"""Prepared-request caches release entries from weakref callbacks.

Garbage collection runs such a callback on whichever thread triggers it,
including a thread inside the cache's own locked region. A callback that
took the cache lock froze the whole gateway (event-loop self-deadlock,
2026-09-14, PR #603); a re-entrant lock would instead let the callback mutate
the dict under a caller midway through a lookup. These tests force a
collection inside the locked region of the real backend methods.
"""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable

import pytest

from kairyu import SamplingParams
from kairyu.engine import openai_backend as openai_backend_module
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.weak_identity_cache import WeakIdentityCache


class _Node:
    """A reference cycle, so only the cyclic collector can free it."""

    def __init__(self) -> None:
        self.cycle = self


def _request(name: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=name,
        prompt=name,
        sampling_params=SamplingParams(temperature=0.2, max_tokens=8),
    )


class _CollectOnGet(dict):
    """Run one full collection from inside the cache's locked lookup."""

    def __init__(self, entries: dict) -> None:
        super().__init__(entries)
        self.armed = True

    def get(self, key, default=None):  # type: ignore[override]
        if self.armed:
            self.armed = False
            gc.collect()
        return super().get(key, default)


def _arm(cache: object) -> _CollectOnGet:
    armed = _CollectOnGet(cache._entries)  # type: ignore[attr-defined]
    cache._entries = armed  # type: ignore[attr-defined]
    return armed


def _within(seconds: float, call: Callable[[], object]) -> object:
    result: list[object] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(call())
        except BaseException as error:  # noqa: BLE001 - re-raised below
            failure.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "cache operation deadlocked under GC"
    if failure:
        raise failure[0]
    return result[0]


@pytest.fixture(autouse=True)
def _manual_gc():
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        openai_backend_module._SHARED_PREPARED_PAYLOADS.clear()


def _backend() -> OpenAICompatBackend:
    return OpenAICompatBackend(
        base_url="https://api.example.com/v1", model="m", api_key_env=None
    )


def _retain_dead(retain: Callable[[GenerationRequest], object]) -> None:
    dead = _request("dead")
    retain(dead)
    holder = _Node()
    holder.owner = dead  # type: ignore[attr-defined]
    dead_ref_holder.append(holder)
    del dead


dead_ref_holder: list[_Node] = []


def _drop_dead_holder() -> None:
    # The cycle keeps the request alive until the collector breaks it.
    dead_ref_holder.clear()


@pytest.mark.parametrize("operation", ["peek", "take", "retain"])
def test_backend_payload_cache_survives_gc_inside_its_lock(operation: str) -> None:
    backend = _backend()
    live = _request("live")
    other = _request("other")
    backend._retain_prepared_payload(live, b"live")
    backend._retain_prepared_payload(other, b"other")
    _retain_dead(lambda request: backend._retain_prepared_payload(request, b"dead"))
    _drop_dead_holder()
    armed = _arm(backend._prepared_payloads)

    if operation == "peek":
        assert _within(5, lambda: backend._peek_prepared_payload(live)) == b"live"
        assert backend._peek_prepared_payload(live) == b"live"
    elif operation == "take":
        assert _within(5, lambda: backend._take_prepared_payload(live)) == b"live"
        assert backend._peek_prepared_payload(live) is None
    else:
        fresh = _request("fresh")
        assert _within(5, lambda: backend._retain_prepared_payload(fresh, b"new")) == b"new"
        assert backend._peek_prepared_payload(fresh) == b"new"

    assert not armed.armed, "the collection did not run inside the lookup"
    assert backend._peek_prepared_payload(other) == b"other"
    # The dead entry is gone; only the live ones remain.
    assert len(backend._prepared_payloads) == {"peek": 2, "take": 1, "retain": 3}[operation]


def test_shared_payload_cache_survives_gc_inside_its_lock() -> None:
    backend = _backend()
    live = _request("live")
    backend._retain_shared_prepared_payload(live, b"live")
    _retain_dead(lambda request: backend._retain_shared_prepared_payload(request, b"dead"))
    _drop_dead_holder()
    armed = _arm(openai_backend_module._SHARED_PREPARED_PAYLOADS)

    assert _within(5, lambda: backend._peek_shared_prepared_payload(live)) == b"live"
    assert not armed.armed
    assert len(openai_backend_module._SHARED_PREPARED_PAYLOADS) == 1


def test_late_callback_for_a_reused_id_keeps_the_new_entry() -> None:
    cache: WeakIdentityCache[GenerationRequest, str] = WeakIdentityCache()
    first = _request("first")
    cache.retain(first, "old")
    old_reference = cache._entries[id(first)][0]
    callback = old_reference.__callback__
    # Simulate id reuse: a new live object now owns the same key.
    replacement = _request("replacement")
    cache._entries[id(first)] = (cache._reference(replacement), "new")
    callback(old_reference)

    assert len(cache) == 1
    assert cache._entries[id(first)][1] == "new"


def test_callback_on_another_thread_does_not_wait_for_the_lock() -> None:
    cache: WeakIdentityCache[GenerationRequest, str] = WeakIdentityCache()
    live = _request("live")
    cache.retain(live, "live")
    _retain_dead(lambda request: cache.retain(request, "dead"))
    _drop_dead_holder()

    with cache._lock:
        # A collection on a worker thread while the owner holds the lock.
        _within(5, gc.collect)
        assert len(cache._entries) == 2  # removal is deferred, not applied

    assert len(cache) == 1
    assert cache.peek(live) == "live"
