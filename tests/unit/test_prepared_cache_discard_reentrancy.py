"""The prepared-request caches drop entries through weakref discard callbacks.
Such a callback runs at garbage-collection time on whichever thread triggers
the collection, including the thread that is inside one of the cache's own
lock-guarded regions. With a non-re-entrant lock that thread then waits on
itself; on the gateway's event loop this froze the whole server (V4.1 tiered
example, judged c16 row, 2026-09-14; py-spy stack in its MEASUREMENTS.md).
"""

import threading

import pytest

from kairyu import SamplingParams
from kairyu.engine import openai_backend as openai_backend_module
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.zmq_backend import ZmqEngineBackend


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="cache-owner",
        prompt="hello",
        sampling_params=SamplingParams(temperature=0.2, max_tokens=8),
    )


def _openai_backend() -> OpenAICompatBackend:
    return OpenAICompatBackend(base_url="https://api.example.com/v1", model="m", api_key_env=None)


def _openai_payload():
    backend = _openai_backend()
    return (
        backend,
        lambda request: backend._retain_prepared_payload(request, b"{}"),
        backend._prepared_payloads_lock,
        backend._prepared_payloads,
    )


def _openai_shared_payload():
    backend = _openai_backend()
    return (
        backend,
        lambda request: backend._retain_shared_prepared_payload(request, b"{}"),
        openai_backend_module._SHARED_PREPARED_PAYLOADS_LOCK,
        openai_backend_module._SHARED_PREPARED_PAYLOADS,
    )


def _openai_image_urls():
    backend = _openai_backend()
    return (
        backend,
        lambda request: backend._retain_prepared_image_urls(request, ("data:x",)),
        backend._prepared_image_urls_lock,
        backend._prepared_image_urls,
    )


def _kairyu_prepared_request():
    backend = KairyuBackend()
    return (
        backend,
        lambda request: backend._retain_prepared_request(request, object()),
        backend._prepared_requests_lock,
        backend._prepared_requests,
    )


def _zmq_prepared_request():
    backend = ZmqEngineBackend()
    return (
        backend,
        lambda request: backend._retain_prepared_request(request, object()),
        backend._prepared_requests_lock,
        backend._prepared_requests,
    )


@pytest.mark.parametrize(
    "cache",
    [
        _openai_payload,
        _openai_shared_payload,
        _openai_image_urls,
        _kairyu_prepared_request,
        _zmq_prepared_request,
    ],
    ids=["openai-payload", "openai-shared-payload", "openai-image-urls", "kairyu", "zmq"],
)
def test_discard_callback_on_the_lock_holding_thread_does_not_deadlock(cache):
    backend, retain, lock, entries = cache()
    holder = [_request()]
    retain(holder[0])
    key = id(holder[0])
    assert key in entries

    def drop_last_reference_under_lock() -> None:
        with lock:
            # The last strong reference dies here, so the discard callback runs
            # synchronously on this thread while it holds the cache lock — the
            # situation garbage collection creates inside _peek_/_take_.
            holder.clear()

    worker = threading.Thread(target=drop_last_reference_under_lock, daemon=True)
    worker.start()
    worker.join(5.0)
    if worker.is_alive():
        try:
            lock.release()  # unblock the self-deadlocked worker (plain Lock only)
        except RuntimeError:
            pass
        pytest.fail("discard callback deadlocked on the cache lock it re-entered")
    assert key not in entries
    del backend
