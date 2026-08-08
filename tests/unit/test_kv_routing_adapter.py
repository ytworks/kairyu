from __future__ import annotations

import threading
from collections.abc import AsyncIterator

import pytest

from kairyu.engine.backend import (
    CacheHint,
    GenerationRequest,
    GenerationResult,
)
from kairyu.orchestration.kv_index import KvEventIndex
from kairyu.orchestration.kv_routing import KvRoutingIndex
from kairyu.orchestration.prefix_index import PrefixIndex, prefix_root_fingerprint
from kairyu.orchestration.replica import ReplicaPool
from kairyu.sampling_params import SamplingParams

_PROMPT = "shared retrieval context " * 24
_HASHES = ("block-1", "block-2")


class _Backend:
    def __init__(self, replica_id: str, selected: list[str]) -> None:
        self.replica_id = replica_id
        self.selected = selected
        self.closed = False

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.selected.append(self.replica_id)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(),
            finished=True,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        yield await self.generate(request)

    async def shutdown(self) -> None:
        self.closed = True


class _PlacementLog:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record_replica(
        self,
        session_id: str | None,
        replica: int,
        reason: str,
        **fields: object,
    ) -> None:
        self.records.append(
            {
                "session_id": session_id,
                "replica": replica,
                "reason": reason,
                **fields,
            }
        )

    async def shutdown(self) -> None:
        return None


class _ExactIndex:
    """Small stateful double for the KvEventIndex routing contract."""

    def __init__(self) -> None:
        self.states = {"r1": "live", "r2": "live"}
        self.hashes = {"r1": {"block-1"}, "r2": set(_HASHES)}
        self.registered: set[str] = set()
        self.forgotten: list[str] = []
        self.routed: list[tuple[str, ...]] = []

    def register_replica(self, replica_id: str) -> None:
        self.registered.add(replica_id)
        self.states.setdefault(replica_id, "unknown")
        self.hashes.setdefault(replica_id, set())

    def forget_replica(self, replica_id: str) -> None:
        self.registered.discard(replica_id)
        self.states.pop(replica_id, None)
        self.hashes.pop(replica_id, None)
        self.forgotten.append(replica_id)

    def all_live(self, replica_ids: tuple[str, ...]) -> bool:
        return all(self.states.get(replica_id) == "live" for replica_id in replica_ids)

    def is_live(self, replica_id: str) -> bool:
        return self.states.get(replica_id) == "live"

    def route_overlaps(
        self,
        replica_ids: tuple[str, ...],
        block_hashes: tuple[str, ...],
    ) -> tuple[int, ...] | None:
        self.routed.append(replica_ids)
        if not self.all_live(replica_ids):
            return None
        overlaps = []
        for replica_id in replica_ids:
            count = 0
            for block_hash in block_hashes:
                if block_hash not in self.hashes[replica_id]:
                    break
                count += 1
            overlaps.append(count)
        return tuple(overlaps)


class _Provider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> tuple[str, ...]:
        self.calls.append(prompt)
        return _HASHES


def _request(request_id: str, *, blank_hint: bool = False) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt=_PROMPT,
        sampling_params=SamplingParams(max_tokens=1),
        cache_hint=CacheHint(
            session_id=f"session-{request_id}",
            prefix_fingerprint=("" if blank_hint else prefix_root_fingerprint(_PROMPT)),
        ),
    )


def _approximate_index() -> PrefixIndex:
    index = PrefixIndex()
    index.observe("r1", _PROMPT)
    return index


def _backends(selected: list[str]) -> dict[str, _Backend]:
    return {replica_id: _Backend(replica_id, selected) for replica_id in ("r1", "r2")}


@pytest.mark.asyncio
async def test_approximate_candidates_do_not_prune_exact_score_space():
    selected: list[str] = []
    exact = _ExactIndex()
    provider = _Provider()
    routing = KvRoutingIndex(_approximate_index(), exact, provider)
    log = _PlacementLog()
    pool = ReplicaPool(
        _backends(selected),
        prefix_index=routing,
        log=log,  # type: ignore[arg-type]
    )

    try:
        await pool.generate(_request("exact"))
    finally:
        await pool.shutdown()

    assert selected == ["r2"]
    assert log.records[-1]["reason"] == "kv_event_match"
    assert pool.decision_counts["kv_event_match"] == 1
    assert routing.mode_counts["exact"] == 1
    assert provider.calls == [_PROMPT]
    assert exact.routed == [("r1", "r2")]


@pytest.mark.asyncio
@pytest.mark.parametrize("dead_state", ["stale", "gap"])
async def test_non_live_feed_globally_uses_approximate_oracle(dead_state):
    exact = _ExactIndex()
    exact.states["r2"] = dead_state
    provider = _Provider()
    routing = KvRoutingIndex(_approximate_index(), exact, provider)
    selected: list[str] = []
    log = _PlacementLog()
    pool = ReplicaPool(
        _backends(selected),
        prefix_index=routing,
        log=log,  # type: ignore[arg-type]
    )
    oracle = ReplicaPool(_backends([]), prefix_index=_approximate_index())
    request = _request(dead_state)
    expected = oracle._select(request)[0]

    try:
        await pool.generate(request)
    finally:
        await pool.shutdown()
        await oracle.shutdown()

    assert selected == [expected] == ["r1"]
    assert log.records[-1]["reason"] == "prefix_match"
    assert routing.mode_counts["approximate_feed_not_live"] == 1
    # Exact prompt hashing is skipped entirely while a feed cannot be used.
    assert provider.calls == []


@pytest.mark.asyncio
async def test_restore_converges_to_exact_on_same_pool_and_index_objects():
    selected: list[str] = []
    exact = _ExactIndex()
    exact.states["r2"] = "gap"
    provider = _Provider()
    routing = KvRoutingIndex(_approximate_index(), exact, provider)
    log = _PlacementLog()
    pool = ReplicaPool(
        _backends(selected),
        prefix_index=routing,
        log=log,  # type: ignore[arg-type]
    )
    identities = (id(pool), id(routing), id(exact))

    try:
        await pool.generate(_request("during-gap"))
        exact.states["r2"] = "live"
        exact.hashes["r2"] = set(_HASHES)
        await pool.generate(_request("after-restore"))
    finally:
        await pool.shutdown()

    assert identities == (id(pool), id(routing), id(exact))
    assert selected == ["r1", "r2"]
    assert [record["reason"] for record in log.records] == [
        "prefix_match",
        "kv_event_match",
    ]
    assert routing.mode_counts["approximate_feed_not_live"] == 1
    assert routing.mode_counts["exact"] == 1


@pytest.mark.asyncio
async def test_blank_cache_hint_bypasses_both_routing_indexes():
    selected: list[str] = []
    exact = _ExactIndex()
    provider = _Provider()
    routing = KvRoutingIndex(_approximate_index(), exact, provider)
    pool = ReplicaPool(_backends(selected), prefix_index=routing)
    oracle = ReplicaPool(_backends([]), prefix_index=_approximate_index())
    request = _request("blank", blank_hint=True)
    expected = oracle._select(request)

    try:
        await pool.generate(request)
    finally:
        await pool.shutdown()
        await oracle.shutdown()

    assert selected == [expected[0]]
    assert pool.decision_counts[expected[1]] == 1
    assert provider.calls == []
    assert sum(routing.mode_counts.values()) == 0


@pytest.mark.asyncio
async def test_dynamic_membership_registers_and_forgets_both_indexes():
    selected: list[str] = []
    exact = _ExactIndex()
    exact.states.pop("r2")
    exact.hashes.pop("r2")
    approximate = _approximate_index()
    routing = KvRoutingIndex(approximate, exact, _Provider())
    pool = ReplicaPool(
        {"r1": _Backend("r1", selected)},
        prefix_index=routing,
    )
    added = _Backend("r2", selected)

    pool.add_replica("r2", added)
    assert exact.registered == {"r1", "r2"}
    await pool.remove_replica("r2")

    assert exact.registered == {"r1"}
    assert exact.forgotten == ["r2"]
    assert added.closed
    await pool.shutdown()


@pytest.mark.parametrize("failure", ["exception", "invalid-vector"])
def test_bulk_exact_failure_globally_falls_back_to_approximate(failure):
    class BrokenBulkIndex(_ExactIndex):
        def route_overlaps(self, replica_ids, block_hashes):
            if failure == "exception":
                raise RuntimeError("exact bulk read failed")
            return (len(block_hashes),)

    routing = KvRoutingIndex(
        _approximate_index(),
        BrokenBulkIndex(),
        _Provider(),
    )

    view = routing.route_view(
        ("r1", "r2"),
        routing.prepare(_PROMPT),
    )

    assert view.mode == "approximate"
    assert view.reason == "approximate_feed_changed"
    assert view.overlaps == ()


def test_provider_iterable_conversion_failure_falls_back_to_approximate():
    class BrokenHashes:
        def __iter__(self):
            raise RuntimeError("block hash iteration failed")

    routing = KvRoutingIndex(
        _approximate_index(),
        _ExactIndex(),
        lambda _prompt: BrokenHashes(),
    )

    view = routing.route_view(
        ("r1", "r2"),
        routing.prepare(_PROMPT),
    )

    assert view.mode == "approximate"
    assert view.reason == "approximate_provider_error"
    assert view.overlaps == ()


def test_exact_scoring_releases_lifecycle_lock_and_revalidates_membership():
    scoring = threading.Event()
    resume = threading.Event()

    class BlockingExactIndex(_ExactIndex):
        def route_overlaps(self, replica_ids, block_hashes):
            scoring.set()
            assert resume.wait(timeout=2)
            return tuple(len(block_hashes) for _replica_id in replica_ids)

    exact = BlockingExactIndex()
    routing = KvRoutingIndex(_approximate_index(), exact, _Provider())
    prepared = routing.prepare(_PROMPT)
    result = []
    route_thread = threading.Thread(
        target=lambda: result.append(routing.route_view(("r1", "r2"), prepared))
    )
    route_thread.start()
    assert scoring.wait(timeout=1)

    changed = threading.Event()

    def forget_candidate() -> None:
        routing.forget_replica("r1")
        changed.set()

    mutation_thread = threading.Thread(target=forget_candidate)
    mutation_thread.start()
    try:
        assert changed.wait(timeout=1), "exact scoring retained the lifecycle lock"
    finally:
        resume.set()
        route_thread.join(timeout=2)
        mutation_thread.join(timeout=2)

    assert not route_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert len(result) == 1
    assert result[0].mode == "approximate"
    assert result[0].reason == "approximate_feed_changed"


def test_legacy_live_index_is_rejected_by_atomic_bulk_routing():
    index = KvEventIndex(now=lambda: 1.0)
    index.register_replica("r1")
    index.apply(
        "r1",
        {"type": "BlockStored", "block_hashes": ["block-1"]},
    )

    assert index.overlap("r1", ["block-1"]) == 1
    assert not index.all_live(("r1",))
    assert index.route_overlaps(("r1",), ("block-1",)) is None


def test_reregistered_replica_rejects_delayed_event_from_retired_epoch():
    index = KvEventIndex(now=lambda: 1.0)
    index.register_replica("r1")
    assert index.install_snapshot("r1", "epoch-old", 0, _HASHES)
    assert index.route_overlaps(("r1",), _HASHES) == (2,)

    index.forget_replica("r1")
    index.register_replica("r1")

    assert (
        index.apply_sequenced(
            "r1",
            "epoch-old",
            1,
            {"type": "BlockStored", "block_hashes": ["late"]},
        )
        == "stale_epoch"
    )
    assert index.view("r1").state == "unknown"
    assert index.route_overlaps(("r1",), _HASHES) is None


def test_register_failure_stays_quarantined_until_forget_register_reset():
    class FailingRegisterIndex(_ExactIndex):
        fail_register = True

        def register_replica(self, replica_id):
            if self.fail_register:
                raise RuntimeError("partial register failure")
            super().register_replica(replica_id)

    exact = FailingRegisterIndex()
    provider = _Provider()
    routing = KvRoutingIndex(_approximate_index(), exact, provider)
    prepared = routing.prepare(_PROMPT)

    routing.register_replica("r2")
    assert routing.route_view(("r1", "r2"), prepared).mode == "approximate"

    # A register retry cannot prove that a partial first call erased stale
    # state, even when the exact index still reports every feed as live.
    exact.fail_register = False
    routing.register_replica("r2")
    retry_view = routing.route_view(("r1", "r2"), prepared)

    assert retry_view.mode == "approximate"
    assert retry_view.reason == "approximate_feed_changed"
    assert provider.calls == []

    routing.forget_replica("r2")
    routing.register_replica("r2")
    exact.states["r2"] = "live"
    exact.hashes["r2"] = set(_HASHES)

    restored = routing.route_view(("r1", "r2"), prepared)
    assert restored.mode == "exact"
    assert restored.overlaps == (1, 2)
    assert provider.calls == [_PROMPT]


def test_forget_failure_stays_quarantined_across_register_success():
    class FailingForgetIndex(_ExactIndex):
        fail_forget = True

        def forget_replica(self, replica_id):
            if self.fail_forget:
                raise RuntimeError("partial forget failure")
            super().forget_replica(replica_id)

    exact = FailingForgetIndex()
    provider = _Provider()
    routing = KvRoutingIndex(_approximate_index(), exact, provider)
    prepared = routing.prepare(_PROMPT)

    routing.forget_replica("r2")
    routing.register_replica("r2")
    quarantined = routing.route_view(("r1", "r2"), prepared)

    assert quarantined.mode == "approximate"
    assert quarantined.reason == "approximate_feed_changed"
    assert provider.calls == []

    exact.fail_forget = False
    routing.forget_replica("r2")
    routing.register_replica("r2")
    exact.states["r2"] = "live"
    exact.hashes["r2"] = set(_HASHES)

    restored = routing.route_view(("r1", "r2"), prepared)
    assert restored.mode == "exact"
    assert restored.overlaps == (1, 2)
    assert provider.calls == [_PROMPT]
