"""Tests for the DP ReplicaPool (design doc m5 D4)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator

import pytest

from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    UpstreamClientError,
)
from kairyu.engine.mock import MockBackend
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.router import JsonlRouterLog
from kairyu.sampling_params import SamplingParams


class FlakyBackend:
    """MockBackend wrapper whose calls can be toggled to fail; counts shutdowns."""

    def __init__(self, latency_s: float = 0.0) -> None:
        self._inner = MockBackend(latency_s=latency_s)
        self.failing = False
        self.client_failing = False  # raises a 4xx-style client error
        self.shutdown_count = 0
        self.shutdown_error = False

    @property
    def prompts_seen(self) -> tuple[str, ...]:
        return self._inner.prompts_seen

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.client_failing:
            raise UpstreamClientError("bad request", status_code=400)
        if self.failing:
            raise RuntimeError("injected failure")
        return await self._inner.generate(request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        if self.failing:
            raise RuntimeError("injected failure")
        async for chunk in self._inner.stream(request):
            yield chunk

    async def shutdown(self) -> None:
        self.shutdown_count += 1
        if self.shutdown_error:
            raise RuntimeError("shutdown failed")


def make_request(prompt: str, session_id: str | None = None) -> GenerationRequest:
    hint = CacheHint(session_id=session_id) if session_id is not None else None
    return GenerationRequest(
        request_id=f"req-{prompt}",
        prompt=prompt,
        sampling_params=SamplingParams(),
        cache_hint=hint,
    )


async def place_session(pool: ReplicaPool, backends: list[FlakyBackend], session: str) -> int:
    """Send one request for ``session`` and return the index of the replica that got it."""
    prompt = f"probe:{session}:{sum(len(b.prompts_seen) for b in backends)}"
    await pool.generate(make_request(prompt, session_id=session))
    return next(i for i, b in enumerate(backends) if prompt in b.prompts_seen)


def test_pool_satisfies_engine_backend_protocol():
    pool = ReplicaPool([MockBackend()])
    assert isinstance(pool, EngineBackend)


def test_constructor_rejects_empty_replicas():
    with pytest.raises(ValueError, match="at least 1 replica"):
        ReplicaPool([])


def test_constructor_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="unhealthy_after"):
        ReplicaPool([MockBackend()], unhealthy_after=0)
    with pytest.raises(ValueError, match="queue_depth_threshold"):
        ReplicaPool([MockBackend()], queue_depth_threshold=-1)


async def test_generate_delegates_and_returns_result():
    backend = MockBackend(responses={"hello": "world"})
    pool = ReplicaPool([backend])
    result = await pool.generate(make_request("hello"))
    assert result.text == "world"
    assert backend.prompts_seen == ("hello",)


async def test_session_affinity_sticks_across_calls():
    backends = [FlakyBackend() for _ in range(3)]
    pool = ReplicaPool(backends)
    first = await place_session(pool, backends, "session-A")
    for _ in range(5):
        assert await place_session(pool, backends, "session-A") == first


async def test_distinct_sessions_spread_over_replicas():
    backends = [FlakyBackend() for _ in range(4)]
    pool = ReplicaPool(backends)
    placements = {
        await place_session(pool, backends, f"session-{i}") for i in range(32)
    }
    assert len(placements) > 1  # rendezvous hashing does not funnel all sessions to one


async def test_rendezvous_remap_only_moves_ejected_sessions():
    backends = [FlakyBackend() for _ in range(3)]
    pool = ReplicaPool(backends, unhealthy_after=2)
    sessions = [f"session-{i}" for i in range(30)]
    before = {s: await place_session(pool, backends, s) for s in sessions}
    ejected = before[sessions[0]]

    # Eject that replica: fail `unhealthy_after` consecutive requests on it.
    backends[ejected].failing = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected failure"):
            await pool.generate(make_request("fail-me", session_id=sessions[0]))
    backends[ejected].failing = False  # healthy again, but still off the ring

    after = {s: await place_session(pool, backends, s) for s in sessions}
    for session in sessions:
        if before[session] == ejected:
            assert after[session] != ejected  # remapped off the ejected replica
        else:
            assert after[session] == before[session]  # everyone else stays put


async def test_client_errors_do_not_eject_the_replica():
    # O1: a bad client request (4xx) repeated on a session-pinned replica must
    # NOT count as a replica failure — otherwise one misbehaving client could
    # cascade-eject the whole pool.
    backends = [FlakyBackend() for _ in range(3)]
    pool = ReplicaPool(backends, unhealthy_after=2)
    target = await place_session(pool, backends, "sticky")
    backends[target].client_failing = True
    for _ in range(5):  # far more than unhealthy_after
        with pytest.raises(UpstreamClientError):
            await pool.generate(make_request("bad", session_id="sticky"))
    backends[target].client_failing = False
    # the replica is still healthy and still serves the session
    assert await place_session(pool, backends, "sticky") == target


async def test_sessionless_goes_to_least_outstanding_with_lowest_index_ties():
    backends = [MockBackend(latency_s=0.05) for _ in range(2)]
    pool = ReplicaPool(backends)
    # All idle: first sessionless request must pick the lowest index.
    task = asyncio.ensure_future(pool.generate(make_request("first")))
    await asyncio.sleep(0.01)
    assert backends[0].prompts_seen == ()  # still in flight; check dispatch below
    # While replica 0 is busy, the next sessionless request goes to replica 1.
    await pool.generate(make_request("second"))
    await task
    assert backends[0].prompts_seen == ("first",)
    assert backends[1].prompts_seen == ("second",)


async def test_queue_depth_fallback_overflows_affine_replica():
    backends = [FlakyBackend(latency_s=0.1) for _ in range(2)]
    pool = ReplicaPool(backends, queue_depth_threshold=0)
    session = "hot-session"
    fast_pool = ReplicaPool(backends)  # shares backends only to discover affinity
    affine = await place_session(fast_pool, backends, session)
    other = 1 - affine

    in_flight = asyncio.ensure_future(
        pool.generate(make_request("occupy", session_id=session))
    )
    await asyncio.sleep(0.02)  # let it dispatch to the affine replica
    await pool.generate(make_request("overflow", session_id=session))
    await in_flight
    assert any("occupy" in p for p in backends[affine].prompts_seen)
    assert any("overflow" in p for p in backends[other].prompts_seen)


async def test_health_ejection_after_consecutive_failures_and_probe_recovery():
    backends = [FlakyBackend() for _ in range(2)]
    pool = ReplicaPool(backends, unhealthy_after=2)
    session = "sticky"
    affine = await place_session(pool, backends, session)
    other = 1 - affine

    backends[affine].failing = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected failure"):
            await pool.generate(make_request("boom", session_id=session))
    backends[affine].failing = False

    # Ejected: the affine session now lands on the surviving replica.
    assert await place_session(pool, backends, session) == other

    # A successful probe returns the replica to the ring; affinity is restored.
    await pool.probe(affine)
    assert await place_session(pool, backends, session) == affine


async def test_remote_replica_requires_probe_then_ejects_and_restores():
    trusted = FlakyBackend()
    remote = FlakyBackend()
    pool = ReplicaPool({"trusted": trusted}, unhealthy_after=2)
    pool.add_replica("remote", remote, health_url="http://remote/readyz")
    await pool.remove_replica("trusted")

    assert pool.validated_by_id() == {"remote": False}
    assert pool.healthy_by_id() == {"remote": False}
    with pytest.raises(RuntimeError, match="eligible"):
        await pool.generate(make_request("before-probe"))

    lease = pool.acquire_drain("remote")
    pool.drain("remote")
    await pool.probe("remote")
    assert pool.validated_by_id() == {"remote": True}
    assert pool.healthy_by_id() == {"remote": True}
    assert pool.is_draining("remote") is True
    assert pool.is_manually_draining("remote") is True
    with pytest.raises(RuntimeError, match="eligible"):
        await pool.generate(make_request("still-draining"))

    pool.cancel_drain("remote")
    assert pool.is_draining("remote") is True
    pool.release_drain("remote", lease)
    assert pool.is_draining("remote") is False
    assert (await pool.generate(make_request("validated"))).finished
    remote.failing = True
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("failure-1"))
    assert pool.healthy_by_id() == {"remote": True}
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("failure-2"))
    assert pool.validated_by_id() == {"remote": True}
    assert pool.healthy_by_id() == {"remote": False}

    remote.failing = False
    await pool.probe("remote")
    assert pool.healthy_by_id() == {"remote": True}
    assert (await pool.generate(make_request("restored"))).finished


async def test_local_replica_starts_validated_and_can_require_probe_by_id():
    pool = ReplicaPool({"local": MockBackend()})

    assert pool.validated_by_id() == {"local": True}
    assert pool.healthy_by_id() == {"local": True}
    assert (await pool.generate(make_request("trusted"))).finished

    pool.require_probe("local")
    assert pool.validated_by_id() == {"local": False}
    assert pool.healthy_by_id() == {"local": False}
    with pytest.raises(RuntimeError, match="eligible"):
        await pool.generate(make_request("unknown"))

    await pool.probe("local")
    assert pool.validated_by_id() == {"local": True}
    assert (await pool.generate(make_request("revalidated"))).finished


async def test_same_id_remote_readd_does_not_inherit_validation():
    pool = ReplicaPool({"local": MockBackend()})
    pool.add_replica("remote", MockBackend(), health_url="http://old/readyz")
    await pool.probe("remote")
    old_generation = pool.entry_generation("remote")
    assert pool.validated_by_id()["remote"] is True

    await pool.remove_replica("remote")
    pool.add_replica("remote", MockBackend(), health_url="http://new/readyz")

    assert pool.entry_generation("remote") is not old_generation
    assert pool.validated_by_id()["remote"] is False
    assert pool.healthy_by_id()["remote"] is False


async def test_failure_count_resets_on_any_success():
    backends = [FlakyBackend()]
    pool = ReplicaPool(backends, unhealthy_after=3)
    for _ in range(2):
        backends[0].failing = True
        with pytest.raises(RuntimeError):
            await pool.generate(make_request("x"))
        backends[0].failing = False
    await pool.generate(make_request("ok"))  # resets the consecutive-failure count
    backends[0].failing = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected failure"):
            await pool.generate(make_request("y"))
    backends[0].failing = False
    # Still healthy (2 failures < 3 after the reset), so it serves again.
    result = await pool.generate(make_request("still-alive"))
    assert result.finished


async def test_all_unhealthy_raises_clear_runtime_error():
    backends = [FlakyBackend()]
    pool = ReplicaPool(backends, unhealthy_after=1)
    backends[0].failing = True
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("boom"))
    with pytest.raises(RuntimeError, match="none of the 1 replicas are eligible"):
        await pool.generate(make_request("nobody-home"))


async def test_probe_rejects_invalid_index():
    pool = ReplicaPool([MockBackend()])
    with pytest.raises(ValueError, match="replica index"):
        await pool.probe(5)


async def test_placement_logged_with_hashed_session(tmp_path):
    log_path = tmp_path / "router.jsonl"
    pool = ReplicaPool([MockBackend(), MockBackend()], log=JsonlRouterLog(log_path))
    await pool.generate(make_request("hello", session_id="secret-session"))
    await pool.generate(make_request("anon"))
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["replica", "replica"]
    assert lines[0]["session_sha256"] == hashlib.sha256(b"secret-session").hexdigest()
    assert lines[0]["reason"] == "session_affinity"
    assert lines[1]["session_sha256"] is None
    assert lines[1]["reason"] == "least_outstanding"
    assert all(isinstance(line["replica"], int) for line in lines)
    assert "secret-session" not in log_path.read_text()  # raw session id never stored


async def test_placement_logged_before_dispatch_even_on_failure(tmp_path):
    log_path = tmp_path / "router.jsonl"
    backends = [FlakyBackend()]
    backends[0].failing = True
    pool = ReplicaPool(backends, log=JsonlRouterLog(log_path))
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("boom", session_id="s"))
    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["kind"] == "replica"


async def test_outstanding_accounting_under_concurrent_load():
    backends = [MockBackend(latency_s=0.05) for _ in range(2)]
    pool = ReplicaPool(backends)
    requests = [make_request(f"load-{i}") for i in range(8)]

    async def snapshot_later() -> tuple[int, ...]:
        await asyncio.sleep(0.02)  # all 8 dispatched, none finished
        return pool.outstanding

    mid_flight, *_ = await asyncio.gather(
        snapshot_later(), *(pool.generate(r) for r in requests)
    )
    assert sum(mid_flight) == 8
    assert mid_flight == (4, 4)  # least-outstanding spreads evenly
    assert pool.outstanding == (0, 0)  # decremented on completion
    assert len(backends[0].prompts_seen) == 4
    assert len(backends[1].prompts_seen) == 4


async def test_outstanding_decremented_on_error():
    backends = [FlakyBackend()]
    backends[0].failing = True
    pool = ReplicaPool(backends)
    with pytest.raises(RuntimeError, match="injected failure"):
        await pool.generate(make_request("boom"))
    assert pool.outstanding == (0,)


async def test_stream_delegates_with_affinity_and_accounting(tmp_path):
    log_path = tmp_path / "router.jsonl"
    backends = [FlakyBackend() for _ in range(2)]
    pool = ReplicaPool(backends, log=JsonlRouterLog(log_path))
    session = "stream-session"
    affine = await place_session(pool, backends, session)

    chunks = []
    async for chunk in pool.stream(make_request("stream me a story", session_id=session)):
        chunks.append(chunk)
        assert sum(pool.outstanding) == 1  # counted while streaming
    assert chunks[-1].finished
    assert len(chunks) > 1  # partials + final, per MockBackend chunking
    assert pool.outstanding == (0, 0)
    assert any("stream me a story" in p for p in backends[affine].prompts_seen)
    last_entry = json.loads(log_path.read_text().splitlines()[-1])
    assert last_entry["kind"] == "replica"


async def test_stream_failure_counts_toward_health_and_decrements():
    backends = [FlakyBackend()]
    pool = ReplicaPool(backends, unhealthy_after=1)
    backends[0].failing = True
    with pytest.raises(RuntimeError, match="injected failure"):
        async for _ in pool.stream(make_request("boom")):
            pass
    assert pool.outstanding == (0,)
    with pytest.raises(RuntimeError, match="unhealthy"):
        await pool.generate(make_request("next"))


async def test_shutdown_shuts_down_all_members():
    backends = [FlakyBackend() for _ in range(3)]
    pool = ReplicaPool(backends)
    await pool.shutdown()
    assert [b.shutdown_count for b in backends] == [1, 1, 1]


async def test_remove_replica_closes_backend_exactly_once():
    removed = FlakyBackend()
    survivor = FlakyBackend()
    pool = ReplicaPool({"removed": removed, "survivor": survivor})

    await pool.remove_replica("removed")

    assert removed.shutdown_count == 1
    assert survivor.shutdown_count == 0
    assert pool.replica_ids == ("survivor",)
    await pool.shutdown()
    assert removed.shutdown_count == 1
    assert survivor.shutdown_count == 1


async def test_forced_remove_closes_inflight_backend_immediately():
    backend = FlakyBackend(latency_s=0.1)
    pool = ReplicaPool({"busy": backend})
    task = asyncio.create_task(pool.generate(make_request("busy")))
    await asyncio.sleep(0.01)

    await pool.remove_replica("busy", force=True)

    assert backend.shutdown_count == 1
    await task


async def test_shutdown_attempts_every_unique_backend_and_aggregates_errors():
    shared = FlakyBackend()
    failing = FlakyBackend()
    failing.shutdown_error = True
    pool = ReplicaPool({"shared-a": shared, "shared-b": shared, "bad": failing})

    with pytest.raises(ExceptionGroup, match="ReplicaPool shutdown") as caught:
        await pool.shutdown()

    assert shared.shutdown_count == 1
    assert failing.shutdown_count == 1
    assert len(caught.value.exceptions) == 1
