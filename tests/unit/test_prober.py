"""Serve-layer health prober restores ejected replicas (design m7 D4, gate C2)."""

import asyncio

import httpx
import pytest

from kairyu.deploy.prober import HealthProber
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.mock import MockBackend
from kairyu.orchestration.replica import ReplicaPool
from kairyu.sampling_params import SamplingParams


class _FailingBackend:
    async def generate(self, request):
        raise RuntimeError("down")

    async def stream(self, request):
        raise RuntimeError("down")
        yield  # pragma: no cover

    async def shutdown(self) -> None:
        return None


def _request() -> GenerationRequest:
    return GenerationRequest(request_id="r", prompt="p", sampling_params=SamplingParams())


async def _eject_first_replica(pool: ReplicaPool) -> None:
    with pytest.raises(RuntimeError):
        await pool.generate(_request())


def _mock_client(status_by_url: dict[str, int]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        status = status_by_url.get(str(request.url))
        if status is None:
            raise httpx.ConnectError("unreachable", request=request)
        return httpx.Response(status)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_probe_restores_replica_when_health_returns_200():
    pool = ReplicaPool([_FailingBackend(), MockBackend()], unhealthy_after=1)
    await _eject_first_replica(pool)
    assert pool.healthy == (False, True)

    prober = HealthProber(
        "p",
        pool,
        ["http://r0/health", None],
        interval_s=1.0,
        client=_mock_client({"http://r0/health": 200}),
    )
    restored = await prober.check_once()
    assert restored == ("0",)  # id-keyed, not ordinal (O2)
    assert pool.healthy == (True, True)


async def test_probe_leaves_replica_ejected_on_failure():
    pool = ReplicaPool([_FailingBackend(), MockBackend()], unhealthy_after=1)
    await _eject_first_replica(pool)

    for status_map in ({"http://r0/health": 503}, {}):  # 503, then unreachable
        prober = HealthProber(
            "p",
            pool,
            ["http://r0/health", None],
            interval_s=1.0,
            client=_mock_client(status_map),
        )
        assert await prober.check_once() == ()
        assert pool.healthy == (False, True)


async def test_unprobeable_member_is_skipped():
    pool = ReplicaPool([_FailingBackend()], unhealthy_after=1)
    await _eject_first_replica(pool)
    prober = HealthProber("p", pool, [None], interval_s=1.0, client=_mock_client({}))
    assert await prober.check_once() == ()
    assert pool.healthy == (False,)


def test_url_count_must_match_replica_count():
    pool = ReplicaPool([MockBackend()])
    with pytest.raises(ValueError, match="health URLs"):
        HealthProber("p", pool, ["a", "b"], interval_s=1.0)


async def test_prober_keys_by_id_across_membership_change():
    # O2: after a membership change shifts ordinals, the prober must still
    # restore the replica that is actually ejected (by id), never IndexError or
    # restore the wrong replica off a stale positional list.
    pool = ReplicaPool({"a": _FailingBackend(), "b": MockBackend()}, unhealthy_after=1)
    await _eject_first_replica(pool)  # ejects "a"
    assert pool.healthy_by_id()["a"] is False
    pool.add_replica("c", MockBackend(), health_url="http://c/readyz")  # shifts order
    prober = HealthProber(
        "p", pool,
        {"a": "http://a/readyz", "b": None},
        interval_s=1.0,
        client=_mock_client({"http://a/readyz": 200}),
    )
    restored = await prober.check_once()
    assert restored == ("a",)
    assert pool.healthy_by_id()["a"] is True


async def test_prober_tick_failure_does_not_kill_the_loop():
    # O2: a raising check_once must be swallowed by run() so the prober keeps
    # probing instead of silently dying and never restoring anything again.
    pool = ReplicaPool([MockBackend()])
    prober = HealthProber("p", pool, ["http://r/readyz"], interval_s=0.0)
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("tick blew up")
        raise asyncio.CancelledError  # stop the loop on the second tick

    prober.check_once = boom  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await prober.run()
    assert calls["n"] == 2  # survived the first failure and ticked again


async def test_constructor_marks_initial_urls_unknown_and_skips_healthy_entries():
    pool = ReplicaPool({"remote": MockBackend(), "trusted": MockBackend()})
    requested = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prober = HealthProber(
        "p",
        pool,
        {"remote": "http://remote/readyz", "trusted": None},
        interval_s=1.0,
        client=client,
    )

    assert pool.validated_by_id() == {"remote": False, "trusted": True}
    assert await prober.check_once() == ("remote",)
    assert requested == ["http://remote/readyz"]
    assert pool.validated_by_id() == {"remote": True, "trusted": True}
    await client.aclose()


async def test_check_once_bounds_concurrent_probe_requests():
    replica_count = 5
    max_concurrency = 2
    pool = ReplicaPool(
        {f"r{index}": MockBackend() for index in range(replica_count)}
    )
    for replica_id in pool.replica_ids:
        pool.require_probe(replica_id)

    active = 0
    peak = 0
    requested = []
    first_wave_started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        requested.append(str(request.url))
        if active == max_concurrency:
            first_wave_started.set()
        try:
            await release.wait()
            return httpx.Response(503)
        finally:
            active -= 1

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prober = HealthProber(
        "p",
        pool,
        {replica_id: f"http://{replica_id}/readyz" for replica_id in pool.replica_ids},
        interval_s=1.0,
        client=client,
        max_concurrency=max_concurrency,
    )
    check = asyncio.create_task(prober.check_once())
    try:
        await asyncio.wait_for(first_wave_started.wait(), timeout=0.2)
        assert check.done() is False
        assert 1 < peak <= max_concurrency
    finally:
        release.set()
        restored = await check

    assert restored == ()
    assert set(requested) == {
        f"http://r{index}/readyz" for index in range(replica_count)
    }
    assert peak == max_concurrency
    await client.aclose()


async def test_check_once_isolates_mixed_probe_results():
    pool = ReplicaPool({"trusted": MockBackend()})
    for replica_id in ("error", "healthy", "unready"):
        pool.add_replica(
            replica_id,
            MockBackend(),
            health_url=f"http://{replica_id}/readyz",
        )

    async def handler(request: httpx.Request) -> httpx.Response:
        replica_id = request.url.host
        if replica_id == "error":
            raise RuntimeError("transport exploded")
        return httpx.Response(200 if replica_id == "healthy" else 503)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prober = HealthProber("p", pool, {}, interval_s=1.0, client=client)

    assert await prober.check_once() == ("healthy",)
    assert pool.validated_by_id() == {
        "trusted": True,
        "error": False,
        "healthy": True,
        "unready": False,
    }
    await client.aclose()


async def test_removed_probe_result_cannot_validate_a_later_same_id_generation():
    pool = ReplicaPool({"trusted": MockBackend()})
    pool.add_replica("same", MockBackend(), health_url="http://old/readyz")
    pool.add_replica("stable", MockBackend(), health_url="http://stable/readyz")
    old_generation = pool.entry_generation("same")
    old_started = asyncio.Event()
    release_old = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "old":
            old_started.set()
            await release_old.wait()
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prober = HealthProber(
        "p",
        pool,
        {},
        interval_s=1.0,
        client=client,
        max_concurrency=2,
    )
    check = asyncio.create_task(prober.check_once())
    await asyncio.wait_for(old_started.wait(), timeout=0.2)

    await pool.remove_replica("same")
    pool.add_replica("same", MockBackend(), health_url="http://new/readyz")
    release_old.set()

    assert await check == ("stable",)
    assert pool.entry_generation("same") is not old_generation
    assert pool.validated_by_id() == {
        "trusted": True,
        "stable": True,
        "same": False,
    }
    await client.aclose()


async def test_dynamic_remote_replica_is_discovered_on_next_snapshot():
    pool = ReplicaPool({"trusted": MockBackend()})
    client = _mock_client({"http://dynamic/readyz": 200})
    prober = HealthProber("p", pool, {}, interval_s=1.0, client=client)
    pool.add_replica(
        "dynamic",
        MockBackend(),
        health_url="http://dynamic/readyz",
    )

    assert await prober.check_once() == ("dynamic",)
    assert pool.validated_by_id()["dynamic"] is True
    await client.aclose()


async def test_run_checks_immediately_and_closes_client_when_cancelled():
    pool = ReplicaPool({"trusted": MockBackend()})
    client = _mock_client({})
    prober = HealthProber("p", pool, {}, interval_s=3600.0, client=client)
    checked = asyncio.Event()

    async def record_tick() -> tuple[str, ...]:
        checked.set()
        return ()

    prober.check_once = record_tick  # type: ignore[method-assign]
    task = asyncio.create_task(prober.run())
    try:
        await asyncio.wait_for(checked.wait(), timeout=0.2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert client.is_closed is True


@pytest.mark.parametrize("max_concurrency", [0, -1])
def test_max_concurrency_must_be_positive(max_concurrency):
    pool = ReplicaPool({"trusted": MockBackend()})
    with pytest.raises(ValueError, match="max_concurrency"):
        HealthProber(
            "p",
            pool,
            {},
            interval_s=1.0,
            max_concurrency=max_concurrency,
        )
