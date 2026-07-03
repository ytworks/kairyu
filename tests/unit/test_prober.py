"""Serve-layer health prober restores ejected replicas (design m7 D4, gate C2)."""

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
        ["http://r0/health", "http://r1/health"],
        interval_s=1.0,
        client=_mock_client({"http://r0/health": 200}),
    )
    restored = await prober.check_once()
    assert restored == (0,)
    assert pool.healthy == (True, True)


async def test_probe_leaves_replica_ejected_on_failure():
    pool = ReplicaPool([_FailingBackend(), MockBackend()], unhealthy_after=1)
    await _eject_first_replica(pool)

    for status_map in ({"http://r0/health": 503}, {}):  # 503, then unreachable
        prober = HealthProber(
            "p",
            pool,
            ["http://r0/health", "http://r1/health"],
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
