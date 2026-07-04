"""/health, /readyz, /metrics (goal G3 gate C6)."""

import httpx

from kairyu.engine.backend import GenerationRequest
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.orchestration.replica import ReplicaPool
from kairyu.sampling_params import SamplingParams


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


class _FailingBackend:
    async def generate(self, request):
        raise RuntimeError("replica down")

    async def stream(self, request):
        raise RuntimeError("replica down")
        yield  # pragma: no cover

    async def shutdown(self) -> None:
        return None


def _chat_body(content: str, model: str = "m", **extra) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


async def test_health_and_readyz_ok():
    app = create_app(engines={"m": MockBackend()})
    async with _client(app) as client:
        health = await client.get("/health")
        ready = await client.get("/readyz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


async def test_readyz_503_when_pool_has_no_healthy_replica():
    pool = ReplicaPool([_FailingBackend()], unhealthy_after=1)
    app = create_app(engines={"m": pool})
    request = GenerationRequest(
        request_id="r1", prompt="p", sampling_params=SamplingParams()
    )
    try:
        await pool.generate(request)
    except RuntimeError:
        pass
    assert not any(pool.healthy)
    async with _client(app) as client:
        ready = await client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["status"] == "unready"
    assert "m" in ready.json()["pools"]


async def test_metrics_exposes_request_and_pool_series():
    pool = ReplicaPool([MockBackend(), MockBackend()])
    app = create_app(engines={"pooled": pool})
    async with _client(app) as client:
        ok = await client.post("/v1/chat/completions", json=_chat_body("hi", model="pooled"))
        assert ok.status_code == 200
        missing = await client.post("/v1/chat/completions", json=_chat_body("hi", model="nope"))
        assert missing.status_code == 404
        metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    assert 'kairyu_requests_total{code="200",model="pooled"} 1.0' in text
    # M1: an unknown model collapses to "unknown" (bounded cardinality)
    assert 'kairyu_requests_total{code="404",model="unknown"} 1.0' in text
    assert "kairyu_request_duration_seconds_bucket" in text
    assert 'kairyu_replica_healthy{pool="pooled",replica="0"} 1.0' in text
    assert 'kairyu_replica_outstanding{pool="pooled",replica="0"} 0.0' in text
    assert 'kairyu_pool_decisions_total{pool="pooled",reason="least_outstanding"} 1.0' in text


async def test_metrics_disabled_by_settings():
    from kairyu.entrypoints.server.settings import ServerSettings

    app = create_app(engines={"m": MockBackend()}, settings=ServerSettings(metrics=False))
    async with _client(app) as client:
        response = await client.get("/metrics")
    assert response.status_code == 404


async def test_access_log_adds_request_id_header():
    app = create_app(engines={"m": MockBackend()})
    async with _client(app) as client:
        response = await client.get("/health")
    assert response.headers.get("x-request-id")


def test_admin_drain_requires_auth_and_flips_readyz(monkeypatch):
    """m10a A5 + security review: /admin/drain is auth-protected when keys are
    configured, and a drained node reports unready."""
    from fastapi.testclient import TestClient

    from kairyu.engine.registry import create_backend
    from kairyu.entrypoints.server.app import create_app
    from kairyu.entrypoints.server.settings import ServerSettings

    monkeypatch.setenv("KAIRYU_TEST_KEYS", "sk-secret")
    app = create_app(
        {"m": create_backend("mock")},
        settings=ServerSettings(api_keys_env="KAIRYU_TEST_KEYS"),
    )
    with TestClient(app) as client:
        assert client.post("/admin/drain").status_code == 401  # no key
        assert client.get("/readyz").status_code == 200
        ok = client.post(
            "/admin/drain", headers={"Authorization": "Bearer sk-secret"}
        )
        assert ok.status_code == 200
        assert client.get("/readyz").status_code == 503  # draining


def test_admin_drain_requires_admin_key_and_undrain_recovers(monkeypatch):
    # S5: with admin keys configured, an ordinary data-plane key cannot drain
    # the node, and /admin/undrain restores readiness (no restart needed).
    from fastapi.testclient import TestClient

    from kairyu.engine.registry import create_backend
    from kairyu.entrypoints.server.app import create_app
    from kairyu.entrypoints.server.settings import ServerSettings

    monkeypatch.setenv("KAIRYU_TEST_KEYS", "sk-user,sk-admin")
    monkeypatch.setenv("KAIRYU_TEST_ADMIN", "sk-admin")
    app = create_app(
        {"m": create_backend("mock")},
        settings=ServerSettings(
            api_keys_env="KAIRYU_TEST_KEYS", admin_keys_env="KAIRYU_TEST_ADMIN"
        ),
    )
    with TestClient(app) as client:
        user = {"Authorization": "Bearer sk-user"}
        admin = {"Authorization": "Bearer sk-admin"}
        assert client.post("/admin/drain", headers=user).status_code == 403  # data-plane key
        assert client.get("/readyz").status_code == 200  # still ready
        assert client.post("/admin/drain", headers=admin).status_code == 200
        assert client.get("/readyz").status_code == 503  # draining
        assert client.post("/admin/undrain", headers=user).status_code == 403
        assert client.post("/admin/undrain", headers=admin).status_code == 200
        assert client.get("/readyz").status_code == 200  # recovered
