"""GET /backends: resolved attention backend, versions, per-engine map (m13)."""

import httpx

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_backends_shape_and_mock_engines():
    app = create_app(engines={"m1": MockBackend(), "m2": MockBackend()})
    async with _client(app) as client:
        resp = await client.get("/backends")

    assert resp.status_code == 200
    body = resp.json()
    # process-level attention resolution (CPU test host -> torch)
    assert body["attention_backend"] in {"torch", "flashinfer"}
    assert body["source"] in {"env", "hw_profile"}
    assert isinstance(body["kernel_tier"], str)
    # torch is always reported; flashinfer only when it is the resolved kernel
    assert "torch" in body["versions"]
    if body["attention_backend"] != "flashinfer":
        assert "flashinfer" not in body["versions"]

    engines = {e["model"]: e for e in body["engines"]}
    assert set(engines) == {"m1", "m2"}
    for entry in engines.values():
        assert entry["engine_backend"] == "mock"
        # mock is a remote/echo engine, not local attention
        assert entry["attention_backend"] is None


async def test_backends_is_open_without_api_key():
    # The BFF calls /backends unauthenticated (trusted-mesh). Even with API keys
    # configured, /backends must be exempt (in middleware _OPEN_PATHS) -> 200.
    app = create_app(
        engines={"m": MockBackend()},
        resolved_api_keys=frozenset({"secret"}),
    )
    async with _client(app) as client:
        open_resp = await client.get("/backends")
        # sanity: a guarded path IS rejected without the key
        guarded = await client.get("/v1/models")

    assert open_resp.status_code == 200
    assert guarded.status_code == 401
