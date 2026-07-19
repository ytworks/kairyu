import httpx

from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app


async def test_routing_config_reports_safe_effective_topology(monkeypatch):
    monkeypatch.setattr("kairyu.dsl.loader._build_worker", lambda worker: MockBackend())
    spec = load_spec(
        """
workers:
  - name: tier1
    backend: mock
    model: small
    base_url: https://secret.internal/v1
    api_key_env: SECRET_KEY
  - name: tier2
    backend: mock
    model: large
roles:
  - name: planner
    worker: tier2
    role_type: planner
    prompt: "secret prompt {query}"
  - name: worker
    worker: tier1
    prompt: "work {planner}"
    depends_on: [planner]
"""
    )
    app = create_app(
        engines={},
        orchestrators={"auto": build_orchestrator(spec)},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/routing")

    assert response.status_code == 200
    descriptor = response.json()["models"]["auto"]
    assert descriptor["configured_engines"] == {
        "tier1": {"backend_type": "mock", "model": "small"},
        "tier2": {"backend_type": "mock", "model": "large"},
    }
    assert descriptor["target_resolution"]["tier1"]["engine"] == "tier1"
    assert descriptor["roles"][1]["depends_on"] == ["planner"]
    serialized = response.text
    assert "secret.internal" not in serialized
    assert "SECRET_KEY" not in serialized
    assert "secret prompt" not in serialized


async def test_routing_config_is_explicitly_empty_without_orchestrators():
    app = create_app(engines={})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/routing")
    assert response.json() == {"models": {}}
