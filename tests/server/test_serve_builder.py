"""DeploymentSpec -> app builder: pool wiring, affinity over HTTP, lifespan (gate C1)."""

import asyncio

import httpx
import pytest

from kairyu.deploy.builder import build_app_from_config, build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.mock import MockBackend
from kairyu.orchestration.orchestrator import Orchestrator

POOLED_YAML = """
engines:
  small: { backend: mock }
pools:
  pooled:
    replicas:
      - { backend: mock }
      - { backend: mock }
      - { backend: mock }
"""


class _ShutdownBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_count = 0

    async def shutdown(self) -> None:
        self.shutdown_count += 1


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _chat_body(content: str, model: str = "pooled", **extra) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


async def test_pool_is_served_and_affinity_sticks():
    app = build_app_from_spec(load_deployment_spec(POOLED_YAML))
    async with _client(app) as client:
        models = await client.get("/v1/models")
        assert {"small", "pooled"} <= {m["id"] for m in models.json()["data"]}

        for _ in range(4):
            response = await client.post(
                "/v1/chat/completions", json=_chat_body("hi", user="alice")
            )
            assert response.status_code == 200
        anonymous = await client.post("/v1/chat/completions", json=_chat_body("hi"))
        assert anonymous.status_code == 200

        metrics = (await client.get("/metrics")).text
    assert 'kairyu_pool_decisions_total{pool="pooled",reason="session_affinity"} 4.0' in metrics
    assert 'kairyu_pool_decisions_total{pool="pooled",reason="least_outstanding"} 1.0' in metrics


async def test_header_session_takes_precedence_over_user():
    app = build_app_from_spec(load_deployment_spec(POOLED_YAML))
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("hi", user="alice"),
            headers={"X-Session-ID": "sess-1"},
        )
        assert response.status_code == 200
        metrics = (await client.get("/metrics")).text
    assert 'reason="session_affinity"} 1.0' in metrics


async def test_lifespan_starts_prober_and_shuts_down_engines():
    yaml_text = """
pools:
  remote:
    replicas:
      - backend: openai
        options: { base_url: "http://gpu-0:8000/v1", model: "m", api_key_env: null }
    probe_interval_s: 0.05
"""
    app = build_app_from_spec(load_deployment_spec(yaml_text))
    assert len(app.state.probers) == 1
    # httpx's ASGITransport never runs the lifespan; drive it directly. The
    # prober task must start and cancel cleanly, and shutdown must close engines.
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            assert (await client.get("/health")).status_code == 200


async def test_lifespan_immediate_probe_validates_only_ready_remote_replicas():
    yaml_text = """
pools:
  remote:
    replicas:
      - backend: openai
        options: { base_url: "http://gpu-0:8000/v1", model: "m", api_key_env: null }
      - backend: openai
        options: { base_url: "http://gpu-1:8000/v1", model: "m", api_key_env: null }
    probe_interval_s: 60
"""
    app = build_app_from_spec(load_deployment_spec(yaml_text))
    prober = app.state.probers[0]
    pool = prober._pool
    both_probed = asyncio.Event()
    requests = 0

    def probe_response(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 2:
            both_probed.set()
        if request.url.host == "gpu-0":
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(503, json={"status": "starting"})

    prober._client = httpx.AsyncClient(transport=httpx.MockTransport(probe_response))

    # Initial URL mappings are applied during construction, before create_app
    # exposes the routes or any backend request can be used as a health signal.
    assert pool.healthy == (False, False)
    async with _client(app) as client:
        assert (await client.get("/readyz")).status_code == 503

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(both_probed.wait(), timeout=1)
        for _ in range(10):
            if pool.healthy == (True, False):
                break
            await asyncio.sleep(0)

        async with _client(app) as client:
            ready = await client.get("/readyz")
            metrics = (await client.get("/metrics")).text

        assert pool.healthy == (True, False)
        assert ready.status_code == 200
        assert 'kairyu_replica_healthy{pool="remote",replica="0"} 1.0' in metrics
        assert 'kairyu_replica_healthy{pool="remote",replica="1"} 0.0' in metrics


async def test_build_from_config_resolves_orchestrator_relative_to_file(tmp_path):
    (tmp_path / "orchestrator.yaml").write_text(
        """
workers:
  - { name: tier1, backend: mock }
  - { name: tier2, backend: mock }
""",
        encoding="utf-8",
    )
    (tmp_path / "deploy.yaml").write_text(
        """
engines:
  m: { backend: mock }
orchestrator:
  spec: orchestrator.yaml
""",
        encoding="utf-8",
    )
    app = build_app_from_config(tmp_path / "deploy.yaml")
    async with _client(app) as client:
        models = await client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
    assert "kairyu-auto" in ids


ORCHESTRATOR_SPEC = """
workers:
  - { name: tier1, backend: mock }
  - { name: tier2, backend: mock }
"""


async def test_named_orchestrators_served_and_answer(tmp_path):
    (tmp_path / "auto.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    (tmp_path / "auto_max.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    (tmp_path / "deploy.yaml").write_text(
        """
engines:
  m: { backend: mock }
orchestrators:
  kairyu-auto: { spec: auto.yaml }
  kairyu-auto-max: { spec: auto_max.yaml }
""",
        encoding="utf-8",
    )
    app = build_app_from_config(tmp_path / "deploy.yaml")
    async with _client(app) as client:
        models = await client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
        assert {"m", "kairyu-auto", "kairyu-auto-max"} <= ids

        for model in ("kairyu-auto", "kairyu-auto-max"):
            response = await client.post(
                "/v1/chat/completions", json=_chat_body("hi", model=model)
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"]


async def test_legacy_orchestrator_composes_with_named(tmp_path):
    (tmp_path / "auto.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    (tmp_path / "auto_max.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    (tmp_path / "deploy.yaml").write_text(
        """
engines:
  m: { backend: mock }
orchestrator: { spec: auto.yaml }
orchestrators:
  kairyu-auto-max: { spec: auto_max.yaml }
""",
        encoding="utf-8",
    )
    app = build_app_from_config(tmp_path / "deploy.yaml")
    async with _client(app) as client:
        models = await client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
    assert {"kairyu-auto", "kairyu-auto-max"} <= ids


async def test_lifespan_attempts_orchestrator_shutdown_after_engine_failure(
    tmp_path, monkeypatch
):
    class _Resource:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail
            self.shutdown_count = 0

        async def shutdown(self) -> None:
            self.shutdown_count += 1
            if self.fail:
                raise RuntimeError("shutdown failed")

    failing_engine = _Resource(fail=True)
    owned_backend = _ShutdownBackend()
    owned_orchestrator = Orchestrator(
        engines={"tier1": owned_backend, "tier2": owned_backend}
    )
    (tmp_path / "auto.yaml").write_text(ORCHESTRATOR_SPEC, encoding="utf-8")
    monkeypatch.setattr(
        "kairyu.deploy.builder.create_backend", lambda *_args, **_kwargs: failing_engine
    )
    monkeypatch.setattr(
        "kairyu.deploy.builder.build_orchestrator", lambda _spec: owned_orchestrator
    )
    spec = load_deployment_spec(
        """
engines:
  bad: { backend: mock }
orchestrator: { spec: auto.yaml }
"""
    )
    app = build_app_from_spec(spec, base_dir=tmp_path)

    with pytest.raises(ExceptionGroup, match="application shutdown"):
        async with app.router.lifespan_context(app):
            pass

    assert failing_engine.shutdown_count == 1
    assert owned_backend.shutdown_count == 1
