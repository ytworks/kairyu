"""DeploymentSpec -> app builder: pool wiring, affinity over HTTP, lifespan (gate C1)."""

import httpx
import pytest

import kairyu.deploy.builder as builder_module
from kairyu.deploy.builder import build_app_from_config, build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.entrypoints.server.tenancy import TenantLimits, UsageLedger

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


def test_builder_wires_distinct_tenant_identities_and_limits(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    spec = load_deployment_spec(
        f"""
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  usage_ledger_path: {tmp_path / "usage.jsonl"}
engines:
  m: {{ backend: mock }}
tenants:
  default_tenant: fallback
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
  limits:
    tenant-a: {{ requests_per_minute: 1, tokens_per_minute: 1000 }}
    tenant-b: {{ requests_per_minute: 3, tokens_per_minute: 2000 }}
"""
    )
    settings_calls = 0
    real_server_settings = builder_module._server_settings

    def recording_server_settings(deployment_spec):
        nonlocal settings_calls
        settings_calls += 1
        return real_server_settings(deployment_spec)

    monkeypatch.setattr(builder_module, "_server_settings", recording_server_settings)

    app = build_app_from_spec(spec)

    config = app.state.tenant_limiter._config
    assert settings_calls == 1
    assert config.tenant_for_key("key-a") == "tenant-a"
    assert config.tenant_for_key("key-b") == "tenant-b"
    assert config.tenant_for_key("unmapped-key") == "fallback"
    assert config.limits_for("tenant-a") == TenantLimits(
        requests_per_minute=1,
        tokens_per_minute=1000,
    )
    assert config.limits_for("tenant-b") == TenantLimits(
        requests_per_minute=3,
        tokens_per_minute=2000,
    )


async def test_deployment_tenants_isolate_rate_limits_and_usage_ledger(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    ledger_path = tmp_path / "usage.jsonl"
    app = build_app_from_spec(
        load_deployment_spec(
            f"""
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  usage_ledger_path: {ledger_path}
engines:
  m: {{ backend: mock }}
tenants:
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
  limits:
    tenant-a: {{ requests_per_minute: 1, tokens_per_minute: 200000 }}
    tenant-b: {{ requests_per_minute: 3, tokens_per_minute: 200000 }}
"""
        )
    )
    payload = _chat_body("hi", model="m")
    tenant_a = {"Authorization": "Bearer key-a"}
    tenant_b = {"Authorization": "Bearer key-b"}

    async with _client(app) as client:
        first_a = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=tenant_a,
        )
        limited_a = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=tenant_a,
        )
        first_b = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers=tenant_b,
        )
        usage_a = await client.get("/admin/usage", headers=tenant_a)
        usage_b = await client.get("/admin/usage", headers=tenant_b)

    assert first_a.status_code == 200
    assert limited_a.status_code == 429
    assert limited_a.json()["error"]["code"] == "tenant_rate_limited"
    assert first_b.status_code == 200
    assert usage_a.status_code == 200
    assert set(usage_a.json()["usage"]) == {"tenant-a"}
    assert usage_a.json()["usage"]["tenant-a"]["requests"] == 1
    assert usage_b.status_code == 200
    assert set(usage_b.json()["usage"]) == {"tenant-b"}
    assert usage_b.json()["usage"]["tenant-b"]["requests"] == 1

    totals = UsageLedger(ledger_path).totals()
    assert set(totals) == {"tenant-a", "tenant-b"}
    assert totals["tenant-a"]["requests"] == 1
    assert totals["tenant-b"]["requests"] == 1
    assert "default" not in totals


def test_tenant_preflight_revalidates_before_constructing_owned_backends(monkeypatch):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    spec = load_deployment_spec(
        """
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
engines:
  m: { backend: mock }
tenants:
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
"""
    )
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a")
    created_backends = []
    real_create_backend = builder_module.create_backend

    def recording_create_backend(name, **options):
        backend = real_create_backend(name, **options)
        created_backends.append(backend)
        return backend

    monkeypatch.setattr(builder_module, "create_backend", recording_create_backend)

    with pytest.raises(ValueError, match="unknown API key 'key-b'"):
        build_app_from_spec(spec)

    assert created_backends == []


def test_builder_without_tenant_section_preserves_legacy_app_state():
    app = build_app_from_spec(load_deployment_spec(POOLED_YAML))

    assert not hasattr(app.state, "tenant_limiter")
