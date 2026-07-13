"""DeploymentSpec -> app builder: pool wiring, affinity over HTTP, lifespan (gate C1)."""

import httpx
import pytest

import kairyu.deploy.builder as builder_module
from kairyu.deploy.builder import build_app_from_config, build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.entrypoints.server.settings import ServerSettings
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
    api_key_resolutions = 0
    admin_key_resolutions = 0
    real_server_settings = builder_module._server_settings
    real_resolve_api_keys = ServerSettings.resolve_api_keys
    real_resolve_admin_keys = ServerSettings.resolve_admin_keys

    def recording_server_settings(deployment_spec):
        nonlocal settings_calls
        settings_calls += 1
        return real_server_settings(deployment_spec)

    def recording_resolve_api_keys(settings):
        nonlocal api_key_resolutions
        api_key_resolutions += 1
        return real_resolve_api_keys(settings)

    def recording_resolve_admin_keys(settings):
        nonlocal admin_key_resolutions
        admin_key_resolutions += 1
        return real_resolve_admin_keys(settings)

    monkeypatch.setattr(builder_module, "_server_settings", recording_server_settings)
    monkeypatch.setattr(
        ServerSettings,
        "resolve_api_keys",
        recording_resolve_api_keys,
    )
    monkeypatch.setattr(
        ServerSettings,
        "resolve_admin_keys",
        recording_resolve_admin_keys,
    )

    app = build_app_from_spec(spec)

    config = app.state.tenant_limiter._config
    assert settings_calls == 1
    assert api_key_resolutions == 1
    assert admin_key_resolutions == 1
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


def test_builder_injects_usage_sinks_into_batch_worker(tmp_path):
    spec = load_deployment_spec(
        f"""
server:
  usage_ledger_path: {tmp_path / "usage.jsonl"}
engines:
  m: {{ backend: mock }}
tenants:
  default_tenant: tenant-a
batch:
  data_dir: {tmp_path / "batch-data"}
  max_concurrency: 1
"""
    )

    app = build_app_from_spec(spec)

    assert app.state.batch_worker._usage_ledger is app.state.usage_ledger
    assert app.state.batch_worker._tenant_limiter is app.state.tenant_limiter


async def test_tenant_auth_uses_the_preflight_key_snapshots(monkeypatch):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a,key-b")
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_ADMIN_KEYS", "admin-a,admin-b")
    spec = load_deployment_spec(
        """
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  admin_keys_env: KAIRYU_DEPLOYMENT_ADMIN_KEYS
engines:
  m: { backend: mock }
tenants:
  key_tenants:
    key-a: tenant-a
    key-b: tenant-b
"""
    )
    api_snapshots = iter(
        (
            frozenset({"key-a", "key-b"}),
            frozenset({"key-a", "key-c"}),
        )
    )
    admin_snapshots = iter(
        (
            frozenset({"admin-a", "admin-b"}),
            frozenset({"admin-a", "admin-c"}),
        )
    )
    api_key_resolutions = 0
    admin_key_resolutions = 0

    def resolve_api_keys(_settings):
        nonlocal api_key_resolutions
        api_key_resolutions += 1
        return next(api_snapshots)

    def resolve_admin_keys(_settings):
        nonlocal admin_key_resolutions
        admin_key_resolutions += 1
        return next(admin_snapshots)

    monkeypatch.setattr(ServerSettings, "resolve_api_keys", resolve_api_keys)
    monkeypatch.setattr(ServerSettings, "resolve_admin_keys", resolve_admin_keys)

    app = build_app_from_spec(spec)
    payload = _chat_body("hi", model="m")
    async with _client(app) as client:
        mapped_data = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer key-b"},
        )
        late_data = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer key-c"},
        )
        mapped_admin = await client.post(
            "/admin/undrain",
            headers={"Authorization": "Bearer admin-b"},
        )
        late_admin = await client.post(
            "/admin/undrain",
            headers={"Authorization": "Bearer admin-c"},
        )

    assert mapped_data.status_code == 200
    assert late_data.status_code == 401
    assert mapped_admin.status_code == 200
    assert late_admin.status_code == 401
    assert api_key_resolutions == 1
    assert admin_key_resolutions == 1


@pytest.mark.parametrize("late_failure", ["data", "admin"])
def test_builder_does_not_attempt_late_key_resolution(monkeypatch, late_failure):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a")
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_ADMIN_KEYS", "admin-a")
    spec = load_deployment_spec(
        """
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  admin_keys_env: KAIRYU_DEPLOYMENT_ADMIN_KEYS
engines:
  m: { backend: mock }
tenants:
  key_tenants:
    key-a: tenant-a
"""
    )
    resolution_counts = {"data": 0, "admin": 0}

    def resolve_api_keys(_settings):
        resolution_counts["data"] += 1
        if late_failure == "data" and resolution_counts["data"] > 1:
            raise ValueError("synthetic late data-key resolution failure")
        return frozenset({"key-a"})

    def resolve_admin_keys(_settings):
        resolution_counts["admin"] += 1
        if late_failure == "admin" and resolution_counts["admin"] > 1:
            raise ValueError("synthetic late admin-key resolution failure")
        return frozenset({"admin-a"})

    monkeypatch.setattr(ServerSettings, "resolve_api_keys", resolve_api_keys)
    monkeypatch.setattr(ServerSettings, "resolve_admin_keys", resolve_admin_keys)

    app = build_app_from_spec(spec)

    assert app.state.tenant_limiter is not None
    assert resolution_counts == {"data": 1, "admin": 1}


@pytest.mark.parametrize("failing_key_set", ["data", "admin"])
def test_builder_key_resolution_failure_precedes_owned_backends(
    monkeypatch,
    failing_key_set,
):
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_KEYS", "key-a")
    monkeypatch.setenv("KAIRYU_DEPLOYMENT_ADMIN_KEYS", "admin-a")
    spec = load_deployment_spec(
        """
server:
  api_keys_env: KAIRYU_DEPLOYMENT_KEYS
  admin_keys_env: KAIRYU_DEPLOYMENT_ADMIN_KEYS
engines:
  m: { backend: mock }
"""
    )
    created_backends = []
    real_create_backend = builder_module.create_backend

    def resolve_api_keys(_settings):
        if failing_key_set == "data":
            raise ValueError("synthetic data-key resolution failure")
        return frozenset({"key-a"})

    def resolve_admin_keys(_settings):
        if failing_key_set == "admin":
            raise ValueError("synthetic admin-key resolution failure")
        return frozenset({"admin-a"})

    def recording_create_backend(name, **options):
        backend = real_create_backend(name, **options)
        created_backends.append(backend)
        return backend

    monkeypatch.setattr(ServerSettings, "resolve_api_keys", resolve_api_keys)
    monkeypatch.setattr(ServerSettings, "resolve_admin_keys", resolve_admin_keys)
    monkeypatch.setattr(builder_module, "create_backend", recording_create_backend)

    with pytest.raises(ValueError, match=f"synthetic {failing_key_set}-key"):
        build_app_from_spec(spec)

    assert created_backends == []


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


def test_builder_without_tenant_section_preserves_legacy_app_state(monkeypatch):
    spec = load_deployment_spec(POOLED_YAML)
    resolution_counts = {"data": 0, "admin": 0}
    real_resolve_api_keys = ServerSettings.resolve_api_keys
    real_resolve_admin_keys = ServerSettings.resolve_admin_keys

    def recording_resolve_api_keys(settings):
        resolution_counts["data"] += 1
        return real_resolve_api_keys(settings)

    def recording_resolve_admin_keys(settings):
        resolution_counts["admin"] += 1
        return real_resolve_admin_keys(settings)

    monkeypatch.setattr(
        ServerSettings,
        "resolve_api_keys",
        recording_resolve_api_keys,
    )
    monkeypatch.setattr(
        ServerSettings,
        "resolve_admin_keys",
        recording_resolve_admin_keys,
    )

    app = build_app_from_spec(spec)

    assert not hasattr(app.state, "tenant_limiter")
    assert resolution_counts == {"data": 1, "admin": 1}
