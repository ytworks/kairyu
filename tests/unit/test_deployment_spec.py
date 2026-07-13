"""DeploymentSpec parsing and validation (design m7 D3)."""

from textwrap import indent

import pytest
import yaml
from pydantic import ValidationError

from kairyu.deploy.spec import (
    BackendSpec,
    _UniqueKeySafeLoader,
    load_deployment_spec,
)

GATEWAY_YAML = """
server:
  port: 8100
  api_keys_env: KAIRYU_API_KEYS
  max_concurrency: 64
engines:
  small:
    backend: mock
pools:
  llama-70b:
    replicas:
      - backend: openai
        options: { base_url: "http://gpu-0:8000/v1", model: "llama", api_key_env: null }
      - backend: openai
        options: { base_url: "http://gpu-1:8000/v1", model: "llama", api_key_env: null }
    unhealthy_after: 2
    probe_interval_s: 1.5
"""


def test_gateway_spec_parses():
    spec = load_deployment_spec(GATEWAY_YAML)
    assert spec.server.port == 8100
    assert spec.server.api_keys_env == "KAIRYU_API_KEYS"
    assert spec.server.max_concurrency == 64
    assert spec.engines["small"].backend == "mock"
    pool = spec.pools["llama-70b"]
    assert len(pool.replicas) == 2
    assert pool.unhealthy_after == 2
    assert pool.probe_interval_s == 1.5
    assert pool.replicas[0].options["api_key_env"] is None  # keyless node-to-node


def test_defaults():
    spec = load_deployment_spec("engines:\n  m:\n    backend: mock\n")
    assert spec.server.host == "0.0.0.0"
    assert spec.server.port == 8000
    assert spec.server.api_keys_env is None
    assert spec.server.metrics is True
    assert spec.orchestrator is None
    assert spec.batch is None
    assert spec.tenants is None


def _deployment_with_tenants(tenants: str) -> str:
    return (
        "server:\n"
        "  api_keys_env: KAIRYU_TENANT_KEYS\n"
        "engines:\n"
        "  m: { backend: mock }\n"
        "tenants:\n"
        f"{indent(tenants, '  ')}\n"
    )


def test_tenant_section_parses_two_tenants(monkeypatch):
    monkeypatch.setenv("KAIRYU_TENANT_KEYS", "key-a,key-b")
    spec = load_deployment_spec(
        _deployment_with_tenants(
            """default_tenant: default
key_tenants:
  key-a: team-a
  key-b: team-b
limits:
  team-a: {requests_per_minute: 60, tokens_per_minute: 10000}
  team-b: {requests_per_minute: 120, tokens_per_minute: 20000}"""
        )
    )

    assert spec.tenants is not None
    assert spec.tenants.default_tenant == "default"
    assert spec.tenants.key_tenants == {"key-a": "team-a", "key-b": "team-b"}
    assert spec.tenants.limits["team-a"].requests_per_minute == 60
    assert spec.tenants.limits["team-b"].tokens_per_minute == 20_000
    assert "key-a" not in repr(spec.tenants)
    with pytest.raises(ValidationError, match="frozen"):
        spec.tenants.default_tenant = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        spec.tenants.limits["team-a"].requests_per_minute = 1


def test_tenant_api_keys_are_internal_but_excluded_from_repr_and_dumps(monkeypatch):
    api_secret = "deployment-api-secret"
    monkeypatch.setenv("KAIRYU_TENANT_KEYS", api_secret)
    spec = load_deployment_spec(
        _deployment_with_tenants(
            f"""key_tenants: {{{api_secret}: team-a}}
limits:
  team-a: {{requests_per_minute: 60}}"""
        )
    )

    assert spec.tenants is not None
    assert spec.tenants.key_tenants == {api_secret: "team-a"}
    assert "key_tenants" not in spec.tenants.model_dump()
    assert "key_tenants" not in spec.model_dump()["tenants"]
    for external_form in (repr(spec.tenants), repr(spec), spec.model_dump_json()):
        assert api_secret not in external_form


def test_tenant_section_rejects_unknown_mapping_without_leaking_resolved_keys(
    monkeypatch,
):
    monkeypatch.setenv("KAIRYU_TENANT_KEYS", "valid-secret")

    with pytest.raises(ValueError) as exc_info:
        load_deployment_spec(
            _deployment_with_tenants(
                """key_tenants:
  valid-secret: team-valid
  unknown-key: team-a"""
            )
        )

    message = str(exc_info.value)
    assert "unknown API key 'unknown-key'" in message
    assert "valid-secret" not in message


@pytest.mark.parametrize(
    "tenants",
    [
        'default_tenant: ""\nkey_tenants: {key-a: team-a}',
        'key_tenants: {"": team-a}',
        'key_tenants: {key-a: ""}',
    ],
)
def test_tenant_section_rejects_empty_names(monkeypatch, tenants):
    monkeypatch.setenv("KAIRYU_TENANT_KEYS", "key-a")

    with pytest.raises(ValueError, match="must not be empty"):
        load_deployment_spec(_deployment_with_tenants(tenants))


def test_tenant_section_rejects_orphan_limits(monkeypatch):
    monkeypatch.setenv("KAIRYU_TENANT_KEYS", "key-a")

    with pytest.raises(ValueError, match="limits reference unknown tenant 'orphan'"):
        load_deployment_spec(
            _deployment_with_tenants(
                """key_tenants: {key-a: team-a}
limits:
  orphan: {requests_per_minute: 60}"""
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("requests_per_minute", 0), ("tokens_per_minute", -1)],
)
def test_tenant_section_rejects_nonpositive_limits(monkeypatch, field, value):
    monkeypatch.setenv("KAIRYU_TENANT_KEYS", "key-a")

    with pytest.raises(ValueError, match=field):
        load_deployment_spec(
            _deployment_with_tenants(
                f"""key_tenants: {{key-a: team-a}}
limits:
  team-a: {{{field}: {value}}}"""
            )
        )


@pytest.mark.parametrize(
    ("tenants", "unknown_field", "input_secret"),
    [
        (
            "key_tenant: {leaked-input-secret: team-a}",
            "key_tenant",
            "leaked-input-secret",
        ),
        (
            """key_tenants: {valid-secret: team-a}
limits:
  team-a: {requests_per_mintue: 60}""",
            "requests_per_mintue",
            "valid-secret",
        ),
    ],
)
def test_tenant_section_rejects_unknown_fields_without_leaking_input(
    monkeypatch,
    tenants,
    unknown_field,
    input_secret,
):
    monkeypatch.setenv("KAIRYU_TENANT_KEYS", "valid-secret")

    with pytest.raises(ValidationError) as exc_info:
        load_deployment_spec(_deployment_with_tenants(tenants))

    message = str(exc_info.value)
    assert unknown_field in message
    assert input_secret not in message


def test_duplicate_nested_tenant_mapping_key_is_rejected(monkeypatch):
    monkeypatch.setenv("KAIRYU_TENANT_KEYS", "key-a")
    yaml_text = _deployment_with_tenants(
        """key_tenants:
  key-a: team-a
  key-a: team-b"""
    )

    with pytest.raises(ValueError) as exc_info:
        load_deployment_spec(yaml_text)

    message = str(exc_info.value)
    assert "duplicate mapping key 'key-a'" in message
    assert "tenants.key_tenants" in message


@pytest.mark.parametrize(
    ("yaml_text", "duplicate_key", "mapping_path"),
    [
        pytest.param(
            """engines:
  m:
    backend: mock
    backend: openai
""",
            "backend",
            "engines.m",
            id="regular-mapping",
        ),
        pytest.param(
            """engines:
  m:
    <<: &backend_defaults
      backend: mock
      backend: openai
""",
            "backend",
            "engines.m.<<",
            id="inline-merge-source",
        ),
        pytest.param(
            """mock_defaults: &mock_defaults
  backend: mock
openai_defaults: &openai_defaults
  backend: openai
engines:
  m:
    <<: *mock_defaults
    <<: *openai_defaults
""",
            "<<",
            "engines.m",
            id="repeated-merge-key",
        ),
    ],
)
def test_duplicate_key_loader_rejects_duplicates_before_merge_flattening(
    yaml_text,
    duplicate_key,
    mapping_path,
):
    with pytest.raises(ValueError) as exc_info:
        load_deployment_spec(yaml_text)

    message = str(exc_info.value)
    assert f"duplicate mapping key {duplicate_key!r}" in message
    assert f"at {mapping_path}" in message


def test_duplicate_key_loader_preserves_safe_yaml_merge_behavior():
    spec = load_deployment_spec(
        """backend_defaults: &backend_defaults
  backend: mock
  health_url: http://default/readyz
engines:
  m:
    <<: *backend_defaults
    health_url: http://override/readyz
"""
    )

    assert spec.engines["m"].backend == "mock"
    assert spec.engines["m"].health_url == "http://override/readyz"


def test_duplicate_key_loader_preserves_recursive_alias_identity():
    yaml_text = """recursive: &recursive
  self: *recursive
engines:
  m: {backend: mock}
"""

    loaded = yaml.load(yaml_text, Loader=_UniqueKeySafeLoader)

    recursive = loaded["recursive"]
    assert recursive["self"] is recursive
    spec = load_deployment_spec(yaml_text)
    assert spec.engines["m"].backend == "mock"


def test_tenants_absent_does_not_resolve_api_key_environment(monkeypatch):
    monkeypatch.delenv("KAIRYU_MISSING_TENANT_KEYS", raising=False)
    spec = load_deployment_spec(
        """server:
  api_keys_env: KAIRYU_MISSING_TENANT_KEYS
engines:
  m: {backend: mock}
"""
    )

    assert spec.tenants is None


def test_health_url_derived_from_base_url():
    entry = BackendSpec(
        backend="openai", options={"base_url": "http://gpu-0:8000/v1", "model": "m"}
    )
    assert entry.resolved_health_url() == "http://gpu-0:8000/readyz"


def test_health_url_explicit_and_absent():
    explicit = BackendSpec(backend="openai", health_url="http://probe:9/x")
    assert explicit.resolved_health_url() == "http://probe:9/x"
    local = BackendSpec(backend="mock")
    assert local.resolved_health_url() is None


def test_empty_spec_rejected():
    with pytest.raises(ValueError, match="at least one engine or pool"):
        load_deployment_spec("server:\n  port: 8000\n")


def test_duplicate_names_rejected():
    yaml_text = """
engines:
  m: { backend: mock }
pools:
  m:
    replicas: [{ backend: mock }]
"""
    with pytest.raises(ValueError, match="unique"):
        load_deployment_spec(yaml_text)


def test_non_mapping_yaml_rejected():
    with pytest.raises(ValueError, match="mapping"):
        load_deployment_spec("- a\n- b\n")


MULTI_ORCH_YAML = """
engines:
  m: { backend: mock }
orchestrators:
  kairyu-auto: { spec: auto.yaml }
  kairyu-auto-max: { spec: auto_max.yaml }
"""


def test_named_orchestrators_parse():
    spec = load_deployment_spec(MULTI_ORCH_YAML)
    assert spec.orchestrator is None
    assert spec.orchestrators["kairyu-auto"].spec == "auto.yaml"
    assert spec.orchestrators["kairyu-auto-max"].spec == "auto_max.yaml"


def test_orchestrator_name_colliding_with_engine_rejected():
    yaml_text = """
engines:
  m: { backend: mock }
orchestrators:
  m: { spec: auto.yaml }
"""
    with pytest.raises(ValueError, match="collide"):
        load_deployment_spec(yaml_text)


def test_legacy_orchestrator_plus_named_kairyu_auto_rejected():
    yaml_text = """
engines:
  m: { backend: mock }
orchestrator: { spec: auto.yaml }
orchestrators:
  kairyu-auto: { spec: other.yaml }
"""
    with pytest.raises(ValueError, match="declare kairyu-auto once"):
        load_deployment_spec(yaml_text)


def test_legacy_orchestrator_composes_with_other_named():
    yaml_text = """
engines:
  m: { backend: mock }
orchestrator: { spec: auto.yaml }
orchestrators:
  kairyu-auto-max: { spec: max.yaml }
"""
    spec = load_deployment_spec(yaml_text)
    assert spec.orchestrator is not None
    assert set(spec.orchestrators) == {"kairyu-auto-max"}


def test_empty_orchestrator_name_rejected():
    yaml_text = """
engines:
  m: { backend: mock }
orchestrators:
  "": { spec: auto.yaml }
"""
    with pytest.raises(ValueError, match="non-empty"):
        load_deployment_spec(yaml_text)
