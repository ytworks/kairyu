"""DeploymentSpec parsing and validation (design m7 D3)."""

import pytest

from kairyu.deploy.spec import BackendSpec, load_deployment_spec

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


def test_health_url_derived_from_base_url():
    entry = BackendSpec(
        backend="openai", options={"base_url": "http://gpu-0:8000/v1", "model": "m"}
    )
    assert entry.resolved_health_url() == "http://gpu-0:8000/health"


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
