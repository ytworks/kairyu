"""DeploymentSpec builder wiring for dynamic Kubernetes replica pools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

import kairyu.deploy.builder as builder_module
from kairyu.deploy.registry import ReplicaConfig
from kairyu.deploy.spec import load_deployment_spec


class _FakeEndpointSliceDiscovery:
    instances: list[_FakeEndpointSliceDiscovery] = []

    def __init__(
        self,
        service: str,
        namespace: str,
        port: str | int,
        **options,
    ) -> None:
        self.service = service
        self.namespace = namespace
        self.port = port
        self.options = options
        self.closed = False
        self.polls = 0
        self.instances.append(self)

    async def poll(self) -> dict[str, ReplicaConfig]:
        self.polls += 1
        return {
            "pod-uid": ReplicaConfig(
                address="http://10.2.0.7:8000/v1",
                model=self.options["model"],
                api_key_env=self.options["api_key_env"],
            )
        }

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_dynamic_pool_wires_discovery_prober_log_and_lifespan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace_path = tmp_path / "namespace"
    namespace_path.write_text("serving-system\n", encoding="utf-8")
    monkeypatch.setattr(
        builder_module,
        "_SERVICE_ACCOUNT_NAMESPACE_PATH",
        namespace_path,
    )
    _FakeEndpointSliceDiscovery.instances = []
    monkeypatch.setattr(
        builder_module,
        "KubernetesEndpointSliceDiscovery",
        _FakeEndpointSliceDiscovery,
    )
    spec = load_deployment_spec(
        """
pools:
  qwen:
    discovery:
      type: kubernetes_endpoints
      service: qwen-replicas
      port: http
      model: Qwen/Qwen3-32B
      api_key_env: null
      poll_interval_s: 0.001
    probe_interval_s: 0.001
    placement_log_path: evidence/membership.jsonl
"""
    )

    app = builder_module.build_app_from_spec(spec, base_dir=tmp_path)
    reconciler = app.state.reconcilers[0]
    prober = app.state.probers[0]
    pool = reconciler._pool
    source = _FakeEndpointSliceDiscovery.instances[0]
    assert pool.replica_count == 0
    assert len(app.state.reconcilers) == 1
    assert len(app.state.probers) == 1
    assert source.service == "qwen-replicas"
    assert source.namespace == "serving-system"
    assert source.port == "http"
    assert source.options == {
        "model": "Qwen/Qwen3-32B",
        "api_key_env": None,
        "scheme": "http",
        "address_family_preference": "ipv4",
    }

    prober._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"status": "ready"})
        )
    )
    async with app.router.lifespan_context(app):
        for _ in range(100):
            if pool.replica_ids == ("pod-uid",) and pool.healthy == (True,):
                break
            await asyncio.sleep(0.001)
        assert pool.replica_ids == ("pod-uid",)
        assert pool.healthy == (True,)
        assert pool.eligible_ids == ("pod-uid",)

    assert source.polls >= 1
    assert source.closed is True
    assert prober._client.is_closed is True
    records = [
        json.loads(line)
        for line in (tmp_path / "evidence/membership.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["reason"] for record in records] == [
        "replica_added",
        "probe_succeeded",
    ]
    assert [record["sequence"] for record in records] == [1, 2]
    assert records[0]["added"] == ["pod-uid"]
    assert records[0]["eligible_ids"] == []
    assert records[1]["eligible"] == ["pod-uid"]
    assert records[1]["healthy_ids"] == ["pod-uid"]
    assert records[1]["eligible_ids"] == ["pod-uid"]
    assert (
        records[0]["replica_generation"]
        == records[1]["replica_generation"]
    )


@pytest.mark.asyncio
async def test_explicit_namespace_and_numeric_port_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeEndpointSliceDiscovery.instances = []
    monkeypatch.setattr(
        builder_module,
        "KubernetesEndpointSliceDiscovery",
        _FakeEndpointSliceDiscovery,
    )
    spec = load_deployment_spec(
        """
pools:
  model:
    discovery:
      type: kubernetes_endpoints
      service: replicas
      namespace: explicit
      port: 9000
      poll_interval_s: 60
"""
    )

    app = builder_module.build_app_from_spec(spec)
    source = _FakeEndpointSliceDiscovery.instances[0]
    assert source.namespace == "explicit"
    assert source.port == 9000
    # No bootstrap backend is synthesized for a discovery-owned pool.
    assert app.state.reconcilers[0]._pool.replica_count == 0
    await app.state.reconcilers[0].aclose()
    assert source.closed is True


@pytest.mark.asyncio
async def test_static_pool_remains_synchronous_and_nonempty() -> None:
    spec = load_deployment_spec(
        """
pools:
  static:
    replicas:
      - {backend: mock}
"""
    )

    app = builder_module.build_app_from_spec(spec)
    assert app.state.reconcilers == ()
    assert app.state.probers == ()
    async with app.router.lifespan_context(app):
        pass


def test_pool_placement_logs_must_resolve_to_unique_files(
    tmp_path: Path,
) -> None:
    spec = load_deployment_spec(
        """
pools:
  first:
    replicas:
      - {backend: mock}
    placement_log_path: evidence/shared.jsonl
  second:
    replicas:
      - {backend: mock}
    placement_log_path: evidence/../evidence/shared.jsonl
"""
    )

    with pytest.raises(ValueError, match="placement_log_path values must be unique"):
        builder_module.build_app_from_spec(spec, base_dir=tmp_path)
