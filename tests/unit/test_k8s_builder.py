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
from kairyu.engine.backend import GenerationRequest
from kairyu.sampling_params import SamplingParams


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


def test_dynamic_http_client_does_not_cap_active_or_warmed_replicas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def construct(**options):
        captured.update(options)
        return sentinel

    monkeypatch.setattr(builder_module.httpx, "AsyncClient", construct)

    assert builder_module._create_dynamic_pool_http_client() is sentinel
    limits = captured["limits"]
    assert limits.max_connections is None
    assert limits.max_keepalive_connections is None
    assert limits.keepalive_expiry == 30.0


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
    requests: list[str] = []
    timeouts: list[tuple[str, float]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        timeouts.append(
            (
                request.url.path,
                request.extensions["timeout"]["read"],
            )
        )
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "shared"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    shared_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    monkeypatch.setattr(
        builder_module,
        "_create_dynamic_pool_http_client",
        lambda: shared_client,
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
    assert app.state.dynamic_pool_http_clients == (shared_client,)
    assert prober._client is shared_client
    assert prober._owns_client is False

    async with app.router.lifespan_context(app):
        for _ in range(100):
            if pool.replica_ids == ("pod-uid",) and pool.healthy == (True,):
                break
            await asyncio.sleep(0.001)
        assert pool.replica_ids == ("pod-uid",)
        assert pool.healthy == (True,)
        assert pool.eligible_ids == ("pod-uid",)
        backend = pool._entries["pod-uid"].backend
        assert backend._client is shared_client
        result = await backend.generate(
            GenerationRequest(
                request_id="shared-client",
                prompt="hello",
                sampling_params=SamplingParams(
                    temperature=0,
                    max_tokens=1,
                ),
            )
        )
        assert result.completions[0].text == "shared"
        assert requests.count("/readyz") >= 1
        assert requests[-1] == "/v1/chat/completions"
        assert {timeout for path, timeout in timeouts if path == "/readyz"} == {
            5.0
        }
        assert timeouts[-1] == ("/v1/chat/completions", 60.0)

        # A reconciler removes individual backends during every churn epoch.
        # Their shutdown must leave the pool/prober's shared transport alive.
        await backend.shutdown()
        assert shared_client.is_closed is False

    assert source.polls >= 1
    assert source.closed is True
    assert shared_client.is_closed is True
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
async def test_dynamic_http_client_concurrent_close_finishes_once_when_cancelled() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient:
        def __init__(self) -> None:
            self.is_closed = False
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            # httpx exposes is_closed as soon as aclose starts. A second owner
            # waiter must still await the in-flight close task.
            self.is_closed = True
            started.set()
            await release.wait()

    client = BlockingClient()
    owner = builder_module._DynamicPoolHTTPClientOwner(client)
    first = asyncio.create_task(owner.shutdown())
    await asyncio.wait_for(started.wait(), timeout=0.2)
    second = asyncio.create_task(owner.shutdown())
    await asyncio.sleep(0)
    assert second.done() is False

    first.cancel()
    await asyncio.sleep(0)
    assert first.done() is False
    assert second.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second

    assert client.is_closed is True
    assert client.close_calls == 1
    await owner.shutdown()
    assert client.close_calls == 1


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
    assert app.state.dynamic_pool_http_clients == ()
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
