"""DeploymentSpec builder wiring for dynamic Kubernetes replica pools."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import httpx
import pytest

import kairyu.deploy.builder as builder_module
from kairyu.deploy.registry import ReplicaConfig
from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.orchestration.replica import ReplicaPool
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


def test_dynamic_http_clients_share_only_tls_and_bound_cross_origin_probe_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tls_context = object()
    ssl_options: list[dict[str, object]] = []
    client_options: list[dict[str, object]] = []

    def create_ssl_context(**options):
        ssl_options.append(options)
        return tls_context

    def construct(**options):
        client_options.append(options)
        return object()

    monkeypatch.setattr(
        builder_module.httpx,
        "create_ssl_context",
        create_ssl_context,
    )
    monkeypatch.setattr(builder_module.httpx, "AsyncClient", construct)

    factory = builder_module._DynamicPoolHTTPClientFactory(
        max_concurrency=128,
        replica_count=lambda: 2,
    )
    first = factory.create_replica_client()
    second = factory.create_replica_client()
    probe = factory.create_probe_client()

    assert len({id(first), id(second), id(probe)}) == 3
    assert ssl_options == [{"verify": True, "trust_env": True}]
    assert len(client_options) == 3
    assert all(options["verify"] is tls_context for options in client_options)
    assert all(options["trust_env"] is False for options in client_options)

    first_limits = client_options[0]["limits"]
    second_limits = client_options[1]["limits"]
    assert first_limits is not second_limits
    for limits in (first_limits, second_limits):
        assert limits.max_connections is None
        assert limits.max_keepalive_connections == 64
        assert limits.keepalive_expiry == 30.0

    probe_limits = client_options[2]["limits"]
    assert probe_limits.max_connections == 16
    assert probe_limits.max_keepalive_connections == 16
    assert probe_limits.keepalive_expiry == 30.0


@pytest.mark.parametrize(
    ("max_concurrency", "replica_count", "expected"),
    [
        (128, 2, 64),
        (128, 0, 64),
        (256, 200, 2),
        (4096, 2, 64),
        (None, 200, 8),
    ],
)
def test_dynamic_replica_keepalive_limit_tracks_gateway_capacity(
    max_concurrency: int | None,
    replica_count: int,
    expected: int,
) -> None:
    assert (
        builder_module._dynamic_replica_keepalive_limit(
            max_concurrency,
            replica_count,
        )
        == expected
    )


@pytest.mark.asyncio
async def test_dynamic_replica_client_keeps_a_concurrent_wave_warm_and_closes_on_removal(
) -> None:
    wave_size = 64
    accepted_connections = 0
    received_requests = 0
    active_handlers = 0
    wave_arrived = (asyncio.Event(), asyncio.Event())
    handlers_done = asyncio.Event()

    async def serve(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        nonlocal accepted_connections, active_handlers, received_requests
        accepted_connections += 1
        active_handlers += 1
        try:
            while True:
                try:
                    await reader.readuntil(b"\r\n\r\n")
                except (
                    asyncio.IncompleteReadError,
                    asyncio.LimitOverrunError,
                    OSError,
                ):
                    return
                wave = received_requests // wave_size
                received_requests += 1
                if wave < len(wave_arrived):
                    if received_requests == (wave + 1) * wave_size:
                        wave_arrived[wave].set()
                    await asyncio.wait_for(
                        wave_arrived[wave].wait(),
                        timeout=5.0,
                    )
                try:
                    writer.write(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Length: 2\r\n"
                        b"Connection: keep-alive\r\n\r\n"
                        b"ok"
                    )
                    await writer.drain()
                except OSError:
                    return
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            active_handlers -= 1
            if active_handlers == 0:
                handlers_done.set()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    factory = builder_module._DynamicPoolHTTPClientFactory(
        max_concurrency=128,
        replica_count=lambda: 2,
    )
    backend = OpenAICompatBackend(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="test-model",
        api_key_env=None,
        client_factory=factory.create_replica_client,
    )
    pool = ReplicaPool({"replica": backend})
    client = backend._get_client()

    try:
        async with server:
            first = await asyncio.gather(
                *(client.get(f"http://127.0.0.1:{port}") for _ in range(wave_size))
            )
            assert [response.text for response in first] == ["ok"] * wave_size
            assert accepted_connections == wave_size

            second = await asyncio.gather(
                *(client.get(f"http://127.0.0.1:{port}") for _ in range(wave_size))
            )
            assert [response.text for response in second] == ["ok"] * wave_size
            assert accepted_connections == wave_size

            await pool.remove_replica("replica")
            assert client.is_closed is True
            await asyncio.wait_for(handlers_done.wait(), timeout=5.0)
    finally:
        if pool.replica_ids:
            await pool.remove_replica("replica")
        if accepted_connections:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(handlers_done.wait(), timeout=5.0)


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
    router_logs = []

    class FakeRouterLog:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.membership_records: list[dict[str, object]] = []
            self.replica_records: list[tuple[tuple, dict]] = []
            self.closed = False
            router_logs.append(self)

        def record_membership(self, actions: dict, **snapshot: object) -> None:
            normalized_snapshot = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in snapshot.items()
            }
            self.membership_records.append({**actions, **normalized_snapshot})

        def record_replica(self, *args, **kwargs) -> None:
            self.replica_records.append((args, kwargs))

        async def shutdown(self) -> None:
            self.closed = True

    monkeypatch.setattr(builder_module, "JsonlRouterLog", FakeRouterLog)
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

    class FakeHTTPClientFactory:
        instances: list[FakeHTTPClientFactory] = []

        def __init__(
            self,
            *,
            max_concurrency: int | None,
            replica_count,
        ) -> None:
            self.max_concurrency = max_concurrency
            self.replica_count = replica_count
            self.replica_clients: list[httpx.AsyncClient] = []
            self.probe_client: httpx.AsyncClient | None = None
            self.instances.append(self)

        def create_replica_client(self) -> httpx.AsyncClient:
            client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            self.replica_clients.append(client)
            return client

        def create_probe_client(self) -> httpx.AsyncClient:
            assert self.probe_client is None
            self.probe_client = httpx.AsyncClient(
                transport=httpx.MockTransport(upstream)
            )
            return self.probe_client

    monkeypatch.setattr(
        builder_module,
        "_DynamicPoolHTTPClientFactory",
        FakeHTTPClientFactory,
    )
    spec = load_deployment_spec(
        """
server:
  max_concurrency: 128
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
    client_factory = FakeHTTPClientFactory.instances[0]
    assert client_factory.max_concurrency == 128
    assert client_factory.replica_count() == 0
    assert app.state.dynamic_pool_http_client_factories == (client_factory,)
    assert prober._client is client_factory.probe_client
    assert prober._owns_client is True

    async with app.router.lifespan_context(app):
        for _ in range(100):
            if pool.replica_ids == ("pod-uid",) and pool.healthy == (True,):
                break
            await asyncio.sleep(0.001)
        assert pool.replica_ids == ("pod-uid",)
        assert client_factory.replica_count() == 1
        assert pool.healthy == (True,)
        assert pool.eligible_ids == ("pod-uid",)
        backend = pool._entries["pod-uid"].backend
        assert backend._client is None
        result = await pool.generate(
            GenerationRequest(
                request_id="origin-local-client",
                prompt="hello",
                sampling_params=SamplingParams(
                    temperature=0,
                    max_tokens=1,
                ),
            )
        )
        assert result.completions[0].text == "shared"
        assert len(client_factory.replica_clients) == 1
        data_client = client_factory.replica_clients[0]
        assert backend._client is data_client
        assert data_client is not client_factory.probe_client
        assert requests.count("/readyz") >= 1
        assert requests[-1] == "/v1/chat/completions"
        assert {timeout for path, timeout in timeouts if path == "/readyz"} == {
            5.0
        }
        assert timeouts[-1] == ("/v1/chat/completions", 60.0)

        # Data and readiness transports are independent while the replica lives.
        assert data_client.is_closed is False
        assert client_factory.probe_client is not None
        assert client_factory.probe_client.is_closed is False

    assert source.polls >= 1
    assert source.closed is True
    assert data_client.is_closed is True
    assert client_factory.probe_client is not None
    assert client_factory.probe_client.is_closed is True
    assert len(router_logs) == 1
    router_log = router_logs[0]
    assert router_log.path == tmp_path / "evidence/membership.jsonl"
    assert router_log.closed is True
    assert len(router_log.replica_records) == 1
    records = router_log.membership_records
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
    assert app.state.dynamic_pool_http_client_factories == ()
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
