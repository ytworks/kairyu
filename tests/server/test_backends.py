"""GET /backends: resolved attention backend, versions, per-engine map (m13)."""

import importlib.metadata

import httpx

from kairyu.engine.core.attention_selector import AttentionBackendDecision
from kairyu.engine.core.hw_profile import HardwareProfile
from kairyu.engine.mock import MockBackend
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.entrypoints.server import health as health_module
from kairyu.entrypoints.server.app import create_app
from kairyu.orchestration.replica import ReplicaPool


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_backends_shape_and_mock_engines(monkeypatch):
    monkeypatch.delenv("KAIRYU_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(health_module, "probe", lambda: HardwareProfile(arch="cpu"))
    app = create_app(engines={"m1": MockBackend(), "m2": MockBackend()})
    async with _client(app) as client:
        resp = await client.get("/backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["requested_attention_backend"] == "auto"
    assert body["attention_backend"] == "torch"
    assert body["attention_components"] == {
        "prefill": "torch",
        "decode": "torch",
        "kv_mode": "tensor-gather",
    }
    assert body["source"] == "hw_profile"
    assert body["architecture"] == {
        "arch": "cpu",
        "device_name": "cpu",
        "sm": None,
        "kernel_tier": "torch",
    }
    assert body["kernel_tier"] == "torch"
    assert isinstance(body["selection_rationale"], str)
    assert "torch" in body["versions"]
    assert "flashinfer" not in body["versions"]

    engines = {e["model"]: e for e in body["engines"]}
    assert set(engines) == {"m1", "m2"}
    for entry in engines.values():
        assert entry["engine_backend"] == "mock"
        # mock is a remote/echo engine, not local attention
        assert entry["attention_backend"] is None
        assert entry["attention_components"] is None
        assert entry["tensor_parallel_size"] == 1


async def test_backends_explicit_auto_reports_env_source(monkeypatch):
    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "auto")
    monkeypatch.setattr(
        health_module,
        "probe",
        lambda: HardwareProfile(arch="cuda", sm=120),
    )
    app = create_app(engines={"m": MockBackend()})

    async with _client(app) as client:
        resp = await client.get("/backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["requested_attention_backend"] == "auto"
    assert body["attention_backend"] == "flashinfer"
    assert body["source"] == "env"
    assert body["attention_components"] == {
        "prefill": "flashinfer",
        "decode": "flashinfer",
        "kv_mode": "paged-direct",
    }
    assert "stable" in body["selection_rationale"]


async def test_backends_reports_the_local_engine_actual_auto_fallback(monkeypatch):
    monkeypatch.delenv("KAIRYU_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(
        health_module,
        "probe",
        lambda: HardwareProfile(arch="cuda", sm=120),
    )

    class KairyuBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.attention_backend_decision = AttentionBackendDecision(
                requested="auto",
                resolved="torch",
                source="hw_profile",
                components={
                    "prefill": "torch",
                    "decode": "torch",
                    "kv_mode": "tensor-gather",
                },
                rationale=(
                    "automatic torch fallback after flashinfer construction "
                    "raised ModuleNotFoundError"
                ),
                architecture={
                    "arch": "cuda",
                    "device_name": "test-gpu",
                    "sm": 120,
                    "kernel_tier": "fa2",
                },
            )
            self.kv_cache_dtype_requested = "bfloat16"
            self.kv_cache_dtype_resolved = "bfloat16"

    app = create_app(engines={"m": KairyuBackend()})
    async with _client(app) as client:
        resp = await client.get("/backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["requested_attention_backend"] == "auto"
    assert body["attention_backend"] == "torch"
    assert body["attention_components"]["prefill"] == "torch"
    assert "ModuleNotFoundError" in body["selection_rationale"]
    assert "flashinfer" not in body["versions"]
    assert body["architecture"]["sm"] == 120
    assert body["kernel_tier"] == "fa2"
    assert body["engines"][0]["attention_backend"] == "torch"
    assert body["engines"][0]["decision_status"] == "actual"
    assert body["engines"][0]["requested_kv_cache_dtype"] == "bfloat16"
    assert body["engines"][0]["kv_cache_dtype"] == "bfloat16"


async def test_backends_reports_ep_topology_locally_and_through_gateway(
    monkeypatch,
):
    monkeypatch.delenv("KAIRYU_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(health_module, "probe", lambda: HardwareProfile(arch="cpu"))
    expected = {
        "parallelism": "expert_parallel",
        "expert_parallel_size": 4,
        "attention_placement": "replicated",
        "attention_output_placement": "row_parallel",
        "attention_output_parallel_size": 4,
        "attention_output_partial_dtype": "bfloat16",
        "execution_mode": "replicated-attention-correctness",
        "pipeline_depth": 1,
        "decode_mode": "eager",
    }

    class KairyuBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.parallelism_metadata = {
                **expected,
                "kv_cache_dtype": "bfloat16",
            }
            self.kv_cache_dtype_requested = "bfloat16"
            self.kv_cache_dtype_resolved = "bfloat16"

    replica_app = create_app(
        engines={
            "default": KairyuBackend(),
            # The gateway backend targets only "default"; topology from this
            # unrelated co-hosted model must not enter the pool aggregate.
            "unrelated": MockBackend(tensor_parallel_size=8),
        }
    )
    async with _client(replica_app) as client:
        replica_response = await client.get("/backends")

    assert replica_response.status_code == 200
    replica_entry = replica_response.json()["engines"][0]
    assert {field: replica_entry[field] for field in expected} == expected
    assert "tensor_parallel_size" not in replica_entry

    replica_backend = OpenAICompatBackend(
        base_url="http://replica/v1",
        model="default",
        api_key_env=None,
        transport=httpx.ASGITransport(app=replica_app),
    )
    gateway_app = create_app(
        engines={"qwen3-235b": ReplicaPool([replica_backend])},
    )
    async with _client(gateway_app) as client:
        gateway_response = await client.get("/backends")

    assert gateway_response.status_code == 200
    pool_entry = gateway_response.json()["engines"][0]
    assert {field: pool_entry[field] for field in expected} == expected
    assert {
        field: pool_entry["via_replica"][field] for field in expected
    } == expected
    assert "tensor_parallel_size" not in pool_entry
    assert "tensor_parallel_size" not in pool_entry["via_replica"]


async def test_backends_never_relabels_malformed_ep_metadata_as_tp():
    class KairyuBackend(MockBackend):
        def __init__(self):
            super().__init__(tensor_parallel_size=1)
            self.expert_parallel_size = 4
            self.parallelism_metadata = {
                "parallelism": "expert_parallel",
                "expert_parallel_size": 4,
                "attention_placement": "replicated",
                # Missing the remaining required runtime topology.
            }

    app = create_app(engines={"qwen3-235b": KairyuBackend()})
    async with _client(app) as client:
        response = await client.get("/backends")

    assert response.status_code == 200
    entry = response.json()["engines"][0]
    assert entry["parallelism"] == "expert_parallel"
    assert entry["expert_parallel_size"] == 4
    assert entry["parallelism_metadata_status"] == "invalid"
    assert "tensor_parallel_size" not in entry


async def test_backends_does_not_invent_torch_after_invalid_selection(monkeypatch):
    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "invalid")
    monkeypatch.setattr(
        health_module,
        "probe",
        lambda: HardwareProfile(arch="cuda", sm=120),
    )
    app = create_app(engines={"m": MockBackend()})

    async with _client(app) as client:
        resp = await client.get("/backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["requested_attention_backend"] == "invalid"
    assert body["attention_backend"] == "unavailable"
    assert body["attention_components"] == {}
    assert "ValueError" in body["selection_rationale"]


async def test_backends_fa4_components_and_missing_versions(monkeypatch):
    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashattention4")
    monkeypatch.setattr(
        health_module,
        "probe",
        lambda: HardwareProfile(arch="cuda", sm=120),
    )

    def package_version(name):
        if name == "torch":
            return "test-torch"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", package_version)

    class KairyuBackend(MockBackend):
        pass

    app = create_app(engines={"m": KairyuBackend()})
    async with _client(app) as client:
        resp = await client.get("/backends")

    assert resp.status_code == 200
    body = resp.json()
    components = {
        "prefill": "flashattention4",
        "decode": "flashinfer",
        "kv_mode": "paged-materialized",
    }
    assert body["requested_attention_backend"] == "flashattention4"
    assert body["attention_backend"] == "flashattention4"
    assert body["attention_components"] == components
    assert body["versions"] == {
        "torch": "test-torch",
        "flashinfer": None,
        "flash-attn-4": None,
    }
    assert body["architecture"]["sm"] == 120
    assert body["engines"][0]["attention_components"] == components


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


async def test_backends_gateway_aggregates_replica_through_pool(monkeypatch):
    monkeypatch.delenv("KAIRYU_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(
        health_module,
        "probe",
        lambda: HardwareProfile(arch="cuda", sm=120),
    )
    # A gateway (all engines are ReplicaPools) runs no local attention, so its own
    # probe reports the process kernel; for each pool it must adopt the replica's
    # /backends. Wire the pool's replica to an in-process "replica" app via ASGI
    # transport so the whole fetch path (URL derivation + transport reuse) runs.
    class KairyuBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.kv_cache_dtype_requested = "auto"
            self.kv_cache_dtype_resolved = "bfloat16"

    replica_app = create_app(engines={"default": KairyuBackend()})
    replica_backend = OpenAICompatBackend(
        base_url="http://replica/v1",
        model="default",
        api_key_env=None,
        transport=httpx.ASGITransport(app=replica_app),
    )
    gateway_app = create_app(engines={"llama": ReplicaPool([replica_backend])})

    async with _client(gateway_app) as client:
        resp = await client.get("/backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "gateway"
    pool = {e["model"]: e for e in body["engines"]}["llama"]
    assert pool["engine_backend"] == "replica-pool"
    # WITHOUT aggregation a replica-pool engine would be null; a non-null value
    # here proves the gateway fetched and adopted the replica's /backends.
    assert pool["attention_backend"] in {"torch", "flashinfer"}
    assert pool["via_replica"]["attention_backend"] == pool["attention_backend"]
    assert pool["attention_components"] == body["attention_components"]
    assert pool["via_replica"]["attention_components"] == body["attention_components"]
    assert "torch" in pool["via_replica"]["versions"]
    assert pool["via_replica"]["architecture"] == body["architecture"]
    assert pool["via_replica"]["requested_attention_backend"] == body["requested_attention_backend"]
    assert pool["via_replica"]["source"] == body["source"]
    assert pool["tensor_parallel_size"] == 1
    assert pool["via_replica"]["tensor_parallel_size"] == 1
    assert pool["requested_kv_cache_dtype"] == "auto"
    assert pool["kv_cache_dtype"] == "bfloat16"
    assert pool["via_replica"]["requested_kv_cache_dtype"] == "auto"
    assert pool["via_replica"]["kv_cache_dtype"] == "bfloat16"


async def test_backends_gateway_pool_without_backends_endpoint_degrades():
    # A replica that does not expose /backends (plain MockBackend, no
    # fetch_backends) -> probe returns None -> attention stays null, no crash.
    gateway_app = create_app(engines={"llama": ReplicaPool([MockBackend()])})
    async with _client(gateway_app) as client:
        resp = await client.get("/backends")

    assert resp.status_code == 200
    pool = {e["model"]: e for e in resp.json()["engines"]}["llama"]
    assert pool["engine_backend"] == "replica-pool"
    assert pool["attention_backend"] is None
    assert "via_replica" not in pool


async def test_backends_does_not_hide_mixed_local_engine_decisions(monkeypatch):
    monkeypatch.delenv("KAIRYU_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(health_module, "probe", lambda: HardwareProfile(arch="cpu"))

    class KairyuBackend(MockBackend):
        def __init__(self, sm, kv_mode):
            super().__init__()
            self.attention_backend_decision = AttentionBackendDecision(
                requested="flashattention4",
                resolved="flashattention4",
                source="env",
                components={
                    "prefill": "flashattention4",
                    "decode": "flashinfer",
                    "kv_mode": kv_mode,
                },
                rationale=f"explicit backend on sm{sm}",
                architecture={
                    "arch": "cuda",
                    "device_name": f"gpu-sm{sm}",
                    "sm": sm,
                    "kernel_tier": "full" if sm == 90 else "fa2",
                },
            )

    app = create_app(
        engines={
            "hopper": KairyuBackend(90, "paged-direct"),
            "blackwell": KairyuBackend(120, "paged-materialized"),
        }
    )
    async with _client(app) as client:
        response = await client.get("/backends")

    body = response.json()
    assert response.status_code == 200
    assert body["attention_backend"] == "mixed"
    assert body["source"] == "engine"
    assert body["kernel_tier"] == "mixed"
    assert set(body["architecture"]) == {"hopper", "blackwell"}
    by_model = {entry["model"]: entry for entry in body["engines"]}
    assert by_model["hopper"]["architecture"]["sm"] == 90
    assert by_model["blackwell"]["architecture"]["sm"] == 120
    assert all(entry["decision_status"] == "actual" for entry in by_model.values())
