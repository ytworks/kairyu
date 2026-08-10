"""GET /backends: resolved attention backend, versions, per-engine map (m13)."""

import importlib.metadata

import httpx
import pytest

from kairyu.engine.core.attention_selector import AttentionBackendDecision
from kairyu.engine.core.hw_profile import HardwareProfile
from kairyu.engine.mock import MockBackend
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.zmq_backend import ZmqEngineBackend
from kairyu.entrypoints.server import health as health_module
from kairyu.entrypoints.server.app import create_app
from kairyu.models.generation import GenerationDefaults
from kairyu.orchestration.orchestrator import Orchestrator
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


async def test_backends_reports_resolved_generation_defaults_for_native_engine(
    monkeypatch,
):
    monkeypatch.setattr(health_module, "probe", lambda: HardwareProfile(arch="cpu"))

    class KairyuBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.generation_defaults = GenerationDefaults(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.05,
                repetition_penalty=1.1,
                mode="auto",
                source="generation_config.json",
            )

    app = create_app({"native": KairyuBackend()})
    async with _client(app) as client:
        response = await client.get("/backends")

    entry = response.json()["engines"][0]
    assert entry["generation_config"] == "auto"
    assert entry["generation_config_source"] == "generation_config.json"
    assert entry["generation_defaults"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
    }


async def test_backends_reports_only_homogeneous_local_pool_generation_defaults(
    monkeypatch,
):
    monkeypatch.setattr(health_module, "probe", lambda: HardwareProfile(arch="cpu"))

    class KairyuBackend(MockBackend):
        def __init__(self, temperature):
            super().__init__()
            self.generation_defaults = GenerationDefaults(
                temperature=temperature,
                mode="auto",
                source="generation_config.json",
            )

    app = create_app(
        {
            "same": ReplicaPool(
                [KairyuBackend(0.6), KairyuBackend(0.6)]
            ),
            "mixed": ReplicaPool(
                [KairyuBackend(0.6), KairyuBackend(0.7)]
            ),
        }
    )
    async with _client(app) as client:
        response = await client.get("/backends")

    entries = {entry["model"]: entry for entry in response.json()["engines"]}
    assert entries["same"]["generation_config"] == "auto"
    assert entries["same"]["generation_defaults"]["temperature"] == 0.6
    assert "generation_config" not in entries["mixed"]
    assert "generation_defaults" not in entries["mixed"]


async def test_backends_reports_orchestrator_defaults_per_worker_and_only_collapses_complete_policy(
    monkeypatch,
):
    monkeypatch.setattr(health_module, "probe", lambda: HardwareProfile(arch="cpu"))

    class KairyuBackend(MockBackend):
        def __init__(self, temperature):
            super().__init__()
            self.generation_defaults = GenerationDefaults(
                temperature=temperature,
                mode="auto",
                source="generation_config.json",
            )

    app = create_app(
        {},
        orchestrators={
            "uniform": Orchestrator(
                {"tier1": KairyuBackend(0.6), "tier2": KairyuBackend(0.6)}
            ),
            "mixed": Orchestrator(
                {"tier1": KairyuBackend(0.6), "tier2": MockBackend()}
            ),
        },
    )
    async with _client(app) as client:
        response = await client.get("/backends")

    entries = {entry["model"]: entry for entry in response.json()["engines"]}
    assert entries["uniform"]["engine_backend"] == "orchestrator"
    assert entries["uniform"]["generation_defaults"]["temperature"] == 0.6
    assert set(entries["uniform"]["generation_defaults_by_worker"]) == {
        "tier1",
        "tier2",
    }
    tier1 = entries["mixed"]["generation_defaults_by_worker"]["tier1"]
    assert tier1["generation_defaults"]["temperature"] == 0.6
    assert entries["mixed"]["generation_defaults_by_worker"]["tier2"] is None
    assert "generation_config" not in entries["mixed"]
    assert "generation_defaults" not in entries["mixed"]


async def test_backends_does_not_report_unstarted_process_worker_placeholder(
    monkeypatch,
):
    monkeypatch.setattr(health_module, "probe", lambda: HardwareProfile(arch="cpu"))
    process_worker = ZmqEngineBackend(
        model_path="/models/not-loaded-yet",
        num_pages=64,
    )
    app = create_app(
        {},
        orchestrators={
            "auto": Orchestrator({"native": process_worker}),
        },
    )

    async with _client(app) as client:
        response = await client.get("/backends")

    entry = response.json()["engines"][0]
    assert entry["generation_defaults_by_worker"] == {"native": None}
    assert "generation_config" not in entry
    assert "generation_defaults" not in entry


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


@pytest.mark.parametrize(
    ("active", "selected", "library", "version", "reason"),
    [
        (
            True,
            "direct_nccl_ctypes",
            "libnccl.so.2",
            "2.29.7",
            "initialized on every rank",
        ),
        (
            False,
            "torch.distributed:nccl",
            None,
            None,
            "rank 2: OSError: library absent",
        ),
        (
            False,
            "unavailable",
            None,
            None,
            "runtime collective fallback is unsafe",
        ),
    ],
)
async def test_backends_reports_attention_dp_collective_transport_only_when_active(
    active,
    selected,
    library,
    version,
    reason,
):
    transport = {
        "selected_backend": selected,
        "fallback_backend": "torch.distributed:nccl",
        "direct_nccl_active": active,
        "direct_nccl_library": library,
        "direct_nccl_version": version,
        "selection_reason": reason,
    }

    class KairyuBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.expert_parallel_size = 4
            self.parallelism_metadata = {
                "parallelism": "expert_parallel",
                "expert_parallel_size": 4,
                "attention_placement": "request_owned_data_parallel",
                "attention_output_placement": "replicated",
                "attention_output_parallel_size": 1,
                "attention_output_partial_dtype": None,
                "execution_mode": "request-owned-attention-dp",
                "pipeline_depth": 5,
                "decode_mode": "eager",
                "kv_cache_dtype": "bfloat16",
                "attention_data_parallel_size": 4,
                "attention_tensor_parallel_size": 1,
                "moe_dispatcher": "nvfp4_allgather_reduce_scatter",
                "sampling_ownership": "request_owner",
                    "kv_cache_ownership": "request_owner",
                    "cuda_graph_decode": False,
                    "cuda_graph_buckets": (),
                    "cuda_graph_captures": 0,
                    "cuda_graph_replays": 0,
                    "cuda_graph_eager_fallbacks": 0,
                    "attention_dp_prefill_scratch_pages": 2,
                    "attention_dp_decode_scratch_pages": 1,
                    "attention_dp_graph_scratch_pages": 0,
                    "attention_dp_scratch_pages": 3,
                "moe_collective_transport": transport,
            }

    app = create_app(engines={"qwen3-235b": KairyuBackend()})
    async with _client(app) as client:
        response = await client.get("/backends")

    assert response.status_code == 200
    entry = response.json()["engines"][0]
    assert entry["execution_mode"] == "request-owned-attention-dp"
    assert entry["pipeline_depth"] == 5
    assert entry["moe_collective_transport"] == transport
    assert "parallelism_metadata_status" not in entry


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


def test_deepseek_v4_native_ep_metadata_accepts_ep8_composite_cache() -> None:
    metadata = health_module._expert_parallel_metadata(
        {
            "parallelism": "expert_parallel",
            "expert_parallel_size": 8,
            "attention_placement": "request_owned_data_parallel",
            "attention_output_placement": "replicated",
            "attention_output_parallel_size": 1,
            "attention_output_partial_dtype": None,
            "execution_mode": "deepseek-v4-native-attention-dp",
            "pipeline_depth": 1,
            "decode_mode": "eager",
            "kv_cache_dtype": "checkpoint-native-hca-csa",
            "attention_data_parallel_size": 8,
            "attention_tensor_parallel_size": 1,
            "moe_dispatcher": "fixed-nccl-all-to-all-packed-fp4",
            "sampling_ownership": "request_owner",
            "kv_cache_ownership": "request_owner",
            "cuda_graph_decode": False,
            "cuda_graph_buckets": (),
            "cuda_graph_captures": 0,
            "cuda_graph_replays": 0,
            "cuda_graph_eager_fallbacks": 0,
            "attention_dp_prefill_scratch_pages": 2,
            "attention_dp_decode_scratch_pages": 1,
            "attention_dp_graph_scratch_pages": 0,
            "attention_dp_scratch_pages": 3,
            "moe_collective_transport": {
                "selected_backend": "torch.distributed:nccl",
                "fallback_backend": "torch.distributed:nccl",
                "direct_nccl_active": False,
                "direct_nccl_library": None,
                "direct_nccl_version": None,
                "selection_reason": "native fixed all-to-all",
            },
        }
    )
    assert metadata is not None
    assert metadata["expert_parallel_size"] == 8
    assert metadata["kv_cache_dtype"] == "checkpoint-native-hca-csa"


async def test_backends_reports_live_attention_dp_cuda_graph_counters():
    transport = {
        "selected_backend": "direct_nccl_ctypes",
        "fallback_backend": "torch.distributed:nccl",
        "direct_nccl_active": True,
        "direct_nccl_library": "libnccl.so.2",
        "direct_nccl_version": "2.29.7",
        "selection_reason": "initialized on every rank",
    }

    class KairyuBackend(MockBackend):
        expert_parallel_size = 4
        parallelism_metadata = None

        @staticmethod
        def parallelism_metadata_snapshot():
            return {
                "parallelism": "expert_parallel",
                "expert_parallel_size": 4,
                "attention_placement": "request_owned_data_parallel",
                "attention_output_placement": "replicated",
                "attention_output_parallel_size": 1,
                "attention_output_partial_dtype": None,
                "execution_mode": "request-owned-attention-dp",
                "pipeline_depth": 1,
                "decode_mode": "cuda_graph",
                "kv_cache_dtype": "bfloat16",
                "attention_data_parallel_size": 4,
                "attention_tensor_parallel_size": 1,
                "moe_dispatcher": "nvfp4_allgather_reduce_scatter",
                "sampling_ownership": "request_owner",
                "kv_cache_ownership": "request_owner",
                "cuda_graph_decode": True,
                "cuda_graph_buckets": (1, 2, 4, 8),
                "cuda_graph_captures": 2,
                "cuda_graph_replays": 17,
                "cuda_graph_eager_fallbacks": 1,
                "attention_dp_prefill_scratch_pages": 2,
                "attention_dp_decode_scratch_pages": 8,
                "attention_dp_graph_scratch_pages": 1,
                "attention_dp_scratch_pages": 11,
                "moe_collective_transport": transport,
            }

    app = create_app(engines={"qwen3-235b": KairyuBackend()})
    async with _client(app) as client:
        response = await client.get("/backends")

    entry = response.json()["engines"][0]
    assert entry["decode_mode"] == "cuda_graph"
    assert entry["cuda_graph_decode"] is True
    assert entry["cuda_graph_buckets"] == [1, 2, 4, 8]
    assert entry["cuda_graph_captures"] == 2
    assert entry["cuda_graph_replays"] == 17
    assert entry["cuda_graph_eager_fallbacks"] == 1


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
            self.generation_defaults = GenerationDefaults(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                mode="auto",
                source="generation_config.json",
            )

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
    expected_generation = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    # One remote replica is an audit sample, not proof that the whole pool is
    # homogeneous. Keep it under via_replica instead of promoting it.
    assert "generation_config" not in pool
    assert "generation_config_source" not in pool
    assert "generation_defaults" not in pool
    assert pool["via_replica"]["generation_config"] == "auto"
    assert (
        pool["via_replica"]["generation_config_source"]
        == "generation_config.json"
    )
    assert pool["via_replica"]["generation_defaults"] == expected_generation


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


async def test_backends_gateway_reports_unanimous_declared_replica_metadata():
    replica = OpenAICompatBackend(
        base_url="http://upstream/v1",
        model="qwen3.6-27b",
        api_key_env=None,
        transport=httpx.MockTransport(lambda request: httpx.Response(404)),
        upstream="vllm",
        model_revision="fixed-revision",
        max_model_len=262144,
        quantization_format="bfloat16",
        cache_descriptor={"family": "qwen3.6-hybrid-deltanet-paged-kv"},
        tensor_parallel_size=1,
        mtp_enabled=False,
        container_image_digest="vllm@sha256:fixed",
    )
    gateway_app = create_app(engines={"qwen3.6-27b": ReplicaPool([replica])})

    async with _client(gateway_app) as client:
        response = await client.get("/backends")

    assert response.status_code == 200
    entry = response.json()["engines"][0]
    assert entry["model_revision"] == "fixed-revision"
    assert entry["max_model_len"] == 262144
    assert entry["quantization_format"] == "bfloat16"
    assert entry["cache_descriptor"] == {
        "family": "qwen3.6-hybrid-deltanet-paged-kv"
    }
    assert entry["tensor_parallel_size"] == 1
    assert entry["mtp_enabled"] is False
    assert entry["container_image_digest"] == "vllm@sha256:fixed"
    assert "via_replica" not in entry


async def test_backends_does_not_adopt_mixed_or_malformed_replica_defaults():
    valid = {
        "generation_config": "auto",
        "generation_config_source": "generation_config.json",
        "generation_defaults": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        },
    }

    class ProbePool(ReplicaPool):
        async def probe_backends(self):
            return {
                "attention_backend": "torch",
                "engines": [
                    valid,
                    {
                        **valid,
                        "generation_defaults": "malformed",
                    },
                ],
            }

    app = create_app({"pool": ProbePool([MockBackend()])})
    async with _client(app) as client:
        response = await client.get("/backends")

    entry = response.json()["engines"][0]
    assert "generation_config" not in entry
    assert "generation_config_source" not in entry
    assert "generation_defaults" not in entry
    assert "generation_config" not in entry["via_replica"]
    assert "generation_defaults" not in entry["via_replica"]


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
