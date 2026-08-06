from pathlib import Path

import pytest

from kairyu.engine import registry as registry_module
from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.mock import MockBackend
from kairyu.engine.registry import available_backends, create_backend, register_backend
from kairyu.engine.vision import ImageInputPolicy


def test_mock_backend_is_preregistered():
    backend = create_backend("mock")
    assert isinstance(backend, MockBackend)


def test_kwargs_forwarded_to_factory():
    backend = create_backend("mock", responses={"x": "y"})
    assert backend._responses == {"x": "y"}


def test_unknown_backend_lists_known_names():
    with pytest.raises(ValueError, match="mock"):
        create_backend("does-not-exist")


def test_lazy_backends_resolve_without_prior_import():
    backend = create_backend("openai", base_url="https://api.example.com/v1", model="m")
    assert type(backend).__name__ == "OpenAICompatBackend"


def test_register_custom_backend():
    register_backend("custom-test", lambda **kwargs: MockBackend(**kwargs))
    assert "custom-test" in available_backends()
    assert isinstance(create_backend("custom-test"), MockBackend)


@pytest.mark.parametrize(
    "options",
    [
        {"tensor_parallel_size": 0},
        {"responses": 1},
    ],
)
def test_mock_rejects_invalid_constructor_options(options):
    with pytest.raises(ValueError):
        validate_backend_options("mock", options)


def test_native_cuda_graph_without_model_is_rejected_before_import():
    with pytest.raises(ValueError, match="cuda_graph"):
        validate_backend_options("kairyu", {"decode_mode": "cuda_graph"})


@pytest.mark.parametrize("backend", ["kairyu", "kairyu-proc"])
def test_native_auto_graph_defaults_pass_static_preflight(backend):
    validate_backend_options(backend, {"decode_mode": None})
    validate_backend_options(backend, {})


@pytest.mark.parametrize("backend", ["kairyu", "kairyu-proc"])
@pytest.mark.parametrize(
    "options",
    [
        {"cuda_graph_max_batch": 0},
        {"cuda_graph_max_pages": 0},
        {"num_pages": 8, "cuda_graph_max_pages": 8},
        {"cuda_graph_warmup_iters": -1},
    ],
)
def test_native_explicit_graph_limits_validate_in_every_decode_mode(
    backend,
    options,
):
    with pytest.raises(ValueError, match="cuda_graph"):
        validate_backend_options(backend, options)


@pytest.mark.parametrize("backend", ["kairyu", "kairyu-proc"])
@pytest.mark.parametrize("mode", ["auto", "vllm", "none"])
def test_native_generation_config_modes_pass_static_preflight(backend, mode):
    validate_backend_options(backend, {"generation_config": mode})


@pytest.mark.parametrize("backend", ["kairyu", "kairyu-proc"])
@pytest.mark.parametrize("mode", ["model", "", None, True])
def test_invalid_native_generation_config_mode_fails_static_preflight(
    backend,
    mode,
):
    with pytest.raises(ValueError, match="generation_config"):
        validate_backend_options(backend, {"generation_config": mode})


@pytest.mark.parametrize("expert_parallel_size", [2, 4])
def test_expert_parallel_options_pass_preflight_without_runtime_import(
    expert_parallel_size,
):
    validate_backend_options(
        "kairyu",
        {
            "model_path": "/models/qwen3-235b",
            "expert_parallel_size": expert_parallel_size,
            "kv_cache_dtype": "bfloat16",
        },
    )


@pytest.mark.parametrize("expert_parallel_size", [0, 3, 8, True])
def test_invalid_expert_parallel_size_fails_static_preflight(
    expert_parallel_size,
):
    with pytest.raises(ValueError, match="expert_parallel_size"):
        validate_backend_options(
            "kairyu",
            {"expert_parallel_size": expert_parallel_size},
        )


def test_expert_parallelism_is_mutually_exclusive_with_tensor_parallelism():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_backend_options(
            "kairyu",
            {
                "model_path": "/models/qwen3-235b",
                "tensor_parallel_size": 2,
                "expert_parallel_size": 2,
                "kv_cache_dtype": "bfloat16",
            },
        )


@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4, 8])
def test_kairyu_proc_accepts_tensor_parallel_degrees(tensor_parallel_size):
    validate_backend_options(
        "kairyu-proc",
        {
            "model_path": "/models/qwen3-32b",
            "tensor_parallel_size": tensor_parallel_size,
        },
    )


@pytest.mark.parametrize("tensor_parallel_size", [0, -1, True, 1.5, "4"])
def test_kairyu_proc_rejects_invalid_tensor_parallel_degrees(
    tensor_parallel_size,
):
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        validate_backend_options(
            "kairyu-proc",
            {"tensor_parallel_size": tensor_parallel_size},
        )


def test_kairyu_proc_model_less_tp_must_shard_the_toy_kv_heads():
    with pytest.raises(ValueError, match="incompatible with KV heads"):
        validate_backend_options(
            "kairyu-proc",
            {"tensor_parallel_size": 3},
        )


@pytest.mark.parametrize(
    "options",
    [
        {"expert_parallel_size": 2},
        {
            "model_path": "/models/qwen3-235b",
            "expert_parallel_size": 2,
        },
        {
            "model_path": "/models/qwen3-235b",
            "expert_parallel_size": 2,
            "kv_cache_dtype": "bfloat16",
            "pipeline_depth": 2,
        },
        {
            "model_path": "/models/qwen3-235b",
            "expert_parallel_size": 2,
            "kv_cache_dtype": "bfloat16",
            "speculative": "ngram",
        },
        {
            "model_path": "/models/qwen3-235b",
            "expert_parallel_size": 2,
            "kv_cache_dtype": "bfloat16",
            "decode_mode": "cuda_graph",
        },
    ],
)
def test_unsupported_expert_parallel_modes_fail_static_preflight(options):
    with pytest.raises(ValueError, match="expert parallelism"):
        validate_backend_options("kairyu", options)


def test_attention_dp_cuda_graph_passes_static_preflight() -> None:
    validate_backend_options(
        "kairyu",
        {
            "model_path": "/models/qwen3-235b",
            "expert_parallel_size": 4,
            "expert_parallel_attention_dp": True,
            "kv_cache_dtype": "bfloat16",
            "pipeline_depth": 5,
            "decode_mode": "cuda_graph",
            "cuda_graph_max_batch": 8,
            "cuda_graph_max_pages": 512,
            "cuda_graph_warmup_iters": 2,
        },
    )


def test_attention_dp_graph_pages_may_cover_full_scheduler_namespace() -> None:
    validate_backend_options(
        "kairyu",
        {
            "model_path": "/models/qwen3-235b",
            "expert_parallel_size": 4,
            "expert_parallel_attention_dp": True,
            "kv_cache_dtype": "bfloat16",
            "decode_mode": "cuda_graph",
            "num_pages": 8,
            "cuda_graph_max_pages": 8,
        },
    )


def test_cross_device_pd_options_pass_preflight_without_runtime_import():
    validate_backend_options(
        "kairyu",
        {
            "model_path": "/models/qwen3-32b",
            "pd_separation": True,
            "pd_prefill_device": "cuda:0",
            "pd_decode_device": "cuda:1",
            "pd_defer_handoff": False,
        },
    )


@pytest.mark.parametrize("backend", ["kairyu", "kairyu-proc"])
@pytest.mark.parametrize("max_model_len", [0, -1, True, 8192.0, "8192"])
def test_native_max_model_len_rejects_non_positive_or_non_integer_values(
    backend,
    max_model_len,
):
    with pytest.raises(ValueError, match="max_model_len.*positive integer"):
        validate_backend_options(
            backend,
            {"max_model_len": max_model_len},
        )


@pytest.mark.parametrize("backend", ["kairyu", "kairyu-proc"])
@pytest.mark.parametrize("max_model_len", [None, 8192])
def test_native_max_model_len_accepts_none_or_positive_integer(
    backend,
    max_model_len,
):
    validate_backend_options(
        backend,
        {"max_model_len": max_model_len},
    )


@pytest.mark.parametrize(
    ("prefill", "decode"),
    [
        ("cpu", "cuda:0"),
        ("mps", "mps"),
        ("not-a-device", "not-a-device"),
    ],
)
def test_pd_device_pair_fails_static_preflight(prefill, decode):
    with pytest.raises(ValueError, match="CPU or CUDA|both be CUDA"):
        validate_backend_options(
            "kairyu",
            {
                "model_path": "/models/qwen3-32b",
                "pd_separation": True,
                "pd_prefill_device": prefill,
                "pd_decode_device": decode,
            },
        )


@pytest.mark.parametrize(
    "options",
    [
        {"pd_prefill_device": 0},
        {"pd_decode_device": False},
        {"pd_defer_handoff": "false"},
        {"pd_prefill_device": "cuda:0"},
        {"pd_defer_handoff": False},
    ],
)
def test_invalid_or_inactive_pd_role_options_fail_preflight(options):
    with pytest.raises(ValueError, match="P-D|pd_"):
        validate_backend_options("kairyu", options)


def test_vllm_non_priority_policy_is_rejected_before_import():
    with pytest.raises(ValueError, match="scheduling_policy"):
        validate_backend_options(
            "vllm",
            {"model": "model", "scheduling_policy": "fcfs"},
        )


def test_openai_multimodal_capability_and_image_policy_must_be_paired():
    base = {
        "base_url": "http://vlm:8000/v1",
        "model": "vlm",
        "upstream": "vllm",
    }
    with pytest.raises(ValueError, match="must be configured together"):
        validate_backend_options(
            "openai",
            {
                **base,
                "capabilities": {"allow_prompt_kinds": ["multimodal"]},
            },
        )
    with pytest.raises(ValueError, match="must be configured together"):
        validate_backend_options(
            "openai",
            {
                **base,
                "image_input_policy": {"max_images": 1},
            },
        )

    validate_backend_options(
        "openai",
        {
            **base,
            "capabilities": {"allow_prompt_kinds": ["multimodal"]},
            "image_input_policy": {
                "max_images": 1,
                "max_processed_prompt_tokens": 8192,
            },
        },
    )
    validate_backend_options(
        "openai",
        {
            **base,
            "capabilities": {"allow_prompt_kinds": ["multimodal"]},
            "image_input_policy": ImageInputPolicy(max_images=1),
        },
    )
    with pytest.raises(ValueError, match="request_stream_usage=true"):
        validate_backend_options(
            "openai",
            {
                **base,
                "capabilities": {"allow_prompt_kinds": ["multimodal"]},
                "image_input_policy": {"max_images": 1},
                "request_stream_usage": False,
            },
        )
    with pytest.raises(ValueError, match="text and multimodal"):
        validate_backend_options(
            "openai",
            {
                **base,
                "capabilities": {"allow_prompt_kinds": ["tokens"]},
            },
        )


@pytest.mark.parametrize(
    "policy",
    [
        "not-a-mapping",
        {"max_images": 0},
        {"unknown_limit": 1},
        {
            "max_image_bytes": 1024,
            "max_total_image_bytes": 512,
        },
    ],
)
def test_openai_image_policy_is_strictly_validated_without_backend_start(policy):
    with pytest.raises(ValueError, match="image_input_policy"):
        validate_backend_options(
            "openai",
            {
                "base_url": "http://vlm:8000/v1",
                "model": "vlm",
                "upstream": "vllm",
                "capabilities": {"allow_prompt_kinds": ["multimodal"]},
                "image_input_policy": policy,
            },
        )


def test_custom_backend_signature_is_checked_without_construction():
    constructed = False

    def custom_factory(*, required):
        nonlocal constructed
        constructed = True
        return MockBackend(responses={"required": str(required)})

    register_backend("signature-test", custom_factory)

    with pytest.raises(ValueError, match="registered factory"):
        validate_backend_options("signature-test", {})

    assert not constructed


def test_builtin_name_override_uses_registered_factory_contract(monkeypatch):
    def replacement_factory(*, replacement):
        return MockBackend(responses={"replacement": str(replacement)})

    monkeypatch.setitem(
        registry_module._FACTORIES,
        "mock",
        replacement_factory,
    )

    validate_backend_options("mock", {"replacement": "accepted"})


@pytest.mark.parametrize(
    ("backend", "options"),
    [
        ("openai", {"base_url": "", "model": ""}),
        ("kairyu", {"speculative_tokens": -1}),
        ("kairyu", {"model_path": Path("")}),
        ("kairyu-proc", {"death_timeout_s": -1}),
    ],
)
def test_preflight_does_not_invent_constructor_constraints(backend, options):
    validate_backend_options(backend, options)
