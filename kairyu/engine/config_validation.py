"""Side-effect-free validation of backend constructor configuration.

Built-in checks here are deliberately limited to contracts that can be decided
without importing a runtime backend, resolving credentials, loading a model, or
probing hardware.  ``create_backend`` and deployment preflight share this one
boundary so configuration rules cannot drift.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping
from os import PathLike

from kairyu.engine.core.kv_cache_dtype import validate_kv_cache_dtype
from kairyu.engine.core.nvfp4_accuracy import NvFp4AccuracyProfile
from kairyu.engine.openai_capabilities import resolve_openai_capabilities
from kairyu.engine.vision import ImageInputPolicy
from kairyu.models.generation import validate_generation_config_mode

_MOCK_OPTIONS = frozenset(
    {
        "responses",
        "latency_s",
        "tensor_parallel_size",
    }
)
_OPENAI_OPTIONS = frozenset(
    {
        "base_url",
        "model",
        "api_key_env",
        "timeout_s",
        "transport",
        "request_stream_usage",
        "upstream",
        "capabilities",
        "image_input_policy",
        "allow_templated_chat_passthrough",
        "completion_reasoning_end_tag",
        "chat_tokenizer",
        "model_revision",
        "max_model_len",
        "quantization_format",
        "cache_descriptor",
        "tensor_parallel_size",
        "expert_parallel_size",
        "attention_data_parallel_size",
        "mtp_enabled",
        "dspark_enabled",
        "container_image_digest",
        "client",
        "client_factory",
    }
)
_KAIRYU_OPTIONS = frozenset(
    {
        "num_pages",
        "page_size",
        "max_num_batched_tokens",
        "max_num_partial_prefills",
        "max_num_seqs",
        "max_model_len",
        "priority_age_s",
        "runner",
        "tensor_parallel_size",
        "expert_parallel_size",
        "expert_parallel_attention_dp",
        "tokenizer",
        "speculative",
        "speculative_tokens",
        "draft_model_path",
        "model_path",
        "pd_separation",
        "pd_prefill_device",
        "pd_decode_device",
        "pd_defer_handoff",
        "pipeline_depth",
        "decode_mode",
        "cuda_graph_max_batch",
        "cuda_graph_max_pages",
        "cuda_graph_warmup_iters",
        "kv_cache_dtype",
        "dram_kv_tier_capacity_pages",
        "dram_kv_tier_profile",
        "generation_config",
        "nvfp4_accuracy_profile",
        "execution_mode",
        "prefix_state_capacity_bytes",
        "model_revision",
        "quantization_format",
        "container_image_digest",
        "mtp_enabled",
        "dspark_enabled",
    }
)
_KAIRYU_PROC_OPTIONS = frozenset(
    {
        "num_pages",
        "page_size",
        "max_num_batched_tokens",
        "max_num_partial_prefills",
        "max_num_seqs",
        "max_model_len",
        "priority_age_s",
        "tensor_parallel_size",
        "tokenizer",
        "speculative",
        "speculative_tokens",
        "draft_model_path",
        "death_timeout_s",
        "model_path",
        "pipeline_depth",
        "decode_mode",
        "cuda_graph_max_batch",
        "cuda_graph_max_pages",
        "cuda_graph_warmup_iters",
        "kv_cache_dtype",
        "dram_kv_tier_capacity_pages",
        "dram_kv_tier_profile",
        "generation_config",
        "nvfp4_accuracy_profile",
    }
)
_DECODE_MODES = frozenset({"eager", "cuda_graph"})
_PD_CUDA_DEVICE = re.compile(r"cuda(?::[0-9]+)?\Z")


def _normalized_options(options: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(options, Mapping) or any(not isinstance(name, str) for name in options):
        raise ValueError("backend options must be a string-keyed mapping")
    return dict(options)


def _require_exact_options(
    backend: str,
    options: Mapping[str, object],
    allowed: frozenset[str],
) -> None:
    unsupported = sorted(set(options) - allowed)
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"{backend} backend has unsupported configuration options: {names}"
        )


def _require_string(
    backend: str,
    options: Mapping[str, object],
    field: str,
) -> None:
    value = options.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{backend} backend requires a string {field}")


def _require_int_at_least(
    backend: str,
    field: str,
    value: object,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{backend} backend {field} must be a {qualifier} integer")
    return value


def _validate_mock(options: Mapping[str, object]) -> None:
    _require_exact_options("mock", options, _MOCK_OPTIONS)
    responses = options.get("responses")
    if responses is not None and (
        not isinstance(responses, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in responses.items()
        )
    ):
        raise ValueError("mock backend responses must map strings to strings")
    latency = options.get("latency_s", 0.0)
    if not isinstance(latency, (int, float)):
        raise ValueError("mock backend latency_s must be numeric")
    _require_int_at_least(
        "mock",
        "tensor_parallel_size",
        options.get("tensor_parallel_size", 1),
        1,
    )


def _validate_openai(options: Mapping[str, object]) -> None:
    _require_exact_options("openai", options, _OPENAI_OPTIONS)
    _require_string("openai", options, "base_url")
    _require_string("openai", options, "model")
    client = options.get("client")
    client_factory = options.get("client_factory")
    transport = options.get("transport")
    if sum(value is not None for value in (client, client_factory, transport)) > 1:
        raise ValueError(
            "openai backend client, client_factory, and transport are mutually exclusive"
        )
    request_stream_usage = options.get("request_stream_usage", True)
    if type(request_stream_usage) is not bool:
        raise ValueError("openai backend request_stream_usage must be a boolean")
    try:
        capabilities = resolve_openai_capabilities(
            options.get("upstream", "generic"),
            options.get("capabilities"),
        )
    except (TypeError, ValueError):
        raise ValueError("openai backend capability configuration is invalid") from None
    unsupported_prompt_kinds = capabilities.prompt_kinds - {
        "text",
        "multimodal",
    }
    if unsupported_prompt_kinds:
        raise ValueError(
            "openai backend currently supports only text and multimodal "
            "prompt kinds"
        )
    image_input_policy = options.get("image_input_policy")
    if image_input_policy is not None:
        if not isinstance(image_input_policy, ImageInputPolicy):
            if not isinstance(image_input_policy, Mapping):
                raise ValueError(
                    "openai backend image_input_policy must be a mapping or "
                    "ImageInputPolicy"
                )
            try:
                ImageInputPolicy(**dict(image_input_policy))
            except (TypeError, ValueError):
                raise ValueError(
                    "openai backend image_input_policy configuration is invalid"
                ) from None
    supports_multimodal = "multimodal" in capabilities.prompt_kinds
    if supports_multimodal != (image_input_policy is not None):
        raise ValueError(
            "openai backend multimodal capability and image_input_policy "
            "must be configured together"
        )
    if supports_multimodal and not request_stream_usage:
        raise ValueError(
            "openai backend multimodal streaming requires "
            "request_stream_usage=true"
        )
    allow_passthrough = options.get("allow_templated_chat_passthrough", False)
    if type(allow_passthrough) is not bool:
        raise ValueError(
            "openai backend allow_templated_chat_passthrough must be a boolean"
        )
    reasoning_end_tag = options.get("completion_reasoning_end_tag")
    if reasoning_end_tag is not None and (
        not isinstance(reasoning_end_tag, str)
        or not reasoning_end_tag
        or len(reasoning_end_tag) > 128
    ):
        raise ValueError(
            "openai backend completion_reasoning_end_tag must be a non-empty "
            "string of at most 128 characters or null"
        )
    if reasoning_end_tag is not None and (
        capabilities.upstream != "vllm" or allow_passthrough is not True
    ):
        raise ValueError(
            "openai backend completion_reasoning_end_tag requires upstream='vllm' "
            "and allow_templated_chat_passthrough=true"
        )
    for field in (
        "chat_tokenizer",
        "model_revision",
        "quantization_format",
        "container_image_digest",
    ):
        value = options.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"openai backend {field} must be a string or null")
    max_model_len = options.get("max_model_len")
    if max_model_len is not None:
        _require_int_at_least("openai", "max_model_len", max_model_len, 1)
    for field in (
        "tensor_parallel_size",
        "expert_parallel_size",
        "attention_data_parallel_size",
    ):
        _require_int_at_least("openai", field, options.get(field, 1), 1)
    cache_descriptor = options.get("cache_descriptor")
    if cache_descriptor is not None and not isinstance(cache_descriptor, Mapping):
        raise ValueError("openai backend cache_descriptor must be a mapping or null")
    for field in ("mtp_enabled", "dspark_enabled"):
        if type(options.get(field, False)) is not bool:
            raise ValueError(f"openai backend {field} must be a boolean")


def _validate_vllm(options: Mapping[str, object]) -> None:
    # VLLMBackend intentionally forwards unknown keys to AsyncEngineArgs.
    _require_string("vllm", options, "model")
    policy = options.get("scheduling_policy", "priority")
    if policy != "priority":
        raise ValueError("vllm backend scheduling_policy must be priority")


def _validate_priority_age(backend: str, value: object) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{backend} backend priority_age_s must be finite and non-negative or null"
        )


def _validate_native_common(
    backend: str,
    options: Mapping[str, object],
    *,
    tensor_parallel_size: int,
    runner: object | None,
    pd_separation: bool,
    graph_scratch_outside_scheduler: bool = False,
) -> None:
    validate_generation_config_mode(options.get("generation_config", "auto"))
    num_pages = _require_int_at_least(
        backend,
        "num_pages",
        options.get("num_pages", 4096),
        1,
    )
    _require_int_at_least(
        backend,
        "page_size",
        options.get("page_size", 16),
        1,
    )
    _require_int_at_least(
        backend,
        "max_num_batched_tokens",
        options.get("max_num_batched_tokens", 2048),
        1,
    )
    _require_int_at_least(
        backend,
        "max_num_partial_prefills",
        options.get("max_num_partial_prefills", 2),
        1,
    )
    _require_int_at_least(
        backend,
        "max_num_seqs",
        options.get("max_num_seqs", 256),
        1,
    )
    max_model_len = options.get("max_model_len")
    if max_model_len is not None:
        _require_int_at_least(
            backend,
            "max_model_len",
            max_model_len,
            1,
        )
    _require_int_at_least(
        backend,
        "pipeline_depth",
        options.get("pipeline_depth", 1),
        1,
    )
    _validate_priority_age(backend, options.get("priority_age_s", 60.0))

    speculative = options.get("speculative")
    if speculative not in (None, "ngram", "eagle", "mtp"):
        raise ValueError("native backend speculative mode is unsupported")
    if speculative is not None:
        _require_int_at_least(
            backend,
            "speculative_tokens",
            options.get("speculative_tokens", 4),
            0,
        )
    decode_mode = options.get("decode_mode")
    if decode_mode is not None and decode_mode not in _DECODE_MODES:
        raise ValueError("native backend decode_mode is unsupported")
    model_path = options.get("model_path")
    if model_path is not None and not isinstance(model_path, (str, PathLike)):
        raise ValueError("native backend model_path must be a local path or null")
    if model_path is not None and runner is not None:
        raise ValueError("native backend model_path and runner are mutually exclusive")
    accuracy_profile = NvFp4AccuracyProfile.parse(
        options.get("nvfp4_accuracy_profile")
    )
    if accuracy_profile.active and model_path is None:
        raise ValueError(
            "native backend NVFP4 accuracy profile requires a real model_path"
        )
    if accuracy_profile.active and pd_separation:
        raise ValueError(
            "native backend NVFP4 accuracy profile does not support P-D separation"
        )
    if accuracy_profile.active and bool(
        options.get("expert_parallel_attention_dp", False)
    ):
        raise ValueError(
            "native backend NVFP4 accuracy profile requires replicated-attention EP"
        )
    draft_model_path = options.get("draft_model_path")
    if draft_model_path is not None and (
        not isinstance(draft_model_path, (str, PathLike))
        or not str(draft_model_path)
    ):
        raise ValueError(
            "native backend draft_model_path must be a non-empty local path or null"
        )
    if speculative in ("eagle", "mtp"):
        if model_path is None or draft_model_path is None:
            raise ValueError(
                "native learned speculation requires model_path and draft_model_path"
            )
        if decode_mode == "cuda_graph":
            raise ValueError(
                "native learned speculation requires decode_mode='eager'"
            )
    elif draft_model_path is not None:
        raise ValueError(
            "native draft_model_path requires speculative='eagle' or 'mtp'"
        )
    kv_cache_dtype = validate_kv_cache_dtype(options.get("kv_cache_dtype", "auto"))
    if kv_cache_dtype != "auto" and model_path is None:
        raise ValueError(
            "native backend explicit KV cache dtype requires a real model_path"
        )
    if kv_cache_dtype != "auto" and pd_separation:
        raise ValueError(
            "native backend explicit KV cache dtype does not support P-D separation"
        )
    dram_pages = _require_int_at_least(
        backend,
        "dram_kv_tier_capacity_pages",
        options.get("dram_kv_tier_capacity_pages", 0),
        0,
    )
    dram_profile = options.get("dram_kv_tier_profile")
    if dram_profile is not None and (
        not isinstance(dram_profile, (str, PathLike))
        or not str(dram_profile)
    ):
        raise ValueError(
            "native backend dram_kv_tier_profile must be a non-empty "
            "local path or null"
        )
    if (dram_pages > 0) != (dram_profile is not None):
        raise ValueError(
            "native backend DRAM KV tier requires both a positive "
            "dram_kv_tier_capacity_pages and dram_kv_tier_profile"
        )
    if dram_pages:
        if model_path is None:
            raise ValueError(
                "native backend DRAM KV tier requires a real model_path"
            )
        if pd_separation:
            raise ValueError(
                "native backend DRAM KV tier does not support P-D separation"
            )
    graph_decode = decode_mode == "cuda_graph"
    graph_batch = options.get("cuda_graph_max_batch")
    if graph_batch is not None:
        _require_int_at_least(
            backend,
            "cuda_graph_max_batch",
            graph_batch,
            1,
        )
    graph_pages = options.get("cuda_graph_max_pages")
    if graph_pages is not None:
        graph_pages = _require_int_at_least(
            backend,
            "cuda_graph_max_pages",
            graph_pages,
            1,
        )
        graph_page_capacity = num_pages - int(
            not graph_scratch_outside_scheduler
        )
        if graph_pages > graph_page_capacity:
            if graph_scratch_outside_scheduler:
                raise ValueError(
                    "native backend cuda_graph_max_pages exceeds the "
                    "scheduler-visible graph page capacity"
                )
            raise ValueError(
                "native backend cuda_graph_max_pages must be smaller than num_pages"
            )
    graph_warmups = options.get("cuda_graph_warmup_iters")
    if graph_warmups is not None:
        _require_int_at_least(
            backend,
            "cuda_graph_warmup_iters",
            graph_warmups,
            0,
        )
    if graph_decode:
        if model_path is None:
            raise ValueError("native backend cuda_graph decode requires model_path")
        if num_pages < 2 and not graph_scratch_outside_scheduler:
            raise ValueError("native backend cuda_graph decode requires at least two pages")

    if pd_separation:
        if model_path is None:
            raise ValueError("native backend pd_separation requires model_path")
        if tensor_parallel_size > 1:
            raise ValueError("native backend pd_separation does not support tensor parallelism")
        if speculative is not None:
            raise ValueError("native backend pd_separation does not support speculative decoding")
        if graph_decode:
            raise ValueError("native backend pd_separation does not support cuda_graph decode")
    if speculative is not None and tensor_parallel_size > 1 and model_path is None:
        raise ValueError("native backend speculative tensor parallelism requires model_path")


def _validate_kairyu(options: Mapping[str, object]) -> None:
    _require_exact_options("kairyu", options, _KAIRYU_OPTIONS)
    execution_mode = options.get("execution_mode", "auto")
    if execution_mode not in {"auto", "native", "reference"}:
        raise ValueError(
            "kairyu backend execution_mode must be one of auto, native, or reference"
        )
    _require_int_at_least(
        "kairyu",
        "prefix_state_capacity_bytes",
        options.get("prefix_state_capacity_bytes", 0),
        0,
    )
    for field in (
        "model_revision",
        "quantization_format",
        "container_image_digest",
    ):
        value = options.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"kairyu backend {field} must be a string or null")
    for field in ("mtp_enabled", "dspark_enabled"):
        if type(options.get(field, False)) is not bool:
            raise ValueError(f"kairyu backend {field} must be a boolean")
    tensor_parallel_size = _require_int_at_least(
        "kairyu",
        "tensor_parallel_size",
        options.get("tensor_parallel_size", 1),
        1,
    )
    expert_parallel_size = _require_int_at_least(
        "kairyu",
        "expert_parallel_size",
        options.get("expert_parallel_size", 1),
        1,
    )
    if expert_parallel_size not in {1, 2, 4}:
        raise ValueError(
            "kairyu backend expert_parallel_size must be one of 1, 2, or 4"
        )
    expert_parallel = expert_parallel_size > 1
    expert_parallel_attention_dp = options.get(
        "expert_parallel_attention_dp",
        False,
    )
    if type(expert_parallel_attention_dp) is not bool:
        raise ValueError(
            "kairyu backend expert_parallel_attention_dp must be a boolean"
        )
    if expert_parallel_attention_dp and expert_parallel_size != 4:
        raise ValueError(
            "kairyu backend expert_parallel_attention_dp requires "
            "expert_parallel_size=4"
        )
    if expert_parallel and tensor_parallel_size > 1:
        raise ValueError(
            "kairyu backend tensor parallelism and expert parallelism are "
            "mutually exclusive"
        )
    pd_separation = options.get("pd_separation", False)
    if not isinstance(pd_separation, bool):
        raise ValueError("kairyu backend pd_separation must be a boolean")
    pd_device_types: dict[str, str] = {}
    for field in ("pd_prefill_device", "pd_decode_device"):
        value = options.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"kairyu backend {field} must be a string or null")
        if isinstance(value, str):
            if value == "cpu":
                pd_device_types[field] = "cpu"
            elif _PD_CUDA_DEVICE.fullmatch(value):
                pd_device_types[field] = "cuda"
            else:
                raise ValueError(
                    f"kairyu backend {field} must name a CPU or CUDA device, got {value!r}"
                )
    if len(pd_device_types) == 2 and len(set(pd_device_types.values())) != 1:
        raise ValueError(
            "kairyu backend pd_prefill_device and pd_decode_device must both "
            "be CUDA or both be CPU"
        )
    pd_defer_handoff = options.get("pd_defer_handoff", True)
    if not isinstance(pd_defer_handoff, bool):
        raise ValueError("kairyu backend pd_defer_handoff must be a boolean")
    if not pd_separation and (
        options.get("pd_prefill_device") is not None
        or options.get("pd_decode_device") is not None
        or pd_defer_handoff is not True
    ):
        raise ValueError("kairyu backend P-D role options require pd_separation")
    _validate_native_common(
        "kairyu",
        options,
        tensor_parallel_size=tensor_parallel_size,
        runner=options.get("runner"),
        pd_separation=pd_separation,
        graph_scratch_outside_scheduler=expert_parallel_attention_dp,
    )
    if expert_parallel:
        if options.get("model_path") is None:
            raise ValueError(
                "kairyu backend expert parallelism requires a real model_path"
            )
        if (
            not expert_parallel_attention_dp
            and options.get("pipeline_depth", 1) != 1
        ):
            raise ValueError(
                "kairyu backend replicated-attention expert parallelism "
                "requires pipeline_depth=1"
            )
        if (
            options.get("decode_mode") == "cuda_graph"
            and not expert_parallel_attention_dp
        ):
            raise ValueError(
                "kairyu backend replicated-attention expert parallelism "
                "requires decode_mode='eager'"
            )
        if options.get("kv_cache_dtype", "auto") != "bfloat16":
            raise ValueError(
                "kairyu backend expert parallelism requires "
                "kv_cache_dtype='bfloat16'"
            )
        if pd_separation:
            raise ValueError(
                "kairyu backend expert parallelism does not support P-D separation"
            )
        if options.get("speculative") is not None:
            raise ValueError(
                "kairyu backend expert parallelism does not support speculative decoding"
            )
        if options.get("dram_kv_tier_capacity_pages", 0) != 0:
            raise ValueError(
                "kairyu backend expert parallelism does not support a DRAM KV tier"
            )
    if options.get("model_path") is None:
        try:
            from kairyu.engine.core.tp_runner import validate_tp_degree

            validate_tp_degree(tensor_parallel_size)
        except ValueError:
            raise ValueError(
                "kairyu backend tensor_parallel_size is incompatible with KV heads"
            ) from None


def _validate_kairyu_proc(options: Mapping[str, object]) -> None:
    _require_exact_options("kairyu-proc", options, _KAIRYU_PROC_OPTIONS)
    tokenizer = options.get("tokenizer")
    if tokenizer is not None and not isinstance(tokenizer, str):
        raise ValueError("kairyu-proc backend tokenizer must be a string or null")
    tensor_parallel_size = _require_int_at_least(
        "kairyu-proc",
        "tensor_parallel_size",
        options.get("tensor_parallel_size", 1),
        1,
    )
    _validate_native_common(
        "kairyu-proc",
        options,
        tensor_parallel_size=tensor_parallel_size,
        runner=None,
        pd_separation=False,
    )
    if options.get("model_path") is None:
        try:
            from kairyu.engine.core.tp_runner import validate_tp_degree

            validate_tp_degree(tensor_parallel_size)
        except ValueError:
            raise ValueError(
                "kairyu-proc backend tensor_parallel_size is incompatible with KV heads"
            ) from None


def _custom_factory(name: str) -> Callable[..., object] | None:
    # Deferred import avoids a module cycle: registry imports this module, while
    # the lookup runs only after registry initialization has completed.
    from kairyu.engine.registry import _factory_for_config_validation

    return _factory_for_config_validation(name)


def _uses_builtin_contract(name: str) -> bool:
    from kairyu.engine.registry import uses_builtin_backend_contract

    return uses_builtin_backend_contract(name)


def _validate_custom(name: str, options: Mapping[str, object]) -> None:
    factory = _custom_factory(name)
    if factory is None:
        raise ValueError("backend is not registered")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(**options)
    except TypeError:
        raise ValueError("custom backend options do not match the registered factory") from None


def validate_backend_options(
    name: str,
    options: Mapping[str, object],
) -> None:
    """Validate one backend configuration without importing or constructing it."""

    normalized = _normalized_options(options)
    validators: dict[str, Callable[[Mapping[str, object]], None]] = {
        "mock": _validate_mock,
        "openai": _validate_openai,
        "vllm": _validate_vllm,
        "kairyu": _validate_kairyu,
        "kairyu-proc": _validate_kairyu_proc,
    }
    validator = validators.get(name)
    if validator is not None and _uses_builtin_contract(name):
        validator(normalized)
        return
    _validate_custom(name, normalized)
