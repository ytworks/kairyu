"""Checkpoint dir → (DenseDecoder, ModelConfig, generation metadata) (m12 D5).

Uses m8's ``CheckpointReader`` (index.json / sharded / single-file safetensors).
Tied embeddings are mandatory to handle — ``save_pretrained`` genuinely omits
``lm_head.weight`` from the file. Quantized checkpoints fail fast ("arrives in
M14"); the ``linear_factory`` hook is where M14's ``QuantizedLinear`` slots in
without touching this body.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from kairyu.engine.core.quant_config import (
    QuantMethod,
    load_checkpoint_quantization,
    validate_model_quantization,
)
from kairyu.engine.core.weights import CheckpointReader
from kairyu.models.config import ModelConfig, parse_model_config
from kairyu.models.generation import (
    GenerationConfigMode,
    GenerationDefaults,
    parse_generation_defaults,
)
from kairyu.models.llama import DenseDecoder

_SUPPORTED_BUILDERS = (
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "DeepseekV3ForCausalLM",
)


def load_generation_defaults(
    model_dir: str,
    generation_config: GenerationConfigMode = "auto",
) -> GenerationDefaults:
    """Public: eos/stop-token defaults from a checkpoint dir (used by the TP
    serve path, which shards the model but still needs the stop config)."""
    directory = Path(model_dir)
    config = json.loads((directory / "config.json").read_text())
    return parse_generation_defaults(directory, config, generation_config)


def build_model(
    config: ModelConfig,
    attention_backend=None,
    linear_factory=None,
    *,
    dtype: torch.dtype | None = None,
) -> DenseDecoder:
    """Registry: architecture -> module (one builder covers the dense family)."""
    if config.architecture not in _SUPPORTED_BUILDERS:
        raise ValueError(f"no builder for architecture {config.architecture!r}")
    return DenseDecoder(
        config,
        attention_backend=attention_backend,
        linear_factory=linear_factory,
        dtype=dtype,
    )


def declared_quantization_config(raw_config: dict) -> dict | None:
    """Return the effective outer or nested checkpoint quantization config."""
    text_config = raw_config.get("text_config")
    nested_quant = (
        text_config.get("quantization_config")
        if isinstance(text_config, dict)
        else None
    )
    declared_quant = raw_config.get("quantization_config") or nested_quant
    return declared_quant if isinstance(declared_quant, dict) else None


def reference_quant_loader_kind(
    raw_config: dict,
    architecture: str,
) -> str | None:
    """Return the official loader for one published reference checkpoint."""

    declared_quant = declared_quantization_config(raw_config)
    if declared_quant is None:
        return None
    method = str(declared_quant.get("quant_method", "")).lower()
    if (
        architecture == "DeepseekV4ForCausalLM"
        and method == "fp8"
        and declared_quant.get("weight_block_size") is not None
    ):
        return "deepseek-v4-block-fp8"
    if (
        architecture in ("KimiLinearForCausalLM", "KimiK3ForConditionalGeneration")
        and method == "compressed-tensors"
        and str(declared_quant.get("format", "")).lower() == "mxfp4-pack-quantized"
    ):
        return "kimi-k3-mxfp4"
    return None


def load_model(
    path: str | Path,
    dtype: torch.dtype = torch.float32,
    attention_backend=None,
    *,
    target_device: str | torch.device = "cpu",
    linear_capabilities=None,
    linear_selection_policy=None,
    nvfp4_accuracy_profile=None,
    generation_config: GenerationConfigMode = "auto",
) -> tuple[torch.nn.Module, ModelConfig, GenerationDefaults]:
    from kairyu.quant.linear import linear_factory

    directory = Path(path)
    config_file = directory / "config.json"
    if not config_file.is_file():
        raise ValueError(f"no config.json at {path}")
    raw_config = json.loads(config_file.read_text())
    generation = parse_generation_defaults(
        directory,
        raw_config,
        generation_config,
    )
    config = parse_model_config(raw_config)
    declared_quant = declared_quantization_config(raw_config)
    if config.requires_full_recompute:
        if linear_selection_policy is not None:
            raise ValueError(
                "hybrid reference execution does not support linear_selection_policy"
            )
        if nvfp4_accuracy_profile is not None:
            raise ValueError(
                "hybrid reference execution does not support nvfp4_accuracy_profile"
            )
    reference_quant_loader = reference_quant_loader_kind(
        raw_config, config.architecture
    )
    if reference_quant_loader == "deepseek-v4-block-fp8":
        from kairyu.models.reference import load_deepseek_public_decoder

        model = load_deepseek_public_decoder(
            directory,
            raw_config,
            config,
            dtype=dtype,
        )
        return model, config, generation
    if reference_quant_loader == "kimi-k3-mxfp4":
        from kairyu.models.reference import load_kimi_public_decoder

        model = load_kimi_public_decoder(
            directory,
            config,
            dtype=dtype,
        )
        return model, config, generation
    quant_config = (
        {**raw_config, "quantization_config": declared_quant}
        if declared_quant is not None
        else raw_config
    )
    quant = load_checkpoint_quantization(directory, quant_config).weights
    if config.requires_full_recompute:
        if quant.method is not QuantMethod.NONE:
            raise ValueError(
                f"{config.architecture} reference execution currently requires "
                "an unquantized checkpoint"
            )
        from kairyu.models.reference import load_reference_decoder

        model = load_reference_decoder(
            directory,
            raw_config,
            config,
            dtype=dtype,
        )
        return model, config, generation
    validate_model_quantization(
        quant,
        is_mla=config.is_mla,
        architecture=config.architecture,
    )
    if linear_selection_policy is not None and nvfp4_accuracy_profile is not None:
        raise ValueError(
            "linear_selection_policy and nvfp4_accuracy_profile are mutually exclusive"
        )
    if nvfp4_accuracy_profile is not None:
        from kairyu.engine.core.nvfp4_accuracy import resolve_nvfp4_accuracy_policy

        _profile, linear_selection_policy = resolve_nvfp4_accuracy_policy(
            quant,
            nvfp4_accuracy_profile,
            num_layers=config.num_hidden_layers,
        )
    model = build_model(
        config,
        attention_backend=attention_backend,
        linear_factory=linear_factory(
            quant,
            device=target_device,
            dtype=dtype,
            capabilities=linear_capabilities,
            selection_policy=linear_selection_policy,
        ),
        dtype=dtype,
    )
    reader = CheckpointReader(directory)
    state: dict[str, torch.Tensor] = {}
    # state_dict() — NOT named_parameters+named_buffers: non-persistent buffers
    # (rotary inv_freq) are absent from checkpoints by contract (m14 A1)
    expected = model.state_dict()
    quantized_buffer_dtypes = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): buffer.dtype
        for module_name, module in model.named_modules()
        if getattr(module, "is_quantized", False)
        for buffer_name, buffer in module.named_buffers(recurse=False)
        if buffer_name not in module._non_persistent_buffers_set
    }
    for name, current in expected.items():
        if name == "lm_head.weight" and config.tie_word_embeddings:
            continue  # tied: the file omits it; re-tied after load
        if name not in reader:
            raise KeyError(f"checkpoint at {path} is missing tensor {name!r}")
        tensor = reader.tensor(name)
        quantized_dtype = quantized_buffer_dtypes.get(name)
        if quantized_dtype is not None and tensor.dtype != quantized_dtype:
            # The registered buffer is the fused-kernel ABI. In particular,
            # real compressed-tensors FP8 checkpoints may serialize
            # per-channel scales in bf16, while torch._scaled_mm requires
            # float32 scales. Converting bf16 -> fp32 is value-preserving and
            # must not follow the model's bf16 compute dtype.
            tensor = tensor.to(quantized_dtype)
        elif (
            name not in quantized_buffer_dtypes
            and current.dtype == torch.float32
            and tensor.is_floating_point()
        ):
            # regular fp params follow the requested compute dtype; quantized
            # payloads follow the dtype declared by their registered buffer
            tensor = tensor.to(dtype)
        state[name] = tensor
    model.load_state_dict(state, strict=False, assign=True)
    if (
        quant.method is QuantMethod.NVFP4
        and config.architecture == "Qwen3MoeForCausalLM"
    ):
        from kairyu.models.moe import apply_nvfp4_moe_global_input_scales

        for layer in model.model.layers:
            block = layer.mlp
            if hasattr(block, "experts"):
                apply_nvfp4_moe_global_input_scales(block.experts)
    if config.tie_word_embeddings:
        # assign=True replaces the embedding tensor; restore the tie
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()
    return model, config, generation
