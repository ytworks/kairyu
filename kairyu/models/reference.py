"""Correctness-first adapters for newly released hybrid decoder families.

These models intentionally run one complete text sequence per step.  Their
recurrent/compressed attention state does not fit Kairyu's paged-KV contract
yet; recomputation keeps the scheduler and sampling semantics correct without
claiming an unvalidated cache or distributed fast path.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from kairyu.engine.core.weights import CheckpointReader
from kairyu.models.config import ModelConfig


class ReferenceDecoder(nn.Module):
    """Small Kairyu-facing wrapper around an official text-only decoder."""

    def __init__(self, model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.hf_model = model
        self.config = config

    @torch.inference_mode()
    def forward_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.hf_model(
            input_ids=token_ids.unsqueeze(0),
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[0]


def _transformers_classes(raw_config: dict, architecture: str):
    try:
        from transformers import (
            DeepseekV4Config,
            DeepseekV4ForCausalLM,
            Qwen3_5ForCausalLM,
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeTextConfig,
            Qwen3_5TextConfig,
        )
    except ImportError as exc:
        raise RuntimeError(f"{architecture} requires transformers>=5.12,<5.13") from exc

    text = raw_config.get("text_config", raw_config)
    if architecture == "DeepseekV4ForCausalLM":
        return DeepseekV4Config.from_dict(raw_config), DeepseekV4ForCausalLM
    if architecture in (
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
    ):
        return Qwen3_5TextConfig.from_dict(text), Qwen3_5ForCausalLM
    if architecture in (
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        return Qwen3_5MoeTextConfig.from_dict(text), Qwen3_5MoeForCausalLM
    raise ValueError(f"no transformers reference class for {architecture!r}")


def _kimi_classes(directory: Path, raw_config: dict):
    text = raw_config.get("text_config", raw_config)
    auto_map = text.get("auto_map") or {}
    config_ref = auto_map.get("AutoConfig")
    model_ref = auto_map.get("AutoModelForCausalLM")
    if not config_ref or not model_ref:
        raise ValueError(
            "Kimi K3 text_config must include AutoConfig and "
            "AutoModelForCausalLM custom-code mappings"
        )
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config_cls = get_class_from_dynamic_module(config_ref, str(directory))
        model_cls = get_class_from_dynamic_module(model_ref, str(directory))
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Kimi K3 requires the checkpoint's public custom Python files and "
            "their fla-core dependency"
        ) from exc
    return config_cls(**text), model_cls


def _checkpoint_candidates(name: str, architecture: str) -> tuple[str, ...]:
    if architecture in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        if name.startswith("model."):
            return ("model.language_model." + name.removeprefix("model."),)
        return (name,)
    if architecture == "KimiK3ForConditionalGeneration":
        return ("language_model." + name,)
    return (name,)


def _packed_expert_tensor(
    reader: CheckpointReader,
    name: str,
    config: ModelConfig,
) -> torch.Tensor | None:
    if config.moe is None:
        return None
    suffix = None
    if name.endswith(".experts.gate_up_proj"):
        suffix = "gate_up"
    elif name.endswith(".experts.down_proj"):
        suffix = "down"
    if suffix is None:
        return None

    packed_prefix = name.rsplit(".experts.", 1)[0] + ".experts"
    mapped_prefix = _checkpoint_candidates(packed_prefix, config.architecture)[0]
    experts: list[torch.Tensor] = []
    for expert_index in range(config.moe.num_experts):
        prefix = f"{mapped_prefix}.{expert_index}"
        if suffix == "gate_up":
            gate_name = f"{prefix}.w1.weight"
            up_name = f"{prefix}.w3.weight"
            if gate_name not in reader or up_name not in reader:
                return None
            experts.append(
                torch.cat(
                    (reader.tensor(gate_name), reader.tensor(up_name)),
                    dim=0,
                )
            )
        else:
            down_name = f"{prefix}.w2.weight"
            if down_name not in reader:
                return None
            experts.append(reader.tensor(down_name))
    return torch.stack(experts)


def _load_reference_weights(
    model: nn.Module,
    directory: Path,
    config: ModelConfig,
    dtype: torch.dtype,
) -> None:
    reader = CheckpointReader(directory)
    state: dict[str, torch.Tensor] = {}
    expected = model.state_dict()
    for name, current in expected.items():
        checkpoint_name = next(
            (
                candidate
                for candidate in _checkpoint_candidates(name, config.architecture)
                if candidate in reader
            ),
            None,
        )
        if checkpoint_name is None:
            tensor = _packed_expert_tensor(reader, name, config)
            if tensor is None:
                if name == "lm_head.weight" and config.tie_word_embeddings:
                    continue
                raise KeyError(f"checkpoint at {directory} is missing text tensor {name!r}")
        else:
            tensor = reader.tensor(checkpoint_name)
        if current.is_floating_point() and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        state[name] = tensor
    model.load_state_dict(state, strict=False, assign=True)
    if config.tie_word_embeddings and "lm_head.weight" not in state:
        model.lm_head.weight = model.model.embed_tokens.weight


def load_deepseek_public_decoder(
    directory: Path,
    raw_config: dict,
    config: ModelConfig,
    *,
    dtype: torch.dtype,
) -> ReferenceDecoder:
    """Load the published block-FP8 checkpoint through transformers."""

    hf_config, model_cls = _transformers_classes(raw_config, config.architecture)
    hf_config._attn_implementation = "eager"
    try:
        model = model_cls.from_pretrained(
            directory,
            config=hf_config,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(
            "published DeepSeek V4 block-FP8 loading requires the "
            "transformers fine-grained FP8 runtime"
        ) from exc
    model.eval()
    return ReferenceDecoder(model, config)


def load_kimi_public_decoder(
    directory: Path,
    config: ModelConfig,
    *,
    dtype: torch.dtype,
) -> ReferenceDecoder:
    """Load the published nested-MXFP4 checkpoint through its official code."""

    try:
        from transformers import AutoModelForCausalLM

        outer = AutoModelForCausalLM.from_pretrained(
            directory,
            trust_remote_code=True,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError(
            "published Kimi K3 MXFP4 loading requires the checkpoint's custom "
            "code plus fla-core and compressed-tensors"
        ) from exc
    model = getattr(outer, "language_model", outer)
    model.eval()
    return ReferenceDecoder(model, config)


def load_reference_decoder(
    directory: Path,
    raw_config: dict,
    config: ModelConfig,
    *,
    dtype: torch.dtype,
) -> ReferenceDecoder:
    """Build and load a single-device text decoder for one hybrid family."""

    if config.architecture in (
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    ):
        hf_config, model_cls = _kimi_classes(directory, raw_config)
    else:
        hf_config, model_cls = _transformers_classes(raw_config, config.architecture)
    # Eager attention is the portable oracle used by the tiny parity suite.
    # Kimi's constructor currently rewrites this value, so set it on both sides.
    hf_config._attn_implementation = "eager"
    if config.architecture == "DeepseekV4ForCausalLM":
        model = model_cls.from_pretrained(
            directory,
            config=hf_config,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="eager",
        )
        model.to(dtype=dtype)
    else:
        model = model_cls(hf_config)
        model.config._attn_implementation = "eager"
        _load_reference_weights(model, directory, config, dtype)
    model.eval()
    return ReferenceDecoder(model, config)
