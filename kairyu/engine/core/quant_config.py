"""Quantization checkpoint detection from HF model configs (design m2 §2.5, m8 D5).

Pure parsing: given a HuggingFace ``config.json`` dict, decide which weight
loading path (FP8 W8A8 / INT8 W8A8 / AWQ / GPTQ / NVFP4 / none) the ModelRunner
must take. Unsupported schemes fail fast at load time with the supported list.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantMethod(Enum):
    NONE = "none"
    FP8 = "fp8"
    INT8 = "int8"
    AWQ = "awq"
    GPTQ = "gptq"
    NVFP4 = "nvfp4"


_SUPPORTED = (
    QuantMethod.FP8,
    QuantMethod.INT8,
    QuantMethod.AWQ,
    QuantMethod.GPTQ,
    QuantMethod.NVFP4,
)


@dataclass(frozen=True)
class QuantConfig:
    method: QuantMethod
    weight_bits: int | None = None
    activation_bits: int | None = None
    group_size: int | None = None


def _detect_compressed_tensors(quant: dict) -> QuantConfig:
    groups = quant.get("config_groups", {})
    first_group = next(iter(groups.values()), {})
    weights = first_group.get("weights", {})
    activations = first_group.get("input_activations", {})
    if weights.get("type") == "float" and weights.get("num_bits") == 8:
        return QuantConfig(
            method=QuantMethod.FP8,
            weight_bits=8,
            activation_bits=activations.get("num_bits"),
        )
    if weights.get("type") == "int" and weights.get("num_bits") == 8:
        return QuantConfig(
            method=QuantMethod.INT8,
            weight_bits=8,
            activation_bits=activations.get("num_bits"),
        )
    if weights.get("type") == "float" and weights.get("num_bits") == 4:
        return QuantConfig(
            method=QuantMethod.NVFP4,
            weight_bits=4,
            group_size=weights.get("group_size"),
        )
    raise ValueError(
        "unsupported compressed-tensors scheme: FP8/INT8 W8A8 and FP4 weights are supported"
    )


def _detect_modelopt(quant: dict) -> QuantConfig:
    """NVIDIA modelopt checkpoints (e.g. nvidia/*-NVFP4): quant_algo selects."""
    algo = str(quant.get("quant_algo", "")).upper()
    if algo == "NVFP4":
        return QuantConfig(
            method=QuantMethod.NVFP4,
            weight_bits=4,
            group_size=quant.get("group_size", 16),
        )
    if algo == "FP8":
        return QuantConfig(method=QuantMethod.FP8, weight_bits=8, activation_bits=8)
    raise ValueError(f"unsupported modelopt quant_algo {algo!r}; supported: NVFP4, FP8")


def detect_quantization(hf_config: dict) -> QuantConfig:
    quant = hf_config.get("quantization_config")
    if not quant:
        return QuantConfig(method=QuantMethod.NONE)
    method = str(quant.get("quant_method", "")).lower()
    if method == "compressed-tensors":
        return _detect_compressed_tensors(quant)
    if method == "modelopt":
        return _detect_modelopt(quant)
    if method == "fp8":
        return QuantConfig(method=QuantMethod.FP8, weight_bits=8, activation_bits=8)
    if method == "awq":
        return QuantConfig(
            method=QuantMethod.AWQ,
            weight_bits=quant.get("bits"),
            group_size=quant.get("group_size"),
        )
    if method == "gptq":
        return QuantConfig(
            method=QuantMethod.GPTQ,
            weight_bits=quant.get("bits"),
            group_size=quant.get("group_size"),
        )
    supported = ", ".join(m.value for m in _SUPPORTED)
    raise ValueError(f"unsupported quant_method {method!r}; supported: {supported}")
