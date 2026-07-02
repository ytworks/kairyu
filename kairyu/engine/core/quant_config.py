"""Quantization checkpoint detection from HF model configs (design m2 §2.5, m3).

Pure parsing: given a HuggingFace ``config.json`` dict, decide which weight
loading path (FP8 W8A8 / AWQ / GPTQ / none) the GPU ModelRunner must take.
Unsupported schemes fail fast at load time with the supported list.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantMethod(Enum):
    NONE = "none"
    FP8 = "fp8"
    AWQ = "awq"
    GPTQ = "gptq"


_SUPPORTED = (QuantMethod.FP8, QuantMethod.AWQ, QuantMethod.GPTQ)


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
    raise ValueError(
        "unsupported compressed-tensors scheme: only FP8 (8-bit float weights) is supported"
    )


def detect_quantization(hf_config: dict) -> QuantConfig:
    quant = hf_config.get("quantization_config")
    if not quant:
        return QuantConfig(method=QuantMethod.NONE)
    method = str(quant.get("quant_method", "")).lower()
    if method == "compressed-tensors":
        return _detect_compressed_tensors(quant)
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
