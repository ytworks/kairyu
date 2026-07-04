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
        # compressed-tensors FP4 uses DIFFERENT tensor names (weight_packed)
        # and an INVERTED global scale vs modelopt — silently flowing it into
        # the modelopt module would corrupt weights (m14 review A8)
        raise ValueError(
            "compressed-tensors FP4 checkpoints are not supported; "
            "use a modelopt NVFP4 checkpoint (quant_method: modelopt)"
        )
    raise ValueError(
        "unsupported compressed-tensors scheme: FP8/INT8 W8A8 are supported"
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
        if quant.get("weight_block_size") is not None:
            # DeepSeek-style block FP8 uses INVERSE-semantics weight_scale_inv
            # tensors; routing it to the per-channel Fp8Linear misreads the scale
            # (KeyError at best, silent multiply-instead-of-divide at worst) (M5)
            raise ValueError(
                "block-wise FP8 (weight_block_size) is not yet supported; only "
                "per-tensor/per-channel FP8 checkpoints load"
            )
        return QuantConfig(method=QuantMethod.FP8, weight_bits=8, activation_bits=8)
    if method == "awq":
        version = str(quant.get("version", "gemm")).lower()
        if version != "gemm":
            raise ValueError(
                f"unsupported AWQ version {version!r}: only 'gemm' layout is supported"
            )
        return QuantConfig(
            method=QuantMethod.AWQ,
            weight_bits=quant.get("bits"),
            group_size=quant.get("group_size"),
        )
    if method == "gptq":
        if quant.get("bits") != 4:
            raise ValueError(f"unsupported GPTQ bits={quant.get('bits')}: only 4-bit")
        if str(quant.get("checkpoint_format", "gptq")).lower() == "gptq_v2":
            raise ValueError(
                "gptq_v2 checkpoints are not supported (v2 drops the zero-point "
                "storage offset; loading as v1 would shift every zero by 1)"
            )
        return QuantConfig(
            method=QuantMethod.GPTQ,
            weight_bits=quant.get("bits"),
            group_size=quant.get("group_size"),
        )
    supported = ", ".join(m.value for m in _SUPPORTED)
    raise ValueError(f"unsupported quant_method {method!r}; supported: {supported}")
