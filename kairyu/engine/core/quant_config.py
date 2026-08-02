"""Quantization checkpoint detection from HF model configs (design m2 §2.5, m8 D5).

Pure parsing: given a HuggingFace ``config.json`` dict, decide which weight
loading path (FP8 W8A8 / INT8 W8A8 / AWQ / GPTQ / NVFP4 / none) the ModelRunner
must take. Unsupported schemes fail fast at load time with the supported list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


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
    activation_dynamic: bool | None = None
    ignored_layers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckpointQuantization:
    """Resolved weight/KV metadata for one local HF checkpoint.

    NVIDIA ModelOpt checkpoints may keep their quantization declaration in a
    sibling ``hf_quant_config.json`` instead of embedding it in
    ``config.json``.  Runtime callers must use this resolved form so a packed
    checkpoint can never be mistaken for an unquantized one.
    """

    weights: QuantConfig
    source: str
    kv_cache_quant_algo: str | None = None
    producer_name: str | None = None
    producer_version: str | None = None


def _ignored_layers(quant: dict) -> tuple[str, ...]:
    raw = quant.get("ignore", ())
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(pattern, str) or not pattern for pattern in raw
    ):
        raise ValueError("quantization ignore must contain non-empty strings")
    return tuple(raw)


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
            activation_dynamic=bool(activations.get("dynamic", False)),
            ignored_layers=_ignored_layers(quant),
        )
    if weights.get("type") == "int" and weights.get("num_bits") == 8:
        return QuantConfig(
            method=QuantMethod.INT8,
            weight_bits=8,
            activation_bits=activations.get("num_bits"),
            ignored_layers=_ignored_layers(quant),
        )
    if weights.get("type") == "float" and weights.get("num_bits") == 4:
        # compressed-tensors FP4 uses DIFFERENT tensor names (weight_packed)
        # and an INVERTED global scale vs modelopt — silently flowing it into
        # the modelopt module would corrupt weights (m14 review A8)
        raise ValueError(
            "compressed-tensors FP4 checkpoints are not supported; "
            "use a modelopt NVFP4 checkpoint (quant_method: modelopt)"
        )
    raise ValueError("unsupported compressed-tensors scheme: FP8/INT8 W8A8 are supported")


def _detect_modelopt(quant: dict) -> QuantConfig:
    """NVIDIA modelopt checkpoints (e.g. nvidia/*-NVFP4): quant_algo selects."""
    algo = str(quant.get("quant_algo", "")).upper()
    if algo == "NVFP4":
        group_size = quant.get("group_size", 16)
        if type(group_size) is not int or group_size != 16:
            raise ValueError(
                "unsupported modelopt NVFP4 group_size "
                f"{group_size!r}: the implemented checkpoint ABI requires 16"
            )
        return QuantConfig(
            method=QuantMethod.NVFP4,
            weight_bits=4,
            group_size=group_size,
            ignored_layers=_ignored_layers(quant),
        )
    if algo == "FP8":
        scheme = str(quant.get("activation_scheme", "dynamic")).lower()
        if scheme not in ("dynamic", "static"):
            raise ValueError(
                f"unsupported modelopt FP8 activation_scheme {scheme!r}: supported: dynamic, static"
            )
        return QuantConfig(
            method=QuantMethod.FP8,
            weight_bits=8,
            activation_bits=8,
            activation_dynamic=scheme == "dynamic",
            ignored_layers=_ignored_layers(quant),
        )
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
        scheme = str(quant.get("activation_scheme", "dynamic")).lower()
        if scheme not in ("dynamic", "static"):
            raise ValueError(
                f"unsupported FP8 activation_scheme {scheme!r}: supported: dynamic, static"
            )
        return QuantConfig(
            method=QuantMethod.FP8,
            weight_bits=8,
            activation_bits=8,
            activation_dynamic=scheme == "dynamic",
            ignored_layers=_ignored_layers(quant),
        )
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
            ignored_layers=_ignored_layers(quant),
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
            ignored_layers=_ignored_layers(quant),
        )
    supported = ", ".join(m.value for m in _SUPPORTED)
    raise ValueError(f"unsupported quant_method {method!r}; supported: {supported}")


def load_checkpoint_quantization(
    model_dir: str | Path,
    hf_config: dict,
) -> CheckpointQuantization:
    """Resolve embedded or external checkpoint quantization metadata.

    ``hf_quant_config.json`` is the public NVIDIA ModelOpt layout used by
    ``nvidia/*-NVFP4`` checkpoints.  It is translated into the same pure
    ``detect_quantization`` contract as embedded metadata.  If both forms are
    present they must agree exactly; silently preferring either one could bind
    packed tensors to the wrong module ABI.
    """

    directory = Path(model_dir)
    external_path = directory / "hf_quant_config.json"
    embedded = detect_quantization(hf_config)
    if not external_path.is_file():
        return CheckpointQuantization(
            weights=embedded,
            source="config.json",
        )

    try:
        payload = json.loads(
            external_path.read_text(),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"invalid external quantization metadata at {external_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("hf_quant_config.json must contain a JSON object")
    producer = payload.get("producer")
    quantization = payload.get("quantization")
    if not isinstance(producer, dict) or not isinstance(quantization, dict):
        raise ValueError(
            "hf_quant_config.json requires object-valued producer and quantization"
        )
    producer_name = producer.get("name")
    producer_version = producer.get("version")
    if (
        producer_name != "modelopt"
        or not isinstance(producer_version, str)
        or not producer_version
        or producer_version.strip() != producer_version
    ):
        raise ValueError(
            "hf_quant_config.json requires producer name 'modelopt' and a version"
        )

    excludes = quantization.get("exclude_modules", ())
    if not isinstance(excludes, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in excludes
    ):
        raise ValueError(
            "hf_quant_config.json exclude_modules must contain non-empty strings"
        )
    translated = {
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": quantization.get("quant_algo"),
            "group_size": quantization.get("group_size", 16),
            "ignore": list(excludes),
        }
    }
    external = detect_quantization(translated)
    if embedded.method is not QuantMethod.NONE and embedded != external:
        raise ValueError(
            "config.json and hf_quant_config.json quantization metadata disagree"
        )

    raw_kv_algo = quantization.get("kv_cache_quant_algo")
    kv_algo = None if raw_kv_algo is None else str(raw_kv_algo).upper()
    if kv_algo not in (None, "FP8"):
        raise ValueError(
            "unsupported ModelOpt kv_cache_quant_algo "
            f"{raw_kv_algo!r}; supported metadata: FP8"
        )
    return CheckpointQuantization(
        weights=external,
        source=(
            "config.json+hf_quant_config.json"
            if embedded.method is not QuantMethod.NONE
            else "hf_quant_config.json"
        ),
        kv_cache_quant_algo=kv_algo,
        producer_name=producer_name,
        producer_version=producer_version,
    )


def validate_tensor_parallel_quantization(quant: QuantConfig) -> None:
    """Validate the quantization formats implemented by the TP shard loader."""

    if quant.method not in (QuantMethod.NONE, QuantMethod.FP8):
        raise ValueError(
            "tensor parallelism currently supports dense and FP8 checkpoints; "
            f"got {quant.method.value}"
        )


def validate_model_quantization(
    quant: QuantConfig,
    *,
    is_mla: bool,
    architecture: str,
) -> None:
    """Validate quantization combinations rejected before model construction."""

    if is_mla and quant.method is not QuantMethod.NONE:
        raise ValueError(
            f"{architecture} MLA does not support {quant.method.value} "
            "checkpoints: its absorbed KV projection requires an unquantized "
            "kv_b_proj weight"
        )
