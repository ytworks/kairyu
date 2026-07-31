"""KV-cache dtype resolution shared by single-rank and tensor-parallel startup.

The public option is intentionally smaller than ``torch.dtype``.  In
particular, FP8 is an execution contract rather than an allocation shortcut:
the hardware, model shape, and selected attention backend must all support it
before a pool can be constructed.
"""

from __future__ import annotations

SUPPORTED_KV_CACHE_DTYPES = ("auto", "bfloat16", "fp8_e4m3")


def validate_kv_cache_dtype(requested: object) -> str:
    """Return a valid public dtype name without importing torch."""

    if not isinstance(requested, str) or requested not in SUPPORTED_KV_CACHE_DTYPES:
        choices = ", ".join(SUPPORTED_KV_CACHE_DTYPES)
        raise ValueError(
            f"kv_cache_dtype must be one of: {choices}; got {requested!r}"
        )
    return requested


def kv_cache_dtype_name(dtype: object) -> str:
    """Canonical public/metadata name for one resolved torch dtype."""

    import torch

    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float8_e4m3fn:
        return "fp8_e4m3"
    name = str(dtype)
    return name.removeprefix("torch.")


def resolve_kv_cache_dtype(
    requested: object,
    compute_dtype: object,
    profile: object,
    attention_backend: object,
    model_config: object,
):
    """Resolve the pool dtype or fail closed before serving.

    ``auto`` preserves the existing compute-dtype cache behavior.  Explicit
    BF16 is available as a stable spelling of the existing CUDA behavior.
    FP8 is deliberately narrower: #170 bakes only E4M3 on real dense models,
    SM120, and a backend that explicitly owns FP8 KV scale semantics.
    """

    import torch

    requested = validate_kv_cache_dtype(requested)
    if requested == "auto":
        return compute_dtype
    if requested == "bfloat16":
        if model_config is None:
            raise ValueError(
                "kv_cache_dtype='bfloat16' requires a real model configuration"
            )
        if compute_dtype is not torch.bfloat16:
            raise RuntimeError(
                "kv_cache_dtype='bfloat16' requires BF16 model compute dtype"
            )
        return torch.bfloat16

    if model_config is None:
        raise ValueError(
            "kv_cache_dtype='fp8_e4m3' requires a real model configuration"
        )
    if getattr(profile, "arch", None) != "cuda" or getattr(profile, "sm", None) != 120:
        raise RuntimeError(
            "kv_cache_dtype='fp8_e4m3' requires CUDA SM120; "
            f"detected arch={getattr(profile, 'arch', None)!r}, "
            f"sm={getattr(profile, 'sm', None)!r}"
        )
    if compute_dtype is not torch.bfloat16:
        raise RuntimeError(
            "kv_cache_dtype='fp8_e4m3' requires BF16 model compute dtype"
        )
    if bool(getattr(model_config, "is_mla", False)):
        raise ValueError("kv_cache_dtype='fp8_e4m3' does not support MLA models")
    if not bool(getattr(attention_backend, "supports_fp8_kv", False)):
        raise RuntimeError(
            "kv_cache_dtype='fp8_e4m3' is unsupported by the selected "
            f"attention backend {type(attention_backend).__name__}"
        )
    return torch.float8_e4m3fn


__all__ = [
    "SUPPORTED_KV_CACHE_DTYPES",
    "kv_cache_dtype_name",
    "resolve_kv_cache_dtype",
    "validate_kv_cache_dtype",
]
