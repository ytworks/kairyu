"""KV-cache dtype resolution shared by single-rank and tensor-parallel startup.

The public option is intentionally smaller than ``torch.dtype``.  In
particular, FP8 is an execution contract rather than an allocation shortcut.
The 2026-07-31 G4 E-KV bake rejected the unit-scale candidate, so the public
request remains disabled while the internal candidate stays available for
calibration work and offline evidence replay.
"""

from __future__ import annotations

SUPPORTED_KV_CACHE_DTYPES = ("auto", "bfloat16")
FP8_KV_CACHE_DTYPE = "fp8_e4m3"
FP8_KV_DISABLED_REASON = (
    "kv_cache_dtype='fp8_e4m3' is disabled because the G4 E-KV "
    "Qwen3-32B SM120 bake failed its output/logprob/cache-quality gates; "
    "calibrated per-layer K/V scales require a new bake before serving"
)


def validate_kv_cache_dtype(requested: object) -> str:
    """Return a valid public dtype name without importing torch."""

    if requested == FP8_KV_CACHE_DTYPE:
        raise RuntimeError(FP8_KV_DISABLED_REASON)
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
    The rejected FP8 candidate is not reachable through this public resolver.
    Its offline pool/kernel path remains testable without weakening startup.
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

    raise AssertionError(f"unreachable KV cache dtype request: {requested!r}")


__all__ = [
    "FP8_KV_CACHE_DTYPE",
    "FP8_KV_DISABLED_REASON",
    "SUPPORTED_KV_CACHE_DTYPES",
    "kv_cache_dtype_name",
    "resolve_kv_cache_dtype",
    "validate_kv_cache_dtype",
]
