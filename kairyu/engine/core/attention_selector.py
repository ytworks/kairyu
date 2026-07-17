"""Pure attention-backend name selection without engine dependencies."""

from __future__ import annotations

import os

from kairyu.engine.core.hw_profile import HardwareProfile

_ENV_OVERRIDE = "KAIRYU_ATTENTION_BACKEND"


def select_backend_name(profile: HardwareProfile | None = None) -> str:
    """Resolve the attention backend name without importing an implementation."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        if override in ("torch", "flashinfer"):
            return override
        raise ValueError(
            f"unknown {_ENV_OVERRIDE}={override!r}; expected 'torch' or 'flashinfer'"
        )
    tier = profile.kernel_tier if profile is not None else "torch"
    return "flashinfer" if tier in ("fa2", "full") else "torch"
