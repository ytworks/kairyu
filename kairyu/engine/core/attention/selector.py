"""Backend selection from the hardware profile (m13 D5)."""

from __future__ import annotations

import os

from kairyu.engine.core.hw_profile import HardwareProfile

_ENV_OVERRIDE = "KAIRYU_ATTENTION_BACKEND"


def select_backend(profile: HardwareProfile | None = None):
    """Env override wins; else the profile's kernel tier; CPU -> torch."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        if override == "torch":
            from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

            return TorchAttentionBackend()
        if override == "flashinfer":
            from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

            return FlashInferBackend()
        raise ValueError(
            f"unknown {_ENV_OVERRIDE}={override!r}; expected 'torch' or 'flashinfer'"
        )
    tier = profile.kernel_tier if profile is not None else "torch"
    if tier in ("fa2", "full"):
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend()
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    return TorchAttentionBackend()
