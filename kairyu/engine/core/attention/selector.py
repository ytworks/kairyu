"""Backend selection from the hardware profile (m13 D5)."""

from __future__ import annotations

import os

from kairyu.engine.core.hw_profile import HardwareProfile

_ENV_OVERRIDE = "KAIRYU_ATTENTION_BACKEND"


def select_backend_name(profile: HardwareProfile | None = None) -> str:
    """Resolve the attention backend NAME ("torch" | "flashinfer") WITHOUT
    importing or instantiating it — safe on CPU/macOS where flashinfer is absent.

    Same precedence as ``select_backend``: an explicit ``KAIRYU_ATTENTION_BACKEND``
    env override wins, else the profile's kernel tier (fa2/full -> flashinfer,
    everything else incl. CPU/None -> torch). Used by the ``/backends`` introspection
    endpoint, which must not pull flashinfer into a CPU process just to name it."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        if override in ("torch", "flashinfer"):
            return override
        raise ValueError(
            f"unknown {_ENV_OVERRIDE}={override!r}; expected 'torch' or 'flashinfer'"
        )
    tier = profile.kernel_tier if profile is not None else "torch"
    return "flashinfer" if tier in ("fa2", "full") else "torch"


def select_backend(profile: HardwareProfile | None = None):
    """Env override wins; else the profile's kernel tier; CPU -> torch."""
    if select_backend_name(profile) == "flashinfer":
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend()
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    return TorchAttentionBackend()
