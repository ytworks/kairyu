"""Backend selection from the hardware profile (m13 D5)."""

from __future__ import annotations

from kairyu.engine.core.attention_selector import select_backend_name
from kairyu.engine.core.hw_profile import HardwareProfile


def select_backend(profile: HardwareProfile | None = None):
    """Env override wins; else the profile's kernel tier; CPU -> torch."""
    name = select_backend_name(profile)
    if name == "flashinfer":
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend()
    if name in ("flashattention3", "flashattention4"):
        from kairyu.engine.core.attention.flashattention_gpu import (
            FlashAttentionBackend,
        )

        generation = 3 if name == "flashattention3" else 4
        return FlashAttentionBackend(generation=generation, profile=profile)
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    return TorchAttentionBackend()
