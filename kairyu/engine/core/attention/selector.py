"""Backend selection from the hardware profile (m13 D5)."""

from __future__ import annotations

from kairyu.engine.core.attention_selector import select_backend_name
from kairyu.engine.core.hw_profile import HardwareProfile


def select_backend(profile: HardwareProfile | None = None):
    """Env override wins; else the profile's kernel tier; CPU -> torch."""
    if select_backend_name(profile) == "flashinfer":
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend()
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    return TorchAttentionBackend()
