"""Backend selection from the hardware profile (m13 D5)."""

from __future__ import annotations

from kairyu.engine.core.attention_selector import select_backend_name
from kairyu.engine.core.hw_profile import HardwareProfile


def select_backend(
    profile: HardwareProfile | None = None,
    *,
    device: object | None = None,
):
    """Env override wins; else the profile's kernel tier; CPU -> torch.

    ``device`` binds stateful GPU backends to one role's device.  Leaving a
    FlashInfer backend at its ``"cuda"`` default makes every later allocation
    follow the thread's current device, which is not stable when one process
    owns distinct prefill and decode GPUs.
    """
    name = select_backend_name(profile)
    selected_device = "cuda" if device is None else device
    if name == "flashinfer":
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend(device=selected_device)
    if name in ("flashattention3", "flashattention4"):
        from kairyu.engine.core.attention.flashattention_gpu import (
            FlashAttentionBackend,
        )

        generation = 3 if name == "flashattention3" else 4
        return FlashAttentionBackend(
            generation=generation,
            profile=profile,
            device=selected_device,
        )
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    return TorchAttentionBackend()
