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
    if select_backend_name(profile) == "flashinfer":
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend(device="cuda" if device is None else device)
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    return TorchAttentionBackend()
