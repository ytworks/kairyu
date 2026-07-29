"""Backend selection from the hardware profile (m13 D5)."""

from __future__ import annotations

from kairyu.engine.core.attention_selector import (
    AttentionBackendDecision,
    select_backend_decision,
)
from kairyu.engine.core.attention_selector import select_backend_name as select_backend_name
from kairyu.engine.core.hw_profile import HardwareProfile


def _construct_backend(
    name: str,
    profile: HardwareProfile | None,
    *,
    device: object | None,
):
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


def _record_decision(backend, decision: AttentionBackendDecision):
    # Attention backends are intentionally plain objects rather than
    # ``nn.Module`` instances (m13 seam safety), so this process-level
    # selection metadata cannot enter a model state_dict.
    backend.selection_decision = decision
    return backend


def select_backend(
    profile: HardwareProfile | None = None,
    *,
    device: object | None = None,
):
    """Construct the selected backend and retain the decision actually used.

    Explicit selections are strict. ``auto`` alone may fall back to torch when
    the profile-selected optional backend cannot be constructed; the sanitized
    failure type remains reportable through ``/backends``. ``device`` binds all
    stateful component workspaces to the role's explicit device.
    """
    decision = select_backend_decision(profile)
    try:
        backend = _construct_backend(decision.resolved, profile, device=device)
    except Exception as error:
        if decision.requested != "auto" or decision.resolved == "torch":
            raise
        fallback = AttentionBackendDecision(
            requested=decision.requested,
            resolved="torch",
            source=decision.source,
            components={
                "prefill": "torch",
                "decode": "torch",
                "kv_mode": "tensor-gather",
            },
            rationale=(
                f"{decision.rationale}; automatic torch fallback after "
                f"{decision.resolved} construction raised "
                f"{type(error).__name__}"
            ),
        )
        return _record_decision(
            _construct_backend("torch", profile, device=device),
            fallback,
        )
    return _record_decision(backend, decision)
