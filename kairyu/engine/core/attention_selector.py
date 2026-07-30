"""Pure attention-backend selection without importing GPU implementations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from kairyu.engine.core.hw_profile import HardwareProfile

_ENV_OVERRIDE = "KAIRYU_ATTENTION_BACKEND"
_EXPLICIT_BACKENDS = (
    "torch",
    "flashinfer",
    "flashattention3",
    "flashattention4",
)
SUPPORTED_ATTENTION_BACKENDS = ("auto", *_EXPLICIT_BACKENDS)


@dataclass(frozen=True)
class AttentionBackendDecision:
    """A reportable selection result.

    ``resolved`` names the public backend while ``components`` makes hybrid
    phase ownership explicit. FlashAttention prefill delegates decode to
    FlashInfer because the latter already owns Kairyu's capture-safe paged
    decode contract.
    """

    requested: str
    resolved: str
    source: str
    components: dict[str, str]
    rationale: str
    architecture: dict[str, object] = field(default_factory=dict)


def _architecture(profile: HardwareProfile | None) -> dict[str, object]:
    """Return the hardware facts that made the selection reproducible."""
    if profile is None:
        return {
            "arch": "unknown",
            "device_name": None,
            "sm": None,
            "kernel_tier": "torch",
        }
    return {
        "arch": profile.arch,
        "device_name": profile.device_name,
        "sm": profile.sm,
        "kernel_tier": profile.kernel_tier,
    }


def attention_backend_identity(decision: AttentionBackendDecision) -> str:
    """Canonical execution identity exchanged by tensor-parallel ranks.

    Free-form rationale is deliberately excluded: it is diagnostic text, not
    an execution contract.  Device indices and host-wide counts are absent
    from ``architecture`` so equivalent rank-local GPUs compare equal.
    """
    return json.dumps(
        {
            "requested": decision.requested,
            "resolved": decision.resolved,
            "source": decision.source,
            "components": dict(decision.components),
            "architecture": dict(decision.architecture),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def compose_backend_decisions(
    prefill: AttentionBackendDecision,
    decode: AttentionBackendDecision,
) -> AttentionBackendDecision:
    """Report role-specific P-D choices without hiding either half."""
    if prefill == decode:
        return prefill
    prefill_kv = prefill.components.get("kv_mode", "unknown")
    decode_kv = decode.components.get("kv_mode", "unknown")
    return AttentionBackendDecision(
        requested=(prefill.requested if prefill.requested == decode.requested else "role-specific"),
        resolved=(prefill.resolved if prefill.resolved == decode.resolved else "composite"),
        source=(prefill.source if prefill.source == decode.source else "role-specific"),
        components={
            "prefill": prefill.components.get("prefill", prefill.resolved),
            "decode": decode.components.get("decode", decode.resolved),
            "kv_mode": (
                prefill_kv
                if prefill_kv == decode_kv
                else f"prefill:{prefill_kv};decode:{decode_kv}"
            ),
        },
        rationale=(f"prefill role: {prefill.rationale}; decode role: {decode.rationale}"),
        architecture={
            "prefill": dict(prefill.architecture),
            "decode": dict(decode.architecture),
        },
    )


def _automatic_backend(profile: HardwareProfile | None) -> tuple[str, str]:
    tier = profile.kernel_tier if profile is not None else "torch"
    if tier in ("fa2", "full"):
        if profile is not None and profile.sm == 120:
            return (
                "flashinfer",
                "retained SM120 Qwen3-32B TP4/TP8 paired evidence confirms "
                "the stable FlashInfer prefill/decode path",
            )
        return (
            "flashinfer",
            "stable hardware-profile fallback; no retained profile evidence "
            "promotes an experimental FlashAttention prefill path",
        )
    return "torch", "CPU or unsupported GPU profile uses the portable torch backend"


def _components(resolved: str, profile: HardwareProfile | None) -> dict[str, str]:
    if resolved == "torch":
        return {
            "prefill": "torch",
            "decode": "torch",
            "kv_mode": "tensor-gather",
        }
    if resolved == "flashinfer":
        return {
            "prefill": "flashinfer",
            "decode": "flashinfer",
            "kv_mode": "paged-direct",
        }
    if resolved == "flashattention3":
        return {
            "prefill": "flashattention3",
            "decode": "flashinfer",
            "kv_mode": "paged-direct",
        }
    if resolved == "flashattention4":
        # The pinned upstream FA4 beta implements direct paged KV on SM90,
        # SM100 and SM110. Its SM120 path rejects page tables, so that profile
        # uses explicit device-to-device page materialization before prefill.
        # This must stay visible in /backends.
        direct = profile is not None and profile.sm in (90, 100, 110)
        return {
            "prefill": "flashattention4",
            "decode": "flashinfer",
            "kv_mode": "paged-direct" if direct else "paged-materialized",
        }
    raise AssertionError(f"unhandled attention backend {resolved!r}")


def select_backend_decision(
    profile: HardwareProfile | None = None,
) -> AttentionBackendDecision:
    """Resolve the requested backend and its phase-level implementation."""
    override = os.environ.get(_ENV_OVERRIDE)
    requested = override if override else "auto"
    if requested not in SUPPORTED_ATTENTION_BACKENDS:
        expected = ", ".join(repr(name) for name in SUPPORTED_ATTENTION_BACKENDS)
        raise ValueError(f"unknown {_ENV_OVERRIDE}={requested!r}; expected one of {expected}")

    if requested == "auto":
        resolved, rationale = _automatic_backend(profile)
    else:
        resolved = requested
        rationale = f"explicit {_ENV_OVERRIDE} override"
    return AttentionBackendDecision(
        requested=requested,
        resolved=resolved,
        source="env" if override else "hw_profile",
        components=_components(resolved, profile),
        rationale=rationale,
        architecture=_architecture(profile),
    )


def select_backend_name(profile: HardwareProfile | None = None) -> str:
    """Resolve the attention backend name without importing an implementation."""
    return select_backend_decision(profile).resolved
