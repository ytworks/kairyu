"""Pure attention-backend selection without importing GPU implementations."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from importlib import metadata

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


def validate_attention_backend_execution_identity(value: object) -> str:
    """Validate the exact canonical execution identity accepted by policy files."""

    if not isinstance(value, str) or not value:
        raise ValueError("attention backend execution identity must be a non-empty string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("attention backend execution identity must be JSON") from error
    expected = {
        "requested",
        "resolved",
        "source",
        "components",
        "component_versions",
        "component_builds",
        "architecture",
    }
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ValueError("attention backend execution identity fields are invalid")
    if any(
        not isinstance(parsed.get(field), str) or not parsed[field]
        for field in ("requested", "resolved", "source")
    ):
        raise ValueError("attention backend execution selection fields are invalid")
    components = parsed["components"]
    architecture = parsed["architecture"]
    versions = parsed["component_versions"]
    builds = parsed["component_builds"]
    if (
        not isinstance(components, dict)
        or not components
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(component, str)
            or not component
            for name, component in components.items()
        )
        or not isinstance(architecture, dict)
        or not isinstance(versions, dict)
        or set(versions) != {"prefill", "decode"}
        or any(
            not isinstance(version, str) or not version or version == "unknown"
            for version in versions.values()
        )
        or not isinstance(builds, dict)
    ):
        raise ValueError("attention backend execution components are invalid")
    uses_flashinfer = "flashinfer" in components.values()
    if uses_flashinfer:
        if set(builds) != {"flashinfer-jit-cache"}:
            raise ValueError("FlashInfer execution build identity is incomplete")
        jit_cache = builds["flashinfer-jit-cache"]
        if not isinstance(jit_cache, dict) or type(jit_cache.get("installed")) is not bool:
            raise ValueError("FlashInfer JIT-cache build identity is invalid")
        if jit_cache["installed"]:
            digest = jit_cache.get("record_sha256")
            if (
                set(jit_cache) != {"installed", "version", "record_sha256"}
                or not isinstance(jit_cache.get("version"), str)
                or not jit_cache["version"]
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("installed FlashInfer JIT-cache build identity is invalid")
        elif set(jit_cache) != {"installed"}:
            raise ValueError("absent FlashInfer JIT-cache build identity is invalid")
    elif builds:
        raise ValueError("non-FlashInfer execution identity has unexpected build metadata")
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    if canonical != value:
        raise ValueError("attention backend execution identity is not canonical")
    return value


def _reported_module_version(module: object, component: str) -> str:
    value = getattr(module, "__version__", None)
    if not isinstance(value, str) or not value or value == "unknown":
        raise RuntimeError(
            f"{component} does not expose an exact runtime __version__; "
            "refusing to construct an attention execution identity"
        )
    return value


def _distribution_build_identity(distribution_name: str) -> dict[str, object]:
    """Exact installed-wheel identity; RECORD hashes distinguish equal versions."""

    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return {"installed": False}
    record = distribution.read_text("RECORD")
    if not isinstance(record, str) or not record:
        raise RuntimeError(
            f"{distribution_name} has no installed RECORD; refusing to construct "
            "an attention execution identity"
        )
    return {
        "installed": True,
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
    }


def attention_backend_execution_identity(
    decision: AttentionBackendDecision,
    backend: object,
) -> str:
    """Canonical rank-invariant identity of the backend that actually runs.

    Selection alone is insufficient for retained performance policy: replacing
    a FlashInfer/FlashAttention wheel can change the restore-versus-recompute
    crossover without changing the public backend name.  This identity extends
    the selection contract with versions read from the constructed backend,
    while deliberately excluding rank-local device indices and state.
    """

    if decision.resolved == "torch":
        import torch

        version = str(torch.__version__)
        component_versions = {"decode": version, "prefill": version}
    elif decision.resolved == "flashinfer":
        module = getattr(backend, "_flashinfer", None)
        version = _reported_module_version(module, "flashinfer")
        component_versions = {"decode": version, "prefill": version}
    elif decision.resolved in {"flashattention3", "flashattention4"}:
        raw_versions = getattr(backend, "component_versions", None)
        if not isinstance(raw_versions, dict):
            raise RuntimeError(
                f"{decision.resolved} does not expose component_versions; "
                "refusing to construct an attention execution identity"
            )
        component_versions = dict(raw_versions)
        decode_backend = getattr(backend, "_decode_backend", None)
        decode_module = getattr(decode_backend, "_flashinfer", None)
        if decode_module is not None:
            component_versions["decode"] = _reported_module_version(
                decode_module,
                "flashinfer delegated decode",
            )
        if set(component_versions) != {"prefill", "decode"} or any(
            not isinstance(value, str) or not value or value == "unknown"
            for value in component_versions.values()
        ):
            raise RuntimeError(
                f"{decision.resolved} component_versions are incomplete; "
                "refusing to construct an attention execution identity"
            )
    else:  # selection validates the public set before constructing a backend
        raise RuntimeError(
            f"unsupported resolved attention backend {decision.resolved!r}"
        )

    selected = json.loads(attention_backend_identity(decision))
    selected["component_versions"] = dict(sorted(component_versions.items()))
    selected["component_builds"] = (
        {"flashinfer-jit-cache": _distribution_build_identity("flashinfer-jit-cache")}
        if "flashinfer" in decision.components.values()
        else {}
    )
    return validate_attention_backend_execution_identity(
        json.dumps(selected, sort_keys=True, separators=(",", ":"))
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
