"""Opt-in NVFP4 accuracy profiles and runtime evidence helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod

_NAMED_SELECTORS = frozenset({"down_proj", "shared_experts"})


def _parse_selectors(field: str, value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"NVFP4 accuracy profile {field} must be a list")
    selectors: list[str] = []
    for selector in value:
        if not isinstance(selector, str) or not selector:
            raise ValueError(
                f"NVFP4 accuracy profile {field} selectors must be non-empty strings"
            )
        if selector in _NAMED_SELECTORS:
            selectors.append(selector)
            continue
        prefix, separator, count = selector.partition(":")
        if separator != ":" or prefix not in {"first", "last"}:
            raise ValueError(f"unknown NVFP4 accuracy selector {selector!r}")
        try:
            parsed = int(count)
        except ValueError:
            parsed = -1
        if parsed < 1 or str(parsed) != count:
            raise ValueError(f"NVFP4 accuracy selector {selector!r} needs a positive count")
        selectors.append(selector)
    if len(set(selectors)) != len(selectors):
        raise ValueError(f"NVFP4 accuracy profile {field} contains duplicate selectors")
    return tuple(selectors)


@dataclass(frozen=True)
class NvFp4AccuracyProfile:
    """Serializable opt-in selectors; an empty profile preserves the checkpoint."""

    fp8: tuple[str, ...] = ()
    dynamic_activation: tuple[str, ...] = ()
    saturation_counters: bool = False

    @classmethod
    def parse(cls, value: object) -> NvFp4AccuracyProfile:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) for key in value
        ):
            raise ValueError("NVFP4 accuracy profile must be a mapping")
        unknown = set(value) - {"fp8", "dynamic_activation", "saturation_counters"}
        if unknown:
            raise ValueError("NVFP4 accuracy profile has unsupported fields")
        counters = value.get("saturation_counters", False)
        if type(counters) is not bool:
            raise ValueError("NVFP4 saturation_counters must be a boolean")
        return cls(
            fp8=_parse_selectors("fp8", value.get("fp8")),
            dynamic_activation=_parse_selectors(
                "dynamic_activation", value.get("dynamic_activation")
            ),
            saturation_counters=counters,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "fp8": list(self.fp8),
            "dynamic_activation": list(self.dynamic_activation),
            "saturation_counters": self.saturation_counters,
        }

    @property
    def active(self) -> bool:
        return bool(self.fp8 or self.dynamic_activation or self.saturation_counters)


def make_nvfp4_accuracy_policy(profile: NvFp4AccuracyProfile, num_layers: int):
    """Return a contextual policy layered over the normal checkpoint policy."""

    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("NVFP4 accuracy policy requires a positive layer count")
    for selector in (*profile.fp8, *profile.dynamic_activation):
        prefix, separator, count = selector.partition(":")
        if separator and prefix in {"first", "last"} and int(count) > num_layers:
            raise ValueError(
                f"NVFP4 accuracy selector {selector!r} exceeds {num_layers} layers"
            )
    from kairyu.quant.linear import (
        ExpertScope,
        LinearRole,
        LinearSelection,
        default_linear_selection,
    )

    def matches(selectors: tuple[str, ...], context) -> bool:
        for selector in selectors:
            if selector == "down_proj" and context.role is LinearRole.MLP_DOWN:
                return True
            if selector == "shared_experts" and context.expert_scope is ExpertScope.SHARED:
                return True
            prefix, separator, count = selector.partition(":")
            if separator and context.layer_index is not None:
                width = int(count)
                if prefix == "first" and context.layer_index < width:
                    return True
                if prefix == "last" and context.layer_index >= num_layers - width:
                    return True
        return False

    def policy(checkpoint: QuantConfig, context):
        baseline = default_linear_selection(checkpoint, context)
        if baseline.quantization.method is not QuantMethod.NVFP4:
            return baseline
        observe = profile.saturation_counters
        if matches(profile.fp8, context):
            return LinearSelection(
                QuantConfig(
                    method=QuantMethod.FP8,
                    weight_bits=8,
                    activation_bits=8,
                    activation_dynamic=True,
                ),
                None,
                "nvfp4-accuracy:fp8",
                # These counters diagnose NVFP4 activation clipping.  An FP8
                # pin uses its own dynamic scale, so reporting zero clipping
                # for it would be vacuous and misleading.
                observe_activation_saturation=False,
            )
        if matches(profile.dynamic_activation, context):
            dynamic = QuantConfig(
                method=QuantMethod.NVFP4,
                weight_bits=4,
                activation_bits=4,
                group_size=16,
                activation_dynamic=True,
                ignored_layers=checkpoint.ignored_layers,
            )
            return LinearSelection(
                dynamic,
                None,
                "nvfp4-accuracy:dynamic-activation",
                observe_activation_saturation=observe,
            )
        if observe:
            return LinearSelection(
                baseline.quantization,
                baseline.kernel,
                baseline.reason,
                observe_activation_saturation=True,
            )
        return baseline

    return policy


def resolve_nvfp4_accuracy_policy(
    checkpoint: QuantConfig,
    profile_value: object,
    *,
    num_layers: int,
):
    profile = NvFp4AccuracyProfile.parse(profile_value)
    if profile.active and checkpoint.method is not QuantMethod.NVFP4:
        raise ValueError("nvfp4_accuracy_profile requires an NVFP4 checkpoint")
    return profile, (make_nvfp4_accuracy_policy(profile, num_layers) if profile.active else None)


def saturation_snapshot(model, *, reset: bool = False) -> list[dict[str, object]]:
    """Synchronizing operator snapshot of opt-in device-resident counters."""

    rows: list[dict[str, object]] = []
    for name, module in model.named_modules():
        if not bool(getattr(module, "observe_activation_saturation", False)):
            continue
        total_tensor = getattr(module, "activation_blocks", None)
        saturated_tensor = getattr(module, "saturated_activation_blocks", None)
        if total_tensor is None or saturated_tensor is None:
            raise RuntimeError(f"observed projection {name!r} has no counters")
        total = int(total_tensor.item())
        saturated = int(saturated_tensor.item())
        rows.append(
            {
                "projection": name,
                "runtime_format": getattr(module, "quant_scheme", None),
                "activation_scale": (
                    "dynamic"
                    if bool(getattr(module, "activation_dynamic", False))
                    else "static"
                ),
                "blocks": total,
                "saturated_blocks": saturated,
                "saturation_rate": saturated / total if total else 0.0,
            }
        )
        if reset:
            total_tensor.zero_()
            saturated_tensor.zero_()
    return rows


def materialize_fp8_weights(model) -> None:
    """Finalize every NVFP4-source/FP8-runtime projection after checkpoint slicing."""

    from kairyu.quant.linear import NvFp4ToFp8Linear

    for module in model.modules():
        if isinstance(module, NvFp4ToFp8Linear):
            module.materialize_runtime_weight()
