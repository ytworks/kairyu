"""Independent quantization contract for EAGLE and MTP draft checkpoints.

Target-model quantization and external draft-head quantization are deliberately
separate decisions.  EAGLE reads the draft checkpoint's standard Hugging Face
``quantization_config`` instead of inheriting the target's.  In-checkpoint MTP
inherits the target format by default and may override it with
``draft_quantization_config`` (an empty object explicitly selects dense).

Only dynamic FP8 W8A8 is admitted initially.  The remaining production linear
kernels are not silently selected for draft heads until their checkpoint and
quality contracts receive the same validation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from kairyu.engine.core.quant_config import (
    QuantConfig,
    QuantMethod,
    detect_quantization,
)
from kairyu.quant.fp8 import quantize_fp8
from kairyu.quant.linear import (
    Fp8Linear,
    LinearBuildContext,
    LinearRole,
    LinearSelection,
    ModelScope,
    linear_factory,
)

MTP_DRAFT_QUANTIZATION_KEY = "draft_quantization_config"
SUPPORTED_QUANTIZED_DRAFT_METHODS = (QuantMethod.FP8,)

_DRAFT_SCOPES = (ModelScope.EAGLE_DRAFT, ModelScope.MTP_DRAFT)


def _validate_draft_compressed_tensors(quant: Mapping[str, object]) -> None:
    """Restrict draft compressed-tensors metadata to one uniform Linear group."""

    if str(quant.get("quant_method", "")).lower() != "compressed-tensors":
        raise ValueError("quantized drafts support only compressed-tensors dynamic per-channel FP8")
    groups = quant.get("config_groups")
    if not isinstance(groups, Mapping) or len(groups) != 1:
        raise ValueError("quantized draft compressed-tensors requires exactly one config group")
    group = next(iter(groups.values()))
    if not isinstance(group, Mapping):
        raise ValueError("quantized draft compressed-tensors group must be a JSON object")
    targets = group.get("targets")
    if not isinstance(targets, (list, tuple)) or tuple(targets) != ("Linear",):
        raise ValueError("quantized draft compressed-tensors targets must be exactly ['Linear']")
    weights = group.get("weights")
    activations = group.get("input_activations")
    if not isinstance(weights, Mapping) or (
        weights.get("type") != "float"
        or weights.get("num_bits") != 8
        or weights.get("strategy") != "channel"
        or weights.get("symmetric") is not True
        or weights.get("dynamic", False) is not False
        or weights.get("group_size") is not None
    ):
        raise ValueError(
            "quantized draft weights must be compressed-tensors FP8 with "
            "channel strategy, symmetric=true, and static weights"
        )
    if not isinstance(activations, Mapping) or (
        activations.get("type") != "float"
        or activations.get("num_bits") != 8
        or activations.get("dynamic") is not True
        or activations.get("strategy") != "token"
        or activations.get("symmetric") is not True
        or activations.get("group_size") is not None
    ):
        raise ValueError(
            "quantized draft activations must be compressed-tensors dynamic FP8 "
            "with token strategy and symmetric=true"
        )
    if group.get("output_activations") not in (None, {}):
        raise ValueError("quantized draft output activations are not supported")


def _validate_quantization_object(raw: object, *, field: str) -> bool:
    if raw is None or raw == {}:
        return False
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field} must be a JSON object or null")
    _validate_draft_compressed_tensors(raw)
    return True


@dataclass(frozen=True)
class DraftQuantization:
    """Validated draft-checkpoint format and its configuration source."""

    model_scope: ModelScope
    quantization: QuantConfig
    source: str

    @property
    def is_quantized(self) -> bool:
        return self.quantization.method is not QuantMethod.NONE


def parse_draft_quantization(
    raw_config: Mapping[str, object],
    *,
    expected_scope: ModelScope,
    target_quantization: QuantConfig | None = None,
) -> DraftQuantization:
    """Resolve the draft format using vLLM-compatible ownership rules."""

    if expected_scope not in _DRAFT_SCOPES:
        raise ValueError(f"{expected_scope.value!r} is not a draft model scope")

    if expected_scope is ModelScope.EAGLE_DRAFT:
        # External draft checkpoint: its own standard metadata is authoritative.
        validated_format = _validate_quantization_object(
            raw_config.get("quantization_config"), field="quantization_config"
        )
        quant = detect_quantization(dict(raw_config))
        source = "quantization_config" if quant.method is not QuantMethod.NONE else "dense-default"
    elif MTP_DRAFT_QUANTIZATION_KEY in raw_config:
        # Same-checkpoint MTP may explicitly diverge from the target.  Null or
        # an empty object is an intentional dense override.
        override = raw_config[MTP_DRAFT_QUANTIZATION_KEY]
        if override is None or override == {}:
            quant = QuantConfig(QuantMethod.NONE)
            validated_format = False
        elif isinstance(override, Mapping):
            validated_format = _validate_quantization_object(
                override, field=MTP_DRAFT_QUANTIZATION_KEY
            )
            quant = detect_quantization({"quantization_config": dict(override)})
        else:
            raise ValueError(f"{MTP_DRAFT_QUANTIZATION_KEY} must be a JSON object or null")
        source = MTP_DRAFT_QUANTIZATION_KEY
    else:
        # MTP tensors live in the target checkpoint and follow its packing
        # unless the checkpoint explicitly provides the override above.
        validated_format = _validate_quantization_object(
            raw_config.get("quantization_config"), field="quantization_config"
        )
        quant = target_quantization or detect_quantization(dict(raw_config))
        source = (
            "target-quantization-inherited"
            if quant.method is not QuantMethod.NONE
            else "dense-default"
        )

    if quant.method is QuantMethod.NONE:
        return DraftQuantization(
            model_scope=expected_scope,
            quantization=quant,
            source=source,
        )
    if not validated_format:
        raise ValueError(
            "quantized draft metadata must declare compressed-tensors dynamic per-channel FP8"
        )
    if quant.method not in SUPPORTED_QUANTIZED_DRAFT_METHODS:
        supported = ", ".join(method.value for method in SUPPORTED_QUANTIZED_DRAFT_METHODS)
        raise ValueError(
            f"unsupported quantized draft method {quant.method.value!r}; "
            f"validated methods: {supported}"
        )
    if quant.activation_dynamic is not True:
        raise ValueError("quantized draft FP8 requires dynamic per-token activations")
    if expected_scope is ModelScope.MTP_DRAFT:
        layer = raw_config.get("num_hidden_layers")
        if type(layer) is not int or layer < 0:
            raise ValueError(
                "quantized MTP draft metadata requires a non-negative num_hidden_layers"
            )
        required_dense = (
            f"model.layers.{layer}.mlp.gate",
            f"model.layers.{layer}.self_attn.kv_b_proj",
        )
        uncovered = [
            name
            for name in required_dense
            if not any(_matches_ignore(pattern, name) for pattern in quant.ignored_layers)
        ]
        if uncovered:
            raise ValueError(
                "quantized MTP draft metadata must ignore dense architectural "
                f"projections: {uncovered}"
            )
    return DraftQuantization(
        model_scope=expected_scope,
        quantization=quant,
        source=source,
    )


def load_draft_quantization(
    checkpoint: str | Path,
    *,
    expected_scope: ModelScope,
    target_quantization: QuantConfig | None = None,
) -> DraftQuantization:
    """Read ``config.json`` and parse its independent draft contract."""

    import json

    path = Path(checkpoint) / "config.json"
    if not path.is_file():
        raise ValueError(f"draft checkpoint has no config.json at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("draft config.json must contain a JSON object")
    return parse_draft_quantization(
        raw,
        expected_scope=expected_scope,
        target_quantization=target_quantization,
    )


def _matches_ignore(pattern: str, qualified_name: str) -> bool:
    try:
        if pattern.startswith("re:"):
            return re.fullmatch(pattern[3:], qualified_name) is not None
    except re.error as error:
        raise ValueError(f"invalid draft quantization ignore pattern {pattern!r}") from error
    return qualified_name == pattern or qualified_name.startswith(f"{pattern}.")


def _draft_selection(
    metadata: DraftQuantization,
    checkpoint_quant: QuantConfig,
    context: LinearBuildContext,
) -> LinearSelection:
    if checkpoint_quant != metadata.quantization:
        raise ValueError("draft linear factory received inconsistent quantization")
    if context.model_scope is not metadata.model_scope:
        raise ValueError(
            f"{metadata.model_scope.value} factory cannot construct "
            f"{context.model_scope.value} projection {context.qualified_name!r}"
        )
    if not context.allow_quantization:
        return LinearSelection(
            QuantConfig(QuantMethod.NONE),
            None,
            "draft-architectural-exclusion",
        )
    for pattern in checkpoint_quant.ignored_layers:
        if _matches_ignore(pattern, context.qualified_name):
            return LinearSelection(
                QuantConfig(QuantMethod.NONE),
                None,
                f"draft-checkpoint-ignore:{pattern}",
            )
    # Router logits are an intentionally high-sensitivity dense boundary.
    # Experts, attention, fusion, and the independent draft output head remain
    # eligible.  The metadata therefore cannot silently repack a router tensor.
    if context.role is LinearRole.MOE_ROUTER:
        return LinearSelection(
            QuantConfig(QuantMethod.NONE),
            None,
            "draft-dense-router",
        )
    return LinearSelection(
        checkpoint_quant,
        None,
        f"draft-checkpoint-{checkpoint_quant.method.value}",
    )


def draft_linear_factory(
    metadata: DraftQuantization,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
    capabilities=None,
):
    """Construct only the declared draft scope with its own format policy."""

    return linear_factory(
        metadata.quantization,
        device=device,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        capabilities=capabilities,
        selection_policy=lambda quant, context: _draft_selection(metadata, quant, context),
    )


def pack_dynamic_fp8_draft_state(
    dense_state: Mapping[str, torch.Tensor],
    quantized_head: nn.Module,
) -> dict[str, torch.Tensor]:
    """Derive the exact packed state expected by an FP8 draft module.

    This is an explicit offline conversion boundary.  Serving loaders consume
    only the packed tensors and never dequantize or silently convert a dense
    checkpoint during startup.
    """

    packed = {name: tensor.detach().contiguous() for name, tensor in dense_state.items()}
    fp8_modules = {
        name: module
        for name, module in quantized_head.named_modules()
        if isinstance(module, Fp8Linear)
    }
    if not fp8_modules:
        raise ValueError("quantized draft head contains no FP8 projections")
    for module_name, module in fp8_modules.items():
        if not module.activation_dynamic:
            raise ValueError("draft checkpoint conversion supports dynamic FP8 only")
        prefix = f"{module_name}." if module_name else ""
        weight_key = f"{prefix}weight"
        try:
            dense_weight = dense_state[weight_key]
        except KeyError as error:
            raise KeyError(f"dense draft checkpoint is missing {weight_key!r}") from error
        if dense_weight.ndim != 2:
            raise ValueError(f"dense draft tensor {weight_key!r} must be a matrix")
        weight, scale = quantize_fp8(dense_weight, per_channel=True)
        packed[weight_key] = weight.contiguous()
        packed[f"{prefix}weight_scale"] = scale.contiguous()

    expected = quantized_head.state_dict()
    missing = set(expected) - set(packed)
    unexpected = set(packed) - set(expected)
    if missing or unexpected:
        raise ValueError(
            "converted draft checkpoint does not match the quantized module: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    packed_payloads = {
        f"{module_name}.{member}" if module_name else member
        for module_name in fp8_modules
        for member in ("weight", "weight_scale")
    }
    for name, tensor in expected.items():
        converted = packed[name]
        if converted.shape != tensor.shape:
            raise ValueError(
                f"converted draft tensor {name!r} has shape "
                f"{tuple(converted.shape)}, expected {tuple(tensor.shape)}"
            )
        if name in packed_payloads and converted.dtype != tensor.dtype:
            raise ValueError(
                f"converted draft tensor {name!r} has dtype "
                f"{converted.dtype}, expected {tensor.dtype}"
            )
        if (
            name not in packed_payloads
            and not converted.is_floating_point()
            and converted.dtype != tensor.dtype
        ):
            raise ValueError(
                f"converted draft tensor {name!r} has dtype "
                f"{converted.dtype}, expected {tensor.dtype}"
            )
    return packed


def coerce_draft_state_for_load(
    head: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Validate packed ABIs and apply compute dtype only to dense members."""

    expected = head.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(
            "draft checkpoint state does not match the constructed head: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    quantized_buffers = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): (
            module,
            buffer_name,
            buffer,
        )
        for module_name, module in head.named_modules()
        if getattr(module, "is_quantized", False)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    coerced: dict[str, torch.Tensor] = {}
    for name, current in expected.items():
        tensor = state[name]
        if tensor.shape != current.shape:
            raise ValueError(
                f"draft checkpoint tensor {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(current.shape)}"
            )
        packed_member = quantized_buffers.get(name)
        if packed_member is not None:
            module, buffer_name, packed = packed_member
            if isinstance(module, Fp8Linear) and buffer_name == "weight_scale":
                # compressed-tensors checkpoints commonly serialize the scale
                # in bf16 even though the fused-kernel registration ABI is fp32.
                # This cast is value-preserving; unlike a packed weight cast it
                # cannot reinterpret or silently requantize payload bits.
                if not tensor.is_floating_point():
                    raise TypeError(
                        f"packed draft tensor {name!r} must be floating-point, got {tensor.dtype}"
                    )
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"packed draft tensor {name!r} must be finite")
                if not bool((tensor > 0).all()):
                    raise ValueError(f"packed draft tensor {name!r} must be strictly positive")
                tensor = tensor.to(packed.dtype)
            elif tensor.dtype != packed.dtype:
                raise TypeError(
                    f"packed draft tensor {name!r} has dtype {tensor.dtype}, "
                    f"expected {packed.dtype}"
                )
        elif tensor.is_floating_point():
            tensor = tensor.to(dtype)
        elif tensor.dtype != current.dtype:
            raise TypeError(
                f"draft checkpoint tensor {name!r} has dtype {tensor.dtype}, "
                f"expected {current.dtype}"
            )
        coerced[name] = tensor
    return coerced
