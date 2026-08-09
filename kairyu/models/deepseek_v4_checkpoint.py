"""DeepSeek-V4 official checkpoint placement and mixed-precision ABI.

The 0731 checkpoint stores routed experts as per-expert packed FP4 tensors and
all other quantized projections as block FP8.  Kairyu consumes that layout
directly: this module never converts, dequantizes, or rewrites the shared model
volume.  Every checkpoint name must match one known architecture member before
workers allocate resident tensors.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

OFFICIAL_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
OFFICIAL_TENSOR_COUNT = 72_317
OFFICIAL_CHECKPOINT_BYTES = 166_878_536_440

_GLOBAL_MEMBERS = frozenset(
    {
        "embed.weight",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "head.weight",
        "norm.weight",
    }
)

_LAYER_MEMBERS = frozenset(
    {
        "attn.attn_sink",
        "attn.compressor.ape",
        "attn.compressor.norm.weight",
        "attn.compressor.wgate.weight",
        "attn.compressor.wkv.weight",
        "attn.indexer.compressor.ape",
        "attn.indexer.compressor.norm.weight",
        "attn.indexer.compressor.wgate.weight",
        "attn.indexer.compressor.wkv.weight",
        "attn.indexer.weights_proj.weight",
        "attn.indexer.wq_b.scale",
        "attn.indexer.wq_b.weight",
        "attn.kv_norm.weight",
        "attn.q_norm.weight",
        "attn.wkv.scale",
        "attn.wkv.weight",
        "attn.wo_a.scale",
        "attn.wo_a.weight",
        "attn.wo_b.scale",
        "attn.wo_b.weight",
        "attn.wq_a.scale",
        "attn.wq_a.weight",
        "attn.wq_b.scale",
        "attn.wq_b.weight",
        "attn_norm.weight",
        "ffn.gate.bias",
        "ffn.gate.tid2eid",
        "ffn.gate.weight",
        "ffn.shared_experts.w1.scale",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w2.scale",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w3.scale",
        "ffn.shared_experts.w3.weight",
        "ffn_norm.weight",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    }
)

_MTP_MEMBERS = frozenset(
    {
        "attn.attn_sink",
        "attn.kv_norm.weight",
        "attn.q_norm.weight",
        "attn.wkv.scale",
        "attn.wkv.weight",
        "attn.wo_a.scale",
        "attn.wo_a.weight",
        "attn.wo_b.scale",
        "attn.wo_b.weight",
        "attn.wq_a.scale",
        "attn.wq_a.weight",
        "attn.wq_b.scale",
        "attn.wq_b.weight",
        "attn_norm.weight",
        "confidence_head.proj.weight",
        "ffn.gate.bias",
        "ffn.gate.weight",
        "ffn.shared_experts.w1.scale",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w2.scale",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w3.scale",
        "ffn.shared_experts.w3.weight",
        "ffn_norm.weight",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "main_norm.weight",
        "main_proj.scale",
        "main_proj.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "norm.weight",
    }
)

_CONTAINER_RE = re.compile(r"^(layers|mtp)\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(
    r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\."
    r"(w1|w2|w3)\.(weight|scale)$"
)
_NORMALIZED_EXPERT_RE = re.compile(
    r"^ffn\.experts\.\d+\.(w1|w2|w3)\.(weight|scale)$"
)

_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E8M0": 1,
    "F8_E8M0FNU": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class DeepseekV4Placement(StrEnum):
    REPLICATED = "replicated"
    RANK_LOCAL_EXPERT = "rank-local-expert"


@dataclass(frozen=True)
class DeepseekV4TensorPlan:
    name: str
    placement: DeepseekV4Placement
    stage: str | None = None
    stage_index: int | None = None
    expert_index: int | None = None


@dataclass(frozen=True)
class DeepseekV4CheckpointPlan:
    """One rank's complete direct-load plan for the official tensor namespace."""

    ep_size: int
    ep_rank: int
    num_hidden_layers: int
    num_experts: int
    tensor_plans: tuple[DeepseekV4TensorPlan, ...]
    mtp_layer_count: int

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def owned_expert_range(self) -> range:
        start = self.ep_rank * self.experts_per_rank
        return range(start, start + self.experts_per_rank)

    @property
    def selected_names(self) -> tuple[str, ...]:
        owned = self.owned_expert_range
        return tuple(
            item.name
            for item in self.tensor_plans
            if item.placement is DeepseekV4Placement.REPLICATED
            or item.expert_index in owned
        )

    @property
    def remote_names(self) -> tuple[str, ...]:
        selected = set(self.selected_names)
        return tuple(item.name for item in self.tensor_plans if item.name not in selected)

    def resident_bytes(
        self,
        specs: Mapping[str, tuple[tuple[int, ...], str]],
    ) -> int:
        total = 0
        for name in self.selected_names:
            try:
                shape, dtype = specs[name]
            except KeyError as error:
                raise ValueError(f"checkpoint specs omit selected tensor {name!r}") from error
            try:
                item_size = _DTYPE_BYTES[dtype.upper()]
            except KeyError as error:
                raise ValueError(f"unsupported safetensors dtype {dtype!r} for {name!r}") from error
            elements = 1
            for extent in shape:
                if type(extent) is not int or extent < 0:
                    raise ValueError(f"invalid checkpoint shape {shape!r} for {name!r}")
                elements *= extent
            total += elements * item_size
        return total

    def validate_specs(
        self,
        specs: Mapping[str, tuple[tuple[int, ...], str]],
    ) -> None:
        planned = {item.name for item in self.tensor_plans}
        if set(specs) != planned:
            missing = sorted(planned - set(specs))
            unexpected = sorted(set(specs) - planned)
            raise ValueError(
                "DeepSeek V4 checkpoint header coverage mismatch: "
                f"missing={missing[:8]} ({len(missing)} total), "
                f"unexpected={unexpected[:8]} ({len(unexpected)} total)"
            )
        for item in self.tensor_plans:
            if item.expert_index is None or not item.name.endswith(".weight"):
                continue
            shape, dtype = specs[item.name]
            if len(shape) != 2 or shape[1] < 1:
                raise ValueError(f"packed FP4 expert {item.name!r} must be a non-empty matrix")
            if dtype.upper() not in {"I8", "F4_E2M1", "F4_E2M1FN_X2"}:
                raise ValueError(
                    f"packed FP4 expert {item.name!r} has dtype {dtype!r}, expected I8/F4"
                )
            scale_name = item.name.removesuffix("weight") + "scale"
            scale_shape, scale_dtype = specs[scale_name]
            logical_k = shape[1] * 2
            expected_scale = (shape[0], (logical_k + 31) // 32)
            if scale_shape != expected_scale:
                raise ValueError(
                    f"packed FP4 scale {scale_name!r} shape {scale_shape} "
                    f"!= {expected_scale}"
                )
            if scale_dtype.upper() not in {"F8_E8M0", "F8_E8M0FNU", "U8"}:
                raise ValueError(
                    f"packed FP4 scale {scale_name!r} has dtype {scale_dtype!r}"
                )
        for name, (shape, dtype) in specs.items():
            if not name.endswith(".weight") or dtype.upper() not in {
                "F8_E4M3",
                "F8_E4M3FN",
            }:
                continue
            scale_name = name.removesuffix("weight") + "scale"
            if scale_name not in specs:
                raise ValueError(f"block FP8 weight {name!r} has no companion scale")
            if len(shape) != 2:
                raise ValueError(f"block FP8 weight {name!r} must be a matrix")
            scale_shape, scale_dtype = specs[scale_name]
            expected_scale = ((shape[0] + 127) // 128, (shape[1] + 127) // 128)
            if scale_shape != expected_scale:
                raise ValueError(
                    f"block FP8 scale {scale_name!r} shape {scale_shape} "
                    f"!= {expected_scale}"
                )
            if scale_dtype.upper() not in {"F8_E8M0", "F8_E8M0FNU", "U8"}:
                raise ValueError(f"block FP8 scale {scale_name!r} has dtype {scale_dtype!r}")


def _classify_name(
    name: str,
    *,
    num_hidden_layers: int,
) -> DeepseekV4TensorPlan:
    canonical = name.removeprefix("model.")
    if canonical in _GLOBAL_MEMBERS:
        return DeepseekV4TensorPlan(name, DeepseekV4Placement.REPLICATED)
    match = _CONTAINER_RE.fullmatch(canonical)
    if match is None:
        raise ValueError(f"unknown DeepSeek V4 checkpoint tensor {name!r}")
    stage, index_text, member = match.groups()
    stage_index = int(index_text)
    if stage == "layers" and not 0 <= stage_index < num_hidden_layers:
        raise ValueError(
            f"DeepSeek V4 layer tensor {name!r} is outside [0, {num_hidden_layers})"
        )
    expert = _EXPERT_RE.fullmatch(canonical)
    if expert is not None:
        expert_index = int(expert.group(3))
        return DeepseekV4TensorPlan(
            name,
            DeepseekV4Placement.RANK_LOCAL_EXPERT,
            stage=stage,
            stage_index=stage_index,
            expert_index=expert_index,
        )
    allowed = _LAYER_MEMBERS if stage == "layers" else _MTP_MEMBERS
    if member not in allowed or _NORMALIZED_EXPERT_RE.fullmatch(member):
        raise ValueError(f"unknown DeepSeek V4 {stage} tensor {name!r}")
    return DeepseekV4TensorPlan(
        name,
        DeepseekV4Placement.REPLICATED,
        stage=stage,
        stage_index=stage_index,
    )


def build_deepseek_v4_checkpoint_plan(
    names: tuple[str, ...] | list[str],
    *,
    num_hidden_layers: int,
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    require_official_count: bool = False,
) -> DeepseekV4CheckpointPlan:
    if ep_size not in {1, 2, 4, 8}:
        raise ValueError("DeepSeek V4 expert parallel size must be one of 1, 2, 4, 8")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"DeepSeek V4 EP rank {ep_rank} is outside size {ep_size}")
    if num_experts < 1 or num_experts % ep_size:
        raise ValueError(
            f"DeepSeek V4 expert count {num_experts} must divide across EP{ep_size}"
        )
    if require_official_count and len(names) != OFFICIAL_TENSOR_COUNT:
        raise ValueError(
            f"official DeepSeek V4 revision requires {OFFICIAL_TENSOR_COUNT} tensors, "
            f"found {len(names)}"
        )
    if len(names) != len(set(names)):
        raise ValueError("DeepSeek V4 checkpoint tensor names must be unique")
    plans = tuple(
        _classify_name(name, num_hidden_layers=num_hidden_layers)
        for name in sorted(names)
    )
    expert_plans = tuple(item for item in plans if item.expert_index is not None)
    invalid_experts = sorted(
        {
            item.expert_index
            for item in expert_plans
            if item.expert_index is not None and item.expert_index >= num_experts
        }
    )
    if invalid_experts:
        raise ValueError(
            f"DeepSeek V4 checkpoint contains expert {invalid_experts[0]} "
            f"but config declares {num_experts}"
        )
    mtp_indices = sorted(
        {item.stage_index for item in plans if item.stage == "mtp"}
    )
    if mtp_indices and mtp_indices != list(range(mtp_indices[-1] + 1)):
        raise ValueError(f"DeepSeek V4 MTP layers are not contiguous: {mtp_indices}")
    return DeepseekV4CheckpointPlan(
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
        tensor_plans=plans,
        mtp_layer_count=len(mtp_indices),
    )
