"""Production DeepSeek-V4 native decoder and EP checkpoint loader.

The decoder math and HCA/CSA cache layer implementations come from the pinned
Transformers release.  Kairyu owns placement, the packed-FP4 kernel,
Attention-DP scheduling, checkpoint attestation, and cache lifetime.  No
checkpoint Python module is imported or executed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from kairyu.engine.core.weights import CheckpointReader
from kairyu.kernels.deepseek_v4_moe_gpu import register_deepseek_v4_experts
from kairyu.models.config import ModelConfig
from kairyu.models.deepseek_v4_checkpoint import (
    OFFICIAL_REPOSITORY,
    OFFICIAL_REVISION,
    build_deepseek_v4_checkpoint_plan,
)
from kairyu.models.moe_parallel import ExpertParallelLoadInfo

_ATTENTION_DP_EP_PLAN = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


def _local(tensor: torch.Tensor) -> torch.Tensor:
    local = getattr(tensor, "to_local", None)
    return local() if callable(local) else tensor


def _validate_official_attestation(directory: Path) -> None:
    path = directory / ".kairyu-model-attestation.json"
    if not path.is_file():
        raise RuntimeError(
            "DeepSeek V4 native execution requires the model-volume attestation "
            "created by the pinned example downloader"
        )
    try:
        attestation = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid DeepSeek V4 model attestation at {path}") from error
    expected = (OFFICIAL_REPOSITORY, OFFICIAL_REVISION)
    observed = (attestation.get("repo"), attestation.get("revision"))
    if observed != expected:
        raise RuntimeError(
            "DeepSeek V4 model attestation does not identify the pinned checkpoint: "
            f"got repo/revision={observed!r}, expected={expected!r}"
        )
    if not isinstance(attestation.get("tree_sha256"), str) or not isinstance(
        attestation.get("files"), list
    ):
        raise RuntimeError("DeepSeek V4 model attestation has no verified file inventory")


class DeepseekV4NativeDecoder(nn.Module):
    """Kairyu-facing incremental wrapper over the built-in V4 implementation."""

    def __init__(self, model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.hf_model = model
        self.config = config
        self._attention_dp_layouts: tuple[tuple[int, ...], ...] | None = None
        self._attention_dp_layout_cursor = 0

    @torch.inference_mode()
    def forward_cached(
        self,
        token_ids: torch.Tensor,
        *,
        past_key_values=None,
        position: int = 0,
    ) -> tuple[torch.Tensor, object]:
        if token_ids.ndim != 1 or token_ids.numel() < 1:
            raise ValueError("DeepSeek V4 cached forward requires non-empty token ids")
        if position < 0 or position + token_ids.numel() > self.config.max_position_embeddings:
            raise ValueError("DeepSeek V4 cached forward exceeds its native 1M context")
        if past_key_values is None:
            from kairyu.models.deepseek_v4_cache import make_deepseek_v4_cache

            past_key_values = make_deepseek_v4_cache(self.hf_model.config)
        position_ids = torch.arange(
            position,
            position + token_ids.numel(),
            dtype=torch.long,
            device=token_ids.device,
        ).unsqueeze(0)
        outputs = self.hf_model(
            input_ids=token_ids.unsqueeze(0),
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("DeepSeek V4 did not return its HCA/CSA cache")
        if cache is not past_key_values:
            raise RuntimeError("DeepSeek V4 replaced Kairyu's owned cache object")
        return outputs.logits[0], cache

    def _kairyu_prepare_attention_dp_layouts(self, layouts: object) -> None:
        if self._attention_dp_layouts is not None:
            raise RuntimeError("DeepSeek V4 Attention-DP layouts are already armed")
        if type(layouts) is not tuple or not layouts:
            raise ValueError("DeepSeek V4 Attention-DP layouts must be a non-empty tuple")
        normalized: list[tuple[int, ...]] = []
        for layout in layouts:
            if type(layout) is not tuple or not layout:
                raise ValueError("DeepSeek V4 Attention-DP layout is malformed")
            if any(type(rows) is not int or rows < 1 for rows in layout):
                raise ValueError("DeepSeek V4 Attention-DP rows must be positive integers")
            normalized.append(layout)
        self._attention_dp_layouts = tuple(normalized)
        self._attention_dp_layout_cursor = 0

    def _kairyu_begin_attention_dp_phase(self) -> tuple[int, ...]:
        layouts = self._attention_dp_layouts
        if layouts is None or self._attention_dp_layout_cursor >= len(layouts):
            raise RuntimeError("DeepSeek V4 Attention-DP phase has no matching layout")
        layout = layouts[self._attention_dp_layout_cursor]
        self._attention_dp_layout_cursor += 1
        return layout

    def _kairyu_assert_attention_dp_layouts_consumed(self) -> None:
        layouts = self._attention_dp_layouts
        if layouts is None or self._attention_dp_layout_cursor != len(layouts):
            raise RuntimeError("DeepSeek V4 did not consume every Attention-DP phase layout")
        self._attention_dp_layouts = None
        self._attention_dp_layout_cursor = 0

    def _kairyu_assert_attention_dp_layouts_idle(self) -> None:
        if self._attention_dp_layouts is not None or self._attention_dp_layout_cursor:
            raise RuntimeError("DeepSeek V4 Attention-DP layout queue is not idle")


def _verify_resident_experts(model: nn.Module, *, ep_size: int) -> None:
    expected = 256 // ep_size
    experts = [
        module
        for module in model.modules()
        if hasattr(module, "gate_up_proj")
        and hasattr(module, "gate_up_proj_scale_inv")
        and hasattr(module, "down_proj")
        and hasattr(module, "down_proj_scale_inv")
    ]
    if not experts:
        raise RuntimeError("DeepSeek V4 load produced no routed expert modules")
    ue8m0 = getattr(torch, "float8_e8m0fnu", None)
    for module in experts:
        gate_up = _local(module.gate_up_proj)
        gate_scale = _local(module.gate_up_proj_scale_inv)
        down = _local(module.down_proj)
        down_scale = _local(module.down_proj_scale_inv)
        if gate_up.shape[0] != expected or down.shape[0] != expected:
            raise RuntimeError(
                "DeepSeek V4 resident expert count disagrees with EP placement: "
                f"got gate/down={gate_up.shape[0]}/{down.shape[0]}, expected={expected}"
            )
        if gate_up.dtype is not torch.int8 or down.dtype is not torch.int8:
            raise RuntimeError("DeepSeek V4 expert weights were not retained as packed FP4")
        if ue8m0 is None or gate_scale.dtype is not ue8m0 or down_scale.dtype is not ue8m0:
            raise RuntimeError("DeepSeek V4 expert scales lost their UE8M0 checkpoint ABI")


def load_deepseek_v4_native_ep(
    directory: str | Path,
    raw_config: dict,
    config: ModelConfig,
    *,
    ep_size: int,
    ep_rank: int,
    comm,
    dtype: torch.dtype,
    device: str | torch.device,
) -> tuple[DeepseekV4NativeDecoder, ExpertParallelLoadInfo]:
    """Validate and directly load one Attention-DP/EP rank."""

    directory = Path(directory)
    if config.architecture != "DeepseekV4ForCausalLM" or config.moe is None:
        raise ValueError("DeepSeek V4 native EP loader received a different architecture")
    if ep_size not in {1, 2, 4, 8}:
        raise ValueError("DeepSeek V4 native execution supports EP1/2/4/8")
    if dtype is not torch.bfloat16 or torch.device(device).type != "cuda":
        raise ValueError("DeepSeek V4 native EP requires CUDA BF16 execution")
    if getattr(comm, "rank", None) != ep_rank or getattr(comm, "world_size", None) != ep_size:
        raise ValueError("DeepSeek V4 communicator does not match its EP rank/size")
    _validate_official_attestation(directory)

    reader = CheckpointReader(directory)
    names = reader.names()
    plan = build_deepseek_v4_checkpoint_plan(
        names,
        num_hidden_layers=config.num_hidden_layers,
        num_experts=config.moe.num_experts,
        ep_size=ep_size,
        ep_rank=ep_rank,
        require_official_count=True,
    )
    plan.validate_specs(reader.specs(names))

    implementation = register_deepseek_v4_experts()
    try:
        from torch.distributed.device_mesh import DeviceMesh
        from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
        from transformers.distributed import DistributedConfig
    except ImportError as error:  # pragma: no cover - production image owns dependencies
        raise RuntimeError(
            "DeepSeek V4 native EP requires Transformers 5.12 and torch 2.12"
        ) from error

    hf_config = DeepseekV4Config.from_dict(raw_config)
    hf_config._attn_implementation = "eager"
    # Attention-DP ranks own different requests.  Therefore the checkpoint's
    # optional indexer-TP entries must be removed: only experts are sharded.
    hf_config.base_model_ep_plan = dict(_ATTENTION_DP_EP_PLAN)
    process_group = comm.group if comm.group is not None else dist.group.WORLD
    mesh = DeviceMesh.from_group(
        process_group,
        "cuda",
        mesh=tuple(range(ep_size)),
        mesh_dim_names=("tp",),
    )
    previous_local_rank = os.environ.get("LOCAL_RANK")
    os.environ["LOCAL_RANK"] = str(ep_rank)
    try:
        model = DeepseekV4ForCausalLM.from_pretrained(
            directory,
            config=hf_config,
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="eager",
            experts_implementation=implementation,
            distributed_config=DistributedConfig(enable_expert_parallel=True),
            device_mesh=mesh,
        )
    finally:
        if previous_local_rank is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = previous_local_rank
    model.eval()
    from kairyu.models.deepseek_v4_cache import install_deepseek_v4_fp4_indexer

    install_deepseek_v4_fp4_indexer(model)
    remaining_meta = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if remaining_meta:
        raise RuntimeError(
            "DeepSeek V4 EP load left meta tensors unresolved: "
            f"{remaining_meta[:8]} ({len(remaining_meta)} total)"
        )
    _verify_resident_experts(model, ep_size=ep_size)

    owned = tuple(plan.owned_expert_range)
    loaded_names = tuple(
        item.name
        for item in plan.tensor_plans
        if item.stage != "mtp"
        and (item.expert_index is None or item.expert_index in plan.owned_expert_range)
    )
    info = ExpertParallelLoadInfo(
        ep_rank=ep_rank,
        ep_size=ep_size,
        owned_expert_indices=owned,
        quantization_source=f"{OFFICIAL_REPOSITORY}@{OFFICIAL_REVISION}",
        quantization_method="mixed-fp4-fp8-direct",
        kv_cache_quant_algo="hca-csa-checkpoint-native",
        producer_name="transformers",
        producer_version="5.12.1",
        checkpoint_tensor_count=len(names),
        rank_loaded_tensor_count=len(loaded_names),
        auxiliary_kv_scale_count=0,
    )
    decoder = DeepseekV4NativeDecoder(model, config)
    object.__setattr__(decoder, "_kairyu_ep_load_info", info)
    return decoder, info


__all__ = [
    "DeepseekV4NativeDecoder",
    "load_deepseek_v4_native_ep",
]
