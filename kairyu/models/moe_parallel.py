"""Expert-parallel dispatch/combine over all_to_all (m16 D3).

Routing runs REPLICATED (fp32, deterministic on CPU/gloo; a deploy-day debug
guard hashes topk_indices across ranks — m16 A8). Tokens permute to
expert-owning ranks via ``tensor_all_to_all_single`` (counts exchange first,
then payload), local experts compute, and reverse all_to_all returns the
expert rows. Each rank finalizes only its owned-expert contribution in fp32,
casts that partial to the model dtype, then all-reduces the rank partials.
This preserves the standalone fused-MoE finalize boundary. Contiguous expert
blocks per rank; gloo and NCCL share this code path. DeepEP/UCCL is the
deploy-day fast path behind the same block interface.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from kairyu.models.moe import (
    apply_nvfp4_moe_global_input_scales,
    route_experts,
)
from kairyu.models.parallel import (
    ParallelShardInfo,
    ReplicatedInputRowParallelLinear,
    checkpoint_member_specs,
)
from kairyu.quant.linear import (
    ExpertScope,
    LinearKernel,
    LinearRole,
    ModelScope,
    NvFp4Linear,
    TensorParallelMode,
)


class RemoteExpertOwnershipError(LookupError):
    """A canonical expert exists, but its tensors belong to another rank."""


class _RemoteExpertProjection(nn.Module):
    """Allocation-free construction placeholder removed before execution."""

    def __init__(self, qualified_name: str, owner_rank: int) -> None:
        super().__init__()
        self.qualified_name = qualified_name
        self.owner_rank = owner_rank

    def forward(self, _hidden: torch.Tensor) -> torch.Tensor:
        raise RemoteExpertOwnershipError(
            f"projection {self.qualified_name!r} belongs to expert rank "
            f"{self.owner_rank}"
        )


class _ExpertShardedLinearFactory:
    """Construct rank-owned experts and an optional attention-output K shard."""

    def __init__(
        self,
        base,
        *,
        num_experts: int,
        ep_rank: int,
        ep_size: int,
        attention_output_factory=None,
        replicate_attention_output: bool = False,
    ) -> None:
        if num_experts % ep_size:
            raise ValueError(
                f"{num_experts} experts do not divide across {ep_size} ranks"
            )
        self._base = base
        self._num_experts = num_experts
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._experts_per_rank = num_experts // ep_size
        self._attention_output_factory = attention_output_factory
        self._replicate_attention_output = replicate_attention_output

    def __call__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        qualified_name: str | None = None,
        model_scope: ModelScope = ModelScope.TARGET,
        role: LinearRole | None = None,
        layer_index: int | None = None,
        expert_index: int | None = None,
        expert_scope: ExpertScope = ExpertScope.NONE,
        shard_dim: int | None = None,
        allow_quantization: bool = True,
    ) -> nn.Module:
        if (
            self._replicate_attention_output
            and model_scope is ModelScope.TARGET
            and role is LinearRole.ATTENTION_OUTPUT
            and expert_scope is ExpertScope.NONE
        ):
            if shard_dim != 1:
                raise ValueError(
                    "attention-DP attention output must originate from checkpoint K dim 1"
                )
            return self._base(
                in_features,
                out_features,
                bias,
                qualified_name=qualified_name,
                model_scope=model_scope,
                role=role,
                layer_index=layer_index,
                expert_index=expert_index,
                expert_scope=expert_scope,
                shard_dim=None,
                allow_quantization=allow_quantization,
            )
        if (
            self._attention_output_factory is not None
            and model_scope is ModelScope.TARGET
            and role is LinearRole.ATTENTION_OUTPUT
            and expert_scope is ExpertScope.NONE
        ):
            if shard_dim != 1:
                raise ValueError("EP attention output must shard checkpoint K on dim 1")
            if in_features % self._ep_size:
                raise ValueError(
                    f"attention output K={in_features} does not divide across "
                    f"EP{self._ep_size}"
                )
            return self._attention_output_factory(
                in_features // self._ep_size,
                out_features,
                bias,
                qualified_name=qualified_name,
                model_scope=model_scope,
                role=role,
                layer_index=layer_index,
                expert_index=expert_index,
                expert_scope=expert_scope,
                shard_dim=shard_dim,
                allow_quantization=allow_quantization,
            )
        if expert_scope is ExpertScope.ROUTED:
            if expert_index is None or not 0 <= expert_index < self._num_experts:
                raise ValueError("routed expert construction requires a valid global index")
            owner = expert_index // self._experts_per_rank
            if owner != self._ep_rank:
                if qualified_name is None:
                    raise ValueError("remote expert projection requires a checkpoint name")
                return _RemoteExpertProjection(qualified_name, owner)
        return self._base(
            in_features,
            out_features,
            bias,
            qualified_name=qualified_name,
            model_scope=model_scope,
            role=role,
            layer_index=layer_index,
            expert_index=expert_index,
            expert_scope=expert_scope,
            shard_dim=shard_dim,
            allow_quantization=allow_quantization,
        )


@dataclass(frozen=True)
class ExpertParallelLoadInfo:
    """Fail-closed provenance from one rank-local expert load."""

    ep_rank: int
    ep_size: int
    owned_expert_indices: tuple[int, ...]
    quantization_source: str
    quantization_method: str
    kv_cache_quant_algo: str | None
    producer_name: str | None
    producer_version: str | None
    checkpoint_tensor_count: int
    rank_loaded_tensor_count: int
    auxiliary_kv_scale_count: int


_ATTENTION_DP_DISPATCHER = "nvfp4_allgather_reduce_scatter"
_ATTENTION_DP_EP_SIZE = 4
_ATTENTION_DP_QWEN3_235B_GEOMETRY = {
    "architecture": "Qwen3MoeForCausalLM",
    "hidden_size": 4096,
    "num_hidden_layers": 94,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 12_288,
    "vocab_size": 151_936,
    "dtype": "bfloat16",
    "tie_word_embeddings": False,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 1536,
    "decoder_sparse_step": 1,
    "mlp_only_layers": (),
    "n_shared_experts": 0,
    "first_k_dense_replace": 0,
}


def _normalize_attention_dp_layouts(
    layouts: object,
    *,
    ep_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate one step's rank-row layouts without coercing caller input."""

    if type(layouts) is not tuple or not layouts:
        raise ValueError(
            "attention-DP layouts must be a non-empty tuple in forward-call order"
        )
    normalized: list[tuple[int, ...]] = []
    for forward_index, row_counts in enumerate(layouts):
        if type(row_counts) is not tuple or len(row_counts) != ep_size:
            raise ValueError(
                "attention-DP layout entry "
                f"{forward_index} must be a rank-order tuple of length {ep_size}"
            )
        if any(type(count) is not int or count <= 0 for count in row_counts):
            raise ValueError(
                "attention-DP layout entry "
                f"{forward_index} must contain positive exact integer row counts"
            )
        normalized.append(row_counts)
    return tuple(normalized)


def _assert_collective_device(
    expected: torch.device,
    **tensors: torch.Tensor,
) -> None:
    if any(tensor.device != expected for tensor in tensors.values()):
        details = ", ".join(
            f"{name}={tensor.device}" for name, tensor in tensors.items()
        )
        raise ValueError(
            f"expert-parallel collective device mismatch: expected {expected}; {details}"
        )


def _ep_rank_partial(
    expert_outputs: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    ep_rank: int,
    experts_per_rank: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Finalize one EP rank in slot order, then cross the output dtype boundary."""

    if expert_outputs.ndim != 3:
        raise ValueError("EP expert outputs must have [tokens, top_k, hidden] shape")
    if (
        topk_indices.shape != expert_outputs.shape[:2]
        or topk_weights.shape != expert_outputs.shape[:2]
    ):
        raise ValueError("EP routing tensors must match expert output token/slot axes")
    if ep_rank < 0 or experts_per_rank < 1:
        raise ValueError("EP rank and experts-per-rank must be valid")

    combine_dtype = torch.promote_types(expert_outputs.dtype, topk_weights.dtype)
    owners = topk_indices // experts_per_rank
    partial = torch.zeros(
        expert_outputs.shape[0],
        expert_outputs.shape[2],
        dtype=combine_dtype,
        device=expert_outputs.device,
    )
    # TensorRT-LLM's standalone finalize walks routing slots in order and
    # accumulates only experts owned by this rank in fp32. Keep that order
    # explicit instead of asking a reduction kernel to choose an order.
    for slot in range(expert_outputs.shape[1]):
        selected = owners[:, slot] == ep_rank
        contribution = expert_outputs[:, slot].to(combine_dtype)
        contribution = contribution * topk_weights[:, slot].to(combine_dtype)[:, None]
        partial = partial + torch.where(
            selected[:, None],
            contribution,
            torch.zeros((), dtype=combine_dtype, device=expert_outputs.device),
        )
    return partial.to(output_dtype)


def _swizzle_nvfp4_moe_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert ``[experts, out, in/16]`` scales to FlashInfer's 128x4 layout."""

    if scale.ndim != 3:
        raise ValueError("NVFP4 MoE block scales must have [experts, out, in/16] shape")
    experts, rows, columns = scale.shape
    rows_padded = (rows + 127) // 128 * 128
    columns_padded = (columns + 3) // 4 * 4
    padded = torch.nn.functional.pad(
        scale,
        (0, columns_padded - columns, 0, rows_padded - rows),
    )
    return (
        padded.reshape(
            experts,
            rows_padded // 128,
            4,
            32,
            columns_padded // 4,
            4,
        )
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, rows_padded, columns_padded)
    )


def _clear_nvfp4_linear_runtime_cache(module: NvFp4Linear) -> None:
    for name in ("_weight_native", "_weight_scale_swizzled"):
        module._buffers.pop(name, None)
        module.__dict__.pop(name, None)
    module.__dict__.pop("_nvfp4_native_source_key", None)


@torch.no_grad()
def _pack_nvfp4_moe_block(
    block,
    *,
    require_flashinfer_selection: bool,
):
    """Pack one rank's Qwen experts without retaining a duplicate weight set."""

    from kairyu.kernels.nvfp4_moe_gpu import PreparedNvFp4Moe

    local_experts = tuple(block.local_experts)
    if not local_experts:
        raise RuntimeError("fused NVFP4 MoE requires at least one local expert")

    projections: list[tuple[NvFp4Linear, NvFp4Linear, NvFp4Linear]] = []
    for expert in local_experts:
        candidate = (
            getattr(expert, "gate_proj", None),
            getattr(expert, "up_proj", None),
            getattr(expert, "down_proj", None),
        )
        if not all(isinstance(module, NvFp4Linear) for module in candidate):
            raise RuntimeError(
                "fused NVFP4 MoE requires homogeneous gate/up/down NvFp4Linear experts"
            )
        gate, up, down = candidate
        if any(module.bias is not None for module in candidate):
            raise RuntimeError("fused Qwen NVFP4 MoE does not support expert bias")
        if require_flashinfer_selection:
            for module in candidate:
                selection = getattr(module, "linear_selection", None)
                if getattr(selection, "kernel", None) is not LinearKernel.FLASHINFER_NVFP4:
                    raise RuntimeError(
                        "fused NVFP4 MoE requires every expert projection to "
                        "resolve to FlashInfer NVFP4"
                    )
        projections.append((gate, up, down))

    template_gate, template_up, template_down = projections[0]
    if (
        template_gate.weight.shape != template_up.weight.shape
        or template_gate.weight_scale.shape != template_up.weight_scale.shape
        or template_gate.in_features != template_up.in_features
        or template_gate.out_features != template_up.out_features
        or template_down.in_features != template_up.out_features
        or template_down.out_features != template_up.in_features
    ):
        raise RuntimeError("fused NVFP4 MoE expert projection geometry is incompatible")

    device = template_gate.weight.device
    if require_flashinfer_selection and device.type != "cuda":
        raise RuntimeError("fused NVFP4 MoE requires CUDA-resident checkpoint tensors")
    expected = (
        template_gate.weight.shape,
        template_gate.weight_scale.shape,
        template_down.weight.shape,
        template_down.weight_scale.shape,
    )
    for gate, up, down in projections:
        if any(module.weight.device != device for module in (gate, up, down)):
            raise RuntimeError("fused NVFP4 MoE expert weights span multiple devices")
        actual = (
            gate.weight.shape,
            gate.weight_scale.shape,
            down.weight.shape,
            down.weight_scale.shape,
        )
        if (
            actual != expected
            or up.weight.shape != gate.weight.shape
            or up.weight_scale.shape != gate.weight_scale.shape
        ):
            raise RuntimeError("fused NVFP4 MoE local experts have unequal geometry")
        if not torch.equal(gate.input_scale, up.input_scale):
            raise RuntimeError("fused NVFP4 FC1 gate/up activation scales differ")
        if not torch.equal(gate.weight_scale_2, up.weight_scale_2):
            raise RuntimeError("fused NVFP4 FC1 gate/up dequant scales differ")

    fc1_input_scale = template_up.input_scale.float().reshape(())
    fc2_input_scale = template_down.input_scale.float().reshape(())
    if (
        not bool(torch.isfinite(fc1_input_scale))
        or not bool(fc1_input_scale > 0)
        or not bool(torch.isfinite(fc2_input_scale))
        or not bool(fc2_input_scale > 0)
    ):
        raise RuntimeError("fused NVFP4 MoE activation scales must be finite and positive")
    for _gate, up, down in projections[1:]:
        if (
            not torch.equal(up.input_scale, template_up.input_scale)
            or not torch.equal(down.input_scale, template_down.input_scale)
        ):
            raise RuntimeError("fused NVFP4 MoE requires layer-global activation scales")

    local_count = len(projections)
    intermediate, packed_hidden = template_up.weight.shape
    hidden, packed_intermediate = template_down.weight.shape
    fc1_weight_u8 = torch.empty(
        local_count,
        intermediate * 2,
        packed_hidden,
        dtype=torch.uint8,
        device=device,
    )
    fc2_weight_u8 = torch.empty(
        local_count,
        hidden,
        packed_intermediate,
        dtype=torch.uint8,
        device=device,
    )
    fc1_scale_linear = torch.empty(
        local_count,
        intermediate * 2,
        template_up.weight_scale.shape[1],
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    fc2_scale_linear = torch.empty(
        local_count,
        hidden,
        template_down.weight_scale.shape[1],
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    fc1_dequant = torch.empty(local_count, dtype=torch.float32, device=device)
    fc2_dequant = torch.empty(local_count, dtype=torch.float32, device=device)

    # FlashInfer CUTLASS consumes the gated FC1 pair as [up, gate].
    for local_index, (gate, up, down) in enumerate(projections):
        fc1_weight_u8[local_index, :intermediate].copy_(up.weight)
        fc1_weight_u8[local_index, intermediate:].copy_(gate.weight)
        fc2_weight_u8[local_index].copy_(down.weight)
        fc1_scale_linear[local_index, :intermediate].copy_(up.weight_scale)
        fc1_scale_linear[local_index, intermediate:].copy_(gate.weight_scale)
        fc2_scale_linear[local_index].copy_(down.weight_scale)
        fc1_dequant[local_index] = up.weight_scale_2.float() * fc1_input_scale
        fc2_dequant[local_index] = down.weight_scale_2.float() * fc2_input_scale

    fc1_scale = _swizzle_nvfp4_moe_scale(fc1_scale_linear)
    fc2_scale = _swizzle_nvfp4_moe_scale(fc2_scale_linear)
    del fc1_scale_linear, fc2_scale_linear

    # Rebind canonical checkpoint buffers to slices of the central packed
    # storage. The dataclass and module buffers now share one allocation, so
    # the 235B model never retains a second model-wide weight copy.
    for local_index, (gate, up, down) in enumerate(projections):
        up._buffers["weight"] = fc1_weight_u8[local_index, :intermediate]
        gate._buffers["weight"] = fc1_weight_u8[local_index, intermediate:]
        down._buffers["weight"] = fc2_weight_u8[local_index]
        for module in (gate, up, down):
            _clear_nvfp4_linear_runtime_cache(module)

    return PreparedNvFp4Moe(
        fc1_weight=fc1_weight_u8.view(torch.long),
        fc2_weight=fc2_weight_u8.view(torch.long),
        fc1_act_scale=fc1_input_scale.reciprocal(),
        fc1_weight_scale=fc1_scale.view(torch.int32),
        fc1_dequant_scale=fc1_dequant,
        fc2_act_scale=fc2_input_scale.reciprocal(),
        fc2_weight_scale=fc2_scale.view(torch.int32),
        fc2_dequant_scale=fc2_dequant,
        num_global_experts=len(block.experts),
        local_expert_start=block.owned_expert_indices[0],
    )


class EpMoeBlock(nn.Module):
    """Canonical MoE tree with rank-local experts + all-to-all execution.

    Router, shared expert, and owned routed experts are registered directly at
    their HF paths. ``experts`` keeps the global length and uses ``None`` for
    remote owners, which PyTorch naturally omits from state/named enumeration
    while retaining global expert indices.
    """

    def __init__(
        self,
        block: nn.Module,
        comm,
        ep_rank: int,
        ep_size: int,
        *,
        attention_dp: bool = False,
    ) -> None:
        super().__init__()
        num_experts = len(block.experts)
        if ep_size < 1:
            raise ValueError("expert-parallel size must be positive")
        if not 0 <= ep_rank < ep_size:
            raise ValueError(
                f"expert-parallel rank {ep_rank} is outside size {ep_size}"
            )
        if num_experts % ep_size != 0:
            raise ValueError(f"{num_experts} experts do not divide across {ep_size} ranks")
        if type(attention_dp) is not bool:
            raise TypeError("attention_dp must be a boolean")
        if attention_dp:
            if ep_size != _ATTENTION_DP_EP_SIZE:
                raise ValueError(
                    "Qwen3-235B NVFP4 attention-DP requires expert-parallel size 4"
                )
            if getattr(comm, "rank", None) != ep_rank:
                raise ValueError(
                    "attention-DP communicator rank must equal the expert rank: "
                    f"{getattr(comm, 'rank', None)!r} != {ep_rank}"
                )
            if getattr(comm, "world_size", None) != ep_size:
                raise ValueError(
                    "attention-DP communicator world_size must equal EP size: "
                    f"{getattr(comm, 'world_size', None)!r} != {ep_size}"
                )
            for primitive in ("tensor_all_gather", "tensor_reduce_scatter"):
                if not callable(getattr(comm, primitive, None)):
                    raise TypeError(
                        f"attention-DP communicator must provide {primitive}()"
                    )
        self._comm = comm
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.attention_dp = attention_dp
        self.experts_per_rank = num_experts // ep_size
        start = ep_rank * self.experts_per_rank
        end = (ep_rank + 1) * self.experts_per_rank
        self.owned_expert_indices = tuple(range(start, end))

        # Register the canonical tree directly. Replacing the original block's
        # list too releases non-local tensors even if a caller retains a
        # reference to that pre-parallel block.
        if hasattr(block, "gate"):
            self.gate = block.gate
            route_override = None
        elif hasattr(block, "_route"):
            # Small/custom blocks may own routing logic without parameters.
            # The original expert list is replaced below before this bound
            # method is retained, so it cannot keep remote expert tensors live.
            self.gate = None
            route_override = block._route
        else:
            raise TypeError(
                f"{type(block).__name__} must expose gate or _route for EP"
            )
        experts = nn.ModuleList(
            expert if index in self.owned_expert_indices else None
            for index, expert in enumerate(block.experts)
        )
        self.experts = experts
        block.experts = experts
        self.shared_experts = getattr(block, "shared_experts", None)
        object.__setattr__(self, "_route_override", route_override)

        # Routing metadata is scalar/plain state, never checkpoint state.
        if route_override is None:
            self.top_k = block.top_k
            self.norm_topk_prob = block.norm_topk_prob
            if hasattr(block, "n_group"):
                self.n_group = block.n_group
                self.topk_group = block.topk_group
                self.routed_scaling_factor = block.routed_scaling_factor
        object.__setattr__(
            self,
            "_kairyu_parallel_shard",
            ParallelShardInfo("expert", ep_rank, ep_size),
        )
        object.__setattr__(self, "_prepared_nvfp4_moe", None)
        object.__setattr__(self, "_fused_nvfp4_executed", False)
        object.__setattr__(self, "_attention_dp_layouts", None)
        object.__setattr__(self, "_attention_dp_layout_cursor", 0)
        object.__setattr__(self, "_attention_dp_completed_steps", 0)
        object.__setattr__(self, "_attention_dp_last_layout", None)
        object.__setattr__(self, "_attention_dp_last_local_rows", None)
        object.__setattr__(self, "_attention_dp_last_padded_rows", None)
        object.__setattr__(self, "_attention_dp_last_total_rows", None)
        object.__setattr__(self, "_attention_dp_last_dispatcher", None)
        object.__setattr__(self, "_attention_dp_last_row_layout", None)
        object.__setattr__(self, "_attention_dp_last_activation_collective", None)
        object.__setattr__(self, "_attention_dp_last_combine_collective", None)
        object.__setattr__(self, "_attention_dp_local_partial_workspaces", {})
        object.__setattr__(
            self,
            "_attention_dp_captured_local_partial_workspaces",
            {},
        )
        object.__setattr__(
            self,
            "_attention_dp_capture_resolver",
            None,
        )

    def _invalidate_prepared_nvfp4(self) -> None:
        """Drop derived fused operands before registered tensors can change."""

        object.__setattr__(self, "_prepared_nvfp4_moe", None)
        object.__setattr__(self, "_fused_nvfp4_executed", False)
        workspaces = getattr(self, "_attention_dp_local_partial_workspaces", None)
        if workspaces is not None:
            workspaces.clear()
        captured = getattr(
            self,
            "_attention_dp_captured_local_partial_workspaces",
            None,
        )
        if captured is not None:
            captured.clear()

    def _attention_dp_local_partial(
        self,
        *,
        rows: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Grow-only per-block output scratch for the direct-NCCL stream.

        The fused local expert writes this tensor and reduce-scatter consumes it
        immediately on the same current stream. Keeping the stream handle in the
        key prevents another stream from overwriting a still-live operation.
        Torch-distributed fallback retains the former allocate-per-call path.
        """

        if getattr(self._comm, "attention_dp_direct_nccl_active", False) is not True:
            return torch.empty(rows, hidden_size, dtype=dtype, device=device)
        stream = torch.cuda.current_stream(device)
        stream_handle = getattr(stream, "cuda_stream", None)
        if type(stream_handle) is not int or stream_handle < 0:
            raise RuntimeError(
                "attention-DP direct NCCL workspace needs a current CUDA stream"
            )
        key = (str(device), dtype, hidden_size, stream_handle)
        capture_resolver = self._attention_dp_capture_resolver
        capturing = (
            capture_resolver()
            if callable(capture_resolver)
            else (
                torch.cuda.is_current_stream_capturing()
                if device.type == "cuda"
                else False
            )
        )
        if type(capturing) is not bool:
            raise RuntimeError(
                "attention-DP CUDA graph capture resolver must return a boolean"
            )
        if capturing:
            captured_key = (*key, rows)
            workspaces = self._attention_dp_captured_local_partial_workspaces
            workspace = workspaces.get(captured_key)
            if workspace is None:
                workspace = torch.empty(
                    rows,
                    hidden_size,
                    dtype=dtype,
                    device=device,
                )
                workspaces[captured_key] = workspace
            return workspace
        workspaces = self._attention_dp_local_partial_workspaces
        workspace = workspaces.get(key)
        if workspace is None or workspace.shape[0] < rows:
            workspace = torch.empty(
                rows,
                hidden_size,
                dtype=dtype,
                device=device,
            )
            workspaces[key] = workspace
        return workspace if workspace.shape[0] == rows else workspace[:rows]

    def _apply(self, fn, recurse: bool = True):
        # ``PreparedNvFp4Moe`` contains unregistered views and derived scale
        # tensors.  Module._apply replaces registered buffers independently,
        # so retaining the prepared object across to()/cuda()/dtype changes
        # would leave it pointing at the previous device/storage generation.
        # Invalidate first so even a failed conversion cannot re-enable stale
        # operands; the normal build path prepares only after model.to(device).
        self._invalidate_prepared_nvfp4()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # The packed activation/dequant scales are values derived from child
        # buffers, not aliases.  A parent model's load_state_dict reaches this
        # hook before it copies or assigns those child buffers.
        self._invalidate_prepared_nvfp4()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def local_experts(self) -> tuple[nn.Module, ...]:
        """Compatibility view; modules stay registered by global index."""

        return tuple(
            self.local_expert(index)
            for index in self.owned_expert_indices
        )

    def owner_rank(self, expert_index: int) -> int:
        if not 0 <= expert_index < len(self.experts):
            raise IndexError(
                f"expert index {expert_index} is outside [0, {len(self.experts)})"
            )
        return expert_index // self.experts_per_rank

    def local_expert(self, expert_index: int) -> nn.Module:
        owner = self.owner_rank(expert_index)
        expert = self.experts[expert_index]
        if expert is None:
            raise RemoteExpertOwnershipError(
                f"expert {expert_index} is owned by rank {owner}, "
                f"not rank {self.ep_rank}"
            )
        return expert

    def prepare_fused_nvfp4(self) -> None:
        """Prepare the rank-local FlashInfer CUTLASS fused-MoE operands."""

        # A failed repack must not leave a previous storage generation active.
        self._invalidate_prepared_nvfp4()
        prepared = _pack_nvfp4_moe_block(
            self,
            require_flashinfer_selection=True,
        )
        object.__setattr__(self, "_prepared_nvfp4_moe", prepared)

    def fused_nvfp4_metadata(self) -> dict[str, object] | None:
        """Return measured preparation/execution metadata without tensor values."""

        prepared = self._prepared_nvfp4_moe
        if prepared is None:
            return None
        metadata = {
            "backend": "flashinfer.cutlass_fused_moe",
            "global_expert_count": prepared.num_global_experts,
            "local_expert_count": prepared.fc2_weight.shape[0],
            "local_expert_start": prepared.local_expert_start,
            "weight_order": "up_gate",
            "scale_layout": "128x4",
            "output_dtype": "bfloat16",
            "ep_partial": True,
            "enable_alltoall": False,
            "successful_forward": self._fused_nvfp4_executed,
        }
        if self.attention_dp:
            metadata.update(self.attention_dp_metadata())
        return metadata

    def attention_dp_metadata(self) -> dict[str, object]:
        """Expose the live opt-in layout state without claiming worker support."""

        layouts = self._attention_dp_layouts
        return {
            "attention_dp": self.attention_dp,
            "attention_placement": (
                "request_owned_data_parallel" if self.attention_dp else "replicated"
            ),
            "attention_output_placement": (
                "replicated" if self.attention_dp else "row_parallel"
            ),
            "dispatcher": (
                self._attention_dp_last_dispatcher
                or _ATTENTION_DP_DISPATCHER
                if self.attention_dp
                else None
            ),
            "row_layout": (
                self._attention_dp_last_row_layout
                or "rank_major_zero_padded"
                if self.attention_dp
                else "replicated"
            ),
            "activation_collective": (
                self._attention_dp_last_activation_collective
                or "tensor_all_gather_many"
                if self.attention_dp
                else None
            ),
            "activation_collective_dtype": (
                "nvfp4_packed_uint8" if self.attention_dp else None
            ),
            "scale_collective_dtype": "uint8" if self.attention_dp else None,
            "combine_collective": (
                self._attention_dp_last_combine_collective
                or "tensor_reduce_scatter"
                if self.attention_dp
                else None
            ),
            "post_combine_all_reduce": False if self.attention_dp else None,
            "layout_queue_depth": len(layouts) if layouts is not None else 0,
            "layout_queue_consumed": self._attention_dp_layout_cursor,
            "completed_layout_steps": self._attention_dp_completed_steps,
            "last_rank_row_layout": self._attention_dp_last_layout,
            "last_local_real_rows": self._attention_dp_last_local_rows,
            "last_padded_rows_per_rank": self._attention_dp_last_padded_rows,
            "total_collective_rows": self._attention_dp_last_total_rows,
        }

    def _check_attention_dp_layout_prepare(
        self,
        layouts: tuple[tuple[int, ...], ...],
    ) -> None:
        if not self.attention_dp:
            raise RuntimeError("attention-DP layouts require an attention-DP EP block")
        if self._attention_dp_layouts is not None:
            raise RuntimeError(
                "attention-DP layout queue is already active; assert complete "
                "consumption before preparing the next step"
            )

    def _prepare_attention_dp_layouts(
        self,
        layouts: tuple[tuple[int, ...], ...],
    ) -> None:
        object.__setattr__(self, "_attention_dp_layouts", layouts)
        object.__setattr__(self, "_attention_dp_layout_cursor", 0)

    def _check_attention_dp_layouts_consumed(self) -> None:
        layouts = self._attention_dp_layouts
        if layouts is None:
            raise RuntimeError("attention-DP layout queue was not prepared")
        consumed = self._attention_dp_layout_cursor
        if consumed != len(layouts):
            raise RuntimeError(
                "attention-DP layout queue was not fully consumed: "
                f"{consumed}/{len(layouts)} forward layouts"
            )

    def _finish_attention_dp_layout_step(self) -> None:
        object.__setattr__(self, "_attention_dp_layouts", None)
        object.__setattr__(self, "_attention_dp_layout_cursor", 0)
        object.__setattr__(
            self,
            "_attention_dp_completed_steps",
            self._attention_dp_completed_steps + 1,
        )

    def _current_attention_dp_layout(self, hidden: torch.Tensor) -> tuple[int, ...]:
        layouts = self._attention_dp_layouts
        if layouts is None:
            raise RuntimeError(
                "attention-DP forward requires prepare_attention_dp_layouts()"
            )
        cursor = self._attention_dp_layout_cursor
        if cursor >= len(layouts):
            raise RuntimeError(
                "attention-DP forward consumed more layouts than the prepared step"
            )
        if hidden.ndim != 2 or hidden.shape[1] <= 0:
            raise ValueError(
                "attention-DP hidden must have non-empty [local_rows, hidden] shape"
            )
        layout = layouts[cursor]
        expected_rows = layout[self.ep_rank]
        if hidden.shape[0] != expected_rows:
            raise ValueError(
                "attention-DP local row layout drift at forward "
                f"{cursor}: rank {self.ep_rank} hidden rows {hidden.shape[0]} "
                f"!= declared {expected_rows}"
            )
        return layout

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.attention_dp:
            return self._forward_attention_dp(hidden)

        device = hidden.device
        topk_indices, topk_weights = self._route(hidden)
        if self._prepared_nvfp4_moe is not None:
            from kairyu.kernels.nvfp4_moe_gpu import fused_moe_forward

            object.__setattr__(self, "_fused_nvfp4_executed", False)
            fused_hidden = hidden.contiguous()
            selected_experts = topk_indices.to(torch.int32).contiguous()
            routing_weights = topk_weights.to(torch.float32).contiguous()
            local_partial = torch.empty_like(fused_hidden)
            fused_moe_forward(
                fused_hidden,
                selected_experts,
                routing_weights,
                self._prepared_nvfp4_moe,
                output=local_partial,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
            )
            _assert_collective_device(device, local_partial=local_partial)
            out = self._comm.tensor_all_reduce(local_partial.contiguous())
            if self.shared_experts is not None:
                out = out + self.shared_experts(hidden)
            object.__setattr__(self, "_fused_nvfp4_executed", True)
            return out

        tokens, k = topk_indices.shape
        flat_expert = topk_indices.reshape(-1)  # [tokens*k]
        owner = flat_expert // self.experts_per_rank
        order = torch.argsort(owner, stable=True)
        send_counts = torch.bincount(owner, minlength=self.ep_size)
        payload = hidden.repeat_interleave(k, dim=0)[order]

        recv_counts = torch.empty(
            self.ep_size, dtype=send_counts.dtype, device=device
        )
        _assert_collective_device(
            device, send_counts=send_counts, recv_counts=recv_counts
        )
        self._comm.tensor_all_to_all_single(
            recv_counts, send_counts.contiguous(), [1] * self.ep_size, [1] * self.ep_size
        )
        recv_total = int(recv_counts.sum().item())
        received = torch.empty(
            recv_total, hidden.shape[-1], dtype=hidden.dtype, device=device
        )
        _assert_collective_device(device, payload=payload, received=received)
        self._comm.tensor_all_to_all_single(
            received,
            payload.contiguous(),
            recv_counts.tolist(),
            send_counts.tolist(),
        )
        # which local expert each received row wants: exchange expert ids too
        expert_ids_out = flat_expert[order].to(torch.int64)
        expert_ids_in = torch.empty(recv_total, dtype=torch.int64, device=device)
        _assert_collective_device(
            device, expert_ids_out=expert_ids_out, expert_ids_in=expert_ids_in
        )
        self._comm.tensor_all_to_all_single(
            expert_ids_in,
            expert_ids_out.contiguous(),
            recv_counts.tolist(),
            send_counts.tolist(),
        )
        computed = torch.zeros_like(received)
        for global_index in expert_ids_in.unique():
            expert_index = int(global_index)
            mask = expert_ids_in == global_index
            computed[mask] = self.local_expert(expert_index)(received[mask])

        returned = torch.empty(
            tokens * k, hidden.shape[-1], dtype=hidden.dtype, device=device
        )
        _assert_collective_device(device, computed=computed, returned=returned)
        self._comm.tensor_all_to_all_single(
            returned,
            computed.contiguous(),
            send_counts.tolist(),
            recv_counts.tolist(),
        )
        # Undo the permutation. Each source rank now has the same complete set
        # of returned expert rows, but contributes only the experts owned by
        # its rank. The BF16 boundary before the all-reduce matches standalone
        # fused-MoE finalization and is observably different from one global
        # FP32 sum followed by a single cast.
        unsorted = torch.empty_like(returned)
        unsorted[order] = returned
        local_partial = _ep_rank_partial(
            unsorted.reshape(tokens, k, -1),
            topk_indices,
            topk_weights,
            ep_rank=self.ep_rank,
            experts_per_rank=self.experts_per_rank,
            output_dtype=hidden.dtype,
        )
        _assert_collective_device(device, local_partial=local_partial)
        out = self._comm.tensor_all_reduce(local_partial.contiguous())
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden)
        return out

    def _forward_attention_dp(self, hidden: torch.Tensor) -> torch.Tensor:
        """Dispatch request-owned rows through the opt-in NVFP4 EP4 path."""

        from kairyu.kernels.nvfp4_moe_gpu import (
            fused_moe_forward_quantized,
            interleave_moe_input_scales,
            quantize_moe_input,
        )

        prepared = self._prepared_nvfp4_moe
        if prepared is None:
            raise RuntimeError(
                "attention-DP requires prepared fused NVFP4 local experts"
            )
        object.__setattr__(self, "_fused_nvfp4_executed", False)
        layout = self._current_attention_dp_layout(hidden)
        if hidden.dtype is not torch.bfloat16:
            raise TypeError(
                "Qwen3-235B NVFP4 attention-DP requires BF16 hidden rows"
            )
        local_rows, hidden_size = hidden.shape
        padded_rows = max(layout)
        ragged_direct = (
            getattr(self._comm, "attention_dp_direct_nccl_active", False) is True
            and len(set(layout)) > 1
        )
        padding = 0 if ragged_direct else padded_rows - local_rows
        padded_hidden = (
            torch.nn.functional.pad(hidden, (0, 0, 0, padding))
            if padding
            else hidden.contiguous()
        )

        # Routing stays rank-local. Dummy rows carry a zero routing weight so
        # their fixed slots cannot contribute even if a packed-zero kernel
        # implementation changes its exact zero-point behavior.
        topk_indices, topk_weights = self._route(hidden)
        selected_local = topk_indices.to(torch.int32).contiguous()
        routing_local = topk_weights.to(torch.float32).contiguous()
        if padding:
            selected_local = torch.nn.functional.pad(
                selected_local,
                (0, 0, 0, padding),
                value=0,
            )
            routing_local = torch.nn.functional.pad(
                routing_local,
                (0, 0, 0, padding),
                value=0.0,
            )

        packed_local, scale_rows_local = quantize_moe_input(
            padded_hidden.contiguous(),
            prepared.fc1_act_scale,
        )
        gather_inputs = (
            packed_local.contiguous(),
            scale_rows_local.contiguous(),
            selected_local.contiguous(),
            routing_local.contiguous(),
        )
        gather_many = getattr(
            self._comm,
            (
                "tensor_all_gatherv_many"
                if ragged_direct
                else "tensor_all_gather_many"
            ),
            None,
        )
        if ragged_direct and not callable(gather_many):
            raise RuntimeError(
                "active direct NCCL attention-DP transport lacks grouped gatherv"
            )
        if callable(gather_many):
            gathered_tensors = (
                gather_many(gather_inputs, layout)
                if ragged_direct
                else gather_many(gather_inputs)
            )
            if (
                not isinstance(gathered_tensors, tuple)
                or len(gathered_tensors) != len(gather_inputs)
                or any(
                    not isinstance(tensor, torch.Tensor)
                    for tensor in gathered_tensors
                )
            ):
                raise RuntimeError(
                    "attention-DP grouped all-gather returned a malformed result"
                )
        else:
            gathered_tensors = tuple(
                self._comm.tensor_all_gather(tensor)
                for tensor in gather_inputs
            )
        (
            packed_global,
            scale_rows_global,
            selected_global,
            routing_global,
        ) = gathered_tensors
        global_rows = sum(layout) if ragged_direct else padded_rows * self.ep_size
        expected_shapes = {
            "packed_global": (global_rows, hidden_size // 2),
            "scale_rows_global": (global_rows, hidden_size // 16),
            "selected_global": (global_rows, selected_local.shape[1]),
            "routing_global": (global_rows, routing_local.shape[1]),
        }
        gathered = {
            "packed_global": packed_global,
            "scale_rows_global": scale_rows_global,
            "selected_global": selected_global,
            "routing_global": routing_global,
        }
        for name, tensor in gathered.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise RuntimeError(
                    f"attention-DP {name} shape {tuple(tensor.shape)} != "
                    f"fixed rank-major layout {expected_shapes[name]}"
                )
        device = hidden.device
        _assert_collective_device(device, **gathered)
        input_sf = interleave_moe_input_scales(scale_rows_global.contiguous())

        local_partial = self._attention_dp_local_partial(
            rows=global_rows,
            hidden_size=hidden_size,
            dtype=hidden.dtype,
            device=device,
        )
        fused_moe_forward_quantized(
            packed_global.contiguous(),
            input_sf,
            selected_global.to(torch.int32).contiguous(),
            routing_global.to(torch.float32).contiguous(),
            prepared,
            output=local_partial,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
        )
        _assert_collective_device(device, local_partial=local_partial)
        reduced_slot = (
            self._comm.tensor_reduce_scatterv(local_partial.contiguous(), layout)
            if ragged_direct
            else self._comm.tensor_reduce_scatter(local_partial.contiguous())
        )
        expected_local_rows = local_rows if ragged_direct else padded_rows
        if tuple(reduced_slot.shape) != (expected_local_rows, hidden_size):
            raise RuntimeError(
                "attention-DP reduce-scatter returned shape "
                f"{tuple(reduced_slot.shape)} != "
                f"{(expected_local_rows, hidden_size)}"
            )
        _assert_collective_device(device, reduced_slot=reduced_slot)
        out = reduced_slot if ragged_direct else reduced_slot[:local_rows]
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden)

        object.__setattr__(self, "_attention_dp_last_layout", layout)
        object.__setattr__(self, "_attention_dp_last_local_rows", local_rows)
        object.__setattr__(
            self,
            "_attention_dp_last_padded_rows",
            None if ragged_direct else padded_rows,
        )
        object.__setattr__(self, "_attention_dp_last_total_rows", global_rows)
        object.__setattr__(
            self,
            "_attention_dp_last_dispatcher",
            (
                "nvfp4_allgatherv_reduce_scatterv"
                if ragged_direct
                else _ATTENTION_DP_DISPATCHER
            ),
        )
        object.__setattr__(
            self,
            "_attention_dp_last_row_layout",
            "rank_major_ragged" if ragged_direct else "rank_major_zero_padded",
        )
        object.__setattr__(
            self,
            "_attention_dp_last_activation_collective",
            (
                "tensor_all_gatherv_many"
                if ragged_direct
                else "tensor_all_gather_many"
            ),
        )
        object.__setattr__(
            self,
            "_attention_dp_last_combine_collective",
            (
                "tensor_reduce_scatterv"
                if ragged_direct
                else "tensor_reduce_scatter"
            ),
        )
        object.__setattr__(
            self,
            "_attention_dp_layout_cursor",
            self._attention_dp_layout_cursor + 1,
        )
        object.__setattr__(self, "_fused_nvfp4_executed", True)
        return out

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._route_override is not None:
            return self._route_override(hidden)
        return route_experts(self, hidden)


def _model_ep_blocks(model: nn.Module) -> tuple[EpMoeBlock, ...]:
    cached = getattr(model, "_kairyu_attention_dp_blocks", None)
    if type(cached) is tuple and cached and all(
        isinstance(block, EpMoeBlock) for block in cached
    ):
        return cached
    return tuple(
        module for module in model.modules() if isinstance(module, EpMoeBlock)
    )


def prepare_attention_dp_layouts(
    model: nn.Module,
    layouts: object,
) -> None:
    """Arm every sparse layer with one immutable per-forward row-layout queue."""

    blocks = _model_ep_blocks(model)
    if not blocks:
        raise ValueError("attention-DP model has no EpMoeBlock layers")
    ep_size = blocks[0].ep_size
    if any(not block.attention_dp or block.ep_size != ep_size for block in blocks):
        raise RuntimeError("model has mixed or inconsistent attention-DP EP blocks")
    normalized = _normalize_attention_dp_layouts(layouts, ep_size=ep_size)
    for block in blocks:
        block._check_attention_dp_layout_prepare(normalized)
    for block in blocks:
        block._prepare_attention_dp_layouts(normalized)


def assert_attention_dp_layouts_consumed(model: nn.Module) -> None:
    """Fail a step with missing model forwards, then release its layout queue."""

    blocks = _model_ep_blocks(model)
    if not blocks:
        raise ValueError("attention-DP model has no EpMoeBlock layers")
    for block in blocks:
        block._check_attention_dp_layouts_consumed()
    for block in blocks:
        block._finish_attention_dp_layout_step()


def assert_attention_dp_layouts_idle(model: nn.Module) -> None:
    """Prove graph replay did not enter Python or arm a layout queue.

    CUDA graph replay launches previously captured kernels and therefore must
    consume zero Python-side MoE layouts.  This explicit check distinguishes a
    clean replay from accidentally leaving a prior step's queue armed.
    """

    blocks = _model_ep_blocks(model)
    if not blocks:
        raise ValueError("attention-DP model has no EpMoeBlock layers")
    for block in blocks:
        if (
            block._attention_dp_layouts is not None
            or block._attention_dp_layout_cursor != 0
        ):
            raise RuntimeError(
                "attention-DP graph replay requires an idle layout queue"
            )


def _checkpoint_shard_coordinates(
    model: nn.Module,
    state_name: str,
    shard_dim: int | None,
) -> tuple[int, int]:
    """Return typed rank/world ownership for one physical checkpoint member."""

    if shard_dim is None:
        return 0, 1
    module_name, separator, _member = state_name.rpartition(".")
    if not separator:
        raise RuntimeError(f"sharded checkpoint member {state_name!r} has no module path")
    module = model.get_submodule(module_name)
    context = getattr(module, "linear_context", None)
    if context is None:
        raise RuntimeError(
            f"sharded checkpoint member {state_name!r} has no linear context"
        )
    placement = context.tensor_parallel
    if placement.mode is TensorParallelMode.REPLICATED:
        raise RuntimeError(
            f"replicated projection {module_name!r} requested a checkpoint shard"
        )
    return placement.rank, placement.world_size


def _ep_checkpoint_shapes(
    model: nn.Module,
    *,
    tie_word_embeddings: bool,
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    tuple[tuple[str, EpMoeBlock], ...],
]:
    """Return rank-local and reconstructed full-checkpoint shape contracts."""

    local = {
        name: tuple(tensor.shape)
        for name, tensor in model.state_dict().items()
    }
    if tie_word_embeddings:
        local.pop("lm_head.weight", None)
    member_specs = checkpoint_member_specs(model)
    expected: dict[str, tuple[int, ...]] = {}
    for name, shape in local.items():
        member = member_specs[name]
        rank, world_size = _checkpoint_shard_coordinates(
            model,
            name,
            member.shard_dim,
        )
        if not 0 <= rank < world_size:
            raise RuntimeError(
                f"checkpoint member {name!r} has invalid shard {rank}/{world_size}"
            )
        full_shape = list(shape)
        if member.shard_dim is not None:
            if not 0 <= member.shard_dim < len(full_shape):
                raise RuntimeError(
                    f"checkpoint member {name!r} has invalid shard dim "
                    f"{member.shard_dim} for shape {shape}"
                )
            full_shape[member.shard_dim] *= world_size
        if member.source_name in expected:
            raise RuntimeError(
                f"duplicate checkpoint source {member.source_name!r} in EP model"
            )
        expected[member.source_name] = tuple(full_shape)
    blocks = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, EpMoeBlock)
    )
    for block_name, block in blocks:
        template_index = block.owned_expert_indices[0]
        template_prefix = f"{block_name}.experts.{template_index}."
        template = {
            name.removeprefix(template_prefix): shape
            for name, shape in local.items()
            if name.startswith(template_prefix)
        }
        if not template:
            raise RuntimeError(
                f"owned expert {template_index} at {block_name!r} has no checkpoint state"
            )
        for expert_index in range(len(block.experts)):
            for suffix, shape in template.items():
                expected[
                    f"{block_name}.experts.{expert_index}.{suffix}"
                ] = shape
    return local, expected, blocks


def _kv_scale_names(model: nn.Module) -> tuple[str, ...]:
    names = []
    for layer_index, _layer in enumerate(model.model.layers):
        names.extend(
            (
                f"model.layers.{layer_index}.self_attn.k_proj.k_scale",
                f"model.layers.{layer_index}.self_attn.v_proj.v_scale",
            )
        )
    return tuple(names)


_NVFP4_SAFETENSORS_DTYPES = {
    torch.uint8: "U8",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float32: "F32",
}


def _global_nvfp4_dtype_contract(
    model: nn.Module,
    blocks: tuple[tuple[str, EpMoeBlock], ...],
) -> dict[str, str]:
    """Reconstruct the packed ABI for owned and remote expert projections."""

    local = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): (
            _NVFP4_SAFETENSORS_DTYPES[buffer.dtype]
        )
        for module_name, module in model.named_modules()
        if isinstance(module, NvFp4Linear)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    expected = dict(local)
    for block_name, block in blocks:
        template_index = block.owned_expert_indices[0]
        template_prefix = f"{block_name}.experts.{template_index}."
        template = {
            name.removeprefix(template_prefix): dtype
            for name, dtype in local.items()
            if name.startswith(template_prefix)
        }
        if not template:
            raise RuntimeError(
                f"owned NVFP4 expert {template_index} at {block_name!r} "
                "has no packed checkpoint members"
            )
        for expert_index in range(len(block.experts)):
            for suffix, dtype in template.items():
                expected[f"{block_name}.experts.{expert_index}.{suffix}"] = dtype
    return expected


def _checkpoint_nvfp4_moe_global_input_scales(
    reader,
    blocks: tuple[tuple[str, EpMoeBlock], ...],
) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Read layer-global FC1/FC2 activation scales, including remote experts."""

    requested: dict[str, tuple[str, str]] = {}
    expected_counts: dict[tuple[str, str], int] = {}
    for block_name, block in blocks:
        template = block.local_expert(block.owned_expert_indices[0])
        w13_projections = tuple(
            projection
            for projection in ("gate_proj", "up_proj")
            if isinstance(getattr(template, projection, None), NvFp4Linear)
        )
        w2_projections = (
            ("down_proj",)
            if isinstance(getattr(template, "down_proj", None), NvFp4Linear)
            else ()
        )
        expected_counts[(block_name, "w13")] = (
            len(block.experts) * len(w13_projections)
        )
        expected_counts[(block_name, "w2")] = (
            len(block.experts) * len(w2_projections)
        )
        for expert_index in range(len(block.experts)):
            prefix = f"{block_name}.experts.{expert_index}"
            for projection in w13_projections:
                requested[f"{prefix}.{projection}.input_scale"] = (
                    block_name,
                    "w13",
                )
            for projection in w2_projections:
                requested[f"{prefix}.{projection}.input_scale"] = (
                    block_name,
                    "w2",
                )

    grouped: dict[tuple[str, str], list[torch.Tensor]] = {}
    for name, scale in reader.selected_items(requested):
        value = scale.detach().to(dtype=torch.float32).reshape(-1)
        if (
            value.numel() != 1
            or not bool(torch.isfinite(value).all())
            or not bool((value > 0).all())
        ):
            raise ValueError(
                f"checkpoint NVFP4 MoE input scale {name!r} must be one "
                "finite positive FP32 value"
            )
        grouped.setdefault(requested[name], []).append(value.reshape(()))

    result: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
    for block_name, _block in blocks:
        w13 = grouped.get((block_name, "w13"), [])
        w2 = grouped.get((block_name, "w2"), [])
        if (
            len(w13) != expected_counts[(block_name, "w13")]
            or len(w2) != expected_counts[(block_name, "w2")]
        ):
            raise RuntimeError(
                f"checkpoint NVFP4 MoE scales for {block_name!r} are incomplete"
            )
        result[block_name] = (
            torch.stack(w13).max() if w13 else None,
            torch.stack(w2).max() if w2 else None,
        )
    return result


def _validate_ep_checkpoint_contract(
    reader,
    model: nn.Module,
    *,
    tie_word_embeddings: bool,
    kv_cache_quant_algo: str | None,
) -> tuple[
    dict[str, tuple[int, ...]],
    tuple[tuple[str, EpMoeBlock], ...],
    tuple[str, ...],
]:
    local, expected, blocks = _ep_checkpoint_shapes(
        model,
        tie_word_embeddings=tie_word_embeddings,
    )
    kv_scale_names = (
        _kv_scale_names(model)
        if kv_cache_quant_algo == "FP8"
        else ()
    )
    required = set(expected) | set(kv_scale_names)
    actual = set(reader.names())
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            "expert-parallel checkpoint tensor layout mismatch: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)"
        )

    actual_specs = reader.specs(expected)
    mismatches = [
        (name, expected[name], actual_specs[name][0])
        for name in expected
        if actual_specs[name][0] != expected[name]
    ]
    if mismatches:
        name, wanted, found = mismatches[0]
        raise ValueError(
            f"checkpoint tensor {name!r} shape {found} != expected {wanted}; "
            f"{len(mismatches)} shape mismatch(es)"
        )
    expected_dtypes = _global_nvfp4_dtype_contract(model, blocks)
    dtype_mismatches = [
        (name, expected_dtype, actual_specs[name][1])
        for name, expected_dtype in expected_dtypes.items()
        if actual_specs[name][1] != expected_dtype
    ]
    if dtype_mismatches:
        name, wanted, found = dtype_mismatches[0]
        raise ValueError(
            f"NVFP4 checkpoint tensor {name!r} dtype {found} != "
            f"required ABI dtype {wanted}; "
            f"{len(dtype_mismatches)} dtype mismatch(es)"
        )
    for name, scale in reader.selected_items(kv_scale_names):
        if (
            scale.numel() != 1
            or not scale.is_floating_point()
            or not torch.isfinite(scale).all()
            or not bool((scale > 0).all())
        ):
            raise ValueError(
                f"checkpoint FP8-KV calibration tensor {name!r} must be one "
                "finite positive floating-point value"
            )
    return local, blocks, kv_scale_names


def _assign_checkpoint_tensor(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
) -> None:
    module_name, separator, member = name.rpartition(".")
    module = model.get_submodule(module_name) if separator else model
    if member in module._parameters:
        current = module._parameters[member]
        if current is None:
            raise RuntimeError(f"checkpoint parameter {name!r} is disabled")
        module._parameters[member] = nn.Parameter(
            tensor,
            requires_grad=current.requires_grad,
        )
        return
    if member in module._buffers:
        module._buffers[member] = tensor
        return
    raise KeyError(f"checkpoint member {name!r} is not registered on the model")


def _validate_attention_dp_build_envelope(
    config,
    quant,
    *,
    ep_size: int,
    ep_rank: int,
    comm,
    dtype: torch.dtype,
    device: str | torch.device,
) -> None:
    """Pin the first attention-DP implementation to its measured formal cell."""

    moe = config.moe
    actual = {
        "architecture": config.architecture,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "dtype": config.dtype,
        "tie_word_embeddings": config.tie_word_embeddings,
        "num_experts": getattr(moe, "num_experts", None),
        "num_experts_per_tok": getattr(moe, "num_experts_per_tok", None),
        "moe_intermediate_size": getattr(moe, "moe_intermediate_size", None),
        "decoder_sparse_step": getattr(moe, "decoder_sparse_step", None),
        "mlp_only_layers": getattr(moe, "mlp_only_layers", None),
        "n_shared_experts": getattr(moe, "n_shared_experts", None),
        "first_k_dense_replace": getattr(moe, "first_k_dense_replace", None),
    }
    mismatches = [
        (name, expected, actual[name])
        for name, expected in _ATTENTION_DP_QWEN3_235B_GEOMETRY.items()
        if actual[name] != expected
    ]
    if mismatches:
        name, expected, found = mismatches[0]
        raise ValueError(
            "attention-DP supports only the pinned Qwen3-235B geometry: "
            f"{name}={found!r}, expected {expected!r} "
            f"({len(mismatches)} mismatch(es))"
        )
    if ep_size != _ATTENTION_DP_EP_SIZE:
        raise ValueError("Qwen3-235B attention-DP requires EP4")
    if getattr(quant.method, "value", None) != "nvfp4" or quant.group_size != 16:
        raise ValueError(
            "Qwen3-235B attention-DP requires ModelOpt NVFP4 group_size=16"
        )
    if dtype is not torch.bfloat16:
        raise ValueError("Qwen3-235B attention-DP requires runtime dtype bfloat16")
    if torch.device(device).type != "cuda":
        raise ValueError("Qwen3-235B attention-DP requires a CUDA device")
    if getattr(comm, "rank", None) != ep_rank:
        raise ValueError("attention-DP communicator rank must equal ep_rank")
    if getattr(comm, "world_size", None) != ep_size:
        raise ValueError("attention-DP communicator world_size must equal ep_size")


def build_ep_model(
    model_dir: str | Path,
    ep_size: int,
    ep_rank: int,
    comm,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    attention_backend=None,
    linear_capabilities=None,
    linear_selection_policy=None,
    attention_dp: bool = False,
) -> tuple[nn.Module, object, ExpertParallelLoadInfo]:
    """Build one real Qwen3-MoE rank without materializing remote experts.

    QKV projection, attention/KV state, router, embeddings, and the output head
    are replicated. The default correctness mode keeps its NVFP4 attention
    ``o_proj`` K shard and all-reduce unchanged. ``attention_dp=True`` is a
    fail-closed Qwen3-235B EP4 opt-in: ``o_proj`` stays fully replicated and
    sparse layers consume request-owned row-layout queues. Routed expert
    modules retain their global HF names, but only this rank's contiguous block
    is constructed and loaded. The complete checkpoint name/shape contract is
    validated on every rank before any request can execute.
    """

    from kairyu.engine.core.quant_config import (
        QuantMethod,
        load_checkpoint_quantization,
        validate_model_quantization,
    )
    from kairyu.engine.core.weights import CheckpointReader
    from kairyu.models.config import parse_model_config
    from kairyu.models.layers import RotaryEmbedding
    from kairyu.models.llama import DenseDecoder
    from kairyu.quant.linear import (
        QuantizedLinearBase,
        linear_factory,
    )

    directory = Path(model_dir)
    raw = json.loads((directory / "config.json").read_text())
    config = parse_model_config(raw)
    if config.architecture != "Qwen3MoeForCausalLM":
        raise ValueError(
            "expert-parallel real-checkpoint loading currently supports "
            f"Qwen3MoeForCausalLM, got {config.architecture!r}"
        )
    if config.moe is None:
        raise ValueError("expert-parallel loading requires a MoE model config")
    if ep_size < 1:
        raise ValueError("expert-parallel size must be positive")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(
            f"expert-parallel rank {ep_rank} is outside size {ep_size}"
        )
    if config.moe.num_experts % ep_size:
        raise ValueError(
            f"{config.moe.num_experts} experts do not divide across {ep_size} ranks"
        )
    if type(attention_dp) is not bool:
        raise TypeError("attention_dp must be a boolean")

    resolved = load_checkpoint_quantization(directory, raw)
    quant = resolved.weights
    validate_model_quantization(
        quant,
        is_mla=config.is_mla,
        architecture=config.architecture,
    )
    if attention_dp:
        _validate_attention_dp_build_envelope(
            config,
            quant,
            ep_size=ep_size,
            ep_rank=ep_rank,
            comm=comm,
            dtype=dtype,
            device=device,
        )
    base_factory = linear_factory(
        quant,
        device=device,
        dtype=dtype,
        capabilities=linear_capabilities,
        selection_policy=linear_selection_policy,
    )
    attention_output_factory = (
        linear_factory(
            quant,
            device=device,
            dtype=dtype,
            tensor_parallel_size=ep_size,
            tensor_parallel_rank=ep_rank,
            capabilities=linear_capabilities,
            selection_policy=linear_selection_policy,
        )
        if quant.method is QuantMethod.NVFP4 and not attention_dp
        else None
    )
    factory = _ExpertShardedLinearFactory(
        base_factory,
        num_experts=config.moe.num_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        attention_output_factory=attention_output_factory,
        replicate_attention_output=attention_dp,
    )
    # The official checkpoint is 134 GB. Even rank-local EP2 dummy buffers
    # would allocate ~70 GB on the host before loading useful values. Construct
    # shape-only modules, drop remote experts, and assign only selected tensors.
    with torch.device("meta"):
        model = DenseDecoder(
            config,
            attention_backend=attention_backend,
            linear_factory=factory,
            dtype=dtype,
        )
    for layer_index, layer in enumerate(model.model.layers):
        if quant.method is QuantMethod.NVFP4:
            output = layer.self_attn.o_proj
            if not isinstance(output, NvFp4Linear):
                raise RuntimeError(
                    "NVFP4 EP attention output requires "
                    f"NvFp4Linear at layer {layer_index}, got {type(output).__name__}"
                )
            if attention_dp:
                placement = output.linear_context.tensor_parallel
                attention_width = (
                    config.num_attention_heads * config.head_dim
                )
                if (
                    placement.mode is not TensorParallelMode.REPLICATED
                    or placement.rank != 0
                    or placement.world_size != 1
                    or output.in_features != attention_width
                    or output.out_features != config.hidden_size
                ):
                    raise RuntimeError(
                        "attention-DP requires a full replicated attention o_proj"
                    )
            else:
                ReplicatedInputRowParallelLinear(output, comm)
        if hasattr(layer.mlp, "experts"):
            layer.mlp = EpMoeBlock(
                layer.mlp,
                comm,
                ep_rank=ep_rank,
                ep_size=ep_size,
                attention_dp=attention_dp,
            )
    reader = CheckpointReader(directory)
    local_shapes, blocks, kv_scale_names = _validate_ep_checkpoint_contract(
        reader,
        model,
        tie_word_embeddings=config.tie_word_embeddings,
        kv_cache_quant_algo=resolved.kv_cache_quant_algo,
    )
    if not blocks:
        raise ValueError("Qwen3-MoE checkpoint has no sparse expert layers")
    nvfp4_moe_input_scales = (
        _checkpoint_nvfp4_moe_global_input_scales(reader, blocks)
        if quant.method is QuantMethod.NVFP4
        else {}
    )

    quantized_dtypes = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): buffer.dtype
        for module_name, module in model.named_modules()
        if isinstance(module, QuantizedLinearBase)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    strict_nvfp4_dtypes = {
        (f"{module_name}.{buffer_name}" if module_name else buffer_name): buffer.dtype
        for module_name, module in model.named_modules()
        if isinstance(module, NvFp4Linear)
        for buffer_name, buffer in module.named_buffers(recurse=False)
    }
    member_specs = checkpoint_member_specs(model)
    replicated_sources: dict[str, str] = {}
    sharded_members: list[tuple[str, str, int, int, int]] = []
    for name in local_shapes:
        member = member_specs[name]
        rank, world_size = _checkpoint_shard_coordinates(
            model,
            name,
            member.shard_dim,
        )
        if member.shard_dim is None or world_size == 1:
            if member.source_name in replicated_sources:
                raise RuntimeError(
                    f"duplicate EP checkpoint source {member.source_name!r}"
                )
            replicated_sources[member.source_name] = name
        else:
            sharded_members.append(
                (name, member.source_name, member.shard_dim, rank, world_size)
            )

    loaded: set[str] = set()

    def assign_checkpoint(
        name: str,
        tensor: torch.Tensor,
        *,
        expected_shape: tuple[int, ...],
    ) -> None:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"rank-local checkpoint tensor {name!r} shape "
                f"{tuple(tensor.shape)} != expected {expected_shape}"
            )
        quantized_dtype = quantized_dtypes.get(name)
        if name in strict_nvfp4_dtypes and tensor.dtype != strict_nvfp4_dtypes[name]:
            raise ValueError(
                f"NVFP4 checkpoint tensor {name!r} dtype {tensor.dtype} != "
                f"required ABI dtype {strict_nvfp4_dtypes[name]}"
            )
        if quantized_dtype is not None and tensor.dtype != quantized_dtype:
            tensor = tensor.to(quantized_dtype)
        elif (
            quantized_dtype is None
            and tensor.is_floating_point()
            and tensor.dtype != dtype
        ):
            tensor = tensor.to(dtype)
        _assign_checkpoint_tensor(model, name, tensor)
        loaded.add(name)

    source_destinations = dict(replicated_sources)
    slice_specs: dict[str, tuple[int, int, int, int]] = {}
    for name, source, shard_dim, rank, world_size in sorted(sharded_members):
        local_span = local_shapes[name][shard_dim]
        start = rank * local_span
        end = start + local_span
        if end != (rank + 1) * local_span or end > local_span * world_size:
            raise RuntimeError(f"invalid EP checkpoint slice for {name!r}")
        if source in source_destinations:
            raise RuntimeError(f"duplicate EP checkpoint source {source!r}")
        source_destinations[source] = name
        slice_specs[source] = (shard_dim, start, end, world_size)
    for source, tensor in reader.selected_items(source_destinations):
        name = source_destinations[source]
        if source in slice_specs:
            shard_dim, _start, _end, world_size = slice_specs[source]
            expected_shape = list(local_shapes[name])
            expected_shape[shard_dim] *= world_size
            assign_checkpoint(
                name,
                tensor,
                expected_shape=tuple(expected_shape),
            )
        else:
            assign_checkpoint(
                name,
                tensor,
                expected_shape=local_shapes[name],
            )
    if loaded != set(local_shapes):
        absent = sorted(set(local_shapes) - loaded)
        raise RuntimeError(
            f"rank-local expert load omitted {absent[:8]} ({len(absent)} total)"
        )
    for block_name, block in blocks:
        if block_name in nvfp4_moe_input_scales:
            w13_input_scale, w2_input_scale = nvfp4_moe_input_scales[block_name]
            apply_nvfp4_moe_global_input_scales(
                block.experts,
                w13_input_scale=w13_input_scale,
                w2_input_scale=w2_input_scale,
            )
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    fresh_rope = RotaryEmbedding(config).to(device)
    model.model.rotary_emb._buffers.update(fresh_rope._buffers)
    model.model.rotary_emb.attention_scaling = fresh_rope.attention_scaling
    remaining_meta = [
        name
        for name, tensor in (
            *model.named_parameters(),
            *model.named_buffers(),
        )
        if tensor.device.type == "meta"
    ]
    if remaining_meta:
        raise RuntimeError(
            "expert-parallel load left meta tensors unresolved: "
            f"{remaining_meta[:8]} ({len(remaining_meta)} total)"
        )
    model.eval()
    model.refresh_dense_packs()
    gc.collect()
    model = model.to(device)
    for source, (shard_dim, start, end, _world_size) in sorted(
        slice_specs.items()
    ):
        name = source_destinations[source]
        full = model.get_buffer(name)
        local = full.narrow(shard_dim, start, end - start).contiguous()
        if tuple(local.shape) != local_shapes[name]:
            raise RuntimeError(
                f"resident checkpoint slice {name!r} shape {tuple(local.shape)} "
                f"!= expected {local_shapes[name]}"
            )
        _assign_checkpoint_tensor(model, name, local)
        del full, local
    if quant.method is QuantMethod.NVFP4 and torch.device(device).type == "cuda":
        for _block_name, block in blocks:
            block.prepare_fused_nvfp4()
    owned = blocks[0][1].owned_expert_indices
    if any(block.owned_expert_indices != owned for _name, block in blocks):
        raise RuntimeError("expert ownership differs across sparse layers")
    info = ExpertParallelLoadInfo(
        ep_rank=ep_rank,
        ep_size=ep_size,
        owned_expert_indices=owned,
        quantization_source=resolved.source,
        quantization_method=quant.method.value,
        kv_cache_quant_algo=resolved.kv_cache_quant_algo,
        producer_name=resolved.producer_name,
        producer_version=resolved.producer_version,
        checkpoint_tensor_count=len(reader.names()),
        rank_loaded_tensor_count=len(loaded),
        auxiliary_kv_scale_count=len(kv_scale_names),
    )
    object.__setattr__(model, "_kairyu_ep_load_info", info)
    if attention_dp:
        object.__setattr__(
            model,
            "_kairyu_attention_dp_blocks",
            tuple(block for _name, block in blocks),
        )
        object.__setattr__(
            model,
            "_kairyu_attention_dp_metadata",
            {
                "enabled": True,
                "dispatcher": _ATTENTION_DP_DISPATCHER,
                "expert_parallel_size": ep_size,
                "expert_parallel_rank": ep_rank,
                "attention_placement": "request_owned_data_parallel",
                "attention_output_placement": "replicated",
                "row_layout": "rank_major_zero_padded",
                "activation_collective_dtype": "nvfp4_packed_uint8",
                "scale_collective_dtype": "uint8",
                "combine_collective": "tensor_reduce_scatter",
            },
        )
    return model, config, info
