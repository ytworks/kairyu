"""DeepSeek-V4 packed-FP4 expert parallel execution for SM120.

The public V4 checkpoint stores two E2M1 values per byte and one UE8M0 scale
per 32 logical K values.  This module keeps that ABI resident.  It uses the
Transformers fine-grained Triton matmul for each local expert and fixed-size
NCCL all-to-all dispatch/combine tiles; neither weights nor scales are
requantized.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

_DISPATCH_TILE_TOKENS = 2048
_IMPLEMENTATION = "kairyu_deepseek_v4"
FINEGRAINED_FP8_REPOSITORY = "kernels-community/finegrained-fp8"
FINEGRAINED_FP8_REVISION = "061130fedf845f320c56de4425f7404f6512c87e"


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return a raw local tensor for ordinary tensors and DTensors."""

    local = getattr(tensor, "to_local", None)
    return local() if callable(local) else tensor


def _fp4_linear(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Run the public 1x32 E2M1/UE8M0 matrix without a dense fallback."""

    if hidden.device.type != "cuda":
        raise RuntimeError("DeepSeek V4 packed FP4 experts require CUDA")
    hidden = hidden.contiguous()
    capability = torch.cuda.get_device_capability(hidden.device)
    if capability != (12, 0):
        raise RuntimeError(
            "DeepSeek V4 packed FP4 experts are production-gated on SM120; "
            f"got SM{capability[0]}{capability[1]}"
        )
    if weight.dtype is not torch.int8:
        raise TypeError(f"DeepSeek V4 expert weight must be packed int8, got {weight.dtype}")
    ue8m0 = getattr(torch, "float8_e8m0fnu", None)
    if ue8m0 is None or scale.dtype is not ue8m0:
        raise TypeError(
            "DeepSeek V4 expert scales must retain the checkpoint's float8_e8m0fnu ABI"
        )
    logical_k = weight.shape[-1] * 2
    expected_scale = (weight.shape[-2], (logical_k + 31) // 32)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(
            f"DeepSeek V4 expert scale shape {tuple(scale.shape)} != {expected_scale}"
        )
    try:
        from transformers.integrations.finegrained_fp8 import finegrained_fp8_linear
    except ImportError as error:  # pragma: no cover - production image owns dependency
        raise RuntimeError(
            "DeepSeek V4 FP4 execution requires transformers>=5.12,<5.13"
        ) from error
    # The DeepGEMM integration rejects SM120 at runtime.  Calling the explicit
    # fine-grained implementation selects the checked Triton FP8xFP4 kernel and
    # cannot silently materialize a BF16 weight.
    return finegrained_fp8_linear(
        hidden,
        weight,
        scale,
        block_size=[128, 128],
        output_dtype=hidden.dtype,
    )


def _all_to_all_fixed(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    group,
) -> None:
    if output.shape != input.shape or not output.is_contiguous() or not input.is_contiguous():
        raise ValueError("DeepSeek V4 fixed all-to-all tensors must be equal contiguous shapes")
    dist.all_to_all_single(output, input, group=group)


def _local_expert_forward(
    module,
    hidden: torch.Tensor,
    local_expert_ids: torch.Tensor,
) -> torch.Tensor:
    """Execute received rows on rank-local official FP4 experts."""

    gate_up = _local(module.gate_up_proj)
    gate_up_scale = _local(module.gate_up_proj_scale_inv)
    down = _local(module.down_proj)
    down_scale = _local(module.down_proj_scale_inv)
    local_experts = gate_up.shape[0]
    if not (
        down.shape[0]
        == gate_up_scale.shape[0]
        == down_scale.shape[0]
        == local_experts
    ):
        raise RuntimeError("DeepSeek V4 rank-local FP4 expert tensors disagree")
    output = torch.zeros(
        (hidden.shape[0], hidden.shape[1]),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    for expert in range(local_experts):
        rows = torch.where(local_expert_ids == expert)[0]
        if rows.numel() == 0:
            continue
        selected = hidden.index_select(0, rows)
        projected = _fp4_linear(selected, gate_up[expert], gate_up_scale[expert])
        projected = module._apply_gate(projected)
        projected = _fp4_linear(projected, down[expert], down_scale[expert])
        output.index_copy_(0, rows, projected.to(output.dtype))
    return output


def _distributed_tile(
    module,
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    token_ids: torch.Tensor,
    route_slots: torch.Tensor,
    *,
    process_group,
    output_token_count: int,
    top_k: int,
) -> torch.Tensor:
    """Dispatch and combine one equal-capacity all-to-all tile."""

    world_size = dist.get_world_size(process_group)
    local_experts = _local(module.gate_up_proj).shape[0]
    global_experts = local_experts * world_size
    hidden_size = hidden.shape[-1]
    device = hidden.device
    valid = (expert_ids >= 0) & (expert_ids < global_experts)
    owners = torch.div(expert_ids.clamp_min(0), local_experts, rounding_mode="floor")
    owner_counts = torch.bincount(owners[valid], minlength=world_size)
    dist.all_reduce(owner_counts, op=dist.ReduceOp.MAX, group=process_group)
    capacity = max(1, int(owner_counts.max().item()))
    send_hidden = hidden.new_zeros((world_size, capacity, hidden_size))
    send_expert = torch.full(
        (world_size, capacity),
        -1,
        dtype=torch.int32,
        device=device,
    )
    send_weight = route_weights.new_zeros((world_size, capacity))
    send_token = torch.full(
        (world_size, capacity),
        -1,
        dtype=torch.int64,
        device=device,
    )
    send_slot = torch.full_like(send_token, -1)
    for owner in range(world_size):
        positions = torch.where(valid & (owners == owner))[0]
        count = positions.numel()
        if count > capacity:
            raise RuntimeError(
                f"DeepSeek V4 dispatch overflow for rank {owner}: {count} > {capacity}"
            )
        if count == 0:
            continue
        send_hidden[owner, :count] = hidden.index_select(0, positions)
        send_expert[owner, :count] = (
            expert_ids.index_select(0, positions) - owner * local_experts
        ).to(torch.int32)
        send_weight[owner, :count] = route_weights.index_select(0, positions)
        send_token[owner, :count] = token_ids.index_select(0, positions)
        send_slot[owner, :count] = route_slots.index_select(0, positions)

    recv_hidden = torch.empty_like(send_hidden)
    recv_expert = torch.empty_like(send_expert)
    recv_weight = torch.empty_like(send_weight)
    for output, input in (
        (recv_hidden, send_hidden),
        (recv_expert, send_expert),
        (recv_weight, send_weight),
    ):
        _all_to_all_fixed(output, input, group=process_group)

    received = _local_expert_forward(
        module,
        recv_hidden.flatten(0, 1),
        recv_expert.flatten().to(torch.long),
    ).view(world_size, capacity, hidden_size)
    received.mul_(recv_weight.unsqueeze(-1).to(received.dtype))
    returned = torch.empty_like(received)
    _all_to_all_fixed(returned, received, group=process_group)

    contributions = torch.zeros(
        (output_token_count, top_k, hidden_size),
        dtype=torch.float32,
        device=device,
    )
    for owner in range(world_size):
        positions = torch.where(send_token[owner] >= 0)[0]
        if positions.numel() == 0:
            continue
        local_tokens = send_token[owner].index_select(0, positions)
        local_slots = send_slot[owner].index_select(0, positions)
        contributions[local_tokens, local_slots] = returned[owner].index_select(
            0, positions
        ).to(torch.float32)
    # Preserve the checkpoint/reference top-k slot order independent of EP
    # ownership.  A single cast follows the deterministic FP32 reduction.
    return contributions.sum(dim=1, dtype=torch.float32).to(hidden.dtype)


def deepseek_v4_fp4_ep_forward(
    module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    process_group=None,
) -> torch.Tensor:
    """Fixed-length all-to-all EP over official packed FP4 expert weights."""

    if process_group is None:
        raise ValueError("DeepSeek V4 Attention-DP requires its EP process group")
    if hidden_states.ndim != 2 or top_k_index.ndim != 2:
        raise ValueError("DeepSeek V4 experts require [tokens, hidden] and [tokens, top-k]")
    if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != hidden_states.shape[0]:
        raise ValueError("DeepSeek V4 route tensor geometry disagrees with hidden rows")
    if hidden_states.dtype is not torch.bfloat16:
        raise TypeError("DeepSeek V4 FP4 EP requires BF16 activations")
    tokens, top_k = top_k_index.shape
    if tokens < 1 or top_k < 1:
        raise ValueError("DeepSeek V4 FP4 EP requires at least one token and route")

    max_tokens = torch.tensor(tokens, dtype=torch.int64, device=hidden_states.device)
    dist.all_reduce(max_tokens, op=dist.ReduceOp.MAX, group=process_group)
    global_tokens = int(max_tokens.item())
    output = torch.zeros_like(hidden_states)
    for start in range(0, global_tokens, _DISPATCH_TILE_TOKENS):
        global_stop = min(global_tokens, start + _DISPATCH_TILE_TOKENS)
        stop = min(tokens, global_stop)
        local_count = max(0, stop - start)
        if local_count:
            tile_hidden = hidden_states[start:stop]
            tile_experts = top_k_index[start:stop].reshape(-1).to(torch.long)
            tile_weights = top_k_weights[start:stop].reshape(-1)
            pair_hidden = tile_hidden.repeat_interleave(top_k, dim=0)
            pair_tokens = torch.arange(
                local_count,
                dtype=torch.int64,
                device=hidden_states.device,
            ).repeat_interleave(top_k)
            pair_slots = torch.arange(
                top_k,
                dtype=torch.int64,
                device=hidden_states.device,
            ).repeat(local_count)
        else:
            pair_hidden = hidden_states.new_empty((0, hidden_states.shape[-1]))
            tile_experts = top_k_index.new_empty((0,), dtype=torch.long)
            tile_weights = top_k_weights.new_empty((0,))
            pair_tokens = top_k_index.new_empty((0,), dtype=torch.int64)
            pair_slots = top_k_index.new_empty((0,), dtype=torch.int64)
        tile_output = _distributed_tile(
            module,
            pair_hidden,
            tile_experts,
            tile_weights,
            pair_tokens,
            pair_slots,
            process_group=process_group,
            output_token_count=local_count,
            top_k=top_k,
        )
        if local_count:
            output[start:stop] = tile_output
    return output


def register_deepseek_v4_experts() -> str:
    """Register the implementation before quantized module construction."""

    try:
        from transformers.integrations.finegrained_fp8 import (
            ALL_FP8_EXPERTS_FUNCTIONS,
            FP8Experts,
        )
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError("DeepSeek V4 native execution requires transformers 5.12") from error
    ALL_FP8_EXPERTS_FUNCTIONS.register(_IMPLEMENTATION, deepseek_v4_fp4_ep_forward)
    FP8Experts._impl_tp_layer_overrides[_IMPLEMENTATION] = {
        "moe_tp_experts": "megamoe_experts",
        "ep_router": "megamoe_router",
    }
    # Dense FP8 projections already use the same fine-grained Triton package.
    # Avoid probing DeepGEMM on every projection: it is hard-rejected on SM120.
    os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
    from transformers.integrations import hub_kernels

    # Transformers otherwise resolves the mutable kernel-version reference at
    # first use.  Production uses the image-baked commit and remains offline.
    hub_kernels._HUB_KERNEL_MAPPING["finegrained-fp8"] = {
        "repo_id": FINEGRAINED_FP8_REPOSITORY,
        "revision": FINEGRAINED_FP8_REVISION,
    }
    return _IMPLEMENTATION


__all__ = [
    "deepseek_v4_fp4_ep_forward",
    "register_deepseek_v4_experts",
]
