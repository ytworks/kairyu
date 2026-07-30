"""One-launch BF16 RMSNorm with Kairyu's historical rounding boundary."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - caller retains the torch fallback
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _round_bf16_to_fp32(value):
        """Round fp32 to BF16 while retaining an fp32 register value."""

        bits = value.to(tl.int32, bitcast=True)
        bias = 0x7FFF + ((bits >> 16) & 1)
        rounded = (bits + bias) & -65536
        return rounded.to(tl.float32, bitcast=True)

    @triton.jit
    def _rms_norm_bf16_kernel(
        source,
        scale,
        output,
        WIDTH: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < WIDTH
        offset = row * WIDTH + columns
        values = tl.load(source + offset, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / WIDTH
        # Preserve the model's established BF16 intermediate before applying
        # the weight, and prevent LLVM from folding the rounding boundary.
        normalized = _round_bf16_to_fp32(
            values * tl.rsqrt(variance + EPSILON)
        )
        scales = tl.load(scale + columns, mask=mask, other=0.0)
        tl.store(output + offset, normalized * scales, mask=mask)

    @triton.jit
    def _joint_qk_rms_norm_bf16_kernel(
        query,
        keys,
        query_scale,
        key_scale,
        query_output,
        key_output,
        QUERY_BATCH_STRIDE: tl.constexpr,
        QUERY_HEAD_STRIDE: tl.constexpr,
        KEY_BATCH_STRIDE: tl.constexpr,
        KEY_HEAD_STRIDE: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        KEY_HEADS: tl.constexpr,
        WIDTH: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        heads_per_batch: tl.constexpr = QUERY_HEADS + KEY_HEADS
        batch = row // heads_per_batch
        head = row % heads_per_batch
        columns = tl.arange(0, BLOCK)
        mask = columns < WIDTH

        if head < QUERY_HEADS:
            query_offset = (
                batch * QUERY_BATCH_STRIDE
                + head * QUERY_HEAD_STRIDE
                + columns
            )
            values = tl.load(
                query + query_offset,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            variance = tl.sum(values * values, axis=0) / WIDTH
            normalized = _round_bf16_to_fp32(
                values * tl.rsqrt(variance + EPSILON)
            )
            scales = tl.load(
                query_scale + columns,
                mask=mask,
                other=0.0,
            )
            output_offset = (
                (batch * QUERY_HEADS + head) * WIDTH + columns
            )
            tl.store(
                query_output + output_offset,
                normalized * scales,
                mask=mask,
            )
        else:
            key_head = head - QUERY_HEADS
            key_offset = (
                batch * KEY_BATCH_STRIDE
                + key_head * KEY_HEAD_STRIDE
                + columns
            )
            values = tl.load(
                keys + key_offset,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            variance = tl.sum(values * values, axis=0) / WIDTH
            normalized = _round_bf16_to_fp32(
                values * tl.rsqrt(variance + EPSILON)
            )
            scales = tl.load(
                key_scale + columns,
                mask=mask,
                other=0.0,
            )
            output_offset = (
                (batch * KEY_HEADS + key_head) * WIDTH + columns
            )
            tl.store(
                key_output + output_offset,
                normalized * scales,
                mask=mask,
            )


def try_rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Return the fused CUDA result, or ``None`` for the safe torch path.

    The model contract rounds the normalized activation to BF16 before the
    learned weight multiplication. PyTorch's weighted fused RMSNorm performs
    that multiplication before its output rounding and measurably changes
    Qwen3-32B logits. This kernel keeps the established intermediate rounding
    while fusing both operations into one launch.
    """

    if (
        hidden.device.type != "cuda"
        or hidden.dtype is not torch.bfloat16
        or weight.device != hidden.device
        or weight.dtype is not torch.bfloat16
        or hidden.ndim < 1
        or hidden.shape[-1] != weight.numel()
        or not hidden.is_contiguous()
        or not weight.is_contiguous()
        or hidden.shape[-1] < 1
        or hidden.shape[-1] > 65536
        or triton is None
    ):
        return None

    width = hidden.shape[-1]
    rows = hidden.numel() // width
    block = triton.next_power_of_2(width)
    output = torch.empty_like(hidden)
    _rms_norm_bf16_kernel[(rows,)](
        hidden,
        weight,
        output,
        WIDTH=width,
        EPSILON=eps,
        BLOCK=block,
        num_warps=8 if width >= 4096 else 4,
    )
    return output


def try_joint_qk_rms_norm(
    query: torch.Tensor,
    keys: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Normalize packed Q/K views in one stride-aware CUDA launch.

    Packed QKV projection outputs are split views into a wider row.  Their
    last dimension is contiguous, but the batch stride still spans Q, K, and
    V, so the single-tensor kernel cannot safely treat the views as dense.
    This kernel accepts that layout explicitly and writes dense Q/K outputs.
    Unsupported inputs return ``None`` so callers retain their established
    per-tensor RMSNorm path.
    """

    if (
        query.device.type != "cuda"
        or keys.device != query.device
        or query_weight.device != query.device
        or key_weight.device != query.device
        or query.dtype is not torch.bfloat16
        or keys.dtype is not torch.bfloat16
        or query_weight.dtype is not torch.bfloat16
        or key_weight.dtype is not torch.bfloat16
        or query.layout is not torch.strided
        or keys.layout is not torch.strided
        or query.ndim != 3
        or keys.ndim != 3
        or query_weight.ndim != 1
        or key_weight.ndim != 1
        or query.shape[0] != keys.shape[0]
        or query.shape[-1] != keys.shape[-1]
        or query.shape[-1] != query_weight.numel()
        or keys.shape[-1] != key_weight.numel()
        or query.shape[0] < 1
        or query.shape[1] < 1
        or keys.shape[1] < 1
        or query.shape[-1] < 1
        or query.shape[-1] > 65536
        or query.stride(-1) != 1
        or keys.stride(-1) != 1
        or not query_weight.is_contiguous()
        or not key_weight.is_contiguous()
        or triton is None
    ):
        return None

    batch, query_heads, width = query.shape
    key_heads = keys.shape[1]
    block = triton.next_power_of_2(width)
    query_output = torch.empty(
        query.shape,
        device=query.device,
        dtype=query.dtype,
    )
    key_output = torch.empty(
        keys.shape,
        device=keys.device,
        dtype=keys.dtype,
    )
    _joint_qk_rms_norm_bf16_kernel[
        (batch * (query_heads + key_heads),)
    ](
        query,
        keys,
        query_weight,
        key_weight,
        query_output,
        key_output,
        QUERY_BATCH_STRIDE=query.stride(0),
        QUERY_HEAD_STRIDE=query.stride(1),
        KEY_BATCH_STRIDE=keys.stride(0),
        KEY_HEAD_STRIDE=keys.stride(1),
        QUERY_HEADS=query_heads,
        KEY_HEADS=key_heads,
        WIDTH=width,
        EPSILON=eps,
        BLOCK=block,
        num_warps=8 if width >= 4096 else 4,
    )
    return query_output, key_output
