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
