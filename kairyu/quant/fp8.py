"""FP8-E4M3 W8A8 (compressed-tensors) reference (m14 D1, review A3/A6).

Storage uses native ``torch.float8_e4m3fn``; ALL compute upcasts first.
torch's CPU cast is NON-saturating (values past ~464 become NaN — verified),
so quantization clamps to ±448 BEFORE the cast; with the clamp, RNE rounding
matches the GPU's saturating kernels.

compressed-tensors variants (review A6): static = per-TENSOR weight_scale
(shape (1,)) + input_scale; FP8_DYNAMIC = per-CHANNEL weight_scale [out, 1],
no input_scale. Symmetric — no zero-point tensor exists.
"""

from __future__ import annotations

import torch

FP8_MAX = 448.0


def quantize_fp8(
    weight: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """weight [out, in] -> (fp8 weight, scale (1,) or [out, 1])."""
    if per_channel:
        amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    else:
        amax = weight.abs().amax().reshape(1).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    scaled = (weight.to(torch.float32) / scale).clamp(-FP8_MAX, FP8_MAX)
    return scaled.to(torch.float8_e4m3fn), scale


def quantize_fp8_activation(
    x: torch.Tensor, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., in] -> (fp8 x, scale), dynamically or with a static scalar."""
    if scale is None:
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (amax / FP8_MAX).to(torch.float32)
    elif scale.numel() != 1:
        raise ValueError(f"static FP8 input_scale must be scalar, got {scale.shape}")
    else:
        scale = scale.to(device=x.device, dtype=torch.float32)
    scaled = (x.to(torch.float32) / scale).clamp(-FP8_MAX, FP8_MAX)
    return scaled.to(torch.float8_e4m3fn), scale


def fp8_w8a8_matmul(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Upcasted CPU oracle for dynamic-token FP8 accumulation."""
    accumulated = torch.matmul(x_q.to(torch.float32), w_q.to(torch.float32).t())
    return accumulated * x_scale * w_scale.reshape(-1)


def dequantize_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.float32) * scale.to(torch.float32)
