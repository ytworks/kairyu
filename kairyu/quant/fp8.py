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


def dequantize_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.float32) * scale.to(torch.float32)
