"""INT8 W8A8 (compressed-tensors) reference (m14 D1, review A7).

Weights: per-channel symmetric int8 (scale [out, 1], no zero point).
Activations: dynamic per-token symmetric (scale = rowmax(|a|)/127).
The reference matmul accumulates in EXACT int32 — the GPU kernel's bit-exact
oracle. CPU torch.matmul rejects int8 operands: both sides upcast to int32
first (int32 matmul verified working on CPU).
"""

from __future__ import annotations

import torch

INT8_MAX = 127


def quantize_int8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """weight [out, in] -> (int8 weight, fp32 scale [out, 1])."""
    amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / INT8_MAX).to(torch.float32)
    q = torch.round(weight.to(torch.float32) / scale).clamp(-INT8_MAX, INT8_MAX)
    return q.to(torch.int8), scale


def quantize_int8_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [T, in] -> (int8 x, fp32 per-token scale [T, 1])."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / INT8_MAX).to(torch.float32)
    q = torch.round(x.to(torch.float32) / scale).clamp(-INT8_MAX, INT8_MAX)
    return q.to(torch.int8), scale


def int8_w8a8_matmul(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Exact int32-accumulated reference: [T, in] @ [out, in]^T -> [T, out]."""
    accumulated = torch.matmul(x_q.to(torch.int32), w_q.to(torch.int32).t())
    return accumulated.to(torch.float32) * x_scale * w_scale.t()


def dequantize_int8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.float32) * scale
