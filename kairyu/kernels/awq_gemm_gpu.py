"""Fused Triton AWQ W4A16 GEMM."""

from __future__ import annotations

import torch


def linear_forward(x: torch.Tensor, module) -> torch.Tensor:
    from kairyu.kernels.quant_gemm_gpu import awq_linear_forward

    return awq_linear_forward(x, module)
