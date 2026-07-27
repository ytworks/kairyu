"""Fused Triton INT8 W8A8 GEMM."""

from __future__ import annotations

import torch


def linear_forward(x: torch.Tensor, module) -> torch.Tensor:
    from kairyu.kernels.quant_gemm_gpu import int8_linear_forward

    return int8_linear_forward(x, module)
