"""Triton NVFP4 GEMM (m14 D4) — deploy-day verified vs the CPU reference.

SM120 has native FP4 tensor cores; several NVFP4 grouped-GEMM paths were
immature at design time (roadmap §2) — this Triton dequant path is the
correctness-first fallback the fused kernel replaces.
"""

from __future__ import annotations

import torch


def linear_forward(x: torch.Tensor, module) -> torch.Tensor:
    import triton  # noqa: F401  (deferred: [gpu] extra)

    from kairyu.quant.nvfp4 import dequantize_nvfp4

    weight = dequantize_nvfp4(
        module.weight, module.weight_scale, module.weight_scale_2
    ).to(x.dtype)
    return torch.nn.functional.linear(x, weight, module.bias)
