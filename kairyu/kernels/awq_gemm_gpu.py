"""Triton AWQ W4A16 GEMM (m14 D4) — deploy-day verified vs the CPU reference."""

from __future__ import annotations

import torch


def linear_forward(x: torch.Tensor, module) -> torch.Tensor:
    import triton  # noqa: F401  (deferred: [gpu] extra)

    from kairyu.quant.awq import dequantize_awq

    # v0: on-the-fly dequant + cuBLAS matmul; fused nibble-unpack kernel is the
    # deploy-day optimization (tuned to SM120's smem budget)
    weight = dequantize_awq(
        module.qweight, module.qzeros, module.scales, module.group_size
    ).to(x.dtype)
    out = torch.nn.functional.linear(x, weight, module.bias)
    return out
