"""Native FlashInfer NVFP4 W4A4 GEMM.

SM100+ has native FP4 tensor cores. Activations and checkpoint weights remain
FP4 through the selected FlashInfer kernel; no W4A16 fallback is enabled.
"""

from __future__ import annotations

import torch


def linear_forward(x: torch.Tensor, module) -> torch.Tensor:
    from kairyu.kernels.quant_gemm_gpu import nvfp4_linear_forward

    return nvfp4_linear_forward(x, module)
