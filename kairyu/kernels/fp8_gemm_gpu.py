"""Selected native/Triton FP8 W8A8 GEMM.

SM120 notes (roadmap §2): ~99 KB shared memory per SM (vs 228 KB on server
dies) — aligned shapes use PyTorch's native scaled MM and ragged shapes use
small Triton tiles.
"""

from __future__ import annotations

import torch


def linear_forward(x: torch.Tensor, module) -> torch.Tensor:
    from kairyu.kernels.quant_gemm_gpu import fp8_linear_forward

    return fp8_linear_forward(x, module)
