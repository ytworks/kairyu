"""SM120 smoke for the official DeepSeek V4 packed-FP4 ABI."""

from __future__ import annotations

import pytest
import torch

from kairyu.kernels.deepseek_v4_moe_gpu import _fp4_linear
from kairyu.models.deepseek_v4_cache import PackedFP4IndexerCache
from kairyu.quant.nvfp4 import E2M1_LUT

pytestmark = pytest.mark.gpu


def _dequant(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    nibbles = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(raw.shape[0], -1)
    values = E2M1_LUT.to(raw.device)[(nibbles & 0x7).long()]
    values *= torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return (
        values.view(values.shape[0], scales.shape[1], 32)
        * scales.float().unsqueeze(-1)
    ).reshape(values.shape)


def test_deepseek_v4_packed_fp4_linear_runs_on_sm120() -> None:
    if not torch.cuda.is_available():
        pytest.skip("DeepSeek V4 FP4 requires CUDA")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("DeepSeek V4 FP4 production gate requires SM120")
    torch.manual_seed(7)
    device = torch.device("cuda")
    source = torch.randn(1, 128, 128, device=device)
    cache = PackedFP4IndexerCache(1, 128)
    cache.append(source)
    packed = cache.packed_chunks[0]
    scales = cache.scale_chunks[0]
    hidden = torch.randn(16, 128, device=device, dtype=torch.bfloat16)

    actual = _fp4_linear(hidden, packed, scales)
    expected = hidden.float() @ _dequant(packed, scales).t()

    assert actual.dtype is torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=1.0, rtol=5e-2)
