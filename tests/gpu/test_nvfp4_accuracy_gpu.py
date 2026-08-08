from __future__ import annotations

import pytest
import torch

from kairyu.quant import nvfp4
from kairyu.quant.linear import NvFp4Linear, NvFp4ToFp8Linear

pytestmark = pytest.mark.gpu


def _loaded(module, source: torch.Tensor):
    packed, scale, scale_2 = nvfp4.quantize_nvfp4(source)
    module.load_state_dict(
        {
            "weight": packed,
            "weight_scale": scale,
            "weight_scale_2": scale_2,
            "input_scale": torch.tensor(0.01),
        },
        assign=True,
    )
    return module


def test_dynamic_per_token_nvfp4_matches_cpu_oracle_and_counts_no_clipping() -> None:
    torch.manual_seed(355)
    weight = torch.randn(32, 32)
    cpu = _loaded(
        NvFp4Linear(
            32,
            32,
            False,
            activation_dynamic=True,
            observe_activation_saturation=True,
        ),
        weight,
    )
    gpu = _loaded(
        NvFp4Linear(
            32,
            32,
            False,
            activation_dynamic=True,
            observe_activation_saturation=True,
        ),
        weight,
    ).to("cuda")
    values = torch.randn(4, 32, dtype=torch.bfloat16)
    expected = cpu(values)
    actual = gpu(values.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.12)
    assert gpu.activation_blocks.item() == 8
    assert gpu.saturated_activation_blocks.item() == 0


def test_nvfp4_source_fp8_runtime_uses_fused_fp8_without_source_buffers() -> None:
    torch.manual_seed(356)
    weight = torch.randn(32, 32)
    cpu = _loaded(NvFp4ToFp8Linear(32, 32, False), weight)
    gpu = _loaded(NvFp4ToFp8Linear(32, 32, False), weight).to("cuda")
    cpu.materialize_runtime_weight()
    gpu.materialize_runtime_weight()
    values = torch.randn(4, 32, dtype=torch.bfloat16)
    expected = cpu(values)
    actual = gpu(values.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.12)
    assert gpu.weight.dtype is torch.float8_e4m3fn
    assert "weight_scale_2" not in gpu._buffers
    assert "input_scale" not in gpu._buffers
