from __future__ import annotations

import torch
from torch import nn

from kairyu.models.moe import (
    _mix_experts,
    apply_nvfp4_moe_global_input_scales,
    route_experts,
)
from kairyu.quant.linear import NvFp4Linear


class _FixedGate(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor(
            [[1.25, -0.5, 0.75, 0.125]],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return logits.expand(hidden.shape[0], -1)


class _QwenRouteBlock(nn.Module):
    top_k = 2
    norm_topk_prob = True

    def __init__(self) -> None:
        super().__init__()
        self.gate = _FixedGate()


class _ConstantExpert(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.full_like(hidden, self.value)


class _NvFp4Expert(nn.Module):
    def __init__(self, gate_scale: float, up_scale: float, down_scale: float) -> None:
        super().__init__()
        self.gate_proj = NvFp4Linear(16, 16, False)
        self.up_proj = NvFp4Linear(16, 16, False)
        self.down_proj = NvFp4Linear(16, 16, False)
        self.gate_proj.input_scale.fill_(gate_scale)
        self.up_proj.input_scale.fill_(up_scale)
        self.down_proj.input_scale.fill_(down_scale)


def test_qwen_router_retains_fp32_weights_for_final_combine() -> None:
    hidden = torch.zeros(3, 16, dtype=torch.bfloat16)

    indices, weights = route_experts(_QwenRouteBlock(), hidden)

    logits = _FixedGate()(hidden)
    expected_weights, expected_indices = torch.softmax(
        logits, dim=-1, dtype=torch.float32
    ).topk(2, dim=-1)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    assert weights.dtype == torch.float32
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def test_local_moe_combines_fp32_router_weights_before_final_bf16_cast() -> None:
    hidden = torch.zeros(1, 1, dtype=torch.bfloat16)
    experts = nn.ModuleList(
        [_ConstantExpert(-5.84375), _ConstantExpert(-7.5625)]
    )
    indices = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.5715058445930481, 0.4284941852092743]])

    actual = _mix_experts(hidden, experts, indices, weights)

    expert_values = torch.tensor([[-5.84375, -7.5625]], dtype=torch.bfloat16)
    expected = (expert_values.float() * weights).sum(dim=1).to(torch.bfloat16)
    premature_bf16 = (
        expert_values * weights.to(torch.bfloat16)
    ).sum(dim=1)
    torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)
    assert not torch.equal(actual[:, 0], premature_bf16)


def test_nvfp4_moe_uses_layer_global_fc1_and_fc2_input_scales() -> None:
    experts = nn.ModuleList(
        [
            _NvFp4Expert(0.5, 0.75, 2.0),
            _NvFp4Expert(1.5, 1.25, 3.0),
        ]
    )

    w13_scale, w2_scale = apply_nvfp4_moe_global_input_scales(experts)

    assert w13_scale.item() == 1.5
    assert w2_scale.item() == 3.0
    for expert in experts:
        assert expert.gate_proj.input_scale.item() == 1.5
        assert expert.up_proj.input_scale.item() == 1.5
        assert expert.down_proj.input_scale.item() == 3.0
