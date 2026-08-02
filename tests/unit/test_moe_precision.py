from __future__ import annotations

import pytest
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


class _MixedExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(16, 16, bias=False)
        self.up_proj = NvFp4Linear(16, 16, False)
        self.down_proj = nn.Linear(16, 16, bias=False)
        self.up_proj.input_scale.fill_(2.5)


class _CopyCommunicator:
    def tensor_all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        _output_split_sizes: list[int],
        _input_split_sizes: list[int],
    ) -> None:
        output.copy_(input_)

    def tensor_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _PrecisionMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [_ConstantExpert(-5.84375), _ConstantExpert(-7.5625)]
        )

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.tensor([[0, 1]], device=hidden.device)
        weights = torch.tensor(
            [[0.5715058445930481, 0.4284941852092743]],
            device=hidden.device,
        )
        return indices, weights


def test_qwen_router_retains_fp32_weights_for_final_combine() -> None:
    hidden = torch.zeros(3, 16, dtype=torch.bfloat16)

    indices, weights = route_experts(_QwenRouteBlock(), hidden)

    logits = _FixedGate()(hidden)
    expected_indices = torch.argsort(
        logits,
        dim=-1,
        descending=True,
        stable=True,
    )[:, :2]
    expected_weights = torch.softmax(
        logits.gather(1, expected_indices).float(),
        dim=-1,
    )
    assert weights.dtype == torch.float32
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


def test_qwen_router_breaks_bf16_logit_ties_by_smaller_expert_id() -> None:
    hidden = torch.zeros(1, 16, dtype=torch.bfloat16)

    class TiedGate(nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                [[1.0, 1.0, 1.0, 0.0]],
                dtype=hidden.dtype,
                device=hidden.device,
            )

    block = _QwenRouteBlock()
    block.gate = TiedGate()
    indices, _weights = route_experts(block, hidden)

    # Experts 0 and 1 are the selected pair. This pins the stable smaller-id
    # ordering independently of torch.topk's backend-specific tie behavior.
    assert indices.tolist() == [[0, 1]]


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

    assert w13_scale is not None
    assert w2_scale is not None
    assert w13_scale.item() == 1.5
    assert w2_scale.item() == 3.0
    for expert in experts:
        assert expert.gate_proj.input_scale.item() == 1.5
        assert expert.up_proj.input_scale.item() == 1.5
        assert expert.down_proj.input_scale.item() == 3.0


def test_nvfp4_moe_globalizes_only_quantized_projections() -> None:
    mixed = _MixedExpert()
    quantized = _NvFp4Expert(1.0, 1.5, 4.0)
    experts = nn.ModuleList([mixed, quantized])

    w13_scale, w2_scale = apply_nvfp4_moe_global_input_scales(experts)

    assert w13_scale is not None
    assert w2_scale is not None
    assert w13_scale.item() == 2.5
    assert w2_scale.item() == 4.0
    assert mixed.up_proj.input_scale.item() == 2.5
    assert quantized.gate_proj.input_scale.item() == 2.5
    assert quantized.up_proj.input_scale.item() == 2.5
    assert quantized.down_proj.input_scale.item() == 4.0


def test_ep_moe_combines_fp32_router_weights_before_final_bf16_cast() -> None:
    from kairyu.models.moe_parallel import EpMoeBlock

    hidden = torch.zeros(1, 1, dtype=torch.bfloat16)
    block = EpMoeBlock(
        _PrecisionMoeBlock(),
        _CopyCommunicator(),
        ep_rank=0,
        ep_size=1,
    )

    actual = block(hidden)

    expert_values = torch.tensor([[-5.84375, -7.5625]], dtype=torch.bfloat16)
    weights = torch.tensor([[0.5715058445930481, 0.4284941852092743]])
    expected = (expert_values.float() * weights).sum(dim=1).to(torch.bfloat16)
    premature_bf16 = (
        expert_values * weights.to(torch.bfloat16)
    ).sum(dim=1)
    torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)
    assert not torch.equal(actual[:, 0], premature_bf16)


def test_ep_finalize_casts_owner_partials_before_rank_reduction() -> None:
    from kairyu.models.moe_parallel import _ep_rank_partial

    expert_outputs = torch.tensor(
        [[[-5.84375], [-7.5625]]],
        dtype=torch.bfloat16,
    )
    indices = torch.tensor([[0, 2]])
    weights = torch.tensor([[0.5715058445930481, 0.4284941852092743]])

    rank_0 = _ep_rank_partial(
        expert_outputs,
        indices,
        weights,
        ep_rank=0,
        experts_per_rank=2,
        output_dtype=torch.bfloat16,
    )
    rank_1 = _ep_rank_partial(
        expert_outputs,
        indices,
        weights,
        ep_rank=1,
        experts_per_rank=2,
        output_dtype=torch.bfloat16,
    )
    actual = rank_0 + rank_1

    expected = (
        (expert_outputs[:, 0].float() * weights[:, 0, None]).to(torch.bfloat16)
        + (expert_outputs[:, 1].float() * weights[:, 1, None]).to(torch.bfloat16)
    )
    one_global_fp32_sum = (
        expert_outputs.float() * weights[:, :, None]
    ).sum(dim=1).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, one_global_fp32_sum)


def test_ep_nvfp4_pack_preserves_canonical_buffers_without_weight_duplication() -> None:
    from kairyu.models.moe_parallel import (
        EpMoeBlock,
        _pack_nvfp4_moe_block,
    )

    class NvFp4Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList(
                [_NvFp4Expert(2.0, 2.0, 3.0) for _ in range(2)]
            )

        def _route(self, hidden: torch.Tensor):
            del hidden
            raise AssertionError("packing must not execute routing")

    block = EpMoeBlock(
        NvFp4Block(),
        _CopyCommunicator(),
        ep_rank=0,
        ep_size=1,
    )
    for local_index, expert in enumerate(block.local_experts):
        expert.up_proj.weight.fill_(10 + local_index)
        expert.gate_proj.weight.fill_(20 + local_index)
        expert.down_proj.weight.fill_(30 + local_index)
        expert.up_proj.weight_scale_2.fill_(local_index + 1)
        expert.gate_proj.weight_scale_2.fill_(local_index + 1)
        expert.down_proj.weight_scale_2.fill_(local_index + 3)
    state_names = tuple(block.state_dict())

    prepared = _pack_nvfp4_moe_block(
        block,
        require_flashinfer_selection=False,
    )

    fc1_bytes = prepared.fc1_weight.view(torch.uint8)
    fc2_bytes = prepared.fc2_weight.view(torch.uint8)
    assert fc1_bytes.shape == (2, 32, 8)
    assert fc2_bytes.shape == (2, 16, 8)
    for local_index, expert in enumerate(block.local_experts):
        assert torch.all(fc1_bytes[local_index, :16] == 10 + local_index)
        assert torch.all(fc1_bytes[local_index, 16:] == 20 + local_index)
        assert torch.all(fc2_bytes[local_index] == 30 + local_index)
        assert (
            expert.up_proj.weight.untyped_storage().data_ptr()
            == expert.gate_proj.weight.untyped_storage().data_ptr()
            == prepared.fc1_weight.untyped_storage().data_ptr()
        )
        assert (
            expert.down_proj.weight.untyped_storage().data_ptr()
            == prepared.fc2_weight.untyped_storage().data_ptr()
        )
    torch.testing.assert_close(
        prepared.fc1_act_scale,
        torch.tensor(0.5),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        prepared.fc1_dequant_scale,
        torch.tensor([2.0, 4.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        prepared.fc2_dequant_scale,
        torch.tensor([9.0, 12.0]),
        rtol=0,
        atol=0,
    )
    assert prepared.fc1_weight_scale.shape == (2, 128, 1)
    assert prepared.fc2_weight_scale.shape == (2, 128, 1)
    assert tuple(block.state_dict()) == state_names
    assert not any("prepared" in name or "packed" in name for name in state_names)


def test_ep_nvfp4_pack_rejects_gate_up_dequant_scale_mismatch() -> None:
    from kairyu.models.moe_parallel import (
        EpMoeBlock,
        _pack_nvfp4_moe_block,
    )

    class NvFp4Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([_NvFp4Expert(2.0, 2.0, 3.0)])

        def _route(self, hidden: torch.Tensor):
            del hidden
            raise AssertionError("packing must not execute routing")

    block = EpMoeBlock(
        NvFp4Block(),
        _CopyCommunicator(),
        ep_rank=0,
        ep_size=1,
    )
    block.local_experts[0].gate_proj.weight_scale_2.fill_(2.0)

    with pytest.raises(RuntimeError, match="gate/up dequant scales differ"):
        _pack_nvfp4_moe_block(
            block,
            require_flashinfer_selection=False,
        )
