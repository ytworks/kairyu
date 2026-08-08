"""CUDA proof for sort-by-expert grouped GEMM and fixed-capacity EP dispatch."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kairyu.models.grouped_moe import GroupedExpertPack
from kairyu.models.moe import _mix_experts
from kairyu.models.moe_parallel import EpMoeBlock

pytestmark = pytest.mark.gpu


class _DenseExpert(nn.Module):
    def __init__(self, hidden: int = 32, intermediate: int = 48) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


class _FixedRouteBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(_DenseExpert() for _ in range(4))

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        first = rows.remainder(4)
        second = (rows * 3 + 1).remainder(4)
        indices = torch.stack((first, second), dim=1)
        weights = torch.empty(
            hidden.shape[0],
            2,
            dtype=torch.float32,
            device=hidden.device,
        )
        weights[:, 0] = 0.625
        weights[:, 1] = 0.375
        return indices, weights


class _CopyCommunicator:
    def __init__(self) -> None:
        self.splits: list[tuple[list[int], list[int]]] = []

    def tensor_all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
    ) -> None:
        self.splits.append((output_split_sizes, input_split_sizes))
        output.copy_(input_)

    def tensor_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("dense grouped EP must not all-reduce")


def _reference(
    hidden: torch.Tensor,
    experts: nn.ModuleList,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for expert_index, expert in enumerate(experts):
        token, slot = (indices == expert_index).nonzero(as_tuple=True)
        output.index_add_(
            0,
            token,
            expert(hidden[token]).float() * weights[token, slot, None],
        )
    return output.to(hidden.dtype)


def test_local_grouped_moe_uses_two_grouped_gemms_without_host_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flashinfer

    torch.manual_seed(11)
    block = _FixedRouteBlock().to(device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(13, 32, device="cuda", dtype=torch.bfloat16)
    indices, weights = block._route(hidden)
    expected = _reference(hidden, block.experts, indices, weights)
    pack = GroupedExpertPack.create(block.experts)
    assert pack is not None
    # Resolve the FlashInfer/cuDNN plan outside the measured hot path.
    _mix_experts(hidden, block.experts, indices, weights, pack)
    torch.cuda.synchronize()
    grouped_calls = 0
    grouped_mm = flashinfer.grouped_mm_bf16

    def recording_grouped_mm(*args, **kwargs):
        nonlocal grouped_calls
        grouped_calls += 1
        return grouped_mm(*args, **kwargs)

    monkeypatch.setattr(flashinfer, "grouped_mm_bf16", recording_grouped_mm)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        acc_events=True,
    ) as profile:
        actual = _mix_experts(hidden, block.experts, indices, weights, pack)
        torch.cuda.synchronize()

    keys = {event.key for event in profile.key_averages()}
    assert grouped_calls == 2
    assert any("grouped" in key.lower() or "cudnn" in key.lower() for key in keys)
    scalar_events = [
        event
        for event in profile.events()
        if event.key == "aten::_local_scalar_dense"
    ]
    assert not scalar_events, [event.stack for event in scalar_events]
    assert "aten::nonzero" not in keys
    assert not any("unique" in key for key in keys)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


def test_ep_grouped_moe_uses_fixed_splits_and_is_cuda_graph_capturable() -> None:
    torch.manual_seed(19)
    block = _FixedRouteBlock().to(device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
    indices, weights = block._route(hidden)
    expected = _reference(hidden, block.experts, indices, weights)
    communicator = _CopyCommunicator()
    ep_block = EpMoeBlock(block, communicator, ep_rank=0, ep_size=1)

    actual = ep_block(hidden)

    capacity = hidden.shape[0] * indices.shape[1]
    assert communicator.splits == [([capacity], [capacity])] * 3
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)

    communicator.splits.clear()
    static_hidden = hidden.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = ep_block(static_hidden)
    static_hidden.copy_(hidden * 0.5)
    graph.replay()
    torch.cuda.synchronize()

    replay_indices, replay_weights = block._route(static_hidden)
    replay_expected = _reference(
        static_hidden,
        block.experts,
        replay_indices,
        replay_weights,
    )
    torch.testing.assert_close(captured, replay_expected, rtol=1e-2, atol=2e-2)
