"""Grouped equal-shape collectives for the attention-DP NVFP4 path."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kairyu.engine.core import dist_comm
from kairyu.engine.core.dist_comm import TorchDistCommunicator
from kairyu.kernels import nvfp4_moe_gpu
from kairyu.models.moe_parallel import (
    EpMoeBlock,
    assert_attention_dp_layouts_consumed,
    prepare_attention_dp_layouts,
)


def test_tensor_all_gather_many_uses_direct_coalesced_rank_major_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = object.__new__(TorchDistCommunicator)
    communicator._group = object()
    communicator._device = torch.device("cuda:0")
    events: list[object] = []

    monkeypatch.setattr(dist_comm.dist, "get_world_size", lambda _group: 4)
    monkeypatch.setattr(dist_comm.dist, "get_backend", lambda _group: "nccl")

    @contextmanager
    def fake_coalescing_manager(*, group):
        events.append(("enter", group))
        yield
        events.append(("exit", group))

    def fake_all_gather_into_tensor(output, tensor, *, group) -> None:
        events.append(("gather", group, tensor.dtype, tuple(tensor.shape)))
        rows = tensor.shape[0]
        for rank in range(4):
            output[rank * rows : (rank + 1) * rows].copy_(tensor + rank)

    monkeypatch.setattr(
        dist_comm.dist,
        "_coalescing_manager",
        fake_coalescing_manager,
    )
    monkeypatch.setattr(
        dist_comm.dist,
        "all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    monkeypatch.setattr(
        dist_comm.dist,
        "all_gather",
        lambda *_args, **_kwargs: pytest.fail("shard-list all_gather was used"),
    )

    inputs = (
        torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8),
        torch.tensor([[5], [6]], dtype=torch.int32),
        torch.tensor([[0.25], [0.5]], dtype=torch.float32),
    )
    outputs = communicator.tensor_all_gather_many(inputs)

    assert events[0] == ("enter", communicator._group)
    assert events[-1] == ("exit", communicator._group)
    assert [event[0] for event in events].count("gather") == len(inputs)
    for source, output in zip(inputs, outputs, strict=True):
        assert tuple(output.shape) == (8, *source.shape[1:])
        expected = torch.cat([source + rank for rank in range(4)], dim=0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


class _Expert(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the test must use the fused MoE path")


class _Moe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_Expert() for _ in range(4)])
        self.shared_experts = None

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(hidden.shape[0], 1, dtype=torch.int64),
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
        )


class _GroupedModelComm:
    rank = 0
    world_size = 4

    def __init__(self) -> None:
        self.grouped_calls: list[tuple[tuple[torch.dtype, tuple[int, ...]], ...]] = []
        self.single_calls = 0

    def tensor_all_gather_many(
        self,
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        self.grouped_calls.append(
            tuple((tensor.dtype, tuple(tensor.shape)) for tensor in tensors)
        )
        return tuple(torch.cat([tensor] * self.world_size, dim=0) for tensor in tensors)

    def tensor_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        self.single_calls += 1
        raise AssertionError("grouped communicator used the single-tensor fallback")

    def tensor_reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[: tensor.shape[0] // self.world_size].contiguous()


class _RaggedDirectModelComm:
    rank = 0
    world_size = 4
    attention_dp_direct_nccl_active = True

    def __init__(self) -> None:
        self.gatherv_layouts: list[tuple[int, ...]] = []
        self.scatterv_layouts: list[tuple[int, ...]] = []

    def tensor_all_gatherv_many(self, tensors, row_sizes):
        self.gatherv_layouts.append(row_sizes)
        outputs = []
        for tensor in tensors:
            rows = [
                tensor[:1].expand(size, *tensor.shape[1:]).contiguous()
                for size in row_sizes
            ]
            outputs.append(torch.cat(rows, dim=0))
        return tuple(outputs)

    def tensor_reduce_scatterv(self, tensor, row_sizes):
        self.scatterv_layouts.append(row_sizes)
        return tensor[: row_sizes[self.rank]].contiguous()

    def tensor_all_gather_many(self, _tensors):
        raise AssertionError("ragged direct path used equal all-gather")

    def tensor_all_gather(self, _tensor):
        raise AssertionError("ragged direct path used single all-gather")

    def tensor_reduce_scatter(self, _tensor):
        raise AssertionError("ragged direct path used equal reduce-scatter")


class _FallbackModelComm:
    rank = 0
    world_size = 4

    def __init__(self) -> None:
        self.single_calls = 0

    def tensor_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        self.single_calls += 1
        return torch.cat([tensor] * self.world_size, dim=0)

    def tensor_reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[: tensor.shape[0] // self.world_size].contiguous()


def _run_attention_dp_block(
    monkeypatch: pytest.MonkeyPatch,
    comm,
    *,
    layout: tuple[int, int, int, int] = (2, 2, 2, 2),
) -> torch.Tensor:
    monkeypatch.setattr(
        nvfp4_moe_gpu,
        "quantize_moe_input",
        lambda hidden, _scale: (
            hidden[:, : hidden.shape[1] // 2].to(torch.uint8).contiguous(),
            torch.zeros(
                hidden.shape[0],
                hidden.shape[1] // 16,
                dtype=torch.uint8,
            ),
        ),
    )
    monkeypatch.setattr(
        nvfp4_moe_gpu,
        "interleave_moe_input_scales",
        lambda scales: scales.reshape(-1).contiguous(),
    )

    def fake_fused(
        packed_hidden,
        input_sf,
        selected_experts,
        routing_weights,
        prepared,
        *,
        output,
        ep_size,
        ep_rank,
    ) -> torch.Tensor:
        ragged = (
            getattr(comm, "attention_dp_direct_nccl_active", False) is True
            and len(set(layout)) > 1
        )
        global_rows = sum(layout) if ragged else max(layout) * 4
        assert packed_hidden.shape == (global_rows, 8)
        assert input_sf.shape == (global_rows,)
        assert selected_experts.shape == (global_rows, 1)
        assert routing_weights.shape == (global_rows, 1)
        assert prepared.local_expert_start == ep_rank == 0
        assert ep_size == 4
        output.fill_(7)
        return output

    monkeypatch.setattr(
        nvfp4_moe_gpu,
        "fused_moe_forward_quantized",
        fake_fused,
    )

    block = EpMoeBlock(_Moe(), comm, ep_rank=0, ep_size=4, attention_dp=True)
    if getattr(comm, "attention_dp_direct_nccl_active", False) is True:
        object.__setattr__(
            block,
            "_attention_dp_local_partial",
            lambda *, rows, hidden_size, dtype, device: torch.empty(
                rows,
                hidden_size,
                dtype=dtype,
                device=device,
            ),
        )
    object.__setattr__(
        block,
        "_prepared_nvfp4_moe",
        SimpleNamespace(
            fc1_act_scale=torch.tensor(1.0, dtype=torch.float32),
            fc2_weight=torch.empty(1, 16, 1),
            num_global_experts=4,
            local_expert_start=0,
        ),
    )
    model = nn.ModuleList([block])
    prepare_attention_dp_layouts(model, (layout,))
    output = block(torch.ones(layout[0], 16, dtype=torch.bfloat16))
    assert_attention_dp_layouts_consumed(model)
    comm.last_attention_dp_metadata = block.attention_dp_metadata()
    return output


def test_attention_dp_prefers_grouped_all_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = _GroupedModelComm()

    output = _run_attention_dp_block(monkeypatch, comm)

    assert len(comm.grouped_calls) == 1
    assert comm.grouped_calls[0] == (
        (torch.uint8, (2, 8)),
        (torch.uint8, (2, 1)),
        (torch.int32, (2, 1)),
        (torch.float32, (2, 1)),
    )
    assert comm.single_calls == 0
    torch.testing.assert_close(output, torch.full_like(output, 7), rtol=0, atol=0)


def test_attention_dp_retains_single_tensor_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = _FallbackModelComm()

    output = _run_attention_dp_block(monkeypatch, comm)

    assert comm.single_calls == 4
    torch.testing.assert_close(output, torch.full_like(output, 7), rtol=0, atol=0)


def test_attention_dp_uses_ragged_direct_collectives_only_for_unequal_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = _RaggedDirectModelComm()
    layout = (2, 1, 1, 3)

    output = _run_attention_dp_block(monkeypatch, comm, layout=layout)

    assert comm.gatherv_layouts == [layout]
    assert comm.scatterv_layouts == [layout]
    assert output.shape == (2, 16)
    assert comm.last_attention_dp_metadata["dispatcher"] == (
        "nvfp4_allgatherv_reduce_scatterv"
    )
    assert comm.last_attention_dp_metadata["row_layout"] == "rank_major_ragged"
    assert comm.last_attention_dp_metadata["activation_collective"] == (
        "tensor_all_gatherv_many"
    )
    assert comm.last_attention_dp_metadata["combine_collective"] == (
        "tensor_reduce_scatterv"
    )
    assert comm.last_attention_dp_metadata["last_padded_rows_per_rank"] is None
    assert comm.last_attention_dp_metadata["total_collective_rows"] == sum(layout)
    torch.testing.assert_close(output, torch.full_like(output, 7), rtol=0, atol=0)
