"""Model-side contracts for the opt-in Qwen3-235B attention-DP dispatcher."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kairyu.engine.core.comm import FakeCommunicator
from kairyu.kernels import nvfp4_moe_gpu
from kairyu.models.moe_parallel import (
    EpMoeBlock,
    assert_attention_dp_layouts_consumed,
    prepare_attention_dp_layouts,
)


class _Expert(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the attention-DP test must use the fused dispatcher")


class _SharedExpert(nn.Module):
    def __init__(self, seen_shapes: list[tuple[int, ...]]) -> None:
        super().__init__()
        self._seen_shapes = seen_shapes

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self._seen_shapes.append(tuple(hidden.shape))
        return hidden


class _MoeBlock(nn.Module):
    def __init__(self, seen_shapes: list[tuple[int, ...]]) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_Expert() for _ in range(4)])
        self.shared_experts = _SharedExpert(seen_shapes)

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected = hidden[:, 0].to(torch.int64).remainder(4)
        return selected[:, None], torch.ones(
            hidden.shape[0],
            1,
            dtype=torch.float32,
            device=hidden.device,
        )


class _TensorFakeComm:
    """Add tensor reduce-scatter to the repository's threaded FakeCommunicator."""

    def __init__(self, base: FakeCommunicator) -> None:
        self._base = base

    @property
    def rank(self) -> int:
        return self._base.rank

    @property
    def world_size(self) -> int:
        return self._base.world_size

    def tensor_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._base.tensor_all_gather(tensor)

    def tensor_reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = self._base.tensor_all_gather(tensor)
        contributions = gathered.reshape(self.world_size, *tensor.shape)
        reduced = contributions.sum(dim=0)
        span = tensor.shape[0] // self.world_size
        start = self.rank * span
        return reduced[start : start + span].contiguous()


class _TwoSparseLayers(nn.Module):
    def __init__(self, blocks: tuple[EpMoeBlock, EpMoeBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(blocks)


def _prepared(rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        fc1_act_scale=torch.tensor(1.0, dtype=torch.float32),
        fc2_weight=torch.empty(1, 16, 1),
        num_global_experts=4,
        local_expert_start=rank,
    )


def _attention_dp_block(
    comm: _TensorFakeComm,
    seen_shapes: list[tuple[int, ...]],
) -> EpMoeBlock:
    block = EpMoeBlock(
        _MoeBlock(seen_shapes),
        comm,
        ep_rank=comm.rank,
        ep_size=comm.world_size,
        attention_dp=True,
    )
    object.__setattr__(block, "_prepared_nvfp4_moe", _prepared(comm.rank))
    return block


def _hidden(rows: tuple[tuple[int, int], ...]) -> torch.Tensor:
    result = torch.zeros(len(rows), 16, dtype=torch.bfloat16)
    for row, (expert, payload) in enumerate(rows):
        result[row, 0] = expert
        result[row, 1] = payload
    return result


def test_attention_dp_gathers_packed_rows_and_reduce_scatters_rank_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quantized_shapes: list[tuple[int, ...]] = []

    def fake_quantize(
        hidden: torch.Tensor,
        _input_global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_shapes.append(tuple(hidden.shape))
        return (
            hidden[:, : hidden.shape[1] // 2].to(torch.uint8).contiguous(),
            torch.zeros(
                hidden.shape[0],
                hidden.shape[1] // 16,
                dtype=torch.uint8,
                device=hidden.device,
            ),
        )

    def fake_interleave(scale_rows: torch.Tensor) -> torch.Tensor:
        return scale_rows.reshape(-1).contiguous()

    def fake_fused(
        packed_hidden: torch.Tensor,
        _input_sf: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        prepared: SimpleNamespace,
        *,
        output: torch.Tensor,
        ep_size: int,
        ep_rank: int,
    ) -> torch.Tensor:
        assert ep_size == 4
        assert ep_rank == prepared.local_expert_start
        selected = selected_experts[:, 0] == ep_rank
        values = (
            packed_hidden[:, 1].to(torch.bfloat16)
            * float(ep_rank + 1)
            * routing_weights[:, 0].to(torch.bfloat16)
        )
        output.zero_()
        output[selected] = values[selected, None]
        return output

    monkeypatch.setattr(nvfp4_moe_gpu, "quantize_moe_input", fake_quantize)
    monkeypatch.setattr(
        nvfp4_moe_gpu,
        "interleave_moe_input_scales",
        fake_interleave,
    )
    monkeypatch.setattr(
        nvfp4_moe_gpu,
        "fused_moe_forward_quantized",
        fake_fused,
    )

    bases = FakeCommunicator.create_group(4)
    seen_shapes = [[] for _ in range(4)]
    models = []
    for rank, base in enumerate(bases):
        comm = _TensorFakeComm(base)
        models.append(
            _TwoSparseLayers(
                (
                    _attention_dp_block(comm, seen_shapes[rank]),
                    _attention_dp_block(comm, seen_shapes[rank]),
                )
            )
        )

    layouts = ((2, 1, 3, 2), (1, 2, 1, 3))
    rows = (
        (
            ((0, 2), (1, 3)),
            ((2, 4),),
            ((3, 5), (0, 6), (2, 7)),
            ((1, 8), (3, 9)),
        ),
        (
            ((3, 2),),
            ((0, 3), (2, 4)),
            ((1, 5),),
            ((2, 6), (3, 7), (0, 8)),
        ),
    )
    for model in models:
        prepare_attention_dp_layouts(model, layouts)

    def run_rank(rank: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        outputs = []
        for forward_index in range(len(layouts)):
            hidden = _hidden(rows[forward_index][rank])
            outputs.append(
                tuple(layer(hidden) for layer in models[rank].layers)
            )
        assert_attention_dp_layouts_consumed(models[rank])
        return tuple(outputs)

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs_by_rank = tuple(executor.map(run_rank, range(4)))

    assert quantized_shapes == [(3, 16)] * 16
    for rank, rank_outputs in enumerate(outputs_by_rank):
        expected_shared_shapes = []
        for forward_index, layer_outputs in enumerate(rank_outputs):
            hidden = _hidden(rows[forward_index][rank])
            routed = torch.empty_like(hidden)
            for row, (expert, payload) in enumerate(rows[forward_index][rank]):
                routed[row].fill_(payload * (expert + 1))
            expected = routed + hidden
            for output in layer_outputs:
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                expected_shared_shapes.append(tuple(hidden.shape))
        assert seen_shapes[rank] == expected_shared_shapes
        for block in models[rank].layers:
            metadata = block.fused_nvfp4_metadata()
            assert metadata is not None
            assert metadata["dispatcher"] == "nvfp4_allgather_reduce_scatter"
            assert metadata["attention_output_placement"] == "replicated"
            assert metadata["last_rank_row_layout"] == layouts[-1]
            assert metadata["completed_layout_steps"] == 1
            assert metadata["successful_forward"] is True


class _ShapeOnlyComm:
    rank = 0
    world_size = 4

    def tensor_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def tensor_reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def test_attention_dp_layout_queue_fails_closed_on_zero_drift_and_lifecycle() -> None:
    model = _TwoSparseLayers(
        (
            _attention_dp_block(_ShapeOnlyComm(), []),
            _attention_dp_block(_ShapeOnlyComm(), []),
        )
    )

    with pytest.raises(ValueError, match="positive exact integer"):
        prepare_attention_dp_layouts(model, ((1, 0, 1, 1),))
    with pytest.raises(ValueError, match="length 4"):
        prepare_attention_dp_layouts(model, ((1, 1),))
    with pytest.raises(ValueError, match="non-empty tuple"):
        prepare_attention_dp_layouts(model, [(1, 1, 1, 1)])

    prepare_attention_dp_layouts(model, ((1, 1, 1, 1),))
    object.__setattr__(model.layers[0], "_fused_nvfp4_executed", True)
    with pytest.raises(ValueError, match="local row layout drift"):
        model.layers[0](torch.zeros(2, 16, dtype=torch.bfloat16))
    assert model.layers[0].fused_nvfp4_metadata()["successful_forward"] is False
    with pytest.raises(RuntimeError, match="not fully consumed"):
        assert_attention_dp_layouts_consumed(model)
    with pytest.raises(RuntimeError, match="already active"):
        prepare_attention_dp_layouts(model, ((1, 1, 1, 1),))


def test_attention_dp_rejects_unprepared_or_non_nvfp4_execution() -> None:
    comm = _ShapeOnlyComm()
    block = EpMoeBlock(
        _MoeBlock([]),
        comm,
        ep_rank=0,
        ep_size=4,
        attention_dp=True,
    )
    model = nn.ModuleList([block])
    prepare_attention_dp_layouts(model, ((1, 1, 1, 1),))

    with pytest.raises(RuntimeError, match="prepared fused NVFP4"):
        block(torch.zeros(1, 16, dtype=torch.bfloat16))


def test_default_ep_block_has_no_attention_dp_collective_requirements() -> None:
    class LegacyComm:
        pass

    block = EpMoeBlock(
        _MoeBlock([]),
        LegacyComm(),
        ep_rank=0,
        ep_size=1,
    )

    assert block.attention_dp is False
    assert block.attention_dp_metadata()["dispatcher"] is None
