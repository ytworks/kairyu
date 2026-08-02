"""Reference-math attention output sharding for NVFP4 expert parallelism."""

from __future__ import annotations

import pytest
import torch

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.models.parallel import ReplicatedInputRowParallelLinear
from kairyu.quant import nvfp4
from kairyu.quant.linear import (
    LinearRole,
    NvFp4Linear,
    linear_factory,
    make_linear,
)


class _PeerReduce:
    rank = 0
    world_size = 2

    def __init__(self, peer_partial: torch.Tensor) -> None:
        self.peer_partial = peer_partial
        self.seen: torch.Tensor | None = None

    def tensor_all_reduce(self, partial: torch.Tensor) -> torch.Tensor:
        self.seen = partial.detach().clone()
        return partial + self.peer_partial


def _local_projection(
    *,
    rank: int,
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
) -> NvFp4Linear:
    factory = linear_factory(
        QuantConfig(QuantMethod.NVFP4, weight_bits=4, group_size=16),
        dtype=torch.bfloat16,
        tensor_parallel_size=2,
        tensor_parallel_rank=rank,
    )
    module = make_linear(
        factory,
        16,
        packed.shape[0],
        False,
        qualified_name="model.layers.0.self_attn.o_proj",
        role=LinearRole.ATTENTION_OUTPUT,
        layer_index=0,
        shard_dim=1,
    )
    assert isinstance(module, NvFp4Linear)
    with torch.no_grad():
        module.weight.copy_(packed[:, rank * 8 : (rank + 1) * 8])
        module.weight_scale.copy_(block_scale[:, rank : rank + 1])
        module.weight_scale_2.copy_(global_weight_scale)
        module.input_scale.copy_(input_scale)
    return module


def test_replicated_attention_nvfp4_materializes_bf16_rank_partials() -> None:
    torch.manual_seed(166)
    full_weight = torch.randn(7, 32, dtype=torch.float32) * 0.25
    packed, block_scale, global_weight_scale = nvfp4.quantize_nvfp4(full_weight)
    hidden = (torch.randn(5, 32, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    input_scale = (
        hidden.float().abs().amax() / (nvfp4.FP4_MAX * 448.0)
    ).clamp(min=1e-12)
    rank0 = _local_projection(
        rank=0,
        packed=packed,
        block_scale=block_scale,
        global_weight_scale=global_weight_scale,
        input_scale=input_scale,
    )
    rank1 = _local_projection(
        rank=1,
        packed=packed,
        block_scale=block_scale,
        global_weight_scale=global_weight_scale,
        input_scale=input_scale,
    )

    local0 = rank0(hidden[:, :16])
    local1 = rank1(hidden[:, 16:])
    assert local0.dtype is torch.bfloat16
    assert local1.dtype is torch.bfloat16
    expected = local0 + local1
    before_state = tuple(rank0.state_dict())

    comm = _PeerReduce(local1)
    bound = ReplicatedInputRowParallelLinear(rank0, comm)
    assert bound._kairyu_row_parallel_executed is False
    assert bound._kairyu_row_parallel_partial_dtype is None
    actual = bound(hidden)

    assert bound._kairyu_row_parallel_executed is True
    assert bound._kairyu_row_parallel_partial_dtype is torch.bfloat16
    assert bound is rank0
    assert tuple(bound.state_dict()) == before_state
    assert torch.equal(comm.seen, local0)
    assert torch.equal(actual, expected)
    assert bound.linear_context.tensor_parallel.local_in_features == 16
    assert bound.linear_context.tensor_parallel.global_in_features == 32


def test_replicated_attention_row_shard_fails_closed_on_wrong_topology() -> None:
    packed, block_scale, global_weight_scale = nvfp4.quantize_nvfp4(
        torch.ones(4, 32)
    )
    module = _local_projection(
        rank=0,
        packed=packed,
        block_scale=block_scale,
        global_weight_scale=global_weight_scale,
        input_scale=torch.tensor(1.0),
    )

    class _WrongRank:
        rank = 1
        world_size = 2

    with pytest.raises(ValueError, match="differs from communicator"):
        ReplicatedInputRowParallelLinear(module, _WrongRank())
