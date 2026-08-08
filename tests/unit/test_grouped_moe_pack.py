from __future__ import annotations

import torch
from torch import nn

from kairyu.models.grouped_moe import GroupedExpertPack
from kairyu.models.moe_parallel import EpMoeBlock


class _DenseExpert(nn.Module):
    def __init__(self, hidden: int = 16, intermediate: int = 24) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(_DenseExpert() for _ in range(4))


def test_grouped_expert_pack_preserves_canonical_names_without_weight_copy() -> None:
    module = _Experts().to(torch.bfloat16)
    names = tuple(module.state_dict())

    pack = GroupedExpertPack.create(module.experts)

    assert pack is not None
    assert pack.is_current()
    assert tuple(module.state_dict()) == names
    assert set(name for name, _value in module.named_parameters()) == set(names)

    gate_up = tuple(
        projection.weight
        for expert in module.experts
        for projection in (expert.gate_proj, expert.up_proj)
    )
    down = tuple(expert.down_proj.weight for expert in module.experts)
    assert len({weight.untyped_storage().data_ptr() for weight in gate_up}) == 1
    assert len({weight.untyped_storage().data_ptr() for weight in down}) == 1
    assert gate_up[0].untyped_storage().nbytes() == sum(
        weight.numel() * weight.element_size() for weight in gate_up
    )
    assert down[0].untyped_storage().nbytes() == sum(
        weight.numel() * weight.element_size() for weight in down
    )


def test_grouped_expert_pack_declines_replaced_canonical_parameter() -> None:
    module = _Experts().to(torch.bfloat16)
    pack = GroupedExpertPack.create(module.experts)
    assert pack is not None and pack.is_current()

    expert = module.experts[0]
    expert.gate_proj.weight = nn.Parameter(expert.gate_proj.weight.detach().clone())

    assert not pack.is_current()


def test_ep_repackages_only_rank_local_dense_experts() -> None:
    class Block(_Experts):
        def _route(self, hidden: torch.Tensor):
            indices = torch.zeros(hidden.shape[0], 1, dtype=torch.int64)
            weights = torch.ones(hidden.shape[0], 1, dtype=hidden.dtype)
            return indices, weights

    block = Block().to(torch.bfloat16)
    object.__setattr__(
        block,
        "_grouped_expert_pack",
        GroupedExpertPack.create(block.experts),
    )

    ep_block = EpMoeBlock(block, object(), ep_rank=0, ep_size=2)

    assert block._grouped_expert_pack is None
    assert ep_block._grouped_expert_pack is not None
    assert ep_block._grouped_expert_pack.is_current()
    local_gate_up = tuple(
        projection.weight
        for expert in ep_block.local_experts
        for projection in (expert.gate_proj, expert.up_proj)
    )
    assert local_gate_up[0].untyped_storage().nbytes() == sum(
        weight.numel() * weight.element_size() for weight in local_gate_up
    )
