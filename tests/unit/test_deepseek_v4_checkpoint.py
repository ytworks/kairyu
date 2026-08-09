from __future__ import annotations

import pytest

from kairyu.models.deepseek_v4_checkpoint import (
    DeepseekV4Placement,
    build_deepseek_v4_checkpoint_plan,
)


def _names() -> list[str]:
    names = [
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
        "layers.0.ffn.gate.weight",
        "layers.0.ffn.shared_experts.w1.weight",
        "mtp.0.main_proj.weight",
        "mtp.0.main_proj.scale",
    ]
    for stage in ("layers.0", "mtp.0"):
        for expert in range(4):
            for projection in ("w1", "w2", "w3"):
                names.append(f"{stage}.ffn.experts.{expert}.{projection}.weight")
                names.append(f"{stage}.ffn.experts.{expert}.{projection}.scale")
    return names


def _specs(names: list[str]):
    specs = {}
    for name in names:
        if ".experts." in name and name.endswith(".weight"):
            specs[name] = ((8, 16), "I8")
        elif ".experts." in name and name.endswith(".scale"):
            specs[name] = ((8, 1), "F8_E8M0")
        elif name in {
            "layers.0.attn.wq_a.weight",
            "mtp.0.main_proj.weight",
        }:
            specs[name] = ((128, 128), "F8_E4M3")
        elif name in {
            "layers.0.attn.wq_a.scale",
            "mtp.0.main_proj.scale",
        }:
            specs[name] = ((1, 1), "F8_E8M0")
        else:
            specs[name] = ((8,), "BF16")
    return specs


@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_checkpoint_plan_assigns_every_expert_once(ep_size: int) -> None:
    names = _names()
    selected_by_rank = []
    for rank in range(ep_size):
        plan = build_deepseek_v4_checkpoint_plan(
            names,
            num_hidden_layers=1,
            num_experts=4,
            ep_size=ep_size,
            ep_rank=rank,
        )
        plan.validate_specs(_specs(names))
        selected_by_rank.append(set(plan.selected_names))
        assert plan.mtp_layer_count == 1
        assert len(plan.owned_expert_range) == 4 // ep_size

    expert_names = {name for name in names if ".experts." in name}
    replicated = {name for name in names if name not in expert_names}
    assert set.union(*selected_by_rank) == set(names)
    for name in expert_names:
        assert sum(name in selected for selected in selected_by_rank) == 1
    for name in replicated:
        assert all(name in selected for selected in selected_by_rank)


def test_checkpoint_plan_reports_placement_and_resident_bytes() -> None:
    names = _names()
    plan = build_deepseek_v4_checkpoint_plan(
        names,
        num_hidden_layers=1,
        num_experts=4,
        ep_size=2,
        ep_rank=1,
    )
    expert = next(item for item in plan.tensor_plans if item.expert_index == 3)
    assert expert.placement is DeepseekV4Placement.RANK_LOCAL_EXPERT
    assert "layers.0.ffn.experts.0.w1.weight" in plan.remote_names
    assert plan.resident_bytes(_specs(names)) > 0


def test_checkpoint_plan_rejects_unknown_and_bad_fp4_scale() -> None:
    with pytest.raises(ValueError, match="unknown DeepSeek V4"):
        build_deepseek_v4_checkpoint_plan(
            ["layers.0.attn.not_a_real_tensor"],
            num_hidden_layers=1,
            num_experts=4,
            ep_size=1,
            ep_rank=0,
        )

    names = _names()
    plan = build_deepseek_v4_checkpoint_plan(
        names,
        num_hidden_layers=1,
        num_experts=4,
        ep_size=1,
        ep_rank=0,
    )
    specs = _specs(names)
    specs["layers.0.ffn.experts.0.w1.scale"] = ((8, 2), "F8_E8M0")
    with pytest.raises(ValueError, match="packed FP4 scale"):
        plan.validate_specs(specs)
