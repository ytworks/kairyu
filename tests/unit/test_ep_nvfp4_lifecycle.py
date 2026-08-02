"""Lifecycle safety for derived expert-parallel NVFP4 fused operands."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kairyu.kernels import nvfp4_moe_gpu
from kairyu.models.moe_parallel import EpMoeBlock, _pack_nvfp4_moe_block
from kairyu.quant.linear import NvFp4Linear


class _NvFp4Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = NvFp4Linear(16, 16, False)
        self.up_proj = NvFp4Linear(16, 16, False)
        self.down_proj = NvFp4Linear(16, 16, False)
        self.gate_proj.input_scale.fill_(2.0)
        self.up_proj.input_scale.fill_(2.0)
        self.down_proj.input_scale.fill_(3.0)


class _SharedExpert(nn.Module):
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        super().__init__()
        self._events = events
        self._fail = fail

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self._events.append("shared")
        if self._fail:
            raise RuntimeError("shared failed")
        return torch.zeros_like(hidden)


class _NvFp4Block(nn.Module):
    def __init__(self, *, shared_experts: nn.Module | None = None) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_NvFp4Expert()])
        self.shared_experts = shared_experts

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (hidden.shape[0], 1)
        return (
            torch.zeros(shape, dtype=torch.int64, device=hidden.device),
            torch.ones(shape, dtype=torch.float32, device=hidden.device),
        )


class _Communicator:
    def __init__(self, events: list[str] | None = None, *, fail: bool = False) -> None:
        self._events = events
        self._fail = fail

    def tensor_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._events is not None:
            self._events.append("all_reduce")
        if self._fail:
            raise RuntimeError("all_reduce failed")
        return tensor


def _prepared_block(
    *,
    communicator: _Communicator | None = None,
    shared_experts: nn.Module | None = None,
) -> tuple[EpMoeBlock, nvfp4_moe_gpu.PreparedNvFp4Moe]:
    block = EpMoeBlock(
        _NvFp4Block(shared_experts=shared_experts),
        communicator or _Communicator(),
        ep_rank=0,
        ep_size=1,
    )
    prepared = _pack_nvfp4_moe_block(
        block,
        require_flashinfer_selection=False,
    )
    object.__setattr__(block, "_prepared_nvfp4_moe", prepared)
    return block, prepared


def test_parent_apply_invalidates_prepared_storage_before_replacing_buffers() -> None:
    block, prepared = _prepared_block()
    parent = nn.Module()
    parent.add_module("moe", block)
    state_names = tuple(parent.state_dict())
    old_fc1_storage = prepared.fc1_weight.untyped_storage().data_ptr()
    old_fc2_storage = prepared.fc2_weight.untyped_storage().data_ptr()
    object.__setattr__(block, "_fused_nvfp4_executed", True)

    parent._apply(lambda tensor: tensor.clone())

    expert = block.local_experts[0]
    assert block._prepared_nvfp4_moe is None
    assert block.fused_nvfp4_metadata() is None
    assert block._fused_nvfp4_executed is False
    assert expert.up_proj.weight.untyped_storage().data_ptr() != old_fc1_storage
    assert expert.gate_proj.weight.untyped_storage().data_ptr() != old_fc1_storage
    assert expert.down_proj.weight.untyped_storage().data_ptr() != old_fc2_storage
    assert tuple(parent.state_dict()) == state_names


def test_parent_to_device_invalidates_cpu_prepared_operands() -> None:
    block, prepared = _prepared_block()
    parent = nn.Module()
    parent.add_module("moe", block)

    parent.to(device="meta")

    assert prepared.fc1_weight.device.type == "cpu"
    assert prepared.fc1_act_scale.device.type == "cpu"
    assert block._prepared_nvfp4_moe is None
    assert block.fused_nvfp4_metadata() is None
    assert all(tensor.device.type == "meta" for tensor in block.buffers())


def test_parent_load_state_dict_invalidates_and_repack_uses_new_scales() -> None:
    block, old_prepared = _prepared_block()
    parent = nn.Module()
    parent.add_module("moe", block)
    state = {
        name: tensor.detach().clone()
        for name, tensor in parent.state_dict().items()
    }
    for name, tensor in state.items():
        if name.endswith(("gate_proj.input_scale", "up_proj.input_scale")):
            tensor.fill_(4.0)
        elif name.endswith("down_proj.input_scale"):
            tensor.fill_(5.0)
        elif name.endswith(("gate_proj.weight_scale_2", "up_proj.weight_scale_2")):
            tensor.fill_(3.0)
        elif name.endswith("down_proj.weight_scale_2"):
            tensor.fill_(7.0)

    parent.load_state_dict(state, strict=True)

    assert block._prepared_nvfp4_moe is None
    assert block.fused_nvfp4_metadata() is None
    torch.testing.assert_close(old_prepared.fc1_act_scale, torch.tensor(0.5))
    torch.testing.assert_close(old_prepared.fc2_act_scale, torch.tensor(1.0 / 3.0))

    new_prepared = _pack_nvfp4_moe_block(
        block,
        require_flashinfer_selection=False,
    )
    torch.testing.assert_close(new_prepared.fc1_act_scale, torch.tensor(0.25))
    torch.testing.assert_close(new_prepared.fc1_dequant_scale, torch.tensor([12.0]))
    torch.testing.assert_close(new_prepared.fc2_act_scale, torch.tensor(0.2))
    torch.testing.assert_close(new_prepared.fc2_dequant_scale, torch.tensor([35.0]))
    assert not any("prepared" in name for name in parent.state_dict())


@pytest.mark.parametrize(
    ("failure_stage", "expected_events"),
    [
        ("kernel", ["kernel"]),
        ("all_reduce", ["kernel", "all_reduce"]),
        ("shared", ["kernel", "all_reduce", "shared"]),
        (None, ["kernel", "all_reduce", "shared"]),
    ],
)
def test_success_marker_requires_the_complete_fused_forward(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str | None,
    expected_events: list[str],
) -> None:
    events: list[str] = []
    shared = _SharedExpert(events, fail=failure_stage == "shared")
    communicator = _Communicator(events, fail=failure_stage == "all_reduce")
    block, _prepared = _prepared_block(
        communicator=communicator,
        shared_experts=shared,
    )
    object.__setattr__(block, "_fused_nvfp4_executed", True)

    def fake_fused_forward(
        hidden: torch.Tensor,
        _selected_experts: torch.Tensor,
        _routing_weights: torch.Tensor,
        _prepared: nvfp4_moe_gpu.PreparedNvFp4Moe,
        *,
        output: torch.Tensor,
        ep_size: int,
        ep_rank: int,
    ) -> torch.Tensor:
        del ep_size, ep_rank
        events.append("kernel")
        if failure_stage == "kernel":
            raise RuntimeError("kernel failed")
        output.copy_(hidden)
        return output

    monkeypatch.setattr(nvfp4_moe_gpu, "fused_moe_forward", fake_fused_forward)
    hidden = torch.ones(2, 16, dtype=torch.bfloat16)

    def invoke() -> torch.Tensor:
        return block(hidden)

    if failure_stage is None:
        torch.testing.assert_close(invoke(), hidden)
    else:
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            invoke()

    assert events == expected_events
    metadata = block.fused_nvfp4_metadata()
    assert metadata is not None
    assert metadata["successful_forward"] is (failure_stage is None)
