from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from kairyu.models import moe_parallel


class _ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.scale


class _TinyMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([_ScaleExpert(2.0), _ScaleExpert(-1.0)])

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_ids = torch.arange(hidden.shape[0], device=hidden.device).remainder(2)
        weights = torch.ones(hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device)
        return expert_ids[:, None], weights

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        expert_ids, _ = self._route(hidden)
        out = torch.zeros_like(hidden)
        for expert_id, expert in enumerate(self.experts):
            mask = expert_ids[:, 0] == expert_id
            out[mask] = expert(hidden[mask])
        return out


class _CopyCommunicator:
    def __init__(self) -> None:
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def tensor_all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
    ) -> None:
        del output_split_sizes, input_split_sizes
        self.pairs.append((output, input_))
        output.copy_(input_)


def test_ep_forward_allocates_every_collective_buffer_on_payload_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-2.0, 5.0]])
    block = _TinyMoeBlock()
    reference = block(hidden)
    communicator = _CopyCommunicator()
    ep_block = moe_parallel.EpMoeBlock(block, communicator, ep_rank=0, ep_size=1)

    real_empty = torch.empty
    empty_calls: list[dict[str, Any]] = []

    def recording_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
        empty_calls.append(kwargs)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(moe_parallel.torch, "empty", recording_empty)

    actual = ep_block(hidden)

    assert len(empty_calls) == 4
    assert [call.get("device") for call in empty_calls] == [hidden.device] * 4
    assert len(communicator.pairs) == 4
    assert all(
        output.device == input_.device == hidden.device
        for output, input_ in communicator.pairs
    )
    torch.testing.assert_close(actual, reference)


def test_collective_device_guard_fails_before_communicator_call() -> None:
    guard = getattr(moe_parallel, "_assert_collective_device", None)
    assert callable(guard), "collective device guard must exist"
    communicator = _CopyCommunicator()
    payload = torch.empty(1, device="cpu")
    received = torch.empty(1, device="meta")

    with pytest.raises(ValueError) as exc_info:
        guard(payload.device, payload=payload, received=received)
        communicator.tensor_all_to_all_single(received, payload, [1], [1])

    message = str(exc_info.value)
    assert "payload=cpu" in message
    assert "received=meta" in message
    assert "expected cpu" in message
    assert communicator.pairs == []
