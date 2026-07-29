"""Focused CPU contract tests for TP rank-0 sampling ownership (#225)."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core.model_runner import (
    PagedModelRunner,
    _DeferredStepOutput,
    _PendingDeviceToken,
)
from kairyu.engine.core.sampler import DeviceSample
from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.spec_runner import _OverlayState
from kairyu.engine.core.step_input import StateSync


def _state(
    request_id: str,
    *,
    prompt: tuple[int, ...] = (1, 2, 3, 4),
    computed_prompt: int | None = None,
    outputs: tuple[int, ...] = (),
    pages: tuple[int, ...] = (0,),
) -> SimpleNamespace:
    if computed_prompt is None:
        computed_prompt = len(prompt)
    request = SimpleNamespace(
        request_id=request_id,
        sampling_identity=request_id,
        prompt_token_ids=prompt,
        sampling=EngineSampling(),
        eos_token_id=None,
        max_new_tokens=16,
    )
    return SimpleNamespace(
        request=request,
        allocation=SimpleNamespace(pages=pages, num_cached_tokens=0),
        decode_pages=(),
        computed_prompt=computed_prompt,
        prefill_done=computed_prompt >= len(prompt),
        outputs=outputs,
        output_epoch=0,
        in_flight=0,
    )


def _packet_runner() -> PagedModelRunner:
    runner = object.__new__(PagedModelRunner)
    runner._device = torch.device("cpu")
    runner._future_tokens = {}
    runner._future_device_tokens = {}
    return runner


def _mixed_step():
    chunks = (
        ScheduledChunk("partial", 2, True, 0),
        ScheduledChunk("final", 2, True, 0),
        ScheduledChunk("decode", 1, False, 3),
    )
    states = {
        "partial": _state("partial", computed_prompt=2),
        "final": _state("final"),
        "decode": _state("decode", outputs=(7, 8, 9)),
    }
    return chunks, states


def test_sampling_packet_has_one_fixed_slot_per_scheduled_chunk():
    runner = _packet_runner()
    chunks, states = _mixed_step()

    packet = runner.make_sampling_token_packet(
        chunks,
        states,
        {
            "decode": (SampledToken(29),),
            "final": (SampledToken(17),),
        },
    )
    follower_buffer = runner.make_sampling_token_packet(chunks, states)

    assert packet.dtype == torch.int64
    assert packet.tolist() == [-1, 17, 29]
    assert follower_buffer.tolist() == [-1, -1, -1]


@pytest.mark.parametrize(
    ("sampled", "missing", "extra"),
    [
        ({"final": (SampledToken(17),)}, "decode", ""),
        (
            {
                "final": (SampledToken(17),),
                "decode": (SampledToken(29),),
                "partial": (SampledToken(31),),
            },
            "",
            "partial",
        ),
    ],
)
def test_sampling_packet_rejects_missing_or_extra_owner_outputs(
    sampled, missing, extra
):
    runner = _packet_runner()
    chunks, states = _mixed_step()

    with pytest.raises(RuntimeError, match="does not match.*layout") as error:
        runner.make_sampling_token_packet(chunks, states, sampled)

    assert missing in str(error.value)
    assert extra in str(error.value)


@pytest.mark.parametrize("values", [(), (SampledToken(1), SampledToken(2))])
def test_sampling_packet_requires_exactly_one_token_per_emitting_chunk(values):
    runner = _packet_runner()
    chunks, states = _mixed_step()

    with pytest.raises(RuntimeError, match="must emit exactly one token"):
        runner.make_sampling_token_packet(
            chunks,
            states,
            {"final": values, "decode": (SampledToken(29),)},
        )


def test_sampling_packet_rejects_duplicate_emitting_slots_for_one_request():
    runner = _packet_runner()
    state = _state("decode", outputs=(7, 8))
    chunks = (
        ScheduledChunk("decode", 1, False, 2),
        ScheduledChunk("decode", 1, False, 3),
    )

    with pytest.raises(RuntimeError, match="more than one emitted token"):
        runner.make_sampling_token_packet(
            chunks, {"decode": state}, {"decode": (SampledToken(9),)}
        )


def test_sampling_packet_reads_deferred_device_id_without_resolving_output():
    runner = _packet_runner()
    chunk = ScheduledChunk("decode", 1, False, 3)
    state = _state("decode", outputs=(7, 8, 9))
    resolved = []
    pending = _PendingDeviceToken(
        DeviceSample(torch.tensor(47, dtype=torch.int64)),
        resolved.append,
    )
    deferred = object.__new__(_DeferredStepOutput)
    deferred._records = {"decode": (pending,)}
    deferred._resolved = None
    deferred._event = None

    packet = runner.make_sampling_token_packet(
        (chunk,), {"decode": state}, deferred
    )

    assert packet.tolist() == [47]
    assert deferred._resolved is None
    assert resolved == []


@pytest.mark.parametrize(
    ("packet", "message"),
    [
        (torch.tensor([-1, 17, 29], dtype=torch.float32), "int64"),
        (torch.tensor([[-1, 17, 29]], dtype=torch.int64), "one-dimensional"),
        (torch.tensor([-1, 17], dtype=torch.int64), "length"),
    ],
)
def test_adopt_rejects_wrong_dtype_shape_or_length(packet, message):
    runner = _packet_runner()
    chunks, states = _mixed_step()

    with pytest.raises(RuntimeError, match=message):
        runner.adopt_sampling_token_packet(chunks, states, packet)


@pytest.mark.parametrize(
    "packet",
    [
        torch.tensor([-1, -1, 29], dtype=torch.int64),
        torch.tensor([99, 17, 29], dtype=torch.int64),
    ],
)
def test_adopt_rejects_missing_or_unexpected_token_slots(packet):
    runner = _packet_runner()
    chunks, states = _mixed_step()

    with pytest.raises(RuntimeError, match="missing|unexpected"):
        runner.adopt_sampling_token_packet(chunks, states, packet)


def test_adopt_inserts_and_overwrites_packet_backed_future_tokens():
    runner = _packet_runner()
    chunks, states = _mixed_step()
    runner._future_device_tokens = {
        "final": {0: torch.tensor(999)},
        "decode": {3: torch.tensor(998)},
    }
    packet = torch.tensor([-1, 17, 29], dtype=torch.int64)

    runner.adopt_sampling_token_packet(chunks, states, packet)

    assert "partial" not in runner._future_device_tokens
    assert int(runner._future_device_tokens["final"][0]) == 17
    assert int(runner._future_device_tokens["decode"][3]) == 29
    packet[2] = 31
    assert int(runner._future_device_tokens["decode"][3]) == 31


class _PassiveModel:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.logit_calls = 0

    def forward_tokens(self, token_ids, *_args, **_kwargs):
        self.forward_calls += 1
        return torch.zeros((len(token_ids), 4))

    def forward_decode_batch(self, token_ids, *_args, **_kwargs):
        self.forward_calls += 1
        return torch.zeros((len(token_ids), 4))

    def logits(self, _hidden):
        self.logit_calls += 1
        pytest.fail("a passive TP follower must not run lm_head")


class _FailSampler:
    def sample(self, *_args, **_kwargs):
        pytest.fail("a passive TP follower must not sample")


def _passive_runner() -> tuple[PagedModelRunner, _PassiveModel]:
    runner = _packet_runner()
    model = _PassiveModel()
    runner._model = model
    runner._pool = object()
    runner._sampler = _FailSampler()
    runner._sampling_owner = False
    runner._deferred_outputs = deque()
    runner._output_copy_stream = None
    runner._decode_slots = None
    runner._decode_positions = None
    runner._slot_staging = None
    runner._slot_copy_done = None
    runner._slot_staging_pool = []
    runner._tensor_decode_supported = False
    runner._graph = None
    return runner, model


def test_passive_eager_prefill_and_decode_skip_lm_head_and_sampling():
    runner, model = _passive_runner()
    chunks = (
        ScheduledChunk("prefill", 4, True, 0),
        ScheduledChunk("decode", 1, False, 1),
    )
    states = {
        "prefill": _state("prefill"),
        "decode": _state("decode", outputs=(13,)),
    }

    result = runner.execute_passive(chunks, states)

    assert result == {}
    assert model.forward_calls == 2
    assert model.logit_calls == 0
    assert runner._future_tokens == {}
    assert runner._future_device_tokens == {}


def test_passive_batched_decode_skips_lm_head_and_sampling():
    runner, model = _passive_runner()
    chunks = (
        ScheduledChunk("a", 1, False, 1),
        ScheduledChunk("b", 1, False, 1),
    )
    states = {
        "a": _state("a", outputs=(13,), pages=(0,)),
        "b": _state("b", outputs=(17,), pages=(1,)),
    }

    result = runner.execute_passive(chunks, states)

    assert result == {}
    assert model.forward_calls == 1
    assert model.logit_calls == 0
    assert runner._future_tokens == {}
    assert runner._future_device_tokens == {}


def test_passive_rank_cannot_accidentally_enter_the_sampling_path():
    runner, _model = _passive_runner()
    state = _state("decode", outputs=(13,))

    with pytest.raises(RuntimeError, match="only rank 0 may sample"):
        runner._sample(state, torch.zeros(32), position=1)


def test_spec_overlay_flag_round_trips_through_state_sync():
    base = _state("request", outputs=(11, 12))
    overlay = _OverlayState(base, (11, 77))
    chunks = (ScheduledChunk("request", 1, False, 2),)
    driver = StateSync()
    worker = StateSync()

    overlaid_delta = driver.diff(chunks, {"request": overlay})
    driver.apply(overlaid_delta)
    reconstructed = worker.apply(overlaid_delta)["request"]

    assert reconstructed.outputs == (11, 77)
    assert reconstructed.outputs_override is True

    restored_delta = driver.diff(chunks, {"request": base})
    assert len(restored_delta.new) == 1
    restored = worker.apply(restored_delta)["request"]
    assert restored.outputs == (11, 12)
    assert restored.outputs_override is False


def test_spec_overlay_output_wins_over_retained_device_token():
    runner = _packet_runner()
    base = _state("request", outputs=(41,))
    overlay = _OverlayState(base, (73,))
    runner._future_device_tokens = {"request": {0: torch.tensor(99)}}

    assert int(runner._previous_token(base, 1)) == 99
    assert runner._previous_token(overlay, 1) == 73
