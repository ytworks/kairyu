"""PP=2 inter-step pipelining: async runner contract + bubble accounting (m6 D5).

CPU tests pin the structure the review made load-bearing: two in-flight
scheduler steps with stage affinity, full decode batches per stage (never
split), sampled tokens landing when a step exits the last stage, and the
mandatory bubble-fraction metric. Fake stage workers are deterministic; the
GPU phase swaps real stage execution behind the same seams.
"""

from __future__ import annotations

import pytest

from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.pipeline import (
    PipelinedEngineCore,
    PipelinedModelRunner,
    SyncRunnerAdapter,
)
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

_VOCAB = 50_000


class _ToyRunner:
    def execute(self, scheduled, states):
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids)
                sampled[chunk.request_id] = (SampledToken((seed + 31 * chunk.position) % _VOCAB),)
        return sampled


class _RecordingStage:
    """Fake PP stage: records the steps it processed, in order."""

    def __init__(self, stage_index: int, sampler: _ToyRunner | None = None) -> None:
        self.stage_index = stage_index
        self.steps_seen: list[int] = []
        self._sampler = sampler

    def execute(self, step_index, scheduled, states):
        self.steps_seen.append(step_index)
        if self._sampler is None:
            return None  # hidden states move stage-to-stage on GPU; nothing on CPU
        return self._sampler.execute(scheduled, states)


def _make_scheduler(num_pages: int = 64) -> Scheduler:
    kv = RadixKVCache(num_pages=num_pages, page_size=4)
    return Scheduler(kv, max_num_batched_tokens=32, page_size=4)


def _requests() -> list[EngineRequest]:
    return [
        EngineRequest("a", prompt_token_ids=tuple(range(1, 6)), max_new_tokens=6),
        EngineRequest("b", prompt_token_ids=tuple(range(10, 30)), max_new_tokens=4),
        EngineRequest("c", prompt_token_ids=(3, 1, 4), max_new_tokens=5),
    ]


def _reference() -> dict[str, tuple[int, ...]]:
    scheduler = _make_scheduler()
    core = EngineCore(scheduler, _ToyRunner())
    for request in _requests():
        core.add_request(request)
    return core.run_to_completion()


def test_sync_adapter_resolves_immediately() -> None:
    scheduler = _make_scheduler()
    runner = SyncRunnerAdapter(_ToyRunner())
    core = PipelinedEngineCore(scheduler, runner, depth=1)
    for request in _requests():
        core.add_request(request)
    assert core.run_to_completion() == _reference()


def test_pipelined_runner_matches_single_core_outputs() -> None:
    stage0 = _RecordingStage(0)
    stage1 = _RecordingStage(1, sampler=_ToyRunner())
    runner = PipelinedModelRunner(stages=(stage0, stage1))
    scheduler = _make_scheduler()
    core = PipelinedEngineCore(scheduler, runner, depth=2)
    for request in _requests():
        core.add_request(request)

    outputs = core.run_to_completion()

    assert outputs == _reference()
    # stage affinity: every step visits stage 0 then stage 1, in order
    assert stage0.steps_seen == sorted(stage0.steps_seen)
    assert stage1.steps_seen == stage0.steps_seen


def test_depth_two_keeps_both_stages_busy() -> None:
    # long decode phase -> steady state; depth 2 must overlap stages
    scheduler = _make_scheduler()
    runner = PipelinedModelRunner(stages=(_RecordingStage(0), _RecordingStage(1, _ToyRunner())))
    core = PipelinedEngineCore(scheduler, runner, depth=2)
    core.add_request(EngineRequest("long", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=24))
    core.run_to_completion()

    assert runner.bubble_fraction < 0.2  # steady-state overlap: stages rarely idle


def test_depth_one_drains_the_pipe_every_step() -> None:
    scheduler = _make_scheduler()
    runner = PipelinedModelRunner(stages=(_RecordingStage(0), _RecordingStage(1, _ToyRunner())))
    core = PipelinedEngineCore(scheduler, runner, depth=1)
    core.add_request(EngineRequest("long", prompt_token_ids=(1, 2, 3, 4), max_new_tokens=24))
    core.run_to_completion()

    # serial mode: while one stage works the other idles -> ~50% bubbles
    assert runner.bubble_fraction == pytest.approx(0.5, abs=0.05)


def test_decode_batches_are_never_split_across_stages() -> None:
    scheduler = _make_scheduler()
    seen_chunk_counts: list[int] = []

    class _CountingStage(_RecordingStage):
        def execute(self, step_index, scheduled, states):
            if self.stage_index == 0:
                seen_chunk_counts.append(len(scheduled))
            return super().execute(step_index, scheduled, states)

    runner = PipelinedModelRunner(
        stages=(_CountingStage(0), _CountingStage(1, _ToyRunner()))
    )
    core = PipelinedEngineCore(scheduler, runner, depth=2)
    for request in _requests():
        core.add_request(request)
    core.run_to_completion()

    # each submitted step arrives at a stage whole — chunk counts match the
    # scheduler's plans, no micro-batch splitting (m6 D5)
    assert seen_chunk_counts == runner.step_chunk_counts


def test_oversized_prompt_rejected_through_pipeline() -> None:
    # C2: a prompt too large to ever fit is rejected (empty output) through the
    # pipelined core too, not a fatal stall.
    kv = RadixKVCache(num_pages=1, page_size=4)
    scheduler = Scheduler(kv, max_num_batched_tokens=8, page_size=4)
    runner = SyncRunnerAdapter(_ToyRunner())
    core = PipelinedEngineCore(scheduler, runner, depth=2)
    core.add_request(EngineRequest("big", prompt_token_ids=tuple(range(50)), max_new_tokens=2))

    outputs = core.run_to_completion()  # no RuntimeError
    assert outputs["big"] == ()
    assert scheduler.finish_reason("big") == "length"


def test_invalid_depth_and_stage_count_rejected() -> None:
    scheduler = _make_scheduler()
    with pytest.raises(ValueError):
        PipelinedEngineCore(scheduler, SyncRunnerAdapter(_ToyRunner()), depth=0)
    with pytest.raises(ValueError):
        PipelinedModelRunner(stages=(_RecordingStage(0),))
