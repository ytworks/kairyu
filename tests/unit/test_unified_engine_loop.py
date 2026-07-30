"""Production EngineLoop parity across synchronous, overlap and PP execution.

These tests deliberately enter through the same loop KairyuBackend and the ZMQ
service construct. The legacy run-to-completion cores have narrower semantics
and are kept only as compatibility harnesses.
"""

from __future__ import annotations

import time

import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.core.pipeline import PipelinedModelRunner
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import Scheduler
from kairyu.engine.core.spec_runner import SpeculativeRunner
from kairyu.engine.core.step_input import RequestSnapshot
from kairyu.engine.engine_loop import EngineLoop, StreamUpdate
from kairyu.engine.kairyu_backend import KairyuBackend, build_engine_loop


class _CharTokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(100 + sum(map(ord, word)) % 1900 for word in text.split())

    def decode(self, token_ids) -> str:
        return "".join(chr(ord("a") + token_id % 26) for token_id in token_ids)

    def vocab(self) -> list[str]:
        return [chr(ord("a") + index % 26) for index in range(2048)]


class _PositionRunner:
    """Snapshot-only runner whose output is determined by scheduled position."""

    def __init__(
        self,
        *,
        base: int = 0,
        delay_s: float = 0.0,
        events: list[str] | None = None,
    ) -> None:
        self.base = base
        self.delay_s = delay_s
        self.events = events if events is not None else []
        self.seen_state_types: list[type] = []
        self.released: list[str] = []
        self._step = 0

    def execute(self, scheduled, states):
        step = self._step
        self._step += 1
        for chunk in scheduled:
            self.seen_state_types.append(type(states[chunk.request_id]))
        if self.delay_s:
            time.sleep(self.delay_s)
        sampled = {
            chunk.request_id: tuple(
                SampledToken(self.base + chunk.position + offset)
                for offset in range(1 if chunk.is_prefill else chunk.num_tokens)
            )
            for chunk in scheduled
            if not chunk.is_prefill or states[chunk.request_id].prefill_done
        }
        self.events.append(f"executed:{step}")
        return sampled

    def release(self, request_id: str) -> None:
        self.released.append(request_id)


def _loop(
    depth: int,
    runner: object,
    *,
    budget: int = 4,
    scheduler: Scheduler | None = None,
) -> tuple[EngineLoop, Scheduler]:
    if scheduler is None:
        cache = RadixKVCache(num_pages=64, page_size=4)
        scheduler = Scheduler(
            cache,
            max_num_batched_tokens=budget,
            max_num_seqs=8,
            page_size=4,
        )
    return EngineLoop(_CharTokenizer(), scheduler, runner, pipeline_depth=depth), scheduler


def _drive(loop: EngineLoop, limit: int = 200) -> list[tuple[str, StreamUpdate]]:
    updates: list[tuple[str, StreamUpdate]] = []
    for _ in range(limit):
        if not loop.has_work():
            return updates
        updates.extend(loop.step())
    raise AssertionError("unified loop did not drain")


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_depths_share_streaming_and_immutable_snapshot_path(depth: int) -> None:
    runner = _PositionRunner(base=1000)
    loop, scheduler = _loop(depth, runner, budget=3)
    loop.submit("a", "one two three four five six seven", SamplingParams(max_tokens=6))
    loop.submit("b", "short prompt", SamplingParams(max_tokens=4))

    updates = _drive(loop)
    finals = {request_id: update for request_id, update in updates if update.finished}

    assert finals["a"].outputs == tuple(range(1000, 1006))
    assert finals["b"].outputs == tuple(range(1000, 1004))
    assert all(state_type is RequestSnapshot for state_type in runner.seen_state_types)
    assert scheduler.states == {}
    assert sorted(runner.released) == ["a", "b"]
    loop.close()


def test_depth_two_schedules_next_snapshot_before_oldest_execute_finishes() -> None:
    events: list[str] = []

    class _RecordingScheduler(Scheduler):
        def schedule(self):
            events.append(f"scheduled:{sum(e.startswith('scheduled:') for e in events)}")
            return super().schedule()

    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = _RecordingScheduler(cache, max_num_batched_tokens=16, page_size=4)
    runner = _PositionRunner(base=1000, delay_s=0.02, events=events)
    loop, _ = _loop(2, runner, scheduler=scheduler)
    loop.submit("overlap", "one two three four", SamplingParams(max_tokens=4))

    _drive(loop)

    assert events.index("scheduled:1") < events.index("executed:0")
    loop.close()


def test_stop_holdback_finishes_stream_before_late_surplus_is_reclaimed() -> None:
    runner = _PositionRunner()
    loop, scheduler = _loop(2, runner)
    loop.submit(
        "stop",
        "one two three",
        SamplingParams(max_tokens=8, stop=("cde",)),
    )

    final = None
    final_had_late_work = False
    while loop.has_work():
        for request_id, update in loop.step():
            if request_id == "stop" and update.finished:
                final = update
                final_had_late_work = loop.has_work()

    assert final is not None
    assert final.text == "ab"
    assert final.finish_reason == "stop"
    assert "cde" not in final.text
    assert final_had_late_work
    assert scheduler.states == {}
    assert runner.released == ["stop"]
    loop.close()


def test_grammar_termination_trims_scheduled_ahead_token_and_releases_late() -> None:
    class _GrammarRunner(_PositionRunner):
        def execute(self, scheduled, states):
            sampled = super().execute(scheduled, states)
            return {
                request_id: tuple(
                    SampledToken(token.token_id, grammar_terminated=True)
                    for token in tokens
                )
                for request_id, tokens in sampled.items()
            }

    runner = _GrammarRunner(base=700)
    loop, scheduler = _loop(2, runner)
    loop.submit("grammar", "json prompt", SamplingParams(max_tokens=8))

    updates = _drive(loop)
    final = next(
        update
        for request_id, update in updates
        if request_id == "grammar" and update.finished
    )

    assert final.outputs == (700,)
    assert final.finish_reason == "stop"
    assert scheduler.states == {}
    assert runner.released == ["grammar"]
    loop.close()


def test_speculative_verification_runs_inside_depth_two_loop() -> None:
    class _MatchingDraft:
        def propose(self, context: tuple[int, ...], max_draft: int) -> tuple[int, ...]:
            last = context[-1]
            if last < 1000:
                return ()
            return tuple(last + offset for offset in range(1, max_draft + 1))

    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=8,
        page_size=4,
        speculative_tokens=2,
    )
    target = _PositionRunner(base=1000)
    speculative = SpeculativeRunner(target, draft_source=_MatchingDraft())
    loop, _ = _loop(2, speculative, scheduler=scheduler)
    loop.submit(
        "spec",
        "repeated repeated",
        SamplingParams(max_tokens=6, temperature=0.0),
    )

    updates = _drive(loop)
    final = next(
        update
        for request_id, update in updates
        if request_id == "spec" and update.finished
    )

    assert final.outputs == tuple(range(1000, 1006))
    assert speculative.draft_proposed > 0
    assert speculative.draft_accepted == speculative.draft_proposed
    loop.close()


def test_preemption_and_chunked_prefill_drain_through_unified_loop() -> None:
    class _PreemptionScheduler(Scheduler):
        preemptions = 0

        def _preempt_for_decode(self, needy_id: str) -> bool:
            preempted = super()._preempt_for_decode(needy_id)
            self.preemptions += int(preempted)
            return preempted

    cache = RadixKVCache(num_pages=3, page_size=4)
    scheduler = _PreemptionScheduler(
        cache,
        max_num_batched_tokens=5,
        max_num_seqs=4,
        page_size=4,
    )
    runner = _PositionRunner(base=1000)
    loop, _ = _loop(2, runner, scheduler=scheduler)
    loop.submit("decode", "one two three four", SamplingParams(max_tokens=3))
    loop.submit(
        "prefill",
        "alpha beta gamma delta epsilon zeta eta theta",
        SamplingParams(max_tokens=2),
    )

    updates = _drive(loop)
    finals = {request_id: update for request_id, update in updates if update.finished}

    assert scheduler.preemptions >= 1
    assert finals["decode"].outputs == tuple(range(1000, 1003))
    assert finals["prefill"].outputs == tuple(range(1000, 1002))
    assert scheduler.states == {}
    loop.close()


def test_native_pipeline_runner_uses_same_production_loop() -> None:
    class _Stage:
        def __init__(self, sampler=None) -> None:
            self.sampler = sampler

        def execute(self, step_index, scheduled, states):
            if self.sampler is None:
                return None
            return self.sampler.execute(scheduled, states)

    target = _PositionRunner(base=1000)
    runner = PipelinedModelRunner((_Stage(), _Stage(target)))
    loop, scheduler = _loop(2, runner, budget=8)
    loop.submit("pp", "one two three four", SamplingParams(max_tokens=16))

    updates = _drive(loop)
    final = next(update for request_id, update in updates if request_id == "pp" and update.finished)

    assert final.outputs == tuple(range(1000, 1016))
    assert runner.bubble_fraction < 0.2
    assert scheduler.states == {}
    loop.close()


def test_abort_trims_late_depth_two_result_before_releasing_state() -> None:
    runner = _PositionRunner(base=1000)
    loop, scheduler = _loop(2, runner)
    loop.submit("abort", "one two three", SamplingParams(max_tokens=64))

    loop.step()
    loop.abort("abort")
    updates = _drive(loop)

    final = next(
        update
        for request_id, update in updates
        if request_id == "abort" and update.finished
    )
    assert final.finish_reason == "abort"
    assert scheduler.states == {}
    assert runner.released == ["abort"]
    loop.close()


def test_failed_oldest_step_settles_queued_work_before_purge_and_recovers() -> None:
    class _FailOnceRunner(_PositionRunner):
        failed = False

        def execute(self, scheduled, states):
            if not self.failed:
                self.failed = True
                raise RuntimeError("injected")
            return super().execute(scheduled, states)

    runner = _FailOnceRunner(base=1000)
    loop, scheduler = _loop(2, runner)
    loop.submit("failed", "one two three", SamplingParams(max_tokens=8))

    with pytest.raises(RuntimeError, match="injected"):
        loop.step()
    loop.purge(("failed",))

    assert scheduler.states == {}
    loop.submit("recovered", "four five", SamplingParams(max_tokens=2))
    updates = _drive(loop)
    final = next(
        update
        for request_id, update in updates
        if request_id == "recovered" and update.finished
    )
    assert final.outputs == (1000, 1001)
    loop.close()


def test_invalid_pipeline_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="pipeline_depth"):
        _loop(0, _PositionRunner())

    with pytest.raises(ValueError, match="pipeline_depth"):
        build_engine_loop(pipeline_depth=0)


def test_production_builder_exposes_pipeline_depth() -> None:
    loop, _cache, _scheduler = build_engine_loop(
        tokenizer=_CharTokenizer(),
        runner=_PositionRunner(),
        pipeline_depth=3,
    )

    assert loop.pipeline_depth == 3
    loop.close()


async def test_backend_streams_stop_holdback_through_depth_two() -> None:
    runner = _PositionRunner()
    backend = KairyuBackend(
        tokenizer=_CharTokenizer(),
        runner=runner,
        pipeline_depth=2,
    )
    request = GenerationRequest(
        request_id="backend",
        prompt="one two three",
        sampling_params=SamplingParams(max_tokens=8, stop=("cde",)),
    )

    partials = [partial async for partial in backend.stream(request)]

    assert partials[-1].finished
    assert partials[-1].completions[0].text == "ab"
    assert partials[-1].completions[0].finish_reason == "stop"
    assert all("cde" not in partial.completions[0].text for partial in partials)
    await backend.shutdown()


def test_qwen_benchmark_generation_uses_configured_production_depth() -> None:
    from bench.future_token_bench import _run_generation

    runner = _PositionRunner(base=1000)
    outputs = _run_generation(
        runner,
        prefix="bench",
        request_count=3,
        max_new_tokens=5,
        num_pages=64,
        page_size=4,
        pipeline_depth=2,
    )

    assert outputs == {
        "bench-r0": tuple(range(1000, 1005)),
        "bench-r1": tuple(range(1000, 1005)),
        "bench-r2": tuple(range(1000, 1005)),
    }
    assert sorted(runner.released) == ["bench-r0", "bench-r1", "bench-r2"]
