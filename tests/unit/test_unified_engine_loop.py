"""Production EngineLoop parity across synchronous, overlap and PP execution.

These tests deliberately enter through the same loop KairyuBackend and the ZMQ
service construct. The legacy run-to-completion cores have narrower semantics
and are kept only as compatibility harnesses.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

import kairyu.engine.engine_loop as engine_loop_module
from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.core.pipeline import PipelinedModelRunner
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.spec_runner import SpeculativeRunner
from kairyu.engine.core.step_input import RequestSnapshot
from kairyu.engine.engine_loop import EngineLoop, StreamUpdate
from kairyu.engine.kairyu_backend import KairyuBackend, build_engine_loop
from kairyu.engine.prompt import TokensPrompt
from kairyu.models.generation import GenerationDefaults
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS


class _CharTokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(100 + sum(map(ord, word)) % 1900 for word in text.split())

    def decode(self, token_ids) -> str:
        return "".join(chr(ord("a") + token_id % 26) for token_id in token_ids)

    def vocab(self) -> list[str]:
        return [chr(ord("a") + index % 26) for index in range(2048)]


class _RawSpecialTokenizer:
    """Tiny tokenizer whose raw pieces intentionally differ from decoded text."""

    eos_token_id = 1
    _pieces = ("[UNK]", "<eos>", "<tool>", "<0xE3>", "x")
    _decoded = {
        0: "?",
        1: "<eos>",
        2: "<tool>",
        3: "�",
        4: "x",
    }
    _special_ids = frozenset({1, 2})

    def __init__(self) -> None:
        self.vocab_calls = 0

    def encode(self, text: str) -> tuple[int, ...]:
        return (0,) if text.startswith("skip") else (4,)

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        return "".join(
            ""
            if skip_special_tokens and token_id in self._special_ids
            else self._decoded.get(token_id, f"fallback{token_id}")
            for token_id in token_ids
        )

    def vocab(self) -> list[str]:
        self.vocab_calls += 1
        return list(self._pieces)


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


class _ScriptedTokenRunner:
    """Return request-specific IDs so opposite decode policies can interleave."""

    def __init__(
        self,
        scripts: dict[str, tuple[int, ...]],
        *,
        with_logprobs: bool = False,
    ) -> None:
        self.scripts = scripts
        self.with_logprobs = with_logprobs

    def execute(self, scheduled, states):
        sampled = {}
        for chunk in scheduled:
            if chunk.is_prefill and not states[chunk.request_id].prefill_done:
                continue
            tokens = []
            for offset in range(chunk.num_tokens):
                position = chunk.position + offset
                token_id = self.scripts[chunk.request_id][position]
                if self.with_logprobs:
                    tokens.append(
                        SampledToken(
                            token_id,
                            logprob=-0.1 * (position + 1),
                            top_logprobs=((1, -1.0), (99, -2.0), (-1, -3.0)),
                        )
                    )
                else:
                    tokens.append(SampledToken(token_id))
            sampled[chunk.request_id] = tuple(tokens)
        return sampled


class _DeferredPositionHandle:
    def __init__(self, runner, scheduled, states) -> None:
        self._runner = runner
        self._scheduled = scheduled
        self._states = states

    def result(self):
        return self._runner.resolve(self._scheduled, self._states)


class _DeferredPositionRunner(_PositionRunner):
    """Native-submit runner that makes unresolved depth deterministic in tests."""

    def __init__(self, *, base: int = 0) -> None:
        super().__init__(base=base)
        self.submitted: list[tuple] = []
        self._outstanding_prefill: list[bool] = []
        self.max_outstanding = 0
        self.max_prefill_horizon = 0
        self.on_next_submit = None
        self.on_first_resolve = None

    def submit(self, _step_index, scheduled, states):
        scheduled = tuple(scheduled)
        self.submitted.append(scheduled)
        self._outstanding_prefill.append(
            any(chunk.is_prefill for chunk in scheduled)
        )
        self.max_outstanding = max(
            self.max_outstanding,
            len(self._outstanding_prefill),
        )
        if any(self._outstanding_prefill):
            self.max_prefill_horizon = max(
                self.max_prefill_horizon,
                len(self._outstanding_prefill),
            )
        callback, self.on_next_submit = self.on_next_submit, None
        if callback is not None:
            callback()
        return _DeferredPositionHandle(self, scheduled, states)

    def resolve(self, scheduled, states):
        callback, self.on_first_resolve = self.on_first_resolve, None
        if callback is not None:
            callback()
        try:
            return super().execute(scheduled, states)
        finally:
            self._outstanding_prefill.pop(0)


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


def _raw_special_loop(
    scripts: dict[str, tuple[int, ...]],
    *,
    with_logprobs: bool = False,
) -> tuple[EngineLoop, _RawSpecialTokenizer]:
    tokenizer = _RawSpecialTokenizer()
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=4,
        max_num_seqs=8,
        page_size=4,
    )
    return (
        EngineLoop(
            tokenizer,
            scheduler,
            _ScriptedTokenRunner(scripts, with_logprobs=with_logprobs),
        ),
        tokenizer,
    )


def _drive(loop: EngineLoop, limit: int = 200) -> list[tuple[str, StreamUpdate]]:
    updates: list[tuple[str, StreamUpdate]] = []
    for _ in range(limit):
        if not loop.has_work():
            return updates
        updates.extend(loop.step())
    raise AssertionError("unified loop did not drain")


def _stage_map(update: StreamUpdate):
    return {metric.stage: metric for metric in update.stage_metrics}


def test_trace_disabled_never_reads_the_stage_clock(monkeypatch) -> None:
    def forbidden_clock() -> int:
        raise AssertionError("trace-disabled engine path read the stage clock")

    monkeypatch.setattr(engine_loop_module, "perf_counter_ns", forbidden_clock)
    loop, _ = _loop(2, _PositionRunner(), budget=2)
    loop.submit(
        "plain",
        TokensPrompt((1, 2, 3, 4, 5)),
        SamplingParams(max_tokens=2, ignore_eos=True),
    )

    final = next(
        update
        for request_id, update in _drive(loop)
        if request_id == "plain" and update.finished
    )

    assert final.stage_metrics == ()
    loop.close()


def test_trace_metrics_cover_mixed_chunked_prefill_and_decode(monkeypatch) -> None:
    ticks = iter(range(10, 100_000, 10))
    monkeypatch.setattr(engine_loop_module, "perf_counter_ns", lambda: next(ticks))
    loop, _ = _loop(2, _PositionRunner(base=1000), budget=2)
    loop.submit(
        "traced",
        TokensPrompt(tuple(range(1, 10))),
        SamplingParams(max_tokens=3, ignore_eos=True),
        trace_requested=True,
    )
    loop.submit(
        "plain",
        TokensPrompt((20, 21, 22)),
        SamplingParams(max_tokens=2, ignore_eos=True),
    )

    updates = _drive(loop)
    finals = {request_id: update for request_id, update in updates if update.finished}
    traced = finals["traced"]
    metrics = _stage_map(traced)

    assert tuple(metric.stage for metric in traced.stage_metrics) == (
        "tokenize",
        "queue_wait",
        "schedule",
        "prefill",
        "decode_step",
        "detokenize",
    )
    assert metrics["tokenize"].occurrences == 1
    assert metrics["queue_wait"].occurrences == 1
    assert metrics["prefill"].occurrences >= 5
    assert metrics["decode_step"].occurrences >= 1
    assert metrics["detokenize"].occurrences >= 2
    assert all(metric.duration_ns > 0 for metric in traced.stage_metrics)
    assert finals["plain"].stage_metrics == ()
    loop.close()


def test_early_eos_trace_excludes_unobserved_scheduled_ahead_decode(
    monkeypatch,
) -> None:
    ticks = iter(range(10, 100_000, 10))
    monkeypatch.setattr(engine_loop_module, "perf_counter_ns", lambda: next(ticks))
    loop, _ = _loop(2, _PositionRunner())
    loop._default_eos = 0
    loop.submit(
        "early-eos",
        TokensPrompt((1, 2, 3)),
        SamplingParams(max_tokens=6),
        trace_requested=True,
    )

    updates = _drive(loop)
    final = next(
        update
        for request_id, update in updates
        if request_id == "early-eos" and update.finished
    )
    metrics = _stage_map(final)

    assert final.outputs == (0,)
    assert final.finish_reason == "stop"
    assert set(metrics) == {
        "tokenize",
        "queue_wait",
        "schedule",
        "prefill",
        "detokenize",
    }
    assert metrics["detokenize"].occurrences == 1
    assert len(metrics) == len(final.stage_metrics)
    loop.close()


def test_loop_resolves_only_model_owned_sampling_omissions_at_submit() -> None:
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=8, page_size=4)
    loop = EngineLoop(
        _CharTokenizer(),
        scheduler,
        _PositionRunner(),
        generation_defaults=GenerationDefaults(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.05,
            repetition_penalty=1.1,
        ),
    )
    params = SamplingParams(top_p=0.8).with_generation_config_omitted(
        GENERATION_CONFIG_SAMPLING_FIELDS - {"top_p"}
    )

    loop.submit("defaults", "hello", params)

    sampling = loop._ops[-1].requests[0].sampling
    assert sampling.temperature == 0.6
    assert sampling.top_p == 0.8
    assert sampling.top_k == 20
    assert sampling.min_p == 0.05
    assert sampling.repetition_penalty == 1.1
    loop.close()


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


def test_cumulative_update_prefixes_remain_frozen_after_later_steps() -> None:
    loop, _ = _loop(1, _PositionRunner(base=1000))
    loop.submit(
        "frozen-prefix",
        TokensPrompt((1, 2, 3, 4)),
        SamplingParams(max_tokens=3, ignore_eos=True),
    )

    token_updates = [
        update
        for request_id, update in _drive(loop)
        if request_id == "frozen-prefix" and update.outputs
    ]

    assert [tuple(update.outputs) for update in token_updates] == [
        (1000,),
        (1000, 1001),
        (1000, 1001, 1002),
    ]
    loop.close()


def test_logprob_presentation_processes_each_token_once(monkeypatch) -> None:
    calls = 0
    original = engine_loop_module._token_logprob

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_loop_module, "_token_logprob", counted)
    loop, _ = _raw_special_loop(
        {"logprobs": (2, 3, 4, 4)},
        with_logprobs=True,
    )
    loop.submit(
        "logprobs",
        "keep prompt",
        SamplingParams(
            max_tokens=4,
            ignore_eos=True,
            logprobs=3,
            skip_special_tokens=False,
        ),
    )

    final = next(
        update
        for request_id, update in _drive(loop)
        if request_id == "logprobs" and update.finished
    )

    assert tuple(entry.token_id for entry in final.logprob_content or ()) == (2, 3, 4, 4)
    assert calls == 4 * 4  # selected token plus three top-logprob entries
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


def test_prefill_horizon_is_two_then_pure_decode_restores_depth_five() -> None:
    runner = _DeferredPositionRunner(base=1000)
    loop, _ = _loop(5, runner, budget=32)
    loop.submit(
        "request",
        TokensPrompt((1, 2, 3, 4)),
        SamplingParams(max_tokens=16, ignore_eos=True),
    )

    loop.step()

    assert len(runner.submitted) == 2
    assert runner.max_prefill_horizon == 2
    assert any(chunk.is_prefill for chunk in runner.submitted[0])
    assert all(not chunk.is_prefill for chunk in runner.submitted[1])

    loop.step()

    assert runner.max_outstanding == 5
    assert len(loop._pending_steps) == 4
    _drive(loop)
    loop.close()


def test_arrivals_during_prefill_join_one_next_admission_cohort() -> None:
    runner = _DeferredPositionRunner(base=1000)
    loop, _ = _loop(5, runner, budget=32)
    newcomer_ids = tuple(f"new-{index}" for index in range(4))

    def submit_newcomers() -> None:
        for index, request_id in enumerate(newcomer_ids):
            loop.submit(
                request_id,
                TokensPrompt((10 + index,)),
                SamplingParams(max_tokens=2, ignore_eos=True),
            )

    runner.on_first_resolve = submit_newcomers
    loop.submit(
        "first",
        TokensPrompt((1, 2, 3, 4)),
        SamplingParams(max_tokens=8, ignore_eos=True),
    )

    loop.step()
    loop.step()

    admitted = tuple(
        chunk.request_id for chunk in runner.submitted[2] if chunk.is_prefill
    )
    assert admitted == newcomer_ids
    assert runner.max_prefill_horizon == 2
    _drive(loop)
    loop.close()


def test_waiter_drains_existing_decode_horizon_before_prefill_schedule() -> None:
    runner = _DeferredPositionRunner(base=1000)
    loop, _ = _loop(5, runner, budget=32)
    loop.submit(
        "decode",
        TokensPrompt((1, 2, 3, 4)),
        SamplingParams(max_tokens=20, ignore_eos=True),
    )
    loop.step()
    loop.step()
    assert len(loop._pending_steps) == 4
    submitted_before_waiter = len(runner.submitted)

    loop.submit(
        "waiter",
        TokensPrompt((9,)),
        SamplingParams(max_tokens=2, ignore_eos=True),
    )
    for expected_pending in (3, 2, 1):
        loop.step()
        assert len(loop._pending_steps) == expected_pending
        assert len(runner.submitted) == submitted_before_waiter

    loop.step()

    assert any(chunk.request_id == "waiter" for chunk in runner.submitted[-1])
    assert any(chunk.is_prefill for chunk in runner.submitted[-1])
    _drive(loop)
    loop.close()


def test_arrival_inside_submit_stops_the_current_decode_fill() -> None:
    runner = _DeferredPositionRunner(base=1000)
    loop, _ = _loop(5, runner, budget=32)
    loop.submit(
        "decode",
        TokensPrompt((1, 2, 3, 4)),
        SamplingParams(max_tokens=20, ignore_eos=True),
    )
    loop.step()
    assert len(loop._pending_steps) == 1
    submitted_before_arrival = len(runner.submitted)

    runner.on_next_submit = lambda: loop.submit(
        "arrival",
        TokensPrompt((9,)),
        SamplingParams(max_tokens=2, ignore_eos=True),
    )
    loop.step()

    # submit() ran from inside the first new runner submission. The producer
    # op is not safe to mutate into the scheduler mid-snapshot, but it must stop
    # this public step from filling the remaining depth-five decode horizon.
    assert len(runner.submitted) == submitted_before_arrival + 1
    assert len(loop._pending_steps) == 1

    loop.step()

    assert any(
        chunk.request_id == "arrival" and chunk.is_prefill
        for chunk in runner.submitted[-1]
    )
    _drive(loop)
    loop.close()


@pytest.mark.parametrize(("depth", "expected_horizon"), [(1, 1), (5, 2)])
def test_chunked_prefill_respects_dynamic_horizon(
    depth: int,
    expected_horizon: int,
) -> None:
    runner = _DeferredPositionRunner(base=1000)
    loop, _ = _loop(depth, runner, budget=2)
    loop.submit(
        "chunked",
        TokensPrompt(tuple(range(1, 10))),
        SamplingParams(max_tokens=3, ignore_eos=True),
    )

    _drive(loop)

    assert sum(
        any(chunk.is_prefill for chunk in scheduled)
        for scheduled in runner.submitted
    ) >= 5
    assert runner.max_prefill_horizon == expected_horizon
    loop.close()


def test_scheduler_without_prefill_signal_stays_on_fail_safe_horizon() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=64, page_size=4),
        max_num_batched_tokens=16,
        max_num_seqs=8,
        page_size=4,
    )
    scheduler.has_prefill_work = None
    runner = _DeferredPositionRunner(base=1000)
    loop, _ = _loop(5, runner, scheduler=scheduler)
    loop.submit(
        "unknown-adapter",
        TokensPrompt((1, 2, 3, 4)),
        SamplingParams(max_tokens=8, ignore_eos=True),
    )

    _drive(loop)

    assert runner.max_outstanding == 2
    loop.close()


def test_terminal_cohort_drains_before_fragmented_prefill_admission() -> None:
    class _RecordingRunner(_PositionRunner):
        def __init__(self) -> None:
            super().__init__(base=1000)
            self.prefill_batches: list[tuple[str, ...]] = []

        def execute(self, scheduled, states):
            prefill_ids = tuple(
                chunk.request_id for chunk in scheduled if chunk.is_prefill
            )
            if prefill_ids:
                self.prefill_batches.append(prefill_ids)
            return super().execute(scheduled, states)

    class _NoCohortDrainScheduler(Scheduler):
        def should_drain_before_admission(self) -> bool:
            return False

    def run(scheduler_type):
        cache = RadixKVCache(num_pages=64, page_size=4)
        scheduler = scheduler_type(
            cache,
            max_num_batched_tokens=2,
            max_num_seqs=2,
            page_size=4,
            max_num_partial_prefills=1,
        )
        runner = _RecordingRunner()
        loop, _ = _loop(2, runner, scheduler=scheduler)
        prompt_lengths = (2, 1, 1, 1, 1)
        for index, prompt_length in enumerate(prompt_lengths):
            loop.submit(
                f"request-{index}",
                TokensPrompt(tuple(range(1, prompt_length + 1))),
                SamplingParams(max_tokens=1, ignore_eos=True),
            )
        updates = _drive(loop)
        finals = {
            request_id: update.outputs
            for request_id, update in updates
            if update.finished
        }
        loop.close()
        return runner.prefill_batches, finals

    baseline_batches, baseline = run(_NoCohortDrainScheduler)
    optimized_batches, optimized = run(Scheduler)

    assert optimized == baseline
    assert baseline_batches == [
        ("request-0",),
        ("request-1",),
        ("request-2",),
        ("request-3",),
        ("request-4",),
    ]
    assert optimized_batches == [
        ("request-0",),
        ("request-1", "request-2"),
        ("request-3", "request-4"),
    ]


def test_terminal_cohort_drain_never_delays_an_outranking_waiter() -> None:
    def build(waiting_priority: int) -> Scheduler:
        cache = RadixKVCache(num_pages=64, page_size=4)
        scheduler = Scheduler(
            cache,
            max_num_batched_tokens=2,
            max_num_seqs=2,
            page_size=4,
            priority_age_s=60.0,
        )
        scheduler.add_request(
            EngineRequest("running-a", (1,), max_new_tokens=1, priority=0)
        )
        scheduler.add_request(
            EngineRequest("running-b", (2,), max_new_tokens=1, priority=0)
        )
        first = scheduler.schedule()
        scheduler.schedule()
        scheduler.update({first.scheduled[0].request_id: (1000,)})
        scheduler.add_request(
            EngineRequest(
                "waiting-a",
                (3,),
                max_new_tokens=1,
                priority=waiting_priority,
            )
        )
        scheduler.add_request(
            EngineRequest(
                "waiting-b",
                (4,),
                max_new_tokens=1,
                priority=waiting_priority,
            )
        )
        return scheduler

    assert build(waiting_priority=0).should_drain_before_admission() is True
    assert build(waiting_priority=-1).should_drain_before_admission() is False


def test_terminal_cohort_drain_preserves_cached_prefix_admission_order() -> None:
    def build(*, cached_waiters: bool) -> Scheduler:
        cache = RadixKVCache(num_pages=64, page_size=4)
        if cached_waiters:
            allocation = cache.allocate((9, 9, 9, 9))
            cache.mark_computed(allocation)
            cache.free(allocation)
        scheduler = Scheduler(
            cache,
            max_num_batched_tokens=10,
            max_num_seqs=2,
            page_size=4,
        )
        scheduler.add_request(EngineRequest("running-a", (1,), max_new_tokens=1))
        scheduler.add_request(EngineRequest("running-b", (2,), max_new_tokens=1))
        first = scheduler.schedule()
        scheduler.schedule()
        scheduler.update({first.scheduled[0].request_id: (1000,)})
        prefix = (9, 9, 9, 9) if cached_waiters else (7, 7, 7, 7)
        scheduler.add_request(
            EngineRequest("waiting-a", prefix + (3,), max_new_tokens=1)
        )
        scheduler.add_request(
            EngineRequest("waiting-b", prefix + (4,), max_new_tokens=1)
        )
        return scheduler

    assert build(cached_waiters=False).should_drain_before_admission() is True
    assert build(cached_waiters=True).should_drain_before_admission() is False


def test_terminal_cohort_drains_when_prefill_shaping_can_grow_batch() -> None:
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=4,
        max_num_seqs=2,
        page_size=4,
    )
    scheduler.add_request(EngineRequest("running-a", (1,), max_new_tokens=1))
    scheduler.add_request(EngineRequest("running-b", (2,), max_new_tokens=1))
    first = scheduler.schedule()
    scheduler.update({first.scheduled[0].request_id: (1000,)})
    scheduler.add_request(
        EngineRequest("waiting-a", tuple(range(10, 18)), max_new_tokens=1)
    )
    scheduler.add_request(
        EngineRequest("waiting-b", tuple(range(20, 28)), max_new_tokens=1)
    )

    # Releasing the terminal slot lets both long prompts split the next budget.
    assert scheduler.should_drain_before_admission() is True
    plan = scheduler.schedule()
    assert [chunk.request_id for chunk in plan.scheduled] == ["waiting-a"]
    assert plan.scheduled[0].num_tokens == 4


def test_terminal_cohort_drain_preserves_bounded_skip_ahead() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=3, page_size=2),
        max_num_batched_tokens=3,
        max_num_seqs=2,
    )
    scheduler.add_request(EngineRequest("terminal", (1, 2, 3), max_new_tokens=1))
    scheduler.schedule()
    scheduler.add_request(
        EngineRequest("blocked", (10, 11, 12, 13, 14), max_new_tokens=1)
    )
    scheduler.add_request(EngineRequest("small", (20, 21), max_new_tokens=1))

    assert scheduler.should_drain_before_admission() is False
    assert [chunk.request_id for chunk in scheduler.schedule().scheduled] == ["small"]


def test_terminal_cohort_drain_requires_a_larger_safe_cohort() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=5, page_size=1),
        max_num_batched_tokens=2,
        max_num_seqs=3,
        max_num_partial_prefills=1,
    )
    scheduler.add_request(EngineRequest("terminal-a", (1,), max_new_tokens=1))
    scheduler.add_request(EngineRequest("terminal-b", (2,), max_new_tokens=1))
    scheduler.schedule()
    scheduler.add_request(EngineRequest("waiting-a", (10,), max_new_tokens=4))
    scheduler.add_request(EngineRequest("waiting-b", (20,), max_new_tokens=2))

    assert scheduler.should_drain_before_admission() is False


def test_terminal_cohort_priority_probe_is_bounded_for_large_waiting_queue(
    monkeypatch,
) -> None:
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=2,
        max_num_seqs=2,
        page_size=4,
        priority_age_s=60.0,
        clock=lambda: 0.0,
    )
    scheduler.add_request(
        EngineRequest("running-a", (1,), max_new_tokens=1, priority=-100)
    )
    scheduler.add_request(
        EngineRequest("running-b", (2,), max_new_tokens=1, priority=-100)
    )
    first = scheduler.schedule()
    scheduler.update({first.scheduled[0].request_id: (1000,)})
    for index in range(10_000):
        priority = (
            1
            if index == 9_876
            else 2
            if index == 8_765
            else 1_000 + index
        )
        scheduler.add_request(
            EngineRequest(
                f"waiting-{index:05d}",
                (10_000 + index,),
                max_new_tokens=1,
                priority=priority,
            )
        )

    probes: list[tuple[int, ...]] = []
    original_probe = cache.peek_cached_tokens

    def counted_probe(tokens):
        probes.append(tokens)
        return original_probe(tokens)

    def fail_full_iteration(_queue):
        raise AssertionError("terminal cohort must not enumerate the full queue")

    def fail_sort(*_args, **_kwargs):
        raise AssertionError("terminal cohort must not sort the full queue")

    monkeypatch.setattr(cache, "peek_cached_tokens", counted_probe)
    monkeypatch.setattr(type(scheduler._waiting), "__iter__", fail_full_iteration)
    monkeypatch.setattr("builtins.sorted", fail_sort)

    assert scheduler.should_drain_before_admission() is True
    assert probes and len(probes) <= 6
    assert set(probes) <= {(19_876,), (18_765,)}


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


@pytest.mark.parametrize("terminal_text", ["a", "axz!"])
async def test_backend_preserves_stable_text_when_terminal_decode_disagrees(
    monkeypatch,
    terminal_text: str,
) -> None:
    class _DisagreeingDetokenizer:
        def __init__(
            self,
            _tokenizer,
            *,
            skip_special_tokens: bool = True,
        ) -> None:
            del skip_special_tokens
            self.stable = ""

        def push(self, token_ids) -> str:
            self.stable += "".join(
                chr(ord("a") + token_id % 26) for token_id in token_ids
            )
            return self.stable

        def finalize(self) -> str:
            return terminal_text

    monkeypatch.setattr(
        engine_loop_module,
        "IncrementalDetokenizer",
        _DisagreeingDetokenizer,
    )
    backend = KairyuBackend(
        tokenizer=_CharTokenizer(),
        runner=_PositionRunner(),
        pipeline_depth=1,
    )
    request = GenerationRequest(
        request_id="terminal-disagreement",
        prompt="one two three",
        sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
    )

    try:
        partials = [partial async for partial in backend.stream(request)]
    finally:
        await backend.shutdown()

    texts = [partial.completions[0].text for partial in partials]
    assembled = ""
    for text in texts:
        assert text.startswith(assembled)
        assembled += text[len(assembled) :]

    assert partials[-1].finished
    assert partials[-1].completions[0].finish_reason == "length"
    assert texts[-1] == assembled == "abc"


def test_terminal_flush_can_complete_a_held_back_stop_string(monkeypatch) -> None:
    class _FlushDetokenizer:
        def __init__(
            self,
            _tokenizer,
            *,
            skip_special_tokens: bool = True,
        ) -> None:
            del skip_special_tokens
            self.stable = ""
            self._stable_by_push = iter(("hello", "helloST", "helloST"))

        def push(self, _token_ids) -> str:
            self.stable = next(self._stable_by_push)
            return self.stable

        def finalize(self) -> str:
            return "helloSTOPtail"

    monkeypatch.setattr(
        engine_loop_module,
        "IncrementalDetokenizer",
        _FlushDetokenizer,
    )
    loop, _ = _loop(1, _PositionRunner())
    loop.submit(
        "terminal-flush-stop",
        "one two three",
        SamplingParams(max_tokens=3, stop=("STOP",), ignore_eos=True),
    )

    updates = [
        update
        for request_id, update in _drive(loop)
        if request_id == "terminal-flush-stop"
    ]

    assert [update.text for update in updates] == ["he", "hell", "hello"]
    assert updates[-1].finished
    assert updates[-1].finish_reason == "stop"
    loop.close()


def test_interleaved_special_policy_and_raw_logprob_metadata_are_request_scoped() -> None:
    scripts = {
        "skip-special": (2, 3),
        "keep-special": (2, 3),
    }
    loop, tokenizer = _raw_special_loop(scripts, with_logprobs=True)
    loop.submit(
        "skip-special",
        "skip prompt",
        SamplingParams(
            max_tokens=2,
            ignore_eos=True,
            logprobs=3,
            skip_special_tokens=True,
        ),
    )
    loop.submit(
        "keep-special",
        "keep prompt",
        SamplingParams(
            max_tokens=2,
            ignore_eos=True,
            logprobs=3,
            skip_special_tokens=False,
        ),
    )

    final = {
        request_id: update
        for request_id, update in _drive(loop)
        if update.finished
    }

    skipped = final["skip-special"]
    kept = final["keep-special"]
    assert skipped.outputs == kept.outputs == (2, 3)
    assert skipped.text == "�"
    assert kept.text == "<tool>�"

    assert skipped.logprob_content is not None
    assert kept.logprob_content is not None
    assert [entry.token for entry in skipped.logprob_content] == [
        "<tool>",
        "<0xE3>",
    ]
    assert [entry.token for entry in kept.logprob_content] == [
        "<tool>",
        "<0xE3>",
    ]
    assert skipped.logprob_content[0].bytes_ == ()
    assert kept.logprob_content[0].bytes_ == tuple(b"<tool>")
    replacement_bytes = tuple("�".encode())
    assert skipped.logprob_content[1].bytes_ == replacement_bytes
    assert kept.logprob_content[1].bytes_ == replacement_bytes

    skipped_top = skipped.logprob_content[0].top
    kept_top = kept.logprob_content[0].top
    assert [entry.token for entry in skipped_top] == [
        "<eos>",
        "fallback99",
        "fallback-1",
    ]
    assert [entry.token for entry in kept_top] == [
        "<eos>",
        "fallback99",
        "fallback-1",
    ]
    assert skipped_top[0].bytes_ == ()
    assert kept_top[0].bytes_ == tuple(b"<eos>")
    assert skipped_top[1].bytes_ == kept_top[1].bytes_ == tuple(b"fallback99")
    # This custom decoder accepts a negative sentinel; prove raw lookup does
    # not alias vocabulary[-1]. Native HF validation for invalid signed/u32
    # IDs remains fail-loud.
    assert skipped_top[2].bytes_ == kept_top[2].bytes_ == tuple(b"fallback-1")
    # Cumulative logprob snapshots are rebuilt every step, but the potentially
    # large raw vocabulary table is materialized only once per engine loop.
    assert tokenizer.vocab_calls == 1
    loop.close()


@pytest.mark.parametrize("depth", [1, 2])
def test_terminating_stop_token_is_not_visible_but_remains_in_outputs(depth: int) -> None:
    runner = _PositionRunner()
    loop, scheduler = _loop(depth, runner)
    loop.submit(
        "token-stop",
        "one two three",
        SamplingParams(max_tokens=6, stop_token_ids=(1,)),
    )

    updates = [
        update
        for request_id, update in _drive(loop)
        if request_id == "token-stop"
    ]
    final = updates[-1]

    assert final.finished
    assert final.finish_reason == "stop"
    assert final.outputs == (0, 1)
    assert final.text == "a"
    assert all("b" not in update.text for update in updates)
    assert scheduler.states == {}
    loop.close()


def test_only_the_occurrence_that_actually_stops_is_hidden() -> None:
    class _RepeatedStopRunner(_PositionRunner):
        sequence = (0, 2, 0)

        def execute(self, scheduled, states):
            sampled = super().execute(scheduled, states)
            return {
                request_id: tuple(
                    SampledToken(self.sequence[token.token_id])
                    for token in tokens
                )
                for request_id, tokens in sampled.items()
            }

    loop, _ = _loop(1, _RepeatedStopRunner())
    loop.submit(
        "repeated",
        "one two three",
        SamplingParams(max_tokens=4, stop_token_ids=(0,), min_tokens=2),
    )

    final = next(
        update
        for request_id, update in _drive(loop)
        if request_id == "repeated" and update.finished
    )

    assert final.outputs == (0, 2, 0)
    assert final.text == "ac"
    assert final.finish_reason == "stop"
    loop.close()


def test_ignored_eos_that_finishes_by_length_remains_visible() -> None:
    loop, _ = _loop(1, _PositionRunner())
    loop._default_eos = 0
    loop.submit(
        "ignored",
        "one two three",
        SamplingParams(max_tokens=1, ignore_eos=True),
    )

    final = next(
        update
        for request_id, update in _drive(loop)
        if request_id == "ignored" and update.finished
    )

    assert final.outputs == (0,)
    assert final.text == "a"
    assert final.finish_reason == "length"
    loop.close()


def test_terminal_special_token_stays_hidden_even_when_specials_are_requested() -> None:
    loop, _ = _raw_special_loop({"terminal-special": (4, 1)})
    loop.submit(
        "terminal-special",
        "keep prompt",
        SamplingParams(max_tokens=3, skip_special_tokens=False),
    )

    final = next(
        update
        for request_id, update in _drive(loop)
        if request_id == "terminal-special" and update.finished
    )

    assert final.outputs == (4, 1)
    assert final.text == "x"
    assert final.finish_reason == "stop"
    loop.close()


@pytest.mark.parametrize(
    ("skip_special_tokens", "expected_text"),
    [(True, ""), (False, "<eos>")],
)
def test_ignored_eos_visibility_obeys_special_token_policy(
    skip_special_tokens: bool,
    expected_text: str,
) -> None:
    request_id = f"ignored-special-{skip_special_tokens}"
    loop, _ = _raw_special_loop({request_id: (1,)})
    loop.submit(
        request_id,
        "keep prompt",
        SamplingParams(
            max_tokens=1,
            ignore_eos=True,
            skip_special_tokens=skip_special_tokens,
        ),
    )

    final = next(
        update
        for update_request_id, update in _drive(loop)
        if update_request_id == request_id and update.finished
    )

    assert final.outputs == (1,)
    assert final.text == expected_text
    assert final.finish_reason == "length"
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


def test_speculative_verification_barrier_overrides_depth_five_loop() -> None:
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
    loop, _ = _loop(5, speculative, scheduler=scheduler)
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
    class _NativePipelineScheduler(Scheduler):
        def should_drain_before_admission(self) -> bool:
            raise AssertionError("native pipeline admission must remain unchanged")

    class _Stage:
        def __init__(self, sampler=None) -> None:
            self.sampler = sampler

        def execute(self, step_index, scheduled, states):
            if self.sampler is None:
                return None
            return self.sampler.execute(scheduled, states)

    target = _PositionRunner(base=1000)
    runner = PipelinedModelRunner((_Stage(), _Stage(target)))
    scheduler = _NativePipelineScheduler(
        RadixKVCache(num_pages=64, page_size=4),
        max_num_batched_tokens=8,
        max_num_seqs=8,
        page_size=4,
    )
    loop, scheduler = _loop(2, runner, scheduler=scheduler)
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


async def test_depth_one_backend_overlaps_output_with_next_step_and_returns_stop(
    monkeypatch,
) -> None:
    detokenize_entered = threading.Event()
    release_detokenize = threading.Event()
    later_device_step_finished = threading.Event()
    detokenize_threads: list[int] = []

    class _GatedDetokenizer:
        def __init__(
            self,
            _tokenizer,
            *,
            skip_special_tokens: bool = True,
        ) -> None:
            del skip_special_tokens
            self.stable = ""

        def push(self, token_ids) -> str:
            detokenize_threads.append(threading.get_ident())
            detokenize_entered.set()
            if not release_detokenize.wait(2):
                raise TimeoutError("test did not release detokenization")
            self.stable += "".join(
                chr(ord("a") + token_id % 26) for token_id in token_ids
            )
            return self.stable

        def finalize(self) -> str:
            return self.stable

        @staticmethod
        def decode(token_ids) -> str:
            return "".join(
                chr(ord("a") + token_id % 26) for token_id in token_ids
            )

    class _OverlapRunner(_PositionRunner):
        def __init__(self) -> None:
            super().__init__()
            self.threads: list[int] = []

        def execute(self, scheduled, states):
            self.threads.append(threading.get_ident())
            if any(
                not chunk.is_prefill and chunk.position >= 1
                for chunk in scheduled
            ):
                if not detokenize_entered.wait(2):
                    raise TimeoutError("detokenization did not start")
                later_device_step_finished.set()
            return super().execute(scheduled, states)

    monkeypatch.setattr(
        engine_loop_module,
        "IncrementalDetokenizer",
        _GatedDetokenizer,
    )
    runner = _OverlapRunner()
    backend = KairyuBackend(
        tokenizer=_CharTokenizer(),
        runner=runner,
        pipeline_depth=1,
    )
    finish_threads: list[int] = []
    original_finish_early = backend._scheduler.finish_early

    def tracked_finish_early(request_id: str) -> None:
        finish_threads.append(threading.get_ident())
        original_finish_early(request_id)

    monkeypatch.setattr(
        backend._scheduler,
        "finish_early",
        tracked_finish_early,
    )
    request = GenerationRequest(
        request_id="output-lane",
        prompt="one two three",
        sampling_params=SamplingParams(max_tokens=4, ignore_eos=True),
    )
    generation = asyncio.create_task(backend.generate(request))

    try:
        detokenize_started = await asyncio.to_thread(
            detokenize_entered.wait,
            2,
        )
        device_progressed = await asyncio.to_thread(
            later_device_step_finished.wait,
            2,
        )
        assert detokenize_started
        assert device_progressed
        assert not generation.done()
        release_detokenize.set()
        result = await asyncio.wait_for(generation, 5)
        assert result.finished
        assert detokenize_threads
        assert set(detokenize_threads).isdisjoint(runner.threads)

        stopped = await backend.generate(
            GenerationRequest(
                request_id="output-stop",
                prompt="one two three",
                sampling_params=SamplingParams(
                    max_tokens=4,
                    stop=("a",),
                    ignore_eos=True,
                ),
            )
        )
        assert stopped.finished
        assert stopped.completions[0].text == ""
        assert stopped.completions[0].finish_reason == "stop"
        assert finish_threads
        assert set(finish_threads) <= set(runner.threads)
        assert set(finish_threads).isdisjoint(detokenize_threads)
    finally:
        release_detokenize.set()
        if not generation.done():
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
        await backend.shutdown()


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
