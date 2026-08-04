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


def _drive(loop: EngineLoop, limit: int = 200) -> list[tuple[str, StreamUpdate]]:
    updates: list[tuple[str, StreamUpdate]] = []
    for _ in range(limit):
        if not loop.has_work():
            return updates
        updates.extend(loop.step())
    raise AssertionError("unified loop did not drain")


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


def test_terminal_cohort_does_not_drain_when_token_budget_cannot_grow_batch() -> None:
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

    # One long prefill consumes the whole token budget with or without the
    # terminal slot, so draining would only make the pipeline shallower.
    assert scheduler.should_drain_before_admission() is False
    plan = scheduler.schedule()
    assert [chunk.request_id for chunk in plan.scheduled] == ["waiting-a"]
    assert plan.scheduled[0].num_tokens == 4


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
    assert probes == [(19_876,), (18_765,)]


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
