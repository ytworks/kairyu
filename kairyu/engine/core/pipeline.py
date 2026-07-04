"""PP inter-step pipelining: async runner contract + stage pipeline (m6 D5).

The review made this structure load-bearing: PP=2 cannot meet B4 under a
synchronous ``execute()`` contract (the pipe drains every step, utilization
caps at ~1.33x). So the runner contract gains the async submit/handle form
reserved by m2 §5 item 3, and ``PipelinedEngineCore`` keeps up to ``depth``
scheduler steps in flight — step N occupies stage 1 while step N+1 occupies
stage 0, each stage always processing a FULL batch (decode is never split into
micro-batches). CPU stage workers are deterministic tick-simulated; the GPU
phase swaps real stage execution (and the Communicator hidden-state hop)
behind the same seams. Bubble-fraction reporting is mandatory (G2 B4).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Protocol

from kairyu.engine.core.engine_core import ModelRunner, StepOutput, token_ids
from kairyu.engine.core.scheduler import EngineRequest, ScheduledChunk, Scheduler


class StepHandle(Protocol):
    def result(self) -> StepOutput:
        """Block until this step's sampled tokens land; return them."""
        ...


class AsyncModelRunner(Protocol):
    """The submit/handle runner contract (m2 §5 item 3, promoted by m6 D5)."""

    def submit(
        self,
        step_index: int,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> StepHandle: ...


class _ResolvedHandle:
    def __init__(self, sampled: StepOutput) -> None:
        self._sampled = sampled

    def result(self) -> StepOutput:
        return self._sampled


class SyncRunnerAdapter:
    """Wraps a legacy synchronous ModelRunner as submit + immediate resolve."""

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner

    def submit(
        self,
        step_index: int,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> StepHandle:
        return _ResolvedHandle(self._runner.execute(scheduled, states))


class StageWorker(Protocol):
    def execute(
        self,
        step_index: int,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> StepOutput | None:
        """Run this stage's layer slice; the final stage returns sampled tokens."""
        ...


class _InFlightStep:
    __slots__ = ("index", "scheduled", "states", "next_stage", "sampled", "done")

    def __init__(
        self,
        index: int,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> None:
        self.index = index
        self.scheduled = scheduled
        self.states = states
        self.next_stage = 0
        self.sampled: StepOutput = {}
        self.done = False


class _PipelineHandle:
    def __init__(self, runner: PipelinedModelRunner, step: _InFlightStep) -> None:
        self._runner = runner
        self._step = step

    def result(self) -> StepOutput:
        while not self._step.done:
            self._runner._tick()
        return self._step.sampled


class PipelinedModelRunner:
    """Tick-simulated stage pipeline with stage affinity and bubble accounting.

    Each tick every stage advances at most one step; steps traverse stages in
    submission order, one stage per tick — so with two steps in flight both
    stages are busy every tick (steady state), and with one step in flight the
    other stage idles (the depth-1 drain the design forbids for B4 runs).
    """

    def __init__(self, stages: tuple[StageWorker, ...]) -> None:
        if len(stages) < 2:
            raise ValueError(f"pipeline needs >= 2 stages, got {len(stages)}")
        self._stages = stages
        self._in_flight: list[_InFlightStep] = []
        self._busy_ticks = 0
        self._total_stage_ticks = 0
        self.step_chunk_counts: list[int] = []

    def submit(
        self,
        step_index: int,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> StepHandle:
        step = _InFlightStep(step_index, scheduled, states)
        self._in_flight.append(step)
        self.step_chunk_counts.append(len(scheduled))
        return _PipelineHandle(self, step)

    @property
    def bubble_fraction(self) -> float:
        """Idle stage-ticks / total stage-ticks while the pipe was non-empty (B4)."""
        if self._total_stage_ticks == 0:
            return 0.0
        return 1.0 - self._busy_ticks / self._total_stage_ticks

    def _tick(self) -> None:
        if not self._in_flight:
            raise RuntimeError("pipeline tick with no in-flight steps")
        last_stage = len(self._stages) - 1
        # advance back-to-front so a step moves exactly one stage per tick
        for stage_index in reversed(range(len(self._stages))):
            self._total_stage_ticks += 1
            step = next(
                (s for s in self._in_flight if s.next_stage == stage_index), None
            )
            if step is None:
                continue  # bubble: this stage idles this tick
            output = self._stages[stage_index].execute(
                step.index, step.scheduled, step.states
            )
            self._busy_ticks += 1
            if stage_index == last_stage:
                step.sampled = output or {}
                step.done = True
                self._in_flight.remove(step)
            else:
                step.next_stage += 1


class PipelinedEngineCore:
    """Engine loop over the async runner contract: up to ``depth`` steps in flight."""

    def __init__(self, scheduler: Scheduler, runner: AsyncModelRunner, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self._scheduler = scheduler
        self._runner = runner
        self._depth = depth
        self._outputs: dict[str, tuple[int, ...]] = {}

    def add_request(self, request: EngineRequest) -> None:
        self._scheduler.add_request(request)

    def run_to_completion(self) -> dict[str, tuple[int, ...]]:
        pending: deque[StepHandle] = deque()
        step_index = 0
        while True:
            if self._scheduler.has_unfinished() and len(pending) < self._depth:
                plan = self._scheduler.schedule()
                # prompts too large to ever fit are rejected in schedule() (C2)
                for request_id in self._scheduler.drain_rejected():
                    self._outputs[request_id] = self._scheduler.output_tokens(request_id)
                if plan.scheduled:
                    pending.append(
                        self._runner.submit(step_index, plan.scheduled, self._scheduler.states)
                    )
                    step_index += 1
                    continue
            if pending:
                sampled = pending.popleft().result()
                finished = self._scheduler.update(token_ids(sampled)) if sampled else ()
                for request_id in finished:
                    self._outputs[request_id] = self._scheduler.output_tokens(request_id)
                continue
            if self._scheduler.has_unfinished():
                # nothing in flight and nothing schedulable: force the stuck
                # waiting head to finish rather than crash the engine (C2)
                head = self._scheduler.reject_waiting_head()
                if head is not None:
                    self._outputs[head] = self._scheduler.output_tokens(head)
                    continue
            return dict(self._outputs)
