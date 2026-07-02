"""EngineCore step loop: scheduler plans, a pluggable ModelRunner executes.

The GPU FlashInfer runner (M2 GPU phase) and CPU test stubs implement the same
``ModelRunner`` protocol, so the loop, scheduling and KV behavior verified here
carry over unchanged. Overlap pipelining (design doc §2.2) wraps this loop in
the GPU phase; the step contract stays the same.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from kairyu.engine.core.scheduler import EngineRequest, ScheduledChunk, Scheduler


class ModelRunner(Protocol):
    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, int]:
        """Run one engine step; return sampled token per prefill-complete request."""
        ...


class EngineCore:
    def __init__(self, scheduler: Scheduler, runner: ModelRunner) -> None:
        self._scheduler = scheduler
        self._runner = runner
        self._outputs: dict[str, tuple[int, ...]] = {}

    def add_request(self, request: EngineRequest) -> None:
        self._scheduler.add_request(request)

    def has_unfinished(self) -> bool:
        return self._scheduler.has_unfinished()

    def step(self) -> tuple[str, ...]:
        plan = self._scheduler.schedule()
        if not plan.scheduled:
            if self._scheduler.has_unfinished():
                raise RuntimeError(
                    "engine stall: unfinished requests but nothing schedulable "
                    "(request larger than KV capacity?)"
                )
            return ()
        sampled = self._runner.execute(plan.scheduled, self._scheduler.states)
        finished = self._scheduler.update(sampled) if sampled else ()
        for request_id in finished:
            self._outputs[request_id] = self._scheduler.output_tokens(request_id)
        return finished

    def run_to_completion(self) -> dict[str, tuple[int, ...]]:
        while self._scheduler.has_unfinished():
            self.step()
        return dict(self._outputs)
