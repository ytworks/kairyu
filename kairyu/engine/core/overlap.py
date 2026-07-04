"""Overlap engine loop: CPU scheduling pipelined with device execution (design m2 §2.2).

While the device executes step N, the scheduler plans step N+1 using the
in-flight token accounting in Scheduler: decode chunks carry an explicit
``position`` so the runner never needs previously-committed token values from
the host (on GPU, the last-token slot is patched device-side — the SGLang
"future token" technique). Sampled tokens are committed via update() while the
next step is already running, so the device never waits on host bookkeeping.

The pipeline structure (schedule-ahead, bounded depth, late finish commit) is
what this module pins with CPU tests; the GPU runner slots into the same
ModelRunner protocol.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from kairyu.engine.core.engine_core import ModelRunner, token_ids
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

_DEFAULT_PIPELINE_DEPTH = 2


class OverlapEngineCore:
    def __init__(
        self,
        scheduler: Scheduler,
        runner: ModelRunner,
        pipeline_depth: int = _DEFAULT_PIPELINE_DEPTH,
    ) -> None:
        if pipeline_depth < 1:
            raise ValueError(f"pipeline_depth must be >= 1, got {pipeline_depth}")
        self._scheduler = scheduler
        self._runner = runner
        self._depth = pipeline_depth
        self._outputs: dict[str, tuple[int, ...]] = {}
        self.events: list[str] = []  # instrumentation: scheduled:N / commit:N ordering

    def add_request(self, request: EngineRequest) -> None:
        self._scheduler.add_request(request)

    def has_unfinished(self) -> bool:
        return self._scheduler.has_unfinished()

    def _commit(self, future: Future) -> None:
        sampled = future.result()
        finished = self._scheduler.update(token_ids(sampled)) if sampled else ()
        for request_id in finished:
            self._outputs[request_id] = self._scheduler.output_tokens(request_id)

    def run_to_completion(self) -> dict[str, tuple[int, ...]]:
        pending: deque[Future] = deque()
        step_index = 0
        with ThreadPoolExecutor(max_workers=1) as device:
            while True:
                if self._scheduler.has_unfinished() and len(pending) < self._depth:
                    plan = self._scheduler.schedule()
                    # prompts too large to ever fit are rejected in schedule() (C2)
                    for request_id in self._scheduler.drain_rejected():
                        self._outputs[request_id] = self._scheduler.output_tokens(request_id)
                    if plan.scheduled:
                        self.events.append(f"scheduled:{step_index}")
                        pending.append(
                            device.submit(
                                self._runner.execute, plan.scheduled, self._scheduler.states
                            )
                        )
                        step_index += 1
                        continue
                if pending:
                    self._commit(pending.popleft())
                    continue
                if self._scheduler.has_unfinished():
                    # nothing in flight and nothing schedulable: force the stuck
                    # waiting head to finish rather than crash the engine (C2)
                    head = self._scheduler.reject_waiting_head()
                    if head is not None:
                        self._outputs[head] = self._scheduler.output_tokens(head)
                        continue
                return dict(self._outputs)
