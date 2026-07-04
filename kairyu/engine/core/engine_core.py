"""EngineCore step loop: scheduler plans, a pluggable ModelRunner executes.

The GPU runner (M12+) and CPU test stubs implement the same ``ModelRunner``
protocol, so the loop, scheduling and KV behavior verified here carry over
unchanged. ``StepOutput`` is the one written name of the runner's return
contract (m8 D2): always a tuple of ``SampledToken`` per request — usually
length 1, longer under speculative decoding (m8 D3/D4).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, ScheduledChunk, Scheduler

StepOutput = dict[str, tuple[SampledToken, ...]]


def token_ids(step_output: Mapping[str, tuple[SampledToken, ...]]) -> dict[str, list[int]]:
    """Convert a runner's StepOutput into the Scheduler.update payload."""
    return {
        request_id: [token.token_id for token in tokens]
        for request_id, tokens in step_output.items()
    }


def grammar_finished(
    step_output: Mapping[str, tuple[SampledToken, ...]], already_finished: tuple[str, ...]
) -> tuple[str, ...]:
    """Requests whose grammar terminated this step and were not otherwise finished."""
    return tuple(
        request_id
        for request_id, tokens in step_output.items()
        if request_id not in already_finished
        and any(token.grammar_terminated for token in tokens)
    )


class ModelRunner(Protocol):
    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> StepOutput:
        """Run one engine step; return sampled tokens per prefill-complete request."""
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
        # Prompts too large to ever fit are rejected during schedule() (C2);
        # surface them as finished-with-empty-output instead of stalling.
        rejected = self._scheduler.drain_rejected()
        for request_id in rejected:
            self._outputs[request_id] = self._scheduler.output_tokens(request_id)
        if not plan.scheduled:
            if self._scheduler.has_unfinished():
                # An empty schedule while unfinished means nothing is running to
                # free pages, so force the stuck waiting head to finish rather
                # than loop or crash the whole engine.
                head = self._scheduler.reject_waiting_head()
                if head is not None:
                    self._outputs[head] = self._scheduler.output_tokens(head)
                    return (*rejected, head)
            return rejected
        sampled = self._runner.execute(plan.scheduled, self._scheduler.states)
        finished = self._scheduler.update(token_ids(sampled)) if sampled else ()
        finished = (*rejected, *finished)
        for request_id in grammar_finished(sampled or {}, finished):
            # between update() and the next schedule(): the safe finish point
            self._scheduler.finish_early(request_id)
            finished = (*finished, request_id)
        for request_id in finished:
            self._outputs[request_id] = self._scheduler.output_tokens(request_id)
        return finished

    def run_to_completion(self) -> dict[str, tuple[int, ...]]:
        while self._scheduler.has_unfinished():
            self.step()
        return dict(self._outputs)
