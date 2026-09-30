"""A generation that degenerates into an endless repetition is stopped early.

DeepSeek V4.1 at its official sampling occasionally starts a legitimate run
of one character ("0.000…", "IIII…") and never leaves it, spending the whole
131,072-token budget (~34 min) inside its reasoning (measured 2026-09-30 on
both the six-GPU overlay and the parent TP8 configuration).
"""

from __future__ import annotations

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.orchestration.conductor import Conductor, RoleSpec
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams


class LoopingBackend:
    """Streams reasoning that collapses into zeros; answers a closed-span retry."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.chunks_sent = 0
        self.stream_closed = False

    async def stream(self, request: GenerationRequest):
        self.requests.append(request)
        if "</think>" in str(request.prompt):
            yield await self._answer(request)
            return
        offset = 0
        try:
            for piece in ["Check the value 0.0", *(["0" * 16] * 100_000)]:
                self.chunks_sent += 1
                yield GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=(),
                            reasoning_delta=piece,
                            reasoning_offset=offset,
                            text_delta="",
                            text_offset=0,
                        ),
                    ),
                )
                offset += len(piece)
        finally:
            self.stream_closed = True

    async def _answer(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text="The value is 0.", token_ids=()),),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return await self._answer(request)

    async def shutdown(self) -> None:
        return None


async def test_a_looping_attempt_is_stopped_and_answered_from_its_prior_reasoning():
    backend = LoopingBackend()
    roles = (
        RoleSpec(
            name="answer",
            worker="w",
            role_type="publisher",
            prompt="[answer] {query}",
            prompt_suffix="<think>",
            reasoning_close_tag="</think>",
            repetition_stop_chars=256,
        ),
    )
    conductor = Conductor(
        roles,
        {"w": backend},
        final_sampling_params=SamplingParams(max_tokens=131072),
        public_output_floor=64,
    )
    result = await conductor.run("task")

    assert backend.chunks_sent < 40  # stopped within a few hundred characters
    assert backend.stream_closed  # the upstream request was cancelled
    assert result.final_text == "The value is 0."
    retry = backend.requests[-1]
    # The continuation keeps the reasoning before the loop and drops the loop.
    assert retry.prompt == "[answer] task<think>Check the value 0.</think>\n\n"
    event = next(e for e in result.trace if e.kind == "retry:empty_output")
    assert event.metadata["stopped"] == "repetition"
