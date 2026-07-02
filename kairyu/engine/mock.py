"""Deterministic in-process backend used by all unit tests and local dev."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.outputs import CompletionOutput

_ECHO_TAIL_CHARS = 48
_STREAM_CHUNK_CHARS = 8


def _fake_token_ids(text: str) -> tuple[int, ...]:
    return tuple(range(len(text.split())))


class MockBackend:
    """Echoes prompts (or substring-matched canned responses) deterministically."""

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        latency_s: float = 0.0,
    ) -> None:
        self._responses = dict(responses) if responses else {}
        self._latency_s = latency_s
        self._prompts_seen: list[str] = []

    @property
    def prompts_seen(self) -> tuple[str, ...]:
        return tuple(self._prompts_seen)

    def _text_for(self, prompt: str, sample_index: int) -> str:
        base = next(
            (text for key, text in self._responses.items() if key in prompt),
            f"mock:{prompt[-_ECHO_TAIL_CHARS:]}",
        )
        return base if sample_index == 0 else f"{base} #{sample_index}"

    def _result_for(self, request: GenerationRequest) -> GenerationResult:
        completions = tuple(
            CompletionOutput(
                index=i,
                text=(text := self._text_for(request.prompt, i)),
                token_ids=_fake_token_ids(text),
                cumulative_logprob=0.0,
                finish_reason="stop",
            )
            for i in range(request.sampling_params.n)
        )
        return GenerationResult(
            request_id=request.request_id, prompt=request.prompt, completions=completions
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._latency_s > 0:
            await asyncio.sleep(self._latency_s)
        self._prompts_seen.append(request.prompt)
        return self._result_for(request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        final = await self.generate(request)
        longest = max(len(completion.text) for completion in final.completions)
        for end in range(_STREAM_CHUNK_CHARS, longest, _STREAM_CHUNK_CHARS):
            partials = tuple(
                CompletionOutput(
                    index=completion.index,
                    text=completion.text[:end],
                    token_ids=_fake_token_ids(completion.text[:end]),
                    cumulative_logprob=0.0,
                )
                for completion in final.completions
            )
            yield GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=partials,
                finished=False,
            )
        yield final

    async def shutdown(self) -> None:
        return None
