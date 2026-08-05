"""Deterministic in-process backend used by all unit tests and local dev."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    prompt_with_tool_intent,
)
from kairyu.engine.prompt import (
    PromptInput,
    prompt_kind,
    prompt_text,
    supplied_prompt_token_ids,
)
from kairyu.engine.tokenizer import (
    ToyTokenizer,
    tokenize_loglikelihood_continuation,
)
from kairyu.outputs import CompletionOutput, TokenLogprob

_ECHO_TAIL_CHARS = 48
_STREAM_CHUNK_CHARS = 8
_LOGLIKELIHOOD_TOKENIZER = ToyTokenizer()


def _fake_token_ids(text: str) -> tuple[int, ...]:
    return tuple(range(len(text.split())))


class MockBackend:
    """Echoes prompts (or substring-matched canned responses) deterministically."""

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        latency_s: float = 0.0,
        tensor_parallel_size: int = 1,
    ) -> None:
        if tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}"
            )
        self._responses = dict(responses) if responses else {}
        self._latency_s = latency_s
        # recorded (not acted on) so tests can assert plumbing, design m5 D3
        self.tensor_parallel_size = tensor_parallel_size
        self._prompts_seen: list[PromptInput] = []

    @property
    def prompts_seen(self) -> tuple[PromptInput, ...]:
        return tuple(self._prompts_seen)

    def _text_for(self, prompt: str, sample_index: int) -> str:
        base = next(
            (text for key, text in self._responses.items() if key in prompt),
            f"mock:{prompt[-_ECHO_TAIL_CHARS:]}",
        )
        return base if sample_index == 0 else f"{base} #{sample_index}"

    @staticmethod
    def _execution_text(request: GenerationRequest) -> str:
        prompt = prompt_with_tool_intent(request)
        if prompt_kind(prompt) == "multimodal":
            raise ValueError(
                "MockBackend does not support multimodal image prompts; "
                "image data was not dispatched"
            )
        token_ids = supplied_prompt_token_ids(prompt)
        if token_ids is not None:
            return "tokens:" + ",".join(str(token_id) for token_id in token_ids)
        text = prompt_text(prompt)
        assert text is not None
        return text

    def validate_request(self, request: GenerationRequest) -> None:
        self._execution_text(request)

    def tokenize_loglikelihood(
        self,
        context: str,
        continuation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Resolve mock continuations through one process-stable tokenizer."""

        return tokenize_loglikelihood_continuation(
            _LOGLIKELIHOOD_TOKENIZER,
            context,
            continuation,
        )

    def _result_for(self, request: GenerationRequest) -> GenerationResult:
        execution_text = self._execution_text(request)
        forced_token_ids = request.sampling_params.forced_token_ids
        if forced_token_ids is None:
            completions = tuple(
                CompletionOutput(
                    index=i,
                    text=(text := self._text_for(execution_text, i)),
                    token_ids=_fake_token_ids(text),
                    cumulative_logprob=0.0,
                    finish_reason="stop",
                )
                for i in range(request.sampling_params.n)
            )
        else:
            token_text = tuple(str(token_id) for token_id in forced_token_ids)
            selected_logprobs = tuple(
                -float(((token_id * 31 + position) % 997) + 1) / 1000.0
                for position, token_id in enumerate(forced_token_ids)
            )
            logprobs = tuple(
                {token_id: logprob}
                for token_id, logprob in zip(
                    forced_token_ids,
                    selected_logprobs,
                    strict=True,
                )
            )
            logprob_content = tuple(
                TokenLogprob(
                    token=token,
                    token_id=token_id,
                    logprob=logprob,
                    bytes_=tuple(token.encode()),
                )
                for token, token_id, logprob in zip(
                    token_text,
                    forced_token_ids,
                    selected_logprobs,
                    strict=True,
                )
            )
            completions = tuple(
                CompletionOutput(
                    index=i,
                    text=" ".join(token_text),
                    token_ids=forced_token_ids,
                    cumulative_logprob=sum(selected_logprobs),
                    logprobs=logprobs,
                    logprob_content=logprob_content,
                    finish_reason="length",
                )
                for i in range(request.sampling_params.n)
            )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            usage=GenerationUsage(
                prompt_tokens=(
                    len(token_ids)
                    if (token_ids := supplied_prompt_token_ids(request.prompt))
                    is not None
                    else len(execution_text.split())
                ),
                completion_tokens=sum(len(c.token_ids) for c in completions),
                cached_tokens=0,
            ),
            prompt_token_ids=supplied_prompt_token_ids(request.prompt) or (),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        # Capability rejection precedes latency and call-log mutation so an
        # unsupported modality is never observable as dispatched work.
        self.validate_request(request)
        if self._latency_s > 0:
            await asyncio.sleep(self._latency_s)
        self._prompts_seen.append(request.prompt)
        return self._result_for(request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        final = await self.generate(request)
        if request.sampling_params.forced_token_ids is not None:
            yield final
            return
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
                prompt_token_ids=supplied_prompt_token_ids(request.prompt) or (),
            )
        yield final

    async def shutdown(self) -> None:
        return None
