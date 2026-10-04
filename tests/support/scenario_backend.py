"""ScenarioBackend: a scripted in-process ``EngineBackend`` (M20 WP-02b).

Gate and conformance tests drive the real server stack with scripted turns:
generic-envelope tool calls, inline or typed reasoning, N-token outputs,
delta-native streaming with chunking and latency, finish reasons, exact
usage, and the native engine's context-length validation. Importing this
module registers nothing; ``register_scenario_backend`` opts a launcher in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from functools import partial

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    prompt_with_tool_intent,
)
from kairyu.engine.prompt import prompt_kind, prompt_text, supplied_prompt_token_ids
from kairyu.engine.registry import register_backend, uses_builtin_backend_contract
from kairyu.outputs import CompletionOutput
from tests.support.scenario_script import (
    Part,
    Piece,
    Reasoning,
    Scenario,
    ScriptedFailure,
    Turn,
    channel_text,
    chunk_delay_s,
    chunk_pieces,
    fails_before,
    inline_part_texts,
    limit_pieces,
    pending_user_text,
    sleep_s,
    toy_tokens,
    unary_delay_s,
)

_TEXT = "text"
_REASONING = "reasoning"


@dataclass(frozen=True)
class ScenarioCall:
    """One dispatched generation, recorded for assertions.

    ``prompt`` is the execution text the engine tokenized, including the
    native tool-intent suffix (``prompt_with_tool_intent``).
    """

    index: int
    request: GenerationRequest
    prompt: str
    prompt_tokens: int
    turn: Turn


@dataclass(frozen=True)
class _Plan:
    turn: Turn
    pieces: tuple[Piece, ...]
    chunks: tuple[tuple[Piece, ...], ...]
    finish_reason: str
    usage: GenerationUsage


def _part_pieces(index: int, part: Part) -> Iterable[Piece]:
    if isinstance(part, Reasoning) and part.typed:
        return (Piece(index, _REASONING, text) for text in toy_tokens(part.text))
    return (Piece(index, _TEXT, text) for text in inline_part_texts(part))


def engine_pieces(turn: Turn) -> tuple[Piece, ...]:
    """The token pieces a native engine would emit for ``turn``."""

    return tuple(
        piece
        for index, part in enumerate(turn.parts)
        for piece in _part_pieces(index, part)
    )


def _delta_output(
    text: str,
    text_delta: str,
    reasoning: str | None,
    reasoning_delta: str,
    token_count: int,
    finish_reason: str | None,
) -> CompletionOutput:
    """A delta-native completion (the native and vLLM adapters' stream shape)."""

    return CompletionOutput(
        index=0,
        text=text,
        token_ids=tuple(range(token_count)),
        finish_reason=finish_reason,
        text_delta=text_delta,
        text_offset=len(text) - len(text_delta),
        reasoning_content=reasoning,
        reasoning_delta=None if reasoning is None else reasoning_delta,
        reasoning_offset=None if reasoning is None else len(reasoning) - len(reasoning_delta),
    )


class ScenarioBackend:
    """Replays a ``Scenario``; the call log is its only mutable state."""

    def __init__(self, scenario: Scenario, *, max_model_len: int | None = None) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError("ScenarioBackend requires a Scenario")
        if max_model_len is not None and (type(max_model_len) is not int or max_model_len < 1):
            raise ValueError("max_model_len must be a positive integer or None")
        self._scenario = scenario
        self.max_model_len = max_model_len
        self._calls: list[ScenarioCall] = []

    @property
    def calls(self) -> tuple[ScenarioCall, ...]:
        return tuple(self._calls)

    def validate_request_before_prepare(self, request: GenerationRequest) -> None:
        """Scalar capability checks; tokenization waits for ``prepare_request``."""

        params = request.sampling_params
        if params.n != 1 or params.best_of not in (None, 1):
            raise ValueError("ScenarioBackend scripts single-candidate generations")
        if params.forced_token_ids is not None:
            raise ValueError("ScenarioBackend does not support forced_token_ids")
        if prompt_kind(request.prompt) == "multimodal":
            raise ValueError(
                "ScenarioBackend does not support multimodal prompts; "
                "image data was not dispatched"
            )

    def validate_request(self, request: GenerationRequest) -> None:
        self.validate_request_before_prepare(request)
        self._admitted_prompt(request)

    async def prepare_request(self, request: GenerationRequest) -> None:
        self.validate_request(request)

    async def count_prompt_tokens_async(self, prompt: str) -> int:
        return len(toy_tokens(prompt))

    def context_length_error(self, prompt_tokens: int, max_tokens: int | None) -> Exception:
        """Build the context-overflow error; the only place it is constructed.

        Mirrors ``EngineLoop._max_new_tokens`` and ``_validate_context_length``
        (``kairyu/engine/engine_loop.py``), which raise ``ValueError`` today.
        WP-04 replaces this with its typed ``ContextLengthExceededError``.
        """

        if max_tokens is None:
            return ValueError(
                f"prompt tokens ({prompt_tokens}) already fill "
                f"max_model_len ({self.max_model_len})"
            )
        return ValueError(
            f"prompt tokens ({prompt_tokens}) plus max_tokens "
            f"({max_tokens}) exceed max_model_len "
            f"({self.max_model_len})"
        )

    def _admitted_prompt(self, request: GenerationRequest) -> tuple[str, int, int | None]:
        """Return (execution text, prompt tokens, output budget) or raise overflow."""

        prompt = prompt_with_tool_intent(request)
        token_ids = supplied_prompt_token_ids(prompt)
        text = prompt_text(prompt) or ""
        prompt_tokens = len(token_ids) if token_ids is not None else len(toy_tokens(text))
        max_tokens = request.sampling_params.max_tokens
        if self.max_model_len is None:
            return text, prompt_tokens, max_tokens
        if max_tokens is None:
            # OpenAI contract for an omitted limit: the remaining context.
            remaining = self.max_model_len - prompt_tokens
            if remaining < 1:
                raise self.context_length_error(prompt_tokens, None)
            return text, prompt_tokens, remaining
        if prompt_tokens + max_tokens > self.max_model_len:
            raise self.context_length_error(prompt_tokens, max_tokens)
        return text, prompt_tokens, max_tokens

    def _dispatch(self, request: GenerationRequest) -> _Plan:
        # Rejections precede the call log so refused work is never "dispatched".
        self.validate_request_before_prepare(request)
        text, prompt_tokens, budget = self._admitted_prompt(request)
        index = len(self._calls)
        turn = self._scenario.select(index, pending_user_text(text))
        self._calls.append(ScenarioCall(index, request, text, prompt_tokens, turn))
        pieces, cut = limit_pieces(engine_pieces(turn), budget)
        usage = turn.usage or GenerationUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(pieces),
        )
        return _Plan(
            turn=turn,
            pieces=pieces,
            chunks=chunk_pieces(pieces, turn.chunk_tokens),
            finish_reason="length" if cut else turn.finish_reason,
            usage=usage,
        )

    @staticmethod
    def _result(
        request: GenerationRequest,
        completion: CompletionOutput,
        *,
        finished: bool,
        usage: GenerationUsage | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(completion,),
            finished=finished,
            usage=usage,
            prompt_token_ids=supplied_prompt_token_ids(request.prompt) or (),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        plan = self._dispatch(request)
        await sleep_s(unary_delay_s(plan.turn, len(plan.chunks)))
        if plan.turn.fail_after_chunks is not None:
            raise ScriptedFailure("scripted generation failure")
        completion = CompletionOutput(
            index=0,
            text=channel_text(plan.pieces, _TEXT),
            token_ids=tuple(range(len(plan.pieces))),
            finish_reason=plan.finish_reason,
            reasoning_content=(
                channel_text(plan.pieces, _REASONING)
                if plan.turn.has_typed_reasoning
                else None
            ),
        )
        return self._result(request, completion, finished=True, usage=plan.usage)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        plan = self._dispatch(request)
        typed = plan.turn.has_typed_reasoning
        count = len(plan.chunks)
        text = reasoning = ""
        tokens = 0
        for position, chunk in enumerate(plan.chunks):
            if fails_before(plan.turn, position, count):
                raise ScriptedFailure(f"scripted failure after {position} chunks")
            await sleep_s(chunk_delay_s(plan.turn, position))
            text_delta = channel_text(chunk, _TEXT)
            reasoning_delta = channel_text(chunk, _REASONING)
            text += text_delta
            reasoning += reasoning_delta
            tokens += len(chunk)
            completion = _delta_output(
                text, text_delta, reasoning if typed else None, reasoning_delta, tokens, None
            )
            yield self._result(request, completion, finished=False)
        if fails_before(plan.turn, count, count):
            raise ScriptedFailure(f"scripted failure after {count} chunks")
        final = _delta_output(
            text, "", reasoning if typed else None, "", tokens, plan.finish_reason
        )
        yield self._result(request, final, finished=True, usage=plan.usage)

    async def shutdown(self) -> None:
        return None


def register_scenario_backend(scenario: Scenario, *, name: str = "scenario") -> None:
    """Expose ``scenario`` to DSL/deployment configs as backend ``name``.

    For test and gate launchers only. Config options bind to the remaining
    constructor parameters (``max_model_len``), which the registry validates
    against this factory's signature.
    """

    if uses_builtin_backend_contract(name):
        raise ValueError(f"refusing to shadow built-in backend {name!r}")
    register_backend(name, partial(ScenarioBackend, scenario))
