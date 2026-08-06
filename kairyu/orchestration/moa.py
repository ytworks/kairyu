"""Mixture-of-Agents: parallel diverse proposals + one synthesis pass.

Each proposer gets a distinct perspective header and seed; the synthesizer
sees all numbered proposals. Prompts share ``shared_prefix`` for KV affinity
(design doc D5).
"""

from __future__ import annotations

import asyncio
import contextvars
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from kairyu.async_thread import run_prompt_work
from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
)
from kairyu.engine.prompt import TemplatedPrompt
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import PARALLEL_TOOL_CALLS_EXTRA_ARG, SamplingParams

_DEFAULT_N_SAMPLES = 3
_PROPOSAL_TEMPERATURE = 0.9

_SYNTHESIS_TEMPLATE = (
    "Synthesize the best single answer to the question below from the candidate "
    "answers. Merge their strengths, drop errors.\n\nQuestion: {query}\n\n{proposals}\n\n"
    "Final answer:"
)


@dataclass(frozen=True)
class MoAResult:
    final_text: str
    proposals: tuple[str, ...]
    completions: tuple[CompletionOutput, ...] = ()
    usage: tuple[int, int] = (0, 0)  # (prompt_tokens, completion_tokens) summed
    cached_tokens: int = 0


@dataclass(frozen=True)
class MoAEvent:
    """A final synthesis delta or the complete MoA result."""

    kind: str  # "delta" | "result"
    text: str = ""
    completions: tuple[CompletionOutput, ...] = ()
    result: MoAResult | None = None


class MoAExecutionError(RuntimeError):
    """Sanitized carrier for partial MoA accounting after a backend failure."""

    def __init__(
        self,
        cause: BaseException,
        *,
        stage: str,
        proposals: tuple[str, ...],
        usage: tuple[int, int],
        cached_tokens: int,
        final_text: str = "",
    ) -> None:
        super().__init__(type(cause).__name__)
        self.cause = cause
        self.stage = stage
        self.proposals = proposals
        self.usage = usage
        self.cached_tokens = cached_tokens
        self.final_text = final_text


@dataclass(frozen=True)
class _MoASetup:
    session: str
    proposal_requests: tuple[GenerationRequest, ...]
    synthesis_params: SamplingParams
    cache_hint: CacheHint


_prepared_setup: contextvars.ContextVar[_MoASetup | None] = contextvars.ContextVar(
    "kairyu_prepared_moa_setup",
    default=None,
)


@contextmanager
def _reuse_prepared_moa_setup(setup: _MoASetup | None):
    """Bind an exact internal setup without changing the public MoA ABI."""

    token = _prepared_setup.set(setup)
    try:
        yield
    finally:
        _prepared_setup.reset(token)


def _build_moa_setup(
    query: str,
    n_samples: int,
    sampling_params: SamplingParams | None,
    final_sampling_params: SamplingParams | None,
    shared_prefix: str,
    session: str | None = None,
) -> _MoASetup:
    """Build request-sized proposal prompts on the prompt CPU lane."""

    if isinstance(query, TemplatedPrompt):
        raise ValueError(
            "MoA cannot derive proposal prompts from a tokenizer-owned pre-rendered chat prompt"
        )
    if isinstance(shared_prefix, TemplatedPrompt):
        raise ValueError("MoA shared_prefix cannot be a tokenizer-owned pre-rendered chat prompt")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    public_params = sampling_params or SamplingParams(
        temperature=_PROPOSAL_TEMPERATURE,
        max_tokens=1024,
    )
    synthesis_params = final_sampling_params or public_params.clone(
        temperature=0.3,
        seed=None,
    )
    internal_extra_args = dict(public_params.extra_args)
    internal_extra_args.pop(PARALLEL_TOOL_CALLS_EXTRA_ARG, None)
    params = public_params.clone(extra_args=internal_extra_args)
    session = session or uuid.uuid4().hex[:12]
    hint = CacheHint(session_id=session)
    requests = tuple(
        GenerationRequest(
            request_id=f"{session}-propose-{index}",
            prompt=(f"{shared_prefix}[proposer {index}] Answer the question: {query}"),
            sampling_params=params.clone(
                seed=index if params.seed is None else params.seed + index
            ),
            cache_hint=hint,
        )
        for index in range(n_samples)
    )
    return _MoASetup(session, requests, synthesis_params, hint)


def _build_synthesis_request(
    query: str,
    proposals: tuple[str, ...],
    setup: _MoASetup,
    shared_prefix: str,
    final_tools: tuple[Mapping[str, object], ...],
    final_tool_choice: str | Mapping[str, object] | None,
    final_tools_in_prompt: bool,
    final_parallel_tool_calls: bool | None,
) -> GenerationRequest:
    numbered = "\n\n".join(
        f"Candidate {index + 1}:\n{proposal}" for index, proposal in enumerate(proposals)
    )
    return GenerationRequest(
        request_id=f"{setup.session}-synthesize",
        prompt=(shared_prefix + _SYNTHESIS_TEMPLATE.format(query=query, proposals=numbered)),
        sampling_params=setup.synthesis_params,
        cache_hint=setup.cache_hint,
        tools=final_tools,
        tool_choice=final_tool_choice,
        tools_in_prompt=final_tools_in_prompt,
        parallel_tool_calls=final_parallel_tool_calls,
    )


async def _prepare_moa(
    backend: EngineBackend,
    query: str,
    *,
    n_samples: int,
    sampling_params: SamplingParams | None,
    final_sampling_params: SamplingParams | None,
    final_tools: tuple[Mapping[str, object], ...],
    final_tool_choice: str | Mapping[str, object] | None,
    final_tools_in_prompt: bool,
    final_parallel_tool_calls: bool | None,
    shared_prefix: str,
    usage_observer: Callable[[GenerationUsage], None] | None,
    prepared_setup: _MoASetup | None,
) -> tuple[tuple[str, ...], list[int], GenerationRequest]:
    setup = prepared_setup
    if setup is None:
        setup = await run_prompt_work(
            _build_moa_setup,
            query,
            n_samples,
            sampling_params,
            final_sampling_params,
            shared_prefix,
        )

    usage_totals = [0, 0, 0]

    def _observe() -> None:
        if usage_observer is not None:
            usage_observer(
                GenerationUsage(
                    prompt_tokens=usage_totals[0],
                    completion_tokens=usage_totals[1],
                    cached_tokens=usage_totals[2],
                )
            )

    def _account(result) -> str:
        if result.usage is not None:
            usage_totals[0] += result.usage.prompt_tokens
            usage_totals[1] += result.usage.completion_tokens
            usage_totals[2] += result.usage.cached_tokens
            _observe()
        return result.text

    async def propose(request: GenerationRequest) -> str:
        return _account(await backend.generate(request))

    proposal_results = await asyncio.gather(
        *(propose(request) for request in setup.proposal_requests),
        return_exceptions=True,
    )
    proposals_list: list[str] = []
    first_error: BaseException | None = None
    for result in proposal_results:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            if first_error is None:
                first_error = result
            continue
        proposals_list.append(result)
    proposals = tuple(proposals_list)
    if first_error is not None:
        raise MoAExecutionError(
            first_error,
            stage="proposal",
            proposals=proposals,
            usage=(usage_totals[0], usage_totals[1]),
            cached_tokens=usage_totals[2],
        ) from first_error
    synthesis_request = await run_prompt_work(
        _build_synthesis_request,
        query,
        proposals,
        setup,
        shared_prefix,
        final_tools,
        final_tool_choice,
        final_tools_in_prompt,
        final_parallel_tool_calls,
    )
    return proposals, usage_totals, synthesis_request


async def run_moa(
    backend: EngineBackend,
    query: str,
    n_samples: int = _DEFAULT_N_SAMPLES,
    synthesizer: EngineBackend | None = None,
    sampling_params: SamplingParams | None = None,
    final_sampling_params: SamplingParams | None = None,
    final_tools: tuple[Mapping[str, object], ...] = (),
    final_tool_choice: str | Mapping[str, object] | None = None,
    final_tools_in_prompt: bool = False,
    shared_prefix: str = "",
    usage_observer: Callable[[GenerationUsage], None] | None = None,
    final_parallel_tool_calls: bool | None = None,
) -> MoAResult:
    proposals, usage_totals, synthesis_request = await _prepare_moa(
        backend,
        query,
        n_samples=n_samples,
        sampling_params=sampling_params,
        final_sampling_params=final_sampling_params,
        final_tools=final_tools,
        final_tool_choice=final_tool_choice,
        final_tools_in_prompt=final_tools_in_prompt,
        final_parallel_tool_calls=final_parallel_tool_calls,
        shared_prefix=shared_prefix,
        usage_observer=usage_observer,
        prepared_setup=_prepared_setup.get(),
    )
    synthesis_backend = synthesizer or backend
    try:
        final_result = await synthesis_backend.generate(synthesis_request)
    except Exception as error:
        raise MoAExecutionError(
            error,
            stage="synthesis",
            proposals=proposals,
            usage=(usage_totals[0], usage_totals[1]),
            cached_tokens=usage_totals[2],
        ) from error
    if final_result.usage is not None:
        usage_totals[0] += final_result.usage.prompt_tokens
        usage_totals[1] += final_result.usage.completion_tokens
        usage_totals[2] += final_result.usage.cached_tokens
        if usage_observer is not None:
            usage_observer(
                GenerationUsage(
                    prompt_tokens=usage_totals[0],
                    completion_tokens=usage_totals[1],
                    cached_tokens=usage_totals[2],
                )
            )
    return MoAResult(
        final_text=final_result.text,
        proposals=proposals,
        completions=final_result.completions,
        usage=(usage_totals[0], usage_totals[1]),
        cached_tokens=usage_totals[2],
    )


def stream_moa(
    backend: EngineBackend,
    query: str,
    n_samples: int = _DEFAULT_N_SAMPLES,
    synthesizer: EngineBackend | None = None,
    sampling_params: SamplingParams | None = None,
    final_sampling_params: SamplingParams | None = None,
    final_tools: tuple[Mapping[str, object], ...] = (),
    final_tool_choice: str | Mapping[str, object] | None = None,
    final_tools_in_prompt: bool = False,
    shared_prefix: str = "",
    usage_observer: Callable[[GenerationUsage], None] | None = None,
    final_parallel_tool_calls: bool | None = None,
) -> AsyncIterator[MoAEvent]:
    """Capture any exact internal setup before returning the public iterator."""

    return _stream_moa(
        backend,
        query,
        n_samples=n_samples,
        synthesizer=synthesizer,
        sampling_params=sampling_params,
        final_sampling_params=final_sampling_params,
        final_tools=final_tools,
        final_tool_choice=final_tool_choice,
        final_tools_in_prompt=final_tools_in_prompt,
        shared_prefix=shared_prefix,
        usage_observer=usage_observer,
        final_parallel_tool_calls=final_parallel_tool_calls,
        prepared_setup=_prepared_setup.get(),
    )


async def _stream_moa(
    backend: EngineBackend,
    query: str,
    *,
    n_samples: int,
    synthesizer: EngineBackend | None,
    sampling_params: SamplingParams | None,
    final_sampling_params: SamplingParams | None,
    final_tools: tuple[Mapping[str, object], ...],
    final_tool_choice: str | Mapping[str, object] | None,
    final_tools_in_prompt: bool,
    shared_prefix: str,
    usage_observer: Callable[[GenerationUsage], None] | None,
    final_parallel_tool_calls: bool | None,
    prepared_setup: _MoASetup | None,
) -> AsyncIterator[MoAEvent]:
    """Generate proposals concurrently and pull synthesis deltas through.

    The synthesis backend iterator is consumed in the caller's task, avoiding
    an intermediate asyncio queue and making cancellation close the real
    backend stream directly.
    """

    proposals, usage_totals, synthesis_request = await _prepare_moa(
        backend,
        query,
        n_samples=n_samples,
        sampling_params=sampling_params,
        final_sampling_params=final_sampling_params,
        final_tools=final_tools,
        final_tool_choice=final_tool_choice,
        final_tools_in_prompt=final_tools_in_prompt,
        final_parallel_tool_calls=final_parallel_tool_calls,
        shared_prefix=shared_prefix,
        usage_observer=usage_observer,
        prepared_setup=prepared_setup,
    )
    synthesis_backend = synthesizer or backend
    emitted = 0
    last_result: GenerationResult | None = None
    latest_usage = None
    previous_text = ""
    try:
        async for partial in synthesis_backend.stream(synthesis_request):
            last_result = partial
            if partial.usage is not None:
                latest_usage = partial.usage
                if usage_observer is not None:
                    usage_observer(
                        GenerationUsage(
                            prompt_tokens=usage_totals[0] + latest_usage.prompt_tokens,
                            completion_tokens=usage_totals[1] + latest_usage.completion_tokens,
                            cached_tokens=usage_totals[2] + latest_usage.cached_tokens,
                        )
                    )
            text = partial.text
            if not text.startswith(previous_text):
                raise RuntimeError("MoA synthesis stream must emit cumulative, prefix-stable text")
            previous_text = text
            yield MoAEvent(
                kind="delta",
                text=text[emitted:],
                completions=partial.completions,
            )
            emitted = len(text)
        if last_result is None:
            raise RuntimeError("MoA synthesis stream produced no result")
    except Exception as error:
        if latest_usage is not None:
            usage_totals[0] += latest_usage.prompt_tokens
            usage_totals[1] += latest_usage.completion_tokens
            usage_totals[2] += latest_usage.cached_tokens
        raise MoAExecutionError(
            error,
            stage="synthesis",
            proposals=proposals,
            usage=(usage_totals[0], usage_totals[1]),
            cached_tokens=usage_totals[2],
            final_text=previous_text,
        ) from error
    if latest_usage is not None:
        usage_totals[0] += latest_usage.prompt_tokens
        usage_totals[1] += latest_usage.completion_tokens
        usage_totals[2] += latest_usage.cached_tokens
    yield MoAEvent(
        kind="result",
        result=MoAResult(
            final_text=last_result.text,
            proposals=proposals,
            completions=last_result.completions,
            usage=(usage_totals[0], usage_totals[1]),
            cached_tokens=usage_totals[2],
        ),
    )
