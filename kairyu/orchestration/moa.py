"""Mixture-of-Agents: parallel diverse proposals + one synthesis pass.

Each proposer gets a distinct perspective header and seed; the synthesizer
sees all numbered proposals. Prompts share ``shared_prefix`` for KV affinity
(design doc D5).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
)
from kairyu.sampling_params import SamplingParams

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
    usage: tuple[int, int] = (0, 0)  # (prompt_tokens, completion_tokens) summed
    cached_tokens: int = 0


@dataclass(frozen=True)
class MoAEvent:
    """A final synthesis delta or the complete MoA result."""

    kind: str  # "delta" | "result"
    text: str = ""
    result: MoAResult | None = None


async def _prepare_moa(
    backend: EngineBackend,
    query: str,
    *,
    n_samples: int,
    sampling_params: SamplingParams | None,
    shared_prefix: str,
) -> tuple[tuple[str, ...], list[int], GenerationRequest]:
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    params = sampling_params or SamplingParams(temperature=_PROPOSAL_TEMPERATURE, max_tokens=1024)
    session = uuid.uuid4().hex[:12]
    hint = CacheHint(session_id=session)

    usage_totals = [0, 0, 0]

    def _account(result) -> str:
        if result.usage is not None:
            usage_totals[0] += result.usage.prompt_tokens
            usage_totals[1] += result.usage.completion_tokens
            usage_totals[2] += result.usage.cached_tokens
        return result.text

    async def propose(index: int) -> str:
        request = GenerationRequest(
            request_id=f"{session}-propose-{index}",
            prompt=f"{shared_prefix}[proposer {index}] Answer the question: {query}",
            sampling_params=params.clone(seed=index),
            cache_hint=hint,
        )
        return _account(await backend.generate(request))

    proposals = tuple(await asyncio.gather(*(propose(i) for i in range(n_samples))))
    numbered = "\n\n".join(
        f"Candidate {i + 1}:\n{proposal}" for i, proposal in enumerate(proposals)
    )
    synthesis_request = GenerationRequest(
        request_id=f"{session}-synthesize",
        prompt=shared_prefix + _SYNTHESIS_TEMPLATE.format(query=query, proposals=numbered),
        sampling_params=params.clone(temperature=0.3, seed=None),
        cache_hint=hint,
    )
    return proposals, usage_totals, synthesis_request


async def run_moa(
    backend: EngineBackend,
    query: str,
    n_samples: int = _DEFAULT_N_SAMPLES,
    synthesizer: EngineBackend | None = None,
    sampling_params: SamplingParams | None = None,
    shared_prefix: str = "",
) -> MoAResult:
    proposals, usage_totals, synthesis_request = await _prepare_moa(
        backend,
        query,
        n_samples=n_samples,
        sampling_params=sampling_params,
        shared_prefix=shared_prefix,
    )
    synthesis_backend = synthesizer or backend
    final_result = await synthesis_backend.generate(synthesis_request)
    if final_result.usage is not None:
        usage_totals[0] += final_result.usage.prompt_tokens
        usage_totals[1] += final_result.usage.completion_tokens
        usage_totals[2] += final_result.usage.cached_tokens
    return MoAResult(
        final_text=final_result.text,
        proposals=proposals,
        usage=(usage_totals[0], usage_totals[1]),
        cached_tokens=usage_totals[2],
    )


async def stream_moa(
    backend: EngineBackend,
    query: str,
    n_samples: int = _DEFAULT_N_SAMPLES,
    synthesizer: EngineBackend | None = None,
    sampling_params: SamplingParams | None = None,
    shared_prefix: str = "",
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
        shared_prefix=shared_prefix,
    )
    synthesis_backend = synthesizer or backend
    emitted = 0
    last_result: GenerationResult | None = None
    latest_usage = None
    previous_text = ""
    async for partial in synthesis_backend.stream(synthesis_request):
        last_result = partial
        if partial.usage is not None:
            latest_usage = partial.usage
        text = partial.text
        if not text.startswith(previous_text):
            raise RuntimeError(
                "MoA synthesis stream must emit cumulative, prefix-stable text"
            )
        previous_text = text
        if len(text) > emitted:
            yield MoAEvent(kind="delta", text=text[emitted:])
            emitted = len(text)
    if last_result is None:
        raise RuntimeError("MoA synthesis stream produced no result")
    if latest_usage is not None:
        usage_totals[0] += latest_usage.prompt_tokens
        usage_totals[1] += latest_usage.completion_tokens
        usage_totals[2] += latest_usage.cached_tokens
    yield MoAEvent(
        kind="result",
        result=MoAResult(
            final_text=last_result.text,
            proposals=proposals,
            usage=(usage_totals[0], usage_totals[1]),
            cached_tokens=usage_totals[2],
        ),
    )
