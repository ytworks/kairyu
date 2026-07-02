"""Mixture-of-Agents: parallel diverse proposals + one synthesis pass.

Each proposer gets a distinct perspective header and seed; the synthesizer
sees all numbered proposals. Prompts share ``shared_prefix`` for KV affinity
(design doc D5).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from kairyu.engine.backend import CacheHint, EngineBackend, GenerationRequest
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


async def run_moa(
    backend: EngineBackend,
    query: str,
    n_samples: int = _DEFAULT_N_SAMPLES,
    synthesizer: EngineBackend | None = None,
    sampling_params: SamplingParams | None = None,
    shared_prefix: str = "",
) -> MoAResult:
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    params = sampling_params or SamplingParams(temperature=_PROPOSAL_TEMPERATURE, max_tokens=1024)
    session = uuid.uuid4().hex[:12]
    hint = CacheHint(session_id=session)

    async def propose(index: int) -> str:
        request = GenerationRequest(
            request_id=f"{session}-propose-{index}",
            prompt=f"{shared_prefix}[proposer {index}] Answer the question: {query}",
            sampling_params=params.clone(seed=index),
            cache_hint=hint,
        )
        return (await backend.generate(request)).text

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
    synthesis_backend = synthesizer or backend
    final = (await synthesis_backend.generate(synthesis_request)).text
    return MoAResult(final_text=final, proposals=proposals)
