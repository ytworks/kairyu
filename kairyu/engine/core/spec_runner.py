"""SpeculativeRunner: n-gram draft + sequential scoring + greedy verify (m8 D4).

A ``ModelRunner`` wrapper (composition — the scheduler is untouched beyond its
D3 reservation contract). For a speculative decode chunk it proposes a draft
from the committed context, scores each draft position through the wrapped
runner using an immutable overlay state whose ``outputs`` are the committed
tokens plus the draft prefix (the wrapped runner reads ``outputs[p-1]`` — draft
tokens are not in scheduler state), verifies with the existing
``verify_greedy``, and returns the accepted prefix + bonus token.

Per-request gating (review amendment): speculation is bypassed unless the
request is pure greedy (no penalties, no grammar) — penalties change the
argmax so equivalence would not hold, and grammar would need per-position
matcher rollback (M17). Bypassed chunks return a single token; the D3
shortfall accounting absorbs the difference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from kairyu.engine.core.draft import DraftSource, NGramDraftSource
from kairyu.engine.core.engine_core import ModelRunner, StepOutput
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.spec_decode import verify_greedy


class _OverlayState:
    """Immutable view of a request state with ``outputs`` overridden."""

    __slots__ = ("_base", "outputs")

    def __init__(self, base: object, outputs: tuple[int, ...]) -> None:
        self._base = base
        self.outputs = list(outputs)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_base"), name)


class SpeculativeRunner:
    """Wraps a ModelRunner; speculative chunks (num_tokens > 1) get draft+verify."""

    def __init__(self, runner: ModelRunner, draft_source: DraftSource | None = None) -> None:
        # default n-gram keeps m8 behavior byte-identical (m17 D3)
        self._draft_source = draft_source or NGramDraftSource()
        self._runner = runner
        self.draft_proposed = 0
        self.draft_accepted = 0

    @property
    def mean_accepted(self) -> float:
        """Accepted draft tokens per proposed draft token (G4 M-A4 lineage)."""
        if self.draft_proposed == 0:
            return 0.0
        return self.draft_accepted / self.draft_proposed

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> StepOutput:
        plain: list[ScheduledChunk] = []
        speculative: list[ScheduledChunk] = []
        for chunk in scheduled:
            if chunk.is_prefill or chunk.num_tokens == 1:
                plain.append(chunk)
            elif not states[chunk.request_id].request.sampling.is_greedy_pure:
                # bypass: score exactly one token; D3 shortfall releases the rest
                plain.append(replace(chunk, num_tokens=1))
            else:
                speculative.append(chunk)
        sampled: StepOutput = {}
        if plain:
            sampled.update(self._runner.execute(tuple(plain), states))
        for chunk in speculative:
            sampled[chunk.request_id] = self._speculate(chunk, states)
        return sampled

    def _score_position(
        self,
        chunk: ScheduledChunk,
        state: object,
        outputs: tuple[int, ...],
        position: int,
    ) -> SampledToken:
        overlay = _OverlayState(state, outputs)
        sub_chunk = ScheduledChunk(
            request_id=chunk.request_id, num_tokens=1, is_prefill=False, position=position
        )
        out = self._runner.execute((sub_chunk,), {chunk.request_id: overlay})
        return out[chunk.request_id][0]

    def _speculate(
        self, chunk: ScheduledChunk, states: Mapping[str, object]
    ) -> tuple[SampledToken, ...]:
        state = states[chunk.request_id]
        committed = tuple(state.outputs)
        context = state.request.prompt_token_ids + committed
        draft = self._draft_source.propose(context, max_draft=chunk.num_tokens - 1)
        draft = draft[: chunk.num_tokens - 1]
        # target_tokens[i] is the model's own next token given the DRAFT prefix
        # of length i (verify_greedy's contract; walkthrough in m8 §6)
        target: list[SampledToken] = []
        prefix = committed
        for index in range(len(draft) + 1):
            target.append(self._score_position(chunk, state, prefix, chunk.position + index))
            if index < len(draft):
                prefix = (*prefix, draft[index])
        result = verify_greedy(draft, tuple(token.token_id for token in target))
        self.draft_proposed += len(draft)
        self.draft_accepted += result.accepted
        return tuple(target[: result.accepted + 1])
