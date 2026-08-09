"""Single-device full-sequence runner for hybrid/recurrent decoder families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk


class RecomputeModelRunner:
    """Correctness-first ModelRunner with no paged/recurrent cache reuse."""

    supports_batched_verification = False
    sampling_owner = True

    def __init__(self, model, sampler: Sampler | None = None) -> None:
        self._model = model
        self._sampler = sampler
        self._device = next(model.parameters()).device
        self._future_tokens: dict[str, dict[int, int]] = {}

    def release(self, request_id: str) -> None:
        if self._sampler is not None:
            self._sampler.release(request_id)
        self._future_tokens.pop(request_id, None)

    @staticmethod
    def _sampling_outputs(state: object, position: int) -> Sequence[int]:
        if getattr(state, "outputs_override", False):
            return tuple(state.outputs[:position])
        return state.outputs

    def _pending_outputs(self, state: object, position: int) -> tuple[int, ...]:
        committed = len(state.outputs)
        if position <= committed:
            return ()
        pending = self._future_tokens.get(state.request.request_id, {})
        values: list[int] = []
        for index in range(committed, position):
            if index not in pending:
                raise RuntimeError(
                    f"no token for {state.request.request_id} at position "
                    f"{index} while recomputing position {position}"
                )
            values.append(pending[index])
        return tuple(values)

    def _history(self, state: object, position: int) -> tuple[int, ...]:
        if getattr(state, "outputs_override", False):
            return tuple(state.outputs[:position])
        committed = tuple(state.outputs)
        return committed[:position] + self._pending_outputs(state, position)

    def _sample(self, state: object, logits: torch.Tensor, position: int) -> SampledToken:
        if self._sampler is None:
            return SampledToken(int(torch.argmax(logits).item()))
        return self._sampler.sample(
            state.request.sampling_identity,
            state.request.sampling,
            position,
            logits,
            prompt=state.request.prompt_token_ids,
            outputs=self._sampling_outputs(state, position),
            pending_outputs=self._pending_outputs(state, position),
            history_epoch=getattr(state, "output_epoch", 0),
            eos_token_id=state.request.eos_token_id,
            stop_token_ids=getattr(state.request, "stop_token_ids", ()),
            min_tokens=getattr(state.request, "min_tokens", 0),
        )

    def _score(self, state: object, position: int) -> SampledToken:
        sequence = state.request.prompt_token_ids + self._history(state, position)
        token_ids = torch.tensor(sequence, dtype=torch.long, device=self._device)
        logits = self._model.forward_sequence(token_ids)[-1]
        token = self._sample(state, logits, position)
        self._future_tokens.setdefault(state.request.request_id, {})[position] = token.token_id
        return token

    def execute(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> Mapping[str, tuple[SampledToken, ...]]:
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if chunk.is_prefill:
                if state.prefill_done and state.computed_prompt == len(
                    state.request.prompt_token_ids
                ):
                    sampled[chunk.request_id] = (self._score(state, 0),)
                continue
            if chunk.num_tokens != 1:
                raise ValueError(
                    "hybrid reference execution does not support speculative verification"
                )
            sampled[chunk.request_id] = (self._score(state, chunk.position),)
        return sampled

    def execute_passive(self, scheduled, states):
        raise RuntimeError("hybrid reference execution is single-device only")
