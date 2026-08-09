"""Incremental Kairyu runner for Qwen3.5/3.6 and DeepSeek V4 cache families."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from kairyu.engine.core.frontier_cache import (
    CacheDescriptor,
    CacheHandle,
    PrefixStateStore,
    cache_descriptor_for_model,
)
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk


class NativeFrontierModelRunner:
    """Own opaque recurrent/compressed state while Kairyu owns scheduling.

    Unlike ``RecomputeModelRunner``, decode advances only the newly committed
    token.  A prefix hit is admitted only when a complete architecture state
    snapshot exists; a generic RadixKV hit alone never skips recurrent work.
    """

    supports_batched_verification = False
    sampling_owner = True
    execution_mode = "native_frontier_state_cache"

    def __init__(
        self,
        model,
        sampler: Sampler | None = None,
        *,
        prefix_state_capacity_bytes: int = 0,
    ) -> None:
        if not callable(getattr(model, "forward_cached", None)):
            raise TypeError("native frontier model must implement forward_cached")
        self._model = model
        self._sampler = sampler
        self._device = next(model.parameters()).device
        self._handles: dict[str, CacheHandle] = {}
        self._future_tokens: dict[str, dict[int, int]] = {}
        self._prefixes = PrefixStateStore(prefix_state_capacity_bytes)
        self.cache_descriptor: CacheDescriptor = cache_descriptor_for_model(model.config)
        self._cache_hits = 0
        self._cache_misses = 0

    def release(self, request_id: str) -> None:
        if self._sampler is not None:
            self._sampler.release(request_id)
        self._future_tokens.pop(request_id, None)
        self._handles.pop(request_id, None)

    def cache_stats(self) -> dict[str, int]:
        return {
            "active_handles": len(self._handles),
            "prefix_state_bytes": self._prefixes.used_bytes,
            "prefix_hits": self._cache_hits,
            "prefix_misses": self._cache_misses,
        }

    def _pending_outputs(self, state: object, position: int) -> tuple[int, ...]:
        committed = len(state.outputs)
        if position <= committed:
            return ()
        pending = self._future_tokens.get(state.request.request_id, {})
        values: list[int] = []
        for index in range(committed, position):
            if index not in pending:
                raise RuntimeError(
                    f"no token for {state.request.request_id} at position {index} "
                    f"while advancing frontier cache to output {position}"
                )
            values.append(pending[index])
        return tuple(values)

    def _history(self, state: object, position: int) -> tuple[int, ...]:
        return tuple(state.outputs[:position]) + self._pending_outputs(state, position)

    def _forget_committed(self, request_id: str, committed: int) -> None:
        pending = self._future_tokens.get(request_id)
        if pending:
            for position in [key for key in pending if key < committed]:
                del pending[position]

    def _restore_prefix(self, handle: CacheHandle, desired: tuple[int, ...]) -> None:
        restored = self._prefixes.longest_prefix(desired)
        if restored is None:
            handle.replace((), None, None, handle.output_epoch)
            self._cache_misses += 1
            return
        prefix, payload = restored
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise RuntimeError("frontier prefix snapshot payload is malformed")
        cache, last_logits = payload
        handle.replace(prefix, cache, last_logits, handle.output_epoch)
        self._cache_hits += 1

    def _ensure_sequence(self, state: object, desired: tuple[int, ...]) -> CacheHandle:
        request_id = state.request.request_id
        epoch = getattr(state, "output_epoch", 0)
        handle = self._handles.get(request_id)
        if handle is None:
            handle = CacheHandle(request_id=request_id, output_epoch=epoch)
            self._handles[request_id] = handle
            self._restore_prefix(handle, desired)
        if handle.output_epoch != epoch or desired[: len(handle.token_ids)] != handle.token_ids:
            handle.output_epoch = epoch
            self._restore_prefix(handle, desired)
        suffix = desired[len(handle.token_ids) :]
        if suffix:
            ids = torch.tensor(suffix, dtype=torch.long, device=self._device)
            logits, cache = self._model.forward_cached(
                ids,
                past_key_values=handle.state,
                position=len(handle.token_ids),
            )
            handle.replace(desired, cache, logits[-1], epoch)
        if handle.last_logits is None:
            raise RuntimeError("frontier cache reached an empty sequence without logits")
        return handle

    def _sample(self, state: object, logits: torch.Tensor, position: int) -> SampledToken:
        if self._sampler is None:
            return SampledToken(int(torch.argmax(logits).item()))
        return self._sampler.sample(
            state.request.sampling_identity,
            state.request.sampling,
            position,
            logits,
            prompt=state.request.prompt_token_ids,
            outputs=state.outputs[:position],
            pending_outputs=self._pending_outputs(state, position),
            history_epoch=getattr(state, "output_epoch", 0),
            eos_token_id=state.request.eos_token_id,
            stop_token_ids=getattr(state.request, "stop_token_ids", ()),
            min_tokens=getattr(state.request, "min_tokens", 0),
        )

    def _score(self, state: object, position: int) -> SampledToken:
        self._forget_committed(state.request.request_id, len(state.outputs))
        prompt = tuple(state.request.prompt_token_ids)
        desired = prompt + self._history(state, position)
        if len(desired) > self.cache_descriptor.max_context_tokens:
            raise ValueError(
                f"request has {len(desired)} tokens but native model context is "
                f"{self.cache_descriptor.max_context_tokens}; truncation is disabled"
            )
        handle = self._ensure_sequence(state, desired)
        if position == 0 and handle.token_ids == prompt:
            self._prefixes.put(prompt, (handle.state, handle.last_logits))
        token = self._sample(state, handle.last_logits, position)
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
                    "frontier native cache requires validated architecture-specific "
                    "draft verification before speculative_tokens can exceed one"
                )
            sampled[chunk.request_id] = (self._score(state, chunk.position),)
        return sampled

    def execute_passive(self, scheduled, states):
        raise RuntimeError(
            "frontier native state cache requires an architecture-specific EP worker"
        )
