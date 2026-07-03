"""PagedModelRunner: the real ModelRunner over DenseDecoder + PagedKVPool (m12 D4).

State-access contract (canonical, m12 review): reads exactly
``request.{prompt_token_ids, request_id, sampling, eos_token_id}``,
``allocation.pages``, ``allocation.num_cached_tokens``, ``decode_pages``,
``computed_prompt``, ``prefill_done``, ``outputs`` (values). The decode input
token comes from the PASSED state (`outputs[p-1]`) at execute time — that is
what keeps SpeculativeRunner's overlay mechanism working unchanged. KV is
written before it is read at every decode position; positions below
``num_cached_tokens`` are never rewritten (shared radix slots).
Requests are processed sequentially within a step (CPU correctness first;
cross-request batching arrives with M13/GPU).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.models.llama import DenseDecoder


class PagedModelRunner:
    def __init__(
        self,
        model: DenseDecoder,
        pool: PagedKVPool,
        sampler: Sampler | None = None,
        cache: object | None = None,
    ) -> None:
        if cache is not None:  # fail-fast sizing agreement (m12 D3)
            if pool.num_pages != cache.num_pages or pool.page_size != cache.page_size:
                raise ValueError(
                    f"pool ({pool.num_pages} pages x {pool.page_size}) disagrees with "
                    f"cache ({cache.num_pages} x {cache.page_size})"
                )
        if pool.num_layers != model.config.num_hidden_layers:
            raise ValueError("pool layer count disagrees with the model config")
        self._model = model
        self._pool = pool
        self._sampler = sampler

    def _sample(self, state: object, logits: torch.Tensor, position: int) -> SampledToken:
        if self._sampler is None:
            return SampledToken(int(torch.argmax(logits).item()))
        return self._sampler.sample(
            state.request.request_id,
            state.request.sampling,
            position,
            logits,
            prompt=state.request.prompt_token_ids,
            outputs=tuple(state.outputs),
            eos_token_id=state.request.eos_token_id,
        )

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            prompt = state.request.prompt_token_ids
            page_table = list(state.allocation.pages) + list(state.decode_pages)
            cached = state.allocation.num_cached_tokens if state.allocation else 0
            if chunk.is_prefill:
                end = state.computed_prompt
                start = end - chunk.num_tokens
                hidden = self._model.forward_tokens(
                    torch.tensor(prompt[start:end], dtype=torch.long),
                    torch.arange(start, end),
                    self._pool,
                    page_table,
                    seq_len=end,
                    write_from=cached,
                )
                if state.prefill_done and end == len(prompt):
                    logits = self._model.logits(hidden[-1])
                    sampled[chunk.request_id] = (self._sample(state, logits, position=0),)
            else:
                position = chunk.position
                input_token = state.outputs[position - 1] if position > 0 else prompt[-1]
                absolute = len(prompt) + position - 1
                hidden = self._model.forward_tokens(
                    torch.tensor([input_token], dtype=torch.long),
                    torch.tensor([absolute]),
                    self._pool,
                    page_table,
                    seq_len=absolute + 1,
                    write_from=cached,
                )
                logits = self._model.logits(hidden[-1])
                sampled[chunk.request_id] = (
                    self._sample(state, logits, position=position),
                )
        return sampled
