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
        # Input tensors (token ids, positions) must be built on the model's device
        # so the GPU forward never mixes CPU inputs with on-device weights/KV.
        self._device = next(model.parameters()).device
        # m2 §2.2 future tokens: under OverlapEngineCore the snapshot for step
        # N+1 is taken BEFORE step N's token is committed, so `state.outputs` is
        # one short and reading `outputs[position - 1]` raised IndexError. The
        # runner already produced that token; it just never kept it. Keyed by
        # (request_id, position) and dropped when the request finishes.
        self._future_tokens: dict[str, dict[int, int]] = {}

    def release(self, request_id: str) -> None:
        """Drop per-request sampler state (seeds + grammar enforcer) on finish (E2)."""
        if self._sampler is not None:
            self._sampler.release(request_id)
        self._future_tokens.pop(request_id, None)

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
        decodes = [chunk for chunk in scheduled if not chunk.is_prefill]
        for chunk in scheduled:
            if chunk.is_prefill:
                self._execute_prefill(chunk, states[chunk.request_id], sampled)
        # C4: single-token decodes for all sequences run as ONE batched forward
        # (byte-identical to per-sequence decode; see test_batched_decode). Below
        # two, the per-sequence path is not worth the batch bookkeeping.
        if len(decodes) >= 2:
            self._execute_decode_batch(decodes, states, sampled)
        else:
            for chunk in decodes:
                self._execute_decode(chunk, states[chunk.request_id], sampled)
        return sampled

    def _execute_prefill(self, chunk: ScheduledChunk, state, sampled: dict) -> None:
        prompt = state.request.prompt_token_ids
        page_table = list(state.allocation.pages) + list(state.decode_pages)
        cached = state.allocation.num_cached_tokens if state.allocation else 0
        end = state.computed_prompt
        start = end - chunk.num_tokens
        hidden = self._model.forward_tokens(
            torch.tensor(prompt[start:end], dtype=torch.long, device=self._device),
            torch.arange(start, end, device=self._device),
            self._pool, page_table, seq_len=end, write_from=cached,
        )
        if state.prefill_done and end == len(prompt):
            logits = self._model.logits(hidden[-1])
            token = self._sample(state, logits, position=0)
            self._remember(chunk.request_id, 0, token)
            sampled[chunk.request_id] = (token,)

    def _remember(self, request_id: str, position: int, token: SampledToken) -> None:
        """Keep this step's token so the NEXT step can read it before commit.

        Only the newest position is retained. A decode reads exactly
        ``position - 1``, so keeping the whole history would be an unbounded
        per-request dict — `EngineLoop` calls `release()` on finish but a bare
        `EngineCore` does not, and the buffer grew for the length of every
        request.
        """
        self._future_tokens[request_id] = {position: token.token_id}

    def _previous_token(self, state, position: int) -> int:
        """The token at ``position - 1``, committed or still in flight.

        Committed outputs win: after `update()` they are the authority, and a
        speculative rollback replaces in-flight values that the future buffer
        would otherwise still hold.
        """
        index = position - 1
        outputs = state.outputs
        if index < len(outputs):
            return outputs[index]
        pending = self._future_tokens.get(state.request.request_id, {})
        if index in pending:
            return pending[index]
        raise RuntimeError(
            f"no token for {state.request.request_id} at position {index}: "
            f"{len(outputs)} committed, in-flight {sorted(pending)}"
        )

    def _decode_inputs(self, chunk: ScheduledChunk, state):
        prompt = state.request.prompt_token_ids
        position = chunk.position
        input_token = self._previous_token(state, position) if position > 0 else prompt[-1]
        absolute = len(prompt) + position - 1
        cached = state.allocation.num_cached_tokens if state.allocation else 0
        page_table = list(state.allocation.pages) + list(state.decode_pages)
        return input_token, absolute, page_table, cached

    def _execute_decode(self, chunk: ScheduledChunk, state, sampled: dict) -> None:
        input_token, absolute, page_table, cached = self._decode_inputs(chunk, state)
        hidden = self._model.forward_tokens(
            torch.tensor([input_token], dtype=torch.long, device=self._device),
            torch.tensor([absolute], device=self._device),
            self._pool, page_table, seq_len=absolute + 1, write_from=cached,
        )
        logits = self._model.logits(hidden[-1])
        token = self._sample(state, logits, position=chunk.position)
        self._remember(chunk.request_id, chunk.position, token)
        sampled[chunk.request_id] = (token,)

    def _execute_decode_batch(
        self, chunks: list[ScheduledChunk], states: Mapping[str, object], sampled: dict
    ) -> None:
        tokens, positions, page_tables, seq_lens, write_from = [], [], [], [], []
        for chunk in chunks:
            input_token, absolute, page_table, cached = self._decode_inputs(
                chunk, states[chunk.request_id]
            )
            tokens.append(input_token)
            positions.append(absolute)
            page_tables.append(page_table)
            seq_lens.append(absolute + 1)
            write_from.append(cached)
        hidden = self._model.forward_decode_batch(
            torch.tensor(tokens, dtype=torch.long, device=self._device),
            torch.tensor(positions, device=self._device),
            self._pool, page_tables, seq_lens, write_from,
        )
        logits = self._model.logits(hidden)  # [B, vocab]
        for i, chunk in enumerate(chunks):
            state = states[chunk.request_id]
            token = self._sample(state, logits[i], position=chunk.position)
            self._remember(chunk.request_id, chunk.position, token)
            sampled[chunk.request_id] = (token,)
