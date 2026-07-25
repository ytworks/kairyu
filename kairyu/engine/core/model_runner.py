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
        graph_backend: object | None = None,
        graph_max_batch: int = 0,
        graph_max_pages: int = 0,
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
        # m17 D1 / runbook §6.3: decode capture. OFF unless a backend is passed —
        # the seam has existed since m17 with no caller, and enabling it by
        # default would change what every deployment executes.
        self._graph = None
        if graph_backend is not None:
            from kairyu.engine.core.step_executor import GraphStepExecutor

            if graph_max_batch < 1 or graph_max_pages < 1:
                raise ValueError(
                    "graph_max_batch and graph_max_pages must be >= 1 when a "
                    f"graph backend is given (got {graph_max_batch}, {graph_max_pages})"
                )
            self._graph = GraphStepExecutor(
                self._graph_decode,
                graph_backend,
                max_batch=graph_max_batch,
                max_pages=graph_max_pages,
                device=self._device,
            )

    def _graph_decode(self, batch) -> torch.Tensor:
        """The captured region: embed -> layers -> norm -> logits, tensors only."""
        hidden = self._model.forward_decode_tensors(
            batch.token_ids, batch.positions, self._pool,
            batch.page_tables, batch.seq_lens,
        )
        return self._model.logits(hidden)

    def invalidate_graphs(self) -> None:
        """Weight swap or pool resize: every capture is stale (m17 D2)."""
        if self._graph is not None:
            self._graph.invalidate()

    def release(self, request_id: str) -> None:
        """Drop per-request sampler state (seeds + grammar enforcer) on finish (E2)."""
        if self._sampler is not None:
            self._sampler.release(request_id)

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
            sampled[chunk.request_id] = (self._sample(state, logits, position=0),)

    def _decode_inputs(self, chunk: ScheduledChunk, state):
        prompt = state.request.prompt_token_ids
        position = chunk.position
        input_token = state.outputs[position - 1] if position > 0 else prompt[-1]
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
        sampled[chunk.request_id] = (self._sample(state, logits, position=chunk.position),)

    def _graph_logits(
        self,
        tokens: list[int],
        positions: list[int],
        page_tables: list[list[int]],
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Route this decode through capture/replay (m17 D1).

        `build_decode_batch` pads the ragged page lists into the [B, max_pages]
        tensor the captured kernels index; `GraphStepExecutor` falls back to
        eager for an oversize batch or a page table wider than the static buffer, so
        this cannot fail on shape.
        """
        from kairyu.engine.core.step_executor import build_decode_batch

        batch = build_decode_batch(
            token_ids=tokens,
            positions=positions,
            page_lists=page_tables,
            seq_lens=seq_lens,
            max_pages=max(len(table) for table in page_tables),
            device=self._device,
        )
        return self._graph.execute_decode(batch)

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
        if self._graph is not None:
            logits = self._graph_logits(tokens, positions, page_tables, seq_lens)
        else:
            hidden = self._model.forward_decode_batch(
                torch.tensor(tokens, dtype=torch.long, device=self._device),
                torch.tensor(positions, device=self._device),
                self._pool, page_tables, seq_lens, write_from,
            )
            logits = self._model.logits(hidden)  # [B, vocab]
        for i, chunk in enumerate(chunks):
            state = states[chunk.request_id]
            sampled[chunk.request_id] = (
                self._sample(state, logits[i], position=chunk.position),
            )
