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

from kairyu.engine.core.attention import graph_capture_gap
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.models.llama import DenseDecoder


def _tensor_decode_gap(model: DenseDecoder) -> str | None:
    """Why this model cannot run the tensor decode contract, or None (m17 [P2]).

    ``_graph_decode`` calls ``forward_decode_tensors``, which unconditionally
    calls ``backend.attend_decode``. ``MlaAttention`` has no tensor decode form
    at all, so an MLA model used to construct fine and then die with an
    ``AttributeError`` on the first BATCHED decode, arbitrarily far into a run.

    Presence of ``attend_decode`` is NOT the question, though (review [P1]).
    A backend can own that method and still be uncapturable, because what
    breaks a capture is host synchronization inside it — FlashInfer's
    ``plan()`` copies ``indptr`` to the CPU and cannot run under capture at
    all. Checking the method existed would have let such a backend construct
    and then fail on the FIRST capture with a D2H ``RuntimeError``, which is
    exactly the late failure this gate exists to prevent. So the question is
    the declared ``GraphDecodeBackend`` capability, enforced in one place by
    ``graph_capture_gap``.

    Checked once, at construction, where the operator can still act on it.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return f"{type(model).__name__} exposes no model.layers to check"
    if not callable(getattr(model, "plan_decode_tensors", None)):
        return (
            f"{type(model).__name__} has no plan_decode_tensors, so the step "
            "boundary never reaches its attention backends"
        )
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if not hasattr(attention, "forward_decode_tensors"):
            return (
                f"layer {index}'s attention ({type(attention).__name__}) has no "
                "forward_decode_tensors"
            )
        gap = graph_capture_gap(getattr(attention, "backend", None))
        if gap is not None:
            return f"layer {index}'s attention backend {gap}"
    return None


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
        self._graph_scratch_page: int | None = None
        if graph_backend is not None:
            from kairyu.engine.core.step_executor import GraphStepExecutor

            if graph_max_batch < 1 or graph_max_pages < 1:
                raise ValueError(
                    "graph_max_batch and graph_max_pages must be >= 1 when a "
                    f"graph backend is given (got {graph_max_batch}, {graph_max_pages})"
                )
            gap = _tensor_decode_gap(model)
            if gap is not None:  # [P2]: fail here, not on the first batched decode
                raise ValueError(
                    f"graph decode needs the tensor decode contract but {gap}. "
                    "A backend must declare supports_graph_capture and provide "
                    "a plan_decode step hook (see GraphDecodeBackend) — build "
                    "this runner without graph_backend."
                )
            # [P1]: the padding rows of every replay write KV somewhere. That
            # page has to come OUT of the scheduler's allocator, so take it from
            # the cache the scheduler allocates from — there is no way to pick a
            # safe id without it.
            if cache is None:
                raise ValueError(
                    "a graph backend needs the RadixKVCache the scheduler "
                    "allocates from: the captured graph's padding rows write KV "
                    "to a scratch page that must be reserved out of the "
                    "allocator (m17 A5), and only the cache can reserve it"
                )
            self._graph_scratch_page = cache.reserve_scratch_page()
            self._graph = GraphStepExecutor(
                self._graph_decode,
                graph_backend,
                max_batch=graph_max_batch,
                max_pages=graph_max_pages,
                scratch_page=self._graph_scratch_page,
                device=self._device,
                plan_fn=self._plan_graph_decode,
            )

    def _plan_graph_decode(self, batch) -> None:
        """Step-boundary host phase, run OUTSIDE the captured region ([P1]).

        The executor calls this before it captures and again after every
        copy-in, so the attention backends plan against the very buffers the
        next ``replay()`` will read. Without it a planning backend such as
        FlashInfer has no live plan at capture time, and every later replay
        would attend over the pages that happened to be in the static buffers
        when the graph was recorded.
        """
        self._model.plan_decode_tensors(self._pool, batch.page_tables, batch.seq_lens)

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

        The width padding uses the SAME reserved scratch page as the row padding:
        a short row's tail entries are only ever read (and then masked off by
        seq_lens), but pointing them at an allocatable page leaves the class of
        bug [P1] found one indexing mistake away.
        """
        from kairyu.engine.core.step_executor import build_decode_batch

        assert self._graph_scratch_page is not None  # set with self._graph
        batch = build_decode_batch(
            token_ids=tokens,
            positions=positions,
            page_lists=page_tables,
            seq_lens=seq_lens,
            max_pages=max(len(table) for table in page_tables),
            scratch_page=self._graph_scratch_page,
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
