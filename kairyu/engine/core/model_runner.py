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

m2 §2.2 status (the "future token" technique), so the scope is stated where the
code is rather than only in the design doc:

- DONE — the decode input slots are persistent device tensors written in place
  (`_decode_input_slots`), shared by the batched and the single-request path, so
  no decode step allocates a device tensor and neither path is a fallback.
- DONE (#136) — the runner keeps its uncommitted tokens, so overlap can resolve
  a decode input one step before the scheduler commits it.
- OPEN — filling those slots DEVICE-to-device, and §2.2's invariant that the
  step loop never blocks on ``.item()``/``.cpu()``. Both require the sampling
  DECISION to move onto the device; today it is host-side by design (m8 D2 pins
  reproducibility to the CPU RNG stream), so the sampled id exists only as a
  Python int and reaches the device by one batched H2D copy per step.
- DONE (#207) — supported eager and captured batched-decode attention share
  the tensor metadata path; KV-write masking and ragged page tables never read
  a per-row device scalar into Python.
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
        graph_scratch_page: int | None = None,
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
        # In-flight tokens: under OverlapEngineCore the snapshot for step N+1 is
        # taken BEFORE step N's token is committed, so `state.outputs` is short
        # and reading `outputs[position - 1]` raised IndexError. The runner
        # already produced those tokens; it just never kept them.
        #
        # This is the HOST-SIDE half of m2 §2.2 (see the module docstring for what
        # of that section is implemented and what cannot be while the sampling
        # decision is host-side).
        self._future_tokens: dict[str, dict[int, int]] = {}
        # The decode input slots themselves: allocated once on the device and
        # written IN PLACE every step (`_decode_input_slots`).
        self._decode_slots: torch.Tensor | None = None
        self._decode_positions: torch.Tensor | None = None
        self._slot_staging: torch.Tensor | None = None
        self._slot_copy_done: torch.cuda.Event | None = None
        # The same tensor-only decode contract serves both graph capture and
        # eager batching. Unsupported model/attention combinations retain the
        # list-based compatibility path.
        self._tensor_decode_supported = _tensor_decode_gap(model) is None

        # m17 D1 / runbook §6.3: decode capture. OFF unless a backend is passed —
        # the seam has existed since m17 with no caller, and enabling it by
        # default would change what every deployment executes.
        self._graph = None
        self._graph_scratch_page: int | None = None
        if graph_backend is None and graph_scratch_page is not None:
            raise ValueError("graph_scratch_page requires a graph_backend")
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
            if graph_scratch_page is None:
                if cache is None:
                    raise ValueError(
                        "a graph backend needs either the scheduler's "
                        "RadixKVCache or an already-reserved scratch page via "
                        "graph_scratch_page: "
                        "captured padding rows write KV to a page the allocator "
                        "must never return (m17 A5)"
                    )
                graph_scratch_page = cache.reserve_scratch_page()
            elif not 0 <= graph_scratch_page < pool.num_pages:
                raise ValueError(
                    f"graph_scratch_page={graph_scratch_page} is outside the "
                    f"KV pool's [0, {pool.num_pages}) range"
                )
            elif cache is not None:
                reserved = cache.reserve_scratch_page()
                if reserved != graph_scratch_page:
                    raise ValueError(
                        f"graph_scratch_page={graph_scratch_page} disagrees with "
                        f"the scheduler cache's reserved page {reserved}"
                    )
            self._graph_scratch_page = graph_scratch_page
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
            batch.page_tables, batch.seq_lens, batch.write_from,
        )
        return self._model.logits(hidden)

    def invalidate_graphs(self) -> None:
        """Weight swap or pool resize: every capture is stale (m17 D2)."""
        if self._graph is not None:
            self._graph.invalidate()

    @property
    def sampler(self) -> Sampler | None:
        """The per-request sampling state a P-D handoff has to carry across."""
        return self._sampler

    def release(self, request_id: str) -> None:
        """Drop per-request sampler state (seeds + grammar enforcer) on finish (E2)."""
        if self._sampler is not None:
            self._sampler.release(request_id)
        self._future_tokens.pop(request_id, None)

    def _sample(self, state: object, logits: torch.Tensor, position: int) -> SampledToken:
        if self._sampler is None:
            return SampledToken(int(torch.argmax(logits).item()))
        return self._sampler.sample(
            state.request.sampling_identity,
            state.request.sampling,
            position,
            logits,
            prompt=state.request.prompt_token_ids,
            # the SAME completion the decode input came from: presence,
            # frequency and repetition penalties are computed over this history,
            # so an overlap snapshot missing the in-flight token would penalise a
            # different set of tokens than the eager path and pick differently
            outputs=self._effective_outputs(state, position),
            eos_token_id=state.request.eos_token_id,
        )

    def _sample_rows(
        self,
        chunks: list[ScheduledChunk],
        states: Mapping[str, object],
        logits: torch.Tensor,
    ) -> tuple[SampledToken, ...]:
        """Select a decode batch without copying every vocab row to the host.

        The common serving request is pure greedy with no logprob report. All
        rows can then argmax on-device and cross to the host as one ``[B]``
        vector, rather than B transfers of ``[vocab]`` float32 rows. Non-greedy,
        penalized, grammar, and logprob requests retain the reviewed CPU sampler.
        """
        direct = self._sampler is None
        if self._sampler is not None:
            direct = all(
                self._sampler.can_argmax_logits(
                    states[chunk.request_id].request.sampling_identity,
                    states[chunk.request_id].request.sampling,
                    states[chunk.request_id].request.eos_token_id,
                )
                for chunk in chunks
            )
        if direct:
            token_ids = torch.argmax(logits, dim=-1).to(device="cpu").tolist()
            return tuple(SampledToken(int(token_id)) for token_id in token_ids)
        return tuple(
            self._sample(
                states[chunk.request_id],
                logits[index],
                position=chunk.position,
            )
            for index, chunk in enumerate(chunks)
        )

    def _effective_outputs(self, state: object, position: int) -> tuple[int, ...]:
        """Committed outputs, extended with any in-flight token before ``position``.

        Under overlap the snapshot for step N+1 is taken before step N commits,
        so `state.outputs` is short by exactly the tokens this runner has already
        sampled. Sampling reads that history for the penalties, so it needs the
        same view the decode input does.
        """
        outputs = tuple(state.outputs)
        if position <= len(outputs):
            return outputs
        pending = self._future_tokens.get(state.request.request_id, {})
        extended = list(outputs)
        for index in range(len(outputs), position):
            if index not in pending:
                # a gap means the history is genuinely unknown; penalties over a
                # silently-short history are worse than a loud failure
                raise RuntimeError(
                    f"no token for {state.request.request_id} at position {index} "
                    f"while sampling position {position}"
                )
            extended.append(pending[index])
        return tuple(extended)

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        decodes = [chunk for chunk in scheduled if not chunk.is_prefill]
        for chunk in scheduled:
            if chunk.is_prefill:
                self._execute_prefill(chunk, states[chunk.request_id], sampled)
        # C4: single-token decodes for all sequences run as ONE tensor forward
        # (byte-identical to per-sequence decode; see test_batched_decode).
        # Eager execution keeps the cheaper per-sequence path for B=1, but graph
        # mode must use its tensor/static-buffer path at every supported bucket:
        # single-stream launch overhead is one of CUDA graph's primary targets.
        if decodes and (self._graph is not None or len(decodes) >= 2):
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
        """Keep this step's token so later steps can read it before it commits.

        A decode input needs `position - 1`; the sampler's penalties need every
        uncommitted token before `position`. Both are served from here.

        Trimming is driven by what has been COMMITTED, not by a fixed depth.
        `OverlapEngineCore` accepts any `pipeline_depth >= 1`, so a constant cap
        would silently drop a position a deeper pipeline still needs — and the
        symptom would be a RuntimeError mid-run, not a wrong answer, but only
        because the lookup fails loudly. Anything at or below the committed
        length is redundant: `_effective_outputs` and `_previous_token` both
        prefer committed values there.
        """
        pending = self._future_tokens.setdefault(request_id, {})
        pending[position] = token.token_id

    def _forget_committed(self, request_id: str, committed: int) -> None:
        """Drop in-flight tokens the scheduler has since committed."""
        pending = self._future_tokens.get(request_id)
        if not pending:
            return
        for position in [key for key in pending if key < committed]:
            del pending[position]

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
        # whatever the scheduler has committed is authoritative from here on, so
        # the in-flight copies of it are dead weight
        self._forget_committed(state.request.request_id, len(state.outputs))
        input_token = self._previous_token(state, position) if position > 0 else prompt[-1]
        absolute = len(prompt) + position - 1
        cached = state.allocation.num_cached_tokens if state.allocation else 0
        page_table = list(state.allocation.pages) + list(state.decode_pages)
        return input_token, absolute, page_table, cached

    def _retire_decode_slots(self) -> None:
        """Drain the in-flight staging DMA before its buffers can be dropped.

        A growth replaces `_slot_staging` and the device slots wholesale, and the
        event recorded by the last `_decode_input_slots` is the ONLY handle on the
        transfer that may still be reading them. Overwriting `_slot_copy_done`
        first destroys that handle: what remains is a freshly recorded event that
        says nothing about the old DMA. Freeing pinned host memory is not
        stream-ordered, so the source rows could go back to the allocator while
        the copy engine is still reading them. Wait here, BEFORE anything is
        released. Growth doubles, so this blocking wait is amortized away.
        """
        if self._slot_copy_done is not None:
            self._slot_copy_done.synchronize()
        self._slot_staging = None
        self._slot_copy_done = None

    def _allocate_decode_slots(self, size: int) -> None:
        """(Re)allocate the persistent slots, doubling so growth is amortized."""
        self._retire_decode_slots()
        current = 0 if self._decode_slots is None else self._decode_slots.numel()
        capacity = max(8, size, 2 * current)
        self._decode_slots = torch.zeros(capacity, dtype=torch.long, device=self._device)
        self._decode_positions = torch.zeros(capacity, dtype=torch.long, device=self._device)
        if self._device.type == "cuda":
            # pinned, so the H2D below is a real async DMA rather than a staged
            # pageable copy that blocks the calling thread
            self._slot_staging = torch.zeros(2, capacity, dtype=torch.long).pin_memory()
            self._slot_copy_done = torch.cuda.Event()
            self._slot_copy_done.record()
        else:
            self._slot_staging = None
            self._slot_copy_done = None

    def _decode_input_slots(
        self, tokens: list[int], positions: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's decode inputs INTO the persistent device slots.

        m2 §5 names this: "the GPU runner owns a future-token device buffer
        indexed by (request, position)". The slots are allocated once and patched
        in place, so no decode step allocates a device tensor and BOTH decode
        paths — batched and single-request — read the same memory. Views are
        returned, so a caller that kept last step's view sees this step's values;
        that aliasing is the property, and `test_decode_input_slots.py` pins it.

        What this does NOT do, and what m2 §2.2 asks for, is fill the slots
        device-to-device. The sampler decides on the host by design (m8 D2 pins
        reproducibility to the CPU RNG stream), so the chosen id exists only as a
        Python int and one H2D copy per step is unavoidable — see the note in
        `sampler.py`. Removing it means moving the sampling decision itself onto
        the device. Per-row scalar copies would be the same round trip B times
        over, so the ids are copied as one batched transfer.
        """
        size = len(tokens)
        if self._decode_slots is None or self._decode_slots.numel() < size:
            self._allocate_decode_slots(size)
        assert self._decode_slots is not None and self._decode_positions is not None
        host_tokens = torch.tensor(tokens, dtype=torch.long)
        host_positions = torch.tensor(positions, dtype=torch.long)
        if self._slot_staging is None:
            self._decode_slots[:size].copy_(host_tokens)
            self._decode_positions[:size].copy_(host_positions)
        else:
            assert self._slot_copy_done is not None
            # the previous step's DMA must have drained before its source rows are
            # overwritten; in practice it long since has (the sampler syncs once
            # per step), so this is a completed-event check, not a stall — but it
            # does not DEPEND on the sampler still syncing. The other way the
            # source rows stop being valid is a capacity growth, and that path
            # waits on THIS event in `_retire_decode_slots` before replacing it.
            self._slot_copy_done.synchronize()
            self._slot_staging[0, :size].copy_(host_tokens)
            self._slot_staging[1, :size].copy_(host_positions)
            self._decode_slots[:size].copy_(self._slot_staging[0, :size], non_blocking=True)
            self._decode_positions[:size].copy_(
                self._slot_staging[1, :size], non_blocking=True
            )
            self._slot_copy_done.record()
        return self._decode_slots[:size], self._decode_positions[:size]

    def _execute_decode(self, chunk: ScheduledChunk, state, sampled: dict) -> None:
        input_token, absolute, page_table, cached = self._decode_inputs(chunk, state)
        # the single-request path uses the SAME slots as the batched one: a
        # workload that drops to one request must not fall back to rebuilding a
        # fresh device tensor every step
        token_slot, position_slot = self._decode_input_slots([input_token], [absolute])
        hidden = self._model.forward_tokens(
            token_slot, position_slot,
            self._pool, page_table, seq_len=absolute + 1, write_from=cached,
        )
        logits = self._model.logits(hidden[-1])
        token = self._sample(state, logits, position=chunk.position)
        self._remember(chunk.request_id, chunk.position, token)
        sampled[chunk.request_id] = (token,)

    def _graph_logits(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_tables: list[list[int]],
        seq_lens: list[int],
        write_from: list[int] | None = None,
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
        if write_from is None:
            write_from = [0] * len(tokens)
        batch = build_decode_batch(
            token_ids=tokens,
            positions=positions,
            page_lists=page_tables,
            seq_lens=seq_lens,
            max_pages=max(len(table) for table in page_tables),
            scratch_page=self._graph_scratch_page,
            write_from=write_from,
            device=self._device,
        )
        return self._graph.execute_decode(batch)

    def _eager_tensor_logits(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_tables: list[list[int]],
        seq_lens: list[int],
        write_from: list[int],
    ) -> torch.Tensor:
        """One tensor-metadata eager forward, with no per-row device reads.

        Ragged page-table tails repeat each request's last owned page. They are
        masked by seq_lens and there are no synthetic rows, so eager execution
        needs no graph scratch-page reservation.
        """
        from kairyu.engine.core.step_executor import build_decode_batch

        batch = build_decode_batch(
            token_ids=tokens,
            positions=positions,
            page_lists=page_tables,
            seq_lens=seq_lens,
            max_pages=max(len(table) for table in page_tables),
            scratch_page=None,
            write_from=write_from,
            device=self._device,
        )
        self._model.plan_decode_tensors(
            self._pool, batch.page_tables, batch.seq_lens
        )
        hidden = self._model.forward_decode_tensors(
            batch.token_ids,
            batch.positions,
            self._pool,
            batch.page_tables,
            batch.seq_lens,
            batch.write_from,
        )
        return self._model.logits(hidden)

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
        # Every branch consumes the same persistent token/position slots. The
        # tensor path must not regress #143 by rebuilding those inputs while it
        # tensorizes page-table metadata.
        token_slots, position_slots = self._decode_input_slots(tokens, positions)
        if self._graph is not None:
            logits = self._graph_logits(
                token_slots, position_slots, page_tables, seq_lens, write_from
            )
        elif self._tensor_decode_supported:
            logits = self._eager_tensor_logits(
                token_slots, position_slots, page_tables, seq_lens, write_from
            )
        else:
            hidden = self._model.forward_decode_batch(
                token_slots, position_slots,
                self._pool, page_tables, seq_lens, write_from,
                position_values=positions,
            )
            logits = self._model.logits(hidden)  # [B, vocab]
        tokens = self._sample_rows(chunks, states, logits)
        for chunk, token in zip(chunks, tokens, strict=True):
            self._remember(chunk.request_id, chunk.position, token)
            sampled[chunk.request_id] = (token,)
