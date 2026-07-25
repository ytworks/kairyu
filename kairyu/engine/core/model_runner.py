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
  Python int and reaches the device by one batched H2D copy per step. The
  batched-decode attention loop also reads ``int(positions[i])`` per row.
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
            # the SAME completion the decode input came from: presence,
            # frequency and repetition penalties are computed over this history,
            # so an overlap snapshot missing the in-flight token would penalise a
            # different set of tokens than the eager path and pick differently
            outputs=self._effective_outputs(state, position),
            eos_token_id=state.request.eos_token_id,
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
        token_slots, position_slots = self._decode_input_slots(tokens, positions)
        hidden = self._model.forward_decode_batch(
            token_slots, position_slots,
            self._pool, page_tables, seq_lens, write_from,
        )
        logits = self._model.logits(hidden)  # [B, vocab]
        for i, chunk in enumerate(chunks):
            state = states[chunk.request_id]
            token = self._sample(state, logits[i], position=chunk.position)
            self._remember(chunk.request_id, chunk.position, token)
            sampled[chunk.request_id] = (token,)
