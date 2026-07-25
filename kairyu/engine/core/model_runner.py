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
        # In-flight tokens: under OverlapEngineCore the snapshot for step N+1 is
        # taken BEFORE step N's token is committed, so `state.outputs` is short
        # and reading `outputs[position - 1]` raised IndexError. The runner
        # already produced those tokens; it just never kept them.
        #
        # This is the HOST-SIDE half of m2 §2.2. That section specifies patching
        # the placeholder slot device-to-device from the sampled tensor, with no
        # host sync in the hot path; this keeps Python ints and rebuilds the
        # input tensor each step. It makes overlap CORRECT on a real runner —
        # which it was not — but the zero-host-sync property is still open.
        self._future_tokens: dict[str, dict[int, int]] = {}
        self._device_tokens: dict[str, dict[int, object]] = {}

    def release(self, request_id: str) -> None:
        """Drop per-request sampler state (seeds + grammar enforcer) on finish (E2)."""
        if self._sampler is not None:
            self._sampler.release(request_id)
        self._future_tokens.pop(request_id, None)
        self._device_tokens.pop(request_id, None)

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
        if token.device_token is not None:
            # m2 §2.2: the next step's input slot is patched from THIS tensor,
            # so the token value never round-trips through the host
            self._device_tokens.setdefault(request_id, {})[position] = token.device_token

    def _forget_committed(self, request_id: str, committed: int) -> None:
        """Drop in-flight tokens the scheduler has since committed."""
        pending = self._future_tokens.get(request_id)
        if not pending:
            return
        for position in [key for key in pending if key < committed]:
            del pending[position]
        device_pending = self._device_tokens.get(request_id)
        if device_pending:
            for position in [key for key in device_pending if key < committed]:
                del device_pending[position]

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

    def _decode_token_tensor(
        self,
        chunks: list[ScheduledChunk],
        states: Mapping[str, object],
        fallback: list[int],
    ) -> torch.Tensor:
        """The decode inputs, built on device where the tokens already are.

        `torch.tensor(list_of_ints, device=...)` is a host-to-device copy of
        values that were produced on the device one step earlier. When every row
        has a device token from the previous step, they are stacked instead —
        m2 §2.2's "patch the last-token slot device-side". A row whose previous
        token was the PROMPT's last (position 0) has no such tensor, so that
        step still comes from the host; it happens once per request, not once
        per token.
        """
        stacked = []
        for chunk in chunks:
            request_id = chunk.request_id
            index = chunk.position - 1
            if index < len(states[request_id].outputs):
                stacked = []  # committed values live on the host; take the copy
                break
            device_token = self._device_tokens.get(request_id, {}).get(index)
            if device_token is None:
                stacked = []
                break
            stacked.append(device_token)
        if len(stacked) == len(chunks) and stacked:
            return torch.stack(stacked).to(torch.long)
        return torch.tensor(fallback, dtype=torch.long, device=self._device)

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
            self._decode_token_tensor(chunks, states, tokens),
            torch.tensor(positions, device=self._device),
            self._pool, page_tables, seq_lens, write_from,
        )
        logits = self._model.logits(hidden)  # [B, vocab]
        for i, chunk in enumerate(chunks):
            state = states[chunk.request_id]
            token = self._sample(state, logits[i], position=chunk.position)
            self._remember(chunk.request_id, chunk.position, token)
            sampled[chunk.request_id] = (token,)
