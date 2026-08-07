"""The Sampler: penalties, temperature, min-p/top-k/top-p, grammar (design m8 D2).

Order of operations (reviewed convention, m8 §6):

1. raw logits → capture ``log_softmax`` for logprob reporting (vLLM v1
   ``raw_logprobs`` default; temperature-independent, OpenAI-style).
2. xgrammar mask FIRST (mask-last can leave zero legal tokens after top-k/top-p
   → zero-support argmax fallback; mask-first plus keep-1 filters guarantees
   support).
3. While ``position < min_tokens``, mask EOS and every stop-token ID.  The
   logical scheduled position is authoritative under overlap/speculation.
4. Penalties — repetition over prompt + committed outputs; presence/frequency
   over committed outputs only (vLLM/HF agreement).
5. A native forced continuation selects its token for this logical position;
   otherwise ``temperature == 0`` → argmax on the masked logits; else scale.
6. min_p, then top-k, then top-p (vLLM v1 order; HF differs — recorded).
7. softmax → stateless seeded Gumbel-max sample.

Determinism: base seed is ``sampling.seed`` or sha256(request_id) (never
Python ``hash()`` — process-randomized); the per-position seed is a splitmix64
mix of (base, position). CPU, structured, and CUDA paths use the same stateless
Gumbel noise derived from (seed, position, vocab index), so changing execution
path does not select a different random stream. TP rank runners sample
identically given identical logits and replaying a position is idempotent
without a host-owned RNG offset. Grammar ``accept`` is likewise idempotent per
position — the matcher advances exactly once per committed token.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from kairyu.engine.core.sampling_types import (
    EngineSampling,
    SampledToken,
    mix_seed,
    stable_request_seed,
)
from kairyu.engine.core.structured import XGrammarEnforcer
from kairyu.engine.tokenizer import GrammarVocabulary
from kairyu.kernels.sampling_gpu import stateless_gumbel_argmax_batched

_MAX_BATCHED_SAMPLING_ELEMENTS = 1 << 22


@dataclass(frozen=True)
class DeviceSample:
    """A sampling result that remains on the logits device.

    The runner owns when these tensors are copied to the host.  Keeping that
    boundary out of ``sample_device`` is what lets overlap enqueue step N+1
    before step N's EOS/streaming bookkeeping resolves on the CPU.
    """

    token_id: torch.Tensor  # scalar int64
    logprob: torch.Tensor | None = None  # scalar float32
    top_indices: torch.Tensor | None = None  # [K] int64
    top_logprobs: torch.Tensor | None = None  # [K] float32


@dataclass(frozen=True)
class DeviceSamplingInput:
    """One grammar-free row supplied to the batched device sampler."""

    request_id: str
    sampling: EngineSampling
    position: int
    prompt: tuple[int, ...] = ()
    outputs: Sequence[int] = ()
    pending_outputs: Sequence[torch.Tensor] = ()
    history_epoch: int | None = None
    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    min_tokens: int = 0


def _sampling_metadata_tensor(
    values: Sequence[int | float | bool],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Copy a small host parameter vector without synchronizing a CUDA stream."""

    host = torch.tensor(
        values,
        dtype=dtype,
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        return host.to(device=device, non_blocking=True)
    return host


class _IncrementalPenaltyState:
    """Dense penalty tensors updated from only the changed history suffix.

    ``committed`` is append-only on the normal scheduler path. ``pending``
    contains the bounded overlap-ahead suffix and may be replaced by an
    authoritative committed token after speculative verification. A shorter
    committed history is the exceptional rollback path; only then do we search
    for the common prefix.
    """

    __slots__ = (
        "_cpu_indices_initialized",
        "_cpu_output_indices",
        "_cpu_output_len",
        "_cpu_output_positions",
        "_cpu_output_ref_counts",
        "_cpu_prompt_ids",
        "_cpu_repetition_indices",
        "_cpu_repetition_len",
        "_cpu_repetition_positions",
        "committed",
        "device",
        "history_epoch",
        "output_counts",
        "output_seen",
        "pending",
        "prompt",
        "prompt_counted",
        "prompt_seen",
        "repetition_seen",
        "vocab_size",
    )

    def __init__(
        self,
        prompt: tuple[int, ...],
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.vocab_size = vocab_size
        self.prompt = prompt
        self.prompt_counted = False
        self.committed: list[int] = []
        self.pending: tuple[int | torch.Tensor, ...] = ()
        self.history_epoch: int | None = None
        self._cpu_prompt_ids: set[int] | None = set() if device.type == "cpu" else None
        self._cpu_output_ref_counts: dict[int, int] | None = {} if device.type == "cpu" else None
        self._cpu_output_positions: dict[int, int] | None = {} if device.type == "cpu" else None
        self._cpu_repetition_positions: dict[int, int] | None = {} if device.type == "cpu" else None
        self._cpu_repetition_indices: torch.Tensor | None = (
            torch.empty(8, dtype=torch.long) if device.type == "cpu" else None
        )
        self._cpu_output_indices: torch.Tensor | None = (
            torch.empty(8, dtype=torch.long) if device.type == "cpu" else None
        )
        self._cpu_repetition_len = 0
        self._cpu_output_len = 0
        self._cpu_indices_initialized = False
        self.prompt_seen = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.repetition_seen = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.output_counts = torch.zeros(vocab_size, dtype=torch.float32, device=device)
        self.output_seen = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    def _indices(self, tokens: Sequence[int | torch.Tensor]) -> tuple[torch.Tensor, ...]:
        host = [token for token in tokens if not isinstance(token, torch.Tensor)]
        device = [token for token in tokens if isinstance(token, torch.Tensor)]
        batches: list[torch.Tensor] = []
        if host:
            batches.append(torch.as_tensor(host, dtype=torch.long, device=self.device))
        if device:
            if len(device) == 1:
                batches.append(device[0].to(device=self.device, dtype=torch.long).reshape(1))
            else:
                batches.append(
                    torch.stack(
                        [
                            token.to(device=self.device, dtype=torch.long).reshape(())
                            for token in device
                        ]
                    )
                )
        return tuple(batches)

    @staticmethod
    def _grow_cpu_indices(
        indices: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        if length < indices.numel():
            return indices
        grown = torch.empty(max(8, indices.numel() * 2), dtype=torch.long)
        grown[:length].copy_(indices[:length])
        return grown

    def _cpu_add_output(self, token_id: int) -> None:
        assert self._cpu_output_positions is not None
        assert self._cpu_output_indices is not None
        if token_id in self._cpu_output_positions:
            return
        self._cpu_output_indices = self._grow_cpu_indices(
            self._cpu_output_indices,
            self._cpu_output_len,
        )
        self._cpu_output_indices[self._cpu_output_len] = token_id
        self._cpu_output_positions[token_id] = self._cpu_output_len
        self._cpu_output_len += 1

    def _cpu_remove_output(self, token_id: int) -> None:
        assert self._cpu_output_positions is not None
        assert self._cpu_output_indices is not None
        position = self._cpu_output_positions.pop(token_id)
        last = self._cpu_output_len - 1
        if position != last:
            moved = int(self._cpu_output_indices[last].item())
            self._cpu_output_indices[position] = moved
            self._cpu_output_positions[moved] = position
        self._cpu_output_len = last

    def _cpu_add_repetition(self, token_id: int) -> None:
        assert self._cpu_repetition_positions is not None
        assert self._cpu_repetition_indices is not None
        if token_id in self._cpu_repetition_positions:
            return
        self._cpu_repetition_indices = self._grow_cpu_indices(
            self._cpu_repetition_indices,
            self._cpu_repetition_len,
        )
        self._cpu_repetition_indices[self._cpu_repetition_len] = token_id
        self._cpu_repetition_positions[token_id] = self._cpu_repetition_len
        self._cpu_repetition_len += 1

    def _cpu_remove_repetition(self, token_id: int) -> None:
        assert self._cpu_repetition_positions is not None
        assert self._cpu_repetition_indices is not None
        position = self._cpu_repetition_positions.pop(token_id)
        last = self._cpu_repetition_len - 1
        if position != last:
            moved = int(self._cpu_repetition_indices[last].item())
            self._cpu_repetition_indices[position] = moved
            self._cpu_repetition_positions[moved] = position
        self._cpu_repetition_len = last

    def _change(
        self,
        tokens: Sequence[int | torch.Tensor],
        delta: int,
        *,
        output: bool,
    ) -> None:
        if not tokens:
            return
        cpu_token_ids: list[int] | None = None
        if self._cpu_prompt_ids is not None:
            assert self._cpu_output_ref_counts is not None
            cpu_token_ids = []
            for token in tokens:
                token_id = int(token.item()) if isinstance(token, torch.Tensor) else int(token)
                cpu_token_ids.append(token_id)
                if output:
                    old_count = self._cpu_output_ref_counts.get(token_id, 0)
                    count = old_count + delta
                    if count < 0:
                        raise AssertionError("sampler output count became negative")
                    if count:
                        self._cpu_output_ref_counts[token_id] = count
                    else:
                        self._cpu_output_ref_counts.pop(token_id, None)
                    if old_count == 0 and count:
                        self._cpu_add_output(token_id)
                        self._cpu_add_repetition(token_id)
                    elif old_count and count == 0:
                        self._cpu_remove_output(token_id)
                        if token_id not in self._cpu_prompt_ids:
                            self._cpu_remove_repetition(token_id)
                else:
                    if token_id not in self._cpu_prompt_ids:
                        self._cpu_prompt_ids.add(token_id)
                        self._cpu_add_repetition(token_id)
            if len(cpu_token_ids) <= 8:
                for token_id in cpu_token_ids:
                    count = self._cpu_output_ref_counts.get(token_id, 0)
                    if output:
                        self.output_counts[token_id] = float(count)
                        self.output_seen[token_id] = count > 0
                    self.repetition_seen[token_id] = token_id in self._cpu_prompt_ids or count > 0
                    if not output:
                        self.prompt_seen[token_id] = True
                return
        for indices in self._indices(tokens):
            if output:
                output_delta = torch.full_like(
                    indices, float(delta), dtype=self.output_counts.dtype
                )
                self.output_counts.scatter_add_(0, indices, output_delta)
                self.output_seen[indices] = self.output_counts[indices] > 0
                self.repetition_seen[indices] = (
                    self.prompt_seen[indices] | self.output_seen[indices]
                )
            else:
                if delta != 1:
                    raise AssertionError("prompt membership is only initialized")
                self.prompt_seen[indices] = True
                self.repetition_seen[indices] = True

    @staticmethod
    def _same_pending(
        left: int | torch.Tensor,
        right: int | torch.Tensor,
    ) -> bool:
        # Pending device tokens are owned by PagedModelRunner's position map.
        # Identity comparison avoids a scalar D2H sync merely to reconcile it.
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return left is right
        return left == right

    def _reset_prompt(self, prompt: tuple[int, ...]) -> None:
        self.prompt_seen.zero_()
        self.repetition_seen.zero_()
        self.output_counts.zero_()
        self.output_seen.zero_()
        self.prompt = prompt
        self.prompt_counted = False
        self.committed.clear()
        self.pending = ()
        self.history_epoch = None
        if self._cpu_prompt_ids is not None:
            assert self._cpu_output_ref_counts is not None
            assert self._cpu_output_positions is not None
            assert self._cpu_repetition_positions is not None
            self._cpu_prompt_ids.clear()
            self._cpu_output_ref_counts.clear()
            self._cpu_output_positions.clear()
            self._cpu_repetition_positions.clear()
            self._cpu_output_len = 0
            self._cpu_repetition_len = 0
            self._cpu_indices_initialized = False

    def _cpu_active_indices(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._cpu_prompt_ids is not None
        assert self._cpu_output_ref_counts is not None
        assert self._cpu_output_indices is not None
        assert self._cpu_repetition_indices is not None
        assert self._cpu_output_positions is not None
        assert self._cpu_repetition_positions is not None
        if not self._cpu_indices_initialized:
            output = self._cpu_output_indices[: self._cpu_output_len]
            repetition = self._cpu_repetition_indices[: self._cpu_repetition_len]
            output.copy_(torch.sort(output).values)
            repetition.copy_(torch.sort(repetition).values)
            self._cpu_output_positions.clear()
            self._cpu_output_positions.update(
                {int(token): index for index, token in enumerate(output.tolist())}
            )
            self._cpu_repetition_positions.clear()
            self._cpu_repetition_positions.update(
                {int(token): index for index, token in enumerate(repetition.tolist())}
            )
            self._cpu_indices_initialized = True
        return (
            self._cpu_repetition_indices[: self._cpu_repetition_len],
            self._cpu_output_indices[: self._cpu_output_len],
        )

    def sync(
        self,
        prompt: tuple[int, ...],
        outputs: Sequence[int],
        pending_outputs: Sequence[int | torch.Tensor],
        *,
        include_prompt: bool,
        history_epoch: int | None,
    ) -> None:
        """Reconcile commit, rollback, and overlap-ahead state incrementally."""
        if prompt != self.prompt:
            self._reset_prompt(prompt)
        if include_prompt and not self.prompt_counted:
            self._change(prompt, 1, output=False)
            self.prompt_counted = True

        old_length = len(self.committed)
        new_length = len(outputs)

        epoch_changed = (
            history_epoch is not None
            and self.history_epoch is not None
            and history_epoch != self.history_epoch
        )
        if epoch_changed:
            # A production owner increments the epoch only when it replaces an
            # already-committed prefix. Rebuild that exceptional transition;
            # normal append-only steps never scan or copy the retained prefix.
            self._change(self.committed, -1, output=True)
            self._change(self.pending, -1, output=True)
            self.committed.clear()
            self.committed.extend(outputs)
            self._change(self.committed, 1, output=True)
            self.pending = ()
        else:
            common = old_length
            if history_epoch is None or new_length < old_length:
                # Unversioned public callers retain fully defensive semantics.
                # Production supplies an epoch and pays this prefix scan only
                # for an actual rollback.
                common = 0
                limit = min(old_length, new_length)
                while common < limit and self.committed[common] == outputs[common]:
                    common += 1

            if common < old_length:
                # Exceptional rollback/restart: replace only the divergent
                # suffix and discard the bounded speculative pending window.
                self._change(self.committed[common:], -1, output=True)
                self._change(self.pending, -1, output=True)
                replacement = outputs[common:]
                self._change(replacement, 1, output=True)
                del self.committed[common:]
                self.committed.extend(replacement)
                self.pending = ()
            else:
                appended = outputs[old_length:]
                # A device-pending value that becomes host-committed is
                # replaced arithmetically, never compared via .item(), so
                # rollback corrections are exact without a device-to-host sync.
                consumed = min(len(appended), len(self.pending))
                if consumed:
                    if history_epoch is None:
                        # Defensive public API: the authoritative host value can
                        # correct an unversioned speculative value.
                        self._change(
                            self.pending[:consumed],
                            -1,
                            output=True,
                        )
                        self._change(
                            appended[:consumed],
                            1,
                            output=True,
                        )
                    # Production's versioned pending token is the exact device
                    # scalar whose batched D2H result the scheduler appended.
                    # It was already counted, so committing it only advances
                    # the boundary and performs no tensor/map mutation.
                    self.pending = self.pending[consumed:]
                if consumed < len(appended):
                    self._change(appended[consumed:], 1, output=True)
                self.committed.extend(appended)

        self.history_epoch = history_epoch

        pending = tuple(pending_outputs)
        common_pending = 0
        limit = min(len(self.pending), len(pending))
        while common_pending < limit and self._same_pending(
            self.pending[common_pending],
            pending[common_pending],
        ):
            common_pending += 1
        self._change(self.pending[common_pending:], -1, output=True)
        self._change(pending[common_pending:], 1, output=True)
        self.pending = pending

    def apply(
        self,
        logits: torch.Tensor,
        sampling: EngineSampling,
    ) -> None:
        if logits.device.type == "cpu":
            repetition_indices, output_indices = self._cpu_active_indices()
            if sampling.repetition_penalty != 1.0 and repetition_indices.numel():
                values = logits[repetition_indices]
                logits[repetition_indices] = torch.where(
                    values > 0,
                    values / sampling.repetition_penalty,
                    values * sampling.repetition_penalty,
                )
            if output_indices.numel():
                values = logits[output_indices]
                if sampling.frequency_penalty != 0.0:
                    values.sub_(sampling.frequency_penalty * self.output_counts[output_indices])
                if sampling.presence_penalty != 0.0:
                    values.sub_(sampling.presence_penalty)
                logits[output_indices] = values
            return
        if sampling.repetition_penalty != 1.0:
            seen = self.repetition_seen
            logits.copy_(
                torch.where(
                    seen,
                    torch.where(
                        logits > 0,
                        logits / sampling.repetition_penalty,
                        logits * sampling.repetition_penalty,
                    ),
                    logits,
                )
            )
        if self.committed or self.pending:
            if sampling.frequency_penalty != 0.0:
                logits.sub_(sampling.frequency_penalty * self.output_counts)
            if sampling.presence_penalty != 0.0:
                logits.sub_(sampling.presence_penalty * self.output_seen.to(logits.dtype))


class _RequestSamplerState:
    __slots__ = (
        "base_seed",
        "enforcer",
        "accepted_position",
        "penalty_states",
    )

    def __init__(self, base_seed: int, enforcer: XGrammarEnforcer | None) -> None:
        self.base_seed = base_seed
        self.enforcer = enforcer
        self.accepted_position = -1
        self.penalty_states: dict[tuple[str, int | None, int], _IncrementalPenaltyState] = {}


class Sampler:
    """Per-request sampling state: stateless seed identity + grammar enforcers.

    ``vocab_provider`` supplies token strings for grammar compilation; required
    only when a request carries ``json_schema``/``json_mode``. State is dropped
    via ``release`` (owners call it on finish). Penalty-free state is a few
    scalars; penalty-active state lazily owns one vocabulary-sized row.
    """

    def __init__(
        self,
        vocab_provider: Callable[[], list[str] | GrammarVocabulary] | None = None,
    ) -> None:
        self._vocab_provider = vocab_provider
        self._states: dict[str, _RequestSamplerState] = {}

    def _state_for(
        self,
        request_id: str,
        sampling: EngineSampling,
        eos_token_id: int | None = None,
    ) -> _RequestSamplerState:
        state = self._states.get(request_id)
        if state is None:
            enforcer = None
            if sampling.needs_grammar:
                if self._vocab_provider is None:
                    raise RuntimeError("structured output requires a Sampler with a vocab_provider")
                if eos_token_id is None:
                    raise RuntimeError(
                        "structured output requires an eos token id — a completed "
                        "grammar terminates by sampling it (m8 D2)"
                    )
                enforcer = XGrammarEnforcer(
                    self._vocab_provider(),
                    json_schema=sampling.json_schema,
                    stop_token_id=eos_token_id,
                )
            base = sampling.seed if sampling.seed is not None else stable_request_seed(request_id)
            state = _RequestSamplerState(base, enforcer)
            self._states[request_id] = state
        return state

    def release(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def hand_over(self, request_id: str, destination: Sampler) -> bool:
        """Move one request's sampling state to another Sampler (P-D, m5 D5).

        The P-D pair samples token 0 on the prefill engine and token 1 onward on
        the decode engine, each with its own ``Sampler``. Recreating the state
        there would restart the grammar matcher from its INITIAL state — one
        that has not accepted token 0 — so every later mask would be computed
        against the wrong position in the grammar. Moving the object instead
        carries the matcher's accept state, and with it the base seed, exactly
        as a single engine would have kept them. Returns False when there is
        nothing to move (no grammar, no explicit state yet — a fresh state is
        then equivalent, because the base seed derives from the same public id).
        """
        state = self._states.pop(request_id, None)
        if state is None:
            return False
        destination._states[request_id] = state
        return True

    @staticmethod
    def _uses_penalties(sampling: EngineSampling) -> bool:
        return (
            sampling.repetition_penalty != 1.0
            or sampling.presence_penalty != 0.0
            or sampling.frequency_penalty != 0.0
        )

    @staticmethod
    def _penalty_key(logits: torch.Tensor) -> tuple[str, int | None, int]:
        return (
            logits.device.type,
            logits.device.index,
            logits.shape[-1],
        )

    def _apply_incremental_penalties(
        self,
        state: _RequestSamplerState,
        logits: torch.Tensor,
        sampling: EngineSampling,
        prompt: tuple[int, ...],
        outputs: Sequence[int],
        pending_outputs: Sequence[int | torch.Tensor] = (),
        history_epoch: int | None = None,
    ) -> None:
        if not self._uses_penalties(sampling):
            return
        key = self._penalty_key(logits)
        penalties = state.penalty_states.get(key)
        if penalties is None:
            # A P-D handoff can move this request to another GPU. The old
            # device row is no longer useful; keeping it would retain one dense
            # vocab-sized state per engine hop.
            state.penalty_states.clear()
            penalties = _IncrementalPenaltyState(
                prompt,
                logits.shape[-1],
                logits.device,
            )
            state.penalty_states[key] = penalties
        penalties.sync(
            prompt,
            outputs,
            pending_outputs,
            include_prompt=sampling.repetition_penalty != 1.0,
            history_epoch=history_epoch,
        )
        penalties.apply(logits, sampling)

    def can_argmax_logits(
        self,
        request_id: str,
        sampling: EngineSampling,
        eos_token_id: int | None = None,
        *,
        position: int = 0,
        stop_token_ids: Sequence[int] = (),
        min_tokens: int = 0,
    ) -> bool:
        """Whether selection can happen on the logits device before host copy.

        Pure greedy sampling needs only the winning index. Copying the entire
        vocabulary row to CPU first is both unnecessary and expensive; callers
        with a batch can argmax every row together and transfer only the small
        index vector. State is still materialized here so lifecycle and grammar
        handoff semantics remain unchanged.
        """
        if not sampling.is_greedy_pure or sampling.logprobs is not None:
            return False
        state = self._state_for(request_id, sampling, eos_token_id)
        if state.enforcer is not None:
            return False
        return not (
            position < min_tokens
            and (eos_token_id is not None or bool(stop_token_ids))
        )

    @staticmethod
    def _apply_min_tokens_mask(
        logits: torch.Tensor,
        *,
        position: int,
        min_tokens: int,
        eos_token_id: int | None,
        stop_token_ids: Sequence[int],
    ) -> None:
        """Prevent a stop-valued token from becoming ordinary model context.

        vLLM's min-token processor masks its complete stop set while the number
        of prior output tokens is below the requested minimum.  ``position`` is
        that prior-token count even when the scheduler snapshot lags behind
        in-flight overlap/speculative work.  EOS remains in this mask when
        ``ignore_eos`` controls termination later; the sampler intentionally
        does not need that flag.
        """
        if position >= min_tokens:
            return
        vocab_size = logits.shape[-1]
        blocked: list[int] = []
        seen: set[int] = set()
        for token_id in (*stop_token_ids, eos_token_id):
            if (
                token_id is not None
                and 0 <= token_id < vocab_size
                and token_id not in seen
            ):
                seen.add(token_id)
                blocked.append(token_id)
        if blocked:
            indices = torch.as_tensor(blocked, dtype=torch.long, device=logits.device)
            logits.index_fill_(0, indices, float("-inf"))

    @staticmethod
    def _forced_token_id(
        sampling: EngineSampling,
        position: int,
        vocab_size: int,
    ) -> int | None:
        forced = sampling.forced_token_ids
        if forced is None:
            return None
        if type(position) is not int or not 0 <= position < len(forced):
            raise ValueError(
                "forced_token_ids has no token for output position "
                f"{position}; sequence length is {len(forced)}"
            )
        token_id = forced[position]
        if type(token_id) is not int or not 0 <= token_id < vocab_size:
            raise ValueError(
                f"forced token ID {token_id!r} at output position {position} "
                f"is outside vocabulary size {vocab_size}"
            )
        return token_id

    def sample(
        self,
        request_id: str,
        sampling: EngineSampling,
        position: int,
        logits: torch.Tensor,
        *,
        prompt: tuple[int, ...] = (),
        outputs: Sequence[int] = (),
        pending_outputs: Sequence[int] = (),
        history_epoch: int | None = None,
        eos_token_id: int | None = None,
        stop_token_ids: Sequence[int] = (),
        min_tokens: int = 0,
    ) -> SampledToken:
        state = self._state_for(request_id, sampling, eos_token_id)
        forced_token_id = self._forced_token_id(
            sampling,
            position,
            logits.shape[-1],
        )
        if self.can_argmax_logits(
            request_id,
            sampling,
            eos_token_id,
            position=position,
            stop_token_ids=stop_token_ids,
            min_tokens=min_tokens,
        ):
            return SampledToken(int(torch.argmax(logits).item()))
        # CPU and structured-output compatibility path. CUDA grammar-free
        # requests use sample_device(), which keeps the decision and penalty
        # history dependency on-device. XGrammar's matcher is stateful host code,
        # so masking/acceptance deliberately stays here until it has a device FSM.
        logits = logits.detach().to(device="cpu", dtype=torch.float32).clone()

        raw_logsoftmax: torch.Tensor | None = None
        if sampling.logprobs is not None or forced_token_id is not None:
            raw_logsoftmax = torch.log_softmax(logits, dim=-1)

        if state.enforcer is not None:
            state.enforcer.mask_logits(logits)

        self._apply_min_tokens_mask(
            logits,
            position=position,
            min_tokens=min_tokens,
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
        )

        self._apply_incremental_penalties(
            state,
            logits,
            sampling,
            prompt,
            outputs,
            pending_outputs,
            history_epoch,
        )

        if forced_token_id is not None:
            token_id = forced_token_id
        elif sampling.temperature == 0.0:
            token_id = int(torch.argmax(logits).item())
        else:
            token_id = self._sample_scaled(logits, sampling, state.base_seed, position)

        logprob, top_logprobs = self._report(raw_logsoftmax, sampling, token_id)
        terminated = False
        if state.enforcer is not None:
            terminated = self._accept_once(state, position, token_id)
        return SampledToken(token_id, logprob, top_logprobs, terminated)

    def sample_device(
        self,
        request_id: str,
        sampling: EngineSampling,
        position: int,
        logits: torch.Tensor,
        *,
        prompt: tuple[int, ...] = (),
        outputs: Sequence[int] = (),
        pending_outputs: Sequence[torch.Tensor] = (),
        history_epoch: int | None = None,
        eos_token_id: int | None = None,
        stop_token_ids: Sequence[int] = (),
        min_tokens: int = 0,
    ) -> DeviceSample:
        """Sample without reading a device value on the host.

        ``pending_outputs`` are the tokens already sampled by overlap but not
        committed into the scheduler snapshot yet.  They stay as device
        scalars, so penalties cover the same effective history as the CPU path
        without first resolving those tokens to Python ints.

        XGrammar's matcher is stateful CPU code: advancing it requires the
        previous token as a host integer.  Structured requests therefore keep
        the reviewed CPU compatibility path rather than silently applying a
        stale grammar mask.  All other supported sampling modes use this path.
        """
        state = self._state_for(request_id, sampling, eos_token_id)
        if state.enforcer is not None:
            raise ValueError("structured sampling requires the CPU matcher path")

        forced_token_id = self._forced_token_id(
            sampling,
            position,
            logits.shape[-1],
        )

        work = self._float_working_copy(logits)
        raw_logsoftmax: torch.Tensor | None = None
        if sampling.logprobs is not None or forced_token_id is not None:
            raw_logsoftmax = torch.log_softmax(work, dim=-1)

        self._apply_min_tokens_mask(
            work,
            position=position,
            min_tokens=min_tokens,
            eos_token_id=eos_token_id,
            stop_token_ids=stop_token_ids,
        )

        self._apply_incremental_penalties(
            state,
            work,
            sampling,
            prompt,
            outputs,
            pending_outputs,
            history_epoch,
        )

        if forced_token_id is not None:
            token_id = torch.tensor(
                forced_token_id,
                dtype=torch.int64,
                device=work.device,
            )
        elif sampling.temperature == 0.0:
            token_id = torch.argmax(work).to(dtype=torch.int64)
        else:
            token_id = self._sample_scaled_device(work, sampling, state.base_seed, position)

        logprob = None
        top_indices = None
        top_logprobs = None
        if raw_logsoftmax is not None:
            logprob = raw_logsoftmax.gather(0, token_id.view(1)).squeeze(0)
            if sampling.logprobs is not None and sampling.logprobs > 0:
                k = min(sampling.logprobs, raw_logsoftmax.shape[-1])
                top_logprobs, top_indices = torch.topk(raw_logsoftmax, k)
        return DeviceSample(
            token_id=token_id,
            logprob=logprob,
            top_indices=top_indices,
            top_logprobs=top_logprobs,
        )

    @staticmethod
    def _float_working_copy(logits: torch.Tensor) -> torch.Tensor:
        """Return one mutable fp32 copy without copying converted logits twice."""

        detached = logits.detach()
        if detached.dtype == torch.float32:
            return detached.clone()
        return detached.to(dtype=torch.float32)

    def sample_batch_device(
        self,
        rows: Sequence[DeviceSamplingInput],
        logits: torch.Tensor,
    ) -> tuple[DeviceSample, ...]:
        """Sample grammar-free rows with shared batched filtering and RNG ops."""

        if logits.ndim != 2:
            raise ValueError(f"batched logits must be 2D, got {tuple(logits.shape)}")
        if len(rows) != logits.shape[0]:
            raise ValueError(
                "sampling row count must match logits batch: "
                f"rows={len(rows)}, batch={logits.shape[0]}"
            )
        if not rows:
            return ()
        if any(row.sampling.needs_grammar for row in rows):
            raise ValueError("structured sampling requires the per-row CPU path")

        states = tuple(
            self._state_for(
                row.request_id,
                row.sampling,
                row.eos_token_id,
            )
            for row in rows
        )
        if any(state.enforcer is not None for state in states):
            raise ValueError("structured sampling requires the per-row CPU path")
        forced_token_ids = tuple(
            self._forced_token_id(
                row.sampling,
                row.position,
                logits.shape[-1],
            )
            for row in rows
        )

        work = self._float_working_copy(logits)
        needs_raw = tuple(
            row.sampling.logprobs is not None or forced_token_id is not None
            for row, forced_token_id in zip(rows, forced_token_ids, strict=True)
        )
        raw_logsoftmax = (
            torch.log_softmax(work, dim=-1) if any(needs_raw) else None
        )

        blocked_rows: list[int] = []
        blocked_tokens: list[int] = []
        vocab_size = work.shape[-1]
        for index, row in enumerate(rows):
            if row.position >= row.min_tokens:
                continue
            seen: set[int] = set()
            for token_id in (*row.stop_token_ids, row.eos_token_id):
                if (
                    token_id is not None
                    and 0 <= token_id < vocab_size
                    and token_id not in seen
                ):
                    seen.add(token_id)
                    blocked_rows.append(index)
                    blocked_tokens.append(token_id)
        if blocked_rows:
            row_indices = _sampling_metadata_tensor(
                blocked_rows,
                dtype=torch.long,
                device=work.device,
            )
            token_indices = _sampling_metadata_tensor(
                blocked_tokens,
                dtype=torch.long,
                device=work.device,
            )
            work[row_indices, token_indices] = float("-inf")

        for index, (row, state) in enumerate(zip(rows, states, strict=True)):
            self._apply_incremental_penalties(
                state,
                work[index],
                row.sampling,
                row.prompt,
                row.outputs,
                row.pending_outputs,
                row.history_epoch,
            )

        stochastic_rows = tuple(
            index
            for index, (row, forced_token_id) in enumerate(
                zip(rows, forced_token_ids, strict=True)
            )
            if forced_token_id is None and row.sampling.temperature != 0.0
        )
        if len(stochastic_rows) == len(rows):
            token_ids = self._sample_scaled_batch_device(
                work,
                tuple(row.sampling for row in rows),
                tuple(state.base_seed for state in states),
                tuple(row.position for row in rows),
            )
        else:
            token_ids = torch.argmax(work, dim=-1).to(dtype=torch.int64)
        if stochastic_rows and len(stochastic_rows) != len(rows):
            indices = _sampling_metadata_tensor(
                stochastic_rows,
                dtype=torch.long,
                device=work.device,
            )
            selected = self._sample_scaled_batch_device(
                work.index_select(0, indices),
                tuple(rows[index].sampling for index in stochastic_rows),
                tuple(states[index].base_seed for index in stochastic_rows),
                tuple(rows[index].position for index in stochastic_rows),
            )
            token_ids = token_ids.index_copy(0, indices, selected)

        forced_rows = tuple(
            index
            for index, token_id in enumerate(forced_token_ids)
            if token_id is not None
        )
        if forced_rows:
            indices = _sampling_metadata_tensor(
                forced_rows,
                dtype=torch.long,
                device=work.device,
            )
            values = _sampling_metadata_tensor(
                [forced_token_ids[index] for index in forced_rows],
                dtype=torch.int64,
                device=work.device,
            )
            token_ids = token_ids.index_copy(0, indices, values)

        selected_logprobs = None
        if raw_logsoftmax is not None:
            selected_logprobs = raw_logsoftmax.gather(
                1,
                token_ids.unsqueeze(1),
            ).squeeze(1)
        max_report = max(
            (
                min(row.sampling.logprobs, vocab_size)
                for row in rows
                if row.sampling.logprobs is not None
                and row.sampling.logprobs > 0
            ),
            default=0,
        )
        top_logprobs = None
        top_indices = None
        if max_report > 0:
            assert raw_logsoftmax is not None
            top_logprobs, top_indices = torch.topk(
                raw_logsoftmax,
                max_report,
                dim=-1,
            )

        samples: list[DeviceSample] = []
        for index, row in enumerate(rows):
            report_count = row.sampling.logprobs
            top_count = (
                0
                if report_count is None
                else min(report_count, vocab_size)
            )
            samples.append(
                DeviceSample(
                    token_id=token_ids[index],
                    logprob=(
                        selected_logprobs[index]
                        if needs_raw[index] and selected_logprobs is not None
                        else None
                    ),
                    top_indices=(
                        top_indices[index, :top_count]
                        if top_count > 0 and top_indices is not None
                        else None
                    ),
                    top_logprobs=(
                        top_logprobs[index, :top_count]
                        if top_count > 0 and top_logprobs is not None
                        else None
                    ),
                )
            )
        return tuple(samples)

    @staticmethod
    def _sample_scaled_batch_device(
        logits: torch.Tensor,
        samplings: Sequence[EngineSampling],
        base_seeds: Sequence[int],
        positions: Sequence[int],
    ) -> torch.Tensor:
        """Apply row-specific filters with batched vocabulary operations."""

        if logits.ndim != 2:
            raise ValueError(f"batched logits must be 2D, got {tuple(logits.shape)}")
        batch_size, vocab_size = logits.shape
        if vocab_size < 1:
            raise ValueError("batched sampling requires a non-empty vocabulary")
        if not (
            len(samplings) == len(base_seeds) == len(positions) == batch_size
        ):
            raise ValueError("batched sampling metadata must match logits rows")
        if any(sampling.temperature == 0.0 for sampling in samplings):
            raise ValueError("scaled batched sampling requires non-zero temperature")
        max_rows = max(1, _MAX_BATCHED_SAMPLING_ELEMENTS // vocab_size)
        if batch_size > max_rows:
            return torch.cat(
                tuple(
                    Sampler._sample_scaled_batch_device(
                        logits[start : start + max_rows],
                        samplings[start : start + max_rows],
                        base_seeds[start : start + max_rows],
                        positions[start : start + max_rows],
                    )
                    for start in range(0, batch_size, max_rows)
                )
            )

        temperatures = _sampling_metadata_tensor(
            [sampling.temperature for sampling in samplings],
            dtype=logits.dtype,
            device=logits.device,
        ).unsqueeze(1)
        probs = torch.softmax(logits / temperatures, dim=-1)

        min_ps = _sampling_metadata_tensor(
            [sampling.min_p for sampling in samplings],
            dtype=probs.dtype,
            device=probs.device,
        ).unsqueeze(1)
        if any(sampling.min_p > 0.0 for sampling in samplings):
            probs = torch.where(
                probs >= min_ps * probs.max(dim=-1, keepdim=True).values,
                probs,
                torch.zeros_like(probs),
            )

        bounded_top_k = tuple(
            index
            for index, sampling in enumerate(samplings)
            if 0 < sampling.top_k < vocab_size
        )
        top_values = None
        top_indices = None
        max_top_k = 0
        if bounded_top_k:
            max_top_k = max(samplings[index].top_k for index in bounded_top_k)
            top_values, top_indices = torch.topk(
                probs,
                max_top_k,
                dim=-1,
                sorted=True,
            )
            per_row_k = _sampling_metadata_tensor(
                [
                    sampling.top_k
                    if 0 < sampling.top_k < vocab_size
                    else max_top_k
                    for sampling in samplings
                ],
                dtype=torch.long,
                device=probs.device,
            )
            thresholds = top_values.gather(
                1,
                (per_row_k - 1).unsqueeze(1),
            )
            bounded = set(bounded_top_k)
            bounded_mask = _sampling_metadata_tensor(
                [index in bounded for index in range(batch_size)],
                dtype=torch.bool,
                device=probs.device,
            ).unsqueeze(1)
            probs = torch.where(
                bounded_mask & (probs < thresholds),
                torch.zeros_like(probs),
                probs,
            )

        bounded_nucleus = tuple(
            index
            for index in bounded_top_k
            if samplings[index].top_p < 1.0
        )
        if bounded_nucleus:
            assert top_values is not None
            assert top_indices is not None
            row_indices = _sampling_metadata_tensor(
                bounded_nucleus,
                dtype=torch.long,
                device=probs.device,
            )
            values = top_values.index_select(0, row_indices)
            indices = top_indices.index_select(0, row_indices)
            row_ks = _sampling_metadata_tensor(
                [samplings[index].top_k for index in bounded_nucleus],
                dtype=torch.long,
                device=probs.device,
            ).unsqueeze(1)
            ranks = torch.arange(max_top_k, device=probs.device).unsqueeze(0)
            values = torch.where(
                ranks < row_ks,
                values,
                torch.zeros_like(values),
            )
            cumulative = torch.cumsum(values, dim=-1)
            top_ps = _sampling_metadata_tensor(
                [samplings[index].top_p for index in bounded_nucleus],
                dtype=probs.dtype,
                device=probs.device,
            ).unsqueeze(1)
            cut = cumulative - values >= top_ps * cumulative[:, -1:].clamp(
                min=1e-12
            )
            values = values.masked_fill(cut, 0.0)
            filtered = torch.zeros(
                (len(bounded_nucleus), vocab_size),
                dtype=probs.dtype,
                device=probs.device,
            ).scatter(1, indices, values)
            probs = probs.index_copy(0, row_indices, filtered)

        unbounded_nucleus = tuple(
            index
            for index, sampling in enumerate(samplings)
            if sampling.top_p < 1.0
            and not (0 < sampling.top_k < vocab_size)
        )
        if unbounded_nucleus:
            row_indices = _sampling_metadata_tensor(
                unbounded_nucleus,
                dtype=torch.long,
                device=probs.device,
            )
            selected = probs.index_select(0, row_indices)
            values, indices = torch.sort(selected, dim=-1, descending=True)
            cumulative = torch.cumsum(values, dim=-1)
            top_ps = _sampling_metadata_tensor(
                [samplings[index].top_p for index in unbounded_nucleus],
                dtype=probs.dtype,
                device=probs.device,
            ).unsqueeze(1)
            cut = cumulative - values >= top_ps * cumulative[:, -1:].clamp(
                min=1e-12
            )
            values = values.masked_fill(cut, 0.0)
            filtered = torch.zeros_like(selected).scatter(1, indices, values)
            probs = probs.index_copy(0, row_indices, filtered)

        totals = probs.sum(dim=-1, keepdim=True)
        if logits.device.type == "cpu" and bool(torch.all(totals > 0)):
            safe = probs / totals
        else:
            argmax = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int64)
            fallback = torch.zeros_like(probs).scatter(
                1,
                argmax,
                torch.ones(
                    (batch_size, 1),
                    dtype=probs.dtype,
                    device=probs.device,
                ),
            )
            safe = torch.where(totals > 0, probs, fallback)
            safe = safe / safe.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return stateless_gumbel_argmax_batched(
            torch.log(safe),
            [
                mix_seed(base_seed, position)
                for base_seed, position in zip(
                    base_seeds,
                    positions,
                    strict=True,
                )
            ],
        )

    @staticmethod
    def _sample_scaled_device(
        logits: torch.Tensor,
        sampling: EngineSampling,
        base_seed: int,
        position: int,
    ) -> torch.Tensor:
        """The reviewed filter order with no scalar extraction."""
        return Sampler._sample_scaled_batch_device(
            logits.unsqueeze(0),
            (sampling,),
            (base_seed,),
            (position,),
        )[0]

    @staticmethod
    def _sample_scaled(
        logits: torch.Tensor, sampling: EngineSampling, base_seed: int, position: int
    ) -> int:
        # The CPU/structured boundary is the only place that materializes the
        # canonical tensor draw as a Python integer.  Filtering and RNG remain
        # exactly the same implementation as the device path.
        return int(
            Sampler._sample_scaled_device(
                logits,
                sampling,
                base_seed,
                position,
            ).item()
        )

    @staticmethod
    def _report(
        raw_logsoftmax: torch.Tensor | None, sampling: EngineSampling, token_id: int
    ) -> tuple[float | None, tuple[tuple[int, float], ...] | None]:
        if raw_logsoftmax is None:
            return None, None
        logprob = float(raw_logsoftmax[token_id].item())
        top: tuple[tuple[int, float], ...] | None = None
        if sampling.logprobs is not None and sampling.logprobs > 0:
            k = min(sampling.logprobs, raw_logsoftmax.shape[-1])
            values, indices = torch.topk(raw_logsoftmax, k)
            top = tuple(
                (int(index.item()), float(value.item()))
                for index, value in zip(indices, values, strict=True)
            )
        return logprob, top

    @staticmethod
    def _accept_once(state: _RequestSamplerState, position: int, token_id: int) -> bool:
        """Advance the grammar matcher exactly once per output position."""
        assert state.enforcer is not None
        if position <= state.accepted_position:
            return state.enforcer.is_terminated()
        state.accepted_position = position
        if not state.enforcer.accept(token_id):
            raise RuntimeError(
                f"grammar rejected token {token_id} sampled under its own mask "
                "(invariant violation)"
            )
        return state.enforcer.is_terminated()
