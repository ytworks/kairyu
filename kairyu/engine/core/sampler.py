"""The Sampler: penalties, temperature, min-p/top-k/top-p, grammar (design m8 D2).

Order of operations (reviewed convention, m8 §6):

1. raw logits → capture ``log_softmax`` for logprob reporting (vLLM v1
   ``raw_logprobs`` default; temperature-independent, OpenAI-style).
2. xgrammar mask FIRST (mask-last can leave zero legal tokens after top-k/top-p
   → NaN multinomial; mask-first plus keep-1 filters guarantees support).
3. Penalties — repetition over prompt + committed outputs; presence/frequency
   over committed outputs only (vLLM/HF agreement).
4. ``temperature == 0`` → argmax on the masked logits; else scale.
5. min_p, then top-k, then top-p (vLLM v1 order; HF differs — recorded).
6. softmax → seeded sample.

Determinism: base seed is ``sampling.seed`` or sha256(request_id) (never
Python ``hash()`` — process-randomized); the per-position seed is a splitmix64
mix of (base, position). CPU compatibility uses a seeded Generator. CUDA uses
stateless Gumbel noise derived from (seed, position, vocab index), so TP rank
runners sample identically given identical logits and replaying a position is
idempotent without a host-owned RNG offset. Grammar ``accept`` is likewise
idempotent per position — the matcher advances exactly once per committed
token.
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


class _RequestSamplerState:
    __slots__ = ("base_seed", "enforcer", "accepted_position")

    def __init__(self, base_seed: int, enforcer: XGrammarEnforcer | None) -> None:
        self.base_seed = base_seed
        self.enforcer = enforcer
        self.accepted_position = -1


class Sampler:
    """Per-request sampling state: seeded generators + grammar enforcers.

    ``vocab_provider`` supplies token strings for grammar compilation; required
    only when a request carries ``json_schema``/``json_mode``. State is dropped
    via ``release`` (owners call it on finish; unreleased state is a few ints).
    """

    def __init__(self, vocab_provider: Callable[[], list[str]] | None = None) -> None:
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
                    raise RuntimeError(
                        "structured output requires a Sampler with a vocab_provider"
                    )
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

    def can_argmax_logits(
        self,
        request_id: str,
        sampling: EngineSampling,
        eos_token_id: int | None = None,
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
        return state.enforcer is None

    def sample(
        self,
        request_id: str,
        sampling: EngineSampling,
        position: int,
        logits: torch.Tensor,
        *,
        prompt: tuple[int, ...] = (),
        outputs: Sequence[int] = (),
        eos_token_id: int | None = None,
    ) -> SampledToken:
        state = self._state_for(request_id, sampling, eos_token_id)
        if self.can_argmax_logits(request_id, sampling, eos_token_id):
            return SampledToken(int(torch.argmax(logits).item()))
        # CPU and structured-output compatibility path. CUDA grammar-free
        # requests use sample_device(), which keeps the decision and penalty
        # history dependency on-device. XGrammar's matcher is stateful host code,
        # so masking/acceptance deliberately stays here until it has a device FSM.
        logits = logits.detach().to(device="cpu", dtype=torch.float32).clone()

        raw_logsoftmax: torch.Tensor | None = None
        if sampling.logprobs is not None:
            raw_logsoftmax = torch.log_softmax(logits, dim=-1)

        if state.enforcer is not None:
            state.enforcer.mask_logits(logits)

        self._apply_penalties(logits, sampling, prompt, outputs)

        if sampling.temperature == 0.0:
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
        eos_token_id: int | None = None,
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

        work = logits.detach().to(dtype=torch.float32).clone()
        raw_logsoftmax: torch.Tensor | None = None
        if sampling.logprobs is not None:
            raw_logsoftmax = torch.log_softmax(work, dim=-1)

        self._apply_penalties_device(
            work,
            sampling,
            prompt,
            outputs,
            pending_outputs,
        )

        if sampling.temperature == 0.0:
            token_id = torch.argmax(work).to(dtype=torch.int64)
        else:
            token_id = self._sample_scaled_device(
                work, sampling, state.base_seed, position
            )

        logprob = None
        top_indices = None
        top_logprobs = None
        if raw_logsoftmax is not None:
            logprob = raw_logsoftmax.gather(0, token_id.view(1)).squeeze(0)
            assert sampling.logprobs is not None
            if sampling.logprobs > 0:
                k = min(sampling.logprobs, raw_logsoftmax.shape[-1])
                top_logprobs, top_indices = torch.topk(raw_logsoftmax, k)
        return DeviceSample(
            token_id=token_id,
            logprob=logprob,
            top_indices=top_indices,
            top_logprobs=top_logprobs,
        )

    @staticmethod
    def _apply_penalties_device(
        logits: torch.Tensor,
        sampling: EngineSampling,
        prompt: tuple[int, ...],
        outputs: Sequence[int],
        pending_outputs: Sequence[torch.Tensor],
    ) -> None:
        """Device equivalent of ``_apply_penalties`` for an overlap history."""
        device = logits.device
        vocab = logits.shape[-1]
        host_outputs = tuple(outputs)

        if sampling.repetition_penalty != 1.0:
            seen = torch.zeros(vocab, dtype=torch.bool, device=device)
            host_seen = tuple(sorted(set(prompt) | set(host_outputs)))
            if host_seen:
                seen.scatter_(
                    0,
                    torch.as_tensor(host_seen, dtype=torch.long, device=device),
                    True,
                )
            for token in pending_outputs:
                seen.scatter_(0, token.to(device=device, dtype=torch.long).view(1), True)
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

        if sampling.presence_penalty != 0.0 or sampling.frequency_penalty != 0.0:
            counts = torch.zeros(vocab, dtype=torch.float32, device=device)
            if host_outputs:
                host_indices = torch.as_tensor(
                    host_outputs, dtype=torch.long, device=device
                )
                counts.scatter_add_(
                    0, host_indices, torch.ones_like(host_indices, dtype=torch.float32)
                )
            for token in pending_outputs:
                index = token.to(device=device, dtype=torch.long).view(1)
                counts.scatter_add_(0, index, torch.ones(1, device=device))
            logits.sub_(sampling.frequency_penalty * counts)
            logits.sub_(sampling.presence_penalty * (counts > 0).to(logits.dtype))

    @staticmethod
    def _sample_scaled_device(
        logits: torch.Tensor,
        sampling: EngineSampling,
        base_seed: int,
        position: int,
    ) -> torch.Tensor:
        """The reviewed filter order with no scalar extraction."""
        probs = torch.softmax(logits / sampling.temperature, dim=-1)
        if sampling.min_p > 0.0:
            probs = torch.where(
                probs >= sampling.min_p * probs.max(), probs, torch.zeros_like(probs)
            )
        if 0 < sampling.top_k < probs.shape[-1]:
            threshold = torch.topk(probs, sampling.top_k).values[-1]
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        if sampling.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cut = (
                cumulative - sorted_probs
                >= sampling.top_p * cumulative[-1].clamp(min=1e-12)
            )
            sorted_probs = sorted_probs.masked_fill(cut, 0.0)
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)

        # Grammar-free filtering always retains one token, but keep the CPU
        # path's degenerate fallback without branching on a device scalar.
        total = probs.sum()
        argmax = torch.argmax(logits).to(dtype=torch.int64)
        fallback = torch.zeros_like(probs).scatter(
            0, argmax.view(1), torch.ones(1, dtype=probs.dtype, device=probs.device)
        )
        safe = torch.where(total > 0, probs, fallback)
        safe = safe / safe.sum().clamp(min=1e-12)
        mixed_seed = mix_seed(base_seed, position)
        if logits.device.type == "cuda":
            from kairyu.kernels.sampling_gpu import stateless_gumbel_argmax

            return stateless_gumbel_argmax(torch.log(safe), mixed_seed)
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(mixed_seed)
        return torch.multinomial(safe, 1, generator=generator).squeeze(0)

    @staticmethod
    def _apply_penalties(
        logits: torch.Tensor,
        sampling: EngineSampling,
        prompt: tuple[int, ...],
        outputs: Sequence[int],
    ) -> None:
        if sampling.repetition_penalty != 1.0:
            seen = torch.tensor(sorted(set(prompt) | set(outputs)), dtype=torch.long)
            if seen.numel():
                values = logits[seen]
                logits[seen] = torch.where(
                    values > 0,
                    values / sampling.repetition_penalty,
                    values * sampling.repetition_penalty,
                )
        if outputs and (sampling.presence_penalty != 0.0 or sampling.frequency_penalty != 0.0):
            counts = torch.bincount(
                torch.tensor(outputs, dtype=torch.long), minlength=logits.shape[-1]
            ).to(logits.dtype)
            logits -= sampling.frequency_penalty * counts
            logits -= sampling.presence_penalty * (counts > 0).to(logits.dtype)

    @staticmethod
    def _sample_scaled(
        logits: torch.Tensor, sampling: EngineSampling, base_seed: int, position: int
    ) -> int:
        probs = torch.softmax(logits / sampling.temperature, dim=-1)
        if sampling.min_p > 0.0:
            probs = torch.where(
                probs >= sampling.min_p * probs.max(), probs, torch.zeros_like(probs)
            )
        if 0 < sampling.top_k < probs.shape[-1]:
            threshold = torch.topk(probs, sampling.top_k).values[-1]
            probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        if sampling.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # drop tokens whose exclusive cumulative mass already reaches top_p;
            # the highest-probability token always survives (keep-1)
            cut = cumulative - sorted_probs >= sampling.top_p * cumulative[-1].clamp(min=1e-12)
            sorted_probs[cut] = 0.0
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        total = probs.sum()
        if total <= 0:  # every candidate filtered (degenerate): fall back to argmax
            return int(torch.argmax(logits).item())
        generator = torch.Generator().manual_seed(mix_seed(base_seed, position))
        return int(torch.multinomial(probs / total, 1, generator=generator).item())

    @staticmethod
    def _report(
        raw_logsoftmax: torch.Tensor | None, sampling: EngineSampling, token_id: int
    ) -> tuple[float | None, tuple[tuple[int, float], ...] | None]:
        if raw_logsoftmax is None:
            return None, None
        logprob = float(raw_logsoftmax[token_id].item())
        top: tuple[tuple[int, float], ...] | None = None
        assert sampling.logprobs is not None
        if sampling.logprobs > 0:
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
