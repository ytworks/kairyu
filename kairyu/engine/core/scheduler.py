"""Chunked-prefill scheduler policy: decode-priority token budget (design doc §2.4).

Pure policy, no GPU: schedule() plans one engine step (which requests compute
how many tokens), update() commits sampled tokens. KV admission goes through
RadixKVCache so waiting requests naturally queue under memory pressure and
finished prompts stay reusable as shared prefixes.

Capacity accounting is at page granularity: before a decode step, a request
must own KV capacity for prompt+outputs+1 slots (the +1 covers the KV entry of
the token being generated).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum

from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache
from kairyu.engine.core.sampling_types import EngineSampling

_DEFAULT_TOKEN_BUDGET = 2048
_DEFAULT_MAX_SEQS = 256


@dataclass(frozen=True)
class EngineRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int = 16
    eos_token_id: int | None = None
    # m8 additions are keyword-only so positional construction can never
    # silently shift meaning across milestones
    _: KW_ONLY
    stop_token_ids: tuple[int, ...] = ()
    min_tokens: int = 0
    ignore_eos: bool = False
    priority: int = 0  # admission ordering lands in M11; field reserved here
    sampling: EngineSampling = field(default_factory=EngineSampling)


class _Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class _RequestState:
    __slots__ = (
        "request",
        "status",
        "computed_prompt",
        "outputs",
        "in_flight",
        "surplus_in_flight",
        "allocation",
        "decode_pages",
        "finish_reason",
    )

    def __init__(self, request: EngineRequest) -> None:
        self.request = request
        self.status = _Status.WAITING
        self.computed_prompt = 0
        self.outputs: list[int] = []
        # sampled tokens scheduled but not yet committed via update() — this is
        # what lets the overlap loop plan step N+1 before step N's tokens land
        self.in_flight = 0
        # tokens still in flight when the request finished/preempted early
        # (EOS under overlap); their late arrivals are trimmed, not errors
        self.surplus_in_flight = 0
        self.allocation: KVAllocation | None = None
        self.decode_pages: list[int] = []
        self.finish_reason: str | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.request.prompt_token_ids)

    @property
    def prefill_done(self) -> bool:
        return self.computed_prompt >= self.prompt_len

    def capacity_tokens(self, page_size: int) -> int:
        pages = len(self.allocation.pages) if self.allocation else 0
        return (pages + len(self.decode_pages)) * page_size


@dataclass(frozen=True)
class ScheduledChunk:
    request_id: str
    num_tokens: int
    is_prefill: bool
    position: int = 0  # decode: output index this chunk produces (overlap-safe)


@dataclass(frozen=True)
class SchedulerOutput:
    scheduled: tuple[ScheduledChunk, ...]


class Scheduler:
    def __init__(
        self,
        kv_cache: RadixKVCache,
        max_num_batched_tokens: int = _DEFAULT_TOKEN_BUDGET,
        max_num_seqs: int = _DEFAULT_MAX_SEQS,
        page_size: int = 16,
        pd_separation: bool = False,
        decode_token_budget: int | None = None,
        decode_watermark_pages: int = 0,
    ) -> None:
        if max_num_batched_tokens < 1:
            raise ValueError(f"max_num_batched_tokens must be >= 1, got {max_num_batched_tokens}")
        if pd_separation and (decode_token_budget is None or decode_token_budget < 1):
            raise ValueError("pd_separation=True requires decode_token_budget >= 1")
        self._kv = kv_cache
        self._budget = max_num_batched_tokens
        self._max_seqs = max_num_seqs
        self._pd_separation = pd_separation
        self._decode_budget = decode_token_budget
        self._decode_watermark = decode_watermark_pages
        self._page_size = getattr(kv_cache, "_page_size", page_size)
        self._states: dict[str, _RequestState] = {}
        self._waiting: list[str] = []
        self._running: list[str] = []

    def add_request(self, request: EngineRequest) -> None:
        if request.request_id in self._states:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        self._states[request.request_id] = _RequestState(request)
        self._waiting.append(request.request_id)

    def has_unfinished(self) -> bool:
        return bool(self._waiting or self._running)

    @property
    def states(self) -> dict[str, _RequestState]:
        """Read view of request states for the ModelRunner (do not mutate)."""
        return self._states

    @property
    def waiting_ids(self) -> tuple[str, ...]:
        return tuple(self._waiting)

    def _release_without_commit(self, state: _RequestState) -> None:
        if state.allocation is not None:
            self._kv.release_preempted(state.allocation, tuple(state.decode_pages))
            state.allocation = None
        elif state.decode_pages:
            self._kv.free_private_pages(tuple(state.decode_pages))
        state.decode_pages.clear()
        state.surplus_in_flight += state.in_flight
        state.in_flight = 0

    def abort(self, request_id: str) -> None:
        """Cancel a request (client disconnect); late in-flight tokens are trimmed."""
        state = self._states.get(request_id)
        if state is None or state.status is _Status.FINISHED:
            return
        if state.status is _Status.WAITING:
            self._waiting.remove(request_id)
        else:
            self._running.remove(request_id)
            self._release_without_commit(state)
        state.status = _Status.FINISHED
        state.finish_reason = "abort"

    def finish_early(self, request_id: str) -> None:
        """Finish a running request now (stop-string / grammar termination, m8 D1).

        Unlike ``abort``, outputs are committed to the radix tree via the normal
        ``_finish`` path so multi-turn prefix reuse survives a stop finish.
        Must be called between ``update()`` and the next ``schedule()`` (the
        step-thread op discipline); with in-flight tokens pending the surplus
        trim covers their late arrival exactly like an EOS finish under overlap.
        """
        state = self._states[request_id]
        if state.status is _Status.FINISHED:
            return
        if state.status is _Status.WAITING:
            self._waiting.remove(request_id)
            state.status = _Status.FINISHED
            state.finish_reason = "stop"
            return
        state.surplus_in_flight += state.in_flight
        state.in_flight = 0
        state.finish_reason = "stop"
        self._finish(state)

    def finish_reason(self, request_id: str) -> str | None:
        return self._states[request_id].finish_reason

    def _preempt_for_decode(self, needy_id: str) -> bool:
        """Recompute-preempt the youngest output-free running request.

        Victims are restricted to requests with no committed outputs so
        requeueing them recomputes the prompt only (output-KV recompute is a
        GPU-phase extension, see design m2 §5).
        """
        for victim_id in reversed(self._running):
            state = self._states[victim_id]
            if victim_id == needy_id or state.outputs:
                continue
            self._running.remove(victim_id)
            self._release_without_commit(state)
            state.computed_prompt = 0
            state.status = _Status.WAITING
            self._waiting.insert(0, victim_id)
            return True
        return False

    def output_tokens(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._states[request_id].outputs)

    def resume_with_kv(
        self, request: EngineRequest, allocation: KVAllocation, first_token: int
    ) -> bool:
        """Adopt a request whose prompt KV was computed elsewhere (P-D handoff, m5 D5).

        The allocation must come from THIS scheduler's cache and cover the
        prompt. State is constructed directly: ``computed_prompt = prompt_len``
        bypasses the recompute-last-token rule (token 0 is adopted, never
        re-sampled) and non-empty ``outputs`` shields the request from
        recompute-preemption — both load-bearing invariants of the P-D design.
        Returns True if the request finished immediately (EOS / max_new_tokens).
        """
        if request.request_id in self._states:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        if allocation.tokens != request.prompt_token_ids:
            raise ValueError(
                f"allocation covers tokens {allocation.tokens!r}, "
                f"not this request's prompt"
            )
        state = _RequestState(request)
        state.status = _Status.RUNNING
        state.computed_prompt = state.prompt_len
        state.outputs.append(first_token)
        state.allocation = allocation
        self._states[request.request_id] = state
        self._running.append(request.request_id)
        is_eos = request.eos_token_id is not None and first_token == request.eos_token_id
        if is_eos or len(state.outputs) >= request.max_new_tokens:
            self._finish(state)
            return True
        return False

    def _ensure_decode_capacity(self, state: _RequestState) -> bool:
        needed_tokens = state.prompt_len + len(state.outputs) + state.in_flight + 1
        while state.capacity_tokens(self._page_size) < needed_tokens:
            try:
                state.decode_pages.append(self._kv.allocate_private_page())
            except KVCacheFull:
                return False
        return True

    def _schedule_decodes(self, budget: int, plan: list[ScheduledChunk]) -> int:
        for request_id in list(self._running):
            state = self._states[request_id]
            if state.status is not _Status.RUNNING:
                continue  # preempted earlier in this pass
            if not state.prefill_done or budget < 1:
                continue
            if len(state.outputs) + state.in_flight >= state.request.max_new_tokens:
                continue  # everything remaining is already in flight
            if not self._ensure_decode_capacity(state):
                # decode must not starve: recompute-preempt a prefilling victim
                if not self._preempt_for_decode(request_id):
                    continue
                if not self._ensure_decode_capacity(state):
                    continue  # still no space; retried next step
            plan.append(
                ScheduledChunk(
                    request_id=request_id,
                    num_tokens=1,
                    is_prefill=False,
                    position=len(state.outputs) + state.in_flight,
                )
            )
            state.in_flight += 1
            budget -= 1
        return budget

    def _schedule_prefills(self, budget: int, plan: list[ScheduledChunk]) -> int:
        for request_id in list(self._running):
            state = self._states[request_id]
            if state.status is not _Status.RUNNING or state.prefill_done or budget < 1:
                continue
            chunk = min(state.prompt_len - state.computed_prompt, budget)
            state.computed_prompt += chunk
            if state.prefill_done:
                state.in_flight += 1  # the prompt-completing chunk samples token 0
                if state.allocation is not None:
                    self._kv.mark_computed(state.allocation)
            plan.append(ScheduledChunk(request_id=request_id, num_tokens=chunk, is_prefill=True))
            budget -= chunk
        return budget

    def _admit_waiting(self, budget: int, plan: list[ScheduledChunk]) -> int:
        while self._waiting and budget > 0 and len(self._running) < self._max_seqs:
            request_id = self._waiting[0]
            state = self._states[request_id]
            if self._decode_watermark > 0:
                estimate = -(-state.prompt_len // self._page_size)  # ceil division
                if self._kv.num_free_pages < estimate + self._decode_watermark:
                    break  # keep pages in reserve so running decodes never starve
            try:
                state.allocation = self._kv.allocate(state.request.prompt_token_ids)
            except KVCacheFull:
                break  # head-of-line waits for pages; FIFO fairness
            self._waiting.pop(0)
            state.status = _Status.RUNNING
            self._running.append(request_id)
            # cached prefix skips prefill compute; the last prompt token is
            # always recomputed so the step produces logits for sampling
            cached = min(state.allocation.num_cached_tokens, state.prompt_len - 1)
            chunk = min(state.prompt_len - cached, budget)
            state.computed_prompt = cached + chunk
            if state.prefill_done:
                state.in_flight += 1  # prompt-completing chunk samples token 0
                self._kv.mark_computed(state.allocation)
            plan.append(ScheduledChunk(request_id=request_id, num_tokens=chunk, is_prefill=True))
            budget -= chunk
        return budget

    def schedule(self) -> SchedulerOutput:
        """Plan one step. With P-D separation, decodes and prefills draw from
        independent token budgets (TPOT and TTFT tuned separately, design m2 §2.4);
        combined mode shares one budget with decode priority."""
        plan: list[ScheduledChunk] = []
        if self._pd_separation:
            assert self._decode_budget is not None
            self._schedule_decodes(self._decode_budget, plan)
            prefill_budget = self._budget
        else:
            prefill_budget = self._schedule_decodes(self._budget, plan)
        prefill_budget = self._schedule_prefills(prefill_budget, plan)
        self._admit_waiting(prefill_budget, plan)
        return SchedulerOutput(scheduled=tuple(plan))

    def _finish(self, state: _RequestState) -> None:
        state.status = _Status.FINISHED
        self._running.remove(state.request.request_id)
        if state.allocation is not None:
            # fold fully-generated pages into the radix tree so the next
            # conversation turn's prompt (prompt + this completion) hits cache
            self._kv.commit_and_release(
                state.allocation, tuple(state.outputs), tuple(state.decode_pages)
            )
            state.decode_pages.clear()
        elif state.decode_pages:
            self._kv.free_private_pages(tuple(state.decode_pages))
            state.decode_pages.clear()

    @staticmethod
    def _normalize_tokens(request_id: str, tokens: int | Sequence[int]) -> list[int]:
        """Widen int-sugar to a list; reject non-int tokens loudly (m8 D2/D3) —
        an unconverted SampledToken would silently defeat the EOS comparison."""
        token_list = [tokens] if isinstance(tokens, int) else list(tokens)
        for token in token_list:
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError(
                    f"request {request_id!r}: sampled tokens must be ints, "
                    f"got {token!r} (convert StepOutput via engine_core.token_ids)"
                )
        return token_list

    def update(self, sampled: Mapping[str, int | Sequence[int]]) -> tuple[str, ...]:
        """Commit sampled tokens for prefill-complete requests; return finished ids.

        A bare int is single-token sugar (all pre-m8 call sites); a sequence
        commits in order with per-token terminal checks — tokens after a
        terminal token are discarded (speculative shortfall, m8 D3), never
        routed through the surplus-trim path.
        """
        finished: list[str] = []
        for request_id, tokens in sampled.items():
            token_list = self._normalize_tokens(request_id, tokens)
            state = self._states.get(request_id)
            if state is None:
                raise ValueError(f"request {request_id!r} is not awaiting a sampled token")
            finished_here = False
            for token_id in token_list:
                if state.status is not _Status.RUNNING:
                    if finished_here:
                        break  # terminal mid-list: discard the rest
                    if state.surplus_in_flight > 0:
                        # scheduled ahead under overlap, then the request finished
                        # (EOS) or was preempted/aborted: trim silently
                        state.surplus_in_flight -= 1
                        continue
                    raise ValueError(f"request {request_id!r} is not awaiting a sampled token")
                if not state.prefill_done or state.in_flight < 1:
                    raise ValueError(f"request {request_id!r} is not awaiting a sampled token")
                state.in_flight -= 1
                state.outputs.append(token_id)
                reason = self._terminal_reason(state, token_id)
                if reason is not None:
                    state.surplus_in_flight += state.in_flight
                    state.in_flight = 0
                    state.finish_reason = reason
                    self._finish(state)
                    finished.append(request_id)
                    finished_here = True
        return tuple(finished)

    @staticmethod
    def _terminal_reason(state: _RequestState, token_id: int) -> str | None:
        """Stop semantics per committed token (m8 D1): EOS/stop_token_ids gated
        by min_tokens and ignore_eos; max_new_tokens is the length backstop."""
        request = state.request
        is_eos = (
            not request.ignore_eos
            and request.eos_token_id is not None
            and token_id == request.eos_token_id
        )
        is_stop = is_eos or token_id in request.stop_token_ids
        if is_stop and len(state.outputs) >= request.min_tokens:
            return "stop"
        if len(state.outputs) >= request.max_new_tokens:
            return "length"
        return None
