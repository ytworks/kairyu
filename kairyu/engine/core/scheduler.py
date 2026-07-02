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

from dataclasses import dataclass
from enum import Enum

from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache

_DEFAULT_TOKEN_BUDGET = 2048
_DEFAULT_MAX_SEQS = 256


@dataclass(frozen=True)
class EngineRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int = 16


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
        "allocation",
        "decode_pages",
    )

    def __init__(self, request: EngineRequest) -> None:
        self.request = request
        self.status = _Status.WAITING
        self.computed_prompt = 0
        self.outputs: list[int] = []
        # sampled tokens scheduled but not yet committed via update() — this is
        # what lets the overlap loop plan step N+1 before step N's tokens land
        self.in_flight = 0
        self.allocation: KVAllocation | None = None
        self.decode_pages: list[int] = []

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

    def output_tokens(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._states[request_id].outputs)

    def _ensure_decode_capacity(self, state: _RequestState) -> bool:
        needed_tokens = state.prompt_len + len(state.outputs) + state.in_flight + 1
        while state.capacity_tokens(self._page_size) < needed_tokens:
            try:
                state.decode_pages.append(self._kv.allocate_private_page())
            except KVCacheFull:
                return False
        return True

    def _schedule_decodes(self, budget: int, plan: list[ScheduledChunk]) -> int:
        for request_id in self._running:
            state = self._states[request_id]
            if not state.prefill_done or budget < 1:
                continue
            if len(state.outputs) + state.in_flight >= state.request.max_new_tokens:
                continue  # everything remaining is already in flight
            if not self._ensure_decode_capacity(state):
                continue  # no KV space this step; retried next step
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
        for request_id in self._running:
            state = self._states[request_id]
            if state.prefill_done or budget < 1:
                continue
            chunk = min(state.prompt_len - state.computed_prompt, budget)
            state.computed_prompt += chunk
            if state.prefill_done:
                state.in_flight += 1  # the prompt-completing chunk samples token 0
            plan.append(ScheduledChunk(request_id=request_id, num_tokens=chunk, is_prefill=True))
            budget -= chunk
        return budget

    def _admit_waiting(self, budget: int, plan: list[ScheduledChunk]) -> int:
        while self._waiting and budget > 0 and len(self._running) < self._max_seqs:
            request_id = self._waiting[0]
            state = self._states[request_id]
            try:
                state.allocation = self._kv.allocate(state.request.prompt_token_ids)
            except KVCacheFull:
                break  # head-of-line waits for pages; FIFO fairness
            self._waiting.pop(0)
            state.status = _Status.RUNNING
            self._running.append(request_id)
            chunk = min(state.prompt_len, budget)
            state.computed_prompt = chunk
            if state.prefill_done:
                state.in_flight += 1  # single-chunk prefill samples token 0
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
            self._kv.free(state.allocation)
        if state.decode_pages:
            self._kv.free_private_pages(tuple(state.decode_pages))
            state.decode_pages.clear()

    def update(self, sampled: dict[str, int]) -> tuple[str, ...]:
        """Commit sampled tokens for prefill-complete requests; return finished ids."""
        finished: list[str] = []
        for request_id, token_id in sampled.items():
            state = self._states.get(request_id)
            if (
                state is None
                or state.status is not _Status.RUNNING
                or not state.prefill_done
                or state.in_flight < 1
            ):
                raise ValueError(f"request {request_id!r} is not awaiting a sampled token")
            state.in_flight -= 1
            state.outputs.append(token_id)
            if len(state.outputs) >= state.request.max_new_tokens:
                self._finish(state)
                finished.append(request_id)
        return tuple(finished)
