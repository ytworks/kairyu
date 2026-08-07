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

import heapq
import math
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum

from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache
from kairyu.engine.core.sampling_types import EngineSampling

_DEFAULT_TOKEN_BUDGET = 2048
_DEFAULT_MAX_SEQS = 256
_DEFAULT_MAX_PARTIAL_PREFILLS = 2
_MAX_WAITING_BYPASSES = 1


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
    # vLLM-compatible ordering: smaller integers are admitted first.
    priority: int = 0
    scheduling_class: str = "interactive"
    sampling: EngineSampling = field(default_factory=EngineSampling)
    # Who this request IS as far as sampling is concerned. Normally itself; the
    # P-D prefill clone sets it to the PUBLIC id, because ``request_id`` there is
    # a scheduler bookkeeping name (``r#p0``) and the sampler derives its base
    # seed from the id when ``seed is None`` — sampling token 0 under the clone's
    # name would put it on a different RNG stream from token 1 onward (m5 D5).
    sampling_id: str | None = None

    @property
    def sampling_identity(self) -> str:
        """The id every sampler must key this request's state under."""
        return self.sampling_id or self.request_id


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
        "output_epoch",
        "in_flight",
        "surplus_in_flight",
        "allocation",
        "decode_pages",
        "finish_reason",
        "spec_in_flight",
        "admission_bypasses",
    )

    def __init__(self, request: EngineRequest) -> None:
        self.request = request
        self.status = _Status.WAITING
        self.computed_prompt = 0
        self.outputs: list[int] = []
        # Committed outputs are append-only in production. Owners must bump
        # this epoch before replacing/truncating an existing prefix so
        # incremental consumers can rebuild only on that exceptional path.
        self.output_epoch = 0
        # sampled tokens scheduled but not yet committed via update() — this is
        # what lets the overlap loop plan step N+1 before step N's tokens land
        self.in_flight = 0
        # tokens still in flight when the request finished/preempted early
        # (EOS under overlap); their late arrivals are trimmed, not errors
        self.surplus_in_flight = 0
        self.allocation: KVAllocation | None = None
        self.decode_pages: list[int] = []
        self.finish_reason: str | None = None
        # reservation size of the one outstanding speculative chunk (m8 D3);
        # 0 when no spec chunk is in flight
        self.spec_in_flight = 0
        # Successful admissions allowed past this request during its current
        # waiting epoch.  Bounded skip-ahead is deliberately independent of
        # priority aging: the common time term cannot change pairwise rank.
        self.admission_bypasses = 0

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


@dataclass
class _PrefillBudget:
    """Work-conserving fair shares for one prefill cohort.

    A short request returns the unused part of its share to later requests.
    Once all initial shares are consumed, any remaining budget can only exist
    because earlier requests completed, so it is safe to give to a replacement
    without exceeding the configured partial-prefill width.
    """

    remaining: int
    shares: int
    public_priority: int | None = None
    cohort_request_ids: frozenset[str] = field(default_factory=frozenset)
    request_limits: dict[str, int] = field(default_factory=dict)
    protected_completions: frozenset[str] = field(default_factory=frozenset)
    persistent_prefills: int = 0
    completion_margin_pages: int = 0

    @property
    def next_chunk_limit(self) -> int:
        if self.remaining < 1:
            return 0
        divisor = max(1, self.shares)
        return max(1, -(-self.remaining // divisor))

    def limit_for(self, request_id: str) -> int:
        return min(
            self.next_chunk_limit,
            self.request_limits.get(request_id, self.remaining),
        )

    def take(self, requested: int, request_id: str | None = None) -> int:
        if requested < 1 or self.remaining < 1:
            return 0
        limit = self.next_chunk_limit if request_id is None else self.limit_for(request_id)
        chunk = min(requested, limit)
        self.remaining -= chunk
        self.shares = max(0, self.shares - 1)
        if request_id is not None:
            self.request_limits.pop(request_id, None)
        return chunk

    def drop_share(self, request_id: str | None = None) -> None:
        """Return one predicted participant that proved ineligible."""

        self.shares = max(0, self.shares - 1)
        if request_id is not None:
            self.request_limits.pop(request_id, None)
            if request_id in self.cohort_request_ids:
                self.request_limits.clear()
                self.protected_completions = frozenset()


class _IndexedWaitingQueue:
    """ID-indexed FIFO or stable priority queue for waiting requests.

    With aging enabled, every effective priority contains the same ``-now/age``
    term. Ordering can therefore use the immutable ``priority + arrival/age``
    component without repeatedly sorting the whole waiting set. Smaller values
    win, matching vLLM request-priority semantics. Heap removals
    are lazy and periodically compacted, making ID cancellation amortized O(1).
    """

    def __init__(self, priority_age_s: float | None) -> None:
        if priority_age_s is not None and (
            not math.isfinite(priority_age_s) or priority_age_s < 0
        ):
            raise ValueError("priority_age_s must be finite and >= 0 or null")
        self._priority_age_s = priority_age_s
        self._priority_age_ns = (
            None
            if priority_age_s is None
            else 0
            if priority_age_s == 0
            else max(1, round(priority_age_s * 1_000_000_000))
        )
        self._fifo: OrderedDict[str, None] = OrderedDict()
        self._heap: list[tuple[int, int, str]] = []
        self._priority_entries: dict[str, tuple[int, int]] = {}
        self._next_back_sequence = 0
        self._next_front_sequence = -1

    @property
    def priority_enabled(self) -> bool:
        return self._priority_age_s is not None

    def __bool__(self) -> bool:
        if self.priority_enabled:
            return bool(self._priority_entries)
        return bool(self._fifo)

    def __len__(self) -> int:
        if self.priority_enabled:
            return len(self._priority_entries)
        return len(self._fifo)

    def __iter__(self):
        if not self.priority_enabled:
            yield from self._fifo
            return
        ordered = sorted(
            (key, sequence, request_id)
            for request_id, (key, sequence) in self._priority_entries.items()
        )
        yield from (request_id for _key, _sequence, request_id in ordered)

    def peek_prefix(self, limit: int) -> tuple[str, ...]:
        """Return at most ``limit`` queue heads without sorting the live queue.

        FIFO mode walks only the requested prefix.  Priority mode performs a
        best-first traversal of the existing heap: a heap child's key cannot
        precede its parent, so the small frontier yields live entries in exactly
        the same order as ``popleft()`` without copying or sorting every waiter.
        Lazy-removal tombstones are skipped just as they are at the queue head.
        """

        if limit <= 0:
            return ()
        if not self.priority_enabled:
            prefix: list[str] = []
            for request_id in self._fifo:
                prefix.append(request_id)
                if len(prefix) >= limit:
                    break
            return tuple(prefix)

        self._drop_stale_head()
        if not self._heap:
            return ()
        # ``frontier`` stores (heap entry, source index).  Expanding only a
        # popped entry's children is the standard k-smallest traversal of a
        # binary heap and avoids the O(W log W) ``__iter__`` materialization.
        frontier: list[tuple[tuple[int, int, str], int]] = [
            (self._heap[0], 0)
        ]
        prefix = []
        while frontier and len(prefix) < limit:
            (key, sequence, request_id), index = heapq.heappop(frontier)
            if self._priority_entries.get(request_id) == (key, sequence):
                prefix.append(request_id)
            left = 2 * index + 1
            right = left + 1
            if left < len(self._heap):
                heapq.heappush(frontier, (self._heap[left], left))
            if right < len(self._heap):
                heapq.heappush(frontier, (self._heap[right], right))
        return tuple(prefix)

    def _priority_key(self, priority: int, arrival: float) -> int:
        """Exact signed-integer rank with nanosecond-resolution aging.

        Multiplying the priority by the aging interval avoids converting public
        signed-int64 values to float, which would collapse adjacent priorities
        outside the 53-bit mantissa. The common denominator does not affect
        ordering.
        """

        if not self._priority_age_ns:
            return priority
        return priority * self._priority_age_ns + round(arrival * 1_000_000_000)

    def priority_key(self, priority: int, arrival: float) -> int:
        """Immutable rank used to compare waiting and running prefills."""
        if not self.priority_enabled:
            raise RuntimeError(
                "priority key requested while priority scheduling is disabled"
            )
        return self._priority_key(priority, arrival)

    def append(
        self,
        request_id: str,
        *,
        priority: int,
        arrival: float,
        front: bool = False,
    ) -> None:
        if not self.priority_enabled:
            if request_id in self._fifo:
                raise ValueError(f"duplicate waiting request_id {request_id!r}")
            self._fifo[request_id] = None
            if front:
                self._fifo.move_to_end(request_id, last=False)
            return

        if request_id in self._priority_entries:
            raise ValueError(f"duplicate waiting request_id {request_id!r}")
        if front:
            sequence = self._next_front_sequence
            self._next_front_sequence -= 1
        else:
            sequence = self._next_back_sequence
            self._next_back_sequence += 1
        key = self._priority_key(priority, arrival)
        self._priority_entries[request_id] = (key, sequence)
        heapq.heappush(self._heap, (key, sequence, request_id))

    def extend(
        self, requests: Sequence[tuple[str, int, float]]
    ) -> None:
        for request_id, priority, arrival in requests:
            self.append(
                request_id,
                priority=priority,
                arrival=arrival,
            )

    def _drop_stale_head(self) -> None:
        while self._heap:
            key, sequence, request_id = self._heap[0]
            if self._priority_entries.get(request_id) == (key, sequence):
                return
            heapq.heappop(self._heap)

    def _maybe_compact(self) -> None:
        if len(self._heap) <= 64 or len(self._heap) <= 2 * len(
            self._priority_entries
        ):
            return
        self._heap = [
            (key, sequence, request_id)
            for request_id, (key, sequence) in self._priority_entries.items()
        ]
        heapq.heapify(self._heap)

    def peek(self) -> str:
        if not self.priority_enabled:
            if not self._fifo:
                raise IndexError("peek from empty waiting queue")
            return next(iter(self._fifo))
        self._drop_stale_head()
        if not self._heap:
            raise IndexError("peek from empty waiting queue")
        return self._heap[0][2]

    def popleft(self) -> str:
        if not self.priority_enabled:
            if not self._fifo:
                raise IndexError("pop from empty waiting queue")
            return self._fifo.popitem(last=False)[0]
        self._drop_stale_head()
        if not self._heap:
            raise IndexError("pop from empty waiting queue")
        key, sequence, request_id = heapq.heappop(self._heap)
        if self._priority_entries.get(request_id) != (key, sequence):
            raise AssertionError("waiting priority index diverged from heap")
        del self._priority_entries[request_id]
        return request_id

    def remove(self, request_id: str) -> None:
        if not self.priority_enabled:
            try:
                del self._fifo[request_id]
            except KeyError:
                raise ValueError(
                    f"{request_id!r} is not in waiting queue"
                ) from None
            return
        try:
            del self._priority_entries[request_id]
        except KeyError:
            raise ValueError(
                f"{request_id!r} is not in waiting queue"
            ) from None
        self._maybe_compact()


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
        speculative_tokens: int = 0,
        priority_age_s: float | None = None,
        clock=None,
        max_num_partial_prefills: int = _DEFAULT_MAX_PARTIAL_PREFILLS,
    ) -> None:
        if max_num_batched_tokens < 1:
            raise ValueError(f"max_num_batched_tokens must be >= 1, got {max_num_batched_tokens}")
        if pd_separation and (decode_token_budget is None or decode_token_budget < 1):
            raise ValueError("pd_separation=True requires decode_token_budget >= 1")
        if speculative_tokens < 0:
            raise ValueError(f"speculative_tokens must be >= 0, got {speculative_tokens}")
        if type(max_num_partial_prefills) is not int or max_num_partial_prefills < 1:
            raise ValueError(
                "max_num_partial_prefills must be a positive integer, "
                f"got {max_num_partial_prefills!r}"
            )
        self._kv = kv_cache
        self._budget = max_num_batched_tokens
        self._max_seqs = max_num_seqs
        self._pd_separation = pd_separation
        self._decode_budget = decode_token_budget
        self._decode_watermark = decode_watermark_pages
        # note: decode_watermark_pages was sized for +1 growth per step; with
        # speculative_tokens=k it should scale with k (m8 D3, documented knob)
        self._spec_k = speculative_tokens
        self._max_num_partial_prefills = max_num_partial_prefills
        self._prefer_prefill_completion = False
        # page_size param kept for back-compat; the cache is the source of truth
        self._page_size = getattr(kv_cache, "page_size", page_size)
        self._states: dict[str, _RequestState] = {}
        self._waiting = _IndexedWaitingQueue(priority_age_s)
        # m11 D6: priority admission — an old larger-number priority improves so
        # lower tiers cannot starve; EngineRequest is frozen, so aging is
        # factored into an immutable heap key from the arrival timestamp
        import time as _time

        self._clock = clock or _time.monotonic
        self._arrivals: dict[str, float] = {}
        self._running: list[str] = []
        self._priority_events: Counter[tuple[str, str]] = Counter(
            {
                (request_class, event): 0
                for request_class in ("interactive", "batch")
                for event in ("enqueue", "admit", "preempt", "complete")
            }
        )
        self._priority_queue_depth: Counter[str] = Counter(
            {"interactive": 0, "batch": 0}
        )
        self._priority_queue_high_watermark: Counter[str] = Counter(
            {"interactive": 0, "batch": 0}
        )
        # requests rejected at admission (C2), drained by the engine core/loop
        # so they surface as finished-with-empty-output like any terminal request
        self._rejected: list[str] = []

    def add_request(self, request: EngineRequest) -> None:
        if request.request_id in self._states:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        self._validate_scheduling_class(request.scheduling_class)
        self._states[request.request_id] = _RequestState(request)
        arrival = self._clock()
        self._arrivals[request.request_id] = arrival
        self._waiting.append(
            request.request_id,
            priority=request.priority,
            arrival=arrival,
        )
        self._record_enqueue(request.scheduling_class)

    def add_requests_atomic(self, requests: Sequence[EngineRequest]) -> None:
        """Admit one producer batch, or mutate nothing if any ID is invalid.

        EngineLoop uses this explicit atomic contract for consecutive adds.
        Scheduler adapters without an equivalent guarantee deliberately stay on
        the one-request path.
        """
        states: dict[str, _RequestState] = {}
        arrivals: dict[str, float] = {}
        request_ids: list[str] = []
        for request in requests:
            request_id = request.request_id
            if request_id in self._states or request_id in states:
                raise ValueError(f"duplicate request_id {request_id!r}")
            self._validate_scheduling_class(request.scheduling_class)
            request_ids.append(request_id)
            states[request_id] = _RequestState(request)
            arrivals[request_id] = self._clock()

        self._states.update(states)
        self._arrivals.update(arrivals)
        self._waiting.extend(
            [
                (
                    request_id,
                    states[request_id].request.priority,
                    arrivals[request_id],
                )
                for request_id in request_ids
            ]
        )
        for request in requests:
            self._record_enqueue(request.scheduling_class)

    @staticmethod
    def _validate_scheduling_class(request_class: str) -> None:
        if request_class not in {"interactive", "batch"}:
            raise ValueError(
                "scheduling_class must be 'interactive' or 'batch', "
                f"got {request_class!r}"
            )

    def _record_enqueue(self, request_class: str) -> None:
        self._priority_events[(request_class, "enqueue")] += 1
        self._priority_queue_depth[request_class] += 1
        self._priority_queue_high_watermark[request_class] = max(
            self._priority_queue_high_watermark[request_class],
            self._priority_queue_depth[request_class],
        )

    def _record_waiting_exit(self, state: _RequestState) -> None:
        self._priority_queue_depth[state.request.scheduling_class] -= 1

    def priority_metrics_snapshot(self) -> dict[str, dict]:
        """Bounded queue/admission evidence for scrape-time collectors."""

        return {
            "events": dict(self._priority_events),
            "queue_depth": {
                request_class: self._priority_queue_depth[request_class]
                for request_class in ("interactive", "batch")
            },
            "queue_high_watermark": {
                request_class: self._priority_queue_high_watermark[request_class]
                for request_class in ("interactive", "batch")
            },
        }

    def has_unfinished(self) -> bool:
        return bool(self._waiting or self._running)

    def has_prefill_work(self) -> bool:
        """Whether admission or an unfinished prompt can change the next plan.

        Waiting requests count even when their prompt is fully radix-cached: the
        scheduler still admits them through one prompt-completing prefill chunk.
        This read-only signal lets ``EngineLoop`` bound immutable schedule-ahead
        without weakening the configured depth once every live request decodes.
        """

        return bool(self._waiting) or any(
            not self._states[request_id].prefill_done
            for request_id in self._running
        )

    def enable_deferred_handoff_overlap(self) -> None:
        """Prefer one prompt completion per cohort for deferred P-D copies."""

        self._prefer_prefill_completion = True

    @property
    def states(self) -> dict[str, _RequestState]:
        """Read view of request states for the ModelRunner (do not mutate)."""
        return self._states

    @property
    def kv_cache(self) -> RadixKVCache:
        """The cache this scheduler admits against — the one a caller assembling
        several schedulers (P-D) must report rather than a freshly built one."""
        return self._kv

    @property
    def speculative_tokens(self) -> int:
        """Configured draft length; used by the loop to place commit barriers."""
        return self._spec_k

    @property
    def waiting_ids(self) -> tuple[str, ...]:
        return tuple(self._waiting)

    def _waiting_compute_and_new_pages(self, state: _RequestState) -> tuple[int, int]:
        """Return compute tokens and newly-owned pages for a waiting prompt."""

        cached = self._kv.peek_cached_tokens(state.request.prompt_token_ids)
        compute_tokens = state.prompt_len - min(cached, state.prompt_len - 1)
        uncached_tokens = state.prompt_len - cached
        new_pages = -(-uncached_tokens // self._page_size)
        return compute_tokens, new_pages

    def _valid_waiting_prefill(self, request_id: str) -> tuple[int, int] | None:
        state = self._states[request_id]
        if state.prompt_len < 1:
            return None
        required_pages = -(-state.prompt_len // self._page_size)
        if required_pages > self._kv.num_pages:
            return None
        return self._waiting_compute_and_new_pages(state)

    def _reclaimable_prefill_pages(self, request_id: str) -> int:
        """Pages a recompute-preempted incomplete prompt makes allocatable."""

        state = self._states[request_id]
        allocation = state.allocation
        if allocation is None:
            return len(state.decode_pages)
        return (
            len(allocation.new_full_pages)
            + (allocation.tail_page is not None)
            + len(state.decode_pages)
        )

    def _decode_liability_pages(
        self,
        state: _RequestState,
        allocation: KVAllocation | None = None,
    ) -> int:
        target_pages = -(-(state.prompt_len + state.request.max_new_tokens) // self._page_size)
        selected = allocation if allocation is not None else state.allocation
        allocation_pages = len(selected.pages) if selected is not None else 0
        owned_pages = allocation_pages + len(state.decode_pages)
        return max(0, target_pages - owned_pages)

    def _can_complete_prefill(
        self,
        request_id: str,
        budget: _PrefillBudget,
        allocation: KVAllocation | None = None,
    ) -> bool:
        """Keep an escape victim until every decode can retain capacity."""

        candidate = self._states[request_id]
        if candidate.request.max_new_tokens <= 1:
            return True
        if budget.persistent_prefills < 1:
            # The first decode leader can reclaim every incomplete prefill.
            return True
        return (
            self._prefill_completion_cost(candidate, allocation) <= budget.completion_margin_pages
        )

    def _prefill_completion_cost(
        self,
        state: _RequestState,
        allocation: KVAllocation | None = None,
    ) -> int:
        selected = allocation if allocation is not None else state.allocation
        reclaimable = len(state.decode_pages)
        if selected is not None:
            reclaimable += len(selected.new_full_pages)
            reclaimable += selected.tail_page is not None
        liability = (
            self._decode_liability_pages(state, selected) if state.request.max_new_tokens > 1 else 0
        )
        return reclaimable + liability

    def _record_prefill_completion(
        self,
        state: _RequestState,
        budget: _PrefillBudget,
    ) -> None:
        budget.completion_margin_pages -= self._prefill_completion_cost(state)
        if state.request.max_new_tokens <= 1:
            return
        budget.persistent_prefills += 1

    def _simple_prefill_cohort(
        self,
        running_prefills: Sequence[str],
        remaining: int,
        available_prefills: int,
    ) -> tuple[str, ...]:
        """Select a bounded leading tier without simulating KV admission."""

        cap = min(
            self._max_num_partial_prefills,
            remaining,
            available_prefills,
        )
        if cap < 1:
            return ()
        waiters = self._waiting.peek_prefix(min(len(self._waiting), cap))
        candidates = tuple(running_prefills) + waiters
        if not candidates:
            return ()
        if not self._waiting.priority_enabled:
            return candidates[:cap]
        indexed = list(enumerate(candidates))
        indexed.sort(
            key=lambda item: (
                self._priority_key_for(item[1]),
                item[0],
            )
        )
        winning_priority = self._states[indexed[0][1]].request.priority
        cohort: list[str] = []
        for _position, request_id in indexed:
            if self._states[request_id].request.priority != winning_priority:
                break
            cohort.append(request_id)
            if len(cohort) >= cap:
                break
        return tuple(cohort)

    def _prefill_budget(self, remaining: int) -> _PrefillBudget:
        running_prefills: list[str] = []
        persistent_prefills = 0
        completion_margin_pages = self._kv.num_free_pages
        for request_id in self._running:
            state = self._states[request_id]
            if state.status is not _Status.RUNNING:
                continue
            if not state.prefill_done:
                running_prefills.append(request_id)
                completion_margin_pages += self._reclaimable_prefill_pages(request_id)
            elif state.request.max_new_tokens > 1:
                persistent_prefills += 1
                completion_margin_pages -= self._decode_liability_pages(state)
        cohort = self._simple_prefill_cohort(
            running_prefills,
            remaining,
            len(running_prefills)
            + max(0, self._max_seqs - len(self._running)),
        )
        shares = max(1, len(cohort))
        request_limits, protected_completions = self._deferred_handoff_limits(cohort, remaining)
        return _PrefillBudget(
            remaining=remaining,
            shares=shares,
            public_priority=(
                self._states[cohort[0]].request.priority
                if self._waiting.priority_enabled and cohort
                else None
            ),
            cohort_request_ids=frozenset(cohort),
            request_limits=request_limits,
            protected_completions=protected_completions,
            persistent_prefills=persistent_prefills,
            completion_margin_pages=completion_margin_pages,
        )

    def _deferred_handoff_limits(
        self, cohort: Sequence[str], remaining: int
    ) -> tuple[dict[str, int], frozenset[str]]:
        """Finish one prompt while retaining work for a deferred-copy overlap."""

        if not self._prefer_prefill_completion or len(cohort) < 2:
            return {}, frozenset()
        prompt_remaining: dict[str, int] = {}
        for request_id in cohort:
            state = self._states[request_id]
            if state.status is _Status.WAITING:
                estimate = self._valid_waiting_prefill(request_id)
                if estimate is None:
                    return {}, frozenset()
                prompt_remaining[request_id] = estimate[0]
            else:
                prompt_remaining[request_id] = state.prompt_len - state.computed_prompt
        completion_budget = remaining - (len(cohort) - 1)
        eligible = [
            request_id for request_id in cohort if prompt_remaining[request_id] <= completion_budget
        ]
        if not eligible:
            return {}, frozenset()
        completion_id = eligible[0]
        limits = {completion_id: prompt_remaining[completion_id]}
        residual = remaining - limits[completion_id]
        shares = len(cohort) - 1
        for request_id in cohort:
            if request_id == completion_id:
                continue
            fair_limit = max(1, -(-residual // shares))
            limit = min(
                fair_limit,
                max(0, prompt_remaining[request_id] - 1),
            )
            limits[request_id] = limit
            residual -= limit
            shares -= 1
        protected = frozenset(request_id for request_id in cohort if request_id != completion_id)
        return limits, protected

    def _terminal_releasable_pages(self, state: _RequestState) -> int:
        """Lower bound on raw pages released by a terminal commit."""

        allocation = state.allocation
        if allocation is None:
            return len(state.decode_pages)
        private_pages = (allocation.tail_page is not None) + len(state.decode_pages)
        written_len = state.prompt_len + len(state.outputs) + state.in_flight - 1
        committed_pages = max(
            0,
            written_len // self._page_size - state.prompt_len // self._page_size,
        )
        return max(0, private_pages - committed_pages)

    def should_drain_before_admission(self) -> bool:
        """Whether committing terminal in-flight work can form a fuller cohort.

        Schedule-ahead can leave a partially free sequence window while every
        remaining runner has already reserved its final token. Admitting into
        those few slots immediately fragments the next prefill cohort even
        though the pending device handles are guaranteed to release the rest.
        The engine loop may therefore commit first and re-evaluate admission.

        This is deliberately a read-only predicate. It never reorders the
        waiting queue, never delays an outranking request, and is disabled for
        speculative chunks whose accepted length is not known until commit.
        It drains only when the prefill token budget can admit more requests
        after the terminal slots are released than it can admit right now.
        """

        if self._spec_k or not self._waiting or not self._running:
            return False
        free_slots = self._max_seqs - len(self._running)
        if free_slots <= 0 or len(self._waiting) <= free_slots:
            return False

        running_states = [self._states[request_id] for request_id in self._running]
        if not all(
            state.status is _Status.RUNNING
            and state.prefill_done
            and state.in_flight > 0
            and len(state.outputs) + state.in_flight
            >= state.request.max_new_tokens
            for state in running_states
        ):
            return False

        if self._waiting.priority_enabled:
            waiting_key = self._priority_key_for(self._waiting.peek())
            if any(
                waiting_key < self._priority_key_for(request_id)
                for request_id in self._running
            ):
                return False

        # One admitted prefill always spends at least one token because the last
        # prompt token is recomputed even on a complete radix hit.  Therefore no
        # plan can inspect/admit more than min(max_seqs, token_budget) waiters.
        # ``peek_prefix`` preserves priority/FIFO order without sorting a large
        # priority queue merely to inspect this bounded cohort.
        candidate_limit = min(
            len(self._waiting),
            self._max_seqs,
            self._budget,
        )
        candidates = self._waiting.peek_prefix(candidate_limit)
        head_estimate = self._valid_waiting_prefill(candidates[0])
        if head_estimate is None:
            return False
        if (
            head_estimate[1] + self._decode_watermark
            > self._kv.num_free_pages
        ):
            # Preserve an immediate bounded skip instead of draining first.
            return False

        cohort = self._simple_prefill_cohort(
            (),
            self._budget,
            self._max_seqs,
        )
        public_priority = (
            self._states[cohort[0]].request.priority
            if self._waiting.priority_enabled and cohort
            else None
        )
        budget = _PrefillBudget(
            remaining=self._budget,
            shares=max(1, len(cohort)),
            public_priority=public_priority,
        )
        pages = self._kv.num_free_pages + sum(
            self._terminal_releasable_pages(state) for state in running_states
        )
        completion_margin = pages
        persistent_prefills = 0
        admitted = 0
        incomplete_winning = False
        for request_id in candidates:
            state = self._states[request_id]
            prompt_len = state.prompt_len
            if prompt_len < 1:
                # Let normal schedule() reject an invalid head immediately.
                return False
            required_pages = -(-prompt_len // self._page_size)
            if required_pages > self._kv.num_pages:
                # Likewise, do not delay rejection of an impossible prompt.
                return False
            cached_tokens = self._kv.peek_cached_tokens(
                state.request.prompt_token_ids
            )
            # Requests that fit the currently free window retain their existing
            # immediate cached-prefix admission.  A cached request deeper than
            # that window is not admissible until a terminal slot is released,
            # so draining does not postpone it.
            if admitted < free_slots and cached_tokens > 0:
                return False
            compute_tokens, new_pages = self._waiting_compute_and_new_pages(state)
            watermark = self._decode_watermark if admitted else 0
            if new_pages + watermark > pages:
                break
            if budget.public_priority is not None:
                if state.request.priority != budget.public_priority:
                    if incomplete_winning:
                        break
                    budget.public_priority = state.request.priority
            requested = compute_tokens
            if compute_tokens <= budget.limit_for(request_id):
                owned_pages = -(-prompt_len // self._page_size)
                target_pages = -(
                    -(prompt_len + state.request.max_new_tokens)
                    // self._page_size
                )
                liability = max(0, target_pages - owned_pages)
                completion_cost = new_pages + (
                    liability if state.request.max_new_tokens > 1 else 0
                )
                safe_completion = (
                    state.request.max_new_tokens <= 1
                    or persistent_prefills < 1
                    or completion_cost <= completion_margin
                )
                if not safe_completion:
                    if compute_tokens == 1:
                        break
                    requested -= 1
                else:
                    completion_margin -= completion_cost
                    persistent_prefills += state.request.max_new_tokens > 1
            chunk = budget.take(requested, request_id)
            if chunk < 1:
                break
            pages -= new_pages
            admitted += 1
            if admitted > free_slots:
                return True
            incomplete_winning |= (
                state.request.priority == budget.public_priority
                and chunk < compute_tokens
            )
            if budget.remaining <= 0:
                break
        return False

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
            self._record_waiting_exit(state)
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
            self._record_waiting_exit(state)
            state.status = _Status.FINISHED
            state.finish_reason = "stop"
            return
        state.surplus_in_flight += state.in_flight
        state.in_flight = 0
        state.finish_reason = "stop"
        self._finish(state)

    def _reject_unadmittable(self, request_id: str, state: _RequestState) -> None:
        """Finish a waiting request that can never be admitted (C2).

        Removes it from the waiting queue and marks it FINISHED with reason
        ``length`` (no output was produced). The engine loop surfaces it as a
        finished stream update like any other terminal request, so an oversized
        prompt degrades to a graceful empty completion instead of a stall.
        """
        self._waiting.remove(request_id)
        self._record_waiting_exit(state)
        state.status = _Status.FINISHED
        state.finish_reason = "length"
        self._rejected.append(request_id)

    def drain_rejected(self) -> tuple[str, ...]:
        """Return and clear requests rejected at admission since the last call."""
        rejected = tuple(self._rejected)
        self._rejected.clear()
        return rejected

    def forget(self, request_id: str) -> None:
        """Drop all per-request state once a finished request is fully consumed.

        Long-running engines (the server, kairyu-proc) must reclaim finished
        ``_RequestState`` objects — each holds the full output token list — and
        their arrival timestamps, or memory grows without bound (E2). Called by
        the engine loop after it emits the terminal update. Only forgets
        already-finished requests; a live id is left untouched.
        """
        state = self._states.get(request_id)
        if state is not None and state.status is not _Status.FINISHED:
            return
        self._states.pop(request_id, None)
        self._arrivals.pop(request_id, None)

    def reject_waiting_head(self) -> str | None:
        """Force the current waiting head to finish (engine-loop stall backstop).

        Called only when a step produced an empty schedule while requests are
        unfinished — which implies nothing is running to free pages, so the head
        is genuinely stuck. Returns the rejected id, or None if nothing waits.
        """
        if not self._waiting:
            return None
        request_id = self._waiting.peek()
        self._reject_unadmittable(request_id, self._states[request_id])
        return request_id

    def finish_reason(self, request_id: str) -> str | None:
        return self._states[request_id].finish_reason

    def num_cached_tokens(self, request_id: str) -> int:
        """Prompt tokens served from the radix cache (usage truth, m8 D6/M9)."""
        state = self._states[request_id]
        if state.allocation is None:
            return 0
        return state.allocation.num_cached_tokens

    def _preempt_for_decode(self, needy_id: str) -> bool:
        """Recompute-preempt a young output-free request that releases pages.

        Victims are restricted to requests with no committed outputs so
        requeueing them recomputes the prompt only (output-KV recompute is a
        GPU-phase extension, see design m2 §5). Cached-only candidates are a
        fallback because unlocking their shared pages may release nothing.
        """
        fallback_id: str | None = None
        for victim_id in reversed(self._running):
            state = self._states[victim_id]
            if victim_id == needy_id or state.outputs or state.in_flight:
                continue
            if self._reclaimable_prefill_pages(victim_id) > 0:
                self._requeue_preempted(victim_id, state)
                return True
            if fallback_id is None:
                fallback_id = victim_id
        if fallback_id is not None:
            self._requeue_preempted(
                fallback_id,
                self._states[fallback_id],
            )
            return True
        return False

    def _requeue_preempted(self, request_id: str, state: _RequestState) -> None:
        """Release output-free running work and restore its original queue rank."""
        self._running.remove(request_id)
        self._release_without_commit(state)
        state.computed_prompt = 0
        state.admission_bypasses = 0
        state.status = _Status.WAITING
        self._waiting.append(
            request_id,
            priority=state.request.priority,
            arrival=self._arrivals[request_id],
            front=True,
        )
        request_class = state.request.scheduling_class
        self._priority_events[(request_class, "preempt")] += 1
        self._priority_queue_depth[request_class] += 1
        self._priority_queue_high_watermark[request_class] = max(
            self._priority_queue_high_watermark[request_class],
            self._priority_queue_depth[request_class],
        )

    def _priority_key_for(self, request_id: str) -> int:
        state = self._states[request_id]
        return self._waiting.priority_key(
            state.request.priority,
            self._arrivals[request_id],
        )

    def _lower_priority_prefill_victim(self, waiting_id: str) -> str | None:
        """Lowest-priority output-free running prefill outranked by ``waiting_id``.

        A waiting interactive request may need both a sequence slot and KV pages
        held by batch prefill work. Recompute-preempting an output-free prefill is
        safe; output-bearing decode requests remain protected by the existing
        decode-priority invariant.
        """
        if not self._waiting.priority_enabled:
            return None
        waiting_key = self._priority_key_for(waiting_id)
        candidates: list[tuple[int, float, int, str]] = []
        for position, request_id in enumerate(self._running):
            state = self._states[request_id]
            if (
                state.status is not _Status.RUNNING
                or state.prefill_done
                or state.outputs
            ):
                continue
            key = self._priority_key_for(request_id)
            if waiting_key < key:
                candidates.append(
                    (key, self._arrivals[request_id], position, request_id)
                )
        if not candidates:
            return None
        # Larger key is lower priority. For equal rank, preempt the younger/later
        # running request, matching the existing youngest-victim discipline.
        return max(candidates)[-1]

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
        # honor ignore_eos / stop_token_ids / min_tokens / finish_reason exactly
        # like the normal decode path, not a bare EOS-or-length check
        reason = self._terminal_reason(state, first_token)
        if reason is not None:
            state.finish_reason = reason
            self._finish(state)
            return True
        return False

    def _ensure_decode_capacity(self, state: _RequestState, extra: int = 1) -> bool:
        needed_tokens = state.prompt_len + len(state.outputs) + state.in_flight + extra
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
            remaining = state.request.max_new_tokens - len(state.outputs) - state.in_flight
            if remaining < 1:
                continue  # everything remaining is already in flight
            # speculative chunks require in_flight == 0 (scheduler-enforced,
            # m8 D3): under overlap-ahead planning a second chunk's position
            # would assume full commit of the first, which rejection breaks
            if self._spec_k > 0 and state.in_flight == 0:
                reserve = min(self._spec_k + 1, remaining, budget)
            else:
                reserve = 1
            if not self._ensure_decode_capacity(state, reserve):
                # decode must not starve: recompute-preempt a prefilling victim
                if not self._preempt_for_decode(request_id) or not self._ensure_decode_capacity(
                    state, reserve
                ):
                    # degrade to a plain 1-token reservation: k+1 must never
                    # stall a workload that completes under k=0 (m8 D3)
                    reserve = 1
                    if not self._ensure_decode_capacity(state, reserve):
                        continue  # still no space; retried next step
            plan.append(
                ScheduledChunk(
                    request_id=request_id,
                    num_tokens=reserve,
                    is_prefill=False,
                    position=len(state.outputs) + state.in_flight,
                )
            )
            state.in_flight += reserve
            if self._spec_k > 0 and reserve > 0 and state.in_flight == reserve:
                state.spec_in_flight = reserve
            budget -= reserve
        return budget

    def _schedule_prefills(self, budget: _PrefillBudget, plan: list[ScheduledChunk]) -> None:
        already_scheduled = {chunk.request_id for chunk in plan if chunk.is_prefill}
        request_ids = [
            request_id
            for request_id in self._running
            if self._states[request_id].status is _Status.RUNNING
            and not self._states[request_id].prefill_done
            and request_id not in already_scheduled
        ]
        if self._waiting.priority_enabled:
            stable_position = {
                request_id: position
                for position, request_id in enumerate(request_ids)
            }
            request_ids.sort(
                key=lambda request_id: (
                    self._priority_key_for(request_id),
                    stable_position[request_id],
                )
            )
        for request_id in request_ids:
            state = self._states[request_id]
            if budget.remaining < 1:
                break
            if budget.public_priority is not None:
                if state.request.priority != budget.public_priority:
                    if self._has_incomplete_cohort_prefill(budget):
                        break
                    budget.public_priority = state.request.priority
                    budget.cohort_request_ids = frozenset()
            if budget.limit_for(request_id) < 1:
                budget.drop_share(request_id)
                continue
            remaining_prompt = state.prompt_len - state.computed_prompt
            requested = remaining_prompt
            if remaining_prompt <= budget.limit_for(request_id) and not self._can_complete_prefill(
                request_id, budget
            ):
                requested -= 1
            if requested < 1:
                budget.drop_share(request_id)
                continue
            chunk = budget.take(requested, request_id)
            state.computed_prompt += chunk
            if state.prefill_done:
                state.in_flight += 1  # the prompt-completing chunk samples token 0
                self._record_prefill_completion(state, budget)
            elif budget.public_priority is not None:
                budget.cohort_request_ids |= frozenset((request_id,))
            plan.append(ScheduledChunk(request_id=request_id, num_tokens=chunk, is_prefill=True))

    def _has_incomplete_cohort_prefill(self, budget: _PrefillBudget) -> bool:
        if budget.public_priority is None:
            return False
        live = frozenset(
            request_id
            for request_id in budget.cohort_request_ids
            if self._states[request_id].status is _Status.RUNNING
            and not self._states[request_id].prefill_done
        )
        budget.cohort_request_ids = live
        return bool(live)

    def _redistribute_prefill_remainder(
        self, budget: _PrefillBudget, plan: list[ScheduledChunk]
    ) -> None:
        """Return an unused trailing share to an existing incomplete chunk."""

        if budget.remaining < 1:
            return
        for index, scheduled in enumerate(plan):
            if not scheduled.is_prefill:
                continue
            state = self._states[scheduled.request_id]
            remaining_prompt = state.prompt_len - state.computed_prompt
            if remaining_prompt < 1:
                continue
            extra = min(remaining_prompt, budget.remaining)
            if scheduled.request_id in budget.protected_completions and extra >= remaining_prompt:
                extra -= 1
            if extra >= remaining_prompt and not self._can_complete_prefill(
                scheduled.request_id,
                budget,
            ):
                extra -= 1
            if extra < 1:
                continue
            state.computed_prompt += extra
            budget.remaining -= extra
            if state.prefill_done:
                state.in_flight += 1
                self._record_prefill_completion(state, budget)
            plan[index] = ScheduledChunk(
                request_id=scheduled.request_id,
                num_tokens=scheduled.num_tokens + extra,
                is_prefill=True,
                position=scheduled.position,
            )
            if budget.remaining < 1:
                break

    def _admit_priority_ahead_of_prefills(
        self, budget: _PrefillBudget, plan: list[ScheduledChunk]
    ) -> None:
        """Run an outranking waiter before lower-priority running prefill work.

        Existing decode work has already consumed its budget when this runs.
        If a lower-priority, output-free prefill owns the last sequence slot or
        needed KV pages, recompute-preempt it and retry the higher-priority head.
        """
        if not self._waiting.priority_enabled:
            return
        while self._waiting and budget.remaining > 0:
            request_id = self._waiting.peek()
            victim_id = self._lower_priority_prefill_victim(request_id)
            if victim_id is None:
                break

            state = self._states[request_id]
            required_pages = -(-state.prompt_len // self._page_size)
            if state.prompt_len == 0 or required_pages > self._kv.num_pages:
                # A full sequence window prevents the ordinary admission loop
                # from entering at all. Reject the already-validated impossible
                # head directly so priority-ahead cannot spin forever.
                self._reject_unadmittable(request_id, state)
                continue
            if len(self._running) >= self._max_seqs:
                self._requeue_preempted(victim_id, self._states[victim_id])

            self._admit_waiting(
                budget,
                plan,
                max_admissions=1,
                allow_skip_ahead=False,
            )
            if state.status is _Status.RUNNING:
                continue
            if request_id not in self._states or state.status is _Status.FINISHED:
                # Empty/oversized head was rejected; reevaluate the new head.
                continue

            # Allocation can still fail despite the page-count estimate (for
            # example, pinned radix pages). Release another outranked prefill.
            victim_id = self._lower_priority_prefill_victim(request_id)
            if victim_id is None:
                break
            self._requeue_preempted(victim_id, self._states[victim_id])
            # Retry immediately after releasing pages.  Waiting until the next
            # loop iteration would first require another eligible victim and
            # could therefore skip the only useful post-release allocation.
            self._admit_waiting(
                budget,
                plan,
                max_admissions=1,
                allow_skip_ahead=False,
            )

    def _admit_waiting(
        self,
        budget: _PrefillBudget,
        plan: list[ScheduledChunk],
        *,
        max_admissions: int | None = None,
        allow_skip_ahead: bool = True,
    ) -> None:
        admitted = 0
        while (
            self._waiting
            and budget.remaining > 0
            and len(self._running) < self._max_seqs
            and (max_admissions is None or admitted < max_admissions)
        ):
            head_id = self._waiting.peek()
            head = self._states[head_id]
            if head.prompt_len == 0:
                self._reject_unadmittable(head_id, head)
                continue
            required_pages = -(-head.prompt_len // self._page_size)
            if required_pages > self._kv.num_pages:
                # This prompt is larger than the whole KV cache: it can NEVER be
                # admitted no matter how much drains. Reject it instead of
                # blocking the head of line forever (C2 — an un-rejectable head
                # turns every subsequent step's empty schedule into a fatal
                # engine stall that kills all concurrent requests).
                self._reject_unadmittable(head_id, head)
                continue
            if budget.public_priority is not None:
                if head.request.priority != budget.public_priority:
                    if self._has_incomplete_cohort_prefill(budget):
                        break
                    budget.public_priority = head.request.priority
                    budget.cohort_request_ids = frozenset()
            if budget.limit_for(head_id) < 1:
                break

            allocation = None
            allocation_free_before = self._kv.num_free_pages
            head_blocked = False
            pre_attempt_free = self._kv.num_free_pages
            _head_compute, head_new_pages = self._waiting_compute_and_new_pages(head)
            watermark = self._decode_watermark if self._running else 0
            successor_before_attempt: tuple[str, int, int] | None = None
            if self._running and pre_attempt_free < head_new_pages + watermark:
                candidates = self._waiting.peek_prefix(_MAX_WAITING_BYPASSES + 1)
                if len(candidates) >= 2:
                    successor_id = candidates[1]
                    successor = self._valid_waiting_prefill(successor_id)
                    if successor is not None:
                        successor_before_attempt = (
                            successor_id,
                            successor[0],
                            successor[1],
                        )
            if self._decode_watermark > 0 and self._running:
                # reserve pages for running decodes only — with nothing running
                # there is no decode to protect, so reserving would deadlock the
                # sole waiting request forever (C2)
                head_blocked = pre_attempt_free < head_new_pages + self._decode_watermark
            if not head_blocked:
                try:
                    allocation = self._kv.allocate(head.request.prompt_token_ids)
                except KVCacheFull:
                    # A failed DRAM restore may still consume destination pages.
                    budget.completion_margin_pages += (
                        self._kv.num_free_pages - allocation_free_before
                    )
                    allocation_free_before = self._kv.num_free_pages
                    if not self._running:
                        # Nothing is running to free pages and the cache still
                        # cannot fit this prompt (for example, pinned sessions
                        # own every page), so waiting is futile.
                        self._reject_unadmittable(head_id, head)
                        continue
                    head_blocked = True

            request_id = head_id
            state = head
            bypassed = False
            if head_blocked:
                if allow_skip_ahead:
                    budget.drop_share(head_id)
                if (
                    not allow_skip_ahead
                    or head.admission_bypasses >= _MAX_WAITING_BYPASSES
                    or successor_before_attempt is None
                ):
                    break
                request_id, prior_compute, prior_new_pages = successor_before_attempt
                state = self._states[request_id]
                if budget.public_priority is not None:
                    if state.request.priority != budget.public_priority:
                        if self._has_incomplete_cohort_prefill(budget):
                            break
                        budget.public_priority = state.request.priority
                        budget.cohort_request_ids = frozenset()
                compute_tokens, new_pages = self._waiting_compute_and_new_pages(state)
                if (
                    prior_compute > budget.limit_for(request_id)
                    or pre_attempt_free < prior_new_pages + watermark
                    or compute_tokens > budget.limit_for(request_id)
                    or self._kv.num_free_pages < new_pages + watermark
                ):
                    break
                allocation_free_before = self._kv.num_free_pages
                try:
                    allocation = self._kv.allocate(state.request.prompt_token_ids)
                except KVCacheFull:
                    budget.completion_margin_pages += (
                        self._kv.num_free_pages - allocation_free_before
                    )
                    break
                bypassed = True

            assert allocation is not None
            private_pages = len(allocation.new_full_pages) + (
                allocation.tail_page is not None
            )
            # Cold allocation converts raw pages into reclaimable private pages;
            # radix eviction adds margin, while a DRAM restore consumes it.
            allocation_gain = private_pages - (
                allocation_free_before - self._kv.num_free_pages
            )
            budget.completion_margin_pages += allocation_gain
            cached = min(
                allocation.num_cached_tokens,
                state.prompt_len - 1,
            )
            compute_tokens = state.prompt_len - cached
            can_finish = True
            if compute_tokens <= budget.limit_for(request_id):
                can_finish = self._can_complete_prefill(
                    request_id,
                    budget,
                    allocation,
                )
            if bypassed:
                if compute_tokens > budget.limit_for(request_id) or not can_finish:
                    self._kv.release_preempted(allocation, ())
                    break
            elif compute_tokens == 1 and not can_finish:
                self._kv.release_preempted(allocation, ())
                break
            if bypassed:
                self._waiting.remove(request_id)
                head.admission_bypasses += 1
            else:
                popped_id = self._waiting.popleft()
                if popped_id != request_id:
                    raise AssertionError("waiting queue head changed during admission")
            state.allocation = allocation
            self._record_waiting_exit(state)
            state.status = _Status.RUNNING
            state.admission_bypasses = 0
            self._running.append(request_id)
            # cached prefix skips prefill compute; the last prompt token is
            # always recomputed so the step produces logits for sampling
            state.computed_prompt = cached
            requested = compute_tokens
            if compute_tokens <= budget.limit_for(request_id) and not can_finish:
                requested -= 1
            assert requested > 0
            chunk = budget.take(requested, request_id)
            state.computed_prompt = cached + chunk
            if state.prefill_done:
                state.in_flight += 1  # prompt-completing chunk samples token 0
                self._record_prefill_completion(state, budget)
            elif budget.public_priority is not None:
                budget.cohort_request_ids |= frozenset((request_id,))
            plan.append(ScheduledChunk(request_id=request_id, num_tokens=chunk, is_prefill=True))
            self._priority_events[(state.request.scheduling_class, "admit")] += 1
            admitted += 1
            if bypassed:
                break

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
        budget = self._prefill_budget(prefill_budget)
        self._admit_priority_ahead_of_prefills(budget, plan)
        self._schedule_prefills(budget, plan)
        self._admit_waiting(budget, plan)
        self._redistribute_prefill_remainder(budget, plan)
        return SchedulerOutput(scheduled=tuple(plan))

    def _finish(self, state: _RequestState) -> None:
        state.status = _Status.FINISHED
        self._running.remove(state.request.request_id)
        self._priority_events[(state.request.scheduling_class, "complete")] += 1
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
            was_list = not isinstance(tokens, int)
            token_list = self._normalize_tokens(request_id, tokens)
            state = self._states.get(request_id)
            if state is None:
                raise ValueError(f"request {request_id!r} is not awaiting a sampled token")
            # a speculative chunk's tokens arrive as one list commit; its
            # recorded reservation is how the rejected-draft shortfall is
            # released exactly (m8 D3 — never via arithmetic on k)
            spec_expected = state.spec_in_flight if was_list and state.spec_in_flight else 0
            if spec_expected:
                state.spec_in_flight = 0
            committed_here = 0
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
                if not state.outputs and state.allocation is not None:
                    # The prompt KV becomes shareable only after the device
                    # result that proves its prompt-completing forward finished.
                    # Publishing during schedule() lets a depth>1 plan reuse
                    # pages that are still queued on the device executor.
                    self._kv.mark_computed(state.allocation)
                state.outputs.append(token_id)
                committed_here += 1
                reason = self._terminal_reason(state, token_id)
                if reason is not None:
                    state.surplus_in_flight += state.in_flight
                    state.in_flight = 0
                    state.finish_reason = reason
                    self._finish(state)
                    finished.append(request_id)
                    finished_here = True
            if spec_expected:
                shortfall = spec_expected - committed_here
                if shortfall > 0:
                    if state.status is _Status.RUNNING:
                        # rejected draft slots never arrive: release directly,
                        # NOT via surplus (surplus means "WILL arrive, trim it")
                        state.in_flight -= shortfall
                    else:
                        # terminal/aborted: the terminal path moved the whole
                        # remainder to surplus — the never-arriving part leaves
                        state.surplus_in_flight = max(
                            0, state.surplus_in_flight - shortfall
                        )
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
