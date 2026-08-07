"""Indexed Scheduler waiting-queue behavior and stress coverage (issue #219)."""

from __future__ import annotations

import random

import pytest

from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache
from kairyu.engine.core.scheduler import (
    EngineRequest,
    Scheduler,
    _IndexedWaitingQueue,
)


def _reference_order(
    request_ids: list[str],
    records: dict[str, tuple[int, float]],
    priority_age_s: float | None,
) -> list[str]:
    if priority_age_s is None:
        return list(request_ids)

    def effective_priority(request_id: str) -> int:
        priority, arrival = records[request_id]
        if priority_age_s:
            age_ns = max(1, round(priority_age_s * 1_000_000_000))
            # The omitted ``-now`` term is common to every request.
            return priority * age_ns + round(arrival * 1_000_000_000)
        return priority

    return sorted(request_ids, key=effective_priority)


@pytest.mark.parametrize("priority_age_s", [None, 0.0, 2.0])
@pytest.mark.parametrize("seed", range(5))
def test_randomized_queue_matches_legacy_list_behavior(
    priority_age_s: float | None, seed: int
) -> None:
    rng = random.Random(seed)
    queue = _IndexedWaitingQueue(priority_age_s)
    legacy: list[str] = []
    records: dict[str, tuple[int, float]] = {}
    next_id = 0

    for operation in range(5_000):
        choice = rng.random()
        if not legacy or choice < 0.50:
            request_id = f"request-{next_id}"
            next_id += 1
            priority = rng.randint(-4, 4)
            arrival = float(rng.randint(0, 100))
            records[request_id] = (priority, arrival)
            legacy.append(request_id)
            queue.append(
                request_id,
                priority=priority,
                arrival=arrival,
            )
        elif choice < 0.72:
            request_id = rng.choice(legacy)
            legacy.remove(request_id)
            queue.remove(request_id)
        elif choice < 0.82:
            # Recompute preemption removes a running ID and reinserts it at the
            # front, retaining its original priority and arrival timestamp.
            request_id = rng.choice(legacy)
            legacy.remove(request_id)
            legacy.insert(0, request_id)
            queue.remove(request_id)
            priority, arrival = records[request_id]
            queue.append(
                request_id,
                priority=priority,
                arrival=arrival,
                front=True,
            )
        else:
            ordered = _reference_order(legacy, records, priority_age_s)
            legacy[:] = ordered  # legacy schedule() sorts the list in place
            expected = ordered[0]
            legacy.pop(0)
            assert queue.peek() == expected
            assert queue.popleft() == expected

        assert len(queue) == len(legacy)
        if operation % 127 == 0:
            legacy[:] = _reference_order(legacy, records, priority_age_s)
            assert tuple(queue) == tuple(legacy)

    while legacy:
        ordered = _reference_order(legacy, records, priority_age_s)
        legacy[:] = ordered
        expected = ordered[0]
        legacy.pop(0)
        assert queue.popleft() == expected
    assert not queue


def test_priority_ties_are_stable_and_front_requeue_wins_the_tie() -> None:
    queue = _IndexedWaitingQueue(priority_age_s=10.0)
    # Both immutable keys are one: 1 + 0/10 == 0 + 10/10.
    queue.append("first", priority=1, arrival=0.0)
    queue.append("second", priority=0, arrival=10.0)
    assert tuple(queue) == ("first", "second")

    queue.remove("second")
    queue.append("second", priority=0, arrival=10.0, front=True)
    assert tuple(queue) == ("second", "first")


def test_smaller_priority_wins_and_aging_improves_an_old_larger_value() -> None:
    strict = _IndexedWaitingQueue(priority_age_s=0.0)
    strict.append("batch", priority=10, arrival=0.0)
    strict.append("interactive", priority=0, arrival=1.0)
    assert strict.popleft() == "interactive"

    aging = _IndexedWaitingQueue(priority_age_s=1.0)
    aging.append("old-batch", priority=5, arrival=0.0)
    aging.append("fresh-interactive", priority=0, arrival=10.0)
    assert aging.popleft() == "old-batch"


@pytest.mark.parametrize("priority_age_s", [0.0, 60.0])
def test_signed_int64_priority_order_is_exact(priority_age_s: float) -> None:
    queue = _IndexedWaitingQueue(priority_age_s=priority_age_s)
    queue.append(
        "adjacent-higher",
        priority=-(2**63) + 1,
        arrival=0.0,
    )
    queue.append(
        "adjacent-lower",
        priority=-(2**63),
        arrival=0.0,
    )

    assert queue.popleft() == "adjacent-lower"


def test_output_bearing_deferred_decode_blocks_later_interactive_at_sequence_cap():
    lowest = 2**63 - 1
    scheduler = Scheduler(
        RadixKVCache(num_pages=16, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=1,
        page_size=4,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest(
            "deferred",
            (1, 2, 3, 4),
            max_new_tokens=4,
            priority=lowest,
            scheduling_class="batch",
        )
    )

    prefill = scheduler.schedule()
    assert [(chunk.request_id, chunk.is_prefill) for chunk in prefill.scheduled] == [
        ("deferred", True)
    ]
    scheduler.update({"deferred": 10})
    assert scheduler.output_tokens("deferred") == (10,)

    scheduler.add_request(
        EngineRequest(
            "interactive",
            (5, 6, 7, 8),
            max_new_tokens=1,
            priority=lowest - 1,
            scheduling_class="interactive",
        )
    )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.is_prefill) for chunk in step.scheduled] == [
        ("deferred", False)
    ]
    assert scheduler.waiting_ids == ("interactive",)


def test_hundred_thousand_ids_support_indexed_removal_and_fifo_drain() -> None:
    queue = _IndexedWaitingQueue(priority_age_s=None)
    count = 100_000
    for index in range(count):
        queue.append(f"request-{index}", priority=0, arrival=float(index))
    for index in range(0, count, 2):
        queue.remove(f"request-{index}")

    assert len(queue) == count // 2
    assert queue.peek() == "request-1"
    assert [queue.popleft() for _ in range(3)] == [
        "request-1",
        "request-3",
        "request-5",
    ]


def test_kv_blocked_priority_head_allows_one_complete_small_bypass() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=3, page_size=4),
        max_num_batched_tokens=64,
        max_num_seqs=3,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest("running", tuple(range(1, 9)), max_new_tokens=1)
    )
    scheduler.schedule()  # occupies two pages and leaves token 0 in flight

    scheduler.add_request(
        EngineRequest(
            "high-large",
            tuple(range(101, 109)),
            max_new_tokens=1,
            priority=-10,
        )
    )
    scheduler.add_request(
        EngineRequest(
            "low-small",
            tuple(range(201, 205)),
            max_new_tokens=1,
            priority=0,
        )
    )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [("low-small", 4)]
    assert scheduler.waiting_ids == ("high-large",)
    assert scheduler.states["high-large"].admission_bypasses == 1


def test_blocked_head_returns_phantom_share_to_completing_successor() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=5, page_size=4),
        max_num_batched_tokens=8,
        max_num_seqs=3,
    )
    scheduler.add_request(EngineRequest("running", tuple(range(1, 9)), max_new_tokens=3))
    first = scheduler.schedule()
    scheduler.update({first.scheduled[0].request_id: 10})
    scheduler.add_request(EngineRequest("blocked-head", tuple(range(101, 113)), max_new_tokens=1))
    scheduler.add_request(EngineRequest("successor", tuple(range(201, 206)), max_new_tokens=1))

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.is_prefill, chunk.num_tokens) for chunk in step.scheduled] == [
        ("running", False, 1),
        ("successor", True, 5),
    ]
    assert scheduler.waiting_ids == ("blocked-head",)
    assert scheduler.states["blocked-head"].admission_bypasses == 1


def test_skip_does_not_cross_tier_while_winning_prefill_is_incomplete() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=7, page_size=4),
        max_num_batched_tokens=8,
        max_num_seqs=3,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest(
            "best-running",
            tuple(range(1, 21)),
            max_new_tokens=1,
            priority=0,
        )
    )
    scheduler.schedule()
    scheduler.add_request(
        EngineRequest(
            "blocked-head",
            tuple(range(101, 113)),
            max_new_tokens=1,
            priority=0,
        )
    )
    scheduler.add_request(
        EngineRequest(
            "lower-successor",
            tuple(range(201, 205)),
            max_new_tokens=1,
            priority=1,
        )
    )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [
        ("best-running", 8)
    ]
    assert scheduler.waiting_ids == ("blocked-head", "lower-successor")
    assert scheduler.states["blocked-head"].admission_bypasses == 0


def test_blocked_head_bypass_is_lifetime_bounded_until_capacity_frees() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=4, page_size=4),
        max_num_batched_tokens=64,
        max_num_seqs=3,
        priority_age_s=0.0,
    )
    scheduler.add_request(EngineRequest("running", tuple(range(1, 9)), max_new_tokens=3))
    scheduler.schedule()
    scheduler.update({"running": 10})
    scheduler.add_request(
        EngineRequest(
            "blocked-head",
            tuple(range(101, 109)),
            max_new_tokens=1,
            priority=-10,
        )
    )
    scheduler.add_request(
        EngineRequest(
            "small-one",
            tuple(range(201, 205)),
            max_new_tokens=1,
            priority=0,
        )
    )

    first = scheduler.schedule()
    assert [chunk.request_id for chunk in first.scheduled] == [
        "running",
        "small-one",
    ]
    scheduler.update({"running": 11, "small-one": 20})
    scheduler.add_request(
        EngineRequest(
            "small-two",
            tuple(range(201, 205)),
            max_new_tokens=1,
            priority=0,
        )
    )

    capped = scheduler.schedule()

    assert [chunk.request_id for chunk in capped.scheduled] == ["running"]
    assert scheduler.waiting_ids == ("blocked-head", "small-two")
    scheduler.update({"running": 12})

    after_release = scheduler.schedule()
    assert [chunk.request_id for chunk in after_release.scheduled] == [
        "blocked-head",
        "small-two",
    ]
    assert scheduler.waiting_ids == ()


def test_decode_watermark_allows_only_candidate_that_preserves_reserve() -> None:
    cache = RadixKVCache(num_pages=5, page_size=4)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=64,
        max_num_seqs=3,
        decode_watermark_pages=1,
        priority_age_s=0.0,
    )
    scheduler.add_request(EngineRequest("running", tuple(range(1, 9)), max_new_tokens=3))
    scheduler.schedule()
    scheduler.update({"running": 10})
    scheduler.add_request(
        EngineRequest(
            "watermark-head",
            tuple(range(101, 109)),
            max_new_tokens=1,
            priority=-10,
        )
    )
    scheduler.add_request(
        EngineRequest(
            "watermark-small",
            tuple(range(201, 205)),
            max_new_tokens=1,
            priority=0,
        )
    )

    step = scheduler.schedule()

    assert [chunk.request_id for chunk in step.scheduled] == [
        "running",
        "watermark-small",
    ]
    assert scheduler.waiting_ids == ("watermark-head",)
    assert cache.num_free_pages == 1


def test_skip_ahead_does_not_admit_zero_token_unsafe_successor() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=4, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=5,
        max_num_partial_prefills=2,
    )
    scheduler.add_request(EngineRequest("decode", (1,), max_new_tokens=6))
    scheduler.add_request(EngineRequest("partial", tuple(range(101, 109)), max_new_tokens=2))
    scheduler.schedule()
    scheduler.update({"decode": 10})
    scheduler.add_request(EngineRequest("blocked-head", tuple(range(201, 209)), max_new_tokens=1))
    scheduler.add_request(EngineRequest("unsafe-successor", (301,), max_new_tokens=12))

    step = scheduler.schedule()

    assert [chunk.request_id for chunk in step.scheduled] == [
        "decode",
        "partial",
    ]
    assert all(chunk.num_tokens > 0 for chunk in step.scheduled)
    assert scheduler.waiting_ids == ("blocked-head", "unsafe-successor")
    assert "unsafe-successor" not in scheduler._running
    assert scheduler.states["blocked-head"].admission_bypasses == 0


def test_cached_head_uses_new_page_estimate_and_is_not_bypassed() -> None:
    cache = RadixKVCache(num_pages=5, page_size=4)
    cached_prompt = tuple(range(101, 109))
    cached = cache.allocate(cached_prompt)
    cache.mark_computed(cached)
    cache.free(cached)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=64,
        max_num_seqs=3,
        decode_watermark_pages=1,
        priority_age_s=0.0,
    )
    scheduler.add_request(EngineRequest("running", tuple(range(1, 5)), max_new_tokens=3))
    scheduler.schedule()
    scheduler.update({"running": 10})
    scheduler.add_request(
        EngineRequest(
            "cached-head",
            cached_prompt,
            max_new_tokens=1,
            priority=-10,
        )
    )
    scheduler.add_request(
        EngineRequest(
            "uncached-small",
            tuple(range(201, 205)),
            max_new_tokens=1,
            priority=0,
        )
    )

    step = scheduler.schedule()

    assert [chunk.request_id for chunk in step.scheduled] == [
        "running",
        "cached-head",
    ]
    assert step.scheduled[1].num_tokens == 1
    assert scheduler.states["cached-head"].admission_bypasses == 0
    assert scheduler.waiting_ids == ("uncached-small",)


def test_skip_ahead_inspects_only_the_immediate_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = RadixKVCache(num_pages=3, page_size=4)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        priority_age_s=0.0,
    )
    scheduler.add_request(EngineRequest("running", tuple(range(1, 9)), max_new_tokens=1))
    scheduler.schedule()
    scheduler.add_request(EngineRequest("blocked-head", tuple(range(101, 109)), priority=-10))
    scheduler.add_request(EngineRequest("blocked-next", tuple(range(201, 209)), priority=0))
    scheduler.add_request(EngineRequest("fitting-third", tuple(range(301, 305)), priority=1))
    probes: list[tuple[int, ...]] = []
    original_probe = cache.peek_cached_tokens

    def counted_probe(tokens: tuple[int, ...]) -> int:
        probes.append(tokens)
        return original_probe(tokens)

    monkeypatch.setattr(cache, "peek_cached_tokens", counted_probe)

    step = scheduler.schedule()

    assert step.scheduled == ()
    assert scheduler.waiting_ids == (
        "blocked-head",
        "blocked-next",
        "fitting-third",
    )
    assert probes
    assert set(probes) <= {
        tuple(range(101, 109)),
        tuple(range(201, 209)),
    }


def test_failed_head_allocation_cannot_create_successor_capacity() -> None:
    cache = RadixKVCache(num_pages=6, page_size=4)
    cached_prompt = tuple(range(901, 905))
    cached = cache.allocate(cached_prompt)
    cache.mark_computed(cached)
    cache.free(cached)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=16,
        max_num_seqs=3,
        priority_age_s=0.0,
    )
    scheduler.add_request(EngineRequest("running", tuple(range(1, 9)), max_new_tokens=3))
    first = scheduler.schedule()
    scheduler.update({first.scheduled[0].request_id: 10})
    scheduler.add_request(
        EngineRequest(
            "blocked-head",
            tuple(range(101, 117)),
            max_new_tokens=1,
            priority=-10,
        )
    )
    scheduler.add_request(
        EngineRequest(
            "successor",
            tuple(range(201, 213)),
            max_new_tokens=1,
            priority=0,
        )
    )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.is_prefill) for chunk in step.scheduled] == [
        ("running", False)
    ]
    assert scheduler.waiting_ids == ("blocked-head", "successor")
    assert scheduler.states["blocked-head"].admission_bypasses == 0


def test_waiting_higher_priority_runs_before_lower_priority_running_prefill() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=8, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=2,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest("batch", tuple(range(1, 9)), max_new_tokens=1, priority=10)
    )
    assert scheduler.schedule().scheduled[0].request_id == "batch"

    scheduler.add_request(
        EngineRequest(
            "interactive",
            tuple(range(101, 109)),
            max_new_tokens=1,
            priority=0,
        )
    )

    step = scheduler.schedule()

    assert [chunk.request_id for chunk in step.scheduled] == ["interactive"]
    assert scheduler.states["batch"].computed_prompt == 4
    assert [chunk.request_id for chunk in scheduler.schedule().scheduled] == [
        "interactive"
    ]
    assert scheduler.states["batch"].computed_prompt == 4


def test_higher_priority_preempts_output_free_prefill_for_sequence_slot() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=8, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=1,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest("batch", tuple(range(1, 9)), max_new_tokens=1, priority=10)
    )
    scheduler.schedule()
    scheduler.add_request(
        EngineRequest(
            "interactive",
            tuple(range(101, 105)),
            max_new_tokens=1,
            priority=0,
        )
    )

    step = scheduler.schedule()

    assert [chunk.request_id for chunk in step.scheduled] == ["interactive"]
    assert scheduler.waiting_ids == ("batch",)
    assert scheduler.states["batch"].computed_prompt == 0


def test_higher_priority_preempts_output_free_prefill_for_kv_pages() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=2, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=2,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest("batch", tuple(range(1, 9)), max_new_tokens=1, priority=10)
    )
    scheduler.schedule()
    scheduler.add_request(
        EngineRequest(
            "interactive",
            tuple(range(101, 105)),
            max_new_tokens=1,
            priority=0,
        )
    )

    step = scheduler.schedule()

    assert [chunk.request_id for chunk in step.scheduled] == ["interactive"]
    assert scheduler.waiting_ids == ("batch",)


def test_priority_allocation_failure_exhausts_victims_without_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=4, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=2,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest("batch", tuple(range(1, 9)), max_new_tokens=1, priority=10)
    )
    scheduler.schedule()
    scheduler.add_request(
        EngineRequest(
            "interactive",
            tuple(range(101, 105)),
            max_new_tokens=1,
            priority=0,
        )
    )

    original_allocate = RadixKVCache.allocate

    def fail_interactive(
        cache: RadixKVCache, tokens: tuple[int, ...]
    ) -> KVAllocation:
        if tokens and tokens[0] == 101:
            raise KVCacheFull("injected persistent allocation failure")
        return original_allocate(cache, tokens)

    monkeypatch.setattr(RadixKVCache, "allocate", fail_interactive)

    step = scheduler.schedule()

    assert scheduler.finish_reason("interactive") == "length"
    assert scheduler.drain_rejected() == ("interactive",)
    assert [chunk.request_id for chunk in step.scheduled] == ["batch"]


def test_priority_preemption_tolerates_two_pending_incomplete_prefills() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=4, page_size=4),
        max_num_batched_tokens=2,
        max_num_seqs=1,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest("batch", tuple(range(1, 9)), max_new_tokens=1, priority=10)
    )

    first = scheduler.schedule()
    second = scheduler.schedule()

    assert [chunk.request_id for chunk in first.scheduled] == ["batch"]
    assert [chunk.request_id for chunk in second.scheduled] == ["batch"]
    assert scheduler.states["batch"].computed_prompt == 4
    assert scheduler.states["batch"].in_flight == 0

    scheduler.add_request(
        EngineRequest(
            "interactive",
            tuple(range(101, 103)),
            max_new_tokens=1,
            priority=0,
        )
    )
    high = scheduler.schedule()

    assert [chunk.request_id for chunk in high.scheduled] == ["interactive"]
    assert scheduler.waiting_ids == ("batch",)
    assert scheduler.states["batch"].computed_prompt == 0
    # Both frozen batch snapshots were incomplete and therefore return no
    # sampled token when they commit after the recompute preemption.
    assert scheduler.update({}) == ()
    assert scheduler.update({}) == ()
    assert scheduler.waiting_ids == ("batch",)
    assert scheduler.states["interactive"].in_flight == 1


def test_equal_priority_waiter_cohorts_without_preempting_running_prefill() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=8, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=2,
        priority_age_s=0.0,
    )
    scheduler.add_request(EngineRequest("first", tuple(range(1, 9)), max_new_tokens=1))
    scheduler.schedule()
    scheduler.add_request(EngineRequest("second", tuple(range(101, 105)), max_new_tokens=1))

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [
        ("first", 2),
        ("second", 2),
    ]
    assert scheduler.waiting_ids == ()


def test_invalid_priority_head_does_not_make_best_tier_donate_budget() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=8, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=3,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest(
            "low-running",
            tuple(range(1, 13)),
            max_new_tokens=1,
            priority=10,
        )
    )
    scheduler.schedule()
    scheduler.add_request(EngineRequest("invalid", (), priority=0))
    scheduler.add_request(
        EngineRequest(
            "high-valid",
            tuple(range(101, 109)),
            max_new_tokens=1,
            priority=0,
        )
    )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [("high-valid", 4)]
    assert scheduler.drain_rejected() == ("invalid",)


def test_multiple_invalid_heads_do_not_consume_cohort_frontier() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=16, page_size=4),
        max_num_batched_tokens=8,
        max_num_seqs=4,
    )
    scheduler.add_request(EngineRequest("invalid-a", ()))
    scheduler.add_request(EngineRequest("invalid-b", ()))
    scheduler.add_request(EngineRequest("long-a", tuple(range(1, 21)), max_new_tokens=1))
    scheduler.add_request(EngineRequest("long-b", tuple(range(101, 121)), max_new_tokens=1))

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [
        ("long-a", 4),
        ("long-b", 4),
    ]
    assert scheduler.drain_rejected() == ("invalid-a", "invalid-b")


def test_priority_remainder_stays_in_incomplete_winning_tier() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=16, page_size=4),
        max_num_batched_tokens=4,
        max_num_seqs=3,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest(
            "high-long",
            tuple(range(1, 11)),
            max_new_tokens=1,
            priority=0,
        )
    )
    scheduler.add_request(EngineRequest("high-short", (101,), max_new_tokens=1, priority=0))
    scheduler.add_request(
        EngineRequest(
            "low-long",
            tuple(range(201, 211)),
            max_new_tokens=1,
            priority=1,
        )
    )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [
        ("high-long", 3),
        ("high-short", 1),
    ]
    assert scheduler.waiting_ids == ("low-long",)


def test_decode_stays_ahead_of_higher_priority_waiting_prefill() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=8, page_size=4),
        max_num_batched_tokens=1,
        max_num_seqs=2,
        priority_age_s=0.0,
    )
    scheduler.add_request(
        EngineRequest("batch-decode", (1,), max_new_tokens=2, priority=10)
    )
    scheduler.schedule()
    scheduler.update({"batch-decode": 7})
    scheduler.add_request(
        EngineRequest("interactive", (101,), max_new_tokens=1, priority=0)
    )

    step = scheduler.schedule()

    assert [chunk.request_id for chunk in step.scheduled] == ["batch-decode"]
    assert step.scheduled[0].is_prefill is False
    assert scheduler.waiting_ids == ("interactive",)
