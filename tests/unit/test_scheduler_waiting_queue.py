"""Indexed Scheduler waiting-queue behavior and stress coverage (issue #219)."""

from __future__ import annotations

import random

import pytest

from kairyu.engine.core.radix_kv import RadixKVCache
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
    now = 10_000.0

    def effective_priority(request_id: str) -> float:
        priority, arrival = records[request_id]
        if priority_age_s:
            return priority + (now - arrival) / priority_age_s
        return float(priority)

    return sorted(request_ids, key=lambda request_id: -effective_priority(request_id))


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
    # Both keys are zero: 0 - 0/10 == 1 - 10/10.
    queue.append("first", priority=0, arrival=0.0)
    queue.append("second", priority=1, arrival=10.0)
    assert tuple(queue) == ("first", "second")

    queue.remove("second")
    queue.append("second", priority=1, arrival=10.0, front=True)
    assert tuple(queue) == ("second", "first")


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


def test_priority_head_blocks_smaller_request_that_would_fit() -> None:
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
            priority=10,
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

    assert scheduler.schedule().scheduled == ()
    assert scheduler.waiting_ids == ("high-large", "low-small")
