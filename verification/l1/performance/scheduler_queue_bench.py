#!/usr/bin/env python3
"""Reproducible 100k-request Scheduler waiting-queue benchmark (issue #219).

Timings compare the indexed queue with the legacy list and are diagnostics.
The ``work`` section is deterministic: at two queue sizes it counts the
queue's store primitives (one unit per call and one per item iterated) and the
hashes and comparisons of request IDs and priority keys, so the growth of work
per operation can be checked without a clock.

Run:
    uv run python verification/l1/performance/scheduler_queue_bench.py --requests 100000 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import OrderedDict

from kairyu.engine.core.scheduler import _IndexedWaitingQueue

WORK_UNIT = (
    "waiting-store primitives (1 per call, 1 per item iterated) plus "
    "request-ID and priority-key hashes and comparisons"
)
# One cancellation per ten queued requests, as in the timed workload.
WORK_REMOVAL_DIVISOR = 10


class _WorkCounter:
    count = 0


def _tick() -> None:
    _WorkCounter.count += 1


class _CountedId(str):
    """Request ID whose hashing and comparisons are counted."""

    __slots__ = ()

    def __hash__(self) -> int:
        _tick()
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        _tick()
        return str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        _tick()
        return str.__ne__(self, other)

    def __lt__(self, other: str) -> bool:
        _tick()
        return str.__lt__(self, other)

    def __gt__(self, other: str) -> bool:
        _tick()
        return str.__gt__(self, other)


def _counted_priority(value):
    return value if value is NotImplemented else _CountedPriority(value)


class _CountedPriority(int):
    """Priority whose derived queue keys count every comparison."""

    __slots__ = ()
    __hash__ = int.__hash__

    def __add__(self, other):
        return _counted_priority(int.__add__(self, other))

    def __radd__(self, other):
        return _counted_priority(int.__radd__(self, other))

    def __mul__(self, other):
        return _counted_priority(int.__mul__(self, other))

    def __rmul__(self, other):
        return _counted_priority(int.__rmul__(self, other))

    def __eq__(self, other: object) -> bool:
        _tick()
        return int.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        _tick()
        return int.__ne__(self, other)

    def __lt__(self, other: int) -> bool:
        _tick()
        return int.__lt__(self, other)

    def __le__(self, other: int) -> bool:
        _tick()
        return int.__le__(self, other)

    def __gt__(self, other: int) -> bool:
        _tick()
        return int.__gt__(self, other)

    def __ge__(self, other: int) -> bool:
        _tick()
        return int.__ge__(self, other)


class _CountingView:
    """Key, value, or item view that counts every item it yields."""

    __slots__ = ("_view",)

    def __init__(self, view) -> None:
        self._view = view

    def __iter__(self):
        for item in self._view:
            _tick()
            yield item

    def __reversed__(self):
        for item in reversed(self._view):
            _tick()
            yield item

    def __len__(self) -> int:
        _tick()
        return len(self._view)

    def __contains__(self, item: object) -> bool:
        _tick()
        return item in self._view


class _CountingOrderedDict(OrderedDict):
    """Waiting store that counts its primitives and the items iterated."""

    def __len__(self) -> int:
        _tick()
        return OrderedDict.__len__(self)

    def __contains__(self, key: object) -> bool:
        _tick()
        return OrderedDict.__contains__(self, key)

    def __getitem__(self, key):
        _tick()
        return OrderedDict.__getitem__(self, key)

    def __setitem__(self, key, value) -> None:
        _tick()
        OrderedDict.__setitem__(self, key, value)

    def __delitem__(self, key) -> None:
        _tick()
        OrderedDict.__delitem__(self, key)

    def __iter__(self):
        for key in OrderedDict.__iter__(self):
            _tick()
            yield key

    def __reversed__(self):
        for key in OrderedDict.__reversed__(self):
            _tick()
            yield key

    def get(self, key, default=None):
        _tick()
        return OrderedDict.get(self, key, default)

    def pop(self, key, *default):
        _tick()
        return OrderedDict.pop(self, key, *default)

    def popitem(self, last: bool = True):
        _tick()
        return OrderedDict.popitem(self, last=last)

    def move_to_end(self, key, last: bool = True) -> None:
        _tick()
        OrderedDict.move_to_end(self, key, last=last)

    def setdefault(self, key, default=None):
        _tick()
        return OrderedDict.setdefault(self, key, default)

    def keys(self):
        return _CountingView(OrderedDict.keys(self))

    def values(self):
        return _CountingView(OrderedDict.values(self))

    def items(self):
        return _CountingView(OrderedDict.items(self))


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _legacy_fifo_drain(request_ids: list[str]) -> None:
    waiting = list(request_ids)
    while waiting:
        waiting.pop(0)


def _indexed_fifo_drain(request_ids: list[str]) -> None:
    waiting = _IndexedWaitingQueue(priority_age_s=None)
    waiting.extend(
        [
            (request_id, 0, float(index))
            for index, request_id in enumerate(request_ids)
        ]
    )
    while waiting:
        waiting.popleft()


def _legacy_abort(request_ids: list[str], victims: list[str]) -> None:
    waiting = list(request_ids)
    for request_id in victims:
        waiting.remove(request_id)


def _indexed_abort(request_ids: list[str], victims: list[str]) -> None:
    waiting = _IndexedWaitingQueue(priority_age_s=None)
    waiting.extend(
        [
            (request_id, 0, float(index))
            for index, request_id in enumerate(request_ids)
        ]
    )
    for request_id in victims:
        waiting.remove(request_id)


def _legacy_priority_drain(
    requests: list[tuple[str, int, float]],
) -> None:
    waiting = list(requests)
    waiting.sort(
        key=lambda item: item[1] * 10_000_000_000
        + round(item[2] * 1_000_000_000)
    )
    while waiting:
        waiting.pop(0)


def _indexed_priority_drain(
    requests: list[tuple[str, int, float]],
) -> None:
    waiting = _IndexedWaitingQueue(priority_age_s=10.0)
    waiting.extend(requests)
    while waiting:
        waiting.popleft()


def _measure(fn, repeats: int) -> dict[str, float | int]:
    samples = [_timed(fn) for _ in range(repeats)]
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "repeats": repeats,
    }


def _comparison(legacy, indexed, repeats: int) -> dict[str, object]:
    legacy_result = _measure(legacy, repeats)
    indexed_result = _measure(indexed, repeats)
    return {
        "legacy": legacy_result,
        "indexed": indexed_result,
        "median_speedup": (
            float(legacy_result["median_seconds"])
            / float(indexed_result["median_seconds"])
        ),
    }


def _victim_indices(requests: int, removals: int) -> list[int]:
    # Evenly spaced removals keep the benchmark representative of arbitrary
    # client cancellation instead of repeatedly selecting only one end.
    return [index * requests // removals for index in range(removals)]


def _counted_fifo_queue(
    size: int,
) -> tuple[_IndexedWaitingQueue, list[str]]:
    waiting = _IndexedWaitingQueue(priority_age_s=None)
    # Instrument this instance's FIFO store before any request enters it.
    waiting._fifo = _CountingOrderedDict()
    request_ids = [_CountedId(f"request-{index}") for index in range(size)]
    return waiting, request_ids


def _work_fifo_drain(size: int) -> tuple[int, int]:
    waiting, request_ids = _counted_fifo_queue(size)
    entries = [
        (request_id, 0, float(index))
        for index, request_id in enumerate(request_ids)
    ]
    _WorkCounter.count = 0
    waiting.extend(entries)
    while waiting:
        waiting.popleft()
    return size, _WorkCounter.count


def _work_removal(size: int) -> tuple[int, int]:
    removals = size // WORK_REMOVAL_DIVISOR
    waiting, request_ids = _counted_fifo_queue(size)
    waiting.extend(
        [
            (request_id, 0, float(index))
            for index, request_id in enumerate(request_ids)
        ]
    )
    # A cancellation supplies an equal ID string, not the queued object.
    victims = [
        _CountedId(str(request_ids[index]))
        for index in _victim_indices(size, removals)
    ]
    _WorkCounter.count = 0
    for request_id in victims:
        waiting.remove(request_id)
    remaining = OrderedDict.__len__(waiting._fifo)
    if remaining != size - removals:
        raise AssertionError("indexed removal left the wrong queue size")
    return removals, _WorkCounter.count


def _work_priority_drain(size: int) -> tuple[int, int]:
    requests = [
        (
            _CountedId(f"request-{index}"),
            _CountedPriority(index % 11 - 5),
            index / 1_000.0,
        )
        for index in range(size)
    ]
    waiting = _IndexedWaitingQueue(priority_age_s=10.0)
    _WorkCounter.count = 0
    waiting.extend(requests)
    while waiting:
        waiting.popleft()
    return size, _WorkCounter.count


def _work_case(sizes: tuple[int, ...], measure) -> dict[str, list[int]]:
    operations: list[int] = []
    work: list[int] = []
    for size in sizes:
        count, units = measure(size)
        operations.append(count)
        work.append(units)
    return {"sizes": list(sizes), "operations": operations, "work": work}


def _work(sizes: tuple[int, ...]) -> dict[str, object]:
    return {
        "unit": WORK_UNIT,
        "fifo_full_drain": _work_case(sizes, _work_fifo_drain),
        "indexed_removal": _work_case(sizes, _work_removal),
        "priority_full_drain": _work_case(sizes, _work_priority_drain),
    }


def _work_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("work sizes must be integers") from error
    if len(sizes) != 2 or not WORK_REMOVAL_DIVISOR <= sizes[0] < sizes[1]:
        raise argparse.ArgumentTypeError(
            "work sizes must be two increasing integers of at least "
            f"{WORK_REMOVAL_DIVISOR}"
        )
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100_000)
    parser.add_argument("--removals", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--work-sizes",
        type=_work_sizes,
        default=(2_000, 16_000),
        help="two queue sizes for the deterministic work counts",
    )
    args = parser.parse_args()
    if args.requests < 1 or args.repeats < 1:
        parser.error("requests and repeats must be positive")
    if not 0 <= args.removals <= args.requests:
        parser.error("removals must be between zero and requests")

    request_ids = [f"request-{index}" for index in range(args.requests)]
    priority_requests = [
        (request_id, index % 11 - 5, index / 1_000.0)
        for index, request_id in enumerate(request_ids)
    ]
    victims = (
        [
            request_ids[index]
            for index in _victim_indices(args.requests, args.removals)
        ]
        if args.removals
        else []
    )
    result = {
        "requests": args.requests,
        "removals": args.removals,
        "fifo_full_drain": _comparison(
            lambda: _legacy_fifo_drain(request_ids),
            lambda: _indexed_fifo_drain(request_ids),
            args.repeats,
        ),
        "indexed_removal": _comparison(
            lambda: _legacy_abort(request_ids, victims),
            lambda: _indexed_abort(request_ids, victims),
            args.repeats,
        ),
        "priority_full_drain": _comparison(
            lambda: _legacy_priority_drain(priority_requests),
            lambda: _indexed_priority_drain(priority_requests),
            args.repeats,
        ),
        "work": _work(args.work_sizes),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
