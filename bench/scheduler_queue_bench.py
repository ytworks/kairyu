#!/usr/bin/env python3
"""Reproducible 100k-request Scheduler waiting-queue benchmark (issue #219).

Run:
    uv run python bench/scheduler_queue_bench.py --requests 100000 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from kairyu.engine.core.scheduler import _IndexedWaitingQueue


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100_000)
    parser.add_argument("--removals", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
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
    # Evenly spaced removals keep the benchmark representative of arbitrary
    # client cancellation instead of repeatedly selecting only one end.
    if args.removals:
        victims = [
            request_ids[index * args.requests // args.removals]
            for index in range(args.removals)
        ]
    else:
        victims = []
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
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
