"""Measure RuleRouter latency (p50/p99) and count its routing work.

Latency percentiles are diagnostics.  The verdict is deterministic: before
and after the measured routes it routes the fixed queries once more and counts
the Python and C calls made (feature candidates, keywords, and markers each
cost calls) plus every hash or comparison of the query itself, and checks that
the work per route stays within a fixed budget and does not grow with the
number of routes already served.

Run: uv run python verification/orchestration/performance/router_latency.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence

from kairyu.orchestration.router import RuleRouter

N_ROUTES = 10_000
QUERIES = (
    "What is the capital of France?",
    "Prove the theorem and explain your reasoning step by step.",
    "Fix this bug:\n```python\nx = [\n```\nwhy?",
    "First, research options. Then design. After that, implement. Finally, verify. " * 5,
)
WORK_UNIT = "Python and C calls plus query hashes and comparisons while routing"
# Rule routing is a few dozen calls per query; a model, tokenizer, or remote
# call in the hot path costs far more than this budget.
MAX_CALLS_PER_ROUTE = 1_000
# The router keeps no growing state, so per-route work must not grow with the
# number of routes served.
MAX_WORK_GROWTH = 2.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routes",
        type=int,
        default=N_ROUTES,
        help=f"number of routes to measure (default: {N_ROUTES})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable metrics",
    )
    parser.add_argument(
        "--assert-gate",
        action="store_true",
        help="exit nonzero when a deterministic work check fails",
    )
    return parser


class _WorkCounter:
    count = 0


class _CountedQuery(str):
    """Query whose hashing and comparisons are counted."""

    __slots__ = ()

    def __hash__(self) -> int:
        _WorkCounter.count += 1
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        _WorkCounter.count += 1
        return str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        _WorkCounter.count += 1
        return str.__ne__(self, other)


def _count_work(fn: Callable[[], object]) -> int:
    def profile(_frame, event: str, _arg) -> None:
        if event in ("call", "c_call"):
            _WorkCounter.count += 1

    _WorkCounter.count = 0
    sys.setprofile(profile)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return _WorkCounter.count


def _route_queries(router: RuleRouter) -> int:
    queries = [_CountedQuery(query) for query in QUERIES]
    return _count_work(lambda: [router.route(query) for query in queries])


def measure(routes: int) -> dict[str, object]:
    if routes < 1:
        raise ValueError("routes must be at least 1")

    router = RuleRouter()
    first = _route_queries(router)
    durations = []
    for index in range(routes):
        query = QUERIES[index % len(QUERIES)]
        start = time.perf_counter()
        router.route(query)
        durations.append(time.perf_counter() - start)
    last = _route_queries(router)
    durations.sort()
    queries = len(QUERIES)
    first_per_route = first / queries
    last_per_route = last / queries
    return {
        "routes": routes,
        "p50_seconds": statistics.median(durations),
        "p99_seconds": durations[int(len(durations) * 0.99)],
        "work": {
            "unit": WORK_UNIT,
            "routes_served": {
                "sizes": [0, routes],
                "operations": [queries, queries],
                "work": [first, last],
            },
        },
        "max_calls_per_route": MAX_CALLS_PER_ROUTE,
        "max_work_growth": MAX_WORK_GROWTH,
        "passed": (
            0 < first_per_route <= MAX_CALLS_PER_ROUTE
            and last_per_route <= MAX_CALLS_PER_ROUTE
            and last_per_route <= MAX_WORK_GROWTH * first_per_route
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.routes < 1:
        parser.error("--routes must be at least 1")

    result = measure(args.routes)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        p50 = float(result["p50_seconds"])
        p99 = float(result["p99_seconds"])
        first, last = result["work"]["routes_served"]["work"]
        print(f"routes={args.routes}")
        print(
            f"router latency p50={p50 * 1e6:.1f}us "
            f"p99={p99 * 1e6:.1f}us (diagnostic)"
        )
        print(
            f"work for {len(QUERIES)} routes: first={first} last={last}"
        )
        verdict = "OK" if result["passed"] else "VIOLATED"
        print(
            f"work: <= {MAX_CALLS_PER_ROUTE} calls per route and growth <= "
            f"{MAX_WORK_GROWTH} -> {verdict}"
        )
    return int(args.assert_gate and not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
