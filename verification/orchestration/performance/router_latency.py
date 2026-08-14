"""Measure RuleRouter latency (p50/p99). Prints only measured values.

Run: uv run python verification/orchestration/performance/router_latency.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence

from kairyu.orchestration.router import RuleRouter

N_ROUTES = 10_000
P99_BUDGET_SECONDS = 0.010
QUERIES = (
    "What is the capital of France?",
    "Prove the theorem and explain your reasoning step by step.",
    "Fix this bug:\n```python\nx = [\n```\nwhy?",
    "First, research options. Then design. After that, implement. Finally, verify. " * 5,
)


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
        help="exit nonzero when the p99 budget is exceeded",
    )
    return parser


def measure(routes: int) -> dict[str, float | int | bool]:
    if routes < 1:
        raise ValueError("routes must be at least 1")

    router = RuleRouter()
    durations = []
    for index in range(routes):
        query = QUERIES[index % len(QUERIES)]
        start = time.perf_counter()
        router.route(query)
        durations.append(time.perf_counter() - start)
    durations.sort()
    p50 = statistics.median(durations)
    p99 = durations[int(len(durations) * 0.99)]
    return {
        "routes": routes,
        "p50_seconds": p50,
        "p99_seconds": p99,
        "p99_budget_seconds": P99_BUDGET_SECONDS,
        "passed": p99 < P99_BUDGET_SECONDS,
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
        print(f"routes={args.routes}")
        print(
            f"router latency p50={p50 * 1e6:.1f}us "
            f"p99={p99 * 1e6:.1f}us"
        )
        verdict = "OK" if result["passed"] else "VIOLATED"
        print(f"budget: p99 < 10ms -> {verdict}")
    return int(args.assert_gate and not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
