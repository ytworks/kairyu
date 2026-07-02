"""Measure RuleRouter latency (p50/p99). Prints only measured values.

Run: uv run python bench/router_latency.py
"""

from __future__ import annotations

import statistics
import time

from kairyu.orchestration.router import RuleRouter

N_ROUTES = 10_000
QUERIES = (
    "What is the capital of France?",
    "Prove the theorem and explain your reasoning step by step.",
    "Fix this bug:\n```python\nx = [\n```\nwhy?",
    "First, research options. Then design. After that, implement. Finally, verify. " * 5,
)


def main() -> None:
    router = RuleRouter()
    durations = []
    for i in range(N_ROUTES):
        query = QUERIES[i % len(QUERIES)]
        start = time.perf_counter()
        router.route(query)
        durations.append(time.perf_counter() - start)
    durations.sort()
    p50 = statistics.median(durations)
    p99 = durations[int(len(durations) * 0.99)]
    print(f"routes={N_ROUTES}")
    print(f"router latency p50={p50 * 1e6:.1f}us p99={p99 * 1e6:.1f}us")
    print(f"budget: p99 < 10ms -> {'OK' if p99 < 0.010 else 'VIOLATED'}")


if __name__ == "__main__":
    main()
