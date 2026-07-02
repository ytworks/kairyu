"""Orchestration overhead harness on the mock backend (no model inference).

Measures the pure framework overhead of routing + DAG execution under
concurrency. Real engine comparisons (vLLM/SGLang, TTFT/TPOT/goodput) land in
M2 with GPU hardware; this script only reports what it actually measured.

Run: uv run python bench/orchestration_mock_bench.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from kairyu.engine.mock import MockBackend
from kairyu.orchestration.orchestrator import Orchestrator

CONCURRENCY = 128
SIMULATED_ENGINE_LATENCY_S = 0.02
QUERY = "First, plan the task. Then execute it. Finally, verify the result."


async def main() -> None:
    engines = {
        "tier1": MockBackend(latency_s=SIMULATED_ENGINE_LATENCY_S),
        "tier2": MockBackend(
            latency_s=SIMULATED_ENGINE_LATENCY_S, responses={"[verifier]": "PASS"}
        ),
    }
    orchestrator = Orchestrator(engines=engines)

    async def one_request() -> float:
        start = time.perf_counter()
        await orchestrator.run(QUERY)
        return time.perf_counter() - start

    wall_start = time.perf_counter()
    latencies = await asyncio.gather(*(one_request() for _ in range(CONCURRENCY)))
    wall = time.perf_counter() - wall_start
    latencies = sorted(latencies)
    print(f"concurrency={CONCURRENCY} simulated_engine_latency={SIMULATED_ENGINE_LATENCY_S}s")
    print(f"wall_time={wall:.3f}s throughput={CONCURRENCY / wall:.1f} req/s (mock)")
    print(
        f"per-request latency p50={statistics.median(latencies) * 1e3:.1f}ms "
        f"p99={latencies[int(len(latencies) * 0.99)] * 1e3:.1f}ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
