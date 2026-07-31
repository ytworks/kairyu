"""Orchestration overhead harness on the mock backend (no model inference).

Measures the pure framework overhead of routing + DAG execution under
concurrency. Real engine comparisons (vLLM/SGLang, TTFT/TPOT/goodput) land in
M2 with GPU hardware; this script only reports what it actually measured.

Run: uv run python bench/orchestration_mock_bench.py
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections.abc import Sequence

from kairyu.engine.mock import MockBackend
from kairyu.orchestration.orchestrator import Orchestrator

CONCURRENCY = 128
SIMULATED_ENGINE_LATENCY_S = 0.02
QUERY = "First, plan the task. Then execute it. Finally, verify the result."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help=f"number of concurrent requests (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--simulated-engine-latency",
        type=float,
        default=SIMULATED_ENGINE_LATENCY_S,
        metavar="SECONDS",
        help=(
            "mock latency for each engine call "
            f"(default: {SIMULATED_ENGINE_LATENCY_S})"
        ),
    )
    return parser


async def _run(
    *,
    concurrency: int,
    simulated_engine_latency_s: float,
) -> None:
    engines = {
        "tier1": MockBackend(latency_s=simulated_engine_latency_s),
        "tier2": MockBackend(
            latency_s=simulated_engine_latency_s, responses={"[verifier]": "PASS"}
        ),
    }
    orchestrator = Orchestrator(engines=engines)

    async def one_request() -> float:
        start = time.perf_counter()
        await orchestrator.run(QUERY)
        return time.perf_counter() - start

    wall_start = time.perf_counter()
    latencies = await asyncio.gather(*(one_request() for _ in range(concurrency)))
    wall = time.perf_counter() - wall_start
    latencies = sorted(latencies)
    print(
        f"concurrency={concurrency} "
        f"simulated_engine_latency={simulated_engine_latency_s}s"
    )
    print(f"wall_time={wall:.3f}s throughput={concurrency / wall:.1f} req/s (mock)")
    print(
        f"per-request latency p50={statistics.median(latencies) * 1e3:.1f}ms "
        f"p99={latencies[int(len(latencies) * 0.99)] * 1e3:.1f}ms"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.simulated_engine_latency < 0:
        parser.error("--simulated-engine-latency must be non-negative")
    asyncio.run(
        _run(
            concurrency=args.concurrency,
            simulated_engine_latency_s=args.simulated_engine_latency,
        )
    )


if __name__ == "__main__":
    main()
