#!/usr/bin/env python3
"""Compare correct fleet-usage responsibility boundaries (issue #194).

Candidate A keeps request-path writes gateway-local and asynchronously drains
them. Candidate B synchronously commits every request to one shared SQLite
store. SQLite is an intentionally favorable same-host stand-in for a remote
shared database: no network hop is included. Both candidates retain raw rows
and produce the same totals.

Run:
    uv run python bench/usage_architecture_bench.py \
      --records 30000 --gateways 3 \
      --output bench/results/usage-architecture-2026-07-27.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from kairyu.entrypoints.server.tenancy import UsageLedger
from kairyu.usage_reconciliation import aggregate_gateway_ledgers


def _percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _latency_summary(durations_ns: list[int]) -> dict[str, float]:
    return {
        "p50_us": _percentile(durations_ns, 0.50) / 1_000,
        "p95_us": _percentile(durations_ns, 0.95) / 1_000,
        "p99_us": _percentile(durations_ns, 0.99) / 1_000,
        "max_us": max(durations_ns) / 1_000,
    }


def _record(index: int) -> tuple[str, str, int, int, int]:
    prompt = 128 + index % 64
    return (
        f"tenant-{index % 8}",
        "qwen",
        prompt,
        16 + index % 8,
        min(prompt, (index % 5) * 32),
    )


def _local_async(
    root: Path,
    *,
    records: int,
    gateways: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    ledgers = {
        f"gateway-{index}": UsageLedger(
            root / f"gateway-{index}.jsonl",
            max_pending_records=records + gateways,
            batch_size=batch_size,
        )
        for index in range(gateways)
    }

    def produce(gateway_index: int) -> list[int]:
        durations = []
        ledger = ledgers[f"gateway-{gateway_index}"]
        for index in range(gateway_index, records, gateways):
            row = _record(index)
            started = time.perf_counter_ns()
            ledger.record(*row)
            durations.append(time.perf_counter_ns() - started)
        return durations

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=gateways) as executor:
        durations = [
            duration
            for result in executor.map(produce, range(gateways))
            for duration in result
        ]
    producer_complete = time.perf_counter()
    for ledger in ledgers.values():
        ledger.close()
    drain_complete = time.perf_counter()
    paths = {
        gateway: root / f"{gateway}.jsonl"
        for gateway in ledgers
    }
    totals = aggregate_gateway_ledgers(paths)
    return (
        {
            **_latency_summary(durations),
            "producer_seconds": producer_complete - started,
            "drain_complete_seconds": drain_complete - started,
            "producer_records_per_second": records / (producer_complete - started),
            "drain_records_per_second": records / (drain_complete - started),
        },
        totals,
    )


def _shared_sync(
    path: Path,
    *,
    records: int,
    gateways: int,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            """
            CREATE TABLE usage (
                gateway TEXT NOT NULL,
                tenant TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL
            )
            """
        )

    def produce(gateway_index: int) -> list[int]:
        durations = []
        connection = sqlite3.connect(path, timeout=30)
        connection.execute("PRAGMA synchronous=OFF")
        try:
            for index in range(gateway_index, records, gateways):
                row = _record(index)
                started = time.perf_counter_ns()
                connection.execute(
                    "INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?)",
                    (f"gateway-{gateway_index}", *row),
                )
                connection.commit()
                durations.append(time.perf_counter_ns() - started)
        finally:
            connection.close()
        return durations

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=gateways) as executor:
        durations = [
            duration
            for result in executor.map(produce, range(gateways))
            for duration in result
        ]
    complete = time.perf_counter()
    totals: dict[str, dict[str, int]] = {}
    with sqlite3.connect(path) as connection:
        for tenant, requests, prompt, completion, cached in connection.execute(
            """
            SELECT tenant, COUNT(*), SUM(prompt_tokens),
                   SUM(completion_tokens), SUM(cached_tokens)
            FROM usage GROUP BY tenant ORDER BY tenant
            """
        ):
            totals[tenant] = {
                "requests": requests,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cached_tokens": cached,
            }
    return (
        {
            **_latency_summary(durations),
            "producer_seconds": complete - started,
            "drain_complete_seconds": complete - started,
            "producer_records_per_second": records / (complete - started),
            "drain_records_per_second": records / (complete - started),
        },
        totals,
    )


def run_benchmark(
    *,
    records: int,
    gateways: int,
    batch_size: int,
    repeats: int,
) -> dict[str, Any]:
    runs = []
    for repeat in range(repeats):
        with tempfile.TemporaryDirectory(
            prefix=f"kairyu-usage-arch-{repeat}-"
        ) as directory:
            root = Path(directory)
            local, local_totals = _local_async(
                root / "local",
                records=records,
                gateways=gateways,
                batch_size=batch_size,
            )
            shared, shared_totals = _shared_sync(
                root / "shared.sqlite3",
                records=records,
                gateways=gateways,
            )
        if local_totals != shared_totals:
            raise AssertionError("architecture candidates produced different totals")
        runs.append(
            {
                "gateway_local_async": local,
                "shared_sync_commit": shared,
            }
        )

    def medians(candidate: str) -> dict[str, float]:
        return {
            field: statistics.median(run[candidate][field] for run in runs)
            for field in runs[0][candidate]
        }

    local = medians("gateway_local_async")
    shared = medians("shared_sync_commit")
    selected = (
        "gateway_local_async"
        if local["p99_us"] < shared["p99_us"]
        else "shared_sync_commit"
    )
    return {
        "schema_version": 1,
        "records": records,
        "gateways": gateways,
        "batch_size": batch_size,
        "repeats": repeats,
        "aggregation": "median of independent runs",
        "shared_store": {
            "implementation": "same-host SQLite WAL, synchronous=OFF",
            "network_latency_included": False,
        },
        "gateway_local_async": local,
        "shared_sync_commit": shared,
        "runs": runs,
        "producer_p99_speedup": shared["p99_us"] / local["p99_us"],
        "producer_throughput_speedup": (
            local["producer_records_per_second"]
            / shared["producer_records_per_second"]
        ),
        "selected": selected,
        "selection_rule": (
            "lowest request-path p99 among candidates that preserve raw rows "
            "and exact fleet totals"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=30_000)
    parser.add_argument("--gateways", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.records < 1
        or args.gateways < 1
        or args.batch_size < 1
        or args.repeats < 1
    ):
        parser.error("records, gateways, batch-size, and repeats must be positive")
    result = run_benchmark(
        records=args.records,
        gateways=args.gateways,
        batch_size=args.batch_size,
        repeats=args.repeats,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
