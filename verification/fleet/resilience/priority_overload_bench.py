#!/usr/bin/env python3
"""Deterministic F5a 2x-overload priority/TTFT gate (issue #190).

The benchmark drives the production ``Scheduler`` with a synthetic service
clock. Interactive work consumes 25% of capacity; batch offers the remaining
175%, for exactly 2x aggregate load. Batch arrives first at every release so
the FCFS control is intentionally adverse but uses the identical trace.

Run:
    uv run python verification/fleet/resilience/priority_overload_bench.py --assert-gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler


@dataclass(frozen=True)
class Workload:
    measurement_ticks: int = 400
    release_period_ticks: int = 4
    interactive_tokens: int = 1
    batch_tokens: int = 7
    interactive_priority: int = 0
    batch_priority: int = 1
    ttft_slo_ticks: int = 2

    @property
    def releases(self) -> int:
        return self.measurement_ticks // self.release_period_ticks

    @property
    def offered_tokens(self) -> int:
        return self.releases * (self.interactive_tokens + self.batch_tokens)

    @property
    def offered_load_ratio(self) -> float:
        return self.offered_tokens / self.measurement_ticks

    def validate(self) -> None:
        if self.measurement_ticks < 1 or self.release_period_ticks < 1:
            raise ValueError("measurement ticks and release period must be positive")
        if self.measurement_ticks % self.release_period_ticks:
            raise ValueError("measurement ticks must be divisible by release period")
        if min(self.interactive_tokens, self.batch_tokens) < 1:
            raise ValueError("request token costs must be positive")
        if self.offered_load_ratio != 2.0:
            raise ValueError(
                f"workload must offer exactly 2x capacity, got {self.offered_load_ratio}"
            )


def percentile(values: list[int], quantile: float) -> int:
    """Nearest-rank percentile, suitable for an SLO gate and raw integer ticks."""

    if not values:
        raise ValueError("cannot compute a percentile of no values")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _request_tokens(index: int, count: int, *, class_offset: int) -> tuple[int, ...]:
    start = class_offset + index * (count + 1)
    return tuple(range(start, start + count))


def run_trace(*, priority_enabled: bool, workload: Workload | None = None) -> dict:
    workload = workload or Workload()
    workload.validate()
    clock = {"tick": 0.0}
    scheduler = Scheduler(
        RadixKVCache(num_pages=4_096, page_size=16),
        max_num_batched_tokens=1,
        max_num_seqs=8,
        page_size=16,
        priority_age_s=10_000.0 if priority_enabled else None,
        clock=lambda: clock["tick"],
    )
    records: dict[str, dict] = {}
    schedule_events: list[dict] = []
    max_waiting_depth = 0
    measurement_state: dict | None = None

    def add_request(request_id: str, class_name: str, tokens: int, priority: int) -> None:
        records[request_id] = {
            "request_id": request_id,
            "class": class_name,
            "priority": priority,
            "arrival_tick": int(clock["tick"]),
            "service_tokens": tokens,
            "first_token_tick": None,
            "completed_tick": None,
        }
        scheduler.add_request(
            EngineRequest(
                request_id,
                _request_tokens(
                    int(request_id.split("-")[-1]),
                    tokens,
                    class_offset=1 if class_name == "interactive" else 1_000_000,
                ),
                max_new_tokens=1,
                priority=priority,
                scheduling_class=class_name,
            )
        )

    tick = 0
    drain_limit = workload.measurement_ticks + workload.offered_tokens * 4
    while tick < drain_limit:
        clock["tick"] = float(tick)
        if tick < workload.measurement_ticks and tick % workload.release_period_ticks == 0:
            release = tick // workload.release_period_ticks
            # Batch arrives first, so FCFS and priority consume one identical trace.
            add_request(
                f"batch-{release}",
                "batch",
                workload.batch_tokens,
                workload.batch_priority,
            )
            add_request(
                f"interactive-{release}",
                "interactive",
                workload.interactive_tokens,
                workload.interactive_priority,
            )

        max_waiting_depth = max(max_waiting_depth, len(scheduler.waiting_ids))
        waiting_depth_before = len(scheduler.waiting_ids)
        plan = scheduler.schedule()
        sampled: dict[str, int] = {}
        for chunk in plan.scheduled:
            record = records[chunk.request_id]
            state = scheduler.states[chunk.request_id]
            produces_token = (not chunk.is_prefill) or state.prefill_done
            if produces_token:
                if record["first_token_tick"] is None:
                    record["first_token_tick"] = tick + 1
                sampled[chunk.request_id] = 1
            schedule_events.append(
                {
                    "tick": tick,
                    "request_id": chunk.request_id,
                    "class": record["class"],
                    "priority": record["priority"],
                    "is_prefill": chunk.is_prefill,
                    "num_tokens": chunk.num_tokens,
                    "waiting_depth_before": waiting_depth_before,
                }
            )
        finished = scheduler.update(sampled) if sampled else ()
        for request_id in finished:
            records[request_id]["completed_tick"] = tick + 1
            scheduler.forget(request_id)

        tick += 1
        if tick == workload.measurement_ticks:
            completed = sum(
                record["completed_tick"] is not None for record in records.values()
            )
            measurement_state = {
                "completed": completed,
                "unfinished": len(records) - completed,
            }
        if tick >= workload.measurement_ticks and not scheduler.has_unfinished():
            break
    else:
        raise RuntimeError("priority overload trace did not drain")

    if measurement_state is None:
        raise AssertionError("measurement state was not captured")
    raw = list(records.values())
    for record in raw:
        first = record["first_token_tick"]
        record["ttft_ticks"] = (
            None if first is None else first - record["arrival_tick"]
        )
    interactive = [record for record in raw if record["class"] == "interactive"]
    batch = [record for record in raw if record["class"] == "batch"]
    interactive_ttft = [int(record["ttft_ticks"]) for record in interactive]
    batch_service_in_window = sum(
        event["num_tokens"]
        for event in schedule_events
        if event["class"] == "batch" and event["tick"] < workload.measurement_ticks
    )
    batch_completed_in_window = sum(
        record["completed_tick"] is not None
        and record["completed_tick"] <= workload.measurement_ticks
        for record in batch
    )
    scheduler_snapshot = scheduler.priority_metrics_snapshot()
    scheduler_evidence = {
        "events": {
            f"{request_class}:{event}": value
            for (request_class, event), value in scheduler_snapshot["events"].items()
        },
        "queue_depth": scheduler_snapshot["queue_depth"],
        "queue_high_watermark": scheduler_snapshot["queue_high_watermark"],
    }
    return {
        "policy": "priority" if priority_enabled else "fcfs",
        "offered_load_ratio": workload.offered_load_ratio,
        "capacity_tokens_per_tick": 1,
        "measurement_ticks": workload.measurement_ticks,
        "offered_requests": len(raw),
        "offered_tokens": workload.offered_tokens,
        "interactive": {
            "offered": len(interactive),
            "completed": sum(record["completed_tick"] is not None for record in interactive),
            "ttft_p50_ticks": percentile(interactive_ttft, 0.50),
            "ttft_p95_ticks": percentile(interactive_ttft, 0.95),
            "ttft_p99_ticks": percentile(interactive_ttft, 0.99),
            "ttft_slo_ticks": workload.ttft_slo_ticks,
            "slo_attainment": (
                sum(value <= workload.ttft_slo_ticks for value in interactive_ttft)
                / len(interactive_ttft)
            ),
        },
        "batch": {
            "offered": len(batch),
            "completed_in_measurement": batch_completed_in_window,
            "service_tokens_in_measurement": batch_service_in_window,
            "backlog_at_measurement_end": measurement_state["unfinished"],
            "completed_after_drain": sum(
                record["completed_tick"] is not None for record in batch
            ),
        },
        "accounting": {
            "completed_at_measurement_end": measurement_state["completed"],
            "queued_at_measurement_end": measurement_state["unfinished"],
            "offered_equals_completed_plus_queued": (
                len(raw)
                == measurement_state["completed"] + measurement_state["unfinished"]
            ),
            "max_waiting_depth": max_waiting_depth,
            "drain_ticks": tick,
        },
        "scheduler": scheduler_evidence,
        "requests": raw,
        "schedule_events": schedule_events,
    }


def run_gate(workload: Workload | None = None) -> dict:
    workload = workload or Workload()
    fcfs = run_trace(priority_enabled=False, workload=workload)
    priority = run_trace(priority_enabled=True, workload=workload)
    checks = {
        "exact_2x_offered_load": priority["offered_load_ratio"] == 2.0,
        "interactive_all_completed": (
            priority["interactive"]["completed"] == priority["interactive"]["offered"]
        ),
        "interactive_p99_slo": (
            priority["interactive"]["ttft_p99_ticks"] <= workload.ttft_slo_ticks
        ),
        "interactive_99pct_slo_attainment": (
            priority["interactive"]["slo_attainment"] >= 0.99
        ),
        "priority_improves_interactive_p99": (
            priority["interactive"]["ttft_p99_ticks"]
            < fcfs["interactive"]["ttft_p99_ticks"]
        ),
        "batch_uses_residual_capacity": (
            priority["batch"]["service_tokens_in_measurement"] > 0
        ),
        "batch_retains_overload_backlog": (
            priority["batch"]["backlog_at_measurement_end"] > 0
        ),
        "batch_drains_after_measurement": (
            priority["batch"]["completed_after_drain"] == priority["batch"]["offered"]
        ),
        "request_accounting_reconciles": (
            priority["accounting"]["offered_equals_completed_plus_queued"]
        ),
        "scheduler_class_evidence_reconciles": (
            priority["scheduler"]["events"]["interactive:enqueue"]
            == priority["interactive"]["offered"]
            == priority["scheduler"]["events"]["interactive:admit"]
            == priority["scheduler"]["events"]["interactive:complete"]
            and priority["scheduler"]["events"]["batch:enqueue"]
            == priority["batch"]["offered"]
            == priority["scheduler"]["events"]["batch:admit"]
            == priority["scheduler"]["events"]["batch:complete"]
        ),
    }
    return {
        "schema_version": 1,
        "gate": "G5-F5a",
        "workload": asdict(workload),
        "checks": checks,
        "passed": all(checks.values()),
        "fcfs": fcfs,
        "priority": priority,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args()
    result = run_gate()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    result["environment"] = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "tracked_dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "benchmark_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if args.assert_gate and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
