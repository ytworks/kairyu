#!/usr/bin/env python3
"""Deterministic G5-F5b tenant-isolation gate (issue #191).

The protected and unprotected runs use the same two-tenant arrival trace and
one shared production Scheduler.  The noisy tenant offers 60 requests/minute
against a 6 requests/minute token bucket (exactly 10x quota); the good tenant
offers 15 requests/minute.  Admission runs before requests enter the shared
replica scheduler.

Run:
    uv run python bench/noisy_neighbor_bench.py --assert-gate
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
from kairyu.entrypoints.server.tenancy import (
    TenantConfig,
    TenantLimiter,
    TenantLimits,
)


@dataclass(frozen=True)
class Workload:
    measurement_seconds: int = 1_200
    good_release_period_s: int = 4
    noisy_release_period_s: int = 1
    good_requests_per_minute: int = 600
    noisy_requests_per_minute: int = 6
    tokens_per_minute: int = 1_000_000
    max_good_p99_increase_seconds: int = 1

    @property
    def good_offered_per_minute(self) -> float:
        return 60.0 / self.good_release_period_s

    @property
    def noisy_offered_per_minute(self) -> float:
        return 60.0 / self.noisy_release_period_s

    @property
    def noisy_offered_to_quota_ratio(self) -> float:
        return self.noisy_offered_per_minute / self.noisy_requests_per_minute

    def validate(self) -> None:
        integer_fields = (
            self.measurement_seconds,
            self.good_release_period_s,
            self.noisy_release_period_s,
            self.good_requests_per_minute,
            self.noisy_requests_per_minute,
            self.tokens_per_minute,
            self.max_good_p99_increase_seconds,
        )
        if any(type(value) is not int or value < 1 for value in integer_fields):
            raise ValueError("workload integer fields must be positive integers")
        if self.measurement_seconds % self.good_release_period_s:
            raise ValueError(
                "measurement_seconds must be divisible by good_release_period_s"
            )
        if self.measurement_seconds % self.noisy_release_period_s:
            raise ValueError(
                "measurement_seconds must be divisible by noisy_release_period_s"
            )
        if not math.isclose(self.noisy_offered_to_quota_ratio, 10.0):
            raise ValueError(
                "noisy tenant must offer exactly 10x request quota, got "
                f"{self.noisy_offered_to_quota_ratio}"
            )
        if self.good_offered_per_minute > self.good_requests_per_minute:
            raise ValueError("good tenant offered rate must remain within quota")


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _scheduler_evidence(scheduler: Scheduler) -> dict:
    snapshot = scheduler.priority_metrics_snapshot()
    return {
        "events": {
            f"{request_class}:{event}": value
            for (request_class, event), value in snapshot["events"].items()
        },
        "queue_depth": snapshot["queue_depth"],
        "queue_high_watermark": snapshot["queue_high_watermark"],
    }


def run_trace(mode: str, workload: Workload | None = None) -> dict:
    if mode not in {"control", "protected", "unprotected"}:
        raise ValueError(f"unknown trace mode {mode!r}")
    workload = workload or Workload()
    workload.validate()
    clock = {"seconds": 0.0}
    tenant_config = TenantConfig(
        limits={
            "good": TenantLimits(
                requests_per_minute=workload.good_requests_per_minute,
                tokens_per_minute=workload.tokens_per_minute,
                request_burst=workload.good_requests_per_minute,
                token_burst=workload.tokens_per_minute,
                max_in_flight=workload.good_requests_per_minute,
            ),
            "noisy": TenantLimits(
                requests_per_minute=workload.noisy_requests_per_minute,
                tokens_per_minute=workload.tokens_per_minute,
                request_burst=1,
                token_burst=workload.tokens_per_minute,
                max_in_flight=1,
            ),
        }
    )
    limiter = TenantLimiter(tenant_config, now=lambda: clock["seconds"])
    scheduler = Scheduler(
        RadixKVCache(num_pages=4_096, page_size=16),
        max_num_batched_tokens=1,
        max_num_seqs=8,
        page_size=16,
    )
    records: dict[str, dict] = {}
    schedule_events: list[dict] = []
    rejected_ids: set[str] = set()
    leases: dict[str, object] = {}
    measurement_state: dict | None = None

    def offer(tenant: str, index: int, *, enforce_admission: bool) -> None:
        request_id = f"{tenant}-{index}"
        admission = limiter.acquire(tenant) if enforce_admission else None
        admitted = admission is None or admission.admitted
        if admission is not None and admitted:
            admitted = admission.reserve_tokens(2)
        record = {
            "request_id": request_id,
            "tenant": tenant,
            "arrival_second": int(clock["seconds"]),
            "admitted": admitted,
            "rejected": not admitted,
            "admission_reason": (
                "bypassed" if admission is None else admission.reason
            ),
            "first_token_second": None,
            "completed_second": None,
        }
        records[request_id] = record
        if not admitted:
            if admission is not None:
                admission.release()
            rejected_ids.add(request_id)
            return
        if admission is not None:
            admission.mark_dispatched()
            leases[request_id] = admission
        token_id = (1 if tenant == "good" else 1_000_000) + index
        scheduler.add_request(
            EngineRequest(
                request_id,
                (token_id,),
                max_new_tokens=1,
                priority=0,
                scheduling_class="interactive",
            )
        )

    tick = 0
    offered_total = (
        workload.measurement_seconds // workload.good_release_period_s
        + (
            0
            if mode == "control"
            else workload.measurement_seconds // workload.noisy_release_period_s
        )
    )
    drain_limit = workload.measurement_seconds + offered_total * 2
    while tick < drain_limit:
        clock["seconds"] = float(tick)
        if tick < workload.measurement_seconds:
            if mode != "control" and tick % workload.noisy_release_period_s == 0:
                offer(
                    "noisy",
                    tick // workload.noisy_release_period_s,
                    enforce_admission=mode == "protected",
                )
            if tick % workload.good_release_period_s == 0:
                offer(
                    "good",
                    tick // workload.good_release_period_s,
                    enforce_admission=True,
                )

        waiting_depth_before = len(scheduler.waiting_ids)
        plan = scheduler.schedule()
        sampled: dict[str, int] = {}
        for chunk in plan.scheduled:
            record = records[chunk.request_id]
            state = scheduler.states[chunk.request_id]
            produces_token = (not chunk.is_prefill) or state.prefill_done
            if produces_token:
                if record["first_token_second"] is None:
                    record["first_token_second"] = tick + 1
                sampled[chunk.request_id] = 1
            schedule_events.append(
                {
                    "second": tick,
                    "request_id": chunk.request_id,
                    "tenant": record["tenant"],
                    "waiting_depth_before": waiting_depth_before,
                    "num_tokens": chunk.num_tokens,
                    "is_prefill": chunk.is_prefill,
                }
            )
        finished = scheduler.update(sampled) if sampled else ()
        for request_id in finished:
            records[request_id]["completed_second"] = tick + 1
            scheduler.forget(request_id)
            admission = leases.pop(request_id, None)
            if admission is not None:
                admission.settle_tokens(2, exact=True)
                admission.release()

        tick += 1
        if tick == workload.measurement_seconds:
            incomplete = [
                record
                for record in records.values()
                if record["admitted"] and record["completed_second"] is None
            ]
            measurement_state = {
                "admitted_incomplete": len(incomplete),
                "good_incomplete": sum(
                    record["tenant"] == "good" for record in incomplete
                ),
                "noisy_incomplete": sum(
                    record["tenant"] == "noisy" for record in incomplete
                ),
            }
        if tick >= workload.measurement_seconds and not scheduler.has_unfinished():
            break
    else:
        raise RuntimeError(f"{mode} trace did not drain")

    if measurement_state is None:
        raise AssertionError("measurement state was not captured")
    if leases:
        raise AssertionError(f"tenant admission leases did not drain: {sorted(leases)}")
    raw = list(records.values())
    for record in raw:
        first = record["first_token_second"]
        record["ttft_seconds"] = (
            None if first is None else first - record["arrival_second"]
        )
    scheduled_ids = {event["request_id"] for event in schedule_events}
    summaries = {}
    for tenant in ("good", "noisy"):
        tenant_records = [record for record in raw if record["tenant"] == tenant]
        admitted = [record for record in tenant_records if record["admitted"]]
        ttfts = [int(record["ttft_seconds"]) for record in admitted]
        summaries[tenant] = {
            "offered": len(tenant_records),
            "admitted": len(admitted),
            "rejected": len(tenant_records) - len(admitted),
            "completed": sum(
                record["completed_second"] is not None for record in admitted
            ),
            "ttft_p50_seconds": percentile(ttfts, 0.50) if ttfts else None,
            "ttft_p95_seconds": percentile(ttfts, 0.95) if ttfts else None,
            "ttft_p99_seconds": percentile(ttfts, 0.99) if ttfts else None,
        }
    scheduler_evidence = _scheduler_evidence(scheduler)
    return {
        "mode": mode,
        "tenants": summaries,
        "shared_replica": {
            "capacity_tokens_per_second": 1,
            "admitted_requests": sum(record["admitted"] for record in raw),
            "completed_requests": sum(
                record["completed_second"] is not None for record in raw
            ),
            "incomplete_at_measurement_end": measurement_state,
            "drain_seconds": tick,
            "scheduler": scheduler_evidence,
            "rejected_request_ids_scheduled": sorted(rejected_ids & scheduled_ids),
            "tenant_reservations_after_drain": limiter.reservation_snapshot(),
        },
        "requests": raw,
        "schedule_events": schedule_events,
    }


def run_gate(workload: Workload | None = None) -> dict:
    workload = workload or Workload()
    workload.validate()
    control = run_trace("control", workload)
    protected = run_trace("protected", workload)
    unprotected = run_trace("unprotected", workload)
    control_good_p99 = control["tenants"]["good"]["ttft_p99_seconds"]
    protected_good_p99 = protected["tenants"]["good"]["ttft_p99_seconds"]
    unprotected_good_p99 = unprotected["tenants"]["good"]["ttft_p99_seconds"]
    assert control_good_p99 is not None
    assert protected_good_p99 is not None
    assert unprotected_good_p99 is not None
    protected_total_admitted = sum(
        protected["tenants"][tenant]["admitted"] for tenant in ("good", "noisy")
    )
    protected_scheduler = protected["shared_replica"]["scheduler"]
    checks = {
        "noisy_offers_exactly_10x_quota": math.isclose(
            workload.noisy_offered_to_quota_ratio,
            10.0,
        ),
        "good_stays_within_quota": (
            workload.good_offered_per_minute <= workload.good_requests_per_minute
        ),
        "noisy_excess_rejected_before_replica": (
            protected["tenants"]["noisy"]["rejected"] > 0
            and not protected["shared_replica"]["rejected_request_ids_scheduled"]
        ),
        "shared_replica_enqueue_matches_admission": (
            protected_scheduler["events"]["interactive:enqueue"]
            == protected_total_admitted
        ),
        "shared_replica_completion_matches_admission": (
            protected_scheduler["events"]["interactive:complete"]
            == protected_total_admitted
            == protected["shared_replica"]["completed_requests"]
        ),
        "good_all_completed": (
            protected["tenants"]["good"]["completed"]
            == protected["tenants"]["good"]["offered"]
        ),
        "good_p99_not_materially_degraded": (
            protected_good_p99
            <= control_good_p99 + workload.max_good_p99_increase_seconds
        ),
        "admission_improves_good_p99": (
            protected_good_p99 < unprotected_good_p99
        ),
        "admission_bounds_shared_queue": (
            protected_scheduler["queue_high_watermark"]["interactive"]
            < unprotected["shared_replica"]["scheduler"]["queue_high_watermark"][
                "interactive"
            ]
        ),
        "protected_trace_drains": (
            protected["shared_replica"]["incomplete_at_measurement_end"][
                "admitted_incomplete"
            ]
            == 0
        ),
        "tenant_reservations_drain": all(
            tokens == 0
            for tokens in protected["shared_replica"][
                "tenant_reservations_after_drain"
            ].values()
        ),
    }
    return {
        "schema_version": 1,
        "gate": "G5-F5b",
        "workload": asdict(workload),
        "rates": {
            "good_offered_per_minute": workload.good_offered_per_minute,
            "noisy_offered_per_minute": workload.noisy_offered_per_minute,
            "noisy_quota_per_minute": workload.noisy_requests_per_minute,
            "noisy_offered_to_quota_ratio": (
                workload.noisy_offered_to_quota_ratio
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "control": control,
        "protected": protected,
        "unprotected": unprotected,
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
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
