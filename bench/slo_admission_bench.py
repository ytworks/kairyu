#!/usr/bin/env python3
"""Deterministic G5-F5c SLO-admission gate (issue #192).

The benchmark gives queue-and-hope and the production ``AdmissionController``
the same immutable arrivals and service requirements.  Actual TTFT comes only
from the production ``Scheduler``.  A post-run forced-admit Scheduler replay
supplies independent counterfactual labels for decisions the treatment did not
admit; predictor output is never used as its own oracle.

Run:
    uv run python bench/slo_admission_bench.py --assert-gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.entrypoints.server.slo import AdmissionController

_CAPACITY_TOKENS_PER_TICK = 1
_INTERACTIVE_PRIORITY = 0
_BATCH_PRIORITY = 1
_PRIORITY_AGE_TICKS = 10_000.0


@dataclass(frozen=True)
class Scenario:
    name: str
    measurement_ticks: int
    release_period_ticks: int
    requests_per_release: int
    service_tokens: int = 1
    ttft_slo_ticks: int = 2

    @property
    def releases(self) -> int:
        return self.measurement_ticks // self.release_period_ticks

    @property
    def offered_requests(self) -> int:
        return self.releases * self.requests_per_release

    @property
    def offered_tokens(self) -> int:
        return self.offered_requests * self.service_tokens

    @property
    def offered_load_ratio(self) -> float:
        return (
            self.offered_tokens
            / self.measurement_ticks
            / _CAPACITY_TOKENS_PER_TICK
        )

    @property
    def saturated(self) -> bool:
        return self.offered_load_ratio > 1.0

    def validate(self) -> None:
        integer_fields = (
            self.measurement_ticks,
            self.release_period_ticks,
            self.requests_per_release,
            self.service_tokens,
            self.ttft_slo_ticks,
        )
        if any(type(value) is not int or value < 1 for value in integer_fields):
            raise ValueError("scenario integer fields must be positive integers")
        if self.measurement_ticks % self.release_period_ticks:
            raise ValueError(
                "measurement_ticks must be divisible by release_period_ticks"
            )


FORMAL_SCENARIOS = (
    Scenario(
        name="underload",
        measurement_ticks=400,
        release_period_ticks=2,
        requests_per_release=1,
    ),
    Scenario(
        name="smooth_2x",
        measurement_ticks=400,
        release_period_ticks=1,
        requests_per_release=2,
    ),
    Scenario(
        name="bursty_2x",
        measurement_ticks=400,
        release_period_ticks=4,
        requests_per_release=8,
    ),
)


def build_trace(scenario: Scenario) -> tuple[dict, ...]:
    scenario.validate()
    trace = []
    index = 0
    for arrival_tick in range(
        0,
        scenario.measurement_ticks,
        scenario.release_period_ticks,
    ):
        for release_order in range(scenario.requests_per_release):
            trace.append(
                {
                    "request_id": f"{scenario.name}-{index:05d}",
                    "index": index,
                    "arrival_tick": arrival_tick,
                    "release_order": release_order,
                    "service_tokens": scenario.service_tokens,
                }
            )
            index += 1
    return tuple(trace)


def _trace_sha256(trace: tuple[dict, ...]) -> str:
    rendered = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def _new_scheduler(clock: dict[str, float]) -> Scheduler:
    return Scheduler(
        RadixKVCache(num_pages=4_096, page_size=16),
        max_num_batched_tokens=_CAPACITY_TOKENS_PER_TICK,
        max_num_seqs=256,
        page_size=16,
        priority_age_s=_PRIORITY_AGE_TICKS,
        clock=lambda: clock["tick"],
    )


def _engine_request(item: dict, request_class: str) -> EngineRequest:
    priority = (
        _INTERACTIVE_PRIORITY
        if request_class == "interactive"
        else _BATCH_PRIORITY
    )
    start = 1 + item["index"] * (item["service_tokens"] + 1)
    return EngineRequest(
        request_id=item["request_id"],
        prompt_token_ids=tuple(
            range(start, start + item["service_tokens"])
        ),
        max_new_tokens=1,
        priority=priority,
        scheduling_class=request_class,
    )


def _queue_snapshot(scheduler: Scheduler) -> dict:
    waiting_ids = scheduler.waiting_ids
    waiting = Counter(
        scheduler.states[request_id].request.scheduling_class
        for request_id in waiting_ids
    )
    active = Counter(
        state.request.scheduling_class
        for state in scheduler.states.values()
    )
    running = {
        request_class: active[request_class] - waiting[request_class]
        for request_class in ("interactive", "batch")
    }
    return {
        "waiting": {
            "total": len(waiting_ids),
            "interactive": waiting["interactive"],
            "batch": waiting["batch"],
        },
        "running": {
            "total": sum(running.values()),
            **running,
        },
        "active": {
            "total": len(scheduler.states),
            "interactive": active["interactive"],
            "batch": active["batch"],
        },
    }


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


def _empty_record(item: dict) -> dict:
    return {
        **item,
        "action": None,
        "reason": "",
        "predicted_ttft_ticks": None,
        "controller_in_flight_before": None,
        "predictor_snapshot_before": None,
        "queue_before_decision": None,
        "scheduler_class": None,
        "enqueue_tick": None,
        "first_token_tick": None,
        "completed_tick": None,
        "ttft_ticks": None,
        "actual_ttft_meets_slo": False,
        "interactive_goodput_success": False,
    }


def run_trace(
    mode: str,
    scenario: Scenario,
    trace: tuple[dict, ...] | None = None,
) -> dict:
    if mode not in {"queue_and_hope", "slo_admission"}:
        raise ValueError(f"unknown trace mode {mode!r}")
    scenario.validate()
    trace = trace or build_trace(scenario)
    clock = {"tick": 0.0}
    scheduler = _new_scheduler(clock)
    controller = (
        AdmissionController(
            ttft_slo_s=float(scenario.ttft_slo_ticks),
            now=lambda: clock["tick"],
        )
        if mode == "slo_admission"
        else None
    )
    by_arrival: dict[int, list[dict]] = {}
    records = {}
    for item in trace:
        by_arrival.setdefault(item["arrival_tick"], []).append(item)
        records[item["request_id"]] = _empty_record(item)

    leases = {}
    tick_events = []
    schedule_events = []
    measurement_state = None
    tick = 0
    drain_limit = scenario.measurement_ticks + scenario.offered_tokens * 2 + 10
    while tick < drain_limit:
        clock["tick"] = float(tick)
        queue_before_arrivals = _queue_snapshot(scheduler)
        arrivals = []
        if tick < scenario.measurement_ticks:
            for item in by_arrival.get(tick, ()):
                request_id = item["request_id"]
                record = records[request_id]
                record["queue_before_decision"] = _queue_snapshot(scheduler)
                if controller is None:
                    action = "admit"
                    predicted = None
                    in_flight_before = None
                    predictor_snapshot = None
                    reason = "queue_and_hope"
                else:
                    snapshot = controller.snapshot()
                    predictor_snapshot = asdict(snapshot)
                    in_flight_before = snapshot.in_flight
                    lease = controller.begin()
                    decision = lease.decision
                    action = decision.action
                    predicted = decision.predicted_ttft_s
                    reason = decision.reason
                    if lease.active:
                        leases[request_id] = lease
                record["action"] = action
                record["predicted_ttft_ticks"] = predicted
                record["controller_in_flight_before"] = in_flight_before
                record["predictor_snapshot_before"] = predictor_snapshot
                record["reason"] = reason

                request_class = None
                if action == "admit":
                    request_class = "interactive"
                elif action == "defer":
                    request_class = "batch"
                elif action != "shed":
                    raise AssertionError(f"unexpected admission action {action!r}")
                if request_class is not None:
                    scheduler.add_request(_engine_request(item, request_class))
                    record["scheduler_class"] = request_class
                    record["enqueue_tick"] = tick
                arrivals.append(
                    {
                        "request_id": request_id,
                        "predicted_ttft_ticks": predicted,
                        "controller_in_flight_before": in_flight_before,
                        "action": action,
                        "scheduler_class": request_class,
                    }
                )

        queue_before_schedule = _queue_snapshot(scheduler)
        plan = scheduler.schedule()
        sampled = {}
        first_tokens = []
        for chunk in plan.scheduled:
            state = scheduler.states[chunk.request_id]
            record = records[chunk.request_id]
            produces_token = (not chunk.is_prefill) or state.prefill_done
            if produces_token:
                if record["first_token_tick"] is None:
                    record["first_token_tick"] = tick + 1
                    first_tokens.append(chunk.request_id)
                sampled[chunk.request_id] = 1
            event = {
                "tick": tick,
                "request_id": chunk.request_id,
                "class": state.request.scheduling_class,
                "priority": state.request.priority,
                "num_tokens": chunk.num_tokens,
                "is_prefill": chunk.is_prefill,
                "queue_before_schedule": queue_before_schedule["waiting"][
                    "total"
                ],
            }
            schedule_events.append(event)

        finished = scheduler.update(sampled) if sampled else ()
        clock["tick"] = float(tick + 1)
        if controller is not None:
            for request_id in first_tokens:
                leases[request_id].finished_first_token()
        for request_id in finished:
            records[request_id]["completed_tick"] = tick + 1
            if controller is not None:
                leases.pop(request_id).completed()
            scheduler.forget(request_id)

        queue_after = _queue_snapshot(scheduler)
        tick_events.append(
            {
                "tick": tick,
                "queue_before_arrivals": queue_before_arrivals,
                "arrivals": arrivals,
                "queue_before_schedule": queue_before_schedule,
                "scheduled": [
                    event
                    for event in schedule_events[-len(plan.scheduled) :]
                ]
                if plan.scheduled
                else [],
                "first_tokens": first_tokens,
                "completed": list(finished),
                "queue_after": queue_after,
                "controller_in_flight_after": (
                    None if controller is None else controller.in_flight
                ),
            }
        )

        tick += 1
        if tick == scenario.measurement_ticks:
            measurement_state = {
                "queue": _queue_snapshot(scheduler),
                "completed": sum(
                    record["completed_tick"] is not None
                    for record in records.values()
                ),
            }
        if tick >= scenario.measurement_ticks and not scheduler.has_unfinished():
            break
    else:
        raise RuntimeError(f"{mode} trace did not drain")

    if measurement_state is None:
        raise AssertionError("measurement state was not captured")
    if leases:
        raise AssertionError(
            f"admission leases did not drain: {sorted(leases)}"
        )
    for record in records.values():
        first_token_tick = record["first_token_tick"]
        record["ttft_ticks"] = (
            None
            if first_token_tick is None
            else first_token_tick - record["arrival_tick"]
        )
        record["actual_ttft_meets_slo"] = (
            record["completed_tick"] is not None
            and record["ttft_ticks"] is not None
            and record["ttft_ticks"] <= scenario.ttft_slo_ticks
        )
        record["interactive_goodput_success"] = (
            record["action"] == "admit"
            and record["scheduler_class"] == "interactive"
            and record["actual_ttft_meets_slo"]
        )

    action_counts = Counter(record["action"] for record in records.values())
    goodput_count = sum(
        record["interactive_goodput_success"]
        for record in records.values()
    )
    deferred_completed = sum(
        record["action"] == "defer"
        and record["completed_tick"] is not None
        for record in records.values()
    )
    return {
        "mode": mode,
        "trace_sha256": _trace_sha256(trace),
        "offered_requests": len(trace),
        "offered_tokens": sum(item["service_tokens"] for item in trace),
        "action_counts": {
            action: action_counts[action]
            for action in ("admit", "defer", "shed")
        },
        "goodput_at_slo": {
            "successful_requests": goodput_count,
            "fixed_window_ticks": scenario.measurement_ticks,
            "requests_per_tick": goodput_count / scenario.measurement_ticks,
        },
        "deferred": {
            "completed": deferred_completed,
            "completed_in_measurement": sum(
                record["action"] == "defer"
                and record["completed_tick"] is not None
                and record["completed_tick"] <= scenario.measurement_ticks
                for record in records.values()
            ),
        },
        "measurement_end": measurement_state,
        "drain_tick": tick,
        "controller_in_flight_after_drain": (
            None if controller is None else controller.in_flight
        ),
        "scheduler": _scheduler_evidence(scheduler),
        "requests": list(records.values()),
        "ticks": tick_events,
        "schedule_events": schedule_events,
    }


def _forced_admit_ttft(
    target: dict,
    trace: tuple[dict, ...],
    treatment_records: dict[str, dict],
) -> int:
    """Replay frozen prior decisions and force the target into interactive.

    Later arrivals cannot advance ahead of this target: all public arrivals
    have the same interactive priority, while prior deferrals are lower-priority
    batch work.  Stopping arrivals after the target therefore gives its exact
    decision-time counterfactual without predictor feedback or future leakage.
    """

    clock = {"tick": 0.0}
    scheduler = _new_scheduler(clock)
    by_arrival: dict[int, list[dict]] = {}
    for item in trace:
        if item["arrival_tick"] > target["arrival_tick"]:
            break
        by_arrival.setdefault(item["arrival_tick"], []).append(item)

    tick = 0
    limit = target["arrival_tick"] + len(trace) + 10
    while tick < limit:
        clock["tick"] = float(tick)
        for item in by_arrival.get(tick, ()):
            is_target = item["request_id"] == target["request_id"]
            if tick == target["arrival_tick"] and (
                item["release_order"] > target["release_order"]
            ):
                break
            if is_target:
                request_class = "interactive"
            else:
                action = treatment_records[item["request_id"]]["action"]
                request_class = (
                    "interactive"
                    if action == "admit"
                    else "batch"
                    if action == "defer"
                    else None
                )
            if request_class is not None:
                scheduler.add_request(_engine_request(item, request_class))

        plan = scheduler.schedule()
        sampled = {}
        target_first_token = False
        for chunk in plan.scheduled:
            state = scheduler.states[chunk.request_id]
            if (not chunk.is_prefill) or state.prefill_done:
                sampled[chunk.request_id] = 1
                if chunk.request_id == target["request_id"]:
                    target_first_token = True
        finished = scheduler.update(sampled) if sampled else ()
        for request_id in finished:
            scheduler.forget(request_id)
        if target_first_token:
            return tick + 1 - target["arrival_tick"]
        tick += 1
    raise RuntimeError(
        f"forced-admit oracle did not produce a token for {target['request_id']}"
    )


def _confusion_matrix(
    scenario: Scenario,
    trace: tuple[dict, ...],
    treatment: dict,
) -> dict:
    records = {
        record["request_id"]: record
        for record in treatment["requests"]
    }
    ids = {"tp": [], "fp": [], "fn": [], "tn": []}
    for item in trace:
        record = records[item["request_id"]]
        oracle_ttft = _forced_admit_ttft(item, trace, records)
        record["forced_admit_oracle_ttft_ticks"] = oracle_ttft
        oracle_violation = oracle_ttft > scenario.ttft_slo_ticks
        record["forced_admit_oracle_violation"] = oracle_violation
        predicted_violation = record["action"] in {"defer", "shed"}
        label = (
            "tp"
            if predicted_violation and oracle_violation
            else "fp"
            if predicted_violation
            else "fn"
            if oracle_violation
            else "tn"
        )
        record["prediction_outcome"] = label
        ids[label].append(item["request_id"])

    counts = {label: len(request_ids) for label, request_ids in ids.items()}
    predicted_positive = counts["tp"] + counts["fp"]
    actual_positive = counts["tp"] + counts["fn"]
    actual_negative = counts["tn"] + counts["fp"]
    return {
        "counts": counts,
        "request_ids": ids,
        "precision": (
            None
            if predicted_positive == 0
            else counts["tp"] / predicted_positive
        ),
        "recall": (
            None
            if actual_positive == 0
            else counts["tp"] / actual_positive
        ),
        "false_positive_rate": (
            None
            if actual_negative == 0
            else counts["fp"] / actual_negative
        ),
        "false_negative_rate": (
            None
            if actual_positive == 0
            else counts["fn"] / actual_positive
        ),
    }


def _scenario_checks(
    scenario: Scenario,
    trace: tuple[dict, ...],
    baseline: dict,
    treatment: dict,
    confusion: dict,
) -> dict[str, bool]:
    treatment_records = {
        record["request_id"]: record
        for record in treatment["requests"]
    }
    action_counts = treatment["action_counts"]
    events = treatment["scheduler"]["events"]
    scheduled_ids = {
        event["request_id"]
        for event in treatment["schedule_events"]
    }
    expected_scheduled_ids = {
        request_id
        for request_id, record in treatment_records.items()
        if record["action"] != "shed"
    }
    oracle_admit_parity = all(
        record["action"] != "admit"
        or record["ttft_ticks"]
        == record["forced_admit_oracle_ttft_ticks"]
        for record in treatment_records.values()
    )
    common = {
        "matched_trace": (
            baseline["trace_sha256"]
            == treatment["trace_sha256"]
            == _trace_sha256(trace)
        ),
        "offered_accounting": (
            action_counts["admit"]
            + action_counts["defer"]
            + action_counts["shed"]
            == len(trace)
        ),
        "baseline_all_enqueued_and_completed": (
            baseline["action_counts"]["admit"] == len(trace)
            and baseline["scheduler"]["events"]["interactive:enqueue"]
            == len(trace)
            and baseline["scheduler"]["events"]["interactive:complete"]
            == len(trace)
        ),
        "treatment_enqueue_matches_decisions": (
            events["interactive:enqueue"] == action_counts["admit"]
            == events["interactive:admit"]
            and events["batch:enqueue"] == action_counts["defer"]
            == events["batch:admit"]
            and events["interactive:complete"] == action_counts["admit"]
            and events["batch:complete"] == action_counts["defer"]
        ),
        "only_non_shed_work_is_scheduled": (
            scheduled_ids == expected_scheduled_ids
        ),
        "shed_never_reaches_scheduler": all(
            record["action"] != "shed"
            or (
                record["enqueue_tick"] is None
                and record["scheduler_class"] is None
                and record["completed_tick"] is None
            )
            for record in treatment_records.values()
        ),
        "deferred_runs_as_batch_and_drains": (
            treatment["deferred"]["completed"] == action_counts["defer"]
            and all(
                record["action"] != "defer"
                or (
                    record["scheduler_class"] == "batch"
                    and record["completed_tick"] is not None
                )
                for record in treatment_records.values()
            )
        ),
        "controller_drains": treatment["controller_in_flight_after_drain"] == 0,
        "scheduler_queues_drain": (
            treatment["scheduler"]["queue_depth"]
            == {"interactive": 0, "batch": 0}
        ),
        "confusion_reconciles": (
            sum(confusion["counts"].values()) == len(trace)
            and {
                request_id
                for request_ids in confusion["request_ids"].values()
                for request_id in request_ids
            }
            == set(treatment_records)
        ),
        "forced_admit_oracle_matches_observed_admits": oracle_admit_parity,
        "fixed_window_goodput_noninferior": (
            treatment["goodput_at_slo"]["requests_per_tick"]
            >= baseline["goodput_at_slo"]["requests_per_tick"]
        ),
    }
    if scenario.saturated:
        common.update(
            {
                "exact_2x_saturation": scenario.offered_load_ratio == 2.0,
                "baseline_goodput_nonzero": (
                    baseline["goodput_at_slo"]["successful_requests"] > 0
                ),
                "treatment_goodput_nonzero": (
                    treatment["goodput_at_slo"]["successful_requests"] > 0
                ),
                "predicted_violations_nonzero": (
                    action_counts["defer"] + action_counts["shed"] > 0
                ),
                "defer_path_is_exercised": action_counts["defer"] > 0,
            }
        )
    else:
        common.update(
            {
                "underload_below_capacity": scenario.offered_load_ratio < 1.0,
                "underload_has_no_false_rejection": (
                    action_counts["defer"] == 0
                    and action_counts["shed"] == 0
                ),
                "underload_goodput_matches_baseline": (
                    treatment["goodput_at_slo"]["successful_requests"]
                    == baseline["goodput_at_slo"]["successful_requests"]
                    == len(trace)
                ),
            }
        )
    return common


def run_scenario(scenario: Scenario) -> dict:
    scenario.validate()
    trace = build_trace(scenario)
    baseline = run_trace("queue_and_hope", scenario, trace)
    treatment = run_trace("slo_admission", scenario, trace)
    confusion = _confusion_matrix(scenario, trace, treatment)
    checks = _scenario_checks(
        scenario,
        trace,
        baseline,
        treatment,
        confusion,
    )
    return {
        "scenario": asdict(scenario),
        "capacity_tokens_per_tick": _CAPACITY_TOKENS_PER_TICK,
        "offered_load_ratio": scenario.offered_load_ratio,
        "trace_sha256": _trace_sha256(trace),
        "trace": list(trace),
        "checks": checks,
        "passed": all(checks.values()),
        "confusion_matrix": confusion,
        "queue_and_hope": baseline,
        "slo_admission": treatment,
    }


def run_gate(
    scenarios: tuple[Scenario, ...] = FORMAL_SCENARIOS,
) -> dict:
    results = [run_scenario(scenario) for scenario in scenarios]
    checks = {
        f"{result['scenario']['name']}:{name}": passed
        for result in results
        for name, passed in result["checks"].items()
    }
    saturated = [
        result
        for result in results
        if result["offered_load_ratio"] > 1.0
    ]
    checks["cross_scenario:saturation_strict_gain_nonvacuous"] = bool(
        saturated
    ) and any(
        result["slo_admission"]["goodput_at_slo"]["successful_requests"]
        > result["queue_and_hope"]["goodput_at_slo"]["successful_requests"]
        for result in saturated
    )
    return {
        "schema_version": 1,
        "gate": "G5-F5c",
        "methodology": {
            "goodput": (
                "completed interactive admissions with actual TTFT <= fixed SLO "
                "divided by the fixed arrival window"
            ),
            "baseline": "queue-and-hope on the same arrivals and service",
            "defer": (
                "caller adapter enqueues the same request as lower-priority batch; "
                "AdmissionController itself only returns a decision"
            ),
            "oracle": (
                "post-decision production Scheduler replay with the target forced "
                "into interactive and no predictor feedback"
            ),
            "predictor_inputs": (
                "production AdmissionController interactive in-flight count and "
                "online TTFT EMA; total active work bounds the deferred backlog"
            ),
            "raw_queue_evidence": (
                "per-tick class counts plus request-id arrival, scheduling, "
                "first-token, and completion deltas; membership is reconstructible"
            ),
            "scope": (
                "homogeneous one-token service isolates admission behavior; "
                "results are not a variable-prompt cost-model claim"
            ),
        },
        "scenarios": results,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _source_environment() -> dict:
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
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args()
    result = run_gate()
    result["environment"] = _source_environment()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if args.assert_gate and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
