#!/usr/bin/env python3
"""Run the deterministic CPU microbenchmark regression gate.

Each child benchmark reports wall-clock timings and deterministic work counts.
The verdict depends only on the code under test: it uses the work counts
(store primitives, items iterated, comparisons and hashes, nodes visited,
history reads, calls per route) and their growth between two sizes eight times
apart, plus exact workload and shape equality checks.  Timings, speed ratios,
and router latency percentiles are recorded under ``diagnostics`` and never
decide pass/fail.  The children run once, in order, under one generous hang
bound for the whole gate; the child that exceeds it is reported as a hang (and
any child not started as not run), never as slow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# One bound for all children together: about ten times their normal total
# (under a minute), and still inside the CI job's 10-minute limit so the report
# is written and uploaded.  Only a hang reaches it.
HANG_TIMEOUT_SECONDS = 420
# Work per operation may at most double between two sizes eight times apart:
# constant work stays at 1x, logarithmic work grows about 1.3x, and linear
# work grows 8x.
MAX_WORK_GROWTH = 2.0
SCHEDULER_WORK_SIZES = (2_000, 16_000)
RADIX_WORK_SIZES = (2_000, 16_000)
OP_QUEUE_WORK_SIZES = (2_000, 16_000)
SAMPLER_WORK_SIZES = (4_096, 32_768)
ROUTER_ROUTES = 10_000
ROUTER_MAX_CALLS_PER_ROUTE = 1_000
PROC_WIRE_LENGTHS = (128, 256, 512, 1024)
PROC_WIRE_EXPECTED_CHECKS = frozenset(
    {
        "clean_tracked_source",
        *(
            f"legacy_growth_{start}_{end}"
            for start, end in pairwise(PROC_WIRE_LENGTHS)
        ),
        *(
            f"v2_growth_{start}_{end}"
            for start, end in pairwise(PROC_WIRE_LENGTHS)
        ),
        *(f"v2_smaller_at_{length}" for length in PROC_WIRE_LENGTHS),
        *(f"v2_structure_at_{length}" for length in PROC_WIRE_LENGTHS),
    }
)


def _sizes_arg(sizes: tuple[int, ...]) -> str:
    return ",".join(str(size) for size in sizes)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    script: str
    args: tuple[str, ...]


BENCHMARKS = (
    BenchmarkSpec(
        name="scheduler_queue",
        script="verification/l1/performance/scheduler_queue_bench.py",
        args=(
            "--requests",
            "30000",
            "--removals",
            "3000",
            "--repeats",
            "5",
            "--work-sizes",
            _sizes_arg(SCHEDULER_WORK_SIZES),
        ),
    ),
    BenchmarkSpec(
        name="radix_eviction",
        script="verification/l1/performance/radix_eviction_bench.py",
        args=(
            "--leaves",
            "10000",
            "--evictions",
            "25",
            "--repeats",
            "3",
            "--work-sizes",
            _sizes_arg(RADIX_WORK_SIZES),
        ),
    ),
    BenchmarkSpec(
        name="op_queue",
        script="verification/l1/performance/op_queue_bench.py",
        args=(
            "--operations",
            "50000",
            "--producers",
            "1",
            "--repeats",
            "5",
            "--work-sizes",
            _sizes_arg(OP_QUEUE_WORK_SIZES),
        ),
    ),
    BenchmarkSpec(
        name="sampler_penalty_state",
        script="verification/l1/performance/sampler_penalty_state_bench.py",
        args=(
            "--device",
            "cpu",
            "--vocab-size",
            "151936",
            "--history-length",
            "32768",
            "--pending-depth",
            "2",
            "--penalties",
            "all",
            "--iterations",
            "20",
            "--repeats",
            "3",
            "--work-sizes",
            _sizes_arg(SAMPLER_WORK_SIZES),
        ),
    ),
    BenchmarkSpec(
        name="proc_wire",
        script="verification/l1/performance/proc_wire_bench.py",
        args=(
            "--lengths",
            ",".join(str(length) for length in PROC_WIRE_LENGTHS),
            "--assert-gate",
        ),
    ),
    BenchmarkSpec(
        name="router_latency",
        script="verification/orchestration/performance/router_latency.py",
        args=("--routes", str(ROUTER_ROUTES), "--json", "--assert-gate"),
    ),
)

# Report-only timing values: (diagnostic name, path inside the results).
TIMING_DIAGNOSTICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "scheduler.fifo_speedup",
        ("scheduler_queue", "fifo_full_drain", "median_speedup"),
    ),
    (
        "scheduler.removal_speedup",
        ("scheduler_queue", "indexed_removal", "median_speedup"),
    ),
    (
        "scheduler.priority_speedup",
        ("scheduler_queue", "priority_full_drain", "median_speedup"),
    ),
    ("radix.eviction_speedup", ("radix_eviction", "median_speedup")),
    (
        "op_queue.add_elapsed_speedup",
        ("op_queue", "add_burst", "elapsed_speedup"),
    ),
    (
        "op_queue.abort_elapsed_speedup",
        ("op_queue", "duplicate_abort_burst", "elapsed_speedup"),
    ),
    ("sampler.legacy_speedup", ("sampler_penalty_state", "legacy_speedup")),
    (
        "sampler.append_legacy_speedup",
        ("sampler_penalty_state", "append_legacy_speedup"),
    ),
    ("router.p50_seconds", ("router_latency", "p50_seconds")),
    ("router.p99_seconds", ("router_latency", "p99_seconds")),
)

# Checks of the protected behaviour, as opposed to fixed workload and shape.
_PROPERTY_CHECK_SUFFIXES = (".work_observed", ".work_growth")
_PROPERTY_CHECK_NAMES = frozenset(
    {
        "proc_wire.all_passed",
        "proc_wire.checks_complete",
        "router.calls_per_route",
        "router.passed",
    }
)


def is_property_check(name: str) -> bool:
    """Whether a check protects behaviour rather than the fixed workload."""
    return name in _PROPERTY_CHECK_NAMES or name.endswith(_PROPERTY_CHECK_SUFFIXES)


@dataclass(frozen=True)
class Check:
    name: str
    actual: bool | float | int | str | tuple[float, ...]
    relation: str
    threshold: bool | float | int | str | tuple[float, ...]
    passed: bool


class BenchmarkFailure(RuntimeError):
    """A benchmark could not produce a trustworthy result."""

    def __init__(self, message: str, *, result: dict[str, Any] | None = None):
        super().__init__(message)
        self.result = result


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def run_benchmark(
    spec: BenchmarkSpec,
    *,
    python: str = sys.executable,
    root: Path = ROOT,
    timeout_seconds: int = HANG_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    command = (python, str(root / spec.script), *spec.args)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=_child_environment(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise BenchmarkFailure(
            f"{spec.name} hang: no result within the remaining "
            f"{timeout_seconds} s of the {HANG_TIMEOUT_SECONDS} s hang bound"
        ) from error
    except OSError as error:
        raise BenchmarkFailure(
            f"{spec.name} did not start: {type(error).__name__}"
        ) from error
    parsed: object | None = None
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pass
    if completed.returncode != 0:
        if isinstance(parsed, dict):
            failed_checks = [
                check
                for check in parsed.get("checks", [])
                if isinstance(check, dict) and check.get("passed") is False
            ]
            detail = json.dumps(failed_checks or parsed, sort_keys=True)
        else:
            detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkFailure(
            f"{spec.name} exited {completed.returncode}: {detail[-4000:]}",
            result=parsed if isinstance(parsed, dict) else None,
        )
    if parsed is None:
        raise BenchmarkFailure(
            f"{spec.name} did not emit one JSON object"
        )
    if not isinstance(parsed, dict):
        raise BenchmarkFailure(f"{spec.name} result must be a JSON object")
    return parsed


def _at(result: Mapping[str, Any], *path: str) -> Any:
    value: Any = result
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise BenchmarkFailure(f"missing result field: {'.'.join(path)}")
        value = value[component]
    return value


def _number(result: Mapping[str, Any], *path: str) -> float:
    value = _at(result, *path)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise BenchmarkFailure(f"result field is not finite: {'.'.join(path)}")
    return float(value)


def _counts(
    case: Mapping[str, Any], field: str, label: str, *, minimum: int
) -> tuple[int, int]:
    values = _at(case, field)
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(type(value) is not int or value < minimum for value in values)
    ):
        raise BenchmarkFailure(
            f"{label}.{field} must be two integers of at least {minimum}"
        )
    return values[0], values[1]


def _equal(
    name: str,
    actual: bool | float | int | str | tuple[float, ...],
    threshold: bool | float | int | str | tuple[float, ...],
) -> Check:
    return Check(
        name,
        actual,
        "==",
        threshold,
        type(actual) is type(threshold) and actual == threshold,
    )


def _work_checks(
    name: str,
    result: Mapping[str, Any],
    case_name: str,
    expected_sizes: tuple[int, int],
) -> list[Check]:
    """Work must be observed and per-operation work must stay sublinear."""
    case = _at(result, "work", case_name)
    if not isinstance(case, Mapping):
        raise BenchmarkFailure(f"{name} work must be an object")
    sizes = _counts(case, "sizes", name, minimum=0)
    operations = _counts(case, "operations", name, minimum=1)
    work = _counts(case, "work", name, minimum=0)
    small, large = (
        units / count for units, count in zip(work, operations, strict=True)
    )
    growth = large / small if small > 0 else 0.0
    return [
        _equal(
            f"{name}.work_sizes",
            tuple(float(size) for size in sizes),
            tuple(float(size) for size in expected_sizes),
        ),
        Check(f"{name}.work_observed", small, ">", 0.0, small > 0),
        Check(
            f"{name}.work_growth",
            growth,
            "<=",
            MAX_WORK_GROWTH,
            small > 0 and growth <= MAX_WORK_GROWTH,
        ),
    ]


def evaluate_results(results: Mapping[str, Mapping[str, Any]]) -> tuple[Check, ...]:
    expected = {spec.name for spec in BENCHMARKS}
    if set(results) != expected:
        raise BenchmarkFailure(
            "benchmark result set mismatch: "
            f"expected={sorted(expected)!r}, actual={sorted(results)!r}"
        )

    scheduler = results["scheduler_queue"]
    radix = results["radix_eviction"]
    op_queue = results["op_queue"]
    sampler = results["sampler_penalty_state"]
    proc_wire = results["proc_wire"]
    router = results["router_latency"]

    measurements = _at(proc_wire, "measurements")
    if not isinstance(measurements, list):
        raise BenchmarkFailure("proc_wire.measurements must be a list")
    proc_lengths: list[float] = []
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, Mapping):
            raise BenchmarkFailure(
                f"proc_wire.measurements.{index} must be an object"
            )
        proc_lengths.append(_number(measurement, "output_tokens"))
    proc_checks = _at(proc_wire, "checks")
    if not isinstance(proc_checks, list):
        raise BenchmarkFailure("proc_wire.checks must be a list")
    proc_check_names: list[str] = []
    proc_checks_passed = True
    for index, check in enumerate(proc_checks):
        if not isinstance(check, Mapping) or type(check.get("name")) is not str:
            raise BenchmarkFailure(f"proc_wire.checks.{index} is malformed")
        proc_check_names.append(check["name"])
        proc_checks_passed = (
            proc_checks_passed
            and type(check.get("passed")) is bool
            and check["passed"] is True
        )
    proc_checks_complete = (
        len(proc_check_names) == len(PROC_WIRE_EXPECTED_CHECKS)
        and set(proc_check_names) == PROC_WIRE_EXPECTED_CHECKS
        and proc_checks_passed
    )

    operations = _number(op_queue, "operations")
    add_containers = _number(
        op_queue,
        "add_burst",
        "coalesced",
        "queued_operation_containers",
    )
    abort_ids = _number(
        op_queue,
        "duplicate_abort_burst",
        "coalesced",
        "queued_abort_ids",
    )
    add_ids = _number(
        op_queue,
        "add_burst",
        "coalesced",
        "queued_ids",
    )
    abort_containers = _number(
        op_queue,
        "duplicate_abort_burst",
        "coalesced",
        "queued_operation_containers_including_add",
    )
    router_first, router_last = _counts(
        _at(router, "work", "routes_served"),
        "work",
        "router.routes_served",
        minimum=0,
    )
    router_routes = _counts(
        _at(router, "work", "routes_served"),
        "operations",
        "router.routes_served",
        minimum=1,
    )
    router_calls_per_route = max(
        router_first / router_routes[0],
        router_last / router_routes[1],
    )
    checks = [
        _equal("scheduler.requests", _number(scheduler, "requests"), 30_000.0),
        _equal("scheduler.removals", _number(scheduler, "removals"), 3_000.0),
        *_work_checks(
            "scheduler.fifo_full_drain",
            scheduler,
            "fifo_full_drain",
            SCHEDULER_WORK_SIZES,
        ),
        *_work_checks(
            "scheduler.indexed_removal",
            scheduler,
            "indexed_removal",
            SCHEDULER_WORK_SIZES,
        ),
        *_work_checks(
            "scheduler.priority_full_drain",
            scheduler,
            "priority_full_drain",
            SCHEDULER_WORK_SIZES,
        ),
        _equal("radix.leaves", _number(radix, "leaves"), 10_000.0),
        _equal("radix.evictions", _number(radix, "evictions"), 25.0),
        *_work_checks("radix.eviction", radix, "eviction", RADIX_WORK_SIZES),
        _equal("op_queue.operations", operations, 50_000.0),
        _equal("op_queue.producers", _number(op_queue, "producers"), 1.0),
        *_work_checks(
            "op_queue.add_burst",
            op_queue,
            "add_burst",
            OP_QUEUE_WORK_SIZES,
        ),
        *_work_checks(
            "op_queue.duplicate_abort_burst",
            op_queue,
            "duplicate_abort_burst",
            OP_QUEUE_WORK_SIZES,
        ),
        _equal(
            "op_queue.add_containers",
            add_containers,
            1.0,
        ),
        _equal("op_queue.add_ids", add_ids, operations),
        _equal(
            "op_queue.abort_containers_including_add",
            abort_containers,
            2.0,
        ),
        _equal("op_queue.abort_ids", abort_ids, 1.0),
        _equal("sampler.device", _at(sampler, "device"), "cpu"),
        _equal("sampler.vocab_size", _number(sampler, "vocab_size"), 151_936.0),
        _equal(
            "sampler.history_length",
            _number(sampler, "history_length"),
            32_768.0,
        ),
        _equal("sampler.pending_depth", _number(sampler, "pending_depth"), 2.0),
        _equal("sampler.penalties", _at(sampler, "penalties"), "all"),
        _equal("sampler.iterations", _number(sampler, "iterations"), 20.0),
        _equal("sampler.repeats", _number(sampler, "repeats"), 3.0),
        _equal(
            "sampler.validated_bitwise",
            _at(sampler, "validated_bitwise"),
            True,
        ),
        *_work_checks(
            "sampler.steady_state",
            sampler,
            "steady_state",
            SAMPLER_WORK_SIZES,
        ),
        *_work_checks("sampler.append", sampler, "append", SAMPLER_WORK_SIZES),
        _equal("proc_wire.schema_version", _number(proc_wire, "schema_version"), 1.0),
        _equal("proc_wire.metric", _at(proc_wire, "metric"), "msgpack_frame_bytes"),
        _equal("proc_wire.protocol_legacy", _number(proc_wire, "protocols", "legacy"), 1.0),
        _equal("proc_wire.protocol_delta", _number(proc_wire, "protocols", "delta"), 2.0),
        _equal(
            "proc_wire.max_v2_growth",
            _number(proc_wire, "thresholds", "max_v2_growth_per_doubling"),
            2.15,
        ),
        _equal(
            "proc_wire.min_legacy_growth",
            _number(proc_wire, "thresholds", "min_legacy_growth_per_doubling"),
            3.5,
        ),
        _equal(
            "proc_wire.lengths",
            tuple(proc_lengths),
            tuple(float(length) for length in PROC_WIRE_LENGTHS),
        ),
        _equal(
            "proc_wire.check_names",
            ",".join(sorted(str(name) for name in proc_check_names)),
            ",".join(sorted(PROC_WIRE_EXPECTED_CHECKS)),
        ),
        _equal("proc_wire.checks_complete", proc_checks_complete, True),
        _equal("proc_wire.tracked_dirty", _at(proc_wire, "source", "tracked_dirty"), False),
        _equal("proc_wire.all_passed", _at(proc_wire, "all_passed"), True),
        _equal("router.routes", _number(router, "routes"), float(ROUTER_ROUTES)),
        _equal(
            "router.max_calls_per_route",
            _number(router, "max_calls_per_route"),
            float(ROUTER_MAX_CALLS_PER_ROUTE),
        ),
        _equal(
            "router.max_work_growth",
            _number(router, "max_work_growth"),
            MAX_WORK_GROWTH,
        ),
        *_work_checks(
            "router.routes_served",
            router,
            "routes_served",
            (0, ROUTER_ROUTES),
        ),
        Check(
            "router.calls_per_route",
            router_calls_per_route,
            "<=",
            float(ROUTER_MAX_CALLS_PER_ROUTE),
            router_calls_per_route <= ROUTER_MAX_CALLS_PER_ROUTE,
        ),
        _equal("router.passed", _at(router, "passed"), True),
    ]
    for case in ("fifo_full_drain", "indexed_removal", "priority_full_drain"):
        for implementation in ("legacy", "indexed"):
            checks.append(
                _equal(
                    f"scheduler.{case}.{implementation}.repeats",
                    _number(scheduler, case, implementation, "repeats"),
                    5.0,
                )
            )
    for implementation in ("scanning", "indexed"):
        checks.append(
            _equal(
                f"radix.{implementation}.repeats",
                _number(radix, implementation, "repeats"),
                3.0,
            )
        )
    for case in ("add_burst", "duplicate_abort_burst"):
        for implementation in ("legacy", "coalesced"):
            checks.append(
                _equal(
                    f"op_queue.{case}.{implementation}.repeats",
                    _number(op_queue, case, implementation, "repeats"),
                    5.0,
                )
            )
    return tuple(checks)


def timing_diagnostics(results: Mapping[str, Any]) -> dict[str, float]:
    """Report-only speed ratios and latencies that are present and finite."""
    diagnostics: dict[str, float] = {}
    for name, path in TIMING_DIAGNOSTICS:
        try:
            diagnostics[name] = _number(results, *path)
        except BenchmarkFailure:
            continue
    return diagnostics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the complete JSON diagnostic report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    elapsed: dict[str, float] = {}
    deadline = time.monotonic() + HANG_TIMEOUT_SECONDS
    for spec in BENCHMARKS:
        started = time.monotonic()
        remaining = math.floor(deadline - started)
        if remaining <= 0:
            errors[spec.name] = (
                f"{spec.name} not run: the {HANG_TIMEOUT_SECONDS} s hang bound "
                "was reached by an earlier benchmark"
            )
            continue
        print(f"running {spec.name}...", file=sys.stderr, flush=True)
        try:
            results[spec.name] = run_benchmark(spec, timeout_seconds=remaining)
        except BenchmarkFailure as error:
            if error.result is not None:
                results[spec.name] = error.result
            errors[spec.name] = str(error)
        elapsed[spec.name] = round(time.monotonic() - started, 3)

    checks: tuple[Check, ...] = ()
    if set(results) == {spec.name for spec in BENCHMARKS}:
        try:
            checks = evaluate_results(results)
        except BenchmarkFailure as error:
            errors["evaluation"] = str(error)
    report = {
        "schema_version": 2,
        "cpu_only": True,
        "benchmarks": results,
        "checks": [asdict(check) for check in checks],
        "diagnostics": {
            "timing": timing_diagnostics(results),
            "elapsed_seconds": elapsed,
            "hang_timeout_seconds": HANG_TIMEOUT_SECONDS,
        },
        "errors": errors,
        "all_passed": not errors
        and bool(checks)
        and all(check.passed for check in checks),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        except OSError as error:
            print(rendered, end="")
            print(
                "could not write CPU microbenchmark report: "
                f"{type(error).__name__}",
                file=sys.stderr,
            )
            return 2
        summary = {
            "schema_version": report["schema_version"],
            "report_path": str(args.output),
            "checks": report["checks"],
            "diagnostics": report["diagnostics"],
            "errors": report["errors"],
            "all_passed": report["all_passed"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(rendered, end="")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
