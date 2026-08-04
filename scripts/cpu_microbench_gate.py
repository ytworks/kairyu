#!/usr/bin/env python3
"""Run the variance-tolerant CPU microbenchmark regression gate.

The gate compares optimized and legacy implementations inside the same child
process wherever possible.  Ratios are intentionally loose: this is a smoke
alarm for large hot-path regressions on shared CI runners, not formal
performance evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TIMEOUT_SECONDS = 45
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


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    script: str
    args: tuple[str, ...]


BENCHMARKS = (
    BenchmarkSpec(
        name="scheduler_queue",
        script="bench/scheduler_queue_bench.py",
        args=(
            "--requests",
            "30000",
            "--removals",
            "3000",
            "--repeats",
            "5",
        ),
    ),
    BenchmarkSpec(
        name="radix_eviction",
        script="bench/radix_eviction_bench.py",
        args=(
            "--leaves",
            "10000",
            "--evictions",
            "25",
            "--repeats",
            "3",
        ),
    ),
    BenchmarkSpec(
        name="op_queue",
        script="bench/op_queue_bench.py",
        args=(
            "--operations",
            "50000",
            "--producers",
            "1",
            "--repeats",
            "5",
        ),
    ),
    BenchmarkSpec(
        name="sampler_penalty_state",
        script="bench/sampler_penalty_state_bench.py",
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
        ),
    ),
    BenchmarkSpec(
        name="proc_wire",
        script="bench/proc_wire_bench.py",
        args=(
            "--lengths",
            ",".join(str(length) for length in PROC_WIRE_LENGTHS),
            "--assert-gate",
        ),
    ),
    BenchmarkSpec(
        name="router_latency",
        script="bench/router_latency.py",
        args=("--routes", "10000", "--json", "--assert-gate"),
    ),
)


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
    timeout_seconds: int = BENCHMARK_TIMEOUT_SECONDS,
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
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BenchmarkFailure(
            f"{spec.name} did not complete: {type(error).__name__}"
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


def _minimum(name: str, actual: float, threshold: float) -> Check:
    return Check(name, actual, ">=", threshold, actual >= threshold)


def _less_than(name: str, actual: float, threshold: float) -> Check:
    return Check(name, actual, "<", threshold, actual < threshold)


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
    checks = [
        _equal("scheduler.requests", _number(scheduler, "requests"), 30_000.0),
        _equal("scheduler.removals", _number(scheduler, "removals"), 3_000.0),
        _minimum(
            "scheduler.fifo_speedup",
            _number(scheduler, "fifo_full_drain", "median_speedup"),
            2.0,
        ),
        _minimum(
            "scheduler.removal_speedup",
            _number(scheduler, "indexed_removal", "median_speedup"),
            10.0,
        ),
        _minimum(
            "scheduler.priority_speedup",
            _number(scheduler, "priority_full_drain", "median_speedup"),
            1.25,
        ),
        _equal("radix.leaves", _number(radix, "leaves"), 10_000.0),
        _equal("radix.evictions", _number(radix, "evictions"), 25.0),
        _minimum(
            "radix.eviction_speedup",
            _number(radix, "median_speedup"),
            100.0,
        ),
        _equal("op_queue.operations", operations, 50_000.0),
        _equal("op_queue.producers", _number(op_queue, "producers"), 1.0),
        _minimum(
            "op_queue.add_elapsed_speedup",
            _number(op_queue, "add_burst", "elapsed_speedup"),
            0.50,
        ),
        _minimum(
            "op_queue.abort_elapsed_speedup",
            _number(op_queue, "duplicate_abort_burst", "elapsed_speedup"),
            0.50,
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
        _minimum(
            "sampler.legacy_speedup",
            _number(sampler, "legacy_speedup"),
            5.0,
        ),
        _minimum(
            "sampler.append_legacy_speedup",
            _number(sampler, "append_legacy_speedup"),
            5.0,
        ),
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
        _equal("router.routes", _number(router, "routes"), 10_000.0),
        _equal(
            "router.p99_budget_seconds",
            _number(router, "p99_budget_seconds"),
            0.010,
        ),
        _less_than(
            "router.p99_seconds",
            _number(router, "p99_seconds"),
            0.010,
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
    for spec in BENCHMARKS:
        print(f"running {spec.name}...", file=sys.stderr, flush=True)
        try:
            results[spec.name] = run_benchmark(spec)
        except BenchmarkFailure as error:
            if error.result is not None:
                results[spec.name] = error.result
            errors[spec.name] = str(error)

    checks: tuple[Check, ...] = ()
    if set(results) == {spec.name for spec in BENCHMARKS}:
        try:
            checks = evaluate_results(results)
        except BenchmarkFailure as error:
            errors["evaluation"] = str(error)
    report = {
        "schema_version": 1,
        "cpu_only": True,
        "benchmarks": results,
        "checks": [asdict(check) for check in checks],
        "errors": errors,
        "all_passed": not errors and all(check.passed for check in checks),
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
            "errors": report["errors"],
            "all_passed": report["all_passed"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(rendered, end="")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
