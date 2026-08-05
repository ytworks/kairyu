#!/usr/bin/env python3
"""Maintain the fail-closed nightly CPU microbenchmark performance series."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

gate = importlib.import_module("scripts.cpu_microbench_gate")


SERIES_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
METHOD = "issue-377-nightly-cpu-perf-v1"
MIN_HISTORY = 5
WINDOW_SIZE = 7
DECLINE_FRACTION = Decimal("0.15")
MAX_SERIES_BYTES = 64 * 1024 * 1024
MAX_REPORT_BYTES = 64 * 1024 * 1024

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_POSITIVE_DECIMAL_RE = re.compile(r"^[1-9][0-9]*$")
_REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_.-])?/[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_.-])?$"
)
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")

_CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

HARD_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "scheduler.fifo_speedup",
        ("benchmarks", "scheduler_queue", "fifo_full_drain", "median_speedup"),
    ),
    (
        "scheduler.removal_speedup",
        ("benchmarks", "scheduler_queue", "indexed_removal", "median_speedup"),
    ),
    (
        "scheduler.priority_speedup",
        ("benchmarks", "scheduler_queue", "priority_full_drain", "median_speedup"),
    ),
    (
        "radix.eviction_speedup",
        ("benchmarks", "radix_eviction", "median_speedup"),
    ),
    (
        "op_queue.add_elapsed_speedup",
        ("benchmarks", "op_queue", "add_burst", "elapsed_speedup"),
    ),
    (
        "op_queue.abort_elapsed_speedup",
        (
            "benchmarks",
            "op_queue",
            "duplicate_abort_burst",
            "elapsed_speedup",
        ),
    ),
    (
        "sampler.legacy_speedup",
        ("benchmarks", "sampler_penalty_state", "legacy_speedup"),
    ),
    (
        "sampler.append_legacy_speedup",
        ("benchmarks", "sampler_penalty_state", "append_legacy_speedup"),
    ),
)
_HARD_NAMES = tuple(name for name, _path in HARD_METRICS)
_RETAINABLE_SOURCE_FAILURES = frozenset(
    {
        *_HARD_NAMES,
        "proc_wire.all_passed",
        "proc_wire.checks_complete",
        "router.p99_seconds",
        "router.passed",
    }
)


class SeriesError(ValueError):
    """A report or series cannot be trusted."""


def _fail(message: str) -> SeriesError:
    return SeriesError(message)


def _reject_constant(value: str) -> None:
    raise _fail(f"non-finite JSON constant {value!r}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_json(value: object, *, path: str = "$") -> None:
    if value is None or isinstance(value, bool | str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(f"{path} contains a non-finite number")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _fail(f"{path} contains a non-string object key")
            _validate_json(item, path=f"{path}.{key}")
        return
    raise _fail(f"{path} contains non-JSON value {type(value).__name__}")


def _strict_json(data: bytes, *, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _fail(f"{label} is not UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except SeriesError:
        raise
    except json.JSONDecodeError as error:
        raise _fail(f"{label} is not valid JSON: {error.msg}") from error
    except ValueError as error:
        raise _fail(f"{label} contains an invalid numeric value") from error
    except RecursionError as error:
        raise _fail(f"{label} exceeds the maximum JSON nesting depth") from error
    try:
        _validate_json(value)
    except RecursionError as error:
        raise _fail(f"{label} exceeds the maximum JSON nesting depth") from error
    return value


def _canonical_bytes(value: object) -> bytes:
    _validate_json(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as error:
        raise _fail("value cannot be serialized as canonical JSON") from error


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _read_limited(path: Path, *, limit: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise _fail(f"cannot stat {label}: {type(error).__name__}") from error
    if not path.is_file():
        raise _fail(f"{label} is not a regular file")
    if size > limit:
        raise _fail(f"{label} exceeds {limit} bytes")
    try:
        return path.read_bytes()
    except OSError as error:
        raise _fail(f"cannot read {label}: {type(error).__name__}") from error


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor = -1
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise _fail(f"cannot publish {path.name}: {type(error).__name__}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _header() -> dict[str, Any]:
    return {
        "format": "kairyu-cpu-perf-series",
        "kind": "header",
        "schema_version": SERIES_SCHEMA_VERSION,
    }


def _genesis_sha256() -> str:
    return _sha256_value(_header())


def _trailer(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "trailer",
        "record_count": len(records),
        "schema_version": SERIES_SCHEMA_VERSION,
        "tip_sha256": (
            records[-1]["record_sha256"] if records else _genesis_sha256()
        ),
    }


def _render_series(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [_canonical_bytes(_header())]
    lines.extend(_canonical_bytes(record) for record in records)
    lines.append(_canonical_bytes(_trailer(records)))
    return b"\n".join(lines) + b"\n"


def _validate_repository(value: object) -> str:
    if not isinstance(value, str) or _REPOSITORY_RE.fullmatch(value) is None:
        raise _fail("repository must be an owner/name pair")
    return value


def _validate_positive_decimal(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _POSITIVE_DECIMAL_RE.fullmatch(value) is None:
        raise _fail(f"{field} must be a canonical positive decimal string")
    return value


def _validate_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise _fail("sha must be a full lowercase Git object ID")
    return value


def _validate_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise _fail(f"{field} must be a lowercase SHA-256")
    return value


def _runtime_identity() -> dict[str, str]:
    values = {
        "hosted_image_class": os.environ.get("ImageOS", "local"),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "system": platform.system(),
    }
    if any(not value for value in values.values()):
        raise _fail("runtime identity contains an empty component")
    return values


def _workload_identity() -> dict[str, Any]:
    return {
        "benchmarks": [
            {
                "args": list(spec.args),
                "name": spec.name,
                "script": spec.script,
            }
            for spec in gate.BENCHMARKS
        ],
        "cpu_environment": dict(_CPU_ENVIRONMENT),
    }


def _policy_identity() -> dict[str, Any]:
    return {
        "decline_boundary_inclusive": True,
        "decline_fraction": str(DECLINE_FRACTION),
        "hard_metrics": [
            {
                "direction": "higher",
                "name": name,
                "report_path": ".".join(path),
            }
            for name, path in HARD_METRICS
        ],
        "minimum_history": MIN_HISTORY,
        "window_size": WINDOW_SIZE,
    }


def _segment(*, repository: str) -> dict[str, Any]:
    return {
        "compatibility": {
            "method": METHOD,
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "repository": repository,
            "series_schema_version": SERIES_SCHEMA_VERSION,
        },
        "policy": _policy_identity(),
        "runtime": _runtime_identity(),
        "workload": _workload_identity(),
    }


def _validate_segment(value: object, *, repository: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "compatibility",
        "policy",
        "runtime",
        "workload",
    }:
        raise _fail("record segment has an unexpected schema")
    compatibility = value["compatibility"]
    if not isinstance(compatibility, dict) or set(compatibility) != {
        "method",
        "report_schema_version",
        "repository",
        "series_schema_version",
    }:
        raise _fail("record compatibility identity is invalid")
    if (
        not isinstance(compatibility["method"], str)
        or not compatibility["method"]
        or type(compatibility["report_schema_version"]) is not int
        or compatibility["report_schema_version"] <= 0
        or compatibility["repository"] != repository
        or type(compatibility["series_schema_version"]) is not int
        or compatibility["series_schema_version"] != SERIES_SCHEMA_VERSION
    ):
        raise _fail("record compatibility identity is invalid")

    policy = value["policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "decline_boundary_inclusive",
        "decline_fraction",
        "hard_metrics",
        "minimum_history",
        "window_size",
    }:
        raise _fail("record comparator policy is invalid")
    if type(policy["decline_boundary_inclusive"]) is not bool:
        raise _fail("record comparator boundary policy is invalid")
    if not isinstance(policy["decline_fraction"], str):
        raise _fail("record comparator decline fraction is invalid")
    try:
        decline = Decimal(policy["decline_fraction"])
        decline_is_valid = (
            decline.is_finite() and Decimal(0) < decline < Decimal(1)
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        raise _fail("record comparator decline fraction is invalid") from error
    if not decline_is_valid:
        raise _fail("record comparator decline fraction is invalid")
    if (
        type(policy["minimum_history"]) is not int
        or policy["minimum_history"] <= 0
        or type(policy["window_size"]) is not int
        or policy["window_size"] < policy["minimum_history"]
    ):
        raise _fail("record comparator window policy is invalid")
    hard_metrics = policy["hard_metrics"]
    if not isinstance(hard_metrics, list) or len(hard_metrics) != len(_HARD_NAMES):
        raise _fail("record comparator hard metric policy is invalid")
    seen_metrics: set[str] = set()
    for metric in hard_metrics:
        if not isinstance(metric, dict) or set(metric) != {
            "direction",
            "name",
            "report_path",
        }:
            raise _fail("record comparator hard metric policy is invalid")
        if (
            metric["direction"] != "higher"
            or metric["name"] not in _HARD_NAMES
            or not isinstance(metric["report_path"], str)
            or not metric["report_path"]
            or metric["name"] in seen_metrics
        ):
            raise _fail("record comparator hard metric policy is invalid")
        seen_metrics.add(metric["name"])
    if seen_metrics != set(_HARD_NAMES):
        raise _fail("record comparator hard metric policy is invalid")

    workload = value["workload"]
    if not isinstance(workload, dict) or set(workload) != {
        "benchmarks",
        "cpu_environment",
    }:
        raise _fail("record workload identity is invalid")
    benchmarks = workload["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks:
        raise _fail("record workload benchmark inventory is invalid")
    seen_benchmarks: set[str] = set()
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict) or set(benchmark) != {
            "args",
            "name",
            "script",
        }:
            raise _fail("record workload benchmark inventory is invalid")
        if (
            not isinstance(benchmark["name"], str)
            or not benchmark["name"]
            or benchmark["name"] in seen_benchmarks
            or not isinstance(benchmark["script"], str)
            or not benchmark["script"]
            or not isinstance(benchmark["args"], list)
            or any(not isinstance(argument, str) for argument in benchmark["args"])
        ):
            raise _fail("record workload benchmark inventory is invalid")
        seen_benchmarks.add(benchmark["name"])
    environment = workload["cpu_environment"]
    if (
        not isinstance(environment, dict)
        or not environment
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in environment.items()
        )
    ):
        raise _fail("record workload CPU environment is invalid")
    runtime = value["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "hosted_image_class",
        "machine",
        "python_implementation",
        "python_major_minor",
        "system",
    }:
        raise _fail("record runtime identity is invalid")
    if any(not isinstance(item, str) or not item for item in runtime.values()):
        raise _fail("record runtime identity contains an empty component")
    return value


def _at(value: object, *path: str) -> Any:
    current = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise _fail(f"missing report field: {'.'.join(path)}")
        current = current[component]
    return current


def _finite_number(value: object, *path: str, positive: bool = False) -> float:
    result = _at(value, *path)
    if isinstance(result, bool) or not isinstance(result, int | float):
        qualifier = "positive finite" if positive else "finite"
        raise _fail(f"report field {'.'.join(path)} must be {qualifier}")
    try:
        numeric = float(result)
    except (OverflowError, ValueError) as error:
        raise _fail(f"report field {'.'.join(path)} must be finite") from error
    if not math.isfinite(numeric) or (positive and numeric <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise _fail(f"report field {'.'.join(path)} must be {qualifier}")
    return numeric


def _hard_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: _finite_number(report, *path, positive=True)
        for name, path in HARD_METRICS
    }


def _diagnostics(report: Mapping[str, Any]) -> dict[str, Any]:
    scheduler: dict[str, Any] = {}
    for public_name, report_name in (
        ("fifo", "fifo_full_drain"),
        ("removal", "indexed_removal"),
        ("priority", "priority_full_drain"),
    ):
        scheduler[public_name] = {
            implementation: _finite_number(
                report,
                "benchmarks",
                "scheduler_queue",
                report_name,
                implementation,
                "median_seconds",
                positive=True,
            )
            for implementation in ("legacy", "indexed")
        }

    radix = {
        implementation: _finite_number(
            report,
            "benchmarks",
            "radix_eviction",
            implementation,
            "median_seconds",
            positive=True,
        )
        for implementation in ("scanning", "indexed")
    }
    op_queue: dict[str, Any] = {}
    for public_name, report_name in (
        ("add", "add_burst"),
        ("abort", "duplicate_abort_burst"),
    ):
        op_queue[public_name] = {
            implementation: _finite_number(
                report,
                "benchmarks",
                "op_queue",
                report_name,
                implementation,
                "elapsed_seconds",
                positive=True,
            )
            for implementation in ("legacy", "coalesced")
        }

    sampler = {
        "append_modes_median_us": {
            name: _finite_number(
                report,
                "benchmarks",
                "sampler_penalty_state",
                "append_modes_median_us",
                name,
                positive=True,
            )
            for name in (
                "committed_plus_overlay",
                "incremental_effective_counts",
                "legacy_full_history",
            )
        },
        "append_overlay_speedup": _finite_number(
            report,
            "benchmarks",
            "sampler_penalty_state",
            "append_overlay_speedup",
            positive=True,
        ),
        "modes_median_us": {
            name: _finite_number(
                report,
                "benchmarks",
                "sampler_penalty_state",
                "modes",
                name,
                "median_us",
                positive=True,
            )
            for name in (
                "committed_plus_overlay",
                "incremental_effective_counts",
                "legacy_full_history",
            )
        },
        "overlay_speedup": _finite_number(
            report,
            "benchmarks",
            "sampler_penalty_state",
            "overlay_speedup",
            positive=True,
        ),
    }

    measurements = _at(report, "benchmarks", "proc_wire", "measurements")
    if not isinstance(measurements, list) or len(measurements) != len(
        gate.PROC_WIRE_LENGTHS
    ):
        raise _fail("proc_wire measurements must match the fixed inventory")
    proc_measurements: list[dict[str, float]] = []
    for index, expected_tokens in enumerate(gate.PROC_WIRE_LENGTHS):
        measurement = measurements[index]
        if not isinstance(measurement, Mapping):
            raise _fail(f"proc_wire measurement {index} must be an object")
        output_tokens = _finite_number(
            measurement, "output_tokens", positive=True
        )
        if output_tokens != float(expected_tokens):
            raise _fail("proc_wire measurements do not match the fixed lengths")
        proc_measurements.append(
            {
                "legacy_total_bytes": _finite_number(
                    measurement, "legacy_total_bytes", positive=True
                ),
                "output_tokens": output_tokens,
                "v2_total_bytes": _finite_number(
                    measurement, "v2_total_bytes", positive=True
                ),
            }
        )

    growth = _at(report, "benchmarks", "proc_wire", "growth")
    expected_growth = tuple(
        zip(gate.PROC_WIRE_LENGTHS, gate.PROC_WIRE_LENGTHS[1:], strict=False)
    )
    if not isinstance(growth, list) or len(growth) != len(expected_growth):
        raise _fail("proc_wire growth must match the fixed inventory")
    proc_growth: list[dict[str, float]] = []
    for index, (item, expected_pair) in enumerate(zip(growth, expected_growth, strict=True)):
        if not isinstance(item, Mapping):
            raise _fail(f"proc_wire growth {index} must be an object")
        rendered_growth = {
            field: _finite_number(item, field, positive=True)
            for field in (
                "from_tokens",
                "legacy_empirical_exponent",
                "legacy_ratio",
                "to_tokens",
                "v2_empirical_exponent",
                "v2_ratio",
            )
        }
        if (
            rendered_growth["from_tokens"],
            rendered_growth["to_tokens"],
        ) != tuple(float(value) for value in expected_pair):
            raise _fail("proc_wire growth does not match the fixed lengths")
        proc_growth.append(rendered_growth)

    router = {
        percentile: _finite_number(
            report,
            "benchmarks",
            "router_latency",
            f"{percentile}_seconds",
            positive=True,
        )
        for percentile in ("p50", "p99")
    }
    return {
        "op_queue_elapsed_seconds": op_queue,
        "proc_wire": {"growth": proc_growth, "measurements": proc_measurements},
        "radix_median_seconds": radix,
        "router_seconds": router,
        "sampler": sampler,
        "scheduler_median_seconds": scheduler,
    }


def _validated_report(
    path: Path,
    *,
    expected_sha: str,
) -> tuple[dict[str, Any], bytes, bool]:
    data = _read_limited(path, limit=MAX_REPORT_BYTES, label="CPU report")
    parsed = _strict_json(data, label="CPU report")
    if not isinstance(parsed, dict) or set(parsed) != {
        "all_passed",
        "benchmarks",
        "checks",
        "cpu_only",
        "errors",
        "schema_version",
    }:
        raise _fail("CPU report has an unexpected top-level schema")
    if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
        raise _fail("CPU report schema_version must be integer 1")
    if parsed["cpu_only"] is not True:
        raise _fail("CPU report must attest cpu_only=true")
    if type(parsed["all_passed"]) is not bool:
        raise _fail("CPU report all_passed must be boolean")
    if not isinstance(parsed["errors"], dict):
        raise _fail("CPU report errors must be an object")
    if parsed["errors"]:
        raise _fail("CPU report contains benchmark or evaluation errors")
    benchmarks = parsed["benchmarks"]
    if not isinstance(benchmarks, dict):
        raise _fail("CPU report benchmarks must be an object")
    try:
        recomputed = gate.evaluate_results(benchmarks)
    except (ArithmeticError, gate.BenchmarkFailure) as error:
        raise _fail(f"CPU report cannot be independently evaluated: {error}") from error
    expected_checks = json.loads(
        _canonical_bytes([asdict(check) for check in recomputed]).decode("utf-8")
    )
    if _canonical_bytes(parsed["checks"]) != _canonical_bytes(expected_checks):
        raise _fail("CPU report checks do not match independent evaluation")
    failed_checks = {check.name for check in recomputed if not check.passed}
    invalid_contract_checks = failed_checks - _RETAINABLE_SOURCE_FAILURES
    if invalid_contract_checks:
        raise _fail(
            "CPU report violates the fixed workload/shape contract: "
            + ", ".join(sorted(invalid_contract_checks))
        )

    proc_wire = _at(parsed, "benchmarks", "proc_wire")
    proc_checks = _at(proc_wire, "checks")
    if not isinstance(proc_checks, list) or len(proc_checks) != len(
        gate.PROC_WIRE_EXPECTED_CHECKS
    ):
        raise _fail("proc_wire checks do not match the fixed inventory")
    proc_outcomes: dict[str, bool] = {}
    for check in proc_checks:
        if (
            not isinstance(check, Mapping)
            or not isinstance(check.get("name"), str)
            or type(check.get("passed")) is not bool
            or check["name"] in proc_outcomes
        ):
            raise _fail("proc_wire checks are malformed or duplicated")
        proc_outcomes[check["name"]] = check["passed"]
    if set(proc_outcomes) != set(gate.PROC_WIRE_EXPECTED_CHECKS):
        raise _fail("proc_wire checks do not match the fixed inventory")
    structural_proc_checks = {
        "clean_tracked_source",
        *(name for name in proc_outcomes if name.startswith("v2_structure_at_")),
    }
    if any(not proc_outcomes[name] for name in structural_proc_checks):
        raise _fail("proc_wire structural/source checks must pass")
    proc_all_passed = _at(proc_wire, "all_passed")
    if type(proc_all_passed) is not bool or proc_all_passed is not all(
        proc_outcomes.values()
    ):
        raise _fail("proc_wire all_passed disagrees with its checks")

    router = _at(parsed, "benchmarks", "router_latency")
    router_p99 = _finite_number(router, "p99_seconds", positive=True)
    router_budget = _finite_number(router, "p99_budget_seconds", positive=True)
    router_passed = _at(router, "passed")
    if type(router_passed) is not bool or router_passed is not (
        router_p99 < router_budget
    ):
        raise _fail("router passed disagrees with its p99 budget")

    source_gate_passed = not failed_checks
    if parsed["all_passed"] is not source_gate_passed:
        raise _fail("CPU report all_passed disagrees with independent evaluation")

    source = _at(parsed, "benchmarks", "proc_wire", "source")
    if not isinstance(source, Mapping):
        raise _fail("proc_wire source must be an object")
    if source.get("commit") != expected_sha:
        raise _fail("CPU report source commit does not match --sha")
    if source.get("tracked_dirty") is not False:
        raise _fail("CPU report source must be tracked-clean")

    child_python = _at(parsed, "benchmarks", "proc_wire", "environment", "python")
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if not isinstance(child_python, str) or not (
        child_python == current_python or child_python.startswith(f"{current_python}.")
    ):
        raise _fail("CPU report Python runtime differs from comparator runtime")
    return parsed, data, source_gate_passed


def _validate_record(
    record: object,
    *,
    sequence: int,
    previous_sha256: str,
    repository: str | None,
) -> str:
    expected_fields = {
        "diagnostics",
        "kind",
        "metrics",
        "previous_sha256",
        "record_sha256",
        "run",
        "schema_version",
        "segment",
        "segment_sha256",
        "sequence",
        "source",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise _fail(f"series record {sequence} has an unexpected schema")
    if (
        record["kind"] != "record"
        or type(record["schema_version"]) is not int
        or record["schema_version"] != SERIES_SCHEMA_VERSION
    ):
        raise _fail(f"series record {sequence} has an invalid kind/schema")
    if type(record["sequence"]) is not int or record["sequence"] != sequence:
        raise _fail(f"series record {sequence} has a non-contiguous sequence")
    if record["previous_sha256"] != previous_sha256:
        raise _fail(f"series record {sequence} breaks the hash chain")
    _validate_digest(record["record_sha256"], field="record_sha256")

    run = record["run"]
    if not isinstance(run, dict) or set(run) != {
        "recorded_at",
        "repository",
        "run_attempt",
        "run_id",
        "sha",
    }:
        raise _fail(f"series record {sequence} has an invalid run identity")
    run_repository = _validate_repository(run["repository"])
    if repository is not None and run_repository != repository:
        raise _fail("one series may not mix repositories")
    _validate_positive_decimal(run["run_id"], field="run_id")
    if type(run["run_attempt"]) is not int or run["run_attempt"] <= 0:
        raise _fail("run_attempt must be a positive integer")
    _validate_commit(run["sha"])
    recorded_at = run["recorded_at"]
    if not isinstance(recorded_at, str) or _UTC_RE.fullmatch(recorded_at) is None:
        raise _fail("recorded_at must be a UTC RFC3339 timestamp")
    try:
        datetime.fromisoformat(recorded_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise _fail("recorded_at must be a valid UTC RFC3339 timestamp") from error

    segment = _validate_segment(record["segment"], repository=run_repository)
    if record["segment_sha256"] != _sha256_value(segment):
        raise _fail(f"series record {sequence} has an invalid segment digest")
    metrics = record["metrics"]
    if not isinstance(metrics, dict) or tuple(metrics) != tuple(sorted(_HARD_NAMES)):
        raise _fail(f"series record {sequence} has an invalid hard metric set")
    for name in _HARD_NAMES:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _fail(f"series record {sequence} metric {name} is invalid")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as error:
            raise _fail(
                f"series record {sequence} metric {name} is invalid"
            ) from error
        if not math.isfinite(numeric) or numeric <= 0:
            raise _fail(f"series record {sequence} metric {name} is invalid")
    _validate_json(record["diagnostics"], path=f"record[{sequence}].diagnostics")
    source = record["source"]
    if not isinstance(source, dict) or set(source) != {
        "report_sha256",
        "source_gate_passed",
    }:
        raise _fail(f"series record {sequence} has an invalid source identity")
    _validate_digest(source["report_sha256"], field="report_sha256")
    if type(source["source_gate_passed"]) is not bool:
        raise _fail("source_gate_passed must be boolean")

    unhashed = dict(record)
    unhashed.pop("record_sha256")
    if record["record_sha256"] != _sha256_value(unhashed):
        raise _fail(f"series record {sequence} has an invalid self hash")
    return run_repository


def _load_series_bytes(data: bytes, *, label: str) -> list[dict[str, Any]]:
    if len(data) > MAX_SERIES_BYTES:
        raise _fail(f"{label} exceeds {MAX_SERIES_BYTES} bytes")
    if not data or not data.endswith(b"\n"):
        raise _fail(f"{label} is empty or truncated")
    raw_lines = data[:-1].split(b"\n")
    if len(raw_lines) < 2 or any(not line for line in raw_lines):
        raise _fail(f"{label} has invalid JSONL framing")
    values: list[Any] = []
    for index, line in enumerate(raw_lines, start=1):
        value = _strict_json(line, label=f"{label} line {index}")
        if line != _canonical_bytes(value):
            raise _fail(f"{label} line {index} is not canonical JSON")
        values.append(value)
    if _canonical_bytes(values[0]) != _canonical_bytes(_header()):
        raise _fail(f"{label} has an invalid header")
    trailer = values[-1]
    records = values[1:-1]
    if not isinstance(trailer, dict) or set(trailer) != {
        "kind",
        "record_count",
        "schema_version",
        "tip_sha256",
    }:
        raise _fail(f"{label} has an invalid trailer schema")
    if (
        trailer["kind"] != "trailer"
        or type(trailer["schema_version"]) is not int
        or trailer["schema_version"] != SERIES_SCHEMA_VERSION
    ):
        raise _fail(f"{label} has an invalid trailer kind/schema")
    if type(trailer["record_count"]) is not int or trailer["record_count"] != len(records):
        raise _fail(f"{label} trailer count does not match the records")

    previous = _genesis_sha256()
    repository: str | None = None
    identities: set[tuple[str, str, int]] = set()
    validated: list[dict[str, Any]] = []
    for sequence, record in enumerate(records, start=1):
        repository = _validate_record(
            record,
            sequence=sequence,
            previous_sha256=previous,
            repository=repository,
        )
        assert isinstance(record, dict)
        run = record["run"]
        identity = (run["repository"], run["run_id"], run["run_attempt"])
        if identity in identities:
            raise _fail(f"{label} repeats run identity {identity!r}")
        identities.add(identity)
        previous = record["record_sha256"]
        validated.append(record)
    if trailer["tip_sha256"] != previous:
        raise _fail(f"{label} trailer tip does not match the chain")
    return validated


def load_series(path: Path) -> list[dict[str, Any]]:
    data = _read_limited(path, limit=MAX_SERIES_BYTES, label="series")
    return _load_series_bytes(data, label="series")


def _tip_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tip: dict[str, Any] | None = None
    if records:
        run = records[-1]["run"]
        tip = {
            "record_sha256": records[-1]["record_sha256"],
            "repository": run["repository"],
            "run_attempt": run["run_attempt"],
            "run_id": run["run_id"],
            "sha": run["sha"],
        }
    return {
        "record_count": len(records),
        "tip": tip,
        "tip_sha256": (
            records[-1]["record_sha256"] if records else _genesis_sha256()
        ),
        "valid": True,
    }


def _record(
    *,
    report_path: Path,
    history_path: Path,
    series_path: Path,
    comparison_path: Path,
    repository: str,
    run_id: str,
    run_attempt: int,
    sha: str,
) -> int:
    roles = {
        "comparison": comparison_path,
        "history": history_path,
        "report": report_path,
        "series": series_path,
    }
    resolved: dict[Path, str] = {}
    for role, path in roles.items():
        try:
            identity_path = path.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise _fail(f"cannot resolve {role} path") from error
        if identity_path in resolved:
            raise _fail(f"{role} path aliases {resolved[identity_path]} path")
        resolved[identity_path] = role

    repository = _validate_repository(repository)
    run_id = _validate_positive_decimal(run_id, field="run_id")
    if run_attempt <= 0:
        raise _fail("run_attempt must be positive")
    sha = _validate_commit(sha)
    report, report_bytes, source_gate_passed = _validated_report(
        report_path,
        expected_sha=sha,
    )
    if not history_path.exists():
        raise _fail("history does not exist; run select-history first")
    records = load_series(history_path)
    identity = (repository, run_id, run_attempt)
    if any(
        (item["run"]["repository"], item["run"]["run_id"], item["run"]["run_attempt"])
        == identity
        for item in records
    ):
        raise _fail(f"run identity {identity!r} already exists")
    if records and records[-1]["run"]["repository"] != repository:
        raise _fail("history belongs to a different repository")

    segment = _segment(repository=repository)
    segment_sha256 = _sha256_value(segment)
    metrics = _hard_metrics(report)
    run = {
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "repository": repository,
        "run_attempt": run_attempt,
        "run_id": run_id,
        "sha": sha,
    }
    current: dict[str, Any] = {
        "diagnostics": _diagnostics(report),
        "kind": "record",
        "metrics": metrics,
        "previous_sha256": (
            records[-1]["record_sha256"] if records else _genesis_sha256()
        ),
        "run": run,
        "schema_version": SERIES_SCHEMA_VERSION,
        "segment": segment,
        "segment_sha256": segment_sha256,
        "sequence": len(records) + 1,
        "source": {
            "report_sha256": _sha256_bytes(report_bytes),
            "source_gate_passed": source_gate_passed,
        },
    }
    current["record_sha256"] = _sha256_value(current)

    compatible = [
        item for item in records if item["segment_sha256"] == segment_sha256
    ]
    window = compatible[-WINDOW_SIZE:]
    metric_comparisons: dict[str, Any] = {}
    regressions: list[str] = []
    for name in _HARD_NAMES:
        samples = [
            {
                "record_sha256": item["record_sha256"],
                "run_attempt": item["run"]["run_attempt"],
                "run_id": item["run"]["run_id"],
                "value": item["metrics"][name],
            }
            for item in window
        ]
        if len(window) < MIN_HISTORY:
            metric_comparisons[name] = {
                "baseline_median": None,
                "baseline_samples": samples,
                "current": metrics[name],
                "decline_fraction": None,
                "regressed": False,
                "status": "warmup",
            }
            continue
        ordered = sorted(Decimal(str(item["metrics"][name])) for item in window)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            baseline = ordered[midpoint]
        else:
            baseline = (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)
        current_decimal = Decimal(str(metrics[name]))
        decline = (baseline - current_decimal) / baseline
        regressed = current_decimal <= baseline * (Decimal(1) - DECLINE_FRACTION)
        if regressed:
            regressions.append(name)
        metric_comparisons[name] = {
            "baseline_median": float(baseline),
            "baseline_samples": samples,
            "current": metrics[name],
            "decline_fraction": str(decline),
            "regressed": regressed,
            "status": "regression" if regressed else "pass",
        }

    if regressions:
        status = "regression"
    elif not source_gate_passed:
        status = "source_gate_failed"
    elif len(window) < MIN_HISTORY:
        status = "warmup"
    else:
        status = "pass"
    comparison = {
        "all_passed": source_gate_passed and not regressions,
        "compatible_history_count": len(compatible),
        "current_record_sha256": current["record_sha256"],
        "decline_boundary_inclusive": True,
        "decline_fraction": float(DECLINE_FRACTION),
        "history_record_count": len(records),
        "maximum_window": WINDOW_SIZE,
        "metrics": metric_comparisons,
        "minimum_history": MIN_HISTORY,
        "regressions": regressions,
        "run": run,
        "schema_version": SERIES_SCHEMA_VERSION,
        "segment_sha256": segment_sha256,
        "source_gate_passed": source_gate_passed,
        "status": status,
        "window_size": len(window),
    }
    updated = [*records, current]
    rendered_series = _render_series(updated)
    if len(rendered_series) > MAX_SERIES_BYTES:
        raise _fail(f"updated series exceeds {MAX_SERIES_BYTES} bytes")
    rendered_comparison = json.dumps(
        comparison,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write(series_path, rendered_series)
    _atomic_write(comparison_path, rendered_comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if comparison["all_passed"] else 1


def _select_history(*, candidates: Path, output: Path) -> int:
    if not candidates.exists():
        raise _fail("candidate root does not exist")
    if candidates.is_symlink() or not candidates.is_dir():
        raise _fail("candidate root must be a directory")
    try:
        children = sorted(candidates.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise _fail("cannot enumerate candidate root") from error
    if not children:
        _atomic_write(output, _render_series([]))
        print(json.dumps({"record_count": 0, "selected": None, "valid": True}))
        return 0
    for child in children:
        candidate = child / "series.jsonl"
        if (
            child.is_symlink()
            or not child.is_dir()
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            print(f"skipping invalid history candidate {child.name}", file=sys.stderr)
            continue
        try:
            data = _read_limited(
                candidate,
                limit=MAX_SERIES_BYTES,
                label=f"history candidate {child.name}",
            )
            records = _load_series_bytes(data, label=f"history candidate {child.name}")
        except SeriesError as error:
            print(f"skipping invalid history candidate {child.name}: {error}", file=sys.stderr)
            continue
        _atomic_write(output, data)
        summary = _tip_summary(records)
        summary["selected"] = child.name
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise _fail("candidate root is non-empty but contains no valid history")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate one complete series")
    validate.add_argument("--series", type=Path, required=True)

    select = commands.add_parser(
        "select-history",
        help="copy the newest valid candidate series, or create genesis",
    )
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    record = commands.add_parser("record", help="append and compare one CPU report")
    record.add_argument("--report", type=Path, required=True)
    record.add_argument("--history", type=Path, required=True)
    record.add_argument("--series", type=Path, required=True)
    record.add_argument("--comparison", type=Path, required=True)
    record.add_argument("--repository", required=True)
    record.add_argument("--run-id", required=True)
    record.add_argument("--run-attempt", type=int, required=True)
    record.add_argument("--sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            records = load_series(args.series)
            print(json.dumps(_tip_summary(records), indent=2, sort_keys=True))
            return 0
        if args.command == "select-history":
            return _select_history(candidates=args.candidates, output=args.output)
        if args.command == "record":
            return _record(
                report_path=args.report,
                history_path=args.history,
                series_path=args.series,
                comparison_path=args.comparison,
                repository=args.repository,
                run_id=args.run_id,
                run_attempt=args.run_attempt,
                sha=args.sha,
            )
        raise AssertionError(f"unhandled command {args.command!r}")
    except SeriesError as error:
        print(f"cpu performance series error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
