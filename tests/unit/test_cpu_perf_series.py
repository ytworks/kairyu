from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import asdict
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from scripts import cpu_perf_series as series

_SHA = "a" * 40
_REPOSITORY = "owner/repository"


def _work_case(
    sizes: tuple[int, int],
    operations: tuple[int, int],
    per_operation: int,
) -> dict[str, list[int]]:
    return {
        "sizes": list(sizes),
        "operations": list(operations),
        "work": [count * per_operation for count in operations],
    }


def _results(value: float = 100.0) -> dict[str, dict[str, object]]:
    proc_lengths = (128, 256, 512, 1024)
    queue_sizes = series.gate.SCHEDULER_WORK_SIZES
    return {
        "scheduler_queue": {
            "requests": 30_000,
            "removals": 3_000,
            "work": {
                "unit": "test",
                "fifo_full_drain": _work_case(queue_sizes, queue_sizes, 7),
                "indexed_removal": _work_case(
                    queue_sizes,
                    (queue_sizes[0] // 10, queue_sizes[1] // 10),
                    4,
                ),
                "priority_full_drain": _work_case(queue_sizes, queue_sizes, 30),
            },
            "fifo_full_drain": {
                "median_speedup": value,
                "legacy": {"median_seconds": 0.3, "repeats": 5},
                "indexed": {"median_seconds": 0.1, "repeats": 5},
            },
            "indexed_removal": {
                "median_speedup": value,
                "legacy": {"median_seconds": 0.3, "repeats": 5},
                "indexed": {"median_seconds": 0.1, "repeats": 5},
            },
            "priority_full_drain": {
                "median_speedup": value,
                "legacy": {"median_seconds": 0.3, "repeats": 5},
                "indexed": {"median_seconds": 0.1, "repeats": 5},
            },
        },
        "radix_eviction": {
            "leaves": 10_000,
            "evictions": 25,
            "median_speedup": value,
            "scanning": {"median_seconds": 0.3, "repeats": 3},
            "indexed": {"median_seconds": 0.1, "repeats": 3},
            "work": {
                "unit": "test",
                "eviction": _work_case(
                    series.gate.RADIX_WORK_SIZES, (25, 25), 31
                ),
            },
        },
        "op_queue": {
            "operations": 50_000,
            "producers": 1,
            "work": {
                "unit": "test",
                "add_burst": _work_case(
                    series.gate.OP_QUEUE_WORK_SIZES,
                    series.gate.OP_QUEUE_WORK_SIZES,
                    8,
                ),
                "duplicate_abort_burst": _work_case(
                    series.gate.OP_QUEUE_WORK_SIZES,
                    series.gate.OP_QUEUE_WORK_SIZES,
                    2,
                ),
            },
            "add_burst": {
                "elapsed_speedup": value,
                "legacy": {"elapsed_seconds": 0.3, "repeats": 5},
                "coalesced": {
                    "elapsed_seconds": 0.1,
                    "queued_operation_containers": 1,
                    "queued_ids": 50_000,
                    "repeats": 5,
                },
            },
            "duplicate_abort_burst": {
                "elapsed_speedup": value,
                "legacy": {"elapsed_seconds": 0.3, "repeats": 5},
                "coalesced": {
                    "elapsed_seconds": 0.1,
                    "queued_operation_containers_including_add": 2,
                    "queued_abort_ids": 1,
                    "repeats": 5,
                },
            },
        },
        "sampler_penalty_state": {
            "device": "cpu",
            "vocab_size": 151_936,
            "history_length": 32_768,
            "pending_depth": 2,
            "penalties": "all",
            "iterations": 20,
            "repeats": 3,
            "validated_bitwise": True,
            "work": {
                "unit": "test",
                "steady_state": _work_case(
                    series.gate.SAMPLER_WORK_SIZES, (20, 20), 2
                ),
                "append": _work_case(
                    series.gate.SAMPLER_WORK_SIZES, (20, 20), 3
                ),
            },
            "legacy_speedup": value,
            "append_legacy_speedup": value,
            "overlay_speedup": 1.1,
            "append_overlay_speedup": 1.2,
            "modes": {
                "committed_plus_overlay": {"median_us": 100.0},
                "incremental_effective_counts": {"median_us": 90.0},
                "legacy_full_history": {"median_us": 1000.0},
            },
            "append_modes_median_us": {
                "committed_plus_overlay": 110.0,
                "incremental_effective_counts": 95.0,
                "legacy_full_history": 1050.0,
            },
        },
        "proc_wire": {
            "schema_version": 1,
            "metric": "msgpack_frame_bytes",
            "protocols": {"legacy": 1, "delta": 2},
            "thresholds": {
                "max_v2_growth_per_doubling": 2.15,
                "min_legacy_growth_per_doubling": 3.5,
            },
            "measurements": [
                {
                    "output_tokens": length,
                    "legacy_total_bytes": length * length * 10,
                    "v2_total_bytes": length * 100,
                }
                for length in proc_lengths
            ],
            "growth": [
                {
                    "from_tokens": start,
                    "to_tokens": end,
                    "legacy_empirical_exponent": 2.0,
                    "legacy_ratio": 4.0,
                    "v2_empirical_exponent": 1.0,
                    "v2_ratio": 2.0,
                }
                for start, end in pairwise(proc_lengths)
            ],
            "checks": [
                {"name": name, "passed": True}
                for name in sorted(series.gate.PROC_WIRE_EXPECTED_CHECKS)
            ],
            "environment": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.0",
                "platform": "test",
                "msgpack": [1, 2, 1],
            },
            "source": {"commit": _SHA, "tracked_dirty": False},
            "all_passed": True,
        },
        "router_latency": {
            "routes": 10_000,
            "p50_seconds": 0.00001,
            "p99_seconds": 0.00002,
            "work": {
                "unit": "test",
                "routes_served": _work_case((0, 10_000), (4, 4), 33),
            },
            "max_calls_per_route": 1_000,
            "max_work_growth": 2.0,
            "passed": True,
        },
    }


def _report(value: float = 100.0) -> dict[str, object]:
    benchmarks = _results(value)
    checks = series.gate.evaluate_results(benchmarks)
    return {
        "schema_version": 2,
        "cpu_only": True,
        "benchmarks": benchmarks,
        "checks": [asdict(check) for check in checks],
        "diagnostics": {"timing": series.gate.timing_diagnostics(benchmarks)},
        "errors": {},
        "all_passed": all(check.passed for check in checks),
    }


def _reevaluate(report: dict[str, object]) -> None:
    checks = series.gate.evaluate_results(report["benchmarks"])
    report["checks"] = [asdict(check) for check in checks]
    report["all_passed"] = all(check.passed for check in checks)


def _write_report(path: Path, report: object) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _record_argv(
    root: Path,
    *,
    history: Path | None = None,
    output: Path | None = None,
    comparison: Path | None = None,
    run_id: int = 1,
) -> list[str]:
    return [
        "record",
        "--report",
        str(root / "report.json"),
        "--history",
        str(root / "history.jsonl" if history is None else history),
        "--series",
        str(root / "series.jsonl" if output is None else output),
        "--comparison",
        str(root / "comparison.json" if comparison is None else comparison),
        "--repository",
        _REPOSITORY,
        "--run-id",
        str(run_id),
        "--run-attempt",
        "1",
        "--sha",
        _SHA,
    ]


def _append(
    root: Path,
    *,
    run_id: int,
    value: float = 100.0,
    report: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    report_path = root / "report.json"
    history_path = root / "history.jsonl"
    output_path = root / "series.jsonl"
    comparison_path = root / "comparison.json"
    _write_report(report_path, _report(value) if report is None else report)
    if not history_path.exists():
        history_path.write_bytes(series._render_series([]))
    code = series.main(_record_argv(root, run_id=run_id))
    if output_path.exists():
        shutil.copyfile(output_path, history_path)
    comparison = (
        json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison_path.exists()
        else {}
    )
    return code, comparison


def test_trend_metrics_and_compatibility_exclude_source_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ImageOS", "ubuntu24")
    report = _report()
    metrics = series._trend_metrics(report)
    segment = series._segment(repository=_REPOSITORY)

    assert tuple(metrics) == tuple(name for name, _path in series.TREND_METRICS)
    assert all(value == 100.0 for value in metrics.values())
    assert set(segment) == {"compatibility", "policy", "runtime", "workload"}
    assert segment["runtime"]["hosted_image_class"] == "ubuntu24"
    assert _SHA not in json.dumps(segment, sort_keys=True)
    assert series.MAX_REPORT_BYTES == series.MAX_SERIES_BYTES == 64 * 1024 * 1024


def test_speed_decline_is_reported_and_never_fails(tmp_path: Path) -> None:
    for run_id in range(1, 6):
        code, comparison = _append(tmp_path, run_id=run_id, value=200.0)
        assert code == 0
        assert comparison["status"] == "warmup"

    code, comparison = _append(tmp_path, run_id=6, value=170.0)

    assert code == 0
    assert comparison["all_passed"] is True
    assert comparison["status"] == "pass"
    assert comparison["trend"] == "report-only"
    assert set(comparison["declines"]) == set(comparison["metrics"])
    for metric in comparison["metrics"].values():
        assert metric["baseline_median"] == 200.0
        assert [sample["value"] for sample in metric["baseline_samples"]] == [
            200.0
        ] * 5
        assert metric["current"] == 170.0
        assert Decimal(metric["decline_fraction"]) == Decimal("0.15")
        assert metric["declined"] is True
        assert metric["status"] == "decline"
    assert (tmp_path / "series.jsonl").is_file()
    assert (tmp_path / "comparison.json").is_file()


def test_trailing_window_is_seven_and_odd_even_medians_are_exact(
    tmp_path: Path,
) -> None:
    for run_id, value in enumerate((10, 20, 30, 40, 50, 60), start=1):
        _append(tmp_path, run_id=run_id, value=float(value))
    _code, even = _append(tmp_path, run_id=7, value=100.0)
    assert even["window_size"] == 6
    assert {
        metric["baseline_median"] for metric in even["metrics"].values()
    } == {35.0}

    _append(tmp_path, run_id=8, value=70.0)
    _code, odd = _append(tmp_path, run_id=9, value=80.0)
    assert odd["compatible_history_count"] == 8
    assert odd["window_size"] == 7
    assert {
        metric["baseline_median"] for metric in odd["metrics"].values()
    } == {50.0}
    assert [
        sample["run_id"]
        for sample in next(iter(odd["metrics"].values()))["baseline_samples"]
    ] == ["2", "3", "4", "5", "6", "7", "8"]


def test_compatibility_segments_ignore_intervening_incompatible_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ImageOS", "ubuntu24")
    for run_id in range(1, 6):
        _append(tmp_path, run_id=run_id, value=200.0)
    monkeypatch.setenv("ImageOS", "ubuntu22")
    _code, changed = _append(tmp_path, run_id=6)
    assert changed["compatible_history_count"] == 0
    assert changed["status"] == "warmup"

    monkeypatch.setenv("ImageOS", "ubuntu24")
    _code, restored = _append(tmp_path, run_id=7, value=170.0)
    assert restored["compatible_history_count"] == 5
    assert set(restored["declines"]) == set(restored["metrics"])


def _linear_scheduler_removal(report: dict[str, object]) -> None:
    case = report["benchmarks"]["scheduler_queue"]["work"]["indexed_removal"]
    # Each removal scans the whole queue: work per removal grows with size.
    case["work"] = [
        count * size
        for count, size in zip(case["operations"], case["sizes"], strict=True)
    ]


def _accumulating_router(report: dict[str, object]) -> None:
    case = report["benchmarks"]["router_latency"]["work"]["routes_served"]
    case["work"] = [132, 132 * 1_000]
    report["benchmarks"]["router_latency"]["passed"] = False


@pytest.mark.parametrize(
    ("regress", "failed_check"),
    [
        (_linear_scheduler_removal, "scheduler.indexed_removal.work_growth"),
        (_accumulating_router, "router.routes_served.work_growth"),
    ],
)
def test_operation_count_regression_fails_and_is_retained(
    tmp_path: Path,
    regress,
    failed_check: str,
) -> None:
    report = _report()
    regress(report)
    _reevaluate(report)
    assert failed_check in {
        check["name"] for check in report["checks"] if not check["passed"]
    }

    code, comparison = _append(tmp_path, run_id=1, report=report)

    assert code == 1
    assert comparison["source_gate_passed"] is False
    assert comparison["status"] == "source_gate_failed"
    assert series.load_series(tmp_path / "series.jsonl")[-1]["source"] == {
        "report_sha256": series._sha256_bytes(
            (tmp_path / "report.json").read_bytes()
        ),
        "source_gate_passed": False,
    }


def test_unavailable_history_is_a_warmup_decided_by_the_gate(
    tmp_path: Path,
) -> None:
    def warmup(report: dict[str, object]) -> tuple[int, dict[str, object]]:
        _write_report(tmp_path / "report.json", report)
        code = series.main(
            [
                "warmup",
                "--report",
                str(tmp_path / "report.json"),
                "--comparison",
                str(tmp_path / "comparison.json"),
                "--repository",
                _REPOSITORY,
                "--run-id",
                "1",
                "--run-attempt",
                "1",
                "--sha",
                _SHA,
            ]
        )
        comparison = json.loads(
            (tmp_path / "comparison.json").read_text(encoding="utf-8")
        )
        return code, comparison

    code, comparison = warmup(_report())
    assert code == 0
    assert comparison["status"] == "warmup"
    assert comparison["history"] == "unavailable"
    assert not (tmp_path / "series.jsonl").exists()

    regressed = _report()
    _linear_scheduler_removal(regressed)
    _reevaluate(regressed)
    code, comparison = warmup(regressed)
    assert code == 1
    assert comparison["status"] == "source_gate_failed"


@pytest.mark.parametrize(
    ("report_path", "value"),
    [
        (("benchmarks", "scheduler_queue", "requests"), 1),
        (
            (
                "benchmarks",
                "scheduler_queue",
                "fifo_full_drain",
                "legacy",
                "repeats",
            ),
            4,
        ),
        (
            (
                "benchmarks",
                "scheduler_queue",
                "work",
                "indexed_removal",
                "sizes",
            ),
            [1_000, 8_000],
        ),
        (("benchmarks", "radix_eviction", "leaves"), 1),
        (("benchmarks", "op_queue", "operations"), 1),
        (("benchmarks", "sampler_penalty_state", "device"), "cuda"),
        (("benchmarks", "proc_wire", "schema_version"), 2),
        (("benchmarks", "proc_wire", "protocols", "delta"), 3),
        (("benchmarks", "router_latency", "routes"), 1),
    ],
)
def test_fixed_workload_or_shape_mismatch_is_never_recorded(
    tmp_path: Path,
    report_path: tuple[str, ...],
    value: object,
) -> None:
    report = _report()
    parent = series._at(report, *report_path[:-1])
    assert isinstance(parent, dict)
    parent[report_path[-1]] = value
    _reevaluate(report)

    code, comparison = _append(tmp_path, run_id=1, report=report)

    assert code == 2
    assert comparison == {}
    assert not (tmp_path / "series.jsonl").exists()


def test_proc_wire_diagnostic_inventory_is_bounded(tmp_path: Path) -> None:
    report = _report()
    proc_wire = report["benchmarks"]["proc_wire"]
    proc_wire["growth"].append(dict(proc_wire["growth"][-1]))

    code, comparison = _append(tmp_path, run_id=1, report=report)

    assert code == 2
    assert comparison == {}
    assert not (tmp_path / "series.jsonl").exists()


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":1,"schema_version":1}\n',
        b'{"schema_version":NaN}\n',
        b'{"schema_version":Infinity}\n',
        b'{"schema_version":1e999}\n',
    ],
)
def test_report_parser_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    report = tmp_path / "report.json"
    report.write_bytes(payload)

    code = series.main(
        [
            "record",
            "--report",
            str(report),
            "--history",
            str(tmp_path / "missing.jsonl"),
            "--series",
            str(tmp_path / "series.jsonl"),
            "--comparison",
            str(tmp_path / "comparison.json"),
            "--repository",
            _REPOSITORY,
            "--run-id",
            "1",
            "--run-attempt",
            "1",
            "--sha",
            _SHA,
        ]
    )

    assert code == 2
    assert not (tmp_path / "series.jsonl").exists()
    assert not (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize("value", [True, 0, -1, float("inf")])
def test_trend_metrics_must_be_positive_finite_numbers(
    tmp_path: Path,
    value: object,
) -> None:
    report = _report()
    report["benchmarks"]["scheduler_queue"]["fifo_full_drain"][
        "median_speedup"
    ] = value
    if type(value) in (int, float) and math.isfinite(value):
        _reevaluate(report)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(report, allow_nan=True),
        encoding="utf-8",
    )

    assert series.main(
        [
            "record",
            "--report",
            str(report_path),
            "--history",
            str(tmp_path / "history.jsonl"),
            "--series",
            str(tmp_path / "series.jsonl"),
            "--comparison",
            str(tmp_path / "comparison.json"),
            "--repository",
            _REPOSITORY,
            "--run-id",
            "1",
            "--run-attempt",
            "1",
            "--sha",
            _SHA,
        ]
    ) == 2
    assert not (tmp_path / "series.jsonl").exists()
    assert not (tmp_path / "comparison.json").exists()


def test_record_requires_authoritative_history_and_distinct_paths(
    tmp_path: Path,
) -> None:
    _write_report(tmp_path / "report.json", _report())

    assert series.main(_record_argv(tmp_path)) == 2
    assert not (tmp_path / "series.jsonl").exists()

    history = tmp_path / "history.jsonl"
    history.write_bytes(series._render_series([]))
    aliased = tmp_path / "aliased-output"
    assert (
        series.main(
            _record_argv(tmp_path, output=aliased, comparison=aliased)
        )
        == 2
    )
    assert not aliased.exists()
    assert series.load_series(history) == []


def test_huge_integer_report_fails_closed_without_an_uncaught_error(
    tmp_path: Path,
) -> None:
    report = _report()
    report["benchmarks"]["scheduler_queue"]["fifo_full_drain"][
        "median_speedup"
    ] = 10**1_000
    _write_report(tmp_path / "report.json", report)
    (tmp_path / "history.jsonl").write_bytes(series._render_series([]))

    assert series.main(_record_argv(tmp_path)) == 2
    assert not (tmp_path / "series.jsonl").exists()
    assert not (tmp_path / "comparison.json").exists()


def test_extreme_finite_metrics_keep_comparison_json_finite(tmp_path: Path) -> None:
    for run_id in range(1, 6):
        code, _comparison = _append(tmp_path, run_id=run_id, value=5e-324)
        assert code == 0

    code, comparison = _append(tmp_path, run_id=6, value=1e308)

    assert code == 0
    for metric in comparison["metrics"].values():
        assert metric["status"] == "steady"
        assert Decimal(metric["decline_fraction"]).is_finite()
        assert Decimal(metric["decline_fraction"]) < 0


def test_updated_series_must_remain_within_the_reader_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genesis = series._render_series([])
    monkeypatch.setattr(series, "MAX_SERIES_BYTES", len(genesis) + 100)

    code, comparison = _append(tmp_path, run_id=1)

    assert code == 2
    assert comparison == {}
    assert not (tmp_path / "series.jsonl").exists()
    assert series.load_series(tmp_path / "history.jsonl") == []


def test_deeply_nested_json_and_invalid_output_parent_fail_closed(
    tmp_path: Path,
) -> None:
    history = tmp_path / "history.jsonl"
    history.write_bytes(series._render_series([]))
    (tmp_path / "report.json").write_bytes(b"[" * 1_100 + b"0" + b"]" * 1_100)
    assert series.main(_record_argv(tmp_path)) == 2

    _write_report(tmp_path / "report.json", _report())
    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("occupied", encoding="utf-8")
    assert (
        series.main(
            _record_argv(tmp_path, output=invalid_parent / "series.jsonl")
        )
        == 2
    )
    assert not (tmp_path / "comparison.json").exists()


def test_series_rejects_tamper_reorder_truncation_and_noncanonical_lines(
    tmp_path: Path,
) -> None:
    for run_id in range(1, 4):
        _append(tmp_path, run_id=run_id)
    valid = (tmp_path / "series.jsonl").read_bytes()
    lines = valid.splitlines(keepends=True)
    mutations = {
        "tamper": valid.replace(b'"run_id":"2"', b'"run_id":"9"', 1),
        "reorder": b"".join([lines[0], lines[2], lines[1], *lines[3:]]),
        "middle_missing": b"".join([lines[0], lines[1], *lines[3:]]),
        "truncated": valid[:-1],
        "trailer_missing": b"".join(lines[:-1]),
        "noncanonical": lines[0].replace(b"{", b"{ ", 1) + b"".join(lines[1:]),
    }
    for name, payload in mutations.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(payload)
        with pytest.raises(series.SeriesError):
            series.load_series(path)


def test_duplicate_run_is_rejected_even_with_a_rehashed_valid_chain(
    tmp_path: Path,
) -> None:
    _append(tmp_path, run_id=1)
    _append(tmp_path, run_id=2)
    records = series.load_series(tmp_path / "series.jsonl")
    records[1]["run"]["run_id"] = "1"
    previous = series._genesis_sha256()
    for sequence, record in enumerate(records, start=1):
        record["sequence"] = sequence
        record["previous_sha256"] = previous
        record.pop("record_sha256")
        record["record_sha256"] = series._sha256_value(record)
        previous = record["record_sha256"]
    path = tmp_path / "duplicate.jsonl"
    path.write_bytes(series._render_series(records))

    with pytest.raises(series.SeriesError, match="repeats run identity"):
        series.load_series(path)


def test_select_history_skips_invalid_newest_and_fails_if_none_valid(
    tmp_path: Path,
) -> None:
    _append(tmp_path, run_id=1)
    valid = (tmp_path / "series.jsonl").read_bytes()
    candidates = tmp_path / "candidates"
    (candidates / "000-newest").mkdir(parents=True)
    (candidates / "000-newest" / "series.jsonl").write_text("broken\n", encoding="utf-8")
    (candidates / "001-older").mkdir()
    (candidates / "001-older" / "series.jsonl").write_bytes(valid)
    selected = tmp_path / "selected.jsonl"

    assert series.main(
        [
            "select-history",
            "--candidates",
            str(candidates),
            "--output",
            str(selected),
        ]
    ) == 0
    assert selected.read_bytes() == valid

    (candidates / "001-older" / "series.jsonl").write_text("also broken\n", encoding="utf-8")
    failed_output = tmp_path / "not-written.jsonl"
    assert series.main(
        [
            "select-history",
            "--candidates",
            str(candidates),
            "--output",
            str(failed_output),
        ]
    ) == 2
    assert not failed_output.exists()


def test_empty_candidate_root_creates_a_valid_genesis(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    output = tmp_path / "history.jsonl"

    assert series.main(
        [
            "select-history",
            "--candidates",
            str(candidates),
            "--output",
            str(output),
        ]
    ) == 0
    assert series.load_series(output) == []
    assert series.main(["validate", "--series", str(output)]) == 0


def test_missing_candidate_root_cannot_silently_reset_history(
    tmp_path: Path,
) -> None:
    output = tmp_path / "history.jsonl"

    assert (
        series.main(
            [
                "select-history",
                "--candidates",
                str(tmp_path / "missing-candidates"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_symlinked_candidate_root_cannot_create_genesis(tmp_path: Path) -> None:
    target = tmp_path / "actual-candidates"
    target.mkdir()
    candidates = tmp_path / "candidates-link"
    candidates.symlink_to(target, target_is_directory=True)
    output = tmp_path / "history.jsonl"

    assert (
        series.main(
            [
                "select-history",
                "--candidates",
                str(candidates),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_candidate_enumeration_error_is_a_hard_input_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    output = tmp_path / "history.jsonl"
    original_iterdir = Path.iterdir

    def denied_iterdir(path: Path):
        if path == candidates:
            raise PermissionError("denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", denied_iterdir)
    assert (
        series.main(
            [
                "select-history",
                "--candidates",
                str(candidates),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_report_checks_and_source_commit_are_recomputed(tmp_path: Path) -> None:
    wrong_checks = _report()
    wrong_checks["checks"] = []
    code, _comparison = _append(tmp_path, run_id=1, report=wrong_checks)
    assert code == 2

    wrong_source = _report()
    wrong_source["benchmarks"]["proc_wire"]["source"]["commit"] = "b" * 40
    code, _comparison = _append(tmp_path, run_id=2, report=wrong_source)
    assert code == 2


def test_validate_outputs_machine_readable_tip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _append(tmp_path, run_id=123)
    capsys.readouterr()

    assert series.main(["validate", "--series", str(tmp_path / "series.jsonl")]) == 0
    summary = json.loads(capsys.readouterr().out)

    assert summary["valid"] is True
    assert summary["record_count"] == 1
    assert summary["tip"]["repository"] == _REPOSITORY
    assert summary["tip"]["run_id"] == "123"
    assert summary["tip"]["run_attempt"] == 1
    assert summary["tip"]["sha"] == _SHA
