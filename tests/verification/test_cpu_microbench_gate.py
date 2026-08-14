from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import cpu_microbench_gate as gate


def _passing_results() -> dict[str, dict[str, object]]:
    return {
        "scheduler_queue": {
            "requests": 30_000,
            "removals": 3_000,
            "fifo_full_drain": {
                "median_speedup": 2.0,
                "legacy": {"repeats": 5},
                "indexed": {"repeats": 5},
            },
            "indexed_removal": {
                "median_speedup": 10.0,
                "legacy": {"repeats": 5},
                "indexed": {"repeats": 5},
            },
            "priority_full_drain": {
                "median_speedup": 0.90,
                "legacy": {"repeats": 5},
                "indexed": {"repeats": 5},
            },
        },
        "radix_eviction": {
            "leaves": 10_000,
            "evictions": 25,
            "median_speedup": 100.0,
            "scanning": {"repeats": 3},
            "indexed": {"repeats": 3},
        },
        "op_queue": {
            "operations": 50_000,
            "producers": 1,
            "add_burst": {
                "elapsed_speedup": 0.50,
                "legacy": {"repeats": 5},
                "coalesced": {
                    "queued_operation_containers": 1,
                    "queued_ids": 50_000,
                    "repeats": 5,
                },
            },
            "duplicate_abort_burst": {
                "elapsed_speedup": 0.50,
                "legacy": {"repeats": 5},
                "coalesced": {
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
            "legacy_speedup": 5.0,
            "append_legacy_speedup": 5.0,
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
                {"output_tokens": length}
                for length in (128, 256, 512, 1024)
            ],
            "checks": [
                {"name": name, "passed": True}
                for name in sorted(gate.PROC_WIRE_EXPECTED_CHECKS)
            ],
            "source": {"tracked_dirty": False},
            "all_passed": True,
        },
        "router_latency": {
            "routes": 10_000,
            "p99_seconds": 0.009,
            "p99_budget_seconds": 0.010,
            "passed": True,
        },
    }


def test_gate_inventory_is_exact_and_cpu_only() -> None:
    assert [spec.name for spec in gate.BENCHMARKS] == [
        "scheduler_queue",
        "radix_eviction",
        "op_queue",
        "sampler_penalty_state",
        "proc_wire",
        "router_latency",
    ]
    assert {spec.script for spec in gate.BENCHMARKS} == {
        "verification/l1/performance/scheduler_queue_bench.py",
        "verification/l1/performance/radix_eviction_bench.py",
        "verification/l1/performance/op_queue_bench.py",
        "verification/l1/performance/sampler_penalty_state_bench.py",
        "verification/l1/performance/proc_wire_bench.py",
        "verification/orchestration/performance/router_latency.py",
    }
    sampler = next(
        spec for spec in gate.BENCHMARKS if spec.name == "sampler_penalty_state"
    )
    assert ("--device", "cpu") == sampler.args[:2]
    assert gate.BENCHMARK_TIMEOUT_SECONDS * len(gate.BENCHMARKS) < 5 * 60


def test_variance_tolerant_bounds_accept_the_boundary() -> None:
    checks = gate.evaluate_results(_passing_results())

    assert checks
    assert all(check.passed for check in checks)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scheduler_queue", "priority_full_drain", "median_speedup"), 0.89),
        (("radix_eviction", "median_speedup"), 99.99),
        (("op_queue", "add_burst", "elapsed_speedup"), 0.49),
        (("op_queue", "producers"), 2),
        (("sampler_penalty_state", "pending_depth"), 1),
        (("sampler_penalty_state", "append_legacy_speedup"), 4.99),
        (("proc_wire", "schema_version"), 2),
        (("proc_wire", "all_passed"), 1),
        (("router_latency", "p99_seconds"), 0.0101),
        (("router_latency", "p99_budget_seconds"), 0.020),
    ],
)
def test_each_benchmark_can_fail_the_gate(
    path: tuple[str, ...],
    value: object,
) -> None:
    results = deepcopy(_passing_results())
    target: object = results
    for component in path[:-1]:
        assert isinstance(target, dict)
        target = target[component]
    assert isinstance(target, dict)
    target[path[-1]] = value

    assert not all(check.passed for check in gate.evaluate_results(results))


def test_proc_wire_fixed_lengths_and_checks_are_binding() -> None:
    wrong_lengths = deepcopy(_passing_results())
    measurements = wrong_lengths["proc_wire"]["measurements"]
    assert isinstance(measurements, list)
    measurements[0] = {"output_tokens": 64}
    assert not all(
        check.passed for check in gate.evaluate_results(wrong_lengths)
    )

    missing_check = deepcopy(_passing_results())
    missing_check["proc_wire"]["checks"] = []
    assert not all(
        check.passed for check in gate.evaluate_results(missing_check)
    )

    integer_passed = deepcopy(_passing_results())
    checks = integer_passed["proc_wire"]["checks"]
    assert isinstance(checks, list)
    assert isinstance(checks[0], dict)
    checks[0]["passed"] = 1
    assert not all(
        check.passed for check in gate.evaluate_results(integer_passed)
    )


def test_run_benchmark_uses_locked_interpreter_and_cpu_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, '{"metric": 1}\n', "")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    spec = gate.BenchmarkSpec("sample", "bench/sample.py", ("--fixed", "1"))

    assert gate.run_benchmark(spec, python="/locked/python", root=tmp_path) == {
        "metric": 1
    }
    assert captured["command"] == (
        "/locked/python",
        str(tmp_path / "bench/sample.py"),
        "--fixed",
        "1",
    )
    assert captured["cwd"] == tmp_path
    assert captured["timeout"] == gate.BENCHMARK_TIMEOUT_SECONDS
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["NUMEXPR_NUM_THREADS"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"


def test_run_benchmark_rejects_nonzero_and_non_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = gate.BenchmarkSpec("sample", "bench/sample.py", ())
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 7, "", "failed"
        ),
    )
    with pytest.raises(gate.BenchmarkFailure, match="exited 7"):
        gate.run_benchmark(spec)

    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "not json", ""
        ),
    )
    with pytest.raises(gate.BenchmarkFailure, match="one JSON object"):
        gate.run_benchmark(spec)


def test_main_retains_one_complete_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _passing_results()
    monkeypatch.setattr(
        gate,
        "run_benchmark",
        lambda spec: deepcopy(results[spec.name]),
    )
    output = tmp_path / "nested" / "report.json"

    assert gate.main(["--output", str(output)]) == 0
    rendered = json.loads(output.read_text(encoding="utf-8"))
    assert rendered["all_passed"] is True
    assert len(rendered["checks"]) > len(gate.BENCHMARKS)
    summary = json.loads(capsys.readouterr().out)
    assert summary["all_passed"] is True
    assert summary["report_path"] == str(output)
    assert summary["checks"] == rendered["checks"]
    assert "benchmarks" not in summary


def test_retryable_benchmarks_names_only_timing_ratio_failures() -> None:
    checks = gate.evaluate_results(_passing_results())
    assert gate.retryable_benchmarks(checks) == frozenset()

    noisy = deepcopy(_passing_results())
    noisy["scheduler_queue"]["priority_full_drain"]["median_speedup"] = 0.85
    noisy["router_latency"]["p99_seconds"] = 0.011
    failed = tuple(
        check for check in gate.evaluate_results(noisy) if not check.passed
    )
    assert {check.relation for check in failed} <= {">=", "<"}
    assert gate.retryable_benchmarks(failed) == frozenset(
        {"scheduler_queue", "router_latency"}
    )

    structural = deepcopy(_passing_results())
    structural["scheduler_queue"]["priority_full_drain"]["median_speedup"] = 0.85
    structural["op_queue"]["producers"] = 2
    failed = tuple(
        check for check in gate.evaluate_results(structural) if not check.passed
    )
    assert gate.retryable_benchmarks(failed) == frozenset()


def test_main_retries_timing_ratio_noise_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _passing_results()
    calls: list[str] = []

    def fake_run(spec: gate.BenchmarkSpec) -> dict[str, object]:
        calls.append(spec.name)
        result = deepcopy(results[spec.name])
        if spec.name == "scheduler_queue" and calls.count("scheduler_queue") == 1:
            result["priority_full_drain"]["median_speedup"] = 0.85
        return result

    monkeypatch.setattr(gate, "run_benchmark", fake_run)

    assert gate.main([]) == 0
    assert calls.count("scheduler_queue") == 2
    assert all(
        calls.count(spec.name) == 1
        for spec in gate.BENCHMARKS
        if spec.name != "scheduler_queue"
    )
    report = json.loads(capsys.readouterr().out)
    assert report["all_passed"] is True
    assert report["errors"] == {}


def test_main_fails_after_exhausting_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _passing_results()
    calls: list[str] = []

    def fake_run(spec: gate.BenchmarkSpec) -> dict[str, object]:
        calls.append(spec.name)
        result = deepcopy(results[spec.name])
        if spec.name == "scheduler_queue":
            result["priority_full_drain"]["median_speedup"] = 0.85
        return result

    monkeypatch.setattr(gate, "run_benchmark", fake_run)

    assert gate.main([]) == 1
    assert calls.count("scheduler_queue") == gate.MAX_BENCHMARK_ATTEMPTS
    report = json.loads(capsys.readouterr().out)
    assert report["all_passed"] is False
    failed = [check for check in report["checks"] if not check["passed"]]
    assert [check["name"] for check in failed] == ["scheduler.priority_speedup"]


def test_main_does_not_retry_structural_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _passing_results()
    calls: list[str] = []

    def fake_run(spec: gate.BenchmarkSpec) -> dict[str, object]:
        calls.append(spec.name)
        result = deepcopy(results[spec.name])
        if spec.name == "op_queue":
            result["producers"] = 2
        if spec.name == "scheduler_queue":
            result["priority_full_drain"]["median_speedup"] = 0.85
        return result

    monkeypatch.setattr(gate, "run_benchmark", fake_run)

    assert gate.main([]) == 1
    assert calls == [spec.name for spec in gate.BENCHMARKS]
    assert json.loads(capsys.readouterr().out)["all_passed"] is False


def test_report_top_level_schema_is_frozen_for_the_nightly_series(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _passing_results()
    monkeypatch.setattr(
        gate,
        "run_benchmark",
        lambda spec: deepcopy(results[spec.name]),
    )

    assert gate.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "all_passed",
        "benchmarks",
        "checks",
        "cpu_only",
        "errors",
        "schema_version",
    }


def test_main_runs_every_benchmark_after_one_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = _passing_results()
    calls: list[str] = []

    def fake_run(spec: gate.BenchmarkSpec) -> dict[str, object]:
        calls.append(spec.name)
        if spec.name == "radix_eviction":
            raise gate.BenchmarkFailure("synthetic failure")
        return deepcopy(results[spec.name])

    monkeypatch.setattr(gate, "run_benchmark", fake_run)

    assert gate.main([]) == 1
    assert calls == [spec.name for spec in gate.BENCHMARKS]
    report = json.loads(capsys.readouterr().out)
    assert report["all_passed"] is False
    assert report["errors"] == {"radix_eviction": "synthetic failure"}
