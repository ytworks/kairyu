"""The example serving report never relabels profiled diagnostics as evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "qwen3-32b-multi-gpu"
    / "benchmark_report.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen_benchmark_report", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
reporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reporter)


def _payload(*, profile: object = False, ttft_p99_ms: float = 20.0) -> dict:
    return {
        "config": {
            "profile": profile,
            "tensor_parallel": 4,
            "concurrency": 8,
            "max_tokens": 16,
            "ttft_slo_s": 1.0,
        },
        "summary": {
            "dataset": "synthetic",
            "requests": 8,
            "wall_s": 2.0,
            "ttft_p50_ms": 10.0,
            "ttft_p99_ms": ttft_p99_ms,
            "tpot_mean_ms": 3.0,
            "tpot_method": "token",
            "throughput_rps": 4.0,
            "goodput_rps": 3.5,
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_skips_profiled_diagnostic_instead_of_rendering_metrics(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "2026-normal-serving.json", _payload())
    _write(
        tmp_path / "2026-profiled-serving.json",
        _payload(profile=True, ttft_p99_ms=987654.0),
    )

    report = reporter.build_report(tmp_path)

    assert "2026-normal" in report
    assert "2026-profiled |" not in report
    assert "987654" not in report
    assert "profiled local-client diagnostics are not timing-comparable" in report


def test_report_fails_when_only_profiled_diagnostics_exist(tmp_path: Path) -> None:
    _write(tmp_path / "2026-profiled-serving.json", _payload(profile=True))

    with pytest.raises(ValueError, match="no timing-comparable.*profiled"):
        reporter.build_report(tmp_path)


def test_report_rejects_non_boolean_profile_marker(tmp_path: Path) -> None:
    _write(tmp_path / "2026-invalid-serving.json", _payload(profile="false"))

    with pytest.raises(ValueError, match="config.profile must be a boolean"):
        reporter.build_report(tmp_path)


def test_report_accepts_historical_result_without_profile_marker(tmp_path: Path) -> None:
    payload = _payload()
    payload["config"].pop("profile")
    _write(tmp_path / "2026-historical-serving.json", payload)

    report = reporter.build_report(tmp_path)

    assert "2026-historical" in report
