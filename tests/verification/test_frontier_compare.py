"""Frontier comparison consumes the shared target and reporting contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "frontier_compare",
    Path(__file__).parents[2]
    / "verification"
    / "product"
    / "performance"
    / "frontier_compare.py",
)
assert _SPEC is not None and _SPEC.loader is not None
frontier_compare = importlib.util.module_from_spec(_SPEC)
sys.modules["frontier_compare"] = frontier_compare
_SPEC.loader.exec_module(frontier_compare)


def test_legacy_target_python_api_remains_available() -> None:
    target = frontier_compare.Target(
        "target",
        "https://api.example.test/v1",
        "model",
    )

    assert target.name == "target"
    assert target.base_url == "https://api.example.test/v1"
    assert target.model == "model"
    assert target.api_key == "sk-local"


def test_summary_uses_nearest_rank_percentiles() -> None:
    report = frontier_compare.TargetReport(name="target", model="model")
    report.trials = [
        frontier_compare.TrialResult(
            ttft_s=float(index),
            tpot_s=float(index) / 10,
            output_chars=index,
            completion_tokens=2,
        )
        for index in range(1, 21)
    ]

    summary = report.summary()

    assert summary["ttft_p50_s"] == 10.0
    assert summary["ttft_p95_s"] == 19.0
    assert summary["tpot_p50_s"] == 1.0


def test_scoreboard_records_percentile_method() -> None:
    scoreboard = frontier_compare.build_scoreboard(
        [frontier_compare.TargetReport(name="target", model="model")]
    )

    assert scoreboard["methodology"]["percentile_method"] == "nearest-rank-v1"


def test_target_cli_fourth_field_is_an_environment_variable_name() -> None:
    args = frontier_compare.build_parser().parse_args(
        ["--target", "remote=https://api.example.test=model=REMOTE_API_KEY"]
    )
    target = frontier_compare.parse_target_spec(args.target[0])

    assert target.base_url == "https://api.example.test/v1"
    assert target.api_key_env == "REMOTE_API_KEY"
    assert "api_key_env" in frontier_compare.build_parser().format_help()
    assert "api_key]" not in frontier_compare.build_parser().format_help()
