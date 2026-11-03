"""Policy tests for the main-only nightly CPU performance series."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _ROOT / ".github" / "workflows" / "nightly-cpu-perf.yml"


def _load_workflow() -> tuple[dict[str, Any], str]:
    text = _WORKFLOW_PATH.read_text(encoding="utf-8")
    parsed = yaml.load(text, Loader=yaml.BaseLoader)
    assert isinstance(parsed, dict)
    return parsed, text


def _only_job(workflow: dict[str, Any]) -> dict[str, Any]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"series"}
    job = jobs["series"]
    assert isinstance(job, dict)
    return job


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        step
        for step in job["steps"]
        if isinstance(step, dict) and step.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_nightly_series_is_serial_main_only_and_least_privilege() -> None:
    workflow, text = _load_workflow()
    triggers = workflow["on"]

    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert len(triggers["schedule"]) == 1
    assert triggers["schedule"][0]["cron"]
    assert "push:" not in text
    assert "pull_request" not in text
    assert workflow["concurrency"] == {
        "group": "nightly-cpu-perf-main",
        "cancel-in-progress": "false",
    }
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}

    job = _only_job(workflow)
    assert job["if"] == "github.ref == 'refs/heads/main'"
    assert job["runs-on"] == "ubuntu-latest"
    assert int(job["timeout-minutes"]) <= 15
    assert "issues: write" not in text
    assert "pull-requests: write" not in text
    assert "contents: write" not in text


def test_nightly_series_uses_locked_runtime_and_exact_cpu_gate() -> None:
    workflow, text = _load_workflow()
    job = _only_job(workflow)
    steps = job["steps"]

    checkout = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("uses")
        == "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
    ]
    assert len(checkout) == 1
    assert checkout[0]["with"]["persist-credentials"] == "false"

    setup = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("uses")
        == "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"
    ]
    assert len(setup) == 1
    assert setup[0]["with"]["python-version"] == "3.12"
    assert _named_step(job, "Sync dependencies")["run"] == (
        "uv sync --frozen --dev"
    )

    measurement = _named_step(job, "Run issue 378 CPU microbenchmarks")
    assert "scripts/cpu_microbench_gate.py" in measurement["run"]
    assert '--output "$REPORT"' in measurement["run"]
    assert text.count("scripts/cpu_microbench_gate.py") == 1
    assert "bench/results" not in text


def test_recovery_includes_failed_runs_and_fails_closed() -> None:
    workflow, text = _load_workflow()
    job = _only_job(workflow)
    recovery = _named_step(job, "Recover newest valid nightly history")
    command = recovery["run"]

    assert recovery["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "ARTIFACT_PREFIX": "nightly-cpu-perf-",
        "CANDIDATE_DIR": (
            "${{ runner.temp }}/kairyu-nightly-cpu-perf/candidates"
        ),
        "DOWNLOAD_DIR": (
            "${{ runner.temp }}/kairyu-nightly-cpu-perf/downloads"
        ),
        "HISTORY": "${{ runner.temp }}/kairyu-nightly-cpu-perf/history.jsonl",
    }
    assert "GH_TOKEN" not in job.get("env", {})
    assert "--paginate" in command
    assert ".created_at" in command
    assert "sort" in command
    assert ".conclusion" not in command
    assert ".status" in command
    assert '"$run_status" != "completed"' in command
    assert ".github/workflows/nightly-cpu-perf.yml" in command
    assert "/attempts/${artifact_attempt}" in command
    assert "workflow_run.head_branch == \"main\"" in command
    assert "GITHUB_RUN_ATTEMPT" not in command
    assert "report.json" in command
    assert "series.jsonl" in command
    assert "comparison.json" in command
    assert "artifact must contain exactly the three root files" in command
    assert "member.file_size >" in command
    assert "uv run --frozen python -" in command
    assert "scripts/cpu_perf_series.py validate" in command
    assert "scripts/cpu_perf_series.py select-history" in command
    assert "series tip does not match Actions provenance" in command
    for identity in (
        "EXPECTED_REPOSITORY",
        "EXPECTED_RUN_ID",
        "EXPECTED_RUN_ATTEMPT",
        "EXPECTED_SHA",
    ):
        assert identity in command
    assert "|| true" not in command
    assert "issue" not in command.lower()


def test_artifacts_are_uploaded_before_the_gate_verdict_is_enforced() -> None:
    workflow, _ = _load_workflow()
    job = _only_job(workflow)
    steps = job["steps"]
    comparison = _named_step(
        job, "Append the nightly record and report the trend"
    )
    warmup = _named_step(
        job, "Report a warm-up run when history is unavailable"
    )
    upload = _named_step(job, "Retain nightly report, series, and comparison")
    enforce = _named_step(job, "Enforce the retained nightly verdict")

    assert comparison["if"] == (
        "always() && steps.recovery.outcome == 'success'"
    )
    assert comparison["continue-on-error"] == "true"
    for argument in (
        "--report",
        "--history",
        "--series",
        "--comparison",
        "--repository",
        "--run-id",
        "--run-attempt",
        "--sha",
    ):
        assert argument in comparison["run"]
    assert warmup["if"] == "always() && steps.recovery.outcome != 'success'"
    assert warmup["continue-on-error"] == "true"
    assert "scripts/cpu_perf_series.py warmup" in warmup["run"]

    assert upload["if"] == "always()"
    assert upload["uses"] == (
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
    )
    assert upload["with"]["name"] == (
        "nightly-cpu-perf-${{ github.run_id }}-${{ github.run_attempt }}"
    )
    assert upload["with"]["retention-days"] == "90"
    assert upload["with"]["if-no-files-found"] == "error"
    paths = upload["with"]["path"]
    assert paths.count("report.json") == 1
    assert paths.count("series.jsonl") == 1
    assert paths.count("comparison.json") == 1

    assert enforce["if"] == "always()"
    assert steps.index(comparison) < steps.index(upload)
    assert steps.index(warmup) < steps.index(upload)
    assert steps.index(upload) < steps.index(enforce)


@pytest.mark.parametrize(
    ("measurement", "recovery", "comparison", "warmup", "passed"),
    [
        ("success", "success", "success", "skipped", True),
        # History could not be fetched: the run is a warm-up, not a failure.
        ("success", "failure", "skipped", "success", True),
        # The deterministic gate failed; the record step reports it too.
        ("failure", "success", "failure", "skipped", False),
        # An invalid report or history cannot be recorded.
        ("success", "success", "failure", "skipped", False),
    ],
)
def test_nightly_verdict_is_the_gate_and_missing_history_is_a_warmup(
    measurement: str,
    recovery: str,
    comparison: str,
    warmup: str,
    passed: bool,
) -> None:
    workflow, _ = _load_workflow()
    enforce = _named_step(_only_job(workflow), "Enforce the retained nightly verdict")
    assert enforce["env"] == {
        "MEASUREMENT_OUTCOME": "${{ steps.measurement.outcome }}",
        "RECOVERY_OUTCOME": "${{ steps.recovery.outcome }}",
        "COMPARISON_OUTCOME": "${{ steps.comparison.outcome }}",
        "WARMUP_OUTCOME": "${{ steps.warmup.outcome }}",
    }

    completed = subprocess.run(
        ["bash", "-c", enforce["run"]],
        env={
            "MEASUREMENT_OUTCOME": measurement,
            "RECOVERY_OUTCOME": recovery,
            "COMPARISON_OUTCOME": comparison,
            "WARMUP_OUTCOME": warmup,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert (completed.returncode == 0) is passed
