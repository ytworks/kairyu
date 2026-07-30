"""Regression tests for truthful, locally runnable GitHub Actions gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"


def _load_workflow(name: str) -> tuple[dict[str, Any], str]:
    text = (_WORKFLOWS / name).read_text(encoding="utf-8")
    parsed = yaml.load(text, Loader=yaml.BaseLoader)
    assert isinstance(parsed, dict)
    return parsed, text


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    matches = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def _run_commands(workflow: dict[str, Any]) -> tuple[str, ...]:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict)
    commands: list[str] = []
    for job in jobs.values():
        assert isinstance(job, dict)
        for step in job.get("steps", []):
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                commands.append(step["run"])
    return tuple(commands)


def test_f1b_pull_requests_always_run_the_local_smoke_gate() -> None:
    workflow, text = _load_workflow("f1b-rollout.yml")
    pull_request = workflow["on"]["pull_request"]
    rollout = workflow["jobs"]["rollout"]

    assert "types" not in pull_request
    assert "if" not in rollout
    assert "labeled" not in text
    assert "f1b-formal" not in text
    assert "github.event.pull_request.labels" not in text
    assert "github.event_name == 'pull_request' && 'smoke'" in rollout["env"]["PROFILE"]
    step = _named_step(rollout, "Run F1b zero-failure rollout gate")
    assert step["run"] == 'bash scripts/kind_rollout_gate.sh "--${PROFILE}"'


def test_f1c_pull_requests_always_run_the_real_gateway_gate() -> None:
    workflow, text = _load_workflow("f1c-gateway.yml")
    pull_request = workflow["on"]["pull_request"]
    jobs = workflow["jobs"]

    assert "types" not in pull_request
    assert set(jobs) == {"gateway"}
    assert "if" not in jobs["gateway"]
    assert "needs" not in jobs["gateway"]
    assert "run-f1c" not in text
    assert "require-label" not in text
    assert "github.event.pull_request.labels" not in text
    step = _named_step(
        jobs["gateway"],
        "Run the F1c Kubernetes integration test",
    )
    assert step["run"] == "bash scripts/kind_gateway_gate.sh"


def test_f2a_pull_requests_measure_and_replay_fresh_formal_evidence() -> None:
    workflow, text = _load_workflow("f2a-prefix-routing.yml")
    benchmark = workflow["jobs"]["benchmark"]
    profile = benchmark["env"]["PROFILE"]

    assert "github.event_name == 'pull_request' && 'formal'" in profile
    assert "HEAD^ HEAD" not in text
    assert "F2A_RETAINED_DIR" not in text
    assert "steps.mode.outputs" not in text
    assert "F2A_EXPECTED_COMMIT" not in benchmark["env"]

    measurement = _named_step(
        benchmark,
        "Run a fresh 500-replica F2a measurement",
    )
    replay = _named_step(benchmark, "Replay the freshly emitted evidence")
    assert "if" not in measurement
    assert "if" not in replay
    assert "--output-dir \"${F2A_RESULTS_DIR}\"" in measurement["run"]
    assert "--verify-artifact \"${F2A_RESULTS_DIR}\"" in replay["run"]


def test_f2b_pull_requests_measure_and_replay_while_manual_replay_stays_local() -> None:
    workflow, text = _load_workflow("f2b-kv-event-chaos.yml")
    benchmark = workflow["jobs"]["benchmark"]
    profile = benchmark["env"]["PROFILE"]

    assert "github.event_name == 'pull_request' && 'formal'" in profile
    assert "steps.mode.outputs" not in text
    assert "gh api" not in text
    assert "gh run download" not in text
    assert "--actions-run-provenance" not in text
    assert "actions" not in workflow["permissions"]
    assert "F2B_EXPECTED_COMMIT" not in benchmark["env"]
    assert "F2B_GITHUB_HEAD_SHA" not in benchmark["env"]

    measurement = _named_step(benchmark, "Run a fresh F2b measurement")
    fresh_replay = _named_step(
        benchmark,
        "Replay freshly emitted evidence without remeasurement",
    )
    manual_replay = _named_step(
        benchmark,
        "Replay a manually selected local artifact",
    )
    for step in (measurement, fresh_replay):
        assert "github.event_name == 'pull_request'" in step["if"]
        assert "inputs.mode == 'measure'" in step["if"]
    assert manual_replay["if"] == (
        "github.event_name == 'workflow_dispatch' && inputs.mode == 'replay'"
    )
    assert "--verify-artifact \"${F2B_RESULTS_DIR}\"" in fresh_replay["run"]
    assert "--require-current-runner-context" in fresh_replay["run"]
    assert "--verify-artifact \"${REPLAY_DIR}\"" in manual_replay["run"]
    assert "--require-current-source" in manual_replay["run"]


def test_ci_jobs_are_bounded_and_uv_uses_the_committed_lockfile() -> None:
    workflow, text = _load_workflow("ci.yml")

    assert "continue-on-error" not in text
    assert workflow["jobs"]["test"]["name"] == (
        "Portable CPU tests (Python ${{ matrix.python-version }})"
    )
    for name, job in workflow["jobs"].items():
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, str), name
        assert timeout.isdecimal() and int(timeout) > 0, name
    for command in _run_commands(workflow):
        if "uv sync" in command or "uv run" in command:
            assert "--frozen" in command, command
    dependency_check = _named_step(
        workflow["jobs"]["test"],
        "Verify required test dependencies",
    )["run"]
    for module in ("torch", "transformers", "xgrammar", "zmq", "yaml"):
        assert f"--require-module {module}" in dependency_check
    test_command = _named_step(
        workflow["jobs"]["test"],
        "Test with coverage gate",
    )["run"]
    lint_command = _named_step(
        workflow["jobs"]["test"],
        "Lint",
    )["run"]
    assert lint_command == "uv run --frozen ruff check ."
    assert "--fail-on-skip" in test_command
    docker_command = _named_step(
        workflow["jobs"]["bench-exec-container"],
        "Real Docker runner conformance",
    )["run"]
    assert "--fail-on-skip" in docker_command


def test_kind_job_runs_helm_tests_as_an_applicable_non_skipping_suite() -> None:
    workflow, _ = _load_workflow("ci.yml")
    steps = workflow["jobs"]["kind-smoke"]["steps"]
    step = _named_step(
        workflow["jobs"]["kind-smoke"],
        "Run Helm integration tests",
    )
    script = (_ROOT / "scripts" / "helm_integration.sh").read_text(
        encoding="utf-8"
    )

    assert step["run"] == "bash scripts/helm_integration.sh"
    assert any(
        isinstance(candidate, dict)
        and candidate.get("uses") == "astral-sh/setup-uv@v5"
        for candidate in steps
    )
    assert "--require-executable helm" in script
    assert "-m helm tests/unit/test_fleet_elastic.py" in script
    assert "--fail-on-skip" in script


def test_postgres_integration_uses_one_local_reproducible_script() -> None:
    workflow, _ = _load_workflow("ci.yml")
    step = _named_step(
        workflow["jobs"]["compose-smoke"],
        "PostgreSQL integration tests",
    )
    script = (_ROOT / "scripts" / "postgres_integration.sh").read_text(
        encoding="utf-8"
    )

    assert step["run"] == "bash scripts/postgres_integration.sh"
    assert "postgres:17.6-bookworm@sha256:" in script
    assert "docker run --detach --rm" in script
    assert 'psycopg.connect(os.environ["KAIRYU_TEST_POSTGRES_DSN"])' in script
    assert "--fail-on-skip" in script
    assert "-m postgres tests/unit/test_postgres_batch_store.py" in script


def test_f2_workflows_use_frozen_local_uv_commands() -> None:
    for name in ("f2a-prefix-routing.yml", "f2b-kv-event-chaos.yml"):
        workflow, text = _load_workflow(name)
        assert "continue-on-error" not in text
        for command in _run_commands(workflow):
            if "uv sync" in command or "uv run" in command:
                assert "--frozen" in command, (name, command)
        for command in _run_commands(workflow):
            if "pytest" in command:
                assert "--fail-on-skip" in command, (name, command)
