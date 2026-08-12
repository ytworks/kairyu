"""Agentic wrappers (SWE-Bench Pro / Terminal-Bench): skip paths + translation."""

import copy
import json
from pathlib import Path

import httpx
import pytest
from conftest import make_config, make_target

from kairyu.bench.adapters import swebench as swe_mod
from kairyu.bench.adapters import terminal_bench as tb_mod
from kairyu.bench.adapters.base import RunContext
from kairyu.bench.adapters.swebench import (
    find_swebench_report,
    load_prediction_ids,
    parse_swebench_report,
)
from kairyu.bench.adapters.swebench_pro import SweBenchProAdapter
from kairyu.bench.adapters.swebench_verified import SweBenchVerifiedAdapter
from kairyu.bench.adapters.terminal_bench import TerminalBenchAdapter, parse_harbor_results
from kairyu.bench.cache import BenchCache
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore

SWEBENCH_REPORT = {
    "schema_version": 2,
    "total_instances": 5,
    "submitted_instances": 4,
    "completed_instances": 2,
    "resolved_instances": 1,
    "unresolved_instances": 1,
    "empty_patch_instances": 1,
    "error_instances": 1,
    "completed_ids": ["resolved-1", "unresolved-1"],
    "incomplete_ids": ["incomplete-1"],
    "empty_patch_ids": ["empty-1"],
    "submitted_ids": ["resolved-1", "unresolved-1", "empty-1", "error-1"],
    "resolved_ids": ["resolved-1"],
    "unresolved_ids": ["unresolved-1"],
    "error_ids": ["error-1"],
}

SWEBENCH_FLOW_REPORT = {
    "schema_version": 2,
    "total_instances": 4,
    "submitted_instances": 4,
    "completed_instances": 3,
    "resolved_instances": 2,
    "unresolved_instances": 1,
    "empty_patch_instances": 0,
    "error_instances": 1,
    "completed_ids": ["astropy-1", "django-2", "flask-3"],
    "incomplete_ids": [],
    "empty_patch_ids": [],
    "submitted_ids": ["astropy-1", "django-2", "flask-3", "numpy-4"],
    "resolved_ids": ["astropy-1", "django-2"],
    "unresolved_ids": ["flask-3"],
    "error_ids": ["numpy-4"],
}

HARBOR_RESULTS = {
    "results": [
        {"task_id": "tb-001", "resolved": True},
        {"task_id": "tb-002", "resolved": False},
        {"task_id": "tb-003", "reward": 1.0},
        {"task_id": "tb-004"},
    ]
}


def _ctx(tmp_path, docker=(True, "docker available"), **overrides) -> RunContext:
    defaults = dict(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        docker=docker,
    )
    defaults.update(overrides)
    return RunContext(**defaults)


def test_parse_swebench_report_preserves_every_official_outcome():
    selected = (*SWEBENCH_REPORT["submitted_ids"], "incomplete-1")
    items, total = parse_swebench_report(SWEBENCH_REPORT, selected_ids=selected)
    assert total == 5
    by_id = {item.item_id: item for item in items}
    assert by_id["resolved-1"].score == 1.0
    assert by_id["unresolved-1"].score == 0.0
    assert by_id["empty-1"].score == 0.0
    assert by_id["empty-1"].details == {"outcome": "empty_patch"}
    assert by_id["error-1"].status == "failed"
    assert by_id["error-1"].error == "SWE-bench harness error"
    assert by_id["incomplete-1"].status == "failed"
    assert by_id["incomplete-1"].error == "SWE-bench evaluation incomplete"


def _mutate_swebench_report(path: str, value) -> dict:
    report = copy.deepcopy(SWEBENCH_REPORT)
    report[path] = value
    return report


@pytest.mark.parametrize(
    ("report", "message"),
    [
        pytest.param(
            _mutate_swebench_report("schema_version", 1),
            "schema_version",
            id="schema-version",
        ),
        pytest.param(
            _mutate_swebench_report("total_instances", True),
            "total_instances",
            id="boolean-count",
        ),
        pytest.param(
            _mutate_swebench_report("error_instances", -1),
            "error_instances",
            id="negative-count",
        ),
        pytest.param(
            _mutate_swebench_report("resolved_instances", 2),
            "resolved_instances.*resolved_ids",
            id="mismatched-count",
        ),
        pytest.param(
            _mutate_swebench_report("resolved_ids", ["resolved-1", "resolved-1"]),
            "resolved_ids.*duplicate",
            id="duplicate-ids",
        ),
        pytest.param(
            _mutate_swebench_report("unresolved_ids", ["resolved-1"]),
            "resolved_ids.*unresolved_ids.*disjoint",
            id="resolved-unresolved-overlap",
        ),
        pytest.param(
            _mutate_swebench_report("completed_ids", ["resolved-1", "other-1"]),
            "completed_ids",
            id="completed-partition",
        ),
        pytest.param(
            _mutate_swebench_report("empty_patch_ids", ["resolved-1"]),
            "outcome IDs.*disjoint",
            id="terminal-overlap",
        ),
        pytest.param(
            _mutate_swebench_report(
                "submitted_ids", ["resolved-1", "unresolved-1", "empty-1", "other-1"]
            ),
            "submitted_ids",
            id="submitted-partition",
        ),
    ],
)
def test_parse_swebench_report_rejects_malformed_schema(report, message):
    with pytest.raises(ValueError, match=message):
        parse_swebench_report(
            report,
            selected_ids=(*SWEBENCH_REPORT["submitted_ids"], "incomplete-1"),
        )


@pytest.mark.parametrize("selected_ids", [("resolved-1",), ()])
def test_parse_swebench_report_rejects_selected_denominator_mismatch(selected_ids):
    with pytest.raises(ValueError, match="selected IDs"):
        parse_swebench_report(SWEBENCH_REPORT, selected_ids=selected_ids)


def test_swebench_load_prediction_ids_validates_and_sorts_mini_swe_mapping(tmp_path):
    path = tmp_path / "preds.json"
    path.write_text(
        json.dumps(
            {
                "b": {
                    "instance_id": "b",
                    "model_name_or_path": "m",
                    "model_patch": "diff",
                },
                "a": {
                    "instance_id": "a",
                    "model_name_or_path": "m",
                    "model_patch": "",
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_prediction_ids(path) == ("a", "b")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param([], id="not-object"),
        pytest.param({}, id="empty"),
        pytest.param({"": {}}, id="empty-id"),
        pytest.param({"a": []}, id="row-not-object"),
        pytest.param(
            {"a": {"instance_id": "b", "model_name_or_path": "m", "model_patch": ""}},
            id="mismatched-instance-id",
        ),
        pytest.param(
            {"a": {"instance_id": "a", "model_patch": ""}},
            id="missing-model",
        ),
        pytest.param(
            {"a": {"instance_id": "a", "model_name_or_path": "m", "model_patch": 1}},
            id="invalid-patch",
        ),
    ],
)
def test_swebench_load_prediction_ids_rejects_malformed_mapping(tmp_path, payload):
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction"):
        load_prediction_ids(path)


def test_find_swebench_report_requires_exactly_one_schema_v2_report(tmp_path):
    report_path = tmp_path / "model.run.json"
    report_path.write_text(json.dumps(SWEBENCH_REPORT), encoding="utf-8")
    assert find_swebench_report(tmp_path) == SWEBENCH_REPORT

    second = tmp_path / "other.run.json"
    second.write_text(json.dumps(SWEBENCH_REPORT), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple"):
        find_swebench_report(tmp_path)


def test_find_swebench_report_rejects_missing_report(tmp_path):
    (tmp_path / "unrelated.json").write_text('{"resolved_ids": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="no SWE-bench schema-v2 report"):
        find_swebench_report(tmp_path)


def test_parse_harbor_results():
    items = parse_harbor_results(HARBOR_RESULTS)
    by_id = {item.item_id: item for item in items}
    assert by_id["tb-001"].score == 1.0
    assert by_id["tb-002"].score == 0.0
    assert by_id["tb-003"].score == 1.0
    assert by_id["tb-004"].status == "failed"
    # bare-list shape
    assert parse_harbor_results([{"resolved": True}])[0].score == 1.0


async def test_both_skip_without_docker(tmp_path):
    ctx = _ctx(tmp_path, docker=(False, "docker unavailable (binary not found)"))
    for adapter in (
        SweBenchProAdapter(),
        SweBenchVerifiedAdapter(),
        TerminalBenchAdapter(),
    ):
        pair = await adapter.run(make_target(), ctx)
        assert pair.status == "skipped"
        assert "docker unavailable" in pair.reason


async def test_swebench_skips_without_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(swe_mod, "harness_missing", lambda: "mini-swe-agent")
    pair = await SweBenchProAdapter().run(make_target(), _ctx(tmp_path))
    assert pair.status == "skipped"
    assert "bench-agentic" in pair.reason


async def test_terminal_bench_skips_without_harbor(tmp_path, monkeypatch):
    monkeypatch.setattr(tb_mod.shutil, "which", lambda name: None)
    pair = await TerminalBenchAdapter().run(make_target(), _ctx(tmp_path))
    assert pair.status == "skipped"
    assert "harbor not installed" in pair.reason


async def test_swebench_two_stage_flow_and_official_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(swe_mod, "harness_missing", lambda: None)
    stages = []
    selected_ids = tuple(SWEBENCH_FLOW_REPORT["submitted_ids"])

    def fake_run(command, capture_output, timeout, env, cwd, check):
        stages.append(list(command))
        if "run_evaluation" not in " ".join(command):
            output_dir = Path(command[command.index("--output") + 1])
            output_dir.mkdir()
            predictions = {
                instance_id: {
                    "instance_id": instance_id,
                    "model_name_or_path": "openai/kairyu-auto",
                    "model_patch": "diff --git a/a b/a\n",
                }
                for instance_id in reversed(selected_ids)
            }
            (output_dir / "preds.json").write_text(
                json.dumps(predictions), encoding="utf-8"
            )
        else:
            predictions = Path(command[command.index("--predictions_path") + 1])
            assert predictions.is_file()
            assert command[command.index("--instance_ids") + 1 :] == sorted(selected_ids)
            assert command[command.index("--max_workers") + 1] == "3"
            assert command[command.index("--report_dir") + 1] == str(cwd)
            (cwd / "kairyu-bench.report.json").write_text(
                json.dumps(SWEBENCH_FLOW_REPORT), encoding="utf-8"
            )

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(swe_mod.subprocess, "run", fake_run)
    pair = await SweBenchProAdapter().run(
        make_target(model="kairyu-auto"),
        _ctx(tmp_path, limit=4, concurrency=3, run_id="test-run"),
    )
    assert pair.status == "partial"  # the error instance keeps it honest
    assert pair.metrics["score"] == 0.5  # 2 resolved / 4 total (official denominator)
    assert stages[0][:2] == ["mini-extra", "swebench"]
    assert "openai/kairyu-auto" in stages[0]
    assert "--slice" in stages[0]
    output_dir = Path(stages[0][stages[0].index("--output") + 1])
    predictions = Path(stages[1][stages[1].index("--predictions_path") + 1])
    assert output_dir.name == "mini-output"
    assert predictions == output_dir / "preds.json"
    assert "swebench.harness.run_evaluation" in " ".join(stages[1])
    assert pair.methodology["selected_instance_ids"] == sorted(selected_ids)
    assert pair.methodology["official_report"]["error_instances"] == 1
    assert "generate_command" in pair.methodology
    assert "evaluate_command" in pair.methodology
    assert "OPENAI_API_KEY" not in json.dumps(pair.methodology)


async def test_swebench_fails_closed_when_generation_writes_no_predictions(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(swe_mod, "harness_missing", lambda: None)

    def fake_run(command, capture_output, timeout, env, cwd, check):
        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(swe_mod.subprocess, "run", fake_run)
    pair = await SweBenchProAdapter().run(make_target(), _ctx(tmp_path, limit=1))
    assert pair.status == "failed"
    assert "mini-output/preds.json" in pair.reason


async def test_swebench_fails_closed_when_predictions_are_malformed(tmp_path, monkeypatch):
    monkeypatch.setattr(swe_mod, "harness_missing", lambda: None)

    def fake_run(command, capture_output, timeout, env, cwd, check):
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir()
        (output_dir / "preds.json").write_text("{}", encoding="utf-8")

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(swe_mod.subprocess, "run", fake_run)
    pair = await SweBenchVerifiedAdapter().run(make_target(), _ctx(tmp_path, limit=1))
    assert pair.status == "failed"
    assert pair.metrics["score"] is None
    assert "prediction" in pair.reason


@pytest.mark.parametrize(
    ("failing_stage", "returncode", "expected"),
    [
        pytest.param("generate", 2, "generate stage failed", id="generate"),
        pytest.param("evaluate", 3, "evaluate stage failed", id="evaluate"),
    ],
)
async def test_swebench_subprocess_failure_is_stage_specific(
    tmp_path, monkeypatch, failing_stage, returncode, expected
):
    monkeypatch.setattr(swe_mod, "harness_missing", lambda: None)

    def fake_run(command, capture_output, timeout, env, cwd, check):
        is_evaluation = "run_evaluation" in " ".join(command)
        stage = "evaluate" if is_evaluation else "generate"
        if not is_evaluation:
            output_dir = Path(command[command.index("--output") + 1])
            output_dir.mkdir()
            (output_dir / "preds.json").write_text(
                json.dumps(
                    {
                        "instance-1": {
                            "instance_id": "instance-1",
                            "model_name_or_path": "m",
                            "model_patch": "diff",
                        }
                    }
                ),
                encoding="utf-8",
            )

        class Completed:
            stdout = b""
            stderr = b"stage exploded"

            def __init__(self):
                self.returncode = returncode if stage == failing_stage else 0

        return Completed()

    monkeypatch.setattr(swe_mod.subprocess, "run", fake_run)
    pair = await SweBenchVerifiedAdapter().run(make_target(), _ctx(tmp_path, limit=1))
    assert pair.status == "failed"
    assert pair.metrics["score"] is None
    assert expected in pair.reason
    assert "stage exploded" in pair.reason


@pytest.mark.parametrize("failing_stage", ["generate", "evaluate"])
async def test_swebench_timeout_is_stage_specific(
    tmp_path, monkeypatch, failing_stage
):
    monkeypatch.setattr(swe_mod, "harness_missing", lambda: None)

    def fake_run(command, capture_output, timeout, env, cwd, check):
        is_evaluation = "run_evaluation" in " ".join(command)
        stage = "evaluate" if is_evaluation else "generate"
        if stage == failing_stage:
            raise swe_mod.subprocess.TimeoutExpired(command, timeout)
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir()
        (output_dir / "preds.json").write_text(
            json.dumps(
                {
                    "instance-1": {
                        "instance_id": "instance-1",
                        "model_name_or_path": "m",
                        "model_patch": "diff",
                    }
                }
            ),
            encoding="utf-8",
        )

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(swe_mod.subprocess, "run", fake_run)
    pair = await SweBenchVerifiedAdapter().run(make_target(), _ctx(tmp_path, limit=1))
    assert pair.status == "failed"
    assert pair.metrics["score"] is None
    assert f"{failing_stage} stage timed out" in pair.reason


async def test_swebench_fails_closed_on_invalid_official_report(tmp_path, monkeypatch):
    monkeypatch.setattr(swe_mod, "harness_missing", lambda: None)

    def fake_run(command, capture_output, timeout, env, cwd, check):
        if "run_evaluation" not in " ".join(command):
            output_dir = Path(command[command.index("--output") + 1])
            output_dir.mkdir()
            (output_dir / "preds.json").write_text(
                json.dumps(
                    {
                        "instance-1": {
                            "instance_id": "instance-1",
                            "model_name_or_path": "m",
                            "model_patch": "diff",
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            report = copy.deepcopy(SWEBENCH_FLOW_REPORT)
            report["schema_version"] = 1
            (cwd / "invalid.json").write_text(json.dumps(report), encoding="utf-8")

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(swe_mod.subprocess, "run", fake_run)
    pair = await SweBenchVerifiedAdapter().run(make_target(), _ctx(tmp_path, limit=1))
    assert pair.status == "failed"
    assert pair.metrics["score"] is None
    assert "report" in pair.reason


async def test_terminal_bench_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(tb_mod.shutil, "which", lambda name: "/usr/bin/harbor")
    seen = {}

    def fake_run(command, capture_output, timeout, env, check):
        seen["command"] = list(command)
        seen["env"] = env
        jobs_dir = command[command.index("--jobs-dir") + 1]
        seen["jobs_dir"] = jobs_dir
        import pathlib

        path = pathlib.Path(jobs_dir) / "results.json"
        path.write_text(json.dumps(HARBOR_RESULTS), encoding="utf-8")

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(tb_mod.subprocess, "run", fake_run)
    pair = await TerminalBenchAdapter().run(make_target(model="m"), _ctx(tmp_path))
    assert pair.status == "partial"  # tb-004 has no verdict
    assert pair.metrics["n_total"] == 4
    assert seen["command"][:3] == ["harbor", "run", "--config"]
    assert "terminal-bench/terminal-bench-2-1" in seen["command"]
    assert seen["env"]["OPENAI_BASE_URL"] == "http://gw/v1"
    assert Path(seen["jobs_dir"]).is_dir()
    assert pair.methodology["harbor_jobs_dir"] == seen["jobs_dir"]
    assert pair.methodology["agent_effective_timeout_cap_s"] == 7200.0


async def test_harness_failure_reports_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(tb_mod.shutil, "which", lambda name: "/usr/bin/harbor")

    def fake_run(command, capture_output, timeout, env, check):
        class Completed:
            returncode = 3
            stdout = b""
            stderr = b"unknown flag --n-tasks"

        return Completed()

    monkeypatch.setattr(tb_mod.subprocess, "run", fake_run)
    pair = await TerminalBenchAdapter().run(make_target(), _ctx(tmp_path, limit=5))
    assert pair.status == "failed"
    assert "unknown flag" in pair.reason


async def test_full_suite_smoke_has_all_eleven_rows(tmp_path, http_factory):
    """Every Accuracy slot appears and unavailable agentic rows skip cleanly."""
    from kairyu.bench.adapters import ACCURACY_ROW_ORDER

    config = make_config(tmp_path, models=("m", "kairyu-auto"))
    runner = SuiteRunner(
        config,
        http_factory=http_factory,
        probe_docker=lambda: (False, "docker unavailable (test)"),
    )
    assert await runner.run() == 0
    scoreboard = json.loads(
        (tmp_path / "results" / "test-run" / "scoreboard.json").read_text(encoding="utf-8")
    )
    assert scoreboard["benchmarks"] == list(ACCURACY_ROW_ORDER)
    store = ResultStore(tmp_path / "results", "test-run")
    for name in ("swe-bench-pro", "terminal-bench"):
        pair = store.load_pair(name, "m")
        assert pair.status == "skipped"
        assert "docker unavailable" in pair.reason
