"""Agentic wrappers (SWE-Bench Pro / Terminal-Bench): skip paths + translation."""

import json

import httpx
from conftest import make_config, make_target

from kairyu.bench.adapters import swebench_pro as swe_mod
from kairyu.bench.adapters import terminal_bench as tb_mod
from kairyu.bench.adapters.base import RunContext
from kairyu.bench.adapters.swebench_pro import SweBenchProAdapter, parse_swebench_report
from kairyu.bench.adapters.terminal_bench import TerminalBenchAdapter, parse_harbor_results
from kairyu.bench.cache import BenchCache
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore

SWEBENCH_REPORT = {
    "total_instances": 4,
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


def test_parse_swebench_report():
    items, total = parse_swebench_report(SWEBENCH_REPORT)
    assert total == 4
    by_id = {item.item_id: item for item in items}
    assert by_id["astropy-1"].score == 1.0
    assert by_id["flask-3"].score == 0.0
    assert by_id["numpy-4"].status == "failed"


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
    for adapter in (SweBenchProAdapter(), TerminalBenchAdapter()):
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

    def fake_run(command, capture_output, timeout, env, cwd, check):
        stages.append(list(command))
        if "run_evaluation" in " ".join(command):
            (cwd / "kairyu-bench.report.json").write_text(
                json.dumps(SWEBENCH_REPORT), encoding="utf-8"
            )

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(swe_mod.subprocess, "run", fake_run)
    pair = await SweBenchProAdapter().run(
        make_target(model="kairyu-auto"), _ctx(tmp_path, limit=4)
    )
    assert pair.status == "partial"  # the error instance keeps it honest
    assert pair.metrics["score"] == 0.5  # 2 resolved / 4 total (official denominator)
    assert stages[0][:2] == ["mini-extra", "swebench"]
    assert "openai/kairyu-auto" in stages[0]
    assert "--slice" in stages[0]
    assert "swebench.harness.run_evaluation" in " ".join(stages[1])


async def test_terminal_bench_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(tb_mod.shutil, "which", lambda name: "/usr/bin/harbor")
    seen = {}

    def fake_run(command, capture_output, timeout, env, check):
        seen["command"] = list(command)
        seen["env"] = env
        output_dir = command[command.index("--output-dir") + 1]
        import pathlib

        path = pathlib.Path(output_dir) / "results.json"
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
    assert seen["command"][:4] == ["harbor", "run", "-d", "terminal-bench/terminal-bench-2-1"]
    assert seen["env"]["OPENAI_BASE_URL"] == "http://gw/v1"


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
    """The headline guarantee: every Fugu slot appears; agentic rows skip cleanly."""
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
    assert len(scoreboard["benchmarks"]) == 11
    store = ResultStore(tmp_path / "results", "test-run")
    for name in ("swe-bench-pro", "terminal-bench"):
        pair = store.load_pair(name, "m")
        assert pair.status == "skipped"
        assert "docker unavailable" in pair.reason
