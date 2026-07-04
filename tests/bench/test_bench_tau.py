"""τ-bench wrapper: harness detection, degradation, results translation."""

import json

import httpx
from conftest import make_config, make_target

from kairyu.bench.adapters import tau_bench
from kairyu.bench.adapters.base import RunContext
from kairyu.bench.adapters.tau_bench import (
    TauBenchBankingAdapter,
    parse_tau_results,
)
from kairyu.bench.cache import BenchCache
from kairyu.bench.judge import JudgeClient
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore
from kairyu.bench.types import JudgeConfig

SAMPLE_RESULTS = {
    "simulations": [
        {"task_id": "banking_001", "reward": 1.0},
        {"task_id": "banking_002", "reward": 0.0},
        {"task_id": "banking_003", "reward": 0.5},
    ]
}


def _ctx(tmp_path, judge_base="http://gw/v1", **overrides) -> RunContext:
    judge = None
    if judge_base is not None:
        judge = JudgeClient(
            JudgeConfig(base_url=judge_base, model="judge-m"),
            http_factory=lambda: httpx.AsyncClient(),
        )
    defaults = dict(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        judge=judge,
        offline_fixtures=True,
    )
    defaults.update(overrides)
    return RunContext(**defaults)


def test_parse_tau_results_shapes():
    items = parse_tau_results(SAMPLE_RESULTS)
    assert [item.score for item in items] == [1.0, 0.0, 0.5]
    assert items[0].item_id == "banking_001"

    bare_list = parse_tau_results([{"reward": 0.25}])
    assert bare_list[0].score == 0.25

    missing = parse_tau_results([{"task_id": "x"}])
    assert missing[0].status == "failed"


async def test_skipped_without_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(tau_bench, "detect_harness", lambda: None)
    adapter = TauBenchBankingAdapter()
    pair = await adapter.run(make_target(), _ctx(tmp_path))
    assert pair.status == "skipped"
    assert "bench-agentic" in pair.reason


async def test_skipped_without_user_simulator(tmp_path, monkeypatch):
    monkeypatch.setattr(tau_bench, "detect_harness", lambda: "tau3")
    adapter = TauBenchBankingAdapter()
    pair = await adapter.run(make_target(), _ctx(tmp_path, judge_base=None))
    assert pair.status == "skipped"
    assert "user-simulator" in pair.reason


async def test_skipped_when_judge_on_other_gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(tau_bench, "detect_harness", lambda: "tau3")
    adapter = TauBenchBankingAdapter()
    pair = await adapter.run(
        make_target(), _ctx(tmp_path, judge_base="http://elsewhere/v1")
    )
    assert pair.status == "skipped"
    assert "same base_url" in pair.reason


async def test_harness_invocation_and_translation(tmp_path, monkeypatch):
    monkeypatch.setattr(tau_bench, "detect_harness", lambda: "tau3")
    seen = {}

    def fake_run(command, capture_output, timeout, env, check):
        seen["command"] = command
        seen["env"] = env
        output = command[command.index("--output") + 1]
        import pathlib

        pathlib.Path(output).write_text(json.dumps(SAMPLE_RESULTS), encoding="utf-8")

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(tau_bench.subprocess, "run", fake_run)
    adapter = TauBenchBankingAdapter()
    pair = await adapter.run(make_target(model="kairyu-auto"), _ctx(tmp_path, limit=3))
    assert pair.status == "completed"
    assert pair.score == 0.5  # mean of 1.0/0.0/0.5
    assert pair.metrics["n_total"] == 3
    assert seen["command"][:3] == ["tau3", "run", "--domain"]
    assert "openai/kairyu-auto" in seen["command"]
    assert "openai/judge-m" in seen["command"]
    assert seen["env"]["OPENAI_BASE_URL"] == "http://gw/v1"
    assert pair.methodology["harness"] == "tau3"


async def test_tau2_fallback_adds_substitute_annotation(tmp_path, monkeypatch):
    monkeypatch.setattr(tau_bench, "detect_harness", lambda: "tau2")

    def fake_run(command, capture_output, timeout, env, check):
        output = command[command.index("--output") + 1]
        import pathlib

        pathlib.Path(output).write_text(json.dumps(SAMPLE_RESULTS), encoding="utf-8")

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(tau_bench.subprocess, "run", fake_run)
    adapter = TauBenchBankingAdapter()
    pair = await adapter.run(make_target(), _ctx(tmp_path))
    assert any("tau2 banking substitute" in note for note in pair.annotations)


async def test_harness_failure_is_failed_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(tau_bench, "detect_harness", lambda: "tau3")

    def fake_run(command, capture_output, timeout, env, check):
        class Completed:
            returncode = 2
            stdout = b""
            stderr = b"unknown argument --output"

        return Completed()

    monkeypatch.setattr(tau_bench.subprocess, "run", fake_run)
    adapter = TauBenchBankingAdapter()
    pair = await adapter.run(make_target(), _ctx(tmp_path))
    assert pair.status == "failed"
    assert "rc=2" in pair.reason


async def test_suite_run_records_tau_skip(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("tau-bench-banking",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0  # no harness installed here -> skipped, not failed
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("tau-bench-banking", "m")
    assert pair.status == "skipped"
