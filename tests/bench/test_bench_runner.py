"""SuiteRunner e2e against the in-process mock gateway (offline fixtures)."""

import json

from conftest import make_config

from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore


def _runner(config, http_factory, docker=(False, "docker unavailable (test)")):
    return SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: docker)


async def test_run_produces_scoreboard_for_all_targets(tmp_path, http_factory, capsys):
    config = make_config(tmp_path, models=("m", "kairyu-auto", "kairyu-auto-max"))
    exit_code = await _runner(config, http_factory).run()
    assert exit_code == 0

    run_dir = tmp_path / "results" / "test-run"
    scoreboard = json.loads((run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["targets"] == ["m", "kairyu-auto", "kairyu-auto-max"]
    assert "gpqa-diamond" in scoreboard["benchmarks"]
    for target in scoreboard["targets"]:
        cell = scoreboard["cells"]["gpqa-diamond"][target]
        assert cell["status"] == "completed"  # mock answers score 0 but complete
    assert (run_dir / "scoreboard.md").exists()
    assert (run_dir / "run.json").exists()
    out = capsys.readouterr().out
    assert "| Benchmark |" in out  # table printed to stdout


async def test_pair_results_carry_item_evidence(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",))
    await _runner(config, http_factory).run()
    store = ResultStore(tmp_path / "results", "test-run")
    pair = store.load_pair("gpqa-diamond", "m")
    assert pair is not None
    assert pair.metrics["n_total"] == 3  # fixture size
    assert all(item.status == "completed" for item in pair.items)
    assert all(item.response_excerpt for item in pair.items)
    assert pair.methodology["source"] == "fixtures"


async def test_resume_skips_stored_pairs(tmp_path, http_factory, capsys):
    config = make_config(tmp_path, models=("m",))
    await _runner(config, http_factory).run()
    capsys.readouterr()

    await _runner(config, http_factory).run()
    out = capsys.readouterr().out
    assert "[cached] gpqa-diamond × m" in out

    rerun_config = config.model_copy(update={"rerun": True})
    await _runner(rerun_config, http_factory).run()
    out = capsys.readouterr().out
    assert "[cached]" not in out


async def test_adapter_crash_becomes_failed_pair_and_exit_1(
    tmp_path, http_factory, monkeypatch
):
    from kairyu.bench.adapters.gpqa import GpqaDiamondAdapter

    async def boom(self, target, ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(GpqaDiamondAdapter, "run", boom)
    config = make_config(tmp_path, models=("m",))
    exit_code = await _runner(config, http_factory).run()
    assert exit_code == 1
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert pair.status == "failed"
    assert "kaboom" in pair.reason


async def test_failed_pair_is_retried_on_resume(tmp_path, http_factory, monkeypatch):
    from kairyu.bench.adapters.gpqa import GpqaDiamondAdapter

    async def boom(self, target, ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(GpqaDiamondAdapter, "run", boom)
    config = make_config(tmp_path, models=("m",))
    assert await _runner(config, http_factory).run() == 1

    monkeypatch.undo()
    assert await _runner(config, http_factory).run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert pair.status == "completed"


async def test_dataset_not_downloaded_becomes_skipped(tmp_path, http_factory):
    config = make_config(
        tmp_path, models=("m",), offline_fixtures=False, download=False
    )
    exit_code = await _runner(config, http_factory).run()
    assert exit_code == 0  # skipped is not a failure
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert pair.status == "skipped"
    assert "dataset not in cache" in pair.reason
