"""SuiteRunner e2e against the in-process mock gateway (offline fixtures)."""

import hashlib
import json
from pathlib import Path

import pytest
from conftest import make_config, make_target

from evals import history
from evals.adapters.base import (
    AdapterInfo,
    DownloadContext,
    RunContext,
    utc_now,
)
from evals.cache import BenchCache
from evals.runner import (
    SuiteRunner,
    _adapter_identity,
)
from evals.store import ResultStore
from evals.types import (
    DownloadReport,
    ExecutionConfig,
    ItemResult,
    PairResult,
)

_FINGERPRINT_EXCLUSIONS = {
    "run_id",
    "results_dir",
    "cache_dir",
    "rerun",
    "download",
    "progress",
}


class _CacheBackedAdapter:
    def __init__(
        self,
        *,
        revision: str = "rev-1",
        rows: list[dict] | None = None,
    ) -> None:
        self.info = AdapterInfo(
            name="pinned-runner",
            display_name="Pinned runner",
            metric="accuracy",
            hf_dataset="org/pinned-runner",
            hf_revision=revision,
        )
        self.rows = rows or [{"id": "1", "answer": "original"}]
        self.download_calls = 0
        self.run_calls = 0

    def download(self, ctx: DownloadContext) -> DownloadReport:
        self.download_calls += 1
        ctx.cache.write_rows(
            self.info.name,
            self.rows,
            {
                "dataset": self.info.hf_dataset,
                "revision": self.info.hf_revision,
            },
        )
        return DownloadReport(adapter=self.info.name, status="ok")

    async def run(self, target, ctx) -> PairResult:
        self.run_calls += 1
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1},
            started_at=now,
            finished_at=now,
        )


def _runner(config, http_factory, docker=(False, "docker unavailable (test)")):
    return SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: docker)


def _clean_source(commit: str = "a" * 40) -> dict:
    return {
        "git_commit": commit,
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "git-checkout",
        "source_tree_clean": True,
        "source_tree_sha256": "c" * 64,
    }


def _clean_environment(commit: str = "a" * 40, created_at: str = "t1") -> dict:
    return {
        **_clean_source(commit),
        "kairyu_version": "test",
        "python": "test",
        "created_at": created_at,
        "execution": {
            "runner": "local",
            "available": True,
            "availability_detail": "available",
        },
    }


def _seed_content_bound_cache(config, benchmark: str) -> None:
    from evals.adapters import suite_adapters
    from evals.adapters.base import cache_pins

    adapter = suite_adapters(config.suite, only=(benchmark,))[0]
    BenchCache(Path(config.cache_dir)).write_rows(
        adapter.info.name,
        [{"id": "bound-row"}],
        cache_pins(adapter.info),
    )


def _disable_history_for_in_memory_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        "evals.runner._source_provenance",
        lambda: {
            "git_commit": None,
            "git_commit_role": "local_benchmark_harness",
            "source_kind": "unverified-package",
            "source_tree_clean": False,
            "source_tree_sha256": "0" * 64,
        },
    )


async def test_selected_execution_runner_is_built_once_and_recorded(
    tmp_path, http_factory, monkeypatch
):
    class RecordingRunner:
        def metadata(self):
            return {
                "runner": "docker",
                "image": "sha256:" + "c" * 64,
                "resolved_runtime": "recording-test",
            }

        def availability(self):
            return False, "pinned image unavailable in test"

    selected = RecordingRunner()
    built_with = []

    def build(config):
        built_with.append(config)
        return selected

    monkeypatch.setattr("evals.execution.build_execution_runner", build)
    execution = ExecutionConfig(
        runner="docker",
        image="sha256:" + "c" * 64,
    )
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        execution=execution,
    )

    assert await _runner(config, http_factory).run() == 0

    assert built_with == [execution]
    metadata = json.loads(
        (tmp_path / "results" / "test-run" / "run.json").read_text(encoding="utf-8")
    )
    assert metadata["config"]["execution"] == execution.model_dump(mode="json")
    assert metadata["identity"]["config"]["execution"] == execution.model_dump(mode="json")
    assert metadata["environment"]["execution"] == {
        "runner": "docker",
        "image": "sha256:" + "c" * 64,
        "resolved_runtime": "recording-test",
        "available": False,
        "availability_detail": "pinned image unavailable in test",
    }


async def test_run_produces_scoreboard_for_all_targets(tmp_path, http_factory, capsys):
    config = make_config(tmp_path, models=("m", "kairyu-auto", "kairyu-auto-max"))
    exit_code = await _runner(config, http_factory).run()
    assert exit_code == 0

    run_dir = tmp_path / "results" / "test-run"
    scoreboard = json.loads((run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert scoreboard["fingerprint"] == metadata["fingerprint"]
    assert scoreboard["environment"] == metadata["environment"]
    assert scoreboard["targets"] == ["m", "kairyu-auto", "kairyu-auto-max"]
    assert "gpqa-diamond" in scoreboard["benchmarks"]
    for target in scoreboard["targets"]:
        cell = scoreboard["cells"]["gpqa-diamond"][target]
        assert cell["status"] == "completed"  # mock answers score 0 but complete
    assert (run_dir / "scoreboard.md").exists()
    assert (run_dir / "run.json").exists()
    out = capsys.readouterr().out
    assert "| Benchmark |" in out  # table printed to stdout


async def test_resume_refuses_a_different_source_before_target_io(
    tmp_path, http_factory, monkeypatch
):
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    common = {
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "git-checkout",
        "source_tree_clean": True,
        "source_tree_sha256": "c" * 64,
        "kairyu_version": "test",
        "python": "test",
    }
    environments = iter(
        (
            {**common, "git_commit": "a" * 40, "created_at": "t1"},
            {**common, "git_commit": "b" * 40, "created_at": "t2"},
        )
    )
    monkeypatch.setattr("evals.runner._environment", lambda **_kwargs: next(environments))

    assert await _runner(config, http_factory).run() == 0
    calls = 0

    def tracked_http_factory():
        nonlocal calls
        calls += 1
        return http_factory()

    with pytest.raises(ValueError, match="environment|source"):
        await _runner(config, tracked_http_factory).run()
    assert calls == 0


async def test_core_offline_runner_completes_all_three_pairs(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), suite="core")
    assert await _runner(config, http_factory).run() == 0

    store = ResultStore(tmp_path / "results", "test-run")
    pairs = {
        benchmark: store.load_pair(benchmark, "m") for benchmark in ("gsm8k", "mmlu", "ifeval")
    }
    assert all(pair is not None and pair.status == "completed" for pair in pairs.values())


async def test_quantization_offline_runner_executes_core_plus_gpqa(tmp_path, http_factory):
    from evals.adapters import QUANTIZATION_ROW_ORDER

    config = make_config(tmp_path, models=("m",), suite="quantization")

    assert await _runner(config, http_factory).run() == 0

    store = ResultStore(tmp_path / "results", "test-run")
    pairs = {
        benchmark: store.load_pair(benchmark, "m")
        for benchmark in QUANTIZATION_ROW_ORDER
    }
    assert all(pair is not None and pair.status == "completed" for pair in pairs.values())


async def test_pair_results_carry_item_evidence(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    await _runner(config, http_factory).run()
    store = ResultStore(tmp_path / "results", "test-run")
    pair = store.load_pair("gpqa-diamond", "m")
    assert pair is not None
    assert pair.metrics["n_total"] == 3  # fixture size
    assert all(item.status == "completed" for item in pair.items)
    assert all(item.response_excerpt for item in pair.items)
    assert pair.methodology["source"] == "fixtures"
    metadata = json.loads((store.run_dir / "run.json").read_text(encoding="utf-8"))
    assert pair.run_fingerprint == metadata["fingerprint"]


async def test_resume_skips_stored_pairs(tmp_path, http_factory, capsys):
    """Reuse is announced by the progress reporter (stderr), not on stdout."""
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    await _runner(config, http_factory).run()
    capsys.readouterr()

    await _runner(config, http_factory).run()
    err = capsys.readouterr().err
    assert "gpqa-diamond × m: cached" in err

    rerun_config = config.model_copy(update={"rerun": True})
    await _runner(rerun_config, http_factory).run()
    err = capsys.readouterr().err
    assert "cached" not in err


@pytest.mark.parametrize(
    ("case", "reason_fragment"),
    [
        pytest.param("huge-score", "metrics.score", id="huge-score"),
        pytest.param("huge-custom", "metrics.custom", id="huge-custom-metric"),
        pytest.param("nan-details", "details", id="nested-nan-details"),
        pytest.param("bad-item-score", "strict PairResult", id="invalid-item-score"),
        pytest.param("bad-latency", "strict PairResult", id="invalid-item-latency"),
        pytest.param("bad-pair-schema", "strict PairResult", id="invalid-pair-schema"),
        pytest.param("non-json", "non-JSON value", id="non-json-evidence"),
        pytest.param("wrong-type", "expected PairResult", id="wrong-result-type"),
    ],
)
async def test_fresh_invalid_pair_evidence_is_failed_without_crashing_progress(
    tmp_path,
    http_factory,
    monkeypatch,
    case,
    reason_fragment,
):
    async def invalid_pair(_runner, adapter, target, _ctx):
        if case == "wrong-type":
            return {"status": "completed", "metrics": {"score": 1.0}}
        pair = PairResult(
            benchmark=adapter.info.name,
            target=target.label(),
            status="completed",
            metrics={
                "score": 10**4000 if case == "huge-score" else 1.0,
                "n_total": 1,
                **({"custom": 10**5000} if case == "huge-custom" else {}),
            },
        )
        if case == "non-json":
            return pair.model_copy(update={"methodology": {"bad": object()}})
        if case == "bad-pair-schema":
            return pair.model_copy(update={"status": "bogus"})
        if case in {"nan-details", "bad-item-score", "bad-latency"}:
            item = ItemResult(
                item_id="item-1",
                status="completed",
                score=1.0,
                latency_s=0.1,
                details={"value": float("nan")} if case == "nan-details" else None,
            )
            if case == "bad-item-score":
                item = item.model_copy(update={"score": float("nan")})
            elif case == "bad-latency":
                item = item.model_copy(update={"latency_s": -1.0})
            return pair.model_copy(update={"items": (item,)})
        return pair

    monkeypatch.setattr(SuiteRunner, "_run_pair", invalid_pair)
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))

    assert await _runner(config, http_factory).run() == 1

    store = ResultStore(tmp_path / "results", "test-run")
    pair = store.load_pair("gpqa-diamond", "m")
    assert pair is not None
    assert pair.status == "failed"
    assert pair.score is None
    assert pair.metrics == {"score": None, "n_total": 0}
    assert "adapter returned invalid pair evidence" in (pair.reason or "")
    assert reason_fragment in (pair.reason or "")
    scoreboard = json.loads((store.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["cells"]["gpqa-diamond"]["m"]["status"] == "failed"


async def test_resume_quarantines_huge_stored_score_before_retry(
    tmp_path,
    http_factory,
    monkeypatch,
):
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    existing = store.load_pair("gpqa-diamond", "m")
    assert existing is not None
    store.save_pair(
        existing.model_copy(
            update={"metrics": {"score": 10**4000, "n_total": 1}}
        )
    )

    saved_statuses: list[str] = []
    real_save_pair = ResultStore.save_pair

    def tracked_save_pair(result_store, result):
        saved_statuses.append(result.status)
        return real_save_pair(result_store, result)

    monkeypatch.setattr(ResultStore, "save_pair", tracked_save_pair)

    assert await _runner(config, http_factory).run() == 0

    assert saved_statuses[:2] == ["failed", "completed"]
    recovered = store.load_pair("gpqa-diamond", "m")
    assert recovered is not None
    assert recovered.status == "completed"
    assert recovered.score is not None


async def test_adapter_crash_becomes_failed_pair_and_exit_1(tmp_path, http_factory, monkeypatch):
    from evals.adapters.gpqa import GpqaDiamondAdapter

    async def boom(self, target, ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(GpqaDiamondAdapter, "run", boom)
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    exit_code = await _runner(config, http_factory).run()
    assert exit_code == 1
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert pair.status == "failed"
    assert "kaboom" in pair.reason
    metadata = json.loads(
        (tmp_path / "results" / "test-run" / "run.json").read_text(encoding="utf-8")
    )
    assert pair.run_fingerprint == metadata["fingerprint"]


async def test_failed_pair_is_retried_on_resume(tmp_path, http_factory, monkeypatch):
    from evals.adapters.gpqa import GpqaDiamondAdapter

    async def boom(self, target, ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(GpqaDiamondAdapter, "run", boom)
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    assert await _runner(config, http_factory).run() == 1

    monkeypatch.undo()
    assert await _runner(config, http_factory).run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert pair.status == "completed"


async def test_dataset_not_downloaded_becomes_skipped(tmp_path, http_factory):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )
    exit_code = await _runner(config, http_factory).run()
    assert exit_code == 0  # skipped is not a failure
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert pair.status == "skipped"
    assert "dataset not in cache" in pair.reason
    metadata = json.loads(
        (tmp_path / "results" / "test-run" / "run.json").read_text(encoding="utf-8")
    )
    assert pair.run_fingerprint == metadata["fingerprint"]


async def test_clean_real_data_run_is_indexed_once_and_rerun_is_preflighted(
    tmp_path, http_factory, monkeypatch
):
    environment = _clean_environment()
    monkeypatch.setattr("evals.runner._environment", lambda **_kwargs: dict(environment))
    monkeypatch.setattr("evals.runner._source_provenance", lambda: _clean_source())
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )

    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    index_path = tmp_path / "results" / "scoreboards.jsonl"
    first_bytes = index_path.read_bytes()
    entry = store.find_scoreboard_entry("test-run")
    assert entry is not None
    assert entry["scoreboard"]["fingerprint"] == entry["fingerprint"]
    assert entry["scoreboard"]["environment"] == entry["run"]["environment"]

    # A normal resume is an idempotent append of the same immutable snapshot.
    assert await _runner(config, http_factory).run() == 0
    assert index_path.read_bytes() == first_bytes

    with pytest.raises(ValueError, match="already indexed"):
        await _runner(config.model_copy(update={"rerun": True}), http_factory).run()
    assert index_path.read_bytes() == first_bytes



async def test_tracked_history_hashes_opaque_extra_body_instead_of_retaining_secret(
    tmp_path, http_factory, monkeypatch
):
    environment = _clean_environment()
    monkeypatch.setattr("evals.runner._environment", lambda **_kwargs: dict(environment))
    monkeypatch.setattr("evals.runner._source_provenance", lambda: _clean_source())
    secret = "dummy-body-token-must-not-persist"
    config = make_config(
        tmp_path,
        models=(),
        targets=(
            make_target(
                "m",
                extra_body_json=json.dumps({"vendor_access_token": secret}),
            ),
        ),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )

    assert await _runner(config, http_factory).run() == 0

    run_bytes = (tmp_path / "results" / "test-run" / "run.json").read_bytes()
    index_bytes = (tmp_path / "results" / "scoreboards.jsonl").read_bytes()
    assert secret.encode() not in run_bytes
    assert secret.encode() not in index_bytes
    metadata = json.loads(run_bytes)
    assert metadata["config"]["targets"][0]["extra_body_json"].startswith("sha256:")


async def test_source_drift_permanently_taints_run_and_blocks_clean_resume(
    tmp_path, http_factory, monkeypatch
):
    from evals.adapters.gpqa import GpqaDiamondAdapter

    run_calls = 0

    async def completed_pair(self, target, ctx):
        nonlocal run_calls
        run_calls += 1
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1, "n_scored": 1},
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(GpqaDiamondAdapter, "run", completed_pair)
    environment = _clean_environment()
    monkeypatch.setattr("evals.runner._environment", lambda **_kwargs: dict(environment))
    clean = _clean_source()
    dirty = {**clean, "source_tree_clean": False, "source_tree_sha256": "d" * 64}
    source_observations = 0

    def drift_after_pair_starts():
        nonlocal source_observations
        source_observations += 1
        # 1: initial history/key gate; 2: immediately before the pair.  Every
        # later gate must continue to see the post-pair dirty tree, including
        # render and pre-append checks, rather than depending on iterator size.
        return clean if source_observations <= 2 else dirty

    monkeypatch.setattr("evals.runner._source_provenance", drift_after_pair_starts)
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )
    _seed_content_bound_cache(config, "gpqa-diamond")

    assert await _runner(config, http_factory).run() == 1
    store = ResultStore(tmp_path / "results", "test-run")
    assert (store.run_dir / "source-taint.json").is_file()
    assert not (tmp_path / "results" / "scoreboards.jsonl").exists()
    assert run_calls == 1
    assert source_observations >= 5

    monkeypatch.setattr("evals.runner._source_provenance", lambda: clean)
    with pytest.raises(ValueError, match="permanently source-tainted"):
        await _runner(config, http_factory).run()
    assert run_calls == 1


async def test_post_pair_evaluator_drift_fails_evidence_and_skips_history(
    tmp_path, http_factory, monkeypatch
):
    from evals.adapters.gpqa import GpqaDiamondAdapter

    run_calls = 0

    async def completed_pair(self, target, ctx):
        nonlocal run_calls
        run_calls += 1
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1, "n_scored": 1},
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(GpqaDiamondAdapter, "run", completed_pair)
    environment = _clean_environment()
    monkeypatch.setattr("evals.runner._environment", lambda **_kwargs: dict(environment))
    monkeypatch.setattr("evals.runner._source_provenance", lambda: _clean_source())

    real_adapter_identity = _adapter_identity
    identity_calls = 0

    def drift_after_pair(*args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        identity = real_adapter_identity(*args, **kwargs)
        if identity_calls < 4:
            return identity
        changed = json.loads(json.dumps(identity))
        changed["evaluation_protocol"]["resources"][0]["sha256"] = "f" * 64
        return changed

    monkeypatch.setattr("evals.runner._adapter_identity", drift_after_pair)
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )
    _seed_content_bound_cache(config, "gpqa-diamond")

    assert await _runner(config, http_factory).run() == 1

    store = ResultStore(tmp_path / "results", "test-run")
    pair = store.load_pair("gpqa-diamond", "m")
    assert identity_calls >= 4
    assert run_calls == 1
    assert pair is not None
    assert pair.status == "failed"
    assert "evaluator identity changed" in (pair.reason or "")
    assert not (tmp_path / "results" / "scoreboards.jsonl").exists()


async def test_evaluator_drift_after_render_skips_history_and_returns_nonzero(
    tmp_path, http_factory, monkeypatch
):
    from evals import runner as runner_module
    from evals.adapters.gpqa import GpqaDiamondAdapter

    run_calls = 0

    async def completed_pair(self, target, ctx):
        nonlocal run_calls
        run_calls += 1
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1, "n_scored": 1},
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(GpqaDiamondAdapter, "run", completed_pair)
    environment = _clean_environment()
    monkeypatch.setattr("evals.runner._environment", lambda **_kwargs: dict(environment))
    monkeypatch.setattr("evals.runner._source_provenance", lambda: _clean_source())

    rendered = False
    drift_enabled = True
    real_render = runner_module.render_markdown

    def render_then_drift(scoreboard):
        nonlocal rendered
        markdown = real_render(scoreboard)
        rendered = True
        return markdown

    monkeypatch.setattr("evals.runner.render_markdown", render_then_drift)
    real_adapter_identity = _adapter_identity

    def identity_with_render_drift(*args, **kwargs):
        identity = real_adapter_identity(*args, **kwargs)
        if not drift_enabled or not rendered:
            return identity
        changed = json.loads(json.dumps(identity))
        changed["evaluation_protocol"]["resources"][0]["sha256"] = "f" * 64
        return changed

    monkeypatch.setattr("evals.runner._adapter_identity", identity_with_render_drift)
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )
    _seed_content_bound_cache(config, "gpqa-diamond")

    assert await _runner(config, http_factory).run() == 1

    store = ResultStore(tmp_path / "results", "test-run")
    pair = store.load_pair("gpqa-diamond", "m")
    assert rendered is True
    assert run_calls == 1
    assert pair is not None and pair.status == "failed"
    scoreboard = json.loads((store.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["cells"]["gpqa-diamond"]["m"]["status"] == "failed"
    assert not (tmp_path / "results" / "scoreboards.jsonl").exists()

    drift_enabled = False
    assert await _runner(config, http_factory).run() == 0

    resumed = store.load_pair("gpqa-diamond", "m")
    assert run_calls == 2
    assert resumed is not None and resumed.status == "completed"
    assert (tmp_path / "results" / "scoreboards.jsonl").is_file()


async def test_evaluator_drift_during_scoreboard_save_is_caught_before_history_append(
    tmp_path, http_factory, monkeypatch
):
    from evals.adapters.gpqa import GpqaDiamondAdapter

    run_calls = 0

    async def completed_pair(self, target, ctx):
        nonlocal run_calls
        run_calls += 1
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1, "n_scored": 1},
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(GpqaDiamondAdapter, "run", completed_pair)
    monkeypatch.setattr(
        "evals.runner._environment", lambda **_kwargs: _clean_environment()
    )
    monkeypatch.setattr("evals.runner._source_provenance", lambda: _clean_source())
    saved_scoreboard = False
    drift_enabled = True
    real_save_scoreboard = ResultStore.save_scoreboard

    def save_then_drift(store, scoreboard, markdown):
        nonlocal saved_scoreboard
        path = real_save_scoreboard(store, scoreboard, markdown)
        saved_scoreboard = True
        return path

    monkeypatch.setattr(ResultStore, "save_scoreboard", save_then_drift)
    real_identity = _adapter_identity

    def identity_with_save_drift(*args, **kwargs):
        identity = real_identity(*args, **kwargs)
        if not drift_enabled or not saved_scoreboard:
            return identity
        changed = json.loads(json.dumps(identity))
        changed["evaluation_protocol"]["resources"][0]["sha256"] = "f" * 64
        return changed

    monkeypatch.setattr("evals.runner._adapter_identity", identity_with_save_drift)
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )
    _seed_content_bound_cache(config, "gpqa-diamond")

    assert await _runner(config, http_factory).run() == 1

    store = ResultStore(tmp_path / "results", "test-run")
    pair = store.load_pair("gpqa-diamond", "m")
    assert saved_scoreboard is True
    assert run_calls == 1
    assert pair is not None and pair.status == "failed"
    scoreboard = json.loads((store.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["cells"]["gpqa-diamond"]["m"]["status"] == "failed"
    marker = store.run_dir / "evaluator-taint.json"
    assert marker.is_file()
    assert not (tmp_path / "results" / "scoreboards.jsonl").exists()

    drift_enabled = False
    assert await _runner(config, http_factory).run() == 0

    resumed = store.load_pair("gpqa-diamond", "m")
    assert run_calls == 2
    assert resumed is not None and resumed.status == "completed"
    assert not marker.exists()
    assert (tmp_path / "results" / "scoreboards.jsonl").is_file()


async def test_source_drift_during_scoreboard_save_is_caught_before_history_append(
    tmp_path, http_factory, monkeypatch
):
    from evals.adapters.gpqa import GpqaDiamondAdapter

    run_calls = 0

    async def completed_pair(self, target, ctx):
        nonlocal run_calls
        run_calls += 1
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1, "n_scored": 1},
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(GpqaDiamondAdapter, "run", completed_pair)
    monkeypatch.setattr(
        "evals.runner._environment", lambda **_kwargs: _clean_environment()
    )
    saved_scoreboard = False
    clean = _clean_source()
    dirty = {**clean, "source_tree_clean": False, "source_tree_sha256": "d" * 64}
    monkeypatch.setattr(
        "evals.runner._source_provenance",
        lambda: dirty if saved_scoreboard else clean,
    )
    real_save_scoreboard = ResultStore.save_scoreboard

    def save_then_drift(store, scoreboard, markdown):
        nonlocal saved_scoreboard
        path = real_save_scoreboard(store, scoreboard, markdown)
        saved_scoreboard = True
        return path

    monkeypatch.setattr(ResultStore, "save_scoreboard", save_then_drift)
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )

    assert await _runner(config, http_factory).run() == 1

    store = ResultStore(tmp_path / "results", "test-run")
    assert saved_scoreboard is True
    assert (store.run_dir / "source-taint.json").is_file()
    assert not (tmp_path / "results" / "scoreboards.jsonl").exists()
    pair = store.load_pair("gpqa-diamond", "m")
    assert pair is not None and pair.status == "failed"
    assert "before history append" in (pair.reason or "")
    scoreboard = json.loads((store.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    cell = scoreboard["cells"]["gpqa-diamond"]["m"]
    assert cell["status"] == "failed"
    assert cell["score"] is None

    monkeypatch.setattr("evals.runner._source_provenance", lambda: clean)
    with pytest.raises(ValueError, match="permanently source-tainted"):
        await _runner(config, http_factory).run()
    assert run_calls == 1


async def test_source_drift_during_final_adapter_scan_is_quarantined(
    tmp_path, http_factory, monkeypatch
):
    from evals.adapters import suite_adapters
    from evals.adapters.base import cache_pins
    from evals.adapters.gpqa import GpqaDiamondAdapter

    async def completed_pair(self, target, ctx):
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1, "n_scored": 1},
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(GpqaDiamondAdapter, "run", completed_pair)
    monkeypatch.setattr(
        "evals.runner._environment", lambda **_kwargs: _clean_environment()
    )
    clean = _clean_source()
    dirty = {**clean, "source_tree_clean": False, "source_tree_sha256": "d" * 64}
    source_dirty = False
    saved_scoreboard = False
    monkeypatch.setattr(
        "evals.runner._source_provenance",
        lambda: dirty if source_dirty else clean,
    )
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )
    adapter = suite_adapters(config.suite, only=("gpqa-diamond",))[0]
    cache = BenchCache(Path(config.cache_dir))
    cache.write_rows(
        adapter.info.name,
        [{"id": "bound-row"}],
        cache_pins(adapter.info),
    )
    real_adapter_identity = _adapter_identity
    real_save_scoreboard = ResultStore.save_scoreboard
    identity_calls = 0

    def save_then_enable_drift(store, scoreboard, markdown):
        nonlocal saved_scoreboard
        path = real_save_scoreboard(store, scoreboard, markdown)
        saved_scoreboard = True
        return path

    def drift_during_final_scan(*args, **kwargs):
        nonlocal identity_calls, source_dirty
        identity_calls += 1
        identity = real_adapter_identity(*args, **kwargs)
        # The pre-append source observation has already happened after the
        # scoreboard save.  Flip only while the following identity scan is in
        # progress, after returning the unchanged evaluator identity.
        if saved_scoreboard:
            source_dirty = True
        return identity

    monkeypatch.setattr(ResultStore, "save_scoreboard", save_then_enable_drift)
    monkeypatch.setattr("evals.runner._adapter_identity", drift_during_final_scan)

    assert await _runner(config, http_factory).run() == 1

    store = ResultStore(tmp_path / "results", "test-run")
    pair = store.load_pair("gpqa-diamond", "m")
    assert identity_calls >= 6
    assert source_dirty is True
    assert pair is not None and pair.status == "failed"
    assert "before history append" in (pair.reason or "")
    assert (store.run_dir / "source-taint.json").is_file()
    scoreboard = json.loads((store.run_dir / "scoreboard.json").read_text(encoding="utf-8"))
    assert scoreboard["cells"]["gpqa-diamond"]["m"]["status"] == "failed"
    assert not (tmp_path / "results" / "scoreboards.jsonl").exists()


async def test_evaluator_taint_forces_reexecution_after_crash_before_failed_pair_save(
    tmp_path, http_factory, monkeypatch
):
    from evals.adapters.gpqa import GpqaDiamondAdapter

    run_calls = 0

    async def completed_pair(self, target, ctx):
        nonlocal run_calls
        run_calls += 1
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1, "n_scored": 1},
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(GpqaDiamondAdapter, "run", completed_pair)
    monkeypatch.setattr(
        "evals.runner._environment", lambda **_kwargs: _clean_environment()
    )
    monkeypatch.setattr("evals.runner._source_provenance", lambda: _clean_source())
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )
    _seed_content_bound_cache(config, "gpqa-diamond")
    real_save_scoreboard = ResultStore.save_scoreboard

    def crash_after_pair(*_args, **_kwargs):
        raise RuntimeError("crash after completed pair save")

    monkeypatch.setattr(ResultStore, "save_scoreboard", crash_after_pair)
    with pytest.raises(RuntimeError, match="after completed pair"):
        await _runner(config, http_factory).run()

    store = ResultStore(tmp_path / "results", "test-run")
    old_pair = store.load_pair("gpqa-diamond", "m")
    assert old_pair is not None and old_pair.status == "completed"
    assert run_calls == 1
    assert not (tmp_path / "results" / "scoreboards.jsonl").exists()

    monkeypatch.setattr(ResultStore, "save_scoreboard", real_save_scoreboard)
    real_identity = _adapter_identity
    identity_calls = 0
    drift_enabled = True

    def drift_after_resume_initialization(*args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        identity = real_identity(*args, **kwargs)
        if not drift_enabled or identity_calls < 2:
            return identity
        changed = json.loads(json.dumps(identity))
        changed["evaluation_protocol"]["resources"][0]["sha256"] = "f" * 64
        return changed

    monkeypatch.setattr(
        "evals.runner._adapter_identity", drift_after_resume_initialization
    )
    real_save_pair = ResultStore.save_pair

    def crash_before_failed_pair_save(_store, result):
        assert result.status == "failed"
        raise RuntimeError("crash before failed pair save")

    monkeypatch.setattr(ResultStore, "save_pair", crash_before_failed_pair_save)
    with pytest.raises(RuntimeError, match="before failed pair"):
        await _runner(config, http_factory).run()

    marker = store.run_dir / "evaluator-taint.json"
    assert marker.is_file()
    still_old = store.load_pair("gpqa-diamond", "m")
    assert still_old is not None and still_old.status == "completed"
    assert run_calls == 1

    drift_enabled = False
    monkeypatch.setattr(ResultStore, "save_pair", real_save_pair)
    assert await _runner(config, http_factory).run() == 0

    recovered = store.load_pair("gpqa-diamond", "m")
    assert recovered is not None and recovered.status == "completed"
    assert run_calls == 2
    assert not marker.exists()
    assert (tmp_path / "results" / "scoreboards.jsonl").is_file()



def test_download_missing_checks_adapter_dataset_and_revision_pins(tmp_path):
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(
        "pinned-runner",
        [{"id": "old"}],
        {"dataset": "org/pinned-runner", "revision": "rev-1"},
    )
    adapter = _CacheBackedAdapter(revision="rev-2")
    config = make_config(
        tmp_path,
        models=("m",),
        offline_fixtures=False,
        download=True,
    )
    runner = _runner(config, lambda: None)
    ctx = RunContext(cache=cache, http_factory=lambda: None)

    runner._download_missing([adapter], cache, ctx)

    assert adapter.download_calls == 1
    assert cache.is_ready(
        adapter.info.name,
        adapter.info.hf_dataset,
        adapter.info.hf_revision,
    )


async def test_run_initializes_canonical_identity_and_stamps_pairs(
    tmp_path, http_factory, monkeypatch
):
    monkeypatch.setenv("RUNNER_TEST_API_KEY", "resolved-secret-must-not-be-hashed")
    config = make_config(
        tmp_path,
        models=(),
        targets=(
            make_target(
                "m",
                name="target-label",
                api_key_env="RUNNER_TEST_API_KEY",
                max_output_tokens=321,
            ),
        ),
        only=("gpqa-diamond",),
        limit=2,
        seed=7,
        retries=4,
    )

    assert await _runner(config, http_factory).run() == 0

    store = ResultStore(tmp_path / "results", "test-run")
    metadata_bytes = (store.run_dir / "run.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    identity = metadata["identity"]
    expected_config = {
        key: value
        for key, value in config.model_dump(mode="json").items()
        if key not in _FINGERPRINT_EXCLUSIONS
    }
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert identity["config"] == expected_config
    assert set(config.model_dump(mode="json")) - set(identity["config"]) == _FINGERPRINT_EXCLUSIONS
    from evals.pins import DATASET_PINS

    adapter_identity = identity["adapters"][0]
    assert adapter_identity["name"] == "gpqa-diamond"
    assert adapter_identity["dataset"] == "Idavidrein/gpqa"
    # the pinned revision is part of the identity, so repinning refuses resume
    # instead of reinterpreting stored evidence
    assert adapter_identity["revision"] == DATASET_PINS["gpqa-diamond"][1]
    assert adapter_identity["sources"] == []
    assert adapter_identity["unavailable"] is True
    bound_resources = {
        (entry["package"], entry["resource"])
        for entry in adapter_identity["evaluation_protocol"]["resources"]
    }
    assert ("evals.adapters", "gpqa.py") in bound_resources
    assert ("evals", "aggregate.py") in bound_resources
    assert metadata["config"] == config.model_dump(mode="json")
    assert metadata["run_id"] == "test-run"
    assert metadata["fingerprint"] == hashlib.sha256(canonical).hexdigest()
    assert b"resolved-secret-must-not-be-hashed" not in metadata_bytes
    pair = store.load_pair(
        "gpqa-diamond",
        "target-label",
        expected_fingerprint=metadata["fingerprint"],
    )
    assert pair is not None
    assert pair.run_fingerprint == metadata["fingerprint"]


@pytest.mark.parametrize("rerun", [False, True])
async def test_changed_target_identity_refuses_before_http_or_artifact_writes(
    tmp_path, http_factory, rerun
):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
    )
    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    run_path = store.run_dir / "run.json"
    pair_path = store.pair_path("gpqa-diamond", "m")
    run_before = run_path.read_bytes()
    pair_before = pair_path.read_bytes()
    http_calls = 0

    def tracked_http_factory():
        nonlocal http_calls
        http_calls += 1
        return http_factory()

    changed = config.model_copy(
        update={
            "targets": (
                config.targets[0].model_copy(update={"base_url": "http://changed.example/v1"}),
            ),
            "rerun": rerun,
        }
    )

    with pytest.raises(ValueError, match="run id 'test-run'.*fingerprint"):
        await _runner(changed, tracked_http_factory).run()

    assert http_calls == 0
    assert run_path.read_bytes() == run_before
    assert pair_path.read_bytes() == pair_before


async def test_dataset_revision_change_refuses_without_running_or_rewriting(
    tmp_path, http_factory, monkeypatch
):
    _disable_history_for_in_memory_adapter(monkeypatch)
    adapter_v1 = _CacheBackedAdapter(revision="rev-1")
    monkeypatch.setattr(
        "evals.runner.suite_adapters",
        lambda *args, **kwargs: [adapter_v1],
    )
    config = make_config(
        tmp_path,
        models=("m",),
        offline_fixtures=False,
        download=True,
    )
    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    run_path = store.run_dir / "run.json"
    pair_path = store.pair_path("pinned-runner", "m")
    run_before = run_path.read_bytes()
    pair_before = pair_path.read_bytes()

    adapter_v2 = _CacheBackedAdapter(revision="rev-2")
    monkeypatch.setattr(
        "evals.runner.suite_adapters",
        lambda *args, **kwargs: [adapter_v2],
    )

    with pytest.raises(ValueError, match="run id 'test-run'.*fingerprint"):
        await _runner(config, http_factory).run()

    assert adapter_v2.download_calls == 1
    assert adapter_v2.run_calls == 0
    assert run_path.read_bytes() == run_before
    assert pair_path.read_bytes() == pair_before


async def test_dataset_digest_change_refuses_without_running_or_rewriting(
    tmp_path, http_factory, monkeypatch
):
    _disable_history_for_in_memory_adapter(monkeypatch)
    adapter = _CacheBackedAdapter()
    monkeypatch.setattr(
        "evals.runner.suite_adapters",
        lambda *args, **kwargs: [adapter],
    )
    config = make_config(
        tmp_path,
        models=("m",),
        offline_fixtures=False,
        download=True,
    )
    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    run_path = store.run_dir / "run.json"
    pair_path = store.pair_path("pinned-runner", "m")
    run_before = run_path.read_bytes()
    pair_before = pair_path.read_bytes()
    cache = BenchCache(Path(config.cache_dir))
    cache.write_rows(
        adapter.info.name,
        [{"id": "1", "answer": "mutated"}],
        {
            "dataset": adapter.info.hf_dataset,
            "revision": adapter.info.hf_revision,
        },
    )
    runs_before = adapter.run_calls

    with pytest.raises(ValueError, match="run id 'test-run'.*fingerprint"):
        await _runner(config, http_factory).run()

    assert adapter.run_calls == runs_before
    assert run_path.read_bytes() == run_before
    assert pair_path.read_bytes() == pair_before


async def test_late_cache_identity_change_is_failed_then_clean_resume_reexecutes(
    tmp_path, http_factory, monkeypatch
):
    from evals.adapters import suite_adapters
    from evals.adapters.base import cache_pins
    from evals.adapters.gpqa import GpqaDiamondAdapter

    run_calls = 0

    async def completed_pair(self, target, ctx):
        nonlocal run_calls
        run_calls += 1
        now = utc_now()
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="completed",
            metrics={"score": 1.0, "n_total": 1, "n_scored": 1},
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr(GpqaDiamondAdapter, "run", completed_pair)
    monkeypatch.setattr(
        "evals.runner._environment", lambda **_kwargs: _clean_environment()
    )
    monkeypatch.setattr("evals.runner._source_provenance", lambda: _clean_source())
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
        offline_fixtures=False,
        download=False,
    )
    adapter = suite_adapters(config.suite, only=("gpqa-diamond",))[0]
    cache = BenchCache(Path(config.cache_dir))
    original_rows = [{"id": "1", "answer": "original"}]
    cache.write_rows(adapter.info.name, original_rows, cache_pins(adapter.info))
    real_initialize = ResultStore.initialize_run

    def initialize_then_mutate(store, metadata):
        persisted = real_initialize(store, metadata)
        cache.write_rows(
            adapter.info.name,
            [{"id": "1", "answer": "late-mutation"}],
            cache_pins(adapter.info),
        )
        assert _adapter_identity(adapter, cache, offline_fixtures=False) != metadata[
            "identity"
        ]["adapters"][0]
        return persisted

    monkeypatch.setattr(ResultStore, "initialize_run", initialize_then_mutate)

    assert await _runner(config, http_factory).run() == 1

    store = ResultStore(tmp_path / "results", "test-run")
    metadata = json.loads((store.run_dir / "run.json").read_text(encoding="utf-8"))
    pair = store.load_pair("gpqa-diamond", "m")
    assert run_calls == 0
    assert pair is not None
    assert pair.status == "failed"
    assert "dataset identity changed" in (pair.reason or "")
    assert pair.run_fingerprint == metadata["fingerprint"]
    assert not (tmp_path / "results" / "scoreboards.jsonl").exists()

    cache.write_rows(adapter.info.name, original_rows, cache_pins(adapter.info))
    monkeypatch.setattr(ResultStore, "initialize_run", real_initialize)

    assert await _runner(config, http_factory).run() == 0

    resumed = store.load_pair("gpqa-diamond", "m")
    assert run_calls == 1
    assert resumed is not None and resumed.status == "completed"
    assert (tmp_path / "results" / "scoreboards.jsonl").is_file()


async def test_legacy_run_metadata_refuses_before_http_and_preserves_bytes(tmp_path, http_factory):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
    )
    store = ResultStore(tmp_path / "results", "test-run")
    store.write_run_config({"config": config.model_dump(), "run_id": "test-run"})
    legacy_pair = PairResult(
        benchmark="gpqa-diamond",
        target="m",
        status="completed",
        metrics={"score": 0.25, "n_total": 1},
    )
    store.save_pair(legacy_pair)
    run_path = store.run_dir / "run.json"
    pair_path = store.pair_path("gpqa-diamond", "m")
    run_before = run_path.read_bytes()
    pair_before = pair_path.read_bytes()
    http_calls = 0

    def tracked_http_factory():
        nonlocal http_calls
        http_calls += 1
        return http_factory()

    with pytest.raises(ValueError, match="run id 'test-run'.*fingerprint"):
        await _runner(config, tracked_http_factory).run()

    assert http_calls == 0
    assert run_path.read_bytes() == run_before
    assert pair_path.read_bytes() == pair_before


async def test_pair_without_fingerprint_is_rerun_and_stamped(tmp_path, http_factory):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
    )
    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    metadata = json.loads((store.run_dir / "run.json").read_text(encoding="utf-8"))
    existing = store.load_pair("gpqa-diamond", "m")
    store.save_pair(existing.model_copy(update={"run_fingerprint": None}))
    http_calls = 0

    def tracked_http_factory():
        nonlocal http_calls
        http_calls += 1
        return http_factory()

    assert await _runner(config, tracked_http_factory).run() == 0

    assert http_calls == 1
    pair = store.load_pair(
        "gpqa-diamond",
        "m",
        expected_fingerprint=metadata["fingerprint"],
    )
    assert pair is not None
    assert pair.run_fingerprint == metadata["fingerprint"]


async def test_pair_without_matching_source_identity_is_rerun(tmp_path, http_factory):
    config = make_config(
        tmp_path,
        models=("m",),
        only=("gpqa-diamond",),
    )
    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    metadata = json.loads((store.run_dir / "run.json").read_text(encoding="utf-8"))
    existing = store.load_pair("gpqa-diamond", "m")
    assert existing is not None
    store.save_pair(existing.model_copy(update={"source_identity": None}))
    http_calls = 0

    def tracked_http_factory():
        nonlocal http_calls
        http_calls += 1
        return http_factory()

    assert await _runner(config, tracked_http_factory).run() == 0

    assert http_calls == 1
    pair = store.load_pair(
        "gpqa-diamond",
        "m",
        expected_fingerprint=metadata["fingerprint"],
        expected_source_identity=history.source_identity(metadata["environment"]),
    )
    assert pair is not None


async def test_resolved_api_key_value_does_not_change_run_fingerprint(
    tmp_path, http_factory, monkeypatch, capsys
):
    target = make_target("m", api_key_env="RUNNER_TEST_API_KEY")
    config = make_config(
        tmp_path,
        models=(),
        targets=(target,),
        only=("gpqa-diamond",),
    )
    monkeypatch.setenv("RUNNER_TEST_API_KEY", "first-secret")
    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    run_before = (store.run_dir / "run.json").read_bytes()
    capsys.readouterr()

    monkeypatch.setenv("RUNNER_TEST_API_KEY", "second-secret")
    assert await _runner(config, http_factory).run() == 0

    assert "gpqa-diamond × m: cached" in capsys.readouterr().err
    assert (store.run_dir / "run.json").read_bytes() == run_before
    assert b"first-secret" not in run_before
    assert b"second-secret" not in run_before
