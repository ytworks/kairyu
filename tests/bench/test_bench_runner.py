"""SuiteRunner e2e against the in-process mock gateway (offline fixtures)."""

import hashlib
import json
from pathlib import Path

import pytest
from conftest import make_config, make_target

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    RunContext,
    utc_now,
)
from kairyu.bench.cache import BenchCache
from kairyu.bench.runner import (
    SuiteRunner,
    _adapter_identity,
    _run_fingerprint,
    _run_identity,
)
from kairyu.bench.store import ResultStore
from kairyu.bench.types import DownloadReport, ExecutionConfig, PairResult

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


def test_selected_judge_templates_are_content_bound_to_adapter_identity(
    tmp_path, monkeypatch
):
    from kairyu.bench import judge_prompts
    from kairyu.bench.adapters import all_adapters

    cache = BenchCache(tmp_path / "cache")
    adapters = all_adapters()
    hle_identity = _adapter_identity(
        adapters["hle"], cache, offline_fixtures=True
    )
    charxiv_identity = _adapter_identity(
        adapters["charxiv-reasoning"], cache, offline_fixtures=True
    )
    assert hle_identity["judge_template"] == {
        "name": "hle",
        "variant": "production",
        "sha256": "0befab1400eba0c4d4d4e16cc018b4d988d775bbf4ad554fed5fe5f76fede74d",
    }
    assert charxiv_identity["judge_template"] == {
        "name": "charxiv-reasoning",
        "variant": "production",
        "sha256": "e8bf1acd527e025da041a43acd729924abc44610194cbf696f8e228c50b2319d",
    }
    assert "judge_template" not in _adapter_identity(
        adapters["gpqa-diamond"], cache, offline_fixtures=True
    )

    config = make_config(tmp_path, models=("m",), only=("hle",))
    before = _run_fingerprint(_run_identity(config, [hle_identity]))
    monkeypatch.setattr(
        judge_prompts,
        "HLE_JUDGE_TEMPLATE",
        judge_prompts.HLE_JUDGE_TEMPLATE + "\n",
    )
    changed_identity = _adapter_identity(
        adapters["hle"], cache, offline_fixtures=True
    )
    assert changed_identity["judge_template"] != hle_identity["judge_template"]
    assert _run_fingerprint(_run_identity(config, [changed_identity])) != before


def test_core_evaluators_are_content_bound_to_adapter_identity(tmp_path, monkeypatch):
    from kairyu.bench.adapters import all_adapters
    from kairyu.bench.adapters import base as adapter_base

    cache = BenchCache(tmp_path / "cache")
    adapters = all_adapters()
    identities = [
        _adapter_identity(adapters[name], cache, offline_fixtures=True)
        for name in ("gsm8k", "mmlu", "ifeval")
    ]
    resources = {
        entry["resource"]
        for identity in identities
        for entry in identity["evaluation_protocol"]["resources"]
    }
    assert {"gsm8k.py", "mmlu.py", "ifeval.py", "checker.py", "instructions.py"} <= resources
    assert {
        (entry["package"], entry["resource"])
        for entry in identities[1]["evaluation_protocol"]["resources"]
    } == {
        ("kairyu.bench.adapters", "mmlu.py"),
        ("kairyu.bench.adapters", "base.py"),
        ("kairyu.bench", "types.py"),
    }

    config = make_config(tmp_path, models=("m",), suite="core")
    before = _run_fingerprint(_run_identity(config, identities))
    original_read = adapter_base._read_evaluation_resource

    def changed_read(package: str, resource: str) -> bytes:
        content = original_read(package, resource)
        return content + b"\n# changed scorer\n" if resource == "checker.py" else content

    monkeypatch.setattr(adapter_base, "_read_evaluation_resource", changed_read)
    changed_identities = [
        _adapter_identity(adapters[name], cache, offline_fixtures=True)
        for name in ("gsm8k", "mmlu", "ifeval")
    ]
    assert _run_fingerprint(_run_identity(config, changed_identities)) != before


def test_production_judge_template_registry_is_complete():
    from kairyu.bench import judge_prompts

    declared = {
        name: value
        for name, value in vars(judge_prompts).items()
        if name.endswith("_JUDGE_TEMPLATE") and isinstance(value, str)
    }
    registry = judge_prompts.judge_templates()
    assert len(registry) == len(declared)
    assert set(registry.values()) == set(declared.values())

    from kairyu.bench.adapters import all_adapters

    selectors = {
        adapter.info.judge_template_name
        for adapter in all_adapters().values()
        if adapter.info.judge_template_name is not None
    }
    assert set(registry) == selectors


async def test_template_change_refuses_same_run_before_backend_or_artifact_write(
    tmp_path, http_factory, monkeypatch
):
    from kairyu.bench import judge_prompts

    config = make_config(tmp_path, models=("m",), only=("hle",))
    assert await _runner(config, http_factory).run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    run_path = store.run_dir / "run.json"
    pair_path = next(store.run_dir.glob("hle*/*.json"))
    run_before = run_path.read_bytes()
    pair_before = pair_path.read_bytes()
    monkeypatch.setattr(
        judge_prompts,
        "HLE_JUDGE_TEMPLATE",
        judge_prompts.HLE_JUDGE_TEMPLATE + "\nchanged",
    )
    calls = 0

    def tracked_factory():
        nonlocal calls
        calls += 1
        return http_factory()

    with pytest.raises(ValueError, match="run id 'test-run'.*fingerprint"):
        await _runner(config, tracked_factory).run()
    assert calls == 0
    assert run_path.read_bytes() == run_before
    assert pair_path.read_bytes() == pair_before


def test_direct_run_context_gets_a_local_execution_runner(tmp_path):
    ctx = RunContext(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: None,
    )

    assert callable(ctx.execution_runner.run_python)
    assert ctx.execution_runner.metadata()["runner"] == "local"


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

    monkeypatch.setattr("kairyu.bench.execution.build_execution_runner", build)
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
    assert metadata["identity"]["config"]["execution"] == execution.model_dump(
        mode="json"
    )
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
    assert scoreboard["targets"] == ["m", "kairyu-auto", "kairyu-auto-max"]
    assert "gpqa-diamond" in scoreboard["benchmarks"]
    for target in scoreboard["targets"]:
        cell = scoreboard["cells"]["gpqa-diamond"][target]
        assert cell["status"] == "completed"  # mock answers score 0 but complete
    assert (run_dir / "scoreboard.md").exists()
    assert (run_dir / "run.json").exists()
    out = capsys.readouterr().out
    assert "| Benchmark |" in out  # table printed to stdout


async def test_core_offline_runner_completes_all_three_pairs(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), suite="core")
    assert await _runner(config, http_factory).run() == 0

    store = ResultStore(tmp_path / "results", "test-run")
    pairs = {
        benchmark: store.load_pair(benchmark, "m")
        for benchmark in ("gsm8k", "mmlu", "ifeval")
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


async def test_adapter_crash_becomes_failed_pair_and_exit_1(
    tmp_path, http_factory, monkeypatch
):
    from kairyu.bench.adapters.gpqa import GpqaDiamondAdapter

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
    from kairyu.bench.adapters.gpqa import GpqaDiamondAdapter

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
    assert (
        set(config.model_dump(mode="json")) - set(identity["config"])
        == _FINGERPRINT_EXCLUSIONS
    )
    from kairyu.bench.pins import DATASET_PINS

    assert identity["adapters"] == [
        {
            "name": "gpqa-diamond",
            "dataset": "Idavidrein/gpqa",
            # the pinned revision is part of the identity, so repinning refuses
            # resume instead of reinterpreting stored evidence
            "revision": DATASET_PINS["gpqa-diamond"][1],
            # secondary sources are part of the identity even when empty
            "sources": [],
            "unavailable": True,
        }
    ]
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
                config.targets[0].model_copy(
                    update={"base_url": "http://changed.example/v1"}
                ),
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
    adapter_v1 = _CacheBackedAdapter(revision="rev-1")
    monkeypatch.setattr(
        "kairyu.bench.runner.suite_adapters",
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
        "kairyu.bench.runner.suite_adapters",
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
    adapter = _CacheBackedAdapter()
    monkeypatch.setattr(
        "kairyu.bench.runner.suite_adapters",
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


async def test_late_cache_identity_change_becomes_stamped_skip_before_adapter_run(
    tmp_path, http_factory, monkeypatch
):
    adapter = _CacheBackedAdapter()
    monkeypatch.setattr(
        "kairyu.bench.runner.suite_adapters",
        lambda *args, **kwargs: [adapter],
    )
    config = make_config(
        tmp_path,
        models=("m",),
        offline_fixtures=False,
        download=True,
    )
    real_initialize = ResultStore.initialize_run

    def initialize_then_mutate(store, metadata):
        real_initialize(store, metadata)
        cache = BenchCache(Path(config.cache_dir))
        cache.write_rows(
            adapter.info.name,
            [{"id": "1", "answer": "late-mutation"}],
            {
                "dataset": adapter.info.hf_dataset,
                "revision": adapter.info.hf_revision,
            },
        )

    monkeypatch.setattr(ResultStore, "initialize_run", initialize_then_mutate)

    assert await _runner(config, http_factory).run() == 0

    store = ResultStore(tmp_path / "results", "test-run")
    metadata = json.loads((store.run_dir / "run.json").read_text(encoding="utf-8"))
    pair = store.load_pair("pinned-runner", "m")
    assert adapter.run_calls == 0
    assert pair is not None
    assert pair.status == "skipped"
    assert "dataset identity changed" in (pair.reason or "")
    assert pair.run_fingerprint == metadata["fingerprint"]


async def test_legacy_run_metadata_refuses_before_http_and_preserves_bytes(
    tmp_path, http_factory
):
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


async def test_pair_without_fingerprint_is_rerun_and_stamped(
    tmp_path, http_factory
):
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
