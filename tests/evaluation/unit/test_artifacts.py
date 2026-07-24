import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

from kairyu.evaluation.artifacts import (
    RUN_ARTIFACT_DIRECTORIES,
    ArtifactConflictError,
    ArtifactStore,
    UnsafeArtifactPath,
)
from kairyu.evaluation.control_store import (
    FinalizationToken,
    LeaseConflictError,
    LeaseToken,
)
from kairyu.evaluation.safety import SecretSafetyError, SecretValueRegistry
from kairyu.evaluation.schemas import Artifact, BenchmarkRun, Metric
from kairyu.evaluation.sqlite_store import SqliteControlStore

_TOKEN = LeaseToken(
    job_id="job-01",
    run_id="run-01",
    worker_id="worker-01",
    attempt=1,
)
_RawArtifactStore = ArtifactStore


@contextmanager
def _publication_guard(publication_token):
    if publication_token != _TOKEN:
        raise LeaseConflictError("artifact lease is not active")
    yield


def _store(root, *, publication_guard=_publication_guard, secret_registry=None):
    store = _RawArtifactStore(
        root,
        publication_guard=publication_guard,
        secret_registry=secret_registry,
    )
    store.write_bytes = partial(store.write_bytes, publication_token=_TOKEN)
    store.write_json = partial(store.write_json, publication_token=_TOKEN)
    store.write_jsonl = partial(store.write_jsonl, publication_token=_TOKEN)
    store.write_text = partial(store.write_text, publication_token=_TOKEN)
    return store


def test_create_run_uses_exact_benchmark_runs_layout(tmp_path):
    root = tmp_path / "benchmark_runs"
    store = _store(root)

    run_dir = store.create_run("run-01")

    assert store.root == root
    assert run_dir == root / "run-01"
    assert {path.name for path in run_dir.iterdir()} == set(RUN_ARTIFACT_DIRECTORIES)


@pytest.mark.parametrize(
    "run_id",
    ("", ".", "..", "../run", "nested/run", "/absolute", r"nested\\run", ".hidden"),
)
def test_run_id_must_be_one_safe_component(tmp_path, run_id):
    store = _store(tmp_path / "benchmark_runs")

    with pytest.raises(UnsafeArtifactPath):
        store.create_run(run_id)


@pytest.mark.parametrize(
    "relative_path",
    (
        "",
        ".",
        "..",
        "../outside",
        "logs/../../outside",
        "/tmp/outside",
        r"logs\\x",
        "./metrics.json",
        "logs//x.json",
        "logs/./x.json",
        "logs/file name.json",
        "logs/日本語.json",
        ".hidden",
    ),
)
def test_artifact_paths_reject_traversal_and_absolute_paths(tmp_path, relative_path):
    store = _store(tmp_path / "benchmark_runs")
    store.create_run("run-01")

    with pytest.raises(UnsafeArtifactPath):
        store.write_bytes("run-01", relative_path, b"secret")

    assert not (tmp_path / "outside").exists()


def test_store_rejects_symlink_as_root_or_root_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(UnsafeArtifactPath):
        _store(link)
    with pytest.raises(UnsafeArtifactPath):
        _store(link / "benchmark_runs")


def test_store_rejects_run_directory_symlink(tmp_path):
    root = tmp_path / "benchmark_runs"
    store = _store(root)
    target = tmp_path / "target"
    target.mkdir()
    (root / "run-01").symlink_to(target, target_is_directory=True)

    with pytest.raises(UnsafeArtifactPath):
        store.create_run("run-01")
    with pytest.raises(UnsafeArtifactPath):
        store.write_bytes("run-01", "manifest.json", b"no")


def test_store_rejects_symlinked_parent_and_destination(tmp_path):
    store = _store(tmp_path / "benchmark_runs")
    run_dir = store.create_run("run-01")
    external = tmp_path / "external"
    external.mkdir()
    (run_dir / "escape").symlink_to(external, target_is_directory=True)

    with pytest.raises(UnsafeArtifactPath):
        store.write_bytes("run-01", "escape/result.json", b"no")

    external_file = tmp_path / "external.json"
    external_file.write_text("unchanged", encoding="utf-8")
    (run_dir / "manifest.json").symlink_to(external_file)
    with pytest.raises(UnsafeArtifactPath):
        store.write_bytes("run-01", "manifest.json", b"changed")
    assert external_file.read_text(encoding="utf-8") == "unchanged"


def test_atomic_byte_write_returns_digest_and_leaves_no_temp_file(tmp_path):
    store = _store(tmp_path / "benchmark_runs")
    run_dir = store.create_run("run-01")

    result = store.write_bytes("run-01", "upstream/result.bin", b"payload")

    assert result.run_id == "run-01"
    assert result.relative_path == "upstream/result.bin"
    assert result.sha256 == ("239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5")
    assert result.size_bytes == 7
    assert store.read_bytes("run-01", "upstream/result.bin") == b"payload"
    assert list(run_dir.rglob("*.tmp")) == []


def test_json_write_is_canonical_create_once_and_idempotent(tmp_path):
    store = _store(tmp_path / "benchmark_runs")
    store.create_run("run-01")

    first = store.write_json("run-01", "manifest.json", {"z": 1, "a": "日本語"})
    retry = store.write_json("run-01", "manifest.json", {"a": "日本語", "z": 1})
    with pytest.raises(ArtifactConflictError, match="different bytes"):
        store.write_json("run-01", "manifest.json", {"version": 2})

    assert retry == first
    content = store.read_bytes("run-01", "manifest.json")
    assert content == '{"a":"日本語","z":1}'.encode()
    assert store.read_json("run-01", "manifest.json") == {
        "a": "日本語",
        "z": 1,
    }


def test_json_write_thaws_immutable_schema_mapping(tmp_path):
    store = _store(tmp_path / "benchmark_runs")
    store.create_run("run-01")
    metric = Metric(
        run_id="run-01",
        name="accuracy",
        display_name="Accuracy",
        dimensions={"groups": ["all"]},
    )

    store.write_json("run-01", "metrics.json", metric.dimensions)

    assert store.read_json("run-01", "metrics.json") == {"groups": ["all"]}


def test_json_write_rejects_nested_secret_before_publishing(tmp_path):
    store = _store(tmp_path / "benchmark_runs")
    store.create_run("run-01")
    secret = "artifact-canary-must-not-appear"

    with pytest.raises(SecretSafetyError) as exc_info:
        store.write_json(
            "run-01",
            "manifest.json",
            {"safe": [{"API-Key": secret}]},
        )

    assert secret not in str(exc_info.value)
    assert not (store.run_dir("run-01") / "manifest.json").exists()


def test_nan_is_not_serialized(tmp_path):
    store = _store(tmp_path / "benchmark_runs")
    store.create_run("run-01")

    with pytest.raises(ValueError):
        store.write_json("run-01", "metrics.json", {"value": float("nan")})

    assert not (store.run_dir("run-01") / "metrics.json").exists()


def test_failed_link_cleans_temporary_file(tmp_path, monkeypatch):
    store = _store(tmp_path / "benchmark_runs")
    run_dir = store.create_run("run-01")

    def fail_link(source, destination, *, follow_symlinks):
        raise OSError("injected link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        store.write_bytes("run-01", "manifest.json", b"payload")

    assert list(run_dir.glob("*.tmp")) == []
    assert not (run_dir / "manifest.json").exists()


def test_read_rejects_artifact_replaced_by_symlink(tmp_path):
    store = _store(tmp_path / "benchmark_runs")
    run_dir = store.create_run("run-01")
    store.write_json("run-01", "manifest.json", {"safe": True})
    (run_dir / "manifest.json").unlink()
    external = tmp_path / "external.json"
    external.write_text(json.dumps({"secret": True}), encoding="utf-8")
    (run_dir / "manifest.json").symlink_to(external)

    with pytest.raises(UnsafeArtifactPath):
        store.read_bytes("run-01", "manifest.json")


def test_read_rejects_directory(tmp_path):
    store = _store(tmp_path / "benchmark_runs")
    store.create_run("run-01")

    with pytest.raises(UnsafeArtifactPath):
        store.read_bytes("run-01", Path("logs"))


def test_all_artifact_writers_reject_registered_secret_before_publish(tmp_path):
    secret = "artifact-provider-value-not-shaped-like-a-key"
    registry = SecretValueRegistry([secret])
    store = _store(
        tmp_path / "benchmark_runs",
        secret_registry=registry,
    )
    store.create_run("run-01")

    attempts = (
        (
            "manifest.json",
            lambda: store.write_json(
                "run-01",
                "manifest.json",
                {"message": f"prefix {secret} suffix"},
            ),
        ),
        (
            "events.jsonl",
            lambda: store.write_jsonl(
                "run-01",
                "events.jsonl",
                ({"message": f"prefix {secret} suffix"},),
            ),
        ),
        (
            "report.md",
            lambda: store.write_text("run-01", "report.md", f"prefix {secret} suffix"),
        ),
        (
            "upstream/raw.bin",
            lambda: store.write_bytes(
                "run-01", "upstream/raw.bin", f"prefix {secret} suffix".encode()
            ),
        ),
    )
    for relative_path, write in attempts:
        with pytest.raises(SecretSafetyError):
            write()
        assert not (store.run_dir("run-01") / relative_path).exists()


def test_write_requires_an_explicit_publication_token(tmp_path):
    store = _RawArtifactStore(
        tmp_path / "benchmark_runs",
        publication_guard=_publication_guard,
    )
    store.create_run("run-01")

    with pytest.raises(TypeError, match="publication_token"):
        store.write_bytes("run-01", "manifest.json", b"payload")


@pytest.mark.parametrize(
    "publication_token",
    (
        LeaseToken(
            job_id="other-job",
            run_id="other-run",
            worker_id="worker-01",
            attempt=1,
        ),
        FinalizationToken(
            run_id="other-run",
            run_version=1,
        ),
    ),
)
def test_write_rejects_publication_token_for_another_run(
    tmp_path,
    publication_token,
):
    store = _RawArtifactStore(
        tmp_path / "benchmark_runs",
        publication_guard=_publication_guard,
    )
    store.create_run("run-01")

    with pytest.raises(LeaseConflictError, match="does not match run"):
        store.write_bytes(
            "run-01",
            "manifest.json",
            b"payload",
            publication_token=publication_token,
        )


def test_stale_worker_cannot_overwrite_current_worker_bytes(tmp_path):
    active_attempt = 2
    guard_held = False

    @contextmanager
    def mutable_guard(publication_token):
        nonlocal guard_held
        if publication_token.run_id != "run-01" or publication_token.attempt != active_attempt:
            raise LeaseConflictError("artifact lease is not active")
        guard_held = True
        try:
            yield
        finally:
            guard_held = False

    current = LeaseToken(
        job_id="job-01",
        run_id="run-01",
        worker_id="worker-current",
        attempt=2,
    )
    stale = LeaseToken(
        job_id="job-01",
        run_id="run-01",
        worker_id="worker-stale",
        attempt=1,
    )
    store = _RawArtifactStore(
        tmp_path / "benchmark_runs",
        publication_guard=mutable_guard,
    )
    store.create_run("run-01")
    store.write_bytes(
        "run-01",
        "manifest.json",
        b"current-worker",
        publication_token=current,
    )

    with pytest.raises(LeaseConflictError):
        store.write_bytes(
            "run-01",
            "manifest.json",
            b"stale-worker",
            publication_token=stale,
        )

    assert not guard_held
    assert store.read_bytes("run-01", "manifest.json") == b"current-worker"


def test_publication_guard_is_held_at_atomic_link_boundary(tmp_path, monkeypatch):
    guard_held = False
    observed = []
    original_link = os.link

    @contextmanager
    def observing_guard(publication_token):
        nonlocal guard_held
        assert publication_token == _TOKEN
        guard_held = True
        try:
            yield
        finally:
            guard_held = False

    def observing_link(source, destination, *, follow_symlinks):
        observed.append(guard_held)
        return original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )

    store = _RawArtifactStore(
        tmp_path / "benchmark_runs",
        publication_guard=observing_guard,
    )
    store.create_run("run-01")
    monkeypatch.setattr(os, "link", observing_link)

    store.write_bytes(
        "run-01",
        "manifest.json",
        b"payload",
        publication_token=_TOKEN,
    )

    assert observed == [True]


def test_registered_secrets_are_rejected_in_artifact_names_and_utf16_bytes(tmp_path):
    secret = "artifact-canary-name"
    registry = SecretValueRegistry([secret])
    store = _RawArtifactStore(
        tmp_path / "benchmark_runs",
        publication_guard=_publication_guard,
        secret_registry=registry,
    )

    with pytest.raises(SecretSafetyError):
        store.create_run(secret)

    store.create_run("run-01")
    with pytest.raises(SecretSafetyError):
        store.write_text(
            "run-01",
            f"logs/{secret}.txt",
            "safe",
            publication_token=_TOKEN,
        )
    with pytest.raises(SecretSafetyError):
        store.write_bytes(
            "run-01",
            "upstream/raw.bin",
            f"prefix {secret} suffix".encode("utf-16-le"),
            publication_token=_TOKEN,
        )

    assert not (store.run_dir("run-01") / "logs" / f"{secret}.txt").exists()
    assert not (store.run_dir("run-01") / "upstream" / "raw.bin").exists()


def test_sqlite_reclaim_fences_actual_artifact_bytes(tmp_path):
    clock_value = [datetime(2026, 7, 24, 12, 0, tzinfo=UTC)]

    def clock():
        return clock_value[0]

    control = SqliteControlStore(
        tmp_path / "control.sqlite3",
        clock=clock,
    )
    control.create_run(
        BenchmarkRun(
            run_id="run-01",
            benchmark_id="gpqa-diamond",
            profile="smoke",
            mode="smoke",
            target_model="fake-model",
            created_at=clock_value[0],
        )
    )
    control.enqueue_job("run-01", job_id="job-01")
    original = control.claim_job("worker-original", lease_seconds=10)
    assert original is not None
    clock_value[0] += timedelta(seconds=10)
    reclaimed = control.claim_job("worker-current", lease_seconds=30)
    assert reclaimed is not None

    artifacts = _RawArtifactStore(
        tmp_path / "benchmark_runs",
        publication_guard=control.publication_guard,
    )
    artifacts.create_run("run-01")
    artifacts.write_bytes(
        "run-01",
        "manifest.json",
        b"current-worker",
        publication_token=reclaimed.lease_token,
    )

    with pytest.raises(LeaseConflictError):
        artifacts.write_bytes(
            "run-01",
            "manifest.json",
            b"stale-worker",
            publication_token=original.lease_token,
        )

    assert artifacts.read_bytes("run-01", "manifest.json") == b"current-worker"


def test_queued_cancel_finalization_token_publishes_source_not_successor(tmp_path):
    now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
    control = SqliteControlStore(
        tmp_path / "control.sqlite3",
        clock=lambda: now,
    )
    control.create_run(
        BenchmarkRun(
            run_id="run-source",
            benchmark_id="gpqa-diamond",
            profile="smoke",
            mode="smoke",
            target_model="fake-model",
            protocol_hash="a" * 64,
            item_input_manifest_sha256="b" * 64,
            created_at=now,
        )
    )
    control.enqueue_job("run-source", job_id="job-source")

    cancelled_job = control.request_cancel("run-source")
    finalization_token = control.finalization_token("run-source")
    successor = control.resume_run("run-source", "run-successor")

    assert cancelled_job.status == "cancelled"
    assert successor.run.run.resumed_from_run_id == "run-source"

    artifacts = _RawArtifactStore(
        tmp_path / "benchmark_runs",
        publication_guard=control.publication_guard,
    )
    artifacts.create_run("run-source")
    artifacts.create_run("run-successor")
    write = artifacts.write_text(
        "run-source",
        "report.md",
        "cancelled report",
        publication_token=finalization_token,
    )
    metadata = Artifact(
        run_id=write.run_id,
        name="report.md",
        relative_path=write.relative_path,
        media_type="text/markdown",
        sha256=write.sha256,
        size_bytes=write.size_bytes,
        created_at=now,
    )

    assert (
        control.register_artifact(
            metadata,
            publication_token=finalization_token,
        )
        == metadata
    )
    assert control.list_artifacts("run-source") == [metadata]
    assert artifacts.read_bytes("run-source", "report.md") == b"cancelled report"

    with pytest.raises(LeaseConflictError, match="does not match run"):
        artifacts.write_text(
            "run-successor",
            "report.md",
            "wrong run",
            publication_token=finalization_token,
        )

    assert not (artifacts.run_dir("run-successor") / "report.md").exists()

    with pytest.raises(LeaseConflictError, match="only report"):
        artifacts.write_jsonl(
            "run-source",
            "predictions.jsonl",
            ({"prediction": "late"},),
            publication_token=finalization_token,
        )
    assert not (artifacts.run_dir("run-source") / "predictions.jsonl").exists()

    predictions = Artifact(
        run_id="run-source",
        name="predictions.jsonl",
        relative_path="predictions.jsonl",
        media_type="application/x-ndjson",
        sha256="c" * 64,
        size_bytes=1,
        created_at=now,
    )
    with pytest.raises(LeaseConflictError, match="only report"):
        control.register_artifact(
            predictions,
            publication_token=finalization_token,
        )
    assert control.list_artifacts("run-source") == [metadata]
