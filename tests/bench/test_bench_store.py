"""ResultStore: atomic pair JSON, resume semantics, filesystem-safe names."""

import hashlib
import json
from pathlib import Path

import pytest

from kairyu.bench.store import ResultStore
from kairyu.bench.types import ItemResult, PairResult


def _pair(
    status: str = "completed",
    target: str = "m",
    run_fingerprint: str | None = None,
) -> PairResult:
    return PairResult(
        benchmark="gpqa-diamond",
        target=target,
        status=status,
        run_fingerprint=run_fingerprint,
        metrics={"score": 0.5, "n_total": 2},
        items=(
            ItemResult(item_id="a", status="completed", score=1.0),
            ItemResult(item_id="b", status="completed", score=0.0),
        ),
        started_at="2026-07-03T00:00:00+00:00",
        finished_at="2026-07-03T00:01:00+00:00",
    )


def _record_path_reads(monkeypatch) -> list[Path]:
    reads: list[Path] = []
    real_read_text = Path.read_text

    def recording_read_text(path: Path, *args, **kwargs):
        reads.append(path)
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)
    return reads


def test_pair_round_trip(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    saved = _pair()
    store.save_pair(saved)
    loaded = store.load_pair("gpqa-diamond", "m")
    assert loaded == saved
    assert loaded.score == 0.5


def test_relative_results_directory_preserves_normal_artifact_round_trip(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    store = ResultStore("results", "run-1")
    pair = _pair()

    pair_path = store.save_pair(pair)
    store.write_run_config({"fingerprint": "fingerprint-a"})
    scoreboard_path = store.save_scoreboard({"cells": {}}, "# table")

    assert pair_path == store.pair_path(pair.benchmark, pair.target)
    assert store.load_pair(pair.benchmark, pair.target) == pair
    assert scoreboard_path == store.run_dir / "scoreboard.md"
    assert not list(store.run_dir.rglob("*.tmp"))


def test_missing_pair_is_none(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    assert store.load_pair("gpqa-diamond", "m") is None


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "/absolute/run",
        "nested/run",
        "nested\\run",
        "./run",
        "run/",
        "C:",
        "C:run",
    ],
)
def test_run_id_must_be_one_non_dot_path_component(tmp_path, run_id):
    with pytest.raises(ValueError) as exc_info:
        ResultStore(tmp_path, run_id)

    message = str(exc_info.value)
    assert "run id" in message
    assert repr(run_id) in message
    assert not list(tmp_path.iterdir())


def test_model_names_with_slashes_are_safe(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    pair = _pair(target="Qwen/Qwen2.5-0.5B")
    path = store.save_pair(pair)
    assert path.parent.parent == store.run_dir  # no extra nesting from the slash
    assert store.load_pair("gpqa-diamond", "Qwen/Qwen2.5-0.5B") == pair


def test_sanitized_name_collisions_use_distinct_stable_hash_suffixes(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    slash_name = "org/model"
    underscore_name = "org__model"

    slash_path = store.save_pair(_pair(target=slash_name))
    underscore_path = store.save_pair(_pair(target=underscore_name))

    assert slash_path != underscore_path
    assert slash_path.parent.parent == store.run_dir
    assert underscore_path.parent.parent == store.run_dir
    assert slash_path.stem.startswith("org__model--")
    assert underscore_path.stem.startswith("org__model--")
    assert slash_path.stem.endswith(
        hashlib.sha256(slash_name.encode()).hexdigest()[:16]
    )
    assert underscore_path.stem.endswith(
        hashlib.sha256(underscore_name.encode()).hexdigest()[:16]
    )
    assert store.load_pair("gpqa-diamond", slash_name) == _pair(target=slash_name)
    assert store.load_pair("gpqa-diamond", underscore_name) == _pair(
        target=underscore_name
    )


def test_legacy_pair_without_run_fingerprint_remains_readable():
    legacy = _pair().model_dump(exclude={"run_fingerprint"})

    loaded = PairResult.model_validate(legacy)

    assert loaded.run_fingerprint is None


def test_pair_resume_requires_exact_expected_fingerprint(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    stamped = _pair(run_fingerprint="fingerprint-a")
    legacy = _pair(target="legacy")
    store.save_pair(stamped)
    store.save_pair(legacy)

    assert (
        store.load_pair(
            "gpqa-diamond", "m", expected_fingerprint="fingerprint-a"
        )
        == stamped
    )
    assert (
        store.load_pair(
            "gpqa-diamond", "m", expected_fingerprint="fingerprint-b"
        )
        is None
    )
    assert (
        store.load_pair(
            "gpqa-diamond", "legacy", expected_fingerprint="fingerprint-a"
        )
        is None
    )
    assert store.load_pair("gpqa-diamond", "legacy") == legacy


def _run_metadata(fingerprint: str, created_at: str) -> dict:
    return {
        "run_id": "run-1",
        "fingerprint": fingerprint,
        "identity": {"config": "canonical"},
        "environment": {"created_at": created_at},
    }


@pytest.mark.parametrize(
    "run_id",
    ["20260713T123456Z", "experiment_alpha-01", "release.v1"],
)
def test_normal_timestamp_and_custom_run_ids_remain_valid(tmp_path, run_id):
    store = ResultStore(tmp_path, run_id)
    metadata = _run_metadata("fingerprint-a", "2026-07-13T00:00:00Z")
    metadata["run_id"] = run_id

    store.initialize_run(metadata)

    assert store.run_dir == tmp_path / run_id
    assert json.loads((store.run_dir / "run.json").read_bytes()) == metadata


def test_initialize_run_refuses_resolved_escape_without_outside_io(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "results"
    outside_dir = tmp_path / "outside"
    results_dir.mkdir()
    outside_dir.mkdir()
    metadata = _run_metadata("fingerprint-a", "2026-07-13T00:00:00Z")
    outside_run_config = outside_dir / "run.json"
    outside_run_config.write_text(json.dumps(metadata), encoding="utf-8")
    before = outside_run_config.read_bytes()
    (results_dir / "run-1").symlink_to(outside_dir, target_is_directory=True)
    store = ResultStore(results_dir, "run-1")

    reads: list[Path] = []
    real_read_text = Path.read_text

    def recording_read_text(path: Path, *args, **kwargs):
        reads.append(path)
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    with pytest.raises(ValueError) as exc_info:
        store.initialize_run(metadata)

    message = str(exc_info.value)
    assert "run id 'run-1'" in message
    assert "fingerprint" in message
    assert reads == []
    assert outside_run_config.read_bytes() == before
    assert not (outside_dir / "run.json.tmp").exists()


@pytest.mark.parametrize("corrupt", [b"{not-json", b"\xff"])
def test_initialize_run_refuses_corrupt_metadata_without_overwrite(tmp_path, corrupt):
    store = ResultStore(tmp_path, "run-1")
    store.ensure()
    run_config = store.run_dir / "run.json"
    run_config.write_bytes(corrupt)
    before = run_config.read_bytes()

    with pytest.raises(ValueError) as exc_info:
        store.initialize_run(
            _run_metadata("fingerprint-a", "2026-07-13T00:00:00Z")
        )

    message = str(exc_info.value)
    assert "run id 'run-1'" in message
    assert "fingerprint" in message
    assert run_config.read_bytes() == before


@pytest.mark.parametrize("metadata", [{}, {"fingerprint": ""}])
def test_initialize_run_fingerprint_error_includes_run_id(tmp_path, metadata):
    store = ResultStore(tmp_path, "run-1")

    with pytest.raises(ValueError) as exc_info:
        store.initialize_run(metadata)

    message = str(exc_info.value)
    assert "run id 'run-1'" in message
    assert "fingerprint" in message
    assert not store.run_dir.exists()


def test_initialize_run_cleans_new_directory_after_atomic_write_failure(
    tmp_path, monkeypatch
):
    store = ResultStore(tmp_path, "run-1")

    def fail_after_temporary_write(path: Path, text: str) -> None:
        path.with_suffix(path.suffix + ".tmp").write_text(text, encoding="utf-8")
        raise OSError("simulated atomic write failure")

    monkeypatch.setattr(
        ResultStore,
        "_atomic_write",
        staticmethod(fail_after_temporary_write),
    )

    with pytest.raises(OSError, match="simulated atomic write failure"):
        store.initialize_run(
            _run_metadata("fingerprint-a", "2026-07-13T00:00:00Z")
        )

    assert not store.run_dir.exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_initialize_run_creates_once_and_keeps_first_environment(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    first = _run_metadata("fingerprint-a", "2026-07-13T00:00:00Z")
    same_identity = _run_metadata("fingerprint-a", "2026-07-14T00:00:00Z")

    store.initialize_run(first)
    before = (store.run_dir / "run.json").read_bytes()
    store.initialize_run(same_identity)

    assert json.loads(before) == first
    assert (store.run_dir / "run.json").read_bytes() == before
    assert not list(store.run_dir.rglob("*.tmp"))


@pytest.mark.parametrize(
    "existing",
    [
        pytest.param({"run_id": "run-1"}, id="missing-fingerprint"),
        pytest.param(
            _run_metadata("fingerprint-b", "2026-07-13T00:00:00Z"),
            id="different-fingerprint",
        ),
    ],
)
def test_initialize_run_refuses_nonmatching_metadata_without_overwrite(
    tmp_path, existing
):
    store = ResultStore(tmp_path, "run-1")
    store.write_run_config(existing)
    before = (store.run_dir / "run.json").read_bytes()

    with pytest.raises(ValueError, match="fingerprint"):
        store.initialize_run(
            _run_metadata("fingerprint-a", "2026-07-14T00:00:00Z")
        )

    assert (store.run_dir / "run.json").read_bytes() == before


def test_initialize_run_refuses_preexisting_directory_without_run_metadata(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    store.ensure()

    with pytest.raises(ValueError, match="fingerprint"):
        store.initialize_run(
            _run_metadata("fingerprint-a", "2026-07-13T00:00:00Z")
        )

    assert not (store.run_dir / "run.json").exists()


def test_no_tmp_files_left_behind(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    store.save_pair(_pair())
    store.save_scoreboard({"cells": {}}, "# table")
    store.write_run_config({"a": 1})
    assert not list(store.run_dir.rglob("*.tmp"))
    assert (store.run_dir / "scoreboard.md").read_text(encoding="utf-8") == "# table"


def test_save_pair_refuses_symlinked_benchmark_directory_without_external_write(
    tmp_path,
):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    pair = _pair()
    pair_path = store.pair_path(pair.benchmark, pair.target)
    outside_dir = tmp_path / "outside-pairs"
    outside_dir.mkdir()
    outside_pair = outside_dir / pair_path.name
    outside_pair.write_bytes(b"external-pair-bytes")
    before = outside_pair.read_bytes()
    pair_path.parent.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.save_pair(pair)

    assert outside_pair.read_bytes() == before
    assert pair_path.parent.is_symlink()


def test_load_pair_refuses_symlinked_benchmark_directory_before_external_read(
    tmp_path,
    monkeypatch,
):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    pair = _pair()
    pair_path = store.pair_path(pair.benchmark, pair.target)
    outside_dir = tmp_path / "outside-pairs"
    outside_dir.mkdir()
    outside_pair = outside_dir / pair_path.name
    outside_pair.write_text(pair.model_dump_json(), encoding="utf-8")
    before = outside_pair.read_bytes()
    pair_path.parent.symlink_to(outside_dir, target_is_directory=True)
    reads = _record_path_reads(monkeypatch)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.load_pair(pair.benchmark, pair.target)

    assert reads == []
    assert outside_pair.read_bytes() == before


def test_save_pair_refuses_symlinked_final_path_without_external_write(tmp_path):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    pair = _pair()
    pair_path = store.pair_path(pair.benchmark, pair.target)
    pair_path.parent.mkdir(parents=True)
    outside_pair = tmp_path / "outside-pair.json"
    outside_pair.write_bytes(b"external-pair-bytes")
    before = outside_pair.read_bytes()
    pair_path.symlink_to(outside_pair)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.save_pair(pair)

    assert outside_pair.read_bytes() == before
    assert pair_path.is_symlink()


def test_load_pair_refuses_symlinked_final_path_before_external_read(
    tmp_path,
    monkeypatch,
):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    pair = _pair()
    pair_path = store.pair_path(pair.benchmark, pair.target)
    pair_path.parent.mkdir(parents=True)
    outside_pair = tmp_path / "outside-pair.json"
    outside_pair.write_text(pair.model_dump_json(), encoding="utf-8")
    before = outside_pair.read_bytes()
    pair_path.symlink_to(outside_pair)
    reads = _record_path_reads(monkeypatch)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.load_pair(pair.benchmark, pair.target)

    assert reads == []
    assert outside_pair.read_bytes() == before


def test_save_pair_refuses_preexisting_tmp_symlink_without_external_write(tmp_path):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    pair = _pair()
    pair_path = store.pair_path(pair.benchmark, pair.target)
    pair_path.parent.mkdir(parents=True)
    outside_tmp = tmp_path / "outside-pair-tmp"
    outside_tmp.write_bytes(b"external-tmp-bytes")
    before = outside_tmp.read_bytes()
    legacy_tmp = pair_path.with_suffix(pair_path.suffix + ".tmp")
    legacy_tmp.symlink_to(outside_tmp)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.save_pair(pair)

    assert outside_tmp.read_bytes() == before
    assert legacy_tmp.is_symlink()
    assert not pair_path.exists()


def test_initialize_run_refuses_symlinked_metadata_before_external_read(
    tmp_path,
    monkeypatch,
):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    metadata = _run_metadata("fingerprint-a", "2026-07-13T00:00:00Z")
    outside_metadata = tmp_path / "outside-run.json"
    outside_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    before = outside_metadata.read_bytes()
    (store.run_dir / "run.json").symlink_to(outside_metadata)
    reads = _record_path_reads(monkeypatch)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.initialize_run(metadata)

    assert reads == []
    assert outside_metadata.read_bytes() == before


def test_write_run_config_refuses_symlinked_metadata_without_external_write(tmp_path):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    outside_metadata = tmp_path / "outside-run.json"
    outside_metadata.write_bytes(b"external-metadata-bytes")
    before = outside_metadata.read_bytes()
    run_config = store.run_dir / "run.json"
    run_config.symlink_to(outside_metadata)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.write_run_config({"fingerprint": "fingerprint-a"})

    assert outside_metadata.read_bytes() == before
    assert run_config.is_symlink()


def test_write_run_config_refuses_preexisting_tmp_symlink_without_external_write(
    tmp_path,
):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    outside_tmp = tmp_path / "outside-run-tmp"
    outside_tmp.write_bytes(b"external-tmp-bytes")
    before = outside_tmp.read_bytes()
    legacy_tmp = store.run_dir / "run.json.tmp"
    legacy_tmp.symlink_to(outside_tmp)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.write_run_config({"fingerprint": "fingerprint-a"})

    assert outside_tmp.read_bytes() == before
    assert legacy_tmp.is_symlink()
    assert not (store.run_dir / "run.json").exists()


@pytest.mark.parametrize(
    "artifact_name",
    [
        "scoreboard.json",
        "scoreboard.md",
        "scoreboard.json.tmp",
        "scoreboard.md.tmp",
    ],
)
def test_save_scoreboard_refuses_symlinked_artifact_before_any_write(
    tmp_path,
    artifact_name,
):
    store = ResultStore(tmp_path / "results", "run-1")
    store.ensure()
    outside_artifact = tmp_path / f"outside-{artifact_name.replace('.', '-')}"
    outside_artifact.write_bytes(b"external-scoreboard-bytes")
    before = outside_artifact.read_bytes()
    linked_artifact = store.run_dir / artifact_name
    linked_artifact.symlink_to(outside_artifact)

    with pytest.raises(ValueError, match="run id 'run-1'"):
        store.save_scoreboard({"cells": {}}, "# table")

    assert outside_artifact.read_bytes() == before
    assert linked_artifact.is_symlink()
    for final_name in ("scoreboard.json", "scoreboard.md"):
        final_path = store.run_dir / final_name
        if final_path != linked_artifact:
            assert not final_path.exists()
