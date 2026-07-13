"""ResultStore: atomic pair JSON, resume semantics, filesystem-safe names."""

import hashlib
import json

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


def test_pair_round_trip(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    saved = _pair()
    store.save_pair(saved)
    loaded = store.load_pair("gpqa-diamond", "m")
    assert loaded == saved
    assert loaded.score == 0.5


def test_missing_pair_is_none(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    assert store.load_pair("gpqa-diamond", "m") is None


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
