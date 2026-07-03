"""ResultStore: atomic pair JSON, resume semantics, filesystem-safe names."""

from kairyu.bench.store import ResultStore
from kairyu.bench.types import ItemResult, PairResult


def _pair(status: str = "completed", target: str = "m") -> PairResult:
    return PairResult(
        benchmark="gpqa-diamond",
        target=target,
        status=status,
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


def test_no_tmp_files_left_behind(tmp_path):
    store = ResultStore(tmp_path, "run-1")
    store.save_pair(_pair())
    store.save_scoreboard({"cells": {}}, "# table")
    store.write_run_config({"a": 1})
    assert not list(store.run_dir.rglob("*.tmp"))
    assert (store.run_dir / "scoreboard.md").read_text(encoding="utf-8") == "# table"
