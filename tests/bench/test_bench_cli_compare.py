"""Public ``kairyu bench compare-runs`` dispatch and failure behavior."""

from kairyu.bench.cli import handle
from kairyu.bench.store import ResultStore
from kairyu.entrypoints import cli


def _entry(run_id: str, commit: str, score: float) -> dict:
    fingerprint = "f" * 64
    environment = {
        "git_commit": commit,
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "git-checkout",
        "source_tree_clean": True,
        "source_tree_sha256": "d" * 64,
        "created_at": f"created-{run_id}",
        "kairyu_version": "0.1.0",
        "python": "3.12.3",
        "execution": {
            "runner": "local",
            "available": True,
            "availability_detail": "available",
        },
    }
    board = {
        "schema_version": 1,
        "run_id": run_id,
        "suite": "core",
        "fingerprint": fingerprint,
        "environment": environment,
        "config": {"suite": "core"},
        "benchmarks": ["gsm8k"],
        "display_names": {"gsm8k": "GSM8K"},
        "targets": ["model"],
        "cells": {
            "gsm8k": {
                "model": {
                    "status": "completed",
                    "score": score,
                    "n": 100,
                    "n_scored": 100,
                    "reason": None,
                    "comparable": True,
                    "incomparable_reasons": [],
                    "cross_run_policy": "allowed",
                    "cross_run_reason": None,
                }
            }
        },
        "footnotes": [],
    }
    return {
        "run_id": run_id,
        "suite": "core",
        "fingerprint": fingerprint,
        "git_commit": commit,
        "run": {
            "run_id": run_id,
            "fingerprint": fingerprint,
            "config": {"suite": "core"},
            "environment": environment,
        },
        "scoreboard": board,
    }


def test_compare_runs_public_parser_dispatches_without_writing(tmp_path, monkeypatch, capsys):
    entries = (
        _entry("base", "a" * 40, 0.8),
        _entry("candidate", "b" * 40, 0.7),
    )
    monkeypatch.setattr(ResultStore, "load_scoreboard_index", lambda _self: entries)
    args = cli._build_parser().parse_args(
        [
            "bench",
            "compare-runs",
            "base",
            "candidate",
            "--suite",
            "core",
            "--results-dir",
            str(tmp_path),
        ]
    )

    assert handle(args) == 0

    output = capsys.readouterr().out
    assert "Accuracy run comparison — base → candidate" in output
    assert "-10.0" in output
    assert "local-harness-only" in output
    assert list(tmp_path.iterdir()) == []


def test_compare_runs_missing_indexed_run_is_a_clear_nonzero_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ResultStore, "load_scoreboard_index", lambda _self: ())
    args = cli._build_parser().parse_args(
        [
            "bench",
            "compare-runs",
            "missing-a",
            "missing-b",
            "--results-dir",
            str(tmp_path),
        ]
    )

    assert handle(args) == 1

    output = capsys.readouterr().out
    assert "compare-runs:" in output
    assert "missing-a" in output
    assert "scoreboards.jsonl" in output
    assert list(tmp_path.iterdir()) == []


def test_compare_runs_rejects_a_results_root_from_another_suite(tmp_path, monkeypatch, capsys):
    entries = (
        _entry("base", "a" * 40, 0.8),
        _entry("candidate", "b" * 40, 0.7),
    )
    monkeypatch.setattr(ResultStore, "load_scoreboard_index", lambda _self: entries)
    args = cli._build_parser().parse_args(
        [
            "bench",
            "compare-runs",
            "base",
            "candidate",
            "--suite",
            "fugu",
            "--results-dir",
            str(tmp_path),
        ]
    )

    assert handle(args) == 1
    assert "belong to suite 'core', not selected suite 'fugu'" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
