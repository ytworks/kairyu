"""Accuracy report: a run's scores next to the published Fugu-release table.

The published numbers live in images on the release page, so they are committed
constants with provenance. The report's job is to make the delta legible without
letting it imply parity it does not have: a missing measurement must never read
as zero, a partial denominator must be marked, and a row kairyu measures
differently must not print a delta at all.
"""

import json

import pytest
from conftest import make_config

from kairyu.bench.compare import build_comparison, render_comparison_markdown
from kairyu.bench.reference import (
    HEADLINE_MODELS,
    NOT_COMPARABLE,
    PROVIDER_REPORTED,
    PUBLISHED_SCORES,
    published_models,
)
from kairyu.bench.runner import SuiteRunner


def _scoreboard(**cells) -> dict:
    benchmarks = list(cells)
    return {
        "run_id": "r1",
        "suite": "fugu",
        "targets": ["qwen3-32b"],
        "benchmarks": benchmarks,
        "display_names": {name: name for name in benchmarks},
        "cells": {
            name: {"qwen3-32b": {"footnotes": [], **cell}} for name, cell in cells.items()
        },
        "footnotes": ["gpqa-diamond: seed-shuffled choices"],
    }


# -- reference data ------------------------------------------------------------


def test_every_suite_row_has_published_scores():
    from kairyu.bench.adapters import FUGU_ROW_ORDER

    assert set(PUBLISHED_SCORES) == set(FUGU_ROW_ORDER)


def test_published_scores_are_percentages_with_fugu_present():
    for benchmark, scores in PUBLISHED_SCORES.items():
        assert "Fugu" in scores, benchmark
        for model, value in scores.items():
            assert 0.0 <= value <= 100.0, f"{benchmark}/{model} = {value}"


def test_published_models_follow_the_headline_order():
    models = published_models("swe-bench-pro")
    assert models[: len(HEADLINE_MODELS)] == HEADLINE_MODELS
    assert models[-1] == "Fable 5"  # figure-only column comes after


def test_baselines_are_declared_provider_reported():
    assert "Opus 4.8" in PROVIDER_REPORTED
    assert "Fugu" not in PROVIDER_REPORTED
    assert "Fugu Ultra" not in PROVIDER_REPORTED


# -- delta semantics -----------------------------------------------------------


def test_delta_is_measured_minus_published_fugu():
    board = _scoreboard(**{"gpqa-diamond": {"status": "completed", "score": 0.90, "n": 198}})
    comparison = build_comparison(board)
    row = comparison["rows"][0]
    assert row["measured"]["qwen3-32b"]["score"] == 90.0
    # published Fugu GPQA-D is 95.5
    assert row["deltas"]["qwen3-32b"] == pytest.approx(-5.5, abs=0.05)


def test_missing_measurement_has_no_delta_and_renders_as_dash():
    board = _scoreboard(
        **{"hle": {"status": "skipped", "score": None, "n": 0, "reason": "gated"}}
    )
    comparison = build_comparison(board)
    row = comparison["rows"][0]
    assert row["measured"]["qwen3-32b"]["score"] is None
    assert row["deltas"]["qwen3-32b"] is None
    markdown = render_comparison_markdown(comparison)
    assert "| hle | — |" in markdown
    assert "not measured — gated" in markdown


def test_substituted_row_withholds_its_delta():
    board = _scoreboard(
        **{"long-context-reasoning": {"status": "completed", "score": 0.5, "n": 10}}
    )
    comparison = build_comparison(board)
    row = comparison["rows"][0]
    assert row["not_comparable"] == NOT_COMPARABLE["long-context-reasoning"]
    assert row["deltas"]["qwen3-32b"] is None
    assert row["measured"]["qwen3-32b"]["comparable"] is False
    markdown = render_comparison_markdown(comparison)
    assert "n/c" in markdown
    assert "NOT COMPARABLE" in markdown


def test_partial_cells_are_marked_in_score_and_delta():
    board = _scoreboard(
        **{
            "hle": {
                "status": "partial",
                "score": 0.10,
                "n": 2500,
                "reason": "312/2500 items unjudgeable",
            }
        }
    )
    markdown = render_comparison_markdown(build_comparison(board))
    assert "10.0*" in markdown  # partial score
    assert "-37.2*" in markdown  # partial delta vs published 47.2
    assert "312/2500 items unjudgeable" in markdown


def test_failed_cells_are_marked_distinctly():
    board = _scoreboard(
        **{"terminal-bench": {"status": "failed", "score": 0.2, "n": 1, "reason": "harbor rc=2"}}
    )
    markdown = render_comparison_markdown(build_comparison(board))
    assert "20.0!" in markdown
    assert "harbor rc=2" in markdown


# -- rendering -----------------------------------------------------------------


def test_report_states_its_provenance_and_the_provider_caveat():
    markdown = render_comparison_markdown(
        build_comparison(_scoreboard(**{"gpqa-diamond": {"status": "completed", "score": 0.9}}))
    )
    assert "https://sakana.ai/fugu-release/" in markdown
    assert "2026-07-25" in markdown
    assert "cannot be fetched programmatically" in markdown
    assert "reported by the model providers" in markdown
    assert "mini-swe-agent" in markdown  # the page's own * footnote


def test_report_surfaces_the_published_hle_text_variant():
    markdown = render_comparison_markdown(
        build_comparison(_scoreboard(**{"hle": {"status": "completed", "score": 0.4}}))
    )
    assert "text-only subset" in markdown
    assert "Fable 5 53.3" in markdown  # the variant's own numbers, not merged in


def test_report_reprints_the_runs_methodology_footnotes():
    markdown = render_comparison_markdown(
        build_comparison(_scoreboard(**{"gpqa-diamond": {"status": "completed", "score": 0.9}}))
    )
    assert "Methodology notes from this run" in markdown
    assert "seed-shuffled choices" in markdown


def test_published_columns_appear_for_every_rendered_row():
    board = _scoreboard(
        **{
            "gpqa-diamond": {"status": "completed", "score": 0.9},
            "mrcr-v2": {"status": "completed", "score": 0.5},
        }
    )
    markdown = render_comparison_markdown(build_comparison(board))
    header = [line for line in markdown.splitlines() if line.startswith("| Benchmark")][0]
    for model in HEADLINE_MODELS:
        assert model in header
    assert "Δ qwen3-32b" in header


# -- runner integration --------------------------------------------------------


async def test_run_writes_the_comparison_next_to_the_scoreboard(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0

    run_dir = tmp_path / "results" / "test-run"
    assert (run_dir / "comparison.md").exists()
    comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["delta_against"] == "Fugu"
    assert comparison["reference"]["retrieved_on"] == "2026-07-25"
    row = comparison["rows"][0]
    assert row["benchmark"] == "gpqa-diamond"
    assert row["published"]["Fugu"] == 95.5


async def test_comparison_is_printed_after_the_scoreboard(tmp_path, http_factory, capsys):
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    await runner.run()
    out = capsys.readouterr().out
    assert out.index("# Fugu benchmark scoreboard") < out.index("# Accuracy vs published")
