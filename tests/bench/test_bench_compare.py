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
        "suite": "accuracy",
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
    from kairyu.bench.adapters import ACCURACY_ROW_ORDER

    assert set(PUBLISHED_SCORES) == set(ACCURACY_ROW_ORDER)


def test_non_accuracy_suite_has_no_published_comparison():
    board = _scoreboard(**{"gsm8k": {"status": "completed", "score": 1.0}})
    board["suite"] = "core"

    with pytest.raises(ValueError, match="has no published comparison"):
        build_comparison(board)


def test_missing_suite_uses_accuracy_as_the_default():
    board = _scoreboard(**{"gpqa-diamond": {"status": "completed", "score": 0.9}})
    del board["suite"]

    comparison = build_comparison(board)

    assert comparison["rows"][0]["published"]["Fugu"] == 95.5


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


def test_partial_cells_are_marked_and_withhold_their_delta():
    """A partial denominator is not the full set, so the delta is withheld."""
    board = _scoreboard(
        **{
            "hle": {
                "status": "partial",
                "score": 0.10,
                "n": 2500,
                "n_scored": 2188,
                "reason": "312/2500 items unjudgeable",
            }
        }
    )
    markdown = render_comparison_markdown(build_comparison(board))
    assert "10.0*" in markdown  # partial score, marked
    assert "(2188/2500)" in markdown  # and its denominator is visible
    assert "-37.2" not in markdown  # no unmarked delta against a full-suite score
    assert "n/c" in markdown
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


async def test_core_run_writes_three_rows_without_a_published_comparison(
    tmp_path, http_factory, capsys
):
    from kairyu.bench.adapters import CORE_ROW_ORDER

    config = make_config(tmp_path, models=("m",), suite="core")
    runner = SuiteRunner(
        config,
        http_factory=http_factory,
        probe_docker=lambda: (False, "t"),
    )

    assert await runner.run() == 0

    run_dir = tmp_path / "results" / "test-run"
    scoreboard = json.loads(
        (run_dir / "scoreboard.json").read_text(encoding="utf-8")
    )
    assert scoreboard["benchmarks"] == list(CORE_ROW_ORDER)
    assert not (run_dir / "comparison.json").exists()
    assert not (run_dir / "comparison.md").exists()
    output = capsys.readouterr().out
    assert "# Core benchmark scoreboard" in output
    assert "# Accuracy vs published" not in output


async def test_comparison_is_printed_after_the_scoreboard(tmp_path, http_factory, capsys):
    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    await runner.run()
    out = capsys.readouterr().out
    assert out.index("# Accuracy benchmark scoreboard") < out.index(
        "# Accuracy vs published"
    )


# -- subset / fixture runs are not full-suite measurements ----------------------


async def test_smoke_run_cells_get_no_unmarked_delta(tmp_path, http_factory):
    """A 20-item smoke cell is legitimately `completed`; it is still not the set."""
    from kairyu.bench.runner import SuiteRunner
    from kairyu.bench.store import ResultStore

    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",), smoke=True)
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0

    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert pair.status == "completed"
    assert pair.comparable is False
    assert any("subset run" in reason for reason in pair.incomparable_reasons)

    comparison = json.loads(
        (tmp_path / "results" / "test-run" / "comparison.json").read_text(encoding="utf-8")
    )
    values = comparison["rows"][0]["measured"]["m"]
    assert values["comparable"] is False
    assert comparison["rows"][0]["deltas"]["m"] is None

    markdown = (tmp_path / "results" / "test-run" / "comparison.md").read_text(
        encoding="utf-8"
    )
    assert "not a full-suite measurement" in markdown
    assert "n/c" in markdown
    # the scoreboard operators read first says so too
    scoreboard = (tmp_path / "results" / "test-run" / "scoreboard.md").read_text(
        encoding="utf-8"
    )
    assert "not a full-suite measurement" in scoreboard


async def test_fixture_run_says_its_scores_are_not_measurements(tmp_path, http_factory):
    from kairyu.bench.runner import SuiteRunner

    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    assert config.offline_fixtures  # the shared helper uses fixtures
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    await runner.run()

    markdown = (tmp_path / "results" / "test-run" / "comparison.md").read_text(
        encoding="utf-8"
    )
    assert "offline fixture mode lacks cache-bound dataset identity" in markdown
    assert "not measurements" in markdown


def test_run_level_reasons_cover_limit_and_fixtures(tmp_path):
    from kairyu.bench.runner import run_level_incomparable_reasons

    config = make_config(tmp_path, models=("m",))
    reasons = run_level_incomparable_reasons(config, 20)
    assert any("at most 20 items" in reason for reason in reasons)
    assert any(
        "offline fixture mode lacks cache-bound dataset identity" in reason
        for reason in reasons
    )

    full = config.model_copy(update={"offline_fixtures": False})
    assert run_level_incomparable_reasons(full, None) == ()


# -- run-time substitutions ----------------------------------------------------


def test_runtime_substitution_withholds_its_delta():
    """The tau2 fallback records a reason on the cell, not just a footnote."""
    board = _scoreboard(
        **{
            "tau-bench-banking": {
                "status": "completed",
                "score": 0.2,
                "n": 10,
                "comparable": False,
                "incomparable_reasons": [
                    "the tau2 banking harness stood in for tau3; scores are not "
                    "directly comparable to Fugu's τ³ number"
                ],
            }
        }
    )
    comparison = build_comparison(board)
    assert comparison["rows"][0]["deltas"]["qwen3-32b"] is None
    markdown = render_comparison_markdown(comparison)
    assert "n/c" in markdown
    assert "tau2 banking harness stood in for tau3" in markdown


def test_legacy_scoreboards_fall_back_to_the_static_map():
    """A scoreboard written before the field existed still reads correctly."""
    board = _scoreboard(
        **{"long-context-reasoning": {"status": "completed", "score": 0.5, "n": 10}}
    )
    for cell in board["cells"]["long-context-reasoning"].values():
        cell.pop("comparable", None)
    comparison = build_comparison(board)
    assert comparison["rows"][0]["deltas"]["qwen3-32b"] is None
    assert "substitutes LongBench v2" in render_comparison_markdown(comparison)


# -- failed cells --------------------------------------------------------------


def test_failed_cell_without_a_score_is_still_marked_failed():
    """The common failed shape has score=None; it must not read as skipped."""
    board = _scoreboard(
        **{
            "terminal-bench": {
                "status": "failed",
                "score": None,
                "n": 0,
                "reason": "harbor failed (rc=2)",
            }
        }
    )
    markdown = render_comparison_markdown(build_comparison(board))
    row = [line for line in markdown.splitlines() if line.startswith("| terminal-bench")][0]
    assert "—!" in row  # visibly failed, not merely absent
    assert "harbor failed (rc=2)" in markdown


def test_failed_cell_with_a_score_gets_no_delta():
    board = _scoreboard(
        **{"terminal-bench": {"status": "failed", "score": 0.2, "n": 4, "reason": "rc=2"}}
    )
    comparison = build_comparison(board)
    assert comparison["rows"][0]["deltas"]["qwen3-32b"] is None
    markdown = render_comparison_markdown(comparison)
    row = [line for line in markdown.splitlines() if line.startswith("| terminal-bench")][0]
    assert "20.0!" in row
    assert "n/c" in row  # not the "*" the legend reserves for partial


async def test_resumed_legacy_pair_gets_this_runs_comparability(tmp_path, http_factory):
    """A pair stored before these fields existed must not resume as comparable.

    Its `comparable` defaults to True and its fingerprint still matches, so
    without re-stamping a resumed subset/fixture run would report a numeric
    published delta with no banner.
    """
    from kairyu.bench.runner import SuiteRunner
    from kairyu.bench.store import ResultStore

    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",), smoke=True)
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0

    store = ResultStore(tmp_path / "results", "test-run")
    fresh = store.load_pair("gpqa-diamond", "m")
    assert fresh.comparable is False

    # rewrite it the way a pre-field artifact looks: comparable, no reasons
    legacy = fresh.model_copy(update={"comparable": True, "incomparable_reasons": ()})
    store.save_pair(legacy)
    assert store.load_pair("gpqa-diamond", "m").comparable is True

    # resuming the same run id must not carry that forward
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    assert await runner.run() == 0

    resumed = store.load_pair("gpqa-diamond", "m")
    assert resumed.comparable is False  # re-stamped on disk, not only in memory
    assert any("subset run" in reason for reason in resumed.incomparable_reasons)

    comparison = json.loads(
        (tmp_path / "results" / "test-run" / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["rows"][0]["deltas"]["m"] is None
    markdown = (tmp_path / "results" / "test-run" / "comparison.md").read_text(
        encoding="utf-8"
    )
    assert "not a full-suite measurement" in markdown


async def test_resumed_pair_reasons_are_not_duplicated(tmp_path, http_factory):
    from kairyu.bench.runner import SuiteRunner
    from kairyu.bench.store import ResultStore

    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",), smoke=True)
    for _ in range(3):
        runner = SuiteRunner(
            config, http_factory=http_factory, probe_docker=lambda: (False, "t")
        )
        await runner.run()
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("gpqa-diamond", "m")
    assert len(pair.incomparable_reasons) == len(set(pair.incomparable_reasons))
