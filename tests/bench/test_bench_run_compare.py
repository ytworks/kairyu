"""Fail-closed cross-commit benchmark scoreboard comparisons."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kairyu.bench.compare import (
    build_run_comparison,
    render_run_comparison_markdown,
)

FINGERPRINT = "f" * 64
BASE_COMMIT = "a" * 40
CANDIDATE_COMMIT = "b" * 40


def _cell(
    score: float | None = 0.5,
    *,
    status: str = "completed",
    n: int | float | None = 10,
    n_scored: int | float | None = 10,
    reason: str | None = None,
    comparable: bool = True,
    incomparable_reasons: list[str] | None = None,
    confidence_interval: dict | None = None,
    cross_run_policy: str = "allowed",
    cross_run_reason: str | None = None,
) -> dict:
    return {
        "status": status,
        "score": score,
        "n": n,
        "n_scored": n_scored,
        "reason": reason,
        "comparable": comparable,
        "incomparable_reasons": list(incomparable_reasons or []),
        "confidence_interval": confidence_interval,
        "cross_run_policy": cross_run_policy,
        "cross_run_reason": cross_run_reason,
        "footnotes": [],
    }


def _entry(
    run_id: str,
    commit: str,
    *,
    suite: str = "core",
    fingerprint: str = FINGERPRINT,
    targets: tuple[str, ...] = ("model",),
    benchmarks: tuple[str, ...] = ("gsm8k",),
    cells: dict | None = None,
    display_names: dict[str, str] | None = None,
    offline_fixtures: bool = False,
) -> dict:
    if cells is None:
        cells = {benchmark: {target: _cell() for target in targets} for benchmark in benchmarks}
    if display_names is None:
        display_names = {benchmark: benchmark.upper() for benchmark in benchmarks}
    environment = {
        "git_commit": commit,
        "created_at": "2026-08-05T00:00:00+00:00",
        "kairyu_version": "0.1.0",
        "python": "3.12.3",
        "execution": {
            "runner": "local",
            "available": True,
            "availability_detail": "available",
        },
    }
    config = {
        "suite": suite,
        "offline_fixtures": offline_fixtures,
    }
    scoreboard = {
        "schema_version": 1,
        "run_id": run_id,
        "suite": suite,
        "fingerprint": fingerprint,
        "environment": dict(environment),
        "config": dict(config),
        "targets": list(targets),
        "benchmarks": list(benchmarks),
        "display_names": dict(display_names),
        "cells": deepcopy(cells),
        "footnotes": [f"methodology for {run_id}"],
    }
    return {
        "index_schema_version": 1,
        "run_id": run_id,
        "suite": suite,
        "git_commit": commit,
        "fingerprint": fingerprint,
        "run": {
            "run_id": run_id,
            "fingerprint": fingerprint,
            "config": dict(config),
            "environment": dict(environment),
        },
        "scoreboard": scoreboard,
    }


def _pair(
    baseline_cell: dict | None = None,
    candidate_cell: dict | None = None,
    **entry_kwargs,
) -> tuple[dict, dict]:
    baseline = _entry(
        "base",
        BASE_COMMIT,
        cells={"gsm8k": {"model": baseline_cell or _cell()}},
        **entry_kwargs,
    )
    candidate = _entry(
        "candidate",
        CANDIDATE_COMMIT,
        cells={"gsm8k": {"model": candidate_cell or _cell()}},
        **entry_kwargs,
    )
    return baseline, candidate


def _row(comparison: dict) -> dict:
    return comparison["rows"][0]


def test_delta_is_candidate_minus_baseline_and_renderer_states_direction():
    baseline, candidate = _pair(_cell(0.61), _cell(0.55))

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["deltas"]["model"] == -6.0
    assert _row(comparison)["delta_states"]["model"] == "comparable"
    markdown = render_run_comparison_markdown(comparison)
    assert markdown.startswith("# Accuracy run comparison — base → candidate")
    assert "candidate minus baseline" in markdown
    assert "negative is a regression" in markdown
    row = next(line for line in markdown.splitlines() if line.startswith("| GSM8K"))
    assert row == "| GSM8K | 55.0 (n=10) | 61.0 (n=10) | -6.0 |"


def test_renderer_labels_commits_as_local_harness_not_served_target_builds():
    baseline, candidate = _pair()

    markdown = render_run_comparison_markdown(build_run_comparison(baseline, candidate))

    assert f"local harness commit `{BASE_COMMIT}`" in markdown
    assert f"local harness commit `{CANDIDATE_COMMIT}`" in markdown
    assert "Commit provenance is local-harness-only" in markdown
    assert "served target build or commit is not attested" in markdown
    assert "Comparison runtime: Python `3.12.3`" in markdown


@pytest.mark.parametrize(
    "mutation",
    [
        lambda environment: environment.__setitem__("python", "3.11.9"),
        lambda environment: environment["execution"].__setitem__("runner", "docker"),
    ],
)
def test_python_or_execution_runtime_mismatch_refuses_commit_delta(mutation):
    baseline, candidate = _pair(_cell(0.6), _cell(0.5))
    mutation(candidate["scoreboard"]["environment"])
    mutation(candidate["run"]["environment"])

    with pytest.raises(ValueError, match="runtime mismatch"):
        build_run_comparison(baseline, candidate)


def test_transient_execution_availability_does_not_change_runtime_identity():
    baseline, candidate = _pair()
    candidate["scoreboard"]["environment"]["execution"]["available"] = False
    candidate["scoreboard"]["environment"]["execution"]["availability_detail"] = (
        "temporarily unavailable"
    )
    candidate["run"]["environment"] = deepcopy(candidate["scoreboard"]["environment"])

    assert _row(build_run_comparison(baseline, candidate))["delta_states"]["model"] == (
        "comparable"
    )


def _docker_execution(*, available: bool, resolved_suffix: str | None = None) -> dict:
    execution = {
        "runner": "docker",
        "image": "registry.example/bench@sha256:" + "d" * 64,
        "pull_policy": "never",
        "cpus": 1.0,
        "pids_limit": 64,
        "disk_mb": 256,
        "available": available,
        "availability_detail": "available" if available else "docker unavailable",
    }
    if resolved_suffix is not None:
        execution.update(
            {
                "resolved_image_id": "sha256:" + resolved_suffix * 64,
                "image_os": "linux",
                "image_architecture": "amd64",
                "image_variant": None,
            }
        )
    return execution


def _set_execution(entry: dict, execution: dict) -> None:
    entry["scoreboard"]["environment"]["execution"] = deepcopy(execution)
    entry["run"]["environment"]["execution"] = deepcopy(execution)


def test_unavailable_docker_withholds_only_execution_dependent_cells():
    benchmarks = ("gsm8k", "scicode")
    baseline = _entry(
        "base",
        BASE_COMMIT,
        benchmarks=benchmarks,
        cells={
            "gsm8k": {"model": _cell(0.4)},
            "scicode": {
                "model": _cell(
                    0.4,
                    cross_run_policy="withheld_unpinned_execution",
                    cross_run_reason=(
                        "code execution lacks a resolved immutable image and platform identity"
                    ),
                )
            },
        },
    )
    candidate = _entry(
        "candidate",
        CANDIDATE_COMMIT,
        benchmarks=benchmarks,
        cells={
            "gsm8k": {"model": _cell(0.5)},
            "scicode": {"model": _cell(0.5)},
        },
    )
    _set_execution(baseline, _docker_execution(available=False))
    _set_execution(candidate, _docker_execution(available=True, resolved_suffix="e"))

    comparison = build_run_comparison(baseline, candidate)
    rows = {row["benchmark"]: row for row in comparison["rows"]}

    assert rows["gsm8k"]["deltas"]["model"] == 10.0
    assert rows["gsm8k"]["delta_states"]["model"] == "comparable"
    assert rows["scicode"]["deltas"]["model"] is None
    assert rows["scicode"]["delta_states"]["model"] == "not_comparable"
    assert "resolved_image_id" not in comparison["comparison_runtime"]["execution"]


def test_two_available_docker_runs_still_require_exact_resolved_identity():
    baseline, candidate = _pair()
    _set_execution(baseline, _docker_execution(available=True, resolved_suffix="d"))
    _set_execution(candidate, _docker_execution(available=True, resolved_suffix="e"))

    with pytest.raises(ValueError, match="runtime mismatch"):
        build_run_comparison(baseline, candidate)


def test_same_normalized_subset_boundary_allows_diagnostic_delta():
    baseline, candidate = _pair(
        _cell(
            0.40,
            comparable=False,
            incomparable_reasons=[" subset   run: at most 20 items ", "duplicate"],
        ),
        _cell(
            0.45,
            comparable=False,
            incomparable_reasons=["duplicate", "subset run: at most 20 items"],
        ),
    )

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["deltas"]["model"] == 5.0
    assert _row(comparison)["delta_states"]["model"] == "comparable"
    markdown = render_run_comparison_markdown(comparison)
    assert "+5.0" in markdown
    assert "both runs carry the same diagnostic boundary" in markdown


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        pytest.param(
            "withheld_unresolved_runtime",
            "runtime dataset, image, or executable content is unresolved",
            id="external-runtime",
        ),
        pytest.param(
            "withheld_unpinned_execution",
            "code execution lacks a resolved immutable image and platform identity",
            id="code-execution",
        ),
    ],
)
def test_cross_run_policy_always_withholds_even_when_both_runs_match(policy, reason):
    baseline, candidate = _pair(
        _cell(0.4, cross_run_policy=policy, cross_run_reason=reason),
        _cell(0.5, cross_run_policy=policy, cross_run_reason=reason),
    )

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["deltas"]["model"] is None
    assert _row(comparison)["delta_states"]["model"] == "not_comparable"
    assert any(
        "cross-run policy withholds" in item
        for item in _row(comparison)["delta_reasons"]["model"]
    )
    assert "n/c" in render_run_comparison_markdown(comparison)


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        pytest.param("unknown", None, id="unknown-policy"),
        pytest.param("withheld_unresolved_runtime", None, id="missing-reason"),
        pytest.param("allowed", "unexpected", id="reason-on-allowed"),
    ],
)
def test_malformed_cross_run_policy_fails_closed(policy, reason):
    baseline, candidate = _pair()
    candidate_cell = candidate["scoreboard"]["cells"]["gsm8k"]["model"]
    candidate_cell["cross_run_policy"] = policy
    candidate_cell["cross_run_reason"] = reason

    with pytest.raises(ValueError, match="cross_run"):
        build_run_comparison(baseline, candidate)


def test_explicit_non_comparable_cells_without_a_boundary_never_get_a_delta():
    baseline, candidate = _pair(
        _cell(0.40, comparable=False),
        _cell(0.45, comparable=False),
    )

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["deltas"]["model"] is None
    assert _row(comparison)["delta_states"]["model"] == "not_comparable"
    assert any(
        "declared non-comparable" in reason for reason in _row(comparison)["delta_reasons"]["model"]
    )
    assert "| GSM8K | 45.0 (n=10) | 40.0 (n=10) | n/c |" in (
        render_run_comparison_markdown(comparison)
    )


def test_offline_fixture_scores_are_not_measurements_and_delta_is_withheld():
    baseline, candidate = _pair(
        _cell(0.4),
        _cell(0.5),
        offline_fixtures=True,
    )

    comparison = build_run_comparison(baseline, candidate)

    assert comparison["offline_fixtures"] is True
    assert _row(comparison)["deltas"]["model"] is None
    assert _row(comparison)["delta_states"]["model"] == "not_comparable"
    assert any(
        "not measurements" in reason for reason in _row(comparison)["delta_reasons"]["model"]
    )
    markdown = render_run_comparison_markdown(comparison)
    assert "Offline fixture scores are not measurements" in markdown
    assert "| GSM8K | 50.0 (n=10) | 40.0 (n=10) | n/c |" in markdown


@pytest.mark.parametrize(
    ("baseline_cell", "candidate_cell", "candidate_text", "delta_text"),
    [
        pytest.param(
            _cell(0.5),
            _cell(None, status="skipped", n=0, n_scored=None, reason="not run"),
            "—",
            "—",
            id="missing-candidate",
        ),
        pytest.param(
            _cell(0.5),
            _cell(0.4, status="partial", reason="one item unjudged"),
            "40.0*",
            "n/c",
            id="partial-candidate",
        ),
        pytest.param(
            _cell(0.5),
            _cell(None, status="failed", n=0, n_scored=None, reason="crashed"),
            "—!",
            "—",
            id="failed-candidate-without-score",
        ),
        pytest.param(
            _cell(float("nan")),
            _cell(0.5),
            "50.0",
            "—",
            id="non-finite-baseline",
        ),
    ],
)
def test_missing_partial_failed_and_nonfinite_scores_are_explicit(
    baseline_cell, candidate_cell, candidate_text, delta_text
):
    baseline, candidate = _pair(baseline_cell, candidate_cell)

    comparison = build_run_comparison(baseline, candidate)
    markdown = render_run_comparison_markdown(comparison)
    rendered_row = next(line for line in markdown.splitlines() if line.startswith("| GSM8K"))

    assert candidate_text in rendered_row
    assert [part.strip() for part in rendered_row.strip("|").split("|")][-1] == delta_text
    assert _row(comparison)["deltas"]["model"] is None
    assert "delta withheld" in markdown


def test_reason_difference_withholds_a_numeric_delta():
    baseline, candidate = _pair(
        _cell(0.4, incomparable_reasons=["tau2 fallback"]),
        _cell(0.5, incomparable_reasons=["tau3 harness"]),
    )

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["delta_states"]["model"] == "not_comparable"
    assert any(
        "incomparable reasons differ" in reason
        for reason in _row(comparison)["delta_reasons"]["model"]
    )
    assert "| GSM8K | 50.0 (n=10) | 40.0 (n=10) | n/c |" in (
        render_run_comparison_markdown(comparison)
    )


def test_completed_cell_reason_difference_also_withholds_delta():
    baseline, candidate = _pair(
        _cell(0.4, reason="same-score diagnostic A"),
        _cell(0.5, reason="same-score diagnostic B"),
    )

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["delta_states"]["model"] == "not_comparable"
    assert any(
        "cell reasons differ" in reason for reason in _row(comparison)["delta_reasons"]["model"]
    )


@pytest.mark.parametrize(
    ("baseline_cell", "candidate_cell", "expected_reason"),
    [
        pytest.param(_cell(0.4, n=10), _cell(0.5, n=11), "denominators differ"),
        pytest.param(
            _cell(0.4, n=10, n_scored=10),
            _cell(0.5, n=10, n_scored=9),
            "denominators differ",
        ),
        pytest.param(
            _cell(0.4, n=0, n_scored=None),
            _cell(0.5, n=0, n_scored=None),
            "no valid positive denominator",
        ),
    ],
)
def test_denominator_difference_or_absence_withholds_delta(
    baseline_cell, candidate_cell, expected_reason
):
    baseline, candidate = _pair(baseline_cell, candidate_cell)

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["delta_states"]["model"] == "not_comparable"
    assert any(expected_reason in reason for reason in _row(comparison)["delta_reasons"]["model"])


def test_implicit_and_explicit_equal_scored_denominators_compare():
    baseline, candidate = _pair(
        _cell(0.4, n=10, n_scored=None),
        _cell(0.5, n=10, n_scored=10),
    )

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["deltas"]["model"] == 10.0


def test_zero_is_a_measured_score_not_a_missing_value():
    baseline, candidate = _pair(_cell(0.0), _cell(0.1))

    comparison = build_run_comparison(baseline, candidate)
    markdown = render_run_comparison_markdown(comparison)

    assert _row(comparison)["deltas"]["model"] == 10.0
    assert "| GSM8K | 10.0 (n=10) | 0.0 (n=10) | +10.0 |" in markdown


def test_one_is_a_valid_accuracy_boundary():
    baseline, candidate = _pair(_cell(0.9), _cell(1.0))

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["deltas"]["model"] == 10.0
    assert _row(comparison)["candidate"]["model"]["score"] == 100.0


@pytest.mark.parametrize(
    "invalid_score",
    [
        pytest.param(True, id="bool"),
        pytest.param(10**400, id="integer-too-large-for-a-float"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(-0.01, id="below-zero"),
        pytest.param(1.01, id="above-one"),
    ],
)
def test_invalid_accuracy_scores_are_missing_and_never_get_a_delta(invalid_score):
    baseline, candidate = _pair(_cell(invalid_score), _cell(0.5))

    comparison = build_run_comparison(baseline, candidate)

    assert _row(comparison)["baseline"]["model"]["score"] is None
    assert _row(comparison)["deltas"]["model"] is None
    assert _row(comparison)["delta_states"]["model"] == "missing"
    assert any(
        "valid finite score in [0, 1]" in reason
        for reason in _row(comparison)["delta_reasons"]["model"]
    )


def test_valid_wilson_intervals_render_for_both_runs():
    from kairyu.bench.aggregate import _wilson_bounds

    def interval(successes: int) -> dict:
        lower, upper = _wilson_bounds(successes, 10)
        return {
            "method": "wilson",
            "confidence": 0.95,
            "successes": successes,
            "trials": 10,
            "lower": lower,
            "upper": upper,
        }

    baseline, candidate = _pair(
        _cell(0.5, confidence_interval=interval(5)),
        _cell(0.6, confidence_interval=interval(6)),
    )

    markdown = render_run_comparison_markdown(build_run_comparison(baseline, candidate))
    rendered_row = next(line for line in markdown.splitlines() if line.startswith("| GSM8K"))

    assert "60.0 [31.3–83.2] (n=10)" in rendered_row
    assert "50.0 [23.7–76.3] (n=10)" in rendered_row
    assert "validated 95% Wilson CIs" in markdown


def test_malformed_wilson_interval_is_not_labelled_or_rendered():
    from kairyu.bench.aggregate import _wilson_bounds

    lower, upper = _wilson_bounds(5, 10)
    malformed = {
        "method": "wilson",
        "confidence": 0.95,
        "successes": 5,
        "trials": 10,
        "lower": lower,
        "upper": upper + 0.1,
    }
    baseline, candidate = _pair(
        _cell(0.5, confidence_interval=malformed),
        _cell(0.6),
    )

    markdown = render_run_comparison_markdown(build_run_comparison(baseline, candidate))
    rendered_row = next(line for line in markdown.splitlines() if line.startswith("| GSM8K"))

    assert "95%" not in rendered_row
    assert "[" not in rendered_row
    assert "50.0 (n=10)" in rendered_row


def test_huge_integer_wilson_fields_are_not_labelled_or_rendered():
    malformed = {
        "method": "wilson",
        "confidence": 10**400,
        "successes": 5,
        "trials": 10,
        "lower": 0.2,
        "upper": 0.8,
    }
    baseline, candidate = _pair(
        _cell(0.5, confidence_interval=malformed),
        _cell(0.6),
    )

    markdown = render_run_comparison_markdown(build_run_comparison(baseline, candidate))
    rendered_row = next(line for line in markdown.splitlines() if line.startswith("| GSM8K"))

    assert "95%" not in rendered_row
    assert "[" not in rendered_row
    assert "50.0 (n=10)" in rendered_row


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param("suite", "suite mismatch", id="suite"),
        pytest.param("fingerprint", "fingerprint mismatch", id="fingerprint"),
        pytest.param("schema", "schema_version 2 is unsupported", id="schema"),
        pytest.param("targets", "target structure mismatch", id="targets"),
        pytest.param("benchmarks", "benchmark structure mismatch", id="benchmarks"),
        pytest.param("display", "display names differ", id="display-names"),
    ],
)
def test_run_identity_and_structure_mismatches_fail_closed(mutation, message):
    baseline, candidate = _pair()
    if mutation == "suite":
        candidate["suite"] = "accuracy"
        candidate["scoreboard"]["suite"] = "accuracy"
        candidate["run"]["config"]["suite"] = "accuracy"
    elif mutation == "fingerprint":
        candidate["fingerprint"] = "e" * 64
        candidate["run"]["fingerprint"] = "e" * 64
        candidate["scoreboard"]["fingerprint"] = "e" * 64
    elif mutation == "schema":
        candidate["scoreboard"]["schema_version"] = 2
    elif mutation == "targets":
        candidate["scoreboard"]["targets"] = ["other"]
        candidate["scoreboard"]["cells"] = {"gsm8k": {"other": _cell()}}
    elif mutation == "benchmarks":
        candidate["scoreboard"]["benchmarks"] = ["mmlu"]
        candidate["scoreboard"]["display_names"] = {"mmlu": "MMLU"}
        candidate["scoreboard"]["cells"] = {"mmlu": {"model": _cell()}}
    else:
        candidate["scoreboard"]["display_names"]["gsm8k"] = "GSM 8K"

    with pytest.raises(ValueError, match=message):
        build_run_comparison(baseline, candidate)


def test_target_order_is_part_of_the_strict_structure():
    targets = ("first", "second")
    baseline = _entry("base", BASE_COMMIT, targets=targets)
    candidate = _entry("candidate", CANDIDATE_COMMIT, targets=targets)
    candidate["scoreboard"]["targets"].reverse()

    with pytest.raises(ValueError, match="target structure mismatch"):
        build_run_comparison(baseline, candidate)


def test_unknown_equal_scoreboard_schema_versions_are_not_interpreted_as_v1():
    baseline, candidate = _pair()
    baseline["scoreboard"]["schema_version"] = 2
    candidate["scoreboard"]["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported"):
        build_run_comparison(baseline, candidate)


def test_missing_cell_and_duplicate_targets_are_malformed_not_silently_aligned():
    baseline, candidate = _pair()
    candidate["scoreboard"]["cells"]["gsm8k"] = {}
    with pytest.raises(ValueError, match="target structure"):
        build_run_comparison(baseline, candidate)

    baseline, candidate = _pair()
    candidate["scoreboard"]["targets"] = ["model", "model"]
    with pytest.raises(ValueError, match="duplicates"):
        build_run_comparison(baseline, candidate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param("bad-fingerprint", "lowercase SHA-256", id="fingerprint-format"),
        pytest.param("board-fingerprint", "scoreboard", id="scoreboard-fingerprint"),
        pytest.param("board-commit", "scoreboard environment", id="board-commit"),
        pytest.param("run-commit", "run metadata", id="run-commit"),
        pytest.param("run-id", "run metadata", id="run-id"),
    ],
)
def test_entry_metadata_must_bind_the_embedded_scoreboard_and_run(mutation, message):
    baseline, candidate = _pair()
    if mutation == "bad-fingerprint":
        candidate["fingerprint"] = "not-a-digest"
    elif mutation == "board-fingerprint":
        candidate["scoreboard"]["fingerprint"] = "e" * 64
    elif mutation == "board-commit":
        candidate["scoreboard"]["environment"]["git_commit"] = "c" * 40
    elif mutation == "run-commit":
        candidate["run"]["environment"]["git_commit"] = "c" * 40
    else:
        candidate["run"]["run_id"] = "someone-else"

    with pytest.raises(ValueError, match=message):
        build_run_comparison(baseline, candidate)


def test_sha256_git_object_ids_are_accepted():
    baseline, candidate = _pair()
    baseline["git_commit"] = "a" * 64
    baseline["run"]["environment"]["git_commit"] = "a" * 64
    baseline["scoreboard"]["environment"]["git_commit"] = "a" * 64

    assert build_run_comparison(baseline, candidate)["baseline"]["git_commit"] == ("a" * 64)


def test_methodology_notes_from_both_runs_are_retained_and_labelled():
    baseline, candidate = _pair()

    comparison = build_run_comparison(baseline, candidate)
    markdown = render_run_comparison_markdown(comparison)

    assert comparison["baseline_methodology_notes"] == ["methodology for base"]
    assert comparison["candidate_methodology_notes"] == ["methodology for candidate"]
    assert "## Baseline methodology notes" in markdown
    assert "## Candidate methodology notes" in markdown
