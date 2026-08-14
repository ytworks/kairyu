"""Scoreboard aggregation: suite row order, cell rendering, and footnotes."""

import json
from argparse import Namespace

import pytest

from evals.adapters import (
    CORE_ROW_ORDER,
    LONG_CONTEXT_ROW_ORDER,
    QUANTIZATION_ROW_ORDER,
    STRUCTURED_ROW_ORDER,
    all_adapters,
    suite_adapters,
)
from evals.aggregate import _wilson_bounds, build_scoreboard, render_markdown
from evals.cli import _handle_report
from evals.types import (
    ItemResult,
    PairResult,
)


def _pair(benchmark, target, status="completed", score=0.5, reason=None, annotations=()):
    metrics = {"score": score, "n_total": 3}
    return PairResult(
        benchmark=benchmark,
        target=target,
        status=status,
        reason=reason,
        metrics=metrics,
        annotations=tuple(annotations),
        started_at="t0",
        finished_at="t1",
    )


def _binary_pair(
    benchmark,
    target,
    *,
    successes,
    trials,
    total=None,
    status="completed",
    reason=None,
):
    total = trials if total is None else total
    scores = [1.0] * successes + [0.0] * (trials - successes)
    items = [
        ItemResult(item_id=f"scored-{index}", status="completed", score=score)
        for index, score in enumerate(scores)
    ]
    items.extend(
        ItemResult(item_id=f"missing-{index}", status="unjudged") for index in range(total - trials)
    )
    return PairResult(
        benchmark=benchmark,
        target=target,
        status=status,
        reason=reason,
        metrics={
            "score": successes / trials,
            "n_total": total,
            "n_scored": trials,
        },
        items=tuple(items),
        started_at="t0",
        finished_at="t1",
    )


def _board(
    pairs,
    targets,
    config=None,
    target_configs=None,
    suite="quantization",
):
    return build_scoreboard(
        run_id="run-1",
        suite=suite,
        config=config or {},
        environment={},
        pairs=pairs,
        targets=targets,
        target_configs=target_configs,
    )


def test_core_scoreboard_uses_its_canonical_row_order_and_title():
    board = _board(
        [
            _pair("ifeval", "m"),
            _pair("gsm8k", "m"),
            _pair("mmlu", "m"),
        ],
        ["m"],
        suite="core",
    )

    assert board["benchmarks"] == list(CORE_ROW_ORDER)
    assert render_markdown(board).startswith("# Core benchmark scoreboard — run run-1")


def test_quantization_suite_preserves_core_and_adds_one_reasoning_row():
    pairs = [_pair(benchmark, "m") for benchmark in reversed(QUANTIZATION_ROW_ORDER)]

    board = _board(pairs, ["m"], suite="quantization")

    assert QUANTIZATION_ROW_ORDER == (*CORE_ROW_ORDER, "gpqa-diamond")
    assert board["benchmarks"] == list(QUANTIZATION_ROW_ORDER)
    assert render_markdown(board).startswith(
        "# Quantization benchmark scoreboard — run run-1"
    )


def test_structured_suite_is_isolated_from_core_and_quantization():
    assert STRUCTURED_ROW_ORDER == ("structured-output",)
    assert [adapter.info.name for adapter in suite_adapters("structured")] == list(
        STRUCTURED_ROW_ORDER
    )
    assert "structured-output" not in CORE_ROW_ORDER
    assert "structured-output" not in QUANTIZATION_ROW_ORDER


def test_long_context_scoreboard_is_the_length_curve():
    pairs = [_pair(benchmark, "m") for benchmark in reversed(LONG_CONTEXT_ROW_ORDER)]

    board = _board(pairs, ["m"], suite="long-context")

    assert board["benchmarks"] == list(LONG_CONTEXT_ROW_ORDER)
    assert render_markdown(board).startswith(
        "# Long Context benchmark scoreboard — run run-1"
    )


def test_only_and_exclude_names_are_validated_within_the_selected_suite():
    assert [adapter.info.name for adapter in suite_adapters("core")] == list(CORE_ROW_ORDER)
    with pytest.raises(ValueError, match="gpqa-diamond"):
        suite_adapters("core", only=("gpqa-diamond",))
    with pytest.raises(ValueError, match="gsm8k"):
        suite_adapters("core", exclude=("gpqa-diamond",))
    with pytest.raises(
        ValueError,
        match="available: core, quantization, structured, long-context",
    ):
        suite_adapters("unknown")


def test_report_resolves_a_core_run_under_its_suite_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "bench" / "results" / "core" / "core-report"
    pair_dir = run_dir / "pair"
    pair_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "core-report",
                "config": {
                    "suite": "core",
                    "targets": [
                        {
                            "base_url": "http://gateway.test/v1",
                            "model": "m",
                        }
                    ],
                },
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "result.json").write_text(_pair("gsm8k", "m").model_dump_json(), encoding="utf-8")

    args = Namespace(
        run="core-report",
        suite="core",
        results_dir=None,
    )
    assert _handle_report(args) == 0
    assert (run_dir / "scoreboard.json").exists()
    assert not (run_dir / "comparison.json").exists()
    assert not (run_dir / "comparison.md").exists()


def test_cells_and_footnotes():
    pairs = [
        _pair("gpqa-diamond", "m", score=0.412),
        _pair(
            "gpqa-diamond",
            "kairyu-auto",
            status="skipped",
            score=None,
            reason="dataset not in cache (gated)",
        ),
    ]
    board = _board(pairs, ["m", "kairyu-auto"])
    cell = board["cells"]["gpqa-diamond"]["m"]
    assert cell["status"] == "completed" and cell["score"] == 0.412
    skipped = board["cells"]["gpqa-diamond"]["kairyu-auto"]
    assert skipped["status"] == "skipped"
    assert skipped["footnotes"]  # skip reason recorded
    note = board["footnotes"][skipped["footnotes"][0] - 1]
    assert "dataset not in cache" in note


def test_annotations_become_footnotes():
    pairs = [_pair("gpqa-diamond", "m", annotations=("substitute suite",))]
    board = _board(pairs, ["m"])
    assert any("substitute suite" in note for note in board["footnotes"])


def test_missing_pair_rendered_as_not_run():
    pairs = [_pair("gpqa-diamond", "m")]
    board = _board(pairs, ["m", "other"])
    assert board["cells"]["gpqa-diamond"]["other"]["reason"] == "not run"
    assert board["cells"]["gpqa-diamond"]["other"]["confidence_interval"] is None


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        (8, 20, (0.2188065324, 0.6134184992)),
        (200, 500, (0.3579789908, 0.4435458773)),
        (5000, 10000, (0.4902020618, 0.5097979382)),
        (0, 1, (0.0, 0.7934506856)),
        (1, 1, (0.2065493144, 1.0)),
    ],
)
def test_wilson_bounds(successes, trials, expected):
    assert _wilson_bounds(successes, trials) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(-1, 1), (2, 1), (0, 0)],
)
def test_wilson_bounds_reject_invalid_counts(successes, trials):
    assert _wilson_bounds(successes, trials) is None


def test_scoreboard_records_machine_readable_wilson_interval():
    pair = _binary_pair("gpqa-diamond", "m", successes=8, trials=20)

    cell = _board([pair], ["m"])["cells"]["gpqa-diamond"]["m"]
    interval = cell["confidence_interval"]

    assert interval["method"] == "wilson"
    assert {key: value for key, value in interval.items() if key != "method"} == pytest.approx(
        {
            "confidence": 0.95,
            "successes": 8,
            "trials": 20,
            "lower": 0.2188065324,
            "upper": 0.6134184992,
        },
        abs=1e-10,
    )


def test_only_contractually_binary_adapters_enable_wilson_intervals():
    assert {name for name, adapter in all_adapters().items() if adapter.info.binary_outcomes} == {
        "gsm8k",
        "gpqa-diamond",
        "ifeval",
        "mmlu",
        *LONG_CONTEXT_ROW_ORDER,
        "structured-output",
    }


@pytest.mark.parametrize(
    "metrics",
    [
        {"score": 0.4, "n_total": 500, "n_scored": 20},
        {"score": 0.4, "n_total": 20, "n_scored": 19},
        {"score": 0.45, "n_total": 20, "n_scored": 20},
    ],
)
def test_inconsistent_binary_evidence_fails_closed(metrics):
    pair = _binary_pair("gpqa-diamond", "m", successes=8, trials=20).model_copy(
        update={"metrics": metrics}
    )

    cell = _board([pair], ["m"])["cells"]["gpqa-diamond"]["m"]

    assert cell["confidence_interval"] is None


def test_markdown_wilson_width_distinguishes_smoke_from_full_sample():
    pairs = [
        _binary_pair("gpqa-diamond", "smoke", successes=8, trials=20),
        _binary_pair("gpqa-diamond", "full", successes=200, trials=500),
    ]

    text = render_markdown(_board(pairs, ["smoke", "full"]))

    assert "40.0 [21.9–61.3] (n=20)" in text
    assert "40.0 [35.8–44.4] (n=500)" in text
    assert "brackets are 95% Wilson CIs for binary item outcomes" in text


def test_fractional_and_legacy_scores_show_n_without_binomial_interval():
    fractional = PairResult(
        benchmark="gpqa-diamond",
        target="fractional",
        status="completed",
        metrics={"score": 0.5, "n_total": 2, "n_scored": 2},
        items=(
            ItemResult(item_id="a", status="completed", score=0.25),
            ItemResult(item_id="b", status="completed", score=0.75),
        ),
        started_at="t0",
        finished_at="t1",
    )
    legacy = _pair("gpqa-diamond", "legacy", score=0.5)

    board = _board([fractional, legacy], ["fractional", "legacy"])
    cells = board["cells"]["gpqa-diamond"]
    text = render_markdown(board)

    assert cells["fractional"]["confidence_interval"] is None
    assert cells["legacy"]["confidence_interval"] is None
    assert "| GPQA Diamond | 50.0 (n=2) | 50.0 (n=3) |" in text


def test_partial_binary_score_withholds_interval_and_keeps_counts_and_markers():
    pair = _binary_pair(
        "gpqa-diamond",
        "partial",
        successes=1,
        trials=2,
        total=4,
        status="partial",
        reason="2/4 unjudged",
    )

    board = _board([pair], ["partial"])
    text = render_markdown(board)
    row = next(line for line in text.splitlines() if line.startswith("| GPQA"))
    header = next(line for line in text.splitlines() if line.startswith("| Benchmark"))

    assert board["cells"]["gpqa-diamond"]["partial"]["confidence_interval"] is None
    assert "50.0* (n=2/4)[^1]" in row
    assert len(row.split("|")) == len(header.split("|"))
    assert "2/4 unjudged" in text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "jeffreys"),
        ("confidence", 0.5),
        ("trials", 19),
        ("successes", 21),
        ("lower", 0.1),
    ],
)
def test_renderer_does_not_mislabel_tampered_interval(field, value):
    pair = _binary_pair("gpqa-diamond", "m", successes=8, trials=20)
    board = _board([pair], ["m"])
    cell = board["cells"]["gpqa-diamond"]["m"]
    cell["confidence_interval"][field] = value

    row = next(line for line in render_markdown(board).splitlines() if line.startswith("| GPQA"))

    assert row == "| GPQA Diamond | 40.0 (n=20) |"


def test_renderer_withholds_interval_when_point_estimate_is_tampered():
    pair = _binary_pair("gpqa-diamond", "m", successes=8, trials=20)
    board = _board([pair], ["m"])
    board["cells"]["gpqa-diamond"]["m"]["score"] = 0.5

    row = next(line for line in render_markdown(board).splitlines() if line.startswith("| GPQA"))

    assert row == "| GPQA Diamond | 50.0 (n=20) |"


def test_legacy_huge_integer_count_does_not_overflow_renderer():
    huge = 10**400
    pair = _pair("gpqa-diamond", "legacy", score=0.5).model_copy(
        update={"metrics": {"score": 0.5, "n_total": huge}}
    )

    text = render_markdown(_board([pair], ["legacy"]))

    assert f"50.0 (n={huge})" in text


def test_huge_primary_score_is_rendered_as_failed_evidence_without_overflow():
    pair = _pair("gpqa-diamond", "malformed").model_copy(
        update={"metrics": {"score": 10**4000, "n_total": 1}}
    )

    assert pair.score is None
    board = _board([pair], ["malformed"])
    cell = board["cells"]["gpqa-diamond"]["malformed"]

    assert cell["status"] == "failed"
    assert cell["score"] is None
    assert cell["n"] == 0
    assert "invalid pair evidence" in cell["reason"]
    assert "n/a[^" in render_markdown(board)


@pytest.mark.parametrize("score", [True, "0.5"], ids=["boolean", "numeric-string"])
def test_pair_metrics_do_not_coerce_non_numeric_score_evidence(score):
    with pytest.raises(ValueError, match="metrics.score"):
        PairResult(
            benchmark="gpqa-diamond",
            target="malformed",
            status="completed",
            metrics={"score": score, "n_total": 1},
        )


def test_markdown_layout():
    pairs = [
        _pair("gpqa-diamond", "m", score=0.955),
        _pair("gpqa-diamond", "auto", status="partial", score=0.5, reason="2/4 unjudged"),
    ]
    text = render_markdown(_board(pairs, ["m", "auto"]))
    assert "| Benchmark | m | auto |" in text
    assert "| GPQA Diamond | 95.5 (n=3) |" in text
    assert "50.0*" in text  # partial marker
    assert "[^1]:" in text  # footnote body present


def test_markdown_skip_cell_is_dash():
    pairs = [_pair("gpqa-diamond", "m", status="skipped", score=None, reason="docker unavailable")]
    text = render_markdown(_board(pairs, ["m"]))
    assert "—[^1]" in text
    assert "docker unavailable" in text
