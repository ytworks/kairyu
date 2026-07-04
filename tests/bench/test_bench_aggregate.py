"""Scoreboard aggregation: Fugu row order, cell rendering, footnotes."""

from kairyu.bench.aggregate import build_scoreboard, render_markdown
from kairyu.bench.types import PairResult


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


def _board(pairs, targets, config=None):
    return build_scoreboard(
        run_id="run-1",
        suite="fugu",
        config=config or {},
        environment={},
        pairs=pairs,
        targets=targets,
    )


def test_self_judged_target_is_flagged():
    # a target graded by a judge that IS that target is flagged as biased.
    board = _board(
        [_pair("gpqa-diamond", "kairyu-auto")],
        targets=["kairyu-auto", "gpt-5"],
        config={"judge": {"model": "kairyu-auto"}},
    )
    assert board["self_judged_targets"] == ["kairyu-auto"]
    cell = board["cells"]["gpqa-diamond"]["kairyu-auto"]
    assert any("self-judged" in board["footnotes"][n - 1] for n in cell["footnotes"])


def test_rows_follow_fugu_order_and_only_present_benchmarks():
    pairs = [
        _pair("gpqa-diamond", "m"),
        _pair("mrcr-v2", "m"),  # later Fugu row than gpqa
    ]
    board = _board(pairs, ["m"])
    assert board["benchmarks"] == ["gpqa-diamond", "mrcr-v2"]


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


def test_markdown_layout():
    pairs = [
        _pair("gpqa-diamond", "m", score=0.955),
        _pair("gpqa-diamond", "auto", status="partial", score=0.5, reason="2/4 unjudged"),
    ]
    text = render_markdown(_board(pairs, ["m", "auto"]))
    assert "| Benchmark | m | auto |" in text
    assert "| GPQA Diamond | 95.5 |" in text
    assert "50.0*" in text  # partial marker
    assert "[^1]:" in text  # footnote body present


def test_markdown_skip_cell_is_dash():
    pairs = [
        _pair("gpqa-diamond", "m", status="skipped", score=None, reason="docker unavailable")
    ]
    text = render_markdown(_board(pairs, ["m"]))
    assert "—[^1]" in text
    assert "docker unavailable" in text
