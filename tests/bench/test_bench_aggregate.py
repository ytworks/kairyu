"""Scoreboard aggregation: Fugu row order, cell rendering, footnotes."""

import json
from argparse import Namespace

import pytest

from kairyu.bench.aggregate import build_scoreboard, render_markdown
from kairyu.bench.cli import _handle_report
from kairyu.bench.types import BenchTarget, JudgeConfig, PairResult


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


def _board(pairs, targets, config=None, target_configs=None, judge=None):
    return build_scoreboard(
        run_id="run-1",
        suite="fugu",
        config=config or {},
        environment={},
        pairs=pairs,
        targets=targets,
        target_configs=target_configs,
        judge=judge,
    )


def test_self_judged_target_is_flagged():
    # Display aliases differ, but resolved URL/model identity is the same.
    target = BenchTarget(
        name="friendly-target",
        base_url="http://gateway.test/v1/",
        model="shared-model",
    )
    judge = JudgeConfig(
        base_url="http://gateway.test/v1", model="shared-model"
    )
    board = _board(
        [_pair("gpqa-diamond", "friendly-target")],
        targets=["friendly-target"],
        config={"judge": judge.model_dump()},
        target_configs=[target],
        judge=judge,
    )
    assert board["self_judged_targets"] == ["friendly-target"]
    cell = board["cells"]["gpqa-diamond"]["friendly-target"]
    assert any("self-judged" in board["footnotes"][n - 1] for n in cell["footnotes"])


def test_self_judged_target_normalizes_default_openai_v1_path():
    target = BenchTarget(
        name="friendly-target",
        base_url="http://gateway.test:8000",
        model="shared-model",
    )
    judge = JudgeConfig(
        base_url="http://gateway.test:8000/v1",
        model="shared-model",
    )

    board = _board(
        [_pair("gpqa-diamond", target.label())],
        targets=[target.label()],
        target_configs=[target],
        judge=judge,
    )

    assert board["self_judged_targets"] == ["friendly-target"]
    assert board["judge_independence_unknown_targets"] == []


@pytest.mark.parametrize(
    ("judge_base_url", "judge_model"),
    [
        pytest.param(
            "http://judge.test:8443/proxy/v1", "judge-model", id="scheme"
        ),
        pytest.param(
            "https://other.test:8443/proxy/v1", "judge-model", id="host"
        ),
        pytest.param(
            "https://judge.test:9443/proxy/v1", "judge-model", id="port"
        ),
        pytest.param(
            "https://judge.test:8443/other/v1", "judge-model", id="path"
        ),
        pytest.param(
            "https://judge.test:8443/proxy/v1", "other-model", id="model"
        ),
    ],
)
def test_self_judge_identity_keeps_other_endpoint_parts_strict(
    judge_base_url, judge_model
):
    target = BenchTarget(
        name="target",
        base_url="https://judge.test:8443/proxy",
        model="judge-model",
    )
    judge = JudgeConfig(base_url=judge_base_url, model=judge_model)

    board = _board(
        [_pair("gpqa-diamond", target.label())],
        targets=[target.label()],
        target_configs=[target],
        judge=judge,
    )

    assert board["self_judged_targets"] == []
    assert board["judge_independence_unknown_targets"] == []


def test_distinct_resolved_judge_identities_are_not_flagged():
    targets = [
        BenchTarget(
            name="different-endpoint",
            base_url="http://other.test/v1",
            model="judge-model",
        ),
        BenchTarget(
            name="different-model",
            base_url="http://judge.test/v1/",
            model="target-model",
        ),
    ]
    judge = JudgeConfig(base_url="http://judge.test/v1", model="judge-model")
    labels = [target.label() for target in targets]
    board = _board(
        [_pair("gpqa-diamond", label) for label in labels],
        labels,
        target_configs=targets,
        judge=judge,
    )

    assert board["self_judged_targets"] == []
    assert board["judge_independence_unknown_targets"] == []


def test_missing_legacy_target_identity_is_annotated_unknown():
    label = "legacy-alias"
    judge = JudgeConfig(base_url="http://judge.test/v1", model="judge-model")
    board = _board(
        [_pair("gpqa-diamond", label)],
        [label],
        config={"judge": judge.model_dump()},
        target_configs=[],
        judge=judge,
    )

    assert board["self_judged_targets"] == []
    assert board["judge_independence_unknown_targets"] == [label]
    cell = board["cells"]["gpqa-diamond"][label]
    assert any(
        "independence unknown" in board["footnotes"][number - 1]
        for number in cell["footnotes"]
    )


_DEFAULT_REPORT_JUDGE = object()


def _write_report_fixture(
    tmp_path, *, target_config, judge_config=_DEFAULT_REPORT_JUDGE
):
    run_dir = tmp_path / "report-run"
    pair_dir = run_dir / "pair"
    pair_dir.mkdir(parents=True)
    if judge_config is _DEFAULT_REPORT_JUDGE:
        judge_config = {
            "base_url": "http://judge.test/v1",
            "model": "judge-model",
        }
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "report-run",
                "config": {
                    "suite": "fugu",
                    "targets": [target_config],
                    "judge": judge_config,
                },
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    label = target_config["name"]
    (pair_dir / "result.json").write_text(
        _pair("gpqa-diamond", label).model_dump_json(), encoding="utf-8"
    )
    args = Namespace(run=str(run_dir), results_dir=str(tmp_path / "unused"))
    assert _handle_report(args) == 0
    return json.loads((run_dir / "scoreboard.json").read_text(encoding="utf-8"))


def test_report_reconstructs_resolved_target_and_judge_identity(tmp_path):
    board = _write_report_fixture(
        tmp_path,
        target_config={
            "name": "alias",
            "base_url": "http://judge.test/v1/",
            "model": "judge-model",
        },
    )

    assert board["self_judged_targets"] == ["alias"]
    assert board["judge_independence_unknown_targets"] == []


def test_legacy_report_without_target_endpoint_fails_closed(tmp_path):
    board = _write_report_fixture(
        tmp_path,
        target_config={"name": "legacy-alias", "model": "judge-model"},
    )

    assert board["self_judged_targets"] == []
    assert board["judge_independence_unknown_targets"] == ["legacy-alias"]


@pytest.mark.parametrize(
    "judge_config",
    [
        pytest.param(None, id="null"),
        pytest.param(JudgeConfig().model_dump(mode="json"), id="serialized-disabled"),
    ],
)
def test_report_does_not_annotate_explicitly_disabled_judge(
    tmp_path, judge_config
):
    board = _write_report_fixture(
        tmp_path,
        target_config={
            "name": "plain-target",
            "base_url": "http://target.test/v1",
            "model": "target-model",
        },
        judge_config=judge_config,
    )

    assert board["self_judged_targets"] == []
    assert board["judge_independence_unknown_targets"] == []


@pytest.mark.parametrize(
    "judge_config",
    [
        pytest.param({"model": "judge-model"}, id="missing-base-url"),
        pytest.param(
            {"base_url": "http://judge.test/v1"}, id="missing-model"
        ),
        pytest.param({"api_key_env": "LEGACY_JUDGE_KEY"}, id="missing-both-keys"),
    ],
)
def test_legacy_report_without_complete_judge_identity_fails_closed(
    tmp_path, judge_config
):
    board = _write_report_fixture(
        tmp_path,
        target_config={
            "name": "legacy-target",
            "base_url": "http://target.test/v1",
            "model": "target-model",
        },
        judge_config=judge_config,
    )

    assert board["self_judged_targets"] == []
    assert board["judge_independence_unknown_targets"] == ["legacy-target"]


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
