"""Public ``kairyu bench compare`` dispatch, artifacts, and exit semantics."""

from __future__ import annotations

import json

import pytest

from kairyu.bench import config_ab
from kairyu.bench.cli import handle
from kairyu.bench.store import ResultStore
from kairyu.entrypoints import cli


def _entry(run_id: str, targets: tuple[str, ...] = ("arm",)) -> dict:
    return {
        "run_id": run_id,
        "suite": "core",
        "scoreboard": {"targets": list(targets)},
    }


def _args(*extra: str):
    return cli._build_parser().parse_args(
        [
            "bench",
            "compare",
            "--baseline",
            "base",
            "--candidate",
            "candidate",
            "--suite",
            "core",
            "--results-dir",
            "/tmp/config-ab-tests",
            "--tolerance",
            "gsm8k=1.25",
            *extra,
        ]
    )


def _wire(monkeypatch, *, verdict: str = "pass") -> list[tuple[dict, str]]:
    entries = (_entry("base"), _entry("candidate"))
    monkeypatch.setattr(ResultStore, "load_scoreboard_index", lambda _self: entries)
    monkeypatch.setattr(
        ResultStore,
        "load_indexed_pair",
        lambda self, entry, benchmark, target: (
            self.run_id,
            entry["run_id"],
            benchmark,
            target,
        ),
        raising=False,
    )

    def build(baseline, candidate, **kwargs):
        assert baseline["run_id"] == "base"
        assert candidate["run_id"] == "candidate"
        assert kwargs["tolerances"] == {"gsm8k": 0.0125}
        assert kwargs["baseline_pairs"]["gsm8k"][0] == "base"
        assert kwargs["candidate_pairs"]["gsm8k"][0] == "candidate"
        return {
            "comparison_type": "config_ab",
            "overall_verdict": verdict,
            "benchmarks": [],
        }

    monkeypatch.setattr(config_ab, "build_config_comparison", build)
    monkeypatch.setattr(
        config_ab,
        "render_config_comparison_markdown",
        lambda comparison: f"CONFIG A/B {comparison['overall_verdict']}",
    )
    saved: list[tuple[dict, str]] = []
    monkeypatch.setattr(
        ResultStore,
        "save_config_comparison",
        lambda _self, comparison, markdown: saved.append((comparison, markdown)),
        raising=False,
    )
    return saved


def test_compare_pass_saves_artifact_then_exits_zero(monkeypatch, capsys):
    saved = _wire(monkeypatch, verdict="pass")

    assert handle(_args()) == 0

    assert saved == [
        (
            {
                "comparison_type": "config_ab",
                "overall_verdict": "pass",
                "benchmarks": [],
            },
            "CONFIG A/B pass",
        )
    ]
    assert capsys.readouterr().out == "CONFIG A/B pass\n"


def test_compare_quality_failure_is_persisted_before_exit_one(monkeypatch, capsys):
    saved = _wire(monkeypatch, verdict="fail")

    assert handle(_args()) == 1

    assert len(saved) == 1
    assert saved[0][0]["overall_verdict"] == "fail"
    assert "CONFIG A/B fail" in capsys.readouterr().out


def test_compare_json_prints_machine_artifact(monkeypatch, capsys):
    _wire(monkeypatch)

    assert handle(_args("--json")) == 0

    assert json.loads(capsys.readouterr().out)["comparison_type"] == "config_ab"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        pytest.param(("--tolerance", "gsm8k=2"), "duplicate", id="duplicate"),
        pytest.param(("--tolerance", "bad"), "BENCHMARK=PP", id="syntax"),
        pytest.param(("--tolerance", "mmlu=nan"), "finite", id="nan"),
        pytest.param(("--tolerance", "mmlu=-1"), "finite", id="negative"),
        pytest.param(("--tolerance", "mmlu=101"), "finite", id="over-100"),
    ],
)
def test_invalid_tolerance_is_input_error_two(monkeypatch, capsys, extra, message):
    saved = _wire(monkeypatch)

    assert handle(_args(*extra)) == 2

    assert saved == []
    output = capsys.readouterr().out
    assert output.startswith("compare:")
    assert message in output


def test_multiple_target_run_requires_explicit_arm_selection(monkeypatch, capsys):
    entries = (_entry("base", ("a", "b")), _entry("candidate"))
    monkeypatch.setattr(ResultStore, "load_scoreboard_index", lambda _self: entries)

    assert handle(_args()) == 2

    assert "select --baseline-target" in capsys.readouterr().out


def test_explicit_targets_support_two_arms_from_the_same_run(monkeypatch, capsys):
    entry = _entry("base", ("bf16", "fp8"))
    monkeypatch.setattr(ResultStore, "load_scoreboard_index", lambda _self: (entry,))
    loaded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ResultStore,
        "load_indexed_pair",
        lambda _self, _entry, benchmark, target: loaded.append((benchmark, target)) or target,
        raising=False,
    )
    monkeypatch.setattr(
        config_ab,
        "build_config_comparison",
        lambda *_args, **_kwargs: {
            "comparison_type": "config_ab",
            "overall_verdict": "pass",
            "benchmarks": [],
        },
    )
    monkeypatch.setattr(config_ab, "render_config_comparison_markdown", lambda _value: "ok")
    monkeypatch.setattr(
        ResultStore,
        "save_config_comparison",
        lambda *_args: None,
        raising=False,
    )
    args = cli._build_parser().parse_args(
        [
            "bench",
            "compare",
            "--baseline",
            "base",
            "--candidate",
            "base",
            "--baseline-target",
            "bf16",
            "--candidate-target",
            "fp8",
            "--suite",
            "core",
            "--results-dir",
            "/tmp/config-ab-tests",
            "--tolerance",
            "gsm8k=1",
        ]
    )

    assert handle(args) == 0
    assert loaded == [("gsm8k", "bf16"), ("gsm8k", "fp8")]
    assert capsys.readouterr().out == "ok\n"
