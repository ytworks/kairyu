"""Public ``kairyu bench quant-sweep`` dispatch and exit semantics."""

from __future__ import annotations

import json

import pytest

from kairyu.bench import quant_sweep
from kairyu.bench.adapters import QUANTIZATION_ROW_ORDER
from kairyu.bench.cli import handle
from kairyu.bench.store import ResultStore
from kairyu.entrypoints import cli


def _args(*extra: str):
    return cli._build_parser().parse_args(
        [
            "bench",
            "quant-sweep",
            "--run",
            "quant-run",
            "--results-dir",
            "/tmp/quant-sweep-tests",
            "--tolerance",
            "gsm8k=1",
            "--tolerance",
            "mmlu=2",
            "--tolerance",
            "ifeval=3",
            "--tolerance",
            "gpqa-diamond=4",
            *extra,
        ]
    )


def _wire(monkeypatch, *, verdict: str = "pass") -> list[tuple[dict, str]]:
    targets = ["bf16", "fp8", "int8", "awq", "gptq", "nvfp4", "fp8-kv"]
    entry = {
        "run_id": "quant-run",
        "suite": "quantization",
        "scoreboard": {
            "targets": targets,
            "benchmarks": list(QUANTIZATION_ROW_ORDER),
        },
    }
    monkeypatch.setattr(ResultStore, "load_scoreboard_index", lambda _self: (entry,))
    loaded: list[tuple[str, str]] = []

    def load(_self, _entry, benchmark, target):
        loaded.append((benchmark, target))
        return (benchmark, target)

    monkeypatch.setattr(ResultStore, "load_indexed_pair", load, raising=False)

    def build(indexed, *, pairs, tolerances):
        assert indexed is entry
        assert tolerances == {
            "gsm8k": 0.01,
            "mmlu": 0.02,
            "ifeval": 0.03,
            "gpqa-diamond": 0.04,
        }
        assert set(pairs) == set(targets)
        assert all(list(cells) == list(QUANTIZATION_ROW_ORDER) for cells in pairs.values())
        assert len(loaded) == len(targets) * len(QUANTIZATION_ROW_ORDER)
        return {
            "sweep_type": "quantization_task_accuracy",
            "overall_verdict": verdict,
        }

    monkeypatch.setattr(quant_sweep, "build_quantization_sweep", build)
    monkeypatch.setattr(
        quant_sweep,
        "render_quantization_sweep_markdown",
        lambda sweep: f"QUANT SWEEP {sweep['overall_verdict']}",
    )
    saved: list[tuple[dict, str]] = []
    monkeypatch.setattr(
        ResultStore,
        "save_quantization_sweep",
        lambda _self, sweep, markdown: saved.append((sweep, markdown)),
        raising=False,
    )
    return saved


def test_pass_saves_artifact_then_exits_zero(monkeypatch, capsys):
    saved = _wire(monkeypatch)

    assert handle(_args()) == 0

    assert saved == [
        (
            {
                "sweep_type": "quantization_task_accuracy",
                "overall_verdict": "pass",
            },
            "QUANT SWEEP pass",
        )
    ]
    assert capsys.readouterr().out == "QUANT SWEEP pass\n"


def test_quality_failure_is_saved_before_exit_one(monkeypatch, capsys):
    saved = _wire(monkeypatch, verdict="fail")

    assert handle(_args()) == 1

    assert len(saved) == 1
    assert saved[0][0]["overall_verdict"] == "fail"
    assert capsys.readouterr().out == "QUANT SWEEP fail\n"


def test_json_mode_prints_machine_artifact(monkeypatch, capsys):
    _wire(monkeypatch)

    assert handle(_args("--json")) == 0

    assert json.loads(capsys.readouterr().out)["sweep_type"] == (
        "quantization_task_accuracy"
    )


def test_invalid_builder_does_not_save_artifact(monkeypatch, capsys):
    saved = _wire(monkeypatch)
    monkeypatch.setattr(
        quant_sweep,
        "build_quantization_sweep",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid evidence")),
    )

    assert handle(_args()) == 2

    assert saved == []
    assert capsys.readouterr().out == "quant-sweep: invalid evidence\n"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        pytest.param(("--tolerance", "gsm8k=2"), "duplicate", id="duplicate"),
        pytest.param(("--tolerance", "other=nan"), "finite", id="nan"),
        pytest.param(("--tolerance", "other=-1"), "finite", id="negative"),
    ],
)
def test_invalid_tolerance_is_input_error_two(monkeypatch, capsys, extra, message):
    saved = _wire(monkeypatch)

    assert handle(_args(*extra)) == 2

    assert saved == []
    output = capsys.readouterr().out
    assert output.startswith("quant-sweep:")
    assert message in output


def test_non_quantization_run_is_rejected_before_pair_reads(monkeypatch, capsys):
    entry = {
        "run_id": "quant-run",
        "suite": "core",
        "scoreboard": {
            "targets": ["m"],
            "benchmarks": list(QUANTIZATION_ROW_ORDER),
        },
    }
    monkeypatch.setattr(ResultStore, "load_scoreboard_index", lambda _self: (entry,))
    monkeypatch.setattr(
        ResultStore,
        "load_indexed_pair",
        lambda *_args: (_ for _ in ()).throw(AssertionError("pair read")),
    )

    assert handle(_args()) == 2

    assert "does not belong to suite" in capsys.readouterr().out
