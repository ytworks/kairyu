"""BenchConfig assembly from CLI args and bench.yaml."""

import argparse

import pytest

from kairyu.bench.cli import add_bench_parser
from kairyu.bench.config import build_config, parse_target_flag


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_bench_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["bench", *argv])


def test_parse_target_flag():
    target = parse_target_flag("gw=http://gw:8000/v1=kairyu-auto=MY_KEY")
    assert target.name == "gw"
    assert target.api_key_env == "MY_KEY"
    with pytest.raises(ValueError, match="expected name=base_url=model"):
        parse_target_flag("just-a-name")


def test_models_shorthand_builds_targets():
    args = _parse(
        ["run", "--base-url", "http://gw:8000", "--model", "m", "--model", "kairyu-auto"]
    )
    config = build_config(args)
    assert [t.model for t in config.targets] == ["m", "kairyu-auto"]
    assert config.targets[0].label() == "m"
    assert config.suite == "fugu"
    assert config.limit is None  # full run is the default


def test_model_without_base_url_rejected():
    args = _parse(["run", "--model", "m"])
    with pytest.raises(ValueError, match="--model requires --base-url"):
        build_config(args)


def test_cli_overrides_yaml(tmp_path):
    (tmp_path / "bench.yaml").write_text(
        """
targets:
  - { base_url: "http://yaml:8000", model: yaml-model }
limit: 5
seed: 7
judge: { base_url: "http://judge:8000", model: judge-m }
""",
        encoding="utf-8",
    )
    args = _parse(
        [
            "run",
            "--config",
            str(tmp_path / "bench.yaml"),
            "--limit",
            "3",
            "--smoke",
            "--only",
            "gpqa-diamond,mrcr-v2",
        ]
    )
    config = build_config(args)
    assert config.targets[0].model == "yaml-model"  # from YAML
    assert config.limit == 3  # CLI wins
    assert config.seed == 7  # YAML survives
    assert config.smoke is True
    assert config.only == ("gpqa-diamond", "mrcr-v2")
    assert config.judge.enabled


def test_no_targets_anywhere_rejected():
    args = _parse(["run"])
    with pytest.raises(ValueError):
        build_config(args)


def test_judge_flags_enable_judge():
    args = _parse(
        [
            "run",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--judge-base-url",
            "http://gw:8000",
            "--judge-model",
            "kairyu-auto",
        ]
    )
    config = build_config(args)
    assert config.judge.enabled
    assert config.judge.model == "kairyu-auto"
