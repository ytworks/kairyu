"""BenchConfig assembly from CLI args and bench.yaml."""

import argparse

import pytest

from kairyu.bench.cli import add_bench_parser
from kairyu.bench.config import build_config, build_judge_config, parse_target_flag
from kairyu.bench.execution import build_execution_runner
from kairyu.bench.types import ExecutionConfig, JudgeConfig, JudgeEndpointConfig


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


def test_repeated_secondary_judge_flags_build_an_ordered_panel():
    args = _parse(
        [
            "run",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--judge-base-url",
            "http://judge:8000",
            "--judge-model",
            "primary",
            "--judge-secondary",
            "http://judge:8000=second=SECOND_KEY",
            "--judge-secondary",
            "http://other:9000/v1=third",
        ]
    )
    config = build_config(args)
    assert [member.model for member in config.judge.additional_judges] == [
        "second",
        "third",
    ]
    assert config.judge.additional_judges[0].api_key_env == "SECOND_KEY"
    assert config.judge.additional_judges[1].base_url == "http://other:9000/v1"


def test_judge_panel_rejects_missing_primary_partial_or_duplicate_members():
    second = JudgeEndpointConfig(base_url="http://judge", model="second")
    with pytest.raises(ValueError, match="required when additional"):
        JudgeConfig(additional_judges=(second,))
    with pytest.raises(ValueError, match="requires both"):
        JudgeConfig(
            base_url="http://judge",
            model="primary",
            additional_judges=(JudgeEndpointConfig(base_url="http://other"),),
        )
    with pytest.raises(ValueError, match="distinct"):
        JudgeConfig(
            base_url="http://judge/",
            model="same",
            additional_judges=(
                JudgeEndpointConfig(base_url="http://judge/v1", model="same"),
            ),
        )
    with pytest.raises(ValueError, match="model must be non-empty"):
        JudgeConfig(base_url="http://judge", model=" ")


def test_calibration_subcommand_loads_a_judge_block_without_targets(tmp_path):
    path = tmp_path / "judge.yaml"
    path.write_text(
        "judge:\n"
        "  base_url: http://primary\n"
        "  model: primary\n"
        "  additional_judges:\n"
        "    - {base_url: http://secondary, model: secondary}\n",
        encoding="utf-8",
    )
    args = _parse(["calibrate-judge", "--config", str(path)])
    config = build_judge_config(args)
    assert config.model == "primary"
    assert config.additional_judges[0].model == "secondary"
    assert args.min_agreement == 11 / 12
    assert args.max_position_flip == 0.0


def test_calibration_subcommand_accepts_a_bound_benchmark_run():
    args = _parse(
        [
            "calibrate-judge",
            "--run",
            "headline-run",
            "--results-dir",
            "/results/fugu",
        ]
    )
    assert args.run == "headline-run"
    assert args.results_dir == "/results/fugu"


def test_execution_config_requires_an_immutable_docker_image():
    digest = "a" * 64
    local = ExecutionConfig()
    assert local.runner == "local"
    assert local.image is None
    assert local.cpus == 1.0
    assert local.pids_limit == 64
    assert local.disk_mb == 256
    assert local.pull_policy == "never"

    for image in (f"sha256:{digest}", f"registry.example/kairyu@sha256:{digest}"):
        config = ExecutionConfig(runner="docker", image=image)
        assert config.image == image

    with pytest.raises(ValueError, match="image is required"):
        ExecutionConfig(runner="docker")
    with pytest.raises(ValueError, match="immutable"):
        ExecutionConfig(runner="docker", image="kairyu-bench:latest")
    with pytest.raises(ValueError, match="only for runner='docker'"):
        ExecutionConfig(image=f"sha256:{digest}")
    with pytest.raises(ValueError, match="resource settings"):
        ExecutionConfig(cpus=2.0)
    with pytest.raises(ValueError, match="finite positive"):
        ExecutionConfig(cpus=True)
    with pytest.raises(ValueError):
        ExecutionConfig(runner="docker", image=f"sha256:{digest}", cpus=float("inf"))
    with pytest.raises(ValueError, match="repository is invalid"):
        ExecutionConfig(runner="docker", image=f"-repo@sha256:{digest}")
    with pytest.raises(ValueError, match="resource settings"):
        build_execution_runner({"runner": "local", "cpus": 2.0})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpus", 0),
        ("pids_limit", 1),
        ("disk_mb", 0),
        ("disk_mb", 65_537),
        ("pull_policy", "always"),
    ],
)
def test_execution_config_rejects_unsafe_resource_policy(field, value):
    with pytest.raises(ValueError):
        ExecutionConfig(**{field: value})


def test_execution_cli_is_explicit_and_overrides_yaml(tmp_path):
    digest = "b" * 64
    config_path = tmp_path / "bench.yaml"
    config_path.write_text(
        f"""
targets:
  - {{ base_url: "http://yaml:8000", model: yaml-model }}
execution:
  runner: docker
  image: sha256:{digest}
  cpus: 1
  pids_limit: 32
  disk_mb: 128
""",
        encoding="utf-8",
    )
    args = _parse(
        [
            "run",
            "--config",
            str(config_path),
            "--exec-cpus",
            "2.5",
            "--exec-pids-limit",
            "48",
            "--exec-disk-mb",
            "512",
        ]
    )
    config = build_config(args)
    assert config.execution.runner == "docker"
    assert config.execution.image == f"sha256:{digest}"
    assert config.execution.cpus == 2.5
    assert config.execution.pids_limit == 48
    assert config.execution.disk_mb == 512

    local = build_config(
        _parse(["run", "--config", str(config_path), "--exec-runner", "local"])
    )
    assert local.execution == ExecutionConfig()
