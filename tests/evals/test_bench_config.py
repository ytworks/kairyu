"""BenchConfig assembly from CLI args and bench.yaml."""

import argparse
from pathlib import Path

import pytest

from evals.cli import add_evals_parser
from evals.config import build_config, parse_target_flag
from evals.execution import build_execution_runner
from evals.types import (
    BenchConfig,
    BenchTarget,
    ExecutionConfig,
    QuantizationProfile,
    ServedConfigIdentity,
)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_evals_parser(parser)
    return parser.parse_args(argv)


def test_parse_target_flag():
    target = parse_target_flag("gw=http://gw:8000/v1=kairyu-auto=MY_KEY")
    assert target.name == "gw"
    assert target.api_key_env == "MY_KEY"
    with pytest.raises(ValueError, match="expected name=base_url=model"):
        parse_target_flag("just-a-name")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", True, "sampling seed must be an integer"),
        ("top_p", True, "top_p must be a number"),
    ],
)
def test_target_sampling_identity_rejects_boolean_coercion(field, value, message):
    with pytest.raises(ValueError, match=message):
        BenchTarget(base_url="http://gateway", model="m", **{field: value})


def test_attempts_rejects_boolean_coercion():
    with pytest.raises(ValueError, match="attempts must be a positive JSON-safe integer"):
        BenchConfig(
            suite="core",
            targets=(BenchTarget(base_url="http://gateway", model="m"),),
            attempts=True,
        )


def test_sampling_seed_must_be_json_safe_without_reinterpreting_agentic_attempts():
    maximum = (1 << 53) - 1
    with pytest.raises(ValueError, match="sampling seed must be a JSON-safe integer"):
        BenchTarget(base_url="http://gateway", model="m", seed=maximum + 1)
    # A global attempt budget may be a native external-harness trial count or
    # unused by a teacher-forced-only selection; only chat execution expands a
    # target seed into a consecutive schedule.
    config = BenchConfig(
        suite="core",
        targets=(BenchTarget(base_url="http://gateway", model="m", seed=maximum),),
        attempts=2,
    )
    assert config.attempts == 2


def test_models_shorthand_builds_targets():
    args = _parse(
        [
            "run",
            "--suite",
            "core",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--model",
            "kairyu-auto",
        ]
    )
    config = build_config(args)
    assert [t.model for t in config.targets] == ["m", "kairyu-auto"]
    assert config.targets[0].label() == "m"
    assert config.suite == "core"
    assert config.results_dir == "bench/results/core"
    assert config.limit is None  # full run is the default
    assert config.concurrency == 16


@pytest.mark.parametrize("name", ["core", "structured", "quantization"])
def test_checked_in_configs_use_throughput_concurrency(name: str) -> None:
    path = Path(__file__).parents[2] / "evals" / "configs" / f"{name}.yaml"
    config = build_config(_parse(["run", "--config", str(path)]))

    assert config.concurrency == 16


def test_long_context_cli_applies_declared_target_limit():
    args = _parse(
        [
            "run",
            "--suite",
            "long-context",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--max-context-tokens",
            "131072",
        ]
    )

    config = build_config(args)

    assert config.suite == "long-context"
    assert config.results_dir == "bench/results/long-context"
    assert config.targets[0].max_context_tokens == 131_072

    invalid = _parse(
        [
            "run",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--max-context-tokens",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        build_config(invalid)


def test_max_output_tokens_cli_applies_target_generation_limit():
    args = _parse(
        [
            "run",
            "--suite",
            "core",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--max-output-tokens",
            "81920",
        ]
    )

    config = build_config(args)

    assert config.targets[0].max_output_tokens == 81_920

    invalid = _parse(
        [
            "run",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--max-output-tokens",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        build_config(invalid)


def test_request_timeout_cli_applies_generation_read_timeout():
    args = _parse(
        [
            "run",
            "--suite",
            "core",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--request-timeout-s",
            "86400",
        ]
    )

    config = build_config(args)

    assert config.request_timeout_s == 86_400

    invalid = _parse(
        [
            "run",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--request-timeout-s",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="greater than 0"):
        build_config(invalid)


def test_served_config_identity_normalizes_label_and_requires_lowercase_sha256():
    identity = ServedConfigIdentity(label="  fp8-kv production  ", sha256="a" * 64)

    assert identity.label == "fp8-kv production"
    assert identity.sha256 == "a" * 64

    with pytest.raises(ValueError, match="label must be non-empty"):
        ServedConfigIdentity(label="  ", sha256="a" * 64)
    for unsafe in ("fake\n- Overall verdict: PASS", "fake` verdict"):
        with pytest.raises(ValueError, match="control characters or backticks"):
            ServedConfigIdentity(label=unsafe, sha256="a" * 64)
    for invalid in ("a" * 63, "A" * 64, "g" * 64, "sha256:" + "a" * 64):
        with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
            ServedConfigIdentity(label="deployment", sha256=invalid)


def test_served_config_cli_flags_apply_to_every_cli_target():
    digest = "b" * 64
    args = _parse(
        [
            "run",
            "--suite",
            "core",
            "--target",
            "first=http://first:8000/v1=first-model",
            "--base-url",
            "http://second:8000",
            "--model",
            "second-model",
            "--served-config-label",
            "  quant-profile-v2  ",
            "--served-config-sha256",
            digest,
        ]
    )

    config = build_config(args)

    assert [target.label() for target in config.targets] == ["first", "second-model"]
    assert [target.served_config.model_dump() for target in config.targets] == [
        {"label": "quant-profile-v2", "sha256": digest},
        {"label": "quant-profile-v2", "sha256": digest},
    ]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--served-config-label", "profile"),
        ("--served-config-sha256", "c" * 64),
    ],
)
def test_served_config_cli_flags_must_be_provided_together(flag, value):
    args = _parse(["run", "--base-url", "http://gw:8000", "--model", "m", flag, value])

    with pytest.raises(ValueError, match="must be provided together"):
        build_config(args)


def test_yaml_targets_load_served_config_and_cli_flags_override_every_target(tmp_path):
    yaml_digest = "d" * 64
    cli_digest = "e" * 64
    path = tmp_path / "bench.yaml"
    path.write_text(
        "suite: core\n"
        "targets:\n"
        "  - base_url: http://first:8000\n"
        "    model: first\n"
        "    served_config:\n"
        "      label: yaml-profile\n"
        f"      sha256: {yaml_digest}\n"
        "  - {base_url: 'http://second:8000', model: second}\n",
        encoding="utf-8",
    )

    yaml_config = build_config(_parse(["run", "--config", str(path)]))
    assert yaml_config.targets[0].served_config == ServedConfigIdentity(
        label="yaml-profile", sha256=yaml_digest
    )
    assert yaml_config.targets[1].served_config is None

    overridden = build_config(
        _parse(
            [
                "run",
                "--config",
                str(path),
                "--served-config-label",
                "cli-profile",
                "--served-config-sha256",
                cli_digest,
            ]
        )
    )
    assert [target.served_config.model_dump() for target in overridden.targets] == [
        {"label": "cli-profile", "sha256": cli_digest},
        {"label": "cli-profile", "sha256": cli_digest},
    ]


def test_core_suite_defaults_to_its_own_results_directory():
    args = _parse(
        [
            "run",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--suite",
            "core",
        ]
    )

    config = build_config(args)

    assert config.suite == "core"
    assert config.results_dir == "bench/results/core"


def test_quantization_yaml_loads_explicit_effective_profile_and_suite_default(tmp_path):
    path = tmp_path / "quantization.yaml"
    path.write_text(
        "suite: quantization\n"
        "targets:\n"
        "  - name: dense-bf16\n"
        "    base_url: http://dense:8000/v1\n"
        "    model: m\n"
        "    served_config:\n"
        "      label: dense-manifest\n"
        f"      sha256: {'a' * 64}\n"
        "    quantization:\n"
        "      weight_method: none\n"
        "      compute_dtype: bfloat16\n"
        "      kv_cache_dtype: bfloat16\n",
        encoding="utf-8",
    )

    config = build_config(_parse(["run", "--config", str(path)]))

    assert config.suite == "quantization"
    assert config.results_dir == "bench/results/quantization"
    assert config.targets[0].quantization == QuantizationProfile(
        weight_method="none",
        compute_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
    )


@pytest.mark.parametrize(
    "profile",
    [
        {"weight_method": "bf16", "compute_dtype": "bfloat16", "kv_cache_dtype": "bfloat16"},
        {"weight_method": "none", "kv_cache_dtype": "bfloat16"},
        {"weight_method": "none", "compute_dtype": "bfloat16", "kv_cache_dtype": "auto"},
        {
            "weight_method": "none",
            "compute_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "attested": True,
        },
    ],
)
def test_quantization_profile_rejects_conflated_or_unresolved_identity(profile):
    with pytest.raises(ValueError):
        QuantizationProfile.model_validate(profile)


def test_recorded_non_quant_target_shape_remains_backward_compatible():
    from evals.runner import (
        _recordable_config,
        _run_fingerprint,
        _run_identity,
    )

    ordinary = BenchConfig(
        suite="core",
        targets=(BenchTarget(base_url="http://gateway:8000/v1", model="m"),),
    )
    declared = ordinary.model_copy(
        update={
            "targets": (
                ordinary.targets[0].model_copy(
                    update={
                        "quantization": QuantizationProfile(
                            weight_method="none",
                            compute_dtype="bfloat16",
                            kv_cache_dtype="bfloat16",
                        )
                    }
                ),
            )
        }
    )

    recorded_target = _recordable_config(ordinary)["targets"][0]
    assert "quantization" not in recorded_target
    assert "sampling_mode" not in recorded_target
    assert "temperature" not in recorded_target
    # Pin the throughput defaults. The concurrency change must produce a
    # distinct identity instead of resuming an older mixed 8/4 run.
    assert _run_fingerprint(_run_identity(ordinary, [])) == (
        "db994e150b073f618dbed8163c78849e5560c145fe721787cea50b86069cb41b"
    )
    assert _recordable_config(declared)["targets"][0]["quantization"] == {
        "weight_method": "none",
        "compute_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
    }


def test_cli_temperature_is_a_fingerprinted_target_override():
    args = _parse(
        [
            "run",
            "--suite",
            "core",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--temperature",
            "0.7",
        ]
    )

    config = build_config(args)
    recorded_target = config.model_dump(mode="json")["targets"][0]

    assert config.targets[0].temperature == 0.7
    assert config.targets[0].sampling_mode == "adapter"
    assert recorded_target["temperature"] == 0.7
    assert "sampling_mode" not in recorded_target


def test_recommended_sampling_omits_yaml_generation_overrides(tmp_path):
    path = tmp_path / "recommended.yaml"
    path.write_text(
        "suite: core\n"
        "targets:\n"
        "  - base_url: http://gw:8000/v1\n"
        "    model: m\n"
        "    temperature: 0.8\n"
        "    top_p: 0.9\n",
        encoding="utf-8",
    )
    args = _parse(
        [
            "run",
            "--config",
            str(path),
            "--recommended-sampling",
        ]
    )

    config = build_config(args)
    target = config.targets[0]
    recorded_target = config.model_dump(mode="json")["targets"][0]

    assert target.sampling_mode == "recommended"
    assert target.temperature is None
    assert target.top_p is None
    assert recorded_target["sampling_mode"] == "recommended"
    assert "temperature" not in recorded_target


def test_recommended_sampling_conflicts_fail_before_a_run_is_built():
    common = ["run", "--suite", "core", "--base-url", "http://gw:8000", "--model", "m"]

    with pytest.raises(SystemExit):
        _parse(
            [
                *common,
                "--temperature",
                "0.7",
                "--recommended-sampling",
            ]
        )

    args = _parse([*common, "--top-p", "0.9", "--recommended-sampling"])
    with pytest.raises(ValueError, match="cannot be combined with.*--top-p"):
        build_config(args)


@pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf"), True])
def test_target_temperature_must_be_a_finite_non_negative_number(temperature):
    with pytest.raises(ValueError, match="finite non-negative"):
        BenchTarget(base_url="http://gw:8000/v1", model="m", temperature=temperature)


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"extra_body_json": '{"top_k": 20}'},
    ],
)
def test_recommended_sampling_rejects_explicit_generation_overrides(overrides):
    with pytest.raises(ValueError, match="recommended sampling cannot be combined"):
        BenchTarget(
            base_url="http://gw:8000/v1",
            model="m",
            sampling_mode="recommended",
            **overrides,
        )


def test_suite_results_directory_preserves_an_explicit_path():
    args = _parse(
        [
            "run",
            "--base-url",
            "http://gw:8000",
            "--model",
            "m",
            "--suite",
            "core",
            "--results-dir",
            "/custom/results",
        ]
    )

    assert build_config(args).results_dir == "/custom/results"


def test_yaml_core_suite_defaults_to_its_own_results_directory(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(
        "suite: core\ntargets:\n  - {base_url: 'http://gw:8000', model: m}\n",
        encoding="utf-8",
    )

    config = build_config(_parse(["run", "--config", str(path)]))

    assert config.suite == "core"
    assert config.results_dir == "bench/results/core"


def test_config_rejects_an_unknown_suite():
    with pytest.raises(ValueError, match="suite"):
        BenchConfig(
            suite="unknown",
            targets=(
                {
                    "base_url": "http://gw:8000",
                    "model": "m",
                },
            ),
        )


def test_report_and_list_accept_core_suite():
    report = _parse(["report", "run-1", "--suite", "core"])
    listing = _parse(["list", "--suite", "core"])

    assert report.suite == "core"
    assert report.results_dir is None
    assert listing.suite == "core"


def test_model_without_base_url_rejected():
    args = _parse(["run", "--model", "m"])
    with pytest.raises(ValueError, match="--model requires --base-url"):
        build_config(args)


def test_cli_overrides_yaml(tmp_path):
    (tmp_path / "bench.yaml").write_text(
        """
suite: quantization
targets:
  - { base_url: "http://yaml:8000", model: yaml-model }
limit: 5
seed: 7
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
            "gpqa-diamond",
        ]
    )
    config = build_config(args)
    assert config.targets[0].model == "yaml-model"  # from YAML
    assert config.limit == 3  # CLI wins
    assert config.seed == 7  # YAML survives
    assert config.smoke is True
    assert config.only == ("gpqa-diamond",)


def test_no_targets_anywhere_rejected():
    args = _parse(["run"])
    with pytest.raises(ValueError):
        build_config(args)


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
suite: core
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

    local = build_config(_parse(["run", "--config", str(config_path), "--exec-runner", "local"]))
    assert local.execution == ExecutionConfig()
