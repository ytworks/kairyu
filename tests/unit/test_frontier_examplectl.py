from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kairyu.engine.config_validation import validate_backend_options

ROOT = Path(__file__).resolve().parents[2]


def _load_examplectl():
    module_spec = importlib.util.spec_from_file_location(
        "frontier_examplectl",
        ROOT / "examples/_shared/examplectl.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _load_benchctl():
    module_spec = importlib.util.spec_from_file_location(
        "frontier_benchctl",
        ROOT / "examples/_shared/benchctl.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_model_storage_root_is_absolute_and_environment_scoped(tmp_path: Path) -> None:
    examplectl = _load_examplectl()
    spec = {"environment": "qwen3.6-27b-1gpu"}
    storage_root = tmp_path / "models"

    directory = examplectl._model_storage_directory(
        spec,
        {"MODEL_STORAGE_ROOT": str(storage_root)},
        tmp_path / "repository",
    )

    assert directory == (storage_root / spec["environment"]).resolve()
    assert directory.is_dir()
    with pytest.raises(SystemExit, match="must be an absolute path"):
        examplectl._model_storage_directory(
            spec,
            {"MODEL_STORAGE_ROOT": "relative/models"},
            tmp_path / "repository",
        )


def test_download_command_uses_python3_and_forwards_token_without_argv_value() -> None:
    examplectl = _load_examplectl()
    secret = "hf_private-do-not-render"
    command = examplectl._download_command(
        "model-volume",
        "vllm@sha256:fixed",
        {
            "repo": "Qwen/Qwen3.6-27B",
            "revision": "fixed-revision",
            "slug": "qwen3.6-27b",
        },
        {"HF_TOKEN": secret},
    )

    assert command[command.index("--entrypoint") + 1] == "python3"
    assert ["--env", "HF_TOKEN"] == command[
        command.index("HF_TOKEN") - 1 : command.index("HF_TOKEN") + 1
    ]
    assert all(secret not in argument for argument in command)
    program = command[command.index("-c") + 1]
    assert "os.environ.get('HF_TOKEN') or None" in program
    assert "HF_TOKEN is required" not in program


def test_download_command_allows_public_anonymous_download() -> None:
    examplectl = _load_examplectl()
    command = examplectl._download_command(
        "model-volume",
        "vllm@sha256:fixed",
        {"repo": "public/model", "revision": "fixed", "slug": "model"},
        {},
    )

    assert "HF_TOKEN" not in command


def test_compose_ignores_dotenv_files(monkeypatch, tmp_path: Path) -> None:
    examplectl = _load_examplectl()
    (tmp_path / ".env").write_text("HF_TOKEN=must-not-be-read\n", encoding="utf-8")
    observed = {}

    def fake_run(command, *, env=None, capture=False):
        observed["command"] = command
        observed["env"] = env
        observed["capture"] = capture
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(examplectl, "_run", fake_run)
    examplectl._compose(
        tmp_path,
        {"environment": "qwen3.6-27b-1gpu"},
        {
            "KAIRYU_CPU_IMAGE": "cpu@sha256:fixed",
            "KAIRYU_GPU_IMAGE": "gpu@sha256:fixed",
            "MODEL_VOLUME": "models",
        },
        "vllm",
        ["ps"],
    )

    assert observed["command"] == [
        "docker",
        "compose",
        "--project-directory",
        str(tmp_path),
        "--file",
        str(tmp_path / "compose.yaml"),
        "ps",
    ]
    assert "HF_TOKEN" not in observed["env"]
    assert observed["env"]["COMPOSE_DISABLE_ENV_FILE"] == "1"
    assert observed["env"]["COMPOSE_PROJECT_NAME"] == (
        "kairyu-qwen3-6-27b-1gpu-vllm"
    )


def test_qwen_vllm_services_cap_sequences_for_hybrid_cache() -> None:
    single = yaml.safe_load(
        (ROOT / "examples/qwen3.6-27b-1gpu/compose.yaml").read_text()
    )
    orchestration = yaml.safe_load(
        (ROOT / "examples/qwen3.6-deepseek-v4-8gpu/compose.yaml").read_text()
    )

    for command in (
        single["services"]["vllm"]["command"],
        orchestration["x-qwen-vllm"]["command"],
    ):
        option = command.index("--max-num-seqs")
        assert command[option + 1] == "64"


def test_deepseek_vllm_avoids_broken_inductor_path() -> None:
    compose = yaml.safe_load(
        (ROOT / "examples/deepseek-v4-flash-0731-8gpu/compose.yaml").read_text()
    )
    command = compose["x-vllm"]["command"]

    assert "--enforce-eager" in command
    assert "--compilation-config" not in command


def test_qwen_quality_command_preserves_documented_thinking_budget(tmp_path: Path) -> None:
    benchctl = _load_benchctl()
    spec = yaml.safe_load(
        (ROOT / "examples/qwen3.6-27b-1gpu/example.json").read_text()
    )

    command = benchctl._quality_command(
        ROOT,
        spec,
        "accuracy:livecodebench",
        8001,
        tmp_path,
    )

    option = command.index("--max-output-tokens")
    assert command[option + 1] == "81920"
    timeout = command.index("--request-timeout-s")
    assert command[timeout + 1] == "86400"
    concurrency = command.index("--concurrency")
    assert command[concurrency + 1] == "16"


def test_qwen_livecodebench_30_contract_is_explicit_and_deterministic(
    tmp_path: Path,
) -> None:
    benchctl = _load_benchctl()
    spec = yaml.safe_load(
        (ROOT / "examples/qwen3.6-27b-1gpu/example.json").read_text()
    )

    command = benchctl._quality_command(
        ROOT,
        spec,
        "accuracy:livecodebench",
        8001,
        tmp_path,
        limit=30,
        seed=0,
    )

    limit = command.index("--limit")
    seed = command.index("--seed")
    assert command[limit + 1] == "30"
    assert command[seed + 1] == "0"
    assert command[command.index("--concurrency") + 1] == "16"

    entrypoint = (
        ROOT / "examples/qwen3.6-27b-1gpu/bench-livecodebench-30.sh"
    ).read_text()
    assert "accuracy:livecodebench --limit 30 --seed 0" in entrypoint


def test_all_frontier_examples_declare_throughput_benchmark_concurrency() -> None:
    specs = sorted((ROOT / "examples").glob("*/example.json"))
    assert specs

    for path in specs:
        spec = yaml.safe_load(path.read_text())
        assert spec["benchmark_concurrency"] == 16, path


def test_orchestration_command_forwards_throughput_concurrency(tmp_path: Path) -> None:
    benchctl = _load_benchctl()
    spec = yaml.safe_load(
        (ROOT / "examples/qwen3.6-deepseek-v4-8gpu/example.json").read_text()
    )

    command = benchctl._orchestration_command(ROOT, spec, 8003, tmp_path)

    concurrency = command.index("--concurrency")
    assert command[concurrency + 1] == "16"


def test_all_frontier_gateway_backend_options_pass_static_validation() -> None:
    gateway_paths = sorted((ROOT / "examples").glob("*/*gateway.yaml"))
    assert gateway_paths
    validated = 0

    for path in gateway_paths:
        config = yaml.safe_load(path.read_text())
        for pool in (config.get("pools") or {}).values():
            for replica in pool["replicas"]:
                validate_backend_options(replica["backend"], replica["options"])
                validated += 1

    assert validated >= 10


def test_all_frontier_native_backend_options_pass_static_validation() -> None:
    config_paths = sorted((ROOT / "examples").glob("**/*kairyu.yaml"))
    assert config_paths
    validated = 0

    for path in config_paths:
        config = yaml.safe_load(path.read_text())
        for engine in (config.get("engines") or {}).values():
            validate_backend_options(engine["backend"], engine["options"])
            validated += 1

    assert validated == 3
