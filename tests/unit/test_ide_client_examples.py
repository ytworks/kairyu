"""Static safety and deployment contracts for the IDE client examples."""

from pathlib import Path

import pytest

from kairyu.deploy.spec import load_deployment_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "deploy" / "ide-client"


@pytest.mark.parametrize(
    ("filename", "model", "model_path", "host_port", "expected_options"),
    [
        (
            "gpu-replica.yaml",
            "llama-3.1-8b-instruct",
            "/models/llama-3.1-8b-instruct/current",
            8002,
            {"model_path": "/models/llama-3.1-8b-instruct/current"},
        ),
        (
            "qwen3-coder-gpu-replica.yaml",
            "qwen3-coder-30b-a3b-instruct",
            "/models/Qwen/Qwen3-Coder-30B-A3B-Instruct",
            8003,
            {
                "model_path": "/models/Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "tensor_parallel_size": 1,
            },
        ),
    ],
)
def test_ide_replica_examples_parse_and_publish_only_on_loopback(
    filename: str,
    model: str,
    model_path: str,
    host_port: int,
    expected_options: dict[str, object],
) -> None:
    path = EXAMPLE_DIR / filename
    text = path.read_text(encoding="utf-8")
    spec = load_deployment_spec(path)

    assert f"-p 127.0.0.1:{host_port}:8000" in text
    assert f"-p {host_port}:8000" not in text
    assert list(spec.engines) == [model]
    assert spec.engines[model].backend == "kairyu"
    assert spec.engines[model].options == expected_options
    assert spec.engines[model].options["model_path"] == model_path
    assert spec.server.host == "0.0.0.0"
    assert spec.server.port == 8000
    assert spec.server.api_keys_env is None


def test_ide_guide_states_non_loopback_security_boundary() -> None:
    guide = (REPO_ROOT / "docs" / "ide-clients.md").read_text(encoding="utf-8")

    assert "Publishing on a non-loopback address requires" in guide
    assert "authentication configuration" in guide
    assert "host firewall rule" in guide
