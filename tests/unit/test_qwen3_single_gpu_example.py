"""Contract for the one-command Qwen3-32B single-GPU example."""

from pathlib import Path

import yaml

from kairyu.deploy.spec import load_deployment_spec

EXAMPLE_DIR = Path("examples/qwen3-32b-single-gpu")
COMPOSE_FILE = EXAMPLE_DIR / "compose.yaml"
CONFIG_FILE = EXAMPLE_DIR / "kairyu.yaml"
RUN_FILE = EXAMPLE_DIR / "run.sh"


def test_qwen3_single_gpu_example_is_one_command_native_kairyu():
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services = compose["services"]

    download = services["model-download"]
    assert "Qwen/Qwen3-32B" in " ".join(download["command"])
    assert download["volumes"] == ["qwen3-32b:/models"]

    kairyu = services["kairyu"]
    assert kairyu["build"] == {
        "context": "../..",
        "dockerfile": "Dockerfile.cuda",
    }
    assert kairyu["depends_on"] == {
        "model-download": {"condition": "service_completed_successfully"}
    }
    assert kairyu["ports"] == ["127.0.0.1:8000:8000"]
    assert kairyu["volumes"] == [
        "./kairyu.yaml:/etc/kairyu/config.yaml:ro",
        "qwen3-32b:/models:ro",
    ]
    devices = kairyu["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [
        {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}
    ]

    spec = load_deployment_spec(CONFIG_FILE)
    assert list(spec.engines) == ["qwen3-32b"]
    engine = spec.engines["qwen3-32b"]
    assert engine.backend == "kairyu"
    assert engine.options == {
        "model_path": "/models/qwen3-32b",
        "tensor_parallel_size": 1,
        "num_pages": 1024,
        "max_num_batched_tokens": 1024,
    }

    launcher = RUN_FILE.read_text(encoding="utf-8")
    assert "docker compose" in launcher
    assert "up --build" in launcher
