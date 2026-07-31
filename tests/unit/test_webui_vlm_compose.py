"""Static contract for the opt-in eight-GPU OpenWebUI VLM stack."""

from pathlib import Path

import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options

COMPOSE = Path("deploy/compose/docker-compose.webui-vlm.yaml")
CONFIG = Path("deploy/compose/config-vlm.yaml")
DOCKERFILE = Path("Dockerfile")
VLLM_IMAGE = (
    "vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404@sha256:"
    "4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8"
)
MODEL_REVISION = "b282fe1ca05915cc8825369f6aa510d7a0982594"
MODEL_NAME = "qwen3-vl-32b"


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _argument_value(command: list[str], option: str) -> str:
    index = command.index(option)
    return command[index + 1]


def test_vlm_overlay_is_opt_in_and_pins_the_real_tp8_runtime() -> None:
    compose = _load_yaml(COMPOSE)
    services = compose["services"]
    assert set(services) == {"kairyu", "webui", "vllm"}

    kairyu = services["kairyu"]
    assert kairyu["build"] == {"args": {"KAIRYU_VISION": "1"}}
    assert kairyu["volumes"] == [
        "./config-vlm.yaml:/etc/kairyu/config.yaml:ro"
    ]
    assert kairyu["depends_on"] == {
        "vllm": {"condition": "service_healthy"}
    }
    assert services["webui"]["environment"] == {
        "DEFAULT_MODELS": MODEL_NAME
    }

    vllm = services["vllm"]
    assert vllm["image"] == VLLM_IMAGE
    assert vllm["ipc"] == "host"
    assert vllm["environment"]["VLLM_MEDIA_URL_ALLOW_REDIRECTS"] == "0"
    assert vllm["environment"]["VLLM_NO_USAGE_STATS"] == "1"
    assert vllm["volumes"] == [
        "qwen3-vl-model-cache:/root/.cache/huggingface"
    ]

    command = vllm["command"]
    assert command[0] == "Qwen/Qwen3-VL-32B-Instruct"
    assert _argument_value(command, "--revision") == MODEL_REVISION
    assert _argument_value(command, "--served-model-name") == MODEL_NAME
    assert _argument_value(command, "--tensor-parallel-size") == "8"
    assert _argument_value(command, "--max-model-len") == "8192"
    assert _argument_value(command, "--mm-processor-kwargs") == (
        '{"min_pixels":65536,"max_pixels":2097152}'
    )
    assert _argument_value(command, "--limit-mm-per-prompt.image") == "1"
    assert _argument_value(command, "--limit-mm-per-prompt.video") == "0"
    assert _argument_value(command, "--mm-encoder-tp-mode") == "data"
    assert _argument_value(command, "--scheduling-policy") == "priority"
    assert "--async-scheduling" in command
    assert "--allowed-local-media-path" not in command
    assert "--allowed-media-domains" not in command

    devices = vllm["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [
        {"driver": "nvidia", "count": 8, "capabilities": ["gpu"]}
    ]
    health = vllm["healthcheck"]
    assert health["test"][:3] == ["CMD", "/usr/bin/python3", "-c"]
    assert "http://127.0.0.1:8000/health" in health["test"][3]
    assert health["retries"] * 10 >= 1800
    assert compose["volumes"] == {"qwen3-vl-model-cache": None}


def test_vlm_gateway_contract_is_strict_and_side_effect_free_to_validate() -> None:
    spec = load_deployment_spec(CONFIG)
    assert spec.server.max_chat_body_bytes == 16_777_216
    assert spec.engines == {}
    assert set(spec.pools) == {MODEL_NAME}
    pool = spec.pools[MODEL_NAME]
    assert pool.unhealthy_after == 1
    assert pool.queue_depth_threshold == 8
    assert pool.probe_interval_s == 2.0
    assert len(pool.replicas) == 1

    replica = pool.replicas[0]
    assert replica.backend == "openai"
    assert replica.health_url == "http://vllm:8000/health"
    assert replica.resolved_health_url() == "http://vllm:8000/health"
    assert replica.options == {
        "base_url": "http://vllm:8000/v1",
        "model": MODEL_NAME,
        "api_key_env": None,
        "upstream": "vllm",
        "capabilities": {"allow_prompt_kinds": ["multimodal"]},
        "image_input_policy": {
            "max_images": 1,
            "max_image_bytes": 8_388_608,
            "max_total_image_bytes": 8_388_608,
            "max_image_pixels": 2_097_152,
            "max_total_image_pixels": 2_097_152,
            "max_image_dimension": 4096,
            "max_image_aspect_ratio": 200,
            "max_processed_prompt_tokens": 8192,
        },
    }
    validate_backend_options(replica.backend, replica.options)

    # The base OpenWebUI image is built with embeddings; keep that endpoint
    # present while the VLM pool becomes the selected chat model.
    assert set(spec.embeddings) == {"embed-small"}


def test_production_dockerfile_installs_the_requested_vision_extra() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG KAIRYU_VISION=0" in dockerfile
    assert dockerfile.count("--extra vision") == 4
    assert dockerfile.count(
        "KAIRYU_EMBEDDINGS and KAIRYU_VISION must each be 0 or 1"
    ) == 2
