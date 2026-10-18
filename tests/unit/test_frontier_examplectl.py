from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options
from kairyu.entrypoints.chat_template import ChatTemplate

ROOT = Path(__file__).resolve().parents[2]
QWEN_EXAMPLE = ROOT / "examples/qwen3.8-27b-1gpu"


def _load(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_qwen_example_is_one_gpu_kairyu_to_vllm_to_webui_on_nvme() -> None:
    spec = json.loads((QWEN_EXAMPLE / "example.json").read_text())
    compose = yaml.safe_load((QWEN_EXAMPLE / "compose.yaml").read_text())

    assert spec["hardware"] == {
        "gpu_count": 1,
        "product": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "minimum_compute_capability": 12.0,
        "minimum_vram_mib": 90000,
    }
    assert spec["storage"]["root"].startswith("/mnt/nvme/")
    assert {
        key: spec["model"][key] for key in ("repo", "revision", "served_name")
    } == {
        "repo": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        "served_name": "qwen3.8-27b",
    }
    assert spec["model"]["tree_sha256"] == (
        "9825ce119c9693172e04dd2a1f2437884503ceab9bf55606141e6662c9fe301e"
    )
    assert set(compose["services"]) == {"vllm", "kairyu", "chat-ui"}
    assert spec["vllm"]["release"] == "v0.31.0"
    assert compose["services"]["vllm"]["image"] == (
        f"${{VLLM_IMAGE:-{spec['vllm']['image']}}}"
    )
    webui = compose["services"]["chat-ui"]
    assert webui["environment"]["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert json.loads(webui["environment"]["DEFAULT_MODEL_PARAMS"]) == {
        "max_tokens": 32768,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    assert webui["depends_on"] == {"kairyu": {"condition": "service_healthy"}}
    assert compose["services"]["kairyu"]["depends_on"] == {"vllm": {"condition": "service_healthy"}}
    devices = compose["services"]["vllm"]["deploy"]["resources"]["reservations"]["devices"][0]
    assert devices["device_ids"] == ["${GPU_ID:-0}"]
    for service in ("vllm", "kairyu"):
        model_mount = compose["services"][service]["volumes"][0]
        assert model_mount["source"] == "${MODEL_STORAGE_PATH}"
    assert compose["services"]["vllm"]["volumes"][1]["source"] == "${VLLM_CACHE_PATH}"
    assert webui["volumes"][0]["source"] == "${WEBUI_STORAGE_PATH}"
    assert compose["services"]["kairyu"]["build"]["args"] == {"KAIRYU_VISION": "1"}


def test_qwen_vllm_command_pins_the_sm120_latency_throughput_contract() -> None:
    compose = yaml.safe_load((QWEN_EXAMPLE / "compose.yaml").read_text())
    service = compose["services"]["vllm"]
    command = service["command"]

    expected_pairs = {
        "--max-model-len": "262144",
        "--dtype": "bfloat16",
        "--kv-cache-dtype": "fp8",
        "--mamba-cache-dtype": "float16",
        "--mamba-ssm-cache-dtype": "float16",
        "--mamba-cache-mode": "align",
        "--max-num-batched-tokens": "32768",
        "--max-num-seqs": "32",
        "--attention-backend": "FLASHINFER",
    }
    for option, value in expected_pairs.items():
        assert command[command.index(option) + 1] == value
    assert "--language-model-only" not in command
    assert json.loads(command[command.index("--mm-processor-kwargs") + 1]) == {
        "min_pixels": 65_536,
        "max_pixels": 2_097_152,
    }
    assert command[command.index("--limit-mm-per-prompt.image") + 1] == "1"
    assert command[command.index("--limit-mm-per-prompt.video") + 1] == "0"
    assert "--enable-prefix-caching" in command
    assert "--enable-chunked-prefill" in command
    assert "--enable-flashinfer-autotune" in command
    assert "--enforce-eager" not in command
    assert "--speculative-config" not in command
    assert command[command.index("--chat-template") + 1] == (
        "/etc/kairyu/qwen3.8-chat.jinja"
    )
    assert json.loads((QWEN_EXAMPLE / "example.json").read_text())["vllm"][
        "mtp_speculative_tokens"
    ] == 0
    assert (
        json.loads(command[command.index("--compilation-config") + 1])["cudagraph_mode"]
        == "PIECEWISE"
    )
    assert "VLLM_USE_FASTOKENS" not in service["environment"]


def test_qwen_kairyu_l3_declares_strict_multimodal_vllm_l1() -> None:
    raw = (QWEN_EXAMPLE / "kairyu.yaml").read_text()
    spec = load_deployment_spec(raw)
    assert set(spec.engines) == {"qwen3.8-27b"}
    entry = spec.engines["qwen3.8-27b"]
    assert entry.backend == "openai"
    validate_backend_options(entry.backend, entry.options)
    assert entry.options["upstream"] == "vllm"
    assert entry.options["capabilities"] == {
        "allow_prompt_kinds": ["multimodal"],
        "allow_chat_template_kwargs": ["enable_thinking"],
    }
    assert entry.options["image_input_policy"] == {
        "max_images": 1,
        "max_image_bytes": 8_388_608,
        "max_total_image_bytes": 8_388_608,
        "max_image_pixels": 2_097_152,
        "max_total_image_pixels": 2_097_152,
        "max_image_dimension": 4096,
        "max_image_aspect_ratio": 200,
        "max_processed_prompt_tokens": 262_144,
    }
    assert entry.options["tensor_parallel_size"] == 1
    assert entry.options["mtp_enabled"] is False
    assert spec.server.max_chat_body_bytes == 16_777_216
    assert spec.chat_templates == {}
    assert spec.legacy_chat_models == frozenset()
    for models in (
        spec.upstream_chat_models,
        spec.upstream_responses_models,
        spec.upstream_messages_models,
    ):
        assert models == frozenset({"qwen3.8-27b"})


def test_qwen_template_defaults_chat_to_direct_and_allows_explicit_thinking() -> None:
    template = ChatTemplate.load(str(QWEN_EXAMPLE / "chat_template.jinja"))
    messages = [{"role": "user", "content": "Return OK."}]
    direct = template.render(messages)
    thinking = template.render(
        messages,
        template_kwargs={"reasoning_effort": "max", "thinking_mode": "thinking"},
    )
    assert direct.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert thinking.endswith("<|im_start|>assistant\n<think>\n")


def test_qwen_control_selects_one_gpu_and_forces_nvme_storage() -> None:
    control = _load(QWEN_EXAMPLE / "control.py", "qwen_example_control")
    text = "0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0, 0000:16:00.0"
    rows = control._gpu_inventory(text)
    assert rows[0]["memory_mib"] == 97887
    assert control.SPEC["storage"]["root"] == "/mnt/nvme/kairyu"


def test_qwen_serving_prompts_do_not_share_the_first_prefix(tmp_path: Path) -> None:
    benchmark = _load(QWEN_EXAMPLE / "verification.py", "qwen_example_serving")
    dataset = tmp_path / "serving.json"
    benchmark._serving_dataset(dataset, requests=4, approximate_tokens=32)
    rows = json.loads(dataset.read_text())
    prompts = [row["conversations"][0]["value"] for row in rows]
    assert len(set(prompts)) == 4
    assert [prompt.split(":", 1)[0] for prompt in prompts] == [
        "Case 0",
        "Case 1",
        "Case 2",
        "Case 3",
    ]


def test_qwen_serving_validation_rejects_partial_streams(tmp_path: Path) -> None:
    benchmark = _load(QWEN_EXAMPLE / "verification.py", "qwen_example_serving_validation")
    row_dir = tmp_path / "serving-c2"
    row_dir.mkdir()
    # Responses rows end naturally, so lengths differ within max_output_tokens.
    result = {
        "summary": {
            "requests": 2,
            "completion_tokens_total": 7,
            "output_tokens_per_s": 1.0,
        },
        "samples": [{"completion_tokens": 3}, {"completion_tokens": 4}],
    }
    output = row_dir / "result-serving.json"
    output.write_text(json.dumps(result))
    assert benchmark._validate_serving_row(row_dir, requests=2, max_output_tokens=4) == 0

    result["summary"]["completion_tokens_total"] = None
    result["samples"][1]["completion_tokens"] = 0
    output.write_text(json.dumps(result))
    assert benchmark._validate_serving_row(row_dir, requests=2, max_output_tokens=4) == 1
