"""CPU contracts for the unmeasured six-GPU V4.1 deployment candidate."""

from pathlib import Path

import yaml

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/qwen3.8-deepseek-v4.1-8gpu"


def read(name):
    return yaml.safe_load((EXAMPLE / name).read_text())


def test_six_gpu_topology_keeps_divisible_attention_and_experts():
    manifest = read("example.json")
    allocation = manifest["allocation"]
    assert allocation["tier1"]["gpu_ids"] == [6, 7]
    assert allocation["tier1"]["replicas"] == 2
    ds = allocation["tier2"]
    assert ds["gpu_ids"] == list(range(6))
    assert ds["tensor_parallel_size"] == 2
    assert ds["attention_data_parallel_size"] == 3
    assert ds["expert_parallel_size"] == 6
    assert ds["pipeline_parallel_size"] == 1
    assert manifest["vllm"]["deepseek"]["engram_cpu_offload"] is True
    assert "deepseek_direct_ttft_p50_ms_fallback" not in manifest["verification"]["coding"]


def test_compose_uses_native_v41_chat_and_no_unverified_dspark():
    services = read("compose.yaml")["services"]
    assert not {"qwen-2", "qwen-3"} & services.keys()
    ds = services["deepseek"]
    command = ds["command"]
    for flag, value in [
        ("--tensor-parallel-size", "2"),
        ("--data-parallel-size", "3"),
        ("--tokenizer-mode", "deepseek_v41"),
        ("--reasoning-parser", "deepseek_v41"),
        ("--tool-call-parser", "deepseek_v41"),
        ("--block-size", "64"),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--enable-expert-parallel" in command
    assert "--speculative-config" not in command
    assert "--chat-template" not in command
    assert ds["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == list(
        map(str, range(6))
    )
    assert "${DEEPSEEK_L1_PORT:-8009}" in ds["ports"][0]
    for index in range(2):
        qwen = services[f"qwen-{index}"]
        assert qwen["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == [
            str(6 + index)
        ]
    for name in ("deepseek", "qwen-0", "qwen-1"):
        assert "--middleware" in services[name]["command"]
        assert "KAIRYU_REQUIREMENTS_CONFIG_SHA256" in services[name]["environment"]


def test_native_multimodal_pools_share_one_deepseek_service():
    config = read("kairyu.yaml")
    assert len(config["pools"]["qwen3.8-27b"]["replicas"]) == 2
    assert not config.get("chat_templates")
    for name in ("deepseek-v4.1-flash", "deepseek-v4.1-flash-thinking"):
        assert name in config["legacy_chat_models"]
        options = config["pools"][name]["replicas"][0]["options"]
        assert options["base_url"] == "http://deepseek:8000/v1"
        assert options["model"] == "deepseek-v4.1-flash"
        assert options["capabilities"]["allow_prompt_kinds"] == ["multimodal"]
        assert not options.get("allow_templated_chat_passthrough")
        assert not options.get("completion_reasoning_end_tag")
        assert options["tensor_parallel_size"] == 2
        assert options["expert_parallel_size"] == 6
        assert options["attention_data_parallel_size"] == 3
        assert options["dspark_enabled"] is False
    assert config["embeddings"]["embed-small"]["dimensions"] == 384
