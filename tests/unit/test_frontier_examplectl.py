from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.config_validation import validate_backend_options
from kairyu.entrypoints.chat_template import ChatTemplate

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/deepseek-v4-flash-0731-8gpu"


def _load(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_examples_surface_contains_exactly_one_environment() -> None:
    environments = sorted(
        path.name
        for path in (ROOT / "examples").iterdir()
        if path.is_dir() and not path.name.startswith("__")
    )
    assert environments == ["deepseek-v4-flash-0731-8gpu"]


def test_example_is_exact_eight_gpu_kairyu_to_vllm_to_webui() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())

    assert spec["hardware"] == {
        "gpu_count": 8,
        "product": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "minimum_compute_capability": 12.0,
        "minimum_vram_mib": 90000,
    }
    assert set(compose["services"]) == {"vllm", "kairyu", "chat-ui"}
    webui = compose["services"]["chat-ui"]
    assert webui["environment"]["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert webui["depends_on"] == {"kairyu": {"condition": "service_healthy"}}
    assert compose["services"]["kairyu"]["depends_on"] == {
        "vllm": {"condition": "service_healthy"}
    }
    devices = compose["services"]["vllm"]["deploy"]["resources"]["reservations"][
        "devices"
    ][0]
    assert devices["device_ids"] == [str(index) for index in range(8)]


def test_vllm_command_pins_the_optimized_sm120_contract() -> None:
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    command = compose["services"]["vllm"]["command"]

    expected_pairs = {
        "--max-model-len": "1048576",
        "--kv-cache-dtype": "fp8",
        "--block-size": "256",
        "--tensor-parallel-size": "8",
        "--moe-backend": "deep_gemm_mega_moe",
        "--max-num-seqs": "64",
        "--max-num-batched-tokens": "16384",
    }
    for option, value in expected_pairs.items():
        assert command[command.index(option) + 1] == value
    assert "--enable-expert-parallel" in command
    assert "--enable-prefix-caching" in command
    assert "--enforce-eager" not in command
    assert json.loads(command[command.index("--attention-config") + 1]) == {
        "use_fp4_indexer_cache": True
    }
    assert json.loads(command[command.index("--speculative-config") + 1])[
        "num_speculative_tokens"
    ] == 5
    assert json.loads(command[command.index("--compilation-config") + 1])[
        "cudagraph_mode"
    ] == "FULL_AND_PIECEWISE"


def test_kairyu_l3_declares_the_attested_vllm_l1() -> None:
    raw = (EXAMPLE / "kairyu.yaml").read_text()
    spec = load_deployment_spec(raw)
    assert set(spec.engines) == {"deepseek-v4-flash-0731"}
    entry = spec.engines["deepseek-v4-flash-0731"]
    assert entry.backend == "openai"
    validate_backend_options(entry.backend, entry.options)
    assert entry.options["upstream"] == "vllm"
    assert entry.options["allow_templated_chat_passthrough"] is True
    assert entry.options["tensor_parallel_size"] == 8
    assert entry.options["expert_parallel_size"] == 8
    assert entry.options["dspark_enabled"] is True
    assert spec.chat_templates == {
        "deepseek-v4-flash-0731": "/etc/kairyu/deepseek-v4-0731.jinja"
    }


def test_deepseek_template_matches_checkpoint_text_encoding() -> None:
    template = ChatTemplate.load(str(EXAMPLE / "deepseek-v4-0731.jinja"))
    maximum = (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to "
        "chance: exhaustively decompose the problem into its most fundamental components, "
        "trace every causal chain to its root, and resolve the underlying cause rather "
        "than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the solution from "
        "multiple angles and are certain that no assumption remains unchecked and no "
        "error remains undiscovered.\n\n"
    )
    rendered = template.render(
        [{"role": "user", "content": "Write Python."}],
        template_kwargs={"reasoning_effort": "max", "thinking_mode": "thinking"},
    )
    assert rendered == (
        "<｜begin▁of▁sentence｜>"
        + maximum
        + "<｜User｜>Write Python.<｜Assistant｜><think>"
    )

    multi = template.render(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
    )
    assert multi == (
        "<｜begin▁of▁sentence｜><｜User｜>A<｜Assistant｜><think>"
        "B<｜end▁of▁sentence｜><｜User｜>C<｜Assistant｜><think>"
    )


def test_deepseek_template_fails_closed_on_tools() -> None:
    template = ChatTemplate.load(str(EXAMPLE / "deepseek-v4-0731.jinja"))
    with pytest.raises(ValueError, match="text chat only"):
        template.render(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )


def test_control_preflight_requires_exact_gpu_inventory() -> None:
    control = _load(EXAMPLE / "control.py", "deepseek_example_control")
    text = "\n".join(
        f"{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0"
        for index in range(8)
    )
    rows = control._gpu_inventory(text)
    assert len(rows) == 8
    assert rows[-1]["index"] == 7


def test_full_livecodebench_command_has_no_subset_escape_hatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "benchmark.py", "deepseek_example_benchmark")
    observed: list[str] = []

    def fake_run(command, *, log=None, check=True):
        observed.extend(command)
        return 0

    monkeypatch.setattr(benchmark, "_run", fake_run)
    monkeypatch.setattr(benchmark, "_execution_image", lambda: "sha256:" + "a" * 64)
    assert benchmark.livecodebench(tmp_path) == 0
    assert observed[observed.index("--only") + 1] == "livecodebench"
    assert observed[observed.index("--concurrency") + 1] == "16"
    assert observed[observed.index("--reasoning-effort") + 1] == "max"
    assert observed[observed.index("--temperature") + 1] == "1.0"
    assert observed[observed.index("--top-p") + 1] == "0.95"
    assert "--limit" not in observed
    assert "--smoke" not in observed
    assert observed[observed.index("--exec-runner") + 1] == "docker"
