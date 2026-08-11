from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from kairyu.deploy.spec import load_deployment_spec
from kairyu.dsl.loader import load_spec
from kairyu.engine.config_validation import validate_backend_options

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/qwen3.6-deepseek-v4-8gpu"


def _load(path: Path, name: str):
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_tiered_example_allocates_four_qwen_replicas_and_one_deepseek_tp4() -> None:
    spec = json.loads((EXAMPLE / "example.json").read_text())
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())

    assert spec["hardware"] == {
        "gpu_count": 8,
        "product": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "minimum_compute_capability": 12.0,
        "minimum_vram_mib": 90000,
    }
    assert spec["allocation"] == {
        "tier1": {
            "model": "qwen3.6-27b",
            "gpu_ids": [0, 1, 2, 3],
            "replicas": 4,
            "tensor_parallel_size": 1,
        },
        "tier2": {
            "model": "deepseek-v4-flash-0731",
            "gpu_ids": [4, 5, 6, 7],
            "replicas": 1,
            "tensor_parallel_size": 4,
            "expert_parallel_size": 4,
        },
    }
    assert set(compose["services"]) == {
        "qwen-0",
        "qwen-1",
        "qwen-2",
        "qwen-3",
        "deepseek",
        "kairyu",
        "chat-ui",
    }
    for index in range(4):
        service = compose["services"][f"qwen-{index}"]
        devices = service["deploy"]["resources"]["reservations"]["devices"][0]
        assert devices["device_ids"] == [str(index)]
        assert "--tensor-parallel-size" not in service["command"]
        assert _option(service["command"], "--max-num-seqs") == "32"
        assert service["volumes"][-1]["target"] == "/root/.cache"
        assert service["environment"] | {
            "XDG_CACHE_HOME": "/root/.cache",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
            "TRITON_CACHE_DIR": "/root/.cache/triton",
            "TILELANG_CACHE_DIR": "/root/.cache/tilelang",
            "TILELANG_TMP_DIR": "/root/.cache/tilelang/tmp",
        } == service["environment"]
    deepseek = compose["services"]["deepseek"]
    devices = deepseek["deploy"]["resources"]["reservations"]["devices"][0]
    assert devices["device_ids"] == ["4", "5", "6", "7"]
    assert _option(deepseek["command"], "--tensor-parallel-size") == "4"
    assert "--enable-expert-parallel" in deepseek["command"]
    assert _option(deepseek["command"], "--max-num-batched-tokens") == "16384"
    assert deepseek["volumes"][1]["target"] == "/root/.cache"
    assert deepseek["environment"] | {
        "XDG_CACHE_HOME": "/root/.cache",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
        "TRITON_CACHE_DIR": "/root/.cache/triton",
        "TILELANG_CACHE_DIR": "/root/.cache/tilelang",
        "TILELANG_TMP_DIR": "/root/.cache/tilelang/tmp",
    } == deepseek["environment"]
    assert json.loads(_option(deepseek["command"], "--speculative-config")) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "greedy",
    }
    assert json.loads(_option(deepseek["command"], "--compilation-config")) == {
        "cudagraph_mode": "NONE",
        "custom_ops": ["all"],
    }


def test_tiered_gateway_owns_l2_pools_templates_and_orchestrators() -> None:
    raw = (EXAMPLE / "kairyu.yaml").read_text()
    deployment = load_deployment_spec(raw, resolve_credentials=False)

    assert set(deployment.pools) == {
        "qwen3.6-27b",
        "deepseek-v4-flash-0731",
        "deepseek-v4-flash-0731-thinking",
    }
    qwen = deployment.pools["qwen3.6-27b"]
    assert len(qwen.replicas) == 4
    assert qwen.prefix_index is True
    assert qwen.queue_depth_threshold == 8
    for replica in qwen.replicas:
        validate_backend_options(replica.backend, replica.options)
        assert replica.options["tensor_parallel_size"] == 1
        assert replica.options["upstream"] == "vllm"
    deepseek = deployment.pools["deepseek-v4-flash-0731"]
    assert len(deepseek.replicas) == 1
    validate_backend_options(deepseek.replicas[0].backend, deepseek.replicas[0].options)
    assert deepseek.replicas[0].options["tensor_parallel_size"] == 4
    assert deepseek.replicas[0].options["expert_parallel_size"] == 4
    assert deepseek.replicas[0].options["dspark_enabled"] is True
    assert set(deployment.orchestrators) == {"kairyu-auto", "kairyu-auto-max"}
    assert deployment.chat_templates == {
        "qwen3.6-27b": "/etc/kairyu/qwen3.6-chat-template.jinja",
        "deepseek-v4-flash-0731": "/etc/kairyu/deepseek-v4-0731.jinja",
        "deepseek-v4-flash-0731-thinking": "/etc/kairyu/deepseek-thinking.jinja",
    }


def test_tiered_l2_pins_moa_fanout_and_budget() -> None:
    standard = load_spec(EXAMPLE / "auto.yaml")
    maximum = load_spec(EXAMPLE / "auto-max.yaml")

    assert [worker.name for worker in standard.workers] == ["tier1", "tier2"]
    assert standard.workers[0].model == "qwen3.6-27b"
    assert standard.workers[1].model == "deepseek-v4-flash-0731"
    assert standard.router.kind == "rules"
    assert standard.router.thresholds is not None
    assert standard.router.thresholds.model_dump() == {
        "multi_step_markers": 64,
        "multi_agent_min_chars": 262144,
        "reasoning_keywords": 64,
        "math_symbols": 512,
        "tier2_min_chars": 131072,
    }
    assert standard.moa_samples == 2
    assert standard.budget.max_steps == 3
    assert standard.budget.max_refine_depth == 0
    assert maximum.router.kind == "calibrated"
    assert maximum.router.target_mode == "auto-max"
    assert maximum.moa_samples == 3
    assert maximum.budget.max_steps == 4


def test_tiered_chat_ui_calls_kairyu_l3() -> None:
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    ui = compose["services"]["chat-ui"]
    assert ui["environment"]["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert ui["environment"]["DEFAULT_MODELS"] == "kairyu-auto"
    assert ui["depends_on"] == {"kairyu": {"condition": "service_healthy"}}
    assert compose["services"]["kairyu"]["depends_on"] == {
        "qwen-0": {"condition": "service_healthy"},
        "qwen-1": {"condition": "service_healthy"},
        "qwen-2": {"condition": "service_healthy"},
        "qwen-3": {"condition": "service_healthy"},
        "deepseek": {"condition": "service_healthy"},
    }


def test_tiered_control_requires_exact_eight_gpu_inventory() -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_control")
    text = "\n".join(
        f"{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0, "
        f"00000000:{16 + index:02x}:00.0"
        for index in range(8)
    )
    rows = control._gpu_inventory(text)
    assert sorted(rows) == list(range(8))


def test_tiered_control_rejects_persistent_storage_outside_nvme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_nvme")
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        control._nvme_root()


def test_tiered_terminalbench_command_is_full_dataset_without_sampling_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "benchmark.py", "tiered_example_benchmark")
    observed: list[str] = []

    def fake_run(command, *, log=None, check=True):
        observed.extend(command)
        return 0

    monkeypatch.setattr(benchmark, "_run", fake_run)
    monkeypatch.setattr(benchmark, "_validate_terminalbench", lambda _path: 0)
    assert benchmark.terminalbench(tmp_path) == 0
    assert observed[observed.index("--only") + 1] == "terminal-bench"
    assert observed[observed.index("--model") + 1] == "kairyu-auto"
    assert observed[observed.index("--attempts") + 1] == "1"
    assert observed[observed.index("--concurrency") + 1] == "4"
    assert "--limit" not in observed
    assert "--temperature" not in observed
    assert "--recommended-sampling" not in observed
    assert "--top-p" not in observed
    assert "--sampling-seed" not in observed


def test_tiered_all_dispatches_every_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "benchmark.py", "tiered_example_all")
    observed: list[str] = []
    monkeypatch.setattr(benchmark, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "_ensure_environment", lambda _no_start: None)
    for name in benchmark.BENCHMARKS:
        monkeypatch.setattr(
            benchmark,
            name.replace("-", "_"),
            lambda _path, selected=name: observed.append(selected) or 0,
        )
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark.py", "all", "--no-start", "--run-id", "all-start"],
    )
    with pytest.raises(SystemExit) as exit_info:
        benchmark.main()
    assert exit_info.value.code == 0
    assert observed == list(benchmark.BENCHMARKS)
    manifest = json.loads((tmp_path / "all-start" / "run.json").read_text())
    assert list(manifest["exit_codes"]) == list(benchmark.BENCHMARKS)
