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
    assert spec["deepseek_l1_loopback_port"] == 8005
    assert spec["benchmarks"]["serving"]["auto_max_combined_max_tokens"] == 4096
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
    assert deepseek["ports"] == [
        "127.0.0.1:${DEEPSEEK_L1_PORT:-8005}:8000"
    ]
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
        "cudagraph_mode": "FULL_AND_PIECEWISE",
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
    assert qwen.queue_depth_threshold == 0
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
    assert "completion_reasoning_end_tag" not in deepseek.replicas[0].options
    thinking = deployment.pools["deepseek-v4-flash-0731-thinking"]
    assert thinking.replicas[0].options["completion_reasoning_end_tag"] == "</think>"
    assert set(deployment.orchestrators) == {
        "kairyu-auto",
        "kairyu-auto-max",
        "kairyu-auto-max-chat",
        "kairyu-auto-max-moa1",
        "kairyu-auto-max-moa2",
        "kairyu-auto-max-moa3",
        "kairyu-auto-max-moa4",
    }
    assert deployment.chat_templates == {
        "qwen3.6-27b": "/etc/kairyu/qwen3.6-chat-template.jinja",
        "deepseek-v4-flash-0731": "/etc/kairyu/deepseek-v4-0731.jinja",
        "deepseek-v4-flash-0731-thinking": "/etc/kairyu/deepseek-thinking.jinja",
    }


def test_tiered_private_reasoning_prompt_converges_without_requesting_a_transcript() -> None:
    template = (EXAMPLE / "deepseek-thinking.jinja").read_text()

    assert "Reason privately and carefully" in template
    assert "converge promptly" in template
    assert "non-empty final answer immediately follows </think>" in template
    assert "entire deliberation process" not in template
    assert "Do not stop reasoning" not in template


def test_tiered_l2_pins_moa_fanout_and_budget() -> None:
    config = json.loads((EXAMPLE / "example.json").read_text())
    standard = load_spec(EXAMPLE / "auto.yaml")
    maximum = load_spec(EXAMPLE / "auto-max.yaml")
    chat = load_spec(EXAMPLE / "auto-max-chat.yaml")

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
    assert chat.router == maximum.router
    assert chat.moa_samples == 2
    assert chat.internal_max_tokens == 1024
    assert config["orchestration"]["auto_max_chat_moa_samples"] == 2
    assert config["orchestration"]["auto_max_chat_internal_max_output_tokens"] == 1024
    assert chat.budget.max_steps == 3
    assert chat.budget.max_refine_depth == maximum.budget.max_refine_depth
    assert chat.workers[0] == maximum.workers[0]
    assert chat.workers[1].model == "deepseek-v4-flash-0731"
    assert maximum.workers[1].model == "deepseek-v4-flash-0731-thinking"
    for samples in range(1, 5):
        candidate = load_spec(EXAMPLE / f"auto-max-moa{samples}.yaml")
        assert candidate.router.target_mode == "auto-max"
        assert candidate.workers == maximum.workers
        assert candidate.moa_samples == samples
        assert candidate.budget.max_steps == samples + 1
        assert candidate.budget.max_refine_depth == 0


def test_tiered_chat_ui_calls_kairyu_l3() -> None:
    compose = yaml.safe_load((EXAMPLE / "compose.yaml").read_text())
    ui = compose["services"]["chat-ui"]
    assert ui["environment"]["OPENAI_API_BASE_URL"] == "http://kairyu:8000/v1"
    assert ui["environment"] | {
        "DEFAULT_MODELS": "kairyu-auto-max",
        "ENABLE_PERSISTENT_CONFIG": "false",
        "ENABLE_SIGNUP": "false",
        "ENABLE_LOGIN_FORM": "false",
        "WEBUI_AUTH": "false",
    } == ui["environment"]
    assert ui["ports"] == ["${CHAT_UI_BIND_ADDRESS:-0.0.0.0}:${CHAT_UI_PORT:-3000}:8080"]
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


def test_tiered_control_uses_explicit_public_ui_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_public_host")
    monkeypatch.setenv("PUBLIC_HOST", "gpu.example.test")

    assert control._public_ui_host() == "gpu.example.test"
    assert control._compose_env()["CHAT_UI_BIND_ADDRESS"] == "0.0.0.0"
    assert control._compose_env()["DEEPSEEK_L1_PORT"] == "8005"


def test_tiered_control_rejects_persistent_storage_outside_nvme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(EXAMPLE / "control.py", "tiered_example_nvme")
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        control._nvme_root()


def test_tiered_benchmark_rejects_storage_outside_nvme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVME_STORAGE_ROOT", "/tmp/not-nvme")
    with pytest.raises(SystemExit, match="/mnt/nvme"):
        _load(EXAMPLE / "benchmark.py", "tiered_example_benchmark_nvme")


def test_tiered_terminalbench_command_is_full_dataset_without_sampling_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "benchmark.py", "tiered_example_benchmark")
    observed: list[str] = []
    observed_env: dict[str, str] = {}
    dataset = tmp_path / "terminal-bench-2-1"
    dataset.mkdir()

    def fake_run(command, *, log=None, check=True, env=None):
        observed.extend(command)
        observed_env.update(env or {})
        return 0

    monkeypatch.setattr(benchmark, "_run", fake_run)
    monkeypatch.setattr(benchmark, "_terminalbench_dataset", lambda _path: dataset)
    monkeypatch.setattr(
        benchmark, "_validate_terminalbench", lambda _path, **_kwargs: 0
    )
    assert benchmark.terminalbench(tmp_path) == 0
    assert observed[observed.index("--only") + 1] == "terminal-bench"
    assert observed[observed.index("--model") + 1] == "kairyu-auto-max"
    assert observed[observed.index("--attempts") + 1] == "1"
    assert observed[observed.index("--concurrency") + 1] == "4"
    assert observed_env["KAIRYU_TERMINAL_BENCH_PATH"] == str(dataset)
    assert "KAIRYU_TERMINAL_BENCH_TASKS" not in observed_env
    assert "--limit" not in observed
    assert "--temperature" not in observed
    assert "--recommended-sampling" not in observed
    assert "--top-p" not in observed
    assert "--sampling-seed" not in observed


def test_tiered_terminalbench_pilot_pins_same_tasks_for_all_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "benchmark.py", "tiered_example_tb_pilot")
    observed: list[str] = []
    observed_env: dict[str, str] = {}
    dataset = tmp_path / "terminal-bench-2-1"
    dataset.mkdir()

    def fake_run(command, *, log=None, check=True, env=None):
        observed.extend(command)
        observed_env.update(env or {})
        return 0

    monkeypatch.setattr(benchmark, "_run", fake_run)
    monkeypatch.setattr(benchmark, "_terminalbench_dataset", lambda _path: dataset)
    monkeypatch.setattr(
        benchmark, "_validate_terminalbench", lambda _path, **_kwargs: 0
    )

    assert benchmark.terminalbench_pilot(tmp_path) == 0
    models = [
        observed[index + 1]
        for index, value in enumerate(observed)
        if value == "--model"
    ]
    config = benchmark.SPEC["benchmarks"]["terminalbench"]
    assert models == config["pilot_models"]
    assert observed[observed.index("--limit") + 1] == str(len(config["pilot_tasks"]))
    assert observed_env["KAIRYU_TERMINAL_BENCH_PATH"] == str(dataset)
    assert observed_env["KAIRYU_TERMINAL_BENCH_TASKS"] == ",".join(
        config["pilot_tasks"]
    )


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


def test_tiered_terminalbench_validator_rejects_partial_or_wrong_denominator(
    tmp_path: Path,
) -> None:
    benchmark = _load(EXAMPLE / "benchmark.py", "tiered_example_tb_validator")
    run = tmp_path / "pilot"
    run.mkdir()
    scoreboard = {
        "cells": {
            "terminal-bench": {
                "deepseek-v4-flash-0731": {
                    "status": "partial",
                    "score": 0.5,
                    "n": 4,
                    "n_scored": 3,
                }
            }
        }
    }
    (run / "scoreboard.json").write_text(json.dumps(scoreboard))

    assert (
        benchmark._validate_terminalbench(
            tmp_path,
            run_id="pilot",
            models=("deepseek-v4-flash-0731",),
            expected_tasks=4,
        )
        == 1
    )


def test_tiered_terminalbench_validator_requires_clean_error_free_raw_items(
    tmp_path: Path,
) -> None:
    benchmark = _load(EXAMPLE / "benchmark.py", "tiered_example_tb_raw_validator")
    run = tmp_path / "pilot"
    result_dir = run / "terminal-bench--suite"
    result_dir.mkdir(parents=True)
    model = "deepseek-v4-flash-0731"
    scoreboard = {
        "cells": {
            "terminal-bench": {
                model: {
                    "status": "completed",
                    "score": 0.75,
                    "n": 4,
                    "n_scored": 4,
                }
            }
        }
    }
    (run / "scoreboard.json").write_text(json.dumps(scoreboard))
    raw = {
        "target": model,
        "status": "completed",
        "source_identity": {"source_tree_clean": True},
        "metrics": {
            "n_total": 4,
            "n_scored": 4,
            "n_unjudged": 0,
            "n_skipped": 0,
            "n_failed": 0,
        },
        "items": [
            {"status": "completed", "score": score, "error": None}
            for score in (1.0, 1.0, 1.0, 0.0)
        ],
    }
    raw_path = result_dir / "deepseek.json"
    raw_path.write_text(json.dumps(raw))

    assert (
        benchmark._validate_terminalbench(
            tmp_path,
            run_id="pilot",
            models=(model,),
            expected_tasks=4,
        )
        == 0
    )

    raw["items"][3]["error"] = "runtime exception"
    raw["metrics"]["n_failed"] = 1
    raw_path.write_text(json.dumps(raw))
    assert (
        benchmark._validate_terminalbench(
            tmp_path,
            run_id="pilot",
            models=(model,),
            expected_tasks=4,
        )
        == 1
    )


def test_tiered_moa_candidate_serving_requires_exact_trace_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _load(EXAMPLE / "benchmark.py", "tiered_example_moa_serving")
    observed: list[tuple[str, dict]] = []

    def fake_serving(model, _run_dir, **kwargs):
        observed.append((model, kwargs))
        return 0

    monkeypatch.setattr(benchmark, "_serving", fake_serving)
    for samples in range(1, 5):
        function = getattr(benchmark, f"serving_auto_max_moa{samples}")
        assert function(tmp_path / str(samples)) == 0

    assert [model for model, _ in observed] == [
        f"kairyu-auto-max-moa{samples}" for samples in range(1, 5)
    ]
    for samples, (_, kwargs) in enumerate(observed, start=1):
        assert kwargs | {
            "tensor_parallel": 4,
            "replicas": 1,
            "expected_route": "moa",
            "expected_role": "moa",
            "expected_kind": "synthesis",
            "expected_moa_samples": samples,
            "warmup_requests": 4,
            "natural_completion": True,
        } == kwargs
