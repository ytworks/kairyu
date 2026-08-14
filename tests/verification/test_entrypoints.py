"""Representative checkout entrypoint invocation and lazy-dependency tests."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest

from verification.l1.performance import vllm_quant_kernel_bench as vllm_quant
from verification.orchestration.performance import (
    orchestration_mock_bench,
    router_latency,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("path", "module", "execution_marker"),
    [
        (
            "verification/orchestration/performance/router_latency.py",
            "verification.orchestration.performance.router_latency",
            "routes=",
        ),
        (
            "verification/orchestration/performance/orchestration_mock_bench.py",
            "verification.orchestration.performance.orchestration_mock_bench",
            "throughput=",
        ),
        (
            "verification/l1/performance/vllm_quant_kernel_bench.py",
            "verification.l1.performance.vllm_quant_kernel_bench",
            '"runtime": "vLLM"',
        ),
    ],
)
@pytest.mark.parametrize("invocation", ["path", "module"])
def test_help_does_not_execute(path, module, execution_marker, invocation) -> None:
    command = (
        [sys.executable, path, "--help"]
        if invocation == "path"
        else [sys.executable, "-m", module, "--help"]
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert execution_marker not in completed.stdout


def test_argparse_defaults_preserve_no_argument_workloads() -> None:
    router_args = router_latency._build_parser().parse_args([])
    orchestration_args = orchestration_mock_bench._build_parser().parse_args([])
    assert router_args.routes == router_latency.N_ROUTES
    assert router_args.json is False
    assert router_args.assert_gate is False
    assert orchestration_args.concurrency == orchestration_mock_bench.CONCURRENCY
    assert (
        orchestration_args.simulated_engine_latency
        == orchestration_mock_bench.SIMULATED_ENGINE_LATENCY_S
    )


def test_router_latency_machine_output_fails_closed(monkeypatch, capsys) -> None:
    result = {
        "routes": 1,
        "p50_seconds": 0.001,
        "p99_seconds": router_latency.P99_BUDGET_SECONDS,
        "p99_budget_seconds": router_latency.P99_BUDGET_SECONDS,
        "passed": False,
    }
    monkeypatch.setattr(router_latency, "measure", lambda _routes: result)
    assert router_latency.main(["--routes", "1", "--json", "--assert-gate"]) == 1
    assert json.loads(capsys.readouterr().out) == result


def test_vllm_runtime_dependency_is_lazy_and_actionable(monkeypatch) -> None:
    real_import = builtins.__import__

    def reject_vllm(name: str, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vllm)
    with pytest.raises(RuntimeError, match="pinned vllm/vllm-openai CUDA image"):
        vllm_quant._load_vllm_runtime()
