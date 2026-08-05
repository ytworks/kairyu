"""Checkout-only benchmark inventory and CLI compatibility tests."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
import tomllib
from importlib import resources
from pathlib import Path

import pytest

from bench import orchestration_mock_bench, router_latency
from bench import vllm_quant_kernel_bench as vllm_quant

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FIELDS = {
    "path",
    "module",
    "kind",
    "requires",
    "documentation",
    "installed",
    "invocations",
}


def test_package_manifest_exactly_inventories_checkout_entrypoints() -> None:
    manifest = resources.files("kairyu.bench").joinpath("entrypoints.toml")
    payload = tomllib.loads(manifest.read_text(encoding="utf-8"))
    entries = payload["entrypoints"]

    actual_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "bench").glob("*.py")
        if path.name != "__init__.py"
    )
    declared_paths = [entry["path"] for entry in entries]

    assert payload["schema_version"] == 1
    assert declared_paths == actual_paths
    assert len(declared_paths) == len(set(declared_paths))
    path_only = {
        "bench/batch_invariance_bench.py",
        "bench/kv_answer_equivalence_bench.py",
    }
    for entry in entries:
        assert set(entry) == EXPECTED_FIELDS
        assert entry["module"] == entry["path"].removesuffix(".py").replace("/", ".")
        assert entry["installed"] is False
        expected_invocations = (
            ["path"]
            if entry["path"] in path_only
            else ["path", "module"]
        )
        assert entry["invocations"] == expected_invocations
        assert entry["requires"]
        assert entry["documentation"]


@pytest.mark.parametrize(
    ("path", "module", "execution_marker"),
    [
        ("bench/router_latency.py", "bench.router_latency", "routes="),
        (
            "bench/orchestration_mock_bench.py",
            "bench.orchestration_mock_bench",
            "throughput=",
        ),
        (
            "bench/vllm_quant_kernel_bench.py",
            "bench.vllm_quant_kernel_bench",
            '"runtime": "vLLM"',
        ),
    ],
)
@pytest.mark.parametrize("invocation", ["path", "module"])
def test_fixed_entrypoint_help_does_not_execute(
    path: str,
    module: str,
    execution_marker: str,
    invocation: str,
) -> None:
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


def test_new_argparse_defaults_preserve_no_argument_workloads() -> None:
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


def test_router_latency_machine_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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


def test_vllm_runtime_dependency_is_lazy_and_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def reject_vllm(name: str, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_vllm)

    with pytest.raises(RuntimeError, match="pinned vllm/vllm-openai CUDA image"):
        vllm_quant._load_vllm_runtime()
