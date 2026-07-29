"""Bench scripts must carry M5 topology flags into their emitted config (design m5 D6)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"


def _load_bench_module(name: str) -> ModuleType:
    module_name = f"bench_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _BENCH_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclass annotation resolution needs the registry
    spec.loader.exec_module(module)
    return module


def test_serving_bench_config_defaults_include_topology():
    bench = _load_bench_module("serving_bench")
    args = bench.build_parser().parse_args([])
    config = bench.build_run_config(args)
    assert config["tensor_parallel"] == 1
    assert config["dp_replicas"] == 1
    assert config["pd"] is False


def test_serving_bench_config_carries_topology_flags():
    bench = _load_bench_module("serving_bench")
    args = bench.build_parser().parse_args(
        ["--tensor-parallel", "4", "--dp-replicas", "2", "--pd"]
    )
    config = bench.build_run_config(args)
    assert config["tensor_parallel"] == 4
    assert config["dp_replicas"] == 2
    assert config["pd"] is True


def test_multiturn_prefix_config_defaults_include_topology():
    bench = _load_bench_module("multiturn_prefix")
    args = bench.build_parser().parse_args([])
    config = bench.build_run_config(args)
    assert config["tensor_parallel"] == 1
    assert config["replicas"] == 1
    assert config["pd"] is False


def test_multiturn_prefix_config_carries_topology_flags():
    bench = _load_bench_module("multiturn_prefix")
    args = bench.build_parser().parse_args(["--replicas", "2", "--tensor-parallel", "8", "--pd"])
    config = bench.build_run_config(args)
    assert config["replicas"] == 2
    assert config["tensor_parallel"] == 8
    assert config["pd"] is True


def test_multiturn_prefix_fixed_workload_geometry_and_theoretical_hits():
    bench = _load_bench_module("multiturn_prefix")
    workload = bench.fixed_workload()

    assert bench.WORKLOAD_SEED == 42
    assert len(workload) == 64 * 8
    assert [request.sequence for request in workload] == list(range(64 * 8))
    for turn in range(8):
        rows = workload[turn * 64 : (turn + 1) * 64]
        assert {request.session for request in rows} == set(range(64))
        assert {request.turn for request in rows} == {turn}
        assert {request.prompt_tokens for request in rows} == {
            512 + (turn + 1) * 128
        }

    assert sum(request.expected_cached_tokens for request in workload) == 491_008
    assert sum(request.prompt_tokens for request in workload) == 557_056
    assert workload == bench.fixed_workload()
