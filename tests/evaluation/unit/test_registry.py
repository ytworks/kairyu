"""The M20 catalog is exact, ordered, and lazily exposes landed adapters."""

import sys

import pytest

from kairyu.evaluation.adapters import available_adapter_ids, get_adapter
from kairyu.evaluation.registry import (
    BENCHMARK_IDS,
    benchmark_catalog,
    get_benchmark,
)
from kairyu.evaluation.schemas import ImplementationStatus

EXPECTED = (
    ("swe-bench-pro", "SWE-Bench Pro", "resolved rate"),
    ("terminal-bench-2.1", "Terminal-Bench 2.1", "task success rate"),
    ("livecodebench-v6", "LiveCodeBench v6", "pass@k"),
    ("livecodebench-pro", "LiveCodeBench Pro", "Accepted rate"),
    ("humanitys-last-exam", "Humanity's Last Exam", "Accuracy"),
    ("charxiv-reasoning", "CharXiv Reasoning", "Judge score"),
    ("gpqa-diamond", "GPQA Diamond", "Accuracy"),
    ("scicode", "SciCode", "resolve rate"),
    ("tau3-banking", "τ³ Banking", "pass@4"),
    (
        "artificial-analysis-lcr",
        "Artificial Analysis Long Context Reasoning",
        "Judge Accuracy",
    ),
    ("mrcr-v2", "MRCR v2", "MRCR score"),
)
AVAILABLE_IDS = ("gpqa-diamond", "humanitys-last-exam")


def test_catalog_is_exact_and_deterministically_ordered():
    entries = benchmark_catalog()

    assert (
        tuple((entry.benchmark_id, entry.display_name, entry.primary_metric) for entry in entries)
        == EXPECTED
    )
    assert BENCHMARK_IDS == tuple(row[0] for row in EXPECTED)
    assert len(BENCHMARK_IDS) == len(set(BENCHMARK_IDS)) == 11


def test_only_landed_adapters_are_available():
    statuses = {entry.benchmark_id: entry.implementation_status for entry in benchmark_catalog()}

    assert available_adapter_ids() == AVAILABLE_IDS
    assert {
        benchmark_id
        for benchmark_id, status in statuses.items()
        if status is ImplementationStatus.AVAILABLE
    } == set(AVAILABLE_IDS)
    assert {
        status for benchmark_id, status in statuses.items() if benchmark_id not in AVAILABLE_IDS
    } == {ImplementationStatus.PLANNED}


def test_hle_adapter_is_loaded_only_when_requested(monkeypatch):
    module_name = "kairyu.evaluation.adapters.humanitys_last_exam"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    assert "humanitys-last-exam" in available_adapter_ids()
    assert module_name not in sys.modules

    adapter = get_adapter("humanitys-last-exam")

    assert type(adapter).__name__ == "HumanitysLastExamAdapter"
    assert module_name in sys.modules


@pytest.mark.parametrize("benchmark_id", AVAILABLE_IDS)
def test_available_catalog_metadata_matches_adapter(benchmark_id):
    assert get_benchmark(benchmark_id) == get_adapter(benchmark_id).metadata()


def test_get_benchmark_returns_the_catalog_entries():
    assert get_benchmark("humanitys-last-exam") is benchmark_catalog()[4]
    assert get_benchmark("gpqa-diamond") is benchmark_catalog()[6]


def test_get_adapter_rejects_planned_benchmark():
    with pytest.raises(KeyError, match="'scicode' has no available adapter"):
        get_adapter("scicode")


def test_get_adapter_rejects_unknown_id_without_aliasing():
    with pytest.raises(KeyError, match="'gpqa' has no available adapter"):
        get_adapter("gpqa")


def test_get_benchmark_rejects_unknown_id_without_aliasing():
    with pytest.raises(KeyError, match="unknown benchmark 'gpqa'"):
        get_benchmark("gpqa")
