"""The M20 catalog is exact, ordered, and metadata-only."""

import pytest

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


def test_catalog_is_exact_and_deterministically_ordered():
    entries = benchmark_catalog()

    assert (
        tuple((entry.benchmark_id, entry.display_name, entry.primary_metric) for entry in entries)
        == EXPECTED
    )
    assert BENCHMARK_IDS == tuple(row[0] for row in EXPECTED)
    assert len(BENCHMARK_IDS) == len(set(BENCHMARK_IDS)) == 11


def test_foundation_catalog_does_not_claim_adapters_are_available():
    assert {entry.implementation_status for entry in benchmark_catalog()} == {
        ImplementationStatus.PLANNED
    }


def test_get_benchmark_returns_the_catalog_entry():
    assert get_benchmark("gpqa-diamond") is benchmark_catalog()[6]


def test_get_benchmark_rejects_unknown_id_without_aliasing():
    with pytest.raises(KeyError, match="unknown benchmark 'gpqa'"):
        get_benchmark("gpqa")
