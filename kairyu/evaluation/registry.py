"""Static catalog for the M20 evaluation platform.

The foundation intentionally contains metadata only: importing this module must
not import an adapter, inspect the host, access the network, or imply that a
benchmark can run. Adapter PRs change an entry to ``available`` only after its
contract and synthetic end-to-end tests land.
"""

from __future__ import annotations

from kairyu.evaluation.schemas import BenchmarkDefinition, ImplementationStatus

_UNRESOLVED_VERSION = "unresolved"

_CATALOG: tuple[BenchmarkDefinition, ...] = (
    BenchmarkDefinition(
        benchmark_id="swe-bench-pro",
        display_name="SWE-Bench Pro",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="resolved rate",
    ),
    BenchmarkDefinition(
        benchmark_id="terminal-bench-2.1",
        display_name="Terminal-Bench 2.1",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="task success rate",
    ),
    BenchmarkDefinition(
        benchmark_id="livecodebench-v6",
        display_name="LiveCodeBench v6",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="pass@k",
    ),
    BenchmarkDefinition(
        benchmark_id="livecodebench-pro",
        display_name="LiveCodeBench Pro",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="Accepted rate",
    ),
    BenchmarkDefinition(
        benchmark_id="humanitys-last-exam",
        display_name="Humanity's Last Exam",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="Accuracy",
    ),
    BenchmarkDefinition(
        benchmark_id="charxiv-reasoning",
        display_name="CharXiv Reasoning",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="Judge score",
    ),
    BenchmarkDefinition(
        benchmark_id="gpqa-diamond",
        display_name="GPQA Diamond",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="Accuracy",
    ),
    BenchmarkDefinition(
        benchmark_id="scicode",
        display_name="SciCode",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="resolve rate",
    ),
    BenchmarkDefinition(
        benchmark_id="tau3-banking",
        display_name="τ³ Banking",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="pass@4",
    ),
    BenchmarkDefinition(
        benchmark_id="artificial-analysis-lcr",
        display_name="Artificial Analysis Long Context Reasoning",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="Judge Accuracy",
    ),
    BenchmarkDefinition(
        benchmark_id="mrcr-v2",
        display_name="MRCR v2",
        benchmark_version=_UNRESOLVED_VERSION,
        primary_metric="MRCR score",
    ),
)

BENCHMARK_IDS: tuple[str, ...] = tuple(entry.benchmark_id for entry in _CATALOG)
_BY_ID = {entry.benchmark_id: entry for entry in _CATALOG}

if len(_BY_ID) != len(_CATALOG):  # pragma: no cover - import-time invariant
    raise RuntimeError("duplicate benchmark ID in the evaluation catalog")
if any(entry.implementation_status is not ImplementationStatus.PLANNED for entry in _CATALOG):
    raise RuntimeError("foundation catalog entries must start as planned")


def benchmark_catalog() -> tuple[BenchmarkDefinition, ...]:
    """Return all supported benchmarks in deterministic report order."""

    return _CATALOG


def get_benchmark(benchmark_id: str) -> BenchmarkDefinition:
    """Return one catalog entry, raising a descriptive error for unknown IDs."""

    try:
        return _BY_ID[benchmark_id]
    except KeyError:
        available = ", ".join(BENCHMARK_IDS)
        raise KeyError(f"unknown benchmark {benchmark_id!r}; available: {available}") from None
