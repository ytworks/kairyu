"""Lazy benchmark-adapter registry.

Importing this package must not inspect the host, load datasets, or access the
network.  Concrete adapters are imported only by :func:`get_adapter`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kairyu.evaluation.adapters.base import BenchmarkAdapter


def _load_gpqa_diamond() -> BenchmarkAdapter:
    from kairyu.evaluation.adapters.gpqa_diamond import GPQADiamondAdapter

    return GPQADiamondAdapter()


_ADAPTER_FACTORIES: dict[str, Callable[[], BenchmarkAdapter]] = {
    "gpqa-diamond": _load_gpqa_diamond,
}


def available_adapter_ids() -> tuple[str, ...]:
    """Return runnable adapter IDs without importing adapter modules."""

    return tuple(_ADAPTER_FACTORIES)


def get_adapter(benchmark_id: str) -> BenchmarkAdapter:
    """Construct one adapter lazily."""

    try:
        factory = _ADAPTER_FACTORIES[benchmark_id]
    except KeyError:
        available = ", ".join(_ADAPTER_FACTORIES)
        raise KeyError(
            f"benchmark {benchmark_id!r} has no available adapter; available: {available}"
        ) from None
    return factory()


__all__ = ["available_adapter_ids", "get_adapter"]
