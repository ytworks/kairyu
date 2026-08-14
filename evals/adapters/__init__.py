"""Adapter registry and canonical benchmark-suite definitions."""

from __future__ import annotations

from dataclasses import dataclass

from evals.adapters.base import BenchmarkAdapter


@dataclass(frozen=True)
class SuiteInfo:
    """Stable identity and presentation policy for one benchmark suite."""

    name: str
    display_name: str
    row_order: tuple[str, ...]


CORE_ROW_ORDER: tuple[str, ...] = (
    "gsm8k",
    "mmlu",
    "ifeval",
)

QUANTIZATION_ROW_ORDER: tuple[str, ...] = (
    *CORE_ROW_ORDER,
    "gpqa-diamond",
)

STRUCTURED_ROW_ORDER: tuple[str, ...] = ("structured-output",)

LONG_CONTEXT_ROW_ORDER: tuple[str, ...] = (
    "ruler-niah-4k",
    "ruler-niah-8k",
    "ruler-niah-16k",
    "ruler-niah-32k",
    "ruler-niah-64k",
    "ruler-niah-128k",
    "ruler-niah-256k",
    "ruler-niah-512k",
    "ruler-niah-1024k",
)

SUITES: dict[str, SuiteInfo] = {
    "core": SuiteInfo(
        name="core",
        display_name="Core",
        row_order=CORE_ROW_ORDER,
    ),
    "quantization": SuiteInfo(
        name="quantization",
        display_name="Quantization",
        row_order=QUANTIZATION_ROW_ORDER,
    ),
    "structured": SuiteInfo(
        name="structured",
        display_name="Structured Output",
        row_order=STRUCTURED_ROW_ORDER,
    ),
    "long-context": SuiteInfo(
        name="long-context",
        display_name="Long Context",
        row_order=LONG_CONTEXT_ROW_ORDER,
    ),
}


def suite_names() -> tuple[str, ...]:
    """Suite names in their stable CLI presentation order."""
    return tuple(SUITES)


def suite_info(name: str) -> SuiteInfo:
    """Return a suite definition, with one consistent validation error."""
    try:
        return SUITES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown suite {name!r}; available: {', '.join(suite_names())}"
        ) from error


def all_adapters() -> dict[str, BenchmarkAdapter]:
    """Fresh adapter instances with pinned dataset revisions applied.

    Pins are attached per instance (see `evals.pins`), so this is the
    entry point the runner, downloader and CLI must use; constructing an adapter
    class directly yields an unpinned instance.
    """
    from evals.adapters.gpqa import GpqaDiamondAdapter
    from evals.adapters.gsm8k import Gsm8kAdapter
    from evals.adapters.ifeval import IfevalAdapter
    from evals.adapters.mmlu import MmluAdapter
    from evals.adapters.ruler_niah import ruler_niah_adapters
    from evals.adapters.structured_output import StructuredOutputAdapter

    adapters: list[BenchmarkAdapter] = [
        Gsm8kAdapter(),
        GpqaDiamondAdapter(),
        IfevalAdapter(),
        MmluAdapter(),
        StructuredOutputAdapter(),
        *ruler_niah_adapters(),
    ]
    from evals.pins import apply_pins

    return {adapter.info.name: adapter for adapter in apply_pins(adapters)}


def suite_adapters(
    suite: str,
    *,
    only: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[BenchmarkAdapter]:
    definition = suite_info(suite)
    unknown = (set(only) | set(exclude)) - set(definition.row_order)
    if unknown:
        raise ValueError(
            f"unknown benchmark names {sorted(unknown)}; "
            f"available: {', '.join(definition.row_order)}"
        )
    registry = all_adapters()
    names = [name for name in definition.row_order if name in registry]
    if only:
        names = [name for name in names if name in only]
    if exclude:
        names = [name for name in names if name not in exclude]
    return [registry[name] for name in names]
