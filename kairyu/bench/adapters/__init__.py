"""Adapter registry and canonical benchmark-suite definitions."""

from __future__ import annotations

from dataclasses import dataclass

from kairyu.bench.adapters.base import BenchmarkAdapter


@dataclass(frozen=True)
class SuiteInfo:
    """Stable identity and presentation policy for one benchmark suite."""

    name: str
    display_name: str
    row_order: tuple[str, ...]
    published_comparison: bool = False


# Row order of the Fugu release table (sakana.ai/fugu-release). Slots land
# phase by phase; the registry below holds the implemented ones and suites
# are filtered to what exists, so the scoreboard grows without reordering.
FUGU_ROW_ORDER: tuple[str, ...] = (
    "swe-bench-pro",
    "terminal-bench",
    "livecodebench",
    "livecodebench-pro",
    "hle",
    "charxiv-reasoning",
    "gpqa-diamond",
    "scicode",
    "tau-bench-banking",
    "long-context-reasoning",
    "mrcr-v2",
)

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

SUITES: dict[str, SuiteInfo] = {
    "fugu": SuiteInfo(
        name="fugu",
        display_name="Fugu",
        row_order=FUGU_ROW_ORDER,
        published_comparison=True,
    ),
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

    Pins are attached per instance (see `kairyu.bench.pins`), so this is the
    entry point the runner, downloader and CLI must use; constructing an adapter
    class directly yields an unpinned instance.
    """
    from kairyu.bench.adapters.charxiv import CharXivAdapter
    from kairyu.bench.adapters.gpqa import GpqaDiamondAdapter
    from kairyu.bench.adapters.gsm8k import Gsm8kAdapter
    from kairyu.bench.adapters.hle import HleAdapter
    from kairyu.bench.adapters.ifeval import IfevalAdapter
    from kairyu.bench.adapters.livecodebench import LiveCodeBenchAdapter
    from kairyu.bench.adapters.livecodebench_pro import LiveCodeBenchProAdapter
    from kairyu.bench.adapters.longbench_v2 import LongBenchV2Adapter
    from kairyu.bench.adapters.mmlu import MmluAdapter
    from kairyu.bench.adapters.mrcr import MrcrAdapter
    from kairyu.bench.adapters.scicode import SciCodeAdapter
    from kairyu.bench.adapters.structured_output import StructuredOutputAdapter
    from kairyu.bench.adapters.swebench_pro import SweBenchProAdapter
    from kairyu.bench.adapters.tau_bench import TauBenchBankingAdapter
    from kairyu.bench.adapters.terminal_bench import TerminalBenchAdapter

    adapters: list[BenchmarkAdapter] = [
        CharXivAdapter(),
        Gsm8kAdapter(),
        GpqaDiamondAdapter(),
        HleAdapter(),
        IfevalAdapter(),
        LiveCodeBenchAdapter(),
        LiveCodeBenchProAdapter(),
        LongBenchV2Adapter(),
        MrcrAdapter(),
        MmluAdapter(),
        SciCodeAdapter(),
        StructuredOutputAdapter(),
        SweBenchProAdapter(),
        TauBenchBankingAdapter(),
        TerminalBenchAdapter(),
    ]
    from kairyu.bench.pins import apply_pins

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
