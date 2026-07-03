"""Adapter registry + the canonical Fugu release table row order."""

from __future__ import annotations

from kairyu.bench.adapters.base import BenchmarkAdapter

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


def all_adapters() -> dict[str, BenchmarkAdapter]:
    from kairyu.bench.adapters.charxiv import CharXivAdapter
    from kairyu.bench.adapters.gpqa import GpqaDiamondAdapter
    from kairyu.bench.adapters.hle import HleAdapter
    from kairyu.bench.adapters.longbench_v2 import LongBenchV2Adapter
    from kairyu.bench.adapters.mrcr import MrcrAdapter

    adapters: list[BenchmarkAdapter] = [
        CharXivAdapter(),
        GpqaDiamondAdapter(),
        HleAdapter(),
        LongBenchV2Adapter(),
        MrcrAdapter(),
    ]
    return {adapter.info.name: adapter for adapter in adapters}


def suite_adapters(
    suite: str,
    *,
    only: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[BenchmarkAdapter]:
    if suite != "fugu":
        raise ValueError(f"unknown suite {suite!r}; available: fugu")
    registry = all_adapters()
    unknown = (set(only) | set(exclude)) - set(FUGU_ROW_ORDER)
    if unknown:
        raise ValueError(
            f"unknown benchmark names {sorted(unknown)}; "
            f"available: {', '.join(FUGU_ROW_ORDER)}"
        )
    names = [name for name in FUGU_ROW_ORDER if name in registry]
    if only:
        names = [name for name in names if name in only]
    if exclude:
        names = [name for name in names if name not in exclude]
    return [registry[name] for name in names]
