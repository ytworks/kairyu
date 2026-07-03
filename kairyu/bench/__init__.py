"""Fugu-suite quality benchmarks: one command against a deployed gateway (G6 P-C1).

`kairyu bench run` downloads missing datasets, runs every benchmark of the
suite against every target model (single engines and orchestrations alike —
a target is just a model name on an OpenAI-compatible endpoint), scores,
and writes per-pair JSON plus an aggregated scoreboard to the results dir.

The perf harnesses in the top-level `bench/` directory are separate: they
measure serving latency/throughput; this package measures answer quality.
"""

from kairyu.bench.types import BenchConfig, BenchTarget, JudgeConfig, PairResult

__all__ = ["BenchConfig", "BenchTarget", "JudgeConfig", "PairResult"]
