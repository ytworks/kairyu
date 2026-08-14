"""Checkout-only model evaluation suites against a deployed gateway.

`python -m evals run` downloads missing datasets, runs every benchmark of the
suite against every target model (single engines and orchestrations alike —
a target is just a model name on an OpenAI-compatible endpoint), scores,
and writes per-pair JSON plus an aggregated scoreboard to the results dir.

Kairyu correctness and performance gates live separately under `verification/`.
Reusable evidence-artifact mechanics live in the package-neutral `evidence/` tree.
"""

from evals.types import BenchConfig, BenchTarget, JudgeConfig, PairResult

__all__ = ["BenchConfig", "BenchTarget", "JudgeConfig", "PairResult"]
