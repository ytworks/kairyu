"""SWE-Bench Pro configured to match Fugu's published mini-SWE method."""

from __future__ import annotations

from kairyu.bench.adapters.swebench import (
    SweBenchAdapter,
    SweBenchSpec,
    parse_swebench_report,
)

_ANNOTATIONS = (
    "scaffold: mini-swe-agent (matches Fugu's published methodology), "
    "agent.step_limit=1000",
    "target reasoning_effort, top_p, and seed are forwarded as "
    "model.model_kwargs.*; explicit temperature and recommended sampling "
    "are rejected because this wrapper has no verified passthrough",
    "vendor extra_body has no harness equivalent and is NOT forwarded",
    "--attempts must remain 1; this wrapper has no verified repeated-trial "
    "or grouped chat seed-sweep path",
)

PRO_SPEC = SweBenchSpec(
    name="swe-bench-pro",
    display_name="SWE-Bench Pro",
    subset="ScaleAI/SWE-bench_Pro",
    dataset="ScaleAI/SWE-bench_Pro",
    step_limit=1000,
    annotations=_ANNOTATIONS,
)


class SweBenchProAdapter(SweBenchAdapter):
    def __init__(self) -> None:
        super().__init__(PRO_SPEC)


__all__ = ["SweBenchProAdapter", "parse_swebench_report"]
