"""SWE-bench Verified with mini-SWE-agent's standard official configuration."""

from __future__ import annotations

from kairyu.bench.adapters.swebench import SweBenchAdapter, SweBenchSpec

_RETIREMENT_URL = "https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/"
_INCOMPARABLE_REASON = (
    "the local result is one mini-SWE-agent trial, while Anthropic's Fable 5 "
    "SWE-bench Verified score is the mean of five trials"
)
_ANNOTATIONS = (
    "scaffold: mini-swe-agent standard SWE-bench configuration, "
    "agent.step_limit=250 and cost_limit=$3",
    "target reasoning_effort, top_p, and seed are forwarded as "
    "model.model_kwargs.*; explicit temperature and recommended sampling "
    "are rejected because this wrapper has no verified passthrough",
    "vendor extra_body has no harness equivalent and is NOT forwarded",
    "--attempts must remain 1; this wrapper has no verified repeated-trial "
    "or grouped chat seed-sweep path",
    "the official SWE-bench Docker images assume Linux x86 architecture",
    "OpenAI retired SWE-bench Verified from model reporting because of "
    f"contamination and test-quality concerns: {_RETIREMENT_URL}",
)

VERIFIED_SPEC = SweBenchSpec(
    name="swe-bench-verified",
    display_name="SWE-bench Verified",
    subset="verified",
    dataset="princeton-nlp/SWE-bench_Verified",
    step_limit=250,
    annotations=_ANNOTATIONS,
    comparable_to_published=False,
    incomparable_reason=_INCOMPARABLE_REASON,
)


class SweBenchVerifiedAdapter(SweBenchAdapter):
    def __init__(self) -> None:
        super().__init__(VERIFIED_SPEC)


__all__ = ["SweBenchVerifiedAdapter"]
