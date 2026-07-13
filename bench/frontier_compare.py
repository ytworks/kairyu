"""m11 D7: multi-target latency comparison (kairyu vs frontier APIs).

Methodology block (printed into the scoreboard): identical prompt set, N
trials per prompt per target, streaming TTFT = first content chunk, TPOT =
(last content chunk-first content chunk)/(final streamed usage completion_tokens-1);
quality proxy = response length + refusal rate only (real quality evals are out
of scope). Targets are OpenAI-protocol endpoints — kairyu, OpenAI, or any proxy
— configured by URL+key+model. Endpoints that omit usage retain TTFT/output
characters but do not publish a TPOT value.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROMPTS = (
    "Explain the difference between processes and threads in one paragraph.",
    "Write a Python function that merges two sorted lists.",
    "Summarize the plot of Hamlet in three sentences.",
)


@dataclass(frozen=True)
class Target:
    name: str
    base_url: str
    model: str
    api_key: str = "sk-local"


@dataclass
class TrialResult:
    ttft_s: float
    tpot_s: float | None
    output_chars: int
    completion_tokens: int | None = None


@dataclass
class TargetReport:
    name: str
    model: str
    trials: list[TrialResult] = field(default_factory=list)
    errors: int = 0

    def summary(self) -> dict:
        ttfts = sorted(t.ttft_s for t in self.trials)
        tpots = sorted(t.tpot_s for t in self.trials if t.tpot_s is not None)

        def pct(values, q):
            if not values:
                return None
            return values[min(int(len(values) * q), len(values) - 1)]

        return {
            "target": self.name,
            "model": self.model,
            "trials": len(self.trials),
            "errors": self.errors,
            "ttft_p50_s": pct(ttfts, 0.5),
            "ttft_p95_s": pct(ttfts, 0.95),
            "tpot_p50_s": pct(tpots, 0.5),
            "tpot_missing_usage_trials": sum(
                trial.completion_tokens is None for trial in self.trials
            ),
            "mean_output_chars": (
                statistics.mean(t.output_chars for t in self.trials)
                if self.trials
                else None
            ),
        }


async def run_trial(client, target: Target, prompt: str) -> TrialResult:
    start = time.perf_counter()
    first_content_time = None
    last_content_time = None
    chars = 0
    completion_tokens = None
    stream = await client.chat.completions.create(
        model=target.model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=128,
    )
    async for chunk in stream:
        usage = getattr(chunk, "usage", None)
        reported_tokens = getattr(usage, "completion_tokens", None)
        if (
            isinstance(reported_tokens, int)
            and not isinstance(reported_tokens, bool)
            and reported_tokens >= 0
        ):
            completion_tokens = reported_tokens
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            content_time = time.perf_counter()
            if first_content_time is None:
                first_content_time = content_time
            last_content_time = content_time
            chars += len(delta)
    end = time.perf_counter()
    ttft = (first_content_time if first_content_time is not None else end) - start
    tpot = None
    if (
        first_content_time is not None
        and last_content_time is not None
        and completion_tokens is not None
        and completion_tokens >= 2
    ):
        tpot = (last_content_time - first_content_time) / (completion_tokens - 1)
    return TrialResult(
        ttft_s=ttft,
        tpot_s=tpot,
        output_chars=chars,
        completion_tokens=completion_tokens,
    )


async def run_target(target: Target, prompts, trials: int) -> TargetReport:
    import openai

    report = TargetReport(name=target.name, model=target.model)
    client = openai.AsyncOpenAI(base_url=target.base_url, api_key=target.api_key)
    for prompt in prompts:
        for _ in range(trials):
            try:
                report.trials.append(await run_trial(client, target, prompt))
            except Exception:
                report.errors += 1
    return report


def build_scoreboard(reports: list[TargetReport]) -> dict:
    return {
        "methodology": {
            "prompts": len(DEFAULT_PROMPTS),
            "metric_definitions": {
                "ttft": "request start -> first content chunk (streaming)",
                "tpot": (
                    "(last content chunk - first content chunk) / "
                    "(final streamed usage completion_tokens - 1); null when usage "
                    "is missing or completion_tokens < 2"
                ),
            },
            "notes": (
                "identical prompt set per target; quality proxy only; missing "
                "streamed usage is counted and never replaced with SSE chunk count"
            ),
        },
        "results": [report.summary() for report in reports],
    }


def render_markdown(scoreboard: dict) -> str:
    lines = [
        "# Frontier comparison",
        "",
        (
            "| target | model | trials | ttft p50 | ttft p95 | "
            "token tpot p50 | tpot missing usage |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for row in scoreboard["results"]:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['trials']} "
            f"| {row['ttft_p50_s']} | {row['ttft_p95_s']} | {row['tpot_p50_s']} "
            f"| {row['tpot_missing_usage_trials']} |"
        )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append", required=True,
                        help="name=base_url=model[=api_key]")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", default="bench/results/frontier.json")
    args = parser.parse_args()
    targets = []
    for spec in args.target:
        parts = spec.split("=")
        targets.append(Target(*parts))
    reports = [await run_target(t, DEFAULT_PROMPTS, args.trials) for t in targets]
    scoreboard = build_scoreboard(reports)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scoreboard, indent=2))
    out.with_suffix(".md").write_text(render_markdown(scoreboard))
    print(render_markdown(scoreboard))


if __name__ == "__main__":
    asyncio.run(main())
