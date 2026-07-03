"""m11 D7: multi-target latency comparison (kairyu vs frontier APIs).

Methodology block (printed into the scoreboard): identical prompt set, N
trials per prompt per target, streaming TTFT = first content chunk, TPOT =
(last-first)/(tokens-1); quality proxy = response length + refusal rate
only (real quality evals are out of scope). Targets are OpenAI-protocol
endpoints — kairyu, OpenAI, or any proxy — configured by URL+key+model.
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
            "mean_output_chars": (
                statistics.mean(t.output_chars for t in self.trials)
                if self.trials
                else None
            ),
        }


async def run_trial(client, target: Target, prompt: str) -> TrialResult:
    start = time.perf_counter()
    first = None
    chunks = 0
    chars = 0
    stream = await client.chat.completions.create(
        model=target.model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=128,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if first is None:
                first = time.perf_counter()
            chunks += 1
            chars += len(delta)
    end = time.perf_counter()
    ttft = (first or end) - start
    tpot = (end - first) / (chunks - 1) if first and chunks > 1 else None
    return TrialResult(ttft_s=ttft, tpot_s=tpot, output_chars=chars)


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
                "tpot": "(last chunk - first chunk) / (chunks - 1)",
            },
            "notes": "identical prompt set per target; quality proxy only",
        },
        "results": [report.summary() for report in reports],
    }


def render_markdown(scoreboard: dict) -> str:
    lines = [
        "# Frontier comparison",
        "",
        "| target | model | trials | ttft p50 | ttft p95 | tpot p50 |",
        "|---|---|---|---|---|---|",
    ]
    for row in scoreboard["results"]:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['trials']} "
            f"| {row['ttft_p50_s']} | {row['ttft_p95_s']} | {row['tpot_p50_s']} |"
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
