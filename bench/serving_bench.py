"""Serving benchmark harness: TTFT p50/p99, TPOT, goodput over an OpenAI-compatible API.

Works against ANY OpenAI-compatible server (kairyu, vLLM, SGLang), so the same
script produces the M2 acceptance comparison on identical hardware. Prints only
measured values and labels the target endpoint; nothing is estimated.

Datasets:
  --dataset sharegpt.json   ShareGPT-format JSON (list of {"conversations": [...]})
  (omitted)                 synthetic prompts, clearly labeled as synthetic

Examples:
  uv run python bench/serving_bench.py --base-url http://localhost:8000 \
      --model kairyu-mock --num-requests 128 --concurrency 128
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

_SSE_PREFIX = "data: "


@dataclass(frozen=True)
class RequestMetrics:
    ttft_s: float
    total_s: float
    output_chunks: int

    @property
    def tpot_s(self) -> float:
        decode_chunks = self.output_chunks - 1
        if decode_chunks <= 0:
            return 0.0
        return (self.total_s - self.ttft_s) / decode_chunks


def load_prompts(dataset: Path | None, num_requests: int) -> tuple[list[str], str]:
    if dataset is None:
        prompts = [
            f"Question {i}: summarize the trade-offs of approach {i % 7} in two sentences."
            for i in range(num_requests)
        ]
        return prompts, "synthetic"
    records = json.loads(dataset.read_text(encoding="utf-8"))
    prompts = []
    for record in records:
        turns = record.get("conversations", [])
        human_turns = [t["value"] for t in turns if t.get("from") in ("human", "user")]
        if human_turns:
            prompts.append(human_turns[0])
        if len(prompts) >= num_requests:
            break
    if len(prompts) < num_requests:
        raise ValueError(
            f"dataset has only {len(prompts)} usable prompts, need {num_requests}"
        )
    return prompts, f"sharegpt:{dataset.name}"


async def run_one(
    client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int
) -> RequestMetrics:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    start = time.perf_counter()
    ttft = None
    chunks = 0
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith(_SSE_PREFIX) or line == f"{_SSE_PREFIX}[DONE]":
                continue
            chunk = json.loads(line[len(_SSE_PREFIX):])
            if any(choice["delta"].get("content") for choice in chunk.get("choices", [])):
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    return RequestMetrics(ttft_s=ttft if ttft is not None else total, total_s=total,
                          output_chunks=chunks)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[min(int(len(sorted_values) * fraction), len(sorted_values) - 1)]


async def run_benchmark(args: argparse.Namespace) -> None:
    prompts, dataset_label = load_prompts(
        Path(args.dataset) if args.dataset else None, args.num_requests
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:

        async def bounded(prompt: str) -> RequestMetrics:
            async with semaphore:
                return await run_one(client, args.model, prompt, args.max_tokens)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(bounded(p) for p in prompts))
        wall = time.perf_counter() - wall_start

    ttfts = sorted(metric.ttft_s for metric in results)
    tpots = [metric.tpot_s for metric in results if metric.tpot_s > 0]
    within_slo = sum(1 for metric in results if metric.ttft_s <= args.ttft_slo_s)
    print(f"target={args.base_url} model={args.model} dataset={dataset_label}")
    print(f"requests={len(results)} concurrency={args.concurrency} wall={wall:.2f}s")
    print(
        f"TTFT p50={statistics.median(ttfts) * 1e3:.1f}ms "
        f"p99={_percentile(ttfts, 0.99) * 1e3:.1f}ms"
    )
    if tpots:
        print(f"TPOT mean={statistics.mean(tpots) * 1e3:.2f}ms/token (chunk-granularity)")
    print(
        f"throughput={len(results) / wall:.1f} req/s; "
        f"goodput(TTFT<={args.ttft_slo_s}s)={within_slo / wall:.1f} req/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="kairyu-mock")
    parser.add_argument("--dataset", default=None, help="ShareGPT-format JSON path")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ttft-slo-s", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    asyncio.run(run_benchmark(parser.parse_args()))


if __name__ == "__main__":
    main()
