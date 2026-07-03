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
import datetime as _datetime
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
    completion_tokens: int | None = None  # from the include_usage final chunk

    @property
    def tpot_s(self) -> float:
        """Token-granularity when the target reported usage (m9 D5); falls
        back to chunk granularity — the method is labeled in the output."""
        units = (
            self.completion_tokens - 1
            if self.completion_tokens is not None
            else self.output_chunks - 1
        )
        if units is None or units <= 0:
            return 0.0
        return (self.total_s - self.ttft_s) / units

    @property
    def token_granular(self) -> bool:
        return self.completion_tokens is not None


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
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
    request_usage: bool = True,
) -> RequestMetrics:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if request_usage:
        body["stream_options"] = {"include_usage": True}
    start = time.perf_counter()
    ttft = None
    chunks = 0
    completion_tokens = None
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        if response.status_code == 400 and request_usage:
            # target rejects stream_options: retry once without (labeled fallback)
            return await run_one(client, model, prompt, max_tokens, request_usage=False)
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith(_SSE_PREFIX) or line == f"{_SSE_PREFIX}[DONE]":
                continue
            chunk = json.loads(line[len(_SSE_PREFIX):])
            if chunk.get("usage"):  # final usage chunk (empty choices)
                completion_tokens = chunk["usage"].get("completion_tokens")
            if any(
                (choice.get("delta") or {}).get("content")
                for choice in chunk.get("choices", [])
            ):
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    return RequestMetrics(
        ttft_s=ttft if ttft is not None else total,
        total_s=total,
        output_chunks=chunks,
        completion_tokens=completion_tokens,
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[min(int(len(sorted_values) * fraction), len(sorted_values) - 1)]


def build_run_config(args: argparse.Namespace) -> dict:
    """Run config embedded in results so files carry topology (G2 §8, design m5 D6).

    The topology args (``--tensor-parallel``, ``--dp-replicas``, ``--pd``) are
    labels for the results file; the GPU phase wires them into engine behavior.
    """
    return {
        "base_url": args.base_url,
        "model": args.model,
        "dataset": args.dataset,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "ttft_slo_s": args.ttft_slo_s,
        "tensor_parallel": args.tensor_parallel,
        "dp_replicas": args.dp_replicas,
        "pd": args.pd,
    }


async def run_benchmark(args: argparse.Namespace) -> None:
    print(f"config={json.dumps(build_run_config(args))}")
    prompts, dataset_label = load_prompts(
        Path(args.dataset) if args.dataset else None, args.num_requests
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=args.timeout, headers=headers
    ) as client:

        async def bounded(prompt: str) -> RequestMetrics:
            async with semaphore:
                return await run_one(client, args.model, prompt, args.max_tokens)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(bounded(p) for p in prompts))
        wall = time.perf_counter() - wall_start

    ttfts = sorted(metric.ttft_s for metric in results)
    tpots = [metric.tpot_s for metric in results if metric.tpot_s > 0]
    token_granular = all(metric.token_granular for metric in results)
    tpot_method = "token" if token_granular else "chunk"
    within_slo = sum(1 for metric in results if metric.ttft_s <= args.ttft_slo_s)
    summary = {
        "dataset": dataset_label,
        "requests": len(results),
        "wall_s": round(wall, 3),
        "ttft_p50_ms": round(statistics.median(ttfts) * 1e3, 2),
        "ttft_p99_ms": round(_percentile(ttfts, 0.99) * 1e3, 2),
        "tpot_mean_ms": round(statistics.mean(tpots) * 1e3, 3) if tpots else None,
        "tpot_method": tpot_method,  # labeled: token (usage-true) vs chunk fallback
        "throughput_rps": round(len(results) / wall, 2),
        "goodput_rps": round(within_slo / wall, 2),
    }
    print(f"target={args.base_url} model={args.model} dataset={dataset_label}")
    print(f"requests={len(results)} concurrency={args.concurrency} wall={wall:.2f}s")
    print(
        f"TTFT p50={summary['ttft_p50_ms']}ms p99={summary['ttft_p99_ms']}ms"
    )
    if tpots:
        print(f"TPOT mean={summary['tpot_mean_ms']}ms/token ({tpot_method}-granularity)")
    print(
        f"throughput={summary['throughput_rps']} req/s; "
        f"goodput(TTFT<={args.ttft_slo_s}s)={summary['goodput_rps']} req/s"
    )
    if args.results_dir:
        stamp = _datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"{stamp}-serving.json"  # timestamped: same-day safe
        out.write_text(
            json.dumps({"config": build_run_config(args), "summary": summary}, indent=2)
        )
        print(f"results written to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="kairyu-mock")
    parser.add_argument("--dataset", default=None, help="ShareGPT-format JSON path")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ttft-slo-s", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key", default=None,
                        help="Bearer token for authenticated targets (m9 D5)")
    parser.add_argument("--results-dir", default="bench/results",
                        help="Write a timestamped results JSON here ('' to disable)")
    # M5 topology labels (design m5 D6); recorded in the run config for G2 §8.
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="TP degree of the target server (results label)")
    parser.add_argument("--dp-replicas", type=int, default=1,
                        help="DP replica count behind the target (results label)")
    parser.add_argument("--pd", action="store_true",
                        help="target runs prefill-decode disaggregated (results label)")
    return parser


def main() -> None:
    asyncio.run(run_benchmark(build_parser().parse_args()))


if __name__ == "__main__":
    main()
