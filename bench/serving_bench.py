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
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from kairyu.bench.reporting import (
    PERCENTILE_METHOD,
    atomic_write_json,
    nearest_rank_percentile,
)
from kairyu.bench.targets import (
    TARGET_SPEC_FORMAT,
    normalize_base_url,
    parse_target_spec,
    target_api_key,
)
from kairyu.bench.types import BenchTarget

_SSE_PREFIX = "data: "
_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_MODEL = "kairyu-mock"


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
    *,
    temperature: float | None = None,
    seed: int | None = None,
    min_tokens: int | None = None,
    ignore_eos: bool = False,
) -> RequestMetrics:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    if min_tokens is not None:
        body["min_tokens"] = min_tokens
    if ignore_eos:
        body["ignore_eos"] = True
    if request_usage:
        body["stream_options"] = {"include_usage": True}
    start = time.perf_counter()
    ttft = None
    chunks = 0
    completion_tokens = None
    # Clients use the shared canonical API root (ending in /v1). Keep the
    # request relative so URL joining cannot produce /v1/v1.
    async with client.stream("POST", "chat/completions", json=body) as response:
        if response.status_code == 400 and request_usage:
            # target rejects stream_options: retry once without (labeled fallback)
            return await run_one(
                client,
                model,
                prompt,
                max_tokens,
                request_usage=False,
                temperature=temperature,
                seed=seed,
                min_tokens=min_tokens,
                ignore_eos=ignore_eos,
            )
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
    """Compatibility wrapper for the package-owned nearest-rank definition."""
    return nearest_rank_percentile(sorted_values, fraction)


def resolve_target(args: argparse.Namespace) -> BenchTarget:
    """Resolve the shared target contract or the legacy split endpoint flags."""
    target_spec = getattr(args, "target", None)
    base_url = getattr(args, "base_url", None)
    model = getattr(args, "model", None)
    api_key_env = getattr(args, "api_key_env", None)
    literal_api_key = getattr(args, "api_key", None)

    if target_spec is not None:
        conflicting = [
            flag
            for flag, value in (
                ("--base-url", base_url),
                ("--model", model),
                ("--api-key-env", api_key_env),
                ("--api-key", literal_api_key),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError(
                f"--target cannot be combined with {', '.join(conflicting)}"
            )
        return parse_target_spec(target_spec)

    if literal_api_key == "":
        raise ValueError("--api-key must not be empty")
    resolved_model = model or _DEFAULT_MODEL
    return BenchTarget(
        name=resolved_model,
        base_url=normalize_base_url(base_url or _DEFAULT_BASE_URL),
        model=resolved_model,
        api_key_env=api_key_env,
    )


def resolve_api_key(args: argparse.Namespace, target: BenchTarget) -> str | None:
    """Resolve auth without ever adding the secret value to run metadata."""
    literal_api_key = getattr(args, "api_key", None)
    if literal_api_key is not None:
        return literal_api_key
    return target_api_key(
        target,
        required=target.api_key_env is not None,
    )


def build_run_config(args: argparse.Namespace) -> dict:
    """Run config embedded in results so files carry topology (G2 §8, design m5 D6).

    The topology args (``--tensor-parallel``, ``--dp-replicas``, ``--pd``) are
    labels for the results file; the GPU phase wires them into engine behavior.
    """
    target = resolve_target(args)
    return {
        "target": target.label(),
        "base_url": normalize_base_url(target.base_url),
        "model": target.model,
        "api_key_env": target.api_key_env,
        "api_key_source": (
            "legacy-cli"
            if getattr(args, "api_key", None) is not None
            else "environment"
            if target.api_key_env is not None
            else "none"
        ),
        "dataset": args.dataset,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "min_tokens": args.min_tokens,
        "ignore_eos": args.ignore_eos,
        "ttft_slo_s": args.ttft_slo_s,
        "tensor_parallel": args.tensor_parallel,
        "dp_replicas": args.dp_replicas,
        "pd": args.pd,
    }


def summarize_results(
    results: list[RequestMetrics],
    *,
    wall_ns: int,
    dataset_label: str,
    ttft_slo_s: float,
) -> tuple[dict, list[dict]]:
    """Build summary plus lossless per-request timing/usage evidence."""
    if wall_ns <= 0:
        raise ValueError(f"wall_ns must be positive, got {wall_ns}")
    wall_s = wall_ns / 1e9
    ttfts = sorted(metric.ttft_s for metric in results)
    tpots = [metric.tpot_s for metric in results if metric.tpot_s > 0]
    token_granular = all(metric.token_granular for metric in results)
    tpot_method = "token" if token_granular else "chunk"
    within_slo = sum(metric.ttft_s <= ttft_slo_s for metric in results)
    completion_tokens_total = (
        sum(metric.completion_tokens or 0 for metric in results)
        if token_granular
        else None
    )
    samples = [
        {
            "request_index": index,
            "ttft_ms": metric.ttft_s * 1e3,
            "total_ms": metric.total_s * 1e3,
            "tpot_ms": metric.tpot_s * 1e3,
            "output_chunks": metric.output_chunks,
            "completion_tokens": metric.completion_tokens,
        }
        for index, metric in enumerate(results)
    ]
    summary = {
        "dataset": dataset_label,
        "percentile_method": PERCENTILE_METHOD,
        "requests": len(results),
        "wall_ns": wall_ns,
        "wall_s": wall_s,
        "ttft_p50_ms": round(nearest_rank_percentile(ttfts, 0.50) * 1e3, 2),
        "ttft_p99_ms": round(nearest_rank_percentile(ttfts, 0.99) * 1e3, 2),
        "tpot_mean_ms": round(statistics.mean(tpots) * 1e3, 3) if tpots else None,
        "tpot_method": tpot_method,
        "throughput_rps": round(len(results) / wall_s, 2),
        "goodput_rps": round(within_slo / wall_s, 2),
        "completion_tokens_total": completion_tokens_total,
        "output_tokens_per_s": (
            round(completion_tokens_total / wall_s, 2)
            if completion_tokens_total is not None
            else None
        ),
    }
    return summary, samples


async def run_benchmark(args: argparse.Namespace) -> None:
    target = resolve_target(args)
    api_key = resolve_api_key(args, target)
    if getattr(args, "api_key", None) is not None:
        print(
            "warning: --api-key is deprecated because command-line arguments may "
            "be visible to other processes; use --api-key-env or --target "
            f"{TARGET_SPEC_FORMAT}",
            file=sys.stderr,
        )
    print(f"config={json.dumps(build_run_config(args))}")
    prompts, dataset_label = load_prompts(
        Path(args.dataset) if args.dataset else None, args.num_requests
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(
        base_url=normalize_base_url(target.base_url),
        timeout=args.timeout,
        headers=headers,
    ) as client:

        async def bounded(prompt: str) -> RequestMetrics:
            async with semaphore:
                return await run_one(
                    client,
                    target.model,
                    prompt,
                    args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    min_tokens=args.min_tokens,
                    ignore_eos=args.ignore_eos,
                )

        wall_start_ns = time.perf_counter_ns()
        results = await asyncio.gather(*(bounded(p) for p in prompts))
        wall_ns = time.perf_counter_ns() - wall_start_ns

    summary, samples = summarize_results(
        results,
        wall_ns=wall_ns,
        dataset_label=dataset_label,
        ttft_slo_s=args.ttft_slo_s,
    )
    print(
        f"target={normalize_base_url(target.base_url)} "
        f"model={target.model} dataset={dataset_label}"
    )
    print(
        f"requests={len(results)} concurrency={args.concurrency} "
        f"wall={summary['wall_s']:.2f}s"
    )
    print(
        f"TTFT p50={summary['ttft_p50_ms']}ms p99={summary['ttft_p99_ms']}ms"
    )
    if summary["tpot_mean_ms"] is not None:
        print(
            f"TPOT mean={summary['tpot_mean_ms']}ms/token "
            f"({summary['tpot_method']}-granularity)"
        )
    print(
        f"throughput={summary['throughput_rps']} req/s; "
        f"output={summary['output_tokens_per_s']} token/s; "
        f"goodput(TTFT<={args.ttft_slo_s}s)={summary['goodput_rps']} req/s"
    )
    if args.results_dir:
        stamp = _datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        results_dir = Path(args.results_dir)
        out = results_dir / f"{stamp}-serving.json"  # timestamped: same-day safe
        atomic_write_json(
            out,
            {
                "config": build_run_config(args),
                "summary": summary,
                "samples": samples,
            },
        )
        print(f"results written to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=None,
        help=f"{TARGET_SPEC_FORMAT}; cannot be combined with split endpoint flags",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None, help="ShareGPT-format JSON path")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Explicit sampling temperature (omitted by default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Explicit per-request sampling seed (omitted by default)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Minimum completion length (omitted by default)",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force generation to the requested max_tokens",
    )
    parser.add_argument("--ttft-slo-s", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the target bearer token",
    )
    auth.add_argument(
        "--api-key",
        default=None,
        help=(
            "Deprecated literal bearer token; may be visible in process arguments. "
            "Prefer --api-key-env"
        ),
    )
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
