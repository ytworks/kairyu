"""G6 P-B4: controlled direct-versus-AUTO serving-performance evidence.

The runner alternates a direct Qwen request with ``kairyu-auto`` on the same
gateway, records TTFT and request latency, and rejects ratios above the declared
limit. Model and product quality evaluation belongs to :mod:`evals`.

Example:
    .venv/bin/python verification/product/performance/tiered_auto_bench.py \
      --base-url http://127.0.0.1:8002/v1 \
      --result verification/results/tiered-auto-qwen3-32b-tp8.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from verification.orchestration.correctness.orchestration_stream_bench import (
    measure_request,
    summarize_requests,
)

SCHEMA_VERSION = 1


async def run_latency(args: argparse.Namespace, client: httpx.AsyncClient) -> dict:
    prompts = [
        f"Reply with one concise sentence describing radix prefix caching. Case {index}."
        for index in range(args.latency_pairs)
    ]
    by_model = {args.direct_model: [], args.auto_model: []}
    for warmup in range(args.latency_warmup):
        prompt = f"Warmup {warmup}: reply with the word ready."
        for model in (args.direct_model, args.auto_model):
            await measure_request(
                client,
                model=model,
                prompt=prompt,
                order=warmup,
                max_tokens=min(args.latency_max_tokens, 8),
            )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(index: int, prompt: str, position: int, model: str):
        async with semaphore:
            measured = await measure_request(
                client,
                model=model,
                prompt=prompt,
                order=position,
                max_tokens=args.latency_max_tokens,
            )
        print(
            f"[latency {index + 1}/{args.latency_pairs}] {model} TTFT={measured.ttft_ms:.2f}ms",
            flush=True,
        )
        return model, measured

    jobs = []
    for index, prompt in enumerate(prompts):
        order = (
            (args.direct_model, args.auto_model)
            if index % 2 == 0
            else (args.auto_model, args.direct_model)
        )
        jobs.extend(run_one(index, prompt, position, model) for position, model in enumerate(order))
    for model, measured in await asyncio.gather(*jobs):
        by_model[model].append(measured)

    direct = summarize_requests(by_model[args.direct_model])
    auto = summarize_requests(by_model[args.auto_model])
    ratios = {
        "ttft_p50_auto_over_direct": round(auto["ttft_p50_ms"] / direct["ttft_p50_ms"], 4),
        "ttft_p99_auto_over_direct": round(auto["ttft_p99_ms"] / direct["ttft_p99_ms"], 4),
    }
    return {
        "config": {
            "external_concurrency": args.concurrency,
            "pairs": args.latency_pairs,
            "warmup": args.latency_warmup,
            "max_tokens": args.latency_max_tokens,
            "pair_order": "alternating",
            "ttft_definition": "first non-empty assistant content delta",
            "max_ttft_ratio": args.max_ttft_ratio,
        },
        "direct": direct,
        "auto": auto,
        "ratios": ratios,
        "passed": all(value <= args.max_ttft_ratio for value in ratios.values()),
        "samples": {model: [asdict(item) for item in items] for model, items in by_model.items()},
    }


async def run(args: argparse.Namespace) -> dict:
    timeout = httpx.Timeout(args.timeout)
    server_root = args.base_url.rstrip("/")
    if server_root.endswith("/v1"):
        server_root = server_root[:-3]
    async with httpx.AsyncClient(base_url=server_root, timeout=timeout) as client:
        models_response = await client.get("/v1/models")
        models_response.raise_for_status()
        served_models = sorted(model["id"] for model in models_response.json()["data"])
        required_models = {args.direct_model, args.auto_model}
        discovery_passed = required_models <= set(served_models)
        if not discovery_passed:
            missing = sorted(required_models - set(served_models))
            raise RuntimeError(f"gateway is missing required models: {missing}")
        latency = await run_latency(args, client)

    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(tz=UTC).isoformat(),
        "issue": 198,
        "revalidation_issue": 208,
        "hardware": args.hardware,
        "served_models": served_models,
        "required_models": sorted(required_models),
        "model_discovery_passed": discovery_passed,
        "latency": latency,
        "passed": discovery_passed and latency["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--direct-model", default="qwen3-32b")
    parser.add_argument("--auto-model", default="kairyu-auto")
    parser.add_argument("--latency-pairs", type=int, default=12)
    parser.add_argument("--latency-warmup", type=int, default=2)
    parser.add_argument("--latency-max-tokens", type=int, default=64)
    parser.add_argument("--max-ttft-ratio", type=float, default=1.5)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--hardware",
        default="8x NVIDIA RTX PRO 6000 Blackwell Server Edition; Qwen3-32B TP8",
    )
    parser.add_argument(
        "--result",
        default="verification/results/tiered-auto-qwen3-32b-tp8.json",
    )
    args = parser.parse_args()
    for name in (
        "latency_pairs",
        "latency_max_tokens",
        "concurrency",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("latency_warmup",):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.max_ttft_ratio <= 0 or args.timeout <= 0:
        parser.error("--max-ttft-ratio and --timeout must be > 0")
    return args


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({key: result[key] for key in ("passed", "served_models")}, indent=2))
    print(f"wrote {path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
