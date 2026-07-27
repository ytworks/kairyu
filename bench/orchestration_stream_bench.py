"""P-B1 streaming-orchestrator acceptance and responsibility-boundary A/B.

The real-model gate compares one gateway's direct model with ``kairyu-auto``
wrapping the same backend.  Requests are paired and their order alternates so
warm-cache and time drift affect both paths.  SSE comments are excluded from
TTFT; only the first non-empty assistant content delta starts the clock.

The isolated architecture A/B compares the selected pull-through iterator with
a task + bounded-queue bridge.  It measures transport overhead only and is
reported separately from the Qwen3-32B end-to-end result.

Example:
    uv run python bench/orchestration_stream_bench.py \
      --base-url http://127.0.0.1:8002 \
      --direct-model qwen3-32b --auto-model kairyu-auto \
      --pairs 24 --max-tokens 64 \
      --result bench/results/orchestration-stream-qwen3-32b-tp8.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

_DONE = object()


@dataclass(frozen=True)
class RequestMeasurement:
    model: str
    order: int
    prompt_sha256: str
    ttft_ms: float
    total_ms: float
    content_chunks: int
    completion_tokens: int | None
    status_comments: int
    trace_comments: int


async def measure_request(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt: str,
    order: int,
    max_tokens: int,
) -> RequestMeasurement:
    started = time.perf_counter()
    first_content_at: float | None = None
    chunks = 0
    completion_tokens = None
    status_comments = 0
    trace_comments = 0
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Kairyu-Trace": "1"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith(": status"):
                status_comments += 1
                continue
            if line.startswith(": trace"):
                trace_comments += 1
                continue
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line.removeprefix("data: "))
            if "error" in payload:
                raise RuntimeError(f"{model} stream failed: {payload['error']}")
            if payload.get("usage"):
                completion_tokens = payload["usage"].get("completion_tokens")
            for choice in payload.get("choices", []):
                content = (choice.get("delta") or {}).get("content")
                if content:
                    chunks += 1
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
    finished = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError(f"{model} emitted no content delta")
    return RequestMeasurement(
        model=model,
        order=order,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        ttft_ms=(first_content_at - started) * 1_000,
        total_ms=(finished - started) * 1_000,
        content_chunks=chunks,
        completion_tokens=completion_tokens,
        status_comments=status_comments,
        trace_comments=trace_comments,
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * percentile), len(ordered) - 1)]


def summarize_requests(
    measurements: list[RequestMeasurement],
) -> dict[str, float | int]:
    ttfts = [item.ttft_ms for item in measurements]
    totals = [item.total_ms for item in measurements]
    completion_tokens = [
        item.completion_tokens
        for item in measurements
        if item.completion_tokens is not None
    ]
    return {
        "requests": len(measurements),
        "ttft_p50_ms": round(statistics.median(ttfts), 3),
        "ttft_p95_ms": round(_percentile(ttfts, 0.95), 3),
        "ttft_p99_ms": round(_percentile(ttfts, 0.99), 3),
        "total_p50_ms": round(statistics.median(totals), 3),
        "completion_tokens_min": min(completion_tokens) if completion_tokens else 0,
        "completion_tokens_max": max(completion_tokens) if completion_tokens else 0,
        "status_comments": sum(item.status_comments for item in measurements),
        "trace_comments": sum(item.trace_comments for item in measurements),
    }


async def run_endpoint_ab(args: argparse.Namespace) -> dict:
    prompts = [
        f"Reply with one concise sentence describing radix prefix caching. Case {index}."
        for index in range(args.pairs)
    ]
    by_model: dict[str, list[RequestMeasurement]] = {
        args.direct_model: [],
        args.auto_model: [],
    }
    async with httpx.AsyncClient(
        base_url=args.base_url,
        timeout=args.timeout,
    ) as client:
        for warmup in range(args.warmup):
            prompt = f"Warmup {warmup}: reply with the word ready."
            for model in (args.direct_model, args.auto_model):
                await measure_request(
                    client,
                    model=model,
                    prompt=prompt,
                    order=warmup,
                    max_tokens=min(args.max_tokens, 8),
                )
        for index, prompt in enumerate(prompts):
            order = (
                (args.direct_model, args.auto_model)
                if index % 2 == 0
                else (args.auto_model, args.direct_model)
            )
            for position, model in enumerate(order):
                measurement = await measure_request(
                    client,
                    model=model,
                    prompt=prompt,
                    order=position,
                    max_tokens=args.max_tokens,
                )
                by_model[model].append(measurement)
                print(
                    f"[{index + 1}/{args.pairs}] {model} "
                    f"TTFT={measurement.ttft_ms:.2f}ms "
                    f"total={measurement.total_ms:.2f}ms",
                    flush=True,
                )

    direct = summarize_requests(by_model[args.direct_model])
    auto = summarize_requests(by_model[args.auto_model])
    ratios = {
        "ttft_p50_auto_over_direct": round(
            auto["ttft_p50_ms"] / direct["ttft_p50_ms"], 4
        ),
        "ttft_p99_auto_over_direct": round(
            auto["ttft_p99_ms"] / direct["ttft_p99_ms"], 4
        ),
    }
    passed = all(value <= args.max_ttft_ratio for value in ratios.values())
    total_latency_comparable = (
        direct["completion_tokens_min"] == auto["completion_tokens_min"]
        and direct["completion_tokens_max"] == auto["completion_tokens_max"]
    )
    return {
        "config": {
            "base_url": args.base_url,
            "direct_model": args.direct_model,
            "auto_model": args.auto_model,
            "pairs": args.pairs,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            "pair_order": "alternating",
            "ttft_definition": "first non-empty assistant content delta",
            "max_ttft_ratio": args.max_ttft_ratio,
        },
        "direct": direct,
        "auto": auto,
        "ratios": ratios,
        "passed": passed,
        "total_latency_comparable": total_latency_comparable,
        "interpretation": (
            "P-B1 gates TTFT only. Total latency is not comparable because "
            "the direct and AUTO paths returned different output-token counts; "
            "AUTO request sampling propagation is tracked by issue #208."
            if not total_latency_comparable
            else "Direct and AUTO returned the same output-token range."
        ),
        "samples": {
            model: [asdict(item) for item in measurements]
            for model, measurements in by_model.items()
        },
    }


async def _token_source(chunks: int):
    for index in range(chunks):
        yield index


async def _consume_pull_through(chunks: int) -> int:
    consumed = 0
    async for _ in _token_source(chunks):
        consumed += 1
    return consumed


async def _consume_queue_bridge(chunks: int) -> int:
    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)

    async def produce() -> None:
        try:
            async for item in _token_source(chunks):
                await queue.put(item)
        finally:
            await queue.put(_DONE)

    producer = asyncio.create_task(produce())
    consumed = 0
    try:
        while await queue.get() is not _DONE:
            consumed += 1
    finally:
        await producer
    return consumed


async def run_transport_ab(*, chunks: int, rounds: int) -> dict:
    samples: dict[str, list[float]] = {"pull_through": [], "queue_bridge": []}
    for round_index in range(rounds):
        order = (
            ("pull_through", "queue_bridge")
            if round_index % 2 == 0
            else ("queue_bridge", "pull_through")
        )
        for candidate in order:
            started = time.perf_counter_ns()
            if candidate == "pull_through":
                consumed = await _consume_pull_through(chunks)
            else:
                consumed = await _consume_queue_bridge(chunks)
            elapsed = time.perf_counter_ns() - started
            if consumed != chunks:
                raise RuntimeError(
                    f"{candidate} consumed {consumed} of {chunks} events"
                )
            samples[candidate].append(elapsed / chunks)
    medians = {
        candidate: round(statistics.median(values), 2)
        for candidate, values in samples.items()
    }
    return {
        "scope": "Python async event transport only; no model inference",
        "chunks_per_round": chunks,
        "rounds": rounds,
        "median_ns_per_event": medians,
        "queue_over_pull_through": round(
            medians["queue_bridge"] / medians["pull_through"],
            4,
        ),
        "selected": min(medians, key=medians.get),
        "samples_ns_per_event": {
            candidate: [round(value, 2) for value in values]
            for candidate, values in samples.items()
        },
    }


async def async_main(args: argparse.Namespace) -> bool:
    transport = await run_transport_ab(
        chunks=args.transport_chunks,
        rounds=args.transport_rounds,
    )
    endpoint = await run_endpoint_ab(args)
    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "hardware_model": args.hardware_model,
        "transport_ab": transport,
        "endpoint_ab": endpoint,
    }
    output = Path(args.result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "transport selected="
        f"{transport['selected']} queue/pull="
        f"{transport['queue_over_pull_through']:.3f}x"
    )
    print(
        "endpoint ratios "
        f"p50={endpoint['ratios']['ttft_p50_auto_over_direct']:.3f}x "
        f"p99={endpoint['ratios']['ttft_p99_auto_over_direct']:.3f}x "
        f"passed={endpoint['passed']}"
    )
    print(f"result={output}")
    return bool(endpoint["passed"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--direct-model", default="qwen3-32b")
    parser.add_argument("--auto-model", default="kairyu-auto")
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-ttft-ratio", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--transport-chunks", type=int, default=100_000)
    parser.add_argument("--transport-rounds", type=int, default=7)
    parser.add_argument("--hardware-model", default="Qwen/Qwen3-32B TP8")
    parser.add_argument(
        "--result",
        default="bench/results/orchestration-stream-qwen3-32b-tp8.json",
    )
    return parser


def main() -> None:
    passed = asyncio.run(async_main(build_parser().parse_args()))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
