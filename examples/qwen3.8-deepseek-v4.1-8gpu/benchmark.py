"""Streaming chat measurements that keep the evidence the gates need.

The shared serving benchmark records first visible content as TTFT but no
per-sample finish reason, no reasoning-effort field, and no raw trace. This
example-local client records, per request: the first model delta (reasoning
or content), the first visible content, completion, the finish reason, the
usage block, the public content, and — when asked — the raw Kairyu trace v2
object, so verification can prove which roles ran and how many tokens each
consumed. It sends requests; it never reorders, audits, or retries them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from pathlib import Path

import httpx

TRACE_HEADER = {"X-Kairyu-Trace": "1"}


async def collect(lines, start: float) -> dict:
    first_model = first_content = None
    content = ""
    reasoning_chars = 0
    tool_calls: dict[int, dict[str, list[str]]] = {}
    finish = usage = trace = None
    done = 0
    for_events = 0
    async for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            done += 1
            continue
        if done:
            raise ValueError("JSON after terminal SSE marker")
        chunk = json.loads(raw)
        for_events += 1
        if "error" in chunk:
            raise ValueError(f"Stream error: {chunk['error']}")
        usage = chunk.get("usage") or usage
        if "kairyu_trace_v2" in chunk:
            trace = chunk["kairyu_trace_v2"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            think = delta.get("reasoning_content") or delta.get("reasoning") or ""
            now = time.perf_counter() - start
            if (text or think) and first_model is None:
                first_model = now
            if text and first_content is None:
                first_content = now
            content += text
            reasoning_chars += len(think)
            for call in delta.get("tool_calls") or ():
                slot = tool_calls.setdefault(
                    int(call.get("index", 0)), {"name": [], "arguments": []}
                )
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"].append(function["name"])
                if function.get("arguments"):
                    slot["arguments"].append(function["arguments"])
                if first_model is None:
                    first_model = now
            finish = choice.get("finish_reason") or finish
    if done != 1:
        raise ValueError("Stream ended without exactly one terminal SSE marker")
    if usage is None:
        raise ValueError("Stream reported no usage")
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or tokens < 0:
        raise ValueError("Missing completion-token count")
    elapsed = time.perf_counter() - start
    calls = [
        {"name": "".join(slot["name"]), "arguments": "".join(slot["arguments"])}
        for _, slot in sorted(tool_calls.items())
    ]
    return {
        "model_ttft_ms": first_model * 1000 if first_model is not None else None,
        "content_ttft_ms": first_content * 1000 if first_content is not None else None,
        "total_ms": elapsed * 1000,
        "tpot_ms": (
            (elapsed - first_model) * 1000 / (tokens - 1)
            if first_model is not None and tokens > 1
            else None
        ),
        "completion_tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "usage": usage,
        "finish_reason": finish,
        "content": content,
        "reasoning_chars": reasoning_chars,
        "tool_calls": calls,
        "trace": trace,
        "sse_events": for_events,
    }


def sample_passed(result: dict, args) -> str | None:
    """Why a sample fails, or None. Fixed-length rows must reach max_tokens;
    natural rows must end with stop/tool_calls and carry visible output."""

    if args.ignore_eos:
        if result["completion_tokens"] != args.max_tokens:
            return "fixed-length row did not reach max_tokens"
        return None
    if result["finish_reason"] not in {"stop", "tool_calls"}:
        return f"finish_reason {result['finish_reason']!r}"
    if not result["content"].strip() and not result["tool_calls"]:
        return "empty visible output"
    if result["content"].strip() and result["content_ttft_ms"] is None:
        return "content without a content timestamp"
    return None


def percentile(values: list[float], fraction: float) -> float | None:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def request_body(args, row: dict) -> dict:
    body = {
        "model": args.model,
        "messages": row.get("messages")
        or [{"role": "user", "content": row["conversations"][0]["value"]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.min_tokens is not None:
        body["min_tokens"] = args.min_tokens
    if args.ignore_eos:
        body["ignore_eos"] = True
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    if row.get("tools"):
        body["tools"] = row["tools"]
    if row.get("response_format"):
        body["response_format"] = row["response_format"]
    return body


async def measure(args) -> int:
    rows = json.loads(args.dataset.read_text())[: args.num_requests]
    if len(rows) != args.num_requests:
        raise ValueError("Dataset contains fewer requests than requested")
    semaphore = asyncio.Semaphore(args.concurrency)
    headers = TRACE_HEADER if args.stage_trace else None
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:

        async def run(index, row):
            async with semaphore:
                start = time.perf_counter()
                try:
                    body = request_body(args, row)
                    async with client.stream(
                        "POST", "chat/completions", json=body, headers=headers
                    ) as response:
                        if response.status_code != 200:
                            detail = (await response.aread()).decode("utf-8", "replace")
                            raise ValueError(f"HTTP {response.status_code}: {detail[:400]}")
                        result = await collect(response.aiter_lines(), start)
                        result["request_id"] = response.headers.get("x-request-id")
                    error = sample_passed(result, args)
                    return {"index": index, "passed": error is None, "error": error, **result}
                except Exception as error:  # noqa: BLE001 - every failure is evidence
                    return {
                        "index": index,
                        "passed": False,
                        "error": str(error),
                        "total_ms": (time.perf_counter() - start) * 1000,
                    }

        start = time.perf_counter()
        samples = await asyncio.gather(*(run(i, row) for i, row in enumerate(rows)))
        elapsed = time.perf_counter() - start
    good = [row for row in samples if row["passed"]]
    tokens = sum(row["completion_tokens"] for row in good)
    orchestration_tokens = sum(
        (row.get("usage") or {}).get("orchestration_output_tokens") or 0 for row in good
    )
    summary = {
        "label": args.label,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "requests": len(samples),
        "successful_requests": len(good),
        "success_rate": len(good) / len(samples),
        "wall_s": elapsed,
        "completion_tokens_total": tokens,
        "orchestration_output_tokens_total": orchestration_tokens,
        "output_tokens_per_s": tokens / elapsed,
        "requests_per_s": len(good) / elapsed,
        "concurrency": args.concurrency,
        "tensor_parallel": args.tensor_parallel,
        "dp_replicas": args.dp_replicas,
        "finish_reasons": dict(Counter(row.get("finish_reason") for row in samples)),
    }
    for field in ("model_ttft_ms", "content_ttft_ms", "total_ms", "tpot_ms"):
        values = [row[field] for row in good if row.get(field) is not None]
        summary[field] = {
            "p50": percentile(values, 0.5),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
            "samples": len(values),
        }
    report = {
        "schema_version": 1,
        "measurement": (
            "fixed_length_including_reasoning" if args.ignore_eos else "completed_answers"
        ),
        "percentile_method": "nearest_rank",
        "summary": summary,
        "samples": samples,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "row-serving.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "finish_reasons"}, indent=2))
    return int(len(good) != len(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--stage-trace", action="store_true")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--dp-replicas", type=int, default=1)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    if min(args.num_requests, args.concurrency, args.max_tokens) < 1:
        parser.error("request, concurrency and token limits must be positive")
    raise SystemExit(asyncio.run(measure(args)))


if __name__ == "__main__":
    main()
