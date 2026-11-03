"""Streaming load generator for this example's serving rows.

Every request is a streamed OpenAI Responses request to Kairyu's public
``/v1/responses`` (m11 D7 amendment 2026-10-17). Two measurements:

* ``capped``: each request ends naturally within ``--max-tokens`` output
  tokens; the Responses API has no ``ignore_eos``/``min_tokens``, so rows are
  not fixed-length and are not comparable with the earlier fixed-length chat
  rows. Throughput includes reasoning tokens; the first model delta
  (reasoning or content) is the model TTFT, and first visible content is
  recorded separately (null when a row ends inside reasoning).
* ``completed``: natural completion at the default effort; a request passes
  only when it completes with visible content. This is what a user waits for.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx

_REASONING_DELTAS = {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}
_TERMINAL = {"response.completed", "response.incomplete"}


async def collect(lines, start: float) -> dict:
    """Fold one Responses event stream into its timing and terminal state."""
    first_model = first_content = None
    content = reasoning = ""
    final: dict | None = None
    async for line in lines:
        if not line.startswith("data:") or not line[5:].strip():
            continue
        if final is not None:
            raise ValueError("event after the terminal Responses event")
        event = json.loads(line[5:])
        kind = event.get("type")
        if kind == "error":
            raise ValueError(f"stream error: {event}")
        if kind == "response.failed":
            raise ValueError(f"stream error: {(event.get('response') or {}).get('error')}")
        text = (event.get("delta") or "") if kind == "response.output_text.delta" else ""
        trace = (event.get("delta") or "") if kind in _REASONING_DELTAS else ""
        now = time.perf_counter() - start
        if (text or trace) and first_model is None:
            first_model = now
        if text and first_content is None:
            first_content = now
        content += text
        reasoning += trace
        if kind in _TERMINAL:
            final = event.get("response") or {}
    usage = (final or {}).get("usage")
    reason = ((final or {}).get("incomplete_details") or {}).get("reason")
    if final is None or not usage or first_model is None:
        raise ValueError("incomplete stream, missing model output, or absent usage")
    if final.get("status") == "completed":
        finish = "stop"
    elif reason == "max_output_tokens":
        finish = "length"
    else:
        raise ValueError(f"response ended {final.get('status')!r} ({reason!r})")
    tokens = usage.get("output_tokens")
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("missing positive completion-token count")
    elapsed = time.perf_counter() - start
    return {
        "model_ttft_ms": first_model * 1000,
        "content_ttft_ms": first_content * 1000 if first_content is not None else None,
        "total_ms": elapsed * 1000,
        "tpot_ms": (elapsed - first_model) * 1000 / (tokens - 1) if tokens > 1 else None,
        "completion_tokens": tokens,
        "prompt_tokens": usage.get("input_tokens"),
        "finish_reason": finish,
        "content": content,
        "reasoning_chars": len(reasoning),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile."""
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def sample_error(result: dict, *, mode: str, max_tokens: int) -> str | None:
    if mode == "capped":
        if result["completion_tokens"] > max_tokens:
            return f"generated {result['completion_tokens']} tokens, above the {max_tokens} cap"
        return None
    if result["finish_reason"] != "stop":
        return f"finish_reason {result['finish_reason']!r} (no completed answer)"
    if not result["content"].strip():
        return "completed without visible content"
    return None


async def measure(args) -> int:
    rows = json.loads(args.dataset.read_text())[: args.num_requests]
    if len(rows) != args.num_requests:
        raise ValueError("dataset holds fewer requests than requested")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/",
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:

        async def run(index: int, row: dict) -> dict:
            async with semaphore:
                start = time.perf_counter()
                body = {
                    "model": args.model,
                    "input": row["prompt"],
                    "stream": True,
                    "store": False,
                    "max_output_tokens": args.max_tokens,
                }
                try:
                    async with client.stream("POST", "responses", json=body) as response:
                        response.raise_for_status()
                        result = await collect(response.aiter_lines(), start)
                        result["request_id"] = response.headers.get("x-request-id")
                except Exception as error:  # noqa: BLE001 - recorded per sample
                    return {"index": index, "passed": False, "error": str(error)}
                error = sample_error(result, mode=args.mode, max_tokens=args.max_tokens)
                return {"index": index, "passed": error is None, "error": error, **result}

        start = time.perf_counter()
        samples = await asyncio.gather(*(run(i, row) for i, row in enumerate(rows)))
        elapsed = time.perf_counter() - start
    good = [row for row in samples if row["passed"]]
    tokens = sum(row["completion_tokens"] for row in good)
    summary = {
        "api": "responses",
        "mode": args.mode,
        "requests": len(samples),
        "successful_requests": len(good),
        "wall_s": elapsed,
        "completion_tokens_total": tokens,
        "output_tokens_per_s": tokens / elapsed,
        "concurrency": args.concurrency,
    }
    for field in ("model_ttft_ms", "content_ttft_ms", "total_ms", "tpot_ms"):
        values = [row[field] for row in good if row.get(field) is not None]
        summary[field] = {
            "p50": percentile(values, 0.5),
            "p99": percentile(values, 0.99),
            "samples": len(values),
        }
    report = {
        "schema_version": 1,
        "percentile_method": "nearest_rank",
        "summary": summary,
        "samples": samples,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "row.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return int(len(good) != len(samples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=("capped", "completed"), required=True)
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_requests, args.concurrency, args.max_tokens) < 1:
        parser.error("request, concurrency and token limits must be positive")
    raise SystemExit(asyncio.run(measure(args)))


if __name__ == "__main__":
    main()
