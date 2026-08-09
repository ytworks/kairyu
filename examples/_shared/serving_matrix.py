#!/usr/bin/env python3
"""Pinned ShareGPT serving matrix with exact tokenizer-visible length bins."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from kairyu.models.tokenizer_metadata import load_tokenizer_chat_metadata

from kairyu.bench.reporting import atomic_write_json, atomic_write_text, nearest_rank_percentile
from kairyu.entrypoints.chat_template import ChatTemplate

SHAREGPT_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
LENGTHS = (1_024, 8_192, 65_536)


@dataclass(frozen=True)
class Sample:
    ttft_s: float
    tpot_s: float | None
    output_tokens: int
    elapsed_s: float
    error: str | None = None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer(repo: str, revision: str):
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    directory = Path(
        snapshot_download(
            repo,
            revision=revision,
            allow_patterns=("tokenizer*", "*.jinja", "vocab*", "merges.txt"),
        )
    )
    metadata = load_tokenizer_chat_metadata(directory, include_templates=True)
    if "default" not in metadata.templates:
        raise RuntimeError("checkpoint tokenizer has no default chat template")
    template = ChatTemplate.from_tokenizer_metadata(
        metadata.templates,
        metadata.special_tokens,
    )
    return Tokenizer.from_file(str(directory / "tokenizer.json")), template, directory


def _exact_prompt(tokenizer, template: ChatTemplate, target: int, seed: str) -> tuple[str, int]:
    prefix = f"Pinned ShareGPT seed material: {seed}\n"

    def measured(content: str) -> int:
        rendered = template.render([{"role": "user", "content": content}])
        return len(tokenizer.encode(rendered, add_special_tokens=False).ids)

    unit = " grass"
    base = measured(prefix)
    if base > target:
        raise RuntimeError(f"chat template overhead {base} exceeds length bin {target}")
    low, high = 0, target - base + 1024
    while low <= high:
        middle = (low + high) // 2
        value = measured(prefix + unit * middle)
        if value < target:
            low = middle + 1
        elif value > target:
            high = middle - 1
        else:
            content = prefix + unit * middle
            return content, value
    # Tokenizer boundary merges may make one unit skip an integer. Search a
    # small deterministic suffix alphabet without changing the length target.
    body = prefix + unit * max(0, high)
    for suffix in (" x", " 0", " a", " あ", ".", "\n"):
        candidate = body
        for _ in range(32):
            value = measured(candidate)
            if value == target:
                return candidate, value
            if value > target:
                break
            candidate += suffix
    raise RuntimeError(f"could not construct an exact {target}-token rendered prompt")


async def _request(client: httpx.AsyncClient, url: str, model: str, prompt: str) -> Sample:
    started = time.perf_counter()
    first = None
    last = None
    output_tokens = 0
    try:
        async with client.stream(
            "POST",
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 128,
                "reasoning_effort": "max",
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                usage = payload.get("usage") or {}
                if isinstance(usage.get("completion_tokens"), int):
                    output_tokens = usage["completion_tokens"]
                choices = payload.get("choices") or ()
                delta = choices[0].get("delta", {}) if choices else {}
                emitted = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if emitted:
                    now = time.perf_counter()
                    first = first or now
                    last = now
        ended = time.perf_counter()
        ttft = (first or ended) - started
        tpot = None
        if first is not None and last is not None and output_tokens > 1:
            tpot = (last - first) / (output_tokens - 1)
        return Sample(ttft, tpot, output_tokens, ended - started)
    except Exception as error:
        ended = time.perf_counter()
        return Sample(ended - started, None, 0, ended - started, type(error).__name__)


async def _cell(url: str, model: str, prompts: list[str], concurrency: int) -> dict:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=3600, limits=limits) as client:
        started = time.perf_counter()
        samples = await asyncio.gather(
            *(_request(client, url, model, prompts[index]) for index in range(concurrency))
        )
        span = time.perf_counter() - started
    successful = [sample for sample in samples if sample.error is None]
    ttft = sorted(sample.ttft_s for sample in successful)
    tpot = sorted(sample.tpot_s for sample in successful if sample.tpot_s is not None)
    output = sum(sample.output_tokens for sample in successful)
    return {
        "concurrency": concurrency,
        "requests": len(samples),
        "errors": len(samples) - len(successful),
        "ttft_p50_s": nearest_rank_percentile(ttft, 0.50) if ttft else None,
        "ttft_p99_s": nearest_rank_percentile(ttft, 0.99) if ttft else None,
        "tpot_p99_s": nearest_rank_percentile(tpot, 0.99) if tpot else None,
        "completed_output_tokens_per_s": output / span if span else None,
        "samples": [asdict(sample) for sample in samples],
    }


async def run(args) -> int:
    dataset = Path(args.dataset)
    if _sha(dataset) != SHAREGPT_SHA256:
        raise RuntimeError("ShareGPT dataset SHA-256 mismatch")
    records = json.loads(dataset.read_text(encoding="utf-8"))
    source_hashes = [
        hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        for row in records[:64]
    ]
    tokenizer, template, tokenizer_dir = _tokenizer(args.repo, args.revision)
    lengths = (*LENGTHS, args.native_max)
    exact: dict[int, list[str]] = {}
    for length in lengths:
        exact[length] = [
            _exact_prompt(tokenizer, template, length, source_hashes[index])[0]
            for index in range(64)
        ]
    cells = []
    for length in lengths:
        for prefix_reuse in (False, True):
            prompts = exact[length]
            if prefix_reuse:
                # Identical, tokenizer-attested prompts are the unambiguous
                # full-prefix-reuse arm. The non-reuse arm has 64 distinct
                # source hashes; no boundary-changing string surgery is used.
                prompts = [prompts[0]] * len(prompts)
            for concurrency in CONCURRENCY:
                cell = await _cell(args.base_url, args.model, prompts, concurrency)
                cell.update(
                    {
                        "rendered_context_tokens": length,
                        "prefix_reuse": prefix_reuse,
                    }
                )
                cells.append(cell)
    payload = {
        "schema_version": 1,
        "model": args.model,
        "repo": args.repo,
        "revision": args.revision,
        "native_max": args.native_max,
        "sharegpt_sha256": SHAREGPT_SHA256,
        "tokenizer_dir": str(tokenizer_dir),
        "cells": cells,
        "passed": all(cell["errors"] == 0 for cell in cells),
    }
    output = Path(args.output)
    atomic_write_json(output, payload)
    atomic_write_text(
        output.with_suffix(".md"),
        "# Serving matrix\n\n"
        f"Status: **{'PASS' if payload['passed'] else 'FAIL'}**\n\n"
        "Every cell records exact rendered token length, raw request samples, "
        "TTFT, TPOT and goodput.\n",
    )
    return 0 if payload["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--native-max", required=True, type=int)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
