"""G6 P-B4: controlled latency/quality evidence for tiered AUTO models.

The latency half alternates a direct Qwen request with ``kairyu-auto`` on the
same gateway. The quality half uses a fixed eight-item LiveCodeBench slice:
items were sampled with seed 198 from the 334 release-v6 prompts that the
default RuleRouter sends to ``multi_agent``. Both AUTO tiers therefore receive
the same prompts and differ only in their configured orchestration:

* ``kairyu-auto``: planner -> worker -> verifier -> synthesizer (Conductor)
* ``kairyu-auto-max``: three parallel proposals -> synthesizer (MoA)

Every quality request opts into the structured trace added by P-B2. The result
records objective sandbox scores, latency, internal call count, cumulative
internal tokens, allocated GPU-seconds, and response hashes.

Example:
    .venv/bin/python bench/tiered_auto_bench.py \
      --base-url http://127.0.0.1:8002/v1 \
      --result bench/results/tiered-auto-qwen3-32b-tp8-2026-07-27.json
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

from bench.orchestration_stream_bench import measure_request, summarize_requests
from kairyu.bench.adapters.base import (
    RunContext,
    extract_code_block,
    normalize_base_url,
)
from kairyu.bench.adapters.livecodebench import LiveCodeBenchAdapter, grade_code
from kairyu.bench.cache import BenchCache
from kairyu.bench.types import BenchItem, BenchTarget, ChatRequestSpec
from kairyu.orchestration.router import RuleRouter

SCHEMA_VERSION = 1
LIVE_CODE_BENCH_DATASET = "livecodebench/code_generation_lite"
LIVE_CODE_BENCH_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
QUALITY_SELECTION_SEED = 198
QUALITY_ELIGIBLE_ITEMS = 334
QUALITY_ITEM_IDS = (
    "3094",
    "3297",
    "3487",
    "abc349_d",
    "abc376_f",
    "abc382_b",
    "abc382_g",
    "abc395_f",
)


@dataclass(frozen=True)
class QualityMeasurement:
    item_id: str
    model: str
    order: int
    latency_s: float
    score: float
    error: str | None
    route: str | None
    internal_calls: int
    orchestration_input_tokens: int
    orchestration_output_tokens: int
    cached_tokens: int
    allocated_gpu_seconds: float
    estimated_cost_usd: float | None
    response_sha256: str
    response_excerpt: str


def _cache_paths(cache_root: str | Path) -> tuple[Path, Path]:
    root = Path(cache_root).expanduser() / "livecodebench"
    return root / "manifest.json", root / "data.jsonl"


def load_fixed_items(
    cache_root: str | Path,
    *,
    item_ids: tuple[str, ...] = QUALITY_ITEM_IDS,
) -> tuple[list[BenchItem], dict]:
    """Read only the fixed rows from the pinned normalized LCB cache."""

    manifest_path, data_path = _cache_paths(cache_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != LIVE_CODE_BENCH_DATASET:
        raise RuntimeError(
            f"unexpected LiveCodeBench dataset: {manifest.get('dataset')!r}"
        )
    if manifest.get("revision") != LIVE_CODE_BENCH_REVISION:
        raise RuntimeError(
            f"unexpected LiveCodeBench revision: {manifest.get('revision')!r}"
        )

    wanted = set(item_ids)
    found: dict[str, BenchItem] = {}
    with data_path.open(encoding="utf-8") as source:
        for line in source:
            # The normalized file puts ``id`` first. Avoid parsing the often
            # large private-test payload for every one of the 1,055 rows.
            prefix = line[:128]
            candidates = [
                item_id
                for item_id in wanted - found.keys()
                if f'"id": "{item_id}"' in prefix
            ]
            if not candidates:
                continue
            row = json.loads(line)
            item_id = row.pop("id")
            if item_id in wanted:
                found[item_id] = BenchItem(id=item_id, payload=row)
                if len(found) == len(wanted):
                    break
    missing = sorted(wanted - found.keys())
    if missing:
        raise RuntimeError(f"fixed LiveCodeBench items missing from cache: {missing}")
    return [found[item_id] for item_id in item_ids], manifest


def count_internal_calls(trace: dict | None) -> int:
    """Recover actual backend call count from the P-B2 structured trace."""

    events = list((trace or {}).get("events") or [])
    for event in events:
        # TraceEvent.metadata is named ``detail`` on the stable v2 wire schema.
        metadata = event.get("metadata") or event.get("detail") or {}
        proposals = metadata.get("proposals")
        if event.get("node") == "moa" and isinstance(proposals, int):
            return proposals + 1
    return sum(
        1
        for event in events
        if event.get("usage") is not None
        and (event.get("operation") or event.get("kind")) != "routing"
    )


def estimated_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> float | None:
    if input_usd_per_mtok == 0 and output_usd_per_mtok == 0:
        return None
    return (
        input_tokens * input_usd_per_mtok
        + output_tokens * output_usd_per_mtok
    ) / 1_000_000


def summarize_quality(
    measurements: list[QualityMeasurement],
) -> dict[str, float | int | None]:
    latencies = [item.latency_s for item in measurements]
    estimated = [
        item.estimated_cost_usd
        for item in measurements
        if item.estimated_cost_usd is not None
    ]
    return {
        "items": len(measurements),
        "passed_items": sum(item.score == 1.0 for item in measurements),
        "score": round(sum(item.score for item in measurements) / len(measurements), 6),
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_total_s": round(sum(latencies), 3),
        "internal_calls": sum(item.internal_calls for item in measurements),
        "orchestration_input_tokens": sum(
            item.orchestration_input_tokens for item in measurements
        ),
        "orchestration_output_tokens": sum(
            item.orchestration_output_tokens for item in measurements
        ),
        "cached_tokens": sum(item.cached_tokens for item in measurements),
        "allocated_gpu_seconds": round(
            sum(item.allocated_gpu_seconds for item in measurements), 3
        ),
        "estimated_cost_usd": round(sum(estimated), 8) if estimated else None,
    }


def _request_for(
    adapter: LiveCodeBenchAdapter,
    item: BenchItem,
    target: BenchTarget,
    context: RunContext,
) -> ChatRequestSpec:
    request = adapter.build_request(item, target, context)
    if not isinstance(request, ChatRequestSpec):
        raise RuntimeError(f"fixed item {item.id} was unexpectedly skipped")
    decision = RuleRouter().route(request.messages[0]["content"])
    if decision.target != "multi_agent":
        raise RuntimeError(
            f"fixed item {item.id} routes to {decision.target}, expected multi_agent"
        )
    return request


async def measure_quality(
    client: httpx.AsyncClient,
    *,
    adapter: LiveCodeBenchAdapter,
    context: RunContext,
    item: BenchItem,
    model: str,
    order: int,
    base_url: str,
    max_tokens: int,
    gpu_count: int,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> QualityMeasurement:
    target = BenchTarget(
        base_url=base_url,
        model=model,
        max_output_tokens=max_tokens,
    )
    request = _request_for(adapter, item, target, context)
    started = time.perf_counter()
    response = await client.post(
        f"{normalize_base_url(base_url)}/chat/completions",
        headers={"X-Kairyu-Trace": "1"},
        json={
            "model": model,
            "messages": list(request.messages),
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    latency_s = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"] or ""
    code = extract_code_block(text)
    if code is None:
        score, error = 0.0, "no code block in response"
    else:
        passed, detail = await asyncio.to_thread(
            grade_code,
            code,
            item.payload["tests"],
            item.payload.get("fn_name"),
        )
        score, error = (1.0 if passed else 0.0), (detail or None)

    usage = payload.get("usage") or {}
    input_tokens = int(usage.get("orchestration_input_tokens") or 0)
    output_tokens = int(usage.get("orchestration_output_tokens") or 0)
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    trace = payload.get("kairyu_trace_v2")
    route = (payload.get("kairyu_route") or {}).get("target")
    return QualityMeasurement(
        item_id=item.id,
        model=model,
        order=order,
        latency_s=round(latency_s, 6),
        score=score,
        error=error,
        route=route,
        internal_calls=count_internal_calls(trace),
        orchestration_input_tokens=input_tokens,
        orchestration_output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        allocated_gpu_seconds=round(latency_s * gpu_count, 6),
        estimated_cost_usd=estimated_cost_usd(
            input_tokens,
            output_tokens,
            input_usd_per_mtok=input_usd_per_mtok,
            output_usd_per_mtok=output_usd_per_mtok,
        ),
        response_sha256=hashlib.sha256(text.encode()).hexdigest(),
        response_excerpt=text[:500],
    )


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
    for index, prompt in enumerate(prompts):
        order = (
            (args.direct_model, args.auto_model)
            if index % 2 == 0
            else (args.auto_model, args.direct_model)
        )
        for position, model in enumerate(order):
            measured = await measure_request(
                client,
                model=model,
                prompt=prompt,
                order=position,
                max_tokens=args.latency_max_tokens,
            )
            by_model[model].append(measured)
            print(
                f"[latency {index + 1}/{args.latency_pairs}] {model} "
                f"TTFT={measured.ttft_ms:.2f}ms",
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
    return {
        "config": {
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
        "samples": {
            model: [asdict(item) for item in items] for model, items in by_model.items()
        },
    }


async def run(args: argparse.Namespace) -> dict:
    items, manifest = load_fixed_items(args.cache_dir)
    context = RunContext(
        cache=BenchCache(Path(args.cache_dir).expanduser()),
        http_factory=httpx.AsyncClient,
        request_timeout_s=args.timeout,
        retries=0,
        concurrency=1,
    )
    adapter = LiveCodeBenchAdapter()
    timeout = httpx.Timeout(args.timeout)
    server_root = args.base_url.rstrip("/")
    if server_root.endswith("/v1"):
        server_root = server_root[:-3]
    async with httpx.AsyncClient(base_url=server_root, timeout=timeout) as client:
        models_response = await client.get("/v1/models")
        models_response.raise_for_status()
        served_models = sorted(model["id"] for model in models_response.json()["data"])
        required_models = {args.direct_model, args.auto_model, args.auto_max_model}
        discovery_passed = required_models <= set(served_models)
        if not discovery_passed:
            missing = sorted(required_models - set(served_models))
            raise RuntimeError(
                f"gateway is missing required models: {missing}"
            )

        latency = await run_latency(args, client)
        measurements: list[QualityMeasurement] = []
        for index, item in enumerate(items):
            order = (
                (args.auto_model, args.auto_max_model)
                if index % 2 == 0
                else (args.auto_max_model, args.auto_model)
            )
            for position, model in enumerate(order):
                measured = await measure_quality(
                    client,
                    adapter=adapter,
                    context=context,
                    item=item,
                    model=model,
                    order=position,
                    base_url=args.base_url,
                    max_tokens=args.quality_max_tokens,
                    gpu_count=args.gpu_count,
                    input_usd_per_mtok=args.input_usd_per_mtok,
                    output_usd_per_mtok=args.output_usd_per_mtok,
                )
                measurements.append(measured)
                print(
                    f"[quality {index + 1}/{len(items)}] {model} item={item.id} "
                    f"score={measured.score:.0f} latency={measured.latency_s:.2f}s "
                    f"calls={measured.internal_calls}",
                    flush=True,
                )

    by_model = {
        model: [item for item in measurements if item.model == model]
        for model in (args.auto_model, args.auto_max_model)
    }
    quality = {
        "selection": {
            "benchmark": "LiveCodeBench release_v6",
            "dataset": LIVE_CODE_BENCH_DATASET,
            "revision": LIVE_CODE_BENCH_REVISION,
            "manifest_sha256": manifest.get("sha256"),
            "eligible_rule": "default RuleRouter target == multi_agent",
            "eligible_items": QUALITY_ELIGIBLE_ITEMS,
            "selection_seed": QUALITY_SELECTION_SEED,
            "item_ids": list(QUALITY_ITEM_IDS),
            "scoring": "pass@1; all public and private tests pass in local sandbox",
        },
        "config": {
            "attempts_per_item": 1,
            "temperature_requested": 0,
            "max_tokens_requested": args.quality_max_tokens,
            "effective_internal_sampling": (
                "Issue #208 remains open: AUTO does not yet forward request sampling. "
                "Conductor uses temperature=1.0/max_tokens=1024; MoA uses "
                "proposal temperature=0.9, synthesis temperature=0.3, max_tokens=1024."
            ),
            "pair_order": "alternating",
            "gpu_count": args.gpu_count,
            "input_usd_per_mtok": args.input_usd_per_mtok,
            "output_usd_per_mtok": args.output_usd_per_mtok,
            "estimated_cost_note": (
                "null when both rates are zero; token totals and allocated GPU-seconds "
                "remain the owned-hardware cost evidence"
            ),
        },
        "models": {
            model: {
                "summary": summarize_quality(model_measurements),
                "samples": [asdict(item) for item in model_measurements],
            }
            for model, model_measurements in by_model.items()
        },
    }
    auto_score = quality["models"][args.auto_model]["summary"]["score"]
    max_score = quality["models"][args.auto_max_model]["summary"]["score"]
    quality["score_delta_auto_max_minus_auto"] = round(max_score - auto_score, 6)
    quality["all_routes_multi_agent"] = all(
        item.route == "multi_agent" for item in measurements
    )
    quality["passed"] = (
        max_score > auto_score and quality["all_routes_multi_agent"]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(tz=UTC).isoformat(),
        "issue": 198,
        "hardware": args.hardware,
        "served_models": served_models,
        "required_models": sorted(required_models),
        "model_discovery_passed": discovery_passed,
        "latency": latency,
        "quality": quality,
        "passed": discovery_passed and latency["passed"] and quality["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--direct-model", default="qwen3-32b")
    parser.add_argument("--auto-model", default="kairyu-auto")
    parser.add_argument("--auto-max-model", default="kairyu-auto-max")
    parser.add_argument("--latency-pairs", type=int, default=12)
    parser.add_argument("--latency-warmup", type=int, default=2)
    parser.add_argument("--latency-max-tokens", type=int, default=64)
    parser.add_argument("--max-ttft-ratio", type=float, default=1.5)
    parser.add_argument("--quality-max-tokens", type=int, default=8192)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--input-usd-per-mtok", type=float, default=0.0)
    parser.add_argument("--output-usd-per-mtok", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--cache-dir",
        default="~/.cache/kairyu/benchmarks",
    )
    parser.add_argument(
        "--hardware",
        default="8x NVIDIA RTX PRO 6000 Blackwell Server Edition; Qwen3-32B TP8",
    )
    parser.add_argument(
        "--result",
        default="bench/results/tiered-auto-qwen3-32b-tp8.json",
    )
    args = parser.parse_args()
    for name in (
        "latency_pairs",
        "latency_max_tokens",
        "quality_max_tokens",
        "gpu_count",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in (
        "latency_warmup",
        "input_usd_per_mtok",
        "output_usd_per_mtok",
    ):
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
