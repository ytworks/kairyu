import argparse
import asyncio
import sys

from verification.orchestration.correctness.orchestration_stream_bench import (
    RequestMeasurement,
)
from verification.product.performance import tiered_auto_bench


def _measurement(model: str, order: int, ttft_ms: float) -> RequestMeasurement:
    return RequestMeasurement(
        model=model,
        order=order,
        prompt_sha256="a" * 64,
        ttft_ms=ttft_ms,
        total_ms=ttft_ms + 10,
        content_chunks=1,
        completion_tokens=8,
        status_comments=0,
        trace_comments=1,
    )


def _args(*, max_ttft_ratio: float = 1.5) -> argparse.Namespace:
    return argparse.Namespace(
        direct_model="qwen3-32b",
        auto_model="kairyu-auto",
        latency_pairs=2,
        latency_warmup=0,
        latency_max_tokens=8,
        concurrency=2,
        max_ttft_ratio=max_ttft_ratio,
    )


def test_latency_gate_preserves_alternating_pairs_and_samples(monkeypatch):
    async def fake_measure_request(client, *, model, prompt, order, max_tokens):
        del client, prompt, max_tokens
        ttft_ms = 100.0 if model == "qwen3-32b" else 120.0
        return _measurement(model, order, ttft_ms)

    monkeypatch.setattr(tiered_auto_bench, "measure_request", fake_measure_request)

    result = asyncio.run(tiered_auto_bench.run_latency(_args(), object()))

    assert result["passed"] is True
    assert result["ratios"] == {
        "ttft_p50_auto_over_direct": 1.2,
        "ttft_p99_auto_over_direct": 1.2,
    }
    assert len(result["samples"]["qwen3-32b"]) == 2
    assert len(result["samples"]["kairyu-auto"]) == 2


def test_latency_gate_rejects_ratio_above_limit(monkeypatch):
    async def fake_measure_request(client, *, model, prompt, order, max_tokens):
        del client, prompt, max_tokens
        ttft_ms = 100.0 if model == "qwen3-32b" else 160.0
        return _measurement(model, order, ttft_ms)

    monkeypatch.setattr(tiered_auto_bench, "measure_request", fake_measure_request)

    result = asyncio.run(tiered_auto_bench.run_latency(_args(), object()))

    assert result["passed"] is False


def test_cli_defaults_to_checkout_only_verification_results(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tiered_auto_bench.py"])

    args = tiered_auto_bench.parse_args()

    assert args.result == "verification/results/tiered-auto-qwen3-32b-tp8.json"
    assert not hasattr(args, "quality_max_tokens")
