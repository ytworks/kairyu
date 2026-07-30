"""m9 D5: serving_bench honesty — token-granularity TPOT via include_usage."""

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

_spec = importlib.util.spec_from_file_location(
    "serving_bench", Path(__file__).parents[2] / "bench" / "serving_bench.py"
)
serving_bench = importlib.util.module_from_spec(_spec)
sys.modules["serving_bench"] = serving_bench  # dataclass annotation resolution
_spec.loader.exec_module(serving_bench)


async def test_run_one_reads_usage_chunk_for_token_tpot():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    calls = []

    @app.post("/v1/chat/completions")
    async def chat(request: dict):
        calls.append(request)

        async def _gen():
            yield 'data: {"choices": [{"index": 0, "delta": {"content": "4"}}]}\n\n'
            yield (
                'data: {"choices": [], '
                '"usage": {"prompt_tokens": 3, "completion_tokens": 6, '
                '"total_tokens": 9}}\n\n'
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        metrics = await serving_bench.run_one(
            client,
            "m",
            "bench me please",
            max_tokens=6,
            temperature=0.0,
            seed=7,
            min_tokens=6,
            ignore_eos=True,
        )
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["seed"] == 7
    assert calls[0]["min_tokens"] == 6
    assert calls[0]["ignore_eos"] is True
    assert metrics.completion_tokens == 6  # from the include_usage final chunk
    assert metrics.token_granular is True
    assert metrics.tpot_s >= 0.0


async def test_run_one_falls_back_when_target_rejects_stream_options():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    calls = []

    @app.post("/v1/chat/completions")
    async def chat(request: dict):
        calls.append(request)
        if "stream_options" in request:
            return JSONResponse(status_code=400, content={"error": "no stream_options"})

        async def _gen():
            yield 'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        metrics = await serving_bench.run_one(
            client,
            "m",
            "fallback",
            max_tokens=4,
            temperature=0.0,
            seed=11,
            min_tokens=4,
            ignore_eos=True,
        )
    assert metrics.token_granular is False  # labeled chunk-granularity fallback
    assert len(calls) == 2  # retried once without stream_options
    assert "stream_options" not in calls[1]
    for field, value in {
        "temperature": 0.0,
        "seed": 11,
        "min_tokens": 4,
        "ignore_eos": True,
    }.items():
        assert calls[1][field] == value


def test_summary_retains_raw_request_timings_and_token_throughput():
    results = [
        serving_bench.RequestMetrics(
            ttft_s=0.1,
            total_s=0.4,
            output_chunks=3,
            completion_tokens=4,
        ),
        serving_bench.RequestMetrics(
            ttft_s=0.2,
            total_s=0.5,
            output_chunks=4,
            completion_tokens=5,
        ),
    ]

    summary, samples = serving_bench.summarize_results(
        results,
        wall_ns=1_000_000_000,
        dataset_label="test",
        ttft_slo_s=0.15,
    )

    assert summary["wall_ns"] == 1_000_000_000
    assert summary["wall_s"] == 1.0
    assert summary["completion_tokens_total"] == 9
    assert summary["output_tokens_per_s"] == 9.0
    assert summary["tpot_method"] == "token"
    assert summary["goodput_rps"] == 1.0
    assert samples == [
        {
            "request_index": 0,
            "ttft_ms": 100.0,
            "total_ms": 400.0,
            "tpot_ms": pytest.approx(100.0),
            "output_chunks": 3,
            "completion_tokens": 4,
        },
        {
            "request_index": 1,
            "ttft_ms": 200.0,
            "total_ms": 500.0,
            "tpot_ms": pytest.approx(75.0),
            "output_chunks": 4,
            "completion_tokens": 5,
        },
    ]
