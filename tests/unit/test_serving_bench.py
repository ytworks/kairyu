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
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t/v1"
    ) as client:
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
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t/v1"
    ) as client:
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
    assert summary["ttft_p50_ms"] == 100.0
    assert summary["completion_tokens_total"] == 9
    assert summary["percentile_method"] == "nearest-rank-v1"
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


def test_shared_target_form_normalizes_url_and_uses_env_secret(monkeypatch):
    monkeypatch.setenv("SERVING_API_KEY", "private-value")
    args = serving_bench.build_parser().parse_args(
        [
            "--target",
            "gateway=http://gateway.test:8000/=model=SERVING_API_KEY",
        ]
    )

    target = serving_bench.resolve_target(args)
    config = serving_bench.build_run_config(args)

    assert target.base_url == "http://gateway.test:8000/v1"
    assert serving_bench.resolve_api_key(args, target) == "private-value"
    assert config["api_key_env"] == "SERVING_API_KEY"
    assert config["api_key_source"] == "environment"
    assert "private-value" not in str(config)


def test_shared_target_refuses_conflicting_legacy_endpoint_flags():
    args = serving_bench.build_parser().parse_args(
        [
            "--target",
            "gateway=http://gateway.test=model",
            "--model",
            "other-model",
        ]
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        serving_bench.resolve_target(args)


def test_missing_explicit_api_key_environment_fails_closed(monkeypatch):
    monkeypatch.delenv("MISSING_SERVING_API_KEY", raising=False)
    args = serving_bench.build_parser().parse_args(
        ["--api-key-env", "MISSING_SERVING_API_KEY"]
    )
    target = serving_bench.resolve_target(args)

    with pytest.raises(ValueError, match="MISSING_SERVING_API_KEY"):
        serving_bench.resolve_api_key(args, target)


def test_legacy_literal_api_key_never_enters_recorded_config():
    secret = "literal-secret-value"
    args = serving_bench.build_parser().parse_args(["--api-key", secret])
    target = serving_bench.resolve_target(args)
    config = serving_bench.build_run_config(args)

    assert serving_bench.resolve_api_key(args, target) == secret
    assert config["api_key_source"] == "legacy-cli"
    assert secret not in str(config)
    assert "Deprecated literal bearer token" in serving_bench.build_parser().format_help()


def test_serving_percentile_uses_nearest_rank_definition():
    assert serving_bench._percentile(list(range(1, 21)), 0.95) == 19
