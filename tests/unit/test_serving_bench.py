"""m9 D5: serving_bench honesty — token-granularity TPOT via include_usage."""

import importlib.util
import sys
from pathlib import Path

import httpx

from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.entrypoints.server.app import create_app

_spec = importlib.util.spec_from_file_location(
    "serving_bench", Path(__file__).parents[2] / "bench" / "serving_bench.py"
)
serving_bench = importlib.util.module_from_spec(_spec)
sys.modules["serving_bench"] = serving_bench  # dataclass annotation resolution
_spec.loader.exec_module(serving_bench)


async def test_run_one_reads_usage_chunk_for_token_tpot():
    app = create_app(engines={"m": KairyuBackend(num_pages=256)})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        metrics = await serving_bench.run_one(client, "m", "bench me please", max_tokens=6)
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
        metrics = await serving_bench.run_one(client, "m", "fallback", max_tokens=4)
    assert metrics.token_granular is False  # labeled chunk-granularity fallback
    assert len(calls) == 2  # retried once without stream_options
