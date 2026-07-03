"""Usage truth through the API (m9 D1, gate P-A1)."""

import json

import httpx
import pytest

from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app

PROMPT_WORDS = 20


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _chat_body(**overrides) -> dict:
    prompt = " ".join(f"word{i}" for i in range(PROMPT_WORDS))
    body = {
        "model": "kairyu-real",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4,
    }
    body.update(overrides)
    return body


@pytest.fixture()
def app():
    return create_app(engines={"kairyu-real": KairyuBackend(num_pages=256)})


async def test_usage_counts_are_tokenizer_true(app):
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body())
    assert response.status_code == 200
    usage = response.json()["usage"]
    # toy tokenizer: one token per word + template framing tokens
    assert usage["completion_tokens"] == 4  # exactly max_tokens committed tokens
    assert usage["prompt_tokens"] >= PROMPT_WORDS  # template adds role framing
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_cached_tokens_reported_on_repeated_prefix(app):
    async with _client(app) as client:
        first = await client.post("/v1/chat/completions", json=_chat_body())
        second = await client.post("/v1/chat/completions", json=_chat_body())
    assert first.status_code == second.status_code == 200
    details = second.json()["usage"].get("prompt_tokens_details")
    assert details is not None
    assert details["cached_tokens"] >= 16  # at least one full radix page reused


async def test_include_usage_final_chunk_contract(app):
    async with _client(app) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=_chat_body(stream=True, stream_options={"include_usage": True}),
        ) as response:
            assert response.status_code == 200
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    chunks.append(json.loads(line[len("data:") :]))
    # every non-final chunk carries usage: null; final carries usage + no choices
    *body, final = chunks
    assert all(chunk["usage"] is None for chunk in body)
    assert final["choices"] == []
    assert final["usage"]["completion_tokens"] == 4
    assert final["usage"]["total_tokens"] > 4


async def test_usage_key_omitted_without_stream_options(app):
    async with _client(app) as client:
        async with client.stream(
            "POST", "/v1/chat/completions", json=_chat_body(stream=True)
        ) as response:
            lines = [line async for line in response.aiter_lines()]
    payloads = [
        json.loads(line[len("data:") :])
        for line in lines
        if line.startswith("data:") and "[DONE]" not in line
    ]
    assert payloads
    assert all("usage" not in chunk for chunk in payloads)


async def test_stream_options_without_stream_is_400(app):
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body(stream_options={"include_usage": True}),
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_max_completion_tokens_alias():
    app = create_app(engines={"kairyu-real": KairyuBackend(num_pages=256)})
    async with _client(app) as client:
        body = _chat_body()
        del body["max_tokens"]
        body["max_completion_tokens"] = 3
        response = await client.post("/v1/chat/completions", json=body)
    assert response.json()["usage"]["completion_tokens"] == 3


async def test_mock_backend_usage_still_flows():
    app = create_app(engines={"m": MockBackend()})
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi there"}]},
        )
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
