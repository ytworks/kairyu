import json

import httpx
import pytest

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.orchestration.orchestrator import Orchestrator

TOOL_CALL_TEXT = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture()
def app():
    engines = {
        "kairyu-mock": MockBackend(responses={"weather": TOOL_CALL_TEXT}),
    }
    orchestrator = Orchestrator(engines={"tier1": MockBackend(), "tier2": MockBackend()})
    return create_app(engines=engines, orchestrator=orchestrator)


def _chat_body(content: str, **extra) -> dict:
    return {
        "model": "kairyu-mock",
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


async def test_models_endpoint_lists_engines_and_auto(app):
    async with _client(app) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "kairyu-mock" in ids
    assert "kairyu-auto" in ids


async def test_chat_completion_happy_path(app):
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body("hello"))
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "kairyu-mock"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] >= 0


async def test_streaming_reassembles_to_full_answer(app):
    async with _client(app) as client:
        full = (await client.post("/v1/chat/completions", json=_chat_body("hello"))).json()
        full_text = full["choices"][0]["message"]["content"]
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("hello", stream=True)
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    deltas = []
    for line in lines[:-1]:
        chunk = json.loads(line[len("data: "):])
        assert chunk["object"] == "chat.completion.chunk"
        delta = chunk["choices"][0]["delta"]
        deltas.append(delta.get("content") or "")
    assert "".join(deltas) == full_text


async def test_tool_call_is_parsed_into_openai_schema(app):
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("what is the weather?", tools=tools)
        )
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Tokyo"}


async def test_kairyu_auto_routes_through_orchestrator(app):
    body = _chat_body("What is the capital of France?")
    body["model"] = "kairyu-auto"
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


async def test_unknown_model_returns_404(app):
    body = _chat_body("hi")
    body["model"] = "ghost"
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 404
    assert "ghost" in response.json()["error"]["message"]


async def test_invalid_body_returns_422(app):
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json={"model": "kairyu-mock"})
    assert response.status_code == 422
