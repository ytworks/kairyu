import json

import httpx
import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.openai_backend import OpenAICompatBackend


def _request(prompt: str = "hi") -> GenerationRequest:
    return GenerationRequest(
        request_id="r1",
        prompt=prompt,
        sampling_params=SamplingParams(temperature=0.2, max_tokens=64),
    )


def _ok_transport(captured: dict) -> httpx.MockTransport:
    def handler(http_request: httpx.Request) -> httpx.Response:
        captured["url"] = str(http_request.url)
        captured["auth"] = http_request.headers.get("authorization")
        captured["body"] = json.loads(http_request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello from api"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return httpx.MockTransport(handler)


async def test_generate_maps_openai_response(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-x",
        api_key_env="TEST_API_KEY",
        transport=_ok_transport(captured),
    )
    result = await backend.generate(_request("say hello"))
    assert result.completions[0].text == "hello from api"
    assert result.completions[0].finish_reason == "stop"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-x"
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["max_tokens"] == 64
    assert captured["body"]["messages"][-1]["content"] == "say hello"


async def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1", model="m", api_key_env="MISSING_KEY"
    )
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        await backend.generate(_request())


async def test_http_error_surfaces_status(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, json={"error": {"message": "rate limited"}})
    )
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env="TEST_API_KEY",
        transport=transport,
    )
    with pytest.raises(RuntimeError, match="429"):
        await backend.generate(_request())
