import json

import httpx
import pytest

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest, UpstreamClientError
from kairyu.engine.openai_backend import OpenAICompatBackend


def _request(
    prompt: str = "hi", sampling_params: SamplingParams | None = None
) -> GenerationRequest:
    return GenerationRequest(
        request_id="r1",
        prompt=prompt,
        sampling_params=sampling_params
        or SamplingParams(temperature=0.2, max_tokens=64),
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


async def test_generate_forwards_representable_sampling_payload():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
    )
    params = SamplingParams(
        presence_penalty=0.4,
        frequency_penalty=-0.3,
        top_k=8,
        min_p=0.15,
        logprobs=3,
        extra_args={"response_format": {"type": "json_object"}},
    )

    await backend.generate(_request(sampling_params=params))

    assert captured["body"]["presence_penalty"] == 0.4
    assert captured["body"]["frequency_penalty"] == -0.3
    assert captured["body"]["top_k"] == 8
    assert captured["body"]["min_p"] == 0.15
    assert captured["body"]["logprobs"] is True
    assert captured["body"]["top_logprobs"] == 3
    assert captured["body"]["response_format"] == {"type": "json_object"}
    await backend.shutdown()


async def test_generate_forwards_logprobs_zero():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
    )

    await backend.generate(
        _request(sampling_params=SamplingParams(logprobs=0))
    )

    assert captured["body"]["logprobs"] is True
    assert captured["body"]["top_logprobs"] == 0
    await backend.shutdown()


async def test_generate_default_payload_omits_unrequested_controls():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport(captured),
    )

    await backend.generate(_request(sampling_params=SamplingParams()))

    assert captured["body"]["presence_penalty"] == 0.0
    assert captured["body"]["frequency_penalty"] == 0.0
    for field in (
        "top_k",
        "min_p",
        "logprobs",
        "top_logprobs",
        "response_format",
    ):
        assert field not in captured["body"]
    await backend.shutdown()


@pytest.mark.parametrize(
    ("field", "params"),
    [
        ("best_of", SamplingParams(best_of=2)),
        ("repetition_penalty", SamplingParams(repetition_penalty=1.1)),
        ("stop_token_ids", SamplingParams(stop_token_ids=[7])),
        ("min_tokens", SamplingParams(min_tokens=1)),
        ("prompt_logprobs", SamplingParams(prompt_logprobs=1)),
        ("ignore_eos", SamplingParams(ignore_eos=True)),
        ("skip_special_tokens", SamplingParams(skip_special_tokens=False)),
        (
            "extra_args.unknown_control",
            SamplingParams(extra_args={"unknown_control": True}),
        ),
        ("logprobs", SamplingParams(logprobs=-1)),
        ("extra_args", SamplingParams(extra_args=[])),
    ],
)
async def test_generate_rejects_unsupported_intent_before_client_or_transport(
    field, params
):
    transport_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(request)
        return httpx.Response(200, json={"choices": []})

    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(UpstreamClientError) as exc_info:
        await backend.generate(_request(sampling_params=params))

    assert exc_info.value.status_code == 400
    assert field in str(exc_info.value)
    assert transport_calls == []
    assert backend._client is None


async def test_stream_rejects_unsupported_intent_before_client_or_transport():
    transport_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(request)
        return httpx.Response(200, content=_SSE_BODY)

    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(UpstreamClientError, match="best_of"):
        async for _ in backend.stream(
            _request(sampling_params=SamplingParams(best_of=2))
        ):
            pass

    assert transport_calls == []
    assert backend._client is None


async def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1", model="m", api_key_env="MISSING_KEY"
    )
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        await backend.generate(_request())


async def test_client_error_surfaces_as_upstream_client_error(monkeypatch):
    # O1: a 4xx is the client's fault, raised as UpstreamClientError so the
    # ReplicaPool does not count it against the replica's health.
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env="TEST_API_KEY",
        transport=transport,
    )
    with pytest.raises(UpstreamClientError) as excinfo:
        await backend.generate(_request())
    assert excinfo.value.status_code == 400


async def test_server_error_surfaces_as_runtime_error(monkeypatch):
    # 5xx is a replica/transport failure the pool SHOULD count.
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    transport = httpx.MockTransport(lambda request: httpx.Response(503, text="unavailable"))
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env="TEST_API_KEY",
        transport=transport,
    )
    with pytest.raises(RuntimeError, match="503"):
        await backend.generate(_request())


# --- m6 D2 fixes: real streaming, pooled client, optional auth, token counts ---

_SSE_BODY = (
    b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


def _sse_transport(captured: dict) -> httpx.MockTransport:
    def handler(http_request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(http_request.content)
        return httpx.Response(
            200, content=_SSE_BODY, headers={"content-type": "text/event-stream"}
        )

    return httpx.MockTransport(handler)


async def test_stream_parses_sse_into_cumulative_partials(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env="TEST_API_KEY",
        transport=_sse_transport(captured),
    )

    results = [result async for result in backend.stream(_request("stream it"))]

    assert captured["body"]["stream"] is True
    texts = [result.completions[0].text for result in results]
    assert texts == ["hel", "hello", "hello"]
    assert [result.finished for result in results] == [False, False, True]
    assert results[-1].completions[0].finish_reason == "stop"
    await backend.shutdown()


async def test_keyless_backend_omits_auth_header():
    captured: dict = {}
    backend = OpenAICompatBackend(
        base_url="http://node-b:8000/v1",
        model="m",
        api_key_env=None,  # node-to-node replica: no auth (design m6 D2)
        transport=_ok_transport(captured),
    )
    result = await backend.generate(_request())
    assert result.completions[0].text == "hello from api"
    assert captured["auth"] is None
    await backend.shutdown()


async def test_async_client_is_reused_across_requests(monkeypatch):
    instances = []
    real_client = httpx.AsyncClient

    class CountingClient(real_client):
        def __init__(self, *args, **kwargs):
            instances.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", CountingClient)
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=_ok_transport({}),
    )
    await backend.generate(_request())
    await backend.generate(_request())
    assert len(instances) == 1  # persistent pooled client, no per-request handshake
    await backend.shutdown()


async def test_usage_completion_tokens_populate_token_ids():
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "five tokens here"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 5},
            },
        )

    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="m",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
    )
    result = await backend.generate(_request())
    assert len(result.completions[0].token_ids) == 5
    await backend.shutdown()
