import asyncio
import base64
import json
import logging
import threading

import httpx
import pytest

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.mock import MockBackend
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.prompt import TokensPrompt
from kairyu.engine.tokenizer import ToyTokenizer
from kairyu.engine.vision import ImageInputPolicy
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.metering import resolve_usage_counts
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.replica import ReplicaPool
from kairyu.outputs import CompletionOutput

TOOL_CALL_TEXT = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
RED_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDA"
    "xMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg=="
)
_red_png_bytes = base64.b64decode(RED_PNG_DATA_URL.partition(",")[2])
TRUNCATED_RED_PNG_DATA_URL = (
    "data:image/png;base64,"
    + base64.b64encode(_red_png_bytes[:-8]).decode("ascii")
)


class _EmptyTokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return ()

    def decode(self, token_ids) -> str:
        return ""

    def vocab(self) -> list[str]:
        return []


class _CountingTokenizer(ToyTokenizer):
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, text: str) -> tuple[int, ...]:
        self.encode_calls += 1
        return super().encode(text)


def _client(app, *, raise_app_exceptions: bool = True) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(
        app=app, raise_app_exceptions=raise_app_exceptions
    )
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _usage_totals(app) -> dict[str, dict[str, int]]:
    """Read through the owning ledger's ordered durability barrier."""
    return await asyncio.to_thread(app.state.usage_ledger.totals)


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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "completions"])
async def test_zero_token_prompt_returns_400_before_streaming(endpoint, stream):
    backend = KairyuBackend(tokenizer=_EmptyTokenizer())
    app = create_app(engines={"empty": backend})
    if endpoint == "chat":
        path = "/v1/chat/completions"
        body = {
            "model": "empty",
            "messages": [{"role": "user", "content": "normal-looking input"}],
            "stream": stream,
        }
    else:
        path = "/v1/completions"
        body = {
            "model": "empty",
            "prompt": "normal-looking input",
            "stream": stream,
        }

    async with _client(app) as client:
        response = await client.post(path, json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "at least one token" in response.json()["error"]["message"]
    assert not response.headers["content-type"].startswith("text/event-stream")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "completions"])
async def test_native_http_text_prompt_is_tokenized_once(endpoint, stream):
    tokenizer = _CountingTokenizer()
    backend = KairyuBackend(tokenizer=tokenizer, num_pages=64)
    app = create_app(engines={"native": backend})
    if endpoint == "chat":
        path = "/v1/chat/completions"
        body = {
            "model": "native",
            "messages": [{"role": "user", "content": "count this prompt"}],
            "max_tokens": 2,
            "stream": stream,
        }
    else:
        path = "/v1/completions"
        body = {
            "model": "native",
            "prompt": "count this prompt",
            "max_tokens": 2,
            "stream": stream,
        }

    async with _client(app) as client:
        response = await client.post(path, json=body)

    assert response.status_code == 200
    assert tokenizer.encode_calls == 1
    assert backend._prepared_requests == {}
    await backend.shutdown()


async def test_backend_without_validate_request_still_serves_requests():
    app = create_app(engines={"mock": MockBackend()})
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions", json={"model": "mock", "prompt": "hello"}
        )
    assert response.status_code == 200


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


async def test_streamed_tool_calls_carry_required_index(app):
    # S6: each streamed delta.tool_calls[] item must include `index` so OpenAI
    # SDK stream accumulators can merge the fragments.
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    async with _client(app) as client:
        async with client.stream(
            "POST", "/v1/chat/completions",
            json=_chat_body("what is the weather?", tools=tools, stream=True),
        ) as response:
            saw_tool_call = False
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[len("data: "):])
                for tc in chunk["choices"][0]["delta"].get("tool_calls") or []:
                    assert "index" in tc
                    saw_tool_call = True
    assert saw_tool_call  # a tool-call delta was actually emitted


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


@pytest.mark.parametrize("chunked", [False, True])
async def test_chat_body_limit_rejects_declared_and_streamed_oversize_bodies(
    chunked,
):
    backend = MockBackend()
    app = create_app(
        engines={"m": backend},
        settings=ServerSettings(max_chat_body_bytes=128),
    )
    encoded = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x" * 256}],
        }
    ).encode()

    async def body_chunks():
        yield encoded[:96]
        yield encoded[96:]

    content = body_chunks() if chunked else encoded
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=content,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert backend.prompts_seen == ()


async def test_invalid_sampling_params_return_400_not_500(app):
    # S2: out-of-range sampling values must be a client error (400), not an
    # unhandled ValueError surfacing as 500 (chat) or a mislabeled 502.
    async with _client(app) as client:
        for bad in ({"top_p": 0}, {"n": 0}, {"temperature": -1}):
            response = await client.post("/v1/chat/completions", json=_chat_body("hi", **bad))
            assert response.status_code == 400, (bad, response.status_code)
            assert response.json()["error"]["type"] == "invalid_request_error"


async def test_completions_invalid_sampling_returns_400(app):
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions", json={"model": "kairyu-mock", "prompt": "hi", "top_p": 0}
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "part",
    [
        pytest.param(
            {"type": "input_audio", "input_audio": {"data": "AA=="}},
            id="unknown-audio",
        ),
        pytest.param(
            {"type": "video_url", "video_url": {"url": "https://example.test/video"}},
            id="unknown-video",
        ),
        pytest.param({"type": "text"}, id="missing-text"),
        pytest.param(
            {
                "type": "text",
                "text": "ambiguous",
                "image_url": {"url": "https://example.test/image"},
            },
            id="conflicting-fields",
        ),
        pytest.param(
            {"type": "text", "text": "hello", "unparsed": {"future": True}},
            id="extra-field",
        ),
        pytest.param(
            {"type": "image_url", "image_url": {}},
            id="missing-image-url",
        ),
    ],
)
async def test_invalid_chat_content_parts_are_rejected_before_dispatch(part):
    backend = MockBackend()
    app = create_app(engines={"parts": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "parts",
                "messages": [{"role": "user", "content": [part]}],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]
    assert backend.prompts_seen == ()


async def test_vlm_chat_preserves_roles_and_content_part_order_through_gateway():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "RED"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 43,
                    "completion_tokens": 1,
                    "total_tokens": 44,
                },
            },
        )

    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="upstream-vlm",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={
            "max_images": 1,
            "max_processed_prompt_tokens": 1024,
        },
    )
    app = create_app(engines={"vlm": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "vlm",
                "messages": [
                    {"role": "system", "content": "Name the color."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "before "},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": RED_PNG_DATA_URL,
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": " after"},
                        ],
                    },
                ],
                "max_tokens": 4,
                "temperature": 0,
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "RED"
    assert response.json()["usage"] == {
        "prompt_tokens": 43,
        "completion_tokens": 1,
        "total_tokens": 44,
        "prompt_tokens_details": None,
    }
    assert captured["body"]["model"] == "upstream-vlm"
    assert captured["body"]["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Name the color."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before "},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": RED_PNG_DATA_URL,
                        "detail": "high",
                    },
                },
                {"type": "text", "text": " after"},
            ],
        },
    ]


async def test_pooled_vlm_decodes_inline_base64_only_once(monkeypatch):
    decode_calls = 0
    decode_threads: list[int] = []
    original_decode = ImageInputPolicy._decode_item

    def recording_decode(self, item, *, index):
        nonlocal decode_calls
        decode_calls += 1
        decode_threads.append(threading.get_ident())
        return original_decode(self, item, index=index)

    monkeypatch.setattr(ImageInputPolicy, "_decode_item", recording_decode)
    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="upstream-vlm",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "RED",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 43,
                        "completion_tokens": 1,
                    },
                },
            )
        ),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )
    app = create_app(engines={"vlm": ReplicaPool([backend])})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "vlm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "name the color"},
                            {
                                "type": "image_url",
                                "image_url": {"url": RED_PNG_DATA_URL},
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert decode_calls == 1
    assert decode_threads[0] != threading.get_ident()


@pytest.mark.parametrize("pooled", [False, True])
async def test_image_input_fails_closed_for_text_model_and_remote_url(pooled):
    text_backend = MockBackend()
    transport_calls: list[httpx.Request] = []
    vision_backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="upstream-vlm",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: (
                transport_calls.append(request)
                or httpx.Response(200, json={"choices": []})
            )
        ),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )
    vision_engine = (
        ReplicaPool({"10.23.45.67": vision_backend})
        if pooled
        else vision_backend
    )
    app = create_app(engines={"text": text_backend, "vlm": vision_engine})
    content = [
        {"type": "text", "text": "describe"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.test/image.png"},
        },
    ]

    async with _client(app) as client:
        text_response = await client.post(
            "/v1/chat/completions",
            json={"model": "text", "messages": [{"role": "user", "content": content}]},
        )
        vision_response = await client.post(
            "/v1/chat/completions",
            json={"model": "vlm", "messages": [{"role": "user", "content": content}]},
        )

    assert text_response.status_code == 400
    assert "multimodal" in text_response.json()["error"]["message"]
    assert text_backend.prompts_seen == ()
    assert vision_response.status_code == 400
    assert vision_response.json()["error"]["code"] == "invalid_image"
    assert "remote and local image URLs are disabled" in (
        vision_response.json()["error"]["message"]
    )
    assert "10.23.45.67" not in vision_response.text
    assert transport_calls == []
    assert vision_backend._client is None


async def test_image_input_rejects_blank_role_as_a_controlled_400():
    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="upstream-vlm",
        api_key_env=None,
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )
    app = create_app(engines={"vlm": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "vlm",
                "messages": [
                    {
                        "role": " ",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {"url": RED_PNG_DATA_URL},
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("stream", [False, True])
async def test_truncated_image_is_rejected_before_stream_or_token_reservation(stream):
    transport_calls: list[httpx.Request] = []
    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="upstream-vlm",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: (
                transport_calls.append(request)
                or httpx.Response(200, json={"choices": []})
            )
        ),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={"max_images": 1},
    )
    app = create_app(
        engines={"vlm": backend},
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=100_000,
                    token_burst=100_000,
                )
            }
        ),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "vlm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": TRUNCATED_RED_PNG_DATA_URL,
                                },
                            },
                        ],
                    }
                ],
                "stream": stream,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"
    assert transport_calls == []
    assert app.state.tenant_limiter.reservation_snapshot()["default"] == 0


async def test_unexpected_upstream_4xx_is_sanitized_and_not_forwarded():
    backend = OpenAICompatBackend(
        base_url="http://internal-vlm.secret:8000/v1",
        model="upstream-vlm",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                400,
                text="SUPER_SECRET upstream diagnostic",
            )
        ),
        upstream="vllm",
    )
    app = create_app(engines={"remote": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "remote",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 502
    payload = response.json()["error"]
    assert payload == {
        "message": "upstream backend rejected the request",
        "type": "upstream_error",
        "code": "backend_error",
    }
    assert "SUPER_SECRET" not in response.text
    assert "internal-vlm" not in response.text


async def test_vlm_stream_failure_without_usage_closes_sanitized_sse(tmp_path):
    backend = OpenAICompatBackend(
        base_url="http://vlm:8000/v1",
        model="upstream-vlm",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    'data: {"choices":[{"index":0,"delta":{"content":"RED"},'
                    '"finish_reason":null}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        ),
        upstream="vllm",
        capabilities={"allow_prompt_kinds": ["multimodal"]},
        image_input_policy={
            "max_images": 1,
            "max_processed_prompt_tokens": 1024,
        },
    )
    app = create_app(
        engines={"vlm": backend},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=20_000,
                    token_burst=20_000,
                )
            }
        ),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "vlm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "name the color"},
                            {
                                "type": "image_url",
                                "image_url": {"url": RED_PNG_DATA_URL},
                            },
                        ],
                    }
                ],
                "max_tokens": 4,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

    assert response.status_code == 200
    assert "RED" in response.text
    assert "upstream backend error (RuntimeError)" in response.text
    assert response.text.rstrip().endswith("data: [DONE]")
    assert await _usage_totals(app) == {}
    assert app.state.tenant_limiter.reservation_snapshot()["default"] == 0
    assert app.state.tenant_limiter.token_balance("default") < 19_000


async def test_unavailable_pool_admission_is_a_sanitized_upstream_error():
    pool = ReplicaPool([MockBackend()])
    pool.drain("0")
    app = create_app(engines={"pool": pool})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "pool",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "backend_error"


@pytest.mark.parametrize(
    "carrier",
    [
        {"prompt_token_ids": [1, 2, 3]},
        {"input_audio": {"data": "AA=="}},
        {"multi_modal_data": {"image": "opaque"}},
    ],
)
async def test_chat_message_alternate_prompt_carriers_are_rejected_before_dispatch(
    carrier,
):
    backend = MockBackend()
    app = create_app(engines={"chat": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "chat",
                "messages": [{"role": "user", "content": "hello", **carrier}],
            },
        )

    assert response.status_code == 400
    assert "messages[0] has unsupported fields" in response.json()["error"]["message"]
    assert backend.prompts_seen == ()


async def test_chat_tool_call_transcript_remains_supported():
    backend = MockBackend()
    app = create_app(engines={"chat": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "chat",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"city":"Tokyo"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": '{"temp":21}',
                        "tool_call_id": "call_1",
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert "<tool_call>" in backend.prompts_seen[0]


async def test_absent_tool_calls_does_not_change_template_message_shape():
    template = ChatTemplate(
        "{{ 'tool_calls' in messages[0] }}|{{ messages[0].content }}"
    )
    backend = MockBackend()
    app = create_app(
        engines={"shape": backend},
        chat_templates={"shape": template},
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "shape",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert backend.prompts_seen == ("False|hello",)


@pytest.mark.parametrize("stream", [False, True])
async def test_completion_token_prompt_is_dispatched_losslessly(stream):
    backend = MockBackend()
    app = create_app(engines={"tokens": backend})
    body = {
        "model": "tokens",
        "prompt": [7, 11, 13],
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=body)

    assert response.status_code == 200
    assert backend.prompts_seen == (TokensPrompt((7, 11, 13)),)
    if stream:
        chunks = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["prompt_tokens"] == 3
    else:
        assert response.json()["usage"]["prompt_tokens"] == 3


async def test_completion_sampling_extensions_are_typed_and_preserved_to_engine():
    class CapturingBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.seen = None

        async def generate(self, request):
            self.seen = request
            return await super().generate(request)

    engine = CapturingBackend()
    app = create_app(engines={"stub": engine})
    body = {
        "model": "stub",
        "prompt": "fixed-length benchmark",
        "max_tokens": 8,
        "top_k": -1,
        "min_p": 0.0,
        "stop_token_ids": [7, 9],
        "min_tokens": 8,
        "ignore_eos": True,
        "repetition_penalty": 1.0,
        "skip_special_tokens": False,
    }

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=body)

    assert response.status_code == 200
    params = engine.seen.sampling_params
    assert params.top_k == -1
    assert params.min_p == 0.0
    assert params.stop_token_ids == (7, 9)
    assert params.min_tokens == 8
    assert params.ignore_eos is True
    assert params.repetition_penalty == 1.0
    assert params.skip_special_tokens is False


async def test_completion_token_batch_preserves_prompt_and_choice_order():
    backend = MockBackend()
    app = create_app(engines={"tokens": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "tokens", "prompt": [[2, 3], [17], [5, 8, 13]]},
        )

    assert response.status_code == 200
    assert backend.prompts_seen == (
        TokensPrompt((2, 3)),
        TokensPrompt((17,)),
        TokensPrompt((5, 8, 13)),
    )
    choices = response.json()["choices"]
    assert [choice["index"] for choice in choices] == [0, 1, 2]
    assert [choice["text"] for choice in choices] == [
        "mock:tokens:2,3",
        "mock:tokens:17",
        "mock:tokens:5,8,13",
    ]


async def test_completion_token_prompt_missing_usage_uses_exact_id_count():
    class TokenBackend(StubBackend):
        def validate_request(self, request):
            if not isinstance(request.prompt, TokensPrompt):
                raise ValueError("test backend requires token IDs")

    backend = TokenBackend(text="derived completion words", usage=None)
    app = create_app(engines={"tokens": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "tokens", "prompt": [3, 5, 8, 13]},
        )

    assert response.status_code == 200
    assert backend.requests[0].prompt == TokensPrompt((3, 5, 8, 13))
    assert response.json()["usage"]["prompt_tokens"] == 4


async def test_token_prompt_requires_an_explicit_backend_capability():
    backend = StubBackend()
    app = create_app(engines={"legacy": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "legacy", "prompt": [1, 2, 3]},
        )

    assert response.status_code == 400
    assert "does not declare support for tokens prompts" in response.json()["error"]["message"]
    assert backend.calls == 0


@pytest.mark.parametrize(
    "carrier",
    [
        {"prompt_token_ids": [1, 2, 3]},
        {"prompt_embeds": [[0.1, 0.2]]},
        {"multi_modal_data": {"image": "opaque"}},
    ],
)
async def test_completion_alternate_prompt_carriers_are_never_silently_ignored(
    carrier,
):
    backend = StubBackend()
    app = create_app(engines={"legacy": backend})
    body = {"model": "legacy", "prompt": "text authority", **carrier}

    async with _client(app) as client:
        response = await client.post("/v1/completions", json=body)

    assert response.status_code == 400
    assert "unsupported request fields" in response.json()["error"]["message"]
    assert backend.calls == 0


@pytest.mark.parametrize(
    ("prompt", "expected_status"),
    [
        pytest.param([], 400, id="empty"),
        pytest.param([1, "2"], 422, id="mixed"),
        pytest.param([1, True], 422, id="boolean"),
        pytest.param([1, -1], 422, id="negative"),
    ],
)
async def test_invalid_completion_token_prompts_are_rejected_before_dispatch(
    prompt, expected_status
):
    backend = StubBackend()
    app = create_app(engines={"tokens": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "tokens", "prompt": prompt},
        )

    assert response.status_code == expected_status
    assert backend.calls == 0


async def test_completion_text_batch_remains_compatible():
    backend = MockBackend()
    app = create_app(engines={"text": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "text", "prompt": ["first", "second"]},
        )

    assert response.status_code == 200
    assert backend.prompts_seen == ("first", "second")
    assert [choice["index"] for choice in response.json()["choices"]] == [0, 1]


class StubBackend:
    """Fixed completions / optional failure, for finish_reason and error tests."""

    def __init__(
        self, text="stub answer", finish_reason="length", error=None, usage=None
    ):
        self._text = text
        self._finish_reason = finish_reason
        self._error = error
        self._usage = usage
        self.calls = 0
        self.requests = []

    async def generate(self, request):
        from kairyu.engine.backend import GenerationResult
        from kairyu.outputs import CompletionOutput

        self.calls += 1
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0, text=self._text, token_ids=(), finish_reason=self._finish_reason
                ),
            ),
            usage=self._usage,
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        return None


class MultiChoiceToolBackend:
    def __init__(self, texts, usage=None):
        self._texts = texts
        self._usage = usage
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=tuple(
                CompletionOutput(
                    index=index,
                    text=text,
                    token_ids=(),
                    finish_reason="stop",
                )
                for index, text in enumerate(self._texts)
            ),
            usage=self._usage,
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        return None


class CompletionUsageBackend:
    """Legacy-completion backend with per-prompt optional usage."""

    def __init__(self, reported_usage):
        self._reported_usage = reported_usage

    async def generate(self, request):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=f"answer for {request.prompt}",
                    token_ids=(),
                    finish_reason="stop",
                ),
            ),
            usage=self._reported_usage.get(request.prompt),
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self):
        return None


class MeteringStreamBackend:
    """Cumulative two-part stream with controllable finalization behavior."""

    def __init__(self, case):
        self.case = case
        self.requests = []
        self.cancelled = False

    async def generate(self, request):
        raise AssertionError("streaming test reached generate")

    async def stream(self, request):
        self.requests.append(request)
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text="partial output",
                    token_ids=(101, 102),
                    finish_reason=None,
                ),
            ),
            finished=False,
            usage=(
                GenerationUsage(prompt_tokens=17, completion_tokens=9)
                if self.case == "reported-then-none"
                else None
            ),
        )
        if self.case == "client-close":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.case == "upstream-error":
            raise RuntimeError("stream failed after partial output")
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text="partial output final",
                    token_ids=(101, 102, 103),
                    finish_reason="stop",
                ),
            ),
            usage=(
                GenerationUsage(prompt_tokens=17, completion_tokens=9)
                if self.case == "reported"
                else None
            ),
        )

    async def shutdown(self):
        return None


def _metered_stream_app(kind, backend, ledger_path):
    settings = ServerSettings(usage_ledger_path=str(ledger_path))
    if kind == "orchestrated-chat":
        app = create_app(
            engines={},
            orchestrators={"auto": Orchestrator({"tier1": backend})},
            settings=settings,
        )
        return app, "/v1/chat/completions", {
            "model": "auto",
            "messages": [{"role": "user", "content": "metering prompt"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    app = create_app(engines={"stream": backend}, settings=settings)
    if kind == "engine-chat":
        return app, "/v1/chat/completions", {
            "model": "stream",
            "messages": [{"role": "user", "content": "metering prompt"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    return app, "/v1/completions", {
        "model": "stream",
        "prompt": "metering prompt",
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def _disconnect_after_partial(app, path, body):
    """Drive the ASGI endpoint until one partial body, then disconnect."""
    request_sent = False
    disconnect = asyncio.Event()
    response_messages = []
    encoded = json.dumps(body).encode()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        response_messages.append(message)
        if (
            message["type"] == "http.response.body"
            and b"partial output" in message.get("body", b"")
        ):
            disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    assert disconnect.is_set()
    return response_messages


class DispatchCountingOrchestrator:
    """Failing spy: invalid requests must never reach either dispatch seam."""

    def __init__(self):
        self.run_calls = 0
        self.run_chat_calls = 0

    async def run(self, prompt):
        self.run_calls += 1
        raise AssertionError("invalid request reached orchestrator.run")

    async def run_chat(self, prompt, *, stream):
        self.run_chat_calls += 1
        raise AssertionError("invalid request reached orchestrator.run_chat")


_WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}},
}


_OUTPUT_LIMIT_CASES = [
    pytest.param(
        "/v1/chat/completions",
        "max_tokens",
        {"model": "stub", "messages": [{"role": "user", "content": "hello"}]},
        id="chat-max-tokens",
    ),
    pytest.param(
        "/v1/chat/completions",
        "max_completion_tokens",
        {"model": "stub", "messages": [{"role": "user", "content": "hello"}]},
        id="chat-max-completion-tokens",
    ),
    pytest.param(
        "/v1/completions",
        "max_tokens",
        {"model": "stub", "prompt": "hello"},
        id="legacy-completions-max-tokens",
    ),
    pytest.param(
        "/v1/responses",
        "max_output_tokens",
        {"model": "stub", "input": "hello"},
        id="responses-max-output-tokens",
    ),
]


@pytest.mark.parametrize(("path", "field", "body"), _OUTPUT_LIMIT_CASES)
@pytest.mark.parametrize("limit", [0, -1])
async def test_nonpositive_output_token_limits_are_predispatch_400(
    path, field, body, limit
):
    engine = StubBackend(text="done", finish_reason="stop")
    app = create_app(engines={"stub": engine})

    async with _client(app, raise_app_exceptions=False) as client:
        response = await client.post(path, json={**body, field: limit})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == "invalid_request"
    assert engine.calls == 0


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("limit", [0, -1])
@pytest.mark.parametrize("stream", [False, True])
async def test_nonpositive_auto_output_limits_are_predispatch_400(
    field, limit, stream
):
    orchestrator = DispatchCountingOrchestrator()
    app = create_app(engines={}, orchestrators={"auto": orchestrator})
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        field: limit,
    }

    async with _client(app, raise_app_exceptions=False) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": f"max_tokens must be >= 1, got {limit}",
        "type": "invalid_request_error",
        "code": "invalid_request",
    }
    assert orchestrator.run_calls == 0
    assert orchestrator.run_chat_calls == 0


@pytest.mark.parametrize(("path", "field", "body"), _OUTPUT_LIMIT_CASES)
async def test_positive_output_token_limits_reach_backend(path, field, body):
    engine = StubBackend(text="done", finish_reason="stop")
    app = create_app(engines={"stub": engine})

    async with _client(app) as client:
        response = await client.post(path, json={**body, field: 7})

    assert response.status_code == 200
    assert engine.calls == 1
    assert engine.requests[0].sampling_params.max_tokens == 7


async def test_chat_max_tokens_precedes_modern_alias_when_both_are_present():
    engine = StubBackend(text="done", finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = {
        "model": "stub",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 5,
        "max_completion_tokens": 9,
    }

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert engine.requests[0].sampling_params.max_tokens == 5


async def test_responses_default_output_token_limit_is_1024():
    engine = StubBackend(text="done", finish_reason="stop")
    app = create_app(engines={"stub": engine})

    async with _client(app) as client:
        response = await client.post(
            "/v1/responses", json={"model": "stub", "input": "hello"}
        )

    assert response.status_code == 200
    assert engine.requests[0].sampling_params.max_tokens == 1024


@pytest.mark.parametrize(
    ("tools", "tool_choice"),
    [
        (None, None),
        ([], None),
        ([_WEATHER_TOOL], None),
        (None, "auto"),
        ([], "auto"),
        ([_WEATHER_TOOL], "auto"),
        (None, "none"),
        ([], "none"),
        ([_WEATHER_TOOL], "none"),
        ([_WEATHER_TOOL], "required"),
        (
            [_WEATHER_TOOL],
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
async def test_coherent_tool_choice_is_accepted(tools, tool_choice):
    engine = StubBackend(text=TOOL_CALL_TEXT, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body("weather", tools=tools, tool_choice=tool_choice)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert engine.calls == 1


@pytest.mark.parametrize(
    ("tools", "tool_choice"),
    [
        (None, "required"),
        ([], "required"),
        ([_WEATHER_TOOL], "any"),
        ([_WEATHER_TOOL], {}),
        ([_WEATHER_TOOL], {"type": "function"}),
        (
            [_WEATHER_TOOL],
            {"type": "other", "function": {"name": "get_weather"}},
        ),
        ([_WEATHER_TOOL], {"type": "function", "function": {}}),
        (
            [_WEATHER_TOOL],
            {"type": "function", "function": {"name": ""}},
        ),
        (
            [_WEATHER_TOOL],
            {"type": "function", "function": {"name": 7}},
        ),
        (
            [_WEATHER_TOOL],
            {"type": "function", "function": {"name": "other"}},
        ),
        ([{"type": "function", "function": {}}], "auto"),
        ([{"type": "function", "function": {"name": ""}}], "none"),
        ([{"type": "function", "function": {"name": "   "}}], None),
        ([{"type": "function", "function": {"name": 7}}], "auto"),
        ([{"type": "other", "function": {"name": "get_weather"}}], "auto"),
        ([{"type": "function", "function": "get_weather"}], "auto"),
        ([_WEATHER_TOOL, _WEATHER_TOOL], "required"),
    ],
)
async def test_invalid_tool_choice_returns_400_before_backend_dispatch(
    tools, tool_choice
):
    engine = StubBackend(text=TOOL_CALL_TEXT, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body("weather", tools=tools, tool_choice=tool_choice)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_request"
    assert error["message"]
    assert engine.calls == 0


def _tool_response_contract(response, stream: bool):
    if not stream:
        choice = response.json()["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        return (
            choice["message"].get("content"),
            [call["function"]["name"] for call in calls],
            choice["finish_reason"],
        )

    chunks = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    choices = [choice for chunk in chunks for choice in chunk["choices"]]
    return (
        "".join(choice["delta"].get("content") or "" for choice in choices) or None,
        [
            call["function"]["name"]
            for choice in choices
            for call in choice["delta"].get("tool_calls") or []
        ],
        next(
            choice["finish_reason"]
            for choice in reversed(choices)
            if choice["finish_reason"] is not None
        ),
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("upstream_finish_reason", "expected_finish_reason"),
    [("stop", "stop"), ("tool_calls", "stop"), ("length", "length")],
)
async def test_tool_choice_none_suppresses_calls_and_keeps_content(
    stream, upstream_finish_reason, expected_finish_reason
):
    engine = StubBackend(text=TOOL_CALL_TEXT, finish_reason=upstream_finish_reason)
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather", tools=[_WEATHER_TOOL], tool_choice="none", stream=stream
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (
        TOOL_CALL_TEXT,
        [],
        expected_finish_reason,
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_choice", [None, "auto"])
async def test_auto_and_omitted_emit_only_declared_calls(stream, tool_choice):
    text = "".join(
        (
            '<tool_call>{"name":"undeclared","arguments":{}}</tool_call>',
            TOOL_CALL_TEXT,
        )
    )
    engine = StubBackend(text=text, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body("weather", tools=[_WEATHER_TOOL], stream=stream)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (
        None,
        ["get_weather"],
        "tool_calls",
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_auto_suppresses_undeclared_model_function_names(stream):
    text = '<tool_call>{"name":"undeclared","arguments":{}}</tool_call>'
    engine = StubBackend(text=text, finish_reason="tool_calls")
    app = create_app(engines={"stub": engine})
    body = _chat_body("weather", tools=[_WEATHER_TOOL], stream=stream)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (text, [], "stop")


@pytest.mark.parametrize("stream", [False, True])
async def test_tool_choice_required_emits_declared_calls(stream):
    engine = StubBackend(text=TOOL_CALL_TEXT, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather", tools=[_WEATHER_TOOL], tool_choice="required", stream=stream
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (
        None,
        ["get_weather"],
        "tool_calls",
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_named_tool_choice_filters_multi_call_output(stream):
    text = "".join(
        (
            TOOL_CALL_TEXT,
            '<tool_call>{"name":"search","arguments":{"q":"rain"}}</tool_call>',
        )
    )
    search_tool = {
        "type": "function",
        "function": {"name": "search", "parameters": {"type": "object"}},
    }
    engine = StubBackend(text=text, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL, search_tool],
        tool_choice={"type": "function", "function": {"name": "search"}},
        stream=stream,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert _tool_response_contract(response, stream) == (
        None,
        ["search"],
        "tool_calls",
    )


@pytest.mark.parametrize(
    ("text", "tool_choice"),
    [
        ("no tool call", "required"),
        (
            '<tool_call>{"name":"other","arguments":{}}</tool_call>',
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
async def test_unsatisfied_tool_choice_is_controlled_upstream_failure(
    text, tool_choice
):
    other_tool = {
        "type": "function",
        "function": {"name": "other", "parameters": {"type": "object"}},
    }
    engine = StubBackend(text=text, finish_reason="stop")
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather", tools=[_WEATHER_TOOL, other_tool], tool_choice=tool_choice
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"
    assert response.json()["error"]["code"] == "tool_choice_not_satisfied"
    assert engine.calls == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("named", [False, True])
async def test_mixed_multi_choice_tool_choice_is_rejected(stream, named):
    search_text = '<tool_call>{"name":"search","arguments":{"q":"rain"}}</tool_call>'
    search_tool = {
        "type": "function",
        "function": {"name": "search", "parameters": {"type": "object"}},
    }
    engine = MultiChoiceToolBackend((TOOL_CALL_TEXT, search_text if named else "plain"))
    app = create_app(engines={"stub": engine})
    tool_choice = (
        {"type": "function", "function": {"name": "get_weather"}}
        if named
        else "required"
    )
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL, search_tool],
        tool_choice=tool_choice,
        n=2,
        stream=stream,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "tool_choice_not_satisfied"
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert engine.calls == 1


async def test_empty_tool_choice_result_is_rejected():
    engine = MultiChoiceToolBackend(())
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL],
        tool_choice="required",
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "tool_choice_not_satisfied"
    assert engine.calls == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("named", [False, True])
async def test_all_valid_multi_choice_tool_choice_is_accepted(stream, named):
    search_text = '<tool_call>{"name":"search","arguments":{"q":"rain"}}</tool_call>'
    search_tool = {
        "type": "function",
        "function": {"name": "search", "parameters": {"type": "object"}},
    }
    if named:
        texts = (
            TOOL_CALL_TEXT + search_text,
            search_text + TOOL_CALL_TEXT,
        )
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        expected_names = ["get_weather", "get_weather"]
    else:
        texts = (TOOL_CALL_TEXT, search_text)
        tool_choice = "required"
        expected_names = ["get_weather", "search"]
    engine = MultiChoiceToolBackend(texts)
    app = create_app(engines={"stub": engine})
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL, search_tool],
        tool_choice=tool_choice,
        n=2,
        stream=stream,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    if not stream:
        choices = response.json()["choices"]
        assert [choice["index"] for choice in choices] == [0, 1]
        assert [choice["finish_reason"] for choice in choices] == [
            "tool_calls",
            "tool_calls",
        ]
        assert [
            choice["message"]["tool_calls"][0]["function"]["name"]
            for choice in choices
        ] == expected_names
    else:
        chunks = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        choices = [choice for chunk in chunks for choice in chunk["choices"]]
        calls = [
            (choice["index"], call["index"], call["function"]["name"])
            for choice in choices
            for call in choice["delta"].get("tool_calls") or []
        ]
        finishes = [
            (choice["index"], choice["finish_reason"])
            for choice in choices
            if choice["finish_reason"] is not None
        ]
        assert calls == [
            (0, 0, expected_names[0]),
            (1, 0, expected_names[1]),
        ]
        assert finishes == [(0, "tool_calls"), (1, "tool_calls")]
    assert engine.calls == 1


async def test_malformed_auto_tool_output_records_actual_backend_usage(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    text = "before <tool_call>[]</tool_call> after"
    engine = StubBackend(
        text=text,
        finish_reason="stop",
        usage=GenerationUsage(prompt_tokens=7, completion_tokens=3),
    )
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )
    body = _chat_body("weather", tools=[_WEATHER_TOOL], tool_choice="auto")
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["tool_calls"] is None
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cached_tokens": 0,
        "uncached_tokens": 7,
    }


async def test_cached_usage_is_exported_to_wire_ledger_and_metrics(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(
        text="cached answer",
        finish_reason="stop",
        usage=GenerationUsage(
            prompt_tokens=17,
            completion_tokens=3,
            cached_tokens=12,
        ),
    )
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "reused prefix"}],
            },
        )
        metrics = (await client.get("/metrics")).text

    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 12
    }
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": 17,
        "completion_tokens": 3,
        "cached_tokens": 12,
        "uncached_tokens": 5,
    }
    assert (
        'kairyu_usage_tokens_total{tenant="default",type="cached"} 12.0'
        in metrics
    )
    assert (
        'kairyu_usage_tokens_total{tenant="default",type="uncached"} 5.0'
        in metrics
    )


async def test_sync_usage_none_records_wire_derived_counts(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(text="derived completion words", finish_reason="stop")
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "rendered prompt words"}],
            },
        )

    assert response.status_code == 200
    wire_usage = response.json()["usage"]
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": wire_usage["prompt_tokens"],
        "completion_tokens": wire_usage["completion_tokens"],
        "cached_tokens": 0,
        "uncached_tokens": wire_usage["prompt_tokens"],
    }


async def test_sync_orchestrator_zero_usage_derives_wire_and_ledger(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(text="derived orchestrated answer", finish_reason="stop")
    app = create_app(
        engines={},
        orchestrators={"auto": Orchestrator({"tier1": engine})},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "brief prompt"}],
            },
        )

    assert response.status_code == 200
    expected = resolve_usage_counts(
        None,
        prompt=engine.requests[0].prompt,
        completions=(
            CompletionOutput(
                index=0,
                text="derived orchestrated answer",
                token_ids=(),
            ),
        ),
    )
    wire_usage = response.json()["usage"]
    assert wire_usage["prompt_tokens"] == expected[0]
    assert wire_usage["completion_tokens"] == expected[1]
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": expected[0],
        "completion_tokens": expected[1],
        "cached_tokens": 0,
        "uncached_tokens": expected[0],
    }


@pytest.mark.parametrize(
    ("prompt", "reported_usage", "expected"),
    [
        pytest.param("single prompt", {}, (2, 4), id="single-missing"),
        pytest.param(
            ["first prompt", "second"],
            {},
            (3, 7),
            id="array-missing",
        ),
        pytest.param(
            ["derived prompt", "reported prompt"],
            {
                "reported prompt": GenerationUsage(
                    prompt_tokens=17,
                    completion_tokens=9,
                )
            },
            (19, 13),
            id="array-mixed",
        ),
    ],
)
async def test_sync_completions_derives_each_missing_usage(
    tmp_path, prompt, reported_usage, expected
):
    ledger_path = tmp_path / "usage.jsonl"
    app = create_app(
        engines={"stub": CompletionUsageBackend(reported_usage)},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "stub", "prompt": prompt},
        )

    assert response.status_code == 200
    wire_usage = response.json()["usage"]
    assert wire_usage["prompt_tokens"] == expected[0]
    assert wire_usage["completion_tokens"] == expected[1]
    assert wire_usage["total_tokens"] == sum(expected)
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": expected[0],
        "completion_tokens": expected[1],
        "cached_tokens": 0,
        "uncached_tokens": expected[0],
    }


@pytest.mark.parametrize(
    "kind", ["engine-chat", "orchestrated-chat", "legacy-completions"]
)
@pytest.mark.parametrize(
    "case", ["reported", "usage-none", "client-close", "upstream-error"]
)
async def test_every_dispatched_stream_is_metered_exactly_once(
    tmp_path, kind, case
):
    ledger_path = tmp_path / "usage.jsonl"
    backend = MeteringStreamBackend(case)
    app, path, body = _metered_stream_app(kind, backend, ledger_path)

    if case == "client-close":
        await _disconnect_after_partial(app, path, body)
        assert backend.cancelled
        response_text = ""
    else:
        async with _client(app) as client:
            response = await client.post(path, json=body)
        assert response.status_code == 200
        response_text = response.text
        assert "data: [DONE]" in response_text

    assert len(backend.requests) == 1
    completion_text = (
        "partial output final" if case in {"reported", "usage-none"}
        else "partial output"
    )
    expected = (
        (17, 9)
        if case == "reported"
        else resolve_usage_counts(
            None,
            prompt=backend.requests[0].prompt,
            completions=(
                CompletionOutput(index=0, text=completion_text, token_ids=()),
            ),
        )
    )
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": expected[0],
        "completion_tokens": expected[1],
        "cached_tokens": 0,
        "uncached_tokens": expected[0],
    }

    if case in {"reported", "usage-none"}:
        usage_chunks = [
            json.loads(line[len("data: "):])["usage"]
            for line in response_text.splitlines()
            if line.startswith("data: {")
            and json.loads(line[len("data: "):]).get("usage") is not None
        ]
        assert usage_chunks[-1]["prompt_tokens"] == expected[0]
        assert usage_chunks[-1]["completion_tokens"] == expected[1]
    elif case == "upstream-error":
        error_chunks = [
            json.loads(line[len("data: "):])["error"]
            for line in response_text.splitlines()
            if line.startswith("data: {")
            and "error" in json.loads(line[len("data: "):])
        ]
        assert len(error_chunks) == 1
        assert "RuntimeError" in error_chunks[0]["message"]


@pytest.mark.parametrize(
    "kind", ["engine-chat", "orchestrated-chat", "legacy-completions"]
)
async def test_stream_retains_reported_usage_when_final_partial_has_none(
    tmp_path, kind
):
    ledger_path = tmp_path / "usage.jsonl"
    backend = MeteringStreamBackend("reported-then-none")
    app, path, body = _metered_stream_app(kind, backend, ledger_path)

    async with _client(app) as client:
        response = await client.post(path, json=body)

    assert response.status_code == 200
    usage_chunks = [
        json.loads(line[len("data: "):])["usage"]
        for line in response.text.splitlines()
        if line.startswith("data: {")
        and json.loads(line[len("data: "):]).get("usage") is not None
    ]
    assert usage_chunks[-1]["prompt_tokens"] == 17
    assert usage_chunks[-1]["completion_tokens"] == 9
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": 17,
        "completion_tokens": 9,
        "cached_tokens": 0,
        "uncached_tokens": 17,
    }


@pytest.mark.parametrize(
    "kind", ["engine-chat", "orchestrated-chat", "legacy-completions"]
)
async def test_closing_unstarted_stream_does_not_dispatch_or_meter(
    tmp_path, kind
):
    from starlette.requests import Request

    from kairyu.engine.backend import GenerationRequest
    from kairyu.entrypoints.server.app import (
        _stream_completions,
        _stream_engine,
        _stream_orchestrator,
    )
    from kairyu.entrypoints.server.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )
    from kairyu.sampling_params import SamplingParams

    ledger_path = tmp_path / "usage.jsonl"
    backend = MeteringStreamBackend("client-close")
    app, _, _ = _metered_stream_app(kind, backend, ledger_path)
    http_request = Request({"type": "http", "app": app, "state": {}})
    generation_request = GenerationRequest(
        request_id="not-started",
        prompt="unstarted prompt",
        sampling_params=SamplingParams(),
    )
    if kind == "engine-chat":
        stream = _stream_engine(
            backend,
            generation_request,
            "stream",
            ChatCompletionRequest(
                model="stream",
                messages=[{"role": "user", "content": "unstarted"}],
                stream=True,
            ),
            http_request,
        )
    elif kind == "orchestrated-chat":
        stream = _stream_orchestrator(
            Orchestrator({"tier1": backend}),
            generation_request.prompt,
            ChatCompletionRequest(
                model="auto",
                messages=[{"role": "user", "content": "unstarted"}],
                stream=True,
            ),
            False,
            False,
            http_request,
        )
    else:
        stream = _stream_completions(
            backend,
            generation_request,
            CompletionRequest(model="stream", prompt="unstarted", stream=True),
            http_request,
        )

    await stream.aclose()

    assert backend.requests == []
    assert not ledger_path.exists()


async def test_generate_fully_tool_stream_keeps_single_sync_metering_owner(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(
        text=TOOL_CALL_TEXT,
        finish_reason="tool_calls",
        usage=GenerationUsage(prompt_tokens=13, completion_tokens=8),
    )
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": 13,
        "completion_tokens": 8,
        "cached_tokens": 0,
        "uncached_tokens": 13,
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("text", "tool_choice"),
    [
        ("no tool call", "required"),
        (
            '<tool_call>{"name":"other","arguments":{}}</tool_call>',
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
async def test_unsatisfied_tool_choice_is_metered_once(
    tmp_path, stream, text, tool_choice
):
    ledger_path = tmp_path / "usage.jsonl"
    engine = StubBackend(
        text=text,
        finish_reason="stop",
        usage=GenerationUsage(prompt_tokens=11, completion_tokens=5),
    )
    app = create_app(
        engines={"stub": engine},
        settings=ServerSettings(usage_ledger_path=str(ledger_path)),
    )
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL],
        tool_choice=tool_choice,
        stream=stream,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "tool_choice_not_satisfied"
    assert engine.calls == 1
    assert (await _usage_totals(app))["default"] == {
        "requests": 1,
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "cached_tokens": 0,
        "uncached_tokens": 11,
    }


async def test_named_tool_stream_includes_per_call_index():
    text = '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>'
    app = create_app(engines={"stub": StubBackend(text=text, finish_reason="stop")})
    body = _chat_body(
        "weather",
        tools=[_WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        stream=True,
    )
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    chunks = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    calls = [
        call
        for chunk in chunks
        for choice in chunk["choices"]
        for call in choice["delta"].get("tool_calls") or []
    ]
    assert [call["index"] for call in calls] == [0]


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "[]",
        "{}",
        '{"arguments": {}}',
        '{"name": 7, "arguments": {}}',
        '{"name": "", "arguments": {}}',
        '{"name": "get_weather", "arguments": []}',
        '{"name": "get_weather", "arguments": 7}',
        '{"name": "get_weather", "arguments": true}',
        pytest.param(
            '{"name": "get_weather", "arguments": ' + ("9" * 5000) + "}",
            id="oversized-json-integer",
        ),
        pytest.param(
            '{"name": "get_weather", "arguments": '
            + ("[" * 1100)
            + "0"
            + ("]" * 1100)
            + "}",
            id="excessive-json-nesting",
        ),
    ],
)
async def test_malformed_generated_tool_payload_stays_content(payload):
    text = f"before <tool_call>{payload}</tool_call> after"
    app = create_app(engines={"stub": StubBackend(text=text, finish_reason="stop")})
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]
    body = _chat_body("weather", tools=tools)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == text
    assert choice["message"]["tool_calls"] is None


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"city": "Tokyo"}', '{"city": "Tokyo"}'),
        ('"{\\"city\\":\\"Tokyo\\"}"', '{"city":"Tokyo"}'),
    ],
)
async def test_generated_tool_arguments_are_serialized_once(arguments, expected):
    text = (
        '<tool_call>{"name": "get_weather", "arguments": '
        f"{arguments}}}</tool_call>"
    )
    app = create_app(engines={"stub": StubBackend(text=text, finish_reason="stop")})
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]
    body = _chat_body("weather", tools=tools)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"] == expected


async def test_generated_tool_calls_keep_order_while_skipping_only_malformed_entries():
    text = "".join(
        (
            '<tool_call>{"name":"first","arguments":{}}</tool_call>',
            '<tool_call>{"name":"broken","arguments":[]}</tool_call>',
            '<tool_call>{"name":"second"}</tool_call>',
        )
    )
    app = create_app(engines={"stub": StubBackend(text=text, finish_reason="stop")})
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("first", "broken", "second")
    ]
    body = _chat_body("tools", tools=tools)
    body["model"] = "stub"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["first", "second"]
    assert [call["function"]["arguments"] for call in calls] == ["{}", "{}"]


async def test_zero_token_prompt_array_is_validated_before_any_generation_starts():
    class ArrayValidationBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.validated: list[str] = []
            self.started: list[str] = []

        def validate_request(self, request):
            self.validated.append(request.prompt)
            if request.prompt == "invalid":
                raise ValueError("prompt must tokenize to at least one token")

        async def generate(self, request):
            self.started.append(request.prompt)
            return await super().generate(request)

    engine = ArrayValidationBackend()
    app = create_app(engines={"stub": engine})
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "stub", "prompt": ["valid", "invalid"]},
        )

    assert response.status_code == 400
    assert engine.validated == ["valid", "invalid"]
    assert engine.started == []


async def test_empty_completion_prompt_array_is_rejected_before_reservation():
    backend = MockBackend()
    app = create_app(
        engines={"m": backend},
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=1_000,
                )
            }
        ),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "m", "prompt": []},
        )

    assert response.status_code == 400
    assert "must not be empty" in response.json()["error"]["message"]
    assert backend.prompts_seen == ()
    assert app.state.tenant_limiter.reservation_snapshot()["default"] == 0


async def test_completion_array_failure_waits_for_sibling_cleanup_before_lease_release():
    class ArrayFailureBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.sibling_started = asyncio.Event()
            self.sibling_cleaned = asyncio.Event()

        async def generate(self, request):
            if request.prompt == "slow":
                self.sibling_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.sibling_cleaned.set()
            await self.sibling_started.wait()
            raise RuntimeError("array element failed")

    backend = ArrayFailureBackend()
    app = create_app(
        engines={"m": backend},
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=10_000,
                    token_burst=10_000,
                    max_in_flight=1,
                )
            }
        ),
    )

    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "m", "prompt": ["slow", "fail"]},
        )
        metrics = (await client.get("/metrics")).text

    assert response.status_code == 502
    assert backend.sibling_cleaned.is_set()
    assert app.state.tenant_limiter.in_flight("default") == 0
    assert app.state.tenant_limiter.reservation_snapshot()["default"] == 0
    assert (
        'kairyu_tenant_in_flight_requests{source="http",tenant="default"} 0.0'
    ) in metrics


async def test_runtime_value_error_remains_an_upstream_error():
    app = create_app(engines={"stub": StubBackend(error=ValueError("runtime failure"))})
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


async def test_streaming_with_n_gt_1_emits_all_choice_indexes(app):
    async with _client(app) as client:
        full = (
            await client.post("/v1/chat/completions", json=_chat_body("hello", n=2))
        ).json()
        response = await client.post(
            "/v1/chat/completions", json=_chat_body("hello", n=2, stream=True)
        )
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    texts: dict[int, str] = {}
    for line in lines[:-1]:
        chunk = json.loads(line[len("data: "):])
        for choice in chunk["choices"]:
            texts[choice["index"]] = texts.get(choice["index"], "") + (
                choice["delta"].get("content") or ""
            )
    expected = {c["index"]: c["message"]["content"] for c in full["choices"]}
    assert texts == expected
    assert set(texts) == {0, 1}


async def test_streaming_with_tools_signals_tool_calls(app):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_body("what is the weather?", tools=tools, stream=True),
        )
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(line[len("data: "):]) for line in lines[:-1]]
    finish_reasons = [
        choice["finish_reason"] for chunk in chunks for choice in chunk["choices"]
    ]
    assert "tool_calls" in finish_reasons
    tool_deltas = [
        choice["delta"].get("tool_calls")
        for chunk in chunks
        for choice in chunk["choices"]
        if choice["delta"].get("tool_calls")
    ]
    assert tool_deltas[0][0]["function"]["name"] == "get_weather"


async def test_backend_finish_reason_passes_through():
    app = create_app(engines={"stub": StubBackend(finish_reason="length")})
    async with _client(app) as client:
        body = _chat_body("hi")
        body["model"] = "stub"
        response = await client.post("/v1/chat/completions", json=body)
    assert response.json()["choices"][0]["finish_reason"] == "length"


async def test_backend_error_returns_openai_error_envelope(caplog):
    app = create_app(
        engines={"stub": StubBackend(error=RuntimeError("secret://host:5432 key missing"))}
    )
    with caplog.at_level(logging.ERROR, logger="kairyu.entrypoints.server.app"):
        async with _client(app) as client:
            body = _chat_body("hi")
            body["model"] = "stub"
            response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "upstream_error"
    # M3: the backend's raw message (which may carry secrets/hosts) must NOT leak
    assert "secret://host:5432" not in error["message"]
    assert "RuntimeError" in error["message"]  # only the class name is disclosed
    # ...but the full traceback IS logged server-side (observability): a replica
    # backend failure must be diagnosable from the logs even though the client
    # only sees the exception class name.
    assert any(
        record.exc_info and record.levelno == logging.ERROR
        for record in caplog.records
    ), "expected the backend traceback to be logged server-side"


async def test_response_format_passes_through_to_engine():
    class CapturingBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.seen = None

        async def generate(self, request):
            self.seen = request
            return await super().generate(request)

    engine = CapturingBackend()
    app = create_app(engines={"stub": engine})
    body = _chat_body("give me json")
    body["model"] = "stub"
    body["response_format"] = {"type": "json_object"}
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert engine.seen.sampling_params.extra_args["response_format"] == {
        "type": "json_object"
    }


async def test_chat_sampling_extensions_are_typed_and_preserved_to_engine():
    class CapturingBackend(StubBackend):
        def __init__(self):
            super().__init__()
            self.seen = None

        async def generate(self, request):
            self.seen = request
            return await super().generate(request)

    engine = CapturingBackend()
    app = create_app(engines={"stub": engine})
    body = _chat_body("sample")
    body.update(
        {
            "model": "stub",
            "best_of": 2,
            "top_k": 8,
            "min_p": 0.15,
            "repetition_penalty": 1.2,
            "stop_token_ids": [7, 9],
            "min_tokens": 3,
            "ignore_eos": True,
            "prompt_logprobs": 2,
            "skip_special_tokens": False,
            "extra_args": {"vendor_cache": {"mode": "ephemeral"}},
        }
    )

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    params = engine.seen.sampling_params
    assert params.best_of == 2
    assert params.top_k == 8
    assert params.min_p == 0.15
    assert params.repetition_penalty == 1.2
    assert params.stop_token_ids == (7, 9)
    assert params.min_tokens == 3
    assert params.ignore_eos is True
    assert params.prompt_logprobs == 2
    assert params.skip_special_tokens is False
    assert params.extra_args == {"vendor_cache": {"mode": "ephemeral"}}


@pytest.mark.parametrize(
    "carrier",
    [
        {"prompt_token_ids": [1, 2, 3]},
        {"prompt_embeds": [[0.1, 0.2]]},
        {"input_image": {"image_url": "https://example.test/image"}},
        {"multi_modal_data": {"image": "opaque"}},
    ],
)
async def test_chat_extra_args_cannot_smuggle_an_alternate_prompt_carrier(carrier):
    engine = StubBackend()
    app = create_app(engines={"stub": engine})
    body = _chat_body("text authority", model="stub", extra_args=carrier)

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "prompt-owned fields" in response.json()["error"]["message"]
    assert engine.calls == 0


async def test_unknown_chat_field_is_clear_predispatch_400():
    engine = StubBackend()
    app = create_app(engines={"stub": engine})
    body = _chat_body("sample", model="stub", misspelled_temprature=0.5)

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "misspelled_temprature" in response.json()["error"]["message"]


async def test_openai_upstream_capability_mismatch_is_predispatch_400():
    transport_calls = []

    def handler(request):
        transport_calls.append(request)
        return httpx.Response(200, json={"choices": []})

    engine = OpenAICompatBackend(
        base_url="https://api.anthropic.example/v1",
        model="claude",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="anthropic",
    )
    app = create_app(engines={"claude": engine})
    body = _chat_body("two", model="claude", n=2)

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "n must be <= 1" in response.json()["error"]["message"]
    assert transport_calls == []
    assert engine._client is None


@pytest.mark.parametrize("wrapper", ["pool", "auto"])
async def test_wrapped_upstream_capability_mismatch_is_predispatch_400(wrapper):
    transport_calls = []

    def handler(request):
        transport_calls.append(request)
        return httpx.Response(200, json={"choices": []})

    upstream = OpenAICompatBackend(
        base_url="https://api.anthropic.example/v1",
        model="claude",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="anthropic",
    )
    body = _chat_body("two", n=2)
    if wrapper == "pool":
        app = create_app(engines={"wrapped": ReplicaPool([upstream])})
        body["model"] = "wrapped"
    else:
        app = create_app(
            engines={},
            orchestrators={
                "auto": Orchestrator({"tier1": upstream, "tier2": upstream})
            },
        )
        body["model"] = "auto"

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "n must be <= 1" in response.json()["error"]["message"]
    assert transport_calls == []
    assert upstream._client is None


@pytest.mark.parametrize("moa_samples", [0, 2])
async def test_auto_best_of_is_forwarded_only_to_the_final_engine(
    moa_samples,
):
    transport_payloads = []

    def handler(request):
        transport_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    proposal = OpenAICompatBackend(
        base_url="https://proposal.example/v1",
        model="proposal",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="generic",
    )
    final = OpenAICompatBackend(
        base_url="https://verified.example/v1",
        model="final",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        upstream="generic",
        capabilities={"allow_sampling_fields": ["best_of"]},
    )
    app = create_app(
        engines={},
        orchestrators={
            "auto": Orchestrator(
                {"tier1": proposal, "tier2": final},
                moa_samples=moa_samples,
            )
        },
    )
    body = _chat_body(
        "First research. Then plan. After that implement. Finally verify.",
        model="auto",
        best_of=2,
    )

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert len(transport_payloads) > 1
    assert all("best_of" not in payload for payload in transport_payloads[:-1])
    assert transport_payloads[-1]["best_of"] == 2
