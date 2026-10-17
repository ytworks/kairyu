"""A request reaches L1 in the API its caller used (m9 D2 amendment 2026-10-17).

Each test sends a public request to a deployment built from YAML and observes
the request its fake OpenAI-compatible upstream received: a Responses caller
reaches an upstream declared in ``upstream_responses_models`` on
``/v1/responses`` with the conversation as it is, a Messages caller prefers
``/v1/messages`` then ``/v1/responses``, everything else stays on chat, and an
AUTO model's role calls follow the caller's API. An assistant prefill skips
Messages (vLLM opens a new turn there) and a single-call limit stays on vLLM's
chat path, the only one that applies it.
"""

from __future__ import annotations

import json

import httpx
import pytest

import kairyu.deploy.builder as builder_module
from kairyu.deploy.builder import build_app_from_spec
from kairyu.deploy.spec import load_deployment_spec
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.registry import create_backend
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS, SamplingParams


def _sse(events: list[dict], *, named: bool = True) -> bytes:
    frames = []
    for event in events:
        prefix = f"event: {event['type']}\n" if named else ""
        frames.append(f"{prefix}data: {json.dumps(event)}\n\n")
    return "".join(frames).encode()


class FakeUpstream:
    """Records every request; answers chat, Responses and Messages with 'ok'."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.url.host, request.url.path, body))
        path = request.url.path
        stream = bool(body.get("stream"))
        if path.endswith("/chat/completions"):
            usage = {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}
            if stream:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse(
                        [
                            {"choices": [{"index": 0, "delta": {"content": "ok"}}]},
                            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                            {"choices": [], "usage": usage},
                        ],
                        named=False,
                    )
                    + b"data: [DONE]\n\n",
                )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-fake",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )
        if not body:
            return httpx.Response(400, json={"error": {"message": "empty body"}})
        if path.endswith("/responses"):
            usage = {
                "input_tokens": 7,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 8,
            }
            message = {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
            response = {"id": "resp_up", "status": "completed", "output": [message], "usage": usage}
            if stream:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse(
                        [
                            {"type": "response.created", "response": {"id": "resp_up"}},
                            {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": {**message, "content": [], "status": "in_progress"},
                            },
                            {
                                "type": "response.output_text.delta",
                                "output_index": 0,
                                "content_index": 0,
                                "delta": "ok",
                            },
                            {
                                "type": "response.output_item.done",
                                "output_index": 0,
                                "item": message,
                            },
                            {"type": "response.completed", "response": response},
                        ]
                    ),
                )
            return httpx.Response(200, json=response)
        assert path.endswith("/messages"), path
        usage = {"input_tokens": 7, "output_tokens": 1}
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(
                    [
                        {"type": "message_start", "message": {"usage": {"input_tokens": 7}}},
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": "ok"},
                        },
                        {"type": "content_block_stop", "index": 0},
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn"},
                            "usage": {"output_tokens": 1},
                        },
                        {"type": "message_stop"},
                    ]
                ),
            )
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": usage,
            },
        )

    def paths(self) -> list[str]:
        return [path for _host, path, _body in self.calls]


@pytest.fixture
def upstream(monkeypatch) -> FakeUpstream:
    fake = FakeUpstream()

    def factory(backend, **options):
        if backend != "openai":
            return create_backend(backend, **options)
        return OpenAICompatBackend(**options, transport=httpx.MockTransport(fake.handler))

    monkeypatch.setattr(builder_module, "create_backend", factory)
    return fake


def _spec(*, responses: bool = False, messages: bool = False, extra: str = "") -> str:
    lists = "upstream_chat_models: [remote]\n"
    if responses:
        lists += "upstream_responses_models: [remote]\n"
    if messages:
        lists += "upstream_messages_models: [remote]\n"
    return (
        """
engines:
  remote:
    backend: openai
    options:
      base_url: http://vllm.test/v1
      model: upstream-model
      api_key_env: null
      upstream: vllm
      max_model_len: 4096
"""
        + lists
        + extra
    )


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


_TOOL = {
    "type": "function",
    "name": "lookup",
    "description": "Look a value up.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
}
_RESPONSES_INPUT = [
    {"role": "user", "content": "What is k?"},
    {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": '{"key":"k"}'},
    {"type": "function_call_output", "call_id": "call_1", "output": "42"},
    {"role": "user", "content": "Answer now."},
]


@pytest.mark.parametrize("stream", [False, True])
async def test_responses_caller_reaches_a_declared_upstream_in_responses(upstream, stream):
    app = build_app_from_spec(load_deployment_spec(_spec(responses=True)))
    async with _client(app) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "remote",
                "instructions": "Be brief.",
                "input": _RESPONSES_INPUT,
                "tools": [_TOOL],
                "reasoning": {"effort": "max"},
                "stream": stream,
            },
        )

    assert response.status_code == 200, response.text
    if stream:
        events = [
            json.loads(line[len("data:") :])
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        final = events[-1]["response"]
        assert events[-1]["type"] == "response.completed"
    else:
        final = response.json()
    assert final["output"][-1]["content"][0]["text"] == "ok"
    ((_host, path, body),) = upstream.calls
    assert path == "/v1/responses"
    assert body["store"] is False
    assert body["reasoning"] == {"effort": "xhigh"}
    assert body["input"] == [
        {"type": "message", "role": "system", "content": "Be brief."},
        {"type": "message", "role": "user", "content": "What is k?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"key":"k"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        {"type": "message", "role": "user", "content": "Answer now."},
    ]
    assert body["tools"] == [_TOOL]
    assert bool(body.get("stream")) is stream


async def test_responses_caller_stays_on_chat_without_a_declared_upstream(upstream):
    app = build_app_from_spec(load_deployment_spec(_spec()))
    async with _client(app) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "remote", "instructions": "Be brief.", "input": _RESPONSES_INPUT},
        )

    assert response.status_code == 200, response.text
    ((_host, path, body),) = upstream.calls
    assert path == "/v1/chat/completions"
    assert [message["role"] for message in body["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]


@pytest.mark.parametrize(
    ("responses", "messages", "expected"),
    [
        (True, True, "/v1/messages"),
        (True, False, "/v1/responses"),
        (False, False, "/v1/chat/completions"),
    ],
)
async def test_messages_caller_prefers_messages_then_responses_then_chat(
    upstream, responses, messages, expected
):
    app = build_app_from_spec(
        load_deployment_spec(_spec(responses=responses, messages=messages))
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "remote",
                "max_tokens": 64,
                "system": "Be brief.",
                "messages": [{"role": "user", "content": "Hi"}],
                "output_config": {"effort": "max"},
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "ok"
    ((_host, path, body),) = upstream.calls
    assert path == expected
    if expected == "/v1/messages":
        assert body["system"] == [{"type": "text", "text": "Be brief."}]
        assert body["messages"] == [{"role": "user", "content": "Hi"}]
        assert body["max_tokens"] == 64
        assert body["output_config"] == {"effort": "max"}


async def test_an_assistant_prefill_continues_on_responses_not_messages(upstream):
    backend = OpenAICompatBackend(
        base_url="http://vllm.test/v1",
        model="m",
        api_key_env=None,
        upstream="vllm",
        upstream_apis=("responses", "messages"),
        transport=httpx.MockTransport(upstream.handler),
    )
    await backend.generate(
        GenerationRequest(
            request_id="prefill",
            prompt="Hi",
            # As the HTTP boundary leaves them: model-owned defaults omitted.
            sampling_params=SamplingParams(max_tokens=16).with_generation_config_omitted(
                GENERATION_CONFIG_SAMPLING_FIELDS
            ),
            assistant_prefill="Partial answer",
            caller_api="messages",
        )
    )
    await backend.shutdown()

    ((_host, path, body),) = upstream.calls
    assert path == "/v1/responses"
    assert body["input"][-1] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Partial answer"}],
        "status": "incomplete",
    }


@pytest.mark.parametrize(
    ("endpoint", "request_body"),
    [
        ("/v1/responses", {"input": "Look k up.", "tools": [_TOOL], "parallel_tool_calls": False}),
        (
            "/v1/messages",
            {
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Look k up."}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Look a value up.",
                        "input_schema": _TOOL["parameters"],
                    }
                ],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            },
        ),
    ],
    ids=["responses", "messages"],
)
async def test_a_single_call_limit_stays_on_vllm_chat(upstream, endpoint, request_body):
    app = build_app_from_spec(load_deployment_spec(_spec(responses=True, messages=True)))
    async with _client(app) as client:
        response = await client.post(endpoint, json={"model": "remote", **request_body})

    assert response.status_code == 200, response.text
    ((_host, path, body),) = upstream.calls
    assert path == "/v1/chat/completions"
    assert body["parallel_tool_calls"] is False


async def test_auto_role_calls_follow_the_callers_api(upstream, tmp_path):
    (tmp_path / "auto.yaml").write_text(
        """
workers:
  - name: writer
    engine_ref: remote
roles:
  - name: answer
    worker: writer
    role_type: synthesizer
    prompt: "Answer the task.\\nTask: {query}"
budget:
  max_steps: 4
"""
    )
    spec = load_deployment_spec(
        _spec(responses=True, extra="orchestrators:\n  auto:\n    spec: auto.yaml\n")
    )
    app = build_app_from_spec(spec, base_dir=tmp_path)
    async with _client(app) as client:
        via_responses = await client.post(
            "/v1/responses", json={"model": "auto", "input": "Say ok."}
        )
        responses_paths = upstream.paths()
        upstream.calls.clear()
        via_chat = await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Say ok."}]},
        )

    assert via_responses.status_code == 200, via_responses.text
    assert via_chat.status_code == 200, via_chat.text
    assert responses_paths and set(responses_paths) == {"/v1/responses"}
    assert upstream.paths() and set(upstream.paths()) == {"/v1/chat/completions"}


async def test_a_declared_api_the_upstream_lacks_fails_startup():
    backend = OpenAICompatBackend(
        base_url="http://vllm.test/v1",
        model="m",
        api_key_env=None,
        upstream="vllm",
        upstream_apis=("responses",),
        transport=httpx.MockTransport(lambda _request: httpx.Response(404)),
    )
    with pytest.raises(RuntimeError, match="does not serve /responses"):
        await backend.startup()
    await backend.shutdown()
