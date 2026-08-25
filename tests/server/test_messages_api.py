"""Anthropic Messages adapter: envelope, tool loop, streaming, errors (#508)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import UsageLedger
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.outputs import CompletionOutput
from tests.server._legacy_chat import create_legacy_app


def _app(tmp_path, backend=None, **kwargs):
    return create_legacy_app(
        {"m": backend or MockBackend({"hello": "streamed hello"})},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        **kwargs,
    )


def _auto_app(tmp_path, backend=None):
    backend = backend or MockBackend({"hello": "orchestrated hello"})
    return create_legacy_app(
        {},
        orchestrators={
            "kairyu-auto": Orchestrator({"tier1": backend, "tier2": backend})
        },
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )


def _events(body: str) -> list[tuple[str, dict]]:
    """Parse SSE frames into (event_name, payload) pairs, skipping comments."""

    events: list[tuple[str, dict]] = []
    name = None
    for line in body.splitlines():
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            assert name is not None, f"data frame without event name: {line!r}"
            events.append((name, json.loads(line.removeprefix("data: "))))
            name = None
    return events


def _text_from_events(events: list[tuple[str, dict]]) -> str:
    return "".join(
        payload["delta"]["text"]
        for name, payload in events
        if name == "content_block_delta"
        and payload["delta"]["type"] == "text_delta"
    )


def _assert_sse_response_headers(response: httpx.Response) -> None:
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "connection" not in response.headers


def _tool() -> dict:
    return {
        "name": "add",
        "description": "Add two integers.",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    }


def _body(**overrides) -> dict:
    body = {
        "model": "m",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(overrides)
    return body


class ToolLoopBackend(MockBackend):
    async def generate(self, request):
        if "tool: 5" in request.prompt:
            text = "The sum is 5."
        else:
            text = '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0, text=text, token_ids=(1, 2, 3), finish_reason="stop"
                ),
            ),
            usage=GenerationUsage(prompt_tokens=11, completion_tokens=3),
        )


class ParallelToolBackend(MockBackend):
    async def generate(self, request):
        text = (
            '<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>'
            '<tool_call>{"name":"add","arguments":{"a":3,"b":4}}</tool_call>'
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0, text=text, token_ids=(1, 2), finish_reason="stop"
                ),
            ),
            usage=GenerationUsage(prompt_tokens=5, completion_tokens=5),
        )


# ---------------------------------------------------------------------------
# Non-streaming envelope


@pytest.mark.parametrize("query", ["", "?beta=true"])
def test_non_streaming_text_message(tmp_path, query):
    # Claude Code posts inference to /v1/messages?beta=true: the route must
    # match on the path and tolerate query parameters.
    client = TestClient(_app(tmp_path))
    response = client.post(
        f"/v1/messages{query}",
        json=_body(),
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200
    message = response.json()
    assert message["id"].startswith("msg_")
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["model"] == "m"
    assert message["content"] == [{"type": "text", "text": "streamed hello"}]
    assert message["stop_reason"] == "end_turn"
    assert message["stop_sequence"] is None
    assert message["usage"]["input_tokens"] > 0
    assert message["usage"]["output_tokens"] > 0


def test_length_maps_to_max_tokens_stop_reason(tmp_path):
    class LengthBackend(MockBackend):
        async def generate(self, request):
            result = await super().generate(request)
            completions = tuple(
                CompletionOutput(
                    index=completion.index,
                    text=completion.text,
                    token_ids=completion.token_ids,
                    finish_reason="length",
                )
                for completion in result.completions
            )
            return GenerationResult(
                request_id=result.request_id,
                prompt=result.prompt,
                completions=completions,
                usage=result.usage,
            )

    client = TestClient(_app(tmp_path, LengthBackend({"hello": "cut"})))
    response = client.post("/v1/messages", json=_body())
    assert response.status_code == 200
    assert response.json()["stop_reason"] == "max_tokens"


# ---------------------------------------------------------------------------
# Tool loop


def test_two_turn_tool_loop(tmp_path):
    client = TestClient(_app(tmp_path, ToolLoopBackend()))
    first = client.post(
        "/v1/messages",
        json=_body(tools=[_tool()], messages=[{"role": "user", "content": "add"}]),
    )
    assert first.status_code == 200
    message = first.json()
    assert message["stop_reason"] == "tool_use"
    (tool_use,) = message["content"]
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "add"
    assert tool_use["input"] == {"a": 2, "b": 3}
    assert tool_use["id"]

    second = client.post(
        "/v1/messages",
        json=_body(
            tools=[_tool()],
            messages=[
                {"role": "user", "content": "add"},
                {"role": "assistant", "content": message["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": "5",
                        }
                    ],
                },
            ],
        ),
    )
    assert second.status_code == 200
    final = second.json()
    assert final["stop_reason"] == "end_turn"
    assert final["content"] == [{"type": "text", "text": "The sum is 5."}]


@pytest.mark.parametrize(
    ("is_error", "expected_tool_line"),
    [
        (True, "tool: [tool_result_error]\nboom"),
        (False, "tool: boom"),
    ],
)
def test_tool_result_error_semantics_reach_model(
    tmp_path, is_error, expected_tool_line
):
    backend = MockBackend({"boom": "handled"})
    client = TestClient(_app(tmp_path, backend))
    response = client.post(
        "/v1/messages",
        json=_body(
            tools=[_tool()],
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_test",
                            "name": "add",
                            "input": {"a": 1, "b": 2},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_test",
                            "content": "boom",
                            "is_error": is_error,
                        }
                    ],
                },
            ],
        ),
    )

    assert response.status_code == 200
    prompt = backend.prompts_seen[-1]
    assert expected_tool_line in prompt
    assert ("[tool_result_error]" in prompt) is is_error


def test_parallel_tool_calls_keep_ids_and_stream_indices(tmp_path):
    client = TestClient(_app(tmp_path, ParallelToolBackend()))
    request = _body(tools=[_tool()], messages=[{"role": "user", "content": "add"}])
    unary = client.post("/v1/messages", json=request)
    assert unary.status_code == 200
    blocks = unary.json()["content"]
    assert [block["type"] for block in blocks] == ["tool_use", "tool_use"]
    assert blocks[0]["input"] == {"a": 1, "b": 2}
    assert blocks[1]["input"] == {"a": 3, "b": 4}
    assert blocks[0]["id"] != blocks[1]["id"]

    streamed = client.post("/v1/messages", json={**request, "stream": True})
    assert streamed.status_code == 200
    events = _events(streamed.text)
    starts = [
        (payload["index"], payload["content_block"])
        for name, payload in events
        if name == "content_block_start"
    ]
    assert [index for index, _block in starts] == [0, 1]
    assert all(block["type"] == "tool_use" for _index, block in starts)
    deltas = [
        (payload["index"], payload["delta"]["partial_json"])
        for name, payload in events
        if name == "content_block_delta"
    ]
    assert [json.loads(partial) for _index, partial in deltas] == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
    # streamed and unary requests reconstruct to the same tool calls
    assert [index for index, _partial in deltas] == [0, 1]
    (delta_payload,) = [
        payload for name, payload in events if name == "message_delta"
    ]
    assert delta_payload["delta"]["stop_reason"] == "tool_use"


def test_disable_parallel_tool_use_violation_is_anthropic_error(tmp_path):
    client = TestClient(_app(tmp_path, ParallelToolBackend()))
    response = client.post(
        "/v1/messages",
        json=_body(
            tools=[_tool()],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=[{"role": "user", "content": "add"}],
        ),
    )
    assert response.status_code == 502
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "api_error"


# ---------------------------------------------------------------------------
# Streaming


def test_live_stream_event_order_and_unary_equivalence(tmp_path):
    client = TestClient(_app(tmp_path))
    unary = client.post("/v1/messages", json=_body())
    streamed = client.post("/v1/messages", json=_body(stream=True))
    assert streamed.status_code == 200
    _assert_sse_response_headers(streamed)
    events = _events(streamed.text)
    names = [name for name, _payload in events]
    assert names[0] == "message_start"
    assert names[1] == "content_block_start"
    assert names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert set(names[2:-3]) == {"content_block_delta"}
    assert _text_from_events(events) == unary.json()["content"][0]["text"]
    (delta_payload,) = [p for name, p in events if name == "message_delta"]
    assert delta_payload["delta"]["stop_reason"] == "end_turn"
    assert delta_payload["usage"]["output_tokens"] > 0


def test_buffered_tool_stream_opens_immediately_and_pings(tmp_path, monkeypatch):
    # Claude Code aborts a stream that goes silent for 300s; the buffered path
    # must flush message_start before generation completes and keep the byte
    # watchdog fed with protocol-valid ping events while generation runs.
    monkeypatch.setattr(
        "kairyu.entrypoints.server.messages_service._KEEPALIVE_SECONDS", 0.02
    )

    class SlowToolBackend(ToolLoopBackend):
        async def generate(self, request):
            await asyncio.sleep(0.2)
            return await super().generate(request)

    client = TestClient(_app(tmp_path, SlowToolBackend()))
    response = client.post(
        "/v1/messages",
        json=_body(
            stream=True,
            tools=[_tool()],
            messages=[{"role": "user", "content": "add"}],
        ),
    )
    assert response.status_code == 200
    _assert_sse_response_headers(response)
    events = _events(response.text)
    names = [name for name, _payload in events]
    assert names[0] == "message_start"
    assert "ping" in names
    assert names.index("ping") < names.index("content_block_start")
    tool_starts = [
        payload["content_block"]
        for name, payload in events
        if name == "content_block_start"
    ]
    assert tool_starts[0]["type"] == "tool_use"
    assert tool_starts[0]["input"] == {}
    assert names[-1] == "message_stop"


def test_mid_stream_failure_emits_anthropic_error_event(tmp_path):
    class ExplodingBackend(MockBackend):
        async def stream(self, request):
            emitted = 0
            async for partial in super().stream(request):
                yield partial
                emitted += 1
                if emitted == 1:
                    raise RuntimeError("boom")

    client = TestClient(_app(tmp_path, ExplodingBackend({"hello": "long answer"})))
    response = client.post("/v1/messages", json=_body(stream=True))
    assert response.status_code == 200
    events = _events(response.text)
    names = [name for name, _payload in events]
    assert names[-1] == "error"
    (error_payload,) = [p for name, p in events if name == "error"]
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"
    assert "RuntimeError" in error_payload["error"]["message"]
    assert "message_stop" not in names


# ---------------------------------------------------------------------------
# Orchestrated (AUTO) models — the primary Claude Code target


def test_orchestrated_model_non_streaming_and_relay_equivalence(tmp_path):
    client = TestClient(_auto_app(tmp_path))
    body = _body(model="kairyu-auto")
    unary = client.post("/v1/messages", json=body)
    assert unary.status_code == 200
    message = unary.json()
    assert message["content"] == [{"type": "text", "text": "orchestrated hello"}]
    assert message["usage"]["input_tokens"] > 0

    streamed = client.post("/v1/messages", json={**body, "stream": True})
    assert streamed.status_code == 200
    _assert_sse_response_headers(streamed)
    events = _events(streamed.text)
    assert _text_from_events(events) == "orchestrated hello"
    (delta_payload,) = [p for name, p in events if name == "message_delta"]
    assert delta_payload["usage"]["output_tokens"] > 0
    # Kairyu's private orchestration reasoning must never surface: the wire
    # carries no reasoning_content key anywhere on this route.
    assert "reasoning_content" not in streamed.text
    assert "reasoning_content" not in unary.text


def test_orchestrated_unknown_model_is_anthropic_404(tmp_path):
    client = TestClient(_auto_app(tmp_path))
    response = client.post("/v1/messages", json=_body(model="kairyu-missing"))
    assert response.status_code == 404
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "not_found_error"


# ---------------------------------------------------------------------------
# Error envelope (never the OpenAI shape, never FastAPI's 422 detail)


@pytest.mark.parametrize(
    ("body", "status", "fragment"),
    [
        (_body(model="nope"), 404, "not found"),
        ({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, 400, "max_tokens"),
        (
            _body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "data": "aaaa"},
                            }
                        ],
                    }
                ]
            ),
            400,
            "image",
        ),
        (
            _body(thinking={"type": "enabled", "budget_tokens": 2048}),
            400,
            "adaptive",
        ),
        (_body(output_config={"format": {"type": "json_schema"}}), 400, "output_config"),
        (_body(context_management={"edits": []}), 400, "context_management"),
        (
            _body(tools=[{**_tool(), "eager_input_streaming": True}]),
            400,
            "eager_input_streaming",
        ),
    ],
)
def test_error_envelope_matrix(tmp_path, body, status, fragment):
    client = TestClient(_app(tmp_path))
    response = client.post("/v1/messages", json=body)
    assert response.status_code == status
    payload = response.json()
    assert payload["type"] == "error"
    assert set(payload) == {"type", "error", "request_id"}
    assert payload["request_id"].startswith("req_")
    assert fragment in payload["error"]["message"]


def test_malformed_json_and_bad_version_are_anthropic_400(tmp_path):
    client = TestClient(_app(tmp_path))
    malformed = client.post(
        "/v1/messages",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"]["type"] == "invalid_request_error"

    bad_version = client.post(
        "/v1/messages", json=_body(), headers={"anthropic-version": "1999-01-01"}
    )
    assert bad_version.status_code == 400
    assert "anthropic-version" in bad_version.json()["error"]["message"]

    # Open beta list: any anthropic-beta value must be accepted, not allowlisted.
    beta = client.post(
        "/v1/messages",
        json=_body(),
        headers={"anthropic-beta": "context-1,future-capability-2099"},
    )
    assert beta.status_code == 200


def test_unknown_extra_fields_are_tolerated(tmp_path):
    # Claude Code gains request fields over releases; unknown extras must not
    # fail parsing (known semantics-changing fields are rejected explicitly).
    client = TestClient(_app(tmp_path))
    response = client.post(
        "/v1/messages", json=_body(some_future_field={"x": 1})
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Usage, probes, count_tokens


def test_usage_ledger_records_public_usage(tmp_path):
    client = TestClient(_app(tmp_path))
    response = client.post("/v1/messages", json=_body())
    assert response.status_code == 200
    usage = response.json()["usage"]
    totals = UsageLedger(tmp_path / "usage.jsonl").totals()["default"]
    assert totals["requests"] == 1
    assert totals["prompt_tokens"] == usage["input_tokens"]
    assert totals["completion_tokens"] == usage["output_tokens"]


def test_hello_probe_is_open_and_count_tokens_is_anthropic_404(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KAIRYU_API_KEYS", "sk-good")
    app = create_legacy_app(
        {"m": MockBackend({"hello": "hi"})},
        settings=ServerSettings(
            usage_ledger_path=str(tmp_path / "usage.jsonl"),
            api_keys_env="KAIRYU_API_KEYS",
        ),
    )
    client = TestClient(app)
    # best-effort startup probe: succeeds without credentials
    assert client.head("/api/hello").status_code == 200
    # optional endpoint: an Anthropic-shaped 404 triggers Claude Code's
    # client-side token-counting fallback
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "sk-good"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_body_limit_413_uses_anthropic_envelope(tmp_path):
    client = TestClient(
        create_legacy_app(
            {"m": MockBackend({"hello": "hi"})},
            settings=ServerSettings(
                usage_ledger_path=str(tmp_path / "usage.jsonl"),
                max_chat_body_bytes=128,
            ),
        )
    )
    response = client.post(
        "/v1/messages",
        content=json.dumps(_body(messages=[{"role": "user", "content": "x" * 256}])),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "request_too_large"


def test_tenant_rate_limit_429_uses_anthropic_envelope(tmp_path):
    from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits

    client = TestClient(
        create_legacy_app(
            {"m": MockBackend({"hello": "hi"})},
            settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
            tenant_config=TenantConfig(
                key_tenants={"noisy-key": "noisy"},
                limits={"noisy": TenantLimits(requests_per_minute=1)},
            ),
            resolved_api_keys=frozenset({"noisy-key"}),
        )
    )
    responses = [
        client.post(
            "/v1/messages",
            json=_body(),
            headers={"x-api-key": "noisy-key"},
        )
        for _ in range(5)
    ]
    rejected = [r for r in responses if r.status_code == 429]
    assert rejected, [r.status_code for r in responses]
    payload = rejected[0].json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "rate_limit_error"
    assert rejected[0].headers["retry-after"] == "1"


def test_mid_conversation_system_messages_are_accepted(tmp_path):
    # Claude Code sends {"role": "system"} entries inside messages (the
    # operator channel); rejecting them breaks every real Claude Code turn.
    client = TestClient(_app(tmp_path))
    response = client.post(
        "/v1/messages",
        json=_body(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "system", "content": [{"type": "text", "text": "Terse."}]},
                {"role": "user", "content": "hello again"},
            ]
        ),
    )
    assert response.status_code == 200
    assert response.json()["content"][0]["type"] == "text"
