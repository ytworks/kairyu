"""Anthropic Messages adapter: envelope, tool loop, streaming, errors (#508)."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.chat_template import ChatTemplate
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


class ReasoningRecordingBackend(MockBackend):
    def __init__(self):
        super().__init__({"hello": "reasoned"})
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return await super().generate(request)


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


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("minimal", "low"),
        ("medium", "high"),
        ("xhigh", "max"),
    ],
)
def test_output_config_effort_reaches_backend(tmp_path, effort, expected):
    backend = ReasoningRecordingBackend()
    client = TestClient(_app(tmp_path, backend))
    response = client.post(
        "/v1/messages",
        json=_body(output_config={"effort": effort}),
    )

    assert response.status_code == 200
    assert backend.requests[-1].reasoning_effort == expected


@pytest.mark.parametrize("thinking_type", ["adaptive", "disabled"])
def test_compatible_thinking_modes_are_accepted_as_noops(tmp_path, thinking_type):
    backend = ReasoningRecordingBackend()
    response = TestClient(_app(tmp_path, backend)).post(
        "/v1/messages",
        json=_body(thinking={"type": thinking_type}),
    )

    assert response.status_code == 200
    assert backend.requests[-1].reasoning_effort is None


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
    deltas: dict[int, str] = {}
    for payload in [
        payload
        for name, payload in events
        if name == "content_block_delta"
    ]:
        index = payload["index"]
        deltas[index] = deltas.get(index, "") + payload["delta"]["partial_json"]
    assert [json.loads(partial) for partial in deltas.values()] == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]
    # streamed and unary requests reconstruct to the same tool calls
    assert list(deltas) == [0, 1]
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
        (
            _body(
                thinking={"type": "disabled"},
                output_config={"effort": "medium"},
            ),
            400,
            "conflicts with output_config.effort",
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
    # The app's bounded async writer flushes in the background; poll briefly
    # so the read side never races the accepted record.
    deadline = time.monotonic() + 5.0
    while True:
        all_totals = UsageLedger(tmp_path / "usage.jsonl").totals()
        if "default" in all_totals or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    totals = all_totals["default"]
    assert totals["requests"] == 1
    assert totals["prompt_tokens"] == usage["input_tokens"]
    assert totals["completion_tokens"] == usage["output_tokens"]


def test_hello_probe_is_open_and_count_tokens_is_served(
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
    # count_tokens is served (behind the same auth) and returns a count
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "sk-good"},
    )
    assert response.status_code == 200
    assert type(response.json()["input_tokens"]) is int


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


# ---------------------------------------------------------------------------
# #573: incremental tool streaming


_QWEN_TEMPLATE = (
    "{{ '<function=n><parameter=v></parameter></function>' if false }}"
    "{{ messages[0].content }}"
)
_DSML_TEMPLATE = (
    "{{ '<｜DSML｜tool_calls>' if false }}{{ messages[0].content }}"
)
_LLAMA_TEMPLATE = "{{ '<|python_tag|>' if false }}{{ messages[0].content }}"
_QWEN_CALL = (
    "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n"
    "<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>"
)
_DSML_CALL = (
    '<｜DSML｜tool_calls><｜DSML｜invoke name="add">'
    '<｜DSML｜parameter name="a" string="false">1</｜DSML｜parameter>'
    '<｜DSML｜parameter name="b" string="false">2</｜DSML｜parameter>'
    "</｜DSML｜invoke></｜DSML｜tool_calls>"
)


def _canned_backend(text: str):
    class CannedBackend(MockBackend):
        async def generate(self, request):
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

    return CannedBackend()


def _blocks_from_events(events: list[tuple[str, dict]]) -> list[dict]:
    """Reconstruct Anthropic content blocks from streamed events."""

    blocks: dict[int, dict] = {}
    for name, payload in events:
        if name == "content_block_start":
            block = dict(payload["content_block"])
            blocks[payload["index"]] = block
            if block["type"] == "tool_use":
                block["_json"] = ""
        elif name == "content_block_delta":
            delta = payload["delta"]
            block = blocks[payload["index"]]
            if delta["type"] == "text_delta":
                block["text"] += delta["text"]
            else:
                block["_json"] += delta["partial_json"]
    out = []
    for index in sorted(blocks):
        block = blocks[index]
        if block["type"] == "tool_use":
            block["input"] = json.loads(block.pop("_json") or "{}")
        out.append(block)
    return out


class StagedArgumentsBackend(MockBackend):
    """Pauses after a declared tool's argument object has started."""

    async def stream(self, request):
        prefix = '<tool_call>{"name":"add","arguments":{"a":1'
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0, text=prefix, token_ids=(1,), finish_reason=None
                ),
            ),
            finished=False,
        )
        await asyncio.sleep(0.2)
        complete = prefix + ',"b":2}}</tool_call>'
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=complete,
                    token_ids=(1, 2),
                    finish_reason="stop",
                ),
            ),
            usage=GenerationUsage(prompt_tokens=5, completion_tokens=5),
        )


class StagedParallelBackend(MockBackend):
    """Emits one committed call, sleeps, then a second call."""

    async def stream(self, request):
        first = '<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>'
        second = '<tool_call>{"name":"add","arguments":{"a":3,"b":4}}</tool_call>'
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0, text=first, token_ids=(1,), finish_reason=None
                ),
            ),
            finished=False,
        )
        await asyncio.sleep(0.2)
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=first + second,
                    token_ids=(1, 2),
                    finish_reason="stop",
                ),
            ),
            usage=GenerationUsage(prompt_tokens=5, completion_tokens=5),
        )


def test_tool_arguments_stream_before_generation_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kairyu.entrypoints.server.messages_service._KEEPALIVE_SECONDS", 0.02
    )
    client = TestClient(_app(tmp_path, StagedArgumentsBackend()))
    response = client.post(
        "/v1/messages",
        json=_body(
            stream=True, tools=[_tool()], messages=[{"role": "user", "content": "add"}]
        ),
    )

    events = _events(response.text)
    names = [name for name, _payload in events]
    start = names.index("content_block_start")
    first_delta = names.index("content_block_delta")
    ping = names.index("ping")
    stop = names.index("content_block_stop")
    assert start < first_delta < ping < stop
    argument_fragments = [
        payload["delta"]["partial_json"]
        for name, payload in events
        if name == "content_block_delta"
        and payload["delta"]["type"] == "input_json_delta"
    ]
    assert len(argument_fragments) >= 2
    assert json.loads("".join(argument_fragments)) == {"a": 1, "b": 2}
    assert names[-1] == "message_stop"


def test_invalid_tool_arguments_after_early_commit_end_stream_with_error(tmp_path):
    class InvalidArgumentsBackend(StagedArgumentsBackend):
        async def stream(self, request):
            prefix = '<tool_call>{"name":"add","arguments":{"a":1'
            yield GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(
                        index=0, text=prefix, token_ids=(1,), finish_reason=None
                    ),
                ),
                finished=False,
            )
            complete = prefix + ',"b":oops}}</tool_call>'
            yield GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(
                        index=0,
                        text=complete,
                        token_ids=(1, 2),
                        finish_reason="stop",
                    ),
                ),
                usage=GenerationUsage(prompt_tokens=5, completion_tokens=5),
            )

    response = TestClient(_app(tmp_path, InvalidArgumentsBackend())).post(
        "/v1/messages",
        json=_body(
            stream=True, tools=[_tool()], messages=[{"role": "user", "content": "add"}]
        ),
    )

    names = [name for name, _payload in _events(response.text)]
    assert "content_block_start" in names
    assert "content_block_stop" not in names
    assert names[-1] == "error"
    assert "message_stop" not in names


def test_tool_fragments_are_emitted_before_generation_completes(
    tmp_path, monkeypatch
):
    # The incrementality contract itself (#573): the first call's blocks are
    # emitted while the model is still generating. Proven by body order: the
    # idle pings generated during the inter-yield sleep must appear AFTER the
    # first call's content_block_stop and BEFORE the second call's
    # content_block_start (a buffered implementation would replay every block
    # after the last ping instead).
    monkeypatch.setattr(
        "kairyu.entrypoints.server.messages_service._KEEPALIVE_SECONDS", 0.02
    )
    client = TestClient(_app(tmp_path, StagedParallelBackend()))
    response = client.post(
        "/v1/messages",
        json=_body(
            stream=True, tools=[_tool()], messages=[{"role": "user", "content": "add"}]
        ),
    )
    names = [name for name, _payload in _events(response.text)]
    first_stop = names.index("content_block_stop")
    second_start = names.index("content_block_start", first_stop)
    pings_between = [
        name for name in names[first_stop:second_start] if name == "ping"
    ]
    assert pings_between, names
    assert names[-1] == "message_stop"


def test_text_and_tool_use_coexist_stream_unary_equivalence(tmp_path):
    # /v1/messages keeps preamble text next to tool_use (the Anthropic
    # contract); stream and unary must reconstruct identically.
    text = (
        'Let me add those. <tool_call>{"name":"add","arguments":{"a":2,"b":3}}'
        "</tool_call>"
    )
    request = _body(tools=[_tool()], messages=[{"role": "user", "content": "add"}])
    client = TestClient(_app(tmp_path, _canned_backend(text)))
    unary = client.post("/v1/messages", json=request).json()
    assert [block["type"] for block in unary["content"]] == ["text", "tool_use"]
    assert unary["content"][0]["text"] == "Let me add those. "
    assert unary["content"][1]["input"] == {"a": 2, "b": 3}
    assert unary["stop_reason"] == "tool_use"

    streamed = client.post("/v1/messages", json={**request, "stream": True})
    events = _events(streamed.text)
    reconstructed = _blocks_from_events(events)
    unary_no_ids = [
        {k: v for k, v in block.items() if k != "id"} for block in unary["content"]
    ]
    stream_no_ids = [
        {k: v for k, v in block.items() if k != "id"} for block in reconstructed
    ]
    assert stream_no_ids == unary_no_ids
    (delta_payload,) = [p for n, p in events if n == "message_delta"]
    assert delta_payload["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_tool_aware_stream_emits_unary_equivalent_text_block(tmp_path, text):
    request = _body(
        tools=[_tool()],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": "say nothing"}],
    )
    client = TestClient(_app(tmp_path, _canned_backend(text)))
    unary = client.post("/v1/messages", json=request)
    streamed = client.post("/v1/messages", json={**request, "stream": True})

    assert unary.status_code == 200
    assert unary.json()["content"] == [{"type": "text", "text": text}]
    events = _events(streamed.text)
    assert _blocks_from_events(events) == unary.json()["content"]
    names = [name for name, _payload in events]
    assert names.count("content_block_start") == 1
    assert names.count("content_block_stop") == 1
    assert names.index("content_block_start") < names.index("content_block_stop")
    assert names[-2:] == ["message_delta", "message_stop"]


@pytest.mark.parametrize(
    ("template", "canned"),
    [
        pytest.param(
            _LLAMA_TEMPLATE,
            '{"name":"add","parameters":{"a":1,"b":2}}',
            id="llama",
        ),
        pytest.param(_QWEN_TEMPLATE, _QWEN_CALL, id="qwen"),
        pytest.param(_DSML_TEMPLATE, _DSML_CALL, id="dsml"),
    ],
)
def test_native_protocol_stream_unary_equivalence(tmp_path, template, canned):
    # Each ToolCallProtocol family owns distinct commit rules; the shared
    # scanner must yield the same tool block on both modes.
    client = TestClient(
        create_legacy_app(
            {"m": _canned_backend(canned)},
            chat_templates={"m": ChatTemplate(template)},
            settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        )
    )
    request = _body(tools=[_tool()], messages=[{"role": "user", "content": "add"}])
    unary = client.post("/v1/messages", json=request).json()
    (block,) = unary["content"]
    assert block["type"] == "tool_use"
    assert block["name"] == "add"
    assert block["input"] == {"a": 1, "b": 2}
    assert unary["stop_reason"] == "tool_use"

    streamed = client.post("/v1/messages", json={**request, "stream": True})
    reconstructed = _blocks_from_events(_events(streamed.text))
    assert [
        {k: v for k, v in b.items() if k != "id"} for b in reconstructed
    ] == [{k: v for k, v in b.items() if k != "id"} for b in unary["content"]]


def test_stream_tool_choice_violation_emits_error_event(tmp_path):
    # required/named tool_choice unsatisfied is only knowable at end of
    # stream: it must surface as an SSE error event with no message_stop.
    client = TestClient(_app(tmp_path))
    response = client.post(
        "/v1/messages",
        json=_body(stream=True, tools=[_tool()], tool_choice={"type": "any"}),
    )
    events = _events(response.text)
    names = [name for name, _payload in events]
    assert names[-1] == "error"
    assert "message_stop" not in names
    (error_payload,) = [p for n, p in events if n == "error"]
    assert error_payload["error"]["type"] == "api_error"
    assert "tool_choice" in error_payload["error"]["message"]


def test_stream_parallel_violation_stops_before_second_block(tmp_path):
    # parallel_tool_calls=false fails the moment a second call would open.
    client = TestClient(_app(tmp_path, ParallelToolBackend()))
    response = client.post(
        "/v1/messages",
        json=_body(
            stream=True,
            tools=[_tool()],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=[{"role": "user", "content": "add"}],
        ),
    )
    events = _events(response.text)
    names = [name for name, _payload in events]
    assert names.count("content_block_start") == 1
    assert names[-1] == "error"
    assert "message_stop" not in names


def test_qwen_post_commit_invalidation_stream_errors_unary_downgrades(tmp_path):
    # QWEN's whole-text rule voids committed calls on trailing prose: the
    # stream (bytes already sent) emits an error event; the unary fold keeps
    # the silent downgrade to text. The divergence is itself the contract.
    canned = _QWEN_CALL + "\nAlso, some prose."
    client = TestClient(
        create_legacy_app(
            {"m": _canned_backend(canned)},
            chat_templates={"m": ChatTemplate(_QWEN_TEMPLATE)},
            settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        )
    )
    request = _body(tools=[_tool()], messages=[{"role": "user", "content": "add"}])
    unary = client.post("/v1/messages", json=request).json()
    (block,) = unary["content"]
    assert block["type"] == "text"
    assert _QWEN_CALL in block["text"]
    assert unary["stop_reason"] == "end_turn"

    streamed = client.post("/v1/messages", json={**request, "stream": True})
    names = [name for name, _payload in _events(streamed.text)]
    assert names[-1] == "error"
    assert "message_stop" not in names


def test_undeclared_call_flushes_as_text(tmp_path):
    # A call to an undeclared tool never becomes a block; its envelope
    # survives verbatim as text on both modes (mirrors the unary filter).
    canned = '<tool_call>{"name":"other","arguments":{}}</tool_call>'
    request = _body(tools=[_tool()], messages=[{"role": "user", "content": "add"}])
    client = TestClient(_app(tmp_path, _canned_backend(canned)))
    unary = client.post("/v1/messages", json=request).json()
    assert unary["content"] == [{"type": "text", "text": canned}]
    assert unary["stop_reason"] == "end_turn"
    streamed = client.post("/v1/messages", json={**request, "stream": True})
    assert _blocks_from_events(_events(streamed.text)) == unary["content"]


def test_orchestrated_tool_stream_unary_equivalence_and_gates(tmp_path):
    # AUTO models use the internal raw stream (#573 sentinel) for both modes;
    # public /v1/chat behavior is covered by its own untouched tests.
    text = (
        'Sure. <tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'
    )
    backend = _canned_backend(text)
    client = TestClient(_auto_app(tmp_path, backend))
    request = _body(
        model="kairyu-auto",
        tools=[_tool()],
        messages=[{"role": "user", "content": "add"}],
    )
    unary = client.post("/v1/messages", json=request).json()
    assert [block["type"] for block in unary["content"]] == ["text", "tool_use"]
    assert unary["content"][1]["input"] == {"a": 2, "b": 3}
    assert unary["stop_reason"] == "tool_use"
    assert unary["usage"]["input_tokens"] > 0

    streamed = client.post("/v1/messages", json={**request, "stream": True})
    events = _events(streamed.text)
    assert [
        {k: v for k, v in b.items() if k != "id"}
        for b in _blocks_from_events(events)
    ] == [{k: v for k, v in b.items() if k != "id"} for b in unary["content"]]
    (delta_payload,) = [p for n, p in events if n == "message_delta"]
    assert delta_payload["usage"]["output_tokens"] > 0

    violation = client.post(
        "/v1/messages",
        json={**request, "tool_choice": {"type": "tool", "name": "missing"}},
    )
    assert violation.status_code == 400  # undeclared named tool fails validation
    text_only = _auto_app(tmp_path)
    stream_violation = TestClient(text_only).post(
        "/v1/messages",
        json=_body(
            model="kairyu-auto",
            stream=True,
            tools=[_tool()],
            tool_choice={"type": "any"},
        ),
    )
    names = [name for name, _payload in _events(stream_violation.text)]
    assert names[-1] == "error"
    assert "message_stop" not in names
    unary_violation = TestClient(text_only).post(
        "/v1/messages",
        json=_body(model="kairyu-auto", tools=[_tool()], tool_choice={"type": "any"}),
    )
    assert unary_violation.status_code == 502
    assert unary_violation.json()["error"]["type"] == "api_error"


# ---------------------------------------------------------------------------
# count_tokens


def test_count_tokens_matches_billed_usage_direct(tmp_path):
    # The endpoint's one load-bearing property: the count equals the
    # usage.input_tokens the same body would be billed (tools and system
    # included via the tool-intent-augmented rendered prompt).
    client = TestClient(_app(tmp_path))
    body = _body(tools=[_tool()], system="Be terse.")
    usage = client.post("/v1/messages", json=body).json()["usage"]
    count_body = {k: v for k, v in body.items() if k != "max_tokens"}
    counted = client.post("/v1/messages/count_tokens", json=count_body)
    assert counted.status_code == 200
    assert counted.json() == {"input_tokens": usage["input_tokens"]}
    without_tools = {k: v for k, v in count_body.items() if k != "tools"}
    lower = client.post("/v1/messages/count_tokens", json=without_tools).json()
    assert counted.json()["input_tokens"] > lower["input_tokens"]


def test_count_tokens_does_not_substitute_a_non_authoritative_estimate(tmp_path):
    class NoAuthoritativeCountBackend(MockBackend):
        async def count_prompt_tokens_async(self, prompt: str) -> None:
            return None

    client = TestClient(
        _app(tmp_path, backend=NoAuthoritativeCountBackend({"hello": "hi"}))
    )

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hello there friend"}],
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"
    assert "authoritative token counting" in response.json()["error"]["message"]


def test_count_tokens_orchestrated_pins_multi_stage_billing_definition(
    tmp_path,
):
    # AUTO models have no single tokenizer; count_tokens implements the
    # multi-stage billing definition — the word-split of the L2-rendered
    # orchestration prompt (metering's usage-omitted fallback). AUTO direct
    # routes bill the routed engine's own public accounting instead, which
    # is unknowable before routing; that dichotomy lives in AUTO billing
    # itself and is documented on the route.
    from kairyu.engine.prompt import prompt_text
    from kairyu.entrypoints.server.chat_service import (
        validate_orchestration_chat_input,
    )
    from kairyu.entrypoints.server.metering import _approx_tokens
    from kairyu.entrypoints.server.protocol import ChatCompletionRequest

    client = TestClient(_auto_app(tmp_path))
    messages = [{"role": "user", "content": "hello there friend"}]
    counted = client.post(
        "/v1/messages/count_tokens",
        json={"model": "kairyu-auto", "messages": messages},
    )
    assert counted.status_code == 200
    expected_prompt = validate_orchestration_chat_input(
        ChatCompletionRequest(model="kairyu-auto", messages=messages)
    ).prompt
    assert counted.json() == {
        "input_tokens": _approx_tokens(prompt_text(expected_prompt) or "")
    }


def test_count_tokens_rejects_what_the_main_route_rejects(tmp_path):
    client = TestClient(_app(tmp_path))
    unknown = client.post(
        "/v1/messages/count_tokens",
        json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["type"] == "not_found_error"
    image = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": "aa"}}
                    ],
                }
            ],
        },
    )
    assert image.status_code == 400
    assert image.json()["error"]["type"] == "invalid_request_error"
