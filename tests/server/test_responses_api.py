"""Responses API streaming, function tools, state, and SDK compatibility."""

from __future__ import annotations

from dataclasses import replace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, UsageLedger
from kairyu.outputs import CompletionOutput


def _app(tmp_path, backend=None, **kwargs):
    return create_app(
        {"m": backend or MockBackend({"hello": "streamed hello"})},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        **kwargs,
    )


def _sdk(http: TestClient) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=str(http.base_url) + "/v1",
        api_key="sk-local",
        http_client=http,
    )


def _sse_events(body: str) -> list[dict]:
    import json

    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _tool():
    return {
        "type": "function",
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _namespace_tool():
    return {
        "type": "namespace",
        "name": "codex",
        "description": "Local coding tools.",
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command.",
                "defer_loading": True,
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
    }


class ToolLoopBackend(MockBackend):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if "tool: 5" in request.prompt:
            text = "The sum is 5."
        else:
            text = '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=(1, 2, 3),
                    finish_reason="stop",
                ),
            ),
            usage=GenerationUsage(prompt_tokens=11, completion_tokens=3),
        )


class LengthBackend(MockBackend):
    async def generate(self, request):
        result = await super().generate(request)
        return replace(
            result,
            completions=tuple(
                replace(completion, finish_reason="length")
                for completion in result.completions
            ),
        )


def test_official_sdk_typed_text_stream_and_final_response(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        client = _sdk(http)
        with client.responses.stream(model="m", input="hello") as stream:
            events = list(stream)
            final = stream.get_final_response()

    types = [event.type for event in events]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert final.status == "completed"
    assert final.output_text == "streamed hello"
    assert final.usage.input_tokens_details.cached_tokens == 0
    assert final.usage.output_tokens_details.reasoning_tokens == 0


def test_create_stream_true_returns_sdk_typed_events(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        events = list(_sdk(http).responses.create(model="m", input="hello", stream=True))
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    deltas = [
        event.delta for event in events if event.type == "response.output_text.delta"
    ]
    assert "".join(deltas) == "streamed hello"


@pytest.mark.asyncio
async def test_official_async_sdk_stream_is_typed(tmp_path):
    transport = httpx.ASGITransport(app=_app(tmp_path))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as http:
        client = openai.AsyncOpenAI(
            base_url="http://test/v1",
            api_key="sk-local",
            http_client=http,
        )
        stream = await client.responses.create(model="m", input="hello", stream=True)
        events = [event async for event in stream]
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    assert events[-1].response.output_text == "streamed hello"


def test_incomplete_stream_has_consistent_item_and_terminal_status(tmp_path):
    backend = LengthBackend({"hello": "truncated"})
    with TestClient(_app(tmp_path, backend)) as http:
        events = list(_sdk(http).responses.create(model="m", input="hello", stream=True))

    item_done = next(event for event in events if event.type == "response.output_item.done")
    terminal = events[-1]
    assert item_done.item.status == "incomplete"
    assert terminal.type == "response.incomplete"
    assert terminal.response.status == "incomplete"
    assert terminal.response.output[0].status == "incomplete"
    assert terminal.response.incomplete_details.reason == "max_output_tokens"


def test_incomplete_unary_has_consistent_item_and_response_status(tmp_path):
    backend = LengthBackend({"hello": "truncated"})
    with TestClient(_app(tmp_path, backend)) as http:
        response = _sdk(http).responses.create(model="m", input="hello")

    assert response.status == "incomplete"
    assert response.output[0].status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"


def test_function_tool_unary_loop_with_previous_response_id(tmp_path):
    backend = ToolLoopBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        client = _sdk(http)
        first = client.responses.create(
            model="m",
            input="Add 2 and 3.",
            tools=[_tool()],
            tool_choice="required",
        )
        assert len(first.output) == 1
        call = first.output[0]
        assert call.type == "function_call"
        assert call.name == "add"
        assert call.arguments == '{"a": 2, "b": 3}'

        second = client.responses.create(
            model="m",
            previous_response_id=first.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "tool: 5",
                }
            ],
            tools=[_tool()],
        )

    assert second.output_text == "The sum is 5."
    assert len(backend.requests) == 2
    assert backend.requests[0].tools[0]["function"]["name"] == "add"
    assert backend.requests[0].tool_choice == "required"
    assert "tool: 5" in backend.requests[1].prompt


def test_function_tool_stream_has_canonical_argument_lifecycle(tmp_path):
    with TestClient(_app(tmp_path, ToolLoopBackend())) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "Add 2 and 3.",
                "stream": True,
                "tools": [_tool()],
                "tool_choice": {"type": "function", "name": "add"},
            },
        )
    assert response.status_code == 200
    events = _sse_events(response.text)
    types = [event["type"] for event in events]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    added = events[2]["item"]
    done = events[5]["item"]
    assert added["id"] == done["id"]
    assert added["call_id"] == done["call_id"]
    assert added["arguments"] == ""
    assert done["arguments"] == '{"a": 2, "b": 3}'
    assert events[-1]["response"]["output"] == [done]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"mystery": True}, "unsupported request fields"),
        ({"tools": [{"type": "web_search"}]}, "expected 'function'"),
        ({"context_management": [{"type": "compaction"}]}, "context_management"),
        ({"include": ["message.output_text.logprobs"]}, "unsupported include values"),
        ({"service_tier": "priority"}, "service_tier is not supported"),
        (
            {"stream_options": {"include_obfuscation": True}},
            "stream obfuscation is not supported",
        ),
        ({"text": {"verbosity": "maximum"}}, "text.verbosity"),
    ],
)
def test_unsupported_or_unsafe_fields_fail_before_dispatch(tmp_path, payload, message):
    backend = MockBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={"model": "m", "input": "hello", **payload},
        )
    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert backend.prompts_seen == ()


@pytest.mark.parametrize(
    "part",
    [
        {
            "type": "input_text",
            "text": "hello",
            "input_audio": {"data": "AA=="},
        },
        {
            "type": "text",
            "text": "hello",
            "image_url": {"url": "https://example.test/image"},
        },
        {
            "type": "output_text",
            "text": "hello",
            "prompt_token_ids": [1, 2, 3],
        },
    ],
)
def test_response_text_parts_never_drop_alternate_prompt_payloads(
    tmp_path, part
):
    backend = MockBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [part],
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert "unsupported fields" in response.json()["error"]["message"]
    assert backend.prompts_seen == ()


def test_response_output_text_replay_metadata_remains_compatible(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "prior answer",
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                ],
            },
        )

    assert response.status_code == 200


def test_parallel_tool_calls_false_accepts_one_call_and_rejects_multiple(tmp_path):
    one = ToolLoopBackend()
    with TestClient(_app(tmp_path, one)) as http:
        accepted = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "Add 2 and 3.",
                "tools": [_tool()],
                "parallel_tool_calls": False,
            },
        )

    class TwoCallBackend(ToolLoopBackend):
        async def generate(self, request):
            result = await super().generate(request)
            text = (
                '<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'
                '<tool_call>{"name":"add","arguments":{"a":4,"b":5}}</tool_call>'
            )
            return replace(
                result,
                completions=(
                    CompletionOutput(index=0, text=text, token_ids=(1, 2, 3)),
                ),
            )

    with TestClient(_app(tmp_path, TwoCallBackend())) as http:
        rejected = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "Add pairs.",
                "tools": [_tool()],
                "parallel_tool_calls": False,
            },
        )
    assert accepted.status_code == 200
    assert accepted.json()["output"][0]["type"] == "function_call"
    assert rejected.status_code == 502
    assert rejected.json()["error"]["code"] == "parallel_tool_calls_not_satisfied"


def test_namespace_tools_round_trip_without_internal_name_leak(tmp_path):
    class NamespaceBackend(ToolLoopBackend):
        async def generate(self, request):
            self.requests.append(request)
            text = (
                '<tool_call>{"name":"codex__exec_command",'
                '"arguments":{"cmd":"printf PASS"}}</tool_call>'
            )
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(index=0, text=text, token_ids=(1, 2, 3)),
                ),
                usage=GenerationUsage(prompt_tokens=5, completion_tokens=3),
            )

    backend = NamespaceBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "Use the command tool.",
                "tools": [_namespace_tool()],
                "parallel_tool_calls": False,
            },
        )
    assert response.status_code == 200
    assert backend.requests[0].tools[0]["function"]["name"] == "codex__exec_command"
    assert "defer_loading" not in backend.requests[0].tools[0]["function"]
    assert "Call at most one function" in backend.requests[0].prompt
    item = response.json()["output"][0]
    assert item["type"] == "function_call"
    assert item["namespace"] == "codex"
    assert item["name"] == "exec_command"
    assert "codex__" not in str(item)


def test_named_namespace_tool_choice_resolves_to_internal_function(tmp_path):
    backend = ToolLoopBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "Use the command tool.",
                "tools": [_namespace_tool()],
                "tool_choice": {
                    "type": "function",
                    "namespace": "codex",
                    "name": "exec_command",
                },
            },
        )
    assert response.status_code == 502
    # The mock emitted another valid function, proving that the namespace
    # selection reached the shared named-tool enforcement boundary.
    assert response.json()["error"]["message"] == "upstream model did not satisfy tool_choice"
    assert backend.requests[0].tool_choice["function"]["name"] == "codex__exec_command"


def test_namespace_tool_stream_keeps_public_name_and_ids_stable(tmp_path):
    class NamespaceBackend(MockBackend):
        async def generate(self, request):
            text = (
                '<tool_call>{"name":"codex__exec_command",'
                '"arguments":{"cmd":"printf PASS"}}</tool_call>'
            )
            return GenerationResult(
                request_id=request.request_id,
                prompt=request.prompt,
                completions=(
                    CompletionOutput(index=0, text=text, token_ids=(1, 2, 3)),
                ),
                usage=GenerationUsage(prompt_tokens=5, completion_tokens=3),
            )

    with TestClient(_app(tmp_path, NamespaceBackend())) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "Use the command tool.",
                "stream": True,
                "tools": [_namespace_tool()],
            },
        )
    events = _sse_events(response.text)
    added = next(
        event["item"]
        for event in events
        if event["type"] == "response.output_item.added"
    )
    done = next(
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
    )
    completed = events[-1]["response"]["output"][0]
    for item in (added, done, completed):
        assert item["namespace"] == "codex"
        assert item["name"] == "exec_command"
        assert item["id"] == added["id"]
        assert item["call_id"] == added["call_id"]
    assert added["arguments"] == ""
    assert done["arguments"] == '{"cmd": "printf PASS"}'


def test_namespaced_function_history_survives_legacy_chat_rendering(tmp_path):
    backend = MockBackend({"tool: failed": "Recovered."})
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {
                        "type": "function_call",
                        "id": "fc_prior",
                        "call_id": "call_prior",
                        "namespace": "codex",
                        "name": "exec_command",
                        "arguments": '{"cmd":"false"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_prior",
                        "output": "tool: failed",
                    },
                ],
                "tools": [_namespace_tool()],
            },
        )
    assert response.status_code == 200
    prompt = backend.prompts_seen[0]
    assert '"name":"codex__exec_command"' in prompt
    assert "tool: failed" in prompt


def test_phase_and_opaque_function_arguments_survive_stateless_history(tmp_path):
    backend = MockBackend({"tool output": "done"})
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": "working",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_opaque",
                        "name": "add",
                        "arguments": "provider-specific opaque arguments",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_opaque",
                        "output": "tool output",
                    },
                ],
                "tools": [_tool()],
            },
        )
        stored_items = http.app.state.response_store.get(response.json()["id"])
    assert response.status_code == 200
    assert "provider-specific opaque arguments" in backend.prompts_seen[0]
    assert stored_items is not None
    assert stored_items[0]["phase"] == "commentary"


def test_flattened_namespace_name_collisions_fail_before_dispatch(tmp_path):
    backend = MockBackend()
    bare = {
        "type": "function",
        "name": "codex__exec_command",
        "parameters": {"type": "object", "properties": {}},
    }
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "tools": [bare, _namespace_tool()],
            },
        )
    assert response.status_code == 400
    assert "duplicate function name" in response.json()["error"]["message"]
    assert backend.prompts_seen == ()


def test_codex_disabled_web_search_declaration_is_a_truthful_noop(tmp_path):
    backend = MockBackend({"hello": "PASS"})
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "tools": [
                    _tool(),
                    {"type": "web_search", "external_web_access": False},
                ],
            },
        )
    assert response.status_code == 200
    assert len(backend.prompts_seen) == 1
    assert response.json()["tools"][-1] == {
        "type": "web_search",
        "external_web_access": False,
    }


def test_unknown_function_call_output_fails_before_dispatch(tmp_path):
    backend = MockBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_missing",
                        "output": "nope",
                    }
                ],
            },
        )
    assert response.status_code == 400
    assert "unknown call_id" in response.json()["error"]["message"]
    assert backend.prompts_seen == ()


def test_store_false_and_cross_tenant_stream_ids_are_not_readable(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KAIRYU_RESPONSES_KEYS", "key-a,key-b")
    app = create_app(
        {"m": MockBackend({"hello": "tenant response"})},
        settings=ServerSettings(
            api_keys_env="KAIRYU_RESPONSES_KEYS",
            usage_ledger_path=str(tmp_path / "usage.jsonl"),
        ),
        tenant_config=TenantConfig(
            key_tenants={"key-a": "tenant-a", "key-b": "tenant-b"}
        ),
    )
    with TestClient(app) as http:
        first = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={"model": "m", "input": "hello", "stream": True},
        )
        response_id = _sse_events(first.text)[-1]["response"]["id"]
        cross_tenant = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-b"},
            json={
                "model": "m",
                "input": "again",
                "previous_response_id": response_id,
            },
        )
        unstored = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={"model": "m", "input": "hello", "store": False},
        )
        not_found = http.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={
                "model": "m",
                "input": "again",
                "previous_response_id": unstored.json()["id"],
            },
        )
    assert first.status_code == 200
    assert cross_tenant.status_code == 404
    assert not_found.status_code == 404


def test_stream_failure_emits_error_and_failed_without_storing(tmp_path):
    class FailingStreamBackend(MockBackend):
        async def stream(self, request):
            partial = await self.generate(request)
            yield replace(partial, finished=False)
            raise RuntimeError("secret upstream endpoint")

    with TestClient(_app(tmp_path, FailingStreamBackend())) as http:
        response = http.post(
            "/v1/responses",
            json={"model": "m", "input": "hello", "stream": True},
        )
        events = _sse_events(response.text)
        response_id = events[0]["response"]["id"]
        lookup = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "again",
                "previous_response_id": response_id,
            },
        )
    assert [event["type"] for event in events[-2:]] == ["error", "response.failed"]
    failed = events[-1]["response"]
    assert failed["status"] == "failed"
    assert failed["output"][0]["status"] == "incomplete"
    assert failed["output"][0]["content"][0]["text"]
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    try:
        totals = ledger.totals()["default"]
    finally:
        ledger.close()
    assert failed["usage"]["input_tokens"] == totals["prompt_tokens"]
    assert failed["usage"]["output_tokens"] == totals["completion_tokens"]
    assert "secret upstream endpoint" not in response.text
    assert lookup.status_code == 404
