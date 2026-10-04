"""Responses namespace tools: flattened internal names never leak to the client."""

from __future__ import annotations

from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.outputs import CompletionOutput
from tests.server.responses._helpers import ToolLoopBackend, _app, _sse_events


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
