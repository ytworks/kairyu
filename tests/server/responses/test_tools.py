"""Responses function and hosted tools: declaration, call lifecycle, call outputs."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.outputs import CompletionOutput
from tests.server.live_server import openai_client
from tests.server.responses._helpers import ToolLoopBackend, _app, _sse_events, _tool


def test_function_tool_unary_loop_with_previous_response_id(tmp_path):
    backend = ToolLoopBackend()
    with openai_client(_app(tmp_path, backend)) as client:
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


def test_function_call_output_accepts_codex_content_item_arrays(tmp_path):
    # Codex serializes structured tool output (e.g. view_image results) as an
    # array of content items; the text parts must feed the tool message while
    # non-text parts stay an explicit capability rejection.
    backend = ToolLoopBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        first = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "Add 2 and 3.",
                "tools": [_tool()],
                "tool_choice": "required",
            },
        )
        call = first.json()["output"][0]
        second = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "previous_response_id": first.json()["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": [
                            {"type": "input_text", "text": "tool: "},
                            {"type": "input_text", "text": "5"},
                        ],
                    }
                ],
                "tools": [_tool()],
            },
        )
        assert second.status_code == 200
        message = second.json()["output"][0]
        assert message["content"][0]["text"] == "The sum is 5."

        image_output = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "previous_response_id": first.json()["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": [
                            {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
                        ],
                    }
                ],
                "tools": [_tool()],
            },
        )
    assert image_output.status_code == 400
    assert "text tool output only" in image_output.json()["error"]["message"]


_CACHED_WEB_SEARCH = {
    "type": "web_search",
    "external_web_access": False,
    "filters": {"allowed_domains": ["example.test"]},
    "search_context_size": "medium",
}


_LIVE_INDEXED_WEB_SEARCH = {
    "type": "web_search",
    "external_web_access": True,
    "indexed_web_access": True,
}


@pytest.mark.parametrize(
    ("tools", "rendered"),
    [
        ([_tool(), _CACHED_WEB_SEARCH], "tools:add"),
        ([_LIVE_INDEXED_WEB_SEARCH], "plain:hello"),
    ],
    ids=["codex-cached-with-function", "codex-live-indexed-only"],
)
def test_hosted_tool_policy(tmp_path, tools, rendered):
    # Codex declares its hosted web_search on every turn (live under
    # full-access sandboxes). Without a configured executor the declaration is
    # accepted and echoed, but the model sees neither the tool nor, when it is
    # the only tool, any tool scaffolding.
    backend = MockBackend()
    template = ChatTemplate(
        {
            "default": "plain:{{ messages[-1].content }}",
            "tool_use": "tools:{% for tool in tools %}{{ tool.function.name }}{% endfor %}",
        }
    )
    app = create_app(
        {"m": backend},
        chat_templates={"m": template},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/responses", json={"model": "m", "input": "hello", "tools": tools}
        )
    assert response.status_code == 200
    assert response.json()["tools"] == tools
    assert backend.prompts_seen == (rendered,)


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
