"""Responses request surface: accepted input items, fields, and rejections."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from tests.server.responses._helpers import ToolLoopBackend, _app, _tool


def test_codex_internal_passthrough_fields_are_accepted_and_dropped(tmp_path):
    # Codex scrubs internal_chat_message_metadata_passthrough /
    # encrypted_function_args for custom providers but sends them verbatim
    # when it targets an OpenAI-shaped base URL (the Harbor/Terminal-Bench
    # setup); observed live against codex-cli 0.147.0.
    backend = ToolLoopBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "tools": [_tool()],
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Add 2 and 3.",
                        "internal_chat_message_metadata_passthrough": {"x": 1},
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "add",
                        "arguments": "{}",
                        "encrypted_function_args": "opaque",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "tool: 5",
                        "internal_chat_message_metadata_passthrough": {"x": 2},
                    },
                ],
            },
        )
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "The sum is 5."


def test_reasoning_input_items_are_accepted_and_dropped(tmp_path):
    # Codex echoes prior reasoning items back with the full history
    # (store:false); they must not fail the request and must not reach the
    # prompt (Kairyu emits and renders no reasoning items).
    backend = MockBackend({"hello": "hi there"})
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_0199...",
                        "summary": [{"type": "summary_text", "text": "thinking"}],
                        "content": None,
                        "encrypted_content": None,
                    },
                    {"type": "message", "role": "user", "content": "hello"},
                ],
            },
        )
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"]
    assert "thinking" not in backend.prompts_seen[0]


def test_reasoning_effort_is_normalized_onto_l3_levels(tmp_path):
    # Codex sends OpenAI-style efforts (minimal..xhigh); /v1/responses must
    # map them onto Kairyu's L3 levels instead of silently dropping them.
    class RecordingBackend(MockBackend):
        def __init__(self):
            super().__init__({"hello": "hi there"})
            self.requests = []

        async def generate(self, request):
            self.requests.append(request)
            return await super().generate(request)

    backend = RecordingBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "reasoning": {"effort": "medium"},
            },
        )
    assert response.status_code == 200
    assert backend.requests[-1].reasoning_effort == "high"


def test_unknown_reasoning_effort_is_rejected(tmp_path):
    with TestClient(_app(tmp_path)) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "reasoning": {"effort": "ultra"},
            },
        )
    assert response.status_code == 400
    assert "reasoning.effort" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"mystery": True}, "unsupported request fields"),
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
