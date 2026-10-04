"""Shared helpers for the Responses API server tests."""

from __future__ import annotations

import json
from dataclasses import replace

import openai
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.outputs import CompletionOutput
from tests.server._legacy_chat import create_legacy_app


def _app(tmp_path, backend=None, **kwargs):
    return create_legacy_app(
        {"m": backend or MockBackend({"hello": "streamed hello"})},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        **kwargs,
    )


def _sdk(http: TestClient) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=str(http.base_url) + "/v1",
        api_key="sk-local",
        http_client=http,
        _strict_response_validation=True,
    )


def _sse_events(body: str) -> list[dict]:
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
