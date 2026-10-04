"""Prompt overflow is classified on every public surface (M20 WP-04).

Codex compacts only on an in-band ``response.failed`` whose error code is
``context_length_exceeded``; a pre-stream 400 or a 502 is terminal for it.
Chat and Messages clients look for the OpenAI code and Anthropic's "prompt is
too long" text. Each case drives one public surface from the HTTP request to
the rendered error, over the native engine (Kairyu's own tokenizer preflight),
an OpenAI-compatible upstream (vLLM rejects the prompt), or an AUTO model
(orchestration preflight of the client prompt).
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.mock import MockBackend
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.request_errors import ContextLengthExceededError
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.router import RouteThresholds, RuleRouter
from tests.server._legacy_chat import create_legacy_app

_MAX_MODEL_LEN = 8
_LONG_TEXT = " ".join(["overflow"] * 32)
_UPSTREAM_SECRET = "SECRET-UPSTREAM-DETAIL"
# vLLM's nested OpenAI-style error body for a prompt that cannot fit.
_VLLM_OVERFLOW_BODY = {
    "error": {
        "message": (
            "This model's maximum context length is 8 tokens. However, you "
            f"requested 40 tokens in the messages. {_UPSTREAM_SECRET}"
        ),
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    }
}
_RESPONSES_MESSAGE = (
    "Your input exceeds the context window of this model. "
    "Please adjust your input and try again."
)
_TOOL = {
    "type": "function",
    "name": "add",
    "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {}},
}


def _upstream_backend() -> OpenAICompatBackend:
    return OpenAICompatBackend(
        base_url="http://vllm.internal:8000/v1",
        model="served",
        api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json=_VLLM_OVERFLOW_BODY)
        ),
        upstream="vllm",
    )


def _overflow_app(tmp_path, source: str):
    """Return ``(app, model, owned_backends)`` for one overflow source."""

    settings = ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl"))
    if source == "upstream":
        backend = _upstream_backend()
        return create_legacy_app({"m": backend}, settings=settings), "m", [backend]
    native = KairyuBackend(num_pages=64, max_model_len=_MAX_MODEL_LEN)
    if source == "native":
        return create_legacy_app({"m": native}, settings=settings), "m", [native]
    app = create_legacy_app(
        {},
        orchestrators={
            "kairyu-auto": Orchestrator({"tier1": native, "tier2": native})
        },
        settings=settings,
    )
    return app, "kairyu-auto", [native]


async def _post(app, path: str, payload: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=payload)


def _responses_events(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _anthropic_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    name = None
    for line in body.splitlines():
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            events.append((name, json.loads(line.removeprefix("data: "))))
    return events


def _chat_error_payloads(body: str) -> list[dict]:
    payloads = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            payload = json.loads(line.removeprefix("data: "))
            if "error" in payload:
                payloads.append(payload["error"])
    return payloads


async def _assert_responses_overflow(app, response, *, stream, source) -> None:
    expected = {
        "message": _RESPONSES_MESSAGE,
        "type": "invalid_request_error",
        "param": "input",
        "code": "context_length_exceeded",
    }
    if not stream:
        assert response.status_code == 400
        assert response.json() == {"error": expected}
        return
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _responses_events(response.text)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    types = [event["type"] for event in events]
    if source == "upstream":
        # vLLM reports the overflow only after dispatch; the stream is open.
        assert types[:2] == ["response.created", "response.in_progress"]
        assert types[-2:] == ["error", "response.failed"]
    else:
        # Pre-dispatch overflow: no generation, no metering, no stored state.
        assert [kind for kind in types if kind != "response.in_progress"] == [
            "response.created",
            "error",
            "response.failed",
        ]
        assert types[1] == "response.in_progress"
        assert await asyncio.to_thread(app.state.usage_ledger.totals) == {}
        assert app.state.response_store.get(events[-1]["response"]["id"]) is None
    error_event = events[-2]
    assert {key: error_event[key] for key in ("code", "message", "param")} == {
        "code": "context_length_exceeded",
        "message": _RESPONSES_MESSAGE,
        "param": "input",
    }
    failed = events[-1]["response"]
    assert failed["status"] == "failed"
    assert failed["error"] == {
        "code": "context_length_exceeded",
        "message": _RESPONSES_MESSAGE,
    }


def _assert_chat_overflow(response, *, stream) -> None:
    if stream:
        assert response.status_code == 200
        errors = _chat_error_payloads(response.text)
        assert len(errors) == 1
        error = errors[0]
    else:
        assert response.status_code == 400
        error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "context_length_exceeded"
    assert error["param"] == "messages"
    assert error["message"].startswith("This model's maximum context length")


def _assert_messages_overflow(response, *, stream, source) -> None:
    if stream:
        assert response.status_code == 200
        errors = [
            payload
            for name, payload in _anthropic_events(response.text)
            if name == "error"
        ]
        assert len(errors) == 1
        error = errors[0]["error"]
    else:
        assert response.status_code == 400
        assert response.json()["type"] == "error"
        error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    if source == "native":
        # Kairyu counted the prompt itself, so the client sees Anthropic's
        # exact shape with the numbers it uses to trim.
        assert re.fullmatch(
            rf"prompt is too long: \d+ tokens > {_MAX_MODEL_LEN} maximum",
            error["message"],
        )
    else:
        assert error["message"] == "prompt is too long"


@pytest.mark.parametrize(
    ("surface", "source", "stream", "tools"),
    [
        ("responses", "native", False, False),
        ("responses", "native", True, True),
        ("responses", "upstream", False, False),
        ("responses", "upstream", True, False),
        ("responses", "upstream", True, True),
        ("responses", "auto", False, False),
        ("responses", "auto", True, False),
        ("responses", "auto", True, True),
        ("chat", "native", False, False),
        ("chat", "upstream", False, False),
        ("chat", "upstream", True, False),
        ("messages", "native", False, False),
        ("messages", "upstream", False, False),
        ("messages", "upstream", True, False),
        ("messages", "auto", False, False),
    ],
    ids=[
        "responses-unary-native",
        "responses-stream-native",
        "responses-unary-upstream",
        "responses-stream-upstream-live",
        "responses-stream-upstream-buffered-tools",
        "responses-unary-auto",
        "responses-stream-auto-relay",
        "responses-stream-auto-buffered-tools",
        "chat-unary-native",
        "chat-unary-upstream",
        "chat-stream-upstream",
        "messages-unary-native",
        "messages-unary-upstream",
        "messages-stream-upstream",
        "messages-unary-auto",
    ],
)
async def test_context_overflow_classification(tmp_path, surface, source, stream, tools):
    app, model, backends = _overflow_app(tmp_path, source)
    try:
        if surface == "responses":
            payload = {"model": model, "input": _LONG_TEXT, "stream": stream}
            if tools:
                payload["tools"] = [_TOOL]
            response = await _post(app, "/v1/responses", payload)
            await _assert_responses_overflow(app, response, stream=stream, source=source)
        elif surface == "chat":
            response = await _post(
                app,
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": _LONG_TEXT}],
                    "stream": stream,
                },
            )
            _assert_chat_overflow(response, stream=stream)
        else:
            response = await _post(
                app,
                "/v1/messages",
                {
                    "model": model,
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": _LONG_TEXT}],
                    "stream": stream,
                },
            )
            _assert_messages_overflow(response, stream=stream, source=source)
        assert _UPSTREAM_SECRET not in response.text
        assert "vllm.internal" not in response.text
    finally:
        for backend in backends:
            await backend.shutdown()


async def test_internal_stage_overflow_is_a_server_error(tmp_path, caplog):
    # A synthesis prompt is built from candidates, not from the client's
    # prompt: reporting its overflow as context_length_exceeded would make
    # Codex compact a conversation that already fits, then fail again.
    class SynthesisOverflow(MockBackend):
        async def generate(self, request):
            raise ContextLengthExceededError(
                "prompt tokens (90) plus max_tokens (16) exceed max_model_len (64)",
                prompt_tokens=90,
                max_tokens=16,
                max_model_len=64,
            )

    app = create_legacy_app(
        {},
        orchestrators={
            "kairyu-auto": Orchestrator(
                {"tier1": MockBackend(), "tier2": SynthesisOverflow()},
                # Every prompt routes to MoA: tier1 proposes, tier2 synthesizes.
                router=RuleRouter(RouteThresholds(multi_agent_min_chars=1)),
                moa_samples=2,
            )
        },
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )

    response = await _post(
        app,
        "/v1/chat/completions",
        {"model": "kairyu-auto", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert error["code"] == "server_error"
    assert "context" not in error["message"].lower()
    # Operators still see why the stage failed.
    assert "internal_stage_context_overflow" in caplog.text
