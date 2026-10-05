"""Responses streaming: reasoning, incremental calls, heartbeats, overflow."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult, GenerationUsage, UpstreamClientError
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server import responses_auto, responses_events
from kairyu.entrypoints.server.responses_events import OutputAssembler, ResponseEmitter
from kairyu.entrypoints.server.responses_protocol import ResponsesRequest
from kairyu.entrypoints.server.responses_store import PendingSave
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.outputs import CompletionOutput
from tests.server._legacy_chat import create_legacy_app

_ADD = {
    "type": "function",
    "name": "add",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    },
}
_CALL = '<tool_call>{"name":"add","arguments":{"a":1,"b":1}}</tool_call>'
# Renders replayed reasoning and leaves the generation prompt inside <think>,
# like reasoning-model templates that open the block themselves.
_THINKING_TEMPLATE = ChatTemplate(
    "{% for m in messages %}<|{{ m.role }}|>"
    "{% if m.reasoning_content %}[reasoning:{{ m.reasoning_content }}]{% endif %}"
    "{{ m.content or '' }}{% for c in m.tool_calls or [] %}[call:{{ c.function.name }}]"
    "{% endfor %}\n{% endfor %}<|assistant|><think>\n"
)


class ScriptedBackend(MockBackend):
    """Streams cumulative text step by step: a float step is a pause and a
    ``{"reasoning": ...}`` step sets backend-separated reasoning so far."""

    def __init__(self, *script, reasoning: str | None = None, finish: str = "stop"):
        super().__init__()
        self.script = script
        self.reasoning = reasoning
        self.finish = finish
        self.prompts: list[str] = []
        self.max_model_len = 4096

    def _result(self, text: str, *, final: bool, reasoning: str | None = None) -> GenerationResult:
        return GenerationResult(
            request_id="r",
            prompt="",
            completions=(
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=(1,) * len(text),
                    finish_reason=self.finish if final else None,
                    reasoning_content=reasoning or self.reasoning,
                ),
            ),
            finished=final,
            usage=GenerationUsage(prompt_tokens=3, completion_tokens=len(text)) if final else None,
        )

    async def generate(self, request):
        self.prompts.append(str(request.prompt))
        steps = [step for step in self.script if isinstance(step, dict)]
        return self._result(
            "".join(s for s in self.script if isinstance(s, str)),
            final=True,
            reasoning=steps[-1]["reasoning"] if steps else None,
        )

    async def stream(self, request):
        self.prompts.append(str(request.prompt))
        text, reasoning = "", None
        for step in self.script:
            if isinstance(step, float):
                await asyncio.sleep(step)
                continue
            if isinstance(step, dict):
                reasoning = step["reasoning"]
            else:
                text += step
            yield self._result(text, final=False, reasoning=reasoning)
        yield self._result(text, final=True, reasoning=reasoning)


class _NodeAccumulator:
    """openai-node ResponseStream: lifecycle events replace the snapshot and
    deltas must append at exactly the indices already announced."""

    def __init__(self) -> None:
        self.snapshot: dict | None = None

    def feed(self, event: dict) -> None:
        kind = event["type"]
        if "response" in event:
            assert (self.snapshot is None) == (kind == "response.created")
            self.snapshot = copy.deepcopy(event["response"])
            return
        if kind == "error":
            return
        output = self.snapshot["output"]
        if kind == "response.output_item.added":
            assert event["output_index"] == len(output)
            output.append(copy.deepcopy(event["item"]))
            return
        item = output[event["output_index"]]
        assert item["id"] == event.get("item_id", item["id"])
        if kind == "response.content_part.added":
            assert event["content_index"] == len(item["content"])
            item["content"].append(copy.deepcopy(event["part"]))
        elif kind in ("response.output_text.delta", "response.reasoning_text.delta"):
            item["content"][event["content_index"]]["text"] += event["delta"]
        elif kind == "response.function_call_arguments.delta":
            item["arguments"] += event["delta"]
        elif kind == "response.output_item.done":
            output[event["output_index"]] = copy.deepcopy(event["item"])


def _app(tmp_path, backend, **kwargs):
    return create_legacy_app(
        {"m": backend},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        **kwargs,
    )


def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def _shape(output: list[dict]) -> list[dict]:
    """Items without per-response ids, for stream/unary comparisons.

    Arguments compare as JSON: the shared scanner streams a call's raw bytes
    when it spans deltas but re-serializes one that arrives whole.
    """

    shaped = []
    for item in output:
        item = {k: v for k, v in item.items() if k not in ("id", "call_id")}
        if item["type"] == "function_call":
            item["arguments"] = json.loads(item["arguments"])
        shaped.append(item)
    return shaped


def _run(http, body: dict) -> tuple[list[dict], dict]:
    stream = http.post("/v1/responses", json={**body, "stream": True})
    unary = http.post("/v1/responses", json=body)
    events = _events(stream.text)
    node = _NodeAccumulator()
    for event in events:
        node.feed(event)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert node.snapshot == events[-1]["response"]
    return events, unary.json()


@pytest.mark.parametrize(
    "backend, templates",
    [
        (ScriptedBackend("<think>plan ", _CALL, "</think>", "Answer"), None),
        (ScriptedBackend("plan ", _CALL, "</think>", "Answer"), {"m": _THINKING_TEMPLATE}),
        (ScriptedBackend("Answer", reasoning="plan " + _CALL), None),
    ],
    ids=["leading-think", "template-opened-think", "backend-separated"],
)
def test_reasoning_is_split_without_effort_and_never_parsed_for_calls(
    tmp_path, backend, templates
):
    with TestClient(_app(tmp_path, backend, chat_templates=templates)) as http:
        events, unary = _run(http, {"model": "m", "input": "hi", "tools": [_ADD]})

    streamed = events[-1]["response"]["output"]
    assert _shape(streamed) == _shape(unary["output"])
    assert [item["type"] for item in streamed] == ["reasoning", "message"]
    assert streamed[0]["content"] == [{"type": "reasoning_text", "text": "plan " + _CALL}]
    assert streamed[1]["content"][0]["text"] == "Answer"


def test_calls_stream_incrementally_beside_their_preamble(tmp_path):
    backend = ScriptedBackend(
        "Let me add. ",
        '<tool_call>{"name":"add","arguments":{"a":',
        0.05,
        '2,"b":3}}</tool_call>',
        _CALL,
    )
    with TestClient(_app(tmp_path, backend)) as http:
        events, unary = _run(http, {"model": "m", "input": "add", "tools": [_ADD]})

    output = events[-1]["response"]["output"]
    deltas = [e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta"]
    assert [item["type"] for item in output] == ["message", "function_call", "function_call"]
    assert output[0]["content"][0]["text"] == "Let me add. "
    assert [json.loads(item["arguments"]) for item in output[1:]] == [
        {"a": 2, "b": 3},
        {"a": 1, "b": 1},
    ]
    assert deltas[0] == '{"a":'  # committed before the model finished the call
    assert len({item["id"] for item in output}) == len({item["call_id"] for item in output[1:]}) + 1
    assert _shape(output) == _shape(unary["output"])


@pytest.mark.parametrize(
    "body, code",
    [
        ({"parallel_tool_calls": False}, "parallel_tool_calls_not_satisfied"),
        ({"tool_choice": "required"}, "tool_choice_not_satisfied"),
    ],
)
def test_tool_gates_fail_in_band_and_as_unary_errors(tmp_path, body, code):
    script = (_CALL, _CALL) if "parallel_tool_calls" in body else ("no call",)
    request = {"model": "m", "input": "add", "tools": [_ADD], **body}
    with TestClient(_app(tmp_path, ScriptedBackend(*script))) as http:
        stream = http.post("/v1/responses", json={**request, "stream": True})
        unary = http.post("/v1/responses", json=request)
    events = _events(stream.text)
    assert [e["type"] for e in events[-2:]] == ["error", "response.failed"]
    assert events[-2]["code"] == code  # response.error.code stays in the spec enum
    assert unary.status_code == 502
    assert unary.json()["error"]["code"] == code


def test_reasoning_interleaved_into_a_call_waits_for_the_call_to_close(tmp_path):
    backend = ScriptedBackend(
        '<tool_call>{"name":"add","arguments":{"a":', {"reasoning": "late"}, '1,"b":1}}</tool_call>'
    )
    with TestClient(_app(tmp_path, backend)) as http:
        events, _unary = _run(http, {"model": "m", "input": "add", "tools": [_ADD]})

    output = events[-1]["response"]["output"]
    assert [item["type"] for item in output] == ["function_call", "reasoning"]
    assert json.loads(output[0]["arguments"]) == {"a": 1, "b": 1}
    assert output[1]["content"][0]["text"] == "late"


def test_a_failed_gate_settles_tenant_usage_like_the_unary_turn(tmp_path):
    # The engine finished, so the stream settles exact usage instead of
    # keeping the whole max_model_len reservation.
    debits = []
    limits = {"default": TenantLimits(tokens_per_minute=1, token_burst=100_000)}
    for stream in (False, True):
        app = _app(tmp_path, ScriptedBackend("no call"), tenant_config=TenantConfig(limits=limits))
        with TestClient(app) as http:
            limiter = http.app.state.tenant_limiter
            before = limiter.token_balance("default")
            body = {"model": "m", "input": "add", "tools": [_ADD], "tool_choice": "required"}
            http.post("/v1/responses", json={**body, "stream": stream})
            debits.append(before - limiter.token_balance("default"))
    assert debits[1] == pytest.approx(debits[0], abs=0.5)  # not the ~4k reservation


def test_unary_keeps_the_chat_downgrade_of_retroactively_invalid_calls(tmp_path):
    # Prose after a QWEN call voids it: chat and Messages unary return text.
    qwen = ChatTemplate(
        "use <function=NAME><parameter=K>V</parameter></function>"
        "{% for m in messages %}<|{{ m.role }}|>{{ m.content or '' }}\n{% endfor %}"
    )
    text = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n</function>\n"
    text += "</tool_call>\nI will wait."
    with TestClient(_app(tmp_path, ScriptedBackend(text), chat_templates={"m": qwen})) as http:
        unary = http.post("/v1/responses", json={"model": "m", "input": "add", "tools": [_ADD]})
    assert unary.status_code == 200
    assert [item["content"][0]["text"] for item in unary.json()["output"]] == [text]


def test_heartbeats_are_data_events_carrying_the_streamed_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(responses_events, "_HEARTBEAT_SECONDS", 0.03)
    backend = ScriptedBackend("<think>plan", 0.15, " more</think>Hello", 0.15, " world")
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True})

    assert not [line for line in response.text.splitlines() if line.startswith(":")]
    node = _NodeAccumulator()
    heartbeats = []
    for event in _events(response.text):
        if event["type"] == "response.in_progress" and node.snapshot is not None:
            # openai-node swaps in this snapshot: it must equal what it built.
            assert event["response"]["output"] == node.snapshot["output"]
            heartbeats.append([item["type"] for item in event["response"]["output"]])
        node.feed(event)
    assert ["reasoning", "message"] in heartbeats
    assert node.snapshot["status"] == "completed"


async def test_auto_relay_sends_data_heartbeats_while_only_status_comments_arrive(
    monkeypatch,
):
    # The orchestrator's ": status" comments keep the upstream busy but are
    # not data for Codex's idle timer, so the relay must still heartbeat.
    monkeypatch.setattr(responses_events, "_HEARTBEAT_SECONDS", 0.03)

    async def upstream():
        for _ in range(10):
            await asyncio.sleep(0.01)
            yield ": status working\n\n"
        yield 'data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}'
        yield "\n\ndata: [DONE]\n\n"

    emitter = ResponseEmitter(ResponsesRequest(model="auto"), response_id="resp_x", created_at=0)
    frames = [
        frame if isinstance(frame, str) else frame.decode()
        async for frame in responses_auto._relay(
            upstream(),
            emitter=emitter,
            assembler=OutputAssembler(emitter),
            saver=PendingSave(None, "default", [], []),
        )
    ]
    text = "".join(frames)
    assert text.count(": status working") == 10
    assert text.count("event: response.in_progress") > 2
    assert "event: response.completed" in text


def test_reasoning_round_trips_through_encrypted_content(tmp_path):
    backend = ScriptedBackend("secret plan</think>", "Answer")
    with TestClient(_app(tmp_path, backend, chat_templates={"m": _THINKING_TEMPLATE})) as http:
        first = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hi",
                "store": False,
                "include": ["reasoning.encrypted_content"],
            },
        ).json()
        reasoning, answer = first["output"]
        sealed_only = {key: reasoning[key] for key in ("type", "id", "summary")}
        sealed_only["encrypted_content"] = reasoning["encrypted_content"]
        tampered = {**reasoning, "encrypted_content": "krs1.tampered"}
        foreign = {"type": "reasoning", "summary": [], "encrypted_content": "gAAAA-foreign"}
        for history in ([sealed_only], [tampered], [foreign]):
            follow_up = http.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "input": [
                        {"role": "user", "content": "hi"},
                        *history,
                        answer,
                        {"role": "user", "content": "next"},
                    ],
                },
            )
            assert follow_up.status_code == 200

    assert reasoning["encrypted_content"].startswith("krs1.")
    replayed = "<|assistant|>[reasoning:secret plan]Answer"
    assert replayed in backend.prompts[1]
    assert replayed in backend.prompts[2]  # a bad token falls back to the visible text
    assert "[reasoning:" not in backend.prompts[3]  # nothing recoverable is dropped


def test_auto_models_do_not_replay_reasoning(tmp_path):
    backend = MockBackend({"next": "done"})
    app = create_legacy_app(
        {},
        orchestrators={"kairyu-auto": Orchestrator({"tier1": backend, "tier2": backend})},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )
    reasoning = {
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "private stage plan"}],
    }
    with TestClient(app) as http:
        response = http.post(
            "/v1/responses",
            json={
                "model": "kairyu-auto",
                "input": [
                    {"role": "user", "content": "hi"},
                    reasoning,
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "next"},
                ],
            },
        )
    assert response.status_code == 200
    assert not any("private stage plan" in str(prompt) for prompt in backend.prompts_seen)


class _PrepareFailure(MockBackend):
    def __init__(self, error: Exception):
        super().__init__()
        self.error = error

    async def prepare_request(self, request):
        raise self.error


class _StreamFailure(ScriptedBackend):
    async def stream(self, request):
        raise ValueError("prompt tokens (9) already fill max_model_len (8)")
        yield  # pragma: no cover


@pytest.mark.parametrize(
    "backend, body",
    [
        (_PrepareFailure(ValueError("prompt tokens (9) already fill max_model_len (8)")), {}),
        (
            _PrepareFailure(
                UpstreamClientError(
                    "backend http://replica returned HTTP 400: This model's maximum "
                    "context length is 8 tokens.",
                    400,
                )
            ),
            {},
        ),
        (_StreamFailure("unused"), {}),
        (ScriptedBackend("cut", finish="length"), {}),
    ],
    ids=["prepare-native", "prepare-vllm", "during-stream", "context-exhausted"],
)
def test_context_overflow_is_in_band_for_streams_and_400_for_unary(tmp_path, backend, body):
    request = {"model": "m", "input": "long", **body}
    with TestClient(_app(tmp_path, backend)) as http:
        stream = http.post("/v1/responses", json={**request, "stream": True})
        unary = http.post("/v1/responses", json=request)

    events = _events(stream.text)
    assert stream.status_code == 200
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["code"] == "context_length_exceeded"
    if isinstance(backend, _StreamFailure):
        return  # unary never streams
    assert unary.status_code == 400
    assert unary.json()["error"]["code"] == "context_length_exceeded"
    assert unary.json()["error"]["param"] == "input"


def test_an_explicit_cap_still_ends_incomplete(tmp_path):
    with TestClient(_app(tmp_path, ScriptedBackend("cut", finish="length"))) as http:
        unary = http.post(
            "/v1/responses", json={"model": "m", "input": "x", "max_output_tokens": 3}
        )
    assert unary.json()["status"] == "incomplete"
    assert unary.json()["incomplete_details"] == {"reason": "max_output_tokens"}


def test_auto_tool_streams_commit_calls_incrementally(tmp_path):
    text = 'Checking. <tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>'
    backend = MockBackend({"add": text})
    app = create_legacy_app(
        {},
        orchestrators={"kairyu-auto": Orchestrator({"tier1": backend, "tier2": backend})},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )
    with TestClient(app) as http:
        events, unary = _run(http, {"model": "kairyu-auto", "input": "add", "tools": [_ADD]})

    output = events[-1]["response"]["output"]
    deltas = [e for e in events if e["type"] == "response.function_call_arguments.delta"]
    assert [item["type"] for item in output] == ["message", "function_call"]
    assert output[1]["name"] == "add"
    assert json.loads(output[1]["arguments"]) == {"a": 2, "b": 3}
    assert len(deltas) > 1
    assert [item["type"] for item in unary["output"]] == ["message", "function_call"]
