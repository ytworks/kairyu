"""The L1/L3 Responses equivalence gate can pass and can fail (m11 D4 amendment)."""

import argparse
import json

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.responses_events import ResponseEmitter
from kairyu.entrypoints.server.responses_protocol import ResponsesRequest
from verification.product.correctness import upstream_api_equivalence
from verification.product.correctness.upstream_api_equivalence import (
    Capture,
    Target,
    compare,
    run_case,
    summarize,
)


def _kairyu_capture() -> Capture:
    app = create_app(engines={"m": MockBackend()}, legacy_chat_models={"m"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "m", "input": "hello", "stream": True, "store": False},
        )
    events = [
        json.loads(line[len("data:") :])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    return Capture(response.status_code, events, events[-1]["response"], response.text)


def _stream(*items: tuple[str, str], vllm_part_events: bool = False) -> Capture:
    """Kairyu's own stream of (reasoning | message | function_call, text or name) items.

    With ``vllm_part_events`` the reasoning part events carry vLLM's names.
    """

    request = ResponsesRequest.model_validate(
        {"model": "m", "input": "hello", "stream": True, "store": False}
    )
    emitter = ResponseEmitter(request, response_id="resp_1", created_at=0)
    frames = emitter.start()
    for kind, value in items:
        if kind == "reasoning":
            frames += emitter.begin_reasoning() + emitter.reasoning_delta(value)
        elif kind == "message":
            frames += emitter.begin_message() + emitter.text_delta(value)
        else:
            frames += emitter.begin_call("call_1", value, None)
            frames += emitter.arguments_delta('{"city": "Paris"}')
        frames += emitter.close()
    usage = {
        "input_tokens": 1,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 3,
    }
    envelope, tail = emitter.complete("completed", usage)
    events = []
    for frame in frames + tail:
        text = frame.decode() if isinstance(frame, bytes) else frame
        event = json.loads(text.split("data: ", 1)[1])
        part = event.get("part") or {}
        if vllm_part_events and part.get("type") == "reasoning_text":
            event["type"] = event["type"].replace("content_part", "reasoning_part")
        events.append(event)
    return Capture(200, events, envelope, "")


def test_a_vllm_reasoning_stream_matches_kairyus():
    items = (("reasoning", "think"), ("message", "ok"))
    l1, l1_findings = summarize(_stream(*items, vllm_part_events=True), strict=False)
    l3, l3_findings = summarize(_stream(*items), strict=True)

    assert l1_findings == [] and l3_findings == []
    assert l1["items"] == ["reasoning", "message"]
    assert compare(l1, l3) == []


def test_a_spec_conformant_stream_matches_itself():
    summary, findings = summarize(_kairyu_capture(), strict=True)

    assert findings == []
    assert compare(summary, summary, exact_input_tokens=True) == []


def test_a_stream_without_its_terminal_event_fails():
    capture = _kairyu_capture()
    broken = Capture(capture.status, capture.events[:-1], capture.response, capture.raw)
    summary, findings = summarize(broken, strict=True)
    good, _ = summarize(capture, strict=True)

    assert any("terminal event" in finding for finding in findings)
    assert compare(good, summary)


def test_an_empty_message_item_carries_no_answer():
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "content": []}
    empty = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "incomplete",
        "content": [{"type": "output_text", "text": "", "annotations": []}],
    }
    base = {
        "id": "resp_1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    l1, _ = summarize(Capture(200, [], {**base, "output": [reasoning]}), strict=False)
    l3, _ = summarize(Capture(200, [], {**base, "output": [reasoning, empty]}), strict=False)

    assert l1["items"] == l3["items"] == ["reasoning"]
    assert compare(l1, l3) == []


_CALL = ("function_call", "get_weather")
_STRAY = ("function_call", "unregistered_weather")
_ANSWER = ("message", "18C and sunny.")
_THOUGHT = ("reasoning", "think")
_STRAY_FINDINGS = [
    "l3: function_call ['unregistered_weather'] is not a declared tool",
    "functions: L1 ['get_weather'] != L3 ['unregistered_weather']",
]


@pytest.mark.parametrize(
    ("l1_turns", "l3_turns", "findings"),
    [
        pytest.param([[_CALL], [_ANSWER]], [[_STRAY], [_ANSWER]], _STRAY_FINDINGS, id="call-turn"),
        pytest.param(
            [[_CALL], [_ANSWER, _CALL]],
            [[_CALL], [_ANSWER, _STRAY]],
            _STRAY_FINDINGS,
            id="answer-turn",
        ),
        # Whether the model reasons before a call or an answer is sampled per request.
        pytest.param(
            [[_THOUGHT, _CALL], [_ANSWER]], [[_CALL], [_THOUGHT, _ANSWER]], [], id="reasoning"
        ),
        # So is how many parallel calls it makes.
        pytest.param([[_CALL, _CALL], [_ANSWER]], [[_CALL], [_ANSWER]], [], id="parallel-calls"),
    ],
)
def test_the_tools_case_compares_both_turns(monkeypatch, l1_turns, l3_turns, findings):
    turns = {"l1": l1_turns, "l3": l3_turns}
    monkeypatch.setattr(
        upstream_api_equivalence,
        "capture",
        lambda target, body: _stream(*turns[target.name][isinstance(body["input"], list)]),
    )
    targets = (Target("l1", "", "m"), Target("l3", "", "m"))
    args = argparse.Namespace(max_output_tokens=64, l3_orchestrated=False)

    assert run_case("tools", targets, args)["findings"] == findings


@pytest.mark.parametrize(
    ("orchestrated", "findings"), [(False, ["l3: answer 'White' is not red"]), (True, [])]
)
def test_an_auto_models_image_answer_is_not_judged(monkeypatch, orchestrated, findings):
    answers = {"l1": "Red", "l3": "White"}
    monkeypatch.setattr(
        upstream_api_equivalence,
        "capture",
        lambda target, body: _stream(("message", answers[target.name])),
    )
    targets = (Target("l1", "", "m"), Target("l3", "", "m"))
    args = argparse.Namespace(max_output_tokens=64, l3_orchestrated=orchestrated)

    assert run_case("image", targets, args)["findings"] == findings


@pytest.mark.parametrize(
    ("failed", "finding"),
    [
        # A Quyet AUTO model answered the cap case with HTTP 502 (GPU 2026-10-18);
        # the case's own check then read fields a failed capture does not have.
        (Capture(502, [], None, '{"error": {"code": "backend_error"}}'), "l3: HTTP 502"),
        # A 200 stream cut before its terminal event carries no response either.
        (Capture(200, [], None, "data: {}"), "l3: HTTP 200"),
    ],
    ids=["http-502", "no-terminal-event"],
)
def test_a_capture_without_a_response_is_a_finding_not_a_crash(monkeypatch, failed, finding):
    monkeypatch.setattr(
        upstream_api_equivalence,
        "capture",
        lambda target, body: _stream(("message", "1")) if target.name == "l1" else failed,
    )
    targets = (Target("l1", "", "m"), Target("l3", "", "m"))
    args = argparse.Namespace(max_output_tokens=64, l3_orchestrated=True)

    findings = run_case("cap", targets, args)["findings"]
    assert any(item.startswith(finding) for item in findings)
