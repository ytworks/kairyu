"""The L1/L3 Responses equivalence gate can pass and can fail (m11 D4 amendment)."""

import json

from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.responses_events import ResponseEmitter
from kairyu.entrypoints.server.responses_protocol import ResponsesRequest
from verification.product.correctness.upstream_api_equivalence import (
    Capture,
    compare,
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


def _reasoning_capture(*, vllm_part_events: bool) -> Capture:
    """Kairyu's own reasoning + text stream; vLLM names the reasoning part events."""

    request = ResponsesRequest.model_validate(
        {"model": "m", "input": "hello", "stream": True, "store": False}
    )
    emitter = ResponseEmitter(request, response_id="resp_1", created_at=0)
    frames = emitter.start() + emitter.begin_reasoning() + emitter.reasoning_delta("think")
    frames += emitter.close() + emitter.begin_message() + emitter.text_delta("ok")
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
    l1, l1_findings = summarize(_reasoning_capture(vllm_part_events=True), strict=False)
    l3, l3_findings = summarize(_reasoning_capture(vllm_part_events=False), strict=True)

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
