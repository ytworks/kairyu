"""The L1/L3 Responses equivalence gate can pass and can fail (m11 D4 amendment)."""

import json

from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
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
