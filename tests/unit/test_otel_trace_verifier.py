from __future__ import annotations

import json

import pytest

from scripts.verify_otel_trace import _load_spans, _response_output, verify


def _span(
    name: str,
    service: str,
    span_id: str,
    parent: str | None,
    attributes: dict,
    *,
    kind: str = "INTERNAL",
) -> dict:
    return {
        "name": name,
        "service_name": service,
        "trace_id": "1" * 32,
        "span_id": span_id,
        "parent_span_id": parent,
        "kind": kind,
        "status": "UNSET",
        "attributes": attributes,
    }


def _valid_trace() -> list[dict]:
    root = "a" * 16
    call = "c" * 16
    request_id = "response-request-id"
    return [
        _span(
            "kairyu.request",
            "kairyu-gateway",
            root,
            None,
            {
                "kairyu.request_id": request_id,
                "http.route": "/v1/chat/completions",
                "http.response.status_code": 200,
                "kairyu.response.complete": True,
            },
            kind="SERVER",
        ),
        _span(
            "kairyu.pool.place",
            "kairyu-gateway",
            "b" * 16,
            root,
            {
                "kairyu.request_id": request_id,
                "kairyu.replica.id": "0",
                "kairyu.placement.reason": "least_load",
                "kairyu.pool.size": 3,
                "kairyu.pool.eligible_size": 3,
                "kairyu.placement.latency_ns": 12,
            },
        ),
        _span(
            "kairyu.replica.call",
            "kairyu-gateway",
            call,
            root,
            {
                "kairyu.request_id": request_id,
                "kairyu.replica.id": "0",
                "kairyu.call.status": "success",
            },
            kind="CLIENT",
        ),
        _span(
            "kairyu.request",
            "kairyu-replica-1",
            "d" * 16,
            call,
            {
                "http.route": "/v1/chat/completions",
                "http.response.status_code": 200,
                "kairyu.response.complete": True,
            },
            kind="SERVER",
        ),
    ]


def test_valid_deployed_trace_joins_by_ids_not_log_order():
    verify(
        list(reversed(_valid_trace())),
        request_id="response-request-id",
        forbidden=("prompt-canary", "output-canary"),
    )


@pytest.mark.parametrize(
    ("span_name", "field", "value"),
    [
        ("kairyu.pool.place", "parent_span_id", "f" * 16),
        ("kairyu.replica.call", "kind", "INTERNAL"),
        ("kairyu.request", "service_name", "wrong-gateway"),
    ],
)
def test_invalid_deployed_trace_fails_closed(span_name, field, value):
    spans = _valid_trace()
    target = next(
        span
        for span in spans
        if span["name"] == span_name
        and (
            span_name != "kairyu.request"
            or span["service_name"] == "kairyu-gateway"
        )
    )
    target[field] = value

    with pytest.raises(SystemExit, match="OTel trace verification failed"):
        verify(
            spans,
            request_id="response-request-id",
            forbidden=(),
        )


def test_trace_verifier_rejects_prompt_or_output_canary():
    spans = _valid_trace()
    spans[1]["attributes"]["unsafe"] = "output-canary\nwith-json-escaping"

    with pytest.raises(SystemExit, match="canary leaked"):
        verify(
            spans,
            request_id="response-request-id",
            forbidden=("prompt-canary", "output-canary\nwith-json-escaping"),
        )


@pytest.mark.parametrize(
    ("span_name", "attribute", "value"),
    [
        ("kairyu.pool.place", "kairyu.request_id", "wrong-request"),
        ("kairyu.replica.call", "kairyu.request_id", "wrong-request"),
        ("kairyu.replica.call", "kairyu.replica.id", "1"),
    ],
)
def test_trace_verifier_rejects_misattributed_replica_work(
    span_name,
    attribute,
    value,
):
    spans = _valid_trace()
    target = next(span for span in spans if span["name"] == span_name)
    target["attributes"][attribute] = value

    with pytest.raises(SystemExit, match="OTel trace verification failed"):
        verify(
            spans,
            request_id="response-request-id",
            forbidden=(),
        )


def test_trace_verifier_rejects_remote_service_that_is_not_selected():
    spans = _valid_trace()
    remote = next(
        span
        for span in spans
        if span["name"] == "kairyu.request"
        and span["service_name"].startswith("kairyu-replica-")
    )
    remote["service_name"] = "kairyu-replica-2"

    with pytest.raises(SystemExit, match="OTel trace verification failed"):
        verify(
            spans,
            request_id="response-request-id",
            forbidden=(),
        )


def test_marker_loader_and_response_output(tmp_path):
    logs = tmp_path / "compose.log"
    span = _valid_trace()[0]
    logs.write_text(
        "ordinary log\n"
        f"gateway | KAIRYU_OTEL_SPAN {json.dumps(span)}\n",
        encoding="utf-8",
    )
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps({"choices": [{"message": {"content": "private output"}}]}),
        encoding="utf-8",
    )

    assert _load_spans(logs) == [span]
    assert _response_output(response) == "private output"
