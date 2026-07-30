#!/usr/bin/env python3
"""Verify the deployed Compose gateway-to-replica OTel trace.

The console exporter emits one marker-prefixed JSON object per finished span.
This verifier joins spans by IDs rather than log order or timing so the smoke is
deterministic across concurrent containers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_MARKER = "KAIRYU_OTEL_SPAN "
_COMPOSE_REPLICA_SERVICES = {
    "0": "kairyu-replica-1",
    "1": "kairyu-replica-2",
    "2": "kairyu-replica-3",
}


def _fail(message: str) -> None:
    raise SystemExit(f"OTel trace verification failed: {message}")


def _load_spans(path: Path) -> list[dict]:
    spans: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        marker = line.find(_MARKER)
        if marker < 0:
            continue
        try:
            span = json.loads(line[marker + len(_MARKER) :])
        except json.JSONDecodeError as error:
            _fail(f"{path}:{line_number}: invalid marker JSON: {error}")
        if not isinstance(span, dict):
            _fail(f"{path}:{line_number}: span payload must be an object")
        spans.append(span)
    if not spans:
        _fail(f"{path} contains no {_MARKER.strip()!r} records")
    return spans


def _attributes(span: dict) -> dict:
    attributes = span.get("attributes")
    if not isinstance(attributes, dict):
        _fail(f"span {span.get('name')!r} has no attribute object")
    return attributes


def _single(spans: list[dict], description: str) -> dict:
    if len(spans) != 1:
        _fail(f"expected exactly one {description}, found {len(spans)}")
    return spans[0]


def _string_values(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _string_values(key)
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)


def verify(
    spans: list[dict],
    *,
    request_id: str,
    forbidden: tuple[str, ...],
) -> None:
    gateway = _single(
        [
            span
            for span in spans
            if span.get("name") == "kairyu.request"
            and span.get("service_name") == "kairyu-gateway"
            and _attributes(span).get("kairyu.request_id") == request_id
        ],
        "gateway request root for the response X-Request-ID",
    )
    trace_id = gateway.get("trace_id")
    if not isinstance(trace_id, str) or len(trace_id) != 32:
        _fail("gateway root has an invalid trace_id")
    trace = [span for span in spans if span.get("trace_id") == trace_id]
    trace_strings = tuple(_string_values(trace))
    for value in forbidden:
        if value and any(value in candidate for candidate in trace_strings):
            _fail("prompt/output canary leaked into exported span data")

    by_id: dict[str, dict] = {}
    for span in trace:
        span_id = span.get("span_id")
        if not isinstance(span_id, str) or len(span_id) != 16:
            _fail(f"span {span.get('name')!r} has an invalid span_id")
        if span_id in by_id:
            _fail(f"duplicate span_id {span_id}")
        by_id[span_id] = span

    gateway_id = gateway["span_id"]
    gateway_attributes = _attributes(gateway)
    if gateway.get("kind") != "SERVER":
        _fail("gateway request span is not SERVER")
    if gateway.get("parent_span_id") is not None:
        _fail("gateway request unexpectedly has a parent")
    if gateway_attributes.get("http.response.status_code") != 200:
        _fail("gateway request did not record HTTP 200")
    if gateway_attributes.get("kairyu.response.complete") is not True:
        _fail("gateway response did not reach its final ASGI body")
    if gateway_attributes.get("http.route") != "/v1/chat/completions":
        _fail("gateway root recorded the wrong HTTP route")

    placements = [
        span
        for span in trace
        if span.get("name") == "kairyu.pool.place"
        and span.get("service_name") == "kairyu-gateway"
    ]
    placement = _single(placements, "gateway pool placement span")
    if placement.get("parent_span_id") != gateway_id:
        _fail("pool placement is not a child of the gateway request")
    placement_attributes = _attributes(placement)
    for key in (
        "kairyu.request_id",
        "kairyu.replica.id",
        "kairyu.placement.reason",
        "kairyu.pool.size",
        "kairyu.pool.eligible_size",
        "kairyu.placement.latency_ns",
    ):
        if key not in placement_attributes:
            _fail(f"pool placement is missing {key!r}")
    if placement_attributes["kairyu.request_id"] != request_id:
        _fail("pool placement request ID does not match the gateway response")

    calls = [
        span
        for span in trace
        if span.get("name") == "kairyu.replica.call"
        and span.get("service_name") == "kairyu-gateway"
    ]
    call = _single(calls, "gateway replica call span")
    if call.get("kind") != "CLIENT":
        _fail("replica call span is not CLIENT")
    if call.get("parent_span_id") != gateway_id:
        _fail("replica call is not a child of the gateway request")
    call_attributes = _attributes(call)
    if call_attributes.get("kairyu.request_id") != request_id:
        _fail("replica-call request ID does not match the gateway response")
    if call_attributes.get("kairyu.call.status") != "success":
        _fail("replica call did not record success")
    if call_attributes.get("kairyu.replica.id") != placement_attributes.get(
        "kairyu.replica.id"
    ):
        _fail("placement and replica-call IDs disagree")
    replica_id = call_attributes.get("kairyu.replica.id")
    expected_service = _COMPOSE_REPLICA_SERVICES.get(replica_id)
    if expected_service is None:
        _fail(f"selected replica ID {replica_id!r} is not in the Compose topology")

    replica_request = _single(
        [
            span
            for span in trace
            if span.get("name") == "kairyu.request"
            and span.get("service_name") == expected_service
            and span.get("parent_span_id") == call["span_id"]
        ],
        f"{expected_service} SERVER span parented by the CLIENT call",
    )
    if replica_request.get("kind") != "SERVER":
        _fail("remote replica request span is not SERVER")
    replica_attributes = _attributes(replica_request)
    if replica_attributes.get("http.route") != "/v1/chat/completions":
        _fail("remote replica recorded the wrong HTTP route")
    if replica_attributes.get("http.response.status_code") != 200:
        _fail("remote replica request did not record HTTP 200")
    if replica_attributes.get("kairyu.response.complete") is not True:
        _fail("remote replica response was not completed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--response-body", type=Path)
    parser.add_argument("--forbidden", action="append", default=[])
    return parser.parse_args()


def _response_output(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        output = payload["choices"][0]["message"]["content"]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        _fail(f"cannot read response output from {path}: {type(error).__name__}")
    if not isinstance(output, str) or not output:
        _fail(f"{path} contains no non-empty assistant output")
    return output


def main() -> None:
    args = _parse_args()
    forbidden = list(args.forbidden)
    if args.response_body is not None:
        forbidden.append(_response_output(args.response_body))
    verify(
        _load_spans(args.logs),
        request_id=args.request_id,
        forbidden=tuple(forbidden),
    )


if __name__ == "__main__":
    main()
