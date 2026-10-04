"""Responses rendering of a ``ClassifiedError`` (M20 WP-04).

Placement-aware: a ``pre_stream`` error, and any error of a unary request, is
an HTTP error body. An ``in_band_on_stream`` error of a streaming request is a
complete HTTP 200 SSE stream (``response.created`` -> ``response.in_progress``
-> ``error`` -> ``response.failed``) produced before anything is dispatched,
metered, or stored, because Codex acts on the code of ``response.failed`` and
treats a pre-stream 400 as terminal. WP-06 moves this module into the
``responses`` package.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse, Response

from kairyu.entrypoints.server.error_classifier import (
    ClassifiedError,
    classify_error_body,
    classify_request_error,
)
from kairyu.entrypoints.server.responses_envelope import (
    response_envelope,
    usage_payload_from_wire,
)
from kairyu.entrypoints.server.sse_response import sse_response
from kairyu.entrypoints.server.stream_util import sse_event

if TYPE_CHECKING:
    from kairyu.entrypoints.server.responses_service import ResponsesRequest


def responses_sse(event_type: str, sequence_number: int, **payload) -> str:
    """One Responses stream event with its gapless sequence number."""

    return sse_event(
        event_type,
        {"type": event_type, "sequence_number": sequence_number, **payload},
    )


def request_failure(request: ResponsesRequest, error: BaseException) -> Response | None:
    """The whole HTTP response for a classified pre-dispatch failure, else None."""

    classified = classify_request_error(error, "responses")
    return None if classified is None else _render(request, classified)


def delegated_failure(request: ResponsesRequest, delegated: JSONResponse) -> Response:
    """Re-render a delegated Chat error (AUTO) the Responses way when classified.

    The Chat handler rejects an overflowing AUTO prompt before any dispatch;
    other delegated errors are returned unchanged.
    """

    try:
        body = json.loads(bytes(delegated.body))
    except ValueError:
        return delegated
    classified = classify_error_body(body, "responses")
    return delegated if classified is None else _render(request, classified)


def _render(request: ResponsesRequest, classified: ClassifiedError) -> Response:
    if not (request.stream and classified.placement == "in_band_on_stream"):
        return JSONResponse(
            status_code=classified.status,
            content={"error": classified.openai_payload()},
        )
    return sse_response(_in_band_failure(request, classified))


async def _in_band_failure(
    request: ResponsesRequest,
    classified: ClassifiedError,
) -> AsyncIterator[str]:
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    in_progress = response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status="in_progress",
        output=[],
        usage=None,
    )
    yield responses_sse("response.created", 0, response=in_progress)
    yield responses_sse("response.in_progress", 1, response=in_progress)
    yield responses_sse(
        "error",
        2,
        code=classified.code,
        message=classified.message,
        param=classified.param,
    )
    failed = response_envelope(
        request,
        response_id=response_id,
        created_at=created_at,
        status="failed",
        output=[],
        # Nothing ran, so every count is zero.
        usage=usage_payload_from_wire(None),
        error={"code": classified.code, "message": classified.message},
    )
    yield responses_sse("response.failed", 3, response=failed)
