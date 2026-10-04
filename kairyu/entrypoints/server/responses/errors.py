"""Responses rendering of request and generation errors (M20 WP-04).

Placement-aware: a ``pre_stream`` error, and any error of a unary request, is
an HTTP error body. An ``in_band_on_stream`` error of a streaming request is a
complete HTTP 200 SSE stream (``response.created`` -> ``response.in_progress``
-> ``error`` -> ``response.failed``) produced before anything is dispatched,
metered, or stored, because Codex acts on the code of ``response.failed`` and
treats a pre-stream 400 as terminal.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse, Response

from kairyu.entrypoints.server.chat_service import ChatRequestError
from kairyu.entrypoints.server.error_classifier import (
    ClassifiedError,
    classify_error_body,
    classify_request_error,
)
from kairyu.entrypoints.server.responses.envelope import (
    response_envelope,
    usage_payload_from_wire,
)
from kairyu.entrypoints.server.responses.events import responses_sse
from kairyu.entrypoints.server.sse_response import sse_response

if TYPE_CHECKING:
    from kairyu.entrypoints.server.responses.request import ResponsesRequest


class BufferedFailure(Exception):
    """Generation failure carrying the exact wire error payload.

    The buffered path surfaces it as an HTTP error before streaming starts
    (non-stream requests) or as ``error`` + ``response.failed`` SSE events
    once the stream is already open.
    """

    def __init__(self, payload: dict, status_code: int) -> None:
        classified = classify_error_body({"error": payload}, "responses")
        if classified is not None:
            payload, status_code = classified.openai_payload(), classified.status
        super().__init__(payload.get("message") or "generation failed")
        self.payload = payload
        self.status_code = status_code

    @classmethod
    def from_chat_error(cls, error: ChatRequestError) -> BufferedFailure:
        return cls(error.payload(), error.status_code)

    def json_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code, content={"error": self.payload}
        )


def request_error(message: str, *, status_code: int = 400, code: str | None = None):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
            }
        },
    )


def chat_error(error: ChatRequestError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": error.payload()})


def request_failure(request: ResponsesRequest, error: BaseException) -> Response | None:
    """The whole HTTP response for a classified pre-dispatch failure, else None."""

    classified = classify_request_error(error, "responses")
    return None if classified is None else _render(request, classified)


def delegated_failure(request: ResponsesRequest, delegated: JSONResponse) -> Response:
    """Re-render a delegated Chat error (AUTO) the Responses way when classified.

    The Chat handler reports an overflow of the client prompt (AUTO preflight,
    or a direct route's upstream on a unary run); other delegated errors are
    returned unchanged.
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
