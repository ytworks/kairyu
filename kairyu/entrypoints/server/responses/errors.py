"""Responses rendering of request and generation errors (M20 WP-04, WP-07).

Every failure is a ``ClassifiedError`` rendered in the Responses dialect: the
OpenAI envelope with ``param`` always present, and transient backpressure (a
retryable 429) as 503 ``slow_down`` with ``Retry-After`` and
``retry-after-ms`` (O-2). A Chat error delegated by an AUTO turn is
re-rendered the same way: extras stripped, code mapped, ``param`` added.

Placement-aware: a ``pre_stream`` error, and any error of a unary request, is
an HTTP error body. An ``in_band_on_stream`` error of a streaming request is a
complete HTTP 200 SSE stream (``response.created`` -> ``response.in_progress``
-> ``error`` -> ``response.failed``) produced before anything is dispatched,
metered, or stored, because Codex acts on the code of ``response.failed`` and
treats a pre-stream 400 as terminal. A failure after a buffered stream opened
ends it in-band with a ``ResponseErrorCode``: backpressure is
``rate_limit_exceeded`` (Codex retries it), a generation failure
``server_error`` (Principle 4).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse, Response

from kairyu.entrypoints.server.chat_errors import ChatRequestError
from kairyu.entrypoints.server.error_classifier import (
    ClassifiedError,
    backpressure,
    classify_error_body,
    classify_request_error,
    pre_stream_error,
    tenant_reservation_refused,
)
from kairyu.entrypoints.server.errors import classified_response
from kairyu.entrypoints.server.responses.envelope import (
    response_envelope,
    usage_payload_from_wire,
)
from kairyu.entrypoints.server.responses.events import responses_sse
from kairyu.entrypoints.server.sse_response import sse_response

if TYPE_CHECKING:
    from kairyu.entrypoints.server.engine_admission import TenantRefusal
    from kairyu.entrypoints.server.responses.request import ResponsesRequest
    from kairyu.entrypoints.server.tenancy import TenantAdmission

_DEFAULT_MESSAGE = "generation failed"


class BufferedFailure(Exception):
    """Generation failure carrying its classified wire error.

    The buffered path surfaces it as an HTTP error before streaming starts
    (non-stream requests) or as ``error`` + ``response.failed`` SSE events
    once the stream is already open.
    """

    def __init__(self, error: ClassifiedError) -> None:
        super().__init__(error.message)
        self.error = error

    @classmethod
    def from_payload(cls, payload: Mapping, status_code: int) -> BufferedFailure:
        return cls(delegated_error(status_code, {"error": payload}))

    @classmethod
    def from_chat_error(cls, error: ChatRequestError) -> BufferedFailure:
        return cls.from_payload(error.payload(), error.status_code)

    @property
    def in_band_code(self) -> str:
        """The ``response.failed`` code (a ``ResponseErrorCode`` member)."""

        if self.error.status == 429:
            return "rate_limit_exceeded"
        if self.error.code in (None, "backend_error"):
            return "server_error"
        return self.error.code

    def json_response(self) -> JSONResponse:
        return responses_error(self.error)


def responses_error(error: ClassifiedError) -> JSONResponse:
    return classified_response(error, responses=True)


def request_error(
    message: str,
    *,
    status_code: int = 400,
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    return responses_error(
        pre_stream_error(status_code, "invalid_request_error", code, message, param=param)
    )


def chat_error(error: ChatRequestError) -> JSONResponse:
    return responses_error(error.classified())


def previous_response_not_found(response_id: str) -> JSONResponse:
    """Unknown, unstored and other tenants' ids answer alike (D-d, tenant-blind)."""

    return request_error(
        f"Previous response with id {response_id!r} not found.",
        code="previous_response_not_found",
        param="previous_response_id",
    )


def tenant_refused(refusal: TenantRefusal) -> JSONResponse:
    return responses_error(
        tenant_reservation_refused(
            refusal.tenant,
            refusal.reason,
            retry_after_s=refusal.retry_after_s,
            exceeds_capacity=refusal.exceeds_token_capacity,
        )
    )


def request_failure(request: ResponsesRequest, error: BaseException) -> Response | None:
    """The whole HTTP response for a classified pre-dispatch failure, else None."""

    classified = classify_request_error(error, "responses")
    return None if classified is None else _render(request, classified)


def delegated_failure(
    request: ResponsesRequest,
    delegated: JSONResponse,
    admission: TenantAdmission | None,
) -> Response:
    """Re-render a delegated Chat error (AUTO) the Responses way.

    The Chat handler reports an overflow of the client prompt (AUTO preflight,
    or a direct route's upstream on a unary run) and refuses tenant
    reservations; ``admission`` is the request's own tenant lease, which knows
    whether a refused reservation could ever fit.
    """

    try:
        body = json.loads(bytes(delegated.body))
    except ValueError:
        return delegated
    return _render(request, delegated_error(delegated.status_code, body, admission))


def delegated_error(
    status: int,
    body: object,
    admission: TenantAdmission | None = None,
) -> ClassifiedError:
    """A Chat-rendered error body as a Responses classification (extras stripped)."""

    overflow = classify_error_body(body, "responses")
    if overflow is not None:
        return overflow
    member = body.get("error") if isinstance(body, Mapping) else None
    member = member if isinstance(member, Mapping) else {}
    message = _text(member.get("message")) or _DEFAULT_MESSAGE
    code = _text(member.get("code"))
    if status == 429 and code == "tenant_rate_limited" and admission is not None:
        return tenant_reservation_refused(
            admission.tenant,
            admission.reason,
            retry_after_s=admission.retry_after_s,
            exceeds_capacity=admission.exceeds_token_capacity,
        )
    if status == 429:
        return backpressure(message, code or "rate_limit_exceeded")
    error_type = _text(member.get("type")) or (
        "invalid_request_error" if status < 500 else "server_error"
    )
    return pre_stream_error(status, error_type, code, message, param=_text(member.get("param")))


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _render(request: ResponsesRequest, classified: ClassifiedError) -> Response:
    if not (request.stream and classified.placement == "in_band_on_stream"):
        return responses_error(classified)
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
