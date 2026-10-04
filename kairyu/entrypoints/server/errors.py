"""Shared OpenAI-style error payloads and HTTP responses.

Every body is rendered from an ``error_classifier.ClassifiedError``, so the
error member always carries ``message``, ``type``, ``param`` and ``code``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from kairyu.entrypoints.server.chat_errors import ChatRequestError, chat_error_from_value_error
from kairyu.entrypoints.server.error_classifier import (
    ClassifiedError,
    backend_failure,
    backpressure,
    classify_request_error,
    pre_stream_error,
    render_openai,
    tenant_rate_limited,
)

if TYPE_CHECKING:
    from kairyu.entrypoints.server.engine_admission import AdmissionStage, TenantRefusal
    from kairyu.orchestration.orchestrator import OrchestratorExecutionError

logger = logging.getLogger(__name__)


def classified_response(error: ClassifiedError, *, responses: bool = False) -> JSONResponse:
    """The HTTP answer for ``error`` in the Chat or the Responses dialect."""

    rendered = render_openai(error, responses=responses)
    return JSONResponse(
        status_code=rendered.status, content=rendered.body, headers=dict(rendered.headers)
    )


def invalid_request_payload(message: str, code: str = "invalid_request") -> dict:
    return pre_stream_error(400, "invalid_request_error", code, message).openai_payload()


def invalid_request(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": invalid_request_payload(message)},
    )


def model_not_found(model: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": invalid_request_payload(
                f"model {model!r} not found", code="model_not_found"
            )
        },
    )


def sanitize_backend_error(error: BaseException) -> dict:
    """Return a tenant-safe backend failure without arbitrary exception text."""
    return backend_failure(error).openai_payload()


def upstream_error(error: BaseException) -> JSONResponse:
    # The full traceback remains server-side; arbitrary exception strings can
    # contain replica URLs, credentials, and local filesystem paths.
    logger.exception("upstream backend error")
    return JSONResponse(
        status_code=502,
        content={"error": sanitize_backend_error(error)},
    )


def chat_error_response(error: ChatRequestError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": error.payload()})


@dataclass(frozen=True)
class ChatAdmissionErrors:
    """OpenAI rendering of the engine admission chain's failures (Chat family)."""

    def slo_shed(self) -> JSONResponse:
        return classified_response(
            backpressure(
                "predicted TTFT exceeds the configured SLO",
                "slo_admission_shed",
                retry_after_s=1.0,
            )
        )

    def invalid(self, message: str) -> JSONResponse:
        return invalid_request(message)

    def rejected(self, error: ValueError | ChatRequestError) -> JSONResponse:
        if isinstance(error, ChatRequestError):
            return chat_error_response(error)
        return chat_error_response(chat_error_from_value_error(error))

    def upstream_failed(self, stage: AdmissionStage, error: RuntimeError) -> JSONResponse:
        # Every stage logs under upstream_error's single message.
        return upstream_error(error)

    def tenant_limited(self, refusal: TenantRefusal) -> JSONResponse:
        # Chat answers every tenant refusal with a retryable 429 (O-2).
        return classified_response(tenant_rate_limited(refusal.tenant, refusal.reason))


def orchestration_failure(error: OrchestratorExecutionError) -> tuple[int, dict]:
    """Status and error member of a failed AUTO run.

    A direct route that overflowed sent the client prompt itself: 400
    ``context_length_exceeded``. An internal stage that overflowed is a server
    error (WP-04), never reported as the client's context window.
    """

    classified = classify_request_error(error, "chat")
    if classified is not None:
        return classified.status, classified.openai_payload()
    return 502, sanitize_backend_error(error.cause)
