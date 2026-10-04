"""Shared OpenAI-style error payloads and HTTP responses."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from kairyu.entrypoints.server.chat_errors import ChatRequestError, chat_error_from_value_error
from kairyu.entrypoints.server.error_classifier import classify_request_error

if TYPE_CHECKING:
    from kairyu.entrypoints.server.engine_admission import AdmissionStage, TenantRefusal
    from kairyu.orchestration.orchestrator import OrchestratorExecutionError

logger = logging.getLogger(__name__)


def invalid_request_payload(message: str, code: str = "invalid_request") -> dict:
    return {
        "message": message,
        "type": "invalid_request_error",
        "code": code,
    }


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
    return {
        "message": f"upstream backend error ({type(error).__name__})",
        "type": "upstream_error",
        "code": "backend_error",
    }


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


def _rate_limited(message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "1"},
        content={"error": {"message": message, "type": "rate_limit_error", "code": code}},
    )


@dataclass(frozen=True)
class ChatAdmissionErrors:
    """OpenAI rendering of the engine admission chain's failures (Chat family)."""

    def slo_shed(self) -> JSONResponse:
        return _rate_limited("predicted TTFT exceeds the configured SLO", "slo_admission_shed")

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
        return _rate_limited(
            f"tenant {refusal.tenant!r} admission limit exceeded ({refusal.reason})",
            "tenant_rate_limited",
        )


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
