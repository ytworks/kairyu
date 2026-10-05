"""Shared OpenAI-style error payloads and HTTP responses."""

from __future__ import annotations

import logging

from fastapi.responses import JSONResponse

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


RESPONSES_PATH = "/v1/responses"


def wants_responses_envelope(path: str) -> bool:
    """True for ``/v1/responses`` and its sub-resources (OpenAI envelope + param)."""

    return path == RESPONSES_PATH or path.startswith(RESPONSES_PATH + "/")


def openai_error_payload(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    param: str | None = None,
) -> dict:
    """The full OpenAI error object: ``message``, ``type``, ``param``, ``code``."""

    return {"message": message, "type": error_type, "param": param, "code": code}


def responses_slow_down_payload(message: str) -> dict:
    """503 body for a transient overload on ``/v1/responses``.

    Codex treats HTTP 429 as terminal but waits out ``Retry-After`` and retries
    a 503 whose code is ``slow_down``; rate quotas keep the OpenAI 429.
    """

    return {"error": openai_error_payload(message, error_type="server_error", code="slow_down")}


def tenant_rejection(path: str, message: str, reason: str | None) -> tuple[int, dict]:
    """Status and OpenAI-envelope body for a tenant admission rejection."""

    if wants_responses_envelope(path) and reason == "in_flight":
        return 503, responses_slow_down_payload(message)
    payload = {"message": message, "type": "rate_limit_error", "code": "tenant_rate_limited"}
    if wants_responses_envelope(path):
        payload["param"] = None
    return 429, {"error": payload}
