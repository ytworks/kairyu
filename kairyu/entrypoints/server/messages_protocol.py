"""Anthropic Messages wire schema and error envelope helpers (issue #508).

Kept separate from ``protocol.py`` so the OpenAI contract stays readable and
independently testable. This module has no imports from the rest of the server
package so ``middleware.py`` and ``tenancy.py`` can share the per-path error
envelope selection without cycles.
"""

from __future__ import annotations

import uuid

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ANTHROPIC_VERSION = "2023-06-01"

_MESSAGES_PATH = "/v1/messages"


def wants_anthropic_envelope(path: str) -> bool:
    """True when errors on ``path`` must use the Anthropic error envelope.

    ASGI ``scope["path"]`` excludes the query string, so Claude Code's
    ``POST /v1/messages?beta=true`` matches the exact path.
    """

    return path == _MESSAGES_PATH or path.startswith(_MESSAGES_PATH + "/")


# HTTP status -> Anthropic error type. 5xx (including Kairyu's 502 upstream
# errors) collapse to api_error; 529 overloaded_error is reserved but never
# emitted in the first slice (Kairyu signals overload via 429 + Retry-After,
# which Anthropic SDKs retry).
_ERROR_TYPES_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def anthropic_error_type_for_status(status_code: int) -> str:
    return _ERROR_TYPES_BY_STATUS.get(status_code, "api_error")


def anthropic_error_payload(
    message: str,
    *,
    error_type: str,
    request_id: str | None = None,
) -> dict:
    payload: dict = {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }
    if request_id:
        payload["request_id"] = f"req_{request_id}"
    return payload


def anthropic_error_response(
    message: str,
    *,
    status_code: int = 400,
    error_type: str | None = None,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content=anthropic_error_payload(
            message,
            error_type=error_type or anthropic_error_type_for_status(status_code),
            request_id=request_id,
        ),
    )


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


class MessagesRequest(BaseModel):
    """Supported Anthropic Messages request surface used by Claude Code.

    The outer envelope tolerates unknown extra fields (``extra="allow"``) so a
    newly added Claude Code field does not fail before compatibility policy can
    inspect it; known fields that change inference semantics are declared below
    and are explicitly mapped or rejected in ``messages_service`` — never
    silently dropped. This deliberately differs from the Responses adapter,
    which rejects every extra.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    max_tokens: int = Field(ge=1)
    messages: list[dict]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    tools: list[dict] | None = None
    tool_choice: dict | None = None
    metadata: dict | None = None
    # Known Anthropic capability fields (each mapped or rejected explicitly).
    thinking: dict | None = None
    output_config: dict | None = None
    context_management: object | None = None
    mcp_servers: object | None = None
    container: object | None = None
    betas: object | None = None
    service_tier: str | None = None


class MessagesCountTokensRequest(MessagesRequest):
    """count_tokens body: the Messages surface without generation controls.

    ``max_tokens`` is not part of the Anthropic count_tokens request; the
    harmless default keeps the shared conversion path intact, and a client
    that replays it anyway still validates.
    """

    max_tokens: int = Field(default=1, ge=1)
