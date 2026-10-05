"""Responses request surface, error envelopes, and response/usage shapes.

The Responses adapter (``responses_service``) normalizes wire requests into the
shared Chat Completions boundary; this module owns the request model, the
supported-field policy, and the response envelope and usage payloads.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from kairyu.entrypoints.server.chat_service import ChatRequestError
from kairyu.entrypoints.server.metering import resolve_cached_tokens, resolve_usage_counts

_ALLOWED_INCLUDE = {"reasoning.encrypted_content"}


class _BufferedFailure(Exception):
    """Generation failure carrying the exact wire error payload.

    The buffered path surfaces it as an HTTP error before streaming starts
    (non-stream requests) or as ``error`` + ``response.failed`` SSE events
    once the stream is already open.
    """

    def __init__(self, payload: dict, status_code: int) -> None:
        super().__init__(payload.get("message") or "generation failed")
        self.payload = payload
        self.status_code = status_code

    @classmethod
    def from_chat_error(cls, error: ChatRequestError) -> _BufferedFailure:
        return cls(error.payload(), error.status_code)

    def json_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code, content={"error": self.payload}
        )


class ResponsesRequest(BaseModel):
    """Supported Responses request surface used by the OpenAI SDK and Codex."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict] = ""
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool = True
    max_output_tokens: int | None = None
    stream: bool = False
    metadata: dict = Field(default_factory=dict)
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool = True
    temperature: float | None = None
    top_p: float | None = None
    text: dict | None = None
    include: list[str] | None = None
    reasoning: dict | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    stream_options: dict | None = None
    client_metadata: dict | None = None
    context_management: list[dict] | None = None
    safety_identifier: str | None = None
    user: str | None = None
    top_logprobs: int | None = None
    truncation: str | None = None
    background: bool | None = None
    conversation: str | dict | None = None
    prompt: dict | None = None
    max_tool_calls: int | None = None
    moderation: dict | None = None
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)


def _request_error(message: str, *, status_code: int = 400, code: str | None = None):
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


def _chat_error(error: ChatRequestError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": error.payload()})


def _validate_request_surface(request: ResponsesRequest) -> None:
    if request.model_extra:
        raise ChatRequestError(
            "unsupported request fields: " + ", ".join(sorted(request.model_extra))
        )
    if request.background:
        raise ChatRequestError("background responses are not supported")
    if request.conversation is not None:
        raise ChatRequestError("conversation is not supported; use previous_response_id")
    if request.prompt is not None:
        raise ChatRequestError("prompt templates are not supported")
    if request.max_tool_calls is not None:
        raise ChatRequestError("max_tool_calls is not supported")
    if request.moderation is not None:
        raise ChatRequestError("moderation is not supported")
    if request.top_logprobs is not None:
        raise ChatRequestError("top_logprobs is not supported by the Responses adapter")
    if request.service_tier not in (None, "auto"):
        raise ChatRequestError("service_tier is not supported; omit it or use 'auto'")
    if request.truncation not in (None, "disabled"):
        raise ChatRequestError("truncation='auto' is not supported")
    unsupported_includes = set(request.include or ()) - _ALLOWED_INCLUDE
    if unsupported_includes:
        raise ChatRequestError(
            "unsupported include values: " + ", ".join(sorted(unsupported_includes))
        )
    if request.prompt_cache_retention not in (None, "in_memory", "24h"):
        raise ChatRequestError("prompt_cache_retention must be 'in_memory' or '24h'")
    if request.context_management:
        raise ChatRequestError("context_management is not supported")
    if request.stream_options is not None:
        if not isinstance(request.stream_options, dict):
            raise ChatRequestError("stream_options must be an object")
        unknown_stream_options = set(request.stream_options) - {"include_obfuscation"}
        if unknown_stream_options:
            raise ChatRequestError(
                "unsupported stream_options fields: "
                + ", ".join(sorted(unknown_stream_options))
            )
        if request.stream_options.get("include_obfuscation") not in (None, False):
            raise ChatRequestError("stream obfuscation is not supported")


def _usage_payload(
    prompt: str,
    completions,
    usage,
) -> dict:
    input_tokens, output_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    cached_tokens = resolve_cached_tokens(usage)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _response_envelope(
    request: ResponsesRequest,
    *,
    response_id: str,
    created_at: int,
    status: str,
    output: list[dict],
    usage: dict | None,
    error: dict | None = None,
    incomplete_details: dict | None = None,
) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "instructions": request.instructions,
        "model": request.model,
        "output": output,
        "parallel_tool_calls": request.parallel_tool_calls,
        "tool_choice": request.tool_choice or "auto",
        "tools": request.tools or [],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "previous_response_id": request.previous_response_id,
        "metadata": request.metadata,
        "max_output_tokens": request.max_output_tokens,
        "reasoning": None,
        "service_tier": "default" if request.service_tier == "auto" else None,
        "text": request.text,
        "usage": usage,
    }


def _usage_payload_from_wire(usage: dict | None) -> dict:
    """Map public Chat Completions usage onto the Responses usage shape."""
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": int(details.get("cached_tokens") or 0)},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }
