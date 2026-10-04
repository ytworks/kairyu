"""The accepted Responses request surface and its pre-dispatch validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kairyu.entrypoints.server.chat_service import ChatRequestError

_ALLOWED_INCLUDE = {"reasoning.encrypted_content"}


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


def validate_request_surface(request: ResponsesRequest) -> None:
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
