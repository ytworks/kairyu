"""Responses request surface, error envelopes, and response/usage shapes.

The Responses adapter (``responses_service``) normalizes wire requests into the
shared Chat Completions boundary; this module owns the request model, the
supported-field policy, and the response envelope and usage payloads.

Every ``/v1/responses*`` error uses the full OpenAI error object (``message``,
``type``, ``param``, ``code``). Fields of the pinned OpenAI spec that Kairyu
cannot honor in L3 fail before dispatch with ``param`` naming the field;
fields whose semantics are vacuous here (no hosted tool ever runs, the prefix
cache is automatic) are accepted and echoed.
"""

from __future__ import annotations

import json
import time
from typing import TypeVar

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kairyu.entrypoints.server.chat_service import ChatRequestError
from kairyu.entrypoints.server.errors import openai_error_payload
from kairyu.entrypoints.server.metering import resolve_cached_tokens, resolve_usage_counts

# Every ResponseIncludable value of the pinned spec. Hosted-tool values are
# vacuous (no hosted tool runs); reasoning.encrypted_content is honored.
_INCLUDABLE = frozenset(
    {
        "file_search_call.results",
        "web_search_call.results",
        "web_search_call.action.sources",
        "message.input_image.image_url",
        "computer_call_output.output.image_url",
        "code_interpreter_call.outputs",
        "reasoning.encrypted_content",
        "message.output_text.logprobs",
    }
)
_UNSUPPORTED_INCLUDES = frozenset({"message.output_text.logprobs"})
_SERVICE_TIERS = frozenset({"auto", "default", "flex", "scale", "priority", "fast"})
_REASONING_FIELDS = frozenset({"effort", "summary", "generate_summary", "context", "mode"})
_REASONING_SUMMARIES = frozenset({"auto", "concise", "detailed"})
_UNSUPPORTED_FEATURES = {
    "conversation": "conversation objects are not supported; use previous_response_id",
    "prompt": "stored prompt templates are not supported",
    "moderation": "moderation is not supported",
    "access_programs": "access programs are not supported",
}
_ModelT = TypeVar("_ModelT", bound=BaseModel)
_JSON_BODY_ERROR = (
    "We could not parse the JSON body of your request. The Responses API "
    "expects a JSON object."
)


class ResponsesError(ChatRequestError):
    """A Responses failure rendered in the OpenAI envelope with ``param``."""

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, code=code, error_type=error_type)
        self.param = param
        self.headers = headers

    def payload(self) -> dict:
        return openai_error_payload(
            str(self), error_type=self.error_type, code=self.code, param=self.param
        )


def responses_error_payload(error: ChatRequestError) -> dict:
    """The OpenAI error object for a chat-boundary or Responses error."""

    if isinstance(error, ResponsesError):
        return error.payload()
    param = "model" if error.code == "model_not_found" else None
    return openai_error_payload(
        str(error), error_type=error.error_type, code=error.code, param=param
    )


def responses_error_response(error: ChatRequestError) -> JSONResponse:
    headers = error.headers if isinstance(error, ResponsesError) else None
    return JSONResponse(
        status_code=error.status_code,
        content={"error": responses_error_payload(error)},
        headers=headers,
    )


def not_found(response_id: str) -> ResponsesError:
    return ResponsesError(f"Response with id '{response_id}' not found.", status_code=404)


class _BufferedFailure(Exception):
    """Generation failure carrying the exact wire error payload.

    The buffered path surfaces it as an HTTP error before streaming starts
    (non-stream requests) or as ``error`` + ``response.failed`` SSE events
    once the stream is already open.
    """

    def __init__(self, payload: dict, status_code: int) -> None:
        super().__init__(payload.get("message") or "generation failed")
        self.payload = {"param": None, **payload}
        self.status_code = status_code

    @classmethod
    def from_chat_error(cls, error: ChatRequestError) -> _BufferedFailure:
        return cls(responses_error_payload(error), error.status_code)

    def json_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code, content={"error": self.payload}
        )


class ResponsesRequest(BaseModel):
    """Responses request surface used by the OpenAI SDK and Codex."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[dict] = ""
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool = True
    max_output_tokens: int | None = None
    stream: bool = False
    metadata: dict | None = None
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
    prompt_cache_options: dict | None = None
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
    max_tool_calls: int | None = Field(default=None, ge=0)
    moderation: dict | None = None
    access_programs: dict | None = None
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)


class ResponsesInputTokensRequest(BaseModel):
    """``POST /v1/responses/input_tokens`` body (every field optional upstream)."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    input: str | list[dict] | None = None
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    reasoning: dict | None = None
    text: dict | None = None
    truncation: str | None = None
    conversation: str | dict | None = None
    personality: str | None = None

    def as_responses_request(self) -> ResponsesRequest:
        if self.model is None:
            raise ResponsesError(
                "Missing required parameter: 'model'.",
                param="model",
                code="missing_required_parameter",
            )
        for field in ("conversation", "personality"):
            if getattr(self, field) is not None:
                raise _unsupported(field, f"{field} is not supported", code="unsupported_parameter")
        request = ResponsesRequest(
            model=self.model,
            input=self.input if self.input is not None else "",
            instructions=self.instructions,
            previous_response_id=self.previous_response_id,
            tools=self.tools,
            tool_choice=self.tool_choice,
            parallel_tool_calls=(
                self.parallel_tool_calls if self.parallel_tool_calls is not None else True
            ),
            reasoning=self.reasoning,
            text=self.text,
            truncation=self.truncation,
        )
        _validate_request_surface(request)
        return request


class ResponsesCompactRequest(BaseModel):
    """``POST /v1/responses/compact`` body."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[dict] | None = None
    instructions: str | None = None
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_options: dict | None = None
    prompt_cache_retention: str | None = None
    service_tier: str | None = None

    def as_responses_request(self) -> ResponsesRequest:
        """The equivalent unstored turn ending in a compaction_trigger item."""

        if self.input is None:
            items: list = []
        elif isinstance(self.input, str):
            items = [{"type": "message", "role": "user", "content": self.input}]
        else:
            items = list(self.input)
        request = ResponsesRequest(
            model=self.model,
            input=[*items, {"type": "compaction_trigger"}],
            instructions=self.instructions,
            previous_response_id=self.previous_response_id,
            prompt_cache_key=self.prompt_cache_key,
            prompt_cache_options=self.prompt_cache_options,
            prompt_cache_retention=self.prompt_cache_retention,
            service_tier=self.service_tier,
            store=False,
        )
        _validate_request_surface(request)
        return request


async def parse_json_object(http_request: Request) -> dict:
    try:
        body = json.loads(await http_request.body())
    except (UnicodeDecodeError, ValueError):
        raise ResponsesError(_JSON_BODY_ERROR) from None
    if not isinstance(body, dict):
        raise ResponsesError(_JSON_BODY_ERROR)
    return body


def validate_model(model_cls: type[_ModelT], body: dict) -> _ModelT:
    """Validate a parsed body, mapping the first pydantic error onto ``param``."""

    try:
        return model_cls.model_validate(body)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        param = _param_from_loc(first.get("loc", ()))
        kind = first.get("type", "")
        if kind == "missing":
            raise ResponsesError(
                f"Missing required parameter: '{param}'.",
                param=param,
                code="missing_required_parameter",
            ) from None
        if kind == "extra_forbidden":
            raise ResponsesError(
                f"Unknown parameter: '{param}'.", param=param, code="unknown_parameter"
            ) from None
        raise ResponsesError(
            f"Invalid value for '{param}': {first.get('msg', 'invalid value')}.",
            param=param,
            code="invalid_type" if kind.endswith("_type") else "invalid_value",
        ) from None


def _param_from_loc(loc: tuple) -> str | None:
    """``("input", "list[dict[any,any]]", 0, "role")`` -> ``input[0].role``.

    Union-branch and validator tags in pydantic locations are not wire paths.
    """

    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and part.isidentifier():
            path += f".{part}" if path else part
    return path or None


def _unsupported(param: str, message: str, *, code: str = "unsupported_value") -> ResponsesError:
    return ResponsesError(message, param=param, code=code)


def _validate_request_surface(request: ResponsesRequest) -> None:
    if request.background:
        raise _unsupported(
            "background", "background responses are not supported", code="unsupported_parameter"
        )
    for field, message in _UNSUPPORTED_FEATURES.items():
        if getattr(request, field) is not None:
            raise _unsupported(field, message, code="unsupported_parameter")
    if request.context_management:
        raise _unsupported(
            "context_management",
            "context_management is not supported; send a compaction_trigger item "
            "or call /v1/responses/compact",
            code="unsupported_parameter",
        )
    if request.top_logprobs:
        raise _unsupported("top_logprobs", "top_logprobs is not supported")
    if request.service_tier is not None and request.service_tier not in _SERVICE_TIERS:
        raise ResponsesError(
            f"Invalid value: '{request.service_tier}'.", param="service_tier", code="invalid_value"
        )
    if request.truncation not in (None, "disabled"):
        raise _unsupported(
            "truncation",
            f"truncation={request.truncation!r} is not supported; only 'disabled' is",
        )
    validate_include(request.include or ())
    if request.prompt_cache_retention not in (None, "in_memory", "24h"):
        raise ResponsesError(
            "prompt_cache_retention must be 'in_memory' or '24h'",
            param="prompt_cache_retention",
            code="invalid_value",
        )
    _validate_prompt_cache_options(request.prompt_cache_options)
    _validate_stream_options(request.stream_options)
    _validate_reasoning(request.reasoning)


def validate_include(values) -> None:
    for index, value in enumerate(values):
        if value not in _INCLUDABLE:
            raise ResponsesError(
                f"Invalid value: '{value}'.", param=f"include[{index}]", code="invalid_value"
            )
        if value in _UNSUPPORTED_INCLUDES:
            raise _unsupported(f"include[{index}]", f"include value '{value}' is not supported")


def _validate_prompt_cache_options(options: dict | None) -> None:
    if options is None:
        return
    for key in sorted(set(options) - {"mode", "ttl", "prewarm", "comparison_response_id"}):
        raise ResponsesError(
            f"Unknown parameter: 'prompt_cache_options.{key}'.",
            param=f"prompt_cache_options.{key}",
            code="unknown_parameter",
        )
    if options.get("mode") not in (None, "implicit", "explicit"):
        raise ResponsesError(
            "prompt_cache_options.mode must be 'implicit' or 'explicit'",
            param="prompt_cache_options.mode",
            code="invalid_value",
        )
    if options.get("ttl") not in (None, "30m"):
        raise ResponsesError(
            "prompt_cache_options.ttl must be '30m'",
            param="prompt_cache_options.ttl",
            code="invalid_value",
        )
    if options.get("prewarm"):
        raise _unsupported(
            "prompt_cache_options.prewarm", "prompt cache prewarming is not supported"
        )
    if options.get("comparison_response_id") is not None:
        raise _unsupported(
            "prompt_cache_options.comparison_response_id",
            "prompt cache diagnostics are not supported",
        )


def _validate_stream_options(options: dict | None) -> None:
    if options is None:
        return
    for key in sorted(set(options) - {"include_obfuscation"}):
        raise ResponsesError(
            f"Unknown parameter: 'stream_options.{key}'.",
            param=f"stream_options.{key}",
            code="unknown_parameter",
        )
    if options.get("include_obfuscation") not in (None, False):
        raise _unsupported(
            "stream_options.include_obfuscation", "stream obfuscation is not supported"
        )


def _validate_reasoning(reasoning: dict | None) -> None:
    if reasoning is None:
        return
    for key in sorted(set(reasoning) - _REASONING_FIELDS):
        raise ResponsesError(
            f"Unknown parameter: 'reasoning.{key}'.",
            param=f"reasoning.{key}",
            code="unknown_parameter",
        )
    for key in ("summary", "generate_summary"):
        if reasoning.get(key) not in (None, *_REASONING_SUMMARIES):
            raise ResponsesError(
                f"reasoning.{key} must be auto, concise, or detailed",
                param=f"reasoning.{key}",
                code="invalid_value",
            )
    if reasoning.get("effort") == "none":
        raise _unsupported(
            "reasoning.effort",
            "reasoning.effort='none' is not supported: disabling thinking is "
            "model-template specific",
        )
    if reasoning.get("context") not in (None, "auto"):
        raise _unsupported("reasoning.context", "only reasoning.context='auto' is supported")
    if reasoning.get("mode") not in (None, "standard"):
        raise _unsupported("reasoning.mode", "only reasoning.mode='standard' is supported")


def _usage_shape(
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
) -> dict:
    return {
        "input_tokens": input_tokens,
        # Backends report cache reads, never cache writes: the field the spec
        # requires stays 0 instead of claiming an unmeasured value.
        "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _usage_payload(
    prompt: str,
    completions,
    usage,
) -> dict:
    input_tokens, output_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    return _usage_shape(input_tokens, resolve_cached_tokens(usage), output_tokens)


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
        "background": False,
        "completed_at": int(time.time()) if status == "completed" else None,
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
        "top_logprobs": request.top_logprobs,
        "previous_response_id": request.previous_response_id,
        "metadata": request.metadata or {},
        "max_output_tokens": request.max_output_tokens,
        "max_tool_calls": request.max_tool_calls,
        "reasoning": dict(request.reasoning) if request.reasoning else None,
        # The tier that actually served the request (the spec's echo rule).
        "service_tier": "default",
        "text": request.text,
        "truncation": request.truncation or "disabled",
        "prompt_cache_key": request.prompt_cache_key,
        "prompt_cache_retention": request.prompt_cache_retention,
        "safety_identifier": request.safety_identifier,
        "user": request.user,
        "access_programs": None,
        "conversation": None,
        "prompt": None,
        "usage": usage,
    }


def _usage_payload_from_wire(usage: dict | None) -> dict:
    """Map public Chat Completions usage onto the Responses usage shape."""
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return _usage_shape(
        int(usage.get("prompt_tokens") or 0),
        int(details.get("cached_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(completion_details.get("reasoning_tokens") or 0),
    )


# Context-window overflow as L3 can see it: L1 raises no typed overflow error,
# so these are the stable messages of the native engine (#496 contract), vLLM
# replicas, and llama.cpp. Failures sanitized before L3 sees them (AUTO
# delegations behind a 502, L2 stage failures) stay generic (documented).
_OVERFLOW_MARKERS = (
    "already fill max_model_len",
    "exceed max_model_len",
    "maximum context length",
    "longer than the maximum model length",
    "exceed_context_size_error",
    "exceeds the available context size",
    "context_length_exceeded",
)


def overflow_text(message: str | None) -> bool:
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)


def is_context_overflow(error: BaseException | None) -> bool:
    """Whether a failure (or what it wraps) reports a context-window overflow."""

    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if overflow_text(str(error)):
            return True
        error = error.__cause__ or error.__context__
    return False


def context_overflow_error(*, exhausted: bool = False) -> ResponsesError:
    """``context_length_exceeded``: Codex auto-compacts on this code in-band."""

    message = (
        "The response filled the model's context window before it finished."
        if exhausted
        else "Your input exceeds the context window of this model. "
        "Please adjust your input and try again."
    )
    return ResponsesError(message, param="input", code="context_length_exceeded")
