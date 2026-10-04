"""Per-surface classification of request failures (M20 WP-04; WP-07 extends).

``classify_request_error`` maps one failure to a frozen ``ClassifiedError`` for
the Chat, Messages, or Responses surface, or returns ``None`` for a failure
this table does not own yet (the caller keeps its existing mapping).

Prompt overflow is classified by the public prompt it concerns:

* the client-derived prompt (an engine or orchestration preflight, or an
  upstream rejecting the request it was sent) becomes
  ``context_length_exceeded``. Responses delivers it in-band on a stream
  (``placement="in_band_on_stream"``) because Codex compacts only on
  ``response.failed``; Chat and Messages reject before the stream opens
  (``pre_stream``) when the overflow is found before dispatch. An upstream
  reports it only after the SSE headers are committed, so there it is the
  stream's in-band error (a recorded deviation, m9 D6 amendment 2026-10-05).
* an internal orchestration stage prompt (built from candidates, judged
  outputs, …) is a server failure. An L2 error declares that boundary with a
  ``context_overflow_reason`` attribute; the overflow is logged with it and
  never reported as the client's context window.

Ingress and admission failures are classified here too (WP-07): middleware,
the Chat handler and the Responses routes build a ``ClassifiedError`` and
``render_openai`` renders its OpenAI envelope, ``param`` always present. On a
Responses-dialect path (C25: decided by the path alone, so nothing inspects a
body) a retryable 429 becomes 503 ``slow_down`` with ``Retry-After`` and
``retry-after-ms`` (O-2, m7 D5 amendment); Chat and Messages keep their 429.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from kairyu.engine.request_errors import (
    CONTEXT_LENGTH_EXCEEDED,
    ContextLengthExceededError,
)

logger = logging.getLogger(__name__)

Surface = Literal["chat", "messages", "responses"]
Placement = Literal["pre_stream", "in_band_on_stream"]

# Exception chains are short (wrapper -> backend error); bound the walk anyway.
_MAX_CAUSE_DEPTH = 8
_RESPONSES_OVERFLOW_MESSAGE = (
    "Your input exceeds the context window of this model. "
    "Please adjust your input and try again."
)
_CHAT_OVERFLOW_MESSAGE = (
    "This model's maximum context length was exceeded. "
    "Please reduce the length of the messages or completion."
)
_SERVER_ERROR_MESSAGE = "The server had an error while processing your request."
# /v1/alpha holds Codex's standalone search endpoint (POST {base}/alpha/search).
_RESPONSES_DIALECT_PREFIXES = ("/v1/responses", "/v1/conversations", "/v1/alpha")
_MS_PER_S = 1000
# Chat, Messages and Jev keep their historical 429 + Retry-After: 1 (m7 D5).
_LEGACY_RETRY_HEADERS = {"retry-after": "1"}
_TENANT_BUDGET_TOO_SMALL = (
    "tenant {tenant!r} token budget can never hold this request's reservation; "
    "set a lower max_output_tokens (an omitted max_output_tokens reserves the "
    "model's whole remaining context)"
)


@dataclass(frozen=True)
class ClassifiedError:
    """One failure as a surface renders it (immutable across modules)."""

    status: int
    type: str
    code: str | None
    message: str
    param: str | None
    retryable: bool
    retry_after_s: float | None
    placement: Placement

    def openai_payload(self) -> dict:
        return {
            "message": self.message,
            "type": self.type,
            "param": self.param,
            "code": self.code,
        }


@dataclass(frozen=True)
class _Overflow:
    counts: ContextLengthExceededError | None
    internal_reason: str | None


def classify_request_error(
    error: BaseException,
    surface: Surface,
) -> ClassifiedError | None:
    """Classify one raised failure for ``surface``; ``None`` if not owned."""

    overflow = _find_overflow(error)
    if overflow is None:
        return None
    if overflow.internal_reason is not None:
        # The caller logs the failure itself; this records why it is a 5xx.
        logger.warning(
            "internal orchestration stage overflowed its context window "
            "(reason=%s)",
            overflow.internal_reason,
        )
        return _server_error(surface)
    return _context_length_exceeded(surface, overflow.counts)


def classify_error_body(body: object, surface: Surface) -> ClassifiedError | None:
    """Classify a delegated OpenAI-style error body (``{"error": {...}}``).

    AUTO surfaces reuse the Chat handler and receive its rendered body, so the
    overflow is recognised by its public code alone.
    """

    error = body.get("error") if isinstance(body, Mapping) else None
    if not isinstance(error, Mapping) or error.get("code") != CONTEXT_LENGTH_EXCEEDED:
        return None
    return _context_length_exceeded(surface, None)


def anthropic_stream_failure(error: BaseException) -> tuple[str, str]:
    """Return ``(message, error_type)`` for a Messages in-band error event."""

    classified = classify_request_error(error, "messages")
    if classified is not None:
        return classified.message, classified.type
    # M3: only the class name crosses the wire, never the backend's text.
    return f"upstream backend error ({type(error).__name__})", "api_error"


@dataclass(frozen=True)
class HttpError:
    """One error as an HTTP response: status, JSON body and headers."""

    status: int
    body: dict
    headers: Mapping[str, str]


def speaks_responses_dialect(path: str) -> bool:
    """Whether errors on ``path`` render the Responses way (C25)."""

    return any(
        path == prefix or path.startswith(prefix + "/") for prefix in _RESPONSES_DIALECT_PREFIXES
    )


def pre_stream_error(
    status: int,
    error_type: str,
    code: str | None,
    message: str,
    *,
    param: str | None = None,
) -> ClassifiedError:
    """A failure decided before any output: rendered as the HTTP answer, never retried."""

    return ClassifiedError(
        status=status,
        type=error_type,
        code=code,
        message=message,
        param=param,
        retryable=False,
        retry_after_s=None,
        placement="pre_stream",
    )


def backend_failure(error: BaseException) -> ClassifiedError:
    """A tenant-safe upstream failure: only the class name crosses the wire (M3)."""

    return pre_stream_error(
        502, "upstream_error", "backend_error", f"upstream backend error ({type(error).__name__})"
    )


def backpressure(message: str, code: str, *, retry_after_s: float) -> ClassifiedError:
    """Transient overload or quota: a retryable 429 (503 on the Responses dialect)."""

    return replace(
        pre_stream_error(429, "rate_limit_error", code, message),
        retryable=True,
        retry_after_s=retry_after_s,
    )


def tenant_rate_limited(
    tenant: str, reason: str, *, retry_after_s: float | None = None
) -> ClassifiedError:
    """A tenant quota refusal that a later attempt can pass."""

    return backpressure(
        f"tenant {tenant!r} admission limit exceeded ({reason})",
        "tenant_rate_limited",
        retry_after_s=retry_after_s or 1.0,
    )


def tenant_reservation_refused(
    tenant: str,
    reason: str,
    *,
    retry_after_s: float | None,
    exceeds_capacity: bool,
) -> ClassifiedError:
    """Responses: a reservation above the bucket's capacity is terminal (O-2).

    Retrying it can never succeed, so it is a non-retryable 429 naming
    ``max_output_tokens``; anything else is transient backpressure. Chat and
    Messages render every tenant refusal with ``tenant_rate_limited``.
    """

    if exceeds_capacity:
        message = _TENANT_BUDGET_TOO_SMALL.format(tenant=tenant)
        return pre_stream_error(429, "rate_limit_error", "tenant_budget_too_small", message)
    return tenant_rate_limited(tenant, reason, retry_after_s=retry_after_s)


def render_openai(error: ClassifiedError, *, responses: bool) -> HttpError:
    """The OpenAI-envelope HTTP answer, in the Responses or the Chat dialect.

    Responses (O-2): Codex treats every 429 as terminal and 503
    ``server_is_overloaded`` as non-retryable, but retries 503 ``slow_down``
    after ``Retry-After``; the OpenAI SDKs retry any 503 and read
    ``retry-after-ms`` first. Chat keeps its 429 and ``Retry-After: 1``.
    """

    if not responses:
        headers = dict(_LEGACY_RETRY_HEADERS) if error.retryable else {}
        return HttpError(error.status, {"error": error.openai_payload()}, headers)
    if error.retryable and error.status == 429:
        error = replace(error, status=503, type="service_unavailable_error", code="slow_down")
    return HttpError(error.status, {"error": error.openai_payload()}, _retry_headers(error))


def _retry_headers(error: ClassifiedError) -> dict[str, str]:
    if not error.retryable or error.retry_after_s is None:
        return {}
    wait_ms = max(1, math.ceil(error.retry_after_s * _MS_PER_S))
    return {
        "retry-after": str(math.ceil(wait_ms / _MS_PER_S)),
        "retry-after-ms": str(wait_ms),
    }


def _find_overflow(error: BaseException) -> _Overflow | None:
    internal_reason = None
    found = False
    counts = None
    current: BaseException | None = error
    for _ in range(_MAX_CAUSE_DEPTH):
        if current is None:
            break
        internal_reason = internal_reason or getattr(
            current, "context_overflow_reason", None
        )
        if getattr(current, "code", None) == CONTEXT_LENGTH_EXCEEDED:
            found = True
            if isinstance(current, ContextLengthExceededError):
                counts = current
                break
        following = current.__cause__ or getattr(current, "cause", None)
        current = following if isinstance(following, BaseException) else None
    if not found:
        return None
    return _Overflow(counts=counts, internal_reason=internal_reason)


def _server_error(surface: Surface) -> ClassifiedError:
    return ClassifiedError(
        status=500,
        type="api_error" if surface == "messages" else "server_error",
        code="server_error",
        message=_SERVER_ERROR_MESSAGE,
        param=None,
        retryable=False,
        retry_after_s=None,
        placement="in_band_on_stream",
    )


def _context_length_exceeded(
    surface: Surface,
    counts: ContextLengthExceededError | None,
) -> ClassifiedError:
    if surface == "responses":
        message, param, placement = _RESPONSES_OVERFLOW_MESSAGE, "input", "in_band_on_stream"
    elif surface == "chat":
        message, param, placement = _chat_message(counts), "messages", "pre_stream"
    else:
        message, param, placement = _anthropic_message(counts), None, "pre_stream"
    return ClassifiedError(
        status=400,
        type="invalid_request_error",
        code=CONTEXT_LENGTH_EXCEEDED,
        message=message,
        param=param,
        retryable=False,
        retry_after_s=None,
        placement=placement,
    )


def _chat_message(counts: ContextLengthExceededError | None) -> str:
    """OpenAI's wording, which client libraries match to detect overflow."""

    if counts is None:
        return _CHAT_OVERFLOW_MESSAGE
    if counts.max_tokens is None:
        return (
            f"This model's maximum context length is {counts.max_model_len} "
            f"tokens. However, your messages resulted in {counts.prompt_tokens} "
            "tokens. Please reduce the length of the messages."
        )
    return (
        f"This model's maximum context length is {counts.max_model_len} tokens. "
        "However, you requested "
        f"{counts.prompt_tokens + counts.max_tokens} tokens "
        f"({counts.prompt_tokens} in the messages, {counts.max_tokens} in the "
        "completion). Please reduce the length of the messages or completion."
    )


def _anthropic_message(counts: ContextLengthExceededError | None) -> str:
    """Anthropic's wording; Claude Code parses the counts to trim history."""

    if counts is None:
        return "prompt is too long"
    if counts.max_tokens is None:
        # At least one output token must fit after the prompt.
        maximum = counts.max_model_len - 1
        return f"prompt is too long: {counts.prompt_tokens} tokens > {maximum} maximum"
    if counts.prompt_tokens > counts.max_model_len:
        return (
            f"prompt is too long: {counts.prompt_tokens} tokens > "
            f"{counts.max_model_len} maximum"
        )
    return (
        "input length and `max_tokens` exceed context limit: "
        f"{counts.prompt_tokens} + {counts.max_tokens} > {counts.max_model_len}, "
        "decrease input length or `max_tokens` and try again"
    )
