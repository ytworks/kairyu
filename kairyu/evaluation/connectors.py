"""Secret-safe, bounded connectors for evaluation model calls.

The connector boundary deliberately has no persistence or logging behavior.
Callers receive immutable structured results and decide what evidence to store
through the evaluation artifact and control stores.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from ipaddress import ip_address
from typing import Annotated, Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ConfigDict, Field, field_validator, model_validator

from kairyu.evaluation.safety import (
    SecretSafetyError,
    SecretValueRegistry,
    ensure_secret_free_bytes,
    ensure_secret_free_json,
)
from kairyu.evaluation.schemas import FrozenModel

CancelCallback = Callable[[], bool]
BackoffCallback = Callable[[int], None]
Clock = Callable[[], float]
SecretResolver = Callable[[str], str | None]

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MAX_RETRIES = 10
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class _ConnectorModel(FrozenModel):
    """Frozen model that preserves output-affecting message whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        str_strip_whitespace=False,
    )


class ConnectorMessage(_ConnectorModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=4_000_000)]


class ConnectorRequest(_ConnectorModel):
    request_id: Annotated[str, Field(min_length=1, max_length=256)]
    model: Annotated[str, Field(min_length=1, max_length=512)]
    messages: Annotated[tuple[ConnectorMessage, ...], Field(min_length=1, max_length=256)]
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0.0
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 1.0
    max_tokens: Annotated[int, Field(ge=1, le=10_000_000)] = 1_024
    seed: int | None = None
    timeout_seconds: Annotated[float, Field(gt=0, le=86_400, allow_inf_nan=False)] = 600.0

    @field_validator("request_id", "model")
    @classmethod
    def _identifier_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("connector identifiers must be non-blank")
        return value


class ConnectorUsage(_ConnectorModel):
    prompt_tokens: Annotated[int, Field(ge=0)]
    completion_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _total_covers_each_component(self) -> ConnectorUsage:
        if self.total_tokens < max(self.prompt_tokens, self.completion_tokens):
            raise ValueError("total token count cannot be smaller than a component")
        return self


class ConnectorResponse(_ConnectorModel):
    request_id: Annotated[str, Field(min_length=1, max_length=256)]
    content: Annotated[str, Field(min_length=1, max_length=16_000_000)]
    finish_reason: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    provider_request_id: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    provider_model: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    usage: ConnectorUsage | None = None
    latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    attempts: Annotated[int, Field(ge=1, le=_MAX_RETRIES + 1)] = 1


class ConnectorErrorCode(StrEnum):
    CANCELLED = "cancelled"
    CREDENTIAL_MISSING = "credential_missing"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    TRANSPORT = "transport"
    HTTP_ERROR = "http_error"
    MALFORMED = "malformed"
    EMPTY_RESPONSE = "empty_response"
    RESPONSE_TOO_LARGE = "response_too_large"


class ConnectorError(_ConnectorModel):
    request_id: Annotated[str, Field(min_length=1, max_length=256)]
    code: ConnectorErrorCode
    detail: Annotated[str, Field(min_length=1, max_length=200)]
    retryable: bool = False
    attempts: Annotated[int, Field(ge=0, le=_MAX_RETRIES + 1)] = 0
    status_code: Annotated[int, Field(ge=100, le=599)] | None = None
    provider_request_id: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    latency_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0


class ConnectorResult(_ConnectorModel):
    response: ConnectorResponse | None = None
    error: ConnectorError | None = None

    @model_validator(mode="after")
    def _has_exactly_one_outcome(self) -> ConnectorResult:
        if (self.response is None) == (self.error is None):
            raise ValueError("connector result must contain exactly one outcome")
        return self

    @property
    def request_id(self) -> str:
        outcome = self.response if self.response is not None else self.error
        assert outcome is not None
        return outcome.request_id


# Adapter-facing names keep the boundary independent from one wire provider.
ModelRequest = ConnectorRequest
ModelResponse = ConnectorResult


@runtime_checkable
class ModelConnector(Protocol):
    def complete(
        self,
        request: ModelRequest,
        *,
        cancel_requested: CancelCallback | None = None,
    ) -> ModelResponse: ...


class FakeOpenAIConnector:
    """Completely offline connector returning fixed results by request ID."""

    def __init__(
        self,
        responses: Mapping[str, ConnectorResult],
        *,
        secret_registry: SecretValueRegistry | None = None,
    ) -> None:
        self._secret_registry = secret_registry
        validated: dict[str, ConnectorResult] = {}
        context = _validation_context(secret_registry)
        for request_id, result in responses.items():
            ensure_secret_free_json(request_id, secret_registry=secret_registry)
            snapshot = ConnectorResult.model_validate(
                result.model_dump(mode="python"),
                context=context,
            )
            if snapshot.request_id != request_id:
                raise ValueError("fake response key and request ID must match")
            validated[request_id] = snapshot
        self._responses = validated
        self._requests: list[ConnectorRequest] = []

    @property
    def requests(self) -> tuple[ConnectorRequest, ...]:
        return tuple(self._requests)

    def complete(
        self,
        request: ConnectorRequest,
        *,
        cancel_requested: CancelCallback | None = None,
    ) -> ConnectorResult:
        cancelled = cancel_requested or _never_cancelled
        request = ConnectorRequest.model_validate(
            request.model_dump(mode="python"),
            context=_validation_context(self._secret_registry),
        )
        if cancelled():
            return _error_result(
                request.request_id,
                ConnectorErrorCode.CANCELLED,
                "request cancelled",
                attempts=0,
                secret_registry=self._secret_registry,
            )
        self._requests.append(request)
        result = self._responses.get(request.request_id)
        if cancelled():
            return _error_result(
                request.request_id,
                ConnectorErrorCode.CANCELLED,
                "request cancelled",
                attempts=0,
                secret_registry=self._secret_registry,
            )
        if result is None:
            return _error_result(
                request.request_id,
                ConnectorErrorCode.MALFORMED,
                "no fixed fake response for request",
                attempts=0,
                secret_registry=self._secret_registry,
            )
        return result


class OpenAICompatibleConnector:
    """Synchronous bounded OpenAI-compatible chat-completions connector."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.Client,
        secret_env_name: str | None = None,
        secret_registry: SecretValueRegistry | None = None,
        secret_resolver: SecretResolver | None = None,
        max_response_bytes: int = 1_048_576,
        max_retries: int = 2,
        backoff: BackoffCallback | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(client, httpx.Client):
            raise TypeError("client must be an injected httpx.Client")
        if secret_env_name is not None and not _ENV_NAME.fullmatch(secret_env_name):
            raise ValueError("secret environment name is invalid")
        if (
            not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or not 0 < max_response_bytes <= _MAX_RESPONSE_BYTES
        ):
            raise ValueError("max_response_bytes is outside the supported bound")
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or not 0 <= max_retries <= _MAX_RETRIES
        ):
            raise ValueError("max_retries is outside the supported bound")
        self._url = _chat_completions_url(endpoint)
        self._client = client
        self._secret_env_name = secret_env_name
        self._secret_registry = (
            secret_registry if secret_registry is not None else SecretValueRegistry()
        )
        self._resolve_secret = secret_resolver or os.environ.get
        self._max_response_bytes = max_response_bytes
        self._max_retries = max_retries
        self._backoff = backoff or _default_backoff
        self._clock = clock or time.monotonic

    @property
    def endpoint(self) -> str:
        return self._url

    def complete(
        self,
        request: ConnectorRequest,
        *,
        cancel_requested: CancelCallback | None = None,
    ) -> ConnectorResult:
        cancelled = cancel_requested or _never_cancelled
        started = self._clock()
        if cancelled():
            return self._cancelled(request.request_id, attempts=0, started=started)

        secret: str | None = None
        if self._secret_env_name is not None:
            secret = self._resolve_secret(self._secret_env_name)
            if not secret:
                return _error_result(
                    request.request_id,
                    ConnectorErrorCode.CREDENTIAL_MISSING,
                    "configured credential is unavailable",
                    attempts=0,
                    latency_seconds=self._elapsed(started),
                    secret_registry=self._secret_registry,
                )
            self._secret_registry.register(secret)

        safe_request_id = _safe_local_request_id(
            request.request_id,
            self._secret_registry,
        )
        try:
            ensure_secret_free_json(
                self._url,
                secret_registry=self._secret_registry,
            )
            request = ConnectorRequest.model_validate(
                request.model_dump(mode="python"),
                context=_validation_context(self._secret_registry),
            )
        except (SecretSafetyError, ValueError):
            return _error_result(
                safe_request_id,
                ConnectorErrorCode.MALFORMED,
                "request payload is unsafe or invalid",
                attempts=0,
                latency_seconds=self._elapsed(started),
                secret_registry=self._secret_registry,
            )

        body: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.seed is not None:
            body["seed"] = request.seed
        headers = {"Authorization": f"Bearer {secret}"} if secret is not None else {}

        attempts = 0
        while True:
            if cancelled():
                return self._cancelled(request.request_id, attempts=attempts, started=started)
            attempts += 1
            try:
                with self._client.stream(
                    "POST",
                    self._url,
                    json=body,
                    headers=headers,
                    timeout=request.timeout_seconds,
                ) as response:
                    provider_request_id = _safe_provider_request_id(
                        response.headers.get("x-request-id") or response.headers.get("request-id"),
                        self._secret_registry,
                    )
                    status_code = response.status_code
                    if status_code == 429:
                        raw = b""
                    elif status_code < 200 or status_code >= 300:
                        raw = b""
                    else:
                        raw = self._read_response(response, cancelled)
            except _ConnectorCancelled:
                return self._cancelled(request.request_id, attempts=attempts, started=started)
            except httpx.TimeoutException:
                if cancelled():
                    return self._cancelled(
                        request.request_id,
                        attempts=attempts,
                        started=started,
                    )
                if attempts <= self._max_retries:
                    if self._prepare_retry(attempts, cancelled):
                        continue
                    return self._cancelled(
                        request.request_id,
                        attempts=attempts,
                        started=started,
                    )
                return _error_result(
                    request.request_id,
                    ConnectorErrorCode.TIMEOUT,
                    "model request timed out",
                    attempts=attempts,
                    retryable=True,
                    latency_seconds=self._elapsed(started),
                    secret_registry=self._secret_registry,
                )
            except httpx.HTTPError:
                if cancelled():
                    return self._cancelled(
                        request.request_id,
                        attempts=attempts,
                        started=started,
                    )
                return _error_result(
                    request.request_id,
                    ConnectorErrorCode.TRANSPORT,
                    "model transport failed",
                    attempts=attempts,
                    retryable=True,
                    latency_seconds=self._elapsed(started),
                    secret_registry=self._secret_registry,
                )
            except _ResponseTooLarge:
                return _error_result(
                    request.request_id,
                    ConnectorErrorCode.RESPONSE_TOO_LARGE,
                    "model response exceeded the configured byte limit",
                    attempts=attempts,
                    latency_seconds=self._elapsed(started),
                    secret_registry=self._secret_registry,
                )

            if cancelled():
                return self._cancelled(request.request_id, attempts=attempts, started=started)
            if status_code == 429:
                if attempts <= self._max_retries:
                    if self._prepare_retry(attempts, cancelled):
                        continue
                    return self._cancelled(
                        request.request_id,
                        attempts=attempts,
                        started=started,
                    )
                return _error_result(
                    request.request_id,
                    ConnectorErrorCode.RATE_LIMIT,
                    "model endpoint rate limited the request",
                    attempts=attempts,
                    retryable=True,
                    status_code=429,
                    provider_request_id=provider_request_id,
                    latency_seconds=self._elapsed(started),
                    secret_registry=self._secret_registry,
                )
            if status_code < 200 or status_code >= 300:
                return _error_result(
                    request.request_id,
                    ConnectorErrorCode.HTTP_ERROR,
                    "model endpoint returned an HTTP error",
                    attempts=attempts,
                    status_code=status_code,
                    provider_request_id=provider_request_id,
                    latency_seconds=self._elapsed(started),
                    secret_registry=self._secret_registry,
                )
            if cancelled():
                return self._cancelled(request.request_id, attempts=attempts, started=started)
            return self._parse_response(
                request,
                raw,
                attempts=attempts,
                provider_request_id=provider_request_id,
                started=started,
            )

    def _read_response(
        self,
        response: httpx.Response,
        cancelled: CancelCallback,
    ) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_response_bytes:
                    raise _ResponseTooLarge
            except ValueError:
                pass
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            if cancelled():
                raise _ConnectorCancelled
            size += len(chunk)
            if size > self._max_response_bytes:
                raise _ResponseTooLarge
            chunks.append(chunk)
        if cancelled():
            raise _ConnectorCancelled
        return b"".join(chunks)

    def _parse_response(
        self,
        request: ConnectorRequest,
        raw: bytes,
        *,
        attempts: int,
        provider_request_id: str | None,
        started: float,
    ) -> ConnectorResult:
        try:
            ensure_secret_free_bytes(raw, secret_registry=self._secret_registry)
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError
            choices = payload.get("choices")
            if not isinstance(choices, list):
                raise ValueError
            if not choices:
                return _error_result(
                    request.request_id,
                    ConnectorErrorCode.EMPTY_RESPONSE,
                    "model response contained no choices",
                    attempts=attempts,
                    provider_request_id=provider_request_id,
                    latency_seconds=self._elapsed(started),
                    secret_registry=self._secret_registry,
                )
            if len(choices) != 1 or not isinstance(choices[0], Mapping):
                raise ValueError
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise ValueError
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError
            if not content:
                return _error_result(
                    request.request_id,
                    ConnectorErrorCode.EMPTY_RESPONSE,
                    "model response content was empty",
                    attempts=attempts,
                    provider_request_id=provider_request_id,
                    latency_seconds=self._elapsed(started),
                    secret_registry=self._secret_registry,
                )
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise ValueError
            usage = _parse_usage(payload.get("usage"))
            payload_request_id = payload.get("id")
            if provider_request_id is None and isinstance(payload_request_id, str):
                provider_request_id = _safe_provider_request_id(
                    payload_request_id,
                    self._secret_registry,
                )
            provider_model = payload.get("model")
            if provider_model is not None and not isinstance(provider_model, str):
                raise ValueError
            response = ConnectorResponse.model_validate(
                {
                    "request_id": request.request_id,
                    "content": content,
                    "finish_reason": finish_reason,
                    "provider_request_id": provider_request_id,
                    "provider_model": provider_model,
                    "usage": usage,
                    "latency_seconds": self._elapsed(started),
                    "attempts": attempts,
                },
                context=_validation_context(self._secret_registry),
            )
            return ConnectorResult.model_validate(
                {"response": response},
                context=_validation_context(self._secret_registry),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, SecretSafetyError, ValueError):
            return _error_result(
                request.request_id,
                ConnectorErrorCode.MALFORMED,
                "model response was malformed or unsafe",
                attempts=attempts,
                provider_request_id=provider_request_id,
                latency_seconds=self._elapsed(started),
                secret_registry=self._secret_registry,
            )

    def _prepare_retry(self, attempt: int, cancelled: CancelCallback) -> bool:
        if cancelled():
            return False
        self._backoff(attempt)
        return not cancelled()

    def _cancelled(
        self,
        request_id: str,
        *,
        attempts: int,
        started: float,
    ) -> ConnectorResult:
        return _error_result(
            request_id,
            ConnectorErrorCode.CANCELLED,
            "request cancelled",
            attempts=attempts,
            latency_seconds=self._elapsed(started),
            secret_registry=self._secret_registry,
        )

    def _elapsed(self, started: float) -> float:
        return max(0.0, self._clock() - started)


class _ResponseTooLarge(RuntimeError):
    pass


class _ConnectorCancelled(RuntimeError):
    pass


def _parse_usage(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("usage must be an object")
    expected = ("prompt_tokens", "completion_tokens", "total_tokens")
    parsed: dict[str, int] = {}
    for key in expected:
        token_count = value.get(key)
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
            raise ValueError("usage token counts must be non-negative integers")
        parsed[key] = token_count
    return parsed


def normalize_openai_base_url(endpoint: str) -> str:
    """Validate and canonicalize a credential-free OpenAI-compatible base URL."""

    if not isinstance(endpoint, str):
        raise TypeError("endpoint must be a string")
    try:
        ensure_secret_free_json(endpoint)
        parsed = urlsplit(endpoint)
    except (SecretSafetyError, ValueError):
        raise ValueError("endpoint contains forbidden credential material") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint must be a credential-free HTTP base URL")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("plain HTTP model endpoints are restricted to loopback hosts")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        raise ValueError("endpoint must be a base URL, not a completion route")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _chat_completions_url(endpoint: str) -> str:
    parsed = urlsplit(normalize_openai_base_url(endpoint))
    path = f"{parsed.path}/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _is_loopback_host(hostname: str) -> bool:
    lowered = hostname.casefold().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    try:
        return ip_address(lowered).is_loopback
    except ValueError:
        return False


def _safe_provider_request_id(
    value: str | None,
    secret_registry: SecretValueRegistry,
) -> str | None:
    if value is None or not value:
        return None
    try:
        ensure_secret_free_json(value, secret_registry=secret_registry)
    except SecretSafetyError:
        return None
    return value[:512]


def _safe_local_request_id(
    value: str,
    secret_registry: SecretValueRegistry,
) -> str:
    try:
        ensure_secret_free_json(value, secret_registry=secret_registry)
    except SecretSafetyError:
        return "unsafe-request"
    return value


def _error_result(
    request_id: str,
    code: ConnectorErrorCode,
    detail: str,
    *,
    attempts: int,
    retryable: bool = False,
    status_code: int | None = None,
    provider_request_id: str | None = None,
    latency_seconds: float = 0.0,
    secret_registry: SecretValueRegistry | None = None,
) -> ConnectorResult:
    return ConnectorResult.model_validate(
        {
            "error": {
                "request_id": request_id,
                "code": code,
                "detail": detail,
                "retryable": retryable,
                "attempts": attempts,
                "status_code": status_code,
                "provider_request_id": provider_request_id,
                "latency_seconds": latency_seconds,
            }
        },
        context=_validation_context(secret_registry),
    )


def _validation_context(
    registry: SecretValueRegistry | None,
) -> dict[str, SecretValueRegistry] | None:
    return {"secret_registry": registry} if registry is not None else None


def _never_cancelled() -> bool:
    return False


def _default_backoff(attempt: int) -> None:
    time.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))


__all__ = [
    "ConnectorError",
    "ConnectorErrorCode",
    "ConnectorMessage",
    "ConnectorRequest",
    "ConnectorResponse",
    "ConnectorResult",
    "ConnectorUsage",
    "FakeOpenAIConnector",
    "ModelConnector",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatibleConnector",
    "normalize_openai_base_url",
]
