"""Secret-safe, bounded connectors for evaluation model calls.

The connector boundary deliberately has no persistence or logging behavior.
Callers receive immutable structured results and decide what evidence to store
through the evaluation artifact and control stores.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from ipaddress import ip_address
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from kairyu.evaluation.safety import (
    SecretSafetyError,
    SecretValueRegistry,
    ensure_secret_free_bytes,
    ensure_secret_free_json,
)
from kairyu.evaluation.schemas import FrozenModel, freeze_json_value, thaw_json_value

CancelCallback = Callable[[], bool]
BackoffCallback = Callable[[int], None]
Clock = Callable[[], float]
SecretResolver = Callable[[str], str | None]

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_JSON_SCHEMA_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_INLINE_IMAGE_DATA_URL = re.compile(
    r"data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/]*={0,2})\Z"
)
_MAX_RETRIES = 10
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_INLINE_IMAGE_URL_CHARS = 28_000_000
_MAX_JSON_SCHEMA_BYTES = 256 * 1024


class _ConnectorModel(FrozenModel):
    """Frozen model that preserves output-affecting message whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        populate_by_name=True,
        serialize_by_alias=True,
        str_strip_whitespace=False,
    )


class ConnectorTextPart(_ConnectorModel):
    """One OpenAI chat text content part."""

    type: Literal["text"] = "text"
    text: Annotated[str, Field(min_length=1, max_length=4_000_000)]


class ConnectorImageURL(_ConnectorModel):
    """One bounded inline image URL; remote provider fetches are forbidden."""

    url: Annotated[str, Field(min_length=1, max_length=_MAX_INLINE_IMAGE_URL_CHARS)]
    detail: Literal["auto", "low", "high"] | None = None

    @field_validator("url")
    @classmethod
    def _validate_inline_image(
        cls,
        value: str,
        info: ValidationInfo,
    ) -> str:
        _decode_inline_image(
            value,
            secret_registry=_registry_from_validation_info(info),
        )
        return value

    @property
    def decoded_size(self) -> int:
        """Return validated decoded image bytes without retaining image data."""

        return _decoded_base64_size(self.url.rsplit(",", 1)[1])


class ConnectorImagePart(_ConnectorModel):
    """One OpenAI chat inline image content part."""

    type: Literal["image_url"] = "image_url"
    image_url: ConnectorImageURL


ConnectorContentPart: TypeAlias = Annotated[
    ConnectorTextPart | ConnectorImagePart,
    Field(discriminator="type"),
]
ConnectorMessageContent: TypeAlias = (
    str
    | Annotated[
        tuple[ConnectorContentPart, ...],
        Field(min_length=1, max_length=256),
    ]
)


class ConnectorMessage(_ConnectorModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: ConnectorMessageContent

    @model_validator(mode="after")
    def _images_are_user_input_and_bounded(self) -> ConnectorMessage:
        if isinstance(self.content, str):
            if not self.content or len(self.content) > 4_000_000:
                raise ValueError("text message content is outside the supported bound")
            return self
        images = [part for part in self.content if isinstance(part, ConnectorImagePart)]
        if images and self.role != "user":
            raise ValueError("image content parts are restricted to user messages")
        if sum(part.image_url.decoded_size for part in images) > _MAX_INLINE_IMAGE_BYTES:
            raise ValueError("decoded inline images exceed the per-message byte limit")
        return self


class ConnectorJSONSchema(_ConnectorModel):
    """Named strict JSON Schema for an OpenAI structured response."""

    name: Annotated[str, Field(min_length=1, max_length=64)]
    schema_definition: Mapping[str, JsonValue] = Field(alias="schema")
    strict: Literal[True] = True

    @field_validator("name")
    @classmethod
    def _name_is_portable(cls, value: str) -> str:
        if _JSON_SCHEMA_NAME.fullmatch(value) is None:
            raise ValueError("JSON Schema name must use portable characters")
        return value

    @field_validator("schema_definition", mode="before")
    @classmethod
    def _schema_is_bounded_json(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            raise ValueError("JSON Schema must be an object")
        encoded = _canonical_json_bytes(value)
        if len(encoded) > _MAX_JSON_SCHEMA_BYTES:
            raise ValueError("JSON Schema exceeds the supported byte limit")
        return value

    @model_validator(mode="after")
    def _freeze_schema(self) -> ConnectorJSONSchema:
        object.__setattr__(
            self,
            "schema_definition",
            freeze_json_value(self.schema_definition),
        )
        return self

    @field_serializer("schema_definition")
    def _serialize_schema(self, value: Mapping[str, JsonValue]) -> JsonValue:
        return thaw_json_value(value)


class ConnectorResponseFormat(_ConnectorModel):
    """OpenAI chat-completions JSON Schema response format."""

    type: Literal["json_schema"] = "json_schema"
    json_schema: ConnectorJSONSchema


class ConnectorRequest(_ConnectorModel):
    request_id: Annotated[str, Field(min_length=1, max_length=256)]
    model: Annotated[str, Field(min_length=1, max_length=512)]
    messages: Annotated[tuple[ConnectorMessage, ...], Field(min_length=1, max_length=256)]
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0.0
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 1.0
    max_tokens: Annotated[int, Field(ge=1, le=10_000_000)] = 1_024
    seed: int | None = None
    timeout_seconds: Annotated[float, Field(gt=0, le=86_400, allow_inf_nan=False)] = 600.0
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    response_format: ConnectorResponseFormat | None = None
    allow_empty_content: bool = False

    @field_validator("request_id", "model")
    @classmethod
    def _identifier_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("connector identifiers must be non-blank")
        return value

    @model_validator(mode="after")
    def _inline_images_are_request_bounded(self) -> ConnectorRequest:
        total = 0
        for message in self.messages:
            if isinstance(message.content, str):
                continue
            total += sum(
                part.image_url.decoded_size
                for part in message.content
                if isinstance(part, ConnectorImagePart)
            )
        if total > _MAX_INLINE_IMAGE_BYTES:
            raise ValueError("decoded inline images exceed the per-request byte limit")
        return self


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
    content: Annotated[str, Field(max_length=16_000_000)]
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


def canonical_connector_request_json(
    request: ConnectorRequest,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> str:
    """Serialize one validated request deterministically for protocol evidence."""

    if not isinstance(request, ConnectorRequest):
        raise TypeError("request must be a ConnectorRequest")
    registry = secret_registry if secret_registry is not None else request._secret_registry
    snapshot = ConnectorRequest.model_validate(
        request.model_dump(mode="python"),
        context=_validation_context(registry),
    )
    payload = snapshot.model_dump(mode="json")
    ensure_secret_free_json(payload, secret_registry=registry)
    return _canonical_json_bytes(payload).decode("utf-8")


def canonical_connector_request_sha256(
    request: ConnectorRequest,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> str:
    """Hash the canonical request snapshot used as protocol evidence."""

    canonical = canonical_connector_request_json(
        request,
        secret_registry=secret_registry,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        max_attempts: int | None = None,
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
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        cancelled = cancel_requested or _never_cancelled
        attempt_limit = _validate_max_attempts(max_attempts)
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
        attempts = (
            result.response.attempts if result.response is not None else result.error.attempts
        )
        if attempts > attempt_limit:
            return _error_result(
                request.request_id,
                ConnectorErrorCode.MALFORMED,
                "fixed fake response exceeds the per-call attempt budget",
                attempts=attempt_limit,
                secret_registry=self._secret_registry,
            )
        if (
            result.response is not None
            and not result.response.content
            and not request.allow_empty_content
        ):
            return _error_result(
                request.request_id,
                ConnectorErrorCode.EMPTY_RESPONSE,
                "model response content was empty",
                attempts=result.response.attempts,
                provider_request_id=result.response.provider_request_id,
                latency_seconds=result.response.latency_seconds,
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
        max_attempts: int | None = None,
    ) -> ConnectorResult:
        cancelled = cancel_requested or _never_cancelled
        attempt_limit = min(self._max_retries + 1, _validate_max_attempts(max_attempts))
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

        wire_body = _canonical_wire_request_bytes(
            request,
            secret_registry=self._secret_registry,
        )
        headers = {"Content-Type": "application/json"}
        if secret is not None:
            headers["Authorization"] = f"Bearer {secret}"

        attempts = 0
        while True:
            if cancelled():
                return self._cancelled(request.request_id, attempts=attempts, started=started)
            attempts += 1
            try:
                with self._client.stream(
                    "POST",
                    self._url,
                    content=wire_body,
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
                if attempts < attempt_limit:
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
                if attempts < attempt_limit:
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
            if not content and not request.allow_empty_content:
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


def _validate_max_attempts(value: int | None) -> int:
    if value is None:
        return _MAX_RETRIES + 1
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_attempts must be a positive integer")
    return value


def _canonical_wire_request_bytes(
    request: ConnectorRequest,
    *,
    secret_registry: SecretValueRegistry,
) -> bytes:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            message.model_dump(mode="json", exclude_none=True) for message in request.messages
        ],
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stream": False,
    }
    if request.seed is not None:
        payload["seed"] = request.seed
    if request.reasoning_effort is not None:
        payload["reasoning_effort"] = request.reasoning_effort
    if request.response_format is not None:
        payload["response_format"] = request.response_format.model_dump(mode="json")
    encoded = _canonical_json_bytes(payload)
    ensure_secret_free_bytes(encoded, secret_registry=secret_registry)
    return encoded


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError):
        raise ValueError("connector JSON must be finite and serializable") from None


def _registry_from_validation_info(
    info: ValidationInfo,
) -> SecretValueRegistry | None:
    if not isinstance(info.context, Mapping):
        return None
    registry = info.context.get("secret_registry")
    return registry if isinstance(registry, SecretValueRegistry) else None


def _decode_inline_image(
    value: str,
    *,
    secret_registry: SecretValueRegistry | None,
) -> bytes:
    match = _INLINE_IMAGE_DATA_URL.fullmatch(value)
    if match is None:
        raise ValueError("image URL must be a supported inline data:image/...;base64 value")
    mime_type, payload = match.groups()
    decoded_size = _decoded_base64_size(payload)
    if decoded_size == 0:
        raise ValueError("inline image must not be empty")
    if decoded_size > _MAX_INLINE_IMAGE_BYTES:
        raise ValueError("decoded inline image exceeds the supported byte limit")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("inline image base64 is malformed") from None
    if len(decoded) != decoded_size or base64.b64encode(decoded).decode("ascii") != payload:
        raise ValueError("inline image base64 must use canonical padding")
    _validate_image_signature(mime_type, decoded)
    ensure_secret_free_bytes(decoded, secret_registry=secret_registry)
    return decoded


def _decoded_base64_size(payload: str) -> int:
    if len(payload) % 4:
        raise ValueError("inline image base64 must use canonical padding")
    padding = len(payload) - len(payload.rstrip("="))
    return (len(payload) // 4) * 3 - padding


def _validate_image_signature(mime_type: str, decoded: bytes) -> None:
    valid = {
        "image/png": decoded.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": decoded.startswith(b"\xff\xd8\xff"),
        "image/gif": decoded.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": (
            len(decoded) >= 12 and decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP"
        ),
    }
    if not valid[mime_type]:
        raise ValueError("inline image bytes do not match the declared MIME type")


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
    "ConnectorContentPart",
    "ConnectorError",
    "ConnectorErrorCode",
    "ConnectorImagePart",
    "ConnectorImageURL",
    "ConnectorJSONSchema",
    "ConnectorMessage",
    "ConnectorMessageContent",
    "ConnectorRequest",
    "ConnectorResponse",
    "ConnectorResponseFormat",
    "ConnectorResult",
    "ConnectorTextPart",
    "ConnectorUsage",
    "FakeOpenAIConnector",
    "ModelConnector",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatibleConnector",
    "canonical_connector_request_json",
    "canonical_connector_request_sha256",
    "normalize_openai_base_url",
]
