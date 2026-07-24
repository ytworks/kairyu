"""Secret-safety checks for persisted evaluation payloads.

Evaluation metadata is durable and frequently copied into reports. This module
rejects credential material before it crosses a schema, protocol, or persistence
boundary. Errors intentionally contain neither the offending key nor value.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit


class SecretSafetyError(RuntimeError):
    """Raised without exposing credential-bearing input through Pydantic errors."""


class SecretValueRegistry:
    """Known secret values retained only as SHA-256 digest-and-length pairs."""

    def __init__(self, values: Iterable[str] = ()) -> None:
        self._digests_by_byte_length: dict[int, set[bytes]] = {}
        self._registered_secret_digests: set[bytes] = set()
        for value in values:
            self.register(value)

    def register(self, value: str) -> None:
        """Register one non-empty secret without retaining its plaintext."""
        if not isinstance(value, str) or not value:
            raise ValueError("registered secrets must be non-empty strings")
        utf8_value = value.encode("utf-8")
        self._registered_secret_digests.add(_secret_digest_bytes(utf8_value))
        for encoded in (
            utf8_value,
            value.encode("utf-16-le"),
            value.encode("utf-16-be"),
        ):
            self._digests_by_byte_length.setdefault(len(encoded), set()).add(
                _secret_digest_bytes(encoded)
            )

    def matches(self, value: str) -> bool:
        """Return whether ``value`` exactly matches a registered secret."""
        return self.matches_bytes(value.encode("utf-8"))

    def matches_bytes(self, value: bytes) -> bool:
        digests = self._digests_by_byte_length.get(len(value), ())
        return _secret_digest_bytes(value) in digests

    def contains(self, value: str) -> bool:
        """Return whether a registered secret occurs anywhere in text."""
        return self.contains_bytes(value.encode("utf-8"))

    def contains_bytes(self, value: bytes) -> bool:
        """Find embedded secrets without retaining or exposing their plaintext."""
        for length, digests in self._digests_by_byte_length.items():
            if length > len(value):
                continue
            for offset in range(len(value) - length + 1):
                if _secret_digest_bytes(value[offset : offset + length]) in digests:
                    return True
        return self.matches_bytes(value)

    def __len__(self) -> int:
        return len(self._registered_secret_digests)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(count={len(self)})"


# Generic benchmark evidence may legitimately use keys such as ``token``. Only
# unambiguous credential aliases are rejected by key; actual credential values
# are caught by the run-scoped registry and credential-shape checks.
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "bearer",
        "bearertoken",
        "apikey",
        "apikeys",
        "authtoken",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "sessiontoken",
        "clientsecret",
        "consumersecret",
        "secret",
        "password",
        "passwd",
        "pwd",
        "cookie",
        "setcookie",
        "sessioncookie",
        "credential",
        "credentials",
        "privatekey",
        "awsaccesskeyid",
        "awssecretaccesskey",
        "xamzcredential",
        "xamzsignature",
        "xamzsecuritytoken",
        "xgoogcredential",
        "xgoogsignature",
        "googleaccessid",
        "sig",
    }
)
_SENSITIVE_KEY_MARKERS = (
    "apikey",
    "authtoken",
    "accesstoken",
    "refreshtoken",
    "sessiontoken",
    "clientsecret",
    "consumersecret",
    "secretkey",
    "privatekey",
    "authorization",
    "password",
)
_PROVIDER_TOKEN_PREFIXES = (
    "api",
    "auth",
    "aws",
    "azure",
    "gcp",
    "github",
    "gitlab",
    "google",
    "hf",
    "openai",
    "anthropic",
)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_BEARER_VALUE = re.compile(
    r"(?<![A-Za-z0-9])bearer[ \t]+[A-Za-z0-9._~+/=-]{8,}"
    r"(?![A-Za-z0-9._~+/=-])",
    re.IGNORECASE,
)
_BASIC_VALUE = re.compile(
    r"(?<![A-Za-z0-9])basic[ \t]+[A-Za-z0-9+/]{12,}={0,2}"
    r"(?![A-Za-z0-9+/=])",
    re.IGNORECASE,
)
_SK_VALUE = re.compile(
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9][A-Za-z0-9._-]{11,}",
    re.IGNORECASE,
)
_SENSITIVE_PARAMETER = re.compile(
    r"(?:^|[?&#;])(?:x-amz-signature|x-amz-credential|x-goog-signature|"
    r"googleaccessid|api[-_]?key|access[-_]?token|auth[-_]?token|sig)=[^&#\s]+",
    re.IGNORECASE,
)
_ASSIGNMENT_KEY = re.compile(
    r"(?:^|[\s{,;])['\"]?([A-Za-z][A-Za-z0-9_.-]{1,63})['\"]?\s*[:=]",
    re.MULTILINE,
)
_URL = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"']+",
    re.IGNORECASE,
)


def ensure_secret_free_json(
    value: object,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> object:
    """Validate a JSON-like tree recursively and return it unchanged."""
    stack = [value]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if isinstance(current, str):
            _check_string(current, secret_registry)
            continue
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            for key, child in current.items():
                if isinstance(key, str):
                    if _is_sensitive_key(key):
                        _reject()
                    _check_string(key, secret_registry)
                stack.append(child)
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            stack.extend(current)

    return value


def ensure_secret_free_bytes(
    value: bytes,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> bytes:
    """Reject credentials in the exact bytes about to cross storage."""
    if not isinstance(value, bytes):
        raise TypeError("secret-safe byte payload must be bytes")
    if secret_registry is not None and secret_registry.contains_bytes(value):
        _reject()
    _check_string(value.decode("utf-8", errors="ignore"), secret_registry)
    return value


def ensure_secret_free_serialized_json(
    value: bytes,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> bytes:
    """Scan one immutable canonical JSON snapshot and its decoded tree."""
    ensure_secret_free_bytes(value, secret_registry=secret_registry)
    try:
        payload = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("serialized evaluation JSON is invalid") from exc
    ensure_secret_free_json(payload, secret_registry=secret_registry)
    return value


def secret_registry_from_context(context: object) -> SecretValueRegistry | None:
    """Extract the optional Pydantic validation-context registry safely."""
    if not isinstance(context, Mapping):
        return None
    registry = context.get("secret_registry")
    return registry if isinstance(registry, SecretValueRegistry) else None


def _check_string(
    value: str,
    secret_registry: SecretValueRegistry | None,
) -> None:
    if _BEARER_VALUE.search(value) or _BASIC_VALUE.search(value) or _SK_VALUE.search(value):
        _reject()
    if secret_registry is not None and secret_registry.contains(value):
        _reject()
    if _SENSITIVE_PARAMETER.search(value):
        _reject()
    if any(_is_sensitive_key(match.group(1)) for match in _ASSIGNMENT_KEY.finditer(value)):
        _reject()
    for match in _URL.finditer(value):
        _check_url(match.group(0), secret_registry)


def _check_url(
    value: str,
    secret_registry: SecretValueRegistry | None,
) -> None:
    authority = value.split("://", 1)[1]
    authority = re.split(r"[/\?#]", authority, maxsplit=1)[0]
    if "@" in authority:
        _reject()
    try:
        parsed = urlsplit(value)
        parameters = (
            *parse_qsl(parsed.query, keep_blank_values=True),
            *parse_qsl(parsed.fragment, keep_blank_values=True),
        )
    except ValueError:
        _reject()
    if parsed.username is not None or parsed.password is not None:
        _reject()
    for key, parameter_value in parameters:
        if _is_sensitive_key(key):
            _reject()
        if secret_registry is not None and secret_registry.contains(parameter_value):
            _reject()


def _is_sensitive_key(value: str) -> bool:
    normalised = _NON_ALPHANUMERIC.sub("", value.casefold())
    return (
        normalised in _SENSITIVE_KEYS
        or any(marker in normalised for marker in _SENSITIVE_KEY_MARKERS)
        or (normalised.endswith("token") and normalised.startswith(_PROVIDER_TOKEN_PREFIXES))
    )


def _secret_digest_bytes(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def _reject() -> None:
    raise SecretSafetyError("evaluation payload contains forbidden credential material")


__all__ = [
    "SecretSafetyError",
    "SecretValueRegistry",
    "ensure_secret_free_bytes",
    "ensure_secret_free_json",
    "ensure_secret_free_serialized_json",
    "secret_registry_from_context",
]
