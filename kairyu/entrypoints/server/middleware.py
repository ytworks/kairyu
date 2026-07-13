"""Pure-ASGI middleware: auth, concurrency guard, metrics, JSON access log.

Pure ASGI (not ``BaseHTTPMiddleware``) so the concurrency guard holds its slot
until the last body byte of an SSE stream is sent, and metrics measure the
full streamed response (design m7 D4/D5/D8).
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable

_ASGIApp = Callable[..., Awaitable[None]]

_OPEN_PATHS = ("/health", "/readyz")
_GUARDED_PREFIX = "/v1/"

# collapse per-object id path segments (file-…, batch_…, uuids, long hex/digits)
# to {id} so a Prometheus path label cannot explode in cardinality (M1)
_ID_SEGMENT = re.compile(r"^(file-|batch_|resp_|chatcmpl-|cmpl-|[0-9a-f-]{16,}|\d{6,})")


def _template_path(path: str) -> str:
    return "/".join(
        "{id}" if _ID_SEGMENT.match(segment) else segment for segment in path.split("/")
    )

access_logger = logging.getLogger("kairyu.access")


async def _send_json(send: Callable, status: int, payload: dict, headers: dict[str, str]) -> None:
    body = json.dumps(payload).encode()
    raw_headers = [(k.encode(), v.encode()) for k, v in headers.items()]
    raw_headers.append((b"content-type", b"application/json"))
    raw_headers.append((b"content-length", str(len(body)).encode()))
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": body})


def _state(scope: dict) -> dict:
    return scope.setdefault("state", {})


class AuthMiddleware:
    """Static API keys (env-sourced), constant-time compare; /health and /readyz open."""

    def __init__(
        self,
        app: _ASGIApp,
        *,
        api_keys: Iterable[str],
        admin_keys: Iterable[str] = (),
        protect_metrics: bool = False,
    ) -> None:
        self.app = app
        self._api_keys = tuple(api_keys)
        self._admin_keys = tuple(admin_keys)
        self._protect_metrics = protect_metrics

    def _authorized(self, scope: dict) -> bool:
        header = dict(scope.get("headers") or ()).get(b"authorization", b"")
        prefix, _, token = header.decode("latin-1").partition(" ")
        if prefix.lower() != "bearer" or not token:
            return False
        # hmac.compare_digest rejects non-ASCII strings with TypeError; a
        # non-ASCII token can never match an ASCII key, so 401 not 500 (M5)
        if not token.isascii():
            return False
        is_data_plane = False
        for key in self._api_keys:
            is_data_plane |= hmac.compare_digest(token, key)
        is_admin = False
        for key in self._admin_keys:
            is_admin |= hmac.compare_digest(token, key)
        if not (is_data_plane or is_admin):
            return False
        state = _state(scope)
        state["api_key"] = token
        state["is_data_plane"] = is_data_plane
        state["is_admin"] = is_admin
        return True

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        exempt = path in _OPEN_PATHS or (path == "/metrics" and not self._protect_metrics)
        if exempt:
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            if path.startswith(_GUARDED_PREFIX) and not _state(scope)["is_data_plane"]:
                await _send_json(
                    send,
                    403,
                    {
                        "error": {
                            "message": "data-plane API key required",
                            "type": "invalid_request_error",
                            "code": "data_plane_required",
                        }
                    },
                    {},
                )
                return
            await self.app(scope, receive, send)
            return
        await _send_json(
            send,
            401,
            {
                "error": {
                    "message": "missing or invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
            {"www-authenticate": "Bearer"},
        )


class ConcurrencyLimitMiddleware:
    """Global in-flight cap on /v1/*; saturation returns 429 + Retry-After (m7 D5).

    Fine-grained per-client rate limiting is the edge WAF/LB's job — this guard
    only protects the process from overload.
    """

    def __init__(self, app: _ASGIApp, *, limit: int) -> None:
        self.app = app
        self._limit = limit
        self._active = 0

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(_GUARDED_PREFIX):
            await self.app(scope, receive, send)
            return
        if self._active >= self._limit:
            await _send_json(
                send,
                429,
                {
                    "error": {
                        "message": f"server is at max concurrency ({self._limit})",
                        "type": "rate_limit_error",
                        "code": "concurrency_exceeded",
                    }
                },
                {"retry-after": "1"},
            )
            return
        self._active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self._active -= 1


class MetricsMiddleware:
    """Record kairyu_requests_total{model,code} and duration; model set by the handler."""

    def __init__(self, app: _ASGIApp, *, metrics) -> None:
        self.app = app
        self._metrics = metrics

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = {"code": 500}

        async def wrapped_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            path = _template_path(scope["path"])  # bounded-cardinality label (M1)
            self._metrics.request_duration_seconds.labels(path=path).observe(
                time.perf_counter() - started
            )
            if scope["path"].startswith(_GUARDED_PREFIX):
                # an unknown model (404) collapses to "unknown" so an attacker
                # looping random model names can't grow the timeseries (M1)
                model = _state(scope).get("model", "-")
                if status["code"] == 404:
                    model = "unknown"
                self._metrics.requests_total.labels(
                    model=model, code=str(status["code"])
                ).inc()


class AccessLogMiddleware:
    """One JSON line per request; assigns and echoes X-Request-ID."""

    def __init__(self, app: _ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:16]
        _state(scope)["request_id"] = request_id
        started = time.perf_counter()
        status = {"code": 500}

        async def wrapped_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            access_logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "code": status["code"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )


class JsonLogFormatter(logging.Formatter):
    """Stdlib JSON formatter (m7 D8 — no structlog/OTel dependency)."""

    _RESERVED = frozenset(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Route root logging through the JSON formatter (used by `kairyu serve`)."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


class TracingMiddleware:
    """Gateway request span (m10a D4): one span per /v1/* request.

    Pure ASGI like the rest of this file — streaming responses must not be
    buffered by a BaseHTTPMiddleware."""

    def __init__(self, app: _ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        from kairyu.telemetry import traced_span

        status = {"code": 500}

        async def wrapped_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        with traced_span(
            "kairyu.request",
            {"http.route": scope["path"], "http.method": scope.get("method", "")},
        ) as span:
            await self.app(scope, receive, wrapped_send)
            if span is not None:
                span.set_attribute("http.status_code", status["code"])
