"""Pure-ASGI middleware: auth, concurrency guard, metrics, JSON access log.

Pure ASGI (not ``BaseHTTPMiddleware``) so the concurrency guard holds its slot
until the last body byte of an SSE stream is sent, and metrics measure the
full streamed response (design m7 D4/D5/D8).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable

_ASGIApp = Callable[..., Awaitable[None]]

# /backends is an open introspection endpoint (kept out of /v1/ so the
# concurrency cap and per-model metrics labeling don't apply); its disclosure
# level matches the already-open /readyz and /metrics.
_OPEN_PATHS = ("/health", "/readyz", "/backends")
_GUARDED_PREFIX = "/v1/"
_SLO_ADMISSION_LEASE_STATE_KEY = "slo_admission_lease"

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
    """Bound active and queued /v1/* work; saturation returns 429 (m7 D5).

    Fine-grained per-client rate limiting is the edge WAF/LB's job — this guard
    only protects the process from overload.
    """

    def __init__(
        self,
        app: _ASGIApp,
        *,
        limit: int,
        total_limit: int,
        wait_timeout_s: float,
    ) -> None:
        if not 1 <= limit <= total_limit:
            raise ValueError("active limit must be within the total concurrency limit")
        if wait_timeout_s <= 0:
            raise ValueError("admission wait timeout must be positive")
        self.app = app
        self._limit = limit
        self._total_limit = total_limit
        self._queue_limit = total_limit - limit
        self._wait_timeout_s = wait_timeout_s
        self._slots = asyncio.Semaphore(limit)
        self._active = 0
        self._waiting = 0

    async def _reject(self, send: Callable, message: str) -> None:
        await _send_json(
            send,
            429,
            {
                "error": {
                    "message": message,
                    "type": "rate_limit_error",
                    "code": "concurrency_exceeded",
                }
            },
            {"retry-after": "1"},
        )

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(_GUARDED_PREFIX):
            await self.app(scope, receive, send)
            return
        if self._slots.locked():
            if self._waiting >= self._queue_limit:
                message = (
                    f"server is at max concurrency ({self._limit})"
                    if self._queue_limit == 0
                    else f"server admission queue is full ({self._total_limit})"
                )
                await self._reject(
                    send,
                    message,
                )
                return
            self._waiting += 1
            timed_out = False
            try:
                await asyncio.wait_for(
                    self._slots.acquire(),
                    timeout=self._wait_timeout_s,
                )
            except TimeoutError:
                timed_out = True
            finally:
                self._waiting -= 1
            if timed_out:
                await self._reject(
                    send,
                    "server admission queue wait timed out",
                )
                return
        else:
            await self._slots.acquire()
        self._active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self._active -= 1
            self._slots.release()


class ChatBodyLimitMiddleware:
    """Bound Chat Completions JSON before Starlette materializes the body."""

    _PATH = "/v1/chat/completions"

    def __init__(self, app: _ASGIApp, *, limit: int) -> None:
        self.app = app
        self._limit = limit

    async def _reject(self, send: Callable) -> None:
        await _send_json(
            send,
            413,
            {
                "error": {
                    "message": (
                        "chat request body exceeds the configured "
                        f"{self._limit}-byte limit"
                    ),
                    "type": "invalid_request_error",
                    "code": "request_too_large",
                }
            },
            {},
        )

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self._PATH
        ):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or ())
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                content_length = None
            if content_length is not None and content_length > self._limit:
                await self._reject(send)
                return

        received = 0
        buffered: deque[dict] = deque()
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            received += len(message.get("body", b""))
            if received > self._limit:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        async def replay_receive() -> dict:
            if buffered:
                return buffered.popleft()
            return await receive()

        await self.app(scope, replay_receive, send)


class MetricsMiddleware:
    """Record kairyu_requests_total{model,code} and duration; model set by the handler."""

    def __init__(self, app: _ASGIApp, *, metrics) -> None:
        self.app = app
        self._metrics = metrics

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started_ns = time.perf_counter_ns()
        # RequestIngressMiddleware owns the true outer-boundary timestamp.
        # Retain this fallback for direct middleware tests and embedded stacks.
        _state(scope).setdefault("placement_started_ns", started_ns)
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
                (time.perf_counter_ns() - started_ns) / 1_000_000_000
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


class RequestIngressMiddleware:
    """Own request-boundary timing and direct-chat SLO lease cleanup."""

    def __init__(self, app: _ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "http":
            _state(scope)["placement_started_ns"] = time.perf_counter_ns()
        try:
            await self.app(scope, receive, send)
        finally:
            if scope["type"] == "http":
                lease = _state(scope).pop(_SLO_ADMISSION_LEASE_STATE_KEY, None)
                if lease is not None and lease.active:
                    lease.completed()


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
    # Successful outbound requests are already represented by Kairyu's
    # authoritative structured access line. Keep dependency warnings/errors
    # without serializing another INFO line for every replica dispatch.
    logging.getLogger("httpx").setLevel(logging.WARNING)


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
        from kairyu.telemetry import (
            extract_trace_context,
            mark_span_error,
            traced_span,
        )

        status = {"code": 500, "complete": False, "disconnected": False}

        async def wrapped_receive() -> dict:
            message = await receive()
            if message["type"] == "http.disconnect":
                status["disconnected"] = True
            return message

        async def wrapped_send(message: dict) -> None:
            try:
                await send(message)
            except OSError:
                # ASGI 2.4+ reports a disconnected client from send(); older
                # servers expose the same condition through http.disconnect.
                status["disconnected"] = True
                raise
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            elif (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                # Complete means the downstream server accepted the final
                # body, not merely that the application attempted to send it.
                status["complete"] = True

        attributes: dict[str, str] = {
            # Keep the legacy names for existing dashboards while also
            # emitting the current HTTP semantic-convention spelling.
            "http.route": scope["path"],
            "http.method": scope.get("method", ""),
            "http.request.method": scope.get("method", ""),
        }
        request_id = _state(scope).get("request_id")
        if isinstance(request_id, str):
            attributes["kairyu.request_id"] = request_id
        carrier = {
            key.decode("latin-1"): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        with traced_span(
            "kairyu.request",
            attributes,
            context=extract_trace_context(carrier),
            kind="server",
        ) as span:
            try:
                await self.app(scope, wrapped_receive, wrapped_send)
            finally:
                if span is not None:
                    span.set_attribute("http.status_code", status["code"])
                    span.set_attribute("http.response.status_code", status["code"])
                    span.set_attribute(
                        "kairyu.response.complete",
                        status["complete"],
                    )
                    if not status["complete"]:
                        mark_span_error(
                            span,
                            error_type=(
                                "ClientDisconnect"
                                if status["disconnected"]
                                else "incomplete_response"
                            ),
                            cancelled=status["disconnected"],
                        )
                    elif status["code"] >= 500:
                        mark_span_error(span, error_type=str(status["code"]))
