"""Small, auditable HRW HTTP load balancer for the disposable F1c kind gate."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_REPLACED_RESPONSE_HEADERS = frozenset(
    {
        "content-length",
        "date",
        "server",
        "x-kairyu-gateway-id",
        "x-kairyu-lb-request-id",
    }
)
_GATEWAY_ID = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")


@dataclass(frozen=True)
class Gateway:
    gateway_id: str
    host: str
    port: int


def _parse_gateways(raw: str) -> tuple[Gateway, ...]:
    gateways: list[Gateway] = []
    seen: set[str] = set()
    for raw_entry in raw.split(","):
        entry = raw_entry.strip()
        gateway_id, separator, address = entry.partition("=")
        if not separator or not _GATEWAY_ID.fullmatch(gateway_id):
            raise ValueError(f"invalid F1C_GATEWAYS entry: {entry!r}")
        if gateway_id in seen:
            raise ValueError(f"duplicate gateway id: {gateway_id!r}")
        parsed = urlsplit(address)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or parsed.port is None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(f"gateway must be an http://host:port origin: {address!r}")
        seen.add(gateway_id)
        gateways.append(Gateway(gateway_id, parsed.hostname, parsed.port))
    if not gateways:
        raise ValueError("F1C_GATEWAYS must contain at least one gateway")
    return tuple(gateways)


def _rank_gateways(key: str, gateways: tuple[Gateway, ...]) -> tuple[Gateway, ...]:
    # Match Kairyu ReplicaPool's frozen rendezvous input exactly. The gateway
    # ID is the second-stage member identity; the first digest byte is the most
    # significant because byte strings compare lexicographically.
    return tuple(
        sorted(
            gateways,
            key=lambda gateway: (
                hashlib.sha256(f"{key}:{gateway.gateway_id}".encode()).digest(),
                gateway.gateway_id,
            ),
            reverse=True,
        )
    )


class AuditWriter:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._fd = os.open(
            path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )

    def append(self, record: dict[str, object]) -> None:
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
        )
        with self._lock:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(self._fd, remaining)
                if written <= 0:
                    raise OSError(f"short audit write to {self._path}")
                remaining = remaining[written:]


class F1cLoadBalancer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        gateways: tuple[Gateway, ...],
        audit: AuditWriter,
        *,
        upstream_timeout_s: float,
        max_body_bytes: int,
    ) -> None:
        self.gateways = gateways
        self.audit = audit
        self.upstream_timeout_s = upstream_timeout_s
        self.max_body_bytes = max_body_bytes
        super().__init__(address, ProxyHandler)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kairyu-f1c-lb/1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.partition("?")[0] in {"/health", "/readyz"}:
            self._serve_health()
            return
        self._proxy()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy()

    def log_message(self, message: str, *args: object) -> None:
        print(
            json.dumps(
                {
                    "kind": "http_log",
                    "message": message % args,
                    "time_ns": time.time_ns(),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    @property
    def _lb(self) -> F1cLoadBalancer:
        assert isinstance(self.server, F1cLoadBalancer)
        return self.server

    def _serve_health(self) -> None:
        payload = json.dumps(
            {
                "status": "ready",
                "gateway_ids": [
                    gateway.gateway_id for gateway in self._lb.gateways
                ],
            },
            separators=(",", ":"),
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _read_request_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding") is not None:
            self._send_json_error(501, "chunked request bodies are not supported")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            self._send_json_error(400, "invalid Content-Length")
            return None
        if length < 0 or length > self._lb.max_body_bytes:
            self._send_json_error(413, "request body exceeds the F1c fixture limit")
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._send_json_error(400, "incomplete request body")
            return None
        return body

    def _upstream_headers(
        self,
        gateway: Gateway,
        request_id: str,
        body: bytes,
    ) -> dict[str, str]:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS
            and name.lower()
            not in {
                "content-length",
                "host",
                "x-kairyu-gateway-id",
                "x-kairyu-lb-gateway-id",
                "x-kairyu-lb-request-id",
            }
        }
        headers["Host"] = f"{gateway.host}:{gateway.port}"
        headers["Content-Length"] = str(len(body))
        headers["X-Kairyu-LB-Request-ID"] = request_id
        headers["X-Kairyu-LB-Gateway-ID"] = gateway.gateway_id
        return headers

    def _proxy(self) -> None:
        if not self.path.startswith("/") or self.path.startswith("//"):
            self._send_json_error(400, "only origin-form request targets are supported")
            return
        body = self._read_request_body()
        if body is None:
            return

        request_id = uuid.uuid4().hex
        session_id = self.headers.get("X-Session-ID")
        affinity_key = session_id if session_id else request_id
        candidates = _rank_gateways(affinity_key, self._lb.gateways)
        started_ns = time.time_ns()
        attempts: list[dict[str, object]] = []

        for index, gateway in enumerate(candidates):
            attempt_started_ns = time.perf_counter_ns()
            connection = http.client.HTTPConnection(
                gateway.host,
                gateway.port,
                timeout=self._lb.upstream_timeout_s,
            )
            try:
                connection.request(
                    self.command,
                    self.path,
                    body=body,
                    headers=self._upstream_headers(gateway, request_id, body),
                )
                response = connection.getresponse()
                response_body = response.read(self._lb.max_body_bytes + 1)
                if len(response_body) > self._lb.max_body_bytes:
                    raise http.client.HTTPException(
                        "upstream response exceeds the F1c fixture limit"
                    )
                response_headers = response.getheaders()
                upstream_request_id = response.getheader("X-Request-ID")
                retryable = response.status >= 500 and index + 1 < len(candidates)
                attempts.append(
                    {
                        "gateway_id": gateway.gateway_id,
                        "elapsed_ns": time.perf_counter_ns() - attempt_started_ns,
                        "outcome": "retryable_5xx" if retryable else "response",
                        "status": response.status,
                        "upstream_request_id": upstream_request_id,
                    }
                )
                if retryable:
                    continue
                record = self._audit_record(
                    request_id,
                    session_id,
                    body,
                    candidates,
                    attempts,
                    started_ns,
                    gateway.gateway_id,
                    response.status,
                )
                self._lb.audit.append(record)
                self._send_upstream_response(
                    response.status,
                    response.reason,
                    response_headers,
                    response_body,
                    gateway.gateway_id,
                    request_id,
                )
                return
            except (OSError, http.client.HTTPException) as error:
                attempts.append(
                    {
                        "gateway_id": gateway.gateway_id,
                        "elapsed_ns": time.perf_counter_ns() - attempt_started_ns,
                        "error_type": type(error).__name__,
                        "outcome": "transport_error",
                        "status": None,
                        "upstream_request_id": None,
                    }
                )
            finally:
                connection.close()

        record = self._audit_record(
            request_id,
            session_id,
            body,
            candidates,
            attempts,
            started_ns,
            None,
            503,
        )
        self._lb.audit.append(record)
        self._send_json_error(
            503,
            "all F1c gateways are unavailable",
            gateway_id=None,
            request_id=request_id,
        )

    def _audit_record(
        self,
        request_id: str,
        session_id: str | None,
        body: bytes,
        candidates: tuple[Gateway, ...],
        attempts: list[dict[str, object]],
        started_ns: int,
        selected_gateway_id: str | None,
        final_status: int,
    ) -> dict[str, object]:
        return {
            "attempts": attempts,
            "body_bytes": len(body),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "candidate_gateway_ids": [
                gateway.gateway_id for gateway in candidates
            ],
            "final_status": final_status,
            "finished_ns": time.time_ns(),
            "kind": "f1c_lb_decision",
            "method": self.command,
            "path": self.path,
            "request_id": request_id,
            "selected_gateway_id": selected_gateway_id,
            "selection_reason": (
                "session_hrw" if session_id else "request_id_hrw"
            ),
            "session_sha256": (
                hashlib.sha256(session_id.encode()).hexdigest()
                if session_id
                else None
            ),
            "started_ns": started_ns,
        }

    def _send_upstream_response(
        self,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
        body: bytes,
        gateway_id: str,
        request_id: str,
    ) -> None:
        self.send_response(status, reason)
        for name, value in headers:
            lowered = name.lower()
            if (
                lowered not in _HOP_BY_HOP_HEADERS
                and lowered not in _REPLACED_RESPONSE_HEADERS
            ):
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Kairyu-Gateway-ID", gateway_id)
        self.send_header("X-Kairyu-LB-Request-ID", request_id)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json_error(
        self,
        status: int,
        message: str,
        *,
        gateway_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        payload = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "f1c_load_balancer_error",
                }
            },
            separators=(",", ":"),
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if gateway_id is not None:
            self.send_header("X-Kairyu-Gateway-ID", gateway_id)
        if request_id is not None:
            self.send_header("X-Kairyu-LB-Request-ID", request_id)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)


def _positive_float(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int(name: str, default: str) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> None:
    gateways = _parse_gateways(
        os.environ.get(
            "F1C_GATEWAYS",
            (
                "a=http://f1c-gateway-a:8000,"
                "b=http://f1c-gateway-b:8000,"
                "c=http://f1c-gateway-c:8000"
            ),
        )
    )
    audit = AuditWriter(
        os.environ.get("F1C_AUDIT_PATH", "/evidence/decisions.jsonl")
    )
    server = F1cLoadBalancer(
        (
            os.environ.get("F1C_LISTEN_HOST", "0.0.0.0"),
            _positive_int("F1C_LISTEN_PORT", "8082"),
        ),
        gateways,
        audit,
        upstream_timeout_s=_positive_float("F1C_UPSTREAM_TIMEOUT_S", "10"),
        max_body_bytes=_positive_int("F1C_MAX_BODY_BYTES", str(64 * 1024 * 1024)),
    )
    print(
        json.dumps(
            {
                "kind": "startup",
                "gateway_ids": [gateway.gateway_id for gateway in gateways],
                "listen": list(server.server_address),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()
