"""Responses transport: SSE heartbeats, headers, wire escaping, failures; WebSocket probe."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path

import httpx
import pytest
import uvicorn
import uvicorn.protocols.websockets.auto as uvicorn_ws_auto
from fastapi.testclient import TestClient

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints import cli
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import UsageLedger
from kairyu.orchestration.orchestrator import Orchestrator
from tests.server._legacy_chat import create_legacy_app
from tests.server.live_server import LOOPBACK_HOST, SERVER_TIMEOUT_S, serve
from tests.server.responses._helpers import _app, _sse_events, _tool


def _assert_sse_response_headers(response: httpx.Response) -> None:
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert "connection" not in response.headers


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        ("m", {"tools": [_tool()], "tool_choice": "auto"}),
        ("m", {}),
        ("kairyu-auto", {}),
    ],
    ids=["buffered-tool-stream", "live-text-stream", "auto-relay"],
)
def test_every_sse_path_emits_data_heartbeat(tmp_path, monkeypatch, model, payload):
    # Codex resets its 300 s SSE idle timer only on events that carry data;
    # comment keep-alives never reach its parser. Every Responses stream path
    # must emit a real data event (a repeated response.in_progress) while
    # generation is silent, without breaking the gapless sequence numbers.
    from kairyu.entrypoints.server.responses import events

    monkeypatch.setattr(events, "HEARTBEAT_SECONDS", 0.05)
    backend = MockBackend({"hello": "done"}, latency_s=0.3)
    app = create_legacy_app(
        {"m": backend},
        orchestrators={
            "kairyu-auto": Orchestrator({"tier1": backend, "tier2": backend})
        },
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={"model": model, "input": "hello", "stream": True, **payload},
        )
    assert response.status_code == 200
    assert not any(line.startswith(":") for line in response.text.splitlines())
    events = _sse_events(response.text)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert events[0]["type"] == "response.created"
    assert sum(event["type"] == "response.in_progress" for event in events) >= 2
    assert events[-1]["type"] == "response.completed"
    _assert_lifecycle_snapshots_match_events(events)


def _assert_lifecycle_snapshots_match_events(events: list[dict]) -> None:
    # openai-node's ResponseStream replaces its accumulated snapshot with the
    # response of every lifecycle event, so a repeated response.in_progress
    # must carry exactly the output implied by the events sent before it.
    output: list[dict] = []
    for event in events:
        kind = event["type"]
        if kind == "response.output_item.added":
            output.append(copy.deepcopy(event["item"]))
        elif kind == "response.content_part.added":
            output[event["output_index"]]["content"].append(copy.deepcopy(event["part"]))
        elif kind == "response.output_text.delta":
            part = output[event["output_index"]]["content"][event["content_index"]]
            part["text"] += event["delta"]
        elif kind == "response.output_item.done":
            output[event["output_index"]] = copy.deepcopy(event["item"])
        elif kind == "response.in_progress":
            assert event["response"]["output"] == output


@pytest.mark.parametrize("with_tools", [False, True], ids=["live", "buffered"])
def test_every_responses_sse_path_sets_transport_headers(
    tmp_path,
    with_tools: bool,
) -> None:
    payload = {"model": "m", "input": "hello", "stream": True}
    if with_tools:
        payload["tools"] = [_tool()]

    with TestClient(_app(tmp_path)) as http:
        response = http.post("/v1/responses", json=payload)

    assert response.status_code == 200
    _assert_sse_response_headers(response)
    assert "event: response.completed" in response.text


def test_responses_stream_escapes_unicode_line_separators_on_the_wire(tmp_path):
    separators = "before\u0085middle\u2028after\u2029"
    backend = MockBackend({"unicode": separators})
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={"model": "m", "input": "unicode", "stream": True},
        )

    assert response.status_code == 200
    assert not any(character in response.text for character in "\u0085\u2028\u2029")
    assert all(escape in response.text for escape in ("\\u0085", "\\u2028", "\\u2029"))
    deltas = [
        event["delta"]
        for event in _sse_events(response.text)
        if event["type"] == "response.output_text.delta"
    ]
    assert "".join(deltas) == separators


_WEBSOCKET_UPGRADE = {
    "Connection": "Upgrade",
    "Upgrade": "websocket",
    "Sec-WebSocket-Version": "13",
    "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
}


class _TransitiveWebSocketLibrary(asyncio.Protocol):
    """What uvicorn's ``ws="auto"`` finds once a dependency installs websockets
    or wsproto: a protocol that takes over every upgrade before the app runs."""

    def __init__(self, **_uvicorn_state: object) -> None:
        super().__init__()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport

    def data_received(self, data: bytes) -> None:
        self.transport.write(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")
        self.transport.close()


@contextmanager
def _kairyu_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    """Serve what ``kairyu serve`` launches, on an ephemeral port; yield the port."""

    config = tmp_path / "deploy.yaml"
    config.write_text("engines: {m: {backend: mock}}\n", encoding="utf-8")
    launch: dict[str, object] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **options: launch.update(app=app, **options))
    # Keep pytest's logging: serve would replace the root handlers.
    monkeypatch.setattr("kairyu.entrypoints.server.middleware.configure_json_logging", lambda: None)
    cli.main(["serve", str(config)])
    with serve(uvicorn.Config(**{**launch, "host": LOOPBACK_HOST, "port": 0})) as port:
        yield port


@pytest.mark.parametrize(
    "headers", [_WEBSOCKET_UPGRADE, {}], ids=["websocket-upgrade", "plain-get"]
)
def test_websocket_upgrade_get_returns_426(tmp_path, monkeypatch, headers):
    # Codex (built-in openai provider, the Harbor shape) tries a WebSocket
    # upgrade first; 426 triggers its immediate, silent HTTPS fallback. Until
    # WebSocket mode exists (m20 D-e) `kairyu serve` must give the upgrade to
    # the app as plain HTTP even when a dependency installs a WebSocket
    # library: a websocket scope would skip AuthMiddleware and tenancy.
    monkeypatch.setattr(uvicorn_ws_auto, "AutoWebSocketsProtocol", _TransitiveWebSocketLibrary)
    with _kairyu_serve(tmp_path, monkeypatch) as port:
        connection = HTTPConnection(LOOPBACK_HOST, port, timeout=SERVER_TIMEOUT_S)
        try:
            connection.request("GET", "/v1/responses", headers=headers)
            response = connection.getresponse()
            status, body = response.status, response.read()
        finally:
            connection.close()
    assert status == 426
    assert json.loads(body)["error"]["code"] == "upgrade_required"


def test_stream_failure_emits_error_and_failed_without_storing(tmp_path):
    class FailingStreamBackend(MockBackend):
        async def stream(self, request):
            partial = await self.generate(request)
            yield replace(partial, finished=False)
            raise RuntimeError("secret upstream endpoint")

    with TestClient(_app(tmp_path, FailingStreamBackend())) as http:
        response = http.post(
            "/v1/responses",
            json={"model": "m", "input": "hello", "stream": True},
        )
        events = _sse_events(response.text)
        response_id = events[0]["response"]["id"]
        lookup = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "again",
                "previous_response_id": response_id,
            },
        )
    assert [event["type"] for event in events[-2:]] == ["error", "response.failed"]
    failed = events[-1]["response"]
    assert failed["status"] == "failed"
    assert failed["output"][0]["status"] == "incomplete"
    assert failed["output"][0]["content"][0]["text"]
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    try:
        totals = ledger.totals()["default"]
    finally:
        ledger.close()
    assert failed["usage"]["input_tokens"] == totals["prompt_tokens"]
    assert failed["usage"]["output_tokens"] == totals["completion_tokens"]
    assert "secret upstream endpoint" not in response.text
    assert lookup.status_code == 400
