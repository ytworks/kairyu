"""Per-exchange log of the recording proxy (M20 WP-05, D1).

The Codex matrix (``run_matrix.py``) asserts on what crossed the wire, not on
what Codex chose to report: every HTTP status, how many data events of each
type a stream carried (a repeated ``response.in_progress`` is a heartbeat), the
in-band ``response.failed`` code, the error code of a non-streamed error body,
and the request facts a scenario checks (model, tool count, a live
``web_search`` declaration). One JSON object per exchange, appended to a JSONL
file when the response body ends.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bytes kept of a non-SSE body, enough for any error envelope.
ERROR_BODY_LIMIT = 64 * 1024
_FAILED = "response.failed"


def request_facts(body: bytes) -> dict[str, Any]:
    """Model, tool count, live ``web_search`` and ``compaction_trigger`` of a request."""

    try:
        payload = json.loads(body) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"json": False}
    if not isinstance(payload, Mapping):
        return {}
    tools = payload.get("tools")
    tool_list = tools if isinstance(tools, list) else []
    items = payload.get("input")
    item_list = items if isinstance(items, list) else []
    return {
        "model": payload.get("model"),
        "tools": len(tool_list) if isinstance(tools, list) else None,
        "live_web_search": any(
            isinstance(tool, Mapping)
            and tool.get("type") == "web_search"
            and tool.get("external_web_access") is True
            for tool in tool_list
        ),
        "compaction_trigger": any(
            isinstance(item, Mapping) and item.get("type") == "compaction_trigger"
            for item in item_list
        ),
    }


@dataclass
class StreamObserver:
    """Single-owner accumulator for one response body (SSE or JSON)."""

    sse: bool
    events: Counter = field(default_factory=Counter)
    failed_code: str | None = None
    _pending: bytes = b""
    _body: bytes = b""

    def feed(self, chunk: bytes) -> None:
        if not self.sse:
            room = ERROR_BODY_LIMIT - len(self._body)
            if room > 0:
                self._body += chunk[:room]
            return
        lines = (self._pending + chunk).split(b"\n")
        self._pending = lines.pop()
        for line in lines:
            self._line(line.rstrip(b"\r"))

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[len(b"data:") :].strip()
        if not data or data == b"[DONE]":
            return
        try:
            event = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.events["<unparsed>"] += 1
            return
        kind = event.get("type") if isinstance(event, Mapping) else None
        self.events[str(kind)] += 1
        if kind == _FAILED:
            error = (event.get("response") or {}).get("error") or {}
            self.failed_code = error.get("code") if isinstance(error, Mapping) else None

    def error_code(self) -> str | None:
        """The ``error.code`` of a non-streamed JSON error body, if any."""

        try:
            payload = json.loads(self._body) if self._body else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        error = payload.get("error") if isinstance(payload, Mapping) else None
        if isinstance(error, Mapping):
            return error.get("code")
        return None

    def summary(self) -> dict[str, Any]:
        if self._pending:
            self._line(self._pending.rstrip(b"\r"))
            self._pending = b""
        return {
            "events": dict(sorted(self.events.items())),
            "failed_code": self.failed_code,
            "error_code": None if self.sse else self.error_code(),
        }


def append_exchange(path: Path, entry: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_exchanges(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
