"""Opt-in wire capture for refactor-only work packages (m20 DoD #9).

Run a suite with ``KAIRYU_WIRE_CAPTURE=<dir>`` at the base commit and again at
the head commit, then compare the two captures::

    python -m tests.contracts.wire_capture diff <base-dir> <head-dir>

Each test writes one JSON-lines file holding every HTTP exchange it made, on
every route, in completion order. ``diff`` masks only values that differ
between two runs of the same commit (generated ids, clocks and per-request
headers) and requires every other byte to match.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.contracts.openai_contract import Exchange

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_LIMIT = 150
_VOLATILE_HEADERS = frozenset({"date", "x-request-id", "content-length"})
# Routes whose body is a live process snapshot (timings, clocks).
_VOLATILE_BODY_PATHS = frozenset({"/metrics"})
# Values that change between two runs of the same commit, and their stand-ins.
_VOLATILE_VALUES = (
    (
        re.compile(
            r"\b(?:resp|msg|fc|rs|cmp|cmpl|call|chatcmpl|batch|file|req|item)[-_]"
            r"[0-9A-Za-z]{8,}"
        ),
        "<id>",
    ),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<id>"),
    (
        re.compile(r'"(created_at|created|completed_at|expires_at|timestamp)":\s*[0-9.]+'),
        r'"\1": "<time>"',
    ),
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"), "<time>"),
    (re.compile(r"\bkcp1\.[A-Za-z0-9_\-=.]+"), "<sealed>"),
    (re.compile(r'((?:duration|latency|elapsed)_(?:ns|ms|s)"?\s*[=:]\s*)[0-9.]+'), r"\1<dur>"),
)


def _text_or_b64(data: bytes) -> dict[str, str]:
    try:
        return {"text": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"b64": base64.b64encode(data).decode("ascii")}


@dataclass(frozen=True)
class WireRecord:
    """One completed HTTP exchange as seen on the wire."""

    method: str
    path: str
    query: str
    request_body: bytes
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes

    @classmethod
    def from_asgi(
        cls,
        scope: Mapping[str, Any],
        request_body: bytes,
        start: Mapping[str, Any],
        body: bytes,
    ) -> WireRecord:
        return cls(
            method=scope["method"],
            path=scope["path"],
            query=scope.get("query_string", b"").decode("latin-1"),
            request_body=request_body,
            status=start["status"],
            headers=tuple(
                (name.decode("latin-1").lower(), value.decode("latin-1"))
                for name, value in start.get("headers", ())
            ),
            body=body,
        )

    def exchange(self) -> Exchange:
        content_type = next((value for name, value in self.headers if name == "content-type"), "")
        return Exchange(self.method, self.path, self.status, content_type, self.body)

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "query": self.query,
            "request_body": _text_or_b64(self.request_body),
            "status": self.status,
            "headers": [list(header) for header in self.headers],
            "body": _text_or_b64(self.body),
        }


def capture_file_name(test_id: str) -> str:
    digest = hashlib.sha1(test_id.encode("utf-8")).hexdigest()[:12]
    return f"{_SAFE.sub('_', test_id)[:_NAME_LIMIT]}-{digest}.jsonl"


class WireCapture:
    """Writes one capture file per test into ``directory``."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, test_id: str, records: Iterable[WireRecord]) -> None:
        lines = [
            json.dumps({"test": test_id, **record.to_json()}, sort_keys=True) for record in records
        ]
        if lines:
            target = self.directory / capture_file_name(test_id)
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mask(text: str) -> str:
    for pattern, replacement in _VOLATILE_VALUES:
        text = pattern.sub(replacement, text)
    return text


def normalize(line: str) -> dict[str, Any]:
    record = json.loads(line)
    record["headers"] = [
        [name, "<volatile>" if name in _VOLATILE_HEADERS else _mask(value)]
        for name, value in record["headers"]
    ]
    for field in ("request_body", "body"):
        if "text" in record[field]:
            record[field] = {"text": _mask(record[field]["text"])}
    if record["path"] in _VOLATILE_BODY_PATHS:
        record["body"] = {"text": "<volatile>"}
    return record


def _load(directory: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        path.name: [normalize(line) for line in path.read_text("utf-8").splitlines()]
        for path in sorted(directory.glob("*.jsonl"))
    }


def diff(base: Path, head: Path) -> list[str]:
    """Return one line per test whose normalized wire output differs."""

    before, after = _load(base), _load(head)
    problems = []
    for name in sorted(set(before) | set(after)):
        if name not in after or name not in before:
            side = "head" if name not in after else "base"
            problems.append(f"{name}: missing at {side}")
            continue
        if before[name] != after[name]:
            pairs = zip(before[name], after[name], strict=False)
            index = next((i for i, (a, b) in enumerate(pairs) if a != b), None)
            where = f"record {index}" if index is not None else "record count"
            problems.append(f"{name}: differs at {where}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("diff", help="compare base and head captures")
    compare.add_argument("base", type=Path)
    compare.add_argument("head", type=Path)
    args = parser.parse_args(argv)
    problems = diff(args.base, args.head)
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{len(problems)} test(s) with differing wire output")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
