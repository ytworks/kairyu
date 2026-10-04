"""Opt-in wire capture for refactor-only work packages (m20 DoD #9).

Run a suite with ``KAIRYU_WIRE_CAPTURE=<dir>`` at the base commit and again at
the head commit, then compare the two captures::

    python -m tests.contracts.wire_capture diff <base-dir> <head-dir>

Each test writes one JSON-lines file holding every HTTP exchange it made, on
every route. ``diff`` requires every byte to match except what differs between
two runs of the same commit:

- the random part of a generated id (its prefix, such as ``resp_``, is kept),
  UUIDs and bare 32/64-digit hex runs (multipart boundaries, ``uuid4().hex``,
  digests over volatile data). Each distinct value is numbered per test in
  order of first appearance, so a swapped id or a broken cross-reference
  (``previous_response_id``, ``call_id``) still differs;
- clocks, durations, per-request headers and the ``/metrics`` body;
- the order of a test's exchanges (concurrent requests finish in any order).

Request bodies are test input: only their first ``REQUEST_BODY_LIMIT`` bytes
and their length are kept, so capture leaves memory-bound tests unchanged.
``NONDETERMINISTIC_TESTS`` names the tests whose output varies between runs of
one commit; ``diff`` skips them and says so. ``--key test`` pairs captures by
the node id without its module path, for a refactor that moves tests between
modules (WP-06).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tests.contracts.openai_contract import Exchange

REQUEST_BODY_LIMIT = 8 * 1024
_LOGPROBS = "tests/server/test_logprobs_completions_n.py::"
_STRUCTURED = "tests/server/test_structured_output_api.py::"
# Node id -> why its wire output differs between two runs of one commit.
NONDETERMINISTIC_TESTS: Mapping[str, str] = MappingProxyType(
    {
        f"{_LOGPROBS}test_n_greater_than_one_distinct_and_indexed": "unseeded torch sampling",
        f"{_LOGPROBS}test_n_streaming_interleaves_indices": "unseeded torch sampling",
        "tests/server/test_openai_api.py::test_native_chat_stage_trace_is_opt_in_and_terminal": (
            "native engine stream chunking depends on scheduling"
        ),
        "tests/server/test_usage_truth.py::test_include_usage_final_chunk_contract": (
            "native engine stream chunking depends on scheduling"
        ),
        "tests/server/test_usage_truth.py::test_usage_key_omitted_without_stream_options": (
            "native engine stream chunking depends on scheduling"
        ),
        f"{_LOGPROBS}test_streaming_logprobs_on_chunk_choice": (
            "native engine stream chunking depends on scheduling"
        ),
        "tests/server/test_batches.py::test_batch_lifecycle_end_to_end": (
            "status poll count depends on worker timing"
        ),
        "tests/server/test_serve_builder.py::"
        "test_builder_wires_async_request_routes_worker_and_store_lifespan": (
            "status poll count depends on worker timing"
        ),
        f"{_STRUCTURED}test_invalid_strict_tool_schema_is_400_and_engine_stays_healthy": (
            "unseeded grammar-constrained sampling"
        ),
        f"{_STRUCTURED}test_response_format_text_passes_through": "unseeded sampling",
        "tests/unit/test_runner_startup_admission_runtime.py::"
        "test_runtime_assembles_app_checks_dependencies_and_closes[200-True]": (
            "base64 JSONPatch embeds a random binding id"
        ),
    }
)
KEYS: Mapping[str, Callable[[str], str]] = MappingProxyType(
    {"node": lambda test_id: test_id, "test": lambda test_id: test_id.split("::", 1)[-1]}
)
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_LIMIT = 150
_VOLATILE_HEADERS = frozenset({"date", "x-request-id", "content-length"})
# Routes whose body is a live process snapshot (timings, clocks).
_VOLATILE_BODY_PATHS = frozenset({"/metrics"})
_ID_PREFIXES = (
    "chatcmpl", "cmpl", "cmp", "resp", "msg", "fc", "rs", "call", "toolu", "batch_req",
    "batch", "file", "req", "item", "orch", "kinv", "direct", "http",
)  # fmt: skip
# A generated id is a known prefix, "-" or "_", and a random suffix with a digit.
_GENERATED_ID = re.compile(rf"\b({'|'.join(_ID_PREFIXES)})([-_])([0-9A-Za-z]{{8,}})")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_HEX_RUN = re.compile(r"\b(?:[0-9a-f]{64}|[0-9a-f]{32})\b")
# Clocks, durations and sealed tokens, and their stand-ins (not numbered).
_VOLATILE_VALUES = (
    (
        re.compile(r'"(\w+_at(?:_ns|_ms)?|created|timestamp)":\s*[0-9.]+'),
        r'"\1": "<time>"',
    ),
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"), "<time>"),
    (re.compile(r"\[\d{2}:\d{2}:\d{2}\]"), "[<clock>]"),
    (re.compile(r"\bk(?:cp1|st2)\.[A-Za-z0-9_\-=.]+"), "<sealed>"),
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
    request_head: bytes  # at most REQUEST_BODY_LIMIT bytes
    request_size: int
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes

    @classmethod
    def from_asgi(
        cls,
        scope: Mapping[str, Any],
        request: tuple[bytes, int],
        start: Mapping[str, Any],
        body: bytes,
    ) -> WireRecord:
        request_head, request_size = request
        return cls(
            method=scope["method"],
            path=scope["path"],
            query=scope.get("query_string", b"").decode("latin-1"),
            request_head=request_head[:REQUEST_BODY_LIMIT],
            request_size=request_size,
            status=start["status"],
            headers=tuple(
                (name.decode("latin-1").lower(), value.decode("latin-1"))
                for name, value in start.get("headers", ())
            ),
            body=body,
        )

    def exchange(self) -> Exchange:
        content_type = next((value for name, value in self.headers if name == "content-type"), "")
        return Exchange(
            self.method, self.path, self.status, content_type, self.body, query=self.query
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "query": self.query,
            # A cut can split a UTF-8 sequence; keep the prefix maskable as text.
            "request_body": {"text": self.request_head.decode("utf-8", "backslashreplace")},
            "request_size": self.request_size,
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


Label = Callable[[str], str]


def _unnumbered(_token: str) -> str:
    return ""


def _numbering() -> Label:
    """A label that numbers distinct tokens in order of first appearance (one capture)."""

    numbers: dict[str, int] = {}

    def label(token: str) -> str:
        return f":{numbers.setdefault(token, len(numbers) + 1)}"

    return label


def _mask(text: str, label: Label) -> str:
    def generated_id(match: re.Match[str]) -> str:
        prefix, separator, suffix = match.groups()
        if not any(char.isdigit() for char in suffix):  # a word such as item_reference
            return match.group(0)
        return f"{prefix}{separator}<id{label(suffix)}>"

    text = _GENERATED_ID.sub(generated_id, text)
    text = _UUID.sub(lambda match: f"<uuid{label(match.group(0))}>", text)
    text = _HEX_RUN.sub(lambda match: f"<hex{label(match.group(0))}>", text)
    for pattern, replacement in _VOLATILE_VALUES:
        text = pattern.sub(replacement, text)
    return text


def _normalize_record(record: Mapping[str, Any], label: Label) -> dict[str, Any]:
    # Masking order (path, query, headers, request, body) fixes the numbering order.
    normalized = {name: value for name, value in record.items() if name != "test"}
    normalized["path"] = _mask(record["path"], label)
    normalized["query"] = _mask(record["query"], label)
    normalized["headers"] = [
        [name, "<volatile>" if name in _VOLATILE_HEADERS else _mask(value, label)]
        for name, value in record["headers"]
    ]
    for field in ("request_body", "body"):
        if "text" in record[field]:
            normalized[field] = {"text": _mask(record[field]["text"], label)}
    if record["path"] in _VOLATILE_BODY_PATHS:
        normalized["body"] = {"text": "<volatile>"}
    return normalized


def normalize_capture(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Mask one test's capture and order its exchanges independently of completion order."""

    records = [json.loads(line) for line in lines]
    shapes = [
        json.dumps(_normalize_record(record, _unnumbered), sort_keys=True) for record in records
    ]
    label = _numbering()
    return [
        _normalize_record(records[index], label)
        for index in sorted(range(len(records)), key=shapes.__getitem__)
    ]


@dataclass(frozen=True)
class _Capture:
    test_id: str
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DiffReport:
    compared: int
    skipped: tuple[str, ...]
    problems: tuple[str, ...]


def _load(directory: Path) -> tuple[_Capture, ...]:
    captures = []
    for path in sorted(directory.glob("*.jsonl")):
        lines = path.read_text("utf-8").splitlines()
        captures.append(_Capture(json.loads(lines[0])["test"], tuple(normalize_capture(lines))))
    return tuple(captures)


def _index(
    captures: Iterable[_Capture], key: Callable[[str], str], side: str
) -> tuple[dict[str, _Capture], list[str]]:
    grouped: dict[str, list[_Capture]] = {}
    for capture in captures:
        grouped.setdefault(key(capture.test_id), []).append(capture)
    problems = [
        f"{name}: {len(group)} captures share this key at {side}"
        for name, group in grouped.items()
        if len(group) > 1
    ]
    return {name: group[0] for name, group in grouped.items() if len(group) == 1}, problems


def _first_difference(old: tuple[Any, ...], new: tuple[Any, ...]) -> str:
    index = next((i for i, (a, b) in enumerate(zip(old, new, strict=False)) if a != b), None)
    return "record count" if index is None else f"record {index}"


def diff(base: Path, head: Path, key: str = "node") -> DiffReport:
    """Compare two capture directories; one problem line per differing test."""

    key_of = KEYS[key]
    loaded = {"base": _load(base), "head": _load(head)}
    skipped_keys = {key_of(test_id) for test_id in NONDETERMINISTIC_TESTS}
    skipped = {
        capture.test_id
        for captures in loaded.values()
        for capture in captures
        if key_of(capture.test_id) in skipped_keys
    }
    (before, base_problems), (after, head_problems) = (
        _index((c for c in captures if c.test_id not in skipped), key_of, side)
        for side, captures in loaded.items()
    )
    problems = [*base_problems, *head_problems]
    for name in sorted(set(before) | set(after)):
        if name not in after or name not in before:
            problems.append(f"{name}: missing at {'head' if name not in after else 'base'}")
        elif before[name].records != after[name].records:
            where = _first_difference(before[name].records, after[name].records)
            problems.append(f"{name}: differs at {where}")
    compared = len(set(before) & set(after))
    return DiffReport(compared, tuple(sorted(skipped)), tuple(problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("diff", help="compare base and head captures")
    compare.add_argument("base", type=Path)
    compare.add_argument("head", type=Path)
    compare.add_argument(
        "--key",
        choices=sorted(KEYS),
        default="node",
        help="pair captures by full node id (default) or by node id without the module path",
    )
    args = parser.parse_args(argv)
    report = diff(args.base, args.head, args.key)
    for problem in report.problems:
        print(problem, file=sys.stderr)
    for test_id in report.skipped:
        print(f"skipped as nondeterministic: {test_id}", file=sys.stderr)
    print(
        f"{report.compared} test(s) compared, {len(report.skipped)} skipped as "
        f"nondeterministic; {len(report.problems)} with differing wire output"
    )
    return 1 if report.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
