#!/usr/bin/env python3
"""Recording reverse proxy for Codex request fixtures (M20 WP-02, D1).

``Codex -> record_proxy -> Kairyu``. Every request is forwarded and every
upstream response is streamed back unchanged. Requests on the routes Codex
calls (``POST .../responses``, ``POST .../responses/compact``,
``GET .../models``) are also written to ``--out`` as normalized captures, one
JSON file per request, numbered in arrival order.

Normalization makes a capture reproducible and safe to commit:

- credentials are never recorded: only allowlisted headers are kept, any
  credential-like header is dropped even when allowlisted, and the value of a
  credential-like query parameter (a provider's ``query_params``) becomes
  ``{{CREDENTIAL}}``;
- volatile values become numbered placeholders, consistently within one
  capture: UUIDs (session, thread, installation, window ids), server-issued
  item and call ids (``fc_…``, ``call_…``), ``prompt_cache_key``, and the ids
  inside the JSON-encoded ``x-codex-turn-metadata``; ``create_time`` becomes 0;
- environment-specific text becomes placeholders: the paths given with
  ``--redact`` and the home directory, the user name as a path segment, the date,
  timezone and shell of ``<environment_context>``, other timestamps, and
  version-specific prose -- Codex's base ``instructions``, descriptions over
  240 characters and input text over 4096 characters -- which is replaced by
  its digest;
- JSON object keys are sorted; arrays keep their order.

Recording procedure (D1; repeat within 7 days of a new Codex stable):

1. start a Kairyu server; any server answers the single-turn shapes. The
   multi-turn shapes (tool loop, namespace loop, view_image, compaction) need a
   server scripted over ``tests/support/scenario_backend.py``: WP-05's
   ``scripts/codex_gate`` scenarios launcher provides it; until it lands those
   fixtures keep their recording at the tag they name;
2. ``python -m scripts.codex_gate.record_proxy serve --upstream URL --port 8010
   --out DIR --codex-version 0.160.0 --provider-shape custom-responses
   --scenario default-turn --redact /work/dir={{CWD}}``;
3. run Codex with an isolated ``CODEX_HOME`` and ``HOME``, a dummy API key and
   a provider whose base URL is ``http://127.0.0.1:8010/v1``;
4. ``python -m scripts.codex_gate.record_proxy promote DIR/default-turn-01.json
   tests/fixtures/codex/rust-v0.160.0/default-turn.json`` replaces the
   fixture's ``provenance`` and ``request`` and keeps its authored keys; onto a
   derived fixture it re-applies ``provenance.derivation`` (``derivation.py``)
   to the new capture, whose scenario is the one ``derived_from`` names.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import json
import os
import re
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from scripts.codex_gate.derivation import DerivationError, apply_derivation

RECORDER = "scripts/codex_gate/record_proxy.py"
PROVIDER_SHAPES = ("custom-responses", "openai-base-url", "oss-lmstudio", "oss-ollama")
RECORDED_SUFFIXES = ("/responses", "/responses/compact", "/models")
# Authored fixture keys that ``promote`` keeps; it replaces everything else.
AUTHORED_KEYS = ("id", "gap_ids", "contract", "replay")
FIXTURE_KEY_ORDER = ("id", "gap_ids", "contract", "provenance", "request", "replay")
# Authored provenance of a derived fixture, kept and re-applied by ``promote``.
DERIVED_KEYS = ("edit", "codex_rs", "derivation")
RECORDED_HEADERS = frozenset(
    {
        "accept",
        "content-encoding",
        "content-type",
        "openai-beta",
        "originator",
        "session-id",
        "thread-id",
        "user-agent",
        "version",
        "x-client-request-id",
        "x-codex-beta-features",
        "x-codex-parent-thread-id",
        "x-codex-turn-metadata",
        "x-codex-turn-state",
        "x-codex-window-id",
        "x-openai-internal-codex-responses-lite",
        "x-openai-memgen-request",
        "x-openai-subagent",
    }
)
_CREDENTIAL_NAME = re.compile(
    r"auth|cookie|token|secret|key|attestation|account|organization|project|fedramp",
    re.IGNORECASE,
)
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_SERVER_ID = re.compile(r"\b(resp|msg|fc|fco|rs|cmp|call|ctc|at)_([A-Za-z0-9]{12,})\b")
_ENV_TAGS = {"current_date": "{{DATE}}", "timezone": "{{TIMEZONE}}", "shell": "{{SHELL}}"}
_ENV_TAG = re.compile(r"<(current_date|timezone|shell)>[^<]*</\1>")
_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z| UTC|[+-]\d{2}:?\d{2})?"
)
_USER_AGENT = re.compile(r"^(codex_[A-Za-z_]+/[0-9][^ ]*) .*$")
_ID_KEYS = frozenset({"prompt_cache_key"})
_TIME_KEY = re.compile(r"^(create_time|created_at)$|_unix_ms$|_start_ms$|_at_ms$")
_JSON_STRING_KEYS = frozenset({"x-codex-turn-metadata"})
# Descriptions and input text longer than these become a digest: the text is
# version-specific prose (tool docs, base instructions moved into a developer
# message by responses-lite), not part of the wire contract.
DESCRIPTION_LIMIT = 240
INPUT_TEXT_LIMIT = 4096


class Normalizer:
    """Single-owner placeholder table for one capture (consistent numbering)."""

    def __init__(self, redactions: Sequence[tuple[str, str]], user: str = "") -> None:
        # Longest first, so a working directory under HOME keeps its own name.
        self._redactions = tuple(sorted(redactions, key=lambda pair: -len(pair[0])))
        # The user name only as a path segment: a bare word ("user") is content.
        self._user = re.compile(rf"(?<=[/\\]){re.escape(user)}(?=[/\\]|$)") if user else None
        self._ids: dict[str, str] = {}
        self.applied: set[str] = set()

    def _volatile(self, value: str) -> str:
        if value not in self._ids:
            self._ids[value] = f"{{{{ID_{len(self._ids) + 1}}}}}"
        self.applied.add("{{ID_n}}")
        return self._ids[value]

    def text(self, value: str) -> str:
        for literal, placeholder in self._redactions:
            if literal and literal in value:
                value = value.replace(literal, placeholder)
                self.applied.add(placeholder)
        if self._user is not None and self._user.search(value):
            value = self._user.sub("{{USER}}", value)
            self.applied.add("{{USER}}")
        value = _UUID.sub(lambda match: self._volatile(match.group(0).lower()), value)
        value = _SERVER_ID.sub(
            lambda match: f"{match.group(1)}_{self._volatile(match.group(0))}", value
        )
        if _TIMESTAMP.search(value):
            value = _TIMESTAMP.sub("{{TIMESTAMP}}", value)
            self.applied.add("{{TIMESTAMP}}")
        return _ENV_TAG.sub(self._env_tag, value)

    def _env_tag(self, match: re.Match[str]) -> str:
        placeholder = _ENV_TAGS[match.group(1)]
        self.applied.add(placeholder)
        return f"<{match.group(1)}>{placeholder}</{match.group(1)}>"

    def digest(self, value: str, label: str) -> str:
        redacted = self.text(value)
        digest = hashlib.sha256(redacted.encode()).hexdigest()[:16]
        self.applied.add(f"{{{{{label}}}}}")
        return f"{{{{{label} sha256:{digest} chars:{len(redacted)}}}}}"

    def value(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {name: self.value(value[name], name) for name in sorted(value)}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if key is not None and _TIME_KEY.search(key) and isinstance(value, (int, float)):
            self.applied.add("{{TIME}}=0")
            return 0
        if not isinstance(value, str):
            return value
        if key in _ID_KEYS:
            return self._volatile(value)
        if key in _JSON_STRING_KEYS:
            return self._json_string(value)
        if key == "description" and len(value) > DESCRIPTION_LIMIT:
            return self.digest(value, "DESCRIPTION")
        if key == "text" and len(value) > INPUT_TEXT_LIMIT:
            return self.digest(value, "TEXT")
        return self.text(value)

    def _json_string(self, value: str) -> str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return self.text(value)
        return json.dumps(self.value(parsed), sort_keys=True, separators=(",", ":"))

    def body(self, body: Any) -> Any:
        if isinstance(body, Mapping) and isinstance(body.get("instructions"), str):
            instructions = body["instructions"]
            rest = self.value({k: v for k, v in body.items() if k != "instructions"})
            if instructions:
                rest["instructions"] = self.digest(instructions, "CODEX_INSTRUCTIONS")
            else:
                rest["instructions"] = instructions
            return {name: rest[name] for name in sorted(rest)}
        return self.value(body)

    def headers(self, headers: Sequence[tuple[str, str]]) -> dict[str, str]:
        recorded: dict[str, str] = {}
        for raw_name, raw_value in headers:
            name = raw_name.lower()
            if name not in RECORDED_HEADERS or _CREDENTIAL_NAME.search(name):
                continue
            if name == "user-agent":
                recorded[name] = _USER_AGENT.sub(r"\1 ({{PLATFORM}})", raw_value)
            else:
                recorded[name] = self.value(raw_value, name)
        return {name: recorded[name] for name in sorted(recorded)}

    def query(self, query: str) -> str:
        """Normalize a query string; a credential-like parameter keeps only its name."""

        pairs = []
        for name, value in parse_qsl(query, keep_blank_values=True):
            if _CREDENTIAL_NAME.search(name):
                self.applied.add("{{CREDENTIAL}}")
                pairs.append(f"{self.text(name)}={{{{CREDENTIAL}}}}")
            else:
                pairs.append(f"{self.text(name)}={self.text(value)}")
        return "&".join(pairs)


@dataclass(frozen=True)
class CaptureContext:
    codex_version: str
    provider_shape: str
    scenario: str
    invocation: str
    redactions: tuple[tuple[str, str], ...]
    user: str = ""

    @property
    def codex_tag(self) -> str:
        return f"rust-v{self.codex_version}"


def is_recorded(path: str) -> bool:
    return path.endswith(RECORDED_SUFFIXES)


def build_capture(
    context: CaptureContext,
    *,
    sequence: int,
    method: str,
    path: str,
    query: str,
    headers: Sequence[tuple[str, str]],
    body: bytes,
    observed_status: int,
) -> dict[str, Any]:
    """Return the normalized capture of one request (pure; no I/O)."""

    normalizer = Normalizer(context.redactions, context.user)
    if body:
        try:
            payload: Any = normalizer.body(json.loads(body))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {"undecoded_bytes": len(body)}
    else:
        payload = None
    request = {
        "method": method,
        "path": path,
        "query": normalizer.query(query),
        "headers": normalizer.headers(headers),
        "body": payload,
    }
    provenance = {
        "source": "recorded",
        "recorder": RECORDER,
        "codex_version": context.codex_version,
        "codex_tag": context.codex_tag,
        "provider_shape": context.provider_shape,
        "scenario": context.scenario,
        "invocation": Normalizer(context.redactions, context.user).text(context.invocation),
        "sequence": sequence,
        "observed_status": observed_status,
        "placeholders": sorted(normalizer.applied),
    }
    return {"provenance": provenance, "request": request}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def promote(capture_path: Path, fixture_path: Path) -> dict[str, Any]:
    """Merge a capture into a fixture, keeping the fixture's authored keys.

    A derived fixture keeps its ``provenance.derivation`` (see
    ``derivation.py``), which is re-applied to the new capture.
    """

    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    existing = json.loads(fixture_path.read_text(encoding="utf-8")) if fixture_path.exists() else {}
    merged = {key: existing[key] for key in AUTHORED_KEYS if key in existing}
    merged.setdefault("id", fixture_path.stem)
    merged.update(_promoted(capture, existing.get("provenance", {}), fixture_path))
    ordered = {key: merged[key] for key in FIXTURE_KEY_ORDER if key in merged}
    write_json(fixture_path, ordered)
    return ordered


def _promoted(
    capture: Mapping[str, Any], previous: Mapping[str, Any], fixture_path: Path
) -> dict[str, Any]:
    """The fixture's ``provenance`` and ``request`` from a capture."""

    provenance, request = capture["provenance"], capture["request"]
    if previous.get("source") != "derived":
        return {"provenance": provenance, "request": request}
    missing = [key for key in DERIVED_KEYS if key not in previous]
    if missing:
        raise DerivationError(f"{fixture_path}: derived fixture lacks provenance {missing}")
    origin = f"recorded capture {provenance['scenario']} (sequence {provenance['sequence']})"
    return {
        "provenance": {
            **provenance,
            "source": "derived",
            "derived_from": origin,
            **{key: previous[key] for key in DERIVED_KEYS},
        },
        "request": {**request, "body": apply_derivation(request["body"], previous["derivation"])},
    }


def create_proxy_app(upstream: str, out_dir: Path, context: CaptureContext):
    """Starlette app that forwards everything and records the Codex routes."""

    import httpx
    from starlette.applications import Starlette
    from starlette.background import BackgroundTask
    from starlette.requests import Request
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    client = httpx.AsyncClient(base_url=upstream, timeout=None)
    sequence = 0

    async def forward(request: Request) -> StreamingResponse:
        nonlocal sequence
        body = await request.body()
        headers = [
            (name, value)
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP
        ]
        query = request.url.query
        target = request.url.path + (f"?{query}" if query else "")
        upstream_request = client.build_request(
            request.method, target, headers=headers, content=body
        )
        response = await client.send(upstream_request, stream=True)
        if is_recorded(request.url.path):
            sequence += 1
            capture = build_capture(
                context,
                sequence=sequence,
                method=request.method,
                path=request.url.path,
                query=query,
                headers=headers,
                body=body,
                observed_status=response.status_code,
            )
            write_json(out_dir / f"{context.scenario}-{sequence:02d}.json", capture)
        response_headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() not in _HOP_BY_HOP
        }
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=response_headers,
            background=BackgroundTask(response.aclose),
        )

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    return Starlette(routes=[Route("/{path:path}", forward, methods=methods)], lifespan=lifespan)


def _redactions(pairs: Sequence[str]) -> tuple[tuple[str, str], ...]:
    parsed = [(str(Path.home()), "{{HOME}}")]
    for pair in pairs:
        literal, separator, placeholder = pair.partition("=")
        if not separator or not literal or not placeholder:
            raise SystemExit(f"--redact expects LITERAL=PLACEHOLDER, got {pair!r}")
        parsed.append((literal, placeholder))
        resolved = os.path.realpath(literal)
        if resolved != literal:
            parsed.append((resolved, placeholder))
    return tuple(parsed)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the recording proxy")
    serve.add_argument("--upstream", required=True, help="Kairyu base URL (no /v1)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--out", type=Path, required=True)
    serve.add_argument("--codex-version", required=True, help="e.g. 0.160.0")
    serve.add_argument("--provider-shape", choices=PROVIDER_SHAPES, required=True)
    serve.add_argument("--scenario", required=True)
    serve.add_argument(
        "--invocation", default="", help="the Codex command line and config, for provenance"
    )
    serve.add_argument("--redact", action="append", default=[], metavar="LITERAL=PLACEHOLDER")
    promote_cmd = commands.add_parser("promote", help="merge a capture into a fixture")
    promote_cmd.add_argument("capture", type=Path)
    promote_cmd.add_argument("fixture", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "promote":
        promote(args.capture, args.fixture)
        return 0
    import uvicorn

    context = CaptureContext(
        codex_version=args.codex_version,
        provider_shape=args.provider_shape,
        scenario=args.scenario,
        invocation=args.invocation,
        redactions=_redactions(args.redact),
        user=getpass.getuser(),
    )
    app = create_proxy_app(args.upstream, args.out, context)
    # ws="none": a Codex WebSocket probe reaches Kairyu as a plain GET (426).
    uvicorn.run(app, host=args.host, port=args.port, ws="none", log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
