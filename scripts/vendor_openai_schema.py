#!/usr/bin/env python3
"""Vendor the Responses and Conversations closure of OpenAI's pinned OpenAPI spec.

The M20 schema gate (docs/design/m20-responses-compat.md, D1) validates every
Responses/Conversations body and SSE event emitted in tests against a pinned
snapshot of ``openai/openai-openapi``. This script fetches the spec at a pinned
commit and writes the ``$ref`` closure of the gate's roots, plus the
``/responses*`` and ``/conversations*`` operations, to
``tests/contracts/openai/responses-schema@<sha10>.json``.

Only annotation keywords (``description``, ``title``, ``example(s)`` and the
``x-*`` vendor extensions) are stripped; validation keywords are kept verbatim.
The generation date is an input (``--generated-on`` or ``SOURCE_DATE_EPOCH``)
so that re-running the script on the same pin reproduces the file byte for byte.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

REPO = "openai/openai-openapi"
PINNED_REF = "13fa6e7ab9301b00c03af2a5d2f584e7a9b84391"
SPEC_PATH = "openapi.json"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
COMMIT_API_URL = "https://api.github.com/repos/{repo}/commits/{ref}"
PATH_PREFIXES = ("/responses", "/conversations")
ROOT_SCHEMAS = (
    "Response",
    "ResponseStreamEvent",
    "CreateResponse",
    "ErrorResponse",
    "ResponseItemList",
    "TokenCountsBody",
    "TokenCountsResource",
    "CompactResponseMethodPublicBody",
    "CompactResource",
    "ResponsesClientEvent",
    "ResponsesServerEvent",
    "ResponsesWebSocketStreamEvent",
    "ConversationResource",
    "CreateConversationBody",
    "UpdateConversationBody",
    "DeletedConversationResource",
    "ConversationItem",
    "ConversationItemList",
)
ANNOTATION_KEYWORDS = frozenset({"description", "title", "example", "examples"})
# JSON Schema keywords whose value is a schema, a list of schemas, or a map of them.
SCHEMA_VALUED = frozenset(
    {
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
SCHEMA_LISTS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
SCHEMA_MAPS = frozenset({"$defs", "dependentSchemas", "patternProperties", "properties"})
REF_PATTERN = re.compile(r"^#/components/(?P<section>[A-Za-z]+)/(?P<name>[^/]+)$")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def strip_annotations(schema: Any) -> Any:
    """Return a copy of ``schema`` without annotation keywords at schema positions."""

    if isinstance(schema, list):
        return [strip_annotations(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    stripped: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ANNOTATION_KEYWORDS or key.startswith("x-"):
            continue
        if key in SCHEMA_VALUED or key in SCHEMA_LISTS:
            stripped[key] = strip_annotations(value)
        elif key in SCHEMA_MAPS and isinstance(value, dict):
            stripped[key] = {name: strip_annotations(sub) for name, sub in value.items()}
        else:
            stripped[key] = value
    return stripped


def iter_refs(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from iter_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_refs(item)


def ref_target(ref: str) -> tuple[str, str]:
    match = REF_PATTERN.match(ref)
    if match is None:
        raise ValueError(f"unsupported $ref outside #/components/<section>/<name>: {ref}")
    return match["section"], match["name"]


def closure(spec: dict[str, Any], seeds: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Return ``{section: {name: component}}`` reachable from the ``seeds`` refs."""

    components = spec["components"]
    pending = list(seeds)
    found: dict[str, dict[str, Any]] = {}
    while pending:
        section, name = ref_target(pending.pop())
        if name in found.get(section, {}):
            continue
        component = components[section][name]
        found.setdefault(section, {})[name] = component
        pending.extend(iter_refs(component))
    return found


def _media_schemas(content: dict[str, Any]) -> dict[str, Any]:
    return {
        media_type: strip_annotations(media["schema"])
        for media_type, media in sorted(content.items())
        if "schema" in media
    }


def select_operations(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the gate-relevant view of the Responses and Conversations operations."""

    operations: dict[str, dict[str, Any]] = {}
    for path, item in sorted(spec["paths"].items()):
        if "?" in path or not path.startswith(PATH_PREFIXES):
            continue
        for method, operation in sorted(item.items()):
            if not isinstance(operation, dict) or "responses" not in operation:
                continue
            request = operation.get("requestBody", {}).get("content", {})
            operations.setdefault(path, {})[method] = {
                "operationId": operation.get("operationId"),
                "requestBody": _media_schemas(request),
                "responses": {
                    status: _media_schemas(response.get("content", {}))
                    for status, response in sorted(operation["responses"].items())
                },
            }
    if not operations:
        raise ValueError("the spec has no /responses or /conversations operations")
    return operations


def check_stream_variants(schemas: dict[str, Any]) -> None:
    """Fail unless every stream-event variant declares exactly one ``type`` value.

    The contract validator dispatches each SSE event to its variant by ``type``.
    """

    for ref in iter_refs(schemas["ResponseStreamEvent"]["anyOf"]):
        _, name = ref_target(ref)
        values = schemas[name].get("properties", {}).get("type", {}).get("enum", [])
        if len(values) != 1:
            raise ValueError(f"stream event {name} has no single `type` value: {values}")


def build_document(
    spec: dict[str, Any], *, sha: str, source_url: str, generated_on: str
) -> dict[str, Any]:
    operations = select_operations(spec)
    seeds = [f"#/components/schemas/{name}" for name in ROOT_SCHEMAS]
    seeds.extend(iter_refs(operations))
    components = {
        section: {name: strip_annotations(found[name]) for name in sorted(found)}
        for section, found in sorted(closure(spec, seeds).items())
    }
    check_stream_variants(components["schemas"])
    return {
        "provenance": {
            "repository": REPO,
            "commit": sha,
            "source_url": source_url,
            "openapi": spec["openapi"],
            "info_version": spec["info"]["version"],
            "generated_on": generated_on,
            "generator": "scripts/vendor_openai_schema.py",
            "roots": list(ROOT_SCHEMAS),
            "path_prefixes": list(PATH_PREFIXES),
            "stripped": "annotation keywords (description, title, example(s), x-*)",
        },
        "paths": operations,
        "components": components,
    }


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "kairyu-vendor-schema"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed hosts
        return response.read()


def resolve_sha(ref: str) -> str:
    if FULL_SHA.match(ref):
        return ref
    payload = json.loads(_fetch(COMMIT_API_URL.format(repo=REPO, ref=ref)))
    sha = payload.get("sha", "")
    if not FULL_SHA.match(sha):
        raise ValueError(f"GitHub did not resolve {REPO}@{ref} to a commit: {sha!r}")
    return sha


def generation_date(explicit: str | None) -> str:
    if explicit:
        return dt.date.fromisoformat(explicit).isoformat()
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        return dt.datetime.fromtimestamp(int(epoch), tz=dt.UTC).date().isoformat()
    raise SystemExit("pass --generated-on YYYY-MM-DD or set SOURCE_DATE_EPOCH")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--ref", default=PINNED_REF, help=f"{REPO} commit (default: pin)")
    parser.add_argument("--generated-on", help="ISO date recorded in the provenance")
    parser.add_argument(
        "--spec-file",
        type=Path,
        help="read an already downloaded openapi.json for --ref instead of fetching it",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "contracts" / "openai",
    )
    args = parser.parse_args(argv)
    generated_on = generation_date(args.generated_on)
    sha = resolve_sha(args.ref)
    source_url = RAW_URL.format(repo=REPO, sha=sha, path=SPEC_PATH)
    raw = args.spec_file.read_bytes() if args.spec_file else _fetch(source_url)
    document = build_document(
        json.loads(raw), sha=sha, source_url=source_url, generated_on=generated_on
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / f"responses-schema@{sha[:10]}.json"
    text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_text(text, encoding="utf-8")
    schemas = len(document["components"]["schemas"])
    print(f"wrote {target} ({schemas} schemas, {len(text)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
