#!/usr/bin/env python3
"""Inventory everything Codex can put on the Responses wire (M20 WP-02, R-1).

Walks the codex-rs serde request types at a pinned tag -- ``ResponsesApiRequest``,
the WebSocket ``response.create`` and, where it still exists, the V1
``CompactionInput`` (``codex-api/src/common.rs``); the ``ResponseItem``,
``ContentItem`` and ``FunctionCallOutputPayload`` family
(``protocol/src/models.rs``); ``ToolSpec`` and ``ResponsesApiTool``
(``tools/src``; web search is built in ``core/src/tools/hosted_spec.rs``) --
in step with the vendored OpenAPI closure, and writes
``tests/fixtures/codex/rust-v<ver>/extensions.json``: one entry per field,
item/content/tool kind and enum value, with its ``path@tag:line`` source and its
classification, ``spec`` (the pinned spec declares it at that position) or
``codex-extension``. WP-08a accepts the extensions by this list.

Sources are fetched once with ``gh api`` into ``--cache/<tag>/``. The output
depends only on the tag's sources and the spec file, so a re-run reproduces it.

    python -m scripts.codex_gate.extension_inventory --tag rust-v0.160.0 \\
        --cache /tmp/codex-src
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.codex_gate.rust_serde import (
    Member,
    TypeDef,
    TypeRef,
    parse_type,
    parse_types,
    serialized,
    wire_name,
)

REPO = "openai/codex"
ROOT_DIR = Path(__file__).resolve().parents[2]
SPEC_FILE = ROOT_DIR / "tests/contracts/openai/responses-schema@13fa6e7ab9.json"
FIXTURE_DIR = ROOT_DIR / "tests/fixtures/codex"
SOURCE_FILES = (
    "codex-rs/codex-api/src/common.rs",
    "codex-rs/protocol/src/models.rs",
    "codex-rs/protocol/src/models/configuration_update.rs",
    "codex-rs/protocol/src/models/item_metadata.rs",
    "codex-rs/protocol/src/config_types.rs",
    "codex-rs/protocol/src/openai_models.rs",
    "codex-rs/protocol/src/openai_models/reasoning_effort.rs",
    "codex-rs/protocol/src/turn_input.rs",
    "codex-rs/tools/src/tool_spec.rs",
    "codex-rs/tools/src/responses_api.rs",
    "codex-rs/core/src/tools/hosted_spec.rs",
)
# (codex root type, wire path, spec schema, route); roots absent at a tag are skipped.
ROOTS = (
    ("ResponsesApiRequest", "request", "CreateResponse", "POST /responses"),
    ("ResponsesWsRequest", "websocket", "ResponsesClientEvent", "WebSocket /responses"),
    ("CompactionInput", "compact", "CompactResponseMethodPublicBody", "POST /responses/compact"),
)
# Raw JSON fields whose content is a serialized ToolSpec.
RAW_TOOL_FIELDS = frozenset(
    {
        ("ResponseCreateWsRequest", "tools"),
        ("AdditionalTools", "tools"),
        ("ToolSearchOutput", "tools"),
    }
)
_USE_ALIAS = re.compile(r"^use [\w:]+::(\w+) as (\w+);", re.M)
TOOLS = TypeRef("array", inner=TypeRef("named", "ToolSpec"))
# Variants built outside their type's crate, cited as ``constructed_at``.
CONSTRUCTORS = {("ToolSpec", "WebSearch"): "codex-rs/core/src/tools/hosted_spec.rs"}
NOTE = (
    "Type-level inventory: every field, kind and value the serde request types can "
    "serialize at this tag, classified against the pinned OpenAPI closure at the same "
    "wire position. Builder gating (which fields a given provider or model actually "
    "sends) is recorded by the request fixtures beside this file."
)
STRING = TypeRef("scalar", "String")


@dataclass(frozen=True)
class Override:
    """A type whose wire form a serde derive does not describe."""

    shapes: tuple[TypeRef, ...] = ()
    # (wire value, file, anchor regex): the value exists at a tag iff its anchor does.
    values: tuple[tuple[str, str, str], ...] = ()
    note: str = ""


_MODELS = "codex-rs/protocol/src/openai_models.rs"
OVERRIDES: Mapping[str, Override] = {
    "ResponsesApiTools": Override(shapes=(TOOLS,), note="raw JSON array of ToolSpec"),
    "ResponseItemId": Override(shapes=(STRING,), note="prefixed item id string"),
    "JsonSchema": Override(note="JSON Schema object (tool parameters)"),
    "FunctionCallOutputPayload": Override(
        shapes=(STRING, TypeRef("array", inner=TypeRef("named", "FunctionCallOutputContentItem"))),
        note="custom Serialize: a string or an array of content items",
    ),
    "ReasoningEffort": Override(
        values=(
            *(
                (value, _MODELS, rf'Self::\w+ => "{value}"')
                for value in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
            ),
            ("disabled", "", r'Custom\("disabled"'),
            ("<custom string>", _MODELS, r"Custom\(String\)"),
            ("<integer>", "codex-rs/codex-api/src/common.rs", r"serialize_u64"),
        ),
        note=(
            'as_str values; ultra resolves to max and persistent to "disabled" before the '
            "wire; numeric custom efforts serialize as integers"
        ),
    ),
}


def fetch_sources(tag: str, cache: Path) -> dict[str, str]:
    """``{repo path: source}`` for the files present at ``tag`` (cached)."""

    sources = {}
    for path in SOURCE_FILES:
        target = cache / tag / path
        absent = target.with_name(target.name + ".absent")
        if not target.exists() and not absent.exists():
            result = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{REPO}/contents/{path}?ref={tag}",
                    "-H",
                    "Accept: application/vnd.github.raw",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if result.returncode == 0:
                target.write_text(result.stdout, encoding="utf-8")
            elif "Not Found" in result.stdout + result.stderr:
                absent.write_text("", encoding="utf-8")
            else:
                raise SystemExit(f"gh api failed for {path}@{tag}: {result.stderr.strip()}")
        if target.exists():
            sources[path] = target.read_text(encoding="utf-8")
    return sources


class Spec:
    """Read-only view of the vendored OpenAPI closure."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self._schemas = document["components"]["schemas"]

    def schema(self, name: str) -> Any:
        return self._schemas.get(name)

    def resolve(self, node: Any) -> Any:
        while isinstance(node, Mapping) and "$ref" in node:
            node = self._schemas[node["$ref"].rsplit("/", 1)[-1]]
        return node

    def _alternatives(self, node: Any) -> list[Mapping[str, Any]]:
        node = self.resolve(node)
        if not isinstance(node, Mapping):
            return []
        nested = [*node.get("anyOf", ()), *node.get("oneOf", ()), *node.get("allOf", ())]
        return [node, *(alt for child in nested for alt in self._alternatives(child))]

    def prop(self, node: Any, name: str) -> Any:
        found = [
            alt["properties"][name]
            for alt in self._alternatives(node)
            if name in alt.get("properties", {})
        ]
        return {"anyOf": found} if found else None

    def items(self, node: Any) -> Any:
        found = [alt["items"] for alt in self._alternatives(node) if "items" in alt]
        return {"anyOf": found} if found else None

    def values(self, node: Any) -> tuple[frozenset[str], bool]:
        """Declared enum values, and whether any string is accepted."""

        values: set[str] = set()
        open_string = False
        for alt in self._alternatives(node):
            values.update(str(value) for value in alt.get("enum", ()))
            if "const" in alt:
                values.add(str(alt["const"]))
            if alt.get("type") == "string" and "enum" not in alt and "const" not in alt:
                open_string = True
        return frozenset(values), open_string

    def _branches(self, node: Any) -> list[Mapping[str, Any]]:
        """The members of a (nested) union; an object with properties is a leaf."""

        node = self.resolve(node)
        if not isinstance(node, Mapping):
            return []
        union = [*node.get("anyOf", ()), *node.get("oneOf", ())]
        if union and "properties" not in node and "allOf" not in node:
            return [branch for child in union for branch in self._branches(child)]
        return [node]

    def variants(self, node: Any, tag: str) -> dict[str, Any]:
        found: dict[str, list[Any]] = {}
        for branch in self._branches(node):
            tag_schema = self.prop(branch, tag)
            if tag_schema is None:
                continue
            for value in self.values(tag_schema)[0]:
                found.setdefault(value, []).append(branch)
        return {value: {"anyOf": branches} for value, branches in found.items()}


class Inventory:
    """Single-owner walk of one tag's wire types alongside the spec."""

    def __init__(self, tag: str, sources: Mapping[str, str], spec: Spec) -> None:
        self.tag = tag
        self._sources = sources
        self._spec = spec
        self._types: dict[str, TypeDef] = {}
        self._aliases: dict[str, str] = {}
        for path, source in sources.items():
            for name, definition in parse_types(path, source).items():
                self._types.setdefault(name, definition)
            for original, alias in _USE_ALIAS.findall(source):
                self._aliases.setdefault(alias, original)
        self.entries: dict[str, dict[str, Any]] = {}
        # (type, spec position) -> first path: a repeat is cited, not re-expanded.
        # Named types expand breadth-first, so the shallowest path is the first.
        self._expanded: dict[tuple[str, str], str] = {}
        self._pending: deque[tuple[str, str, Any, tuple[str, ...]]] = deque()

    def cite(self, path: str, line: int) -> str:
        return f"{path}@{self.tag}:{line}"

    def _anchor(self, path: str, pattern: str) -> str | None:
        candidates = [path] if path else sorted(self._sources)
        for candidate in candidates:
            match = re.search(pattern, self._sources.get(candidate, ""))
            if match is not None:
                line = self._sources[candidate].count("\n", 0, match.start()) + 1
                return self.cite(candidate, line)
        return None

    def _record(self, path: str, kind: str, source: str, in_spec: bool, **extra: Any) -> None:
        self.entries.setdefault(
            path,
            {
                "path": path,
                "kind": kind,
                "classification": "spec" if in_spec else "codex-extension",
                "source": source,
                **extra,
            },
        )

    def walk_root(self, root: str, path: str, spec_root: str) -> bool:
        definition = self._types.get(root)
        if definition is None:
            return False
        self._record(path, "root", self.cite(definition.path, definition.line), True)
        self._walk_def(definition, path, self._spec.schema(spec_root), (root,))
        while self._pending:
            self._walk_named(*self._pending.popleft())
        return True

    def _walk_ref(self, ref: TypeRef, path: str, spec_node: Any, stack: tuple[str, ...]) -> None:
        if ref.kind == "array":
            items = self._spec.items(spec_node) if spec_node is not None else None
            self._walk_ref(ref.inner, f"{path}[]", items, stack)
        elif ref.kind == "map" and ref.inner is not None:
            self._walk_ref(ref.inner, f"{path}{{*}}", None, stack)
        elif ref.kind == "named" and ref.name not in stack:
            job = (ref.name, path, spec_node, (*stack, ref.name))
            if self._aliases.get(ref.name, ref.name) in OVERRIDES:
                self._walk_named(*job)  # a wire-form override adds no nesting level
            else:
                self._pending.append(job)

    def _walk_named(self, name: str, path: str, spec_node: Any, stack: tuple[str, ...]) -> None:
        name = self._aliases.get(name, name)
        override = OVERRIDES.get(name)
        if override is None:
            definition = self._types.get(name)
            if definition is None:
                return
            key = (name, json.dumps(spec_node, sort_keys=True))
            first = self._expanded.setdefault(key, path)
            if first != path:
                self._record(
                    path,
                    "same_as",
                    self.cite(definition.path, definition.line),
                    spec_node is not None,
                )
                self.entries[path]["same_as"] = first
                return
            self._walk_def(definition, path, spec_node, stack)
            return
        for shape in override.shapes:
            self._walk_ref(shape, path, spec_node, stack)
        declared, open_string = (
            self._spec.values(spec_node) if spec_node is not None else (frozenset(), False)
        )
        for value, file, anchor in override.values:
            source = self._anchor(file, anchor)
            if source is not None:
                in_spec = value in declared or (open_string and not value.startswith("<"))
                self._record(f"{path}={value}", "value", source, in_spec, note=override.note)

    def _walk_def(
        self, definition: TypeDef, path: str, spec_node: Any, stack: tuple[str, ...]
    ) -> None:
        if definition.kind == "struct":
            self._walk_fields(
                definition.name,
                definition.path,
                definition.members,
                definition.serde,
                path,
                spec_node,
                stack,
            )
            return
        tag = definition.serde.get("tag")
        rename_all = definition.serde.get("rename_all")
        rename_all = rename_all if isinstance(rename_all, str) else None
        if definition.serde.get("untagged"):
            for variant in definition.members:
                self._walk_variant_body(definition, variant, path, spec_node, stack)
            return
        if isinstance(tag, str):
            spec_variants = self._spec.variants(spec_node, tag) if spec_node is not None else {}
            for variant in filter(serialized, definition.members):
                value = wire_name(variant.name, variant.serde, rename_all)
                variant_path = f"{path}{{{tag}={value}}}"
                variant_spec = spec_variants.get(value)
                extra = {}
                constructor = CONSTRUCTORS.get((definition.name, variant.name))
                if constructor is not None:
                    built = self._anchor(constructor, rf"{definition.name}::{variant.name}\b")
                    if built is not None:
                        extra["constructed_at"] = built
                self._record(
                    variant_path,
                    "variant",
                    self.cite(definition.path, variant.line),
                    variant_spec is not None,
                    **extra,
                )
                self._walk_variant_body(definition, variant, variant_path, variant_spec, stack)
            return
        declared, open_string = (
            self._spec.values(spec_node) if spec_node is not None else (frozenset(), False)
        )
        for variant in filter(serialized, definition.members):
            value = wire_name(variant.name, variant.serde, rename_all)
            in_spec = spec_node is not None and (value in declared or open_string)
            self._record(
                f"{path}={value}", "value", self.cite(definition.path, variant.line), in_spec
            )

    def _walk_variant_body(
        self,
        definition: TypeDef,
        variant: Member,
        path: str,
        spec_node: Any,
        stack: tuple[str, ...],
    ) -> None:
        if variant.shape == "struct":
            self._walk_fields(
                variant.name, definition.path, variant.fields, {}, path, spec_node, stack
            )
        elif variant.shape == "newtype":
            self._walk_ref(parse_type(variant.type_text), path, spec_node, stack)

    def _walk_fields(
        self,
        owner: str,
        file: str,
        fields: Iterable[Member],
        serde: Mapping[str, Any],
        path: str,
        spec_node: Any,
        stack: tuple[str, ...],
    ) -> None:
        rename_all = serde.get("rename_all") if isinstance(serde.get("rename_all"), str) else None
        for member in filter(serialized, fields):
            ref = TOOLS if (owner, member.name) in RAW_TOOL_FIELDS else parse_type(member.type_text)
            if member.serde.get("flatten"):
                self._walk_ref(ref, path, spec_node, stack)
                continue
            name = wire_name(member.name, member.serde, rename_all)
            child_path = f"{path}.{name}"
            child_spec = self._spec.prop(spec_node, name) if spec_node is not None else None
            optional = ref.optional or "skip_serializing_if" in member.serde
            self._record(
                child_path,
                "field",
                self.cite(file, member.line),
                child_spec is not None,
                optional=optional,
            )
            self._walk_ref(ref, child_path, child_spec, stack)


def build_inventory(tag: str, sources: Mapping[str, str], spec_document: Mapping[str, Any]) -> dict:
    inventory = Inventory(tag, sources, Spec(spec_document))
    routes = [
        {"root": root, "path": path, "spec_schema": spec_root, "route": route}
        for root, path, spec_root, route in ROOTS
        if inventory.walk_root(root, path, spec_root)
    ]
    entries = [inventory.entries[path] for path in sorted(inventory.entries)]
    extensions = [entry["path"] for entry in entries if entry["classification"] != "spec"]
    return {
        "codex_tag": tag,
        "generator": "scripts/codex_gate/extension_inventory.py",
        "note": NOTE,
        "spec": SPEC_FILE.name,
        "sources": sorted(sources),
        "routes": routes,
        "summary": {"entries": len(entries), "codex_extensions": len(extensions)},
        "codex_extensions": extensions,
        "entries": entries,
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="e.g. rust-v0.160.0")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: tests/fixtures/codex/<tag>/")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not re.fullmatch(r"rust-v\d+\.\d+\.\d+", args.tag):
        raise SystemExit("--tag must look like rust-v0.160.0")
    sources = fetch_sources(args.tag, args.cache)
    document = json.loads(SPEC_FILE.read_text(encoding="utf-8"))
    inventory = build_inventory(args.tag, sources, document)
    out = (args.out or FIXTURE_DIR / args.tag) / "extensions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{out}: {inventory['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
