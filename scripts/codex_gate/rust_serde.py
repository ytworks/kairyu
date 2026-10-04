"""Minimal reader for the serde-derived wire types of codex-rs (M20 WP-02).

Only what ``extension_inventory`` needs: ``struct`` and ``enum`` items with
their ``#[serde(...)]`` attributes, named fields, unit / tuple / struct-like
variants, and the type text of every field. Line numbers point at the field or
variant declaration so inventory entries cite ``path@tag:line``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

_ITEM = re.compile(r"\b(?:pub(?:\([^)]*\))?\s+)?(struct|enum)\s+(\w+)\s*(?:<[^>{;]*>)?\s*\{")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_ATTR = re.compile(r"#\[\s*serde\s*\((.*)\)\s*\]\s*$", re.S)
_FIELD = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(r#)?(\w+)\s*:\s*(.+)$", re.S)
_VARIANT = re.compile(r"^(\w+)\s*(.*)$", re.S)
_OPEN = "([{<"
_CLOSE = ")]}>"
SCALARS = frozenset(
    {
        "String",
        "str",
        "bool",
        "u8",
        "u16",
        "u32",
        "u64",
        "usize",
        "i8",
        "i16",
        "i32",
        "i64",
        "f32",
        "f64",
        "Number",
        "char",
    }
)
JSON_TYPES = frozenset({"Value", "RawValue"})
WRAPPERS = frozenset({"Box", "Arc", "Rc", "Cow"})
MAPS = frozenset({"HashMap", "BTreeMap", "IndexMap"})
ARRAYS = frozenset({"Vec", "VecDeque"})


@dataclass(frozen=True)
class TypeRef:
    """A field type reduced to its wire shape."""

    kind: str  # "named" | "array" | "map" | "scalar" | "json" | "none"
    name: str = ""
    inner: TypeRef | None = None
    optional: bool = False


@dataclass(frozen=True)
class Member:
    """A struct field, or an enum variant (``shape`` unit | newtype | struct)."""

    name: str
    line: int
    serde: Mapping[str, str | bool]
    type_text: str = ""
    shape: str = "field"
    fields: tuple[Member, ...] = ()


@dataclass(frozen=True)
class TypeDef:
    name: str
    kind: str  # "struct" | "enum"
    path: str
    line: int
    serde: Mapping[str, str | bool]
    members: tuple[Member, ...] = field(default=())


def strip_comments(source: str) -> str:
    """Blank out comments, keeping offsets (and so line numbers) intact."""

    source = _BLOCK_COMMENT.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), source)
    lines = []
    for line in source.split("\n"):
        in_string = False
        for index, char in enumerate(line):
            if char == '"' and (index == 0 or line[index - 1] != "\\"):
                in_string = not in_string
            elif not in_string and line.startswith("//", index):
                line = line[:index] + " " * (len(line) - index)
                break
        lines.append(line)
    return "\n".join(lines)


def _matching(text: str, start: int) -> int:
    """Index of the bracket closing ``text[start]`` (strings skipped)."""

    depth = 0
    in_string = False
    for index in range(start, len(text)):
        char = text[index]
        if char == '"' and text[index - 1] != "\\":
            in_string = not in_string
        elif in_string:
            continue
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unbalanced bracket at offset {start}")


def split_top_level(text: str, separator: str = ",") -> Iterator[tuple[int, str]]:
    """Yield ``(offset, chunk)`` for chunks split at depth-0 separators."""

    depth = 0
    in_string = False
    start = 0
    for index, char in enumerate(text):
        if char == '"' and (index == 0 or text[index - 1] != "\\"):
            in_string = not in_string
        elif in_string:
            continue
        elif char in _OPEN and not (char == "<" and text[index - 1 : index] == "-"):
            depth += 1
        elif char in _CLOSE and not (char == ">" and text[index - 1 : index] == "-"):
            depth -= 1
        elif char == separator and depth == 0:
            yield start, text[start:index]
            start = index + 1
    if text[start:].strip():
        yield start, text[start:]


def _leading_attrs(chunk: str) -> tuple[list[str], int]:
    """Split leading ``#[...]`` attributes from a member chunk."""

    attrs = []
    position = 0
    while True:
        stripped = len(chunk[position:]) - len(chunk[position:].lstrip())
        position += stripped
        if not chunk.startswith("#[", position):
            return attrs, position
        end = _matching(chunk, position + 1)
        attrs.append(chunk[position : end + 1])
        position = end + 1


def parse_serde(attrs: list[str]) -> dict[str, str | bool]:
    """Merge ``#[serde(...)]`` attributes into ``{key: value or True}``."""

    merged: dict[str, str | bool] = {}
    for attr in attrs:
        match = _ATTR.match(attr.strip())
        if match is None:
            continue
        for _offset, part in split_top_level(match.group(1)):
            key, separator, value = part.partition("=")
            key = key.strip()
            if key:
                merged[key] = value.strip().strip('"') if separator else True
    return merged


def _attrs_before(text: str, start: int) -> list[str]:
    """The attribute blocks directly above an item starting at ``start``."""

    attrs = []
    end = start
    while True:
        before = text[:end].rstrip()
        if not before.endswith("]"):
            return list(reversed(attrs))
        depth = 0
        for index in range(len(before) - 1, -1, -1):
            if before[index] == "]":
                depth += 1
            elif before[index] == "[":
                depth -= 1
                if depth == 0:
                    break
        if index == 0 or before[index - 1] != "#":
            return list(reversed(attrs))
        attrs.append(before[index - 1 :])
        end = index - 1


def _members(body: str, body_offset: int, text: str, enum: bool) -> tuple[Member, ...]:
    members = []
    for offset, chunk in split_top_level(body):
        attrs, position = _leading_attrs(chunk)
        declaration = chunk[position:].strip()
        if not declaration:
            continue
        absolute = body_offset + offset + position
        line = text.count("\n", 0, absolute) + 1
        serde = parse_serde(attrs)
        if not enum:
            match = _FIELD.match(declaration)
            if match is None:
                continue
            members.append(Member(match.group(2), line, serde, " ".join(match.group(3).split())))
            continue
        match = _VARIANT.match(declaration)
        if match is None:
            continue
        name, rest = match.group(1), match.group(2).strip()
        if rest.startswith("{"):
            inner_start = chunk.index("{", position)
            inner = chunk[inner_start + 1 : _matching(chunk, inner_start)]
            fields = _members(inner, body_offset + offset + inner_start + 1, text, enum=False)
            members.append(Member(name, line, serde, shape="struct", fields=fields))
        elif rest.startswith("("):
            members.append(Member(name, line, serde, " ".join(rest[1:-1].split()), "newtype"))
        else:
            members.append(Member(name, line, serde, shape="unit"))
    return tuple(members)


def parse_types(path: str, source: str) -> dict[str, TypeDef]:
    """Every ``struct``/``enum`` with a braced body in ``source``."""

    text = strip_comments(source)
    types = {}
    for match in _ITEM.finditer(text):
        kind, name = match.group(1), match.group(2)
        open_brace = match.end() - 1
        body = text[open_brace + 1 : _matching(text, open_brace)]
        line = text.count("\n", 0, match.start()) + 1
        types[name] = TypeDef(
            name=name,
            kind=kind,
            path=path,
            line=line,
            serde=parse_serde(_attrs_before(text, match.start())),
            members=_members(body, open_brace + 1, text, enum=kind == "enum"),
        )
    return types


def _generic_args(text: str) -> tuple[str, list[str]]:
    head, _, rest = text.partition("<")
    args = [chunk.strip() for _offset, chunk in split_top_level(rest[: rest.rfind(">")])]
    return head.strip(), args


def parse_type(text: str) -> TypeRef:
    """Reduce a Rust field type to its wire shape."""

    text = re.sub(r"&\s*'\w+\s*|&\s*|'\w+\s*,?\s*|\bmut\s+|\bdyn\s+", "", text).strip()
    if text.startswith("[") and text.endswith("]"):
        return TypeRef("array", inner=parse_type(text[1:-1]))
    if "<" not in text:
        name = text.split("::")[-1].strip()
        if name in SCALARS:
            return TypeRef("scalar", name)
        if name in JSON_TYPES:
            return TypeRef("json", name)
        return TypeRef("named", name)
    head, args = _generic_args(text)
    head = head.split("::")[-1]
    if head == "Option":
        inner = parse_type(args[0])
        return TypeRef(inner.kind, inner.name, inner.inner, optional=True)
    if head in ARRAYS:
        return TypeRef("array", inner=parse_type(args[0]))
    if head in MAPS:
        return TypeRef("map", inner=parse_type(args[-1]))
    if head in WRAPPERS:
        return parse_type(args[-1])
    return TypeRef("named", head)


_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def wire_name(name: str, serde: Mapping[str, str | bool], rename_all: str | None) -> str:
    """The serialized name of a field or variant."""

    renamed = serde.get("rename")
    if isinstance(renamed, str):
        return renamed
    name = name.removeprefix("r#")
    if rename_all == "snake_case":
        return _CASE_BOUNDARY.sub("_", name).lower()
    if rename_all == "lowercase":
        return name.lower()
    if rename_all == "camelCase":
        return name[:1].lower() + name[1:]
    return name


def serialized(member: Member) -> bool:
    """Whether a member can appear in serialized output."""

    return not any(member.serde.get(key) for key in ("skip", "skip_serializing", "other"))
