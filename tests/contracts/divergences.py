"""Load and match ``divergences.toml``, the schema gate's allowlist.

Every entry names the gap it tracks and the work package that removes it
(``kind = "temporary"``), or records a deliberate extension
(``codex-extension`` / ``kairyu-extension``). Entries are validated when the
file is loaded, so a malformed allowlist fails the session instead of
silently allowing violations.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from tests.contracts.openai_contract import Violation

DIVERGENCES_FILE = Path(__file__).with_name("divergences.toml")
KINDS = frozenset({"temporary", "codex-extension", "kairyu-extension"})
REQUIRED_FIELDS = ("id", "schema", "pointer_glob", "keyword", "gap_id", "owner_wp", "kind")
OPTIONAL_FIELDS = ("value", "note")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_OWNER_WP = re.compile(r"^WP-\d{2}[a-c]?$")
# Research gap IDs, verifier-added gaps, review requirements, and m20 decisions.
_GAP_ID = re.compile(r"^(G-[a-z-]+-\d+|M-(W|SR|ST)-\d+|R-\d+|D\d+|D-[a-j])$")


class DivergenceFileError(ValueError):
    """``divergences.toml`` violates the entry contract."""


@dataclass(frozen=True)
class Divergence:
    id: str
    schema: str
    pointer_glob: str
    keyword: str
    gap_id: str
    owner_wp: str
    kind: str
    value: str | None = None
    note: str = ""

    def matches(self, violation: Violation) -> bool:
        return (
            violation.keyword == self.keyword
            and fnmatchcase(violation.pointer, self.pointer_glob)
            and fnmatchcase(violation.schema, self.schema)
            and (self.value is None or violation.value == self.value)
        )


def _entry(raw: Mapping[str, Any], index: int) -> Divergence:
    where = f"divergence #{index} ({raw.get('id', '?')})"
    missing = [name for name in REQUIRED_FIELDS if name not in raw]
    unknown = sorted(set(raw) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if missing or unknown:
        raise DivergenceFileError(f"{where}: missing {missing}, unknown {unknown}")
    text = {name: raw[name] if isinstance(raw[name], str) else "" for name in REQUIRED_FIELDS}
    checks = {
        "id": _ID.match(text["id"]),
        "schema": text["schema"],
        "pointer_glob": text["pointer_glob"],
        "keyword": text["keyword"],
        "gap_id": _GAP_ID.match(text["gap_id"]),
        "owner_wp": _OWNER_WP.match(text["owner_wp"]),
        "kind": text["kind"] in KINDS,
        "value": isinstance(raw.get("value", ""), str),
        "note": isinstance(raw.get("note", ""), str),
    }
    invalid = [name for name, ok in checks.items() if not ok]
    if invalid:
        raise DivergenceFileError(f"{where}: invalid {invalid}")
    return Divergence(
        id=raw["id"],
        schema=raw["schema"],
        pointer_glob=raw["pointer_glob"],
        keyword=raw["keyword"],
        gap_id=raw["gap_id"],
        owner_wp=raw["owner_wp"],
        kind=raw["kind"],
        value=raw.get("value"),
        note=raw.get("note", ""),
    )


def load_divergences(path: Path = DIVERGENCES_FILE) -> tuple[Divergence, ...]:
    raw = tomllib.loads(path.read_text(encoding="utf-8")).get("divergence", [])
    entries = tuple(_entry(item, index) for index, item in enumerate(raw))
    ids = [entry.id for entry in entries]
    duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
    if duplicates:
        raise DivergenceFileError(f"duplicate divergence ids: {duplicates}")
    return entries


def triage(
    violations: Iterable[Violation], divergences: Iterable[Divergence]
) -> tuple[tuple[Violation, ...], frozenset[str]]:
    """Split violations into unallowed ones and the ids of the entries that matched."""

    entries = tuple(divergences)
    unallowed: list[Violation] = []
    matched: set[str] = set()
    for violation in violations:
        hits = [entry.id for entry in entries if entry.matches(violation)]
        if hits:
            matched.update(hits)
        else:
            unallowed.append(violation)
    return tuple(unallowed), frozenset(matched)
