"""Machine-applied derivations of Codex request fixtures (M20 WP-02, D1).

A derived fixture is a recorded capture plus input items, cited from codex-rs
at the tag, that the recording scenario cannot make Codex send (a replayed
reasoning item, a ``configuration_update``, an ``input_audio`` part, ...). Its
``provenance.derivation`` lists those edits as data, and
``record_proxy promote`` re-applies them to a fresh capture, so a refresh keeps
what the fixture tests instead of erasing it. Edits apply in order to
``request.body.input``:

- ``{"op": "insert", "before": ANCHOR, "items": [...]}`` inserts the items
  before the input item ANCHOR selects;
- ``{"op": "append", "items": [...]}`` appends the items;
- ``{"op": "extend_content", "of": ANCHOR, "parts": [...]}`` appends content
  parts to the message ANCHOR selects.

ANCHOR selects the first input item whose fields equal the given ones (``type``,
``role``, ...), or the final one with ``"last": true``. An anchor that matches
nothing fails, so a capture that lost the anchored item is reported instead of
silently becoming a different fixture.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

ANCHOR_LAST = "last"


class DerivationError(ValueError):
    """A derivation edit is malformed or does not apply to the capture."""


def apply_derivation(body: Mapping[str, Any], edits: Sequence[Mapping[str, Any]]) -> dict:
    """Return a copy of a request body with the edits applied in order (pure)."""

    items = copy.deepcopy(list(body["input"]))
    for edit in edits:
        items = _apply(items, edit)
    return {**copy.deepcopy(dict(body)), "input": items}


def _apply(items: list[dict], edit: Mapping[str, Any]) -> list[dict]:
    op = edit.get("op")
    if op == "insert":
        index = _anchor_index(items, edit["before"])
        return [*items[:index], *copy.deepcopy(edit["items"]), *items[index:]]
    if op == "append":
        return [*items, *copy.deepcopy(edit["items"])]
    if op == "extend_content":
        index = _anchor_index(items, edit["of"])
        message = items[index]
        extended = {**message, "content": [*message["content"], *copy.deepcopy(edit["parts"])]}
        return [*items[:index], extended, *items[index + 1 :]]
    raise DerivationError(f"unknown derivation op {op!r}")


def _anchor_index(items: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]) -> int:
    fields = {name: value for name, value in anchor.items() if name != ANCHOR_LAST}
    matches = [
        index
        for index, item in enumerate(items)
        if all(item.get(name) == value for name, value in fields.items())
    ]
    if not matches:
        raise DerivationError(f"no input item matches the anchor {dict(anchor)}")
    return matches[-1] if anchor.get(ANCHOR_LAST) else matches[0]
