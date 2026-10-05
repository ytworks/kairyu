"""Bounded, tenant-scoped Responses state for retrieval and continuation.

One record per stored response keeps the returned envelope (for ``GET``), the
request's own input items (for ``GET .../input_items``), and the continuation
context (for ``previous_response_id``). The record is deep-copied once on save
so its envelope output and the tail of its context share item objects instead
of duplicating them. The store is per-process memory: it is not shared across
gateways (documented in ``docs/deployment.md``).
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass

from kairyu.entrypoints.server.responses_protocol import ResponsesError

_MAX_LIST_LIMIT = 100
_DEFAULT_LIST_LIMIT = 20


@dataclass(frozen=True)
class StoredResponse:
    owner: str
    response: dict
    input_items: list[dict]
    context: list[dict]


class ResponseStore:
    """Bounded LRU of stored responses, isolated per tenant."""

    def __init__(self, max_items: int = 4096) -> None:
        self._items: OrderedDict[str, StoredResponse] = OrderedDict()
        self._max = max_items

    def save(
        self,
        response_id: str,
        items: list[dict],
        owner: str = "default",
        *,
        response: dict | None = None,
        input_items: list[dict] | None = None,
    ) -> None:
        response, input_items, items = copy.deepcopy(
            (response or {}, input_items or [], items)
        )
        self._items[response_id] = StoredResponse(owner, response, input_items, items)
        self._items.move_to_end(response_id)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def _record(self, response_id: str, owner: str) -> StoredResponse | None:
        entry = self._items.get(response_id)
        if entry is None or entry.owner != owner:
            return None
        self._items.move_to_end(response_id)
        return entry

    def has(self, response_id: str, owner: str = "default") -> bool:
        return self._record(response_id, owner) is not None

    def get(self, response_id: str, owner: str = "default") -> list[dict] | None:
        """The continuation context for ``previous_response_id``."""

        entry = self._record(response_id, owner)
        return None if entry is None else copy.deepcopy(entry.context)

    def response(self, response_id: str, owner: str = "default") -> dict | None:
        entry = self._record(response_id, owner)
        return None if entry is None else copy.deepcopy(entry.response)

    def input_items(self, response_id: str, owner: str = "default") -> list[dict] | None:
        entry = self._record(response_id, owner)
        return None if entry is None else copy.deepcopy(entry.input_items)

    def delete(self, response_id: str, owner: str = "default") -> bool:
        if self._record(response_id, owner) is None:
            return False
        del self._items[response_id]
        return True


def list_page(items: list[dict], query) -> dict:
    """A ``ResponseItemList`` page per the ``after``/``limit``/``order`` query."""

    order = query.get("order", "desc")
    if order not in ("asc", "desc"):
        raise ResponsesError("order must be 'asc' or 'desc'", param="order", code="invalid_value")
    raw_limit = query.get("limit", str(_DEFAULT_LIST_LIMIT))
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 0
    if not 1 <= limit <= _MAX_LIST_LIMIT:
        raise ResponsesError(
            f"limit must be an integer between 1 and {_MAX_LIST_LIMIT}",
            param="limit",
            code="invalid_value",
        )
    ordered = items if order == "asc" else list(reversed(items))
    after = query.get("after")
    if after is not None:
        ids = [item.get("id") for item in ordered]
        if after not in ids:
            raise ResponsesError(
                f"No item with id '{after}' in this response's input.",
                param="after",
                code="invalid_value",
            )
        ordered = ordered[ids.index(after) + 1 :]
    page = ordered[:limit]
    return {
        "object": "list",
        "data": page,
        "first_id": page[0]["id"] if page else None,
        "last_id": page[-1]["id"] if page else None,
        "has_more": len(ordered) > limit,
    }


@dataclass(frozen=True)
class PendingSave:
    """Where one finished response is stored; ``store`` is None for store=false."""

    store: ResponseStore | None
    owner: str
    context: list[dict]
    input_items: list[dict]

    def commit(self, response: dict) -> None:
        if self.store is None:
            return
        # Reasoning tokens are minted per read (``include``), never stored.
        output = [
            {key: value for key, value in item.items() if key != "encrypted_content"}
            if item.get("type") == "reasoning"
            else item
            for item in response["output"]
        ]
        self.store.save(
            response["id"],
            self.context + output,
            self.owner,
            response={**response, "output": output},
            input_items=self.input_items,
        )
