"""Bounded, tenant-scoped Responses state for ``previous_response_id``."""

from __future__ import annotations

import copy
from collections import OrderedDict


class ResponseStore:
    """Bounded, tenant-scoped state for ``previous_response_id``."""

    def __init__(self, max_items: int = 4096) -> None:
        self._items: OrderedDict[str, tuple[str, list[dict]]] = OrderedDict()
        self._max = max_items

    def save(self, response_id: str, items: list[dict], owner: str = "default") -> None:
        self._items[response_id] = (owner, copy.deepcopy(items))
        self._items.move_to_end(response_id)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def get(self, response_id: str, owner: str = "default") -> list[dict] | None:
        entry = self._items.get(response_id)
        if entry is None or entry[0] != owner:
            return None
        self._items.move_to_end(response_id)
        return copy.deepcopy(entry[1])
