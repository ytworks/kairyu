"""Fixed-size KV page pool: allocation/free bookkeeping only.

Refcounting and sharing policy live in the radix tree (radix_kv.py); the pool
just tracks which physical page ids are free.
"""

from __future__ import annotations

from collections.abc import Iterable


class PagePool:
    def __init__(self, num_pages: int) -> None:
        if num_pages < 1:
            raise ValueError(f"num_pages must be >= 1, got {num_pages}")
        self._num_pages = num_pages
        self._free: list[int] = list(range(num_pages - 1, -1, -1))
        self._allocated: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, count: int) -> tuple[int, ...]:
        if count == 0:
            return ()
        if count > len(self._free):
            raise MemoryError(
                f"requested {count} pages but only {len(self._free)} of "
                f"{self._num_pages} pages are free"
            )
        pages = tuple(self._free.pop() for _ in range(count))
        self._allocated.update(pages)
        return pages

    def free(self, pages: Iterable[int]) -> None:
        page_list = list(pages)
        not_allocated = [page for page in page_list if page not in self._allocated]
        if not_allocated:
            raise ValueError(f"pages {not_allocated} are not allocated")
        if len(set(page_list)) != len(page_list):
            # a duplicate id would append the same physical page to _free twice,
            # allowing it to be handed out to two requests at once
            raise ValueError(f"duplicate page ids in free batch: {page_list}")
        for page in page_list:
            self._allocated.discard(page)
            self._free.append(page)
