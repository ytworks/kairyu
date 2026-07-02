"""Radix tree over paged KV blocks: page-granular prefix sharing (design doc §2.3).

Structure follows SGLang's RadixAttention adapted to page alignment:
- Node keys are token sequences whose length is a multiple of ``page_size``;
  children are keyed by their first page's token tuple, so partial *pages* are
  never shared and copy-on-write is unnecessary (tail pages are private).
- Nodes carry refcounts (live allocations + session pins) and an LRU stamp;
  eviction removes refcount-0 leaves, oldest first.
- Hit-rate metrics (``hit_tokens / total_tokens``) implement the M2 acceptance
  measurement for shared-prefix workloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kairyu.engine.core.pages import PagePool


class KVCacheFull(MemoryError):
    """Not enough free or evictable pages to admit the request."""


class _Node:
    __slots__ = ("key", "pages", "children", "parent", "ref_count", "last_access")

    def __init__(
        self,
        key: tuple[int, ...],
        pages: tuple[int, ...],
        parent: _Node | None,
    ) -> None:
        self.key = key
        self.pages = pages
        self.children: dict[tuple[int, ...], _Node] = {}
        self.parent = parent
        self.ref_count = 0
        self.last_access = 0


@dataclass(frozen=True)
class KVAllocation:
    tokens: tuple[int, ...]
    cached_pages: tuple[int, ...]
    new_full_pages: tuple[int, ...]
    tail_page: int | None
    _node: _Node = field(repr=False)
    _freed: bool = field(default=False, repr=False)

    @property
    def pages(self) -> tuple[int, ...]:
        tail = (self.tail_page,) if self.tail_page is not None else ()
        return self.cached_pages + self.new_full_pages + tail


class RadixKVCache:
    def __init__(self, num_pages: int, page_size: int = 16) -> None:
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        self._pool = PagePool(num_pages)
        self._page_size = page_size
        self._root = _Node(key=(), pages=(), parent=None)
        self._root.ref_count = 1  # never evictable
        self._clock = 0
        self._pins: dict[str, _Node] = {}
        self._hit_tokens = 0
        self._total_tokens = 0

    @property
    def num_free_pages(self) -> int:
        return self._pool.num_free

    @property
    def hit_rate(self) -> float:
        if self._total_tokens == 0:
            return 0.0
        return self._hit_tokens / self._total_tokens

    def _touch(self, node: _Node) -> None:
        self._clock += 1
        node.last_access = self._clock

    def _split(self, node: _Node, keep_pages: int) -> _Node:
        """Split ``node`` so its first ``keep_pages`` pages become a new parent.

        The original node object keeps the tail segment so external references
        (allocations, pins) stay valid; the new upper node inherits the
        refcount because every lock on the old node locked its whole key.
        """
        split_tokens = keep_pages * self._page_size
        upper = _Node(
            key=node.key[:split_tokens],
            pages=node.pages[:keep_pages],
            parent=node.parent,
        )
        upper.ref_count = node.ref_count
        upper.last_access = node.last_access
        parent = node.parent
        assert parent is not None  # root is never split (its key is empty)
        parent.children[upper.key[: self._page_size]] = upper
        node.key = node.key[split_tokens:]
        node.pages = node.pages[keep_pages:]
        node.parent = upper
        upper.children[node.key[: self._page_size]] = node
        return upper

    def _match_and_lock(self, tokens: tuple[int, ...]) -> tuple[int, tuple[int, ...], _Node]:
        """Walk the tree, splitting on partial matches; lock and stamp the path."""
        node = self._root
        self._touch(node)
        matched_tokens = 0
        matched_pages: list[int] = []
        position = 0
        while True:
            first_page = tuple(tokens[position : position + self._page_size])
            if len(first_page) < self._page_size:
                break
            child = node.children.get(first_page)
            if child is None:
                break
            whole_pages = len(child.key) // self._page_size
            common_pages = 0
            for i in range(whole_pages):
                start = i * self._page_size
                segment = tokens[position + start : position + start + self._page_size]
                if tuple(segment) != child.key[start : start + self._page_size]:
                    break
                common_pages += 1
            if common_pages < whole_pages:
                child = self._split(child, common_pages)
            child.ref_count += 1
            self._touch(child)
            matched_pages.extend(child.pages)
            matched_tokens += len(child.key)
            position += len(child.key)
            node = child
        return matched_tokens, tuple(matched_pages), node

    def _unlock_path(self, node: _Node) -> None:
        current: _Node | None = node
        while current is not None and current is not self._root:
            current.ref_count -= 1
            current = current.parent

    def _evictable_leaves(self) -> list[_Node]:
        leaves: list[_Node] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is not self._root and not node.children and node.ref_count == 0:
                leaves.append(node)
        return leaves

    def _ensure_free(self, needed: int) -> bool:
        while self._pool.num_free < needed:
            leaves = self._evictable_leaves()
            if not leaves:
                return False
            victim = min(leaves, key=lambda leaf: leaf.last_access)
            self._pool.free(victim.pages)
            assert victim.parent is not None
            del victim.parent.children[victim.key[: self._page_size]]
        return True

    def allocate(self, tokens: tuple[int, ...]) -> KVAllocation:
        tokens = tuple(tokens)
        matched_tokens, cached_pages, node = self._match_and_lock(tokens)
        suffix_len = len(tokens) - matched_tokens
        full_pages_needed = suffix_len // self._page_size
        tail_tokens = suffix_len % self._page_size
        needed = full_pages_needed + (1 if tail_tokens else 0)
        if not self._ensure_free(needed):
            self._unlock_path(node)
            raise KVCacheFull(
                f"need {needed} pages, {self._pool.num_free} free and nothing evictable"
            )
        new_full_pages = self._pool.allocate(full_pages_needed)
        tail_page = self._pool.allocate(1)[0] if tail_tokens else None
        if full_pages_needed:
            key = tokens[matched_tokens : matched_tokens + full_pages_needed * self._page_size]
            child = _Node(key=key, pages=new_full_pages, parent=node)
            child.ref_count = 1
            self._touch(child)
            node.children[key[: self._page_size]] = child
            node = child
        self._hit_tokens += matched_tokens
        self._total_tokens += len(tokens)
        return KVAllocation(
            tokens=tokens,
            cached_pages=cached_pages,
            new_full_pages=new_full_pages,
            tail_page=tail_page,
            _node=node,
        )

    def free(self, allocation: KVAllocation) -> None:
        if allocation._freed:
            raise ValueError("allocation was already freed")
        object.__setattr__(allocation, "_freed", True)
        self._unlock_path(allocation._node)
        if allocation.tail_page is not None:
            self._pool.free((allocation.tail_page,))

    def pin(self, session_id: str, tokens: tuple[int, ...]) -> None:
        """Hold the matched prefix path against eviction (orchestration sessions)."""
        if session_id in self._pins:
            raise ValueError(f"session {session_id!r} is already pinned")
        _, _, node = self._match_and_lock(tuple(tokens))
        self._pins[session_id] = node

    def unpin(self, session_id: str) -> None:
        node = self._pins.pop(session_id, None)
        if node is None:
            raise ValueError(f"session {session_id!r} is not pinned")
        self._unlock_path(node)
