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
    __slots__ = (
        "key",
        "pages",
        "children",
        "parent",
        "ref_count",
        "last_access",
        "computed",
        "publishing",
    )

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
        # KV for these pages has actually been written. Matching skips
        # uncomputed nodes so chunked prefill in progress is never shared
        # as if it were valid cache (garbage-KV protection).
        self.computed = False
        # Suppress same-node reentry while BlockStored is being delivered.
        # This is reset even when the sink fails so callers can retry.
        self.publishing = False


@dataclass(frozen=True)
class KVAllocation:
    tokens: tuple[int, ...]
    cached_pages: tuple[int, ...]
    new_full_pages: tuple[int, ...]
    tail_page: int | None
    _node: _Node = field(repr=False)
    # Publication and release form one allocation lifecycle: event-sink
    # reentry must not start a nested terminal operation on this allocation.
    _freed: bool = field(default=False, repr=False)
    _publishing: bool = field(default=False, repr=False)
    _releasing: bool = field(default=False, repr=False)
    _tree_inserted: bool = field(default=True, repr=False)
    _page_size: int = field(default=16, repr=False)

    @property
    def pages(self) -> tuple[int, ...]:
        tail = (self.tail_page,) if self.tail_page is not None else ()
        return self.cached_pages + self.new_full_pages + tail

    @property
    def num_cached_tokens(self) -> int:
        """Prompt tokens whose KV is already computed — prefill can skip them."""
        return len(self.cached_pages) * self._page_size


class RadixKVCache:
    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def page_size(self) -> int:
        return self._page_size

    def __init__(
        self, num_pages: int, page_size: int = 16, event_sink=None
    ) -> None:
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        self._pool = PagePool(num_pages)
        self._num_pages = num_pages
        self._page_size = page_size
        self._root = _Node(key=(), pages=(), parent=None)
        self._root.ref_count = 1  # never evictable
        self._root.computed = True
        self._clock = 0
        self._alloc_tick = 0
        self._pins: dict[str, tuple[_Node, int | None]] = {}  # node, expiry tick
        self._hit_tokens = 0
        self._total_tokens = 0
        # m10b D7: optional KV-event sink (BlockStored/BlockRemoved)
        self._event_sink = event_sink

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
        upper.computed = node.computed
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
            if child is None or not child.computed:
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
            self._emit_removed(victim)  # the ONLY BlockRemoved source (A13)
            self._pool.free(victim.pages)
            assert victim.parent is not None
            del victim.parent.children[victim.key[: self._page_size]]
        return True

    def _expire_pins(self) -> None:
        expired = [
            session_id
            for session_id, (_, expiry) in self._pins.items()
            if expiry is not None and self._alloc_tick >= expiry
        ]
        for session_id in expired:
            node, _ = self._pins.pop(session_id)
            self._unlock_path(node)

    def allocate(self, tokens: tuple[int, ...]) -> KVAllocation:
        tokens = tuple(tokens)
        self._alloc_tick += 1
        self._expire_pins()
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
        tree_inserted = True
        if full_pages_needed:
            key = tokens[matched_tokens : matched_tokens + full_pages_needed * self._page_size]
            if key[: self._page_size] in node.children:
                # an uncomputed sibling owns this key (in-flight prefill of the
                # same prefix); keep our pages private instead of colliding
                tree_inserted = False
            else:
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
            _tree_inserted=tree_inserted,
            _page_size=self._page_size,
        )

    def _full_prefix(self, node) -> tuple[int, ...]:
        parts = []
        current = node
        while current is not None and current.key:
            parts.append(tuple(current.key))
            current = current.parent
        tokens: tuple[int, ...] = ()
        for part in reversed(parts):
            tokens += part
        return tokens

    def _emit_stored(self, node) -> None:
        """BlockStored on the computed False->True transition (m10b A13)."""
        if self._event_sink is None:
            return
        import hashlib as _hashlib

        prefix = self._full_prefix(node)
        start = len(prefix) - len(node.key)
        hashes = []
        for offset in range(0, len(node.key), self._page_size):
            end = start + offset + self._page_size
            hashes.append(
                _hashlib.sha256(repr(prefix[:end]).encode()).hexdigest()[:16]
            )
        parent_hash = (
            _hashlib.sha256(repr(prefix[:start]).encode()).hexdigest()[:16]
            if start
            else None
        )
        self._event_sink(
            {
                "type": "BlockStored",
                "block_hashes": hashes,
                "parent_block_hash": parent_hash,
                "token_ids": list(node.key),
                "block_size": self._page_size,
            }
        )

    def _emit_removed(self, node) -> None:
        if self._event_sink is None:
            return
        import hashlib as _hashlib

        prefix = self._full_prefix(node)
        start = len(prefix) - len(node.key)
        hashes = [
            _hashlib.sha256(repr(prefix[: start + offset + self._page_size]).encode())
            .hexdigest()[:16]
            for offset in range(0, len(node.key), self._page_size)
        ]
        self._event_sink({"type": "BlockRemoved", "block_hashes": hashes})

    def mark_computed(self, allocation: KVAllocation) -> None:
        """Record that the allocation's prefill KV has been written (prefill done)."""
        if allocation._tree_inserted and allocation.new_full_pages:
            node = allocation._node
            if not node.computed and not node.publishing:
                node.publishing = True
                object.__setattr__(allocation, "_publishing", True)
                try:
                    self._emit_stored(node)
                    node.computed = True
                finally:
                    object.__setattr__(allocation, "_publishing", False)
                    node.publishing = False

    def _begin_release(self, allocation: KVAllocation) -> bool:
        """Start a release, suppressing same-stack terminal reentry."""
        if allocation._freed:
            raise ValueError("allocation was already freed")
        if allocation._publishing or allocation._releasing:
            return False
        object.__setattr__(allocation, "_releasing", True)
        return True

    def _finish_release(self, allocation: KVAllocation) -> None:
        object.__setattr__(allocation, "_freed", True)
        object.__setattr__(allocation, "_releasing", False)

    def _abort_release(self, allocation: KVAllocation) -> None:
        object.__setattr__(allocation, "_releasing", False)

    def _release(self, allocation: KVAllocation) -> None:
        # free()/commit run only at request completion, so the KV is written;
        # a future preemption path must release WITHOUT marking computed
        self.mark_computed(allocation)
        self._unlock_path(allocation._node)
        if not allocation._tree_inserted and allocation.new_full_pages:
            self._pool.free(allocation.new_full_pages)

    def free(self, allocation: KVAllocation) -> None:
        if not self._begin_release(allocation):
            return
        try:
            self._release(allocation)
            if allocation.tail_page is not None:
                self._pool.free((allocation.tail_page,))
        except Exception:
            self._abort_release(allocation)
            raise
        self._finish_release(allocation)

    def commit_and_release(
        self,
        allocation: KVAllocation,
        output_tokens: tuple[int, ...],
        decode_pages: tuple[int, ...],
    ) -> None:
        """Finish a request: fold fully-generated pages into the radix tree.

        Turn N+1 of a conversation prompts with turn N's completion appended,
        so caching generated tokens is what makes multi-turn prefixes hit.
        Partially-filled pages are returned to the pool.
        """
        if not self._begin_release(allocation):
            return
        sequence = allocation.tokens + tuple(output_tokens)
        prompt_full = len(allocation.tokens) // self._page_size
        # The decode loop writes the KV of the *previous* token each step, so
        # the last sampled token's KV slot is never written (C1). Cap the
        # committable length below that final token so a page ending exactly on
        # the sequence boundary is not folded as computed — otherwise the next
        # turn matches it as cache and reads a garbage KV row for that token.
        written_len = len(sequence) - 1 if output_tokens else len(sequence)
        sequence_full = written_len // self._page_size
        extra_full = sequence_full - prompt_full
        candidates = (
            ((allocation.tail_page,) if allocation.tail_page is not None else ())
            + tuple(decode_pages)
        )
        node = allocation._node
        kept: tuple[int, ...] = ()
        try:
            if extra_full > 0 and allocation._tree_inserted:
                key = sequence[
                    prompt_full * self._page_size : sequence_full * self._page_size
                ]
                first_page = key[: self._page_size]
                candidate_pages = candidates[:extra_full]
                existing = node.children.get(first_page)
                if existing is None:
                    child = _Node(key=key, pages=candidate_pages, parent=node)
                    # Keep an uncomputed, in-delivery child invisible to matching
                    # and protected from reentrant eviction until the sink accepts it.
                    child.ref_count = 1
                    child.publishing = True
                    self._touch(child)
                    node.children[first_page] = child
                    try:
                        self._emit_stored(child)  # decode-extension store (A13)
                        child.computed = True
                        kept = candidate_pages
                    finally:
                        child.publishing = False
                        child.ref_count = 0
                        if (
                            not child.computed
                            and node.children.get(first_page) is child
                        ):
                            del node.children[first_page]
                elif existing.key == key and existing.pages == candidate_pages:
                    # A prior attempt delivered this transition but failed later.
                    kept = candidate_pages
            leftover = tuple(page for page in candidates if page not in kept)
            self._release(allocation)
            if leftover:
                self._pool.free(leftover)
        except Exception:
            self._abort_release(allocation)
            raise
        self._finish_release(allocation)

    def allocate_private_page(self) -> int:
        """Allocate one non-shared page (decode growth beyond the prompt allocation)."""
        if not self._ensure_free(1):
            raise KVCacheFull("no free or evictable page for decode growth")
        return self._pool.allocate(1)[0]

    def free_private_pages(self, pages: tuple[int, ...]) -> None:
        self._pool.free(pages)

    def release_preempted(
        self, allocation: KVAllocation, decode_pages: tuple[int, ...] = ()
    ) -> None:
        """Release a preempted/aborted request WITHOUT marking KV computed.

        The request did not complete, so its uncomputed tree node (if any)
        stays unmatched until LRU eviction reclaims it; private pages return
        to the pool immediately.
        """
        if not self._begin_release(allocation):
            return
        try:
            self._unlock_path(allocation._node)
            if not allocation._tree_inserted and allocation.new_full_pages:
                self._pool.free(allocation.new_full_pages)
            loose = (
                ((allocation.tail_page,) if allocation.tail_page is not None else ())
                + tuple(decode_pages)
            )
            if loose:
                self._pool.free(loose)
        except Exception:
            self._abort_release(allocation)
            raise
        self._finish_release(allocation)

    def pin(
        self,
        session_id: str,
        tokens: tuple[int, ...],
        ttl_allocations: int | None = None,
    ) -> None:
        """Hold the matched prefix path against eviction (orchestration sessions).

        ``ttl_allocations`` bounds the pin's lifetime in allocate() calls so
        abandoned sessions drain instead of pinning pages forever.
        """
        if session_id in self._pins:
            raise ValueError(f"session {session_id!r} is already pinned")
        _, _, node = self._match_and_lock(tuple(tokens))
        expiry = self._alloc_tick + ttl_allocations if ttl_allocations is not None else None
        self._pins[session_id] = (node, expiry)

    def unpin(self, session_id: str) -> None:
        entry = self._pins.pop(session_id, None)
        if entry is None:
            raise ValueError(f"session {session_id!r} is not pinned")
        self._unlock_path(entry[0])
