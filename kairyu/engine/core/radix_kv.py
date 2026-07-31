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

import hashlib
import heapq
from dataclasses import dataclass, field

from kairyu.engine.core.pages import PagePool


class KVCacheFull(MemoryError):
    """Not enough free or evictable pages to admit the request."""


def _snapshot_prefix_hash(prefix_hasher, token_count: int) -> str:
    snapshot = prefix_hasher.copy()
    # repr((token,)) carries a trailing comma; every longer tuple does not.
    if token_count == 1:
        snapshot.update(b",")
    snapshot.update(b")")
    return snapshot.hexdigest()[:16]


def _snapshot_prefix_digest(prefix_hasher, token_count: int) -> str:
    """Full-strength identity for one page-aligned prefix.

    Fleet routing keeps the historical 64-bit display hash above.  A DRAM
    payload key is an integrity boundary, however: a collision must never make
    one prefix restore another prefix's KV.  The tier therefore uses the full
    SHA-256 while public KV events retain their established compact identity.
    """

    snapshot = prefix_hasher.copy()
    if token_count == 1:
        snapshot.update(b",")
    snapshot.update(b")")
    return snapshot.hexdigest()


def _extend_prefix_hash_chain(
    prefix_hasher,
    prefix_token_count: int,
    token_ids: tuple[int, ...],
    page_size: int,
    *,
    include_tier_keys: bool = True,
):
    """Extend compatible repr(tuple(prefix)) block hashes in one pass."""
    extended = prefix_hasher.copy()
    token_count = prefix_token_count
    hashes: list[str] = []
    digests: list[str] = []
    for offset, token_id in enumerate(token_ids, start=1):
        if token_count:
            extended.update(b", ")
        extended.update(repr(token_id).encode())
        token_count += 1
        if offset % page_size == 0:
            hashes.append(_snapshot_prefix_hash(extended, token_count))
            if include_tier_keys:
                digests.append(_snapshot_prefix_digest(extended, token_count))
    return extended, token_count, tuple(hashes), tuple(digests)


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
        "evicting",
        "prefix_hasher",
        "prefix_token_count",
        "block_hashes",
        "tier_keys",
        "eviction_version",
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
        # A selected eviction victim remains in the tree until BlockRemoved is
        # accepted, but its PagePool pages are already promised to the outer
        # allocation.  Reentrant event sinks must not match or pin those pages.
        self.evicting = False
        # Event-enabled caches maintain the canonical repr(tuple(prefix))
        # SHA-256 stream without its closing punctuation. A copy can append the
        # tuple terminator at each block boundary without rebuilding prefixes.
        self.prefix_hasher = None
        self.prefix_token_count = 0
        self.block_hashes: tuple[str, ...] = ()
        self.tier_keys: tuple[str, ...] = ()
        # Every eligibility/LRU transition invalidates older heap entries.
        self.eviction_version = 0


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
        # m10b D7: optional KV-event sink (BlockStored/BlockRemoved)
        self._event_sink = event_sink
        self._pool = PagePool(num_pages)
        self._num_pages = num_pages
        self._page_size = page_size
        self._clock = 0
        self._eviction_sequence = 0
        self._eviction_heap: list[tuple[int, int, int, _Node]] = []
        self._evictable_index: dict[_Node, tuple[int, int]] = {}
        # The full prefix chain has a measurable per-token cost.  Keep the
        # historical zero-tier path free of that work; an event sink needs the
        # compact hashes immediately, while a DRAM tier enables the same chain
        # before the first allocation.
        self._hash_chain_enabled = event_sink is not None
        self._tier_hash_chain_enabled = False
        self._root = _Node(key=(), pages=(), parent=None)
        self._initialize_hash_chain(self._root)
        self._root.ref_count = 1  # never evictable
        self._root.computed = True
        self._alloc_tick = 0
        self._pins: dict[str, tuple[_Node, int | None]] = {}  # node, expiry tick
        self._hit_tokens = 0
        self._total_tokens = 0
        self._dram_tier = None
        self._dram_min_restore_tokens = 0
        self._dram_stats = {
            "offload_pages": 0,
            "offload_bypassed_pages": 0,
            "restore_pages": 0,
            "restore_attempts": 0,
            "restore_fallbacks": 0,
            "ownership_failures": 0,
        }

    def attach_dram_tier(self, tier, *, min_restore_tokens: int) -> None:
        """Attach one physical DRAM tier after its ``PagedKVPool`` exists.

        Radix owns logical page ids while the runner owns the tensors indexed by
        them, so construction is necessarily two-stage.  Replacement is
        forbidden: outstanding host entries are fingerprinted to the original
        pool/rank and cannot safely migrate to another tier implicitly.
        """

        if self._dram_tier is not None:
            raise ValueError("a DRAM KV tier is already attached")
        if tier is None:
            raise TypeError("DRAM KV tier must not be None")
        if (
            isinstance(min_restore_tokens, bool)
            or not isinstance(min_restore_tokens, int)
            or min_restore_tokens < self._page_size
            or min_restore_tokens % self._page_size
        ):
            raise ValueError(
                "min_restore_tokens must be a positive page-aligned token count"
            )
        if self._total_tokens or self._root.children:
            raise ValueError("a DRAM KV tier must be attached before the cache is used")
        self._hash_chain_enabled = True
        self._tier_hash_chain_enabled = True
        self._initialize_hash_chain(self._root)
        self._dram_tier = tier
        self._dram_min_restore_tokens = min_restore_tokens

    @property
    def dram_tier_stats(self) -> dict[str, int]:
        """A snapshot of ownership transitions, never the mutable counters."""

        return dict(self._dram_stats)
    @property
    def num_free_pages(self) -> int:
        return self._pool.num_free

    @property
    def hit_rate(self) -> float:
        if self._total_tokens == 0:
            return 0.0
        return self._hit_tokens / self._total_tokens

    def peek_cached_tokens(self, tokens: tuple[int, ...]) -> int:
        """Return the computed page-aligned prefix length without mutation.

        Admission policy occasionally needs to distinguish expensive uncached
        prefill from a cheap radix hit before it allocates or locks pages. A
        normal allocation walk splits nodes, updates LRU state, refcounts the
        path, and contributes to hit-rate accounting; this probe deliberately
        does none of those things.
        """

        node = self._root
        matched_tokens = 0
        position = 0
        while True:
            first_page = tuple(tokens[position : position + self._page_size])
            if len(first_page) < self._page_size:
                return matched_tokens
            child = node.children.get(first_page)
            if child is None or not child.computed or child.evicting:
                return matched_tokens
            whole_pages = len(child.key) // self._page_size
            common_pages = 0
            for page_index in range(whole_pages):
                start = page_index * self._page_size
                segment = tokens[
                    position + start : position + start + self._page_size
                ]
                if tuple(segment) != child.key[start : start + self._page_size]:
                    break
                common_pages += 1
            matched_tokens += common_pages * self._page_size
            if common_pages < whole_pages:
                return matched_tokens
            position += len(child.key)
            node = child

    def _touch(self, node: _Node) -> None:
        self._clock += 1
        node.last_access = self._clock
        if node in self._evictable_index or self._is_evictable(node):
            self._refresh_evictable(node)

    def _is_evictable(self, node: _Node) -> bool:
        return (
            node is not self._root
            and not node.children
            and node.ref_count == 0
            and not node.evicting
        )

    def _maybe_compact_eviction_heap(self) -> None:
        if len(self._eviction_heap) <= 64 or len(
            self._eviction_heap
        ) <= 2 * len(self._evictable_index):
            return
        self._eviction_heap = [
            (node.last_access, sequence, version, node)
            for node, (version, sequence) in self._evictable_index.items()
        ]
        heapq.heapify(self._eviction_heap)

    def _refresh_evictable(self, node: _Node) -> None:
        """Invalidate stale heap state and index the node if eligible now."""
        node.eviction_version += 1
        self._evictable_index.pop(node, None)
        if self._is_evictable(node):
            self._eviction_sequence += 1
            sequence = self._eviction_sequence
            self._evictable_index[node] = (
                node.eviction_version,
                sequence,
            )
            heapq.heappush(
                self._eviction_heap,
                (
                    node.last_access,
                    sequence,
                    node.eviction_version,
                    node,
                ),
            )
        self._maybe_compact_eviction_heap()

    def _invalidate_evictable(self, node: _Node) -> None:
        node.eviction_version += 1
        self._evictable_index.pop(node, None)
        self._maybe_compact_eviction_heap()

    def _pop_oldest_evictable(self) -> _Node | None:
        while self._eviction_heap:
            last_access, sequence, version, node = heapq.heappop(
                self._eviction_heap
            )
            if self._evictable_index.get(node) != (version, sequence):
                continue
            if node.last_access != last_access or not self._is_evictable(node):
                self._invalidate_evictable(node)
                continue
            del self._evictable_index[node]
            return node
        return None

    def _initialize_hash_chain(self, node: _Node) -> None:
        """Derive one node's compatible block hashes from its parent once."""
        if not self._hash_chain_enabled:
            return
        if node.parent is None:
            node.prefix_hasher = hashlib.sha256(b"(")
            node.prefix_token_count = 0
            node.block_hashes = ()
            node.tier_keys = ()
            return

        parent = node.parent
        assert parent.prefix_hasher is not None
        (
            node.prefix_hasher,
            node.prefix_token_count,
            node.block_hashes,
            node.tier_keys,
        ) = _extend_prefix_hash_chain(
            parent.prefix_hasher,
            parent.prefix_token_count,
            node.key,
            self._page_size,
            include_tier_keys=self._tier_hash_chain_enabled,
        )

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
        self._initialize_hash_chain(upper)
        retained_hashes = node.block_hashes[:keep_pages]
        retained_tier_keys = node.tier_keys[:keep_pages]
        if upper.block_hashes != retained_hashes:
            raise AssertionError("split prefix hash diverged from stored chain")
        if upper.tier_keys != retained_tier_keys:
            raise AssertionError("split DRAM identity diverged from stored chain")
        parent = node.parent
        assert parent is not None  # root is never split (its key is empty)
        parent.children[upper.key[: self._page_size]] = upper
        node.key = node.key[split_tokens:]
        node.pages = node.pages[keep_pages:]
        node.block_hashes = node.block_hashes[keep_pages:]
        node.tier_keys = node.tier_keys[keep_pages:]
        node.parent = upper
        upper.children[node.key[: self._page_size]] = node
        self._refresh_evictable(parent)
        self._refresh_evictable(upper)
        self._refresh_evictable(node)
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
            if child is None or not child.computed or child.evicting:
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
            self._refresh_evictable(current)
            current = current.parent

    def _ensure_free(self, needed: int, *, offload_to_dram: bool = True) -> bool:
        while self._pool.num_free < needed:
            victim = self._pop_oldest_evictable()
            if victim is None:
                return False
            victim.evicting = True
            try:
                if (
                    offload_to_dram
                    and self._dram_tier is not None
                    and victim.computed
                ):
                    try:
                        retained = self._dram_tier.offload(
                            victim.tier_keys,
                            victim.pages,
                        )
                    except BaseException as error:
                        # Capacity pressure and a physically settled D2H failure
                        # are cache misses, not engine failures: the HBM victim
                        # is still safe to evict and will be recomputed later.
                        # Unknown DMA completion is different and must retain the
                        # source pages indefinitely.
                        if getattr(error, "source_reusable", False) is not True:
                            raise
                        retained = False
                    if retained:
                        self._dram_stats["offload_pages"] += len(victim.pages)
                    else:
                        self._dram_stats["offload_bypassed_pages"] += len(
                            victim.pages
                        )
                elif self._dram_tier is not None and victim.computed:
                    self._dram_stats["offload_bypassed_pages"] += len(victim.pages)
                self._emit_removed(victim)  # the ONLY BlockRemoved source (A13)
            except BaseException:
                # The old scan found the same leaf again on retry. Restore that
                # ownership when either the tier or event sink refuses removal.
                # In particular, an unknown D2H completion must never let these
                # physical pages return to PagePool.
                self._dram_stats["ownership_failures"] += 1
                victim.evicting = False
                self._refresh_evictable(victim)
                raise
            self._pool.free(victim.pages)
            assert victim.parent is not None
            parent = victim.parent
            self._invalidate_evictable(victim)
            del parent.children[victim.key[: self._page_size]]
            victim.evicting = False
            self._refresh_evictable(parent)
        return True

    def _restore_from_dram(
        self,
        tokens: tuple[int, ...],
        matched_tokens: int,
        cached_pages: tuple[int, ...],
        node: _Node,
    ) -> tuple[int, tuple[int, ...], _Node]:
        """Restore a consecutive suffix and publish it only after H2D completes.

        The allocation already locks ``node`` and every ancestor.  Restored
        pages become one computed child with the same lock, so the ordinary
        allocation continuation can treat them exactly like an HBM radix hit.
        A safely settled transfer failure falls back to recompute; an unknown
        destination completion deliberately leaks/quarantines the reserved page
        ids and raises rather than exposing or recycling them.
        """

        tier = self._dram_tier
        if tier is None:
            return matched_tokens, cached_pages, node
        full_suffix_tokens = (
            (len(tokens) - matched_tokens) // self._page_size
        ) * self._page_size
        if full_suffix_tokens < self._dram_min_restore_tokens:
            return matched_tokens, cached_pages, node
        first_page = tuple(
            tokens[matched_tokens : matched_tokens + self._page_size]
        )
        # An uncomputed sibling belongs to an in-flight prefill.  Restoring a
        # second child under the same radix edge would overwrite its ownership.
        if first_page in node.children:
            return matched_tokens, cached_pages, node
        assert node.prefix_hasher is not None
        _, _, _, candidate_keys = _extend_prefix_hash_chain(
            node.prefix_hasher,
            node.prefix_token_count,
            tokens[matched_tokens : matched_tokens + full_suffix_tokens],
            self._page_size,
            include_tier_keys=True,
        )
        minimum_pages = -(
            -self._dram_min_restore_tokens // self._page_size
        )
        available = tier.available_prefix(
            candidate_keys,
            min_pages=minimum_pages,
        )
        if not isinstance(available, int) or isinstance(available, bool):
            raise TypeError("DRAM tier available_prefix must return an int")
        if not 0 <= available <= len(candidate_keys):
            raise ValueError(
                "DRAM tier available_prefix returned an out-of-range page count"
            )
        if available < minimum_pages:
            return matched_tokens, cached_pages, node
        # Protect the selected host entries while obtaining destination ids.
        # Offloading the HBM victim into a full bounded tier could evict the
        # exact DRAM pages we are about to restore, turning every swap into
        # deterministic self-thrashing. The displaced HBM victim is therefore
        # an ordinary eviction for this one page-exchange operation.
        if not self._ensure_free(available, offload_to_dram=False):
            return matched_tokens, cached_pages, node

        destination_pages = self._pool.allocate(available)
        selected_keys = candidate_keys[:available]
        self._dram_stats["restore_attempts"] += 1
        try:
            restored = tier.restore(selected_keys, destination_pages)
        except BaseException as error:
            reusable = getattr(error, "destination_reusable", False) is True
            if reusable:
                self._pool.free(destination_pages)
                self._dram_stats["restore_fallbacks"] += 1
                return matched_tokens, cached_pages, node
            self._dram_stats["ownership_failures"] += 1
            # Unknown H2D completion: PagePool intentionally never sees these
            # ids again. They remain quarantined for this cache's lifetime.
            raise
        if restored is not True:
            self._pool.free(destination_pages)
            self._dram_stats["restore_fallbacks"] += 1
            return matched_tokens, cached_pages, node

        restored_tokens = available * self._page_size
        key = tokens[matched_tokens : matched_tokens + restored_tokens]
        child = _Node(key=key, pages=destination_pages, parent=node)
        self._initialize_hash_chain(child)
        if child.tier_keys != selected_keys:
            # This is an internal identity error, not a transfer failure.  The
            # destination copy is physically complete and can be recycled.
            self._pool.free(destination_pages)
            raise AssertionError("restored DRAM identities differ from radix keys")
        child.ref_count = 1
        child.publishing = True
        self._touch(child)
        node.children[first_page] = child
        self._refresh_evictable(node)
        try:
            self._emit_stored(child)
            child.computed = True
        except BaseException:
            del node.children[first_page]
            self._invalidate_evictable(child)
            self._refresh_evictable(node)
            self._pool.free(destination_pages)
            raise
        finally:
            child.publishing = False
        self._refresh_evictable(child)
        self._dram_stats["restore_pages"] += available
        return (
            matched_tokens + restored_tokens,
            cached_pages + destination_pages,
            child,
        )

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
        try:
            matched_tokens, cached_pages, node = self._restore_from_dram(
                tokens,
                matched_tokens,
                cached_pages,
                node,
            )
        except BaseException:
            self._unlock_path(node)
            raise
        suffix_len = len(tokens) - matched_tokens
        full_pages_needed = suffix_len // self._page_size
        tail_tokens = suffix_len % self._page_size
        needed = full_pages_needed + (1 if tail_tokens else 0)
        try:
            has_capacity = self._ensure_free(needed)
        except BaseException:
            # A successful DRAM restore extends the locked match path before
            # suffix admission.  If a later eviction/offload raises, release
            # that updated path just like every other failed allocation while
            # leaving the victim's unknown physical ownership untouched.
            self._unlock_path(node)
            raise
        if not has_capacity:
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
                self._initialize_hash_chain(child)
                child.ref_count = 1
                self._touch(child)
                node.children[key[: self._page_size]] = child
                self._refresh_evictable(node)
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

    def _emit_stored(self, node) -> None:
        """BlockStored on the computed False->True transition (m10b A13)."""
        if self._event_sink is None:
            return
        parent = node.parent
        assert parent is not None
        parent_hash = parent.block_hashes[-1] if parent.block_hashes else None
        self._event_sink(
            {
                "type": "BlockStored",
                "block_hashes": list(node.block_hashes),
                "parent_block_hash": parent_hash,
                "token_ids": list(node.key),
                "block_size": self._page_size,
            }
        )

    def _emit_removed(self, node) -> None:
        if self._event_sink is None:
            return
        self._event_sink(
            {"type": "BlockRemoved", "block_hashes": list(node.block_hashes)}
        )

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
                    self._initialize_hash_chain(child)
                    # Keep an uncomputed, in-delivery child invisible to matching
                    # and protected from reentrant eviction until the sink accepts it.
                    child.ref_count = 1
                    child.publishing = True
                    self._touch(child)
                    node.children[first_page] = child
                    self._refresh_evictable(node)
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
                            self._invalidate_evictable(child)
                            self._refresh_evictable(node)
                        else:
                            self._refresh_evictable(child)
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

    def reserve_scratch_page(self) -> int:
        """Take one page PERMANENTLY out of circulation (m17 A5).

        A captured CUDA graph replays a fixed bucket size: the padding rows past
        the real batch still execute their KV write, and it lands on whichever
        page id their page table holds. That id must therefore be one this cache
        can never hand out again — otherwise every partial-bucket replay
        overwrites a live (or about-to-be-allocated) request's K/V.

        Unlike ``allocate_private_page`` the page is never returned: the capture
        outlives every request, so ``free_private_pages`` must not be called on
        it. Capacity drops by exactly one page for the process lifetime.
        """
        if not self._ensure_free(1):
            raise KVCacheFull("no free or evictable page to reserve as graph scratch")
        return self._pool.allocate(1)[0]

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
