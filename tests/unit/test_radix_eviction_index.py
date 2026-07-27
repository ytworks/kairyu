"""Indexed RadixKV leaf eviction equivalence and stress (issue #214)."""

from __future__ import annotations

import random

import pytest

from kairyu.engine.core.radix_kv import RadixKVCache


class _ScanningRadixKVCache(RadixKVCache):
    """Pre-#214 full-tree victim selection used as the behavior oracle."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.full_tree_scans = 0

    def _pop_oldest_evictable(self):
        self.full_tree_scans += 1
        leaves = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if (
                node is not self._root
                and not node.children
                and node.ref_count == 0
            ):
                leaves.append(node)
        if not leaves:
            return None
        victim = min(leaves, key=lambda leaf: leaf.last_access)
        self._evictable_index.pop(victim, None)
        return victim


@pytest.mark.parametrize("seed", range(10))
def test_randomized_indexed_eviction_matches_full_tree_scan(seed: int) -> None:
    rng = random.Random(seed)
    indexed_events: list[dict] = []
    scanning_events: list[dict] = []
    indexed = RadixKVCache(
        num_pages=48,
        page_size=4,
        event_sink=indexed_events.append,
    )
    scanning = _ScanningRadixKVCache(
        num_pages=48,
        page_size=4,
        event_sink=scanning_events.append,
    )
    known_prefixes: list[tuple[int, ...]] = [()]

    for request_index in range(1_000):
        known = rng.choice(known_prefixes)
        new_pages = rng.randint(1, 6)
        max_base_pages = min(
            len(known) // 4,
            36 - new_pages,
        )
        base_pages = rng.randint(0, max_base_pages)
        base = known[: base_pages * 4]
        suffix = (request_index + seed * 10_000_000,) + tuple(
            rng.randint(-1_000_000, 1_000_000)
            for _ in range(new_pages * 4 - 1)
        )
        tail = tuple(
            rng.randint(-1_000_000, 1_000_000)
            for _ in range(rng.randrange(4))
        )
        token_ids = base + suffix + tail

        indexed_allocation = indexed.allocate(token_ids)
        scanning_allocation = scanning.allocate(token_ids)
        assert indexed_allocation.pages == scanning_allocation.pages
        assert (
            indexed_allocation.num_cached_tokens
            == scanning_allocation.num_cached_tokens
        )

        indexed.mark_computed(indexed_allocation)
        scanning.mark_computed(scanning_allocation)
        indexed.free(indexed_allocation)
        scanning.free(scanning_allocation)

        assert indexed.num_free_pages == scanning.num_free_pages
        assert indexed.hit_rate == scanning.hit_rate
        assert indexed_events == scanning_events
        known_prefixes.append(token_ids[: len(token_ids) // 4 * 4])

    assert scanning.full_tree_scans > 100


def test_repeated_hits_compact_stale_heap_entries() -> None:
    cache = RadixKVCache(num_pages=4, page_size=4)
    token_ids = (1, 2, 3, 4)
    allocation = cache.allocate(token_ids)
    cache.mark_computed(allocation)
    cache.free(allocation)

    for _ in range(10_000):
        hit = cache.allocate(token_ids)
        assert hit.num_cached_tokens == 4
        cache.free(hit)

    assert len(cache._evictable_index) == 1
    assert len(cache._eviction_heap) <= 64
    replacement = cache.allocate(tuple(range(100, 116)))
    assert replacement.num_cached_tokens == 0
    cache.release_preempted(replacement)


def test_removed_event_failure_restores_victim_for_retry() -> None:
    events: list[dict] = []
    removed_attempts = 0

    def fail_first_removed(event: dict) -> None:
        nonlocal removed_attempts
        if event["type"] == "BlockRemoved":
            removed_attempts += 1
            if removed_attempts == 1:
                raise RuntimeError("removed event failed")
        events.append(event)

    cache = RadixKVCache(
        num_pages=1,
        page_size=4,
        event_sink=fail_first_removed,
    )
    first = cache.allocate((1, 2, 3, 4))
    cache.mark_computed(first)
    cache.free(first)

    with pytest.raises(RuntimeError, match="removed event failed"):
        cache.allocate((5, 6, 7, 8))

    second = cache.allocate((5, 6, 7, 8))
    cache.mark_computed(second)
    cache.free(second)
    assert removed_attempts == 2
    assert [event["type"] for event in events].count("BlockRemoved") == 1
