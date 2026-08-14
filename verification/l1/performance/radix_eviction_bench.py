#!/usr/bin/env python3
"""Reproducible large-tree RadixKV eviction benchmark for issue #214.

Run:
    uv run python verification/l1/performance/radix_eviction_bench.py \
        --leaves 100000 --evictions 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from kairyu.engine.core.radix_kv import RadixKVCache


class _ScanningRadixKVCache(RadixKVCache):
    """Exact pre-#214 selection: collect every leaf, then take min(LRU)."""

    def _pop_oldest_evictable(self):
        leaves = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is not self._root and not node.children and node.ref_count == 0:
                leaves.append(node)
        if not leaves:
            return None
        victim = min(leaves, key=lambda leaf: leaf.last_access)
        self._evictable_index.pop(victim, None)
        return victim


def _build(cache_type, leaves: int) -> RadixKVCache:
    cache = cache_type(num_pages=leaves, page_size=1)
    for token_id in range(leaves):
        allocation = cache.allocate((token_id,))
        cache.mark_computed(allocation)
        cache.free(allocation)
    return cache


def _measure(
    cache_type,
    leaves: int,
    evictions: int,
    repeats: int,
) -> dict[str, float | int]:
    samples = []
    for _ in range(repeats):
        cache = _build(cache_type, leaves)
        start = time.perf_counter()
        assert cache._ensure_free(evictions)
        samples.append(time.perf_counter() - start)
        assert cache.num_free_pages == evictions
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaves", type=int, default=100_000)
    parser.add_argument("--evictions", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.leaves < 1 or args.evictions < 1 or args.repeats < 1:
        parser.error("leaves, evictions, and repeats must be positive")
    if args.evictions > args.leaves:
        parser.error("evictions cannot exceed leaves")

    scanning = _measure(
        _ScanningRadixKVCache,
        args.leaves,
        args.evictions,
        args.repeats,
    )
    indexed = _measure(
        RadixKVCache,
        args.leaves,
        args.evictions,
        args.repeats,
    )
    result = {
        "leaves": args.leaves,
        "evictions": args.evictions,
        "scanning": scanning,
        "indexed": indexed,
        "median_speedup": (float(scanning["median_seconds"]) / float(indexed["median_seconds"])),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
