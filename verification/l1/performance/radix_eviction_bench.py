#!/usr/bin/env python3
"""Reproducible large-tree RadixKV eviction benchmark for issue #214.

Timings compare indexed eviction with the pre-#214 full scan and are
diagnostics.  The ``work`` section is deterministic: at two tree sizes it
counts the nodes and index entries eviction touches (one unit per children-map
or eviction-index primitive and one per item iterated) plus every LRU-stamp
comparison, so the growth of work per eviction can be checked without a clock.

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

WORK_UNIT = (
    "children-map and eviction-index primitives (1 per call, 1 per item "
    "iterated) plus LRU-stamp comparisons"
)


class _WorkCounter:
    count = 0


def _tick() -> None:
    _WorkCounter.count += 1


def _counted_stamp(value):
    return value if value is NotImplemented else _CountedStamp(value)


class _CountedStamp(int):
    """LRU clock value whose comparisons are counted."""

    __slots__ = ()
    __hash__ = int.__hash__

    def __add__(self, other):
        return _counted_stamp(int.__add__(self, other))

    def __radd__(self, other):
        return _counted_stamp(int.__radd__(self, other))

    def __eq__(self, other: object) -> bool:
        _tick()
        return int.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        _tick()
        return int.__ne__(self, other)

    def __lt__(self, other: int) -> bool:
        _tick()
        return int.__lt__(self, other)

    def __le__(self, other: int) -> bool:
        _tick()
        return int.__le__(self, other)

    def __gt__(self, other: int) -> bool:
        _tick()
        return int.__gt__(self, other)

    def __ge__(self, other: int) -> bool:
        _tick()
        return int.__ge__(self, other)


class _CountingView:
    """Key, value, or item view that counts every item it yields."""

    __slots__ = ("_view",)

    def __init__(self, view) -> None:
        self._view = view

    def __iter__(self):
        for item in self._view:
            _tick()
            yield item

    def __len__(self) -> int:
        _tick()
        return len(self._view)

    def __contains__(self, item: object) -> bool:
        _tick()
        return item in self._view


class _CountingDict(dict):
    """Children map or eviction index that counts what eviction touches."""

    __slots__ = ()

    def __len__(self) -> int:
        _tick()
        return dict.__len__(self)

    def __contains__(self, key: object) -> bool:
        _tick()
        return dict.__contains__(self, key)

    def __getitem__(self, key):
        _tick()
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value) -> None:
        _tick()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key) -> None:
        _tick()
        dict.__delitem__(self, key)

    def __iter__(self):
        for key in dict.__iter__(self):
            _tick()
            yield key

    def get(self, key, default=None):
        _tick()
        return dict.get(self, key, default)

    def pop(self, key, *default):
        _tick()
        return dict.pop(self, key, *default)

    def popitem(self):
        _tick()
        return dict.popitem(self)

    def setdefault(self, key, default=None):
        _tick()
        return dict.setdefault(self, key, default)

    def keys(self):
        return _CountingView(dict.keys(self))

    def values(self):
        return _CountingView(dict.values(self))

    def items(self):
        return _CountingView(dict.items(self))


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


def _build(cache_type, leaves: int, *, counted: bool = False) -> RadixKVCache:
    cache = cache_type(num_pages=leaves, page_size=1)
    if counted:
        cache._clock = _CountedStamp(cache._clock)
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


def _instrument(cache: RadixKVCache) -> None:
    """Count every children map and the eviction index of a built tree."""
    stack = [cache._root]
    while stack:
        node = stack.pop()
        children = list(node.children.values())
        node.children = _CountingDict(node.children)
        stack.extend(children)
    cache._evictable_index = _CountingDict(cache._evictable_index)


def _work_eviction(leaves: int, evictions: int) -> tuple[int, int]:
    cache = _build(RadixKVCache, leaves, counted=True)
    _instrument(cache)
    _WorkCounter.count = 0
    if not cache._ensure_free(evictions):
        raise AssertionError("indexed eviction could not free the pages")
    units = _WorkCounter.count
    if cache.num_free_pages != evictions:
        raise AssertionError("indexed eviction freed the wrong page count")
    return evictions, units


def _work(sizes: tuple[int, ...], evictions: int) -> dict[str, object]:
    operations: list[int] = []
    work: list[int] = []
    for leaves in sizes:
        count, units = _work_eviction(leaves, evictions)
        operations.append(count)
        work.append(units)
    return {
        "unit": WORK_UNIT,
        "eviction": {
            "sizes": list(sizes),
            "operations": operations,
            "work": work,
        },
    }


def _work_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("work sizes must be integers") from error
    if len(sizes) != 2 or not 1 <= sizes[0] < sizes[1]:
        raise argparse.ArgumentTypeError(
            "work sizes must be two increasing positive integers"
        )
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaves", type=int, default=100_000)
    parser.add_argument("--evictions", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--work-sizes",
        type=_work_sizes,
        default=(2_000, 16_000),
        help="two leaf counts for the deterministic work counts",
    )
    args = parser.parse_args()
    if args.leaves < 1 or args.evictions < 1 or args.repeats < 1:
        parser.error("leaves, evictions, and repeats must be positive")
    if args.evictions > min(args.leaves, args.work_sizes[0]):
        parser.error("evictions cannot exceed leaves or work sizes")

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
        "work": _work(args.work_sizes, args.evictions),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
