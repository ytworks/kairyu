#!/usr/bin/env python3
"""Reproducible long-prefix KV event hash benchmark for issue #220.

Run:
    uv run python bench/kv_event_hash_bench.py --tokens 32768 --repeats 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

from kairyu.engine.core.radix_kv import _extend_prefix_hash_chain


def _legacy_hashes(
    token_ids: tuple[int, ...], page_size: int
) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(repr(token_ids[:end]).encode()).hexdigest()[:16]
        for end in range(page_size, len(token_ids) + 1, page_size)
    )


def _incremental_hashes(
    token_ids: tuple[int, ...], page_size: int
) -> tuple[str, ...]:
    _hasher, _count, hashes, _tier_keys = _extend_prefix_hash_chain(
        hashlib.sha256(b"("),
        0,
        token_ids,
        page_size,
        include_tier_keys=False,
    )
    return hashes


def _measure(fn, repeats: int) -> dict[str, float | int]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32_768)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.tokens < 1 or args.page_size < 1 or args.repeats < 1:
        parser.error("tokens, page-size, and repeats must be positive")
    if args.tokens % args.page_size:
        parser.error("tokens must be divisible by page-size")

    token_ids = tuple(
        (index * 104_729) % 2_000_003 - 1_000_001
        for index in range(args.tokens)
    )
    expected = _legacy_hashes(token_ids, args.page_size)
    actual = _incremental_hashes(token_ids, args.page_size)
    if actual != expected:
        raise AssertionError("incremental hash chain is not legacy-compatible")

    legacy = _measure(
        lambda: _legacy_hashes(token_ids, args.page_size),
        args.repeats,
    )
    incremental = _measure(
        lambda: _incremental_hashes(token_ids, args.page_size),
        args.repeats,
    )
    result = {
        "tokens": args.tokens,
        "page_size": args.page_size,
        "blocks": args.tokens // args.page_size,
        "legacy": legacy,
        "incremental": incremental,
        "median_speedup": (
            float(legacy["median_seconds"])
            / float(incremental["median_seconds"])
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
