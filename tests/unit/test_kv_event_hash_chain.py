"""Incremental radix KV event hash compatibility and stress (issue #220)."""

from __future__ import annotations

import hashlib
import random

import pytest

from kairyu.engine.core.radix_kv import RadixKVCache


def _legacy_hashes(
    token_ids: tuple[int, ...], page_size: int
) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(repr(token_ids[:end]).encode()).hexdigest()[:16]
        for end in range(page_size, len(token_ids) + 1, page_size)
    )


@pytest.mark.parametrize("token_id", [-10, -1, 0, 7, 123456789])
def test_singleton_tuple_hash_keeps_legacy_trailing_comma(token_id: int) -> None:
    events: list[dict] = []
    cache = RadixKVCache(num_pages=2, page_size=1, event_sink=events.append)
    allocation = cache.allocate((token_id,))

    cache.mark_computed(allocation)

    assert events[0]["block_hashes"] == list(
        _legacy_hashes((token_id,), page_size=1)
    )
    assert events[0]["parent_block_hash"] is None
    cache.free(allocation)


@pytest.mark.parametrize("page_size", [1, 2, 4, 7])
@pytest.mark.parametrize("seed", range(5))
def test_randomized_branch_and_split_events_match_legacy_hashes(
    page_size: int, seed: int
) -> None:
    rng = random.Random(seed)
    events: list[dict] = []
    cache = RadixKVCache(
        num_pages=8_192,
        page_size=page_size,
        event_sink=events.append,
    )
    prefixes: list[tuple[int, ...]] = [()]

    for _ in range(200):
        known = rng.choice(prefixes)
        known_pages = len(known) // page_size
        # Cutting through an existing multi-page node forces radix splits;
        # appending fresh pages then exercises child hash-state inheritance.
        base_pages = rng.randint(0, known_pages)
        base = known[: base_pages * page_size]
        new_pages = rng.randint(1, 6)
        new_tokens = tuple(
            rng.randint(-1_000_000, 1_000_000)
            for _ in range(new_pages * page_size)
        )
        tail = tuple(
            rng.randint(-1_000_000, 1_000_000)
            for _ in range(rng.randrange(page_size))
        )
        token_ids = base + new_tokens + tail
        safe_prefix = token_ids[: len(token_ids) // page_size * page_size]
        before = len(events)

        allocation = cache.allocate(token_ids)
        cache.mark_computed(allocation)
        cache.free(allocation)

        if len(events) == before:
            continue
        event = events[-1]
        assert event["type"] == "BlockStored"
        expected = _legacy_hashes(safe_prefix, page_size)
        published_count = len(event["block_hashes"])
        assert event["block_hashes"] == list(expected[-published_count:])
        expected_parent = (
            expected[-published_count - 1]
            if published_count < len(expected)
            else None
        )
        assert event["parent_block_hash"] == expected_parent
        assert event["token_ids"] == list(
            safe_prefix[-published_count * page_size :]
        )
        prefixes.append(safe_prefix)


def test_decode_extension_hashes_continue_prompt_chain_exactly() -> None:
    events: list[dict] = []
    cache = RadixKVCache(num_pages=8, page_size=4, event_sink=events.append)
    prompt = tuple(range(8))
    allocation = cache.allocate(prompt)
    cache.mark_computed(allocation)
    decode_page = cache.allocate_private_page()
    outputs = (100, 101, 102, 103, 104)

    cache.commit_and_release(
        allocation,
        output_tokens=outputs,
        decode_pages=(decode_page,),
    )

    assert len(events) == 2
    expected = _legacy_hashes(prompt + outputs[:4], page_size=4)
    assert events[1]["block_hashes"] == [expected[-1]]
    assert events[1]["parent_block_hash"] == expected[-2]
    assert events[1]["token_ids"] == list(outputs[:4])


def test_randomized_eviction_only_removes_exactly_stored_hashes() -> None:
    rng = random.Random(220)
    events: list[dict] = []
    cache = RadixKVCache(num_pages=24, page_size=4, event_sink=events.append)
    seen = 0
    stored_hashes: set[str] = set()
    removed = 0

    for request_index in range(200):
        page_count = rng.randint(1, 6)
        token_ids = (request_index + 10_000,) + tuple(
            rng.randint(-1_000_000, 1_000_000)
            for _ in range(page_count * 4 - 1)
        )
        allocation = cache.allocate(token_ids)
        cache.mark_computed(allocation)
        cache.free(allocation)

        for event in events[seen:]:
            if event["type"] == "BlockStored":
                assert event["block_hashes"] == list(
                    _legacy_hashes(token_ids, page_size=4)
                )
                stored_hashes.update(event["block_hashes"])
            else:
                removed += 1
                assert set(event["block_hashes"]) <= stored_hashes
        seen = len(events)

    assert removed > 100
