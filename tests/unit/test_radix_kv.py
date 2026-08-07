import pytest

import kairyu.engine.core.radix_kv as radix_kv_module
from kairyu.engine.core.radix_kv import KVCacheFull, RadixKVCache

PAGE = 4  # tokens per page in these tests


def _tokens(*pages: list[int]) -> tuple[int, ...]:
    flat: list[int] = []
    for page in pages:
        flat.extend(page)
    return tuple(flat)


def test_first_allocation_is_all_misses():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    allocation = cache.allocate(_tokens([1, 2, 3, 4], [5, 6, 7, 8]))
    assert allocation.cached_pages == ()
    assert len(allocation.new_full_pages) == 2
    assert allocation.tail_page is None
    assert cache.hit_rate == 0.0


def test_allocation_pages_are_concatenated_once():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    allocation = cache.allocate(_tokens([1, 2, 3, 4], [5, 6, 7, 8]))

    assert allocation.pages is allocation.pages


def test_event_only_hash_chain_skips_full_dram_tier_digests(monkeypatch):
    events: list[dict] = []

    def unexpected_tier_digest(*_args, **_kwargs):
        raise AssertionError("event-only cache computed a DRAM tier digest")

    monkeypatch.setattr(
        radix_kv_module,
        "_snapshot_prefix_digest",
        unexpected_tier_digest,
    )
    cache = RadixKVCache(num_pages=2, page_size=PAGE, event_sink=events.append)
    tokens = _tokens([1, 2, 3, 4])
    allocation = cache.allocate(tokens)
    cache.mark_computed(allocation)

    node = cache._root.children[tokens]
    assert node.block_hashes
    assert node.tier_keys == ()
    assert [event["type"] for event in events] == ["BlockStored"]


def test_second_allocation_hits_shared_prefix():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    first = cache.allocate(_tokens([1, 2, 3, 4], [5, 6, 7, 8]))
    cache.mark_computed(first)
    second = cache.allocate(_tokens([1, 2, 3, 4], [9, 9, 9, 9]))
    assert second.cached_pages == (first.new_full_pages[0],)
    assert len(second.new_full_pages) == 1
    # 4 hit tokens out of 8+8 total
    assert cache.hit_rate == pytest.approx(4 / 16)


def test_cached_prefix_probe_is_read_only_and_matches_allocation():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    first = cache.allocate(_tokens([1, 2, 3, 4], [5, 6, 7, 8]))
    cache.mark_computed(first)
    cache.free(first)
    hit_rate = cache.hit_rate
    free_pages = cache.num_free_pages

    assert cache.peek_cached_tokens(
        _tokens([1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9])
    ) == 8
    assert cache.peek_cached_tokens(_tokens([1, 2, 3, 4], [7, 7, 7, 7])) == 4
    assert cache.peek_cached_tokens((1, 2, 3)) == 0
    assert cache.peek_cached_tokens(_tokens([9, 9, 9, 9])) == 0
    assert cache.hit_rate == hit_rate
    assert cache.num_free_pages == free_pages

    allocation = cache.allocate(
        _tokens([1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9])
    )
    assert allocation.num_cached_tokens == 8


def test_partial_page_tail_is_private():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    first = cache.allocate((1, 2, 3, 4, 5, 6))  # 1 full page + 2-token tail
    cache.mark_computed(first)
    assert len(first.new_full_pages) == 1
    assert first.tail_page is not None
    second = cache.allocate((1, 2, 3, 4, 5, 6))
    assert second.cached_pages == first.new_full_pages
    assert second.tail_page is not None
    assert second.tail_page != first.tail_page  # tails never shared


def test_multi_page_node_splits_on_partial_match():
    cache = RadixKVCache(num_pages=16, page_size=PAGE)
    first = cache.allocate(_tokens([1] * 4, [2] * 4, [3] * 4))
    cache.mark_computed(first)
    second = cache.allocate(_tokens([1] * 4, [2] * 4, [7] * 4))
    assert second.cached_pages == first.new_full_pages[:2]


def test_referenced_pages_survive_pressure_and_full_cache_raises():
    cache = RadixKVCache(num_pages=2, page_size=PAGE)
    held = cache.allocate(_tokens([1] * 4, [2] * 4))
    with pytest.raises(KVCacheFull):
        cache.allocate(_tokens([9] * 4,))
    assert held.new_full_pages  # still valid


def test_eviction_reclaims_freed_pages_lru_first():
    cache = RadixKVCache(num_pages=4, page_size=PAGE)
    old = cache.allocate(_tokens([1] * 4, [1, 1, 1, 2]))
    recent = cache.allocate(_tokens([5] * 4,))
    cache.free(old)
    cache.free(recent)
    cache.allocate(_tokens([5] * 4,))  # touch "recent" branch -> old becomes LRU
    # needs 3 pages: must evict the two "old" pages, keeping the recent branch
    big = cache.allocate(_tokens([6] * 4, [6, 6, 6, 7], [8] * 4))
    hit = cache.allocate(_tokens([5] * 4,))
    assert hit.cached_pages  # recent branch survived eviction
    cache.free(big)
    miss = cache.allocate(_tokens([1] * 4,))
    assert miss.cached_pages == ()  # old branch was evicted


def test_block_removed_reentry_cannot_reacquire_the_evicting_page():
    victim_tokens = _tokens([1, 2, 3, 4])
    replacement_tokens = _tokens([5] * 4, [6] * 4)
    events: list[dict] = []
    reentrant_allocations = []

    def reentrant_sink(event: dict) -> None:
        events.append(event)
        if event["type"] != "BlockRemoved":
            return
        assert cache.peek_cached_tokens(victim_tokens) == 0
        cache.pin("during-removal", victim_tokens)
        reentrant_allocations.append(cache.allocate(victim_tokens))

    cache = RadixKVCache(num_pages=2, page_size=PAGE, event_sink=reentrant_sink)
    victim = cache.allocate(victim_tokens)
    cache.mark_computed(victim)
    victim_pages = victim.new_full_pages
    cache.free(victim)

    # The sink consumes the one spare page while the original victim remains
    # reserved by the outer eviction.  The outer two-page request therefore
    # fails safely instead of sharing either physical page.
    with pytest.raises(KVCacheFull):
        cache.allocate(replacement_tokens)

    assert len(reentrant_allocations) == 1
    reentrant = reentrant_allocations[0]
    assert reentrant.num_cached_tokens == 0
    assert set(reentrant.pages).isdisjoint(victim_pages)
    assert [event["type"] for event in events].count("BlockRemoved") == 1
    cache.unpin("during-removal")
    cache.release_preempted(reentrant)
    assert cache.num_free_pages == 2


def test_block_removed_failure_restores_victim_visibility_and_retryability():
    victim_tokens = _tokens([7, 7, 7, 7])
    replacement_tokens = _tokens([8, 8, 8, 8])
    removed_attempts = 0
    accepted: list[dict] = []

    def fail_first_removed(event: dict) -> None:
        nonlocal removed_attempts
        if event["type"] == "BlockRemoved":
            removed_attempts += 1
            assert cache.peek_cached_tokens(victim_tokens) == 0
            if removed_attempts == 1:
                raise RuntimeError("removed sink failed")
        accepted.append(event)

    cache = RadixKVCache(num_pages=1, page_size=PAGE, event_sink=fail_first_removed)
    victim = cache.allocate(victim_tokens)
    cache.mark_computed(victim)
    cache.free(victim)

    with pytest.raises(RuntimeError, match="removed sink failed"):
        cache.allocate(replacement_tokens)

    assert cache.peek_cached_tokens(victim_tokens) == PAGE
    hit = cache.allocate(victim_tokens)
    assert hit.num_cached_tokens == PAGE
    cache.free(hit)

    replacement = cache.allocate(replacement_tokens)
    assert replacement.num_cached_tokens == 0
    assert removed_attempts == 2
    assert [event["type"] for event in accepted].count("BlockRemoved") == 1


def test_free_releases_tail_page_to_pool():
    cache = RadixKVCache(num_pages=2, page_size=PAGE)
    allocation = cache.allocate((1, 2, 3, 4, 5))  # 1 full + tail
    assert cache.num_free_pages == 0
    cache.free(allocation)
    assert cache.num_free_pages == 1  # tail returned; full page stays cached


def test_pinned_session_survives_eviction():
    cache = RadixKVCache(num_pages=4, page_size=PAGE)
    prefix = _tokens([1] * 4, [2] * 4)
    allocation = cache.allocate(prefix)
    cache.free(allocation)
    cache.pin("session-1", prefix)
    with pytest.raises(KVCacheFull):
        cache.allocate(_tokens([7] * 4, [8] * 4, [9] * 4))
    cache.unpin("session-1")
    cache.allocate(_tokens([7] * 4, [8] * 4, [9] * 4))  # now evictable


def test_shared_prefix_workload_hits_above_80_percent():
    """Mini version of the M2 acceptance criterion: 50% shared prefix workload."""
    cache = RadixKVCache(num_pages=256, page_size=PAGE)
    shared = tuple(range(32))  # 8 pages shared system prompt
    allocations = []
    for i in range(20):
        unique = tuple(1000 + i * PAGE + j for j in range(PAGE))  # 1 unique page each
        allocation = cache.allocate(shared + unique)
        cache.mark_computed(allocation)  # prefill completes before the next arrival
        allocations.append(allocation)
    for allocation in allocations:
        cache.free(allocation)
    assert cache.hit_rate > 0.80


def test_double_free_rejected():
    cache = RadixKVCache(num_pages=4, page_size=PAGE)
    allocation = cache.allocate((1, 2, 3, 4))
    cache.free(allocation)
    with pytest.raises(ValueError, match="freed"):
        cache.free(allocation)


def test_preempted_uncomputed_pages_are_discarded_before_cached_prefix() -> None:
    cache = RadixKVCache(num_pages=3, page_size=PAGE)
    cold_tokens = tuple(range(101, 105))
    cold = cache.allocate(cold_tokens)
    cache.mark_computed(cold)
    cache.free(cold)
    victim = cache.allocate(tuple(range(1, 9)))
    assert cache.num_free_pages == 0

    cache.release_preempted(victim)

    assert cache.num_free_pages == 2
    assert cache.peek_cached_tokens(cold_tokens) == PAGE


def test_preempted_suffix_discard_preserves_its_cached_parent() -> None:
    cache = RadixKVCache(num_pages=3, page_size=PAGE)
    parent_tokens = tuple(range(1, 5))
    parent = cache.allocate(parent_tokens)
    cache.mark_computed(parent)
    cache.free(parent)
    full_tokens = tuple(range(1, 9))
    suffix = cache.allocate(full_tokens)
    assert suffix.num_cached_tokens == PAGE

    cache.release_preempted(suffix)

    assert cache.peek_cached_tokens(full_tokens) == PAGE
    assert cache.num_free_pages == 2


def test_private_pages_for_decode_growth():
    cache = RadixKVCache(num_pages=2, page_size=PAGE)
    page = cache.allocate_private_page()
    assert cache.num_free_pages == 1
    cache.free_private_pages((page,))
    assert cache.num_free_pages == 2


def test_private_page_allocation_evicts_when_needed():
    cache = RadixKVCache(num_pages=1, page_size=PAGE)
    allocation = cache.allocate((1, 2, 3, 4))
    cache.free(allocation)  # page stays cached with refcount 0
    page = cache.allocate_private_page()  # must evict the cached page
    assert page is not None
    with pytest.raises(KVCacheFull):
        cache.allocate_private_page()


def test_multiturn_history_reuse_hits_above_80_percent():
    """Multi-turn shape: shared system prompt + growing per-session history."""
    cache = RadixKVCache(num_pages=2048, page_size=PAGE)
    system_prompt = tuple(range(64))
    histories: dict[int, tuple[int, ...]] = {s: () for s in range(8)}
    for turn in range(6):
        for session in range(8):
            new_turn = tuple(
                1_000_000 + session * 100_000 + turn * 1_000 + j for j in range(16)
            )
            prompt = system_prompt + histories[session] + new_turn
            cache.free(cache.allocate(prompt))
            histories[session] = histories[session] + new_turn
    assert cache.hit_rate > 0.80


def test_allocation_reports_cached_token_count():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    first = cache.allocate(_tokens([1] * 4, [2] * 4))
    cache.free(first)
    second = cache.allocate(_tokens([1] * 4, [2] * 4, [3] * 4))
    assert second.num_cached_tokens == 8


def test_uncomputed_pages_are_not_matched_by_later_requests():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    first = cache.allocate(_tokens([1] * 4, [2] * 4))  # KV not computed yet
    second = cache.allocate(_tokens([1] * 4, [2] * 4))
    assert second.num_cached_tokens == 0  # must not share garbage KV
    cache.mark_computed(first)
    third = cache.allocate(_tokens([1] * 4, [2] * 4))
    assert third.num_cached_tokens == 8


def test_commit_and_release_folds_outputs_into_cache():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    allocation = cache.allocate((1, 2, 3, 4))  # exactly one page, no tail
    cache.mark_computed(allocation)
    decode_page = cache.allocate_private_page()
    # Five outputs: the decode page (positions 4-7) is fully KV-written; the
    # fifth token (5) lands in a partial tail page whose KV is not yet written.
    cache.commit_and_release(allocation, output_tokens=(9, 8, 7, 6, 5), decode_pages=(decode_page,))
    reuse = cache.allocate((1, 2, 3, 4, 9, 8, 7, 6, 5))
    assert reuse.num_cached_tokens == 8  # prompt AND fully-written decode page hit


def test_commit_does_not_cache_final_unwritten_token_on_page_boundary():
    # Regression (C1 radix poisoning): the decode loop writes the KV of the
    # *previous* token each step, so the last sampled token's KV is never
    # written. When (prompt + outputs) lands exactly on a page boundary, that
    # final page must NOT be folded as computed — otherwise the next turn's
    # prompt matches it as cached and reuses a garbage KV slot for that token.
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    allocation = cache.allocate((1, 2, 3, 4))  # one full, computed prompt page
    cache.mark_computed(allocation)
    decode_page = cache.allocate_private_page()
    # Four outputs exactly fill one decode page; the fourth (token 6) has no KV.
    cache.commit_and_release(allocation, output_tokens=(9, 8, 7, 6), decode_pages=(decode_page,))
    reuse = cache.allocate((1, 2, 3, 4, 9, 8, 7, 6))
    # Only the prompt page is safely cached; the boundary decode page is not.
    assert reuse.num_cached_tokens == 4


def test_commit_frees_partially_filled_decode_pages():
    cache = RadixKVCache(num_pages=2, page_size=PAGE)
    allocation = cache.allocate((1, 2, 3, 4))
    cache.mark_computed(allocation)
    decode_page = cache.allocate_private_page()
    assert cache.num_free_pages == 0
    cache.commit_and_release(allocation, output_tokens=(9, 8), decode_pages=(decode_page,))
    assert cache.num_free_pages == 1  # half-filled decode page returned to pool
