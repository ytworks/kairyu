import pytest

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


def test_second_allocation_hits_shared_prefix():
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    first = cache.allocate(_tokens([1, 2, 3, 4], [5, 6, 7, 8]))
    cache.mark_computed(first)
    second = cache.allocate(_tokens([1, 2, 3, 4], [9, 9, 9, 9]))
    assert second.cached_pages == (first.new_full_pages[0],)
    assert len(second.new_full_pages) == 1
    # 4 hit tokens out of 8+8 total
    assert cache.hit_rate == pytest.approx(4 / 16)


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
    cache.commit_and_release(allocation, output_tokens=(9, 8, 7, 6), decode_pages=(decode_page,))
    reuse = cache.allocate((1, 2, 3, 4, 9, 8, 7, 6))
    assert reuse.num_cached_tokens == 8  # prompt AND generated tokens hit


def test_commit_frees_partially_filled_decode_pages():
    cache = RadixKVCache(num_pages=2, page_size=PAGE)
    allocation = cache.allocate((1, 2, 3, 4))
    cache.mark_computed(allocation)
    decode_page = cache.allocate_private_page()
    assert cache.num_free_pages == 0
    cache.commit_and_release(allocation, output_tokens=(9, 8), decode_pages=(decode_page,))
    assert cache.num_free_pages == 1  # half-filled decode page returned to pool
