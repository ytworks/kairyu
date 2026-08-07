"""#187: Radix publication and physical-page ownership around a DRAM tier."""

from __future__ import annotations

import pytest

from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler


class _RestoreError(RuntimeError):
    def __init__(self, *, destination_reusable: bool) -> None:
        super().__init__("injected restore failure")
        self.destination_reusable = destination_reusable


class _OffloadError(RuntimeError):
    def __init__(self, *, source_reusable: bool) -> None:
        super().__init__("injected offload failure")
        self.source_reusable = source_reusable


class _FakeTier:
    def __init__(self) -> None:
        self.entries: dict[str, int] = {}
        self.calls: list[tuple[str, tuple[str, ...], tuple[int, ...]]] = []
        self.offload_error: BaseException | None = None
        self.restore_error: BaseException | None = None
        self.restore_calls = 0

    def available_prefix(
        self,
        keys: tuple[str, ...],
        *,
        min_pages: int = 1,
    ) -> int:
        count = 0
        for key in keys:
            if key not in self.entries:
                break
            count += 1
        return count if count >= min_pages else 0

    def offload(
        self,
        keys: tuple[str, ...],
        page_ids: tuple[int, ...],
    ) -> bool:
        self.calls.append(("offload", keys, page_ids))
        if self.offload_error is not None:
            raise self.offload_error
        assert len(keys) == len(page_ids)
        self.entries.update(zip(keys, page_ids, strict=True))
        return True

    def restore(
        self,
        keys: tuple[str, ...],
        page_ids: tuple[int, ...],
    ) -> bool:
        self.calls.append(("restore", keys, page_ids))
        self.restore_calls += 1
        if self.restore_error is not None:
            raise self.restore_error
        if any(key not in self.entries for key in keys):
            return False
        for key in keys:
            del self.entries[key]
        return True


def _publish(cache: RadixKVCache, tokens: tuple[int, ...]):
    allocation = cache.allocate(tokens)
    cache.mark_computed(allocation)
    cache.free(allocation)
    return allocation


def test_eviction_offloads_before_the_physical_page_is_reused() -> None:
    cache = RadixKVCache(num_pages=1, page_size=2)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=2)

    first = _publish(cache, (1, 2))
    second = cache.allocate((3, 4))

    operation, keys, source_pages = tier.calls[0]
    assert operation == "offload"
    assert len(keys) == 1
    assert len(keys[0]) == 64
    assert source_pages == first.pages
    assert second.new_full_pages == first.pages
    assert cache.dram_tier_stats["offload_pages"] == 1


def test_restore_is_a_cached_hit_only_after_the_copy_succeeds() -> None:
    cache = RadixKVCache(num_pages=1, page_size=2)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=2)

    first = _publish(cache, (1, 2))
    _publish(cache, (3, 4))  # evicts first into DRAM
    restored = cache.allocate((1, 2))

    assert restored.cached_pages == first.pages
    assert restored.new_full_pages == ()
    assert restored.num_cached_tokens == 2
    assert tier.restore_calls == 1
    assert cache.dram_tier_stats["restore_pages"] == 1
    assert cache.peek_cached_tokens((1, 2)) == 2


def test_policy_restores_the_available_prefix_above_the_crossover_only() -> None:
    cache = RadixKVCache(num_pages=4, page_size=2)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=4)

    _publish(cache, (1, 2, 3, 4, 5, 6))
    _publish(cache, (11, 12, 13, 14, 15, 16))  # stores the three-page prefix
    allocation = cache.allocate((1, 2, 3, 4, 5, 6, 7, 8))

    assert allocation.num_cached_tokens == 6
    assert len(allocation.cached_pages) == 3
    assert len(allocation.new_full_pages) == 1

    cache.release_preempted(allocation)
    one_page = cache.allocate((11, 12))
    assert one_page.num_cached_tokens == 0


def test_scheduler_margin_charges_pages_restored_from_dram() -> None:
    cache = RadixKVCache(num_pages=4, page_size=4)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=4)
    first, second = (1, 2, 3, 4), (5, 6, 7, 8)
    _publish(cache, first)
    _publish(cache, second)
    eviction = cache.allocate(tuple(range(100, 116)))
    cache.release_preempted(eviction)
    pinned = tuple(range(20, 28))
    _publish(cache, pinned)
    cache.pin("pinned", pinned)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=2,
        max_num_seqs=2,
        max_num_partial_prefills=2,
    )
    scheduler.add_request(EngineRequest("a", first, max_new_tokens=2))
    scheduler.add_request(EngineRequest("b", second, max_new_tokens=2))

    plan = scheduler.schedule()

    assert [chunk.request_id for chunk in plan.scheduled] == ["a"]
    assert scheduler.waiting_ids == ("b",)
    scheduler.update({"a": 1000})
    assert scheduler.schedule().scheduled[0].request_id == "a"


def test_failed_dram_restore_reduces_scheduler_completion_margin() -> None:
    cache = RadixKVCache(num_pages=6, page_size=5)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=5)
    first = tuple(range(1, 11))
    second = tuple(range(21, 31))
    short = tuple(range(41, 46))
    _publish(cache, first)
    _publish(cache, second)
    _publish(cache, short)
    eviction = cache.allocate(tuple(range(101, 131)))
    cache.release_preempted(eviction)
    pinned = tuple(range(201, 211))
    _publish(cache, pinned)
    cache.pin("pinned", pinned)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=4,
        max_num_seqs=3,
        max_num_partial_prefills=3,
    )
    scheduler.add_request(EngineRequest("q0", short[:2], max_new_tokens=1))
    scheduler.add_request(EngineRequest("q1", second[:9], max_new_tokens=3))
    scheduler.add_request(EngineRequest("q2", first[:2], max_new_tokens=2))
    scheduler.add_request(EngineRequest("q3", first[:6], max_new_tokens=5))

    scheduler.schedule()
    scheduler.update({"q0": 0})
    plan = scheduler.schedule()

    assert [
        (chunk.request_id, chunk.num_tokens, chunk.is_prefill)
        for chunk in plan.scheduled
    ] == [("q1", 2, True), ("q2", 1, True)]
    assert scheduler.states["q1"].computed_prompt == 8
    assert scheduler.states["q1"].prefill_done is False
    assert scheduler.states["q1"].in_flight == 0
    assert scheduler.states["q2"].computed_prompt == 2
    assert scheduler.states["q2"].prefill_done is True
    assert scheduler.states["q2"].in_flight == 1
    assert scheduler.waiting_ids == ("q3",)


def test_safely_failed_restore_falls_back_to_recompute_without_publication() -> None:
    cache = RadixKVCache(num_pages=1, page_size=2)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=2)

    _publish(cache, (1, 2))
    _publish(cache, (3, 4))
    tier.restore_error = _RestoreError(destination_reusable=True)
    allocation = cache.allocate((1, 2))

    assert allocation.cached_pages == ()
    assert len(allocation.new_full_pages) == 1
    assert cache.peek_cached_tokens((1, 2)) == 0
    assert cache.dram_tier_stats["restore_fallbacks"] == 1


def test_unknown_restore_completion_quarantines_the_destination_page() -> None:
    cache = RadixKVCache(num_pages=1, page_size=2)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=2)

    _publish(cache, (1, 2))
    _publish(cache, (3, 4))
    tier.restore_error = _RestoreError(destination_reusable=False)

    with pytest.raises(_RestoreError, match="injected restore failure"):
        cache.allocate((1, 2))
    assert cache.num_free_pages == 0
    assert cache.peek_cached_tokens((1, 2)) == 0
    assert cache.dram_tier_stats["ownership_failures"] == 1


def test_failed_offload_retains_the_victim_and_its_page() -> None:
    cache = RadixKVCache(num_pages=1, page_size=2)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=2)
    _publish(cache, (1, 2))
    tier.offload_error = RuntimeError("injected D2H failure")

    with pytest.raises(RuntimeError, match="injected D2H failure"):
        cache.allocate((3, 4))
    assert cache.num_free_pages == 0
    assert cache.peek_cached_tokens((1, 2)) == 2


def test_post_restore_offload_failure_unlocks_the_restored_match_path() -> None:
    cache = RadixKVCache(num_pages=2, page_size=2)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=2)
    restored_tokens = (1, 2)
    retained_tokens = (5, 6)

    _publish(cache, restored_tokens)
    _publish(cache, (3, 4))
    _publish(cache, retained_tokens)  # moves restored_tokens to DRAM
    tier.offload_error = KeyboardInterrupt("injected unknown D2H completion")

    with pytest.raises(KeyboardInterrupt, match="unknown D2H completion"):
        cache.allocate((*restored_tokens, 9, 10))

    restored_node = cache._root.children[restored_tokens]
    assert restored_node.ref_count == 0
    assert cache.peek_cached_tokens(restored_tokens) == 2
    assert cache.peek_cached_tokens(retained_tokens) == 2
    assert cache.num_free_pages == 0


def test_safely_failed_offload_becomes_an_ordinary_eviction() -> None:
    cache = RadixKVCache(num_pages=1, page_size=2)
    tier = _FakeTier()
    cache.attach_dram_tier(tier, min_restore_tokens=2)
    first = _publish(cache, (1, 2))
    tier.offload_error = _OffloadError(source_reusable=True)

    replacement = cache.allocate((3, 4))

    assert replacement.new_full_pages == first.pages
    assert cache.peek_cached_tokens((1, 2)) == 0
    assert cache.dram_tier_stats["offload_bypassed_pages"] == 1
    assert cache.dram_tier_stats["ownership_failures"] == 0


@pytest.mark.parametrize("value", [0, 1, 3, -2, True])
def test_crossover_must_be_positive_and_page_aligned(value: object) -> None:
    cache = RadixKVCache(num_pages=2, page_size=2)
    with pytest.raises(ValueError, match="page-aligned"):
        cache.attach_dram_tier(_FakeTier(), min_restore_tokens=value)


def test_tier_cannot_be_replaced_while_host_entries_may_exist() -> None:
    cache = RadixKVCache(num_pages=2, page_size=2)
    cache.attach_dram_tier(_FakeTier(), min_restore_tokens=2)
    with pytest.raises(ValueError, match="already attached"):
        cache.attach_dram_tier(_FakeTier(), min_restore_tokens=2)


@pytest.mark.parametrize("with_event_sink", [False, True], ids=["plain", "events"])
def test_tier_must_attach_before_prefix_identity_history_exists(
    with_event_sink: bool,
) -> None:
    event_sink = (lambda _event: None) if with_event_sink else None
    cache = RadixKVCache(num_pages=2, page_size=2, event_sink=event_sink)
    _publish(cache, (1, 2))

    with pytest.raises(ValueError, match="before the cache is used"):
        cache.attach_dram_tier(_FakeTier(), min_restore_tokens=2)
