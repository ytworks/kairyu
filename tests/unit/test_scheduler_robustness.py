from kairyu.engine.core.overlap import OverlapEngineCore
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

PAGE = 4
EOS = 777


def _setup(num_pages=64, budget=8, max_seqs=4, **kwargs):
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    scheduler = Scheduler(
        cache, max_num_batched_tokens=budget, max_num_seqs=max_seqs, **kwargs
    )
    return scheduler, cache


def test_eos_token_finishes_request_early():
    scheduler, _ = _setup()
    scheduler.add_request(
        EngineRequest("a", (1, 2, 3, 4), max_new_tokens=8, eos_token_id=EOS)
    )
    scheduler.schedule()
    finished = scheduler.update({"a": EOS})
    assert finished == ("a",)
    assert scheduler.output_tokens("a") == (EOS,)


def test_overlap_surplus_token_after_eos_is_trimmed():
    class EosAtPositionOne:
        def execute(self, scheduled, states):
            return {
                chunk.request_id: (
                    SampledToken(EOS if chunk.position == 1 else 1000 + chunk.position),
                )
                for chunk in scheduled
                if not chunk.is_prefill or states[chunk.request_id].prefill_done
            }

    cache = RadixKVCache(num_pages=64, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=8)
    engine = OverlapEngineCore(scheduler=scheduler, runner=EosAtPositionOne())
    engine.add_request(
        EngineRequest("a", (1, 2, 3, 4), max_new_tokens=8, eos_token_id=EOS)
    )
    outputs = engine.run_to_completion()
    # position 2 was scheduled ahead under overlap; its token must be trimmed
    assert outputs["a"] == (1000, EOS)


def test_abort_releases_kv_and_removes_request():
    scheduler, cache = _setup(num_pages=2)
    scheduler.add_request(EngineRequest("a", tuple(range(1, 9)), max_new_tokens=8))
    scheduler.schedule()
    assert cache.num_free_pages == 0
    scheduler.abort("a")
    assert scheduler.has_unfinished() is False
    # pages are evictable again: a full new allocation must succeed
    allocation = cache.allocate(tuple(range(101, 109)))
    assert len(allocation.new_full_pages) == 2


def test_decode_pressure_preempts_prefilling_victim():
    scheduler, _ = _setup(num_pages=3, budget=4, max_seqs=4)
    scheduler.add_request(
        EngineRequest("b", tuple(range(101, 105)), max_new_tokens=4)
    )
    scheduler.schedule()  # b admitted, prefill complete (1 page)
    scheduler.update({"b": 1})
    scheduler.add_request(EngineRequest("a", tuple(range(1, 9)), max_new_tokens=4))
    scheduler.schedule()  # a admitted (2 pages), prefill incomplete: 0 free pages left
    step = scheduler.schedule()  # b's decode needs a page -> preempt a
    scheduled_ids = [c.request_id for c in step.scheduled]
    assert "b" in scheduled_ids  # decode proceeded thanks to preemption
    assert "a" in scheduler.waiting_ids  # a was requeued, not lost


def test_pin_ttl_expires_after_allocations():
    cache = RadixKVCache(num_pages=3, page_size=PAGE)
    prefix = (1, 2, 3, 4, 5, 6, 7, 8)
    allocation = cache.allocate(prefix)
    cache.free(allocation)
    cache.pin("s", prefix, ttl_allocations=2)
    cache.free(cache.allocate((11, 12, 13, 14)))  # tick 1 (uses eviction? 0 free... )
    cache.free(cache.allocate((11, 12, 13, 14)))  # tick 2 (cache hit)
    # pin now expired: evicting the pinned prefix must be possible
    big = cache.allocate((21, 22, 23, 24, 25, 26, 27, 28))
    assert len(big.new_full_pages) == 2


def test_decode_watermark_defers_admission():
    scheduler, _ = _setup(num_pages=3, budget=64, decode_watermark_pages=1)
    scheduler.add_request(
        EngineRequest("a", tuple(range(1, 9)), max_new_tokens=2)
    )  # needs 2 pages, leaves 1 = watermark
    scheduler.add_request(
        EngineRequest("b", tuple(range(101, 105)), max_new_tokens=2)
    )  # would eat the reserve -> deferred
    step = scheduler.schedule()
    assert [c.request_id for c in step.scheduled] == ["a"]
    scheduler.update({"a": 1})
    step = scheduler.schedule()  # a's decode uses the reserved page, no stall
    assert [c.request_id for c in step.scheduled] == ["a"]
