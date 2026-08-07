import pytest

from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

PAGE = 4


def _setup(num_pages=64, budget=8, max_seqs=4):
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=budget, max_num_seqs=max_seqs)
    return scheduler, cache


def _request(request_id: str, prompt_len: int, max_new_tokens: int = 4) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        prompt_token_ids=tuple(range(1, prompt_len + 1)),
        max_new_tokens=max_new_tokens,
    )


def test_prefill_is_chunked_by_token_budget():
    scheduler, _ = _setup(budget=8)
    scheduler.add_request(_request("a", prompt_len=20))
    chunks = [scheduler.schedule().scheduled[0].num_tokens for _ in range(3)]
    assert chunks == [8, 8, 4]
    assert all(c.is_prefill for step in range(0) for c in [])  # placeholder no-op
    # prompt fully computed -> next step is a decode of 1 token after sampling
    scheduler.update({"a": 100})
    step = scheduler.schedule()
    assert step.scheduled[0].is_prefill is False
    assert step.scheduled[0].num_tokens == 1


def test_prefill_cohort_shares_budget_from_initial_admission() -> None:
    scheduler, _ = _setup(budget=8, max_seqs=4)
    scheduler.add_request(_request("long-a", prompt_len=20))
    scheduler.add_request(
        EngineRequest(
            "short-b",
            prompt_token_ids=tuple(range(101, 105)),
            max_new_tokens=1,
        )
    )

    first = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in first.scheduled] == [
        ("long-a", 4),
        ("short-b", 4),
    ]
    assert scheduler.states["long-a"].prefill_done is False
    assert scheduler.states["short-b"].prefill_done is True

    scheduler.update({"short-b": 200})
    second = scheduler.schedule()
    assert [(chunk.request_id, chunk.num_tokens) for chunk in second.scheduled] == [("long-a", 8)]


def test_prefill_cohort_redistributes_short_share_without_underfill() -> None:
    scheduler, _ = _setup(budget=8, max_seqs=4)
    scheduler.add_request(_request("almost-done", prompt_len=9))
    assert scheduler.schedule().scheduled[0].num_tokens == 8
    scheduler.add_request(
        EngineRequest(
            "long",
            prompt_token_ids=tuple(range(101, 121)),
            max_new_tokens=1,
        )
    )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [
        ("almost-done", 1),
        ("long", 7),
    ]
    assert sum(chunk.num_tokens for chunk in step.scheduled) == 8


def test_prefill_cohort_returns_share_when_kv_waiter_stays_blocked() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=3, page_size=PAGE),
        max_num_batched_tokens=4,
        max_num_seqs=2,
    )
    scheduler.add_request(_request("running", prompt_len=8))
    assert scheduler.schedule().scheduled[0].num_tokens == 4
    scheduler.add_request(
        EngineRequest(
            "blocked",
            prompt_token_ids=tuple(range(101, 109)),
            max_new_tokens=1,
        )
    )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [("running", 4)]
    assert scheduler.waiting_ids == ("blocked",)


def test_prefill_cohort_uses_configured_width_after_radix_eviction() -> None:
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    cold = cache.allocate(tuple(range(901, 933)))
    cache.mark_computed(cold)
    cache.free(cold)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=9,
        max_num_seqs=3,
        max_num_partial_prefills=3,
    )
    for index in range(3):
        start = 100 * (index + 1)
        scheduler.add_request(
            EngineRequest(
                f"request-{index}",
                prompt_token_ids=tuple(range(start, start + 8)),
                max_new_tokens=1,
            )
        )

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [
        ("request-0", 3),
        ("request-1", 3),
        ("request-2", 3),
    ]


def test_prefill_cohort_limits_partial_width_and_singleton_keeps_budget() -> None:
    scheduler, _ = _setup(budget=8, max_seqs=4)
    for index in range(3):
        scheduler.add_request(
            EngineRequest(
                f"long-{index}",
                prompt_token_ids=tuple(range(100 * index + 1, 100 * index + 21)),
                max_new_tokens=1,
            )
        )

    first = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in first.scheduled] == [
        ("long-0", 4),
        ("long-1", 4),
    ]
    assert scheduler.waiting_ids == ("long-2",)

    singleton = Scheduler(
        RadixKVCache(num_pages=16, page_size=PAGE),
        max_num_batched_tokens=8,
        max_num_partial_prefills=2,
    )
    singleton.add_request(_request("only", prompt_len=20))
    assert singleton.schedule().scheduled[0].num_tokens == 8


def test_prefill_cohort_does_not_count_unavailable_sequence_slots() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=64, page_size=PAGE),
        max_num_batched_tokens=8,
        max_num_seqs=2,
        max_num_partial_prefills=8,
    )
    for index in range(3):
        scheduler.add_request(
            EngineRequest(
                str(index),
                tuple(range(index * 100 + 1, index * 100 + 21)),
                max_new_tokens=1,
            )
        )

    assert [
        (chunk.request_id, chunk.num_tokens)
        for chunk in scheduler.schedule().scheduled
    ] == [("0", 4), ("1", 4)]
    assert scheduler.waiting_ids == ("2",)


def test_prefill_cohort_budget_one_never_emits_zero_chunk() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=16, page_size=PAGE),
        max_num_batched_tokens=1,
        max_num_partial_prefills=2,
    )
    scheduler.add_request(_request("a", prompt_len=8))
    scheduler.add_request(EngineRequest("b", tuple(range(101, 109)), max_new_tokens=1))

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.num_tokens) for chunk in step.scheduled] == [("a", 1)]


def test_prefill_cohort_keeps_recompute_victim_until_decode_can_finish() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=5, page_size=2),
        max_num_batched_tokens=2,
        max_num_seqs=4,
        max_num_partial_prefills=2,
        priority_age_s=None,
    )
    scheduler.add_request(EngineRequest("a", tuple(range(1, 4)), max_new_tokens=6))
    scheduler.add_request(EngineRequest("b", tuple(range(101, 105)), max_new_tokens=4))

    for token in range(32):
        if not scheduler.has_unfinished():
            break
        step = scheduler.schedule()
        assert step.scheduled
        sampled = {
            chunk.request_id: token
            for chunk in step.scheduled
            if not chunk.is_prefill or scheduler.states[chunk.request_id].prefill_done
        }
        scheduler.update(sampled)

    assert not scheduler.has_unfinished()
    assert len(scheduler.output_tokens("a")) == 6
    assert len(scheduler.output_tokens("b")) == 4


def test_decode_preemption_prefers_victim_that_releases_private_pages() -> None:
    cache = RadixKVCache(num_pages=10, page_size=2)
    cold_tokens = tuple(range(9000, 9016))
    cold = cache.allocate(cold_tokens)
    cache.mark_computed(cold)
    cache.free(cold)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=8,
        max_num_partial_prefills=5,
        pd_separation=True,
        decode_token_budget=3,
        priority_age_s=0.0,
    )
    requests = (
        EngineRequest("r0", tuple(range(9)), max_new_tokens=2, priority=-1),
        EngineRequest("r1", tuple(range(100, 109)), max_new_tokens=1, priority=2),
        EngineRequest("r2", cold_tokens[:6], max_new_tokens=7, priority=-1),
        EngineRequest("r3", cold_tokens[:4], max_new_tokens=5, priority=-1),
        EngineRequest("r4", tuple(range(400, 408)), max_new_tokens=10, priority=-2),
    )
    for request in requests:
        scheduler.add_request(request)

    for token in range(96):
        if not scheduler.has_unfinished():
            break
        step = scheduler.schedule()
        assert step.scheduled
        sampled = {
            chunk.request_id: token
            for chunk in step.scheduled
            if not chunk.is_prefill or scheduler.states[chunk.request_id].prefill_done
        }
        scheduler.update(sampled)

    assert not scheduler.has_unfinished()
    assert {
        request.request_id: len(scheduler.output_tokens(request.request_id)) for request in requests
    } == {request.request_id: request.max_new_tokens for request in requests}


def test_decode_does_not_preempt_a_pending_prompt_completion() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=2, page_size=2),
        max_num_batched_tokens=4,
        max_num_seqs=2,
        max_num_partial_prefills=2,
    )
    scheduler.add_request(EngineRequest("pending", (1, 2), max_new_tokens=1))
    scheduler.add_request(EngineRequest("decode", (3, 4), max_new_tokens=3))
    first = scheduler.schedule()
    assert [chunk.request_id for chunk in first.scheduled] == ["pending", "decode"]
    scheduler.update({"decode": 10})

    blocked = scheduler.schedule()

    assert blocked.scheduled == ()
    assert scheduler.waiting_ids == ()
    assert scheduler.states["pending"].in_flight == 1
    assert scheduler.states["pending"].surplus_in_flight == 0


def test_unsafe_one_token_waiter_stays_waiting_instead_of_zero_work_admission() -> None:
    scheduler = Scheduler(
        RadixKVCache(num_pages=7, page_size=1),
        max_num_batched_tokens=8,
        max_num_seqs=4,
        max_num_partial_prefills=2,
    )
    requests = (
        EngineRequest("r0", tuple(range(3)), max_new_tokens=2),
        EngineRequest("r1", tuple(range(100, 103)), max_new_tokens=4),
        EngineRequest("r2", (200,), max_new_tokens=2),
        EngineRequest("r3", (300,), max_new_tokens=3),
    )
    for request in requests:
        scheduler.add_request(request)

    first = scheduler.schedule()
    assert [chunk.request_id for chunk in first.scheduled] == ["r0", "r1"]
    assert scheduler.waiting_ids == ("r2", "r3")

    for token in range(32):
        sampled = {
            chunk.request_id: token
            for chunk in first.scheduled
            if not chunk.is_prefill or scheduler.states[chunk.request_id].prefill_done
        }
        scheduler.update(sampled)
        if not scheduler.has_unfinished():
            break
        first = scheduler.schedule()
        assert first.scheduled

    assert not scheduler.has_unfinished()


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "2"])
def test_max_num_partial_prefills_must_be_positive_integer(value) -> None:
    with pytest.raises(ValueError, match="max_num_partial_prefills"):
        Scheduler(
            RadixKVCache(num_pages=16, page_size=PAGE),
            max_num_partial_prefills=value,
        )


def test_scheduler_preserves_legacy_positional_clock_argument() -> None:
    def clock() -> float:
        return 123.0

    scheduler = Scheduler(
        RadixKVCache(num_pages=16, page_size=PAGE),
        2048,
        256,
        PAGE,
        False,
        None,
        0,
        0,
        None,
        clock,
    )

    scheduler.add_request(_request("clocked", prompt_len=1))

    assert scheduler._arrivals["clocked"] == 123.0


def test_decode_has_priority_over_prefill():
    scheduler, _ = _setup(budget=8)
    scheduler.add_request(_request("a", prompt_len=4))
    scheduler.schedule()  # a completes prefill
    scheduler.update({"a": 100})  # first sampled token
    # b's prompt is disjoint from a's so prefix caching doesn't shrink its chunk
    scheduler.add_request(
        EngineRequest("b", prompt_token_ids=tuple(range(101, 109)), max_new_tokens=4)
    )
    step = scheduler.schedule()
    kinds = [(c.request_id, c.is_prefill, c.num_tokens) for c in step.scheduled]
    assert kinds[0] == ("a", False, 1)
    assert kinds[1] == ("b", True, 7)  # remaining budget after the decode


def test_decode_precedes_and_reduces_prefill_cohort_budget() -> None:
    scheduler, _ = _setup(budget=8, max_seqs=4)
    scheduler.add_request(_request("decode", prompt_len=4))
    scheduler.schedule()
    scheduler.update({"decode": 100})
    scheduler.add_request(EngineRequest("prefill-a", tuple(range(101, 121)), max_new_tokens=1))
    scheduler.add_request(EngineRequest("prefill-b", tuple(range(201, 221)), max_new_tokens=1))

    step = scheduler.schedule()

    assert [(chunk.request_id, chunk.is_prefill, chunk.num_tokens) for chunk in step.scheduled] == [
        ("decode", False, 1),
        ("prefill-a", True, 4),
        ("prefill-b", True, 3),
    ]
    assert sum(chunk.num_tokens for chunk in step.scheduled) == 8


def test_max_num_seqs_limits_admission():
    scheduler, _ = _setup(budget=64, max_seqs=1)
    scheduler.add_request(_request("a", prompt_len=4, max_new_tokens=1))
    scheduler.add_request(_request("b", prompt_len=4))
    step = scheduler.schedule()
    assert [c.request_id for c in step.scheduled] == ["a"]
    finished = scheduler.update({"a": 100})  # max_new_tokens=1 -> finished
    assert finished == ("a",)
    step = scheduler.schedule()
    assert [c.request_id for c in step.scheduled] == ["b"]


def test_finished_request_prompt_is_reusable_from_cache():
    scheduler, cache = _setup()
    scheduler.add_request(_request("a", prompt_len=8, max_new_tokens=1))
    scheduler.schedule()
    scheduler.update({"a": 100})
    assert scheduler.has_unfinished() is False
    reuse = cache.allocate(tuple(range(1, 9)))
    assert len(reuse.cached_pages) == 2  # radix reuse across requests


def test_forget_reclaims_finished_state():
    # E2: finished requests must be evictable from the scheduler, or a
    # long-running engine grows _states/_arrivals without bound.
    scheduler, _ = _setup(budget=64)
    scheduler.add_request(_request("a", prompt_len=4, max_new_tokens=1))
    scheduler.schedule()
    scheduler.update({"a": 100})  # max_new_tokens=1 -> finished
    assert scheduler.has_unfinished() is False
    scheduler.forget("a")
    assert "a" not in scheduler.states
    assert "a" not in scheduler._arrivals


def test_forget_leaves_a_live_request_untouched():
    # forget() must not drop a still-running request's state.
    scheduler, _ = _setup(budget=64)
    scheduler.add_request(_request("a", prompt_len=4, max_new_tokens=4))
    scheduler.schedule()
    scheduler.update({"a": 100})  # still running (max_new_tokens=4)
    scheduler.forget("a")
    assert "a" in scheduler.states


def test_oversized_prompt_is_rejected_not_blocking():
    # C2: a prompt needing more pages than the cache can EVER hold must be
    # rejected at admission, not left blocking the head of line forever (which
    # turns an empty schedule into a fatal engine stall).
    scheduler, _ = _setup(num_pages=2, budget=64)  # capacity = 8 tokens
    scheduler.add_request(_request("big", prompt_len=20))  # needs 5 pages
    scheduler.add_request(_request("small", prompt_len=4, max_new_tokens=1))
    step = scheduler.schedule()
    assert scheduler.finish_reason("big") == "length"  # rejected, not scheduled
    assert "big" not in [c.request_id for c in step.scheduled]
    # the normal request behind it still makes progress
    assert "small" in [c.request_id for c in step.scheduled]


def test_unadmittable_head_does_not_stall_scheduler():
    # C2: with only an unadmittable request, schedule() returns an empty plan
    # AND the request is finished, so has_unfinished() is False and the engine
    # loop never trips its "nothing schedulable" stall guard.
    scheduler, _ = _setup(num_pages=2, budget=64)
    scheduler.add_request(_request("big", prompt_len=20))
    step = scheduler.schedule()
    assert step.scheduled == ()
    assert scheduler.finish_reason("big") == "length"
    assert scheduler.has_unfinished() is False


def test_empty_prompt_is_rejected_without_prefill_work():
    scheduler, _ = _setup()
    scheduler.add_request(_request("empty", prompt_len=0))

    step = scheduler.schedule()

    assert step.scheduled == ()
    assert scheduler.has_unfinished() is False
    assert scheduler.finish_reason("empty") == "length"
    assert scheduler.output_tokens("empty") == ()
    assert scheduler.drain_rejected() == ("empty",)


def test_kv_pressure_keeps_request_waiting_then_admits():
    scheduler, _ = _setup(num_pages=2, budget=64, max_seqs=4)
    scheduler.add_request(
        EngineRequest("a", prompt_token_ids=tuple(range(1, 9)), max_new_tokens=1)
    )  # 2 pages
    scheduler.add_request(
        EngineRequest("b", prompt_token_ids=tuple(range(101, 105)), max_new_tokens=1)
    )  # distinct prompt, needs 1 page
    step = scheduler.schedule()
    assert [c.request_id for c in step.scheduled] == ["a"]  # b: no KV space, stays waiting
    finished = scheduler.update({"a": 100})
    assert finished == ("a",)
    step = scheduler.schedule()  # a's pages now evictable -> b admitted
    assert [c.request_id for c in step.scheduled] == ["b"]


def test_identical_prompts_wait_until_computed_pages_are_published():
    scheduler, cache = _setup(num_pages=1, budget=64, max_seqs=4)
    scheduler.add_request(_request("a", prompt_len=4, max_new_tokens=1))
    scheduler.add_request(_request("b", prompt_len=4, max_new_tokens=1))
    step = scheduler.schedule()
    # Scheduling a's forward does not prove its KV exists yet, so b cannot
    # reuse that page while the device result is still in flight.
    assert [c.request_id for c in step.scheduled] == ["a"]
    assert cache.peek_cached_tokens((1, 2, 3, 4)) == 0

    scheduler.update({"a": 100})
    assert cache.peek_cached_tokens((1, 2, 3, 4)) == 4
    step = scheduler.schedule()
    assert [c.request_id for c in step.scheduled] == ["b"]
    assert cache.hit_rate == 0.5


def test_decode_output_tokens_are_recorded():
    scheduler, _ = _setup()
    scheduler.add_request(_request("a", prompt_len=4, max_new_tokens=3))
    scheduler.schedule()
    scheduler.update({"a": 100})
    scheduler.schedule()
    scheduler.update({"a": 101})
    scheduler.schedule()
    finished = scheduler.update({"a": 102})
    assert finished == ("a",)
    assert scheduler.output_tokens("a") == (100, 101, 102)


def test_pd_separation_gives_prefill_and_decode_independent_budgets():
    cache = RadixKVCache(num_pages=256, page_size=PAGE)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=8,
        max_num_seqs=8,
        pd_separation=True,
        decode_token_budget=2,
    )
    # three requests reach decode phase
    for i in range(3):
        prompt = tuple(range(i * 100 + 1, i * 100 + 3))
        scheduler.add_request(EngineRequest(f"d{i}", prompt, max_new_tokens=4))
    scheduler.schedule()
    scheduler.update({"d0": 1, "d1": 1, "d2": 1})
    scheduler.add_request(_request("p", prompt_len=20))
    scheduler.add_request(EngineRequest("p2", tuple(range(501, 521)), max_new_tokens=1))
    step = scheduler.schedule()
    decodes = [c for c in step.scheduled if not c.is_prefill]
    prefills = [c for c in step.scheduled if c.is_prefill]
    assert len(decodes) == 2  # capped by decode budget, third decode waits
    assert [(chunk.request_id, chunk.num_tokens) for chunk in prefills] == [
        ("p", 4),
        ("p2", 4),
    ]  # full independent prefill budget, not reduced by decodes


def test_combined_mode_decodes_consume_shared_budget():
    cache = RadixKVCache(num_pages=256, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=8, max_num_seqs=8)
    for i in range(3):
        prompt = tuple(range(i * 100 + 1, i * 100 + 3))
        scheduler.add_request(EngineRequest(f"d{i}", prompt, max_new_tokens=4))
    scheduler.schedule()
    scheduler.update({"d0": 1, "d1": 1, "d2": 1})
    scheduler.add_request(_request("p", prompt_len=20))
    step = scheduler.schedule()
    prefills = [c for c in step.scheduled if c.is_prefill]
    assert prefills[0].num_tokens == 5  # 8 - 3 decodes


def test_cached_prefix_skips_prefill_compute():
    scheduler, _ = _setup(budget=64)
    scheduler.add_request(_request("a", prompt_len=8, max_new_tokens=1))
    scheduler.schedule()
    scheduler.update({"a": 100})  # finished; prompt pages committed to cache
    scheduler.add_request(_request("b", prompt_len=8, max_new_tokens=1))
    step = scheduler.schedule()
    chunk = step.scheduled[0]
    assert chunk.request_id == "b"
    assert chunk.num_tokens == 1  # 7 of 8 tokens cached; only last token recomputed
