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


def test_identical_prompts_share_pages_under_pressure():
    scheduler, cache = _setup(num_pages=1, budget=64, max_seqs=4)
    scheduler.add_request(_request("a", prompt_len=4, max_new_tokens=1))
    scheduler.add_request(_request("b", prompt_len=4, max_new_tokens=1))
    step = scheduler.schedule()
    # both fit in ONE page because b radix-hits a's identical prompt
    assert [c.request_id for c in step.scheduled] == ["a", "b"]
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
    step = scheduler.schedule()
    decodes = [c for c in step.scheduled if not c.is_prefill]
    prefills = [c for c in step.scheduled if c.is_prefill]
    assert len(decodes) == 2  # capped by decode budget, third decode waits
    assert prefills[0].num_tokens == 8  # full prefill budget, not reduced by decodes


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
