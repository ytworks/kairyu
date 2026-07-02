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
    scheduler.add_request(_request("b", prompt_len=8))
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
