"""Stop semantics in the scheduler: stop_token_ids, min_tokens, ignore_eos,
finish_reason, finish_early (design m8 D1)."""

import pytest

from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

PAGE = 4


def _setup(num_pages=64, budget=32):
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    return Scheduler(cache, max_num_batched_tokens=budget, page_size=PAGE), cache


def _run_prefill(scheduler: Scheduler) -> None:
    scheduler.schedule()


def test_eos_sets_finish_reason_stop():
    scheduler, _ = _setup()
    scheduler.add_request(
        EngineRequest("a", (1, 2, 3), max_new_tokens=8, eos_token_id=99)
    )
    _run_prefill(scheduler)
    finished = scheduler.update({"a": 99})
    assert finished == ("a",)
    assert scheduler.finish_reason("a") == "stop"


def test_length_sets_finish_reason_length():
    scheduler, _ = _setup()
    scheduler.add_request(EngineRequest("a", (1, 2, 3), max_new_tokens=1))
    _run_prefill(scheduler)
    finished = scheduler.update({"a": 7})
    assert finished == ("a",)
    assert scheduler.finish_reason("a") == "length"


def test_stop_token_ids_finish_like_eos():
    scheduler, _ = _setup()
    scheduler.add_request(
        EngineRequest("a", (1, 2, 3), max_new_tokens=8, stop_token_ids=(42, 43))
    )
    _run_prefill(scheduler)
    assert scheduler.update({"a": 7}) == ()
    scheduler.schedule()
    assert scheduler.update({"a": 43}) == ("a",)
    assert scheduler.finish_reason("a") == "stop"


def test_min_tokens_suppresses_eos():
    scheduler, _ = _setup()
    scheduler.add_request(
        EngineRequest("a", (1, 2, 3), max_new_tokens=4, eos_token_id=99, min_tokens=2)
    )
    _run_prefill(scheduler)
    assert scheduler.update({"a": 99}) == ()  # below min_tokens: EOS ignored
    scheduler.schedule()
    assert scheduler.update({"a": 99}) == ("a",)  # at min_tokens: EOS fires
    assert scheduler.output_tokens("a") == (99, 99)


def test_ignore_eos_runs_to_length():
    scheduler, _ = _setup()
    scheduler.add_request(
        EngineRequest("a", (1, 2, 3), max_new_tokens=2, eos_token_id=99, ignore_eos=True)
    )
    _run_prefill(scheduler)
    assert scheduler.update({"a": 99}) == ()
    scheduler.schedule()
    assert scheduler.update({"a": 99}) == ("a",)
    assert scheduler.finish_reason("a") == "length"


def test_stop_token_still_honors_min_tokens():
    scheduler, _ = _setup()
    scheduler.add_request(
        EngineRequest("a", (1, 2, 3), max_new_tokens=4, stop_token_ids=(42,), min_tokens=2)
    )
    _run_prefill(scheduler)
    assert scheduler.update({"a": 42}) == ()
    scheduler.schedule()
    assert scheduler.update({"a": 42}) == ("a",)
    assert scheduler.finish_reason("a") == "stop"


def test_finish_early_commits_outputs_to_radix():
    scheduler, cache = _setup()
    prompt = (1, 2, 3, 4)
    scheduler.add_request(EngineRequest("a", prompt, max_new_tokens=8))
    _run_prefill(scheduler)
    scheduler.update({"a": 10})
    scheduler.schedule()
    scheduler.update({"a": 11})
    scheduler.finish_early("a")
    assert scheduler.finish_reason("a") == "stop"
    assert not scheduler.has_unfinished()
    # committed to the radix tree: a follow-up prompt extending prompt+outputs hits cache
    allocation = cache.allocate(prompt + (10, 11, 5, 6))
    assert allocation.num_cached_tokens >= PAGE  # at least the first full page reused


def test_finish_early_on_finished_request_is_noop():
    scheduler, _ = _setup()
    scheduler.add_request(EngineRequest("a", (1, 2, 3), max_new_tokens=1))
    _run_prefill(scheduler)
    scheduler.update({"a": 7})
    scheduler.finish_early("a")  # already finished: no error
    assert scheduler.finish_reason("a") == "length"  # original reason preserved


def test_finish_early_unknown_request_raises():
    scheduler, _ = _setup()
    with pytest.raises(KeyError):
        scheduler.finish_early("ghost")


def test_finish_reason_none_while_running():
    scheduler, _ = _setup()
    scheduler.add_request(EngineRequest("a", (1, 2, 3), max_new_tokens=4))
    _run_prefill(scheduler)
    assert scheduler.finish_reason("a") is None


def test_abort_sets_finish_reason_abort():
    scheduler, _ = _setup()
    scheduler.add_request(EngineRequest("a", (1, 2, 3), max_new_tokens=4))
    _run_prefill(scheduler)
    scheduler.abort("a")
    assert scheduler.finish_reason("a") == "abort"
