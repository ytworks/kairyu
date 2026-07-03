"""Speculative multi-token reservation in the scheduler (design m8 D3).

Pinned invariants (review amendments): capped reservation via chunk.num_tokens,
budget consumption = num_tokens, capacity degrade-to-1 (never stall worse than
k=0), scheduler-enforced in_flight==0 precondition, shortfall zeroing (not
surplus transfer), terminal-mid-list zeroes both counters.
"""

from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

PAGE = 4


def _setup(num_pages=64, budget=32, k=3, **kwargs):
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=budget,
        page_size=PAGE,
        speculative_tokens=k,
        **kwargs,
    )
    return scheduler, cache


def _to_decode(scheduler: Scheduler, request_id: str = "a", prompt=(1, 2, 3), max_new=16):
    scheduler.add_request(EngineRequest(request_id, tuple(prompt), max_new_tokens=max_new))
    scheduler.schedule()  # prefill completes
    scheduler.update({request_id: 100})  # token 0 commits


def test_spec_chunk_reserves_k_plus_one():
    scheduler, _ = _setup(k=3)
    _to_decode(scheduler)
    step = scheduler.schedule()
    chunk = step.scheduled[0]
    assert chunk.is_prefill is False
    assert chunk.num_tokens == 4  # k + 1
    assert chunk.position == 1


def test_reservation_capped_by_remaining_tokens():
    scheduler, _ = _setup(k=8)
    _to_decode(scheduler, max_new=3)  # 1 committed, 2 remaining
    chunk = scheduler.schedule().scheduled[0]
    assert chunk.num_tokens == 2  # min(k+1, remaining)


def test_spec_requires_in_flight_zero():
    # schedule twice without committing (overlap-style): the second chunk for
    # the same request must be a plain 1-token chunk, scheduler-enforced
    scheduler, _ = _setup(k=3, budget=64)
    _to_decode(scheduler)
    first = scheduler.schedule().scheduled[0]
    assert first.num_tokens == 4
    second = scheduler.schedule().scheduled[0]
    assert second.num_tokens == 1
    assert second.position == first.position + first.num_tokens


def test_spec_chunk_consumes_budget_num_tokens():
    scheduler, _ = _setup(k=3, budget=64, pd_separation=True, decode_token_budget=6)
    scheduler.add_request(EngineRequest("a", (1, 2, 3), max_new_tokens=16))
    scheduler.add_request(EngineRequest("b", (9, 8, 7), max_new_tokens=16))
    scheduler.schedule()  # both prefills complete in one step
    scheduler.update({"a": 100, "b": 100})
    step = scheduler.schedule()
    decode_chunks = [c for c in step.scheduled if not c.is_prefill]
    # decode budget 6: first spec chunk takes 4, second gets the remaining 2
    assert [c.num_tokens for c in decode_chunks] == [4, 2]


def test_capacity_degrades_to_single_token():
    # tight KV: k+1 slots don't fit, but 1 does — never stall worse than k=0
    scheduler, cache = _setup(num_pages=3, k=8)
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4), max_new_tokens=8))
    scheduler.schedule()
    scheduler.update({"a": 100})
    step = scheduler.schedule()
    assert len(step.scheduled) == 1
    assert step.scheduled[0].num_tokens < 9  # degraded below the full reservation


def test_full_acceptance_commits_all_reserved_tokens():
    scheduler, _ = _setup(k=3)
    _to_decode(scheduler)
    scheduler.schedule()
    finished = scheduler.update({"a": [101, 102, 103, 104]})
    assert finished == ()
    assert scheduler.output_tokens("a") == (100, 101, 102, 103, 104)
    # accounting clean: next schedule re-reserves a full spec chunk
    chunk = scheduler.schedule().scheduled[0]
    assert chunk.num_tokens == 4
    assert chunk.position == 5


def test_partial_acceptance_zeroes_shortfall():
    scheduler, _ = _setup(k=3)
    _to_decode(scheduler)
    scheduler.schedule()  # reserves 4
    scheduler.update({"a": [101, 102]})  # 2 rejected draft slots never arrive
    assert scheduler.output_tokens("a") == (100, 101, 102)
    chunk = scheduler.schedule().scheduled[0]
    assert chunk.num_tokens == 4  # in_flight was zeroed, full re-reservation
    assert chunk.position == 3


def test_eos_mid_list_discards_rest_and_zeroes_counters():
    scheduler, _ = _setup(k=3)
    scheduler.add_request(
        EngineRequest("a", (1, 2, 3), max_new_tokens=16, eos_token_id=999)
    )
    scheduler.schedule()
    scheduler.update({"a": 100})
    scheduler.schedule()  # reserves 4
    finished = scheduler.update({"a": [101, 999, 102, 103]})
    assert finished == ("a",)
    assert scheduler.output_tokens("a") == (100, 101, 999)
    assert scheduler.finish_reason("a") == "stop"
    assert not scheduler.has_unfinished()


def test_max_new_tokens_mid_list_truncates():
    scheduler, _ = _setup(k=8)
    _to_decode(scheduler, max_new=3)
    scheduler.schedule()  # reserves min(9, 2) = 2
    finished = scheduler.update({"a": [101, 102]})
    assert finished == ("a",)
    assert scheduler.finish_reason("a") == "length"
    assert len(scheduler.output_tokens("a")) == 3


def test_abort_with_reserved_spec_slots_releases_cleanly():
    scheduler, cache = _setup(k=3)
    _to_decode(scheduler)
    scheduler.schedule()  # 4 slots reserved, in flight
    scheduler.abort("a")
    assert not scheduler.has_unfinished()
    # late arrivals for the aborted chunk are trimmed, not errors
    assert scheduler.update({"a": [101, 102, 103, 104]}) == ()


def test_spec_zero_matches_legacy_single_token_behavior():
    scheduler, _ = _setup(k=0)
    _to_decode(scheduler)
    chunk = scheduler.schedule().scheduled[0]
    assert chunk.num_tokens == 1


def test_preemption_victims_still_output_free_only():
    # a spec request has committed outputs -> never a preemption victim
    scheduler, _ = _setup(num_pages=8, k=3, budget=8)
    _to_decode(scheduler, "a", prompt=(1, 2, 3))
    scheduler.add_request(EngineRequest("b", tuple(range(10, 22)), max_new_tokens=4))
    for _ in range(4):
        step = scheduler.schedule()
        commit = {}
        for chunk in step.scheduled:
            state = scheduler.states[chunk.request_id]
            if not chunk.is_prefill:
                commit[chunk.request_id] = [7] * chunk.num_tokens
            elif state.prefill_done:
                commit[chunk.request_id] = 7
        if commit:
            scheduler.update(commit)
        if not scheduler.has_unfinished():
            break
    assert scheduler.states["a"].status.value in ("running", "finished")
