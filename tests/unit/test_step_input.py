"""StepInput snapshot: torn-free immutable step state (design m5 D2, m2 §5 item 3)."""

import dataclasses

import pytest

from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.step_input import RequestSnapshot, snapshot_step

PAGE = 4


def _scheduler(num_pages: int = 64, budget: int = 32, max_seqs: int = 4) -> Scheduler:
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    return Scheduler(
        cache, max_num_batched_tokens=budget, max_num_seqs=max_seqs, page_size=PAGE
    )


def test_snapshot_copies_request_fields_from_live_state():
    scheduler = _scheduler()
    scheduler.add_request(
        EngineRequest("a", (1, 2, 3, 4, 5), max_new_tokens=3, eos_token_id=9)
    )
    plan = scheduler.schedule()
    snapshot = snapshot_step(plan, scheduler.states)
    assert snapshot.chunks == plan.scheduled
    entry = snapshot.states_view()["a"]
    assert entry.request_id == "a"
    assert entry.prompt_token_ids == (1, 2, 3, 4, 5)
    assert entry.computed_prompt == 5
    assert entry.output_len == 0
    assert entry.in_flight == 1
    assert entry.eos_token_id == 9
    assert entry.max_new_tokens == 3
    assert isinstance(entry.page_ids, tuple) and len(entry.page_ids) == 2
    assert isinstance(entry.decode_page_ids, tuple)


def test_snapshot_accepts_raw_chunk_tuple():
    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4)))
    plan = scheduler.schedule()
    from_output = snapshot_step(plan, scheduler.states)
    from_chunks = snapshot_step(plan.scheduled, scheduler.states)
    assert from_output == from_chunks


def test_snapshot_is_torn_free_after_scheduler_mutation():
    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4, 5), max_new_tokens=4))
    plan = scheduler.schedule()
    snapshot = snapshot_step(plan, scheduler.states)
    scheduler.update({"a": 7})  # live state mutates: outputs grows, in_flight drops
    entry = snapshot.states_view()["a"]
    assert entry.output_len == 0
    assert entry.in_flight == 1
    live = scheduler.states["a"]
    assert live.outputs == [7] and live.in_flight == 0  # sanity: live state moved on


def test_snapshot_dataclasses_are_frozen():
    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4)))
    snapshot = snapshot_step(scheduler.schedule(), scheduler.states)
    entry = snapshot.states_view()["a"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.output_len = 5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.chunks = ()  # type: ignore[misc]


def test_request_property_provides_toy_runner_compatible_view():
    entry = RequestSnapshot(
        request_id="a",
        prompt_token_ids=(1, 2, 3),
        computed_prompt=3,
        outputs=(),
        in_flight=1,
        page_ids=(0,),
        decode_page_ids=(),
        eos_token_id=None,
        max_new_tokens=4,
    )
    assert entry.request is entry
    assert entry.request.prompt_token_ids == (1, 2, 3)
    assert entry.prefill_done is True


def test_prefill_done_is_false_for_partial_prefill():
    scheduler = _scheduler(budget=2)
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4, 5)))
    snapshot = snapshot_step(scheduler.schedule(), scheduler.states)
    entry = snapshot.states_view()["a"]
    assert entry.computed_prompt == 2
    assert entry.prefill_done is False
    assert entry.in_flight == 0


def test_snapshot_includes_only_scheduled_requests():
    scheduler = _scheduler(max_seqs=1)
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4)))
    scheduler.add_request(EngineRequest("b", (5, 6, 7, 8)))
    snapshot = snapshot_step(scheduler.schedule(), scheduler.states)
    assert set(snapshot.states_view()) == {"a"}  # "b" still waiting, not snapshotted


def test_snapshot_rejects_chunk_for_unknown_request():
    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4)))
    plan = scheduler.schedule()
    with pytest.raises(ValueError, match="unknown request"):
        snapshot_step(plan.scheduled, {})
