"""StepInput snapshot: torn-free immutable step state (design m5 D2, m2 §5 item 3)."""

import dataclasses

import pytest

from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import EngineRequest, ScheduledChunk, Scheduler
from kairyu.engine.core.step_input import (
    RequestDelta,
    RequestSnapshot,
    StepDelta,
    decode_step_delta,
    encode_step_delta,
    snapshot_step,
)

PAGE = 4


class _ObservedList(list):
    def __init__(self, values):
        super().__init__(values)
        self.index_reads = 0

    def __getitem__(self, index):
        self.index_reads += 1
        return super().__getitem__(index)


def _scheduler(num_pages: int = 64, budget: int = 32, max_seqs: int = 4) -> Scheduler:
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    return Scheduler(
        cache, max_num_batched_tokens=budget, max_num_seqs=max_seqs, page_size=PAGE
    )


def test_snapshot_copies_request_fields_from_live_state():
    scheduler = _scheduler()
    scheduler.add_request(
        EngineRequest(
            "a",
            (1, 2, 3, 4, 5),
            max_new_tokens=3,
            eos_token_id=9,
            stop_token_ids=(10, 11),
            min_tokens=2,
            ignore_eos=True,
        )
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
    assert entry.stop_token_ids == (10, 11)
    assert entry.min_tokens == 2
    assert entry.ignore_eos is True
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


def test_snapshot_freezes_output_length_without_copying_the_prefix():
    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4)))
    plan = scheduler.schedule()
    outputs = _ObservedList(range(4096))
    scheduler.states["a"].outputs = outputs

    snapshot = snapshot_step(plan, scheduler.states).states_view()["a"]

    assert outputs.index_reads == 0
    outputs.append(4096)
    assert len(snapshot.outputs) == 4096
    assert snapshot.outputs[-1] == 4095


def test_state_sync_steady_delta_reads_only_the_new_output_tail():
    from kairyu.engine.core.step_input import StateSync

    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4)))
    plan = scheduler.schedule()
    outputs = _ObservedList(range(4096))
    scheduler.states["a"].outputs = outputs
    sync = StateSync()
    sync.apply(sync.diff(plan.scheduled, scheduler.states))
    outputs.index_reads = 0
    outputs.append(4096)

    delta = sync.diff(plan.scheduled, scheduler.states)

    assert delta.new == ()
    assert delta.updates[0].new_outputs == (4096,)
    assert outputs.index_reads == 1


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


def test_step_delta_tensor_codec_round_trips_every_field():
    assert tuple(field.name for field in dataclasses.fields(EngineSampling)) == (
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "logprobs",
        "json_schema",
        "json_mode",
        "regex",
        "grammar",
        "structural_tag",
        "forced_token_ids",
    )
    assert tuple(field.name for field in dataclasses.fields(RequestSnapshot)) == (
        "request_id",
        "prompt_token_ids",
        "computed_prompt",
        "outputs",
        "in_flight",
        "page_ids",
        "decode_page_ids",
        "eos_token_id",
        "max_new_tokens",
        "num_cached_tokens",
        "sampling",
        "output_epoch",
        "outputs_override",
        "sampling_id",
        "stop_token_ids",
        "min_tokens",
        "ignore_eos",
    )
    assert tuple(field.name for field in dataclasses.fields(RequestDelta)) == (
        "request_id",
        "computed_prompt",
        "new_outputs",
        "in_flight",
        "new_decode_page_ids",
        "num_cached_tokens",
    )
    assert tuple(field.name for field in dataclasses.fields(ScheduledChunk)) == (
        "request_id",
        "num_tokens",
        "is_prefill",
        "position",
    )
    assert tuple(field.name for field in dataclasses.fields(StepDelta)) == (
        "chunks",
        "new",
        "updates",
        "dropped",
        "released_ids",
    )
    sampling = EngineSampling(
        temperature=0.75,
        top_k=17,
        top_p=0.875,
        min_p=0.125,
        presence_penalty=0.25,
        frequency_penalty=-0.5,
        repetition_penalty=1.125,
        seed=(1 << 63) - 1,
        logprobs=4,
        json_schema={"type": "object", "required": ["資料"]},
        json_mode=True,
        regex="資料.+",
        grammar='root ::= "資料"',
        structural_tag={
            "type": "structural_tag",
            "format": {"type": "const_string", "value": "資料"},
        },
        forced_token_ids=(7, 8),
    )
    snapshot = RequestSnapshot(
        request_id="新規-request",
        prompt_token_ids=(1, 2, 3),
        computed_prompt=2,
        outputs=(9, 10),
        in_flight=1,
        page_ids=(11, 12),
        decode_page_ids=(13,),
        eos_token_id=14,
        max_new_tokens=32,
        num_cached_tokens=2,
        sampling=sampling,
        output_epoch=3,
        outputs_override=True,
        sampling_id="public-id",
        stop_token_ids=(15, 16),
        min_tokens=2,
        ignore_eos=True,
    )
    delta = StepDelta(
        chunks=(ScheduledChunk("新規-request", 3, True, 4),),
        new=(snapshot,),
        updates=(
            RequestDelta(
                request_id="existing",
                computed_prompt=8,
                new_outputs=(17, 18),
                in_flight=2,
                new_decode_page_ids=(19,),
                num_cached_tokens=7,
            ),
        ),
        dropped=("完了",),
        released_ids=("released",),
    )

    assert decode_step_delta(encode_step_delta(delta)) == delta


def test_step_delta_tensor_codec_rejects_trailing_values():
    delta = StepDelta(chunks=(), new=(), updates=(), dropped=())

    with pytest.raises(ValueError, match="trailing"):
        decode_step_delta((*encode_step_delta(delta), 0))


def test_state_sync_delta_reconstructs_full_snapshot_each_step():
    # F4/TP: the delta broadcaster must reconstruct EXACTLY what snapshot_step
    # would produce every step — over prefill, decodes, a second request joining,
    # and the first finishing — so delta-broadcast TP == full-broadcast TP.
    from kairyu.engine.core.step_input import StateSync

    scheduler = _scheduler(max_seqs=2, budget=8)
    driver, worker = StateSync(), StateSync()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4, 5), max_new_tokens=3))

    def drive_one_step(token_map):
        plan = scheduler.schedule()
        chunks = plan.scheduled
        full = snapshot_step(chunks, scheduler.states).states_view()
        delta = driver.diff(chunks, scheduler.states)
        driver_view = driver.apply(delta)
        worker_view = worker.apply(delta)  # same delta over the wire
        # both reconstructions equal the full snapshot, field for field
        assert driver_view == full
        assert worker_view == full
        if chunks:
            scheduler.update(token_map)

    drive_one_step({})  # prefill "a" (samples token 0)
    scheduler.add_request(EngineRequest("b", (6, 7, 8, 9), max_new_tokens=2))
    drive_one_step({"a": 100})  # a decodes, b prefills
    drive_one_step({"a": 101, "b": 200})  # both decode; a hits max_new_tokens -> finishes
    drive_one_step({"b": 201})  # only b remains; "a" must be dropped from sync
    assert "a" not in worker._states


def test_state_sync_release_diff_is_transactional_until_apply():
    """A failed diff must not consume the old identity before it crosses the wire."""
    from kairyu.engine.core.step_input import StateSync

    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("reused", (1, 2, 3, 4)))
    plan = scheduler.schedule()
    sync = StateSync()
    sync.apply(sync.diff(plan.scheduled, scheduler.states))
    retained = dict(sync._states)

    with pytest.raises(KeyError):
        sync.diff(
            (*plan.scheduled, ScheduledChunk("missing", 1, True, 0)),
            scheduler.states,
            released_ids=("reused",),
        )

    assert sync._states == retained
    retry = sync.diff(
        plan.scheduled,
        scheduler.states,
        released_ids=("reused",),
    )
    assert [snapshot.request_id for snapshot in retry.new] == ["reused"]
    assert retry.updates == ()
    assert sync._states == retained
    sync.apply(retry)


def test_state_sync_propagates_output_epoch_as_full_snapshot():
    from kairyu.engine.core.step_input import StateSync

    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4)))
    plan = scheduler.schedule()
    sync = StateSync()
    sync.apply(sync.diff(plan.scheduled, scheduler.states))

    scheduler.states["a"].output_epoch += 1
    delta = sync.diff(plan.scheduled, scheduler.states)

    assert len(delta.new) == 1
    assert not delta.updates
    assert sync.apply(delta)["a"].output_epoch == 1
