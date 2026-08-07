"""Issue #229: model-runner wiring for reusable decode page tables."""

from __future__ import annotations

import torch

# Reuse the small native model/state fixtures already used by the decode and
# speculative-verification contract suites.
from test_batched_prefill import _model, _pool, _state

from kairyu.engine.core.model_runner import PagedModelRunner
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_executor import DecodeRowOwner


def _runner() -> PagedModelRunner:
    model = _model()
    return PagedModelRunner(model, _pool(model))


def _decode_state(
    request_id: str,
    pages: tuple[int, ...],
    outputs: tuple[int, ...],
):
    state = _state(request_id, (1, 2, 3, 4), pages)
    state.outputs = list(outputs)
    return state


def _record_tensor_batches(runner: PagedModelRunner, monkeypatch):
    batches = []

    def record(batch):
        batches.append(batch)
        return torch.empty(
            (batch.batch_size, runner._model.config.hidden_size),
            dtype=torch.float32,
        )

    monkeypatch.setattr(runner, "_eager_tensor_hidden", record)
    return batches


def test_normal_decode_uses_stable_owners_and_exact_on_off_wiring(monkeypatch):
    runner = _runner()
    batches = _record_tensor_batches(runner, monkeypatch)
    chunks = [
        ScheduledChunk("a", 1, False, 1),
        ScheduledChunk("b", 1, False, 1),
    ]
    states = {
        "a": _decode_state("a", (2,), (11,)),
        "b": _decode_state("b", (3,), (12,)),
    }

    runner._execute_decode_batch(chunks, states, None)
    first = batches[-1]
    first_stats = runner.decode_page_table_cache_stats(reset=True)
    assert tuple(row.owner for row in first.page_table_rows) == (
        DecodeRowOwner("a"),
        DecodeRowOwner("b"),
    )
    assert first.page_tables.tolist() == [[2], [3]]
    assert first_stats["enabled"] is True
    assert first_stats["builds"] == 1
    assert first_stats["rows"] == 2
    assert first_stats["legacy_outer_allocations"] == 0
    assert first_stats["cache"]["storage_allocations"] == 1
    assert first_stats["cache"]["rows_uploaded"] == 2

    runner._execute_decode_batch(chunks, states, None)
    second = batches[-1]
    hit_stats = runner.decode_page_table_cache_stats(reset=True)
    assert second.page_tables.data_ptr() == first.page_tables.data_ptr()
    assert hit_stats["cache"]["storage_allocations"] == 0
    assert hit_stats["cache"]["rows_uploaded"] == 0
    assert hit_stats["cache"]["row_hits"] == 2

    runner.set_decode_page_table_cache_enabled(False)
    runner._execute_decode_batch(chunks, states, None)
    legacy = batches[-1]
    legacy_stats = runner.decode_page_table_cache_stats()
    assert legacy.page_tables.tolist() == first.page_tables.tolist()
    assert legacy.page_table_rows == ()
    assert legacy.page_tables.data_ptr() != first.page_tables.data_ptr()
    assert legacy_stats["enabled"] is False
    assert legacy_stats["builds"] == 1
    assert legacy_stats["rows"] == 2
    assert legacy_stats["host_page_ids_visited"] == 2
    assert legacy_stats["legacy_outer_allocations"] == 1
    assert legacy_stats["legacy_row_allocations"] == 2
    assert legacy_stats["legacy_elements_written"] == 4
    assert legacy_stats["legacy_owned_elements_uploaded"] == 2
    assert legacy_stats["cache"]["storage_allocations"] == 0
    assert legacy_stats["cache"]["rows_uploaded"] == 0
    assert legacy_stats["cache"]["row_hits"] == 0


def test_eager_tensor_plan_receives_authoritative_host_lengths(monkeypatch):
    runner = _runner()
    batch = runner._build_tensor_decode_batch(
        tokens=torch.tensor([11, 12]),
        positions=torch.tensor([4, 6]),
        page_tables=[(2, 3), (4, 5)],
        seq_lens=[5, 7],
        max_pages=2,
        scratch_page=None,
        write_from=[0, 0],
        row_owners=[DecodeRowOwner("a"), DecodeRowOwner("b")],
    )
    plan_kwargs = []

    def plan(_pool, _tables, _lengths, **kwargs):
        plan_kwargs.append(kwargs)

    monkeypatch.setattr(runner._model, "plan_decode_tensors", plan)
    monkeypatch.setattr(
        runner._model,
        "forward_decode_tensors",
        lambda *_args: torch.empty((2, runner._model.config.hidden_size)),
    )

    runner._eager_tensor_hidden(batch)

    assert plan_kwargs == [{"host_seq_lens": (5, 7)}]


def test_decode_inputs_use_the_scheduler_tuple_snapshot_without_list_walks():
    class _UnwalkablePages(list):
        def __iter__(self):
            raise AssertionError("live decode pages must not be walked per step")

    runner = _runner()
    state = _decode_state("request", (2, 3), (11,))
    state.decode_pages = _UnwalkablePages([99])
    state.decode_pages_snapshot = (4, 5)

    _, _, page_table, _ = runner._decode_inputs(
        ScheduledChunk("request", 1, False, 1),
        state,
    )

    assert page_table == (2, 3, 4, 5)
    assert isinstance(page_table, tuple)


def test_speculative_rows_use_request_local_lanes_starting_at_one(monkeypatch):
    runner = _runner()
    batches = _record_tensor_batches(runner, monkeypatch)
    chunk = ScheduledChunk("request", 3, False, 1)
    state = _decode_state("request", (4, 5), (11, 12, 13))

    rows = runner._verification_decode_rows(
        [chunk],
        {"request": state},
    )
    assert rows[0] == [
        ScheduledChunk("request", 1, False, 1),
        ScheduledChunk("request", 1, False, 2),
        ScheduledChunk("request", 1, False, 3),
    ]
    assert rows[-1] == [
        DecodeRowOwner("request", 1),
        DecodeRowOwner("request", 2),
        DecodeRowOwner("request", 3),
    ]

    runner._execute_verification_batch(
        [chunk],
        {"request": state},
        None,
    )
    batch = batches[-1]
    assert tuple(row.owner for row in batch.page_table_rows) == tuple(rows[-1])
    stats = runner.decode_page_table_cache_stats()
    assert stats["builds"] == 1
    assert stats["rows"] == 3
    assert stats["cache"]["rows_uploaded"] == 3
    assert stats["cache"]["elements_uploaded"] == 6


def test_release_invalidates_cached_owner_before_request_id_reuse():
    runner = _runner()
    owner = DecodeRowOwner("request")
    build = {
        "tokens": torch.tensor([11]),
        "positions": torch.tensor([4]),
        "page_tables": [[4, 5]],
        "seq_lens": [5],
        "max_pages": 2,
        "scratch_page": None,
        "write_from": [0],
        "row_owners": [owner],
    }

    first = runner._build_tensor_decode_batch(**build)
    runner.decode_page_table_cache_stats(reset=True)
    runner.release("request")
    release_stats = runner.decode_page_table_cache_stats(reset=True)
    assert release_stats["cache"]["released_rows"] == 1

    reused = runner._build_tensor_decode_batch(**build)
    reuse_stats = runner.decode_page_table_cache_stats()
    assert reused.page_tables.data_ptr() == first.page_tables.data_ptr()
    assert reused.page_tables.tolist() == [[4, 5]]
    assert reuse_stats["cache"]["storage_allocations"] == 0
    assert reuse_stats["cache"]["rows_uploaded"] == 1
    assert reuse_stats["cache"]["elements_uploaded"] == 2
    assert reuse_stats["cache"]["row_hits"] == 0
