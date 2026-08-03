"""m17 D1/D2 gates: bucket policy + capture/replay lifecycle on the fake graph."""

import pytest
import torch

from kairyu.engine.core.graph_buckets import bucket_for, decode_buckets
from kairyu.engine.core.step_executor import (
    DecodeBatch,
    DecodePageTableCache,
    DecodeRowOwner,
    EagerStepExecutor,
    FakeGraphBackend,
    GraphStepExecutor,
    SnapshotGraphBackend,
    build_decode_batch,
)

# These executors run a synthetic decode_fn with no KV pool behind the page
# ids, so no id can corrupt anything; the runner gates pin the real reservation
# (m17 A5). The argument is required precisely so that choice is never implicit.
SCRATCH = 0


def _owned_batch(
    page_lists: list[tuple[int, ...]],
    owners: list[DecodeRowOwner],
    *,
    max_pages: int,
    scratch_page: int | None,
    cache: DecodePageTableCache | None = None,
) -> DecodeBatch:
    size = len(page_lists)
    return build_decode_batch(
        token_ids=[10 + row for row in range(size)],
        positions=[20 + row for row in range(size)],
        page_lists=page_lists,
        seq_lens=[21 + row for row in range(size)],
        max_pages=max_pages,
        scratch_page=scratch_page,
        page_table_cache=cache,
        row_owners=owners,
    )


def _batch(size: int, max_pages: int = 1) -> DecodeBatch:
    return build_decode_batch(
        token_ids=[10 + i for i in range(size)],
        positions=[5 + i for i in range(size)],
        page_lists=[(i,) for i in range(size)],
        seq_lens=[6] * size,
        max_pages=max_pages,
        scratch_page=SCRATCH,
    )


class TestDecodePageTableCache:
    def test_preserves_eager_and_graph_padding_contracts(self):
        owners = [DecodeRowOwner("a"), DecodeRowOwner("b")]
        page_lists = [(3, 4), (8,)]

        graph = _owned_batch(
            page_lists,
            owners,
            max_pages=4,
            scratch_page=91,
            cache=DecodePageTableCache(),
        )
        eager = _owned_batch(
            page_lists,
            owners,
            max_pages=4,
            scratch_page=None,
            cache=DecodePageTableCache(),
        )

        assert graph.page_tables.tolist() == [
            [3, 4, 91, 91],
            [8, 91, 91, 91],
        ]
        assert [row.pad_page for row in graph.page_table_rows] == [91, 91]
        assert eager.page_tables.tolist() == [
            [3, 4, 4, 4],
            [8, 8, 8, 8],
        ]
        assert [row.pad_page for row in eager.page_table_rows] == [4, 8]

    def test_reuses_storage_and_uploads_only_a_changed_suffix(self):
        cache = DecodePageTableCache()
        owners = [DecodeRowOwner("a"), DecodeRowOwner("b")]
        first = _owned_batch(
            [(3, 4), (8,)],
            owners,
            max_pages=4,
            scratch_page=91,
            cache=cache,
        )
        pointer = first.page_tables.data_ptr()
        first_stats = cache.stats(reset=True)

        same = _owned_batch(
            [(3, 4), (8,)],
            owners,
            max_pages=4,
            scratch_page=91,
            cache=cache,
        )
        same_stats = cache.stats(reset=True)
        grown = _owned_batch(
            [(3, 4, 5), (8,)],
            owners,
            max_pages=4,
            scratch_page=91,
            cache=cache,
        )
        grown_stats = cache.stats()

        assert first_stats["storage_allocations"] == 1
        assert first_stats["rows_uploaded"] == 2
        assert first_stats["elements_uploaded"] == 8
        assert same.page_tables.data_ptr() == pointer
        assert same_stats["storage_allocations"] == 0
        assert same_stats["rows_uploaded"] == 0
        assert same_stats["elements_uploaded"] == 0
        assert same_stats["row_hits"] == 2
        assert grown.page_tables.data_ptr() == pointer
        assert grown.page_tables.tolist()[0] == [3, 4, 5, 91]
        assert grown_stats["rows_uploaded"] == 1
        assert grown_stats["elements_uploaded"] == 1
        assert grown_stats["row_hits"] == 1

    def test_owner_and_page_changes_cannot_reuse_stale_rows(self):
        cache = DecodePageTableCache()
        owners = [DecodeRowOwner("a"), DecodeRowOwner("b")]
        _owned_batch(
            [(10, 11), (20, 21)],
            owners,
            max_pages=3,
            scratch_page=91,
            cache=cache,
        )
        cache.stats(reset=True)

        # A preempted slot can be allocated to another request with exactly the
        # same page ids. Ownership, not coincidental bytes, decides the hit.
        replacement_owners = [DecodeRowOwner("c"), owners[1]]
        _owned_batch(
            [(10, 11), (20, 21)],
            replacement_owners,
            max_pages=3,
            scratch_page=91,
            cache=cache,
        )
        replacement_stats = cache.stats(reset=True)
        assert replacement_stats["rows_uploaded"] == 1
        assert replacement_stats["elements_uploaded"] == 3
        assert replacement_stats["row_hits"] == 1

        reallocated = _owned_batch(
            [(10, 12), (20, 21)],
            replacement_owners,
            max_pages=3,
            scratch_page=91,
            cache=cache,
        )
        reallocated_stats = cache.stats(reset=True)
        assert reallocated.page_tables.tolist()[0] == [10, 12, 91]
        assert reallocated_stats["rows_uploaded"] == 1
        assert reallocated_stats["elements_uploaded"] == 1
        assert reallocated_stats["row_hits"] == 1

        reordered = _owned_batch(
            [(20, 21), (10, 12)],
            [replacement_owners[1], replacement_owners[0]],
            max_pages=3,
            scratch_page=91,
            cache=cache,
        )
        reordered_stats = cache.stats()
        assert reordered.page_tables.tolist() == [
            [20, 21, 91],
            [10, 12, 91],
        ]
        assert reordered_stats["rows_uploaded"] == 2
        assert reordered_stats["elements_uploaded"] == 6
        assert reordered_stats["row_hits"] == 0

    def test_shape_shrink_regrow_forgets_removed_rows_and_hidden_columns(self):
        cache = DecodePageTableCache()
        owners = [DecodeRowOwner("a"), DecodeRowOwner("b")]
        first = _owned_batch(
            [(1, 2, 3, 4), (5, 6, 7, 8)],
            owners,
            max_pages=4,
            scratch_page=91,
            cache=cache,
        )
        pointer = first.page_tables.data_ptr()
        cache.stats(reset=True)

        shrunk = _owned_batch(
            [(1, 2)],
            owners[:1],
            max_pages=2,
            scratch_page=91,
            cache=cache,
        )
        shrink_stats = cache.stats(reset=True)
        assert shrunk.page_tables.tolist() == [[1, 2]]
        assert shrink_stats["rows_uploaded"] == 0
        assert shrink_stats["row_hits"] == 1

        regrown = _owned_batch(
            [(1, 2, 3, 4), (5, 6, 7, 8)],
            owners,
            max_pages=4,
            scratch_page=91,
            cache=cache,
        )
        regrow_stats = cache.stats()
        assert regrown.page_tables.data_ptr() == pointer
        assert regrown.page_tables.tolist() == [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
        # Row a must rewrite the two columns hidden by the narrow view. Row b
        # was removed and must rewrite its complete row on request-id reuse.
        assert regrow_stats["rows_uploaded"] == 2
        assert regrow_stats["elements_uploaded"] == 6
        assert regrow_stats["row_hits"] == 0

    def test_release_invalidates_every_lane_before_request_id_reuse(self):
        cache = DecodePageTableCache()
        owners = [
            DecodeRowOwner("request", lane=0),
            DecodeRowOwner("request", lane=1),
            DecodeRowOwner("survivor"),
        ]
        _owned_batch(
            [(1,), (2,), (3,)],
            owners,
            max_pages=2,
            scratch_page=91,
            cache=cache,
        )
        cache.stats(reset=True)

        cache.release("request")
        release_stats = cache.stats(reset=True)
        assert release_stats["released_rows"] == 2

        reused = _owned_batch(
            [(1,), (2,), (3,)],
            owners,
            max_pages=2,
            scratch_page=91,
            cache=cache,
        )
        reuse_stats = cache.stats()
        assert reused.page_tables.tolist() == [
            [1, 91],
            [2, 91],
            [3, 91],
        ]
        assert reuse_stats["rows_uploaded"] == 2
        assert reuse_stats["elements_uploaded"] == 4
        assert reuse_stats["row_hits"] == 1

    def test_capacity_growth_allocates_once_then_remains_bounded(self):
        cache = DecodePageTableCache()
        first = _owned_batch(
            [(1,)],
            [DecodeRowOwner("a")],
            max_pages=1,
            scratch_page=91,
            cache=cache,
        )
        first_pointer = first.page_tables.data_ptr()
        cache.stats(reset=True)

        grown = _owned_batch(
            [(1, 2, 3, 4), (5,), (6, 7)],
            [
                DecodeRowOwner("a"),
                DecodeRowOwner("b"),
                DecodeRowOwner("c"),
            ],
            max_pages=4,
            scratch_page=91,
            cache=cache,
        )
        growth_stats = cache.stats(reset=True)
        assert grown.page_tables.data_ptr() != first_pointer
        assert growth_stats["storage_allocations"] == 1
        assert growth_stats["storage_elements_copied"] == 1
        assert growth_stats["rows_uploaded"] == 3
        # The old row is copied D2D into the grown storage, so only its newly
        # visible suffix plus the two new rows are uploaded from the host.
        assert growth_stats["elements_uploaded"] == 11
        assert growth_stats["row_capacity"] == 4
        assert growth_stats["page_capacity"] == 4

        within_capacity = _owned_batch(
            [(1, 2, 3), (5,)],
            [DecodeRowOwner("a"), DecodeRowOwner("b")],
            max_pages=3,
            scratch_page=91,
            cache=cache,
        )
        bounded_stats = cache.stats()
        assert within_capacity.page_tables.data_ptr() == grown.page_tables.data_ptr()
        assert bounded_stats["storage_allocations"] == 0
        assert bounded_stats["row_capacity"] == 4
        assert bounded_stats["page_capacity"] == 4


def _decode_fn(batch: DecodeBatch) -> torch.Tensor:
    # deterministic function of the inputs: exposes any copy-in mistake
    return batch.token_ids[:, None].float() * 2 + batch.positions[:, None].float()


class TestBuckets:
    def test_decode_buckets_shape(self):
        assert decode_buckets(32) == (1, 2, 4, 8, 16, 24, 32)
        assert decode_buckets(48) == (1, 2, 4, 8, 16, 24, 32, 40, 48)
        assert decode_buckets(3) == (1, 2, 3)
        with pytest.raises(ValueError):
            decode_buckets(0)

    def test_bucket_for(self):
        buckets = decode_buckets(32)
        assert bucket_for(1, buckets) == 1
        assert bucket_for(5, buckets) == 8
        assert bucket_for(32, buckets) == 32
        assert bucket_for(33, buckets) is None


class TestGraphStepExecutor:
    def test_captures_once_per_bucket_and_replays(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=32, scratch_page=SCRATCH)
        for _ in range(4):
            executor.execute_decode(_batch(3))  # bucket 4
        assert backend.captures == 1
        assert backend.replays == 4
        executor.execute_decode(_batch(10))  # bucket 16 -> new capture
        assert backend.captures == 2

    def test_outputs_match_eager_and_padding_dropped(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=32, scratch_page=SCRATCH)
        batch = _batch(5)  # padded to bucket 8
        graph_out = executor.execute_decode(batch)
        eager_out = EagerStepExecutor(_decode_fn).execute_decode(batch)
        assert graph_out.shape[0] == 5  # padding rows dropped
        assert torch.equal(graph_out, eager_out)

    def test_copy_in_refreshes_values_between_replays(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=8, scratch_page=SCRATCH)
        first = executor.execute_decode(_batch(2))
        shifted = build_decode_batch(
            token_ids=[99, 100], positions=[7, 8],
            page_lists=[(0,), (1,)], seq_lens=[8, 9], max_pages=1,
            scratch_page=SCRATCH,
        )
        second = executor.execute_decode(shifted)
        assert not torch.equal(first, second)
        assert torch.equal(second[:, 0], torch.tensor([99.0 * 2 + 7, 100.0 * 2 + 8]))

    def test_oversize_batch_falls_back_to_eager(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=4, scratch_page=SCRATCH)
        out = executor.execute_decode(_batch(6))
        assert backend.captures == 0  # never captured
        assert out.shape[0] == 6

    def test_wide_page_table_falls_back_to_eager(self):
        # a page table wider than the captured static buffer runs eager, never
        # silently truncated (C5)
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(
            _decode_fn, backend, max_batch=8, max_pages=2, scratch_page=SCRATCH
        )
        wide = build_decode_batch(
            token_ids=[1], positions=[1], page_lists=[(0, 1, 2)], seq_lens=[9],
            max_pages=3, scratch_page=SCRATCH,
        )
        executor.execute_decode(wide)
        assert backend.captures == 0

    def test_invalidate_forces_recapture(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=8, scratch_page=SCRATCH)
        executor.execute_decode(_batch(2))
        executor.invalidate()
        executor.execute_decode(_batch(2))
        assert backend.captures == 2

    def test_pad_rows_use_scratch_page(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=8, scratch_page=7)
        executor.execute_decode(_batch(3))
        _, static = executor._captured[4]
        assert int(static.page_tables[3, 0]) == 7  # padding row -> scratch page
        assert int(static.seq_lens[3]) == 1

    def test_copy_in_refreshes_write_masks_between_replays(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(
            _decode_fn, backend, max_batch=8, scratch_page=SCRATCH
        )
        first = build_decode_batch(
            token_ids=[1, 2],
            positions=[3, 4],
            page_lists=[(0,), (1,)],
            seq_lens=[4, 5],
            max_pages=1,
            scratch_page=SCRATCH,
            write_from=[0, 5],
        )
        second = build_decode_batch(
            token_ids=[1, 2],
            positions=[4, 5],
            page_lists=[(0,), (1,)],
            seq_lens=[5, 6],
            max_pages=1,
            scratch_page=SCRATCH,
            write_from=[5, 0],
        )

        executor.execute_decode(first)
        executor.execute_decode(second)
        _, static = executor._captured[2]

        assert static.write_from.tolist() == [5, 0]


def _decode_fn_pages(batch: DecodeBatch) -> torch.Tensor:
    """Output depends on each sequence's first page id — a proxy for 'attends
    over the right KV pages'. If the page table doesn't reach replay, it's wrong."""
    return batch.page_tables[:, 0].to(torch.float32)[:, None]


def test_graph_replay_reflects_current_page_tables():
    # C5 (fixed): page_tables is now a static device buffer written in place, so
    # a faithful graph (SnapshotGraphBackend, which sees in-place writes but not
    # attribute rebinds) attends over the request's REAL pages, not the
    # capture-time scratch page.
    backend = SnapshotGraphBackend()
    executor = GraphStepExecutor(_decode_fn_pages, backend, max_batch=8, scratch_page=0)
    batch = build_decode_batch(
        token_ids=[10, 11, 12], positions=[5, 6, 7],
        page_lists=[(3,), (4,), (5,)], seq_lens=[6, 6, 6], max_pages=1,
        scratch_page=0,
    )
    out = executor.execute_decode(batch)
    assert out.flatten().tolist() == [3.0, 4.0, 5.0]  # not the scratch page (0)


def test_graph_page_table_copy_tracks_hits_suffix_growth_and_removal():
    scratch = 91
    backend = SnapshotGraphBackend()
    executor = GraphStepExecutor(
        _decode_fn_pages,
        backend,
        max_batch=4,
        max_pages=4,
        scratch_page=scratch,
    )
    owners = [DecodeRowOwner(name) for name in ("a", "b", "c", "d")]
    first = _owned_batch(
        [(1,), (2,), (3,), (4,)],
        owners,
        max_pages=2,
        scratch_page=scratch,
    )

    executor.execute_decode(first)
    _, static = executor._captured[4]
    static_pointer = static.page_tables.data_ptr()
    first_stats = executor.page_table_execution_stats(reset=True)
    assert static.page_tables.tolist() == [
        [1, scratch, scratch, scratch],
        [2, scratch, scratch, scratch],
        [3, scratch, scratch, scratch],
        [4, scratch, scratch, scratch],
    ]
    assert first_stats["rows_copied"] == 4
    assert first_stats["elements_copied"] == 16
    assert first_stats["row_hits"] == 0

    executor.execute_decode(first)
    hit_stats = executor.page_table_execution_stats(reset=True)
    assert hit_stats["rows_copied"] == 0
    assert hit_stats["elements_copied"] == 0
    assert hit_stats["row_hits"] == 4

    grown = _owned_batch(
        [(1, 8), (2,), (3,), (4,)],
        owners,
        max_pages=2,
        scratch_page=scratch,
    )
    executor.execute_decode(grown)
    suffix_stats = executor.page_table_execution_stats(reset=True)
    assert static.page_tables[0].tolist() == [1, 8, scratch, scratch]
    assert suffix_stats["rows_copied"] == 1
    assert suffix_stats["elements_copied"] == 1
    assert suffix_stats["row_hits"] == 3

    executor.release("a")
    released_stats = executor.page_table_execution_stats(reset=True)
    assert released_stats["released_rows"] == 1
    assert released_stats["live_rows"] == 3
    executor.execute_decode(grown)
    reused_stats = executor.page_table_execution_stats(reset=True)
    assert reused_stats["rows_copied"] == 1
    assert reused_stats["elements_copied"] == 4
    assert reused_stats["row_hits"] == 3

    # Releasing a row forgets ownership before the next step, but its captured
    # bytes still need an ordered scratch clear if that row becomes padding.
    executor.release("d")
    release_before_shrink = executor.page_table_execution_stats(reset=True)
    assert release_before_shrink["released_rows"] == 1

    removed = _owned_batch(
        [(1, 8), (2,), (3,)],
        owners[:3],
        max_pages=2,
        scratch_page=scratch,
    )
    executor.execute_decode(removed)
    removal_stats = executor.page_table_execution_stats()
    assert static.page_tables.data_ptr() == static_pointer
    assert static.page_tables[3].tolist() == [scratch] * 4
    assert removal_stats["rows_copied"] == 1
    assert removal_stats["elements_copied"] == 4
    assert removal_stats["row_hits"] == 3


def test_graph_batches_without_owner_metadata_keep_full_copy_compatibility():
    scratch = 91
    executor = GraphStepExecutor(
        _decode_fn_pages,
        SnapshotGraphBackend(),
        max_batch=4,
        max_pages=3,
        scratch_page=scratch,
    )
    batch = build_decode_batch(
        token_ids=[10, 11, 12],
        positions=[5, 6, 7],
        page_lists=[(3,), (4,), (5,)],
        seq_lens=[6, 6, 6],
        max_pages=2,
        scratch_page=scratch,
    )
    assert batch.page_table_rows == ()

    executor.execute_decode(batch)
    first_stats = executor.page_table_execution_stats(reset=True)
    executor.execute_decode(batch)
    second_stats = executor.page_table_execution_stats()

    for stats in (first_stats, second_stats):
        assert stats["rows_copied"] == 4
        assert stats["elements_copied"] == 12
        assert stats["row_hits"] == 0


def test_page_table_dispatch_reset_does_not_erase_graph_dispatch_stats():
    executor = GraphStepExecutor(
        _decode_fn_pages,
        SnapshotGraphBackend(),
        max_batch=2,
        max_pages=2,
        scratch_page=91,
    )
    executor.execute_decode(_owned_batch(
        [(3,)],
        [DecodeRowOwner("request")],
        max_pages=1,
        scratch_page=91,
    ))

    page_dispatch = executor.page_table_dispatch_stats(reset=True)
    graph_dispatch = executor.execution_stats()
    assert page_dispatch == {
        "graph_executions": 1,
        "eager_fallbacks": 0,
        "captured_buckets": (1,),
    }
    assert graph_dispatch == page_dispatch
    assert executor.page_table_dispatch_stats()["graph_executions"] == 0
    assert executor.execution_stats()["graph_executions"] == 1


# --- the step-boundary plan hook (GraphDecodeBackend, review [P1]) ------------


class _PlanRecorder:
    """Stands in for a backend that MUST be planned (FlashInfer).

    Records what the buffers held at each plan call, so the tests can pin not
    just that the hook fires but WHEN — a plan taken before ``_copy_in`` would
    describe the previous step and is the failure mode that matters.
    """

    def __init__(self) -> None:
        self.plans: list[tuple[list[list[int]], list[int]]] = []

    def __call__(self, batch: DecodeBatch) -> None:
        self.plans.append(
            (batch.page_tables.tolist(), batch.seq_lens.tolist())
        )


def test_the_plan_hook_runs_before_the_first_capture():
    """A planning backend cannot plan for itself under capture, so the capture
    must be preceded by one. Without this the first capture raises."""
    backend = FakeGraphBackend()
    plan = _PlanRecorder()
    captured_at_plan: list[int] = []

    def _fn(batch: DecodeBatch) -> torch.Tensor:
        captured_at_plan.append(len(plan.plans))  # plans seen when the fn ran
        return _decode_fn(batch)

    executor = GraphStepExecutor(
        _fn, backend, max_batch=8, scratch_page=SCRATCH, plan_fn=plan
    )
    executor.execute_decode(_batch(2))
    assert backend.captures == 1
    assert captured_at_plan and captured_at_plan[0] >= 1, (
        "the capture ran decode_fn before any plan_decode"
    )


def test_the_plan_hook_runs_after_copy_in_before_every_replay():
    """The reviewer's exact requirement: the backend is re-planned against the
    SAME padded static buffers the replay is about to read."""
    backend = SnapshotGraphBackend()
    plan = _PlanRecorder()
    executor = GraphStepExecutor(
        _decode_fn_pages, backend, max_batch=8, max_pages=2,
        scratch_page=9, plan_fn=plan,
    )
    first = build_decode_batch(
        token_ids=[1, 2, 3], positions=[0, 1, 2], page_lists=[(3,), (4,), (5,)],
        seq_lens=[4, 4, 4], max_pages=1, scratch_page=9,
    )
    second = build_decode_batch(
        token_ids=[1, 2, 3], positions=[1, 2, 3], page_lists=[(6,), (7,), (8,)],
        seq_lens=[5, 5, 5], max_pages=1, scratch_page=9,
    )
    executor.execute_decode(first)
    executor.execute_decode(second)

    assert backend.captures == 1
    assert backend.replays == 2
    assert len(plan.plans) == 3  # one for the capture, one per replay
    # every replay's plan saw the post-copy-in buffers: real rows, then the
    # scratch-page padding row of bucket 4
    assert plan.plans[1] == ([[3, 9], [4, 9], [5, 9], [9, 9]], [4, 4, 4, 1])
    assert plan.plans[2] == ([[6, 9], [7, 9], [8, 9], [9, 9]], [5, 5, 5, 1])


def test_replay_plan_is_distinct_and_receives_current_padded_host_lengths():
    """Capture stays stock while replay gets immutable scheduler metadata."""

    stock = _PlanRecorder()
    replay_batches: list[DecodeBatch] = []

    def replay(batch: DecodeBatch) -> None:
        replay_batches.append(batch)

    executor = GraphStepExecutor(
        _decode_fn_pages,
        SnapshotGraphBackend(),
        max_batch=8,
        max_pages=2,
        scratch_page=9,
        plan_fn=stock,
        replay_plan_fn=replay,
    )
    batch = build_decode_batch(
        token_ids=[1, 2, 3],
        positions=[3, 4, 5],
        page_lists=[(3,), (4,), (5,)],
        seq_lens=[4, 5, 6],
        max_pages=1,
        scratch_page=9,
    )

    executor.execute_decode(batch)
    _, static = executor._captured[4]

    assert len(stock.plans) == 1  # initial wrapper/capture plan only
    assert len(replay_batches) == 1
    replay_batch = replay_batches[0]
    assert replay_batch.host_seq_lens == (4, 5, 6, 1)
    assert replay_batch.seq_lens.tolist() == [4, 5, 6, 1]
    assert replay_batch.seq_lens.data_ptr() == static.seq_lens.data_ptr()
    # The frozen capture seed remains honest; no stale metadata was mutated.
    assert static.host_seq_lens == (1, 1, 1, 1)
    assert batch.host_seq_lens == (4, 5, 6)


def test_replay_without_host_lengths_falls_back_to_stock_plan():
    stock = _PlanRecorder()
    replay_calls = 0

    def replay(_batch: DecodeBatch) -> None:
        nonlocal replay_calls
        replay_calls += 1

    executor = GraphStepExecutor(
        _decode_fn,
        SnapshotGraphBackend(),
        max_batch=2,
        scratch_page=0,
        plan_fn=stock,
        replay_plan_fn=replay,
    )
    batch = DecodeBatch(
        token_ids=torch.tensor([1]),
        positions=torch.tensor([1]),
        page_tables=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        write_from=torch.tensor([0]),
    )

    executor.execute_decode(batch)

    assert replay_calls == 0
    assert len(stock.plans) == 2  # capture initialization plus safe replay fallback


def test_the_plan_hook_also_runs_on_the_eager_fallback():
    """An oversize batch skips capture but still reaches attend_decode, which on
    a planning backend needs a live plan for THOSE buffers."""
    backend = FakeGraphBackend()
    plan = _PlanRecorder()
    executor = GraphStepExecutor(
        _decode_fn, backend, max_batch=4, scratch_page=SCRATCH, plan_fn=plan
    )
    executor.execute_decode(_batch(6))  # beyond the largest bucket
    assert backend.captures == 0
    assert len(plan.plans) == 1


def test_no_plan_hook_is_still_valid():
    """Backends with no host phase (the torch path) construct unchanged."""
    executor = GraphStepExecutor(
        _decode_fn, FakeGraphBackend(), max_batch=8, scratch_page=SCRATCH
    )
    executor.execute_decode(_batch(2))  # must not raise
