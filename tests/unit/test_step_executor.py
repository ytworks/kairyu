"""m17 D1/D2 gates: bucket policy + capture/replay lifecycle on the fake graph."""

import pytest
import torch

from kairyu.engine.core.graph_buckets import bucket_for, decode_buckets
from kairyu.engine.core.step_executor import (
    DecodeBatch,
    EagerStepExecutor,
    FakeGraphBackend,
    GraphStepExecutor,
    SnapshotGraphBackend,
    build_decode_batch,
)


def _batch(size: int, max_pages: int = 1) -> DecodeBatch:
    return build_decode_batch(
        token_ids=[10 + i for i in range(size)],
        positions=[5 + i for i in range(size)],
        page_lists=[(i,) for i in range(size)],
        seq_lens=[6] * size,
        max_pages=max_pages,
    )


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
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=32)
        for _ in range(4):
            executor.execute_decode(_batch(3))  # bucket 4
        assert backend.captures == 1
        assert backend.replays == 4
        executor.execute_decode(_batch(10))  # bucket 16 -> new capture
        assert backend.captures == 2

    def test_outputs_match_eager_and_padding_dropped(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=32)
        batch = _batch(5)  # padded to bucket 8
        graph_out = executor.execute_decode(batch)
        eager_out = EagerStepExecutor(_decode_fn).execute_decode(batch)
        assert graph_out.shape[0] == 5  # padding rows dropped
        assert torch.equal(graph_out, eager_out)

    def test_copy_in_refreshes_values_between_replays(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=8)
        first = executor.execute_decode(_batch(2))
        shifted = build_decode_batch(
            token_ids=[99, 100], positions=[7, 8],
            page_lists=[(0,), (1,)], seq_lens=[8, 9], max_pages=1,
        )
        second = executor.execute_decode(shifted)
        assert not torch.equal(first, second)
        assert torch.equal(second[:, 0], torch.tensor([99.0 * 2 + 7, 100.0 * 2 + 8]))

    def test_oversize_batch_falls_back_to_eager(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=4)
        out = executor.execute_decode(_batch(6))
        assert backend.captures == 0  # never captured
        assert out.shape[0] == 6

    def test_wide_page_table_falls_back_to_eager(self):
        # a page table wider than the captured static buffer runs eager, never
        # silently truncated (C5)
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=8, max_pages=2)
        wide = build_decode_batch(
            token_ids=[1], positions=[1], page_lists=[(0, 1, 2)], seq_lens=[9],
            max_pages=3,
        )
        executor.execute_decode(wide)
        assert backend.captures == 0

    def test_invalidate_forces_recapture(self):
        backend = FakeGraphBackend()
        executor = GraphStepExecutor(_decode_fn, backend, max_batch=8)
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
    )
    out = executor.execute_decode(batch)
    assert out.flatten().tolist() == [3.0, 4.0, 5.0]  # not the scratch page (0)
