"""m17 D1/D2 gates: bucket policy + capture/replay lifecycle on the fake graph."""

import pytest
import torch

from kairyu.engine.core.graph_buckets import bucket_for, decode_buckets
from kairyu.engine.core.step_executor import (
    DecodeBatch,
    EagerStepExecutor,
    FakeGraphBackend,
    GraphStepExecutor,
)


def _batch(size: int) -> DecodeBatch:
    return DecodeBatch(
        token_ids=torch.arange(size, dtype=torch.int64) + 10,
        positions=torch.arange(size, dtype=torch.int64) + 5,
        page_tables=tuple((i,) for i in range(size)),
        seq_lens=tuple(6 for _ in range(size)),
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
        shifted = DecodeBatch(
            token_ids=torch.tensor([99, 100]),
            positions=torch.tensor([7, 8]),
            page_tables=((0,), (1,)),
            seq_lens=(8, 9),
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
        assert static.page_tables[3] == (7,)
        assert static.seq_lens[3] == 1
