"""StepExecutor: the capture/replay seam around decode execution (m17 D1).

ALL policy (bucketing, capture-once, padding, invalidation, eager fallback)
lives here and is CPU-tested against ``FakeGraphBackend``; the only
CUDA-touching lines are in ``cuda_graph_gpu.CudaGraphBackend``. The CUDA-graph
contract: per-bucket STATIC device buffers, inputs copied in place before
replay, outputs read from the static output buffer.

All four decode inputs — token_ids, positions, page_tables, seq_lens — are
static device tensors written IN PLACE by ``_copy_in`` (C5). A real CUDA graph
replays fixed kernels over fixed memory, so nothing may be a Python attribute
rebound after capture — it would be invisible to the graph and every replay
would attend over the capture-time scratch page.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from kairyu.engine.core.graph_buckets import bucket_for, decode_buckets


@dataclass(frozen=True)
class DecodeBatch:
    """Decode-shaped step input: one new token per sequence. Every field is a
    static device tensor so in-place writes are visible to a captured graph."""

    token_ids: torch.Tensor  # [B] int64
    positions: torch.Tensor  # [B] int64
    page_tables: torch.Tensor  # [B, max_pages] int32, padded with the scratch page
    seq_lens: torch.Tensor  # [B] int32

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def max_pages(self) -> int:
        return int(self.page_tables.shape[1])


def build_decode_batch(
    token_ids: Sequence[int],
    positions: Sequence[int],
    page_lists: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
    max_pages: int,
    *,
    scratch_page: int = 0,
    device: str | torch.device = "cpu",
) -> DecodeBatch:
    """Pad ragged per-sequence page lists into a [B, max_pages] int32 tensor."""
    batch = len(seq_lens)
    page_tables = torch.full(
        (batch, max_pages), scratch_page, dtype=torch.int32, device=device
    )
    for row, pages in enumerate(page_lists):
        if pages:
            page_tables[row, : len(pages)] = torch.tensor(
                list(pages[:max_pages]), dtype=torch.int32, device=device
            )
    return DecodeBatch(
        token_ids=torch.as_tensor(token_ids, dtype=torch.int64, device=device),
        positions=torch.as_tensor(positions, dtype=torch.int64, device=device),
        page_tables=page_tables,
        seq_lens=torch.as_tensor(seq_lens, dtype=torch.int32, device=device),
    )


DecodeFn = Callable[[DecodeBatch], torch.Tensor]  # -> logits/hidden [B, ...]


class EagerStepExecutor:
    """Default: run the model directly (today's behavior)."""

    def __init__(self, decode_fn: DecodeFn) -> None:
        self._decode_fn = decode_fn

    def execute_decode(self, batch: DecodeBatch) -> torch.Tensor:
        return self._decode_fn(batch)

    def invalidate(self) -> None:  # nothing captured
        return None


class FakeGraphBackend:
    """CPU test double honoring the real contract: capture binds the STATIC
    device buffers; replay re-runs the fn on those SAME buffers (reading their
    current in-place contents) and asserts the frozen shape."""

    def __init__(self) -> None:
        self.captures = 0
        self.replays = 0

    def capture(self, fn: DecodeFn, static_batch: DecodeBatch):
        self.captures += 1
        backend = self
        frozen_shape = static_batch.token_ids.shape

        class _Replayable:
            def replay(self) -> torch.Tensor:
                assert static_batch.token_ids.shape == frozen_shape, (
                    "static buffer shape drifted after capture"
                )
                backend.replays += 1
                return fn(static_batch)

        return _Replayable()


class SnapshotGraphBackend:
    """Faithful CUDA-graph model: captures the batch's static buffer OBJECTS and
    replays against them, so it sees IN-PLACE writes (what a real graph reads
    from fixed device memory) but never an attribute rebind. With all four
    inputs now in-place-written device tensors (C5), replay reflects the
    request's real page tables — which ``test_graph_replay_reflects_current_
    page_tables`` pins (it caught the pre-fix Python-attribute rebind)."""

    def __init__(self) -> None:
        self.captures = 0
        self.replays = 0

    def capture(self, fn: DecodeFn, static_batch: DecodeBatch):
        self.captures += 1
        backend = self

        class _Replayable:
            def replay(self) -> torch.Tensor:
                backend.replays += 1
                return fn(static_batch)  # reads the static buffers' live contents

        return _Replayable()


class GraphStepExecutor:
    """Bucketed capture/replay with padding and eager fallback (m17 D1/D2)."""

    def __init__(
        self,
        decode_fn: DecodeFn,
        graph_backend,
        max_batch: int,
        max_pages: int = 1,
        scratch_page: int = 0,
    ) -> None:
        self._decode_fn = decode_fn
        self._backend = graph_backend
        self._buckets = decode_buckets(max_batch)
        self._max_pages = max_pages
        self._scratch_page = scratch_page
        self._captured: dict[int, tuple[object, DecodeBatch]] = {}

    def execute_decode(self, batch: DecodeBatch) -> torch.Tensor:
        bucket = bucket_for(batch.batch_size, self._buckets)
        # oversize batch OR a page table wider than the captured static buffer:
        # never crash, run eager (D2)
        if bucket is None or batch.max_pages > self._max_pages:
            return self._decode_fn(batch)
        if bucket not in self._captured:
            self._capture(bucket)
        replayable, static = self._captured[bucket]
        self._copy_in(static, batch)
        out = replayable.replay()
        return out[: batch.batch_size]  # padding rows dropped

    def invalidate(self) -> None:
        """Weight swap / pool resize: every capture is stale."""
        self._captured.clear()

    def _capture(self, bucket: int) -> None:
        static = DecodeBatch(
            token_ids=torch.zeros(bucket, dtype=torch.int64),
            positions=torch.zeros(bucket, dtype=torch.int64),
            page_tables=torch.full(
                (bucket, self._max_pages), self._scratch_page, dtype=torch.int32
            ),
            seq_lens=torch.ones(bucket, dtype=torch.int32),
        )
        replayable = self._backend.capture(self._decode_fn, static)
        self._captured[bucket] = (replayable, static)

    def _copy_in(self, static: DecodeBatch, batch: DecodeBatch) -> None:
        """Copy real rows into the static device buffers IN PLACE (C5); padding
        rows point at the scratch page with seq_len 1 (their outputs dropped)."""
        size = batch.batch_size
        static.token_ids[:size] = batch.token_ids
        static.token_ids[size:] = 0
        static.positions[:size] = batch.positions
        static.positions[size:] = 0
        static.page_tables[:size, : batch.max_pages] = batch.page_tables
        static.page_tables[:size, batch.max_pages :] = self._scratch_page
        static.page_tables[size:] = self._scratch_page
        static.seq_lens[:size] = batch.seq_lens
        static.seq_lens[size:] = 1
