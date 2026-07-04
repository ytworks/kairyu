"""StepExecutor: the capture/replay seam around decode execution (m17 D1).

ALL policy (bucketing, capture-once, padding, invalidation, eager fallback)
lives here and is CPU-tested against ``FakeGraphBackend``; the only
CUDA-touching lines are in ``cuda_graph_gpu.CudaGraphBackend``. The CUDA-graph
contract: per-bucket STATIC input buffers, inputs copied in before replay,
outputs read from the static output buffer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from kairyu.engine.core.graph_buckets import bucket_for, decode_buckets


@dataclass(frozen=True)
class DecodeBatch:
    """Decode-shaped step input: one new token per sequence."""

    token_ids: torch.Tensor  # [B] int64
    positions: torch.Tensor  # [B] int64
    page_tables: tuple[tuple[int, ...], ...]  # per-sequence
    seq_lens: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])


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
    """CPU test double honoring the real contract: capture returns a replayable
    bound to STATIC buffers; replay re-runs the fn on those buffers and
    asserts the frozen shape (a real graph would silently corrupt instead).

    NOTE: this fake re-invokes ``fn(static_batch)`` at replay time, so it reads
    whatever the executor most recently wrote onto ``static_batch`` — including
    the page_tables/seq_lens that ``GraphStepExecutor._copy_in`` rebinds via
    ``object.__setattr__``. A REAL CUDA graph cannot: it replays fixed kernels
    over fixed device memory and never sees post-capture Python-attribute
    mutation. ``SnapshotGraphBackend`` models that faithfully (see C5).
    """

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
    """Faithful CUDA-graph model for the contract test (C5).

    Captures the batch's inputs AT CAPTURE TIME and replays against that frozen
    snapshot, ignoring any later Python-attribute mutation — exactly what a real
    graph does. Inputs a real graph reads from static DEVICE buffers (token_ids,
    positions) are read live from the captured tensor object (in-place writes are
    visible); inputs the executor rebinds as Python attributes (page_tables,
    seq_lens) are frozen at capture, so post-capture changes are INVISIBLE.

    Used to prove GraphStepExecutor currently mis-handles page_tables/seq_lens:
    until they become in-place-written static device buffers, replay attends over
    the capture-time scratch page instead of the request's real pages.
    """

    def __init__(self) -> None:
        self.captures = 0
        self.replays = 0

    def capture(self, fn: DecodeFn, static_batch: DecodeBatch):
        self.captures += 1
        backend = self
        frozen_pages = static_batch.page_tables  # snapshot: real graph bakes these in
        frozen_seq_lens = static_batch.seq_lens
        live_tokens = static_batch.token_ids  # device buffer: in-place writes seen
        live_positions = static_batch.positions

        class _Replayable:
            def replay(self) -> torch.Tensor:
                backend.replays += 1
                # replay sees live device buffers but the CAPTURE-TIME page table
                snapshot = DecodeBatch(
                    token_ids=live_tokens,
                    positions=live_positions,
                    page_tables=frozen_pages,
                    seq_lens=frozen_seq_lens,
                )
                return fn(snapshot)

        return _Replayable()


class GraphStepExecutor:
    """Bucketed capture/replay with padding and eager fallback (m17 D1/D2)."""

    def __init__(
        self,
        decode_fn: DecodeFn,
        graph_backend,
        max_batch: int,
        scratch_page: int = 0,
    ) -> None:
        self._decode_fn = decode_fn
        self._backend = graph_backend
        self._buckets = decode_buckets(max_batch)
        self._scratch_page = scratch_page
        self._captured: dict[int, tuple[object, DecodeBatch]] = {}

    def execute_decode(self, batch: DecodeBatch) -> torch.Tensor:
        bucket = bucket_for(batch.batch_size, self._buckets)
        if bucket is None:  # oversize: never crash, run eager (D2)
            return self._decode_fn(batch)
        if bucket not in self._captured:
            self._capture(bucket)
        replayable, static = self._captured[bucket]
        self._copy_in(static, batch, bucket)
        out = replayable.replay()
        return out[: batch.batch_size]  # padding rows dropped

    def invalidate(self) -> None:
        """Weight swap / pool resize: every capture is stale."""
        self._captured.clear()

    def _capture(self, bucket: int) -> None:
        static = DecodeBatch(
            token_ids=torch.zeros(bucket, dtype=torch.int64),
            positions=torch.zeros(bucket, dtype=torch.int64),
            page_tables=((self._scratch_page,),) * bucket,
            seq_lens=(1,) * bucket,
        )
        replayable = self._backend.capture(self._decode_fn, static)
        self._captured[bucket] = (replayable, static)

    def _copy_in(self, static: DecodeBatch, batch: DecodeBatch, bucket: int) -> None:
        """The static-buffer contract: copy real rows, point padding rows at
        the scratch page with seq_len 1 (their outputs are dropped)."""
        static.token_ids[: batch.batch_size] = batch.token_ids
        static.token_ids[batch.batch_size :] = 0
        static.positions[: batch.batch_size] = batch.positions
        static.positions[batch.batch_size :] = 0
        pad = ((self._scratch_page,),) * (bucket - batch.batch_size)
        # frozen dataclass: page tables/seq_lens are rebuilt via object.__setattr__
        object.__setattr__(static, "page_tables", batch.page_tables + pad)
        object.__setattr__(
            static, "seq_lens", batch.seq_lens + (1,) * (bucket - batch.batch_size)
        )
