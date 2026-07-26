"""StepExecutor: the capture/replay seam around decode execution (m17 D1).

ALL policy (bucketing, capture-once, padding, invalidation, eager fallback)
lives here and is CPU-tested against ``FakeGraphBackend``; the only
CUDA-touching lines are in ``cuda_graph_gpu.CudaGraphBackend``. The CUDA-graph
contract: per-bucket STATIC device buffers, inputs copied in place before
replay, outputs read from the static output buffer.

All five decode inputs — token_ids, positions, page_tables, seq_lens,
write_from — are
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
    write_from: torch.Tensor  # [B] int64; positions below this retain cached KV

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
    scratch_page: int | None,
    write_from: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
) -> DecodeBatch:
    """Pad ragged per-sequence page lists into a [B, max_pages] int32 tensor.

    ``scratch_page`` is REQUIRED and has no default. Graph callers pass a page
    the allocator can never return (m17 A5). Eager tensor decode passes ``None``
    and repeats each row's final owned page into its masked tail; it has no
    synthetic padding rows and therefore needs no capacity reservation.
    """
    batch = len(seq_lens)
    if write_from is None:
        write_from = [0] * batch
    if len(write_from) != batch:
        raise ValueError("write_from must have one entry per decode row")
    if scratch_page is None:
        if any(not pages for pages in page_lists):
            raise ValueError("eager decode page lists must not be empty")
        page_tables = torch.empty(
            (batch, max_pages), dtype=torch.int32, device=device
        )
    else:
        page_tables = torch.full(
            (batch, max_pages), scratch_page, dtype=torch.int32, device=device
        )
    for row, pages in enumerate(page_lists):
        if scratch_page is None:
            page_tables[row].fill_(pages[-1])
        if pages:
            page_tables[row, : len(pages)] = torch.tensor(
                list(pages[:max_pages]), dtype=torch.int32, device=device
            )
    return DecodeBatch(
        token_ids=torch.as_tensor(token_ids, dtype=torch.int64, device=device),
        positions=torch.as_tensor(positions, dtype=torch.int64, device=device),
        page_tables=page_tables,
        seq_lens=torch.as_tensor(seq_lens, dtype=torch.int32, device=device),
        write_from=torch.as_tensor(write_from, dtype=torch.int64, device=device),
    )


DecodeFn = Callable[[DecodeBatch], torch.Tensor]  # -> logits/hidden [B, ...]
PlanFn = Callable[[DecodeBatch], None]  # host phase, OUTSIDE the captured region


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
        *,
        scratch_page: int,
        max_pages: int = 1,
        device: str | torch.device = "cpu",
        plan_fn: PlanFn | None = None,
    ) -> None:
        """``scratch_page`` is REQUIRED (m17 A5, review [P1]).

        Every replay executes the FULL bucket: the rows past the real batch still
        run their KV write, and it lands on whatever page id the padded page
        table holds. A defaulted 0 pointed those writes at the first page a
        ``PagePool`` hands out, so a partial bucket overwrote a live request's
        K/V at slot 0 — and the capture warmup did it even for a full batch.
        There is no safe default; the owner of the pool must name a page its
        allocator can never return.
        """
        self._decode_fn = decode_fn
        # The step-boundary host hook (``GraphDecodeBackend``, review [P1]).
        # An attention backend that plans on the CPU — FlashInfer builds its
        # split-KV schedule there and cannot do it under capture — needs the
        # step boundary to reach it, and this executor is the only object that
        # knows where those boundaries are: it owns the static buffers, so it
        # is the only one that knows WHICH tensors the next replay will read.
        # Optional, so a decode_fn whose backends have no host phase (every
        # FakeGraphBackend test) constructs exactly as before.
        self._plan_fn = plan_fn
        self._backend = graph_backend
        self._buckets = decode_buckets(max_batch)
        self._max_pages = max_pages
        self._scratch_page = scratch_page
        # The static buffers must live where the captured kernels run. Without
        # this they were allocated on the host regardless of backend, so a real
        # CUDA graph would replay over memory the GPU never writes — the module
        # docstring already required device tensors, nothing enforced it, and
        # `build_decode_batch` (the other constructor) has taken a device all
        # along. Defaults to cpu so the FakeGraphBackend tests are unchanged.
        self._device = torch.device(device)
        self._captured: dict[int, tuple[object, DecodeBatch]] = {}

    def execute_decode(self, batch: DecodeBatch) -> torch.Tensor:
        bucket = bucket_for(batch.batch_size, self._buckets)
        # oversize batch OR a page table wider than the captured static buffer:
        # never crash, run eager (D2)
        if bucket is None or batch.max_pages > self._max_pages:
            self._plan(batch)  # eager still needs a live plan for THESE buffers
            return self._decode_fn(batch)
        if bucket not in self._captured:
            self._capture(bucket)
        replayable, static = self._captured[bucket]
        self._copy_in(static, batch)
        # AFTER copy-in, BEFORE replay: the plan describes the page table and
        # lengths the kernels are about to read, and _copy_in just rewrote both
        # (including the padding rows). Planning before the copy would schedule
        # the PREVIOUS step. This ordering is the whole point of the hook.
        self._plan(static)
        out = replayable.replay()
        return out[: batch.batch_size]  # padding rows dropped

    def invalidate(self) -> None:
        """Weight swap / pool resize: every capture is stale."""
        for replayable, _static in self._captured.values():
            close = getattr(replayable, "close", None)
            if close is not None:
                close()
        self._captured.clear()
        invalidate_backend = getattr(self._backend, "invalidate", None)
        if invalidate_backend is not None:
            invalidate_backend()

    def _plan(self, batch: DecodeBatch) -> None:
        """Backend host phase over ``batch``'s buffers — never under capture."""
        if self._plan_fn is not None:
            self._plan_fn(batch)

    def _capture(self, bucket: int) -> None:
        static = DecodeBatch(
            token_ids=torch.zeros(bucket, dtype=torch.int64, device=self._device),
            positions=torch.zeros(bucket, dtype=torch.int64, device=self._device),
            page_tables=torch.full(
                (bucket, self._max_pages),
                self._scratch_page,
                dtype=torch.int32,
                device=self._device,
            ),
            seq_lens=torch.ones(bucket, dtype=torch.int32, device=self._device),
            write_from=torch.zeros(bucket, dtype=torch.int64, device=self._device),
        )
        # Plan BEFORE the backend captures: the warmup passes and the capture
        # itself run decode_fn, which reaches attend_decode, which under capture
        # may not plan for itself. Without this the very first capture of a
        # planning backend dies with "no live plan".
        self._plan(static)
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
        static.write_from[:size] = batch.write_from
        static.write_from[size:] = 0
