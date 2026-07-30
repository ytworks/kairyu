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
class DecodeRowOwner:
    """Stable ownership identity for one row of decode metadata.

    ``lane`` distinguishes flattened speculative rows of the same request while
    remaining stable as their absolute positions advance.  Request completion
    can therefore discard every lane without keeping a scheduler object alive.
    """

    request_id: str
    lane: int = 0


@dataclass(frozen=True)
class PageTableRow:
    """Host-side signature for one materialized page-table row."""

    owner: DecodeRowOwner
    pages: tuple[int, ...]
    pad_page: int


@dataclass(frozen=True)
class DecodeBatch:
    """Decode-shaped step input: one new token per sequence. Every field is a
    static device tensor so in-place writes are visible to a captured graph."""

    token_ids: torch.Tensor  # [B] int64
    positions: torch.Tensor  # [B] int64
    page_tables: torch.Tensor  # [B, max_pages] int32, padded with the scratch page
    seq_lens: torch.Tensor  # [B] int32
    write_from: torch.Tensor  # [B] int64; positions below this retain cached KV
    # Optional host metadata.  It never participates in model execution; a
    # reusable builder and graph bucket use it to prove which page-table ranges
    # are unchanged without reading a CUDA tensor back to the host.
    page_table_rows: tuple[PageTableRow, ...] = ()

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def max_pages(self) -> int:
        return int(self.page_tables.shape[1])


@dataclass
class _CachedPageTableRow:
    signature: PageTableRow
    visible_width: int


def _page_at(signature: PageTableRow, column: int) -> int:
    if column < len(signature.pages):
        return signature.pages[column]
    return signature.pad_page


def _dirty_span(
    previous: _CachedPageTableRow | None,
    current: PageTableRow,
    width: int,
) -> tuple[int, int] | None:
    """Return the smallest changed half-open range in ``current``.

    Columns outside the previous view are unknown even when an older, wider
    step happened to leave matching bytes in storage.  Treating them as dirty
    is what makes width shrink/regrow deterministic rather than stale-data
    dependent.
    """

    if width == 0:
        return None
    if previous is None or previous.signature.owner != current.owner:
        return 0, width
    if previous.visible_width == width and previous.signature == current:
        return None
    first: int | None = None
    last = 0
    for column in range(width):
        changed = (
            column >= previous.visible_width
            or _page_at(previous.signature, column) != _page_at(current, column)
        )
        if changed:
            first = column if first is None else first
            last = column + 1
    return None if first is None else (first, last)


class DecodePageTableCache:
    """One bounded, grow-only page-table buffer owned by a model runner.

    The cache has one tensor, not an entry per request or shape.  Capacity grows
    geometrically to the largest observed compatible batch and page width, so
    cache cardinality is fixed while steady decode preserves the tensor's data
    pointer.  Only host ownership signatures are retained; ``release`` removes
    them immediately and a new owner must rewrite its row before use.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = torch.device(device)
        self._buffer: torch.Tensor | None = None
        self._row_capacity = 0
        self._page_capacity = 0
        self._rows: list[_CachedPageTableRow | None] = []
        self._active_rows = 0
        self._storage_allocations = 0
        self._storage_elements_copied = 0
        self._rows_uploaded = 0
        self._elements_uploaded = 0
        self._row_hits = 0
        self._released_rows = 0
        self._removed_rows = 0
        self._full_row_uploads = 0
        self._partial_row_uploads = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def _grow(current: int, required: int) -> int:
        if required <= current:
            return current
        capacity = max(current, 1)
        while capacity < required:
            capacity *= 2
        return capacity

    def _ensure_capacity(self, rows: int, pages: int) -> None:
        next_rows = self._grow(self._row_capacity, rows)
        next_pages = self._grow(self._page_capacity, pages)
        if self._buffer is not None and (
            next_rows == self._row_capacity
            and next_pages == self._page_capacity
        ):
            return
        previous_buffer = self._buffer
        previous_rows = self._rows
        previous_page_capacity = self._page_capacity
        next_buffer = torch.empty(
            (next_rows, next_pages),
            dtype=torch.int32,
            device=self._device,
        )
        if previous_buffer is not None:
            copied_rows = min(self._active_rows, next_rows)
            copied_pages = min(
                max(
                    (
                        row.visible_width
                        for row in previous_rows[:copied_rows]
                        if row is not None
                    ),
                    default=0,
                ),
                previous_page_capacity,
                next_pages,
            )
            if copied_rows and copied_pages:
                next_buffer[:copied_rows, :copied_pages].copy_(
                    previous_buffer[:copied_rows, :copied_pages]
                )
                self._storage_elements_copied += copied_rows * copied_pages
        self._buffer = next_buffer
        self._row_capacity = next_rows
        self._page_capacity = next_pages
        self._rows = previous_rows[:next_rows] + [None] * max(
            0, next_rows - len(previous_rows)
        )
        self._storage_allocations += 1

    def materialize(
        self,
        page_lists: Sequence[Sequence[int]],
        *,
        max_pages: int,
        scratch_page: int | None,
        row_owners: Sequence[DecodeRowOwner],
    ) -> tuple[torch.Tensor, tuple[PageTableRow, ...]]:
        """Return a live view and update only changed page-table ranges."""

        batch = len(page_lists)
        if len(row_owners) != batch:
            raise ValueError("row_owners must have one entry per decode row")
        if max_pages < 0:
            raise ValueError("max_pages must be non-negative")
        self._ensure_capacity(batch, max_pages)
        assert self._buffer is not None

        signatures: list[PageTableRow] = []
        for row, (pages, owner) in enumerate(
            zip(page_lists, row_owners, strict=True)
        ):
            owned = tuple(int(page) for page in pages[:max_pages])
            if scratch_page is None:
                if not owned:
                    raise ValueError("eager decode page lists must not be empty")
                pad_page = owned[-1]
            else:
                pad_page = int(scratch_page)
            signature = PageTableRow(owner, owned, pad_page)
            signatures.append(signature)
            dirty = _dirty_span(self._rows[row], signature, max_pages)
            if dirty is None:
                self._row_hits += 1
            else:
                start, end = dirty
                values = torch.tensor(
                    [_page_at(signature, column) for column in range(start, end)],
                    dtype=torch.int32,
                )
                self._buffer[row, start:end].copy_(values)
                self._rows_uploaded += 1
                self._elements_uploaded += end - start
                if start == 0 and end == max_pages:
                    self._full_row_uploads += 1
                else:
                    self._partial_row_uploads += 1
            self._rows[row] = _CachedPageTableRow(signature, max_pages)

        # A row outside the returned view cannot be read.  Forget its ownership
        # now so a removed/preempted request never survives as a false hit when
        # that slot is reused later.
        for row in range(batch, self._active_rows):
            if self._rows[row] is not None:
                self._removed_rows += 1
            self._rows[row] = None
        self._active_rows = batch
        return self._buffer[:batch, :max_pages], tuple(signatures)

    def release(self, request_id: str) -> None:
        """Drop every speculative lane owned by ``request_id``."""

        for index, row in enumerate(self._rows):
            if row is None or row.signature.owner.request_id != request_id:
                continue
            self._rows[index] = None
            self._released_rows += 1

    def invalidate(self) -> None:
        """Forget ownership while retaining bounded reusable storage."""
        for index, row in enumerate(self._rows):
            if row is None:
                continue
            self._rows[index] = None
            self._removed_rows += 1
        self._active_rows = 0

    def stats(self, *, reset: bool = False) -> dict[str, int]:
        result = {
            "storage_allocations": self._storage_allocations,
            "storage_elements_copied": self._storage_elements_copied,
            "rows_uploaded": self._rows_uploaded,
            "elements_uploaded": self._elements_uploaded,
            "row_hits": self._row_hits,
            "released_rows": self._released_rows,
            "removed_rows": self._removed_rows,
            "full_row_uploads": self._full_row_uploads,
            "partial_row_uploads": self._partial_row_uploads,
            "live_rows": sum(row is not None for row in self._rows),
            "row_capacity": self._row_capacity,
            "page_capacity": self._page_capacity,
        }
        if reset:
            self._storage_allocations = 0
            self._storage_elements_copied = 0
            self._rows_uploaded = 0
            self._elements_uploaded = 0
            self._row_hits = 0
            self._released_rows = 0
            self._removed_rows = 0
            self._full_row_uploads = 0
            self._partial_row_uploads = 0
        return result


@dataclass
class _GraphPageTableRow:
    signature: PageTableRow
    input_width: int


def _graph_page_at(
    row: _GraphPageTableRow,
    column: int,
    scratch_page: int,
) -> int:
    if column >= row.input_width:
        return scratch_page
    return _page_at(row.signature, column)


def _graph_dirty_span(
    previous: _GraphPageTableRow | None,
    current: _GraphPageTableRow,
    *,
    width: int,
    scratch_page: int,
) -> tuple[int, int] | None:
    if width == 0:
        return None
    if previous is None or previous.signature.owner != current.signature.owner:
        return 0, width
    if (
        previous.input_width == current.input_width
        and previous.signature == current.signature
    ):
        return None
    first: int | None = None
    last = 0
    for column in range(width):
        if _graph_page_at(
            previous, column, scratch_page
        ) != _graph_page_at(current, column, scratch_page):
            first = column if first is None else first
            last = column + 1
    return None if first is None else (first, last)


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
    page_table_cache: DecodePageTableCache | None = None,
    row_owners: Sequence[DecodeRowOwner] | None = None,
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
    if page_table_cache is not None:
        if row_owners is None:
            raise ValueError("row_owners are required with a page_table_cache")
        if page_table_cache.device != torch.device(device):
            raise ValueError(
                "page-table cache device disagrees with decode batch device"
            )
        page_tables, page_table_rows = page_table_cache.materialize(
            page_lists,
            max_pages=max_pages,
            scratch_page=scratch_page,
            row_owners=row_owners,
        )
    else:
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
        if row_owners is None:
            page_table_rows = ()
        else:
            if len(row_owners) != batch:
                raise ValueError("row_owners must have one entry per decode row")
            page_table_rows = tuple(
                PageTableRow(
                    owner,
                    tuple(int(page) for page in pages[:max_pages]),
                    int(scratch_page) if scratch_page is not None else int(pages[-1]),
                )
                for owner, pages in zip(row_owners, page_lists, strict=True)
            )
    return DecodeBatch(
        token_ids=torch.as_tensor(token_ids, dtype=torch.int64, device=device),
        positions=torch.as_tensor(positions, dtype=torch.int64, device=device),
        page_tables=page_tables,
        seq_lens=torch.as_tensor(seq_lens, dtype=torch.int32, device=device),
        write_from=torch.as_tensor(write_from, dtype=torch.int64, device=device),
        page_table_rows=page_table_rows,
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
        self._captured_page_rows: dict[
            int, list[_GraphPageTableRow | None]
        ] = {}
        # A release forgets ownership immediately, but clearing the captured
        # tensor is deferred until the next ordered copy-in.  The tombstone is
        # necessary because ``None`` alone also means "already scratch": after
        # release the bytes still contain the old request's page IDs.
        self._pending_page_clears: dict[int, set[int]] = {}
        self._graph_executions = 0
        self._eager_fallbacks = 0
        self._page_table_graph_executions = 0
        self._page_table_eager_fallbacks = 0
        self._page_table_rows_copied = 0
        self._page_table_elements_copied = 0
        self._page_table_row_hits = 0
        self._page_table_released_rows = 0

    def can_execute_graph(self, batch: DecodeBatch) -> bool:
        """Whether ``batch`` fits one configured static graph bucket."""
        return (
            bucket_for(batch.batch_size, self._buckets) is not None
            and batch.max_pages <= self._max_pages
        )

    def execution_stats(self, *, reset: bool = False) -> dict[str, object]:
        """Structural dispatch counts, independent of wall-clock jitter."""
        result = {
            "graph_executions": self._graph_executions,
            "eager_fallbacks": self._eager_fallbacks,
            "captured_buckets": tuple(sorted(self._captured)),
        }
        if reset:
            self._graph_executions = 0
            self._eager_fallbacks = 0
        return result

    def page_table_execution_stats(
        self, *, reset: bool = False
    ) -> dict[str, int]:
        """Return only page-table copy counters.

        Keeping these counters separate preserves the established graph
        dispatch-stats schema while allowing matched cache ON/OFF probes to
        reset copy evidence without erasing verification dispatch evidence.
        """
        result = {
            "rows_copied": self._page_table_rows_copied,
            "elements_copied": self._page_table_elements_copied,
            "row_hits": self._page_table_row_hits,
            "released_rows": self._page_table_released_rows,
            "live_rows": sum(
                row is not None
                for rows in self._captured_page_rows.values()
                for row in rows
            ),
        }
        if reset:
            self._page_table_rows_copied = 0
            self._page_table_elements_copied = 0
            self._page_table_row_hits = 0
            self._page_table_released_rows = 0
        return result

    def page_table_dispatch_stats(
        self, *, reset: bool = False
    ) -> dict[str, object]:
        """Dispatch counters reset independently from verification stats."""
        result = {
            "graph_executions": self._page_table_graph_executions,
            "eager_fallbacks": self._page_table_eager_fallbacks,
            "captured_buckets": tuple(sorted(self._captured)),
        }
        if reset:
            self._page_table_graph_executions = 0
            self._page_table_eager_fallbacks = 0
        return result

    def execute_decode(self, batch: DecodeBatch) -> torch.Tensor:
        bucket = bucket_for(batch.batch_size, self._buckets)
        # oversize batch OR a page table wider than the captured static buffer:
        # never crash, run eager (D2)
        if bucket is None or batch.max_pages > self._max_pages:
            self._eager_fallbacks += 1
            self._page_table_eager_fallbacks += 1
            self._plan(batch)  # eager still needs a live plan for THESE buffers
            return self._decode_fn(batch)
        if bucket not in self._captured:
            self._capture(bucket)
        replayable, static = self._captured[bucket]
        self._copy_in(bucket, static, batch)
        # AFTER copy-in, BEFORE replay: the plan describes the page table and
        # lengths the kernels are about to read, and _copy_in just rewrote both
        # (including the padding rows). Planning before the copy would schedule
        # the PREVIOUS step. This ordering is the whole point of the hook.
        self._plan(static)
        out = replayable.replay()
        self._graph_executions += 1
        self._page_table_graph_executions += 1
        return out[: batch.batch_size]  # padding rows dropped

    def invalidate(self) -> None:
        """Weight swap / pool resize: every capture is stale."""
        for replayable, _static in self._captured.values():
            close = getattr(replayable, "close", None)
            if close is not None:
                close()
        self._captured.clear()
        self._captured_page_rows.clear()
        self._pending_page_clears.clear()
        invalidate_backend = getattr(self._backend, "invalidate", None)
        if invalidate_backend is not None:
            invalidate_backend()

    def release(self, request_id: str) -> None:
        """Forget every captured row owned by a completed request."""
        for bucket, rows in self._captured_page_rows.items():
            for index, row in enumerate(rows):
                if (
                    row is None
                    or row.signature.owner.request_id != request_id
                ):
                    continue
                rows[index] = None
                self._pending_page_clears[bucket].add(index)
                self._page_table_released_rows += 1

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
        self._captured_page_rows[bucket] = [None] * bucket
        self._pending_page_clears[bucket] = set()

    def _copy_in(
        self,
        bucket: int,
        static: DecodeBatch,
        batch: DecodeBatch,
    ) -> None:
        """Copy real rows into the static device buffers IN PLACE (C5); padding
        rows point at the scratch page with seq_len 1 (their outputs dropped)."""
        size = batch.batch_size
        static.token_ids[:size] = batch.token_ids
        static.token_ids[size:] = 0
        static.positions[:size] = batch.positions
        static.positions[size:] = 0
        page_rows = self._captured_page_rows[bucket]
        pending_clears = self._pending_page_clears[bucket]
        if len(batch.page_table_rows) == size:
            for row, signature in enumerate(batch.page_table_rows):
                current = _GraphPageTableRow(signature, batch.max_pages)
                dirty = _graph_dirty_span(
                    page_rows[row],
                    current,
                    width=self._max_pages,
                    scratch_page=self._scratch_page,
                )
                if dirty is None:
                    self._page_table_row_hits += 1
                else:
                    start, end = dirty
                    copied_end = min(end, batch.max_pages)
                    if start < copied_end:
                        static.page_tables[row, start:copied_end].copy_(
                            batch.page_tables[row, start:copied_end]
                        )
                    if end > batch.max_pages:
                        static.page_tables[
                            row, max(start, batch.max_pages) : end
                        ].fill_(self._scratch_page)
                    self._page_table_rows_copied += 1
                    self._page_table_elements_copied += end - start
                page_rows[row] = current
                pending_clears.discard(row)
            for row in range(size, bucket):
                if page_rows[row] is None and row not in pending_clears:
                    continue
                static.page_tables[row].fill_(self._scratch_page)
                page_rows[row] = None
                pending_clears.discard(row)
                self._page_table_rows_copied += 1
                self._page_table_elements_copied += self._max_pages
        else:
            # Compatibility batches have no trustworthy host signature.  Keep
            # the original unconditional copy rather than inferring identity
            # by synchronously reading a device tensor.
            static.page_tables[:size, : batch.max_pages] = batch.page_tables
            static.page_tables[:size, batch.max_pages :] = self._scratch_page
            static.page_tables[size:] = self._scratch_page
            self._page_table_rows_copied += bucket
            self._page_table_elements_copied += bucket * self._max_pages
            page_rows[:] = [None] * bucket
            pending_clears.clear()
        static.seq_lens[:size] = batch.seq_lens
        static.seq_lens[size:] = 1
        static.write_from[:size] = batch.write_from
        static.write_from[size:] = 0
