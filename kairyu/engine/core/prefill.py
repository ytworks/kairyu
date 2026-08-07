"""Ragged cross-request prefill input (issue #224).

The model sees one flat token tensor, while every row retains its own logical
sequence bounds and physical page table.  Writable KV slots are validated on
the host before the device kernels run: cached prefix pages may be shared, but
two requests may never write the same physical slot in one batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefillSequence:
    """One request's tail chunk before it is packed into a ragged batch."""

    request_id: str
    token_ids: tuple[int, ...]
    page_table: tuple[int, ...]
    chunk_start: int
    seq_len: int
    write_from: int


@dataclass(frozen=True)
class PrefillBatch:
    """One flat model invocation over independent variable-length sequences."""

    request_ids: tuple[str, ...]
    token_ids: torch.Tensor  # [sum(query_lens)] int64
    positions: torch.Tensor  # [sum(query_lens)] absolute positions
    row_ids: torch.Tensor  # [sum(query_lens)] owner row for every token
    page_tables: torch.Tensor  # [B, max_pages] int32
    seq_lens: tuple[int, ...]
    chunk_starts: tuple[int, ...]
    query_lens: tuple[int, ...]
    qo_indptr: tuple[int, ...]
    write_from: torch.Tensor  # [B] int64
    page_lists: tuple[tuple[int, ...], ...]

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def num_tokens(self) -> int:
        return int(self.token_ids.shape[0])

    def split(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split a flat token-major tensor back into request order."""
        if value.shape[0] != self.num_tokens:
            raise ValueError(
                "prefill result token count does not match the batch: "
                f"got {value.shape[0]}, expected {self.num_tokens}"
            )
        return tuple(torch.split(value, self.query_lens, dim=0))


def _metadata_views(
    buffer: torch.Tensor,
    *,
    num_tokens: int,
    num_rows: int,
    max_pages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split one byte buffer into the typed prefill metadata tensors."""

    cursor = 0

    def take(count: int, dtype: torch.dtype) -> torch.Tensor:
        nonlocal cursor
        byte_count = count * dtype.itemsize
        view = buffer[cursor : cursor + byte_count].view(dtype)
        cursor += byte_count
        return view

    token_ids = take(num_tokens, torch.int64)
    positions = take(num_tokens, torch.int64)
    row_ids = take(num_tokens, torch.int64)
    write_from = take(num_rows, torch.int64)
    page_tables = take(num_rows * max_pages, torch.int32).view(
        num_rows, max_pages
    )
    if cursor != buffer.numel():
        raise AssertionError("prefill metadata buffer layout is inconsistent")
    return token_ids, positions, row_ids, write_from, page_tables


def build_prefill_batch(
    sequences: Sequence[PrefillSequence],
    *,
    page_size: int,
    device: str | torch.device = "cpu",
) -> PrefillBatch:
    """Pack tail-aligned chunks and reject ambiguous KV ownership.

    Page-table tails repeat the final owned page only to make a rectangular
    device tensor.  They are unreachable because every token position and
    ``seq_len`` is validated against the row's real page count.
    """
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
    rows = tuple(sequences)
    if len(rows) < 1:
        raise ValueError("prefill batch must contain at least one sequence")

    request_ids = tuple(row.request_id for row in rows)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("prefill batch request_ids must be unique")

    query_lens: list[int] = []
    qo_indptr = [0]
    # A physical page may be shared only while every owner treats it as cached
    # read-only state. One map makes that page-level invariant O(total pages)
    # instead of walking prompt tokens or comparing every pair of rows.
    page_owners: dict[int, tuple[int, bool]] = {}
    for row_index, row in enumerate(rows):
        query_len = len(row.token_ids)
        if query_len < 1:
            raise ValueError(
                f"prefill row {row.request_id!r} must contain at least one token"
            )
        if row.chunk_start < 0 or row.seq_len < 1:
            raise ValueError(
                f"prefill row {row.request_id!r} has invalid bounds "
                f"[{row.chunk_start}, {row.seq_len})"
            )
        if row.chunk_start + query_len != row.seq_len:
            raise ValueError(
                f"prefill row {row.request_id!r} is not the sequence tail: "
                f"chunk_start={row.chunk_start}, query_len={query_len}, "
                f"seq_len={row.seq_len}"
            )
        if not 0 <= row.write_from <= row.seq_len:
            raise ValueError(
                f"prefill row {row.request_id!r} has write_from={row.write_from} "
                f"outside [0, {row.seq_len}]"
            )
        required_pages = -(-row.seq_len // page_size)
        if len(row.page_table) < required_pages:
            raise ValueError(
                f"prefill row {row.request_id!r} needs {required_pages} pages "
                f"for seq_len={row.seq_len}, got {len(row.page_table)}"
            )
        if any(page < 0 for page in row.page_table):
            raise ValueError(
                f"prefill row {row.request_id!r} has a negative page id"
            )
        used_pages = row.page_table[:required_pages]
        if len(set(used_pages)) != required_pages:
            raise ValueError(
                f"prefill row {row.request_id!r} aliases one physical page "
                "from multiple logical page positions"
            )
        first_writable = max(row.chunk_start, row.write_from)
        first_writable_page = (
            first_writable // page_size
            if first_writable < row.seq_len
            else required_pages
        )
        for logical_page, page in enumerate(used_pages):
            writable = logical_page >= first_writable_page
            owner = page_owners.get(page)
            if owner is not None and (writable or owner[1]):
                other = rows[owner[0]]
                raise ValueError(
                    "prefill rows would cross KV ownership: writable pages "
                    f"[{page}] overlap between {other.request_id!r} and "
                    f"{row.request_id!r}"
                )
            if owner is None:
                page_owners[page] = (row_index, writable)
        query_lens.append(query_len)
        qo_indptr.append(qo_indptr[-1] + query_len)

    selected = torch.device(device)
    max_pages = max(len(row.page_table) for row in rows)
    num_tokens = qo_indptr[-1]
    total_bytes = (
        (3 * num_tokens + len(rows)) * torch.int64.itemsize
        + len(rows) * max_pages * torch.int32.itemsize
    )
    host_buffer = torch.empty(
        total_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=selected.type == "cuda",
    )
    (
        host_token_ids,
        host_positions,
        host_row_ids,
        host_write_from,
        host_page_tables,
    ) = _metadata_views(
        host_buffer,
        num_tokens=num_tokens,
        num_rows=len(rows),
        max_pages=max_pages,
    )
    for index, row in enumerate(rows):
        start, end = qo_indptr[index], qo_indptr[index + 1]
        host_token_ids[start:end].copy_(
            torch.as_tensor(row.token_ids, dtype=torch.int64)
        )
        torch.arange(
            row.chunk_start,
            row.seq_len,
            dtype=torch.int64,
            out=host_positions[start:end],
        )
        host_row_ids[start:end].fill_(index)
        host_write_from[index] = row.write_from
        host_page_tables[index].fill_(row.page_table[-1])
        host_page_tables[index, : len(row.page_table)].copy_(
            torch.as_tensor(row.page_table, dtype=torch.int32)
        )

    device_buffer = host_buffer.to(
        device=selected,
        non_blocking=selected.type == "cuda",
    )
    token_ids, positions, row_ids, write_from, page_tables = _metadata_views(
        device_buffer,
        num_tokens=num_tokens,
        num_rows=len(rows),
        max_pages=max_pages,
    )
    return PrefillBatch(
        request_ids=request_ids,
        token_ids=token_ids,
        positions=positions,
        row_ids=row_ids,
        page_tables=page_tables,
        seq_lens=tuple(row.seq_len for row in rows),
        chunk_starts=tuple(row.chunk_start for row in rows),
        query_lens=tuple(query_lens),
        qo_indptr=tuple(qo_indptr),
        write_from=write_from,
        page_lists=tuple(row.page_table for row in rows),
    )
