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
    used_pages_by_row: list[set[int]] = []
    writable_pages_by_row: list[set[int]] = []
    writable_slots: dict[tuple[int, int], str] = {}
    for row in rows:
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
        used_pages = set(row.page_table[:required_pages])
        if len(used_pages) != required_pages:
            raise ValueError(
                f"prefill row {row.request_id!r} aliases one physical page "
                "from multiple logical page positions"
            )
        first_writable = max(row.chunk_start, row.write_from)
        writable_pages = {
            row.page_table[position // page_size]
            for position in range(first_writable, row.seq_len)
        }
        for other_index, (other_used, other_writable) in enumerate(
            zip(used_pages_by_row, writable_pages_by_row, strict=True)
        ):
            if writable_pages & other_used or other_writable & used_pages:
                other = rows[other_index]
                overlap = sorted(
                    (writable_pages & other_used) | (other_writable & used_pages)
                )
                raise ValueError(
                    "prefill rows would cross KV ownership: writable pages "
                    f"{overlap} overlap between {other.request_id!r} and "
                    f"{row.request_id!r}"
                )
        used_pages_by_row.append(used_pages)
        writable_pages_by_row.append(writable_pages)

        for position in range(first_writable, row.seq_len):
            page = row.page_table[position // page_size]
            slot = (page, position % page_size)
            owner = writable_slots.get(slot)
            if owner is not None:
                raise ValueError(
                    "prefill rows would cross KV ownership: physical slot "
                    f"{slot} is writable by both {owner!r} and {row.request_id!r}"
                )
            writable_slots[slot] = row.request_id
        query_lens.append(query_len)
        qo_indptr.append(qo_indptr[-1] + query_len)

    selected = torch.device(device)
    max_pages = max(len(row.page_table) for row in rows)
    page_tables = torch.empty(
        (len(rows), max_pages), dtype=torch.int32, device=selected
    )
    for index, row in enumerate(rows):
        page_tables[index].fill_(row.page_table[-1])
        page_tables[index, : len(row.page_table)] = torch.tensor(
            row.page_table, dtype=torch.int32, device=selected
        )

    token_ids = torch.tensor(
        [token for row in rows for token in row.token_ids],
        dtype=torch.int64,
        device=selected,
    )
    positions = torch.cat(
        [
            torch.arange(
                row.chunk_start,
                row.seq_len,
                dtype=torch.int64,
                device=selected,
            )
            for row in rows
        ]
    )
    row_ids = torch.repeat_interleave(
        torch.arange(len(rows), dtype=torch.int64, device=selected),
        torch.tensor(query_lens, dtype=torch.int64, device=selected),
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
        write_from=torch.tensor(
            [row.write_from for row in rows],
            dtype=torch.int64,
            device=selected,
        ),
        page_lists=tuple(row.page_table for row in rows),
    )
