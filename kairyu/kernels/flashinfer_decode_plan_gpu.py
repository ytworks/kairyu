"""Device-only metadata packing for FlashInfer CUDA-graph decode replay.

FlashInfer consumes CSR page indices, while Kairyu's captured input is a fixed
rectangular page table.  Stock ``wrapper.plan`` converts that table with
boolean indexing and then reads indptr/last-page lengths back to the host.
Replay already has authoritative scheduler lengths on the CPU, so this one
kernel only refreshes the wrapper's persistent device buffers on the current
stream; the CPU schedule is built independently from those host lengths.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - GPU extra is required for the fast path
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _pack_decode_metadata_kernel(
        page_tables,
        seq_lens,
        indptr,
        indices,
        last_page_len,
        table_stride_0,
        table_stride_1,
        seq_stride,
        NUM_ROWS: tl.constexpr,
        NUM_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_PAGES: tl.constexpr,
    ):
        row = tl.program_id(0)
        prior_rows = tl.arange(0, BLOCK_ROWS)
        lengths = tl.load(
            seq_lens + prior_rows * seq_stride,
            mask=prior_rows < NUM_ROWS,
            other=1,
        ).to(tl.int32)
        page_counts = (lengths + PAGE_SIZE - 1) // PAGE_SIZE
        # The graph executor guarantees 1 <= seq_len <= P * page_size from the
        # same host tuple used by the CPU plan.  Clamp here as a final memory
        # safety boundary without synchronously validating a device scalar.
        page_counts = tl.maximum(1, tl.minimum(page_counts, NUM_PAGES))
        offset = tl.sum(
            tl.where(prior_rows < row, page_counts, 0),
            axis=0,
        )
        row_length = tl.load(seq_lens + row * seq_stride).to(tl.int32)
        row_pages = (row_length + PAGE_SIZE - 1) // PAGE_SIZE
        row_pages = tl.maximum(1, tl.minimum(row_pages, NUM_PAGES))

        tl.store(indptr + row, offset)
        if row == NUM_ROWS - 1:
            tl.store(indptr + NUM_ROWS, offset + row_pages)
        tl.store(last_page_len + row, (row_length - 1) % PAGE_SIZE + 1)

        columns = tl.arange(0, BLOCK_PAGES)
        mask = columns < row_pages
        pages = tl.load(
            page_tables
            + row * table_stride_0
            + columns * table_stride_1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        tl.store(indices + offset + columns, pages, mask=mask)


def pack_flashinfer_decode_metadata(
    page_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    *,
    page_size: int,
) -> None:
    """Pack rectangular decode metadata into persistent FlashInfer buffers.

    This function intentionally has no PyTorch fallback: using ordinary
    boolean indexing would reintroduce ``nonzero`` and its device-to-host
    synchronization.  The caller instead falls back truthfully to stock
    FlashInfer planning when Triton or a compatible CUDA layout is absent.
    """

    if triton is None:
        raise RuntimeError("Triton is unavailable for FlashInfer fast replay planning")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if page_tables.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("page_tables must be [B, P] and seq_lens must be [B]")
    rows, pages = page_tables.shape
    if rows < 1 or pages < 1 or seq_lens.shape[0] != rows:
        raise ValueError("decode metadata must contain matching non-empty rows")
    device = page_tables.device
    tensors = (page_tables, seq_lens, indptr, indices, last_page_len)
    if device.type != "cuda" or any(tensor.device != device for tensor in tensors):
        raise ValueError("FlashInfer fast replay metadata must share one CUDA device")
    if page_tables.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError("page_tables and seq_lens must use torch.int32")
    if any(
        tensor.dtype != torch.int32
        for tensor in (indptr, indices, last_page_len)
    ):
        raise ValueError("FlashInfer persistent metadata buffers must use torch.int32")
    if indptr.ndim != 1 or indptr.shape[0] < rows + 1:
        raise ValueError("FlashInfer indptr buffer is too small")
    if indices.ndim != 1 or indices.shape[0] < rows * pages:
        raise ValueError("FlashInfer indices buffer is too small")
    if last_page_len.ndim != 1 or last_page_len.shape[0] < rows:
        raise ValueError("FlashInfer last-page-length buffer is too small")

    _pack_decode_metadata_kernel[(rows,)](
        page_tables,
        seq_lens,
        indptr,
        indices,
        last_page_len,
        page_tables.stride(0),
        page_tables.stride(1),
        seq_lens.stride(0),
        NUM_ROWS=rows,
        NUM_PAGES=pages,
        PAGE_SIZE=page_size,
        BLOCK_ROWS=triton.next_power_of_2(rows),
        BLOCK_PAGES=triton.next_power_of_2(pages),
        num_warps=4,
    )
