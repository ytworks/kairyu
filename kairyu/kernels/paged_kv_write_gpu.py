"""One-launch CUDA writes into the paged K/V cache.

The generic PyTorch implementation is the portability/reference path.  These
Triton kernels cover the dense GQA layout used by CUDA serving, including a
packed (non-contiguous across tokens) V projection.  Cached/shared slots are
left untouched by masking the stores rather than reading and writing them
back.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the caller retains the torch fallback
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _write_batched_kernel(
        k_pool,
        v_pool,
        page_tables,
        positions,
        keys,
        values,
        write_from,
        table_stride_0,
        table_stride_1,
        position_stride,
        key_stride_0,
        key_stride_1,
        key_stride_2,
        value_stride_0,
        value_stride_1,
        value_stride_2,
        write_from_stride,
        NUM_ROWS: tl.constexpr,
        NUM_TABLE_PAGES: tl.constexpr,
        NUM_POOL_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_WRITE_FROM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        position = tl.load(positions + row * position_stride)
        page_index = position // PAGE_SIZE
        slot = position - page_index * PAGE_SIZE
        valid = (
            (row < NUM_ROWS)
            & (position >= 0)
            & (page_index >= 0)
            & (page_index < NUM_TABLE_PAGES)
        )
        page = tl.load(
            page_tables + row * table_stride_0 + page_index * table_stride_1,
            mask=valid,
            other=-1,
        )
        valid &= (page >= 0) & (page < NUM_POOL_PAGES)
        if HAS_WRITE_FROM:
            valid &= position >= tl.load(
                write_from + row * write_from_stride,
                mask=row < NUM_ROWS,
                other=position + 1,
            )

        offsets = tl.arange(0, BLOCK)
        payload = offsets < NUM_HEADS * HEAD_DIM
        head = offsets // HEAD_DIM
        column = offsets - head * HEAD_DIM
        source_mask = valid & payload
        key = tl.load(
            keys
            + row * key_stride_0
            + head * key_stride_1
            + column * key_stride_2,
            mask=source_mask,
            other=0.0,
        )
        value = tl.load(
            values
            + row * value_stride_0
            + head * value_stride_1
            + column * value_stride_2,
            mask=source_mask,
            other=0.0,
        )
        destination = (
            (page * PAGE_SIZE + slot) * (NUM_HEADS * HEAD_DIM) + offsets
        )
        tl.store(k_pool + destination, key, mask=source_mask)
        tl.store(v_pool + destination, value, mask=source_mask)

    @triton.jit
    def _write_ragged_kernel(
        k_pool,
        v_pool,
        page_tables,
        row_ids,
        positions,
        keys,
        values,
        write_from,
        table_stride_0,
        table_stride_1,
        row_id_stride,
        position_stride,
        key_stride_0,
        key_stride_1,
        key_stride_2,
        value_stride_0,
        value_stride_1,
        value_stride_2,
        write_from_stride,
        NUM_TOKENS: tl.constexpr,
        NUM_ROWS: tl.constexpr,
        NUM_TABLE_PAGES: tl.constexpr,
        NUM_POOL_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        owner = tl.load(row_ids + token * row_id_stride)
        position = tl.load(positions + token * position_stride)
        page_index = position // PAGE_SIZE
        slot = position - page_index * PAGE_SIZE
        valid = (
            (token < NUM_TOKENS)
            & (owner >= 0)
            & (owner < NUM_ROWS)
            & (position >= 0)
            & (page_index >= 0)
            & (page_index < NUM_TABLE_PAGES)
        )
        page = tl.load(
            page_tables + owner * table_stride_0 + page_index * table_stride_1,
            mask=valid,
            other=-1,
        )
        retain_from = tl.load(
            write_from + owner * write_from_stride,
            mask=valid,
            other=position + 1,
        )
        valid &= (
            (page >= 0)
            & (page < NUM_POOL_PAGES)
            & (position >= retain_from)
        )

        offsets = tl.arange(0, BLOCK)
        payload = offsets < NUM_HEADS * HEAD_DIM
        head = offsets // HEAD_DIM
        column = offsets - head * HEAD_DIM
        source_mask = valid & payload
        key = tl.load(
            keys
            + token * key_stride_0
            + head * key_stride_1
            + column * key_stride_2,
            mask=source_mask,
            other=0.0,
        )
        value = tl.load(
            values
            + token * value_stride_0
            + head * value_stride_1
            + column * value_stride_2,
            mask=source_mask,
            other=0.0,
        )
        destination = (
            (page * PAGE_SIZE + slot) * (NUM_HEADS * HEAD_DIM) + offsets
        )
        tl.store(k_pool + destination, key, mask=source_mask)
        tl.store(v_pool + destination, value, mask=source_mask)


_FLOAT_DTYPES = (torch.float16, torch.bfloat16)
_INDEX_DTYPES = (torch.int32, torch.int64)


def _supported_payload(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> bool:
    if triton is None or k_pool.device.type != "cuda":
        return False
    if (
        k_pool.ndim != 4
        or v_pool.ndim != 4
        or keys.ndim != 3
        or values.ndim != 3
    ):
        return False
    if k_pool.shape != v_pool.shape or keys.shape != values.shape:
        return False
    if keys.shape[1:] != k_pool.shape[2:]:
        return False
    if (
        k_pool.dtype not in _FLOAT_DTYPES
        or v_pool.dtype != k_pool.dtype
        or keys.dtype != k_pool.dtype
        or values.dtype != k_pool.dtype
    ):
        return False
    if any(
        tensor.device != k_pool.device
        for tensor in (v_pool, keys, values)
    ):
        return False
    if not k_pool.is_contiguous() or not v_pool.is_contiguous():
        return False
    head_dim = keys.shape[2]
    elements = keys.shape[1] * head_dim
    if not (0 < elements <= 4096):
        return False
    # A packed projection can make stride(0) wider than the visible tensor.
    # Each token's [head, dim] payload must still be densely addressable.
    return (
        keys.stride(2) == 1
        and keys.stride(1) == head_dim
        and values.stride(2) == 1
        and values.stride(1) == head_dim
    )


def _supported_index(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    ndim: int,
) -> bool:
    return (
        tensor.device == device
        and tensor.dtype in _INDEX_DTYPES
        and tensor.ndim == ndim
    )


def try_write_batched(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_size: int,
    page_tables: torch.Tensor,
    positions: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    write_from: torch.Tensor | None,
) -> bool:
    """Write a dense decode batch and report whether the fused path ran."""

    if not _supported_payload(k_pool, v_pool, keys, values):
        return False
    device = k_pool.device
    rows = keys.shape[0]
    if (
        not _supported_index(page_tables, device=device, ndim=2)
        or not _supported_index(positions, device=device, ndim=1)
        or page_tables.shape[0] != rows
        or positions.shape[0] != rows
        or (
            write_from is not None
            and (
                not _supported_index(write_from, device=device, ndim=1)
                or write_from.shape[0] != rows
            )
        )
    ):
        return False
    if rows == 0:
        return True

    elements = keys.shape[1] * keys.shape[2]
    block = triton.next_power_of_2(elements)
    dummy_write_from = positions if write_from is None else write_from
    _write_batched_kernel[(rows,)](
        k_pool,
        v_pool,
        page_tables,
        positions,
        keys,
        values,
        dummy_write_from,
        page_tables.stride(0),
        page_tables.stride(1),
        positions.stride(0),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        dummy_write_from.stride(0),
        NUM_ROWS=rows,
        NUM_TABLE_PAGES=page_tables.shape[1],
        NUM_POOL_PAGES=k_pool.shape[0],
        PAGE_SIZE=page_size,
        NUM_HEADS=keys.shape[1],
        HEAD_DIM=keys.shape[2],
        HAS_WRITE_FROM=write_from is not None,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return True


def try_write_ragged(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_size: int,
    page_tables: torch.Tensor,
    row_ids: torch.Tensor,
    positions: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    write_from: torch.Tensor,
) -> bool:
    """Write a ragged prefill batch and report whether the fused path ran."""

    if not _supported_payload(k_pool, v_pool, keys, values):
        return False
    device = k_pool.device
    tokens = keys.shape[0]
    rows = page_tables.shape[0] if page_tables.ndim == 2 else -1
    if (
        not _supported_index(page_tables, device=device, ndim=2)
        or not _supported_index(row_ids, device=device, ndim=1)
        or not _supported_index(positions, device=device, ndim=1)
        or not _supported_index(write_from, device=device, ndim=1)
        or row_ids.shape[0] != tokens
        or positions.shape[0] != tokens
        or write_from.shape[0] != rows
    ):
        return False
    if tokens == 0:
        return True

    elements = keys.shape[1] * keys.shape[2]
    block = triton.next_power_of_2(elements)
    _write_ragged_kernel[(tokens,)](
        k_pool,
        v_pool,
        page_tables,
        row_ids,
        positions,
        keys,
        values,
        write_from,
        page_tables.stride(0),
        page_tables.stride(1),
        row_ids.stride(0),
        positions.stride(0),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        write_from.stride(0),
        NUM_TOKENS=tokens,
        NUM_ROWS=rows,
        NUM_TABLE_PAGES=page_tables.shape[1],
        NUM_POOL_PAGES=k_pool.shape[0],
        PAGE_SIZE=page_size,
        NUM_HEADS=keys.shape[1],
        HEAD_DIM=keys.shape[2],
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return True
