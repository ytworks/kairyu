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

    @triton.jit(
        do_not_specialize=(
            "table_stride_0",
            "NUM_ROWS",
            "NUM_TABLE_PAGES",
            "k_scale",
            "v_scale",
        )
    )
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
        k_scale,
        v_scale,
        NUM_ROWS,
        NUM_TABLE_PAGES,
        NUM_POOL_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_WRITE_FROM: tl.constexpr,
        FP8_DEST: tl.constexpr,
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
        if FP8_DEST:
            key = tl.clamp(key / k_scale, -448.0, 448.0)
            value = tl.clamp(value / v_scale, -448.0, 448.0)
        destination = (
            (page * PAGE_SIZE + slot) * (NUM_HEADS * HEAD_DIM) + offsets
        )
        tl.store(k_pool + destination, key, mask=source_mask)
        tl.store(v_pool + destination, value, mask=source_mask)

    @triton.jit(
        do_not_specialize=(
            "table_stride_0",
            "NUM_TOKENS",
            "NUM_ROWS",
            "NUM_TABLE_PAGES",
            "k_scale",
            "v_scale",
        )
    )
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
        k_scale,
        v_scale,
        NUM_TOKENS,
        NUM_ROWS,
        NUM_TABLE_PAGES,
        NUM_POOL_PAGES: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        FP8_DEST: tl.constexpr,
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
        if FP8_DEST:
            key = tl.clamp(key / k_scale, -448.0, 448.0)
            value = tl.clamp(value / v_scale, -448.0, 448.0)
        destination = (
            (page * PAGE_SIZE + slot) * (NUM_HEADS * HEAD_DIM) + offsets
        )
        tl.store(k_pool + destination, key, mask=source_mask)
        tl.store(v_pool + destination, value, mask=source_mask)


_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)
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
        or values.dtype != keys.dtype
    ):
        return False
    source_matches_pool = keys.dtype == k_pool.dtype
    source_can_fuse_fp8 = (
        k_pool.dtype == torch.float8_e4m3fn
        and keys.dtype in (torch.float16, torch.bfloat16)
    )
    if not source_matches_pool and not source_can_fuse_fp8:
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
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> bool:
    """Write a dense decode batch and report whether the fused path ran."""

    if not _supported_payload(k_pool, v_pool, keys, values):
        return False
    # Runtime calibrated scales stay on the torch oracle path: Triton's FP8
    # cast is not byte-identical for non-unit scales on the reviewed SM120.
    if k_pool.dtype == torch.float8_e4m3fn and (
        k_scale != 1.0 or v_scale != 1.0
    ):
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
    # Request-dependent batch and page-table bounds must remain runtime kernel
    # arguments.  ``table_stride_0`` is the live page width for contiguous
    # tables, so specializing it would still recompile on width changes even
    # when ``NUM_TABLE_PAGES`` itself is runtime-valued.
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
        k_scale,
        v_scale,
        NUM_ROWS=rows,
        NUM_TABLE_PAGES=page_tables.shape[1],
        NUM_POOL_PAGES=k_pool.shape[0],
        PAGE_SIZE=page_size,
        NUM_HEADS=keys.shape[1],
        HEAD_DIM=keys.shape[2],
        HAS_WRITE_FROM=write_from is not None,
        FP8_DEST=k_pool.dtype == torch.float8_e4m3fn,
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
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> bool:
    """Write a ragged prefill batch and report whether the fused path ran."""

    if not _supported_payload(k_pool, v_pool, keys, values):
        return False
    # Keep this fused FP8 writer unit-scale-only; see the batched guard above.
    if k_pool.dtype == torch.float8_e4m3fn and (
        k_scale != 1.0 or v_scale != 1.0
    ):
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
    # Request-dependent ragged bounds must remain runtime kernel arguments.
    # Marking them constexpr makes every unseen token/row/page-table shape pay
    # a separate Triton compilation pause on the serving path.
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
        k_scale,
        v_scale,
        NUM_TOKENS=tokens,
        NUM_ROWS=rows,
        NUM_TABLE_PAGES=page_tables.shape[1],
        NUM_POOL_PAGES=k_pool.shape[0],
        PAGE_SIZE=page_size,
        NUM_HEADS=keys.shape[1],
        HEAD_DIM=keys.shape[2],
        FP8_DEST=k_pool.dtype == torch.float8_e4m3fn,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return True
