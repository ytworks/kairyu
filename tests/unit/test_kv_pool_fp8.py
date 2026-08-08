"""Focused CPU contract tests for unit-scale FP8 paged KV storage."""

import pytest
import torch

from kairyu.engine.core.kv_pool import PagedKVPool


def _pool(
    *,
    dtype: torch.dtype = torch.float8_e4m3fn,
    k_scales=None,
    v_scales=None,
) -> PagedKVPool:
    return PagedKVPool(
        num_layers=2,
        num_pages=4,
        page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=dtype,
        k_scales=k_scales,
        v_scales=v_scales,
    )


def test_fp8_pool_exposes_unit_scales_per_layer():
    pool = _pool()

    assert [pool.k_scale(layer) for layer in range(2)] == [1.0, 1.0]
    assert [pool.v_scale(layer) for layer in range(2)] == [1.0, 1.0]
    assert _pool(dtype=torch.bfloat16).k_scale(0) is None
    assert _pool(dtype=torch.bfloat16).v_scale(0) is None
    with pytest.raises(IndexError, match="outside"):
        pool.k_scale(2)
    with pytest.raises(IndexError, match="outside"):
        pool.v_scale(-1)


def test_fp8_pool_validates_and_exposes_calibrated_layer_scales():
    pool = _pool(k_scales=[0.25, 0.5], v_scales=[0.125, 0.75])

    assert [pool.k_scale(layer) for layer in range(2)] == [0.25, 0.5]
    assert [pool.v_scale(layer) for layer in range(2)] == [0.125, 0.75]
    with pytest.raises(ValueError, match="both K and V"):
        _pool(k_scales=[1.0, 1.0])
    with pytest.raises(ValueError, match="contain 2 layers"):
        _pool(k_scales=[1.0], v_scales=[1.0])
    with pytest.raises(ValueError, match="finite and positive"):
        _pool(k_scales=[1.0, 0.0], v_scales=[1.0, 1.0])
    with pytest.raises(ValueError, match="require an FP8 cache"):
        _pool(dtype=torch.bfloat16, k_scales=[1.0, 1.0], v_scales=[1.0, 1.0])


def test_fp8_direct_write_applies_independent_kv_layer_scales():
    pool = _pool(k_scales=[1.0, 0.5], v_scales=[1.0, 0.25])
    keys = torch.tensor([[[0.5, -1.0]]], dtype=torch.bfloat16)
    values = torch.tensor([[[0.25, -0.5]]], dtype=torch.bfloat16)

    pool.write(1, [0], torch.tensor([0]), keys, values)

    assert torch.equal(pool.k[1, 0, 0], (keys / 0.5).to(torch.float8_e4m3fn)[0])
    assert torch.equal(pool.v[1, 0, 0], (values / 0.25).to(torch.float8_e4m3fn)[0])


def test_fp8_direct_write_explicitly_quantizes_sources():
    pool = _pool()
    keys = torch.tensor(
        [[[0.1, -0.2]], [[3.3, -4.7]]],
        dtype=torch.bfloat16,
    )
    values = torch.tensor(
        [[[-1.1, 2.2]], [[0.4, 0.8]]],
        dtype=torch.bfloat16,
    )

    pool.write(
        1,
        [2, 3],
        torch.tensor([0, 3]),
        keys,
        values,
    )

    assert torch.equal(pool.k[1, 2, 0], keys[0].to(torch.float8_e4m3fn))
    assert torch.equal(pool.k[1, 3, 1], keys[1].to(torch.float8_e4m3fn))
    assert torch.equal(pool.v[1, 2, 0], values[0].to(torch.float8_e4m3fn))
    assert torch.equal(pool.v[1, 3, 1], values[1].to(torch.float8_e4m3fn))


def test_fp8_write_uses_satfinite_e4m3_conversion():
    pool = PagedKVPool(
        num_layers=1,
        num_pages=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=6,
        dtype=torch.float8_e4m3fn,
    )
    source = torch.tensor(
        [-1000.0, -449.0, -448.0, 448.0, 449.0, 1000.0]
    ).reshape(1, 1, 6)

    pool.write(0, [0], torch.tensor([0]), source, -source)

    stored_k = pool.k[0, 0, 0, 0]
    stored_v = pool.v[0, 0, 0, 0]
    assert stored_k.view(torch.uint8).tolist() == [254, 254, 254, 126, 126, 126]
    assert stored_v.view(torch.uint8).tolist() == [126, 126, 126, 254, 254, 254]
    assert torch.isfinite(stored_k.float()).all()
    assert torch.isfinite(stored_v.float()).all()


def test_fp8_batched_write_quantizes_and_preserves_cached_rows():
    pool = _pool()
    page_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    positions = torch.tensor([1, 3], dtype=torch.int64)
    keys = torch.tensor(
        [[[0.3, -0.6]], [[7.1, -8.2]]],
        dtype=torch.bfloat16,
    )
    values = torch.tensor(
        [[[-0.9, 1.4]], [[2.5, 3.5]]],
        dtype=torch.bfloat16,
    )

    pool.write_batched(
        0,
        page_tables,
        positions,
        keys,
        values,
        write_from=torch.tensor([0, 4]),
    )

    assert torch.equal(pool.k[0, 0, 1], keys[0].to(torch.float8_e4m3fn))
    assert torch.equal(pool.v[0, 0, 1], values[0].to(torch.float8_e4m3fn))
    assert torch.equal(pool.k[0, 3, 1], torch.zeros_like(pool.k[0, 3, 1]))
    assert torch.equal(pool.v[0, 3, 1], torch.zeros_like(pool.v[0, 3, 1]))


def test_fp8_ragged_write_quantizes_only_writable_slots():
    pool = _pool()
    page_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    row_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    positions = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
    keys = torch.tensor(
        [[[0.1, 0.2]], [[1.3, -1.7]], [[2.1, 2.9]], [[3.2, -3.8]]],
        dtype=torch.bfloat16,
    )
    values = -keys

    pool.write_ragged(
        0,
        page_tables,
        row_ids,
        positions,
        keys,
        values,
        write_from=torch.tensor([2, 3]),
    )

    assert torch.equal(pool.k[0, 1, 0], keys[1].to(torch.float8_e4m3fn))
    assert torch.equal(pool.v[0, 1, 0], values[1].to(torch.float8_e4m3fn))
    assert torch.equal(pool.k[0, 3, 1], keys[3].to(torch.float8_e4m3fn))
    assert torch.equal(pool.v[0, 3, 1], values[3].to(torch.float8_e4m3fn))
    assert torch.equal(pool.k[0, 0, 0], torch.zeros_like(pool.k[0, 0, 0]))
    assert torch.equal(pool.k[0, 2, 1], torch.zeros_like(pool.k[0, 2, 1]))
