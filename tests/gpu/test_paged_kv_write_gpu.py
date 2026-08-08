"""CUDA gates for the fused dense paged-KV write path."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():  # pragma: no cover - CPU CI
        pytest.skip("fused paged-KV writes need CUDA")
    return torch.device("cuda:0")


def _pool(
    *,
    pages: int = 8,
    page_size: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    k_scales=None,
    v_scales=None,
):
    from kairyu.engine.core.kv_pool import PagedKVPool

    return PagedKVPool(
        num_layers=1,
        num_pages=pages,
        page_size=page_size,
        num_kv_heads=2,
        head_dim=128,
        dtype=dtype,
        device="cuda:0",
        k_scales=k_scales,
        v_scales=v_scales,
    )


def _randomize_pool(pool) -> None:
    pool.k.copy_(
        torch.randn(pool.k.shape, device=pool.k.device, dtype=torch.bfloat16)
    )
    pool.v.copy_(
        torch.randn(pool.v.shape, device=pool.v.device, dtype=torch.bfloat16)
    )


def _packed_values(
    rows: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    # Models split V out of a packed QKV projection.  Its visible [H,D]
    # payload is dense, but the next token starts after the complete QKV row.
    storage = torch.randn(
        rows,
        2 * 128 + 37,
        device=device,
        dtype=torch.bfloat16,
    )
    values = storage[:, : 2 * 128].view(rows, 2, 128)
    assert not values.is_contiguous()
    assert values.stride()[1:] == (128, 1)
    return values


def _reference_write(
    expected_k: torch.Tensor,
    expected_v: torch.Tensor,
    *,
    page_size: int,
    page_tables: torch.Tensor,
    row_ids: torch.Tensor,
    positions: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    write_from: torch.Tensor,
) -> None:
    for token, (owner, position) in enumerate(
        zip(row_ids.tolist(), positions.tolist(), strict=True)
    ):
        if position < int(write_from[owner]):
            continue
        page = int(page_tables[owner, position // page_size])
        slot = position % page_size
        expected_k[0, page, slot].copy_(keys[token])
        expected_v[0, page, slot].copy_(values[token])


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8-e4m3"],
)
def test_batched_write_handles_packed_v_and_cached_rows(monkeypatch, dtype):
    from kairyu.kernels import paged_kv_write_gpu

    device = _require_cuda()
    torch.manual_seed(211)
    pool = _pool(dtype=dtype)
    _randomize_pool(pool)
    expected_k = pool.k.clone()
    expected_v = pool.v.clone()

    page_tables = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.tensor([5, 2, 6, 0], dtype=torch.int64, device=device)
    write_from = torch.tensor([0, 3, 6, 1], dtype=torch.int32, device=device)
    keys = torch.randn(4, 2, 128, device=device, dtype=torch.bfloat16)
    values = _packed_values(4, device=device)
    fused_results = []
    fused_source_dtypes = []
    fused_write = paged_kv_write_gpu.try_write_batched

    def spy_fused_write(*args, **kwargs):
        fused_source_dtypes.append((args[5].dtype, args[6].dtype))
        result = fused_write(*args, **kwargs)
        fused_results.append(result)
        return result

    monkeypatch.setattr(
        paged_kv_write_gpu, "try_write_batched", spy_fused_write
    )

    _reference_write(
        expected_k,
        expected_v,
        page_size=pool.page_size,
        page_tables=page_tables,
        row_ids=torch.arange(4, device=device),
        positions=positions,
        keys=keys,
        values=values,
        write_from=write_from,
    )
    pool.write_batched(
        0,
        page_tables,
        positions,
        keys,
        values,
        write_from,
    )
    torch.cuda.synchronize()

    assert fused_results == [True]
    assert fused_source_dtypes == [(torch.bfloat16, torch.bfloat16)]
    assert torch.equal(pool.k, expected_k)
    assert torch.equal(pool.v, expected_v)


def test_fp8_batched_write_uses_satfinite_bytes(monkeypatch):
    from kairyu.kernels import paged_kv_write_gpu

    device = _require_cuda()
    pool = _pool(dtype=torch.float8_e4m3fn)
    page_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    positions = torch.tensor([0], dtype=torch.int64, device=device)
    keys = torch.zeros(1, 2, 128, dtype=torch.float16, device=device)
    keys[0, 0, :4] = torch.tensor(
        [-1000.0, -449.0, 449.0, 1000.0],
        device=device,
    )
    values = -keys
    fused_results = []
    fused_source_dtypes = []
    fused_write = paged_kv_write_gpu.try_write_batched

    def spy_fused_write(*args, **kwargs):
        fused_source_dtypes.append((args[5].dtype, args[6].dtype))
        result = fused_write(*args, **kwargs)
        fused_results.append(result)
        return result

    monkeypatch.setattr(
        paged_kv_write_gpu, "try_write_batched", spy_fused_write
    )

    pool.write_batched(
        0,
        page_tables,
        positions,
        keys,
        values,
    )
    torch.cuda.synchronize()

    stored_k = pool.k[0, 0, 0, 0, :4]
    stored_v = pool.v[0, 0, 0, 0, :4]
    assert fused_results == [True]
    assert fused_source_dtypes == [(torch.float16, torch.float16)]
    assert stored_k.view(torch.uint8).tolist() == [254, 254, 126, 126]
    assert stored_v.view(torch.uint8).tolist() == [126, 126, 254, 254]
    assert torch.isfinite(stored_k.float()).all()
    assert torch.isfinite(stored_v.float()).all()


def test_fp8_fused_writes_apply_independent_calibrated_scales():
    device = _require_cuda()
    k_scale = 0.27773437500000003
    v_scale = 0.0013641357421875
    pool = _pool(
        dtype=torch.float8_e4m3fn,
        k_scales=[k_scale],
        v_scales=[v_scale],
    )
    page_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    positions = torch.tensor([0], dtype=torch.int64, device=device)
    torch.manual_seed(357)
    keys = torch.randn(1, 2, 128, dtype=torch.bfloat16, device=device) * 30
    values = torch.randn_like(keys) * 0.1

    def quantized(tensor, scale):
        return (tensor.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    pool.write_batched(0, page_tables, positions, keys, values)
    torch.cuda.synchronize()
    assert torch.equal(
        pool.k[0, 0, 0], quantized(keys, k_scale)[0]
    )
    assert torch.equal(
        pool.v[0, 0, 0], quantized(values, v_scale)[0]
    )

    positions.fill_(1)
    row_ids = torch.tensor([0], dtype=torch.int32, device=device)
    write_from = torch.tensor([1], dtype=torch.int32, device=device)
    pool.write_ragged(
        0,
        page_tables,
        row_ids,
        positions,
        keys * 2,
        values * 2,
        write_from,
    )
    torch.cuda.synchronize()
    assert torch.equal(
        pool.k[0, 0, 1], quantized(keys * 2, k_scale)[0]
    )
    assert torch.equal(
        pool.v[0, 0, 1], quantized(values * 2, v_scale)[0]
    )


def test_batched_write_cuda_graph_replay_reads_current_buffers(monkeypatch):
    from kairyu.kernels import paged_kv_write_gpu

    device = _require_cuda()
    torch.manual_seed(223)
    pool = _pool()
    page_tables = torch.tensor(
        [[0, 1], [2, 3]], dtype=torch.int32, device=device
    )
    positions = torch.tensor([0, 0], dtype=torch.int64, device=device)
    write_from = torch.zeros(2, dtype=torch.int64, device=device)
    keys = torch.randn(2, 2, 128, device=device, dtype=torch.bfloat16)
    values = _packed_values(2, device=device)
    fused_results = []
    fused_write = paged_kv_write_gpu.try_write_batched

    def spy_fused_write(*args, **kwargs):
        result = fused_write(*args, **kwargs)
        fused_results.append(result)
        return result

    monkeypatch.setattr(
        paged_kv_write_gpu, "try_write_batched", spy_fused_write
    )

    # Compile before capture, as the production graph backend's warmups do.
    pool.write_batched(
        0, page_tables, positions, keys, values, write_from
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.write_batched(
            0, page_tables, positions, keys, values, write_from
        )
    assert fused_results == [True, True]

    pool.k.zero_()
    pool.v.zero_()
    page_tables.copy_(
        torch.tensor([[4, 5], [6, 7]], dtype=torch.int32, device=device)
    )
    positions.copy_(torch.tensor([5, 6], dtype=torch.int64, device=device))
    write_from.copy_(torch.tensor([0, 7], dtype=torch.int64, device=device))
    keys.copy_(
        torch.randn(2, 2, 128, device=device, dtype=torch.bfloat16)
    )
    values.copy_(_packed_values(2, device=device))
    expected_k = pool.k.clone()
    expected_v = pool.v.clone()
    _reference_write(
        expected_k,
        expected_v,
        page_size=pool.page_size,
        page_tables=page_tables,
        row_ids=torch.arange(2, device=device),
        positions=positions,
        keys=keys,
        values=values,
        write_from=write_from,
    )

    graph.replay()
    torch.cuda.synchronize()
    # Replay runs the captured Triton node, not Python or the torch fallback.
    assert fused_results == [True, True]
    assert torch.equal(pool.k, expected_k)
    assert torch.equal(pool.v, expected_v)


def test_batched_request_shape_bounds_are_runtime_kernel_arguments():
    from kairyu.kernels import paged_kv_write_gpu

    _require_cuda()
    parameters = {
        parameter.name: parameter
        for parameter in paged_kv_write_gpu._write_batched_kernel.params
    }

    for name in ("NUM_ROWS", "NUM_TABLE_PAGES"):
        assert parameters[name].is_constexpr is False
    for name in (
        "NUM_POOL_PAGES",
        "PAGE_SIZE",
        "NUM_HEADS",
        "HEAD_DIM",
        "HAS_WRITE_FROM",
        "FP8_DEST",
        "BLOCK",
    ):
        assert parameters[name].is_constexpr is True
    assert {
        parameters[index].name
        for index in paged_kv_write_gpu._write_batched_kernel.do_not_specialize
    } == {
        "table_stride_0",
        "NUM_ROWS",
        "NUM_TABLE_PAGES",
        "k_scale",
        "v_scale",
    }


def test_batched_request_shapes_share_one_compiled_kernel(monkeypatch):
    from triton import knobs

    from kairyu.kernels import paged_kv_write_gpu

    device = _require_cuda()
    compiled = []

    def record_compile(**kwargs):
        if (
            kwargs["fn"].jit_function
            is paged_kv_write_gpu._write_batched_kernel
        ):
            compiled.append(kwargs["repr"])

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", record_compile)
    paged_kv_write_gpu._write_batched_kernel.device_caches.clear()

    for rows, table_pages, positions_list in (
        (2, 1, [0, 1]),
        (5, 3, [0, 4, 8, 5, 2]),
    ):
        pool = _pool(pages=32)
        _randomize_pool(pool)
        expected_k = pool.k.clone()
        expected_v = pool.v.clone()
        page_tables = torch.arange(
            rows * table_pages,
            dtype=torch.int32,
            device=device,
        ).view(rows, table_pages)
        positions = torch.tensor(
            positions_list, dtype=torch.int64, device=device
        )
        write_from = torch.zeros(rows, dtype=torch.int32, device=device)
        keys = torch.randn(
            rows, 2, 128, device=device, dtype=torch.bfloat16
        )
        values = _packed_values(rows, device=device)
        _reference_write(
            expected_k,
            expected_v,
            page_size=pool.page_size,
            page_tables=page_tables,
            row_ids=torch.arange(rows, device=device),
            positions=positions,
            keys=keys,
            values=values,
            write_from=write_from,
        )

        assert paged_kv_write_gpu.try_write_batched(
            pool.k[0],
            pool.v[0],
            pool.page_size,
            page_tables,
            positions,
            keys,
            values,
            write_from,
        )
        torch.cuda.synchronize()
        assert torch.equal(pool.k, expected_k)
        assert torch.equal(pool.v, expected_v)
        assert len(compiled) == 1


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float8_e4m3fn],
    ids=["bf16", "fp8-e4m3"],
)
def test_ragged_write_preserves_shared_pages_and_writes_private_slots(
    monkeypatch, dtype
):
    from kairyu.kernels import paged_kv_write_gpu

    device = _require_cuda()
    torch.manual_seed(227)
    pool = _pool(dtype=dtype)
    _randomize_pool(pool)
    expected_k = pool.k.clone()
    expected_v = pool.v.clone()
    shared_k = pool.k[:, 0].clone()
    shared_v = pool.v[:, 0].clone()

    # Rows 0 and 1 retain shared page 0.  Their writable second pages differ.
    page_tables = torch.tensor(
        [[0, 1], [0, 2], [3, 4]], dtype=torch.int32, device=device
    )
    row_ids = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int32, device=device
    )
    positions = torch.tensor(
        [0, 1, 4, 2, 3, 4, 0, 5], dtype=torch.int64, device=device
    )
    write_from = torch.tensor([4, 4, 2], dtype=torch.int32, device=device)
    keys = torch.randn(8, 2, 128, device=device, dtype=torch.bfloat16)
    values = _packed_values(8, device=device)
    fused_results = []
    fused_source_dtypes = []
    fused_write = paged_kv_write_gpu.try_write_ragged

    def spy_fused_write(*args, **kwargs):
        fused_source_dtypes.append((args[6].dtype, args[7].dtype))
        result = fused_write(*args, **kwargs)
        fused_results.append(result)
        return result

    monkeypatch.setattr(
        paged_kv_write_gpu, "try_write_ragged", spy_fused_write
    )
    _reference_write(
        expected_k,
        expected_v,
        page_size=pool.page_size,
        page_tables=page_tables,
        row_ids=row_ids,
        positions=positions,
        keys=keys,
        values=values,
        write_from=write_from,
    )

    pool.write_ragged(
        0,
        page_tables,
        row_ids,
        positions,
        keys,
        values,
        write_from,
    )
    torch.cuda.synchronize()

    assert fused_results == [True]
    assert fused_source_dtypes == [(torch.bfloat16, torch.bfloat16)]
    assert torch.equal(pool.k, expected_k)
    assert torch.equal(pool.v, expected_v)
    assert torch.equal(pool.k[:, 0], shared_k)
    assert torch.equal(pool.v[:, 0], shared_v)


def test_fp8_ragged_write_uses_satfinite_bytes(monkeypatch):
    from kairyu.kernels import paged_kv_write_gpu

    device = _require_cuda()
    pool = _pool(dtype=torch.float8_e4m3fn)
    page_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    row_ids = torch.tensor([0], dtype=torch.int32, device=device)
    positions = torch.tensor([0], dtype=torch.int64, device=device)
    write_from = torch.tensor([0], dtype=torch.int32, device=device)
    keys = torch.zeros(1, 2, 128, dtype=torch.bfloat16, device=device)
    keys[0, 0, :4] = torch.tensor(
        [-1000.0, -449.0, 449.0, 1000.0],
        dtype=torch.bfloat16,
        device=device,
    )
    values = -keys
    fused_results = []
    fused_source_dtypes = []
    fused_write = paged_kv_write_gpu.try_write_ragged

    def spy_fused_write(*args, **kwargs):
        fused_source_dtypes.append((args[6].dtype, args[7].dtype))
        result = fused_write(*args, **kwargs)
        fused_results.append(result)
        return result

    monkeypatch.setattr(
        paged_kv_write_gpu, "try_write_ragged", spy_fused_write
    )

    pool.write_ragged(
        0,
        page_tables,
        row_ids,
        positions,
        keys,
        values,
        write_from,
    )
    torch.cuda.synchronize()

    stored_k = pool.k[0, 0, 0, 0, :4]
    stored_v = pool.v[0, 0, 0, 0, :4]
    assert fused_results == [True]
    assert fused_source_dtypes == [(torch.bfloat16, torch.bfloat16)]
    assert stored_k.view(torch.uint8).tolist() == [254, 254, 126, 126]
    assert stored_v.view(torch.uint8).tolist() == [126, 126, 254, 254]
    assert torch.isfinite(stored_k.float()).all()
    assert torch.isfinite(stored_v.float()).all()


def test_ragged_request_shape_bounds_are_runtime_kernel_arguments():
    from kairyu.kernels import paged_kv_write_gpu

    _require_cuda()
    parameters = {
        parameter.name: parameter
        for parameter in paged_kv_write_gpu._write_ragged_kernel.params
    }

    for name in ("NUM_TOKENS", "NUM_ROWS", "NUM_TABLE_PAGES"):
        assert parameters[name].is_constexpr is False
    for name in (
        "NUM_POOL_PAGES",
        "PAGE_SIZE",
        "NUM_HEADS",
        "HEAD_DIM",
        "BLOCK",
    ):
        assert parameters[name].is_constexpr is True
    assert {
        parameters[index].name
        for index in paged_kv_write_gpu._write_ragged_kernel.do_not_specialize
    } == {
        "table_stride_0",
        "NUM_TOKENS",
        "NUM_ROWS",
        "NUM_TABLE_PAGES",
        "k_scale",
        "v_scale",
    }
