"""CUDA gates for the fused dense paged-KV write path."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():  # pragma: no cover - CPU CI
        pytest.skip("fused paged-KV writes need CUDA")
    return torch.device("cuda:0")


def _pool(*, pages: int = 8, page_size: int = 4):
    from kairyu.engine.core.kv_pool import PagedKVPool

    return PagedKVPool(
        num_layers=1,
        num_pages=pages,
        page_size=page_size,
        num_kv_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda:0",
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


def test_batched_write_handles_packed_v_and_cached_rows(monkeypatch):
    from kairyu.kernels import paged_kv_write_gpu

    device = _require_cuda()
    torch.manual_seed(211)
    pool = _pool()
    pool.k.normal_()
    pool.v.normal_()
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
    fused_write = paged_kv_write_gpu.try_write_batched

    def spy_fused_write(*args, **kwargs):
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
    assert torch.equal(pool.k, expected_k)
    assert torch.equal(pool.v, expected_v)


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


def test_ragged_write_preserves_shared_pages_and_writes_private_slots(
    monkeypatch,
):
    from kairyu.kernels import paged_kv_write_gpu

    device = _require_cuda()
    torch.manual_seed(227)
    pool = _pool()
    pool.k.normal_()
    pool.v.normal_()
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
    fused_write = paged_kv_write_gpu.try_write_ragged

    def spy_fused_write(*args, **kwargs):
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
    assert torch.equal(pool.k, expected_k)
    assert torch.equal(pool.v, expected_v)
    assert torch.equal(pool.k[:, 0], shared_k)
    assert torch.equal(pool.v[:, 0], shared_v)


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
    }
