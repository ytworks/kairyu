"""FlashInfer honours the tensor decode contract too (m13 seam, m17 D1).

#137 added `attend_decode` for the torch backend so decode could be captured.
FlashInfer is the kernel a GPU deployment actually selects, so the contract has
to hold there or the capturable path is a fallback nobody runs.

FlashInfer splits the work in two: `plan()` builds the split-KV schedule on the
CPU and "cannot be used in Cuda Graph" (its own docstring), while `run()` reads
the plan out of device buffers. So the gates below are the two halves — eager
parity for `plan_decode` + `attend_decode` together, and a REAL captured graph
whose replay picks up an in-place page-table change once the step is re-planned.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

PAGE_SIZE = 16
HEAD_DIM = 64
KV_HEADS = 2
QO_HEADS = 4


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("FlashInfer gates need CUDA")


def _pool(*, seed: int = 3):
    from kairyu.engine.core.kv_pool import PagedKVPool

    torch.manual_seed(seed)
    pool = PagedKVPool(
        num_layers=1, num_pages=8, page_size=PAGE_SIZE, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, dtype=torch.bfloat16, device="cuda:0",
    )
    with torch.no_grad():
        pool.k.copy_(torch.randn(pool.k.shape, device="cuda:0").to(torch.bfloat16) * 0.1)
        pool.v.copy_(torch.randn(pool.v.shape, device="cuda:0").to(torch.bfloat16) * 0.1)
    return pool


@pytest.mark.parametrize(
    ("tables", "seq_lens"),
    [
        ([[0, 1], [2, 3]], [20, 30]),
        ([[0, 1], [2, 3]], [16, 32]),  # exact page multiples
        ([[4, 5]], [17]),  # single row, one token into the second page
    ],
    ids=["ragged", "aligned", "single"],
)
def test_flashinfer_matches_torch_on_the_tensor_contract(tables, seq_lens):
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    _require_cuda()
    pool = _pool()
    rows = len(tables)
    query = torch.randn(rows, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    page_tables = torch.tensor(tables, dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device="cuda:0")

    reference = TorchAttentionBackend().attend_decode(query, pool, 0, page_tables, lengths)
    actual = FlashInferBackend().attend_decode(query, pool, 0, page_tables, lengths)

    assert actual.shape == reference.shape
    assert torch.allclose(actual.float(), reference.float(), atol=5e-2)


def test_the_paged_arrays_come_from_the_tensors_not_a_python_list():
    """A page table mutated IN PLACE must change the answer — the property the
    list-based `attend_batched` cannot have."""
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

    _require_cuda()
    pool = _pool()
    backend = FlashInferBackend()
    query = torch.randn(1, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor([20], dtype=torch.int32, device="cuda:0")

    first = backend.attend_decode(query, pool, 0, tables, lengths)
    tables.copy_(torch.tensor([[4, 5]], dtype=torch.int32, device="cuda:0"))
    second = backend.attend_decode(query, pool, 0, tables, lengths)

    assert not torch.allclose(first.float(), second.float())


def test_a_captured_graph_replays_against_the_current_pages():
    """The gate the eager tests structurally cannot give: a REAL CUDA graph.

    Capture fails outright on any `.tolist()`/`.cpu()` inside the region
    ("Cannot copy between CPU and CUDA tensors during CUDA graph capture"). It
    succeeding is half the property; the other half is that after the static
    page-table/seq-len buffers are rewritten in place and the step is
    re-planned on the host, `replay()` attends over the NEW pages — not the
    ones that happened to be there at capture time.
    """
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

    _require_cuda()
    pool = _pool()
    backend = FlashInferBackend()
    query = torch.randn(2, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    # STATIC buffers: the graph is captured against these exact tensors
    tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor([20, 30], dtype=torch.int32, device="cuda:0")

    def decode(_batch=None):
        return backend.attend_decode(query, pool, 0, tables, lengths)

    backend.plan_decode(
        pool, tables, lengths, num_qo_heads=QO_HEADS, q_dtype=query.dtype
    )
    replayable = CudaGraphBackend(warmup_iters=2).capture(decode, None)

    captured = replayable.replay().clone()
    torch.cuda.synchronize()
    assert torch.allclose(
        captured.float(),
        TorchAttentionBackend().attend_decode(query, pool, 0, tables, lengths).float(),
        atol=5e-2,
    )

    tables.copy_(torch.tensor([[4, 5], [6, 7]], dtype=torch.int32, device="cuda:0"))
    lengths.copy_(torch.tensor([16, 17], dtype=torch.int32, device="cuda:0"))
    backend.plan_decode(  # the host half of the step, OUTSIDE the graph
        pool, tables, lengths, num_qo_heads=QO_HEADS, q_dtype=query.dtype
    )
    replayed = replayable.replay()
    torch.cuda.synchronize()

    expected = TorchAttentionBackend().attend_decode(query, pool, 0, tables, lengths)
    assert torch.allclose(replayed.float(), expected.float(), atol=5e-2)
    assert not torch.allclose(replayed.float(), captured.float())


def test_planning_inside_a_capture_fails_instead_of_corrupting_the_graph():
    """FlashInfer's plan cannot be captured; the adapter says so itself rather
    than letting the capture die inside the kernel library."""
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

    _require_cuda()
    pool = _pool()
    backend = FlashInferBackend()
    query = torch.randn(1, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor([20], dtype=torch.int32, device="cuda:0")

    def decode(_batch=None):
        return backend.attend_decode(query, pool, 0, tables, lengths)

    decode()  # warm the kernels up eagerly (this plans)
    backend._decode_tensor_key = None  # ...then let the step's plan go missing

    with pytest.raises(RuntimeError, match="plan_decode"):
        CudaGraphBackend(warmup_iters=0).capture(decode, None)
