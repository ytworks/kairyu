"""FlashInfer honours the tensor decode contract too (m13 seam, m17 D1).

#137 added `attend_decode` for the torch backend so decode could be captured.
FlashInfer is the kernel a GPU deployment actually selects, so the contract has
to hold there or the capturable path is a fallback nobody runs.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("FlashInfer gates need CUDA")


def _pool(*, seed: int = 3):
    from kairyu.engine.core.kv_pool import PagedKVPool

    torch.manual_seed(seed)
    pool = PagedKVPool(
        num_layers=1, num_pages=8, page_size=16, num_kv_heads=2, head_dim=64,
        dtype=torch.bfloat16, device="cuda:0",
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
    query = torch.randn(rows, 4, 64, device="cuda:0", dtype=torch.bfloat16)
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
    query = torch.randn(1, 4, 64, device="cuda:0", dtype=torch.bfloat16)
    tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor([20], dtype=torch.int32, device="cuda:0")

    first = backend.attend_decode(query, pool, 0, tables, lengths)
    tables.copy_(torch.tensor([[4, 5]], dtype=torch.int32, device="cuda:0"))
    second = backend.attend_decode(query, pool, 0, tables, lengths)

    assert not torch.allclose(first.float(), second.float())
