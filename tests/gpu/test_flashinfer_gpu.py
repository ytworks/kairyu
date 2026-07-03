"""Deploy-day mirror of the fake-flashinfer contract suite (m13 D4).

Runs real FlashInfer kernels with TorchAttentionBackend as the oracle.
Executed via `pytest -m gpu` / scripts/gpu_gates on CUDA hardware.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

PAGE = 16


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
        pytest.skip("CUDA required")
    return "cuda"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seq_len,chunk_len", [(10, 10), (37, 5), (PAGE * 3, 1)]  # prefill/chunk/decode
)
def test_flashinfer_matches_torch_backend(cuda, dtype, seq_len, chunk_len):
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.kv_pool import PagedKVPool

    torch.manual_seed(0)
    heads, kv_heads, head_dim = 8, 2, 64
    pool = PagedKVPool(1, 64, PAGE, kv_heads, head_dim, dtype=dtype, device=cuda)
    keys = torch.randn(seq_len, kv_heads, head_dim, dtype=dtype, device=cuda)
    values = torch.randn(seq_len, kv_heads, head_dim, dtype=dtype, device=cuda)
    page_table = list(range(-(-seq_len // PAGE)))
    pool.write(0, page_table, torch.arange(seq_len, device=cuda), keys, values)
    chunk_start = seq_len - chunk_len
    query = torch.randn(chunk_len, heads, head_dim, dtype=dtype, device=cuda)

    reference = TorchAttentionBackend().attend(
        query, pool, 0, page_table, seq_len, chunk_start
    )
    out = FlashInferBackend(device=cuda).attend(
        query, pool, 0, page_table, seq_len, chunk_start
    )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
    assert (out - reference).abs().max().item() < tolerance
