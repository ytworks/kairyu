"""Real SM120 FlashAttention-4 adapter proof against the torch oracle."""

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_flashattention4_sm120_prefill_and_delegated_decode_match_torch():
    if not torch.cuda.is_available():  # pragma: no cover - GPU gate only
        pytest.skip("CUDA required")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm != 120:  # pragma: no cover - this gate targets the retained profile
        pytest.skip(f"SM120 required, found SM{sm}")

    from kairyu.engine.core.attention.flashattention_gpu import (
        FlashAttentionBackend,
    )
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.hw_profile import HardwareProfile
    from kairyu.engine.core.kv_pool import PagedKVPool

    torch.manual_seed(277)
    device = "cuda"
    dtype = torch.bfloat16
    page_size = 16
    seq_len = 273
    chunk_len = 64
    heads, kv_heads, head_dim = 8, 1, 128  # Qwen3-32B TP8 local shape
    page_table = [
        29,
        3,
        41,
        7,
        17,
        2,
        37,
        11,
        23,
        5,
        31,
        13,
        43,
        19,
        47,
        0,
        53,
        59,
    ]
    pool = PagedKVPool(
        1,
        64,
        page_size,
        kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    positions = torch.arange(seq_len, device=device)
    keys = torch.randn(
        seq_len, kv_heads, head_dim, dtype=dtype, device=device
    )
    values = torch.randn(
        seq_len, kv_heads, head_dim, dtype=dtype, device=device
    )
    pool.write(0, page_table, positions, keys, values)

    backend = FlashAttentionBackend(
        generation=4,
        profile=HardwareProfile(arch="cuda", sm=120),
        device=device,
    )
    oracle = TorchAttentionBackend()
    assert backend.components == {
        "prefill": "flashattention4",
        "decode": "flashinfer",
        "kv_mode": "paged-materialized",
    }

    prefill_query = torch.randn(
        chunk_len, heads, head_dim, dtype=dtype, device=device
    )
    chunk_start = seq_len - chunk_len
    expected_prefill = oracle.attend(
        prefill_query, pool, 0, page_table, seq_len, chunk_start
    )
    actual_prefill = backend.attend(
        prefill_query, pool, 0, page_table, seq_len, chunk_start
    )
    torch.testing.assert_close(
        actual_prefill, expected_prefill, rtol=0.02, atol=0.02
    )

    decode_query = torch.randn(
        1, heads, head_dim, dtype=dtype, device=device
    )
    expected_decode = oracle.attend(
        decode_query, pool, 0, page_table, seq_len, seq_len - 1
    )
    actual_decode = backend.attend(
        decode_query, pool, 0, page_table, seq_len, seq_len - 1
    )
    torch.testing.assert_close(
        actual_decode, expected_decode, rtol=0.02, atol=0.02
    )
