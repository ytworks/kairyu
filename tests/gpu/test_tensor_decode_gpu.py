"""The tensor decode path on real devices, bf16 (m17 D1).

The CPU gate covers the math; this covers the dtype and device a capture would
actually run under, and that the pool write lands in the same slots there too.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("tensor decode gate needs CUDA")


@pytest.fixture(scope="module")
def model_and_config(tmp_path_factory):
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.models.loader import load_model

    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("gpu-tensor-decode")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    # the torch backend on purpose: this gate is about the MODEL-level tensor
    # path, so it uses the backend that needs no plan step; FlashInfer's own
    # half of the contract is gated in test_flashinfer_tensor_decode.py
    model, config, _ = load_model(
        str(path), dtype=torch.bfloat16, attention_backend=TorchAttentionBackend()
    )
    return model.to("cuda:0"), config


def _pool(config, *, seed: int = 5):
    from kairyu.engine.core.kv_pool import PagedKVPool

    torch.manual_seed(seed)
    pool = PagedKVPool(
        num_layers=config.num_hidden_layers, num_pages=16, page_size=4,
        num_kv_heads=config.kv_cache_num_heads, head_dim=config.kv_cache_head_dim,
        dtype=torch.bfloat16, device="cuda:0",
    )
    with torch.no_grad():
        pool.k.copy_(torch.randn(pool.k.shape, device="cuda:0").to(torch.bfloat16) * 0.1)
        pool.v.copy_(torch.randn(pool.v.shape, device="cuda:0").to(torch.bfloat16) * 0.1)
    return pool


def test_tensor_decode_matches_list_decode_on_gpu(model_and_config):
    model, config = model_and_config
    _require_cuda()
    listed, tensored = _pool(config), _pool(config)

    tables = [[0, 1, 2], [4, 5, 6]]
    tokens = torch.tensor([5, 9], device="cuda:0")
    positions = torch.tensor([6, 10], device="cuda:0")
    seq_lens = [7, 11]

    reference = model.forward_decode_batch(
        tokens, positions, listed, tables, seq_lens, [0, 0]
    )
    actual = model.forward_decode_tensors(
        tokens,
        positions,
        tensored,
        torch.tensor(tables, dtype=torch.int32, device="cuda:0"),
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda:0"),
    )

    assert actual.device.type == "cuda"
    assert torch.allclose(reference.float(), actual.float(), atol=2e-2)
    assert torch.equal(listed.k, tensored.k), "the KV write landed elsewhere"
    assert torch.equal(listed.v, tensored.v)
