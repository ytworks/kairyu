"""A decode path with no host values, so a CUDA graph can capture it (m17 D1).

`forward_decode_batch` takes Python page lists and int lengths. A captured graph
replays recorded kernels over recorded memory, so anything read out of a Python
object is frozen at capture time — every replay would attend over the
capture-time pages. That is why `GraphStepExecutor` had a seam but no caller.

These pin that the tensor path is numerically the SAME computation, so switching
to it is a capture enabler and not a behaviour change.
"""


import pytest
import torch

transformers = pytest.importorskip("transformers")

TINY = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def model_and_config(tmp_path_factory):
    from kairyu.models.loader import load_model

    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("tensor-decode")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    model, config, _ = load_model(str(path))
    return model, config


def _pool(config, *, seed: int = 5):
    from kairyu.engine.core.kv_pool import PagedKVPool

    torch.manual_seed(seed)
    pool = PagedKVPool(
        num_layers=config.num_hidden_layers, num_pages=16, page_size=4,
        num_kv_heads=config.kv_cache_num_heads, head_dim=config.kv_cache_head_dim,
    )
    with torch.no_grad():
        pool.k.copy_(torch.randn_like(pool.k) * 0.1)
        pool.v.copy_(torch.randn_like(pool.v) * 0.1)
    return pool


@pytest.mark.parametrize(
    ("tables", "positions", "seq_lens"),
    [
        ([[0, 1, 2], [4, 5, 6]], [6, 10], [7, 11]),
        ([[0, 1], [2, 3]], [4, 5], [5, 6]),  # ragged: different lengths, same pages
        ([[7, 8, 9]], [3], [4]),  # batch of one
    ],
    ids=["two-rows", "ragged", "single"],
)
def test_tensor_decode_matches_list_decode(model_and_config, tables, positions, seq_lens):
    model, config = model_and_config
    listed, tensored = _pool(config), _pool(config)

    tokens = torch.tensor([5, 9][: len(tables)])
    position_tensor = torch.tensor(positions)

    reference = model.forward_decode_batch(
        tokens, position_tensor, listed, tables, seq_lens, [0] * len(tables)
    )
    actual = model.forward_decode_tensors(
        tokens,
        position_tensor,
        tensored,
        torch.tensor(tables, dtype=torch.int32),
        torch.tensor(seq_lens, dtype=torch.int32),
    )

    assert torch.allclose(reference, actual, atol=1e-5)
    # the KV write must land in the same slots, or the NEXT step diverges even
    # though this one matched
    assert torch.allclose(listed.k, tensored.k, atol=1e-6)
    assert torch.allclose(listed.v, tensored.v, atol=1e-6)


def test_the_tensor_path_reads_no_python_values(model_and_config):
    """Rebinding the page-table TENSOR in place changes the result; that is the
    property a graph needs. A Python list would have been frozen at capture."""
    model, config = model_and_config
    pool = _pool(config)

    tokens = torch.tensor([5])
    positions = torch.tensor([6])
    seq_lens = torch.tensor([7], dtype=torch.int32)
    tables = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    first = model.forward_decode_tensors(tokens, positions, pool, tables, seq_lens)
    tables.copy_(torch.tensor([[4, 5, 6]], dtype=torch.int32))  # IN PLACE
    second = model.forward_decode_tensors(tokens, positions, pool, tables, seq_lens)

    assert not torch.allclose(first, second), "the page table was not actually read"


def test_the_gather_is_shape_stable_across_lengths(model_and_config):
    """Slicing to `seq_len` would make the shape depend on a host value, which a
    graph cannot re-read; the tail is masked instead."""
    _model, config = model_and_config
    pool = _pool(config)
    tables = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int32)

    keys, values = pool.gather_batched(0, tables)
    assert keys.shape == (2, 3 * pool.page_size, pool.num_kv_heads, pool.head_dim)
    assert values.shape[:2] == keys.shape[:2]


def test_write_batched_lands_where_write_does(model_and_config):
    _model, config = model_and_config
    listed, tensored = _pool(config), _pool(config)

    keys = torch.randn(2, config.kv_cache_num_heads, config.kv_cache_head_dim)
    values = torch.randn_like(keys)
    positions = torch.tensor([6, 10])
    tables = [[0, 1, 2], [4, 5, 6]]

    for row, table in enumerate(tables):
        listed.write(0, table, positions[row : row + 1], keys[row : row + 1], values[row : row + 1])
    tensored.write_batched(
        0, torch.tensor(tables, dtype=torch.int32), positions, keys, values
    )

    assert torch.equal(listed.k, tensored.k)
    assert torch.equal(listed.v, tensored.v)
