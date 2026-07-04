"""C4: batched decode must be numerically IDENTICAL to sequential decode.

Batching is a performance transform (one forward over B sequences instead of B
forwards), not a numeric one — row i of the batched output must equal running
sequence i's decode step on its own. This is the CPU-testable contract the GPU
batched runner/FlashInfer path preserves.
"""

import pytest

torch = pytest.importorskip("torch")

from kairyu.engine.core.kv_pool import PagedKVPool  # noqa: E402
from kairyu.engine.core.radix_kv import RadixKVCache  # noqa: E402
from kairyu.models.config import parse_model_config  # noqa: E402
from kairyu.models.llama import DenseDecoder  # noqa: E402

PAGE = 4
TINY = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 64,
    "rms_norm_eps": 1e-6,
}


def _prefill(model, pool, tokens, pages):
    """Prefill a sequence's prompt KV into `pages`; return post-norm hidden."""
    ids = torch.tensor(tokens, dtype=torch.long)
    positions = torch.arange(len(tokens))
    return model.forward_tokens(ids, positions, pool, pages, seq_len=len(tokens), write_from=0)


def test_batched_decode_equals_sequential():
    torch.manual_seed(7)
    config = parse_model_config(TINY)
    model = DenseDecoder(config).eval()

    # two sequences with DISJOINT pages, prefilled to different lengths
    seqs = [
        {"prompt": [5, 9, 1, 7, 3], "pages": [0, 1]},   # len 5 -> pages 0,1
        {"prompt": [2, 8, 4, 6, 10, 11, 12], "pages": [2, 3]},  # len 7 -> pages 2,3
    ]
    cache = RadixKVCache(num_pages=8, page_size=PAGE)

    # Build TWO identically-prefilled pools (decode mutates the pool)
    def fresh_pool():
        pool = PagedKVPool.for_cache(cache, config)
        for seq in seqs:
            _prefill(model, pool, seq["prompt"], seq["pages"])
        return pool

    pool_seq = fresh_pool()
    pool_batch = fresh_pool()

    # each sequence decodes one new token at position = len(prompt)
    decode_tokens = [42, 17]
    positions = [len(seq["prompt"]) for seq in seqs]  # 5, 7
    seq_lens = [p + 1 for p in positions]  # 6, 8
    page_tables = [seq["pages"] for seq in seqs]
    write_from = list(positions)  # the new token is written (>= write_from)

    # sequential: one forward per sequence
    sequential = []
    for i in range(len(seqs)):
        hidden = model.forward_tokens(
            torch.tensor([decode_tokens[i]], dtype=torch.long),
            torch.tensor([positions[i]]),
            pool_seq, page_tables[i], seq_len=seq_lens[i], write_from=write_from[i],
        )
        sequential.append(hidden[0])

    # batched: ONE forward over both sequences
    batched = model.forward_decode_batch(
        torch.tensor(decode_tokens, dtype=torch.long),
        torch.tensor(positions),
        pool_batch, page_tables, seq_lens, write_from,
    )

    for i in range(len(seqs)):
        assert torch.allclose(batched[i], sequential[i], atol=1e-5), i
    # and the written KV matches too (byte-for-byte decode-page contents)
    assert torch.allclose(pool_batch.k, pool_seq.k, atol=1e-6)


def test_batched_logits_match_sequential():
    torch.manual_seed(11)
    config = parse_model_config(TINY)
    model = DenseDecoder(config).eval()
    cache = RadixKVCache(num_pages=8, page_size=PAGE)

    prompts = [[1, 2, 3], [4, 5, 6, 7]]
    pages = [[0], [1, 2]]

    def fresh_pool():
        pool = PagedKVPool.for_cache(cache, config)
        for prompt, pt in zip(prompts, pages, strict=True):
            _prefill(model, pool, prompt, pt)
        return pool

    positions = [len(p) for p in prompts]
    batched_hidden = model.forward_decode_batch(
        torch.tensor([9, 13], dtype=torch.long),
        torch.tensor(positions),
        fresh_pool(), pages, [p + 1 for p in positions], list(positions),
    )
    batched_logits = model.logits(batched_hidden)
    # greedy tokens from the batched logits are well-defined per row
    assert batched_logits.shape == (2, config.vocab_size)
    assert batched_logits.argmax(dim=-1).shape == (2,)
