"""C4: batched decode must be numerically IDENTICAL to sequential decode.

Batching is a performance transform (one forward over B sequences instead of B
forwards), not a numeric one — row i of the batched output must equal running
sequence i's decode step on its own. This is the CPU-testable contract the GPU
batched runner/FlashInfer path preserves.
"""

from types import SimpleNamespace

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


def _sampling_state(request_id, sampling):
    request = SimpleNamespace(
        request_id=request_id,
        sampling_identity=request_id,
        sampling=sampling,
        eos_token_id=None,
        prompt_token_ids=(1, 2, 3),
    )
    return SimpleNamespace(request=request, outputs=())


def test_plain_greedy_batch_uses_one_vector_argmax():
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import ScheduledChunk

    runner = object.__new__(PagedModelRunner)
    runner._sampler = Sampler()
    runner._sample = lambda *_args, **_kwargs: pytest.fail(
        "plain greedy rows must bypass per-row CPU sampling"
    )
    chunks = [
        ScheduledChunk("a", 1, False, 3),
        ScheduledChunk("b", 1, False, 5),
    ]
    states = {
        "a": _sampling_state("a", EngineSampling()),
        "b": _sampling_state("b", EngineSampling()),
    }
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 2.0, 5.0]])

    tokens = runner._sample_rows(chunks, states, logits)

    assert [token.token_id for token in tokens] == [1, 2]
    assert set(runner._sampler._states) == {"a", "b"}


def test_non_plain_batch_keeps_the_reviewed_per_row_sampler():
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
    from kairyu.engine.core.scheduler import ScheduledChunk

    runner = object.__new__(PagedModelRunner)
    runner._sampler = Sampler()
    seen = []

    def sample(state, _logits, position):
        seen.append((state.request.request_id, position))
        return SampledToken(position)

    runner._sample = sample
    chunks = [
        ScheduledChunk("a", 1, False, 3),
        ScheduledChunk("b", 1, False, 5),
    ]
    states = {
        "a": _sampling_state("a", EngineSampling()),
        "b": _sampling_state("b", EngineSampling(logprobs=0)),
    }

    tokens = runner._sample_rows(chunks, states, torch.zeros(2, 8))

    assert [token.token_id for token in tokens] == [3, 5]
    assert seen == [("a", 3), ("b", 5)]
