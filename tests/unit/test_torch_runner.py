import pytest

torch = pytest.importorskip("torch")

from kairyu.engine.core.engine_core import EngineCore  # noqa: E402
from kairyu.engine.core.radix_kv import RadixKVCache  # noqa: E402
from kairyu.engine.core.scheduler import EngineRequest, Scheduler  # noqa: E402
from kairyu.engine.core.torch_runner import TinyAttentionLM, TorchPagedRunner  # noqa: E402

PAGE = 4
PROMPT = (5, 17, 3, 99, 42, 7, 63, 11, 28, 4)


def _engine(model: TinyAttentionLM, num_pages=128, budget=64):
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=budget, page_size=PAGE)
    runner = TorchPagedRunner(model=model, num_pages=num_pages, page_size=PAGE)
    return EngineCore(scheduler=scheduler, runner=runner), cache


def test_paged_kv_write_and_gather_matches_contiguous():
    model = TinyAttentionLM(seed=1)
    runner = TorchPagedRunner(model=model, num_pages=8, page_size=PAGE)
    tokens = list(PROMPT)
    pages = [0, 1, 2]
    for position, token in enumerate(tokens):
        runner._write_kv(token, position, pages)
    gathered_k, gathered_v = runner._gather_kv(pages, len(tokens))
    reference_k, reference_v = model.kv_for(torch.tensor(tokens))
    assert torch.allclose(gathered_k, reference_k, atol=1e-6)
    assert torch.allclose(gathered_v, reference_v, atol=1e-6)


def test_engine_greedy_output_equals_unpaged_reference():
    model = TinyAttentionLM(seed=2)
    engine, _ = _engine(model)
    engine.add_request(EngineRequest("a", PROMPT, max_new_tokens=6))
    outputs = engine.run_to_completion()
    assert outputs["a"] == model.reference_greedy(PROMPT, steps=6)


def test_chunked_prefill_does_not_change_outputs():
    model = TinyAttentionLM(seed=2)
    engine, _ = _engine(model, budget=3)  # prompt of 10 forced into 4 chunks
    engine.add_request(EngineRequest("a", PROMPT, max_new_tokens=6))
    outputs = engine.run_to_completion()
    assert outputs["a"] == model.reference_greedy(PROMPT, steps=6)


def test_prefix_cache_reuse_preserves_correctness():
    """Second identical prompt reuses cached KV pages and must produce
    identical tokens — validates that compute-skip serves REAL cached KV."""
    model = TinyAttentionLM(seed=3)
    engine, cache = _engine(model)
    engine.add_request(EngineRequest("first", PROMPT, max_new_tokens=5))
    first = engine.run_to_completion()["first"]
    engine.add_request(EngineRequest("second", PROMPT, max_new_tokens=5))
    second = engine.run_to_completion()["second"]
    assert cache.hit_rate > 0.0  # second request actually hit the radix cache
    assert second == first == model.reference_greedy(PROMPT, steps=5)
