"""End-to-end speculative decoding: spec ≡ non-spec greedy (design m8 D4)."""

from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.spec_runner import SpeculativeRunner
from kairyu.engine.core.torch_runner import TinyAttentionLM, TorchPagedRunner

PAGE = 4
PROMPTS = [
    (5, 9, 2, 11, 7),
    (1, 2, 3, 1, 2, 3, 1, 2, 3),  # repetitive: n-gram drafts should hit
    (42,),
    (17, 3, 17, 3, 17, 3),
    (100, 101, 102, 103, 104, 105, 106, 107),
]


def _plain_outputs(prompt, max_new=12, seed=0):
    model = TinyAttentionLM(seed=seed)
    cache = RadixKVCache(num_pages=128, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=PAGE)
    engine = EngineCore(scheduler, TorchPagedRunner(model, num_pages=128, page_size=PAGE))
    engine.add_request(EngineRequest("a", prompt, max_new_tokens=max_new))
    return engine.run_to_completion()["a"]


def _spec_engine(k=3, seed=0, sampler=None):
    model = TinyAttentionLM(seed=seed)
    cache = RadixKVCache(num_pages=128, page_size=PAGE)
    scheduler = Scheduler(
        cache, max_num_batched_tokens=64, page_size=PAGE, speculative_tokens=k
    )
    runner = SpeculativeRunner(
        TorchPagedRunner(model, num_pages=128, page_size=PAGE, sampler=sampler)
    )
    return EngineCore(scheduler, runner), runner


def test_spec_equals_plain_greedy_across_prompts():
    total_accepted = 0
    for prompt in PROMPTS:
        reference = _plain_outputs(prompt)
        engine, runner = _spec_engine()
        engine.add_request(EngineRequest("a", prompt, max_new_tokens=12))
        assert engine.run_to_completion()["a"] == reference, f"diverged on {prompt}"
        total_accepted += runner.draft_accepted
    # tiny greedy models cycle; at least one prompt must exercise acceptance
    assert total_accepted > 0


def test_spec_equals_plain_with_eos():
    prompt = (1, 2, 3, 1, 2, 3)
    reference = _plain_outputs(prompt, max_new=16)
    eos = reference[5] if len(reference) > 5 else reference[-1]
    plain_engine_out = None
    for k in (0, 3):
        model = TinyAttentionLM(seed=0)
        cache = RadixKVCache(num_pages=128, page_size=PAGE)
        scheduler = Scheduler(
            cache, max_num_batched_tokens=64, page_size=PAGE, speculative_tokens=k
        )
        base = TorchPagedRunner(model, num_pages=128, page_size=PAGE)
        runner = SpeculativeRunner(base) if k else base
        engine = EngineCore(scheduler, runner)
        engine.add_request(
            EngineRequest("a", prompt, max_new_tokens=16, eos_token_id=eos)
        )
        out = engine.run_to_completion()["a"]
        if plain_engine_out is None:
            plain_engine_out = out
        else:
            assert out == plain_engine_out  # EOS mid-draft: identical truncation


def test_non_greedy_requests_bypass_speculation():
    engine, runner = _spec_engine(sampler=Sampler())
    engine.add_request(
        EngineRequest(
            "a",
            (1, 2, 3, 1, 2, 3),
            max_new_tokens=8,
            sampling=EngineSampling(temperature=1.0, seed=3),
        )
    )
    engine.run_to_completion()
    assert runner.draft_proposed == 0  # bypassed: no draft ever scored


def test_penalized_greedy_bypasses_speculation():
    engine, runner = _spec_engine(sampler=Sampler())
    engine.add_request(
        EngineRequest(
            "a",
            (1, 2, 3, 1, 2, 3),
            max_new_tokens=8,
            sampling=EngineSampling(temperature=0.0, repetition_penalty=1.5),
        )
    )
    engine.run_to_completion()
    assert runner.draft_proposed == 0


def test_acceptance_counters_track_rate():
    engine, runner = _spec_engine()
    engine.add_request(EngineRequest("a", (1, 2, 3, 1, 2, 3, 1, 2, 3), max_new_tokens=12))
    engine.run_to_completion()
    assert 0.0 <= runner.mean_accepted <= 1.0


def test_spec_with_multiple_concurrent_requests():
    references = {f"r{i}": _plain_outputs(p) for i, p in enumerate(PROMPTS)}
    engine, _ = _spec_engine()
    for i, prompt in enumerate(PROMPTS):
        engine.add_request(EngineRequest(f"r{i}", prompt, max_new_tokens=12))
    outputs = engine.run_to_completion()
    assert outputs == references


async def test_backend_speculative_matches_plain():
    from kairyu import SamplingParams
    from kairyu.engine.backend import GenerationRequest
    from kairyu.engine.kairyu_backend import KairyuBackend
    from kairyu.engine.tokenizer import ToyTokenizer

    class _SmallVocabTokenizer(ToyTokenizer):
        """Token ids bounded by the tiny model's 128-token vocab."""

        def encode(self, text: str) -> tuple[int, ...]:
            return tuple(t % 128 for t in super().encode(text))

    def _req(rid):
        return GenerationRequest(
            request_id=rid,
            prompt="speculative backend parity",
            sampling_params=SamplingParams(max_tokens=8),
        )

    model = TinyAttentionLM(seed=1)
    plain = KairyuBackend(
        num_pages=256,
        runner=TorchPagedRunner(model, num_pages=256, page_size=16),
        tokenizer=_SmallVocabTokenizer(),
    )
    reference = await plain.generate(_req("a"))
    model2 = TinyAttentionLM(seed=1)
    spec = KairyuBackend(
        num_pages=256,
        runner=TorchPagedRunner(model2, num_pages=256, page_size=16),
        tokenizer=_SmallVocabTokenizer(),
        speculative="ngram",
        speculative_tokens=3,
    )
    result = await spec.generate(_req("a"))
    assert result.completions[0].token_ids == reference.completions[0].token_ids


def test_backend_rejects_spec_with_tp():
    import pytest as _pytest

    from kairyu.engine.kairyu_backend import KairyuBackend

    with _pytest.raises(ValueError, match="tensor_parallel_size"):
        KairyuBackend(num_pages=64, tensor_parallel_size=2, speculative="ngram")
