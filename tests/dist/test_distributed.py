"""m16 spawn gates: communicator contract, TP=2, EP=2, PP=2 parity (@dist)."""

import pytest
import torch

from tests.dist import dist_targets

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.dist

JSON_VOCAB = ["{", "}", "[", "]", '"', "a", "1", ":", ",", " ", "true", "null", "<eos>"]
TP_VOCAB = JSON_VOCAB + [f"unused-{i}" for i in range(256 - len(JSON_VOCAB))]

TINY_LLAMA = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
TINY_QWEN2 = dict(  # biases exercise the A4 row/column bias rules
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
TINY_MOE = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
    max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY_LLAMA))
    path = tmp_path_factory.mktemp("dist-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="module")
def qwen2_dir(tmp_path_factory):
    torch.manual_seed(73)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**TINY_QWEN2))
    path = tmp_path_factory.mktemp("dist-qwen2")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="module")
def moe_dir(tmp_path_factory):
    torch.manual_seed(79)
    model = transformers.Qwen3MoeForCausalLM(transformers.Qwen3MoeConfig(**TINY_MOE))
    with torch.no_grad():  # decisive routing margins (m15 fixture lesson)
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3MoeSparseMoeBlock":
                module.gate.weight.mul_(6.0)
    path = tmp_path_factory.mktemp("dist-moe")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _single_process_greedy(model_dir: str, prompt: list[int], max_new: int) -> list[int]:
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(model_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=6, page_size=4)
    pool = PagedKVPool.for_cache(cache, config)
    engine = EngineCore(scheduler, PagedModelRunner(model, pool, sampler=Sampler()))
    engine.add_request(
        EngineRequest("a", tuple(prompt), max_new_tokens=max_new, sampling=EngineSampling())
    )
    return list(engine.run_to_completion()["a"])


def test_communicator_contract_on_gloo(spawn2):
    results = spawn2(dist_targets.comm_contract)
    for result in results:
        assert result["broadcast"] == {"step": 7}
        assert result["reduced"] == [3.0, 20.0]  # (1+2, 10+10)
        assert result["gathered"] == ["r0", "r1"]
    assert results[1]["received"] == {"hello": 0}
    # rank0 keeps its row 0 + gets rank1's first 2; rank1 gets rank0's rows 1..3 + own tail
    assert results[0]["a2a"] == [0.0, 100.0, 101.0]
    assert results[1]["a2a"] == [1.0, 2.0, 3.0, 102.0, 103.0]


@pytest.mark.parametrize("fixture_name", ["llama_dir", "qwen2_dir"])
def test_tp2_engine_greedy_matches_single_process(spawn2, fixture_name, request):
    model_dir = request.getfixturevalue(fixture_name)
    torch.manual_seed(83)
    prompt = torch.randint(0, 256, (11,)).tolist()
    reference = _single_process_greedy(model_dir, prompt, max_new=12)
    results = spawn2(dist_targets.tp_engine_parity, model_dir, prompt, 12, TP_VOCAB)
    assert results[0]["outputs"] == reference
    assert results[1]["steps"] > 0  # the worker actually executed the steps


def test_tp_structured_sampling_and_release_on_every_rank(spawn2, llama_dir):
    results = spawn2(dist_targets.tp_structured_release, llama_dir, TP_VOCAB)
    for rank, result in enumerate(results):
        assert "error" not in result, f"rank {rank} failed: {result.get('error')}"
    assert results[0]["structured_completed"] is True
    assert results[0]["sampler_states"] == 0
    assert results[1]["sampler_states"] == 0
    assert results[0]["released_requests"] == 34
    assert results[1]["released_requests"] == 34


def test_dist_tp_launcher_serve_path_matches_single_process(llama_dir):
    # Deploy wiring: DistTPLauncher (rank 0 here + spawned worker) is what
    # `kairyu serve --tp 2` uses. Driving EngineCore through it must reproduce the
    # single-process greedy output, and shutdown() must stop the worker cleanly.
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.engine.core.worker import DistTPLauncher

    torch.manual_seed(83)
    prompt = torch.randint(0, 256, (11,)).tolist()
    reference = _single_process_greedy(llama_dir, prompt, max_new=12)

    # force_cpu: the reference is a single-process fp32 HOST run, so on a GPU
    # box the launcher must not follow the probe onto bf16 devices
    launcher = DistTPLauncher(
        llama_dir, tp=2, num_pages=64, page_size=4, vocab=TP_VOCAB, force_cpu=True
    )
    try:
        scheduler = Scheduler(
            RadixKVCache(num_pages=64, page_size=4), max_num_batched_tokens=6, page_size=4
        )
        engine = EngineCore(scheduler, launcher.runner)
        engine.add_request(
            EngineRequest("a", tuple(prompt), max_new_tokens=12, sampling=EngineSampling())
        )
        outputs = list(engine.run_to_completion()["a"])
    finally:
        launcher.shutdown()
    assert outputs == reference


def test_ep2_moe_block_matches_single_process(spawn2, moe_dir):
    results = spawn2(dist_targets.ep_block_parity, moe_dir)
    for result in results:
        assert result["local_experts"] == 4  # 8 experts / 2 ranks
        assert result["maxdiff"] < 1e-5  # accumulation-order tolerance (A7)


def test_pp2_greedy_matches_single_process(spawn2, llama_dir):
    torch.manual_seed(89)
    prompt = torch.randint(0, 256, (9,)).tolist()
    reference = _single_process_greedy(llama_dir, prompt, max_new=10)
    results = spawn2(dist_targets.pp_greedy_parity, llama_dir, prompt, 10)
    assert results[0]["outputs"] == reference
    assert results[1]["outputs"] == reference


def test_tp_speculative_matches_single_process_greedy(llama_dir):
    # spec == greedy is pinned single-process (m8 D4); it must survive sharding.
    # SpeculativeRunner composes over DistTPModelRunner, so each scoring pass it
    # issues is broadcast like any other step and every rank replays the same
    # draft — the m5 D1 agreement invariant is untouched.
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.engine.core.spec_runner import SpeculativeRunner
    from kairyu.engine.core.worker import DistTPLauncher

    torch.manual_seed(83)
    prompt = torch.randint(0, 256, (11,)).tolist()
    reference = _single_process_greedy(llama_dir, prompt, max_new=12)

    launcher = DistTPLauncher(
        llama_dir, tp=2, num_pages=64, page_size=4, vocab=TP_VOCAB, force_cpu=True
    )
    try:
        scheduler = Scheduler(
            RadixKVCache(num_pages=64, page_size=4),
            max_num_batched_tokens=6,
            page_size=4,
            speculative_tokens=4,
        )
        engine = EngineCore(scheduler, SpeculativeRunner(launcher.runner))
        engine.add_request(
            EngineRequest("a", tuple(prompt), max_new_tokens=12, sampling=EngineSampling())
        )
        outputs = list(engine.run_to_completion()["a"])
    finally:
        launcher.shutdown()
    assert outputs == reference


class _TinyTokenizer:
    """256 ids, matching the fixture's vocab_size (the toy tokenizer's 50k
    would trip the vocab-agreement guard before the wiring is reached)."""

    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(ch) % 256 for ch in text)

    def decode(self, token_ids) -> str:
        return "".join(chr(int(t) % 128) for t in token_ids)

    def vocab(self) -> list[str]:
        return [f"t{i}" for i in range(256)]


def test_build_engine_loop_accepts_speculative_with_real_model_tp(llama_dir):
    from kairyu.engine.core.spec_runner import SpeculativeRunner
    from kairyu.engine.kairyu_backend import build_engine_loop

    loop, _cache, scheduler = build_engine_loop(
        model_path=llama_dir,
        tensor_parallel_size=2,
        num_pages=64,
        page_size=4,
        max_num_batched_tokens=6,
        tokenizer=_TinyTokenizer(),
        speculative="ngram",
        speculative_tokens=4,
    )
    try:
        assert isinstance(loop._runner, SpeculativeRunner)
        # the scheduler must reserve for the draft, or every speculative chunk
        # is silently downgraded to a single token
        assert scheduler._spec_k == 4
    finally:
        loop.tp_launcher.shutdown()


def test_in_process_tp_still_rejects_speculative():
    # the toy path shares ONE runner across ranks: drafting there would score
    # against itself, so the guard stays for that case
    from kairyu.engine.kairyu_backend import build_engine_loop

    with pytest.raises(ValueError, match="requires a real model"):
        build_engine_loop(tensor_parallel_size=2, speculative="ngram")
