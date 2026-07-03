"""m12 flagship gate: full-engine greedy == transformers generate (D6).

Chunked prefill (budget smaller than the prompt), radix reuse, and decode
paging all engaged — the whole engine, not just the model forward.
"""

import json

import pytest
import torch

from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.model_runner import PagedModelRunner
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder

transformers = pytest.importorskip("transformers")

PAGE = 4
TINY = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=512,
)


@pytest.fixture(scope="module", params=["llama", "qwen2", "qwen3"])
def arch(request, tmp_path_factory):
    torch.manual_seed(11)
    if request.param == "llama":
        hf_model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    elif request.param == "qwen2":
        hf_model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**TINY))
    else:
        hf_model = transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(**TINY, head_dim=32)
        )
    hf_model = hf_model.to(torch.float32).eval()
    path = tmp_path_factory.mktemp(f"e-{request.param}")
    hf_model.save_pretrained(path, safe_serialization=True)
    config = parse_model_config(json.loads((path / "config.json").read_text()))
    ours = DenseDecoder(config).eval()
    ours.load_state_dict(hf_model.state_dict(), strict=False)
    return request.param, hf_model, ours, config


def _engine(ours, config, budget=7, num_pages=128):
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=budget, page_size=PAGE)
    pool = PagedKVPool.for_cache(cache, config)
    runner = PagedModelRunner(ours, pool, sampler=Sampler(), cache=cache)
    return EngineCore(scheduler, runner), scheduler, cache


def _hf_greedy(hf_model, prompt: list[int], max_new: int, eos: int | None) -> list[int]:
    output = hf_model.generate(
        torch.tensor([prompt]),
        max_new_tokens=max_new,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=0,
    )
    return output[0, len(prompt):].tolist()


def test_full_engine_greedy_matches_hf_generate(arch):
    name, hf_model, ours, config = arch
    torch.manual_seed(13)
    prompt = torch.randint(0, config.vocab_size, (23,)).tolist()
    reference = _hf_greedy(hf_model, prompt, max_new=32, eos=None)
    engine, scheduler, _ = _engine(ours, config, budget=7)  # forces 4 prefill chunks
    engine.add_request(
        EngineRequest(
            "a",
            tuple(prompt),
            max_new_tokens=32,
            sampling=EngineSampling(temperature=0.0),
        )
    )
    outputs = engine.run_to_completion()["a"]
    assert list(outputs) == reference, f"{name} diverged from hf.generate"


def test_full_engine_greedy_matches_hf_with_eos(arch):
    name, hf_model, ours, config = arch
    torch.manual_seed(17)
    prompt = torch.randint(0, config.vocab_size, (12,)).tolist()
    no_eos = _hf_greedy(hf_model, prompt, max_new=24, eos=None)
    eos = no_eos[7]  # a token that WILL be generated: both sides must stop there
    reference = _hf_greedy(hf_model, prompt, max_new=24, eos=eos)
    engine, scheduler, _ = _engine(ours, config)
    engine.add_request(
        EngineRequest(
            "a",
            tuple(prompt),
            max_new_tokens=24,
            eos_token_id=eos,
            sampling=EngineSampling(temperature=0.0),
        )
    )
    outputs = engine.run_to_completion()["a"]
    assert list(outputs) == reference, f"{name} EOS handling diverged"
    assert scheduler.finish_reason("a") == "stop"


def test_radix_reuse_second_request_identical_with_cached_tokens(arch):
    name, hf_model, ours, config = arch
    torch.manual_seed(19)
    prompt = tuple(torch.randint(0, config.vocab_size, (16,)).tolist())  # page-aligned
    engine, scheduler, cache = _engine(ours, config, budget=32)
    engine.add_request(
        EngineRequest("first", prompt, max_new_tokens=8, sampling=EngineSampling())
    )
    first = engine.run_to_completion()["first"]
    engine.add_request(
        EngineRequest("second", prompt, max_new_tokens=8, sampling=EngineSampling())
    )
    engine.run_to_completion()
    second = scheduler.output_tokens("second")
    assert tuple(second) == tuple(first), f"{name}: radix-reused run diverged"
    # the full page-aligned prompt was served from the radix tree (compute
    # recomputed only the capped last token; KV writes skipped, m12 D4)
    assert scheduler.num_cached_tokens("second") == len(prompt)
    assert cache.hit_rate > 0


def test_decode_crosses_page_boundaries(arch):
    name, hf_model, ours, config = arch
    torch.manual_seed(23)
    prompt = torch.randint(0, config.vocab_size, (5,)).tolist()
    reference = _hf_greedy(hf_model, prompt, max_new=20, eos=None)  # 5 pages of decode
    engine, _, _ = _engine(ours, config, budget=64)
    engine.add_request(
        EngineRequest("a", tuple(prompt), max_new_tokens=20, sampling=EngineSampling())
    )
    assert list(engine.run_to_completion()["a"]) == reference, name


def test_seeded_sampling_reproducible_on_real_model(arch):
    _, _, ours, config = arch
    prompt = tuple(range(1, 9))
    results = []
    for _ in range(2):
        engine, _, _ = _engine(ours, config)
        engine.add_request(
            EngineRequest(
                "a",
                prompt,
                max_new_tokens=8,
                sampling=EngineSampling(temperature=1.0, seed=5),
            )
        )
        results.append(engine.run_to_completion()["a"])
    assert results[0] == results[1]
