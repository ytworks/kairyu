"""Overlap ON against a real model on real GPUs (m2 §2.2, G2 A1's second half).

The CPU gate covers the logic; this covers the path an A1 run actually takes —
bf16 weights, the flashinfer kernel, and the KV pool on device.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    path = tmp_path_factory.mktemp("gpu-overlap-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _generate(model_dir: str, core_cls, prompts, max_new: int):
    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(
        model_dir, dtype=torch.bfloat16, attention_backend=select_backend(probe())
    )
    model = model.to("cuda:0")
    cache = RadixKVCache(num_pages=128, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    core = core_cls(scheduler, PagedModelRunner(model, pool, sampler=Sampler()))
    for index, tokens in enumerate(prompts):
        core.add_request(
            EngineRequest(
                f"r{index}", tuple(tokens), max_new_tokens=max_new,
                sampling=EngineSampling(),
            )
        )
    return core.run_to_completion()


def test_overlap_on_matches_overlap_off_on_gpu(llama_dir):
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("overlap-on-GPU gate needs CUDA")

    prompts = [list(range(11)), list(range(5, 20)), list(range(3, 12))]
    off = _generate(llama_dir, EngineCore, prompts, 12)
    on = _generate(llama_dir, OverlapEngineCore, prompts, 12)

    assert on == off, "overlap ON diverged from OFF on device"
    assert all(len(tokens) == 12 for tokens in off.values())
