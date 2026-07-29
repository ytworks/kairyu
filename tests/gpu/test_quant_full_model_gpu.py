"""Every advertised quant scheme must load and generate through CUDA."""

from __future__ import annotations

import pytest
import torch

from tests.quant_checkpoint_fixtures import TINY, write_quantized_checkpoint

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def quantized_checkpoints(tmp_path_factory):
    torch.manual_seed(41)
    hf_model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    hf_model = hf_model.to(torch.float32).eval()
    source = tmp_path_factory.mktemp("gpu-quant-source")
    hf_model.save_pretrained(source, safe_serialization=True)
    paths = {}
    for scheme in ("fp8", "int8", "awq", "gptq", "nvfp4"):
        target = tmp_path_factory.mktemp(f"gpu-quant-{scheme}")
        write_quantized_checkpoint(scheme, source, hf_model, target)
        paths[scheme] = target
    return paths


def _generate(path, device: str, dtype: torch.dtype, prompt, max_new=8):
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(
        path, dtype=dtype, attention_backend=TorchAttentionBackend(),
        target_device=device,
    )
    model = model.to(device)
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=32, page_size=4)
    pool = PagedKVPool.for_cache(
        cache, config, dtype=dtype, device=device
    )
    engine = EngineCore(scheduler, PagedModelRunner(model, pool, sampler=Sampler()))
    engine.add_request(
        EngineRequest(
            "quant",
            prompt,
            max_new_tokens=max_new,
            sampling=EngineSampling(),
        )
    )
    return engine.run_to_completion()["quant"]


@pytest.mark.parametrize("scheme", ["fp8", "int8", "awq", "gptq", "nvfp4"])
def test_quantized_checkpoint_runs_full_model_on_gpu(scheme, quantized_checkpoints):
    if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
        pytest.skip("CUDA required")
    prompt = tuple(range(10))
    path = quantized_checkpoints[scheme]
    cpu_tokens = _generate(path, "cpu", torch.float32, prompt)
    gpu_tokens = _generate(path, "cuda:0", torch.bfloat16, prompt)

    assert len(gpu_tokens) == len(cpu_tokens) == 8
    agreement = sum(
        actual == expected
        for actual, expected in zip(gpu_tokens, cpu_tokens, strict=True)
    ) / len(cpu_tokens)
    assert agreement >= 0.5, f"{scheme}: CPU/GPU greedy agreement {agreement}"
    assert len(set(gpu_tokens)) > 1, f"{scheme}: degenerate output {gpu_tokens}"
