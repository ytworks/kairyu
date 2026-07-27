"""Multi-process TP puts its shards on the GPU (m5 D1/D3, m16 D1, runbook §6.1).

The regression these guard: `build_engine_loop` returns into `_build_dist_tp_loop`
before the block that probes the hardware and calls `model.to(...)`, so every
spawned rank used to keep the CPU/fp32 defaults of `DenseDecoder` and
`PagedKVPool`. Nothing in the CPU suite can observe that — on a CPU box those
defaults are the correct answer — which is why a real 8-GPU deployment served a
32B model from host memory in fp32 with all eight GPUs idle.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

# head_dim 64, not the 16 the CPU dist fixtures use: on a CUDA profile the
# selector picks FlashInfer, whose MMA tiles reject a 16-wide head outright, so a
# smaller fixture would test the torch fallback instead of the real GPU path
TINY_LLAMA = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
TP_VOCAB = [f"tok-{i}" for i in range(256)]
WORLD_SIZE = 2


def _require_devices(count: int) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("TP placement gates need CUDA")
    if torch.cuda.device_count() < count:  # pragma: no cover - small box
        pytest.skip(f"need {count} CUDA devices, found {torch.cuda.device_count()}")


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY_LLAMA))
    path = tmp_path_factory.mktemp("gpu-tp-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="module")
def fp8_llama_dir(tmp_path_factory):
    from tests.quant_checkpoint_fixtures import write_quantized_checkpoint

    torch.manual_seed(73)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY_LLAMA))
    source = tmp_path_factory.mktemp("gpu-tp-fp8-source")
    target = tmp_path_factory.mktemp("gpu-tp-fp8")
    model.to(torch.float32).eval().save_pretrained(source, safe_serialization=True)
    write_quantized_checkpoint("fp8", source, model, target)
    return str(target)


def _single_process_gpu_greedy(model_dir: str, prompt: list[int], max_new: int):
    """The TP=1 reference, placed exactly the way the single-process path places it."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(model_dir, dtype=torch.bfloat16)
    model = model.to("cuda:0")
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=6, page_size=4)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    engine = EngineCore(scheduler, PagedModelRunner(model, pool, sampler=Sampler()))
    engine.add_request(
        EngineRequest("a", tuple(prompt), max_new_tokens=max_new, sampling=EngineSampling())
    )
    return list(engine.run_to_completion()["a"])


def test_placement_assigns_one_device_per_rank():
    from kairyu.engine.core.worker import tp_placement

    _require_devices(WORLD_SIZE)
    first, second = tp_placement(WORLD_SIZE, 0), tp_placement(WORLD_SIZE, 1)
    assert (first.device, second.device) == ("cuda:0", "cuda:1")
    # bf16 + NCCL is what the single-process path picks for a CUDA profile; gloo
    # here would route every RowParallelLinear all_reduce through host memory
    assert first.dtype is torch.bfloat16
    assert first.backend == "nccl"


def test_force_cpu_keeps_the_host_defaults():
    from kairyu.engine.core.worker import tp_placement

    _require_devices(1)
    placement = tp_placement(WORLD_SIZE, 0, force_cpu=True)
    assert (placement.device, placement.dtype, placement.backend) == (
        "cpu",
        torch.float32,
        "gloo",
    )


def test_overcommitted_tp_degree_fails_fast():
    from kairyu.engine.core.worker import tp_placement

    _require_devices(1)
    too_many = torch.cuda.device_count() + 1
    # two shards sharing one GPU would silently halve the memory each expects
    with pytest.raises(RuntimeError, match="CUDA devices"):
        tp_placement(too_many, 0)


def test_launcher_places_shards_on_the_gpu(llama_dir):
    """The direct regression: weights must leave host memory."""
    from kairyu.engine.core.worker import DistTPLauncher

    _require_devices(WORLD_SIZE)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(0)
    launcher = DistTPLauncher(
        llama_dir, tp=WORLD_SIZE, num_pages=64, page_size=4, vocab=TP_VOCAB
    )
    try:
        # rank 0's shard lives in this process; ranks 1.. are spawned
        local = launcher.runner._local
        weight_devices = {p.device.type for p in local._model.parameters()}
        assert weight_devices == {"cuda"}, f"shard left on {weight_devices}"
        assert local._pool.k.device.type == "cuda"
        assert local._pool.k.dtype is torch.bfloat16
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(0) > before
    finally:
        launcher.shutdown()


def test_tp2_gpu_greedy_matches_single_process(llama_dir):
    """Gate A1 shape: sharding must not change the answer (m5 D1)."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.engine.core.worker import DistTPLauncher

    _require_devices(WORLD_SIZE)
    torch.manual_seed(83)
    prompt = torch.randint(0, 256, (11,)).tolist()
    # both sides bf16-on-device: a CPU fp32 reference would compare placements,
    # not sharding
    reference = _single_process_gpu_greedy(llama_dir, prompt, max_new=12)

    launcher = DistTPLauncher(
        llama_dir, tp=WORLD_SIZE, num_pages=64, page_size=4, vocab=TP_VOCAB
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


def test_tp2_fp8_checkpoint_runs_fused_on_every_rank(fp8_llama_dir):
    """FP8 weight and scale shards must survive NCCL production execution."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.engine.core.worker import DistTPLauncher
    from kairyu.quant.linear import Fp8Linear

    _require_devices(WORLD_SIZE)
    launcher = DistTPLauncher(
        fp8_llama_dir,
        tp=WORLD_SIZE,
        num_pages=64,
        page_size=4,
        vocab=TP_VOCAB,
    )
    try:
        local_q = launcher.runner._local._model.model.layers[0].self_attn.q_proj
        assert isinstance(local_q, Fp8Linear)
        assert local_q.weight.device.type == "cuda"
        assert local_q.weight_scale.shape == (TINY_LLAMA["hidden_size"] // 2, 1)

        scheduler = Scheduler(
            RadixKVCache(num_pages=64, page_size=4),
            max_num_batched_tokens=6,
            page_size=4,
        )
        engine = EngineCore(scheduler, launcher.runner)
        engine.add_request(
            EngineRequest(
                "fp8",
                tuple(range(11)),
                max_new_tokens=8,
                sampling=EngineSampling(),
            )
        )
        generated = engine.run_to_completion()["fp8"]
        assert len(generated) == 8
        assert len(set(generated)) > 1
    finally:
        launcher.shutdown()
