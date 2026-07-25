"""Decode through a REAL CUDA graph, end to end (m17 D1, runbook §6.3).

#132 proved `CudaGraphBackend` captures and replays; nothing constructed one.
This runs the engine's decode through it and compares against eager.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
PROMPTS = [list(range(9)), list(range(4, 15)), list(range(2, 11))]


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("CUDA graph decode gate needs CUDA")


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("graph-decode")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _generate(model_dir: str, graph_backend, max_new: int = 6):
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    # torch backend: flashinfer's attend_* still take Python lists, so the
    # capturable tensor path is only defined for this one
    model, config, _ = load_model(
        model_dir, dtype=torch.bfloat16, attention_backend=TorchAttentionBackend()
    )
    model = model.to("cuda:0")
    cache = RadixKVCache(num_pages=128, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    extra = (
        {"graph_backend": graph_backend, "graph_max_batch": 8, "graph_max_pages": 8}
        if graph_backend is not None
        else {}
    )
    runner = PagedModelRunner(model, pool, sampler=Sampler(), **extra)
    core = EngineCore(scheduler, runner)
    for index, prompt in enumerate(PROMPTS):
        core.add_request(
            EngineRequest(
                f"r{index}", tuple(prompt), max_new_tokens=max_new,
                sampling=EngineSampling(),
            )
        )
    return core.run_to_completion(), runner


def test_captured_decode_matches_eager(llama_dir):
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

    _require_cuda()
    eager, _ = _generate(llama_dir, None)
    graphed, runner = _generate(llama_dir, CudaGraphBackend())

    assert runner._graph is not None
    assert runner._graph._captured, "nothing was captured; the graph path was skipped"
    assert graphed == eager
    assert all(len(tokens) == 6 for tokens in eager.values())


def test_the_graph_is_off_unless_a_backend_is_given(llama_dir):
    """The seam must not change what an existing deployment executes."""
    _require_cuda()
    _outputs, runner = _generate(llama_dir, None)
    assert runner._graph is None


def test_an_incomplete_graph_config_is_rejected(llama_dir):
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.loader import load_model

    _require_cuda()
    model, config, _ = load_model(llama_dir, dtype=torch.bfloat16)
    cache = RadixKVCache(num_pages=32, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    with pytest.raises(ValueError, match="must be >= 1"):
        PagedModelRunner(model.to("cuda:0"), pool, graph_backend=CudaGraphBackend())
