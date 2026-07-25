"""PagedModelRunner routes decode through the capture seam (m17 D1).

CPU-side, against the fake backends: the wiring, the opt-in, and the fallbacks.
The real-capture gate lives in tests/gpu.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

TINY = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
PROMPTS = [list(range(9)), list(range(4, 15)), list(range(2, 11))]


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("graph-wiring")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _generate(model_dir, graph_backend, *, max_batch=8, max_pages=8, max_new=6):
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
    scheduler = Scheduler(cache, max_num_batched_tokens=16, page_size=4)
    pool = PagedKVPool.for_cache(cache, config)
    extra = (
        {
            "graph_backend": graph_backend,
            "graph_max_batch": max_batch,
            "graph_max_pages": max_pages,
        }
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


def test_graph_decode_matches_eager(llama_dir):
    """SnapshotGraphBackend replays against the STATIC buffers, so this catches a
    runner that writes its inputs anywhere the captured region cannot see."""
    from kairyu.engine.core.step_executor import SnapshotGraphBackend

    eager, _ = _generate(llama_dir, None)
    backend = SnapshotGraphBackend()
    graphed, _runner = _generate(llama_dir, backend)

    assert backend.replays > 0, "the graph path was never taken"
    assert graphed == eager


def test_the_seam_is_off_by_default(llama_dir):
    _outputs, runner = _generate(llama_dir, None)
    assert runner._graph is None


def test_an_oversize_batch_falls_back_to_eager(llama_dir):
    """D2: never crash on a batch beyond the largest bucket."""
    from kairyu.engine.core.step_executor import SnapshotGraphBackend

    eager, _ = _generate(llama_dir, None)
    backend = SnapshotGraphBackend()
    graphed, _runner = _generate(llama_dir, backend, max_batch=2)
    assert graphed == eager


def test_a_page_table_wider_than_the_buffer_falls_back(llama_dir):
    from kairyu.engine.core.step_executor import SnapshotGraphBackend

    eager, _ = _generate(llama_dir, None)
    backend = SnapshotGraphBackend()
    graphed, _runner = _generate(llama_dir, backend, max_pages=1)
    assert graphed == eager


def test_a_backend_without_sizes_is_rejected(llama_dir):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.step_executor import SnapshotGraphBackend
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    cache = RadixKVCache(num_pages=16, page_size=4)
    with pytest.raises(ValueError, match="must be >= 1"):
        PagedModelRunner(
            model, PagedKVPool.for_cache(cache, config),
            graph_backend=SnapshotGraphBackend(),
        )


def test_invalidate_graphs_is_safe_without_a_backend(llama_dir):
    _outputs, runner = _generate(llama_dir, None)
    runner.invalidate_graphs()  # must not raise
