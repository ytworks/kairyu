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
    runner = PagedModelRunner(model, pool, sampler=Sampler(), cache=cache, **extra)
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


# --- [P1]: the scratch page the padding rows write to -------------------------


def _graph_runner(llama_dir, *, num_pages=8, page_size=4, max_batch=4, max_pages=1):
    """A runner over a REAL cache/pool, so the scratch page has real KV behind it."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.step_executor import SnapshotGraphBackend
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    pool = PagedKVPool.for_cache(cache, config)
    runner = PagedModelRunner(
        model, pool, cache=cache, graph_backend=SnapshotGraphBackend(),
        graph_max_batch=max_batch, graph_max_pages=max_pages,
    )
    return runner, cache, pool


def test_the_graph_scratch_page_is_reserved_out_of_the_allocator(llama_dir):
    """m17 A5: the page the padding rows write to must be one the scheduler can
    never hand out. Draining the cache is the only honest way to ask."""
    runner, cache, _pool = _graph_runner(llama_dir)

    handed_out = [cache.allocate_private_page() for _ in range(cache.num_free_pages)]

    assert runner._graph_scratch_page not in handed_out
    assert cache.num_free_pages == 0  # the scratch page was the only one held back


def test_padding_rows_never_write_kv_into_an_allocatable_page(llama_dir):
    """The [P1] corruption, end to end.

    A batch of 3 replays in bucket 4, so one padding row executes with
    position 0 / seq_len 1 and writes its K/V wherever the scratch page points.
    With an unreserved scratch page that is page 0 — a page a future request
    gets handed. Nothing the graph does may touch a page still in the free list.
    """
    runner, cache, pool = _graph_runner(llama_dir, num_pages=8)

    # four requests take a page each; one of them finishes and returns its page
    handed_out = [cache.allocate_private_page() for _ in range(4)]
    finished, rows = handed_out[0], handed_out[1:]
    cache.free_private_pages((finished,))

    pool.k.fill_(7.0)  # a value no projection will coincidentally produce
    pool.v.fill_(7.0)
    before_k, before_v = pool.k.clone(), pool.v.clone()

    runner._graph_logits([1, 2, 3], [0, 0, 0], [[page] for page in rows], [1, 1, 1])

    free_pages = [cache.allocate_private_page() for _ in range(cache.num_free_pages)]
    assert finished in free_pages, "the returned page is not back in circulation"
    for page in free_pages:
        assert torch.equal(pool.k[:, page], before_k[:, page]), f"K of page {page} clobbered"
        assert torch.equal(pool.v[:, page], before_v[:, page]), f"V of page {page} clobbered"
    # and the real rows really did run (otherwise the check above is vacuous)
    assert not torch.equal(pool.k[:, rows[0]], before_k[:, rows[0]])


def test_a_graph_backend_without_a_cache_is_rejected(llama_dir):
    """No cache means no allocator to reserve from — that must fail loudly, not
    fall back to a page id the scheduler is about to hand out."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.step_executor import SnapshotGraphBackend
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    cache = RadixKVCache(num_pages=16, page_size=4)
    with pytest.raises(ValueError, match="scratch page"):
        PagedModelRunner(
            model, PagedKVPool.for_cache(cache, config),
            graph_backend=SnapshotGraphBackend(),
            graph_max_batch=4, graph_max_pages=2,
        )


# --- [P2]: backends that cannot run the tensor decode contract ----------------


class _ProtocolOnlyBackend:
    """An AttentionBackend at the m13 protocol and nothing more — no
    ``attend_decode``. That is FlashInfer today (flashinfer_gpu.py)."""

    def attend(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("unused")

    def attend_batched(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("unused")


def test_an_attention_backend_without_attend_decode_is_rejected(llama_dir):
    """Constructing used to succeed and then die with AttributeError on the
    first batched decode, arbitrarily deep into a run."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.step_executor import SnapshotGraphBackend
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    for layer in model.model.layers:
        layer.self_attn.backend = _ProtocolOnlyBackend()
    cache = RadixKVCache(num_pages=16, page_size=4)
    with pytest.raises(ValueError, match="attend_decode"):
        PagedModelRunner(
            model, PagedKVPool.for_cache(cache, config), cache=cache,
            graph_backend=SnapshotGraphBackend(),
            graph_max_batch=4, graph_max_pages=2,
        )


def _mla_config():
    from kairyu.models.config import MlaConfig, ModelConfig

    return ModelConfig(
        architecture="DeepseekV3ForCausalLM", hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, vocab_size=64, rms_norm_eps=1e-6, rope_theta=10000.0,
        rope_scaling=None, tie_word_embeddings=False, dtype="float32",
        mla=MlaConfig(
            kv_lora_rank=16, q_lora_rank=None, qk_nope_head_dim=8,
            qk_rope_head_dim=4, v_head_dim=8,
        ),
    )


def test_an_mla_model_is_rejected():
    """MlaAttention has no ``forward_decode_tensors`` at all (m13 scope)."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.step_executor import SnapshotGraphBackend
    from kairyu.models.llama import DenseDecoder

    config = _mla_config()
    model = DenseDecoder(config)
    cache = RadixKVCache(num_pages=16, page_size=4)
    with pytest.raises(ValueError, match="forward_decode_tensors"):
        PagedModelRunner(
            model, PagedKVPool.for_cache(cache, config), cache=cache,
            graph_backend=SnapshotGraphBackend(),
            graph_max_batch=4, graph_max_pages=2,
        )
