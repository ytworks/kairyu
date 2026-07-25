"""The P-D construction seam (m18 D3 / G2 stage 5.3 entry).

`PDCoordinator` existed only in tests, so no deployment could reach a
`KVHandoff` — which is why `CudaStreamProvider` had no production caller. These
pin the placement decision, the KV copy the handoff owes decode, and the
assembled engine.

Placement is INJECTED, never probed: a unit test that called `probe()` would
select FlashInfer on any CUDA-visible machine and fail with ModuleNotFoundError
wherever the optional kernel is absent. The real probe path is
tests/gpu/test_handoff_stream_gpu.py.
"""

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

TINY = dict(
    vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)


def _cpu_placement():
    """The injected placement: a CPU profile AND an explicit torch backend, so
    neither the hardware nor `KAIRYU_ATTENTION_BACKEND` can reach in."""
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.hw_profile import HardwareProfile

    return dict(
        profile=HardwareProfile(arch="cpu"),
        device="cpu",
        dtype=torch.float32,
        attention_backend=TorchAttentionBackend(),
    )


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("pd-factory")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    # written locally, not fetched: a unit test must not reach the Hub
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0", "truncation": None, "padding": None,
                "added_tokens": [], "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None, "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": {f"<{index}>": index for index in range(256)},
                    "unk_token": "<0>",
                },
            }
        )
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<0>"})
    )
    return str(path)


def _pool(device="cpu"):
    from kairyu.engine.core.kv_pool import PagedKVPool

    return PagedKVPool(
        num_layers=1, num_pages=4, page_size=4, num_kv_heads=1, head_dim=4,
        device=device,
    )


class _Inner:
    def transfer(self, tokens, first_token, pages=()):
        return ("allocation", tokens, first_token, pages)


def test_a_host_pool_gets_the_plain_handoff():
    """A stream window around a host copy buys nothing, and CudaStreamProvider
    requires CUDA."""
    from kairyu.engine.core.pd_factory import build_kv_handoff

    inner = _Inner()
    assert build_kv_handoff(inner, _pool()) is inner


def test_forcing_the_side_stream_without_cuda_is_an_error():
    """Rather than silently degrading to the plain handoff."""
    from kairyu.engine.core.pd_factory import build_kv_handoff

    with pytest.raises(ValueError, match="needs a CUDA KV pool"):
        build_kv_handoff(_Inner(), _pool(), force_side_stream=True)


def test_the_side_stream_can_be_declined_explicitly():
    from kairyu.engine.core.pd_factory import build_kv_handoff

    inner = _Inner()
    assert build_kv_handoff(inner, _pool(), force_side_stream=False) is inner


# --- the copy itself ----------------------------------------------------------


def _copy_fixture(page_size=4, num_pages=8):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.radix_kv import RadixKVCache

    def pool():
        return PagedKVPool(
            num_layers=2, num_pages=num_pages, page_size=page_size,
            num_kv_heads=2, head_dim=4,
        )

    return (
        pool(),
        pool(),
        RadixKVCache(num_pages=num_pages, page_size=page_size),
        RadixKVCache(num_pages=num_pages, page_size=page_size),
    )


def _fill(pool, allocation):
    """What prefill writes: content no zeroed pool could match, into the pages
    this allocation actually computed — a cached prefix is left alone."""
    computed = (*allocation.new_full_pages, allocation.tail_page)
    for page in computed:
        if page is None:
            continue
        pool.k[:, page] = torch.randn_like(pool.k[:, page])
        pool.v[:, page] = torch.randn_like(pool.v[:, page])


def _assert_pages_equal(source_pool, source_pages, dest_pool, dest_pages):
    assert len(source_pages) == len(dest_pages)
    for source, dest in zip(source_pages, dest_pages, strict=True):
        assert torch.equal(dest_pool.k[:, dest], source_pool.k[:, source]), (
            f"k of source page {source} did not reach destination page {dest}"
        )
        assert torch.equal(dest_pool.v[:, dest], source_pool.v[:, source]), (
            f"v of source page {source} did not reach destination page {dest}"
        )


def test_the_local_copy_handoff_moves_the_kv_bytes():
    """The [P1] regression: the accounting-only handoff published the
    destination's untouched (zeroed) pages as computed."""
    from kairyu.engine.core.pd import LocalCopyKVHandoff

    torch.manual_seed(3)
    source_pool, dest_pool, source_cache, dest_cache = _copy_fixture()
    tokens = tuple(range(1, 10))  # 9 tokens over page_size 4 -> 3 pages
    source_allocation = source_cache.allocate(tokens)
    source_cache.mark_computed(source_allocation)
    _fill(source_pool, source_allocation)

    allocation = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool).transfer(
        tokens, first_token=7, pages=tuple(source_allocation.pages)
    )

    _assert_pages_equal(
        source_pool, source_allocation.pages, dest_pool, allocation.pages
    )
    # and it copied the prompt's pages, not the whole pool
    for page in set(range(8)) - set(allocation.pages):
        assert not dest_pool.k[:, page].any(), f"page {page} was written"


def test_a_destination_side_prefix_hit_still_leaves_every_page_correct():
    """Receiver-side dedup (m6 D4) skips the COPY for pages already cached in
    the destination — the leading source pages must be dropped, not the trailing
    ones, or the prompt lands page-shifted."""
    from kairyu.engine.core.pd import LocalCopyKVHandoff

    torch.manual_seed(5)
    source_pool, dest_pool, source_cache, dest_cache = _copy_fixture(num_pages=16)
    handoff = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool)

    first = tuple(range(1, 9))  # 2 full pages
    second = first + tuple(range(100, 105))  # shares both, then 2 more pages
    for tokens in (first, second):
        source_allocation = source_cache.allocate(tokens)
        source_cache.mark_computed(source_allocation)
        _fill(source_pool, source_allocation)
        allocation = handoff.transfer(
            tokens, first_token=7, pages=tuple(source_allocation.pages)
        )
        if tokens is second:
            assert allocation.cached_pages, "no destination prefix hit to dedup"
        _assert_pages_equal(
            source_pool, source_allocation.pages, dest_pool, allocation.pages
        )


def test_a_source_page_count_that_is_not_the_prompts_is_refused():
    """Silently copying the wrong pages would publish a wrong prefix."""
    from kairyu.engine.core.pd import KVHandoffError, LocalCopyKVHandoff

    source_pool, dest_pool, _source_cache, dest_cache = _copy_fixture()
    handoff = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool)

    with pytest.raises(KVHandoffError, match="expected 3 source pages"):
        handoff.transfer(tuple(range(1, 10)), first_token=7, pages=(0,))
    assert dest_cache.num_free_pages == 8, "the refused transfer leaked pages"


def test_mismatched_pool_geometry_is_refused_at_construction():
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd import LocalCopyKVHandoff

    source_pool, _dest_pool, _source_cache, dest_cache = _copy_fixture()
    wider = PagedKVPool(
        num_layers=2, num_pages=8, page_size=4, num_kv_heads=2, head_dim=8
    )
    with pytest.raises(ValueError, match="same geometry"):
        LocalCopyKVHandoff(dest_cache, source_pool, wider)


# --- the assembled engine -----------------------------------------------------


def _request(request_id="a"):
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest

    return EngineRequest(
        request_id, tuple(range(9)), max_new_tokens=4, sampling=EngineSampling()
    )


def _single_engine_outputs(model_dir):
    """One ordinary engine over the same checkpoint: the greedy reference."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.tokenizer import resolve_tokenizer
    from kairyu.models.loader import load_model

    placement = _cpu_placement()
    model, config, _generation = load_model(
        model_dir, dtype=placement["dtype"],
        attention_backend=placement["attention_backend"],
    )
    cache = RadixKVCache(num_pages=64, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=2048, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=placement["dtype"], device="cpu")
    runner = PagedModelRunner(
        model, pool, sampler=Sampler(vocab_provider=resolve_tokenizer(model_dir).vocab),
        cache=cache,
    )
    core = EngineCore(scheduler, runner)
    core.add_request(_request())
    return core.run_to_completion()


def test_the_coordinator_assembles_and_generates(model_dir):
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cpu_placement()
    )
    coordinator.add_request(_request())
    outputs = coordinator.run_to_completion()
    assert len(outputs["a"]) == 4
    assert not coordinator.failed_requests


def test_the_assembled_engine_decodes_from_the_transferred_kv(model_dir):
    """The [P1] finding at the level a user sees it: with no byte copy the
    decode half continues from a zeroed cache and generates different tokens,
    while every count-based assertion above still passes."""
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cpu_placement()
    )
    coordinator.add_request(_request())
    outputs = coordinator.run_to_completion()

    assert outputs["a"] == _single_engine_outputs(model_dir)["a"]


def test_the_factory_wires_the_copying_handoff(model_dir):
    """Named explicitly: `LocalKVHandoff` satisfies the same Protocol and every
    token-count assertion, so only the type check pins which one landed."""
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cpu_placement()
    )
    assert isinstance(coordinator._handoff, LocalCopyKVHandoff)
