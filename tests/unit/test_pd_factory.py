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


def _cuda_placement():
    """The same injection pointed at a device pool, which is what a deferred copy
    needs to exist at all — there is no side stream to defer onto on a host pool."""
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.hw_profile import HardwareProfile

    if not torch.cuda.is_available():
        pytest.skip("a deferred copy needs a device pool")
    return dict(
        profile=HardwareProfile(arch="cuda"),
        device="cuda:0",
        dtype=torch.float32,
        attention_backend=TorchAttentionBackend(),
    )


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


def test_the_deferred_handoff_records_an_event_instead_of_blocking():
    """The ordering contract, on the recording provider."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = CpuNoopStream()
    handoff = StreamCopyKVHandoff(_Inner(), provider, defer=True)
    handoff.transfer((1, 2), 0)

    assert provider.events == ["begin", "record"], provider.events
    assert handoff.pending_event is not None
    handoff.wait_for_pending()
    assert provider.events == ["begin", "record", "wait"]
    assert handoff.pending_event is None


def test_the_default_handoff_still_blocks_before_returning():
    """m6 D4: the commit point must not run ahead of the copy unless the caller
    explicitly takes that responsibility."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = CpuNoopStream()
    handoff = StreamCopyKVHandoff(_Inner(), provider)
    handoff.transfer((1, 2), 0)

    assert provider.events == ["begin", "synchronize"]
    assert handoff.pending_event is None


def test_a_raising_deferred_transfer_still_closes_the_window():
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    class _Boom:
        def transfer(self, tokens, first_token, pages=()):
            raise RuntimeError("transfer failed")

    provider = CpuNoopStream()
    with pytest.raises(RuntimeError, match="transfer failed"):
        StreamCopyKVHandoff(_Boom(), provider, defer=True).transfer((1,), 0)
    assert provider.events == ["begin", "record"]


def test_every_deferred_copy_is_settled_not_just_the_last():
    """A prefill step transfers every prompt that completed in it, so several
    copies can be in flight at once; keeping one slot drops the earlier ones."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = CpuNoopStream()
    handoff = StreamCopyKVHandoff(_Inner(), provider, defer=True)
    handoff.transfer((1,), 0)
    handoff.transfer((2,), 0)

    assert len(handoff.pending_events) == 2
    handoff.gate_pending()
    assert provider.events.count("wait") == 2, provider.events
    assert handoff.pending_events == ()


def test_gating_settles_the_copies_without_a_host_wait():
    """`gate_pending` expresses the dependency on the caller's stream — that is
    what leaves the producer's already-queued step running."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = CpuNoopStream()
    handoff = StreamCopyKVHandoff(_Inner(), provider, defer=True)
    handoff.transfer((1, 2), 0)
    handoff.gate_pending()

    assert provider.events == ["begin", "record", "wait"]
    # never the blocking form's stream-wide stop
    assert "synchronize" not in provider.events


def test_deferring_is_opt_in_because_it_is_a_consumer_contract():
    from kairyu.engine.core.pd_factory import build_cpu_kv_handoff

    assert build_cpu_kv_handoff(_Inner()).defers is False
    assert build_cpu_kv_handoff(_Inner(), defer=True).defers is True


def test_a_host_pool_stays_plain_even_when_deferring_is_asked_for():
    from kairyu.engine.core.pd_factory import build_kv_handoff

    inner = _Inner()
    assert build_kv_handoff(inner, _pool(), defer=True) is inner


def test_a_host_pool_coordinator_does_not_pretend_to_defer(model_dir):
    """There is no side stream to defer onto, so the plain blocking handoff is
    what the caller gets — and the coordinator has nothing to gate."""
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cpu_placement()
    )
    assert not isinstance(coordinator._handoff, StreamCopyKVHandoff)
    assert coordinator._gate_pending is None


def test_the_production_coordinator_enables_and_settles_the_deferred_copy(model_dir):
    """The finding this closes: nothing enabled or waited on the deferred path.

    `build_pd_coordinator` is that caller — it turns defer on wherever a side
    stream exists, and `PDCoordinator` is what holds the prefill-side lease until
    the copy's completion event releases it.
    """
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cuda_placement()
    )
    handoff = coordinator._handoff

    assert isinstance(handoff, StreamCopyKVHandoff)
    assert handoff.defers, "the one caller that settles the copy did not enable it"
    assert coordinator._gate_pending is not None


def test_the_deferred_copy_can_be_declined(model_dir):
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, defer_handoff=False,
        **_cuda_placement(),
    )
    assert not coordinator._handoff.defers
    assert coordinator._gate_pending is None


def test_the_deferred_and_blocking_copies_generate_the_same_tokens(model_dir):
    """The end-to-end gate: this runs the real deferred handoff through
    `PDCoordinator` against device pools, and overlap must not change the answer.

    It is the whole chain the review asked for — factory → coordinator →
    `LocalCopyKVHandoff`'s real page copy on a side stream — not a synthetic
    clone wrapped in a stream window.
    """
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    placement = _cuda_placement()

    def _generate(defer_handoff: bool):
        coordinator = build_pd_coordinator(
            model_path=model_dir, num_pages=64, page_size=16,
            defer_handoff=defer_handoff, **placement,
        )
        coordinator.add_request(_request())
        outputs = coordinator.run_to_completion()
        assert not coordinator.failed_requests
        return outputs

    assert _generate(True) == _generate(False)


def test_the_deferred_copy_decodes_from_the_transferred_kv_on_device(model_dir):
    """The strongest gate this PR can offer: a REAL `LocalCopyKVHandoff` page
    copy, deferred onto a side stream, with the source pages released only behind
    the gate — and the tokens still match a single engine's."""
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cuda_placement()
    )
    assert coordinator._handoff.defers
    coordinator.add_request(_request())
    outputs = coordinator.run_to_completion()

    assert outputs["a"] == _single_engine_outputs(model_dir)["a"]


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


def test_the_deferred_wrapper_still_wraps_the_copying_handoff(model_dir):
    """The lease work must compose with #140's copy, not replace it: there is one
    copying handoff in the stack, and the side stream only wraps it."""
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cuda_placement()
    )
    assert isinstance(coordinator._handoff._inner, LocalCopyKVHandoff)


# --- the serving path: pd_separation through EngineLoop (G2 stage 5.3) -------

def test_pd_separation_builds_a_coordinator_backed_loop(model_dir):
    """m2 §2.4 reserved `pd_separation` as a config surface and never wired it;
    a deployment could only reach `PDCoordinator` from Python."""
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd_loop import PDLoopAdapter
    from kairyu.engine.kairyu_backend import build_engine_loop

    loop, cache, scheduler = build_engine_loop(
        model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    coordinator = loop.pd_coordinator
    assert coordinator is not None
    # the loop drives the coordinator through the adapter, and reports the cache
    # and scheduler it ACTUALLY drives -- not a third, unrelated pair
    assert isinstance(scheduler, PDLoopAdapter)
    assert loop._scheduler is scheduler and loop._runner is scheduler
    assert scheduler.coordinator is coordinator
    assert cache is coordinator.decode_cache
    if torch.cuda.is_available():
        assert isinstance(coordinator._handoff, StreamCopyKVHandoff)


def _run(loop, request_id, prompt, max_tokens):
    """Drive the loop like the backend pump does, returning the last update."""
    from kairyu.sampling_params import SamplingParams

    loop.submit(request_id, prompt, SamplingParams(max_tokens=max_tokens, temperature=0.0))
    last = None
    for _ in range(200):
        if not loop.has_work():
            break
        for updated_id, update in loop.step():
            if updated_id == request_id:
                last = update
    assert last is not None, "the loop produced no update at all"
    return last


def test_a_request_flows_prefill_handoff_then_decode(model_dir):
    """The wiring test the construction assertions could not do: submit through
    the P-D loop and require actual output.

    Handing `EngineLoop` the bare coordinator cannot serve even one request --
    `_drain_ops` adds submissions to the scheduler it was given, so they land in
    the DECODE scheduler with no prompt KV and `PDCoordinator.add_request` is
    never called.
    """
    from kairyu.engine.kairyu_backend import build_engine_loop

    loop, _cache, scheduler = build_engine_loop(
        model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    coordinator = loop.pd_coordinator

    update = _run(loop, "pd1", "<1> <2> <3> <4>", max_tokens=5)

    assert update.finished and update.outputs
    assert coordinator.failed_requests == ()
    # the request really went through prefill: the coordinator's clone finished
    # there, and decode adopted the transferred KV
    assert coordinator.internal_id_for("pd1") is None  # reclaimed on finish
    assert scheduler.states == {}  # both halves reclaimed (E2)
    assert coordinator.prefill_scheduler.states == {}


def test_pd_output_matches_the_single_engine_path(model_dir):
    """Greedy equivalence is what proves the KV BYTES moved.

    The two halves own two pools. An accounting-only handoff marks the
    destination pages computed without copying anything into them, so decode
    attends over a zero-initialised pool and diverges from the reference.
    """
    from kairyu.engine.kairyu_backend import build_engine_loop

    prompt = "<7> <11> <13> <17> <19>"
    combined, _c, _s = build_engine_loop(
        model_path=model_dir, num_pages=64, page_size=16
    )
    reference = _run(combined, "ref", prompt, max_tokens=6)

    separated, _c2, _s2 = build_engine_loop(
        model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    served = _run(separated, "pd", prompt, max_tokens=6)

    assert len(reference.outputs) == 6
    assert served.outputs == reference.outputs

def test_pd_separation_needs_a_model():
    from kairyu.engine.kairyu_backend import build_engine_loop

    with pytest.raises(ValueError, match="needs a model_path"):
        build_engine_loop(pd_separation=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tensor_parallel_size": 2}, "tensor_parallel_size"),
        ({"speculative": "ngram"}, "speculative"),
    ],
    ids=["tp", "speculative"],
)
def test_pd_separation_rejects_combinations_it_does_not_implement(
    model_dir, kwargs, match
):
    """Fail loudly rather than silently serving a different topology."""
    from kairyu.engine.kairyu_backend import build_engine_loop

    with pytest.raises(ValueError, match=match):
        build_engine_loop(model_path=model_dir, pd_separation=True, **kwargs)


def test_the_backend_accepts_the_option(model_dir):
    """`backend: kairyu` with `options: {pd_separation: true}` in a deployment
    YAML has to reach the constructor."""
    from kairyu.engine.registry import create_backend

    backend = create_backend(
        "kairyu", model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    assert backend._loop.pd_coordinator is not None


async def test_the_backend_serves_a_generation_through_the_pair(model_dir):
    """The whole deployment path, not just the constructor: an API request has
    to come back with tokens."""
    from kairyu.engine.backend import GenerationRequest
    from kairyu.engine.registry import create_backend
    from kairyu.sampling_params import SamplingParams

    backend = create_backend(
        "kairyu", model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    result = await backend.generate(
        GenerationRequest(
            request_id="served",
            prompt="<3> <5> <8>",
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )

    assert result.finished
    assert len(result.completions[0].token_ids) == 4
    assert result.usage is not None and result.usage.prompt_tokens > 0
    # E2: nothing left behind in either half
    assert backend._scheduler.states == {}
    assert backend._loop.pd_coordinator.prefill_scheduler.states == {}
    await backend.shutdown()
