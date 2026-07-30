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
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


@pytest.fixture(autouse=True)
def _no_optional_kernels(monkeypatch):
    """Hold the module docstring's promise for the tests that cannot inject.

    The `build_engine_loop` / `create_backend` cases below go through the real
    probe, which selects FlashInfer on any CUDA host that has it — and then
    rejects this fixture's 32-wide head outright (`Invalid configuration :
    NUM_MMA_Q=1 ...`). These tests are about the P-D seam, not kernel selection;
    the probed backend choice is pinned in tests/gpu/test_handoff_stream_gpu.py.
    """
    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "torch")


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
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
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
        num_layers=1,
        num_pages=4,
        page_size=4,
        num_kv_heads=1,
        head_dim=4,
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
            num_layers=2,
            num_pages=num_pages,
            page_size=page_size,
            num_kv_heads=2,
            head_dim=4,
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

    _assert_pages_equal(source_pool, source_allocation.pages, dest_pool, allocation.pages)
    # and it copied the prompt's pages, not the whole pool
    for page in set(range(8)) - set(allocation.pages):
        assert not dest_pool.k[:, page].any(), f"page {page} was written"


def test_a_deferred_copy_is_not_cache_visible_before_publication():
    """A second same-prefix handoff must not reuse pages still being copied."""
    from kairyu.engine.core.pd import LocalCopyKVHandoff

    source_pool, dest_pool, source_cache, dest_cache = _copy_fixture(num_pages=16)
    tokens = tuple(range(1, 10))
    source_allocation = source_cache.allocate(tokens)
    source_cache.mark_computed(source_allocation)
    _fill(source_pool, source_allocation)
    handoff = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool, defer_publish=True)

    allocation = handoff.transfer(tokens, first_token=7, pages=tuple(source_allocation.pages))
    before = dest_cache.allocate(tokens)
    assert before.cached_pages == ()
    dest_cache.release_preempted(before)

    handoff.publish(allocation)
    after = dest_cache.allocate(tokens)
    assert after.cached_pages == allocation.new_full_pages
    dest_cache.release_preempted(after)


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
        allocation = handoff.transfer(tokens, first_token=7, pages=tuple(source_allocation.pages))
        if tokens is second:
            assert allocation.cached_pages, "no destination prefix hit to dedup"
        _assert_pages_equal(source_pool, source_allocation.pages, dest_pool, allocation.pages)


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
    wider = PagedKVPool(num_layers=2, num_pages=8, page_size=4, num_kv_heads=2, head_dim=8)
    with pytest.raises(ValueError, match="same geometry"):
        LocalCopyKVHandoff(dest_cache, source_pool, wider)


def test_bare_copy_sync_failure_retains_the_destination_allocation():
    """Without a completion event, failed synchronization forbids page reuse."""
    from kairyu.engine.core.pd import (
        HandoffCompletionLostError,
        LocalCopyKVHandoff,
    )

    source_pool, dest_pool, source_cache, dest_cache = _copy_fixture()
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    _fill(source_pool, source)
    handoff = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool)
    allocations = []
    original_allocate = dest_cache.allocate

    def capture_allocate(*args, **kwargs):
        allocation = original_allocate(*args, **kwargs)
        allocations.append(allocation)
        return allocation

    def fail_sync():
        raise RuntimeError("injected synchronization failure")

    dest_cache.allocate = capture_allocate
    handoff._synchronize_copy_devices = fail_sync

    with pytest.raises(
        HandoffCompletionLostError,
        match="could not prove completion",
    ) as caught:
        handoff.transfer(tokens, 7, tuple(source.pages))

    assert caught.value.allocation is allocations[0]
    assert allocations[0]._freed is False
    assert getattr(caught.value, "handoff_completion_lost", False)


def test_bare_copy_cleanup_failure_retains_the_destination_allocation():
    """A successful sync does not excuse losing the allocation during cleanup."""
    from kairyu.engine.core.pd import (
        HandoffCompletionLostError,
        LocalCopyKVHandoff,
    )

    source_pool, dest_pool, source_cache, dest_cache = _copy_fixture()
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    _fill(source_pool, source)
    handoff = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool)
    allocations = []
    original_allocate = dest_cache.allocate

    def capture_allocate(*args, **kwargs):
        allocation = original_allocate(*args, **kwargs)
        allocations.append(allocation)
        return allocation

    def fail_copy(_source_page, _dest_page):
        raise RuntimeError("injected copy failure")

    def fail_cleanup(_allocation):
        raise RuntimeError("injected cleanup failure")

    dest_cache.allocate = capture_allocate
    dest_cache.release_preempted = fail_cleanup
    handoff._copy_page = fail_copy

    with pytest.raises(
        HandoffCompletionLostError,
        match="could not prove completion",
    ) as caught:
        handoff.transfer(tokens, 7, tuple(source.pages))

    assert caught.value.allocation is allocations[0]
    assert allocations[0]._freed is False


# --- the assembled engine -----------------------------------------------------


def _request(request_id="a"):
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest

    return EngineRequest(request_id, tuple(range(9)), max_new_tokens=4, sampling=EngineSampling())


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
        model_dir,
        dtype=placement["dtype"],
        attention_backend=placement["attention_backend"],
    )
    cache = RadixKVCache(num_pages=64, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=2048, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=placement["dtype"], device="cpu")
    runner = PagedModelRunner(
        model,
        pool,
        sampler=Sampler(vocab_provider=resolve_tokenizer(model_dir).vocab),
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


class _ManualEvent:
    def __init__(self, log):
        self.log = log
        self.ready = False

    def query(self):
        self.log.append(f"query:{self.ready}")
        return self.ready

    def wait(self, stream=None):  # noqa: ARG002
        self.log.append("stream-wait")

    def synchronize(self):
        self.log.append("event-sync")
        self.ready = True


class _ManualProvider:
    def __init__(self):
        self.log = []
        self.events = []

    def begin(self):
        self.log.append("begin")

    def record(self):
        self.log.append("record")
        event = _ManualEvent(self.log)
        self.events.append(event)
        return event

    def gate(self, event):
        event.wait()

    def synchronize(self):
        self.log.append("stream-sync")


def test_stream_wait_does_not_publish_until_completion_query_is_true():
    """A cudaStreamWaitEvent is ordering, not host-visible completion."""
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    source_pool, dest_pool, source_cache, _ = _copy_fixture(num_pages=16)
    stored = []
    dest_cache = RadixKVCache(num_pages=16, page_size=dest_pool.page_size, event_sink=stored.append)
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    _fill(source_pool, source)
    provider = _ManualProvider()
    handoff = StreamCopyKVHandoff(
        LocalCopyKVHandoff(dest_cache, source_pool, dest_pool),
        provider,
        defer=True,
    )

    allocation = handoff.transfer(tokens, 7, tuple(source.pages))
    completion = handoff.completion_for(allocation)
    assert stored == []
    handoff.gate_pending()
    assert stored == []
    assert handoff.pending_events, "stream wait consumed physical ownership"
    assert handoff.finalize_completion(completion) is False
    assert allocation._freed is False

    provider.events[-1].ready = True
    assert handoff.finalize_completion(completion) is True
    assert [event["type"] for event in stored] == ["BlockStored"]
    matched = dest_cache.allocate(tokens)
    assert matched.cached_pages == allocation.new_full_pages
    dest_cache.release_preempted(matched)


def test_blocking_stream_publishes_only_after_stream_synchronization():
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    source_pool, dest_pool, source_cache, _ = _copy_fixture(num_pages=16)
    provider = _ManualProvider()
    dest_cache = RadixKVCache(
        num_pages=16,
        page_size=dest_pool.page_size,
        event_sink=lambda event: provider.log.append(event["type"]),
    )
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    _fill(source_pool, source)
    handoff = StreamCopyKVHandoff(
        LocalCopyKVHandoff(dest_cache, source_pool, dest_pool),
        provider,
        defer=False,
    )

    handoff.transfer(tokens, 7, tuple(source.pages))

    assert provider.log[-2:] == ["stream-sync", "BlockStored"]


def test_partial_copy_failure_keeps_pages_until_completion_then_cleans_up():
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd import KVHandoffError, LocalCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    source_pool, dest_pool, source_cache, _ = _copy_fixture(num_pages=16)
    stored = []
    dest_cache = RadixKVCache(num_pages=16, page_size=dest_pool.page_size, event_sink=stored.append)
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    _fill(source_pool, source)
    inner = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool)
    provider = _ManualProvider()
    handoff = StreamCopyKVHandoff(inner, provider, defer=True)
    original_copy = inner._copy_page
    calls = 0

    def fail_after_one(source_page, dest_page):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected partial copy")
        original_copy(source_page, dest_page)

    inner._copy_page = fail_after_one
    with pytest.raises(KVHandoffError, match="destination KV copy failed") as caught:
        handoff.transfer(tokens, 7, tuple(source.pages))
    allocation = caught.value.allocation
    completion = caught.value.handoff_completion
    assert allocation is not None and completion is not None
    assert allocation._freed is False
    assert stored == []
    assert handoff.finalize_completion(completion) is False

    provider.events[-1].ready = True
    with pytest.raises(KVHandoffError, match="destination KV copy failed"):
        handoff.finalize_completion(completion)
    assert allocation._freed is True
    assert stored == []
    assert handoff.pending_events == ()


def test_publish_and_cleanup_failure_retains_completion_ownership():
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd import KVHandoffError, LocalCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    source_pool, dest_pool, source_cache, _ = _copy_fixture(num_pages=16)

    def reject_publication(event):  # noqa: ARG001 - failure injection
        raise RuntimeError("injected publication failure")

    dest_cache = RadixKVCache(
        num_pages=16,
        page_size=dest_pool.page_size,
        event_sink=reject_publication,
    )
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    _fill(source_pool, source)
    provider = _ManualProvider()
    handoff = StreamCopyKVHandoff(
        LocalCopyKVHandoff(dest_cache, source_pool, dest_pool),
        provider,
        defer=True,
    )
    allocation = handoff.transfer(tokens, 7, tuple(source.pages))
    completion = handoff.completion_for(allocation)
    provider.events[-1].ready = True
    original_release = dest_cache.release_preempted
    fail_cleanup = True

    def release_once(*args, **kwargs):
        nonlocal fail_cleanup
        if fail_cleanup:
            fail_cleanup = False
            raise RuntimeError("injected cleanup failure")
        return original_release(*args, **kwargs)

    dest_cache.release_preempted = release_once
    with pytest.raises(RuntimeError, match="cleanup failure"):
        handoff.finalize_completion(completion)

    assert allocation._freed is False
    assert handoff.completion_for(allocation) == completion
    assert handoff.pending_events

    with pytest.raises(KVHandoffError, match="publication failed"):
        handoff.finalize_completion(completion)
    assert allocation._freed is True
    assert handoff.pending_events == ()


@pytest.mark.parametrize("defer", [False, True])
def test_completion_primitive_failure_discards_an_unpublished_allocation(defer):
    """A missing event or failed sync must not strand or publish destination KV."""
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    source_pool, dest_pool, source_cache, _ = _copy_fixture(num_pages=16)
    stored = []
    dest_cache = RadixKVCache(num_pages=16, page_size=dest_pool.page_size, event_sink=stored.append)
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    _fill(source_pool, source)
    inner = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool)
    captured = []
    original_transfer = inner.transfer

    def capture_allocation(*args, **kwargs):
        allocation = original_transfer(*args, **kwargs)
        captured.append(allocation)
        return allocation

    inner.transfer = capture_allocation

    class _CompletionFailureProvider(_ManualProvider):
        def record(self):
            self.log.append("record-error")
            raise RuntimeError("injected event record failure")

        def synchronize(self):
            self.log.append("stream-sync-error" if not defer else "stream-sync-fallback")
            if not defer:
                raise RuntimeError("injected stream synchronization failure")

    handoff = StreamCopyKVHandoff(inner, _CompletionFailureProvider(), defer=defer)
    expected = "event record failure" if defer else "blocking handoff synchronization failed"
    with pytest.raises(RuntimeError, match=expected):
        handoff.transfer(tokens, 7, tuple(source.pages))

    assert len(captured) == 1
    assert stored == []
    assert handoff.pending_events == ()
    if defer:
        assert captured[0]._freed is True
        # Full pages remain as an uncomputed, evictable radix node; prove they
        # are reclaimable rather than mistaking num_free_pages alone for a leak.
        replacement = dest_cache.allocate(tuple(range(100, 164)))
        assert len(replacement.pages) == 16
        dest_cache.release_preempted(replacement)
        assert handoff.poisoned_allocations == ()
    else:
        # A failed sync is not proof that queued writes stopped. Fail closed:
        # never publish, but retain ownership rather than recycling the pages.
        assert captured[0]._freed is False
        assert handoff.poisoned_allocations == (captured[0],)


def test_record_fallback_cleanup_failure_retains_the_allocation():
    from kairyu.engine.core.handoff_stream import (
        CpuNoopStream,
        HandoffCompletionLostError,
        StreamCopyKVHandoff,
    )

    allocation = object()

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return allocation

        def discard(self, allocation):  # noqa: ARG002 - failure injection
            raise RuntimeError("injected cleanup failure")

    class _RecordFailure(CpuNoopStream):
        def record(self):
            raise RuntimeError("injected event record failure")

    handoff = StreamCopyKVHandoff(_Inner(), _RecordFailure(), defer=True)
    with pytest.raises(HandoffCompletionLostError, match="cleanup"):
        handoff.transfer((1,), 0)

    assert handoff.pending_events == ()
    assert handoff.poisoned_allocations == (allocation,)


def test_deferred_failed_transfer_reconciles_post_release_cleanup_error():
    """A released allocation must never be discarded twice on finalizer re-entry."""
    from kairyu.engine.core.handoff_stream import (
        CpuNoopStream,
        StreamCopyKVHandoff,
    )
    from kairyu.engine.core.pd import KVHandoffError

    class _Allocation:
        _freed = False

    allocation = _Allocation()
    fail_after_release = True

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            raise KVHandoffError(
                "injected transfer failure",
                allocation=allocation,
            )

        def discard(self, candidate):
            nonlocal fail_after_release
            if candidate._freed:
                raise ValueError("allocation was already freed")
            candidate._freed = True
            if fail_after_release:
                fail_after_release = False
                raise RuntimeError("injected post-release cleanup failure")

    handoff = StreamCopyKVHandoff(
        _Inner(),
        CpuNoopStream(),
        defer=True,
    )
    with pytest.raises(KVHandoffError) as caught:
        handoff.transfer((1,), 0)
    completion = caught.value.handoff_completion

    with pytest.raises(RuntimeError, match="post-release cleanup failure"):
        handoff.finalize_completion(completion)

    assert allocation._freed is True
    assert handoff.pending_events == ()
    with pytest.raises(KVHandoffError, match="transfer failure"):
        handoff.finalize_completion(completion)
    handoff.acknowledge_completion(completion)
    assert handoff._settled == {}


def test_blocking_transfer_cleanup_failure_retains_the_allocation():
    from kairyu.engine.core.handoff_stream import (
        CpuNoopStream,
        HandoffCompletionLostError,
        StreamCopyKVHandoff,
    )
    from kairyu.engine.core.pd import KVHandoffError

    allocation = object()

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            raise KVHandoffError("injected transfer failure", allocation=allocation)

        def discard(self, allocation):  # noqa: ARG002 - failure injection
            raise RuntimeError("injected cleanup failure")

    handoff = StreamCopyKVHandoff(_Inner(), CpuNoopStream())
    with pytest.raises(HandoffCompletionLostError, match="could not release"):
        handoff.transfer((1,), 0)

    assert handoff.pending_events == ()
    assert handoff.poisoned_allocations == (allocation,)


def test_blocking_publication_cleanup_failure_fails_closed():
    from kairyu.engine.core.handoff_stream import (
        CpuNoopStream,
        HandoffCompletionLostError,
        StreamCopyKVHandoff,
    )

    allocation = object()

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return allocation

        def publish(self, allocation):  # noqa: ARG002 - failure injection
            raise RuntimeError("injected publication failure")

        def discard(self, allocation):  # noqa: ARG002 - failure injection
            raise RuntimeError("injected cleanup failure")

    handoff = StreamCopyKVHandoff(_Inner(), CpuNoopStream())
    with pytest.raises(HandoffCompletionLostError, match="could not release"):
        handoff.transfer((1,), 0)

    assert handoff.pending_events == ()
    assert handoff.poisoned_allocations == (allocation,)


@pytest.mark.parametrize("defer", [False, True])
def test_nested_completion_loss_remains_fail_closed(defer):
    from kairyu.engine.core.handoff_stream import (
        CpuNoopStream,
        HandoffCompletionLostError,
        StreamCopyKVHandoff,
    )

    allocation = object()
    discarded = False

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            raise HandoffCompletionLostError(
                "injected inner completion loss",
                allocation=allocation,
            )

        def discard(self, allocation):  # noqa: ARG002
            nonlocal discarded
            discarded = True

    handoff = StreamCopyKVHandoff(_Inner(), CpuNoopStream(), defer=defer)
    with pytest.raises(HandoffCompletionLostError, match="inner completion loss"):
        handoff.transfer((1,), 0)

    assert discarded is False
    assert handoff.pending_events == ()
    assert handoff.poisoned_allocations == (allocation,)


def test_decode_adoption_failure_releases_destination_and_retries(model_dir):
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir,
        num_pages=64,
        page_size=16,
        **_cpu_placement(),
    )
    handoff = coordinator._handoff
    captured = []
    original_transfer = handoff.transfer

    def capture_transfer(*args, **kwargs):
        allocation = original_transfer(*args, **kwargs)
        captured.append(allocation)
        return allocation

    handoff.transfer = capture_transfer
    original_resume = coordinator.decode_scheduler.resume_with_kv
    fail_resume = True

    def resume_once(*args, **kwargs):
        nonlocal fail_resume
        if fail_resume:
            fail_resume = False
            raise RuntimeError("injected decode adoption failure")
        return original_resume(*args, **kwargs)

    coordinator.decode_scheduler.resume_with_kv = resume_once
    request = _request()
    coordinator.add_request(request)

    outputs = coordinator.run_to_completion()

    assert outputs == {"a": _single_engine_outputs(model_dir)["a"]}
    assert len(captured) == 2
    assert captured[0]._freed is True
    assert coordinator.failed_requests == ()


def test_a_raising_deferred_transfer_still_closes_the_window():
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    class _Boom:
        def transfer(self, tokens, first_token, pages=()):
            raise RuntimeError("transfer failed")

    provider = CpuNoopStream()
    with pytest.raises(RuntimeError, match="transfer failed"):
        StreamCopyKVHandoff(_Boom(), provider, defer=True).transfer((1,), 0)
    assert provider.events == ["begin", "record"]


@pytest.mark.parametrize("defer", [False, True])
def test_base_exception_still_closes_the_stream_window(defer):
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    class _Interrupted:
        def transfer(self, tokens, first_token, pages=()):
            raise KeyboardInterrupt

    provider = CpuNoopStream()
    with pytest.raises(KeyboardInterrupt):
        StreamCopyKVHandoff(_Interrupted(), provider, defer=defer).transfer((1,), 0)
    assert provider.events == ["begin", "synchronize"]


@pytest.mark.parametrize("defer", [False, True])
def test_base_exception_plus_sync_failure_retains_ownership(defer):
    from kairyu.engine.core.handoff_stream import (
        CpuNoopStream,
        HandoffCompletionLostError,
        StreamCopyKVHandoff,
    )

    allocation = object()

    class _Interrupted:
        def transfer(self, tokens, first_token, pages=()):
            interruption = KeyboardInterrupt()
            interruption.allocation = allocation
            raise interruption

        def discard(self, allocation):  # pragma: no cover - sync fails first
            raise AssertionError("unsafe cleanup")

    class _SyncFailure(CpuNoopStream):
        def synchronize(self):
            raise RuntimeError("injected synchronization failure")

    handoff = StreamCopyKVHandoff(_Interrupted(), _SyncFailure(), defer=defer)
    with pytest.raises(HandoffCompletionLostError, match="ownership remains retained"):
        handoff.transfer((1,), 0)

    assert handoff.pending_events == ()
    assert handoff.poisoned_allocations == (allocation,)


def test_record_base_exception_synchronizes_and_reclaims_before_propagating():
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    allocation = object()
    discarded = []

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return allocation

        def discard(self, allocation):
            discarded.append(allocation)

    class _RecordInterrupted(CpuNoopStream):
        def record(self):
            self.events.append("record-interrupt")
            raise KeyboardInterrupt

    provider = _RecordInterrupted()
    handoff = StreamCopyKVHandoff(_Inner(), provider, defer=True)
    with pytest.raises(KeyboardInterrupt):
        handoff.transfer((1,), 0)

    assert provider.events == ["begin", "record-interrupt", "synchronize"]
    assert discarded == [allocation]
    assert handoff.poisoned_allocations == ()


@pytest.mark.parametrize("defer", [False, True])
def test_completion_base_exception_fails_closed_when_sync_cannot_prove_safety(defer):
    from kairyu.engine.core.handoff_stream import (
        CpuNoopStream,
        HandoffCompletionLostError,
        StreamCopyKVHandoff,
    )

    allocation = object()

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return allocation

        def discard(self, allocation):  # pragma: no cover - sync fails first
            raise AssertionError("unsafe cleanup")

    class _CompletionInterrupted(CpuNoopStream):
        def record(self):
            if defer:
                raise KeyboardInterrupt
            return super().record()

        def synchronize(self):
            raise KeyboardInterrupt

    handoff = StreamCopyKVHandoff(
        _Inner(),
        _CompletionInterrupted(),
        defer=defer,
    )
    with pytest.raises(HandoffCompletionLostError, match="synchronization"):
        handoff.transfer((1,), 0)

    assert handoff.pending_events == ()
    assert handoff.poisoned_allocations == (allocation,)


def test_interrupted_local_copy_reclaims_its_destination_after_sync():
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff
    from kairyu.engine.core.pd import LocalCopyKVHandoff

    source_pool, dest_pool, source_cache, dest_cache = _copy_fixture(num_pages=16)
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    inner = LocalCopyKVHandoff(dest_cache, source_pool, dest_pool)
    interruption = KeyboardInterrupt()

    def interrupt_copy(source_page, dest_page):  # noqa: ARG001
        raise interruption

    inner._copy_page = interrupt_copy
    handoff = StreamCopyKVHandoff(inner, CpuNoopStream(), defer=True)

    with pytest.raises(KeyboardInterrupt):
        handoff.transfer(tokens, 7, tuple(source.pages))

    allocation = interruption.allocation
    assert allocation._freed is True
    assert handoff.poisoned_allocations == ()


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
    assert len(handoff.pending_events) == 2, "a stream wait was mistaken for completion"
    handoff.wait_for_pending()
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
    assert handoff.pending_events, "stream ordering consumed physical ownership"
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
    assert coordinator._completion_ready is None


@pytest.mark.gpu
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
    assert coordinator._completion_ready is not None


@pytest.mark.gpu
def test_the_deferred_copy_can_be_declined(model_dir):
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir,
        num_pages=64,
        page_size=16,
        defer_handoff=False,
        **_cuda_placement(),
    )
    assert not coordinator._handoff.defers
    assert coordinator._completion_ready is None


@pytest.mark.gpu
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
            model_path=model_dir,
            num_pages=64,
            page_size=16,
            defer_handoff=defer_handoff,
            **placement,
        )
        coordinator.add_request(_request())
        outputs = coordinator.run_to_completion()
        assert not coordinator.failed_requests
        return outputs

    assert _generate(True) == _generate(False)


@pytest.mark.gpu
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


@pytest.mark.gpu
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


def _run(loop, request_id, prompt, max_tokens=None, params=None):
    """Drive the loop like the backend pump does, returning the last update."""
    from kairyu.sampling_params import SamplingParams

    if params is None:
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    loop.submit(request_id, prompt, params)
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
    combined, _c, _s = build_engine_loop(model_path=model_dir, num_pages=64, page_size=16)
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


def test_pd_role_placement_and_blocking_control_reach_the_factory(model_dir):
    from kairyu.engine.kairyu_backend import build_engine_loop

    loop, _cache, _scheduler = build_engine_loop(
        model_path=model_dir,
        pd_separation=True,
        pd_prefill_device="cpu",
        pd_decode_device="cpu",
        pd_defer_handoff=False,
        num_pages=64,
        page_size=16,
    )

    assert loop.pd_coordinator._prefill_runner._device.type == "cpu"
    assert loop.pd_coordinator._decode_runner._device.type == "cpu"
    assert loop.pd_coordinator._completion_ready is None
    decision = loop.pd_coordinator.attention_backend_decision
    assert decision.requested == decision.resolved == "torch"
    assert decision.components == {
        "prefill": "torch",
        "decode": "torch",
        "kv_mode": "tensor-gather",
    }
    assert decision.architecture["arch"] == "cpu"


def test_pd_role_options_are_not_silently_ignored_without_pd(model_dir):
    from kairyu.engine.kairyu_backend import build_engine_loop

    with pytest.raises(ValueError, match="require pd_separation=True"):
        build_engine_loop(model_path=model_dir, pd_prefill_device="cpu")


def test_pd_role_options_do_not_shift_legacy_backend_positionals():
    import inspect

    from kairyu.engine.kairyu_backend import KairyuBackend

    names = tuple(inspect.signature(KairyuBackend).parameters)
    legacy_tail = names.index("cuda_graph_warmup_iters")
    assert names[legacy_tail + 1 :] == (
        "pd_prefill_device",
        "pd_decode_device",
        "pd_defer_handoff",
    )


def test_pd_rejects_a_mixed_cpu_cuda_role_pair(model_dir):
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.hw_profile import HardwareProfile
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    with pytest.raises(ValueError, match="both be CUDA or both be CPU"):
        build_pd_coordinator(
            model_path=model_dir,
            profile=HardwareProfile(arch="cpu"),
            prefill_device="cpu",
            decode_device="cuda:0",
            dtype=torch.float32,
            prefill_attention_backend=TorchAttentionBackend(),
            decode_attention_backend=TorchAttentionBackend(),
        )


def test_pd_rejects_an_unsupported_role_device_before_model_loading(model_dir):
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.hw_profile import HardwareProfile
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    with pytest.raises(ValueError, match="unsupported device"):
        build_pd_coordinator(
            model_path=model_dir,
            profile=HardwareProfile(arch="cpu"),
            prefill_device="mps",
            decode_device="mps",
            dtype=torch.float32,
            prefill_attention_backend=TorchAttentionBackend(),
            decode_attention_backend=TorchAttentionBackend(),
        )


def test_pd_rejects_a_role_backend_bound_to_the_wrong_gpu(model_dir):
    from kairyu.engine.core.hw_profile import HardwareProfile
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    class _StatefulBackend:
        def __init__(self, device):
            self.device = torch.device(device)

    with pytest.raises(ValueError, match="not its role device"):
        build_pd_coordinator(
            model_path=model_dir,
            profile=HardwareProfile(arch="cuda"),
            prefill_device="cuda:0",
            decode_device="cuda:1",
            dtype=torch.float32,
            prefill_attention_backend=_StatefulBackend("cuda:1"),
            decode_attention_backend=_StatefulBackend("cuda:1"),
        )


def test_pd_rejects_one_unknown_stateful_backend_shared_across_gpus(model_dir):
    from kairyu.engine.core.hw_profile import HardwareProfile
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    shared = object()
    with pytest.raises(ValueError, match="stateful attention backend"):
        build_pd_coordinator(
            model_path=model_dir,
            profile=HardwareProfile(arch="cuda"),
            prefill_device="cuda:0",
            decode_device="cuda:1",
            dtype=torch.float32,
            prefill_attention_backend=shared,
            decode_attention_backend=shared,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tensor_parallel_size": 2}, "tensor_parallel_size"),
        ({"speculative": "ngram"}, "speculative"),
    ],
    ids=["tp", "speculative"],
)
def test_pd_separation_rejects_combinations_it_does_not_implement(model_dir, kwargs, match):
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


# --- token 0 keeps its sampling identity across the handoff (m5 D5) ----------


def _single_engine_run(model_dir, request):
    """One ordinary engine over the same checkpoint, for any request."""
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
        model_dir,
        dtype=placement["dtype"],
        attention_backend=placement["attention_backend"],
    )
    cache = RadixKVCache(num_pages=64, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=2048, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=placement["dtype"], device="cpu")
    runner = PagedModelRunner(
        model,
        pool,
        sampler=Sampler(vocab_provider=resolve_tokenizer(model_dir).vocab),
        cache=cache,
    )
    core = EngineCore(scheduler, runner)
    core.add_request(request)
    return core.run_to_completion()


def _pd_run(model_dir, request):
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cpu_placement()
    )
    coordinator.add_request(request)
    outputs = coordinator.run_to_completion()
    assert not coordinator.failed_requests
    return outputs


def _sampled_request(request_id="a", **sampling):
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest

    return EngineRequest(
        request_id,
        tuple(range(9)),
        max_new_tokens=sampling.pop("max_new_tokens", 4),
        eos_token_id=sampling.pop("eos_token_id", None),
        sampling=EngineSampling(**sampling),
    )


def test_stochastic_sampling_matches_the_single_engine_path(model_dir):
    """[P1] token 0 was sampled under the prefill CLONE's id (`a#p0`).

    `Sampler._state_for` derives the base seed from the request id when
    `seed is None`, so the default stochastic case put token 0 on a different
    RNG stream from token 1 onward -- invisible to a greedy-only parity test,
    because argmax never consults the seed at all.
    """
    request = _sampled_request(temperature=1.0)  # seed=None: id-derived stream
    assert _pd_run(model_dir, request)["a"] == _single_engine_run(model_dir, request)["a"]


def test_the_clone_samples_under_the_public_request_id(model_dir):
    """Named directly, because a token comparison can coincide."""
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cpu_placement()
    )
    coordinator.add_request(_sampled_request(temperature=1.0))
    coordinator.step_prefill()

    clone_id = coordinator.internal_id_for("a")
    clone = coordinator.prefill_scheduler.states[clone_id].request
    assert clone.request_id == clone_id != "a"
    assert clone.sampling_identity == "a"


# The `<0>`..`<255>` fixture vocab can spell no JSON at all, so every token is
# grammar-illegal and sampling degenerates before the handoff is even reached.
# This one can, which is what makes a structured-output parity test mean
# something. `_SCHEMA` is deliberately small: the point is the accept state
# crossing the handoff, not grammar coverage.
_JSON_VOCAB = [
    "{",
    "}",
    "[",
    "]",
    '"',
    ":",
    ",",
    " ",
    "a",
    "b",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "true",
    "false",
    "null",
    "-",
]
_SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


@pytest.fixture(scope="module")
def grammar_model_dir(tmp_path_factory):
    """The same tiny checkpoint with a vocab that can spell JSON."""
    pytest.importorskip("xgrammar")
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("pd-grammar")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    vocab = {token: index for index, token in enumerate(_JSON_VOCAB)}
    vocab.update({f"z{index}": index for index in range(len(_JSON_VOCAB), 256)})
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "z255"},
            }
        )
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "z255"})
    )
    return str(path)


def test_structured_output_matches_the_single_engine_path(grammar_model_dir):
    """The grammar enforcer's accept state has to cross the handoff.

    Rebuilt on the decode side it starts from the INITIAL grammar state, one
    that has never accepted token 0, so every later mask is computed against the
    wrong position and the completion diverges from a single engine's.
    """
    request = _sampled_request(json_schema=_SCHEMA, max_new_tokens=6, eos_token_id=255)
    served = _pd_run(grammar_model_dir, request)["a"]
    assert served == _single_engine_run(grammar_model_dir, request)["a"]


def test_the_grammar_state_moves_to_the_decode_sampler(grammar_model_dir):
    """The mechanism, so a coincidental token match cannot pass for it."""
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    coordinator = build_pd_coordinator(
        model_path=grammar_model_dir, num_pages=64, page_size=16, **_cpu_placement()
    )
    coordinator.add_request(
        _sampled_request(json_schema=_SCHEMA, max_new_tokens=6, eos_token_id=255)
    )
    coordinator.step_prefill()

    prefill_states = coordinator._prefill_runner.sampler._states
    decode_states = coordinator._decode_runner.sampler._states
    assert "a" not in prefill_states, "the prefill half kept the enforcer"
    carried = decode_states["a"]
    assert carried.enforcer is not None
    assert carried.accepted_position == 0, "decode restarts from an unaccepted grammar"


def _pd_engine_loop(model_dir):
    """An EngineLoop over the P-D pair with the placement injected, so the
    logprob assertions do not depend on what kernels the host happens to have."""
    from kairyu.engine.core.pd_factory import build_pd_coordinator
    from kairyu.engine.core.pd_loop import PDLoopAdapter
    from kairyu.engine.engine_loop import EngineLoop
    from kairyu.engine.tokenizer import resolve_tokenizer

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, **_cpu_placement()
    )
    adapter = PDLoopAdapter(coordinator)
    return EngineLoop(resolve_tokenizer(model_dir), adapter, adapter)


def _single_engine_loop(model_dir):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.engine_loop import EngineLoop
    from kairyu.engine.tokenizer import resolve_tokenizer
    from kairyu.models.loader import load_model

    placement = _cpu_placement()
    model, config, _generation = load_model(
        model_dir,
        dtype=placement["dtype"],
        attention_backend=placement["attention_backend"],
    )
    cache = RadixKVCache(num_pages=64, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=2048, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=placement["dtype"], device="cpu")
    resolved = resolve_tokenizer(model_dir)
    runner = PagedModelRunner(
        model, pool, sampler=Sampler(vocab_provider=resolved.vocab), cache=cache
    )
    return EngineLoop(resolved, scheduler, runner)


def test_token0_logprobs_survive_a_single_token_completion(model_dir):
    """[P1] `max_tokens=1` finishes DURING the handoff.

    The only token of the completion is sampled on the prefill runner and
    committed by `resume_with_kv`, so no decode `execute()` ever runs -- and a
    loop that builds its logprob stream out of runner output alone reports
    nothing at all for a completion that has one.
    """
    from kairyu.sampling_params import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=3)
    served = _run(_pd_engine_loop(model_dir), "one", "<3> <5> <8>", params=params)
    reference = _run(_single_engine_loop(model_dir), "one", "<3> <5> <8>", params=params)

    assert served.finished and len(served.outputs) == 1
    assert reference.logprobs is not None, "the reference reported none either"
    assert served.logprobs == reference.logprobs
    assert served.cumulative_logprob == reference.cumulative_logprob
    assert served.logprob_content == reference.logprob_content


def test_token0_logprobs_lead_the_stream_of_a_longer_completion(model_dir):
    """And in order: token 0's metadata must pair with token 0, not token 1."""
    from kairyu.sampling_params import SamplingParams

    params = SamplingParams(max_tokens=4, temperature=0.0, logprobs=2)
    served = _run(_pd_engine_loop(model_dir), "many", "<3> <5> <8>", params=params)
    reference = _run(_single_engine_loop(model_dir), "many", "<3> <5> <8>", params=params)

    assert len(served.outputs) == 4
    assert served.logprobs == reference.logprobs
    assert served.logprob_content == reference.logprob_content
