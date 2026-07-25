"""CudaStreamProvider on real devices (m18 D3's deploy-day half).

`CpuNoopStream` records call order and nothing else — it cannot show that work
lands on a side stream, that the side stream waits for the caller's queued work
before reading, or that a raising transfer still closes the window. Those are
the three ways this can go wrong on hardware.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("CudaStreamProvider needs CUDA")


def test_work_inside_the_window_runs_off_the_default_stream():
    from kairyu.engine.core.handoff_stream import CudaStreamProvider

    _require_cuda()
    provider = CudaStreamProvider()
    default = torch.cuda.current_stream()

    provider.begin()
    inside = torch.cuda.current_stream()
    provider.synchronize()

    assert inside != default, "the copy ran on the caller's stream; nothing overlaps"
    assert torch.cuda.current_stream() == default, "the window did not close"


def test_the_side_stream_waits_for_already_queued_work():
    """The ordering that matters: a copy must not read pages the forward on the
    caller's stream has not finished writing."""
    from kairyu.engine.core.handoff_stream import CudaStreamProvider

    _require_cuda()
    provider = CudaStreamProvider()
    source = torch.zeros(1 << 22, device="cuda:0")

    # queue enough work on the default stream that an unsynchronized side stream
    # would very likely read the pre-write value
    for _ in range(50):
        source.add_(1.0)

    provider.begin()
    copied = source.clone()
    provider.synchronize()

    assert torch.equal(copied, torch.full_like(copied, 50.0))


def test_a_raising_transfer_still_closes_the_window():
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()

    class _Boom:
        def transfer(self, tokens, first_token, pages=()):
            raise RuntimeError("transfer failed")

    default = torch.cuda.current_stream()
    provider = CudaStreamProvider()
    with pytest.raises(RuntimeError, match="transfer failed"):
        StreamCopyKVHandoff(_Boom(), provider).transfer((1, 2), 0)

    # a leaked stream context would silently redirect every later op on this
    # thread onto the side stream
    assert torch.cuda.current_stream() == default
    # and the provider must be reusable afterwards
    provider.begin()
    provider.synchronize()


def test_the_allocation_is_complete_when_transfer_returns():
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()
    payload = torch.arange(1 << 20, dtype=torch.float32, device="cuda:0")

    class _Copying:
        def __init__(self) -> None:
            self.result = None

        def transfer(self, tokens, first_token, pages=()):
            self.result = payload * 2
            return self.result

    inner = _Copying()
    returned = StreamCopyKVHandoff(inner, CudaStreamProvider()).transfer((1,), 0)
    # no extra synchronize here on purpose: the commit point must never run ahead
    # of the copy (m18 D3), so the values must already be right
    assert torch.equal(returned, payload * 2)


def test_a_cuda_kv_pool_round_trips_through_the_real_handoff():
    """The failure the arbitrary-arithmetic test above could not see.

    `RemoteKVHandoff.transfer()` reaches `kv_serde._to_bytes()`, which called
    `.numpy()` on the tensor directly — `TypeError: can't convert cuda:0 device
    type tensor to numpy`. A side stream is irrelevant if extraction cannot read
    a device pool at all.
    """
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.kv_serde import extract_pages, inject_page

    _require_cuda()
    source = PagedKVPool(
        num_layers=2, num_pages=4, page_size=4, num_kv_heads=1, head_dim=2,
        dtype=torch.bfloat16, device="cuda:0",
    )
    destination = PagedKVPool(
        num_layers=2, num_pages=4, page_size=4, num_kv_heads=1, head_dim=2,
        dtype=torch.bfloat16, device="cuda:0",
    )
    with torch.no_grad():
        source.k.copy_(torch.randn_like(source.k))
        source.v.copy_(torch.randn_like(source.v))

    frames = extract_pages(source, (1, 2))
    for frame in frames:
        inject_page(destination, frame.page_id, frame)

    for page in (1, 2):
        assert torch.equal(destination.k[:, page], source.k[:, page]), page
        assert torch.equal(destination.v[:, page], source.v[:, page]), page
    # and an untouched page must NOT have been written
    assert not torch.equal(destination.k[:, 0], source.k[:, 0])


def test_extraction_inside_the_window_sees_pages_written_on_the_callers_stream():
    """End to end: the D2H copy runs in the window and still reads the values the
    caller's queued writes produced."""
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.kv_serde import extract_pages

    _require_cuda()
    pool = PagedKVPool(
        num_layers=2, num_pages=4, page_size=4, num_kv_heads=1, head_dim=2,
        dtype=torch.float32, device="cuda:0",
    )

    class _Extracting:
        def transfer(self, tokens, first_token, pages=()):
            return extract_pages(pool, pages)

    with torch.no_grad():
        for _ in range(50):  # queue enough that an unsynchronized read would race
            pool.k.add_(1.0)

    frames = StreamCopyKVHandoff(_Extracting(), CudaStreamProvider()).transfer(
        (1, 2), 0, (1,)
    )
    restored = torch.frombuffer(bytearray(frames[0].fragments[0]), dtype=torch.uint8)
    values = restored.view(torch.float32)
    assert torch.equal(values, torch.full_like(values, 50.0))


def test_a_provider_on_a_second_device_waits_on_that_devices_stream():
    """Review [P2]: `current_stream()` with no argument follows the THREAD's
    current device, not the provider's."""
    from kairyu.engine.core.handoff_stream import CudaStreamProvider

    if torch.cuda.device_count() < 2:  # pragma: no cover - single-GPU box
        pytest.skip("needs 2 CUDA devices")

    torch.cuda.set_device(0)  # thread is on cuda:0 ...
    provider = CudaStreamProvider(device="cuda:1")  # ... provider is not

    source = torch.zeros(1 << 20, device="cuda:1")
    with torch.cuda.device(1):
        for _ in range(50):
            source.add_(1.0)

    provider.begin()
    copied = source.clone()
    provider.synchronize()

    assert torch.equal(copied, torch.full_like(copied, 50.0))


def test_a_device_pool_gets_the_side_stream_handoff():
    """The production selection this seam existed without.

    `PDCoordinator` had no production constructor, so nothing chose between the
    plain and side-stream handoffs. `build_kv_handoff` is that choice, keyed off
    where the KV actually lives.
    """
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd_factory import build_kv_handoff

    _require_cuda()

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return "allocation"

    device_pool = PagedKVPool(
        num_layers=1, num_pages=4, page_size=4, num_kv_heads=1, head_dim=4,
        device="cuda:0",
    )
    wrapped = build_kv_handoff(_Inner(), device_pool)
    assert isinstance(wrapped, StreamCopyKVHandoff)
    # and it works, rather than merely being constructed
    assert wrapped.transfer((1, 2), 0, (0,)) == "allocation"


def test_a_device_to_device_page_copy_lands_inside_the_stream_window():
    """The production handoff on hardware: `LocalCopyKVHandoff` is what
    `StreamCopyKVHandoff` wraps, so the D2D copy must be complete — and correct —
    by the time transfer() returns, with no synchronize of our own."""
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_kv_handoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    _require_cuda()

    def pool():
        return PagedKVPool(
            num_layers=2, num_pages=8, page_size=4, num_kv_heads=2, head_dim=64,
            dtype=torch.bfloat16, device="cuda:0",
        )

    source_pool, dest_pool = pool(), pool()
    source_cache = RadixKVCache(num_pages=8, page_size=4)
    dest_cache = RadixKVCache(num_pages=8, page_size=4)

    tokens = tuple(range(1, 10))  # 3 pages
    source_allocation = source_cache.allocate(tokens)
    source_cache.mark_computed(source_allocation)
    with torch.no_grad():
        for page in source_allocation.pages:
            source_pool.k[:, page] = torch.randn_like(source_pool.k[:, page])
            source_pool.v[:, page] = torch.randn_like(source_pool.v[:, page])

    handoff = build_kv_handoff(
        LocalCopyKVHandoff(dest_cache, source_pool, dest_pool), source_pool
    )
    assert isinstance(handoff, StreamCopyKVHandoff)
    # no synchronize here on purpose: the commit point must never run ahead of
    # the copy (m18 D3), so the destination must already hold the bytes
    allocation = handoff.transfer(tokens, 7, tuple(source_allocation.pages))

    for source, dest in zip(source_allocation.pages, allocation.pages, strict=True):
        assert torch.equal(dest_pool.k[:, dest], source_pool.k[:, source])
        assert torch.equal(dest_pool.v[:, dest], source_pool.v[:, source])


def test_the_probed_default_places_the_pair_on_the_gpu(tmp_path):
    """The live path the unit tests deliberately do not take: no injected
    placement, so probe() picks the device and the selector picks FlashInfer.

    head_dim 64 is deliberate — FlashInfer's MMA tiles reject a 16-wide head
    outright, so a smaller fixture would only exercise the torch fallback.
    """
    import json

    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_pd_coordinator
    from kairyu.engine.core.scheduler import EngineRequest

    _require_cuda()
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(71)
    transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=256, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512,
        )
    ).to(torch.float32).eval().save_pretrained(tmp_path, safe_serialization=True)
    (tmp_path / "tokenizer.json").write_text(
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
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<0>"})
    )

    coordinator = build_pd_coordinator(
        model_path=str(tmp_path), num_pages=64, page_size=16
    )
    handoff = coordinator._handoff
    assert isinstance(handoff, StreamCopyKVHandoff), "a device pool got the plain handoff"
    assert isinstance(handoff._inner, LocalCopyKVHandoff), "no byte copy behind it"

    coordinator.add_request(EngineRequest("a", tuple(range(9)), max_new_tokens=4))
    outputs = coordinator.run_to_completion()
    assert len(outputs["a"]) == 4
    assert not coordinator.failed_requests


def test_the_provider_follows_the_pool_onto_a_second_device():
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd_factory import build_kv_handoff

    if torch.cuda.device_count() < 2:  # pragma: no cover - single-GPU box
        pytest.skip("needs 2 CUDA devices")

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return "allocation"

    pool = PagedKVPool(
        num_layers=1, num_pages=4, page_size=4, num_kv_heads=1, head_dim=4,
        device="cuda:1",
    )
    wrapped = build_kv_handoff(_Inner(), pool)
    assert wrapped._provider._stream.device == torch.device("cuda", 1)


def test_a_deferred_transfer_returns_before_the_copy_finishes():
    """The overlap this seam is named for.

    With `defer=True` the host is not blocked, so the producer can queue its next
    step while the copy runs. Timed against the blocking form on the same work.
    """
    import time

    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()
    payload = torch.randn(1 << 24, device="cuda:0")  # 64 MB, long enough to time

    class _SlowCopy:
        def transfer(self, tokens, first_token, pages=()):
            out = payload.clone()
            for _ in range(20):
                out = out * 1.000001
            return out

    torch.cuda.synchronize()
    blocking = StreamCopyKVHandoff(_SlowCopy(), CudaStreamProvider())
    start = time.perf_counter()
    blocking.transfer((1,), 0)
    blocking_elapsed = time.perf_counter() - start

    torch.cuda.synchronize()
    deferred = StreamCopyKVHandoff(_SlowCopy(), CudaStreamProvider(), defer=True)
    start = time.perf_counter()
    result = deferred.transfer((1,), 0)
    deferred_elapsed = time.perf_counter() - start

    assert deferred.pending_event is not None, "no completion event was recorded"
    assert deferred_elapsed < blocking_elapsed, (
        f"deferred {deferred_elapsed:.4f}s did not beat blocking {blocking_elapsed:.4f}s"
    )

    # and the values are correct once the consumer waits
    deferred.wait_for_pending()
    assert deferred.pending_event is None
    assert torch.isfinite(result).all()


def test_waiting_on_the_event_makes_the_copy_visible():
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()
    source = torch.zeros(1 << 22, device="cuda:0")
    with torch.no_grad():
        for _ in range(50):
            source.add_(1.0)

    class _Copying:
        def transfer(self, tokens, first_token, pages=()):
            return source.clone()

    handoff = StreamCopyKVHandoff(_Copying(), CudaStreamProvider(), defer=True)
    copied = handoff.transfer((1,), 0)
    handoff.wait_for_pending()
    assert torch.equal(copied, torch.full_like(copied, 50.0))


def test_wait_for_pending_is_safe_with_nothing_pending():
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return "allocation"

    handoff = StreamCopyKVHandoff(_Inner(), CudaStreamProvider())  # blocking form
    handoff.transfer((1,), 0)
    assert handoff.pending_event is None
    handoff.wait_for_pending()  # must not raise


def test_gating_orders_a_source_reuse_after_the_copy():
    """The use-after-free this seam's defer opened, in miniature.

    `PDCoordinator` hands the prefill-side pages back once the transfer returns;
    with `defer=True` the copy is still READING them, and the next prefill step
    allocates the same page and writes it on the caller's stream. `gate_pending()`
    is what orders that rewrite after the copy — without stopping the host, so the
    step the producer already queued keeps running.
    """
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()
    source = torch.full((1 << 24,), 7.0, device="cuda:0")  # 64 MB "source page"

    class _SlowCopy:
        def transfer(self, tokens, first_token, pages=()):
            out = source.clone()
            for _ in range(20):  # keep READING the source, widening the window
                out = out + source
            return out

    handoff = StreamCopyKVHandoff(_SlowCopy(), CudaStreamProvider(), defer=True)
    torch.cuda.synchronize()
    copied = handoff.transfer((1,), 0)

    handoff.gate_pending()  # the gate PDCoordinator applies before releasing
    source.fill_(-1.0)  # the next step, reusing the page it just got back
    torch.cuda.synchronize()

    assert torch.equal(copied, torch.full_like(copied, 7.0 * 21))


def test_gating_does_not_stop_the_host_the_way_waiting_does():
    """Both settle the copy; only one of them gives up the overlap to do it."""
    import time

    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()
    payload = torch.randn(1 << 24, device="cuda:0")

    class _SlowCopy:
        def transfer(self, tokens, first_token, pages=()):
            out = payload.clone()
            for _ in range(20):
                out = out * 1.000001
            return out

    def _settle(settle_name):
        torch.cuda.synchronize()
        handoff = StreamCopyKVHandoff(_SlowCopy(), CudaStreamProvider(), defer=True)
        handoff.transfer((1,), 0)
        start = time.perf_counter()
        getattr(handoff, settle_name)()
        return time.perf_counter() - start

    _settle("gate_pending")  # warm up allocator/kernels before timing
    blocking = _settle("wait_for_pending")
    gated = _settle("gate_pending")

    assert gated < blocking, (
        f"gate_pending {gated:.4f}s did not beat wait_for_pending {blocking:.4f}s; "
        "it is supposed to express the dependency, not wait on it"
    )
    torch.cuda.synchronize()


def test_the_copy_overlaps_the_next_prefill_forward_on_the_coordinator():
    """[P1] a non-blocking host is not overlap.

    The gate used to sit at the end of the very step that recorded the copy, so
    every later kernel — the decode step, the next prefill forward — queued
    behind it. The host no longer stopped, but the device timeline was still
    serial. This measures the thing that actually matters, on the device clock:
    the interval the copy occupies on the side stream against the interval the
    NEXT prefill forward occupies on the caller's stream.
    """
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff
    from kairyu.engine.core.pd import PDCoordinator
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import SampledToken
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler

    _require_cuda()

    def _burn(payload):
        out = payload * 1.000001
        for _ in range(40):
            out = out * 1.000001
        return out

    class _TimedProvider(CudaStreamProvider):
        """CudaStreamProvider that also brackets each copy with timing events."""

        def __init__(self) -> None:
            super().__init__()
            self.spans: list[tuple] = []
            self._opened = None

        def begin(self) -> None:
            super().begin()
            self._opened = torch.cuda.Event(enable_timing=True)
            self._opened.record(self._stream)

        def record(self):
            self._close_window()
            done = torch.cuda.Event(enable_timing=True)
            done.record(self._stream)
            self.spans.append((self._opened, done))
            return done

    class _SlowCopy:
        """Accounting-only transfer plus device work long enough to be timed."""

        def __init__(self, dest_kv, payload) -> None:
            self._dest, self._payload, self.sink = dest_kv, payload, None

        def transfer(self, tokens, first_token, pages=()):
            self.sink = _burn(self._payload)
            allocation = self._dest.allocate(tuple(tokens))
            self._dest.mark_computed(allocation)
            return allocation

    class _KernelRunner:
        """A runner whose forward is real device work on the caller's stream."""

        def __init__(self, payload) -> None:
            self._payload, self.sink = payload, None
            self.spans: list[tuple] = []

        def execute(self, scheduled, states):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.sink = _burn(self._payload)
            done = torch.cuda.Event(enable_timing=True)
            done.record()
            self.spans.append((start, done))
            return {
                chunk.request_id: (SampledToken(7),)
                for chunk in scheduled
                if not chunk.is_prefill or states[chunk.request_id].prefill_done
            }

    payload = torch.randn(1 << 23, device="cuda:0")  # 32 MB per kernel
    prefill_kv = RadixKVCache(num_pages=64, page_size=4)
    decode_kv = RadixKVCache(num_pages=64, page_size=4)
    provider = _TimedProvider()
    prefill_runner = _KernelRunner(payload)
    coordinator = PDCoordinator(
        prefill_scheduler=Scheduler(prefill_kv, max_num_batched_tokens=32, page_size=4),
        prefill_runner=prefill_runner,
        decode_scheduler=Scheduler(decode_kv, max_num_batched_tokens=32, page_size=4),
        decode_runner=_KernelRunner(payload),
        handoff=StreamCopyKVHandoff(_SlowCopy(decode_kv, payload), provider, defer=True),
    )
    # 24 + 24 prompt tokens against a 32-token budget: `a` finishes prefilling in
    # step 1, `b` spills into step 2 — so step 2 has a forward to run while a's
    # copy is still going, if the gate lets it through
    coordinator.add_request(EngineRequest("a", tuple(range(1, 25)), max_new_tokens=2))
    coordinator.add_request(EngineRequest("b", tuple(range(50, 74)), max_new_tokens=2))
    base = torch.cuda.Event(enable_timing=True)
    base.record()

    coordinator.run_to_completion()

    torch.cuda.synchronize()
    assert len(provider.spans) >= 1, "the handoff never deferred"
    assert len(prefill_runner.spans) >= 2, "no second prefill forward to overlap with"
    copy_start, copy_end = (base.elapsed_time(e) for e in provider.spans[0])
    next_start, next_end = (base.elapsed_time(e) for e in prefill_runner.spans[1])
    assert copy_start < next_end and next_start < copy_end, (
        f"copy ran [{copy_start:.3f}, {copy_end:.3f}] ms and the next prefill "
        f"forward ran [{next_start:.3f}, {next_end:.3f}] ms: the two never "
        "overlapped, so the gate is still fencing the producer"
    )


def test_the_production_coordinator_defers_on_a_device_pool():
    """[P1] there was no production caller that enabled the deferred path."""
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd_factory import build_kv_handoff

    _require_cuda()

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return "allocation"

    pool = PagedKVPool(
        num_layers=1, num_pages=4, page_size=4, num_kv_heads=1, head_dim=4,
        device="cuda:0",
    )
    handoff = build_kv_handoff(_Inner(), pool, defer=True)
    assert isinstance(handoff, StreamCopyKVHandoff)
    assert handoff.defers
    assert handoff.transfer((1, 2), 0, (0,)) == "allocation"
    assert handoff.pending_events, "no completion event to gate on"
    handoff.gate_pending()
    assert handoff.pending_events == ()
