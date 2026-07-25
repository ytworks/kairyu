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
