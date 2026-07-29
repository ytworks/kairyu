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


def _write_tiny_model(path) -> None:
    import json

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(71)
    transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
    ).to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
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
        num_layers=2,
        num_pages=4,
        page_size=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    destination = PagedKVPool(
        num_layers=2,
        num_pages=4,
        page_size=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.bfloat16,
        device="cuda:0",
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
        num_layers=2,
        num_pages=4,
        page_size=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cuda:0",
    )

    class _Extracting:
        def transfer(self, tokens, first_token, pages=()):
            return extract_pages(pool, pages)

    with torch.no_grad():
        for _ in range(50):  # queue enough that an unsynchronized read would race
            pool.k.add_(1.0)

    frames = StreamCopyKVHandoff(_Extracting(), CudaStreamProvider()).transfer((1, 2), 0, (1,))
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
        num_layers=1,
        num_pages=4,
        page_size=4,
        num_kv_heads=1,
        head_dim=4,
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
            num_layers=2,
            num_pages=8,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
            device="cuda:0",
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

    handoff = build_kv_handoff(LocalCopyKVHandoff(dest_cache, source_pool, dest_pool), source_pool)
    assert isinstance(handoff, StreamCopyKVHandoff)
    # no synchronize here on purpose: the commit point must never run ahead of
    # the copy (m18 D3), so the destination must already hold the bytes
    allocation = handoff.transfer(tokens, 7, tuple(source_allocation.pages))

    for source, dest in zip(source_allocation.pages, allocation.pages, strict=True):
        assert torch.equal(dest_pool.k[:, dest], source_pool.k[:, source])
        assert torch.equal(dest_pool.v[:, dest], source_pool.v[:, source])


def test_cross_device_copy_gates_source_reuse_and_destination_read():
    """A cross-GPU stream wait does not publish before physical completion."""
    import time

    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_kv_handoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    if torch.cuda.device_count() < 2:  # pragma: no cover - single-GPU box
        pytest.skip("needs two CUDA devices")

    def pool(device):
        return PagedKVPool(
            num_layers=2,
            num_pages=8,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
            device=device,
        )

    source_pool, dest_pool = pool("cuda:0"), pool("cuda:1")
    source_cache = RadixKVCache(num_pages=8, page_size=4)
    stored = []
    dest_cache = RadixKVCache(num_pages=8, page_size=4, event_sink=stored.append)
    tokens = tuple(range(1, 10))
    source_allocation = source_cache.allocate(tokens)
    source_cache.mark_computed(source_allocation)
    for value, page in enumerate(source_allocation.pages, start=1):
        source_pool.k[:, page].fill_(value)
        source_pool.v[:, page].fill_(-value)
    # Remove one-time peer-access setup from the deliberate pending window.
    peer_probe = torch.empty(1, device="cuda:1")
    peer_probe.copy_(torch.ones(1, device="cuda:0"))
    torch.cuda.synchronize(1)
    # Keep the source-ready event pending long enough to prove transfer() really
    # returned before the destination copy completed.
    with torch.cuda.device(0):
        torch.cuda._sleep(1_000_000_000)

    handoff = build_kv_handoff(
        LocalCopyKVHandoff(dest_cache, source_pool, dest_pool),
        source_pool,
        defer=True,
        transfer_device=dest_pool.k.device,
        gate_devices=(source_pool.k.device, dest_pool.k.device),
    )
    allocation = handoff.transfer(tokens, first_token=7, pages=tuple(source_allocation.pages))
    completion = handoff.completion_for(allocation)
    assert handoff._provider._stream.device == torch.device("cuda:1")
    assert not handoff.completion_ready(completion)
    assert stored == []

    # Installing waits on both compute streams is still not completion and must
    # not expose the prefix to cache observers.
    handoff.gate_pending()
    assert stored == []
    assert handoff.pending_events
    deadline = time.perf_counter() + 5.0
    while not handoff.completion_ready(completion):
        assert time.perf_counter() < deadline, "cross-device completion never became ready"
    assert handoff.finalize_completion(completion)
    assert [event["type"] for event in stored] == ["BlockStored"]

    # Physical completion now makes both destination reads and source reuse safe,
    # independent of which current stream a later caller selects.
    for page in source_allocation.pages:
        source_pool.k[:, page].fill_(-99)
        source_pool.v[:, page].fill_(-99)
    for value, page in enumerate(allocation.pages, start=1):
        assert torch.equal(dest_pool.k[:, page], torch.full_like(dest_pool.k[:, page], value))
        assert torch.equal(dest_pool.v[:, page], torch.full_like(dest_pool.v[:, page], -value))


def test_cross_device_copy_uses_dedicated_streams_on_both_gpus():
    """Real P2P must not fall back onto either role's compute stream.

    PyTorch's cross-device ``copy_`` consults the current stream on both GPUs.
    Making only the destination side stream current therefore queues the DMA on
    the source/default stream and serializes the next prefill forward.  This
    brackets a production-layout copy on the source-copy and destination
    dependency streams, then queues independent role work on both defaults.
    """
    import time

    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    if torch.cuda.device_count() < 2:  # pragma: no cover - single-GPU box
        pytest.skip("needs two CUDA devices")
    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("needs CUDA peer access between the role GPUs")

    class _TimedCrossProvider(CudaStreamProvider):
        def __init__(self) -> None:
            super().__init__(
                device="cuda:0",
                transfer_device="cuda:1",
                gate_devices=("cuda:0", "cuda:1"),
                require_peer_access=True,
            )
            self.source_start = torch.cuda.Event(enable_timing=True)
            self.source_done = torch.cuda.Event(enable_timing=True)
            self.dest_start = torch.cuda.Event(enable_timing=True)
            self.dest_done = None

        def begin(self) -> None:
            super().begin()
            self.source_start.record(self._source_stream)
            self.dest_start.record(self._stream)
            # Widen the structural window without making wall-clock performance
            # a gate. If source/default is accidentally used, the "next prefill"
            # interval below will start only after this delay and the P2P copy.
            with torch.cuda.device(self._source_device):
                torch.cuda._sleep(1_000_000_000)

        def record(self):
            self.source_done.record(self._source_stream)
            event = super().record()
            self.dest_done = torch.cuda.Event(enable_timing=True)
            self.dest_done.record(self._stream)
            return event

    def pool(device):
        return PagedKVPool(
            num_layers=8,
            num_pages=256,
            page_size=16,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.bfloat16,
            device=device,
        )

    source_pool, dest_pool = pool("cuda:0"), pool("cuda:1")
    source_cache = RadixKVCache(num_pages=256, page_size=16)
    stored = []
    dest_cache = RadixKVCache(num_pages=256, page_size=16, event_sink=stored.append)
    tokens = tuple(range(1, 4097))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    source_pool.k.fill_(3)
    source_pool.v.fill_(-3)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    source_base = torch.cuda.Event(enable_timing=True)
    dest_base = torch.cuda.Event(enable_timing=True)
    source_base.record(torch.cuda.current_stream(0))
    dest_base.record(torch.cuda.current_stream(1))
    source_base.synchronize()
    dest_base.synchronize()
    provider = _TimedCrossProvider()
    handoff = StreamCopyKVHandoff(
        LocalCopyKVHandoff(dest_cache, source_pool, dest_pool),
        provider,
        defer=True,
    )

    allocation = handoff.transfer(tokens, 7, tuple(source.pages))
    completion = handoff.completion_for(allocation)
    assert not handoff.completion_ready(completion), "the deferred P2P copy already blocked"
    assert stored == []

    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_done = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(0):
        prefill_start.record()
        torch.cuda._sleep(500_000_000)
        prefill_done.record()
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_done = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(1):
        decode_start.record()
        torch.cuda._sleep(500_000_000)
        decode_done.record()

    deadline = time.perf_counter() + 5.0
    while not handoff.completion_ready(completion):
        assert time.perf_counter() < deadline, "cross-device completion never became ready"
    assert handoff.finalize_completion(completion)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    source_copy = (
        source_base.elapsed_time(provider.source_start),
        source_base.elapsed_time(provider.source_done),
    )
    next_prefill = (
        source_base.elapsed_time(prefill_start),
        source_base.elapsed_time(prefill_done),
    )
    assert source_copy[0] < next_prefill[1] and next_prefill[0] < source_copy[1], (
        f"source-copy stream {source_copy} ms did not overlap source/default "
        f"prefill work {next_prefill} ms"
    )
    dest_copy = (
        dest_base.elapsed_time(provider.dest_start),
        dest_base.elapsed_time(provider.dest_done),
    )
    next_decode = (
        dest_base.elapsed_time(decode_start),
        dest_base.elapsed_time(decode_done),
    )
    assert dest_copy[0] < next_decode[1] and next_decode[0] < dest_copy[1], (
        f"destination dependency stream {dest_copy} ms did not overlap "
        f"destination/default decode work {next_decode} ms"
    )
    assert [event["type"] for event in stored] == ["BlockStored"]
    assert bool(torch.all(dest_pool.k == 3))
    assert bool(torch.all(dest_pool.v == -3))


def test_cross_device_plain_control_blocks_before_publication():
    """Declining the side stream remains a safe serialized control."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_kv_handoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    if torch.cuda.device_count() < 2:  # pragma: no cover - single-GPU box
        pytest.skip("needs two CUDA devices")

    def pool(device):
        return PagedKVPool(
            num_layers=2,
            num_pages=8,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
            device=device,
        )

    source_pool, dest_pool = pool("cuda:0"), pool("cuda:1")
    source_cache = RadixKVCache(num_pages=8, page_size=4)
    stored = []
    dest_cache = RadixKVCache(num_pages=8, page_size=4, event_sink=stored.append)
    tokens = tuple(range(1, 10))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    source_pool.k.fill_(5)
    source_pool.v.fill_(-5)
    source_done = torch.cuda.Event()
    with torch.cuda.device(0):
        torch.cuda._sleep(500_000_000)
        source_done.record()
    handoff = build_kv_handoff(
        LocalCopyKVHandoff(dest_cache, source_pool, dest_pool),
        source_pool,
        force_side_stream=False,
    )

    allocation = handoff.transfer(tokens, 7, tuple(source.pages))

    assert source_done.query(), "plain control published before source work completed"
    assert [event["type"] for event in stored] == ["BlockStored"]
    for page in allocation.pages:
        assert bool(torch.all(dest_pool.k[:, page] == 5))
        assert bool(torch.all(dest_pool.v[:, page] == -5))


def test_deferred_cross_device_bytes_match_the_blocking_control():
    """The overlapped P2P path must land exactly the serialized KV bytes."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_kv_handoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    if torch.cuda.device_count() < 2:  # pragma: no cover - single-GPU box
        pytest.skip("needs two CUDA devices")
    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("needs CUDA peer access between the role GPUs")

    def pool(device):
        return PagedKVPool(
            num_layers=3,
            num_pages=16,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            dtype=torch.bfloat16,
            device=device,
        )

    source_pool = pool("cuda:0")
    blocking_pool = pool("cuda:1")
    deferred_pool = pool("cuda:1")
    source_cache = RadixKVCache(num_pages=16, page_size=4)
    blocking_cache = RadixKVCache(num_pages=16, page_size=4)
    deferred_cache = RadixKVCache(num_pages=16, page_size=4)
    tokens = tuple(range(1, 14))
    source = source_cache.allocate(tokens)
    source_cache.mark_computed(source)
    generator = torch.Generator(device="cuda:0").manual_seed(223)
    source_pool.k.copy_(
        torch.randn(
            source_pool.k.shape,
            dtype=source_pool.k.dtype,
            device=source_pool.k.device,
            generator=generator,
        )
    )
    source_pool.v.copy_(
        torch.randn(
            source_pool.v.shape,
            dtype=source_pool.v.dtype,
            device=source_pool.v.device,
            generator=generator,
        )
    )

    blocking = build_kv_handoff(
        LocalCopyKVHandoff(blocking_cache, source_pool, blocking_pool),
        source_pool,
        defer=False,
        transfer_device=blocking_pool.k.device,
    )
    deferred = build_kv_handoff(
        LocalCopyKVHandoff(deferred_cache, source_pool, deferred_pool),
        source_pool,
        defer=True,
        transfer_device=deferred_pool.k.device,
        gate_devices=(source_pool.k.device, deferred_pool.k.device),
    )

    blocking_allocation = blocking.transfer(tokens, 7, tuple(source.pages))
    deferred_allocation = deferred.transfer(tokens, 7, tuple(source.pages))
    deferred.wait_for_pending()

    assert torch.equal(
        blocking_pool.k[:, blocking_allocation.pages],
        deferred_pool.k[:, deferred_allocation.pages],
    )
    assert torch.equal(
        blocking_pool.v[:, blocking_allocation.pages],
        deferred_pool.v[:, deferred_allocation.pages],
    )
    assert torch.equal(
        deferred_pool.k[:, deferred_allocation.pages],
        source_pool.k[:, source.pages].to(deferred_pool.k.device),
    )
    assert torch.equal(
        deferred_pool.v[:, deferred_allocation.pages],
        source_pool.v[:, source.pages].to(deferred_pool.v.device),
    )


def test_the_probed_default_places_the_pair_on_the_gpu(tmp_path):
    """The live path the unit tests deliberately do not take: no injected
    placement, so probe() picks the device and the selector picks FlashInfer.

    head_dim 64 is deliberate — FlashInfer's MMA tiles reject a 16-wide head
    outright, so a smaller fixture would only exercise the torch fallback.
    """
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_pd_coordinator
    from kairyu.engine.core.scheduler import EngineRequest

    _require_cuda()
    _write_tiny_model(tmp_path)

    coordinator = build_pd_coordinator(model_path=str(tmp_path), num_pages=64, page_size=16)
    handoff = coordinator._handoff
    assert isinstance(handoff, StreamCopyKVHandoff), "a device pool got the plain handoff"
    assert isinstance(handoff._inner, LocalCopyKVHandoff), "no byte copy behind it"

    coordinator.add_request(EngineRequest("a", tuple(range(9)), max_new_tokens=4))
    outputs = coordinator.run_to_completion()
    assert len(outputs["a"]) == 4
    assert not coordinator.failed_requests


def test_production_coordinator_runs_cross_device_with_distinct_backends(tmp_path):
    from kairyu.engine.core.pd_factory import build_pd_coordinator
    from kairyu.engine.core.scheduler import EngineRequest

    if torch.cuda.device_count() < 2:  # pragma: no cover - single-GPU box
        pytest.skip("needs two CUDA devices")
    _write_tiny_model(tmp_path)

    def run(defer_handoff):
        coordinator = build_pd_coordinator(
            model_path=str(tmp_path),
            num_pages=64,
            page_size=16,
            prefill_device="cuda:0",
            decode_device="cuda:1",
            defer_handoff=defer_handoff,
        )
        prefill_backend = coordinator._prefill_runner._model.model.layers[0].self_attn.backend
        decode_backend = coordinator._decode_runner._model.model.layers[0].self_attn.backend
        assert coordinator._prefill_runner._device == torch.device("cuda:0")
        assert coordinator._decode_runner._device == torch.device("cuda:1")
        assert prefill_backend is not decode_backend
        coordinator.add_request(EngineRequest("a", tuple(range(1, 25)), max_new_tokens=4))
        coordinator.add_request(EngineRequest("b", tuple(range(50, 74)), max_new_tokens=4))
        outputs = coordinator.run_to_completion()
        assert not coordinator.failed_requests
        return outputs

    assert run(True) == run(False)


def test_the_provider_follows_the_pool_onto_a_second_device():
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd_factory import build_kv_handoff

    if torch.cuda.device_count() < 2:  # pragma: no cover - single-GPU box
        pytest.skip("needs 2 CUDA devices")

    class _Inner:
        def transfer(self, tokens, first_token, pages=()):
            return "allocation"

    pool = PagedKVPool(
        num_layers=1,
        num_pages=4,
        page_size=4,
        num_kv_heads=1,
        head_dim=4,
        device="cuda:1",
    )
    wrapped = build_kv_handoff(_Inner(), pool)
    assert wrapped._provider._stream.device == torch.device("cuda", 1)


def test_a_deferred_transfer_returns_before_the_copy_finishes():
    """The overlap this seam is named for.

    With `defer=True` the host is not blocked, so the producer can queue its next
    step while the copy runs. Event state proves that contract directly; host
    timing is deliberately not a gate.
    """
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()
    payload = torch.randn(1 << 20, device="cuda:0")

    class _SlowCopy:
        def transfer(self, tokens, first_token, pages=()):
            torch.cuda._sleep(500_000_000)
            return payload.clone()

    torch.cuda.synchronize()
    blocking = StreamCopyKVHandoff(_SlowCopy(), CudaStreamProvider())
    blocking_result = blocking.transfer((1,), 0)
    assert blocking.pending_event is None
    assert torch.isfinite(blocking_result).all()

    torch.cuda.synchronize()
    deferred = StreamCopyKVHandoff(_SlowCopy(), CudaStreamProvider(), defer=True)
    result = deferred.transfer((1,), 0)
    completion = deferred.completion_for(result)

    assert deferred.pending_event is not None, "no completion event was recorded"
    assert not deferred.completion_ready(completion), "deferred transfer blocked until done"

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

    This is the low-level stream-ordering primitive, not the coordinator's
    publication rule. `gate_pending()` orders a same-stream rewrite after the
    copy without stopping the host; physical ownership remains pending.
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

    handoff.gate_pending()
    source.fill_(-1.0)  # the next step, reusing the page it just got back
    torch.cuda.synchronize()

    assert torch.equal(copied, torch.full_like(copied, 7.0 * 21))


def test_stream_gating_does_not_stop_the_host_the_way_waiting_does():
    """A stream dependency returns early; physical finalization still polls."""
    from kairyu.engine.core.handoff_stream import CudaStreamProvider, StreamCopyKVHandoff

    _require_cuda()
    payload = torch.randn(1 << 20, device="cuda:0")

    class _SlowCopy:
        def transfer(self, tokens, first_token, pages=()):
            torch.cuda._sleep(500_000_000)
            return payload.clone()

    torch.cuda.synchronize()
    handoff = StreamCopyKVHandoff(_SlowCopy(), CudaStreamProvider(), defer=True)
    allocation = handoff.transfer((1,), 0)
    completion = handoff.completion_for(allocation)
    assert not handoff.completion_ready(completion)

    handoff.gate_pending()

    assert not handoff.completion_ready(completion), (
        "a stream wait was mistaken for physical completion"
    )
    assert handoff.pending_events
    handoff.wait_for_pending()
    assert handoff.pending_events == ()


def test_the_copy_overlaps_the_next_prefill_forward_on_the_coordinator():
    """[P1] a non-blocking host is not overlap.

    The gate used to sit at the end of the very step that recorded the copy, so
    every later kernel — the decode step, the next prefill forward — queued
    behind it. The host no longer stopped, but the device timeline was still
    serial. This measures the thing that actually matters, on the device clock:
    the interval the copy occupies on the side stream against the interval the
    NEXT prefill forward occupies on the caller's stream.

    The transfer interval includes one deliberately long, low-occupancy CUDA
    sleep.  That makes this an ordering assertion rather than a performance
    threshold: two bandwidth-saturating kernels are allowed to serialize at the
    GPU scheduler's discretion, and a sub-millisecond boundary must not turn OS
    or clock jitter into a false regression.
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
        """Accounting-only transfer plus a stable side-stream ordering window."""

        def __init__(self, dest_kv, payload) -> None:
            self._dest, self._payload, self.sink = dest_kv, payload, None

        def transfer(self, tokens, first_token, pages=()):
            # One spin kernel occupies little of the device, so independent
            # caller-stream work can run while it remains visibly in flight.
            # The real P2P timeline and byte parity are covered by the dedicated
            # cross-device production-layout test above.
            torch.cuda._sleep(500_000_000)
            self.sink = self._payload
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
    decode_runner = _KernelRunner(payload)
    coordinator = PDCoordinator(
        prefill_scheduler=Scheduler(prefill_kv, max_num_batched_tokens=32, page_size=4),
        prefill_runner=prefill_runner,
        decode_scheduler=Scheduler(decode_kv, max_num_batched_tokens=32, page_size=4),
        decode_runner=decode_runner,
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
    assert len(provider.spans) >= 2, "no later copy for decode to overlap"
    assert decode_runner.spans, "no eligible decode forward was queued"
    later_copy_start, later_copy_end = (base.elapsed_time(event) for event in provider.spans[1])
    decode_start, decode_end = (base.elapsed_time(event) for event in decode_runner.spans[0])
    assert later_copy_start < decode_end and decode_start < later_copy_end, (
        f"later copy ran [{later_copy_start:.3f}, {later_copy_end:.3f}] ms and "
        f"another request's decode ran [{decode_start:.3f}, {decode_end:.3f}] "
        "ms: the consumer gate fenced all eligible decode work"
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
        num_layers=1,
        num_pages=4,
        page_size=4,
        num_kv_heads=1,
        head_dim=4,
        device="cuda:0",
    )
    handoff = build_kv_handoff(_Inner(), pool, defer=True)
    assert isinstance(handoff, StreamCopyKVHandoff)
    assert handoff.defers
    assert handoff.transfer((1, 2), 0, (0,)) == "allocation"
    assert handoff.pending_events, "no completion event to gate on"
    handoff.gate_pending()
    assert handoff.pending_events, "stream wait was mistaken for physical completion"
    handoff.wait_for_pending()
    assert handoff.pending_events == ()
