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
