"""m18 gates: serde round-trips, remote handoff ordering, stream wrapper,
fake-nixl contract."""

import sys
import types

import pytest
import torch

from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.kv_serde import (
    extract_page,
    inject_page,
    pool_fingerprint,
)
from kairyu.engine.core.kv_transport import (
    KVTransportError,
    LocalFabric,
    PageFrame,
    SequenceMeta,
)
from kairyu.engine.core.pd import KVHandoffError
from kairyu.engine.core.pd_remote import RemoteKVHandoff, RemoteKVReceiver
from kairyu.engine.core.radix_kv import RadixKVCache

PAGE = 4


def _filled_pool(layers=2, kv_heads=2, head_dim=8, v_head_dim=None) -> PagedKVPool:
    torch.manual_seed(0)
    pool = PagedKVPool(layers, 8, PAGE, kv_heads, head_dim, v_head_dim=v_head_dim)
    pool.k.copy_(torch.randn_like(pool.k))
    if pool.v_head_dim:
        pool.v.copy_(torch.randn_like(pool.v))
    return pool


class TestSerde:
    def test_round_trip_gqa(self):
        source = _filled_pool()
        target = PagedKVPool(2, 8, PAGE, 2, 8)
        frame = extract_page(source, 3)
        assert len(frame.fragments) == 4  # 2 layers x (k, v)
        inject_page(target, 5, frame)  # page ids remap (m18 A3)
        for layer in range(2):
            assert torch.equal(target.k[layer, 5], source.k[layer, 3])
            assert torch.equal(target.v[layer, 5], source.v[layer, 3])

    def test_round_trip_mla_empty_v(self):
        source = _filled_pool(v_head_dim=0)
        target = PagedKVPool(2, 8, PAGE, 2, 8, v_head_dim=0)
        frame = extract_page(source, 1)
        assert frame.fragments[1] == b""  # empty v fragment
        inject_page(target, 2, frame)
        assert torch.equal(target.k[0, 2], source.k[0, 1])

    def test_round_trip_bfloat16(self):
        # Phase 6: bf16 (which numpy cannot represent) must round-trip byte-exact
        # through the uint8-view serde.
        source = PagedKVPool(2, 8, PAGE, 2, 8)
        source.k.copy_(torch.randn_like(source.k))
        source.v.copy_(torch.randn_like(source.v))
        source.k = source.k.to(torch.bfloat16)
        source.v = source.v.to(torch.bfloat16)
        target = PagedKVPool(2, 8, PAGE, 2, 8)
        target.k = target.k.to(torch.bfloat16)
        target.v = target.v.to(torch.bfloat16)
        frame = extract_page(source, 3)
        inject_page(target, 5, frame)
        assert torch.equal(target.k[0, 5], source.k[0, 3])
        assert torch.equal(target.v[1, 5], source.v[1, 3])

    def test_length_and_count_mismatches_fail_loudly(self):
        pool = _filled_pool()
        with pytest.raises(KVTransportError, match="fragments"):
            inject_page(pool, 0, PageFrame(page_id=0, fragments=(b"x",)))
        good = extract_page(pool, 0)
        bad = PageFrame(page_id=0, fragments=(b"short",) + good.fragments[1:])
        with pytest.raises(KVTransportError, match="bytes"):
            inject_page(pool, 0, bad)
        mla = PagedKVPool(2, 8, PAGE, 2, 8, v_head_dim=0)
        poisoned = PageFrame(
            page_id=0,
            fragments=(good.fragments[0], b"not-empty") + (good.fragments[0], b"")[0:2],
        )
        with pytest.raises(KVTransportError):
            inject_page(mla, 0, poisoned)

    def test_fingerprint_distinguishes_pools(self):
        assert pool_fingerprint(_filled_pool()) != pool_fingerprint(
            _filled_pool(v_head_dim=0)
        )


class TestRemoteHandoff:
    def _pair(self):
        fabric = LocalFabric()
        prefill = fabric.endpoint("prefill")
        decode = fabric.endpoint("decode")
        prefill.register(8)
        decode.register(8)
        return prefill, decode

    def test_bytes_land_and_allocation_returned(self):
        prefill_pool = _filled_pool()
        decode_pool = PagedKVPool(2, 16, PAGE, 2, 8)
        decode_cache = RadixKVCache(num_pages=16, page_size=PAGE)
        prefill_transport, decode_transport = self._pair()
        receiver = RemoteKVReceiver(decode_cache, decode_pool)
        handoff = RemoteKVHandoff(
            prefill_transport, "decode", prefill_pool, receiver,
            decode_transport, "prefill",
        )
        tokens = tuple(range(9))  # 2 full pages + tail
        allocation = handoff.transfer(tokens, first_token=42, pages=(1, 3, 5))
        assert allocation.num_cached_tokens == 0
        targets = tuple(allocation.new_full_pages) + (allocation.tail_page,)
        for source_page, local_page in zip((1, 3, 5), targets, strict=True):
            for layer in range(2):
                assert torch.equal(
                    decode_pool.k[layer, local_page], prefill_pool.k[layer, source_page]
                )
        assert receiver.injected_pages == 3

    def test_receiver_dedup_skips_cached_pages(self):
        prefill_pool = _filled_pool()
        decode_pool = PagedKVPool(2, 16, PAGE, 2, 8)
        decode_cache = RadixKVCache(num_pages=16, page_size=PAGE)
        prefill_transport, decode_transport = self._pair()
        receiver = RemoteKVReceiver(decode_cache, decode_pool)
        handoff = RemoteKVHandoff(
            prefill_transport, "decode", prefill_pool, receiver,
            decode_transport, "prefill",
        )
        tokens = tuple(range(9))
        first = handoff.transfer(tokens, 42, pages=(1, 3, 5))
        decode_cache.commit_and_release(first, output_tokens=(), decode_pages=())
        second = handoff.transfer(tokens, 42, pages=(1, 3, 5))
        assert second.num_cached_tokens == 8  # 2 full pages radix-hit
        assert receiver.injected_pages == 3 + 1  # only the tail re-injected (A4)

    @pytest.mark.parametrize("frame_count", [0, 1])
    def test_incomplete_adopt_does_not_publish_cache(self, frame_count):
        source = _filled_pool()
        pool = PagedKVPool(2, 16, PAGE, 2, 8)
        cache = RadixKVCache(num_pages=16, page_size=PAGE)
        receiver = RemoteKVReceiver(cache, pool)
        tokens = tuple(range(9))  # two full pages plus one tail page
        frames = tuple(extract_page(source, page) for page in range(frame_count))

        with pytest.raises(KVTransportError, match="expected 3 page frames"):
            receiver.adopt(frames, SequenceMeta(token_ids=tokens, first_token=42))

        allocation = cache.allocate(tokens)
        assert allocation.num_cached_tokens == 0
        cache.release_preempted(allocation)

    def test_failed_injection_does_not_publish_partial_cache(self):
        source = _filled_pool()
        pool = PagedKVPool(2, 16, PAGE, 2, 8)
        cache = RadixKVCache(num_pages=16, page_size=PAGE)
        receiver = RemoteKVReceiver(cache, pool)
        tokens = tuple(range(5))  # one full page plus one tail page
        good = extract_page(source, 0)
        bad_source = extract_page(source, 1)
        bad = PageFrame(
            page_id=bad_source.page_id,
            fragments=(b"short",) + bad_source.fragments[1:],
        )

        with pytest.raises(KVTransportError):
            receiver.adopt((good, bad), SequenceMeta(token_ids=tokens, first_token=42))

        allocation = cache.allocate(tokens)
        assert allocation.num_cached_tokens == 0
        cache.release_preempted(allocation)

    def test_event_sink_failure_does_not_publish_cache(self):
        source = _filled_pool()
        pool = PagedKVPool(2, 16, PAGE, 2, 8)

        def fail_event_sink(_event):
            raise RuntimeError("event sink failed")

        cache = RadixKVCache(
            num_pages=16, page_size=PAGE, event_sink=fail_event_sink
        )
        receiver = RemoteKVReceiver(cache, pool)
        tokens = tuple(range(5))  # one full page plus one tail page
        frames = tuple(extract_page(source, page) for page in (0, 1))

        with pytest.raises(RuntimeError, match="event sink failed"):
            receiver.adopt(frames, SequenceMeta(token_ids=tokens, first_token=42))

        allocation = cache.allocate(tokens)
        assert allocation.num_cached_tokens == 0
        cache.release_preempted(allocation)

    def test_empty_token_metadata_is_rejected_before_allocation(self):
        pool = PagedKVPool(2, 16, PAGE, 2, 8)
        cache = RadixKVCache(num_pages=16, page_size=PAGE)
        before = cache.num_free_pages

        with pytest.raises(KVTransportError, match="non-empty token_ids"):
            RemoteKVReceiver(cache, pool).adopt(
                (), SequenceMeta(token_ids=(), first_token=None)
            )

        assert cache.num_free_pages == before

    def test_excess_frames_are_rejected_before_allocation(self, monkeypatch):
        source = _filled_pool()
        decode_pool = PagedKVPool(2, 16, PAGE, 2, 8)
        decode_cache = RadixKVCache(num_pages=16, page_size=PAGE)
        receiver = RemoteKVReceiver(decode_cache, decode_pool)
        before = decode_cache.num_free_pages
        allocation_calls = 0
        real_allocate = decode_cache.allocate

        def recording_allocate(tokens):
            nonlocal allocation_calls
            allocation_calls += 1
            return real_allocate(tokens)

        monkeypatch.setattr(decode_cache, "allocate", recording_allocate)
        # A two-token prompt needs one page, but the sender provides three.
        too_many = tuple(extract_page(source, page) for page in (0, 1, 2))
        with pytest.raises(KVTransportError, match="expected 1 page frames"):
            receiver.adopt(too_many, SequenceMeta(token_ids=(1, 2), first_token=0))
        assert allocation_calls == 0
        assert decode_cache.num_free_pages == before

    def test_missing_pages_is_a_handoff_error(self):
        prefill_transport, decode_transport = self._pair()
        handoff = RemoteKVHandoff(
            prefill_transport, "decode", _filled_pool(),
            RemoteKVReceiver(RadixKVCache(num_pages=8, page_size=PAGE),
                             PagedKVPool(2, 8, PAGE, 2, 8)),
            decode_transport, "prefill",
        )
        with pytest.raises(KVHandoffError, match="page ids"):
            handoff.transfer((1, 2, 3), 0, pages=())


class TestStreamCopy:
    def test_transfer_runs_inside_stream_window(self):
        events = []
        provider = CpuNoopStream()
        provider.events = events

        class _Inner:
            def transfer(self, tokens, first_token, pages=()):
                events.append("copy")
                return "allocation"

        wrapped = StreamCopyKVHandoff(_Inner(), provider)
        assert wrapped.transfer((1,), 0, (0,)) == "allocation"
        assert events == ["begin", "copy", "synchronize"]

    def test_synchronize_runs_even_on_failure(self):
        provider = CpuNoopStream()

        class _Boom:
            def transfer(self, tokens, first_token, pages=()):
                raise KVHandoffError("copy died")

        wrapped = StreamCopyKVHandoff(_Boom(), provider)
        with pytest.raises(KVHandoffError):
            wrapped.transfer((1,), 0, (0,))
        assert provider.events == ["begin", "synchronize"]


class TestNixlContract:
    @pytest.fixture()
    def fake_nixl(self, monkeypatch):
        module = types.ModuleType("nixl")

        class _Agent:
            def __init__(self, name):
                self.name = name
                self.registered = []
                self.sends = []

            def register_memory(self, num_pages):
                self.registered.append(num_pages)

            def post_send(self, dst, descriptors, token_ids):
                self.sends.append((dst, descriptors, tuple(token_ids)))
                return len(self.sends)

            def is_complete(self, handle):
                return True

            def wait_recv(self, src):
                return (), (), None

        module.Agent = _Agent
        monkeypatch.setitem(sys.modules, "nixl", module)
        return module

    def test_register_once_and_descriptor_math(self, fake_nixl):
        import asyncio

        from kairyu.engine.core.kv_transport import SequenceMeta
        from kairyu.engine.core.kv_transport_nixl_gpu import NixlTransport

        transport = NixlTransport("prefill-0")
        transport.register(64)
        with pytest.raises(KVTransportError, match="already registered"):
            transport.register(64)

        frame = PageFrame(page_id=7, fragments=(b"abcd", b""))
        asyncio.run(transport.send("decode-0", (frame,), SequenceMeta((1, 2))))
        _, descriptors, token_ids = transport._agent.sends[0]
        assert descriptors == [
            {"page_id": 7, "fragment_index": 0, "length": 4},
            {"page_id": 7, "fragment_index": 1, "length": 0},
        ]
        assert token_ids == (1, 2)

    def test_send_before_register_fails(self, fake_nixl):
        import asyncio

        from kairyu.engine.core.kv_transport import SequenceMeta
        from kairyu.engine.core.kv_transport_nixl_gpu import NixlTransport

        transport = NixlTransport("prefill-0")
        with pytest.raises(KVTransportError, match="register"):
            asyncio.run(transport.send("d", (PageFrame(0, (b"x",)),), SequenceMeta((1,))))
