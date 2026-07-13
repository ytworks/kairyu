"""m10b gates: prefix trie routing, radix KV events, event index staleness,
zmq pub/sub chaos, offline weight tuning."""

import pytest

from kairyu.engine.backend import GenerationRequest, GenerationResult, SamplingParams
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.orchestration.kv_index import KvEventIndex
from kairyu.orchestration.learning.dataset import PlacementRecord, tune_prefix_weights
from kairyu.orchestration.prefix_index import PrefixIndex, prompt_chunks
from kairyu.orchestration.replica import ReplicaPool


class MockBackend:
    async def generate(self, request):
        return GenerationResult(request_id="r", prompt="p", completions=(), finished=True)

    async def stream(self, request):
        yield GenerationResult(request_id="r", prompt="p", completions=(), finished=True)

    async def shutdown(self) -> None:
        return None


def _request(prompt: str) -> GenerationRequest:
    return GenerationRequest(
        request_id="req", prompt=prompt, sampling_params=SamplingParams()
    )


class TestPrefixIndex:
    def test_prefix_chained_chunks(self):
        keys_short = prompt_chunks("a" * 512, chunk_chars=256)
        keys_long = prompt_chunks("a" * 512 + "b" * 256, chunk_chars=256)
        assert len(keys_short) == 2
        assert keys_long[:2] == keys_short  # shared prefix -> shared keys

    def test_incremental_hashing_matches_whole_prefix_hash(self):
        # P5: the streaming sha256 chain must be byte-identical to hashing each
        # whole prefix (incl. multibyte chars split across chunk boundaries).
        import hashlib

        prompt = "こんにちは world 日本語" * 40
        for cc in (1, 3, 256):
            expected = tuple(
                hashlib.sha256(prompt[:end].encode()).hexdigest()[:16]
                for end in range(cc, len(prompt) + 1, cc)
            )
            assert prompt_chunks(prompt, chunk_chars=cc) == expected

    def test_overlap_stops_at_first_miss(self):
        index = PrefixIndex(chunk_chars=4)
        index.observe("r1", "aaaabbbbcccc")
        assert index.overlap("r1", "aaaabbbbcccc") == 3
        assert index.overlap("r1", "aaaabbbbXXXX") == 2
        assert index.overlap("r1", "XXXXbbbbcccc") == 0
        assert index.overlap("ghost", "aaaa") == 0

    def test_lru_cap(self):
        index = PrefixIndex(chunk_chars=1, max_chunks_per_replica=4)
        index.observe("r1", "abcdef")  # 6 chunks -> capped to 4
        assert len(index._chunks["r1"]) == 4


class TestPrefixRouting:
    @pytest.mark.asyncio
    async def test_shared_prefix_routes_to_warm_replica(self):
        index = PrefixIndex(chunk_chars=4)
        pool = ReplicaPool(
            {"a": MockBackend(), "b": MockBackend()}, prefix_index=index
        )
        shared = "SYSTEMPROMPT" * 4
        await pool.generate(_request(shared + "-user1"))
        first_target = max(
            ("a", "b"), key=lambda rid: index.overlap(rid, shared)
        )
        for i in range(5):
            await pool.generate(_request(shared + f"-user{i + 2}"))
        counts = pool.decision_counts
        assert counts["prefix_match"] == 5  # every follow-up hit the warm replica
        assert index.overlap(first_target, shared) > 0

    @pytest.mark.asyncio
    async def test_no_overlap_falls_back_to_least_outstanding(self):
        pool = ReplicaPool(
            {"a": MockBackend(), "b": MockBackend()}, prefix_index=PrefixIndex()
        )
        await pool.generate(_request("first-ever-prompt"))
        assert pool.decision_counts["least_outstanding"] == 1

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        pool = ReplicaPool({"a": MockBackend()})
        await pool.generate(_request("x" * 600))
        assert pool.decision_counts["prefix_match"] == 0


class TestRadixKvEvents:
    def _cache_with_sink(self):
        events: list[dict] = []
        cache = RadixKVCache(num_pages=8, page_size=4, event_sink=events.append)
        return cache, events

    def test_block_stored_on_computed_transition_only(self):
        cache, events = self._cache_with_sink()
        allocation = cache.allocate(tuple(range(9)))  # 2 full + tail
        assert events == []  # allocate never emits (A13)
        cache.mark_computed(allocation)
        assert [e["type"] for e in events] == ["BlockStored"]
        assert len(events[0]["block_hashes"]) == 2
        assert events[0]["block_size"] == 4
        cache.mark_computed(allocation)  # idempotent: no double fire
        cache.commit_and_release(allocation, output_tokens=(), decode_pages=())
        assert [e["type"] for e in events] == ["BlockStored"]

    def test_decode_extension_emits_stored(self):
        cache, events = self._cache_with_sink()
        allocation = cache.allocate(tuple(range(8)))
        cache.mark_computed(allocation)
        decode_pages = [cache.allocate_private_page() for _ in range(1)]
        # Five outputs: the decode page (positions 8-11) is fully KV-written;
        # the fifth token spills into a partial tail whose KV is not yet
        # written, so the full decode page can be safely folded (C1).
        cache.commit_and_release(
            allocation, output_tokens=(100, 101, 102, 103, 104), decode_pages=tuple(decode_pages)
        )
        kinds = [e["type"] for e in events]
        assert kinds.count("BlockStored") == 2  # prefill + decode extension

    def test_eviction_emits_removed(self):
        cache, events = self._cache_with_sink()
        first = cache.allocate(tuple(range(16)))
        cache.mark_computed(first)
        cache.commit_and_release(first, output_tokens=(), decode_pages=())
        # exhaust the pool so eviction must fire (6 pages needed, 4 free)
        second = cache.allocate(tuple(range(100, 124)))
        kinds = [e["type"] for e in events]
        assert "BlockRemoved" in kinds
        removed = next(e for e in events if e["type"] == "BlockRemoved")
        stored = next(e for e in events if e["type"] == "BlockStored")
        assert removed["block_hashes"] == stored["block_hashes"]
        cache.release_preempted(second)


class TestKvEventIndex:
    def test_apply_and_overlap(self):
        clock = {"t": 0.0}
        index = KvEventIndex(now=lambda: clock["t"])
        index.apply(
            "r1",
            {
                "type": "BlockStored",
                "block_hashes": ["h1", "h2"],
                "block_size": 4,
            },
        )
        assert index.overlap("r1", ["h1", "h2", "h3"]) == 2
        index.apply("r1", {"type": "BlockRemoved", "block_hashes": ["h2"]})
        assert index.overlap("r1", ["h1", "h2"]) == 1

    @pytest.mark.parametrize(
        "event",
        [
            pytest.param([], id="decoded-array"),
            pytest.param("event", id="decoded-string"),
            pytest.param(3, id="decoded-number"),
            pytest.param(True, id="decoded-bool"),
            pytest.param(None, id="decoded-null"),
            pytest.param({}, id="missing-type"),
            pytest.param({"type": 3}, id="number-type"),
            pytest.param({"type": True}, id="bool-type"),
            pytest.param({"type": []}, id="array-type"),
            pytest.param({"type": "Mystery"}, id="unknown-type"),
            pytest.param({"type": "BlockStored"}, id="stored-missing-hashes"),
            pytest.param(
                {"type": "BlockStored", "block_hashes": None},
                id="stored-null-hashes",
            ),
            pytest.param(
                {"type": "BlockStored", "block_hashes": "h2"},
                id="stored-string-hashes",
            ),
            pytest.param(
                {"type": "BlockStored", "block_hashes": 3},
                id="stored-number-hashes",
            ),
            pytest.param(
                {"type": "BlockStored", "block_hashes": True},
                id="stored-bool-hashes",
            ),
            pytest.param(
                {"type": "BlockStored", "block_hashes": {}},
                id="stored-object-hashes",
            ),
            pytest.param({"type": "BlockRemoved"}, id="removed-missing-hashes"),
            pytest.param(
                {"type": "BlockRemoved", "block_hashes": None},
                id="removed-null-hashes",
            ),
            pytest.param(
                {"type": "BlockRemoved", "block_hashes": "h1"},
                id="removed-string-hashes",
            ),
            pytest.param(
                {"type": "BlockRemoved", "block_hashes": 3},
                id="removed-number-hashes",
            ),
            pytest.param(
                {"type": "BlockRemoved", "block_hashes": True},
                id="removed-bool-hashes",
            ),
            pytest.param(
                {"type": "BlockRemoved", "block_hashes": {}},
                id="removed-object-hashes",
            ),
            pytest.param(
                {"type": "BlockStored", "block_hashes": ["new", 3]},
                id="stored-number-hash-member",
            ),
            pytest.param(
                {"type": "BlockStored", "block_hashes": ["new", {}]},
                id="stored-object-hash-member",
            ),
            pytest.param(
                {"type": "BlockRemoved", "block_hashes": ["h1", 3]},
                id="removed-number-hash-member",
            ),
            pytest.param(
                {"type": "BlockRemoved", "block_hashes": ["h1", {}]},
                id="removed-object-hash-member",
            ),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": None},
                id="cleared-null-hashes",
            ),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": "h1"},
                id="cleared-string-hashes",
            ),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": 3},
                id="cleared-number-hashes",
            ),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": True},
                id="cleared-bool-hashes",
            ),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": {}},
                id="cleared-object-hashes",
            ),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": ["h1", 3]},
                id="cleared-number-hash-member",
            ),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": ["h1", {}]},
                id="cleared-object-hash-member",
            ),
            pytest.param(
                {"type": "BlockStored", "block_hashes": ["new"], "block_size": 0},
                id="zero-block-size",
            ),
            pytest.param(
                {
                    "type": "BlockStored",
                    "block_hashes": ["new"],
                    "block_size": -1,
                },
                id="negative-block-size",
            ),
            pytest.param(
                {
                    "type": "BlockStored",
                    "block_hashes": ["new"],
                    "block_size": True,
                },
                id="true-block-size",
            ),
            pytest.param(
                {
                    "type": "BlockStored",
                    "block_hashes": ["new"],
                    "block_size": False,
                },
                id="false-block-size",
            ),
            pytest.param(
                {
                    "type": "BlockStored",
                    "block_hashes": ["new"],
                    "block_size": 1.5,
                },
                id="float-block-size",
            ),
            pytest.param(
                {
                    "type": "BlockStored",
                    "block_hashes": ["new"],
                    "block_size": "4",
                },
                id="string-block-size",
            ),
            pytest.param(
                {
                    "type": "BlockStored",
                    "block_hashes": ["new"],
                    "block_size": None,
                },
                id="null-block-size",
            ),
        ],
    )
    def test_invalid_event_is_controlled_and_fully_atomic(self, event):
        clock = {"t": 1.0}
        index = KvEventIndex(now=lambda: clock["t"])
        index.apply("existing", {"type": "BlockStored", "block_hashes": ["h1"]})
        before = {
            replica_id: (frozenset(entry.hashes), entry.last_event)
            for replica_id, entry in index._replicas.items()
        }
        clock["t"] = 2.0

        with pytest.raises(ValueError):
            index.apply("existing", event)
        assert {
            replica_id: (frozenset(entry.hashes), entry.last_event)
            for replica_id, entry in index._replicas.items()
        } == before

        with pytest.raises(ValueError):
            index.apply("new", event)
        assert {
            replica_id: (frozenset(entry.hashes), entry.last_event)
            for replica_id, entry in index._replicas.items()
        } == before

    def test_staleness_returns_none_for_fallback(self):
        clock = {"t": 0.0}
        index = KvEventIndex(staleness_s=0.5, now=lambda: clock["t"])
        index.apply("r1", {"type": "BlockStored", "block_hashes": ["h1"]})
        assert index.overlap("r1", ["h1"]) == 1
        clock["t"] = 1.0  # publisher died: > 500 ms silence
        assert index.overlap("r1", ["h1"]) is None  # graceful fallback signal
        index.heartbeat("r1")
        assert index.overlap("r1", ["h1"]) == 1

    def test_unknown_event_rejected(self):
        index = KvEventIndex()
        with pytest.raises(ValueError, match="unknown"):
            index.apply("r1", {"type": "Mystery"})

    @pytest.mark.parametrize(
        "clear_event",
        [
            pytest.param({"type": "AllBlocksCleared"}, id="hashes-absent"),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": []},
                id="hashes-empty",
            ),
            pytest.param(
                {"type": "AllBlocksCleared", "block_hashes": ["h1"]},
                id="hashes-present",
            ),
        ],
    )
    def test_all_blocks_cleared_is_handled(self, clear_event):
        # M4: vLLM emits AllBlocksCleared on a cache reset; it must clear the
        # replica's blocks, not crash the subscriber.
        index = KvEventIndex()
        index.apply("r1", {"type": "BlockStored", "block_hashes": ["h1", "h2"]})
        index.apply("r1", clear_event)
        assert index.overlap("r1", ["h1", "h2"]) == 0

    def test_garbage_event_does_not_keep_replica_fresh(self):
        # M4: freshness must be stamped only after a valid apply — a rejected
        # event must not reset the staleness clock and mask a dead feed.
        clock = {"t": 0.0}
        index = KvEventIndex(staleness_s=0.5, now=lambda: clock["t"])
        index.apply("r1", {"type": "BlockStored", "block_hashes": ["h1"]})
        clock["t"] = 1.0  # > 500 ms since the last VALID event
        with pytest.raises(ValueError):
            index.apply("r1", {"type": "Mystery"})  # rejected, must not refresh
        assert index.overlap("r1", ["h1"]) is None  # still stale -> trie fallback


class TestZmqTransport:
    def test_pub_sub_round_trip_and_chaos_staleness(self):
        zmq = pytest.importorskip("zmq")
        del zmq
        import time as _time

        from kairyu.orchestration.kv_index import (
            ZmqKvEventPublisher,
            ZmqKvEventSubscriber,
        )

        endpoint = "tcp://127.0.0.1:29471"
        clock = {"t": 0.0}
        index = KvEventIndex(staleness_s=0.5, now=lambda: clock["t"])
        publisher = ZmqKvEventPublisher(endpoint, replica_id="r1")
        subscriber = ZmqKvEventSubscriber([endpoint], index)
        _time.sleep(0.2)  # PUB/SUB slow-joiner
        publisher({"type": "BlockStored", "block_hashes": ["h1"]})
        deadline = _time.monotonic() + 2
        while index.overlap("r1", ["h1"]) != 1 and _time.monotonic() < deadline:
            subscriber.drain()
            _time.sleep(0.01)
        assert index.overlap("r1", ["h1"]) == 1
        # chaos: kill the publisher; staleness must flip to fallback
        publisher.close()
        clock["t"] = 1.0
        assert index.overlap("r1", ["h1"]) is None
        subscriber.close()


class TestOfflineTuning:
    def test_grid_prefers_weights_matching_good_outcomes(self):
        records = [
            PlacementRecord("a", "prefix_match", overlap_chunks=4, outstanding=1, ttft_s=0.05),
            PlacementRecord("a", "prefix_match", overlap_chunks=3, outstanding=2, ttft_s=0.06),
            PlacementRecord("b", "least_outstanding", overlap_chunks=0, outstanding=0, ttft_s=0.30),
            PlacementRecord("b", "least_outstanding", overlap_chunks=0, outstanding=1, ttft_s=0.28),
        ]
        alpha, beta = tune_prefix_weights(records)
        assert alpha > 0
        with pytest.raises(ValueError):
            tune_prefix_weights([])
