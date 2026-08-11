"""m10b gates: prefix trie routing, radix KV events, event index staleness,
zmq pub/sub chaos, offline weight tuning."""

import asyncio

import pytest

from kairyu.engine.backend import (
    CacheHint,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    UpstreamClientError,
)
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.prompt import PromptInput, TextPrompt, TokensPrompt
from kairyu.orchestration.kv_index import KvEventIndex, ZmqKvEventSubscriber
from kairyu.orchestration.prefix_index import (
    PREFIX_HASH_VERSION,
    PrefixIndex,
    prefix_root_fingerprint,
    prompt_chunks,
)
from kairyu.orchestration.replica import ReplicaPool


class MockBackend:
    async def generate(self, request):
        return GenerationResult(request_id="r", prompt="p", completions=(), finished=True)

    async def stream(self, request):
        yield GenerationResult(request_id="r", prompt="p", completions=(), finished=True)

    async def shutdown(self) -> None:
        return None


def _request(
    prompt: PromptInput,
    *,
    session_id: str | None = None,
    prefix_fingerprint: str = "",
) -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        prompt=prompt,
        sampling_params=SamplingParams(),
        cache_hint=(
            CacheHint(
                session_id=session_id,
                prefix_fingerprint=prefix_fingerprint,
            )
            if session_id is not None
            else None
        ),
    )


class TestPrefixIndex:
    def test_prefix_chained_chunks(self):
        keys_short = prompt_chunks("a" * 512, chunk_chars=256)
        keys_long = prompt_chunks("a" * 512 + "b" * 256, chunk_chars=256)
        assert len(keys_short) == 2
        assert keys_long[:2] == keys_short  # shared prefix -> shared keys

    def test_incremental_hashing_matches_whole_prefix_hash(self):
        # P5: the streaming XXH3 chain must be byte-identical to hashing each
        # whole prefix (incl. multibyte chars split across chunk boundaries).
        import xxhash

        prompt = "こんにちは world 日本語" * 40
        assert PREFIX_HASH_VERSION == "xxh3-64-v1"
        for cc in (1, 3, 256):
            expected = tuple(
                xxhash.xxh3_64(prompt[:end].encode()).hexdigest()
                for end in range(cc, len(prompt) + 1, cc)
            )
            assert prompt_chunks(prompt, chunk_chars=cc) == expected

    def test_carried_root_requires_the_exact_default_chunk_contract(self):
        prompt = "a" * 256 + "b" * 256
        exact_root = prefix_root_fingerprint(prompt)

        assert prefix_root_fingerprint("a" * 255) == ""
        assert exact_root == prompt_chunks(prompt)[0]
        assert PrefixIndex().prepare(prompt, "not-a-root") == exact_root
        assert PrefixIndex(chunk_chars=4).prepare(prompt, exact_root) == (
            prompt_chunks(prompt, chunk_chars=4)[0]
        )

    def test_overlap_stops_at_first_miss(self):
        index = PrefixIndex(chunk_chars=4)
        index.observe("r1", "aaaabbbbcccc")
        assert index.overlap("r1", "aaaabbbbcccc") == 3
        assert index.overlap("r1", "aaaabbbbXXXX") == 2
        assert index.overlap("r1", "XXXXbbbbcccc") == 0
        assert index.overlap("ghost", "aaaa") == 0

    @pytest.mark.parametrize(
        ("replica_id", "prompt"),
        [
            pytest.param("r1", "aaaabbbb", id="warm-prefix"),
            pytest.param("r1", "zzzz", id="cold-prefix"),
            pytest.param("ghost", "aaaabbbb", id="missing-replica"),
            pytest.param("r1", "aaaabbbbcccc", id="full-prefix"),
            pytest.param("r1", "XXXXbbbbcccc", id="first-miss"),
        ],
    )
    def test_overlap_keys_matches_prompt_overlap(self, replica_id, prompt):
        index = PrefixIndex(chunk_chars=4)
        index.observe("r1", "aaaabbbbcccc")
        keys = index.chunk_keys(prompt)
        before = tuple(keys)

        assert index.overlap_keys(replica_id, keys) == index.overlap(replica_id, prompt)
        assert keys == before  # caller-owned immutable key sequence is read-only

    def test_lru_cap(self):
        index = PrefixIndex(chunk_chars=1, max_chunks_per_replica=4)
        index.observe("r1", "abcdef")  # 6 chunks -> capped to 4
        assert len(index._chunks["r1"]) == 4
        assert tuple(index._chunks["r1"]) == prompt_chunks("abcdef", chunk_chars=1)[:4]
        assert index.overlap("r1", "abcdef") == 4

    @pytest.mark.parametrize("limit", [0, -1])
    def test_chunk_cap_must_be_positive(self, limit):
        with pytest.raises(ValueError, match="max_chunks_per_replica must be >= 1"):
            PrefixIndex(max_chunks_per_replica=limit)

    def test_candidate_reverse_map_tracks_eviction_and_forget(self):
        index = PrefixIndex(chunk_chars=1, max_chunks_per_replica=2)
        first_keys = index.chunk_keys("ab")
        replacement_keys = index.chunk_keys("cd")
        assert index.candidate_ids(()) == ()

        index.observe("r1", "ab")
        index.observe("r2", "ab")
        assert index.candidate_ids(first_keys) == ("r1", "r2")

        index.observe("r1", "cd")
        assert index.candidate_ids(first_keys) == ("r2",)
        assert index.candidate_ids(replacement_keys) == ("r1",)

        index.forget_replica("r2")
        index.forget_replica("unknown")
        assert index.candidate_ids(first_keys) == ()
        assert index.candidate_ids(replacement_keys) == ("r1",)

        index.forget_replica("r1")
        assert index.candidate_ids(replacement_keys) == ()
        assert index._replicas_by_first_key == {}

    def test_concurrent_cold_preparations_merge_completed_candidates(self):
        index = PrefixIndex(chunk_chars=4)
        first = index.prepare("aaaabbbb")
        second = index.prepare("aaaabbbb")
        assert isinstance(first, str)
        assert first == second

        index.observe_prepared("r0", first, False)
        index.observe_prepared("r1", second, False)

        assert index.candidate_ids((first,)) == ("r0", "r1")


class TestPrefixRouting:
    def test_native_index_preallocates_dynamic_replica_stores(self):
        index = PrefixIndex(chunk_chars=4)
        pool = ReplicaPool(
            {"r0": MockBackend(), "r1": MockBackend()},
            prefix_index=index,
        )

        assert set(index._chunks) == {"r0", "r1"}
        assert all(not store for store in index._chunks.values())

        pool.add_replica("r2", MockBackend())

        assert set(index._chunks) == {"r0", "r1", "r2"}
        assert index._chunks["r2"] == {}

    def test_cold_selection_skips_500_overlap_scans_and_runs_one_hrw(
        self, monkeypatch
    ):
        import kairyu.orchestration.replica as replica_module

        class SpyIndex(PrefixIndex):
            def __init__(self):
                super().__init__(chunk_chars=32)
                self.overlap_key_calls: list[str] = []

            def overlap_keys(self, replica_id, keys):
                self.overlap_key_calls.append(replica_id)
                return super().overlap_keys(replica_id, keys)

        index = SpyIndex()
        replica_ids = tuple(f"r{i}" for i in range(500))
        pool = ReplicaPool(
            {replica_id: MockBackend() for replica_id in replica_ids},
            prefix_index=index,
        )
        real_hrw = replica_module._rendezvous_winner
        hrw_calls = 0

        def counted_hrw(session_id, candidates, encoded):
            nonlocal hrw_calls
            hrw_calls += 1
            return real_hrw(session_id, candidates, encoded)

        monkeypatch.setattr(replica_module, "_rendezvous_winner", counted_hrw)

        selected, reason, session_id = pool._select(
            _request("long prompt " * 512, session_id="cold-session")
        )

        assert selected in replica_ids
        assert reason == "session_affinity"
        assert session_id == "cold-session"
        assert index.overlap_key_calls == []
        assert hrw_calls == 1

    @pytest.mark.asyncio
    async def test_successful_cold_then_warm_hashes_each_chunk_at_most_once(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        import kairyu.orchestration.prefix_index as prefix_module

        real_xxh3 = prefix_module.xxhash.xxh3_64
        hash_instances = 0
        hashed_parts: list[bytes] = []

        class CountingHash:
            def __init__(self, payload=b""):
                self._inner = real_xxh3(payload)
                if payload:
                    hashed_parts.append(payload)

            def update(self, payload):
                hashed_parts.append(payload)
                return self._inner.update(payload)

            def hexdigest(self):
                return self._inner.hexdigest()

        def counted_xxh3(payload=b""):
            nonlocal hash_instances
            hash_instances += 1
            return CountingHash(payload)

        monkeypatch.setattr(
            prefix_module,
            "xxhash",
            SimpleNamespace(xxh3_64=counted_xxh3),
        )

        index = PrefixIndex(chunk_chars=4)
        replica_ids = ("r0", "r1")
        pool = ReplicaPool(
            {replica_id: MockBackend() for replica_id in replica_ids},
            prefix_index=index,
        )
        prompt = "aaaabbbbccccddddeeee"
        root_key = real_xxh3(b"aaaa").hexdigest()

        await pool.generate(_request(prompt))

        # Cold placement and success publish only the reusable root: one XXH3
        # object and one chunk update, not two full five-chunk passes.
        assert hash_instances == 1
        assert hashed_parts == [b"aaaa"]
        selected = index.candidate_ids((root_key,))
        assert len(selected) == 1
        assert tuple(index._chunks[selected[0]]) == (root_key,)

        hash_instances = 0
        hashed_parts.clear()
        await pool.generate(_request(prompt))

        # The root candidate makes this a warm route. Selection lazily extends
        # the same request-local hash chain; successful promotion consumes it,
        # so every prompt chunk is hashed exactly once overall.
        assert pool.decision_counts["prefix_match"] == 1
        assert hash_instances == 1
        assert hashed_parts == [b"aaaa", b"bbbb", b"cccc", b"dddd", b"eeee"]
        assert len(index._chunks[selected[0]]) == 5

    @pytest.mark.asyncio
    async def test_trusted_prefix_fingerprint_interoperates_with_local_fallback(self):
        index = PrefixIndex()
        pool = ReplicaPool(
            {"r0": MockBackend(), "r1": MockBackend()},
            prefix_index=index,
        )
        prompt = "a" * 256 + "b" * 256
        fingerprint = prefix_root_fingerprint(prompt)

        await pool.generate(
            _request(
                prompt,
                session_id="cold-session",
                prefix_fingerprint=fingerprint,
            )
        )

        selected = index.candidate_ids((fingerprint,))
        assert len(selected) == 1
        assert tuple(index._chunks[selected[0]]) == (fingerprint,)

        await pool.generate(
            _request(
                "a" * 256 + "c" * 256,
                session_id="warm-session",
                prefix_fingerprint="not-a-root",
            )
        )

        assert pool.decision_counts["prefix_match"] == 1

    @pytest.mark.asyncio
    async def test_prefix_index_subclass_keeps_chunk_and_observe_overrides(self):
        class CustomIndex(PrefixIndex):
            def __init__(self):
                super().__init__(chunk_chars=4)
                self.chunk_calls: list[str] = []
                self.observe_calls: list[tuple[str, str]] = []

            def chunk_keys(self, prompt):
                self.chunk_calls.append(prompt)
                return (f"custom:{prompt[:4]}",)

            def observe(self, replica_id, prompt):
                self.observe_calls.append((replica_id, prompt))
                self.observe_keys(replica_id, self.chunk_keys(prompt))

        index = CustomIndex()
        index.observe("r0", "shared-seed")
        index.chunk_calls.clear()
        index.observe_calls.clear()
        pool = ReplicaPool(
            {"r0": MockBackend(), "r1": MockBackend()},
            prefix_index=index,
        )

        await pool.generate(_request("shared-request", session_id="new-session"))

        assert pool.decision_counts["prefix_match"] == 1
        assert index.chunk_calls == ["shared-request", "shared-request"]
        assert index.observe_calls == [("r0", "shared-request")]

    def test_warm_selection_scores_only_first_key_candidates(self):
        class SpyIndex(PrefixIndex):
            def __init__(self):
                super().__init__(chunk_chars=4)
                self.overlap_key_calls: list[str] = []

            def overlap_keys(self, replica_id, keys):
                self.overlap_key_calls.append(replica_id)
                return super().overlap_keys(replica_id, keys)

        index = SpyIndex()
        replica_ids = tuple(f"r{i}" for i in range(32))
        pool = ReplicaPool(
            {replica_id: MockBackend() for replica_id in replica_ids},
            prefix_index=index,
        )
        prompt = "shared-prefix"
        index.observe("r19", prompt)
        index.observe("r7", prompt)

        assert pool._prefix_select(replica_ids, prompt) == "r7"
        assert index.overlap_key_calls == ["r7", "r19"]

    def test_prefix_selection_preserves_legacy_overlap_only_index(self):
        class LegacyIndex:
            def __init__(self):
                self.calls: list[tuple[str, str]] = []

            def overlap(self, replica_id, prompt):
                self.calls.append((replica_id, prompt))
                return {"a": 1, "b": 2}[replica_id]

        index = LegacyIndex()
        pool = ReplicaPool(
            {"a": MockBackend(), "b": MockBackend()}, prefix_index=index
        )

        assert pool._prefix_select(("a", "b"), "prompt") == "b"
        assert index.calls == [("a", "prompt"), ("b", "prompt")]

    def test_prefix_selection_does_not_mask_advertised_key_api_errors(self):
        class BrokenIndex:
            def chunk_keys(self, _prompt):
                raise RuntimeError("key generation failed")

            def overlap_keys(self, _replica_id, _keys):  # pragma: no cover
                return 0

        pool = ReplicaPool({"a": MockBackend()}, prefix_index=BrokenIndex())

        with pytest.raises(RuntimeError, match="key generation failed"):
            pool._prefix_select(("a",), "prompt")

    def test_prefix_selection_keeps_first_candidate_on_score_tie(self):
        class TiedIndex:
            def chunk_keys(self, _prompt):
                return ("key",)

            def overlap_keys(self, _replica_id, _keys):
                return 2

        pool = ReplicaPool(
            {"a": MockBackend(), "b": MockBackend()}, prefix_index=TiedIndex()
        )

        assert pool._prefix_select(("a", "b"), "prompt") == "a"

    def test_prefix_selection_score_tie_uses_session_hrw(self):
        index = PrefixIndex(chunk_chars=4)
        replicas = {"a": MockBackend(), "b": MockBackend(), "c": MockBackend()}
        pool = ReplicaPool(replicas, prefix_index=index)
        baseline = ReplicaPool(replicas)
        prompt = "shared-prefix"
        for replica_id in replicas:
            index.observe(replica_id, prompt)
        session_id = next(
            f"session-{candidate}"
            for candidate in range(100)
            if baseline._select(
                _request(prompt, session_id=f"session-{candidate}")
            )[0]
            != "a"
        )
        expected = baseline._select(_request(prompt, session_id=session_id))[0]

        selected, reason, observed_session = pool._select(
            _request(
                prompt,
                session_id=session_id,
                prefix_fingerprint="0" * 16,
            )
        )

        assert selected == expected
        assert reason == "prefix_match"
        assert observed_session == session_id

    def test_warm_candidate_wins_zero_score_tie_with_cold_baseline(self):
        index = PrefixIndex(chunk_chars=4)
        replicas = {"a": MockBackend(), "b": MockBackend(), "c": MockBackend()}
        pool = ReplicaPool(
            replicas,
            prefix_index=index,
            prefix_beta=0.25,
            queue_depth_threshold=1_000,
        )
        baseline = ReplicaPool(replicas)
        prompt = "shared-prefix"
        index.observe("a", prompt)
        pool._entries["a"].outstanding = index.overlap("a", prompt) * 4
        session_id = next(
            f"session-{candidate}"
            for candidate in range(100)
            if baseline._select(
                _request(prompt, session_id=f"session-{candidate}")
            )[0]
            != "a"
        )

        selected, reason, _observed_session = pool._select(
            _request(
                prompt,
                session_id=session_id,
                prefix_fingerprint="0" * 16,
            )
        )

        assert selected == "a"
        assert reason == "prefix_match"

    def test_negative_warm_score_uses_cold_session_fallback(self):
        index = PrefixIndex(chunk_chars=4)
        replicas = {"a": MockBackend(), "b": MockBackend(), "c": MockBackend()}
        pool = ReplicaPool(replicas, prefix_index=index, prefix_beta=0.25)
        baseline = ReplicaPool(replicas)
        prompt = "shared-prefix"
        index.observe("a", prompt)
        pool._entries["a"].outstanding = index.overlap("a", prompt) * 4 + 1
        session_id = next(
            f"session-{candidate}"
            for candidate in range(100)
            if baseline._select(
                _request(prompt, session_id=f"session-{candidate}")
            )[0]
            != "a"
        )
        request = _request(
            prompt,
            session_id=session_id,
            prefix_fingerprint="0" * 16,
        )

        assert pool._select(request) == baseline._select(request)

    def test_hinted_request_prefers_warm_prefix_over_session_affinity(self):
        index = PrefixIndex(chunk_chars=4)
        replicas = {"a": MockBackend(), "b": MockBackend(), "c": MockBackend()}
        pool = ReplicaPool(replicas, prefix_index=index)
        request = _request(
            "shared-prefix-user",
            session_id="sticky-session",
            prefix_fingerprint="0" * 16,
        )
        affine, reason, _session_id = pool._select(request)
        assert reason == "session_affinity"
        warm = next(replica_id for replica_id in replicas if replica_id != affine)
        index.observe(warm, request.prompt)

        selected, reason, observed_session = pool._select(request)

        assert selected == warm
        assert reason == "prefix_match"
        assert observed_session == "sticky-session"

    def test_hinted_request_without_overlap_preserves_session_affinity(self):
        replicas = {"a": MockBackend(), "b": MockBackend(), "c": MockBackend()}
        request = _request(
            "cold-prompt",
            session_id="sticky-session",
            prefix_fingerprint="0" * 16,
        )
        baseline = ReplicaPool(replicas)
        pool = ReplicaPool(replicas, prefix_index=PrefixIndex(chunk_chars=4))

        assert pool._select(request) == baseline._select(request)

    def test_hinted_cold_request_preserves_queue_depth_fallback(self):
        replicas = {"a": MockBackend(), "b": MockBackend(), "c": MockBackend()}
        request = _request(
            "cold-prompt",
            session_id="sticky-session",
            prefix_fingerprint="0" * 16,
        )
        baseline = ReplicaPool(replicas, queue_depth_threshold=0)
        pool = ReplicaPool(
            replicas,
            prefix_index=PrefixIndex(chunk_chars=4),
            queue_depth_threshold=0,
        )
        affine = baseline._select(request)[0]
        baseline._entries[affine].outstanding = 1
        pool._entries[affine].outstanding = 1

        assert pool._select(request) == baseline._select(request)
        assert pool._select(request)[1] == "queue_depth_fallback"

    def test_busy_warm_prefix_obeys_queue_depth_fallback(self):
        index = PrefixIndex(chunk_chars=4)
        replicas = {"a": MockBackend(), "b": MockBackend(), "c": MockBackend()}
        pool = ReplicaPool(
            replicas,
            prefix_index=index,
            queue_depth_threshold=0,
        )
        request = _request(
            "shared-prefix-user",
            session_id="sticky-session",
            prefix_fingerprint="0" * 16,
        )
        index.observe("a", request.prompt)
        pool._entries["a"].outstanding = 1

        selected, reason, observed_session = pool._select(request)

        assert selected == "b"
        assert reason == "queue_depth_fallback"
        assert observed_session == "sticky-session"

    def test_prefix_selection_keeps_queue_depth_penalty_semantics(self):
        class ScoredIndex:
            def chunk_keys(self, _prompt):
                return ("key",)

            def overlap_keys(self, replica_id, _keys):
                return {"a": 3, "b": 2}[replica_id]

        pool = ReplicaPool(
            {"a": MockBackend(), "b": MockBackend()},
            prefix_index=ScoredIndex(),
            prefix_beta=0.25,
        )
        pool._entries["a"].outstanding = 8

        assert pool._prefix_select(("a", "b"), "prompt") == "b"

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
    async def test_blank_session_hint_bypasses_native_prefix_tracking(
        self, monkeypatch
    ):
        index = PrefixIndex()
        replicas = {"a": MockBackend(), "b": MockBackend()}
        pool = ReplicaPool(
            replicas,
            prefix_index=index,
        )
        baseline = ReplicaPool(replicas)
        request = _request("x" * 512, session_id="session-only")

        def unexpected_prefix_work(*_args, **_kwargs):
            pytest.fail("blank native CacheHint performed prefix work")

        monkeypatch.setattr(pool, "_prefix_prepare", unexpected_prefix_work)
        monkeypatch.setattr(pool, "_prefix_select", unexpected_prefix_work)
        monkeypatch.setattr(pool, "_prefix_observe_root", unexpected_prefix_work)
        monkeypatch.setattr(
            pool,
            "_prefix_observe_prepared",
            unexpected_prefix_work,
        )
        monkeypatch.setattr(index, "observe", unexpected_prefix_work)

        assert pool._select(request) == baseline._select(request)
        await pool.generate(request)
        chunks = [chunk async for chunk in pool.stream(request)]

        assert len(chunks) == 1
        assert pool.decision_counts["session_affinity"] == 2
        assert pool.decision_counts["prefix_match"] == 0
        assert index._replicas_by_first_key == {}
        assert all(not store for store in index._chunks.values())

    @pytest.mark.asyncio
    async def test_token_prompt_bypasses_text_prefix_index_but_keeps_session_affinity(
        self, monkeypatch
    ):
        class TokenBackend(MockBackend):
            def validate_request(self, _request):
                return None

        index = PrefixIndex(chunk_chars=1)
        replicas = {"a": TokenBackend(), "b": TokenBackend()}
        pool = ReplicaPool(replicas, prefix_index=index)
        baseline = ReplicaPool(replicas)
        request = _request(TokensPrompt((1, 2, 3)), session_id="token-session")

        def unexpected_text_prefix_work(*_args, **_kwargs):
            pytest.fail("token prompt entered the text PrefixIndex")

        monkeypatch.setattr(pool, "_prefix_prepare", unexpected_text_prefix_work)
        monkeypatch.setattr(pool, "_prefix_select", unexpected_text_prefix_work)
        monkeypatch.setattr(pool, "_prefix_observe_root", unexpected_text_prefix_work)
        monkeypatch.setattr(
            pool,
            "_prefix_observe_prepared",
            unexpected_text_prefix_work,
        )
        monkeypatch.setattr(index, "observe", unexpected_text_prefix_work)

        assert pool._select(request) == baseline._select(request)
        await pool.generate(request)
        chunks = [chunk async for chunk in pool.stream(request)]

        assert len(chunks) == 1
        assert pool.decision_counts["session_affinity"] == 2
        assert pool.decision_counts["prefix_match"] == 0
        assert index._replicas_by_first_key == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prompt",
        [TextPrompt("typed text"), TokensPrompt((1, 2, 3))],
    )
    async def test_typed_prompt_fails_closed_for_legacy_pool_members(
        self, prompt
    ):
        pool = ReplicaPool({"legacy": MockBackend()})
        request = _request(prompt)

        with pytest.raises(ValueError, match="text-only"):
            pool.validate_request(request)
        with pytest.raises(ValueError, match="text-only"):
            await pool.generate(request)
        with pytest.raises(ValueError, match="text-only"):
            async for _ in pool.stream(request):
                pass

    @pytest.mark.asyncio
    async def test_pool_direct_dispatch_revalidates_declared_backend_contract(self):
        class RejectingTokenBackend(MockBackend):
            def __init__(self):
                self.calls = 0

            def validate_request(self, request):
                if isinstance(request.prompt, TokensPrompt):
                    raise ValueError("token prompts are disabled")

            async def generate(self, request):
                self.calls += 1
                return await super().generate(request)

            async def stream(self, request):
                self.calls += 1
                async for chunk in super().stream(request):
                    yield chunk

        backend = RejectingTokenBackend()
        pool = ReplicaPool({"declared": backend})
        request = _request(TokensPrompt((1, 2, 3)))

        with pytest.raises(ValueError, match="token prompts are disabled"):
            await pool.generate(request)
        with pytest.raises(ValueError, match="token prompts are disabled"):
            async for _ in pool.stream(request):
                pass

        assert backend.calls == 0

    @pytest.mark.asyncio
    async def test_explicit_text_prompt_uses_the_string_prefix_domain(self):
        class TypedTextBackend(MockBackend):
            def validate_request(self, _request):
                return None

        index = PrefixIndex(chunk_chars=4)
        pool = ReplicaPool(
            {"a": TypedTextBackend(), "b": TypedTextBackend()},
            prefix_index=index,
        )
        request = _request(TextPrompt("shared-prefix"))

        await pool.generate(request)
        chunks = [chunk async for chunk in pool.stream(request)]

        assert len(chunks) == 1
        assert pool.decision_counts["prefix_match"] == 1
        assert index.candidate_ids(index.chunk_keys("shared-prefix")) == ("a",)

    @pytest.mark.asyncio
    async def test_only_successful_generate_advertises_the_prefix(self):
        class FailingBackend(MockBackend):
            async def generate(self, request):
                raise RuntimeError("backend failed")

        class ClientErrorBackend(MockBackend):
            async def generate(self, request):
                raise UpstreamClientError("bad request", status_code=400)

        prompt = "successful-prefix"
        failed_index = PrefixIndex(chunk_chars=4)
        failed_pool = ReplicaPool(
            {"failed": FailingBackend()},
            prefix_index=failed_index,
        )
        with pytest.raises(RuntimeError, match="backend failed"):
            await failed_pool.generate(_request(prompt))
        assert failed_index.overlap("failed", prompt) == 0

        client_error_index = PrefixIndex(chunk_chars=4)
        client_error_pool = ReplicaPool(
            {"client-error": ClientErrorBackend()},
            prefix_index=client_error_index,
        )
        with pytest.raises(UpstreamClientError, match="bad request"):
            await client_error_pool.generate(_request(prompt))
        assert client_error_index.overlap("client-error", prompt) == 0

        successful_index = PrefixIndex(chunk_chars=4)
        successful_pool = ReplicaPool(
            {"successful": MockBackend()},
            prefix_index=successful_index,
        )
        await successful_pool.generate(_request(prompt))
        assert successful_index.overlap("successful", prompt) > 0

    @pytest.mark.asyncio
    async def test_force_removed_generation_cannot_republish_prefix_after_readd(self):
        class DelayedBackend(MockBackend):
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def generate(self, request):
                self.started.set()
                await self.release.wait()
                return GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(),
                    finished=True,
                )

        backend = DelayedBackend()
        index = PrefixIndex(chunk_chars=4)
        pool = ReplicaPool({"same-id": backend}, prefix_index=index)
        prompt = "old-generation-prefix"
        task = asyncio.create_task(pool.generate(_request(prompt)))
        await backend.started.wait()

        await pool.remove_replica("same-id", force=True)
        pool.add_replica("same-id", MockBackend())
        backend.release.set()
        await task

        keys = index.chunk_keys(prompt)
        assert index.candidate_ids(keys) == ()
        assert index.overlap("same-id", prompt) == 0

        await pool.generate(_request(prompt))
        assert index.candidate_ids(keys) == ("same-id",)
        assert index.overlap("same-id", prompt) > 0

    @pytest.mark.asyncio
    async def test_cancelled_stream_does_not_advertise_the_prefix(self):
        class BlockingStreamBackend(MockBackend):
            def __init__(self):
                self.started = asyncio.Event()
                self.never = asyncio.Event()

            async def stream(self, request):
                self.started.set()
                await self.never.wait()
                yield GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(),
                    finished=True,
                )

        backend = BlockingStreamBackend()
        index = PrefixIndex(chunk_chars=4)
        pool = ReplicaPool({"stream": backend}, prefix_index=index)
        request = _request("cancelled-prefix")

        async def consume() -> None:
            async for _chunk in pool.stream(request):
                pass

        task = asyncio.create_task(consume())
        await backend.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert index.overlap("stream", request.prompt) == 0

    @pytest.mark.asyncio
    async def test_first_stream_chunk_advertises_root_before_completion(self):
        class DecodingBackend(MockBackend):
            def __init__(self):
                self.never = asyncio.Event()

            async def stream(self, request):
                yield GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(),
                    finished=False,
                )
                await self.never.wait()

        index = PrefixIndex(chunk_chars=4)
        pool = ReplicaPool(
            {"a": DecodingBackend(), "b": MockBackend()},
            prefix_index=index,
        )
        prompt = "shared-prefix"
        stream = pool.stream(_request(prompt + "-first"))

        await anext(stream)

        keys = index.chunk_keys(prompt)
        assert index.candidate_ids(keys) == ("a",)
        await pool.generate(_request(prompt + "-second"))
        assert pool.decision_counts["prefix_match"] == 1

        await stream.aclose()
        assert index.candidate_ids(keys) == ("a",)

    @pytest.mark.asyncio
    async def test_stream_failure_before_first_chunk_does_not_advertise(self):
        class FailingStreamBackend(MockBackend):
            async def stream(self, request):
                raise RuntimeError("stream failed before output")
                yield  # pragma: no cover

        index = PrefixIndex(chunk_chars=4)
        pool = ReplicaPool(
            {"stream": FailingStreamBackend()},
            prefix_index=index,
        )
        request = _request("failed-stream-prefix")

        with pytest.raises(RuntimeError, match="stream failed before output"):
            async for _chunk in pool.stream(request):
                pass

        assert index.overlap("stream", request.prompt) == 0

    @pytest.mark.asyncio
    async def test_warm_stream_promotes_root_only_after_completion(self):
        class PausedBackend(MockBackend):
            def __init__(self):
                self.release = asyncio.Event()

            async def stream(self, request):
                yield GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(),
                    finished=False,
                )
                await self.release.wait()

        backend = PausedBackend()
        index = PrefixIndex(chunk_chars=4)
        index.observe_root("stream", index.chunk_keys("aaaaseed")[0])
        pool = ReplicaPool({"stream": backend}, prefix_index=index)
        request = _request("aaaabbbbcccc")
        stream = pool.stream(request)

        await anext(stream)
        assert index.overlap("stream", request.prompt) == 1

        backend.release.set()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert index.overlap("stream", request.prompt) == 3

    @pytest.mark.asyncio
    async def test_warm_stream_failure_after_first_chunk_keeps_only_root(self):
        class FailingAfterFirstChunkBackend(MockBackend):
            async def stream(self, request):
                yield GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    completions=(),
                    finished=False,
                )
                raise RuntimeError("decode failed")

        index = PrefixIndex(chunk_chars=4)
        index.observe_root("stream", index.chunk_keys("aaaaseed")[0])
        pool = ReplicaPool(
            {"stream": FailingAfterFirstChunkBackend()},
            prefix_index=index,
            unhealthy_after=1,
        )
        request = _request("aaaabbbbcccc")
        stream = pool.stream(request)

        await anext(stream)
        with pytest.raises(RuntimeError, match="decode failed"):
            await anext(stream)

        assert index.overlap("stream", request.prompt) == 1
        assert pool.healthy_by_id() == {"stream": False}

    @pytest.mark.asyncio
    async def test_completed_stream_advertises_the_prefix(self):
        index = PrefixIndex(chunk_chars=4)
        pool = ReplicaPool({"stream": MockBackend()}, prefix_index=index)
        request = _request("completed-prefix")

        chunks = [chunk async for chunk in pool.stream(request)]

        assert len(chunks) == 1
        assert index.overlap("stream", request.prompt) > 0

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

    def test_reentrant_block_stored_sink_emits_once_and_publishes_cache(self):
        events: list[dict] = []
        allocation_holder = []
        attempts = 0

        def reentrant_sink(event):
            nonlocal attempts
            attempts += 1
            cache.mark_computed(allocation_holder[0])
            if attempts == 1:
                raise RuntimeError("event sink failed")
            events.append(event)

        cache = RadixKVCache(num_pages=8, page_size=4, event_sink=reentrant_sink)
        allocation = cache.allocate(tuple(range(8)))
        allocation_holder.append(allocation)

        with pytest.raises(RuntimeError, match="event sink failed"):
            cache.mark_computed(allocation)
        cache.mark_computed(allocation)

        assert attempts == 2
        assert [event["type"] for event in events] == ["BlockStored"]
        cache.free(allocation)
        cache_hit = cache.allocate(tuple(range(8)))
        assert cache_hit.num_cached_tokens == 8
        cache.release_preempted(cache_hit)

    def test_failed_free_event_is_retryable_and_releases_once(self):
        events: list[dict] = []
        attempts = 0

        def fail_once_sink(event):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("event sink failed")
            events.append(event)

        cache = RadixKVCache(num_pages=4, page_size=4, event_sink=fail_once_sink)
        allocation = cache.allocate((1, 2, 3, 4, 5))

        with pytest.raises(RuntimeError, match="event sink failed"):
            cache.free(allocation)

        assert not allocation._freed
        assert cache.num_free_pages == 2
        cache.free(allocation)

        assert allocation._freed
        assert cache.num_free_pages == 3
        assert attempts == 2
        assert [event["type"] for event in events] == ["BlockStored"]
        cache_hit = cache.allocate((1, 2, 3, 4, 5))
        assert cache_hit.num_cached_tokens == 4
        cache.release_preempted(cache_hit)

    def test_failed_mark_computed_ignores_reentrant_free_until_retry(self):
        events: list[dict] = []
        attempts = 0
        allocation_holder = []

        def fail_once_after_reentrant_free(event):
            nonlocal attempts
            attempts += 1
            cache.free(allocation_holder[0])
            if attempts == 1:
                raise RuntimeError("event sink failed")
            events.append(event)

        cache = RadixKVCache(
            num_pages=4, page_size=4, event_sink=fail_once_after_reentrant_free
        )
        allocation = cache.allocate((1, 2, 3, 4, 5))
        allocation_holder.append(allocation)

        with pytest.raises(RuntimeError, match="event sink failed"):
            cache.mark_computed(allocation)

        assert not allocation._freed
        assert cache.num_free_pages == 2
        cache.mark_computed(allocation)

        assert not allocation._freed
        assert cache.num_free_pages == 2
        cache.free(allocation)
        assert allocation._freed
        assert cache.num_free_pages == 3
        assert attempts == 2
        assert [event["type"] for event in events] == ["BlockStored"]
        cache_hit = cache.allocate((1, 2, 3, 4, 5))
        assert cache_hit.num_cached_tokens == 4
        cache.release_preempted(cache_hit)

    def test_free_ignores_reentrant_free_from_event_sink(self):
        events: list[dict] = []
        allocation_holder = []

        def reentrant_free_sink(event):
            cache.free(allocation_holder[0])
            events.append(event)

        cache = RadixKVCache(
            num_pages=4, page_size=4, event_sink=reentrant_free_sink
        )
        allocation = cache.allocate((1, 2, 3, 4, 5))
        allocation_holder.append(allocation)

        cache.free(allocation)

        assert allocation._freed
        assert cache.num_free_pages == 3
        assert [event["type"] for event in events] == ["BlockStored"]
        cache_hit = cache.allocate((1, 2, 3, 4, 5))
        assert cache_hit.num_cached_tokens == 4
        cache.release_preempted(cache_hit)

    def test_mark_computed_ignores_reentrant_preempted_release(self):
        events: list[dict] = []
        allocation_holder = []

        def reentrant_preempt_sink(event):
            cache.release_preempted(allocation_holder[0])
            events.append(event)

        cache = RadixKVCache(
            num_pages=4, page_size=4, event_sink=reentrant_preempt_sink
        )
        allocation = cache.allocate((1, 2, 3, 4, 5))
        allocation_holder.append(allocation)

        cache.mark_computed(allocation)

        assert not allocation._freed
        assert cache.num_free_pages == 2
        cache.free(allocation)
        assert allocation._freed
        assert cache.num_free_pages == 3
        assert [event["type"] for event in events] == ["BlockStored"]
        cache_hit = cache.allocate((1, 2, 3, 4, 5))
        assert cache_hit.num_cached_tokens == 4
        cache.release_preempted(cache_hit)

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

    def test_failed_decode_extension_event_rolls_back_and_retries(self):
        events: list[dict] = []
        attempts = 0
        output_tokens = (100, 101, 102, 103, 104)
        allocation_holder = []
        decode_page_holder = []

        def fail_decode_once_sink(event):
            nonlocal attempts
            attempts += 1
            if event["token_ids"] == list(output_tokens[:4]):
                cache.commit_and_release(
                    allocation_holder[0],
                    output_tokens=output_tokens,
                    decode_pages=(decode_page_holder[0],),
                )
            if attempts == 2:
                raise RuntimeError("event sink failed")
            events.append(event)

        cache = RadixKVCache(num_pages=8, page_size=4, event_sink=fail_decode_once_sink)
        allocation = cache.allocate(tuple(range(8)))
        allocation_holder.append(allocation)
        cache.mark_computed(allocation)
        decode_page = cache.allocate_private_page()
        decode_page_holder.append(decode_page)

        with pytest.raises(RuntimeError, match="event sink failed"):
            cache.commit_and_release(
                allocation, output_tokens=output_tokens, decode_pages=(decode_page,)
        )

        assert not allocation._freed
        assert allocation._node.children == {}

        cache.commit_and_release(
            allocation, output_tokens=output_tokens, decode_pages=(decode_page,)
        )

        assert allocation._freed
        assert cache.num_free_pages == 5
        assert attempts == 3
        assert [event["type"] for event in events] == ["BlockStored", "BlockStored"]
        cache_hit = cache.allocate(allocation.tokens + output_tokens)
        assert cache_hit.num_cached_tokens == 12
        assert decode_page in cache_hit.cached_pages
        cache.release_preempted(cache_hit)

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
    def test_drain_quarantines_each_malformed_message_and_respects_boundary(
        self, caplog
    ):
        import json
        import logging

        zmq = pytest.importorskip("zmq")

        class FakeSocket:
            def __init__(self, messages):
                self.messages = list(messages)

            def recv_multipart(self, *, flags):
                del flags
                if not self.messages:
                    raise zmq.Again()
                return self.messages.pop(0)

        def valid(event):
            return [b"r1", json.dumps(event).encode()]

        messages = [
            valid({"type": "BlockStored", "block_hashes": ["a"]}),
            [b"r1", b'["array-marker"]'],
            valid({"type": "BlockStored", "block_hashes": ["b"]}),
            [
                b"\xffREPLICA_SECRET",
                json.dumps(
                    {"type": "BlockStored", "block_hashes": ["not-applied-id"]}
                ).encode(),
            ],
            valid({"type": "BlockRemoved", "block_hashes": ["a"]}),
            [b"r1", b"\xffPAYLOAD_SECRET"],
            valid({"type": "BlockStored", "block_hashes": ["c"]}),
            [b"r1", b"INVALID_JSON_SECRET"],
            valid({"type": "BlockRemoved", "block_hashes": ["b"]}),
            [b"ONE_FRAME_SECRET"],
            valid({"type": "BlockStored", "block_hashes": ["d"]}),
            [
                b"r1",
                json.dumps(
                    {
                        "type": "BlockStored",
                        "block_hashes": ["not-applied-three-frames"],
                    }
                ).encode(),
                b"THREE_FRAME_SECRET",
            ],
            valid({"type": "BlockStored", "block_hashes": ["e"]}),
        ]
        clock = {"t": 0.0}

        def now():
            clock["t"] += 1.0
            return clock["t"]

        index = KvEventIndex(now=now)
        socket = FakeSocket(messages)
        subscriber = object.__new__(ZmqKvEventSubscriber)
        subscriber._socket = socket
        subscriber._index = index

        with caplog.at_level(logging.WARNING, logger="kairyu.kv_index"):
            assert subscriber.drain(max_events=6) == 6
            assert index._replicas["r1"].hashes == {"b"}
            assert index._replicas["r1"].last_event == 3.0
            assert len(socket.messages) == 7

            assert subscriber.drain(max_events=100) == 7

        assert index._replicas["r1"].hashes == {"c", "d", "e"}
        assert index._replicas["r1"].last_event == 7.0
        assert len(caplog.records) == 6
        warnings = "\n".join(caplog.messages)
        assert warnings.count("ValueError") == 1
        assert warnings.count("UnicodeDecodeError") == 2
        assert warnings.count("JSONDecodeError") == 1
        assert "frame_count=1" in warnings
        assert "frame_count=3" in warnings
        assert "array-marker" not in warnings
        assert "REPLICA_SECRET" not in warnings
        assert "not-applied-id" not in warnings
        assert "PAYLOAD_SECRET" not in warnings
        assert "INVALID_JSON_SECRET" not in warnings
        assert "ONE_FRAME_SECRET" not in warnings
        assert "not-applied-three-frames" not in warnings
        assert "THREE_FRAME_SECRET" not in warnings

    @pytest.mark.parametrize(
        "error_type",
        [
            pytest.param(KeyboardInterrupt, id="keyboard-interrupt"),
            pytest.param(MemoryError, id="memory-error"),
            pytest.param(asyncio.CancelledError, id="cancelled-error"),
        ],
    )
    def test_drain_does_not_quarantine_process_errors(self, error_type):
        import json

        pytest.importorskip("zmq")

        class OneMessageSocket:
            def recv_multipart(self, *, flags):
                del flags
                return [
                    b"r1",
                    json.dumps(
                        {"type": "BlockStored", "block_hashes": ["h1"]}
                    ).encode(),
                ]

        def raise_process_error():
            raise error_type()

        index = KvEventIndex(now=raise_process_error)
        subscriber = object.__new__(ZmqKvEventSubscriber)
        subscriber._socket = OneMessageSocket()
        subscriber._index = index

        with pytest.raises(error_type):
            subscriber.drain()

    def test_pub_sub_round_trip_and_chaos_staleness(self):
        zmq = pytest.importorskip("zmq")
        del zmq
        import time as _time

        from kairyu.orchestration.kv_index import (
            ZmqKvEventPublisher,
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
