"""m10a gates: dynamic membership, HRW remap property, drain, registry,
reconciler, tracing spans, helm render."""

import shutil
import subprocess

import pytest

from kairyu.deploy import registry as registry_module
from kairyu.deploy.registry import (
    PoolReconciler,
    RegistryDiscovery,
    ReplicaRegistry,
    StaticDiscovery,
)
from kairyu.engine.backend import (
    CacheHint,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
)
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.orchestration.replica import ReplicaPool

pytestmark = pytest.mark.asyncio


class MockBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        return GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

    async def stream(self, request):
        yield GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

    async def shutdown(self) -> None:
        return None


class ShutdownRecordingBackend(MockBackend):
    def __init__(self, name: str, *, fail_shutdown: bool = False) -> None:
        super().__init__()
        self.name = name
        self.shutdown_calls = 0
        self.fail_shutdown = fail_shutdown

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError(f"{self.name} shutdown failed")


class MutableDiscovery:
    def __init__(self, members) -> None:
        self.members = dict(members)

    def poll(self):
        return dict(self.members)


class RecordingFactory:
    def __init__(
        self,
        pool: ReplicaPool,
        replica_id: str,
        old_backend: ShutdownRecordingBackend,
        *,
        fail: bool = False,
    ) -> None:
        self.pool = pool
        self.replica_id = replica_id
        self.old_backend = old_backend
        self.fail = fail
        self.identities = []
        self.candidates = []

    def __call__(self, identity):
        self.identities.append(identity)
        assert self.pool._entries[self.replica_id].backend is self.old_backend
        if not self.candidates:
            assert self.pool.is_draining(self.replica_id) is False
        if self.fail:
            raise RuntimeError("candidate factory failed")
        candidate = ShutdownRecordingBackend(
            f"candidate-{self.replica_id}-{len(self.candidates) + 1}"
        )
        self.candidates.append(candidate)
        return candidate, f"{identity.address.removesuffix('/v1')}/readyz"


def _request(session: str | None = None) -> GenerationRequest:
    hint = CacheHint(session_id=session) if session else None
    return GenerationRequest(
        request_id="req", prompt="p", sampling_params=SamplingParams(), cache_hint=hint
    )


class TestDynamicMembership:
    async def test_add_drain_remove_lifecycle(self):
        pool = ReplicaPool({"a": MockBackend(), "b": MockBackend()})
        pool.add_replica("c", MockBackend(), health_url="http://c/health")
        assert pool.replica_ids == ("a", "b", "c")
        assert pool.health_url("c") == "http://c/health"

        pool.drain("a")
        assert pool.is_draining("a")
        # drained replicas take no NEW placements
        for _ in range(8):
            await pool.generate(_request())
        assert pool.outstanding_by_id()["a"] == 0

        await pool.remove_replica("a")
        assert pool.replica_ids == ("b", "c")
        with pytest.raises(ValueError, match="already"):
            pool.add_replica("b", MockBackend())

    async def test_cancel_drain_preserves_health_and_outstanding(self):
        backend = MockBackend()
        pool = ReplicaPool({"replica": backend}, unhealthy_after=2)
        entry = pool._entries["replica"]
        entry.consecutive_failures = 2
        entry.outstanding = 3
        health_before = pool.healthy_by_id()
        outstanding_before = pool.outstanding_by_id()

        pool.drain("replica")
        pool.cancel_drain("replica")

        assert pool._entries["replica"] is entry
        assert pool._entries["replica"].backend is backend
        assert pool.is_draining("replica") is False
        assert pool.healthy_by_id() == health_before
        assert pool.outstanding_by_id() == outstanding_before

    async def test_manual_cancel_does_not_release_active_drain_lease(self):
        pool = ReplicaPool({"replica": MockBackend()})
        lease = pool.acquire_drain("replica")
        pool.drain("replica")

        pool.cancel_drain("replica")

        assert pool.is_draining("replica") is True
        with pytest.raises(RuntimeError, match="eligible"):
            pool._select(_request())
        pool.release_drain("replica", lease)
        assert pool.is_draining("replica") is False

    async def test_manual_drain_query_ignores_active_lease(self):
        pool = ReplicaPool({"replica": MockBackend()})

        assert pool.is_manually_draining("replica") is False
        lease = pool.acquire_drain("replica")
        assert pool.is_draining("replica") is True
        assert pool.is_manually_draining("replica") is False
        pool.drain("replica")
        assert pool.is_manually_draining("replica") is True
        pool.cancel_drain("replica")
        assert pool.is_manually_draining("replica") is False
        assert pool.is_draining("replica") is True
        pool.release_drain("replica", lease)

    async def test_release_drain_lease_preserves_state_and_restores_eligibility(self):
        backend = MockBackend()
        pool = ReplicaPool({"replica": backend}, unhealthy_after=2)
        entry = pool._entries["replica"]
        entry.consecutive_failures = 1
        entry.outstanding = 3
        health_before = pool.healthy_by_id()
        outstanding_before = pool.outstanding_by_id()
        lease = pool.acquire_drain("replica")

        pool.release_drain("replica", lease)

        assert pool._entries["replica"] is entry
        assert pool._entries["replica"].backend is backend
        assert pool.is_draining("replica") is False
        assert pool.healthy_by_id() == health_before
        assert pool.outstanding_by_id() == outstanding_before
        assert pool._select(_request())[0] == "replica"

    async def test_remove_refuses_inflight_then_force(self):
        import asyncio

        class SlowBackend(MockBackend):
            def __init__(self):
                super().__init__()
                self.release = asyncio.Event()

            async def generate(self, request):
                await self.release.wait()
                return GenerationResult(request_id="req", prompt="p", completions=(), finished=True)

        slow = SlowBackend()
        pool = ReplicaPool({"s": slow})
        task = asyncio.create_task(pool.generate(_request()))
        await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="in-flight"):
            await pool.remove_replica("s")
        await pool.remove_replica("s", force=True)
        slow.release.set()
        await task  # late completion on removed id is a no-op (A2)

    async def test_all_draining_raises(self):
        pool = ReplicaPool([MockBackend()])
        pool.drain("0")
        with pytest.raises(RuntimeError, match="eligible"):
            await pool.generate(_request())

    async def test_probe_never_clears_draining(self):
        pool = ReplicaPool([MockBackend(), MockBackend()])
        pool.drain("1")
        await pool.probe(1)  # legacy ordinal accepted
        assert pool.is_draining("1")


class TestHrwRemapProperty:
    def _mapping(self, pool: ReplicaPool, sessions: list[str]) -> dict[str, str]:
        return {
            session: pool._select(_request(session))[0] for session in sessions
        }

    async def test_removal_remaps_only_departed_sessions(self):
        backends = {str(i): MockBackend() for i in range(8)}
        pool = ReplicaPool(backends)
        sessions = [f"s{i}" for i in range(400)]
        before = self._mapping(pool, sessions)
        await pool.remove_replica("3")
        after = self._mapping(pool, sessions)
        moved = [s for s in sessions if before[s] != after[s]]
        assert all(before[s] == "3" for s in moved)  # only its own sessions

    async def test_addition_remaps_about_one_over_n(self):
        pool = ReplicaPool({str(i): MockBackend() for i in range(8)})
        sessions = [f"s{i}" for i in range(400)]
        before = self._mapping(pool, sessions)
        pool.add_replica("8", MockBackend())
        after = self._mapping(pool, sessions)
        moved = sum(before[s] != after[s] for s in sessions)
        assert moved <= 400 / 9 * 1.6  # ~1/N with slack
        assert all(after[s] == "8" for s in sessions if before[s] != after[s])


class TestRegistryAndReconciler:
    def test_ttl_expiry(self):
        clock = {"t": 0.0}
        registry = ReplicaRegistry(now=lambda: clock["t"])
        registry.register("r1", "http://r1/v1", ttl_s=10)
        assert registry.alive() == {
            "r1": registry_module.ReplicaConfig(address="http://r1/v1")
        }
        clock["t"] = 9.0
        registry.heartbeat("r1")
        clock["t"] = 18.0
        assert registry.alive() == {
            "r1": registry_module.ReplicaConfig(address="http://r1/v1")
        }
        clock["t"] = 30.0
        assert registry.alive() == {}
        with pytest.raises(KeyError):
            registry.heartbeat("ghost")

    async def test_reconciler_adds_and_drain_removes(self):
        pool = ReplicaPool({"old": MockBackend()})
        members = {"old": "http://old/v1", "new": "http://new/v1"}
        source = StaticDiscovery(members)
        reconciler = PoolReconciler(
            pool,
            source,
            default_model="llama",
            factory=lambda identity: (
                MockBackend(),
                f"{identity.address}/health",
            ),
        )
        result = await reconciler.reconcile()
        assert result["added"] == ["new"]
        assert pool.replica_ids == ("old", "new")

        members.pop("old")
        source._members.pop("old")
        result = await reconciler.reconcile()
        assert result["removed"] == ["old"]
        assert pool.replica_ids == ("new",)

    async def test_reconciler_retries_inflight_removal(self):
        pool = ReplicaPool({"busy": MockBackend(), "idle": MockBackend()})
        pool._entries["busy"].outstanding = 1  # simulate in-flight
        source = StaticDiscovery({"idle": "http://idle/v1"})
        reconciler = PoolReconciler(
            pool,
            source,
            default_model="llama",
            factory=lambda identity: (MockBackend(), None),
        )
        result = await reconciler.reconcile()
        assert result["removed"] == []
        assert "busy" in result["draining"]
        assert pool.is_draining("busy")
        pool._entries["busy"].outstanding = 0
        result = await reconciler.reconcile()
        assert result["removed"] == ["busy"]

    def test_registry_discovery_bridges(self):
        clock = {"t": 0.0}
        registry = ReplicaRegistry(now=lambda: clock["t"])
        registry.register(
            "r1",
            "http://r1/v1",
            model="llama",
            api_key_env=None,
        )
        assert RegistryDiscovery(registry).poll() == {
            "r1": registry_module.ReplicaConfig(
                address="http://r1/v1",
                model="llama",
                api_key_env=None,
            )
        }

    def test_default_openai_factory_receives_model_and_auth_identity(self):
        identity = registry_module.ReplicaIdentity(
            address="http://replica/v1",
            model="llama",
            api_key_env=None,
        )

        backend, health_url = registry_module.openai_replica_factory(identity)

        assert isinstance(backend, OpenAICompatBackend)
        assert backend._base_url == "http://replica/v1"
        assert backend._model == "llama"
        assert backend._api_key_env is None
        assert health_url == "http://replica/readyz"

    async def test_initial_scale_uses_reconciler_default_model(self):
        pool = ReplicaPool({"seed": MockBackend()})
        source = StaticDiscovery({"replica": "http://replica/v1"})
        reconciler = PoolReconciler(pool, source, default_model="llama")

        result = await reconciler.reconcile()

        assert result["added"] == ["replica"]
        backend = pool._entries["replica"].backend
        assert isinstance(backend, OpenAICompatBackend)
        assert backend._model == "llama"

    async def test_discovery_identity_overrides_reconciler_model_and_auth_default(self):
        identities = []
        source_identity = registry_module.ReplicaIdentity(
            address="http://replica/v1",
            model="source-model",
            api_key_env=None,
        )
        source = StaticDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address=source_identity.address,
                    model=source_identity.model,
                    api_key_env=source_identity.api_key_env,
                )
            }
        )
        pool = ReplicaPool({"seed": MockBackend()})

        def recording_factory(identity):
            identities.append(identity)
            return MockBackend(), None

        reconciler = PoolReconciler(
            pool,
            source,
            factory=recording_factory,
            default_model="fallback-model",
        )

        await reconciler.reconcile()

        assert identities == [source_identity]

    async def test_unchanged_identity_is_a_complete_noop_on_second_reconcile(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1",
                    model="llama",
                    api_key_env="OLD_API_KEY",
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)

        assert await reconciler.reconcile() == {
            "added": [],
            "draining": [],
            "removed": [],
        }
        result = await reconciler.reconcile()

        assert result == {"added": [], "draining": [], "removed": []}
        assert factory.identities == []
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert old.shutdown_calls == 0

    async def test_reconciler_does_not_cancel_unowned_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        pool.drain("replica")
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(pool, source)

        result = await reconciler.reconcile()

        assert result == {"added": [], "draining": [], "removed": []}
        assert pool.is_draining("replica") is True
        assert reconciler._draining == {}

    async def test_address_change_replacement_constructs_before_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        old_config = registry_module.ReplicaConfig(
            address="http://old/v1", model="llama", api_key_env="API_KEY"
        )
        source = MutableDiscovery({"replica": old_config})
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()  # seed the pre-existing identity baseline
        new_config = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama", api_key_env="API_KEY"
        )
        source.members["replica"] = new_config

        result = await reconciler.reconcile()

        candidate = factory.candidates[0]
        assert factory.identities == [
            registry_module.ReplicaIdentity(
                address="http://new/v1", model="llama", api_key_env="API_KEY"
            )
        ]
        assert result == {
            "added": ["replica"],
            "draining": [],
            "removed": ["replica"],
        }
        assert pool._entries["replica"].backend is candidate
        assert pool.health_url("replica") == "http://new/readyz"
        assert old.shutdown_calls == 1
        assert candidate.shutdown_calls == 0

    async def test_successful_replacement_preserves_manual_drain_on_new_backend(self):
        old = ShutdownRecordingBackend("old")
        candidate = ShutdownRecordingBackend("candidate")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )

        def factory(identity):
            assert pool._entries["replica"].backend is old
            assert pool.is_draining("replica") is True
            return candidate, None

        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        pool.drain("replica")
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )

        result = await reconciler.reconcile()

        assert result == {
            "added": ["replica"],
            "draining": [],
            "removed": ["replica"],
        }
        assert pool._entries["replica"].backend is candidate
        assert pool.is_draining("replica") is True
        assert pool.is_manually_draining("replica") is True
        with pytest.raises(RuntimeError, match="eligible"):
            pool._select(_request())
        assert pool.healthy_by_id() == {"replica": True}
        assert pool.outstanding_by_id() == {"replica": 0}
        assert pool._entries["replica"].drain_leases == set()
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1
        assert candidate.shutdown_calls == 0

    async def test_model_change_triggers_replacement(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="old-model"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://replica/v1", model="new-model"
        )

        result = await reconciler.reconcile()

        assert result["removed"] == ["replica"]
        assert result["added"] == ["replica"]
        assert factory.identities[0].model == "new-model"
        assert pool._entries["replica"].backend is factory.candidates[0]
        assert old.shutdown_calls == 1

    async def test_auth_change_triggers_replacement(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1",
                    model="llama",
                    api_key_env="OLD_API_KEY",
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://replica/v1",
            model="llama",
            api_key_env="NEW_API_KEY",
        )

        result = await reconciler.reconcile()

        assert result["removed"] == ["replica"]
        assert result["added"] == ["replica"]
        assert factory.identities[0].api_key_env == "NEW_API_KEY"
        assert pool._entries["replica"].backend is factory.candidates[0]
        assert old.shutdown_calls == 1

    async def test_replacement_factory_failure_keeps_old_replica_eligible(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old, fail=True)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )

        result = await reconciler.reconcile()

        assert result == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert pool._select(_request())[0] == "replica"
        assert old.shutdown_calls == 0

    async def test_inflight_replacement_cleans_candidate_then_retries(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1

        blocked = await reconciler.reconcile()

        first_candidate = factory.candidates[0]
        assert blocked == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is True
        assert set(reconciler._draining) == {"replica"}
        assert first_candidate.shutdown_calls == 1
        assert old.shutdown_calls == 0

        pool._entries["replica"].outstanding = 0
        completed = await reconciler.reconcile()

        second_candidate = factory.candidates[1]
        assert completed == {
            "added": ["replica"],
            "draining": [],
            "removed": ["replica"],
        }
        assert pool._entries["replica"].backend is second_candidate
        assert first_candidate.shutdown_calls == 1
        assert second_candidate.shutdown_calls == 0
        assert old.shutdown_calls == 1
        assert reconciler._draining == {}

    async def test_reverted_replacement_cancels_owned_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        applied = registry_module.ReplicaConfig(
            address="http://old/v1", model="llama"
        )
        source = MutableDiscovery({"replica": applied})
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1
        await reconciler.reconcile()
        source.members["replica"] = applied

        result = await reconciler.reconcile()

        assert result == {"added": [], "draining": [], "removed": []}
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert pool._select(_request())[0] == "replica"
        assert reconciler._draining == {}
        assert factory.candidates[0].shutdown_calls == 1
        assert old.shutdown_calls == 0

    async def test_removed_desire_reappears_with_same_identity_cancels_owned_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        config = registry_module.ReplicaConfig(
            address="http://replica/v1", model="llama"
        )
        source = MutableDiscovery({"replica": config})
        reconciler = PoolReconciler(pool, source)
        await reconciler.reconcile()
        pool._entries["replica"].outstanding = 1
        source.members.clear()
        blocked = await reconciler.reconcile()
        source.members["replica"] = config

        restored = await reconciler.reconcile()

        assert blocked == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert restored == {"added": [], "draining": [], "removed": []}
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert pool._select(_request())[0] == "replica"
        assert reconciler._draining == {}
        assert old.shutdown_calls == 0

    async def test_retry_factory_failure_cancels_owned_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1
        await reconciler.reconcile()
        factory.fail = True

        result = await reconciler.reconcile()

        assert result == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is False
        assert pool._select(_request())[0] == "replica"
        assert reconciler._draining == {}
        assert factory.candidates[0].shutdown_calls == 1
        assert old.shutdown_calls == 0

    async def test_reverted_replacement_preserves_later_manual_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        applied = registry_module.ReplicaConfig(
            address="http://old/v1", model="llama"
        )
        source = MutableDiscovery({"replica": applied})
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1
        await reconciler.reconcile()
        pool.drain("replica")
        source.members["replica"] = applied

        result = await reconciler.reconcile()

        assert result == {"added": [], "draining": [], "removed": []}
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is True
        with pytest.raises(RuntimeError, match="eligible"):
            pool._select(_request())
        assert reconciler._draining == {}
        assert old.shutdown_calls == 0

    async def test_retry_factory_failure_preserves_later_manual_drain(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )
        pool._entries["replica"].outstanding = 1
        await reconciler.reconcile()
        pool.drain("replica")
        factory.fail = True

        result = await reconciler.reconcile()

        assert result == {
            "added": [],
            "draining": ["replica"],
            "removed": [],
        }
        assert pool._entries["replica"].backend is old
        assert pool.is_draining("replica") is True
        with pytest.raises(RuntimeError, match="eligible"):
            pool._select(_request())
        assert reconciler._draining == {}
        assert old.shutdown_calls == 0

    async def test_replacement_add_failure_cleans_candidate_and_applied_identity(
        self, monkeypatch
    ):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )
        factory = RecordingFactory(pool, "replica", old)
        reconciler = PoolReconciler(pool, source, factory=factory)
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )

        def reject_add(replica_id, backend, health_url=None):
            raise ValueError("replacement add failed")

        monkeypatch.setattr(pool, "add_replica", reject_add)

        with pytest.raises(ValueError, match="replacement add failed"):
            await reconciler.reconcile()

        candidate = factory.candidates[0]
        assert pool.replica_ids == ()
        assert reconciler._applied == {}
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1
        assert candidate.shutdown_calls == 1

    async def test_replacement_double_shutdown_failure_clears_applied_identity(self):
        old = ShutdownRecordingBackend("old", fail_shutdown=True)
        candidate = ShutdownRecordingBackend("candidate", fail_shutdown=True)
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://old/v1", model="llama"
                )
            }
        )

        def failing_candidate_factory(identity):
            assert pool._entries["replica"].backend is old
            assert pool.is_draining("replica") is False
            return candidate, None

        reconciler = PoolReconciler(
            pool, source, factory=failing_candidate_factory
        )
        await reconciler.reconcile()
        source.members["replica"] = registry_module.ReplicaConfig(
            address="http://new/v1", model="llama"
        )

        with pytest.raises(ExceptionGroup) as raised:
            await reconciler.reconcile()

        assert "replacement candidate 'replica' shutdown failed" in str(raised.value)
        assert str(raised.value.exceptions[0]) == "candidate shutdown failed"
        old_failure = raised.value.__context__
        assert isinstance(old_failure, ExceptionGroup)
        assert "replica 'replica' shutdown failed" in str(old_failure)
        assert str(old_failure.exceptions[0]) == "old shutdown failed"
        assert old.shutdown_calls == 1
        assert candidate.shutdown_calls == 1
        assert pool.replica_ids == ()
        assert reconciler._applied == {}
        assert reconciler._draining == {}

    async def test_reconciliation_result_lists_preserve_phase_order(self):
        remove_z = ShutdownRecordingBackend("remove-z")
        replace_b = ShutdownRecordingBackend("replace-b")
        remove_a = ShutdownRecordingBackend("remove-a")
        replace_a = ShutdownRecordingBackend("replace-a")
        keep = ShutdownRecordingBackend("keep")
        pool = ReplicaPool(
            {
                "remove-z": remove_z,
                "replace-b": replace_b,
                "remove-a": remove_a,
                "replace-a": replace_a,
                "keep": keep,
            }
        )

        def config(address):
            return registry_module.ReplicaConfig(address=address, model="llama")

        source = MutableDiscovery(
            {
                "remove-z": config("http://remove-z/v1"),
                "replace-b": config("http://replace-b-old/v1"),
                "remove-a": config("http://remove-a/v1"),
                "replace-a": config("http://replace-a-old/v1"),
                "keep": config("http://keep/v1"),
            }
        )
        factory_calls = []
        candidates = {}

        def ordered_factory(identity):
            factory_calls.append(identity.address)
            if identity.address == "http://replace-a-new/v1":
                assert pool._entries["replace-a"].backend is replace_a
                assert pool.is_draining("replace-a") is False
            if identity.address == "http://replace-b-new/v1":
                assert pool._entries["replace-b"].backend is replace_b
                assert pool.is_draining("replace-b") is False
            candidate = ShutdownRecordingBackend(identity.address)
            candidates[identity.address] = candidate
            return candidate, None

        reconciler = PoolReconciler(pool, source, factory=ordered_factory)
        await reconciler.reconcile()
        source.members = {
            "replace-a": config("http://replace-a-new/v1"),
            "new-b": config("http://new-b/v1"),
            "replace-b": config("http://replace-b-new/v1"),
            "new-a": config("http://new-a/v1"),
            "keep": config("http://keep/v1"),
        }

        result = await reconciler.reconcile()

        assert factory_calls == [
            "http://new-b/v1",
            "http://new-a/v1",
            "http://replace-a-new/v1",
            "http://replace-b-new/v1",
        ]
        assert result == {
            "added": ["new-b", "new-a", "replace-a", "replace-b"],
            "draining": ["remove-z", "remove-a"],
            "removed": ["replace-a", "replace-b", "remove-z", "remove-a"],
        }
        assert remove_z.shutdown_calls == 1
        assert replace_b.shutdown_calls == 1
        assert remove_a.shutdown_calls == 1
        assert replace_a.shutdown_calls == 1
        assert keep.shutdown_calls == 0
        assert all(candidate.shutdown_calls == 0 for candidate in candidates.values())

    async def test_desired_removal_drains_and_shuts_down_backend_once(self):
        old = ShutdownRecordingBackend("old")
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(pool, source)
        await reconciler.reconcile()
        source.members.clear()

        result = await reconciler.reconcile()

        assert result == {
            "added": [],
            "draining": ["replica"],
            "removed": ["replica"],
        }
        assert "replica" not in pool.replica_ids
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1

    async def test_missing_model_validation_precedes_all_pool_mutation(self):
        seed = ShutdownRecordingBackend("seed")
        pool = ReplicaPool({"seed": seed})
        source = MutableDiscovery(
            {
                "valid": registry_module.ReplicaConfig(
                    address="http://valid/v1", model="llama"
                ),
                "invalid": registry_module.ReplicaConfig(
                    address="http://invalid/v1", model=None
                ),
            }
        )
        factory_calls = []
        reconciler = PoolReconciler(
            pool,
            source,
            factory=lambda identity: (factory_calls.append(identity) or MockBackend(), None),
        )

        with pytest.raises(ValueError, match="'invalid' requires a model"):
            await reconciler.reconcile()

        assert pool.replica_ids == ("seed",)
        assert pool.is_draining("seed") is False
        assert factory_calls == []
        assert reconciler._applied == {}

    async def test_add_failure_cleans_unused_candidate_and_does_not_record_identity(
        self, monkeypatch
    ):
        seed = ShutdownRecordingBackend("seed")
        candidate = ShutdownRecordingBackend("candidate")
        pool = ReplicaPool({"seed": seed})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(
            pool, source, factory=lambda identity: (candidate, None)
        )

        def reject_add(replica_id, backend, health_url=None):
            raise ValueError("add failed")

        monkeypatch.setattr(pool, "add_replica", reject_add)

        with pytest.raises(ValueError, match="add failed"):
            await reconciler.reconcile()

        assert pool.replica_ids == ("seed",)
        assert reconciler._applied == {}
        assert candidate.shutdown_calls == 1

    async def test_removal_shutdown_failure_does_not_leave_stale_applied_identity(self):
        old = ShutdownRecordingBackend("old", fail_shutdown=True)
        pool = ReplicaPool({"replica": old})
        source = MutableDiscovery(
            {
                "replica": registry_module.ReplicaConfig(
                    address="http://replica/v1", model="llama"
                )
            }
        )
        reconciler = PoolReconciler(pool, source)
        await reconciler.reconcile()
        source.members.clear()

        with pytest.raises(ExceptionGroup, match="replica 'replica' shutdown failed"):
            await reconciler.reconcile()

        assert "replica" not in pool.replica_ids
        assert "replica" not in reconciler._applied
        assert reconciler._draining == {}
        assert old.shutdown_calls == 1


class TestTracing:
    async def test_spans_recorded_when_enabled(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from kairyu.telemetry import configure_tracing, traced_span

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        configure_tracing(True)
        try:
            with traced_span("kairyu.pool.place", {"replica_id": "3", "reason": "x"}):
                pass
        finally:
            configure_tracing(False)
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["kairyu.pool.place"]
        assert spans[0].attributes["replica_id"] == "3"

    async def test_disabled_is_noop_without_otel_import(self):
        from kairyu.telemetry import traced_span, tracing_enabled

        assert not tracing_enabled()
        with traced_span("anything") as span:
            assert span is None


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")
def test_helm_chart_renders():
    rendered = subprocess.run(
        ["helm", "template", "kairyu", "deploy/helm/kairyu"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "kind: Deployment" in rendered
    assert "path: /readyz" in rendered
    assert "mountPath: /etc/kairyu" in rendered  # the Dockerfile CMD path (A11)


def test_helm_chart_config_is_a_valid_deployment_spec():
    """kind-smoke root cause (PR #16): the chart shipped 'models:' which is
    not a DeploymentSpec field — the pod crash-looped at validation. Pin the
    embedded config to the real schema, no helm binary needed."""
    import yaml

    from kairyu.deploy.spec import load_deployment_spec

    values = yaml.safe_load(open("deploy/helm/kairyu/values.yaml"))
    spec = load_deployment_spec(values["config"])
    assert spec.engines, "chart config must declare at least one engine"
