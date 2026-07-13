"""Replica discovery + reconciliation (m10a D2).

``ReplicaRegistry`` is TTL-heartbeat membership (clock injected — A7).
``DiscoverySource`` is the seam: ``StaticDiscovery`` (spec lists),
``RegistryDiscovery`` (the TTL registry); the k8s-endpoints source is a thin
deploy-day adapter behind the same ``poll()`` shape. ``PoolReconciler`` diffs
desired vs current membership: additions attach immediately; removals go
drain-then-remove and TOLERATE in-flight refusal (retried next tick — A6).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from kairyu.engine.backend import EngineBackend, shutdown_all
from kairyu.orchestration.replica import ReplicaPool


@dataclass(frozen=True)
class ReplicaConfig:
    address: str
    model: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"


@dataclass(frozen=True)
class ReplicaIdentity:
    address: str
    model: str
    api_key_env: str | None


def _copy_config(config: ReplicaConfig) -> ReplicaConfig:
    return ReplicaConfig(
        address=config.address,
        model=config.model,
        api_key_env=config.api_key_env,
    )


class DiscoverySource(Protocol):
    def poll(self) -> dict[str, ReplicaConfig]:
        """Current desired membership: replica_id -> backend configuration."""
        ...


class StaticDiscovery:
    def __init__(self, members: Mapping[str, str | ReplicaConfig]) -> None:
        self._members = {
            replica_id: (
                ReplicaConfig(address=config)
                if isinstance(config, str)
                else _copy_config(config)
            )
            for replica_id, config in members.items()
        }

    def poll(self) -> dict[str, ReplicaConfig]:
        return {
            replica_id: _copy_config(config)
            for replica_id, config in self._members.items()
        }


@dataclass
class _Registration:
    config: ReplicaConfig
    ttl_s: float
    last_heartbeat: float


class ReplicaRegistry:
    """TTL-heartbeat membership; replicas self-register and heartbeat."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._members: dict[str, _Registration] = {}

    def register(
        self,
        replica_id: str,
        address: str,
        ttl_s: float = 15.0,
        model: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
    ) -> None:
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be > 0, got {ttl_s}")
        config = ReplicaConfig(
            address=address,
            model=model,
            api_key_env=api_key_env,
        )
        self._members[replica_id] = _Registration(config, ttl_s, self._now())

    def heartbeat(self, replica_id: str) -> None:
        member = self._members.get(replica_id)
        if member is None:
            raise KeyError(f"unknown replica {replica_id!r}; register first")
        member.last_heartbeat = self._now()

    def deregister(self, replica_id: str) -> None:
        self._members.pop(replica_id, None)

    def alive(self) -> dict[str, ReplicaConfig]:
        current = self._now()
        return {
            replica_id: _copy_config(member.config)
            for replica_id, member in self._members.items()
            if current - member.last_heartbeat <= member.ttl_s
        }


class RegistryDiscovery:
    def __init__(self, registry: ReplicaRegistry) -> None:
        self._registry = registry

    def poll(self) -> dict[str, ReplicaConfig]:
        return self._registry.alive()


# factory: complete identity -> (backend, readiness_url) — closes over
# create_backend and the resolved_health_url /v1-strip rule (m10a A6)
ReplicaFactory = Callable[[ReplicaIdentity], tuple[EngineBackend, str | None]]


def openai_replica_factory(
    identity: ReplicaIdentity,
) -> tuple[EngineBackend, str | None]:
    """Default factory for a fully identified OpenAI-protocol replica."""
    from kairyu.engine.registry import create_backend

    backend = create_backend(
        "openai",
        base_url=identity.address,
        model=identity.model,
        api_key_env=identity.api_key_env,
    )
    base = identity.address.removesuffix("/v1")
    # readiness, not liveness: a drained/wedged replica must stay ejected (O3)
    return backend, f"{base}/readyz"


class PoolReconciler:
    """One tick: converge the pool's membership toward the discovery source."""

    def __init__(
        self,
        pool: ReplicaPool,
        source: DiscoverySource,
        factory: ReplicaFactory = openai_replica_factory,
        default_model: str | None = None,
    ) -> None:
        self._pool = pool
        self._source = source
        self._factory = factory
        self._default_model = default_model
        self._applied: dict[str, ReplicaIdentity] = {}

    def _resolve_identity(
        self, replica_id: str, config: ReplicaConfig
    ) -> ReplicaIdentity:
        model = config.model or self._default_model
        if not model:
            raise ValueError(f"replica {replica_id!r} requires a model")
        return ReplicaIdentity(
            address=config.address,
            model=model,
            api_key_env=config.api_key_env,
        )

    async def reconcile(self) -> dict[str, list[str]]:
        """Returns {'added': [...], 'draining': [...], 'removed': [...]}."""
        desired = {
            replica_id: self._resolve_identity(replica_id, config)
            for replica_id, config in self._source.poll().items()
        }
        current_ids = self._pool.replica_ids
        current = set(current_ids)
        for replica_id in current_ids:
            if replica_id not in desired:
                continue
            self._applied.setdefault(replica_id, desired[replica_id])

        added: list[str] = []
        draining: list[str] = []
        removed: list[str] = []

        for replica_id, identity in desired.items():
            if replica_id not in current:
                backend, health_url = self._factory(identity)
                try:
                    self._pool.add_replica(
                        replica_id, backend, health_url=health_url
                    )
                except BaseException:
                    await shutdown_all((backend,), f"replica candidate {replica_id!r}")
                    raise
                self._applied[replica_id] = identity
                added.append(replica_id)

        for replica_id, identity in desired.items():
            if replica_id not in current or self._applied[replica_id] == identity:
                continue
            try:
                candidate, health_url = self._factory(identity)
            except Exception:
                draining.append(replica_id)
                continue

            self._pool.drain(replica_id)
            try:
                await self._pool.remove_replica(replica_id)
            except RuntimeError:
                await shutdown_all(
                    (candidate,), f"replacement candidate {replica_id!r}"
                )
                draining.append(replica_id)
                continue
            except BaseException:
                await shutdown_all(
                    (candidate,), f"replacement candidate {replica_id!r}"
                )
                if replica_id not in self._pool.replica_ids:
                    self._applied.pop(replica_id, None)
                raise

            self._applied.pop(replica_id, None)
            try:
                self._pool.add_replica(
                    replica_id, candidate, health_url=health_url
                )
            except BaseException:
                await shutdown_all(
                    (candidate,), f"replacement candidate {replica_id!r}"
                )
                raise
            self._applied[replica_id] = identity
            removed.append(replica_id)
            added.append(replica_id)

        for replica_id in current_ids:
            if replica_id in desired:
                continue
            if not self._pool.is_draining(replica_id):
                self._pool.drain(replica_id)
                draining.append(replica_id)
            try:
                await self._pool.remove_replica(replica_id)
            except RuntimeError:
                # in-flight work: drain holds, removal retries next tick (A6)
                if replica_id not in draining:
                    draining.append(replica_id)
            except BaseException:
                if replica_id not in self._pool.replica_ids:
                    self._applied.pop(replica_id, None)
                raise
            else:
                self._applied.pop(replica_id, None)
                removed.append(replica_id)
        return {"added": added, "draining": draining, "removed": removed}
