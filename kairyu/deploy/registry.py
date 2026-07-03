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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from kairyu.orchestration.replica import ReplicaPool


class DiscoverySource(Protocol):
    def poll(self) -> dict[str, str]:
        """Current desired membership: replica_id -> address."""
        ...


class StaticDiscovery:
    def __init__(self, members: dict[str, str]) -> None:
        self._members = dict(members)

    def poll(self) -> dict[str, str]:
        return dict(self._members)


@dataclass
class _Registration:
    address: str
    ttl_s: float
    last_heartbeat: float


class ReplicaRegistry:
    """TTL-heartbeat membership; replicas self-register and heartbeat."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._members: dict[str, _Registration] = {}

    def register(self, replica_id: str, address: str, ttl_s: float = 15.0) -> None:
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be > 0, got {ttl_s}")
        self._members[replica_id] = _Registration(address, ttl_s, self._now())

    def heartbeat(self, replica_id: str) -> None:
        member = self._members.get(replica_id)
        if member is None:
            raise KeyError(f"unknown replica {replica_id!r}; register first")
        member.last_heartbeat = self._now()

    def deregister(self, replica_id: str) -> None:
        self._members.pop(replica_id, None)

    def alive(self) -> dict[str, str]:
        current = self._now()
        return {
            replica_id: member.address
            for replica_id, member in self._members.items()
            if current - member.last_heartbeat <= member.ttl_s
        }


class RegistryDiscovery:
    def __init__(self, registry: ReplicaRegistry) -> None:
        self._registry = registry

    def poll(self) -> dict[str, str]:
        return self._registry.alive()


# factory: address -> (backend, health_url) — closes over create_backend and
# the resolved_health_url /v1-strip rule (m10a A6)
ReplicaFactory = Callable[[str], tuple[object, str | None]]


def openai_replica_factory(address: str) -> tuple[object, str | None]:
    """Default factory: an OpenAI-protocol replica at ``address`` (…/v1)."""
    from kairyu.engine.registry import create_backend

    backend = create_backend("openai", base_url=address)
    base = address[: -len("/v1")] if address.endswith("/v1") else address
    return backend, f"{base}/health"


class PoolReconciler:
    """One tick: converge the pool's membership toward the discovery source."""

    def __init__(
        self,
        pool: ReplicaPool,
        source: DiscoverySource,
        factory: ReplicaFactory = openai_replica_factory,
    ) -> None:
        self._pool = pool
        self._source = source
        self._factory = factory

    def reconcile(self) -> dict[str, list[str]]:
        """Returns {'added': [...], 'draining': [...], 'removed': [...]}."""
        desired = self._source.poll()
        current = set(self._pool.replica_ids)
        added: list[str] = []
        draining: list[str] = []
        removed: list[str] = []
        for replica_id, address in desired.items():
            if replica_id not in current:
                backend, health_url = self._factory(address)
                self._pool.add_replica(replica_id, backend, health_url=health_url)
                added.append(replica_id)
        for replica_id in current - set(desired):
            if not self._pool.is_draining(replica_id):
                self._pool.drain(replica_id)
                draining.append(replica_id)
            try:
                self._pool.remove_replica(replica_id)
                removed.append(replica_id)
            except RuntimeError:
                # in-flight work: drain holds, removal retries next tick (A6)
                if replica_id not in draining:
                    draining.append(replica_id)
        return {"added": added, "draining": draining, "removed": removed}
