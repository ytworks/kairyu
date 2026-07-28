"""Replica discovery + reconciliation (m10a D2).

``ReplicaRegistry`` is TTL-heartbeat membership (clock injected — A7).
``DiscoverySource`` is the seam: ``StaticDiscovery`` (spec lists),
``RegistryDiscovery`` (the TTL registry); the Kubernetes EndpointSlice source
is the production in-cluster adapter behind the same ``poll()`` shape.
``PoolReconciler`` diffs desired vs current membership: additions attach
immediately; removals go drain-then-remove and TOLERATE in-flight refusal
(retried next tick — A6).
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from kairyu.engine.backend import (
    EngineBackend,
    shutdown_all_cancellation_safe,
)
from kairyu.orchestration.replica import DrainLease, ReplicaPool

logger = logging.getLogger(__name__)


async def _await_task_cancellation_safe(task: asyncio.Task[None]) -> None:
    """Let lifecycle cleanup finish before propagating caller cancellation."""
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        try:
            task.result()
        except BaseException as cleanup_error:
            raise cancelled from cleanup_error
        raise


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
    def poll(
        self,
    ) -> dict[str, ReplicaConfig] | Awaitable[dict[str, ReplicaConfig]]:
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


class KubernetesEndpointSliceDiscovery:
    """Discover ready replicas from the in-cluster EndpointSlice API.

    The service-account token is read for every request so projected-token
    rotation does not require restarting the gateway.  A supplied client is
    caller-owned by default; clients created here are closed by ``aclose``.
    """

    _SERVICE_ACCOUNT_DIR = Path(
        "/var/run/secrets/kubernetes.io/serviceaccount"
    )

    def __init__(
        self,
        service: str,
        namespace: str,
        port: str | int,
        *,
        model: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        scheme: str = "http",
        address_family_preference: str = "ipv4",
        base_path: str = "/v1",
        api_server: str | None = None,
        token_path: str | Path | None = None,
        ca_path: str | Path | None = None,
        client: httpx.AsyncClient | None = None,
        close_client: bool | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        if not service:
            raise ValueError("service must not be empty")
        if not namespace:
            raise ValueError("namespace must not be empty")
        if isinstance(port, str) and port.isdecimal():
            port = int(port)
        if isinstance(port, int):
            if not 1 <= port <= 65535:
                raise ValueError(f"port must be in 1..65535, got {port}")
        elif not port:
            raise ValueError("port name must not be empty")
        if scheme not in {"http", "https"}:
            raise ValueError(f"scheme must be http or https, got {scheme!r}")
        if address_family_preference not in {"ipv4", "ipv6"}:
            raise ValueError(
                "address_family_preference must be 'ipv4' or 'ipv6', got "
                f"{address_family_preference!r}"
            )
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")

        self._service = service
        self._namespace = namespace
        self._port = port
        self._model = model
        self._api_key_env = api_key_env
        self._scheme = scheme
        self._address_family_preference = address_family_preference
        self._base_path = (
            f"/{base_path.strip('/')}" if base_path.strip("/") else ""
        )
        self._token_path = Path(token_path or self._SERVICE_ACCOUNT_DIR / "token")
        resolved_ca_path = Path(ca_path or self._SERVICE_ACCOUNT_DIR / "ca.crt")

        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        api_port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if api_server is None:
            if not host:
                api_server = "https://kubernetes.default.svc"
            else:
                api_host = f"[{host}]" if ":" in host else host
                api_server = f"https://{api_host}:{api_port}"
        self._api_server = api_server.rstrip("/")
        self._slices_url = (
            f"{self._api_server}/apis/discovery.k8s.io/v1/namespaces/"
            f"{quote(namespace, safe='')}/endpointslices"
        )

        if client is None:
            self._client = httpx.AsyncClient(
                verify=str(resolved_ca_path),
                timeout=timeout_s,
            )
            self._owns_client = True if close_client is None else close_client
        else:
            self._client = client
            self._owns_client = False if close_client is None else close_client
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @staticmethod
    def _select_port(ports: Any, selected: str | int) -> int:
        if not isinstance(ports, list):
            raise ValueError("EndpointSlice ports must be a list")
        matches: list[int] = []
        for entry in ports:
            if not isinstance(entry, dict):
                raise ValueError("EndpointSlice port entries must be objects")
            protocol = entry.get("protocol", "TCP")
            value = entry.get("port")
            if protocol != "TCP" or not isinstance(value, int):
                continue
            if isinstance(selected, int):
                matched = value == selected
            else:
                matched = entry.get("name") == selected
            if matched and 1 <= value <= 65535:
                matches.append(value)
        if len(matches) != 1:
            raise ValueError(
                f"EndpointSlice must expose exactly one matching TCP port "
                f"for {selected!r}"
            )
        return matches[0]

    @staticmethod
    def _replica_id(endpoint: Mapping[str, Any], address: str) -> str:
        target = endpoint.get("targetRef")
        if isinstance(target, dict):
            uid = target.get("uid")
            if isinstance(uid, str) and uid:
                return uid
            name = target.get("name")
            if isinstance(name, str) and name:
                return name
        return address

    def _parse(self, payload: Any) -> dict[str, ReplicaConfig]:
        if not isinstance(payload, dict) or not isinstance(
            payload.get("items"), list
        ):
            raise ValueError("EndpointSlice response must contain an items list")

        candidates: dict[str, list[tuple[int, int, int, str]]] = {}
        for item in payload["items"]:
            if not isinstance(item, dict):
                raise ValueError("EndpointSlice items must be objects")
            address_type = item.get("addressType")
            if address_type not in {"IPv4", "IPv6"}:
                continue
            port = self._select_port(item.get("ports"), self._port)
            endpoints = item.get("endpoints")
            if not isinstance(endpoints, list):
                raise ValueError("EndpointSlice endpoints must be a list")
            for endpoint in endpoints:
                if not isinstance(endpoint, dict):
                    raise ValueError("EndpointSlice endpoints must be objects")
                conditions = endpoint.get("conditions")
                if not isinstance(conditions, dict):
                    continue
                if conditions.get("ready") is not True:
                    continue
                if conditions.get("terminating") is True:
                    continue
                addresses = endpoint.get("addresses")
                if not isinstance(addresses, list) or not addresses:
                    raise ValueError(
                        "ready EndpointSlice endpoints must have addresses"
                    )
                for address in addresses:
                    if not isinstance(address, str):
                        raise ValueError(
                            "EndpointSlice addresses must be strings"
                        )
                    parsed = ipaddress.ip_address(address)
                    expected = 4 if address_type == "IPv4" else 6
                    if parsed.version != expected:
                        raise ValueError(
                            f"{address!r} does not match {address_type}"
                        )
                    replica_id = self._replica_id(endpoint, address)
                    preferred_version = (
                        6
                        if self._address_family_preference == "ipv6"
                        else 4
                    )
                    candidates.setdefault(replica_id, []).append(
                        (
                            0 if parsed.version == preferred_version else 1,
                            int(parsed),
                            port,
                            address,
                        )
                    )

        members: dict[str, ReplicaConfig] = {}
        for replica_id in sorted(candidates):
            _family_rank, _address_int, port, address = min(
                candidates[replica_id]
            )
            parsed = ipaddress.ip_address(address)
            host = f"[{address}]" if parsed.version == 6 else address
            members[replica_id] = ReplicaConfig(
                address=f"{self._scheme}://{host}:{port}{self._base_path}",
                model=self._model,
                api_key_env=self._api_key_env,
            )
        return members

    async def poll(self) -> dict[str, ReplicaConfig]:
        if self._closed:
            raise RuntimeError("KubernetesEndpointSliceDiscovery is closed")
        token = self._token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Kubernetes service-account token is empty")
        response = await self._client.get(
            self._slices_url,
            params={"labelSelector": f"kubernetes.io/service-name={self._service}"},
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        response.raise_for_status()
        return self._parse(response.json())

    async def aclose(self) -> None:
        if self._closed:
            return
        if not self._owns_client:
            self._closed = True
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._client.aclose())
        try:
            await _await_task_cancellation_safe(self._close_task)
        finally:
            if (
                self._close_task.done()
                and not self._close_task.cancelled()
                and self._close_task.exception() is None
            ):
                self._closed = True

    async def __aenter__(self) -> KubernetesEndpointSliceDiscovery:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()


# factory: complete identity -> (backend, readiness_url) — closes over
# create_backend and the resolved_health_url /v1-strip rule (m10a A6)
ReplicaFactory = Callable[[ReplicaIdentity], tuple[EngineBackend, str | None]]
MembershipEventSink = Callable[
    [dict[str, object]], None | Awaitable[None]
]


def openai_replica_factory(
    identity: ReplicaIdentity,
    *,
    client: httpx.AsyncClient | None = None,
) -> tuple[EngineBackend, str | None]:
    """Default factory for a fully identified OpenAI-protocol replica."""
    from kairyu.engine.registry import create_backend

    backend = create_backend(
        "openai",
        base_url=identity.address,
        model=identity.model,
        api_key_env=identity.api_key_env,
        upstream="kairyu",
        client=client,
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
        event_sink: MembershipEventSink | None = None,
    ) -> None:
        self._pool = pool
        self._source = source
        self._factory = factory
        self._default_model = default_model
        self._event_sink = event_sink
        self._applied: dict[str, ReplicaIdentity] = {}
        self._draining: dict[str, DrainLease] = {}
        self._entry_generations: dict[str, str] = {}
        self._event_source_id = uuid.uuid4().hex
        self._event_sequence = 0
        self._pending_events: deque[dict[str, object]] = deque()
        self._pool_event_callback = self._capture_pool_event
        if event_sink is not None:
            self._pool.set_membership_event_sink(self._pool_event_callback)
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def _capture_pool_event(self, transition: dict[str, object]) -> None:
        """Append one post-mutation snapshot without performing sink I/O."""
        self._event_sequence += 1
        replica_ids = self._pool.replica_ids
        healthy_by_id = self._pool.healthy_by_id()
        event = {
            **transition,
            # Capture the state-transition time, not the later outbox flush
            # time. The sink may be delayed or retried while placements already
            # consume this exact post-transition pool snapshot.
            "observed_ns": time.perf_counter_ns(),
            "event_source_id": self._event_source_id,
            "sequence": self._event_sequence,
            "event_id": (
                f"{self._event_source_id}:{self._event_sequence:020d}"
            ),
            "replica_ids": list(replica_ids),
            "healthy_ids": [
                replica_id
                for replica_id in replica_ids
                if healthy_by_id[replica_id]
            ],
            "eligible_ids": list(self._pool.eligible_ids),
            "generation_by_id": {
                replica_id: self._pool.entry_generation(replica_id)
                for replica_id in replica_ids
            },
        }
        self._pending_events.append(event)

    @staticmethod
    def _copy_event(event: Mapping[str, object]) -> dict[str, object]:
        copied: dict[str, object] = {}
        for key, value in event.items():
            if isinstance(value, list):
                copied[key] = list(value)
            elif isinstance(value, dict):
                copied[key] = dict(value)
            else:
                copied[key] = value
        return copied

    async def _flush_event_outbox(self) -> None:
        if self._event_sink is None:
            self._pending_events.clear()
            return
        while self._pending_events:
            event = self._pending_events[0]
            emitted = self._event_sink(self._copy_event(event))
            if inspect.isawaitable(emitted):
                await emitted
            self._pending_events.popleft()

    def _forget_tracking(self, replica_id: str) -> None:
        self._applied.pop(replica_id, None)
        self._draining.pop(replica_id, None)
        self._entry_generations.pop(replica_id, None)

    def _release_drain(self, replica_id: str) -> None:
        lease = self._draining.pop(replica_id, None)
        if lease is not None:
            self._pool.release_drain(replica_id, lease)

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
        if self._closed:
            raise RuntimeError("PoolReconciler is closed")
        await self._flush_event_outbox()
        try:
            result = await self._reconcile_once()
        except asyncio.CancelledError:
            raise
        except BaseException:
            try:
                await self._flush_event_outbox()
            except Exception:
                logger.exception(
                    "membership event flush failed after reconciliation error"
                )
            raise
        await self._flush_event_outbox()
        return result

    async def _reconcile_once(self) -> dict[str, list[str]]:
        polled = self._source.poll()
        if inspect.isawaitable(polled):
            polled = await polled
        desired = {
            replica_id: self._resolve_identity(replica_id, config)
            for replica_id, config in polled.items()
        }
        current_ids = self._pool.replica_ids
        current = set(current_ids)
        current_generations = {
            replica_id: self._pool.entry_generation(replica_id)
            for replica_id in current_ids
        }
        tracked = (
            set(self._applied) | set(self._draining) | set(self._entry_generations)
        )
        for replica_id in tracked - current:
            self._forget_tracking(replica_id)
        for replica_id in current_ids:
            generation = current_generations[replica_id]
            previous_generation = self._entry_generations.get(replica_id)
            if (
                previous_generation is not None
                and previous_generation != generation
            ):
                self._applied.pop(replica_id, None)
                self._draining.pop(replica_id, None)
            self._entry_generations[replica_id] = generation
            if replica_id not in desired:
                continue
            self._applied.setdefault(replica_id, desired[replica_id])

        added: list[str] = []
        draining: list[str] = []
        removed: list[str] = []

        for replica_id, identity in desired.items():
            if (
                replica_id in current
                and self._applied[replica_id] == identity
                and replica_id in self._draining
            ):
                self._release_drain(replica_id)

        for replica_id, identity in desired.items():
            if replica_id not in current:
                backend, health_url = self._factory(identity)
                try:
                    self._pool.add_replica(
                        replica_id, backend, health_url=health_url
                    )
                except BaseException:
                    await shutdown_all_cancellation_safe(
                        (backend,),
                        f"replica candidate {replica_id!r}",
                    )
                    raise
                self._applied[replica_id] = identity
                self._draining.pop(replica_id, None)
                self._entry_generations[replica_id] = self._pool.entry_generation(
                    replica_id
                )
                added.append(replica_id)

        for replica_id, identity in desired.items():
            if replica_id not in current or self._applied[replica_id] == identity:
                continue
            try:
                candidate, health_url = self._factory(identity)
            except Exception:
                if replica_id in self._draining:
                    self._release_drain(replica_id)
                draining.append(replica_id)
                continue

            manual_draining = self._pool.is_manually_draining(replica_id)
            if replica_id not in self._draining:
                self._draining[replica_id] = self._pool.acquire_drain(replica_id)
            try:
                await self._pool.remove_replica(replica_id)
            except RuntimeError:
                await shutdown_all_cancellation_safe(
                    (candidate,), f"replacement candidate {replica_id!r}"
                )
                draining.append(replica_id)
                continue
            except BaseException:
                if replica_id not in self._pool.replica_ids:
                    self._forget_tracking(replica_id)
                await shutdown_all_cancellation_safe(
                    (candidate,), f"replacement candidate {replica_id!r}"
                )
                raise

            self._forget_tracking(replica_id)
            try:
                if manual_draining:
                    self._pool.add_replica(
                        replica_id,
                        candidate,
                        health_url=health_url,
                        manual_draining=True,
                    )
                else:
                    self._pool.add_replica(
                        replica_id, candidate, health_url=health_url
                    )
            except BaseException:
                await shutdown_all_cancellation_safe(
                    (candidate,), f"replacement candidate {replica_id!r}"
                )
                raise
            self._applied[replica_id] = identity
            self._draining.pop(replica_id, None)
            self._entry_generations[replica_id] = self._pool.entry_generation(
                replica_id
            )
            removed.append(replica_id)
            added.append(replica_id)

        for replica_id in current_ids:
            if replica_id in desired:
                continue
            was_draining = self._pool.is_draining(replica_id)
            if replica_id not in self._draining:
                self._draining[replica_id] = self._pool.acquire_drain(replica_id)
            if not was_draining:
                draining.append(replica_id)
            try:
                await self._pool.remove_replica(replica_id)
            except RuntimeError:
                # in-flight work: drain holds, removal retries next tick (A6)
                if replica_id not in draining:
                    draining.append(replica_id)
            except BaseException:
                if replica_id not in self._pool.replica_ids:
                    self._forget_tracking(replica_id)
                raise
            else:
                self._forget_tracking(replica_id)
                removed.append(replica_id)
        result = {"added": added, "draining": draining, "removed": removed}
        return result

    async def run(self, interval_s: float) -> None:
        """Reconcile immediately and then periodically until cancelled."""
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        try:
            while True:
                try:
                    await self.reconcile()
                except Exception:
                    logger.exception("replica membership reconciliation failed")
                await asyncio.sleep(interval_s)
        finally:
            try:
                await self.aclose()
            except Exception:
                logger.exception("replica discovery shutdown failed")

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._aclose_impl())
        await _await_task_cancellation_safe(self._close_task)

    async def _aclose_impl(self) -> None:
        current = set(self._pool.replica_ids)
        for replica_id in tuple(self._draining):
            if replica_id in current:
                self._release_drain(replica_id)
            else:
                self._draining.pop(replica_id, None)
        cleanup_error: BaseException | None = None
        try:
            await self._flush_event_outbox()
        except BaseException as error:
            cleanup_error = error
        self._pool.clear_membership_event_sink(self._pool_event_callback)
        close = getattr(self._source, "aclose", None)
        if close is not None:
            try:
                closed = close()
                if inspect.isawaitable(closed):
                    await closed
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    logger.exception(
                        "replica discovery close failed after event flush error"
                    )
        self._closed = True
        if cleanup_error is not None:
            raise cleanup_error
