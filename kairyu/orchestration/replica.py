"""DP replica pool: N engine backends behind one ``EngineBackend`` (m5 D4, m10a D1).

``ReplicaPool`` is an L2 orchestration component — a sibling of the ``Router``
per the m5 D4 seam amendment. Placement policy, in order:

1. **Prefix-aware score** when enabled and declared by a non-empty root hint
   (or by a sessionless legacy caller) — maximize
   ``alpha * overlap - beta * outstanding`` over warm eligible replicas.
   Equal-score warm candidates use session HRW, or insertion order without a
   session. A warm/cold score tie prefers the reusable prefix.
2. **Cold session affinity** — requests carrying ``cache_hint.session_id`` map
   by rendezvous (HRW) hashing over the ELIGIBLE replicas (healthy ∧ not
   draining), so removing or draining one replica only remaps its sessions.
   A blank root hint takes this path directly without speculative prefix work.
3. **Load-skew valve** — if the prefix-aware or affine replica's
   outstanding-request count exceeds ``queue_depth_threshold``, fall back to
   least-outstanding.  A warm prefix remains sticky while it is within the
   configured queue budget, but cannot bypass the pool's overload policy.
4. **Least outstanding** for cold session-less traffic (ties break to
   insertion order).

m10a dynamic membership: entries are id-keyed; legacy sequence construction
auto-ids "0".."N-1" — HRW strings and Prometheus labels are IDENTICAL to the
index era (review A1). ``add_replica``/``drain``/``remove_replica`` never drop
in-flight work: drain stops NEW placements, completion on a removed id is a
no-op (A2), and remove refuses while outstanding > 0 unless forced.

Health: replicas with a declared readiness URL stay off the ring until ``probe()``
validates them. ``unhealthy_after`` consecutive failures eject a validated replica
until another probe (which resets failures but NEVER clears draining — A4).
Placement is pure hashing plus one optional JSONL append. No background tasks.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from kairyu.engine.backend import (
    AdmissionUpperBound,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    UpstreamClientError,
    backend_admission_upper_bound,
    backend_admission_upper_bound_async,
    backend_admission_upper_bound_key,
    backend_supports_prompt_kind,
    backend_supports_slo_defer,
    prepare_backend_request,
    shutdown_all,
    shutdown_all_cancellation_safe,
    validate_backend_request,
    validate_backend_request_before_prepare,
)
from kairyu.engine.prompt import prompt_kind, prompt_text
from kairyu.orchestration.kv_routing import KvRoutingIndex, PreparedKvRouting
from kairyu.orchestration.prefix_index import PrefixIndex, PreparedPrefixKeys
from kairyu.orchestration.router import JsonlRouterLog

_DEFAULT_UNHEALTHY_AFTER = 3
_DEFAULT_QUEUE_DEPTH_THRESHOLD = 8
_DECLARED_METADATA_FIELDS = (
    "model_revision",
    "max_model_len",
    "quantization_format",
    "cache_descriptor",
    "execution_mode",
    "tensor_parallel_size",
    "expert_parallel_size",
    "attention_data_parallel_size",
    "mtp_enabled",
    "dspark_enabled",
    "container_image_digest",
)
logger = logging.getLogger(__name__)


def _supports_native_prepared_prefix(index: object) -> bool:
    """Keep inherited legacy overrides on their original observe/hash path."""
    if not isinstance(index, PrefixIndex):
        return False
    return all(
        getattr(getattr(index, name, None), "__func__", None) is getattr(PrefixIndex, name)
        for name in (
            "candidate_ids",
            "chunk_keys",
            "observe",
            "observe_keys",
            "observe_prepared",
            "observe_root",
            "prepare",
            "register_replica",
        )
    )


def _rendezvous_score(session_id: str, replica_id: str) -> bytes:
    """HRW score for one (session, replica) pair; max over replicas wins."""
    return hashlib.sha256(f"{session_id}:{replica_id}".encode()).digest()


def _rendezvous_winner(
    session_id: str,
    replica_ids: Sequence[str],
    encoded_replica_ids: Mapping[str, bytes],
) -> str:
    """Return the byte-identical HRW winner without re-hashing the session prefix.

    SHA-256 is incremental, so the state after ``"<session>:"`` can be copied
    for every candidate.  This preserves the exact historical digest and
    first-candidate tie behavior while avoiding one formatted string and one
    UTF-8 encoding per replica on every placement.
    """

    prefix = hashlib.sha256()
    prefix.update(session_id.encode())
    prefix.update(b":")
    winner = replica_ids[0]
    winner_score: bytes | None = None
    for replica_id in replica_ids:
        candidate = prefix.copy()
        candidate.update(encoded_replica_ids[replica_id])
        score = candidate.digest()
        if winner_score is None or score > winner_score:
            winner = replica_id
            winner_score = score
    return winner


@dataclass
class _ReplicaEntry:
    backend: EngineBackend
    health_url: str | None = None
    validated: bool = True
    generation: str = field(
        default_factory=lambda: uuid.uuid4().hex,
        compare=False,
    )
    outstanding: int = 0
    consecutive_failures: int = 0
    manual_draining: bool = False
    drain_leases: set[DrainLease] = field(default_factory=set)
    removed: bool = field(default=False, compare=False)

    @property
    def draining(self) -> bool:
        return self.manual_draining or bool(self.drain_leases)


@dataclass
class _PreparedPlacement:
    request_ref: weakref.ReferenceType[GenerationRequest]
    candidates: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _PreparationFlight:
    request_ref: weakref.ReferenceType[GenerationRequest]
    task: asyncio.Task[tuple[str, ...]]


class DrainLease:
    """Opaque ownership token for one independently reversible pool drain."""

    __slots__ = ()


class _ReplicaRequestValidationError(ValueError):
    """Aggregated placeable-replica rejection with a stable public code."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ReplicaPool:
    """Routes requests across N ``EngineBackend`` replicas; itself an ``EngineBackend``."""

    def __init__(
        self,
        replicas: Sequence[EngineBackend] | Mapping[str, EngineBackend],
        *,
        log: JsonlRouterLog | None = None,
        unhealthy_after: int = _DEFAULT_UNHEALTHY_AFTER,
        queue_depth_threshold: int = _DEFAULT_QUEUE_DEPTH_THRESHOLD,
        prefix_index=None,
        prefix_alpha: float = 1.0,
        prefix_beta: float = 0.25,
        allow_empty: bool = False,
        membership_event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if len(replicas) < 1 and not allow_empty:
            raise ValueError("ReplicaPool requires at least 1 replica")
        if unhealthy_after < 1:
            raise ValueError(f"unhealthy_after must be >= 1, got {unhealthy_after}")
        if queue_depth_threshold < 0:
            raise ValueError(f"queue_depth_threshold must be >= 0, got {queue_depth_threshold}")
        if isinstance(replicas, Mapping):
            items = list(replicas.items())
        else:  # legacy sequence: auto-ids match the old index labels (A1)
            items = [(str(index), backend) for index, backend in enumerate(replicas)]
        self._entries: dict[str, _ReplicaEntry] = {
            replica_id: _ReplicaEntry(backend=backend) for replica_id, backend in items
        }
        # Placement is far hotter than membership mutation.  Keep immutable
        # snapshots and stable metadata at the mutation boundary so a
        # 200-replica request does not repeatedly scan/encode the same ring.
        self._encoded_replica_ids = {
            replica_id: replica_id.encode() for replica_id in self._entries
        }
        self._ordinal_by_id = {
            replica_id: ordinal for ordinal, replica_id in enumerate(self._entries)
        }
        self._eligible_snapshot = tuple(self._entries)
        self._log = log
        self._unhealthy_after = unhealthy_after
        self._queue_depth_threshold = queue_depth_threshold
        # m10b D6: KV-aware scoring is OPT-IN (prefix_index=None keeps m5
        # behavior byte-identical); score = alpha*overlap - beta*outstanding
        self._prefix_index = prefix_index
        self._prefix_alpha = prefix_alpha
        self._prefix_beta = prefix_beta
        native_prepared = isinstance(
            prefix_index, KvRoutingIndex
        ) or _supports_native_prepared_prefix(prefix_index)
        self._prefix_prepare = prefix_index.prepare if native_prepared else None
        self._prefix_overlap_keys = prefix_index.overlap_keys if native_prepared else None
        self._prefix_observe_prepared = prefix_index.observe_prepared if native_prepared else None
        self._prefix_observe_root = prefix_index.observe_root if native_prepared else None
        self._prefix_register_replica = prefix_index.register_replica if native_prepared else None
        if self._prefix_register_replica is not None:
            for replica_id in self._entries:
                self._prefix_register_replica(replica_id)
        self._membership_event_sink = membership_event_sink
        self._decision_counts: dict[str, int] = {
            "session_affinity": 0,
            "queue_depth_fallback": 0,
            "least_outstanding": 0,
            "prefix_match": 0,
            "kv_event_match": 0,
        }
        # Request-identity placement leases bind a successful async preflight
        # to the exact eligible generations it covered. Newly eligible members
        # cannot slip into placement after HTTP headers are committed.
        self._prepared_placements: dict[int, _PreparedPlacement] = {}
        # Exact-request preparation is single-flight. Besides avoiding duplicate
        # tokenizer work, this prevents a later concurrent snapshot from
        # overwriting an already narrowed placement lease with new members.
        self._preparation_flights: dict[int, _PreparationFlight] = {}
        # m13: cached /backends payload of one replica for the gateway's own
        # /backends aggregation. Only successful probes are cached (a process's
        # attention backend is constant for its lifetime); None stays uncached.
        self._backends_cache: dict | None = None

    @property
    def generation_defaults(self):
        """Expose one resolved policy only when every local replica agrees."""

        values = tuple(
            getattr(entry.backend, "generation_defaults", None)
            for entry in self._entries.values()
        )
        if not values or values[0] is None:
            return None
        return values[0] if all(value == values[0] for value in values[1:]) else None

    def declared_metadata_snapshot(self) -> dict[str, object]:
        """Return constructor metadata only when every current replica agrees.

        External engines such as vLLM do not expose Kairyu's ``/backends``
        endpoint. Their OpenAI-compatible adapters still carry the pinned
        deployment metadata, so a gateway can publish that unanimous contract
        without pretending it was probed from the upstream runtime.
        """

        backends = tuple(entry.backend for entry in self._entries.values())
        if not backends:
            return {}
        metadata: dict[str, object] = {}
        for metadata_field in _DECLARED_METADATA_FIELDS:
            values = tuple(
                getattr(backend, metadata_field, None) for backend in backends
            )
            first = values[0]
            if first is None or not all(value == first for value in values[1:]):
                continue
            metadata[metadata_field] = (
                dict(first) if isinstance(first, Mapping) else first
            )
        return metadata

    # -- membership (m10a D1) -------------------------------------------------

    @property
    def replica_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def decode_graph_metric_sources_snapshot(
        self,
    ) -> tuple[tuple[str, str, int | str | None, dict[str, object] | None], ...]:
        """Return generation-stable CUDA graph metric sources in this pool.

        The serve metrics collector owns counter continuity across backend
        restarts and membership changes.  Keeping the replica generation next
        to each snapshot lets it distinguish a restarted/re-added backend from
        the prior process without exposing mutable pool entries.  A source that
        temporarily fails to answer remains present with a ``None`` snapshot so
        one failed scrape cannot retire its last observed counter value.
        """

        sources: list[
            tuple[str, str, int | str | None, dict[str, object] | None]
        ] = []
        for replica_id, entry in tuple(self._entries.items()):
            getter = getattr(entry.backend, "decode_graph_metadata_snapshot", None)
            if not callable(getter):
                continue
            generation_getter = getattr(
                entry.backend,
                "decode_graph_metric_generation_snapshot",
                None,
            )
            service_generation: int | str | None = None
            if callable(generation_getter):
                service_generation = generation_getter()
                if (
                    service_generation is not None
                    and type(service_generation) not in (int, str)
                ):
                    raise TypeError("invalid graph metric service generation")
            try:
                snapshot = getter()
            except Exception:
                snapshot = None
            if isinstance(snapshot, Mapping):
                snapshot = dict(snapshot)
            else:
                snapshot = None
            sources.append(
                (replica_id, entry.generation, service_generation, snapshot)
            )
        return tuple(sources)

    def add_replica(
        self,
        replica_id: str,
        backend: EngineBackend,
        health_url: str | None = None,
        *,
        manual_draining: bool = False,
    ) -> None:
        if replica_id in self._entries:
            raise ValueError(f"replica {replica_id!r} already in the pool")
        self._entries[replica_id] = _ReplicaEntry(
            backend=backend,
            health_url=health_url,
            validated=health_url is None,
            manual_draining=manual_draining,
        )
        self._encoded_replica_ids[replica_id] = replica_id.encode()
        self._ordinal_by_id[replica_id] = len(self._entries) - 1
        if self._prefix_register_replica is not None:
            self._prefix_register_replica(replica_id)
        self._refresh_eligible_snapshot()
        entry = self._entries[replica_id]
        self._emit_membership_event(
            "replica_added",
            replica_id,
            entry.generation,
            added=(replica_id,),
            eligible=((replica_id,) if self._is_eligible(entry) else ()),
        )

    def drain(self, replica_id: str) -> None:
        """Acquire the manual drain owner; in-flight requests complete normally."""
        entry = self._entry(replica_id)
        was_eligible = self._is_eligible(entry)
        entry.manual_draining = True
        if was_eligible and not self._is_eligible(entry):
            self._refresh_eligible_snapshot()
            self._emit_membership_event(
                "manual_drain",
                replica_id,
                entry.generation,
                draining=(replica_id,),
                ineligible=(replica_id,),
            )

    def cancel_drain(self, replica_id: str) -> None:
        """Release only the manual drain owner without changing other state."""
        entry = self._entry(replica_id)
        was_eligible = self._is_eligible(entry)
        entry.manual_draining = False
        if not was_eligible and self._is_eligible(entry):
            self._refresh_eligible_snapshot()
            self._emit_membership_event(
                "manual_undrain",
                replica_id,
                entry.generation,
                undraining=(replica_id,),
                eligible=(replica_id,),
            )

    def acquire_drain(self, replica_id: str) -> DrainLease:
        """Acquire an independently reversible drain lease."""
        entry = self._entry(replica_id)
        was_eligible = self._is_eligible(entry)
        lease = DrainLease()
        entry.drain_leases.add(lease)
        if was_eligible and not self._is_eligible(entry):
            self._refresh_eligible_snapshot()
            self._emit_membership_event(
                "reconciler_drain",
                replica_id,
                entry.generation,
                draining=(replica_id,),
                ineligible=(replica_id,),
            )
        return lease

    def release_drain(self, replica_id: str, lease: DrainLease) -> None:
        """Release only *lease*; manual and other lease owners remain active."""
        entry = self._entry(replica_id)
        was_eligible = self._is_eligible(entry)
        entry.drain_leases.discard(lease)
        if not was_eligible and self._is_eligible(entry):
            self._refresh_eligible_snapshot()
            self._emit_membership_event(
                "reconciler_undrain",
                replica_id,
                entry.generation,
                undraining=(replica_id,),
                eligible=(replica_id,),
            )

    def is_draining(self, replica_id: str) -> bool:
        return self._entry(replica_id).draining

    def is_manually_draining(self, replica_id: str) -> bool:
        """Whether the manual owner is active, independent of drain leases."""
        return self._entry(replica_id).manual_draining

    async def remove_replica(self, replica_id: str, force: bool = False) -> None:
        entry = self._entry(replica_id)
        if entry.outstanding > 0 and not force:
            raise RuntimeError(
                f"replica {replica_id!r} has {entry.outstanding} in-flight requests; "
                "drain first (or force=True)"
            )
        entry.removed = True  # guarded decrement: late completions are no-ops (A2)
        del self._entries[replica_id]
        self._encoded_replica_ids.pop(replica_id)
        self._ordinal_by_id = {
            current_id: ordinal for ordinal, current_id in enumerate(self._entries)
        }
        self._refresh_eligible_snapshot()
        # drop this id's prefix history (M5): a re-added replica with the same id
        # must not inherit phantom prefixes and route shared traffic to a cold cache
        if self._prefix_index is not None:
            self._prefix_index.forget_replica(replica_id)
        self._emit_membership_event(
            "replica_removed",
            replica_id,
            entry.generation,
            removed=(replica_id,),
        )
        await shutdown_all_cancellation_safe(
            (entry.backend,),
            f"replica {replica_id!r}",
        )

    def health_url(self, replica_id: str) -> str | None:
        return self._entry(replica_id).health_url

    def require_probe(self, replica_id: str) -> None:
        """Mark an existing id unknown until a readiness probe succeeds."""
        entry = self._entry(replica_id)
        was_eligible = self._is_eligible(entry)
        entry.validated = False
        if was_eligible and not self._is_eligible(entry):
            self._refresh_eligible_snapshot()
            self._emit_membership_event(
                "probe_required",
                replica_id,
                entry.generation,
                ineligible=(replica_id,),
            )

    def entry_generation(self, replica_id: str) -> str:
        """Serializable token that changes whenever a replica ID is re-added."""
        return self._entry(replica_id).generation

    def set_membership_event_sink(
        self,
        sink: Callable[[dict[str, object]], None] | None,
    ) -> None:
        """Install a non-blocking observer for effective membership changes."""
        self._membership_event_sink = sink

    def clear_membership_event_sink(
        self,
        sink: Callable[[dict[str, object]], None],
    ) -> None:
        if self._membership_event_sink is sink:
            self._membership_event_sink = None

    def _emit_membership_event(
        self,
        reason: str,
        replica_id: str,
        generation: str,
        *,
        added: tuple[str, ...] = (),
        draining: tuple[str, ...] = (),
        undraining: tuple[str, ...] = (),
        removed: tuple[str, ...] = (),
        eligible: tuple[str, ...] = (),
        ineligible: tuple[str, ...] = (),
    ) -> None:
        sink = self._membership_event_sink
        if sink is None:
            return
        try:
            sink(
                {
                    "reason": reason,
                    "replica_id": replica_id,
                    "replica_generation": generation,
                    "added": list(added),
                    "draining": list(draining),
                    "undraining": list(undraining),
                    "removed": list(removed),
                    "eligible": list(eligible),
                    "ineligible": list(ineligible),
                }
            )
        except Exception:
            # Membership observers are downstream audit consumers. A broken
            # observer must not turn an already-applied pool mutation into a
            # false failure or strand a detached backend before shutdown.
            logger.exception(
                "membership event observer failed for %s (%s)",
                replica_id,
                reason,
            )

    def _entry(self, replica_id: str) -> _ReplicaEntry:
        entry = self._entries.get(replica_id)
        if entry is None:
            raise ValueError(f"unknown replica {replica_id!r}")
        return entry

    def _resolve_id(self, key: int | str) -> str:
        if isinstance(key, str):
            return key
        ids = tuple(self._entries)
        if not 0 <= key < len(ids):
            raise ValueError(f"replica index {key} out of range [0, {len(ids)})")
        return ids[key]

    # -- read-only surface (insertion order — A1) -----------------------------

    @property
    def outstanding(self) -> tuple[int, ...]:
        """In-flight request count per replica (dispatch-incremented, completion-decremented)."""
        return tuple(entry.outstanding for entry in self._entries.values())

    @property
    def replica_count(self) -> int:
        return len(self._entries)

    @property
    def healthy(self) -> tuple[bool, ...]:
        """Per-replica health (read-only; consumed by /readyz, /metrics, the prober)."""
        return tuple(self._is_healthy(entry) for entry in self._entries.values())

    def healthy_by_id(self) -> dict[str, bool]:
        return dict(zip(self._entries, self.healthy, strict=True))

    def membership_snapshot(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        dict[str, str],
    ]:
        """Capture one internally consistent state for audit publication."""

        replica_ids = []
        healthy_ids = []
        generation_by_id = {}
        for replica_id, entry in self._entries.items():
            replica_ids.append(replica_id)
            if self._is_healthy(entry):
                healthy_ids.append(replica_id)
            generation_by_id[replica_id] = entry.generation
        return (
            tuple(replica_ids),
            tuple(healthy_ids),
            self._eligible_snapshot,
            generation_by_id,
        )

    def validated_by_id(self) -> dict[str, bool]:
        """Read-only probe-validation state keyed by stable replica id."""
        return {rid: entry.validated for rid, entry in self._entries.items()}

    def draining_by_id(self) -> dict[str, bool]:
        return {rid: entry.draining for rid, entry in self._entries.items()}

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        return self._eligible_ids()

    def outstanding_by_id(self) -> dict[str, int]:
        return {rid: entry.outstanding for rid, entry in self._entries.items()}

    @property
    def decision_counts(self) -> dict[str, int]:
        """Cumulative placement decisions by reason (read-only, for /metrics)."""
        return dict(self._decision_counts)

    async def probe(self, key: int | str) -> None:
        """Declare a replica healthy again, returning it to the hash ring.

        Accepts the legacy ordinal or the replica id (A1). Validates the entry
        and resets its failure count — a probe NEVER clears draining (A4).
        """
        replica_id = self._resolve_id(key)
        entry = self._entry(replica_id)
        was_eligible = self._is_eligible(entry)
        entry.validated = True
        entry.consecutive_failures = 0
        if not was_eligible and self._is_eligible(entry):
            self._refresh_eligible_snapshot()
            self._emit_membership_event(
                "probe_succeeded",
                replica_id,
                entry.generation,
                eligible=(replica_id,),
            )

    async def probe_backends(self) -> dict | None:
        """Best-effort ``/backends`` payload of ONE replica, for the gateway's own
        ``/backends`` aggregation (m13).

        The gateway process runs no local attention (it forwards), so it cannot
        name the kernel its replicas use; it asks one. Prefers a healthy replica
        but falls back to any (this is introspection, not request placement).
        Delegates the actual fetch to the replica backend (``fetch_backends`` —
        the pool holds no URLs itself), and caches the first success since a
        replica's attention backend is constant for its lifetime. Returns ``None``
        if no replica exposes ``/backends`` (e.g. non-kairyu or unreachable)."""
        if self._backends_cache is not None:
            return self._backends_cache
        healthy_first = [rid for rid, e in self._entries.items() if self._is_healthy(e)]
        rest = [rid for rid in self._entries if rid not in healthy_first]
        for rid in healthy_first + rest:
            fetch = getattr(self._entries[rid].backend, "fetch_backends", None)
            if fetch is None:
                continue
            payload = await fetch()
            if payload is not None:
                self._backends_cache = payload
                return payload
        return None

    # -- placement -------------------------------------------------------------

    def _is_healthy(self, entry: _ReplicaEntry) -> bool:
        return entry.validated and entry.consecutive_failures < self._unhealthy_after

    def _is_eligible(self, entry: _ReplicaEntry) -> bool:
        return self._is_healthy(entry) and not entry.draining

    def _eligible_ids(self) -> tuple[str, ...]:
        return self._eligible_snapshot

    @property
    def supports_slo_defer(self) -> bool:
        """Whether every placeable replica isolates running deferred work."""

        return self._candidates_support_slo_defer(self._eligible_ids())

    def supports_slo_defer_for_request(self, request: GenerationRequest) -> bool:
        """Check the exact prepared placement lease, or current candidates."""

        candidates = self._prepared_placement(request, consume=False)
        if candidates is None:
            candidates = self._eligible_ids()
        return self._candidates_support_slo_defer(candidates, request=request)

    def supports_prompt_kind(self, kind: str) -> bool:
        """Expose the intersection of every currently placeable replica."""

        candidates = self._eligible_ids()
        return bool(candidates) and all(
            backend_supports_prompt_kind(self._entries[replica_id].backend, kind)
            for replica_id in candidates
        )

    def _candidates_support_slo_defer(
        self,
        candidates: tuple[str, ...],
        *,
        request: GenerationRequest | None = None,
    ) -> bool:
        if not candidates:
            return False
        return all(
            backend_supports_slo_defer(
                self._entries[replica_id].backend,
                request,
            )
            for replica_id in candidates
        )

    def _refresh_eligible_snapshot(self) -> None:
        self._eligible_snapshot = tuple(
            rid for rid, entry in self._entries.items() if self._is_eligible(entry)
        )

    def validate_request(self, request: GenerationRequest) -> None:
        """Validate against the intersection of all placeable replicas.

        Placement may select any eligible member after preflight, so accepting
        a request that only some members can execute would make behavior depend
        on load or affinity. Backends without a validator remain compatible
        with legacy text but fail closed for every typed prompt. A backend may
        explicitly publish an immutable ``request_validation_key``; equal keys
        on the same backend type promise identical validation semantics, so one
        representative is sufficient. Unknown or unhashable keys retain
        per-replica validation.
        """

        self._validate_request_candidates(request, self._eligible_ids())

    def validate_request_before_prepare(self, request: GenerationRequest) -> None:
        """Validate the current replica intersection without deep CPU work."""

        self._validate_request_candidates(
            request,
            self._eligible_ids(),
            before_prepare=True,
        )

    def _validate_request_candidates(
        self,
        request: GenerationRequest,
        candidates: tuple[str, ...],
        *,
        before_prepare: bool = False,
    ) -> None:
        """Validate against one immutable pre-placement membership snapshot."""

        failures: list[tuple[str, ValueError]] = []
        validated_keys: set[tuple[type, object]] = set()
        for replica_id in candidates:
            backend = self._entries[replica_id].backend
            validate = getattr(backend, "validate_request", None)
            fast_validate = getattr(
                backend,
                "validate_request_before_prepare",
                None,
            )
            if not callable(validate) and not (
                before_prepare and callable(fast_validate)
            ):
                try:
                    if before_prepare:
                        validate_backend_request_before_prepare(backend, request)
                    else:
                        validate_backend_request(backend, request)
                except ValueError as error:
                    failures.append((replica_id, error))
                continue
            validation_key = getattr(backend, "request_validation_key", None)
            if validation_key is not None:
                typed_key = (type(backend), validation_key)
                try:
                    if typed_key in validated_keys:
                        continue
                    validated_keys.add(typed_key)
                except TypeError:
                    # A malformed optional key must not weaken validation.
                    # Validate this backend independently instead.
                    pass
            try:
                if before_prepare:
                    validate_backend_request_before_prepare(backend, request)
                else:
                    validate(request)
            except ValueError as error:
                failures.append((replica_id, error))
        if failures:
            codes = {
                getattr(error, "code", "invalid_request")
                for _replica_id, error in failures
            }
            logger.warning(
                "request rejected by placeable replica contract",
                extra={
                    "replica_validation_failures": [
                        f"{replica_id}: {error}"
                        for replica_id, error in failures
                    ]
                },
            )
            messages = {str(error) for _replica_id, error in failures}
            public_message = "request is not supported by every eligible replica"
            if len(messages) == 1:
                public_message += f": {messages.pop()}"
            raise _ReplicaRequestValidationError(
                public_message,
                code=codes.pop() if len(codes) == 1 else "invalid_request",
            )

    def admission_upper_bound(self, request: GenerationRequest) -> AdmissionUpperBound:
        """Return the safe maximum across every currently placeable replica."""

        prepared = self._prepared_placement(request, consume=False)
        eligible = self._eligible_ids() if prepared is None else prepared
        if not eligible:
            raise RuntimeError(
                f"ReplicaPool: none of the {len(self._entries)} replicas are eligible"
            )
        backends = self._admission_backends(eligible)
        bounds = [
            backend_admission_upper_bound(
                backend,
                request,
            )
            for backend in backends
        ]
        return AdmissionUpperBound(
            tokens=max(bound.tokens for bound in bounds),
            refundable_on_exact_usage=all(
                bound.refundable_on_exact_usage for bound in bounds
            ),
        )

    async def admission_upper_bound_async(
        self,
        request: GenerationRequest,
    ) -> AdmissionUpperBound:
        """Bound the monotonic prepared lease despite rolling membership changes."""

        if self._prepared_placement(request, consume=False) is None:
            await self.prepare_request(request)

        # Every restart strictly shrinks the prepared lease. A removed,
        # replaced, or drained representative may fail while its async bound is
        # running; that failure cannot reject a request which still has a
        # prepared survivor. Recompute from the survivor contracts instead.
        while True:
            eligible = self._prepared_placement(request, consume=False)
            if not eligible:
                raise RuntimeError(
                    f"ReplicaPool: none of the {len(self._entries)} replicas are eligible"
                )
            members = self._admission_members(eligible)
            bounds: list[AdmissionUpperBound] = []
            restart = False
            for replica_id, entry, backend in members:
                current = self._prepared_placement(request, consume=False)
                if current != eligible or self._entries.get(replica_id) is not entry:
                    restart = True
                    break
                try:
                    bound = await backend_admission_upper_bound_async(
                        backend,
                        request,
                    )
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
                    surviving = self._prepared_placement(request, consume=False)
                    if (
                        surviving is not None
                        and replica_id in surviving
                        and self._entries.get(replica_id) is entry
                    ):
                        raise
                    restart = True
                    break
                except Exception:
                    surviving = self._prepared_placement(request, consume=False)
                    if (
                        surviving is not None
                        and replica_id in surviving
                        and self._entries.get(replica_id) is entry
                    ):
                        raise
                    restart = True
                    break

                surviving = self._prepared_placement(request, consume=False)
                if surviving != eligible:
                    restart = True
                    break
                bounds.append(bound)
            if restart:
                continue
            return AdmissionUpperBound(
                tokens=max(bound.tokens for bound in bounds),
                refundable_on_exact_usage=all(
                    bound.refundable_on_exact_usage for bound in bounds
                ),
            )

    def _admission_backends(
        self,
        eligible: tuple[str, ...],
    ) -> tuple[EngineBackend, ...]:
        """Keep one representative per explicit immutable bound contract."""

        representatives: list[EngineBackend] = []
        seen: set[object] = set()
        for replica_id in eligible:
            backend = self._entries[replica_id].backend
            key = backend_admission_upper_bound_key(backend)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            representatives.append(backend)
        return tuple(representatives)

    def _admission_members(
        self,
        eligible: tuple[str, ...],
    ) -> tuple[tuple[str, _ReplicaEntry, EngineBackend], ...]:
        """Retain representative identity so async bounds can revalidate it."""

        representatives: list[tuple[str, _ReplicaEntry, EngineBackend]] = []
        seen: set[object] = set()
        for replica_id in eligible:
            entry = self._entries[replica_id]
            backend = entry.backend
            key = backend_admission_upper_bound_key(backend)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            representatives.append((replica_id, entry, backend))
        return tuple(representatives)

    def _prepared_placement(
        self,
        request: GenerationRequest,
        *,
        consume: bool,
    ) -> tuple[str, ...] | None:
        key = id(request)
        prepared = self._prepared_placements.get(key)
        if prepared is None or prepared.request_ref() is not request:
            self._prepared_placements.pop(key, None)
            return None
        surviving = tuple(
            (replica_id, generation)
            for replica_id, generation in prepared.candidates
            if (
                (entry := self._entries.get(replica_id)) is not None
                and entry.generation == generation
                and self._is_eligible(entry)
            )
        )
        # A lease may shrink as members leave eligibility, but the exact same
        # generation can never re-enter after admission observed it absent.
        prepared.candidates = surviving
        if consume:
            self._prepared_placements.pop(key, None)
        return tuple(replica_id for replica_id, _generation in surviving)

    def _retain_prepared_placement(
        self,
        request: GenerationRequest,
        candidates: tuple[tuple[str, str], ...],
    ) -> None:
        key = id(request)

        def discard(reference: weakref.ReferenceType[GenerationRequest]) -> None:
            retained = self._prepared_placements.get(key)
            if retained is not None and retained.request_ref is reference:
                self._prepared_placements.pop(key, None)

        retained = self._prepared_placements.get(key)
        if retained is not None and retained.request_ref() is request:
            candidate_set = set(candidates)
            # A late preflight result may only confirm members which remain in
            # the lease; it can never re-add a member admission already removed.
            retained.candidates = tuple(
                candidate
                for candidate in retained.candidates
                if candidate in candidate_set
            )
            return
        self._prepared_placements[key] = _PreparedPlacement(
            weakref.ref(request, discard),
            candidates,
        )

    async def prepare_request(
        self,
        request: GenerationRequest,
    ) -> tuple[str, ...]:
        """Single-flight preparation of exact replica generations."""

        cached = self._prepared_placement(request, consume=False)
        if cached is not None:
            return cached
        key = id(request)
        flight = self._preparation_flights.get(key)
        if flight is None or flight.request_ref() is not request:
            self._preparation_flights.pop(key, None)
            task = asyncio.create_task(self._prepare_request_once(request))
            reference = weakref.ref(request)
            flight = _PreparationFlight(reference, task)
            self._preparation_flights[key] = flight

            def finish(completed: asyncio.Task[tuple[str, ...]]) -> None:
                current = self._preparation_flights.get(key)
                if current is not None and current.task is completed:
                    self._preparation_flights.pop(key, None)
                # A sole waiter may be cancelled while shielded preparation
                # safely finishes. Retrieve any exception in that case.
                if not completed.cancelled():
                    completed.exception()

            task.add_done_callback(finish)
        return await asyncio.shield(flight.task)

    async def _prepare_request_once(
        self,
        request: GenerationRequest,
    ) -> tuple[str, ...]:
        """Prepare one immutable membership snapshot and publish its lease."""

        eligible = self._eligible_ids()
        if not eligible:
            raise RuntimeError(
                f"ReplicaPool: none of the {len(self._entries)} replicas are eligible "
                f"(unvalidated, unhealthy after >= {self._unhealthy_after} consecutive "
                "failures, or draining); call probe() once a replica recovers"
            )
        snapshot = tuple(
            (replica_id, self._entries[replica_id])
            for replica_id in eligible
        )
        self._validate_request_candidates(
            request,
            eligible,
            before_prepare=True,
        )
        prepared_entries: list[tuple[str, _ReplicaEntry]] = []
        for _replica_id, entry in snapshot:
            if (
                self._entries.get(_replica_id) is not entry
                or not self._is_eligible(entry)
            ):
                continue
            backend = entry.backend
            prepare = getattr(backend, "prepare_request", None)
            if not callable(prepare):
                prepared_entries.append((_replica_id, entry))
                continue
            try:
                await prepare_backend_request(backend, request)
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                if (
                    self._entries.get(_replica_id) is entry
                    and self._is_eligible(entry)
                ):
                    raise
            except Exception:
                if (
                    self._entries.get(_replica_id) is entry
                    and self._is_eligible(entry)
                ):
                    raise
            else:
                prepared_entries.append((_replica_id, entry))
        candidates = tuple(
            (replica_id, entry.generation)
            for replica_id, entry in prepared_entries
            if (
                self._entries.get(replica_id) is entry
                and self._is_eligible(entry)
            )
        )
        if not candidates:
            raise RuntimeError(
                "ReplicaPool: every replica eligible at preflight became "
                "unavailable before placement"
            )
        self._retain_prepared_placement(request, candidates)
        return tuple(replica_id for replica_id, _generation in candidates)

    def _least_outstanding(self, candidates: Sequence[str]) -> str:
        # ``candidates`` is already in insertion order and ``min`` keeps the
        # first equal key, preserving the historical stable tie break.
        return min(candidates, key=lambda rid: self._entries[rid].outstanding)

    def _apply_queue_depth_valve(
        self,
        selected: str,
        eligible: tuple[str, ...],
        reason: str,
        session_id: str | None,
    ) -> tuple[str, str, str | None]:
        """Overflow a busy affinity choice without weakening idle KV reuse."""

        if self._entries[selected].outstanding > self._queue_depth_threshold:
            return self._least_outstanding(eligible), "queue_depth_fallback", session_id
        return selected, reason, session_id

    def _select(
        self,
        request: GenerationRequest,
        *,
        prefix_keys: str | Sequence[str] | PreparedKvRouting | None = None,
        track_prefix: bool | None = None,
        eligible: tuple[str, ...] | None = None,
    ) -> tuple[str, str, str | None]:
        """Pick a replica; returns (replica_id, reason, session_id). Pure hashing."""
        eligible = self._eligible_ids() if eligible is None else eligible
        if not eligible:
            raise RuntimeError(
                f"ReplicaPool: none of the {len(self._entries)} replicas are eligible "
                f"(unvalidated, unhealthy after >= {self._unhealthy_after} consecutive "
                "failures, or draining); call probe() once a replica recovers"
            )
        session_id = request.cache_hint.session_id if request.cache_hint else None
        if track_prefix is None:
            track_prefix = self._should_track_prefix(request)
        if track_prefix:
            prefix_prompt = prompt_text(request.prompt)
            if prefix_prompt is None:  # Defensive: _should_track_prefix owns this invariant.
                raise RuntimeError("prefix tracking requires a text prompt")
            prefix_reason = "prefix_match"
            if isinstance(prefix_keys, PreparedKvRouting):
                view = self._prefix_index.route_view(eligible, prefix_keys)
                if view.mode == "exact":
                    best = self._kv_event_select(
                        eligible,
                        view.overlaps,
                        session_id=session_id or None,
                    )
                    prefix_reason = "kv_event_match"
                else:
                    approximate = prefix_keys.approximate
                    best = (
                        None
                        if isinstance(approximate, str)
                        or (
                            isinstance(approximate, PreparedPrefixKeys)
                            and not approximate.candidate_ids
                        )
                        else self._prefix_select(
                            eligible,
                            prefix_prompt,
                            session_id=session_id or None,
                            prepared_keys=approximate,
                        )
                    )
            else:
                best = (
                    None
                    if isinstance(prefix_keys, str)
                    or (
                        isinstance(prefix_keys, PreparedPrefixKeys)
                        and not prefix_keys.candidate_ids
                    )
                    else self._prefix_select(
                        eligible,
                        prefix_prompt,
                        session_id=session_id or None,
                        prepared_keys=prefix_keys,
                    )
                )
            if best is not None:
                return self._apply_queue_depth_valve(
                    best,
                    eligible,
                    prefix_reason,
                    session_id,
                )
        if session_id:
            affine = _rendezvous_winner(
                session_id,
                eligible,
                self._encoded_replica_ids,
            )
            return self._apply_queue_depth_valve(
                affine,
                eligible,
                "session_affinity",
                session_id,
            )
        return self._least_outstanding(eligible), "least_outstanding", None

    def _should_track_prefix(self, request: GenerationRequest) -> bool:
        """Whether this request declares enough affinity for native tracking.

        A blank CacheHint already has session affinity but declares no reusable
        cross-session root. Charging that common path for speculative hashing
        and publication only lowers goodput. Sessionless callers retain the
        legacy local-prefix discovery path, while non-empty hints opt in.
        """
        if self._prefix_index is None or prompt_kind(request.prompt) != "text":
            return False
        return not (
            self._prefix_prepare is not None
            and request.cache_hint is not None
            and not request.cache_hint.prefix_fingerprint
        )

    def _prefix_select(
        self,
        eligible: tuple[str, ...],
        prompt: str,
        *,
        session_id: str | None = None,
        prepared_keys: str | Sequence[str] | None = None,
    ) -> str | None:
        """alpha*overlap - beta*outstanding; None when no candidate has overlap
        (fall through to the existing cold-placement policy).

        Candidate order remains the tie break for requests without a session.
        A session chooses its HRW winner among equal-score candidates, so
        affinity remains deterministic without bypassing prefix-aware scoring.
        """
        if prepared_keys is None and self._prefix_prepare is not None:
            prepared_keys = self._prefix_prepare(prompt)
        if isinstance(prepared_keys, str):
            return None
        native_precomputed = (
            isinstance(prepared_keys, PreparedPrefixKeys) and self._prefix_overlap_keys is not None
        )
        advertised: tuple[str, ...] | None = None
        if native_precomputed:
            keys = prepared_keys
            overlap_keys = self._prefix_overlap_keys
            use_precomputed = True
            advertised = prepared_keys.candidate_ids
        else:
            chunk_keys = getattr(self._prefix_index, "chunk_keys", None)
            overlap_keys = getattr(self._prefix_index, "overlap_keys", None)
            candidate_ids = getattr(self._prefix_index, "candidate_ids", None)
            use_precomputed = callable(chunk_keys) and callable(overlap_keys)
            keys = chunk_keys(prompt) if use_precomputed else None
            if use_precomputed and callable(candidate_ids):
                advertised = candidate_ids(keys)
        score_candidates = eligible
        if advertised is not None:
            if not advertised:
                return None
            if eligible is self._eligible_snapshot:
                score_candidates = tuple(
                    sorted(
                        (
                            rid
                            for rid in advertised
                            if rid in self._entries and self._is_eligible(self._entries[rid])
                        ),
                        key=self._ordinal_by_id.__getitem__,
                    )
                )
            else:
                advertised_set = set(advertised)
                score_candidates = tuple(rid for rid in eligible if rid in advertised_set)
            if not score_candidates:
                return None
        best_score: float | None = None
        tied: list[tuple[str, int]] = []
        for rid in score_candidates:
            if use_precomputed:
                overlap = overlap_keys(rid, keys)
            else:
                overlap = self._prefix_index.overlap(rid, prompt)
            score = (
                self._prefix_alpha * overlap - self._prefix_beta * self._entries[rid].outstanding
            )
            if best_score is None or score > best_score:
                best_score = score
                tied = [(rid, overlap)]
            elif score == best_score:
                tied.append((rid, overlap))
        if best_score is None or best_score < 0:
            return None
        warm_tied = [rid for rid, overlap in tied if overlap > 0]
        if not warm_tied:
            return None
        selected = (
            warm_tied[0]
            if session_id is None or len(warm_tied) == 1
            else _rendezvous_winner(
                session_id,
                warm_tied,
                self._encoded_replica_ids,
            )
        )
        return selected

    def _kv_event_select(
        self,
        eligible: tuple[str, ...],
        overlaps: Sequence[int],
        *,
        session_id: str | None = None,
    ) -> str | None:
        """Score one immutable exact-overlap view using the existing weights."""

        if len(overlaps) != len(eligible):
            raise ValueError("exact overlap view must align with eligible replicas")
        best_score: float | None = None
        tied: list[tuple[str, int]] = []
        for replica_id, overlap in zip(eligible, overlaps, strict=True):
            score = (
                self._prefix_alpha * overlap
                - self._prefix_beta * self._entries[replica_id].outstanding
            )
            if best_score is None or score > best_score:
                best_score = score
                tied = [(replica_id, overlap)]
            elif score == best_score:
                tied.append((replica_id, overlap))
        if best_score is None or best_score < 0:
            return None
        warm_tied = [replica_id for replica_id, overlap in tied if overlap > 0]
        if not warm_tied:
            return None
        return (
            warm_tied[0]
            if session_id is None or len(warm_tied) == 1
            else _rendezvous_winner(
                session_id,
                warm_tied,
                self._encoded_replica_ids,
            )
        )

    def _place_prepared(
        self,
        request: GenerationRequest,
        *,
        eligible: tuple[str, ...] | None = None,
    ) -> tuple[
        str,
        str,
        str | Sequence[str] | PreparedKvRouting | None,
        bool,
    ]:
        """Select/log once and retain request-local prefix work for completion."""
        from kairyu.telemetry import traced_span

        eligible = self._eligible_ids() if eligible is None else eligible
        if not eligible:
            raise RuntimeError(
                "ReplicaPool: no preflighted replica generation remains eligible"
            )

        started_ns = (
            request.placement_started_ns
            if request.placement_started_ns is not None
            else time.perf_counter_ns()
        )
        prepare = self._prefix_prepare
        track_prefix = self._should_track_prefix(request)
        prefix_prompt = prompt_text(request.prompt) if track_prefix else None
        if track_prefix and prefix_prompt is None:  # Defensive invariant.
            raise RuntimeError("prefix tracking requires a text prompt")
        prefix_fingerprint = ""
        if track_prefix and request.cache_hint is not None:
            prefix_fingerprint = request.cache_hint.prefix_fingerprint
        prefix_keys = (
            prepare(prefix_prompt, prefix_fingerprint or None)
            if prepare is not None and track_prefix
            else None
        )
        with traced_span(
            "kairyu.pool.place",
            {
                "kairyu.request_id": request.request_id,
                "kairyu.pool.size": len(self._entries),
                "kairyu.pool.eligible_size": len(eligible),
            },
        ) as span:
            replica_id, reason, session_id = self._select(
                request,
                prefix_keys=prefix_keys,
                track_prefix=track_prefix,
                eligible=eligible,
            )
            selected_at_ns = time.perf_counter_ns()
            placement_latency_ns = max(0, selected_at_ns - started_ns)
            if span is not None:
                # Preserve the original short keys for existing consumers.
                span.set_attribute("replica_id", replica_id)
                span.set_attribute("reason", reason)
                span.set_attribute("kairyu.replica.id", replica_id)
                span.set_attribute("kairyu.placement.reason", reason)
                span.set_attribute(
                    "kairyu.placement.latency_ns",
                    placement_latency_ns,
                )
        self._decision_counts[reason] += 1
        if self._log is not None:
            # legacy field stays the ordinal; replica_id added alongside (A1)
            self._log.record_replica(
                session_id,
                self._ordinal_by_id[replica_id],
                reason,
                replica_id=replica_id,
                replica_generation=self._entries[replica_id].generation,
                placement_latency_ns=placement_latency_ns,
                request_id=request.request_id,
                placement_started_ns=started_ns,
                selected_at_ns=selected_at_ns,
                pool_size=len(self._entries),
                eligible_size=len(eligible),
            )
        return replica_id, reason, prefix_keys, track_prefix

    def _place(self, request: GenerationRequest) -> str:
        """Select a replica and log the decision before dispatch (m5 D4)."""
        replica_id, _reason, _prefix_keys, _track_prefix = self._place_prepared(request)
        return replica_id

    def _finish(self, entry: _ReplicaEntry) -> None:
        if not entry.removed:  # late completion on a removed id is a no-op (A2)
            entry.outstanding -= 1

    def _observe_prefix(
        self,
        replica_id: str,
        entry: _ReplicaEntry,
        request: GenerationRequest,
        prefix_keys: str | Sequence[str] | PreparedKvRouting | None,
        *,
        promote: bool,
    ) -> None:
        """Publish a prefix only for the generation that received the request."""
        if self._entries.get(replica_id) is not entry:
            return
        observe_root = self._prefix_observe_root
        if isinstance(prefix_keys, str) and prefix_keys and observe_root is not None:
            observe_root(replica_id, prefix_keys)
        elif prefix_keys is not None and self._prefix_observe_prepared is not None:
            self._prefix_observe_prepared(replica_id, prefix_keys, promote)
        else:
            prefix_prompt = prompt_text(request.prompt)
            if prefix_prompt is None:  # Defensive: prefix tracking is text-only.
                raise RuntimeError("prefix tracking requires a text prompt")
            self._prefix_index.observe(replica_id, prefix_prompt)

    def _record_backend_failure(
        self,
        replica_id: str,
        entry: _ReplicaEntry,
    ) -> None:
        was_eligible = self._is_eligible(entry)
        entry.consecutive_failures += 1
        if was_eligible and not self._is_eligible(entry):
            self._refresh_eligible_snapshot()
            self._emit_membership_event(
                "health_ejected",
                replica_id,
                entry.generation,
                ineligible=(replica_id,),
            )

    def _record_backend_success(self, entry: _ReplicaEntry) -> None:
        # Once the failure threshold ejects an entry, only the readiness prober
        # may restore it. A slower request that was dispatched before ejection
        # must not silently put the replica back on the ring without emitting
        # the corresponding probe-succeeded membership transition.
        if self._is_healthy(entry):
            entry.consecutive_failures = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        # Legacy strings remain the universally compatible hot path and have
        # already crossed normal server/offline preflight. Typed variants need
        # an in-pool check as well so direct EngineBackend use cannot bypass the
        # intersection contract. Avoid re-tokenizing text across the fleet.
        await self.prepare_request(request)
        eligible = self._prepared_placement(request, consume=True)
        if eligible is None:  # Request is strongly held across both calls.
            raise RuntimeError("prepared replica placement lease was lost")
        replica_id, reason, prefix_keys, track_prefix = self._place_prepared(
            request,
            eligible=eligible,
        )
        entry = self._entries[replica_id]
        entry.outstanding += 1
        try:
            from kairyu.telemetry import traced_span

            with traced_span(
                "kairyu.replica.call",
                {
                    "kairyu.request_id": request.request_id,
                    "kairyu.replica.id": replica_id,
                    "kairyu.backend.type": type(entry.backend).__name__,
                    "kairyu.stream": False,
                },
                kind="client",
            ) as span:
                result = await entry.backend.generate(request)
                if span is not None:
                    span.set_attribute("kairyu.call.status", "success")
        except UpstreamClientError:
            # a bad client request (4xx) is not a replica health signal: do NOT
            # count it, or one misbehaving client would eject the whole pool (O1)
            raise
        except Exception:
            self._record_backend_failure(replica_id, entry)
            raise
        else:
            self._record_backend_success(entry)
            if track_prefix:
                self._observe_prefix(
                    replica_id,
                    entry,
                    request,
                    prefix_keys,
                    promote=reason in ("prefix_match", "kv_event_match"),
                )
            return result
        finally:
            self._finish(entry)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        await self.prepare_request(request)
        eligible = self._prepared_placement(request, consume=True)
        if eligible is None:
            raise RuntimeError("prepared replica placement lease was lost")
        replica_id, reason, prefix_keys, track_prefix = self._place_prepared(
            request,
            eligible=eligible,
        )
        entry = self._entries[replica_id]
        entry.outstanding += 1  # streams stay in-flight until generator close (A2)
        publish_root_on_first_chunk = (
            track_prefix
            and prefix_keys is not None
            and self._prefix_observe_prepared is not None
        )
        published_on_first_chunk = False
        try:
            from kairyu.telemetry import traced_span

            with traced_span(
                "kairyu.replica.call",
                {
                    "kairyu.request_id": request.request_id,
                    "kairyu.replica.id": replica_id,
                    "kairyu.backend.type": type(entry.backend).__name__,
                    "kairyu.stream": True,
                },
                kind="client",
            ) as span:
                async for chunk in entry.backend.stream(request):
                    if publish_root_on_first_chunk:
                        # The first backend result proves that prefill landed.
                        # Publish the cheap root before a long decode finishes.
                        self._observe_prefix(
                            replica_id,
                            entry,
                            request,
                            prefix_keys,
                            promote=False,
                        )
                        publish_root_on_first_chunk = False
                        published_on_first_chunk = True
                    yield chunk
                if span is not None:
                    span.set_attribute("kairyu.call.status", "success")
        except UpstreamClientError:
            raise  # client-side 4xx, not a replica failure (O1)
        except Exception:
            self._record_backend_failure(replica_id, entry)
            raise
        else:
            self._record_backend_success(entry)
            promote = reason in ("prefix_match", "kv_event_match")
            if track_prefix and (
                not published_on_first_chunk
                or (
                    promote
                    and prefix_keys is not None
                    and self._prefix_observe_prepared is not None
                )
            ):
                # Empty streams still publish on normal completion. Warm native
                # streams promote their first-token root to the full key chain.
                self._observe_prefix(
                    replica_id,
                    entry,
                    request,
                    prefix_keys,
                    promote=promote,
                )
        finally:
            self._finish(entry)

    async def shutdown(self) -> None:
        preparation_tasks = tuple(
            flight.task for flight in self._preparation_flights.values()
        )
        for task in preparation_tasks:
            task.cancel()
        if preparation_tasks:
            await asyncio.gather(*preparation_tasks, return_exceptions=True)
        self._preparation_flights.clear()
        resources = [entry.backend for entry in self._entries.values()]
        if self._log is not None:
            resources.append(self._log)
        try:
            await shutdown_all(resources, "ReplicaPool")
        finally:
            self._prepared_placements.clear()
