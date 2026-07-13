"""DP replica pool: N engine backends behind one ``EngineBackend`` (m5 D4, m10a D1).

``ReplicaPool`` is an L2 orchestration component — a sibling of the ``Router``
per the m5 D4 seam amendment. Placement policy, in order:

1. **Session affinity** — requests carrying ``cache_hint.session_id`` map to a
   replica by rendezvous (HRW) hashing over the ELIGIBLE replicas (healthy ∧
   not draining), so removing or draining one replica only remaps the sessions
   that lived on it (G2 A8; m10a property test).
2. **Load-skew valve** — if the affine replica's outstanding-request count
   exceeds ``queue_depth_threshold``, fall back to least-outstanding.
3. **Least outstanding** for session-less traffic (ties break to insertion
   order).

m10a dynamic membership: entries are id-keyed; legacy sequence construction
auto-ids "0".."N-1" — HRW strings and Prometheus labels are IDENTICAL to the
index era (review A1). ``add_replica``/``drain``/``remove_replica`` never drop
in-flight work: drain stops NEW placements, completion on a removed id is a
no-op (A2), and remove refuses while outstanding > 0 unless forced.

Health: ``unhealthy_after`` consecutive failures removes a replica from the
ring until ``probe()`` (which resets failures but NEVER clears draining — A4).
Placement is pure hashing plus one optional JSONL append. No background tasks.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field

from kairyu.engine.backend import (
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    UpstreamClientError,
    shutdown_all,
)
from kairyu.orchestration.router import JsonlRouterLog

_DEFAULT_UNHEALTHY_AFTER = 3
_DEFAULT_QUEUE_DEPTH_THRESHOLD = 8


def _rendezvous_score(session_id: str, replica_id: str) -> bytes:
    """HRW score for one (session, replica) pair; max over replicas wins."""
    return hashlib.sha256(f"{session_id}:{replica_id}".encode()).digest()


@dataclass
class _ReplicaEntry:
    backend: EngineBackend
    health_url: str | None = None
    outstanding: int = 0
    consecutive_failures: int = 0
    draining: bool = False
    removed: bool = field(default=False, compare=False)


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
    ) -> None:
        if len(replicas) < 1:
            raise ValueError("ReplicaPool requires at least 1 replica")
        if unhealthy_after < 1:
            raise ValueError(f"unhealthy_after must be >= 1, got {unhealthy_after}")
        if queue_depth_threshold < 0:
            raise ValueError(
                f"queue_depth_threshold must be >= 0, got {queue_depth_threshold}"
            )
        if isinstance(replicas, Mapping):
            items = list(replicas.items())
        else:  # legacy sequence: auto-ids match the old index labels (A1)
            items = [(str(index), backend) for index, backend in enumerate(replicas)]
        self._entries: dict[str, _ReplicaEntry] = {
            replica_id: _ReplicaEntry(backend=backend) for replica_id, backend in items
        }
        self._log = log
        self._unhealthy_after = unhealthy_after
        self._queue_depth_threshold = queue_depth_threshold
        # m10b D6: KV-aware scoring is OPT-IN (prefix_index=None keeps m5
        # behavior byte-identical); score = alpha*overlap - beta*outstanding
        self._prefix_index = prefix_index
        self._prefix_alpha = prefix_alpha
        self._prefix_beta = prefix_beta
        self._decision_counts: dict[str, int] = {
            "session_affinity": 0,
            "queue_depth_fallback": 0,
            "least_outstanding": 0,
            "prefix_match": 0,
        }

    # -- membership (m10a D1) -------------------------------------------------

    @property
    def replica_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def add_replica(
        self, replica_id: str, backend: EngineBackend, health_url: str | None = None
    ) -> None:
        if replica_id in self._entries:
            raise ValueError(f"replica {replica_id!r} already in the pool")
        self._entries[replica_id] = _ReplicaEntry(backend=backend, health_url=health_url)

    def drain(self, replica_id: str) -> None:
        """Stop NEW placements; in-flight requests complete normally."""
        self._entry(replica_id).draining = True

    def cancel_drain(self, replica_id: str) -> None:
        """Return a draining replica to placement without changing other state."""
        self._entry(replica_id).draining = False

    def is_draining(self, replica_id: str) -> bool:
        return self._entry(replica_id).draining

    async def remove_replica(self, replica_id: str, force: bool = False) -> None:
        entry = self._entry(replica_id)
        if entry.outstanding > 0 and not force:
            raise RuntimeError(
                f"replica {replica_id!r} has {entry.outstanding} in-flight requests; "
                "drain first (or force=True)"
            )
        entry.removed = True  # guarded decrement: late completions are no-ops (A2)
        del self._entries[replica_id]
        # drop this id's prefix history (M5): a re-added replica with the same id
        # must not inherit phantom prefixes and route shared traffic to a cold cache
        if self._prefix_index is not None:
            self._prefix_index.forget_replica(replica_id)
        await shutdown_all((entry.backend,), f"replica {replica_id!r}")

    def health_url(self, replica_id: str) -> str | None:
        return self._entry(replica_id).health_url

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
        return tuple(
            entry.consecutive_failures < self._unhealthy_after
            for entry in self._entries.values()
        )

    def healthy_by_id(self) -> dict[str, bool]:
        return dict(zip(self._entries, self.healthy, strict=True))

    def outstanding_by_id(self) -> dict[str, int]:
        return {rid: entry.outstanding for rid, entry in self._entries.items()}

    @property
    def decision_counts(self) -> dict[str, int]:
        """Cumulative placement decisions by reason (read-only, for /metrics)."""
        return dict(self._decision_counts)

    async def probe(self, key: int | str) -> None:
        """Declare a replica healthy again, returning it to the hash ring.

        Accepts the legacy ordinal or the replica id (A1). Resets the failure
        count only — a probe NEVER clears draining (A4).
        """
        self._entry(self._resolve_id(key)).consecutive_failures = 0

    # -- placement -------------------------------------------------------------

    def _eligible_ids(self) -> tuple[str, ...]:
        return tuple(
            rid
            for rid, entry in self._entries.items()
            if entry.consecutive_failures < self._unhealthy_after and not entry.draining
        )

    def _least_outstanding(self, candidates: Sequence[str]) -> str:
        order = {rid: position for position, rid in enumerate(self._entries)}
        return min(candidates, key=lambda rid: (self._entries[rid].outstanding, order[rid]))

    def _select(self, request: GenerationRequest) -> tuple[str, str, str | None]:
        """Pick a replica; returns (replica_id, reason, session_id). Pure hashing."""
        eligible = self._eligible_ids()
        if not eligible:
            raise RuntimeError(
                f"ReplicaPool: none of the {len(self._entries)} replicas are eligible "
                f"(unhealthy after >= {self._unhealthy_after} consecutive failures, "
                "or draining); call probe() once a replica recovers"
            )
        session_id = request.cache_hint.session_id if request.cache_hint else None
        if session_id:
            affine = max(eligible, key=lambda rid: _rendezvous_score(session_id, rid))
            if self._entries[affine].outstanding > self._queue_depth_threshold:
                return self._least_outstanding(eligible), "queue_depth_fallback", session_id
            return affine, "session_affinity", session_id
        if self._prefix_index is not None:
            best = self._prefix_select(eligible, request.prompt)
            if best is not None:
                return best, "prefix_match", None
        return self._least_outstanding(eligible), "least_outstanding", None

    def _prefix_select(self, eligible: tuple[str, ...], prompt: str) -> str | None:
        """alpha*overlap - beta*outstanding; None when no candidate has overlap
        (fall through to least-outstanding rather than pay a random placement)."""
        scored = []
        for rid in eligible:
            overlap = self._prefix_index.overlap(rid, prompt)  # compute once per replica (P5)
            score = (
                self._prefix_alpha * overlap
                - self._prefix_beta * self._entries[rid].outstanding
            )
            scored.append((score, overlap, rid))
        best_score, best_overlap, best_rid = max(scored, key=lambda item: item[0])
        if best_overlap == 0:
            return None
        return best_rid

    def _place(self, request: GenerationRequest) -> str:
        """Select a replica and log the decision before dispatch (m5 D4)."""
        from kairyu.telemetry import traced_span

        replica_id, reason, session_id = self._select(request)
        self._decision_counts[reason] += 1
        with traced_span(
            "kairyu.pool.place", {"replica_id": replica_id, "reason": reason}
        ):
            pass
        if self._prefix_index is not None:
            self._prefix_index.observe(replica_id, request.prompt)
        if self._log is not None:
            # legacy field stays the ordinal; replica_id added alongside (A1)
            ordinal = list(self._entries).index(replica_id)
            self._log.record_replica(session_id, ordinal, reason, replica_id=replica_id)
        return replica_id

    def _finish(self, entry: _ReplicaEntry) -> None:
        if not entry.removed:  # late completion on a removed id is a no-op (A2)
            entry.outstanding -= 1

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        replica_id = self._place(request)
        entry = self._entries[replica_id]
        entry.outstanding += 1
        try:
            result = await entry.backend.generate(request)
        except UpstreamClientError:
            # a bad client request (4xx) is not a replica health signal: do NOT
            # count it, or one misbehaving client would eject the whole pool (O1)
            raise
        except Exception:
            entry.consecutive_failures += 1
            raise
        else:
            entry.consecutive_failures = 0
            return result
        finally:
            self._finish(entry)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        replica_id = self._place(request)
        entry = self._entries[replica_id]
        entry.outstanding += 1  # streams stay in-flight until generator close (A2)
        try:
            async for chunk in entry.backend.stream(request):
                yield chunk
        except UpstreamClientError:
            raise  # client-side 4xx, not a replica failure (O1)
        except Exception:
            entry.consecutive_failures += 1
            raise
        else:
            entry.consecutive_failures = 0
        finally:
            self._finish(entry)

    async def shutdown(self) -> None:
        await shutdown_all(
            (entry.backend for entry in self._entries.values()), "ReplicaPool"
        )
