"""DP replica pool: N engine backends behind one ``EngineBackend`` (design doc m5 D4).

``ReplicaPool`` is an L2 orchestration component — a sibling of the ``Router``
per the m5 D4 seam amendment — so data parallelism never touches the engine and
L3 / the OpenAI server are unchanged. Placement policy, in order:

1. **Session affinity** — requests carrying ``cache_hint.session_id`` map to a
   replica by rendezvous (HRW) hashing over the healthy replicas, so removing
   one replica only remaps the sessions that lived on it (preserves multi-turn
   radix hits, G2 A8).
2. **Load-skew valve** — if the affine replica's outstanding-request count
   exceeds ``queue_depth_threshold``, fall back to the least-outstanding
   healthy replica (design m5 §5).
3. **Least outstanding** for session-less traffic (ties break to the lowest
   index).

Health: a replica that fails ``unhealthy_after`` consecutive requests is
removed from the hash ring until a probe succeeds (design m5 D4); see
``probe``. Placement is pure hashing plus one optional JSONL append (logged
before dispatch), keeping the router layer's <10 ms p99 budget. No background
tasks.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence

from kairyu.engine.backend import EngineBackend, GenerationRequest, GenerationResult
from kairyu.orchestration.router import JsonlRouterLog

_DEFAULT_UNHEALTHY_AFTER = 3
_DEFAULT_QUEUE_DEPTH_THRESHOLD = 8


def _rendezvous_score(session_id: str, replica_index: int) -> bytes:
    """HRW score for one (session, replica) pair; max over replicas wins."""
    return hashlib.sha256(f"{session_id}:{replica_index}".encode()).digest()


class ReplicaPool:
    """Routes requests across N ``EngineBackend`` replicas; itself an ``EngineBackend``."""

    def __init__(
        self,
        replicas: Sequence[EngineBackend],
        *,
        log: JsonlRouterLog | None = None,
        unhealthy_after: int = _DEFAULT_UNHEALTHY_AFTER,
        queue_depth_threshold: int = _DEFAULT_QUEUE_DEPTH_THRESHOLD,
    ) -> None:
        if len(replicas) < 1:
            raise ValueError("ReplicaPool requires at least 1 replica")
        if unhealthy_after < 1:
            raise ValueError(f"unhealthy_after must be >= 1, got {unhealthy_after}")
        if queue_depth_threshold < 0:
            raise ValueError(
                f"queue_depth_threshold must be >= 0, got {queue_depth_threshold}"
            )
        self._replicas: tuple[EngineBackend, ...] = tuple(replicas)
        self._log = log
        self._unhealthy_after = unhealthy_after
        self._queue_depth_threshold = queue_depth_threshold
        self._outstanding: list[int] = [0] * len(self._replicas)
        self._consecutive_failures: list[int] = [0] * len(self._replicas)
        self._decision_counts: dict[str, int] = {
            "session_affinity": 0,
            "queue_depth_fallback": 0,
            "least_outstanding": 0,
        }

    @property
    def outstanding(self) -> tuple[int, ...]:
        """In-flight request count per replica (dispatch-incremented, completion-decremented)."""
        return tuple(self._outstanding)

    @property
    def replica_count(self) -> int:
        return len(self._replicas)

    @property
    def healthy(self) -> tuple[bool, ...]:
        """Per-replica health (read-only; consumed by /readyz, /metrics, the prober)."""
        healthy = set(self._healthy_indices())
        return tuple(index in healthy for index in range(len(self._replicas)))

    @property
    def decision_counts(self) -> dict[str, int]:
        """Cumulative placement decisions by reason (read-only, for /metrics)."""
        return dict(self._decision_counts)

    async def probe(self, index: int) -> None:
        """Declare replica ``index`` healthy again, returning it to the hash ring.

        Sends nothing; it only resets the consecutive-failure count. The caller
        decides when a probe has succeeded (e.g. after an out-of-band health
        check). The count also resets automatically whenever the replica
        completes any request successfully.
        """
        if not 0 <= index < len(self._replicas):
            raise ValueError(
                f"replica index {index} out of range [0, {len(self._replicas)})"
            )
        self._consecutive_failures[index] = 0

    def _healthy_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(len(self._replicas))
            if self._consecutive_failures[index] < self._unhealthy_after
        )

    def _least_outstanding(self, candidates: Sequence[int]) -> int:
        return min(candidates, key=lambda index: (self._outstanding[index], index))

    def _select(self, request: GenerationRequest) -> tuple[int, str, str | None]:
        """Pick a replica; returns (index, reason, session_id). Pure hashing, no I/O."""
        healthy = self._healthy_indices()
        if not healthy:
            raise RuntimeError(
                f"ReplicaPool: all {len(self._replicas)} replicas are unhealthy "
                f"(each failed >= {self._unhealthy_after} consecutive requests); "
                "call probe(index) once a replica recovers"
            )
        session_id = request.cache_hint.session_id if request.cache_hint else None
        if not session_id:
            return self._least_outstanding(healthy), "least_outstanding", None
        affine = max(healthy, key=lambda index: _rendezvous_score(session_id, index))
        if self._outstanding[affine] > self._queue_depth_threshold:
            return self._least_outstanding(healthy), "queue_depth_fallback", session_id
        return affine, "session_affinity", session_id

    def _place(self, request: GenerationRequest) -> int:
        """Select a replica and log the decision before dispatch (design m5 D4)."""
        index, reason, session_id = self._select(request)
        self._decision_counts[reason] += 1
        if self._log is not None:
            self._log.record_replica(session_id, index, reason)
        return index

    def _record_failure(self, index: int) -> None:
        self._consecutive_failures[index] += 1

    def _record_success(self, index: int) -> None:
        self._consecutive_failures[index] = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        index = self._place(request)
        self._outstanding[index] += 1
        try:
            result = await self._replicas[index].generate(request)
        except Exception:
            self._record_failure(index)
            raise
        else:
            self._record_success(index)
            return result
        finally:
            self._outstanding[index] -= 1

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        index = self._place(request)
        self._outstanding[index] += 1
        try:
            async for chunk in self._replicas[index].stream(request):
                yield chunk
        except Exception:
            self._record_failure(index)
            raise
        else:
            self._record_success(index)
        finally:
            self._outstanding[index] -= 1

    async def shutdown(self) -> None:
        for replica in self._replicas:
            await replica.shutdown()
