"""Pluggable query router: tier1 (light) / tier2 (frontier) / multi_agent.

``RuleRouter`` is the first implementation on the pluggable seam (design doc
D3); the M4 learned classifier and contextual bandit implement the same
``Router`` protocol. Every decision can be logged (features only, no raw text)
as the M4 training corpus.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Literal, Protocol

from kairyu.audit_io import BoundedJsonlWriter
from kairyu.orchestration.features import QueryFeatures, extract_features

RouteTarget = Literal["tier1", "tier2", "multi_agent"]


@dataclass(frozen=True)
class RouteThresholds:
    multi_step_markers: int = 3
    multi_agent_min_chars: int = 2000
    reasoning_keywords: int = 2
    math_symbols: int = 3
    tier2_min_chars: int = 600


DEFAULT_THRESHOLDS = RouteThresholds()


@dataclass(frozen=True)
class RouteDecision:
    target: RouteTarget
    confidence: float
    features: QueryFeatures
    reason: str


class Router(Protocol):
    def route(self, query: str, context: dict | None = None) -> RouteDecision: ...

    def preview(self, query: str, context: dict | None = None) -> RouteDecision: ...

    def describe(self) -> dict[str, object]: ...


class RuleRouter:
    """Threshold rules over extracted features; no model in the hot path."""

    def __init__(self, thresholds: RouteThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def _decide(self, query: str) -> RouteDecision:
        features = extract_features(query)
        t = self._thresholds
        if (
            features.multi_step_marker_count >= t.multi_step_markers
            or features.char_len >= t.multi_agent_min_chars
        ):
            return RouteDecision(
                target="multi_agent",
                confidence=0.7,
                features=features,
                reason=(
                    f"multi_step_markers={features.multi_step_marker_count} "
                    f"char_len={features.char_len}"
                ),
            )
        if (
            features.has_code_fence
            or features.reasoning_keyword_count >= t.reasoning_keywords
            or features.math_symbol_count >= t.math_symbols
            or features.char_len >= t.tier2_min_chars
        ):
            return RouteDecision(
                target="tier2",
                confidence=0.7,
                features=features,
                reason=(
                    f"code_fence={features.has_code_fence} "
                    f"reasoning_keywords={features.reasoning_keyword_count} "
                    f"math_symbols={features.math_symbol_count}"
                ),
            )
        return RouteDecision(
            target="tier1",
            confidence=0.8,
            features=features,
            reason="no heavy signals; short query",
        )

    def route(self, query: str, context: dict | None = None) -> RouteDecision:
        return self._decide(query)

    def preview(self, query: str, context: dict | None = None) -> RouteDecision:
        return self._decide(query)

    def describe(self) -> dict[str, object]:
        return {
            "router_type": type(self).__name__,
            "thresholds": asdict(self._thresholds),
        }


class JsonlRouterLog:
    """Bounded asynchronous JSONL; raw query text is never stored.

    The constructing router/pool owns ``close``. ``flush`` provides an ordered
    read-after-write barrier for training/inspection consumers."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_pending_records: int = 4_096,
        batch_size: int = 128,
        _open_file: Callable[..., IO[str]] = open,
    ) -> None:
        self._path = Path(path)
        self._writer = BoundedJsonlWriter(
            self._path,
            max_pending=max_pending_records,
            batch_size=batch_size,
            open_file=_open_file,
        )

    def _append(self, entry: dict) -> None:
        self._writer.append(json.dumps(entry))

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    async def shutdown(self) -> None:
        """Drain without blocking an async lifecycle's event loop."""
        await asyncio.to_thread(self.close)

    def __enter__(self) -> JsonlRouterLog:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def record(self, query: str, decision: RouteDecision) -> None:
        self._append(
            {
                "kind": "decision",
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "target": decision.target,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "features": decision.features.as_dict(),
            }
        )

    def record_replica(
        self,
        session_id: str | None,
        replica_index: int,
        reason: str,
        replica_id: str | None = None,
        replica_generation: str | None = None,
        placement_latency_ns: int | None = None,
        request_id: str | None = None,
        placement_started_ns: int | None = None,
        selected_at_ns: int | None = None,
        pool_size: int | None = None,
        eligible_size: int | None = None,
    ) -> None:
        """Log a ``ReplicaPool`` placement decision (design doc m5 D4).

        The raw session id is never stored — only its SHA-256, matching the
        no-raw-text rule of the other records. The M4 dataset builder filters
        records by ``kind``, so ``replica`` entries never enter the training
        corpus.
        """
        session_sha256 = (
            hashlib.sha256(session_id.encode()).hexdigest()
            if session_id is not None
            else None
        )
        record = {
            "kind": "replica",
            "session_sha256": session_sha256,
            "replica": replica_index,
            "reason": reason,
        }
        if replica_id is not None:  # m10a A1: id alongside the legacy ordinal
            record["replica_id"] = replica_id
        if replica_generation is not None:
            record["replica_generation"] = replica_generation
        if placement_latency_ns is not None:
            record["placement_latency_ns"] = placement_latency_ns
        if request_id is not None:
            record["request_id"] = request_id
        if placement_started_ns is not None:
            record["placement_started_ns"] = placement_started_ns
        if selected_at_ns is not None:
            record["selected_at_ns"] = selected_at_ns
        if pool_size is not None:
            record["pool_size"] = pool_size
        if eligible_size is not None:
            record["eligible_size"] = eligible_size
        self._append(record)

    def record_membership(
        self,
        actions: Mapping[str, object],
        *,
        replica_ids: tuple[str, ...],
        healthy_ids: tuple[str, ...],
        eligible_ids: tuple[str, ...],
        generation_by_id: Mapping[str, str] | None = None,
    ) -> None:
        """Record one non-empty discovery/reconciliation transition."""

        action_keys = (
            "added",
            "draining",
            "undraining",
            "removed",
            "eligible",
            "ineligible",
        )
        if "event_id" not in actions and not any(
            actions.get(key) for key in action_keys
        ):
            return
        transition_ns = actions.get("observed_ns")
        if not isinstance(transition_ns, int):
            transition_ns = time.perf_counter_ns()
        record: dict[str, object] = {
            "kind": "membership",
            "observed_ns": transition_ns,
            **{
                key: list(actions.get(key, ()))
                for key in action_keys
            },
            "replica_ids": list(replica_ids),
            "healthy_ids": list(healthy_ids),
            "eligible_ids": list(eligible_ids),
        }
        for key in (
            "event_id",
            "event_source_id",
            "sequence",
            "reason",
            "replica_id",
            "replica_generation",
        ):
            if key in actions:
                record[key] = actions[key]
        if generation_by_id is not None:
            record["generation_by_id"] = dict(generation_by_id)
        self._append(record)

    def record_outcome(
        self,
        query: str,
        target: RouteTarget,
        quality: float,
        cost_usd: float,
        latency_s: float | None = None,
    ) -> None:
        """Log the observed outcome of a routed request (M4 training signal)."""
        if not 0.0 <= quality <= 1.0:
            raise ValueError(f"quality must be in [0, 1], got {quality}")
        if cost_usd < 0.0:
            raise ValueError(f"cost_usd must be >= 0, got {cost_usd}")
        self._append(
            {
                "kind": "outcome",
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "target": target,
                "quality": quality,
                "cost_usd": cost_usd,
                "latency_s": latency_s,
            }
        )
