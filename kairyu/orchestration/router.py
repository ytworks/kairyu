"""Pluggable query router: tier1 (light) / tier2 (frontier) / multi_agent.

``RuleRouter`` is the first implementation on the pluggable seam (design doc
D3); the M4 learned classifier and contextual bandit implement the same
``Router`` protocol. Every decision can be logged (features only, no raw text)
as the M4 training corpus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

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


class RuleRouter:
    """Threshold rules over extracted features; no model in the hot path."""

    def __init__(self, thresholds: RouteThresholds = DEFAULT_THRESHOLDS) -> None:
        self._thresholds = thresholds

    def route(self, query: str, context: dict | None = None) -> RouteDecision:
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


class JsonlRouterLog:
    """Appends routing decisions as JSONL; raw query text is never stored."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _append(self, entry: dict) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

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

    def record_outcome(
        self,
        query: str,
        target: RouteTarget,
        quality: float,
        cost_usd: float,
        latency_s: float | None = None,
    ) -> None:
        """Log the observed outcome of a routed request (M4 training signal)."""
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
