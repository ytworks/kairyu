"""Contextual bandit for online routing refinement (design doc m4 §2.4).

One linear reward model per arm over the query features, SGD-updated from
observed rewards, epsilon-greedy exploration. update() is O(features), safe on
the serving hot path. ``BanditRouter`` adapts it to the Router protocol with a
cold-start deferral to a base router.
"""

from __future__ import annotations

import random

from kairyu.orchestration.features import QueryFeatures, extract_features
from kairyu.orchestration.learning.dataset import FEATURE_ORDER
from kairyu.orchestration.router import RouteDecision, Router

_ARMS = ("tier1", "tier2", "multi_agent")
_DEFAULT_EPSILON = 0.1
_DEFAULT_LR = 0.2
_CLIP = 4.0  # scaled features are clipped so one outlier can't blow up SGD

# Fixed feature scaling so SGD is stable without maintaining running statistics.
# Derived from FEATURE_ORDER so the bandit and the classifier can never
# silently diverge in feature space.
_SCALES = {
    "char_len": 500.0,
    "word_count": 100.0,
    "has_code_fence": 1.0,
    "math_symbol_count": 5.0,
    "reasoning_keyword_count": 5.0,
    "multi_step_marker_count": 5.0,
    "question_count": 5.0,
}
if set(_SCALES) != set(FEATURE_ORDER):  # pragma: no cover - import-time invariant
    raise AssertionError("bandit _SCALES out of sync with dataset FEATURE_ORDER")

_DIMENSION = len(FEATURE_ORDER) + 1  # + bias


def _vectorize(features: QueryFeatures) -> list[float]:
    values = features.as_dict()
    scaled = [
        min(float(values[name]) / _SCALES[name], _CLIP) for name in FEATURE_ORDER
    ]
    return [*scaled, 1.0]  # bias term


class GreedyLinearBandit:
    def __init__(
        self,
        epsilon: float = _DEFAULT_EPSILON,
        lr: float = _DEFAULT_LR,
        seed: int = 0,
        weights: dict[str, list[float]] | None = None,
        arms: tuple[str, ...] = _ARMS,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self._epsilon = epsilon
        self._lr = lr
        self._rng = random.Random(seed)
        self._arms = arms
        if weights is not None:
            missing = [arm for arm in arms if arm not in weights]
            if missing:
                raise ValueError(f"weights missing arms: {missing}")
            bad = [arm for arm in arms if len(weights[arm]) != _DIMENSION]
            if bad:
                raise ValueError(
                    f"weights for arms {bad} have wrong dimension (expected {_DIMENSION})"
                )
            self._weights = {arm: list(weights[arm]) for arm in arms}
        else:
            self._weights = {arm: [0.0] * _DIMENSION for arm in arms}

    @property
    def weights(self) -> dict[str, list[float]]:
        return {arm: list(row) for arm, row in self._weights.items()}

    def _score(self, arm: str, x: list[float]) -> float:
        return sum(w * v for w, v in zip(self._weights[arm], x, strict=True))

    def select(self, features: QueryFeatures) -> str:
        if self._rng.random() < self._epsilon:
            return self._rng.choice(self._arms)
        x = _vectorize(features)
        return max(self._arms, key=lambda arm: self._score(arm, x))

    def update(self, features: QueryFeatures, arm: str, reward: float) -> None:
        if arm not in self._weights:
            raise ValueError(f"unknown arm {arm!r}; known arms: {self._arms}")
        x = _vectorize(features)
        error = reward - self._score(arm, x)
        row = self._weights[arm]
        for i, value in enumerate(x):
            row[i] += self._lr * error * value


class BanditRouter:
    """Router-protocol adapter: bandit policy with cold-start deferral to a base.

    Until every arm has received ``min_updates_per_arm`` rewards, decisions
    defer to the base router (M1 RuleRouter by default seam) so an untrained
    bandit never degrades routing.
    """

    def __init__(
        self,
        bandit: GreedyLinearBandit,
        base: Router,
        min_updates_per_arm: int = 10,
    ) -> None:
        self._bandit = bandit
        self._base = base
        self._min_updates = min_updates_per_arm
        self._update_counts: dict[str, int] = dict.fromkeys(bandit._arms, 0)

    @property
    def is_warm(self) -> bool:
        return all(count >= self._min_updates for count in self._update_counts.values())

    def route(self, query: str, context: dict | None = None) -> RouteDecision:
        features = extract_features(query)
        if not self.is_warm:
            base = self._base.route(query, context)
            return RouteDecision(
                target=base.target,
                confidence=base.confidence,
                features=features,
                reason=f"bandit:cold_start({dict(self._update_counts)}); {base.reason}",
            )
        target = self._bandit.select(features)
        return RouteDecision(
            target=target,  # type: ignore[arg-type]
            confidence=0.5,
            features=features,
            reason="bandit",
        )

    def record_reward(self, query: str, target: str, reward: float) -> None:
        features = extract_features(query)
        self._bandit.update(features, target, reward)
        self._update_counts[target] = self._update_counts.get(target, 0) + 1
