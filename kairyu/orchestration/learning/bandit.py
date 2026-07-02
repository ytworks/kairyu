"""Contextual bandit for online routing refinement (design doc m4 §2.4).

One linear reward model per arm over the query features, SGD-updated from
observed rewards, epsilon-greedy exploration. update() is O(features), safe on
the serving hot path.
"""

from __future__ import annotations

import random

from kairyu.orchestration.features import QueryFeatures

_ARMS = ("tier1", "tier2", "multi_agent")
_DEFAULT_EPSILON = 0.1
_DEFAULT_LR = 0.2

# Fixed feature scaling so SGD is stable without maintaining running statistics.
_SCALES = {
    "char_len": 500.0,
    "word_count": 100.0,
    "has_code_fence": 1.0,
    "math_symbol_count": 5.0,
    "reasoning_keyword_count": 5.0,
    "multi_step_marker_count": 5.0,
    "question_count": 5.0,
}


def _vectorize(features: QueryFeatures) -> list[float]:
    values = features.as_dict()
    scaled = [float(values[name]) / scale for name, scale in _SCALES.items()]
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
        dimension = len(_SCALES) + 1
        self._weights = {
            arm: list(weights[arm]) if weights else [0.0] * dimension for arm in arms
        }

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
