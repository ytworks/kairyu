"""Distilled routing classifier: multinomial logistic regression, pure Python.

7 features x 3 classes: numpy/torch would be dependencies for nothing, and
inference is microseconds — comfortably inside the <10ms routing budget
(design doc m4 §2.3).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

from kairyu.orchestration.features import extract_features
from kairyu.orchestration.learning.dataset import FEATURE_ORDER, LabeledExample, feature_vector
from kairyu.orchestration.router import RouteDecision, Router

TARGETS = ("tier1", "tier2", "multi_agent")

_DEFAULT_EPOCHS = 100
_DEFAULT_LR = 0.1
# Softmax max-prob is always >= 1/3 for 3 classes, so a near-uniform threshold
# would never fire. 0.55 is a starting point; calibrate on held-out data for a
# target selective accuracy before trusting it in serving (design doc m4 §2.3).
_DEFAULT_MIN_CONFIDENCE = 0.55


def _softmax(scores: Sequence[float]) -> list[float]:
    peak = max(scores)
    exps = [math.exp(score - peak) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


class RouterModel:
    def __init__(
        self,
        weights: list[list[float]],  # per target: len(FEATURE_ORDER) + 1 (bias)
        means: list[float],
        stds: list[float],
    ) -> None:
        self.weights = weights
        self.means = means
        self.stds = stds

    def _standardize(self, features: dict) -> list[float]:
        raw = feature_vector(features)
        return [
            (value - mean) / std
            for value, mean, std in zip(raw, self.means, self.stds, strict=True)
        ]

    def _scores(self, x: list[float]) -> list[float]:
        return [
            sum(w * v for w, v in zip(row[:-1], x, strict=True)) + row[-1]
            for row in self.weights
        ]

    def predict(self, features: dict) -> tuple[str, float]:
        probabilities = _softmax(self._scores(self._standardize(features)))
        best = max(range(len(TARGETS)), key=lambda i: probabilities[i])
        return TARGETS[best], probabilities[best]

    def save(self, path: str | Path) -> None:
        payload = {"weights": self.weights, "means": self.means, "stds": self.stds}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RouterModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights=payload["weights"], means=payload["means"], stds=payload["stds"])


def train_model(
    examples: Sequence[LabeledExample],
    epochs: int = _DEFAULT_EPOCHS,
    lr: float = _DEFAULT_LR,
    seed: int = 0,
) -> RouterModel:
    if not examples:
        raise ValueError("cannot train a router model on an empty dataset")
    vectors = [feature_vector(example.features) for example in examples]
    n_features = len(FEATURE_ORDER)
    means = [sum(v[i] for v in vectors) / len(vectors) for i in range(n_features)]
    stds = []
    for i in range(n_features):
        variance = sum((v[i] - means[i]) ** 2 for v in vectors) / len(vectors)
        stds.append(math.sqrt(variance) or 1.0)
    model = RouterModel(
        weights=[[0.0] * (n_features + 1) for _ in TARGETS], means=means, stds=stds
    )
    standardized = [model._standardize(example.features) for example in examples]
    labels = [TARGETS.index(example.label) for example in examples]
    rng = random.Random(seed)
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            x = standardized[index]
            probabilities = _softmax(model._scores(x))
            for target_index in range(len(TARGETS)):
                gradient = probabilities[target_index] - (
                    1.0 if target_index == labels[index] else 0.0
                )
                row = model.weights[target_index]
                for i in range(n_features):
                    row[i] -= lr * gradient * x[i]
                row[n_features] -= lr * gradient
    return model


class LearnedRouter:
    """Router-protocol implementation backed by a RouterModel with rule fallback."""

    def __init__(
        self,
        model: RouterModel,
        fallback: Router | None = None,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self._model = model
        self._fallback = fallback
        self._min_confidence = min_confidence

    def _decide(
        self,
        query: str,
        context: dict | None = None,
        *,
        preview: bool = False,
    ) -> RouteDecision:
        features = extract_features(query)
        label, confidence = self._model.predict(features.as_dict())
        if confidence < self._min_confidence and self._fallback is not None:
            fallback = (
                getattr(self._fallback, "preview", None)
                if preview
                else self._fallback.route
            )
            if fallback is None:
                raise NotImplementedError("fallback router does not support preview")
            base = fallback(query, context)
            return RouteDecision(
                target=base.target,
                confidence=base.confidence,
                features=features,
                reason=f"fallback:low_confidence({confidence:.2f}); {base.reason}",
            )
        return RouteDecision(
            target=label,  # type: ignore[arg-type]
            confidence=confidence,
            features=features,
            reason="learned",
        )

    def route(self, query: str, context: dict | None = None) -> RouteDecision:
        return self._decide(query, context)

    def preview(self, query: str, context: dict | None = None) -> RouteDecision:
        return self._decide(query, context, preview=True)

    def describe(self) -> dict[str, object]:
        return {
            "router_type": type(self).__name__,
            "min_confidence": self._min_confidence,
            "fallback_type": (
                type(self._fallback).__name__ if self._fallback is not None else None
            ),
        }
