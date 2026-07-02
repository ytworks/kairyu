"""Build a labeled routing dataset from JSONL serving logs (design doc m4 §2.2).

Joins decision records (features) with outcome records (quality/cost) on the
query hash and labels each query with the highest mean-utility observed target,
utility = quality - cost_weight * cost_usd.

Known bias (documented in design doc m4 §2.2): counterfactual arms are observed
only when the exact query recurs under exploration, so with mostly-single-arm
observations this distills "the chosen arm cleared the utility floor" rather
than the utility-optimal policy. Decision records carry router/confidence for
future inverse-propensity weighting.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

FEATURE_ORDER = (
    "char_len",
    "word_count",
    "has_code_fence",
    "math_symbol_count",
    "reasoning_keyword_count",
    "multi_step_marker_count",
    "question_count",
)

_DEFAULT_UTILITY_FLOOR = 0.0


@dataclass(frozen=True)
class LabeledExample:
    query_hash: str
    features: dict
    label: str


def feature_vector(features: dict) -> list[float]:
    return [float(features[name]) for name in FEATURE_ORDER]


def build_dataset(
    records: Iterable[dict],
    cost_weight: float = 1.0,
    utility_floor: float = _DEFAULT_UTILITY_FLOOR,
) -> tuple[LabeledExample, ...]:
    features_by_hash: dict[str, dict] = {}
    utility_sums: dict[str, dict[str, list[float]]] = {}
    for record in records:
        query_hash = record.get("query_sha256")
        if not query_hash:
            continue
        if record.get("kind") == "decision" and "features" in record:
            features_by_hash.setdefault(query_hash, record["features"])
        elif record.get("kind") == "outcome":
            utility = record["quality"] - cost_weight * record["cost_usd"]
            per_target = utility_sums.setdefault(query_hash, {})
            per_target.setdefault(record["target"], []).append(utility)
    examples = []
    for query_hash, per_target in utility_sums.items():
        features = features_by_hash.get(query_hash)
        if features is None:
            continue
        # mean, not max: max is winner's-curse optimistic under noisy judge scores
        mean_utilities = {
            target: sum(values) / len(values) for target, values in per_target.items()
        }
        label, best_utility = max(mean_utilities.items(), key=lambda item: item[1])
        if best_utility <= utility_floor:
            continue
        examples.append(
            LabeledExample(query_hash=query_hash, features=dict(features), label=label)
        )
    return tuple(examples)
