"""Build a labeled routing dataset from JSONL serving logs (design doc m4 §2.2).

Joins decision records (features) with outcome records (quality/cost) on the
query hash and labels each query with the highest-utility observed target,
utility = quality - cost_weight * cost_usd.
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
    utilities: dict[str, dict[str, float]] = {}
    for record in records:
        query_hash = record.get("query_sha256")
        if not query_hash:
            continue
        if record.get("kind") == "decision" and "features" in record:
            features_by_hash.setdefault(query_hash, record["features"])
        elif record.get("kind") == "outcome":
            utility = record["quality"] - cost_weight * record["cost_usd"]
            per_target = utilities.setdefault(query_hash, {})
            per_target[record["target"]] = max(
                per_target.get(record["target"], float("-inf")), utility
            )
    examples = []
    for query_hash, per_target in utilities.items():
        features = features_by_hash.get(query_hash)
        if features is None:
            continue
        label, best_utility = max(per_target.items(), key=lambda item: item[1])
        if best_utility <= utility_floor:
            continue
        examples.append(
            LabeledExample(query_hash=query_hash, features=dict(features), label=label)
        )
    return tuple(examples)
