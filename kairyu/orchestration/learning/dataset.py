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

from collections.abc import Iterable, Sequence
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


# --- m10b D8: placement records + offline (alpha, beta) grid tuning ---------


@dataclass(frozen=True)
class PlacementRecord:
    """One placement decision joined with its outcome (TTFT seconds)."""

    replica_id: str
    reason: str
    overlap_chunks: int
    outstanding: int
    ttft_s: float


def tune_prefix_weights(
    records: Sequence[PlacementRecord],
    alphas: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    betas: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
) -> tuple[float, float]:
    """Offline grid: pick (alpha, beta) minimizing predicted-mean TTFT.

    Pure counterfactual scoring over the logged records: a candidate weighting
    is judged by the mean TTFT of the records whose decision it AGREES with
    (records where alpha*overlap - beta*outstanding would have picked the same
    replica class — approximated by whether the record's own score ranks
    positive). Deliberately simple: the dataset is small and the goal is a
    starting point for the online bandit, not the bandit itself.
    """
    if not records:
        raise ValueError("no placement records to tune on")
    best: tuple[float, tuple[float, float]] | None = None
    for alpha in alphas:
        for beta in betas:
            agreeing = [
                record.ttft_s
                for record in records
                if (alpha * record.overlap_chunks - beta * record.outstanding > 0)
                == (record.reason == "prefix_match")
            ]
            if not agreeing:
                continue
            mean_ttft = sum(agreeing) / len(agreeing)
            coverage = len(agreeing) / len(records)
            score = mean_ttft / max(coverage, 1e-9)  # penalize tiny agreement sets
            if best is None or score < best[0]:
                best = (score, (alpha, beta))
    assert best is not None
    return best[1]
