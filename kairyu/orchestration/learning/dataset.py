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

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

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


# --- m10b D8: placement records + offline policy replay tuning --------------


PlacementSplit = Literal["train", "heldout"]


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def canonicalize_prefix_weights(alpha: float, beta: float) -> tuple[float, float]:
    """Return the one identifiable prefix-placement policy representation.

    ``alpha * overlap - beta * outstanding`` is unchanged when both weights
    are multiplied by the same positive constant.  The policy is therefore
    identified by ``beta / alpha``, not by the two magnitudes independently.
    Canonical records always use ``alpha=1``.
    """

    finite_alpha = _finite_number(alpha, "alpha")
    finite_beta = _finite_number(beta, "beta")
    if finite_alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if finite_beta < 0.0:
        raise ValueError("beta must be >= 0")
    ratio = finite_beta / finite_alpha
    if not math.isfinite(ratio):
        raise ValueError("beta / alpha must be finite")
    return 1.0, 0.0 if ratio == 0.0 else ratio


@dataclass(frozen=True)
class PlacementRecord:
    """One actually replayed placement decision joined with its outcome.

    This is an observed full-policy rollout row, not a counterfactual inferred
    from the chosen replica.  ``trace_item_id`` pairs the same exogenous work
    across independently reset policy arms; policy-specific ``request_id``
    joins the production placement JSONL row to the benchmark outcome row.
    """

    trace_item_id: str
    family_id: str
    split: PlacementSplit
    alpha: float
    beta: float
    request_id: str
    replica_id: str
    reason: str
    ttft_s: float
    success: bool

    def __post_init__(self) -> None:
        for field in (
            "trace_item_id",
            "family_id",
            "request_id",
            "replica_id",
            "reason",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        if self.split not in ("train", "heldout"):
            raise ValueError("split must be 'train' or 'heldout'")
        alpha, beta = canonicalize_prefix_weights(self.alpha, self.beta)
        ttft_s = _finite_number(self.ttft_s, "ttft_s")
        if ttft_s < 0.0:
            raise ValueError("ttft_s must be >= 0")
        if type(self.success) is not bool:
            raise ValueError("success must be a bool")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "ttft_s", ttft_s)


def _required_string(record: Mapping[str, object], field: str, *, kind: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} {field} must be a non-empty string")
    return value


def _validate_family_splits(records: Sequence[PlacementRecord]) -> None:
    split_by_family: dict[str, PlacementSplit] = {}
    identity_by_trace_item: dict[str, tuple[str, PlacementSplit]] = {}
    for record in records:
        family_split = split_by_family.setdefault(record.family_id, record.split)
        if family_split != record.split:
            raise ValueError(
                f"family {record.family_id!r} appears in both train and heldout"
            )
        identity = (record.family_id, record.split)
        previous = identity_by_trace_item.setdefault(record.trace_item_id, identity)
        if previous != identity:
            raise ValueError(
                f"trace item {record.trace_item_id!r} has inconsistent family or split"
            )


def build_placement_dataset(
    rows: Iterable[Mapping[str, object]],
) -> tuple[PlacementRecord, ...]:
    """Strictly join production placements with benchmark TTFT outcomes.

    Expected production row::

        {"kind": "replica", "request_id": "...", "replica_id": "...",
         "reason": "prefix_match", ...}

    Expected benchmark row::

        {"kind": "placement_outcome", "request_id": "...",
         "trace_item_id": "...", "family_id": "...",
         "split": "train", "alpha": 1.0, "beta": 0.25,
         "replica_id": "...", "ttft_s": 0.012, "success": True}

    Unrelated JSONL kinds are ignored.  Within the two placement kinds every
    request ID must occur exactly once on each side, and the backend that
    produced the outcome must equal the production-selected replica.
    """

    routes: dict[str, Mapping[str, object]] = {}
    outcomes: dict[str, Mapping[str, object]] = {}
    route_order: list[str] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_index} must be a mapping")
        kind = row.get("kind")
        if kind not in ("replica", "placement_outcome"):
            continue
        request_id = _required_string(row, "request_id", kind=str(kind))
        destination = routes if kind == "replica" else outcomes
        if request_id in destination:
            raise ValueError(f"duplicate {kind} row for request_id {request_id!r}")
        destination[request_id] = row
        if kind == "replica":
            route_order.append(request_id)

    if not routes and not outcomes:
        raise ValueError("no placement rows")
    missing_outcomes = set(routes) - set(outcomes)
    missing_routes = set(outcomes) - set(routes)
    if missing_outcomes or missing_routes:
        details = []
        if missing_outcomes:
            details.append(
                "missing outcomes for " + ", ".join(sorted(missing_outcomes))
            )
        if missing_routes:
            details.append("missing routes for " + ", ".join(sorted(missing_routes)))
        raise ValueError("placement rows are not one-to-one: " + "; ".join(details))

    joined: list[PlacementRecord] = []
    for request_id in route_order:
        route = routes[request_id]
        outcome = outcomes[request_id]
        route_replica = _required_string(route, "replica_id", kind="replica")
        outcome_replica = _required_string(
            outcome, "replica_id", kind="placement_outcome"
        )
        if route_replica != outcome_replica:
            raise ValueError(
                f"request {request_id!r} route selected {route_replica!r} "
                f"but outcome came from {outcome_replica!r}"
            )
        joined.append(
            PlacementRecord(
                trace_item_id=_required_string(
                    outcome, "trace_item_id", kind="placement_outcome"
                ),
                family_id=_required_string(
                    outcome, "family_id", kind="placement_outcome"
                ),
                split=outcome.get("split"),  # type: ignore[arg-type]
                alpha=outcome.get("alpha"),  # type: ignore[arg-type]
                beta=outcome.get("beta"),  # type: ignore[arg-type]
                request_id=request_id,
                replica_id=route_replica,
                reason=_required_string(route, "reason", kind="replica"),
                ttft_s=outcome.get("ttft_s"),  # type: ignore[arg-type]
                success=outcome.get("success"),  # type: ignore[arg-type]
            )
        )
    _validate_family_splits(joined)
    return tuple(joined)


def _balanced_training_panel(
    records: Sequence[PlacementRecord],
) -> dict[tuple[float, float], dict[str, PlacementRecord]]:
    if not records:
        raise ValueError("no placement records to tune on")
    if any(not isinstance(record, PlacementRecord) for record in records):
        raise ValueError("all records must be PlacementRecord instances")
    _validate_family_splits(records)

    request_ids: set[str] = set()
    for record in records:
        if record.request_id in request_ids:
            raise ValueError(f"duplicate request_id {record.request_id!r}")
        request_ids.add(record.request_id)
        if not record.success:
            raise ValueError(f"request {record.request_id!r} did not succeed")

    panel: dict[tuple[float, float], dict[str, PlacementRecord]] = {}
    for record in records:
        if record.split != "train":
            continue
        policy = canonicalize_prefix_weights(record.alpha, record.beta)
        arm = panel.setdefault(policy, {})
        if record.trace_item_id in arm:
            raise ValueError(
                f"policy {policy} has duplicate trace item {record.trace_item_id!r}"
            )
        arm[record.trace_item_id] = record
    if len(panel) < 2:
        raise ValueError("tuning requires at least two train policy arms")

    expected_items: set[str] | None = None
    expected_policy: tuple[float, float] | None = None
    for policy, arm in panel.items():
        items = set(arm)
        if expected_items is None:
            expected_items = items
            expected_policy = policy
        elif items != expected_items:
            missing = expected_items - items
            extra = items - expected_items
            raise ValueError(
                f"unbalanced train panel for policy {policy}: "
                f"missing={sorted(missing)}, extra={sorted(extra)} "
                f"relative to {expected_policy}"
            )
    if not expected_items:
        raise ValueError("train policy arms contain no trace items")
    return panel


def tune_prefix_weights(
    records: Sequence[PlacementRecord],
) -> tuple[float, float]:
    """Choose the lowest observed mean-TTFT arm on a balanced train replay.

    Every candidate policy must have been independently rolled out over the
    exact same train ``trace_item_id`` set.  Held-out rows are validated for
    integrity and family isolation but never enter selection.  Equal observed
    means choose the smaller canonical ``beta / alpha`` ratio.
    """

    panel = _balanced_training_panel(records)
    means = {
        policy: math.fsum(record.ttft_s for record in arm.values()) / len(arm)
        for policy, arm in panel.items()
    }
    return min(panel, key=lambda policy: (means[policy], policy[1]))
