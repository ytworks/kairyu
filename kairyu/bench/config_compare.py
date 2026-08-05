"""Paired statistical primitives for configuration A/B quality comparisons.

The functions in this module deliberately operate on complete, already paired
item evidence.  Loading runs, proving provenance, and deciding which rows gate
belong to the comparison orchestration rather than to these calculations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from statistics import NormalDist

_CONFIDENCE = 0.95
_TWO_SIDED_Z = NormalDist().inv_cdf(0.5 + _CONFIDENCE / 2)
_ONE_SIDED_Z = NormalDist().inv_cdf(_CONFIDENCE)
_BOOTSTRAP_RESAMPLES = 20_000
_BOOTSTRAP_PRNG = "splitmix64_rejection_v1"
_MAX_SAFE_INTEGER = (1 << 53) - 1
_UINT64_RANGE = 1 << 64
_UINT64_MASK = _UINT64_RANGE - 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class _SplitMix64:
    """Small versioned PRNG used only for reproducible bootstrap indices."""

    def __init__(self, seed: int) -> None:
        self._state = seed & _UINT64_MASK

    def _next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _UINT64_MASK
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        return (value ^ (value >> 31)) & _UINT64_MASK

    def randbelow(self, bound: int) -> int:
        """Return an unbiased value in ``range(bound)`` by rejection sampling."""
        if type(bound) is not int or not 0 < bound <= _UINT64_RANGE:
            raise ValueError("SplitMix64 bound must be within [1, 2**64]")
        cutoff = _UINT64_RANGE - (_UINT64_RANGE % bound)
        while True:
            value = self._next_u64()
            if value < cutoff:
                return value % bound


def _strict_count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if value > _MAX_SAFE_INTEGER:
        raise ValueError(f"{name} must be a JSON-safe integer")
    return value


def _strict_score(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number within [0, 1]")
    if not 0 <= value <= 1:
        raise ValueError(f"{label} must be a finite number within [0, 1]")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number within [0, 1]")
    return number


def _wilson_bounds(successes: int, trials: int, z: float) -> tuple[float, float]:
    proportion = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z_squared / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _newcombe_bounds(
    *,
    both_pass: int,
    candidate_win: int,
    candidate_loss: int,
    both_fail: int,
    z: float,
) -> tuple[float, float]:
    """Newcombe (1998) paired-proportion method 10 for one z critical value."""
    trials = both_pass + candidate_win + candidate_loss + both_fail
    candidate = (both_pass + candidate_win) / trials
    baseline = (both_pass + candidate_loss) / trials
    candidate_lower, candidate_upper = _wilson_bounds(
        both_pass + candidate_win, trials, z
    )
    baseline_lower, baseline_upper = _wilson_bounds(
        both_pass + candidate_loss, trials, z
    )

    candidate_lower_distance = candidate - candidate_lower
    candidate_upper_distance = candidate_upper - candidate
    baseline_lower_distance = baseline - baseline_lower
    baseline_upper_distance = baseline_upper - baseline

    correlation_denominator = math.sqrt(
        (both_pass + candidate_win)
        * (candidate_loss + both_fail)
        * (both_pass + candidate_loss)
        * (candidate_win + both_fail)
    )
    correlation_numerator = both_pass * both_fail - candidate_win * candidate_loss
    if correlation_denominator == 0:
        correlation = 0.0
    elif correlation_numerator > 0:
        # Method 10 is method 8 with this positive-correlation continuity
        # correction.  Negative correlations retain their signed numerator.
        correlation = max(correlation_numerator - trials / 2, 0.0) / (
            correlation_denominator
        )
    else:
        correlation = correlation_numerator / correlation_denominator
    correlation = max(-1.0, min(1.0, correlation))

    delta = candidate - baseline
    lower_radicand = (
        candidate_lower_distance**2
        - 2
        * correlation
        * candidate_lower_distance
        * baseline_upper_distance
        + baseline_upper_distance**2
    )
    upper_radicand = (
        candidate_upper_distance**2
        - 2
        * correlation
        * candidate_upper_distance
        * baseline_lower_distance
        + baseline_lower_distance**2
    )
    lower = delta - math.sqrt(max(0.0, lower_radicand))
    upper = delta + math.sqrt(max(0.0, upper_radicand))
    return max(-1.0, lower), min(1.0, upper)


def newcombe_paired_binary(
    *,
    both_pass: int,
    candidate_win: int,
    candidate_loss: int,
    both_fail: int,
) -> dict[str, object]:
    """Return paired binary delta, two-sided 95% CI, and one-sided 95% bounds.

    The delta is always candidate minus baseline.  The lower member of
    ``one_sided`` is the binding non-inferiority lower confidence bound.
    """
    counts = {
        "both_pass": _strict_count(both_pass, "both_pass"),
        "candidate_win": _strict_count(candidate_win, "candidate_win"),
        "candidate_loss": _strict_count(candidate_loss, "candidate_loss"),
        "both_fail": _strict_count(both_fail, "both_fail"),
    }
    trials = sum(counts.values())
    if trials <= 0:
        raise ValueError("paired binary evidence must contain at least one trial")
    if trials > _MAX_SAFE_INTEGER:
        raise ValueError("paired binary trial total must be a JSON-safe integer")

    two_sided = _newcombe_bounds(**counts, z=_TWO_SIDED_Z)
    one_sided = _newcombe_bounds(**counts, z=_ONE_SIDED_Z)
    delta = (counts["candidate_win"] - counts["candidate_loss"]) / trials
    if delta == 0:
        delta = 0.0
    result: dict[str, object] = {
        "method": "newcombe_wilson_paired_method_10",
        "confidence": _CONFIDENCE,
        "trials": trials,
        "counts": counts,
        "delta": delta,
        "two_sided": {"lower": two_sided[0], "upper": two_sided[1]},
        "one_sided": {"lower": one_sided[0], "upper": one_sided[1]},
    }
    # Keep the public return contract safely serializable even if a future
    # calculation change accidentally introduces NaN or infinity.
    json.dumps(result, allow_nan=False)
    return result


def _digest_item_ids(item_ids: Sequence[str]) -> str:
    canonical = json.dumps(
        sorted(item_ids),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _bootstrap_seed(
    *, baseline_pair_sha256: str, candidate_pair_sha256: str, item_ids_sha256: str
) -> str:
    pair_hashes = sorted((baseline_pair_sha256, candidate_pair_sha256))
    payload = {
        "item_ids_sha256": item_ids_sha256,
        "method": "paired_percentile_bootstrap_v1",
        "pair_sha256s": pair_hashes,
        "prng": _BOOTSTRAP_PRNG,
        "resamples": _BOOTSTRAP_RESAMPLES,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _cluster_bootstrap_seed(
    *,
    baseline_pair_sha256: str,
    candidate_pair_sha256: str,
    item_clusters_sha256: str,
) -> str:
    payload = {
        "item_clusters_sha256": item_clusters_sha256,
        "method": "paired_cluster_percentile_bootstrap_v1",
        "pair_sha256s": sorted((baseline_pair_sha256, candidate_pair_sha256)),
        "prng": _BOOTSTRAP_PRNG,
        "resamples": _BOOTSTRAP_RESAMPLES,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _lower_nearest_rank(
    sorted_values: Sequence[float], *, numerator: int, denominator: int
) -> float:
    """Return the lower nearest-rank quantile ``numerator / denominator``."""
    if not sorted_values:
        raise ValueError("nearest-rank input must not be empty")
    if numerator <= 0 or denominator <= 0 or numerator >= denominator:
        raise ValueError("nearest-rank probability must be strictly within (0, 1)")
    rank = (numerator * len(sorted_values) + denominator - 1) // denominator
    return float(sorted_values[rank - 1])


def _symmetric_nearest_rank_interval(
    sorted_values: Sequence[float], *, tail_numerator: int, tail_denominator: int
) -> tuple[float, float]:
    """Central interval with equal, nearest-rank empirical tails.

    Selecting the upper order statistic by the reflected lower-tail rank makes
    arm reversal exact: an interval for ``candidate - baseline`` negates to the
    reversed interval without a one-order-statistic asymmetry.
    """
    lower = _lower_nearest_rank(
        sorted_values,
        numerator=tail_numerator,
        denominator=tail_denominator,
    )
    rank = (
        tail_numerator * len(sorted_values) + tail_denominator - 1
    ) // tail_denominator
    upper = float(sorted_values[len(sorted_values) - rank])
    return lower, upper


def paired_bounded_bootstrap(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    item_ids: Sequence[str],
    baseline_pair_sha256: str,
    candidate_pair_sha256: str,
) -> dict[str, object]:
    """Return a deterministic paired percentile-bootstrap interval.

    Scores are bounded to ``[0, 1]`` and aligned by the supplied item IDs.  The
    function sorts those identities before resampling, so persisted evidence
    order cannot change the result.  Both arms always reuse the same sampled
    indices.
    """
    baseline_values = tuple(baseline_scores)
    candidate_values = tuple(candidate_scores)
    identities = tuple(item_ids)
    if not baseline_values:
        raise ValueError("paired bootstrap evidence must contain at least one item")
    if len(baseline_values) != len(candidate_values) or len(baseline_values) != len(
        identities
    ):
        raise ValueError("baseline scores, candidate scores, and item IDs must have equal lengths")
    if any(not isinstance(item_id, str) or not item_id for item_id in identities):
        raise ValueError("item IDs must be non-empty strings")
    if len(identities) != len(set(identities)):
        raise ValueError("item IDs must be unique")
    for label, digest in (
        ("baseline_pair_sha256", baseline_pair_sha256),
        ("candidate_pair_sha256", candidate_pair_sha256),
    ):
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    paired: list[tuple[str, float, float]] = []
    for index, (item_id, baseline, candidate) in enumerate(
        zip(identities, baseline_values, candidate_values, strict=True)
    ):
        paired.append(
            (
                item_id,
                _strict_score(baseline, f"baseline_scores[{index}]"),
                _strict_score(candidate, f"candidate_scores[{index}]"),
            )
        )
    paired.sort(key=lambda entry: entry[0])
    differences = tuple(candidate - baseline for _, baseline, candidate in paired)
    count = len(differences)
    delta = math.fsum(differences) / count
    if delta == 0:
        delta = 0.0

    item_ids_sha256 = _digest_item_ids(tuple(item_id for item_id, _, _ in paired))
    seed_sha256 = _bootstrap_seed(
        baseline_pair_sha256=baseline_pair_sha256,
        candidate_pair_sha256=candidate_pair_sha256,
        item_ids_sha256=item_ids_sha256,
    )
    rng = _SplitMix64(int(seed_sha256, 16))
    samples = [
        math.fsum(differences[rng.randbelow(count)] for _ in range(count)) / count
        for _ in range(_BOOTSTRAP_RESAMPLES)
    ]
    samples.sort()
    two_sided = _symmetric_nearest_rank_interval(
        samples,
        tail_numerator=1,
        tail_denominator=40,
    )
    one_sided = _symmetric_nearest_rank_interval(
        samples,
        tail_numerator=1,
        tail_denominator=20,
    )
    result: dict[str, object] = {
        "method": "paired_percentile_bootstrap_v1",
        "confidence": _CONFIDENCE,
        "n_pairs": count,
        "delta": delta,
        "two_sided": {"lower": two_sided[0], "upper": two_sided[1]},
        "one_sided": {"lower": one_sided[0], "upper": one_sided[1]},
        "resamples": _BOOTSTRAP_RESAMPLES,
        "quantile_method": "symmetric_nearest_rank_v1",
        "prng": _BOOTSTRAP_PRNG,
        "seed_sha256": seed_sha256,
        "item_ids_sha256": item_ids_sha256,
    }
    json.dumps(result, allow_nan=False)
    return result


def paired_clustered_bounded_bootstrap(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    item_ids: Sequence[str],
    cluster_ids: Sequence[str],
    baseline_pair_sha256: str,
    candidate_pair_sha256: str,
) -> dict[str, object]:
    """Bootstrap paired item differences by dependent evidence clusters.

    Clusters are sampled with replacement, while every item in a selected
    cluster travels together. The replicate remains an item-weighted ratio so
    the original per-item score estimand is unchanged when cluster sizes differ.
    """
    baseline_values = tuple(baseline_scores)
    candidate_values = tuple(candidate_scores)
    identities = tuple(item_ids)
    clusters = tuple(cluster_ids)
    if not baseline_values:
        raise ValueError("paired cluster bootstrap evidence must contain at least one item")
    if not (
        len(baseline_values)
        == len(candidate_values)
        == len(identities)
        == len(clusters)
    ):
        raise ValueError(
            "baseline scores, candidate scores, item IDs, and cluster IDs "
            "must have equal lengths"
        )
    if any(not isinstance(item_id, str) or not item_id for item_id in identities):
        raise ValueError("item IDs must be non-empty strings")
    if len(identities) != len(set(identities)):
        raise ValueError("item IDs must be unique")
    if any(not isinstance(cluster_id, str) or not cluster_id for cluster_id in clusters):
        raise ValueError("cluster IDs must be non-empty strings")
    for label, digest in (
        ("baseline_pair_sha256", baseline_pair_sha256),
        ("candidate_pair_sha256", candidate_pair_sha256),
    ):
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    paired: list[tuple[str, str, float]] = []
    for index, (item_id, cluster_id, baseline, candidate) in enumerate(
        zip(
            identities,
            clusters,
            baseline_values,
            candidate_values,
            strict=True,
        )
    ):
        baseline_score = _strict_score(baseline, f"baseline_scores[{index}]")
        candidate_score = _strict_score(candidate, f"candidate_scores[{index}]")
        paired.append((item_id, cluster_id, candidate_score - baseline_score))
    paired.sort(key=lambda entry: entry[0])

    grouped: dict[str, list[float]] = {}
    for _, cluster_id, difference in paired:
        grouped.setdefault(cluster_id, []).append(difference)
    if len(grouped) < 2:
        raise ValueError("paired cluster bootstrap requires at least two clusters")
    cluster_summaries = tuple(
        (cluster_id, math.fsum(grouped[cluster_id]), len(grouped[cluster_id]))
        for cluster_id in sorted(grouped)
    )
    total_items = len(paired)
    delta = math.fsum(difference for _, _, difference in paired) / total_items
    if delta == 0:
        delta = 0.0

    item_ids_sha256 = _digest_item_ids(tuple(item_id for item_id, _, _ in paired))
    item_clusters = [(item_id, cluster_id) for item_id, cluster_id, _ in paired]
    item_clusters_sha256 = hashlib.sha256(
        json.dumps(
            item_clusters,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    seed_sha256 = _cluster_bootstrap_seed(
        baseline_pair_sha256=baseline_pair_sha256,
        candidate_pair_sha256=candidate_pair_sha256,
        item_clusters_sha256=item_clusters_sha256,
    )
    rng = _SplitMix64(int(seed_sha256, 16))
    cluster_count = len(cluster_summaries)
    samples: list[float] = []
    for _ in range(_BOOTSTRAP_RESAMPLES):
        selected = [
            cluster_summaries[rng.randbelow(cluster_count)]
            for _ in range(cluster_count)
        ]
        numerator = math.fsum(summary[1] for summary in selected)
        denominator = sum(summary[2] for summary in selected)
        samples.append(numerator / denominator)
    samples.sort()
    two_sided = _symmetric_nearest_rank_interval(
        samples,
        tail_numerator=1,
        tail_denominator=40,
    )
    one_sided = _symmetric_nearest_rank_interval(
        samples,
        tail_numerator=1,
        tail_denominator=20,
    )
    result: dict[str, object] = {
        "method": "paired_cluster_percentile_bootstrap_v1",
        "confidence": _CONFIDENCE,
        "n_pairs": total_items,
        "n_clusters": cluster_count,
        "delta": delta,
        "two_sided": {"lower": two_sided[0], "upper": two_sided[1]},
        "one_sided": {"lower": one_sided[0], "upper": one_sided[1]},
        "resamples": _BOOTSTRAP_RESAMPLES,
        "quantile_method": "symmetric_nearest_rank_v1",
        "prng": _BOOTSTRAP_PRNG,
        "seed_sha256": seed_sha256,
        "item_ids_sha256": item_ids_sha256,
        "item_clusters_sha256": item_clusters_sha256,
    }
    json.dumps(result, allow_nan=False)
    return result


__all__ = [
    "newcombe_paired_binary",
    "paired_bounded_bootstrap",
    "paired_clustered_bounded_bootstrap",
]
