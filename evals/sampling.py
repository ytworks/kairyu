"""Fail-closed sampling-sensitivity evidence and statistics.

Chat attempts are correlated repeats of one dataset item, not additional
dataset items.  This module keeps that boundary explicit and recomputes every
reported statistic from the retained child outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import mean, stdev

from evals.types import (
    JSON_SAFE_INTEGER_MAX,
    ItemResult,
    SamplingAttemptResult,
)

PROTOCOL_ID = "grouped-chat-seeds-v1"


def seed_schedule(target_seed: int | None, attempts: int) -> tuple[int | None, ...]:
    """Return the exact chat seed sent for each attempt.

    A single attempt preserves omission: an unset target seed remains absent
    from the wire.  A sensitivity run needs distinct streams, so attempts > 1
    use consecutive seeds from the declared target seed, or zero when absent.
    """

    if (
        type(attempts) is not int
        or attempts < 1
        or attempts > JSON_SAFE_INTEGER_MAX
    ):
        raise ValueError("attempts must be a positive JSON-safe integer")
    if target_seed is not None and type(target_seed) is not int:
        raise ValueError("target seed must be an integer")
    if attempts == 1:
        if target_seed is not None and abs(target_seed) > JSON_SAFE_INTEGER_MAX:
            raise ValueError("target seed must be a JSON-safe integer")
        return (target_seed,)
    start = target_seed if target_seed is not None else 0
    if (
        abs(start) > JSON_SAFE_INTEGER_MAX
        or start + attempts - 1 > JSON_SAFE_INTEGER_MAX
    ):
        raise ValueError("sampling seed schedule must contain only JSON-safe integers")
    return tuple(start + index for index in range(attempts))


def combine_attempts(
    item_id: str,
    attempts: Sequence[SamplingAttemptResult],
) -> ItemResult:
    """Build one source-item result from its ordered seed-specific outcomes."""

    retained = tuple(attempts)
    if not retained:
        raise ValueError("sampling attempts must not be empty")
    results = [attempt.result for attempt in retained]
    complete = all(result.status == "completed" and result.score is not None for result in results)
    latency_values = [result.latency_s for result in results if result.latency_s is not None]
    latency = round(sum(latency_values), 3) if latency_values else None
    if complete:
        scores = [float(result.score) for result in results if result.score is not None]
        return ItemResult(
            item_id=item_id,
            status="completed",
            score=sum(scores) / len(scores),
            latency_s=latency,
            sampling_attempts=retained,
        )

    counts = {
        status: sum(result.status == status for result in results)
        for status in ("failed", "unjudged", "skipped")
    }
    missing_scores = sum(
        result.status == "completed" and result.score is None for result in results
    )
    if missing_scores:
        counts["invalid"] = missing_scores
    if counts["failed"] or missing_scores:
        status = "failed"
    elif counts["unjudged"]:
        status = "unjudged"
    else:
        status = "skipped"
    reason = ", ".join(f"{count} {name}" for name, count in counts.items() if count)
    return ItemResult(
        item_id=item_id,
        status=status,
        error=f"sampling sweep incomplete: {reason}",
        latency_s=latency,
        sampling_attempts=retained,
    )


def pass_at_k(successes: int, attempts: int, k: int) -> float:
    """Unbiased pass@k estimator ``1 - C(n-c,k) / C(n,k)``.

    The product form avoids constructing large combinations and retains exact
    zero/one endpoints before the final floating-point conversion.
    """

    if any(type(value) is not int for value in (successes, attempts, k)):
        raise ValueError("pass@k inputs must be integers")
    if attempts < 1 or not 0 <= successes <= attempts or not 1 <= k <= attempts:
        raise ValueError("pass@k inputs are outside their valid ranges")
    failures = attempts - successes
    if failures < k:
        return 1.0
    all_fail_probability = 1.0
    for offset in range(k):
        all_fail_probability *= (failures - offset) / (attempts - offset)
    return 1.0 - all_fail_probability


def reported_k_values(attempts: int) -> tuple[int, ...]:
    """Report powers of two plus the actual attempt budget."""

    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    values: list[int] = []
    value = 1
    while value <= attempts:
        values.append(value)
        value *= 2
    if values[-1] != attempts:
        values.append(attempts)
    return tuple(values)


def sampling_summary(
    items: Sequence[ItemResult],
    *,
    seeds: Sequence[int],
    declared_binary: bool,
) -> dict:
    """Recompute the complete seed matrix summary or raise ``ValueError``.

    No statistic is returned for partial/malformed evidence.  Callers may keep
    the ordinary visibly-partial pair score, but must not label it a sampling
    sensitivity measurement.
    """

    expected_seeds = tuple(seeds)
    if type(declared_binary) is not bool:
        raise ValueError("declared_binary must be a boolean")
    if any(type(seed) is not int for seed in expected_seeds):
        raise ValueError("sampling sensitivity seeds must be integers")
    if any(abs(seed) > JSON_SAFE_INTEGER_MAX for seed in expected_seeds):
        raise ValueError("sampling sensitivity seeds must be JSON-safe integers")
    attempts = len(expected_seeds)
    if attempts < 2 or len(set(expected_seeds)) != attempts:
        raise ValueError("sampling sensitivity requires at least two unique seeds")
    if not items:
        raise ValueError("sampling sensitivity requires at least one source item")
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("sampling sensitivity source item ids must be unique")

    columns: list[list[float]] = [[] for _ in expected_seeds]
    success_counts: list[int] = []
    for item in items:
        retained = item.sampling_attempts
        if item.status != "completed" or item.score is None or retained is None:
            raise ValueError("sampling sensitivity requires complete source items")
        if len(retained) != attempts:
            raise ValueError("sampling attempt count does not match the seed schedule")
        if tuple(attempt.seed for attempt in retained) != expected_seeds:
            raise ValueError("sampling attempt seeds do not match the ordered schedule")
        scores: list[float] = []
        for index, attempt in enumerate(retained):
            result = attempt.result
            if (
                attempt.attempt != index + 1
                or result.item_id != item.item_id
                or result.status != "completed"
                or result.score is None
            ):
                raise ValueError("sampling attempt evidence is incomplete or misaligned")
            score = float(result.score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("sampling attempt score is outside [0, 1]")
            if declared_binary and score not in (0.0, 1.0):
                raise ValueError("binary sampling evidence contains a non-binary score")
            columns[index].append(score)
            scores.append(score)
        if not math.isclose(item.score, mean(scores), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("sampling parent score does not equal its attempt mean")
        if declared_binary:
            success_counts.append(sum(score == 1.0 for score in scores))

    seed_scores = [mean(column) for column in columns]
    result = {
        "attempts": attempts,
        "seeds": list(expected_seeds),
        "n_items": len(items),
        "seed_scores": [
            {"seed": seed, "score": score}
            for seed, score in zip(expected_seeds, seed_scores, strict=True)
        ],
        "mean": mean(seed_scores),
        "sample_sd": stdev(seed_scores),
        "min": min(seed_scores),
        "max": max(seed_scores),
        "pass_at_k": None,
        "item_successes": None,
    }
    if declared_binary:
        result["item_successes"] = success_counts
        result["pass_at_k"] = [
            {
                "k": k,
                "estimate": mean(
                    pass_at_k(successes, attempts, k) for successes in success_counts
                ),
            }
            for k in reported_k_values(attempts)
        ]
    return result


def sampling_metrics(summary: dict) -> dict[str, float | int]:
    """Flatten a validated summary into PairResult's scalar metric namespace."""

    metrics: dict[str, float | int] = {
        "sampling_attempts": int(summary["attempts"]),
        "sampling_base_items": int(summary["n_items"]),
        "sampling_seed_score_mean": float(summary["mean"]),
        "sampling_seed_score_sample_sd": float(summary["sample_sd"]),
        "sampling_seed_score_min": float(summary["min"]),
        "sampling_seed_score_max": float(summary["max"]),
    }
    for entry in summary.get("pass_at_k") or ():
        metrics[f"pass_at_{entry['k']}"] = float(entry["estimate"])
    return metrics
