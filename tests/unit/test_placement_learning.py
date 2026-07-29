from __future__ import annotations

import math

import pytest

from kairyu.orchestration.learning.dataset import (
    PlacementRecord,
    build_placement_dataset,
    canonicalize_prefix_weights,
    tune_prefix_weights,
)


def _route(
    request_id: str,
    replica_id: str = "warm",
    reason: str = "prefix_match",
) -> dict[str, object]:
    return {
        "kind": "replica",
        "request_id": request_id,
        "replica_id": replica_id,
        "reason": reason,
    }


def _outcome(
    request_id: str,
    *,
    trace_item_id: str = "item-0",
    family_id: str = "family-0",
    split: str = "train",
    alpha: float = 1.0,
    beta: float = 0.25,
    replica_id: str = "warm",
    ttft_s: float = 0.1,
    success: bool = True,
) -> dict[str, object]:
    return {
        "kind": "placement_outcome",
        "request_id": request_id,
        "trace_item_id": trace_item_id,
        "family_id": family_id,
        "split": split,
        "alpha": alpha,
        "beta": beta,
        "replica_id": replica_id,
        "ttft_s": ttft_s,
        "success": success,
    }


def _record(
    policy_beta: float,
    trace_item_id: str,
    ttft_s: float,
    *,
    split: str = "train",
    family_id: str | None = None,
    alpha: float = 1.0,
    success: bool = True,
) -> PlacementRecord:
    family = family_id or f"family-{trace_item_id}"
    return PlacementRecord(
        trace_item_id=trace_item_id,
        family_id=family,
        split=split,
        alpha=alpha,
        beta=policy_beta,
        request_id=f"{split}-a{alpha}-b{policy_beta}-{trace_item_id}",
        replica_id="warm",
        reason="prefix_match",
        ttft_s=ttft_s,
        success=success,
    )


def test_build_placement_dataset_strictly_joins_route_and_outcome():
    rows = [
        {"kind": "run", "schema": 1},
        _route("request-1"),
        _outcome("request-1", alpha=2.0, beta=1.0),
    ]

    assert build_placement_dataset(rows) == (
        PlacementRecord(
            trace_item_id="item-0",
            family_id="family-0",
            split="train",
            alpha=1.0,
            beta=0.5,
            request_id="request-1",
            replica_id="warm",
            reason="prefix_match",
            ttft_s=0.1,
            success=True,
        ),
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_route("r")], "missing outcomes"),
        ([_outcome("r")], "missing routes"),
        ([_route("r"), _route("r"), _outcome("r")], "duplicate replica"),
        (
            [_route("r"), _outcome("r"), _outcome("r")],
            "duplicate placement_outcome",
        ),
        (
            [_route("r", replica_id="a"), _outcome("r", replica_id="b")],
            "outcome came from",
        ),
    ],
)
def test_build_placement_dataset_rejects_non_one_to_one_rows(rows, message):
    with pytest.raises(ValueError, match=message):
        build_placement_dataset(rows)


@pytest.mark.parametrize(
    ("alpha", "beta", "message"),
    [
        (0.0, 0.0, "alpha must be > 0"),
        (-1.0, 0.0, "alpha must be > 0"),
        (math.inf, 0.0, "alpha must be a finite"),
        (1.0, -0.1, "beta must be >= 0"),
        (1.0, math.nan, "beta must be a finite"),
    ],
)
def test_prefix_weight_canonicalization_rejects_invalid_weights(
    alpha, beta, message
):
    with pytest.raises(ValueError, match=message):
        canonicalize_prefix_weights(alpha, beta)


def test_prefix_weight_canonicalization_identifies_only_beta_over_alpha():
    assert canonicalize_prefix_weights(0.5, 0.25) == (1.0, 0.5)
    assert canonicalize_prefix_weights(2.0, 1.0) == (1.0, 0.5)


def test_tuner_uses_balanced_observed_train_rollouts_and_not_heldout():
    records = [
        _record(0.25, "a", 0.4),
        _record(0.25, "b", 0.6),
        _record(1.0, "a", 0.2),
        _record(1.0, "b", 0.3),
        _record(
            0.25,
            "heldout-a",
            0.01,
            split="heldout",
            family_id="heldout-family-a",
        ),
        _record(
            1.0,
            "heldout-a",
            100.0,
            split="heldout",
            family_id="heldout-family-a",
        ),
    ]

    assert tune_prefix_weights(records) == (1.0, 1.0)


def test_tuner_tie_breaks_toward_smaller_canonical_ratio():
    records = [
        _record(1.0, "a", 0.2),
        _record(1.0, "b", 0.4),
        _record(0.25, "a", 0.1),
        _record(0.25, "b", 0.5),
    ]

    assert tune_prefix_weights(records) == (1.0, 0.25)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "no placement records"),
        ([_record(0.25, "a", 0.1)], "at least two"),
        (
            [
                _record(0.25, "a", 0.1),
                _record(0.25, "b", 0.1),
                _record(1.0, "a", 0.1),
            ],
            "unbalanced train panel",
        ),
        (
            [
                _record(0.25, "a", 0.1),
                _record(0.5, "a", 0.2, alpha=2.0),
                _record(1.0, "a", 0.1),
            ],
            "duplicate trace item",
        ),
        (
            [
                _record(0.25, "a", 0.1, success=False),
                _record(1.0, "a", 0.1),
            ],
            "did not succeed",
        ),
    ],
)
def test_tuner_rejects_incomplete_or_failed_panels(records, message):
    with pytest.raises(ValueError, match=message):
        tune_prefix_weights(records)


def test_records_reject_nonfinite_or_negative_ttft():
    with pytest.raises(ValueError, match="finite"):
        _record(0.25, "a", math.nan)
    with pytest.raises(ValueError, match=">= 0"):
        _record(0.25, "a", -0.1)


def test_dataset_rejects_family_leakage_across_splits():
    rows = [
        _route("train"),
        _outcome("train", family_id="shared-family", split="train"),
        _route("heldout"),
        _outcome(
            "heldout",
            trace_item_id="heldout-item",
            family_id="shared-family",
            split="heldout",
        ),
    ]

    with pytest.raises(ValueError, match="both train and heldout"):
        build_placement_dataset(rows)


def test_equivalent_raw_weight_arms_cannot_duplicate_a_trace_item():
    records = [
        _record(0.25, "a", 0.1, alpha=0.5),
        _record(0.5, "a", 0.1, alpha=1.0),
        _record(0.0, "a", 0.2),
    ]

    with pytest.raises(ValueError, match="duplicate trace item"):
        tune_prefix_weights(records)
