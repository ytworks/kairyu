"""Paired statistical primitives for configuration quality comparisons."""

from __future__ import annotations

import json

import pytest

from kairyu.bench.config_compare import (
    _lower_nearest_rank,
    _SplitMix64,
    _symmetric_nearest_rank_interval,
    newcombe_paired_binary,
    paired_bounded_bootstrap,
    paired_clustered_bounded_bootstrap,
)


def test_newcombe_method_10_matches_published_table_value():
    result = newcombe_paired_binary(
        both_pass=36,
        candidate_win=12,
        candidate_loss=2,
        both_fail=0,
    )

    assert result["delta"] == pytest.approx(0.2)
    assert result["two_sided"] == pytest.approx(
        {"lower": 0.0569295840032388, "upper": 0.3404276029749407}
    )
    assert result["one_sided"]["lower"] == pytest.approx(0.0816654356579504)
    json.dumps(result, allow_nan=False)


def test_newcombe_method_10_exercises_published_positive_correlation_correction():
    result = newcombe_paired_binary(
        both_pass=20,
        candidate_win=12,
        candidate_loss=2,
        both_fail=16,
    )

    assert result["two_sided"] == pytest.approx(
        {"lower": 0.05615642484, "upper": 0.32920699540}
    )


def test_newcombe_arm_reversal_negates_delta_and_bounds():
    forward = newcombe_paired_binary(
        both_pass=60,
        candidate_win=10,
        candidate_loss=15,
        both_fail=15,
    )
    reverse = newcombe_paired_binary(
        both_pass=60,
        candidate_win=15,
        candidate_loss=10,
        both_fail=15,
    )

    assert reverse["delta"] == pytest.approx(-forward["delta"])
    assert reverse["two_sided"]["lower"] == pytest.approx(
        -forward["two_sided"]["upper"]
    )
    assert reverse["two_sided"]["upper"] == pytest.approx(
        -forward["two_sided"]["lower"]
    )
    assert reverse["one_sided"]["lower"] == pytest.approx(
        -forward["one_sided"]["upper"]
    )


def test_newcombe_all_concordant_items_retain_sampling_uncertainty():
    result = newcombe_paired_binary(
        both_pass=54,
        candidate_win=0,
        candidate_loss=0,
        both_fail=0,
    )

    assert result["delta"] == 0.0
    assert result["two_sided"] == pytest.approx(
        {"lower": -0.06641358809089626, "upper": 0.06641358809089626}
    )
    assert result["one_sided"]["lower"] == pytest.approx(-0.0477121510401467)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("both_pass", True, "non-negative integer"),
        ("candidate_win", 1.0, "non-negative integer"),
        ("candidate_loss", -1, "non-negative integer"),
        ("both_fail", 1 << 53, "JSON-safe integer"),
    ],
)
def test_newcombe_rejects_non_strict_counts(field, value, message):
    counts = {
        "both_pass": 1,
        "candidate_win": 1,
        "candidate_loss": 1,
        "both_fail": 1,
    }
    counts[field] = value

    with pytest.raises(ValueError, match=message):
        newcombe_paired_binary(**counts)


def test_newcombe_rejects_empty_or_unsafe_total():
    with pytest.raises(ValueError, match="at least one trial"):
        newcombe_paired_binary(
            both_pass=0,
            candidate_win=0,
            candidate_loss=0,
            both_fail=0,
        )
    with pytest.raises(ValueError, match="trial total.*JSON-safe"):
        newcombe_paired_binary(
            both_pass=(1 << 53) - 1,
            candidate_win=1,
            candidate_loss=0,
            both_fail=0,
        )


def _bootstrap(
    baseline=(0.0, 0.2, 0.8, 1.0),
    candidate=(0.1, 0.5, 0.4, 0.9),
    item_ids=("z", "a", "m", "q"),
    baseline_hash="a" * 64,
    candidate_hash="b" * 64,
):
    return paired_bounded_bootstrap(
        baseline,
        candidate,
        item_ids=item_ids,
        baseline_pair_sha256=baseline_hash,
        candidate_pair_sha256=candidate_hash,
    )


def test_paired_bootstrap_is_reproducible_order_independent_and_json_safe():
    expected = _bootstrap()
    repeated = _bootstrap()
    reordered = _bootstrap(
        baseline=(0.8, 0.0, 1.0, 0.2),
        candidate=(0.4, 0.1, 0.9, 0.5),
        item_ids=("m", "z", "q", "a"),
    )

    assert repeated == expected
    assert reordered == expected
    assert expected["item_ids_sha256"] == (
        "b4917cf3da212736a02783a6479dc12ad5baa6fd282f9da69712e39d8e2d3405"
    )
    assert expected["seed_sha256"] == (
        "f866cb4506a13d04874008dbd1749af01c2b7a570aa22fa9aaa1f9dab913bc5b"
    )
    assert expected["delta"] == pytest.approx(-0.025)
    assert expected["two_sided"] == pytest.approx({"lower": -0.275, "upper": 0.2})
    assert expected["one_sided"] == pytest.approx({"lower": -0.25, "upper": 0.2})
    assert expected["resamples"] == 20_000
    assert expected["quantile_method"] == "symmetric_nearest_rank_v1"
    assert expected["prng"] == "splitmix64_rejection_v1"
    json.dumps(expected, allow_nan=False)


def test_splitmix64_has_a_fixed_cross_python_reference_sequence():
    generator = _SplitMix64(0)

    assert [generator._next_u64() for _ in range(3)] == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]


def test_paired_bootstrap_arm_reversal_reuses_seed_and_negates_interval():
    forward = _bootstrap()
    reverse = _bootstrap(
        baseline=(0.1, 0.5, 0.4, 0.9),
        candidate=(0.0, 0.2, 0.8, 1.0),
        baseline_hash="b" * 64,
        candidate_hash="a" * 64,
    )

    assert reverse["seed_sha256"] == forward["seed_sha256"]
    assert reverse["item_ids_sha256"] == forward["item_ids_sha256"]
    assert reverse["delta"] == pytest.approx(-forward["delta"])
    assert reverse["two_sided"]["lower"] == pytest.approx(
        -forward["two_sided"]["upper"]
    )
    assert reverse["two_sided"]["upper"] == pytest.approx(
        -forward["two_sided"]["lower"]
    )
    assert reverse["one_sided"]["lower"] == pytest.approx(
        -forward["one_sided"]["upper"]
    )


def test_nearest_rank_uses_integer_tail_ranks_and_symmetric_endpoints():
    values = tuple(float(value) for value in range(80))

    assert _lower_nearest_rank(values, numerator=1, denominator=40) == 1.0
    assert _lower_nearest_rank(values, numerator=1, denominator=20) == 3.0
    assert _symmetric_nearest_rank_interval(
        values, tail_numerator=1, tail_denominator=40
    ) == (1.0, 78.0)


@pytest.mark.parametrize(
    ("baseline", "candidate", "item_ids", "message"),
    [
        ((), (), (), "at least one item"),
        ((0.0,), (0.0, 1.0), ("a",), "equal lengths"),
        ((0.0,), (0.0,), ("",), "non-empty strings"),
        ((0.0, 1.0), (0.0, 1.0), ("a", "a"), "unique"),
        ((True,), (0.0,), ("a",), "finite number"),
        ((float("nan"),), (0.0,), ("a",), "finite number"),
        ((0.0,), (float("inf"),), ("a",), "finite number"),
        ((-0.01,), (0.0,), ("a",), "finite number"),
        ((0.0,), (1.01,), ("a",), "finite number"),
    ],
)
def test_paired_bootstrap_rejects_malformed_evidence(
    baseline, candidate, item_ids, message
):
    with pytest.raises(ValueError, match=message):
        _bootstrap(baseline=baseline, candidate=candidate, item_ids=item_ids)


@pytest.mark.parametrize("digest", ["not-a-digest", "A" * 64, 123])
def test_paired_bootstrap_requires_canonical_pair_hashes(digest):
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _bootstrap(baseline_hash=digest)


def _cluster_bootstrap(*, reverse: bool = False):
    baseline = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    candidate = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    baseline_hash = "a" * 64
    candidate_hash = "b" * 64
    if reverse:
        baseline, candidate = candidate, baseline
        baseline_hash, candidate_hash = candidate_hash, baseline_hash
    return paired_clustered_bounded_bootstrap(
        baseline,
        candidate,
        item_ids=("p1.1", "p1.2", "p2.1", "p3.1", "p3.2", "p3.3"),
        cluster_ids=("p1", "p1", "p2", "p3", "p3", "p3"),
        baseline_pair_sha256=baseline_hash,
        candidate_pair_sha256=candidate_hash,
    )


def test_cluster_bootstrap_resamples_whole_clusters_and_preserves_item_estimand():
    result = _cluster_bootstrap()
    reverse = _cluster_bootstrap(reverse=True)
    reordered = paired_clustered_bounded_bootstrap(
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        item_ids=("p3.3", "p3.2", "p3.1", "p2.1", "p1.2", "p1.1"),
        cluster_ids=("p3", "p3", "p3", "p2", "p1", "p1"),
        baseline_pair_sha256="a" * 64,
        candidate_pair_sha256="b" * 64,
    )

    assert reordered == result
    assert result["method"] == "paired_cluster_percentile_bootstrap_v1"
    assert result["n_pairs"] == 6
    assert result["n_clusters"] == 3
    assert result["delta"] == pytest.approx(2 / 6)
    assert result["one_sided"]["lower"] == pytest.approx(-0.6)
    assert result["prng"] == "splitmix64_rejection_v1"
    assert result["seed_sha256"] == (
        "83f4283f14b1bd5812eb7aafad78f63b72e9fa9424ebd8fbd71df4cfe81c54e5"
    )
    assert result["item_clusters_sha256"] == (
        "0c89f7f695a16260438681b7d3ad9cc041af378c1e0c24191a8dbc9ea1cfd6bd"
    )
    assert reverse["seed_sha256"] == result["seed_sha256"]
    assert reverse["delta"] == pytest.approx(-result["delta"])
    assert reverse["two_sided"]["lower"] == pytest.approx(
        -result["two_sided"]["upper"]
    )
    assert reverse["two_sided"]["upper"] == pytest.approx(
        -result["two_sided"]["lower"]
    )


def test_cluster_bootstrap_rejects_a_single_dependent_cluster():
    with pytest.raises(ValueError, match="at least two clusters"):
        paired_clustered_bounded_bootstrap(
            (0.0, 1.0),
            (1.0, 0.0),
            item_ids=("p1.1", "p1.2"),
            cluster_ids=("p1", "p1"),
            baseline_pair_sha256="a" * 64,
            candidate_pair_sha256="b" * 64,
        )
