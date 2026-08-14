import pytest

from verification.fleet.resilience.noisy_neighbor_bench import (
    Workload,
    percentile,
    run_gate,
    run_trace,
)


def test_noisy_neighbor_gate_protects_good_tenant_p99() -> None:
    result = run_gate()

    assert result["passed"]
    assert all(result["checks"].values())
    assert result["rates"]["noisy_offered_to_quota_ratio"] == 10.0
    assert (
        result["protected"]["tenants"]["good"]["ttft_p99_seconds"]
        - result["control"]["tenants"]["good"]["ttft_p99_seconds"]
        == 1
    )
    assert (
        result["protected"]["tenants"]["good"]["ttft_p99_seconds"]
        < result["unprotected"]["tenants"]["good"]["ttft_p99_seconds"]
    )


def test_rejected_noisy_requests_never_reach_shared_scheduler() -> None:
    protected = run_trace("protected")

    assert protected["tenants"]["noisy"]["rejected"] > 0
    assert protected["shared_replica"]["rejected_request_ids_scheduled"] == []
    assert (
        protected["shared_replica"]["scheduler"]["events"]["interactive:enqueue"]
        == protected["shared_replica"]["admitted_requests"]
    )


def test_workload_requires_exact_ten_x_noisy_rate() -> None:
    with pytest.raises(ValueError, match="exactly 10x"):
        Workload(noisy_release_period_s=2).validate()


def test_nearest_rank_percentile_rejects_invalid_inputs() -> None:
    assert percentile(list(range(1, 101)), 0.99) == 99
    with pytest.raises(ValueError, match="no values"):
        percentile([], 0.99)
    with pytest.raises(ValueError, match=r"in \(0, 1]"):
        percentile([1], 0.0)
