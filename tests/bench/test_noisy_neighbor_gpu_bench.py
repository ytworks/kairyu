from types import SimpleNamespace

from bench.noisy_neighbor_gpu_bench import (
    _contains_tensor_parallel_size,
    _gateway_contract,
    _labeled_count,
    _single_labeled_value,
    metric_delta,
    percentile,
)


def test_gpu_bench_percentile_uses_raw_nearest_rank_samples() -> None:
    assert percentile([float(index) for index in range(1, 101)], 0.99) == 99.0


def test_gpu_bench_metric_delta_and_bounded_label_matching() -> None:
    admitted = (
        'kairyu_tenant_admission_total{decision="admitted",'
        'reason="quota_available",source="http",tenant="noisy"}'
    )
    rejected = (
        'kairyu_tenant_admission_total{decision="rejected",'
        'reason="request_quota",source="http",tenant="noisy"}'
    )
    before = f"{admitted} 2\n"
    after = f"{admitted} 5\n{rejected} 7\n"
    delta = metric_delta(before, after, "kairyu_tenant_admission_total")

    assert _labeled_count(
        delta,
        tenant="noisy",
        source="http",
        decision="admitted",
    ) == 3.0
    assert _labeled_count(
        delta,
        tenant="noisy",
        source="http",
        decision="rejected",
    ) == 7.0


def test_gpu_bench_reads_limits_from_the_executed_gateway_config(tmp_path) -> None:
    config = tmp_path / "gateway.yaml"
    config.write_text(
        """
tenants:
  key_tenants: {good-key: good, noisy-key: noisy}
  limits:
    good:
      requests_per_minute: 600
      tokens_per_minute: 1000000
      request_burst: 64
      token_burst: 1000000
      max_in_flight: 64
      interactive_priority: 0
    noisy:
      requests_per_minute: 60
      tokens_per_minute: 1000000
      request_burst: 1
      token_burst: 1000000
      max_in_flight: 1
      interactive_priority: 0
"""
    )
    args = SimpleNamespace(
        gateway_config=config,
        good_key="good-key",
        noisy_key="noisy-key",
        noisy_quota_per_minute=60,
    )

    contract = _gateway_contract(args)

    assert all(contract["checks"].values())
    assert contract["limits"]["noisy"]["token_burst"] == 1_000_000


def test_gpu_bench_requires_resolved_tp8_in_runtime_topology() -> None:
    assert _contains_tensor_parallel_size(
        {"engines": [{"options": {"tensor_parallel_size": 8}}]},
        8,
    )
    assert not _contains_tensor_parallel_size(
        {"engines": [{"options": {"tensor_parallel_size": 4}}]},
        8,
    )


def test_gpu_bench_zero_metric_requires_one_expected_series() -> None:
    assert _single_labeled_value({}, tenant="good") is None
    assert (
        _single_labeled_value(
            {'metric{tenant="good"}': 0.0},
            tenant="good",
        )
        == 0.0
    )
    assert (
        _single_labeled_value(
            {
                'metric{source="http",tenant="good"}': 0.0,
                'metric{source="batch",tenant="good"}': 0.0,
            },
            tenant="good",
        )
        is None
    )
