import math
from types import SimpleNamespace

import verification.fleet.resilience.noisy_neighbor_gpu_bench as gpu_bench
from verification.fleet.resilience.noisy_neighbor_gpu_bench import (
    _contains_tensor_parallel_size,
    _gateway_contract,
    _labeled_count,
    _single_labeled_value,
    metric_delta,
    percentile,
)


async def test_two_tenant_window_uses_one_origin_and_distinct_phase_labels(
    monkeypatch,
) -> None:
    calls = []
    # Adding an offset crosses a binary exponent boundary here, so subtracting
    # the absolute timestamps would introduce one ULP of cancellation error.
    boundary_origin = math.nextafter(512.0, 0.0)
    clock_calls = 0

    def fake_perf_counter() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return boundary_origin

    async def fake_one_stream(
        client,
        headers,
        model,
        phase,
        index,
        target_s,
        origin_s,
    ):
        calls.append(
            (headers["tenant"], phase, index, target_s, origin_s)
        )
        return {"phase": phase, "index": index}

    monkeypatch.setattr(gpu_bench, "_one_stream", fake_one_stream)
    monkeypatch.setattr(gpu_bench.time, "perf_counter", fake_perf_counter)

    good, noisy, duration = await gpu_bench._two_tenant_window(
        object(),
        {"tenant": "good"},
        {"tenant": "noisy"},
        "model",
        phase="control",
        good_count=4,
        good_rate=2.0,
        noisy_rate=1.0,
    )

    assert duration == 2.0
    assert clock_calls == 1
    assert [item["phase"] for item in good] == ["control-good"] * 4
    assert [item["phase"] for item in noisy] == ["control-noisy"] * 2
    assert [
        (tenant, call_phase, index)
        for tenant, call_phase, index, _, _ in calls
    ] == [
        ("good", "control-good", 0),
        ("good", "control-good", 1),
        ("good", "control-good", 2),
        ("good", "control-good", 3),
        ("noisy", "control-noisy", 0),
        ("noisy", "control-noisy", 1),
    ]
    origin = calls[0][4]
    assert origin == boundary_origin
    assert [call_origin for *_, call_origin in calls] == [origin] * 6
    assert [target for *_, target, _ in calls] == [
        origin + 0.0,
        origin + 0.5,
        origin + 1.0,
        origin + 1.5,
        origin + 0.0,
        origin + 1.0,
    ]


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
