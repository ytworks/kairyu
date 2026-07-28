from bench.priority_overload_gpu_bench import (
    _scheduler_event_count,
    metric_delta,
    percentile,
)


def test_gpu_bench_percentile_uses_raw_nearest_rank_samples() -> None:
    assert percentile([float(index) for index in range(1, 101)], 0.99) == 99.0


def test_gpu_bench_metric_delta_preserves_bounded_labels() -> None:
    before = """
kairyu_priority_requests_total{request_class="interactive",source="http"} 2
"""
    after = """
kairyu_priority_requests_total{request_class="interactive",source="http"} 7
kairyu_priority_requests_total{request_class="batch",source="batch"} 3
"""

    assert metric_delta(
        before,
        after,
        "kairyu_priority_requests_total",
    ) == {
        'kairyu_priority_requests_total{request_class="interactive",source="http"}': 5.0,
        'kairyu_priority_requests_total{request_class="batch",source="batch"}': 3.0,
    }


def test_scheduler_event_count_uses_bounded_class_and_event_labels() -> None:
    delta = {
        (
            'kairyu_scheduler_priority_events_total{event="admit",model="m",'
            'request_class="batch"}'
        ): 12.0,
        (
            'kairyu_scheduler_priority_events_total{event="complete",model="m",'
            'request_class="batch"}'
        ): 9.0,
    }

    assert _scheduler_event_count(delta, "batch", "admit") == 12.0
    assert _scheduler_event_count(delta, "batch", "complete") == 9.0
