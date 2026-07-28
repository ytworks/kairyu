from bench.priority_overload_gpu_bench import (
    _has_scheduler_queue_evidence,
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


def test_gpu_gate_requires_positive_batch_queue_evidence() -> None:
    empty_queue = {
        'kairyu_scheduler_queue_depth{model="m",request_class="batch"}': 0.0,
        'kairyu_scheduler_queue_depth{model="m",request_class="interactive"}': 0.0,
    }
    empty_high_watermark = {
        (
            'kairyu_scheduler_queue_high_watermark{model="m",'
            'request_class="batch"}'
        ): 0.0,
        (
            'kairyu_scheduler_queue_high_watermark{model="m",'
            'request_class="interactive"}'
        ): 0.0,
    }

    assert not _has_scheduler_queue_evidence(empty_queue, empty_high_watermark)

    batch_queue = dict(empty_queue)
    batch_queue[
        'kairyu_scheduler_queue_depth{model="m",request_class="batch"}'
    ] = 1.0
    batch_high_watermark = dict(empty_high_watermark)
    batch_high_watermark[
        'kairyu_scheduler_queue_high_watermark{model="m",request_class="batch"}'
    ] = 8.0
    assert _has_scheduler_queue_evidence(batch_queue, batch_high_watermark)


def test_gpu_gate_rejects_inconsistent_batch_queue_high_watermark() -> None:
    assert not _has_scheduler_queue_evidence(
        {'kairyu_scheduler_queue_depth{request_class="batch"}': 2.0},
        {
            (
                'kairyu_scheduler_queue_high_watermark{'
                'request_class="batch"}'
            ): 1.0
        },
    )
