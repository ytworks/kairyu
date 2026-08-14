from verification.fleet.resilience.priority_overload_bench import (
    Workload,
    percentile,
    run_gate,
    run_trace,
)


def test_percentile_uses_nearest_rank() -> None:
    assert percentile(list(range(1, 101)), 0.99) == 99


def test_workload_is_exactly_two_x_and_interactive_is_feasible() -> None:
    workload = Workload()

    assert workload.offered_load_ratio == 2.0
    assert (
        workload.interactive_tokens / workload.release_period_ticks
        < 1.0
    )


def test_priority_gate_holds_interactive_slo_and_batch_absorbs_residual() -> None:
    result = run_gate()

    assert result["passed"], {
        name: passed for name, passed in result["checks"].items() if not passed
    }
    assert result["priority"]["interactive"]["ttft_p99_ticks"] <= 2
    assert (
        result["priority"]["interactive"]["ttft_p99_ticks"]
        < result["fcfs"]["interactive"]["ttft_p99_ticks"]
    )
    assert result["priority"]["batch"]["service_tokens_in_measurement"] > 0
    assert result["priority"]["batch"]["backlog_at_measurement_end"] > 0


def test_trace_reconciles_every_offered_request_after_drain() -> None:
    result = run_trace(priority_enabled=True, workload=Workload(measurement_ticks=40))

    assert result["offered_requests"] == 20
    assert result["accounting"]["offered_equals_completed_plus_queued"]
    assert result["interactive"]["completed"] == result["interactive"]["offered"]
    assert result["batch"]["completed_after_drain"] == result["batch"]["offered"]
