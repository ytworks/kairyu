import pytest

from bench.slo_admission_bench import (
    FORMAL_SCENARIOS,
    Scenario,
    build_trace,
    run_gate,
    run_scenario,
)


@pytest.fixture(scope="module")
def formal_result() -> dict:
    return run_gate()


def test_formal_profiles_pin_underload_and_two_distinct_exact_2x_traces() -> None:
    scenarios = {scenario.name: scenario for scenario in FORMAL_SCENARIOS}

    assert set(scenarios) == {"underload", "smooth_2x", "bursty_2x"}
    assert scenarios["underload"].offered_load_ratio == 0.5
    assert scenarios["smooth_2x"].offered_load_ratio == 2.0
    assert scenarios["bursty_2x"].offered_load_ratio == 2.0
    assert scenarios["smooth_2x"].release_period_ticks == 1
    assert scenarios["bursty_2x"].release_period_ticks == 4
    assert all(scenario.ttft_slo_ticks == 2 for scenario in FORMAL_SCENARIOS)


def test_trace_is_deterministic_and_preserves_release_order() -> None:
    scenario = Scenario(
        name="trace",
        measurement_ticks=8,
        release_period_ticks=4,
        requests_per_release=3,
    )

    first = build_trace(scenario)
    second = build_trace(scenario)

    assert first == second
    assert [item["arrival_tick"] for item in first] == [0, 0, 0, 4, 4, 4]
    assert [item["release_order"] for item in first] == [0, 1, 2, 0, 1, 2]
    assert len({item["request_id"] for item in first}) == len(first)


def test_formal_gate_is_nonvacuous_and_goodput_is_noninferior(
    formal_result: dict,
) -> None:
    assert formal_result["passed"], {
        name: passed
        for name, passed in formal_result["checks"].items()
        if not passed
    }
    assert formal_result["checks"][
        "cross_scenario:saturation_strict_gain_nonvacuous"
    ]

    results = {
        result["scenario"]["name"]: result
        for result in formal_result["scenarios"]
    }
    underload = results["underload"]
    assert underload["slo_admission"]["action_counts"] == {
        "admit": underload["scenario"]["measurement_ticks"] // 2,
        "defer": 0,
        "shed": 0,
    }
    assert (
        underload["slo_admission"]["goodput_at_slo"]
        == underload["queue_and_hope"]["goodput_at_slo"]
    )

    for name in ("smooth_2x", "bursty_2x"):
        result = results[name]
        baseline = result["queue_and_hope"]["goodput_at_slo"]
        treatment = result["slo_admission"]["goodput_at_slo"]
        actions = result["slo_admission"]["action_counts"]
        assert baseline["successful_requests"] > 0
        assert treatment["successful_requests"] > 0
        assert treatment["requests_per_tick"] >= baseline["requests_per_tick"]
        assert actions["defer"] + actions["shed"] > 0


def test_arms_share_input_trace_and_goodput_uses_the_fixed_window(
    formal_result: dict,
) -> None:
    for result in formal_result["scenarios"]:
        scenario = result["scenario"]
        baseline = result["queue_and_hope"]
        treatment = result["slo_admission"]

        assert (
            result["trace_sha256"]
            == baseline["trace_sha256"]
            == treatment["trace_sha256"]
        )
        baseline_input = [
            (
                record["request_id"],
                record["arrival_tick"],
                record["release_order"],
                record["service_tokens"],
            )
            for record in baseline["requests"]
        ]
        treatment_input = [
            (
                record["request_id"],
                record["arrival_tick"],
                record["release_order"],
                record["service_tokens"],
            )
            for record in treatment["requests"]
        ]
        assert baseline_input == treatment_input
        for arm in (baseline, treatment):
            goodput = arm["goodput_at_slo"]
            assert goodput["fixed_window_ticks"] == scenario["measurement_ticks"]
            assert goodput["requests_per_tick"] == (
                goodput["successful_requests"] / scenario["measurement_ticks"]
            )
            assert arm["drain_tick"] >= scenario["measurement_ticks"]


def test_treatment_applies_each_decision_before_scheduler_dispatch(
    formal_result: dict,
) -> None:
    observed_actions = set()
    for result in formal_result["scenarios"]:
        treatment = result["slo_admission"]
        records = treatment["requests"]
        observed_actions.update(record["action"] for record in records)

        for record in records:
            if record["action"] == "admit":
                assert record["scheduler_class"] == "interactive"
                assert record["enqueue_tick"] == record["arrival_tick"]
                assert record["completed_tick"] is not None
            elif record["action"] == "defer":
                assert record["scheduler_class"] == "batch"
                assert record["enqueue_tick"] == record["arrival_tick"]
                assert record["completed_tick"] is not None
                assert not record["interactive_goodput_success"]
            else:
                assert record["action"] == "shed"
                assert record["scheduler_class"] is None
                assert record["enqueue_tick"] is None
                assert record["first_token_tick"] is None
                assert record["completed_tick"] is None

        events = treatment["scheduler"]["events"]
        actions = treatment["action_counts"]
        assert events["interactive:enqueue"] == actions["admit"]
        assert events["interactive:admit"] == actions["admit"]
        assert events["interactive:complete"] == actions["admit"]
        assert events["batch:enqueue"] == actions["defer"]
        assert events["batch:admit"] == actions["defer"]
        assert events["batch:complete"] == actions["defer"]
        assert treatment["controller_in_flight_after_drain"] == 0

    assert observed_actions == {"admit", "defer", "shed"}


def test_raw_queue_prediction_decision_and_outcome_evidence_is_complete(
    formal_result: dict,
) -> None:
    for result in formal_result["scenarios"]:
        treatment = result["slo_admission"]
        assert treatment["ticks"]
        assert treatment["schedule_events"]
        for event in treatment["ticks"]:
            assert set(event["queue_before_arrivals"]) == {
                "waiting",
                "running",
                "active",
            }
            assert set(event["queue_after"]) == {
                "waiting",
                "running",
                "active",
            }
            assert "arrivals" in event
            assert "scheduled" in event
            assert "completed" in event

        for record in treatment["requests"]:
            snapshot = record["predictor_snapshot_before"]
            assert snapshot is not None
            assert snapshot["in_flight"] == record["controller_in_flight_before"]
            assert (
                record["predicted_ttft_ticks"]
                == snapshot["predicted_ttft_s"]
            )
            assert record["queue_before_decision"] is not None
            assert record["action"] in {"admit", "defer", "shed"}
            assert "actual_ttft_meets_slo" in record


def test_forced_admit_scheduler_oracle_is_independent_and_reconciled(
    formal_result: dict,
) -> None:
    for result in formal_result["scenarios"]:
        treatment = result["slo_admission"]
        confusion = result["confusion_matrix"]
        records = treatment["requests"]

        assert sum(confusion["counts"].values()) == len(records)
        assert set(confusion["request_ids"]) == {"tp", "fp", "fn", "tn"}
        for record in records:
            assert record["forced_admit_oracle_ttft_ticks"] >= 1
            assert record["prediction_outcome"] in {"tp", "fp", "fn", "tn"}
            if record["action"] == "admit":
                assert (
                    record["forced_admit_oracle_ttft_ticks"]
                    == record["ttft_ticks"]
                )

        counts = confusion["counts"]
        assert confusion["false_positive_rate"] == (
            counts["fp"] / (counts["fp"] + counts["tn"])
            if counts["fp"] + counts["tn"]
            else None
        )
        assert confusion["false_negative_rate"] == (
            counts["fn"] / (counts["fn"] + counts["tp"])
            if counts["fn"] + counts["tp"]
            else None
        )


def test_scenario_rejects_non_integral_measurement_window() -> None:
    scenario = Scenario(
        name="invalid",
        measurement_ticks=5,
        release_period_ticks=2,
        requests_per_release=1,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        run_scenario(scenario)
