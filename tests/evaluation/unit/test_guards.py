import pytest

from kairyu.evaluation.guards import RunGuardError, validate_run_guard
from kairyu.evaluation.schemas import RunMode


def test_smoke_is_offline_safe_and_never_official():
    decision = validate_run_guard("smoke", environ={})

    assert decision.mode is RunMode.SMOKE
    assert decision.official_eligible is False


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({}, "sample_scope_required"),
        ({"limit": 0}, "invalid_limit"),
        ({"limit": True}, "invalid_limit"),
        ({"sample_ids": [""]}, "invalid_sample_ids"),
        ({"sample_ids": ["one", "one"]}, "duplicate_sample_ids"),
    ],
)
def test_sample_requires_an_explicit_valid_subset(kwargs, code):
    with pytest.raises(RunGuardError) as raised:
        validate_run_guard("sample", environ={}, **kwargs)
    assert raised.value.code == code


def test_sample_accepts_limit_or_ids_but_is_unofficial():
    by_limit = validate_run_guard("sample", limit=2, environ={})
    by_id = validate_run_guard("sample", sample_ids=("one", "two"), environ={})

    assert by_limit.limit == 2
    assert by_id.sample_ids == ("one", "two")
    assert not by_limit.official_eligible
    assert not by_id.official_eligible


@pytest.mark.parametrize(
    ("confirm", "environment", "code"),
    [
        (False, {}, "full_confirmation_required"),
        (True, {}, "full_environment_opt_in_required"),
        (True, {"BENCHMARK_ALLOW_FULL_RUN": "true"}, "full_environment_opt_in_required"),
    ],
)
def test_full_requires_both_explicit_opt_ins(confirm, environment, code):
    with pytest.raises(RunGuardError) as raised:
        validate_run_guard(
            "full",
            confirm_full_run=confirm,
            environ=environment,
        )
    assert raised.value.code == code


@pytest.mark.parametrize("ci_value", ["1", "true", "TRUE", "yes", "on"])
def test_ci_always_rejects_full_even_with_both_opt_ins(ci_value):
    with pytest.raises(RunGuardError) as raised:
        validate_run_guard(
            RunMode.FULL,
            confirm_full_run=True,
            environ={"CI": ci_value, "BENCHMARK_ALLOW_FULL_RUN": "1"},
        )
    assert raised.value.code == "full_forbidden_in_ci"


def test_full_rejects_subset_selectors():
    with pytest.raises(RunGuardError) as raised:
        validate_run_guard(
            "full",
            confirm_full_run=True,
            limit=1,
            environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
        )
    assert raised.value.code == "full_with_subset"


def test_full_is_official_only_after_all_non_ci_guards_pass():
    decision = validate_run_guard(
        "full",
        confirm_full_run=True,
        environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
    )

    assert decision.mode is RunMode.FULL
    assert decision.official_eligible is True


def test_invalid_mode_is_controlled():
    with pytest.raises(RunGuardError) as raised:
        validate_run_guard("everything", environ={})
    assert raised.value.code == "invalid_mode"
