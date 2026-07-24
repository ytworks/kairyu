from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from kairyu.evaluation.protocol import protocol_hash
from kairyu.evaluation.safety import SecretSafetyError
from kairyu.evaluation.schemas import (
    Artifact,
    BenchmarkDefinition,
    BenchmarkProfile,
    BenchmarkRun,
    Comparability,
    ImplementationStatus,
    Metric,
    ProtocolSignature,
    ReferenceResult,
    RunItem,
    RunMode,
    RunState,
)


def protocol(**changes):
    values = {
        "benchmark_id": "gpqa-diamond",
        "benchmark_version": "fugu-2026",
        "dataset_revision": "sha256:fixture",
        "split": "smoke",
        "harness_name": "synthetic",
        "harness_version": "1",
        "prompt_version": "v1",
        "metric_implementation": "accuracy-v1",
    }
    values.update(changes)
    return ProtocolSignature(**values)


def test_definition_is_strict_frozen_and_explicitly_planned():
    definition = BenchmarkDefinition(
        benchmark_id="gpqa-diamond",
        display_name="GPQA Diamond",
        benchmark_version="profile-pinned",
        primary_metric="Accuracy",
    )

    assert definition.implementation_status is ImplementationStatus.PLANNED
    with pytest.raises(ValidationError):
        definition.display_name = "changed"
    with pytest.raises(SecretSafetyError):
        BenchmarkDefinition(
            benchmark_id="gpqa-diamond",
            display_name="GPQA Diamond",
            benchmark_version="profile-pinned",
            primary_metric="Accuracy",
            api_key="must-not-be-a-schema-field",
        )


def test_profile_requires_matching_protocol_benchmark():
    with pytest.raises(ValidationError, match="must match"):
        BenchmarkProfile(
            name="smoke",
            benchmark_id="mrcr-v2",
            protocol=protocol(),
        )


def test_protocol_rejects_unknown_or_duplicate_unresolved_fields():
    with pytest.raises(ValidationError, match="unknown protocol"):
        protocol(unresolved_fields=("api_key",))
    with pytest.raises(ValidationError, match="must be unique"):
        protocol(unresolved_fields=("judge_model", "judge_model"))


def test_protocol_json_is_deeply_immutable_and_hash_stable():
    signature = protocol(generation_parameters={"nested": {"ids": ["one", "two"]}})
    original_hash = protocol_hash(signature)
    nested = signature.generation_parameters["nested"]
    assert isinstance(nested, Mapping)
    ids = nested["ids"]
    assert isinstance(ids, Sequence)

    with pytest.raises(TypeError, match="immutable"):
        signature.generation_parameters["temperature"] = 1
    with pytest.raises(TypeError, match="immutable"):
        nested["other"] = True
    with pytest.raises(TypeError, match="immutable"):
        ids.append("three")
    with pytest.raises(TypeError):
        dict.__setitem__(signature.generation_parameters, "bypass", True)
    with pytest.raises(TypeError):
        list.append(ids, "bypass")
    with pytest.raises(TypeError, match="immutable"):
        nested._values = {}  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="immutable"):
        ids._values = ()  # type: ignore[attr-defined]

    assert protocol_hash(signature) == original_hash
    assert signature.model_dump(mode="json")["generation_parameters"] == {
        "nested": {"ids": ["one", "two"]}
    }
    assert '"ids":["one","two"]' in signature.model_dump_json()

    mutable_update = {"nested": {"ids": ["copy"]}}
    copied = signature.model_copy(update={"generation_parameters": mutable_update})
    copied_hash = protocol_hash(copied)
    mutable_update["nested"]["ids"].append("outside")
    copied_ids = copied.generation_parameters["nested"]["ids"]
    assert copied_ids == ["copy"]
    with pytest.raises(TypeError, match="immutable"):
        copied_ids.append("inside")
    assert protocol_hash(copied) == copied_hash

    with pytest.raises(SecretSafetyError):
        signature.model_copy(update={"judge_model": "sk-" + "x" * 24})


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_protocol_rejects_non_finite_nested_numbers(value):
    with pytest.raises(ValidationError, match="finite"):
        protocol(generation_parameters={"temperature": value})


def test_run_lifecycle_fields_are_consistent():
    now = datetime.now(UTC)
    common = {
        "run_id": "run-01",
        "benchmark_id": "gpqa-diamond",
        "profile": "smoke",
        "mode": "smoke",
        "target_model": "fake",
        "protocol_hash": "a" * 64,
        "item_input_manifest_sha256": "b" * 64,
        "created_at": now,
    }

    with pytest.raises(ValidationError, match="started_at"):
        BenchmarkRun(**common, state=RunState.RUNNING)
    with pytest.raises(ValidationError, match="finished_at"):
        BenchmarkRun(**common, state=RunState.COMPLETED, started_at=now)
    with pytest.raises(ValidationError, match="partial"):
        BenchmarkRun(
            **common,
            state=RunState.PARTIAL,
            started_at=now,
            finished_at=now,
            partial=False,
        )


def test_prepared_states_require_protocol_and_item_manifest_identity():
    now = datetime.now(UTC)
    common = {
        "run_id": "run-01",
        "benchmark_id": "gpqa-diamond",
        "profile": "smoke",
        "mode": "smoke",
        "target_model": "fake",
        "created_at": now,
    }

    for state in (RunState.READY, RunState.RUNNING, RunState.FAILED):
        with pytest.raises(ValidationError, match="manifest hashes"):
            BenchmarkRun(**common, state=state)
        with pytest.raises(ValidationError, match="manifest hashes"):
            BenchmarkRun(**common, state=state, protocol_hash="a" * 64)
        with pytest.raises(ValidationError, match="manifest hashes"):
            BenchmarkRun(
                **common,
                state=state,
                item_input_manifest_sha256="b" * 64,
            )

    cancelled = BenchmarkRun(
        **common,
        state=RunState.CANCELLED,
        started_at=now,
        finished_at=now,
    )
    assert cancelled.protocol_hash is None
    assert cancelled.item_input_manifest_sha256 is None
    for state in (RunState.BLOCKED, RunState.NEEDS_USER_ACTION):
        paused_during_prepare = BenchmarkRun(**common, state=state)
        assert paused_during_prepare.protocol_hash is None
        assert paused_during_prepare.item_input_manifest_sha256 is None


def test_selected_item_ids_are_nonblank_bounded_and_unique():
    common = {
        "run_id": "run-01",
        "benchmark_id": "gpqa-diamond",
        "profile": "smoke",
        "mode": "smoke",
        "target_model": "fake",
    }

    with pytest.raises(ValidationError, match="at least 1"):
        BenchmarkRun(**common, selected_item_ids=("   ",))
    with pytest.raises(ValidationError, match="at most 512"):
        BenchmarkRun(**common, selected_item_ids=("x" * 513,))
    with pytest.raises(ValidationError, match="must be unique"):
        BenchmarkRun(**common, selected_item_ids=("item-1", "item-1"))


def test_scores_and_metric_dimensions_are_deeply_immutable():
    item = RunItem(
        run_id="run-01",
        item_id="item-01",
        ordinal=0,
        input_sha256="a" * 64,
        scores={"accuracy": 1.0},
    )
    metric = Metric(
        run_id="run-01",
        name="accuracy",
        display_name="Accuracy",
        dimensions={"groups": [{"name": "all"}]},
    )

    with pytest.raises(TypeError, match="immutable"):
        item.scores["accuracy"] = 0.0
    groups = metric.dimensions["groups"]
    assert isinstance(groups, Sequence)
    with pytest.raises(TypeError, match="immutable"):
        groups.append({"name": "other"})
    group = groups[0]
    assert isinstance(group, Mapping)
    with pytest.raises(TypeError, match="immutable"):
        group["name"] = "changed"
    with pytest.raises(TypeError):
        dict.__setitem__(item.scores, "bypass", 0.0)
    with pytest.raises(TypeError):
        list.append(groups, {"name": "bypass"})
    assert item.model_dump(mode="json")["scores"] == {"accuracy": 1.0}
    assert metric.model_dump(mode="json")["dimensions"] == {"groups": [{"name": "all"}]}


@pytest.mark.parametrize(
    "state",
    (
        RunState.CANCELLING,
        RunState.CANCELLED,
        RunState.FAILED,
        RunState.BLOCKED,
        RunState.NEEDS_USER_ACTION,
    ),
)
def test_interrupted_states_may_retain_partial_evidence(state):
    now = datetime.now(UTC)
    timestamps = (
        {"started_at": now}
        if state is RunState.CANCELLING
        else {"started_at": now, "finished_at": now}
        if state in {RunState.CANCELLED, RunState.FAILED}
        else {}
    )
    run = BenchmarkRun(
        run_id="run-01",
        benchmark_id="gpqa-diamond",
        profile="smoke",
        mode="smoke",
        state=state,
        partial=True,
        protocol_hash="a" * 64,
        item_input_manifest_sha256="b" * 64,
        selected_item_ids=("item-1",),
        completed_count=1,
        target_model="fake",
        created_at=now,
        **timestamps,
    )

    assert run.partial is True


@pytest.mark.parametrize(
    "state",
    (
        RunState.PENDING,
        RunState.PREPARING,
        RunState.READY,
        RunState.RUNNING,
        RunState.COMPLETED,
    ),
)
def test_non_interrupted_states_reject_partial_evidence(state):
    now = datetime.now(UTC)
    values = {
        "run_id": "run-01",
        "benchmark_id": "gpqa-diamond",
        "profile": "smoke",
        "mode": "smoke",
        "state": state,
        "partial": True,
        "protocol_hash": "a" * 64,
        "item_input_manifest_sha256": "b" * 64,
        "target_model": "fake",
        "created_at": now,
    }
    if state in {RunState.RUNNING, RunState.COMPLETED}:
        values["started_at"] = now
    if state is RunState.COMPLETED:
        values["finished_at"] = now

    with pytest.raises(ValidationError, match="partial evidence is invalid"):
        BenchmarkRun(**values)


def test_resume_lineage_is_explicit_and_immutable():
    now = datetime.now(UTC)
    common = {
        "run_id": "run-02",
        "benchmark_id": "gpqa-diamond",
        "profile": "smoke",
        "mode": "smoke",
        "target_model": "fake",
        "created_at": now,
    }

    with pytest.raises(ValidationError, match="source run"):
        BenchmarkRun(**common, attempt=2)
    with pytest.raises(ValidationError, match="first attempt"):
        BenchmarkRun(**common, resumed_from_run_id="run-01")
    with pytest.raises(ValidationError, match="itself"):
        BenchmarkRun(
            **common,
            attempt=2,
            resumed_from_run_id="run-02",
        )

    resumed = BenchmarkRun(
        **common,
        attempt=2,
        resumed_from_run_id="run-01",
    )
    assert resumed.resumed_from_run_id == "run-01"


def test_reference_comparability_requires_protocol_evidence():
    with pytest.raises(ValidationError, match="protocol hash"):
        ReferenceResult(
            reference_id="reference-01",
            benchmark_id="gpqa-diamond",
            benchmark_version="fugu-2026",
            profile="full",
            model_name="provider-model",
            score=42.0,
            metric_name="accuracy",
            source_id="source-01",
            source_type="provider_system_card",
            retrieved_at=datetime.now(UTC),
            evidence_hash="a" * 64,
            comparability=Comparability.EXACT,
        )


def test_run_counts_and_timezone_are_validated():
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="exceed selected"):
        BenchmarkRun(
            run_id="run-01",
            benchmark_id="gpqa-diamond",
            profile="smoke",
            mode=RunMode.SMOKE,
            selected_item_ids=("one",),
            completed_count=2,
            target_model="fake",
            created_at=now,
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        BenchmarkRun(
            run_id="run-01",
            benchmark_id="gpqa-diamond",
            profile="smoke",
            mode="smoke",
            target_model="fake",
            created_at=datetime.now(),
        )
    with pytest.raises(ValidationError, match="precedes"):
        BenchmarkRun(
            run_id="run-01",
            benchmark_id="gpqa-diamond",
            profile="smoke",
            mode="smoke",
            target_model="fake",
            created_at=now,
            started_at=now - timedelta(seconds=1),
        )


@pytest.mark.parametrize(
    "relative_path",
    (
        "../secret",
        "/tmp/secret",
        r"logs\\secret",
        "logs/../secret",
        "./metrics.json",
        "logs//x.json",
        "logs/./x.json",
        "logs/file name.json",
        "logs/日本語.json",
        ".hidden",
    ),
)
def test_artifact_rejects_unsafe_paths(relative_path):
    with pytest.raises(ValidationError, match="canonical portable"):
        Artifact(
            run_id="run-01",
            name="manifest.json",
            relative_path=relative_path,
            media_type="application/json",
            sha256="a" * 64,
            size_bytes=1,
        )


def test_artifact_accepts_canonical_nested_safe_components():
    artifact = Artifact(
        run_id="run-01",
        name="log",
        relative_path="logs/x.json",
        media_type="application/json",
        sha256="a" * 64,
        size_bytes=1,
    )

    assert artifact.relative_path == "logs/x.json"


def test_artifact_requires_canonical_sha256_and_aware_time():
    with pytest.raises(ValidationError):
        Artifact(
            run_id="run-01",
            name="manifest.json",
            relative_path="manifest.json",
            media_type="application/json",
            sha256="A" * 64,
            size_bytes=1,
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        Artifact(
            run_id="run-01",
            name="manifest.json",
            relative_path="manifest.json",
            media_type="application/json",
            sha256="a" * 64,
            size_bytes=1,
            created_at=datetime.now(),
        )
