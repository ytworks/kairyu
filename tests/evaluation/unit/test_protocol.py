import json
from collections.abc import Mapping, Sequence

import pytest

from kairyu.evaluation.protocol import (
    ProtocolDifference,
    canonical_protocol_json,
    compare_protocols,
    protocol_hash,
)
from kairyu.evaluation.safety import SecretSafetyError, SecretValueRegistry
from kairyu.evaluation.schemas import Comparability, ProtocolSignature


def signature(**changes):
    values = {
        "benchmark_id": "gpqa-diamond",
        "benchmark_version": "fugu-2026",
        "dataset_revision": "dataset-sha",
        "split": "train",
        "sample_filter": {"ids": ["b", "a"]},
        "harness_name": "evalscope",
        "harness_version": "1.8.1",
        "prompt_version": "default-v1",
        "generation_parameters": {"temperature": 0.0, "seed": 7},
        "metric_implementation": "accuracy-v1",
    }
    values.update(changes)
    return ProtocolSignature(**values)


def test_protocol_difference_values_are_deeply_immutable():
    difference = ProtocolDifference(
        field="sample_filter",
        left={"ids": ["one"]},
        right={"ids": ["two"]},
        critical=True,
    )
    left = difference.left
    assert isinstance(left, Mapping)
    ids = left["ids"]
    assert isinstance(ids, Sequence)

    with pytest.raises(TypeError, match="immutable"):
        left["other"] = True
    with pytest.raises(TypeError, match="immutable"):
        ids.append("three")
    with pytest.raises(TypeError):
        dict.__setitem__(left, "bypass", True)
    with pytest.raises(TypeError):
        list.append(ids, "bypass")
    assert difference.model_dump(mode="json")["left"] == {"ids": ["one"]}
    assert '"ids":["one"]' in difference.model_dump_json()


def test_canonical_json_and_hash_are_stable():
    first = signature(generation_parameters={"seed": 7, "temperature": 0.0})
    second = signature(generation_parameters={"temperature": 0.0, "seed": 7})

    encoded = canonical_protocol_json(first)
    assert encoded == canonical_protocol_json(second)
    assert protocol_hash(first) == protocol_hash(second)
    assert len(protocol_hash(first)) == 64
    assert json.loads(encoded)["benchmark_id"] == "gpqa-diamond"
    assert " " not in encoded


def test_equal_fully_known_protocols_are_exact():
    comparison = compare_protocols(signature(), signature())

    assert comparison.comparability is Comparability.EXACT
    assert comparison.differences == ()
    assert comparison.unresolved_fields == ()


def test_reviewed_harness_version_difference_is_near():
    comparison = compare_protocols(
        signature(harness_version="1.8.1"),
        signature(harness_version="1.8.2"),
    )

    assert comparison.comparability is Comparability.NEAR
    assert [difference.field for difference in comparison.differences] == ["harness_version"]
    assert comparison.differences[0].critical is False


def test_dependency_compatibility_patch_difference_is_incompatible():
    comparison = compare_protocols(
        signature(),
        signature(dependency_compatibility_patches=("stdin-buffer.patch",)),
    )

    assert comparison.comparability is Comparability.INCOMPATIBLE
    assert [difference.field for difference in comparison.differences] == [
        "dependency_compatibility_patches"
    ]
    assert comparison.differences[0].critical is True


def test_metric_or_dataset_difference_is_incompatible():
    comparison = compare_protocols(
        signature(),
        signature(dataset_revision="different", metric_implementation="other"),
    )

    assert comparison.comparability is Comparability.INCOMPATIBLE
    assert {difference.field for difference in comparison.differences} == {
        "dataset_revision",
        "metric_implementation",
    }
    assert all(difference.critical for difference in comparison.differences)


def test_unresolved_critical_evidence_prevents_exact_comparison():
    unresolved = signature(unresolved_fields=("judge_model",))
    comparison = compare_protocols(unresolved, unresolved)

    assert comparison.left_hash == comparison.right_hash
    assert comparison.comparability is Comparability.INCOMPATIBLE
    assert comparison.unresolved_fields == ("judge_model",)


def test_compare_protocols_scans_unvalidated_construction_with_registry():
    secret = "ordinary-provider-secret-value"
    registry = SecretValueRegistry([secret])
    base = signature()
    tainted = ProtocolSignature.model_construct(**{**base.model_dump(), "judge_model": secret})

    with pytest.raises(SecretSafetyError) as exc_info:
        compare_protocols(signature(), tainted, secret_registry=registry)

    assert secret not in str(exc_info.value)


def test_secret_fields_are_not_accepted_by_protocol_schema():
    payload = signature().model_dump()
    payload["api_key"] = "canary-secret"

    with pytest.raises(SecretSafetyError) as exc_info:
        ProtocolSignature.model_validate(payload)

    assert "canary-secret" not in str(exc_info.value)
    assert "canary-secret" not in repr(exc_info.value)
