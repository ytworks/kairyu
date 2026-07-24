import pytest
from pydantic import ValidationError

from kairyu.evaluation.protocol import ProtocolDifference, canonical_protocol_json
from kairyu.evaluation.safety import (
    SecretSafetyError,
    SecretValueRegistry,
    ensure_secret_free_bytes,
    ensure_secret_free_json,
)
from kairyu.evaluation.schemas import BenchmarkRun, Metric, ProtocolSignature, RunItem


def protocol(**changes):
    values = {
        "benchmark_id": "gpqa-diamond",
        "benchmark_version": "fugu-2026",
        "dataset_revision": "fixture",
        "split": "smoke",
        "harness_name": "synthetic",
        "harness_version": "1",
        "prompt_version": "v1",
        "metric_implementation": "accuracy-v1",
    }
    values.update(changes)
    return ProtocolSignature(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sample_filter", {"nested": [{"Authorization": "not-shown"}]}),
        ("generation_parameters", {"outer": {"OPENAI_API_KEY": "not-shown"}}),
        ("code_execution_sandbox", {"env": {"HF_TOKEN": "not-shown"}}),
        (
            "sample_filter",
            {"asset": "https://example.test/x?X-Amz-Signature=not-shown"},
        ),
    ),
)
def test_protocol_rejects_nested_secrets_without_pydantic_error_leak(field, value):
    with pytest.raises(SecretSafetyError) as exc_info:
        protocol(**{field: value})

    assert "not-shown" not in str(exc_info.value)
    assert "not-shown" not in repr(exc_info.value)
    assert not isinstance(exc_info.value, ValidationError)


@pytest.mark.parametrize(
    "value",
    (
        {"nested": ["Bearer not-shown"]},
        {"nested": [{"value": "sk-not-shown-value"}]},
        {"nested": ["Basic dXNlcjpwYXNzd29yZA=="]},
    ),
)
def test_protocol_rejects_credential_shaped_nested_values(value):
    with pytest.raises(SecretSafetyError) as exc_info:
        protocol(generation_parameters=value)

    assert "not-shown" not in str(exc_info.value)


@pytest.mark.parametrize(
    "value",
    (
        {"token": "hello"},
        {"max_tokens": 1024, "tokenizer": "fixture-tokenizer"},
        {"dependency": "sk-learn"},
        {"sentence": "Bearer of bad news arrived."},
    ),
)
def test_benign_token_and_prose_values_are_not_false_positives(value):
    signature = protocol(generation_parameters=value)

    assert signature.generation_parameters == value


@pytest.mark.parametrize(
    "alias",
    (
        "OPENAI_API_KEY",
        "x-api-key",
        "HF_TOKEN",
        "anthropic_api_key",
        "secret",
        "credential",
        "client_password",
        "database_password",
    ),
)
def test_common_credential_key_aliases_are_rejected(alias):
    with pytest.raises(SecretSafetyError):
        ensure_secret_free_json({alias: "not-shown"})


def test_registered_secret_substrings_are_rejected_without_plaintext_retention():
    secret = "ordinary-provider-value-not-shaped-like-a-key"
    registry = SecretValueRegistry([secret])

    assert secret not in repr(registry)
    assert registry.contains(f"prefix:{secret}:suffix")
    with pytest.raises(SecretSafetyError) as exc_info:
        ensure_secret_free_json(
            {"provider_value": f"https://example.test/{secret}#fragment"},
            secret_registry=registry,
        )
    assert secret not in str(exc_info.value)

    with pytest.raises(SecretSafetyError):
        ensure_secret_free_bytes(
            f"log prefix {secret} suffix".encode(),
            secret_registry=registry,
        )


def test_whole_model_scan_covers_free_text_and_validation_context():
    shaped = "sk-" + "x" * 24
    with pytest.raises(SecretSafetyError):
        protocol(judge_model=shaped)

    secret = "ordinary-provider-secret-value"
    registry = SecretValueRegistry([secret])
    payload = {
        "run_id": "run-01",
        "benchmark_id": "gpqa-diamond",
        "profile": "smoke",
        "mode": "smoke",
        "target_model": "fake-model",
        "termination_reason": f"provider returned {secret}",
    }
    with pytest.raises(SecretSafetyError) as exc_info:
        BenchmarkRun.model_validate(payload, context={"secret_registry": registry})
    assert secret not in str(exc_info.value)
    assert not isinstance(exc_info.value, ValidationError)


@pytest.mark.parametrize("encoding", ("utf-16-le", "utf-16-be"))
def test_registered_secret_is_rejected_in_common_text_encodings(encoding):
    secret = "ordinary-provider-secret-value"
    registry = SecretValueRegistry([secret])

    with pytest.raises(SecretSafetyError):
        ensure_secret_free_bytes(
            f"log prefix {secret} suffix".encode(encoding),
            secret_registry=registry,
        )


@pytest.mark.parametrize(
    "value",
    (
        "postgresql://alice:hunter2@db.example/app",
        "https://alice:hunter2@[::1",
    ),
)
def test_all_uri_userinfo_and_malformed_authorities_fail_closed(value):
    with pytest.raises(SecretSafetyError):
        ensure_secret_free_json({"endpoint": value})


@pytest.mark.parametrize(
    "value",
    (
        "Bearer abcdefgh1234, request failed",
        "Basic dXNlcjpwYXNzd29yZA==, request failed",
    ),
)
def test_credential_shapes_are_rejected_before_punctuation(value):
    with pytest.raises(SecretSafetyError):
        ensure_secret_free_json({"message": value})


def test_registry_count_tracks_secrets_not_encoded_representations():
    registry = SecretValueRegistry(["first", "second", "first"])

    assert len(registry) == 2


def test_registered_secret_in_mapping_key_never_reaches_validation_errors():
    secret = "mapping-key-provider-secret"
    registry = SecretValueRegistry([secret])
    payload = {
        "run_id": "run-01",
        "benchmark_id": "gpqa-diamond",
        "profile": "smoke",
        "mode": "smoke",
        "target_model": "fake-model",
        secret: "ordinary value",
    }

    with pytest.raises(SecretSafetyError) as exc_info:
        BenchmarkRun.model_validate(payload, context={"secret_registry": registry})

    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
    assert not isinstance(exc_info.value, ValidationError)


def test_canonical_snapshot_rescan_blocks_unvalidated_model_construction():
    shaped = "sk-" + "z" * 24
    base = protocol()
    tainted = ProtocolSignature.model_construct(**{**base.model_dump(), "judge_model": shaped})

    with pytest.raises(SecretSafetyError):
        canonical_protocol_json(tainted)

    secret = "generic-provider-secret-value"
    registry = SecretValueRegistry([secret])
    tainted_generic = ProtocolSignature.model_construct(
        **{**base.model_dump(), "judge_model": secret}
    )
    with pytest.raises(SecretSafetyError):
        canonical_protocol_json(tainted_generic, secret_registry=registry)


def test_other_arbitrary_schema_json_fields_are_secret_safe():
    with pytest.raises(SecretSafetyError):
        Metric(
            run_id="run-01",
            name="accuracy",
            display_name="Accuracy",
            dimensions={"nested": {"Cookie": "not-shown"}},
        )

    with pytest.raises(SecretSafetyError):
        ProtocolDifference(
            field="sample_filter",
            left={"api_key": "not-shown"},
            right={},
            critical=True,
        )


@pytest.mark.parametrize(
    ("model_type", "payload", "update"),
    (
        (
            RunItem,
            {
                "run_id": "run-01",
                "item_id": "item-01",
                "ordinal": 0,
                "input_sha256": "a" * 64,
                "scores": {"safe": 1.0},
            },
            {"scores": {"ordinary-provider-secret-value": 1.0}},
        ),
        (
            Metric,
            {
                "run_id": "run-01",
                "name": "accuracy",
                "display_name": "Accuracy",
                "dimensions": {"group": "safe"},
            },
            {"dimensions": {"group": "ordinary-provider-secret-value"}},
        ),
        (
            ProtocolDifference,
            {
                "field": "sample_filter",
                "left": {"safe": True},
                "right": {},
                "critical": True,
            },
            {"left": {"value": "ordinary-provider-secret-value"}},
        ),
    ),
)
def test_model_copy_preserves_registered_secret_context(model_type, payload, update):
    secret = "ordinary-provider-secret-value"
    registry = SecretValueRegistry([secret])
    model = model_type.model_validate(
        payload,
        context={"secret_registry": registry},
    )

    assert "_secret_registry" not in model.model_dump()
    assert secret not in model.model_dump_json()
    with pytest.raises(SecretSafetyError) as exc_info:
        model.model_copy(update=update)
    assert secret not in str(exc_info.value)
