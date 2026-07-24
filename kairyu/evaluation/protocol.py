"""Canonical protocol identity and conservative comparison rules."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import Field, JsonValue, ValidationInfo, field_serializer, field_validator

from kairyu.evaluation.safety import (
    SecretValueRegistry,
    ensure_secret_free_json,
    ensure_secret_free_serialized_json,
    secret_registry_from_context,
)
from kairyu.evaluation.schemas import (
    Comparability,
    FrozenModel,
    ProtocolSignature,
    freeze_json_value,
    thaw_json_value,
)

# Only these reviewed provenance/implementation-version differences may be NEAR.
# Every task, data, scoring, prompt, model-interaction, unknown, and newly added
# field remains fail-closed as incompatible.
REVIEWED_NON_CRITICAL_PROTOCOL_FIELDS = frozenset({"schema_version", "harness_version"})
_PROTOCOL_FIELDS = frozenset(ProtocolSignature.model_fields) - {"unresolved_fields"}
if not REVIEWED_NON_CRITICAL_PROTOCOL_FIELDS <= _PROTOCOL_FIELDS:
    raise RuntimeError("reviewed protocol fields must exist in ProtocolSignature")
CRITICAL_PROTOCOL_FIELDS = _PROTOCOL_FIELDS - REVIEWED_NON_CRITICAL_PROTOCOL_FIELDS


class ProtocolDifference(FrozenModel):
    field: str = Field(min_length=1)
    left: JsonValue
    right: JsonValue
    critical: bool

    @field_validator("left", "right")
    @classmethod
    def _values_are_secret_free(
        cls,
        value: JsonValue,
        info: ValidationInfo,
    ) -> JsonValue:
        ensure_secret_free_json(
            value,
            secret_registry=secret_registry_from_context(info.context),
        )
        return freeze_json_value(value)

    @field_serializer("left", "right")
    def _serialize_values(self, value: object) -> JsonValue:
        return thaw_json_value(value)


class ProtocolComparison(FrozenModel):
    comparability: Comparability
    left_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    differences: tuple[ProtocolDifference, ...] = ()
    unresolved_fields: tuple[str, ...] = ()


def canonical_protocol_json(
    signature: ProtocolSignature,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> str:
    """Return the one canonical serialization used for protocol identity."""
    payload = signature.model_dump(mode="json", exclude_none=False)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ensure_secret_free_serialized_json(encoded, secret_registry=secret_registry)
    return encoded.decode("utf-8")


def protocol_hash(
    signature: ProtocolSignature,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> str:
    canonical = canonical_protocol_json(signature, secret_registry=secret_registry)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compare_protocols(
    left: ProtocolSignature,
    right: ProtocolSignature,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> ProtocolComparison:
    """Compare protocols without treating missing evidence as equivalence."""
    left_hash = protocol_hash(left, secret_registry=secret_registry)
    right_hash = protocol_hash(right, secret_registry=secret_registry)
    unresolved = tuple(sorted(set(left.unresolved_fields) | set(right.unresolved_fields)))

    left_payload = left.model_dump(mode="json", exclude={"unresolved_fields"})
    right_payload = right.model_dump(mode="json", exclude={"unresolved_fields"})
    ensure_secret_free_json(left_payload, secret_registry=secret_registry)
    ensure_secret_free_json(right_payload, secret_registry=secret_registry)
    differences = tuple(
        ProtocolDifference.model_validate(
            {
                "field": field,
                "left": _json_value(left_payload.get(field)),
                "right": _json_value(right_payload.get(field)),
                "critical": field in CRITICAL_PROTOCOL_FIELDS,
            },
            context={"secret_registry": secret_registry},
        )
        for field in sorted(set(left_payload) | set(right_payload))
        if left_payload.get(field) != right_payload.get(field)
    )

    if unresolved or any(difference.critical for difference in differences):
        comparability = Comparability.INCOMPATIBLE
    elif left_hash == right_hash:
        comparability = Comparability.EXACT
    else:
        comparability = Comparability.NEAR

    return ProtocolComparison.model_validate(
        {
            "comparability": comparability,
            "left_hash": left_hash,
            "right_hash": right_hash,
            "differences": differences,
            "unresolved_fields": unresolved,
        },
        context={"secret_registry": secret_registry},
    )


def _json_value(value: Any) -> JsonValue:
    """Assert that a model-dumped value remains canonical-JSON compatible."""
    # Round-tripping also makes tuple/list representation consistent in diff output.
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
