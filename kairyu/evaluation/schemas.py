"""Strict, secret-free schemas for the M20 evaluation control plane."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Annotated, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from kairyu.evaluation.safety import (
    SecretValueRegistry,
    ensure_secret_free_json,
    secret_registry_from_context,
)

Identifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")]
RunIdentifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
ItemIdentifier = Annotated[str, Field(min_length=1, max_length=512)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]


class RunState(StrEnum):
    PENDING = "pending"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_USER_ACTION = "needs_user_action"


class ItemState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class RunMode(StrEnum):
    SMOKE = "smoke"
    SAMPLE = "sample"
    FULL = "full"


class Comparability(StrEnum):
    EXACT = "exact"
    NEAR = "near"
    INCOMPATIBLE = "incompatible"


class ImplementationStatus(StrEnum):
    PLANNED = "planned"
    AVAILABLE = "available"
    BLOCKED = "blocked"


class SourceType(StrEnum):
    ORIGINAL_PAPER = "original_paper"
    BENCHMARK_PAPER = "benchmark_paper"
    OFFICIAL_LEADERBOARD = "official_leaderboard"
    PROVIDER_SYSTEM_CARD = "provider_system_card"
    PROVIDER_BLOG = "provider_blog"
    PAPER_COMPILATION = "paper_compilation"
    VERIFIED_THIRD_PARTY = "verified_third_party"


class _FrozenJsonMapping(Mapping[str, JsonValue]):
    """Read-only JSON mapping backed by an unmodifiable proxy."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, JsonValue]) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("evaluation JSON values are immutable")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("evaluation JSON values are immutable")

    def __getitem__(self, key: str) -> JsonValue:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return repr(dict(self._values))

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("evaluation JSON values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


class _FrozenJsonSequence(Sequence[JsonValue]):
    """Read-only JSON sequence backed by a tuple."""

    __slots__ = ("_values",)

    def __init__(self, values: Sequence[JsonValue]) -> None:
        object.__setattr__(self, "_values", tuple(values))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("evaluation JSON values are immutable")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("evaluation JSON values are immutable")

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sequence)
            and not isinstance(other, (str, bytes, bytearray))
            and tuple(self) == tuple(other)
        )

    def __repr__(self) -> str:
        return repr(list(self._values))

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("evaluation JSON values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def freeze_json_value(value: JsonValue) -> JsonValue:
    """Recursively copy JSON into non-builtin immutable containers."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("protocol JSON numbers must be finite")
    if isinstance(value, Mapping):
        return cast(
            JsonValue,
            _FrozenJsonMapping({key: freeze_json_value(child) for key, child in value.items()}),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast(
            JsonValue,
            _FrozenJsonSequence([freeze_json_value(child) for child in value]),
        )
    return value


def thaw_json_value(value: object) -> JsonValue:
    """Recursively copy immutable evaluation JSON into canonical builtin JSON."""

    if isinstance(value, Mapping):
        return cast(
            JsonValue,
            {str(key): thaw_json_value(child) for key, child in value.items()},
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast(JsonValue, [thaw_json_value(child) for child in value])
    return cast(JsonValue, value)


class FrozenModel(BaseModel):
    _secret_registry: SecretValueRegistry | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        str_strip_whitespace=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, object] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a fully revalidated copy instead of Pydantic's unsafe update."""
        payload = self.model_dump(round_trip=True)
        if deep:
            payload = copy.deepcopy(payload)
        if update is not None:
            payload.update(update)
        context = (
            {"secret_registry": self._secret_registry}
            if self._secret_registry is not None
            else None
        )
        return type(self).model_validate(payload, context=context)

    @model_validator(mode="before")
    @classmethod
    def _raw_input_is_secret_free(cls, value, info: ValidationInfo):
        ensure_secret_free_json(
            value,
            secret_registry=secret_registry_from_context(info.context),
        )
        return value

    @model_validator(mode="after")
    def _normalised_snapshot_is_secret_free(
        self,
        info: ValidationInfo,
    ) -> FrozenModel:
        registry = secret_registry_from_context(info.context)
        ensure_secret_free_json(
            self.model_dump(mode="json"),
            secret_registry=registry,
        )
        self._secret_registry = registry
        return self


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class BenchmarkDefinition(FrozenModel):
    schema_version: PositiveInt = 1
    benchmark_id: Identifier
    display_name: Annotated[str, Field(min_length=1, max_length=200)]
    description: str = ""
    benchmark_version: Annotated[str, Field(min_length=1)]
    licenses: tuple[str, ...] = ()
    data_sources: tuple[str, ...] = ()
    required_auth: tuple[str, ...] = ()
    primary_metric: Annotated[str, Field(min_length=1)]
    auxiliary_metrics: tuple[str, ...] = ()
    higher_is_better: bool = True
    modalities: tuple[str, ...] = ("text",)
    required_capabilities: tuple[str, ...] = ()
    supports_resume: bool = False
    implementation_status: ImplementationStatus = ImplementationStatus.PLANNED

    @field_validator("licenses", "data_sources", "required_auth", "modalities")
    @classmethod
    def _non_blank_entries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("list entries must be non-blank")
        return value


class ProtocolSignature(FrozenModel):
    schema_version: PositiveInt = 1
    benchmark_id: Identifier
    benchmark_version: Annotated[str, Field(min_length=1)]
    dataset_revision: Annotated[str, Field(min_length=1)]
    split: Annotated[str, Field(min_length=1)]
    sample_filter: dict[str, JsonValue] = Field(default_factory=dict)
    harness_name: Annotated[str, Field(min_length=1)]
    harness_version: Annotated[str, Field(min_length=1)]
    agent_scaffold: str | None = None
    prompt_version: Annotated[str, Field(min_length=1)]
    modalities: tuple[str, ...] = ("text",)
    tools: tuple[str, ...] = ()
    web_access: bool = False
    max_turns: NonNegativeInt | None = None
    retries: NonNegativeInt = 0
    timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    context_length: PositiveInt | None = None
    needle_count: PositiveInt | None = None
    judge_model: str | None = None
    judge_prompt_version: str | None = None
    judge_reasoning_mode: str | None = None
    simulator_model: str | None = None
    generation_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reasoning_effort: str | None = None
    metric_implementation: Annotated[str, Field(min_length=1)]
    dependency_compatibility_patches: tuple[str, ...] = ()
    code_execution_sandbox: dict[str, JsonValue] = Field(default_factory=dict)
    unresolved_fields: tuple[str, ...] = ()

    @field_validator(
        "sample_filter",
        "generation_parameters",
        "code_execution_sandbox",
    )
    @classmethod
    def _json_fields_are_secret_free(
        cls,
        value: dict[str, JsonValue],
        info: ValidationInfo,
    ) -> dict[str, JsonValue]:
        ensure_secret_free_json(
            value,
            secret_registry=secret_registry_from_context(info.context),
        )
        return cast(dict[str, JsonValue], freeze_json_value(value))

    @field_serializer(
        "sample_filter",
        "generation_parameters",
        "code_execution_sandbox",
    )
    def _serialize_json_fields(self, value: object) -> JsonValue:
        return thaw_json_value(value)

    @field_validator("unresolved_fields")
    @classmethod
    def _known_unique_unresolved_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        known = set(cls.model_fields) - {"unresolved_fields"}
        unknown = set(value) - known
        if unknown:
            raise ValueError("unresolved_fields contains unknown protocol field names")
        if len(set(value)) != len(value):
            raise ValueError("unresolved protocol fields must be unique")
        return tuple(sorted(value))


class BenchmarkProfile(FrozenModel):
    schema_version: PositiveInt = 1
    name: Identifier
    benchmark_id: Identifier
    description: str = ""
    protocol: ProtocolSignature
    expected_full_count: NonNegativeInt | None = None
    source_urls: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _protocol_matches_benchmark(self) -> BenchmarkProfile:
        if self.protocol.benchmark_id != self.benchmark_id:
            raise ValueError("profile and protocol benchmark IDs must match")
        return self


class BenchmarkRun(FrozenModel):
    schema_version: PositiveInt = 1
    run_id: RunIdentifier
    benchmark_id: Identifier
    profile: Identifier
    mode: RunMode
    state: RunState = RunState.PENDING
    partial: bool = False
    termination_reason: str | None = None
    protocol_hash: Sha256 | None = None
    item_input_manifest_sha256: Sha256 | None = None
    selected_item_ids: tuple[ItemIdentifier, ...] = ()
    expected_full_count: NonNegativeInt | None = None
    completed_count: NonNegativeInt = 0
    failed_count: NonNegativeInt = 0
    skipped_count: NonNegativeInt = 0
    target_model: Annotated[str, Field(min_length=1)]
    judge_model: str | None = None
    simulator_model: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt: PositiveInt = 1
    resumed_from_run_id: RunIdentifier | None = None

    @field_validator("selected_item_ids")
    @classmethod
    def _selected_items_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("selected item IDs must be unique")
        return value

    @field_validator("created_at", "started_at", "finished_at")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime | None, info):
        return _aware(value, info.field_name)

    @model_validator(mode="after")
    def _counts_and_times_are_consistent(self) -> BenchmarkRun:
        identity_states = {
            RunState.READY,
            RunState.RUNNING,
            RunState.PARTIAL,
            RunState.COMPLETED,
            RunState.FAILED,
        }
        if self.state in identity_states and (
            self.protocol_hash is None or self.item_input_manifest_sha256 is None
        ):
            raise ValueError("prepared run states require protocol and item-input manifest hashes")
        selected_count = len(self.selected_item_ids)
        accounted = self.completed_count + self.failed_count + self.skipped_count
        if selected_count and accounted > selected_count:
            raise ValueError("completed, failed, and skipped counts exceed selected items")
        if self.started_at and self.started_at < self.created_at:
            raise ValueError("started_at precedes created_at")
        terminal_states = {
            RunState.CANCELLED,
            RunState.PARTIAL,
            RunState.COMPLETED,
            RunState.FAILED,
        }
        requires_start = {RunState.RUNNING, RunState.CANCELLING, *terminal_states}
        if self.state in requires_start and self.started_at is None:
            raise ValueError("executing and terminal states require started_at")
        if self.state in terminal_states and self.finished_at is None:
            raise ValueError("terminal states require finished_at")
        if self.state not in terminal_states and self.finished_at is not None:
            raise ValueError("finished_at is only valid for terminal states")
        if self.finished_at and self.started_at and self.finished_at < self.started_at:
            raise ValueError("finished_at precedes started_at")
        partial_states = {
            RunState.CANCELLING,
            RunState.CANCELLED,
            RunState.PARTIAL,
            RunState.FAILED,
            RunState.BLOCKED,
            RunState.NEEDS_USER_ACTION,
        }
        if self.state is RunState.PARTIAL and not self.partial:
            raise ValueError("partial runs must retain partial evidence")
        if self.partial and self.state not in partial_states:
            raise ValueError("partial evidence is invalid for the current run state")
        if self.attempt == 1 and self.resumed_from_run_id is not None:
            raise ValueError("first attempt cannot have a resumed-from run")
        if self.attempt > 1 and self.resumed_from_run_id is None:
            raise ValueError("resumed attempts require a source run")
        if self.resumed_from_run_id == self.run_id:
            raise ValueError("a run cannot resume from itself")
        return self


class RunItem(FrozenModel):
    schema_version: PositiveInt = 1
    run_id: RunIdentifier
    item_id: Annotated[str, Field(min_length=1, max_length=512)]
    ordinal: NonNegativeInt
    state: ItemState = ItemState.PENDING
    attempt: PositiveInt = 1
    input_sha256: Sha256
    checkpoint: str | None = None
    error_class: str | None = None
    scores: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = Field(default_factory=dict)

    @field_validator("scores")
    @classmethod
    def _scores_are_secret_free_and_frozen(
        cls,
        value: dict[str, float],
        info: ValidationInfo,
    ) -> dict[str, float]:
        ensure_secret_free_json(
            value,
            secret_registry=secret_registry_from_context(info.context),
        )
        return cast(dict[str, float], freeze_json_value(value))

    @field_serializer("scores")
    def _serialize_scores(self, value: object) -> JsonValue:
        return thaw_json_value(value)


class Artifact(FrozenModel):
    schema_version: PositiveInt = 1
    run_id: RunIdentifier
    name: Annotated[str, Field(min_length=1, max_length=255)]
    relative_path: Annotated[str, Field(min_length=1, max_length=1024)]
    media_type: Annotated[str, Field(min_length=1, max_length=255)]
    sha256: Sha256
    size_bytes: NonNegativeInt
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("relative_path")
    @classmethod
    def _portable_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        safe_component = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
        if (
            "\\" in value
            or path.is_absolute()
            or path.as_posix() != value
            or not path.parts
            or any(not safe_component.fullmatch(part) for part in path.parts)
        ):
            raise ValueError("artifact path must be canonical portable safe ASCII components")
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at_is_aware(cls, value: datetime) -> datetime:
        normalised = _aware(value, "created_at")
        assert normalised is not None
        return normalised


class Metric(FrozenModel):
    schema_version: PositiveInt = 1
    run_id: RunIdentifier
    name: Identifier
    display_name: Annotated[str, Field(min_length=1)]
    value: Annotated[float, Field(allow_inf_nan=False)] | None = None
    numerator: NonNegativeInt | None = None
    denominator: NonNegativeInt | None = None
    scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 100.0
    unit: Annotated[str, Field(min_length=1)] = "percent"
    primary: bool = False
    higher_is_better: bool = True
    dimensions: dict[str, JsonValue] = Field(default_factory=dict)
    official_eligible: bool = False

    @field_validator("dimensions")
    @classmethod
    def _dimensions_are_secret_free(
        cls,
        value: dict[str, JsonValue],
        info: ValidationInfo,
    ) -> dict[str, JsonValue]:
        ensure_secret_free_json(
            value,
            secret_registry=secret_registry_from_context(info.context),
        )
        return cast(dict[str, JsonValue], freeze_json_value(value))

    @field_serializer("dimensions")
    def _serialize_dimensions(self, value: object) -> JsonValue:
        return thaw_json_value(value)


class Source(FrozenModel):
    schema_version: PositiveInt = 1
    source_id: Identifier
    source_type: SourceType
    title: Annotated[str, Field(min_length=1)]
    url: Annotated[str, Field(min_length=1)]
    locator: Annotated[str, Field(min_length=1)]
    release_page: str | None = None
    publication_date: date | None = None
    retrieved_at: datetime
    evidence_hash: Sha256
    notes: str = ""

    @field_validator("retrieved_at")
    @classmethod
    def _retrieved_at_is_aware(cls, value: datetime) -> datetime:
        normalised = _aware(value, "retrieved_at")
        assert normalised is not None
        return normalised


class ReferenceResult(FrozenModel):
    schema_version: PositiveInt = 1
    reference_id: Identifier
    benchmark_id: Identifier
    benchmark_version: Annotated[str, Field(min_length=1)]
    profile: Identifier
    protocol_hash: Sha256 | None = None
    model_name: Annotated[str, Field(min_length=1)]
    model_version: str | None = None
    score: Annotated[float, Field(allow_inf_nan=False)]
    score_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 100.0
    metric_name: Annotated[str, Field(min_length=1)]
    sample_count: NonNegativeInt | None = None
    source_id: Identifier
    source_type: SourceType
    provider_reported: bool | None = None
    independently_reproduced: bool = False
    publication_date: date | None = None
    retrieved_at: datetime
    notes: str = ""
    comparability: Comparability = Comparability.INCOMPATIBLE
    evidence_hash: Sha256

    @model_validator(mode="after")
    def _comparability_requires_protocol_evidence(self) -> ReferenceResult:
        if self.comparability is not Comparability.INCOMPATIBLE and self.protocol_hash is None:
            raise ValueError("comparable reference results require a protocol hash")
        return self

    @field_validator("retrieved_at")
    @classmethod
    def _reference_retrieved_at_is_aware(cls, value: datetime) -> datetime:
        normalised = _aware(value, "retrieved_at")
        assert normalised is not None
        return normalised


def is_sha256(value: str) -> bool:
    """Return whether ``value`` is the lowercase canonical SHA-256 form."""
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None
