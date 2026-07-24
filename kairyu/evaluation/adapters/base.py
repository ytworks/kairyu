"""Common contract for independently runnable benchmark adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import Field, JsonValue, field_serializer, field_validator, model_validator

from kairyu.evaluation.schemas import (
    BenchmarkDefinition,
    BenchmarkProfile,
    FrozenModel,
    Metric,
    ProtocolSignature,
    RunMode,
    freeze_json_value,
    thaw_json_value,
)

if TYPE_CHECKING:
    from kairyu.evaluation.connectors import ModelConnector


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class PreparationStatus(StrEnum):
    DRY_RUN = "dry_run"
    READY = "ready"
    NEEDS_USER_ACTION = "needs_user_action"
    BLOCKED = "blocked"


class DoctorCheck(FrozenModel):
    """One structured, non-secret environment diagnostic."""

    check_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    status: CheckStatus
    summary: str = Field(min_length=1, max_length=500)
    action: str | None = Field(default=None, max_length=1_000)


class DoctorReport(FrozenModel):
    """Adapter diagnostics; one benchmark's failure never hides other adapters."""

    benchmark_id: str
    profile: str
    runnable: bool
    checks: tuple[DoctorCheck, ...]


class PreparationResult(FrozenModel):
    """Result of an explicit, non-implicit preparation request."""

    benchmark_id: str
    profile: str
    status: PreparationStatus
    dry_run: bool
    dataset_revision: str | None = None
    dataset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    item_count: int | None = Field(default=None, ge=0)
    actions: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()


class RunSelection(FrozenModel):
    """Raw execution selection revalidated at every external boundary."""

    profile: str
    mode: RunMode = RunMode.SMOKE
    target_model: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1)
    sample_ids: tuple[str, ...] = ()
    seed: int = 42
    confirm_full_run: bool = False
    dataset_path: str | None = None
    dataset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    accepted_access: bool = False
    generation_parameters: dict[str, JsonValue] = Field(
        default_factory=lambda: {
            "temperature": 0.0,
            "repeats": 1,
        }
    )

    @field_validator("sample_ids")
    @classmethod
    def _sample_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("sample IDs must be non-blank")
        if len(set(value)) != len(value):
            raise ValueError("sample IDs must be unique")
        return value

    @field_validator("generation_parameters")
    @classmethod
    def _generation_parameters_are_immutable(
        cls,
        value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], freeze_json_value(value))

    @field_serializer("generation_parameters")
    def _serialize_generation_parameters(self, value: object) -> JsonValue:
        return thaw_json_value(value)


class RunEstimate(FrozenModel):
    """Truthful preflight estimates; unknown values always carry assumptions."""

    selected_item_count: int = Field(ge=1)
    model_calls: int = Field(ge=0)
    maximum_model_calls: int = Field(ge=0)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    maximum_output_tokens: int | None = Field(default=None, ge=0)
    estimated_duration_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    maximum_duration_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    estimated_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _maximum_covers_estimate(self) -> RunEstimate:
        if self.maximum_model_calls < self.model_calls:
            raise ValueError("maximum model calls cannot be smaller than estimated calls")
        return self


class ResourceRequirements(FrozenModel):
    """Minimum local resources and execution boundaries for one plan."""

    cpu_cores: int = Field(ge=1)
    ram_bytes: int = Field(ge=0)
    disk_bytes: int = Field(ge=0)
    docker_required: bool
    network_policy: str = Field(min_length=1, max_length=500)


class ExecutionSpec(FrozenModel):
    """How the durable plan reaches its adapter execution entry point."""

    kind: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    command: tuple[str, ...]
    container_image_digest: str | None = None


@dataclass(frozen=True, slots=True)
class AdapterItem:
    """Prepared item kept in worker memory.

    Prompt and choice text may contain licensed benchmark material and must not
    be copied into control metadata or reports.  Only ``item_id`` and
    ``input_sha256`` cross those boundaries.
    """

    item_id: str
    ordinal: int
    input_sha256: str
    prompt: str
    target: str
    choice_permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AdapterRunPlan:
    benchmark_id: str
    profile: str
    mode: RunMode
    target_model: str
    items: tuple[AdapterItem, ...]
    protocol: ProtocolSignature
    protocol_hash: str
    item_input_manifest_sha256: str
    expected_full_count: int | None
    estimated_model_calls: int
    estimate: RunEstimate
    resources: ResourceRequirements
    execution: ExecutionSpec
    official_eligible: bool
    selection: RunSelection


class ItemResult(FrozenModel):
    """Secret-free per-item result suitable for a checkpoint artifact."""

    item_id: str
    ordinal: int = Field(ge=0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_text: str
    extracted_answer: str | None = None
    target: str
    correct: bool = False
    error_class: str | None = None
    latency_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    finish_reason: str | None = None
    provider_request_id: str | None = None
    provider_model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class CollectedResult(FrozenModel):
    """Adapter-normalized aggregate generated from immutable item results."""

    metrics: tuple[Metric, ...]
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    error_counts: dict[str, int] = Field(default_factory=dict)


CancelCheck = Callable[[], bool]


class BenchmarkAdapter(ABC):
    """Stable adapter boundary used by the controller and worker."""

    @abstractmethod
    def metadata(self) -> BenchmarkDefinition:
        """Return static metadata without host or network access."""

    @abstractmethod
    def profiles(self) -> tuple[BenchmarkProfile, ...]:
        """Return immutable, versioned execution profiles."""

    @abstractmethod
    def doctor(
        self,
        profile: str,
        *,
        dataset_path: Path | None = None,
    ) -> DoctorReport:
        """Inspect prerequisites without downloading or executing the benchmark."""

    @abstractmethod
    def prepare(
        self,
        profile: str,
        *,
        dry_run: bool,
        dataset_path: Path | None = None,
        dataset_sha256: str | None = None,
        accepted_access: bool = False,
    ) -> PreparationResult:
        """Validate an approved local snapshot; never auto-accept gated terms."""

    @abstractmethod
    def build_run_plan(
        self,
        selection: RunSelection,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> AdapterRunPlan:
        """Revalidate scope before dataset reads or model/executor startup."""

    @abstractmethod
    def protocol_signature(self, plan: AdapterRunPlan) -> ProtocolSignature:
        """Return the exact critical protocol represented by a plan."""

    @abstractmethod
    def run(
        self,
        plan: AdapterRunPlan,
        item: AdapterItem,
        connector: ModelConnector,
        *,
        cancel_check: CancelCheck,
    ) -> ItemResult:
        """Execute one prepared item with graceful cancellation."""

    @abstractmethod
    def collect(
        self,
        run_id: str,
        plan: AdapterRunPlan,
        results: tuple[ItemResult, ...],
    ) -> CollectedResult:
        """Normalize saved item results without changing upstream semantics."""

    @abstractmethod
    def render_report_data(
        self,
        collected: CollectedResult,
    ) -> Mapping[str, Any]:
        """Return adapter-specific report fields from collected evidence."""


__all__ = [
    "AdapterItem",
    "AdapterRunPlan",
    "BenchmarkAdapter",
    "CancelCheck",
    "CheckStatus",
    "CollectedResult",
    "DoctorCheck",
    "DoctorReport",
    "ExecutionSpec",
    "ItemResult",
    "PreparationResult",
    "PreparationStatus",
    "ResourceRequirements",
    "RunEstimate",
    "RunSelection",
]
