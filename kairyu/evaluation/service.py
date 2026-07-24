"""Controller-facing submission and lifecycle service."""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from kairyu.evaluation.adapters import get_adapter
from kairyu.evaluation.adapters.base import (
    AdapterRunPlan,
    ExecutionSpec,
    ResourceRequirements,
    RunEstimate,
    RunSelection,
)
from kairyu.evaluation.artifacts import ArtifactStore
from kairyu.evaluation.connectors import normalize_openai_base_url
from kairyu.evaluation.guards import validate_run_guard
from kairyu.evaluation.protocol import protocol_hash
from kairyu.evaluation.safety import SecretValueRegistry
from kairyu.evaluation.schemas import (
    BenchmarkRun,
    FrozenModel,
    ProtocolSignature,
    RunMode,
)
from kairyu.evaluation.sqlite_store import SqliteControlStore


class ConnectorConfig(FrozenModel):
    kind: str = Field(pattern=r"^(fake|openai)$")
    endpoint: str | None = None
    secret_env_name: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    max_response_bytes: int = Field(default=1_048_576, ge=1, le=16_777_216)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("endpoint")
    @classmethod
    def _canonical_endpoint(cls, value: str | None) -> str | None:
        return normalize_openai_base_url(value) if value is not None else None

    @model_validator(mode="after")
    def _kind_specific_fields(self) -> ConnectorConfig:
        if self.kind == "fake" and (self.endpoint is not None or self.secret_env_name is not None):
            raise ValueError("fake connector cannot have endpoint credentials")
        if self.kind == "openai" and self.endpoint is None:
            raise ValueError("OpenAI-compatible connector requires an endpoint")
        return self


class BenchmarkJobPayload(FrozenModel):
    """Validated, secret-free durable input bound to one run identity."""

    benchmark_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    connector: ConnectorConfig
    official_eligible: bool
    protocol: ProtocolSignature
    selection: RunSelection

    @model_validator(mode="after")
    def _cross_fields_match(self) -> BenchmarkJobPayload:
        if self.protocol.benchmark_id != self.benchmark_id:
            raise ValueError("job protocol benchmark ID does not match")
        if self.official_eligible and self.protocol.unresolved_fields:
            raise ValueError("unresolved protocol cannot be official eligible")
        return self


class SubmissionResult(FrozenModel):
    run: BenchmarkRun
    job_id: str
    estimated_model_calls: int = Field(ge=0)
    estimate: RunEstimate
    resources: ResourceRequirements
    execution: ExecutionSpec
    official_eligible: bool


def validate_runtime_connector(
    runtime: EvaluationRuntime,
    connector: ConnectorConfig,
) -> ConnectorConfig:
    """Register an available runtime credential before metadata validation."""

    if connector.secret_env_name is not None:
        secret = os.environ.get(connector.secret_env_name)
        if secret:
            runtime.secret_registry.register(secret)
    return ConnectorConfig.model_validate(
        connector.model_dump(mode="python"),
        context={"secret_registry": runtime.secret_registry},
    )


def bind_connector_to_plan(
    plan: AdapterRunPlan,
    connector: ConnectorConfig,
) -> AdapterRunPlan:
    """Bind connector identity, effective retries, and final planning evidence."""

    protocol_payload = plan.protocol.model_dump(mode="json")
    connector_payload = connector.model_dump(mode="json")
    effective_retries = connector.max_retries if connector.kind == "openai" else 0
    retry_backoff_seconds = _default_connector_backoff_seconds(effective_retries)
    maximum_duration_seconds = plan.estimate.maximum_duration_seconds
    if maximum_duration_seconds is not None:
        maximum_duration_seconds *= effective_retries + 1
        maximum_duration_seconds += plan.estimate.model_calls * retry_backoff_seconds
    estimate = plan.estimate.model_copy(
        update={
            "maximum_model_calls": plan.estimate.model_calls * (effective_retries + 1),
            "maximum_duration_seconds": maximum_duration_seconds,
            "assumptions": (
                *plan.estimate.assumptions,
                "Maximum calls and duration include all configured connector attempts "
                f"and {retry_backoff_seconds:g} seconds of default backoff per item.",
            ),
        }
    )
    reviewed_connector = protocol_payload["adapter_configuration"].get("reviewed_model_connector")
    connector_matches_review = (
        reviewed_connector is not None and reviewed_connector == connector_payload
    )
    planning_evidence = {
        "estimate": estimate.model_dump(mode="json"),
        "resources": plan.resources.model_dump(mode="json"),
        "execution": plan.execution.model_dump(mode="json"),
    }
    adapter_configuration = {
        **dict(protocol_payload["adapter_configuration"]),
        "model_connector": connector_payload,
        "model_connector_review": {
            "status": "verified" if connector_matches_review else "unresolved",
            "reason": (
                "Connector configuration matches the reviewed profile identity."
                if connector_matches_review
                else "The profile does not pin this exact connector configuration."
            ),
        },
        "planning_evidence": planning_evidence,
    }
    protocol = plan.protocol.model_copy(
        update={
            "adapter_configuration": adapter_configuration,
            "retries": effective_retries,
        }
    )
    return replace(
        plan,
        protocol=protocol,
        protocol_hash=protocol_hash(protocol),
        estimate=estimate,
        official_eligible=(
            plan.official_eligible
            and connector_matches_review
            and effective_retries == plan.protocol.retries
        ),
    )


def _default_connector_backoff_seconds(retries: int) -> float:
    return sum(min(0.5 * (2 ** (attempt - 1)), 4.0) for attempt in range(1, retries + 1))


class EvaluationRuntime:
    """One local controller instance backed by SQLite and immutable artifacts."""

    def __init__(
        self,
        state_directory: str | Path,
        *,
        secret_registry: SecretValueRegistry | None = None,
    ) -> None:
        state_root = Path(state_directory).expanduser().absolute()
        state_root.mkdir(parents=True, exist_ok=True)
        self.state_directory = state_root
        self.secret_registry = secret_registry or SecretValueRegistry()
        self.store = SqliteControlStore(
            state_root / "control.sqlite3",
            secret_registry=self.secret_registry,
        )
        self.artifacts = ArtifactStore(
            state_root / "benchmark_runs",
            publication_guard=self.store.publication_guard,
            secret_registry=self.secret_registry,
        )


class BenchmarkService:
    def __init__(self, runtime: EvaluationRuntime) -> None:
        self.runtime = runtime

    def submit(
        self,
        benchmark_id: str,
        selection: RunSelection,
        connector: ConnectorConfig,
        *,
        run_id: str | None = None,
    ) -> SubmissionResult:
        """Validate, plan, and durably enqueue without executing a model call."""

        # Controller authorization is raw and local; neither a plan nor a job
        # payload is accepted as proof that a full run was authorized.
        validate_run_guard(
            selection.mode,
            confirm_full_run=selection.confirm_full_run,
            limit=selection.limit,
            sample_ids=selection.sample_ids,
        )
        connector = validate_runtime_connector(self.runtime, connector)
        if connector.kind == "fake" and selection.mode is not RunMode.SMOKE:
            raise ValueError("the fixed fake connector is smoke-only")
        adapter = get_adapter(benchmark_id)
        plan = bind_connector_to_plan(
            adapter.build_run_plan(selection),
            connector,
        )
        selected_run_id = run_id or f"run-{uuid.uuid4().hex}"
        run = BenchmarkRun(
            run_id=selected_run_id,
            benchmark_id=benchmark_id,
            profile=selection.profile,
            mode=selection.mode,
            protocol_hash=plan.protocol_hash,
            item_input_manifest_sha256=plan.item_input_manifest_sha256,
            selected_item_ids=tuple(item.item_id for item in plan.items),
            expected_full_count=plan.expected_full_count,
            target_model=selection.target_model,
        )
        job_payload = BenchmarkJobPayload(
            benchmark_id=benchmark_id,
            connector=connector,
            official_eligible=plan.official_eligible,
            protocol=plan.protocol,
            selection=plan.selection,
        )
        self.runtime.artifacts.create_run(run.run_id)
        created = self.runtime.store.create_run_and_enqueue(
            run,
            payload=job_payload.model_dump(mode="json"),
        )
        return SubmissionResult(
            run=created.run.run,
            job_id=created.job.job_id,
            estimated_model_calls=plan.estimated_model_calls,
            estimate=plan.estimate,
            resources=plan.resources,
            execution=plan.execution,
            official_eligible=plan.official_eligible,
        )

    def cancel(self, run_id: str):
        job = self.runtime.store.request_cancel(run_id)
        if self.runtime.store.get_run(run_id).run.state.value == "cancelled":
            from kairyu.evaluation.worker import render_cancelled_job_report

            render_cancelled_job_report(self.runtime, run_id, job.payload)
        return job

    def resume(
        self,
        run_id: str,
        *,
        new_run_id: str | None = None,
    ):
        successor_id = new_run_id or f"run-{uuid.uuid4().hex}"
        self.runtime.artifacts.create_run(successor_id)
        result = self.runtime.store.resume_run(run_id, successor_id)
        return result

    def status(self, run_id: str):
        return self.runtime.store.get_run(run_id)


def rebuild_plan_from_job(payload: dict) -> AdapterRunPlan:
    """Recreate and reauthorize a plan from untrusted durable job metadata."""

    snapshot = BenchmarkJobPayload.model_validate(payload)
    selection = snapshot.selection
    # Worker-level guard is deliberately before adapter lookup/import.
    validate_run_guard(
        selection.mode,
        confirm_full_run=selection.confirm_full_run,
        limit=selection.limit,
        sample_ids=selection.sample_ids,
    )
    plan = bind_connector_to_plan(
        get_adapter(snapshot.benchmark_id).build_run_plan(selection),
        snapshot.connector,
    )
    if snapshot.protocol != plan.protocol or protocol_hash(snapshot.protocol) != plan.protocol_hash:
        raise ValueError("durable protocol snapshot does not match reconstructed plan")
    if snapshot.selection != plan.selection:
        raise ValueError("durable selection snapshot is not canonical")
    if snapshot.official_eligible is not plan.official_eligible:
        raise ValueError("durable official-eligibility flag does not match plan")
    return plan


__all__ = [
    "BenchmarkJobPayload",
    "BenchmarkService",
    "ConnectorConfig",
    "EvaluationRuntime",
    "SubmissionResult",
    "bind_connector_to_plan",
    "validate_runtime_connector",
    "rebuild_plan_from_job",
]
