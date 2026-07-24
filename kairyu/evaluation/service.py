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
    ModelRole,
    ModelRolePlan,
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
    judge_connector: ConnectorConfig | None = None
    simulator_connector: ConnectorConfig | None = None
    official_eligible: bool
    protocol: ProtocolSignature
    selection: RunSelection

    @model_validator(mode="after")
    def _cross_fields_match(self) -> BenchmarkJobPayload:
        if self.protocol.benchmark_id != self.benchmark_id:
            raise ValueError("job protocol benchmark ID does not match")
        if self.selection.judge_model != self.protocol.judge_model:
            raise ValueError("job judge model does not match the protocol")
        if self.selection.simulator_model != self.protocol.simulator_model:
            raise ValueError("job simulator model does not match the protocol")
        if (self.judge_connector is None) != (self.selection.judge_model is None):
            raise ValueError("judge model and connector must be configured together")
        if (self.simulator_connector is None) != (self.selection.simulator_model is None):
            raise ValueError("simulator model and connector must be configured together")
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
    *,
    judge_connector: ConnectorConfig | None = None,
    simulator_connector: ConnectorConfig | None = None,
) -> AdapterRunPlan:
    """Bind role-separated connector identity, retries, and planning evidence."""

    role_plans = _normalized_model_roles(plan)
    retry_budget_scope = plan.protocol.adapter_configuration.get("retry_budget_scope")
    max_attempts_per_request = plan.protocol.adapter_configuration.get("max_attempts_per_request")
    has_adapter_retry_budget = (
        retry_budget_scope is not None or max_attempts_per_request is not None
    )
    adapter_attempt_budget: int | None = None
    if has_adapter_retry_budget:
        if (
            retry_budget_scope != "per-model-request-total-attempts"
            or not isinstance(max_attempts_per_request, int)
            or isinstance(max_attempts_per_request, bool)
            or max_attempts_per_request < 1
            or max_attempts_per_request != plan.protocol.retries + 1
        ):
            raise ValueError("adapter retry budget marker is invalid or inconsistent")
        required_model_calls = sum(role.model_calls for role in role_plans) * (
            max_attempts_per_request
        )
        if plan.estimate.maximum_model_calls < required_model_calls:
            raise ValueError("adapter estimate does not cover its declared retry budget")
        adapter_attempt_budget = max_attempts_per_request
    connector, judge_connector, simulator_connector = _complete_smoke_connector_roles(
        plan,
        connector,
        judge_connector,
        simulator_connector,
        role_plans=role_plans,
    )
    configs = {
        ModelRole.TARGET: connector,
        ModelRole.JUDGE: judge_connector,
        ModelRole.SIMULATOR: simulator_connector,
    }
    required_roles = {role.role for role in role_plans if role.model_calls}
    for role in required_roles:
        if configs[role] is None:
            raise ValueError(f"{role.value} model calls require a connector")
    for role, configured in configs.items():
        if configured is not None and role not in required_roles:
            raise ValueError(f"{role.value} connector is configured but the plan has no calls")

    effective_retries: dict[ModelRole, int] = {}
    connector_payloads: dict[str, dict] = {}
    maximum_model_calls = 0
    maximum_duration_seconds = 0.0
    total_backoff_seconds = 0.0
    role_backoff_seconds: dict[ModelRole, float] = {}
    for role_plan in role_plans:
        configured = configs[role_plan.role]
        if configured is None or not role_plan.model_calls:
            continue
        retries = configured.max_retries if configured.kind == "openai" else 0
        if has_adapter_retry_budget:
            assert adapter_attempt_budget is not None
            backoff = _default_connector_backoff_for_attempt_budget(
                total_attempts=adapter_attempt_budget,
                retries_per_call=retries,
            )
        else:
            backoff = _default_connector_backoff_seconds(retries)
        effective_retries[role_plan.role] = retries
        connector_payloads[role_plan.role.value] = configured.model_dump(mode="json")
        maximum_model_calls += role_plan.model_calls * (retries + 1)
        maximum_duration_seconds += role_plan.model_calls * (
            role_plan.timeout_seconds * (retries + 1) + backoff
        )
        role_backoff_seconds[role_plan.role] = role_plan.model_calls * backoff
        total_backoff_seconds += role_backoff_seconds[role_plan.role]

    if has_adapter_retry_budget:
        assert adapter_attempt_budget is not None
        backoff_summary = ", ".join(
            f"{role.value}={role_backoff_seconds[role]:g}s"
            for role in sorted(role_backoff_seconds, key=lambda value: value.value)
        )
        retry_assumption = (
            f"Adapter-owned total attempt budgets cap each model request at "
            f"{adapter_attempt_budget} attempts; connector-internal retries add "
            f"{total_backoff_seconds:g} seconds of worst-case default backoff "
            f"across role calls ({backoff_summary}) without increasing model-call "
            "or output-token ceilings."
        )
    elif len(effective_retries) == 1 and ModelRole.TARGET in effective_retries:
        per_item_backoff = _default_connector_backoff_seconds(effective_retries[ModelRole.TARGET])
        retry_assumption = (
            "Maximum calls and duration include all configured connector attempts "
            f"and {per_item_backoff:g} seconds of default backoff per item."
        )
    else:
        retry_summary = ", ".join(
            f"{role.value}={effective_retries[role]}"
            for role in sorted(effective_retries, key=lambda value: value.value)
        )
        retry_assumption = (
            "Maximum calls and duration include role-specific connector attempts "
            f"({retry_summary}) and {total_backoff_seconds:g} seconds of total backoff."
        )
    if has_adapter_retry_budget:
        estimate = plan.estimate.model_copy(
            update={
                "maximum_duration_seconds": (
                    plan.estimate.maximum_duration_seconds + total_backoff_seconds
                    if plan.estimate.maximum_duration_seconds is not None
                    else None
                ),
                "assumptions": (*plan.estimate.assumptions, retry_assumption),
            }
        )
    else:
        estimate = plan.estimate.model_copy(
            update={
                "maximum_model_calls": max(
                    plan.estimate.maximum_model_calls,
                    maximum_model_calls,
                ),
                "maximum_duration_seconds": max(
                    plan.estimate.maximum_duration_seconds or 0.0,
                    maximum_duration_seconds,
                ),
                "assumptions": (*plan.estimate.assumptions, retry_assumption),
            }
        )

    protocol_payload = plan.protocol.model_dump(mode="json")
    base_configuration = dict(protocol_payload["adapter_configuration"])
    reviewed_by_role = base_configuration.get("reviewed_model_connectors")
    if not isinstance(reviewed_by_role, dict):
        reviewed_by_role = {}
    if "target" not in reviewed_by_role:
        legacy_reviewed = base_configuration.get("reviewed_model_connector")
        if legacy_reviewed is not None:
            reviewed_by_role["target"] = legacy_reviewed
    connector_reviews: dict[str, dict[str, str]] = {}
    for role_name, payload in connector_payloads.items():
        matches = reviewed_by_role.get(role_name) == payload
        connector_reviews[role_name] = {
            "status": "verified" if matches else "unresolved",
            "reason": (
                "Connector configuration matches the reviewed profile identity."
                if matches
                else "The profile does not pin this exact connector configuration."
            ),
        }

    planning_evidence = {
        "estimate": estimate.model_dump(mode="json"),
        "resources": plan.resources.model_dump(mode="json"),
        "execution": plan.execution.model_dump(mode="json"),
    }
    adapter_configuration = {
        **base_configuration,
        "model_connector": connector_payloads[ModelRole.TARGET.value],
        "model_connector_review": connector_reviews[ModelRole.TARGET.value],
        "planning_evidence": planning_evidence,
    }
    if len(connector_payloads) > 1:
        adapter_configuration.update(
            {
                "model_connectors": connector_payloads,
                "model_connector_reviews": connector_reviews,
                "model_calls_by_role": {role.role.value: role.model_calls for role in role_plans},
            }
        )
    maximum_retries = (
        plan.protocol.retries
        if has_adapter_retry_budget
        else max(
            plan.protocol.retries,
            max(effective_retries.values(), default=0),
        )
    )
    protocol = plan.protocol.model_copy(
        update={
            "adapter_configuration": adapter_configuration,
            "retries": maximum_retries,
        }
    )
    connectors_match_review = all(
        review["status"] == "verified" for review in connector_reviews.values()
    )
    retries_match_review = has_adapter_retry_budget or all(
        retries == plan.protocol.retries for retries in effective_retries.values()
    )
    return replace(
        plan,
        protocol=protocol,
        protocol_hash=protocol_hash(protocol),
        estimate=estimate,
        official_eligible=(
            plan.official_eligible and connectors_match_review and retries_match_review
        ),
    )


def _normalized_model_roles(plan: AdapterRunPlan) -> tuple[ModelRolePlan, ...]:
    roles = plan.model_roles
    if not roles:
        timeout = plan.protocol.timeout_seconds
        if timeout is None:
            timeout = float(plan.protocol.generation_parameters.get("timeout_seconds", 600.0))
        roles = (
            ModelRolePlan(
                role=ModelRole.TARGET,
                model=plan.target_model,
                model_calls=plan.estimate.model_calls,
                timeout_seconds=timeout,
                maximum_output_tokens=plan.estimate.maximum_output_tokens or 0,
            ),
        )
    role_names = [entry.role for entry in roles]
    if len(set(role_names)) != len(role_names):
        raise ValueError("model role plans must be unique")
    if sum(entry.model_calls for entry in roles) != plan.estimate.model_calls:
        raise ValueError("model role calls do not sum to the run estimate")
    expected_models = {
        ModelRole.TARGET: plan.target_model,
        ModelRole.JUDGE: plan.selection.judge_model,
        ModelRole.SIMULATOR: plan.selection.simulator_model,
    }
    for entry in roles:
        if expected_models[entry.role] != entry.model:
            raise ValueError(f"{entry.role.value} role model does not match the selection")
    return roles


def _complete_smoke_connector_roles(
    plan: AdapterRunPlan,
    connector: ConnectorConfig,
    judge_connector: ConnectorConfig | None,
    simulator_connector: ConnectorConfig | None,
    *,
    role_plans: tuple[ModelRolePlan, ...] | None = None,
) -> tuple[ConnectorConfig, ConnectorConfig | None, ConnectorConfig | None]:
    roles = role_plans or _normalized_model_roles(plan)
    configs = {
        ModelRole.TARGET: connector,
        ModelRole.JUDGE: judge_connector,
        ModelRole.SIMULATOR: simulator_connector,
    }
    for role_plan in roles:
        if (
            role_plan.model_calls
            and configs[role_plan.role] is None
            and connector.kind == "fake"
            and plan.mode is RunMode.SMOKE
        ):
            configs[role_plan.role] = connector
    return (
        configs[ModelRole.TARGET],
        configs[ModelRole.JUDGE],
        configs[ModelRole.SIMULATOR],
    )


def _default_connector_backoff_seconds(retries: int) -> float:
    return sum(min(0.5 * (2 ** (attempt - 1)), 4.0) for attempt in range(1, retries + 1))


def _default_connector_backoff_for_attempt_budget(
    *,
    total_attempts: int,
    retries_per_call: int,
) -> float:
    attempts_per_call = retries_per_call + 1
    full_calls, final_attempts = divmod(total_attempts, attempts_per_call)
    return full_calls * _default_connector_backoff_seconds(retries_per_call) + (
        _default_connector_backoff_seconds(final_attempts - 1) if final_attempts else 0.0
    )


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
        judge_connector: ConnectorConfig | None = None,
        simulator_connector: ConnectorConfig | None = None,
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
        judge_connector = (
            validate_runtime_connector(self.runtime, judge_connector)
            if judge_connector is not None
            else None
        )
        simulator_connector = (
            validate_runtime_connector(self.runtime, simulator_connector)
            if simulator_connector is not None
            else None
        )
        adapter = get_adapter(benchmark_id)
        unbound_plan = adapter.build_run_plan(selection)
        connector, judge_connector, simulator_connector = _complete_smoke_connector_roles(
            unbound_plan,
            connector,
            judge_connector,
            simulator_connector,
        )
        configured_connectors = tuple(
            value
            for value in (connector, judge_connector, simulator_connector)
            if value is not None
        )
        if (
            any(value.kind == "fake" for value in configured_connectors)
            and selection.mode is not RunMode.SMOKE
        ):
            raise ValueError("fixed fake connectors are smoke-only")
        plan = bind_connector_to_plan(
            unbound_plan,
            connector,
            judge_connector=judge_connector,
            simulator_connector=simulator_connector,
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
            judge_model=plan.selection.judge_model,
            simulator_model=plan.selection.simulator_model,
        )
        job_payload = BenchmarkJobPayload(
            benchmark_id=benchmark_id,
            connector=connector,
            judge_connector=judge_connector,
            simulator_connector=simulator_connector,
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


def rebuild_plan_from_job(
    payload: dict,
    *,
    secret_registry: SecretValueRegistry | None = None,
) -> AdapterRunPlan:
    """Recreate and reauthorize a plan from untrusted durable job metadata."""

    context = {"secret_registry": secret_registry} if secret_registry is not None else None
    snapshot = BenchmarkJobPayload.model_validate(payload, context=context)
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
        judge_connector=snapshot.judge_connector,
        simulator_connector=snapshot.simulator_connector,
    )
    if (
        snapshot.protocol.model_dump(mode="json") != plan.protocol.model_dump(mode="json")
        or protocol_hash(snapshot.protocol) != plan.protocol_hash
    ):
        raise ValueError("durable protocol snapshot does not match reconstructed plan")
    if snapshot.selection.model_dump(mode="json") != plan.selection.model_dump(mode="json"):
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
