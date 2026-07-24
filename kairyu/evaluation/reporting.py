"""Deterministic, evidence-only evaluation report rendering."""

from __future__ import annotations

import html as html_module
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from kairyu.evaluation.protocol import protocol_hash
from kairyu.evaluation.schemas import (
    BenchmarkRun,
    Comparability,
    FrozenModel,
    ItemState,
    Metric,
    ProtocolSignature,
    ReferenceResult,
    RunItem,
    RunMode,
    RunState,
    Source,
)

_REPORTABLE_STATES = frozenset(
    {RunState.CANCELLED, RunState.PARTIAL, RunState.COMPLETED, RunState.FAILED}
)
_FORBIDDEN_RAW_CONTENT_KEYS = frozenset(
    {
        "answerchoices",
        "choice",
        "choices",
        "officialproblem",
        "officialquestion",
        "problemstatement",
        "prompt",
        "prompttext",
        "question",
        "questions",
        "rawchoices",
        "rawprompt",
        "rawquestion",
    }
)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_REPRODUCIBILITY_NOTICE = (
    "REPRODUCIBILITY NOTE: Provider APIs may be nondeterministic; storing a seed "
    "does not guarantee identical responses."
)


class UsageEvidence(FrozenModel):
    """Typed aggregate of the usage measurements stored for one run."""

    schema_version: int = 1
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_latency_seconds: float = Field(ge=0, allow_inf_nan=False)
    measurement_status: Literal["complete", "partial"]
    measurement_unavailable_reasons: tuple[str, ...] = ()
    actual_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    actual_cost_unavailable_reason: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
    )

    @model_validator(mode="after")
    def _availability_is_explicit(self) -> UsageEvidence:
        if self.measurement_status == "complete" and self.measurement_unavailable_reasons:
            raise ValueError("complete usage cannot have measurement gaps")
        if self.measurement_status == "partial" and not self.measurement_unavailable_reasons:
            raise ValueError("partial usage requires a measurement gap reason")
        if (self.actual_cost_usd is None) == (self.actual_cost_unavailable_reason is None):
            raise ValueError("actual cost must have either a value or an unavailable reason")
        return self


class StringEvidence(FrozenModel):
    value: str | None = None
    unavailable_reason: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def _availability_is_explicit(self) -> StringEvidence:
        _validate_nullable_evidence(self.value, self.unavailable_reason)
        return self


class IntegerEvidence(FrozenModel):
    value: int | None = Field(default=None, ge=0)
    unavailable_reason: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def _availability_is_explicit(self) -> IntegerEvidence:
        _validate_nullable_evidence(self.value, self.unavailable_reason)
        return self


class SignedIntegerEvidence(FrozenModel):
    value: int | None = None
    unavailable_reason: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def _availability_is_explicit(self) -> SignedIntegerEvidence:
        _validate_nullable_evidence(self.value, self.unavailable_reason)
        return self


class FloatEvidence(FrozenModel):
    value: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    unavailable_reason: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def _availability_is_explicit(self) -> FloatEvidence:
        _validate_nullable_evidence(self.value, self.unavailable_reason)
        return self


class BooleanEvidence(FrozenModel):
    value: bool | None = None
    unavailable_reason: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def _availability_is_explicit(self) -> BooleanEvidence:
        _validate_nullable_evidence(self.value, self.unavailable_reason)
        return self


class ManifestBenchmark(FrozenModel):
    benchmark_version: str
    dataset_id: str
    dataset_revision: str
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestUpstream(FrozenModel):
    repository: StringEvidence
    commit: StringEvidence


class ManifestContainer(FrozenModel):
    image_digest: StringEvidence


class ManifestGeneration(FrozenModel):
    temperature: FloatEvidence
    top_p: FloatEvidence
    max_input_tokens: IntegerEvidence
    max_output_tokens: IntegerEvidence
    seed: SignedIntegerEvidence
    retries: int = Field(ge=0)
    tools: tuple[str, ...]
    scaffold: StringEvidence
    max_turns: IntegerEvidence
    timeout_seconds: FloatEvidence
    reasoning_effort: StringEvidence


class ManifestResources(FrozenModel):
    status: Literal["known", "partial", "unknown"]
    cpu_cores: IntegerEvidence
    ram_bytes: IntegerEvidence
    disk_bytes: IntegerEvidence
    docker_required: BooleanEvidence
    network_policy: StringEvidence


class ManifestEstimates(FrozenModel):
    model_calls: IntegerEvidence
    maximum_model_calls: IntegerEvidence
    estimated_input_tokens: IntegerEvidence
    maximum_output_tokens: IntegerEvidence
    estimated_duration_seconds: FloatEvidence
    maximum_duration_seconds: FloatEvidence
    estimated_cost_usd: FloatEvidence
    assumptions: tuple[str, ...]


class ManifestSoftware(FrozenModel):
    status: Literal["known", "partial", "unknown"]
    harness_name: str
    harness_version: str
    dependency_lock_sha256: StringEvidence
    compatibility_layer_identity: StringEvidence
    compatibility_layer_sha256: StringEvidence
    compatibility_patches: tuple[str, ...]
    runtime_dependency_environment_status: StringEvidence
    runtime_dependency_environment_reason: StringEvidence


class ManifestGit(FrozenModel):
    commit: StringEvidence
    dirty: BooleanEvidence


class RunManifest(FrozenModel):
    """Typed, deterministic run manifest shared by every terminal path."""

    schema_version: int = 1
    run: BenchmarkRun
    benchmark: ManifestBenchmark
    upstream: ManifestUpstream
    container: ManifestContainer
    generation: ManifestGeneration
    resources: ManifestResources
    estimates: ManifestEstimates
    usage: UsageEvidence
    software: ManifestSoftware
    git: ManifestGit
    observed_provider_models: tuple[str, ...] = ()


class ReportError(FrozenModel):
    error_class: str = Field(min_length=1, max_length=128)
    count: int = Field(ge=1)


class ReportInputs(FrozenModel):
    """Stored, normalized inputs accepted by the renderer.

    Raw predictions, official questions, answer choices, and prompt text have no
    field in this contract.  They belong in separately protected execution
    evidence, not in a report renderer.
    """

    run: BenchmarkRun
    protocol: ProtocolSignature
    metrics: tuple[Metric, ...]
    items: tuple[RunItem, ...] = ()
    error_counts: tuple[ReportError, ...] = ()
    usage: UsageEvidence
    sources: tuple[Source, ...]
    references: tuple[ReferenceResult, ...]

    @model_validator(mode="before")
    @classmethod
    def _reject_raw_benchmark_content(cls, value: Any) -> Any:
        _ensure_no_raw_benchmark_content(value)
        return value

    @model_validator(mode="after")
    def _evidence_is_coherent(self) -> ReportInputs:
        run = self.run
        if run.state not in _REPORTABLE_STATES:
            raise ValueError("reports require a completed, cancelled, partial, or failed run")
        if self.protocol.benchmark_id != run.benchmark_id:
            raise ValueError("run and protocol benchmark IDs must match")
        if (
            run.judge_model is not None
            and self.protocol.judge_model is not None
            and run.judge_model != self.protocol.judge_model
        ):
            raise ValueError("run and protocol judge models must match")
        if (
            run.simulator_model is not None
            and self.protocol.simulator_model is not None
            and run.simulator_model != self.protocol.simulator_model
        ):
            raise ValueError("run and protocol simulator models must match")
        computed_protocol_hash = protocol_hash(self.protocol)
        if run.protocol_hash is not None and run.protocol_hash != computed_protocol_hash:
            raise ValueError("stored run protocol hash does not match the protocol")

        if not self.metrics:
            raise ValueError("reports require one primary metric")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("report metric names must be unique")
        if any(metric.run_id != run.run_id for metric in self.metrics):
            raise ValueError("report metrics must belong to the run")
        primary_metrics = [metric for metric in self.metrics if metric.primary]
        if len(primary_metrics) != 1:
            raise ValueError("reports require exactly one primary metric")
        primary = primary_metrics[0]
        _validate_metric(primary)
        if run.completed_count == 0 and primary.value is not None:
            raise ValueError("a run with zero completed items must have a null metric value")

        item_ids = [item.item_id for item in self.items]
        ordinals = [item.ordinal for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("report item IDs must be unique")
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("report item ordinals must be unique")
        selected = set(run.selected_item_ids)
        for item in self.items:
            if item.run_id != run.run_id:
                raise ValueError("report items must belong to the run")
            if selected and item.item_id not in selected:
                raise ValueError("report item is absent from the stored run selection")
            if item.error_class is not None and (
                not item.error_class.strip() or len(item.error_class) > 128
            ):
                raise ValueError("item error classes must be nonblank and bounded")
        _validate_run_counts(run, self.items)
        error_classes = [error.error_class for error in self.error_counts]
        if len(error_classes) != len(set(error_classes)):
            raise ValueError("report error classes must be unique")
        item_error_counts = Counter(
            item.error_class for item in self.items if item.error_class is not None
        )
        stored_error_counts = {error.error_class: error.count for error in self.error_counts}
        if any(
            stored_error_counts.get(error_class, 0) < count
            for error_class, count in item_error_counts.items()
        ):
            raise ValueError("stored error counts omit item error evidence")
        if sum(stored_error_counts.values()) > len(self.items):
            raise ValueError("stored error counts exceed reported item evidence")

        source_by_id: dict[str, Source] = {}
        for source in self.sources:
            if source.source_id in source_by_id:
                raise ValueError("report source IDs must be unique")
            source_by_id[source.source_id] = source
        if not source_by_id:
            raise ValueError("reports require at least one versioned source")

        reference_ids: set[str] = set()
        for reference in self.references:
            if reference.reference_id in reference_ids:
                raise ValueError("report reference IDs must be unique")
            reference_ids.add(reference.reference_id)
            if reference.benchmark_id != run.benchmark_id:
                raise ValueError("report references must match the run benchmark")
            source = source_by_id.get(reference.source_id)
            if source is None:
                raise ValueError("report reference is missing its source")
            if source.source_type is not reference.source_type:
                raise ValueError("report reference and source types must match")
        return self


class ReportProtocol(FrozenModel):
    benchmark_id: str
    benchmark_version: str
    dataset_revision: str
    split: str
    harness_name: str
    harness_version: str
    harness_commit: str | None
    prompt_version: str
    retries: int = Field(ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    reasoning_effort: str | None
    metric_implementation: str
    unresolved_fields: tuple[str, ...] = ()


class ReportCounts(FrozenModel):
    expected_full: int | None
    selected: int
    reported_items: int
    completed: int
    failed: int
    skipped: int
    cancelled: int
    pending: int
    running: int


class ReportMetric(FrozenModel):
    name: str
    display_name: str
    value: float | None
    numerator: int | None
    denominator: int | None
    scale: float
    unit: str
    primary: bool
    higher_is_better: bool
    official_eligible: bool


class ReportItem(FrozenModel):
    item_id: str = Field(min_length=1, max_length=512)
    state: ItemState
    score: float | None = Field(default=None, allow_inf_nan=False)


class ReportReference(FrozenModel):
    reference_id: str
    model_name: str
    score: float = Field(allow_inf_nan=False)
    score_scale: float = Field(gt=0, allow_inf_nan=False)
    metric_name: str
    sample_count: int | None
    source_id: str
    source_type: str
    provider_reported: bool | None
    independently_reproduced: bool
    publication_date: date | None
    retrieved_at: datetime
    comparability: Comparability
    protocol_hash: str | None
    evidence_hash: str
    delta: float | None = Field(default=None, allow_inf_nan=False)
    rank: int | None = Field(default=None, ge=1)


class ReportSource(FrozenModel):
    source_id: str
    source_type: str
    title: str
    url: str
    locator: str
    release_page: str | None
    publication_date: date | None
    retrieved_at: datetime
    evidence_hash: str


class EvaluationReport(FrozenModel):
    """Normalized report document shared by all three renderers."""

    notice: str
    reproducibility_notice: str
    schema_version: int = 2
    report_date: date
    evidence_as_of: datetime
    run_id: str
    benchmark_id: str
    profile: str
    mode: RunMode
    state: RunState
    partial: bool
    unofficial: bool
    target_model: str
    judge_model: str | None
    simulator_model: str | None
    termination_reason: str | None
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol: ReportProtocol
    counts: ReportCounts
    metrics: tuple[ReportMetric, ...]
    usage: UsageEvidence
    errors: tuple[ReportError, ...]
    items: tuple[ReportItem, ...]
    sources: tuple[ReportSource, ...]
    references: tuple[ReportReference, ...]
    comparison_eligible: bool
    rank: int | None = Field(default=None, ge=1)


@dataclass(frozen=True)
class RenderedReports:
    """The same validated report rendered in artifact-ready text formats."""

    report: EvaluationReport
    json: str
    markdown: str
    html: str


def build_run_manifest(
    run: BenchmarkRun,
    protocol: ProtocolSignature,
    usage: UsageEvidence,
    *,
    observed_provider_models: Sequence[str] = (),
) -> RunManifest:
    """Build one typed manifest without consulting mutable ambient state."""

    digest = protocol_hash(protocol)
    if run.protocol_hash != digest:
        raise ValueError("manifest run protocol hash does not match the protocol")
    generation = protocol.generation_parameters
    adapter_configuration = protocol.adapter_configuration
    sandbox = protocol.code_execution_sandbox
    planning_evidence = _mapping_mapping(adapter_configuration, "planning_evidence") or {}
    estimate = _mapping_mapping(planning_evidence, "estimate") or {}
    resources = _mapping_mapping(planning_evidence, "resources") or {}
    execution = _mapping_mapping(planning_evidence, "execution") or {}

    repository = _mapping_string(adapter_configuration, "upstream_repository")
    container_digest = _first_mapping_string(
        execution,
        "container_image_digest",
        "container_digest",
    ) or _first_mapping_string(
        sandbox,
        "container_image_digest",
        "container_digest",
    )
    temperature = _mapping_float(generation, "temperature")
    top_p = _mapping_float(generation, "top_p")
    max_input_tokens = _mapping_integer(generation, "max_input_tokens")
    max_output_tokens = _mapping_integer(generation, "max_output_tokens")
    if max_output_tokens is None:
        max_output_tokens = _mapping_integer(generation, "max_tokens")
    seed = _mapping_signed_integer(generation, "seed")
    timeout_seconds = protocol.timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = _mapping_float(generation, "timeout_seconds")

    cpu_cores = _mapping_integer(resources, "cpu_cores")
    if cpu_cores is None:
        cpu_cores = _mapping_integer(sandbox, "cpu_cores")
    ram_bytes = _mapping_integer(resources, "ram_bytes")
    if ram_bytes is None:
        ram_bytes = _mapping_integer(sandbox, "ram_bytes")
    disk_bytes = _mapping_integer(resources, "disk_bytes")
    docker_required = _mapping_boolean(resources, "docker_required")
    network_policy = _mapping_string(resources, "network_policy")
    resource_values = (
        cpu_cores,
        ram_bytes,
        disk_bytes,
        docker_required,
        network_policy,
    )
    known_resources = sum(value is not None for value in resource_values)
    if known_resources == len(resource_values):
        resource_status: Literal["known", "partial", "unknown"] = "known"
    elif known_resources:
        resource_status = "partial"
    else:
        resource_status = "unknown"

    estimate_reason = (
        "run plan records this estimate as unknown; see estimates.assumptions"
        if estimate
        else "protocol does not record run planning evidence"
    )
    resource_reason = "protocol planning evidence does not record this resource requirement"
    container_reason = (
        "run plan does not require a container image"
        if execution
        else "protocol does not record a container image digest"
    )
    git_reason = "run evidence does not capture the Kairyu repository state"

    dependency_lock = protocol.dependency_lock_sha256
    compatibility_sha256 = _mapping_string(
        adapter_configuration,
        "compatibility_layer_sha256",
    )
    compatibility_identity, patch_sha256 = _compatibility_layer_identity(
        protocol.dependency_compatibility_patches,
        compatibility_sha256,
    )
    if compatibility_sha256 is None:
        compatibility_sha256 = patch_sha256
    reproducibility = _mapping_mapping(adapter_configuration, "reproducibility_evidence") or {}
    runtime_dependency = _mapping_mapping(reproducibility, "runtime_dependency_environment") or {}
    runtime_dependency_status = _mapping_string(runtime_dependency, "status")
    runtime_dependency_reason = _mapping_string(runtime_dependency, "reason")
    software_components = (
        dependency_lock,
        compatibility_identity,
        compatibility_sha256,
        runtime_dependency_status,
    )
    if all(value is not None for value in software_components) and (
        runtime_dependency_status == "verified"
    ):
        software_status: Literal["known", "partial", "unknown"] = "known"
    elif any(value is not None for value in software_components):
        software_status = "partial"
    else:
        software_status = "unknown"

    return RunManifest(
        run=run,
        benchmark=ManifestBenchmark(
            benchmark_version=protocol.benchmark_version,
            dataset_id=protocol.dataset_id,
            dataset_revision=protocol.dataset_revision,
            protocol_hash=digest,
        ),
        upstream=ManifestUpstream(
            repository=_string_evidence(
                repository,
                "protocol does not record an upstream repository",
            ),
            commit=_string_evidence(
                protocol.harness_commit,
                "protocol does not record an upstream commit",
            ),
        ),
        container=ManifestContainer(
            image_digest=_string_evidence(container_digest, container_reason)
        ),
        generation=ManifestGeneration(
            temperature=_float_evidence(
                temperature,
                "protocol does not record generation temperature",
            ),
            top_p=_float_evidence(top_p, "protocol does not record generation top_p"),
            max_input_tokens=_integer_evidence(
                max_input_tokens,
                "protocol does not record a maximum input token count",
            ),
            max_output_tokens=_integer_evidence(
                max_output_tokens,
                "protocol does not record a maximum output token count",
            ),
            seed=_signed_integer_evidence(
                seed,
                "protocol does not record a generation seed",
            ),
            retries=protocol.retries,
            tools=protocol.tools,
            scaffold=_string_evidence(
                protocol.agent_scaffold,
                "protocol does not record an agent scaffold",
            ),
            max_turns=_integer_evidence(
                protocol.max_turns,
                "protocol does not record a maximum turn count",
            ),
            timeout_seconds=_float_evidence(
                timeout_seconds,
                "protocol does not record a timeout",
            ),
            reasoning_effort=_string_evidence(
                protocol.reasoning_effort,
                "protocol does not record reasoning effort",
            ),
        ),
        resources=ManifestResources(
            status=resource_status,
            cpu_cores=_integer_evidence(cpu_cores, resource_reason),
            ram_bytes=_integer_evidence(ram_bytes, resource_reason),
            disk_bytes=_integer_evidence(disk_bytes, resource_reason),
            docker_required=_boolean_evidence(docker_required, resource_reason),
            network_policy=_string_evidence(network_policy, resource_reason),
        ),
        estimates=ManifestEstimates(
            model_calls=_integer_evidence(
                _mapping_integer(estimate, "model_calls"),
                estimate_reason,
            ),
            maximum_model_calls=_integer_evidence(
                _mapping_integer(estimate, "maximum_model_calls"),
                estimate_reason,
            ),
            estimated_input_tokens=_integer_evidence(
                _mapping_integer(estimate, "estimated_input_tokens"),
                estimate_reason,
            ),
            maximum_output_tokens=_integer_evidence(
                _mapping_integer(estimate, "maximum_output_tokens"),
                estimate_reason,
            ),
            estimated_duration_seconds=_float_evidence(
                _mapping_float(estimate, "estimated_duration_seconds"),
                estimate_reason,
            ),
            maximum_duration_seconds=_float_evidence(
                _mapping_float(estimate, "maximum_duration_seconds"),
                estimate_reason,
            ),
            estimated_cost_usd=_float_evidence(
                _mapping_float(estimate, "estimated_cost_usd"),
                estimate_reason,
            ),
            assumptions=_mapping_string_tuple(estimate, "assumptions"),
        ),
        usage=usage,
        software=ManifestSoftware(
            status=software_status,
            harness_name=protocol.harness_name,
            harness_version=protocol.harness_version,
            dependency_lock_sha256=_string_evidence(
                dependency_lock,
                "protocol does not record a dependency lock digest",
            ),
            compatibility_layer_identity=_string_evidence(
                compatibility_identity,
                "protocol does not record a compatibility layer identity",
            ),
            compatibility_layer_sha256=_string_evidence(
                compatibility_sha256,
                "protocol does not record a compatibility layer digest",
            ),
            compatibility_patches=protocol.dependency_compatibility_patches,
            runtime_dependency_environment_status=_string_evidence(
                runtime_dependency_status,
                "protocol does not attest the runtime dependency environment",
            ),
            runtime_dependency_environment_reason=_string_evidence(
                runtime_dependency_reason,
                "protocol does not explain runtime dependency attestation status",
            ),
        ),
        git=ManifestGit(
            commit=_string_evidence(None, git_reason),
            dirty=BooleanEvidence(value=None, unavailable_reason=git_reason),
        ),
        observed_provider_models=tuple(sorted(set(observed_provider_models))),
    )


def build_report(inputs: ReportInputs) -> EvaluationReport:
    """Build a normalized report solely from stored input models."""

    primary = next(metric for metric in inputs.metrics if metric.primary)
    protocol_digest = protocol_hash(inputs.protocol)
    comparison_base, notice, unofficial = _comparison_policy(inputs.run, primary)
    compatible = tuple(
        reference for reference in inputs.references if _reference_is_comparable(reference, primary)
    )
    comparison_eligible = comparison_base and primary.value is not None and bool(compatible)
    if comparison_base and not compatible:
        notice = (
            "COMPARISON UNAVAILABLE: all stored reference protocols are incompatible; "
            "deltas and ranks are not reported."
        )

    target_rank = (
        _rank(
            primary.value,
            [reference.score / reference.score_scale * primary.scale for reference in compatible],
            primary,
        )
        if comparison_eligible and primary.value is not None
        else None
    )
    reference_summaries = tuple(
        _summarize_reference(
            reference,
            primary,
            compatible,
            comparison_eligible=comparison_eligible,
        )
        for reference in inputs.references
    )
    ordered_items = tuple(sorted(inputs.items, key=lambda item: item.ordinal))
    finished_at = inputs.run.finished_at
    if finished_at is None:  # guarded by BenchmarkRun terminal-state validation
        raise ValueError("reportable run is missing its deterministic evidence date")

    return EvaluationReport(
        notice=notice,
        reproducibility_notice=_REPRODUCIBILITY_NOTICE,
        report_date=finished_at.date(),
        evidence_as_of=finished_at,
        run_id=inputs.run.run_id,
        benchmark_id=inputs.run.benchmark_id,
        profile=inputs.run.profile,
        mode=inputs.run.mode,
        state=inputs.run.state,
        partial=inputs.run.partial,
        unofficial=unofficial,
        target_model=inputs.run.target_model,
        judge_model=inputs.run.judge_model or inputs.protocol.judge_model,
        simulator_model=inputs.run.simulator_model or inputs.protocol.simulator_model,
        termination_reason=inputs.run.termination_reason,
        protocol_hash=protocol_digest,
        protocol=ReportProtocol(
            benchmark_id=inputs.protocol.benchmark_id,
            benchmark_version=inputs.protocol.benchmark_version,
            dataset_revision=inputs.protocol.dataset_revision,
            split=inputs.protocol.split,
            harness_name=inputs.protocol.harness_name,
            harness_version=inputs.protocol.harness_version,
            harness_commit=inputs.protocol.harness_commit,
            prompt_version=inputs.protocol.prompt_version,
            retries=inputs.protocol.retries,
            timeout_seconds=(
                inputs.protocol.timeout_seconds
                if inputs.protocol.timeout_seconds is not None
                else _mapping_float(
                    inputs.protocol.generation_parameters,
                    "timeout_seconds",
                )
            ),
            reasoning_effort=inputs.protocol.reasoning_effort,
            metric_implementation=inputs.protocol.metric_implementation,
            unresolved_fields=inputs.protocol.unresolved_fields,
        ),
        counts=_counts(inputs.run, inputs.items),
        metrics=tuple(_summarize_metric(metric) for metric in inputs.metrics),
        usage=inputs.usage,
        errors=tuple(sorted(inputs.error_counts, key=lambda error: error.error_class)),
        items=tuple(
            ReportItem(
                item_id=item.item_id,
                state=item.state,
                score=item.scores.get(primary.name),
            )
            for item in ordered_items
        ),
        sources=tuple(
            _summarize_source(source)
            for source in sorted(inputs.sources, key=lambda source: source.source_id)
        ),
        references=reference_summaries,
        comparison_eligible=comparison_eligible,
        rank=target_rank,
    )


def render_report(inputs: ReportInputs) -> RenderedReports:
    """Build and render deterministic JSON, Markdown, and self-contained HTML."""

    report = build_report(inputs)
    return RenderedReports(
        report=report,
        json=render_json(report),
        markdown=render_markdown(report),
        html=render_html(report),
    )


def render_json(report: EvaluationReport) -> str:
    return (
        json.dumps(
            report.model_dump(mode="json", exclude_none=False),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            indent=2,
        )
        + "\n"
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        f"> **{_md(report.notice)}**",
        "",
        f"> **{_md(report.reproducibility_notice)}**",
        "",
        "# Evaluation Report",
        "",
        f"- Run: `{_md(report.run_id)}`",
        f"- Benchmark: `{_md(report.benchmark_id)}`",
        f"- Profile / mode: `{_md(report.profile)}` / `{report.mode.value}`",
        f"- State: `{report.state.value}` (partial: `{str(report.partial).lower()}`)",
        f"- Target model: `{_md(report.target_model)}`",
        f"- Judge model: `{_md(report.judge_model)}`",
        f"- Simulator model: `{_md(report.simulator_model)}`",
        f"- Report date: `{report.report_date.isoformat()}`",
        f"- Evidence as of: `{report.evidence_as_of.isoformat()}`",
        f"- Protocol SHA-256: `{report.protocol_hash}`",
        f"- Comparison eligible: `{str(report.comparison_eligible).lower()}`",
        f"- Rank: `{_display(report.rank)}`",
        "",
        "## Counts",
        "",
        "| Expected full | Selected | Reported | Completed | Failed | Skipped | "
        "Cancelled | Pending | Running |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(_display(value) for value in report.counts.model_dump(mode="python").values())
        + " |",
        "",
        "## Protocol Details",
        "",
        "| Harness | Version | Upstream commit | Retries | Timeout (seconds) | Reasoning effort |",
        "|---|---|---|---:|---:|---|",
        "| "
        + " | ".join(
            (
                _md(report.protocol.harness_name),
                _md(report.protocol.harness_version),
                _md(report.protocol.harness_commit),
                _display(report.protocol.retries),
                _display(report.protocol.timeout_seconds),
                _md(report.protocol.reasoning_effort),
            )
        )
        + " |",
        "",
        "## Usage / Cost",
        "",
        "| Input tokens | Output tokens | Total latency (seconds) | "
        "Measurement status | Actual cost (USD) | Cost unavailable reason |",
        "|---:|---:|---:|---|---:|---|",
        "| "
        + " | ".join(
            (
                _display(report.usage.input_tokens),
                _display(report.usage.output_tokens),
                _display(report.usage.total_latency_seconds),
                report.usage.measurement_status,
                _display(report.usage.actual_cost_usd),
                _md(report.usage.actual_cost_unavailable_reason),
            )
        )
        + " |",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Numerator | Denominator | Scale | Unit | Primary | Official |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            (
                _md(metric.display_name),
                _display(metric.value),
                _display(metric.numerator),
                _display(metric.denominator),
                _display(metric.scale),
                _md(metric.unit),
                str(metric.primary).lower(),
                str(metric.official_eligible).lower(),
            )
        )
        + " |"
        for metric in report.metrics
    )
    lines.extend(_markdown_errors(report))
    lines.extend(_markdown_items(report))
    lines.extend(_markdown_sources(report))
    lines.extend(_markdown_references(report))
    return "\n".join(lines) + "\n"


def render_html(report: EvaluationReport) -> str:
    counts = "".join(
        f"<td>{_html(value)}</td>" for value in report.counts.model_dump(mode="python").values()
    )
    metrics = "".join(
        "<tr>"
        f"<td>{_html(metric.display_name)}</td>"
        f"<td>{_html(metric.value)}</td>"
        f"<td>{_html(metric.numerator)}</td>"
        f"<td>{_html(metric.denominator)}</td>"
        f"<td>{_html(metric.scale)}</td>"
        f"<td>{_html(metric.unit)}</td>"
        f"<td>{_html(metric.primary)}</td>"
        f"<td>{_html(metric.official_eligible)}</td>"
        "</tr>"
        for metric in report.metrics
    )
    errors = (
        "".join(
            f"<tr><td>{_html(error.error_class)}</td><td>{error.count}</td></tr>"
            for error in report.errors
        )
        or '<tr><td colspan="2">None</td></tr>'
    )
    items = (
        "".join(
            "<tr>"
            f"<td>{_html(item.item_id)}</td>"
            f"<td>{_html(item.state.value)}</td>"
            f"<td>{_html(item.score)}</td>"
            "</tr>"
            for item in report.items
        )
        or '<tr><td colspan="3">No item evidence</td></tr>'
    )
    sources = "".join(
        "<tr>"
        f"<td>{_html(source.source_id)}</td>"
        f"<td>{_html(source.source_type)}</td>"
        f"<td>{_html(source.title)}</td>"
        f"<td>{_html(source.url)}</td>"
        f"<td>{_html(source.locator)}</td>"
        f"<td>{_html(source.publication_date)}</td>"
        f"<td>{_html(source.retrieved_at)}</td>"
        f"<td><code>{source.evidence_hash}</code></td>"
        "</tr>"
        for source in report.sources
    )
    references = (
        "".join(
            "<tr>"
            f"<td>{_html(reference.model_name)}</td>"
            f"<td>{_html(reference.score)}</td>"
            f"<td>{_html(reference.score_scale)}</td>"
            f"<td>{_html(reference.source_id)}</td>"
            f"<td>{_html(reference.publication_date)}</td>"
            f"<td>{_html(reference.retrieved_at)}</td>"
            f"<td>{_html(reference.comparability.value)}</td>"
            f"<td>{_html(reference.protocol_hash)}</td>"
            f"<td>{_html(reference.delta)}</td>"
            f"<td>{_html(reference.rank)}</td>"
            "</tr>"
            for reference in report.references
        )
        or '<tr><td colspan="10">No references</td></tr>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;
color:#172033}}.notice{{border:2px solid #9a6700;background:#fff8c5;padding:1rem;
font-weight:700}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
th,td{{border:1px solid #d0d7de;padding:.45rem;text-align:left;vertical-align:top}}
th{{background:#f6f8fa}}code{{overflow-wrap:anywhere}}dl{{display:grid;
grid-template-columns:max-content 1fr;gap:.4rem 1rem}}dt{{font-weight:700}}
</style>
</head>
<body>
<div class="notice">{_html(report.notice)}</div>
<div class="notice">{_html(report.reproducibility_notice)}</div>
<h1>Evaluation Report</h1>
<dl>
<dt>Run</dt><dd><code>{_html(report.run_id)}</code></dd>
<dt>Benchmark</dt><dd><code>{_html(report.benchmark_id)}</code></dd>
<dt>Profile / mode</dt><dd>{_html(report.profile)} / {_html(report.mode.value)}</dd>
<dt>State</dt><dd>{_html(report.state.value)}; partial={_html(report.partial)}</dd>
<dt>Target model</dt><dd>{_html(report.target_model)}</dd>
<dt>Judge model</dt><dd>{_html(report.judge_model)}</dd>
<dt>Simulator model</dt><dd>{_html(report.simulator_model)}</dd>
<dt>Report date</dt><dd>{report.report_date.isoformat()}</dd>
<dt>Evidence as of</dt><dd>{_html(report.evidence_as_of)}</dd>
<dt>Protocol SHA-256</dt><dd><code>{report.protocol_hash}</code></dd>
<dt>Comparison eligible</dt><dd>{_html(report.comparison_eligible)}</dd>
<dt>Rank</dt><dd>{_html(report.rank)}</dd>
</dl>
<h2>Counts</h2>
<table><thead><tr><th>Expected full</th><th>Selected</th><th>Reported</th>
<th>Completed</th><th>Failed</th><th>Skipped</th><th>Cancelled</th>
<th>Pending</th><th>Running</th></tr></thead><tbody><tr>{counts}</tr></tbody></table>
<h2>Protocol Details</h2>
<dl>
<dt>Harness</dt><dd>{_html(report.protocol.harness_name)}</dd>
<dt>Harness version</dt><dd>{_html(report.protocol.harness_version)}</dd>
<dt>Upstream commit</dt><dd>{_html(report.protocol.harness_commit)}</dd>
<dt>Retries</dt><dd>{_html(report.protocol.retries)}</dd>
<dt>Timeout (seconds)</dt><dd>{_html(report.protocol.timeout_seconds)}</dd>
<dt>Reasoning effort</dt><dd>{_html(report.protocol.reasoning_effort)}</dd>
</dl>
<h2>Usage / Cost</h2>
<dl>
<dt>Input tokens</dt><dd>{_html(report.usage.input_tokens)}</dd>
<dt>Output tokens</dt><dd>{_html(report.usage.output_tokens)}</dd>
<dt>Total latency (seconds)</dt><dd>{_html(report.usage.total_latency_seconds)}</dd>
<dt>Measurement status</dt><dd>{_html(report.usage.measurement_status)}</dd>
<dt>Actual cost (USD)</dt><dd>{_html(report.usage.actual_cost_usd)}</dd>
<dt>Cost unavailable reason</dt>
<dd>{_html(report.usage.actual_cost_unavailable_reason)}</dd>
</dl>
<h2>Metrics</h2>
<table><thead><tr><th>Metric</th><th>Value</th><th>Numerator</th><th>Denominator</th>
<th>Scale</th><th>Unit</th><th>Primary</th><th>Official</th></tr></thead>
<tbody>{metrics}</tbody></table>
<h2>Errors</h2><table><thead><tr><th>Class</th><th>Count</th></tr></thead>
<tbody>{errors}</tbody></table>
<h2>Items</h2><table><thead><tr><th>ID</th><th>State</th><th>Score</th></tr></thead>
<tbody>{items}</tbody></table>
<h2>Sources</h2><table><thead><tr><th>ID</th><th>Type</th><th>Title</th><th>URL</th>
<th>Locator</th><th>Publication date</th><th>Retrieved at</th><th>Evidence hash</th></tr>
</thead><tbody>{sources}</tbody></table>
<h2>References</h2><table><thead><tr><th>Model</th><th>Score</th><th>Scale</th>
<th>Source</th><th>Publication date</th><th>Retrieved at</th><th>Comparability</th>
<th>Protocol hash</th><th>Delta</th><th>Rank</th></tr></thead>
<tbody>{references}</tbody></table>
</body>
</html>
"""


def _validate_nullable_evidence(value: object | None, reason: str | None) -> None:
    if (value is None) == (reason is None):
        raise ValueError("evidence must have either a value or an unavailable reason")


def _string_evidence(value: str | None, reason: str) -> StringEvidence:
    return StringEvidence(
        value=value,
        unavailable_reason=reason if value is None else None,
    )


def _integer_evidence(value: int | None, reason: str) -> IntegerEvidence:
    return IntegerEvidence(
        value=value,
        unavailable_reason=reason if value is None else None,
    )


def _signed_integer_evidence(value: int | None, reason: str) -> SignedIntegerEvidence:
    return SignedIntegerEvidence(
        value=value,
        unavailable_reason=reason if value is None else None,
    )


def _float_evidence(value: float | None, reason: str) -> FloatEvidence:
    return FloatEvidence(
        value=value,
        unavailable_reason=reason if value is None else None,
    )


def _boolean_evidence(value: bool | None, reason: str) -> BooleanEvidence:
    return BooleanEvidence(
        value=value,
        unavailable_reason=reason if value is None else None,
    )


def _mapping_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _mapping_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object] | None:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else None


def _mapping_string_tuple(
    mapping: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _first_mapping_string(mapping: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = _mapping_string(mapping, key)
        if value is not None:
            return value
    return None


def _mapping_float(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping_integer(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _mapping_signed_integer(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mapping_boolean(mapping: Mapping[str, object], key: str) -> bool | None:
    value = mapping.get(key)
    return value if isinstance(value, bool) else None


def _compatibility_layer_identity(
    patches: tuple[str, ...],
    expected_sha256: str | None,
) -> tuple[str | None, str | None]:
    for patch in patches:
        identity, marker, digest = patch.partition("@sha256:")
        if (
            marker
            and identity
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            and (expected_sha256 is None or digest == expected_sha256)
        ):
            return identity, digest
    return None, None


def _validate_metric(metric: Metric) -> None:
    numerator = metric.numerator
    denominator = metric.denominator
    if (numerator is None) != (denominator is None):
        raise ValueError("metric numerator and denominator must be stored together")
    if denominator == 0:
        if numerator not in (None, 0) or metric.value is not None:
            raise ValueError("zero-denominator metrics must have a null value")
        return
    if denominator is not None:
        if numerator is None or numerator > denominator:
            raise ValueError("metric numerator exceeds its denominator")
        expected = numerator / denominator * metric.scale
        if metric.value is None or abs(metric.value - expected) > 1e-9:
            raise ValueError("metric value does not match its stored counts and scale")


def _validate_run_counts(run: BenchmarkRun, items: tuple[RunItem, ...]) -> None:
    counts = Counter(item.state for item in items)
    if counts[ItemState.COMPLETED] != run.completed_count:
        raise ValueError("completed item evidence does not match the run count")
    if counts[ItemState.FAILED] != run.failed_count:
        raise ValueError("failed item evidence does not match the run count")
    if counts[ItemState.SKIPPED] != run.skipped_count:
        raise ValueError("skipped item evidence does not match the run count")


def _counts(run: BenchmarkRun, items: tuple[RunItem, ...]) -> ReportCounts:
    states = Counter(item.state for item in items)
    return ReportCounts(
        expected_full=run.expected_full_count,
        selected=len(run.selected_item_ids),
        reported_items=len(items),
        completed=run.completed_count,
        failed=run.failed_count,
        skipped=run.skipped_count,
        cancelled=states[ItemState.CANCELLED],
        pending=states[ItemState.PENDING],
        running=states[ItemState.RUNNING],
    )


def _summarize_metric(metric: Metric) -> ReportMetric:
    return ReportMetric(
        name=metric.name,
        display_name=metric.display_name,
        value=metric.value,
        numerator=metric.numerator,
        denominator=metric.denominator,
        scale=metric.scale,
        unit=metric.unit,
        primary=metric.primary,
        higher_is_better=metric.higher_is_better,
        official_eligible=metric.official_eligible,
    )


def _summarize_source(source: Source) -> ReportSource:
    return ReportSource(
        source_id=source.source_id,
        source_type=source.source_type.value,
        title=source.title,
        url=source.url,
        locator=source.locator,
        release_page=source.release_page,
        publication_date=source.publication_date,
        retrieved_at=source.retrieved_at,
        evidence_hash=source.evidence_hash,
    )


def _comparison_policy(run: BenchmarkRun, metric: Metric) -> tuple[bool, str, bool]:
    reasons: list[str] = []
    if run.mode is not RunMode.FULL:
        reasons.append(f"{run.mode.value} mode")
    if run.state is not RunState.COMPLETED:
        reasons.append(f"{run.state.value} state")
    if run.partial:
        reasons.append("partial evidence")
    if not metric.official_eligible:
        reasons.append("metric is not official-eligible")
    if reasons:
        reason = ", ".join(reasons)
        return (
            False,
            "UNOFFICIAL: this report contains "
            f"{reason}; it is not full-suite accuracy and cannot be used for "
            "published-score deltas or ranking.",
            True,
        )
    return True, "OFFICIAL-ELIGIBLE: stored protocol evidence permits comparison.", False


def _reference_is_comparable(reference: ReferenceResult, metric: Metric) -> bool:
    return (
        reference.comparability in {Comparability.EXACT, Comparability.NEAR}
        and reference.protocol_hash is not None
        and reference.metric_name.casefold() == metric.name.casefold()
    )


def _summarize_reference(
    reference: ReferenceResult,
    primary: Metric,
    compatible: tuple[ReferenceResult, ...],
    *,
    comparison_eligible: bool,
) -> ReportReference:
    is_compatible = comparison_eligible and reference in compatible
    scaled_score = reference.score / reference.score_scale * primary.scale
    delta = (
        round(primary.value - scaled_score, 12)
        if is_compatible and primary.value is not None
        else None
    )
    rank = (
        _rank(
            scaled_score,
            [other.score / other.score_scale * primary.scale for other in compatible]
            + ([primary.value] if primary.value is not None else []),
            primary,
        )
        if is_compatible
        else None
    )
    return ReportReference(
        reference_id=reference.reference_id,
        model_name=reference.model_name,
        score=reference.score,
        score_scale=reference.score_scale,
        metric_name=reference.metric_name,
        sample_count=reference.sample_count,
        source_id=reference.source_id,
        source_type=reference.source_type.value,
        provider_reported=reference.provider_reported,
        independently_reproduced=reference.independently_reproduced,
        publication_date=reference.publication_date,
        retrieved_at=reference.retrieved_at,
        comparability=reference.comparability,
        protocol_hash=reference.protocol_hash,
        evidence_hash=reference.evidence_hash,
        delta=delta,
        rank=rank,
    )


def _rank(value: float, population: Sequence[float], metric: Metric) -> int:
    if metric.higher_is_better:
        return 1 + sum(candidate > value for candidate in population)
    return 1 + sum(candidate < value for candidate in population)


def _ensure_no_raw_benchmark_content(value: Any) -> None:
    stack = [value]
    seen: set[int] = set()
    # Keep visited containers alive.  ``model_dump`` creates temporary dicts;
    # retaining them prevents CPython from reusing an ID and accidentally
    # treating a later nested mapping as already visited.
    retained: list[object] = []
    while stack:
        current = stack.pop()
        if isinstance(current, BaseModel):
            current = current.model_dump(mode="python")
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            retained.append(current)
            for key, child in current.items():
                if isinstance(key, str):
                    normalized = _NON_ALPHANUMERIC.sub("", key.casefold())
                    if normalized in _FORBIDDEN_RAW_CONTENT_KEYS:
                        raise ValueError(
                            "raw benchmark questions, choices, and prompts are forbidden"
                        )
                stack.append(child)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            retained.append(current)
            stack.extend(current)


def _display(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _md(value: object) -> str:
    text = _display(value).replace("\n", " ").replace("\r", " ")
    for character in "\\`*_|[]()<>#":
        text = text.replace(character, "\\" + character)
    return text


def _html(value: object) -> str:
    return html_module.escape(_display(value), quote=True)


def _markdown_errors(report: EvaluationReport) -> list[str]:
    lines = ["", "## Errors", "", "| Class | Count |", "|---|---:|"]
    if report.errors:
        lines.extend(f"| {_md(error.error_class)} | {error.count} |" for error in report.errors)
    else:
        lines.append("| None | 0 |")
    return lines


def _markdown_items(report: EvaluationReport) -> list[str]:
    lines = ["", "## Items", "", "| ID | State | Score |", "|---|---|---:|"]
    if report.items:
        lines.extend(
            f"| {_md(item.item_id)} | {item.state.value} | {_display(item.score)} |"
            for item in report.items
        )
    else:
        lines.append("| No item evidence | null | null |")
    return lines


def _markdown_sources(report: EvaluationReport) -> list[str]:
    lines = [
        "",
        "## Sources",
        "",
        "| ID | Type | Title | URL | Locator | Publication date | Retrieved at | Evidence hash |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            (
                _md(source.source_id),
                source.source_type,
                _md(source.title),
                _md(source.url),
                _md(source.locator),
                _md(source.publication_date),
                _md(source.retrieved_at),
                source.evidence_hash,
            )
        )
        + " |"
        for source in report.sources
    )
    return lines


def _markdown_references(report: EvaluationReport) -> list[str]:
    lines = [
        "",
        "## References",
        "",
        "| Model | Score | Scale | Source | Publication date | Retrieved at | "
        "Comparability | Protocol hash | Delta | Rank |",
        "|---|---:|---:|---|---|---|---|---|---:|---:|",
    ]
    if report.references:
        lines.extend(
            "| "
            + " | ".join(
                (
                    _md(reference.model_name),
                    _display(reference.score),
                    _display(reference.score_scale),
                    _md(reference.source_id),
                    _md(reference.publication_date),
                    _md(reference.retrieved_at),
                    reference.comparability.value,
                    _display(reference.protocol_hash),
                    _display(reference.delta),
                    _display(reference.rank),
                )
            )
            + " |"
            for reference in report.references
        )
    else:
        lines.append(
            "| None | null | null | null | null | null | incompatible | null | null | null |"
        )
    return lines


__all__ = [
    "EvaluationReport",
    "RenderedReports",
    "ReportInputs",
    "build_report",
    "render_html",
    "render_json",
    "render_markdown",
    "render_report",
]
