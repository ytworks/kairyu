"""Humanity's Last Exam adapter pinned to the reviewed CAIS simple-evals harness."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kairyu.evaluation.adapters.base import (
    AdapterItem,
    AdapterRunPlan,
    BenchmarkAdapter,
    CheckStatus,
    CollectedResult,
    DoctorCheck,
    DoctorReport,
    ExecutionSpec,
    ItemResult,
    ModelConnectorSet,
    ModelRole,
    ModelRolePlan,
    PreparationResult,
    PreparationStatus,
    ResourceRequirements,
    RunEstimate,
    RunSelection,
)
from kairyu.evaluation.adapters.hle_official_2026 import (
    MAX_ATTEMPTS,
    PROMPT_BUNDLE_SHA256,
    SYSTEM_PROMPT,
    PreparedHLERecord,
    ScoredPrediction,
    build_judge_prompt,
    compute_metrics,
    load_records_with_sha256,
    manifest_sha256,
    parse_judge_response,
    prepare_records,
    sha256_file,
)
from kairyu.evaluation.connectors import (
    ConnectorError,
    ConnectorErrorCode,
    ConnectorImagePart,
    ConnectorImageURL,
    ConnectorMessage,
    ConnectorResponse,
    ConnectorResult,
    ConnectorTextPart,
    ConnectorUsage,
    ModelConnector,
    ModelRequest,
    canonical_connector_request_sha256,
)
from kairyu.evaluation.guards import validate_run_guard
from kairyu.evaluation.profiles import ProfileLock, get_profile_lock, load_profiles
from kairyu.evaluation.protocol import protocol_hash
from kairyu.evaluation.schemas import (
    BenchmarkDefinition,
    BenchmarkProfile,
    ImplementationStatus,
    Metric,
    ProtocolSignature,
    RunMode,
)

_BENCHMARK_ID = "humanitys-last-exam"
_BENCHMARK_VERSION = "humanitys-last-exam-cais-simple-evals-2026.07"
_EXPECTED_OFFICIAL_COUNT = 2_500
_MIN_RAM_BYTES = 2 * 1024 * 1024 * 1024
_MIN_DISK_BYTES = 1024 * 1024 * 1024
_SMOKE_RESOURCE = (
    Path(__file__).parents[1] / "resources" / "fixtures" / "humanitys-last-exam-smoke.jsonl"
)
_COMPATIBILITY_LAYER_RESOURCE = Path(__file__).with_name("hle_official_2026.py")


@dataclass(slots=True)
class _AttemptEvidence:
    attempts: int = 0
    budget_used: int = 0
    latency_seconds: float = 0.0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    has_usage: bool = False
    last_response_provider_model: str | None = None

    def consume(
        self,
        outcome: ConnectorResult,
        *,
        remaining_attempts: int,
    ) -> None:
        result = outcome.response if outcome.response is not None else outcome.error
        assert result is not None
        reported_attempts = result.attempts
        cancelled_without_attempt = (
            outcome.error is not None
            and outcome.error.code is ConnectorErrorCode.CANCELLED
            and reported_attempts == 0
        )
        consumed_attempts = 0 if cancelled_without_attempt else max(1, reported_attempts)
        if consumed_attempts > remaining_attempts:
            raise ValueError("HLE connector exceeded the remaining attempt budget")
        self.attempts += reported_attempts
        self.budget_used += consumed_attempts
        self.latency_seconds += result.latency_seconds
        if outcome.response is not None and outcome.response.provider_model is not None:
            self.last_response_provider_model = outcome.response.provider_model
        if outcome.response is not None and outcome.response.usage is not None:
            self.has_usage = True
            self.input_tokens_total += outcome.response.usage.prompt_tokens
            self.output_tokens_total += outcome.response.usage.completion_tokens

    @property
    def input_tokens(self) -> int | None:
        return self.input_tokens_total if self.has_usage else None

    @property
    def output_tokens(self) -> int | None:
        return self.output_tokens_total if self.has_usage else None


class HumanitysLastExamAdapter(BenchmarkAdapter):
    """Offline-first multimodal HLE target-and-judge runner."""

    def metadata(self) -> BenchmarkDefinition:
        return BenchmarkDefinition(
            benchmark_id=_BENCHMARK_ID,
            display_name="Humanity's Last Exam",
            description=(
                "The gated 2,500-item CAIS expert benchmark with text and image "
                "questions, an LLM judge, accuracy, calibration error, and Wald intervals."
            ),
            benchmark_version=_BENCHMARK_VERSION,
            licenses=(
                "CAIS HLE dataset: MIT with original-provider access conditions",
                "CAIS simple-evals harness: MIT",
                "Kairyu synthetic smoke fixture: CC0-1.0",
            ),
            data_sources=(
                "https://huggingface.co/datasets/cais/hle",
                "https://github.com/centerforaisafety/simple-evals/tree/"
                "8e53435ff2985b0f32ea7ceb7e92c3a175f2c0f3/hle",
                "https://arxiv.org/abs/2501.14249",
            ),
            required_auth=("manually approved local CAIS HLE snapshot",),
            primary_metric="Accuracy",
            auxiliary_metrics=(
                "Calibration Error",
                "Success-only Accuracy",
                "95% Wald confidence interval half-width",
                "judge and API error counts",
            ),
            higher_is_better=True,
            modalities=("text", "image"),
            required_capabilities=(
                "chat completions",
                "inline image input",
                "separate judge connector",
            ),
            supports_resume=True,
            implementation_status=ImplementationStatus.AVAILABLE,
        )

    def profiles(self) -> tuple[BenchmarkProfile, ...]:
        return load_profiles(_BENCHMARK_ID)

    def doctor(
        self,
        profile: str,
        *,
        dataset_path: Path | None = None,
    ) -> DoctorReport:
        lock = get_profile_lock(_BENCHMARK_ID, profile)
        cpu_count = os.cpu_count()
        total_memory = _total_memory_bytes()
        disk_free = _disk_free_bytes(dataset_path)
        compatibility_sha256 = _compatibility_layer_sha256()
        compatibility_matches = compatibility_sha256 == lock.compatibility_layer_sha256
        prompt_matches = PROMPT_BUNDLE_SHA256 == lock.prompt_sha256
        upstream_source_matches = (
            lock.adapter_configuration is not None
            and lock.adapter_configuration.get("upstream_source_sha256")
            == "d276f725ecc5ea2c08f73e161f97760881332c3d33d20b197c2ffbac5f55edfe"
        )
        checks = [
            DoctorCheck(
                check_id="python",
                status=(
                    CheckStatus.PASS
                    if tuple(map(int, os.sys.version_info[:2])) >= (3, 11)
                    else CheckStatus.FAIL
                ),
                summary=f"Python {os.sys.version_info.major}.{os.sys.version_info.minor}",
                action=(
                    None
                    if tuple(map(int, os.sys.version_info[:2])) >= (3, 11)
                    else "Use Python 3.11 or newer."
                ),
            ),
            DoctorCheck(
                check_id="cpu",
                status=(
                    CheckStatus.WARN
                    if cpu_count is None
                    else CheckStatus.PASS
                    if cpu_count >= 2
                    else CheckStatus.FAIL
                ),
                summary=(
                    "CPU count could not be measured"
                    if cpu_count is None
                    else f"{cpu_count} logical CPU(s) detected; HLE requires at least 2"
                ),
                action=(
                    "Verify that the worker has at least two CPUs." if cpu_count is None else None
                ),
            ),
            DoctorCheck(
                check_id="memory",
                status=(
                    CheckStatus.WARN
                    if total_memory is None
                    else CheckStatus.PASS
                    if total_memory >= _MIN_RAM_BYTES
                    else CheckStatus.FAIL
                ),
                summary=(
                    "System RAM could not be measured"
                    if total_memory is None
                    else (
                        f"{_format_bytes(total_memory)} total RAM; "
                        f"minimum {_format_bytes(_MIN_RAM_BYTES)}"
                    )
                ),
                action=(
                    "Verify worker RAM before loading the approved multimodal snapshot."
                    if total_memory is None
                    else None
                ),
            ),
            DoctorCheck(
                check_id="disk",
                status=(
                    CheckStatus.WARN
                    if disk_free is None
                    else CheckStatus.PASS
                    if disk_free >= _MIN_DISK_BYTES
                    else CheckStatus.FAIL
                ),
                summary=(
                    "Free disk space could not be measured"
                    if disk_free is None
                    else (
                        f"{_format_bytes(disk_free)} free; minimum {_format_bytes(_MIN_DISK_BYTES)}"
                    )
                ),
                action=(
                    "Verify free space for the local snapshot and reports."
                    if disk_free is None
                    else None
                ),
            ),
            DoctorCheck(
                check_id="docker",
                status=CheckStatus.PASS,
                summary="Docker is not required by the HLE adapter",
            ),
            DoctorCheck(
                check_id="target-capability",
                status=CheckStatus.WARN,
                summary=(
                    "Target chat and inline-image capabilities require connector-time validation"
                ),
                action="Probe the selected target endpoint before a real-data run.",
            ),
            DoctorCheck(
                check_id="judge-capability",
                status=CheckStatus.WARN,
                summary="The separate judge endpoint requires connector-time validation",
                action="Configure and probe the profile's exact judge model.",
            ),
            DoctorCheck(
                check_id="harness-pin",
                status=(
                    CheckStatus.PASS
                    if prompt_matches and upstream_source_matches
                    else CheckStatus.FAIL
                ),
                summary=(
                    f"CAIS source {lock.harness_commit[:12]} and both prompt pins match"
                    if prompt_matches and upstream_source_matches
                    else "The checked-in CAIS source or prompt pin changed"
                ),
                action=(
                    None
                    if prompt_matches and upstream_source_matches
                    else "Review the profile and HLE compatibility module together."
                ),
            ),
            DoctorCheck(
                check_id="compatibility-layer",
                status=(CheckStatus.PASS if compatibility_matches else CheckStatus.FAIL),
                summary=(
                    "HLE compatibility module checksum verified"
                    if compatibility_matches
                    else "HLE compatibility module is missing or its checksum changed"
                ),
                action=(
                    None
                    if compatibility_matches
                    else "Review the module, tests, and all HLE profile locks together."
                ),
            ),
            DoctorCheck(
                check_id="protocol-completeness",
                status=(CheckStatus.PASS if not lock.unresolved_fields else CheckStatus.WARN),
                summary=(
                    "Profile has no unresolved protocol fields"
                    if not lock.unresolved_fields
                    else (
                        "Formal comparison is disabled; unresolved fields: "
                        + ", ".join(lock.unresolved_fields)
                    )
                ),
                action=(
                    None
                    if not lock.unresolved_fields
                    else "Resolve the published protocol evidence before formal comparison."
                ),
            ),
        ]
        if profile == "smoke":
            fixture_sha256 = (
                _sha256_file_or_none(_SMOKE_RESOURCE)
                if _SMOKE_RESOURCE.is_file() and not _SMOKE_RESOURCE.is_symlink()
                else None
            )
            fixture_ok = fixture_sha256 == lock.dataset_sha256
            checks.append(
                DoctorCheck(
                    check_id="synthetic-fixture",
                    status=CheckStatus.PASS if fixture_ok else CheckStatus.FAIL,
                    summary=(
                        "Two-item text/image synthetic fixture checksum verified"
                        if fixture_ok
                        else "Synthetic HLE fixture is missing or changed"
                    ),
                )
            )
        else:
            local_ok = (
                dataset_path is not None
                and dataset_path.is_file()
                and not dataset_path.is_symlink()
            )
            checks.append(
                DoctorCheck(
                    check_id="approved-local-dataset",
                    status=CheckStatus.PASS if local_ok else CheckStatus.FAIL,
                    summary=(
                        "Approved local HLE JSONL path is present"
                        if local_ok
                        else "No approved local HLE snapshot was provided"
                    ),
                    action=(
                        None
                        if local_ok
                        else (
                            "Accept the CAIS terms manually, export the approved test "
                            "snapshot to local JSONL, and pass its SHA-256."
                        )
                    ),
                )
            )
        return DoctorReport(
            benchmark_id=_BENCHMARK_ID,
            profile=profile,
            runnable=all(check.status is not CheckStatus.FAIL for check in checks),
            checks=tuple(checks),
        )

    def prepare(
        self,
        profile: str,
        *,
        dry_run: bool,
        dataset_path: Path | None = None,
        dataset_sha256: str | None = None,
        accepted_access: bool = False,
    ) -> PreparationResult:
        lock = get_profile_lock(_BENCHMARK_ID, profile)
        notices = (
            "CAIS HLE requires manual acceptance of the original provider's access terms.",
            "Do not publish, re-upload, or include official benchmark inputs in reports.",
            "Kairyu never downloads an ungated mirror or accepts terms on a user's behalf.",
            "The current public metadata declares 2,500 test items; other counts are distinct.",
        )
        if dry_run:
            return PreparationResult(
                benchmark_id=_BENCHMARK_ID,
                profile=profile,
                status=PreparationStatus.DRY_RUN,
                dry_run=True,
                dataset_revision=lock.dataset_revision,
                dataset_sha256=lock.dataset_sha256,
                item_count=(2 if profile == "smoke" else lock.expected_full_count),
                actions=(
                    "verify packaged CC0 fixture and image checksum"
                    if profile == "smoke"
                    else "verify approved local JSONL SHA-256, schema, images, and count",
                ),
                notices=notices,
            )

        if profile == "smoke":
            records, actual_sha256 = load_records_with_sha256(_SMOKE_RESOURCE)
            if actual_sha256 != lock.dataset_sha256:
                return PreparationResult(
                    benchmark_id=_BENCHMARK_ID,
                    profile=profile,
                    status=PreparationStatus.BLOCKED,
                    dry_run=False,
                    actions=("restore the checked-in synthetic HLE fixture",),
                    notices=notices,
                )
            if len(prepare_records(records)) != 2:
                raise ValueError("HLE synthetic smoke fixture must contain exactly two items")
            return PreparationResult(
                benchmark_id=_BENCHMARK_ID,
                profile=profile,
                status=PreparationStatus.READY,
                dry_run=False,
                dataset_revision=lock.dataset_revision,
                dataset_sha256=actual_sha256,
                item_count=2,
                notices=notices,
            )

        if not accepted_access or dataset_path is None or dataset_sha256 is None:
            return PreparationResult(
                benchmark_id=_BENCHMARK_ID,
                profile=profile,
                status=PreparationStatus.NEEDS_USER_ACTION,
                dry_run=False,
                actions=(
                    "manually accept the CAIS HLE access conditions",
                    "provide an approved local UTF-8 JSONL snapshot",
                    "provide the snapshot's lowercase SHA-256",
                ),
                notices=notices,
            )
        records, actual_sha256 = load_records_with_sha256(dataset_path)
        if actual_sha256 != dataset_sha256:
            raise ValueError("approved HLE snapshot SHA-256 does not match")
        if len(prepare_records(records)) != _EXPECTED_OFFICIAL_COUNT:
            raise ValueError("approved HLE snapshot must contain exactly 2,500 records")
        return PreparationResult(
            benchmark_id=_BENCHMARK_ID,
            profile=profile,
            status=PreparationStatus.READY,
            dry_run=False,
            dataset_revision=f"{lock.dataset_revision}:sha256:{actual_sha256}",
            dataset_sha256=actual_sha256,
            item_count=len(records),
            notices=notices,
        )

    def build_run_plan(
        self,
        selection: RunSelection,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> AdapterRunPlan:
        # The scope guard must precede profile/resource loading and dataset reads.
        decision = validate_run_guard(
            selection.mode,
            confirm_full_run=selection.confirm_full_run,
            limit=selection.limit,
            sample_ids=selection.sample_ids,
            environ=environ,
        )
        if selection.limit is not None and selection.sample_ids:
            raise ValueError("choose either limit or sample IDs, not both")
        if decision.mode is RunMode.SMOKE and selection.profile != "smoke":
            raise ValueError("smoke mode requires the smoke profile")
        if decision.mode is not RunMode.SMOKE and selection.profile == "smoke":
            raise ValueError("the synthetic smoke profile cannot run real-data modes")

        lock = get_profile_lock(_BENCHMARK_ID, selection.profile)
        judge_model = _resolve_judge_model(selection, lock)
        target_parameters = _validated_target_parameters(selection)
        judge_parameters = _validated_judge_parameters(selection, judge_model)
        compatibility_sha256 = _validate_compatibility_layer(lock)
        if decision.mode is RunMode.SMOKE:
            records, source_sha256 = load_records_with_sha256(_SMOKE_RESOURCE)
            if source_sha256 != lock.dataset_sha256:
                raise ValueError("HLE synthetic fixture checksum changed")
        else:
            source_path = Path(selection.dataset_path) if selection.dataset_path else None
            if (
                not selection.accepted_access
                or source_path is None
                or selection.dataset_sha256 is None
            ):
                raise ValueError("approved HLE local snapshot requires user action")
            records, source_sha256 = load_records_with_sha256(source_path)
            if source_sha256 != selection.dataset_sha256:
                raise ValueError("approved HLE snapshot SHA-256 does not match")
            if len(records) != _EXPECTED_OFFICIAL_COUNT:
                raise ValueError("approved HLE snapshot must contain exactly 2,500 records")

        all_prepared = prepare_records(records)
        if decision.mode is RunMode.FULL and len(all_prepared) != _EXPECTED_OFFICIAL_COUNT:
            raise ValueError("full HLE run requires exactly 2,500 records")

        selected = all_prepared
        canonical_sample_ids: tuple[str, ...] = ()
        if decision.sample_ids:
            selected, canonical_sample_ids = _resolve_sample_selection(
                all_prepared,
                decision.sample_ids,
            )
        elif decision.limit is not None:
            selected = all_prepared[: decision.limit]
        if not selected:
            raise ValueError("HLE run selection contains no items")

        items = tuple(
            AdapterItem(
                item_id=_public_item_id(record.record_id),
                ordinal=ordinal,
                input_sha256=record.input_sha256,
                prompt=record.question,
                target=record.answer,
                choice_permutation=(),
                image_data_uri=record.image,
                category=record.answer_type,
            )
            for ordinal, record in enumerate(selected)
        )
        base_protocol = lock.to_profile(_BENCHMARK_ID).protocol
        base_protocol_json = base_protocol.model_dump(mode="json")
        base_configuration = base_protocol_json["adapter_configuration"]
        if (
            base_protocol.retries != MAX_ATTEMPTS - 1
            or base_configuration.get("retry_budget_scope") != "per-model-request-total-attempts"
            or base_configuration.get("max_attempts_per_request") != MAX_ATTEMPTS
        ):
            raise ValueError("HLE retry policy does not match the pinned evaluator")
        dataset_snapshot_reviewed = (
            lock.dataset_sha256 is not None and source_sha256 == lock.dataset_sha256
        )
        reproducibility_evidence = {
            "hardware_conditions": {
                "status": "unresolved",
                "reason": "Configured target and judge providers do not attest hardware.",
            },
            "provider_api_version": {
                "status": "unresolved",
                "reason": "The connector contract does not expose canonical provider API versions.",
            },
            "runtime_dependency_environment": {
                "status": "unresolved",
                "reason": (
                    "The dependency lock is recorded but the active worker environment "
                    "is not attested."
                ),
            },
            "source_retrieval_date": {
                "status": "unresolved",
                "reason": "The profile lock does not carry a reviewed retrieval timestamp.",
            },
        }
        evidence_complete = all(
            evidence["status"] == "verified" for evidence in reproducibility_evidence.values()
        )
        adapter_configuration = {
            **dict(base_protocol_json["adapter_configuration"]),
            "compatibility_layer_sha256": compatibility_sha256,
            "dataset_snapshot_reviewed": dataset_snapshot_reviewed,
            "item_input_manifest_sha256": manifest_sha256(selected),
            "judge_generation_parameters": judge_parameters,
            "judge_model": judge_model,
            "record_order": "approved-snapshot-order",
            "reproducibility_evidence": reproducibility_evidence,
            "reproducibility_evidence_complete": evidence_complete,
            "target_model": selection.target_model,
        }
        protocol = base_protocol.model_copy(
            update={
                "dataset_revision": f"{lock.dataset_revision}:sha256:{source_sha256}",
                "sample_filter": {
                    "limit": decision.limit,
                    "mode": decision.mode.value,
                    "sample_ids": list(canonical_sample_ids),
                },
                "judge_model": judge_model,
                "judge_reasoning_mode": judge_parameters.get("reasoning_effort"),
                "generation_parameters": target_parameters,
                "timeout_seconds": max(
                    float(target_parameters["timeout_seconds"]),
                    float(judge_parameters["timeout_seconds"]),
                ),
                "adapter_configuration": adapter_configuration,
            }
        )
        signature_hash = protocol_hash(protocol)
        selection_payload = selection.model_dump(mode="json")
        selection_payload.update(
            {
                "generation_parameters": target_parameters,
                "judge_generation_parameters": judge_parameters,
                "judge_model": judge_model,
                "sample_ids": list(canonical_sample_ids),
            }
        )
        canonical_selection = RunSelection.model_validate(selection_payload)
        official_eligible = (
            decision.official_eligible
            and not protocol.unresolved_fields
            and dataset_snapshot_reviewed
            and evidence_complete
        )

        target_max_tokens = int(target_parameters["max_tokens"])
        judge_max_tokens = int(judge_parameters["max_tokens"])
        target_timeout = float(target_parameters["timeout_seconds"])
        judge_timeout = float(judge_parameters["timeout_seconds"])
        target_calls = len(items)
        judge_calls = len(items)
        total_calls = target_calls + judge_calls
        maximum_model_calls = total_calls * MAX_ATTEMPTS
        maximum_output_tokens = len(items) * (target_max_tokens + judge_max_tokens) * MAX_ATTEMPTS
        estimated_artifact_bytes = sum(
            len(item.prompt.encode("utf-8"))
            + (len(item.image_data_uri.encode("ascii")) if item.image_data_uri else 0)
            for item in items
        )
        estimated_artifact_bytes += maximum_output_tokens * 8
        estimate = RunEstimate(
            selected_item_count=len(items),
            model_calls=total_calls,
            maximum_model_calls=maximum_model_calls,
            estimated_input_tokens=None,
            maximum_output_tokens=maximum_output_tokens,
            estimated_duration_seconds=None,
            maximum_duration_seconds=(target_calls * target_timeout + judge_calls * judge_timeout)
            * MAX_ATTEMPTS,
            estimated_cost_usd=None,
            assumptions=(
                "Each selected item makes one logical target request and, after target "
                "success, one logical judge request.",
                "Each logical target and judge request has a strict five-attempt total "
                "budget including connector-internal attempts.",
                "Judge input tokens are unknown before the target response exists.",
                "Maximum output tokens sum the configured successful target and judge ceilings.",
                "Expected duration is unknown; the pre-connector ceiling is serial and "
                "connector binding expands it for role-specific retries and backoff.",
                "Cost is unknown because target and judge provider pricing is not configured.",
            ),
        )
        resources = ResourceRequirements(
            cpu_cores=2,
            ram_bytes=_MIN_RAM_BYTES,
            disk_bytes=max(_MIN_DISK_BYTES, estimated_artifact_bytes),
            docker_required=False,
            network_policy=(
                "Only the configured target and judge endpoints may use network; "
                "images are approved inline data and tools/web retrieval are disabled."
            ),
        )
        execution = ExecutionSpec(
            kind="in-process-adapter",
            command=("kairyu", "benchmark", "worker", "--once"),
            container_image_digest=None,
        )
        return AdapterRunPlan(
            benchmark_id=_BENCHMARK_ID,
            profile=selection.profile,
            mode=decision.mode,
            target_model=selection.target_model,
            items=items,
            protocol=protocol,
            protocol_hash=signature_hash,
            item_input_manifest_sha256=manifest_sha256(selected),
            expected_full_count=_EXPECTED_OFFICIAL_COUNT,
            estimated_model_calls=total_calls,
            estimate=estimate,
            resources=resources,
            execution=execution,
            official_eligible=official_eligible,
            selection=canonical_selection,
            model_roles=(
                ModelRolePlan(
                    role=ModelRole.TARGET,
                    model=selection.target_model,
                    model_calls=target_calls,
                    timeout_seconds=target_timeout,
                    maximum_output_tokens=len(items) * target_max_tokens * MAX_ATTEMPTS,
                ),
                ModelRolePlan(
                    role=ModelRole.JUDGE,
                    model=judge_model,
                    model_calls=judge_calls,
                    timeout_seconds=judge_timeout,
                    maximum_output_tokens=len(items) * judge_max_tokens * MAX_ATTEMPTS,
                ),
            ),
        )

    def protocol_signature(self, plan: AdapterRunPlan) -> ProtocolSignature:
        if plan.benchmark_id != _BENCHMARK_ID:
            raise ValueError("HLE adapter received a foreign run plan")
        return plan.protocol

    def run(
        self,
        plan: AdapterRunPlan,
        item: AdapterItem,
        connector: ModelConnector,
        *,
        cancel_check,
    ) -> ItemResult:
        validate_run_guard(
            plan.selection.mode,
            confirm_full_run=plan.selection.confirm_full_run,
            limit=plan.selection.limit,
            sample_ids=plan.selection.sample_ids,
        )
        compatibility_sha256 = _validate_compatibility_layer(
            get_profile_lock(_BENCHMARK_ID, plan.profile)
        )
        if (
            plan.protocol.adapter_configuration["compatibility_layer_sha256"]
            != compatibility_sha256
        ):
            raise ValueError("HLE plan compatibility-layer identity changed")
        if not isinstance(connector, ModelConnectorSet) or connector.judge is None:
            return _failed_item(item, "judge_connector_missing")
        judge_model = plan.selection.judge_model
        if judge_model is None:
            return _failed_item(item, "judge_model_missing")

        try:
            target_request = ModelRequest(
                request_id=_target_request_id(item),
                model=plan.target_model,
                messages=_target_messages(item),
                temperature=float(plan.selection.generation_parameters["temperature"]),
                top_p=float(plan.selection.generation_parameters["top_p"]),
                max_tokens=int(plan.selection.generation_parameters["max_tokens"]),
                seed=plan.selection.seed,
                timeout_seconds=float(plan.selection.generation_parameters["timeout_seconds"]),
                allow_empty_content=True,
            )
            target_request_sha256 = canonical_connector_request_sha256(target_request)
        except ValueError:
            return ItemResult(
                item_id=item.item_id,
                ordinal=item.ordinal,
                input_sha256=item.input_sha256,
                response_text="",
                target=item.target,
                correct=False,
                error_class="target_request_invalid",
                latency_seconds=0.0,
                target_attempts=0,
            )
        target_evidence = _AttemptEvidence()
        while True:
            remaining_attempts = MAX_ATTEMPTS - target_evidence.budget_used
            target_outcome = connector.target.complete(
                target_request,
                cancel_requested=cancel_check,
                max_attempts=remaining_attempts,
            )
            target_evidence.consume(
                target_outcome,
                remaining_attempts=remaining_attempts,
            )
            if target_outcome.error is not None:
                target_error = target_outcome.error
                if (
                    target_error.code is ConnectorErrorCode.CANCELLED
                    or target_evidence.budget_used >= MAX_ATTEMPTS
                ):
                    return ItemResult(
                        item_id=item.item_id,
                        ordinal=item.ordinal,
                        input_sha256=item.input_sha256,
                        response_text="",
                        target=item.target,
                        correct=False,
                        error_class=_connector_error_class(
                            ModelRole.TARGET,
                            target_error,
                        ),
                        latency_seconds=target_evidence.latency_seconds,
                        provider_request_id=target_error.provider_request_id,
                        target_attempts=target_evidence.attempts,
                        target_request_sha256=target_request_sha256,
                    )
                continue
            target_response = target_outcome.response
            assert target_response is not None
            break

        target_fields = _target_response_fields(
            target_response,
            target_evidence,
            target_request_sha256,
        )
        try:
            judge_request = ModelRequest(
                request_id=_judge_request_id(item),
                model=judge_model,
                messages=(
                    ConnectorMessage(
                        role="user",
                        content=build_judge_prompt(
                            question=item.prompt,
                            response=target_response.content,
                            correct_answer=item.target,
                        ),
                    ),
                ),
                temperature=float(plan.selection.judge_generation_parameters["temperature"]),
                top_p=float(plan.selection.judge_generation_parameters["top_p"]),
                max_tokens=int(plan.selection.judge_generation_parameters["max_tokens"]),
                reasoning_effort=plan.selection.judge_generation_parameters.get("reasoning_effort"),
                seed=plan.selection.seed,
                timeout_seconds=float(
                    plan.selection.judge_generation_parameters["timeout_seconds"]
                ),
            )
            judge_request_sha256 = canonical_connector_request_sha256(judge_request)
        except ValueError:
            return ItemResult(
                item_id=item.item_id,
                ordinal=item.ordinal,
                input_sha256=item.input_sha256,
                target=item.target,
                correct=False,
                error_class="judge_request_invalid",
                judge_latency_seconds=0.0,
                judge_attempts=0,
                **target_fields,
            )
        judge_evidence = _AttemptEvidence()
        while True:
            remaining_attempts = MAX_ATTEMPTS - judge_evidence.budget_used
            judge_outcome = connector.judge.complete(
                judge_request,
                cancel_requested=cancel_check,
                max_attempts=remaining_attempts,
            )
            judge_evidence.consume(
                judge_outcome,
                remaining_attempts=remaining_attempts,
            )
            if judge_outcome.error is not None:
                judge_error = judge_outcome.error
                if (
                    judge_error.code is ConnectorErrorCode.CANCELLED
                    or judge_evidence.budget_used >= MAX_ATTEMPTS
                ):
                    return ItemResult(
                        item_id=item.item_id,
                        ordinal=item.ordinal,
                        input_sha256=item.input_sha256,
                        target=item.target,
                        correct=False,
                        error_class=_connector_error_class(
                            ModelRole.JUDGE,
                            judge_error,
                        ),
                        **target_fields,
                        **_judge_error_fields(
                            judge_error,
                            judge_evidence,
                            judge_request_sha256,
                        ),
                    )
                continue

            judge_response = judge_outcome.response
            assert judge_response is not None
            try:
                verdict = parse_judge_response(judge_response.content)
            except (AssertionError, ValueError):
                verdict = None
            invalid_error = (
                "judge_parse_error"
                if verdict is None
                else "judge_confidence_out_of_range"
                if not 0 <= verdict.confidence <= 100
                else None
            )
            judge_fields = _judge_response_fields(
                judge_response,
                judge_evidence,
                judge_request_sha256,
            )
            if invalid_error is not None:
                if judge_evidence.budget_used >= MAX_ATTEMPTS:
                    return ItemResult(
                        item_id=item.item_id,
                        ordinal=item.ordinal,
                        input_sha256=item.input_sha256,
                        target=item.target,
                        correct=False,
                        error_class=invalid_error,
                        **target_fields,
                        **judge_fields,
                    )
                continue

            assert verdict is not None
            correct = verdict.correct == "yes"
            confidence = verdict.confidence / 100.0
            return ItemResult(
                item_id=item.item_id,
                ordinal=item.ordinal,
                input_sha256=item.input_sha256,
                extracted_answer=verdict.extracted_final_answer,
                target=item.target,
                correct=correct,
                scores={
                    "accuracy": 1.0 if correct else 0.0,
                    "confidence": confidence,
                },
                confidence=confidence,
                **target_fields,
                **judge_fields,
            )

    def collect(
        self,
        run_id: str,
        plan: AdapterRunPlan,
        results: tuple[ItemResult, ...],
    ) -> CollectedResult:
        scored = tuple(
            result
            for result in results
            if result.error_class is None and result.confidence is not None
        )
        has_results = bool(results)
        failed = len(results) - len(scored)
        correct_count = sum(result.correct for result in scored)
        upstream = compute_metrics(
            tuple(
                ScoredPrediction(
                    correct=result.correct,
                    confidence=round(result.confidence * 100),
                )
                for result in scored
                if result.confidence is not None
            ),
            total_questions=len(results),
            num_failed=failed,
        )
        eligible = (
            plan.official_eligible
            and bool(results)
            and all(result.error_class is None for result in results)
            and all(result.provider_model == plan.target_model for result in results)
            and all(result.judge_provider_model == plan.selection.judge_model for result in results)
        )
        metrics = (
            Metric(
                run_id=run_id,
                name="accuracy",
                display_name="Accuracy",
                value=(correct_count / len(results) * 100.0 if has_results else None),
                numerator=correct_count,
                denominator=len(results),
                scale=100.0,
                unit="percent",
                primary=True,
                higher_is_better=True,
                dimensions={"aggregation": "overall", "failed_as_incorrect": True},
                official_eligible=eligible,
            ),
            Metric(
                run_id=run_id,
                name="calibration-error",
                display_name="Calibration Error",
                value=upstream.calibration_error if has_results else None,
                denominator=len(scored),
                scale=100.0,
                unit="percent",
                primary=False,
                higher_is_better=False,
                dimensions={"p": "2", "beta": 100, "successful_judgements_only": True},
                official_eligible=eligible,
            ),
            Metric(
                run_id=run_id,
                name="accuracy-success-only",
                display_name="Success-only Accuracy",
                value=(correct_count / len(scored) * 100.0 if scored else None),
                numerator=correct_count,
                denominator=len(scored),
                scale=100.0,
                unit="percent",
                primary=False,
                higher_is_better=True,
                dimensions={"aggregation": "success-only"},
                official_eligible=eligible,
            ),
            Metric(
                run_id=run_id,
                name="confidence-interval",
                display_name="Overall 95% Wald CI half-width",
                value=upstream.confidence_interval if has_results else None,
                denominator=len(results),
                scale=100.0,
                unit="percentage points",
                primary=False,
                higher_is_better=False,
                dimensions={"confidence_level": 0.95, "estimator": "wald"},
                official_eligible=eligible,
            ),
            Metric(
                run_id=run_id,
                name="confidence-interval-success-only",
                display_name="Success-only 95% Wald CI half-width",
                value=upstream.confidence_interval_success_only if has_results else None,
                denominator=len(scored),
                scale=100.0,
                unit="percentage points",
                primary=False,
                higher_is_better=False,
                dimensions={"confidence_level": 0.95, "estimator": "wald"},
                official_eligible=eligible,
            ),
        )
        error_counts: dict[str, int] = {}
        for result in results:
            error = result.error_class or result.report_error_class
            if error is not None:
                error_counts[error] = error_counts.get(error, 0) + 1
        return CollectedResult(
            metrics=metrics,
            completed_count=len(scored),
            failed_count=failed,
            skipped_count=0,
            error_counts=error_counts,
        )

    def smoke_connector_results(
        self,
        plan: AdapterRunPlan,
        role: ModelRole,
    ) -> Mapping[str, ConnectorResult]:
        responses: dict[str, ConnectorResult] = {}
        for index, item in enumerate(plan.items):
            if role is ModelRole.TARGET:
                answer = item.target if index == 0 else "red"
                request_id = _target_request_id(item)
                responses[request_id] = _fixed_response(
                    request_id=request_id,
                    content=(
                        "Explanation: synthetic fixture response\n"
                        f"Answer: {answer}\n"
                        f"Confidence: {80 if index == 0 else 60}%"
                    ),
                    provider_request_id=f"fake-target-{item.ordinal}",
                    provider_model=plan.target_model,
                )
            elif role is ModelRole.JUDGE:
                request_id = _judge_request_id(item)
                extracted = item.target if index == 0 else "red"
                correct = "yes" if index == 0 else "no"
                confidence = 80 if index == 0 else 60
                responses[request_id] = _fixed_response(
                    request_id=request_id,
                    content=(
                        f"<extracted_final_answer>{extracted}</extracted_final_answer>"
                        "<reasoning>Synthetic fixture verdict only.</reasoning>"
                        f"<correct>{correct}</correct>"
                        f"<confidence>{confidence}</confidence>"
                    ),
                    provider_request_id=f"fake-judge-{item.ordinal}",
                    provider_model=plan.selection.judge_model or "missing-judge",
                )
        return responses

    def render_report_data(
        self,
        collected: CollectedResult,
    ) -> Mapping[str, Any]:
        metrics = {metric.name: metric for metric in collected.metrics}
        primary = metrics["accuracy"]
        return {
            "benchmark_metric": primary.display_name,
            "calibration_error": metrics["calibration-error"].value,
            "confidence_interval": metrics["confidence-interval"].value,
            "error_counts": dict(collected.error_counts),
            "metric_scale": primary.scale,
            "metric_value": primary.value,
            "success_only_accuracy": metrics["accuracy-success-only"].value,
        }


def _resolve_judge_model(selection: RunSelection, lock: ProfileLock) -> str:
    if "judge_model" in lock.unresolved_fields and selection.judge_model is None:
        raise ValueError("HLE profile requires an explicit judge model")
    judge_model = selection.judge_model or lock.judge_model
    if judge_model is None:
        raise ValueError("HLE requires a separate judge model")
    if (
        "judge_model" not in lock.unresolved_fields
        and lock.judge_model is not None
        and judge_model != lock.judge_model
    ):
        raise ValueError("HLE judge model does not match the selected profile")
    return judge_model


def _validated_target_parameters(selection: RunSelection) -> dict[str, Any]:
    parameters = dict(selection.model_dump(mode="json")["generation_parameters"])
    allowed = {"max_tokens", "repeats", "seed", "temperature", "timeout_seconds", "top_p"}
    if set(parameters) - allowed:
        raise ValueError("HLE target generation parameters contain unsupported fields")
    repeats = parameters.get("repeats", 1)
    if type(repeats) is not int or repeats != 1:
        raise ValueError("HLE target repeats must be exactly 1")
    if "seed" in parameters and parameters["seed"] != selection.seed:
        raise ValueError("HLE target seed must match the run selection")
    try:
        request = ModelRequest(
            request_id="hle-target-generation-validation",
            model=selection.target_model,
            messages=({"role": "user", "content": "HLE target validation"},),
            temperature=parameters.get("temperature", 0.0),
            top_p=parameters.get("top_p", 1.0),
            max_tokens=parameters.get("max_tokens", 4_096),
            seed=selection.seed,
            timeout_seconds=parameters.get("timeout_seconds", 600.0),
        )
    except ValueError as exc:
        raise ValueError("HLE target generation parameters are invalid") from exc
    return {
        "max_tokens": request.max_tokens,
        "repeats": 1,
        "seed": selection.seed,
        "temperature": request.temperature,
        "timeout_seconds": request.timeout_seconds,
        "top_p": request.top_p,
    }


def _validated_judge_parameters(
    selection: RunSelection,
    judge_model: str,
) -> dict[str, Any]:
    parameters = dict(selection.model_dump(mode="json")["judge_generation_parameters"])
    allowed = {
        "max_tokens",
        "reasoning_effort",
        "seed",
        "temperature",
        "timeout_seconds",
        "top_p",
    }
    if set(parameters) - allowed:
        raise ValueError("HLE judge generation parameters contain unsupported fields")
    if "seed" in parameters and parameters["seed"] != selection.seed:
        raise ValueError("HLE judge seed must match the run selection")
    try:
        request = ModelRequest(
            request_id="hle-judge-generation-validation",
            model=judge_model,
            messages=({"role": "user", "content": "HLE judge validation"},),
            temperature=parameters.get("temperature", 0.0),
            top_p=parameters.get("top_p", 1.0),
            max_tokens=parameters.get("max_tokens", 2_048),
            reasoning_effort=parameters.get("reasoning_effort"),
            seed=selection.seed,
            timeout_seconds=parameters.get("timeout_seconds", 600.0),
        )
    except ValueError as exc:
        raise ValueError("HLE judge generation parameters are invalid") from exc
    validated: dict[str, Any] = {
        "max_tokens": request.max_tokens,
        "seed": selection.seed,
        "temperature": request.temperature,
        "timeout_seconds": request.timeout_seconds,
        "top_p": request.top_p,
    }
    if request.reasoning_effort is not None:
        validated["reasoning_effort"] = request.reasoning_effort
    return validated


def _target_messages(item: AdapterItem) -> tuple[ConnectorMessage, ...]:
    content: list[ConnectorTextPart | ConnectorImagePart] = [ConnectorTextPart(text=item.prompt)]
    if item.image_data_uri is not None:
        content.append(ConnectorImagePart(image_url=ConnectorImageURL(url=item.image_data_uri)))
    return (
        ConnectorMessage(role="system", content=SYSTEM_PROMPT),
        ConnectorMessage(role="user", content=tuple(content)),
    )


def _target_request_id(item: AdapterItem) -> str:
    return f"hle-target-{item.ordinal}-{item.input_sha256[:16]}"


def _judge_request_id(item: AdapterItem) -> str:
    return f"hle-judge-{item.ordinal}-{item.input_sha256[:16]}"


def _failed_item(item: AdapterItem, error_class: str) -> ItemResult:
    return ItemResult(
        item_id=item.item_id,
        ordinal=item.ordinal,
        input_sha256=item.input_sha256,
        response_text="",
        target=item.target,
        correct=False,
        error_class=error_class,
    )


def _connector_error_class(role: ModelRole, error: ConnectorError) -> str:
    if error.code is ConnectorErrorCode.CANCELLED:
        return ConnectorErrorCode.CANCELLED.value
    return f"{role.value}_{error.code.value}"


def _target_response_fields(
    response: ConnectorResponse,
    evidence: _AttemptEvidence,
    request_sha256: str,
) -> dict[str, Any]:
    return {
        "response_text": response.content,
        "latency_seconds": evidence.latency_seconds,
        "finish_reason": response.finish_reason,
        "provider_request_id": response.provider_request_id,
        "provider_model": response.provider_model,
        "input_tokens": evidence.input_tokens,
        "output_tokens": evidence.output_tokens,
        "target_attempts": evidence.attempts,
        "target_request_sha256": request_sha256,
    }


def _judge_response_fields(
    response: ConnectorResponse,
    evidence: _AttemptEvidence,
    request_sha256: str,
) -> dict[str, Any]:
    return {
        "judge_response_text": response.content,
        "judge_finish_reason": response.finish_reason,
        "judge_provider_request_id": response.provider_request_id,
        "judge_provider_model": response.provider_model,
        "judge_input_tokens": evidence.input_tokens,
        "judge_output_tokens": evidence.output_tokens,
        "judge_latency_seconds": evidence.latency_seconds,
        "judge_attempts": evidence.attempts,
        "judge_request_sha256": request_sha256,
    }


def _judge_error_fields(
    error: ConnectorError,
    evidence: _AttemptEvidence,
    request_sha256: str,
) -> dict[str, Any]:
    return {
        "judge_provider_request_id": error.provider_request_id,
        "judge_provider_model": evidence.last_response_provider_model,
        "judge_input_tokens": evidence.input_tokens,
        "judge_output_tokens": evidence.output_tokens,
        "judge_latency_seconds": evidence.latency_seconds,
        "judge_attempts": evidence.attempts,
        "judge_request_sha256": request_sha256,
    }


def _fixed_response(
    *,
    request_id: str,
    content: str,
    provider_request_id: str,
    provider_model: str,
) -> ConnectorResult:
    return ConnectorResult(
        response=ConnectorResponse(
            request_id=request_id,
            content=content,
            finish_reason="stop",
            provider_request_id=provider_request_id,
            provider_model=provider_model,
            usage=ConnectorUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            latency_seconds=0.0,
            attempts=1,
        )
    )


def _public_item_id(source_record_id: str) -> str:
    digest = hashlib.sha256(source_record_id.encode("utf-8")).hexdigest()
    return f"hle-{digest[:32]}"


def _resolve_sample_selection(
    records: tuple[PreparedHLERecord, ...],
    requested_ids: tuple[str, ...],
) -> tuple[tuple[PreparedHLERecord, ...], tuple[str, ...]]:
    resolved_ids: list[str] = []
    missing_count = 0
    ambiguous_count = 0
    for requested_id in requested_ids:
        matches = tuple(
            record
            for record in records
            if requested_id in {record.record_id, _public_item_id(record.record_id)}
        )
        if not matches:
            missing_count += 1
            continue
        if len(matches) != 1:
            ambiguous_count += 1
            continue
        resolved_ids.append(_public_item_id(matches[0].record_id))
    if missing_count:
        raise ValueError(f"{missing_count} requested HLE sample ID(s) were not found")
    if ambiguous_count:
        raise ValueError(f"{ambiguous_count} requested HLE sample ID(s) were ambiguous")
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("requested HLE sample IDs identify duplicate items")
    selected_ids = set(resolved_ids)
    selected = tuple(
        record for record in records if _public_item_id(record.record_id) in selected_ids
    )
    canonical_ids = tuple(_public_item_id(record.record_id) for record in selected)
    return selected, canonical_ids


def _sha256_file_or_none(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except (OSError, ValueError):
        return None


def _compatibility_layer_sha256() -> str | None:
    return _sha256_file_or_none(_COMPATIBILITY_LAYER_RESOURCE)


def _validate_compatibility_layer(lock: ProfileLock) -> str:
    observed = _compatibility_layer_sha256()
    if observed != lock.compatibility_layer_sha256:
        raise ValueError("HLE compatibility module checksum does not match the profile lock")
    return observed


def _total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(page_count, int):
        return None
    total = page_size * page_count
    return total if total > 0 else None


def _disk_free_bytes(dataset_path: Path | None) -> int | None:
    probe = Path.cwd() if dataset_path is None else dataset_path
    if not probe.exists():
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _format_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


__all__ = ["HumanitysLastExamAdapter"]
