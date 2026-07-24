"""GPQA Diamond adapter pinned to EvalScope 1.8.1 semantics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
from collections.abc import Mapping
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
    PreparationResult,
    PreparationStatus,
    ResourceRequirements,
    RunEstimate,
    RunSelection,
)
from kairyu.evaluation.adapters.gpqa_v181 import (
    PROMPT_SHA256,
    PreparedGPQARecord,
    extract_answer,
    load_records_with_sha256,
    manifest_sha256,
    prepare_records,
    sha256_file,
)
from kairyu.evaluation.connectors import ModelConnector, ModelRequest
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

_BENCHMARK_ID = "gpqa-diamond"
_EXPECTED_OFFICIAL_COUNT = 198
_MIN_RAM_BYTES = 512 * 1024 * 1024
_MIN_DISK_BYTES = 64 * 1024 * 1024
_SMOKE_RESOURCE = Path(__file__).parents[1] / "resources" / "fixtures" / "gpqa-diamond-smoke.jsonl"
_COMPATIBILITY_LAYER_RESOURCE = Path(__file__).with_name("gpqa_v181.py")


class GPQADiamondAdapter(BenchmarkAdapter):
    """Offline-first GPQA runner with explicit gated-data boundaries."""

    def metadata(self) -> BenchmarkDefinition:
        return BenchmarkDefinition(
            benchmark_id=_BENCHMARK_ID,
            display_name="GPQA Diamond",
            description=(
                "198 expert-written science multiple-choice questions; official "
                "data access is gated and examples are not emitted in reports."
            ),
            benchmark_version="gpqa-diamond-evalscope-v1.8.1",
            licenses=(
                "GPQA data: CC-BY-4.0 with original-provider access terms",
                "EvalScope compatibility source: Apache-2.0",
                "Kairyu synthetic smoke fixture: CC0-1.0",
            ),
            data_sources=(
                "https://huggingface.co/datasets/Idavidrein/gpqa",
                "https://github.com/idavidrein/gpqa",
                "https://github.com/modelscope/evalscope/tree/v1.8.1/evalscope/benchmarks/gpqa",
            ),
            required_auth=("manually approved local GPQA snapshot",),
            primary_metric="Accuracy",
            auxiliary_metrics=("invalid answer count", "API error count"),
            higher_is_better=True,
            modalities=("text",),
            required_capabilities=("chat completions",),
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
        prompt_pin_matches = PROMPT_SHA256 == lock.prompt_sha256
        compatibility_sha256 = _compatibility_layer_sha256()
        compatibility_layer_matches = compatibility_sha256 == lock.compatibility_layer_sha256
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
                    if cpu_count >= 1
                    else CheckStatus.FAIL
                ),
                summary=(
                    "CPU count could not be measured"
                    if cpu_count is None
                    else f"{cpu_count} logical CPU(s) detected; GPQA requires at least 1"
                ),
                action=(
                    "Verify that the worker has at least one CPU." if cpu_count is None else None
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
                    "Verify the worker RAM limit before execution."
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
                    "Verify free space for checkpoints and reports." if disk_free is None else None
                ),
            ),
            DoctorCheck(
                check_id="docker",
                status=CheckStatus.PASS,
                summary="Docker is not required by the GPQA adapter",
            ),
            DoctorCheck(
                check_id="model-capability",
                status=CheckStatus.WARN,
                summary=(
                    "Chat-completions capability is not measurable without a connector and model"
                ),
                action="Validate the selected OpenAI-compatible endpoint at run time.",
            ),
            DoctorCheck(
                check_id="harness-pin",
                status=CheckStatus.PASS if prompt_pin_matches else CheckStatus.FAIL,
                summary=(
                    f"Compatibility source {lock.harness_commit[:12]} and prompt checksum match"
                    if prompt_pin_matches
                    else "Checked-in prompt checksum does not match the profile lock"
                ),
                action=(
                    "Restore or review the profile and compatibility layer together."
                    if not prompt_pin_matches
                    else None
                ),
            ),
            DoctorCheck(
                check_id="compatibility-layer",
                status=(CheckStatus.PASS if compatibility_layer_matches else CheckStatus.FAIL),
                summary=(
                    "GPQA compatibility module checksum verified"
                    if compatibility_layer_matches
                    else "GPQA compatibility module is missing or its checksum changed"
                ),
                action=(
                    None
                    if compatibility_layer_matches
                    else "Review the compatibility module and update every profile lock together."
                ),
            ),
        ]
        if profile == "smoke":
            fixture_ok = (
                _SMOKE_RESOURCE.is_file()
                and not _SMOKE_RESOURCE.is_symlink()
                and sha256_file(_SMOKE_RESOURCE) == lock.dataset_sha256
            )
            checks.append(
                DoctorCheck(
                    check_id="synthetic-fixture",
                    status=CheckStatus.PASS if fixture_ok else CheckStatus.FAIL,
                    summary=(
                        "Synthetic fixture checksum verified"
                        if fixture_ok
                        else "Synthetic fixture is missing or its checksum changed"
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
                        "Approved local dataset path is present"
                        if local_ok
                        else "No approved local GPQA snapshot was provided"
                    ),
                    action=(
                        None
                        if local_ok
                        else (
                            "Accept the original provider terms manually, then pass "
                            "the local .csv or .jsonl path and SHA-256 to prepare."
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
            "GPQA data is CC-BY-4.0 and subject to the original provider's manual access terms.",
            "Kairyu never downloads an ungated mirror to bypass those terms.",
            "Official question and choice text is excluded from reports.",
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
                    "verify package fixture checksum"
                    if profile == "smoke"
                    else "verify manually approved local snapshot SHA-256 and schema",
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
                    actions=("restore the checked-in synthetic fixture",),
                    notices=notices,
                )
            if len(records) != 2:
                raise ValueError("GPQA synthetic smoke fixture must contain exactly two items")
            return PreparationResult(
                benchmark_id=_BENCHMARK_ID,
                profile=profile,
                status=PreparationStatus.READY,
                dry_run=False,
                dataset_revision=lock.dataset_revision,
                dataset_sha256=actual_sha256,
                item_count=len(records),
                notices=notices,
            )

        if not accepted_access or dataset_path is None or dataset_sha256 is None:
            return PreparationResult(
                benchmark_id=_BENCHMARK_ID,
                profile=profile,
                status=PreparationStatus.NEEDS_USER_ACTION,
                dry_run=False,
                actions=(
                    "manually accept the original provider's terms",
                    "provide an approved local .csv or .jsonl snapshot",
                    "provide the snapshot's lowercase SHA-256",
                ),
                notices=notices,
            )
        records, actual_sha256 = load_records_with_sha256(dataset_path)
        if actual_sha256 != dataset_sha256:
            raise ValueError("approved GPQA snapshot SHA-256 does not match")
        if len(records) != _EXPECTED_OFFICIAL_COUNT:
            raise ValueError("approved GPQA Diamond snapshot must contain exactly 198 records")
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
        # This must remain the first operation: it precedes profile/resource
        # loading, local dataset reads, connector construction, and execution.
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

        generation_parameters = _validated_generation_parameters(selection)
        lock = get_profile_lock(_BENCHMARK_ID, selection.profile)
        compatibility_sha256 = _validate_compatibility_layer(lock)
        if decision.mode is RunMode.SMOKE:
            source_path = _SMOKE_RESOURCE
            records, source_sha256 = load_records_with_sha256(source_path)
            if source_sha256 != lock.dataset_sha256:
                raise ValueError("GPQA synthetic fixture checksum changed")
        else:
            source_path = Path(selection.dataset_path) if selection.dataset_path else None
            if (
                not selection.accepted_access
                or source_path is None
                or selection.dataset_sha256 is None
            ):
                raise ValueError("approved GPQA local snapshot requires user action")
            records, source_sha256 = load_records_with_sha256(source_path)
            if source_sha256 != selection.dataset_sha256:
                raise ValueError("approved GPQA snapshot SHA-256 does not match")
            if len(records) != _EXPECTED_OFFICIAL_COUNT:
                raise ValueError("approved GPQA Diamond snapshot must contain exactly 198 records")

        all_prepared = prepare_records(records, seed=selection.seed)
        if decision.mode is RunMode.FULL and len(all_prepared) != _EXPECTED_OFFICIAL_COUNT:
            raise ValueError("full GPQA Diamond run requires exactly 198 records")

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
            raise ValueError("GPQA run selection contains no items")

        items = tuple(
            AdapterItem(
                item_id=_public_item_id(item.record_id),
                ordinal=selection_ordinal,
                input_sha256=item.input_sha256,
                prompt=item.prompt,
                target=item.target,
                choice_permutation=item.choice_permutation,
            )
            for selection_ordinal, item in enumerate(selected)
        )
        permutation_digest = hashlib.sha256(
            _canonical_json(
                [
                    {
                        "item_id": _public_item_id(item.record_id),
                        "permutation": item.choice_permutation,
                    }
                    for item in all_prepared
                ]
            )
        ).hexdigest()
        base_protocol = lock.to_profile(_BENCHMARK_ID).protocol
        base_protocol_json = base_protocol.model_dump(mode="json")
        dataset_snapshot_reviewed = (
            lock.dataset_sha256 is not None and source_sha256 == lock.dataset_sha256
        )
        reproducibility_evidence = {
            "hardware_conditions": {
                "status": "unresolved",
                "reason": (
                    "The OpenAI-compatible connector does not expose worker or provider hardware."
                ),
            },
            "provider_api_version": {
                "status": "unresolved",
                "reason": (
                    "No canonical provider API version is supplied by the current "
                    "connector contract."
                ),
            },
            "runtime_dependency_environment": {
                "status": "unresolved",
                "reason": (
                    "The reviewed dependency lock is recorded but the active "
                    "environment is not attested."
                ),
            },
            "source_retrieval_date": {
                "status": "unresolved",
                "reason": "The profile lock does not yet carry a reviewed source retrieval date.",
            },
        }
        reproducibility_evidence_complete = all(
            evidence["status"] == "verified" for evidence in reproducibility_evidence.values()
        )
        adapter_configuration = {
            **dict(base_protocol_json["adapter_configuration"]),
            "all_item_permutations_sha256": permutation_digest,
            "evalscope_wheel_sha256": lock.evalscope_wheel_sha256,
            "compatibility_layer_sha256": compatibility_sha256,
            "record_order": "approved-snapshot-order",
            "dataset_snapshot_reviewed": dataset_snapshot_reviewed,
            "model_request_policy": "one-chat-completion-per-item-v1",
            "request_concurrency": 1,
            "reproducibility_evidence": reproducibility_evidence,
            "reproducibility_evidence_complete": reproducibility_evidence_complete,
            "target_model": selection.target_model,
        }
        protocol = base_protocol.model_copy(
            update={
                "dataset_revision": (f"{lock.dataset_revision}:sha256:{source_sha256}"),
                "sample_filter": {
                    "limit": decision.limit,
                    "mode": decision.mode.value,
                    "sample_ids": list(canonical_sample_ids),
                },
                "generation_parameters": generation_parameters,
                "adapter_configuration": adapter_configuration,
            }
        )
        signature_hash = protocol_hash(protocol)
        canonical_selection_payload = selection.model_dump(mode="json")
        canonical_selection_payload.update(
            {
                "generation_parameters": generation_parameters,
                "sample_ids": list(canonical_sample_ids),
            }
        )
        canonical_selection = RunSelection.model_validate(canonical_selection_payload)
        expected_official_generation = {
            "max_tokens": 1_024,
            "repeats": 1,
            "seed": lock.seed,
            "temperature": 0.0,
            "timeout_seconds": 120.0,
            "top_p": 1.0,
        }
        official_eligible = (
            decision.official_eligible
            and not protocol.unresolved_fields
            and dataset_snapshot_reviewed
            and reproducibility_evidence_complete
            and generation_parameters == expected_official_generation
        )
        max_tokens = int(generation_parameters["max_tokens"])
        timeout_seconds = float(generation_parameters["timeout_seconds"])
        estimated_input_tokens = sum(_estimate_prompt_tokens(item.prompt) for item in items)
        maximum_output_tokens = len(items) * max_tokens
        estimated_artifact_bytes = sum(len(item.prompt.encode("utf-8")) for item in items)
        estimated_artifact_bytes += maximum_output_tokens * 8
        estimate = RunEstimate(
            selected_item_count=len(items),
            model_calls=len(items),
            maximum_model_calls=len(items),
            estimated_input_tokens=estimated_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
            estimated_duration_seconds=None,
            maximum_duration_seconds=len(items) * timeout_seconds,
            estimated_cost_usd=None,
            assumptions=(
                "Input tokens use a UTF-8-bytes/4 planning heuristic; "
                "the target tokenizer is not loaded.",
                "Maximum output tokens are the configured per-call ceiling, not expected usage.",
                "Expected duration is unknown because provider throughput is not "
                "configured; the pre-connector duration is a one-attempt serial timeout "
                "ceiling that connector binding expands for retries and backoff.",
                "Cost is unknown because provider pricing is not configured.",
            ),
        )
        resources = ResourceRequirements(
            cpu_cores=1,
            ram_bytes=_MIN_RAM_BYTES,
            disk_bytes=max(_MIN_DISK_BYTES, estimated_artifact_bytes),
            docker_required=False,
            network_policy=(
                "Only the configured model endpoint may require network access; "
                "the fixed smoke connector is offline."
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
            item_input_manifest_sha256=manifest_sha256(tuple(item for item in selected)),
            expected_full_count=_EXPECTED_OFFICIAL_COUNT,
            estimated_model_calls=len(items),
            estimate=estimate,
            resources=resources,
            execution=execution,
            official_eligible=official_eligible,
            selection=canonical_selection,
        )

    def protocol_signature(self, plan: AdapterRunPlan) -> ProtocolSignature:
        if plan.benchmark_id != _BENCHMARK_ID:
            raise ValueError("GPQA adapter received a foreign run plan")
        return plan.protocol

    def run(
        self,
        plan: AdapterRunPlan,
        item: AdapterItem,
        connector: ModelConnector,
        *,
        cancel_check,
    ) -> ItemResult:
        # A direct caller cannot bypass the scope guard.
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
            raise ValueError("GPQA plan compatibility-layer identity changed")
        request = ModelRequest(
            request_id=f"gpqa-{item.ordinal}-{item.input_sha256[:16]}",
            model=plan.target_model,
            messages=({"role": "user", "content": item.prompt},),
            temperature=float(plan.selection.generation_parameters.get("temperature", 0.0)),
            top_p=float(plan.selection.generation_parameters.get("top_p", 1.0)),
            max_tokens=int(plan.selection.generation_parameters.get("max_tokens", 1_024)),
            seed=plan.selection.seed,
            timeout_seconds=float(
                plan.selection.generation_parameters.get("timeout_seconds", 120.0)
            ),
        )
        outcome = connector.complete(request, cancel_requested=cancel_check)
        if outcome.error is not None:
            return ItemResult(
                item_id=item.item_id,
                ordinal=item.ordinal,
                input_sha256=item.input_sha256,
                response_text="",
                target=item.target,
                correct=False,
                error_class=outcome.error.code.value,
            )
        response = outcome.response
        assert response is not None
        extracted = extract_answer(response.content)
        usage = response.usage
        return ItemResult(
            item_id=item.item_id,
            ordinal=item.ordinal,
            input_sha256=item.input_sha256,
            response_text=response.content,
            extracted_answer=extracted,
            target=item.target,
            correct=extracted == item.target,
            latency_seconds=response.latency_seconds,
            finish_reason=response.finish_reason,
            provider_request_id=response.provider_request_id,
            provider_model=response.provider_model,
            input_tokens=(usage.prompt_tokens if usage is not None else None),
            output_tokens=(usage.completion_tokens if usage is not None else None),
        )

    def collect(
        self,
        run_id: str,
        plan: AdapterRunPlan,
        results: tuple[ItemResult, ...],
    ) -> CollectedResult:
        scored = tuple(result for result in results if result.error_class is None)
        correct = sum(result.correct for result in scored)
        denominator = len(scored)
        value = (correct / denominator * 100.0) if denominator else None
        error_counts: dict[str, int] = {}
        for result in results:
            if result.error_class is not None:
                error_counts[result.error_class] = error_counts.get(result.error_class, 0) + 1
            elif result.extracted_answer not in {"A", "B", "C", "D"}:
                error_counts["invalid_answer"] = error_counts.get("invalid_answer", 0) + 1
        metric = Metric(
            run_id=run_id,
            name="accuracy",
            display_name="Accuracy",
            value=value,
            numerator=correct,
            denominator=denominator,
            scale=100.0,
            unit="percent",
            primary=True,
            higher_is_better=True,
            dimensions={"aggregation": "mean", "shots": 0},
            official_eligible=(
                plan.official_eligible
                and bool(scored)
                and all(result.provider_model == plan.target_model for result in scored)
            ),
        )
        return CollectedResult(
            metrics=(metric,),
            completed_count=denominator,
            failed_count=len(results) - denominator,
            skipped_count=0,
            error_counts=error_counts,
        )

    def render_report_data(
        self,
        collected: CollectedResult,
    ) -> Mapping[str, Any]:
        metric = collected.metrics[0]
        return {
            "benchmark_metric": metric.display_name,
            "error_counts": dict(collected.error_counts),
            "metric_value": metric.value,
            "metric_scale": metric.scale,
        }


def _validated_generation_parameters(selection: RunSelection) -> dict[str, Any]:
    parameters = dict(selection.model_dump(mode="json")["generation_parameters"])
    repeats = parameters.get("repeats", 1)
    if type(repeats) is not int or repeats != 1:
        raise ValueError("GPQA repeats must be exactly 1")
    if "seed" in parameters and parameters["seed"] != selection.seed:
        raise ValueError("GPQA generation seed must match the run selection seed")
    try:
        request = ModelRequest(
            request_id="gpqa-generation-validation",
            model=selection.target_model,
            messages=({"role": "user", "content": "GPQA generation validation"},),
            temperature=parameters.get("temperature", 0.0),
            top_p=parameters.get("top_p", 1.0),
            max_tokens=parameters.get("max_tokens", 1_024),
            seed=selection.seed,
            timeout_seconds=parameters.get("timeout_seconds", 120.0),
        )
    except ValueError as exc:
        raise ValueError(
            "GPQA generation parameters do not satisfy ModelRequest constraints"
        ) from exc
    parameters.update(
        {
            "max_tokens": request.max_tokens,
            "repeats": 1,
            "seed": selection.seed,
            "temperature": request.temperature,
            "timeout_seconds": request.timeout_seconds,
            "top_p": request.top_p,
        }
    )
    return parameters


def _public_item_id(source_record_id: str) -> str:
    digest = hashlib.sha256(source_record_id.encode("utf-8")).hexdigest()
    return f"gpqa-{digest[:32]}"


def _resolve_sample_selection(
    records: tuple[PreparedGPQARecord, ...],
    requested_ids: tuple[str, ...],
) -> tuple[tuple[PreparedGPQARecord, ...], tuple[str, ...]]:
    """Resolve source or already-public IDs without retaining source IDs."""

    resolved_public_ids: list[str] = []
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
        resolved_public_ids.append(_public_item_id(matches[0].record_id))
    if missing_count:
        raise ValueError(f"{missing_count} requested GPQA sample ID(s) were not found")
    if ambiguous_count:
        raise ValueError(f"{ambiguous_count} requested GPQA sample ID(s) were ambiguous")
    if len(set(resolved_public_ids)) != len(resolved_public_ids):
        raise ValueError("requested GPQA sample IDs identify duplicate items")

    selected_ids = set(resolved_public_ids)
    selected = tuple(
        record for record in records if _public_item_id(record.record_id) in selected_ids
    )
    canonical_public_ids = tuple(_public_item_id(record.record_id) for record in selected)
    return selected, canonical_public_ids


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _compatibility_layer_sha256() -> str | None:
    try:
        return sha256_file(_COMPATIBILITY_LAYER_RESOURCE)
    except ValueError:
        return None


def _validate_compatibility_layer(lock: ProfileLock) -> str:
    observed = _compatibility_layer_sha256()
    if observed != lock.compatibility_layer_sha256:
        raise ValueError("GPQA compatibility module checksum does not match the profile lock")
    return observed


def _estimate_prompt_tokens(prompt: str) -> int:
    byte_count = len(prompt.encode("utf-8"))
    return max(1, (byte_count + 3) // 4)


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


def evalscope_distribution_version() -> str | None:
    """Return an installed EvalScope version without importing it."""

    try:
        return importlib.metadata.version("evalscope")
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = ["GPQADiamondAdapter", "evalscope_distribution_version"]
