"""Lifecycle commands for the reproducible evaluation platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kairyu.evaluation.adapters.base import RunSelection
from kairyu.evaluation.guards import validate_run_guard
from kairyu.evaluation.references import load_reference_snapshot
from kairyu.evaluation.registry import benchmark_catalog
from kairyu.evaluation.schemas import RunMode
from kairyu.evaluation.service import (
    BenchmarkService,
    ConnectorConfig,
    EvaluationRuntime,
    bind_connector_to_plan,
)
from kairyu.evaluation.worker import render_saved_report, run_worker_once

_DEFAULT_STATE_DIR = Path(".kairyu/evaluation")


def add_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register benchmark lifecycle commands without loading concrete adapters."""

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run reproducible, independently managed accuracy benchmarks.",
    )
    commands = benchmark.add_subparsers(dest="benchmark_command", required=True)

    list_command = commands.add_parser("list", help="List the exact benchmark catalog.")
    _add_format(list_command)

    doctor = commands.add_parser("doctor", help="Inspect one benchmark's prerequisites.")
    doctor.add_argument("benchmark_id")
    doctor.add_argument("--profile", default="smoke")
    doctor.add_argument("--dataset-path", type=Path)
    _add_format(doctor)

    prepare = commands.add_parser(
        "prepare",
        help="Validate an approved local snapshot; never auto-accept data terms.",
    )
    prepare.add_argument("benchmark_id")
    prepare.add_argument("--profile", default="smoke")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--dataset-path", type=Path)
    prepare.add_argument("--dataset-sha256")
    prepare.add_argument("--accepted-access", action="store_true")
    _add_format(prepare)

    plan = commands.add_parser("plan", help="Build a guarded run plan without enqueueing.")
    plan.add_argument("benchmark_id")
    _add_selection_arguments(plan)
    _add_connector_arguments(plan)
    _add_format(plan)

    run = commands.add_parser("run", help="Validate and enqueue one durable benchmark job.")
    run.add_argument("benchmark_id")
    _add_selection_arguments(run)
    _add_connector_arguments(run)
    _add_state_dir(run)
    run.add_argument(
        "--wait",
        action="store_true",
        help="Claim this smoke job in-process and wait for its report.",
    )
    _add_format(run)

    worker = commands.add_parser("worker", help="Claim one durable evaluation job.")
    _add_state_dir(worker)
    worker.add_argument("--once", action="store_true", required=True)
    worker.add_argument("--worker-id", default="worker-cli")
    worker.add_argument("--lease-seconds", type=float, default=900.0)
    _add_format(worker)

    status = commands.add_parser("status", help="Show durable run progress.")
    status.add_argument("run_id")
    _add_state_dir(status)
    _add_format(status)

    cancel = commands.add_parser("cancel", help="Persist cancellation intent.")
    cancel.add_argument("run_id")
    _add_state_dir(cancel)
    _add_format(cancel)

    resume = commands.add_parser("resume", help="Create an immutable successor attempt.")
    resume.add_argument("run_id")
    resume.add_argument("--new-run-id")
    _add_state_dir(resume)
    _add_format(resume)

    report = commands.add_parser(
        "report",
        help="Regenerate reports from saved local evidence only.",
    )
    report.add_argument("run_id")
    _add_state_dir(report)
    _add_format(report)

    references = commands.add_parser("references", help="Inspect versioned reference data.")
    reference_commands = references.add_subparsers(
        dest="reference_command",
        required=True,
    )
    reference_list = reference_commands.add_parser("list")
    reference_list.add_argument("--benchmark", default="gpqa-diamond")
    _add_format(reference_list)


def handle(args: argparse.Namespace) -> int:
    """Dispatch a parsed evaluation command."""

    command = args.benchmark_command
    if command == "list":
        return _handle_list(args.format)
    if command == "doctor":
        from kairyu.evaluation.adapters import get_adapter

        report = get_adapter(args.benchmark_id).doctor(
            args.profile,
            dataset_path=args.dataset_path,
        )
        _emit(report.model_dump(mode="json"), args.format)
        return 0 if report.runnable else 2
    if command == "prepare":
        from kairyu.evaluation.adapters import get_adapter

        result = get_adapter(args.benchmark_id).prepare(
            args.profile,
            dry_run=args.dry_run,
            dataset_path=args.dataset_path,
            dataset_sha256=args.dataset_sha256,
            accepted_access=args.accepted_access,
        )
        _emit(result.model_dump(mode="json"), args.format)
        return 0 if result.status.value in {"ready", "dry_run"} else 2
    if command == "plan":
        selection = _selection_from_args(args)
        # CLI guard precedes concrete adapter import.
        _guard_selection(selection)
        from kairyu.evaluation.adapters import get_adapter

        adapter = get_adapter(args.benchmark_id)
        plan = bind_connector_to_plan(
            adapter.build_run_plan(selection),
            _connector_from_args(args),
            judge_connector=_judge_connector_from_args(args),
        )
        _emit(_plan_payload(plan, adapter.metadata()), args.format)
        return 0
    if command == "run":
        selection = _selection_from_args(args)
        _guard_selection(selection)
        connector = _connector_from_args(args)
        judge_connector = _judge_connector_from_args(args)
        if selection.mode is RunMode.FULL:
            from kairyu.evaluation.adapters import get_adapter

            adapter = get_adapter(args.benchmark_id)
            preview = bind_connector_to_plan(
                adapter.build_run_plan(selection),
                connector,
                judge_connector=judge_connector,
            )
            _emit_full_run_preflight(
                _preflight_payload(preview, adapter.metadata()),
                args.format,
            )
        runtime = EvaluationRuntime(args.state_dir)
        service = BenchmarkService(runtime)
        submitted = service.submit(
            args.benchmark_id,
            selection,
            connector,
            judge_connector=judge_connector,
        )
        payload = submitted.model_dump(mode="json")
        if args.wait:
            run_worker_once(
                runtime,
                worker_id=f"worker-{submitted.run.run_id[-16:]}",
            )
            payload["run"] = runtime.store.get_run(submitted.run.run_id).run.model_dump(mode="json")
            payload["report_paths"] = ["report.json", "report.md", "report.html"]
        _emit(payload, args.format)
        return 0
    if command == "worker":
        runtime = EvaluationRuntime(args.state_dir)
        run_id = run_worker_once(
            runtime,
            worker_id=args.worker_id,
            lease_seconds=args.lease_seconds,
        )
        _emit({"claimed_run_id": run_id}, args.format)
        return 0
    if command == "status":
        runtime = EvaluationRuntime(args.state_dir)
        stored = runtime.store.get_run(args.run_id)
        payload = stored.run.model_dump(mode="json")
        payload["version"] = stored.version
        payload["items"] = [
            item.item.model_dump(mode="json") for item in _list_all_run_items(runtime, args.run_id)
        ]
        payload["artifacts"] = [
            artifact.model_dump(mode="json")
            for artifact in runtime.store.list_artifacts(args.run_id)
        ]
        _emit(payload, args.format)
        return 0
    if command == "cancel":
        runtime = EvaluationRuntime(args.state_dir)
        job = BenchmarkService(runtime).cancel(args.run_id)
        _emit(_job_payload(job), args.format)
        return 0
    if command == "resume":
        runtime = EvaluationRuntime(args.state_dir)
        result = BenchmarkService(runtime).resume(
            args.run_id,
            new_run_id=args.new_run_id,
        )
        _emit(
            {
                "job": _job_payload(result.job),
                "run": result.run.run.model_dump(mode="json"),
            },
            args.format,
        )
        return 0
    if command == "report":
        runtime = EvaluationRuntime(args.state_dir)
        report = render_saved_report(runtime, args.run_id)
        _emit(report.model_dump(mode="json"), args.format)
        return 0
    if command == "references" and args.reference_command == "list":
        snapshot = load_reference_snapshot(benchmark_id=args.benchmark)
        _emit(snapshot.model_dump(mode="json"), args.format)
        return 0
    raise ValueError(f"unknown benchmark command {command!r}")


def _handle_list(output_format: str) -> int:
    entries = benchmark_catalog()
    if output_format == "json":
        _emit([entry.model_dump(mode="json") for entry in entries], output_format)
        return 0

    print(f"evaluation benchmark catalog ({len(entries)} entries)")
    for entry in entries:
        print(
            f"  {entry.benchmark_id:25s} {entry.display_name} "
            f"— {entry.primary_metric} [{entry.implementation_status.value}]"
        )
    return 0


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
    )


def _add_state_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", type=Path, default=_DEFAULT_STATE_DIR)


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--mode", choices=tuple(RunMode), default=RunMode.SMOKE.value)
    parser.add_argument("--model", default="kairyu-synthetic-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", "--sample-ids", dest="sample_ids", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--judge-temperature", type=float)
    parser.add_argument("--judge-top-p", type=float)
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument("--judge-timeout-seconds", type=float)
    parser.add_argument("--judge-reasoning-effort")
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--accepted-access", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")


def _add_connector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--connector", choices=("fake", "openai"), default="fake")
    parser.add_argument("--endpoint")
    parser.add_argument("--secret-env-name")
    parser.add_argument("--max-response-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--judge-connector", choices=("fake", "openai"))
    parser.add_argument("--judge-endpoint")
    parser.add_argument("--judge-secret-env-name")
    parser.add_argument("--judge-max-response-bytes", type=int, default=1_048_576)
    parser.add_argument("--judge-max-retries", type=int, default=2)


def _selection_from_args(args: argparse.Namespace) -> RunSelection:
    judge_generation_parameters = {
        key: value
        for key, value in {
            "max_tokens": args.judge_max_tokens,
            "reasoning_effort": args.judge_reasoning_effort,
            "temperature": args.judge_temperature,
            "timeout_seconds": args.judge_timeout_seconds,
            "top_p": args.judge_top_p,
        }.items()
        if value is not None
    }
    return RunSelection(
        profile=args.profile,
        mode=RunMode(args.mode),
        target_model=args.model,
        judge_model=args.judge_model,
        limit=args.limit,
        sample_ids=tuple(args.sample_ids or ()),
        seed=args.seed,
        confirm_full_run=args.confirm_full_run,
        dataset_path=args.dataset_path,
        dataset_sha256=args.dataset_sha256,
        accepted_access=args.accepted_access,
        generation_parameters={
            "max_tokens": args.max_tokens,
            "repeats": 1,
            "temperature": args.temperature,
            "timeout_seconds": args.timeout_seconds,
            "top_p": args.top_p,
        },
        judge_generation_parameters=judge_generation_parameters,
    )


def _guard_selection(selection: RunSelection) -> None:
    validate_run_guard(
        selection.mode,
        confirm_full_run=selection.confirm_full_run,
        limit=selection.limit,
        sample_ids=selection.sample_ids,
    )


def _connector_from_args(args: argparse.Namespace) -> ConnectorConfig:
    return ConnectorConfig(
        kind=args.connector,
        endpoint=args.endpoint,
        secret_env_name=args.secret_env_name,
        max_response_bytes=args.max_response_bytes,
        max_retries=args.max_retries,
    )


def _judge_connector_from_args(args: argparse.Namespace) -> ConnectorConfig | None:
    configured = (
        args.judge_connector is not None
        or args.judge_endpoint is not None
        or args.judge_secret_env_name is not None
    )
    if not configured:
        return None
    if args.judge_connector is None:
        raise ValueError("judge endpoint or credential requires --judge-connector")
    return ConnectorConfig(
        kind=args.judge_connector,
        endpoint=args.judge_endpoint,
        secret_env_name=args.judge_secret_env_name,
        max_response_bytes=args.judge_max_response_bytes,
        max_retries=args.judge_max_retries,
    )


def _plan_payload(plan, metadata) -> dict:
    return {
        "benchmark_id": plan.benchmark_id,
        "estimated_model_calls": plan.estimated_model_calls,
        "effective_retries": plan.protocol.retries,
        "maximum_model_calls": plan.estimate.maximum_model_calls,
        "estimate": plan.estimate.model_dump(mode="json"),
        "execution": plan.execution.model_dump(mode="json"),
        "expected_full_count": plan.expected_full_count,
        "item_input_manifest_sha256": plan.item_input_manifest_sha256,
        "mode": plan.mode.value,
        "official_eligible": plan.official_eligible,
        "preflight": _preflight_payload(plan, metadata),
        "profile": plan.profile,
        "protocol_hash": plan.protocol_hash,
        "required_resources": plan.resources.model_dump(mode="json"),
        "selected_item_ids": [item.item_id for item in plan.items],
        "target_model": plan.target_model,
    }


def _preflight_payload(plan, metadata) -> dict:
    protocol = plan.protocol.model_dump(mode="json")
    reproducibility = protocol["adapter_configuration"].get(
        "reproducibility_evidence",
        {},
    )
    return {
        "benchmark_id": plan.benchmark_id,
        "profile": plan.profile,
        "mode": plan.mode.value,
        "problem_count": len(plan.items),
        "estimated_api_calls": plan.estimate.model_calls,
        "effective_retries": plan.protocol.retries,
        "maximum_api_calls": plan.estimate.maximum_model_calls,
        "estimated_input_tokens": plan.estimate.estimated_input_tokens,
        "maximum_output_tokens": plan.estimate.maximum_output_tokens,
        "estimated_cost_usd": plan.estimate.estimated_cost_usd,
        "estimated_duration_seconds": plan.estimate.estimated_duration_seconds,
        "maximum_duration_seconds": plan.estimate.maximum_duration_seconds,
        "estimate_assumptions": list(plan.estimate.assumptions),
        "required_resources": plan.resources.model_dump(mode="json"),
        "execution": plan.execution.model_dump(mode="json"),
        "models": {
            "target": plan.target_model,
            "judge": plan.protocol.judge_model,
            "simulator": plan.protocol.simulator_model,
        },
        "data_licenses": list(metadata.licenses),
        "cancellation_supported": True,
        "resume_supported": metadata.supports_resume,
        "official_eligible": plan.official_eligible,
        "score_claim": (
            "Eligible for a formal full-run report; this is not a reproduced leaderboard claim."
            if plan.official_eligible
            else "Unofficial: unresolved evidence or non-full scope prevents a formal score claim."
        ),
        "unresolved_reproducibility_evidence": {
            key: value
            for key, value in reproducibility.items()
            if value.get("status") != "verified"
        },
    }


def _emit_full_run_preflight(payload: dict, output_format: str) -> None:
    envelope = {"full_run_preflight": payload}
    if output_format == "human":
        print("FULL RUN PREFLIGHT (before enqueue)", file=sys.stderr)
    print(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _job_payload(job) -> dict:
    return {
        "attempts": job.attempts,
        "cancel_requested": job.cancel_requested,
        "job_id": job.job_id,
        "run_id": job.run_id,
        "status": job.status,
    }


def _list_all_run_items(runtime: EvaluationRuntime, run_id: str):
    items = []
    after_ordinal = -1
    while True:
        page = runtime.store.list_run_items(
            run_id,
            after_ordinal=after_ordinal,
            limit=1_000,
        )
        if not page:
            return tuple(items)
        items.extend(page)
        after_ordinal = page[-1].item.ordinal


def _emit(payload, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if isinstance(payload, list):
        for item in payload:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
