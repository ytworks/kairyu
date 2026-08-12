"""Shared, fail-closed evidence contracts for official SWE-bench adapters."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    RunContext,
    external_harness_sampling_incompatibility,
    normalize_base_url,
    skipped_pair,
    summarize_items,
    target_api_key,
    utc_now,
)
from kairyu.bench.hub import load_hf_rows
from kairyu.bench.types import (
    BenchTarget,
    DownloadReport,
    ItemResult,
    PairResult,
)

_STAGE_TIMEOUT_S = 8 * 3600
_BASE_CONFIG = "swebench.yaml"
_SAFE_RUN_ID = re.compile(r"[^A-Za-z0-9_.-]+")

_COUNT_TO_IDS = {
    "submitted_instances": "submitted_ids",
    "completed_instances": "completed_ids",
    "resolved_instances": "resolved_ids",
    "unresolved_instances": "unresolved_ids",
    "empty_patch_instances": "empty_patch_ids",
    "error_instances": "error_ids",
}


def _required_count(report: dict, name: str) -> int:
    value = report.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_id_set(report: dict, name: str) -> set[str]:
    value = report.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate IDs")
    return set(value)


def _selected_id_set(selected_ids: Sequence[str] | None) -> set[str] | None:
    if selected_ids is None:
        return None
    if any(not isinstance(item, str) or not item for item in selected_ids):
        raise ValueError("selected IDs must be non-empty strings")
    selected = set(selected_ids)
    if len(selected) != len(selected_ids):
        raise ValueError("selected IDs contain duplicates")
    return selected


def parse_swebench_report(
    report: dict,
    *,
    selected_ids: Sequence[str] | None = None,
) -> tuple[list[ItemResult], int]:
    """Validate schema-v2 partitions and retain every official outcome."""

    if not isinstance(report, dict):
        raise ValueError("SWE-bench report must be an object")
    if report.get("schema_version") != 2:
        raise ValueError("SWE-bench report schema_version must be 2")

    counts = {
        name: _required_count(report, name)
        for name in ("total_instances", *_COUNT_TO_IDS)
    }
    ids = {
        name: _required_id_set(report, name)
        for name in (*_COUNT_TO_IDS.values(), "incomplete_ids")
    }
    for count_name, ids_name in _COUNT_TO_IDS.items():
        if counts[count_name] != len(ids[ids_name]):
            raise ValueError(f"{count_name} does not match {ids_name}")

    resolved_outcomes = ids["resolved_ids"] | ids["unresolved_ids"]
    if ids["resolved_ids"] & ids["unresolved_ids"]:
        raise ValueError("resolved_ids and unresolved_ids must be disjoint")
    if resolved_outcomes & ids["error_ids"]:
        raise ValueError(
            "submitted outcome IDs must be disjoint except for the official "
            "completed/error overlap"
        )
    if ids["completed_ids"] & ids["empty_patch_ids"] or ids["error_ids"] & ids[
        "empty_patch_ids"
    ]:
        raise ValueError(
            "submitted outcome IDs must be disjoint except for the official "
            "completed/error overlap"
        )
    # SWE-bench 4.1 adds an ID to completed_ids as soon as its report file
    # exists, then also to error_ids when that file is empty or malformed.
    # Every non-error completed report must still have exactly one verdict.
    if ids["completed_ids"] - ids["error_ids"] != resolved_outcomes:
        raise ValueError(
            "completed_ids minus error_ids must equal resolved_ids plus unresolved_ids"
        )

    submitted_outcomes = (
        ids["completed_ids"] | ids["empty_patch_ids"] | ids["error_ids"]
    )
    if ids["submitted_ids"] != submitted_outcomes:
        raise ValueError(
            "submitted_ids must equal completed, empty-patch, and error IDs"
        )
    if ids["submitted_ids"] & ids["incomplete_ids"]:
        raise ValueError("submitted_ids and incomplete_ids must be disjoint")

    official_ids = ids["submitted_ids"] | ids["incomplete_ids"]
    selected = (
        official_ids if selected_ids is None else _selected_id_set(selected_ids)
    )
    assert selected is not None
    if selected != official_ids:
        raise ValueError("selected IDs must equal submitted_ids plus incomplete_ids")
    if counts["total_instances"] != len(selected):
        raise ValueError("total_instances does not match selected IDs")

    items = []
    for item_id in sorted(selected):
        if item_id in ids["resolved_ids"]:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="completed",
                    score=1.0,
                    details={"outcome": "resolved"},
                )
            )
        elif item_id in ids["unresolved_ids"]:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="completed",
                    score=0.0,
                    details={"outcome": "unresolved"},
                )
            )
        elif item_id in ids["empty_patch_ids"]:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="completed",
                    score=0.0,
                    details={"outcome": "empty_patch"},
                )
            )
        elif item_id in ids["error_ids"]:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="failed",
                    error="SWE-bench harness error",
                    details={"outcome": "error"},
                )
            )
        else:
            items.append(
                ItemResult(
                    item_id=item_id,
                    status="failed",
                    error="SWE-bench evaluation incomplete",
                    details={"outcome": "incomplete"},
                )
            )
    return items, counts["total_instances"]


def load_prediction_ids(path: Path) -> tuple[str, ...]:
    """Read mini-SWE-agent's prediction mapping and validate its row identity."""

    try:
        predictions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"prediction file is not readable JSON: {error}") from error
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("prediction file must be a non-empty object")

    for instance_id, prediction in predictions.items():
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("prediction IDs must be non-empty strings")
        if not isinstance(prediction, dict):
            raise ValueError(f"prediction {instance_id!r} must be an object")
        if prediction.get("instance_id") != instance_id:
            raise ValueError(f"prediction {instance_id!r} has a mismatched instance_id")
        model = prediction.get("model_name_or_path")
        if not isinstance(model, str) or not model:
            raise ValueError(f"prediction {instance_id!r} has an invalid model_name_or_path")
        if "model_patch" not in prediction:
            raise ValueError(f"prediction {instance_id!r} has no model_patch")
        patch = prediction["model_patch"]
        if patch is not None and not isinstance(patch, str):
            raise ValueError(f"prediction {instance_id!r} has an invalid model_patch")
    return tuple(sorted(predictions))


def load_selected_instance_ids(dataset: str, limit: int | None) -> tuple[str, ...]:
    """Load the exact official split order before generation fixes the denominator."""

    rows = load_hf_rows(dataset, split="test")
    if limit is not None:
        rows = rows[:limit]
    instance_ids = []
    for row in rows:
        instance_id = row.get("instance_id") if isinstance(row, dict) else None
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("SWE-bench dataset contains an invalid instance_id")
        instance_ids.append(instance_id)
    if not instance_ids:
        raise ValueError("SWE-bench selection is empty")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("SWE-bench selection contains duplicate instance IDs")
    return tuple(instance_ids)


def find_swebench_report(workdir: Path) -> dict:
    """Return the sole root schema-v2 report produced by the official harness."""

    return _find_swebench_report(workdir)[1]


def _find_swebench_report(workdir: Path) -> tuple[Path, dict]:
    candidates: list[tuple[Path, dict]] = []
    for path in sorted(workdir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("schema_version") == 2:
            candidates.append((path, data))
    if not candidates:
        raise ValueError("no SWE-bench schema-v2 report produced")
    if len(candidates) > 1:
        raise ValueError("multiple SWE-bench schema-v2 reports produced")
    return candidates[0]


def _safe_component(value: str) -> str:
    prefix = _SAFE_RUN_ID.sub("-", value).strip("-.")[:80] or "value"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _redact_output(value: bytes | str | None, secrets: Sequence[str]) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = value
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def _write_stage_logs(
    workdir: Path,
    stage: str,
    stdout: bytes | str | None,
    stderr: bytes | str | None,
    secrets: Sequence[str],
) -> dict[str, str]:
    log_dir = workdir / "stage-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for stream, value in (("stdout", stdout), ("stderr", stderr)):
        path = log_dir / f"{stage}.{stream}.log"
        path.write_text(_redact_output(value, secrets), encoding="utf-8")
        paths[f"{stage}_{stream}"] = str(path)
    return paths


def _persist_selection(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"stored selection manifest is unreadable: {error}") from error
        if existing != payload:
            raise ValueError("stored selection manifest does not match this run")
        return
    path.write_text(text, encoding="utf-8")


def _model_kwargs(target: BenchTarget) -> dict[str, object]:
    fields = {
        "reasoning_effort": target.reasoning_effort,
        "top_p": target.top_p,
        "seed": target.seed,
    }
    return {name: value for name, value in fields.items() if value is not None}


def harness_missing() -> str | None:
    for module, package in (
        ("minisweagent", "mini-swe-agent"),
        ("swebench", "swebench"),
    ):
        if importlib.util.find_spec(module) is None:
            return package
    if shutil.which("mini-extra") is None:
        return "mini-swe-agent executable"
    return None


def unsupported_platform() -> str | None:
    system = platform.system()
    machine = platform.machine()
    if system == "Linux" and machine.lower() in {"x86_64", "amd64"}:
        return None
    return f"requires a Linux x86-64 host; detected {system or 'unknown'} {machine or 'unknown'}"


@dataclass(frozen=True)
class SweBenchSpec:
    name: str
    display_name: str
    subset: str
    dataset: str
    step_limit: int
    annotations: tuple[str, ...]
    comparable_to_published: bool = True
    incomparable_reason: str = ""


class SweBenchAdapter:
    """Official mini-SWE-agent generation followed by SWE-bench Docker eval."""

    def __init__(self, spec: SweBenchSpec) -> None:
        self.spec = spec
        self.info = AdapterInfo(
            name=spec.name,
            display_name=spec.display_name,
            metric="resolved rate",
            binary_outcomes=True,
            hf_dataset=spec.dataset,
            needs_docker=True,
            agentic=True,
            annotations=spec.annotations,
            comparable_to_published=spec.comparable_to_published,
            incomparable_reason=spec.incomparable_reason,
            evaluation_distributions=("mini-swe-agent", "swebench"),
            evaluation_executables=("mini-extra",),
            history_provenance_complete=False,
            history_provenance_reason=(
                "the harness fetches a mutable remote dataset and task container "
                "images without resolved content identities"
            ),
        )

    def download(self, ctx: DownloadContext) -> DownloadReport:
        return DownloadReport(
            adapter=self.info.name,
            status="ok",
            detail="instances and images are fetched by the SWE-bench harness at run time",
        )

    def _preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        sampling_reason = external_harness_sampling_incompatibility(
            target,
            harness="mini-swe-agent",
        )
        if sampling_reason is not None:
            return sampling_reason
        if ctx.attempts != 1:
            return (
                "mini-swe-agent has no verified --attempts passthrough; "
                f"{self.info.display_name} requires attempts=1"
            )
        available, reason = ctx.docker
        if not available:
            return reason
        platform_reason = unsupported_platform()
        if platform_reason is not None:
            return platform_reason
        missing = harness_missing()
        if missing is not None:
            return f"{missing} not installed (pip install 'kairyu[bench-agentic]')"
        return None

    def _generate_command(
        self,
        target: BenchTarget,
        ctx: RunContext,
        output_dir: Path,
    ) -> list[str]:
        command = [
            "mini-extra",
            "swebench",
            "--model",
            f"openai/{target.model}",
            "--subset",
            self.spec.subset,
            "--split",
            "test",
            "--output",
            str(output_dir),
            "--workers",
            str(ctx.concurrency),
            "--config",
            _BASE_CONFIG,
            "--config",
            f"agent.step_limit={self.spec.step_limit}",
        ]
        for field, value in _model_kwargs(target).items():
            command += ["--config", f"model.model_kwargs.{field}={value}"]
        if ctx.limit is not None:
            command += ["--slice", f"0:{ctx.limit}"]
        return command

    def _evaluate_command(
        self,
        predictions: Path,
        instance_ids: Sequence[str],
        run_id: str,
        report_dir: Path,
        workers: int,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.spec.dataset,
            "--split",
            "test",
            "--predictions_path",
            str(predictions),
            "--max_workers",
            str(workers),
            "--run_id",
            run_id,
            "--report_dir",
            str(report_dir),
            "--instance_ids",
            *instance_ids,
        ]

    def _run_id(self, ctx: RunContext) -> str:
        source = ctx.run_id or "bench"
        sanitized = _SAFE_RUN_ID.sub("-", source).strip("-.") or "bench"
        return f"kairyu-{sanitized}-{self.info.name}"

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        started_at = utc_now()
        reason = self._preconditions(target, ctx)
        incomparable_reasons = (
            (self.info.incomparable_reason,)
            if not self.info.comparable_to_published
            else ()
        )
        if reason is not None:
            return skipped_pair(
                self.info.name,
                target.label(),
                reason,
                annotations=self.info.annotations,
                comparable=self.info.comparable_to_published,
                incomparable_reasons=incomparable_reasons,
            )

        artifact_root = ctx.artifacts_dir or (
            ctx.cache.root / "run-artifacts" / (ctx.run_id or "direct")
        )
        stable_workdir = (
            Path(artifact_root) / self.info.name / _safe_component(target.label())
        ).resolve()
        if ctx.rerun:
            stable_workdir.mkdir(parents=True, exist_ok=True)
            workdir = Path(
                tempfile.mkdtemp(prefix="rerun-", dir=stable_workdir)
            ).resolve()
        else:
            workdir = stable_workdir
        workdir.mkdir(parents=True, exist_ok=True)
        generation_dir = workdir / "mini-output"
        predictions = generation_dir / "preds.json"
        selection_path = workdir / "selected-instances.json"
        artifacts: dict[str, object] = {
            "root": str(workdir),
            "selection": str(selection_path),
            "generation_dir": str(generation_dir),
            "predictions": str(predictions),
            "mini_swe_agent_log": str(generation_dir / "minisweagent.log"),
            "evaluation_logs": str(workdir / "logs" / "run_evaluation"),
            "official_report": None,
            "stage_logs": {},
        }
        methodology = {
            "metric": self.info.metric,
            "dataset": self.spec.dataset,
            "subset": self.spec.subset,
            "split": "test",
            "scaffold": "mini-swe-agent",
            "base_config": _BASE_CONFIG,
            "step_limit": self.spec.step_limit,
            "concurrency": ctx.concurrency,
            "evaluation": "swebench.harness.run_evaluation (docker)",
            "artifacts": artifacts,
        }
        try:
            selected_ids = load_selected_instance_ids(self.spec.dataset, ctx.limit)
            selection = {
                "schema_version": 1,
                "dataset": self.spec.dataset,
                "split": "test",
                "limit": ctx.limit,
                "instance_ids": list(selected_ids),
            }
            _persist_selection(selection_path, selection)
        except Exception as error:  # noqa: BLE001 - dataset backends vary
            return self._failed(
                target,
                started_at,
                f"could not establish SWE-bench selection: {error}",
                methodology=methodology,
            )
        methodology["selected_instance_ids"] = list(selected_ids)

        env = dict(os.environ)
        base = normalize_base_url(target.base_url)
        env["OPENAI_BASE_URL"] = base
        env["OPENAI_API_BASE"] = base
        api_key = target_api_key(target)
        env["OPENAI_API_KEY"] = api_key or "sk-local"
        secrets = (api_key,) if api_key else ()

        def invoke(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
            return subprocess.run(
                command,
                capture_output=True,
                timeout=_STAGE_TIMEOUT_S,
                env=env,
                cwd=cwd,
                check=False,
            )

        generate_command = self._generate_command(target, ctx, generation_dir)
        methodology["generate_command"] = shlex.join(generate_command)
        failure = await self._invoke_stage(
            target,
            started_at,
            "generate",
            generate_command,
            workdir,
            invoke,
            methodology,
            secrets,
        )
        if failure is not None:
            return failure
        if not predictions.is_file():
            return self._failed(
                target,
                started_at,
                "generate stage produced no mini-output/preds.json file",
                methodology=methodology,
            )
        try:
            prediction_ids = load_prediction_ids(predictions)
        except ValueError as error:
            return self._failed(
                target,
                started_at,
                f"invalid mini-output/preds.json prediction file: {error}",
                methodology=methodology,
            )
        unexpected_ids = sorted(set(prediction_ids) - set(selected_ids))
        if unexpected_ids:
            return self._failed(
                target,
                started_at,
                "mini-output/preds.json contains IDs outside the persisted selection: "
                + ", ".join(unexpected_ids),
                methodology=methodology,
            )
        methodology["prediction_instance_ids"] = list(prediction_ids)
        methodology["missing_prediction_ids"] = sorted(
            set(selected_ids) - set(prediction_ids)
        )

        evaluate_command = self._evaluate_command(
            predictions,
            selected_ids,
            self._run_id(ctx),
            workdir,
            ctx.concurrency,
        )
        methodology["evaluate_command"] = shlex.join(evaluate_command)
        failure = await self._invoke_stage(
            target,
            started_at,
            "evaluate",
            evaluate_command,
            workdir,
            invoke,
            methodology,
            secrets,
        )
        if failure is not None:
            return failure
        try:
            report_path, report = _find_swebench_report(workdir)
            artifacts["official_report"] = str(report_path)
            items, total = parse_swebench_report(
                report,
                selected_ids=selected_ids,
            )
        except ValueError as error:
            return self._failed(
                target,
                started_at,
                f"invalid evaluation report: {error}",
                methodology=methodology,
            )

        resolved = report["resolved_instances"]
        methodology["official_report"] = report
        pair = summarize_items(
            self.info.name,
            target.label(),
            items,
            methodology=methodology,
            annotations=self.info.annotations,
            started_at=started_at,
            score_fn=lambda _: resolved / total if total else None,
            comparable=self.info.comparable_to_published,
            incomparable_reasons=incomparable_reasons,
        )
        category_metrics = {
            "n_resolved": report["resolved_instances"],
            "n_unresolved": report["unresolved_instances"],
            "n_empty_patch": report["empty_patch_instances"],
            "n_error": report["error_instances"],
            "n_incomplete": len(report["incomplete_ids"]),
        }
        return pair.model_copy(update={"metrics": {**pair.metrics, **category_metrics}})

    async def _invoke_stage(
        self,
        target: BenchTarget,
        started_at: str,
        stage: str,
        command: list[str],
        workdir: Path,
        invoke,
        methodology: dict,
        secrets: Sequence[str],
    ) -> PairResult | None:
        try:
            completed = await asyncio.to_thread(invoke, command, workdir)
        except subprocess.TimeoutExpired as error:
            stage_logs = _write_stage_logs(
                workdir,
                stage,
                error.stdout,
                error.stderr,
                secrets,
            )
            methodology["artifacts"]["stage_logs"].update(stage_logs)
            return self._failed(
                target,
                started_at,
                f"{stage} stage timed out ({_STAGE_TIMEOUT_S}s)",
                methodology=methodology,
            )
        except OSError as error:
            return self._failed(
                target,
                started_at,
                f"{stage} stage could not start: {error}",
                methodology=methodology,
            )
        stage_logs = _write_stage_logs(
            workdir,
            stage,
            completed.stdout,
            completed.stderr,
            secrets,
        )
        methodology["artifacts"]["stage_logs"].update(stage_logs)
        if completed.returncode != 0:
            stderr = _redact_output(completed.stderr, secrets)[-500:]
            return self._failed(
                target,
                started_at,
                f"{stage} stage failed (rc={completed.returncode}): {stderr}",
                methodology=methodology,
            )
        return None

    def _failed(
        self,
        target: BenchTarget,
        started_at: str,
        reason: str,
        *,
        methodology: dict | None = None,
    ) -> PairResult:
        incomparable_reasons = (
            (self.info.incomparable_reason,)
            if not self.info.comparable_to_published
            else ()
        )
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="failed",
            reason=reason,
            metrics={"score": None, "n_total": 0},
            methodology=methodology or {},
            annotations=self.info.annotations,
            comparable=self.info.comparable_to_published,
            incomparable_reasons=incomparable_reasons,
            started_at=started_at,
            finished_at=utc_now(),
        )


__all__ = [
    "SweBenchAdapter",
    "SweBenchSpec",
    "find_swebench_report",
    "harness_missing",
    "load_prediction_ids",
    "load_selected_instance_ids",
    "parse_swebench_report",
]
