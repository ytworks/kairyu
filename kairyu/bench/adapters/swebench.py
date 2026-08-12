"""Shared, fail-closed evidence contracts for official SWE-bench adapters."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
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

    if ids["resolved_ids"] & ids["unresolved_ids"]:
        raise ValueError("resolved_ids and unresolved_ids must be disjoint")
    if ids["completed_ids"] != ids["resolved_ids"] | ids["unresolved_ids"]:
        raise ValueError("completed_ids must equal resolved_ids plus unresolved_ids")

    terminal = (ids["completed_ids"], ids["empty_patch_ids"], ids["error_ids"])
    if any(
        left & right
        for index, left in enumerate(terminal)
        for right in terminal[index + 1 :]
    ):
        raise ValueError("submitted outcome IDs must be disjoint")
    if ids["submitted_ids"] != set().union(*terminal):
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
        patch = prediction.get("model_patch")
        if patch is not None and not isinstance(patch, str):
            raise ValueError(f"prediction {instance_id!r} has an invalid model_patch")
    return tuple(sorted(predictions))


def find_swebench_report(workdir: Path) -> dict:
    """Return the sole root schema-v2 report produced by the official harness."""

    candidates = []
    for path in sorted(workdir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("schema_version") == 2:
            candidates.append(data)
    if not candidates:
        raise ValueError("no SWE-bench schema-v2 report produced")
    if len(candidates) > 1:
        raise ValueError("multiple SWE-bench schema-v2 reports produced")
    return candidates[0]


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

        env = dict(os.environ)
        base = normalize_base_url(target.base_url)
        env["OPENAI_BASE_URL"] = base
        env["OPENAI_API_BASE"] = base
        env["OPENAI_API_KEY"] = target_api_key(target) or "sk-local"

        def invoke(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
            return subprocess.run(
                command,
                capture_output=True,
                timeout=_STAGE_TIMEOUT_S,
                env=env,
                cwd=cwd,
                check=False,
            )

        with tempfile.TemporaryDirectory(prefix=f"kairyu-{self.info.name}-") as tmp:
            workdir = Path(tmp)
            generation_dir = workdir / "mini-output"
            predictions = generation_dir / "preds.json"
            generate_command = self._generate_command(target, ctx, generation_dir)
            failure = await self._invoke_stage(
                target,
                started_at,
                "generate",
                generate_command,
                workdir,
                invoke,
            )
            if failure is not None:
                return failure
            if not predictions.is_file():
                return self._failed(
                    target,
                    started_at,
                    "generate stage produced no mini-output/preds.json file",
                )
            try:
                instance_ids = load_prediction_ids(predictions)
            except ValueError as error:
                return self._failed(
                    target,
                    started_at,
                    f"invalid mini-output/preds.json prediction file: {error}",
                )

            evaluate_command = self._evaluate_command(
                predictions,
                instance_ids,
                self._run_id(ctx),
                workdir,
                ctx.concurrency,
            )
            failure = await self._invoke_stage(
                target,
                started_at,
                "evaluate",
                evaluate_command,
                workdir,
                invoke,
            )
            if failure is not None:
                return failure
            try:
                report = find_swebench_report(workdir)
                items, total = parse_swebench_report(
                    report,
                    selected_ids=instance_ids,
                )
            except ValueError as error:
                return self._failed(
                    target,
                    started_at,
                    f"invalid evaluation report: {error}",
                )

        resolved = report["resolved_instances"]
        pair = summarize_items(
            self.info.name,
            target.label(),
            items,
            methodology={
                "metric": self.info.metric,
                "dataset": self.spec.dataset,
                "subset": self.spec.subset,
                "split": "test",
                "scaffold": "mini-swe-agent",
                "base_config": _BASE_CONFIG,
                "step_limit": self.spec.step_limit,
                "concurrency": ctx.concurrency,
                "selected_instance_ids": list(instance_ids),
                "official_report": report,
                "evaluation": "swebench.harness.run_evaluation (docker)",
                "generate_command": shlex.join(generate_command),
                "evaluate_command": shlex.join(evaluate_command),
            },
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
    ) -> PairResult | None:
        try:
            completed = await asyncio.to_thread(invoke, command, workdir)
        except subprocess.TimeoutExpired:
            return self._failed(
                target,
                started_at,
                f"{stage} stage timed out ({_STAGE_TIMEOUT_S}s)",
            )
        except OSError as error:
            return self._failed(
                target,
                started_at,
                f"{stage} stage could not start: {error}",
            )
        if completed.returncode != 0:
            stderr = completed.stderr.decode(errors="replace")[-500:]
            return self._failed(
                target,
                started_at,
                f"{stage} stage failed (rc={completed.returncode}): {stderr}",
            )
        return None

    def _failed(
        self,
        target: BenchTarget,
        started_at: str,
        reason: str,
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
    "parse_swebench_report",
]
