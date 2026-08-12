"""SWE-Bench Pro: official two-stage flow — mini-swe-agent patches, docker eval.

Stage 1 points mini-swe-agent's OpenAI-compatible client at the target
gateway to generate patches for the public split; stage 2 runs ScaleAI's
pinned SWE-bench Pro evaluator with local Docker and translates its report.
Docker or the [bench-agentic] extra missing -> skipped, never a crash.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    RunContext,
    adapter_cache_ready,
    cache_pins,
    external_harness_sampling_incompatibility,
    normalize_base_url,
    skipped_pair,
    summarize_items,
    target_api_key,
    utc_now,
)
from kairyu.bench.types import (
    BenchTarget,
    DatasetUnavailable,
    DownloadReport,
    ItemResult,
    PairResult,
)

_STAGE_TIMEOUT_S = 72 * 3600
_DATASET = "ScaleAI/SWE-bench_Pro"
_DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
_DATASET_ROWS = 731
_EVALUATOR_REPOSITORY = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
_EVALUATOR_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
_DOCKERHUB_USERNAME = "jefzda"
# Fugu runs mini-swe-agent with a 1,000-turn budget; the harness default
# (config/benchmarks/swebench.yaml) is agent.step_limit: 250, so a long trace
# would be cut off four times earlier than the published condition.
_STEP_LIMIT = 1000
# mini-swe-agent merges -c specs recursively but DROPS its default config as
# soon as -c is given, so the default file has to be restated by name.
_BASE_CONFIG = "swebench_backticks.yaml"
_MODEL_CLASS = "kairyu.bench.mini_swe_agent.OpenAICompatTextbasedModel"
_PULL_TIMEOUT_S = 30 * 60
_AGENT_OUTCOMES = frozenset(
    {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"}
)


def _model_kwargs(target: BenchTarget) -> dict[str, object]:
    """Named sampling fields the harness can forward to its LLM client."""
    fields = {
        # mini-swe-agent otherwise omits max_tokens. OpenAI-compatible
        # /completions endpoints commonly default that omission to 16 tokens,
        # which can truncate both ordinary answers and private-reasoning
        # prefixes before their public answer boundary.
        "max_tokens": target.max_output_tokens,
        "reasoning_effort": target.reasoning_effort,
        "top_p": target.top_p,
        "seed": target.seed,
    }
    return {name: value for name, value in fields.items() if value is not None}


def harness_missing() -> str | None:
    for module, package in (
        ("minisweagent", "mini-swe-agent"),
        ("docker", "docker"),
        ("pandas", "pandas"),
    ):
        if importlib.util.find_spec(module) is None:
            return package
    return None


def mini_extra_executable() -> str:
    """Resolve the console script when Python's venv is not on ``PATH``."""
    executable = shutil.which("mini-extra")
    if executable is not None:
        return executable
    return str(Path(sys.executable).with_name("mini-extra"))


def _evaluator_root(cache) -> Path:
    override = os.environ.get("KAIRYU_SWE_BENCH_PRO_EVAL_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return cache.assets_dir("swe-bench-pro") / "SWE-bench_Pro-os"


def _git_head(path: Path) -> str | None:
    if not (path / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _evaluator_ready(cache) -> bool:
    root = _evaluator_root(cache)
    return (
        _git_head(root) == _EVALUATOR_REVISION
        and (root / "swe_bench_pro_eval.py").is_file()
        and (root / "run_scripts").is_dir()
    )


def _ensure_evaluator(cache) -> Path:
    root = _evaluator_root(cache)
    if _evaluator_ready(cache):
        return root
    if os.environ.get("KAIRYU_SWE_BENCH_PRO_EVAL_PATH"):
        raise DatasetUnavailable(
            "KAIRYU_SWE_BENCH_PRO_EVAL_PATH must be a git checkout at "
            f"{_EVALUATOR_REVISION}"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                _EVALUATOR_REPOSITORY,
                str(root),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise DatasetUnavailable(
                f"could not clone SWE-bench Pro evaluator: {completed.stderr[-500:]}"
            )
    if not (root / ".git").is_dir():
        raise DatasetUnavailable(f"SWE-bench Pro evaluator path is not a git checkout: {root}")
    commands = (
        ["git", "fetch", "--depth", "1", "origin", _EVALUATOR_REVISION],
        ["git", "-c", "advice.detachedHead=false", "switch", "--detach", _EVALUATOR_REVISION],
    )
    for command in commands:
        completed = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise DatasetUnavailable(
                f"could not pin SWE-bench Pro evaluator: {completed.stderr[-500:]}"
            )
    if not _evaluator_ready(cache):
        raise DatasetUnavailable("pinned SWE-bench Pro evaluator is incomplete")
    return root


def _agent_problem(row: dict) -> str:
    return (
        f"{row['problem_statement']}\n\nRequirements:\n{row.get('requirements') or ''}"
        f"\n\nNew interfaces introduced:\n{row.get('interface') or ''}"
    )


def _normalize_rows(rows: list[dict]) -> list[dict]:
    if len(rows) != _DATASET_ROWS:
        raise DatasetUnavailable(
            f"{_DATASET}@{_DATASET_REVISION} yielded {len(rows)} rows, "
            f"expected {_DATASET_ROWS}"
        )
    normalized = []
    seen: set[str] = set()
    for row in rows:
        instance_id = str(row["instance_id"])
        if instance_id in seen:
            raise DatasetUnavailable(f"duplicate SWE-bench Pro instance {instance_id}")
        seen.add(instance_id)
        dockerhub_tag = str(row["dockerhub_tag"])
        normalized.append(
            {
                **row,
                "instance_id": instance_id,
                "problem_statement": _agent_problem(row),
                "image_name": f"{_DOCKERHUB_USERNAME}/sweap-images:{dockerhub_tag}",
            }
        )
    return normalized


def _write_generation_dataset(rows: list[dict], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "test.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination


def _prediction_list(predictions: Path, destination: Path, expected_ids: set[str]) -> Path:
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != expected_ids:
        observed = len(payload) if isinstance(payload, dict) else None
        raise ValueError(
            f"generation produced {observed} predictions; expected {len(expected_ids)}"
        )
    converted = []
    for instance_id in sorted(expected_ids):
        prediction = payload[instance_id]
        if not isinstance(prediction, dict):
            raise ValueError(f"prediction {instance_id} is not an object")
        converted.append(
            {
                "instance_id": instance_id,
                "model_patch": str(prediction.get("model_patch") or ""),
                "prefix": "kairyu",
            }
        )
    destination.write_text(
        json.dumps(converted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def _validate_generation_trajectories(
    output_dir: Path,
    expected_ids: set[str],
) -> None:
    """Reject harness/runtime crashes that mini-swe-agent exits with status 0."""

    failures: list[str] = []
    for instance_id in sorted(expected_ids):
        path = output_dir / instance_id / f"{instance_id}.traj.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            exit_status = payload["info"]["exit_status"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            failures.append(f"{instance_id}: invalid trajectory ({type(error).__name__})")
            continue
        if exit_status not in _AGENT_OUTCOMES:
            failures.append(f"{instance_id}: {exit_status!r}")
    if failures:
        preview = ", ".join(failures[:5])
        suffix = "" if len(failures) <= 5 else f", and {len(failures) - 5} more"
        raise ValueError(
            f"generation had {len(failures)} harness/runtime failures: {preview}{suffix}"
        )


def _prune_incomplete_predictions(
    output_dir: Path,
    expected_ids: set[str],
) -> None:
    """Keep only predictions backed by a valid, completed agent trajectory."""

    predictions = output_dir / "preds.json"
    if not predictions.is_file():
        return
    try:
        payload = json.loads(predictions.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        predictions.unlink(missing_ok=True)
        return
    if not isinstance(payload, dict):
        predictions.unlink(missing_ok=True)
        return

    reusable = {}
    for instance_id, prediction in payload.items():
        if instance_id not in expected_ids or not isinstance(prediction, dict):
            continue
        trajectory = output_dir / instance_id / f"{instance_id}.traj.json"
        try:
            trajectory_payload = json.loads(trajectory.read_text(encoding="utf-8"))
            exit_status = trajectory_payload["info"]["exit_status"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if exit_status in _AGENT_OUTCOMES:
            reusable[instance_id] = prediction

    temporary = predictions.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(reusable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(predictions)


def parse_swebench_report(report: dict) -> tuple[list[ItemResult], int]:
    """Report -> per-instance items + official denominator (all submitted)."""
    resolved = report.get("resolved_ids") or []
    unresolved = report.get("unresolved_ids") or []
    errors = report.get("error_ids") or []
    items = (
        [ItemResult(item_id=str(i), status="completed", score=1.0) for i in resolved]
        + [ItemResult(item_id=str(i), status="completed", score=0.0) for i in unresolved]
        + [ItemResult(item_id=str(i), status="failed", error="evaluation error") for i in errors]
    )
    total = report.get("total_instances") or len(items)
    return items, int(total)


class SweBenchProAdapter:
    info = AdapterInfo(
        name="swe-bench-pro",
        display_name="SWE-Bench Pro",
        metric="resolved rate",
        binary_outcomes=True,
        hf_dataset=_DATASET,
        hf_revision=_DATASET_REVISION,
        extra_sources=((_EVALUATOR_REPOSITORY, _EVALUATOR_REVISION),),
        needs_docker=True,
        agentic=True,
        annotations=(
            "scaffold: mini-swe-agent text actions (official bundled SWE-bench "
            f"backticks config), agent.step_limit={_STEP_LIMIT}",
            "target max_output_tokens, reasoning_effort, top_p, and seed are forwarded "
            "as model.model_kwargs.*; local-model cost lookup errors are ignored; "
            "explicit temperature and recommended sampling are rejected because this "
            "wrapper has no verified passthrough",
            "vendor extra_body has no harness equivalent and is NOT forwarded",
            f"official ScaleAI evaluator {_EVALUATOR_REVISION[:12]} with local Docker",
            "task images first pulled by this run are removed after each generation "
            "and evaluation task to bound local Docker storage",
            "--attempts must remain 1; this wrapper has no verified repeated-trial "
            "or grouped chat seed-sweep path",
        ),
        evaluation_distributions=("mini-swe-agent",),
        evaluation_executables=("mini-extra",),
        history_provenance_complete=False,
        history_provenance_reason=(
            "the official evaluator uses mutable Docker Hub image tags"
        ),
    )

    def additional_cache_ready(self, cache) -> bool:
        return _evaluator_ready(cache)

    def download(self, ctx: DownloadContext) -> DownloadReport:
        if not ctx.force and adapter_cache_ready(self, ctx.cache):
            return DownloadReport(adapter=self.info.name, status="cached")
        try:
            if not ctx.cache.is_ready(self.info.name, **cache_pins(self.info)):
                from kairyu.bench.hub import load_hf_rows

                rows = _normalize_rows(
                    load_hf_rows(
                        _DATASET,
                        split="test",
                        revision=_DATASET_REVISION,
                    )
                )
                ctx.cache.write_rows(self.info.name, rows, cache_pins(self.info))
            rows = ctx.cache.read_rows(self.info.name)
            evaluator = _ensure_evaluator(ctx.cache)
            missing = [
                row["instance_id"]
                for row in rows
                if not (evaluator / "run_scripts" / row["instance_id"]).is_dir()
            ]
            if missing:
                raise DatasetUnavailable(
                    f"official evaluator has no run script for {len(missing)} instances"
                )
        except Exception as error:  # noqa: BLE001 - external source failures are data
            return DownloadReport(
                adapter=self.info.name,
                status="unavailable",
                detail=f"{type(error).__name__}: {error}",
            )
        return DownloadReport(
            adapter=self.info.name,
            status="ok",
            detail=f"{len(rows)} rows; evaluator {_EVALUATOR_REVISION[:12]}",
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
                "SWE-Bench Pro requires attempts=1"
            )
        available, reason = ctx.docker
        if not available:
            return reason
        missing = harness_missing()
        if missing is not None:
            return f"{missing} not installed (pip install 'kairyu[bench-agentic]')"
        if self.info.name in ctx.download_failures:
            return ctx.download_failures[self.info.name]
        if not adapter_cache_ready(self, ctx.cache):
            return "SWE-bench Pro dataset or official evaluator is unavailable"
        return None

    def _generate_command(
        self,
        target: BenchTarget,
        ctx: RunContext,
        output_dir: Path,
        dataset_dir: Path | None = None,
    ) -> list[str]:
        command = [
            mini_extra_executable(),
            "swebench",
            "--model",
            f"openai/{target.model}",
            "--model-class",
            _MODEL_CLASS,
            "--subset",
            str(dataset_dir) if dataset_dir is not None else _DATASET,
            "--split",
            "test",
            "--output",
            str(output_dir),
            "--workers",
            str(ctx.concurrency),
            "--config",
            _BASE_CONFIG,
            "--config",
            f"agent.step_limit={_STEP_LIMIT}",
            "--config",
            "environment.cwd=/app",
            "--config",
            'environment.run_args=["--rm","--entrypoint",""]',
            "--config",
            f"environment.pull_timeout={_PULL_TIMEOUT_S}",
            "--config",
            "model.cost_tracking=ignore_errors",
        ]
        # The harness forwards `model.model_kwargs.*` to its LLM client, so the
        # endpoint's named sampling policy can reach the agent's requests. Vendor
        # `extra_body` has no equivalent key and is annotated instead of guessed.
        for field, value in _model_kwargs(target).items():
            command += ["--config", f"model.model_kwargs.{field}={value}"]
        if ctx.limit is not None:
            command += ["--slice", f"0:{ctx.limit}"]
        return command

    def _evaluate_command(
        self,
        evaluator: Path,
        dataset: Path,
        predictions: Path,
        output_dir: Path,
        workers: int,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "kairyu.bench.swebench_pro_eval",
            f"--evaluator-script={evaluator / 'swe_bench_pro_eval.py'}",
            f"--raw_sample_path={dataset}",
            f"--patch_path={predictions}",
            f"--output_dir={output_dir}",
            f"--scripts_dir={evaluator / 'run_scripts'}",
            f"--num_workers={workers}",
            f"--dockerhub_username={_DOCKERHUB_USERNAME}",
            "--use_local_docker",
        ]

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        started_at = utc_now()
        reason = self._preconditions(target, ctx)
        if reason is not None:
            return skipped_pair(
                self.info.name, target.label(), reason, annotations=self.info.annotations
            )

        env = dict(os.environ)
        base = normalize_base_url(target.base_url)
        env["OPENAI_BASE_URL"] = base
        env["OPENAI_API_BASE"] = base
        env["OPENAI_API_KEY"] = target_api_key(target) or "sk-local"

        def _invoke(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
            return subprocess.run(
                command,
                capture_output=True,
                timeout=_STAGE_TIMEOUT_S,
                env=env,
                cwd=cwd,
                check=False,
            )

        rows = ctx.cache.read_rows(self.info.name)
        selected_rows = rows[: ctx.limit] if ctx.limit is not None else rows
        run_key = hashlib.sha256(
            f"{ctx.run_id}\0{target.label()}".encode()
        ).hexdigest()[:16]
        workdir = ctx.cache.adapter_dir(self.info.name) / "runs" / run_key
        workdir.mkdir(parents=True, exist_ok=True)
        dataset_dir = _write_generation_dataset(rows, workdir / "mini-dataset")
        generation_dir = workdir / "mini-output"
        predictions = generation_dir / "preds.json"
        evaluator = _evaluator_root(ctx.cache)
        evaluation_predictions = workdir / "predictions.json"
        evaluation_dir = workdir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        expected_ids = {str(row["instance_id"]) for row in selected_rows}
        _prune_incomplete_predictions(generation_dir, expected_ids)

        generate_command = self._generate_command(
            target, ctx, generation_dir, dataset_dir
        )
        try:
            completed = await asyncio.to_thread(_invoke, generate_command, workdir)
        except subprocess.TimeoutExpired:
            return self._failed(
                target, started_at, f"generate stage timed out ({_STAGE_TIMEOUT_S}s)"
            )
        if completed.returncode != 0:
            stderr = completed.stderr.decode(errors="replace")[-500:]
            return self._failed(
                target,
                started_at,
                f"generate stage failed (rc={completed.returncode}): {stderr}",
            )
        if not predictions.is_file():
            return self._failed(
                target,
                started_at,
                "generate stage produced no mini-output/preds.json file",
            )
        try:
            _validate_generation_trajectories(generation_dir, expected_ids)
            _prediction_list(
                predictions,
                evaluation_predictions,
                expected_ids,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return self._failed(target, started_at, f"invalid predictions: {error}")

        evaluate_command = self._evaluate_command(
            evaluator,
            ctx.cache.data_path(self.info.name),
            evaluation_predictions,
            evaluation_dir,
            ctx.concurrency,
        )
        try:
            completed = await asyncio.to_thread(_invoke, evaluate_command, evaluator)
        except subprocess.TimeoutExpired:
            return self._failed(
                target, started_at, f"evaluate stage timed out ({_STAGE_TIMEOUT_S}s)"
            )
        if completed.returncode != 0:
            stderr = completed.stderr.decode(errors="replace")[-500:]
            return self._failed(
                target,
                started_at,
                f"evaluate stage failed (rc={completed.returncode}): {stderr}",
            )
        report_path = evaluation_dir / "eval_results.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            return self._failed(target, started_at, f"invalid evaluation report: {error}")
        if not isinstance(report, dict) or set(report) != {
            str(row["instance_id"]) for row in selected_rows
        }:
            return self._failed(
                target,
                started_at,
                "evaluation report does not cover every submitted instance",
            )
        if any(not isinstance(resolved, bool) for resolved in report.values()):
            return self._failed(
                target, started_at, "evaluation report contains a non-boolean verdict"
            )
        items = [
            ItemResult(
                item_id=instance_id,
                status="completed",
                score=1.0 if resolved else 0.0,
            )
            for instance_id, resolved in sorted(report.items())
        ]
        total = len(selected_rows)

        resolved = sum(1 for item in items if item.score == 1.0)
        return summarize_items(
            self.info.name,
            target.label(),
            items,
            methodology={
                "metric": self.info.metric,
                "dataset": _DATASET,
                "dataset_revision": _DATASET_REVISION,
                "scaffold": "mini-swe-agent",
                "step_limit": _STEP_LIMIT,
                "evaluation": "ScaleAI SWE-bench_Pro-os (local Docker)",
                "evaluator_revision": _EVALUATOR_REVISION,
                "artifacts": str(workdir),
                "generate_command": " ".join(generate_command),
                "evaluate_command": " ".join(evaluate_command),
            },
            annotations=self.info.annotations,
            started_at=started_at,
            # official resolved rate divides by ALL submitted instances
            score_fn=lambda _: (resolved / total) if total else None,
        )

    def _failed(self, target: BenchTarget, started_at: str, reason: str) -> PairResult:
        return PairResult(
            benchmark=self.info.name,
            target=target.label(),
            status="failed",
            reason=reason,
            metrics={"score": None, "n_total": 0},
            annotations=self.info.annotations,
            started_at=started_at,
            finished_at=utc_now(),
        )
