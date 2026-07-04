"""SWE-Bench Pro: official two-stage flow — mini-swe-agent patches, docker eval.

Stage 1 points mini-swe-agent's OpenAI-compatible client at the target
gateway to generate patches for the public split; stage 2 runs the swebench
docker evaluation harness and this adapter translates its report. Docker or
the [bench-agentic] extra missing -> skipped, never a crash (user decision 1).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    RunContext,
    normalize_base_url,
    skipped_pair,
    summarize_items,
    target_api_key,
    utc_now,
)
from kairyu.bench.types import BenchTarget, DownloadReport, ItemResult, PairResult

_STAGE_TIMEOUT_S = 8 * 3600
_DATASET = "ScaleAI/SWE-bench_Pro"


def harness_missing() -> str | None:
    for module, package in (("minisweagent", "mini-swe-agent"), ("swebench", "swebench")):
        if importlib.util.find_spec(module) is None:
            return package
    return None


def parse_swebench_report(report: dict) -> tuple[list[ItemResult], int]:
    """Report -> per-instance items + official denominator (all submitted)."""
    resolved = report.get("resolved_ids") or []
    unresolved = report.get("unresolved_ids") or []
    errors = report.get("error_ids") or []
    items = (
        [ItemResult(item_id=str(i), status="completed", score=1.0) for i in resolved]
        + [ItemResult(item_id=str(i), status="completed", score=0.0) for i in unresolved]
        + [
            ItemResult(item_id=str(i), status="failed", error="evaluation error")
            for i in errors
        ]
    )
    total = report.get("total_instances") or len(items)
    return items, int(total)


class SweBenchProAdapter:
    info = AdapterInfo(
        name="swe-bench-pro",
        display_name="SWE-Bench Pro",
        metric="resolved rate",
        hf_dataset=_DATASET,
        needs_docker=True,
        agentic=True,
        annotations=("scaffold: mini-swe-agent (matches Fugu's published methodology)",),
    )

    def download(self, ctx: DownloadContext) -> DownloadReport:
        return DownloadReport(
            adapter=self.info.name,
            status="ok",
            detail="instances and images are fetched by the swebench harness at run time",
        )

    def _preconditions(self, ctx: RunContext) -> str | None:
        available, reason = ctx.docker
        if not available:
            return reason
        missing = harness_missing()
        if missing is not None:
            return f"{missing} not installed (pip install 'kairyu[bench-agentic]')"
        return None

    def _generate_command(
        self, target: BenchTarget, ctx: RunContext, output: Path
    ) -> list[str]:
        command = [
            "mini-extra",
            "swebench",
            "--model",
            f"openai/{target.model}",
            "--subset",
            _DATASET,
            "--split",
            "test",
            "--output",
            str(output),
            "--workers",
            str(ctx.concurrency),
        ]
        if ctx.limit is not None:
            command += ["--slice", f"0:{ctx.limit}"]
        return command

    def _evaluate_command(self, predictions: Path, run_id: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            _DATASET,
            "--split",
            "test",
            "--predictions_path",
            str(predictions),
            "--run_id",
            run_id,
        ]

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        started_at = utc_now()
        reason = self._preconditions(ctx)
        if reason is not None:
            return skipped_pair(
                self.info.name, target.label(), reason, annotations=self.info.annotations
            )

        import os

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

        with tempfile.TemporaryDirectory(prefix="kairyu-swebench-") as tmp:
            workdir = Path(tmp)
            predictions = workdir / "preds.json"
            for stage, command in (
                ("generate", self._generate_command(target, ctx, predictions)),
                ("evaluate", self._evaluate_command(predictions, "kairyu-bench")),
            ):
                try:
                    completed = await asyncio.to_thread(_invoke, command, workdir)
                except subprocess.TimeoutExpired:
                    return self._failed(
                        target, started_at, f"{stage} stage timed out ({_STAGE_TIMEOUT_S}s)"
                    )
                if completed.returncode != 0:
                    stderr = completed.stderr.decode(errors="replace")[-500:]
                    return self._failed(
                        target,
                        started_at,
                        f"{stage} stage failed (rc={completed.returncode}): {stderr}",
                    )
            report = self._find_report(workdir)
            if report is None:
                return self._failed(target, started_at, "no evaluation report produced")
            items, total = parse_swebench_report(report)

        resolved = sum(1 for item in items if item.score == 1.0)
        return summarize_items(
            self.info.name,
            target.label(),
            items,
            methodology={
                "metric": self.info.metric,
                "dataset": _DATASET,
                "scaffold": "mini-swe-agent",
                "evaluation": "swebench.harness.run_evaluation (docker)",
            },
            annotations=self.info.annotations,
            started_at=started_at,
            # official resolved rate divides by ALL submitted instances
            score_fn=lambda _: (resolved / total) if total else None,
        )

    @staticmethod
    def _find_report(workdir: Path) -> dict | None:
        for path in sorted(workdir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and "resolved_ids" in data:
                return data
        return None

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
