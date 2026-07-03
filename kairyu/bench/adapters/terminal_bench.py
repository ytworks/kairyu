"""Terminal-Bench 2.1 via the official Harbor harness (`harbor run`).

The agent (terminus-2) talks to the target gateway through litellm's
OpenAI-compatible env vars; Harbor runs each task in its own container, so
docker is a hard precondition (skip, never crash — user decision 1).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
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

_HARNESS_TIMEOUT_S = 8 * 3600
_DATASET = "terminal-bench/terminal-bench-2-1"


def parse_harbor_results(data) -> list[ItemResult]:
    """Harbor results -> per-task items ({"results": [...]}, or a bare list).

    Entries carry either a boolean `resolved` or a float `reward`.
    """
    entries = data.get("results", data) if isinstance(data, dict) else data
    items: list[ItemResult] = []
    for index, entry in enumerate(entries):
        task_id = str(entry.get("task_id", entry.get("id", index)))
        if "resolved" in entry:
            score = 1.0 if entry["resolved"] else 0.0
        elif entry.get("reward") is not None:
            score = float(entry["reward"])
        else:
            items.append(
                ItemResult(item_id=task_id, status="failed", error="no verdict in entry")
            )
            continue
        items.append(ItemResult(item_id=task_id, status="completed", score=score))
    return items


class TerminalBenchAdapter:
    info = AdapterInfo(
        name="terminal-bench",
        display_name="Terminal-Bench 2.1",
        metric="accuracy",
        needs_docker=True,
        agentic=True,
        annotations=("agent scaffold: terminus-2 via the Harbor harness",),
    )

    def download(self, ctx: DownloadContext) -> DownloadReport:
        return DownloadReport(
            adapter=self.info.name,
            status="ok",
            detail="tasks are fetched by the Harbor harness at run time",
        )

    def _preconditions(self, ctx: RunContext) -> str | None:
        available, reason = ctx.docker
        if not available:
            return reason
        if shutil.which("harbor") is None:
            return "harbor not installed (pip install 'kairyu[bench-agentic]')"
        return None

    def _command(self, target: BenchTarget, ctx: RunContext, output_dir: Path) -> list[str]:
        command = [
            "harbor",
            "run",
            "-d",
            _DATASET,
            "-a",
            "terminus-2",
            "-m",
            f"openai/{target.model}",
            "--output-dir",
            str(output_dir),
        ]
        if ctx.limit is not None:
            command += ["--n-tasks", str(ctx.limit)]
        return command

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

        with tempfile.TemporaryDirectory(prefix="kairyu-tb-") as tmp:
            output_dir = Path(tmp)
            command = self._command(target, ctx, output_dir)

            def _invoke() -> subprocess.CompletedProcess:
                return subprocess.run(
                    command,
                    capture_output=True,
                    timeout=_HARNESS_TIMEOUT_S,
                    env=env,
                    check=False,
                )

            try:
                completed = await asyncio.to_thread(_invoke)
            except subprocess.TimeoutExpired:
                return self._failed(
                    target, started_at, f"harbor timed out after {_HARNESS_TIMEOUT_S}s"
                )
            if completed.returncode != 0:
                stderr = completed.stderr.decode(errors="replace")[-500:]
                return self._failed(
                    target, started_at, f"harbor failed (rc={completed.returncode}): {stderr}"
                )
            data = self._find_results(output_dir)
            if data is None:
                return self._failed(target, started_at, "no harbor results file found")
            items = parse_harbor_results(data)

        return summarize_items(
            self.info.name,
            target.label(),
            items,
            methodology={
                "metric": self.info.metric,
                "dataset": _DATASET,
                "harness": "harbor",
                "agent": "terminus-2",
                "command": " ".join(command),
            },
            annotations=self.info.annotations,
            started_at=started_at,
        )

    @staticmethod
    def _find_results(output_dir: Path):
        for path in sorted(output_dir.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and "results" in data:
                return data
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if "resolved" in data[0] or "reward" in data[0]:
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
