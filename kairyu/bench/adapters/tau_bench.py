"""τ³-Bench Banking: official-harness wrapper (agent + user-simulator LLM loop).

The τ-bench family ships its environment (tools, DB, reward function) as a
package with a `run` CLI; reimplementing it locally would not be comparable,
so this adapter shells out to the installed harness — `tau3` preferred,
`tau2` accepted with a substitute annotation — and translates its results
file. Missing harness / missing user-simulator config degrade to skipped.
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
from kairyu.bench.types import (
    BenchTarget,
    DownloadReport,
    ItemResult,
    PairResult,
)

_HARNESS_TIMEOUT_S = 4 * 3600


def detect_harness() -> str | None:
    """Console script of the installed τ-bench flavor ('tau3' > 'tau2')."""
    for flavor in ("tau3", "tau2"):
        if shutil.which(flavor) is not None:
            return flavor
    return None


def parse_tau_results(data) -> list[ItemResult]:
    """Translate a τ-bench results JSON into per-task ItemResults.

    Accepts both shapes seen across versions: {"simulations": [...]} and a
    bare list; each entry needs a reward (0..1) and ideally a task id.
    """
    if isinstance(data, dict):
        entries = data.get("simulations") or data.get("results") or []
    else:
        entries = data
    items: list[ItemResult] = []
    for index, entry in enumerate(entries):
        reward = entry.get("reward")
        task_id = str(entry.get("task_id", entry.get("id", index)))
        if reward is None:
            items.append(
                ItemResult(item_id=task_id, status="failed", error="no reward in entry")
            )
        else:
            items.append(
                ItemResult(item_id=task_id, status="completed", score=float(reward))
            )
    return items


class TauBenchBankingAdapter:
    info = AdapterInfo(
        name="tau-bench-banking",
        display_name="τ³-Bench Banking",
        metric="pass^1 (avg reward)",
        agentic=True,
        judge_preferred=True,  # the user simulator rides the judge config
    )

    def download(self, ctx: DownloadContext) -> DownloadReport:
        # Tasks/environment ship inside the harness package itself.
        return DownloadReport(
            adapter=self.info.name,
            status="ok",
            detail="dataset ships with the tau3/tau2 package",
        )

    def _preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        if detect_harness() is None:
            return "tau harness not installed (pip install 'kairyu[bench-agentic]')"
        if ctx.judge is None:
            return (
                "requires a user-simulator LLM: configure the judge endpoint "
                "(--judge-base-url/--judge-model)"
            )
        judge_base = normalize_base_url(ctx.judge.config.base_url)
        if judge_base != normalize_base_url(target.base_url):
            return (
                "user simulator must be served by the target gateway (the harness "
                "takes one OPENAI_BASE_URL); point the judge at the same base_url"
            )
        return None

    def _command(
        self, flavor: str, target: BenchTarget, ctx: RunContext, output: Path
    ) -> list[str]:
        command = [
            flavor,
            "run",
            "--domain",
            "banking",
            "--agent-llm",
            f"openai/{target.model}",
            "--user-llm",
            f"openai/{ctx.judge.config.model}",
            "--output",
            str(output),
        ]
        if ctx.limit is not None:
            command += ["--num-tasks", str(ctx.limit)]
        return command

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        started_at = utc_now()
        reason = self._preconditions(target, ctx)
        if reason is not None:
            return skipped_pair(
                self.info.name, target.label(), reason, annotations=self.info.annotations
            )
        flavor = detect_harness()
        annotations = self.info.annotations
        if flavor == "tau2":
            annotations = annotations + (
                "tau2 banking substitute — the tau3 harness is not installed; "
                "scores are NOT directly comparable to Fugu's τ³ number",
            )

        import os

        env = dict(os.environ)
        base = normalize_base_url(target.base_url)
        env["OPENAI_BASE_URL"] = base
        env["OPENAI_API_BASE"] = base
        env["OPENAI_API_KEY"] = target_api_key(target) or "sk-local"

        with tempfile.TemporaryDirectory(prefix="kairyu-tau-") as tmp:
            output = Path(tmp) / "results.json"
            command = self._command(flavor, target, ctx, output)

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
                return PairResult(
                    benchmark=self.info.name,
                    target=target.label(),
                    status="failed",
                    reason=f"{flavor} harness timed out after {_HARNESS_TIMEOUT_S}s",
                    metrics={"score": None, "n_total": 0},
                    annotations=annotations,
                    started_at=started_at,
                    finished_at=utc_now(),
                )
            if completed.returncode != 0 or not output.exists():
                stderr = completed.stderr.decode(errors="replace")[-500:]
                return PairResult(
                    benchmark=self.info.name,
                    target=target.label(),
                    status="failed",
                    reason=f"{flavor} harness failed (rc={completed.returncode}): {stderr}",
                    metrics={"score": None, "n_total": 0},
                    annotations=annotations,
                    started_at=started_at,
                    finished_at=utc_now(),
                )
            data = json.loads(output.read_text(encoding="utf-8"))

        items = parse_tau_results(data)
        return summarize_items(
            self.info.name,
            target.label(),
            items,
            methodology={
                "metric": self.info.metric,
                "harness": flavor,
                "domain": "banking",
                "user_simulator": ctx.judge.config.model,
                "command": " ".join(command),
            },
            annotations=annotations,
            started_at=started_at,
        )
