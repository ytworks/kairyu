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
    external_harness_sampling_incompatibility,
    normalize_base_url,
    skipped_pair,
    summarize_items,
    target_api_key,
    utc_now,
)
from kairyu.bench.types import BenchTarget, DownloadReport, ItemResult, PairResult

# A complete Terminal-Bench 2.1 package at concurrency one can legitimately
# exceed a workday. Keep a finite fail-closed bound without truncating the
# requested full task set in the parent process.
_HARNESS_TIMEOUT_S = 72 * 3600
# Terminal-Bench 2.1 is published on Harbor Hub under this organization/package
# identifier. `terminal-bench@2.1` addresses the legacy registry instead and is
# not a published dataset there.
_DATASET = "terminal-bench/terminal-bench-2-1"
_AGENT = "terminus-2"
# Fugu's condition. Harbor's terminus-2 default turn budget is lower, which
# truncates long traces well before the published limit.
_MAX_TURNS = 500


# Harbor's JobResult holds `trial_results`, and each TrialResult carries its
# verdict under `verifier_result.rewards` — a task-defined dict, not a fixed
# field. These are the keys Terminal-Bench-style verifiers use, in preference
# order; a dict with exactly one entry is unambiguous whatever it is called.
_REWARD_KEYS = ("reward", "resolved", "accuracy", "score", "passed")


def trial_reward(rewards: dict) -> float | None:
    """One reward value from a Harbor reward dict, or None when ambiguous."""
    for key in _REWARD_KEYS:
        if key in rewards:
            value = rewards[key]
            return float(bool(value)) if isinstance(value, bool) else float(value)
    if len(rewards) == 1:
        value = next(iter(rewards.values()))
        return float(bool(value)) if isinstance(value, bool) else float(value)
    return None


def harbor_mean(items: list[ItemResult]) -> float | None:
    """Harbor's own `Mean`: every trial counts, a missing reward as zero.

    `aggregate_reward_dicts()` maps `None` to zero before averaging, so a trial
    that raised is a zero in the denominator, not an exclusion from it. Averaging
    only the completed trials would report a crashed run as a better score.
    """
    if not items:
        return None
    return sum(item.score or 0.0 for item in items) / len(items)


def _trial_name(entry: dict, index: int) -> str:
    """Prefer `trial_name`: with `-k > 1` it distinguishes attempts of one task."""
    if entry.get("trial_name"):
        return str(entry["trial_name"])
    task_id = entry.get("task_id")
    if isinstance(task_id, dict):
        for key in ("name", "task_name", "path", "id"):
            if task_id.get(key):
                return str(task_id[key])
    return str(task_id or entry.get("id") or index)


def parse_harbor_results(data) -> list[ItemResult]:
    """Harbor `result.json` -> per-trial items.

    Accepts the real `JobResult` shape (`trial_results` with
    `verifier_result.rewards`) plus the flatter `{"results": [...]}` / bare-list
    shapes older adapters wrote, so a schema move degrades to a visible
    `failed` item rather than a silently empty cell.
    """
    if isinstance(data, dict):
        entries = data.get("trial_results")
        if entries is None:
            entries = data.get("results", [])
    else:
        entries = data
    items: list[ItemResult] = []
    for index, entry in enumerate(entries):
        name = _trial_name(entry, index)
        if entry.get("exception_info"):
            exception = entry["exception_info"] or {}
            items.append(
                ItemResult(
                    item_id=name,
                    status="failed",
                    error=f"trial raised {exception.get('exception_type', 'error')}",
                )
            )
            continue
        verifier = entry.get("verifier_result") or {}
        rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
        if isinstance(rewards, dict) and rewards:
            score = trial_reward(rewards)
            if score is None:
                items.append(
                    ItemResult(
                        item_id=name,
                        status="failed",
                        error=f"ambiguous reward keys: {sorted(rewards)}",
                    )
                )
                continue
            items.append(ItemResult(item_id=name, status="completed", score=score))
            continue
        # legacy/flat shapes
        if "resolved" in entry:
            items.append(
                ItemResult(
                    item_id=name,
                    status="completed",
                    score=1.0 if entry["resolved"] else 0.0,
                )
            )
        elif entry.get("reward") is not None:
            items.append(ItemResult(item_id=name, status="completed", score=float(entry["reward"])))
        else:
            items.append(ItemResult(item_id=name, status="failed", error="no verdict in trial"))
    return items


class TerminalBenchAdapter:
    info = AdapterInfo(
        name="terminal-bench",
        display_name="Terminal-Bench 2.1",
        metric="accuracy",
        needs_docker=True,
        agentic=True,
        annotations=(
            f"agent scaffold: {_AGENT} via the Harbor harness, "
            f"max_turns={_MAX_TURNS} (Fugu's condition)",
            "one attempt per task by default (--attempts); the official "
            "leaderboard requires at least five; Harbor -k is a harness trial "
            "count, not grouped chat seed-sweep sensitivity",
            "target max_output_tokens is forwarded through terminus-2's documented "
            "llm_call_kwargs; without it an OpenAI-compatible endpoint may apply a "
            "16-token default and truncate every JSON command batch",
            "target reasoning_effort, top_p, seed, and vendor extra_body are NOT "
            "forwarded: Harbor's agent kwargs are agent-defined and terminus-2 "
            "does not expose those fields as portable harness controls; explicit "
            "temperature and recommended sampling are rejected",
            "score is Harbor's Mean over EVERY trial, errored trials as zero",
        ),
        evaluation_distributions=("harbor",),
        evaluation_executables=("harbor",),
        history_provenance_complete=False,
        history_provenance_reason=(
            "Harbor fetches mutable remote tasks and task container images without "
            "resolved content identities"
        ),
    )

    def download(self, ctx: DownloadContext) -> DownloadReport:
        return DownloadReport(
            adapter=self.info.name,
            status="ok",
            detail=f"{_DATASET} tasks are fetched by the Harbor harness at run time",
        )

    def _preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        sampling_reason = external_harness_sampling_incompatibility(
            target,
            harness="Harbor/terminus-2",
        )
        if sampling_reason is not None:
            return sampling_reason
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
            _AGENT,
            "-m",
            f"openai/{target.model}",
            # `harbor run` writes into --jobs-dir; --output-dir belongs to
            # `harbor jobs download` and is rejected here.
            "--jobs-dir",
            str(output_dir),
            "--ak",
            f"max_turns={_MAX_TURNS}",
            "--ak",
            "llm_call_kwargs="
            + json.dumps(
                {"max_tokens": target.max_output_tokens},
                separators=(",", ":"),
            ),
            "-k",
            str(ctx.attempts),
            "--n-concurrent",
            str(ctx.concurrency),
        ]
        if ctx.limit is not None:
            command += ["--n-tasks", str(ctx.limit)]
        return command

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        started_at = utc_now()
        reason = self._preconditions(target, ctx)
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
                "agent": _AGENT,
                "max_turns": _MAX_TURNS,
                "attempts": ctx.attempts,
                "denominator": (
                    "every trial, errored ones as zero (Harbor's own Mean maps a "
                    "missing reward to zero before averaging)"
                ),
                "command": " ".join(command),
            },
            annotations=self.info.annotations,
            started_at=started_at,
            score_fn=harbor_mean,
        )

    @staticmethod
    def _find_results(output_dir: Path):
        """Harbor writes a job-level `result.json` holding `trial_results`."""
        candidates = sorted(
            output_dir.rglob("*.json"),
            # prefer the job-level file over any per-trial artifact
            key=lambda path: (path.name != "result.json", len(path.parts), str(path)),
        )
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and ("trial_results" in data or "results" in data):
                return data
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if {"resolved", "reward", "verifier_result"} & set(data[0]):
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
