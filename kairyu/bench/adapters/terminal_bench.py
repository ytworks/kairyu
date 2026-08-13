"""Terminal-Bench 2.1 via the official Harbor harness (`harbor run`).

The agent (terminus-2) talks to the target gateway through litellm's
OpenAI-compatible env vars; Harbor runs each task in its own container, so
docker is a hard precondition (skip, never crash — user decision 1).
"""

from __future__ import annotations

import asyncio
import json
import os
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
_LOCAL_DATASET_ENV = "KAIRYU_TERMINAL_BENCH_PATH"
_INCLUDE_TASKS_ENV = "KAIRYU_TERMINAL_BENCH_TASKS"
_AGENT = "terminus-2"
# Fugu's condition. Harbor's terminus-2 default turn budget is lower, which
# truncates long traces well before the published limit.
_MAX_TURNS = 500
# Terminal-Bench 2.1 task metadata commonly gives the agent 900 seconds. That
# is too short for a 500-turn local-model trajectory with 32K responses, and a
# timeout is a failed task rather than a meaningful accuracy verdict. Harbor's
# agent-only multiplier preserves verifier/build timeouts while allowing two
# hours for the agent phase of each task.
_AGENT_TIMEOUT_MULTIPLIER = 8.0
# Harbor applies ``max_timeout_sec`` before the multiplier.  Without this cap,
# an outlier task declaring a 3600-second agent budget receives eight hours,
# even though the intended benchmark-wide allowance is two hours.
_AGENT_MAX_TIMEOUT_S = 900.0
# Terminus expects one compact JSON command batch. Letting an orchestration
# publisher inherit the suite-wide 32K answer ceiling can spend ~13 minutes on
# a single malformed/unterminated batch, exhausting the official agent timeout
# before Harbor can obtain a verdict. This remains well above the agent's
# command payload needs while preserving the public target declaration for
# benchmarks that genuinely require long prose/code answers.
_AGENT_MAX_OUTPUT_TOKENS = 4096
_AGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "terminus_command_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "maxLength": 512},
                "plan": {"type": "string", "maxLength": 512},
                "commands": {
                    "type": "array",
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "keystrokes": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 2048,
                                "pattern": "(?:\\n$|^C-[cd]$)",
                            },
                            "duration": {"type": "number"},
                        },
                        "required": ["keystrokes", "duration"],
                        "additionalProperties": False,
                    },
                },
                "task_complete": {"type": "boolean"},
            },
            "required": ["analysis", "plan", "commands"],
            "additionalProperties": False,
        },
    },
}


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


def _failed_schema_item(index: int, error: str) -> ItemResult:
    return ItemResult(
        item_id=f"harbor-schema-{index}",
        status="failed",
        error=error,
    )


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
        expected_total = data.get("n_total_trials")
    else:
        entries = data
        expected_total = None
    if not isinstance(entries, list):
        return [_failed_schema_item(0, "Harbor trial results are not a list")]
    items: list[ItemResult] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            items.append(
                _failed_schema_item(
                    index,
                    f"Harbor trial result is {type(entry).__name__}, not an object",
                )
            )
            continue
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
    if isinstance(expected_total, int) and expected_total >= 0:
        for missing_index in range(len(entries), expected_total):
            items.append(
                _failed_schema_item(
                    missing_index,
                    "Harbor job omitted a declared trial result",
                )
            )
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
            "Harbor agent max_timeout_sec=900 is applied before the 8.0x agent "
            "execution multiplier, so every task has a 7200-second effective "
            "agent cap while verifier and build timeouts stay task-defined",
            "one attempt per task by default (--attempts); the official "
            "leaderboard requires at least five; Harbor -k is a harness trial "
            "count, not grouped chat seed-sweep sensitivity",
            "terminus-2's documented llm_call_kwargs receives the target output "
            f"limit capped at {_AGENT_MAX_OUTPUT_TOKENS} tokens; without an explicit "
            "limit an OpenAI-compatible endpoint may apply a 16-token default, while "
            "the suite-wide 32K ceiling can consume the agent timeout on an "
            "unterminated JSON command batch",
            "terminus-2's required JSON command object is declared as a strict "
            "OpenAI json_schema with bounded analysis, plan, and one atomic "
            "non-empty command per turn; ordinary commands are newline-terminated "
            "and C-c/C-d remain supported as standalone control keystrokes, so the "
            "target executes a complete batch without concatenating commands",
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
        local_dataset = os.environ.get(_LOCAL_DATASET_ENV)
        if local_dataset:
            path = Path(local_dataset)
            if not path.is_absolute() or not path.is_dir():
                return f"{_LOCAL_DATASET_ENV} must name an existing absolute directory"
        return None

    @staticmethod
    def _agent_config(target: BenchTarget, output_dir: Path) -> Path:
        """Write the granular Harbor agent config needed for a hard timeout cap.

        Harbor's ``-a``/``-m`` flags replace every agent loaded from ``--config``.
        The cap therefore has to live in a complete dynamic AgentConfig rather
        than in a partial config combined with those convenience flags.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "kairyu-agent-config.json"
        path.write_text(
            json.dumps(
                {
                    "agents": [
                        {
                            "name": _AGENT,
                            "model_name": f"openai/{target.model}",
                            "max_timeout_sec": _AGENT_MAX_TIMEOUT_S,
                            "kwargs": {
                                "max_turns": _MAX_TURNS,
                                "llm_call_kwargs": {
                                    "max_tokens": min(
                                        target.max_output_tokens,
                                        _AGENT_MAX_OUTPUT_TOKENS,
                                    ),
                                    "response_format": _AGENT_RESPONSE_FORMAT,
                                },
                            },
                        }
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def _command(self, target: BenchTarget, ctx: RunContext, output_dir: Path) -> list[str]:
        agent_config = self._agent_config(target, output_dir)
        command = [
            "harbor",
            "run",
            "--config",
            str(agent_config),
            *(self._dataset_arguments()),
            # `harbor run` writes into --jobs-dir; --output-dir belongs to
            # `harbor jobs download` and is rejected here.
            "--jobs-dir",
            str(output_dir),
            "-k",
            str(ctx.attempts),
            "--n-concurrent",
            str(ctx.concurrency),
            "--agent-timeout-multiplier",
            str(_AGENT_TIMEOUT_MULTIPLIER),
        ]
        if ctx.limit is not None:
            command += ["--n-tasks", str(ctx.limit)]
        include_tasks = tuple(
            task.strip()
            for task in os.environ.get(_INCLUDE_TASKS_ENV, "").split(",")
            if task.strip()
        )
        for task in include_tasks:
            command += ["--include-task-name", task]
        return command

    @staticmethod
    def _dataset_arguments() -> list[str]:
        local_dataset = os.environ.get(_LOCAL_DATASET_ENV)
        if local_dataset:
            return ["--path", str(Path(local_dataset).resolve())]
        return ["-d", _DATASET]

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

        # Keep the raw Harbor job directory.  It is the authoritative per-task
        # evidence and lets an interrupted long run be resumed with
        # ``harbor job resume``.  The caller controls its filesystem through
        # TMPDIR (the production example pins that to /mnt/nvme).
        output_dir = Path(tempfile.mkdtemp(prefix="kairyu-tb-"))
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
                target,
                started_at,
                f"harbor timed out after {_HARNESS_TIMEOUT_S}s; raw job retained at "
                f"{output_dir}",
            )
        if completed.returncode != 0:
            stderr = completed.stderr.decode(errors="replace")[-500:]
            return self._failed(
                target,
                started_at,
                f"harbor failed (rc={completed.returncode}): {stderr}; raw job "
                f"retained at {output_dir}",
            )
        data = self._find_results(output_dir)
        if data is None:
            return self._failed(
                target,
                started_at,
                f"no harbor results file found; raw job retained at {output_dir}",
            )
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
                "agent_max_output_tokens": min(
                    target.max_output_tokens,
                    _AGENT_MAX_OUTPUT_TOKENS,
                ),
                "agent_response_format": _AGENT_RESPONSE_FORMAT,
                "agent_max_timeout_s_before_multiplier": _AGENT_MAX_TIMEOUT_S,
                "agent_timeout_multiplier": _AGENT_TIMEOUT_MULTIPLIER,
                "agent_effective_timeout_cap_s": (
                    _AGENT_MAX_TIMEOUT_S * _AGENT_TIMEOUT_MULTIPLIER
                ),
                "harbor_jobs_dir": str(output_dir),
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
        """Find complete Harbor verdicts across supported on-disk layouts.

        Harbor <=0.16 embedded ``trial_results`` in the job result. Harbor
        0.17 intentionally excludes that heavy field from the final job-level
        file and leaves one authoritative ``result.json`` under each trial
        directory. Legacy wrappers may instead write ``{"results": [...]}``.
        Only accept typed lists of trial objects; agent artifacts also use a
        top-level ``results`` key and must never be mistaken for verdicts.
        """
        candidates = sorted(
            output_dir.rglob("*.json"),
            # prefer the job-level file over any per-trial artifact
            key=lambda path: (path.name != "result.json", len(path.parts), str(path)),
        )
        expected_total: int | None = None
        per_trial: list[tuple[Path, dict]] = []
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict):
                declared_total = data.get("n_total_trials")
                if isinstance(declared_total, int) and declared_total >= 0:
                    expected_total = (
                        declared_total
                        if expected_total is None
                        else max(expected_total, declared_total)
                    )
                embedded = data.get("trial_results")
                if (
                    isinstance(embedded, list)
                    and (embedded or isinstance(declared_total, int))
                    and all(isinstance(entry, dict) for entry in embedded)
                ):
                    return data
                legacy = data.get("results")
                if isinstance(legacy, list) and legacy and all(
                    isinstance(entry, dict) for entry in legacy
                ):
                    return data
                if (
                    path.name == "result.json"
                    and data.get("trial_name")
                    and (data.get("task_name") or data.get("task_id"))
                    and ("verifier_result" in data or "exception_info" in data)
                ):
                    per_trial.append((path, data))
                    continue
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if {"resolved", "reward", "verifier_result"} & set(data[0]):
                    return data
        if per_trial or expected_total is not None:
            per_trial.sort(key=lambda pair: str(pair[0]))
            return {
                "trial_results": [data for _path, data in per_trial],
                "n_total_trials": (
                    expected_total if expected_total is not None else len(per_trial)
                ),
            }
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
