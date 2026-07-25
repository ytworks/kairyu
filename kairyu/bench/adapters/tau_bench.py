"""τ³-Bench Banking: official-harness wrapper (agent + user-simulator LLM loop).

The τ-bench family ships its environment (tools, DB, reward function) as a
package with a `run` CLI; reimplementing it locally would not be comparable,
so this adapter shells out to the installed harness — `tau3` preferred,
`tau2` accepted with a substitute annotation — and translates its results
file. Missing harness / missing user-simulator config degrade to skipped.

Three harness facts drive the command built here: the banking domain is named
`banking_knowledge` (there is no `banking`), results are addressed by `--save-to
<name>` under the harness data directory rather than by an output path, and the
user simulator's sampling is set through `--user-llm-args` JSON.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
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
# The harness exposes banking as `banking_knowledge`; `banking` is not one of
# its domains, so the previous command could only ever fail.
_DOMAIN = "banking_knowledge"
# Fugu enables every knowledge-retrieval tool. `alltools` is also the harness
# default, but pinning it makes the condition part of the record.
_RETRIEVAL_CONFIG = "alltools"
_RESULTS_NAME = "results.json"


def detect_harness() -> str | None:
    """Console script of the installed τ-bench flavor ('tau3' > 'tau2')."""
    for flavor in ("tau3", "tau2"):
        if shutil.which(flavor) is not None:
            return flavor
    return None


def data_dir_candidates(flavor: str) -> list[Path]:
    """Roots that may hold `simulations/<save_to>/results.json`.

    The harness resolves its data directory from `TAU2_DATA_DIR`, falling back
    to a directory inside the installed package, so the results file is not at
    a path the caller chose.
    """
    roots: list[Path] = []
    env_dir = os.environ.get("TAU2_DATA_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    for module in dict.fromkeys((flavor, "tau2")):
        try:
            import importlib.util

            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        for location in getattr(spec, "submodule_search_locations", None) or []:
            package = Path(location)
            roots += [package / "data", package.parent / "data"]
    roots.append(Path.cwd() / "data")
    return list(dict.fromkeys(roots))


def find_results(flavor: str, save_to: str) -> Path | None:
    for root in data_dir_candidates(flavor):
        candidate = root / "simulations" / save_to / _RESULTS_NAME
        if candidate.is_file():
            return candidate
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
        annotations=(
            f"domain {_DOMAIN}, retrieval config {_RETRIEVAL_CONFIG} "
            "(Fugu enables every knowledge-retrieval tool)",
            "one trial per task by default (--attempts); Fugu reports pass@4",
        ),
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
        self, flavor: str, target: BenchTarget, ctx: RunContext, save_to: str
    ) -> list[str]:
        command = [
            flavor,
            "run",
            "--domain",
            _DOMAIN,
            "--retrieval-config",
            _RETRIEVAL_CONFIG,
            "--agent-llm",
            f"openai/{target.model}",
            "--user-llm",
            f"openai/{ctx.judge.config.model}",
            "--num-trials",
            str(ctx.attempts),
            # a NAME under the harness data dir, not a path the caller picks
            "--save-to",
            save_to,
        ]
        agent_args = self._llm_args(target)
        if agent_args:
            command += ["--agent-llm-args", agent_args]
        user_args = self._llm_args(ctx.judge.config)
        if user_args:
            command += ["--user-llm-args", user_args]
        if ctx.limit is not None:
            command += ["--num-tasks", str(ctx.limit)]
        return command

    @staticmethod
    def _llm_args(options) -> str | None:
        """Sampling policy as the JSON the harness passes to litellm.

        Fugu ran the user simulator at low reasoning effort; that is configured
        on the judge/user-simulator endpoint like any other sampling knob.
        """
        args = options.wire_overrides()
        return json.dumps(args, sort_keys=True) if args else None

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

        env = dict(os.environ)
        base = normalize_base_url(target.base_url)
        env["OPENAI_BASE_URL"] = base
        env["OPENAI_API_BASE"] = base
        env["OPENAI_API_KEY"] = target_api_key(target) or "sk-local"

        save_to = f"kairyu-{self.info.name}-{target.label()}".replace("/", "_")
        command = self._command(flavor, target, ctx, save_to)

        def _invoke() -> subprocess.CompletedProcess:
            return subprocess.run(
                command,
                capture_output=True,
                timeout=_HARNESS_TIMEOUT_S,
                env=env,
                check=False,
            )

        def failed(reason: str) -> PairResult:
            return PairResult(
                benchmark=self.info.name,
                target=target.label(),
                status="failed",
                reason=reason,
                metrics={"score": None, "n_total": 0},
                annotations=annotations,
                started_at=started_at,
                finished_at=utc_now(),
            )

        try:
            completed = await asyncio.to_thread(_invoke)
        except subprocess.TimeoutExpired:
            return failed(f"{flavor} harness timed out after {_HARNESS_TIMEOUT_S}s")
        if completed.returncode != 0:
            stderr = completed.stderr.decode(errors="replace")[-500:]
            return failed(f"{flavor} harness failed (rc={completed.returncode}): {stderr}")
        results = find_results(flavor, save_to)
        if results is None:
            searched = ", ".join(
                str(root / "simulations" / save_to / _RESULTS_NAME)
                for root in data_dir_candidates(flavor)
            )
            return failed(f"{flavor} produced no results file; looked in: {searched}")
        data = json.loads(results.read_text(encoding="utf-8"))

        items = parse_tau_results(data)
        metric = (
            "pass^1 (avg reward)"
            if ctx.attempts == 1
            else f"avg reward over {ctx.attempts} trials"
        )
        return summarize_items(
            self.info.name,
            target.label(),
            items,
            methodology={
                "metric": metric,
                "harness": flavor,
                "domain": _DOMAIN,
                "retrieval_config": _RETRIEVAL_CONFIG,
                "num_trials": ctx.attempts,
                "user_simulator": ctx.judge.config.model,
                "user_simulator_sampling": self._llm_args(ctx.judge.config),
                "results_file": str(results),
                "command": " ".join(command),
            },
            annotations=annotations,
            started_at=started_at,
        )
