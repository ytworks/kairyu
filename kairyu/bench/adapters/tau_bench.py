"""τ³-Bench Banking: official-harness wrapper (agent + user-simulator LLM loop).

The τ-bench family ships its environment (tools, DB, reward function) as a
package with a `run` CLI; reimplementing it locally would not be comparable,
so this adapter shells out to the installed harness and translates its results
file. Official τ³ v1.x retains the `tau2` package and CLI name; only pre-1.0
`tau2` is the older substitute. Missing harness / missing user-simulator config
degrade to skipped.

Three harness facts drive the command built here: the banking domain is named
`banking_knowledge` (there is no `banking`), results are addressed by `--save-to
<name>` under the harness data directory rather than by an output path, and the
user simulator's sampling is set through `--user-llm-args` JSON.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

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

_HARNESS_TIMEOUT_S = 4 * 3600
# The harness exposes banking as `banking_knowledge`; `banking` is not one of
# its domains, so the previous command could only ever fail.
_DOMAIN = "banking_knowledge"
# ``alltools`` is the harness default and Fugu condition, but it hard-codes a
# second public OpenAI embeddings model (text-embedding-3-large). Kairyu's
# single-public-model pilot cannot expose that service without violating its
# target boundary, so use the harness's official no-retrieval-tool alternative
# and mark the resulting score incomparable below.
_RETRIEVAL_CONFIG = "full_kb"
_RESULTS_NAME = "results.json"


def detect_harness() -> str | None:
    """Console script exposed by the installed τ-bench package.

    Official τ³ v1.x intentionally retains the historical ``tau2`` package and
    CLI names. A hypothetical/legacy ``tau3`` wrapper remains accepted first.
    """
    for flavor in ("tau3", "tau2"):
        if shutil.which(flavor) is not None:
            return flavor
    return None


def harness_version(flavor: str) -> str | None:
    """Installed distribution version, independent of its console-script name."""
    distributions = ("tau2", "tau3") if flavor == "tau2" else ("tau3", "tau2")
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def is_tau3_release(flavor: str) -> bool:
    """Whether the harness is official τ³ rather than the pre-1.0 τ² fallback."""
    if flavor == "tau3":
        return True
    version = harness_version(flavor)
    if version is None:
        return False
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        return False
    return major >= 1


def harness_data_dir(flavor: str) -> Path | None:
    """The harness's own resolved `DATA_DIR`, asked of the harness itself.

    `TAU2_DATA_DIR` wins there; otherwise it is `Path(utils.__file__).parents[3]
    / "data"`, which for an installed package sits beside `site-packages` — not
    under it. Guessing that layout is how a successful run gets reported as
    producing no results, so the value is imported rather than reconstructed.
    """
    for module in dict.fromkeys((f"{flavor}.utils.utils", "tau2.utils.utils")):
        try:
            imported = importlib.import_module(module)
        except Exception:  # noqa: BLE001 - a missing/partial harness is not fatal
            continue
        data_dir = getattr(imported, "DATA_DIR", None)
        if data_dir is not None:
            return Path(data_dir)
    return None


def data_dir_candidates(flavor: str) -> list[Path]:
    """Roots that may hold `simulations/<save_to>/results.json`.

    The harness's own `DATA_DIR` comes first; the rest are fallbacks for when it
    cannot be imported (a partially installed harness, or a flavor that renamed
    the module).
    """
    roots: list[Path] = []
    resolved = harness_data_dir(flavor)
    if resolved is not None:
        roots.append(resolved)
    env_dir = os.environ.get("TAU2_DATA_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    for module in dict.fromkeys((flavor, "tau2")):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        for location in getattr(spec, "submodule_search_locations", None) or []:
            package = Path(location)
            # source checkout: <root>/src/tau2 -> <root>/data
            roots += [package / "data", package.parent / "data", package.parents[1] / "data"]
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
        # τ³ v1.x moved the official aggregate under reward_info while older
        # τ² results and a few exported forms keep it at the top level.
        reward_info = entry.get("reward_info")
        if reward is None and isinstance(reward_info, dict):
            reward = reward_info.get("reward")
        task_id = str(entry.get("task_id", entry.get("id", index)))
        if reward is None:
            items.append(ItemResult(item_id=task_id, status="failed", error="no reward in entry"))
        else:
            items.append(ItemResult(item_id=task_id, status="completed", score=float(reward)))
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
            "(the official full knowledge base is inlined; Fugu instead uses "
            "the alltools retrieval condition)",
            "one trial per task by default (--attempts); Fugu reports pass@4; "
            "--num-trials is a harness trial count, not grouped chat seed-sweep "
            "sensitivity",
            "target and user-simulator reasoning_effort, top_p, seed, and vendor "
            "extra_body are forwarded through harness LLM args; explicit target "
            "temperature and recommended sampling are rejected because the harness "
            "has no verified equivalent",
        ),
        evaluation_distributions=("tau2", "tau3"),
        evaluation_executables=("tau3", "tau2"),
        history_provenance_complete=False,
        history_provenance_reason=(
            "the harness may use mutable TAU2_DATA_DIR content"
        ),
    )

    def download(self, ctx: DownloadContext) -> DownloadReport:
        # Tasks/environment ship inside the harness package itself.
        return DownloadReport(
            adapter=self.info.name,
            status="ok",
            detail="dataset ships with the official tau-three/tau2 package",
        )

    def _preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        sampling_reason = external_harness_sampling_incompatibility(
            target,
            harness="tau",
        )
        if sampling_reason is not None:
            return sampling_reason
        if detect_harness() is None:
            return (
                "tau harness not installed (install official tau2-bench v1.x "
                "with the knowledge extra)"
            )
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

    def _save_to_name(self, target: BenchTarget, ctx: RunContext) -> str:
        """Unique per invocation, with the kairyu run id for traceability.

        The harness PROMPTS before resuming when the results file already
        exists, so a fixed name makes a second run interactive — or, if
        accepted, resumes simulations produced under a different configuration.
        kairyu's own pair-level resume is what reuses finished work.
        """
        return "-".join(
            part
            for part in (
                "kairyu",
                self.info.name,
                target.label().replace("/", "_"),
                ctx.run_id or None,
                uuid4().hex[:8],
            )
            if part
        )

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
        version = harness_version(flavor)
        annotations = self.info.annotations
        incomparable: tuple[str, ...] = (
            "the official full_kb retrieval policy differs from Fugu's alltools "
            "condition; scores are not directly comparable",
        )
        if not is_tau3_release(flavor):
            annotations = annotations + (
                "pre-1.0 tau2 banking substitute — the tau3 release is not installed; "
                "scores are NOT directly comparable to Fugu's τ³ number",
            )
            # A run-time substitution: the comparison report must withhold this
            # cell's delta, not merely footnote it.
            incomparable += (
                "the tau2 banking harness stood in for tau3; scores are not "
                "directly comparable to Fugu's τ³ number",
            )

        env = dict(os.environ)
        base = normalize_base_url(target.base_url)
        env["OPENAI_BASE_URL"] = base
        env["OPENAI_API_BASE"] = base
        env["OPENAI_API_KEY"] = target_api_key(target) or "sk-local"

        save_to = self._save_to_name(target, ctx)
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
                comparable=not incomparable,
                incomparable_reasons=incomparable,
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
            "pass^1 (avg reward)" if ctx.attempts == 1 else f"avg reward over {ctx.attempts} trials"
        )
        return summarize_items(
            self.info.name,
            target.label(),
            items,
            methodology={
                "metric": metric,
                "harness": flavor,
                "harness_version": version,
                "harness_release": "tau3" if is_tau3_release(flavor) else "tau2",
                "domain": _DOMAIN,
                "retrieval_config": _RETRIEVAL_CONFIG,
                "num_trials": ctx.attempts,
                "user_simulator": ctx.judge.config.model,
                "user_simulator_sampling": self._llm_args(ctx.judge.config),
                "results_file": str(results),
                "command": " ".join(command),
            },
            annotations=annotations,
            comparable=not incomparable,
            incomparable_reasons=incomparable,
            started_at=started_at,
        )
