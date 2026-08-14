"""SuiteRunner: the one-command loop — download, run pairs, aggregate, report.

Targets run sequentially within a benchmark (item-level concurrency already
saturates a shared gateway; sequential pairs keep per-cell numbers
uncontended and comparable). Resume: same --run-id reuses every stored pair
whose run fingerprint matches and status is not "failed". Exit code 1 only
when a pair hard-failed.

Live per-pair output belongs to the progress reporter (stderr); stdout carries
the artifacts — download notes and the final scoreboard — so a run can be piped
without the play-by-play interleaving into it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from evals.adapters import suite_adapters, suite_info
from evals.adapters.base import (
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    adapter_cache_ready,
    cache_pins,
    evaluation_protocol_identity,
    utc_now,
)
from evals.aggregate import build_scoreboard, render_markdown
from evals.cache import BenchCache, resolve_cache_root
from evals.compare import build_comparison, render_comparison_markdown
from evals.history import cross_run_policies
from evals.sampling import PROTOCOL_ID as SAMPLING_PROTOCOL_ID
from evals.store import ResultStore
from evals.structured import structured_run_expectation
from evals.types import (
    SMOKE_LIMIT,
    BenchConfig,
    PairResult,
    pair_result_evidence_error,
    run_configuration_incomparable_reasons,
)

# How often a running pair reports that it is still alive.
_HEARTBEAT_INTERVAL_S = 15.0

_FINGERPRINT_EXCLUSIONS = frozenset(
    # location, resume and display controls: none of them change a score
    {"run_id", "results_dir", "cache_dir", "rerun", "download", "progress"}
)

_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SOURCE_SCOPES = ("kairyu", "pyproject.toml", "uv.lock")
_SOURCE_IDENTITY_FIELDS = (
    "git_commit",
    "git_commit_role",
    "source_kind",
    "source_tree_clean",
    "source_tree_sha256",
)


def _git_bytes(cwd: Path, *args: str) -> bytes | None:
    """Run one read-only git query without inheriting the caller's cwd."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _git_text(cwd: Path, *args: str) -> str | None:
    raw = _git_bytes(cwd, *args)
    if raw is None:
        return None
    try:
        value = raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return None
    return value or None


def _source_provenance() -> dict:
    """Attest the local benchmark harness, never the served target build.

    The module path anchors every git query.  Using the process cwd here would
    let an unrelated checkout lend its HEAD to an installed Kairyu package.
    """
    module_path = Path(__file__).resolve()
    digest = hashlib.sha256()
    try:
        digest.update(module_path.read_bytes())
    except OSError:
        digest.update(str(module_path).encode("utf-8", errors="surrogatepass"))
    unverified = {
        "git_commit": None,
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "unverified-package",
        "source_tree_clean": False,
        "source_tree_sha256": digest.hexdigest(),
    }

    root_text = _git_text(module_path.parent, "rev-parse", "--show-toplevel")
    if root_text is None:
        return unverified
    try:
        root = Path(root_text).resolve(strict=True)
        relative_module = module_path.relative_to(root)
    except (OSError, ValueError):
        return unverified
    relative_text = relative_module.as_posix()
    if _git_bytes(root, "ls-files", "--error-unmatch", "--", relative_text) is None:
        return unverified

    commit = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git_text(root, "rev-parse", "--verify", "HEAD^{tree}")
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        *_SOURCE_SCOPES,
    )
    if (
        commit is None
        or _GIT_COMMIT_RE.fullmatch(commit) is None
        or tree is None
        or _GIT_COMMIT_RE.fullmatch(tree) is None
        or status is None
    ):
        return unverified

    source_digest = hashlib.sha256()
    source_digest.update(b"kairyu-benchmark-source-v1\0")
    source_digest.update(commit.encode("ascii"))
    source_digest.update(b"\0")
    source_digest.update(tree.encode("ascii"))
    source_digest.update(b"\0")
    source_digest.update(status)
    if status:
        diff = _git_bytes(root, "diff", "--binary", "HEAD", "--", *_SOURCE_SCOPES)
        untracked = _git_bytes(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *_SOURCE_SCOPES,
        )
        if diff is None or untracked is None:
            return unverified
        source_digest.update(diff)
        for raw_name in sorted(name for name in untracked.split(b"\0") if name):
            try:
                name = raw_name.decode("utf-8", errors="strict")
                candidate = (root / name).resolve(strict=True)
                candidate.relative_to(root)
                payload = candidate.read_bytes()
            except (OSError, UnicodeDecodeError, ValueError):
                return unverified
            source_digest.update(b"\0untracked\0")
            source_digest.update(raw_name)
            source_digest.update(b"\0")
            source_digest.update(payload)

    return {
        "git_commit": commit,
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "git-checkout",
        "source_tree_clean": not status,
        "source_tree_sha256": source_digest.hexdigest(),
    }


def _environment(*, execution_runner=None) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    try:
        kairyu_version = version("kairyu")
    except PackageNotFoundError:
        kairyu_version = "unknown"
    environment = {
        **_source_provenance(),
        "kairyu_version": kairyu_version,
        "python": platform.python_version(),
        "created_at": utc_now(),
    }
    if execution_runner is not None:
        available, detail = execution_runner.availability()
        environment["execution"] = {
            **execution_runner.metadata(),
            "available": available,
            "availability_detail": detail,
        }
    return environment


def _history_ineligibility(config: BenchConfig, environment: dict) -> str | None:
    if config.offline_fixtures:
        return "offline fixture mode lacks cache-bound dataset identity and is diagnostic only"
    if environment.get("source_kind") != "git-checkout":
        return "the loaded benchmark package is not a tracked Git checkout"
    if environment.get("source_tree_clean") is not True:
        return "the local benchmark source tree is not clean"
    if environment.get("git_commit_role") != "local_benchmark_harness":
        return "the Git commit is not attested as the local benchmark harness"
    commit = environment.get("git_commit")
    digest = environment.get("source_tree_sha256")
    if not isinstance(commit, str) or _GIT_COMMIT_RE.fullmatch(commit) is None:
        return "the local benchmark harness has no full Git commit identity"
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return "the local benchmark harness has no source-tree SHA-256"
    return None


def _methodology_history_ineligibility(adapter_identities: list[dict]) -> str | None:
    for identity in adapter_identities:
        protocol = identity.get("evaluation_protocol")
        if not isinstance(protocol, dict):
            return f"benchmark {identity.get('name')!r} has no evaluator content identity"
        distributions = protocol.get("distributions")
        if not isinstance(distributions, list):
            return f"benchmark {identity.get('name')!r} has malformed harness identity"
        for distribution in distributions:
            if (
                isinstance(distribution, dict)
                and distribution.get("installed") is True
                and distribution.get("content_verified") is not True
            ):
                return (
                    f"benchmark {identity.get('name')!r} has an installed but "
                    "unverifiable external harness"
                )
        executables = protocol.get("executables")
        if not isinstance(executables, list):
            return f"benchmark {identity.get('name')!r} has malformed executable identity"
        for executable in executables:
            if not isinstance(executable, dict):
                return f"benchmark {identity.get('name')!r} has malformed executable identity"
            if executable.get("found") is not True:
                continue
            if executable.get("content_verified") is not True or not executable.get(
                "owned_by_distribution"
            ):
                return (
                    f"benchmark {identity.get('name')!r} resolves an external harness "
                    "executable that is unverified or PATH-shadowed"
                )
    return None


def _default_run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _adapter_identity(adapter, cache: BenchCache, *, offline_fixtures: bool) -> dict:
    info = adapter.info
    identity = {
        "name": info.name,
        "dataset": info.hf_dataset,
        "revision": info.hf_revision,
        "binary_outcomes": info.binary_outcomes,
        "paired_cluster_key": info.paired_cluster_key,
        "uses_judge_template": info.judge_template_name is not None,
        "history_provenance": {
            "complete": info.history_provenance_complete,
            "reason": info.history_provenance_reason or None,
            "requires_content_addressed_execution": info.needs_execution,
        },
    }
    if info.judge_template_name is not None:
        from evals.judge import judge_protocol_identity
        from evals.judge_prompts import judge_template_identity

        identity["judge_template"] = judge_template_identity(info.judge_template_name)
        identity["judge_protocol"] = judge_protocol_identity()
    identity["evaluation_protocol"] = evaluation_protocol_identity(adapter)
    pins = cache_pins(info)
    identity = {**identity, "sources": pins["sources"]}
    if offline_fixtures or not adapter_cache_ready(adapter, cache):
        return {**identity, "unavailable": True}
    try:
        manifest = cache.read_manifest(info.name)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {**identity, "unavailable": True}
    digest = manifest.get("sha256") if isinstance(manifest, dict) else None
    if not isinstance(digest, str):
        return {**identity, "unavailable": True}
    assets = manifest.get("assets") if isinstance(manifest, dict) else None
    if not isinstance(assets, list):
        return {**identity, "unavailable": True}
    if info.hf_dataset is not None and manifest.get("dataset") != info.hf_dataset:
        return {**identity, "unavailable": True}
    if info.hf_revision is not None and manifest.get("revision") != info.hf_revision:
        return {**identity, "unavailable": True}
    return {**identity, "sha256": digest, "assets": assets}


def _recordable_config(config: BenchConfig) -> dict:
    """Hash opaque request bodies before metadata can enter tracked history.

    ``extra_body_json`` can carry arbitrary vendor fields and therefore may
    contain credentials despite configuration guidance.  Runtime keeps the
    validated body in memory; durable metadata stores only a semantic JSON
    digest, which still makes it fingerprint-bearing without retaining a
    secret literal in ``run.json`` or ``scoreboards.jsonl``.
    """

    def sanitize(value: object) -> object:
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if not isinstance(value, dict):
            return value
        sanitized: dict = {}
        for key, item in value.items():
            if key == "extra_body_json" and isinstance(item, str):
                parsed = json.loads(item)
                canonical = json.dumps(
                    parsed,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                sanitized[key] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
            else:
                sanitized[key] = sanitize(item)
        return sanitized

    recorded = sanitize(config.model_dump(mode="json"))
    assert isinstance(recorded, dict)
    if config.attempts > 1:
        # ``attempts`` predates grouped chat sweeps and was already used by
        # external agentic harnesses.  This durable marker lets history accept
        # those legacy multi-trial records while still recognizing every new
        # run whose evaluator surface includes the #371 protocol.
        recorded["sampling_protocol"] = SAMPLING_PROTOCOL_ID
    return recorded


def _run_identity(config: BenchConfig, adapter_identities: list[dict]) -> dict:
    full_config = _recordable_config(config)
    immutable_config = {
        key: value for key, value in full_config.items() if key not in _FINGERPRINT_EXCLUSIONS
    }
    return {"config": immutable_config, "adapters": adapter_identities}


def _run_fingerprint(identity: dict) -> str:
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_level_incomparable_reasons(config: BenchConfig, ctx_limit: int | None) -> tuple[str, ...]:
    """Reasons a whole run cannot be compared with a published full-suite score.

    A limited run legitimately reports `completed` -- the limit is applied before
    aggregation -- so without this a 20-item smoke cell renders as a plain number
    with an unmarked delta against Fugu's full-suite result.
    """
    return run_configuration_incomparable_reasons(
        limit=ctx_limit,
        offline_fixtures=config.offline_fixtures,
    )


def _stamp_run_reasons(
    pair: PairResult,
    fingerprint: str,
    run_reasons: tuple[str, ...],
    source_identity: dict,
    cross_run_policy: tuple[str, str | None],
) -> PairResult:
    """Attach this run's fingerprint and run-level comparability to a pair.

    Applied to freshly executed AND resumed pairs: a stored pair carries its own
    reasons, but the run-level ones (subset, fixtures) belong to the run doing the
    reporting.
    """
    reasons = tuple(dict.fromkeys(tuple(pair.incomparable_reasons) + run_reasons))
    policy, policy_reason = cross_run_policy
    return pair.model_copy(
        update={
            "run_fingerprint": fingerprint,
            "source_identity": source_identity,
            "comparable": pair.comparable and not run_reasons,
            "incomparable_reasons": reasons,
            "cross_run_policy": policy,
            "cross_run_reason": policy_reason,
        }
    )


def _failed_evidence(pair: PairResult, reason: str) -> PairResult:
    """Discard a score whose source or methodology changed around execution."""
    return PairResult(
        benchmark=pair.benchmark,
        target=pair.target,
        status="failed",
        reason=reason,
        metrics={"score": None, "n_total": 0},
        annotations=pair.annotations,
        started_at=pair.started_at,
        finished_at=utc_now(),
    )


def _validated_pair_evidence(
    value: object,
    *,
    benchmark: str,
    target: str,
    annotations: tuple[str, ...],
    context: str,
    expected_sampling: dict[str, object],
    expected_structured_run: dict[str, object] | None,
    require_history_safe_counts: bool,
) -> PairResult:
    """Return trustworthy pair evidence or a failed pair at the scheduled key."""
    error = pair_result_evidence_error(
        value,
        expected_benchmark=benchmark,
        expected_target=target,
        expected_sampling=expected_sampling,
        expected_structured_run=expected_structured_run,
        require_history_safe_counts=require_history_safe_counts,
    )
    if error is None:
        assert isinstance(value, PairResult)
        return value
    if isinstance(value, PairResult):
        evidence_annotations = value.annotations
        started_at = value.started_at
    else:
        evidence_annotations = annotations
        started_at = ""
    return PairResult(
        benchmark=benchmark,
        target=target,
        status="failed",
        reason=f"{context}: {error}",
        metrics={"score": None, "n_total": 0},
        annotations=evidence_annotations,
        started_at=started_at,
        finished_at=utc_now(),
    )


class SuiteRunner:
    def __init__(
        self,
        config: BenchConfig,
        *,
        http_factory=None,
        probe_docker=None,
        progress=None,
    ) -> None:
        self.config = config
        self._http_factory = http_factory or (lambda: httpx.AsyncClient())
        self._probe_docker = probe_docker
        from evals.progress import SafeReporter, make_reporter

        if progress is None:
            progress = make_reporter(enabled=config.progress)
        # An injected reporter is guarded too: no reporter may end a run.
        self._progress = progress if isinstance(progress, SafeReporter) else SafeReporter(progress)

    def _build_context(
        self,
        cache: BenchCache,
        run_id: str = "",
        *,
        force_fresh_artifacts: bool = False,
    ) -> RunContext:
        config = self.config
        from evals.execution import build_execution_runner

        execution_runner = build_execution_runner(config.execution)
        limit = config.limit
        if config.smoke:
            limit = min(limit or SMOKE_LIMIT, SMOKE_LIMIT)
        if self._probe_docker is not None:
            docker = self._probe_docker()
        else:
            from evals.docker_probe import docker_available

            docker = docker_available()
        judge = None
        if config.judge.enabled:
            from evals.judge import build_judge_client

            judge = build_judge_client(config.judge, http_factory=self._http_factory)
        return RunContext(
            cache=cache,
            http_factory=self._http_factory,
            judge=judge,
            limit=limit,
            seed=config.seed,
            attempts=config.attempts,
            run_id=run_id,
            rerun=config.rerun or force_fresh_artifacts,
            artifacts_dir=Path(config.results_dir).resolve() / run_id / "artifacts",
            concurrency=config.concurrency,
            retries=config.retries,
            request_timeout_s=config.request_timeout_s,
            offline_fixtures=config.offline_fixtures,
            smoke=config.smoke,
            docker=docker,
            execution_runner=execution_runner,
            progress=self._progress,
            exec_semaphore=asyncio.Semaphore(max(1, (os.cpu_count() or 4) - 1)),
        )

    async def _run_pair(self, adapter, target, ctx: RunContext):
        """Run one pair, keeping the log moving while it is in progress.

        External harnesses never report items and their subprocess output is
        captured, so a heartbeat is the only thing distinguishing an 8-hour run
        from a hung one in a non-TTY log.
        """
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            return await adapter.run(target, ctx)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            self._progress.pair_heartbeat()

    def _download_missing(self, adapters, cache: BenchCache, ctx: RunContext) -> None:
        download_ctx = DownloadContext(cache=cache)
        for adapter in adapters:
            if adapter_cache_ready(adapter, cache):
                continue
            report = adapter.download(download_ctx)
            if report.status in ("gated", "unavailable", "extras_missing"):
                ctx.download_failures[adapter.info.name] = report.detail or report.status
                print(f"[download] {adapter.info.name}: {report.status} — {report.detail}")
            else:
                print(f"[download] {adapter.info.name}: {report.status} {report.detail}")

    async def run(self) -> int:
        """Run the suite; the reporter is closed even on failure or cancellation."""
        try:
            return await self._run()
        finally:
            self._progress.close()

    async def _run(self) -> int:
        config = self.config
        adapters = suite_adapters(config.suite, only=config.only, exclude=config.exclude)
        cache = BenchCache(resolve_cache_root(config.cache_dir))
        run_id = config.run_id or _default_run_id()
        store = ResultStore(config.results_dir, run_id)
        force_pair_rerun = store.has_evaluator_taint()

        ctx = self._build_context(
            cache,
            run_id,
            force_fresh_artifacts=force_pair_rerun,
        )
        current_environment = _environment(execution_runner=ctx.execution_runner)
        # Reject a cross-source resume before downloads or target requests.  A
        # second validation in initialize_run closes the time-of-check gap.
        store.require_resumable_environment(current_environment)
        if config.download and not config.offline_fixtures:
            self._download_missing(adapters, cache, ctx)

        adapter_identities = [
            _adapter_identity(
                adapter,
                cache,
                offline_fixtures=config.offline_fixtures,
            )
            for adapter in adapters
        ]
        identity = _run_identity(config, adapter_identities)
        fingerprint = _run_fingerprint(identity)
        recorded_config = _recordable_config(config)
        persisted = store.initialize_run(
            {
                "fingerprint": fingerprint,
                "identity": identity,
                "config": recorded_config,
                "environment": current_environment,
                "run_id": run_id,
            }
        )
        # The first run.json is authoritative.  In particular, a resume must
        # never mix old pairs with a later commit/environment in its scoreboard.
        fingerprint = persisted["fingerprint"]
        persisted_config = persisted["config"]
        environment = persisted["environment"]
        expected_source_identity = {
            field: environment.get(field) for field in _SOURCE_IDENTITY_FIELDS
        }
        history_ineligibility = _history_ineligibility(config, environment)
        if history_ineligibility is None:
            history_ineligibility = _methodology_history_ineligibility(adapter_identities)
        history_candidate = history_ineligibility is None
        source_tainted = False
        evaluator_tainted = False

        def mark_evaluator_drift(reason: str, changed_adapters: set[str]) -> None:
            nonlocal evaluator_tainted, history_ineligibility
            store.mark_evaluator_tainted(reason=reason, adapters=sorted(changed_adapters))
            evaluator_tainted = True
            history_ineligibility = reason

        def observe_source() -> dict:
            nonlocal history_ineligibility, source_tainted
            if not history_candidate:
                return expected_source_identity
            observed = _source_provenance()
            observed_identity = {field: observed.get(field) for field in _SOURCE_IDENTITY_FIELDS}
            if observed_identity != expected_source_identity:
                store.mark_source_tainted(
                    expected=expected_source_identity,
                    observed=observed_identity,
                )
                source_tainted = True
                history_ineligibility = (
                    "the local benchmark source identity changed while the suite ran"
                )
            return observed_identity

        observe_source()
        if history_ineligibility is None:
            store.ensure_scoreboard_key_available(persisted, rerun=config.rerun)
        expected_adapter_identities = {
            adapter.info.name: adapter_identity
            for adapter, adapter_identity in zip(adapters, adapter_identities, strict=True)
        }
        adapter_cross_run_policies = cross_run_policies(adapter_identities, environment)

        targets = [target.label() for target in config.targets]
        run_reasons = run_level_incomparable_reasons(config, ctx.limit)
        pairs: list[PairResult] = []

        def quarantine_source_pairs(reason: str) -> bool:
            """Persist failed evidence for every score observed after source drift."""
            changed = False
            for index, pair in enumerate(pairs):
                if pair.status not in {"completed", "partial"}:
                    continue
                failed_pair = _stamp_run_reasons(
                    _failed_evidence(pair, reason),
                    fingerprint,
                    run_reasons,
                    pair.source_identity or expected_source_identity,
                    adapter_cross_run_policies[pair.benchmark],
                )
                store.save_pair(failed_pair)
                pairs[index] = failed_pair
                changed = True
            return changed

        self._progress.suite_start(len(adapters) * len(config.targets))
        for adapter in adapters:
            for target in config.targets:
                label = target.label()
                expected_sampling = {
                    "attempts": config.attempts,
                    "seed": target.seed,
                    "mode": target.sampling_mode,
                    "temperature": target.temperature,
                    "top_p": target.top_p,
                    "required": isinstance(adapter, GenerativeAdapter),
                }
                expected_structured_run = (
                    structured_run_expectation(
                        limit=config.limit,
                        smoke=config.smoke,
                        selection_seed=config.seed,
                        extra_body_present=target.extra_body_json is not None,
                    )
                    if adapter.info.name == "structured-output"
                    else None
                )
                source_before = observe_source()
                current_adapter_identity = _adapter_identity(
                    adapter,
                    cache,
                    offline_fixtures=config.offline_fixtures,
                )
                dataset_identity_changed = (
                    current_adapter_identity != expected_adapter_identities[adapter.info.name]
                )
                if dataset_identity_changed:
                    mark_evaluator_drift(
                        "a dataset or evaluator identity changed after run initialization",
                        {adapter.info.name},
                    )
                if (
                    not config.rerun
                    and not force_pair_rerun
                    and not dataset_identity_changed
                    and not source_tainted
                ):
                    try:
                        existing = store.load_pair(
                            adapter.info.name,
                            label,
                            expected_fingerprint=fingerprint,
                            expected_source_identity=expected_source_identity,
                        )
                    except ValidationError:
                        invalid_existing = PairResult(
                            benchmark=adapter.info.name,
                            target=label,
                            status="failed",
                            reason="stored pair does not match the PairResult schema",
                            metrics={"score": None, "n_total": 0},
                            annotations=adapter.info.annotations,
                            finished_at=utc_now(),
                        )
                        store.save_pair(
                            _stamp_run_reasons(
                                invalid_existing,
                                fingerprint,
                                run_reasons,
                                expected_source_identity,
                                adapter_cross_run_policies[adapter.info.name],
                            )
                        )
                        existing = None
                    if existing is not None:
                        existing_error = pair_result_evidence_error(
                            existing,
                            expected_benchmark=adapter.info.name,
                            expected_target=label,
                            expected_sampling=expected_sampling,
                            expected_structured_run=expected_structured_run,
                            require_history_safe_counts=history_candidate,
                        )
                        if existing_error is not None:
                            # Persist the quarantine before retrying.  If the
                            # process is interrupted, a later resume can never
                            # mistake the malformed completed pair for cache.
                            invalid_existing = _validated_pair_evidence(
                                existing,
                                benchmark=adapter.info.name,
                                target=label,
                                annotations=adapter.info.annotations,
                                context="stored pair contains invalid evidence",
                                expected_sampling=expected_sampling,
                                expected_structured_run=expected_structured_run,
                                require_history_safe_counts=history_candidate,
                            )
                            store.save_pair(
                                _stamp_run_reasons(
                                    invalid_existing,
                                    fingerprint,
                                    run_reasons,
                                    expected_source_identity,
                                    adapter_cross_run_policies[adapter.info.name],
                                )
                            )
                            existing = None
                    if existing is not None and existing.status != "failed":
                        self._progress.pair_start(adapter.info.name, label)
                        self._progress.pair_done(existing.status, existing.score, cached=True)
                        # A pair stored before these fields existed validates with
                        # comparable=True by model default and carries the same
                        # fingerprint, so without this a resumed subset or fixture
                        # run would be reported with a numeric published delta and
                        # no banner. Re-stamped and re-saved so the evidence on
                        # disk matches what the report says.
                        stamped = _stamp_run_reasons(
                            existing,
                            fingerprint,
                            run_reasons,
                            expected_source_identity,
                            adapter_cross_run_policies[adapter.info.name],
                        )
                        if stamped != existing:
                            store.save_pair(stamped)
                        pairs.append(stamped)
                        continue
                self._progress.pair_start(
                    adapter.info.name,
                    label,
                    note="agentic harness" if adapter.info.agentic else "",
                )
                if source_tainted:
                    result = PairResult(
                        benchmark=adapter.info.name,
                        target=label,
                        status="failed",
                        reason="local benchmark source identity changed while the suite ran",
                        metrics={"score": None, "n_total": 0},
                        annotations=adapter.info.annotations,
                        started_at=utc_now(),
                        finished_at=utc_now(),
                    )
                elif dataset_identity_changed:
                    result = PairResult(
                        benchmark=adapter.info.name,
                        target=label,
                        status="failed",
                        reason="dataset identity changed after run initialization",
                        metrics={"score": None, "n_total": 0},
                        annotations=adapter.info.annotations,
                        started_at=utc_now(),
                        finished_at=utc_now(),
                    )
                else:
                    try:
                        result = await self._run_pair(adapter, target, ctx)
                    except Exception as error:  # noqa: BLE001 - isolate each pair
                        result = PairResult(
                            benchmark=adapter.info.name,
                            target=label,
                            status="failed",
                            reason=f"adapter crashed: {error}",
                            metrics={"score": None, "n_total": 0},
                            annotations=adapter.info.annotations,
                            started_at=utc_now(),
                            finished_at=utc_now(),
                        )
                result = _validated_pair_evidence(
                    result,
                    benchmark=adapter.info.name,
                    target=label,
                    annotations=adapter.info.annotations,
                    context="adapter returned invalid pair evidence",
                    expected_sampling=expected_sampling,
                    expected_structured_run=expected_structured_run,
                    require_history_safe_counts=history_candidate,
                )
                source_after = observe_source()
                if source_tainted and result.status != "failed":
                    result = _failed_evidence(
                        result,
                        "local benchmark source identity changed while the pair ran",
                    )
                final_pair_identity = _adapter_identity(
                    adapter,
                    cache,
                    offline_fixtures=config.offline_fixtures,
                )
                if not dataset_identity_changed and (
                    final_pair_identity != expected_adapter_identities[adapter.info.name]
                ):
                    mark_evaluator_drift(
                        "a dataset or evaluator identity changed while a pair ran",
                        {adapter.info.name},
                    )
                    if result.status != "failed":
                        result = _failed_evidence(
                            result,
                            "dataset or evaluator identity changed while the pair ran",
                        )
                result = _stamp_run_reasons(
                    result,
                    fingerprint,
                    run_reasons,
                    source_after if history_candidate else source_before,
                    adapter_cross_run_policies[adapter.info.name],
                )
                store.save_pair(result)
                self._progress.pair_done(result.status, result.score)
                pairs.append(result)

        changed_adapters = {
            adapter.info.name
            for adapter in adapters
            if _adapter_identity(
                adapter,
                cache,
                offline_fixtures=config.offline_fixtures,
            )
            != expected_adapter_identities[adapter.info.name]
        }
        if changed_adapters:
            mark_evaluator_drift(
                "a dataset or evaluator identity changed while the suite ran",
                changed_adapters,
            )
            for index, pair in enumerate(pairs):
                if pair.benchmark not in changed_adapters or pair.status not in {
                    "completed",
                    "partial",
                }:
                    continue
                failed_pair = _stamp_run_reasons(
                    _failed_evidence(
                        pair,
                        "dataset or evaluator identity changed while the suite ran",
                    ),
                    fingerprint,
                    run_reasons,
                    pair.source_identity or expected_source_identity,
                    adapter_cross_run_policies[pair.benchmark],
                )
                store.save_pair(failed_pair)
                pairs[index] = failed_pair

        published_comparison = suite_info(config.suite).published_comparison

        def render_reports() -> tuple[dict, str, dict | None, str | None]:
            board = build_scoreboard(
                run_id=run_id,
                suite=config.suite,
                config=persisted_config,
                environment=environment,
                fingerprint=fingerprint,
                pairs=pairs,
                targets=targets,
                target_configs=config.targets,
                judge=config.judge,
            )
            board_markdown = render_markdown(board)
            if not published_comparison:
                return board, board_markdown, None, None
            published = build_comparison(board)
            return (
                board,
                board_markdown,
                published,
                render_comparison_markdown(published),
            )

        scoreboard, markdown, comparison, comparison_markdown = render_reports()

        observe_source()
        if source_tainted and quarantine_source_pairs(
            "local benchmark source identity changed after scoreboard rendering"
        ):
            scoreboard, markdown, comparison, comparison_markdown = render_reports()
        late_changed_adapters = {
            adapter.info.name
            for adapter in adapters
            if _adapter_identity(
                adapter,
                cache,
                offline_fixtures=config.offline_fixtures,
            )
            != expected_adapter_identities[adapter.info.name]
        }
        if late_changed_adapters:
            mark_evaluator_drift(
                "a dataset or evaluator identity changed after scoreboard rendering",
                late_changed_adapters,
            )
            for index, pair in enumerate(pairs):
                if pair.benchmark not in late_changed_adapters or pair.status not in {
                    "completed",
                    "partial",
                }:
                    continue
                failed_pair = _stamp_run_reasons(
                    _failed_evidence(
                        pair,
                        "dataset or evaluator identity changed after scoreboard rendering",
                    ),
                    fingerprint,
                    run_reasons,
                    pair.source_identity or expected_source_identity,
                    adapter_cross_run_policies[pair.benchmark],
                )
                store.save_pair(failed_pair)
                pairs[index] = failed_pair
            scoreboard, markdown, comparison, comparison_markdown = render_reports()

        path = store.save_scoreboard(scoreboard, markdown)
        print()
        print(markdown)
        if comparison is not None and comparison_markdown is not None:
            store.save_comparison(comparison, comparison_markdown)
            print(comparison_markdown)

        observe_source()
        source_pairs_quarantined = source_tainted and quarantine_source_pairs(
            "local benchmark source identity changed before history append"
        )
        pre_append_changed_adapters = {
            adapter.info.name
            for adapter in adapters
            if _adapter_identity(
                adapter,
                cache,
                offline_fixtures=config.offline_fixtures,
            )
            != expected_adapter_identities[adapter.info.name]
        }
        # The final evaluator scan may itself be slow enough for the checkout
        # to change after the observation above.  Re-attest immediately after
        # that scan so its result can never be appended under stale source
        # provenance.
        observe_source()
        if source_tainted:
            source_pairs_quarantined = quarantine_source_pairs(
                "local benchmark source identity changed before history append"
            ) or source_pairs_quarantined
        if pre_append_changed_adapters:
            mark_evaluator_drift(
                "a dataset or evaluator identity changed before history append",
                pre_append_changed_adapters,
            )
            for index, pair in enumerate(pairs):
                if pair.benchmark not in pre_append_changed_adapters or pair.status not in {
                    "completed",
                    "partial",
                }:
                    continue
                failed_pair = _stamp_run_reasons(
                    _failed_evidence(
                        pair,
                        "dataset or evaluator identity changed before history append",
                    ),
                    fingerprint,
                    run_reasons,
                    pair.source_identity or expected_source_identity,
                    adapter_cross_run_policies[pair.benchmark],
                )
                store.save_pair(failed_pair)
                pairs[index] = failed_pair
        if source_pairs_quarantined or pre_append_changed_adapters:
            scoreboard, markdown, comparison, comparison_markdown = render_reports()
            path = store.save_scoreboard(scoreboard, markdown)
            print()
            print(markdown)
            if comparison is not None and comparison_markdown is not None:
                store.save_comparison(comparison, comparison_markdown)
                print(comparison_markdown)

        failed = (
            source_tainted or evaluator_tainted or any(pair.status == "failed" for pair in pairs)
        )
        if history_ineligibility is not None:
            print(f"[history] not indexed: {history_ineligibility}", file=sys.stderr)
        elif failed:
            print("[history] not indexed: the run contains failed pairs", file=sys.stderr)
        else:
            index_path = store.append_scoreboard_index(scoreboard, persisted, pairs)
            print(f"history: {index_path}")

        if force_pair_rerun and not failed:
            store.clear_evaluator_taint()

        print(f"results: {path.parent}")
        return 1 if failed else 0
