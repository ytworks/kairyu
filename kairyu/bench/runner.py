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
import subprocess
from datetime import UTC, datetime

import httpx

from kairyu.bench.adapters import suite_adapters
from kairyu.bench.adapters.base import (
    DownloadContext,
    RunContext,
    cache_pins,
    skipped_pair,
    utc_now,
)
from kairyu.bench.aggregate import build_scoreboard, render_markdown
from kairyu.bench.cache import BenchCache, resolve_cache_root
from kairyu.bench.compare import build_comparison, render_comparison_markdown
from kairyu.bench.store import ResultStore
from kairyu.bench.types import SMOKE_LIMIT, BenchConfig, PairResult

# How often a running pair reports that it is still alive.
_HEARTBEAT_INTERVAL_S = 15.0

_FINGERPRINT_EXCLUSIONS = frozenset(
    # location, resume and display controls: none of them change a score
    {"run_id", "results_dir", "cache_dir", "rerun", "download", "progress"}
)


def _environment(*, execution_runner=None) -> dict:
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, timeout=5, check=False
            )
            .stdout.decode()
            .strip()
            or None
        )
    except OSError:
        commit = None
    from importlib.metadata import PackageNotFoundError, version

    try:
        kairyu_version = version("kairyu")
    except PackageNotFoundError:
        kairyu_version = "unknown"
    environment = {
        "git_commit": commit,
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


def _default_run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _adapter_identity(adapter, cache: BenchCache, *, offline_fixtures: bool) -> dict:
    info = adapter.info
    identity = {
        "name": info.name,
        "dataset": info.hf_dataset,
        "revision": info.hf_revision,
    }
    pins = cache_pins(info)
    identity = {**identity, "sources": pins["sources"]}
    if offline_fixtures or not cache.is_ready(info.name, **pins):
        return {**identity, "unavailable": True}
    try:
        manifest = cache.read_manifest(info.name)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {**identity, "unavailable": True}
    digest = manifest.get("sha256") if isinstance(manifest, dict) else None
    if not isinstance(digest, str):
        return {**identity, "unavailable": True}
    if info.hf_dataset is not None and manifest.get("dataset") != info.hf_dataset:
        return {**identity, "unavailable": True}
    if info.hf_revision is not None and manifest.get("revision") != info.hf_revision:
        return {**identity, "unavailable": True}
    return {**identity, "sha256": digest}


def _run_identity(config: BenchConfig, adapter_identities: list[dict]) -> dict:
    full_config = config.model_dump(mode="json")
    immutable_config = {
        key: value
        for key, value in full_config.items()
        if key not in _FINGERPRINT_EXCLUSIONS
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
    reasons: list[str] = []
    if ctx_limit is not None:
        reasons.append(
            f"subset run: at most {ctx_limit} items per benchmark, not the full set"
        )
    if config.offline_fixtures:
        reasons.append(
            "synthetic offline fixtures stood in for the real datasets; scores are "
            "not measurements"
        )
    return tuple(reasons)



def _stamp_run_reasons(
    pair: PairResult, fingerprint: str, run_reasons: tuple[str, ...]
) -> PairResult:
    """Attach this run's fingerprint and run-level comparability to a pair.

    Applied to freshly executed AND resumed pairs: a stored pair carries its own
    reasons, but the run-level ones (subset, fixtures) belong to the run doing the
    reporting.
    """
    reasons = tuple(
        dict.fromkeys(tuple(pair.incomparable_reasons) + run_reasons)
    )
    return pair.model_copy(
        update={
            "run_fingerprint": fingerprint,
            "comparable": pair.comparable and not run_reasons,
            "incomparable_reasons": reasons,
        }
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
        from kairyu.bench.progress import SafeReporter, make_reporter

        if progress is None:
            progress = make_reporter(enabled=config.progress)
        # An injected reporter is guarded too: no reporter may end a run.
        self._progress = (
            progress if isinstance(progress, SafeReporter) else SafeReporter(progress)
        )

    def _build_context(self, cache: BenchCache, run_id: str = "") -> RunContext:
        config = self.config
        from kairyu.bench.execution import build_execution_runner

        execution_runner = build_execution_runner(config.execution)
        limit = config.limit
        if config.smoke:
            limit = min(limit or SMOKE_LIMIT, SMOKE_LIMIT)
        if self._probe_docker is not None:
            docker = self._probe_docker()
        else:
            from kairyu.bench.docker_probe import docker_available

            docker = docker_available()
        judge = None
        if config.judge.enabled:
            from kairyu.bench.judge import JudgeClient

            judge = JudgeClient(config.judge, http_factory=self._http_factory)
        return RunContext(
            cache=cache,
            http_factory=self._http_factory,
            judge=judge,
            limit=limit,
            seed=config.seed,
            attempts=config.attempts,
            run_id=run_id,
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
            if cache.is_ready(adapter.info.name, **cache_pins(adapter.info)):
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

        ctx = self._build_context(cache, run_id)
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
        environment = _environment(execution_runner=ctx.execution_runner)
        store.initialize_run(
            {
                "fingerprint": fingerprint,
                "identity": identity,
                "config": config.model_dump(mode="json"),
                "environment": environment,
                "run_id": run_id,
            }
        )
        expected_adapter_identities = {
            adapter.info.name: adapter_identity
            for adapter, adapter_identity in zip(
                adapters, adapter_identities, strict=True
            )
        }

        targets = [target.label() for target in config.targets]
        run_reasons = run_level_incomparable_reasons(config, ctx.limit)
        pairs: list[PairResult] = []
        self._progress.suite_start(len(adapters) * len(config.targets))
        for adapter in adapters:
            for target in config.targets:
                label = target.label()
                current_adapter_identity = _adapter_identity(
                    adapter,
                    cache,
                    offline_fixtures=config.offline_fixtures,
                )
                dataset_identity_changed = (
                    current_adapter_identity
                    != expected_adapter_identities[adapter.info.name]
                )
                if not config.rerun and not dataset_identity_changed:
                    existing = store.load_pair(
                        adapter.info.name,
                        label,
                        expected_fingerprint=fingerprint,
                    )
                    if existing is not None and existing.status != "failed":
                        self._progress.pair_start(adapter.info.name, label)
                        self._progress.pair_done(
                            existing.status, existing.score, cached=True
                        )
                        # A pair stored before these fields existed validates with
                        # comparable=True by model default and carries the same
                        # fingerprint, so without this a resumed subset or fixture
                        # run would be reported with a numeric published delta and
                        # no banner. Re-stamped and re-saved so the evidence on
                        # disk matches what the report says.
                        stamped = _stamp_run_reasons(existing, fingerprint, run_reasons)
                        if stamped != existing:
                            store.save_pair(stamped)
                        pairs.append(stamped)
                        continue
                self._progress.pair_start(
                    adapter.info.name,
                    label,
                    note="agentic harness" if adapter.info.agentic else "",
                )
                if dataset_identity_changed:
                    result = skipped_pair(
                        adapter.info.name,
                        label,
                        "dataset identity changed after run initialization",
                        annotations=adapter.info.annotations,
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
                result = _stamp_run_reasons(result, fingerprint, run_reasons)
                store.save_pair(result)
                self._progress.pair_done(result.status, result.score)
                pairs.append(result)

        scoreboard = build_scoreboard(
            run_id=run_id,
            suite=config.suite,
            config=config.model_dump(),
            environment=environment,
            pairs=pairs,
            targets=targets,
            target_configs=config.targets,
            judge=config.judge,
        )
        markdown = render_markdown(scoreboard)
        path = store.save_scoreboard(scoreboard, markdown)
        print()
        print(markdown)

        comparison = build_comparison(scoreboard)
        comparison_markdown = render_comparison_markdown(comparison)
        store.save_comparison(comparison, comparison_markdown)
        print(comparison_markdown)

        print(f"results: {path.parent}")
        return 1 if any(pair.status == "failed" for pair in pairs) else 0
