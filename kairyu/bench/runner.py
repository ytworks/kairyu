"""SuiteRunner: the one-command loop — download, run pairs, aggregate, report.

Targets run sequentially within a benchmark (item-level concurrency already
saturates a shared gateway; sequential pairs keep per-cell numbers
uncontended and comparable). Resume: same --run-id reuses every stored pair
whose status is not "failed". Exit code 1 only when a pair hard-failed.
"""

from __future__ import annotations

import asyncio
import os
import platform
import subprocess
from datetime import UTC, datetime

import httpx

from kairyu.bench.adapters import suite_adapters
from kairyu.bench.adapters.base import DownloadContext, RunContext, utc_now
from kairyu.bench.aggregate import build_scoreboard, render_markdown
from kairyu.bench.cache import BenchCache, resolve_cache_root
from kairyu.bench.store import ResultStore
from kairyu.bench.types import SMOKE_LIMIT, BenchConfig, PairResult


def _environment() -> dict:
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
    return {
        "git_commit": commit,
        "kairyu_version": kairyu_version,
        "python": platform.python_version(),
        "created_at": utc_now(),
    }


def _default_run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


class SuiteRunner:
    def __init__(self, config: BenchConfig, *, http_factory=None, probe_docker=None) -> None:
        self.config = config
        self._http_factory = http_factory or (lambda: httpx.AsyncClient())
        self._probe_docker = probe_docker

    def _build_context(self, cache: BenchCache) -> RunContext:
        config = self.config
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
            concurrency=config.concurrency,
            retries=config.retries,
            request_timeout_s=config.request_timeout_s,
            offline_fixtures=config.offline_fixtures,
            smoke=config.smoke,
            docker=docker,
            exec_semaphore=asyncio.Semaphore(max(1, (os.cpu_count() or 4) - 1)),
        )

    def _download_missing(self, adapters, cache: BenchCache, ctx: RunContext) -> None:
        download_ctx = DownloadContext(cache=cache)
        for adapter in adapters:
            if cache.is_ready(adapter.info.name):
                continue
            report = adapter.download(download_ctx)
            if report.status in ("gated", "unavailable", "extras_missing"):
                ctx.download_failures[adapter.info.name] = report.detail or report.status
                print(f"[download] {adapter.info.name}: {report.status} — {report.detail}")
            else:
                print(f"[download] {adapter.info.name}: {report.status} {report.detail}")

    async def run(self) -> int:
        config = self.config
        adapters = suite_adapters(config.suite, only=config.only, exclude=config.exclude)
        cache = BenchCache(resolve_cache_root(config.cache_dir))
        run_id = config.run_id or _default_run_id()
        store = ResultStore(config.results_dir, run_id)
        environment = _environment()
        store.write_run_config(
            {"config": config.model_dump(), "environment": environment, "run_id": run_id}
        )

        ctx = self._build_context(cache)
        if config.download and not config.offline_fixtures:
            self._download_missing(adapters, cache, ctx)

        targets = [target.label() for target in config.targets]
        pairs: list[PairResult] = []
        for adapter in adapters:
            for target in config.targets:
                label = target.label()
                if not config.rerun:
                    existing = store.load_pair(adapter.info.name, label)
                    if existing is not None and existing.status != "failed":
                        print(f"[cached] {adapter.info.name} × {label}: {existing.status}")
                        pairs.append(existing)
                        continue
                print(f"[run] {adapter.info.name} × {label} ...")
                try:
                    result = await adapter.run(target, ctx)
                except Exception as error:  # noqa: BLE001 - a pair must never kill the suite
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
                store.save_pair(result)
                score = f"{result.score * 100:.1f}" if result.score is not None else "n/a"
                print(f"       -> {result.status} (score={score})")
                pairs.append(result)

        scoreboard = build_scoreboard(
            run_id=run_id,
            suite=config.suite,
            config=config.model_dump(),
            environment=environment,
            pairs=pairs,
            targets=targets,
        )
        markdown = render_markdown(scoreboard)
        path = store.save_scoreboard(scoreboard, markdown)
        print()
        print(markdown)
        print(f"results: {path.parent}")
        return 1 if any(pair.status == "failed" for pair in pairs) else 0
