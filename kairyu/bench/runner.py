"""SuiteRunner: the one-command loop — download, run pairs, aggregate, report.

Targets run sequentially within a benchmark (item-level concurrency already
saturates a shared gateway; sequential pairs keep per-cell numbers
uncontended and comparable). Resume: same --run-id reuses every stored pair
whose run fingerprint matches and status is not "failed". Exit code 1 only
when a pair hard-failed.
"""

from __future__ import annotations

import asyncio
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
    skipped_pair,
    utc_now,
)
from kairyu.bench.aggregate import build_scoreboard, render_markdown
from kairyu.bench.cache import BenchCache, resolve_cache_root
from kairyu.bench.store import ResultStore
from kairyu.bench.types import SMOKE_LIMIT, BenchConfig, PairResult

_FINGERPRINT_EXCLUSIONS = frozenset(
    {"run_id", "results_dir", "cache_dir", "rerun", "download"}
)


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


def _adapter_identity(adapter, cache: BenchCache, *, offline_fixtures: bool) -> dict:
    info = adapter.info
    identity = {
        "name": info.name,
        "dataset": info.hf_dataset,
        "revision": info.hf_revision,
    }
    if offline_fixtures or not cache.is_ready(
        info.name, info.hf_dataset, info.hf_revision
    ):
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
            if cache.is_ready(
                adapter.info.name,
                adapter.info.hf_dataset,
                adapter.info.hf_revision,
            ):
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

        ctx = self._build_context(cache)
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
        environment = _environment()
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
        pairs: list[PairResult] = []
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
                        print(f"[cached] {adapter.info.name} × {label}: {existing.status}")
                        pairs.append(existing)
                        continue
                print(f"[run] {adapter.info.name} × {label} ...")
                if dataset_identity_changed:
                    result = skipped_pair(
                        adapter.info.name,
                        label,
                        "dataset identity changed after run initialization",
                        annotations=adapter.info.annotations,
                    )
                else:
                    try:
                        result = await adapter.run(target, ctx)
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
                result = result.model_copy(update={"run_fingerprint": fingerprint})
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
            target_configs=config.targets,
            judge=config.judge,
        )
        markdown = render_markdown(scoreboard)
        path = store.save_scoreboard(scoreboard, markdown)
        print()
        print(markdown)
        print(f"results: {path.parent}")
        return 1 if any(pair.status == "failed" for pair in pairs) else 0
