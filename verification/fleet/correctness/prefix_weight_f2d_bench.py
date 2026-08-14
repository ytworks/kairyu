#!/usr/bin/env python3
"""G5 F2d: deterministic full-policy replay for prefix-placement weights.

This gate deliberately measures no host wall time.  Each policy is run from a
fresh ``ReplicaPool`` + ``PrefixIndex`` state against blocking virtual
backends.  All requests in one shared-prefix family are placed before any can
complete, so the pool's real outstanding counters form a deterministic queue.
One virtual tick is one uncached 256-character prompt chunk; a request's TTFT
is the queued prompt work ahead of it plus its own uncached prompt work.

The workload reuses F2a's exact shared-prefix prompt/session generator and the
200 eligible replica IDs proven by F2b's retained final fleet snapshot.  Whole
families are assigned to train or held-out once.  Every lambda=beta/alpha arm
is replayed on train; only the selected arm and the declared production
baseline are replayed on held-out data.

Two raw files are retained:

* ``prefix-weight-f2d-router.jsonl`` is emitted by the production
  :class:`JsonlRouterLog`.
* ``prefix-weight-f2d-raw.jsonl`` contains the deterministic virtual outcomes
  and frozen trace descriptors.

The verifier hashes both, joins every production placement to exactly one
outcome through ``learning.dataset``, replays every state transition, reruns
the tuner from train rows only, and recomputes the held-out comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from kairyu.engine.backend import (
    CacheHint,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
)
from kairyu.orchestration.learning.dataset import (
    build_placement_dataset,
    canonicalize_prefix_weights,
    tune_prefix_weights,
)
from kairyu.orchestration.prefix_index import PrefixIndex
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.router import JsonlRouterLog
from kairyu.sampling_params import SamplingParams
from verification.fleet.performance import prefix_routing_f2a_bench as f2a

SCHEMA_VERSION = "kairyu.f2d.prefix-weight-replay.v1"
GATE = "G5-F2d"
RAW_NAME = "prefix-weight-f2d-raw.jsonl"
ROUTER_NAME = "prefix-weight-f2d-router.jsonl"
MANIFEST_NAME = "prefix-weight-f2d-manifest.json"

ALPHA = 1.0
BASELINE_LAMBDA = 0.25
# Unique beta/alpha policies in the original 4x4 (alpha, beta) grid.
FORMAL_LAMBDAS = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)
DEFAULT_SEED = f2a.DEFAULT_SEED
QUEUE_DEPTH_THRESHOLD = 8
VIRTUAL_TICK_SECONDS = 1.0

_REPO_ROOT = Path(__file__).resolve().parents[3]
_F2A_DIR = _REPO_ROOT / "bench/results/f2a-prefix-routing-500-2026-07-28"
_F2B_DIR = _REPO_ROOT / "bench/results/f2b-kv-event-retained"
_F2A_MANIFEST = _F2A_DIR / "prefix-routing-f2a-manifest.json"
_F2A_RAW = _F2A_DIR / "prefix-routing-f2a-raw.jsonl"
_F2B_MANIFEST = _F2B_DIR / "kv-event-f2b-manifest.json"
_F2B_RAW = _F2B_DIR / "kv-event-f2b-raw.jsonl"

_PINNED_INPUT_SHA256 = {
    "f2a_manifest": "fb2858ed213be3f6b98463d30623e937d74403a7a7dd97356311439a7f7585f4",
    "f2a_raw": "82bb2a2ff420dbd4e244685ce2a83f38379028604be3c1077e85daf5b31cd0f3",
    "f2b_manifest": "e83f44dc1bac2338691b20aba4470aed6317dc51e6af45029b005cdbdeae7c43",
    "f2b_raw": "6a43544f672af438b7eb2d6cb74e35dd907cf357a0191bf42a54e8093ac0a6bb",
}
_SOURCE_PATHS = (
    "verification/fleet/correctness/prefix_weight_f2d_bench.py",
    "verification/fleet/performance/prefix_routing_f2a_bench.py",
    "verification/fleet/correctness/kv_event_f2b_bench.py",
    "kairyu/orchestration/learning/dataset.py",
    "kairyu/orchestration/replica.py",
    "kairyu/orchestration/prefix_index.py",
    "kairyu/orchestration/router.py",
    "kairyu/engine/backend.py",
    "kairyu/sampling_params.py",
)
_LEGACY_SOURCE_PATHS = tuple(
    path.replace(
        "verification/fleet/correctness/prefix_weight_f2d_bench.py",
        "bench/prefix_weight_f2d_bench.py",
    ).replace(
        "verification/fleet/performance/prefix_routing_f2a_bench.py",
        "bench/prefix_routing_f2a_bench.py",
    ).replace(
        "verification/fleet/correctness/kv_event_f2b_bench.py",
        "bench/kv_event_f2b_bench.py",
    )
    for path in _SOURCE_PATHS
)
_HEX = frozenset("0123456789abcdef")
_PASS_CONTRACT = (
    "manifest.passed is the online gate+source verdict; offline acceptance "
    "additionally requires every artifact-integrity check"
)
_GATE_CHECK_NAMES = frozenset(
    {
        "families_are_disjoint_between_train_and_heldout",
        "production_routes_join_outcomes_one_to_one",
        "every_policy_state_and_virtual_ttft_replays",
        "no_extra_or_missing_measured_requests",
        "dataset_tuner_recomputes_frozen_selection_from_train",
        "all_train_outcomes_precede_tuning_and_all_heldout_follow",
        "train_policy_panel_is_exactly_balanced",
        "heldout_contains_only_tuned_and_baseline_on_identical_trace",
        "all_replayed_requests_succeeded",
        "production_log_contains_prefix_match",
        "heldout_mean_ttft_strictly_improves",
    }
)


@dataclass(frozen=True)
class WorkloadConfig:
    profile: str
    seed: int
    replicas: int
    families: int
    requests_per_family: int
    train_families: int
    heldout_families: int
    lambdas: tuple[float, ...] = FORMAL_LAMBDAS
    baseline_lambda: float = BASELINE_LAMBDA
    alpha: float = ALPHA
    queue_depth_threshold: int = QUEUE_DEPTH_THRESHOLD
    chunk_chars: int = 256
    prefix_chunks: int = 4
    virtual_tick_seconds: float = VIRTUAL_TICK_SECONDS

    def validate(self) -> None:
        if self.profile not in {"smoke", "formal"}:
            raise ValueError(f"unknown profile {self.profile!r}")
        if self.replicas < 2:
            raise ValueError("F2d needs at least two replicas")
        if self.families != self.train_families + self.heldout_families:
            raise ValueError("train and held-out family counts must cover the trace")
        if min(self.requests_per_family, self.train_families, self.heldout_families) < 1:
            raise ValueError("trace counts must be positive")
        if self.alpha != 1.0:
            raise ValueError("F2d canonical alpha must equal 1")
        if self.baseline_lambda not in self.lambdas:
            raise ValueError("the hand baseline must be one of the tuned arms")
        if tuple(sorted(set(self.lambdas))) != self.lambdas:
            raise ValueError("lambda grid must be unique and increasing")
        for value in self.lambdas:
            if not math.isfinite(value) or value < 0:
                raise ValueError("lambda grid values must be finite and non-negative")


def profile_config(profile: str) -> WorkloadConfig:
    if profile == "formal":
        config = WorkloadConfig(
            profile="formal",
            seed=DEFAULT_SEED,
            replicas=200,
            families=64,
            requests_per_family=16,
            train_families=48,
            heldout_families=16,
        )
    elif profile == "smoke":
        config = WorkloadConfig(
            profile="smoke",
            seed=DEFAULT_SEED,
            replicas=8,
            families=8,
            requests_per_family=8,
            train_families=6,
            heldout_families=2,
        )
    else:
        raise ValueError(f"unknown profile {profile!r}")
    config.validate()
    return config


@dataclass(frozen=True)
class FamilyEpisode:
    family_id: str
    split: str
    seed_replica_id: str
    seed_prompt: str
    prefix_fingerprint: str
    items: tuple[f2a.TraceItem, ...]

    @property
    def trace_item_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self.items)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX
    )


def _config_dict(config: WorkloadConfig) -> dict[str, object]:
    result = asdict(config)
    result["lambdas"] = list(config.lambdas)
    return result


def _config_from_dict(value: Mapping[str, object]) -> WorkloadConfig:
    profile = value.get("profile")
    if not isinstance(profile, str):
        raise ValueError("config profile is missing")
    expected = profile_config(profile)
    if dict(value) != _config_dict(expected):
        raise ValueError("artifact config does not match the frozen profile")
    return expected


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"invalid JSONL framing at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{path}"),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_clean() -> bool:
    lines = _git_output("status", "--porcelain", "--untracked-files=all").splitlines()
    return all(
        line.startswith("?? .claude/worktrees/")
        or line == "?? .claude/worktrees"
        for line in lines
    )


def _source_snapshot(*, include_clean: bool) -> dict[str, object]:
    commit = _git_output("rev-parse", "HEAD")
    expected_commit = os.environ.get("F2D_EXPECTED_COMMIT")
    result: dict[str, object] = {
        "commit": commit,
        "expected_commit": expected_commit,
        "expected_commit_matches": expected_commit is None or expected_commit == commit,
        "files": [
            {"path": path, "sha256": sha256_file(_REPO_ROOT / path)}
            for path in _SOURCE_PATHS
        ],
    }
    if include_clean:
        result["git_clean_at_start"] = _git_clean()
    return result


def source_descriptor() -> dict[str, object]:
    return _source_snapshot(include_clean=True)


def source_end_descriptor() -> dict[str, object]:
    return _source_snapshot(include_clean=False)


def environment_descriptor() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "note": "host timing is diagnostic only; all F2d metrics use integer virtual ticks",
    }


def _input_paths() -> dict[str, Path]:
    return {
        "f2a_manifest": _F2A_MANIFEST,
        "f2a_raw": _F2A_RAW,
        "f2b_manifest": _F2B_MANIFEST,
        "f2b_raw": _F2B_RAW,
    }


@lru_cache(maxsize=1)
def _final_f2b_topology() -> tuple[str, ...]:
    final_rows = [
        row
        for row in _read_jsonl(_F2B_RAW)
        if row.get("type") == "fleet_snapshot" and row.get("phase") == "final"
    ]
    if len(final_rows) != 1:
        raise ValueError("retained F2b evidence must contain one final fleet snapshot")
    ids = final_rows[0].get("eligible_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != 200
        or not all(isinstance(value, str) for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("retained F2b final eligible topology is invalid")
    # Membership replacement changes dictionary insertion order in the live
    # pool.  F2d needs one policy-independent canonical order for tie breaks,
    # so use the same 200 proven IDs in lexical ordinal order.
    return tuple(sorted(ids))


def input_descriptor(config: WorkloadConfig) -> dict[str, object]:
    paths = _input_paths()
    actual = {name: sha256_file(path) for name, path in paths.items()}
    topology = _final_f2b_topology()[: config.replicas]
    return {
        "files": {
            name: {
                "path": str(path.relative_to(_REPO_ROOT)),
                "sha256": actual[name],
                "pinned_sha256": _PINNED_INPUT_SHA256[name],
            }
            for name, path in paths.items()
        },
        "all_pinned_digests_match": actual == _PINNED_INPUT_SHA256,
        "f2a_trace_contract": {
            "seed": config.seed,
            "families": config.families,
            "requests_per_family": config.requests_per_family,
            "source_profile": "formal",
        },
        "f2b_final_eligible_count": len(topology),
        "f2b_final_eligible_digest": sha256_text(canonical_json(topology)),
    }


def _f2a_config(config: WorkloadConfig) -> f2a.WorkloadConfig:
    base = f2a.profile_config("formal", seed=config.seed)
    return replace(
        base,
        profile=f"f2d-{config.profile}",
        replicas=config.replicas,
        shared_families=config.families,
        shared_requests_per_family=config.requests_per_family,
    )


@lru_cache(maxsize=2)
def build_episodes(config: WorkloadConfig) -> tuple[FamilyEpisode, ...]:
    """Build family-isolated train/held-out episodes from the F2a generator."""

    config.validate()
    topology = _final_f2b_topology()[: config.replicas]
    f2a_config = _f2a_config(config)
    trace = f2a.shared_trace(f2a_config)
    grouped: dict[str, list[f2a.TraceItem]] = defaultdict(list)
    for item in trace:
        grouped[item.family_id].append(item)
    seeds = {
        str(row["family_id"]): row
        for row in f2a.cache_seed_rows(f2a_config)
    }
    if set(grouped) != set(seeds) or len(grouped) != config.families:
        raise RuntimeError("F2a trace and cache-seed families diverged")

    split_order = sorted(
        grouped,
        key=lambda family_id: sha256_text(f"f2d-split-v1:{family_id}"),
    )
    train_ids = set(split_order[: config.train_families])
    episodes = []
    for family_id in sorted(grouped):
        items = tuple(grouped[family_id])
        if len(items) != config.requests_per_family:
            raise RuntimeError("F2a family request count diverged")
        seed = seeds[family_id]
        seed_replica = str(seed["replica_id"])
        if seed_replica not in topology:
            raise RuntimeError("F2a seed escaped the reused F2b topology")
        prefix_fingerprints = {item.prefix_fingerprint for item in items}
        if prefix_fingerprints != {str(seed["prefix_fingerprint"])}:
            raise RuntimeError("F2a family and seed roots diverged")
        episodes.append(
            FamilyEpisode(
                family_id=family_id,
                split=("train" if family_id in train_ids else "heldout"),
                seed_replica_id=seed_replica,
                seed_prompt=f2a._cache_seed_prompt(f2a_config, family_id),
                prefix_fingerprint=str(seed["prefix_fingerprint"]),
                items=items,
            )
        )
    return tuple(episodes)


def episode_descriptor(episode: FamilyEpisode) -> dict[str, object]:
    return {
        "type": "family",
        "family_id": episode.family_id,
        "split": episode.split,
        "seed_replica_id": episode.seed_replica_id,
        "seed_prompt_sha256": sha256_text(episode.seed_prompt),
        "prefix_fingerprint_sha256": sha256_text(episode.prefix_fingerprint),
        "trace_items": [
            {
                "trace_item_id": item.request_id,
                "prompt_sha256": item.prompt_sha256,
                "session_sha256": item.session_sha256,
            }
            for item in episode.items
        ],
    }


@lru_cache(maxsize=2)
def trace_descriptor(config: WorkloadConfig) -> dict[str, object]:
    episodes = build_episodes(config)
    rows = [episode_descriptor(episode) for episode in episodes]
    return {
        "families": config.families,
        "requests": config.families * config.requests_per_family,
        "train_families": sum(row["split"] == "train" for row in rows),
        "heldout_families": sum(row["split"] == "heldout" for row in rows),
        "sha256": sha256_text(canonical_json(rows)),
    }


def _arm_token(beta: float) -> str:
    return format(beta, ".10g").replace(".", "p")


def request_id_for(split: str, beta: float, item: f2a.TraceItem) -> str:
    return f"f2d:{split}:b{_arm_token(beta)}:{item.request_id}"


def nearest_rank_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return float(ordered[math.ceil(quantile * len(ordered)) - 1])


def _rendezvous_winner(session_id: str, replica_ids: Sequence[str]) -> str:
    return max(
        replica_ids,
        key=lambda replica_id: hashlib.sha256(
            f"{session_id}:{replica_id}".encode()
        ).digest(),
    )


@dataclass
class _VirtualPending:
    entered: asyncio.Event
    release: asyncio.Future[GenerationResult] | None = None


class _VirtualController:
    """Deterministic FIFO prompt-work queues shared by virtual backends."""

    def __init__(
        self,
        *,
        prefix_index: PrefixIndex,
        config: WorkloadConfig,
        episode: FamilyEpisode,
        beta: float,
    ) -> None:
        self.prefix_index = prefix_index
        self.config = config
        self.episode = episode
        self.beta = beta
        self.pending: dict[str, _VirtualPending] = {}
        self.queue_work: dict[str, int] = defaultdict(int)
        self.outcomes: dict[str, dict[str, object]] = {}

    def prepare(self, request_id: str) -> asyncio.Event:
        if request_id in self.pending:
            raise RuntimeError(f"duplicate virtual request {request_id}")
        event = asyncio.Event()
        self.pending[request_id] = _VirtualPending(entered=event)
        return event

    async def enter(
        self,
        replica_id: str,
        request: GenerationRequest,
    ) -> GenerationResult:
        pending = self.pending[request.request_id]
        overlap = self.prefix_index.overlap(replica_id, request.prompt)
        prompt_chunks = math.ceil(len(request.prompt) / self.config.chunk_chars)
        uncached_chunks = prompt_chunks - overlap
        if uncached_chunks < 1:
            raise RuntimeError("virtual request must retain one uncached prompt chunk")
        queue_before = self.queue_work[replica_id]
        ttft_ticks = queue_before + uncached_chunks
        self.queue_work[replica_id] += uncached_chunks
        self.outcomes[request.request_id] = {
            "kind": "placement_outcome",
            "request_id": request.request_id,
            "trace_item_id": request.request_id.rsplit(":", 1)[-1],
            "family_id": self.episode.family_id,
            "split": self.episode.split,
            "alpha": ALPHA,
            "beta": self.beta,
            "replica_id": replica_id,
            "ttft_s": float(ttft_ticks) * self.config.virtual_tick_seconds,
            "ttft_ticks": ttft_ticks,
            "success": True,
            "prompt_sha256": sha256_text(request.prompt),
            "overlap_chunks": overlap,
            "prompt_chunks": prompt_chunks,
            "uncached_chunks": uncached_chunks,
            "virtual_queue_work_before": queue_before,
        }
        pending.release = asyncio.get_running_loop().create_future()
        pending.entered.set()
        return await pending.release

    def release_all(self) -> None:
        for request_id, pending in self.pending.items():
            if pending.release is None:
                raise RuntimeError(f"request {request_id} never reached a backend")
            outcome = self.outcomes[request_id]
            if not pending.release.done():
                pending.release.set_result(
                    GenerationResult(
                        request_id=request_id,
                        prompt="",
                        completions=(),
                        finished=True,
                        usage=GenerationUsage(
                            prompt_tokens=int(outcome["prompt_chunks"]),
                            completion_tokens=1,
                            cached_tokens=int(outcome["overlap_chunks"]),
                        ),
                    )
                )


class _VirtualBackend:
    def __init__(self, replica_id: str, controller: _VirtualController) -> None:
        self.replica_id = replica_id
        self.controller = controller

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return await self.controller.enter(self.replica_id, request)

    async def stream(self, request: GenerationRequest):
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


async def _await_placement(task: asyncio.Task[GenerationResult], entered: asyncio.Event) -> None:
    waiter = asyncio.create_task(entered.wait())
    done, _pending = await asyncio.wait(
        (task, waiter),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if task in done:
        waiter.cancel()
        # A virtual backend cannot complete before release.  Reaching here is
        # either a placement exception or a broken controller contract.
        task.result()
        raise RuntimeError("virtual request completed before the placement barrier")
    await waiter


async def _run_episode(
    config: WorkloadConfig,
    episode: FamilyEpisode,
    beta: float,
    router_log: JsonlRouterLog,
) -> list[dict[str, object]]:
    topology = _final_f2b_topology()[: config.replicas]
    index = PrefixIndex(chunk_chars=config.chunk_chars)
    index.observe(episode.seed_replica_id, episode.seed_prompt)
    controller = _VirtualController(
        prefix_index=index,
        config=config,
        episode=episode,
        beta=beta,
    )
    backends = {
        replica_id: _VirtualBackend(replica_id, controller)
        for replica_id in topology
    }
    pool = ReplicaPool(
        backends,
        log=router_log,
        queue_depth_threshold=config.queue_depth_threshold,
        prefix_index=index,
        prefix_alpha=ALPHA,
        prefix_beta=beta,
    )
    tasks: list[asyncio.Task[GenerationResult]] = []
    try:
        for item in episode.items:
            request_id = request_id_for(episode.split, beta, item)
            entered = controller.prepare(request_id)
            task = asyncio.create_task(
                pool.generate(
                    GenerationRequest(
                        request_id=request_id,
                        prompt=item.prompt,
                        sampling_params=SamplingParams(
                            temperature=0.0,
                            max_tokens=1,
                            min_tokens=1,
                            ignore_eos=True,
                        ),
                        cache_hint=CacheHint(
                            session_id=item.session_id,
                            prefix_fingerprint=item.prefix_fingerprint,
                        ),
                    )
                )
            )
            tasks.append(task)
            await _await_placement(task, entered)
        controller.release_all()
        results = await asyncio.gather(*tasks)
        if len(results) != len(episode.items) or not all(result.finished for result in results):
            raise RuntimeError("virtual episode did not finish every request")
        return [
            controller.outcomes[request_id_for(episode.split, beta, item)]
            for item in episode.items
        ]
    except BaseException:
        controller.release_all()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_policies(
    config: WorkloadConfig,
    episodes: Sequence[FamilyEpisode],
    betas: Sequence[float],
    router_log: JsonlRouterLog,
) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    for beta in betas:
        canonical_alpha, canonical_beta = canonicalize_prefix_weights(ALPHA, beta)
        if canonical_alpha != ALPHA or canonical_beta != beta:
            raise RuntimeError("frozen lambda grid is not canonical")
        for episode in episodes:
            outcomes.extend(await _run_episode(config, episode, beta, router_log))
    return outcomes


def _expected_episode_rows(
    config: WorkloadConfig,
    episode: FamilyEpisode,
    beta: float,
) -> list[dict[str, object]]:
    """Replay production placement plus the virtual FIFO without wall time."""

    topology = _final_f2b_topology()[: config.replicas]
    outstanding = dict.fromkeys(topology, 0)
    queue_work = dict.fromkeys(topology, 0)
    expected: list[dict[str, object]] = []
    for item in episode.items:
        warm_score = config.prefix_chunks - beta * outstanding[episode.seed_replica_id]
        if warm_score >= 0.0:
            replica_id = episode.seed_replica_id
            reason = "prefix_match"
        else:
            affine = _rendezvous_winner(item.session_id, topology)
            if outstanding[affine] > config.queue_depth_threshold:
                replica_id = min(topology, key=outstanding.__getitem__)
                reason = "queue_depth_fallback"
            else:
                replica_id = affine
                reason = "session_affinity"
        overlap = config.prefix_chunks if replica_id == episode.seed_replica_id else 0
        prompt_chunks = math.ceil(len(item.prompt) / config.chunk_chars)
        uncached_chunks = prompt_chunks - overlap
        ttft_ticks = queue_work[replica_id] + uncached_chunks
        expected.append(
            {
                "request_id": request_id_for(episode.split, beta, item),
                "trace_item_id": item.request_id,
                "family_id": episode.family_id,
                "split": episode.split,
                "replica_id": replica_id,
                "reason": reason,
                "prompt_sha256": item.prompt_sha256,
                "session_sha256": item.session_sha256,
                "overlap_chunks": overlap,
                "prompt_chunks": prompt_chunks,
                "uncached_chunks": uncached_chunks,
                "virtual_queue_work_before": queue_work[replica_id],
                "ttft_ticks": ttft_ticks,
                "ttft_s": float(ttft_ticks) * config.virtual_tick_seconds,
                "outstanding_before": outstanding[replica_id],
            }
        )
        outstanding[replica_id] += 1
        queue_work[replica_id] += uncached_chunks
    return expected


def _rows_by_request(
    rows: Iterable[Mapping[str, object]],
    *,
    kind: str,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if row.get("kind") != kind:
            continue
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"{kind} row has no request_id")
        if request_id in result:
            raise ValueError(f"duplicate {kind} request_id {request_id!r}")
        result[request_id] = row
    return result


def _tuning_row(raw_rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    rows = [row for row in raw_rows if row.get("type") == "tuning"]
    if len(rows) != 1:
        raise ValueError("raw evidence must contain exactly one tuning row")
    return rows[0]


def _policy_stats(
    records: Sequence[Any],
    outcomes: Mapping[str, Mapping[str, object]],
    *,
    split: str,
    beta: float,
) -> dict[str, object]:
    selected = [
        record
        for record in records
        if record.split == split and record.alpha == ALPHA and record.beta == beta
    ]
    ticks = [int(outcomes[record.request_id]["ttft_ticks"]) for record in selected]
    return {
        "requests": len(selected),
        "successes": sum(record.success for record in selected),
        "total_ttft_ticks": sum(ticks),
        "mean_ttft_ticks": (sum(ticks) / len(ticks) if ticks else None),
        "p95_ttft_ticks": (nearest_rank_percentile(ticks, 0.95) if ticks else None),
        "prefix_match_count": sum(record.reason == "prefix_match" for record in selected),
        "queue_depth_fallback_count": sum(
            record.reason == "queue_depth_fallback" for record in selected
        ),
        "session_affinity_count": sum(
            record.reason == "session_affinity" for record in selected
        ),
        "cached_prompt_chunks": sum(
            int(outcomes[record.request_id]["overlap_chunks"])
            for record in selected
        ),
    }


def _derive_metrics(
    config: WorkloadConfig,
    router_rows: Sequence[Mapping[str, object]],
    raw_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    outcomes = _rows_by_request(raw_rows, kind="placement_outcome")
    records = build_placement_dataset([*router_rows, *outcomes.values()])
    selected_alpha, selected_beta = tune_prefix_weights(records)
    if selected_alpha != ALPHA:
        raise ValueError("tuner returned non-canonical alpha")

    train = {
        format(beta, ".10g"): _policy_stats(
            records,
            outcomes,
            split="train",
            beta=beta,
        )
        for beta in config.lambdas
    }
    heldout_baseline = _policy_stats(
        records,
        outcomes,
        split="heldout",
        beta=config.baseline_lambda,
    )
    heldout_tuned = _policy_stats(
        records,
        outcomes,
        split="heldout",
        beta=selected_beta,
    )
    baseline_by_item = {
        record.trace_item_id: record
        for record in records
        if record.split == "heldout" and record.beta == config.baseline_lambda
    }
    tuned_by_item = {
        record.trace_item_id: record
        for record in records
        if record.split == "heldout" and record.beta == selected_beta
    }
    shared_items = sorted(set(baseline_by_item) & set(tuned_by_item))
    disagreements = sum(
        (
            baseline_by_item[item].replica_id,
            baseline_by_item[item].reason,
        )
        != (
            tuned_by_item[item].replica_id,
            tuned_by_item[item].reason,
        )
        for item in shared_items
    )
    baseline_mean = heldout_baseline["mean_ttft_ticks"]
    tuned_mean = heldout_tuned["mean_ttft_ticks"]
    ratio = (
        float(tuned_mean) / float(baseline_mean)
        if isinstance(tuned_mean, (int, float))
        and isinstance(baseline_mean, (int, float))
        and baseline_mean > 0
        else None
    )
    return {
        "selected": {"alpha": selected_alpha, "beta": selected_beta},
        "baseline": {"alpha": ALPHA, "beta": config.baseline_lambda},
        "train": train,
        "heldout": {
            "baseline": heldout_baseline,
            "tuned": heldout_tuned,
            "mean_ttft_ratio": ratio,
            "mean_ttft_improvement_ticks": (
                float(baseline_mean) - float(tuned_mean)
                if isinstance(tuned_mean, (int, float))
                and isinstance(baseline_mean, (int, float))
                else None
            ),
            "action_disagreement_count": disagreements,
            "action_disagreement_rate": (
                disagreements / len(shared_items) if shared_items else None
            ),
        },
    }


def _evaluate_rows(
    config: WorkloadConfig,
    router_rows: Sequence[Mapping[str, object]],
    raw_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, bool], dict[str, object]]:
    """Independently replay raw routes/outcomes and derive the binding gate."""

    checks: dict[str, bool] = {}
    checks["raw_row_indices_are_contiguous"] = all(
        row.get("row_index") == index for index, row in enumerate(raw_rows)
    )
    checks["raw_row_types_are_known"] = all(
        row.get("type") in {"run", "family", "tuning"}
        or row.get("kind") == "placement_outcome"
        for row in raw_rows
    )
    run_rows = [row for row in raw_rows if row.get("type") == "run"]
    checks["exactly_one_run_row_is_first"] = (
        len(run_rows) == 1 and bool(raw_rows) and raw_rows[0] is run_rows[0]
    )

    episodes = build_episodes(config)
    expected_family_rows = [episode_descriptor(episode) for episode in episodes]
    actual_family_rows = [
        {key: value for key, value in row.items() if key != "row_index"}
        for row in raw_rows
        if row.get("type") == "family"
    ]
    checks["family_trace_and_split_match_generator"] = (
        actual_family_rows == expected_family_rows
    )
    checks["families_are_disjoint_between_train_and_heldout"] = (
        {episode.family_id for episode in episodes if episode.split == "train"}
        .isdisjoint(
            {
                episode.family_id
                for episode in episodes
                if episode.split == "heldout"
            }
        )
    )

    outcomes = _rows_by_request(raw_rows, kind="placement_outcome")
    routes = _rows_by_request(router_rows, kind="replica")
    checks["production_routes_join_outcomes_one_to_one"] = (
        bool(routes) and set(routes) == set(outcomes)
    )
    checks["router_rows_are_all_production_replica_records"] = (
        len(routes) == len(router_rows)
    )

    tuning = _tuning_row(raw_rows)
    tuning_index = raw_rows.index(tuning)
    outcome_positions = [
        (index, row)
        for index, row in enumerate(raw_rows)
        if row.get("kind") == "placement_outcome"
    ]
    train_positions = [
        index for index, row in outcome_positions if row.get("split") == "train"
    ]
    heldout_positions = [
        index for index, row in outcome_positions if row.get("split") == "heldout"
    ]
    checks["all_train_outcomes_precede_tuning_and_all_heldout_follow"] = (
        bool(train_positions)
        and bool(heldout_positions)
        and max(train_positions) < tuning_index < min(heldout_positions)
        and all(
            row.get("split") == ("train" if index < tuning_index else "heldout")
            for index, row in outcome_positions
        )
    )
    selected_alpha = tuning.get("selected_alpha")
    selected_beta = tuning.get("selected_beta")
    checks["tuning_row_is_canonical"] = (
        selected_alpha == ALPHA
        and isinstance(selected_beta, (int, float))
        and canonicalize_prefix_weights(
            float(selected_alpha),
            float(selected_beta),
        )
        == (ALPHA, float(selected_beta))
    )
    train_episodes = [episode for episode in episodes if episode.split == "train"]
    heldout_episodes = [episode for episode in episodes if episode.split == "heldout"]
    heldout_betas = (
        tuple(dict.fromkeys((config.baseline_lambda, float(selected_beta))))
        if isinstance(selected_beta, (int, float))
        else ()
    )

    replay_ok = True
    expected_request_ids: set[str] = set()
    for split, selected_episodes, betas in (
        ("train", train_episodes, config.lambdas),
        ("heldout", heldout_episodes, heldout_betas),
    ):
        for beta in betas:
            for episode in selected_episodes:
                for expected in _expected_episode_rows(config, episode, beta):
                    request_id = str(expected["request_id"])
                    expected_request_ids.add(request_id)
                    outcome = outcomes.get(request_id)
                    route = routes.get(request_id)
                    if outcome is None or route is None:
                        replay_ok = False
                        continue
                    outcome_ok = (
                        outcome.get("trace_item_id") == expected["trace_item_id"]
                        and outcome.get("family_id") == episode.family_id
                        and outcome.get("split") == split
                        and outcome.get("alpha") == ALPHA
                        and outcome.get("beta") == beta
                        and outcome.get("replica_id") == expected["replica_id"]
                        and outcome.get("ttft_ticks") == expected["ttft_ticks"]
                        and outcome.get("ttft_s") == expected["ttft_s"]
                        and outcome.get("success") is True
                        and outcome.get("prompt_sha256") == expected["prompt_sha256"]
                        and outcome.get("overlap_chunks") == expected["overlap_chunks"]
                        and outcome.get("prompt_chunks") == expected["prompt_chunks"]
                        and outcome.get("uncached_chunks") == expected["uncached_chunks"]
                        and outcome.get("virtual_queue_work_before")
                        == expected["virtual_queue_work_before"]
                    )
                    route_ok = (
                        route.get("request_id") == request_id
                        and route.get("replica_id") == expected["replica_id"]
                        and route.get("reason") == expected["reason"]
                        and route.get("session_sha256") == expected["session_sha256"]
                        and route.get("pool_size") == config.replicas
                        and route.get("eligible_size") == config.replicas
                        and isinstance(route.get("replica_generation"), str)
                        and bool(route.get("replica_generation"))
                        and isinstance(route.get("placement_started_ns"), int)
                        and isinstance(route.get("selected_at_ns"), int)
                        and isinstance(route.get("placement_latency_ns"), int)
                        and int(route["placement_started_ns"])
                        <= int(route["selected_at_ns"])
                        and route.get("placement_latency_ns")
                        == int(route["selected_at_ns"])
                        - int(route["placement_started_ns"])
                    )
                    replay_ok = replay_ok and outcome_ok and route_ok
    checks["every_policy_state_and_virtual_ttft_replays"] = replay_ok
    checks["no_extra_or_missing_measured_requests"] = (
        set(routes) == expected_request_ids == set(outcomes)
    )

    records = build_placement_dataset([*router_rows, *outcomes.values()])
    tuned = tune_prefix_weights(records)
    checks["dataset_tuner_recomputes_frozen_selection_from_train"] = tuned == (
        selected_alpha,
        selected_beta,
    )
    train_items_by_beta = {
        beta: {
            record.trace_item_id
            for record in records
            if record.split == "train" and record.beta == beta
        }
        for beta in config.lambdas
    }
    expected_train_items = {
        item.request_id for episode in train_episodes for item in episode.items
    }
    checks["train_policy_panel_is_exactly_balanced"] = (
        set(train_items_by_beta) == set(config.lambdas)
        and all(items == expected_train_items for items in train_items_by_beta.values())
    )
    heldout_items_by_beta = {
        beta: {
            record.trace_item_id
            for record in records
            if record.split == "heldout" and record.beta == beta
        }
        for beta in heldout_betas
    }
    expected_heldout_items = {
        item.request_id for episode in heldout_episodes for item in episode.items
    }
    checks["heldout_contains_only_tuned_and_baseline_on_identical_trace"] = (
        set(heldout_items_by_beta) == set(heldout_betas)
        and all(
            items == expected_heldout_items
            for items in heldout_items_by_beta.values()
        )
    )
    checks["all_replayed_requests_succeeded"] = all(record.success for record in records)
    checks["production_log_contains_prefix_match"] = any(
        record.reason == "prefix_match" for record in records
    )

    metrics = _derive_metrics(config, router_rows, raw_rows)
    baseline_mean = metrics["heldout"]["baseline"]["mean_ttft_ticks"]  # type: ignore[index]
    tuned_mean = metrics["heldout"]["tuned"]["mean_ttft_ticks"]  # type: ignore[index]
    checks["heldout_mean_ttft_strictly_improves"] = (
        isinstance(tuned_mean, (int, float))
        and isinstance(baseline_mean, (int, float))
        and tuned_mean < baseline_mean
    )
    return checks, metrics


def _source_end_matches_start(
    source: object,
    source_end: object,
) -> bool:
    if not isinstance(source, Mapping) or not isinstance(source_end, Mapping):
        return False
    return dict(source_end) == {
        key: source.get(key)
        for key in ("commit", "expected_commit", "expected_commit_matches", "files")
    }


def _verify_source_descriptor(
    config: WorkloadConfig,
    source: object,
) -> dict[str, bool]:
    """Verify source identity independently of manifest self-reporting."""

    if not isinstance(source, Mapping):
        return {
            "source_descriptor_schema_is_frozen": False,
            "source_commit_is_full_sha": False,
            "source_expected_commit_relation_recomputes": False,
            "source_file_set_and_hash_schema_are_frozen": False,
            "formal_source_hashes_match_commit_blobs": False,
        }
    checks: dict[str, bool] = {}
    checks["source_descriptor_schema_is_frozen"] = set(source) == {
        "commit",
        "expected_commit",
        "expected_commit_matches",
        "files",
        "git_clean_at_start",
    }
    commit = source.get("commit")
    commit_valid = (
        isinstance(commit, str)
        and len(commit) == 40
        and set(commit) <= _HEX
    )
    checks["source_commit_is_full_sha"] = commit_valid
    commit_exists = False
    if commit_valid:
        commit_exists = (
            subprocess.run(
                ("git", "cat-file", "-e", f"{commit}^{{commit}}"),
                cwd=_REPO_ROOT,
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )
    checks["source_commit_exists"] = commit_exists
    expected = source.get("expected_commit")
    expected_valid = expected is None or (
        isinstance(expected, str)
        and len(expected) == 40
        and set(expected) <= _HEX
    )
    checks["source_expected_commit_relation_recomputes"] = (
        expected_valid
        and source.get("expected_commit_matches")
        is (expected is None or expected == commit)
    )
    files = source.get("files")
    files_valid = (
        isinstance(files, list)
        and [row.get("path") for row in files if isinstance(row, Mapping)]
        in (list(_SOURCE_PATHS), list(_LEGACY_SOURCE_PATHS))
        and len(files) == len(_SOURCE_PATHS)
        and all(
            isinstance(row, Mapping)
            and set(row) == {"path", "sha256"}
            and _valid_sha256(row.get("sha256"))
            for row in files
        )
    )
    checks["source_file_set_and_hash_schema_are_frozen"] = files_valid
    blobs_match = False
    if commit_valid and files_valid:
        try:
            blobs_match = all(
                _git_blob_sha256(str(commit), str(row["path"]))
                == row.get("sha256")
                for row in files
            )
        except subprocess.CalledProcessError:
            blobs_match = False
    # Development smoke artifacts are intentionally measurable before their
    # source is committed.  The retained formal gate has no such exemption.
    checks["formal_source_hashes_match_commit_blobs"] = (
        blobs_match if config.profile == "formal" else True
    )
    if config.profile == "smoke" and files_valid:
        checks["smoke_source_commit_is_current_head"] = (
            commit_valid and commit == _git_output("rev-parse", "HEAD")
        )
        checks["smoke_source_hashes_match_current_files"] = all(
            sha256_file(_REPO_ROOT / str(row["path"])) == row.get("sha256")
            for row in files
        )
    return checks


def _formal_source_gate(config: WorkloadConfig, source: object, source_end: object) -> bool:
    if config.profile != "formal":
        return True
    source_checks = _verify_source_descriptor(config, source)
    return (
        isinstance(source, Mapping)
        and all(source_checks.values())
        and source.get("git_clean_at_start") is True
        and isinstance(source.get("expected_commit"), str)
        and bool(source.get("expected_commit"))
        and source.get("expected_commit_matches") is True
        and _source_end_matches_start(source, source_end)
    )


def _artifact_path(path: Path, name: str) -> Path:
    candidate = (path / name).resolve() if path.is_dir() else path.resolve()
    if candidate.name != name:
        raise ValueError(f"artifact path must resolve to {name}")
    return candidate


def verify_artifact(
    path: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    metrics: dict[str, object] | None = None
    manifest_path = _artifact_path(path, MANIFEST_NAME)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        checks["manifest_schema_is_frozen"] = set(manifest) == {
            "schema_version",
            "gate",
            "config",
            "inputs",
            "trace",
            "source",
            "source_end",
            "environment",
            "raw",
            "router",
            "metrics",
            "gate_checks",
            "pass_contract",
            "passed",
        }
        checks["schema_and_gate_match"] = (
            manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("gate") == GATE
        )
        config_raw = manifest.get("config")
        if not isinstance(config_raw, Mapping):
            raise ValueError("manifest config is missing")
        config = _config_from_dict(config_raw)
        checks["config_matches_frozen_profile"] = (
            dict(config_raw) == _config_dict(config)
        )
        raw_descriptor = manifest.get("raw")
        router_descriptor = manifest.get("router")
        if not isinstance(raw_descriptor, Mapping) or not isinstance(
            router_descriptor, Mapping
        ):
            raise ValueError("manifest raw descriptors are missing")
        checks["raw_descriptor_schemas_are_frozen"] = (
            set(raw_descriptor) == {"path", "sha256", "rows"}
            and set(router_descriptor) == {"path", "sha256", "rows"}
        )
        raw_path = (manifest_path.parent / str(raw_descriptor.get("path"))).resolve()
        router_path = (
            manifest_path.parent / str(router_descriptor.get("path"))
        ).resolve()
        checks["artifact_paths_are_adjacent_and_frozen"] = (
            raw_path.parent == manifest_path.parent
            and router_path.parent == manifest_path.parent
            and raw_path.name == RAW_NAME
            and router_path.name == ROUTER_NAME
        )
        raw_rows = _read_jsonl(raw_path)
        router_rows = _read_jsonl(router_path)
        checks["raw_hash_and_row_count_match"] = (
            _valid_sha256(raw_descriptor.get("sha256"))
            and raw_descriptor.get("sha256") == sha256_file(raw_path)
            and raw_descriptor.get("rows") == len(raw_rows)
        )
        checks["router_hash_and_row_count_match"] = (
            _valid_sha256(router_descriptor.get("sha256"))
            and router_descriptor.get("sha256") == sha256_file(router_path)
            and router_descriptor.get("rows") == len(router_rows)
        )
        inputs = input_descriptor(config)
        checks["retained_f2a_f2b_inputs_match_pinned_digests"] = (
            inputs.get("all_pinned_digests_match") is True
            and manifest.get("inputs") == inputs
        )
        checks["trace_descriptor_matches_generator"] = (
            manifest.get("trace") == trace_descriptor(config)
        )
        run_rows = [row for row in raw_rows if row.get("type") == "run"]
        checks["raw_run_row_is_first_unique_and_matches_manifest"] = (
            len(run_rows) == 1
            and bool(raw_rows)
            and raw_rows[0] is run_rows[0]
            and set(run_rows[0])
            == {
                "type",
                "schema_version",
                "gate",
                "config",
                "inputs",
                "trace",
                "source",
                "source_end",
                "environment",
                "row_index",
            }
            and {
                key: run_rows[0].get(key)
                for key in (
                    "schema_version",
                    "gate",
                    "config",
                    "inputs",
                    "trace",
                    "source",
                    "source_end",
                    "environment",
                )
            }
            == {
                key: manifest.get(key)
                for key in (
                    "schema_version",
                    "gate",
                    "config",
                    "inputs",
                    "trace",
                    "source",
                    "source_end",
                    "environment",
                )
            }
        )
        row_checks, metrics = _evaluate_rows(config, router_rows, raw_rows)
        checks.update(row_checks)
        checks["manifest_metrics_recompute_exactly"] = (
            manifest.get("metrics") == metrics
        )
        gate_checks = {
            name: value
            for name, value in row_checks.items()
            if name in _GATE_CHECK_NAMES
        }
        checks["gate_check_name_set_is_frozen"] = (
            set(gate_checks) == _GATE_CHECK_NAMES
        )
        checks["manifest_gate_checks_recompute_exactly"] = (
            manifest.get("gate_checks") == gate_checks
        )
        source = manifest.get("source")
        source_end = manifest.get("source_end")
        checks.update(_verify_source_descriptor(config, source))
        checks["source_did_not_drift_during_measurement"] = (
            _source_end_matches_start(source, source_end)
        )
        checks["formal_source_is_clean_exact_commit"] = _formal_source_gate(
            config,
            source,
            source_end,
        )
        expected_passed = (
            all(gate_checks.values())
            and _formal_source_gate(config, source, source_end)
        )
        checks["manifest_pass_contract_is_explicit"] = (
            manifest.get("pass_contract") == _PASS_CONTRACT
        )
        checks["manifest_passed_recomputes_exactly"] = (
            manifest.get("passed") is expected_passed
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        errors.append(f"{type(error).__name__}: {error}")

    passed = bool(checks) and all(checks.values()) and not errors
    if not passed:
        errors.extend(name for name, value in checks.items() if not value)
    result: dict[str, object] = {
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "errors": sorted(set(errors)),
        "manifest": str(manifest_path),
    }
    if assert_gate and not passed:
        raise RuntimeError("F2d artifact failed: " + ", ".join(result["errors"]))  # type: ignore[arg-type]
    return result


async def _measure(
    config: WorkloadConfig,
    router_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], tuple[float, float]]:
    episodes = build_episodes(config)
    train = [episode for episode in episodes if episode.split == "train"]
    heldout = [episode for episode in episodes if episode.split == "heldout"]
    router_log = JsonlRouterLog(
        router_path,
        max_pending_records=32_768,
        batch_size=256,
    )
    try:
        train_outcomes = await _run_policies(
            config,
            train,
            config.lambdas,
            router_log,
        )
        router_log.flush()
        train_routes = _read_jsonl(router_path)
        train_records = build_placement_dataset([*train_routes, *train_outcomes])
        selected = tune_prefix_weights(train_records)
        heldout_betas = tuple(
            dict.fromkeys((config.baseline_lambda, selected[1]))
        )
        heldout_outcomes = await _run_policies(
            config,
            heldout,
            heldout_betas,
            router_log,
        )
        router_log.flush()
    finally:
        router_log.close()
    return train_outcomes, heldout_outcomes, selected


def run_benchmark(
    output_dir: Path,
    *,
    profile: str = "formal",
    assert_gate: bool = False,
) -> Path:
    config = profile_config(profile)
    source = source_descriptor()
    environment = environment_descriptor()
    inputs = input_descriptor(config)
    trace = trace_descriptor(config)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    router_path = output_dir / ROUTER_NAME
    manifest_path = output_dir / MANIFEST_NAME
    existing = [path.name for path in (raw_path, router_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "F2d output is single-use; files already exist: " + ", ".join(existing)
        )

    train_outcomes, heldout_outcomes, selected = asyncio.run(
        _measure(config, router_path)
    )
    source_end = source_end_descriptor()
    run_row = {
        "type": "run",
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "config": _config_dict(config),
        "inputs": inputs,
        "trace": trace,
        "source": source,
        "source_end": source_end,
        "environment": environment,
    }
    family_rows = [episode_descriptor(episode) for episode in build_episodes(config)]
    tuning_row = {
        "type": "tuning",
        "selected_alpha": selected[0],
        "selected_beta": selected[1],
        "baseline_alpha": ALPHA,
        "baseline_beta": config.baseline_lambda,
        "selection_input": "train_only",
    }
    raw_rows: list[dict[str, object]] = [
        run_row,
        *family_rows,
        *train_outcomes,
        tuning_row,
        *heldout_outcomes,
    ]
    for index, row in enumerate(raw_rows):
        row["row_index"] = index
    _write_jsonl(raw_path, raw_rows)
    router_rows = _read_jsonl(router_path)
    row_checks, metrics = _evaluate_rows(config, router_rows, raw_rows)
    gate_checks = {
        name: value
        for name, value in row_checks.items()
        if name in _GATE_CHECK_NAMES
    }
    if set(gate_checks) != _GATE_CHECK_NAMES:
        raise RuntimeError("internal F2d gate-name set drifted")
    passed = all(gate_checks.values()) and _formal_source_gate(
        config,
        source,
        source_end,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "config": _config_dict(config),
        "inputs": inputs,
        "trace": trace,
        "source": source,
        "source_end": source_end,
        "environment": environment,
        "raw": {
            "path": RAW_NAME,
            "sha256": sha256_file(raw_path),
            "rows": len(raw_rows),
        },
        "router": {
            "path": ROUTER_NAME,
            "sha256": sha256_file(router_path),
            "rows": len(router_rows),
        },
        "metrics": metrics,
        "gate_checks": gate_checks,
        "pass_contract": _PASS_CONTRACT,
        "passed": passed,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verify_artifact(manifest_path, assert_gate=assert_gate)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--profile", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args(argv)

    if args.output_dir is not None:
        manifest_path = run_benchmark(
            args.output_dir,
            profile=args.profile,
            assert_gate=args.assert_gate,
        )
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        result = verify_artifact(
            args.verify_artifact,
            assert_gate=args.assert_gate,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
