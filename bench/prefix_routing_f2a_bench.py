#!/usr/bin/env python3
"""F2a: replayable 500-replica prefix-routing evidence gate.

The benchmark drives the production ``ReplicaPool.generate`` placement path
against 500 concrete mock backends.  Prefix-aware and session-HRW policies see
the same deterministic traces and the same initial simulated cache placement.
The selected backend, rather than a post-hoc model, reports whether the request
hit its cache.

Raw JSONL retains cache placement, request/family IDs, prompt hashes, actual
placement decisions/timings, paired round order, and round timings.  Prompt
bodies are deliberately absent.  ``--verify-artifact`` hashes and independently
replays that raw evidence instead of trusting manifest verdicts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from kairyu.engine.backend import (
    CacheHint,
    GenerationRequest,
    GenerationResult,
)
from kairyu.orchestration.prefix_index import PrefixIndex
from kairyu.orchestration.replica import ReplicaPool
from kairyu.sampling_params import SamplingParams

SCHEMA_VERSION = "kairyu.f2a.prefix-routing.v1"
REPLICA_COUNT = 500
DEFAULT_SEED = 0xF2A_2026
PREFIX_POLICY = "prefix_aware"
HRW_POLICY = "session_hrw"
POLICIES = (PREFIX_POLICY, HRW_POLICY)
RAW_NAME = "prefix-routing-f2a-raw.jsonl"
MANIFEST_NAME = "prefix-routing-f2a-manifest.json"
_CHUNK_CHARS = 256
_PREFIX_CHUNKS = 4
_HEX_SHA256 = frozenset("0123456789abcdef")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATHS = (
    "kairyu/orchestration/replica.py",
    "kairyu/orchestration/prefix_index.py",
    "bench/prefix_routing_f2a_bench.py",
    ".github/workflows/f2a-prefix-routing.yml",
    "kairyu/engine/backend.py",
    "kairyu/sampling_params.py",
    "kairyu/telemetry.py",
    "kairyu/orchestration/router.py",
    "pyproject.toml",
    "uv.lock",
)
# One-sided 95% Student-t critical values for the two frozen paired profiles.
_T_CRITICAL_95_ONE_SIDED = {
    2: 2.919985580353725,
    20: 1.7247182429207863,
}


@dataclass(frozen=True)
class WorkloadConfig:
    profile: str
    seed: int
    replicas: int
    shared_families: int
    shared_requests_per_family: int
    uniform_rounds: int
    uniform_requests_per_round: int
    uniform_sign_min_rounds: int
    chunk_chars: int = _CHUNK_CHARS
    prefix_chunks: int = _PREFIX_CHUNKS
    hit_rate_ratio_threshold: float = 2.0
    placement_p99_limit_ms: float = 10.0
    uniform_equivalence_margin: float = 0.01


_PROFILE_COUNTS = {
    "smoke": {
        "shared_families": 16,
        "shared_requests_per_family": 8,
        "uniform_rounds": 3,
        "uniform_requests_per_round": 64,
        "uniform_sign_min_rounds": 2,
    },
    "formal": {
        "shared_families": 64,
        "shared_requests_per_family": 16,
        "uniform_rounds": 21,
        "uniform_requests_per_round": 512,
        "uniform_sign_min_rounds": 15,
    },
}


@dataclass(frozen=True)
class TraceItem:
    request_id: str
    family_id: str
    prompt: str
    session_id: str

    @property
    def prompt_sha256(self) -> str:
        return _sha256_text(self.prompt)

    @property
    def session_sha256(self) -> str:
        return _sha256_text(self.session_id)


def profile_config(profile: str, *, seed: int = DEFAULT_SEED) -> WorkloadConfig:
    try:
        counts = _PROFILE_COUNTS[profile]
    except KeyError as error:
        raise ValueError(f"unknown profile {profile!r}") from error
    return WorkloadConfig(
        profile=profile,
        seed=seed,
        replicas=REPLICA_COUNT,
        **counts,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{relative_path}"),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_is_ancestor(commit: str) -> bool:
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _git_clean_at_start() -> bool:
    lines = _git_output(
        "status", "--porcelain", "--untracked-files=all"
    ).splitlines()
    # Codex owns this persistent control directory.  It is not product source
    # and must neither invalidate evidence nor be staged/deleted by the gate.
    relevant = [
        line
        for line in lines
        if not (
            line.startswith("?? .claude/worktrees/")
            or line == "?? .claude/worktrees"
        )
    ]
    return not relevant


def _frozen_source_snapshot() -> dict[str, object]:
    commit = _git_output("rev-parse", "HEAD")
    expected_commit = os.environ.get("F2A_EXPECTED_COMMIT")
    return {
        "commit": commit,
        "expected_commit": expected_commit,
        "expected_commit_matches": (
            expected_commit is None or expected_commit == commit
        ),
        "files": [
            {
                "path": relative_path,
                "sha256": _sha256_file(_REPO_ROOT / relative_path),
            }
            for relative_path in _SOURCE_PATHS
        ],
    }


def source_descriptor() -> dict[str, object]:
    return {
        **_frozen_source_snapshot(),
        "git_clean_at_start": _git_clean_at_start(),
    }


def source_end_descriptor() -> dict[str, object]:
    """Re-read commit and frozen files after the measured workload."""
    return _frozen_source_snapshot()


def environment_descriptor() -> dict[str, object]:
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "sched_affinity": affinity,
        "github": {
            name: os.environ.get(name)
            for name in (
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_EVENT_NAME",
                "GITHUB_SHA",
            )
        },
    }


def _source_end_matches_start(source: object, source_end: object) -> bool:
    if not isinstance(source, dict) or not isinstance(source_end, dict):
        return False
    expected_end = {
        key: source.get(key)
        for key in (
            "commit",
            "expected_commit",
            "expected_commit_matches",
            "files",
        )
    }
    return source_end == expected_end


def _formal_context_complete(
    config: WorkloadConfig,
    source: object,
    environment: object,
) -> bool:
    if config.profile != "formal":
        return True
    if not isinstance(source, dict) or not isinstance(environment, dict):
        return False
    commit = source.get("commit")
    expected_commit = source.get("expected_commit")
    github = environment.get("github")
    if not isinstance(github, dict):
        return False
    run_id = github.get("GITHUB_RUN_ID")
    run_attempt = github.get("GITHUB_RUN_ATTEMPT")
    event_name = github.get("GITHUB_EVENT_NAME")
    github_sha = github.get("GITHUB_SHA")
    return (
        isinstance(commit, str)
        and isinstance(expected_commit, str)
        and bool(expected_commit)
        and expected_commit == commit
        and source.get("expected_commit_matches") is True
        and all(
            isinstance(value, str)
            and value.isascii()
            and value.isdecimal()
            and int(value) > 0
            for value in (run_id, run_attempt)
        )
        and event_name in {"pull_request", "workflow_dispatch"}
        and isinstance(github_sha, str)
        and len(github_sha) == 40
        and set(github_sha.lower()) <= _HEX_SHA256
    )


def _replica_ids(config: WorkloadConfig) -> tuple[str, ...]:
    return tuple(f"replica-{index:03d}" for index in range(config.replicas))


def _family_prefix(config: WorkloadConfig, family_id: str) -> str:
    token = _sha256_text(f"f2a:{config.seed}:family:{family_id}")
    unit = f"family={family_id}|{token}|"
    length = config.chunk_chars * config.prefix_chunks
    return (unit * math.ceil(length / len(unit)))[:length]


def _cache_seed_prompt(config: WorkloadConfig, family_id: str) -> str:
    suffix = _sha256_text(f"f2a:{config.seed}:seed:{family_id}")
    return _family_prefix(config, family_id) + f"|seed|{suffix}"


def _shared_prompt(
    config: WorkloadConfig, family_id: str, request_number: int
) -> str:
    suffix = _sha256_text(
        f"f2a:{config.seed}:shared:{family_id}:{request_number}"
    )
    return _family_prefix(config, family_id) + f"|query={request_number}|{suffix}"


def _uniform_prompt(config: WorkloadConfig, family_id: str) -> str:
    token = _sha256_text(f"f2a:{config.seed}:uniform:{family_id}")
    unit = f"uniform={family_id}|{token}|"
    length = config.chunk_chars * (config.prefix_chunks + 1)
    return (unit * math.ceil(length / len(unit)))[:length]


def _prompt_work_chunks(config: WorkloadConfig, prompt: str) -> int:
    """Prompt work units include the final partial text chunk."""
    return math.ceil(len(prompt) / config.chunk_chars)


def _session_id(config: WorkloadConfig, request_id: str) -> str:
    return f"f2a-session:{config.seed}:{request_id}"


def cache_seed_rows(config: WorkloadConfig) -> list[dict[str, object]]:
    rng = random.Random(config.seed ^ 0xCACE)
    replicas = _replica_ids(config)
    rows: list[dict[str, object]] = []
    for index in range(config.shared_families):
        family_id = f"shared-{index:03d}"
        prompt = _cache_seed_prompt(config, family_id)
        rows.append(
            {
                "type": "cache_seed",
                "family_id": family_id,
                "replica_id": replicas[rng.randrange(len(replicas))],
                "prompt_sha256": _sha256_text(prompt),
            }
        )
    return rows


def shared_trace(config: WorkloadConfig) -> tuple[TraceItem, ...]:
    items = []
    for family_index in range(config.shared_families):
        family_id = f"shared-{family_index:03d}"
        for request_number in range(config.shared_requests_per_family):
            request_id = f"shared-{family_index:03d}-{request_number:03d}"
            items.append(
                TraceItem(
                    request_id=request_id,
                    family_id=family_id,
                    prompt=_shared_prompt(config, family_id, request_number),
                    session_id=_session_id(config, request_id),
                )
            )
    random.Random(config.seed ^ 0x51A2ED).shuffle(items)
    return tuple(items)


def uniform_trace(
    config: WorkloadConfig, round_index: int
) -> tuple[TraceItem, ...]:
    items = []
    for request_number in range(config.uniform_requests_per_round):
        family_id = f"uniform-{round_index:03d}-{request_number:05d}"
        request_id = family_id
        items.append(
            TraceItem(
                request_id=request_id,
                family_id=family_id,
                prompt=_uniform_prompt(config, family_id),
                session_id=_session_id(config, request_id),
            )
        )
    random.Random(config.seed ^ (0xA110 + round_index)).shuffle(items)
    return tuple(items)


class _PlacementLog:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def record_replica(
        self,
        session_id: str | None,
        replica: int,
        reason: str,
        **fields: object,
    ) -> None:
        request_id = fields.get("request_id")
        if not isinstance(request_id, str) or request_id in self.records:
            raise RuntimeError("placement log request IDs must be unique strings")
        self.records[request_id] = {
            "session_id": session_id,
            "replica_ordinal": replica,
            "reason": reason,
            **fields,
        }

    async def shutdown(self) -> None:
        return None


class _MockReplica:
    """Concrete backend whose own cache state produces the observed hit."""

    def __init__(
        self,
        replica_id: str,
        cached_families: set[str],
        prompt_catalog: dict[str, str],
        observations: dict[str, dict[str, object]],
        config: WorkloadConfig,
    ) -> None:
        self.replica_id = replica_id
        self.cached_families = set(cached_families)
        self.prompt_catalog = prompt_catalog
        self.observations = observations
        self.config = config

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt_sha256 = _sha256_text(request.prompt)
        family_id = self.prompt_catalog.get(prompt_sha256)
        if family_id is None:
            raise RuntimeError("mock backend received a prompt outside the frozen trace")
        cache_hit = family_id in self.cached_families
        prompt_chunks = _prompt_work_chunks(self.config, request.prompt)
        cached_prefix_chunks_before = (
            self.config.prefix_chunks if cache_hit else 0
        )
        self.cached_families.add(family_id)
        self.observations[request.request_id] = {
            "backend_replica_id": self.replica_id,
            "cache_hit": cache_hit,
            "cached_prefix_chunks_before": cached_prefix_chunks_before,
            "prompt_chunks": prompt_chunks,
            "family_id": family_id,
            "prompt_sha256": prompt_sha256,
        }
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(),
            finished=True,
        )

    async def stream(self, request: GenerationRequest):
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


def _build_pool(
    config: WorkloadConfig,
    *,
    policy: str,
    seeds: list[dict[str, object]],
    trace: tuple[TraceItem, ...],
) -> tuple[
    ReplicaPool,
    _PlacementLog,
    dict[str, dict[str, object]],
]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    cached_by_replica = {replica_id: set() for replica_id in _replica_ids(config)}
    for seed in seeds:
        cached_by_replica[str(seed["replica_id"])].add(str(seed["family_id"]))
    catalog = {item.prompt_sha256: item.family_id for item in trace}
    observations: dict[str, dict[str, object]] = {}
    backends = {
        replica_id: _MockReplica(
            replica_id,
            cached_by_replica[replica_id],
            catalog,
            observations,
            config,
        )
        for replica_id in _replica_ids(config)
    }
    placement_log = _PlacementLog()
    prefix_index = PrefixIndex(chunk_chars=config.chunk_chars)
    if policy == PREFIX_POLICY:
        for seed in seeds:
            prefix_index.observe(
                str(seed["replica_id"]),
                _cache_seed_prompt(config, str(seed["family_id"])),
            )
    pool = ReplicaPool(
        backends,
        log=placement_log,  # type: ignore[arg-type]
        prefix_index=(prefix_index if policy == PREFIX_POLICY else None),
    )
    return pool, placement_log, observations


async def _dispatch_trace(
    config: WorkloadConfig,
    *,
    trace_kind: str,
    round_index: int | None,
    policy: str,
    policy_order_index: int,
    seeds: list[dict[str, object]],
    trace: tuple[TraceItem, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    pool, placement_log, observations = _build_pool(
        config,
        policy=policy,
        seeds=seeds,
        trace=trace,
    )
    try:
        placement_rows: list[dict[str, object]] = []
        wall_started_ns = time.perf_counter_ns()
        for sequence, item in enumerate(trace):
            request = GenerationRequest(
                request_id=item.request_id,
                prompt=item.prompt,
                sampling_params=SamplingParams(max_tokens=1),
                # Both arms carry the identical production request contract.
                # The treatment differs only by enabling the pool prefix index.
                cache_hint=CacheHint(session_id=item.session_id),
                placement_started_ns=time.perf_counter_ns(),
            )
            await pool.generate(request)
            # End the dispatch interval before reading or validating evidence so
            # reporting work cannot enter the uniform-arm goodput denominator.
            dispatch_finished_ns = time.perf_counter_ns()
            logged = placement_log.records.pop(item.request_id)
            observed = observations.pop(item.request_id)
            selected = str(logged["replica_id"])
            if observed["backend_replica_id"] != selected:
                raise RuntimeError("ReplicaPool log and selected backend disagree")
            if logged["session_id"] != item.session_id:
                raise RuntimeError(
                    "ReplicaPool log lost the frozen session identity"
                )
            placement_rows.append(
                {
                    "type": "placement",
                    "trace": trace_kind,
                    "round_index": round_index,
                    "policy": policy,
                    "policy_order_index": policy_order_index,
                    "sequence": sequence,
                    "request_id": item.request_id,
                    "family_id": item.family_id,
                    "prompt_sha256": item.prompt_sha256,
                    "session_sha256": item.session_sha256,
                    "selected_replica_id": selected,
                    "backend_replica_id": observed["backend_replica_id"],
                    "replica_ordinal": logged["replica_ordinal"],
                    "reason": logged["reason"],
                    "cache_hit": observed["cache_hit"],
                    "cached_prefix_chunks_before": observed[
                        "cached_prefix_chunks_before"
                    ],
                    "prompt_chunks": observed["prompt_chunks"],
                    "placement_started_ns": logged["placement_started_ns"],
                    "selected_at_ns": logged["selected_at_ns"],
                    "placement_latency_ns": logged["placement_latency_ns"],
                    "dispatch_finished_ns": dispatch_finished_ns,
                    "dispatch_latency_ns": (
                        dispatch_finished_ns - int(request.placement_started_ns)
                    ),
                    "pool_size": logged["pool_size"],
                    "eligible_size": logged["eligible_size"],
                }
            )
        wall_finished_ns = time.perf_counter_ns()
        if placement_log.records or observations:
            raise RuntimeError("undrained placement observations")
        placement_slo_ns = int(config.placement_p99_limit_ms * 1_000_000)
        summary = {
            "type": "round_summary",
            "trace": trace_kind,
            "round_index": round_index,
            "policy": policy,
            "policy_order_index": policy_order_index,
            "request_count": len(trace),
            "wall_started_ns": wall_started_ns,
            "wall_finished_ns": wall_finished_ns,
            "wall_ns": wall_finished_ns - wall_started_ns,
            "wall_role": "audit_only",
            "placement_sum_ns": sum(
                int(row["placement_latency_ns"]) for row in placement_rows
            ),
            "dispatch_sum_ns": sum(
                int(row["dispatch_latency_ns"]) for row in placement_rows
            ),
            "placement_slo_ns": placement_slo_ns,
            "placement_slo_success_count": sum(
                int(row["placement_latency_ns"]) < placement_slo_ns
                for row in placement_rows
            ),
        }
        return placement_rows, summary
    finally:
        # Cleanup 500 mock entries and the log after the measured wall interval.
        await pool.shutdown()


async def _collect_raw_rows(
    config: WorkloadConfig,
    source: dict[str, object],
    environment: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": SCHEMA_VERSION,
            "config": asdict(config),
            "source": source,
            "environment": environment,
        }
    ]
    seeds = cache_seed_rows(config)
    rows.extend(seeds)

    shared = shared_trace(config)
    for order_index, policy in enumerate(POLICIES):
        placements, summary = await _dispatch_trace(
            config,
            trace_kind="shared",
            round_index=None,
            policy=policy,
            policy_order_index=order_index,
            seeds=seeds,
            trace=shared,
        )
        rows.extend(placements)
        rows.append(summary)

    for round_index in range(config.uniform_rounds):
        order = (
            POLICIES if round_index % 2 == 0 else tuple(reversed(POLICIES))
        )
        rows.append(
            {
                "type": "paired_round",
                "trace": "uniform",
                "round_index": round_index,
                "policy_order": list(order),
            }
        )
        trace = uniform_trace(config, round_index)
        for order_index, policy in enumerate(order):
            placements, summary = await _dispatch_trace(
                config,
                trace_kind="uniform",
                round_index=round_index,
                policy=policy,
                policy_order_index=order_index,
                seeds=seeds,
                trace=trace,
            )
            rows.extend(placements)
            rows.append(summary)

    for row_index, row in enumerate(rows):
        row["row_index"] = row_index
    return rows


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def derive_metrics(
    rows: list[dict[str, object]], config: WorkloadConfig
) -> dict[str, object]:
    placements = [row for row in rows if row.get("type") == "placement"]
    shared = [row for row in placements if row.get("trace") == "shared"]
    prefix_shared = [
        row for row in shared if row.get("policy") == PREFIX_POLICY
    ]
    hrw_shared = [row for row in shared if row.get("policy") == HRW_POLICY]
    prefix_hits = sum(row.get("cache_hit") is True for row in prefix_shared)
    hrw_hits = sum(row.get("cache_hit") is True for row in hrw_shared)
    prefix_cached_chunks = sum(
        int(row["cached_prefix_chunks_before"]) for row in prefix_shared
    )
    hrw_cached_chunks = sum(
        int(row["cached_prefix_chunks_before"]) for row in hrw_shared
    )
    prefix_prompt_chunks = sum(
        int(row["prompt_chunks"]) for row in prefix_shared
    )
    hrw_prompt_chunks = sum(int(row["prompt_chunks"]) for row in hrw_shared)
    prefix_rate = (
        prefix_cached_chunks / prefix_prompt_chunks
        if prefix_prompt_chunks
        else 0.0
    )
    hrw_rate = (
        hrw_cached_chunks / hrw_prompt_chunks if hrw_prompt_chunks else 0.0
    )
    infinite_ratio = prefix_rate > 0.0 and hrw_rate == 0.0
    hit_ratio = None if infinite_ratio or hrw_rate == 0.0 else prefix_rate / hrw_rate

    prefix_latencies_by_trace = {
        trace_kind: [
            int(row["placement_latency_ns"])
            for row in placements
            if row.get("policy") == PREFIX_POLICY
            and row.get("trace") == trace_kind
        ]
        for trace_kind in ("shared", "uniform")
    }
    placement_p99_by_trace = {
        trace_kind: _nearest_rank(latencies, 0.99)
        for trace_kind, latencies in prefix_latencies_by_trace.items()
    }
    prefix_latencies = [
        latency
        for latencies in prefix_latencies_by_trace.values()
        for latency in latencies
    ]
    placement_p99_ns = _nearest_rank(prefix_latencies, 0.99)

    summaries = {
        (int(row["round_index"]), str(row["policy"])): row
        for row in rows
        if row.get("type") == "round_summary"
        and row.get("trace") == "uniform"
        and isinstance(row.get("round_index"), int)
    }
    paired_ratios = []
    paired_rounds = []
    placement_slo_ns = int(config.placement_p99_limit_ms * 1_000_000)
    for round_index in range(config.uniform_rounds):
        prefix = summaries[(round_index, PREFIX_POLICY)]
        hrw = summaries[(round_index, HRW_POLICY)]
        prefix_round_placements = [
            row
            for row in placements
            if row.get("trace") == "uniform"
            and row.get("round_index") == round_index
            and row.get("policy") == PREFIX_POLICY
        ]
        hrw_round_placements = [
            row
            for row in placements
            if row.get("trace") == "uniform"
            and row.get("round_index") == round_index
            and row.get("policy") == HRW_POLICY
        ]
        prefix_qualifying = [
            row
            for row in prefix_round_placements
            if int(row["placement_latency_ns"]) < placement_slo_ns
        ]
        hrw_qualifying = [
            row
            for row in hrw_round_placements
            if int(row["placement_latency_ns"]) < placement_slo_ns
        ]
        prefix_successes = len(prefix_qualifying)
        hrw_successes = len(hrw_qualifying)
        # All dispatched work remains in the denominator, including requests
        # whose placement missed the SLO; only the numerator is SLO-qualified.
        prefix_dispatch_sum_ns = int(prefix["dispatch_sum_ns"])
        hrw_dispatch_sum_ns = int(hrw["dispatch_sum_ns"])
        if prefix_dispatch_sum_ns <= 0 or hrw_dispatch_sum_ns <= 0:
            raise ValueError(
                "paired placement-SLO goodput requires positive dispatch "
                "time in both arms"
            )
        prefix_goodput = (
            prefix_successes * 1_000_000_000 / prefix_dispatch_sum_ns
        )
        hrw_goodput = (
            hrw_successes * 1_000_000_000 / hrw_dispatch_sum_ns
        )
        if prefix_goodput <= 0.0 or hrw_goodput <= 0.0:
            raise ValueError(
                "paired placement-SLO goodput requires a success in both arms"
            )
        ratio = prefix_goodput / hrw_goodput
        paired_ratios.append(ratio)
        paired_rounds.append(
            {
                "round_index": round_index,
                "policy_order": (
                    [PREFIX_POLICY, HRW_POLICY]
                    if round_index % 2 == 0
                    else [HRW_POLICY, PREFIX_POLICY]
                ),
                "placement_slo_ns": placement_slo_ns,
                "prefix_slo_success_count": prefix_successes,
                "hrw_slo_success_count": hrw_successes,
                "prefix_dispatch_sum_ns": prefix_dispatch_sum_ns,
                "hrw_dispatch_sum_ns": hrw_dispatch_sum_ns,
                "prefix_placement_slo_goodput_rps": prefix_goodput,
                "hrw_placement_slo_goodput_rps": hrw_goodput,
                "prefix_over_hrw_ratio": ratio,
            }
        )
    log_ratios = [math.log(ratio) for ratio in paired_ratios]
    log_mean = statistics.mean(log_ratios)
    log_stdev = statistics.stdev(log_ratios)
    degrees_of_freedom = len(log_ratios) - 1
    try:
        t_critical = _T_CRITICAL_95_ONE_SIDED[degrees_of_freedom]
    except KeyError as error:
        raise ValueError(
            f"no frozen one-sided t critical value for df={degrees_of_freedom}"
        ) from error
    log_lcb95 = log_mean - t_critical * log_stdev / math.sqrt(len(log_ratios))
    ratio_lcb95 = math.exp(log_lcb95)
    noninferiority_floor = 1.0 - config.uniform_equivalence_margin
    rounds_at_or_above_floor = sum(
        ratio >= noninferiority_floor for ratio in paired_ratios
    )
    return {
        "shared": {
            "request_count_per_policy": len(prefix_shared),
            "prefix_request_hits": prefix_hits,
            "hrw_request_hits": hrw_hits,
            "prefix_cached_prefix_chunks": prefix_cached_chunks,
            "prefix_prompt_chunks": prefix_prompt_chunks,
            "hrw_cached_prefix_chunks": hrw_cached_chunks,
            "hrw_prompt_chunks": hrw_prompt_chunks,
            "prefix_hit_rate": prefix_rate,
            "hrw_hit_rate": hrw_rate,
            "hit_rate_ratio": hit_ratio,
            "hit_rate_ratio_infinite": infinite_ratio,
        },
        "placement": {
            "prefix_sample_count": len(prefix_latencies),
            "prefix_p99_ns": placement_p99_ns,
            "prefix_p99_ms": placement_p99_ns / 1_000_000,
            "prefix_shared_sample_count": len(
                prefix_latencies_by_trace["shared"]
            ),
            "prefix_shared_p99_ns": placement_p99_by_trace["shared"],
            "prefix_shared_p99_ms": (
                placement_p99_by_trace["shared"] / 1_000_000
            ),
            "prefix_uniform_sample_count": len(
                prefix_latencies_by_trace["uniform"]
            ),
            "prefix_uniform_p99_ns": placement_p99_by_trace["uniform"],
            "prefix_uniform_p99_ms": (
                placement_p99_by_trace["uniform"] / 1_000_000
            ),
            "prefix_worst_trace_p99_ns": max(
                placement_p99_by_trace.values()
            ),
            "prefix_worst_trace_p99_ms": (
                max(placement_p99_by_trace.values()) / 1_000_000
            ),
        },
        "uniform": {
            "equivalence_margin": config.uniform_equivalence_margin,
            "goodput_definition": (
                "successful placements below the fixed 10ms placement SLO "
                "divided by summed raw dispatch intervals for every request "
                "in the arm; round wall is audit-only"
            ),
            "placement_slo_ns": placement_slo_ns,
            "noninferiority_floor": noninferiority_floor,
            "paired_log_ratio_lcb_method": "one-sided-95pct-student-t",
            "paired_log_ratio_degrees_of_freedom": degrees_of_freedom,
            "paired_log_ratio_t_critical": t_critical,
            "paired_log_ratio_mean": log_mean,
            "paired_log_ratio_sample_stdev": log_stdev,
            "paired_log_ratio_lcb95": log_lcb95,
            "paired_ratio_lcb95": ratio_lcb95,
            "paired_rounds": paired_rounds,
            "paired_round_count": len(paired_ratios),
            "rounds_at_or_above_noninferiority_floor": (
                rounds_at_or_above_floor
            ),
            "sign_gate_min_rounds": config.uniform_sign_min_rounds,
            "paired_median_prefix_over_hrw_ratio": statistics.median(
                paired_ratios
            ),
        },
    }


def threshold_checks(
    metrics: dict[str, object], config: WorkloadConfig
) -> dict[str, bool]:
    shared = metrics["shared"]
    placement = metrics["placement"]
    uniform = metrics["uniform"]
    assert isinstance(shared, dict)
    assert isinstance(placement, dict)
    assert isinstance(uniform, dict)
    ratio = shared["hit_rate_ratio"]
    ratio_pass = shared["hit_rate_ratio_infinite"] is True or (
        isinstance(ratio, (int, float))
        and ratio >= config.hit_rate_ratio_threshold
    )
    return {
        "shared_prefix_hit_rate_ratio_at_least_2": ratio_pass,
        "session_hrw_shared_hits_are_nonzero": (
            int(shared["hrw_request_hits"]) > 0
        ),
        "prefix_shared_and_uniform_p99_below_10ms": (
            float(placement["prefix_worst_trace_p99_ms"])
            < config.placement_p99_limit_ms
        ),
        "uniform_paired_goodput_lcb_noninferior": (
            float(uniform["paired_ratio_lcb95"])
            >= 1.0 - config.uniform_equivalence_margin
        ),
        "uniform_paired_goodput_sign_gate": (
            int(uniform["rounds_at_or_above_noninferiority_floor"])
            >= config.uniform_sign_min_rounds
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX_SHA256
    )


def _expected_hrw(
    session_id: str, replica_ids: tuple[str, ...]
) -> str:
    return max(
        replica_ids,
        key=lambda replica_id: hashlib.sha256(
            f"{session_id}:{replica_id}".encode()
        ).digest(),
    )


def _chunk_keys(prompt: str, chunk_chars: int) -> tuple[str, ...]:
    keys = []
    hasher = hashlib.sha256()
    position = 0
    for end in range(chunk_chars, len(prompt) + 1, chunk_chars):
        hasher.update(prompt[position:end].encode())
        keys.append(hasher.hexdigest()[:16])
        position = end
    return tuple(keys)


def _overlap(keys: tuple[str, ...], stored: set[str]) -> int:
    count = 0
    for key in keys:
        if key not in stored:
            break
        count += 1
    return count


def _replay_rows(
    rows: list[dict[str, object]], config: WorkloadConfig
) -> tuple[dict[str, bool], list[str]]:
    errors: list[str] = []
    replicas = _replica_ids(config)
    checks: dict[str, bool] = {}

    checks["row_indices_are_contiguous"] = all(
        row.get("row_index") == index for index, row in enumerate(rows)
    )
    checks["prompt_bodies_are_absent"] = all("prompt" not in row for row in rows)
    checks["raw_record_types_are_known"] = all(
        row.get("type")
        in {
            "run",
            "cache_seed",
            "placement",
            "paired_round",
            "round_summary",
        }
        for row in rows
    )

    expected_seeds = cache_seed_rows(config)
    actual_seeds = [
        {key: row.get(key) for key in ("type", "family_id", "replica_id", "prompt_sha256")}
        for row in rows
        if row.get("type") == "cache_seed"
    ]
    checks["initial_cache_placement_is_frozen"] = actual_seeds == expected_seeds
    seeds = actual_seeds if actual_seeds == expected_seeds else expected_seeds

    expected_shared = {item.request_id: item for item in shared_trace(config)}
    expected_uniform = {
        (round_index, item.request_id): item
        for round_index in range(config.uniform_rounds)
        for item in uniform_trace(config, round_index)
    }

    placement_rows = [row for row in rows if row.get("type") == "placement"]
    expected_shared_count = (
        config.shared_families * config.shared_requests_per_family * 2
    )
    expected_uniform_count = (
        config.uniform_rounds * config.uniform_requests_per_round * 2
    )
    checks["placement_row_count_matches_profile"] = len(placement_rows) == (
        expected_shared_count + expected_uniform_count
    )

    round_rows = [row for row in rows if row.get("type") == "paired_round"]
    checks["paired_round_order_alternates"] = len(round_rows) == config.uniform_rounds
    for round_index, row in enumerate(round_rows):
        expected_order = (
            [PREFIX_POLICY, HRW_POLICY]
            if round_index % 2 == 0
            else [HRW_POLICY, PREFIX_POLICY]
        )
        if (
            row.get("round_index") != round_index
            or row.get("policy_order") != expected_order
        ):
            checks["paired_round_order_alternates"] = False

    cache_seed_families = {str(seed["family_id"]) for seed in seeds}

    def replay(policy: str, trace_kind: str, round_index: int | None) -> bool:
        selected_rows = [
            row
            for row in placement_rows
            if row.get("policy") == policy
            and row.get("trace") == trace_kind
            and row.get("round_index") == round_index
        ]
        selected_rows.sort(key=lambda row: int(row.get("sequence", -1)))
        expected_count = (
            config.shared_families * config.shared_requests_per_family
            if trace_kind == "shared"
            else config.uniform_requests_per_round
        )
        if len(selected_rows) != expected_count:
            return False

        families_by_replica = {replica_id: set() for replica_id in replicas}
        keys_by_replica = {replica_id: set() for replica_id in replicas}
        for seed in seeds:
            replica_id = str(seed["replica_id"])
            family_id = str(seed["family_id"])
            families_by_replica[replica_id].add(family_id)
            keys_by_replica[replica_id].update(
                _chunk_keys(
                    _cache_seed_prompt(config, family_id),
                    config.chunk_chars,
                )
            )

        previous_dispatch_finished_ns: int | None = None
        for sequence, row in enumerate(selected_rows):
            request_id = row.get("request_id")
            item = (
                expected_shared.get(str(request_id))
                if trace_kind == "shared"
                else expected_uniform.get((int(round_index), str(request_id)))
            )
            if item is None or row.get("sequence") != sequence:
                return False
            prompt_keys = _chunk_keys(item.prompt, config.chunk_chars)
            if policy == HRW_POLICY:
                expected_replica = _expected_hrw(item.session_id, replicas)
                expected_reason = "session_affinity"
                expected_session_hash = item.session_sha256
            else:
                scored = [
                    (_overlap(prompt_keys, keys_by_replica[replica_id]), replica_id)
                    for replica_id in replicas
                ]
                best_overlap = max(score for score, _replica_id in scored)
                if best_overlap > 0:
                    warm = tuple(
                        replica_id
                        for score, replica_id in scored
                        if score == best_overlap
                    )
                    expected_replica = _expected_hrw(item.session_id, warm)
                    expected_reason = "prefix_match"
                else:
                    expected_replica = _expected_hrw(item.session_id, replicas)
                    expected_reason = "session_affinity"
                expected_session_hash = item.session_sha256
            expected_hit = item.family_id in families_by_replica[expected_replica]
            expected_prompt_chunks = _prompt_work_chunks(config, item.prompt)
            expected_cached_prefix_chunks = (
                config.prefix_chunks if expected_hit else 0
            )
            valid = (
                row.get("family_id") == item.family_id
                and row.get("prompt_sha256") == item.prompt_sha256
                and row.get("session_sha256") == expected_session_hash
                and row.get("selected_replica_id") == expected_replica
                and row.get("backend_replica_id") == expected_replica
                and row.get("replica_ordinal") == replicas.index(expected_replica)
                and row.get("reason") == expected_reason
                and row.get("cache_hit") is expected_hit
                and row.get("cached_prefix_chunks_before")
                == expected_cached_prefix_chunks
                and row.get("prompt_chunks") == expected_prompt_chunks
                and (
                    trace_kind != "shared"
                    or expected_prompt_chunks == config.prefix_chunks + 1
                )
                and row.get("pool_size") == config.replicas
                and row.get("eligible_size") == config.replicas
                and isinstance(row.get("placement_started_ns"), int)
                and isinstance(row.get("selected_at_ns"), int)
                and isinstance(row.get("placement_latency_ns"), int)
                and isinstance(row.get("dispatch_finished_ns"), int)
                and isinstance(row.get("dispatch_latency_ns"), int)
                and int(row["placement_started_ns"])
                <= int(row["selected_at_ns"])
                <= int(row["dispatch_finished_ns"])
                and row.get("placement_latency_ns")
                == int(row["selected_at_ns"]) - int(row["placement_started_ns"])
                and row.get("dispatch_latency_ns")
                == int(row["dispatch_finished_ns"])
                - int(row["placement_started_ns"])
                and int(row["dispatch_latency_ns"]) > 0
                and (
                    previous_dispatch_finished_ns is None
                    or previous_dispatch_finished_ns
                    <= int(row["placement_started_ns"])
                )
            )
            if not valid:
                return False
            previous_dispatch_finished_ns = int(row["dispatch_finished_ns"])
            families_by_replica[expected_replica].add(item.family_id)
            keys_by_replica[expected_replica].update(prompt_keys)
        return True

    replay_results = []
    for policy in POLICIES:
        replay_results.append(replay(policy, "shared", None))
        for round_index in range(config.uniform_rounds):
            replay_results.append(replay(policy, "uniform", round_index))
    checks["every_actual_selection_and_cache_hit_replays"] = all(replay_results)

    shared_rows = [
        row for row in placement_rows if row.get("trace") == "shared"
    ]
    trace_by_policy = {
        policy: [
            (
                row.get("sequence"),
                row.get("request_id"),
                row.get("family_id"),
                row.get("prompt_sha256"),
                row.get("session_sha256"),
            )
            for row in sorted(
                (
                    candidate
                    for candidate in shared_rows
                    if candidate.get("policy") == policy
                ),
                key=lambda candidate: int(candidate["sequence"]),
            )
        ]
        for policy in POLICIES
    }
    checks["shared_trace_is_identical_between_policies"] = (
        trace_by_policy[PREFIX_POLICY] == trace_by_policy[HRW_POLICY]
    )
    uniform_trace_pairs_match = True
    for round_index in range(config.uniform_rounds):
        by_policy = {
            policy: [
                (
                    row.get("sequence"),
                    row.get("request_id"),
                    row.get("family_id"),
                    row.get("prompt_sha256"),
                    row.get("session_sha256"),
                )
                for row in sorted(
                    (
                        candidate
                        for candidate in placement_rows
                        if candidate.get("trace") == "uniform"
                        and candidate.get("round_index") == round_index
                        and candidate.get("policy") == policy
                    ),
                    key=lambda candidate: int(candidate["sequence"]),
                )
            ]
            for policy in POLICIES
        }
        if by_policy[PREFIX_POLICY] != by_policy[HRW_POLICY]:
            uniform_trace_pairs_match = False
    checks["uniform_traces_and_sessions_match_between_policies"] = (
        uniform_trace_pairs_match
    )
    checks["all_prompt_hashes_are_sha256"] = all(
        _valid_sha256(row.get("prompt_sha256"))
        for row in placement_rows
    ) and all(_valid_sha256(seed.get("prompt_sha256")) for seed in seeds)
    expected_shared_families = {
        item.family_id for item in expected_shared.values()
    }
    checks["only_frozen_shared_families_are_seeded"] = (
        {str(seed["family_id"]) for seed in seeds}
        == cache_seed_families
        == expected_shared_families
    )

    summary_rows = [row for row in rows if row.get("type") == "round_summary"]
    expected_summary_keys = {
        ("shared", None, policy) for policy in POLICIES
    } | {
        ("uniform", round_index, policy)
        for round_index in range(config.uniform_rounds)
        for policy in POLICIES
    }
    checks["round_summaries_are_complete"] = (
        len(summary_rows) == len(expected_summary_keys)
    )
    expected_summary_order = [
        ("shared", None, policy) for policy in POLICIES
    ]
    for round_index in range(config.uniform_rounds):
        policy_order = (
            POLICIES
            if round_index % 2 == 0
            else tuple(reversed(POLICIES))
        )
        expected_summary_order.extend(
            ("uniform", round_index, policy) for policy in policy_order
        )
    checks["round_summaries_follow_emitted_arm_and_round_order"] = [
        (row.get("trace"), row.get("round_index"), row.get("policy"))
        for row in summary_rows
    ] == expected_summary_order
    summary_keys: set[tuple[object, object, object]] = set()
    summary_by_key: dict[tuple[object, object, object], dict[str, object]] = {}
    placement_slo_ns = int(config.placement_p99_limit_ms * 1_000_000)
    for row in summary_rows:
        summary_key = (
            row.get("trace"),
            row.get("round_index"),
            row.get("policy"),
        )
        matching = [
            placement
            for placement in placement_rows
            if (
                placement.get("trace"),
                placement.get("round_index"),
                placement.get("policy"),
            )
            == summary_key
        ]
        expected_order_index = (
            POLICIES.index(str(row.get("policy")))
            if row.get("trace") == "shared"
            and row.get("policy") in POLICIES
            else (
                (
                    POLICIES
                    if int(row.get("round_index", -1)) % 2 == 0
                    else tuple(reversed(POLICIES))
                ).index(str(row.get("policy")))
                if row.get("trace") == "uniform"
                and isinstance(row.get("round_index"), int)
                and row.get("policy") in POLICIES
                else -1
            )
        )
        placement_started = [
            int(placement["placement_started_ns"])
            for placement in matching
            if isinstance(placement.get("placement_started_ns"), int)
        ]
        placement_selected = [
            int(placement["selected_at_ns"])
            for placement in matching
            if isinstance(placement.get("selected_at_ns"), int)
        ]
        dispatch_finished = [
            int(placement["dispatch_finished_ns"])
            for placement in matching
            if isinstance(placement.get("dispatch_finished_ns"), int)
        ]
        valid = (
            summary_key not in summary_keys
            and row.get("policy") in POLICIES
            and isinstance(row.get("request_count"), int)
            and isinstance(row.get("wall_started_ns"), int)
            and isinstance(row.get("wall_finished_ns"), int)
            and isinstance(row.get("wall_ns"), int)
            and row.get("wall_role") == "audit_only"
            and isinstance(row.get("placement_sum_ns"), int)
            and isinstance(row.get("dispatch_sum_ns"), int)
            and row.get("placement_slo_ns") == placement_slo_ns
            and isinstance(row.get("placement_slo_success_count"), int)
            and row.get("wall_ns")
            == int(row["wall_finished_ns"]) - int(row["wall_started_ns"])
            and int(row["wall_ns"]) > 0
            and int(row["placement_sum_ns"]) > 0
            and int(row["dispatch_sum_ns"]) > 0
            and row.get("request_count") == len(matching)
            and row.get("placement_sum_ns")
            == sum(int(placement["placement_latency_ns"]) for placement in matching)
            and row.get("dispatch_sum_ns")
            == sum(int(placement["dispatch_latency_ns"]) for placement in matching)
            and row.get("placement_slo_success_count")
            == sum(
                int(placement["placement_latency_ns"]) < placement_slo_ns
                for placement in matching
            )
            and row.get("policy_order_index") == expected_order_index
            and len(placement_started) == len(matching)
            and len(placement_selected) == len(matching)
            and len(dispatch_finished) == len(matching)
            and (
                not matching
                or (
                    int(row["wall_started_ns"]) <= min(placement_started)
                    and int(row["wall_finished_ns"]) >= max(dispatch_finished)
                )
            )
        )
        summary_keys.add(summary_key)
        summary_by_key[summary_key] = row
        if not valid:
            checks["round_summaries_are_complete"] = False
    checks["round_summary_key_set_is_exact_and_unique"] = (
        summary_keys == expected_summary_keys
        and len(summary_keys) == len(summary_rows)
    )
    intervals_are_ordered = checks["round_summary_key_set_is_exact_and_unique"]
    interval_groups = [
        ("shared", None, POLICIES),
        *[
            (
                "uniform",
                round_index,
                (
                    POLICIES
                    if round_index % 2 == 0
                    else tuple(reversed(POLICIES))
                ),
            )
            for round_index in range(config.uniform_rounds)
        ],
    ]
    if intervals_are_ordered:
        for trace_kind, round_index, policy_order in interval_groups:
            first = summary_by_key[(trace_kind, round_index, policy_order[0])]
            second = summary_by_key[(trace_kind, round_index, policy_order[1])]
            if not (
                first.get("policy_order_index") == 0
                and second.get("policy_order_index") == 1
                and int(first["wall_finished_ns"])
                <= int(second["wall_started_ns"])
            ):
                intervals_are_ordered = False
    checks["paired_arm_intervals_follow_order_without_overlap"] = (
        intervals_are_ordered
    )
    global_intervals_are_ordered = (
        checks["round_summary_key_set_is_exact_and_unique"]
        and checks["round_summaries_follow_emitted_arm_and_round_order"]
    )
    previous_wall_finished_ns: int | None = None
    if global_intervals_are_ordered:
        for summary_key in expected_summary_order:
            summary = summary_by_key[summary_key]
            wall_started_ns = summary.get("wall_started_ns")
            wall_finished_ns = summary.get("wall_finished_ns")
            if (
                not isinstance(wall_started_ns, int)
                or not isinstance(wall_finished_ns, int)
                or (
                    previous_wall_finished_ns is not None
                    and previous_wall_finished_ns > wall_started_ns
                )
            ):
                global_intervals_are_ordered = False
                break
            previous_wall_finished_ns = wall_finished_ns
    checks["summary_intervals_follow_global_order_without_overlap"] = (
        global_intervals_are_ordered
    )
    if not all(checks.values()):
        errors.extend(name for name, passed in checks.items() if not passed)
    return checks, errors


def _resolve_manifest(path: Path) -> Path:
    return path / MANIFEST_NAME if path.is_dir() else path


def _verify_source_descriptor(source: object) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if not isinstance(source, dict):
        return {"source_descriptor_is_object": False}
    commit = source.get("commit")
    commit_is_sha = (
        isinstance(commit, str)
        and len(commit) == 40
        and set(commit) <= _HEX_SHA256
    )
    checks["source_commit_is_exact_sha"] = commit_is_sha
    checks["source_commit_is_ancestor_of_verifier_head"] = (
        commit_is_sha and _git_is_ancestor(str(commit))
    )
    expected_commit = source.get("expected_commit")
    checks["workflow_expected_commit_matches_when_present"] = (
        (expected_commit is None or expected_commit == commit)
        and source.get("expected_commit_matches") is True
    )
    # This is a fact captured before the run, not current verifier state:
    # committing the evidence naturally changes HEAD/cleanliness afterward.
    checks["source_was_clean_at_run_start"] = (
        source.get("git_clean_at_start") is True
    )
    files = source.get("files")
    expected_paths = list(_SOURCE_PATHS)
    checks["source_file_descriptors_are_exact"] = (
        isinstance(files, list)
        and [item.get("path") for item in files if isinstance(item, dict)]
        == expected_paths
    )
    if isinstance(files, list):
        for relative_path, descriptor in zip(
            expected_paths, files, strict=False
        ):
            checks[f"source_file_hash:{relative_path}"] = (
                isinstance(descriptor, dict)
                and descriptor.get("path") == relative_path
                and _valid_sha256(descriptor.get("sha256"))
                and descriptor.get("sha256")
                == _git_blob_sha256(str(commit), relative_path)
                == _sha256_file(_REPO_ROOT / relative_path)
            )
    return checks


def _verify_environment_descriptor(environment: object) -> dict[str, bool]:
    if not isinstance(environment, dict):
        return {"environment_descriptor_is_object": False}
    github = environment.get("github")
    affinity = environment.get("sched_affinity")
    return {
        "environment_platform_is_recorded": (
            isinstance(environment.get("platform"), str)
            and bool(environment["platform"])
        ),
        "environment_python_is_recorded": (
            isinstance(environment.get("python"), str)
            and bool(environment["python"])
        ),
        "environment_cpu_count_is_recorded": (
            isinstance(environment.get("cpu_count"), int)
            and int(environment["cpu_count"]) > 0
        ),
        "environment_affinity_is_recorded": (
            affinity is None
            or (
                isinstance(affinity, list)
                and bool(affinity)
                and all(isinstance(cpu, int) and cpu >= 0 for cpu in affinity)
            )
        ),
        "environment_github_fields_are_exact": (
            isinstance(github, dict)
            and set(github)
            == {
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_EVENT_NAME",
                "GITHUB_SHA",
            }
            and all(
                value is None or isinstance(value, str)
                for value in github.values()
            )
        ),
    }


def verify_artifact(path: Path) -> dict[str, object]:
    manifest_path = _resolve_manifest(path).resolve()
    errors: list[str] = []
    checks: dict[str, bool] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not an object")
        config_raw = manifest.get("config")
        if not isinstance(config_raw, dict):
            raise ValueError("manifest config is missing")
        profile = config_raw.get("profile")
        seed = config_raw.get("seed")
        if not isinstance(profile, str) or not isinstance(seed, int):
            raise ValueError("manifest profile/seed is invalid")
        config = profile_config(profile, seed=seed)
        checks["config_matches_frozen_profile"] = config_raw == asdict(config)
        checks["schema_version_matches"] = (
            manifest.get("schema_version") == SCHEMA_VERSION
        )
        raw_descriptor = manifest.get("raw")
        if not isinstance(raw_descriptor, dict):
            raise ValueError("manifest raw descriptor is missing")
        raw_name = raw_descriptor.get("path")
        if (
            not isinstance(raw_name, str)
            or Path(raw_name).name != raw_name
            or raw_name != RAW_NAME
        ):
            raise ValueError("raw path must be the frozen adjacent filename")
        raw_path = (manifest_path.parent / raw_name).resolve()
        if raw_path.parent != manifest_path.parent:
            raise ValueError("raw path escapes the artifact directory")
        checks["raw_sha256_matches"] = (
            _valid_sha256(raw_descriptor.get("sha256"))
            and _sha256_file(raw_path) == raw_descriptor.get("sha256")
        )
        rows = _read_jsonl(raw_path)
        checks["raw_row_count_matches"] = (
            raw_descriptor.get("rows") == len(rows)
        )
        run_rows = [row for row in rows if row.get("type") == "run"]
        checks["one_run_record_exists"] = len(run_rows) == 1
        if len(run_rows) != 1:
            raise ValueError("raw evidence must contain exactly one run record")
        run = run_rows[0]
        checks["raw_run_metadata_matches_manifest"] = (
            run.get("schema_version") == SCHEMA_VERSION
            and run.get("config") == asdict(config)
            and run.get("source") == manifest.get("source")
            and run.get("source_end") == manifest.get("source_end")
            and run.get("environment") == manifest.get("environment")
        )
        checks.update(_verify_source_descriptor(manifest.get("source")))
        checks["source_commit_and_files_did_not_drift_during_run"] = (
            _source_end_matches_start(
                manifest.get("source"),
                manifest.get("source_end"),
            )
        )
        checks.update(
            _verify_environment_descriptor(manifest.get("environment"))
        )
        checks["formal_expected_commit_and_github_context_are_complete"] = (
            _formal_context_complete(
                config,
                manifest.get("source"),
                manifest.get("environment"),
            )
        )
        replay_checks, replay_errors = _replay_rows(rows, config)
        checks.update(replay_checks)
        errors.extend(replay_errors)
        metrics = derive_metrics(rows, config)
        gates = threshold_checks(metrics, config)
        checks["manifest_metrics_recomputed_exactly"] = (
            manifest.get("metrics") == metrics
        )
        checks["manifest_gate_checks_recomputed_exactly"] = (
            manifest.get("gate_checks") == gates
        )
        recorded_source = manifest.get("source")
        source_gate = (
            isinstance(recorded_source, dict)
            and recorded_source.get("git_clean_at_start") is True
            and recorded_source.get("expected_commit_matches") is True
            and _source_end_matches_start(
                recorded_source,
                manifest.get("source_end"),
            )
            and _formal_context_complete(
                config,
                recorded_source,
                manifest.get("environment"),
            )
        )
        checks["manifest_passed_recomputed_exactly"] = (
            manifest.get("passed")
            is (source_gate and all(gates.values()))
        )
        checks.update(gates)
        passed = all(checks.values())
        if not passed:
            errors.extend(name for name, value in checks.items() if not value)
        return {
            "passed": passed,
            "checks": checks,
            "metrics": metrics,
            "errors": sorted(set(errors)),
            "manifest": str(manifest_path),
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        errors.append(f"{type(error).__name__}: {error}")
        return {
            "passed": False,
            "checks": checks,
            "metrics": None,
            "errors": errors,
            "manifest": str(manifest_path),
        }


def run_benchmark(
    output_dir: Path,
    *,
    profile: str = "formal",
    seed: int = DEFAULT_SEED,
) -> Path:
    config = profile_config(profile, seed=seed)
    # Capture provenance before creating output: a results directory inside the
    # repository must not turn a clean measurement source into its own dirt.
    source = source_descriptor()
    environment = environment_descriptor()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    manifest_path = output_dir / MANIFEST_NAME
    rows = asyncio.run(_collect_raw_rows(config, source, environment))
    source_end = source_end_descriptor()
    run_row = next(row for row in rows if row.get("type") == "run")
    run_row["source_end"] = source_end
    _write_jsonl(raw_path, rows)
    metrics = derive_metrics(rows, config)
    gates = threshold_checks(metrics, config)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "source": source,
        "source_end": source_end,
        "environment": environment,
        "raw": {
            "path": RAW_NAME,
            "sha256": _sha256_file(raw_path),
            "rows": len(rows),
        },
        "metrics": metrics,
        "gate_checks": gates,
        "passed": (
            source.get("git_clean_at_start") is True
            and source.get("expected_commit_matches") is True
            and _source_end_matches_start(source, source_end)
            and _formal_context_complete(config, source, environment)
            and all(gates.values())
        ),
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--profile", choices=tuple(_PROFILE_COUNTS), default="formal")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_artifact is not None:
        result = verify_artifact(args.verify_artifact)
    else:
        manifest = run_benchmark(
            args.output_dir,
            profile=args.profile,
            seed=args.seed,
        )
        result = verify_artifact(manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
