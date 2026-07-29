#!/usr/bin/env python3
"""G5 F2b: replayable KV-event freshness and stream-chaos CPU gate.

The formal profile reuses F1a's exact 200-replica, seed-175 permutation and
ten disjoint batches of twenty.  It compresses the one-minute epoch to one
second, so the complete lifecycle is exercised at sixty times F1a's replica
churn intensity without repeating Kind, Docker, or GPU work.  F1a remains the
binding evidence for the real ten-minute Kubernetes envelope; this gate adds
only the distinct KV-event transport, stale fallback, replay, and routing
claims.

Raw JSONL is authoritative.  ``--verify-artifact`` rehashes it, reconstructs
the frozen schedule and causal joins, and recomputes every gate instead of
trusting the manifest's metrics or pass bit.
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
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from bench.fleet_churn_bench import FORMAL_CONFIG, churn_plan
from kairyu.engine.backend import (
    CacheHint,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
)
from kairyu.orchestration.kv_index import (
    KvEventIndex,
    ZmqKvEventPublisher,
    ZmqKvEventSubscriber,
)
from kairyu.orchestration.kv_routing import KvRoutingIndex
from kairyu.orchestration.prefix_index import (
    PrefixIndex,
    prefix_root_fingerprint,
)
from kairyu.orchestration.replica import ReplicaPool

SCHEMA_VERSION = "kairyu.f2b.kv-event-chaos.v1"
RAW_NAME = "kv-event-f2b-raw.jsonl"
MANIFEST_NAME = "kv-event-f2b-manifest.json"
GATE = "G5-F2b"
F1A_RUN_ID = "30374404150"
F2A_RUN_ID = "30411111758"
F2A_COMMIT = "c067cb8a6d1aa661c82a56a007fc325011426459"
F2A_RAW_SHA256 = "82bb2a2ff420dbd4e244685ce2a83f38379028604be3c1077e85daf5b31cd0f3"
F2A_MANIFEST_SHA256 = "fb2858ed213be3f6b98463d30623e937d74403a7a7dd97356311439a7f7585f4"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_F2A_RETAINED_DIR = _REPO_ROOT / "bench/results/f2a-prefix-routing-500-2026-07-28"
_SOURCE_PATHS = (
    "kairyu/__init__.py",
    "kairyu/audit_io.py",
    "kairyu/outputs.py",
    "kairyu/sampling_params.py",
    "kairyu/telemetry.py",
    "kairyu/engine/__init__.py",
    "kairyu/orchestration/__init__.py",
    "kairyu/orchestration/kv_index.py",
    "kairyu/orchestration/kv_routing.py",
    "kairyu/orchestration/features.py",
    "kairyu/orchestration/prefix_index.py",
    "kairyu/orchestration/replica.py",
    "kairyu/orchestration/router.py",
    "kairyu/engine/backend.py",
    "bench/fleet_churn_bench.py",
    "bench/kv_event_f2b_bench.py",
    ".github/workflows/f2b-kv-event-chaos.yml",
    "pyproject.toml",
    "uv.lock",
)
_HEX_SHA256 = frozenset("0123456789abcdef")
_NS_PER_SECOND = 1_000_000_000
_WORKFLOW_NAME = "F2b KV-event freshness and chaos"
_WORKFLOW_PATH = ".github/workflows/f2b-kv-event-chaos.yml"
_GITHUB_CONTEXT_NAMES = (
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_EVENT_NAME",
    "GITHUB_REPOSITORY",
    "GITHUB_WORKFLOW",
    "GITHUB_WORKFLOW_REF",
    "GITHUB_REF",
    "GITHUB_SHA",
    "F2B_GITHUB_HEAD_SHA",
)


@dataclass(frozen=True)
class WorkloadConfig:
    profile: str
    seed: int
    replicas: int
    churn_batch_size: int
    churn_epochs: int
    epoch_seconds: float
    source_epoch_seconds: float
    source_churn_window_seconds: float
    warmup_seconds: float
    cooldown_seconds: float
    route_rate: float
    heartbeat_seconds: float
    stale_after_seconds: float
    freshness_limit_seconds: float
    kill_offset_seconds: float
    restore_offset_seconds: float
    replay_buffer_steps: int
    route_timeout_seconds: float

    @property
    def measurement_seconds(self) -> float:
        return self.churn_epochs * self.epoch_seconds

    @property
    def churn_acceleration(self) -> float:
        return self.source_epoch_seconds / self.epoch_seconds

    @property
    def churn_window_seconds(self) -> float:
        return self.source_churn_window_seconds / self.churn_acceleration

    @property
    def expected_measurement_routes(self) -> int:
        return math.floor(self.measurement_seconds * self.route_rate)

    def validate(self) -> None:
        if self.profile not in {"formal", "smoke"}:
            raise ValueError("profile must be formal or smoke")
        integer_values = (
            self.seed,
            self.replicas,
            self.churn_batch_size,
            self.churn_epochs,
            self.replay_buffer_steps,
        )
        if any(type(value) is not int for value in integer_values):
            raise TypeError("integer configuration fields must be integers")
        if self.replicas != self.churn_batch_size * self.churn_epochs:
            raise ValueError("replicas must equal churn_batch_size * churn_epochs")
        positive_values = (
            self.epoch_seconds,
            self.source_epoch_seconds,
            self.source_churn_window_seconds,
            self.route_rate,
            self.heartbeat_seconds,
            self.stale_after_seconds,
            self.freshness_limit_seconds,
            self.restore_offset_seconds,
            self.replay_buffer_steps,
            self.route_timeout_seconds,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("durations, rates, and capacities must be positive")
        if self.warmup_seconds < 0 or self.cooldown_seconds < 0:
            raise ValueError("warmup/cooldown must be non-negative")
        if not 0 < self.stale_after_seconds < self.freshness_limit_seconds:
            raise ValueError("stale lease must leave strict freshness headroom")
        if self.heartbeat_seconds >= self.stale_after_seconds:
            raise ValueError("heartbeat interval must be below the stale lease")
        if not (
            0 < self.kill_offset_seconds < self.restore_offset_seconds < self.measurement_seconds
        ):
            raise ValueError("kill/restore offsets must lie inside measurement")


_FORMAL = WorkloadConfig(
    profile="formal",
    seed=FORMAL_CONFIG.seed,
    replicas=FORMAL_CONFIG.replica_count,
    churn_batch_size=FORMAL_CONFIG.churn_batch_size,
    churn_epochs=FORMAL_CONFIG.churn_epochs,
    epoch_seconds=1.0,
    source_epoch_seconds=FORMAL_CONFIG.epoch_seconds,
    source_churn_window_seconds=FORMAL_CONFIG.churn_window_seconds,
    warmup_seconds=1.0,
    cooldown_seconds=1.0,
    route_rate=50.0,
    heartbeat_seconds=0.05,
    stale_after_seconds=0.25,
    freshness_limit_seconds=0.5,
    kill_offset_seconds=3.25,
    restore_offset_seconds=6.25,
    replay_buffer_steps=4_096,
    route_timeout_seconds=2.0,
)

_SMOKE = WorkloadConfig(
    profile="smoke",
    seed=FORMAL_CONFIG.seed,
    replicas=8,
    churn_batch_size=2,
    churn_epochs=4,
    epoch_seconds=0.25,
    source_epoch_seconds=FORMAL_CONFIG.epoch_seconds,
    source_churn_window_seconds=FORMAL_CONFIG.churn_window_seconds,
    warmup_seconds=0.15,
    cooldown_seconds=0.35,
    route_rate=20.0,
    heartbeat_seconds=0.02,
    stale_after_seconds=0.10,
    freshness_limit_seconds=0.5,
    kill_offset_seconds=0.30,
    restore_offset_seconds=0.72,
    replay_buffer_steps=256,
    route_timeout_seconds=2.0,
)


def profile_config(profile: str) -> WorkloadConfig:
    if profile == "formal":
        config = _FORMAL
    elif profile == "smoke":
        config = _SMOKE
    else:
        raise ValueError(f"unknown profile {profile!r}")
    config.validate()
    return config


def compressed_churn_plan(config: WorkloadConfig) -> list[dict[str, object]]:
    """Return the frozen ordinal batches with only wall pacing compressed."""

    config.validate()
    if config.profile == "formal":
        source = churn_plan(FORMAL_CONFIG)
        if (
            config.replicas != FORMAL_CONFIG.replica_count
            or config.churn_batch_size != FORMAL_CONFIG.churn_batch_size
            or config.churn_epochs != FORMAL_CONFIG.churn_epochs
            or config.seed != FORMAL_CONFIG.seed
        ):
            raise ValueError("formal F2b identity schedule must equal F1a")
        ordinal_batches = [list(row["ordinals"]) for row in source]
    else:
        ordinals = list(range(config.replicas))
        random.Random(config.seed).shuffle(ordinals)
        ordinal_batches = [
            ordinals[epoch * config.churn_batch_size : (epoch + 1) * config.churn_batch_size]
            for epoch in range(config.churn_epochs)
        ]
    return [
        {
            "epoch": epoch,
            "source_deadline_offset_s": epoch * config.source_epoch_seconds,
            "deadline_offset_s": epoch * config.epoch_seconds,
            "ordinals": batch,
        }
        for epoch, batch in enumerate(ordinal_batches)
    ]


def _replica_id(ordinal: int) -> str:
    return f"replica-{ordinal:03d}"


def _representative_epoch(config: WorkloadConfig) -> int:
    return min(
        config.churn_epochs - 1,
        max(0, math.ceil(config.kill_offset_seconds / config.epoch_seconds)),
    )


def representative_replica_id(config: WorkloadConfig) -> str:
    plan = compressed_churn_plan(config)
    ordinal = plan[_representative_epoch(config)]["ordinals"][0]
    if type(ordinal) is not int:
        raise RuntimeError("frozen churn ordinal must be an integer")
    return _replica_id(ordinal)


def topology_descriptor(config: WorkloadConfig) -> dict[str, object]:
    return {
        "logical_replicas": config.replicas,
        "in_process_feeds": config.replicas - 1,
        "physical_zmq_feeds": 1,
        "representative_replica_id": representative_replica_id(config),
        "wire_scope": (
            "one representative production ZMQ source gates exact routing for "
            "the same 200-entry logical pool; this is not 200 physical streams"
        ),
        "fault": "same-subscriber-object live socket close/reconnect",
    }


def _epoch_prompt(epoch: int) -> str:
    marker = chr(ord("A") + epoch)
    return marker * 512 + f":f2b-epoch-{epoch:02d}"


def _epoch_block_hashes(epoch: int) -> tuple[str, str]:
    return (f"f2b-block-{epoch:02d}-0", f"f2b-block-{epoch:02d}-1")


def _epoch_targets(
    config: WorkloadConfig,
    epoch: int,
) -> tuple[str, str]:
    plan = compressed_churn_plan(config)
    exact_ordinal = plan[epoch]["ordinals"][0]
    other_epoch = (epoch + 1) % config.churn_epochs
    approximate_ordinal = plan[other_epoch]["ordinals"][0]
    if type(exact_ordinal) is not int or type(approximate_ordinal) is not int:
        raise RuntimeError("frozen churn ordinals must be integers")
    return _replica_id(exact_ordinal), _replica_id(approximate_ordinal)


def _session_for_approximate(
    epoch: int,
    exact_replica: str,
    approximate_replica: str,
) -> str:
    for attempt in range(10_000):
        session_id = f"f2b-session-{epoch:02d}-{attempt:04d}"
        exact_score = hashlib.sha256(f"{session_id}:{exact_replica}".encode()).digest()
        approximate_score = hashlib.sha256(f"{session_id}:{approximate_replica}".encode()).digest()
        if approximate_score > exact_score:
            return session_id
    raise RuntimeError("could not freeze an approximate HRW winner")


def reused_evidence_descriptor() -> dict[str, object]:
    return {
        "f1a": {
            "actions_run_id": F1A_RUN_ID,
            "evidence_status": "external_context_only",
            "scope": (
                "Kind/NodePort/discovery/provenance/host-capacity and the real "
                "200-replica 10x20 one-minute churn envelope only"
            ),
            "claims": {
                "replicas": 200,
                "churn_batch_size": 20,
                "churn_epochs": 10,
                "epoch_seconds": 60.0,
                "request_2xx": 33_000,
                "request_total": 33_000,
                "placement_p99_ms": 5.221,
                "max_withdrawal_seconds": 1.117,
            },
            "artifact_digests_locally_verified": False,
            "binding_to_f2b_kv_claims": False,
            "kv_measurements_reused": False,
        },
        "f2a": {
            "actions_run_id": F2A_RUN_ID,
            "commit": F2A_COMMIT,
            "raw_sha256": F2A_RAW_SHA256,
            "manifest_sha256": F2A_MANIFEST_SHA256,
            "scope": (
                "production ReplicaPool.generate and approximate PrefixIndex routing precedent only"
            ),
            "claims": {
                "logical_replicas": 500,
                "worst_trace_placement_p99_ms": 0.145979,
            },
            "fallback_measurements_reused": False,
        },
    }


def _reused_f2a_digests_match() -> bool:
    manifest = _F2A_RETAINED_DIR / "prefix-routing-f2a-manifest.json"
    raw = _F2A_RETAINED_DIR / "prefix-routing-f2a-raw.jsonl"
    return (
        manifest.is_file()
        and raw.is_file()
        and _sha256_file(manifest) == F2A_MANIFEST_SHA256
        and _sha256_file(raw) == F2A_RAW_SHA256
    )


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


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


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
    lines = _git_output("status", "--porcelain", "--untracked-files=all").splitlines()
    relevant = [
        line
        for line in lines
        if not (line.startswith("?? .claude/worktrees/") or line == "?? .claude/worktrees")
    ]
    return not relevant


def _frozen_source_snapshot() -> dict[str, object]:
    commit = _git_output("rev-parse", "HEAD")
    expected = os.environ.get("F2B_EXPECTED_COMMIT")
    return {
        "commit": commit,
        "expected_commit": expected,
        "expected_commit_matches": expected is None or expected == commit,
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
    return _frozen_source_snapshot()


def _current_github_context() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _GITHUB_CONTEXT_NAMES}


def environment_descriptor() -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "sched_affinity": affinity,
        "github": _current_github_context(),
    }


def _source_end_matches_start(source: object, source_end: object) -> bool:
    if not isinstance(source, dict) or not isinstance(source_end, dict):
        return False
    return source_end == {
        key: source.get(key)
        for key in (
            "commit",
            "expected_commit",
            "expected_commit_matches",
            "files",
        )
    }


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _HEX_SHA256


def _verify_source_descriptor(source: object) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if not isinstance(source, dict):
        return {"source_descriptor_shape": False}
    commit = source.get("commit")
    files = source.get("files")
    checks["source_commit_is_recorded"] = (
        isinstance(commit, str) and len(commit) == 40 and set(commit.lower()) <= _HEX_SHA256
    )
    checks["source_commit_is_ancestor"] = checks["source_commit_is_recorded"] and _git_is_ancestor(
        commit
    )
    checks["source_files_are_exact"] = isinstance(files, list) and [
        item.get("path") for item in files if isinstance(item, dict)
    ] == list(_SOURCE_PATHS)
    if checks["source_commit_is_recorded"] and isinstance(files, list):
        try:
            checks["source_file_hashes_match_commit"] = all(
                isinstance(item, dict)
                and item.get("path") in _SOURCE_PATHS
                and _valid_sha256(item.get("sha256"))
                and _git_blob_sha256(commit, str(item["path"])) == item["sha256"]
                for item in files
            )
        except (OSError, subprocess.CalledProcessError):
            checks["source_file_hashes_match_commit"] = False
    else:
        checks["source_file_hashes_match_commit"] = False
    checks["source_clean_at_start"] = source.get("git_clean_at_start") is True
    checks["source_expected_commit_matches"] = source.get("expected_commit_matches") is True
    return checks


def _formal_context_complete(
    config: WorkloadConfig,
    source: object,
    environment: object,
) -> bool:
    if config.profile != "formal":
        return True
    if not isinstance(source, dict) or not isinstance(environment, dict):
        return False
    github = environment.get("github")
    if not isinstance(github, dict):
        return False
    return (
        isinstance(source.get("expected_commit"), str)
        and source.get("expected_commit") == source.get("commit")
        and source.get("expected_commit_matches") is True
        and all(
            isinstance(github.get(name), str)
            and str(github[name]).isascii()
            and str(github[name]).isdecimal()
            and int(str(github[name])) > 0
            for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
        )
        and github.get("GITHUB_EVENT_NAME") in {"pull_request", "workflow_dispatch"}
        and isinstance(github.get("GITHUB_SHA"), str)
        and len(str(github["GITHUB_SHA"])) == 40
        and set(str(github["GITHUB_SHA"]).lower()) <= _HEX_SHA256
        and github.get("F2B_GITHUB_HEAD_SHA") == source.get("commit")
        and github.get("GITHUB_REPOSITORY") == "ytworks/kairyu"
        and github.get("GITHUB_WORKFLOW") == _WORKFLOW_NAME
        and isinstance(github.get("GITHUB_WORKFLOW_REF"), str)
        and f"/{_WORKFLOW_PATH}@" in str(github["GITHUB_WORKFLOW_REF"])
        and isinstance(github.get("GITHUB_REF"), str)
        and bool(github["GITHUB_REF"])
    )


def _github_context_matches_current_runner(environment: object) -> bool:
    """Bind freshly measured evidence to this exact Actions runner context."""

    return isinstance(environment, dict) and environment.get("github") == _current_github_context()


def _positive_decimal(value: object) -> bool:
    return isinstance(value, str) and value.isascii() and value.isdecimal() and int(value) > 0


def _verify_actions_run_provenance(
    provenance: object,
    source: object,
    environment: object,
) -> dict[str, bool]:
    """Bind retained evidence to a trusted GitHub Actions run API response."""

    github = environment.get("github") if isinstance(environment, dict) else None
    repository = provenance.get("repository") if isinstance(provenance, dict) else None
    run_id = provenance.get("id") if isinstance(provenance, dict) else None
    run_attempt = provenance.get("run_attempt") if isinstance(provenance, dict) else None
    recorded_run_id = github.get("GITHUB_RUN_ID") if isinstance(github, dict) else None
    recorded_attempt = github.get("GITHUB_RUN_ATTEMPT") if isinstance(github, dict) else None
    recorded_repository = github.get("GITHUB_REPOSITORY") if isinstance(github, dict) else None
    workflow_ref = github.get("GITHUB_WORKFLOW_REF") if isinstance(github, dict) else None
    workflow_path = provenance.get("path") if isinstance(provenance, dict) else None
    source_commit = source.get("commit") if isinstance(source, dict) else None
    repository_full_name = repository.get("full_name") if isinstance(repository, dict) else None
    return {
        "actions_run_id_matches_artifact": (
            type(run_id) is int
            and _positive_decimal(recorded_run_id)
            and run_id == int(recorded_run_id)
        ),
        "actions_run_head_sha_matches_artifact_source": (
            isinstance(source_commit, str)
            and provenance.get("head_sha") == source_commit
            and isinstance(github, dict)
            and github.get("F2B_GITHUB_HEAD_SHA") == source_commit
        )
        if isinstance(provenance, dict)
        else False,
        "actions_run_event_matches_artifact": (
            isinstance(provenance, dict)
            and isinstance(github, dict)
            and provenance.get("event") == github.get("GITHUB_EVENT_NAME")
        ),
        "actions_run_name_matches_artifact": (
            isinstance(provenance, dict)
            and isinstance(github, dict)
            and provenance.get("name") == github.get("GITHUB_WORKFLOW")
            and provenance.get("name") == _WORKFLOW_NAME
        ),
        "actions_run_path_matches_artifact": (
            workflow_path == _WORKFLOW_PATH
            and isinstance(workflow_ref, str)
            and isinstance(recorded_repository, str)
            and workflow_ref.startswith(f"{recorded_repository}/{_WORKFLOW_PATH}@")
        ),
        "actions_run_repository_matches_artifact": (
            isinstance(recorded_repository, str) and repository_full_name == recorded_repository
        ),
        "actions_run_attempt_matches_artifact": (
            type(run_attempt) is int
            and _positive_decimal(recorded_attempt)
            and run_attempt == int(recorded_attempt)
        ),
        "actions_run_is_completed": (
            isinstance(provenance, dict) and provenance.get("status") == "completed"
        ),
        "actions_run_conclusion_is_success": (
            isinstance(provenance, dict) and provenance.get("conclusion") == "success"
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _resolve_manifest(path: Path) -> Path:
    return path / MANIFEST_NAME if path.is_dir() else path


def artifact_source_matches_current(
    path: Path,
    *,
    required_profile: str | None = None,
) -> bool:
    """Whether retained evidence was produced by byte-identical gate sources."""

    try:
        manifest = json.loads(_resolve_manifest(path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False
        config = manifest.get("config")
        if required_profile is not None and (
            not isinstance(config, dict) or config.get("profile") != required_profile
        ):
            return False
        source = manifest.get("source")
        files = source.get("files") if isinstance(source, dict) else None
        if not isinstance(files, list):
            return False
        if [item.get("path") for item in files if isinstance(item, dict)] != list(_SOURCE_PATHS):
            return False
        return all(
            isinstance(item, dict)
            and _valid_sha256(item.get("sha256"))
            and (_REPO_ROOT / str(item["path"])).is_file()
            and _sha256_file(_REPO_ROOT / str(item["path"])) == item["sha256"]
            for item in files
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _rows_by_type(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        kind = row.get("type")
        if isinstance(kind, str):
            grouped.setdefault(kind, []).append(row)
    return grouped


def _single_row(
    grouped: Mapping[str, list[dict[str, object]]],
    kind: str,
) -> dict[str, object]:
    candidates = grouped.get(kind, [])
    return candidates[0] if len(candidates) == 1 else {}


def _snapshot_by_phase(
    grouped: Mapping[str, list[dict[str, object]]],
    phase: str,
) -> dict[str, object]:
    candidates = [row for row in grouped.get("state_snapshot", []) if row.get("phase") == phase]
    return candidates[0] if len(candidates) == 1 else {}


def _kill_by_action(
    grouped: Mapping[str, list[dict[str, object]]],
) -> dict[str, dict[str, object]]:
    return {
        str(row.get("action")): row
        for row in grouped.get("kill", [])
        if isinstance(row.get("action"), str)
    }


def _binding_recovery(
    grouped: Mapping[str, list[dict[str, object]]],
) -> dict[str, object]:
    resume = _kill_by_action(grouped).get("resume", {})
    resume_ns = resume.get("observed_ns")
    candidates = [
        row
        for row in grouped.get("replay", [])
        if type(resume_ns) is int
        and type(row.get("requested_ns")) is int
        and int(row["requested_ns"]) >= int(resume_ns)
    ]
    return candidates[0] if len(candidates) == 1 else {}


def _trusted_wire_refreshes(
    grouped: Mapping[str, list[dict[str, object]]],
) -> list[int]:
    refreshes = [
        int(row["applied_ns"])
        for row in grouped.get("heartbeat", [])
        if row.get("result") in {"refreshed", "recovered"} and type(row.get("applied_ns")) is int
    ]
    refreshes.extend(
        int(row["applied_ns"])
        for row in grouped.get("event_apply", [])
        if row.get("result") == "applied"
        and row.get("replayed") is False
        and type(row.get("applied_ns")) is int
    )
    return sorted(set(refreshes))


def _latest_at_or_before(values: Sequence[int], timestamp: int) -> int | None:
    latest = None
    for value in values:
        if value > timestamp:
            break
        latest = value
    return latest


def _route_facts(
    rows: list[dict[str, object]],
    config: WorkloadConfig,
) -> list[dict[str, object]]:
    grouped = _rows_by_type(rows)
    kill = _kill_by_action(grouped)
    pause = kill.get("pause", {})
    resume = kill.get("resume", {})
    recovery = _binding_recovery(grouped)
    pause_ns = pause.get("observed_ns")
    resume_ns = resume.get("observed_ns")
    recovery_ns = recovery.get("completed_ns")
    representative_id = representative_replica_id(config)
    refreshes = _trusted_wire_refreshes(grouped)
    pre_pause_refresh = (
        _latest_at_or_before(refreshes, int(pause_ns)) if type(pause_ns) is int else None
    )
    stale_deadline_ns = (
        pre_pause_refresh + int(config.stale_after_seconds * _NS_PER_SECOND)
        if pre_pause_refresh is not None
        else None
    )
    invalidations = [
        int(row["observed_ns"])
        for row in grouped.get("membership", [])
        if row.get("replica_id") == representative_id
        and row.get("reason") in {"manual_drain", "replica_removed"}
        and type(row.get("observed_ns")) is int
        and type(pause_ns) is int
        and int(row["observed_ns"]) >= int(pause_ns)
    ]
    invalidations.extend(
        int(row["received_ns"])
        for row in grouped.get("event_receive", [])
        if row.get("result") == "gap" and type(row.get("received_ns")) is int
    )
    first_invalidation_ns = min(invalidations, default=None)
    stale_boundary_candidates = [
        value for value in (stale_deadline_ns, first_invalidation_ns) if value is not None
    ]
    stale_boundary_ns = min(stale_boundary_candidates) if stale_boundary_candidates else None
    logical_refreshes = sorted(
        int(row["finished_ns"])
        for row in grouped.get("logical_heartbeat", [])
        if row.get("all_refreshed") is True and type(row.get("finished_ns")) is int
    )
    facts = []
    for row in sorted(
        grouped.get("route", []),
        key=lambda item: int(item.get("sequence", -1)),
    ):
        selected_ns = row.get("selected_ns")
        sequence = row.get("sequence")
        if type(selected_ns) is not int or type(sequence) is not int:
            facts.append({"row": row, "phase": "invalid"})
            continue
        if type(recovery_ns) is int and selected_ns >= int(recovery_ns):
            phase = "restored"
        elif (
            type(pause_ns) is int
            and selected_ns >= int(pause_ns)
            and stale_boundary_ns is not None
            and selected_ns >= stale_boundary_ns
        ):
            phase = "stale"
        else:
            phase = "fresh"
        wire_refresh = _latest_at_or_before(refreshes, selected_ns)
        logical_refresh = _latest_at_or_before(logical_refreshes, selected_ns)
        epoch = min(
            config.churn_epochs - 1,
            int((sequence / config.route_rate) // config.epoch_seconds),
        )
        exact_target, approximate_target = _epoch_targets(config, epoch)
        facts.append(
            {
                "row": row,
                "phase": phase,
                "epoch": epoch,
                "wire_truth_age_ns": (
                    selected_ns - wire_refresh if wire_refresh is not None else None
                ),
                "logical_truth_age_ns": (
                    selected_ns - logical_refresh if logical_refresh is not None else None
                ),
                "expected_exact_replica": exact_target,
                "expected_approximate_replica": approximate_target,
                "stale_boundary_ns": stale_boundary_ns,
                "pause_ns": pause_ns,
                "resume_ns": resume_ns,
                "recovery_ns": recovery_ns,
            }
        )
    return facts


def _rebuild_authoritative_state(
    rows: list[dict[str, object]],
    config: WorkloadConfig,
) -> dict[str, list[str]] | None:
    state = {_replica_id(ordinal): set() for ordinal in range(config.replicas)}
    logical = sorted(
        (row for row in rows if row.get("type") == "logical_apply"),
        key=lambda row: int(row.get("applied_ns", -1)),
    )
    wire = sorted(
        (row for row in rows if row.get("type") == "event_emit"),
        key=lambda row: int(row.get("emitted_ns", -1)),
    )
    try:
        for row in logical:
            replica_id = row.get("replica_id")
            event = row.get("event")
            if replica_id not in state or not isinstance(event, dict):
                return None
            _apply_hash_event(state[str(replica_id)], event)
        representative_id = representative_replica_id(config)
        for row in wire:
            event = row.get("event")
            if row.get("replica_id") != representative_id or not isinstance(event, dict):
                return None
            _apply_hash_event(state[representative_id], event)
    except RuntimeError:
        return None
    return {replica_id: sorted(hashes) for replica_id, hashes in sorted(state.items())}


def _rebuild_rotated_stream(
    rows: list[dict[str, object]],
    *,
    replica_id: object,
    replica_generation: object,
    stream_epoch: object,
    high_sequence: object,
) -> list[str] | None:
    if (
        not isinstance(replica_id, str)
        or not isinstance(replica_generation, str)
        or not isinstance(stream_epoch, str)
        or type(high_sequence) is not int
        or high_sequence < 0
    ):
        return None
    emits = sorted(
        (
            row
            for row in rows
            if row.get("type") == "event_emit"
            and row.get("replica_id") == replica_id
            and row.get("replica_generation") == replica_generation
            and row.get("stream_epoch") == stream_epoch
        ),
        key=lambda row: int(row.get("sequence", -1)),
    )
    if [row.get("sequence") for row in emits] != list(range(high_sequence + 1)):
        return None
    hashes: set[str] = set()
    try:
        for row in emits:
            event = row.get("event")
            if not isinstance(event, dict):
                return None
            _apply_hash_event(hashes, event)
    except RuntimeError:
        return None
    return sorted(hashes)


def derive_metrics(
    rows: list[dict[str, object]],
    config: WorkloadConfig,
) -> dict[str, object]:
    grouped = _rows_by_type(rows)
    emits = grouped.get("event_emit", [])
    applies = grouped.get("event_apply", [])
    logical = grouped.get("logical_apply", [])
    heartbeats = grouped.get("heartbeat", [])
    routes = grouped.get("route", [])
    route_facts = _route_facts(rows, config)
    binding_replay = _binding_recovery(grouped)
    kill = _kill_by_action(grouped)
    pause_ns = kill.get("pause", {}).get("observed_ns")
    resume_ns = kill.get("resume", {}).get("observed_ns")

    live_wire_latencies = [
        int(row["applied_ns"]) - int(row["emitted_ns"])
        for row in applies
        if row.get("result") == "applied"
        and row.get("replayed") is False
        and type(row.get("emitted_ns")) is int
        and type(row.get("applied_ns")) is int
    ]
    receive_latencies = [
        int(row["received_ns"]) - int(row["emitted_ns"])
        for row in grouped.get("event_receive", [])
        if row.get("replayed") is False
        and type(row.get("emitted_ns")) is int
        and type(row.get("received_ns")) is int
    ]
    heartbeat_latencies = [
        int(row["applied_ns"]) - int(row["emitted_ns"])
        for row in heartbeats
        if row.get("result") == "refreshed"
        and type(row.get("emitted_ns")) is int
        and type(row.get("applied_ns")) is int
    ]
    logical_latencies = [
        int(row["applied_ns"]) - int(row["emitted_ns"])
        for row in logical
        if type(row.get("emitted_ns")) is int and type(row.get("applied_ns")) is int
    ]
    selected_times = [
        int(row["selected_ns"])
        for row in sorted(routes, key=lambda item: int(item.get("sequence", -1)))
        if type(row.get("selected_ns")) is int
    ]
    route_gaps = [
        later - earlier for earlier, later in zip(selected_times, selected_times[1:], strict=False)
    ]
    lateness = [
        int(row["selected_ns"]) - int(row["scheduled_ns"])
        for row in routes
        if type(row.get("selected_ns")) is int and type(row.get("scheduled_ns")) is int
    ]
    exact_ages = [
        max(
            int(fact["wire_truth_age_ns"]),
            int(fact["logical_truth_age_ns"]),
        )
        for fact in route_facts
        if fact.get("row", {}).get("reason") == "kv_event_match"
        and type(fact.get("wire_truth_age_ns")) is int
        and type(fact.get("logical_truth_age_ns")) is int
    ]
    stale_approximate_selected = [
        int(fact["row"]["selected_ns"])
        for fact in route_facts
        if fact.get("phase") == "stale"
        and fact.get("row", {}).get("reason") == "prefix_match"
        and fact.get("row", {}).get("mode") == "approximate"
        and type(fact.get("row", {}).get("selected_ns")) is int
    ]
    replay_completed_ns = binding_replay.get("completed_ns")
    recovery_snapshot = _snapshot_by_phase(grouped, "recovery")
    action_lateness = [
        int(row["observed_ns"]) - int(row["scheduled_ns"])
        for row in kill.values()
        if type(row.get("observed_ns")) is int and type(row.get("scheduled_ns")) is int
    ]
    return {
        "logical_index": {
            "replicas": config.replicas,
            "in_process_feeds": config.replicas - 1,
            "apply_samples": len(logical_latencies),
            "max_apply_ns": (max(logical_latencies) if logical_latencies else None),
        },
        "representative_wire": {
            "physical_feeds": 1,
            "emitted_events": len(emits),
            "received_events": len(grouped.get("event_receive", [])),
            "applied_events": len(applies),
            "live_apply_samples": len(live_wire_latencies),
            "max_live_apply_ns": (max(live_wire_latencies) if live_wire_latencies else None),
            "max_live_receive_ns": (max(receive_latencies) if receive_latencies else None),
            "heartbeat_samples": len(heartbeat_latencies),
            "max_heartbeat_apply_ns": (max(heartbeat_latencies) if heartbeat_latencies else None),
        },
        "routes": {
            "total": len(routes),
            "fresh": sum(fact.get("phase") == "fresh" for fact in route_facts),
            "stale": sum(fact.get("phase") == "stale" for fact in route_facts),
            "restored": sum(fact.get("phase") == "restored" for fact in route_facts),
            "max_exact_truth_age_ns": (max(exact_ages) if exact_ages else None),
            "max_schedule_lateness_ns": (max(lateness) if lateness else None),
            "max_actual_selected_gap_ns": (max(route_gaps) if route_gaps else None),
        },
        "chaos": {
            "first_stale_approximate_route_after_pause_ns": (
                min(stale_approximate_selected) - int(pause_ns)
                if stale_approximate_selected and type(pause_ns) is int
                else None
            ),
            "max_action_lateness_ns": (max(action_lateness) if action_lateness else None),
            "recovery_convergence_ns": (
                int(replay_completed_ns) - int(resume_ns)
                if type(replay_completed_ns) is int and type(resume_ns) is int
                else None
            ),
            "replay_mode": binding_replay.get("mode"),
            "replay_requested_from": binding_replay.get("requested_from"),
            "replay_high_sequence": binding_replay.get("high_sequence"),
            "replay_event_count": binding_replay.get("event_count"),
            "recovery_authoritative_digest": recovery_snapshot.get("authoritative_digest"),
            "recovery_index_digest": recovery_snapshot.get("index_digest"),
        },
    }


def threshold_checks(
    rows: list[dict[str, object]],
    config: WorkloadConfig,
) -> dict[str, bool]:
    grouped = _rows_by_type(rows)
    run = _single_row(grouped, "run")
    schedules = grouped.get("schedule", [])
    emits = sorted(
        grouped.get("event_emit", []),
        key=lambda row: int(row.get("enqueued_ns", -1)),
    )
    receives = grouped.get("event_receive", [])
    applies = grouped.get("event_apply", [])
    logical = grouped.get("logical_apply", [])
    routes = sorted(
        grouped.get("route", []),
        key=lambda row: int(row.get("sequence", -1)),
    )
    route_facts = _route_facts(rows, config)
    metric = derive_metrics(rows, config)
    topology = topology_descriptor(config)
    recorded_topology = run.get("topology")
    transport = (
        recorded_topology.get("zmq_transport") if isinstance(recorded_topology, dict) else None
    )
    expected_topology = {**topology, "zmq_transport": transport}
    freshness_ns = int(config.freshness_limit_seconds * _NS_PER_SECOND)
    stale_ns = int(config.stale_after_seconds * _NS_PER_SECOND)
    representative_id = representative_replica_id(config)
    replica_ids = [_replica_id(ordinal) for ordinal in range(config.replicas)]
    origin_ns = run.get("measurement_origin_ns")
    interval_ns = int(_NS_PER_SECOND / config.route_rate)

    schedule_payloads = [
        {key: value for key, value in row.items() if key not in {"type", "row_index"}}
        for row in schedules
    ]
    emit_by_key = {(row.get("stream_epoch"), row.get("sequence")): row for row in emits}
    emits_by_epoch: dict[str, list[dict[str, object]]] = {}
    for row in emits:
        stream_epoch = row.get("stream_epoch")
        if isinstance(stream_epoch, str):
            emits_by_epoch.setdefault(stream_epoch, []).append(row)

    def delivery_matches_emit(row: Mapping[str, object]) -> bool:
        emit = emit_by_key.get((row.get("stream_epoch"), row.get("sequence")))
        return bool(
            emit is not None
            and row.get("replica_id") == representative_id
            and row.get("replica_generation") == emit.get("replica_generation")
            and row.get("event") == emit.get("event")
            and row.get("event_kind") == emit.get("event_kind")
            and row.get("emitted_ns") == emit.get("emitted_ns")
            and type(row.get("received_ns")) is int
            and type(row.get("applied_ns")) is int
            and type(row.get("emitted_ns")) is int
            and int(row["emitted_ns"]) <= int(row["received_ns"]) <= int(row["applied_ns"])
        )

    route_sequence_and_ticks = (
        type(origin_ns) is int
        and [row.get("sequence") for row in routes]
        == list(range(config.expected_measurement_routes))
        and all(
            row.get("scheduled_ns") == int(origin_ns) + sequence * interval_ns
            and type(row.get("selected_ns")) is int
            and int(row["selected_ns"]) >= int(row["scheduled_ns"])
            and type(row.get("completed_ns")) is int
            and int(row["completed_ns"]) >= int(row["selected_ns"])
            for sequence, row in enumerate(routes)
        )
        and all(
            int(earlier["selected_ns"]) <= int(later["selected_ns"])
            for earlier, later in zip(routes, routes[1:], strict=False)
        )
    )
    route_oracle_inputs = all(
        fact.get("epoch") == row.get("epoch")
        and row.get("prompt_sha256")
        == hashlib.sha256(_epoch_prompt(int(fact["epoch"])).encode()).hexdigest()
        and row.get("prefix_fingerprint")
        == prefix_root_fingerprint(_epoch_prompt(int(fact["epoch"])))
        and row.get("session_id")
        == _session_for_approximate(
            int(fact["epoch"]),
            str(fact["expected_exact_replica"]),
            str(fact["expected_approximate_replica"]),
        )
        and row.get("expected_exact_replica") == fact.get("expected_exact_replica")
        and row.get("expected_approximate_replica") == fact.get("expected_approximate_replica")
        and row.get("backend_replica") == row.get("selected_replica")
        for fact in route_facts
        for row in [fact["row"]]
        if fact.get("phase") != "invalid"
    ) and len(route_facts) == len(routes)
    fresh_facts = [fact for fact in route_facts if fact.get("phase") == "fresh"]
    stale_facts = [fact for fact in route_facts if fact.get("phase") == "stale"]
    restored_facts = [fact for fact in route_facts if fact.get("phase") == "restored"]

    expected_churn = []
    if type(origin_ns) is int:
        for epoch, plan_row in enumerate(compressed_churn_plan(config)):
            ordinals = plan_row["ordinals"]
            if not isinstance(ordinals, list):
                continue
            spacing_ns = int(
                config.churn_window_seconds * _NS_PER_SECOND / max(1, len(ordinals) - 1)
            )
            for batch_index, ordinal in enumerate(ordinals):
                expected_churn.append(
                    {
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "ordinal": ordinal,
                        "replica_id": (_replica_id(ordinal) if type(ordinal) is int else None),
                        "scheduled_ns": (
                            int(origin_ns)
                            + int(epoch * config.epoch_seconds * _NS_PER_SECOND)
                            + batch_index * spacing_ns
                        ),
                    }
                )
    churn_rows = sorted(
        grouped.get("churn", []),
        key=lambda row: (
            int(row.get("epoch", -1)),
            int(row.get("batch_index", -1)),
        ),
    )
    churn_shape = len(churn_rows) == len(expected_churn) and all(
        all(row.get(key) == expected.get(key) for key in expected)
        and isinstance(row.get("old_generation"), str)
        and isinstance(row.get("new_generation"), str)
        and row.get("old_generation") != row.get("new_generation")
        and type(row.get("started_ns")) is int
        and type(row.get("completed_ns")) is int
        and int(row["started_ns"]) >= int(row["scheduled_ns"])
        and int(row["completed_ns"]) >= int(row["started_ns"])
        for row, expected in zip(churn_rows, expected_churn, strict=True)
    )
    membership = grouped.get("membership", [])
    membership_join = churn_shape
    if membership_join:
        for churn in churn_rows:
            joined = [
                row
                for row in membership
                if row.get("epoch") == churn.get("epoch")
                and row.get("ordinal") == churn.get("ordinal")
            ]
            if (
                [row.get("reason") for row in joined]
                != ["manual_drain", "replica_removed", "replica_added"]
                or joined[0].get("replica_generation") != churn.get("old_generation")
                or joined[1].get("replica_generation") != churn.get("old_generation")
                or joined[2].get("replica_generation") != churn.get("new_generation")
                or any(
                    type(row.get("observed_ns")) is not int
                    or not (
                        int(churn["started_ns"])
                        <= int(row["observed_ns"])
                        <= int(churn["completed_ns"])
                    )
                    for row in joined
                )
            ):
                membership_join = False
                break
    epoch_route_overlap = churn_shape and all(
        any(
            type(route.get("selected_ns")) is int
            and int(epoch_churn[0]["started_ns"])
            <= int(route["selected_ns"])
            <= int(epoch_churn[-1]["completed_ns"])
            for route in routes
        )
        for epoch in range(config.churn_epochs)
        for epoch_churn in [[row for row in churn_rows if row.get("epoch") == epoch]]
        if epoch_churn
    )

    initial_direct = {
        row.get("replica_id") for row in logical if row.get("purpose") == "initialization"
    }
    initial_wire = {
        row.get("replica_id") for row in emits if row.get("purpose") == "initialization"
    }
    replacement_direct = {
        row.get("replica_id") for row in logical if row.get("purpose") == "replacement_clear"
    }
    replacement_wire = {
        row.get("replica_id") for row in emits if row.get("purpose") == "replacement_clear"
    }
    prefix_seeds = sorted(
        grouped.get("prefix_seed", []),
        key=lambda row: int(row.get("epoch", -1)),
    )
    prefix_seed_exact = len(prefix_seeds) == config.churn_epochs and all(
        row.get("epoch") == epoch
        and row.get("replica_id") == _epoch_targets(config, epoch)[1]
        and row.get("prompt_sha256") == hashlib.sha256(_epoch_prompt(epoch).encode()).hexdigest()
        and row.get("prefix_fingerprint") == prefix_root_fingerprint(_epoch_prompt(epoch))
        for epoch, row in enumerate(prefix_seeds)
    )

    logical_stream_rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    logical_generation_epochs: dict[tuple[str, str], set[str]] = {}
    logical_shape = True
    for row in sorted(
        logical,
        key=lambda item: int(item.get("applied_ns", -1)),
    ):
        replica_id = row.get("replica_id")
        generation = row.get("replica_generation")
        stream_epoch = row.get("stream_epoch")
        sequence = row.get("sequence")
        if (
            not isinstance(replica_id, str)
            or not isinstance(generation, str)
            or not isinstance(stream_epoch, str)
            or not stream_epoch
            or type(sequence) is not int
            or row.get("result") != "applied"
            or row.get("synchronized") is not True
        ):
            logical_shape = False
            continue
        key = (replica_id, generation, stream_epoch)
        logical_stream_rows.setdefault(key, []).append(row)
        logical_generation_epochs.setdefault((replica_id, generation), set()).add(stream_epoch)
    logical_shape = (
        logical_shape
        and all(
            [row.get("sequence") for row in stream_rows] == list(range(len(stream_rows)))
            for stream_rows in logical_stream_rows.values()
        )
        and all(len(stream_epochs) == 1 for stream_epochs in logical_generation_epochs.values())
    )
    logical_heartbeats = grouped.get("logical_heartbeat", [])
    for heartbeat in logical_heartbeats:
        heartbeat_ids = heartbeat.get("replica_ids")
        high_water = heartbeat.get("stream_high_water")
        finished_ns = heartbeat.get("finished_ns")
        if (
            not isinstance(heartbeat_ids, list)
            or len(heartbeat_ids) != config.replicas - 1
            or set(heartbeat_ids) != set(replica_ids) - {representative_id}
            or heartbeat.get("replica_count") != config.replicas - 1
            or heartbeat.get("all_refreshed") is not True
            or not isinstance(high_water, dict)
            or set(high_water) != set(heartbeat_ids)
            or type(finished_ns) is not int
        ):
            logical_shape = False
            break
        for replica_id in heartbeat_ids:
            candidates = [
                row
                for row in logical
                if row.get("replica_id") == replica_id
                and type(row.get("applied_ns")) is int
                and int(row["applied_ns"]) <= int(finished_ns)
            ]
            latest = max(
                candidates,
                key=lambda item: int(item["applied_ns"]),
                default=None,
            )
            recorded = high_water.get(replica_id)
            if (
                latest is None
                or not isinstance(recorded, dict)
                or recorded.get("stream_epoch") != latest.get("stream_epoch")
                or recorded.get("high_sequence") != latest.get("sequence")
            ):
                logical_shape = False
                break
        if not logical_shape:
            break

    snapshots = grouped.get("fleet_snapshot", [])
    final_fleet_candidates = [row for row in snapshots if row.get("phase") == "final"]
    final_fleet = final_fleet_candidates[0] if len(final_fleet_candidates) == 1 else {}
    final_generations = final_fleet.get("generation_by_id")
    final_replica_ids = final_fleet.get("replica_ids")
    final_healthy_ids = final_fleet.get("healthy_ids")
    final_eligible_ids = final_fleet.get("eligible_ids")
    final_membership = (
        isinstance(final_replica_ids, list)
        and isinstance(final_healthy_ids, list)
        and isinstance(final_eligible_ids, list)
        and sorted(final_replica_ids) == replica_ids
        and sorted(final_healthy_ids) == replica_ids
        and sorted(final_eligible_ids) == replica_ids
        and final_fleet.get("replica_digest") == _sha256_json(final_replica_ids)
        and final_fleet.get("eligible_digest") == _sha256_json(final_eligible_ids)
        and isinstance(final_generations, dict)
        and set(final_generations) == set(replica_ids)
        and all(
            final_generations.get(row["replica_id"]) == row.get("new_generation")
            for row in churn_rows
        )
    )

    kill = _kill_by_action(grouped)
    pause = kill.get("pause", {})
    resume = kill.get("resume", {})
    kill_schedule = (
        len(grouped.get("kill", [])) == 2
        and type(origin_ns) is int
        and pause.get("fault") == "subscriber_live_socket_close"
        and resume.get("fault") == "subscriber_live_socket_reconnect"
        and pause.get("scheduled_ns")
        == int(origin_ns) + int(config.kill_offset_seconds * _NS_PER_SECOND)
        and resume.get("scheduled_ns")
        == int(origin_ns) + int(config.restore_offset_seconds * _NS_PER_SECOND)
        and all(
            type(row.get("requested_ns")) is int
            and type(row.get("observed_ns")) is int
            and int(row["scheduled_ns"]) <= int(row["requested_ns"]) <= int(row["observed_ns"])
            for row in (pause, resume)
        )
    )
    process = grouped.get("process_identity", [])
    stable_fields = (
        "pid",
        "pool_id",
        "routing_id",
        "index_id",
        "publisher_id",
        "publisher_thread_id",
        "subscriber_id",
    )
    process_stable = (
        [row.get("phase") for row in process] == ["ready", "outage", "restored"]
        and all(
            row.get(field) == process[0].get(field) for row in process for field in stable_fields
        )
        and process[0].get("subscriber_socket_id")
        == process[1].get("subscriber_socket_id")
        == pause.get("subscriber_socket_id")
        and process[2].get("subscriber_socket_id") == resume.get("subscriber_socket_id")
        and process[2].get("subscriber_socket_id") != process[0].get("subscriber_socket_id")
    )

    binding_replay = _binding_recovery(grouped)
    binding_generation = None
    representative_churn = [row for row in churn_rows if row.get("replica_id") == representative_id]
    if len(representative_churn) == 1:
        binding_generation = representative_churn[0].get("new_generation")
    stream_rotations = [row for row in grouped.get("stream", []) if row.get("action") == "rotate"]
    rotation = stream_rotations[0] if len(stream_rotations) == 1 else {}
    initial_stream_epoch = emits[0].get("stream_epoch") if emits else None
    final_stream_epoch = rotation.get("stream_epoch")
    physical_stream_shape = (
        len(representative_churn) == 1
        and len(stream_rotations) == 1
        and len(emits_by_epoch) == 2
        and isinstance(initial_stream_epoch, str)
        and isinstance(final_stream_epoch, str)
        and initial_stream_epoch != final_stream_epoch
        and set(emits_by_epoch) == {initial_stream_epoch, final_stream_epoch}
        and rotation.get("previous_stream_epoch") == initial_stream_epoch
        and rotation.get("replica_id") == representative_id
        and rotation.get("epoch") == representative_churn[0].get("epoch")
        and rotation.get("ordinal") == representative_churn[0].get("ordinal")
        and type(rotation.get("observed_ns")) is int
        and int(representative_churn[0]["started_ns"])
        <= int(rotation["observed_ns"])
        <= int(representative_churn[0]["completed_ns"])
        and all(
            [row.get("sequence") for row in stream_rows] == list(range(len(stream_rows)))
            and len({row.get("replica_generation") for row in stream_rows}) == 1
            and all(
                row.get("replica_id") == representative_id
                and isinstance(row.get("event"), dict)
                and type(row.get("enqueued_ns")) is int
                and type(row.get("emitted_ns")) is int
                and int(row["enqueued_ns"]) <= int(row["emitted_ns"])
                for row in stream_rows
            )
            for stream_rows in emits_by_epoch.values()
        )
        and {row.get("replica_generation") for row in emits_by_epoch[initial_stream_epoch]}
        == {representative_churn[0].get("old_generation")}
        and {row.get("replica_generation") for row in emits_by_epoch[final_stream_epoch]}
        == {representative_churn[0].get("new_generation")}
    )
    requested_from = binding_replay.get("requested_from")
    high_sequence = binding_replay.get("high_sequence")
    replayed_applies = sorted(
        (
            row
            for row in applies
            if row.get("replayed") is True
            and row.get("replica_id") == representative_id
            and row.get("replica_generation") == binding_generation
            and row.get("stream_epoch") == final_stream_epoch
        ),
        key=lambda row: int(row.get("sequence", -1)),
    )
    replay_sequences = [row.get("sequence") for row in replayed_applies]
    replay_requested_ns = binding_replay.get("requested_ns")
    replay_response_ns = binding_replay.get("response_received_ns")
    replay_completed_ns = binding_replay.get("completed_ns")
    replay_causality = (
        type(replay_requested_ns) is int
        and type(replay_response_ns) is int
        and type(replay_completed_ns) is int
        and int(replay_requested_ns) <= int(replay_response_ns) <= int(replay_completed_ns)
        and all(
            type(row.get("received_ns")) is int
            and type(row.get("applied_ns")) is int
            and int(row["received_ns"]) == int(replay_response_ns)
            and int(replay_response_ns) <= int(row["applied_ns"]) <= int(replay_completed_ns)
            for row in replayed_applies
        )
    )
    replay_complete = (
        bool(binding_replay)
        and binding_replay.get("replica_id") == representative_id
        and binding_replay.get("replica_generation") == binding_generation
        and binding_replay.get("stream_epoch") == final_stream_epoch
        and binding_replay.get("mode") == "events"
        and type(requested_from) is int
        and type(high_sequence) is int
        and requested_from <= high_sequence
        and replay_sequences == list(range(int(requested_from), int(high_sequence) + 1))
        and binding_replay.get("event_count") == len(replayed_applies)
        and binding_replay.get("last_available") == high_sequence
        and type(binding_replay.get("first_available")) is int
        and int(binding_replay["first_available"]) <= int(requested_from)
        and replay_causality
        and all(row.get("result") in {"applied", "duplicate"} for row in replayed_applies)
    )
    applied_keys = [
        (
            row.get("replica_generation"),
            row.get("stream_epoch"),
            row.get("sequence"),
        )
        for row in applies
        if row.get("result") == "applied"
    ]
    unique_apply_keys = len(applied_keys) == len(set(applied_keys))

    rebuilt_state = _rebuild_authoritative_state(rows, config)
    final_snapshot = _snapshot_by_phase(grouped, "final")
    measured_authoritative = final_snapshot.get("authoritative_hashes_by_replica")
    measured_index = final_snapshot.get("index_hashes_by_replica")
    final_state_converged = (
        rebuilt_state is not None
        and rebuilt_state == measured_authoritative == measured_index
        and final_snapshot.get("authoritative_digest") == _sha256_json(rebuilt_state)
        and final_snapshot.get("index_digest") == _sha256_json(rebuilt_state)
    )
    recovery_snapshot = _snapshot_by_phase(grouped, "recovery")
    recovery_view = recovery_snapshot.get("index_view")
    rebuilt_recovery = _rebuild_rotated_stream(
        rows,
        replica_id=representative_id,
        replica_generation=binding_generation,
        stream_epoch=final_stream_epoch,
        high_sequence=high_sequence,
    )
    recovery_authoritative = recovery_snapshot.get("authoritative_hashes")
    recovery_index = recovery_snapshot.get("index_hashes")
    recovery_state_converged = (
        rebuilt_recovery is not None
        and isinstance(recovery_view, dict)
        and recovery_snapshot.get("replica_id") == representative_id
        and recovery_snapshot.get("replica_generation") == binding_generation
        and recovery_snapshot.get("stream_epoch") == final_stream_epoch
        and recovery_snapshot.get("high_sequence") == high_sequence
        and recovery_snapshot.get("replay_completed_ns") == replay_completed_ns
        and type(recovery_snapshot.get("observed_ns")) is int
        and type(replay_completed_ns) is int
        and int(replay_completed_ns)
        <= int(recovery_snapshot["observed_ns"])
        < int(replay_completed_ns) + freshness_ns
        and recovery_view.get("replica_id") == representative_id
        and recovery_view.get("stream_epoch") == final_stream_epoch
        and recovery_view.get("last_sequence") == high_sequence
        and recovery_view.get("state") == "live"
        and recovery_view.get("trusted") is True
        and recovery_view.get("synchronized") is True
        and recovery_view.get("block_hashes") == rebuilt_recovery
        and rebuilt_recovery == recovery_authoritative == recovery_index
        and recovery_snapshot.get("authoritative_digest") == _sha256_json(rebuilt_recovery)
        and recovery_snapshot.get("index_digest") == _sha256_json(rebuilt_recovery)
    )
    transport_summaries = grouped.get("transport_summary", [])
    transport_summary = transport_summaries[0] if len(transport_summaries) == 1 else {}

    logical_max = metric["logical_index"]["max_apply_ns"]
    wire_max = metric["representative_wire"]["max_live_apply_ns"]
    heartbeat_max = metric["representative_wire"]["max_heartbeat_apply_ns"]
    exact_age_max = metric["routes"]["max_exact_truth_age_ns"]
    route_lateness_max = metric["routes"]["max_schedule_lateness_ns"]
    route_gap_max = metric["routes"]["max_actual_selected_gap_ns"]
    fallback_after_pause_ns = metric["chaos"]["first_stale_approximate_route_after_pause_ns"]
    action_lateness_max = metric["chaos"]["max_action_lateness_ns"]
    recovery_ns = metric["chaos"]["recovery_convergence_ns"]
    return {
        "topology_is_exactly_200_logical_199_in_process_1_physical": (
            recorded_topology == expected_topology
            and topology["logical_replicas"] == config.replicas
            and topology["in_process_feeds"] == config.replicas - 1
            and topology["physical_zmq_feeds"] == 1
            and (config.profile != "formal" or transport == "ipc")
            and transport in {"ipc", "inproc"}
        ),
        "schedule_matches_frozen_plan": (schedule_payloads == compressed_churn_plan(config)),
        "all_planned_churn_operations_execute_once_with_new_generations": (churn_shape),
        "membership_audit_joins_drain_remove_and_add": membership_join,
        "routing_overlaps_every_churn_epoch": epoch_route_overlap,
        "final_fleet_is_exactly_the_same_eligible_id_set": final_membership,
        "all_200_feed_identities_initialize_and_churn": (
            initial_direct == set(replica_ids) - {representative_id}
            and initial_wire == {representative_id}
            and replacement_direct == set(replica_ids) - {representative_id}
            and replacement_wire == {representative_id}
        ),
        "logical_feeds_are_sequenced_synchronized_and_heartbeat_bound": (
            logical_shape and bool(logical_heartbeats)
        ),
        "approximate_oracle_seeds_are_frozen": prefix_seed_exact,
        "physical_feed_rotates_once_and_is_contiguous_per_cache_lifetime": (physical_stream_shape),
        "every_wire_receive_and_apply_causally_joins_an_emit": (
            bool(receives)
            and bool(applies)
            and all(delivery_matches_emit(row) for row in receives)
            and all(delivery_matches_emit(row) for row in applies)
        ),
        "applied_sequence_keys_are_unique_per_replica_generation": (unique_apply_keys),
        "logical_index_apply_freshness_is_strictly_below_500ms": (
            type(logical_max) is int and 0 <= int(logical_max) < freshness_ns
        ),
        "representative_wire_live_freshness_is_strictly_below_500ms": (
            type(wire_max) is int
            and 0 <= int(wire_max) < freshness_ns
            and type(heartbeat_max) is int
            and 0 <= int(heartbeat_max) < freshness_ns
        ),
        "kill_and_restore_follow_the_frozen_actual_time_schedule": (kill_schedule),
        "pause_and_resume_action_lateness_is_strictly_below_500ms": (
            type(action_lateness_max) is int and 0 <= int(action_lateness_max) < freshness_ns
        ),
        "pool_router_index_publisher_and_subscriber_never_restart": (process_stable),
        "route_ticks_are_absolute_complete_and_monotonic": (route_sequence_and_ticks),
        "max_route_schedule_lateness_is_strictly_below_500ms": (
            type(route_lateness_max) is int and 0 <= int(route_lateness_max) < freshness_ns
        ),
        "max_actual_selected_route_gap_is_strictly_below_500ms": (
            type(route_gap_max) is int and 0 <= int(route_gap_max) < freshness_ns
        ),
        "route_oracle_inputs_recompute_from_the_frozen_trace": (route_oracle_inputs),
        "fresh_routes_use_the_exact_oracle": (
            bool(fresh_facts)
            and all(
                fact["row"].get("reason") == "kv_event_match"
                and fact["row"].get("mode") == "exact"
                and fact["row"].get("selected_replica") == fact.get("expected_exact_replica")
                for fact in fresh_facts
            )
        ),
        "every_route_after_actual_lease_expiry_uses_approximate_oracle": (
            bool(stale_facts)
            and all(
                fact["row"].get("reason") == "prefix_match"
                and fact["row"].get("mode") == "approximate"
                and fact["row"].get("selected_replica") == fact.get("expected_approximate_replica")
                and fact["row"].get("selected_replica") != fact.get("expected_exact_replica")
                for fact in stale_facts
            )
        ),
        "first_stale_approximate_route_after_pause_is_strictly_below_500ms": (
            type(fallback_after_pause_ns) is int
            and 0 <= int(fallback_after_pause_ns) < freshness_ns
        ),
        "restored_routes_use_the_exact_oracle": (
            bool(restored_facts)
            and all(
                fact["row"].get("reason") == "kv_event_match"
                and fact["row"].get("mode") == "exact"
                and fact["row"].get("selected_replica") == fact.get("expected_exact_replica")
                for fact in restored_facts
            )
        ),
        "every_exact_decision_uses_truth_younger_than_the_250ms_lease": (
            type(exact_age_max) is int
            and 0 <= int(exact_age_max) < stale_ns < freshness_ns
            and all(
                fact.get("row", {}).get("reason") != "kv_event_match"
                or (
                    type(fact.get("wire_truth_age_ns")) is int
                    and type(fact.get("logical_truth_age_ns")) is int
                    and 0 <= int(fact["wire_truth_age_ns"]) < stale_ns
                    and 0 <= int(fact["logical_truth_age_ns"]) < stale_ns
                )
                for fact in route_facts
            )
        ),
        "binding_replay_is_contiguous_complete_and_causally_applied": (replay_complete),
        "recovery_snapshot_rebuilds_rotated_epoch_at_replay_completion": (recovery_state_converged),
        "final_state_rebuilds_the_authoritative_200_replica_digest": (final_state_converged),
        "restore_recovery_completes_strictly_below_500ms": (
            type(recovery_ns) is int and 0 <= int(recovery_ns) < freshness_ns
        ),
        "publisher_reports_no_unjournaled_live_drop": (
            transport_summary.get("publisher_dropped_live_events") == 0
            and transport_summary.get("publisher_stream_epoch") == final_stream_epoch
            and transport_summary.get("publisher_high_sequence")
            == len(emits_by_epoch.get(str(final_stream_epoch), [])) - 1
            and transport_summary.get("recovery_count") == len(grouped.get("replay", []))
        ),
    }


def _replay_rows(
    rows: list[dict[str, object]],
    config: WorkloadConfig,
) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    checks["row_indices_are_contiguous"] = all(
        row.get("row_index") == index for index, row in enumerate(rows)
    )
    allowed_types = {
        "run",
        "schedule",
        "process_identity",
        "fleet_snapshot",
        "membership",
        "churn",
        "logical_apply",
        "logical_heartbeat",
        "prefix_seed",
        "event_emit",
        "event_receive",
        "event_apply",
        "heartbeat",
        "kill",
        "stream",
        "replay",
        "route",
        "state_snapshot",
        "transport_summary",
    }
    checks["raw_row_types_are_known"] = all(row.get("type") in allowed_types for row in rows)
    run_rows = [row for row in rows if row.get("type") == "run"]
    checks["exactly_one_run_row"] = len(run_rows) == 1
    if len(run_rows) != 1:
        errors.append("raw evidence must contain exactly one run row")
    checks.update(threshold_checks(rows, config))
    return checks, errors


def verify_artifact(
    path: Path,
    *,
    required_profile: str | None = None,
    require_current_source: bool = False,
    require_current_runner_context: bool = False,
    actions_run_provenance: Path | None = None,
) -> dict[str, object]:
    manifest_path = _resolve_manifest(path).resolve()
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        config_raw = manifest.get("config")
        if not isinstance(config_raw, dict):
            raise ValueError("manifest config is missing")
        profile = config_raw.get("profile")
        if not isinstance(profile, str):
            raise ValueError("manifest profile is invalid")
        config = profile_config(profile)
        checks["required_profile_matches"] = required_profile is None or profile == required_profile
        checks["artifact_sources_match_current_worktree"] = (
            not require_current_source
            or artifact_source_matches_current(
                manifest_path,
                required_profile=required_profile,
            )
        )
        checks["artifact_github_context_matches_current_runner"] = (
            not require_current_runner_context
            or _github_context_matches_current_runner(manifest.get("environment"))
        )
        if actions_run_provenance is not None:
            provenance = json.loads(actions_run_provenance.resolve().read_text(encoding="utf-8"))
            checks.update(
                _verify_actions_run_provenance(
                    provenance,
                    manifest.get("source"),
                    manifest.get("environment"),
                )
            )
        checks["schema_version_matches"] = manifest.get("schema_version") == SCHEMA_VERSION
        checks["gate_matches"] = manifest.get("gate") == GATE
        checks["config_matches_frozen_profile"] = config_raw == asdict(config)
        checks["schedule_matches_frozen_profile"] = manifest.get(
            "schedule"
        ) == compressed_churn_plan(config)
        checks["topology_matches_frozen_profile"] = (
            isinstance(manifest.get("topology"), dict)
            and manifest["topology"].get("logical_replicas") == config.replicas
            and manifest["topology"].get("in_process_feeds") == config.replicas - 1
            and manifest["topology"].get("physical_zmq_feeds") == 1
            and manifest["topology"].get("representative_replica_id")
            == representative_replica_id(config)
        )
        checks["reused_evidence_descriptor_is_exact"] = (
            manifest.get("reused_evidence") == reused_evidence_descriptor()
        )
        f2a_manifest = _F2A_RETAINED_DIR / "prefix-routing-f2a-manifest.json"
        f2a_raw = _F2A_RETAINED_DIR / "prefix-routing-f2a-raw.jsonl"
        checks["reused_f2a_manifest_digest_matches"] = (
            f2a_manifest.is_file() and _sha256_file(f2a_manifest) == F2A_MANIFEST_SHA256
        )
        checks["reused_f2a_raw_digest_matches"] = (
            f2a_raw.is_file() and _sha256_file(f2a_raw) == F2A_RAW_SHA256
        )
        raw_descriptor = manifest.get("raw")
        if not isinstance(raw_descriptor, dict):
            raise ValueError("raw descriptor is missing")
        raw_name = raw_descriptor.get("path")
        if not isinstance(raw_name, str) or Path(raw_name).name != raw_name or raw_name != RAW_NAME:
            raise ValueError("raw path is not the frozen adjacent filename")
        raw_path = (manifest_path.parent / raw_name).resolve()
        if raw_path.parent != manifest_path.parent:
            raise ValueError("raw path escapes artifact directory")
        rows = _read_jsonl(raw_path)
        checks["raw_sha256_matches"] = _valid_sha256(raw_descriptor.get("sha256")) and _sha256_file(
            raw_path
        ) == raw_descriptor.get("sha256")
        checks["raw_row_count_matches"] = raw_descriptor.get("rows") == len(rows)
        run_rows = [row for row in rows if row.get("type") == "run"]
        checks["one_run_record_exists"] = len(run_rows) == 1
        if len(run_rows) != 1:
            raise ValueError("raw evidence must contain one run row")
        run = run_rows[0]
        checks["raw_run_metadata_matches_manifest"] = (
            run.get("schema_version") == SCHEMA_VERSION
            and run.get("config") == asdict(config)
            and run.get("schedule_digest") == _sha256_json(compressed_churn_plan(config))
            and run.get("topology") == manifest.get("topology")
            and run.get("source") == manifest.get("source")
            and run.get("source_end") == manifest.get("source_end")
            and run.get("environment") == manifest.get("environment")
            and run.get("reused_evidence") == manifest.get("reused_evidence")
        )
        checks.update(_verify_source_descriptor(manifest.get("source")))
        checks["source_did_not_drift_during_run"] = _source_end_matches_start(
            manifest.get("source"), manifest.get("source_end")
        )
        checks["formal_github_context_is_complete"] = _formal_context_complete(
            config,
            manifest.get("source"),
            manifest.get("environment"),
        )
        replay_checks, replay_errors = _replay_rows(rows, config)
        checks.update(replay_checks)
        errors.extend(replay_errors)
        metrics = derive_metrics(rows, config)
        gates = threshold_checks(rows, config)
        checks["manifest_metrics_recomputed_exactly"] = manifest.get("metrics") == metrics
        checks["manifest_gate_checks_recomputed_exactly"] = manifest.get("gate_checks") == gates
        source = manifest.get("source")
        source_gate = (
            isinstance(source, dict)
            and source.get("git_clean_at_start") is True
            and source.get("expected_commit_matches") is True
            and _source_end_matches_start(source, manifest.get("source_end"))
            and _formal_context_complete(config, source, manifest.get("environment"))
        )
        checks["manifest_passed_recomputed_exactly"] = manifest.get("passed") is (
            source_gate
            and all(gates.values())
            and checks["reused_f2a_manifest_digest_matches"]
            and checks["reused_f2a_raw_digest_matches"]
        )
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
            raise RuntimeError("placement request IDs must be unique strings")
        self.records[request_id] = {
            "session_id": session_id,
            "replica_ordinal": replica,
            "reason": reason,
            **fields,
        }

    async def shutdown(self) -> None:
        return None


class _MockReplica:
    def __init__(
        self,
        replica_id: str,
        observations: dict[str, str],
    ) -> None:
        self.replica_id = replica_id
        self.observations = observations

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        self.observations[request.request_id] = self.replica_id
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(),
            finished=True,
        )

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationResult]:
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


async def _wait_until_ns(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        await asyncio.sleep(min(0.01, remaining_ns / _NS_PER_SECOND))


def _membership_snapshot_row(
    pool: ReplicaPool,
    *,
    phase: str,
    epoch: int | None,
) -> dict[str, object]:
    replica_ids, healthy_ids, eligible_ids, generations = pool.membership_snapshot()
    return {
        "type": "fleet_snapshot",
        "phase": phase,
        "epoch": epoch,
        "observed_ns": time.perf_counter_ns(),
        "replica_ids": list(replica_ids),
        "healthy_ids": list(healthy_ids),
        "eligible_ids": list(eligible_ids),
        "generation_by_id": generations,
        "replica_digest": _sha256_json(list(replica_ids)),
        "eligible_digest": _sha256_json(list(eligible_ids)),
    }


def _apply_hash_event(state: set[str], event: Mapping[str, object]) -> None:
    kind = event.get("type")
    hashes = event.get("block_hashes", [])
    if not isinstance(hashes, list) or any(not isinstance(value, str) for value in hashes):
        raise RuntimeError("fixture emitted a malformed hash event")
    if kind == "BlockStored":
        state.update(hashes)
    elif kind == "BlockRemoved":
        state.difference_update(hashes)
    elif kind == "AllBlocksCleared":
        state.clear()
    else:
        raise RuntimeError(f"fixture emitted an unknown event {kind!r}")


async def _collect_raw_rows(
    config: WorkloadConfig,
    source: dict[str, object],
    environment: dict[str, object],
) -> list[dict[str, object]]:
    """Exercise production selection with one representative physical stream."""

    config.validate()
    topology = topology_descriptor(config)
    schedule = compressed_churn_plan(config)
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": SCHEMA_VERSION,
            "config": asdict(config),
            "topology": topology,
            "schedule_digest": _sha256_json(schedule),
            "source": source,
            "environment": environment,
            "reused_evidence": reused_evidence_descriptor(),
            "measurement_origin_ns": None,
        }
    ]
    rows.extend({"type": "schedule", **item} for item in schedule)

    representative_id = str(topology["representative_replica_id"])
    exact_index = KvEventIndex(staleness_s=config.stale_after_seconds)
    approximate_index = PrefixIndex()
    current_epoch = {"value": 0}
    routing_index = KvRoutingIndex(
        approximate=approximate_index,
        exact=exact_index,
        block_hash_provider=lambda _prompt: _epoch_block_hashes(current_epoch["value"]),
    )
    observations: dict[str, str] = {}
    placement_log = _PlacementLog()
    backends = {
        _replica_id(ordinal): _MockReplica(_replica_id(ordinal), observations)
        for ordinal in range(config.replicas)
    }
    active_churn: dict[str, object] = {}

    def membership_sink(event: dict[str, object]) -> None:
        rows.append(
            {
                "type": "membership",
                "epoch": active_churn.get("epoch"),
                "ordinal": active_churn.get("ordinal"),
                "scheduled_ns": active_churn.get("scheduled_ns"),
                "observed_ns": time.perf_counter_ns(),
                **event,
            }
        )

    pool = ReplicaPool(
        backends,
        log=placement_log,  # type: ignore[arg-type]
        prefix_index=routing_index,
        membership_event_sink=membership_sink,
    )
    membership_lock = asyncio.Lock()
    authoritative: dict[str, set[str]] = {replica_id: set() for replica_id in pool.replica_ids}
    logical_streams: dict[str, tuple[str, str, int]] = {}
    wire_events: dict[tuple[str, int], dict[str, object]] = {}
    delivery_records: list[dict[str, object]] = []

    def on_delivery(delivery: dict[str, object]) -> None:
        emission = wire_events.get(
            (
                str(delivery.get("stream_epoch")),
                int(delivery.get("sequence", -1)),
            )
        )
        generation = (
            emission.get("replica_generation")
            if emission is not None
            else (
                pool.entry_generation(representative_id)
                if representative_id in pool.replica_ids
                else None
            )
        )
        delivery_records.append(
            {
                **delivery,
                "replica_generation": generation,
            }
        )

    with tempfile.TemporaryDirectory(prefix="kairyu-f2b-") as temporary:
        transport = os.environ.get("F2B_ZMQ_TRANSPORT", "ipc")
        if transport == "ipc":
            endpoint = f"ipc://{temporary}/events.sock"
            replay_endpoint = f"ipc://{temporary}/replay.sock"
        elif transport == "inproc" and config.profile == "smoke":
            transport_id = uuid.uuid4().hex
            endpoint = f"inproc://kairyu-f2b-{transport_id}-events"
            replay_endpoint = f"inproc://kairyu-f2b-{transport_id}-replay"
        else:
            raise ValueError("F2B_ZMQ_TRANSPORT must be ipc; smoke also permits inproc")
        topology["zmq_transport"] = transport
        publisher = ZmqKvEventPublisher(
            endpoint,
            representative_id,
            replay_endpoint=replay_endpoint,
            buffer_events=config.replay_buffer_steps,
            heartbeat_s=config.heartbeat_seconds,
        )
        wire_sink = publisher.event_sink()
        subscriber = ZmqKvEventSubscriber(
            [endpoint],
            exact_index,
            replay_endpoints={representative_id: replay_endpoint},
            replay_timeout_s=config.route_timeout_seconds,
            on_delivery=on_delivery,
        )

        def wire_publish(
            event: dict[str, object],
            *,
            epoch: int | None,
            purpose: str,
        ) -> int:
            enqueued_ns = time.perf_counter_ns()
            wire_sink(event)
            sequence = publisher.high_sequence
            stream_epoch = publisher.stream_epoch
            generation = (
                pool.entry_generation(representative_id)
                if representative_id in pool.replica_ids
                else None
            )
            wire_events[(stream_epoch, sequence)] = {
                "type": "event_emit",
                "channel": "representative_zmq",
                "replica_id": representative_id,
                "replica_generation": generation,
                "stream_epoch": stream_epoch,
                "sequence": sequence,
                "epoch": epoch,
                "purpose": purpose,
                "event_kind": event["type"],
                "event": json.loads(json.dumps(event)),
                "enqueued_ns": enqueued_ns,
                "emitted_ns": None,
            }
            _apply_hash_event(authoritative[representative_id], event)
            return sequence

        def direct_apply(
            replica_id: str,
            event: dict[str, object],
            *,
            epoch: int | None,
            purpose: str,
        ) -> None:
            generation = pool.entry_generation(replica_id)
            previous = logical_streams.get(replica_id)
            if previous is None or previous[0] != generation:
                stream_epoch = f"logical:{replica_id}:{generation}"
                sequence = 0
            else:
                stream_epoch = previous[1]
                sequence = previous[2] + 1
            emitted_ns = time.perf_counter_ns()
            result = exact_index.apply_sequenced(
                replica_id,
                stream_epoch,
                sequence,
                event,
            )
            synchronized = exact_index.mark_synchronized(
                replica_id,
                stream_epoch,
                sequence,
            )
            applied_ns = time.perf_counter_ns()
            if result != "applied" or not synchronized:
                raise RuntimeError(
                    f"logical KV feed failed to synchronize {replica_id!r}: "
                    f"{result=}, {synchronized=}"
                )
            logical_streams[replica_id] = (
                generation,
                stream_epoch,
                sequence,
            )
            _apply_hash_event(authoritative[replica_id], event)
            rows.append(
                {
                    "type": "logical_apply",
                    "channel": "in_process",
                    "replica_id": replica_id,
                    "replica_generation": generation,
                    "stream_epoch": stream_epoch,
                    "sequence": sequence,
                    "result": result,
                    "synchronized": synchronized,
                    "epoch": epoch,
                    "purpose": purpose,
                    "event_kind": event["type"],
                    "event": json.loads(json.dumps(event)),
                    "emitted_ns": emitted_ns,
                    "applied_ns": applied_ns,
                }
            )

        def identity_row(phase: str) -> dict[str, object]:
            return {
                "type": "process_identity",
                "phase": phase,
                "observed_ns": time.perf_counter_ns(),
                "pid": os.getpid(),
                "pool_id": id(pool),
                "routing_id": id(routing_index),
                "index_id": id(exact_index),
                "publisher_id": id(publisher),
                "publisher_thread_id": id(publisher._thread),
                "subscriber_id": id(subscriber),
                "subscriber_socket_id": id(subscriber._socket),
            }

        try:
            clear = {"type": "AllBlocksCleared", "block_hashes": []}
            for replica_id in pool.replica_ids:
                if replica_id == representative_id:
                    wire_publish(clear, epoch=None, purpose="initialization")
                else:
                    direct_apply(
                        replica_id,
                        clear,
                        epoch=None,
                        purpose="initialization",
                    )

            ready_deadline_ns = time.perf_counter_ns() + int(
                config.route_timeout_seconds * _NS_PER_SECOND
            )
            while not exact_index.all_live((representative_id,)):
                subscriber.drain()
                if time.perf_counter_ns() >= ready_deadline_ns:
                    raise RuntimeError("representative ZMQ feed did not cross its ready barrier")
                await asyncio.sleep(0.005)
            rows.append(
                {
                    "type": "stream",
                    "action": "ready",
                    "observed_ns": time.perf_counter_ns(),
                    "stream_epoch": publisher.stream_epoch,
                    "high_sequence": publisher.high_sequence,
                }
            )
            rows.append(identity_row("ready"))
            rows.append(_membership_snapshot_row(pool, phase="initial", epoch=None))
            # The synchronized ready barrier proves the SUB connection is
            # established; emit one no-op cache baseline so the live-wire
            # freshness gate never depends on PUB/SUB slow-joiner luck.
            wire_publish(clear, epoch=None, purpose="wire_live_probe")

            stop_maintenance = asyncio.Event()

            async def maintain_feeds() -> None:
                heartbeat_index = 0
                next_tick_ns = time.perf_counter_ns()
                while not stop_maintenance.is_set():
                    await _wait_until_ns(next_tick_ns)
                    started_ns = time.perf_counter_ns()
                    async with membership_lock:
                        subscriber.drain(max_events=10_000)
                        replica_ids = [
                            replica_id
                            for replica_id in pool.replica_ids
                            if replica_id != representative_id
                        ]
                        high_water = {
                            replica_id: {
                                "stream_epoch": logical_streams[replica_id][1],
                                "high_sequence": logical_streams[replica_id][2],
                            }
                            for replica_id in replica_ids
                        }
                        refreshed = []
                        for replica_id in replica_ids:
                            stream = high_water[replica_id]
                            if exact_index.heartbeat(
                                replica_id,
                                stream_epoch=str(stream["stream_epoch"]),
                                high_sequence=int(stream["high_sequence"]),
                            ):
                                refreshed.append(replica_id)
                        finished_ns = time.perf_counter_ns()
                        rows.append(
                            {
                                "type": "logical_heartbeat",
                                "sequence": heartbeat_index,
                                "scheduled_ns": next_tick_ns,
                                "started_ns": started_ns,
                                "finished_ns": finished_ns,
                                "replica_ids": replica_ids,
                                "replica_count": len(replica_ids),
                                "stream_high_water": high_water,
                                "all_refreshed": refreshed == replica_ids,
                            }
                        )
                    heartbeat_index += 1
                    next_tick_ns += int(config.heartbeat_seconds * _NS_PER_SECOND)

            maintenance_task = asyncio.create_task(maintain_feeds())
            await _wait_until_ns(
                time.perf_counter_ns() + int(config.warmup_seconds * _NS_PER_SECOND)
            )
            measurement_origin_ns = time.perf_counter_ns()
            rows[0]["measurement_origin_ns"] = measurement_origin_ns
            epoch_mutated = [asyncio.Event() for _ in range(config.churn_epochs)]

            async def execute_churn() -> None:
                nonlocal wire_sink
                for epoch, plan_row in enumerate(schedule):
                    current_epoch["value"] = epoch
                    epoch_origin_ns = measurement_origin_ns + int(
                        epoch * config.epoch_seconds * _NS_PER_SECOND
                    )
                    await _wait_until_ns(epoch_origin_ns)
                    exact_target, approximate_target = _epoch_targets(config, epoch)
                    prompt = _epoch_prompt(epoch)
                    approximate_index.observe(approximate_target, prompt)
                    rows.append(
                        {
                            "type": "prefix_seed",
                            "epoch": epoch,
                            "replica_id": approximate_target,
                            "observed_ns": time.perf_counter_ns(),
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "prefix_fingerprint": (prefix_root_fingerprint(prompt)),
                        }
                    )
                    stored = {
                        "type": "BlockStored",
                        "block_hashes": list(_epoch_block_hashes(epoch)),
                    }
                    rows.append(_membership_snapshot_row(pool, phase="epoch_start", epoch=epoch))
                    ordinals = plan_row["ordinals"]
                    if not isinstance(ordinals, list):
                        raise RuntimeError("churn ordinals must be a list")
                    spacing_ns = int(
                        config.churn_window_seconds * _NS_PER_SECOND / max(1, len(ordinals) - 1)
                    )
                    for batch_index, ordinal in enumerate(ordinals):
                        if type(ordinal) is not int:
                            raise RuntimeError("churn ordinal must be an integer")
                        scheduled_ns = epoch_origin_ns + (batch_index * spacing_ns)
                        await _wait_until_ns(scheduled_ns)
                        replica_id = _replica_id(ordinal)
                        async with membership_lock:
                            active_churn.update(
                                {
                                    "epoch": epoch,
                                    "ordinal": ordinal,
                                    "scheduled_ns": scheduled_ns,
                                }
                            )
                            started_ns = time.perf_counter_ns()
                            old_generation = pool.entry_generation(replica_id)
                            pool.drain(replica_id)
                            await pool.remove_replica(replica_id)
                            authoritative[replica_id].clear()
                            if replica_id == representative_id:
                                previous_stream_epoch = publisher.stream_epoch
                                next_stream_epoch = publisher.rotate_epoch()
                                wire_sink = publisher.event_sink()
                                rows.append(
                                    {
                                        "type": "stream",
                                        "action": "rotate",
                                        "epoch": epoch,
                                        "ordinal": ordinal,
                                        "replica_id": replica_id,
                                        "observed_ns": time.perf_counter_ns(),
                                        "previous_stream_epoch": (previous_stream_epoch),
                                        "stream_epoch": next_stream_epoch,
                                    }
                                )
                            pool.add_replica(
                                replica_id,
                                _MockReplica(replica_id, observations),
                            )
                            if replica_id == representative_id:
                                wire_publish(
                                    clear,
                                    epoch=epoch,
                                    purpose="replacement_clear",
                                )
                            else:
                                direct_apply(
                                    replica_id,
                                    clear,
                                    epoch=epoch,
                                    purpose="replacement_clear",
                                )
                            new_generation = pool.entry_generation(replica_id)
                            if replica_id == exact_target:
                                if replica_id == representative_id:
                                    wire_publish(
                                        stored,
                                        epoch=epoch,
                                        purpose="replacement_exact_seed",
                                    )
                                else:
                                    direct_apply(
                                        replica_id,
                                        stored,
                                        epoch=epoch,
                                        purpose="replacement_exact_seed",
                                    )
                            completed_ns = time.perf_counter_ns()
                            rows.append(
                                {
                                    "type": "churn",
                                    "epoch": epoch,
                                    "batch_index": batch_index,
                                    "ordinal": ordinal,
                                    "replica_id": replica_id,
                                    "scheduled_ns": scheduled_ns,
                                    "started_ns": started_ns,
                                    "completed_ns": completed_ns,
                                    "old_generation": old_generation,
                                    "new_generation": new_generation,
                                }
                            )
                            if batch_index == 0:
                                epoch_mutated[epoch].set()
                            active_churn.clear()
                    rows.append(_membership_snapshot_row(pool, phase="epoch_end", epoch=epoch))

            async def route_requests() -> None:
                interval_ns = int(_NS_PER_SECOND / config.route_rate)
                for sequence in range(config.expected_measurement_routes):
                    scheduled_ns = measurement_origin_ns + (sequence * interval_ns)
                    epoch = min(
                        config.churn_epochs - 1,
                        int((sequence / config.route_rate) // config.epoch_seconds),
                    )
                    await _wait_until_ns(scheduled_ns + 1_000_000)
                    # The first ordinal is the exact target.  Join the request
                    # to that real pool replacement instead of assuming a
                    # fixed 5 ms scheduler margin; remaining churn still
                    # overlaps route traffic.
                    await epoch_mutated[epoch].wait()
                    prompt = _epoch_prompt(epoch)
                    exact_target, approximate_target = _epoch_targets(config, epoch)
                    session_id = _session_for_approximate(epoch, exact_target, approximate_target)
                    request_id = f"f2b-route-{sequence:05d}"
                    request = GenerationRequest(
                        request_id=request_id,
                        prompt=prompt,
                        sampling_params=SamplingParams(max_tokens=1),
                        cache_hint=CacheHint(
                            session_id=session_id,
                            prefix_fingerprint=prefix_root_fingerprint(prompt),
                        ),
                        placement_started_ns=time.perf_counter_ns(),
                    )
                    async with membership_lock:
                        current_epoch["value"] = epoch
                        await pool.generate(request)
                        completed_ns = time.perf_counter_ns()
                        placement = placement_log.records.pop(request_id)
                        backend_replica = observations.pop(request_id)
                        selected_replica = str(placement["replica_id"])
                        if backend_replica != selected_replica:
                            raise RuntimeError("ReplicaPool log and mock backend disagree")
                        reason = str(placement["reason"])
                        representative_view = exact_index.view(representative_id)
                    rows.append(
                        {
                            "type": "route",
                            "sequence": sequence,
                            "epoch": epoch,
                            "scheduled_ns": scheduled_ns,
                            "selected_ns": placement["selected_at_ns"],
                            "completed_ns": completed_ns,
                            "request_id": request_id,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "prefix_fingerprint": (prefix_root_fingerprint(prompt)),
                            "session_id": session_id,
                            "selected_replica": selected_replica,
                            "backend_replica": backend_replica,
                            "reason": reason,
                            "mode": ("exact" if reason == "kv_event_match" else "approximate"),
                            "expected_exact_replica": exact_target,
                            "expected_approximate_replica": (approximate_target),
                            "pool_size": placement["pool_size"],
                            "eligible_size": placement["eligible_size"],
                            "representative_view": asdict(representative_view),
                        }
                    )
                if placement_log.records or observations:
                    raise RuntimeError("undrained placement observations")

            async def stream_chaos() -> None:
                kill_scheduled_ns = measurement_origin_ns + int(
                    config.kill_offset_seconds * _NS_PER_SECOND
                )
                await _wait_until_ns(kill_scheduled_ns)
                old_socket_id = id(subscriber._socket)
                requested_ns = time.perf_counter_ns()
                subscriber.pause()
                observed_ns = time.perf_counter_ns()
                rows.append(
                    {
                        "type": "kill",
                        "action": "pause",
                        "fault": "subscriber_live_socket_close",
                        "scheduled_ns": kill_scheduled_ns,
                        "requested_ns": requested_ns,
                        "observed_ns": observed_ns,
                        "subscriber_socket_id": old_socket_id,
                    }
                )
                rows.append(identity_row("outage"))

                restore_scheduled_ns = measurement_origin_ns + int(
                    config.restore_offset_seconds * _NS_PER_SECOND
                )
                await _wait_until_ns(restore_scheduled_ns)
                requested_ns = time.perf_counter_ns()
                subscriber.resume()
                observed_ns = time.perf_counter_ns()
                rows.append(
                    {
                        "type": "kill",
                        "action": "resume",
                        "fault": "subscriber_live_socket_reconnect",
                        "scheduled_ns": restore_scheduled_ns,
                        "requested_ns": requested_ns,
                        "observed_ns": observed_ns,
                        "subscriber_socket_id": id(subscriber._socket),
                    }
                )
                recovery_deadline_ns = observed_ns + int(
                    config.route_timeout_seconds * _NS_PER_SECOND
                )
                while not exact_index.all_live((representative_id,)):
                    if time.perf_counter_ns() >= recovery_deadline_ns:
                        raise RuntimeError("representative stream did not recover")
                    await asyncio.sleep(0.005)
                binding_recoveries = [
                    recovery
                    for recovery in subscriber.recovery_history
                    if type(recovery.get("requested_ns")) is int
                    and int(recovery["requested_ns"]) >= observed_ns
                ]
                if len(binding_recoveries) != 1:
                    raise RuntimeError("stream restore must produce exactly one binding replay")
                binding_recovery = binding_recoveries[0]
                async with membership_lock:
                    recovery_generation = pool.entry_generation(representative_id)
                    recovery_view = exact_index.view(representative_id)
                    recovery_epoch = publisher.stream_epoch
                    recovery_high_sequence = publisher.high_sequence
                    recovery_authoritative = list(publisher.authoritative_hashes())
                    recovery_index = list(recovery_view.block_hashes)
                    recovery_view_payload = asdict(recovery_view)
                    recovery_view_payload["block_hashes"] = recovery_index
                    rows.append(
                        {
                            "type": "state_snapshot",
                            "phase": "recovery",
                            "replica_id": representative_id,
                            "replica_generation": recovery_generation,
                            "stream_epoch": recovery_epoch,
                            "high_sequence": recovery_high_sequence,
                            "replay_completed_ns": binding_recovery["completed_ns"],
                            "observed_ns": time.perf_counter_ns(),
                            "index_view": recovery_view_payload,
                            "authoritative_hashes": recovery_authoritative,
                            "index_hashes": recovery_index,
                            "authoritative_digest": _sha256_json(recovery_authoritative),
                            "index_digest": _sha256_json(recovery_index),
                        }
                    )
                rows.append(
                    {
                        "type": "stream",
                        "action": "recovered",
                        "observed_ns": time.perf_counter_ns(),
                        "stream_epoch": publisher.stream_epoch,
                        "high_sequence": publisher.high_sequence,
                    }
                )
                rows.append(identity_row("restored"))

            await asyncio.gather(
                execute_churn(),
                route_requests(),
                stream_chaos(),
            )
            await _wait_until_ns(
                time.perf_counter_ns() + int(config.cooldown_seconds * _NS_PER_SECOND)
            )
            stop_maintenance.set()
            await maintenance_task
            subscriber.drain(max_events=100_000)

            for delivery in delivery_records:
                event_kind = delivery.get("event_kind")
                if event_kind == "heartbeat":
                    rows.append(
                        {
                            "type": "heartbeat",
                            "channel": "representative_zmq",
                            **delivery,
                        }
                    )
                    continue
                sequence = delivery.get("sequence")
                stream_epoch = delivery.get("stream_epoch")
                wire_key = (
                    (stream_epoch, sequence)
                    if isinstance(stream_epoch, str) and type(sequence) is int
                    else None
                )
                if wire_key is not None and wire_key in wire_events:
                    emitted_ns = delivery.get("emitted_ns")
                    if wire_events[wire_key]["emitted_ns"] is None:
                        wire_events[wire_key]["emitted_ns"] = emitted_ns
                rows.append(
                    {
                        "type": "event_receive",
                        "channel": "representative_zmq",
                        **delivery,
                    }
                )
                if delivery.get("result") in {"applied", "duplicate"}:
                    rows.append(
                        {
                            "type": "event_apply",
                            "channel": "representative_zmq",
                            **delivery,
                        }
                    )
            rows.extend(
                sorted(
                    wire_events.values(),
                    key=lambda row: int(row["enqueued_ns"]),
                )
            )

            final_authoritative = {
                replica_id: (
                    list(publisher.authoritative_hashes())
                    if replica_id == representative_id
                    else sorted(authoritative[replica_id])
                )
                for replica_id in sorted(authoritative)
            }
            final_index = {
                replica_id: list(exact_index.view(replica_id).block_hashes)
                for replica_id in sorted(authoritative)
            }
            rows.append(
                {
                    "type": "state_snapshot",
                    "phase": "final",
                    "observed_ns": time.perf_counter_ns(),
                    "authoritative_hashes_by_replica": final_authoritative,
                    "index_hashes_by_replica": final_index,
                    "authoritative_digest": _sha256_json(final_authoritative),
                    "index_digest": _sha256_json(final_index),
                }
            )
            for recovery in subscriber.recovery_history:
                recovery_epoch = recovery.get("stream_epoch")
                recovery_generations = {
                    row.get("replica_generation")
                    for row in wire_events.values()
                    if row.get("replica_id") == recovery.get("replica_id")
                    and row.get("stream_epoch") == recovery_epoch
                }
                recovery_generation = (
                    next(iter(recovery_generations)) if len(recovery_generations) == 1 else None
                )
                rows.append(
                    {
                        "type": "replay",
                        **recovery,
                        "replica_generation": recovery_generation,
                    }
                )
            rows.append(
                {
                    "type": "transport_summary",
                    "publisher_stream_epoch": publisher.stream_epoch,
                    "publisher_high_sequence": publisher.high_sequence,
                    "publisher_dropped_live_events": (publisher.dropped_live_events),
                    "delivery_count": len(delivery_records),
                    "recovery_count": len(subscriber.recovery_history),
                }
            )
            rows.append(_membership_snapshot_row(pool, phase="final", epoch=None))
        finally:
            subscriber.close()
            publisher.close()
            await pool.shutdown()
    return rows


def run_benchmark(
    output_dir: Path,
    *,
    profile: str,
) -> Path:
    config = profile_config(profile)
    source = source_descriptor()
    environment = environment_descriptor()
    rows = asyncio.run(_collect_raw_rows(config, source, environment))
    source_end = source_end_descriptor()
    run = next(row for row in rows if row.get("type") == "run")
    run["source_end"] = source_end
    for index, row in enumerate(rows):
        row["row_index"] = index
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    manifest_path = output_dir / MANIFEST_NAME
    _write_jsonl(raw_path, rows)
    metrics = derive_metrics(rows, config)
    gates = threshold_checks(rows, config)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "config": asdict(config),
        "schedule": compressed_churn_plan(config),
        "topology": run.get("topology"),
        "reused_evidence": reused_evidence_descriptor(),
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
            and _reused_f2a_digests_match()
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
    mode.add_argument("--check-current-source", type=Path)
    parser.add_argument(
        "--profile",
        choices=("formal", "smoke"),
        default="smoke",
    )
    parser.add_argument(
        "--required-profile",
        choices=("formal", "smoke"),
        help="Reject replay/source evidence from any other frozen profile",
    )
    parser.add_argument(
        "--require-current-source",
        action="store_true",
        help="Reject replay evidence unless every frozen source byte matches",
    )
    parser.add_argument(
        "--require-current-runner-context",
        action="store_true",
        help="Reject freshly measured evidence unless its GitHub context matches this runner",
    )
    parser.add_argument(
        "--actions-run-provenance",
        type=Path,
        help="Trusted GitHub Actions run API JSON required for retained evidence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    replay_policies = (
        args.required_profile is not None
        or args.require_current_source
        or args.require_current_runner_context
        or args.actions_run_provenance is not None
    )
    if args.output_dir is not None and replay_policies:
        parser.error("replay provenance policies cannot be used while measuring")
    if args.check_current_source is not None and (
        args.require_current_runner_context or args.actions_run_provenance is not None
    ):
        parser.error("run provenance policies require --verify-artifact")
    if args.check_current_source is not None:
        matches = artifact_source_matches_current(
            args.check_current_source,
            required_profile=args.required_profile,
        )
        print(
            json.dumps(
                {
                    "required_profile": args.required_profile,
                    "source_matches_current": matches,
                }
            )
        )
        return 0 if matches else 1
    if args.verify_artifact is not None:
        result = verify_artifact(
            args.verify_artifact,
            required_profile=args.required_profile,
            require_current_source=args.require_current_source,
            require_current_runner_context=(args.require_current_runner_context),
            actions_run_provenance=args.actions_run_provenance,
        )
    else:
        manifest = run_benchmark(args.output_dir, profile=args.profile)
        result = verify_artifact(manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
