#!/usr/bin/env python3
"""G5 F2c: real-engine KV-aware TTFT crossover and offline evidence verifier.

The measurement drives the production ``ReplicaPool`` and
``OpenAICompatBackend`` streaming path directly.  A candidate
``PrefixIndex`` pool and a session-HRW baseline receive byte-identical prompts
on isolated two-replica cohorts.  Cohorts cross over on every round.

Prompt bodies and generated text never enter retained evidence.  The raw JSONL
contains their SHA-256 identities, exact engine usage, production router-log
decisions, and monotonic timestamps.  ``--verify-artifact`` performs an
offline replay and recomputes all binding metrics without trusting the
manifest verdict.
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
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from kairyu.engine.backend import CacheHint, GenerationRequest, GenerationUsage
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.orchestration.prefix_index import PrefixIndex, prefix_root_fingerprint
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.router import JsonlRouterLog
from kairyu.sampling_params import SamplingParams

SCHEMA_VERSION = "kairyu.f2c.kv-aware-ttft.v1"
GATE = "G5-F2c"
RAW_NAME = "kv-aware-ttft-f2c-raw.jsonl"
MANIFEST_NAME = "kv-aware-ttft-f2c-manifest.json"
CANDIDATE_POLICY = "kv_aware"
BASELINE_POLICY = "session_hrw"
POLICIES = (CANDIDATE_POLICY, BASELINE_POLICY)
REPLICA_IDS = ("r0", "r1")
DEFAULT_SEED = 0xF2C_2026
DEFAULT_COHORT_A_ENDPOINTS = (
    "http://127.0.0.1:8100/v1",
    "http://127.0.0.1:8101/v1",
)
DEFAULT_COHORT_B_ENDPOINTS = (
    "http://127.0.0.1:8102/v1",
    "http://127.0.0.1:8103/v1",
)
DEFAULT_CONFIG_PATHS = (
    "bench/deploy/qwen3-32b-multi-gpu/f2c-compose.yaml",
    "bench/deploy/qwen3-32b-multi-gpu/f2c-replica.yaml",
    "bench/deploy/qwen3-32b-multi-gpu/f2c-stack.sh",
)
_LEGACY_CONFIG_PATHS = tuple(
    path.replace(
        "verification/fleet/performance/kv_aware_ttft_f2c_bench.py",
        "bench/kv_aware_ttft_f2c_bench.py",
    ).replace(
        "bench/deploy/qwen3-32b-multi-gpu/",
        "examples/qwen3-32b-multi-gpu/",
    )
    for path in DEFAULT_CONFIG_PATHS
)
F2C_SERVICES = ("replica-a0", "replica-a1", "replica-b0", "replica-b1")
F2C_SERVICE_GPU_IDS = {
    "replica-a0": ("0", "1"),
    "replica-a1": ("2", "3"),
    "replica-b0": ("4", "5"),
    "replica-b1": ("6", "7"),
}
F2C_SERVICE_HOST_PORTS = {
    "replica-a0": "8100",
    "replica-a1": "8101",
    "replica-b0": "8102",
    "replica-b1": "8103",
}
F2C_SERVICE_CPUSETS = {
    "replica-a0": "0-15,64-79",
    "replica-a1": "16-31,80-95",
    "replica-b0": "32-47,96-111",
    "replica-b1": "48-63,112-127",
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PATHS = (
    "verification/fleet/performance/kv_aware_ttft_f2c_bench.py",
    "Dockerfile.cuda",
    "bench/deploy/qwen3-32b-multi-gpu/f2c-compose.yaml",
    "bench/deploy/qwen3-32b-multi-gpu/f2c-replica.yaml",
    "bench/deploy/qwen3-32b-multi-gpu/f2c-stack.sh",
    "bench/results/env-2026-07-25.json",
    "bench/results/parity-tp-qwen3-32b-2026-07-26.json",
    "bench/results/hf-reference-qwen3-32b.json",
    "kairyu/orchestration/replica.py",
    "kairyu/orchestration/prefix_index.py",
    "kairyu/orchestration/router.py",
    "kairyu/engine/backend.py",
    "kairyu/engine/openai_backend.py",
    "kairyu/sampling_params.py",
    "pyproject.toml",
    "uv.lock",
)
_LEGACY_SOURCE_PATHS = tuple(
    path.replace(
        "verification/fleet/performance/kv_aware_ttft_f2c_bench.py",
        "bench/kv_aware_ttft_f2c_bench.py",
    ).replace(
        "bench/deploy/qwen3-32b-multi-gpu/",
        "examples/qwen3-32b-multi-gpu/",
    )
    for path in _SOURCE_PATHS
)
_HEX = frozenset("0123456789abcdef")
_NANOSECONDS = 1_000_000_000
_DOCUMENT_VOCABULARY = (
    "amber",
    "bridge",
    "candle",
    "delta",
    "ember",
    "forest",
    "garden",
    "harbor",
    "island",
    "jungle",
    "kernel",
    "lantern",
    "meadow",
    "needle",
    "ocean",
    "prairie",
    "quartz",
    "river",
    "silver",
    "temple",
    "umber",
    "valley",
    "willow",
    "xenon",
    "yellow",
    "zenith",
)


@dataclass(frozen=True)
class WorkloadConfig:
    profile: str
    seed: int
    rounds: int
    families_per_round: int
    document_words: int
    arrival_interval_s: float
    trace_namespace: str = "smoke"
    max_tokens: int = 8
    min_tokens: int = 8
    ignore_eos: bool = True
    temperature: float = 0.0
    request_timeout_s: float = 180.0
    ttft_slo_s: float = 60.0
    ttft_ratio_limit: float = 0.70
    goodput_ratio_floor: float = 0.99
    prompt_token_min: int = 1_500
    prompt_token_max: int = 3_000

    def validate(self) -> None:
        if self.profile not in {"smoke", "formal"}:
            raise ValueError("profile must be smoke or formal")
        if min(self.rounds, self.families_per_round, self.document_words) < 1:
            raise ValueError("round, family, and document counts must be positive")
        if not (
            1 <= len(self.trace_namespace) <= 64
            and self.trace_namespace[0].isalnum()
            and self.trace_namespace[-1].isalnum()
            and all(
                character in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in self.trace_namespace
            )
        ):
            raise ValueError(
                "trace_namespace must be a 1..64 character lower-case safe token"
            )
        if self.profile == "formal" and (
            self.seed != DEFAULT_SEED
            or self.rounds != 8
            or self.families_per_round != 16
            or self.document_words != 2_048
            or self.arrival_interval_s != 1.0
            or self.max_tokens != 8
            or self.min_tokens != 8
            or self.ignore_eos is not True
            or self.temperature != 0.0
            or self.request_timeout_s != 180.0
            or self.ttft_slo_s != 60.0
            or self.ttft_ratio_limit != 0.70
            or self.goodput_ratio_floor != 0.99
            or self.prompt_token_min != 1_500
            or self.prompt_token_max != 3_000
        ):
            raise ValueError(
                "formal F2c freezes the complete workload and gate identity"
            )
        if self.max_tokens != 8 or self.min_tokens != 8:
            raise ValueError("F2c freezes exactly eight decode tokens")
        if not self.ignore_eos or self.temperature != 0.0:
            raise ValueError("F2c requires fixed-work greedy decoding")
        if self.arrival_interval_s <= 0 or self.request_timeout_s <= 0:
            raise ValueError("durations must be positive")
        if not 0 < self.ttft_slo_s <= self.request_timeout_s:
            raise ValueError("TTFT SLO must be positive and no larger than timeout")
        if not 0 < self.ttft_ratio_limit < 1:
            raise ValueError("TTFT ratio limit must be between zero and one")
        if not 0 < self.goodput_ratio_floor <= 1:
            raise ValueError("goodput floor must be in (0, 1]")
        if not 0 < self.prompt_token_min <= self.prompt_token_max:
            raise ValueError("invalid prompt-token envelope")


@dataclass(frozen=True)
class FamilyTrace:
    round_index: int
    family_index: int
    family_id: str
    seed_replica_id: str
    measured_replica_id: str
    seed_session_id: str
    measured_session_id: str
    shared_prefix: str
    seed_prompt: str
    turn1_prompt: str
    canonical_assistant_continuation: str
    prefix_fingerprint: str

    @property
    def shared_prefix_sha256(self) -> str:
        return sha256_text(self.shared_prefix)

    @property
    def seed_prompt_sha256(self) -> str:
        return sha256_text(self.seed_prompt)

    @property
    def turn1_prompt_sha256(self) -> str:
        return sha256_text(self.turn1_prompt)

    @property
    def canonical_assistant_continuation_sha256(self) -> str:
        return sha256_text(self.canonical_assistant_continuation)

    @property
    def turn2_prompt_sha256(self) -> str:
        return sha256_text(turn2_prompt(self))

    @property
    def seed_session_sha256(self) -> str:
        return sha256_text(self.seed_session_id)

    @property
    def measured_session_sha256(self) -> str:
        return sha256_text(self.measured_session_id)


@dataclass(frozen=True)
class EndpointTopology:
    cohort_a: tuple[str, str]
    cohort_b: tuple[str, str]

    def validate(self, *, formal: bool = False) -> None:
        if len(self.cohort_a) != 2 or len(self.cohort_b) != 2:
            raise ValueError("each F2c cohort must contain exactly two endpoints")
        endpoints = (*self.cohort_a, *self.cohort_b)
        if len(set(endpoints)) != 4:
            raise ValueError("F2c requires four distinct endpoint origins")
        if any(not value.endswith("/v1") for value in endpoints):
            raise ValueError("OpenAI-compatible endpoint bases must end in /v1")
        if formal and (
            self.cohort_a != DEFAULT_COHORT_A_ENDPOINTS
            or self.cohort_b != DEFAULT_COHORT_B_ENDPOINTS
        ):
            raise ValueError(
                "formal F2c freezes localhost ports 8100..8103 so host "
                "GPU topology remains causally bound to the endpoints"
            )


@dataclass(frozen=True)
class RequestObservation:
    row: dict[str, object]
    output: str


def profile_config(
    profile: str,
    *,
    seed: int = DEFAULT_SEED,
    trace_namespace: str | None = None,
) -> WorkloadConfig:
    if profile == "formal":
        if trace_namespace is None:
            raise ValueError("formal F2c requires a fresh trace_namespace")
        config = WorkloadConfig(
            profile="formal",
            seed=seed,
            rounds=8,
            families_per_round=16,
            document_words=2_048,
            arrival_interval_s=1.0,
            trace_namespace=trace_namespace,
        )
    elif profile == "smoke":
        config = WorkloadConfig(
            profile="smoke",
            seed=seed,
            rounds=2,
            families_per_round=2,
            document_words=128,
            arrival_interval_s=0.05,
            trace_namespace=trace_namespace or "smoke",
            prompt_token_min=1,
            prompt_token_max=1_000,
        )
    else:
        raise ValueError(f"unknown profile {profile!r}")
    config.validate()
    return config


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_runtime_image(value: object) -> bool:
    """Return whether *value* is an immutable lower-case OCI digest reference."""

    if not isinstance(value, str) or any(character.isspace() for character in value):
        return False
    repository, separator, digest = value.rpartition("@sha256:")
    return (
        separator == "@sha256:"
        and bool(repository)
        and "@sha256:" not in repository
        and len(digest) == 64
        and digest == digest.lower()
        and set(digest) <= _HEX
    )


def _valid_container_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX
    )


def _valid_image_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == len("sha256:") + 64
        and set(value.removeprefix("sha256:")) <= _HEX
    )


def valid_runtime_attestation(
    value: object,
    *,
    runtime_image: str,
    expected_commit: str,
) -> bool:
    """Validate the normalized Compose/container/image causal binding."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "compose_file",
        "runtime_image",
        "services",
        "image",
    }:
        return False
    services = value["services"]
    image = value["image"]
    if not (
        value["schema_version"] == 1
        and value["compose_file"] in (DEFAULT_CONFIG_PATHS[0], _LEGACY_CONFIG_PATHS[0])
        and value["runtime_image"] == runtime_image
        and valid_runtime_image(runtime_image)
        and _valid_git_commit(expected_commit)
        and isinstance(services, list)
        and len(services) == len(F2C_SERVICES)
        and [item.get("service") for item in services if isinstance(item, dict)]
        == list(F2C_SERVICES)
        and isinstance(image, dict)
        and set(image)
        == {
            "id",
            "ref",
            "id_inspect_id",
            "ref_inspect_id",
            "revision",
        }
        and _valid_image_id(image["id"])
        and image["ref"] == runtime_image
        and image["id_inspect_id"] == image["id"]
        and image["ref_inspect_id"] == image["id"]
        and image["revision"] == expected_commit
    ):
        return False
    image_id = image["id"]
    return all(
        isinstance(item, dict)
        and set(item)
        == {
            "service",
            "container_id",
            "config_image",
            "image_id",
            "status",
            "health",
            "device_ids",
            "cpuset_cpus",
            "port",
        }
        and _valid_container_id(item["container_id"])
        and item["config_image"] == runtime_image
        and item["image_id"] == image_id
        and item["status"] == "running"
        and item["health"] == "healthy"
        and item["device_ids"] == list(F2C_SERVICE_GPU_IDS[item["service"]])
        and item["cpuset_cpus"] == F2C_SERVICE_CPUSETS[item["service"]]
        and item["port"]
        == {
            "container_port": "8000/tcp",
            "host_ip": "127.0.0.1",
            "host_port": F2C_SERVICE_HOST_PORTS[item["service"]],
        }
        for item in services
    )


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def frozen_hrw(
    session_id: str,
    replica_ids: tuple[str, ...] = REPLICA_IDS,
) -> str:
    """Byte-identical SHA-256 HRW used by production ``ReplicaPool``."""

    if not session_id or not replica_ids:
        raise ValueError("HRW requires a session and at least one replica")
    return max(
        replica_ids,
        key=lambda replica_id: hashlib.sha256(
            f"{session_id}:{replica_id}".encode()
        ).digest(),
    )


def _session_for_target(
    config: WorkloadConfig,
    family_id: str,
    role: str,
    target: str,
) -> str:
    for nonce in range(1_000_000):
        value = f"f2c:{config.seed}:{family_id}:{role}:{nonce}"
        if frozen_hrw(value) == target:
            return value
    raise RuntimeError("could not construct deterministic HRW session")


def _family_document(config: WorkloadConfig, family_id: str) -> str:
    seed_bytes = hashlib.sha256(
        (
            f"f2c-document:{config.profile}:{config.trace_namespace}:"
            f"{config.seed}:{family_id}"
        ).encode()
    ).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
    words = " ".join(
        _DOCUMENT_VOCABULARY[rng.randrange(len(_DOCUMENT_VOCABULARY))]
        for _ in range(config.document_words)
    )
    # The family identity and digest intentionally appear immediately after a
    # short /no_think prefix, so every document differs inside the first
    # 256-character PrefixIndex root.
    identity = sha256_text(f"f2c-family-root:{config.seed}:{family_id}")
    return (
        f"/no_think\nRAG family {family_id}; root {identity}.\n"
        f"Evidence document:\n{words}\n"
    )


def build_trace(config: WorkloadConfig) -> tuple[FamilyTrace, ...]:
    """Build deterministic, disjoint RAG families without retaining prompts."""

    config.validate()
    trace: list[FamilyTrace] = []
    roots: set[str] = set()
    for round_index in range(config.rounds):
        for family_index in range(config.families_per_round):
            family_id = (
                f"{config.profile}-{config.trace_namespace}-"
                f"round-{round_index:02d}-family-{family_index:02d}"
            )
            seed_replica_id = REPLICA_IDS[family_index % len(REPLICA_IDS)]
            measured_replica_id = REPLICA_IDS[
                (family_index + 1) % len(REPLICA_IDS)
            ]
            shared_prefix = _family_document(config, family_id)
            fingerprint = prefix_root_fingerprint(shared_prefix)
            if not fingerprint or fingerprint in roots:
                raise RuntimeError("RAG families require unique complete roots")
            roots.add(fingerprint)
            seed_session = _session_for_target(
                config, family_id, "seed", seed_replica_id
            )
            measured_session = _session_for_target(
                config, family_id, "measure", measured_replica_id
            )
            trace.append(
                FamilyTrace(
                    round_index=round_index,
                    family_index=family_index,
                    family_id=family_id,
                    seed_replica_id=seed_replica_id,
                    measured_replica_id=measured_replica_id,
                    seed_session_id=seed_session,
                    measured_session_id=measured_session,
                    shared_prefix=shared_prefix,
                    seed_prompt=(
                        shared_prefix
                        + "Seed query: identify the RAG family in eight tokens.\n"
                    ),
                    turn1_prompt=(
                        shared_prefix
                        + "Question one: identify the RAG family in eight tokens.\n"
                    ),
                    canonical_assistant_continuation=(
                        f"Canonical family answer {family_id} is fixed for crossover."
                    ),
                    prefix_fingerprint=fingerprint,
                )
            )
    return tuple(trace)


def turn2_prompt(family: FamilyTrace) -> str:
    return (
        family.turn1_prompt
        + "Assistant answer: "
        + family.canonical_assistant_continuation
        + "\n"
        + "Question two: confirm the same RAG family in eight tokens.\n"
    )


def trace_descriptor(
    config: WorkloadConfig,
    trace: tuple[FamilyTrace, ...] | None = None,
) -> dict[str, object]:
    trace = build_trace(config) if trace is None else trace
    rows = [
        {
            "round_index": family.round_index,
            "family_index": family.family_index,
            "family_id": family.family_id,
            "seed_replica_id": family.seed_replica_id,
            "measured_replica_id": family.measured_replica_id,
            "seed_session_sha256": family.seed_session_sha256,
            "measured_session_sha256": family.measured_session_sha256,
            "shared_prefix_sha256": family.shared_prefix_sha256,
            "seed_prompt_sha256": family.seed_prompt_sha256,
            "turn1_prompt_sha256": family.turn1_prompt_sha256,
            "canonical_assistant_continuation_sha256": (
                family.canonical_assistant_continuation_sha256
            ),
            "turn2_prompt_sha256": family.turn2_prompt_sha256,
            "prefix_fingerprint": family.prefix_fingerprint,
            "prefix_fingerprint_sha256": sha256_text(
                family.prefix_fingerprint
            ),
            "shared_prefix_chars": len(family.shared_prefix),
            "document_words": config.document_words,
        }
        for family in trace
    ]
    identity = {
        "profile": config.profile,
        "trace_namespace": config.trace_namespace,
        "families": rows,
    }
    return {
        **identity,
        "sha256": sha256_text(canonical_json(identity)),
    }


def nearest_rank_percentile(
    values: list[int | float] | tuple[int | float, ...],
    quantile: float,
) -> int | float:
    if not values:
        raise ValueError("nearest-rank percentile requires samples")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive samples")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def cohort_for(round_index: int, policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    candidate = "a" if round_index % 2 == 0 else "b"
    if policy == CANDIDATE_POLICY:
        return candidate
    return "b" if candidate == "a" else "a"


def request_id_for(
    family: FamilyTrace,
    policy: str,
    turn: int,
) -> str:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    phase = "seed" if turn == 0 else f"turn{turn}"
    return (
        f"f2c-r{family.round_index:02d}-f{family.family_index:02d}-"
        f"{policy}-{phase}"
    )


def hashed_descriptor(value: object) -> dict[str, object]:
    return {"value": value, "sha256": sha256_text(canonical_json(value))}


def _descriptor_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"value", "sha256"}
        and isinstance(value["sha256"], str)
        and value["sha256"]
        == sha256_text(canonical_json(value["value"]))
    )


def _binding_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("type") == "request" and row.get("binding") is True
    ]


def _round_policy_rows(
    binding: list[dict[str, object]],
    round_index: int,
    policy: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in binding
        if row.get("round_index") == round_index
        and row.get("policy") == policy
    ]


def _arm_metrics(
    rows: list[dict[str, object]],
    config: WorkloadConfig,
) -> dict[str, object]:
    ttfts = [int(row["ttft_ns"]) for row in rows]
    receipts = [int(row["receipt_ns"]) for row in rows]
    dones = [int(row["done_ns"]) for row in rows]
    span_ns = max(dones) - min(receipts)
    if span_ns <= 0:
        raise ValueError("arm wall interval must be positive")
    slo_ns = int(config.ttft_slo_s * _NANOSECONDS)
    slo_successes = sum(ttft <= slo_ns for ttft in ttfts)
    prompt_tokens = sum(int(row["usage"]["prompt_tokens"]) for row in rows)
    cached_tokens = sum(int(row["usage"]["cached_tokens"]) for row in rows)
    completion_tokens = sum(
        int(row["usage"]["completion_tokens"]) for row in rows
    )
    return {
        "request_count": len(rows),
        "ttft_p95_ns": int(nearest_rank_percentile(ttfts, 0.95)),
        "ttft_p95_s": nearest_rank_percentile(ttfts, 0.95)
        / _NANOSECONDS,
        "ttft_slo_ns": slo_ns,
        "ttft_slo_successes": slo_successes,
        "wall_ns": span_ns,
        "goodput_rps": slo_successes * _NANOSECONDS / span_ns,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "token_weighted_cache_rate": (
            cached_tokens / prompt_tokens if prompt_tokens else 0.0
        ),
    }


def derive_metrics(
    rows: list[dict[str, object]],
    config: WorkloadConfig,
) -> dict[str, object]:
    """Recompute every binding statistic from raw request rows."""

    binding = _binding_rows(rows)
    output_pairs: dict[tuple[int, int, int], dict[str, object]] = {}
    for row in binding:
        key = (
            int(row["round_index"]),
            int(row["family_index"]),
            int(row["turn"]),
        )
        output_pairs.setdefault(key, {})[str(row["policy"])] = row.get(
            "output_sha256"
        )
    complete_output_pairs = [
        pair for pair in output_pairs.values() if set(pair) == set(POLICIES)
    ]
    output_match_total = len(complete_output_pairs)
    output_match_count = sum(
        pair[CANDIDATE_POLICY] == pair[BASELINE_POLICY]
        for pair in complete_output_pairs
    )
    failures = [
        row
        for row in rows
        if row.get("type") == "request" and row.get("status") != "success"
    ]
    diagnostics = {
        "max_paired_receipt_skew_ns": max(
            (
                int(row.get("paired_receipt_skew_ns", 0))
                for row in binding
            ),
            default=0,
        ),
        "max_schedule_lateness_ns": max(
            (int(row.get("schedule_lateness_ns", 0)) for row in binding),
            default=0,
        ),
        "timing_fields_are_diagnostic_only": True,
        "output_match_count": output_match_count,
        "output_match_total": output_match_total,
        "output_match_rate": (
            output_match_count / output_match_total
            if output_match_total
            else 0.0
        ),
        "output_match_is_diagnostic_only": True,
    }
    if failures:
        return {
            "computable": False,
            "request_count": len(binding),
            "failure_count": len(failures),
            "diagnostics": diagnostics,
            "rounds": [],
        }

    expected_binding = config.rounds * config.families_per_round * 2 * 2
    if len(binding) != expected_binding:
        raise ValueError(
            f"binding request count {len(binding)} != {expected_binding}"
        )
    rounds: list[dict[str, object]] = []
    ttft_ratios: list[float] = []
    goodput_ratios: list[float] = []
    candidate_prompt = candidate_cached = 0
    baseline_prompt = baseline_cached = 0
    candidate_slo = baseline_slo = 0
    candidate_wall_sum = baseline_wall_sum = 0
    for round_index in range(config.rounds):
        candidate = _arm_metrics(
            _round_policy_rows(binding, round_index, CANDIDATE_POLICY),
            config,
        )
        baseline = _arm_metrics(
            _round_policy_rows(binding, round_index, BASELINE_POLICY),
            config,
        )
        ttft_ratio = (
            float(candidate["ttft_p95_ns"])
            / float(baseline["ttft_p95_ns"])
        )
        goodput_ratio = (
            float(candidate["goodput_rps"])
            / float(baseline["goodput_rps"])
        )
        ttft_ratios.append(ttft_ratio)
        goodput_ratios.append(goodput_ratio)
        candidate_prompt += int(candidate["prompt_tokens"])
        candidate_cached += int(candidate["cached_tokens"])
        baseline_prompt += int(baseline["prompt_tokens"])
        baseline_cached += int(baseline["cached_tokens"])
        candidate_slo += int(candidate["ttft_slo_successes"])
        baseline_slo += int(baseline["ttft_slo_successes"])
        candidate_wall_sum += int(candidate["wall_ns"])
        baseline_wall_sum += int(baseline["wall_ns"])
        rounds.append(
            {
                "round_index": round_index,
                "candidate_cohort": cohort_for(
                    round_index, CANDIDATE_POLICY
                ),
                "baseline_cohort": cohort_for(
                    round_index, BASELINE_POLICY
                ),
                "candidate": candidate,
                "baseline": baseline,
                "candidate_over_baseline_ttft_p95_ratio": ttft_ratio,
                "candidate_over_baseline_goodput_ratio": goodput_ratio,
                "cache_rate_guard": (
                    float(candidate["token_weighted_cache_rate"])
                    >= float(baseline["token_weighted_cache_rate"])
                ),
            }
        )

    candidate_rows = [
        row for row in binding if row.get("policy") == CANDIDATE_POLICY
    ]
    baseline_rows = [
        row for row in binding if row.get("policy") == BASELINE_POLICY
    ]
    candidate_pooled_p95 = int(
        nearest_rank_percentile(
            [int(row["ttft_ns"]) for row in candidate_rows], 0.95
        )
    )
    baseline_pooled_p95 = int(
        nearest_rank_percentile(
            [int(row["ttft_ns"]) for row in baseline_rows], 0.95
        )
    )
    upper_rank = max(1, config.rounds - 1)
    goodput_lower_rank = min(2, config.rounds)
    upper_order_coverage = sum(
        math.comb(config.rounds, count)
        for count in range(upper_rank)
    ) / (2**config.rounds)
    candidate_cache_rate = candidate_cached / candidate_prompt
    baseline_cache_rate = baseline_cached / baseline_prompt
    candidate_pooled_goodput = (
        candidate_slo * _NANOSECONDS / candidate_wall_sum
    )
    baseline_pooled_goodput = (
        baseline_slo * _NANOSECONDS / baseline_wall_sum
    )
    return {
        "computable": True,
        "request_count": len(binding),
        "failure_count": 0,
        "ttft": {
            "ratio_definition": "candidate nearest-rank p95 / baseline nearest-rank p95",
            "round_ratios": ttft_ratios,
            "round_count": len(ttft_ratios),
            "upper_order_rank": upper_rank,
            "upper_order_achieved_coverage": upper_order_coverage,
            "upper_order_ratio": sorted(ttft_ratios)[upper_rank - 1],
            "geometric_mean_ratio": geometric_mean(ttft_ratios),
            "candidate_pooled_p95_ns": candidate_pooled_p95,
            "baseline_pooled_p95_ns": baseline_pooled_p95,
            "pooled_ratio": candidate_pooled_p95 / baseline_pooled_p95,
        },
        "goodput": {
            "definition": (
                "requests with TTFT <= frozen SLO divided by first-receipt "
                "to last-terminal wall; every planned request remains in scope"
            ),
            "round_ratios": goodput_ratios,
            "round_count": len(goodput_ratios),
            "lower_order_rank": goodput_lower_rank,
            "lower_order_achieved_coverage": upper_order_coverage,
            "lower_order_ratio": sorted(goodput_ratios)[
                goodput_lower_rank - 1
            ],
            "geometric_mean_ratio": geometric_mean(goodput_ratios),
            "candidate_pooled_rps": candidate_pooled_goodput,
            "baseline_pooled_rps": baseline_pooled_goodput,
            "pooled_ratio": (
                candidate_pooled_goodput / baseline_pooled_goodput
            ),
        },
        "cache": {
            "definition": "sum(engine cached_tokens) / sum(engine prompt_tokens)",
            "candidate_prompt_tokens": candidate_prompt,
            "candidate_cached_tokens": candidate_cached,
            "candidate_token_weighted_rate": candidate_cache_rate,
            "baseline_prompt_tokens": baseline_prompt,
            "baseline_cached_tokens": baseline_cached,
            "baseline_token_weighted_rate": baseline_cache_rate,
            "candidate_strictly_above_baseline": (
                candidate_cache_rate > baseline_cache_rate
            ),
            "all_round_lower_guards": all(
                bool(row["cache_rate_guard"]) for row in rounds
            ),
        },
        "rounds": rounds,
        "diagnostics": diagnostics,
    }


def threshold_checks(
    metrics: dict[str, object],
    config: WorkloadConfig,
) -> dict[str, bool]:
    if config.profile == "smoke":
        return {}
    if metrics.get("computable") is not True:
        return {
            "zero_failures": False,
            "ttft_upper_order_at_most_0_70": False,
            "ttft_geometric_mean_at_most_0_70": False,
            "ttft_pooled_at_most_0_70": False,
            "goodput_lower_order_at_least_0_99": False,
            "goodput_geometric_mean_at_least_0_99": False,
            "goodput_pooled_at_least_0_99": False,
            "engine_cache_rate_strictly_improved": False,
            "engine_cache_rate_every_round_noninferior": False,
        }
    ttft = metrics["ttft"]
    goodput = metrics["goodput"]
    cache = metrics["cache"]
    assert isinstance(ttft, dict)
    assert isinstance(goodput, dict)
    assert isinstance(cache, dict)
    return {
        "zero_failures": metrics["failure_count"] == 0,
        "ttft_upper_order_at_most_0_70": (
            float(ttft["upper_order_ratio"]) <= config.ttft_ratio_limit
        ),
        "ttft_geometric_mean_at_most_0_70": (
            float(ttft["geometric_mean_ratio"]) <= config.ttft_ratio_limit
        ),
        "ttft_pooled_at_most_0_70": (
            float(ttft["pooled_ratio"]) <= config.ttft_ratio_limit
        ),
        "goodput_lower_order_at_least_0_99": (
            float(goodput["lower_order_ratio"])
            >= config.goodput_ratio_floor
        ),
        "goodput_geometric_mean_at_least_0_99": (
            float(goodput["geometric_mean_ratio"])
            >= config.goodput_ratio_floor
        ),
        "goodput_pooled_at_least_0_99": (
            float(goodput["pooled_ratio"]) >= config.goodput_ratio_floor
        ),
        "engine_cache_rate_strictly_improved": (
            cache["candidate_strictly_above_baseline"] is True
        ),
        "engine_cache_rate_every_round_noninferior": (
            cache["all_round_lower_guards"] is True
        ),
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.lower()) <= _HEX
    )


def _contains_forbidden_body_key(value: object) -> bool:
    if isinstance(value, dict):
        if any(key in {"prompt", "output", "shared_prefix"} for key in value):
            return True
        return any(_contains_forbidden_body_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_body_key(item) for item in value)
    return False


def _valid_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and set(value.lower()) <= _HEX
    )


def _valid_repo_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(_REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"configuration file is outside the repository: {resolved}"
        ) from error
    return relative.as_posix()


def _valid_source_value(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "commit",
        "expected_commit",
        "expected_commit_matches",
        "git_clean",
        "git_status",
        "files",
    }:
        return False
    expected_commit = value["expected_commit"]
    if not (
        _valid_git_commit(value["commit"])
        and (
            expected_commit is None
            or _valid_git_commit(expected_commit)
        )
        and type(value["expected_commit_matches"]) is bool
        and value["expected_commit_matches"]
        is (
            expected_commit is None
            or expected_commit == value["commit"]
        )
        and type(value["git_clean"]) is bool
        and isinstance(value["git_status"], list)
        and all(isinstance(line, str) for line in value["git_status"])
        and value["git_clean"] is (not value["git_status"])
    ):
        return False
    files = value["files"]
    files_valid = (
        isinstance(files, list)
        and [item.get("path") for item in files if isinstance(item, dict)]
        in (list(_SOURCE_PATHS), list(_LEGACY_SOURCE_PATHS))
        and all(
            isinstance(item, dict)
            and set(item) == {"path", "sha256"}
            and _valid_sha256(item["sha256"])
            for item in files
        )
    )
    return files_valid


def _valid_configuration_value(
    value: object,
    config: WorkloadConfig,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "workload",
        "cohorts",
        "model",
        "model_revision",
        "model_digests",
        "runtime_image",
        "runtime_attestation",
        "files",
    }:
        return False
    cohorts = value["cohorts"]
    if not (
        value["workload"] == asdict(config)
        and isinstance(cohorts, dict)
        and set(cohorts) == {"a", "b"}
        and all(
            isinstance(cohorts[name], list)
            and len(cohorts[name]) == 2
            and all(
                isinstance(endpoint, str) and endpoint.endswith("/v1")
                for endpoint in cohorts[name]
            )
            for name in ("a", "b")
        )
        and len(set((*cohorts["a"], *cohorts["b"]))) == 4
        and isinstance(value["model"], str)
        and bool(value["model"])
        and (
            value["model_revision"] is None
            or isinstance(value["model_revision"], str)
        )
        and isinstance(value["model_digests"], dict)
        and all(
            isinstance(label, str)
            and bool(label)
            and _valid_sha256(digest)
            for label, digest in value["model_digests"].items()
        )
        and (
            value["runtime_image"] is None
            or isinstance(value["runtime_image"], str)
        )
        and (
            value["runtime_attestation"] is None
            or isinstance(value["runtime_attestation"], dict)
        )
    ):
        return False
    files = value["files"]
    files_valid = (
        isinstance(files, list)
        and bool(files)
        and len(
            {
                item.get("path")
                for item in files
                if isinstance(item, dict)
            }
        )
        == len(files)
        and all(
            isinstance(item, dict)
            and set(item) == {"path", "sha256"}
            and _valid_repo_relative_path(item["path"])
            and _valid_sha256(item["sha256"])
            for item in files
        )
    )
    if not files_valid:
        return False
    if config.profile == "formal":
        return (
            value["model"] == "qwen3-32b"
            and cohorts
            == {
                "a": list(DEFAULT_COHORT_A_ENDPOINTS),
                "b": list(DEFAULT_COHORT_B_ENDPOINTS),
            }
            and _valid_git_commit(value["model_revision"])
            and [item["path"] for item in files]
            in (list(DEFAULT_CONFIG_PATHS), list(_LEGACY_CONFIG_PATHS))
        )
    return True


def _valid_environment_value(value: object) -> bool:
    github_names = {
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_EVENT_NAME",
        "GITHUB_REPOSITORY",
        "GITHUB_SHA",
    }
    if not isinstance(value, dict) or set(value) != {
        "platform",
        "python",
        "python_executable",
        "cpu_count",
        "sched_affinity",
        "cuda_visible_devices",
        "github",
        "nvidia_smi_gpus",
        "nvidia_smi_topology",
        "runtime_attestation",
    }:
        return False
    github = value["github"]
    affinity = value["sched_affinity"]
    return (
        isinstance(value["platform"], str)
        and bool(value["platform"])
        and isinstance(value["python"], str)
        and bool(value["python"])
        and isinstance(value["python_executable"], str)
        and bool(value["python_executable"])
        and (value["cpu_count"] is None or type(value["cpu_count"]) is int)
        and (
            affinity is None
            or (
                isinstance(affinity, list)
                and all(type(cpu) is int and cpu >= 0 for cpu in affinity)
            )
        )
        and (
            value["cuda_visible_devices"] is None
            or isinstance(value["cuda_visible_devices"], str)
        )
        and isinstance(github, dict)
        and set(github) == github_names
        and all(item is None or isinstance(item, str) for item in github.values())
        and isinstance(value["nvidia_smi_gpus"], list)
        and all(isinstance(line, str) for line in value["nvidia_smi_gpus"])
        and isinstance(value["nvidia_smi_topology"], list)
        and all(
            isinstance(line, str) for line in value["nvidia_smi_topology"]
        )
        and (
            value["runtime_attestation"] is None
            or isinstance(value["runtime_attestation"], dict)
        )
    )


def _expected_request_specs(
    config: WorkloadConfig,
) -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    for family in build_trace(config):
        for policy in POLICIES:
            for turn in (0, 1, 2):
                request_id = request_id_for(family, policy, turn)
                session_sha256 = (
                    family.seed_session_sha256
                    if turn == 0
                    else family.measured_session_sha256
                )
                selected = (
                    family.seed_replica_id
                    if turn == 0 or policy == CANDIDATE_POLICY
                    else family.measured_replica_id
                )
                reason = (
                    "prefix_match"
                    if turn > 0 and policy == CANDIDATE_POLICY
                    else "session_affinity"
                )
                specs[request_id] = {
                    "request_id": request_id,
                    "round_index": family.round_index,
                    "family_index": family.family_index,
                    "family_id": family.family_id,
                    "policy": policy,
                    "cohort": cohort_for(family.round_index, policy),
                    "turn": turn,
                    "phase": "seed" if turn == 0 else "measure",
                    "binding": turn > 0,
                    "session_sha256": session_sha256,
                    "prefix_fingerprint": family.prefix_fingerprint,
                    "expected_prompt_sha256": (
                        family.seed_prompt_sha256
                        if turn == 0
                        else family.turn1_prompt_sha256
                        if turn == 1
                        else family.turn2_prompt_sha256
                    ),
                    "canonical_assistant_continuation_sha256": (
                        family.canonical_assistant_continuation_sha256
                    ),
                    "turn2_prompt_sha256": family.turn2_prompt_sha256,
                    "selected_replica_id": selected,
                    "reason": reason,
                }
    return specs


def replay_checks(
    rows: list[dict[str, object]],
    config: WorkloadConfig,
) -> dict[str, bool]:
    """Validate raw identity/joins and replay expected policy evolution."""

    if len(rows) < 2 or rows[0].get("type") != "run":
        raise ValueError("raw evidence must start with one run row")
    if rows[-1].get("type") != "run_end":
        raise ValueError("raw evidence must end with one run_end row")
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise ValueError("raw row indices are not contiguous")
    if any(
        row.get("type") not in {"run", "request", "run_end"}
        for row in rows
    ):
        raise ValueError("raw evidence contains an unknown record type")
    if _contains_forbidden_body_key(rows):
        raise ValueError("raw evidence contains a prompt/output body")

    run = rows[0]
    run_end = rows[-1]
    if run.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("raw schema version mismatch")
    if run.get("gate") != GATE:
        raise ValueError("raw gate mismatch")
    try:
        raw_config = WorkloadConfig(**run["config"])
    except (KeyError, TypeError) as error:
        raise ValueError("invalid raw workload config") from error
    raw_config.validate()
    if raw_config != config:
        raise ValueError("raw workload config mismatch")
    expected_trace = trace_descriptor(config)
    if run.get("trace") != expected_trace:
        raise ValueError("deterministic trace identity mismatch")

    start = run.get("provenance_start")
    end = run_end.get("provenance_end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise ValueError("missing start/end provenance")
    if set(start) != {"source", "configuration", "environment"}:
        raise ValueError("invalid start provenance categories")
    if set(end) != set(start):
        raise ValueError("invalid end provenance categories")
    if not all(_descriptor_valid(start[name]) for name in start):
        raise ValueError("invalid start provenance hash")
    if not all(_descriptor_valid(end[name]) for name in end):
        raise ValueError("invalid end provenance hash")
    if not _valid_source_value(start["source"]["value"]) or not _valid_source_value(
        end["source"]["value"]
    ):
        raise ValueError("invalid source provenance schema or frozen file set")
    if not _valid_configuration_value(
        start["configuration"]["value"], config
    ) or not _valid_configuration_value(
        end["configuration"]["value"], config
    ):
        raise ValueError("invalid configuration provenance schema")
    if not _valid_environment_value(
        start["environment"]["value"]
    ) or not _valid_environment_value(end["environment"]["value"]):
        raise ValueError("invalid environment provenance schema")

    request_rows = [row for row in rows if row.get("type") == "request"]
    if (
        type(run_end.get("request_count")) is not int
        or run_end["request_count"] != len(request_rows)
    ):
        raise ValueError("run_end request_count does not match raw requests")
    by_id: dict[str, dict[str, object]] = {}
    for row in request_rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or request_id in by_id:
            raise ValueError("request IDs must be unique strings")
        by_id[request_id] = row
    expected = _expected_request_specs(config)
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        extra = sorted(set(by_id) - set(expected))
        raise ValueError(f"request set mismatch: missing={missing}, extra={extra}")

    structural_keys = (
        "request_id",
        "round_index",
        "family_index",
        "family_id",
        "policy",
        "cohort",
        "turn",
        "phase",
        "binding",
        "session_sha256",
        "prefix_fingerprint",
    )
    route_reasons_exact = True
    selected_replicas_exact = True
    timestamp_chains_valid = True
    usage_exact = True
    fixed_decode_work = True
    prompt_token_envelope = True
    all_terminal = True
    zero_failures = True
    router_decisions_exact = True
    successful_error_types_clear = True
    fixed_length_finishes = True
    for request_id, spec in expected.items():
        row = by_id[request_id]
        if any(row.get(key) != spec[key] for key in structural_keys):
            raise ValueError(f"trace mismatch for {request_id}")
        expected_prompt = spec["expected_prompt_sha256"]
        if expected_prompt is not None and row.get("prompt_sha256") != expected_prompt:
            raise ValueError(f"prompt identity mismatch for {request_id}")
        if not _valid_sha256(row.get("prompt_sha256")):
            raise ValueError(f"invalid prompt hash for {request_id}")
        if not _valid_sha256(row.get("output_sha256")):
            raise ValueError(f"invalid output hash for {request_id}")
        if row.get("terminal") is not True:
            all_terminal = False
        if row.get("status") != "success":
            zero_failures = False
        route_reasons_exact &= row.get("reason") == spec["reason"]
        expected_ordinal = REPLICA_IDS.index(
            str(spec["selected_replica_id"])
        )
        selected_replicas_exact &= (
            row.get("selected_replica_id")
            == spec["selected_replica_id"]
            and row.get("replica_ordinal") == expected_ordinal
        )
        decision = row.get("router_decision")
        router_decisions_exact &= (
            isinstance(decision, dict)
            and set(decision)
            == {
                "kind",
                "session_sha256",
                "replica",
                "reason",
                "replica_id",
                "replica_generation",
                "placement_latency_ns",
                "request_id",
                "placement_started_ns",
                "selected_at_ns",
                "pool_size",
                "eligible_size",
            }
            and decision.get("kind") == "replica"
            and decision.get("request_id") == request_id
            and decision.get("session_sha256") == row.get("session_sha256")
            and decision.get("replica_id")
            == row.get("selected_replica_id")
            and decision.get("replica") == row.get("replica_ordinal")
            and isinstance(decision.get("replica_generation"), str)
            and bool(decision.get("replica_generation"))
            and decision.get("replica_generation")
            == row.get("replica_generation")
            and decision.get("reason") == row.get("reason")
            and decision.get("placement_started_ns")
            == row.get("receipt_ns")
            and decision.get("selected_at_ns")
            == row.get("selected_at_ns")
            and type(decision.get("placement_latency_ns")) is int
            and type(decision.get("placement_started_ns")) is int
            and type(decision.get("selected_at_ns")) is int
            and int(decision.get("placement_latency_ns", -1)) >= 0
            and decision.get("placement_latency_ns")
            == int(decision.get("selected_at_ns", 0))
            - int(decision.get("placement_started_ns", 0))
            and decision.get("pool_size") == 2
            and decision.get("eligible_size") == 2
        )
        if row.get("status") == "success":
            successful_error_types_clear &= row.get("error_type") is None
            fixed_length_finishes &= row.get("finish_reason") == "length"
            timing = tuple(
                row.get(key)
                for key in (
                    "planned_ns",
                    "receipt_ns",
                    "selected_at_ns",
                    "first_content_ns",
                    "done_ns",
                )
            )
            timestamp_chains_valid &= all(
                isinstance(value, int) for value in timing
            ) and all(
                int(left) <= int(right)
                for left, right in zip(timing[:-1], timing[1:], strict=True)
            )
            if all(isinstance(value, int) for value in timing):
                timestamp_chains_valid &= (
                    row.get("ttft_ns")
                    == int(row["first_content_ns"]) - int(row["receipt_ns"])
                    and row.get("schedule_lateness_ns")
                    == max(
                        0,
                        int(row["receipt_ns"]) - int(row["planned_ns"]),
                    )
                )
            usage = row.get("usage")
            usage_valid = (
                isinstance(usage, dict)
                and set(usage)
                == {"prompt_tokens", "cached_tokens", "completion_tokens"}
                and all(
                    type(usage.get(key)) is int and int(usage[key]) >= 0
                    for key in usage
                )
                and int(usage["cached_tokens"])
                <= int(usage["prompt_tokens"])
            )
            usage_exact &= usage_valid
            if usage_valid:
                fixed_decode_work &= (
                    int(usage["completion_tokens"]) == config.max_tokens
                )
                if row.get("binding") and row.get("turn") == 1:
                    prompt_token_envelope &= (
                        config.prompt_token_min
                        <= int(usage["prompt_tokens"])
                        <= config.prompt_token_max
                    )

    paired_prompt_and_work_equal = True
    frozen_turn2_transcript_exact = True
    paired_skews_exact = True
    for family in build_trace(config):
        for turn in (0, 1, 2):
            candidate = by_id[
                request_id_for(family, CANDIDATE_POLICY, turn)
            ]
            baseline = by_id[
                request_id_for(family, BASELINE_POLICY, turn)
            ]
            candidate_usage = candidate.get("usage")
            baseline_usage = baseline.get("usage")
            paired_prompt_and_work_equal &= (
                candidate.get("prompt_sha256")
                == baseline.get("prompt_sha256")
                and candidate.get("prompt_chars")
                == baseline.get("prompt_chars")
                and isinstance(candidate_usage, dict)
                and isinstance(baseline_usage, dict)
                and candidate_usage.get("prompt_tokens")
                == baseline_usage.get("prompt_tokens")
                and candidate_usage.get("completion_tokens")
                == baseline_usage.get("completion_tokens")
                == config.max_tokens
            )
            if turn == 2:
                frozen_turn2_prompt = turn2_prompt(family)
                frozen_turn2_transcript_exact &= (
                    candidate.get("prompt_sha256")
                    == baseline.get("prompt_sha256")
                    == family.turn2_prompt_sha256
                    and candidate.get("prompt_chars")
                    == baseline.get("prompt_chars")
                    == len(frozen_turn2_prompt)
                )
            if all(
                isinstance(row.get("receipt_ns"), int)
                for row in (candidate, baseline)
            ):
                skew = abs(
                    int(candidate["receipt_ns"])
                    - int(baseline["receipt_ns"])
                )
                paired_skews_exact &= all(
                    row.get("paired_receipt_skew_ns") == skew
                    for row in (candidate, baseline)
                )
            else:
                paired_skews_exact = False

    source_value = start["source"]["value"]
    source_clean = (
        isinstance(source_value, dict)
        and source_value.get("git_clean") is True
    )
    source_pinned = (
        isinstance(source_value, dict)
        and isinstance(source_value.get("expected_commit"), str)
        and bool(source_value["expected_commit"])
        and source_value.get("expected_commit_matches") is True
        and source_value.get("commit") == source_value.get("expected_commit")
    )
    model_revision = run.get("model_revision")
    model_digests = run.get("model_digests")
    runtime_image = run.get("runtime_image")
    model_pinned = (
        _valid_git_commit(model_revision)
        and isinstance(model_digests, dict)
        and bool(model_digests)
        and all(
            isinstance(label, str)
            and bool(label)
            and _valid_sha256(digest)
            for label, digest in model_digests.items()
        )
    )
    configuration_value = start["configuration"]["value"]
    environment_value = start["environment"]["value"]
    topology = run.get("topology")
    configured_cohorts = (
        configuration_value.get("cohorts")
        if isinstance(configuration_value, dict)
        else None
    )
    configured_endpoints = (
        [
            *configured_cohorts.get("a", ()),
            *configured_cohorts.get("b", ()),
        ]
        if isinstance(configured_cohorts, dict)
        else []
    )
    topology_exact = (
        isinstance(topology, dict)
        and topology.get("logical_replica_ids") == list(REPLICA_IDS)
        and topology.get("cohorts") == configured_cohorts
        and topology.get("physical_endpoints") == configured_endpoints
        and len(set(configured_endpoints)) == 4
        and topology.get("round_crossover")
        == [
            {
                "round_index": round_index,
                CANDIDATE_POLICY: cohort_for(
                    round_index, CANDIDATE_POLICY
                ),
                BASELINE_POLICY: cohort_for(
                    round_index, BASELINE_POLICY
                ),
            }
            for round_index in range(config.rounds)
        ]
        and isinstance(environment_value, dict)
        and topology.get("nvidia_smi_topology")
        == environment_value.get("nvidia_smi_topology")
        and topology.get("nvidia_smi_gpus")
        == environment_value.get("nvidia_smi_gpus")
        and bool(topology.get("nvidia_smi_topology"))
        and len(topology.get("nvidia_smi_gpus", [])) == 8
        and [
            line.split(",", 1)[0].strip()
            for line in topology.get("nvidia_smi_gpus", [])
            if isinstance(line, str)
        ]
        == [str(index) for index in range(8)]
        and (
            config.profile != "formal"
            or (
                configured_cohorts
                == {
                    "a": list(DEFAULT_COHORT_A_ENDPOINTS),
                    "b": list(DEFAULT_COHORT_B_ENDPOINTS),
                }
                and topology.get("physical_endpoints")
                == [
                    *DEFAULT_COHORT_A_ENDPOINTS,
                    *DEFAULT_COHORT_B_ENDPOINTS,
                ]
            )
        )
    )
    configuration_pinned = (
        _valid_configuration_value(configuration_value, config)
        and configuration_value.get("model") == run.get("model")
        and configuration_value.get("model_revision") == model_revision
        and configuration_value.get("model_digests") == model_digests
        and configuration_value.get("runtime_image") == runtime_image
    )
    runtime_image_pinned = (
        valid_runtime_image(runtime_image)
        and isinstance(configuration_value, dict)
        and configuration_value.get("runtime_image") == runtime_image
    )
    runtime_attestation = (
        configuration_value.get("runtime_attestation")
        if isinstance(configuration_value, dict)
        else None
    )
    runtime_attestation_pinned = (
        isinstance(runtime_image, str)
        and isinstance(source_value, dict)
        and isinstance(source_value.get("expected_commit"), str)
        and valid_runtime_attestation(
            runtime_attestation,
            runtime_image=runtime_image,
            expected_commit=source_value["expected_commit"],
        )
        and isinstance(environment_value, dict)
        and environment_value.get("runtime_attestation")
        == runtime_attestation
    )
    roots = [
        family.prefix_fingerprint for family in build_trace(config)
    ]
    return {
        "all_planned_requests_present_once": True,
        "all_requests_terminal": all_terminal,
        "zero_failures": zero_failures,
        "timestamp_chains_valid": timestamp_chains_valid,
        "paired_receipt_skews_recomputed": paired_skews_exact,
        "production_reason_guards_exact": route_reasons_exact,
        "production_selected_replicas_exact": selected_replicas_exact,
        "embedded_router_decisions_exact": router_decisions_exact,
        "exact_engine_usage_present": usage_exact,
        "fixed_eight_token_decode_work": fixed_decode_work,
        "successful_rows_have_no_error_type": successful_error_types_clear,
        "fixed_decode_finishes_by_length": fixed_length_finishes,
        "formal_prompt_token_envelope": prompt_token_envelope,
        "paired_prompts_and_exact_work_identical": (
            paired_prompt_and_work_equal
        ),
        "turn2_uses_frozen_common_transcript": (
            frozen_turn2_transcript_exact
        ),
        "rag_roots_are_unique_and_complete": (
            all(bool(root) for root in roots) and len(set(roots)) == len(roots)
        ),
        "source_and_environment_unchanged": start == end,
        "formal_source_clean": source_clean or config.profile != "formal",
        "formal_source_commit_pinned": (
            source_pinned or config.profile != "formal"
        ),
        "formal_model_revision_and_digests_pinned": (
            model_pinned or config.profile != "formal"
        ),
        "formal_exact_configuration_pinned": (
            configuration_pinned or config.profile != "formal"
        ),
        "formal_runtime_image_digest_pinned": (
            runtime_image_pinned or config.profile != "formal"
        ),
        "formal_runtime_container_attestation": (
            runtime_attestation_pinned or config.profile != "formal"
        ),
        "eight_gpu_four_endpoint_topology_recorded": (
            topology_exact or config.profile != "formal"
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
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


def assemble_manifest(
    rows: list[dict[str, object]],
    raw_sha256: str,
) -> dict[str, object]:
    if not _valid_sha256(raw_sha256):
        raise ValueError("manifest requires a raw SHA-256")
    try:
        config = WorkloadConfig(**rows[0]["config"])
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("raw run config missing") from error
    config.validate()
    replay = replay_checks(rows, config)
    metrics = derive_metrics(rows, config)
    checks = {**replay, **threshold_checks(metrics, config)}
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "config": asdict(config),
        "trace_sha256": rows[0]["trace"]["sha256"],
        "raw": {
            "name": RAW_NAME,
            "sha256": raw_sha256,
            "row_count": len(rows),
        },
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_artifact(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Write canonical raw evidence and its wholly-derived manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        row["row_index"] = index
    raw_path = output_dir / RAW_NAME
    _write_jsonl(raw_path, rows)
    manifest = assemble_manifest(rows, sha256_file(raw_path))
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _artifact_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / RAW_NAME, path / MANIFEST_NAME
    if path.name == RAW_NAME:
        return path, path.with_name(MANIFEST_NAME)
    if path.name == MANIFEST_NAME:
        return path.with_name(RAW_NAME), path
    raise ValueError(
        f"artifact path must be a directory, {RAW_NAME}, or {MANIFEST_NAME}"
    )


def verify_artifact(
    path: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    """Independently hash, replay, and recompute a retained F2c artifact."""

    raw_path, manifest_path = _artifact_paths(path)
    rows = _read_jsonl(raw_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    raw_sha256 = sha256_file(raw_path)
    raw_descriptor = manifest.get("raw")
    if not isinstance(raw_descriptor, dict):
        raise ValueError("manifest raw descriptor missing")
    if raw_descriptor.get("sha256") != raw_sha256:
        raise ValueError("raw SHA-256 does not match manifest")
    recomputed = assemble_manifest(rows, raw_sha256)
    if manifest != recomputed:
        raise ValueError("manifest differs from independent raw replay")
    if assert_gate and recomputed["passed"] is not True:
        raise RuntimeError("F2c artifact does not pass every gate")
    return recomputed


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_snapshot(expected_commit: str | None) -> dict[str, object]:
    status = _git_output(
        "status", "--porcelain", "--untracked-files=all"
    ).splitlines()
    relevant_status = [
        line
        for line in status
        if not (
            line.startswith("?? .claude/worktrees/")
            or line == "?? .claude/worktrees"
        )
    ]
    commit = _git_output("rev-parse", "HEAD")
    return {
        "commit": commit,
        "expected_commit": expected_commit,
        "expected_commit_matches": (
            expected_commit is None or expected_commit == commit
        ),
        "git_clean": not relevant_status,
        "git_status": relevant_status,
        "files": [
            {
                "path": relative_path,
                "sha256": sha256_file(_REPO_ROOT / relative_path),
            }
            for relative_path in _SOURCE_PATHS
        ],
    }


def _docker_json(
    command: tuple[str, ...],
    *,
    environment: dict[str, str],
) -> object:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def capture_runtime_attestation(
    *,
    runtime_image: str,
    expected_commit: str,
) -> dict[str, object]:
    """Resolve the four live Compose services to one immutable source image."""

    compose_path = (_REPO_ROOT / DEFAULT_CONFIG_PATHS[0]).resolve()
    environment = dict(os.environ)
    environment["KAIRYU_F2C_IMAGE"] = runtime_image
    ps = subprocess.run(
        (
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "ps",
            "-q",
            *F2C_SERVICES,
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    container_ids = tuple(line for line in ps.stdout.splitlines() if line)
    if len(container_ids) != len(F2C_SERVICES) or len(set(container_ids)) != len(
        F2C_SERVICES
    ):
        raise ValueError("formal F2c requires exactly four live Compose containers")
    inspected = _docker_json(
        ("docker", "inspect", *container_ids),
        environment=environment,
    )
    if not isinstance(inspected, list) or len(inspected) != len(F2C_SERVICES):
        raise ValueError("docker inspect did not return four containers")
    by_service: dict[str, dict[str, object]] = {}
    for container in inspected:
        if not isinstance(container, dict):
            raise ValueError("docker inspect container row is not an object")
        config = container.get("Config")
        state = container.get("State")
        host_config = container.get("HostConfig")
        network = container.get("NetworkSettings")
        if not all(
            isinstance(value, dict)
            for value in (config, state, host_config, network)
        ):
            raise ValueError(
                "docker inspect omitted Config, State, HostConfig, or NetworkSettings"
            )
        assert isinstance(config, dict)
        assert isinstance(state, dict)
        assert isinstance(host_config, dict)
        assert isinstance(network, dict)
        labels = config.get("Labels")
        health = state.get("Health")
        service = (
            labels.get("com.docker.compose.service")
            if isinstance(labels, dict)
            else None
        )
        if (
            service not in F2C_SERVICES
            or service in by_service
            or not isinstance(health, dict)
        ):
            raise ValueError("docker inspect service/health identity is invalid")
        device_requests = host_config.get("DeviceRequests")
        ports = network.get("Ports")
        if (
            not isinstance(device_requests, list)
            or len(device_requests) != 1
            or not isinstance(device_requests[0], dict)
            or not isinstance(ports, dict)
        ):
            raise ValueError("docker inspect GPU/port topology is invalid")
        device_ids = device_requests[0].get("DeviceIDs")
        bindings = ports.get("8000/tcp")
        if (
            not isinstance(device_ids, list)
            or not isinstance(bindings, list)
            or len(bindings) != 1
            or not isinstance(bindings[0], dict)
        ):
            raise ValueError("docker inspect omitted GPU IDs or port binding")
        by_service[str(service)] = {
            "service": service,
            "container_id": container.get("Id"),
            "config_image": config.get("Image"),
            "image_id": container.get("Image"),
            "status": state.get("Status"),
            "health": health.get("Status"),
            "device_ids": device_ids,
            "cpuset_cpus": host_config.get("CpusetCpus"),
            "port": {
                "container_port": "8000/tcp",
                "host_ip": bindings[0].get("HostIp"),
                "host_port": bindings[0].get("HostPort"),
            },
        }
    if set(by_service) != set(F2C_SERVICES):
        raise ValueError("Compose service set differs from frozen F2c topology")
    services = [by_service[service] for service in F2C_SERVICES]
    image_ids = {item["image_id"] for item in services}
    config_images = {item["config_image"] for item in services}
    if len(image_ids) != 1 or config_images != {runtime_image}:
        raise ValueError("F2c containers do not share the frozen runtime image")
    image_id = next(iter(image_ids))
    by_id = _docker_json(
        ("docker", "image", "inspect", str(image_id)),
        environment=environment,
    )
    by_ref = _docker_json(
        ("docker", "image", "inspect", runtime_image),
        environment=environment,
    )
    if (
        not isinstance(by_id, list)
        or len(by_id) != 1
        or not isinstance(by_ref, list)
        or len(by_ref) != 1
        or not isinstance(by_id[0], dict)
        or not isinstance(by_ref[0], dict)
    ):
        raise ValueError("docker image inspect returned an invalid result")
    labels = (by_id[0].get("Config") or {}).get("Labels") or {}
    ref_labels = (by_ref[0].get("Config") or {}).get("Labels") or {}
    revision = (
        labels.get("org.opencontainers.image.revision")
        if isinstance(labels, dict)
        else None
    )
    ref_revision = (
        ref_labels.get("org.opencontainers.image.revision")
        if isinstance(ref_labels, dict)
        else None
    )
    if revision != expected_commit or ref_revision != expected_commit:
        raise ValueError("runtime OCI revision does not equal expected commit")
    attestation = {
        "schema_version": 1,
        "compose_file": DEFAULT_CONFIG_PATHS[0],
        "runtime_image": runtime_image,
        "services": services,
        "image": {
            "id": image_id,
            "ref": runtime_image,
            "id_inspect_id": by_id[0].get("Id"),
            "ref_inspect_id": by_ref[0].get("Id"),
            "revision": revision,
        },
    }
    if not valid_runtime_attestation(
        attestation,
        runtime_image=runtime_image,
        expected_commit=expected_commit,
    ):
        raise ValueError("runtime attestation does not satisfy the frozen schema")
    return attestation


def formal_preflight(
    config: WorkloadConfig,
    topology: EndpointTopology,
    *,
    model: str,
    model_revision: str | None,
    model_digests: dict[str, str],
    runtime_image: str | None,
    config_paths: tuple[Path, ...],
    expected_commit: str | None,
) -> dict[str, object] | None:
    """Reject unbound formal runs before querying or allocating any GPU."""

    config.validate()
    topology.validate(formal=config.profile == "formal")
    missing_configs = [str(path) for path in config_paths if not path.is_file()]
    if missing_configs:
        raise ValueError("missing config files: " + ", ".join(missing_configs))
    if config.profile != "formal":
        return None
    expected_config_paths = tuple(
        (_REPO_ROOT / relative_path).resolve()
        for relative_path in DEFAULT_CONFIG_PATHS
    )
    if tuple(path.resolve() for path in config_paths) != expected_config_paths:
        raise ValueError("formal F2c freezes the three default config files")
    if model != "qwen3-32b":
        raise ValueError("formal F2c requires model qwen3-32b")
    if not _valid_git_commit(expected_commit):
        raise ValueError("formal F2c requires --expected-commit as a full SHA")
    source = _source_snapshot(expected_commit)
    if not _valid_source_value(source):
        raise ValueError("formal F2c source descriptor is incomplete")
    if source["expected_commit_matches"] is not True:
        raise ValueError("formal F2c expected commit does not equal HEAD")
    if source["git_clean"] is not True:
        raise ValueError("formal F2c requires a clean source tree")
    if not _valid_git_commit(model_revision):
        raise ValueError(
            "formal F2c requires model revision as a full 40-character "
            "lower-case git SHA"
        )
    if not model_digests or not all(
        isinstance(label, str)
        and bool(label)
        and _valid_sha256(digest)
        for label, digest in model_digests.items()
    ):
        raise ValueError("formal F2c requires immutable model digests")
    if not valid_runtime_image(runtime_image):
        raise ValueError(
            "formal F2c requires --runtime-image repository@sha256:<64 lower hex>"
        )
    assert isinstance(runtime_image, str)
    assert isinstance(expected_commit, str)
    return capture_runtime_attestation(
        runtime_image=runtime_image,
        expected_commit=expected_commit,
    )


def _configuration_snapshot(
    config: WorkloadConfig,
    topology: EndpointTopology,
    *,
    model: str,
    model_revision: str | None,
    model_digests: dict[str, str],
    runtime_image: str | None,
    runtime_attestation: dict[str, object] | None,
    config_paths: tuple[Path, ...],
) -> dict[str, object]:
    return {
        "workload": asdict(config),
        "cohorts": {
            "a": list(topology.cohort_a),
            "b": list(topology.cohort_b),
        },
        "model": model,
        "model_revision": model_revision,
        "model_digests": model_digests,
        "runtime_image": runtime_image,
        "runtime_attestation": runtime_attestation,
        "files": [
            {
                "path": _repo_relative_path(path),
                "sha256": sha256_file(path.resolve()),
            }
            for path in config_paths
        ],
    }


def _run_output(command: tuple[str, ...]) -> list[str]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().splitlines()


def _environment_snapshot(
    runtime_attestation: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "sched_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "github": {
            name: os.environ.get(name)
            for name in (
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_EVENT_NAME",
                "GITHUB_REPOSITORY",
                "GITHUB_SHA",
            )
        },
        "nvidia_smi_gpus": _run_output(
            (
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,pci.bus_id,uuid",
                "--format=csv,noheader",
            )
        ),
        "nvidia_smi_topology": _run_output(("nvidia-smi", "topo", "-m")),
        "runtime_attestation": runtime_attestation,
    }


def capture_provenance(
    config: WorkloadConfig,
    topology: EndpointTopology,
    *,
    model: str,
    model_revision: str | None,
    model_digests: dict[str, str],
    runtime_image: str | None,
    runtime_attestation: dict[str, object] | None,
    config_paths: tuple[Path, ...],
    expected_commit: str | None,
) -> dict[str, object]:
    return {
        "source": hashed_descriptor(_source_snapshot(expected_commit)),
        "configuration": hashed_descriptor(
            _configuration_snapshot(
                config,
                topology,
                model=model,
                model_revision=model_revision,
                model_digests=model_digests,
                runtime_image=runtime_image,
                runtime_attestation=runtime_attestation,
                config_paths=config_paths,
            )
        ),
        "environment": hashed_descriptor(
            _environment_snapshot(runtime_attestation)
        ),
    }


def _physical_topology(
    config: WorkloadConfig,
    topology: EndpointTopology,
    environment: dict[str, object],
) -> dict[str, object]:
    env_value = environment["environment"]["value"]
    assert isinstance(env_value, dict)
    return {
        "logical_replica_ids": list(REPLICA_IDS),
        "cohorts": {
            "a": list(topology.cohort_a),
            "b": list(topology.cohort_b),
        },
        "physical_endpoints": [
            *topology.cohort_a,
            *topology.cohort_b,
        ],
        "round_crossover": [
            {
                "round_index": round_index,
                CANDIDATE_POLICY: cohort_for(
                    round_index, CANDIDATE_POLICY
                ),
                BASELINE_POLICY: cohort_for(
                    round_index, BASELINE_POLICY
                ),
            }
            for round_index in range(config.rounds)
        ],
        "nvidia_smi_gpus": env_value["nvidia_smi_gpus"],
        "nvidia_smi_topology": env_value["nvidia_smi_topology"],
    }


def _build_pool(
    *,
    policy: str,
    endpoints: tuple[str, str],
    model: str,
    timeout_s: float,
    router_log_path: Path,
) -> ReplicaPool:
    backends = {
        replica_id: OpenAICompatBackend(
            endpoint,
            model,
            api_key_env=None,
            timeout_s=timeout_s,
            upstream="kairyu",
            request_stream_usage=True,
        )
        for replica_id, endpoint in zip(REPLICA_IDS, endpoints, strict=True)
    }
    return ReplicaPool(
        backends,
        log=JsonlRouterLog(router_log_path),
        prefix_index=(PrefixIndex() if policy == CANDIDATE_POLICY else None),
    )


async def _stream_request(
    pool: ReplicaPool,
    config: WorkloadConfig,
    family: FamilyTrace,
    *,
    policy: str,
    turn: int,
    prompt: str,
    planned_ns: int,
) -> RequestObservation:
    receipt_ns = time.perf_counter_ns()
    session_id = (
        family.seed_session_id if turn == 0 else family.measured_session_id
    )
    request_id = request_id_for(family, policy, turn)
    request = GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            min_tokens=config.min_tokens,
            ignore_eos=config.ignore_eos,
        ),
        cache_hint=CacheHint(
            session_id=session_id,
            prefix_fingerprint=family.prefix_fingerprint,
        ),
        placement_started_ns=receipt_ns,
    )
    first_content_ns: int | None = None
    final = None
    error_type: str | None = None
    try:
        async with asyncio.timeout(config.request_timeout_s):
            async for chunk in pool.stream(request):
                observed_ns = time.perf_counter_ns()
                if first_content_ns is None and chunk.text:
                    first_content_ns = observed_ns
                if chunk.finished:
                    final = chunk
    except Exception as error:
        error_type = type(error).__name__
    done_ns = time.perf_counter_ns()
    usage: GenerationUsage | None = None if final is None else final.usage
    output = "" if final is None else final.text
    success = (
        error_type is None
        and final is not None
        and final.finished
        and first_content_ns is not None
        and usage is not None
        and bool(output)
    )
    if not success and error_type is None:
        error_type = "IncompleteStream"
    row: dict[str, object] = {
        "type": "request",
        "binding": turn > 0,
        "phase": "seed" if turn == 0 else "measure",
        "turn": turn,
        "round_index": family.round_index,
        "family_index": family.family_index,
        "family_id": family.family_id,
        "policy": policy,
        "cohort": cohort_for(family.round_index, policy),
        "request_id": request_id,
        "session_sha256": sha256_text(session_id),
        "prefix_fingerprint": family.prefix_fingerprint,
        "prefix_fingerprint_sha256": sha256_text(
            family.prefix_fingerprint
        ),
        "prompt_sha256": sha256_text(prompt),
        "prompt_chars": len(prompt),
        "prompt_word_count": len(prompt.split()),
        "planned_ns": planned_ns,
        "receipt_ns": receipt_ns,
        "selected_at_ns": None,
        "first_content_ns": first_content_ns,
        "done_ns": done_ns,
        "schedule_lateness_ns": max(0, receipt_ns - planned_ns),
        "paired_receipt_skew_ns": None,
        "ttft_ns": (
            None
            if first_content_ns is None
            else first_content_ns - receipt_ns
        ),
        "reason": None,
        "selected_replica_id": None,
        "replica_ordinal": None,
        "replica_generation": None,
        "router_decision": None,
        "status": "success" if success else "failure",
        "terminal": True,
        "error_type": error_type,
        "usage": (
            None
            if usage is None
            else {
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": usage.cached_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        ),
        "finish_reason": (
            None
            if final is None or not final.completions
            else final.completions[0].finish_reason
        ),
        "output_sha256": sha256_text(output),
        "output_chars": len(output),
    }
    return RequestObservation(row=row, output=output)


async def _launch_pair(
    candidate_pool: ReplicaPool,
    baseline_pool: ReplicaPool,
    config: WorkloadConfig,
    family: FamilyTrace,
    *,
    turn: int,
    prompt: str,
    planned_ns: int,
) -> tuple[RequestObservation, RequestObservation]:
    candidate_task = asyncio.create_task(
        _stream_request(
            candidate_pool,
            config,
            family,
            policy=CANDIDATE_POLICY,
            turn=turn,
            prompt=prompt,
            planned_ns=planned_ns,
        )
    )
    baseline_task = asyncio.create_task(
        _stream_request(
            baseline_pool,
            config,
            family,
            policy=BASELINE_POLICY,
            turn=turn,
            prompt=prompt,
            planned_ns=planned_ns,
        )
    )
    candidate, baseline = await asyncio.gather(
        candidate_task, baseline_task
    )
    skew = abs(
        int(candidate.row["receipt_ns"]) - int(baseline.row["receipt_ns"])
    )
    candidate.row["paired_receipt_skew_ns"] = skew
    baseline.row["paired_receipt_skew_ns"] = skew
    return candidate, baseline


async def _scheduled_pair(
    candidate_pool: ReplicaPool,
    baseline_pool: ReplicaPool,
    config: WorkloadConfig,
    family: FamilyTrace,
    *,
    turn: int,
    prompt: str,
    planned_ns: int,
) -> tuple[RequestObservation, RequestObservation]:
    delay_s = (planned_ns - time.perf_counter_ns()) / _NANOSECONDS
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    return await _launch_pair(
        candidate_pool,
        baseline_pool,
        config,
        family,
        turn=turn,
        prompt=prompt,
        planned_ns=planned_ns,
    )


def _read_router_decisions(path: Path) -> dict[str, dict[str, object]]:
    decisions: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path):
        if row.get("kind") != "replica":
            raise ValueError("F2c router log contains a non-replica record")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or request_id in decisions:
            raise ValueError("F2c router decisions require unique request IDs")
        decisions[request_id] = row
    return decisions


def _join_router_decisions(
    rows: list[dict[str, object]],
    decisions: dict[str, dict[str, object]],
) -> None:
    expected_ids = {
        str(row["request_id"])
        for row in rows
        if row.get("type") == "request"
    }
    if set(decisions) != expected_ids:
        raise ValueError("router-log/request causal join is incomplete")
    for row in rows:
        if row.get("type") != "request":
            continue
        decision = decisions[str(row["request_id"])]
        if decision.get("placement_started_ns") != row["receipt_ns"]:
            raise ValueError("router placement start differs from receipt")
        row["selected_at_ns"] = decision["selected_at_ns"]
        row["reason"] = decision["reason"]
        row["selected_replica_id"] = decision["replica_id"]
        row["replica_ordinal"] = decision["replica"]
        row["replica_generation"] = decision["replica_generation"]
        row["router_decision"] = decision


async def _run_round(
    config: WorkloadConfig,
    topology: EndpointTopology,
    trace: tuple[FamilyTrace, ...],
    *,
    round_index: int,
    model: str,
    temp_dir: Path,
) -> list[dict[str, object]]:
    families = tuple(
        family for family in trace if family.round_index == round_index
    )
    if len(families) != config.families_per_round:
        raise RuntimeError("round trace is incomplete")
    candidate_cohort = cohort_for(round_index, CANDIDATE_POLICY)
    baseline_cohort = cohort_for(round_index, BASELINE_POLICY)
    endpoints = {
        "a": topology.cohort_a,
        "b": topology.cohort_b,
    }
    candidate_log = temp_dir / f"round-{round_index:02d}-candidate.jsonl"
    baseline_log = temp_dir / f"round-{round_index:02d}-baseline.jsonl"
    candidate_pool = _build_pool(
        policy=CANDIDATE_POLICY,
        endpoints=endpoints[candidate_cohort],
        model=model,
        timeout_s=config.request_timeout_s,
        router_log_path=candidate_log,
    )
    baseline_pool = _build_pool(
        policy=BASELINE_POLICY,
        endpoints=endpoints[baseline_cohort],
        model=model,
        timeout_s=config.request_timeout_s,
        router_log_path=baseline_log,
    )
    observations: list[RequestObservation] = []
    try:
        # Non-binding seeds are serialized by family so HRW target identity,
        # rather than queue-depth fallback, places every initial cache copy.
        for family in families:
            planned_ns = time.perf_counter_ns()
            pair = await _launch_pair(
                candidate_pool,
                baseline_pool,
                config,
                family,
                turn=0,
                prompt=family.seed_prompt,
                planned_ns=planned_ns,
            )
            observations.extend(pair)

        interval_ns = int(config.arrival_interval_s * _NANOSECONDS)
        phase1_anchor_ns = time.perf_counter_ns() + interval_ns
        phase1 = await asyncio.gather(
            *(
                _scheduled_pair(
                    candidate_pool,
                    baseline_pool,
                    config,
                    family,
                    turn=1,
                    prompt=family.turn1_prompt,
                    planned_ns=(
                        phase1_anchor_ns + family.family_index * interval_ns
                    ),
                )
                for family in families
            )
        )
        observations.extend(
            observation for pair in phase1 for observation in pair
        )

        for family, pair in zip(families, phase1, strict=True):
            candidate, baseline = pair
            if (
                candidate.row["status"] != "success"
                or baseline.row["status"] != "success"
            ):
                raise RuntimeError(
                    "turn1 failed before frozen transcript barrier: "
                    f"{family.family_id}"
                )

        # The complete phase-one success barrier closes before phase two.
        # Both arms then receive the predeclared family-specific transcript;
        # neither post-treatment output becomes input to a later request.
        phase2_anchor_ns = time.perf_counter_ns() + interval_ns
        phase2 = await asyncio.gather(
            *(
                _scheduled_pair(
                    candidate_pool,
                    baseline_pool,
                    config,
                    family,
                    turn=2,
                    prompt=turn2_prompt(family),
                    planned_ns=(
                        phase2_anchor_ns + family.family_index * interval_ns
                    ),
                )
                for family in families
            )
        )
        observations.extend(
            observation for pair in phase2 for observation in pair
        )
    finally:
        await asyncio.gather(
            candidate_pool.shutdown(),
            baseline_pool.shutdown(),
        )

    rows = [observation.row for observation in observations]
    decisions = {
        **_read_router_decisions(candidate_log),
        **_read_router_decisions(baseline_log),
    }
    if len(decisions) != len(rows):
        raise ValueError("candidate/baseline router-log IDs overlap")
    _join_router_decisions(rows, decisions)
    return rows


async def measure(
    config: WorkloadConfig,
    topology: EndpointTopology,
    *,
    model: str,
    model_revision: str | None,
    model_digests: dict[str, str],
    runtime_image: str | None,
    config_paths: tuple[Path, ...],
    expected_commit: str | None = None,
) -> list[dict[str, object]]:
    """Run one complete, frozen F2c crossover measurement."""

    runtime_attestation_start = formal_preflight(
        config,
        topology,
        model=model,
        model_revision=model_revision,
        model_digests=model_digests,
        runtime_image=runtime_image,
        config_paths=config_paths,
        expected_commit=expected_commit,
    )
    trace = build_trace(config)
    provenance_start = capture_provenance(
        config,
        topology,
        model=model,
        model_revision=model_revision,
        model_digests=model_digests,
        runtime_image=runtime_image,
        runtime_attestation=runtime_attestation_start,
        config_paths=config_paths,
        expected_commit=expected_commit,
    )
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "config": asdict(config),
            "trace": trace_descriptor(config, trace),
            "model": model,
            "model_revision": model_revision,
            "model_digests": model_digests,
            "runtime_image": runtime_image,
            "topology": _physical_topology(
                config, topology, provenance_start
            ),
            "provenance_start": provenance_start,
        }
    ]
    with tempfile.TemporaryDirectory(prefix="kairyu-f2c-") as raw_temp:
        temp_dir = Path(raw_temp)
        for round_index in range(config.rounds):
            rows.extend(
                await _run_round(
                    config,
                    topology,
                    trace,
                    round_index=round_index,
                    model=model,
                    temp_dir=temp_dir,
                )
            )
    runtime_attestation_end = (
        capture_runtime_attestation(
            runtime_image=runtime_image,
            expected_commit=expected_commit,
        )
        if config.profile == "formal"
        and isinstance(runtime_image, str)
        and isinstance(expected_commit, str)
        else None
    )
    provenance_end = capture_provenance(
        config,
        topology,
        model=model,
        model_revision=model_revision,
        model_digests=model_digests,
        runtime_image=runtime_image,
        runtime_attestation=runtime_attestation_end,
        config_paths=config_paths,
        expected_commit=expected_commit,
    )
    rows.append(
        {
            "type": "run_end",
            "request_count": sum(
                row.get("type") == "request" for row in rows
            ),
            "provenance_end": provenance_end,
        }
    )
    return rows


def _parse_endpoint_pair(value: str) -> tuple[str, str]:
    endpoints = tuple(item.strip().rstrip("/") for item in value.split(","))
    if len(endpoints) != 2 or any(not endpoint for endpoint in endpoints):
        raise argparse.ArgumentTypeError(
            "cohort endpoints must be two comma-separated URLs"
        )
    return endpoints  # type: ignore[return-value]


def _parse_model_digests(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, value in enumerate(values):
        if "=" in value:
            label, digest = value.split("=", 1)
        else:
            label, digest = f"model-{index:03d}", value
        digest = digest.lower()
        if not label or label in result or not _valid_sha256(digest):
            raise ValueError(
                "--model-digest requires unique LABEL=64_HEX values"
            )
        result[label] = digest
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    parser.add_argument("--model", default="qwen3-32b")
    parser.add_argument("--model-revision")
    parser.add_argument("--trace-namespace")
    parser.add_argument(
        "--runtime-image",
        default=os.environ.get("KAIRYU_F2C_IMAGE"),
    )
    parser.add_argument(
        "--model-digest",
        action="append",
        default=[],
        metavar="LABEL=SHA256",
    )
    parser.add_argument(
        "--cohort-a-endpoints",
        type=_parse_endpoint_pair,
        default=DEFAULT_COHORT_A_ENDPOINTS,
    )
    parser.add_argument(
        "--cohort-b-endpoints",
        type=_parse_endpoint_pair,
        default=DEFAULT_COHORT_B_ENDPOINTS,
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--expected-commit",
        default=os.environ.get("F2C_EXPECTED_COMMIT"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify_artifact is not None:
        result = verify_artifact(
            args.verify_artifact,
            assert_gate=args.assert_gate,
        )
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required for measurement")
        try:
            model_digests = _parse_model_digests(args.model_digest)
        except ValueError as error:
            parser.error(str(error))
        config_paths = tuple(
            args.config_file
            if args.config_file is not None
            else (_REPO_ROOT / path for path in DEFAULT_CONFIG_PATHS)
        )
        missing = [str(path) for path in config_paths if not path.is_file()]
        if missing:
            parser.error("missing config files: " + ", ".join(missing))
        if args.profile == "formal" and args.trace_namespace is None:
            parser.error("--trace-namespace is required for formal measurement")
        config = profile_config(
            args.profile,
            trace_namespace=args.trace_namespace,
        )
        topology = EndpointTopology(
            cohort_a=tuple(args.cohort_a_endpoints),
            cohort_b=tuple(args.cohort_b_endpoints),
        )
        rows = asyncio.run(
            measure(
                config,
                topology,
                model=args.model,
                model_revision=args.model_revision,
                model_digests=model_digests,
                runtime_image=args.runtime_image,
                config_paths=config_paths,
                expected_commit=args.expected_commit,
            )
        )
        result = write_artifact(args.output_dir, rows)
        if (
            result["passed"] is not True
            and (args.assert_gate or config.profile == "formal")
        ):
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
