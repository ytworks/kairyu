#!/usr/bin/env python3
"""Objective SM120 FlashAttention-4 versus FlashInfer prefill evidence.

The benchmark measures the same one-layer paged-KV work with both backends.
Warmups are disjoint, while timed pairs alternate AB/BA order to limit drift
and order bias.  Every ``perf_counter_ns`` reading is retained.  The artifact
is self-verifying: ``verify`` reconstructs measurements, statistics, the exact
paired sign test, and the backend recommendation from the raw samples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

from kairyu.engine.core.hw_profile import HardwareProfile  # noqa: E402
from kairyu.engine.core.kv_pool import PagedKVPool  # noqa: E402

SCHEMA_VERSION = 1
BENCHMARK_ID = "qwen3-32b-sm120-attention-prefill-ab"
MODEL = "Qwen/Qwen3-32B"
DEVICE = "cuda"
DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"
NUM_LAYERS = 1
PAGE_SIZE = 16
CONTEXT_TOKENS = 2_048
CHUNK_TOKENS = 512
HEAD_DIM = 128
SEED = 277
WARMUPS_PER_BACKEND = 5
PAIRED_BLOCKS = 24
PARITY_RTOL = 0.02
PARITY_ATOL = 0.02

FLASHINFER = "flashinfer"
FLASHATTENTION4 = "flashattention4"
BACKENDS = (FLASHINFER, FLASHATTENTION4)
_ALPHA_NUMERATOR = 1
_ALPHA_DENOMINATOR = 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SM120_PROFILE = HardwareProfile(arch="cuda", sm=120)


@dataclass(frozen=True)
class Scenario:
    name: str
    tensor_parallel_size: int
    num_query_heads: int
    num_kv_heads: int


SCENARIOS = (
    Scenario("qwen3-32b-tp4", 4, 16, 2),
    Scenario("qwen3-32b-tp8", 8, 8, 1),
)


def nearest_rank_percentile(values: Sequence[int], quantile: float) -> int:
    """Return the nearest-rank percentile without interpolation."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered))
    return ordered[rank - 1]


def _scenario_geometry(scenario: Scenario) -> dict[str, Any]:
    sequence_tokens = CONTEXT_TOKENS + CHUNK_TOKENS
    return {
        "name": scenario.name,
        "tensor_parallel_size": scenario.tensor_parallel_size,
        "num_query_heads": scenario.num_query_heads,
        "num_kv_heads": scenario.num_kv_heads,
        "head_dim": HEAD_DIM,
        "query_shape": [
            CHUNK_TOKENS,
            scenario.num_query_heads,
            HEAD_DIM,
        ],
        "output_shape": [
            CHUNK_TOKENS,
            scenario.num_query_heads * HEAD_DIM,
        ],
        "sequence_tokens": sequence_tokens,
        "num_pages": math.ceil(sequence_tokens / PAGE_SIZE),
    }


def benchmark_geometry() -> dict[str, Any]:
    return {
        "model": MODEL,
        "dtype": DTYPE_NAME,
        "num_layers": NUM_LAYERS,
        "page_size": PAGE_SIZE,
        "context_tokens": CONTEXT_TOKENS,
        "chunk_tokens": CHUNK_TOKENS,
        "head_dim": HEAD_DIM,
        "seed": SEED,
        "scenarios": [_scenario_geometry(scenario) for scenario in SCENARIOS],
    }


def methodology() -> dict[str, Any]:
    return {
        "clock": "time.perf_counter_ns",
        "synchronization": (
            "torch.cuda.synchronize immediately before and after every timed call"
        ),
        "warmups": {
            "kind": "disjoint per backend and scenario",
            "calls_per_backend": WARMUPS_PER_BACKEND,
        },
        "paired_blocks": PAIRED_BLOCKS,
        "order": (
            "AB on even blocks and BA on odd blocks; "
            "A=flashinfer, B=flashattention4"
        ),
        "latency_summary": "median and nearest-rank p99 over raw elapsed_ns",
        "paired_ratio": (
            "flashinfer elapsed_ns / flashattention4 elapsed_ns; "
            "values above 1 favor flashattention4"
        ),
        "inference": {
            "test": "exact one-sided paired sign test",
            "alternative": "flashattention4 is faster",
            "ties": "excluded",
            "alpha": {
                "numerator": _ALPHA_NUMERATOR,
                "denominator": _ALPHA_DENOMINATOR,
            },
        },
        "recommendation_rule": (
            "recommend flashattention4 only when output parity passes and the "
            "exact one-sided paired sign test favors it at alpha=0.05 in every "
            "required scenario; there is no minimum effect-size threshold"
        ),
    }


def backend_contract() -> dict[str, Any]:
    return {
        FLASHINFER: {
            "prefill": FLASHINFER,
            "decode": FLASHINFER,
            "kv_mode": "paged-direct",
        },
        FLASHATTENTION4: {
            "prefill": FLASHATTENTION4,
            "decode": FLASHINFER,
            "kv_mode": "paged-materialized",
        },
        "decode_delegation": (
            "flashattention4 is measured for prefill only; decode remains "
            "delegated to flashinfer"
        ),
    }


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return package_version(name)
        except PackageNotFoundError:
            continue
    return None


def _environment(device: str) -> dict[str, Any]:
    torch_device = torch.device(device)
    properties = torch.cuda.get_device_properties(torch_device)
    major, minor = torch.cuda.get_device_capability(torch_device)
    return {
        "device": device,
        "packages": {
            "torch": _package_version("torch") or torch.__version__,
            "flashinfer": _package_version("flashinfer", "flashinfer-python"),
            "flash-attn-4": _package_version("flash-attn-4"),
        },
        "cuda": {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "gpu": {
            "name": properties.name,
            "compute_capability": [major, minor],
            "sm": major * 10 + minor,
            "total_memory_bytes": int(properties.total_memory),
        },
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _script_sha256() -> str:
    return _sha256_bytes(Path(__file__).resolve().read_bytes())


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _provenance() -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    # Agent worktree control directories are persistent host state, not source
    # inputs. Excluding that one path lets a clean measurement bind to the exact
    # committed implementation while preserving every relevant dirty entry.
    status = "\n".join(
        line
        for line in _git_output("status", "--porcelain=v1").splitlines()
        if not line[3:].startswith(".claude/worktrees/")
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "commit": commit,
            "dirty": bool(status),
            "status_sha256": _sha256_bytes(status.encode()),
            "script_sha256": _script_sha256(),
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    unhashed = dict(payload)
    unhashed.pop("artifact_sha256", None)
    return json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def seal_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied artifact with its canonical content hash."""
    sealed = copy.deepcopy(dict(payload))
    sealed.pop("artifact_sha256", None)
    sealed["artifact_sha256"] = _sha256_bytes(_canonical_payload_bytes(sealed))
    return sealed


def _expected_order(block: int) -> tuple[str, tuple[str, str]]:
    if block % 2 == 0:
        return "AB", (FLASHINFER, FLASHATTENTION4)
    return "BA", (FLASHATTENTION4, FLASHINFER)


def _validated_pairs(
    measurements: object,
) -> list[tuple[int, int]]:
    if not isinstance(measurements, list):
        raise ValueError("measurements must be a list")
    if len(measurements) != PAIRED_BLOCKS * len(BACKENDS):
        raise ValueError("wrong number of timed measurements")

    pairs: list[tuple[int, int]] = []
    previous_finished = -1
    expected_keys = {
        "block",
        "pair_order",
        "position",
        "backend",
        "started_ns",
        "finished_ns",
        "elapsed_ns",
    }
    for block in range(PAIRED_BLOCKS):
        pair_order, backend_order = _expected_order(block)
        durations: dict[str, int] = {}
        for position, backend in enumerate(backend_order):
            item = measurements[block * 2 + position]
            if not isinstance(item, dict) or set(item) != expected_keys:
                raise ValueError("measurement shape mismatch")
            if (
                type(item["block"]) is not int
                or item["block"] != block
                or item["pair_order"] != pair_order
                or type(item["position"]) is not int
                or item["position"] != position
                or item["backend"] != backend
            ):
                raise ValueError("measurement order mismatch")
            started = item["started_ns"]
            finished = item["finished_ns"]
            elapsed = item["elapsed_ns"]
            if (
                type(started) is not int
                or type(finished) is not int
                or type(elapsed) is not int
                or started < 0
                or finished <= started
                or elapsed != finished - started
                or started < previous_finished
            ):
                raise ValueError("invalid perf_counter_ns sample")
            durations[backend] = elapsed
            previous_finished = finished
        pairs.append((durations[FLASHINFER], durations[FLASHATTENTION4]))
    return pairs


def exact_fa4_faster_sign_test(
    pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Exact one-sided sign test; each pair is (FlashInfer ns, FA4 ns)."""
    wins = sum(fa4 < flashinfer for flashinfer, fa4 in pairs)
    losses = sum(fa4 > flashinfer for flashinfer, fa4 in pairs)
    ties = len(pairs) - wins - losses
    trials = wins + losses
    tail_numerator = sum(
        math.comb(trials, count) for count in range(wins, trials + 1)
    )
    tail_denominator = 2**trials
    significant = (
        tail_numerator * _ALPHA_DENOMINATOR
        <= tail_denominator * _ALPHA_NUMERATOR
    )
    return {
        "alternative": "flashattention4_faster",
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "non_tied_pairs": trials,
        "tail_numerator": tail_numerator,
        "tail_denominator": tail_denominator,
        "one_sided_p_value": tail_numerator / tail_denominator,
        "alpha": {
            "numerator": _ALPHA_NUMERATOR,
            "denominator": _ALPHA_DENOMINATOR,
        },
        "significant": significant,
    }


def compute_statistics(measurements: object) -> dict[str, Any]:
    pairs = _validated_pairs(measurements)
    flashinfer_samples = [pair[0] for pair in pairs]
    fa4_samples = [pair[1] for pair in pairs]
    ratios = [
        flashinfer / fa4 for flashinfer, fa4 in pairs
    ]
    return {
        "sample_count_per_backend": len(pairs),
        "latency_ns": {
            FLASHINFER: {
                "median": statistics.median(flashinfer_samples),
                "p99_nearest_rank": nearest_rank_percentile(
                    flashinfer_samples, 0.99
                ),
            },
            FLASHATTENTION4: {
                "median": statistics.median(fa4_samples),
                "p99_nearest_rank": nearest_rank_percentile(
                    fa4_samples, 0.99
                ),
            },
        },
        "paired_ratio_flashinfer_over_flashattention4": {
            "values": ratios,
            "median": statistics.median(ratios),
            "geometric_mean": math.exp(
                math.fsum(math.log(value) for value in ratios) / len(ratios)
            ),
        },
        "exact_paired_sign_test": exact_fa4_faster_sign_test(pairs),
    }


def _parity_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    reference_f32 = reference.detach().float()
    candidate_f32 = candidate.detach().float()
    if reference_f32.shape != candidate_f32.shape:
        return {
            "reference": FLASHINFER,
            "candidate": FLASHATTENTION4,
            "shape": list(reference_f32.shape),
            "candidate_shape": list(candidate_f32.shape),
            "rtol": PARITY_RTOL,
            "atol": PARITY_ATOL,
            "max_abs_error": None,
            "max_normalized_error": None,
            "reference_sha256": _tensor_sha256(reference_f32),
            "candidate_sha256": _tensor_sha256(candidate_f32),
            "passed": False,
        }
    absolute = (candidate_f32 - reference_f32).abs()
    allowance = PARITY_ATOL + PARITY_RTOL * reference_f32.abs()
    normalized = absolute / allowance
    max_abs_error = float(absolute.max().item())
    max_normalized_error = float(normalized.max().item())
    return {
        "reference": FLASHINFER,
        "candidate": FLASHATTENTION4,
        "shape": list(reference_f32.shape),
        "candidate_shape": list(candidate_f32.shape),
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "max_abs_error": max_abs_error,
        "max_normalized_error": max_normalized_error,
        "reference_sha256": _tensor_sha256(reference_f32),
        "candidate_sha256": _tensor_sha256(candidate_f32),
        "passed": math.isfinite(max_normalized_error)
        and max_normalized_error <= 1.0,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return _sha256_bytes(contiguous.numpy().tobytes())


def _invoke(
    backend: object,
    query: torch.Tensor,
    pool: PagedKVPool,
    page_table: list[int],
    seq_len: int,
) -> torch.Tensor:
    return backend.attend(
        query,
        pool,
        0,
        page_table,
        seq_len,
        CONTEXT_TOKENS,
    )


def _measure_once(
    *,
    backend: object,
    query: torch.Tensor,
    pool: PagedKVPool,
    page_table: list[int],
    seq_len: int,
    device: str,
) -> tuple[torch.Tensor, int, int]:
    torch.cuda.synchronize(device)
    started_ns = time.perf_counter_ns()
    output = _invoke(backend, query, pool, page_table, seq_len)
    torch.cuda.synchronize(device)
    finished_ns = time.perf_counter_ns()
    return output, started_ns, finished_ns


def _scenario_inputs(
    scenario: Scenario,
    device: str,
) -> tuple[PagedKVPool, torch.Tensor, list[int], int]:
    sequence_tokens = CONTEXT_TOKENS + CHUNK_TOKENS
    num_pages = math.ceil(sequence_tokens / PAGE_SIZE)
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + scenario.tensor_parallel_size)
    pool = PagedKVPool(
        num_layers=NUM_LAYERS,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        num_kv_heads=scenario.num_kv_heads,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        device=device,
    )
    pool.k.copy_(
        torch.randn(
            pool.k.shape,
            dtype=DTYPE,
            device=device,
            generator=generator,
        )
    )
    pool.v.copy_(
        torch.randn(
            pool.v.shape,
            dtype=DTYPE,
            device=device,
            generator=generator,
        )
    )
    query = torch.randn(
        (CHUNK_TOKENS, scenario.num_query_heads, HEAD_DIM),
        dtype=DTYPE,
        device=device,
        generator=generator,
    )
    return pool, query, list(range(num_pages)), sequence_tokens


def _measure_scenario(
    scenario: Scenario,
    backends: Mapping[str, object],
    device: str,
) -> dict[str, Any]:
    pool, query, page_table, seq_len = _scenario_inputs(scenario, device)

    warm_outputs: dict[str, torch.Tensor] = {}
    for backend_name in BACKENDS:
        output = None
        for _ in range(WARMUPS_PER_BACKEND):
            output = _invoke(
                backends[backend_name],
                query,
                pool,
                page_table,
                seq_len,
            )
        torch.cuda.synchronize(device)
        assert output is not None
        warm_outputs[backend_name] = output.detach()

    parity = _parity_summary(
        warm_outputs[FLASHINFER],
        warm_outputs[FLASHATTENTION4],
    )

    measurements: list[dict[str, Any]] = []
    for block in range(PAIRED_BLOCKS):
        pair_order, backend_order = _expected_order(block)
        for position, backend_name in enumerate(backend_order):
            _, started_ns, finished_ns = _measure_once(
                backend=backends[backend_name],
                query=query,
                pool=pool,
                page_table=page_table,
                seq_len=seq_len,
                device=device,
            )
            measurements.append(
                {
                    "block": block,
                    "pair_order": pair_order,
                    "position": position,
                    "backend": backend_name,
                    "started_ns": started_ns,
                    "finished_ns": finished_ns,
                    "elapsed_ns": finished_ns - started_ns,
                }
            )

    return {
        "name": scenario.name,
        "geometry": _scenario_geometry(scenario),
        "parity": parity,
        "measurements": measurements,
        "statistics": compute_statistics(measurements),
    }


def recommendation_for(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evidence = []
    all_scenarios_favor_fa4 = len(scenarios) == len(SCENARIOS)
    for scenario in scenarios:
        parity = scenario.get("parity")
        statistics_payload = scenario.get("statistics")
        sign_test = (
            statistics_payload.get("exact_paired_sign_test")
            if isinstance(statistics_payload, dict)
            else None
        )
        ratio = (
            statistics_payload.get(
                "paired_ratio_flashinfer_over_flashattention4"
            )
            if isinstance(statistics_payload, dict)
            else None
        )
        parity_passed = isinstance(parity, dict) and parity.get("passed") is True
        fa4_faster = (
            isinstance(sign_test, dict)
            and sign_test.get("significant") is True
            and isinstance(ratio, dict)
            and isinstance(ratio.get("median"), (int, float))
            and not isinstance(ratio.get("median"), bool)
            and ratio["median"] > 1.0
        )
        name = scenario.get("name")
        evidence.append(
            {
                "name": name,
                "parity_passed": parity_passed,
                "fa4_faster_by_exact_paired_evidence": fa4_faster,
            }
        )
        all_scenarios_favor_fa4 = (
            all_scenarios_favor_fa4 and parity_passed and fa4_faster
        )

    prefill = (
        FLASHATTENTION4 if all_scenarios_favor_fa4 else FLASHINFER
    )
    reason = (
        "fa4_faster_in_every_required_scenario"
        if all_scenarios_favor_fa4
        else "stable_flashinfer_without_unanimous_paired_evidence"
    )
    return {
        "prefill_backend": prefill,
        "decode_backend": FLASHINFER,
        "reason": reason,
        "required_scenarios": [scenario.name for scenario in SCENARIOS],
        "scenario_evidence": evidence,
        "minimum_effect_size_threshold": None,
    }


def assemble_artifact(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    scenario_copies = copy.deepcopy(list(scenarios))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_ID,
        "methodology": methodology(),
        "geometry": benchmark_geometry(),
        "backend_contract": backend_contract(),
        "environment": copy.deepcopy(dict(environment)),
        "provenance": copy.deepcopy(dict(provenance)),
        "scenarios": scenario_copies,
        "recommendation": recommendation_for(scenario_copies),
    }
    return seal_artifact(payload)


def measure_artifact(device: str = DEVICE) -> dict[str, Any]:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("the objective attention benchmark requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    major, minor = torch.cuda.get_device_capability(torch_device)
    if major * 10 + minor != 120:
        raise RuntimeError(
            "this evidence profile is fixed to SM120; "
            f"found compute capability {major}.{minor}"
        )

    from kairyu.engine.core.attention.flashattention_gpu import (
        FlashAttentionBackend,
    )
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

    with torch.cuda.device(torch_device):
        backends = {
            FLASHINFER: FlashInferBackend(device=device),
            FLASHATTENTION4: FlashAttentionBackend(
                generation=4,
                profile=_SM120_PROFILE,
            ),
        }
        measured = [
            _measure_scenario(scenario, backends, device)
            for scenario in SCENARIOS
        ]
    return assemble_artifact(
        measured,
        environment=_environment(device),
        provenance=_provenance(),
    )


def _parity_record_valid(parity: object, expected_shape: list[int]) -> bool:
    if not isinstance(parity, dict):
        return False
    expected_keys = {
        "reference",
        "candidate",
        "shape",
        "candidate_shape",
        "rtol",
        "atol",
        "max_abs_error",
        "max_normalized_error",
        "reference_sha256",
        "candidate_sha256",
        "passed",
    }
    if set(parity) != expected_keys:
        return False
    normalized = parity["max_normalized_error"]
    maximum = parity["max_abs_error"]
    numeric = (
        isinstance(normalized, (int, float))
        and not isinstance(normalized, bool)
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and math.isfinite(normalized)
        and math.isfinite(maximum)
        and normalized >= 0
        and maximum >= 0
    )
    return (
        parity["reference"] == FLASHINFER
        and parity["candidate"] == FLASHATTENTION4
        and parity["shape"] == expected_shape
        and parity["candidate_shape"] == expected_shape
        and parity["rtol"] == PARITY_RTOL
        and parity["atol"] == PARITY_ATOL
        and numeric
        and isinstance(parity["reference_sha256"], str)
        and _SHA256_RE.fullmatch(parity["reference_sha256"]) is not None
        and isinstance(parity["candidate_sha256"], str)
        and _SHA256_RE.fullmatch(parity["candidate_sha256"]) is not None
        and type(parity["passed"]) is bool
        and parity["passed"] == (normalized <= 1.0)
    )


def _environment_valid(environment: object) -> bool:
    if not isinstance(environment, dict):
        return False
    if set(environment) != {"device", "packages", "cuda", "gpu"}:
        return False
    packages = environment["packages"]
    cuda = environment["cuda"]
    gpu = environment["gpu"]
    if (
        not isinstance(packages, dict)
        or set(packages) != {"torch", "flashinfer", "flash-attn-4"}
        or not isinstance(packages["torch"], str)
        or not packages["torch"]
        or any(
            value is not None and not isinstance(value, str)
            for value in (packages["flashinfer"], packages["flash-attn-4"])
        )
        or not isinstance(cuda, dict)
        or set(cuda) != {"torch_cuda_version", "cudnn_version"}
        or not isinstance(cuda["torch_cuda_version"], str)
        or not cuda["torch_cuda_version"]
        or (
            cuda["cudnn_version"] is not None
            and type(cuda["cudnn_version"]) is not int
        )
        or not isinstance(gpu, dict)
        or set(gpu)
        != {
            "name",
            "compute_capability",
            "sm",
            "total_memory_bytes",
        }
    ):
        return False
    return (
        environment["device"] == DEVICE
        and isinstance(gpu["name"], str)
        and bool(gpu["name"])
        and gpu["compute_capability"] == [12, 0]
        and gpu["sm"] == 120
        and type(gpu["total_memory_bytes"]) is int
        and gpu["total_memory_bytes"] > 0
    )


def _provenance_valid(provenance: object) -> bool:
    if not isinstance(provenance, dict):
        return False
    if set(provenance) != {
        "generated_at_utc",
        "source",
        "python_version",
        "platform",
    }:
        return False
    source = provenance["source"]
    if (
        not isinstance(source, dict)
        or set(source)
        != {"commit", "dirty", "status_sha256", "script_sha256"}
        or not isinstance(source["commit"], str)
        or _COMMIT_RE.fullmatch(source["commit"]) is None
        or type(source["dirty"]) is not bool
        or not isinstance(source["status_sha256"], str)
        or _SHA256_RE.fullmatch(source["status_sha256"]) is None
        or source["script_sha256"] != _script_sha256()
        or not isinstance(provenance["python_version"], str)
        or not provenance["python_version"]
        or not isinstance(provenance["platform"], str)
        or not provenance["platform"]
    ):
        return False
    try:
        generated = datetime.fromisoformat(provenance["generated_at_utc"])
    except (TypeError, ValueError):
        return False
    return generated.tzinfo is not None


def _current_auto_components() -> dict[str, Any]:
    from kairyu.engine.core.attention_selector import select_backend_decision

    sentinel = object()
    previous: object = os.environ.get("KAIRYU_ATTENTION_BACKEND", sentinel)
    os.environ.pop("KAIRYU_ATTENTION_BACKEND", None)
    try:
        decision = select_backend_decision(_SM120_PROFILE)
    finally:
        if previous is not sentinel:
            os.environ["KAIRYU_ATTENTION_BACKEND"] = str(previous)
    return {
        "requested": decision.requested,
        "source": decision.source,
        "prefill": decision.components["prefill"],
        "decode": decision.components["decode"],
    }


def verify_payload(
    artifact: object,
    *,
    assert_policy: bool = False,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    top_keys = {
        "schema_version",
        "benchmark",
        "methodology",
        "geometry",
        "backend_contract",
        "environment",
        "provenance",
        "scenarios",
        "recommendation",
        "artifact_sha256",
    }
    checks["schema"] = (
        isinstance(artifact, dict)
        and set(artifact) == top_keys
        and artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("benchmark") == BENCHMARK_ID
    )
    if not isinstance(artifact, dict):
        return {"valid": False, "checks": checks}

    artifact_hash = artifact.get("artifact_sha256")
    checks["artifact_hash"] = (
        isinstance(artifact_hash, str)
        and _SHA256_RE.fullmatch(artifact_hash) is not None
        and artifact_hash
        == _sha256_bytes(_canonical_payload_bytes(artifact))
    )
    checks["methodology"] = artifact.get("methodology") == methodology()
    checks["geometry"] = artifact.get("geometry") == benchmark_geometry()
    checks["backend_contract"] = (
        artifact.get("backend_contract") == backend_contract()
    )
    checks["environment"] = _environment_valid(artifact.get("environment"))
    checks["provenance"] = _provenance_valid(artifact.get("provenance"))

    scenarios = artifact.get("scenarios")
    expected_scenario_keys = {
        "name",
        "geometry",
        "parity",
        "measurements",
        "statistics",
    }
    scenario_set_valid = (
        isinstance(scenarios, list)
        and len(scenarios) == len(SCENARIOS)
    )
    measurements_valid = scenario_set_valid
    parity_valid = scenario_set_valid
    statistics_valid = scenario_set_valid
    reconstructed: list[dict[str, Any]] = []
    if isinstance(scenarios, list):
        for index, expected in enumerate(SCENARIOS):
            if index >= len(scenarios):
                scenario_set_valid = False
                measurements_valid = False
                parity_valid = False
                statistics_valid = False
                break
            scenario = scenarios[index]
            if (
                not isinstance(scenario, dict)
                or set(scenario) != expected_scenario_keys
                or scenario.get("name") != expected.name
                or scenario.get("geometry") != _scenario_geometry(expected)
            ):
                scenario_set_valid = False
                measurements_valid = False
                parity_valid = False
                statistics_valid = False
                continue
            try:
                computed = compute_statistics(scenario["measurements"])
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                measurements_valid = False
                statistics_valid = False
                continue
            if scenario.get("statistics") != computed:
                statistics_valid = False
            expected_shape = _scenario_geometry(expected)["output_shape"]
            if not _parity_record_valid(scenario.get("parity"), expected_shape):
                parity_valid = False
            rebuilt = copy.deepcopy(scenario)
            rebuilt["statistics"] = computed
            reconstructed.append(rebuilt)
    else:
        scenario_set_valid = False
        measurements_valid = False
        parity_valid = False
        statistics_valid = False

    checks["scenario_set_and_geometry"] = scenario_set_valid
    checks["measurements_complete_and_alternating"] = measurements_valid
    checks["output_parity_record"] = parity_valid
    checks["statistics_reconstructed"] = statistics_valid

    recomputed_recommendation = (
        recommendation_for(reconstructed)
        if len(reconstructed) == len(SCENARIOS)
        else None
    )
    checks["recommendation_reconstructed"] = (
        recomputed_recommendation is not None
        and artifact.get("recommendation") == recomputed_recommendation
    )

    observed_policy = None
    if assert_policy:
        try:
            observed_policy = _current_auto_components()
            recommendation = recomputed_recommendation
            checks["selector_auto_policy_matches"] = (
                recommendation is not None
                and observed_policy["requested"] == "auto"
                and observed_policy["source"] == "hw_profile"
                and observed_policy["prefill"]
                == recommendation["prefill_backend"]
                and observed_policy["decode"]
                == recommendation["decode_backend"]
            )
        except (KeyError, RuntimeError, ValueError):
            checks["selector_auto_policy_matches"] = False

    return {
        "valid": all(checks.values()),
        "checks": checks,
        "recomputed_recommendation": recomputed_recommendation,
        "observed_selector_policy": observed_policy,
    }


def verify_artifact(
    artifact: str | Path | Mapping[str, Any],
    *,
    assert_policy: bool = False,
) -> dict[str, Any]:
    if isinstance(artifact, Mapping):
        payload: object = artifact
    else:
        try:
            payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "valid": False,
                "checks": {"artifact_loadable": False},
                "recomputed_recommendation": None,
                "observed_selector_policy": None,
            }
    return verify_payload(payload, assert_policy=assert_policy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    measure = commands.add_parser("measure", help="run the SM120 GPU benchmark")
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--device", default=DEVICE, choices=(DEVICE,))

    verify = commands.add_parser("verify", help="verify a retained artifact")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--assert-policy", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "measure":
        artifact = measure_artifact(args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "artifact": str(args.output),
                    "recommendation": artifact["recommendation"],
                },
                sort_keys=True,
            )
        )
        return 0

    report = verify_artifact(
        args.artifact,
        assert_policy=args.assert_policy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
