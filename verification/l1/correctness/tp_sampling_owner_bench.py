"""Real-NCCL protocol evidence for rank-0-authoritative TP sampling (#225).

This is deliberately a protocol microbenchmark, not a model throughput test.
It compares the selected fixed-layout rank-0 token broadcast with a valid
all-rank alternative that gathers every rank's sampled tokens and checks exact
equality.  CUDA events time only the collective, while aligned per-rank samples
are reduced to the worst rank after the run.

Examples:

    uv run python verification/l1/correctness/tp_sampling_owner_bench.py \
      --output bench/results/tp-sampling-owner-rtxpro6000.json --assert-gate

    uv run python verification/l1/correctness/tp_sampling_owner_bench.py \
      --verify bench/results/tp-sampling-owner-rtxpro6000.json --assert-gate
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Keep the documented ``python bench/...`` invocation standalone even when the
# local project has not been installed editable into the active environment.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kairyu.engine.core.dist_comm import (  # noqa: E402
    TorchDistCommunicator,
    init_distributed,
)

SCHEMA_VERSION = 1
REQUIRED_WORLD_SIZE = 8
BATCH_SIZES = (1, 8, 16)
PROTOCOLS = ("rank0_broadcast", "all_rank_equality")
TOKEN_BYTES = torch.tensor([], dtype=torch.int64).element_size()
MIN_ROUNDS = 8
MIN_TRIALS_PER_ROUND = 32
MIN_WARMUP = 16
CANDIDATE_STEADY_P95_LIMIT_MS = 1.0
SOURCE_BOUND_PATHS = (
    "verification/l1/correctness/tp_sampling_owner_bench.py",
    "kairyu/engine/core/comm.py",
    "kairyu/engine/core/dist_comm.py",
    "kairyu/engine/core/model_runner.py",
    "kairyu/engine/core/worker.py",
    "kairyu/models/parallel.py",
    "tests/gpu/test_tp_sampling_owner_nccl.py",
)


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("a percentile needs at least one value")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "min_ms": min(values),
        "p50_ms": _nearest_rank(values, 0.50),
        "p95_ms": _nearest_rank(values, 0.95),
        "p99_ms": _nearest_rank(values, 0.99),
        "max_ms": max(values),
    }


def _logical_bytes(world_size: int, batch_size: int) -> dict[str, dict[str, int]]:
    packet = batch_size * TOKEN_BYTES
    return {
        "rank0_broadcast": {
            # One application-level result packet exists.
            "application_payload_bytes": packet,
            # Every non-source rank receives that packet once.
            "aggregate_peer_delivery_bytes": packet * (world_size - 1),
            "materialized_bytes_per_rank": packet,
        },
        "all_rank_equality": {
            # Every rank contributes one complete sampled-token packet.
            "application_payload_bytes": packet * world_size,
            # Each contribution must become visible to every other rank.
            "aggregate_peer_delivery_bytes": (
                packet * world_size * (world_size - 1)
            ),
            "materialized_bytes_per_rank": packet * world_size,
        },
    }


def _rotated_batches(round_index: int) -> tuple[int, ...]:
    offset = round_index % len(BATCH_SIZES)
    return BATCH_SIZES[offset:] + BATCH_SIZES[:offset]


def _protocol_order(round_index: int, batch_order_index: int) -> tuple[str, str]:
    if (round_index + batch_order_index) % 2:
        return PROTOCOLS[1], PROTOCOLS[0]
    return PROTOCOLS


def _canonical(
    batch_size: int, round_index: int, trial_index: int, device: torch.device
) -> torch.Tensor:
    # Stable, non-negative token IDs. Values only need to model the packet.
    base = round_index * 100_000 + trial_index * 100 + batch_size
    return torch.arange(
        base, base + batch_size, dtype=torch.int64, device=device
    )


def _timed_collective(
    comm: TorchDistCommunicator,
    operation,
    start: torch.cuda.Event,
    end: torch.cuda.Event,
) -> float:
    # Align the ranks outside the CUDA-event interval. The reported value is
    # collective device time, not process launch or host scheduler delay.
    comm.barrier()
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
    rounds: int,
    trials_per_round: int,
    warmup: int,
) -> None:
    # Eight otherwise-unpinned Python ranks can be descheduled for one Linux
    # timeslice between a host barrier and NCCL launch. That delay is not GPU
    # collective time, but the late peer makes every GPU event observe it.
    # Pin one lightweight host thread per rank and disable cyclic GC so the
    # steady-state device gate is not secretly a host scheduler benchmark.
    available_cpus = sorted(os.sched_getaffinity(0))
    if len(available_cpus) >= world_size:
        os.sched_setaffinity(0, {available_cpus[rank]})
    torch.set_num_threads(1)
    gc.disable()
    torch.cuda.set_device(rank)
    init_distributed(
        rank,
        world_size,
        f"file://{init_file}",
        backend="nccl",
        timeout_s=180.0,
    )
    comm = TorchDistCommunicator(device=f"cuda:{rank}")
    device = torch.device("cuda", rank)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    gather_buffers = {
        batch_size: torch.empty(
            world_size * batch_size, dtype=torch.int64, device=device
        )
        for batch_size in BATCH_SIZES
    }
    trials: list[dict[str, Any]] = []

    try:
        # Untimed correctness probes bind both protocol semantics. A follower
        # begins divergent and must be overwritten by rank 0. The all-rank
        # alternative must both accept equality and detect one injected rank.
        canonical = _canonical(16, 0, 0, device)
        candidate = canonical.clone() if rank == 0 else torch.full_like(canonical, -77)
        broadcast_before = candidate.detach().cpu().tolist()
        comm.tensor_broadcast(candidate, src=0)
        broadcast_overwrite = bool(torch.equal(candidate, canonical))
        broadcast_after = candidate.detach().cpu().tolist()

        gathered = gather_buffers[16]
        dist.all_gather_into_tensor(gathered, canonical, group=comm.group)
        equal_rows = gathered.view(world_size, 16)
        all_rank_equal_accepts = bool(torch.all(equal_rows == equal_rows[0]))
        equal_gathered = equal_rows.detach().cpu().tolist()

        divergent = canonical.clone()
        if rank == world_size - 1:
            divergent[0].add_(1)
        divergent_input = divergent.detach().cpu().tolist()
        dist.all_gather_into_tensor(gathered, divergent, group=comm.group)
        divergent_rows = gathered.view(world_size, 16)
        all_rank_divergence_detected = bool(
            torch.any(divergent_rows != divergent_rows[0])
        )
        divergent_gathered = divergent_rows.detach().cpu().tolist()

        # Warm both collective shapes before measurement. Warmup order is
        # mirrored by rank, so it cannot change the collective sequence.
        for batch_size in BATCH_SIZES:
            for _ in range(warmup):
                packet = canonical[:batch_size].clone()
                if rank != 0:
                    packet.fill_(-1)
                comm.tensor_broadcast(packet, src=0)
                dist.all_gather_into_tensor(
                    gather_buffers[batch_size],
                    canonical[:batch_size],
                    group=comm.group,
                )
        torch.cuda.synchronize(device)

        trial_mismatches = {protocol: 0 for protocol in PROTOCOLS}
        for round_index in range(rounds):
            for batch_order_index, batch_size in enumerate(
                _rotated_batches(round_index)
            ):
                order = _protocol_order(round_index, batch_order_index)
                for protocol_order_index, protocol in enumerate(order):
                    for trial_index in range(trials_per_round):
                        expected = _canonical(
                            batch_size, round_index, trial_index, device
                        )
                        if protocol == "rank0_broadcast":
                            packet = (
                                expected.clone()
                                if rank == 0
                                else torch.full_like(expected, -(rank + 1))
                            )
                            elapsed = _timed_collective(
                                comm,
                                lambda packet=packet: comm.tensor_broadcast(
                                    packet, src=0
                                ),
                                start,
                                end,
                            )
                            if not torch.equal(packet, expected):
                                trial_mismatches[protocol] += 1
                        else:
                            gathered = gather_buffers[batch_size]
                            elapsed = _timed_collective(
                                comm,
                                lambda gathered=gathered, expected=expected: (
                                    dist.all_gather_into_tensor(
                                        gathered, expected, group=comm.group
                                    )
                                ),
                                start,
                                end,
                            )
                            rows = gathered.view(world_size, batch_size)
                            if not (
                                torch.all(rows == rows[0])
                                and torch.equal(rows[0], expected)
                            ):
                                trial_mismatches[protocol] += 1
                        trials.append(
                            {
                                "round": round_index,
                                "trial": trial_index,
                                "batch_size": batch_size,
                                "batch_order_index": batch_order_index,
                                "protocol": protocol,
                                "protocol_order_index": protocol_order_index,
                                "elapsed_ms": elapsed,
                            }
                        )

        properties = torch.cuda.get_device_properties(rank)
        payload = {
            "rank": rank,
            "device": {
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "uuid": str(properties.uuid),
                "pci": (
                    f"{properties.pci_domain_id:04x}:"
                    f"{properties.pci_bus_id:02x}:"
                    f"{properties.pci_device_id:02x}"
                ),
                "total_memory_bytes": properties.total_memory,
            },
            "correctness": {
                "broadcast_overwrite": broadcast_overwrite,
                "all_rank_equal_accepts": all_rank_equal_accepts,
                "all_rank_divergence_detected": all_rank_divergence_detected,
                "measured_trials": {
                    protocol: trial_mismatches[protocol] == 0
                    for protocol in PROTOCOLS
                },
                "measured_mismatches": trial_mismatches,
                "probe": {
                    "canonical": canonical.detach().cpu().tolist(),
                    "broadcast_before": broadcast_before,
                    "broadcast_after": broadcast_after,
                    "equal_gathered": equal_gathered,
                    "divergent_input": divergent_input,
                    "divergent_gathered": divergent_gathered,
                },
            },
            "trials": trials,
        }
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        comm.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _trial_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["protocol"]),
        int(row["batch_size"]),
        int(row["round"]),
        int(row["trial"]),
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_binding(commit: str) -> tuple[dict[str, str], bool]:
    """Bind the measurement to committed blobs, including this script itself."""
    hashes: dict[str, str] = {}
    matches = True
    for relative in SOURCE_BOUND_PATHS:
        working_path = _REPO_ROOT / relative
        try:
            working = working_path.read_bytes()
            committed = subprocess.check_output(
                ["git", "show", f"{commit}:{relative}"], cwd=_REPO_ROOT
            )
        except (OSError, subprocess.CalledProcessError):
            matches = False
            continue
        hashes[relative] = _sha256(committed)
        matches &= working == committed
    return hashes, matches and set(hashes) == set(SOURCE_BOUND_PATHS)


def _verify_source_binding(source: dict[str, Any]) -> bool:
    commit = str(source.get("commit", ""))
    recorded = source.get("bound_file_sha256")
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(recorded, dict)
        or set(recorded) != set(SOURCE_BOUND_PATHS)
        or source.get("bound_files_match_commit") is not True
    ):
        return False
    for relative in SOURCE_BOUND_PATHS:
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{commit}:{relative}"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return False
        if recorded.get(relative) != _sha256(committed):
            return False
    current_script_sha = _sha256(Path(__file__).read_bytes())
    return (
        source.get("script_sha256") == current_script_sha
        and recorded["verification/l1/correctness/tp_sampling_owner_bench.py"] == current_script_sha
    )


def _aggregate(
    rank_results: list[dict[str, Any]],
    *,
    world_size: int,
    rounds: int,
    trials_per_round: int,
    warmup: int,
) -> dict[str, Any]:
    by_rank: list[dict[tuple[str, int, int, int], dict[str, Any]]] = []
    for result in rank_results:
        indexed = {_trial_key(row): row for row in result["trials"]}
        if len(indexed) != len(result["trials"]):
            raise RuntimeError(f"rank {result['rank']} emitted duplicate trial keys")
        by_rank.append(indexed)

    raw_trials: dict[str, dict[str, list[dict[str, Any]]]] = {
        protocol: {str(batch): [] for batch in BATCH_SIZES}
        for protocol in PROTOCOLS
    }
    for round_index in range(rounds):
        for batch_order_index, batch_size in enumerate(
            _rotated_batches(round_index)
        ):
            for protocol_order_index, protocol in enumerate(
                _protocol_order(round_index, batch_order_index)
            ):
                for trial_index in range(trials_per_round):
                    key = (protocol, batch_size, round_index, trial_index)
                    rows = [rank[key] for rank in by_rank]
                    expected_order = {
                        (
                            int(row["batch_order_index"]),
                            int(row["protocol_order_index"]),
                        )
                        for row in rows
                    }
                    if expected_order != {
                        (batch_order_index, protocol_order_index)
                    }:
                        raise RuntimeError(
                            f"rank execution order diverged for trial {key}: "
                            f"{sorted(expected_order)}"
                        )
                    rank_ms = [float(row["elapsed_ms"]) for row in rows]
                    raw_trials[protocol][str(batch_size)].append(
                        {
                            "round": round_index,
                            "trial": trial_index,
                            "batch_order_index": batch_order_index,
                            "protocol_order_index": protocol_order_index,
                            "rank_ms": rank_ms,
                            "worst_rank_ms": max(rank_ms),
                        }
                    )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
    ).strip()
    bound_file_sha256, bound_files_match_commit = _source_binding(commit)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "commit": commit,
            "script_sha256": _sha256(Path(__file__).read_bytes()),
            "bound_file_sha256": bound_file_sha256,
            "bound_files_match_commit": bound_files_match_commit,
        },
        "environment": {
            "backend": "nccl",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
            "devices": [
                {
                    "rank": int(result["rank"]),
                    **result["device"],
                }
                for result in rank_results
            ],
        },
        "config": {
            "world_size": world_size,
            "batch_sizes": list(BATCH_SIZES),
            "rounds": rounds,
            "trials_per_round": trials_per_round,
            "warmup": warmup,
            "timing": "CUDA events; aligned trial uses max across ranks",
            "binding_percentile": 0.95,
            "candidate_steady_p95_limit_ms": CANDIDATE_STEADY_P95_LIMIT_MS,
        },
        "protocols": {
            "rank0_broadcast": (
                "rank 0 contributes one fixed-layout int64 token packet; every "
                "follower's caller-owned packet is overwritten"
            ),
            "all_rank_equality": (
                "every rank contributes one int64 token packet; all_gather "
                "materializes every contribution and exact row equality is checked"
            ),
        },
        "logical_bytes": {
            str(batch): _logical_bytes(world_size, batch)
            for batch in BATCH_SIZES
        },
        "correctness": [
            {
                "rank": int(result["rank"]),
                **result["correctness"],
            }
            for result in rank_results
        ],
        "execution_order": [
            {
                "round": round_index,
                "batch_order": list(_rotated_batches(round_index)),
                "protocol_order_by_batch": {
                    str(batch): list(
                        _protocol_order(round_index, batch_order_index)
                    )
                    for batch_order_index, batch in enumerate(
                        _rotated_batches(round_index)
                    )
                },
            }
            for round_index in range(rounds)
        ],
        "raw_trials": raw_trials,
    }
    checks, summaries, diagnostics = verify_evidence(artifact)
    artifact["summaries"] = summaries
    artifact["diagnostics"] = diagnostics
    artifact["checks"] = checks
    artifact["gate_passed"] = all(checks.values())
    return artifact


def verify_evidence(
    artifact: dict[str, Any],
    *,
    verify_stored: bool = False,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    """Independently rebuild every binding result from raw aligned trials."""
    config = artifact["config"]
    world_size = int(config["world_size"])
    rounds = int(config["rounds"])
    trials_per_round = int(config["trials_per_round"])
    batch_sizes = tuple(int(value) for value in config["batch_sizes"])
    expected_samples = rounds * trials_per_round

    shape_ok = (
        artifact.get("schema_version") == SCHEMA_VERSION
        and world_size == REQUIRED_WORLD_SIZE
        and batch_sizes == BATCH_SIZES
        and rounds >= MIN_ROUNDS
        and trials_per_round >= MIN_TRIALS_PER_ROUND
        and int(config.get("warmup", -1)) >= MIN_WARMUP
        and float(config.get("binding_percentile", 0.0)) == 0.95
        and float(config.get("candidate_steady_p95_limit_ms", 0.0))
        == CANDIDATE_STEADY_P95_LIMIT_MS
    )
    source = artifact.get("source", {})
    source_ok = _verify_source_binding(source)
    devices = artifact.get("environment", {}).get("devices", [])
    environment_ok = (
        artifact.get("environment", {}).get("backend") == "nccl"
        and len(devices) == world_size
        and {int(device["rank"]) for device in devices}
        == set(range(world_size))
        and len({str(device.get("uuid")) for device in devices}) == world_size
        and len({str(device.get("pci")) for device in devices}) == world_size
        and all(
            isinstance(device.get("name"), str)
            and bool(device["name"])
            and isinstance(device.get("capability"), list)
            and len(device["capability"]) == 2
            and all(int(value) >= 0 for value in device["capability"])
            and isinstance(device.get("uuid"), str)
            and bool(device["uuid"])
            and isinstance(device.get("pci"), str)
            and bool(device["pci"])
            and int(device.get("total_memory_bytes", 0)) > 0
            for device in devices
        )
    )
    correctness_rows = artifact["correctness"]
    canonical = list(range(16, 32))
    equal_gathered = [list(canonical) for _ in range(world_size)]
    divergent_gathered = [list(canonical) for _ in range(world_size)]
    divergent_gathered[-1][0] += 1

    def probe_ok(row: dict[str, Any]) -> bool:
        rank = int(row["rank"])
        probe = row.get("probe", {})
        divergent_input = list(canonical)
        if rank == world_size - 1:
            divergent_input[0] += 1
        return (
            probe.get("canonical") == canonical
            and probe.get("broadcast_before")
            == (canonical if rank == 0 else [-77] * len(canonical))
            and probe.get("broadcast_after") == canonical
            and probe.get("equal_gathered") == equal_gathered
            and probe.get("divergent_input") == divergent_input
            and probe.get("divergent_gathered") == divergent_gathered
        )

    correctness_ok = (
        len(correctness_rows) == world_size
        and {int(row["rank"]) for row in correctness_rows}
        == set(range(world_size))
        and all(
            bool(row["broadcast_overwrite"])
            and bool(row["all_rank_equal_accepts"])
            and bool(row["all_rank_divergence_detected"])
            and set(row["measured_trials"]) == set(PROTOCOLS)
            and all(bool(value) for value in row["measured_trials"].values())
            and set(row.get("measured_mismatches", {})) == set(PROTOCOLS)
            and all(
                int(value) == 0
                for value in row["measured_mismatches"].values()
            )
            and probe_ok(row)
            for row in correctness_rows
        )
    )

    expected_orders = [
        {
            "round": round_index,
            "batch_order": list(_rotated_batches(round_index)),
            "protocol_order_by_batch": {
                str(batch): list(_protocol_order(round_index, batch_index))
                for batch_index, batch in enumerate(
                    _rotated_batches(round_index)
                )
            },
        }
        for round_index in range(rounds)
    ]
    order_ok = artifact["execution_order"] == expected_orders

    expected_bytes = {
        str(batch): _logical_bytes(world_size, batch) for batch in BATCH_SIZES
    }
    logical_bytes_ok = artifact["logical_bytes"] == expected_bytes

    raw_ok = (
        set(artifact["raw_trials"]) == set(PROTOCOLS)
        and all(
            set(artifact["raw_trials"][protocol])
            == {str(batch) for batch in BATCH_SIZES}
            for protocol in PROTOCOLS
        )
    )
    expected_grid = {
        (round_index, trial_index)
        for round_index in range(rounds)
        for trial_index in range(trials_per_round)
    }
    summaries: dict[str, dict[str, dict[str, float | int]]] = {
        protocol: {} for protocol in PROTOCOLS
    }
    for protocol in PROTOCOLS:
        for batch_size in BATCH_SIZES:
            rows = artifact["raw_trials"][protocol][str(batch_size)]
            if len(rows) != expected_samples:
                raw_ok = False
                continue
            seen: set[tuple[int, int]] = set()
            worst_values: list[float] = []
            for row in rows:
                round_index = int(row["round"])
                trial_index = int(row["trial"])
                seen.add((round_index, trial_index))
                rank_ms = [float(value) for value in row["rank_ms"]]
                if (
                    len(rank_ms) != world_size
                    or any(
                        not math.isfinite(value) or value <= 0
                        for value in rank_ms
                    )
                    or float(row["worst_rank_ms"]) != max(rank_ms)
                ):
                    raw_ok = False
                if (round_index, trial_index) not in expected_grid:
                    raw_ok = False
                    continue
                batch_order = _rotated_batches(round_index)
                expected_batch_index = batch_order.index(batch_size)
                expected_protocol_index = _protocol_order(
                    round_index, expected_batch_index
                ).index(protocol)
                if (
                    int(row["batch_order_index"]) != expected_batch_index
                    or int(row["protocol_order_index"])
                    != expected_protocol_index
                ):
                    raw_ok = False
                worst_values.append(float(row["worst_rank_ms"]))
            if seen != expected_grid:
                raw_ok = False
            summaries[protocol][str(batch_size)] = _summary(worst_values)

    candidate_steady_p95_ok = raw_ok and all(
        float(summaries["rank0_broadcast"][str(batch)]["p95_ms"])
        < CANDIDATE_STEADY_P95_LIMIT_MS
        for batch in BATCH_SIZES
    )
    diagnostics = {
        "relative_latency_is_diagnostic_only": True,
        "all_rank_equality_over_broadcast_p99": {
            str(batch): (
                float(summaries["all_rank_equality"][str(batch)]["p99_ms"])
                / float(summaries["rank0_broadcast"][str(batch)]["p99_ms"])
            )
            for batch in BATCH_SIZES
        },
        "candidate_trials_over_1ms": {
            str(batch): sum(
                float(row["worst_rank_ms"]) >= 1.0
                for row in artifact["raw_trials"]["rank0_broadcast"][str(batch)]
            )
            for batch in BATCH_SIZES
        },
        "jitter_policy": (
            "The candidate's fixed 1 ms p95 steady-state ceiling is binding. "
            "p99, max, and >=1 ms tail counts remain visible diagnostics, but "
            "are not a tiny-sample pass/fail threshold because a late host rank "
            "makes every aligned CUDA event include OS launch jitter. Relative "
            "latency is likewise diagnostic only."
        ),
    }
    checks = {
        "artifact_shape": shape_ok,
        "clean_source_and_script_binding": source_ok,
        "eight_rank_nccl_environment": environment_ok,
        "rank_protocol_correctness": correctness_ok,
        "rotating_execution_order": order_ok,
        "raw_trials_and_worst_rank_max": raw_ok,
        "exact_logical_bytes": logical_bytes_ok,
        "candidate_steady_p95_below_1ms": candidate_steady_p95_ok,
    }
    if verify_stored:
        stored_matches = (
            artifact.get("checks") == checks
            and artifact.get("summaries") == summaries
            and artifact.get("diagnostics") == diagnostics
            and artifact.get("gate_passed") == all(checks.values())
        )
        checks["stored_replay_matches"] = stored_matches
    return checks, summaries, diagnostics


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("TP sampling protocol evidence requires CUDA")
    if args.world_size != REQUIRED_WORLD_SIZE:
        raise ValueError(
            f"--world-size must be {REQUIRED_WORLD_SIZE} for the blocking gate"
        )
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"world_size={args.world_size} needs that many CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )
    if args.rounds < MIN_ROUNDS:
        raise ValueError(f"--rounds must be >= {MIN_ROUNDS}")
    if args.trials_per_round < MIN_TRIALS_PER_ROUND:
        raise ValueError(
            f"--trials-per-round must be >= {MIN_TRIALS_PER_ROUND}"
        )
    if args.warmup < MIN_WARMUP:
        raise ValueError(f"--warmup must be >= {MIN_WARMUP}")

    with tempfile.TemporaryDirectory(prefix="kairyu-tp-sampling-") as temp:
        temp_path = Path(temp)
        result_dir = temp_path / "ranks"
        result_dir.mkdir()
        context = mp.start_processes(
            _worker,
            args=(
                args.world_size,
                str(temp_path / "rdv"),
                str(result_dir),
                args.rounds,
                args.trials_per_round,
                args.warmup,
            ),
            nprocs=args.world_size,
            join=False,
            start_method="spawn",
        )
        deadline = time.monotonic() + args.timeout
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                for process in context.processes:
                    process.terminate()
                raise TimeoutError(
                    f"NCCL protocol benchmark timed out after {args.timeout}s"
                )
        results = [
            json.loads((result_dir / f"rank{rank}.json").read_text())
            for rank in range(args.world_size)
        ]
    return _aggregate(
        results,
        world_size=args.world_size,
        rounds=args.rounds,
        trials_per_round=args.trials_per_round,
        warmup=args.warmup,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", type=Path, help="verify an existing JSON artifact")
    mode.add_argument("--output", type=Path, help="write a new JSON artifact here")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--trials-per-round", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--assert-gate",
        action="store_true",
        help="exit non-zero unless every binding check passes",
    )
    args = parser.parse_args()
    if args.verify is None and args.output is None:
        parser.error("one of --output or --verify is required")
    return args


def main() -> None:
    args = _parse_args()
    if args.verify is not None:
        artifact = json.loads(args.verify.read_text())
        checks, summaries, diagnostics = verify_evidence(
            artifact, verify_stored=True
        )
        result = {
            "artifact": str(args.verify),
            "checks": checks,
            "summaries": summaries,
            "diagnostics": diagnostics,
            "gate_passed": all(checks.values()),
        }
    else:
        artifact = _measure(args)
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {
            "artifact": str(args.output),
            "checks": artifact["checks"],
            "summaries": artifact["summaries"],
            "diagnostics": artifact["diagnostics"],
            "gate_passed": artifact["gate_passed"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.assert_gate and not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
