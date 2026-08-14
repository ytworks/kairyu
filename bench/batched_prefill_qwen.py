"""Formal Qwen3-32B TP8 evidence for issue #224 batched prefill.

The binding gate is structural and parity-based, not a wall-clock threshold:
for one eight-request cold burst, every TP rank must reduce model prefill calls
from 8 to 1 and FlashInfer layer runs from 512 to 64 while preserving all eight
selected tokens. Rank-0 CUDA profiler event count must also fall. Timing is
recorded only as diagnostic evidence, so OS jitter cannot decide the verdict.

Measure:

    uv run python bench/batched_prefill_qwen.py \
      --model-path /models/qwen3-32b --tp 8 \
      --output bench/results/issue-224-batched-prefill-qwen3-32b-tp8.json \
      --assert-gate

Replay the artifact checks:

    uv run python bench/batched_prefill_qwen.py \
      --verify bench/results/issue-224-batched-prefill-qwen3-32b-tp8.json \
      --model-path /models/qwen3-32b --assert-gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.profiling import profile_scope

SCHEMA_VERSION = 1
MEASUREMENT_KIND = "qwen3-32b-tp-batched-prefill"
# Keep the complete B=8 group below the reviewed FlashInfer workspace's
# 1,024-query stress point.  The gate is structural (8 -> 1 model calls and
# 512 -> 64 layer runs), so larger prompts add memory risk but no evidence.
PROMPT_LENGTHS = (65, 81, 97, 113, 129, 145, 161, 177)
IMPLEMENTATION_FILES = (
    "evals/profiling.py",
    "kairyu/engine/core/comm.py",
    "kairyu/engine/core/dist_comm.py",
    "kairyu/engine/core/prefill.py",
    "kairyu/engine/core/kv_pool.py",
    "kairyu/engine/core/model_runner.py",
    "kairyu/engine/core/worker.py",
    "kairyu/engine/core/attention/__init__.py",
    "kairyu/engine/core/attention/flashinfer_gpu.py",
    "kairyu/engine/core/attention/torch_backend.py",
    "kairyu/models/attention.py",
    "kairyu/models/llama.py",
    "bench/batched_prefill_qwen.py",
)
_REPO = Path(__file__).resolve().parents[1]
_EXPECTED_CONFIG_SHA256 = "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
_EXPECTED_INDEX_SHA256 = "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
_EXPECTED_WEIGHTS_MANIFEST_SHA256 = (
    "1757cd21c98fbd6e6385910628098f396b9e5c9757a3423f5a280a5770264d8f"
)
_EXPECTED_WEIGHTS = (
    (
        "model-00001-of-00017.safetensors",
        3_957_109_648,
        "52562b2ff97b61764260273e71bf5b4cf8a66f569399398f26dec0300fcf1316",
    ),
    (
        "model-00002-of-00017.safetensors",
        3_900_791_760,
        "e26764b2c6878e3fb7198895fa833ec62838d84a19665e6abfbae43c6daf02b3",
    ),
    (
        "model-00003-of-00017.safetensors",
        3_900_791_760,
        "6c5ba7bed9c52bc121e75cbe8a7be46936d0006cc80f42a6d5886ed40b4c2a62",
    ),
    (
        "model-00004-of-00017.safetensors",
        3_900_791_800,
        "f736f6ac4d8c30866107fb1185a05b3c3cfce9717720082f466fa44e691bcec8",
    ),
    (
        "model-00005-of-00017.safetensors",
        3_900_791_800,
        "a52ed375c083209c54d42ac510afeb1fbb5af4f193be2dc7d103f665a0f212d3",
    ),
    (
        "model-00006-of-00017.safetensors",
        3_900_791_800,
        "37fae28990b0e4a70228549d040c0393e87bee3820d59e58e47844974d8dff5b",
    ),
    (
        "model-00007-of-00017.safetensors",
        3_900_791_800,
        "37776006aeaba29eca8bc73b2b963fe3477e1c2e3f6a27cb9527be75b905e1bf",
    ),
    (
        "model-00008-of-00017.safetensors",
        3_900_791_800,
        "73e74e9129674fe330948005075d70ab4fa0b92b68fb220c5a693f9cea553730",
    ),
    (
        "model-00009-of-00017.safetensors",
        3_900_791_800,
        "a044b3602a01bd8ea62ff51badf9cc038ab1d73d97399480e6a55b4c86fa7fa6",
    ),
    (
        "model-00010-of-00017.safetensors",
        3_900_791_800,
        "9966612ba7ecfc2cd2e592fb95224b86d743271ff88e172cc272a4b26382aa75",
    ),
    (
        "model-00011-of-00017.safetensors",
        3_900_791_800,
        "e2a058a0ac7d4b992b731c29221ecfb4b76b8a48d9004d0e8a62ba44f699845c",
    ),
    (
        "model-00012-of-00017.safetensors",
        3_900_791_800,
        "58a1aa89093fea07325f787072a468e3482a470ff4b7fe5ead5f749683907c40",
    ),
    (
        "model-00013-of-00017.safetensors",
        3_900_791_800,
        "35f3381bab31a23370c37d922290aeecdf603418336058fb86fe42d8f51ac40c",
    ),
    (
        "model-00014-of-00017.safetensors",
        3_900_791_800,
        "8713b062ddc178acf5917610b7f4b64eede833b2ea4aa37bd562dcf2f3a3339d",
    ),
    (
        "model-00015-of-00017.safetensors",
        3_900_791_800,
        "bec439d23931821a236d8f62fa79deecf5551bd25602278aa2ae0ce432b378cf",
    ),
    (
        "model-00016-of-00017.safetensors",
        3_900_791_800,
        "e569139fadd61fe7c8f9eb1c976d9a627cae48c57ddf228cfbd0593c59c64ff7",
    ),
    (
        "model-00017-of-00017.safetensors",
        3_055_341_992,
        "1f47c318fcd7797c0f85b4233cb754438b10e795b8bc874889090c416a94bd38",
    ),
)
_EXPECTED_WEIGHT_ROWS = [
    {"name": name, "size": size, "sha256": digest} for name, size, digest in _EXPECTED_WEIGHTS
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=_REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_show(commit: str, relative: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ("git", "show", f"{commit}:{relative}"),
            cwd=_REPO,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _ensure_python_bin_on_path() -> None:
    """Make FlashInfer's JIT helper visible beside the active interpreter."""
    path = os.environ.get("PATH", "")
    if shutil.which("ninja", path=path) is not None:
        return
    python_bin = Path(sys.executable).parent
    ninja = python_bin / "ninja"
    if os.access(ninja, os.X_OK):
        os.environ["PATH"] = os.pathsep.join(part for part in (str(python_bin), path) if part)


def _source_snapshot() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    files: dict[str, str] = {}
    files_match_commit = isinstance(commit, str)
    if isinstance(commit, str):
        for name in IMPLEMENTATION_FILES:
            current_path = _REPO / name
            committed = _git_show(commit, name)
            if committed is None or not current_path.is_file():
                files_match_commit = False
                continue
            current = current_path.read_bytes()
            files[name] = hashlib.sha256(committed).hexdigest()
            files_match_commit &= current == committed
    return {
        "commit": commit,
        "tracked_dirty": None if dirty is None else bool(dirty),
        "files": files,
        "files_match_commit": (files_match_commit and set(files) == set(IMPLEMENTATION_FILES)),
    }


def _hex_digest(value: object, *, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_snapshot_is_clean_and_bound(snapshot: object) -> bool:
    if not isinstance(snapshot, dict):
        return False
    commit = snapshot.get("commit")
    files = snapshot.get("files")
    if (
        not _hex_digest(commit, length=40)
        or snapshot.get("tracked_dirty") is not False
        or snapshot.get("files_match_commit") is not True
        or not isinstance(files, dict)
        or set(files) != set(IMPLEMENTATION_FILES)
        or any(not _hex_digest(digest) for digest in files.values())
    ):
        return False
    for name in IMPLEMENTATION_FILES:
        committed = _git_show(commit, name)
        current_path = _REPO / name
        if committed is None or not current_path.is_file():
            return False
        digest = hashlib.sha256(committed).hexdigest()
        if files[name] != digest or current_path.read_bytes() != committed:
            return False
    return True


def _source_gate(source: object) -> bool:
    if not isinstance(source, dict):
        return False
    start = source.get("measurement_start")
    end = source.get("measurement_end")
    return (
        source.get("stable_across_measurement") is True
        and start == end
        and _source_snapshot_is_clean_and_bound(start)
        and _source_snapshot_is_clean_and_bound(end)
    )


def _checkpoint_manifest(model_path: Path) -> dict[str, Any]:
    config = json.loads((model_path / "config.json").read_text())
    expected = {
        "architecture": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
    }
    if expected != {
        "architecture": ["Qwen3ForCausalLM"],
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
    }:
        raise RuntimeError(
            f"--model-path is not the reviewed Qwen3-32B checkpoint layout: {expected}"
        )
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors weights under {model_path}")
    with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
        digests = list(pool.map(_sha256, files))
    weights = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(files, digests, strict=True)
    ]
    index = model_path / "model.safetensors.index.json"
    return {
        "layout": expected,
        "config_sha256": _sha256(model_path / "config.json"),
        "index_sha256": _sha256(index) if index.exists() else None,
        "weights": weights,
        "weights_total_bytes": sum(row["size"] for row in weights),
        "weights_manifest_sha256": hashlib.sha256(
            json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _hardware() -> dict[str, Any]:
    import torch

    rows = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        rows.append(
            {
                "index": index,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "uuid": str(props.uuid),
                "pci": (
                    f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}"
                ),
                "total_memory_bytes": props.total_memory,
            }
        )
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": rows,
    }


def _prompt_tokens(index: int, length: int, vocab_size: int) -> tuple[int, ...]:
    # Stable, valid, non-special token IDs. The evidence stores the digest and
    # the first/last values rather than 3,848 integers of noise.
    modulus = vocab_size - 16
    return tuple(
        16 + ((index * 7919 + position * 131 + position * position * 17) % modulus)
        for position in range(length)
    )


def _requests(prefix: str, lengths: tuple[int, ...], vocab_size: int):
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest

    return [
        EngineRequest(
            request_id=f"{prefix}-{index}",
            prompt_token_ids=_prompt_tokens(index, length, vocab_size),
            max_new_tokens=1,
            sampling=EngineSampling(temperature=0.0),
        )
        for index, length in enumerate(lengths)
    ]


def _cuda_event_count(profile) -> int:
    import torch

    return sum(
        event.count
        for event in profile.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
    )


def _normalized_outputs(outputs: dict[str, tuple[int, ...]], prefix: str, count: int) -> list[int]:
    result = []
    for index in range(count):
        values = outputs.get(f"{prefix}-{index}", ())
        if len(values) != 1:
            raise RuntimeError(f"{prefix}-{index} produced {len(values)} tokens, expected 1")
        result.append(int(values[0]))
    return result


def _run_cell(
    runner,
    *,
    enabled: bool,
    prefix: str,
    lengths: tuple[int, ...],
    vocab_size: int,
    num_pages: int,
    page_size: int,
    profile: bool,
) -> dict[str, Any]:
    import torch

    from kairyu.engine.core.engine_core import token_ids
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler

    runner.set_batched_prefill_enabled(enabled)
    runner.prefill_execution_stats(reset=True)
    scheduler = Scheduler(
        RadixKVCache(num_pages=num_pages, page_size=page_size),
        max_num_batched_tokens=sum(lengths),
        max_num_seqs=len(lengths),
        page_size=page_size,
    )
    requests = _requests(prefix, lengths, vocab_size)
    for request in requests:
        scheduler.add_request(request)
    step = scheduler.schedule()
    chunks = step.scheduled
    schedule = [
        {
            "request_id": chunk.request_id,
            "num_tokens": chunk.num_tokens,
            "is_prefill": chunk.is_prefill,
        }
        for chunk in chunks
    ]
    if (
        len(chunks) != len(lengths)
        or any(not chunk.is_prefill for chunk in chunks)
        or tuple(chunk.num_tokens for chunk in chunks) != lengths
    ):
        raise RuntimeError(f"workload did not form one complete prefill group: {schedule}")

    profiler = None
    wall_start = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if profile:
        with profile_scope(
            enabled=True,
            activities=("cpu", "cuda"),
            acc_events=True,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            sampled = runner.execute(chunks, scheduler.states)
    else:
        sampled = runner.execute(chunks, scheduler.states)
    end.record()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall_start
    scheduler.update(token_ids(sampled))
    outputs = {
        request.request_id: tuple(scheduler.output_tokens(request.request_id))
        for request in requests
    }
    for request in requests:
        runner.release(request.request_id)
    stats = runner.prefill_execution_stats()
    return {
        "enabled": enabled,
        "schedule": schedule,
        "outputs": _normalized_outputs(outputs, prefix, len(lengths)),
        "rank_stats": list(stats),
        "rank0_cuda_events": (_cuda_event_count(profiler) if profiler is not None else None),
        "diagnostic": {
            "rank0_cuda_elapsed_ms": start.elapsed_time(end),
            "wall_s": wall_s,
        },
    }


def _backend_counts(rank_row: dict[str, Any]) -> tuple[int, int]:
    backend = rank_row["stats"].get("backend")
    if (
        not isinstance(backend, list)
        or len(backend) != 1
        or backend[0].get("type") != "FlashInferBackend"
    ):
        return -1, -1
    return int(backend[0].get("plans", -1)), int(backend[0].get("runs", -1))


def _schedule_gate(cell: object, *, prefix: str, lengths: tuple[int, ...]) -> bool:
    if not isinstance(cell, dict):
        return False
    schedule = cell.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != len(lengths):
        return False
    return all(
        isinstance(row, dict)
        and set(row) == {"request_id", "num_tokens", "is_prefill"}
        and row["request_id"] == f"{prefix}-{index}"
        and row["num_tokens"] == length
        and row["is_prefill"] is True
        for index, (row, length) in enumerate(zip(schedule, lengths, strict=True))
    )


def _checkpoint_snapshot_gate(checkpoint: object) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    if checkpoint.get("layout") != {
        "architecture": ["Qwen3ForCausalLM"],
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
    }:
        return False
    weights = checkpoint.get("weights")
    if weights != _EXPECTED_WEIGHT_ROWS:
        return False
    digest = hashlib.sha256(
        json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        digest == _EXPECTED_WEIGHTS_MANIFEST_SHA256
        and checkpoint.get("weights_manifest_sha256") == _EXPECTED_WEIGHTS_MANIFEST_SHA256
        and checkpoint.get("weights_total_bytes") == 65_524_328_560
        and checkpoint.get("config_sha256") == _EXPECTED_CONFIG_SHA256
        and checkpoint.get("index_sha256") == _EXPECTED_INDEX_SHA256
    )


def _checkpoint_gate(checkpoint: object) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    start = checkpoint.get("measurement_start")
    end = checkpoint.get("measurement_end")
    return (
        checkpoint.get("stable_across_measurement") is True
        and start == end
        and _checkpoint_snapshot_gate(start)
        and _checkpoint_snapshot_gate(end)
    )


def _prompt_gate(rows: object, lengths: tuple[int, ...], vocab_size: int) -> bool:
    if not isinstance(rows, list) or len(rows) != len(lengths):
        return False
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        if not isinstance(row, dict):
            return False
        tokens = _prompt_tokens(index, length, vocab_size)
        digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        if row != {
            "length": length,
            "first": list(tokens[:4]),
            "last": list(tokens[-4:]),
            "sha256": digest,
        }:
            return False
    return True


def _rank_stats_gate(
    rows: object,
    *,
    tp: int,
    enabled: bool,
    layers: int,
    request_count: int,
) -> bool:
    if not isinstance(rows, list) or len(rows) != tp:
        return False
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("rank") != rank
            or row.get("world_size") != tp
            or row.get("device") != f"cuda:{rank}"
            or not isinstance(row.get("stats"), dict)
        ):
            return False
        stats = row["stats"]
        if type(stats.get("enabled")) is not bool:
            return False
        expected = (
            {
                "enabled": True,
                "rows": request_count,
                "model_calls": 1,
                "batched_groups": 1,
                "sequential_rows": 0,
            }
            if enabled
            else {
                "enabled": False,
                "rows": request_count,
                "model_calls": request_count,
                "batched_groups": 0,
                "sequential_rows": request_count,
            }
        )
        if any(stats.get(key) != value for key, value in expected.items()):
            return False
        if stats.get("capability_gap") is not None:
            return False
        plans, runs = _backend_counts(row)
        if enabled:
            if (plans, runs) != (1, layers):
                return False
        elif (plans, runs) != (request_count, request_count * layers):
            return False
    return True


def _topology_gate(rows: object, tp: int) -> bool:
    if (
        not isinstance(rows, list)
        or len(rows) != tp
        or any(not isinstance(row, dict) for row in rows)
    ):
        return False
    return all(
        row.get("rank") == rank
        and row.get("control_world_size") == tp
        and row.get("model_world_size") == tp
        and row.get("control_backend") == "gloo"
        and row.get("model_backend") == "nccl"
        and row.get("device") == f"cuda:{rank}"
        and type(row.get("sampling_owner")) is bool
        and type(row.get("sampler_present")) is bool
        and row.get("sampling_owner") == (rank == 0)
        and row.get("sampler_present") == (rank == 0)
        for rank, row in enumerate(rows)
    )


def _hardware_gate(hardware: object) -> bool:
    if not isinstance(hardware, dict):
        return False
    devices = hardware.get("devices")
    if not isinstance(devices, list) or len(devices) != 8:
        return False
    uuids = {row.get("uuid") for row in devices if isinstance(row, dict)}
    pci_ids = {row.get("pci") for row in devices if isinstance(row, dict)}
    return (
        len(uuids) == 8
        and None not in uuids
        and len(pci_ids) == 8
        and None not in pci_ids
        and all(
            isinstance(row, dict)
            and set(row)
            == {
                "index",
                "name",
                "compute_capability",
                "uuid",
                "pci",
                "total_memory_bytes",
            }
            and row["index"] == index
            and row["name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
            and row["compute_capability"] == "12.0"
            and isinstance(row["uuid"], str)
            and bool(row["uuid"])
            and isinstance(row["pci"], str)
            and bool(row["pci"])
            and type(row["total_memory_bytes"]) is int
            and row["total_memory_bytes"] >= 100_000_000_000
            for index, row in enumerate(devices)
        )
    )


def _evaluate(payload: dict[str, Any]) -> dict[str, bool]:
    baseline = payload.get("baseline", {})
    candidate = payload.get("candidate", {})
    config = payload.get("config", {})
    tp = config.get("tp")
    checkpoint = payload.get("checkpoint", {})
    checkpoint_start = (
        checkpoint.get("measurement_start", {}) if isinstance(checkpoint, dict) else {}
    )
    layers = checkpoint_start.get("layout", {}).get("num_hidden_layers")
    count = len(config.get("prompt_lengths", ()))
    lengths = tuple(config.get("prompt_lengths", ()))
    outputs = baseline.get("outputs")
    candidate_outputs = candidate.get("outputs")
    return {
        "schema_and_kind": (
            payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("measurement_kind") == MEASUREMENT_KIND
        ),
        "qwen3_32b_tp8": (tp == 8 and layers == 64 and _checkpoint_gate(checkpoint)),
        "clean_committed_source_stable": _source_gate(payload.get("source")),
        "one_group_geometry": (
            lengths == PROMPT_LENGTHS
            and _schedule_gate(baseline, prefix="baseline", lengths=PROMPT_LENGTHS)
            and _schedule_gate(candidate, prefix="candidate", lengths=PROMPT_LENGTHS)
            and config.get("max_new_tokens") == 1
            and config.get("num_pages") == 4096
            and config.get("page_size") == 16
            and config.get("timing_is_diagnostic") is True
        ),
        "all_rank_baseline_counts": _rank_stats_gate(
            baseline.get("rank_stats"),
            tp=tp,
            enabled=False,
            layers=layers,
            request_count=count,
        )
        if type(tp) is int and type(layers) is int
        else False,
        "all_rank_candidate_counts": _rank_stats_gate(
            candidate.get("rank_stats"),
            tp=tp,
            enabled=True,
            layers=layers,
            request_count=count,
        )
        if type(tp) is int and type(layers) is int
        else False,
        "eight_first_tokens_exact": (
            isinstance(outputs, list)
            and len(outputs) == 8
            and all(type(token) is int and 0 <= token < 151936 for token in outputs)
            and outputs == candidate_outputs
        ),
        "rank0_cuda_events_reduced": (
            type(baseline.get("rank0_cuda_events")) is int
            and type(candidate.get("rank0_cuda_events")) is int
            and 0 < candidate["rank0_cuda_events"] < baseline["rank0_cuda_events"]
        ),
        "tp_topology_complete": _topology_gate(payload.get("topology"), tp)
        if type(tp) is int
        else False,
        "prompt_provenance_complete": _prompt_gate(payload.get("prompts"), PROMPT_LENGTHS, 151936),
        "hardware_profile_complete": _hardware_gate(payload.get("hardware")),
    }


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from kairyu.engine.core.worker import DistTPLauncher

    _ensure_python_bin_on_path()
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.tp:
        raise RuntimeError(f"tp={args.tp} requires {args.tp} visible CUDA devices")
    if args.tp != 8 or args.num_pages != 4096 or args.page_size != 16:
        raise RuntimeError("formal #224 evidence requires --tp 8 --num-pages 4096 --page-size 16")
    source_start = _source_snapshot()
    if not _source_snapshot_is_clean_and_bound(source_start):
        raise RuntimeError("formal evidence requires a clean commit containing every bound source")
    hardware = _hardware()
    if not _hardware_gate(hardware):
        raise RuntimeError(
            "formal evidence requires exactly eight visible RTX PRO 6000 "
            "Blackwell Server Edition GPUs"
        )
    model_path = Path(args.model_path).resolve()
    checkpoint_start = _checkpoint_manifest(model_path)
    if not _checkpoint_snapshot_gate(checkpoint_start):
        raise RuntimeError("--model-path is not the exact reviewed Qwen3-32B checkpoint")
    vocab_size = checkpoint_start["layout"]["vocab_size"]
    launcher = DistTPLauncher(
        str(model_path),
        tp=args.tp,
        num_pages=args.num_pages,
        page_size=args.page_size,
        vocab=[],
    )
    try:
        runner = launcher.runner
        # Compile both structural paths before the profiled measurement.
        _run_cell(
            runner,
            enabled=False,
            prefix="warm-sequential",
            lengths=(33, 49),
            vocab_size=vocab_size,
            num_pages=args.num_pages,
            page_size=args.page_size,
            profile=False,
        )
        _run_cell(
            runner,
            enabled=True,
            prefix="warm-batched",
            lengths=(33, 49),
            vocab_size=vocab_size,
            num_pages=args.num_pages,
            page_size=args.page_size,
            profile=False,
        )
        baseline = _run_cell(
            runner,
            enabled=False,
            prefix="baseline",
            lengths=PROMPT_LENGTHS,
            vocab_size=vocab_size,
            num_pages=args.num_pages,
            page_size=args.page_size,
            profile=True,
        )
        candidate = _run_cell(
            runner,
            enabled=True,
            prefix="candidate",
            lengths=PROMPT_LENGTHS,
            vocab_size=vocab_size,
            num_pages=args.num_pages,
            page_size=args.page_size,
            profile=True,
        )
        topology = list(runner.sampling_ownership_metadata())
    finally:
        launcher.shutdown()

    checkpoint_end = _checkpoint_manifest(model_path)
    checkpoint = {
        "measurement_start": checkpoint_start,
        "measurement_end": checkpoint_end,
        "stable_across_measurement": checkpoint_start == checkpoint_end,
    }
    if not _checkpoint_gate(checkpoint):
        raise RuntimeError("checkpoint content changed while formal evidence was running")
    source_end = _source_snapshot()
    source = {
        "measurement_start": source_start,
        "measurement_end": source_end,
        "stable_across_measurement": source_start == source_end,
    }
    if not _source_gate(source):
        raise RuntimeError("bound source or commit changed while formal evidence was running")
    prompt_rows = [
        {
            "length": length,
            "first": list(_prompt_tokens(index, length, vocab_size)[:4]),
            "last": list(_prompt_tokens(index, length, vocab_size)[-4:]),
            "sha256": hashlib.sha256(
                json.dumps(
                    _prompt_tokens(index, length, vocab_size),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        for index, length in enumerate(PROMPT_LENGTHS)
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": MEASUREMENT_KIND,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "tp": args.tp,
            "num_pages": args.num_pages,
            "page_size": args.page_size,
            "prompt_lengths": list(PROMPT_LENGTHS),
            "max_new_tokens": 1,
            "timing_is_diagnostic": True,
        },
        "source": source,
        "checkpoint": checkpoint,
        "hardware": hardware,
        "prompts": prompt_rows,
        "topology": topology,
        "baseline": baseline,
        "candidate": candidate,
    }
    payload["checks"] = _evaluate(payload)
    payload["passed"] = all(payload["checks"].values())
    return payload


def _verify(artifact: Path, model_path: Path | None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text())
    raw_checks = _evaluate(payload)
    stored_matches = payload.get("checks") == raw_checks and payload.get("passed") == all(
        raw_checks.values()
    )
    checks = dict(raw_checks)
    checks["stored_replay_matches"] = stored_matches
    checks["source_commit_replay"] = _source_gate(payload.get("source"))
    checks["hardware_replay"] = _hardware_gate(
        payload.get("hardware")
    ) and _hardware() == payload.get("hardware")
    if model_path is not None:
        replay = _checkpoint_manifest(model_path.resolve())
        checkpoint = payload.get("checkpoint", {})
        checks["checkpoint_replay"] = (
            isinstance(checkpoint, dict)
            and replay == checkpoint.get("measurement_start")
            and replay == checkpoint.get("measurement_end")
            and _checkpoint_snapshot_gate(replay)
        )
    else:
        checks["checkpoint_replay"] = False
    return {
        "artifact": str(artifact),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--num-pages", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args()

    if args.verify is not None:
        result = _verify(
            args.verify,
            Path(args.model_path) if args.model_path else None,
        )
    else:
        if not args.model_path or args.output is None:
            parser.error("measurement requires --model-path and --output")
        result = _measure(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.assert_gate and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
