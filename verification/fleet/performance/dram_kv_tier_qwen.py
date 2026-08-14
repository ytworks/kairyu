#!/usr/bin/env python3
"""Formal Qwen3-32B DRAM-KV crossover evidence for G5 F4a / issue #187.

The live operator records one raw shard per tensor-parallel degree.  This
module owns the fixed experiment schedule and the offline evidence contract:

* TP4 and TP8 are measured in separate, sequential runs;
* prefix lengths 16..8192 use one excluded warm-up pair and nine measured
  AB/BA pairs;
* restore is compared with a real production-model prefill from the same
  empty-device-page boundary; and
* the published crossover is the first member of a stable measured suffix,
  never an interpolated estimate.

``assemble`` seals raw TP4/TP8 JSONL into a replay-derived manifest and one
small runtime policy profile per TP degree.  ``verify`` checks the retained
files against raw replay.  ``replay`` needs only the two raw shards.  The
runtime policy files deliberately contain no bulky provenance; their
``evidence_sha256`` fields bind the complete raw artifact and manifest.

``run`` launches one production ``build_tp_model`` SPMD group, constructs a
``PagedKVPool`` and ``DRAMKVTier`` on every rank, and emits one TP4 or TP8 raw
shard.  Keeping CUDA/model imports inside that live path makes the offline
evidence commands safe on hosts without the checkpoint or GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import ssl
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "kairyu.g5.f4a.dram-kv-tier.v2"
PROFILE_SCHEMA_VERSION = "kairyu.dram-kv-crossover.v2"
MEASUREMENT_KIND = "real-qwen3-32b-dram-kv-tier-crossover"
MODEL = "qwen3-32b"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
TP_DEGREES = (4, 8)
PREFIX_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
PAGE_SIZE = 16
NUM_LAYERS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = "torch.bfloat16"
DTYPE_BYTES = 2
KV_PLANES = 2
MAX_PREFIX_PAGES = max(PREFIX_LENGTHS) // PAGE_SIZE
LIVE_POOL_PAGES = MAX_PREFIX_PAGES * 2
LIVE_TIER_CAPACITY_PAGES = MAX_PREFIX_PAGES
WARMUP_PAIRS = 1
MEASURED_REPEATS = 9
MIN_RESTORE_WINS = 8
SIGN_TEST_P_AT_MIN_WINS = 10 / 512
ARMS = ("restore", "recompute")
RESTORE_PATH = "production-radix-disttp-gloo"
CONTROL_BACKEND = "gloo"
TIER_BACKEND = "cuda-pinned-dram-fragment-major-torch-copy-v1"
RAW_NAMES = {
    4: "dram-kv-tier-qwen3-32b-tp4-raw.jsonl",
    8: "dram-kv-tier-qwen3-32b-tp8-raw.jsonl",
}
PROFILE_NAMES = {
    4: "dram-kv-tier-qwen3-32b-tp4-profile.json",
    8: "dram-kv-tier-qwen3-32b-tp8-profile.json",
}
MANIFEST_NAME = "dram-kv-tier-qwen3-32b-manifest.json"
SOURCE_PATH = "verification/fleet/performance/dram_kv_tier_qwen.py"
LEGACY_SOURCE_PATH = "bench/dram_kv_tier_qwen.py"
EXPECTED_MODEL_WEIGHTS_ROLLUP = "6aa37b7da4e37d45b277d0bca47b00c2bfb58bb60986ba93e4c3e667df123955"
EXPECTED_MODEL_CONFIG_SHA256 = "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
REQUIRED_SOURCE_PATHS = frozenset(
    {
        SOURCE_PATH,
        "Dockerfile.cuda",
        "kairyu/engine/core/attention_selector.py",
        "kairyu/engine/core/dist_comm.py",
        "kairyu/engine/core/kv_tier.py",
        "kairyu/engine/core/kv_tier_policy.py",
        "kairyu/engine/core/kv_pool.py",
        "kairyu/engine/core/model_runner.py",
        "kairyu/engine/core/numa.py",
        "kairyu/engine/core/radix_kv.py",
        "kairyu/engine/core/scheduler.py",
        "kairyu/engine/core/worker.py",
        "pyproject.toml",
        "uv.lock",
    }
)
_HEX = frozenset("0123456789abcdef")


class EvidenceError(ValueError):
    """Raw or retained evidence violates the formal F4a contract."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode())


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
        and value == value.lower()
        and set(value) <= _HEX
    )


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and set(value) <= _HEX
    )


def _exact_int(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    _require(type(value) is int, f"{name} must be an integer")
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{name} must be >= {minimum}")
    return result


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    _require(
        type(value) in {int, float},
        f"{name} must be a finite number",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be a finite number")
    if minimum is not None:
        _require(result >= minimum, f"{name} must be >= {minimum}")
    return result


def expected_rank_bytes(tp_size: int, prefix_tokens: int) -> int:
    _require(tp_size in TP_DEGREES, f"unsupported TP degree: {tp_size}")
    _require(prefix_tokens in PREFIX_LENGTHS, "prefix is outside the fixed grid")
    _require(NUM_KV_HEADS % tp_size == 0, "KV heads do not divide TP degree")
    local_kv_heads = NUM_KV_HEADS // tp_size
    return NUM_LAYERS * prefix_tokens * local_kv_heads * HEAD_DIM * KV_PLANES * DTYPE_BYTES


def formal_config(tp_size: int) -> dict[str, object]:
    _require(tp_size in TP_DEGREES, f"unsupported TP degree: {tp_size}")
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": MEASUREMENT_KIND,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tensor_parallel_size": tp_size,
        "prefix_lengths": list(PREFIX_LENGTHS),
        "page_size": PAGE_SIZE,
        "num_layers": NUM_LAYERS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "dtype": DTYPE,
        "warmup_pairs_per_cell": WARMUP_PAIRS,
        "measured_pairs_per_cell": MEASURED_REPEATS,
        "arm_order": "repeat-even=restore/recompute; repeat-odd=recompute/restore",
        "prefix_order": "repeat-even=ascending; repeat-odd=descending",
        "primary_metric": (
            "rank0 controller empty-pages-to-next-token-ready wall_ns including "
            "checksum, Gloo control, H2D/recompute, Radix publication, and model/output"
        ),
        "recompute_baseline": (
            "production cold prefill includes the final prompt token in its final "
            "chunk and samples directly from that hidden state"
        ),
        "win_rule": {
            "median_paired_restore_over_recompute_lt": 1.0,
            "minimum_restore_wins": MIN_RESTORE_WINS,
            "pairs": MEASURED_REPEATS,
            "one_sided_sign_test_p_at_minimum": SIGN_TEST_P_AT_MIN_WINS,
        },
        "crossover_rule": "first passing member of a stable measured suffix",
        "interpolation": False,
    }


def expected_schedule() -> tuple[dict[str, object], ...]:
    """Return the binding row schedule following the run header."""

    result: list[dict[str, object]] = []
    for repeat in (-1, *range(MEASURED_REPEATS)):
        warmup = repeat == -1
        prefixes = PREFIX_LENGTHS if warmup or repeat % 2 == 0 else tuple(reversed(PREFIX_LENGTHS))
        arm_order = ARMS if warmup or repeat % 2 == 0 else tuple(reversed(ARMS))
        for prefix_tokens in prefixes:
            common = {
                "prefix_tokens": prefix_tokens,
                "pair_index": repeat,
                "warmup": warmup,
                "pair_order": list(arm_order),
            }
            result.append({"row_type": "snapshot", **common})
            result.extend({"row_type": "sample", "arm": arm, **common} for arm in arm_order)
    return tuple(result)


def _median_absolute_deviation(values: Sequence[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def _sign_test_p_value(wins: int, pairs: int = MEASURED_REPEATS) -> float:
    _require(0 <= wins <= pairs, "sign-test wins are out of range")
    return sum(math.comb(pairs, count) for count in range(wins, pairs + 1)) / (2**pairs)


def _validate_descriptor(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} must be an object")
    descriptor = dict(value)
    _require(
        set(descriptor) == {"value", "sha256"},
        f"{name} must contain value and sha256",
    )
    _require(_valid_sha256(descriptor["sha256"]), f"{name} SHA-256 is invalid")
    _require(
        descriptor["sha256"] == sha256_json(descriptor["value"]),
        f"{name} SHA-256 does not match its value",
    )
    return descriptor


def _validate_source(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} must be an object")
    source = dict(value)
    _require(_valid_commit(source.get("commit")), f"{name} commit is invalid")
    _require(source.get("dirty") is False, f"{name} must be a clean commit")
    _require(source.get("source_kind") == "git", f"{name} must be verified by git")
    script_path = source.get("script_path")
    _require(
        script_path in {SOURCE_PATH, LEGACY_SOURCE_PATH},
        f"{name} script path differs",
    )
    bound = source.get("bound_file_sha256")
    _require(isinstance(bound, dict), f"{name} bound files must be an object")
    _require(
        (REQUIRED_SOURCE_PATHS - {SOURCE_PATH} | {script_path}) <= set(bound),
        f"{name} omits a required source path",
    )
    _require(
        all(isinstance(path, str) and _valid_sha256(digest) for path, digest in bound.items()),
        f"{name} contains an invalid bound source digest",
    )
    kairyu_python_paths = source.get("kairyu_python_paths")
    _require(
        isinstance(kairyu_python_paths, list)
        and kairyu_python_paths
        and all(
            isinstance(path, str) and path.startswith("kairyu/") and path.endswith(".py")
            for path in kairyu_python_paths
        ),
        f"{name} Kairyu source path inventory is invalid",
    )
    _require(
        len(kairyu_python_paths) == len(set(kairyu_python_paths)),
        f"{name} Kairyu source path inventory contains duplicates",
    )
    _require(
        set(kairyu_python_paths)
        == {
            path
            for path in bound
            if isinstance(path, str) and path.startswith("kairyu/") and path.endswith(".py")
        },
        f"{name} does not bind its full recorded Kairyu Python inventory",
    )
    _require(
        source.get("bound_files_sha256") == sha256_json(dict(sorted(bound.items()))),
        f"{name} bound-source rollup differs",
    )
    _require(
        source.get("script_sha256") == bound.get(script_path),
        f"{name} script digest differs from the bound tree",
    )
    return source


def _validate_checkpoint(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} must be an object")
    checkpoint = dict(value)
    _require(
        checkpoint.get("model") == MODEL and checkpoint.get("revision") == MODEL_REVISION,
        f"{name} is not the pinned Qwen3-32B revision",
    )
    architecture = checkpoint.get("architecture")
    expected_architecture = {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 64,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": 151936,
    }
    _require(
        architecture == expected_architecture,
        f"{name} Qwen3-32B architecture differs",
    )
    weights = checkpoint.get("weights")
    _require(
        isinstance(weights, list) and len(weights) == 17,
        f"{name} must bind all 17 weight shards",
    )
    seen: set[str] = set()
    for index, item in enumerate(weights):
        _require(isinstance(item, dict), f"{name} weight {index} is invalid")
        path = item.get("file")
        _require(
            isinstance(path, str) and path.endswith(".safetensors") and path not in seen,
            f"{name} weight {index} path is invalid",
        )
        seen.add(path)
        _exact_int(item.get("bytes"), f"{name} weight {index} bytes", minimum=1)
        _require(
            _valid_sha256(item.get("sha256")),
            f"{name} weight {index} SHA-256 is invalid",
        )
    _require(
        checkpoint.get("weights_sha256") == sha256_json(weights),
        f"{name} weight rollup differs",
    )
    _require(
        checkpoint.get("pinned_weights_rollup") == EXPECTED_MODEL_WEIGHTS_ROLLUP,
        f"{name} does not match the pinned Qwen3-32B weight identity",
    )
    metadata = checkpoint.get("metadata_sha256")
    _require(
        isinstance(metadata, dict)
        and {"config.json", "tokenizer.json"} <= set(metadata)
        and all(_valid_sha256(digest) for digest in metadata.values()),
        f"{name} metadata digests are incomplete",
    )
    _require(
        metadata["config.json"] == EXPECTED_MODEL_CONFIG_SHA256,
        f"{name} config.json does not match the pinned Qwen3-32B revision",
    )
    _require(
        _valid_sha256(checkpoint.get("model_config_sha256")),
        f"{name} canonical model-config digest is invalid",
    )
    return checkpoint


def _validate_runtime(value: object, tp_size: int) -> dict[str, object]:
    from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY

    _require(isinstance(value, dict), "runtime must be an object")
    runtime = dict(value)
    _require(
        runtime.get("tensor_parallel_size") == tp_size,
        "runtime TP degree differs",
    )
    _require(runtime.get("page_size") == PAGE_SIZE, "runtime page size differs")
    _require(
        _valid_sha256(runtime.get("engine_source_sha256")),
        "runtime engine source rollup is invalid",
    )
    _require(runtime.get("kv_dtype") == DTYPE, "runtime KV dtype differs")
    _require(
        runtime.get("tier_backend") == TIER_BACKEND,
        "runtime did not use the coalesced pinned-DRAM CUDA tier",
    )
    _require(
        runtime.get("recompute_path") == "production-radix-model-prefill"
        and runtime.get("recompute_cache_disabled") is True,
        "recompute must be an uncached production-model prefill",
    )
    _require(
        runtime.get("restore_path") == RESTORE_PATH
        and runtime.get("control_backend") == CONTROL_BACKEND,
        "restore must use the production Radix/DistTP Gloo control path",
    )
    _require(
        isinstance(runtime.get("attention_backend"), str) and bool(runtime["attention_backend"]),
        "runtime attention backend is missing",
    )
    backend_identity = runtime.get("attention_backend_identity")
    _require(
        isinstance(backend_identity, str) and bool(backend_identity),
        "runtime attention backend identity is missing",
    )
    try:
        parsed_backend_identity = json.loads(backend_identity)
    except json.JSONDecodeError as error:
        raise EvidenceError("runtime attention backend identity is not JSON") from error
    _require(
        isinstance(parsed_backend_identity, dict)
        and set(parsed_backend_identity)
        == {
            "requested",
            "resolved",
            "source",
            "components",
            "component_versions",
            "component_builds",
            "architecture",
        },
        "runtime attention backend identity fields are invalid",
    )
    _require(
        all(
            isinstance(parsed_backend_identity.get(field), str)
            and bool(parsed_backend_identity[field])
            for field in ("requested", "resolved", "source")
        )
        and isinstance(parsed_backend_identity.get("components"), dict)
        and bool(parsed_backend_identity["components"])
        and all(
            isinstance(name, str)
            and bool(name)
            and isinstance(component, str)
            and bool(component)
            for name, component in parsed_backend_identity["components"].items()
        )
        and isinstance(parsed_backend_identity.get("architecture"), dict),
        "runtime attention backend identity values are invalid",
    )
    component_versions = parsed_backend_identity.get("component_versions")
    _require(
        isinstance(component_versions, dict)
        and set(component_versions) == {"prefill", "decode"}
        and all(
            isinstance(component, str)
            and bool(component)
            and component != "unknown"
            for component in component_versions.values()
        ),
        "runtime attention backend component versions are invalid",
    )
    component_builds = parsed_backend_identity.get("component_builds")
    jit_cache = (
        component_builds.get("flashinfer-jit-cache")
        if isinstance(component_builds, dict)
        else None
    )
    _require(
        isinstance(component_builds, dict)
        and set(component_builds) == {"flashinfer-jit-cache"}
        and isinstance(jit_cache, dict)
        and set(jit_cache) == {"installed", "version", "record_sha256"}
        and jit_cache["installed"] is True
        and isinstance(jit_cache["version"], str)
        and bool(jit_cache["version"])
        and _valid_sha256(jit_cache["record_sha256"]),
        "runtime FlashInfer AOT build identity is invalid",
    )
    _require(
        json.dumps(
            parsed_backend_identity,
            sort_keys=True,
            separators=(",", ":"),
        )
        == backend_identity,
        "runtime attention backend identity is not canonical",
    )
    _exact_int(
        runtime.get("max_model_len"),
        "runtime max_model_len",
        minimum=max(PREFIX_LENGTHS) + 1,
    )
    _exact_int(
        runtime.get("max_num_batched_tokens"),
        "runtime max_num_batched_tokens",
        minimum=PAGE_SIZE,
    )
    _require(
        runtime.get("host_memory_placement_policy")
        == GPU_LOCAL_FIRST_TOUCH_POLICY,
        "runtime host-memory placement policy differs",
    )
    _exact_int(
        runtime.get("os_page_size"),
        "runtime OS page size",
        minimum=1,
    )
    _require(
        isinstance(runtime.get("pcie_max_link_speed"), str)
        and bool(runtime["pcie_max_link_speed"]),
        "runtime PCIe maximum link speed is missing",
    )
    for field, label in (
        ("pcie_max_link_width", "PCIe maximum link width"),
        ("pcie_current_link_width", "PCIe current link width"),
        ("local_numa_gpu_count", "local NUMA GPU count"),
    ):
        _exact_int(runtime.get(field), f"runtime {label}", minimum=1)
    _require(
        all(
            isinstance(runtime.get(field), str) and bool(runtime[field])
            for field in ("cpu_model", "python_version", "openssl_version")
        ),
        "runtime CPU/Python/OpenSSL identity is missing",
    )
    _require(
        type(runtime.get("chunked_prefill")) is bool,
        "runtime chunked_prefill must be recorded",
    )
    _require(
        _valid_sha256(runtime.get("kv_layout_fingerprint")),
        "runtime KV-layout fingerprint is invalid",
    )
    rank_fingerprints = runtime.get("rank_pool_fingerprints")
    _require(
        isinstance(rank_fingerprints, dict)
        and set(rank_fingerprints) == {str(rank) for rank in range(tp_size)}
        and all(_valid_sha256(value) for value in rank_fingerprints.values()),
        "runtime rank-pool fingerprints are incomplete",
    )
    _require(
        isinstance(runtime.get("torch_version"), str)
        and bool(runtime["torch_version"])
        and isinstance(runtime.get("torch_cuda_version"), str)
        and bool(runtime["torch_cuda_version"])
        and isinstance(runtime.get("nccl_version"), str)
        and bool(runtime["nccl_version"])
        and isinstance(runtime.get("driver_version"), str)
        and bool(runtime["driver_version"]),
        "runtime torch/CUDA/NCCL/driver versions are missing",
    )
    return runtime


def _validate_hardware(value: object, tp_size: int) -> dict[str, object]:
    from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY

    _require(isinstance(value, dict), "hardware must be an object")
    hardware = dict(value)
    ranks = hardware.get("ranks")
    _require(
        isinstance(ranks, list) and len(ranks) == tp_size,
        "hardware rank inventory differs from TP degree",
    )
    for expected_rank, item in enumerate(ranks):
        _require(isinstance(item, dict), "hardware rank row is invalid")
        _require(item.get("rank") == expected_rank, "hardware ranks are not ordered")
        _require(
            isinstance(item.get("device"), int),
            "hardware device index is invalid",
        )
        _require(
            isinstance(item.get("device_uuid"), str) and item["device_uuid"].startswith("GPU-"),
            "hardware GPU UUID is invalid",
        )
        _require(
            isinstance(item.get("pci_bus_id"), str) and bool(item["pci_bus_id"]),
            "hardware PCI bus id is missing",
        )
        _require(
            isinstance(item.get("pcie_max_link_speed"), str)
            and bool(item["pcie_max_link_speed"]),
            "hardware PCIe maximum link speed is missing",
        )
        for field in (
            "pcie_max_link_width",
            "pcie_current_link_width",
            "local_numa_gpu_count",
        ):
            _exact_int(item.get(field), f"hardware {field}", minimum=1)
        numa_node = _exact_int(item.get("gpu_numa_node"), "GPU NUMA node", minimum=0)
        affinity = item.get("host_cpu_affinity")
        _require(
            isinstance(affinity, list)
            and affinity
            and all(type(cpu) is int and cpu >= 0 for cpu in affinity),
            "rank CPU affinity is invalid",
        )
        _require(
            item.get("host_numa_policy") == GPU_LOCAL_FIRST_TOUCH_POLICY,
            "rank host NUMA policy differs from the production tier policy",
        )
        _require(
            item.get("host_numa_node") == numa_node,
            "rank host NUMA node differs from its GPU-local node",
        )
        page_counts = item.get("host_numa_page_counts")
        resident_pages = _exact_int(
            item.get("host_resident_pages"),
            "rank host resident pages",
            minimum=1,
        )
        slab_bytes = _exact_int(
            item.get("host_slab_bytes"),
            "rank host slab bytes",
            minimum=1,
        )
        os_page_size = _exact_int(
            item.get("host_os_page_size"),
            "rank host OS page size",
            minimum=1,
        )
        _require(
            isinstance(page_counts, dict)
            and page_counts == {str(numa_node): resident_pages}
            and item.get("host_full_slab_local_coverage") is True
            and resident_pages * os_page_size >= slab_bytes,
            "rank host slab attestation does not cover wholly GPU-local memory",
        )
        _require(
            isinstance(item.get("gpu_name"), str)
            and bool(item["gpu_name"])
            and isinstance(item.get("gpu_sm"), list)
            and len(item["gpu_sm"]) == 2
            and all(type(part) is int and part >= 0 for part in item["gpu_sm"]),
            "rank GPU identity is incomplete",
        )
    _require(
        len({item["device_uuid"] for item in ranks}) == tp_size,
        "hardware rank inventory repeats a GPU",
    )
    for item in ranks:
        expected_local_count = sum(
            peer["gpu_numa_node"] == item["gpu_numa_node"] for peer in ranks
        )
        _require(
            item["local_numa_gpu_count"] == expected_local_count,
            "hardware local NUMA GPU count differs from the rank inventory",
        )
    _require(
        isinstance(hardware.get("nvidia_smi_topology"), list) and hardware["nvidia_smi_topology"],
        "NVIDIA topology output is missing",
    )
    _require(
        isinstance(hardware.get("numa_topology"), list) and hardware["numa_topology"],
        "NUMA topology output is missing",
    )
    return hardware


def _validate_prompt_trace(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "prompt_trace must be an object")
    trace = dict(value)
    _require(trace.get("seed") == 187, "prompt trace seed differs")
    _require(
        trace.get("max_prefix_tokens") == max(PREFIX_LENGTHS),
        "prompt trace maximum prefix differs",
    )
    _require(
        isinstance(trace.get("construction"), str) and bool(trace["construction"]),
        "prompt trace construction is missing",
    )
    _require(
        _valid_sha256(trace.get("full_token_ids_sha256")),
        "prompt token digest is invalid",
    )
    prefixes = trace.get("prefix_token_ids_sha256")
    _require(
        isinstance(prefixes, dict)
        and set(prefixes) == {str(prefix) for prefix in PREFIX_LENGTHS}
        and all(_valid_sha256(digest) for digest in prefixes.values()),
        "prompt prefix-token digests are incomplete",
    )
    return trace


def _validate_cuda_span(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} must be an object")
    span = dict(value)
    _require(
        span.get("start_recorded") is True
        and span.get("end_recorded") is True
        and span.get("complete") is True,
        f"{name} CUDA events are incomplete",
    )
    _exact_int(span.get("stream_id"), f"{name} stream id", minimum=0)
    _exact_int(span.get("elapsed_ns"), f"{name} elapsed_ns", minimum=1)
    return span


def _validate_staging(
    value: object,
    *,
    expected_bytes: int,
    expected_numa_node: int,
    name: str,
) -> dict[str, object]:
    from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY

    _require(isinstance(value, dict), f"{name} staging evidence is missing")
    staging = dict(value)
    _require(staging.get("pinned") is True, f"{name} staging memory is not pinned")
    _require(
        staging.get("bytes") == expected_bytes,
        f"{name} staging byte count differs",
    )
    _require(
        staging.get("host_numa_policy") == GPU_LOCAL_FIRST_TOUCH_POLICY
        and staging.get("host_numa_node") == expected_numa_node,
        f"{name} staging allocation does not use the shared GPU-local policy",
    )
    affinity = staging.get("host_cpu_affinity")
    _require(
        isinstance(affinity, list)
        and affinity
        and all(type(cpu) is int and cpu >= 0 for cpu in affinity),
        f"{name} staging CPU affinity is invalid",
    )
    page_counts = staging.get("host_numa_page_counts")
    resident_pages = _exact_int(
        staging.get("host_resident_pages"),
        f"{name} resident pages",
        minimum=1,
    )
    slab_bytes = _exact_int(
        staging.get("host_slab_bytes"),
        f"{name} host slab bytes",
        minimum=expected_bytes,
    )
    os_page_size = _exact_int(
        staging.get("host_os_page_size"),
        f"{name} host OS page size",
        minimum=1,
    )
    _require(
        isinstance(page_counts, dict)
        and set(page_counts) == {str(expected_numa_node)}
        and page_counts[str(expected_numa_node)] == resident_pages,
        f"{name} shared tier evidence is not wholly GPU-local",
    )
    _require(
        staging.get("host_full_slab_local_coverage") is True
        and resident_pages * os_page_size >= slab_bytes,
        f"{name} shared tier evidence does not cover the complete pinned slab",
    )
    return staging


def _validate_rank_events(
    value: object,
    *,
    tp_size: int,
    hardware: Mapping[str, object],
    runtime: Mapping[str, object],
    prefix_tokens: int,
    phase: str,
    expected_intervals: tuple[str, ...],
    transfer: bool,
    wall_elapsed_ns: int | None = None,
) -> tuple[dict[str, object], ...]:
    _require(
        isinstance(value, list) and len(value) == tp_size,
        f"{phase} rank events differ from TP degree",
    )
    ranks = hardware["ranks"]
    _require(isinstance(ranks, list), "validated hardware lost its rank list")
    expected_bytes = expected_rank_bytes(tp_size, prefix_tokens)
    validated: list[dict[str, object]] = []
    for expected_rank, raw_event in enumerate(value):
        _require(isinstance(raw_event, dict), f"{phase} rank event is invalid")
        event = dict(raw_event)
        hardware_rank = ranks[expected_rank]
        _require(isinstance(hardware_rank, dict), "hardware rank is invalid")
        _require(event.get("rank") == expected_rank, f"{phase} ranks are not ordered")
        _require(
            event.get("device_uuid") == hardware_rank["device_uuid"]
            and event.get("pci_bus_id") == hardware_rank["pci_bus_id"]
            and event.get("gpu_numa_node") == hardware_rank["gpu_numa_node"],
            f"{phase} rank identity differs from hardware provenance",
        )
        _require(
            _valid_sha256(event.get("pool_fingerprint"))
            and event["pool_fingerprint"] == runtime["rank_pool_fingerprints"][str(expected_rank)],
            f"{phase} pool/layout fingerprint differs",
        )
        _require(_valid_sha256(event.get("kv_sha256")), f"{phase} KV digest is invalid")
        _require(
            event.get("bytes") == (expected_bytes if transfer else 0),
            f"{phase} rank byte count differs",
        )
        ready = _validate_cuda_span(event.get("cuda_ready_span"), f"{phase} ready span")
        # Only rank 0's CUDA interval is nested inside the retained rank-0
        # controller endpoints.  A follower may return from the leading NCCL
        # barrier before rank 0 is scheduled and record its CUDA start first;
        # comparing that cross-rank interval with rank 0's CPU wall would make
        # ordinary OS scheduling jitter a false evidence failure.
        if wall_elapsed_ns is not None and expected_rank == 0:
            _require(
                ready["elapsed_ns"] <= wall_elapsed_ns,
                f"{phase} rank-0 CUDA ready span exceeds controller wall time",
            )
        intervals = event.get("cuda_intervals")
        _require(
            isinstance(intervals, list)
            and [item.get("name") if isinstance(item, dict) else None for item in intervals]
            == list(expected_intervals),
            f"{phase} CUDA interval inventory differs",
        )
        for index, interval in enumerate(intervals):
            _validate_cuda_span(interval, f"{phase} interval {index}")
            if interval.get("name") in {"offload_d2h", "restore_h2d"}:
                _require(
                    interval.get("scope") == "pure-dma"
                    and interval.get("telemetry")
                    == "tier-transfer.cuda_elapsed_ns",
                    f"{phase} transfer interval is not exact tier DMA telemetry",
                )
        if transfer:
            staging = _validate_staging(
                event.get("staging"),
                expected_bytes=expected_bytes,
                expected_numa_node=int(hardware_rank["gpu_numa_node"]),
                name=phase,
            )
            for field in (
                "host_numa_policy",
                "host_numa_node",
                "host_cpu_affinity",
                "host_numa_page_counts",
                "host_resident_pages",
                "host_slab_bytes",
                "host_os_page_size",
                "host_full_slab_local_coverage",
            ):
                _require(
                    staging.get(field) == hardware_rank.get(field),
                    f"{phase} staging {field} differs from startup attestation",
                )
        else:
            _require(event.get("staging") is None, f"{phase} must not report staging")
        validated.append(event)
    return tuple(validated)


def _rank_digest_map(events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {str(event["rank"]): event["kv_sha256"] for event in events}


def _validate_common_row(
    row: Mapping[str, object],
    expected: Mapping[str, object],
    tp_size: int,
) -> tuple[int, int, bool, tuple[str, str]]:
    _require(row.get("tp_size") == tp_size, "row TP degree differs")
    prefix = _exact_int(row.get("prefix_tokens"), "prefix_tokens", minimum=1)
    pair_index = _exact_int(row.get("pair_index"), "pair_index", minimum=-1)
    warmup = row.get("warmup")
    _require(type(warmup) is bool, "warmup marker must be boolean")
    pair_order = row.get("pair_order")
    _require(
        isinstance(pair_order, list) and len(pair_order) == 2 and set(pair_order) == set(ARMS),
        "pair order is invalid",
    )
    for key in ("row_type", "prefix_tokens", "pair_index", "warmup", "pair_order"):
        _require(row.get(key) == expected.get(key), f"row schedule differs at {key}")
    if expected.get("row_type") == "sample":
        _require(row.get("arm") == expected.get("arm"), "sample arm order differs")
    return prefix, pair_index, warmup, (str(pair_order[0]), str(pair_order[1]))


def _validate_snapshot(
    row: Mapping[str, object],
    *,
    expected: Mapping[str, object],
    tp_size: int,
    hardware: Mapping[str, object],
    runtime: Mapping[str, object],
    prompt_trace: Mapping[str, object],
) -> dict[str, object]:
    prefix, _pair_index, _warmup, _pair_order = _validate_common_row(row, expected, tp_size)
    page_count = prefix // PAGE_SIZE
    _require(row.get("page_count") == page_count, "snapshot page count differs")
    prefix_digests = prompt_trace["prefix_token_ids_sha256"]
    _require(isinstance(prefix_digests, dict), "validated prompt trace lost prefixes")
    _require(
        row.get("token_ids_sha256") == prefix_digests[str(prefix)],
        "snapshot token digest differs",
    )
    source_pages = row.get("source_page_ids")
    destination_pages = row.get("destination_page_ids")
    _require(
        isinstance(source_pages, list)
        and isinstance(destination_pages, list)
        and len(source_pages) == page_count
        and len(destination_pages) == page_count
        and all(type(page) is int and page >= 0 for page in source_pages)
        and all(type(page) is int and page >= 0 for page in destination_pages)
        and len(set(source_pages)) == page_count
        and len(set(destination_pages)) == page_count,
        "snapshot source/destination page ids are invalid",
    )
    _require(
        source_pages == list(range(MAX_PREFIX_PAGES, MAX_PREFIX_PAGES + page_count))
        and destination_pages == list(range(page_count)),
        "snapshot does not use the binding source-high/destination-zero remap",
    )
    _require(
        row.get("restore_path") == RESTORE_PATH
        and row.get("control_backend") == CONTROL_BACKEND,
        "snapshot execution path differs from the production control contract",
    )
    events = _validate_rank_events(
        row.get("rank_events"),
        tp_size=tp_size,
        hardware=hardware,
        runtime=runtime,
        prefix_tokens=prefix,
        phase="offload",
        expected_intervals=("offload_d2h",),
        transfer=True,
    )
    digests = _rank_digest_map(events)
    _require(
        row.get("source_kv_sha256_by_rank") == digests
        and row.get("host_kv_sha256_by_rank") == digests,
        "offloaded host KV bytes differ from the source",
    )
    invalidations = row.get("source_invalidation_rank_receipts")
    _require(
        isinstance(invalidations, list) and len(invalidations) == tp_size,
        "source invalidation rank receipt inventory differs from TP degree",
    )
    expected_bytes = expected_rank_bytes(tp_size, prefix)
    for rank, receipt in enumerate(invalidations):
        _require(isinstance(receipt, dict), "source invalidation receipt is invalid")
        _require(
            receipt
            == {
                "rank": rank,
                "method": "cuda-zero-fill-plus-device-nonzero-check",
                "fill_value": 0,
                "page_count": page_count,
                "bytes": expected_bytes,
                "page_ids_sha256": sha256_json(source_pages),
                "source_kv_sha256_before": digests[str(rank)],
                "synchronized": True,
                "all_zero": True,
            },
            "source HBM invalidation receipt differs or is incomplete",
        )
    _require(
        row.get("async_submission") is True,
        "offload did not exercise asynchronous submission",
    )
    _require(
        row.get("ownership")
        == {
            "before": {
                "source_device_owned_pages": page_count,
                "tier_reserved_pages": 0,
            },
            "pending": {
                "source_device_reusable": False,
                "tier_published": False,
            },
            "after": {
                "source_device_reusable": True,
                "source_device_invalidated": True,
                "tier_resident_pages": page_count,
                "live_transfer": False,
            },
        },
        "offload ownership transition differs",
    )
    _require(
        row.get("retry_count") == 0
        and row.get("error") is None
        and row.get("fallback_used") is False,
        "offload retried, failed, or fell back",
    )
    return dict(row)


def _validate_output(value: object) -> dict[str, object]:
    descriptor = _validate_descriptor(value, "sample output")
    output = descriptor["value"]
    _require(isinstance(output, dict), "sample output value must be an object")
    _require(
        isinstance(output.get("token_ids"), list)
        and output["token_ids"]
        and all(type(token) is int and token >= 0 for token in output["token_ids"]),
        "sample output token ids are invalid",
    )
    _require(isinstance(output.get("text"), str), "sample output text is invalid")
    _require(
        isinstance(output.get("finish_reason"), str),
        "sample finish reason is invalid",
    )
    _require(
        _valid_sha256(output.get("logprobs_sha256")),
        "sample logprob digest is invalid",
    )
    return descriptor


def _validate_sample(
    row: Mapping[str, object],
    *,
    expected: Mapping[str, object],
    tp_size: int,
    hardware: Mapping[str, object],
    runtime: Mapping[str, object],
    prompt_trace: Mapping[str, object],
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    prefix, _pair_index, _warmup, _pair_order = _validate_common_row(row, expected, tp_size)
    arm = row.get("arm")
    _require(arm in ARMS, "sample arm is invalid")
    started = _exact_int(row.get("controller_started_ns"), "controller start", minimum=0)
    ended = _exact_int(row.get("controller_ended_ns"), "controller end", minimum=started + 1)
    elapsed = _exact_int(row.get("wall_elapsed_ns"), "wall elapsed", minimum=1)
    _require(ended - started == elapsed, "sample wall elapsed differs from endpoints")
    prefix_digests = prompt_trace["prefix_token_ids_sha256"]
    _require(isinstance(prefix_digests, dict), "validated prompt trace lost prefixes")
    _require(
        row.get("token_ids_sha256") == prefix_digests[str(prefix)]
        and row.get("token_ids_sha256") == snapshot.get("token_ids_sha256"),
        "sample token digest differs",
    )
    if arm == "restore":
        intervals = ("restore_h2d", "next_token_model")
        transfer = True
    else:
        intervals = ("recompute_prefill",)
        transfer = False
    events = _validate_rank_events(
        row.get("rank_events"),
        tp_size=tp_size,
        hardware=hardware,
        runtime=runtime,
        prefix_tokens=prefix,
        phase=str(arm),
        expected_intervals=intervals,
        transfer=transfer,
        wall_elapsed_ns=elapsed,
    )
    digests = _rank_digest_map(events)
    _require(
        row.get("ready_kv_sha256_by_rank") == digests
        and row.get("source_kv_sha256_by_rank") == snapshot.get("source_kv_sha256_by_rank"),
        "sample KV digests are inconsistent",
    )
    _require(
        row.get("ready_kv_sha256_by_rank") == row.get("source_kv_sha256_by_rank"),
        "sample ready KV differs byte-for-byte from source KV",
    )
    _validate_output(row.get("output"))
    page_count = prefix // PAGE_SIZE
    host_resident_before = (
        page_count
        if arm == "restore" or (arm == "recompute" and row["pair_order"][1] == "restore")
        else 0
    )
    host_resident_after = (
        page_count if arm == "recompute" and row["pair_order"][1] == "restore" else 0
    )
    expected_ownership = {
        "before": {
            "destination_published": False,
            "host_resident_pages": host_resident_before,
        },
        "pending": {"destination_published": False},
        "after": {
            "destination_published_pages": page_count,
            "host_resident_pages": host_resident_after,
            "live_transfer": False,
        },
    }
    _require(row.get("ownership") == expected_ownership, "sample ownership differs")
    _require(
        row.get("retry_count") == 0
        and row.get("error") is None
        and row.get("fallback_used") is False,
        "sample retried, failed, or fell back",
    )
    _require(
        row.get("production_model_forward") is True,
        "sample did not cross a production model forward",
    )
    _require(
        row.get("recompute_cache_hits") == 0,
        "sample used cached recompute tokens",
    )
    _require(
        row.get("restored_from_dram") is (arm == "restore"),
        "sample arm did not execute the requested data path",
    )
    _require(
        row.get("restore_path") == RESTORE_PATH
        and row.get("control_backend") == CONTROL_BACKEND,
        "sample did not use the production Radix/DistTP Gloo path contract",
    )
    _require(
        row.get("destination_page_ids_sha256") == sha256_json(snapshot.get("destination_page_ids")),
        "sample destination pages differ from the remap probe",
    )
    return dict(row)


def _validate_container(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "container provenance must be an object")
    container = dict(value)
    image_id = container.get("image_id")
    _require(
        isinstance(image_id, str)
        and image_id.startswith("sha256:")
        and _valid_sha256(image_id.removeprefix("sha256:")),
        "container image id is invalid",
    )
    _require(
        _valid_sha256(container.get("container_id")),
        "container id is invalid",
    )
    _require(
        isinstance(container.get("command"), list)
        and container["command"]
        and all(isinstance(part, str) for part in container["command"]),
        "container command is missing",
    )
    _require(
        isinstance(container.get("repo_digests"), list)
        and all(isinstance(item, str) and "@sha256:" in item for item in container["repo_digests"]),
        "container repository digest inventory is invalid",
    )
    return container


def _validate_run_header(row: Mapping[str, object], tp_size: int) -> dict[str, object]:
    _require(row.get("row_type") == "run", "raw shard must start with a run row")
    _require(row.get("schema_version") == SCHEMA_VERSION, "run schema differs")
    _require(row.get("measurement_kind") == MEASUREMENT_KIND, "run kind differs")
    _require(row.get("tp_size") == tp_size, "run TP degree differs")
    _require(
        isinstance(row.get("run_id"), str) and bool(row["run_id"]),
        "run id is missing",
    )
    _utc_timestamp(row.get("measurement_started_at"), "measurement start timestamp")
    _require(row.get("config") == formal_config(tp_size), "formal config differs")
    source = _validate_source(row.get("source"), "measurement-start source")
    checkpoint = _validate_checkpoint(row.get("checkpoint"), "measurement-start checkpoint")
    runtime = _validate_runtime(row.get("runtime"), tp_size)
    source_inventory = {
        path: source["bound_file_sha256"][path]
        for path in source["kairyu_python_paths"]
    }
    _require(
        runtime["engine_source_sha256"]
        == sha256_json(dict(sorted(source_inventory.items()))),
        "runtime installed Kairyu source differs from the full committed Python inventory",
    )
    hardware = _validate_hardware(row.get("hardware"), tp_size)
    prompt_trace = _validate_prompt_trace(row.get("prompt_trace"))
    container = _validate_container(row.get("container"))
    return {
        **dict(row),
        "source": source,
        "checkpoint": checkpoint,
        "runtime": runtime,
        "hardware": hardware,
        "prompt_trace": prompt_trace,
        "container": container,
    }


def _validate_run_end(
    row: Mapping[str, object],
    *,
    tp_size: int,
    header: Mapping[str, object],
) -> dict[str, object]:
    _require(row.get("row_type") == "run_end", "raw shard must end with run_end")
    _require(row.get("tp_size") == tp_size, "run_end TP degree differs")
    _require(row.get("run_id") == header["run_id"], "run_end id differs")
    _require(row.get("status") == "complete", "run did not complete")
    _require(row.get("errors") == [], "run recorded errors")
    ended_at = _utc_timestamp(row.get("measurement_ended_at"), "measurement end timestamp")
    started_at = _utc_timestamp(
        header.get("measurement_started_at"),
        "measurement start timestamp",
    )
    _require(ended_at > started_at, "measurement end must follow its start")
    source = _validate_source(row.get("source"), "measurement-end source")
    checkpoint = _validate_checkpoint(row.get("checkpoint"), "measurement-end checkpoint")
    _require(source == header["source"], "source changed during measurement")
    _require(
        checkpoint == header["checkpoint"],
        "checkpoint changed during measurement",
    )
    expected_counts = {
        "snapshot_rows": len(PREFIX_LENGTHS) * (WARMUP_PAIRS + MEASURED_REPEATS),
        "sample_rows": 2 * len(PREFIX_LENGTHS) * (WARMUP_PAIRS + MEASURED_REPEATS),
        "warmup_sample_rows": 2 * len(PREFIX_LENGTHS) * WARMUP_PAIRS,
        "measured_sample_rows": 2 * len(PREFIX_LENGTHS) * MEASURED_REPEATS,
    }
    _require(row.get("row_counts") == expected_counts, "run_end row counts differ")
    return dict(row)


def _validate_shard(
    rows: Sequence[Mapping[str, object]],
    tp_size: int,
) -> tuple[
    dict[str, object],
    dict[tuple[int, int], dict[str, dict[str, object]]],
]:
    _require(tp_size in TP_DEGREES, f"unsupported TP degree: {tp_size}")
    schedule = expected_schedule()
    _require(
        len(rows) == len(schedule) + 2,
        f"TP{tp_size} raw row count differs from the fixed schedule",
    )
    _require(
        all(row.get("row_index") == index for index, row in enumerate(rows)),
        f"TP{tp_size} raw row indices are not contiguous",
    )
    header = _validate_run_header(rows[0], tp_size)
    runtime = header["runtime"]
    hardware = header["hardware"]
    prompt_trace = header["prompt_trace"]
    _require(isinstance(runtime, dict), "validated runtime is invalid")
    _require(isinstance(hardware, dict), "validated hardware is invalid")
    _require(isinstance(prompt_trace, dict), "validated prompt trace is invalid")
    pairs: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    current_snapshot: dict[str, object] | None = None
    for row, expected in zip(rows[1:-1], schedule, strict=True):
        _require(isinstance(row, dict), "raw row must be an object")
        if expected["row_type"] == "snapshot":
            current_snapshot = _validate_snapshot(
                row,
                expected=expected,
                tp_size=tp_size,
                hardware=hardware,
                runtime=runtime,
                prompt_trace=prompt_trace,
            )
            key = (int(expected["prefix_tokens"]), int(expected["pair_index"]))
            _require(key not in pairs, "duplicate snapshot pair")
            pairs[key] = {"snapshot": current_snapshot}
        else:
            _require(current_snapshot is not None, "sample precedes its snapshot")
            sample = _validate_sample(
                row,
                expected=expected,
                tp_size=tp_size,
                hardware=hardware,
                runtime=runtime,
                prompt_trace=prompt_trace,
                snapshot=current_snapshot,
            )
            key = (int(expected["prefix_tokens"]), int(expected["pair_index"]))
            arm = str(expected["arm"])
            _require(arm not in pairs[key], "duplicate sample arm")
            pairs[key][arm] = sample
            if set(pairs[key]) == {"snapshot", *ARMS}:
                restore = pairs[key]["restore"]
                recompute = pairs[key]["recompute"]
                _require(
                    restore["output"] == recompute["output"],
                    "restore/recompute output parity failed",
                )
                _require(
                    restore["ready_kv_sha256_by_rank"] == recompute["ready_kv_sha256_by_rank"],
                    "restore/recompute KV parity failed",
                )
    expected_keys = {
        (prefix, repeat) for prefix in PREFIX_LENGTHS for repeat in (-1, *range(MEASURED_REPEATS))
    }
    _require(set(pairs) == expected_keys, "pair matrix is incomplete")
    _require(
        all(set(pair) == {"snapshot", *ARMS} for pair in pairs.values()),
        "pair matrix omits an arm",
    )
    _validate_run_end(rows[-1], tp_size=tp_size, header=header)
    return header, pairs


def derive_cell_metrics(
    pairs: Mapping[tuple[int, int], Mapping[str, Mapping[str, object]]],
    prefix_tokens: int,
) -> dict[str, object]:
    pair_rows: list[dict[str, object]] = []
    ratios: list[float] = []
    restore_wins = 0
    restore_walls: list[int] = []
    recompute_walls: list[int] = []
    restore_cuda_worst: list[int] = []
    recompute_cuda_worst: list[int] = []
    for repeat in range(MEASURED_REPEATS):
        pair = pairs[(prefix_tokens, repeat)]
        restore = pair["restore"]
        recompute = pair["recompute"]
        restore_wall = int(restore["wall_elapsed_ns"])
        recompute_wall = int(recompute["wall_elapsed_ns"])
        ratio = restore_wall / recompute_wall
        restore_rank_events = restore["rank_events"]
        recompute_rank_events = recompute["rank_events"]
        _require(
            isinstance(restore_rank_events, list) and isinstance(recompute_rank_events, list),
            "validated samples lost rank events",
        )
        restore_worst = max(
            int(event["cuda_ready_span"]["elapsed_ns"]) for event in restore_rank_events
        )
        recompute_worst = max(
            int(event["cuda_ready_span"]["elapsed_ns"]) for event in recompute_rank_events
        )
        pair_rows.append(
            {
                "repeat": repeat,
                "pair_order": restore["pair_order"],
                "restore_wall_ns": restore_wall,
                "recompute_wall_ns": recompute_wall,
                "kv_sha256": sha256_json(restore["ready_kv_sha256_by_rank"]),
                "restore_over_recompute": ratio,
                "restore_worst_rank_cuda_ns": restore_worst,
                "recompute_worst_rank_cuda_ns": recompute_worst,
            }
        )
        ratios.append(ratio)
        restore_walls.append(restore_wall)
        recompute_walls.append(recompute_wall)
        restore_cuda_worst.append(restore_worst)
        recompute_cuda_worst.append(recompute_worst)
        restore_wins += restore_wall < recompute_wall
    median_ratio = statistics.median(ratios)
    passed = median_ratio < 1.0 and restore_wins >= MIN_RESTORE_WINS
    return {
        "prefix_tokens": prefix_tokens,
        "pairs": pair_rows,
        "restore_wins": restore_wins,
        "pair_count": MEASURED_REPEATS,
        "one_sided_sign_test_p": _sign_test_p_value(restore_wins),
        "median_paired_restore_over_recompute": median_ratio,
        "paired_ratio_mad": _median_absolute_deviation(ratios),
        "restore_median_wall_ns": statistics.median(restore_walls),
        "recompute_median_wall_ns": statistics.median(recompute_walls),
        "restore_median_worst_rank_cuda_ns": statistics.median(restore_cuda_worst),
        "recompute_median_worst_rank_cuda_ns": statistics.median(recompute_cuda_worst),
        "passed": passed,
    }


def derive_crossover(cell_metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    _require(
        [cell.get("prefix_tokens") for cell in cell_metrics] == list(PREFIX_LENGTHS),
        "crossover needs the complete ordered prefix grid",
    )
    first_index = next(
        (
            index
            for index in range(len(cell_metrics))
            if all(cell.get("passed") is True for cell in cell_metrics[index:])
        ),
        None,
    )
    if first_index is None:
        return {
            "status": "none_in_measured_range",
            "min_restore_tokens": None,
            "bracket": None,
            "interpolated": False,
        }
    prefix = int(cell_metrics[first_index]["prefix_tokens"])
    if first_index == 0:
        status = "at_or_below_measured_range"
        bracket = {"upper_inclusive_tokens": prefix}
    else:
        status = "bracketed"
        bracket = {
            "lower_exclusive_tokens": int(cell_metrics[first_index - 1]["prefix_tokens"]),
            "upper_inclusive_tokens": prefix,
        }
    return {
        "status": status,
        "min_restore_tokens": prefix,
        "bracket": bracket,
        "interpolated": False,
    }


def _runtime_identity(header: Mapping[str, object], tp_size: int) -> dict[str, object]:
    checkpoint = header["checkpoint"]
    runtime = header["runtime"]
    hardware = header["hardware"]
    _require(isinstance(checkpoint, dict), "validated checkpoint is invalid")
    _require(isinstance(runtime, dict), "validated runtime is invalid")
    _require(isinstance(hardware, dict), "validated hardware is invalid")
    ranks = hardware["ranks"]
    _require(isinstance(ranks, list) and ranks, "validated hardware ranks are invalid")
    gpu_names = {rank["gpu_name"] for rank in ranks if isinstance(rank, dict)}
    gpu_sms = {
        tuple(rank["gpu_sm"])
        for rank in ranks
        if isinstance(rank, dict) and isinstance(rank.get("gpu_sm"), list)
    }
    _require(len(gpu_names) == 1 and len(gpu_sms) == 1, "TP ranks use mixed GPU types")
    return {
        "model_config_sha256": checkpoint["model_config_sha256"],
        "engine_source_sha256": runtime["engine_source_sha256"],
        "tensor_parallel_size": tp_size,
        "attention_backend_identity": runtime["attention_backend_identity"],
        "transfer_backend_identity": runtime["tier_backend"],
        "max_num_batched_tokens": runtime["max_num_batched_tokens"],
        "host_memory_placement_policy": runtime["host_memory_placement_policy"],
        "os_page_size": runtime["os_page_size"],
        "pcie_max_link_speed": runtime["pcie_max_link_speed"],
        "pcie_max_link_width": runtime["pcie_max_link_width"],
        "pcie_current_link_width": runtime["pcie_current_link_width"],
        "local_numa_gpu_count": runtime["local_numa_gpu_count"],
        "cpu_model": runtime["cpu_model"],
        "python_version": runtime["python_version"],
        "openssl_version": runtime["openssl_version"],
        "page_size": PAGE_SIZE,
        "num_layers": NUM_LAYERS,
        "num_kv_heads_per_rank": NUM_KV_HEADS // tp_size,
        "head_dim": HEAD_DIM,
        "v_head_dim": HEAD_DIM,
        "kv_dtype": DTYPE,
        "device_type": "cuda",
        "accelerator_name": next(iter(gpu_names)),
        "compute_capability": list(next(iter(gpu_sms))),
        "torch_version": runtime["torch_version"],
        "cuda_version": runtime["torch_cuda_version"],
    }


def recompute_manifest(
    shards: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    raw_sha256: Mapping[int, str],
) -> dict[str, object]:
    _require(set(shards) == set(TP_DEGREES), "TP4/TP8 raw shards are both required")
    _require(
        set(raw_sha256) == set(TP_DEGREES)
        and all(_valid_sha256(digest) for digest in raw_sha256.values()),
        "raw SHA-256 inventory is invalid",
    )
    headers: dict[int, dict[str, object]] = {}
    pairs_by_tp: dict[
        int,
        dict[tuple[int, int], dict[str, dict[str, object]]],
    ] = {}
    cells_by_tp: dict[int, list[dict[str, object]]] = {}
    crossovers: dict[int, dict[str, object]] = {}
    for tp_size in TP_DEGREES:
        header, pairs = _validate_shard(shards[tp_size], tp_size)
        headers[tp_size] = header
        pairs_by_tp[tp_size] = pairs
        cells = [derive_cell_metrics(pairs, prefix) for prefix in PREFIX_LENGTHS]
        cells_by_tp[tp_size] = cells
        crossovers[tp_size] = derive_crossover(cells)
    _require(
        headers[4]["source"] == headers[8]["source"],
        "TP4/TP8 source provenance differs",
    )
    _require(
        headers[4]["checkpoint"] == headers[8]["checkpoint"],
        "TP4/TP8 checkpoint provenance differs",
    )
    _require(
        headers[4]["prompt_trace"] == headers[8]["prompt_trace"],
        "TP4/TP8 prompt trace differs",
    )
    runtime4 = headers[4]["runtime"]
    runtime8 = headers[8]["runtime"]
    _require(isinstance(runtime4, dict) and isinstance(runtime8, dict), "runtime invalid")
    for key in (
        "attention_backend",
        "attention_backend_identity",
        "engine_source_sha256",
        "max_model_len",
        "max_num_batched_tokens",
        "host_memory_placement_policy",
        "os_page_size",
        "pcie_max_link_speed",
        "pcie_max_link_width",
        "pcie_current_link_width",
        "local_numa_gpu_count",
        "cpu_model",
        "python_version",
        "openssl_version",
        "chunked_prefill",
        "tier_backend",
        "recompute_path",
        "recompute_cache_disabled",
        "torch_version",
        "torch_cuda_version",
        "nccl_version",
        "driver_version",
    ):
        _require(runtime4.get(key) == runtime8.get(key), f"TP runtime differs at {key}")
    container4 = headers[4]["container"]
    container8 = headers[8]["container"]
    _require(
        isinstance(container4, dict) and isinstance(container8, dict),
        "container provenance is invalid",
    )
    _require(
        container4.get("image_id") == container8.get("image_id")
        and container4.get("repo_digests") == container8.get("repo_digests"),
        "TP4/TP8 must use the same immutable container image",
    )
    hardware4 = headers[4]["hardware"]
    hardware8 = headers[8]["hardware"]
    _require(
        isinstance(hardware4, dict) and isinstance(hardware8, dict),
        "hardware provenance is invalid",
    )
    ranks4 = hardware4.get("ranks")
    ranks8 = hardware8.get("ranks")
    _require(
        isinstance(ranks4, list) and isinstance(ranks8, list),
        "hardware rank inventory is invalid",
    )
    _require(
        [rank.get("device_uuid") for rank in ranks4]
        == [rank.get("device_uuid") for rank in ranks8[:4]],
        "TP4 GPUs must be the first four GPUs in the TP8 visible mapping",
    )
    starts = {
        tp_size: _utc_timestamp(
            headers[tp_size]["measurement_started_at"],
            f"TP{tp_size} measurement start",
        )
        for tp_size in TP_DEGREES
    }
    ends = {
        tp_size: _utc_timestamp(
            shards[tp_size][-1].get("measurement_ended_at"),
            f"TP{tp_size} measurement end",
        )
        for tp_size in TP_DEGREES
    }
    _require(
        ends[4] <= starts[8] or ends[8] <= starts[4],
        "TP4/TP8 measurement intervals overlap",
    )
    checks = {
        "tp4_complete_correct_provenance_bound_evidence": True,
        "tp8_complete_correct_provenance_bound_evidence": True,
        "tp4_has_stable_measured_crossover": (crossovers[4]["min_restore_tokens"] is not None),
        "tp8_has_stable_measured_crossover": (crossovers[8]["min_restore_tokens"] is not None),
    }
    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": MEASUREMENT_KIND,
        "gate": "G5-F4a",
        "issue": 187,
        "passed": passed,
        "checks": checks,
        "fixed_method": {
            "prefix_lengths": list(PREFIX_LENGTHS),
            "warmup_pairs_per_cell": WARMUP_PAIRS,
            "measured_pairs_per_cell": MEASURED_REPEATS,
            "minimum_restore_wins": MIN_RESTORE_WINS,
            "median_paired_ratio_must_be_below": 1.0,
            "one_sided_sign_test_p_at_minimum": SIGN_TEST_P_AT_MIN_WINS,
            "crossover": "first passing member of stable measured suffix",
            "interpolation": False,
            "tp_runs_are_sequential": True,
        },
        "raw": {
            f"tp{tp_size}": {
                "name": RAW_NAMES[tp_size],
                "sha256": raw_sha256[tp_size],
                "rows": len(shards[tp_size]),
            }
            for tp_size in TP_DEGREES
        },
        "runtime_identity": {
            f"tp{tp_size}": _runtime_identity(headers[tp_size], tp_size) for tp_size in TP_DEGREES
        },
        "cell_metrics": {f"tp{tp_size}": cells_by_tp[tp_size] for tp_size in TP_DEGREES},
        "crossover": {f"tp{tp_size}": crossovers[tp_size] for tp_size in TP_DEGREES},
        "source": headers[4]["source"],
        "checkpoint": headers[4]["checkpoint"],
        "prompt_trace": headers[4]["prompt_trace"],
        "hardware": {f"tp{tp_size}": headers[tp_size]["hardware"] for tp_size in TP_DEGREES},
        "runtime": {f"tp{tp_size}": headers[tp_size]["runtime"] for tp_size in TP_DEGREES},
        "container": {f"tp{tp_size}": headers[tp_size]["container"] for tp_size in TP_DEGREES},
    }


def _profile(
    manifest: Mapping[str, object],
    *,
    tp_size: int,
    manifest_sha256: str,
) -> dict[str, object]:
    _require(tp_size in TP_DEGREES, f"unsupported TP degree: {tp_size}")
    _require(_valid_sha256(manifest_sha256), "manifest SHA-256 is invalid")
    raw = manifest["raw"]
    runtime_identities = manifest["runtime_identity"]
    cell_metrics = manifest["cell_metrics"]
    crossovers = manifest["crossover"]
    _require(
        isinstance(raw, dict)
        and isinstance(runtime_identities, dict)
        and isinstance(cell_metrics, dict)
        and isinstance(crossovers, dict),
        "manifest profile inputs are invalid",
    )
    key = f"tp{tp_size}"
    identity = runtime_identities[key]
    _require(isinstance(identity, dict), "runtime identity is invalid")
    raw_descriptor = raw[key]
    _require(isinstance(raw_descriptor, dict), "raw descriptor is invalid")
    cells = cell_metrics[key]
    _require(isinstance(cells, list), "cell metrics are invalid")
    crossover = crossovers[key]
    _require(isinstance(crossover, dict), "crossover is invalid")
    samples = [
        {
            "prefix_tokens": cell["prefix_tokens"],
            "repeat": pair["repeat"],
            "order": pair["pair_order"],
            "restore_ns": pair["restore_wall_ns"],
            "recompute_ns": pair["recompute_wall_ns"],
            "restore_sha256": pair["kv_sha256"],
            "recompute_sha256": pair["kv_sha256"],
        }
        for cell in cells
        for pair in cell["pairs"]
    ]
    evidence_sha256 = sha256_json(
        {
            "manifest_sha256": manifest_sha256,
            "raw_sha256": raw_descriptor["sha256"],
        }
    )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "runtime_identity": identity,
        "runtime_fingerprint": sha256_json(identity),
        "samples": samples,
        "policy": {
            "rule": "median_lt_1_and_wins_at_least_8_of_9",
            "min_restore_tokens": crossover["min_restore_tokens"],
        },
        "evidence_sha256": evidence_sha256,
        "passed": manifest["passed"],
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _utc_timestamp(value: object, name: str) -> datetime:
    _require(isinstance(value, str) and bool(value), f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"{name} is not an ISO-8601 timestamp") from error
    _require(
        parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed),
        f"{name} must include an explicit UTC offset",
    )
    return parsed


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise EvidenceError(f"cannot execute {command[0]!r}: {error}") from error
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise EvidenceError(
            f"command failed ({completed.returncode}): {' '.join(command)}: {detail}"
        )
    return completed.stdout.strip()


def _sha256_path(path: Path) -> str:
    return sha256_file(path)


def _source_provenance_live() -> dict[str, object]:
    """Bind the clean commit and every Kairyu Python source used by the run."""

    commit = _run_text(("git", "rev-parse", "HEAD"), cwd=_REPO_ROOT)
    tracked = _run_text(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=_REPO_ROOT,
    )
    _require(_valid_commit(commit), "live source commit is invalid")
    _require(not tracked, "formal measurement requires a clean tracked worktree")
    python_paths = tuple(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in sorted((_REPO_ROOT / "kairyu").rglob("*.py"))
    )
    bound_paths = tuple(
        dict.fromkeys(
            (
                SOURCE_PATH,
                "Dockerfile.cuda",
                "pyproject.toml",
                "uv.lock",
                *python_paths,
            )
        )
    )
    _require(
        REQUIRED_SOURCE_PATHS <= set(bound_paths),
        "live source tree omits required F4a implementation paths",
    )
    bound: dict[str, str] = {}
    for relative in bound_paths:
        working = _REPO_ROOT / relative
        _require(working.is_file(), f"bound source file is missing: {relative}")
        payload = working.read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=False,
        )
        _require(
            committed.returncode == 0 and committed.stdout == payload,
            f"bound source does not match commit: {relative}",
        )
        bound[relative] = sha256_bytes(payload)
    return {
        "commit": commit,
        "dirty": False,
        "source_kind": "git",
        "script_path": SOURCE_PATH,
        "script_sha256": bound[SOURCE_PATH],
        "bound_file_sha256": bound,
        "bound_files_sha256": sha256_json(dict(sorted(bound.items()))),
        "kairyu_python_paths": list(python_paths),
        "python": sys.version,
        "platform": platform.platform(),
    }


def _checkpoint_provenance_live(model_path: Path) -> dict[str, object]:
    """Full-hash the pinned local Qwen3-32B checkpoint."""

    _require(model_path.is_dir(), f"model path is not a directory: {model_path}")
    config_path = model_path / "config.json"
    tokenizer_path = model_path / "tokenizer.json"
    _require(config_path.is_file(), "checkpoint config.json is missing")
    _require(tokenizer_path.is_file(), "checkpoint tokenizer.json is missing")
    config_file_sha256 = sha256_file(config_path)
    _require(
        config_file_sha256 == EXPECTED_MODEL_CONFIG_SHA256,
        "checkpoint config.json differs from the pinned Qwen3-32B revision",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "checkpoint config.json must be an object")
    heads = _exact_int(config.get("num_attention_heads"), "attention heads", minimum=1)
    hidden = _exact_int(config.get("hidden_size"), "hidden size", minimum=1)
    head_dim = _exact_int(config.get("head_dim"), "head dimension", minimum=1)

    shards = sorted(model_path.glob("*.safetensors"))
    _require(len(shards) == 17, "Qwen3-32B checkpoint must contain 17 weight shards")
    with ThreadPoolExecutor(max_workers=min(8, len(shards))) as executor:
        digests = list(executor.map(_sha256_path, shards))
    weights = [
        {
            "file": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": digest,
        }
        for shard, digest in zip(shards, digests, strict=True)
    ]
    parity_rollup = hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()
    _require(
        parity_rollup == EXPECTED_MODEL_WEIGHTS_ROLLUP,
        "checkpoint weights differ from the pinned Qwen3-32B revision",
    )
    architecture = {
        "model_type": config.get("model_type"),
        "hidden_size": hidden,
        "intermediate_size": config.get("intermediate_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": heads,
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": head_dim,
        "vocab_size": config.get("vocab_size"),
    }
    metadata: dict[str, str] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        candidate = model_path / name
        if candidate.is_file():
            metadata[name] = sha256_file(candidate)
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "architecture": architecture,
        "weights": weights,
        "weights_sha256": sha256_json(weights),
        "pinned_weights_rollup": parity_rollup,
        "metadata_sha256": metadata,
        "model_config_sha256": sha256_json(config),
        "read_from": str(model_path.resolve()),
    }


_PROMPT_PARAGRAPH = (
    "A production inference system must preserve model semantics while it "
    "coordinates tensor parallel collectives, paged key value memory, request "
    "scheduling, cancellation, and observable output. Measurements are useful "
    "only when their boundaries, provenance, and failure states are explicit. "
)


def _fixed_prompt_tokens(model_path: Path) -> tuple[tuple[int, ...], dict[str, object]]:
    from kairyu.engine.tokenizer import HFTokenizer

    tokenizer = HFTokenizer(model_path)
    text = _PROMPT_PARAGRAPH
    token_ids = tokenizer.encode(text)
    _require(token_ids, "fixed Qwen prompt tokenized to an empty sequence")
    while len(token_ids) < max(PREFIX_LENGTHS):
        text += _PROMPT_PARAGRAPH
        token_ids = tokenizer.encode(text)
    fixed = tuple(token_ids[: max(PREFIX_LENGTHS)])
    trace = {
        "seed": 187,
        "max_prefix_tokens": max(PREFIX_LENGTHS),
        "construction": "repeated-fixed-natural-language-truncated-v1",
        "full_token_ids_sha256": sha256_json(fixed),
        "prefix_token_ids_sha256": {
            str(prefix): sha256_json(fixed[:prefix]) for prefix in PREFIX_LENGTHS
        },
    }
    return fixed, trace


def _resolve_container_id(
    *,
    container_id: str | None,
    container_id_file: Path | None,
) -> str:
    from_file: str | None = None
    if container_id_file is not None:
        _require(
            container_id_file.is_file(),
            f"container id file is missing: {container_id_file}",
        )
        from_file = container_id_file.read_text(encoding="utf-8").strip()
    if container_id is not None and from_file is not None:
        _require(container_id == from_file, "container id argument and file differ")
    resolved = container_id or from_file
    _require(_valid_sha256(resolved), "full 64-hex Docker container id is required")
    return str(resolved)


def _container_provenance_live(
    *,
    container_id: str,
    image_id: str,
    repo_digests: Sequence[str],
    command: Sequence[str],
) -> dict[str, object]:
    _require(
        image_id.startswith("sha256:") and _valid_sha256(image_id.removeprefix("sha256:")),
        "--image-id must be sha256:<64 lowercase hex>",
    )
    _require(
        all(
            "@sha256:" in digest and _valid_sha256(digest.rsplit("@sha256:", 1)[-1])
            for digest in repo_digests
        ),
        "--repo-digest values must end in @sha256:<64 lowercase hex>",
    )
    hostname = socket.gethostname().strip()
    _require(
        bool(hostname) and container_id.startswith(hostname),
        "current Docker hostname is not a prefix of the supplied container id",
    )
    return {
        "container_id": container_id,
        "image_id": image_id,
        "repo_digests": list(repo_digests),
        "command": list(command),
        "hostname": hostname,
    }


def _visible_gpu_rows() -> tuple[dict[str, str], ...]:
    output = _run_text(
        (
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id,name",
            "--format=csv,noheader,nounits",
        )
    )
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        _require(len(parts) == 3, "nvidia-smi GPU identity row is malformed")
        rows.append({"device_uuid": parts[0], "pci_bus_id": parts[1].lower(), "name": parts[2]})
    _require(rows, "nvidia-smi returned no GPU identities")
    return tuple(rows)


def _normalized_pci_bus_id(pci_bus_id: str) -> str:
    components = pci_bus_id.lower().split(":", 1)
    _require(len(components) == 2, f"invalid PCI bus id: {pci_bus_id}")
    return f"{components[0][-4:]}:{components[1]}"


def _normalized_gpu_uuid(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    return text.replace("-", "")


def _hardware_from_tier(rank: int, tier: object) -> dict[str, object]:
    """Bind hardware provenance to the tier's shared NUMA attestation.

    PCI is the join key between Torch and ``nvidia-smi``.  UUID spellings differ
    across their Python bindings (Torch may omit ``GPU-``), so the normalized
    UUID comparison is an independent reconciliation after that PCI join.
    """

    import torch

    properties = torch.cuda.get_device_properties(rank)
    torch_pci = (
        f"{int(properties.pci_domain_id):04x}:"
        f"{int(properties.pci_bus_id):02x}:"
        f"{int(properties.pci_device_id):02x}.0"
    )
    identities = _visible_gpu_rows()
    matching = [
        row
        for row in identities
        if _normalized_pci_bus_id(row["pci_bus_id"])
        == _normalized_pci_bus_id(torch_pci)
    ]
    _require(len(matching) == 1, f"cannot map CUDA rank {rank} PCI identity to nvidia-smi")
    matched = matching[0]
    _require(
        _normalized_gpu_uuid(properties.uuid)
        == _normalized_gpu_uuid(matched["device_uuid"]),
        f"CUDA rank {rank} Torch/nvidia-smi UUIDs disagree at PCI {torch_pci}",
    )
    node = getattr(tier, "host_numa_node", None)
    policy = getattr(tier, "host_numa_policy", None)
    affinity = tuple(getattr(tier, "host_cpu_affinity", ()))
    raw_page_counts = dict(getattr(tier, "host_numa_page_counts", {}))
    from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY

    _require(policy == GPU_LOCAL_FIRST_TOUCH_POLICY, "tier did not use shared NUMA policy")
    _require(type(node) is int and node >= 0, "tier has no GPU-local NUMA attestation")
    _require(
        affinity and all(type(cpu) is int and cpu >= 0 for cpu in affinity),
        "tier CPU affinity is invalid",
    )
    _require(
        raw_page_counts
        and set(raw_page_counts) == {node}
        and all(type(count) is int and count > 0 for count in raw_page_counts.values()),
        "tier host slab is not wholly resident on its GPU-local NUMA node",
    )
    os_page_size = os.sysconf("SC_PAGE_SIZE")
    slab_bytes = int(getattr(tier, "host_nbytes", 0))
    resident_pages = sum(raw_page_counts.values())
    full_coverage = resident_pages * os_page_size >= slab_bytes > 0
    _require(full_coverage, "tier NUMA receipt does not cover the complete host slab")
    from kairyu.engine.core.kv_tier_policy import pcie_transport_identity

    (
        pcie_max_link_speed,
        pcie_max_link_width,
        pcie_current_link_width,
        link_numa_node,
    ) = pcie_transport_identity(_normalized_pci_bus_id(torch_pci))
    _require(link_numa_node == node, "PCIe and NUMA sysfs identities disagree")
    return {
        "rank": rank,
        "device": rank,
        "device_uuid": matched["device_uuid"],
        "pci_bus_id": _normalized_pci_bus_id(torch_pci),
        "gpu_numa_node": node,
        "pcie_max_link_speed": pcie_max_link_speed,
        "pcie_max_link_width": pcie_max_link_width,
        "pcie_current_link_width": pcie_current_link_width,
        # Reconciled from the gathered rank inventory by ``_startup_runtime``.
        "local_numa_gpu_count": 0,
        "host_numa_policy": policy,
        "host_numa_node": node,
        "host_cpu_affinity": list(affinity),
        "host_numa_page_counts": {str(key): value for key, value in raw_page_counts.items()},
        "host_resident_pages": resident_pages,
        "host_slab_bytes": slab_bytes,
        "host_os_page_size": os_page_size,
        "host_full_slab_local_coverage": full_coverage,
        "gpu_name": properties.name,
        "gpu_sm": [properties.major, properties.minor],
    }


def _staging_residency(tier: object, bytes_transferred: int) -> dict[str, object]:
    node = getattr(tier, "host_numa_node", None)
    counts = dict(getattr(tier, "host_numa_page_counts", {}))
    os_page_size = os.sysconf("SC_PAGE_SIZE")
    slab_bytes = int(getattr(tier, "host_nbytes", 0))
    resident_pages = sum(counts.values())
    full_coverage = resident_pages * os_page_size >= slab_bytes > 0
    _require(
        type(node) is int and set(counts) == {node} and full_coverage,
        "tier's shared NUMA attestation does not prove complete local slab coverage",
    )
    return {
        "pinned": bool(tier.host_is_pinned),
        "bytes": int(bytes_transferred),
        "host_numa_policy": tier.host_numa_policy,
        "host_numa_node": node,
        "host_cpu_affinity": list(tier.host_cpu_affinity),
        "host_numa_page_counts": {str(key): value for key, value in counts.items()},
        "host_resident_pages": resident_pages,
        "host_slab_bytes": slab_bytes,
        "host_os_page_size": os_page_size,
        "host_full_slab_local_coverage": full_coverage,
    }


def _cuda_span(start: object, end: object, stream: object) -> dict[str, object]:
    elapsed_ms = float(start.elapsed_time(end))
    _require(math.isfinite(elapsed_ms) and elapsed_ms > 0, "CUDA span is invalid")
    return {
        "start_recorded": True,
        "end_recorded": True,
        "complete": True,
        "stream_id": int(stream.cuda_stream),
        "elapsed_ns": max(1, round(elapsed_ms * 1_000_000)),
    }


def _timed_tier_transfer(
    tier: object,
    *,
    direction: str,
    keys: tuple[str, ...],
    page_ids: tuple[int, ...],
) -> tuple[object, dict[str, object]]:
    if direction == "offload":
        transfer = tier.begin_offload(keys, page_ids)
    elif direction == "restore":
        transfer = tier.begin_restore(keys, page_ids)
    else:  # pragma: no cover - internal caller contract
        raise ValueError(f"unknown tier direction: {direction}")
    transfer.wait()
    _require(transfer.finalize() is True, f"{direction} finalization failed")
    return transfer, _dma_span_from_transfer(
        transfer,
        direction=direction,
        page_count=len(page_ids),
        bytes_transferred=int(transfer.bytes_transferred),
    )


def _dma_span_from_transfer(
    transfer: object,
    *,
    direction: str,
    page_count: int,
    bytes_transferred: int,
) -> dict[str, object]:
    """Return the tier backend's pure DMA interval, never a broad ready span."""

    actual_direction = getattr(transfer, "direction", None)
    actual_direction = getattr(actual_direction, "value", actual_direction)
    actual_pages = getattr(transfer, "page_count", None)
    if actual_pages is None:
        actual_pages = len(getattr(transfer, "page_ids", ()))
    _require(actual_direction == direction, f"{direction} DMA telemetry direction differs")
    _require(actual_pages == page_count, f"{direction} DMA telemetry page count differs")
    _require(
        int(getattr(transfer, "bytes_transferred", -1)) == bytes_transferred,
        f"{direction} DMA telemetry byte count differs",
    )
    stream_id = getattr(transfer, "stream_id", None)
    elapsed_ns = getattr(transfer, "cuda_elapsed_ns", None)
    _require(type(stream_id) is int and stream_id >= 0, f"{direction} DMA stream is invalid")
    _require(type(elapsed_ns) is int and elapsed_ns > 0, f"{direction} DMA duration is invalid")
    return {
        "start_recorded": True,
        "end_recorded": True,
        "complete": True,
        "stream_id": stream_id,
        "elapsed_ns": elapsed_ns,
        "scope": "pure-dma",
        "telemetry": "tier-transfer.cuda_elapsed_ns",
    }


def _clear_page_range(pool: object, first: int, count: int) -> None:
    pool.k[:, first : first + count].zero_()
    if pool.v_head_dim:
        pool.v[:, first : first + count].zero_()


def _invalidate_source_pages(
    pool: object,
    *,
    rank: int,
    page_ids: tuple[int, ...],
    source_kv_sha256: str,
    bytes_transferred: int,
) -> dict[str, object]:
    """Destroy source HBM and prove the complete physical range is zero."""

    import torch

    _require(page_ids, "source invalidation needs at least one page")
    _require(
        page_ids == tuple(range(page_ids[0], page_ids[0] + len(page_ids))),
        "source pages must be contiguous",
    )
    _clear_page_range(pool, page_ids[0], len(page_ids))
    torch.cuda.synchronize(pool.k.device)
    nonzero = torch.count_nonzero(pool.k[:, page_ids]).to(dtype=torch.int64)
    if pool.v_head_dim:
        nonzero = nonzero + torch.count_nonzero(pool.v[:, page_ids]).to(dtype=torch.int64)
    all_zero = int(nonzero.item()) == 0
    _require(all_zero, "source HBM invalidation left non-zero KV elements")
    return {
        "rank": rank,
        "method": "cuda-zero-fill-plus-device-nonzero-check",
        "fill_value": 0,
        "page_count": len(page_ids),
        "bytes": bytes_transferred,
        "page_ids_sha256": sha256_json(page_ids),
        "source_kv_sha256_before": source_kv_sha256,
        "synchronized": True,
        "all_zero": all_zero,
    }


def _forward_token_range(
    model: object,
    pool: object,
    token_ids: tuple[int, ...],
    page_ids: tuple[int, ...],
    *,
    start: int,
    end: int,
    chunk_tokens: int,
    write_from: int,
) -> object | None:
    import torch

    hidden = None
    for chunk_start in range(start, end, chunk_tokens):
        chunk_end = min(end, chunk_start + chunk_tokens)
        tokens = torch.tensor(
            token_ids[chunk_start:chunk_end],
            dtype=torch.long,
            device=pool.k.device,
        )
        positions = torch.arange(
            chunk_start,
            chunk_end,
            dtype=torch.long,
            device=pool.k.device,
        )
        hidden = model.forward_tokens(
            tokens,
            positions,
            pool,
            list(page_ids),
            seq_len=chunk_end,
            write_from=write_from,
        )
    return hidden


def _sample_hidden(
    model: object,
    comm: object,
    *,
    rank: int,
    hidden: object,
) -> tuple[int, object | None]:
    import torch

    logits = model.logits(hidden[-1]) if rank == 0 else None
    if rank == 0:
        packet = torch.argmax(logits).reshape(1).to(dtype=torch.int64)
    else:
        packet = torch.empty((1,), dtype=torch.int64, device=hidden.device)
    packet = comm.tensor_broadcast(packet, src=0)
    return int(packet.cpu()[0]), logits


def _next_token_stage(
    model: object,
    pool: object,
    comm: object,
    *,
    rank: int,
    token_ids: tuple[int, ...],
    page_ids: tuple[int, ...],
    prefix_tokens: int,
    write_from: int,
) -> tuple[int, object | None]:
    import torch

    hidden = model.forward_tokens(
        torch.tensor(
            (token_ids[prefix_tokens - 1],),
            dtype=torch.long,
            device=pool.k.device,
        ),
        torch.tensor(
            (prefix_tokens - 1,),
            dtype=torch.long,
            device=pool.k.device,
        ),
        pool,
        list(page_ids),
        seq_len=prefix_tokens,
        write_from=write_from,
    )
    return _sample_hidden(model, comm, rank=rank, hidden=hidden)


def _recompute_stage(
    model: object,
    pool: object,
    comm: object,
    *,
    rank: int,
    token_ids: tuple[int, ...],
    page_ids: tuple[int, ...],
    prefix_tokens: int,
    chunk_tokens: int,
) -> tuple[int, object | None]:
    hidden = _forward_token_range(
        model,
        pool,
        token_ids,
        page_ids,
        start=0,
        end=prefix_tokens,
        chunk_tokens=chunk_tokens,
        write_from=0,
    )
    _require(hidden is not None, "recompute prefill produced no hidden state")
    return _sample_hidden(model, comm, rank=rank, hidden=hidden)


def _source_prefix_forward(
    model: object,
    pool: object,
    token_ids: tuple[int, ...],
    page_ids: tuple[int, ...],
    *,
    prefix_tokens: int,
    chunk_tokens: int,
) -> None:
    hidden = _forward_token_range(
        model,
        pool,
        token_ids,
        page_ids,
        start=0,
        end=prefix_tokens,
        chunk_tokens=chunk_tokens,
        write_from=0,
    )
    _require(hidden is not None, "source prefill produced no hidden state")


def _aggregate_checksums(checksums: Sequence[str]) -> str:
    _require(checksums and all(_valid_sha256(value) for value in checksums), "invalid KV checksums")
    return sha256_json(list(checksums))


def _audit_device_pages(
    tier: object,
    *,
    keys: tuple[str, ...],
    page_ids: tuple[int, ...],
) -> str:
    transfer = tier.begin_offload(keys, page_ids)
    transfer.wait()
    _require(transfer.finalize() is True, "audit D2H finalization failed")
    digest = _aggregate_checksums(transfer.checksums)
    _require(tier.discard(keys) is True, "audit tier cleanup failed")
    return digest


def _radix_tier_keys(
    token_ids: tuple[int, ...],
    *,
    prefix_tokens: int,
) -> tuple[str, ...]:
    """Derive the exact full-SHA chain used by ``RadixKVCache``."""

    from kairyu.engine.core.radix_kv import _extend_prefix_hash_chain

    _require(prefix_tokens % PAGE_SIZE == 0, "tier key prefix must be page aligned")
    _require(len(token_ids) >= prefix_tokens, "tier key trace is shorter than the prefix")
    _, count, _, keys = _extend_prefix_hash_chain(
        hashlib.sha256(b"("),
        0,
        token_ids[:prefix_tokens],
        PAGE_SIZE,
    )
    _require(count == prefix_tokens, "Radix tier key chain token count differs")
    _require(
        len(keys) == prefix_tokens // PAGE_SIZE and all(_valid_sha256(key) for key in keys),
        "Radix tier key chain is incomplete",
    )
    return keys


def _service_dram_control_rounds(
    control_comm: object,
    local_runner: object,
    expected_operations: Sequence[str],
) -> tuple[str, ...]:
    """Service the exact rank-0 DistTP tier transactions on a follower rank."""

    from kairyu.engine.core.worker import _dram_kv_local_ack, _dram_kv_operation

    observed: list[str] = []
    for expected in expected_operations:
        payload = control_comm.broadcast(None, src=0)
        operation = _dram_kv_operation(payload)
        _require(operation == expected, f"unexpected DRAM KV control round: {operation}")
        ack = _dram_kv_local_ack(control_comm, local_runner, payload)
        gathered = control_comm.all_gather(ack)
        _require(
            isinstance(gathered, (tuple, list))
            and len(gathered) == control_comm.world_size,
            f"DRAM KV {operation} ACK gather differs from TP degree",
        )
        observed.append(operation)
    return tuple(observed)


def _model_interval(
    callback,
    *,
    stream: object,
) -> tuple[object, dict[str, object]]:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    result = callback()
    end.record(stream)
    end.synchronize()
    return result, _cuda_span(start, end, stream)


def _sample_output(tokenizer: object, token_id: int) -> dict[str, object]:
    value = {
        "token_ids": [token_id],
        "text": tokenizer.decode((token_id,)),
        "finish_reason": "length",
        # This fixed workload does not request public logprobs.  Bind the empty
        # logical output rather than hashing backend-dependent diagnostic logits.
        "logprobs_sha256": sha256_json([]),
    }
    return {"value": value, "sha256": sha256_json(value)}


def _rank_event(
    *,
    startup: Mapping[str, object],
    kv_sha256: str,
    bytes_transferred: int,
    ready_span: Mapping[str, object],
    intervals: Sequence[Mapping[str, object]],
    staging: Mapping[str, object] | None,
) -> dict[str, object]:
    hardware = startup["hardware"]
    _require(isinstance(hardware, dict), "rank startup hardware is invalid")
    return {
        "rank": startup["rank"],
        "device_uuid": hardware["device_uuid"],
        "pci_bus_id": hardware["pci_bus_id"],
        "gpu_numa_node": hardware["gpu_numa_node"],
        "pool_fingerprint": startup["pool_fingerprint"],
        "bytes": bytes_transferred,
        "kv_sha256": kv_sha256,
        "cuda_ready_span": dict(ready_span),
        "cuda_intervals": [dict(value) for value in intervals],
        "staging": None if staging is None else dict(staging),
    }


def _numa_topology_lines(nodes: Sequence[int]) -> list[str]:
    result = []
    for node in sorted(set(nodes)):
        cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist")
        meminfo = Path(f"/sys/devices/system/node/node{node}/meminfo")
        _require(cpulist.is_file() and meminfo.is_file(), f"NUMA node {node} sysfs is incomplete")
        memory = next(
            (
                line.strip()
                for line in meminfo.read_text(encoding="utf-8").splitlines()
                if "MemTotal" in line
            ),
            "",
        )
        result.append(f"node {node} cpus={cpulist.read_text(encoding='utf-8').strip()} {memory}")
    return result


@dataclass(frozen=True)
class _LiveRankOptions:
    tp_size: int
    init_method: str
    model_path: str
    result_path: str
    token_ids: tuple[int, ...]
    prompt_trace: dict[str, object]
    max_num_batched_tokens: int
    timeout_s: float


def _ensure_python_bin_on_path() -> None:
    python_bin = Path(sys.executable).parent
    if (python_bin / "ninja").is_file():
        current = os.environ.get("PATH", "")
        parts = current.split(os.pathsep) if current else []
        if str(python_bin) not in parts:
            os.environ["PATH"] = os.pathsep.join((str(python_bin), *parts))


def _startup_runtime(
    startups: Sequence[Mapping[str, object]],
    *,
    tp_size: int,
    max_num_batched_tokens: int,
) -> tuple[dict[str, object], dict[str, object]]:
    from kairyu.engine.core.kv_tier_policy import (
        cpu_model_identity,
        engine_source_sha256,
    )
    from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY

    _require(
        len(startups) == tp_size
        and [startup.get("rank") for startup in startups] == list(range(tp_size)),
        "rank startup inventory is incomplete",
    )
    hardware_rows = []
    fingerprints: dict[str, str] = {}
    backend_names = set()
    backend_identities = set()
    tier_backends = set()
    page_nbytes = set()
    for rank, startup in enumerate(startups):
        hardware = startup.get("hardware")
        _require(isinstance(hardware, dict), f"rank {rank} hardware is invalid")
        hardware_rows.append(dict(hardware))
        fingerprint = startup.get("pool_fingerprint")
        _require(_valid_sha256(fingerprint), f"rank {rank} pool fingerprint is invalid")
        fingerprints[str(rank)] = str(fingerprint)
        backend_name = startup.get("attention_backend")
        backend_identity = startup.get("attention_backend_identity")
        _require(
            isinstance(backend_name, str) and bool(backend_name),
            f"rank {rank} attention backend is invalid",
        )
        _require(
            isinstance(backend_identity, str) and bool(backend_identity),
            f"rank {rank} attention backend identity is invalid",
        )
        backend_names.add(backend_name)
        backend_identities.add(backend_identity)
        tier_backend = startup.get("tier_backend")
        _require(
            tier_backend == TIER_BACKEND,
            f"rank {rank} did not select the required coalesced tier backend",
        )
        tier_backends.add(tier_backend)
        page_nbytes.add(startup.get("page_nbytes"))
    for hardware in hardware_rows:
        node = hardware.get("gpu_numa_node")
        hardware["local_numa_gpu_count"] = sum(
            peer.get("gpu_numa_node") == node for peer in hardware_rows
        )
    _require(len(backend_names) == 1, "TP ranks selected different attention backends")
    _require(
        len(backend_identities) == 1,
        "TP ranks selected different attention backend identities",
    )
    _require(
        tier_backends == {TIER_BACKEND},
        "TP ranks selected different DRAM transfer runtimes",
    )
    _require(len(page_nbytes) == 1, "TP ranks disagree on KV page bytes")
    transport_cohorts = {
        (
            hardware.get("pcie_max_link_speed"),
            hardware.get("pcie_max_link_width"),
            hardware.get("pcie_current_link_width"),
            hardware.get("local_numa_gpu_count"),
        )
        for hardware in hardware_rows
    }
    _require(len(transport_cohorts) == 1, "TP ranks use different PCIe transport cohorts")
    (
        pcie_max_link_speed,
        pcie_max_link_width,
        pcie_current_link_width,
        local_numa_gpu_count,
    ) = next(iter(transport_cohorts))
    expected_page_nbytes = expected_rank_bytes(tp_size, PAGE_SIZE)
    _require(
        page_nbytes == {expected_page_nbytes},
        "rank KV page bytes differ from Qwen3-32B geometry",
    )
    import torch

    raw_nccl = torch.cuda.nccl.version()
    nccl_version = (
        ".".join(str(part) for part in raw_nccl) if isinstance(raw_nccl, tuple) else str(raw_nccl)
    )
    runtime = {
        "tensor_parallel_size": tp_size,
        "engine_source_sha256": engine_source_sha256(),
        "page_size": PAGE_SIZE,
        "kv_dtype": DTYPE,
        "tier_backend": str(next(iter(tier_backends))),
        "recompute_path": "production-radix-model-prefill",
        "recompute_cache_disabled": True,
        "restore_path": RESTORE_PATH,
        "control_backend": CONTROL_BACKEND,
        "attention_backend": str(next(iter(backend_names))),
        "attention_backend_identity": str(next(iter(backend_identities))),
        "max_model_len": max(PREFIX_LENGTHS) + 1,
        "max_num_batched_tokens": max_num_batched_tokens,
        "host_memory_placement_policy": GPU_LOCAL_FIRST_TOUCH_POLICY,
        "os_page_size": os.sysconf("SC_PAGE_SIZE"),
        "pcie_max_link_speed": pcie_max_link_speed,
        "pcie_max_link_width": pcie_max_link_width,
        "pcie_current_link_width": pcie_current_link_width,
        "local_numa_gpu_count": local_numa_gpu_count,
        "cpu_model": cpu_model_identity(),
        "python_version": platform.python_version(),
        "openssl_version": ssl.OPENSSL_VERSION,
        "chunked_prefill": True,
        "kv_layout_fingerprint": sha256_json(
            {
                "rank_pool_fingerprints": fingerprints,
                "page_nbytes": expected_page_nbytes,
            }
        ),
        "rank_pool_fingerprints": fingerprints,
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "nccl_version": nccl_version,
        "driver_version": _run_text(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            )
        ).splitlines()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    topology = _run_text(("nvidia-smi", "topo", "-m")).splitlines()
    hardware = {
        "ranks": hardware_rows,
        "nvidia_smi_topology": topology,
        "numa_topology": _numa_topology_lines([int(row["gpu_numa_node"]) for row in hardware_rows]),
    }
    return runtime, hardware


def _snapshot_row_from_receipts(
    *,
    tp_size: int,
    prefix_tokens: int,
    pair_index: int,
    warmup: bool,
    pair_order: Sequence[str],
    source_pages: tuple[int, ...],
    destination_pages: tuple[int, ...],
    token_ids_sha256: str,
    receipts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    _require(len(receipts) == tp_size, "snapshot receipt rank count differs")
    rank_events = []
    digests: dict[str, object] = {}
    staging_by_rank: dict[str, object] = {}
    invalidations: list[dict[str, object]] = []
    ownership_after: list[dict[str, object]] = []
    for rank, receipt in enumerate(receipts):
        _require(receipt.get("rank") == rank, "snapshot receipts are not rank ordered")
        event = receipt.get("rank_event")
        staging = receipt.get("staging")
        digest = receipt.get("kv_sha256")
        invalidation = receipt.get("source_invalidation")
        ownership = receipt.get("source_ownership_after")
        _require(isinstance(event, dict), "snapshot rank event is invalid")
        _require(isinstance(staging, dict), "snapshot staging evidence is invalid")
        _require(_valid_sha256(digest), "snapshot rank KV digest is invalid")
        _require(isinstance(invalidation, dict), "snapshot source invalidation is invalid")
        _require(
            ownership
            == {
                "source_device_reusable": True,
                "source_device_invalidated": True,
                "tier_resident_pages": prefix_tokens // PAGE_SIZE,
                "live_transfer": False,
            },
            "snapshot source ownership receipt differs",
        )
        rank_events.append(dict(event))
        digests[str(rank)] = digest
        staging_by_rank[str(rank)] = dict(staging)
        invalidations.append(dict(invalidation))
        ownership_after.append(dict(ownership))
    page_count = prefix_tokens // PAGE_SIZE
    return (
        {
            "row_type": "snapshot",
            "tp_size": tp_size,
            "prefix_tokens": prefix_tokens,
            "pair_index": pair_index,
            "warmup": warmup,
            "pair_order": list(pair_order),
            "page_count": page_count,
            "token_ids_sha256": token_ids_sha256,
            "source_page_ids": list(source_pages),
            "destination_page_ids": list(destination_pages),
            "restore_path": RESTORE_PATH,
            "control_backend": CONTROL_BACKEND,
            "rank_events": rank_events,
            "source_kv_sha256_by_rank": digests,
            "host_kv_sha256_by_rank": dict(digests),
            "source_invalidation_rank_receipts": invalidations,
            "async_submission": True,
            "ownership": {
                "before": {
                    "source_device_owned_pages": page_count,
                    "tier_reserved_pages": 0,
                },
                "pending": {
                    "source_device_reusable": False,
                    "tier_published": False,
                },
                "after": ownership_after[0],
            },
            "retry_count": 0,
            "error": None,
            "fallback_used": False,
        },
        staging_by_rank,
    )


def _sample_row_from_receipts(
    *,
    tp_size: int,
    prefix_tokens: int,
    pair_index: int,
    warmup: bool,
    pair_order: Sequence[str],
    arm: str,
    destination_pages: tuple[int, ...],
    token_ids_sha256: str,
    source_digests: Mapping[str, object],
    controller_started_ns: int,
    controller_ended_ns: int,
    receipts: Sequence[Mapping[str, object]],
    tokenizer: object,
) -> dict[str, object]:
    _require(len(receipts) == tp_size, "sample receipt rank count differs")
    rank_events = []
    ready_digests: dict[str, object] = {}
    token_ids = set()
    host_before_values: set[int] = set()
    host_after_values: set[int] = set()
    published_values: set[int] = set()
    cache_hit_values: set[int] = set()
    restored_values: set[bool] = set()
    control_operations: set[tuple[str, ...]] = set()
    for rank, receipt in enumerate(receipts):
        _require(receipt.get("rank") == rank, "sample receipts are not rank ordered")
        event = receipt.get("rank_event")
        digest = receipt.get("kv_sha256")
        token_id = receipt.get("output_token_id")
        _require(isinstance(event, dict), "sample rank event is invalid")
        _require(_valid_sha256(digest), "sample rank KV digest is invalid")
        _require(type(token_id) is int and token_id >= 0, "sample output token is invalid")
        rank_events.append(dict(event))
        ready_digests[str(rank)] = digest
        token_ids.add(int(token_id))
        host_before = receipt.get("host_resident_pages_before")
        host_after = receipt.get("host_resident_pages_after")
        published = receipt.get("destination_published_pages")
        cache_hits = receipt.get("recompute_cache_hits")
        restored = receipt.get("restored_from_dram")
        operations = receipt.get("control_operations")
        _require(
            type(host_before) is int
            and host_before >= 0
            and type(host_after) is int
            and host_after >= 0
            and type(published) is int
            and published >= 0
            and type(cache_hits) is int
            and cache_hits >= 0
            and type(restored) is bool
            and isinstance(operations, list)
            and all(isinstance(operation, str) for operation in operations),
            "sample ownership/control receipt is invalid",
        )
        host_before_values.add(host_before)
        host_after_values.add(host_after)
        published_values.add(published)
        cache_hit_values.add(cache_hits)
        restored_values.add(restored)
        control_operations.add(tuple(operations))
    _require(ready_digests == source_digests, "restore/recompute KV bytes differ from source")
    _require(len(token_ids) == 1, "TP ranks did not adopt one output token")
    _require(
        all(
            len(values) == 1
            for values in (
                host_before_values,
                host_after_values,
                published_values,
                cache_hit_values,
                restored_values,
                control_operations,
            )
        ),
        "sample TP ranks disagree on ownership or control receipts",
    )
    token_id = next(iter(token_ids))
    host_before = next(iter(host_before_values))
    host_after = next(iter(host_after_values))
    published_pages = next(iter(published_values))
    cache_hits = next(iter(cache_hit_values))
    restored = next(iter(restored_values))
    operations = next(iter(control_operations))
    expected_operations = ("available_prefix", "restore") if arm == "restore" else ()
    _require(operations == expected_operations, "sample control operations differ from its arm")
    return {
        "row_type": "sample",
        "arm": arm,
        "tp_size": tp_size,
        "prefix_tokens": prefix_tokens,
        "pair_index": pair_index,
        "warmup": warmup,
        "pair_order": list(pair_order),
        "controller_started_ns": controller_started_ns,
        "controller_ended_ns": controller_ended_ns,
        "wall_elapsed_ns": controller_ended_ns - controller_started_ns,
        "token_ids_sha256": token_ids_sha256,
        "restore_path": RESTORE_PATH,
        "control_backend": CONTROL_BACKEND,
        "rank_events": rank_events,
        "source_kv_sha256_by_rank": dict(source_digests),
        "ready_kv_sha256_by_rank": ready_digests,
        "output": _sample_output(tokenizer, token_id),
        "ownership": {
            "before": {
                "destination_published": False,
                "host_resident_pages": host_before,
            },
            "pending": {"destination_published": False},
            "after": {
                "destination_published_pages": published_pages,
                "host_resident_pages": host_after,
                "live_transfer": False,
            },
        },
        "retry_count": 0,
        "error": None,
        "fallback_used": False,
        "production_model_forward": True,
        "recompute_cache_hits": cache_hits,
        "restored_from_dram": restored,
        "destination_page_ids_sha256": sha256_json(destination_pages),
    }


def _live_rank_entry(rank: int, options: _LiveRankOptions) -> None:  # pragma: no cover - GPU
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.attention_selector import (
        attention_backend_execution_identity,
    )
    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.kv_tier import DRAMKVTier
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.worker import DistTPModelRunner, serving_group
    from kairyu.engine.tokenizer import HFTokenizer
    from kairyu.models.parallel import build_tp_model

    _require(torch.cuda.device_count() == options.tp_size, "visible GPU count must equal --tp")
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    init_distributed(
        rank,
        options.tp_size,
        options.init_method,
        backend="nccl",
        timeout_s=options.timeout_s,
    )
    comm = TorchDistCommunicator(device=device)
    control_group = serving_group("gloo", timeout_s=options.timeout_s)
    control_comm = TorchDistCommunicator(group=control_group)
    _require(
        str(dist.get_backend()) == "nccl"
        and str(dist.get_backend(control_group)) == CONTROL_BACKEND,
        "live benchmark requires an NCCL model group and Gloo control group",
    )
    succeeded = False
    try:
        profile = probe(device)
        attention_backend = select_backend(profile, device=device)
        decision = getattr(attention_backend, "selection_decision", None)
        _require(
            decision is not None
            and isinstance(decision.components, dict)
            and "flashinfer" in decision.components.values(),
            "formal F4a run requires a FlashInfer-backed attention path; "
            "automatic Torch fallback is invalid",
        )
        attention_identity = attention_backend_execution_identity(
            decision,
            attention_backend,
        )
        model, local_config, full_config = build_tp_model(
            options.model_path,
            options.tp_size,
            rank,
            comm,
            dtype=torch.bfloat16,
            device=device,
            attention_backend=attention_backend,
        )
        _require(
            full_config.num_hidden_layers == NUM_LAYERS
            and full_config.num_key_value_heads == NUM_KV_HEADS
            and full_config.head_dim == HEAD_DIM,
            "live model is not Qwen3-32B",
        )
        pool = PagedKVPool(
            num_layers=local_config.num_hidden_layers,
            num_pages=LIVE_POOL_PAGES,
            page_size=PAGE_SIZE,
            num_kv_heads=local_config.kv_cache_num_heads,
            head_dim=local_config.kv_cache_head_dim,
            v_head_dim=local_config.kv_cache_v_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        config_digest = sha256_file(Path(options.model_path) / "config.json")
        model_id = f"{MODEL}@{MODEL_REVISION}:{config_digest}"
        tier = DRAMKVTier(
            pool,
            model_id=model_id,
            tp_size=options.tp_size,
            tp_rank=rank,
            capacity_pages=LIVE_TIER_CAPACITY_PAGES,
        )
        audit_tier = DRAMKVTier(
            pool,
            model_id=model_id,
            tp_size=options.tp_size,
            tp_rank=rank,
            capacity_pages=LIVE_TIER_CAPACITY_PAGES,
        )
        _require(tier.host_is_pinned and audit_tier.host_is_pinned, "tier slabs are not pinned")
        _require(
            tier.transfer_backend == TIER_BACKEND
            and audit_tier.transfer_backend == TIER_BACKEND,
            "formal F4a run requires one versioned coalesced CUDA transfer backend",
        )
        hardware = _hardware_from_tier(rank, tier)
        local_tier_runner = SimpleNamespace(dram_kv_tier=tier)
        coordinator = (
            DistTPModelRunner(control_comm, local_tier_runner, comm)
            if rank == 0
            else None
        )
        startup = {
            "rank": rank,
            "hardware": hardware,
            "pool_fingerprint": tier.fingerprint_digest,
            "page_nbytes": tier.page_nbytes,
            "tier_backend": tier.transfer_backend,
            "attention_backend": type(attention_backend).__name__,
            "attention_backend_identity": attention_identity,
        }
        startups = comm.all_gather(startup)
        if rank == 0:
            runtime, combined_hardware = _startup_runtime(
                startups,
                tp_size=options.tp_size,
                max_num_batched_tokens=options.max_num_batched_tokens,
            )
            tokenizer = HFTokenizer(options.model_path)
            body_rows: list[dict[str, object]] = []
        else:
            runtime = combined_hardware = tokenizer = body_rows = None

        source_first = MAX_PREFIX_PAGES
        destination_first = 0
        for repeat in (-1, *range(MEASURED_REPEATS)):
            warmup = repeat == -1
            prefixes = (
                PREFIX_LENGTHS if warmup or repeat % 2 == 0 else tuple(reversed(PREFIX_LENGTHS))
            )
            pair_order = ARMS if warmup or repeat % 2 == 0 else tuple(reversed(ARMS))
            for prefix_tokens in prefixes:
                page_count = prefix_tokens // PAGE_SIZE
                source_pages = tuple(range(source_first, source_first + page_count))
                destination_pages = tuple(range(destination_first, destination_first + page_count))
                source_keys = _radix_tier_keys(
                    options.token_ids,
                    prefix_tokens=prefix_tokens,
                )
                comm.barrier()
                _clear_page_range(pool, source_first, page_count)
                _clear_page_range(pool, destination_first, page_count)
                torch.cuda.synchronize(rank)
                _source_prefix_forward(
                    model,
                    pool,
                    options.token_ids,
                    source_pages,
                    prefix_tokens=prefix_tokens,
                    chunk_tokens=options.max_num_batched_tokens,
                )
                torch.cuda.synchronize(rank)
                comm.barrier()

                offload, offload_span = _timed_tier_transfer(
                    tier,
                    direction="offload",
                    keys=source_keys,
                    page_ids=source_pages,
                )
                source_digest = _aggregate_checksums(offload.checksums)
                staging = _staging_residency(tier, int(offload.bytes_transferred))
                source_invalidation = _invalidate_source_pages(
                    pool,
                    rank=rank,
                    page_ids=source_pages,
                    source_kv_sha256=source_digest,
                    bytes_transferred=int(offload.bytes_transferred),
                )
                offload_event = _rank_event(
                    startup=startup,
                    kv_sha256=source_digest,
                    bytes_transferred=offload.bytes_transferred,
                    ready_span=offload_span,
                    intervals=({"name": "offload_d2h", **offload_span},),
                    staging=staging,
                )
                snapshot_receipts = comm.all_gather(
                    {
                        "rank": rank,
                        "rank_event": offload_event,
                        "staging": staging,
                        "kv_sha256": source_digest,
                        "source_invalidation": source_invalidation,
                        "source_ownership_after": {
                            "source_device_reusable": offload.source_reusable,
                            "source_device_invalidated": source_invalidation["all_zero"],
                            "tier_resident_pages": tier.resident_pages,
                            "live_transfer": False,
                        },
                    }
                )
                if rank == 0:
                    snapshot_row, staging_by_rank = _snapshot_row_from_receipts(
                        tp_size=options.tp_size,
                        prefix_tokens=prefix_tokens,
                        pair_index=repeat,
                        warmup=warmup,
                        pair_order=pair_order,
                        source_pages=source_pages,
                        destination_pages=destination_pages,
                        token_ids_sha256=str(
                            options.prompt_trace["prefix_token_ids_sha256"][str(prefix_tokens)]
                        ),
                        receipts=snapshot_receipts,
                    )
                    body_rows.append(snapshot_row)
                    source_digests = snapshot_row["source_kv_sha256_by_rank"]
                else:
                    staging_by_rank = source_digests = None
                staging_by_rank = comm.broadcast(staging_by_rank, src=0)
                source_digests = comm.broadcast(source_digests, src=0)

                for arm in pair_order:
                    cache = RadixKVCache(MAX_PREFIX_PAGES, PAGE_SIZE) if rank == 0 else None
                    if rank == 0:
                        _require(coordinator is not None, "rank 0 DRAM coordinator is missing")
                        cache.attach_dram_tier(
                            coordinator,
                            min_restore_tokens=(
                                PAGE_SIZE
                                if arm == "restore"
                                else max(PREFIX_LENGTHS) + PAGE_SIZE
                            ),
                        )
                    _clear_page_range(pool, destination_first, page_count)
                    torch.cuda.synchronize(rank)
                    comm.barrier()
                    host_resident_before = tier.resident_pages
                    controller_started = time.perf_counter_ns()
                    current = torch.cuda.current_stream(rank)
                    ready_start = torch.cuda.Event(enable_timing=True)
                    ready_end = torch.cuda.Event(enable_timing=True)
                    ready_start.record(current)
                    if arm == "restore":
                        if rank == 0:
                            allocation = cache.allocate(
                                tuple(options.token_ids[:prefix_tokens])
                            )
                            control_operations = ("available_prefix", "restore")
                            _require(
                                allocation.pages == destination_pages
                                and allocation.cached_pages == destination_pages
                                and allocation.num_cached_tokens == prefix_tokens,
                                "production Radix restore did not publish the expected pages",
                            )
                            destination_published_pages = len(
                                allocation.cached_pages
                            )
                            restored_from_dram = (
                                allocation.num_cached_tokens == prefix_tokens
                            )
                        else:
                            allocation = None
                            control_operations = _service_dram_control_rounds(
                                control_comm,
                                local_tier_runner,
                                ("available_prefix", "restore"),
                            )
                            destination_published_pages = page_count
                            restored_from_dram = control_operations == (
                                "available_prefix",
                                "restore",
                            )
                        transfer_metrics = tier.last_completed_transfer
                        _require(transfer_metrics is not None, "restore DMA telemetry is missing")
                        transfer_span = _dma_span_from_transfer(
                            transfer_metrics,
                            direction="restore",
                            page_count=page_count,
                            bytes_transferred=expected_rank_bytes(
                                options.tp_size,
                                prefix_tokens,
                            ),
                        )
                        (output_result, model_span) = _model_interval(
                            partial(
                                _next_token_stage,
                                model,
                                pool,
                                comm,
                                rank=rank,
                                token_ids=options.token_ids,
                                page_ids=destination_pages,
                                prefix_tokens=prefix_tokens,
                                write_from=prefix_tokens,
                            ),
                            stream=current,
                        )
                        interval_rows = (
                            {"name": "restore_h2d", **transfer_span},
                            {"name": "next_token_model", **model_span},
                        )
                        recompute_cache_hits = 0
                    else:
                        if rank == 0:
                            allocation = cache.allocate(
                                tuple(options.token_ids[:prefix_tokens])
                            )
                            _require(
                                allocation.pages == destination_pages
                                and allocation.cached_pages == ()
                                and allocation.num_cached_tokens == 0,
                                "uncached Radix baseline did not allocate destination pages",
                            )
                            recompute_cache_hits = allocation.num_cached_tokens
                        else:
                            allocation = None
                            recompute_cache_hits = 0
                        control_operations = ()
                        (output_result, prefill_span) = _model_interval(
                            partial(
                                _recompute_stage,
                                model,
                                pool,
                                comm,
                                rank=rank,
                                token_ids=options.token_ids,
                                page_ids=destination_pages,
                                prefix_tokens=prefix_tokens,
                                chunk_tokens=options.max_num_batched_tokens,
                            ),
                            stream=current,
                        )
                        interval_rows = ({"name": "recompute_prefill", **prefill_span},)
                        if rank == 0:
                            cache.mark_computed(allocation)
                            destination_published_pages = len(
                                allocation.new_full_pages
                            )
                        else:
                            destination_published_pages = page_count
                        restored_from_dram = False
                    output_token_id, _logits = output_result
                    ready_end.record(current)
                    ready_end.synchronize()
                    comm.barrier()
                    controller_ended = time.perf_counter_ns()
                    ready_span = _cuda_span(ready_start, ready_end, current)
                    host_resident_after = tier.resident_pages

                    ready_digest = _audit_device_pages(
                        audit_tier,
                        keys=source_keys,
                        page_ids=destination_pages,
                    )
                    _require(ready_digest == source_digest, f"{arm} KV digest differs")
                    rank_event = _rank_event(
                        startup=startup,
                        kv_sha256=ready_digest,
                        bytes_transferred=(
                            expected_rank_bytes(options.tp_size, prefix_tokens)
                            if arm == "restore"
                            else 0
                        ),
                        ready_span=ready_span,
                        intervals=interval_rows,
                        staging=(staging_by_rank[str(rank)] if arm == "restore" else None),
                    )
                    sample_receipts = comm.all_gather(
                        {
                            "rank": rank,
                            "rank_event": rank_event,
                            "kv_sha256": ready_digest,
                            "output_token_id": output_token_id,
                            "host_resident_pages_before": host_resident_before,
                            "host_resident_pages_after": host_resident_after,
                            "destination_published_pages": destination_published_pages,
                            "recompute_cache_hits": recompute_cache_hits,
                            "restored_from_dram": restored_from_dram,
                            "control_operations": list(control_operations),
                        }
                    )
                    if rank == 0:
                        body_rows.append(
                            _sample_row_from_receipts(
                                tp_size=options.tp_size,
                                prefix_tokens=prefix_tokens,
                                pair_index=repeat,
                                warmup=warmup,
                                pair_order=pair_order,
                                arm=arm,
                                destination_pages=destination_pages,
                                token_ids_sha256=str(
                                    options.prompt_trace["prefix_token_ids_sha256"][
                                        str(prefix_tokens)
                                    ]
                                ),
                                source_digests=source_digests,
                                controller_started_ns=controller_started,
                                controller_ended_ns=controller_ended,
                                receipts=sample_receipts,
                                tokenizer=tokenizer,
                            )
                        )
                _require(tier.resident_pages == 0, "source tier leaked resident pages")
                _require(audit_tier.resident_pages == 0, "audit tier leaked resident pages")

        if rank == 0:
            _require(runtime is not None and combined_hardware is not None, "startup lost")
            Path(options.result_path).write_text(
                canonical_json(
                    {
                        "runtime": runtime,
                        "hardware": combined_hardware,
                        "body_rows": body_rows,
                    }
                ),
                encoding="utf-8",
            )
        comm.barrier()
        torch.cuda.synchronize(rank)
        succeeded = True
    finally:
        if succeeded:
            dist.destroy_process_group(control_group)
            dist.destroy_process_group()
        else:
            # Do not enter another collective after a rank-local failure.
            try:
                dist.destroy_process_group(control_group)
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _run_live_shard(
    *,
    tp_size: int,
    model_path: Path,
    output: Path,
    container: Mapping[str, object],
    max_num_batched_tokens: int = 2048,
    timeout_s: float = 1800.0,
) -> dict[str, object]:
    """Run one isolated TP degree and write its formal raw JSONL shard."""

    _require(tp_size in TP_DEGREES, "live run supports only TP4 or TP8")
    _require(
        type(max_num_batched_tokens) is int and max_num_batched_tokens >= PAGE_SIZE,
        "max_num_batched_tokens must be an integer >= page size",
    )
    _require(
        type(timeout_s) in {int, float} and math.isfinite(timeout_s) and timeout_s > 0,
        "distributed timeout must be positive and finite",
    )
    model_path = model_path.resolve()
    _require(model_path.is_dir(), f"model path is not a directory: {model_path}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _ensure_python_bin_on_path()

    measurement_started_at = _utc_now()
    source_start = _source_provenance_live()
    checkpoint_start = _checkpoint_provenance_live(model_path)
    token_ids, prompt_trace = _fixed_prompt_tokens(model_path)
    with tempfile.TemporaryDirectory(prefix=f"kairyu-dram-kv-tp{tp_size}-") as temporary:
        temporary_path = Path(temporary)
        init_path = temporary_path / "nccl-rendezvous"
        result_path = temporary_path / "rank0-result.json"
        options = _LiveRankOptions(
            tp_size=tp_size,
            init_method=f"file://{init_path}",
            model_path=str(model_path),
            result_path=str(result_path),
            token_ids=token_ids,
            prompt_trace=prompt_trace,
            max_num_batched_tokens=max_num_batched_tokens,
            timeout_s=float(timeout_s),
        )
        import torch.multiprocessing as mp

        mp.spawn(
            _live_rank_entry,
            args=(options,),
            nprocs=tp_size,
            join=True,
        )
        _require(result_path.is_file(), "rank 0 did not produce live measurement rows")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "live rank result must be an object")
    runtime = payload.get("runtime")
    hardware = payload.get("hardware")
    body_rows = payload.get("body_rows")
    _require(isinstance(runtime, dict), "live rank result omits runtime")
    _require(isinstance(hardware, dict), "live rank result omits hardware")
    _require(
        isinstance(body_rows, list) and len(body_rows) == len(expected_schedule()),
        "live rank result omits the fixed row schedule",
    )
    checkpoint_end = _checkpoint_provenance_live(model_path)
    source_end = _source_provenance_live()
    _require(checkpoint_start == checkpoint_end, "checkpoint changed during live run")
    _require(source_start == source_end, "source changed during live run")

    run_id = f"issue-187-tp{tp_size}-{uuid.uuid4().hex}"
    rows: list[dict[str, object]] = [
        {
            "row_type": "run",
            "schema_version": SCHEMA_VERSION,
            "measurement_kind": MEASUREMENT_KIND,
            "tp_size": tp_size,
            "run_id": run_id,
            "measurement_started_at": measurement_started_at,
            "config": formal_config(tp_size),
            "source": source_start,
            "checkpoint": checkpoint_start,
            "runtime": runtime,
            "hardware": hardware,
            "prompt_trace": prompt_trace,
            "container": dict(container),
        },
        *body_rows,
        {
            "row_type": "run_end",
            "tp_size": tp_size,
            "run_id": run_id,
            "status": "complete",
            "errors": [],
            "measurement_ended_at": _utc_now(),
            "source": source_end,
            "checkpoint": checkpoint_end,
            "row_counts": {
                "snapshot_rows": 100,
                "sample_rows": 200,
                "warmup_sample_rows": 20,
                "measured_sample_rows": 180,
            },
        },
    ]
    write_raw_shard(output, rows)
    stored = read_raw_shard(output)
    _validate_shard(stored, tp_size)
    return {
        "schema_version": SCHEMA_VERSION,
        "tp_size": tp_size,
        "output": str(output),
        "raw_sha256": sha256_file(output),
        "rows": len(stored),
        "passed_contract_validation": True,
    }


def write_raw_shard(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write canonical, newline-framed raw rows with contiguous indices."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            normalized = {**dict(row), "row_index": index}
            handle.write(f"{canonical_json(normalized)}\n")


def read_raw_shard(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            _require(
                raw.endswith(b"\n") and bool(raw.strip()),
                f"invalid JSONL framing at line {line_number}: {path}",
            )
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise EvidenceError(f"invalid JSON at line {line_number}: {path}") from error
            _require(
                isinstance(value, dict),
                f"JSONL row {line_number} is not an object: {path}",
            )
            rows.append(value)
    _require(rows, f"raw evidence is empty: {path}")
    _require(
        all(row.get("row_index") == index for index, row in enumerate(rows)),
        f"raw row indices are not contiguous: {path}",
    )
    return rows


def replay_raw(
    tp4_raw: Path,
    tp8_raw: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    paths = {4: tp4_raw, 8: tp8_raw}
    manifest = recompute_manifest(
        {tp: read_raw_shard(path) for tp, path in paths.items()},
        raw_sha256={tp: sha256_file(path) for tp, path in paths.items()},
    )
    if assert_gate and manifest["passed"] is not True:
        failed = [name for name, passed in manifest["checks"].items() if not passed]
        raise EvidenceError(f"G5 F4a evidence failed: {', '.join(failed)}")
    return manifest


def assemble_artifact(
    tp4_raw: Path,
    tp8_raw: Path,
    output_dir: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    input_paths = {4: tp4_raw, 8: tp8_raw}
    input_rows = {tp: read_raw_shard(path) for tp, path in input_paths.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {tp: output_dir / RAW_NAMES[tp] for tp in TP_DEGREES}
    for tp_size in TP_DEGREES:
        write_raw_shard(output_paths[tp_size], input_rows[tp_size])
    manifest = replay_raw(output_paths[4], output_paths[8], assert_gate=False)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    for tp_size in TP_DEGREES:
        profile = _profile(manifest, tp_size=tp_size, manifest_sha256=manifest_sha)
        (output_dir / PROFILE_NAMES[tp_size]).write_text(
            f"{canonical_json(profile)}\n",
            encoding="utf-8",
        )
    if assert_gate and manifest["passed"] is not True:
        failed = [name for name, passed in manifest["checks"].items() if not passed]
        raise EvidenceError(f"G5 F4a evidence failed: {', '.join(failed)}")
    return manifest


def verify_artifact(
    artifact_dir: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    raw_paths = {tp: artifact_dir / RAW_NAMES[tp] for tp in TP_DEGREES}
    manifest_path = artifact_dir / MANIFEST_NAME
    _require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    for path in raw_paths.values():
        _require(path.is_file(), f"missing raw evidence: {path}")
    recomputed = replay_raw(raw_paths[4], raw_paths[8], assert_gate=False)
    retained = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(retained, dict), "retained manifest is not an object")
    _require(retained == recomputed, "retained manifest differs from raw replay")
    manifest_sha = sha256_file(manifest_path)
    for tp_size in TP_DEGREES:
        profile_path = artifact_dir / PROFILE_NAMES[tp_size]
        _require(profile_path.is_file(), f"missing runtime profile: {profile_path}")
        retained_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        expected_profile = _profile(
            recomputed,
            tp_size=tp_size,
            manifest_sha256=manifest_sha,
        )
        _require(
            retained_profile == expected_profile,
            f"TP{tp_size} runtime profile differs from bound evidence",
        )
    if assert_gate and recomputed["passed"] is not True:
        failed = [name for name, passed in recomputed["checks"].items() if not passed]
        raise EvidenceError(f"G5 F4a evidence failed: {', '.join(failed)}")
    return recomputed


def _schema_summary() -> dict[str, object]:
    return {
        "raw_schema_version": SCHEMA_VERSION,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "formal_config": {f"tp{tp}": formal_config(tp) for tp in TP_DEGREES},
        "raw_rows_per_tp": len(expected_schedule()) + 2,
        "profile_top_level_keys": [
            "schema_version",
            "runtime_identity",
            "runtime_fingerprint",
            "samples",
            "policy",
            "evidence_sha256",
            "passed",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="measure one real Qwen TP degree")
    run.add_argument("--tp", type=int, choices=TP_DEGREES, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--image-id", required=True)
    run.add_argument("--container-id")
    run.add_argument("--container-id-file", type=Path)
    run.add_argument("--repo-digest", action="append", default=[])
    run.add_argument("--max-num-batched-tokens", type=int, default=2048)
    run.add_argument("--timeout-s", type=float, default=1800.0)

    assemble = subparsers.add_parser("assemble", help="seal TP4/TP8 raw evidence")
    assemble.add_argument("--tp4-raw", type=Path, required=True)
    assemble.add_argument("--tp8-raw", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--assert-gate", action="store_true")

    verify = subparsers.add_parser("verify", help="verify retained artifact files")
    verify.add_argument("--artifact-dir", type=Path, required=True)
    verify.add_argument("--assert-gate", action="store_true")

    replay = subparsers.add_parser("replay", help="recompute directly from raw JSONL")
    replay.add_argument("--tp4-raw", type=Path, required=True)
    replay.add_argument("--tp8-raw", type=Path, required=True)
    replay.add_argument("--assert-gate", action="store_true")

    subparsers.add_parser("schema", help="print the fixed evidence schema")
    args = parser.parse_args(argv)
    if args.command == "run":
        container_id = _resolve_container_id(
            container_id=args.container_id,
            container_id_file=args.container_id_file,
        )
        container = _container_provenance_live(
            container_id=container_id,
            image_id=args.image_id,
            repo_digests=args.repo_digest,
            command=list(sys.argv if argv is None else (SOURCE_PATH, *argv)),
        )
        result = _run_live_shard(
            tp_size=args.tp,
            model_path=args.model_path,
            output=args.output,
            container=container,
            max_num_batched_tokens=args.max_num_batched_tokens,
            timeout_s=args.timeout_s,
        )
    elif args.command == "assemble":
        result = assemble_artifact(
            args.tp4_raw,
            args.tp8_raw,
            args.output_dir,
            assert_gate=args.assert_gate,
        )
    elif args.command == "verify":
        result = verify_artifact(args.artifact_dir, assert_gate=args.assert_gate)
    elif args.command == "replay":
        result = replay_raw(
            args.tp4_raw,
            args.tp8_raw,
            assert_gate=args.assert_gate,
        )
    else:
        result = _schema_summary()
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
