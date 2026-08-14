#!/usr/bin/env python3
"""Generate strict engine-capture JSONL for the G4 M-A1 formal gate.

The operator is intentionally split across real lifecycle boundaries:

1. ``prepare`` runs before every engine session and takes the expensive full
   source/checkpoint start snapshot.
2. ``reference-arm`` runs the pinned TensorRT-LLM Python LLM producer inside
   its immutable image and produces ``reference_ep2``/``reference_ep4``.
3. ``kairyu-arm`` uses the production ``DistEPLauncher`` for one EP degree.
4. ``assemble`` runs only after all four sessions, takes a fresh full end
   snapshot, checks time/source/checkpoint stability, and emits the gate's
   canonical raw JSONL (4,428 rows for a successful run).

The reference fragment is a strict JSON object with the same ``session``,
``prompt``, ``free_run`` and ``teacher`` rows consumed by
``g4_ma1_qwen3_235b_nvfp4_bench.py``. It must use TensorRT-LLM's Python LLM
RequestOutput token IDs and token-ID top-64 logprobs; text/HTTP logprobs are
rejected by the formal row validators. Reference top_k=0 and Kairyu top_k=-1
are retained as actual stack arguments and normalize to the same disabled
top-k policy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

from bench import g4_ma1_qwen3_235b_nvfp4_bench as gate  # noqa: E402
from evidence.reporting import atomic_write_json, atomic_write_text  # noqa: E402

CAPTURE_SCHEMA = f"{gate.SCHEMA_VERSION}.capture.v1"
HOST_SNAPSHOT_TYPE = "host_start_snapshot"
ARM_FRAGMENT_TYPE = "arm_fragment"
SUCCESS_RAW_ROWS = 4_428

_ROUTED_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
)
_ROUTER = re.compile(r"^model\.layers\.\d+\.mlp\.gate\.")
_SHARED_EXPERT = re.compile(r"^model\.layers\.\d+\.mlp\.shared_experts\.")
_AUXILIARY_KV_SCALE = re.compile(
    r"^model\.layers\.\d+\.self_attn\.[kv]_proj\.[kv]_scale$"
)
_ATTENTION_OUTPUT_SHARD = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|weight_scale)$"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _package_version(name: str, module: object | None = None) -> str:
    if module is not None:
        value = getattr(module, "__version__", None)
        if isinstance(value, str) and value:
            return value
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required package {name!r} is not installed") from error


def _command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    value = gate.strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        cwd=_REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _source_snapshot() -> dict[str, object]:
    files: dict[str, str] = {}
    for relative in gate.BOUND_SOURCE_FILES:
        path = _REPO_ROOT / relative
        if not path.is_file():
            raise ValueError(f"bound source file is missing: {relative}")
        try:
            tracked = _git("ls-files", "--error-unmatch", "--", relative)
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"bound source file is not tracked by git: {relative}"
            ) from error
        if tracked != relative:
            raise ValueError(
                f"bound source tracking identity is ambiguous: {relative}"
            )
        files[relative] = gate.sha256_file(path)
    snapshot = {
        "commit": _git("rev-parse", "HEAD"),
        "tracked_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=all")
        ),
        "files": files,
    }
    if not gate._source_valid(snapshot):
        raise ValueError("formal capture requires one clean bound source commit")
    return snapshot


def _local_revision(model_path: Path) -> str:
    metadata = (
        model_path / ".cache/huggingface/download/config.json.metadata"
    )
    if not metadata.is_file():
        raise ValueError(
            "checkpoint revision provenance is missing: expected "
            f"{metadata}"
        )
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != gate.MODEL_REVISION:
        raise ValueError("local checkpoint revision does not match the pinned revision")
    return lines[0].strip()


def _strict_file_json(path: Path) -> dict[str, object]:
    value = gate.strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _tokenizer_files(model_path: Path) -> list[Path]:
    fixed = {
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    files = sorted(
        path
        for path in model_path.iterdir()
        if path.is_file()
        and (path.name in fixed or path.name.startswith("tokenizer."))
    )
    names = {path.name for path in files}
    required = {"tokenizer.json", "tokenizer_config.json"}
    if not required <= names:
        raise ValueError(f"checkpoint tokenizer files omit {sorted(required - names)}")
    return files


def _hash_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        digests = tuple(executor.map(gate.sha256_file, paths))
    return [
        {"name": path.name, "size": path.stat().st_size, "sha256": digest}
        for path, digest in zip(paths, digests, strict=True)
    ]


def _checkpoint_snapshot(model_path: Path) -> dict[str, object]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    quant_path = model_path / "hf_quant_config.json"
    config = _strict_file_json(config_path)
    index = _strict_file_json(index_path)
    quant_file = _strict_file_json(quant_path)
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise ValueError("checkpoint index requires weight_map and metadata objects")
    weight_names = sorted(set(weight_map.values()))
    if weight_names != list(gate.EXPECTED_WEIGHT_NAMES):
        raise ValueError("checkpoint index references an unexpected shard set")
    if len(weight_map) != gate.EXPECTED_TENSOR_COUNT:
        raise ValueError("checkpoint index tensor count differs from the formal contract")
    weights = _hash_rows([model_path / name for name in weight_names])
    tokenizer = _hash_rows(_tokenizer_files(model_path))
    architecture = {
        key: config.get(key) for key in gate.EXPECTED_ARCHITECTURE
    }
    producer = quant_file.get("producer")
    quant = quant_file.get("quantization")
    if not isinstance(producer, dict) or not isinstance(quant, dict):
        raise ValueError("hf_quant_config.json omits producer/quantization")
    exclude = quant.get("exclude_modules")
    quantization = {
        "source": "hf_quant_config.json",
        "producer_name": producer.get("name"),
        "producer_version": producer.get("version"),
        "quant_algo": quant.get("quant_algo"),
        "group_size": quant.get("group_size"),
        "checkpoint_kv_cache_quant_algo": quant.get("kv_cache_quant_algo"),
        "runtime_kv_cache_dtype": "bfloat16",
        "excluded_module_count": len(exclude) if isinstance(exclude, list) else None,
        "auxiliary_kv_scale_tensor_count": 188,
    }
    snapshot = {
        "model": gate.MODEL,
        "revision": _local_revision(model_path),
        "config_sha256": gate.sha256_file(config_path),
        "index_sha256": gate.sha256_file(index_path),
        "hf_quant_config_sha256": gate.sha256_file(quant_path),
        "index_total_size": metadata.get("total_size"),
        "tensor_count": len(weight_map),
        "weight_shards": weights,
        "weights_rollup_sha256": gate.sha256_json(weights),
        "tokenizer_files": tokenizer,
        "tokenizer_rollup_sha256": gate.sha256_json(tokenizer),
        "architecture": architecture,
        "quantization": quantization,
    }
    if not gate._checkpoint_valid(snapshot):
        raise ValueError("checkpoint does not satisfy the pinned M-A1 contract")
    return snapshot


def _prompt_rows(
    model_path: Path,
    checkpoint: Mapping[str, object],
    *,
    tokenizer: object | None = None,
) -> list[dict[str, object]]:
    if tokenizer is None:
        from kairyu.engine.tokenizer import resolve_tokenizer

        tokenizer = resolve_tokenizer(str(model_path))
    rows: list[dict[str, object]] = []
    for index, text in enumerate(gate.PROMPTS):
        token_ids = list(tokenizer.encode(text))
        row = {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "prompt",
            "prompt_id": f"p{index}",
            "prompt_index": index,
            "text": text,
            "text_sha256": gate.sha256_bytes(text.encode()),
            "token_ids": token_ids,
            "token_ids_sha256": gate.sha256_json(token_ids),
            "tokenizer_rollup_sha256": checkpoint["tokenizer_rollup_sha256"],
        }
        if not gate._prompt_valid(row):
            raise ValueError(f"invalid formal prompt tokenization at p{index}")
        rows.append(row)
    return rows


def _host_snapshot_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "prepared_at",
        "image_repo_digest",
        "image_config_id",
        "source",
        "checkpoint",
        "prompts",
    }:
        return False
    prepared = gate._utc_timestamp(value["prepared_at"])
    return (
        value["schema_version"] == CAPTURE_SCHEMA
        and value["type"] == HOST_SNAPSHOT_TYPE
        and prepared is not None
        and isinstance(value["image_config_id"], str)
        and gate._IMAGE_ID.fullmatch(value["image_config_id"]) is not None
        and gate._repo_digest_valid(value["image_repo_digest"])
        and gate._source_valid(value["source"])
        and gate._checkpoint_valid(value["checkpoint"])
        and isinstance(value["prompts"], list)
        and len(value["prompts"]) == gate.PROMPT_COUNT
        and all(gate._prompt_valid(row) for row in value["prompts"])
    )


def prepare_host_snapshot(
    *,
    model_path: Path,
    image_repo_digest: str,
    image_config_id: str,
    tokenizer: object | None = None,
) -> dict[str, object]:
    source = _source_snapshot()
    checkpoint = _checkpoint_snapshot(model_path)
    snapshot = {
        "schema_version": CAPTURE_SCHEMA,
        "type": HOST_SNAPSHOT_TYPE,
        "prepared_at": _now(),
        "image_repo_digest": image_repo_digest,
        "image_config_id": image_config_id,
        "source": source,
        "checkpoint": checkpoint,
        "prompts": _prompt_rows(model_path, checkpoint, tokenizer=tokenizer),
    }
    if not _host_snapshot_valid(snapshot):
        raise ValueError("generated host start snapshot is invalid")
    return snapshot


def _visible_gpu_rows(world_size: int) -> list[dict[str, object]]:
    import torch

    query = _command(
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    physical: list[dict[str, object]] = []
    for values in csv.reader(query.splitlines(), skipinitialspace=True):
        if len(values) != 6:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {values!r}")
        index, uuid, pci, name, driver, memory_mib = (item.strip() for item in values)
        physical.append(
            {
                "physical_index": int(index),
                "uuid": uuid,
                "pci_bus_id": pci,
                "name": name,
                "driver": driver,
                "total_memory_bytes": int(memory_mib) * 1024**2,
            }
        )
    by_index = {row["physical_index"]: row for row in physical}
    by_uuid = {row["uuid"]: row for row in physical}
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = (
        [item.strip() for item in raw_visible.split(",")]
        if raw_visible
        else []
    )
    selected: list[dict[str, object]] = []
    if selectors:
        for selector in selectors[:world_size]:
            row = by_uuid.get(selector)
            if row is None and selector.startswith("GPU-"):
                matches = [
                    candidate
                    for uuid, candidate in by_uuid.items()
                    if uuid.startswith(selector)
                ]
                row = matches[0] if len(matches) == 1 else None
            if row is None and selector.isdigit():
                row = by_index.get(int(selector))
            # In an NVIDIA container, nvidia-smi may already expose a renumbered
            # subset while CUDA_VISIBLE_DEVICES still names host indices.
            if row is None and len(physical) == world_size:
                row = physical[len(selected)]
            if row is None:
                raise RuntimeError(f"cannot resolve CUDA visible selector {selector!r}")
            selected.append(row)
    else:
        selected = physical[:world_size]
    if len(selected) != world_size or torch.cuda.device_count() != world_size:
        raise RuntimeError(
            f"arm requires exactly {world_size} visible GPUs; "
            f"torch={torch.cuda.device_count()}, nvidia-smi={len(selected)}"
        )
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(selected):
        properties = torch.cuda.get_device_properties(rank)
        if properties.name != row["name"]:
            raise RuntimeError(
                f"rank {rank} torch/nvidia-smi GPU name mismatch: "
                f"{properties.name!r} != {row['name']!r}"
            )
        pci = str(row["pci_bus_id"])
        linux_pci = pci[-12:].lower()
        numa_path = Path("/sys/bus/pci/devices") / linux_pci / "numa_node"
        if not numa_path.is_file():
            raise RuntimeError(f"NUMA evidence missing for PCI device {pci}")
        numa_node = int(numa_path.read_text(encoding="utf-8").strip())
        rows.append(
            {
                "rank": rank,
                "local_device_index": rank,
                "uuid": row["uuid"],
                "pci_bus_id": pci,
                "name": row["name"],
                "sm": properties.major * 10 + properties.minor,
                "total_memory_bytes": int(properties.total_memory),
                "numa_node": numa_node,
            }
        )
    return rows


def _environment(arm: str) -> dict[str, object]:
    import torch

    world_size = gate._arm_world_size(arm)
    gpus = _visible_gpu_rows(world_size)
    reference = gate._arm_stack(arm) == "reference"
    packages = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    if reference:
        import tensorrt_llm

        packages["tensorrt_llm"] = _package_version(
            "tensorrt_llm", tensorrt_llm
        )
    else:
        import flashinfer

        packages["kairyu"] = _package_version("kairyu")
        packages["flashinfer"] = _package_version(
            "flashinfer-python", flashinfer
        )
    nccl_version = torch.cuda.nccl.version()
    if isinstance(nccl_version, tuple):
        nccl = ".".join(str(item) for item in nccl_version)
    else:
        nccl = str(nccl_version)
    topology_raw = _command("nvidia-smi", "topo", "-m")
    result = {
        "packages": packages,
        "cuda": {
            "runtime": str(torch.version.cuda),
            "driver": _command(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ).splitlines()[0].strip(),
            "nccl": nccl,
        },
        "visibility": {
            "CUDA_VISIBLE_DEVICES": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "<unset:all-visible>"
            ),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get(
                "NVIDIA_VISIBLE_DEVICES", "<unset:all-visible>"
            ),
            "resolved_rank_uuid_order": [row["uuid"] for row in gpus],
        },
        "gpus": gpus,
        "topology": {
            "gpu_uuid_order": [row["uuid"] for row in gpus],
            "gpu_pci_bus_id_order": [row["pci_bus_id"] for row in gpus],
            "gpu_numa_node_order": [row["numa_node"] for row in gpus],
            "nvidia_smi_topology_raw": topology_raw,
            "nvidia_smi_topology_sha256": gate.sha256_bytes(
                topology_raw.encode()
            ),
        },
    }
    if not gate._environment_valid(result, arm=arm):
        raise RuntimeError(f"captured environment is invalid for {arm}")
    return result


def _container_observation(
    path: Path,
    *,
    image_repo_digest: str,
    image_config_id: str,
) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    value = gate.strict_json_loads(raw)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("docker container inspect evidence must be a one-item array")
    inspected = value[0]
    config = inspected.get("Config")
    state = inspected.get("State")
    container_id = inspected.get("Id")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("docker container inspect omits Config/State")
    labels = config.get("Labels")
    if labels is not None and not isinstance(labels, dict):
        raise ValueError("docker container inspect Config.Labels is malformed")
    hostname = os.environ.get("HOSTNAME")
    if (
        not isinstance(container_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        or not isinstance(hostname, str)
        or not container_id.startswith(hostname)
    ):
        raise ValueError("docker inspect container ID does not bind this process hostname")
    if (
        inspected.get("Image") != image_config_id
        or config.get("Image") != image_repo_digest
        or state.get("Running") is not True
    ):
        raise ValueError("docker inspect image identity/running state differs from the arm")
    result = {
        "source": "docker-container-inspect",
        "container_id": container_id,
        "image_repo_digest": image_repo_digest,
        "image_config_id": image_config_id,
        "running": True,
        "started_at": state.get("StartedAt"),
        "inspect_sha256": gate.sha256_bytes(raw.encode()),
        "source_revision": (
            labels.get("org.opencontainers.image.revision")
            if isinstance(labels, dict)
            else None
        ),
    }
    if not gate._container_observation_valid(result):
        raise ValueError("docker container observation is invalid")
    return result


def _tensor_sizes(model_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    index = _strict_file_json(model_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index weight_map is missing")
    mapping = {str(name): str(shard) for name, shard in weight_map.items()}
    sizes: dict[str, int] = {}
    for shard in sorted(set(mapping.values())):
        path = model_path / shard
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"truncated safetensors header: {shard}")
            length = struct.unpack("<Q", raw_length)[0]
            header = gate.strict_json_loads(handle.read(length).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError(f"invalid safetensors header object: {shard}")
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(metadata, dict):
                raise ValueError(f"invalid tensor metadata for {name}")
            offsets = metadata.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(type(item) is int for item in offsets)
                or offsets[1] <= offsets[0]
            ):
                raise ValueError(f"invalid tensor offsets for {name}")
            if name in sizes:
                raise ValueError(f"duplicate tensor header entry {name}")
            sizes[name] = offsets[1] - offsets[0]
    if set(sizes) != set(mapping):
        raise ValueError("safetensors headers and index tensor names differ")
    return mapping, sizes


def _geometry_digest(names: Sequence[str], sizes: Mapping[str, int]) -> str:
    return gate.sha256_json([[name, sizes[name]] for name in sorted(names)])


def _attention_output_sharded_names(names: Sequence[str]) -> list[str]:
    matched: dict[tuple[int, str], str] = {}
    for name in names:
        match = _ATTENTION_OUTPUT_SHARD.fullmatch(name)
        if match is None:
            continue
        key = (int(match.group(1)), match.group(2))
        if key in matched:
            raise ValueError(f"duplicate attention-output checkpoint member {key!r}")
        matched[key] = name
    expected = {
        (layer, member)
        for layer in range(gate.NUM_LAYERS)
        for member in ("weight", "weight_scale")
    }
    if set(matched) != expected:
        missing = sorted(expected - set(matched))
        extra = sorted(set(matched) - expected)
        raise ValueError(
            "checkpoint attention-output shard inventory differs from exactly "
            f"94 x (weight, weight_scale): missing={missing[:4]!r}, "
            f"extra={extra[:4]!r}"
        )
    return sorted(matched.values())


def _attention_output_shard_contracts(
    names: Sequence[str],
    sizes: Mapping[str, int],
    *,
    world_size: int,
) -> list[dict[str, object]]:
    if len(names) != gate.ATTENTION_OUTPUT_SHARDED_TENSOR_COUNT:
        raise ValueError("attention-output shard tensor count is not exact")
    ordered = sorted(names)
    if any(sizes[name] <= 0 or sizes[name] % world_size for name in ordered):
        raise ValueError(
            "attention-output checkpoint members must split evenly across EP ranks"
        )
    full_bytes = sum(sizes[name] for name in ordered)
    rank_slice_bytes = sum(sizes[name] // world_size for name in ordered)
    common = {
        "placement": "row_parallel",
        "checkpoint_axis": 1,
        "world_size": world_size,
        "tensor_count": len(ordered),
        "tensor_names_sha256": gate.sha256_json(ordered),
        "full_geometry_sha256": _geometry_digest(ordered, sizes),
        "full_bytes": full_bytes,
        "slice_denominator": world_size,
        "rank_slice_bytes": rank_slice_bytes,
    }
    rows = [
        {
            **common,
            "rank": rank,
            "slice_start_numerator": rank,
            "slice_end_numerator": rank + 1,
        }
        for rank in range(world_size)
    ]
    rank_slice_digests = [
        gate._attention_output_rank_slice_sha256(row) for row in rows
    ]
    partition_sha256 = gate.sha256_json(rank_slice_digests)
    return [
        {
            **row,
            "rank_slice_sha256": rank_slice_digests[rank],
            "partition_sha256": partition_sha256,
        }
        for rank, row in enumerate(rows)
    ]


def _ownership_rows(
    *,
    model_path: Path,
    arm: str,
    load_info: object | None,
) -> tuple[list[dict[str, object]], list[list[str]]]:
    world_size = gate._arm_world_size(arm)
    _mapping, sizes = _tensor_sizes(model_path)
    all_names = set(sizes)
    routed: dict[int, list[str]] = defaultdict(list)
    for name in all_names:
        match = _ROUTED_EXPERT.match(name)
        if match is not None:
            routed[int(match.group(2))].append(name)
    if set(routed) != set(range(gate.NUM_EXPERTS)):
        raise ValueError("checkpoint routed expert inventory is incomplete")
    replicated_checkpoint = all_names - {
        name for names in routed.values() for name in names
    }
    attention_output = _attention_output_sharded_names(
        sorted(replicated_checkpoint)
    )
    attention_output_set = set(attention_output)
    auxiliary_kv_scales = {
        name for name in replicated_checkpoint if _AUXILIARY_KV_SCALE.match(name)
    }
    if len(auxiliary_kv_scales) != 188:
        raise ValueError(
            "checkpoint auxiliary KV-scale inventory differs from 94 K + 94 V"
        )
    # These calibration scalars remain authoritative checkpoint provenance but
    # BF16 KV runtime deliberately does not load them into rank-local modules.
    replicated = (
        replicated_checkpoint - auxiliary_kv_scales - attention_output_set
    )
    if len(replicated) != gate.FULLY_REPLICATED_TENSOR_COUNT:
        raise ValueError("fully replicated checkpoint tensor count is not exact")
    router = sorted(name for name in replicated if _ROUTER.match(name))
    shared = sorted(name for name in replicated if _SHARED_EXPERT.match(name))
    dense = sorted(replicated - set(router) - set(shared))
    experts_per_rank = gate.NUM_EXPERTS // world_size
    rows: list[dict[str, object]] = []
    projection_names: list[list[str]] = []
    attention_output_contracts = _attention_output_shard_contracts(
        attention_output,
        sizes,
        world_size=world_size,
    )
    fully_replicated_bytes = sum(sizes[name] for name in replicated)
    for rank in range(world_size):
        owned_indices = list(
            range(rank * experts_per_rank, (rank + 1) * experts_per_rank)
        )
        owned_names = sorted(
            name for expert in owned_indices for name in routed[expert]
        )
        loaded_names = sorted((*replicated, *attention_output, *owned_names))
        projections = sorted(
            name[: -len(".weight_scale_2")]
            for name in loaded_names
            if name.endswith(".weight_scale_2")
        )
        projection_names.append(projections)
        owned_expert_bytes = sum(sizes[name] for name in owned_names)
        attention_output_contract = attention_output_contracts[rank]
        row = {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "ownership",
            "arm": arm,
            "rank": rank,
            "world_size": world_size,
            "layers": gate.NUM_LAYERS,
            "num_experts": gate.NUM_EXPERTS,
            "owned_expert_indices": owned_indices,
            "owned_expert_tensor_count": len(owned_names),
            "owned_expert_tensor_names_sha256": gate.sha256_json(owned_names),
            "owned_expert_bytes": owned_expert_bytes,
            "fully_replicated_tensor_count": len(replicated),
            "fully_replicated_bytes": fully_replicated_bytes,
            "fully_replicated_sha256": _geometry_digest(replicated, sizes),
            "attention_output_shard": attention_output_contract,
            "rank_loaded_tensor_count": len(loaded_names),
            "rank_loaded_bytes": (
                fully_replicated_bytes
                + owned_expert_bytes
                + attention_output_contract["rank_slice_bytes"]
            ),
            "loaded_tensor_names_sha256": gate.sha256_json(loaded_names),
            "dense_replication_sha256": _geometry_digest(dense, sizes),
            "router_replication_sha256": _geometry_digest(router, sizes),
            "shared_expert_replication_sha256": _geometry_digest(shared, sizes),
            "remote_expert_tensor_reads": 0,
            "unresolved_meta_tensors": 0,
            "evidence_source": "checkpoint-index-and-build-contract",
            "load_contract": {
                "quantization_source": "hf_quant_config.json",
                "quantization_method": "nvfp4",
                "checkpoint_kv_cache_quant_algo": "FP8",
                "runtime_kv_cache_dtype": "bfloat16",
                "producer_name": "modelopt",
                "producer_version": "0.33.0",
                "checkpoint_tensor_count": gate.EXPECTED_TENSOR_COUNT,
                "auxiliary_kv_scale_count": 188,
                "ownership": "contiguous-global-expert-blocks",
            },
        }
        if not gate._ownership_valid(row):
            raise RuntimeError(f"derived ownership evidence is invalid for rank {rank}")
        rows.append(row)
    if load_info is not None:
        rank_zero = rows[0]
        expected = {
            "ep_rank": 0,
            "ep_size": world_size,
            "owned_expert_indices": tuple(rank_zero["owned_expert_indices"]),
            "quantization_source": "hf_quant_config.json",
            "quantization_method": "nvfp4",
            "kv_cache_quant_algo": "FP8",
            "producer_name": "modelopt",
            "producer_version": "0.33.0",
            "checkpoint_tensor_count": gate.EXPECTED_TENSOR_COUNT,
            "rank_loaded_tensor_count": rank_zero["rank_loaded_tensor_count"],
            "auxiliary_kv_scale_count": 188,
        }
        observed = {name: getattr(load_info, name, None) for name in expected}
        if observed != expected:
            raise RuntimeError(
                f"rank-0 load probe disagrees with checkpoint contract: {observed!r}"
            )
    return rows, projection_names


def _kernel_rows(
    *,
    arm: str,
    projection_names: Sequence[Sequence[str]],
    provider_version: str,
    runtime_probes: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    reference = gate._arm_stack(arm) == "reference"
    if reference and runtime_probes is not None:
        raise ValueError("reference kernel rows do not accept Kairyu runtime probes")
    if not reference and (
        runtime_probes is None
        or len(runtime_probes) != len(projection_names)
    ):
        raise ValueError("Kairyu kernel rows require one runtime probe per rank")
    rows = []
    for rank, names in enumerate(projection_names):
        observed = (
            None
            if runtime_probes is None
            else dict(runtime_probes[rank])
        )
        row = {
            "rank": rank,
            "sm": gate.EXPECTED_SM,
            "requested": "nvfp4",
            "resolved": (
                "tensorrt_llm.pytorch.MoeConfig.CUTLASS"
                if reference
                else "flashinfer.cutlass_fused_moe+flashinfer.mm_fp4"
            ),
            "provider": "cutlass" if reference else "flashinfer",
            "provider_version": provider_version,
            "checkpoint_layout": "modelopt-nvfp4-group16",
            "weight_dtype": "uint8-packed-e2m1",
            "scale_dtype": "float8_e4m3fn",
            "global_scale_dtype": "float32",
            "activation_dtype": "nvfp4-e2m1",
            "compute_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "quantized_projection_count": len(names),
            "native_nvfp4_projection_count": len(names),
            "fallback_projection_count": 0,
            "projection_inventory_sha256": gate.sha256_json(list(names)),
            "fallback": None,
            "execution_evidence": (
                {
                    "kind": "singleton-backend-successful-forward",
                    "projection_inventory_source": "checkpoint-index",
                    "nvfp4_allowed_backends": ["cutlass"],
                    "moe_backend": "CUTLASS",
                    "successful_forward": True,
                }
                if reference
                else {
                    "kind": "all-rank-runtime-module-probe",
                    "projection_inventory_source": (
                        "rank-local-model.named_modules"
                    ),
                    "probe": "DistEPModelRunner.ep_kernel_inventory_metadata",
                    "successful_forward": True,
                    "observed": observed,
                }
            ),
        }
        if not gate._kernel_rank_valid(row, arm=arm, rank=rank):
            raise RuntimeError(f"invalid kernel inventory for {arm} rank {rank}")
        rows.append(row)
    return rows


def _reference_overlay(
    source_model: Path,
    runtime_model: Path,
) -> dict[str, object]:
    source_quant_path = source_model / "hf_quant_config.json"
    source_quant_bytes = source_quant_path.read_bytes()
    source_quant = gate.strict_json_loads(source_quant_bytes.decode("utf-8"))
    if not isinstance(source_quant, dict):
        raise ValueError("source hf_quant_config.json must be an object")
    quantization = source_quant.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("source quantization object is missing")
    removed = quantization.pop("kv_cache_quant_algo", None)
    if removed != "FP8":
        raise ValueError(f"expected source KV calibration FP8, got {removed!r}")
    runtime_model.mkdir(mode=0o755)
    for entry in source_model.iterdir():
        if entry.name == "hf_quant_config.json":
            continue
        (runtime_model / entry.name).symlink_to(
            entry, target_is_directory=entry.is_dir()
        )
    runtime_quant_bytes = (
        json.dumps(source_quant, indent=2, sort_keys=True) + "\n"
    ).encode()
    (runtime_model / "hf_quant_config.json").write_bytes(runtime_quant_bytes)
    weight_links: list[dict[str, object]] = []
    for name in gate.EXPECTED_WEIGHT_NAMES:
        source_weight = source_model / name
        runtime_weight = runtime_model / name
        source_stat = source_weight.stat()
        runtime_stat = runtime_weight.stat()
        if (
            not runtime_weight.is_symlink()
            or source_stat.st_dev != runtime_stat.st_dev
            or source_stat.st_ino != runtime_stat.st_ino
        ):
            raise RuntimeError(f"TRT runtime weight is not an exact link: {name}")
        weight_links.append(
            {
                "name": name,
                "symlink_target": os.readlink(runtime_weight),
                "source_device": source_stat.st_dev,
                "source_inode": source_stat.st_ino,
                "runtime_device": runtime_stat.st_dev,
                "runtime_inode": runtime_stat.st_ino,
                "size_bytes": source_stat.st_size,
            }
        )
    result = {
        "kind": "bf16-kv-metadata-overlay",
        "source_model_path": str(source_model),
        "runtime_model_path": str(runtime_model),
        "source_hf_quant_config_sha256": hashlib.sha256(
            source_quant_bytes
        ).hexdigest(),
        "runtime_hf_quant_config_sha256": hashlib.sha256(
            runtime_quant_bytes
        ).hexdigest(),
        "removed_json_path": "quantization.kv_cache_quant_algo",
        "removed_value": removed,
        "kv_cache_config_dtype_argument": "auto",
        "effective_kv_cache_dtype": "bfloat16",
        "weight_links": weight_links,
    }
    return result


def _ranked_logprobs(
    selected_token_id: int,
    values: Mapping[object, object],
) -> tuple[float, list[dict[str, object]]]:
    normalized: dict[int, float] = {}
    for raw_token_id, raw_logprob in values.items():
        token_id = int(raw_token_id)
        value = getattr(raw_logprob, "logprob", raw_logprob)
        logprob = float(value)
        if not gate._finite_float(logprob):
            raise RuntimeError(f"non-finite logprob for token {token_id}")
        normalized[token_id] = logprob
    if len(normalized) != gate.TOP_LOGPROBS:
        raise RuntimeError(
            f"expected exactly {gate.TOP_LOGPROBS} token-ID logprobs, "
            f"got {len(normalized)}"
        )
    if selected_token_id not in normalized:
        raise RuntimeError(f"selected token {selected_token_id} absent from top64")
    selected_logprob = normalized.pop(selected_token_id)
    rest = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    if rest and rest[0][1] > selected_logprob:
        raise RuntimeError("greedy selected token is not the highest-logprob token")
    rows = [
        {"token_id": selected_token_id, "logprob": selected_logprob},
        *(
            {"token_id": token_id, "logprob": logprob}
            for token_id, logprob in rest
        ),
    ]
    return selected_logprob, rows


def _completion(
    output: object, *, expected_tokens: int
) -> tuple[
    object,
    list[int],
    list[float],
    list[list[dict[str, object]]],
]:
    candidates = getattr(output, "outputs", None)
    if not isinstance(candidates, Sequence) or len(candidates) != 1:
        raise RuntimeError("engine output must contain exactly one completion")
    completion = candidates[0]
    token_ids = [int(token) for token in (getattr(completion, "token_ids", None) or ())]
    logprobs = getattr(completion, "logprobs", None)
    if len(token_ids) != expected_tokens or not isinstance(logprobs, Sequence):
        raise RuntimeError(
            f"expected {expected_tokens} tokens/logprob positions, got "
            f"tokens={len(token_ids)}, logprobs={None if logprobs is None else len(logprobs)}"
        )
    if len(logprobs) != expected_tokens:
        raise RuntimeError("completion logprob position count differs from token count")
    selected_logprobs: list[float] = []
    top_rows: list[list[dict[str, object]]] = []
    for token_id, values in zip(token_ids, logprobs, strict=True):
        if not isinstance(values, Mapping):
            raise RuntimeError("completion logprobs must be token-ID maps")
        selected, top = _ranked_logprobs(token_id, values)
        selected_logprobs.append(selected)
        top_rows.append(top)
    return completion, token_ids, selected_logprobs, top_rows


def _as_output_batch(value: object, *, expected: int) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        outputs = list(value)
    else:
        outputs = [value]
    if len(outputs) != expected:
        raise RuntimeError(f"expected {expected} request outputs, got {len(outputs)}")
    return outputs


def _free_rows_from_outputs(
    *, arm: str, outputs: Sequence[object]
) -> tuple[list[dict[str, object]], list[list[int]]]:
    rows: list[dict[str, object]] = []
    canonical: list[list[int]] = []
    for index, output in enumerate(outputs):
        completion, token_ids, selected, _top = _completion(
            output, expected_tokens=gate.POSITIONS
        )
        finish_reason = getattr(completion, "finish_reason", None)
        if str(finish_reason).lower() != "length":
            raise RuntimeError(
                f"{arm} p{index} free run finish reason is {finish_reason!r}"
            )
        row = {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "free_run",
            "arm": arm,
            "prompt_id": f"p{index}",
            "output_token_ids": token_ids,
            "output_token_ids_sha256": gate.sha256_json(token_ids),
            "selected_logprobs": selected,
            "finish_reason": "length",
            "source_api": gate._result_api(arm),
        }
        if not gate._free_run_valid(row):
            raise RuntimeError(f"invalid free-run evidence for {arm} p{index}")
        rows.append(row)
        canonical.append(token_ids)
    return rows, canonical


def _teacher_rows_from_outputs(
    *,
    arm: str,
    position: int,
    outputs: Sequence[object],
    prompt_tokens: Sequence[Sequence[int]],
    canonical: Sequence[Sequence[int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, output in enumerate(outputs):
        _completion_value, selected_ids, selected_lps, top = _completion(
            output, expected_tokens=1
        )
        prefix = [*prompt_tokens[index], *canonical[index][:position]]
        row = {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "teacher",
            "arm": arm,
            "prompt_id": f"p{index}",
            "position": position,
            "prefix_token_count": len(prefix),
            "prefix_token_ids_sha256": gate.sha256_json(prefix),
            "canonical_token_id": canonical[index][position],
            "selected_token_id": selected_ids[0],
            "selected_logprob": selected_lps[0],
            "top_logprobs": top[0],
            "source_api": gate._result_api(arm),
        }
        if not gate._teacher_valid(row):
            raise RuntimeError(
                f"invalid teacher evidence for {arm} p{index} position {position}"
            )
        rows.append(row)
    return rows


def _reference_teacher_canonical(
    *,
    reference_fragment: Mapping[str, object],
    prompt_tokens: Sequence[Sequence[int]],
) -> list[list[int]]:
    arm = str(reference_fragment["arm"])
    rows = {
        (row["prompt_id"], row["position"]): row
        for row in reference_fragment["teacher"]
    }
    canonical: list[list[int]] = [[] for _ in prompt_tokens]
    for position in range(gate.POSITIONS):
        for index, prompt in enumerate(prompt_tokens):
            prompt_id = f"p{index}"
            row = rows[(prompt_id, position)]
            expected_prefix = [*prompt, *canonical[index]]
            token = row["selected_token_id"]
            if (
                row["arm"] != arm
                or row["canonical_token_id"] != token
                or row["prefix_token_count"] != len(expected_prefix)
                or row["prefix_token_ids_sha256"]
                != gate.sha256_json(expected_prefix)
            ):
                raise ValueError(
                    f"{arm} teacher rollout is not canonical at "
                    f"{prompt_id} position {position}"
                )
            canonical[index].append(token)
    return canonical


def _verify_output_prompts(
    outputs: Sequence[object], expected: Sequence[Sequence[int]]
) -> None:
    for index, (output, tokens) in enumerate(zip(outputs, expected, strict=True)):
        observed = tuple(int(token) for token in getattr(output, "prompt_token_ids", ()))
        if observed != tuple(tokens):
            raise RuntimeError(
                f"engine retokenized p{index}: expected {tuple(tokens)!r}, "
                f"got {observed!r}"
            )


def _reference_bindings() -> dict[str, object]:
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig
    from tensorrt_llm.llmapi.llm_args import Nvfp4GemmConfig

    if str(tensorrt_llm.__version__) != gate.REFERENCE_RUNTIME["version"]:
        raise RuntimeError(
            f"expected TensorRT-LLM {gate.REFERENCE_RUNTIME['version']}, "
            f"got {tensorrt_llm.__version__}"
        )
    return {
        "LLM": LLM,
        "SamplingParams": SamplingParams,
        "KvCacheConfig": KvCacheConfig,
        "MoeConfig": MoeConfig,
        "Nvfp4GemmConfig": Nvfp4GemmConfig,
        "version": str(tensorrt_llm.__version__),
    }


def capture_reference_arm(
    *,
    model_path: Path,
    world_size: int,
    host_snapshot: Mapping[str, object],
    bindings: Mapping[str, object] | None = None,
    image_repo_digest: str | None = None,
    image_config_id: str | None = None,
    container_observation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if world_size not in gate.WORLD_SIZES:
        raise ValueError("reference arm world_size must be 2 or 4")
    arm = gate.REFERENCE_ARM[world_size]
    if not _host_snapshot_valid(host_snapshot):
        raise ValueError("invalid host start snapshot")
    image_repo_digest = image_repo_digest or str(
        gate.REFERENCE_RUNTIME["image_repo_digest"]
    )
    image_config_id = image_config_id or str(
        gate.REFERENCE_RUNTIME["image_config_id"]
    )
    if (
        image_repo_digest != gate.REFERENCE_RUNTIME["image_repo_digest"]
        or image_config_id != gate.REFERENCE_RUNTIME["image_config_id"]
    ):
        raise ValueError("reference image RepoDigest/config ID differs from the pin")
    if (
        not gate._container_observation_valid(container_observation)
        or container_observation["image_repo_digest"] != image_repo_digest
        or container_observation["image_config_id"] != image_config_id
    ):
        raise ValueError("reference arm requires matching docker inspect evidence")
    prompt_rows = host_snapshot["prompts"]
    prompt_tokens = [list(row["token_ids"]) for row in prompt_rows]
    max_input_len = max(len(tokens) for tokens in prompt_tokens) + gate.POSITIONS
    required_kv_tokens = sum(
        len(tokens) + gate.POSITIONS for tokens in prompt_tokens
    )
    if required_kv_tokens > gate.KV_CACHE_CAPACITY_TOKENS:
        raise RuntimeError(
            "fixed reference KV capacity is smaller than the formal workload: "
            f"required={required_kv_tokens}, "
            f"capacity={gate.KV_CACHE_CAPACITY_TOKENS}"
        )
    started_at = _now()
    checkpoint_start = _checkpoint_snapshot(model_path)
    if checkpoint_start != host_snapshot["checkpoint"]:
        raise RuntimeError("reference session start checkpoint differs from prepare")
    environment_start = _environment(arm)
    if bindings is None:
        bindings = _reference_bindings()
    llm_type = bindings["LLM"]
    sampling_type = bindings["SamplingParams"]
    kv_type = bindings["KvCacheConfig"]
    moe_type = bindings["MoeConfig"]
    nvfp4_type = bindings["Nvfp4GemmConfig"]
    provider_version = f"bundled-with-tensorrt_llm-{bindings['version']}"
    _unused_ownership, projections = _ownership_rows(
        model_path=model_path,
        arm=gate.KAIRYU_ARM[world_size],
        load_info=None,
    )
    llm = None
    with tempfile.TemporaryDirectory(
        prefix=f"kairyu-g4-ma1-trt-ep{world_size}-", dir="/tmp"
    ) as temporary:
        runtime_model = Path(temporary) / "model"
        checkpoint_view = _reference_overlay(model_path, runtime_model)
        if not gate._checkpoint_view_valid(
            checkpoint_view,
            arm=arm,
            checkpoint=host_snapshot["checkpoint"],
        ):
            raise RuntimeError("generated TRT checkpoint overlay evidence is invalid")
        kv_config = kv_type(
            enable_block_reuse=False,
            enable_partial_reuse=False,
            copy_on_partial_reuse=False,
            max_tokens=gate.KV_CACHE_CAPACITY_TOKENS,
            free_gpu_memory_fraction=(
                gate.REFERENCE_KV_FREE_GPU_MEMORY_FRACTION
            ),
            dtype="auto",
        )
        moe_config = moe_type(
            backend="CUTLASS",
            disable_finalize_fusion=True,
            use_low_precision_moe_combine=False,
        )
        nvfp4_config = nvfp4_type(allowed_backends=["cutlass"])
        llm_arguments = {
            "model": str(runtime_model),
            "tensor_parallel_size": world_size,
            "pipeline_parallel_size": 1,
            "context_parallel_size": 1,
            "gpus_per_node": world_size,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": world_size,
            "enable_attention_dp": False,
            "attention_dp_config": None,
            "dtype": "bfloat16",
            "kv_cache_config": kv_config,
            "max_batch_size": gate.PROMPT_COUNT,
            "max_input_len": max_input_len,
            "max_seq_len": max_input_len + gate.POSITIONS,
            "max_num_tokens": gate.KV_CACHE_CAPACITY_TOKENS,
            "moe_config": moe_config,
            "nvfp4_gemm_config": nvfp4_config,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "attn_backend": "TRTLLM",
            "enable_autotuner": True,
            "print_iter_log": True,
        }
        try:
            llm = llm_type(**llm_arguments)
            free_params = sampling_type(
                max_tokens=gate.POSITIONS,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                seed=gate.SEED,
                logprobs=gate.TOP_LOGPROBS,
                ignore_eos=True,
                detokenize=False,
                add_special_tokens=True,
                skip_special_tokens=False,
            )
            free_outputs = _as_output_batch(
                llm.generate(
                    [row["text"] for row in prompt_rows],
                    free_params,
                    use_tqdm=False,
                ),
                expected=gate.PROMPT_COUNT,
            )
            _verify_output_prompts(free_outputs, prompt_tokens)
            free_rows, _free_diagnostic = _free_rows_from_outputs(
                arm=arm, outputs=free_outputs
            )
            canonical: list[list[int]] = [[] for _ in prompt_tokens]
            teacher_params = sampling_type(
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                seed=gate.SEED,
                logprobs=gate.TOP_LOGPROBS,
                ignore_eos=True,
                detokenize=False,
                add_special_tokens=False,
                skip_special_tokens=False,
            )
            teacher_rows: list[dict[str, object]] = []
            for position in range(gate.POSITIONS):
                prefixes = [
                    [*prompt_tokens[index], *canonical[index][:position]]
                    for index in range(gate.PROMPT_COUNT)
                ]
                outputs = _as_output_batch(
                    llm.generate(prefixes, teacher_params, use_tqdm=False),
                    expected=gate.PROMPT_COUNT,
                )
                _verify_output_prompts(outputs, prefixes)
                for index, output in enumerate(outputs):
                    _completion_value, selected_ids, _selected_lps, _top = (
                        _completion(output, expected_tokens=1)
                    )
                    canonical[index].append(selected_ids[0])
                wave = _teacher_rows_from_outputs(
                    arm=arm,
                    position=position,
                    outputs=outputs,
                    prompt_tokens=prompt_tokens,
                    canonical=canonical,
                )
                teacher_rows.extend(wave)
        finally:
            if llm is not None:
                llm.shutdown()
    environment_end = _environment(arm)
    checkpoint_end = _checkpoint_snapshot(model_path)
    if checkpoint_end != checkpoint_start:
        raise RuntimeError("reference checkpoint changed during the engine session")
    finished_at = _now()
    session = {
        "schema_version": gate.SCHEMA_VERSION,
        "type": "session",
        "arm": arm,
        "sequence_index": gate.ARMS.index(arm),
        "started_at": started_at,
        "finished_at": finished_at,
        "config": gate._session_config(arm),
        "runtime": {
            **gate.REFERENCE_RUNTIME,
            "image_repo_digest": image_repo_digest,
            "image_config_id": image_config_id,
            "container_observation": dict(container_observation),
        },
        "environment_start": environment_start,
        "environment_end": environment_end,
        "checkpoint_start": checkpoint_start,
        "checkpoint_end": checkpoint_end,
        "kernel_ranks": _kernel_rows(
            arm=arm,
            projection_names=projections,
            provider_version=provider_version,
        ),
        "sampling_ownership": [],
        "attention_backend": {
            "requested": "tensorrt_llm",
            "resolved": "tensorrt_llm",
            "components": {
                "prefill": "tensorrt_llm",
                "decode": "tensorrt_llm",
                "kv_mode": "paged-direct",
            },
            "identity": gate.canonical_json(
                {
                    "backend": "TRTLLM",
                    "version": bindings["version"],
                    "overlap": False,
                    "cuda_graph": False,
                }
            ),
        },
        "checkpoint_view": checkpoint_view,
    }
    fragment = {
        "schema_version": CAPTURE_SCHEMA,
        "type": ARM_FRAGMENT_TYPE,
        "arm": arm,
        "prompts": prompt_rows,
        "session": session,
        "ownership": [],
        "free_run": free_rows,
        "teacher": teacher_rows,
        "failures": [],
    }
    if not _fragment_valid(fragment, arm=arm, host_snapshot=host_snapshot):
        raise RuntimeError(f"generated {arm} fragment is invalid")
    return fragment


def _run_kairyu_batch(
    *,
    tokenizer: object,
    runner: object,
    prompts: Sequence[Sequence[int]],
    max_tokens: int,
    num_pages: int,
    page_size: int,
    request_prefix: str,
) -> list[object]:
    from types import SimpleNamespace

    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.engine_loop import EngineLoop
    from kairyu.engine.prompt import TokensPrompt
    from kairyu.sampling_params import SamplingParams

    # A fresh radix tree per wave is the executable no-reuse contract. All 64
    # adds are queued before the first step, and the token budget admits the
    # complete wave together, so no completed request can seed another request.
    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=max(sum(len(prompt) for prompt in prompts), 64),
        max_num_seqs=gate.PROMPT_COUNT,
        page_size=page_size,
    )
    loop = EngineLoop(
        tokenizer,
        scheduler,
        runner,
        pipeline_depth=1,
        max_model_len=max(len(prompt) + max_tokens for prompt in prompts),
    )
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=gate.SEED,
        logprobs=gate.TOP_LOGPROBS,
        ignore_eos=True,
        skip_special_tokens=False,
    )
    for index, prompt in enumerate(prompts):
        loop.submit(
            f"{request_prefix}-p{index}",
            TokensPrompt(tuple(prompt)),
            params,
        )
    finished: dict[str, object] = {}
    steps = 0
    try:
        while loop.has_work():
            steps += 1
            if steps > 100_000:
                raise RuntimeError("Kairyu capture exceeded the bounded step count")
            for request_id, update in loop.step():
                if update.error is not None:
                    raise update.error
                if update.finished:
                    finished[request_id] = update
    finally:
        loop.close()
    outputs: list[object] = []
    for index, prompt in enumerate(prompts):
        request_id = f"{request_prefix}-p{index}"
        update = finished.get(request_id)
        if update is None:
            raise RuntimeError(f"Kairyu produced no terminal update for {request_id}")
        if update.logprobs is None:
            raise RuntimeError(f"Kairyu omitted logprobs for {request_id}")
        completion = SimpleNamespace(
            token_ids=tuple(update.outputs),
            logprobs=tuple(update.logprobs),
            finish_reason=update.finish_reason,
        )
        outputs.append(
            SimpleNamespace(
                outputs=(completion,),
                prompt_token_ids=tuple(prompt),
            )
        )
    return outputs


def _formal_num_pages(
    *, prompts: Sequence[Sequence[int]], max_tokens: int, page_size: int
) -> int:
    if page_size != gate.KAIRYU_KV_PAGE_SIZE:
        raise ValueError(
            f"formal Kairyu capture requires page_size={gate.KAIRYU_KV_PAGE_SIZE}"
        )
    required_pages = sum(
        (len(prompt) + max_tokens + page_size - 1) // page_size
        for prompt in prompts
    )
    if required_pages > gate.KAIRYU_KV_NUM_PAGES:
        raise RuntimeError(
            "fixed Kairyu KV capacity is smaller than the formal workload: "
            f"required={required_pages}, capacity={gate.KAIRYU_KV_NUM_PAGES}"
        )
    return gate.KAIRYU_KV_NUM_PAGES


def _validate_ep_kernel_probe(
    value: object,
    *,
    projection_names: Sequence[Sequence[str]],
    ownership: Sequence[Mapping[str, object]],
) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != len(projection_names):
        raise RuntimeError("EP kernel probe rank count is incomplete")
    for rank, (row, names, owned) in enumerate(
        zip(value, projection_names, ownership, strict=True)
    ):
        if not isinstance(row, dict) or set(row) != {
            "rank",
            "projection_count",
            "projection_inventory_sha256",
            "kernel_counts",
            "fused_moe",
            "attention_output",
            "meta_count",
            "load_info",
        }:
            raise RuntimeError(f"EP kernel probe row {rank} has an invalid schema")
        expected_count = len(names)
        owned_indices = list(owned["owned_expert_indices"])
        routed_matches = [
            _ROUTED_EXPERT.match(f"{name}.")
            for name in names
        ]
        routed_projection_count = sum(
            match is not None for match in routed_matches
        )
        dense_projection_count = expected_count - routed_projection_count
        fused_block_count = len(
            {
                int(match.group(1))
                for match in routed_matches
                if match is not None
            }
        )
        expected_fused_moe = (
            {
                "backend": "flashinfer.cutlass_fused_moe",
                "global_expert_count": gate.NUM_EXPERTS,
                "local_expert_count": len(owned_indices),
                "local_expert_start": owned_indices[0],
                "weight_order": "up_gate",
                "scale_layout": "128x4",
                "output_dtype": "bfloat16",
                "ep_partial": True,
                "enable_alltoall": False,
                "block_count": fused_block_count,
                "successful_forward_block_count": fused_block_count,
            }
            if routed_projection_count
            else None
        )
        world_size = len(projection_names)
        expected_attention_output = {
            "backend": "flashinfer.mm_fp4",
            "placement": "row_parallel",
            "parallel_size": world_size,
            "rank": rank,
            "shard_axis": 1,
            "global_in_features": 8_192,
            "local_in_features": 8_192 // world_size,
            "out_features": 4_096,
            "partial_dtype": "bfloat16",
            "block_count": gate.NUM_LAYERS,
            "successful_forward_block_count": gate.NUM_LAYERS,
        }
        expected_kernel_counts = {
            "nvfp4:flashinfer_nvfp4": dense_projection_count,
        }
        if routed_projection_count:
            expected_kernel_counts[
                "nvfp4:flashinfer_cutlass_fused_moe"
            ] = routed_projection_count
        expected_kernel_counts = dict(sorted(expected_kernel_counts.items()))
        expected_load = {
            "ep_rank": rank,
            "ep_size": len(projection_names),
            "owned_expert_count": len(owned_indices),
            "owned_expert_indices_sha256": gate.sha256_json(owned_indices),
            "first_owned_expert": owned_indices[0],
            "last_owned_expert": owned_indices[-1],
            "quantization_source": "hf_quant_config.json",
            "quantization_method": "nvfp4",
            "kv_cache_quant_algo": "FP8",
            "producer_name": "modelopt",
            "producer_version": "0.33.0",
            "checkpoint_tensor_count": gate.EXPECTED_TENSOR_COUNT,
            "rank_loaded_tensor_count": owned["rank_loaded_tensor_count"],
            "auxiliary_kv_scale_count": 188,
        }
        if (
            row["rank"] != rank
            or row["projection_count"] != expected_count
            or row["projection_inventory_sha256"]
            != gate.sha256_json(list(names))
            or row["kernel_counts"]
            != expected_kernel_counts
            or row["fused_moe"] != expected_fused_moe
            or row["attention_output"] != expected_attention_output
            or row["meta_count"]
            != {"parameters": 0, "buffers": 0, "total": 0}
            or row["load_info"] != expected_load
        ):
            raise RuntimeError(
                f"EP rank {rank} runtime kernel/load probe differs from the "
                "checkpoint-derived contract"
            )


def capture_kairyu_arm(
    *,
    model_path: Path,
    world_size: int,
    host_snapshot: Mapping[str, object],
    reference_fragment: Mapping[str, object],
    page_size: int = 16,
    num_pages: int | None = None,
    launcher_type: object | None = None,
    tokenizer: object | None = None,
    container_observation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if world_size not in gate.WORLD_SIZES:
        raise ValueError("Kairyu arm world_size must be 2 or 4")
    arm = gate.KAIRYU_ARM[world_size]
    reference_arm = gate.REFERENCE_ARM[world_size]
    if not _host_snapshot_valid(host_snapshot):
        raise ValueError("invalid host start snapshot")
    if (
        not gate._container_observation_valid(container_observation)
        or container_observation["image_repo_digest"]
        != host_snapshot["image_repo_digest"]
        or container_observation["image_config_id"]
        != host_snapshot["image_config_id"]
    ):
        raise ValueError("Kairyu arm requires matching docker inspect evidence")
    if not _fragment_valid(
        reference_fragment,
        arm=reference_arm,
        host_snapshot=host_snapshot,
    ) or reference_fragment["failures"]:
        raise ValueError(f"valid successful {reference_arm} fragment is required")
    if page_size != gate.KAIRYU_KV_PAGE_SIZE:
        raise ValueError(
            f"formal Kairyu capture requires page_size={gate.KAIRYU_KV_PAGE_SIZE}"
        )
    if tokenizer is None:
        from kairyu.engine.tokenizer import resolve_tokenizer

        tokenizer = resolve_tokenizer(str(model_path))
    if launcher_type is None:
        from kairyu.engine.core.worker import DistEPLauncher

        launcher_type = DistEPLauncher
    prompt_rows = host_snapshot["prompts"]
    prompt_tokens = [list(row["token_ids"]) for row in prompt_rows]
    canonical = _reference_teacher_canonical(
        reference_fragment=reference_fragment,
        prompt_tokens=prompt_tokens,
    )
    started_at = _now()
    checkpoint_start = _checkpoint_snapshot(model_path)
    if checkpoint_start != host_snapshot["checkpoint"]:
        raise RuntimeError("Kairyu session start checkpoint differs from prepare")
    environment_start = _environment(arm)
    formal_pages = _formal_num_pages(
        prompts=prompt_tokens,
        max_tokens=gate.POSITIONS,
        page_size=page_size,
    )
    if num_pages is None:
        num_pages = formal_pages
    elif num_pages != formal_pages:
        raise ValueError(
            f"formal KV policy requires {formal_pages} pages, got override {num_pages}"
        )
    launcher = None
    try:
        launcher = launcher_type(
            str(model_path),
            world_size,
            num_pages,
            page_size,
            tokenizer.vocab(),
            pipeline_depth=1,
            decode_mode="eager",
            kv_cache_dtype="bfloat16",
        )
        metadata = launcher.parallelism_metadata()
        if metadata != {
            "parallelism": "expert_parallel",
            "expert_parallel_size": world_size,
            "attention_placement": "replicated",
            "attention_output_placement": "row_parallel",
            "attention_output_parallel_size": world_size,
            "attention_output_partial_dtype": "bfloat16",
            "execution_mode": "replicated-attention-correctness",
            "pipeline_depth": 1,
            "decode_mode": "eager",
            "kv_cache_dtype": "bfloat16",
        }:
            raise RuntimeError(f"unexpected DistEPLauncher topology: {metadata!r}")
        probe = launcher.runner.sampling_ownership_metadata()
        expected_sampling_ownership = tuple(
            {
                "rank": rank,
                "control_world_size": world_size,
                "control_backend": "gloo",
                "model_world_size": world_size,
                "model_backend": "nccl",
                "sampling_owner": rank == 0,
                "sampler_present": rank == 0,
                "device": f"cuda:{rank}",
            }
            for rank in range(world_size)
        )
        if probe != expected_sampling_ownership:
            raise RuntimeError(f"rank ownership probe is invalid: {probe!r}")
        sampling_ownership = [dict(row) for row in probe]
        ownership, projections = _ownership_rows(
            model_path=model_path,
            arm=arm,
            load_info=launcher.expert_parallel_load_info,
        )
        free_outputs = _run_kairyu_batch(
            tokenizer=tokenizer,
            runner=launcher.runner,
            prompts=prompt_tokens,
            max_tokens=gate.POSITIONS,
            num_pages=num_pages,
            page_size=page_size,
            request_prefix=f"{arm}-free",
        )
        free_rows, _free_canonical = _free_rows_from_outputs(
            arm=arm, outputs=free_outputs
        )
        teacher_rows: list[dict[str, object]] = []
        for position in range(gate.POSITIONS):
            prefixes = [
                [*prompt_tokens[index], *canonical[index][:position]]
                for index in range(gate.PROMPT_COUNT)
            ]
            outputs = _run_kairyu_batch(
                tokenizer=tokenizer,
                runner=launcher.runner,
                prompts=prefixes,
                max_tokens=1,
                num_pages=num_pages,
                page_size=page_size,
                request_prefix=f"{arm}-teacher-{position}",
            )
            teacher_rows.extend(
                _teacher_rows_from_outputs(
                    arm=arm,
                    position=position,
                    outputs=outputs,
                    prompt_tokens=prompt_tokens,
                    canonical=canonical,
                )
            )
        runtime_kernel_probes = launcher.ep_kernel_inventory_metadata()
        _validate_ep_kernel_probe(
            runtime_kernel_probes,
            projection_names=projections,
            ownership=ownership,
        )
        decision = launcher.attention_backend_decision
        attention = {
            "requested": decision.requested,
            "resolved": decision.resolved,
            "components": dict(decision.components),
            "identity": launcher.attention_backend_identity,
        }
        checkpoint_view = {
            "kind": "original",
            "model_path": str(model_path),
            "hf_quant_config_sha256": host_snapshot["checkpoint"][
                "hf_quant_config_sha256"
            ],
        }
        kernel_rows = _kernel_rows(
            arm=arm,
            projection_names=projections,
            provider_version=_package_version("flashinfer-python"),
            runtime_probes=runtime_kernel_probes,
        )
    finally:
        if launcher is not None:
            launcher.shutdown()
    environment_end = _environment(arm)
    checkpoint_end = _checkpoint_snapshot(model_path)
    if checkpoint_end != checkpoint_start:
        raise RuntimeError("Kairyu checkpoint changed during the engine session")
    finished_at = _now()
    runtime = {
        "name": "kairyu",
        "version": _package_version("kairyu"),
        "image_repo_digest": host_snapshot["image_repo_digest"],
        "image_config_id": host_snapshot["image_config_id"],
        "backend": "pytorch",
        "result_api": gate.KAIRYU_RESULT_API,
        "source_commit": host_snapshot["source"]["commit"],
        "container_observation": dict(container_observation),
    }
    session = {
        "schema_version": gate.SCHEMA_VERSION,
        "type": "session",
        "arm": arm,
        "sequence_index": gate.ARMS.index(arm),
        "started_at": started_at,
        "finished_at": finished_at,
        "config": gate._session_config(arm),
        "runtime": runtime,
        "environment_start": environment_start,
        "environment_end": environment_end,
        "checkpoint_start": checkpoint_start,
        "checkpoint_end": checkpoint_end,
        "kernel_ranks": kernel_rows,
        "sampling_ownership": sampling_ownership,
        "attention_backend": attention,
        "checkpoint_view": checkpoint_view,
    }
    fragment = {
        "schema_version": CAPTURE_SCHEMA,
        "type": ARM_FRAGMENT_TYPE,
        "arm": arm,
        "prompts": prompt_rows,
        "session": session,
        "ownership": ownership,
        "free_run": free_rows,
        "teacher": teacher_rows,
        "failures": [],
    }
    if not _fragment_valid(fragment, arm=arm, host_snapshot=host_snapshot):
        raise RuntimeError(f"generated {arm} fragment is invalid")
    return fragment


def _fragment_valid(
    value: object,
    *,
    arm: str,
    host_snapshot: Mapping[str, object],
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "arm",
        "prompts",
        "session",
        "ownership",
        "free_run",
        "teacher",
        "failures",
    }:
        return False
    if (
        value["schema_version"] != CAPTURE_SCHEMA
        or value["type"] != ARM_FRAGMENT_TYPE
        or value["arm"] != arm
        or value["prompts"] != host_snapshot.get("prompts")
        or not isinstance(value["ownership"], list)
        or not isinstance(value["free_run"], list)
        or not isinstance(value["teacher"], list)
        or not isinstance(value["failures"], list)
        or not all(gate._failure_valid(row) for row in value["failures"])
    ):
        return False
    if value["failures"]:
        return value["session"] is None
    run = {
        "image_repo_digest": host_snapshot.get("image_repo_digest"),
        "image_config_id": host_snapshot.get("image_config_id"),
        "source_start": host_snapshot.get("source"),
        "checkpoint_start": host_snapshot.get("checkpoint"),
    }
    session = value["session"]
    if not gate._session_valid(session, arm=arm, run=run):
        return False
    expected_free = {(arm, f"p{index}") for index in range(gate.PROMPT_COUNT)}
    free_keys = [(row.get("arm"), row.get("prompt_id")) for row in value["free_run"]]
    expected_teacher = {
        (arm, f"p{index}", position)
        for index in range(gate.PROMPT_COUNT)
        for position in range(gate.POSITIONS)
    }
    teacher_keys = [
        (row.get("arm"), row.get("prompt_id"), row.get("position"))
        for row in value["teacher"]
    ]
    ownership_ok = (
        not value["ownership"]
        if gate._arm_stack(arm) == "reference"
        else (
            len(value["ownership"]) == gate._arm_world_size(arm)
            and all(gate._ownership_valid(row) for row in value["ownership"])
        )
    )
    return (
        ownership_ok
        and len(free_keys) == len(set(free_keys)) == gate.PROMPT_COUNT
        and set(free_keys) == expected_free
        and all(gate._free_run_valid(row) for row in value["free_run"])
        and len(teacher_keys) == len(set(teacher_keys)) == gate.EXPECTED_POSITIONS
        and set(teacher_keys) == expected_teacher
        and all(gate._teacher_valid(row) for row in value["teacher"])
    )


def validate_reference_fragment(
    fragment: Mapping[str, object],
    host_snapshot: Mapping[str, object],
) -> dict[str, object]:
    arm = fragment.get("arm")
    if arm not in gate.REFERENCE_ARM.values():
        raise ValueError("reference fragment arm must be reference_ep2 or reference_ep4")
    if not _host_snapshot_valid(host_snapshot):
        raise ValueError("invalid host start snapshot")
    if not _fragment_valid(fragment, arm=arm, host_snapshot=host_snapshot):
        raise ValueError(f"invalid {arm} fragment")
    return {
        "schema_version": CAPTURE_SCHEMA,
        "arm": arm,
        "valid": True,
        "failed": bool(fragment["failures"]),
        "free_run_rows": len(fragment["free_run"]),
        "teacher_rows": len(fragment["teacher"]),
    }


def _row_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["type"]) for row in rows).items()))


def assemble_capture(
    *,
    model_path: Path,
    host_snapshot: Mapping[str, object],
    fragments: Mapping[str, Mapping[str, object]],
    tokenizer: object | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not _host_snapshot_valid(host_snapshot):
        raise ValueError("invalid host start snapshot")
    if set(fragments) != set(gate.ARMS):
        raise ValueError(f"assemble requires exactly {list(gate.ARMS)}")
    for arm in gate.ARMS:
        if not _fragment_valid(
            fragments[arm], arm=arm, host_snapshot=host_snapshot
        ):
            raise ValueError(f"invalid {arm} fragment")

    # This full snapshot is intentionally taken here, after every retained
    # session fragment already has a finish timestamp.
    source_end = _source_snapshot()
    checkpoint_end = _checkpoint_snapshot(model_path)
    prompts_end = _prompt_rows(model_path, checkpoint_end, tokenizer=tokenizer)
    finalized_at = datetime.now(UTC)
    if source_end != host_snapshot["source"]:
        raise ValueError("bound source changed between prepare and assemble")
    if checkpoint_end != host_snapshot["checkpoint"]:
        raise ValueError("checkpoint changed between prepare and assemble")
    if prompts_end != host_snapshot["prompts"]:
        raise ValueError("tokenizer output changed between prepare and assemble")

    prepared_at = gate._utc_timestamp(host_snapshot["prepared_at"])
    sessions = [
        fragments[arm]["session"]
        for arm in gate.ARMS
        if fragments[arm]["session"] is not None
    ]
    if sessions:
        first_started = min(gate._utc_timestamp(row["started_at"]) for row in sessions)
        last_finished = max(gate._utc_timestamp(row["finished_at"]) for row in sessions)
        if prepared_at > first_started:
            raise ValueError("host start snapshot was taken after a session started")
        if finalized_at < last_finished:
            raise ValueError("host end snapshot was taken before a session finished")

    rows: list[dict[str, object]] = [
        {
            "schema_version": gate.SCHEMA_VERSION,
            "type": "run",
            "created_at": host_snapshot["prepared_at"],
            "config": gate.formal_config(),
            "image_repo_digest": host_snapshot["image_repo_digest"],
            "image_config_id": host_snapshot["image_config_id"],
            "source_start": host_snapshot["source"],
            "checkpoint_start": host_snapshot["checkpoint"],
        },
        *host_snapshot["prompts"],
    ]
    rows.extend(
        fragments[arm]["session"]
        for arm in gate.ARMS
        if fragments[arm]["session"] is not None
    )
    for arm in gate.ARMS:
        rows.extend(fragments[arm]["ownership"])
    for arm in gate.ARMS:
        rows.extend(
            sorted(
                fragments[arm]["free_run"],
                key=lambda row: int(row["prompt_id"][1:]),
            )
        )
    for arm in gate.ARMS:
        rows.extend(
            sorted(
                fragments[arm]["teacher"],
                key=lambda row: (
                    int(row["prompt_id"][1:]),
                    row["position"],
                ),
            )
        )
    for arm in gate.ARMS:
        rows.extend(fragments[arm]["failures"])
    run_end = {
        "schema_version": gate.SCHEMA_VERSION,
        "type": "run_end",
        "source_end": source_end,
        "checkpoint_end": checkpoint_end,
        "row_counts_before_end": _row_counts(rows),
    }
    rows.append(run_end)
    raw = "".join(gate.canonical_json(row) + "\n" for row in rows)
    manifest = gate.recompute_manifest(
        rows, raw_sha256=hashlib.sha256(raw.encode()).hexdigest()
    )
    if not any(fragment["failures"] for fragment in fragments.values()):
        if len(rows) != SUCCESS_RAW_ROWS:
            raise ValueError(
                f"successful capture has {len(rows)} rows, expected {SUCCESS_RAW_ROWS}"
            )
    return rows, manifest


def _failure_fragment(
    *, arm: str, host_snapshot: Mapping[str, object], error: BaseException
) -> dict[str, object]:
    fragment = {
        "schema_version": CAPTURE_SCHEMA,
        "type": ARM_FRAGMENT_TYPE,
        "arm": arm,
        "prompts": host_snapshot["prompts"],
        "session": None,
        "ownership": [],
        "free_run": [],
        "teacher": [],
        "failures": [
            {
                "schema_version": gate.SCHEMA_VERSION,
                "type": "failure",
                "arm": arm,
                "error_type": type(error).__name__,
                "message": str(error) or repr(error),
            }
        ],
    }
    if not _fragment_valid(fragment, arm=arm, host_snapshot=host_snapshot):
        raise RuntimeError("could not encode the formal arm failure") from error
    return fragment


def _write_object(path: Path, value: object) -> None:
    atomic_write_text(path, gate.canonical_json(value) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="take the pre-session full source/checkpoint snapshot"
    )
    prepare.add_argument("--model-path", type=Path, required=True)
    prepare.add_argument("--image-repo-digest", required=True)
    prepare.add_argument("--image-config-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)

    reference = subparsers.add_parser(
        "reference-arm", help="run one pinned TensorRT-LLM TP/MoE-EP arm"
    )
    reference.add_argument("--model-path", type=Path, required=True)
    reference.add_argument("--host-snapshot", type=Path, required=True)
    reference.add_argument("--world-size", type=int, choices=gate.WORLD_SIZES, required=True)
    reference.add_argument("--image-repo-digest", required=True)
    reference.add_argument("--image-config-id", required=True)
    reference.add_argument("--container-inspect", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)

    kairyu = subparsers.add_parser(
        "kairyu-arm", help="run one production DistEPLauncher arm"
    )
    kairyu.add_argument("--model-path", type=Path, required=True)
    kairyu.add_argument("--host-snapshot", type=Path, required=True)
    kairyu.add_argument("--reference-fragment", type=Path, required=True)
    kairyu.add_argument("--world-size", type=int, choices=gate.WORLD_SIZES, required=True)
    kairyu.add_argument("--page-size", type=int, default=16)
    kairyu.add_argument("--container-inspect", type=Path, required=True)
    kairyu.add_argument("--output", type=Path, required=True)

    assemble = subparsers.add_parser(
        "assemble", help="take the final full snapshot and emit canonical JSONL"
    )
    assemble.add_argument("--model-path", type=Path, required=True)
    assemble.add_argument("--host-snapshot", type=Path, required=True)
    for arm in gate.ARMS:
        assemble.add_argument(
            f"--{arm.replace('_', '-')}",
            dest=arm,
            type=Path,
            required=True,
        )
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--manifest", type=Path)
    assemble.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_host_snapshot(
            model_path=args.model_path.resolve(),
            image_repo_digest=args.image_repo_digest,
            image_config_id=args.image_config_id,
        )
        _write_object(args.output, result)
        print(gate.canonical_json({"prepared": str(args.output), "valid": True}))
        return 0

    host_snapshot = _read_json(args.host_snapshot)
    if not _host_snapshot_valid(host_snapshot):
        raise ValueError("invalid host start snapshot")
    if args.command in {"reference-arm", "kairyu-arm"}:
        arm = (
            gate.REFERENCE_ARM[args.world_size]
            if args.command == "reference-arm"
            else gate.KAIRYU_ARM[args.world_size]
        )
        try:
            if args.command == "reference-arm":
                observation = _container_observation(
                    args.container_inspect,
                    image_repo_digest=args.image_repo_digest,
                    image_config_id=args.image_config_id,
                )
                result = capture_reference_arm(
                    model_path=args.model_path.resolve(),
                    world_size=args.world_size,
                    host_snapshot=host_snapshot,
                    image_repo_digest=args.image_repo_digest,
                    image_config_id=args.image_config_id,
                    container_observation=observation,
                )
            else:
                reference_fragment = _read_json(args.reference_fragment)
                observation = _container_observation(
                    args.container_inspect,
                    image_repo_digest=host_snapshot["image_repo_digest"],
                    image_config_id=host_snapshot["image_config_id"],
                )
                result = capture_kairyu_arm(
                    model_path=args.model_path.resolve(),
                    world_size=args.world_size,
                    host_snapshot=host_snapshot,
                    reference_fragment=reference_fragment,
                    page_size=args.page_size,
                    container_observation=observation,
                )
        except BaseException as error:
            failure = _failure_fragment(
                arm=arm, host_snapshot=host_snapshot, error=error
            )
            _write_object(args.output, failure)
            print(
                gate.canonical_json(
                    {
                        "arm": arm,
                        "error_type": type(error).__name__,
                        "failure_fragment": str(args.output),
                    }
                ),
                file=sys.stderr,
            )
            return 1
        _write_object(args.output, result)
        print(gate.canonical_json({"arm": arm, "valid": True}))
        return 0

    fragments = {arm: _read_json(getattr(args, arm)) for arm in gate.ARMS}
    rows, manifest = assemble_capture(
        model_path=args.model_path.resolve(),
        host_snapshot=host_snapshot,
        fragments=fragments,
    )
    atomic_write_text(
        args.output,
        "".join(gate.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    if args.manifest is not None:
        atomic_write_json(
            args.manifest,
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    print(gate.canonical_json(manifest))
    return 1 if args.assert_gate and manifest["passed"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
