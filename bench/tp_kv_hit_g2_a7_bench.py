#!/usr/bin/env python3
"""G2 A7: deterministic real-engine KV-hit invariance gate for TP4 and TP8.

The fixed workload geometry comes from :mod:`bench.multiturn_prefix`: 64
sessions, eight turns, a 512-token shared prefix, and 128 appended tokens per
turn in the seed-42 order.  Synthetic token IDs are replaced by tokenizer
round-tripping, one-token text pieces before requests reach the production HTTP
API.

``measure`` records one topology/path shard. ``assemble`` combines exactly the
TP4/TP8 x direct/gateway matrix into canonical raw JSONL plus a wholly-derived
manifest. ``verify`` re-hashes the raw file and recomputes every binding check.

Only engine-reported ``prompt_tokens_details.cached_tokens`` is binding.
Latency, output equality, routing proxies, affinity counts, and OS timing are
intentionally outside this gate.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402
import yaml  # noqa: E402

from bench.multiturn_prefix import (  # noqa: E402
    NUM_SESSIONS,
    SYSTEM_PROMPT_TOKENS,
    TURN_TOKENS,
    TURNS_PER_SESSION,
    WORKLOAD_SEED,
    fixed_workload,
)
from kairyu.engine.tokenizer import HFTokenizer  # noqa: E402

SCHEMA_VERSION = "kairyu.g2.a7.tp-kv-hit.v1"
GATE = "G2-A7"
RAW_NAME = "tp-kv-hit-g2-a7-raw.jsonl"
MANIFEST_NAME = "tp-kv-hit-g2-a7-manifest.json"
MODEL = "qwen3-32b"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
MODEL_WEIGHTS_ROLLUP = (
    "6aa37b7da4e37d45b277d0bca47b00c2bfb58bb60986ba93e4c3e667df123955"
)
TP_DEGREES = (4, 8)
PATHS = ("direct", "gateway")
ENDPOINTS = {
    "direct": "http://127.0.0.1:8001",
    "gateway": "http://127.0.0.1:8002",
}
EXPECTED_SCENARIOS = tuple(
    (tensor_parallel_size, path)
    for tensor_parallel_size in TP_DEGREES
    for path in PATHS
)
EXPECTED_REQUESTS = NUM_SESSIONS * TURNS_PER_SESSION
CACHE_RATE_FLOOR = 0.80
TEMPERATURE = 0.0
MAX_TOKENS = 1
MIN_TOKENS = 1
IGNORE_EOS = True
DEFAULT_TIMEOUT_S = 180.0
_AFFINITY_METRIC_PREFIX = (
    f'kairyu_pool_decisions_total{{pool="{MODEL}",'
    'reason="session_affinity"} '
)

_SOURCE_PATHS = (
    "bench/tp_kv_hit_g2_a7_bench.py",
    "bench/multiturn_prefix.py",
    "bench/results/gate1-hf-parity-tp8-2026-07-26.json",
    "bench/results/parity-tp-qwen3-32b-2026-07-26.json",
    "kairyu/engine/backend.py",
    "kairyu/engine/engine_loop.py",
    "kairyu/engine/kairyu_backend.py",
    "kairyu/engine/openai_backend.py",
    "kairyu/engine/core/radix_kv.py",
    "kairyu/engine/core/scheduler.py",
    "kairyu/entrypoints/server/app.py",
    "kairyu/entrypoints/server/health.py",
    "pyproject.toml",
    "uv.lock",
)
_HEX = frozenset("0123456789abcdef")
_TRACE_CONSTRUCTION = "stable-single-token-repeat-v1"
_CHECKPOINT_ROOT = "/models/qwen3-32b"
_ENGINE_CONFIG_PATH = "/tmp/kairyu.yaml"
_GATEWAY_CONFIG_PATH = (
    "/workspace/bench/deploy/qwen3-32b-multi-gpu/auto-gateway.yaml"
)
_ENGINE_SERVICE = "kairyu"
_GATEWAY_SERVICE = "auto-gateway"
_PARITY_GATE1_PATH = (
    _REPO_ROOT / "bench/results/gate1-hf-parity-tp8-2026-07-26.json"
)
_PARITY_TP_PATH = (
    _REPO_ROOT / "bench/results/parity-tp-qwen3-32b-2026-07-26.json"
)
_COMMON_CONFIGURATION_KEYS = (
    "model",
    "model_revision",
    "model_digests",
    "tokenizer_sha256",
    "config_files",
    "checkpoint_evidence",
)


class _Tokenizer(Protocol):
    def encode(self, text: str) -> tuple[int, ...]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def vocab(self) -> list[str]: ...


@dataclass(frozen=True)
class TextWorkloadPrompt:
    sequence: int
    session: int
    turn: int
    prompt: str
    prompt_token_ids: tuple[int, ...]
    expected_cached_tokens: int

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt)

    @property
    def prompt_token_ids_sha256(self) -> str:
        return sha256_json(list(self.prompt_token_ids))


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


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_descriptor(value: object) -> dict[str, object]:
    return {"value": value, "sha256": sha256_json(value)}


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX
    )


def _valid_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and set(value) <= _HEX
    )


def _descriptor_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"value", "sha256"}
        and _valid_sha256(value["sha256"])
        and value["sha256"] == sha256_json(value["value"])
    )


def _valid_container_id(value: object) -> bool:
    return isinstance(value, str) and _valid_sha256(value)


def _valid_image_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _valid_sha256(value.removeprefix("sha256:"))
    )


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _expected_checkpoint_identity() -> dict[str, object]:
    parity = _read_json_object(_PARITY_TP_PATH)
    gate1 = _read_json_object(_PARITY_GATE1_PATH)
    try:
        checkpoint = parity["config"]["checkpoint"]
        gate1_files = gate1["config"]["reference_provenance"][
            "checkpoint_weight_files"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("pinned parity artifacts omit checkpoint identity") from error
    if not isinstance(checkpoint, dict) or not isinstance(gate1_files, dict):
        raise ValueError("pinned parity checkpoint identity has invalid types")
    weights = checkpoint.get("weights")
    if not isinstance(weights, list) or len(weights) != 17:
        raise ValueError("pinned parity checkpoint must contain 17 weight shards")
    shard_digests = {
        item.get("file"): item.get("sha256")
        for item in weights
        if isinstance(item, dict)
    }
    if (
        len(shard_digests) != 17
        or any(gate1_files.get(name) != digest for name, digest in shard_digests.items())
        or checkpoint.get("weights_sha256") != MODEL_WEIGHTS_ROLLUP
        or hashlib.sha256(
            json.dumps(weights, sort_keys=True).encode()
        ).hexdigest()
        != MODEL_WEIGHTS_ROLLUP
    ):
        raise ValueError("pinned parity artifacts disagree on checkpoint weights")
    index_sha256 = gate1_files.get("model.safetensors.index.json")
    if not _valid_sha256(index_sha256):
        raise ValueError("pinned parity artifact omits the safetensors index")
    expected = {
        "weights_sha256": checkpoint["weights_sha256"],
        "weights": weights,
        "index_sha256": index_sha256,
        "metadata_sha256": checkpoint.get("metadata_sha256"),
        "tokenizer_sha256": checkpoint.get("tokenizer_sha256"),
    }
    if not (
        isinstance(expected["metadata_sha256"], dict)
        and isinstance(expected["tokenizer_sha256"], dict)
        and all(
            _valid_sha256(digest)
            for group in (
                expected["metadata_sha256"],
                expected["tokenizer_sha256"],
            )
            for digest in group.values()
        )
    ):
        raise ValueError("pinned parity metadata/tokenizer identity is invalid")
    return expected


def _checkpoint_references() -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(_REPO_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in (_PARITY_GATE1_PATH, _PARITY_TP_PATH)
    ]


def _mount_descriptor(
    container: Mapping[str, object],
    destination: str,
) -> dict[str, object]:
    mounts = container.get("Mounts")
    if not isinstance(mounts, list):
        raise ValueError("docker inspect omitted Mounts")
    matches = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == destination
    ]
    if len(matches) != 1:
        raise ValueError(
            f"running container requires exactly one {destination} mount"
        )
    mount = matches[0]
    source = _portable_mount_source(mount)
    return {
        "type": mount.get("Type"),
        "name": mount.get("Name"),
        "source": source,
        "destination": mount.get("Destination"),
        "rw": mount.get("RW"),
    }


def _portable_mount_source(mount: Mapping[str, object]) -> str:
    mount_type = mount.get("Type")
    raw_source = mount.get("Source")
    if mount_type == "bind" and isinstance(raw_source, str):
        source = Path(raw_source).resolve()
        try:
            relative = source.relative_to(_REPO_ROOT.resolve())
        except ValueError:
            return f"external:{sha256_text(str(source))}"
        portable = relative.as_posix() if relative.parts else "."
        return f"repo:{portable}"
    name = mount.get("Name")
    if mount_type == "volume" and isinstance(name, str) and name:
        return f"volume:{name}"
    return f"external:{sha256_text(str(raw_source))}"


def _all_mount_descriptors(
    container: Mapping[str, object],
) -> list[dict[str, object]]:
    mounts = container.get("Mounts")
    if not isinstance(mounts, list):
        raise ValueError("docker inspect omitted Mounts")
    return sorted(
        (
            {
                "type": mount.get("Type"),
                "name": mount.get("Name"),
                "source": _portable_mount_source(mount),
                "destination": mount.get("Destination"),
                "rw": mount.get("RW"),
            }
            for mount in mounts
            if isinstance(mount, dict)
        ),
        key=lambda item: str(item["destination"]),
    )


def _gateway_mount_set_valid(
    value: object,
) -> bool:
    if not isinstance(value, list) or not all(
        isinstance(mount, dict)
        and set(mount) == {"type", "name", "source", "destination", "rw"}
        and isinstance(mount["source"], str)
        and isinstance(mount["destination"], str)
        and type(mount["rw"]) is bool
        for mount in value
    ):
        return False
    destinations = [mount["destination"] for mount in value]
    if len(destinations) != len(set(destinations)):
        return False
    for mount in value:
        if mount["rw"] is not True:
            continue
        source = str(mount["source"])
        destination = str(mount["destination"])
        if (
            source.startswith("repo:")
            or destination == "/workspace"
            or destination.startswith("/workspace/")
            or destination == "/app/kairyu"
            or destination.startswith("/app/kairyu/")
        ):
            return False
    return True


def _source_mount_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"type", "name", "source", "destination", "rw"}
        and value["type"] == "bind"
        and value["name"] in {None, ""}
        and value["destination"] == "/app/kairyu"
        and value["rw"] is False
        and value["source"] == "repo:kairyu"
    )


def _model_mount_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"type", "name", "source", "destination", "rw"}
        and value["type"] == "volume"
        and isinstance(value["name"], str)
        and bool(value["name"])
        and isinstance(value["source"], str)
        and value["source"] == f"volume:{value['name']}"
        and value["destination"] == "/models"
        and value["rw"] is False
    )


def _docker_container(container_id: str) -> dict[str, object]:
    completed = subprocess.run(
        ("docker", "inspect", container_id),
        check=True,
        capture_output=True,
        text=True,
    )
    inspected = json.loads(completed.stdout)
    if (
        not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], dict)
    ):
        raise ValueError("docker inspect must resolve exactly one container")
    return inspected[0]


def _container_core(
    container_id: str,
    *,
    expected_service: str,
    include_model_mount: bool,
) -> dict[str, object]:
    container = _docker_container(container_id)
    config = container.get("Config")
    state = container.get("State")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("docker inspect omitted Config or State")
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        raise ValueError("docker inspect omitted container labels")
    service = labels.get("com.docker.compose.service")
    if service != expected_service:
        raise ValueError(
            f"container service is {service!r}, expected {expected_service!r}"
        )
    value = {
        "container_id": container.get("Id"),
        "service": service,
        "config_image": config.get("Image"),
        "image_id": container.get("Image"),
        "source_revision": labels.get("org.opencontainers.image.revision"),
        "working_dir": config.get("WorkingDir"),
        "running": state.get("Running"),
        "source_mount": _mount_descriptor(container, "/app/kairyu"),
        "model_mount": (
            _mount_descriptor(container, "/models")
            if include_model_mount
            else None
        ),
    }
    if not _container_core_valid(
        value,
        expected_service=expected_service,
        include_model_mount=include_model_mount,
    ):
        raise ValueError("running container identity does not satisfy A7")
    return value


def _container_core_valid(
    value: object,
    *,
    expected_service: str,
    include_model_mount: bool,
) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "container_id",
            "service",
            "config_image",
            "image_id",
            "source_revision",
            "working_dir",
            "running",
            "source_mount",
            "model_mount",
        }
        and _valid_container_id(value["container_id"])
        and value["service"] == expected_service
        and isinstance(value["config_image"], str)
        and bool(value["config_image"])
        and _valid_image_id(value["image_id"])
        and _valid_git_commit(value["source_revision"])
        and isinstance(value["working_dir"], str)
        and PurePosixPath(value["working_dir"]).is_absolute()
        and value["running"] is True
        and _source_mount_valid(value["source_mount"])
        and (
            _model_mount_valid(value["model_mount"])
            if include_model_mount
            else value["model_mount"] is None
        )
    )


_CHECKPOINT_HASH_SCRIPT = r"""
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

weight_paths = sorted(root.glob("model-*.safetensors"))
with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(weight_paths))) as pool:
    weight_hashes = list(pool.map(sha256, weight_paths))
weights = [
    {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}
    for path, digest in zip(weight_paths, weight_hashes, strict=True)
]
metadata_names = ("config.json", "generation_config.json")
tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
revision_files = []
for path in sorted((root / ".cache/huggingface/download").glob("*.metadata")):
    lines = path.read_text(encoding="utf-8").splitlines()
    revision_files.append({
        "file": path.relative_to(root).as_posix(),
        "revision": lines[0].strip() if lines else None,
    })
result = {
    "weights": weights,
    "index_sha256": sha256(root / "model.safetensors.index.json"),
    "metadata_sha256": {
        name: sha256(root / name) for name in metadata_names
    },
    "tokenizer_sha256": {
        name: sha256(root / name) for name in tokenizer_names
    },
    "revision_files": revision_files,
}
print(json.dumps(result, sort_keys=True))
"""


def attest_checkpoint(container_id: str) -> dict[str, object]:
    """Full-hash the live model volume once and bind it to parity evidence."""

    container = _container_core(
        container_id,
        expected_service=_ENGINE_SERVICE,
        include_model_mount=True,
    )
    completed = subprocess.run(
        (
            "docker",
            "exec",
            str(container["container_id"]),
            "python",
            "-c",
            _CHECKPOINT_HASH_SCRIPT,
            _CHECKPOINT_ROOT,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    measured = json.loads(completed.stdout)
    if not isinstance(measured, dict):
        raise ValueError("checkpoint hasher returned a non-object")
    weights = measured.get("weights")
    if not isinstance(weights, list):
        raise ValueError("checkpoint hasher omitted weight rows")
    measured["weights_sha256"] = hashlib.sha256(
        json.dumps(weights, sort_keys=True).encode()
    ).hexdigest()
    revision_files = measured.pop("revision_files", None)
    if not isinstance(revision_files, list) or not revision_files:
        raise ValueError("checkpoint has no Hugging Face revision metadata")
    if measured != _expected_checkpoint_identity():
        raise ValueError("live checkpoint bytes differ from pinned parity evidence")
    revisions = {
        item.get("revision")
        for item in revision_files
        if isinstance(item, dict)
    }
    if revisions != {MODEL_REVISION} or len(revision_files) != sum(
        isinstance(item, dict) for item in revision_files
    ):
        raise ValueError("checkpoint revision metadata is absent or inconsistent")
    evidence = hashed_descriptor(
        {
            "schema_version": 1,
            "model": MODEL,
            "checkpoint_root": _CHECKPOINT_ROOT,
            "model_revision": MODEL_REVISION,
            "checkpoint": measured,
            "revision_files": revision_files,
            "container": container,
            "reference_artifacts": _checkpoint_references(),
        }
    )
    if not _checkpoint_evidence_valid(evidence):
        raise ValueError("checkpoint evidence does not satisfy the frozen schema")
    return evidence


def _checkpoint_evidence_valid(value: object) -> bool:
    if not _descriptor_valid(value):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "model",
        "checkpoint_root",
        "model_revision",
        "checkpoint",
        "revision_files",
        "container",
        "reference_artifacts",
    }:
        return False
    revisions = raw["revision_files"]
    return (
        raw["schema_version"] == 1
        and raw["model"] == MODEL
        and raw["checkpoint_root"] == _CHECKPOINT_ROOT
        and raw["model_revision"] == MODEL_REVISION
        and raw["checkpoint"] == _expected_checkpoint_identity()
        and isinstance(revisions, list)
        and bool(revisions)
        and len(
            {
                item.get("file")
                for item in revisions
                if isinstance(item, dict)
            }
        )
        == len(revisions)
        and all(
            isinstance(item, dict)
            and set(item) == {"file", "revision"}
            and isinstance(item["file"], str)
            and item["file"].startswith(
                ".cache/huggingface/download/"
            )
            and item["file"].endswith(".metadata")
            and item["revision"] == MODEL_REVISION
            for item in revisions
        )
        and _container_core_valid(
            raw["container"],
            expected_service=_ENGINE_SERVICE,
            include_model_mount=True,
        )
        and raw["reference_artifacts"] == _checkpoint_references()
    )


def _load_checkpoint_evidence(path: Path) -> dict[str, object]:
    value = _read_json_object(path)
    if not _checkpoint_evidence_valid(value):
        raise ValueError("checkpoint evidence is invalid")
    return value


def _bind_mount_valid(
    value: object,
    *,
    destination: str,
) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"type", "name", "source", "destination", "rw"}
        and value["type"] == "bind"
        and value["name"] in {None, ""}
        and value["destination"] == destination
        and value["rw"] is False
        and isinstance(value["source"], str)
        and value["source"].startswith("repo:")
    )


def _device_request(
    container: Mapping[str, object],
) -> dict[str, object]:
    host_config = container.get("HostConfig")
    if not isinstance(host_config, dict):
        raise ValueError("docker inspect omitted HostConfig")
    requests = host_config.get("DeviceRequests")
    if not isinstance(requests, list):
        raise ValueError("docker inspect omitted DeviceRequests")
    matches = [
        request
        for request in requests
        if isinstance(request, dict) and request.get("Driver") == "nvidia"
    ]
    if len(matches) != 1:
        raise ValueError("engine must have one NVIDIA device request")
    request = matches[0]
    return {
        "driver": request.get("Driver"),
        "count": request.get("Count"),
        "device_ids": request.get("DeviceIDs"),
        "capabilities": request.get("Capabilities"),
    }


def _engine_port_binding(
    container: Mapping[str, object],
) -> dict[str, object]:
    network = container.get("NetworkSettings")
    if not isinstance(network, dict):
        raise ValueError("docker inspect omitted NetworkSettings")
    ports = network.get("Ports")
    if not isinstance(ports, dict):
        raise ValueError("docker inspect omitted published ports")
    bindings = ports.get("8000/tcp")
    if (
        not isinstance(bindings, list)
        or len(bindings) != 1
        or not isinstance(bindings[0], dict)
    ):
        raise ValueError("engine must publish exactly one 8000/tcp binding")
    return {
        "container_port": "8000/tcp",
        "host_ip": bindings[0].get("HostIp"),
        "host_port": bindings[0].get("HostPort"),
    }


def _engine_port_binding_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"container_port", "host_ip", "host_port"}
        and value["container_port"] == "8000/tcp"
        and value["host_ip"] == "127.0.0.1"
        and value["host_port"] == "8001"
    )


def _device_request_valid(value: object, tensor_parallel_size: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "driver",
        "count",
        "device_ids",
        "capabilities",
    }:
        return False
    device_ids = value["device_ids"]
    if not isinstance(device_ids, list) or not all(
        isinstance(item, str) and item.startswith("GPU-")
        for item in device_ids
    ):
        return False
    return (
        value["driver"] == "nvidia"
        and isinstance(value["capabilities"], list)
        and ["gpu"] in value["capabilities"]
        and len(device_ids) == tensor_parallel_size
        and len(set(device_ids)) == tensor_parallel_size
    )


def _docker_exec_text(container_id: str, *command: str) -> str:
    completed = subprocess.run(
        ("docker", "exec", container_id, *command),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _docker_pid1_cmdline(container_id: str) -> list[str]:
    completed = subprocess.run(
        ("docker", "exec", container_id, "cat", "/proc/1/cmdline"),
        check=True,
        capture_output=True,
    )
    try:
        return [
            part.decode()
            for part in completed.stdout.split(b"\0")
            if part
        ]
    except UnicodeDecodeError as error:
        raise ValueError("container PID 1 cmdline is not UTF-8") from error


def _serve_cmdline_valid(
    value: object,
    config_path: str,
    working_dir: object,
) -> bool:
    if not (
        isinstance(value, list)
        and len(value) >= 3
        and all(isinstance(item, str) and bool(item) for item in value)
        and Path(value[-3]).name == "kairyu"
        and value[-2] == "serve"
        and isinstance(working_dir, str)
        and PurePosixPath(working_dir).is_absolute()
    ):
        return False
    declared = PurePosixPath(value[-1])
    resolved = (
        declared
        if declared.is_absolute()
        else PurePosixPath(working_dir) / declared
    )
    return resolved == PurePosixPath(config_path) and ".." not in declared.parts


def _container_gpu_inventory(container_id: str) -> list[str]:
    output = _docker_exec_text(
        container_id,
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader",
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _gpu_inventory_valid(value: object, tensor_parallel_size: int) -> bool:
    if not isinstance(value, list) or len(value) != tensor_parallel_size:
        return False
    parsed = [
        tuple(part.strip() for part in line.split(",", 1))
        for line in value
        if isinstance(line, str) and "," in line
    ]
    return (
        len(parsed) == tensor_parallel_size
        and all(index and uuid.startswith("GPU-") for index, uuid in parsed)
        and len({index for index, _ in parsed}) == tensor_parallel_size
        and len({uuid for _, uuid in parsed}) == tensor_parallel_size
    )


def _engine_config_valid(text: object, tensor_parallel_size: int) -> bool:
    if not isinstance(text, str) or not text:
        return False
    expected = (
        _REPO_ROOT
        / "bench/deploy/qwen3-32b-multi-gpu/kairyu.template.yaml"
    ).read_text(encoding="utf-8").replace(
        "__TENSOR_PARALLEL_SIZE__",
        str(tensor_parallel_size),
    )
    if text != expected:
        return False
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    if not isinstance(value, dict):
        return False
    engines = value.get("engines")
    if not isinstance(engines, dict) or set(engines) != {MODEL}:
        return False
    engine = engines[MODEL]
    if not isinstance(engine, dict) or engine.get("backend") != "kairyu":
        return False
    options = engine.get("options")
    return (
        isinstance(options, dict)
        and options.get("model_path") == _CHECKPOINT_ROOT
        and options.get("tensor_parallel_size") == tensor_parallel_size
    )


def _gateway_config_valid(text: object) -> bool:
    if not isinstance(text, str) or not text:
        return False
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    if not isinstance(value, dict):
        return False
    server = value.get("server")
    if not (
        isinstance(server, dict)
        and server.get("host") == "127.0.0.1"
        and server.get("port") == 8002
    ):
        return False
    pools = value.get("pools")
    if not isinstance(pools, dict) or set(pools) != {MODEL}:
        return False
    pool = pools[MODEL]
    if not isinstance(pool, dict):
        return False
    replicas = pool.get("replicas")
    if (
        not isinstance(replicas, list)
        or len(replicas) != 1
        or not isinstance(replicas[0], dict)
    ):
        return False
    replica = replicas[0]
    options = replica.get("options")
    return (
        replica.get("backend") == "openai"
        and replica.get("health_url") == "http://127.0.0.1:8001/readyz"
        and isinstance(options, dict)
        and options.get("base_url") == "http://127.0.0.1:8001/v1"
        and options.get("model") == MODEL
        and options.get("upstream") == "kairyu"
    )


def _runtime_attestation_valid(
    value: object,
    *,
    tensor_parallel_size: int,
    path: str,
    expected_commit: str,
    checkpoint_evidence: object,
) -> bool:
    if not _descriptor_valid(value) or not _checkpoint_evidence_valid(
        checkpoint_evidence
    ):
        return False
    assert isinstance(value, dict)
    assert isinstance(checkpoint_evidence, dict)
    raw = value["value"]
    evidence = checkpoint_evidence["value"]
    if not isinstance(raw, dict) or not isinstance(evidence, dict):
        return False
    if set(raw) != {
        "schema_version",
        "tensor_parallel_size",
        "path",
        "checkpoint_evidence_sha256",
        "engine",
        "gateway",
    }:
        return False
    engine = raw["engine"]
    if not isinstance(engine, dict) or set(engine) != {
        "container",
        "template_mount",
        "port_binding",
        "device_request",
        "gpu_inventory",
        "pid1_cmdline",
        "effective_config",
    }:
        return False
    engine_container = engine["container"]
    effective = engine["effective_config"]
    if not (
        raw["schema_version"] == 1
        and raw["tensor_parallel_size"] == tensor_parallel_size
        and raw["path"] == path
        and raw["checkpoint_evidence_sha256"] == checkpoint_evidence["sha256"]
        and _container_core_valid(
            engine_container,
            expected_service=_ENGINE_SERVICE,
            include_model_mount=True,
        )
        and isinstance(engine_container, dict)
        and engine_container["source_revision"] == expected_commit
        and _bind_mount_valid(
            engine["template_mount"],
            destination="/etc/kairyu/config.template.yaml",
        )
        and engine["template_mount"]["source"]
        == "repo:bench/deploy/qwen3-32b-multi-gpu/kairyu.template.yaml"
        and _engine_port_binding_valid(engine["port_binding"])
        and _device_request_valid(
            engine["device_request"],
            tensor_parallel_size,
        )
        and _gpu_inventory_valid(
            engine["gpu_inventory"],
            tensor_parallel_size,
        )
        and _serve_cmdline_valid(
            engine["pid1_cmdline"],
            _ENGINE_CONFIG_PATH,
            engine_container["working_dir"],
        )
        and _descriptor_valid(effective)
        and isinstance(effective, dict)
        and _engine_config_valid(effective["value"], tensor_parallel_size)
    ):
        return False
    evidence_container = evidence["container"]
    if not isinstance(evidence_container, dict):
        return False
    for key in ("image_id", "source_revision", "source_mount", "model_mount"):
        if engine_container[key] != evidence_container.get(key):
            return False
    gateway = raw["gateway"]
    if path == "direct":
        return gateway is None
    if path != "gateway" or not isinstance(gateway, dict) or set(gateway) != {
        "container",
        "workspace_mount",
        "all_mounts",
        "network_mode",
        "pid1_cmdline",
        "effective_config",
    }:
        return False
    gateway_container = gateway["container"]
    gateway_config = gateway["effective_config"]
    return (
        _container_core_valid(
            gateway_container,
            expected_service=_GATEWAY_SERVICE,
            include_model_mount=False,
        )
        and isinstance(gateway_container, dict)
        and gateway_container["source_revision"] == expected_commit
        and gateway_container["image_id"] == engine_container["image_id"]
        and gateway_container["source_mount"] == engine_container["source_mount"]
        and _bind_mount_valid(
            gateway["workspace_mount"],
            destination="/workspace",
        )
        and gateway["workspace_mount"]["source"] == "repo:."
        and _gateway_mount_set_valid(gateway["all_mounts"])
        and gateway["network_mode"] == "host"
        and _serve_cmdline_valid(
            gateway["pid1_cmdline"],
            _GATEWAY_CONFIG_PATH,
            gateway_container["working_dir"],
        )
        and _descriptor_valid(gateway_config)
        and isinstance(gateway_config, dict)
        and _gateway_config_valid(gateway_config["value"])
        and gateway_config["value"]
        == (
            _REPO_ROOT
            / "bench/deploy/qwen3-32b-multi-gpu/auto-gateway.yaml"
        ).read_text(encoding="utf-8")
    )


def capture_runtime_attestation(
    *,
    tensor_parallel_size: int,
    path: str,
    runtime_container: str,
    gateway_container: str | None,
    expected_commit: str,
    checkpoint_evidence: dict[str, object],
) -> dict[str, object]:
    if path == "direct" and gateway_container is not None:
        raise ValueError("direct A7 measurement must not declare a gateway")
    if path == "gateway" and gateway_container is None:
        raise ValueError("gateway A7 measurement requires --gateway-container")
    engine_raw = _docker_container(runtime_container)
    engine_core = _container_core(
        runtime_container,
        expected_service=_ENGINE_SERVICE,
        include_model_mount=True,
    )
    template_mount = _mount_descriptor(
        engine_raw,
        "/etc/kairyu/config.template.yaml",
    )
    if (
        template_mount["source"]
        != "repo:bench/deploy/qwen3-32b-multi-gpu/kairyu.template.yaml"
    ):
        raise ValueError("engine template mount is not this checkout")
    engine = {
        "container": engine_core,
        "template_mount": template_mount,
        "port_binding": _engine_port_binding(engine_raw),
        "device_request": _device_request(engine_raw),
        "gpu_inventory": _container_gpu_inventory(runtime_container),
        "pid1_cmdline": _docker_pid1_cmdline(runtime_container),
        "effective_config": hashed_descriptor(
            _docker_exec_text(runtime_container, "cat", _ENGINE_CONFIG_PATH)
        ),
    }
    gateway: dict[str, object] | None = None
    if gateway_container is not None:
        gateway_raw = _docker_container(gateway_container)
        host_config = gateway_raw.get("HostConfig")
        if not isinstance(host_config, dict):
            raise ValueError("gateway docker inspect omitted HostConfig")
        workspace_mount = _mount_descriptor(gateway_raw, "/workspace")
        if workspace_mount["source"] != "repo:.":
            raise ValueError("gateway workspace mount is not this checkout")
        gateway = {
            "container": _container_core(
                gateway_container,
                expected_service=_GATEWAY_SERVICE,
                include_model_mount=False,
            ),
            "workspace_mount": workspace_mount,
            "all_mounts": _all_mount_descriptors(gateway_raw),
            "network_mode": host_config.get("NetworkMode"),
            "pid1_cmdline": _docker_pid1_cmdline(gateway_container),
            "effective_config": hashed_descriptor(
                _docker_exec_text(
                    gateway_container,
                    "cat",
                    _GATEWAY_CONFIG_PATH,
                )
            ),
        }
    result = hashed_descriptor(
        {
            "schema_version": 1,
            "tensor_parallel_size": tensor_parallel_size,
            "path": path,
            "checkpoint_evidence_sha256": checkpoint_evidence["sha256"],
            "engine": engine,
            "gateway": gateway,
        }
    )
    if not _runtime_attestation_valid(
        result,
        tensor_parallel_size=tensor_parallel_size,
        path=path,
        expected_commit=expected_commit,
        checkpoint_evidence=checkpoint_evidence,
    ):
        raise ValueError("runtime container attestation does not satisfy A7")
    return result


def _scenario_id(tensor_parallel_size: int, path: str) -> str:
    return f"tp{tensor_parallel_size}-{path}"


def _fixed_geometry() -> tuple[dict[str, int], ...]:
    rows = fixed_workload()
    if len(rows) != EXPECTED_REQUESTS:
        raise RuntimeError(
            f"fixed A7 workload has {len(rows)} rows, expected {EXPECTED_REQUESTS}"
        )
    return tuple(
        {
            "sequence": row.sequence,
            "session": row.session,
            "turn": row.turn,
            "prompt_tokens": row.prompt_tokens,
            "expected_cached_tokens": row.expected_cached_tokens,
        }
        for row in rows
    )


def workload_config() -> dict[str, object]:
    geometry = _fixed_geometry()
    return {
        "model": MODEL,
        "seed": WORKLOAD_SEED,
        "sessions": NUM_SESSIONS,
        "turns_per_session": TURNS_PER_SESSION,
        "shared_prefix_tokens": SYSTEM_PROMPT_TOKENS,
        "turn_tokens": TURN_TOKENS,
        "request_count": EXPECTED_REQUESTS,
        "theoretical_prompt_tokens": sum(
            row["prompt_tokens"] for row in geometry
        ),
        "theoretical_cached_tokens": sum(
            row["expected_cached_tokens"] for row in geometry
        ),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "min_tokens": MIN_TOKENS,
        "ignore_eos": IGNORE_EOS,
        "cache_rate_threshold": {
            "operator": ">",
            "value": CACHE_RATE_FLOOR,
        },
    }


def _stable_token_pieces(
    tokenizer: _Tokenizer,
    *,
    required: int = NUM_SESSIONS + len(PATHS),
) -> tuple[tuple[int, str], ...]:
    """Find distinct text pieces that each remain one token when repeated."""

    pieces: list[tuple[int, str]] = []
    seen_text: set[str] = set()
    for token_id in range(len(tokenizer.vocab())):
        text = tokenizer.decode((token_id,))
        if (
            not text
            or text in seen_text
            or not text.startswith(" ")
            or not text.strip()
            or not text.isprintable()
            or "\ufffd" in text
        ):
            continue
        if tokenizer.encode(text) != (token_id,):
            continue
        if tokenizer.encode(text * 4) != (token_id,) * 4:
            continue
        seen_text.add(text)
        pieces.append((token_id, text))
        if len(pieces) == required:
            return tuple(pieces)
    raise ValueError(
        f"tokenizer exposes {len(pieces)} stable one-token text pieces; "
        f"A7 requires {required}"
    )


def build_text_workload(
    tokenizer: _Tokenizer,
    *,
    path: str,
) -> tuple[TextWorkloadPrompt, ...]:
    """Replace synthetic IDs with exact-length, tokenizer-valid text."""

    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}, got {path!r}")
    pieces = _stable_token_pieces(tokenizer)
    shared_token_id, shared_piece = pieces[PATHS.index(path)]
    session_pieces = pieces[len(PATHS) :]
    geometry = _fixed_geometry()
    prompts: list[TextWorkloadPrompt] = []
    for row in geometry:
        session_token_id, session_piece = session_pieces[row["session"]]
        history_tokens = (row["turn"] + 1) * TURN_TOKENS
        prompt = (
            shared_piece * SYSTEM_PROMPT_TOKENS
            + session_piece * history_tokens
        )
        token_ids = (
            (shared_token_id,) * SYSTEM_PROMPT_TOKENS
            + (session_token_id,) * history_tokens
        )
        encoded = tokenizer.encode(prompt)
        if encoded != token_ids:
            raise ValueError(
                "tokenizer text construction lost one-token repeat identity "
                f"at sequence {row['sequence']}"
            )
        if len(encoded) != row["prompt_tokens"]:
            raise ValueError(
                f"tokenizer prompt length {len(encoded)} differs from fixed "
                f"geometry {row['prompt_tokens']} at sequence {row['sequence']}"
            )
        prompts.append(
            TextWorkloadPrompt(
                sequence=row["sequence"],
                session=row["session"],
                turn=row["turn"],
                prompt=prompt,
                prompt_token_ids=encoded,
                expected_cached_tokens=row["expected_cached_tokens"],
            )
        )
    return tuple(prompts)


def trace_descriptor(
    prompts: Sequence[TextWorkloadPrompt],
    *,
    path: str,
    tokenizer_sha256: str,
) -> dict[str, object]:
    if len(prompts) != EXPECTED_REQUESTS:
        raise ValueError(
            f"trace has {len(prompts)} prompts, expected {EXPECTED_REQUESTS}"
        )
    first = prompts[0].prompt_token_ids
    shared_token_id = first[0]
    session_token_ids: list[int | None] = [None] * NUM_SESSIONS
    rows = []
    for prompt in prompts:
        session_token_id = prompt.prompt_token_ids[SYSTEM_PROMPT_TOKENS]
        current = session_token_ids[prompt.session]
        if current is None:
            session_token_ids[prompt.session] = session_token_id
        elif current != session_token_id:
            raise ValueError("one session uses multiple trace token identities")
        rows.append(
            {
                "sequence": prompt.sequence,
                "session": prompt.session,
                "turn": prompt.turn,
                "prompt_tokens": len(prompt.prompt_token_ids),
                "expected_cached_tokens": prompt.expected_cached_tokens,
                "prompt_sha256": prompt.prompt_sha256,
                "prompt_token_ids_sha256": prompt.prompt_token_ids_sha256,
            }
        )
    if any(token_id is None for token_id in session_token_ids):
        raise ValueError("trace omits at least one session")
    value = {
        "construction": _TRACE_CONSTRUCTION,
        "path": path,
        "tokenizer_sha256": tokenizer_sha256,
        "shared_token_id": shared_token_id,
        "session_token_ids": session_token_ids,
        "rows": rows,
    }
    return hashed_descriptor(value)


def _trace_valid(value: object, *, path: str) -> bool:
    if not _descriptor_valid(value):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    if not isinstance(raw, dict) or set(raw) != {
        "construction",
        "path",
        "tokenizer_sha256",
        "shared_token_id",
        "session_token_ids",
        "rows",
    }:
        return False
    session_ids = raw["session_token_ids"]
    rows = raw["rows"]
    if not (
        raw["construction"] == _TRACE_CONSTRUCTION
        and raw["path"] == path
        and _valid_sha256(raw["tokenizer_sha256"])
        and type(raw["shared_token_id"]) is int
        and isinstance(session_ids, list)
        and len(session_ids) == NUM_SESSIONS
        and all(type(token_id) is int for token_id in session_ids)
        and len(set(session_ids)) == NUM_SESSIONS
        and raw["shared_token_id"] not in session_ids
        and isinstance(rows, list)
        and len(rows) == EXPECTED_REQUESTS
    ):
        return False
    geometry = _fixed_geometry()
    for expected, row in zip(geometry, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "sequence",
            "session",
            "turn",
            "prompt_tokens",
            "expected_cached_tokens",
            "prompt_sha256",
            "prompt_token_ids_sha256",
        }:
            return False
        immutable = {
            key: row.get(key)
            for key in (
                "sequence",
                "session",
                "turn",
                "prompt_tokens",
                "expected_cached_tokens",
            )
        }
        if immutable != expected or not _valid_sha256(row["prompt_sha256"]):
            return False
        token_ids = (
            [raw["shared_token_id"]] * SYSTEM_PROMPT_TOKENS
            + [session_ids[expected["session"]]]
            * ((expected["turn"] + 1) * TURN_TOKENS)
        )
        if row["prompt_token_ids_sha256"] != sha256_json(token_ids):
            return False
    return True


def backends_topology_exact(
    backends: object,
    *,
    tensor_parallel_size: int,
    path: str,
) -> bool:
    """Bind the public /backends answer to the requested production path."""

    if not isinstance(backends, dict):
        return False
    engines = backends.get("engines")
    if not isinstance(engines, list):
        return False
    matches = [
        engine
        for engine in engines
        if isinstance(engine, dict) and engine.get("model") == MODEL
    ]
    if len(matches) != 1:
        return False
    engine = matches[0]
    if (
        type(engine.get("tensor_parallel_size")) is not int
        or engine["tensor_parallel_size"] != tensor_parallel_size
    ):
        return False
    if path == "direct":
        return (
            backends.get("role") == "engine-host"
            and engine.get("engine_backend") == "kairyu"
        )
    if path != "gateway":
        return False
    via = engine.get("via_replica")
    return (
        backends.get("role") == "gateway"
        and engine.get("engine_backend") == "replica-pool"
        and isinstance(via, dict)
        and via.get("tensor_parallel_size") == tensor_parallel_size
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


def _run_output(command: tuple[str, ...]) -> list[str]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().splitlines()


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(_REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"configuration file is outside the repository: {resolved}"
        ) from error
    return relative.as_posix()


def _source_snapshot(expected_commit: str) -> dict[str, object]:
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
        "expected_commit_matches": commit == expected_commit,
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


def _configuration_snapshot(
    *,
    tensor_parallel_size: int,
    path: str,
    endpoint: str,
    tokenizer_sha256: str,
    config_paths: Sequence[Path],
    checkpoint_evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "scenario": {
            "tensor_parallel_size": tensor_parallel_size,
            "path": path,
            "endpoint": endpoint,
        },
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_digests": {
            "weights-rollup": MODEL_WEIGHTS_ROLLUP,
        },
        "tokenizer_sha256": tokenizer_sha256,
        "config_files": [
            {
                "path": _repo_relative_path(config_path),
                "sha256": sha256_file(config_path.resolve()),
            }
            for config_path in config_paths
        ],
        "checkpoint_evidence": checkpoint_evidence,
    }


def _environment_snapshot() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_gpus": _run_output(
            (
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,pci.bus_id,uuid",
                "--format=csv,noheader",
            )
        ),
        "nvidia_smi_topology": _run_output(("nvidia-smi", "topo", "-m")),
    }


def capture_provenance(
    *,
    tensor_parallel_size: int,
    path: str,
    endpoint: str,
    tokenizer_sha256: str,
    config_paths: Sequence[Path],
    expected_commit: str,
    runtime_container: str,
    gateway_container: str | None,
    checkpoint_evidence_path: Path,
) -> dict[str, object]:
    checkpoint_evidence = _load_checkpoint_evidence(
        checkpoint_evidence_path
    )
    return {
        "source": hashed_descriptor(_source_snapshot(expected_commit)),
        "configuration": hashed_descriptor(
            _configuration_snapshot(
                tensor_parallel_size=tensor_parallel_size,
                path=path,
                endpoint=endpoint,
                tokenizer_sha256=tokenizer_sha256,
                config_paths=config_paths,
                checkpoint_evidence=checkpoint_evidence,
            )
        ),
        "environment": hashed_descriptor(_environment_snapshot()),
        "runtime": capture_runtime_attestation(
            tensor_parallel_size=tensor_parallel_size,
            path=path,
            runtime_container=runtime_container,
            gateway_container=gateway_container,
            expected_commit=expected_commit,
            checkpoint_evidence=checkpoint_evidence,
        ),
    }


def _source_value_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "commit",
        "expected_commit",
        "expected_commit_matches",
        "git_clean",
        "git_status",
        "files",
    }:
        return False
    files = value["files"]
    return (
        _valid_git_commit(value["commit"])
        and value["expected_commit"] == value["commit"]
        and value["expected_commit_matches"] is True
        and value["git_clean"] is True
        and value["git_status"] == []
        and isinstance(files, list)
        and [item.get("path") for item in files if isinstance(item, dict)]
        == list(_SOURCE_PATHS)
        and all(
            isinstance(item, dict)
            and set(item) == {"path", "sha256"}
            and _valid_sha256(item["sha256"])
            for item in files
        )
    )


def _configuration_value_valid(
    value: object,
    *,
    tensor_parallel_size: int,
    path: str,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "scenario",
        "model",
        "model_revision",
        "model_digests",
        "tokenizer_sha256",
        "config_files",
        "checkpoint_evidence",
    }:
        return False
    scenario = value["scenario"]
    digests = value["model_digests"]
    configs = value["config_files"]
    evidence = value["checkpoint_evidence"]
    tokenizer_sha256 = None
    if _checkpoint_evidence_valid(evidence):
        assert isinstance(evidence, dict)
        evidence_value = evidence["value"]
        assert isinstance(evidence_value, dict)
        checkpoint = evidence_value["checkpoint"]
        assert isinstance(checkpoint, dict)
        tokenizers = checkpoint["tokenizer_sha256"]
        if isinstance(tokenizers, dict):
            tokenizer_sha256 = tokenizers.get("tokenizer.json")
    return (
        isinstance(scenario, dict)
        and set(scenario) == {"tensor_parallel_size", "path", "endpoint"}
        and scenario["tensor_parallel_size"] == tensor_parallel_size
        and scenario["path"] == path
        and scenario["endpoint"] == ENDPOINTS[path]
        and value["model"] == MODEL
        and value["model_revision"] == MODEL_REVISION
        and digests == {"weights-rollup": MODEL_WEIGHTS_ROLLUP}
        and _valid_sha256(value["tokenizer_sha256"])
        and value["tokenizer_sha256"] == tokenizer_sha256
        and isinstance(configs, list)
        and bool(configs)
        and all(
            isinstance(item, dict)
            and set(item) == {"path", "sha256"}
            and isinstance(item["path"], str)
            and bool(item["path"])
            and _valid_sha256(item["sha256"])
            for item in configs
        )
        and _checkpoint_evidence_valid(evidence)
    )


def _environment_value_valid(value: object) -> bool:
    host_gpus = (
        _host_gpu_inventory(value.get("nvidia_smi_gpus"))
        if isinstance(value, dict)
        else None
    )
    topology = value.get("nvidia_smi_topology") if isinstance(value, dict) else None
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "platform",
            "python",
            "python_executable",
            "cuda_visible_devices",
            "nvidia_smi_gpus",
            "nvidia_smi_topology",
        }
        and isinstance(value["platform"], str)
        and bool(value["platform"])
        and isinstance(value["python"], str)
        and bool(value["python"])
        and isinstance(value["python_executable"], str)
        and bool(value["python_executable"])
        and (
            value["cuda_visible_devices"] is None
            or isinstance(value["cuda_visible_devices"], str)
        )
        and host_gpus is not None
        and len(host_gpus) == 8
        and isinstance(topology, list)
        and all(isinstance(line, str) for line in topology)
        and all(
            any(line.lstrip().startswith(f"GPU{index}") for line in topology)
            for index in range(8)
        )
    )


def _host_gpu_inventory(value: object) -> dict[str, str] | None:
    if not isinstance(value, list):
        return None
    result: dict[str, str] = {}
    for line in value:
        if not isinstance(line, str):
            return None
        parts = [part.strip() for part in line.split(",", 4)]
        if (
            len(parts) != 5
            or not parts[0]
            or not parts[4].startswith("GPU-")
            or parts[0] in result
            or parts[4] in result.values()
        ):
            return None
        result[parts[0]] = parts[4]
    return result


def _container_gpu_uuids(value: object) -> set[str] | None:
    if not isinstance(value, list):
        return None
    uuids: set[str] = set()
    for line in value:
        if not isinstance(line, str) or "," not in line:
            return None
        _, uuid = (part.strip() for part in line.split(",", 1))
        if not uuid.startswith("GPU-") or uuid in uuids:
            return None
        uuids.add(uuid)
    return uuids


def _physical_gpu_assignment_valid(
    provenance: Mapping[str, object],
    tensor_parallel_size: int,
) -> bool:
    environment = provenance["environment"]["value"]
    runtime = provenance["runtime"]["value"]
    if not isinstance(environment, dict) or not isinstance(runtime, dict):
        return False
    engine = runtime.get("engine")
    if not isinstance(engine, dict):
        return False
    host = _host_gpu_inventory(environment.get("nvidia_smi_gpus"))
    container = _container_gpu_uuids(engine.get("gpu_inventory"))
    device_request = engine.get("device_request")
    if (
        host is None
        or len(host) != 8
        or container is None
        or len(container) != tensor_parallel_size
        or not container <= set(host.values())
        or not isinstance(device_request, dict)
    ):
        return False
    if tensor_parallel_size == 8 and container != set(host.values()):
        return False
    device_ids = device_request.get("device_ids")
    return isinstance(device_ids, list) and set(device_ids) == container


def _provenance_valid(
    value: object,
    *,
    tensor_parallel_size: int,
    path: str,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "source",
        "configuration",
        "environment",
        "runtime",
    }:
        return False
    if not all(_descriptor_valid(value[name]) for name in value):
        return False
    source = value["source"]["value"]
    configuration = value["configuration"]["value"]
    if not (
        _source_value_valid(source)
        and _configuration_value_valid(
            configuration,
            tensor_parallel_size=tensor_parallel_size,
            path=path,
        )
        and _environment_value_valid(value["environment"]["value"])
    ):
        return False
    assert isinstance(source, dict)
    assert isinstance(configuration, dict)
    return (
        _runtime_attestation_valid(
            value["runtime"],
            tensor_parallel_size=tensor_parallel_size,
            path=path,
            expected_commit=str(source["commit"]),
            checkpoint_evidence=configuration["checkpoint_evidence"],
        )
        and _physical_gpu_assignment_valid(value, tensor_parallel_size)
    )


def _common_configuration(value: dict[str, object]) -> dict[str, object]:
    return {key: value[key] for key in _COMMON_CONFIGURATION_KEYS}


def _provenance_matrix_exact(
    starts: Mapping[tuple[int, str], object],
    ends: Mapping[tuple[int, str], object],
) -> bool:
    if set(starts) != set(EXPECTED_SCENARIOS) or set(ends) != set(starts):
        return False
    for scenario in EXPECTED_SCENARIOS:
        start = starts[scenario]
        if (
            start != ends[scenario]
            or not _provenance_valid(
                start,
                tensor_parallel_size=scenario[0],
                path=scenario[1],
            )
        ):
            return False
    values = [starts[scenario] for scenario in EXPECTED_SCENARIOS]
    if not all(isinstance(item, dict) for item in values):
        return False
    values = [item for item in values if isinstance(item, dict)]
    source = values[0]["source"]
    environment = values[0]["environment"]
    configuration = values[0]["configuration"]["value"]
    assert isinstance(configuration, dict)
    common_configuration = _common_configuration(configuration)
    for value in values[1:]:
        candidate_configuration = value["configuration"]["value"]
        if not isinstance(candidate_configuration, dict):
            return False
        if (
            value["source"] != source
            or value["environment"] != environment
            or _common_configuration(candidate_configuration)
            != common_configuration
        ):
            return False
    for tensor_parallel_size in TP_DEGREES:
        direct_runtime = values[
            EXPECTED_SCENARIOS.index((tensor_parallel_size, "direct"))
        ]["runtime"]["value"]
        gateway_runtime = values[
            EXPECTED_SCENARIOS.index((tensor_parallel_size, "gateway"))
        ]["runtime"]["value"]
        if not isinstance(direct_runtime, dict) or not isinstance(
            gateway_runtime,
            dict,
        ):
            return False
        if direct_runtime.get("engine") != gateway_runtime.get("engine"):
            return False
    return True


def _tokenizer_file(path: Path) -> Path:
    candidate = path / "tokenizer.json" if path.is_dir() else path
    if not candidate.is_file():
        raise ValueError(f"no tokenizer.json at {path}")
    return candidate


def _normalize_endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[: -len("/v1")]
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("endpoint must be an HTTP(S) origin")
    return endpoint


def _formal_preflight(
    *,
    tensor_parallel_size: int,
    path: str,
    endpoint: str,
    tokenizer_sha256: str,
    config_paths: Sequence[Path],
    expected_commit: str,
    runtime_container: str,
    gateway_container: str | None,
    checkpoint_evidence_path: Path,
) -> None:
    if tensor_parallel_size not in TP_DEGREES:
        raise ValueError(f"A7 tensor parallel size must be one of {TP_DEGREES}")
    if path not in PATHS:
        raise ValueError(f"A7 path must be one of {PATHS}")
    if endpoint != ENDPOINTS[path]:
        raise ValueError(
            f"A7 {path} endpoint must be exactly {ENDPOINTS[path]}"
        )
    if not config_paths or any(not item.is_file() for item in config_paths):
        raise ValueError("A7 requires every declared config file to exist")
    if not _valid_git_commit(expected_commit):
        raise ValueError("A7 requires --expected-commit as a full SHA")
    source = _source_snapshot(expected_commit)
    if source["expected_commit_matches"] is not True:
        raise ValueError("A7 expected commit does not equal HEAD")
    if source["git_clean"] is not True:
        raise ValueError("A7 requires a clean source tree")
    evidence = _load_checkpoint_evidence(checkpoint_evidence_path)
    evidence_value = evidence["value"]
    assert isinstance(evidence_value, dict)
    checkpoint = evidence_value["checkpoint"]
    assert isinstance(checkpoint, dict)
    tokenizer_digests = checkpoint["tokenizer_sha256"]
    if (
        not isinstance(tokenizer_digests, dict)
        or tokenizer_digests.get("tokenizer.json") != tokenizer_sha256
    ):
        raise ValueError("measurement tokenizer differs from checkpoint evidence")
    capture_runtime_attestation(
        tensor_parallel_size=tensor_parallel_size,
        path=path,
        runtime_container=runtime_container,
        gateway_container=gateway_container,
        expected_commit=expected_commit,
        checkpoint_evidence=evidence,
    )


def _usage_from_response(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    details = value.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        return None
    usage = {
        "prompt_tokens": value.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": value.get("completion_tokens"),
    }
    if not all(
        type(item) is int and item >= 0 for item in usage.values()
    ):
        return None
    if usage["cached_tokens"] > usage["prompt_tokens"]:
        return None
    return usage  # type: ignore[return-value]


def _usage_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {"prompt_tokens", "cached_tokens", "completion_tokens"}
        and all(type(item) is int and item >= 0 for item in value.values())
        and value["prompt_tokens"] > 0
        and value["cached_tokens"] <= value["prompt_tokens"]
    )


def _affinity_metric_lines(text: str) -> tuple[list[str], float | None]:
    lines = [
        line
        for line in text.splitlines()
        if line.startswith(_AFFINITY_METRIC_PREFIX)
    ]
    if len(lines) != 1:
        return lines, None
    sample_text = lines[0][len(_AFFINITY_METRIC_PREFIX) :].split(maxsplit=1)[0]
    try:
        sample = float(sample_text)
    except ValueError:
        return lines, None
    return lines, sample if math.isfinite(sample) else None


async def _capture_affinity_metric(
    client: httpx.AsyncClient,
    endpoint: str,
) -> dict[str, object]:
    """Retain a non-binding gateway affinity diagnostic."""

    try:
        response = await client.get(f"{endpoint}/metrics")
        lines, value = _affinity_metric_lines(response.text)
        return {
            "http_status": response.status_code,
            "relevant_lines": lines,
            "value": value,
            "error_type": None if response.status_code == 200 else "HTTPError",
        }
    except Exception as error:
        return {
            "http_status": None,
            "relevant_lines": [],
            "value": None,
            "error_type": type(error).__name__,
        }


def _affinity_metric_delta(
    before: object,
    after: object,
) -> float | None:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    before_value = before.get("value")
    after_value = after.get("value")
    if (
        isinstance(before_value, (int, float))
        and not isinstance(before_value, bool)
        and math.isfinite(before_value)
        and isinstance(after_value, (int, float))
        and not isinstance(after_value, bool)
        and math.isfinite(after_value)
    ):
        return float(after_value) - float(before_value)
    return None


async def measure(
    *,
    tensor_parallel_size: int,
    path: str,
    endpoint: str,
    tokenizer_path: Path,
    config_paths: Sequence[Path],
    expected_commit: str,
    runtime_container: str,
    gateway_container: str | None,
    checkpoint_evidence_path: Path,
    request_timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[dict[str, object]]:
    """Measure one TP/path shard through the production HTTP API."""

    if request_timeout_s <= 0:
        raise ValueError("request timeout must be positive")
    endpoint = _normalize_endpoint(endpoint)
    tokenizer_file = _tokenizer_file(tokenizer_path)
    tokenizer_sha256 = sha256_file(tokenizer_file)
    _formal_preflight(
        tensor_parallel_size=tensor_parallel_size,
        path=path,
        endpoint=endpoint,
        tokenizer_sha256=tokenizer_sha256,
        config_paths=config_paths,
        expected_commit=expected_commit,
        runtime_container=runtime_container,
        gateway_container=gateway_container,
        checkpoint_evidence_path=checkpoint_evidence_path,
    )
    tokenizer = HFTokenizer(tokenizer_path)
    prompts = build_text_workload(tokenizer, path=path)
    trace = trace_descriptor(
        prompts,
        path=path,
        tokenizer_sha256=tokenizer_sha256,
    )
    scenario_id = _scenario_id(tensor_parallel_size, path)

    async with httpx.AsyncClient(timeout=request_timeout_s) as client:
        ready = await client.get(f"{endpoint}/readyz")
        if ready.status_code != 200:
            raise RuntimeError(
                f"{scenario_id} endpoint is not ready: HTTP {ready.status_code}"
            )
        topology_response = await client.get(f"{endpoint}/backends")
        if topology_response.status_code != 200:
            raise RuntimeError(
                f"{scenario_id} /backends returned HTTP "
                f"{topology_response.status_code}"
            )
        backends = topology_response.json()
        if not backends_topology_exact(
            backends,
            tensor_parallel_size=tensor_parallel_size,
            path=path,
        ):
            raise ValueError(
                f"{scenario_id} /backends does not bind the requested topology"
            )
        provenance_start = capture_provenance(
            tensor_parallel_size=tensor_parallel_size,
            path=path,
            endpoint=endpoint,
            tokenizer_sha256=tokenizer_sha256,
            config_paths=config_paths,
            expected_commit=expected_commit,
            runtime_container=runtime_container,
            gateway_container=gateway_container,
            checkpoint_evidence_path=checkpoint_evidence_path,
        )
        affinity_before = (
            await _capture_affinity_metric(client, endpoint)
            if path == "gateway"
            else None
        )
        rows: list[dict[str, object]] = [
            {
                "type": "run",
                "schema_version": SCHEMA_VERSION,
                "gate": GATE,
                "scenario_id": scenario_id,
                "tensor_parallel_size": tensor_parallel_size,
                "path": path,
                "workload": workload_config(),
                "trace": trace,
                "backends": backends,
                "provenance_start": provenance_start,
                "gateway_session_affinity_before": affinity_before,
            }
        ]
        for prompt in prompts:
            error_type: str | None = None
            http_status: int | None = None
            response_request_id: str | None = None
            usage: dict[str, int] | None = None
            try:
                response = await client.post(
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "user", "content": prompt.prompt}
                        ],
                        "user": f"g2-a7-session-{prompt.session:02d}",
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "min_tokens": MIN_TOKENS,
                        "ignore_eos": IGNORE_EOS,
                    },
                )
                http_status = response.status_code
                response_request_id = response.headers.get("x-request-id")
                if response.status_code == 200:
                    payload = response.json()
                    usage = _usage_from_response(payload.get("usage"))
                    if usage is None:
                        error_type = "IncompleteEngineUsage"
                else:
                    error_type = "HTTPError"
            except Exception as error:  # retain the sanitized failure class
                error_type = type(error).__name__
            success = error_type is None and usage is not None
            rows.append(
                {
                    "type": "request",
                    "scenario_id": scenario_id,
                    "tensor_parallel_size": tensor_parallel_size,
                    "path": path,
                    "sequence": prompt.sequence,
                    "session": prompt.session,
                    "turn": prompt.turn,
                    "prompt_tokens": len(prompt.prompt_token_ids),
                    "expected_cached_tokens": prompt.expected_cached_tokens,
                    "prompt_sha256": prompt.prompt_sha256,
                    "prompt_token_ids_sha256": (
                        prompt.prompt_token_ids_sha256
                    ),
                    "response_request_id": response_request_id,
                    "http_status": http_status,
                    "status": "success" if success else "failure",
                    "error_type": error_type,
                    "usage": usage,
                }
            )
            if not success:
                break
        affinity_after = (
            await _capture_affinity_metric(client, endpoint)
            if path == "gateway"
            else None
        )

    provenance_end = capture_provenance(
        tensor_parallel_size=tensor_parallel_size,
        path=path,
        endpoint=endpoint,
        tokenizer_sha256=tokenizer_sha256,
        config_paths=config_paths,
        expected_commit=expected_commit,
        runtime_container=runtime_container,
        gateway_container=gateway_container,
        checkpoint_evidence_path=checkpoint_evidence_path,
    )
    rows.append(
        {
            "type": "run_end",
            "scenario_id": scenario_id,
            "tensor_parallel_size": tensor_parallel_size,
            "path": path,
            "request_count": sum(
                row.get("type") == "request" for row in rows
            ),
            "provenance_end": provenance_end,
            "gateway_session_affinity_after": affinity_after,
            "gateway_session_affinity_delta": _affinity_metric_delta(
                affinity_before,
                affinity_after,
            ),
        }
    )
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        normalized.append({**row, "row_index": index})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in normalized),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(
                    f"invalid JSONL framing at {path}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row is not an object at {path}:{line_number}"
                )
            rows.append(value)
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise ValueError(f"non-contiguous row indices in {path}")
    return rows


def write_measurement_shard(
    path: Path,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    _write_jsonl(path, rows)
    stored = _read_jsonl(path)
    start, requests, end = _parse_measurement_shard(stored)
    prompt_tokens = sum(
        int(row["usage"]["prompt_tokens"])
        for row in requests
        if _usage_valid(row.get("usage"))
    )
    cached_tokens = sum(
        int(row["usage"]["cached_tokens"])
        for row in requests
        if _usage_valid(row.get("usage"))
    )
    return {
        "scenario_id": start["scenario_id"],
        "request_count": len(requests),
        "successful_requests": sum(
            row.get("status") == "success" for row in requests
        ),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_rate": (
            cached_tokens / prompt_tokens if prompt_tokens else None
        ),
        "provenance_stable": (
            start.get("provenance_start") == end.get("provenance_end")
        ),
        "sha256": sha256_file(path),
    }


def _parse_measurement_shard(
    rows: Sequence[dict[str, object]],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    if len(rows) < 2:
        raise ValueError("measurement shard is empty")
    start = rows[0]
    end = rows[-1]
    if start.get("type") != "run" or end.get("type") != "run_end":
        raise ValueError("measurement shard must be framed by run/run_end")
    if (
        start.get("schema_version") != SCHEMA_VERSION
        or start.get("gate") != GATE
    ):
        raise ValueError("measurement shard schema/gate mismatch")
    tensor_parallel_size = start.get("tensor_parallel_size")
    path = start.get("path")
    if type(tensor_parallel_size) is not int or not isinstance(path, str):
        raise ValueError("measurement shard scenario identity is invalid")
    scenario_id = _scenario_id(tensor_parallel_size, path)
    if (
        start.get("scenario_id") != scenario_id
        or end.get("scenario_id") != scenario_id
    ):
        raise ValueError("measurement shard scenario identity is inconsistent")
    requests = [row for row in rows[1:-1] if row.get("type") == "request"]
    if len(requests) != len(rows) - 2:
        raise ValueError("measurement shard contains an unknown row type")
    if any(row.get("scenario_id") != scenario_id for row in requests):
        raise ValueError("measurement request belongs to another scenario")
    return start, requests, end


def assemble_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Combine measurement shards into one canonical raw evidence stream."""

    shards: dict[
        tuple[int, str],
        tuple[
            dict[str, object],
            list[dict[str, object]],
            dict[str, object],
            str,
        ],
    ] = {}
    for path in paths:
        rows = _read_jsonl(path)
        start, requests, end = _parse_measurement_shard(rows)
        scenario = (int(start["tensor_parallel_size"]), str(start["path"]))
        if scenario in shards:
            raise ValueError(f"duplicate measurement shard for {scenario}")
        shards[scenario] = (
            start,
            requests,
            end,
            sha256_file(path),
        )
    combined: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "workload": workload_config(),
        }
    ]
    for scenario in sorted(shards):
        start, requests, end, shard_sha256 = shards[scenario]
        combined.append(
            {
                **{
                    key: value
                    for key, value in start.items()
                    if key not in {"type", "row_index"}
                },
                "type": "scenario_start",
                "measurement_raw_sha256": shard_sha256,
            }
        )
        combined.extend(
            {
                key: value
                for key, value in request.items()
                if key != "row_index"
            }
            for request in requests
        )
        combined.append(
            {
                **{
                    key: value
                    for key, value in end.items()
                    if key not in {"type", "row_index"}
                },
                "type": "scenario_end",
            }
        )
    combined.append(
        {
            "type": "run_end",
            "scenario_count": len(shards),
            "request_count": sum(
                len(shard[1]) for shard in shards.values()
            ),
        }
    )
    return combined


def _parse_artifact_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[
    dict[str, object],
    dict[tuple[int, str], dict[str, object]],
    dict[tuple[int, str], list[dict[str, object]]],
    dict[tuple[int, str], dict[str, object]],
    dict[str, object],
]:
    if len(rows) < 2 or rows[0].get("type") != "run":
        raise ValueError("artifact must start with one run row")
    if rows[-1].get("type") != "run_end":
        raise ValueError("artifact must end with one run_end row")
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise ValueError("artifact row indices are not contiguous")
    run = rows[0]
    run_end = rows[-1]
    if (
        run.get("schema_version") != SCHEMA_VERSION
        or run.get("gate") != GATE
    ):
        raise ValueError("artifact schema/gate mismatch")
    starts: dict[tuple[int, str], dict[str, object]] = {}
    ends: dict[tuple[int, str], dict[str, object]] = {}
    requests: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows[1:-1]:
        row_type = row.get("type")
        tensor_parallel_size = row.get("tensor_parallel_size")
        path = row.get("path")
        if type(tensor_parallel_size) is not int or not isinstance(path, str):
            raise ValueError("artifact scenario row has no typed identity")
        scenario = (tensor_parallel_size, path)
        if row.get("scenario_id") != _scenario_id(*scenario):
            raise ValueError("artifact scenario ID mismatch")
        if row_type == "scenario_start":
            if scenario in starts:
                raise ValueError(f"duplicate scenario_start for {scenario}")
            starts[scenario] = row
        elif row_type == "scenario_end":
            if scenario in ends:
                raise ValueError(f"duplicate scenario_end for {scenario}")
            ends[scenario] = row
        elif row_type == "request":
            requests[scenario].append(row)
        else:
            raise ValueError(f"unknown artifact row type {row_type!r}")
    return run, starts, requests, ends, run_end


def _request_trace_exact(
    trace: object,
    requests: Sequence[dict[str, object]],
) -> bool:
    if not isinstance(trace, dict) or not _descriptor_valid(trace):
        return False
    value = trace["value"]
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        return False
    expected = value["rows"]
    if len(requests) != len(expected):
        return False
    immutable = (
        "sequence",
        "session",
        "turn",
        "prompt_tokens",
        "expected_cached_tokens",
        "prompt_sha256",
        "prompt_token_ids_sha256",
    )
    return all(
        isinstance(expected_row, dict)
        and all(
            actual_row.get(key) == expected_row.get(key)
            for key in immutable
        )
        for actual_row, expected_row in zip(requests, expected, strict=True)
    )


def _trace_tokenizer_matches_provenance(
    trace: object,
    provenance: object,
) -> bool:
    if not isinstance(trace, dict) or not isinstance(provenance, dict):
        return False
    trace_value = trace.get("value")
    configuration = provenance.get("configuration")
    if not isinstance(trace_value, dict) or not isinstance(
        configuration,
        dict,
    ):
        return False
    configuration_value = configuration.get("value")
    return (
        isinstance(configuration_value, dict)
        and trace_value.get("tokenizer_sha256")
        == configuration_value.get("tokenizer_sha256")
    )


def evaluate_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[dict[str, bool], dict[str, dict[str, object]], dict[str, object]]:
    run, starts, requests, ends, run_end = _parse_artifact_rows(rows)
    required_scenarios_exact = (
        set(starts)
        == set(requests)
        == set(ends)
        == set(EXPECTED_SCENARIOS)
        and run_end.get("scenario_count") == len(EXPECTED_SCENARIOS)
    )
    fixed_workload_exact = run.get("workload") == workload_config()

    scenario_metrics: dict[str, dict[str, object]] = {}
    counts_exact = required_scenarios_exact
    successes_exact = required_scenarios_exact
    usages_exact = required_scenarios_exact
    topology_exact = required_scenarios_exact
    traces_exact = required_scenarios_exact and fixed_workload_exact
    starts_provenance: dict[tuple[int, str], object] = {}
    ends_provenance: dict[tuple[int, str], object] = {}
    affinity_diagnostics: dict[str, dict[str, object]] = {}
    for scenario in sorted(set(starts) | set(requests) | set(ends)):
        tensor_parallel_size, path = scenario
        scenario_id = _scenario_id(*scenario)
        scenario_requests = requests.get(scenario, [])
        start = starts.get(scenario, {})
        end = ends.get(scenario, {})
        valid_usages = [
            row["usage"]
            for row in scenario_requests
            if _usage_valid(row.get("usage"))
        ]
        prompt_tokens = sum(
            int(usage["prompt_tokens"]) for usage in valid_usages
        )
        cached_tokens = sum(
            int(usage["cached_tokens"]) for usage in valid_usages
        )
        rate = cached_tokens / prompt_tokens if prompt_tokens else None
        success_count = sum(
            row.get("status") == "success" for row in scenario_requests
        )
        scenario_metrics[scenario_id] = {
            "tensor_parallel_size": tensor_parallel_size,
            "path": path,
            "request_count": len(scenario_requests),
            "success_count": success_count,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "pooled_engine_cache_rate": rate,
            "definition": (
                "sum(engine cached_tokens) / sum(engine prompt_tokens)"
            ),
        }
        counts_exact &= (
            len(scenario_requests) == EXPECTED_REQUESTS
            and end.get("request_count") == EXPECTED_REQUESTS
        )
        successes_exact &= (
            len(scenario_requests) == EXPECTED_REQUESTS
            and success_count == EXPECTED_REQUESTS
            and all(
                row.get("http_status") == 200
                and row.get("error_type") is None
                for row in scenario_requests
            )
        )
        usages_exact &= (
            len(scenario_requests) == EXPECTED_REQUESTS
            and len(valid_usages) == EXPECTED_REQUESTS
        )
        topology_exact &= backends_topology_exact(
            start.get("backends"),
            tensor_parallel_size=tensor_parallel_size,
            path=path,
        )
        trace = start.get("trace")
        traces_exact &= (
            _trace_valid(trace, path=path)
            and _request_trace_exact(trace, scenario_requests)
            and _trace_tokenizer_matches_provenance(
                trace,
                start.get("provenance_start"),
            )
        )
        starts_provenance[scenario] = start.get("provenance_start")
        ends_provenance[scenario] = end.get("provenance_end")
        affinity_diagnostics[scenario_id] = {
            "before": start.get("gateway_session_affinity_before"),
            "after": end.get("gateway_session_affinity_after"),
            "delta": end.get("gateway_session_affinity_delta"),
        }

    if required_scenarios_exact:
        for path in PATHS:
            traces_exact &= (
                starts[(4, path)].get("trace")
                == starts[(8, path)].get("trace")
            )
    provenance_exact = _provenance_matrix_exact(
        starts_provenance,
        ends_provenance,
    )
    counts_exact &= (
        run_end.get("request_count")
        == EXPECTED_REQUESTS * len(EXPECTED_SCENARIOS)
    )
    rates_above_floor = required_scenarios_exact and all(
        metric["pooled_engine_cache_rate"] is not None
        and float(metric["pooled_engine_cache_rate"]) > CACHE_RATE_FLOOR
        for metric in scenario_metrics.values()
    )
    checks = {
        "required_tp_path_matrix_exact": required_scenarios_exact,
        "every_scenario_has_512_requests": counts_exact,
        "all_requests_successful": successes_exact,
        "exact_engine_usage_complete": usages_exact,
        "backends_topology_exact": topology_exact,
        "every_pooled_engine_cache_rate_strictly_above_0_80": (
            rates_above_floor
        ),
        "trace_identity_and_fixed_geometry_exact": traces_exact,
        "provenance_start_end_and_cross_scenario_exact": provenance_exact,
    }
    diagnostics = {
        "latency_is_binding": False,
        "output_equality_is_binding": False,
        "routing_proxy_is_binding": False,
        "affinity_is_binding": False,
        "repeat_count_is_binding": False,
        "gateway_session_affinity_by_scenario": affinity_diagnostics,
    }
    return checks, scenario_metrics, diagnostics


def assemble_manifest(
    rows: Sequence[dict[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    if not _valid_sha256(raw_sha256):
        raise ValueError("manifest requires a raw SHA-256")
    checks, scenario_metrics, diagnostics = evaluate_rows(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "workload": workload_config(),
        "raw": {
            "name": RAW_NAME,
            "sha256": raw_sha256,
            "row_count": len(rows),
        },
        "scenarios": scenario_metrics,
        "checks": checks,
        "diagnostics": diagnostics,
        "passed": all(checks.values()),
    }


def write_artifact(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    _write_jsonl(raw_path, rows)
    stored_rows = _read_jsonl(raw_path)
    manifest = assemble_manifest(
        stored_rows,
        raw_sha256=sha256_file(raw_path),
    )
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
    raw_path, manifest_path = _artifact_paths(path)
    rows = _read_jsonl(raw_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    raw_sha256 = sha256_file(raw_path)
    raw_descriptor = manifest.get("raw")
    if (
        not isinstance(raw_descriptor, dict)
        or raw_descriptor.get("sha256") != raw_sha256
    ):
        raise ValueError("raw SHA-256 does not match manifest")
    recomputed = assemble_manifest(rows, raw_sha256=raw_sha256)
    if manifest != recomputed:
        raise ValueError("manifest differs from independent raw replay")
    if assert_gate and recomputed["passed"] is not True:
        raise RuntimeError("G2 A7 artifact does not pass every binding check")
    return recomputed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    attest_parser = subparsers.add_parser(
        "attest-checkpoint",
        help="full-hash one live model volume against pinned parity evidence",
    )
    attest_parser.add_argument("--container", required=True)
    attest_parser.add_argument("--output", type=Path, required=True)

    measure_parser = subparsers.add_parser(
        "measure",
        help="measure one TP/path scenario and write a raw shard",
    )
    measure_parser.add_argument("--tp", type=int, choices=TP_DEGREES, required=True)
    measure_parser.add_argument("--path", choices=PATHS, required=True)
    measure_parser.add_argument("--endpoint", required=True)
    measure_parser.add_argument("--tokenizer-path", type=Path, required=True)
    measure_parser.add_argument("--runtime-container", required=True)
    measure_parser.add_argument("--gateway-container")
    measure_parser.add_argument(
        "--checkpoint-evidence",
        type=Path,
        required=True,
    )
    measure_parser.add_argument(
        "--config-file",
        type=Path,
        action="append",
        required=True,
    )
    measure_parser.add_argument("--expected-commit", required=True)
    measure_parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
    )
    measure_parser.add_argument("--output", type=Path, required=True)

    assemble_parser = subparsers.add_parser(
        "assemble",
        help="assemble four measurement shards into retained evidence",
    )
    assemble_parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        required=True,
    )
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    assemble_parser.add_argument("--assert-gate", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify",
        help="independently replay one retained artifact",
    )
    verify_parser.add_argument("--artifact", type=Path, required=True)
    verify_parser.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "attest-checkpoint":
        evidence = attest_checkpoint(args.container)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    if args.command == "measure":
        rows = asyncio.run(
            measure(
                tensor_parallel_size=args.tp,
                path=args.path,
                endpoint=args.endpoint,
                tokenizer_path=args.tokenizer_path,
                config_paths=tuple(args.config_file),
                expected_commit=args.expected_commit,
                runtime_container=args.runtime_container,
                gateway_container=args.gateway_container,
                checkpoint_evidence_path=args.checkpoint_evidence,
                request_timeout_s=args.request_timeout_s,
            )
        )
        result = write_measurement_shard(args.output, rows)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if (
            result["request_count"] == EXPECTED_REQUESTS
            and result["successful_requests"] == EXPECTED_REQUESTS
            and result["cache_rate"] is not None
            and float(result["cache_rate"]) > CACHE_RATE_FLOOR
            and result["provenance_stable"] is True
        ) else 1
    if args.command == "assemble":
        rows = assemble_rows(tuple(args.raw))
        result = write_artifact(args.output_dir, rows)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] is True or not args.assert_gate else 1
    result = verify_artifact(
        args.artifact,
        assert_gate=args.assert_gate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
