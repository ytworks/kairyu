#!/usr/bin/env python3
"""G4 E-KV: replayable SM120 long-context FP8 KV correctness bake.

The formal run compares one loaded Qwen3-32B model with two fresh paged-cache
arms on one RTX PRO 6000 Blackwell GPU:

* BF16 KV, which remains the product default.
* explicit E4M3 KV with a fixed unit K/V scale.

Both arms execute the same exact 8K/16K/32K text prompts through B=1 native
FlashInfer ragged prefill in 2,048-token chunks, followed by sixteen greedy
tokens through tensor decode.  Timing is deliberately absent from the verdict.

The FP8 arm instruments every production ragged/decode cache write.  It checks
the complete input tensors for finite/range safety, compares the stored E4M3
bytes with an independent torch-cast oracle, and checks the written values
against the format's explicit error bound.  Small, fixed cache samples retain
raw BF16 and FP8 bytes for independent cross-arm replay.

Run inside the retained CUDA image (see ``docs/gpu-runbook.md``):

    python bench/fp8_kv_g4_ekv_bench.py measure \
      --model-path /models/qwen3-32b \
      --image-id sha256:<64-hex> \
      --output-dir /results/g4-ekv \
      --assert-gate

Then independently replay the retained raw JSONL:

    python bench/fp8_kv_g4_ekv_bench.py verify \
      --artifact /results/g4-ekv --assert-gate
    python bench/fp8_kv_g4_ekv_bench.py replay \
      --artifact /results/g4-ekv --assert-gate

A completed measurement is always written.  A model/runtime exception or a
failed correctness check therefore produces retained FAIL evidence; it can
never be presented as PASS.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "kairyu.g4.e-kv.fp8-kv.v1"
GATE = "G4-E-KV"
RAW_NAME = "fp8-kv-g4-ekv-raw.jsonl"
MANIFEST_NAME = "fp8-kv-g4-ekv-manifest.json"

MODEL = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
EXPECTED_CONFIG_SHA256 = (
    "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
)
EXPECTED_INDEX_SHA256 = (
    "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
)
EXPECTED_TOKENIZER_SHA256 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)

PROMPT_LENGTHS = (8_192, 16_384, 32_768)
PREFILL_CHUNK_TOKENS = 2_048
OUTPUT_TOKENS = 16
PAGE_SIZE = 16
NUM_LAYERS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 151_936
ARMS = ("bf16", "fp8_e4m3")
FP8_SCALE = 1.0
FP8_MAX = 448.0
FP8_MIN_ABS_ERROR = 2.0**-10
LOGPROB_ABS_DELTA_MAX = 0.25
CACHE_NRMSE_MAX = 0.05
CACHE_COSINE_MIN = 0.99
SAMPLE_LAYERS = (0, 1, 7, 15, 31, 47, 62, 63)
SAMPLE_POSITIONS_PER_PROMPT = 16
_PROMPT_CONSTRUCTION = "stable-single-token-repeat-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")

BOUND_SOURCE_FILES = (
    "Dockerfile.cuda",
    "bench/fp8_kv_g4_ekv_bench.py",
    "kairyu/engine/config_validation.py",
    "kairyu/engine/core/attention/__init__.py",
    "kairyu/engine/core/attention/flashinfer_gpu.py",
    "kairyu/engine/core/attention/selector.py",
    "kairyu/engine/core/attention_selector.py",
    "kairyu/engine/core/engine_service.py",
    "kairyu/engine/core/kv_cache_dtype.py",
    "kairyu/engine/core/kv_pool.py",
    "kairyu/engine/core/prefill.py",
    "kairyu/engine/core/worker.py",
    "kairyu/engine/kairyu_backend.py",
    "kairyu/engine/zmq_backend.py",
    "kairyu/entrypoints/server/health.py",
    "kairyu/kernels/paged_kv_write_gpu.py",
    "kairyu/models/attention.py",
    "kairyu/models/llama.py",
    "kairyu/models/loader.py",
    "kairyu/engine/tokenizer.py",
    "pyproject.toml",
    "uv.lock",
)

EXPECTED_WEIGHTS = (
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
EXPECTED_WEIGHT_ROWS = tuple(
    {"name": name, "size": size, "sha256": digest}
    for name, size, digest in EXPECTED_WEIGHTS
)


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


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_snapshot() -> dict[str, object]:
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    files: dict[str, str] = {}
    for relative in BOUND_SOURCE_FILES:
        path = _REPO_ROOT / relative
        if path.is_file():
            files[relative] = sha256_file(path)
    return {
        "commit": commit,
        "tracked_dirty": None if dirty is None else bool(dirty),
        "files": files,
    }


def _source_snapshot_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"commit", "tracked_dirty", "files"}
        and isinstance(value["commit"], str)
        and _COMMIT.fullmatch(value["commit"]) is not None
        and value["tracked_dirty"] is False
        and isinstance(value["files"], dict)
        and set(value["files"]) == set(BOUND_SOURCE_FILES)
        and all(
            isinstance(digest, str) and _HEX64.fullmatch(digest) is not None
            for digest in value["files"].values()
        )
    )


def _checkpoint_snapshot(model_path: Path) -> dict[str, object]:
    weights = [model_path / name for name, _size, _digest in EXPECTED_WEIGHTS]
    if any(not path.is_file() for path in weights):
        missing = [path.name for path in weights if not path.is_file()]
        raise ValueError(f"checkpoint omits weight shards: {missing}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        digests = tuple(executor.map(sha256_file, weights))
    weight_rows = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(weights, digests, strict=True)
    ]
    revision_file = (
        model_path / ".cache/huggingface/download/config.json.metadata"
    )
    revision_lines = (
        revision_file.read_text(encoding="utf-8").splitlines()
        if revision_file.is_file()
        else []
    )
    return {
        "model": MODEL,
        "revision": revision_lines[0].strip() if revision_lines else None,
        "config_sha256": sha256_file(model_path / "config.json"),
        "index_sha256": sha256_file(
            model_path / "model.safetensors.index.json"
        ),
        "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
        "weights": weight_rows,
        "weights_rollup_sha256": sha256_json(weight_rows),
    }


def _checkpoint_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "model",
            "revision",
            "config_sha256",
            "index_sha256",
            "tokenizer_sha256",
            "weights",
            "weights_rollup_sha256",
        }
        and value["model"] == MODEL
        and value["revision"] == MODEL_REVISION
        and value["config_sha256"] == EXPECTED_CONFIG_SHA256
        and value["index_sha256"] == EXPECTED_INDEX_SHA256
        and value["tokenizer_sha256"] == EXPECTED_TOKENIZER_SHA256
        and value["weights"] == list(EXPECTED_WEIGHT_ROWS)
        and value["weights_rollup_sha256"]
        == sha256_json(list(EXPECTED_WEIGHT_ROWS))
    )


def _nvidia_smi_inventory() -> list[dict[str, object]]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"invalid nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": fields[2],
                "name": fields[3],
                "driver_version": fields[4],
                "memory_total_mib": int(fields[5]),
            }
        )
    return rows


def _stable_cc1plus_provenance(value: str) -> str:
    """Drop GCC's per-invocation timing while retaining compiler identity."""

    return value.partition("\nTime variable")[0].rstrip()


def _stable_environment_identity(value: object) -> object:
    """Canonicalize the one known non-identity field in a valid snapshot."""

    if not isinstance(value, dict):
        return value
    normalized = json.loads(json.dumps(value))
    try:
        cc1plus = normalized["compiler"]["host_compiler"]["cc1plus"]
        cc1plus["version"] = _stable_cc1plus_provenance(
            cc1plus["version"]
        )
    except (KeyError, TypeError):
        return normalized
    return normalized


def _compiler_snapshot() -> dict[str, object]:
    def tool(
        name: str,
        *,
        path: str | None = None,
        version_args: tuple[str, ...] = ("--version",),
    ) -> dict[str, object]:
        selected = path or shutil.which(name)
        if selected is None:
            raise RuntimeError(
                f"formal G4 E-KV source-JIT execution requires {name}"
            )
        resolved = Path(selected).resolve()
        completed = subprocess.run(
            (str(resolved), *version_args),
            check=True,
            capture_output=True,
            text=True,
        )
        version = "\n".join(
            line.rstrip()
            for line in (
                completed.stdout + completed.stderr
            ).splitlines()
        )
        if not version:
            raise RuntimeError(f"{name} --version returned no provenance")
        if name == "cc1plus":
            version = _stable_cc1plus_provenance(version)
        return {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "version": version,
        }

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError(
            "formal G4 E-KV source-JIT execution requires nvcc on PATH"
        )
    nvcc_path = Path(nvcc).resolve()
    completed = subprocess.run(
        (str(nvcc_path), "--version"),
        check=True,
        capture_output=True,
        text=True,
    )
    version = "\n".join(
        line.rstrip() for line in completed.stdout.splitlines()
    )
    if not version:
        raise RuntimeError("nvcc --version returned no provenance")
    gcc = tool("gcc")
    gxx = tool("g++")
    cc1plus_path = subprocess.run(
        (str(gxx["path"]), "-print-prog-name=cc1plus"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    host_compiler = {
        "gcc": gcc,
        "g++": gxx,
        "cc1plus": tool(
            "cc1plus",
            path=cc1plus_path,
            # GCC 13's cc1plus emits nothing for `--version`. Syntax-checking
            # /dev/null with `-version` emits provenance, exits zero, and does
            # not create an assembly file in the read-only source checkout.
            version_args=("-version", "-fsyntax-only", "/dev/null"),
        ),
    }
    home = os.environ.get("HOME")
    workspace_base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    cache_base = workspace_base or home
    if not home or not cache_base:
        raise RuntimeError(
            "formal G4 E-KV requires HOME and a resolvable FlashInfer cache"
        )
    cache_root = (Path(cache_base) / ".cache" / "flashinfer").resolve()
    shared_objects = [
        {
            "path": path.relative_to(cache_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(cache_root.rglob("*.so"))
    ]
    return {
        "mode": "container-with-mounted-source-jit-runtime",
        "python_executable": str(Path(sys.executable).resolve()),
        "nvcc_path": str(nvcc_path),
        "nvcc_sha256": sha256_file(nvcc_path),
        "nvcc_version": version,
        "host_compiler": host_compiler,
        "HOME": home,
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "FLASHINFER_WORKSPACE_BASE": workspace_base,
        # Retain this legacy/operator spelling too. FlashInfer 0.6.14 resolves
        # its cache from FLASHINFER_WORKSPACE_BASE, not from this variable.
        "FLASHINFER_WORKSPACE_FOLDER": os.environ.get(
            "FLASHINFER_WORKSPACE_FOLDER"
        ),
        "resolved_flashinfer_cache_root": str(cache_root),
        "flashinfer_shared_objects": shared_objects,
        "flashinfer_shared_objects_rollup_sha256": sha256_json(
            shared_objects
        ),
    }


def _environment_snapshot(
    *,
    backend_decision: object | None = None,
) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("G4 E-KV requires CUDA")
    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise RuntimeError(
            "formal G4 E-KV requires exactly one visible CUDA GPU, "
            f"found {device_count}"
        )
    properties = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    decision = None
    if backend_decision is not None:
        decision = {
            "requested": getattr(backend_decision, "requested", None),
            "resolved": getattr(backend_decision, "resolved", None),
            "source": getattr(backend_decision, "source", None),
            "components": dict(
                getattr(backend_decision, "components", {}) or {}
            ),
            "architecture": dict(
                getattr(backend_decision, "architecture", {}) or {}
            ),
        }
    return {
        "packages": {
            "python": ".".join(
                str(part)
                for part in (
                    sys.version_info.major,
                    sys.version_info.minor,
                    sys.version_info.micro,
                )
            ),
            "torch": _package_version("torch") or torch.__version__,
            "flashinfer": _package_version(
                "flashinfer-python", "flashinfer"
            ),
            "triton": _package_version("triton"),
        },
        "cuda": {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "visible_device_count": device_count,
        },
        "gpu": {
            "torch_name": properties.name,
            "compute_capability": [major, minor],
            "sm": major * 10 + minor,
            "total_memory_bytes": int(properties.total_memory),
            "nvidia_smi": _nvidia_smi_inventory(),
        },
        "visibility": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get(
                "NVIDIA_VISIBLE_DEVICES"
            ),
        },
        "compiler": _compiler_snapshot(),
        "attention_backend": decision,
    }


def _environment_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "packages",
        "cuda",
        "gpu",
        "visibility",
        "compiler",
        "attention_backend",
    }:
        return False
    packages = value["packages"]
    cuda = value["cuda"]
    gpu = value["gpu"]
    compiler = value["compiler"]
    decision = value["attention_backend"]
    if not all(
        isinstance(item, dict)
        for item in (packages, cuda, gpu, compiler)
    ):
        return False
    inventory = gpu.get("nvidia_smi")
    shared_objects = compiler.get("flashinfer_shared_objects")
    host_compiler = compiler.get("host_compiler")
    shared_object_paths = (
        [
            row["path"]
            for row in shared_objects
            if isinstance(row, dict)
            and isinstance(row.get("path"), str)
        ]
        if isinstance(shared_objects, list)
        else []
    )
    return (
        set(packages) == {"python", "torch", "flashinfer", "triton"}
        and set(cuda)
        == {
            "torch_cuda_version",
            "cudnn_version",
            "visible_device_count",
        }
        and set(gpu)
        == {
            "torch_name",
            "compute_capability",
            "sm",
            "total_memory_bytes",
            "nvidia_smi",
        }
        and isinstance(value["visibility"], dict)
        and set(value["visibility"])
        == {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
        and isinstance(packages.get("python"), str)
        and bool(packages["python"])
        and isinstance(packages.get("torch"), str)
        and bool(packages["torch"])
        and isinstance(packages.get("flashinfer"), str)
        and bool(packages["flashinfer"])
        and isinstance(packages.get("triton"), str)
        and bool(packages["triton"])
        and cuda.get("visible_device_count") == 1
        and isinstance(cuda.get("torch_cuda_version"), str)
        and gpu.get("torch_name") == EXPECTED_GPU_NAME
        and gpu.get("compute_capability") == [12, 0]
        and gpu.get("sm") == 120
        and type(gpu.get("total_memory_bytes")) is int
        and gpu["total_memory_bytes"] > 90 * 1024**3
        and isinstance(inventory, list)
        and len(inventory) == 1
        and isinstance(inventory[0], dict)
        and set(inventory[0])
        == {
            "index",
            "uuid",
            "pci_bus_id",
            "name",
            "driver_version",
            "memory_total_mib",
        }
        and type(inventory[0].get("index")) is int
        and inventory[0].get("name") == EXPECTED_GPU_NAME
        and isinstance(inventory[0].get("uuid"), str)
        and inventory[0]["uuid"].startswith("GPU-")
        and isinstance(inventory[0].get("pci_bus_id"), str)
        and bool(inventory[0]["pci_bus_id"])
        and isinstance(inventory[0].get("driver_version"), str)
        and bool(inventory[0]["driver_version"])
        and type(inventory[0].get("memory_total_mib")) is int
        and inventory[0]["memory_total_mib"] > 90 * 1024
        and compiler.get("mode")
        == "container-with-mounted-source-jit-runtime"
        and set(compiler)
        == {
            "mode",
            "python_executable",
            "nvcc_path",
            "nvcc_sha256",
            "nvcc_version",
            "host_compiler",
            "HOME",
            "CUDA_HOME",
            "TORCH_EXTENSIONS_DIR",
            "FLASHINFER_WORKSPACE_BASE",
            "FLASHINFER_WORKSPACE_FOLDER",
            "resolved_flashinfer_cache_root",
            "flashinfer_shared_objects",
            "flashinfer_shared_objects_rollup_sha256",
        }
        and isinstance(compiler.get("python_executable"), str)
        and Path(compiler["python_executable"]).is_absolute()
        and isinstance(compiler.get("nvcc_path"), str)
        and Path(compiler["nvcc_path"]).is_absolute()
        and isinstance(compiler.get("nvcc_sha256"), str)
        and _HEX64.fullmatch(compiler["nvcc_sha256"]) is not None
        and isinstance(compiler.get("nvcc_version"), str)
        and "Cuda compilation tools" in compiler["nvcc_version"]
        and isinstance(host_compiler, dict)
        and set(host_compiler) == {"gcc", "g++", "cc1plus"}
        and all(
            isinstance(tool, dict)
            and set(tool) == {"path", "sha256", "version"}
            and isinstance(tool["path"], str)
            and Path(tool["path"]).is_absolute()
            and isinstance(tool["sha256"], str)
            and _HEX64.fullmatch(tool["sha256"]) is not None
            and isinstance(tool["version"], str)
            and bool(tool["version"])
            for tool in host_compiler.values()
        )
        and isinstance(compiler.get("HOME"), str)
        and Path(compiler["HOME"]).is_absolute()
        and isinstance(compiler.get("CUDA_HOME"), str)
        and Path(compiler["CUDA_HOME"]).is_absolute()
        and isinstance(compiler.get("TORCH_EXTENSIONS_DIR"), str)
        and Path(compiler["TORCH_EXTENSIONS_DIR"]).is_absolute()
        and isinstance(
            compiler.get("FLASHINFER_WORKSPACE_BASE"), str
        )
        and Path(
            compiler["FLASHINFER_WORKSPACE_BASE"]
        ).is_absolute()
        and (
            compiler.get("FLASHINFER_WORKSPACE_FOLDER") is None
            or isinstance(
                compiler.get("FLASHINFER_WORKSPACE_FOLDER"), str
            )
        )
        and isinstance(
            compiler.get("resolved_flashinfer_cache_root"), str
        )
        and Path(
            compiler["resolved_flashinfer_cache_root"]
        ).is_absolute()
        and Path(compiler["resolved_flashinfer_cache_root"])
        == Path(compiler["FLASHINFER_WORKSPACE_BASE"])
        / ".cache"
        / "flashinfer"
        and isinstance(shared_objects, list)
        and len(shared_objects) >= 2
        and all(
            isinstance(row, dict)
            and set(row) == {"path", "size", "sha256"}
            and isinstance(row["path"], str)
            and bool(row["path"])
            and type(row["size"]) is int
            and row["size"] > 0
            and isinstance(row["sha256"], str)
            and _HEX64.fullmatch(row["sha256"]) is not None
            for row in shared_objects
        )
        and len(shared_object_paths) == len(shared_objects)
        and len(set(shared_object_paths)) == len(shared_object_paths)
        and any("dtype_kv_bf16" in path for path in shared_object_paths)
        and any("dtype_kv_e4m3" in path for path in shared_object_paths)
        and compiler.get("flashinfer_shared_objects_rollup_sha256")
        == sha256_json(shared_objects)
        and isinstance(decision, dict)
        and set(decision)
        == {
            "requested",
            "resolved",
            "source",
            "components",
            "architecture",
        }
        and decision.get("requested") == "flashinfer"
        and decision.get("resolved") == "flashinfer"
        and decision.get("source") == "env"
        and decision.get("components")
        == {
            "prefill": "flashinfer",
            "decode": "flashinfer",
            "kv_mode": "paged-direct",
        }
        and isinstance(decision.get("architecture"), dict)
    )


def sample_positions(length: int) -> tuple[int, ...]:
    candidates = (
        0,
        1,
        15,
        16,
        63,
        255,
        1_023,
        2_047,
        length // 4,
        length // 2 - 1,
        length // 2,
        3 * length // 4 - 1,
        3 * length // 4,
        length - 3,
        length - 2,
        length - 1,
    )
    if len(set(candidates)) != SAMPLE_POSITIONS_PER_PROMPT:
        raise AssertionError(f"sample positions collide for length {length}")
    if any(position < 0 or position >= length for position in candidates):
        raise AssertionError(f"sample positions escape length {length}")
    return candidates


def formal_config() -> dict[str, object]:
    return {
        "gate": GATE,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tensor_parallel_size": 1,
        "visible_gpu_count": 1,
        "required_sm": 120,
        "compute_dtype": "bfloat16",
        "attention_backend": "flashinfer",
        "arms": {
            "bf16": {
                "kv_dtype": "bfloat16",
                "k_scale": None,
                "v_scale": None,
                "default": True,
            },
            "fp8_e4m3": {
                "kv_dtype": "float8_e4m3fn",
                "k_scale": FP8_SCALE,
                "v_scale": FP8_SCALE,
                "default": False,
            },
        },
        "prompt_construction": _PROMPT_CONSTRUCTION,
        "prompt_lengths": list(PROMPT_LENGTHS),
        "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
        "page_size": PAGE_SIZE,
        "greedy_output_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "sample_layers": list(SAMPLE_LAYERS),
        "sample_positions": {
            str(length): list(sample_positions(length))
            for length in PROMPT_LENGTHS
        },
        "cache_geometry": {
            "num_layers": NUM_LAYERS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
        },
        "thresholds": {
            "output_token_ids": "exact",
            "finish_reason": "exact-length",
            "selected_logprob_max_abs_delta": LOGPROB_ABS_DELTA_MAX,
            "fp8_max_finite": FP8_MAX,
            "fp8_quantization": "SATFINITE clamp[-448,448] then E4M3",
            "fp8_dequant_error": (
                "abs(stored.float32-input.float32) <= "
                "max(abs(input.float32)/16, 2^-10)"
            ),
            "cache_sample_nrmse_max": CACHE_NRMSE_MAX,
            "cache_sample_cosine_min": CACHE_COSINE_MIN,
        },
        "timing_is_binding": False,
    }


def _expected_audit_counts() -> dict[tuple[str, str], dict[str, int]]:
    prefill_tokens = sum(PROMPT_LENGTHS)
    decode_tokens = len(PROMPT_LENGTHS) * (OUTPUT_TOKENS - 1)
    scalars_per_token = NUM_KV_HEADS * HEAD_DIM
    return {
        ("ragged_prefill", kind): {
            "calls": sum(
                length // PREFILL_CHUNK_TOKENS
                for length in PROMPT_LENGTHS
            )
            * NUM_LAYERS,
            "values": prefill_tokens * NUM_LAYERS * scalars_per_token,
        }
        for kind in ("k", "v")
    } | {
        ("tensor_decode", kind): {
            "calls": decode_tokens * NUM_LAYERS,
            "values": decode_tokens * NUM_LAYERS * scalars_per_token,
        }
        for kind in ("k", "v")
    }


def _stable_token_pieces(
    tokenizer: object,
    *,
    required: int,
) -> tuple[tuple[int, str], ...]:
    pieces: list[tuple[int, str]] = []
    seen: set[str] = set()
    vocab = tokenizer.vocab()
    for token_id in range(len(vocab)):
        text = tokenizer.decode((token_id,))
        if (
            not text
            or text in seen
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
        seen.add(text)
        pieces.append((token_id, text))
        if len(pieces) == required:
            return tuple(pieces)
    raise ValueError(
        f"tokenizer provides only {len(pieces)} stable repeated pieces; "
        f"{required} are required"
    )


def _prompt_rows(tokenizer: object) -> list[dict[str, object]]:
    pieces = _stable_token_pieces(tokenizer, required=len(PROMPT_LENGTHS))
    rows: list[dict[str, object]] = []
    for index, (length, (token_id, piece)) in enumerate(
        zip(PROMPT_LENGTHS, pieces, strict=True)
    ):
        text = piece * length
        token_ids = tokenizer.encode(text)
        expected = (token_id,) * length
        if token_ids != expected:
            raise ValueError(
                f"prompt {index} lost stable-token identity at length {length}"
            )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "type": "prompt",
                "prompt_id": f"long-{length}",
                "length": length,
                "piece_token_id": token_id,
                "piece_text": piece,
                "text": text,
                "text_sha256": sha256_text(text),
                "token_ids": list(token_ids),
                "token_ids_sha256": sha256_json(list(token_ids)),
            }
        )
    return rows


def _tensor_bytes(tensor: object) -> bytes:
    import torch

    value = tensor.detach().contiguous().view(torch.uint8).cpu()
    return value.numpy().tobytes()


def _cache_sample_payload(
    pool: object,
    *,
    prompt_id: str,
    length: int,
    page_table: Sequence[int],
) -> dict[tuple[int, int, str], bytes]:
    rows: dict[tuple[int, int, str], bytes] = {}
    for layer in SAMPLE_LAYERS:
        for position in sample_positions(length):
            page = page_table[position // PAGE_SIZE]
            slot = position % PAGE_SIZE
            for kind in ("k", "v"):
                tensor = getattr(pool, kind)[layer, page, slot]
                rows[(layer, position, kind)] = _tensor_bytes(tensor)
    expected = (
        len(SAMPLE_LAYERS) * SAMPLE_POSITIONS_PER_PROMPT * 2
    )
    if len(rows) != expected:
        raise AssertionError(
            f"{prompt_id} retained {len(rows)} cache samples, expected {expected}"
        )
    return rows


class _FP8WriteAuditor:
    """Test-only observer around the production FP8 ragged/decode writes."""

    def __init__(self, pool: object) -> None:
        import torch

        self.pool = pool
        self.torch = torch
        self._original_write = pool.write
        self._original_ragged = pool.write_ragged
        self._original_batched = pool.write_batched
        self._stats: dict[tuple[str, str], dict[str, object]] = {}
        for key in _expected_audit_counts():
            self._stats[key] = {
                "calls": 0,
                "values": 0,
                "nonfinite": torch.zeros(
                    (), dtype=torch.int64, device=pool.k.device
                ),
                "overrange": torch.zeros(
                    (), dtype=torch.int64, device=pool.k.device
                ),
                "input_dtype_mismatch": 0,
                "stored_nonfinite": torch.zeros(
                    (), dtype=torch.int64, device=pool.k.device
                ),
                "byte_mismatch": torch.zeros(
                    (), dtype=torch.int64, device=pool.k.device
                ),
                "error_violation": torch.zeros(
                    (), dtype=torch.int64, device=pool.k.device
                ),
                "max_abs_input": torch.zeros(
                    (), dtype=torch.float32, device=pool.k.device
                ),
                "max_abs_error": torch.zeros(
                    (), dtype=torch.float32, device=pool.k.device
                ),
                "max_bound_excess": torch.zeros(
                    (), dtype=torch.float32, device=pool.k.device
                ),
            }

    def install(self) -> None:
        self.pool.write = self._unexpected_write
        self.pool.write_ragged = self.write_ragged
        self.pool.write_batched = self.write_batched

    def restore(self) -> None:
        self.pool.write = self._original_write
        self.pool.write_ragged = self._original_ragged
        self.pool.write_batched = self._original_batched

    def _unexpected_write(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "formal E-KV execution escaped ragged prefill/tensor decode"
        )

    def _audit(
        self,
        phase: str,
        kind: str,
        source: object,
        actual: object,
    ) -> None:
        torch = self.torch
        source_f32 = source.to(torch.float32)
        finite_source = torch.isfinite(source_f32)
        source_for_math = torch.where(
            finite_source, source_f32, torch.zeros_like(source_f32)
        )
        expected = source_f32.clamp(-FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn
        )
        actual_f32 = actual.to(torch.float32)
        error = (actual_f32 - source_for_math).abs()
        bound = torch.maximum(
            source_for_math.abs() / 16.0,
            torch.full_like(source_for_math, FP8_MIN_ABS_ERROR),
        )
        stats = self._stats[(phase, kind)]
        stats["calls"] += 1
        stats["values"] += source.numel()
        if source.dtype != torch.bfloat16:
            stats["input_dtype_mismatch"] += source.numel()
        stats["nonfinite"].add_(
            torch.count_nonzero(~finite_source)
        )
        stats["overrange"].add_(
            torch.count_nonzero(source_for_math.abs() > FP8_MAX)
        )
        stats["stored_nonfinite"].add_(
            torch.count_nonzero(~torch.isfinite(actual_f32))
        )
        stats["byte_mismatch"].add_(
            torch.count_nonzero(
                actual.view(torch.uint8) != expected.view(torch.uint8)
            )
        )
        stats["error_violation"].add_(
            torch.count_nonzero(error > bound)
        )
        stats["max_abs_input"].copy_(
            torch.maximum(
                stats["max_abs_input"], source_for_math.abs().max()
            )
        )
        stats["max_abs_error"].copy_(
            torch.maximum(stats["max_abs_error"], error.max())
        )
        stats["max_bound_excess"].copy_(
            torch.maximum(
                stats["max_bound_excess"],
                torch.clamp_min(error - bound, 0.0).max(),
            )
        )

    def write_ragged(
        self,
        layer: int,
        page_tables: object,
        row_ids: object,
        positions: object,
        keys: object,
        values: object,
        write_from: object,
    ) -> None:
        self._original_ragged(
            layer,
            page_tables,
            row_ids,
            positions,
            keys,
            values,
            write_from,
        )
        owners = row_ids.long()
        page_index = self.torch.div(
            positions, self.pool.page_size, rounding_mode="floor"
        ).long()
        pages = page_tables[owners, page_index].long()
        slots = positions.long() - page_index * self.pool.page_size
        flat = pages * self.pool.page_size + slots
        writable = positions >= write_from[owners]
        flat = flat[writable]
        actual_k = self.pool.k[layer].reshape(
            -1, self.pool.num_kv_heads, self.pool.head_dim
        )[flat]
        actual_v = self.pool.v[layer].reshape(
            -1, self.pool.num_kv_heads, self.pool.v_head_dim
        )[flat]
        self._audit("ragged_prefill", "k", keys[writable], actual_k)
        self._audit("ragged_prefill", "v", values[writable], actual_v)

    def write_batched(
        self,
        layer: int,
        page_tables: object,
        positions: object,
        keys: object,
        values: object,
        write_from: object | None = None,
    ) -> None:
        self._original_batched(
            layer,
            page_tables,
            positions,
            keys,
            values,
            write_from,
        )
        page_index = self.torch.div(
            positions, self.pool.page_size, rounding_mode="floor"
        ).long()
        slots = positions.long() - page_index * self.pool.page_size
        pages = page_tables.gather(
            1, page_index.unsqueeze(1)
        ).squeeze(1).long()
        flat = pages * self.pool.page_size + slots
        writable = (
            self.torch.ones_like(positions, dtype=self.torch.bool)
            if write_from is None
            else positions >= write_from
        )
        flat = flat[writable]
        actual_k = self.pool.k[layer].reshape(
            -1, self.pool.num_kv_heads, self.pool.head_dim
        )[flat]
        actual_v = self.pool.v[layer].reshape(
            -1, self.pool.num_kv_heads, self.pool.v_head_dim
        )[flat]
        self._audit("tensor_decode", "k", keys[writable], actual_k)
        self._audit("tensor_decode", "v", values[writable], actual_v)

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for phase, kind in sorted(self._stats):
            stats = self._stats[(phase, kind)]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "type": "write_audit",
                    "phase": phase,
                    "kind": kind,
                    "input_dtype": "bfloat16",
                    "storage_dtype": "float8_e4m3fn",
                    "scale": FP8_SCALE,
                    "calls": stats["calls"],
                    "values": stats["values"],
                    "input_nonfinite": int(stats["nonfinite"].item()),
                    "input_overrange": int(stats["overrange"].item()),
                    "input_dtype_mismatch": stats[
                        "input_dtype_mismatch"
                    ],
                    "stored_nonfinite": int(
                        stats["stored_nonfinite"].item()
                    ),
                    "stored_byte_mismatch": int(
                        stats["byte_mismatch"].item()
                    ),
                    "dequant_error_violation": int(
                        stats["error_violation"].item()
                    ),
                    "max_abs_input": float(
                        stats["max_abs_input"].item()
                    ),
                    "max_abs_error": float(
                        stats["max_abs_error"].item()
                    ),
                    "max_bound_excess": float(
                        stats["max_bound_excess"].item()
                    ),
                }
            )
        return rows


def _new_pool(config: object, *, dtype: object, device: object) -> object:
    from kairyu.engine.core.kv_pool import PagedKVPool

    num_pages = math.ceil(
        (max(PROMPT_LENGTHS) + OUTPUT_TOKENS) / PAGE_SIZE
    )
    return PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        num_kv_heads=config.kv_cache_num_heads,
        head_dim=config.kv_cache_head_dim,
        dtype=dtype,
        device=str(device),
        v_head_dim=config.kv_cache_v_head_dim,
    )


def _selected_token_and_logprob(logits: object) -> tuple[int, float]:
    import torch

    row = logits.reshape(-1, logits.shape[-1])[-1].to(torch.float32)
    token = int(torch.argmax(row).item())
    logprob = float(torch.log_softmax(row, dim=-1)[token].item())
    if not math.isfinite(logprob):
        raise RuntimeError("selected token logprob is not finite")
    return token, logprob


def _run_prompt(
    model: object,
    pool: object,
    prompt: Mapping[str, object],
    *,
    device: object,
) -> tuple[dict[str, object], dict[tuple[int, int, str], bytes]]:
    import torch

    from kairyu.engine.core.prefill import (
        PrefillSequence,
        build_prefill_batch,
    )

    prompt_id = str(prompt["prompt_id"])
    token_ids = tuple(int(token) for token in prompt["token_ids"])
    length = len(token_ids)
    page_table = tuple(range(pool.num_pages))
    hidden = None
    for chunk_start in range(0, length, PREFILL_CHUNK_TOKENS):
        chunk = token_ids[
            chunk_start : chunk_start + PREFILL_CHUNK_TOKENS
        ]
        seq_len = chunk_start + len(chunk)
        batch = build_prefill_batch(
            (
                PrefillSequence(
                    request_id=prompt_id,
                    token_ids=chunk,
                    page_table=page_table,
                    chunk_start=chunk_start,
                    seq_len=seq_len,
                    write_from=chunk_start,
                ),
            ),
            page_size=PAGE_SIZE,
            device=device,
        )
        hidden = model.forward_prefill_batch(batch, pool)
    if hidden is None:
        raise AssertionError("formal prompt unexpectedly had no prefill")

    outputs: list[int] = []
    selected_logprobs: list[float] = []
    token, logprob = _selected_token_and_logprob(
        model.logits(hidden[-1:])
    )
    outputs.append(token)
    selected_logprobs.append(logprob)

    page_tables = torch.tensor(
        [page_table], dtype=torch.int32, device=device
    )
    for output_index in range(1, OUTPUT_TOKENS):
        position = length + output_index - 1
        seq_len = position + 1
        positions = torch.tensor(
            [position], dtype=torch.int64, device=device
        )
        seq_lens = torch.tensor(
            [seq_len], dtype=torch.int64, device=device
        )
        write_from = torch.tensor(
            [position], dtype=torch.int64, device=device
        )
        model.plan_decode_tensors(pool, page_tables, seq_lens)
        hidden = model.forward_decode_tensors(
            torch.tensor(
                [outputs[-1]], dtype=torch.int64, device=device
            ),
            positions,
            pool,
            page_tables,
            seq_lens,
            write_from,
        )
        token, logprob = _selected_token_and_logprob(model.logits(hidden))
        outputs.append(token)
        selected_logprobs.append(logprob)

    samples = _cache_sample_payload(
        pool,
        prompt_id=prompt_id,
        length=length,
        page_table=page_table,
    )
    return (
        {
            "output_token_ids": outputs,
            "selected_logprobs": selected_logprobs,
            "finish_reason": "length",
        },
        samples,
    )


def _run_arm(
    model: object,
    config: object,
    tokenizer: object,
    prompts: Sequence[Mapping[str, object]],
    *,
    arm: str,
    device: object,
) -> tuple[
    list[dict[str, object]],
    dict[tuple[str, int, int, str], bytes],
    list[dict[str, object]],
]:
    import torch

    if arm == "bf16":
        dtype = torch.bfloat16
    elif arm == "fp8_e4m3":
        dtype = torch.float8_e4m3fn
    else:
        raise ValueError(f"unknown arm {arm!r}")
    pool = _new_pool(config, dtype=dtype, device=device)
    auditor = _FP8WriteAuditor(pool) if arm == "fp8_e4m3" else None
    if auditor is not None:
        if (
            pool.k.dtype is not torch.float8_e4m3fn
            or pool.v.dtype is not torch.float8_e4m3fn
            or any(pool.k_scale(layer) != FP8_SCALE for layer in SAMPLE_LAYERS)
            or any(pool.v_scale(layer) != FP8_SCALE for layer in SAMPLE_LAYERS)
        ):
            raise RuntimeError("FP8 pool does not expose the unit-scale contract")
        auditor.install()

    output_rows: list[dict[str, object]] = []
    samples: dict[tuple[str, int, int, str], bytes] = {}
    try:
        for prompt in prompts:
            result, prompt_samples = _run_prompt(
                model, pool, prompt, device=device
            )
            output_text = tokenizer.decode(
                tuple(int(token) for token in result["output_token_ids"])
            )
            output_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "type": "output",
                    "prompt_id": prompt["prompt_id"],
                    "prompt_length": prompt["length"],
                    "arm": arm,
                    "kv_dtype": (
                        "bfloat16"
                        if arm == "bf16"
                        else "float8_e4m3fn"
                    ),
                    "k_scale": None if arm == "bf16" else FP8_SCALE,
                    "v_scale": None if arm == "bf16" else FP8_SCALE,
                    "output_token_ids": result["output_token_ids"],
                    "output_token_ids_sha256": sha256_json(
                        result["output_token_ids"]
                    ),
                    "selected_logprobs": result["selected_logprobs"],
                    "finish_reason": result["finish_reason"],
                    "output_text": output_text,
                    "output_text_sha256": sha256_text(output_text),
                }
            )
            for (layer, position, kind), raw in prompt_samples.items():
                samples[
                    (
                        str(prompt["prompt_id"]),
                        layer,
                        position,
                        kind,
                    )
                ] = raw
    finally:
        if auditor is not None:
            auditor.restore()
    audit_rows = auditor.rows() if auditor is not None else []
    del pool
    torch.cuda.empty_cache()
    return output_rows, samples, audit_rows


def _cache_rows(
    prompts: Sequence[Mapping[str, object]],
    bf16: Mapping[tuple[str, int, int, str], bytes],
    fp8: Mapping[tuple[str, int, int, str], bytes],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        length = int(prompt["length"])
        for layer in SAMPLE_LAYERS:
            for position in sample_positions(length):
                for kind in ("k", "v"):
                    key = (prompt_id, layer, position, kind)
                    bf16_raw = bf16[key]
                    fp8_raw = fp8[key]
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "type": "cache_sample",
                            "prompt_id": prompt_id,
                            "prompt_length": length,
                            "layer": layer,
                            "position": position,
                            "kind": kind,
                            "shape": [NUM_KV_HEADS, HEAD_DIM],
                            "bf16_dtype": "bfloat16",
                            "bf16_base64": base64.b64encode(
                                bf16_raw
                            ).decode("ascii"),
                            "bf16_sha256": sha256_bytes(bf16_raw),
                            "fp8_dtype": "float8_e4m3fn",
                            "fp8_scale": FP8_SCALE,
                            "fp8_base64": base64.b64encode(
                                fp8_raw
                            ).decode("ascii"),
                            "fp8_sha256": sha256_bytes(fp8_raw),
                        }
                    )
    return rows


def _sanitize_failure(error: BaseException) -> dict[str, object]:
    message = " ".join(str(error).split())
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "failure",
        "error_type": type(error).__name__,
        "message": message[:2_000],
    }


def measure(
    *,
    model_path: Path,
    image_id: str,
) -> list[dict[str, object]]:
    import gc

    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.tokenizer import HFTokenizer
    from kairyu.models.loader import load_model

    rows: list[dict[str, object]] = []
    source_start: dict[str, object] = {}
    environment_start: dict[str, object] = {}
    checkpoint: dict[str, object] = {}
    model = None
    try:
        if _IMAGE_ID.fullmatch(image_id) is None:
            raise ValueError("--image-id must be sha256:<64 lowercase hex>")
        if os.environ.get("KAIRYU_ATTENTION_BACKEND") != "flashinfer":
            raise RuntimeError(
                "formal G4 E-KV requires explicit "
                "KAIRYU_ATTENTION_BACKEND=flashinfer"
            )
        source_start = _source_snapshot()
        if not _source_snapshot_valid(source_start):
            raise RuntimeError(
                "formal G4 E-KV requires a clean commit containing every "
                "bound source file"
            )
        checkpoint = _checkpoint_snapshot(model_path)
        if not _checkpoint_valid(checkpoint):
            raise RuntimeError(
                "--model-path is not the exact reviewed Qwen3-32B checkpoint"
            )
        backend = select_backend(probe(), device="cuda:0")
        environment_start = _environment_snapshot(
            backend_decision=backend.selection_decision
        )
        if not _environment_valid(environment_start):
            raise RuntimeError(
                "formal G4 E-KV requires one visible SM120 RTX PRO 6000 "
                "and explicit FlashInfer"
            )
        tokenizer = HFTokenizer(model_path)
        prompts = _prompt_rows(tokenizer)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "type": "run",
                "created_at": datetime.now(UTC).isoformat(
                    timespec="seconds"
                ),
                "config": formal_config(),
                "image_id": image_id,
                "source_start": source_start,
                "checkpoint": checkpoint,
                "environment_start": environment_start,
            }
        )
        rows.extend(prompts)

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        model, config, _generation = load_model(
            model_path,
            dtype=torch.bfloat16,
            attention_backend=backend,
            target_device=device,
        )
        model = model.to(device)
        if (
            config.architecture != "Qwen3ForCausalLM"
            or config.num_hidden_layers != NUM_LAYERS
            or config.num_key_value_heads != NUM_KV_HEADS
            or config.kv_cache_head_dim != HEAD_DIM
            or config.vocab_size != VOCAB_SIZE
        ):
            raise RuntimeError("loaded model geometry is not Qwen3-32B")

        bf16_outputs, bf16_samples, _ = _run_arm(
            model,
            config,
            tokenizer,
            prompts,
            arm="bf16",
            device=device,
        )
        rows.extend(bf16_outputs)
        fp8_outputs, fp8_samples, audits = _run_arm(
            model,
            config,
            tokenizer,
            prompts,
            arm="fp8_e4m3",
            device=device,
        )
        rows.extend(fp8_outputs)
        rows.extend(audits)
        rows.extend(_cache_rows(prompts, bf16_samples, fp8_samples))
    except BaseException as error:
        if not rows or rows[0].get("type") != "run":
            rows.insert(
                0,
                {
                    "schema_version": SCHEMA_VERSION,
                    "type": "run",
                    "created_at": datetime.now(UTC).isoformat(
                        timespec="seconds"
                    ),
                    "config": formal_config(),
                    "image_id": image_id,
                    "source_start": source_start,
                    "checkpoint": checkpoint,
                    "environment_start": environment_start,
                },
            )
        rows.append(_sanitize_failure(error))
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        source_end = _source_snapshot()
    except BaseException:
        source_end = {}
    try:
        environment_end = _environment_snapshot()
        if (
            isinstance(environment_start, dict)
            and environment_start.get("attention_backend") is not None
        ):
            environment_end["attention_backend"] = environment_start[
                "attention_backend"
            ]
    except BaseException:
        environment_end = {}
    counts = defaultdict(int)
    for row in rows:
        counts[str(row.get("type"))] += 1
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "type": "run_end",
            "source_end": source_end,
            "environment_end": environment_end,
            "row_counts_before_end": dict(sorted(counts.items())),
        }
    )
    return rows


def _decode_sample(
    row: Mapping[str, object],
) -> tuple[object, object]:
    import torch

    bf16_raw = base64.b64decode(
        str(row["bf16_base64"]), validate=True
    )
    fp8_raw = base64.b64decode(
        str(row["fp8_base64"]), validate=True
    )
    expected_values = NUM_KV_HEADS * HEAD_DIM
    if len(bf16_raw) != expected_values * 2:
        raise ValueError("cache sample BF16 byte length is invalid")
    if len(fp8_raw) != expected_values:
        raise ValueError("cache sample FP8 byte length is invalid")
    if sha256_bytes(bf16_raw) != row["bf16_sha256"]:
        raise ValueError("cache sample BF16 digest mismatch")
    if sha256_bytes(fp8_raw) != row["fp8_sha256"]:
        raise ValueError("cache sample FP8 digest mismatch")
    bf16 = torch.frombuffer(
        bytearray(bf16_raw), dtype=torch.bfloat16
    ).to(torch.float64)
    fp8 = (
        torch.frombuffer(bytearray(fp8_raw), dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .to(torch.float64)
    )
    return bf16, fp8


def _run_valid(row: object) -> bool:
    if not isinstance(row, dict) or set(row) != {
        "schema_version",
        "type",
        "created_at",
        "config",
        "image_id",
        "source_start",
        "checkpoint",
        "environment_start",
    }:
        return False
    created_at = row["created_at"]
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return False
    return (
        row["schema_version"] == SCHEMA_VERSION
        and row["type"] == "run"
        and created.tzinfo is not None
        and created.utcoffset() == UTC.utcoffset(created)
    )


def _run_end_valid(row: object) -> bool:
    if not isinstance(row, dict) or set(row) != {
        "schema_version",
        "type",
        "source_end",
        "environment_end",
        "row_counts_before_end",
    }:
        return False
    counts = row["row_counts_before_end"]
    return (
        row["schema_version"] == SCHEMA_VERSION
        and row["type"] == "run_end"
        and isinstance(counts, dict)
        and all(
            isinstance(kind, str)
            and bool(kind)
            and type(count) is int
            and count >= 0
            for kind, count in counts.items()
        )
    )


def _prompt_valid(row: object) -> bool:
    if not isinstance(row, dict) or set(row) != {
        "schema_version",
        "type",
        "prompt_id",
        "length",
        "piece_token_id",
        "piece_text",
        "text",
        "text_sha256",
        "token_ids",
        "token_ids_sha256",
    }:
        return False
    length = row["length"]
    token_id = row["piece_token_id"]
    piece = row["piece_text"]
    text = row["text"]
    token_ids = row["token_ids"]
    return (
        row["schema_version"] == SCHEMA_VERSION
        and row["type"] == "prompt"
        and type(length) is int
        and length in PROMPT_LENGTHS
        and row["prompt_id"] == f"long-{length}"
        and type(token_id) is int
        and 0 <= token_id < VOCAB_SIZE
        and isinstance(piece, str)
        and bool(piece)
        and isinstance(text, str)
        and text == piece * length
        and row["text_sha256"] == sha256_text(text)
        and isinstance(token_ids, list)
        and len(token_ids) == length
        and all(token == token_id for token in token_ids)
        and row["token_ids_sha256"] == sha256_json(token_ids)
    )


def _output_valid(row: object) -> bool:
    if not isinstance(row, dict) or set(row) != {
        "schema_version",
        "type",
        "prompt_id",
        "prompt_length",
        "arm",
        "kv_dtype",
        "k_scale",
        "v_scale",
        "output_token_ids",
        "output_token_ids_sha256",
        "selected_logprobs",
        "finish_reason",
        "output_text",
        "output_text_sha256",
    }:
        return False
    arm = row["arm"]
    expected_dtype = (
        "bfloat16" if arm == "bf16" else "float8_e4m3fn"
    )
    expected_scale = None if arm == "bf16" else FP8_SCALE
    tokens = row["output_token_ids"]
    logprobs = row["selected_logprobs"]
    output_text = row["output_text"]
    return (
        row["schema_version"] == SCHEMA_VERSION
        and row["type"] == "output"
        and arm in ARMS
        and type(row["prompt_length"]) is int
        and row["prompt_length"] in PROMPT_LENGTHS
        and row["prompt_id"] == f"long-{row['prompt_length']}"
        and row["kv_dtype"] == expected_dtype
        and row["k_scale"] == expected_scale
        and row["v_scale"] == expected_scale
        and (
            arm == "bf16"
            or (
                type(row["k_scale"]) is float
                and type(row["v_scale"]) is float
            )
        )
        and isinstance(tokens, list)
        and len(tokens) == OUTPUT_TOKENS
        and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in tokens)
        and row["output_token_ids_sha256"] == sha256_json(tokens)
        and isinstance(logprobs, list)
        and len(logprobs) == OUTPUT_TOKENS
        and all(type(value) is float and math.isfinite(value) for value in logprobs)
        and row["finish_reason"] == "length"
        and isinstance(output_text, str)
        and row["output_text_sha256"] == sha256_text(output_text)
    )


def _audit_valid(row: object) -> bool:
    if not isinstance(row, dict) or set(row) != {
        "schema_version",
        "type",
        "phase",
        "kind",
        "input_dtype",
        "storage_dtype",
        "scale",
        "calls",
        "values",
        "input_nonfinite",
        "input_overrange",
        "input_dtype_mismatch",
        "stored_nonfinite",
        "stored_byte_mismatch",
        "dequant_error_violation",
        "max_abs_input",
        "max_abs_error",
        "max_bound_excess",
    }:
        return False
    key = (row["phase"], row["kind"])
    expected = _expected_audit_counts().get(key)
    floats = (
        row["max_abs_input"],
        row["max_abs_error"],
        row["max_bound_excess"],
    )
    integers = (
        row["calls"],
        row["values"],
        row["input_nonfinite"],
        row["input_overrange"],
        row["input_dtype_mismatch"],
        row["stored_nonfinite"],
        row["stored_byte_mismatch"],
        row["dequant_error_violation"],
    )
    return (
        row["schema_version"] == SCHEMA_VERSION
        and row["type"] == "write_audit"
        and expected is not None
        and row["input_dtype"] == "bfloat16"
        and row["storage_dtype"] == "float8_e4m3fn"
        and type(row["scale"]) is float
        and row["scale"] == FP8_SCALE
        and all(type(value) is int and value >= 0 for value in integers)
        and row["calls"] == expected["calls"]
        and row["values"] == expected["values"]
        and row["input_nonfinite"] == 0
        and row["input_overrange"] == 0
        and row["input_dtype_mismatch"] == 0
        and row["stored_nonfinite"] == 0
        and row["stored_byte_mismatch"] == 0
        and row["dequant_error_violation"] == 0
        and all(type(value) is float and math.isfinite(value) for value in floats)
        and 0.0 <= row["max_abs_input"] <= FP8_MAX
        and row["max_abs_error"] >= 0.0
        and row["max_bound_excess"] == 0.0
    )


def _cache_row_valid(row: object) -> bool:
    if not isinstance(row, dict) or set(row) != {
        "schema_version",
        "type",
        "prompt_id",
        "prompt_length",
        "layer",
        "position",
        "kind",
        "shape",
        "bf16_dtype",
        "bf16_base64",
        "bf16_sha256",
        "fp8_dtype",
        "fp8_scale",
        "fp8_base64",
        "fp8_sha256",
    }:
        return False
    length = row["prompt_length"]
    if (
        row["schema_version"] != SCHEMA_VERSION
        or row["type"] != "cache_sample"
        or type(length) is not int
        or length not in PROMPT_LENGTHS
        or row["prompt_id"] != f"long-{length}"
        or type(row["layer"]) is not int
        or row["layer"] not in SAMPLE_LAYERS
        or type(row["position"]) is not int
        or row["position"] not in sample_positions(length)
        or row["kind"] not in ("k", "v")
        or row["shape"] != [NUM_KV_HEADS, HEAD_DIM]
        or row["bf16_dtype"] != "bfloat16"
        or row["fp8_dtype"] != "float8_e4m3fn"
        or type(row["fp8_scale"]) is not float
        or row["fp8_scale"] != FP8_SCALE
        or not isinstance(row["bf16_base64"], str)
        or not isinstance(row["fp8_base64"], str)
        or not isinstance(row["bf16_sha256"], str)
        or _HEX64.fullmatch(row["bf16_sha256"]) is None
        or not isinstance(row["fp8_sha256"], str)
        or _HEX64.fullmatch(row["fp8_sha256"]) is None
    ):
        return False
    try:
        bf16, fp8 = _decode_sample(row)
    except (ValueError, TypeError):
        return False
    return bool(bf16.isfinite().all() and fp8.isfinite().all())


def _row_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["type"])] += 1
    return dict(sorted(counts.items()))


def recompute_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    if _HEX64.fullmatch(raw_sha256) is None:
        raise ValueError("raw_sha256 must be 64 lowercase hex")
    allowed_types = {
        "run",
        "prompt",
        "output",
        "write_audit",
        "cache_sample",
        "failure",
        "run_end",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"raw row {index} is not an object")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"raw row {index} has the wrong schema")
        if row.get("type") not in allowed_types:
            raise ValueError(f"raw row {index} has unknown type")

    by_type: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["type"])].append(row)
    run = by_type["run"][0] if len(by_type["run"]) == 1 else {}
    run_end = (
        by_type["run_end"][0] if len(by_type["run_end"]) == 1 else {}
    )
    checks: dict[str, bool] = {}
    checks["one_run_and_one_run_end"] = (
        len(by_type["run"]) == 1
        and len(by_type["run_end"]) == 1
        and _run_valid(run)
        and _run_end_valid(run_end)
    )
    checks["formal_config_exact"] = (
        isinstance(run, dict)
        and run.get("config") == formal_config()
        and isinstance(run.get("image_id"), str)
        and _IMAGE_ID.fullmatch(str(run.get("image_id"))) is not None
    )
    checks["clean_source_stable"] = (
        _source_snapshot_valid(run.get("source_start"))
        and _source_snapshot_valid(run_end.get("source_end"))
        and run.get("source_start") == run_end.get("source_end")
    )
    checks["full_checkpoint_identity"] = _checkpoint_valid(
        run.get("checkpoint")
    )
    checks["sm120_flashinfer_environment_stable"] = (
        _environment_valid(run.get("environment_start"))
        and _environment_valid(run_end.get("environment_end"))
        and _stable_environment_identity(run.get("environment_start"))
        == _stable_environment_identity(run_end.get("environment_end"))
    )
    checks["no_runtime_failure"] = len(by_type["failure"]) == 0
    checks["run_end_counts_exact"] = (
        isinstance(run_end, dict)
        and run_end.get("row_counts_before_end")
        == _row_counts(
            [row for row in rows if row.get("type") != "run_end"]
        )
    )

    prompts = by_type["prompt"]
    checks["exact_raw_prompts"] = (
        len(prompts) == len(PROMPT_LENGTHS)
        and all(_prompt_valid(row) for row in prompts)
        and {row["length"] for row in prompts} == set(PROMPT_LENGTHS)
    )

    outputs = by_type["output"]
    output_by_key: dict[tuple[int, str], Mapping[str, object]] = {}
    duplicate_output = False
    for row in outputs:
        key = (row.get("prompt_length"), row.get("arm"))
        if key in output_by_key:
            duplicate_output = True
        output_by_key[key] = row
    exact_output_keys = {
        (length, arm) for length in PROMPT_LENGTHS for arm in ARMS
    }
    checks["complete_output_rows"] = (
        not duplicate_output
        and len(outputs) == len(exact_output_keys)
        and set(output_by_key) == exact_output_keys
        and all(_output_valid(row) for row in outputs)
    )
    output_metrics: list[dict[str, object]] = []
    exact_tokens = checks["complete_output_rows"]
    finite_deltas = checks["complete_output_rows"]
    within_delta = checks["complete_output_rows"]
    if checks["complete_output_rows"]:
        for length in PROMPT_LENGTHS:
            bf16 = output_by_key[(length, "bf16")]
            fp8 = output_by_key[(length, "fp8_e4m3")]
            tokens_equal = (
                bf16["output_token_ids"] == fp8["output_token_ids"]
                and bf16["finish_reason"] == fp8["finish_reason"]
                and bf16["output_text"] == fp8["output_text"]
            )
            common_prefix = 0
            for left, right in zip(
                bf16["output_token_ids"],
                fp8["output_token_ids"],
                strict=True,
            ):
                if left != right:
                    break
                common_prefix += 1
            deltas = [
                abs(float(left) - float(right))
                for left, right in zip(
                    bf16["selected_logprobs"][:common_prefix],
                    fp8["selected_logprobs"][:common_prefix],
                    strict=True,
                )
            ]
            maximum = max(deltas, default=math.inf)
            exact_tokens &= tokens_equal
            finite_deltas &= all(math.isfinite(value) for value in deltas)
            within_delta &= maximum <= LOGPROB_ABS_DELTA_MAX
            output_metrics.append(
                {
                    "prompt_length": length,
                    "tokens_exact": tokens_equal,
                    "positions": OUTPUT_TOKENS,
                    "common_prefix_tokens": common_prefix,
                    "compared_logprob_positions": len(deltas),
                    "max_selected_logprob_abs_delta": maximum,
                }
            )
    checks["exact_output_tokens_and_stopping"] = bool(exact_tokens)
    checks["selected_logprobs_finite"] = bool(finite_deltas)
    checks["selected_logprob_delta_at_most_0_25"] = bool(within_delta)

    audits = by_type["write_audit"]
    audit_keys = {
        (row.get("phase"), row.get("kind")) for row in audits
    }
    checks["all_fp8_writes_audited"] = (
        len(audits) == len(_expected_audit_counts())
        and audit_keys == set(_expected_audit_counts())
        and all(_audit_valid(row) for row in audits)
    )

    cache_rows = by_type["cache_sample"]
    expected_cache_keys = {
        (length, layer, position, kind)
        for length in PROMPT_LENGTHS
        for layer in SAMPLE_LAYERS
        for position in sample_positions(length)
        for kind in ("k", "v")
    }
    cache_by_key: dict[
        tuple[int, int, int, str], Mapping[str, object]
    ] = {}
    duplicate_cache = False
    for row in cache_rows:
        key = (
            row.get("prompt_length"),
            row.get("layer"),
            row.get("position"),
            row.get("kind"),
        )
        if key in cache_by_key:
            duplicate_cache = True
        cache_by_key[key] = row
    checks["complete_cache_samples"] = (
        not duplicate_cache
        and len(cache_rows) == len(expected_cache_keys)
        and set(cache_by_key) == expected_cache_keys
        and all(_cache_row_valid(row) for row in cache_rows)
    )
    cache_metrics: list[dict[str, object]] = []
    nrmse_ok = checks["complete_cache_samples"]
    cosine_ok = checks["complete_cache_samples"]
    if checks["complete_cache_samples"]:
        import torch

        grouped: dict[
            tuple[int, int, str], list[tuple[object, object]]
        ] = defaultdict(list)
        for key in sorted(cache_by_key):
            length, layer, _position, kind = key
            grouped[(length, layer, kind)].append(
                _decode_sample(cache_by_key[key])
            )
        for (length, layer, kind), pairs in sorted(grouped.items()):
            reference = torch.cat([pair[0] for pair in pairs])
            candidate = torch.cat([pair[1] for pair in pairs])
            delta = candidate - reference
            reference_rms = float(
                torch.sqrt(torch.mean(reference.square())).item()
            )
            delta_rms = float(
                torch.sqrt(torch.mean(delta.square())).item()
            )
            nrmse = (
                delta_rms / reference_rms
                if reference_rms > 0.0
                else (0.0 if delta_rms == 0.0 else math.inf)
            )
            reference_norm = float(torch.linalg.vector_norm(reference).item())
            candidate_norm = float(torch.linalg.vector_norm(candidate).item())
            if reference_norm == 0.0 or candidate_norm == 0.0:
                cosine = (
                    1.0
                    if reference_norm == candidate_norm == 0.0
                    else 0.0
                )
            else:
                cosine = float(
                    torch.dot(reference, candidate).item()
                    / (reference_norm * candidate_norm)
                )
            nrmse_ok &= math.isfinite(nrmse) and nrmse <= CACHE_NRMSE_MAX
            cosine_ok &= (
                math.isfinite(cosine) and cosine >= CACHE_COSINE_MIN
            )
            cache_metrics.append(
                {
                    "prompt_length": length,
                    "layer": layer,
                    "kind": kind,
                    "values": int(reference.numel()),
                    "nrmse": nrmse,
                    "cosine": cosine,
                }
            )
    checks["cache_sample_nrmse_at_most_0_05"] = bool(nrmse_ok)
    checks["cache_sample_cosine_at_least_0_99"] = bool(cosine_ok)

    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "config": formal_config(),
        "raw": {
            "name": RAW_NAME,
            "sha256": raw_sha256,
            "rows": len(rows),
        },
        "checks": checks,
        "metrics": {
            "outputs": output_metrics,
            "cache_samples": cache_metrics,
            "write_audits": audits,
        },
        "failures": by_type["failure"],
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


def _artifact_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / RAW_NAME, path / MANIFEST_NAME
    if path.name == RAW_NAME:
        return path, path.with_name(MANIFEST_NAME)
    if path.name == MANIFEST_NAME:
        return path.with_name(RAW_NAME), path
    raise ValueError(
        f"artifact must be a directory, {RAW_NAME}, or {MANIFEST_NAME}"
    )


def _read_raw(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise ValueError(f"{path}:{number}: blank JSONL row")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: row is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def write_artifact(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    raw_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = recompute_manifest(
        rows, raw_sha256=sha256_file(raw_path)
    )
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_artifact(path: Path) -> dict[str, object]:
    raw_path, manifest_path = _artifact_paths(path)
    rows = _read_raw(raw_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replayed = recompute_manifest(
        rows, raw_sha256=sha256_file(raw_path)
    )
    if manifest != replayed:
        raise ValueError("manifest differs from independent raw replay")
    return replayed


def replay_artifact(path: Path) -> dict[str, object]:
    raw_path, _manifest_path = _artifact_paths(path)
    rows = _read_raw(raw_path)
    return recompute_manifest(
        rows, raw_sha256=sha256_file(raw_path)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_parser = subparsers.add_parser(
        "measure", help="run the real one-GPU BF16/FP8 bake"
    )
    measure_parser.add_argument("--model-path", type=Path, required=True)
    measure_parser.add_argument("--image-id", required=True)
    measure_parser.add_argument("--output-dir", type=Path, required=True)
    measure_parser.add_argument("--assert-gate", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="verify manifest and independently replay raw JSONL"
    )
    verify_parser.add_argument("--artifact", type=Path, required=True)
    verify_parser.add_argument("--assert-gate", action="store_true")

    replay_parser = subparsers.add_parser(
        "replay", help="recompute a verdict from raw JSONL only"
    )
    replay_parser.add_argument("--artifact", type=Path, required=True)
    replay_parser.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "measure":
        rows = measure(
            model_path=args.model_path.resolve(),
            image_id=args.image_id,
        )
        result = write_artifact(args.output_dir.resolve(), rows)
    elif args.command == "verify":
        result = verify_artifact(args.artifact.resolve())
    else:
        result = replay_artifact(args.artifact.resolve())
    print(canonical_json(result))
    return 1 if args.assert_gate and result["passed"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
