#!/usr/bin/env python3
"""Fail-closed native A12 batch/cache/chunk invariance evidence.

``run-native`` executes the fixed Qwen3-32B TP8 matrix.  ``verify`` compares a
retained manifest with an independent replay of its canonical raw JSONL;
``replay`` ignores the retained manifest and derives the verdict from raw
evidence alone.  No verification command constructs an engine.

Formal evidence is direct-path only and the *initial* interpreter must already
be isolated with ``-I -B``.  Engine construction, torch, tokenizer resolution,
and CUDA-facing imports remain inside the native execution path.
"""

# ruff: noqa: E402

from __future__ import annotations

# A late re-exec would happen after normal Python may already have evaluated a
# hostile sitecustomize.  Evidence therefore refuses a non-isolated initial
# interpreter.  Help is non-evidence and may re-exec solely for inventory
# compatibility.
_bootstrap_sys = __import__("sys")
_bootstrap_help = any(
    argument in {"-h", "--help"} for argument in _bootstrap_sys.argv[1:]
)
if __name__ == "__main__":
    if __spec__ is not None and not _bootstrap_help:
        raise RuntimeError(
            "A12 evidence commands require the direct checkout path, not python -m"
        )
    _bootstrap_posix = __import__("posix")
    if not _bootstrap_sys.flags.isolated or not _bootstrap_sys.dont_write_bytecode:
        if not _bootstrap_help:
            raise RuntimeError(
                "A12 evidence commands must start as: python -I -B "
                "bench/batch_invariance_bench.py ..."
            )
        _bootstrap_posix.execv(
            _bootstrap_sys.executable,
            [
                _bootstrap_sys.executable,
                "-I",
                "-B",
                __file__,
                *_bootstrap_sys.argv[1:],
            ],
        )

import argparse
import asyncio
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_TEXT = str(_REPO_ROOT)
_IMPORT_CACHE_DIRECTORY = None
if __name__ == "__main__" and __spec__ is None:
    _IMPORT_CACHE_DIRECTORY = tempfile.TemporaryDirectory(
        prefix="kairyu-a12-import-cache-",
        dir="/tmp",
    )
    sys.pycache_prefix = _IMPORT_CACHE_DIRECTORY.name

_IMPORT_ARTIFACT_SUFFIXES = (".py", ".pyc", ".pth", ".so")
_GIT_OTHER_PATHSPEC = (".", ":(glob,exclude).venv/**")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_other_paths(*, ignored: bool) -> tuple[str, ...]:
    command = ["git", "ls-files", "--others", "-z"]
    if ignored:
        command.append("--ignored")
    command.extend(("--exclude-standard", "--", *_GIT_OTHER_PATHSPEC))
    try:
        raw = subprocess.check_output(
            command,
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot enumerate non-versioned source artifacts") from error
    return tuple(os.fsdecode(item) for item in raw.split(b"\0") if item)


def _safe_generated_source_path(path: str) -> bool:
    parts = PurePosixPath(path.rstrip("/")).parts
    return bool(parts) and parts[0] == ".venv"


def _could_be_repo_import(path: str) -> bool:
    parts = PurePosixPath(path.rstrip("/")).parts
    if not parts or _safe_generated_source_path(path):
        return False
    if path.endswith("/"):
        return all(part.isidentifier() for part in parts)
    return parts[-1].endswith(_IMPORT_ARTIFACT_SUFFIXES)


def _assert_no_import_shadows() -> None:
    offenders: set[str] = set()
    for ignored in (False, True):
        for path in _git_other_paths(ignored=ignored):
            if _safe_generated_source_path(path):
                continue
            working = _REPO_ROOT / path
            if working.is_symlink() or _could_be_repo_import(path):
                offenders.add(path)
    _require(
        not offenders,
        "native evidence rejects untracked or ignored import artifacts: "
        + ", ".join(sorted(offenders)),
    )


def _assert_tracked_tree_clean_before_repo_import() -> None:
    try:
        completed = subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--"),
            cwd=_REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise ValueError("cannot inspect tracked source before repository import") from error
    _require(
        completed.returncode == 0,
        "native evidence requires a clean tracked source tree before repository import",
    )


_FORMAL_DIRECT_EXECUTION = (
    __name__ == "__main__" and __spec__ is None and not _bootstrap_help
)
if _FORMAL_DIRECT_EXECUTION:
    _assert_tracked_tree_clean_before_repo_import()
    _assert_no_import_shadows()

sys.path[:] = [entry for entry in sys.path if entry != _REPO_ROOT_TEXT]
sys.path.insert(0, _REPO_ROOT_TEXT)

from kairyu.bench import batch_invariance as gate  # noqa: E402

for _module_name, _relative_origin in (
    ("kairyu", "kairyu/__init__.py"),
    ("kairyu.bench", "kairyu/bench/__init__.py"),
    ("kairyu.bench.batch_invariance", "kairyu/bench/batch_invariance.py"),
):
    _module = sys.modules.get(_module_name)
    _origin = getattr(_module, "__file__", None)
    try:
        _resolved_origin = Path(_origin).resolve(strict=True)
        _expected_origin = (_REPO_ROOT / _relative_origin).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            f"cannot establish checkout origin for {_module_name}"
        ) from error
    if _resolved_origin != _expected_origin:
        raise RuntimeError(
            f"refusing non-checkout {_module_name} origin: {_resolved_origin}"
        )

_CHECKOUT_MODULE_PREFIXES = ("bench", "kairyu")
_WEIGHT_INDEX = "model.safetensors.index.json"


def _same_json_value(left: object, right: object) -> bool:
    try:
        return gate.canonical_json(left) == gate.canonical_json(right)
    except (TypeError, ValueError):
        return False


def _read_exact_bytes(path: Path, *, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description}: {path}") from error


def _strict_json_object(path: Path) -> dict[str, object]:
    raw = _read_exact_bytes(path, description="JSON file")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{path}: non-finite JSON number {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse strict JSON: {path}") from error
    _require(isinstance(parsed, dict), f"{path}: JSON root must be an object")
    return parsed


def _resolve_existing_path(
    path: Path,
    *,
    description: str,
    directory: bool,
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cannot resolve {description}: {path}") from error
    _require(
        resolved.is_dir() if directory else resolved.is_file(),
        f"{description} is not a {'directory' if directory else 'file'}: {resolved}",
    )
    return resolved


def _head_identical_source(path: str, *, description: str) -> str:
    working_path = _REPO_ROOT / path
    _require(working_path.is_file(), f"{description} is missing: {path}")
    tracked = subprocess.run(
        ("git", "ls-files", "--error-unmatch", "--", path),
        cwd=_REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _require(tracked.returncode == 0, f"{description} is untracked: {path}")
    try:
        committed = subprocess.check_output(
            ("git", "show", f"HEAD:{path}"),
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot read {description} from HEAD: {path}") from error
    current = _read_exact_bytes(working_path, description=description)
    _require(current == committed, f"{description} bytes differ from HEAD: {path}")
    return gate.sha256_bytes(current)


def _assert_loaded_repo_modules_head_identical() -> None:
    repo_root = _REPO_ROOT.resolve(strict=True)
    checked: set[str] = set()
    for module_name, module in tuple(sys.modules.items()):
        checkout_prefix = next(
            (
                prefix
                for prefix in _CHECKOUT_MODULE_PREFIXES
                if module_name == prefix or module_name.startswith(f"{prefix}.")
            ),
            None,
        )
        checkout_owned = checkout_prefix is not None
        module_spec = getattr(module, "__spec__", None)
        origins = {
            getattr(module, "__file__", None),
            getattr(module_spec, "origin", None),
        }
        concrete = {
            origin
            for origin in origins
            if isinstance(origin, str) and origin not in {"built-in", "frozen"}
        }
        _require(
            not checkout_owned or bool(concrete),
            f"checkout-owned module has no concrete origin: {module_name}",
        )
        for origin in concrete:
            candidate = Path(origin)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            candidate = candidate.absolute()
            try:
                lexical_relative = candidate.relative_to(repo_root)
            except ValueError:
                lexical_relative = None
            resolved = candidate.resolve(strict=False)
            try:
                resolved_relative = resolved.relative_to(repo_root)
            except ValueError:
                resolved_relative = None
            _require(
                not checkout_owned
                or lexical_relative is not None
                or resolved_relative is not None,
                f"checkout-owned module loaded outside the repository: {module_name}",
            )
            if lexical_relative is not None and resolved_relative is None:
                raise ValueError(
                    f"loaded repo module escapes through a symlink: {module_name}"
                )
            relative = resolved_relative or lexical_relative
            if relative is None:
                continue
            relative_name = relative.as_posix()
            _require(
                not checkout_owned
                or (
                    bool(relative.parts)
                    and relative.parts[0] == checkout_prefix
                    and not _safe_generated_source_path(relative_name)
                ),
                f"checkout-owned module is outside its tracked package: {module_name}",
            )
            if _safe_generated_source_path(relative_name) or relative_name in checked:
                continue
            _head_identical_source(
                relative_name,
                description=f"loaded repo module {module_name}",
            )
            checked.add(relative_name)


def _source_snapshot() -> dict[str, object]:
    try:
        git_head = subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        clean = (
            subprocess.run(
                ("git", "diff", "--quiet", "HEAD", "--"),
                cwd=_REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot establish native source identity") from error
    _require(
        len(git_head) == 40
        and all(character in "0123456789abcdef" for character in git_head),
        "git HEAD is not a full lowercase commit identity",
    )
    _require(clean, "native evidence requires a clean tracked source tree")
    _assert_no_import_shadows()
    file_sha256 = {
        path: _head_identical_source(path, description="required source path")
        for path in gate.REQUIRED_SOURCE_PATHS
    }
    _assert_loaded_repo_modules_head_identical()
    return {
        "git_head": git_head,
        "tracked_tree_clean": True,
        "file_sha256": file_sha256,
        "files_sha256": gate.sha256_json(file_sha256),
        "python_isolated": sys.flags.isolated == 1,
        "bytecode_disabled": sys.dont_write_bytecode is True,
    }


def _live_checkpoint(model_path: Path) -> dict[str, object]:
    metadata_names = tuple(gate.EXPECTED_METADATA_SHA256)
    for name in (*metadata_names, _WEIGHT_INDEX):
        _require((model_path / name).is_file(), f"checkpoint lacks {name}")
    config = _strict_json_object(model_path / "config.json")
    architecture = {
        name: config[name]
        for name in (
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        )
    }
    index = _strict_json_object(model_path / _WEIGHT_INDEX)
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, Mapping) and weight_map, "weight map is missing")
    indexed_names = set(weight_map.values())
    _require(
        all(isinstance(name, str) and Path(name).name == name for name in indexed_names),
        "weight map contains an invalid shard name",
    )
    shard_paths = sorted(model_path.glob("*.safetensors"), key=lambda path: path.name)
    _require(
        {path.name for path in shard_paths} == indexed_names,
        "weight index and checkpoint shard set differ",
    )
    with ThreadPoolExecutor(max_workers=min(8, len(shard_paths))) as pool:
        shard_digests = list(pool.map(gate.sha256_file, shard_paths))
    weights = [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}
        for path, digest in zip(shard_paths, shard_digests, strict=True)
    ]
    pinned_rollup = hashlib.sha256(
        json.dumps(weights, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint = {
        "model": gate.MODEL,
        "revision": gate.MODEL_REVISION,
        "architecture": architecture,
        "weights": weights,
        "weights_sha256": gate.sha256_json(weights),
        "pinned_weights_rollup": pinned_rollup,
        "metadata_sha256": {
            name: gate.sha256_file(model_path / name) for name in metadata_names
        },
        "weight_index_sha256": gate.sha256_file(model_path / _WEIGHT_INDEX),
        "model_config_sha256": gate.sha256_json(config),
    }
    _require(
        _same_json_value(checkpoint["architecture"], gate.EXPECTED_ARCHITECTURE),
        "live checkpoint architecture differs from pinned Qwen3-32B",
    )
    _require(
        _same_json_value(checkpoint["weights"], list(gate.EXPECTED_WEIGHTS))
        and checkpoint["weights_sha256"] == gate.EXPECTED_WEIGHTS_SHA256
        and checkpoint["pinned_weights_rollup"] == gate.EXPECTED_WEIGHTS_ROLLUP,
        "live checkpoint weight inventory differs from pinned Qwen3-32B",
    )
    _require(
        _same_json_value(
            checkpoint["metadata_sha256"], gate.EXPECTED_METADATA_SHA256
        )
        and checkpoint["weight_index_sha256"]
        == gate.EXPECTED_WEIGHT_INDEX_SHA256
        and checkpoint["model_config_sha256"]
        == gate.EXPECTED_MODEL_CONFIG_SHA256,
        "live checkpoint metadata differs from pinned Qwen3-32B",
    )
    return checkpoint


def _command_csv(command: Sequence[str], *, description: str) -> list[list[str]]:
    try:
        completed = subprocess.run(
            tuple(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError(f"cannot execute {description}") from error
    _require(
        completed.returncode == 0,
        f"cannot collect {description}: {completed.stderr.strip()}",
    )
    return [
        [field.strip() for field in row]
        for row in csv.reader(completed.stdout.splitlines())
        if row and any(field.strip() for field in row)
    ]


def _assert_gpu_job_exclusive() -> None:
    rows = _command_csv(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        description="active NVIDIA compute-process inventory",
    )
    foreign: list[str] = []
    for row in rows:
        _require(len(row) == 2, "NVIDIA compute-process inventory is malformed")
        try:
            process_id = int(row[1])
        except ValueError as error:
            raise ValueError("NVIDIA compute-process PID is malformed") from error
        if process_id != os.getpid():
            foreign.append(f"{row[0]}:{process_id}")
    _require(
        not foreign,
        "formal A12 run requires exclusive GPUs; active foreign jobs: "
        + ", ".join(foreign),
    )


def _hardware() -> dict[str, object]:
    """Bind visible rank order to NVIDIA identities and exact software versions."""

    # Probe before constructing an engine.  A context owned by this process is
    # acceptable on a repeated stability probe, but another PID is never folded
    # into formal evidence as an "exclusive" allocation.
    _assert_gpu_job_exclusive()
    inventory_rows = _command_csv(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,pci.bus_id,driver_version",
            "--format=csv,noheader,nounits",
        ),
        description="NVIDIA GPU identity inventory",
    )
    inventory: dict[str, dict[str, str]] = {}
    for row in inventory_rows:
        _require(len(row) == 4, "NVIDIA GPU identity inventory is malformed")
        name, gpu_uuid, pci_bus_id, driver = row
        key = gpu_uuid.lower()
        _require(key not in inventory, "NVIDIA GPU inventory repeats a UUID")
        inventory[key] = {
            "name": name,
            "uuid": gpu_uuid,
            "pci_bus_id": pci_bus_id,
            "driver_version": driver,
        }

    import torch

    _require(torch.cuda.is_available(), "CUDA is unavailable")
    count = torch.cuda.device_count()
    _require(
        count == gate.TENSOR_PARALLEL_SIZE,
        "formal A12 run requires exactly eight visible CUDA devices",
    )
    devices: list[dict[str, object]] = []
    drivers: set[str] = set()
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        raw_uuid = str(properties.uuid)
        lookup_uuid = raw_uuid if raw_uuid.lower().startswith("gpu-") else f"GPU-{raw_uuid}"
        smi = inventory.get(lookup_uuid.lower())
        _require(smi is not None, f"visible CUDA GPU {index} is absent from nvidia-smi")
        assert smi is not None
        _require(
            properties.name == smi["name"],
            f"visible CUDA GPU {index} name differs from nvidia-smi",
        )
        drivers.add(smi["driver_version"])
        devices.append(
            {
                "rank": index,
                "visible_index": index,
                "name": properties.name,
                "uuid": smi["uuid"],
                "pci_bus_id": smi["pci_bus_id"],
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
            }
        )
    _require(len(drivers) == 1, "visible CUDA GPUs report different driver versions")
    nccl = torch.cuda.nccl.version()
    _require(
        isinstance(nccl, tuple) and nccl and all(type(item) is int for item in nccl),
        "PyTorch exposes no exact NCCL version",
    )
    try:
        flashinfer_version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("flashinfer-python distribution is not installed") from error
    _require(isinstance(torch.version.cuda, str) and torch.version.cuda, "CUDA version is missing")
    return {
        "gpu_count": count,
        "gpu_exclusive": True,
        "gpus": devices,
        "software": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "driver_version": next(iter(drivers)),
            "nccl_version": ".".join(str(item) for item in nccl),
            "flashinfer_version": flashinfer_version,
        },
    }


def _json_safe(value: object, *, description: str = "runtime value") -> object:
    """Convert reviewed runtime containers without stringifying unknown types."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value), description=description)
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            _require(type(key) is str, f"{description} has a non-string key")
            result[key] = _json_safe(item, description=description)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, description=description) for item in value]
    raise ValueError(
        f"{description} contains unsupported {type(value).__name__}"
    )


class _ScheduleObserver:
    """Snapshot scheduler decisions without altering the returned plan."""

    def __init__(self, scheduler: object) -> None:
        original = getattr(scheduler, "schedule", None)
        _require(callable(original), "native scheduler has no schedule method")
        self._scheduler = scheduler
        self._original = original
        self._lock = threading.Lock()
        self._active: str | None = None
        self._steps: list[dict[str, object]] = []

        def observed() -> object:
            plan = self._original()
            with self._lock:
                phase = self._active
                if phase is not None:
                    scheduled = getattr(plan, "scheduled", None)
                    _require(
                        isinstance(scheduled, (tuple, list)),
                        "native scheduler plan has malformed scheduled rows",
                    )
                    self._steps.append(
                        {
                            "step_index": len(self._steps),
                            "kind": (
                                "prefill"
                                if scheduled and all(chunk.is_prefill for chunk in scheduled)
                                else "decode"
                                if scheduled
                                and all(not chunk.is_prefill for chunk in scheduled)
                                else "mixed"
                            ),
                            "request_ids": [chunk.request_id for chunk in scheduled],
                            "num_tokens": [chunk.num_tokens for chunk in scheduled],
                        }
                    )
            return plan

        scheduler.schedule = observed

    def begin(self, phase: str) -> None:
        with self._lock:
            _require(self._active is None, "schedule observer phase is already active")
            _require(bool(phase), "schedule observer phase is empty")
            self._active = phase
            self._steps = []

    def finish(self, phase: str) -> list[dict[str, object]]:
        with self._lock:
            _require(self._active == phase, "schedule observer phase changed")
            steps = [dict(step) for step in self._steps]
            self._active = None
            self._steps = []
        return steps

    def close(self) -> None:
        with self._lock:
            _require(self._active is None, "cannot close an active schedule observer")
        self._scheduler.schedule = self._original


def _parallel_runner(backend: object) -> object:
    loop = getattr(backend, "_loop", None)
    launcher = getattr(loop, "tp_launcher", None)
    runner = getattr(launcher, "runner", None)
    _require(runner is not None, "native TP runtime exposes no rank-complete runner")
    return runner


def _reset_structural_stats(runner: object) -> None:
    prefill = getattr(runner, "prefill_execution_stats", None)
    decode = getattr(runner, "verification_execution_stats", None)
    _require(callable(prefill), "native TP runner exposes no prefill stats")
    _require(callable(decode), "native TP runner exposes no decode/graph stats")
    prefill(reset=True)
    decode(reset=True)


def _collect_structure(
    runner: object,
    scheduler_steps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    prefill = getattr(runner, "prefill_execution_stats", None)
    decode = getattr(runner, "verification_execution_stats", None)
    _require(callable(prefill) and callable(decode), "native structural stats disappeared")
    return {
        "scheduler_steps": _json_safe(
            list(scheduler_steps), description="scheduler steps"
        ),
        "prefill_rank_stats": _json_safe(
            prefill(reset=False), description="all-rank prefill stats"
        ),
        # This existing all-rank probe includes general graph dispatch and
        # FlashInfer decode counters even when speculative request counters are
        # all zero.  A12 does not reinterpret those zero verification counters.
        "decode_rank_stats": _json_safe(
            decode(reset=False), description="all-rank decode stats"
        ),
    }


def _cache_probe(cache: object, token_ids: tuple[int, ...]) -> int:
    probe = getattr(cache, "peek_cached_tokens", None)
    _require(callable(probe), "native radix cache exposes no non-mutating probe")
    value = probe(token_ids)
    _require(
        type(value) is int and 0 <= value <= len(token_ids),
        "native radix cache returned an invalid target-token count",
    )
    return value


def _prompt_descriptor(text: str, token_ids: Sequence[int]) -> dict[str, object]:
    _require(type(text) is str and bool(text), "prompt text is empty")
    copied = list(token_ids)
    _require(
        copied and all(type(token_id) is int for token_id in copied),
        "prompt token IDs are invalid",
    )
    _require(
        all(0 <= token_id < gate.TOKENIZER_VOCAB_SIZE for token_id in copied),
        "prompt token ID is outside the tokenizer vocabulary",
    )
    return {
        "text": text,
        "text_sha256": gate.sha256_bytes(text.encode("utf-8")),
        "token_ids": copied,
        "token_count": len(copied),
        "token_ids_sha256": gate.sha256_json(copied),
    }


def _token_pieces(tokenizer: object, token_ids: Sequence[int]) -> list[str]:
    vocab_getter = getattr(tokenizer, "vocab", None)
    _require(callable(vocab_getter), "native tokenizer exposes no raw vocabulary")
    vocab = vocab_getter()
    _require(
        isinstance(vocab, (tuple, list))
        and len(vocab) == gate.TOKENIZER_VOCAB_SIZE,
        "native tokenizer vocabulary size differs from the pinned ID domain",
    )
    pieces: list[str] = []
    for token_id in token_ids:
        _require(
            type(token_id) is int and 0 <= token_id < len(vocab),
            "native output token is outside the tokenizer vocabulary",
        )
        piece = vocab[token_id]
        _require(type(piece) is str and bool(piece), "native token piece is invalid")
        pieces.append(piece)
    return pieces


def _usage_dict(value: object, *, prompt_tokens: int) -> dict[str, int]:
    _require(value is not None, "native result exposes no usage accounting")
    result = {
        "prompt_tokens": getattr(value, "prompt_tokens", None),
        "completion_tokens": getattr(value, "completion_tokens", None),
        "cached_tokens": getattr(value, "cached_tokens", None),
    }
    for name, item in result.items():
        _require(type(item) is int and item >= 0, f"native usage {name} is invalid")
    return result  # type: ignore[return-value]


async def _request_observation(
    *,
    backend: object,
    tokenizer: object,
    request_factory: Any,
    scenario: str,
    runtime_id: str,
    runtime_nonce: str,
    request_id: str,
    role: str,
    cohort_index: int,
    cohort_size: int,
    prompt: Mapping[str, object],
    sampling: Mapping[str, object],
    cache_probe_before: int,
) -> dict[str, object]:
    prompt_ids_value = prompt.get("token_ids")
    _require(
        isinstance(prompt_ids_value, list)
        and all(type(token_id) is int for token_id in prompt_ids_value),
        "request prompt descriptor has invalid token IDs",
    )
    prompt_ids = tuple(prompt_ids_value)
    error_type: str | None = None
    try:
        result = await backend.generate(
            request_factory(request_id, prompt_ids, sampling)
        )
        _require(result.finished is True, "native request did not finish")
        _require(
            len(result.completions) == 1,
            "native request did not return exactly one completion",
        )
        processed = result.prompt_token_ids
        _require(
            isinstance(processed, (tuple, list))
            and all(type(token_id) is int for token_id in processed)
            and tuple(processed) == prompt_ids,
            "native backend processed different prompt token IDs",
        )
        completion = result.completions[0]
        native_ids = completion.token_ids
        _require(
            isinstance(native_ids, (tuple, list))
            and all(type(token_id) is int for token_id in native_ids),
            "native output token IDs have invalid types",
        )
        token_ids = list(native_ids)
        response = {
            "status": "success",
            "token_ids": token_ids,
            "token_pieces": _token_pieces(tokenizer, token_ids),
            "text": completion.text,
            "finish_reason": completion.finish_reason,
            "usage": _usage_dict(result.usage, prompt_tokens=len(prompt_ids)),
        }
    except Exception as error:  # retain one explicit failed observation
        error_type = type(error).__name__
        response = {
            "status": "error",
            "token_ids": [],
            "token_pieces": [],
            "text": "",
            "finish_reason": None,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": 0,
                "cached_tokens": 0,
            },
        }
    return {
        "scenario": scenario,
        "runtime_id": runtime_id,
        "runtime_nonce": runtime_nonce,
        "request_id": request_id,
        "role": role,
        "cohort_index": cohort_index,
        "cohort_size": cohort_size,
        "prompt": dict(prompt),
        "cache": {
            "probe_before": cache_probe_before,
            "native_cached_tokens": response["usage"]["cached_tokens"],
        },
        "sampling": dict(sampling),
        "response": response,
        "error_type": error_type,
        "retry_count": 0,
    }


def _fixed_prompt_descriptors(tokenizer: object) -> dict[str, object]:
    """Bind runtime tokenizer replay to the pure contract's exact prompt set."""

    encode = getattr(tokenizer, "encode", None)
    _require(callable(encode), "native tokenizer exposes no encode method")

    def exact(text: str, expected_ids: Sequence[int], name: str) -> dict[str, object]:
        encoded = encode(text)
        _require(
            isinstance(encoded, (tuple, list))
            and all(type(token_id) is int for token_id in encoded),
            f"{name} tokenizer output is invalid",
        )
        _require(tuple(encoded) == tuple(expected_ids), f"{name} token IDs drifted")
        return _prompt_descriptor(text, expected_ids)

    target = exact(
        gate.TARGET_PROMPT_TEXT,
        gate.TARGET_PROMPT_TOKEN_IDS,
        "target prompt",
    )
    _require(
        target["token_count"] == gate.TARGET_PROMPT_TOKENS
        and target["text_sha256"] == gate.TARGET_PROMPT_TEXT_SHA256
        and target["token_ids_sha256"] == gate.TARGET_PROMPT_TOKEN_IDS_SHA256,
        "target prompt identity differs from the pure contract",
    )
    _require(
        len(gate.DISTRACTOR_TEXTS) == len(gate.DISTRACTOR_TOKEN_IDS) == 31,
        "fixed distractor set must contain exactly 31 prompts",
    )
    distractors = [
        exact(text, (token_id,), f"distractor prompt {index}")
        for index, (text, token_id) in enumerate(
            zip(
                gate.DISTRACTOR_TEXTS,
                gate.DISTRACTOR_TOKEN_IDS,
                strict=True,
            )
        )
    ]
    _require(
        len(gate.PRESSURE_PROMPT_TEXTS)
        == len(gate.PRESSURE_PROMPT_TOKEN_IDS)
        == gate.MAX_PRESSURE_REQUESTS,
        "fixed pressure prompt set differs",
    )
    pressure = [
        exact(text, token_ids, f"pressure prompt {index}")
        for index, (text, token_ids) in enumerate(
            zip(
                gate.PRESSURE_PROMPT_TEXTS,
                gate.PRESSURE_PROMPT_TOKEN_IDS,
                strict=True,
            )
        )
    ]
    return {
        "target": target,
        "distractors": distractors,
        "pressure": pressure,
    }


async def _capture_group(
    *,
    backend: object,
    tokenizer: object,
    request_factory: Any,
    observer: _ScheduleObserver,
    scenario: str,
    runtime_id: str,
    runtime_nonce: str,
    requests: Sequence[Mapping[str, object]],
    sampling: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    runner = _parallel_runner(backend)
    cache = getattr(backend, "_cache", None)
    _require(cache is not None, "native runtime exposes no radix cache")
    observations: list[Any] = []
    for request in requests:
        prompt = request.get("prompt")
        _require(isinstance(prompt, Mapping), "capture request prompt is missing")
        prompt_ids = prompt.get("token_ids")
        _require(isinstance(prompt_ids, list), "capture request token IDs are missing")
        observations.append(
            _request_observation(
                backend=backend,
                tokenizer=tokenizer,
                request_factory=request_factory,
                scenario=scenario,
                runtime_id=runtime_id,
                runtime_nonce=runtime_nonce,
                request_id=str(request["request_id"]),
                role=str(request["role"]),
                cohort_index=int(request["cohort_index"]),
                cohort_size=len(requests),
                prompt=prompt,
                sampling=sampling,
                cache_probe_before=_cache_probe(cache, tuple(prompt_ids)),
            )
        )
    _reset_structural_stats(runner)
    observer.begin(scenario)
    try:
        captured = list(await asyncio.gather(*observations))
    finally:
        steps = observer.finish(scenario)
    structure = _collect_structure(runner, steps)
    return captured, structure


async def _capture_pressure(
    *,
    backend: object,
    tokenizer: object,
    request_factory: Any,
    observer: _ScheduleObserver,
    runtime_id: str,
    runtime_nonce: str,
    target: Mapping[str, object],
    prompts: Sequence[Mapping[str, object]],
    sampling: Mapping[str, object],
) -> dict[str, object]:
    scenario = "pressure"
    runner = _parallel_runner(backend)
    cache = getattr(backend, "_cache", None)
    _require(cache is not None, "native runtime exposes no radix cache")
    target_ids_value = target.get("token_ids")
    _require(isinstance(target_ids_value, list), "target prompt token IDs are missing")
    target_ids = tuple(target_ids_value)
    target_before = _cache_probe(cache, target_ids)
    observations: list[dict[str, object]] = []
    _reset_structural_stats(runner)
    observer.begin(scenario)
    try:
        for index, prompt in enumerate(prompts):
            prompt_ids = prompt.get("token_ids")
            _require(isinstance(prompt_ids, list), "pressure prompt token IDs are missing")
            observation = await _request_observation(
                backend=backend,
                tokenizer=tokenizer,
                request_factory=request_factory,
                scenario=scenario,
                runtime_id=runtime_id,
                runtime_nonce=runtime_nonce,
                request_id=f"pressure-{index:02d}",
                role="pressure",
                cohort_index=index,
                cohort_size=len(prompts),
                prompt=prompt,
                sampling=sampling,
                cache_probe_before=_cache_probe(cache, tuple(prompt_ids)),
            )
            observations.append(observation)
    finally:
        steps = observer.finish(scenario)
    target_after = _cache_probe(cache, target_ids)
    return {
        "scenario": scenario,
        "runtime_id": runtime_id,
        "runtime_nonce": runtime_nonce,
        "target_cache_before": target_before,
        "target_cache_after": target_after,
        "requests": observations,
        "structure": _collect_structure(runner, steps),
    }


async def _capture_full_backend(
    *,
    backend: object,
    tokenizer: object,
    request_factory: Any,
    runtime_id: str,
    runtime_nonce: str,
    prompts: Mapping[str, object],
    sampling: Mapping[str, object],
) -> dict[str, object]:
    target = prompts.get("target")
    distractors = prompts.get("distractors")
    pressure_prompts = prompts.get("pressure")
    _require(isinstance(target, Mapping), "target prompt is missing")
    _require(
        isinstance(distractors, list) and len(distractors) == 31,
        "distractor prompt set differs",
    )
    _require(
        isinstance(pressure_prompts, list)
        and len(pressure_prompts) == gate.MAX_PRESSURE_REQUESTS,
        "pressure prompt set differs",
    )
    observer = _ScheduleObserver(getattr(backend, "_scheduler", None))
    try:
        batch_requests: list[dict[str, object]] = [
            {
                "request_id": "batch32-target",
                "role": "target",
                "cohort_index": 0,
                "prompt": target,
            }
        ]
        batch_requests.extend(
            {
                "request_id": f"batch32-distractor-{index:02d}",
                "role": "distractor",
                "cohort_index": index + 1,
                "prompt": prompt,
            }
            for index, prompt in enumerate(distractors)
        )
        batch, batch_structure = await _capture_group(
            backend=backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            observer=observer,
            scenario="batch32-cold",
            runtime_id=runtime_id,
            runtime_nonce=runtime_nonce,
            requests=batch_requests,
            sampling=sampling,
        )
        pressure = await _capture_pressure(
            backend=backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            observer=observer,
            runtime_id=runtime_id,
            runtime_nonce=runtime_nonce,
            target=target,
            prompts=pressure_prompts,
            sampling=sampling,
        )
        alone_cold, cold_structure = await _capture_group(
            backend=backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            observer=observer,
            scenario="alone-cold",
            runtime_id=runtime_id,
            runtime_nonce=runtime_nonce,
            requests=(
                {
                    "request_id": "alone-cold-target",
                    "role": "target",
                    "cohort_index": 0,
                    "prompt": target,
                },
            ),
            sampling=sampling,
        )
        alone_warm, warm_structure = await _capture_group(
            backend=backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            observer=observer,
            scenario="alone-warm",
            runtime_id=runtime_id,
            runtime_nonce=runtime_nonce,
            requests=(
                {
                    "request_id": "alone-warm-target",
                    "role": "target",
                    "cohort_index": 0,
                    "prompt": target,
                },
            ),
            sampling=sampling,
        )
    finally:
        observer.close()
    return {
        "batch": batch,
        "batch_structure": batch_structure,
        "pressure": pressure,
        "alone_cold": alone_cold[0],
        "cold_structure": cold_structure,
        "alone_warm": alone_warm[0],
        "warm_structure": warm_structure,
    }


async def _capture_chunk_backend(
    *,
    backend: object,
    tokenizer: object,
    request_factory: Any,
    runtime_id: str,
    runtime_nonce: str,
    target: Mapping[str, object],
    sampling: Mapping[str, object],
) -> dict[str, object]:
    observer = _ScheduleObserver(getattr(backend, "_scheduler", None))
    try:
        observations, structure = await _capture_group(
            backend=backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            observer=observer,
            scenario="chunked-alone-cold",
            runtime_id=runtime_id,
            runtime_nonce=runtime_nonce,
            requests=(
                {
                    "request_id": "chunked-alone-cold-target",
                    "role": "target",
                    "cohort_index": 0,
                    "prompt": target,
                },
            ),
            sampling=sampling,
        )
    finally:
        observer.close()
    return {"request": observations[0], "structure": structure}


def _runtime_record(
    *,
    backend: object,
    runtime_id: str,
    runtime_nonce: str,
    hardware: Mapping[str, object],
) -> dict[str, object]:
    """Attest the native objects, rather than only echoing constructor inputs."""

    _require(runtime_id in {"full", "chunk"}, "unknown A12 runtime identity")
    expected = (
        gate.FULL_RUNTIME_CONFIG
        if runtime_id == "full"
        else gate.CHUNK_RUNTIME_CONFIG
    )
    loop = getattr(backend, "_loop", None)
    cache = getattr(backend, "_cache", None)
    scheduler = getattr(backend, "_scheduler", None)
    launcher = getattr(loop, "tp_launcher", None)
    runner = getattr(launcher, "runner", None)
    _require(
        loop is not None and cache is not None and scheduler is not None,
        f"{runtime_id} native engine objects are incomplete",
    )
    _require(
        launcher is not None and runner is not None,
        f"{runtime_id} native TP launcher is missing",
    )

    actual_geometry = {
        "tensor_parallel_size": getattr(backend, "tensor_parallel_size", None),
        "expert_parallel_size": getattr(backend, "expert_parallel_size", None),
        "pipeline_depth": getattr(loop, "_pipeline_depth", None),
        "page_size": getattr(cache, "page_size", None),
        "num_pages": getattr(cache, "num_pages", None),
        "max_model_len": getattr(loop, "_max_model_len", None),
        "max_num_seqs": getattr(scheduler, "_max_seqs", None),
        "max_num_batched_tokens": getattr(scheduler, "_budget", None),
        "kv_cache_dtype": getattr(backend, "kv_cache_dtype_resolved", None),
    }
    for name, actual in actual_geometry.items():
        _require(
            actual == expected[name],
            f"{runtime_id} native {name} differs from the frozen engine",
        )
    _require(
        getattr(backend, "kv_cache_dtype_requested", None)
        == expected["kv_cache_dtype"],
        f"{runtime_id} requested KV dtype differs",
    )
    defaults = getattr(backend, "generation_defaults", None)
    _require(
        getattr(defaults, "mode", None) == "none",
        f"{runtime_id} unexpectedly loaded generation_config defaults",
    )

    local = getattr(runner, "_local", None)
    graph = getattr(local, "_graph", None)
    _require(graph is not None, f"{runtime_id} native CUDA graph executor is missing")
    _require(
        tuple(getattr(graph, "configured_buckets", ()))
        == gate.CUDA_GRAPH_BUCKETS
        and getattr(graph, "max_pages", None) == expected["cuda_graph_max_pages"],
        f"{runtime_id} CUDA graph geometry differs",
    )
    graph_backend = getattr(graph, "_backend", None)
    _require(
        getattr(graph_backend, "_warmup_iters", None)
        == expected["cuda_graph_warmup_iters"],
        f"{runtime_id} CUDA graph warmup count differs",
    )

    ownership_probe = getattr(runner, "sampling_ownership_metadata", None)
    _require(callable(ownership_probe), f"{runtime_id} exposes no rank topology probe")
    raw_ownership = ownership_probe()
    _require(
        isinstance(raw_ownership, (tuple, list))
        and len(raw_ownership) == gate.TENSOR_PARALLEL_SIZE,
        f"{runtime_id} rank topology is incomplete",
    )
    ownership: list[dict[str, object]] = []
    for rank, raw in enumerate(raw_ownership):
        _require(isinstance(raw, Mapping), f"{runtime_id} rank {rank} topology is invalid")
        _require(
            raw.get("rank") == rank
            and raw.get("control_world_size") == gate.TENSOR_PARALLEL_SIZE
            and raw.get("control_backend") == "gloo"
            and raw.get("model_world_size") == gate.TENSOR_PARALLEL_SIZE
            and raw.get("model_backend") == "nccl"
            and raw.get("sampling_owner") is (rank == 0)
            and raw.get("sampler_present") is (rank == 0)
            and raw.get("device") == f"cuda:{rank}",
            f"{runtime_id} rank {rank} topology differs",
        )
        ownership.append(
            {
                "rank": rank,
                "world_size": gate.TENSOR_PARALLEL_SIZE,
                "sampling_owner": rank == 0,
            }
        )

    hardware_gpus = hardware.get("gpus")
    _require(
        isinstance(hardware_gpus, list)
        and len(hardware_gpus) == gate.TENSOR_PARALLEL_SIZE
        and all(isinstance(row, Mapping) for row in hardware_gpus),
        "hardware rank inventory is unavailable",
    )
    rank_gpu_uuid_order = [str(row["uuid"]) for row in hardware_gpus]

    decision = getattr(backend, "attention_backend_decision", None)
    components = getattr(decision, "components", None)
    _require(
        getattr(decision, "requested", None) == "flashinfer"
        and getattr(decision, "resolved", None) == "flashinfer"
        and isinstance(components, Mapping)
        and components.get("prefill") == "flashinfer"
        and components.get("decode") == "flashinfer"
        and components.get("kv_mode") == "paged-direct",
        f"{runtime_id} did not construct exact FlashInfer attention",
    )
    from kairyu.engine.core.attention_selector import (
        attention_backend_execution_identity,
        attention_backend_identity,
    )

    selected_identity = attention_backend_identity(decision)
    _require(
        getattr(launcher, "attention_backend_identity", None) == selected_identity,
        f"{runtime_id} launcher attention selection identity differs",
    )
    model = getattr(local, "_model", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    try:
        layer_values = tuple(layers)
    except TypeError as error:
        raise ValueError(
            f"{runtime_id} rank-0 model exposes no iterable attention layers"
        ) from error
    _require(
        bool(layer_values),
        f"{runtime_id} rank-0 model exposes no attention layers",
    )
    layer_backends = [
        getattr(getattr(layer, "self_attn", None), "backend", None)
        for layer in layer_values
    ]
    _require(
        all(item is not None for item in layer_backends)
        and len({id(item) for item in layer_backends}) == 1,
        f"{runtime_id} rank-0 layers do not share one selected attention backend",
    )
    execution_identity = attention_backend_execution_identity(
        decision,
        layer_backends[0],
    )
    return {
        "runtime_id": runtime_id,
        "nonce": runtime_nonce,
        "engine": dict(expected),
        "topology": {
            "parallelism": {"tensor": 8, "expert": 1, "pipeline": 1},
            "control_backend": "gloo",
            "model_backend": "nccl",
            "rank_gpu_uuid_order": rank_gpu_uuid_order,
            "sampling_ownership": ownership,
        },
        "attention": {
            "requested_backend": "flashinfer",
            "selected_backend": "flashinfer",
            "execution_identity": execution_identity,
            "fallback_reason": None,
        },
    }


def _native_backend(model_path: Path, config: Mapping[str, object]) -> object:
    # Explicitly configure the selector before importing any engine/GPU module.
    os.environ["KAIRYU_ATTENTION_BACKEND"] = "flashinfer"
    from kairyu.engine.kairyu_backend import KairyuBackend

    return KairyuBackend(
        model_path=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=int(config["tensor_parallel_size"]),
        expert_parallel_size=int(config["expert_parallel_size"]),
        pipeline_depth=int(config["pipeline_depth"]),
        page_size=int(config["page_size"]),
        num_pages=int(config["num_pages"]),
        max_model_len=int(config["max_model_len"]),
        max_num_seqs=int(config["max_num_seqs"]),
        max_num_batched_tokens=int(config["max_num_batched_tokens"]),
        priority_age_s=None,
        kv_cache_dtype=str(config["kv_cache_dtype"]),
        decode_mode=str(config["decode_mode"]),
        cuda_graph_max_batch=int(config["cuda_graph_max_batch"]),
        cuda_graph_max_pages=int(config["cuda_graph_max_pages"]),
        cuda_graph_warmup_iters=int(config["cuda_graph_warmup_iters"]),
        generation_config=str(config["generation_config"]),
    )


def _request_factory() -> tuple[object, Any]:
    from kairyu.engine.backend import GenerationRequest
    from kairyu.engine.prompt import TokensPrompt
    from kairyu.sampling_params import SamplingParams

    kwargs = dict(gate.SAMPLING_CONFIG)
    omitted = kwargs.pop("generation_config_omitted")
    _require(omitted == [], "A12 sampling omission contract differs")
    params = SamplingParams(**kwargs)

    def build(
        request_id: str,
        token_ids: tuple[int, ...],
        sampling: Mapping[str, object],
    ) -> object:
        _require(
            _same_json_value(sampling, gate.SAMPLING_CONFIG),
            "request sampling differs from the frozen A12 contract",
        )
        return GenerationRequest(request_id, TokensPrompt(token_ids), params)

    return params, build


def _request_row(
    *,
    run_id: str,
    row_index: int,
    observation: Mapping[str, object],
    structure: Mapping[str, object] | None,
    structure_sha256: str,
) -> dict[str, object]:
    return {
        "row_type": "request",
        "row_index": row_index,
        "run_id": run_id,
        "scenario": observation["scenario"],
        "runtime_id": observation["runtime_id"],
        "runtime_nonce": observation["runtime_nonce"],
        "request_id": observation["request_id"],
        "role": observation["role"],
        "cohort_index": observation["cohort_index"],
        "cohort_size": observation["cohort_size"],
        "prompt": observation["prompt"],
        "sampling": observation["sampling"],
        "cache": observation["cache"],
        "response": observation["response"],
        "error_type": observation["error_type"],
        "retry_count": observation["retry_count"],
        "structure": dict(structure) if structure is not None else None,
        "structure_sha256": structure_sha256,
    }


def _pressure_request_row(observation: Mapping[str, object]) -> dict[str, object]:
    return {
        "request_id": observation["request_id"],
        "prompt": observation["prompt"],
        "sampling": observation["sampling"],
        "cache": observation["cache"],
        "response": observation["response"],
        "error_type": observation["error_type"],
        "retry_count": observation["retry_count"],
    }


def _assemble_rows(
    *,
    run_id: str,
    prompts: Mapping[str, object],
    source_start: Mapping[str, object],
    source_end: Mapping[str, object],
    checkpoint_start: Mapping[str, object],
    checkpoint_end: Mapping[str, object],
    hardware: Mapping[str, object],
    runtimes: Mapping[str, object],
    full: Mapping[str, object],
    chunk: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "row_type": "run_start",
            "row_index": 0,
            "schema_version": gate.SCHEMA_VERSION,
            "gate": gate.GATE,
            "run_id": run_id,
            "config": gate.frozen_config(),
            "prompts": dict(prompts),
            "source": {
                "measurement_start": dict(source_start),
                "measurement_end": dict(source_end),
                "stable_across_measurement": _same_json_value(
                    source_start, source_end
                ),
            },
            "checkpoint": {
                "measurement_start": dict(checkpoint_start),
                "measurement_end": dict(checkpoint_end),
                "stable_across_measurement": _same_json_value(
                    checkpoint_start, checkpoint_end
                ),
            },
            "hardware": dict(hardware),
            "runtimes": dict(runtimes),
        }
    ]

    batch = full.get("batch")
    batch_structure = full.get("batch_structure")
    _require(
        isinstance(batch, list)
        and len(batch) == gate.BATCH_SIZE
        and isinstance(batch_structure, Mapping),
        "full capture lost the batch phase",
    )
    batch_sha = gate.sha256_json(batch_structure)
    for index, observation in enumerate(batch):
        _require(isinstance(observation, Mapping), "batch observation is invalid")
        rows.append(
            _request_row(
                run_id=run_id,
                row_index=len(rows),
                observation=observation,
                structure=batch_structure if index == 0 else None,
                structure_sha256=batch_sha,
            )
        )

    pressure = full.get("pressure")
    _require(isinstance(pressure, Mapping), "full capture lost pressure evidence")
    pressure_observations = pressure.get("requests")
    pressure_structure = pressure.get("structure")
    _require(
        isinstance(pressure_observations, list)
        and len(pressure_observations) == gate.MAX_PRESSURE_REQUESTS
        and all(isinstance(item, Mapping) for item in pressure_observations)
        and isinstance(pressure_structure, Mapping),
        "pressure capture is incomplete",
    )
    rows.append(
        {
            "row_type": "pressure_summary",
            "row_index": len(rows),
            "run_id": run_id,
            "scenario": pressure["scenario"],
            "runtime_id": pressure["runtime_id"],
            "runtime_nonce": pressure["runtime_nonce"],
            "target_cache_before": pressure["target_cache_before"],
            "target_cache_after": pressure["target_cache_after"],
            "requests": [
                _pressure_request_row(observation)
                for observation in pressure_observations
            ],
            "structure": dict(pressure_structure),
            "structure_sha256": gate.sha256_json(pressure_structure),
        }
    )

    singleton_specs = (
        (full.get("alone_cold"), full.get("cold_structure")),
        (full.get("alone_warm"), full.get("warm_structure")),
        (chunk.get("request"), chunk.get("structure")),
    )
    for observation, structure in singleton_specs:
        _require(
            isinstance(observation, Mapping) and isinstance(structure, Mapping),
            "singleton capture is incomplete",
        )
        rows.append(
            _request_row(
                run_id=run_id,
                row_index=len(rows),
                observation=observation,
                structure=structure,
                structure_sha256=gate.sha256_json(structure),
            )
        )

    observations = [
        *batch,
        *pressure_observations,
        *(observation for observation, _structure in singleton_specs),
    ]
    errors = [
        f"{observation['request_id']}:{observation['error_type']}"
        for observation in observations
        if observation["error_type"] is not None
    ]
    rows.append(
        {
            "row_type": "run_end",
            "row_index": len(rows),
            "run_id": run_id,
            "status": "complete" if not errors else "failed",
            "errors": errors,
            "runtime_count": 2,
            "scenario_count": len(gate.SCENARIOS),
        }
    )
    _require(len(rows) == 38, "assembled A12 raw row count differs")
    return rows


async def _run_native(args: argparse.Namespace) -> list[dict[str, object]]:
    model_path = _resolve_existing_path(
        args.model_path,
        description="model path",
        directory=True,
    )
    source_start = _source_snapshot()
    checkpoint_start = _live_checkpoint(model_path)
    hardware_start = _hardware()

    # These imports are intentionally below all direct-execution/source
    # preflight checks.  Resolving and replaying the tokenizer happens before
    # either GPU runtime is constructed.
    from kairyu.engine.tokenizer import resolve_tokenizer

    tokenizer = resolve_tokenizer(str(model_path))
    prompts = _fixed_prompt_descriptors(tokenizer)
    _require(
        _same_json_value(prompts, gate.fixed_prompts()),
        "native tokenizer prompt inventory differs from the pure contract",
    )
    _params, request_factory = _request_factory()
    sampling = dict(gate.SAMPLING_CONFIG)

    full_nonce = uuid.uuid4().hex
    full_backend = _native_backend(model_path, gate.FULL_RUNTIME_CONFIG)
    try:
        full_runtime = _runtime_record(
            backend=full_backend,
            runtime_id="full",
            runtime_nonce=full_nonce,
            hardware=hardware_start,
        )
        full_capture = await _capture_full_backend(
            backend=full_backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            runtime_id="full",
            runtime_nonce=full_nonce,
            prompts=prompts,
            sampling=sampling,
        )
    finally:
        await full_backend.shutdown()
    del full_backend

    # The chunk arm must own a genuinely fresh cache, scheduler, graph executor,
    # worker group, and nonce.  Release all rank-0 references before constructing
    # the second Qwen3-32B runtime.
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    chunk_nonce = uuid.uuid4().hex
    chunk_backend = _native_backend(model_path, gate.CHUNK_RUNTIME_CONFIG)
    try:
        chunk_runtime = _runtime_record(
            backend=chunk_backend,
            runtime_id="chunk",
            runtime_nonce=chunk_nonce,
            hardware=hardware_start,
        )
        target = prompts.get("target")
        _require(isinstance(target, Mapping), "fixed target prompt disappeared")
        chunk_capture = await _capture_chunk_backend(
            backend=chunk_backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            runtime_id="chunk",
            runtime_nonce=chunk_nonce,
            target=target,
            sampling=sampling,
        )
    finally:
        await chunk_backend.shutdown()
    del chunk_backend
    gc.collect()
    torch.cuda.empty_cache()

    source_end = _source_snapshot()
    checkpoint_end = _live_checkpoint(model_path)
    hardware_end = _hardware()
    _require(
        _same_json_value(source_start, source_end),
        "native source identity changed during A12 capture",
    )
    _require(
        _same_json_value(checkpoint_start, checkpoint_end),
        "live checkpoint identity changed during A12 capture",
    )
    _require(
        _same_json_value(hardware_start, hardware_end),
        "hardware or software identity changed during A12 capture",
    )
    return _assemble_rows(
        run_id=f"a12-{uuid.uuid4().hex}",
        prompts=prompts,
        source_start=source_start,
        source_end=source_end,
        checkpoint_start=checkpoint_start,
        checkpoint_end=checkpoint_end,
        hardware=hardware_start,
        runtimes={"full": full_runtime, "chunk": chunk_runtime},
        full=full_capture,
        chunk=chunk_capture,
    )


def _planned_output_directory(path: Path) -> Path:
    candidate = Path(path)
    _require(candidate.name not in {"", ".", ".."}, "output directory is invalid")
    parent = _resolve_existing_path(
        candidate.parent,
        description="output parent directory",
        directory=True,
    )
    target = parent / candidate.name
    _require(
        not target.exists() and not target.is_symlink(),
        f"refusing to overwrite existing A12 output directory: {target}",
    )
    return target


def _create_output_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(
            f"refusing to overwrite existing A12 output directory: {path}"
        ) from error
    except OSError as error:
        raise ValueError(f"cannot create A12 output directory: {path}") from error


def _raw_artifact_path(path: Path) -> Path:
    target = Path(path)
    if target.is_dir():
        return target / gate.RAW_NAME
    if target.name == gate.MANIFEST_NAME:
        return target.with_name(gate.RAW_NAME)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and independently replay the A12 batch-invariance gate."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    native = commands.add_parser(
        "run-native",
        help="run the fixed Qwen3-32B TP8 native evidence matrix",
    )
    native.add_argument("--model-path", type=Path, required=True)
    native.add_argument("--output-dir", type=Path, required=True)
    native.add_argument("--assert-gate", action="store_true")

    verify = commands.add_parser(
        "verify",
        help="verify retained raw and manifest bytes by independent replay",
    )
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)
    verify.add_argument("--assert-gate", action="store_true")

    replay = commands.add_parser(
        "replay",
        help="derive the verdict from canonical raw JSONL without a manifest",
    )
    replay.add_argument("--artifact", type=Path, required=True)
    replay.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run-native":
        output_dir = _planned_output_directory(args.output_dir)
        rows = asyncio.run(_run_native(args))
        # Reserve the whole result directory after measurement.  Both child
        # files are then additionally opened with O_EXCL by the pure writer.
        _create_output_directory(output_dir)
        manifest = gate.write_artifact(
            output_dir / gate.RAW_NAME,
            output_dir / gate.MANIFEST_NAME,
            rows,
            assert_gate=args.assert_gate,
        )
    elif args.command == "verify":
        manifest = gate.verify_artifact(
            args.artifact,
            args.manifest,
            assert_gate=args.assert_gate,
        )
    else:
        manifest = gate.replay_raw(
            _raw_artifact_path(args.artifact),
            assert_gate=args.assert_gate,
        )
    print(gate.canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
