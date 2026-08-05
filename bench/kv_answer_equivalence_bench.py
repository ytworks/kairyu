#!/usr/bin/env python3
"""Checkout-only native KV cold/warm answer-equivalence operator.

``run-native`` records one fixed production-topology child capture.  The five
cells cover F2c TP2, F2d TP2, F4a TP4/TP8, and F4b TP4.  ``assemble`` combines
exactly those five children into the package-owned replayable artifact;
``verify`` and ``replay`` never construct an engine.

Engine imports are deliberately confined to the ``run-native`` execution
path.  In particular, ``--help``, assembly, and retained-evidence replay do
not import torch, model loaders, or CUDA-facing modules.
"""
# ruff: noqa: E402

from __future__ import annotations

# Formal commands are path-only and require the *initial* interpreter to have
# ``-I -B``.  An in-script restart would be too late because normal Python may
# already have executed ``sitecustomize``.  Help is non-evidence and may
# restart solely to preserve inventory compatibility.  A fresh cache prefix
# installed after isolation makes existing repository ``.pyc`` files
# ineligible for import; ``-B`` keeps that prefix from being populated.
_bootstrap_sys = __import__("sys")
_bootstrap_help = any(argument in {"-h", "--help"} for argument in _bootstrap_sys.argv[1:])
if __name__ == "__main__":
    if __spec__ is not None and not _bootstrap_help:
        raise RuntimeError("B7 evidence commands require the direct checkout path, not python -m")
    _bootstrap_posix = __import__("posix")
    if not _bootstrap_sys.flags.isolated or not _bootstrap_sys.dont_write_bytecode:
        if not _bootstrap_help:
            raise RuntimeError(
                "B7 evidence commands must start as: python -I -B "
                "bench/kv_answer_equivalence_bench.py ..."
            )
        _bootstrap_posix.execv(
            _bootstrap_sys.executable,
            [_bootstrap_sys.executable, "-I", "-B", __file__, *_bootstrap_sys.argv[1:]],
        )

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

# A formal clean-source run rejects repository bytecode rather than trusting
# ignored ``__pycache__`` entries.  Set both the current interpreter flag and
# the inherited worker-process environment before importing any repo package.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_TEXT = str(_REPO_ROOT)

# Isolated mode has not exposed the checkout on sys.path yet.  Allocate an
# unpredictable, exclusive 0700 cache root now and redirect all later cache
# lookups there before making the repository importable.  The object remains
# live until interpreter shutdown and removes the empty directory then.
_IMPORT_CACHE_DIRECTORY = None
if __name__ == "__main__" and __spec__ is None:
    _IMPORT_CACHE_DIRECTORY = tempfile.TemporaryDirectory(
        prefix="kairyu-b7-import-cache-",
        dir="/tmp",
    )
    sys.pycache_prefix = _IMPORT_CACHE_DIRECTORY.name

_IMPORT_ARTIFACT_SUFFIXES = (".py", ".pyc", ".pth", ".so")
_GIT_OTHER_PATHSPEC = (
    ".",
    ":(glob,exclude).venv/**",
)


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
    """Return whether an unversioned path can carry executable import code."""

    parts = PurePosixPath(path.rstrip("/")).parts
    if not parts or _safe_generated_source_path(path):
        return False
    # ``git ls-files --others`` reports an embedded untracked repository as a
    # directory rather than enumerating its files.  A directory whose entire
    # path is importable can still provide a shadow package and must fail.
    if path.endswith("/"):
        return all(part.isidentifier() for part in parts)
    return parts[-1].endswith(_IMPORT_ARTIFACT_SUFFIXES)


def _assert_no_import_shadows() -> None:
    offenders: set[str] = set()
    for ignored in (False, True):
        for path in _git_other_paths(ignored=ignored):
            if _safe_generated_source_path(path):
                continue
            working_path = _REPO_ROOT / path
            if working_path.is_symlink() or _could_be_repo_import(path):
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


_FORMAL_DIRECT_EXECUTION = __name__ == "__main__" and __spec__ is None and not _bootstrap_help
if _FORMAL_DIRECT_EXECUTION:
    # This runs while isolated mode still excludes the checkout from sys.path.
    # No repository package can execute until both checks have passed.
    _assert_tracked_tree_clean_before_repo_import()
    _assert_no_import_shadows()

sys.path[:] = [entry for entry in sys.path if entry != _REPO_ROOT_TEXT]
sys.path.insert(0, _REPO_ROOT_TEXT)

from kairyu.bench import kv_equivalence as gate  # noqa: E402

for _module_name, _relative_origin in (
    ("kairyu", "kairyu/__init__.py"),
    ("kairyu.bench", "kairyu/bench/__init__.py"),
    ("kairyu.bench.kv_equivalence", "kairyu/bench/kv_equivalence.py"),
):
    _module = sys.modules.get(_module_name)
    _origin = getattr(_module, "__file__", None)
    try:
        _resolved_origin = Path(_origin).resolve(strict=True)
        _expected_origin = (_REPO_ROOT / _relative_origin).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise RuntimeError(f"cannot establish checkout origin for {_module_name}") from error
    if _resolved_origin != _expected_origin:
        raise RuntimeError(f"refusing non-checkout {_module_name} origin: {_resolved_origin}")

RAW_NAME = "kv-answer-equivalence-raw.jsonl"
MANIFEST_NAME = "kv-answer-equivalence-manifest.json"
_MAX_PRESSURE_REQUESTS = gate.MAX_PRESSURE_REQUESTS
_PINNED_MODEL = gate.MODEL
_PINNED_REVISION = gate.MODEL_REVISION
_CHECKPOINT_METADATA = tuple(gate.EXPECTED_METADATA_SHA256)
_WEIGHT_INDEX = "model.safetensors.index.json"
_F2D_ROUTER_NAME = "prefix-weight-f2d-router.jsonl"
_F4B_QUALITY_MANIFEST_NAME = "agentic-kv-tier-f4b-quality-manifest.json"
_F4B_QUALITY_RAW_NAME = "agentic-kv-tier-f4b-quality-raw.jsonl"
_RUNTIME_SOURCE_PATHS = gate.REQUIRED_SOURCE_PATHS

# B7 is a topology companion, not merely a four-label checklist.  F4a's TP4
# and TP8 parent rows exercise different reviewed deployments and therefore
# remain distinct native cells.
_CELL_SPECS = gate.CELL_SPECS
_CELL_IDS = gate.CELL_IDS

_CHECKOUT_MODULE_PREFIXES = ("bench", "kairyu")


def _same_json_value(left: object, right: object) -> bool:
    """Compare JSON values without Python bool/int/float coercion."""

    try:
        return gate.canonical_json(left) == gate.canonical_json(right)
    except (TypeError, ValueError):
        return False


def _strict_json_object_bytes(raw: bytes, *, description: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{description}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"{description}: non-finite JSON number {value!r}")

    try:
        text = raw.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{description}: JSON is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{description}: cannot parse JSON") from error
    _require(
        isinstance(parsed, dict),
        f"{description}: JSON root must be an object",
    )
    return parsed


def _read_exact_bytes(path: Path, *, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {description}: {path}") from error


def _strict_json_object(path: Path) -> dict[str, object]:
    return _strict_json_object_bytes(
        _read_exact_bytes(path, description="JSON file"),
        description=str(path),
    )


def _strict_jsonl_bytes(
    raw: bytes,
    *,
    description: str,
    canonical: bool = True,
) -> list[dict[str, object]]:
    """Parse one exact canonical JSONL byte snapshot without reopening its path."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description}: raw JSONL is not UTF-8") from error
    _require(bool(text), f"{description}: raw JSONL is empty")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        _require(
            line.endswith("\n") and not line.endswith("\r\n"),
            f"{description}: invalid JSONL framing at line {line_number}",
        )
        payload = line[:-1]
        _require(bool(payload), f"{description}: empty JSONL row at line {line_number}")
        value = _strict_json_object_bytes(
            payload.encode("utf-8"),
            description=f"{description} line {line_number}",
        )
        if canonical:
            _require(
                payload == gate.canonical_json(value),
                f"{description}: JSONL line {line_number} is not canonical",
            )
        rows.append(value)
    _require(bool(rows), f"{description}: raw JSONL is empty")
    return rows


def _assert_exact_bytes(path: Path, expected: bytes, *, description: str) -> None:
    _require(
        _read_exact_bytes(path, description=description) == expected,
        f"{description} changed during validation: {path}",
    )


def _resolve_existing_path(path: Path, *, description: str, directory: bool) -> Path:
    """Resolve one input once so later symlink retargets cannot change its target."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{description} does not exist: {path}") from error
    _require(
        resolved.is_dir() if directory else resolved.is_file(),
        f"{description} is not a {'directory' if directory else 'file'}: {resolved}",
    )
    return resolved


def _cell_spec(cell_id: str) -> Mapping[str, object]:
    try:
        return _CELL_SPECS[cell_id]
    except KeyError as error:
        raise ValueError(f"unknown B7 cell: {cell_id}") from error


def _checkpoint_parent_view(value: object, *, description: str) -> dict[str, object]:
    """Copy the retained full checkpoint fields used for live comparison."""

    _require(isinstance(value, Mapping), f"{description}: checkpoint is missing")
    checkpoint = dict(value)
    required = {
        "model",
        "revision",
        "architecture",
        "metadata_sha256",
        "model_config_sha256",
        "weights",
        "weights_sha256",
        "pinned_weights_rollup",
    }
    _require(
        required <= set(checkpoint),
        f"{description}: checkpoint identity is incomplete",
    )
    return {name: checkpoint[name] for name in sorted(required)}


def _parent_checkpoint_binding(
    *,
    cell_id: str,
    manifest: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[str, object],
    dict[str, object],
    int | None,
    dict[str, object] | None,
]:
    """Extract the reviewed model/TP constraint from one parent raw file."""

    spec = _cell_spec(cell_id)
    feature = str(spec["feature_id"])
    tensor_parallel_size = int(spec["tensor_parallel_size"])
    first = rows[0]
    expected_schema = gate.FEATURE_SPECS[feature]["parent_schema_version"]

    if feature == "f2c":
        _require(
            first.get("schema_version") == expected_schema,
            "f2c: parent raw schema differs",
        )
        _require(first.get("type") == "run", "f2c: parent raw run row is missing")
        _require(first.get("gate") == "G5-F2c", "f2c: parent raw gate differs")
        _require(
            first.get("model") == _PINNED_MODEL,
            "f2c: parent model differs from the pinned companion checkpoint",
        )
        _require(
            first.get("model_revision") == _PINNED_REVISION,
            "f2c: parent model revision differs",
        )
        model_digests = first.get("model_digests")
        _require(isinstance(model_digests, Mapping), "f2c: model digests are missing")
        rollup = model_digests.get("weights-rollup")
        _require(
            isinstance(rollup, str) and len(rollup) == 64,
            "f2c: weights rollup is invalid",
        )
        provenance = first.get("provenance_start")
        _require(isinstance(provenance, Mapping), "f2c: parent provenance is missing")
        configuration = provenance.get("configuration")
        _require(isinstance(configuration, Mapping), "f2c: configuration proof is missing")
        configuration_value = configuration.get("value")
        _require(
            isinstance(configuration_value, Mapping),
            "f2c: configuration value is missing",
        )
        attestation = configuration_value.get("runtime_attestation")
        _require(isinstance(attestation, Mapping), "f2c: runtime attestation is missing")
        services = attestation.get("services")
        _require(
            isinstance(services, list) and len(services) == 4,
            "f2c: exactly four runtime services are required",
        )
        service_device_ids: dict[str, list[str]] = {}
        all_device_ids: list[str] = []
        for ordinal, service_value in enumerate(services):
            _require(
                isinstance(service_value, Mapping),
                f"f2c: service {ordinal} is invalid",
            )
            service = service_value.get("service")
            device_ids = service_value.get("device_ids")
            _require(
                isinstance(service, str) and bool(service),
                f"f2c: service {ordinal} identity is invalid",
            )
            _require(
                isinstance(device_ids, list)
                and len(device_ids) == 2
                and all(isinstance(device_id, str) for device_id in device_ids)
                and len(set(device_ids)) == 2,
                f"f2c: service {service} does not prove TP2",
            )
            _require(service not in service_device_ids, "f2c: duplicate runtime service")
            service_device_ids[service] = list(device_ids)
            all_device_ids.extend(device_ids)
        _require(
            len(set(all_device_ids)) == 8,
            "f2c: service GPU assignments are not globally unique",
        )
        parent_topology = {
            "kind": "direct",
            "tensor_parallel_size": 2,
            "service_count": 4,
            "service_device_ids": service_device_ids,
            "unique_device_count": 8,
        }
        return (
            {
                "binding_kind": "direct-partial",
                "model": first["model"],
                "revision": first["model_revision"],
                "pinned_weights_rollup": rollup,
            },
            parent_topology,
            None,
            None,
        )

    if feature == "f2d":
        _require(
            first.get("schema_version") == expected_schema,
            "f2d: parent raw schema differs",
        )
        _require(first.get("gate") == "G5-F2d", "f2d: parent raw gate differs")
        _require(first.get("type") == "run", "f2d: parent raw run row is missing")
        return (
            {"binding_kind": "aggregate-common-f2c"},
            dict(_cell_spec(cell_id)["parent_topology"]),
            None,
            None,
        )

    if feature == "f4a":
        _require(
            first.get("schema_version") == expected_schema,
            "f4a: parent raw schema differs",
        )
        _require(first.get("row_type") == "run", "f4a: parent run row is missing")
        _require(
            _same_json_value(first.get("tp_size"), tensor_parallel_size),
            f"{cell_id}: parent TP size differs",
        )
        config = first.get("config")
        _require(isinstance(config, Mapping), "f4a: parent config is missing")
        _require(
            _same_json_value(
                config.get("tensor_parallel_size"),
                tensor_parallel_size,
            ),
            f"{cell_id}: parent config TP size differs",
        )
        checkpoint = _checkpoint_parent_view(first.get("checkpoint"), description=cell_id)
        crossover = manifest.get("crossover")
        _require(isinstance(crossover, Mapping), "f4a: crossover verdict is missing")
        crossover_cell = crossover.get(f"tp{tensor_parallel_size}")
        _require(
            isinstance(crossover_cell, Mapping),
            f"{cell_id}: crossover verdict is missing",
        )
        minimum = crossover_cell.get("min_restore_tokens")
        _require(
            type(minimum) is int and minimum > 0,
            f"{cell_id}: crossover restore threshold is invalid",
        )
        return (
            {"binding_kind": "direct-full", **checkpoint},
            dict(_cell_spec(cell_id)["parent_topology"]),
            int(minimum),
            None,
        )

    _require(feature == "f4b", f"{cell_id}: unsupported parent feature")
    shard_starts = [row for row in rows if row.get("row_type") == "shard_start"]
    _require(len(shard_starts) == 4, "f4b: exactly four shard_start rows are required")
    checkpoints: list[dict[str, object]] = []
    profile_descriptors: list[dict[str, object]] = []
    for ordinal, row in enumerate(shard_starts):
        _require(
            row.get("schema_version") == expected_schema,
            f"f4b: shard {ordinal} schema differs",
        )
        config = row.get("config")
        runtime = row.get("runtime")
        _require(isinstance(config, Mapping), f"f4b: shard {ordinal} config is missing")
        _require(isinstance(runtime, Mapping), f"f4b: shard {ordinal} runtime is missing")
        _require(
            _same_json_value(
                config.get("tensor_parallel_size"),
                tensor_parallel_size,
            )
            and _same_json_value(
                runtime.get("tensor_parallel_size"),
                tensor_parallel_size,
            ),
            f"f4b: shard {ordinal} TP size differs",
        )
        checkpoints.append(
            _checkpoint_parent_view(row.get("checkpoint"), description=f"f4b shard {ordinal}")
        )
        if row.get("arm") == "on":
            profile = row.get("profile")
            _require(
                isinstance(profile, Mapping),
                f"f4b: tier-on shard {ordinal} profile is missing",
            )
            digest = profile.get("file_sha256")
            _require(
                isinstance(digest, str) and len(digest) == 64,
                f"f4b: tier-on shard {ordinal} profile digest is invalid",
            )
            _require(
                runtime.get("dram_kv_tier_profile_sha256") == digest,
                f"f4b: tier-on shard {ordinal} runtime profile differs",
            )
            profile_descriptors.append(dict(profile))
    _require(
        all(_same_json_value(checkpoint, checkpoints[0]) for checkpoint in checkpoints[1:]),
        "f4b: shard checkpoint identities differ",
    )
    _require(
        len(profile_descriptors) == 2
        and _same_json_value(profile_descriptors[0], profile_descriptors[1]),
        "f4b: tier-on profile identities differ",
    )
    return (
        {"binding_kind": "direct-full", **checkpoints[0]},
        dict(_cell_spec(cell_id)["parent_topology"]),
        None,
        profile_descriptors[0],
    )


def _spec(feature: str) -> Mapping[str, object]:
    value = gate.FEATURE_SPECS[feature]
    _require(isinstance(value, Mapping), f"{feature}: feature spec is not a mapping")
    return value


def _require_exact_row_indices(
    rows: Sequence[Mapping[str, object]],
    *,
    description: str,
) -> None:
    _require(
        all(
            type(row.get("row_index")) is int and row["row_index"] == index
            for index, row in enumerate(rows)
        ),
        f"{description}: row indices must be exact contiguous integers",
    )


def _validate_f2d_parent_rows(
    manifest: Mapping[str, object],
    raw_rows: Sequence[Mapping[str, object]],
    router_rows: Sequence[Mapping[str, object]],
) -> None:
    """Close the legacy F2d owner's Python bool/int/float equality gaps."""

    _require_exact_row_indices(raw_rows, description="F2d parent raw")
    expected_run = {
        "type": "run",
        **{
            field: manifest[field]
            for field in (
                "schema_version",
                "gate",
                "config",
                "inputs",
                "trace",
                "source",
                "source_end",
                "environment",
            )
        },
        "row_index": 0,
    }
    _require(
        _same_json_value(raw_rows[0], expected_run),
        "F2d parent run row differs from its manifest",
    )

    family_keys = {
        "type",
        "family_id",
        "split",
        "seed_replica_id",
        "seed_prompt_sha256",
        "prefix_fingerprint_sha256",
        "trace_items",
        "row_index",
    }
    trace_item_keys = {"trace_item_id", "prompt_sha256", "session_sha256"}
    outcome_keys = {
        "kind",
        "request_id",
        "trace_item_id",
        "family_id",
        "split",
        "alpha",
        "beta",
        "replica_id",
        "ttft_s",
        "ttft_ticks",
        "success",
        "prompt_sha256",
        "overlap_chunks",
        "prompt_chunks",
        "uncached_chunks",
        "virtual_queue_work_before",
        "row_index",
    }
    outcome_strings = {
        "kind",
        "request_id",
        "trace_item_id",
        "family_id",
        "split",
        "replica_id",
        "prompt_sha256",
    }
    outcome_floats = {"alpha", "beta", "ttft_s"}
    outcome_ints = {
        "ttft_ticks",
        "overlap_chunks",
        "prompt_chunks",
        "uncached_chunks",
        "virtual_queue_work_before",
        "row_index",
    }
    tuning_keys = {
        "type",
        "selected_alpha",
        "selected_beta",
        "baseline_alpha",
        "baseline_beta",
        "selection_input",
        "row_index",
    }
    for ordinal, row in enumerate(raw_rows[1:], 1):
        if row.get("type") == "family":
            _require(set(row) == family_keys, f"F2d family row {ordinal} schema differs")
            _require(
                all(
                    isinstance(row.get(field), str) and bool(row[field])
                    for field in (
                        "family_id",
                        "split",
                        "seed_replica_id",
                        "seed_prompt_sha256",
                        "prefix_fingerprint_sha256",
                    )
                ),
                f"F2d family row {ordinal} string fields differ",
            )
            trace_items = row.get("trace_items")
            _require(
                isinstance(trace_items, list)
                and bool(trace_items)
                and all(
                    isinstance(item, Mapping)
                    and set(item) == trace_item_keys
                    and all(isinstance(value, str) and bool(value) for value in item.values())
                    for item in trace_items
                ),
                f"F2d family row {ordinal} trace items differ",
            )
        elif row.get("kind") == "placement_outcome":
            _require(set(row) == outcome_keys, f"F2d outcome row {ordinal} schema differs")
            _require(
                all(
                    isinstance(row.get(field), str) and bool(row[field])
                    for field in outcome_strings
                )
                and all(
                    type(row.get(field)) is float
                    and math.isfinite(row[field])
                    and row[field] >= 0
                    and (row[field] != 0.0 or math.copysign(1.0, row[field]) > 0)
                    for field in outcome_floats
                )
                and all(type(row.get(field)) is int for field in outcome_ints)
                and row.get("success") is True,
                f"F2d outcome row {ordinal} field types differ",
            )
        elif row.get("type") == "tuning":
            _require(set(row) == tuning_keys, "F2d tuning row schema differs")
            selected = manifest.get("metrics")
            _require(isinstance(selected, Mapping), "F2d manifest metrics are missing")
            selected = selected.get("selected")
            config = manifest.get("config")
            _require(
                isinstance(selected, Mapping) and isinstance(config, Mapping),
                "F2d selected policy is missing",
            )
            expected_tuning = {
                "type": "tuning",
                "selected_alpha": selected.get("alpha"),
                "selected_beta": selected.get("beta"),
                "baseline_alpha": config.get("alpha"),
                "baseline_beta": config.get("baseline_lambda"),
                "selection_input": "train_only",
                "row_index": ordinal,
            }
            _require(
                all(
                    type(row.get(field)) is float
                    for field in (
                        "selected_alpha",
                        "selected_beta",
                        "baseline_alpha",
                        "baseline_beta",
                    )
                )
                and _same_json_value(row, expected_tuning),
                "F2d tuning row differs from the replayed policy",
            )
        else:
            raise ValueError(f"F2d raw row {ordinal} has an unknown schema")

    router_keys = {
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
    router_strings = {
        "kind",
        "session_sha256",
        "reason",
        "replica_id",
        "replica_generation",
        "request_id",
    }
    router_ints = {
        "replica",
        "placement_latency_ns",
        "placement_started_ns",
        "selected_at_ns",
        "pool_size",
        "eligible_size",
    }
    config = manifest["config"]
    assert isinstance(config, Mapping)
    for ordinal, row in enumerate(router_rows):
        _require(set(row) == router_keys, f"F2d router row {ordinal} schema differs")
        _require(
            all(isinstance(row.get(field), str) and bool(row[field]) for field in router_strings)
            and all(type(row.get(field)) is int for field in router_ints),
            f"F2d router row {ordinal} field types differ",
        )
        replica_id = row["replica_id"]
        assert isinstance(replica_id, str)
        _require(
            replica_id.startswith("replica-")
            and replica_id.removeprefix("replica-").isdigit()
            and row["replica"] == int(replica_id.removeprefix("replica-"))
            and row["pool_size"] == config.get("replicas")
            and row["eligible_size"] == config.get("replicas")
            and row["placement_started_ns"] >= 0
            and row["selected_at_ns"] >= row["placement_started_ns"]
            and row["placement_latency_ns"] == row["selected_at_ns"] - row["placement_started_ns"],
            f"F2d router row {ordinal} numeric relationships differ",
        )

    _validate_feature_numeric_types(
        list(raw_rows),
        exact_int_fields={
            "chunk_chars",
            "cpu_count",
            "f2b_final_eligible_count",
            "families",
            "heldout_families",
            "prefix_chunks",
            "queue_depth_threshold",
            "replicas",
            "requests",
            "requests_per_family",
            "seed",
            "train_families",
        },
        exact_float_fields={
            "alpha",
            "baseline_lambda",
            "virtual_tick_seconds",
        },
        exact_bool_fields={
            "all_pinned_digests_match",
            "expected_commit_matches",
            "git_clean_at_start",
        },
        exact_int_lists=set(),
        exact_float_lists={"lambdas"},
        exact_int_maps=set(),
        description="F2d parent raw",
    )


_F4B_ROW_KEYS = {
    "run": {
        "config",
        "container_lifecycle",
        "measurement_kind",
        "row_index",
        "row_type",
        "schema_version",
    },
    "shard_start": {
        "arm",
        "checkpoint",
        "cohort",
        "config",
        "container",
        "hardware",
        "input_raw_sha256",
        "measurement_kind",
        "measurement_started_at",
        "measurement_started_unix_ns",
        "output_path",
        "profile",
        "round",
        "row_index",
        "row_type",
        "run_id",
        "runtime",
        "schema_version",
        "source",
        "trace",
    },
    "request": {
        "events",
        "final_num_free_pages",
        "final_tier_stats",
        "initial_num_free_pages",
        "initial_tier_stats",
        "output_token_ids",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "prompt_tokens",
        "request_id",
        "retry_count",
        "row_index",
        "row_type",
        "sequence",
        "session",
        "status",
        "turn",
        "usage",
    },
    "shard_end": {
        "arm",
        "cohort",
        "errors",
        "final_tier_stats",
        "measurement_ended_at",
        "measurement_ended_unix_ns",
        "request_count",
        "round",
        "row_index",
        "row_type",
        "run_id",
        "source",
        "status",
    },
    "run_end": {"request_count", "row_index", "row_type", "shard_count"},
    "quality_run": {
        "config",
        "container_lifecycle",
        "measurement_kind",
        "parent_performance_raw_sha256",
        "row_index",
        "row_type",
        "schema_version",
    },
    "quality_shard_start": {
        "arm",
        "checkpoint",
        "cohort",
        "config",
        "container",
        "hardware",
        "input_raw_sha256",
        "measurement_kind",
        "measurement_started_at",
        "measurement_started_unix_ns",
        "output_path",
        "parent_performance_raw_sha256",
        "profile",
        "row_index",
        "row_type",
        "run_id",
        "runtime",
        "schema_version",
        "source",
        "trace",
    },
    "quality_request": {
        "final_tier_stats",
        "initial_tier_stats",
        "logprob_records",
        "output_token_ids",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "prompt_tokens",
        "request_id",
        "retry_count",
        "row_index",
        "row_type",
        "sequence",
        "session",
        "status",
        "turn",
        "usage",
    },
    "quality_shard_end": {
        "arm",
        "cohort",
        "errors",
        "final_tier_stats",
        "measurement_ended_at",
        "measurement_ended_unix_ns",
        "request_count",
        "row_index",
        "row_type",
        "run_id",
        "source",
        "status",
    },
    "quality_run_end": {"request_count", "row_index", "row_type", "shard_count"},
}

_F4B_EXACT_INT_FIELDS = {
    "Count",
    "ExitCode",
    "canonical_agent_tool_tokens_per_turn",
    "completion_tokens",
    "dram_kv_tier_capacity_pages",
    "dram_kv_tier_min_restore_tokens",
    "dram_pages_when_enabled",
    "eligible_size",
    "head_dim",
    "hbm_pages",
    "hidden_size",
    "intermediate_size",
    "logprobs",
    "max_model_len",
    "max_num_batched_tokens",
    "max_tokens",
    "min_restore_tokens",
    "min_tokens",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_pages",
    "output_count",
    "page_size",
    "pipeline_depth",
    "position",
    "prompt_tokens",
    "rank",
    "request_count",
    "retry_count",
    "round",
    "row_index",
    "seed",
    "selected_token_id",
    "sequence",
    "session",
    "sessions",
    "shard_count",
    "shared_prefix_tokens",
    "step_index",
    "tensor_parallel_size",
    "turn",
    "turn_tokens",
    "turns_per_session",
    "vocab_size",
}
_F4B_OPTIONAL_INT_FIELDS = {"dram_kv_tier_min_restore_tokens", "min_restore_tokens"}
_F4B_EXACT_FLOAT_FIELDS = {
    "logprob_abs_tolerance",
    "temperature",
    "tpot_noninferiority_ratio_limit",
}
_F4B_EXACT_BOOL_FIELDS = {
    "dirty",
    "ignore_eos",
    "timing_is_binding",
}


def _validate_f4b_parent_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    description: str,
) -> None:
    """Enforce type-sensitive fixed fields around the legacy F4b replay."""

    _require_exact_row_indices(rows, description=description)
    for ordinal, row in enumerate(rows):
        row_type = row.get("row_type")
        expected_keys = _F4B_ROW_KEYS.get(str(row_type))
        _require(
            expected_keys is not None and set(row) == expected_keys,
            f"{description}: row {ordinal} schema differs",
        )

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key)
                item_path = (*path, key)
                if key in _F4B_EXACT_INT_FIELDS:
                    _require(
                        (item is None and key in _F4B_OPTIONAL_INT_FIELDS) or type(item) is int,
                        f"{description}: {'.'.join(item_path)} must be an exact integer",
                    )
                elif key in _F4B_EXACT_FLOAT_FIELDS:
                    _require(
                        type(item) is float
                        and math.isfinite(item)
                        and (item != 0.0 or math.copysign(1.0, item) > 0),
                        f"{description}: {'.'.join(item_path)} must be an exact float",
                    )
                elif key in _F4B_EXACT_BOOL_FIELDS:
                    _require(
                        type(item) is bool,
                        f"{description}: {'.'.join(item_path)} must be a boolean",
                    )
                if key == "schema_version" and len(path) >= 1 and path[-1] == "container_lifecycle":
                    _require(
                        type(item) is int,
                        f"{description}: {'.'.join(item_path)} must be an exact integer",
                    )
                if key == "prompt_tokens_by_turn":
                    _require(
                        isinstance(item, list) and all(type(token) is int for token in item),
                        f"{description}: {'.'.join(item_path)} must contain exact integers",
                    )
                if key == "identity" and "container_lifecycle" in path:
                    _require(
                        isinstance(item, list)
                        and (
                            (
                                len(item) == 3
                                and type(item[0]) is int
                                and all(isinstance(part, str) for part in item[1:])
                            )
                            or (len(item) == 2 and all(isinstance(part, str) for part in item))
                        ),
                        f"{description}: {'.'.join(item_path)} has invalid scalar types",
                    )
                visit(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))

    visit(list(rows), ())


def _validate_feature_numeric_types(
    value: object,
    *,
    exact_int_fields: set[str],
    exact_float_fields: set[str],
    exact_bool_fields: set[str],
    exact_int_lists: set[str],
    exact_float_lists: set[str],
    exact_int_maps: set[str],
    description: str,
) -> None:
    def visit(item: object, path: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                child_path = (*path, key)
                if key in exact_int_fields:
                    _require(
                        type(child) is int,
                        f"{description}: {'.'.join(child_path)} must be an exact integer",
                    )
                elif key in exact_float_fields:
                    _require(
                        type(child) is float
                        and math.isfinite(child)
                        and (child != 0.0 or math.copysign(1.0, child) > 0),
                        f"{description}: {'.'.join(child_path)} must be an exact float",
                    )
                elif key in exact_bool_fields:
                    _require(
                        type(child) is bool,
                        f"{description}: {'.'.join(child_path)} must be a boolean",
                    )
                if key in exact_int_lists:
                    _require(
                        isinstance(child, list) and all(type(part) is int for part in child),
                        f"{description}: {'.'.join(child_path)} must contain exact integers",
                    )
                if key in exact_float_lists:
                    _require(
                        isinstance(child, list)
                        and all(
                            type(part) is float
                            and math.isfinite(part)
                            and (part != 0.0 or math.copysign(1.0, part) > 0)
                            for part in child
                        ),
                        f"{description}: {'.'.join(child_path)} must contain exact floats",
                    )
                if key in exact_int_maps:
                    _require(
                        isinstance(child, Mapping)
                        and all(type(part) is int for part in child.values()),
                        f"{description}: {'.'.join(child_path)} values must be exact integers",
                    )
                if key == "schema_version" and path and path[-1] == "runtime_attestation":
                    _require(
                        type(child) is int,
                        f"{description}: {'.'.join(child_path)} must be an exact integer",
                    )
                visit(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, (*path, str(index)))

    visit(value, ())


def _validate_f2c_parent_rows(rows: Sequence[Mapping[str, object]]) -> None:
    _require_exact_row_indices(rows, description="f2c parent raw")
    _validate_feature_numeric_types(
        list(rows),
        exact_int_fields={
            "cached_tokens",
            "completion_tokens",
            "cpu_count",
            "document_words",
            "done_ns",
            "eligible_size",
            "families_per_round",
            "family_index",
            "first_content_ns",
            "max_tokens",
            "min_tokens",
            "output_chars",
            "paired_receipt_skew_ns",
            "placement_latency_ns",
            "placement_started_ns",
            "planned_ns",
            "pool_size",
            "prompt_chars",
            "prompt_token_max",
            "prompt_token_min",
            "prompt_tokens",
            "prompt_word_count",
            "receipt_ns",
            "replica",
            "replica_ordinal",
            "request_count",
            "round_index",
            "rounds",
            "row_index",
            "schedule_lateness_ns",
            "seed",
            "selected_at_ns",
            "shared_prefix_chars",
            "ttft_ns",
            "turn",
        },
        exact_float_fields={
            "arrival_interval_s",
            "goodput_ratio_floor",
            "request_timeout_s",
            "temperature",
            "ttft_ratio_limit",
            "ttft_slo_s",
        },
        exact_bool_fields={
            "binding",
            "expected_commit_matches",
            "git_clean",
            "ignore_eos",
            "terminal",
        },
        exact_int_lists={"sched_affinity"},
        exact_float_lists=set(),
        exact_int_maps=set(),
        description="f2c parent raw",
    )


def _validate_f4a_parent_rows(rows: Sequence[Mapping[str, object]]) -> None:
    _require_exact_row_indices(rows, description="f4a parent raw")
    _validate_feature_numeric_types(
        list(rows),
        exact_int_fields={
            "bytes",
            "controller_ended_ns",
            "controller_started_ns",
            "destination_published_pages",
            "device",
            "elapsed_ns",
            "fill_value",
            "gpu_numa_node",
            "head_dim",
            "hidden_size",
            "host_numa_node",
            "host_os_page_size",
            "host_resident_pages",
            "host_slab_bytes",
            "intermediate_size",
            "local_numa_gpu_count",
            "max_model_len",
            "max_num_batched_tokens",
            "max_prefix_tokens",
            "measured_pairs_per_cell",
            "measured_sample_rows",
            "minimum_restore_wins",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_kv_heads",
            "num_layers",
            "os_page_size",
            "page_count",
            "page_size",
            "pair_index",
            "pairs",
            "pcie_current_link_width",
            "pcie_max_link_width",
            "prefix_tokens",
            "rank",
            "recompute_cache_hits",
            "retry_count",
            "sample_rows",
            "seed",
            "snapshot_rows",
            "source_device_owned_pages",
            "stream_id",
            "tensor_parallel_size",
            "tier_published_pages",
            "tier_reserved_pages",
            "tier_resident_pages",
            "tp_size",
            "vocab_size",
            "wall_elapsed_ns",
            "warmup_pairs_per_cell",
            "warmup_sample_rows",
        },
        exact_float_fields={
            "median_paired_restore_over_recompute_lt",
            "one_sided_sign_test_p_at_minimum",
        },
        exact_bool_fields={
            "all_zero",
            "async_submission",
            "chunked_prefill",
            "complete",
            "destination_published",
            "end_recorded",
            "fallback_used",
            "host_full_slab_local_coverage",
            "interpolation",
            "live_transfer",
            "pinned",
            "production_model_forward",
            "recompute_cache_disabled",
            "restored_from_dram",
            "source_device_invalidated",
            "source_device_reusable",
            "start_recorded",
            "synchronized",
            "tier_published",
            "warmup",
        },
        exact_int_lists={
            "destination_page_ids",
            "gpu_sm",
            "host_cpu_affinity",
            "prefix_lengths",
            "source_page_ids",
            "token_ids",
        },
        exact_float_lists=set(),
        exact_int_maps={"host_numa_page_counts"},
        description="f4a parent raw",
    )


def _parent_context(
    cell_id: str,
    manifest_path: Path,
    raw_path: Path,
) -> tuple[dict[str, object], int | None, dict[str, object] | None]:
    """Validate and content-bind one reviewed parent manifest/raw pair."""

    cell = _cell_spec(cell_id)
    feature = str(cell["feature_id"])
    spec = _spec(feature)
    manifest_path = _resolve_existing_path(
        manifest_path,
        description=f"{feature} parent manifest",
        directory=False,
    )
    raw_path = _resolve_existing_path(
        raw_path,
        description=f"{feature} parent raw",
        directory=False,
    )
    manifest_bytes = _read_exact_bytes(
        manifest_path,
        description=f"{feature} parent manifest",
    )
    raw_bytes = _read_exact_bytes(raw_path, description=f"{feature} parent raw")
    f2d_router_path: Path | None = None
    f2d_router_bytes: bytes | None = None
    sibling_profile_path: Path | None = None
    sibling_profile_bytes: bytes | None = None
    quality_manifest_path: Path | None = None
    quality_manifest_bytes: bytes | None = None
    quality_manifest: dict[str, object] | None = None
    quality_raw_path: Path | None = None
    quality_raw_bytes: bytes | None = None
    quality_rows: list[dict[str, object]] | None = None
    if feature == "f2d":
        f2d_router_path = _resolve_existing_path(
            manifest_path.parent / _F2D_ROUTER_NAME,
            description="f2d parent router raw",
            directory=False,
        )
        f2d_router_bytes = _read_exact_bytes(
            f2d_router_path,
            description="f2d parent router raw",
        )
    elif feature == "f4a":
        tensor_parallel_size = int(cell["tensor_parallel_size"])
        sibling_profile_path = _resolve_existing_path(
            manifest_path.parent / f"dram-kv-tier-qwen3-32b-tp{tensor_parallel_size}-profile.json",
            description=f"{cell_id} parent DRAM profile",
            directory=False,
        )
        sibling_profile_bytes = _read_exact_bytes(
            sibling_profile_path,
            description=f"{cell_id} parent DRAM profile",
        )
    elif feature == "f4b":
        quality_manifest_path = _resolve_existing_path(
            manifest_path.parent / _F4B_QUALITY_MANIFEST_NAME,
            description="f4b parent quality manifest",
            directory=False,
        )
        quality_raw_path = _resolve_existing_path(
            manifest_path.parent / _F4B_QUALITY_RAW_NAME,
            description="f4b parent quality raw",
            directory=False,
        )
        quality_manifest_bytes = _read_exact_bytes(
            quality_manifest_path,
            description="f4b parent quality manifest",
        )
        quality_raw_bytes = _read_exact_bytes(
            quality_raw_path,
            description="f4b parent quality raw",
        )
        quality_rows = _strict_jsonl_bytes(
            quality_raw_bytes,
            description=str(quality_raw_path),
        )
        quality_manifest = _strict_json_object_bytes(
            quality_manifest_bytes,
            description=str(quality_manifest_path),
        )
    manifest = _strict_json_object_bytes(
        manifest_bytes,
        description=str(manifest_path),
    )
    _verify_parent_artifact(
        feature=feature,
        manifest_path=manifest_path,
        supplied_manifest=manifest,
        supplied_quality_manifest=quality_manifest,
    )
    # Owner replay is path-based.  Require every input it could consume to be
    # byte-for-byte identical to the snapshots used below before trusting the
    # replay result or extracting any binding.
    _assert_exact_bytes(
        manifest_path,
        manifest_bytes,
        description=f"{feature} parent manifest",
    )
    _assert_exact_bytes(raw_path, raw_bytes, description=f"{feature} parent raw")
    if f2d_router_path is not None and f2d_router_bytes is not None:
        _assert_exact_bytes(
            f2d_router_path,
            f2d_router_bytes,
            description="f2d parent router raw",
        )
    if sibling_profile_path is not None and sibling_profile_bytes is not None:
        _assert_exact_bytes(
            sibling_profile_path,
            sibling_profile_bytes,
            description=f"{cell_id} parent DRAM profile",
        )
    if quality_manifest_path is not None and quality_manifest_bytes is not None:
        _assert_exact_bytes(
            quality_manifest_path,
            quality_manifest_bytes,
            description="f4b parent quality manifest",
        )
    if quality_raw_path is not None and quality_raw_bytes is not None:
        _assert_exact_bytes(
            quality_raw_path,
            quality_raw_bytes,
            description="f4b parent quality raw",
        )
        assert quality_rows is not None
        _validate_f4b_parent_rows(
            quality_rows,
            description="f4b parent quality raw",
        )
    # Owner replay keeps its established failure precedence.  Once every path
    # it could consume is proven byte-identical to our snapshots, parse the raw
    # snapshot once and enforce F2d's type-exact manifest/raw duplication.
    rows = _strict_jsonl_bytes(raw_bytes, description=str(raw_path))
    _require_exact_row_indices(rows, description=f"{feature} parent raw")
    if feature == "f2c":
        _validate_f2c_parent_rows(rows)
    elif feature == "f2d":
        assert f2d_router_path is not None
        assert f2d_router_bytes is not None
        router_rows = _strict_jsonl_bytes(
            f2d_router_bytes,
            description=str(f2d_router_path),
            canonical=False,
        )
        _validate_f2d_parent_rows(manifest, rows, router_rows)
    elif feature == "f4a":
        _validate_f4a_parent_rows(rows)
    else:
        assert feature == "f4b"
        _validate_f4b_parent_rows(rows, description="f4b parent performance raw")
    expected_gate = spec["parent_gate"]
    expected_schema = spec["parent_schema_version"]
    _require(
        manifest.get("schema_version") == expected_schema,
        f"{feature}: parent schema_version differs",
    )
    _require(manifest.get("passed") is True, f"{feature}: parent gate did not pass")
    # F4b predates a top-level ``gate`` field.  Every other parent requires
    # the exact gate field; F4b alone permits absence after exact schema
    # identification.  A conflicting value is never relabelled.
    parent_gate = manifest.get("gate")
    if feature == "f4b":
        _require(
            parent_gate is None or parent_gate == expected_gate,
            f"{feature}: parent gate differs",
        )
    else:
        _require(parent_gate == expected_gate, f"{feature}: parent gate differs")
    manifest_sha256 = gate.sha256_bytes(manifest_bytes)
    raw_sha256 = gate.sha256_bytes(raw_bytes)

    declared_raw = manifest.get("raw")
    declared_digests: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            digest = value.get("sha256")
            if isinstance(digest, str):
                declared_digests.add(digest)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(declared_raw)
    _require(
        raw_sha256 in declared_digests,
        f"{feature}: supplied parent raw is not bound by its manifest",
    )
    # Parent artifacts are canonical retained evidence too.  Reuse B7's
    # duplicate-key/framing/canonical JSONL reader instead of accepting a
    # merely parseable parent stream.
    (
        checkpoint_binding,
        parent_topology,
        minimum_restore_tokens,
        profile_descriptor,
    ) = _parent_checkpoint_binding(cell_id=cell_id, manifest=manifest, rows=rows)
    _require(
        _same_json_value(parent_topology, cell["parent_topology"]),
        f"{cell_id}: parent topology differs from the frozen cell",
    )
    _require(
        checkpoint_binding.get("binding_kind") == cell["parent_binding_kind"],
        f"{cell_id}: parent checkpoint binding kind differs",
    )
    if feature == "f4a":
        assert sibling_profile_path is not None
        assert sibling_profile_bytes is not None
        sibling_profile = _strict_json_object_bytes(
            sibling_profile_bytes,
            description=str(sibling_profile_path),
        )
        profile_descriptor = {
            "file_sha256": gate.sha256_bytes(sibling_profile_bytes),
            "value_sha256": gate.sha256_json(sibling_profile),
            "value": sibling_profile,
        }
    # Reject drift during semantic extraction as well as drift caused by the
    # owner verifier.  Evidence hashes and extracted semantics therefore bind
    # one immutable byte snapshot.
    _assert_exact_bytes(
        manifest_path,
        manifest_bytes,
        description=f"{feature} parent manifest",
    )
    _assert_exact_bytes(raw_path, raw_bytes, description=f"{feature} parent raw")
    if f2d_router_path is not None and f2d_router_bytes is not None:
        _assert_exact_bytes(
            f2d_router_path,
            f2d_router_bytes,
            description="f2d parent router raw",
        )
    if sibling_profile_path is not None and sibling_profile_bytes is not None:
        _assert_exact_bytes(
            sibling_profile_path,
            sibling_profile_bytes,
            description=f"{cell_id} parent DRAM profile",
        )
    if quality_manifest_path is not None and quality_manifest_bytes is not None:
        _assert_exact_bytes(
            quality_manifest_path,
            quality_manifest_bytes,
            description="f4b parent quality manifest",
        )
    if quality_raw_path is not None and quality_raw_bytes is not None:
        _assert_exact_bytes(
            quality_raw_path,
            quality_raw_bytes,
            description="f4b parent quality raw",
        )
    return (
        {
            "gate": expected_gate,
            "schema_version": expected_schema,
            "manifest_sha256": manifest_sha256,
            "raw_sha256": raw_sha256,
            "quality_manifest_sha256": (
                gate.sha256_bytes(quality_manifest_bytes)
                if quality_manifest_bytes is not None
                else None
            ),
            "quality_raw_sha256": (
                gate.sha256_bytes(quality_raw_bytes) if quality_raw_bytes is not None else None
            ),
            "parent_topology": parent_topology,
            "checkpoint_binding": checkpoint_binding,
        },
        minimum_restore_tokens,
        profile_descriptor,
    )


def _parent_evidence(
    cell_id: str,
    manifest_path: Path,
    raw_path: Path,
) -> dict[str, object]:
    """Return only the retained evidence part of :func:`_parent_context`."""

    evidence, _minimum_restore_tokens, _profile_descriptor = _parent_context(
        cell_id, manifest_path, raw_path
    )
    return evidence


def _live_checkpoint(model_path: Path) -> dict[str, object]:
    """Hash the complete pinned checkpoint before any GPU is constructed."""

    for name in (*_CHECKPOINT_METADATA, _WEIGHT_INDEX):
        _require((model_path / name).is_file(), f"checkpoint lacks {name}")

    config = _strict_json_object(model_path / "config.json")
    architecture_names = (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    )
    _require(
        all(name in config for name in architecture_names),
        "checkpoint config lacks the pinned Qwen architecture identity",
    )
    architecture = {name: config[name] for name in architecture_names}

    index = _strict_json_object(model_path / _WEIGHT_INDEX)
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, Mapping) and bool(weight_map), "weight map is missing")
    indexed_names = set(weight_map.values())
    _require(
        all(isinstance(name, str) and Path(name).name == name for name in indexed_names),
        "weight map contains an invalid shard name",
    )
    shard_paths = sorted(model_path.glob("*.safetensors"), key=lambda path: path.name)
    _require(len(shard_paths) == 17, "checkpoint must contain exactly 17 weight shards")
    _require(
        {path.name for path in shard_paths} == indexed_names,
        "weight index and checkpoint shard set differ",
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        shard_digests = list(pool.map(gate.sha256_file, shard_paths))
    weights = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(shard_paths, shard_digests, strict=True)
    ]
    # The retained F2c/F4a/F4b parents use json.dumps' historical default
    # separators for this rollup.  Keep both that parity digest and the B7
    # canonical list digest visible.
    pinned_weights_rollup = hashlib.sha256(
        json.dumps(weights, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint = {
        "model": _PINNED_MODEL,
        "revision": _PINNED_REVISION,
        "architecture": architecture,
        "weights": weights,
        "weights_sha256": gate.sha256_json(weights),
        "pinned_weights_rollup": pinned_weights_rollup,
        "metadata_sha256": {
            name: gate.sha256_file(model_path / name) for name in _CHECKPOINT_METADATA
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
            checkpoint["metadata_sha256"],
            gate.EXPECTED_METADATA_SHA256,
        )
        and checkpoint["weight_index_sha256"] == gate.EXPECTED_WEIGHT_INDEX_SHA256
        and checkpoint["model_config_sha256"] == gate.EXPECTED_MODEL_CONFIG_SHA256,
        "live checkpoint metadata differs from pinned Qwen3-32B",
    )
    return checkpoint


def _assert_checkpoint_constraint(
    checkpoint: Mapping[str, object],
    checkpoint_constraint: Mapping[str, object],
) -> None:
    """Reject a wrong model before tokenizer/backend construction."""

    kind = checkpoint_constraint.get("binding_kind")
    if kind == "aggregate-common-f2c":
        _require(
            checkpoint.get("model") == _PINNED_MODEL
            and checkpoint.get("revision") == _PINNED_REVISION,
            "F2d companion checkpoint differs from the pinned Qwen identity",
        )
        return
    if kind == "direct-partial":
        comparisons = {
            "model": checkpoint_constraint.get("model"),
            "revision": checkpoint_constraint.get("revision"),
            "pinned_weights_rollup": checkpoint_constraint.get("pinned_weights_rollup"),
        }
        _require(
            all(
                _same_json_value(checkpoint.get(name), value) for name, value in comparisons.items()
            ),
            "live checkpoint differs from the F2c parent identity",
        )
        return
    _require(kind == "direct-full", "parent checkpoint constraint kind differs")
    retained = checkpoint_constraint
    for name in (
        "model",
        "revision",
        "architecture",
        "model_config_sha256",
        "weights",
        "weights_sha256",
        "pinned_weights_rollup",
    ):
        _require(
            _same_json_value(checkpoint.get(name), retained.get(name)),
            f"live checkpoint {name} differs from the parent",
        )
    live_metadata = checkpoint.get("metadata_sha256")
    parent_metadata = retained.get("metadata_sha256")
    _require(
        isinstance(live_metadata, Mapping) and isinstance(parent_metadata, Mapping),
        "checkpoint metadata identity is missing",
    )
    _require(
        set(live_metadata) == set(parent_metadata) == set(_CHECKPOINT_METADATA),
        "live and parent checkpoint metadata file sets differ",
    )
    for name in _CHECKPOINT_METADATA:
        _require(
            live_metadata.get(name) == parent_metadata.get(name),
            f"live checkpoint {name} differs from the parent",
        )


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
        committed_bytes = subprocess.check_output(
            ("git", "show", f"HEAD:{path}"),
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot read {description} from HEAD: {path}") from error
    working_bytes = _read_exact_bytes(working_path, description=description)
    _require(
        working_bytes == committed_bytes,
        f"{description} bytes differ from HEAD: {path}",
    )
    return gate.sha256_bytes(working_bytes)


def _assert_loaded_repo_modules_head_identical() -> None:
    """Bind every already-loaded repo module, including lazy native imports."""

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
        origins: set[object] = {getattr(module, "__file__", None)}
        module_spec = getattr(module, "__spec__", None)
        origins.add(getattr(module_spec, "origin", None))
        concrete_origins = {
            origin
            for origin in origins
            if isinstance(origin, str) and origin not in {"built-in", "frozen"}
        }
        _require(
            not checkout_owned or bool(concrete_origins),
            f"checkout-owned module has no concrete origin: {module_name}",
        )
        for origin in concrete_origins:
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
                not checkout_owned or lexical_relative is not None or resolved_relative is not None,
                f"checkout-owned module loaded outside the repository: {module_name}",
            )
            if lexical_relative is not None and resolved_relative is None:
                raise ValueError(f"loaded repo module escapes through a symlink: {module_name}")
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
    """Return a fail-closed clean-source identity for the native child."""

    try:
        git_head = subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tracked_tree_clean = (
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
        raise ValueError("cannot establish the native source identity") from error
    _require(
        len(git_head) == 40 and all(character in "0123456789abcdef" for character in git_head),
        "git HEAD is not a full lowercase commit identity",
    )
    _require(tracked_tree_clean, "native evidence requires a clean tracked source tree")
    _assert_no_import_shadows()
    required_paths = _RUNTIME_SOURCE_PATHS
    file_sha256: dict[str, str] = {}
    for path in required_paths:
        file_sha256[path] = _head_identical_source(
            path,
            description="required source path",
        )
    _assert_loaded_repo_modules_head_identical()
    return {
        "git_head": git_head,
        "tracked_tree_clean": True,
        "file_sha256": file_sha256,
        "files_sha256": gate.sha256_json(file_sha256),
    }


def _verify_parent_artifact(
    *,
    feature: str,
    manifest_path: Path,
    supplied_manifest: Mapping[str, object],
    supplied_quality_manifest: Mapping[str, object] | None,
) -> None:
    """Run the owning checkout verifier before trusting a parent verdict."""

    if feature == "f2c":
        from bench import kv_aware_ttft_f2c_bench as owner

        recomputed = owner.verify_artifact(manifest_path, assert_gate=True)
        _require(
            _same_json_value(recomputed, supplied_manifest),
            "F2c parent differs from owner replay",
        )
        return
    if feature == "f2d":
        from bench import prefix_weight_f2d_bench as owner

        recomputed = owner.verify_artifact(manifest_path, assert_gate=True)
        _require(recomputed.get("passed") is True, "F2d parent owner replay did not pass")
        config_value = supplied_manifest.get("config")
        _require(isinstance(config_value, Mapping), "F2d parent config is missing")
        profile = config_value.get("profile")
        _require(profile == "formal", "F2d parent is not the formal profile")
        config = owner.profile_config(profile)
        expected_config = owner._config_dict(config)
        _require(
            _same_json_value(config_value, expected_config),
            "F2d parent config types differ from the frozen profile",
        )
        for descriptor_name, expected_path in (
            ("raw", owner.RAW_NAME),
            ("router", owner.ROUTER_NAME),
        ):
            descriptor = supplied_manifest.get(descriptor_name)
            _require(
                isinstance(descriptor, Mapping) and set(descriptor) == {"path", "sha256", "rows"},
                f"F2d parent {descriptor_name} descriptor differs",
            )
            _require(
                descriptor.get("path") == expected_path
                and type(descriptor.get("rows")) is int
                and descriptor["rows"] > 0,
                f"F2d parent {descriptor_name} descriptor types differ",
            )
        _require(
            _same_json_value(supplied_manifest.get("inputs"), owner.input_descriptor(config)),
            "F2d parent inputs differ from owner replay",
        )
        _require(
            _same_json_value(supplied_manifest.get("trace"), owner.trace_descriptor(config)),
            "F2d parent trace differs from owner replay",
        )
        _require(
            _same_json_value(supplied_manifest.get("metrics"), recomputed.get("metrics")),
            "F2d parent metrics differ from owner replay",
        )
        owner_checks = recomputed.get("checks")
        gate_checks = supplied_manifest.get("gate_checks")
        _require(
            isinstance(owner_checks, Mapping)
            and isinstance(gate_checks, Mapping)
            and set(gate_checks) == set(owner._GATE_CHECK_NAMES)
            and all(type(value) is bool for value in gate_checks.values()),
            "F2d parent gate checks have invalid types",
        )
        replayed_gate_checks = {name: owner_checks.get(name) for name in owner._GATE_CHECK_NAMES}
        _require(
            _same_json_value(gate_checks, replayed_gate_checks),
            "F2d parent gate checks differ from owner replay",
        )
        _require(
            supplied_manifest.get("pass_contract") == owner._PASS_CONTRACT
            and supplied_manifest.get("passed") is True,
            "F2d parent pass contract differs",
        )
        return
    if feature == "f4a":
        from bench import dram_kv_tier_qwen as owner

        recomputed = owner.verify_artifact(manifest_path.parent, assert_gate=True)
        _require(
            _same_json_value(recomputed, supplied_manifest),
            "F4a parent differs from owner replay",
        )
        return
    _require(feature == "f4b", f"unsupported parent owner: {feature}")
    from bench import agentic_kv_tier_f4b_bench as owner

    recomputed = owner.verify_artifact(manifest_path, assert_gate=True)
    _require(
        _same_json_value(recomputed, supplied_manifest),
        "F4b parent differs from owner replay",
    )
    sealed = owner.verify_quality_artifact(
        manifest_path.parent,
        assert_gate=True,
    )
    _require(
        isinstance(sealed, Mapping) and sealed.get("passed") is True,
        "F4b sealed performance+quality parent did not pass",
    )
    _require(
        _same_json_value(sealed.get("performance"), supplied_manifest),
        "F4b quality closure refers to different performance evidence",
    )
    quality = sealed.get("quality")
    _require(
        isinstance(quality, Mapping) and quality.get("passed") is True,
        "F4b quality parent did not pass",
    )
    _require(
        supplied_quality_manifest is not None
        and _same_json_value(quality, supplied_quality_manifest),
        "F4b quality manifest differs from owner replay",
    )


def _validate_dram_profile(
    *,
    cell_id: str,
    profile_path: Path,
    checkpoint: Mapping[str, object],
    parent_evidence: Mapping[str, object],
    expected_min_restore_tokens: int | None,
    expected_parent_profile: Mapping[str, object] | None,
    profile_bytes: bytes | None = None,
) -> dict[str, object]:
    """Bind the supplied runtime policy to the selected parent and live model."""

    profile_path = _resolve_existing_path(
        profile_path,
        description=f"{cell_id} DRAM profile",
        directory=False,
    )
    retained_bytes = (
        _read_exact_bytes(profile_path, description=f"{cell_id} DRAM profile")
        if profile_bytes is None
        else profile_bytes
    )
    _assert_exact_bytes(
        profile_path,
        retained_bytes,
        description=f"{cell_id} DRAM profile",
    )
    profile = _strict_json_object_bytes(
        retained_bytes,
        description=str(profile_path),
    )
    _require(
        profile.get("schema_version") == "kairyu.dram-kv-crossover.v2",
        f"{cell_id}: DRAM profile schema differs",
    )
    _require(profile.get("passed") is True, f"{cell_id}: DRAM profile did not pass")
    runtime_identity = profile.get("runtime_identity")
    policy = profile.get("policy")
    _require(
        isinstance(runtime_identity, Mapping),
        f"{cell_id}: DRAM profile runtime identity is missing",
    )
    _require(isinstance(policy, Mapping), f"{cell_id}: DRAM profile policy is missing")
    cell = _cell_spec(cell_id)
    tensor_parallel_size = int(cell["tensor_parallel_size"])
    _require(
        _same_json_value(
            runtime_identity.get("tensor_parallel_size"),
            tensor_parallel_size,
        ),
        f"{cell_id}: DRAM profile TP size differs",
    )
    _require(
        runtime_identity.get("model_config_sha256") == checkpoint.get("model_config_sha256"),
        f"{cell_id}: DRAM profile model config differs",
    )
    _require(
        _same_json_value(
            policy.get("min_restore_tokens"),
            cell["dram_min_restore_tokens"],
        ),
        f"{cell_id}: DRAM profile restore threshold differs from frozen topology",
    )
    _require(
        expected_parent_profile is not None,
        f"{cell_id}: parent profile descriptor is missing",
    )
    _require(
        gate.sha256_bytes(retained_bytes) == expected_parent_profile.get("file_sha256"),
        f"{cell_id}: supplied profile file digest differs from the parent",
    )
    _require(
        _same_json_value(profile, expected_parent_profile.get("value")),
        f"{cell_id}: supplied profile value differs from the parent",
    )
    _require(
        gate.sha256_json(profile) == expected_parent_profile.get("value_sha256"),
        f"{cell_id}: supplied profile value digest differs from the parent",
    )
    if cell_id.startswith("f4a-"):
        _require(
            expected_min_restore_tokens is not None
            and _same_json_value(
                policy.get("min_restore_tokens"),
                expected_min_restore_tokens,
            ),
            f"{cell_id}: DRAM profile restore threshold differs from F4a",
        )
        expected_evidence_sha256 = gate.sha256_json(
            {
                "manifest_sha256": parent_evidence["manifest_sha256"],
                "raw_sha256": parent_evidence["raw_sha256"],
            }
        )
        _require(
            profile.get("evidence_sha256") == expected_evidence_sha256,
            f"{cell_id}: DRAM profile is not bound to the selected F4a parent raw",
        )
    _assert_exact_bytes(
        profile_path,
        retained_bytes,
        description=f"{cell_id} DRAM profile",
    )
    return profile


def _prompt_descriptor(token_ids: Sequence[int]) -> dict[str, object]:
    copied = [int(token_id) for token_id in token_ids]
    return {
        "token_ids": copied,
        "token_ids_sha256": gate.sha256_json(copied),
        "token_count": len(copied),
    }


def _stable_token_pieces(
    tokenizer: object,
    *,
    required: int,
) -> tuple[tuple[int, str], ...]:
    """Find deterministic, mutually disjoint, repeat-stable token pieces."""

    vocab = tokenizer.vocab()
    selected: list[tuple[int, str]] = []
    seen: set[str] = set()
    for token_id in range(len(vocab)):
        piece = tokenizer.decode((token_id,))
        if (
            not isinstance(piece, str)
            or not piece
            or piece in seen
            or not piece.startswith(" ")
            or not piece.strip()
            or not piece.isprintable()
            or "\ufffd" in piece
        ):
            continue
        if tuple(tokenizer.encode(piece)) != (token_id,):
            continue
        if tuple(tokenizer.encode(piece * 4)) != (token_id,) * 4:
            continue
        seen.add(piece)
        selected.append((token_id, piece))
        if len(selected) == required:
            return tuple(selected)
    raise ValueError(
        f"tokenizer exposes {len(selected)} repeat-stable token pieces; need {required}"
    )


def _tier_stats(cache: object) -> dict[str, int]:
    value = cache.dram_tier_stats
    _require(isinstance(value, Mapping), "native DRAM tier stats are missing")
    _require(
        set(value) == set(gate.TIER_STATS_FIELDS),
        "native DRAM tier stats fields differ",
    )
    result: dict[str, int] = {}
    for field in gate.TIER_STATS_FIELDS:
        item = value[field]
        _require(type(item) is int and item >= 0, f"invalid DRAM tier stat {field}")
        result[field] = item
    return result


def _cached_prefix_probe(
    probe: Callable[[tuple[int, ...]], object],
    prompt_token_ids: tuple[int, ...],
) -> int:
    value = probe(prompt_token_ids)
    _require(
        type(value) is int and 0 <= value <= len(prompt_token_ids),
        "native cache-residency probe returned an invalid token count",
    )
    return value


def _usage_dict(value: object, *, prompt_tokens: int) -> dict[str, int]:
    if value is None:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
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
    request_factory: Callable[[str, tuple[int, ...]], object],
    cell_id: str,
    phase: str,
    ordinal: int,
    prompt_token_ids: tuple[int, ...],
    runtime_nonce: str,
) -> dict[str, object]:
    """Run one native request and retain a schema-complete failure if it errs."""

    request_id = f"kv-equivalence-{cell_id}-{phase}-{ordinal}-{uuid.uuid4().hex}"
    error_type: str | None = None
    try:
        result = await backend.generate(request_factory(request_id, prompt_token_ids))
        _require(result.finished is True, "native request did not finish")
        _require(len(result.completions) == 1, "native request did not return one completion")
        processed_prompt_token_ids = result.prompt_token_ids
        _require(
            isinstance(processed_prompt_token_ids, (list, tuple))
            and all(type(token_id) is int for token_id in processed_prompt_token_ids)
            and tuple(processed_prompt_token_ids) == prompt_token_ids,
            "native backend processed different prompt token ids",
        )
        completion = result.completions[0]
        native_token_ids = completion.token_ids
        _require(
            isinstance(native_token_ids, (list, tuple))
            and all(type(token_id) is int for token_id in native_token_ids),
            "native output token ids have invalid types",
        )
        token_ids = list(native_token_ids)
        vocab = tokenizer.vocab()
        _require(
            all(0 <= token_id < len(vocab) for token_id in token_ids),
            "native output token id is outside the tokenizer vocabulary",
        )
        token_pieces = [vocab[token_id] for token_id in token_ids]
        _require(
            all(isinstance(piece, str) and bool(piece) for piece in token_pieces),
            "native output token piece is invalid",
        )
        response = {
            "status": "success",
            "token_ids": token_ids,
            "token_pieces": token_pieces,
            "text": completion.text,
            "finish_reason": completion.finish_reason,
            "usage": _usage_dict(result.usage, prompt_tokens=len(prompt_token_ids)),
        }
    except Exception as error:  # retain FAIL evidence; never turn it into PASS
        error_type = type(error).__name__
        response = {
            "status": "error",
            "token_ids": [],
            "token_pieces": [],
            "text": "",
            "finish_reason": None,
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": 0,
                "cached_tokens": 0,
            },
        }
    return {
        "phase": phase,
        "runtime_nonce": runtime_nonce,
        "prompt_token_ids": list(prompt_token_ids),
        "response": response,
        "error_type": error_type,
        "retry_count": 0,
    }


def _proof(
    *,
    feature: str,
    observations: Sequence[Mapping[str, object]],
    tier_before: Mapping[str, int] | None,
    tier_pre_warm: Mapping[str, int] | None,
    tier_after: Mapping[str, int] | None,
    target_cached_tokens_pre_warm: int | None,
) -> dict[str, object]:
    spec = _spec(feature)
    kind = spec["proof_kind"]
    if kind == "radix_reuse":
        _require(
            target_cached_tokens_pre_warm is not None,
            f"{feature}: native radix pre-warm cache proof is missing",
        )
        runtime_nonces = {row["runtime_nonce"] for row in observations}
        return {
            "kind": "radix_reuse",
            "same_runtime": len(runtime_nonces) == 1,
            "cache_hit_tokens": target_cached_tokens_pre_warm,
        }
    _require(kind == "dram_restore", f"{feature}: unknown proof kind")
    assert (
        tier_before is not None
        and tier_pre_warm is not None
        and tier_after is not None
        and target_cached_tokens_pre_warm is not None
    )
    before = {field: int(tier_before[field]) for field in gate.TIER_STATS_FIELDS}
    pre_warm = {field: int(tier_pre_warm[field]) for field in gate.TIER_STATS_FIELDS}
    after = {field: int(tier_after[field]) for field in gate.TIER_STATS_FIELDS}
    return {
        "kind": "dram_restore",
        "target_cached_tokens_pre_warm": target_cached_tokens_pre_warm,
        "before": before,
        "pre_warm": pre_warm,
        "after": after,
        "pressure_delta": {
            field: pre_warm[field] - before[field] for field in gate.TIER_STATS_FIELDS
        },
        "warm_delta": {field: after[field] - pre_warm[field] for field in gate.TIER_STATS_FIELDS},
    }


def _capture_to_rows(
    *,
    run_id: str,
    cell_id: str,
    feature: str,
    parent_evidence: Mapping[str, object],
    runtime: Mapping[str, object],
    sampling: Mapping[str, object],
    main_prompt_token_ids: Sequence[int],
    observations: Sequence[Mapping[str, object]],
    feature_proof: Mapping[str, object],
) -> list[dict[str, object]]:
    """Pure conversion seam used by CPU tests and the real native runner."""

    main_prompt = _prompt_descriptor(main_prompt_token_ids)
    rows: list[dict[str, object]] = [
        {
            "row_type": "cell_start",
            "run_id": run_id,
            "cell_id": cell_id,
            "feature_id": feature,
            "transition": _spec(feature)["transition"],
            "prompt": main_prompt,
            "sampling": dict(sampling),
            "parent_evidence": dict(parent_evidence),
            "runtime": dict(runtime),
        }
    ]
    for observation in observations:
        prompt_ids = observation["prompt_token_ids"]
        assert isinstance(prompt_ids, Sequence)
        rows.append(
            {
                "row_type": "request",
                "run_id": run_id,
                "cell_id": cell_id,
                "feature_id": feature,
                "phase": observation["phase"],
                "runtime_nonce": observation["runtime_nonce"],
                "prompt": _prompt_descriptor(prompt_ids),
                "response": dict(observation["response"]),
                "error_type": observation["error_type"],
                "retry_count": observation["retry_count"],
            }
        )
    rows.append(
        {
            "row_type": "cell_end",
            "run_id": run_id,
            "cell_id": cell_id,
            "feature_id": feature,
            "feature_proof": dict(feature_proof),
        }
    )
    return rows


def _capture_passed(rows: Sequence[Mapping[str, object]]) -> bool:
    try:
        return gate.validate_single_cell(rows)["passed"] is True
    except (gate.EvidenceError, KeyError, TypeError, ValueError):
        return False


def _prompt_token_count(
    *,
    feature: str,
    page_size: int,
    num_pages: int,
    minimum_restore_tokens: int,
) -> int:
    comparison_tokens = gate.COMPARISON_PROMPT_TOKENS
    _require(
        comparison_tokens > 0 and comparison_tokens % page_size == 0,
        "fixed comparison prompt is not page aligned",
    )
    output_pages = math.ceil(gate.OUTPUT_TOKENS / page_size)
    maximum_prompt_pages = num_pages - output_pages - 1
    comparison_pages = comparison_tokens // page_size
    _require(
        comparison_pages <= maximum_prompt_pages,
        "device KV capacity cannot fit the fixed comparison prompt",
    )
    if feature in {"f4a", "f4b"}:
        _require(
            minimum_restore_tokens > 0 and minimum_restore_tokens % page_size == 0,
            "DRAM profile restore threshold is not page aligned",
        )
        _require(
            minimum_restore_tokens <= comparison_tokens,
            "DRAM restore threshold exceeds the fixed comparison prompt",
        )
    else:
        _require(
            minimum_restore_tokens == 0,
            "radix-only cell has an unexpected DRAM restore threshold",
        )
    return comparison_tokens


async def _capture_backend(
    *,
    cell_id: str,
    backend: object,
    tokenizer: object,
    request_factory: Callable[[str, tuple[int, ...]], object],
    parent_evidence: Mapping[str, object],
    runtime: Mapping[str, object],
    sampling: Mapping[str, object],
    page_size: int,
    num_pages: int,
) -> list[dict[str, object]]:
    """Capture one feature using a supplied real or fake native backend."""

    feature = str(_cell_spec(cell_id)["feature_id"])
    tiered = feature in {"f4a", "f4b"}
    minimum_restore_tokens = (
        int(getattr(backend, "dram_kv_tier_min_restore_tokens", 0)) if tiered else 0
    )
    prompt_tokens = _prompt_token_count(
        feature=feature,
        page_size=page_size,
        num_pages=num_pages,
        minimum_restore_tokens=minimum_restore_tokens,
    )
    prompt_pages = prompt_tokens // page_size
    pressure_limit = min(
        _MAX_PRESSURE_REQUESTS,
        max(2, math.ceil(num_pages / prompt_pages) + 2),
    )
    pieces = _stable_token_pieces(
        tokenizer,
        required=pressure_limit + 1 if tiered else 1,
    )
    main_prompt = (pieces[0][0],) * prompt_tokens
    run_id = f"kv-equivalence-{cell_id}-{uuid.uuid4().hex}"
    runtime_nonce = runtime.get("run_nonce")
    _require(
        isinstance(runtime_nonce, str) and bool(runtime_nonce),
        f"{cell_id}: runtime nonce is missing",
    )
    observations: list[dict[str, object]] = []
    tier_before: dict[str, int] | None = None
    tier_pre_warm: dict[str, int] | None = None
    target_cached_tokens_pre_warm: int | None = None
    observations.append(
        await _request_observation(
            backend=backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            cell_id=cell_id,
            phase="cold",
            ordinal=0,
            prompt_token_ids=main_prompt,
            runtime_nonce=runtime_nonce,
        )
    )
    if tiered:
        # The causal pressure interval starts after the cold request.  A cold
        # request's unexpected tier activity therefore cannot masquerade as
        # pressure-induced offload.
        tier_before = _tier_stats(backend._cache)
        peek_cached_tokens = getattr(backend._cache, "peek_cached_tokens", None)
        _require(
            callable(peek_cached_tokens),
            "native radix cache cannot prove target HBM eviction",
        )
        for ordinal in range(1, pressure_limit + 1):
            current = _tier_stats(backend._cache)
            if (
                current["offload_pages"] > tier_before["offload_pages"]
                and _cached_prefix_probe(peek_cached_tokens, main_prompt) == 0
            ):
                break
            pressure_prompt = (pieces[ordinal][0],) * prompt_tokens
            observations.append(
                await _request_observation(
                    backend=backend,
                    tokenizer=tokenizer,
                    request_factory=request_factory,
                    cell_id=cell_id,
                    phase="pressure",
                    ordinal=ordinal,
                    prompt_token_ids=pressure_prompt,
                    runtime_nonce=runtime_nonce,
                )
            )
        tier_pre_warm = _tier_stats(backend._cache)
        target_cached_tokens_pre_warm = _cached_prefix_probe(
            peek_cached_tokens,
            main_prompt,
        )
    else:
        peek_cached_tokens = getattr(backend._cache, "peek_cached_tokens", None)
        _require(
            callable(peek_cached_tokens),
            "native radix cache cannot prove the target prefix residency",
        )
        target_cached_tokens_pre_warm = _cached_prefix_probe(
            peek_cached_tokens,
            main_prompt,
        )
    observations.append(
        await _request_observation(
            backend=backend,
            tokenizer=tokenizer,
            request_factory=request_factory,
            cell_id=cell_id,
            phase="warm",
            ordinal=len(observations),
            prompt_token_ids=main_prompt,
            runtime_nonce=runtime_nonce,
        )
    )
    tier_after = _tier_stats(backend._cache) if tiered else None
    feature_proof = _proof(
        feature=feature,
        observations=observations,
        tier_before=tier_before,
        tier_pre_warm=tier_pre_warm,
        tier_after=tier_after,
        target_cached_tokens_pre_warm=target_cached_tokens_pre_warm,
    )
    return _capture_to_rows(
        run_id=run_id,
        cell_id=cell_id,
        feature=feature,
        parent_evidence=parent_evidence,
        runtime=runtime,
        sampling=sampling,
        main_prompt_token_ids=main_prompt,
        observations=observations,
        feature_proof=feature_proof,
    )


async def _run_native(args: argparse.Namespace) -> tuple[list[dict[str, object]], bool]:
    cell_id = args.cell
    cell = _cell_spec(cell_id)
    feature = str(cell["feature_id"])
    tiered = feature in {"f4a", "f4b"}
    model_path = _resolve_existing_path(
        args.model_path,
        description="model path",
        directory=True,
    )
    parent_manifest = _resolve_existing_path(
        args.parent_manifest,
        description="parent manifest",
        directory=False,
    )
    parent_raw = _resolve_existing_path(
        args.parent_raw,
        description="parent raw",
        directory=False,
    )
    if tiered:
        _require(args.dram_profile is not None, f"{feature} requires --dram-profile")
        dram_profile = _resolve_existing_path(
            args.dram_profile,
            description="DRAM profile",
            directory=False,
        )
    else:
        _require(args.dram_profile is None, f"{feature} does not accept --dram-profile")
        dram_profile = None

    parent, expected_min_restore_tokens, expected_parent_profile = _parent_context(
        cell_id, parent_manifest, parent_raw
    )
    checkpoint = _live_checkpoint(model_path)
    checkpoint_constraint = parent["checkpoint_binding"]
    _require(
        isinstance(checkpoint_constraint, Mapping),
        f"{cell_id}: parent checkpoint constraint is missing",
    )
    _assert_checkpoint_constraint(checkpoint, checkpoint_constraint)
    source = _source_snapshot()
    if tiered:
        assert dram_profile is not None
        profile_bytes = _read_exact_bytes(
            dram_profile,
            description=f"{cell_id} DRAM profile",
        )
        profile = _validate_dram_profile(
            cell_id=cell_id,
            profile_path=dram_profile,
            checkpoint=checkpoint,
            parent_evidence=parent,
            expected_min_restore_tokens=expected_min_restore_tokens,
            expected_parent_profile=expected_parent_profile,
            profile_bytes=profile_bytes,
        )
        dram_profile_sha256 = gate.sha256_bytes(profile_bytes)
    else:
        profile = None
        profile_bytes = None
        dram_profile_sha256 = None
    _require(
        dram_profile_sha256 == cell["dram_profile_sha256"],
        f"{cell_id}: supplied DRAM profile differs from the frozen topology",
    )

    engine = {
        "cell_id": cell_id,
        "tensor_parallel_size": int(cell["tensor_parallel_size"]),
        "page_size": int(cell["page_size"]),
        "num_pages": int(cell["num_pages"]),
        "decode_mode": str(cell["decode_mode"]),
        "max_num_batched_tokens": int(cell["max_num_batched_tokens"]),
        "max_model_len": int(cell["max_model_len"]),
        "dram_profile_sha256": dram_profile_sha256,
        "dram_capacity_pages": int(cell["dram_capacity_pages"]),
        "dram_min_restore_tokens": int(cell["dram_min_restore_tokens"]),
    }
    runtime = {
        "run_nonce": uuid.uuid4().hex,
        "source": source,
        "checkpoint": checkpoint,
        "engine": engine,
    }

    # Lazy by contract: these imports may initialize native/CUDA-facing code.
    from kairyu.engine.backend import GenerationRequest
    from kairyu.engine.kairyu_backend import KairyuBackend
    from kairyu.engine.prompt import TokensPrompt
    from kairyu.engine.tokenizer import resolve_tokenizer
    from kairyu.sampling_params import SamplingParams

    config = gate.frozen_config()
    sampling = config["sampling"]
    _require(isinstance(sampling, Mapping), "frozen sampling config is invalid")
    params = SamplingParams(**sampling)
    tokenizer = resolve_tokenizer(str(model_path))
    backend = KairyuBackend(
        model_path=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=engine["tensor_parallel_size"],
        num_pages=engine["num_pages"],
        page_size=engine["page_size"],
        max_num_batched_tokens=engine["max_num_batched_tokens"],
        max_num_seqs=1,
        max_model_len=engine["max_model_len"],
        pipeline_depth=1,
        decode_mode=engine["decode_mode"],
        dram_kv_tier_capacity_pages=engine["dram_capacity_pages"],
        dram_kv_tier_profile=dram_profile,
    )
    try:
        _require(
            _same_json_value(
                backend.tensor_parallel_size,
                engine["tensor_parallel_size"],
            ),
            "native backend TP size differs from the frozen cell",
        )
        _require(
            _same_json_value(backend._cache.page_size, engine["page_size"])
            and _same_json_value(backend._cache.num_pages, engine["num_pages"]),
            "native backend KV geometry differs from the frozen cell",
        )
        _require(
            _same_json_value(
                backend._scheduler._budget,
                engine["max_num_batched_tokens"],
            )
            and _same_json_value(
                backend._loop.max_model_len,
                engine["max_model_len"],
            ),
            "native backend scheduling geometry differs from the frozen cell",
        )
        if tiered:
            _require(backend.dram_kv_tier_enabled is True, "native DRAM tier is disabled")
            _require(
                _same_json_value(
                    backend.dram_kv_tier_capacity_pages,
                    engine["dram_capacity_pages"],
                ),
                "native DRAM tier capacity differs",
            )
            _require(
                backend.dram_kv_tier_profile_sha256 == dram_profile_sha256,
                "native DRAM tier profile digest differs",
            )
            assert isinstance(profile, Mapping)
            policy = profile["policy"]
            assert isinstance(policy, Mapping)
            _require(
                _same_json_value(
                    backend.dram_kv_tier_min_restore_tokens,
                    policy["min_restore_tokens"],
                )
                and _same_json_value(
                    policy["min_restore_tokens"],
                    engine["dram_min_restore_tokens"],
                ),
                "native DRAM tier restore threshold differs",
            )
        else:
            _require(
                backend.dram_kv_tier_enabled is False
                and backend.dram_kv_tier_capacity_pages == 0
                and backend.dram_kv_tier_profile_sha256 is None,
                "radix-only cell unexpectedly enabled the DRAM tier",
            )
        rows = await _capture_backend(
            cell_id=cell_id,
            backend=backend,
            tokenizer=tokenizer,
            request_factory=lambda request_id, token_ids: GenerationRequest(
                request_id,
                TokensPrompt(token_ids),
                params,
            ),
            parent_evidence=parent,
            runtime=runtime,
            sampling=sampling,
            page_size=int(engine["page_size"]),
            num_pages=int(engine["num_pages"]),
        )
    finally:
        try:
            await backend.shutdown()
        finally:
            # Evidence is emitted only when the exact source, checkpoint, and
            # runtime policy observed before native imports remain unchanged
            # after the backend has fully shut down.  This applies equally to
            # successful capture and validation/error paths.
            source_after = _source_snapshot()
            _require(
                _same_json_value(source_after, source),
                "native source identity changed during capture",
            )
            checkpoint_after = _live_checkpoint(model_path)
            _require(
                _same_json_value(checkpoint_after, checkpoint),
                "live checkpoint identity changed during capture",
            )
            if dram_profile is not None and profile_bytes is not None:
                _assert_exact_bytes(
                    dram_profile,
                    profile_bytes,
                    description=f"{cell_id} DRAM profile",
                )
    return rows, _capture_passed(rows)


def _read_capture(path: Path, cell_id: str) -> list[dict[str, object]]:
    rows = gate.read_jsonl(path)
    _require(len(rows) >= 3, f"{cell_id}: capture is incomplete")
    _require(rows[0].get("row_type") == "cell_start", f"{cell_id}: missing cell_start")
    _require(rows[-1].get("row_type") == "cell_end", f"{cell_id}: missing cell_end")
    _require(
        all(row.get("cell_id") == cell_id for row in rows),
        f"{cell_id}: capture cell identity differs",
    )
    gate.validate_single_cell(rows, expected_cell_id=cell_id)
    return rows


def _assemble_rows(paths: Mapping[str, Path]) -> list[dict[str, object]]:
    run_id = f"kv-answer-equivalence-{uuid.uuid4().hex}"
    rows: list[dict[str, object]] = [
        {
            "row_type": "run",
            "schema_version": gate.SCHEMA_VERSION,
            "gate": gate.GATE,
            "run_id": run_id,
            "required_cells": list(gate.CELL_IDS),
            "config": gate.frozen_config(),
        }
    ]
    for cell_id in gate.CELL_IDS:
        for child in _read_capture(paths[cell_id], cell_id):
            rebound = dict(child)
            rebound["run_id"] = run_id
            rows.append(rebound)
    rows.append(
        {
            "row_type": "run_end",
            "run_id": run_id,
            "status": "complete",
            "errors": [],
            "cell_count": len(gate.CELL_IDS),
        }
    )
    return rows


def _artifact_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / RAW_NAME, path / MANIFEST_NAME
    if path.name == RAW_NAME:
        return path, path.with_name(MANIFEST_NAME)
    if path.name == MANIFEST_NAME:
        return path.with_name(RAW_NAME), path
    raise ValueError(f"artifact must be a directory, {RAW_NAME}, or {MANIFEST_NAME}")


def _write_jsonl_exclusive(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Create a canonical capture without an exists-check/write race."""

    _require(bool(rows), "capture JSONL is empty")
    payload = "".join(f"{gate.canonical_json(row)}\n" for row in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite existing capture: {path}") from error
    except OSError as error:
        raise ValueError(f"cannot create capture: {path}") from error
    _require(
        _read_exact_bytes(path, description="written capture") == payload,
        f"capture path changed while writing: {path}",
    )


def _reserve_output_directory(path: Path) -> None:
    """Atomically reserve a new aggregate directory; even an empty one is owned."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"refusing to use existing output directory: {path}") from error
    except OSError as error:
        raise ValueError(f"cannot reserve output directory: {path}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    native = commands.add_parser(
        "run-native", help="capture one real native cold/warm feature cell"
    )
    native.add_argument("--cell", choices=gate.CELL_IDS, required=True)
    native.add_argument("--model-path", type=Path, required=True)
    native.add_argument("--parent-manifest", type=Path, required=True)
    native.add_argument("--parent-raw", type=Path, required=True)
    native.add_argument("--dram-profile", type=Path)
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--assert-gate", action="store_true")

    assemble = commands.add_parser(
        "assemble", help="assemble exactly one child capture per fixed KV topology cell"
    )
    for cell_id in gate.CELL_IDS:
        assemble.add_argument(
            f"--{cell_id}",
            dest=cell_id.replace("-", "_"),
            type=Path,
            required=True,
        )
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--assert-gate", action="store_true")

    for command in ("verify", "replay"):
        retained = commands.add_parser(
            command,
            help=(
                "verify manifest and replay retained raw evidence"
                if command == "verify"
                else "independently replay retained raw evidence"
            ),
        )
        retained.add_argument("--artifact", type=Path, required=True)
        retained.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Help exits in argparse before this point.  Every evidence-producing or
    # replaying CLI surface rejects repository bytecode/import artifacts even
    # though only run-native records the full source descriptor.
    _assert_no_import_shadows()
    if args.command == "run-native":
        output = args.output.resolve()
        _require(not output.exists(), f"refusing to overwrite existing capture: {output}")
        rows, passed = asyncio.run(_run_native(args))
        _write_jsonl_exclusive(output, rows)
        result: dict[str, object] = {
            "cell_id": args.cell,
            "feature_id": gate.CELL_SPECS[args.cell]["feature_id"],
            "output": str(output),
            "raw_sha256": gate.sha256_file(output),
            "passed": passed,
        }
    elif args.command == "assemble":
        paths = {
            cell_id: getattr(args, cell_id.replace("-", "_")).resolve() for cell_id in gate.CELL_IDS
        }
        rows = _assemble_rows(paths)
        output_dir = args.output_dir.resolve()
        _reserve_output_directory(output_dir)
        raw_output = output_dir / RAW_NAME
        manifest_output = output_dir / MANIFEST_NAME
        written = gate.write_artifact(
            raw_output,
            manifest_output,
            rows,
            assert_gate=args.assert_gate,
        )
        result = gate.verify_artifact(
            raw_output,
            manifest_output,
            assert_gate=args.assert_gate,
        )
        _require(
            result == written,
            "assembled artifact changed immediately after creation",
        )
    elif args.command == "verify":
        raw_path, manifest_path = _artifact_paths(args.artifact.resolve())
        result = gate.verify_artifact(
            raw_path,
            manifest_path,
            assert_gate=args.assert_gate,
        )
    else:
        raw_path, _manifest_path = _artifact_paths(args.artifact.resolve())
        result = gate.replay_raw(raw_path, assert_gate=args.assert_gate)
    print(gate.canonical_json(result))
    return 1 if args.assert_gate and result.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
