#!/usr/bin/env python3
"""Run the complete G2 A6 fresh-server matrix with safe, hash-bound resume.

This committed operator script never modifies the source checkout: Kairyu's
package and the full checkout are mounted read-only into every Kairyu container,
and every source identity is rechecked before and after a scenario.  All
generated configs, provenance, shards, logs, state, and the assembled artifact
stay below ``--work-dir`` (which must be under /tmp).

Dry-run is read-only.  It validates the pinned dataset, tokenizer, trace bundle,
environment artifact, exact 32+20 plan, provenance schema, any existing resume
shards, and command construction without invoking Docker or creating files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

SCRIPT_SCHEMA = "kairyu.g2.a6.formal-runner.v1"
STATE_NAME = "session-state.json"
_SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO = (
    _SCRIPT_PATH.parents[1]
    if _SCRIPT_PATH.parent.name == "bench"
    else Path.cwd()
)
DEFAULT_WORK_DIR = Path("/tmp/g2-a6-formal")
DEFAULT_TRACE_BUNDLE = Path("/tmp/a6-traces/g2-a6-traces.json")
DEFAULT_DATASET = Path("/tmp/ShareGPT_V3_unfiltered_cleaned_split.json")
DEFAULT_TOKENIZER = Path("/tmp/kairyu-a7-qwen3-32b-tokenizer.json")
DEFAULT_ENV_ARTIFACT = DEFAULT_REPO / "bench/results/env-2026-07-30.json"
DEFAULT_KAIRYU_TEMPLATE = (
    DEFAULT_REPO / "examples/qwen3-32b-multi-gpu/g2-a6-kairyu.template.yaml"
)
DEFAULT_PYTHON = DEFAULT_REPO / ".venv/bin/python"
DEFAULT_KAIRYU_IMAGE = "kairyu-qwen3-32b-kairyu:latest"
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404"
DEFAULT_VLLM_VERSION = "0.26.0+cu129"
DEFAULT_MODEL_VOLUME = "kairyu-qwen3-32b_qwen3-32b"
DEFAULT_PORT = 18080
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Scenario:
    kind: str
    tp: int
    arm: str
    workload: str = "sharegpt"
    repeat: int | None = None
    rate_rps: int | None = None

    @property
    def key(self) -> str:
        if self.kind == "binding":
            assert self.repeat is not None
            return f"tp{self.tp}-{self.workload}-r{self.repeat}-{self.arm}"
        assert self.rate_rps is not None
        return f"tp{self.tp}-sharegpt-open-loop-{self.rate_rps}rps-{self.arm}"

    @property
    def shard_relative(self) -> Path:
        return Path("binding" if self.kind == "binding" else "sweep") / (
            self.key + ".jsonl"
        )

    def identity(self) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": self.kind,
            "tp": self.tp,
            "arm": self.arm,
            "workload": self.workload,
        }
        if self.repeat is not None:
            value["repeat"] = self.repeat
        if self.rate_rps is not None:
            value["rate_rps"] = self.rate_rps
        return value


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    driver_version: str
    pci_bus_id: str


class FormalRunError(RuntimeError):
    pass


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
    announce: bool = True,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if announce:
        print(
            f"[{time.strftime('%H:%M:%S')}] $ {command_text(command)}",
            flush=True,
        )
    try:
        result = subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise FormalRunError(
            f"command timed out after {timeout_s}s: {command_text(command)}"
        ) from error
    if check and result.returncode != 0:
        stdout = (result.stdout or "")[-6000:]
        stderr = (result.stderr or "")[-6000:]
        raise FormalRunError(
            f"command failed ({result.returncode}): {command_text(command)}\n"
            f"stdout tail:\n{stdout}\nstderr tail:\n{stderr}"
        )
    return result


def load_harness(repo: Path) -> ModuleType:
    path = repo / "bench/g2_a6_vllm_bench.py"
    if not path.is_file():
        raise FormalRunError(f"missing harness: {path}")
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("g2_a6_formal_harness", path)
    if spec is None or spec.loader is None:
        raise FormalRunError(f"cannot load harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def binding_plan(a6: ModuleType) -> list[Scenario]:
    result: list[Scenario] = []
    for tp in a6.TP_DEGREES:
        for workload in a6.WORKLOADS:
            for repeat, pair in enumerate(a6.PAIR_ORDER):
                for arm in pair:
                    result.append(
                        Scenario(
                            kind="binding",
                            tp=tp,
                            arm=arm,
                            workload=workload,
                            repeat=repeat,
                        )
                    )
    if len(result) != 32 or len({item.key for item in result}) != 32:
        raise FormalRunError("binding plan is not the exact unique 32-cell matrix")
    return result


def sweep_plan(a6: ModuleType) -> list[Scenario]:
    result = [
        Scenario(
            kind="sweep",
            tp=tp,
            arm=arm,
            rate_rps=rate,
        )
        for tp in a6.TP_DEGREES
        for rate in a6.OPEN_LOOP_RATES_RPS
        for arm in a6.ARMS
    ]
    if len(result) != 20 or len({item.key for item in result}) != 20:
        raise FormalRunError("sweep plan is not the exact unique 20-point grid")
    return result


def plan_sha256(binding: Sequence[Scenario], sweep: Sequence[Scenario]) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "binding": [item.identity() for item in binding],
                "sweep": [item.identity() for item in sweep],
            }
        ).encode()
    )


def git_output(repo: Path, *arguments: str) -> str:
    result = run_command(("git", *arguments), cwd=repo)
    return (result.stdout or "").strip()


def unexpected_source_status_entries(status: bytes) -> list[bytes]:
    entries = status.split(b"\0")
    if entries[-1:] != [b""]:
        raise FormalRunError("git status -z returned invalid framing")
    untracked_prefix = b"?? "
    allowed_root = b".claude/worktrees/"
    return [
        entry
        for entry in entries[:-1]
        if not (
            entry.startswith(untracked_prefix)
            and (
                entry[len(untracked_prefix) :] == allowed_root
                or entry[len(untracked_prefix) :].startswith(allowed_root)
            )
            and b"/../" not in entry[len(untracked_prefix) :]
        )
    ]


def source_identity(repo: Path, *, require_clean: bool) -> tuple[str, bool]:
    commit = git_output(repo, "rev-parse", "HEAD")
    if not HEX40.fullmatch(commit):
        raise FormalRunError(f"HEAD is not a full lowercase commit: {commit!r}")
    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ),
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise FormalRunError(
            "cannot inspect source status:\n"
            + status.stderr.decode(errors="replace")[-6000:]
        )
    unexpected = unexpected_source_status_entries(status.stdout)
    clean = not unexpected
    if require_clean and not clean:
        rendered = ", ".join(repr(item.decode(errors="surrogateescape")) for item in unexpected)
        raise FormalRunError(
            "formal execution requires a clean committed source tree; only "
            "untracked .claude/worktrees/ or a path below it is exempt, "
            f"found: {rendered}"
        )
    return commit, clean


def assert_source_identity(repo: Path, expected_commit: str) -> None:
    commit, clean = source_identity(repo, require_clean=True)
    if commit != expected_commit:
        raise FormalRunError(
            f"source moved during formal run: expected {expected_commit}, got {commit}"
        )
    if not clean:
        raise FormalRunError("source became dirty during formal run")


def project_version(repo: Path) -> str:
    value = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    version = value.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise FormalRunError("pyproject.toml has no project.version")
    return version


def parse_environment(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalRunError(f"cannot read environment artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise FormalRunError("environment artifact must be a JSON object")
    measurement_session_id = value.get("measurement_session_id")
    measured_at_ns = value.get("measured_at_ns")
    if (
        not isinstance(measurement_session_id, str)
        or not measurement_session_id
        or type(measured_at_ns) is not int
        or measured_at_ns <= 0
    ):
        raise FormalRunError(
            "environment artifact must record its own measurement_session_id "
            "and measured_at_ns"
        )
    profile = value.get("profile")
    notes = value.get("notes")
    if not isinstance(profile, dict) or not isinstance(notes, str):
        raise FormalRunError("environment artifact lacks profile/notes")
    matrix = profile.get("p2p_matrix")
    count = profile.get("device_count")
    if (
        type(count) is not int
        or count < 8
        or not isinstance(matrix, list)
        or len(matrix) != count
        or any(not isinstance(row, list) or len(row) != count for row in matrix)
    ):
        raise FormalRunError("environment artifact has no complete measured P2P matrix")
    peer_rates: list[float] = []
    for source, row in enumerate(matrix):
        for destination, rate in enumerate(row):
            if (
                type(rate) not in {int, float}
                or isinstance(rate, bool)
                or not math.isfinite(float(rate))
                or float(rate) <= 0
            ):
                raise FormalRunError("environment P2P matrix contains an invalid rate")
            if source != destination:
                peer_rates.append(float(rate))
    inventory: list[str] = []
    in_inventory = False
    for line in notes.splitlines():
        if line.strip() == "inventory:":
            in_inventory = True
            continue
        if not in_inventory or not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3 and fields[0].isdigit():
            inventory.append(fields[2])
    if len(inventory) != count or len(set(inventory)) != count:
        raise FormalRunError("environment artifact inventory is incomplete")
    measured = profile.get("measured_bandwidth_gbs")
    if (
        type(measured) not in {int, float}
        or isinstance(measured, bool)
        or not math.isfinite(float(measured))
        or float(measured) <= 0
    ):
        raise FormalRunError("environment artifact lacks measured bandwidth")
    return {
        "raw": value,
        "sha256": sha256_file(path),
        "notes_sha256": sha256_bytes(notes.encode()),
        "matrix_sha256": sha256_bytes(canonical_json(matrix).encode()),
        "minimum_peer_bandwidth_gbs": min(peer_rates),
        "pci_bus_ids": tuple(inventory),
    }


def environment_value(
    *,
    repo: Path,
    path: Path,
    parsed: dict[str, Any],
) -> dict[str, object]:
    try:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise FormalRunError("environment artifact must live inside the repository") from error
    if not relative.startswith("bench/results/env-") or not relative.endswith(".json"):
        raise FormalRunError("environment artifact must be bench/results/env-*.json")
    raw = parsed["raw"]
    profile = raw["profile"]
    return {
        "measurement_session_id": raw["measurement_session_id"],
        "measured_at_ns": raw["measured_at_ns"],
        "artifact_date": raw["date"],
        "artifact": {
            "path": relative,
            "sha256": parsed["sha256"],
        },
        "physical_topology_sha256": parsed["notes_sha256"],
        "fabric": {
            "kind": profile["interconnect"],
            "raw_p2p_matrix_sha256": parsed["matrix_sha256"],
            "minimum_peer_bandwidth_gbs": parsed[
                "minimum_peer_bandwidth_gbs"
            ],
        },
    }


def parse_gpu_inventory(text: str) -> list[GPU]:
    result: list[GPU] = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise FormalRunError(f"invalid nvidia-smi inventory line: {line!r}")
        try:
            index = int(fields[0])
        except ValueError as error:
            raise FormalRunError(f"invalid GPU index in {line!r}") from error
        result.append(
            GPU(
                index=index,
                uuid=fields[1],
                name=fields[2],
                driver_version=fields[3],
                pci_bus_id=fields[4],
            )
        )
    result.sort(key=lambda item: item.index)
    if [item.index for item in result[:8]] != list(range(8)):
        raise FormalRunError("formal run requires host GPU indices 0..7")
    if len({item.uuid for item in result[:8]}) != 8:
        raise FormalRunError("host GPU UUIDs are not unique")
    return result[:8]


def live_gpu_inventory() -> list[GPU]:
    result = run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
            "--format=csv,noheader",
        )
    )
    return parse_gpu_inventory(result.stdout or "")


def dry_gpu_inventory(parsed_env: dict[str, Any]) -> list[GPU]:
    raw = parsed_env["raw"]
    profile = raw["profile"]
    return [
        GPU(
            index=index,
            uuid=f"GPU-dry-{index}",
            name=profile["device_name"],
            driver_version=raw["driver"],
            pci_bus_id=parsed_env["pci_bus_ids"][index],
        )
        for index in range(8)
    ]


def validate_inventory_against_environment(
    inventory: Sequence[GPU],
    parsed_env: dict[str, Any],
) -> None:
    raw = parsed_env["raw"]
    profile = raw["profile"]
    expected_pci = parsed_env["pci_bus_ids"]
    for gpu in inventory:
        if gpu.name != profile["device_name"]:
            raise FormalRunError(
                f"GPU {gpu.index} product differs from env artifact: {gpu.name}"
            )
        if gpu.driver_version != raw["driver"]:
            raise FormalRunError(
                f"GPU {gpu.index} driver differs from env artifact: "
                f"{gpu.driver_version} != {raw['driver']}"
            )
        if gpu.pci_bus_id != expected_pci[gpu.index]:
            raise FormalRunError(
                f"GPU {gpu.index} PCI address differs from env artifact: "
                f"{gpu.pci_bus_id} != {expected_pci[gpu.index]}"
            )


def docker_image_id(reference: str) -> str:
    result = run_command(
        ("docker", "image", "inspect", reference, "--format", "{{.Id}}")
    )
    image_id = (result.stdout or "").strip()
    if not image_id.startswith("sha256:") or not HEX64.fullmatch(
        image_id.removeprefix("sha256:")
    ):
        raise FormalRunError(f"invalid Docker image ID for {reference}: {image_id}")
    return image_id


def docker_volume_exists(name: str) -> None:
    result = run_command(("docker", "volume", "inspect", name), check=False)
    if result.returncode != 0:
        raise FormalRunError(f"missing model volume: {name}")


def checkpoint_attestation(
    *,
    a6: ModuleType,
    model_volume: str,
    image: str,
) -> dict[str, object]:
    """Full-hash the live read-only checkpoint volume and verify every pin."""

    result = run_command(
        (
            "docker",
            "run",
            "--rm",
            "-v",
            f"{model_volume}:/models:ro",
            "--entrypoint",
            "/app/.venv/bin/python",
            image,
            "-c",
            a6.CHECKPOINT_HASH_SCRIPT,
            a6.CHECKPOINT_ROOT,
        )
    )
    try:
        measured = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise FormalRunError("cannot parse live checkpoint hash output") from error
    if not isinstance(measured, dict) or not isinstance(measured.get("weights"), list):
        raise FormalRunError("live checkpoint hasher returned an invalid object")
    measured["weights_sha256"] = sha256_bytes(
        json.dumps(measured["weights"], sort_keys=True).encode()
    )
    revision_files = measured.pop("revision_files", None)
    if measured != a6._expected_checkpoint_identity():
        raise FormalRunError("live checkpoint bytes differ from pinned parity evidence")
    if (
        not isinstance(revision_files, list)
        or not revision_files
        or any(
            not isinstance(item, dict)
            or item.get("revision") != a6.MODEL_REVISION
            or not isinstance(item.get("file"), str)
            or not item["file"]
            for item in revision_files
        )
    ):
        raise FormalRunError(
            "live checkpoint Hugging Face revision metadata is absent or inconsistent"
        )
    descriptor = a6.hashed_descriptor(
        {
            "schema_version": 1,
            "model": a6.MODEL,
            "model_revision": a6.MODEL_REVISION,
            "checkpoint_root": a6.CHECKPOINT_ROOT,
            "checkpoint": measured,
            "revision_files": revision_files,
        }
    )
    if not a6._checkpoint_attestation_valid(descriptor):
        raise FormalRunError("live checkpoint attestation failed the harness schema")
    return descriptor


def dry_checkpoint_attestation(a6: ModuleType) -> dict[str, object]:
    descriptor = a6.hashed_descriptor(
        {
            "schema_version": 1,
            "model": a6.MODEL,
            "model_revision": a6.MODEL_REVISION,
            "checkpoint_root": a6.CHECKPOINT_ROOT,
            "checkpoint": a6._expected_checkpoint_identity(),
            "revision_files": [
                {
                    "file": (
                        ".cache/huggingface/download/"
                        "model.safetensors.index.json.metadata"
                    ),
                    "revision": a6.MODEL_REVISION,
                }
            ],
        }
    )
    if not a6._checkpoint_attestation_valid(descriptor):
        raise FormalRunError("dry checkpoint attestation failed the harness schema")
    return descriptor


def dry_attestation_identities(
    *,
    a6: ModuleType,
    args: argparse.Namespace,
    version: str,
    resume_state: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Return schema-valid dry identities, reusing retained live identities."""
    if resume_state is not None:
        checkpoint = resume_state.get("checkpoint")
        runtime = resume_state.get("runtime")
        if not a6._checkpoint_attestation_valid(checkpoint):
            raise FormalRunError("resume state checkpoint fails the formal schema")
        if not isinstance(runtime, dict) or set(runtime) != {
            "kairyu",
            "vllm-stock",
        }:
            raise FormalRunError("resume state runtime identities are invalid")
        for arm, expected_core in (
            (
                "kairyu",
                {
                    "cuda_version": args.kairyu_cuda_version,
                    "nccl_version": args.kairyu_nccl_version,
                },
            ),
            (
                "vllm-stock",
                {
                    "cuda_version": args.vllm_cuda_version,
                    "nccl_version": args.vllm_nccl_version,
                },
            ),
        ):
            retained_runtime = runtime.get(arm)
            if not isinstance(retained_runtime, dict):
                raise FormalRunError(
                    f"resume state runtime identity is invalid for {arm}"
                )
            validate_runtime_versions(
                observed=retained_runtime,
                expected=expected_core,
                arm=arm,
            )
        assert isinstance(checkpoint, dict)
        return checkpoint, runtime
    return (
        dry_checkpoint_attestation(a6),
        {
            "kairyu": {
                "cuda_version": args.kairyu_cuda_version,
                "nccl_version": args.kairyu_nccl_version,
                "package_versions": {
                    "torch": "dry-attested",
                    "flashinfer": "dry-attested",
                    "uvloop": a6.UVLOOP_VERSION,
                    "httptools": a6.HTTPTOOLS_VERSION,
                    "uvicorn": a6.KAIRYU_UVICORN_VERSION,
                    "kairyu": version,
                },
            },
            "vllm-stock": {
                "cuda_version": args.vllm_cuda_version,
                "nccl_version": args.vllm_nccl_version,
                "package_versions": {
                    "torch": "dry-attested",
                    "flashinfer": "dry-attested",
                    "uvloop": a6.UVLOOP_VERSION,
                    "httptools": a6.HTTPTOOLS_VERSION,
                    "uvicorn": a6.VLLM_UVICORN_VERSION,
                    "vllm": args.vllm_version,
                },
            },
        },
    )


def static_state(
    *,
    session_id: str,
    commit: str,
    plan_hash: str,
    env_path: Path,
    env_sha256: str,
    kairyu_ref: str,
    kairyu_id: str,
    vllm_ref: str,
    vllm_id: str,
    kairyu_version: str,
    checkpoint: dict[str, object],
    runtime_identities: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema_version": SCRIPT_SCHEMA,
        "session_id": session_id,
        "repo_commit": commit,
        "source_clean": True,
        "source_status_policy": {
            "format": "porcelain-v1-z-untracked-files-all",
            "only_allowed_exception": ".claude/worktrees/",
        },
        "plan_sha256": plan_hash,
        "environment_artifact": {
            "path": str(env_path.resolve()),
            "sha256": env_sha256,
        },
        "images": {
            "kairyu": {"ref": kairyu_ref, "id": kairyu_id},
            "vllm-stock": {"ref": vllm_ref, "id": vllm_id},
        },
        "versions": {
            "kairyu": kairyu_version,
            "vllm-stock": args.vllm_version,
        },
        "checkpoint": checkpoint,
        "runtime": runtime_identities,
        "last_server_started_ns": 0,
    }


def state_static_projection(value: dict[str, object]) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key != "last_server_started_ns"
    }


def read_resume_state(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalRunError(f"cannot read resume state {path}: {error}") from error
    if not isinstance(existing, dict):
        raise FormalRunError("resume state must be a JSON object")
    session_id = existing.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise FormalRunError("resume state has invalid session_id")
    return existing


def resume_image_id(
    state: dict[str, object] | None,
    arm: str,
    default: str,
) -> str:
    if state is None:
        return default
    images = state.get("images")
    if not isinstance(images, dict):
        raise FormalRunError("resume state has invalid images")
    image = images.get(arm)
    if not isinstance(image, dict):
        raise FormalRunError(f"resume state has invalid image record for {arm}")
    image_id = image.get("id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise FormalRunError(f"resume state has invalid image ID for {arm}")
    if not HEX64.fullmatch(image_id.removeprefix("sha256:")):
        raise FormalRunError(f"resume state has invalid image ID for {arm}")
    return image_id


def load_or_create_state(
    *,
    path: Path,
    expected: dict[str, object],
    shard_paths: Sequence[Path],
    dry_run: bool,
) -> dict[str, object]:
    existing = read_resume_state(path)
    if existing is not None:
        if state_static_projection(existing) != state_static_projection(expected):
            raise FormalRunError(
                "resume state does not match source/env/images/plan; "
                "use a new --work-dir"
            )
        if type(existing.get("last_server_started_ns")) is not int:
            raise FormalRunError("resume state has invalid last_server_started_ns")
        return existing
    if any(item.exists() for item in shard_paths):
        raise FormalRunError(
            "shards exist without session-state.json; refusing unsafe resume"
        )
    if not dry_run:
        atomic_write_json(path, expected)
    return expected


def allocate_server_started_ns(state: dict[str, object], state_path: Path) -> int:
    previous = int(state["last_server_started_ns"])
    value = max(time.time_ns(), previous + 1)
    state["last_server_started_ns"] = value
    atomic_write_json(state_path, state)
    return value


def configuration(a6: ModuleType, scenario: Scenario) -> dict[str, object]:
    return {
        "served_model_name": a6.MODEL,
        "dtype": a6.MODEL_DTYPE,
        "tensor_parallel_size": scenario.tp,
        "max_num_batched_tokens": a6.MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": a6.MAX_NUM_SEQS,
        "enable_prefix_caching": True,
        "cache_block_size": a6.CACHE_BLOCK_SIZE,
        "allocated_cache_blocks": (
            a6.KAIRYU_ALLOCATED_CACHE_BLOCKS
            if scenario.arm == "kairyu"
            else a6.VLLM_ALLOCATED_CACHE_BLOCKS
        ),
        "reserved_graph_scratch_blocks": (
            a6.KAIRYU_GRAPH_SCRATCH_BLOCKS
            if scenario.arm == "kairyu"
            else a6.VLLM_GRAPH_SCRATCH_BLOCKS
        ),
        "reserved_null_blocks": (
            a6.KAIRYU_NULL_BLOCKS
            if scenario.arm == "kairyu"
            else a6.VLLM_NULL_BLOCKS
        ),
        "usable_cache_blocks": a6.USABLE_CACHE_BLOCKS,
        "cache_tokens": a6.CACHE_TOKENS,
        "kv_cache_dtype": a6.KV_CACHE_DTYPE,
        "allocated_kv_cache_bytes_per_rank": (
            a6.ALLOCATED_KV_CACHE_BYTES_PER_RANK[scenario.tp]
        ),
        "reserved_kv_cache_bytes_per_rank": (
            a6.RESERVED_KV_CACHE_BYTES_PER_RANK[scenario.tp]
        ),
        "usable_kv_cache_bytes_per_rank": (
            a6.USABLE_KV_CACHE_BYTES_PER_RANK[scenario.tp]
        ),
        "chunked_prefill_enabled": True,
        "scheduler_policy": "fcfs",
        "access_log_enabled": a6.ACCESS_LOG_ENABLED,
        "max_model_len": a6.MAX_MODEL_LEN,
        "pipeline_depth": (
            a6.KAIRYU_PIPELINE_DEPTH
            if scenario.arm == "kairyu"
            else None
        ),
        "async_scheduling": (
            False
            if scenario.arm == "kairyu"
            else a6.VLLM_ASYNC_SCHEDULING
        ),
        "attention_backend": (
            "flashinfer"
            if scenario.arm == "kairyu"
            else a6.VLLM_ATTENTION_BACKEND
        ),
        "distributed_executor_backend": (
            "torch-distributed-nccl"
            if scenario.arm == "kairyu"
            else a6.VLLM_DISTRIBUTED_EXECUTOR_BACKEND
        ),
        "custom_all_reduce": False,
        "compilation": (
            None
            if scenario.arm == "kairyu"
            else a6.vllm_compilation_config()
        ),
        "cuda_graph": {
            "enabled": True,
            "mode": (
                "decode-only"
                if scenario.arm == "kairyu"
                else a6.VLLM_CUDAGRAPH_MODE
            ),
            "active_decode_graph_cap": a6.ACTIVE_DECODE_GRAPH_CAP,
            "capture_sizes": (
                list(a6.KAIRYU_CUDA_GRAPH_CAPTURE_SIZES)
                if scenario.arm == "kairyu"
                else list(a6.VLLM_CUDA_GRAPH_CAPTURE_SIZES)
            ),
        },
    }


def provenance_value(
    *,
    a6: ModuleType,
    scenario: Scenario,
    state: dict[str, object],
    env_value: dict[str, object],
    inventory: Sequence[GPU],
    host_id: str,
    server_started_ns: int,
    server_generation: str,
    observed_runtime: dict[str, object],
    container_launch: dict[str, object],
    resolved_backend: dict[str, object],
    container_gpus: dict[str, list[str]],
    configuration_source: dict[str, object] | None,
    startup_log: dict[str, object] | None = None,
) -> dict[str, object]:
    images = state["images"]
    versions = state["versions"]
    runtime_versions = state["runtime"]
    assert isinstance(images, dict)
    assert isinstance(versions, dict)
    assert isinstance(runtime_versions, dict)
    image = images[scenario.arm]
    runtime_version = runtime_versions[scenario.arm]
    assert isinstance(image, dict)
    assert isinstance(runtime_version, dict)
    selected = list(inventory[: scenario.tp])
    runtime: dict[str, object] = {
        "arm": scenario.arm,
        "version": versions[scenario.arm],
        "build_commit": (
            state["repo_commit"]
            if scenario.arm == "kairyu"
            else a6.VLLM_BUILD_COMMIT
        ),
        "image_ref": image["ref"],
        "image_id": image["id"],
        "stock_image": scenario.arm == "vllm-stock",
        "cuda_version": runtime_version["cuda_version"],
        "nccl_version": runtime_version["nccl_version"],
        "package_versions": observed_runtime["package_versions"],
        "container_launch": container_launch,
        "resolved_backend": resolved_backend,
    }
    if scenario.arm == "kairyu":
        runtime["source_clean"] = True
        runtime["source_status_policy"] = {
            "format": "porcelain-v1-z-untracked-files-all",
            "only_allowed_exception": (
                f"?? {a6.SOURCE_CLEAN_EXCEPTION_PREFIX}*"
            ),
        }
    else:
        runtime["v1_engine"] = True
        runtime["startup_log"] = startup_log
    return {
        "schema_version": 1,
        "server_generation": server_generation,
        "server_started_ns": server_started_ns,
        "host": {
            "id": host_id,
            "environment": env_value,
        },
        "gpu": {
            "uuids": [item.uuid for item in selected],
            "pci_bus_ids": [item.pci_bus_id for item in selected],
            "product_name": selected[0].name,
            "driver_version": selected[0].driver_version,
            "container_visible_uuids": container_gpus["uuids"],
            "container_visible_pci_bus_ids": container_gpus["pci_bus_ids"],
        },
        "model": a6.model_identity(),
        "checkpoint": state["checkpoint"],
        "runtime": runtime,
        "configuration": configuration(a6, scenario),
        "configuration_source": configuration_source,
    }


def render_kairyu_config(a6: ModuleType, template: str, tp: int) -> str:
    marker = "__TENSOR_PARALLEL_SIZE__"
    if template.count(marker) != 1:
        raise FormalRunError("Kairyu template must contain one TP marker")
    result = template.replace(marker, str(tp))
    if marker in result:
        raise FormalRunError("Kairyu template TP marker was not fully rendered")
    try:
        a6.kairyu_configuration_source(result, tp=tp)
    except ValueError as error:
        raise FormalRunError(str(error)) from error
    return result


def container_name(session_id: str, scenario: Scenario) -> str:
    session = (
        re.sub(r"[^a-z0-9]+", "-", session_id.lower())
        .strip("-")[-16:]
        .strip("-")
    )
    key = re.sub(r"[^a-z0-9]+", "-", scenario.key.lower()).strip("-")
    value = f"g2a6-{session}-{key}"
    if len(value) > 120:
        value = value[:120].rstrip("-")
    return value


def common_docker_run(
    *,
    args: argparse.Namespace,
    scenario: Scenario,
    name: str,
) -> list[str]:
    gpu_ids = ",".join(str(index) for index in range(scenario.tp))
    return [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        f'"device={gpu_ids}"',
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "-p",
        f"127.0.0.1:{args.port}:8000",
        "-v",
        f"{args.model_volume}:/models:ro",
        "-v",
        f"{args.repo.resolve()}:/workspace:ro",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
    ]


def kairyu_docker_command(
    *,
    args: argparse.Namespace,
    scenario: Scenario,
    name: str,
    config_path: Path,
    image_id: str,
) -> list[str]:
    return [
        *common_docker_run(args=args, scenario=scenario, name=name),
        "-v",
        f"{(args.repo / 'kairyu').resolve()}:/app/kairyu:ro",
        "-v",
        f"{config_path.resolve()}:/etc/kairyu/config.yaml:ro",
        "-e",
        "KAIRYU_ATTENTION_BACKEND=flashinfer",
        "--entrypoint",
        "/app/.venv/bin/kairyu",
        image_id,
        "serve",
        "/etc/kairyu/config.yaml",
    ]


def vllm_docker_command(
    *,
    a6: ModuleType,
    args: argparse.Namespace,
    scenario: Scenario,
    name: str,
    image_id: str,
) -> list[str]:
    return [
        *common_docker_run(args=args, scenario=scenario, name=name),
        "-e",
        "VLLM_USE_V1=1",
        "--entrypoint",
        "/usr/local/bin/vllm",
        image_id,
        *a6.expected_engine_argv(arm="vllm-stock", tp=scenario.tp),
    ]


def measure_command(
    *,
    a6: ModuleType,
    args: argparse.Namespace,
    scenario: Scenario,
    provenance_path: Path,
    output_path: Path,
) -> list[str]:
    base = [
        str(args.python),
        str(args.repo / "bench/g2_a6_vllm_bench.py"),
    ]
    common = [
        "--tp",
        str(scenario.tp),
        "--arm",
        scenario.arm,
        "--endpoint",
        f"http://127.0.0.1:{args.port}",
        "--tokenizer",
        str(args.tokenizer),
        "--trace-bundle",
        str(args.trace_bundle),
        "--dataset-sha256",
        a6.SHAREGPT_DATASET_SHA256,
        "--provenance",
        str(provenance_path),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--output",
        str(output_path),
    ]
    if args.api_key:
        common.extend(("--api-key", args.api_key))
    if scenario.kind == "binding":
        assert scenario.repeat is not None
        return [
            *base,
            "measure",
            *common,
            "--workload",
            scenario.workload,
            "--repeat",
            str(scenario.repeat),
        ]
    assert scenario.rate_rps is not None
    return [
        *base,
        "measure-sweep",
        *common,
        "--rate-rps",
        str(scenario.rate_rps),
    ]


def container_running(name: str, *, command_timeout_s: float | None = None) -> bool:
    result = run_command(
        ("docker", "inspect", name, "--format", "{{.State.Running}}"),
        check=False,
        announce=False,
        timeout_s=command_timeout_s,
    )
    return result.returncode == 0 and (result.stdout or "").strip() == "true"


def save_container_logs(name: str, path: Path) -> None:
    result = run_command(
        ("docker", "logs", "--timestamps", name),
        check=False,
    )
    content = (result.stdout or "") + (result.stderr or "")
    atomic_write_text(path, content)


def cleanup_container(name: str) -> None:
    if not name.startswith("g2a6-"):
        raise FormalRunError(f"refusing to clean non-G2-A6 container {name!r}")
    run_command(("docker", "rm", "-f", name), check=False)


def wait_for_health(
    *,
    name: str,
    endpoint: str,
    model: str,
    timeout_s: float,
    command_timeout_s: float | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if not container_running(name, command_timeout_s=command_timeout_s):
            raise FormalRunError(
                f"container {name} exited before /v1/models became healthy"
            )
        try:
            with urllib.request.urlopen(
                f"{endpoint}/v1/models",
                timeout=5,
            ) as response:
                payload = json.loads(response.read())
            models = {
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict)
            }
            if model in models:
                print(
                    f"[{time.strftime('%H:%M:%S')}] healthy: {name} serves {model}",
                    flush=True,
                )
                return
            last_error = f"model list did not contain {model}: {sorted(models)}"
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(2)
    raise FormalRunError(
        f"timed out after {timeout_s}s waiting for {name}: {last_error}"
    )


def normalize_nccl_version(value: object) -> str:
    if isinstance(value, list) and all(type(item) is int for item in value):
        return ".".join(str(item) for item in value)
    if type(value) is int:
        major = value // 10000
        minor = (value // 100) % 100
        patch = value % 100
        return f"{major}.{minor}.{patch}"
    return str(value)


def runtime_probe_code(arm: str) -> str:
    engine_distribution = "kairyu" if arm == "kairyu" else "vllm"
    return (
        "import importlib.metadata as m, json, torch\n"
        "def version(*names):\n"
        "  for name in names:\n"
        "    try: return m.version(name)\n"
        "    except m.PackageNotFoundError: pass\n"
        "  raise RuntimeError('missing distribution: ' + ','.join(names))\n"
        f"engine={engine_distribution!r}\n"
        "print(json.dumps({'cuda':torch.version.cuda,"
        "'nccl':torch.cuda.nccl.version(),"
        "'packages':{'torch':version('torch'),"
        "'flashinfer':version('flashinfer','flashinfer-python'),"
        "'uvloop':version('uvloop'),"
        "'httptools':version('httptools'),"
        "'uvicorn':version('uvicorn'),"
        "engine:version(engine)}}))"
    )


def parse_runtime_probe(output: str) -> dict[str, object]:
    try:
        value = json.loads(output.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise FormalRunError("cannot parse container CUDA/NCCL versions") from error
    normalized = {
        "cuda_version": str(value["cuda"]),
        "nccl_version": normalize_nccl_version(value["nccl"]),
        "package_versions": value["packages"],
    }
    if not isinstance(normalized["package_versions"], dict):
        raise FormalRunError("container runtime package versions are invalid")
    return normalized


def container_runtime_versions(
    name: str,
    arm: str,
    *,
    command_timeout_s: float | None = None,
) -> dict[str, object]:
    python = "/app/.venv/bin/python" if arm == "kairyu" else "python3"
    result = run_command(
        ("docker", "exec", name, python, "-c", runtime_probe_code(arm)),
        timeout_s=command_timeout_s,
    )
    return parse_runtime_probe(result.stdout or "")


def image_runtime_versions(image: str, arm: str) -> dict[str, object]:
    python = "/app/.venv/bin/python" if arm == "kairyu" else "python3"
    result = run_command(
        (
            "docker",
            "run",
            "--rm",
            "--gpus",
            '"device=0"',
            "--entrypoint",
            python,
            image,
            "-c",
            runtime_probe_code(arm),
        )
    )
    return parse_runtime_probe(result.stdout or "")


def validate_runtime_versions(
    *,
    observed: dict[str, object],
    expected: dict[str, object],
    arm: str,
) -> None:
    observed_core = {
        "cuda_version": observed.get("cuda_version"),
        "nccl_version": observed.get("nccl_version"),
    }
    expected_core = {
        "cuda_version": expected.get("cuda_version"),
        "nccl_version": expected.get("nccl_version"),
    }
    if observed_core != expected_core:
        raise FormalRunError(
            f"{arm} runtime differs from pinned provenance: "
            f"observed={observed}, expected={expected}"
        )
    packages = observed.get("package_versions")
    if not isinstance(packages, dict) or set(packages) != {
        "torch",
        "flashinfer",
        "uvloop",
        "httptools",
        "uvicorn",
        "kairyu" if arm == "kairyu" else "vllm",
    }:
        raise FormalRunError(f"{arm} runtime package attestation is incomplete")
    expected_packages = expected.get("package_versions")
    if expected_packages is not None and packages != expected_packages:
        raise FormalRunError(
            f"{arm} runtime package versions differ from resume pin: "
            f"observed={packages}, expected={expected_packages}"
        )


def container_launch_attestation(
    *,
    a6: ModuleType,
    args: argparse.Namespace,
    name: str,
    arm: str,
    tp: int,
    image_id: str,
    configuration_source: dict[str, object] | None,
    config_path: Path | None,
) -> dict[str, object]:
    result = run_command(("docker", "inspect", name))
    try:
        inspected = json.loads((result.stdout or "").strip())
    except json.JSONDecodeError as error:
        raise FormalRunError("cannot parse post-start Docker inspection") from error
    if (
        not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], dict)
    ):
        raise FormalRunError("post-start Docker inspection is not one object")
    container = inspected[0]
    config = container.get("Config")
    host_config = container.get("HostConfig")
    mounts = container.get("Mounts")
    if (
        not isinstance(config, dict)
        or not isinstance(host_config, dict)
        or not isinstance(mounts, list)
    ):
        raise FormalRunError("post-start Docker inspection omitted Config/HostConfig/Mounts")
    env: dict[str, str] = {}
    for item in config.get("Env", ()):
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            if key in env:
                raise FormalRunError(f"container environment repeats {key!r}")
            env[key] = value
    expected = a6.expected_container_launch(arm=arm, tp=tp)
    expected_env = expected["environment"]
    assert isinstance(expected_env, dict)
    if any(env.get(key) != value for key, value in expected_env.items()):
        raise FormalRunError(
            f"{arm} post-start environment omits a required formal value"
        )

    normalized_mounts: list[dict[str, object]] = []
    expected_sources = {
        "/workspace": args.repo.resolve(),
        **(
            {
                "/app/kairyu": (args.repo / "kairyu").resolve(),
                "/etc/kairyu/config.yaml": (
                    config_path.resolve() if config_path is not None else None
                ),
            }
            if arm == "kairyu"
            else {}
        ),
    }
    for mount in mounts:
        if not isinstance(mount, dict):
            raise FormalRunError("container mount inspection contains a non-object")
        destination = mount.get("Destination")
        mount_type = mount.get("Type")
        source = mount.get("Source")
        name_value = mount.get("Name")
        if destination == "/models" and mount_type == "volume":
            if name_value != args.model_volume:
                raise FormalRunError(
                    f"container model volume is {name_value!r}, expected {args.model_volume!r}"
                )
            portable_source = f"volume:{name_value}"
        elif destination in expected_sources and mount_type == "bind":
            expected_source = expected_sources[destination]
            if (
                expected_source is None
                or not isinstance(source, str)
                or Path(source).resolve() != expected_source
            ):
                raise FormalRunError(
                    f"container mount {destination} does not use the expected source"
                )
            if destination == "/workspace":
                portable_source = "repo:."
            elif destination == "/app/kairyu":
                portable_source = "repo:kairyu"
            else:
                if configuration_source is None:
                    raise FormalRunError("Kairyu config mount has no retained source")
                raw_source = configuration_source.get("value")
                if not isinstance(raw_source, dict):
                    raise FormalRunError("Kairyu config source descriptor is invalid")
                portable_source = (
                    f"generated-config:{raw_source.get('text_sha256')}"
                )
        else:
            portable_source = f"external:{sha256_bytes(str(source).encode())}"
        normalized_mounts.append(
            {
                "type": mount_type,
                "name": name_value,
                "source": portable_source,
                "destination": destination,
                "rw": mount.get("RW"),
            }
        )
    normalized_mounts.sort(key=lambda item: str(item["destination"]))

    ulimits = host_config.get("Ulimits")
    memlock = [
        item
        for item in ulimits or ()
        if isinstance(item, dict) and item.get("Name") == "memlock"
    ]
    port_bindings = host_config.get("PortBindings")
    port_rows = (
        port_bindings.get("8000/tcp")
        if isinstance(port_bindings, dict)
        else None
    )
    device_requests = host_config.get("DeviceRequests")
    gpu_requests = [
        item
        for item in device_requests or ()
        if isinstance(item, dict)
        and (
            item.get("Driver") == "nvidia"
            or "gpu" in (item.get("Capabilities") or [[]])[0]
        )
    ]
    if (
        len(memlock) != 1
        or not isinstance(port_rows, list)
        or len(port_rows) != 1
        or not isinstance(port_rows[0], dict)
        or len(gpu_requests) != 1
        or not isinstance(gpu_requests[0].get("DeviceIDs"), list)
    ):
        raise FormalRunError("container HostConfig does not retain the formal resource shape")
    try:
        published_port = int(port_rows[0].get("HostPort"))
    except (TypeError, ValueError) as error:
        raise FormalRunError("container published host port is invalid") from error

    python = "/app/.venv/bin/python" if arm == "kairyu" else "python3"
    distribution = "kairyu" if arm == "kairyu" else "vllm"
    import_code = (
        "import hashlib, importlib, json\n"
        "from pathlib import Path\n"
        "def item(name):\n"
        "  module=importlib.import_module(name)\n"
        "  path=Path(module.__file__).resolve()\n"
        "  return {'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        f"result={{'package':item({distribution!r})}}\n"
        + (
            "result['engine']=item('kairyu.engine.engine_loop')\n"
            if arm == "kairyu"
            else ""
        )
        + "print(json.dumps(result,sort_keys=True))"
    )
    import_result = run_command(
        ("docker", "exec", name, python, "-c", import_code)
    )
    try:
        imported = json.loads((import_result.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise FormalRunError("cannot parse container import-source attestation") from error
    if not isinstance(imported, dict) or not isinstance(imported.get("package"), dict):
        raise FormalRunError("container import-source attestation is invalid")
    package = imported["package"]
    engine_import: dict[str, object] = {
        "distribution": distribution,
        "package_file": package.get("file"),
        "package_sha256": package.get("sha256"),
    }
    if arm == "kairyu":
        engine = imported.get("engine")
        if not isinstance(engine, dict):
            raise FormalRunError("Kairyu engine import-source attestation is absent")
        engine_import.update(
            {
                "engine_file": engine.get("file"),
                "engine_sha256": engine.get("sha256"),
            }
        )
        if (
            config_path is None
            or configuration_source is None
            or not isinstance(configuration_source.get("value"), dict)
        ):
            raise FormalRunError("Kairyu config attestation inputs are absent")
        config_probe = run_command(
            (
                "docker",
                "exec",
                name,
                python,
                "-c",
                (
                    "import hashlib,json\n"
                    "from pathlib import Path\n"
                    "value=Path('/etc/kairyu/config.yaml').read_bytes()\n"
                    "print(json.dumps({'sha256':hashlib.sha256(value).hexdigest(),"
                    "'text':value.decode('utf-8')}))"
                ),
            )
        )
        try:
            config_measured = json.loads(
                (config_probe.stdout or "").strip().splitlines()[-1]
            )
        except (json.JSONDecodeError, IndexError) as error:
            raise FormalRunError("cannot parse mounted Kairyu YAML attestation") from error
        source_raw = configuration_source["value"]
        if (
            not isinstance(config_measured, dict)
            or config_measured.get("text") != source_raw.get("text")
            or config_measured.get("sha256") != source_raw.get("text_sha256")
        ):
            raise FormalRunError("mounted Kairyu YAML bytes differ from provenance")

    measured = {
        "container_id": container.get("Id"),
        "image_id": container.get("Image"),
        "config_image": config.get("Image"),
        "entrypoint": config.get("Entrypoint"),
        "argv": config.get("Cmd"),
        "environment": env,
        "working_dir": config.get("WorkingDir"),
        "mounts": normalized_mounts,
        "host_config": {
            "ipc_mode": host_config.get("IpcMode"),
            "memlock_soft": memlock[0].get("Soft"),
            "memlock_hard": memlock[0].get("Hard"),
            "published_container_port": "8000/tcp",
            "published_host_ip": port_rows[0].get("HostIp"),
            "published_host_port": published_port,
            "gpu_device_ids": gpu_requests[0].get("DeviceIDs"),
        },
        "engine_import": engine_import,
    }
    descriptor = a6.hashed_descriptor(measured)
    if container.get("Image") != image_id or config.get("Image") != image_id:
        raise FormalRunError(
            f"{arm} container image differs from immutable state ID: "
            f"observed={container.get('Image')}/{config.get('Image')}, expected={image_id}"
        )
    if not a6._container_launch_valid(
        descriptor,
        arm=arm,
        tp=tp,
        image_id=image_id,
        configuration_source=configuration_source,
    ):
        raise FormalRunError(
            f"{arm} post-start container launch/mount/import attestation is invalid"
        )
    return descriptor


def container_gpu_attestation(
    name: str,
    selected: Sequence[GPU],
    *,
    command_timeout_s: float | None = None,
) -> dict[str, list[str]]:
    result = run_command(
        (
            "docker",
            "exec",
            name,
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader",
        ),
        timeout_s=command_timeout_s,
    )
    rows = [
        [field.strip() for field in line.split(",", 1)]
        for line in (result.stdout or "").splitlines()
        if line.strip()
    ]
    if any(len(row) != 2 for row in rows):
        raise FormalRunError("container GPU inventory has invalid rows")
    measured = {
        "uuids": [row[0] for row in rows],
        "pci_bus_ids": [row[1] for row in rows],
    }
    expected = {
        "uuids": [item.uuid for item in selected],
        "pci_bus_ids": [item.pci_bus_id for item in selected],
    }
    if measured != expected:
        raise FormalRunError(
            "container-visible GPUs differ from the selected host assignment: "
            f"observed={measured}, expected={expected}"
        )
    return measured


def resolved_backend_attestation(
    *,
    a6: ModuleType,
    endpoint: str,
    arm: str,
    tp: int,
    startup_log: dict[str, object] | None = None,
    container_launch: dict[str, object] | None = None,
    image_id: str | None = None,
) -> dict[str, object]:
    expected = a6.expected_resolved_backend(arm=arm, tp=tp)
    if arm == "vllm-stock":
        if (
            startup_log is None
            or not a6._vllm_startup_log_valid(startup_log, tp=tp)
            or container_launch is None
            or not a6._container_launch_valid(
                container_launch,
                arm=arm,
                tp=tp,
                image_id=image_id,
                configuration_source=None,
            )
        ):
            raise FormalRunError(
                "vLLM resolved backend requires valid startup messages and "
                "post-start inspected immutable argv"
            )
        return a6.hashed_descriptor(expected)
    try:
        with urllib.request.urlopen(f"{endpoint}/backends", timeout=10) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise FormalRunError(f"cannot attest Kairyu /backends: {error}") from error
    if not isinstance(payload, dict):
        raise FormalRunError("Kairyu /backends returned a non-object")
    engines = payload.get("engines")
    if not isinstance(engines, list) or len(engines) != 1 or not isinstance(engines[0], dict):
        raise FormalRunError("Kairyu /backends must report exactly one engine")
    engine = engines[0]
    measured = {
        "attestation": "GET /backends after readiness",
        "attention_backend": payload.get("attention_backend"),
        "requested_attention_backend": payload.get("requested_attention_backend"),
        "attention_components": payload.get("attention_components"),
        "source": payload.get("source"),
        "role": payload.get("role"),
        "engine_backend": engine.get("engine_backend"),
        "engine_attention_backend": engine.get("attention_backend"),
        "engine_attention_components": engine.get("attention_components"),
        "engine_decision_status": engine.get("decision_status"),
        "tensor_parallel_size": engine.get("tensor_parallel_size"),
    }
    if measured != expected:
        raise FormalRunError(
            "Kairyu resolved backend differs from the formal pin: "
            f"observed={measured}, expected={expected}"
        )
    return a6.hashed_descriptor(measured)


def vllm_startup_log_attestation(
    *,
    a6: ModuleType,
    name: str,
    tp: int,
) -> dict[str, object]:
    result = run_command(("docker", "logs", name), check=False)
    content = (result.stdout or "") + "\n" + (result.stderr or "")
    config_lines = [
        line
        for line in content.splitlines()
        if "Initializing a V1 LLM engine" in line and "with config:" in line
    ]
    attention_lines = [
        line
        for line in content.splitlines()
        if a6.VLLM_ATTENTION_BACKEND_LOG_MARKER in line
    ]
    attention_version_lines = [
        line
        for line in content.splitlines()
        if (
            f"Using FlashAttention version {a6.VLLM_FLASH_ATTN_VERSION}"
            in line
        )
    ]
    async_scheduling_lines = [
        line
        for line in content.splitlines()
        if "Asynchronous scheduling is enabled." in line
    ]
    if (
        len(config_lines) != 1
        or not attention_lines
        or not attention_version_lines
        or not async_scheduling_lines
    ):
        raise FormalRunError(
            "vLLM startup logs do not uniquely attest the V1 config and "
            "resolved async scheduling/FLASH_ATTN version"
        )
    config_line = config_lines[0]
    config_message = config_line[config_line.index("Initializing a V1 LLM engine") :]
    attention_messages = sorted(
        {
            line[line.index(a6.VLLM_ATTENTION_BACKEND_LOG_MARKER) :]
            for line in attention_lines
        }
    )
    attention_version_messages = sorted(
        {
            line[line.index("Using FlashAttention version") :]
            for line in attention_version_lines
        }
    )
    async_scheduling_messages = sorted(
        {
            line[line.index("Asynchronous scheduling is enabled.") :]
            for line in async_scheduling_lines
        }
    )
    markers = a6._vllm_startup_markers(
        config_message,
        attention_messages,
        attention_version_messages,
        async_scheduling_messages,
        tp=tp,
    )
    if not all(markers.values()):
        failed = sorted(key for key, passed in markers.items() if not passed)
        raise FormalRunError(
            f"vLLM startup logs omit resolved formal settings: {failed}"
        )
    descriptor = a6.hashed_descriptor(
        {
            "engine_config_message": config_message,
            "engine_config_line_sha256": sha256_bytes(config_message.encode()),
            "attention_backend_messages": attention_messages,
            "attention_backend_line_sha256": sha256_bytes(
                canonical_json(attention_messages).encode()
            ),
            "attention_version_messages": attention_version_messages,
            "attention_version_line_sha256": sha256_bytes(
                canonical_json(attention_version_messages).encode()
            ),
            "async_scheduling_messages": async_scheduling_messages,
            "async_scheduling_line_sha256": sha256_bytes(
                canonical_json(async_scheduling_messages).encode()
            ),
            "resolved_markers": markers,
        }
    )
    if not a6._vllm_startup_log_valid(descriptor, tp=tp):
        raise FormalRunError("vLLM startup log attestation failed the harness schema")
    return descriptor


def shard_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256.json")


def validate_shard(
    *,
    a6: ModuleType,
    scenario: Scenario,
    path: Path,
    trace_bundle: dict[str, object],
    state: dict[str, object],
    write_missing_sidecar: bool,
) -> tuple[str, str, int]:
    rows = a6._read_jsonl(path)
    if scenario.kind == "binding":
        start, requests, end = a6._parse_measurement_shard(rows)
        expected_identity = (
            scenario.tp,
            scenario.workload,
            scenario.repeat,
            scenario.arm,
        )
        actual_identity = (
            start.get("tensor_parallel_size"),
            start.get("workload"),
            start.get("repeat"),
            start.get("arm"),
        )
        if actual_identity != expected_identity:
            raise FormalRunError(
                f"{path} identity mismatch: {actual_identity} != {expected_identity}"
            )
        expected_formal = (
            a6.SHAREGPT_REQUESTS
            if scenario.workload == "sharegpt"
            else a6.SHARED_PREFIX_REQUESTS
        )
        if (
            len(requests)
            != a6.WARMUP_REQUESTS + a6.GRAPH_WARMUP_REQUESTS + expected_formal
            or end.get("warmup_request_count")
            != a6.WARMUP_REQUESTS + a6.GRAPH_WARMUP_REQUESTS
            or end.get("serial_warmup_request_count") != a6.WARMUP_REQUESTS
            or end.get("graph_warmup_request_count")
            != a6.GRAPH_WARMUP_REQUESTS
            or end.get("measurement_request_count") != expected_formal
        ):
            raise FormalRunError(f"{path} has incomplete request counts")
    else:
        start, requests, end = a6._parse_sweep_shard(rows)
        expected_identity = (scenario.tp, scenario.rate_rps, scenario.arm)
        actual_identity = (
            start.get("tensor_parallel_size"),
            start.get("arrival_rate_rps"),
            start.get("arm"),
        )
        if actual_identity != expected_identity:
            raise FormalRunError(
                f"{path} identity mismatch: {actual_identity} != {expected_identity}"
            )
        if (
            len(requests)
            != a6.WARMUP_REQUESTS
            + a6.GRAPH_WARMUP_REQUESTS
            + a6.SHAREGPT_REQUESTS
            or end.get("warmup_request_count")
            != a6.WARMUP_REQUESTS + a6.GRAPH_WARMUP_REQUESTS
            or end.get("serial_warmup_request_count") != a6.WARMUP_REQUESTS
            or end.get("graph_warmup_request_count")
            != a6.GRAPH_WARMUP_REQUESTS
            or end.get("sweep_request_count") != a6.SHAREGPT_REQUESTS
        ):
            raise FormalRunError(f"{path} has incomplete request counts")
    if start.get("benchmark") != trace_bundle["benchmark"]:
        raise FormalRunError(f"{path} benchmark configuration differs")
    traces = trace_bundle["traces"]
    assert isinstance(traces, dict)
    trace = start.get("trace")
    if trace != traces[scenario.workload]:
        raise FormalRunError(f"{path} trace differs from pinned trace bundle")
    if start.get("graph_warmup") != trace_bundle.get("graph_warmup"):
        raise FormalRunError(f"{path} graph warmup differs from pinned trace bundle")
    provenance_start = start.get("provenance_start")
    provenance_end = end.get("provenance_end")
    if provenance_start != provenance_end:
        raise FormalRunError(f"{path} provenance changed during measurement")
    if not a6._provenance_valid(
        provenance_start,
        tp=scenario.tp,
        arm=scenario.arm,
    ):
        raise FormalRunError(f"{path} provenance is invalid")
    assert isinstance(provenance_start, dict)
    value = provenance_start["value"]
    assert isinstance(value, dict)
    host = value["host"]
    runtime = value["runtime"]
    assert isinstance(host, dict)
    assert isinstance(runtime, dict)
    environment = host["environment"]
    assert isinstance(environment, dict)
    if environment.get("measurement_session_id") != state["session_id"]:
        raise FormalRunError(f"{path} belongs to another measurement session")
    expected_image = state["images"][scenario.arm]
    assert isinstance(expected_image, dict)
    if (
        runtime.get("image_ref") != expected_image["ref"]
        or runtime.get("image_id") != expected_image["id"]
    ):
        raise FormalRunError(f"{path} Docker image differs from resume state")
    if scenario.arm == "kairyu" and (
        runtime.get("build_commit") != state["repo_commit"]
        or runtime.get("source_clean") is not True
    ):
        raise FormalRunError(f"{path} Kairyu source identity differs")
    digest = sha256_file(path)
    summary = {
        "schema_version": SCRIPT_SCHEMA,
        "identity": scenario.identity(),
        "sha256": digest,
    }
    sidecar = shard_sidecar(path)
    if sidecar.exists():
        try:
            retained = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FormalRunError(f"invalid sidecar {sidecar}: {error}") from error
        if retained != summary:
            raise FormalRunError(f"{path} content SHA/identity sidecar mismatch")
    elif write_missing_sidecar:
        atomic_write_json(sidecar, summary)
    generation = value.get("server_generation")
    started_ns = value.get("server_started_ns")
    if not isinstance(generation, str) or type(started_ns) is not int:
        raise FormalRunError(f"{path} generation/start provenance is invalid")
    return digest, generation, started_ns


def validate_resume(
    *,
    a6: ModuleType,
    binding: Sequence[Scenario],
    sweep: Sequence[Scenario],
    args: argparse.Namespace,
    trace_bundle: dict[str, object],
    state: dict[str, object],
) -> dict[str, bool]:
    status: dict[str, bool] = {}
    generations: set[str] = set()
    max_started = int(state["last_server_started_ns"])
    for scenario in [*binding, *sweep]:
        path = args.work_dir / scenario.shard_relative
        status[scenario.key] = path.exists()
        if not path.exists():
            continue
        _, generation, started_ns = validate_shard(
            a6=a6,
            scenario=scenario,
            path=path,
            trace_bundle=trace_bundle,
            state=state,
            write_missing_sidecar=not args.dry_run,
        )
        if generation in generations:
            raise FormalRunError(
                f"server_generation {generation!r} is reused across shards"
            )
        generations.add(generation)
        max_started = max(max_started, started_ns)
    for tp in a6.TP_DEGREES:
        for workload in a6.WORKLOADS:
            group = [
                item
                for item in binding
                if item.tp == tp and item.workload == workload
            ]
            saw_missing = False
            for scenario in group:
                if not status[scenario.key]:
                    saw_missing = True
                elif saw_missing:
                    raise FormalRunError(
                        "unsafe binding resume hole: existing shards must be a "
                        f"prefix of pair order for TP{tp}/{workload}; found "
                        f"{scenario.key} after a missing predecessor"
                    )
    if max_started > int(state["last_server_started_ns"]) and not args.dry_run:
        state["last_server_started_ns"] = max_started
        atomic_write_json(args.work_dir / STATE_NAME, state)
    return status


def promote_completed_shard(
    *,
    a6: ModuleType,
    scenario: Scenario,
    temporary: Path,
    final: Path,
    trace_bundle: dict[str, object],
    state: dict[str, object],
) -> None:
    validate_shard(
        a6=a6,
        scenario=scenario,
        path=temporary,
        trace_bundle=trace_bundle,
        state=state,
        write_missing_sidecar=False,
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)
    validate_shard(
        a6=a6,
        scenario=scenario,
        path=final,
        trace_bundle=trace_bundle,
        state=state,
        write_missing_sidecar=True,
    )


def run_scenario(
    *,
    a6: ModuleType,
    args: argparse.Namespace,
    scenario: Scenario,
    state: dict[str, object],
    parsed_env: dict[str, Any],
    inventory: Sequence[GPU],
    trace_bundle: dict[str, object],
    kairyu_template: str,
) -> None:
    assert_source_identity(args.repo, str(state["repo_commit"]))
    state_path = args.work_dir / STATE_NAME
    started_ns = allocate_server_started_ns(state, state_path)
    generation = f"{state['session_id']}:{scenario.key}:{uuid.uuid4().hex}"
    name = container_name(str(state["session_id"]), scenario)
    final = args.work_dir / scenario.shard_relative
    temporary = final.with_name(f".{final.name}.partial-{os.getpid()}")
    provenance_path = args.work_dir / "provenance" / f"{scenario.key}.json"
    config_path = args.work_dir / "configs" / f"{scenario.key}.yaml"
    log_path = args.work_dir / "logs" / f"{scenario.key}.container.log"
    if temporary.exists():
        temporary.unlink()
    images = state.get("images")
    if not isinstance(images, dict) or not isinstance(images.get(scenario.arm), dict):
        raise FormalRunError(f"resume state has no image for {scenario.arm}")
    image_id = images[scenario.arm].get("id")
    if (
        not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or not HEX64.fullmatch(image_id.removeprefix("sha256:"))
    ):
        raise FormalRunError(f"resume state image ID is invalid for {scenario.arm}")
    configuration_source: dict[str, object] | None = None
    if scenario.arm == "kairyu":
        rendered_config = render_kairyu_config(
            a6,
            kairyu_template,
            scenario.tp,
        )
        configuration_source = a6.kairyu_configuration_source(
            rendered_config,
            tp=scenario.tp,
        )
        atomic_write_text(
            config_path,
            rendered_config,
        )
        docker_command = kairyu_docker_command(
            args=args,
            scenario=scenario,
            name=name,
            config_path=config_path,
            image_id=image_id,
        )
    else:
        docker_command = vllm_docker_command(
            a6=a6,
            args=args,
            scenario=scenario,
            name=name,
            image_id=image_id,
        )
    print(
        f"\n[{time.strftime('%H:%M:%S')}] START {scenario.key} "
        f"(fresh generation {generation})",
        flush=True,
    )
    cleanup_container(name)
    started = False
    try:
        run_command(docker_command)
        started = True
        endpoint = f"http://127.0.0.1:{args.port}"
        wait_for_health(
            name=name,
            endpoint=endpoint,
            model=a6.MODEL,
            timeout_s=args.startup_timeout_s,
        )
        observed_runtime = container_runtime_versions(name, scenario.arm)
        expected_runtime = state["runtime"][scenario.arm]
        assert isinstance(expected_runtime, dict)
        validate_runtime_versions(
            observed=observed_runtime,
            expected=expected_runtime,
            arm=scenario.arm,
        )
        selected = list(inventory[: scenario.tp])
        container_launch = container_launch_attestation(
            a6=a6,
            args=args,
            name=name,
            arm=scenario.arm,
            tp=scenario.tp,
            image_id=image_id,
            configuration_source=configuration_source,
            config_path=(config_path if scenario.arm == "kairyu" else None),
        )
        container_gpus = container_gpu_attestation(name, selected)
        startup_log = (
            vllm_startup_log_attestation(
                a6=a6,
                name=name,
                tp=scenario.tp,
            )
            if scenario.arm == "vllm-stock"
            else None
        )
        resolved_backend = resolved_backend_attestation(
            a6=a6,
            endpoint=endpoint,
            arm=scenario.arm,
            tp=scenario.tp,
            startup_log=startup_log,
            container_launch=container_launch,
            image_id=image_id,
        )
        env_value = environment_value(
            repo=args.repo,
            path=args.env_artifact,
            parsed=parsed_env,
        )
        provenance = provenance_value(
            a6=a6,
            scenario=scenario,
            state=state,
            env_value=env_value,
            inventory=inventory,
            host_id=socket.gethostname(),
            server_started_ns=started_ns,
            server_generation=generation,
            observed_runtime=observed_runtime,
            container_launch=container_launch,
            resolved_backend=resolved_backend,
            container_gpus=container_gpus,
            configuration_source=configuration_source,
            startup_log=startup_log,
        )
        descriptor = a6.hashed_descriptor(provenance)
        if not a6._provenance_valid(
            descriptor,
            tp=scenario.tp,
            arm=scenario.arm,
        ):
            raise FormalRunError(
                f"generated provenance failed harness schema for {scenario.key}"
            )
        atomic_write_json(provenance_path, provenance)
        measure = measure_command(
            a6=a6,
            args=args,
            scenario=scenario,
            provenance_path=provenance_path,
            output_path=temporary,
        )
        result = run_command(measure)
        if result.stdout:
            print(result.stdout.strip(), flush=True)
        assert_source_identity(args.repo, str(state["repo_commit"]))
        promote_completed_shard(
            a6=a6,
            scenario=scenario,
            temporary=temporary,
            final=final,
            trace_bundle=trace_bundle,
            state=state,
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] COMPLETE {scenario.key} "
            f"sha256={sha256_file(final)}",
            flush=True,
        )
    finally:
        if started or container_running(name):
            save_container_logs(name, log_path)
        cleanup_container(name)
        if temporary.exists():
            temporary.unlink()


def assembly_command(
    *,
    args: argparse.Namespace,
    binding: Sequence[Scenario],
    sweep: Sequence[Scenario],
) -> list[str]:
    command = [
        str(args.python),
        str(args.repo / "bench/g2_a6_vllm_bench.py"),
        "assemble",
    ]
    for scenario in binding:
        command.extend(("--raw", str(args.work_dir / scenario.shard_relative)))
    for scenario in sweep:
        command.extend(
            ("--sweep-raw", str(args.work_dir / scenario.shard_relative))
        )
    command.extend(
        (
            "--output-dir",
            str(args.work_dir / "artifact"),
            "--assert-gate",
        )
    )
    return command


def verify_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(args.repo / "bench/g2_a6_vllm_bench.py"),
        "verify",
        "--artifact",
        str(args.work_dir / "artifact"),
        "--assert-gate",
    ]


def dry_validate_provenance(
    *,
    a6: ModuleType,
    binding: Sequence[Scenario],
    sweep: Sequence[Scenario],
    state: dict[str, object],
    parsed_env: dict[str, Any],
    inventory: Sequence[GPU],
    repo: Path,
    env_path: Path,
    kairyu_template: str,
    model_volume: str,
    published_host_port: int,
) -> None:
    env_value = environment_value(
        repo=repo,
        path=env_path,
        parsed=parsed_env,
    )
    starts: dict[tuple[int, str, int, str], dict[str, object]] = {}
    ends: dict[tuple[int, str, int, str], dict[str, object]] = {}
    generations: set[str] = set()

    def dry_attestations(
        scenario: Scenario,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, list[str]],
        dict[str, object] | None,
        dict[str, object] | None,
    ]:
        selected = list(inventory[: scenario.tp])
        image = state["images"][scenario.arm]
        assert isinstance(image, dict)
        image_id = image["id"]
        configuration_source = (
            a6.kairyu_configuration_source(
                render_kairyu_config(
                    a6,
                    kairyu_template,
                    scenario.tp,
                ),
                tp=scenario.tp,
            )
            if scenario.arm == "kairyu"
            else None
        )
        expected_launch = a6.expected_container_launch(
            arm=scenario.arm,
            tp=scenario.tp,
        )
        mounts: list[dict[str, object]] = [
            {
                "type": "volume",
                "name": model_volume,
                "source": f"volume:{model_volume}",
                "destination": "/models",
                "rw": False,
            },
            {
                "type": "bind",
                "name": None,
                "source": "repo:.",
                "destination": "/workspace",
                "rw": False,
            },
        ]
        if scenario.arm == "kairyu":
            assert isinstance(configuration_source, dict)
            source_value = configuration_source["value"]
            assert isinstance(source_value, dict)
            mounts.extend(
                (
                    {
                        "type": "bind",
                        "name": None,
                        "source": "repo:kairyu",
                        "destination": "/app/kairyu",
                        "rw": False,
                    },
                    {
                        "type": "bind",
                        "name": None,
                        "source": (
                            f"generated-config:{source_value['text_sha256']}"
                        ),
                        "destination": "/etc/kairyu/config.yaml",
                        "rw": False,
                    },
                )
            )
        imports = {
            "distribution": (
                "kairyu" if scenario.arm == "kairyu" else "vllm"
            ),
            "package_file": (
                "/app/kairyu/__init__.py"
                if scenario.arm == "kairyu"
                else "/usr/local/lib/python/site-packages/vllm/__init__.py"
            ),
            "package_sha256": sha256_bytes(
                f"dry-package-{scenario.arm}".encode()
            ),
        }
        if scenario.arm == "kairyu":
            imports.update(
                {
                    "engine_file": "/app/kairyu/engine/engine_loop.py",
                    "engine_sha256": sha256_bytes(b"dry-engine"),
                }
            )
        launch = a6.hashed_descriptor(
            {
                "container_id": sha256_bytes(
                    f"dry-container-{scenario.key}".encode()
                ),
                "image_id": image_id,
                "config_image": image_id,
                "entrypoint": expected_launch["entrypoint"],
                "argv": expected_launch["argv"],
                "environment": expected_launch["environment"],
                "working_dir": (
                    "/app" if scenario.arm == "kairyu" else "/root"
                ),
                "mounts": sorted(
                    mounts,
                    key=lambda item: str(item["destination"]),
                ),
                "host_config": {
                    "ipc_mode": "host",
                    "memlock_soft": -1,
                    "memlock_hard": -1,
                    "published_container_port": "8000/tcp",
                    "published_host_ip": "127.0.0.1",
                    "published_host_port": published_host_port,
                    "gpu_device_ids": [
                        str(index) for index in range(scenario.tp)
                    ],
                },
                "engine_import": imports,
            }
        )
        backend = a6.hashed_descriptor(
            a6.expected_resolved_backend(arm=scenario.arm, tp=scenario.tp)
        )
        gpus = {
            "uuids": [item.uuid for item in selected],
            "pci_bus_ids": [item.pci_bus_id for item in selected],
        }
        startup_log = (
            a6.hashed_descriptor(
                {
                    "engine_config_message": (
                        "Initializing a V1 LLM engine with config: "
                        f"tensor_parallel_size={scenario.tp}, "
                        f"max_seq_len={a6.MAX_MODEL_LEN}, "
                        "enable_prefix_caching=True, "
                        "enable_chunked_prefill=True, "
                        "disable_custom_all_reduce=True, "
                        f"VLLM_COMPILE {a6.VLLM_CUDAGRAPH_MODE} "
                        f"{list(a6.VLLM_CUDA_GRAPH_CAPTURE_SIZES)}"
                    ),
                    "engine_config_line_sha256": sha256_bytes(b"dry-config"),
                        "attention_backend_messages": [
                            a6.VLLM_ATTENTION_BACKEND_LOG_MARKER
                        ],
                        "attention_backend_line_sha256": sha256_bytes(
                            canonical_json(
                                [a6.VLLM_ATTENTION_BACKEND_LOG_MARKER]
                            ).encode()
                        ),
                    "attention_version_messages": [
                        (
                            "Using FlashAttention version "
                            f"{a6.VLLM_FLASH_ATTN_VERSION}"
                        )
                    ],
                    "attention_version_line_sha256": sha256_bytes(
                        canonical_json(
                            [
                                (
                                    "Using FlashAttention version "
                                    f"{a6.VLLM_FLASH_ATTN_VERSION}"
                                )
                            ]
                        ).encode()
                    ),
                    "async_scheduling_messages": [
                        "Asynchronous scheduling is enabled."
                    ],
                    "async_scheduling_line_sha256": sha256_bytes(
                        canonical_json(
                            ["Asynchronous scheduling is enabled."]
                        ).encode()
                    ),
                    "resolved_markers": {},
                }
            )
            if scenario.arm == "vllm-stock"
            else None
        )
        if startup_log is not None:
            log_value = startup_log["value"]
            assert isinstance(log_value, dict)
            config_message = log_value["engine_config_message"]
            attention_messages = log_value["attention_backend_messages"]
            attention_version_messages = log_value[
                "attention_version_messages"
            ]
            async_scheduling_messages = log_value[
                "async_scheduling_messages"
            ]
            assert isinstance(config_message, str)
            assert isinstance(attention_messages, list)
            assert isinstance(attention_version_messages, list)
            assert isinstance(async_scheduling_messages, list)
            log_value["engine_config_line_sha256"] = sha256_bytes(
                config_message.encode()
            )
            log_value["resolved_markers"] = a6._vllm_startup_markers(
                config_message,
                attention_messages,
                attention_version_messages,
                async_scheduling_messages,
                tp=scenario.tp,
            )
            startup_log = a6.hashed_descriptor(log_value)
        return launch, backend, gpus, startup_log, configuration_source

    for index, scenario in enumerate(binding, 1):
        generation = f"dry-generation-{index}"
        generations.add(generation)
        launch, backend, gpus, startup_log, configuration_source = (
            dry_attestations(scenario)
        )
        provenance = provenance_value(
            a6=a6,
            scenario=scenario,
            state=state,
            env_value=env_value,
            inventory=inventory,
            host_id="dry-host",
            server_started_ns=index,
            server_generation=generation,
            observed_runtime=state["runtime"][scenario.arm],
            container_launch=launch,
            resolved_backend=backend,
            container_gpus=gpus,
            configuration_source=configuration_source,
            startup_log=startup_log,
        )
        descriptor = a6.hashed_descriptor(provenance)
        if not a6._provenance_valid(
            descriptor,
            tp=scenario.tp,
            arm=scenario.arm,
        ):
            raise FormalRunError(
                f"dry provenance failed for binding {scenario.key}"
            )
        assert scenario.repeat is not None
        identity = (
            scenario.tp,
            scenario.workload,
            scenario.repeat,
            scenario.arm,
        )
        starts[identity] = {"provenance_start": descriptor}
        ends[identity] = {"provenance_end": descriptor}
    checks, _ = a6._provenance_matrix(starts, ends)
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise FormalRunError(f"dry binding provenance checks failed: {failed}")
    for index, scenario in enumerate(sweep, len(binding) + 1):
        generation = f"dry-generation-{index}"
        if generation in generations:
            raise FormalRunError("dry generation collision")
        generations.add(generation)
        launch, backend, gpus, startup_log, configuration_source = (
            dry_attestations(scenario)
        )
        provenance = provenance_value(
            a6=a6,
            scenario=scenario,
            state=state,
            env_value=env_value,
            inventory=inventory,
            host_id="dry-host",
            server_started_ns=index,
            server_generation=generation,
            observed_runtime=state["runtime"][scenario.arm],
            container_launch=launch,
            resolved_backend=backend,
            container_gpus=gpus,
            configuration_source=configuration_source,
            startup_log=startup_log,
        )
        if not a6._provenance_valid(
            a6.hashed_descriptor(provenance),
            tp=scenario.tp,
            arm=scenario.arm,
        ):
            raise FormalRunError(f"dry sweep provenance failed for {scenario.key}")


def validate_paths(args: argparse.Namespace, a6: ModuleType) -> dict[str, object]:
    if not args.repo.resolve().is_dir():
        raise FormalRunError(f"missing repository: {args.repo}")
    if not args.work_dir.resolve().is_relative_to(Path("/tmp")):
        raise FormalRunError("--work-dir must resolve below /tmp")
    for label, path in (
        ("python", args.python),
        ("dataset", args.dataset),
        ("tokenizer", args.tokenizer),
        ("trace bundle", args.trace_bundle),
        ("environment artifact", args.env_artifact),
        ("Kairyu template", args.kairyu_template),
    ):
        if not path.is_file():
            raise FormalRunError(f"missing {label}: {path}")
    if sha256_file(args.dataset) != a6.SHAREGPT_DATASET_SHA256:
        raise FormalRunError("ShareGPT dataset SHA-256 differs from the pin")
    tokenizer_file = (
        args.tokenizer
        if args.tokenizer.is_file()
        else args.tokenizer / "tokenizer.json"
    )
    if sha256_file(tokenizer_file) != a6.PINNED_TOKENIZER_SHA256:
        raise FormalRunError("tokenizer.json SHA-256 differs from the pin")
    trace_bundle = a6.load_trace_bundle(
        args.trace_bundle,
        dataset_sha256=a6.SHAREGPT_DATASET_SHA256,
    )
    tokenizer = a6.HFTokenizer(args.tokenizer)
    regenerated = {
        "schema_version": a6.TRACE_BUNDLE_SCHEMA,
        "benchmark": a6.benchmark_config(a6.SHAREGPT_DATASET_SHA256),
        "graph_warmup": a6.build_graph_warmup_trace(tokenizer),
        "traces": {
            "sharegpt": a6.build_sharegpt_trace(
                args.dataset,
                tokenizer,
                expected_dataset_sha256=a6.SHAREGPT_DATASET_SHA256,
            ),
            "shared-prefix": a6.build_shared_prefix_trace(tokenizer),
        },
    }
    if regenerated != trace_bundle:
        raise FormalRunError(
            "retained trace bundle differs from exact regeneration of the "
            "pinned dataset/tokenizer"
        )
    return trace_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--trace-bundle", type=Path, default=DEFAULT_TRACE_BUNDLE)
    parser.add_argument("--env-artifact", type=Path, default=DEFAULT_ENV_ARTIFACT)
    parser.add_argument(
        "--kairyu-template",
        type=Path,
        default=DEFAULT_KAIRYU_TEMPLATE,
    )
    parser.add_argument("--kairyu-image", default=DEFAULT_KAIRYU_IMAGE)
    parser.add_argument(
        "--vllm-image",
        default=DEFAULT_VLLM_IMAGE,
    )
    parser.add_argument("--model-volume", default=DEFAULT_MODEL_VOLUME)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key")
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--vllm-version", default=DEFAULT_VLLM_VERSION)
    parser.add_argument("--kairyu-cuda-version", default="13.0")
    parser.add_argument("--kairyu-nccl-version", default="2.29.7")
    parser.add_argument("--vllm-cuda-version", default="12.9")
    parser.add_argument("--vllm-nccl-version", default="2.28.9")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the 52-cell plan without Docker or writes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.repo = args.repo.resolve()
    args.work_dir = args.work_dir.resolve()
    # Preserve the venv launcher path. Resolving its symlink to /usr/bin/python
    # would lose pyvenv.cfg discovery and therefore the harness dependencies.
    args.python = Path(os.path.abspath(args.python))
    args.dataset = args.dataset.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.trace_bundle = args.trace_bundle.resolve()
    args.env_artifact = args.env_artifact.resolve()
    args.kairyu_template = args.kairyu_template.resolve()
    if not 1 <= args.port <= 65535:
        raise FormalRunError("--port must be in 1..65535")
    if args.startup_timeout_s <= 0 or args.request_timeout_s <= 0:
        raise FormalRunError("timeouts must be positive")

    a6 = load_harness(args.repo)
    binding = binding_plan(a6)
    sweep = sweep_plan(a6)
    all_scenarios = [*binding, *sweep]
    trace_bundle = validate_paths(args, a6)
    parsed_env = parse_environment(args.env_artifact)
    kairyu_template = args.kairyu_template.read_text(encoding="utf-8")
    render_kairyu_config(a6, kairyu_template, 4)
    render_kairyu_config(a6, kairyu_template, 8)
    commit, clean = source_identity(args.repo, require_clean=not args.dry_run)
    version = project_version(args.repo)
    plan_hash = plan_sha256(binding, sweep)
    state_path = args.work_dir / STATE_NAME
    shard_paths = [
        args.work_dir / scenario.shard_relative for scenario in all_scenarios
    ]
    resume_state = read_resume_state(state_path)
    session_id = str(parsed_env["raw"]["measurement_session_id"])
    if resume_state is not None and resume_state.get("session_id") != session_id:
        raise FormalRunError(
            "resume state session differs from the raw environment measurement session"
        )

    if args.dry_run:
        inventory = dry_gpu_inventory(parsed_env)
        validate_inventory_against_environment(inventory, parsed_env)
        checkpoint_value, runtime_identities_for_dry = (
            dry_attestation_identities(
                a6=a6,
                args=args,
                version=version,
                resume_state=resume_state,
            )
        )
        expected = static_state(
            session_id=session_id,
            commit=commit,
            plan_hash=plan_hash,
            env_path=args.env_artifact,
            env_sha256=parsed_env["sha256"],
            kairyu_ref=args.kairyu_image,
            kairyu_id=resume_image_id(
                resume_state,
                "kairyu",
                "sha256:" + "1" * 64,
            ),
            vllm_ref=args.vllm_image,
            vllm_id=resume_image_id(
                resume_state,
                "vllm-stock",
                a6.VLLM_IMAGE_ID,
            ),
            kairyu_version=version,
            checkpoint=checkpoint_value,
            runtime_identities=runtime_identities_for_dry,
            args=args,
        )
    else:
        assert clean
        inventory = live_gpu_inventory()
        validate_inventory_against_environment(inventory, parsed_env)
        docker_volume_exists(args.model_volume)
        if resume_state is None:
            kairyu_id = docker_image_id(args.kairyu_image)
            vllm_id = docker_image_id(args.vllm_image)
        else:
            kairyu_id = resume_image_id(
                resume_state,
                "kairyu",
                "",
            )
            vllm_id = resume_image_id(
                resume_state,
                "vllm-stock",
                "",
            )
            if docker_image_id(kairyu_id) != kairyu_id:
                raise FormalRunError("retained Kairyu image ID is unavailable")
            if docker_image_id(vllm_id) != vllm_id:
                raise FormalRunError("retained vLLM image ID is unavailable")
        if args.vllm_image != a6.VLLM_IMAGE_REF or vllm_id != a6.VLLM_IMAGE_ID:
            raise FormalRunError(
                "stock vLLM ref/image ID differs from the exact harness pin"
            )
        checkpoint = checkpoint_attestation(
            a6=a6,
            model_volume=args.model_volume,
            image=kairyu_id,
        )
        runtime_identities = {
            "kairyu": image_runtime_versions(kairyu_id, "kairyu"),
            "vllm-stock": image_runtime_versions(
                vllm_id,
                "vllm-stock",
            ),
        }
        for arm, expected_runtime in (
            (
                "kairyu",
                {
                    "cuda_version": args.kairyu_cuda_version,
                    "nccl_version": args.kairyu_nccl_version,
                },
            ),
            (
                "vllm-stock",
                {
                    "cuda_version": args.vllm_cuda_version,
                    "nccl_version": args.vllm_nccl_version,
                },
            ),
        ):
            validate_runtime_versions(
                observed=runtime_identities[arm],
                expected=expected_runtime,
                arm=arm,
            )
        if runtime_identities["kairyu"]["package_versions"]["kairyu"] != version:
            raise FormalRunError("Kairyu image package version differs from pyproject")
        if (
            runtime_identities["vllm-stock"]["package_versions"]["vllm"]
            != args.vllm_version
        ):
            raise FormalRunError("vLLM image package version differs from the pin")
        for arm, uvicorn_version in (
            ("kairyu", a6.KAIRYU_UVICORN_VERSION),
            ("vllm-stock", a6.VLLM_UVICORN_VERSION),
        ):
            packages = runtime_identities[arm]["package_versions"]
            expected_http = {
                "uvloop": a6.UVLOOP_VERSION,
                "httptools": a6.HTTPTOOLS_VERSION,
                "uvicorn": uvicorn_version,
            }
            observed_http = {
                key: packages.get(key) for key in expected_http
            }
            if observed_http != expected_http:
                raise FormalRunError(
                    f"{arm} HTTP runtime differs from the formal pin: "
                    f"observed={observed_http}, expected={expected_http}"
                )
        expected = static_state(
            session_id=session_id,
            commit=commit,
            plan_hash=plan_hash,
            env_path=args.env_artifact,
            env_sha256=parsed_env["sha256"],
            kairyu_ref=args.kairyu_image,
            kairyu_id=kairyu_id,
            vllm_ref=args.vllm_image,
            vllm_id=vllm_id,
            kairyu_version=version,
            checkpoint=checkpoint,
            runtime_identities=runtime_identities,
            args=args,
        )

    state = load_or_create_state(
        path=state_path,
        expected=expected,
        shard_paths=shard_paths,
        dry_run=args.dry_run,
    )
    status = validate_resume(
        a6=a6,
        binding=binding,
        sweep=sweep,
        args=args,
        trace_bundle=trace_bundle,
        state=state,
    )

    if args.dry_run:
        dry_validate_provenance(
            a6=a6,
            binding=binding,
            sweep=sweep,
            state=state,
            parsed_env=parsed_env,
            inventory=inventory,
            repo=args.repo,
            env_path=args.env_artifact,
            kairyu_template=kairyu_template,
            model_volume=args.model_volume,
            published_host_port=args.port,
        )
        for index, scenario in enumerate(all_scenarios, 1):
            action = "RESUME-SKIP" if status[scenario.key] else "RUN"
            name = container_name(str(state["session_id"]), scenario)
            config = (
                args.work_dir / "configs" / f"{scenario.key}.yaml"
            )
            docker_command = (
                kairyu_docker_command(
                    args=args,
                    scenario=scenario,
                    name=name,
                    config_path=config,
                    image_id=str(state["images"]["kairyu"]["id"]),
                )
                if scenario.arm == "kairyu"
                else vllm_docker_command(
                    a6=a6,
                    args=args,
                    scenario=scenario,
                    name=name,
                    image_id=str(state["images"]["vllm-stock"]["id"]),
                )
            )
            measure = measure_command(
                a6=a6,
                args=args,
                scenario=scenario,
                provenance_path=(
                    args.work_dir / "provenance" / f"{scenario.key}.json"
                ),
                output_path=args.work_dir / scenario.shard_relative,
            )
            print(
                f"DRY {index:02d}/52 {action} {scenario.key}\n"
                f"  server: {command_text(docker_command)}\n"
                f"  measure: {command_text(measure)}"
            )
        assemble = assembly_command(args=args, binding=binding, sweep=sweep)
        print(f"DRY assemble: {command_text(assemble)}")
        print(f"DRY verify: {command_text(verify_command(args))}")
        print(
            "DRY VALIDATION PASSED: pinned inputs, exact 32+20 plan, "
            "provenance schema, resume safety, and commands are valid."
        )
        return 0

    for index, scenario in enumerate(all_scenarios, 1):
        if status[scenario.key]:
            print(f"[{index:02d}/52] RESUME-SKIP {scenario.key}", flush=True)
            continue
        print(f"[{index:02d}/52] scheduled {scenario.key}", flush=True)
        run_scenario(
            a6=a6,
            args=args,
            scenario=scenario,
            state=state,
            parsed_env=parsed_env,
            inventory=inventory,
            trace_bundle=trace_bundle,
            kairyu_template=kairyu_template,
        )

    assert_source_identity(args.repo, str(state["repo_commit"]))
    checkpoint_after = checkpoint_attestation(
        a6=a6,
        model_volume=args.model_volume,
        image=str(state["images"]["kairyu"]["id"]),
    )
    if checkpoint_after != state["checkpoint"]:
        raise FormalRunError("live checkpoint changed between preflight and assembly")
    assemble = run_command(
        assembly_command(args=args, binding=binding, sweep=sweep)
    )
    atomic_write_text(
        args.work_dir / "assemble.stdout.json",
        (assemble.stdout or "").strip() + "\n",
    )
    verify = run_command(verify_command(args))
    atomic_write_text(
        args.work_dir / "verify.stdout.json",
        (verify.stdout or "").strip() + "\n",
    )
    manifest_path = args.work_dir / "artifact" / a6.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed") is not True:
        raise FormalRunError("assembled manifest did not pass every gate")
    atomic_write_json(
        args.work_dir / "completed.json",
        {
            "schema_version": SCRIPT_SCHEMA,
            "session_id": state["session_id"],
            "repo_commit": state["repo_commit"],
            "checkpoint_attestation_sha256": checkpoint_after["sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "completed_at_ns": time.time_ns(),
        },
    )
    print(
        f"FORMAL G2 A6 PASSED: {manifest_path} "
        f"sha256={sha256_file(manifest_path)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormalRunError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
