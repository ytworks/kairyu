"""Formal G2 A9 DP=2xTP4 versus TP8 crossover report.

The DP arm is the immutable, independently replayed G2 A8 measurement.  The
live ``measure`` command runs only the missing TP8 arm with the same Qwen3-32B
checkpoint, source package, image, trace, request namespace, per-engine
cache/scheduling envelope, and open-loop grid.  The report separately discloses
the two-replica DP arm's doubled aggregate configured capacity.  ``verify`` and
``replay`` recompute every value from the A8 raw artifact and this operator's
TP8 raw JSONL.

A9 is report-only: ``passed`` means the evidence is complete and internally
consistent, never that one topology met a post-hoc performance threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import tarfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from bench import dp_scaling_g2_a8_bench as a8
from kairyu.bench.reporting import nearest_rank_percentile

SCHEMA_VERSION = "kairyu.g2.a9.dp-tp-crossover.v1"
GATE = "G2-A9"
MODEL = a8.MODEL
MODEL_REVISION = a8.MODEL_REVISION
RAW_NAME = "g2-a9-tp8-raw.jsonl"
MANIFEST_NAME = "g2-a9-dp-tp-crossover-manifest.json"
TPOT_METHOD = "stream-terminal-token-v1"
TPOT_DEFINITION = (
    "(stream_terminal_ns - first_content_ns) / (completion_tokens - 1); "
    "includes final usage, finish, [DONE], and stream-terminal tail"
)

A8_SOURCE_COMMIT = "4924b4d71b8fae0af087979908819aed6939a871"
A8_SOURCE_ARCHIVE_SHA256 = (
    "3e4adc43898caba5d820728926d83c16a97bd3c33debe52f9164ab8056f8db29"
)
A8_RUNTIME_FILE_COUNT = 196
A8_RUNTIME_FILES_SHA256 = (
    "e253ab1c33cf281bae6f0989459214ae195e024827244e35116c48ecfe0d907a"
)
A8_IMAGE_ID = "sha256:2c73b577b5213264493c6aeba24c1f6318214d20bf6df7780158fd1ef2c70a50"
A8_RAW_SHA256 = "b637489302a9b818a0c34790c4059946994ff6070f76ac9e1bc7d128bbbd803f"
A8_MANIFEST_SHA256 = (
    "da439f153c04d05178ddf96c489aca7fb1cc270ba982736c1bc98197730e3946"
)
A8_PLACEMENTS_SHA256 = (
    "76ca1a5f709ccf8492238bdcc4193776f94fdb7a88a40a7e970a517db30270c3"
)
A8_ARTIFACT_DIR = Path(
    "bench/results/g2-a8-dp-qwen3-32b-rtxpro6000-2026-07-31"
)
A8_ACCEPTED_FALSE_CHECK = "median_peak_goodput_ratio_at_least_1_9"
A8_HELPER_PATH = Path("bench/dp_scaling_g2_a8_bench.py")
A8_HELPER_SHA256 = (
    "d86be05d8815ec50f65f3ab51d450764aa4c20b7acd17c219c4190f0287192f5"
)

RATES_RPS = a8.RATES_RPS
REPEATS = a8.REPEATS
REQUESTS_PER_CELL = a8.SCALING_REQUESTS
OUTPUT_TOKENS = a8.SCALING_OUTPUT_TOKENS
WARMUP_REQUESTS = a8.WARMUP_REQUESTS
TTFT_SLO_NS = a8.TTFT_SLO_NS
PERCENTILE_METHOD = a8.PERCENTILE_METHOD
IMAGE_ID_RE = a8.IMAGE_ID_RE
HEX64_RE = a8.HEX64_RE
COMMIT_RE = a8.COMMIT_RE

CONFIG_PATHS = (
    Path("bench/g2_a9_dp_tp_crossover_bench.py"),
    A8_HELPER_PATH,
    Path("examples/qwen3-32b-multi-gpu/a9-tp8-compose.yaml"),
    Path("examples/qwen3-32b-multi-gpu/a9-tp8-replica.yaml"),
    Path("examples/qwen3-32b-multi-gpu/a9-tp8-gateway.yaml"),
    Path("examples/qwen3-32b-multi-gpu/a9-tp8-stack.sh"),
)
RUNTIME_PACKAGE_HASH_SCRIPT = r"""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    payload = path.read_bytes()
    files.append(
        {
            "path": f"kairyu/{path.relative_to(root).as_posix()}",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
encoded = json.dumps(
    files,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode()
print(
    json.dumps(
        {
            "file_count": len(files),
            "files_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        sort_keys=True,
    )
)
"""


class GateEvidenceError(ValueError):
    """Raised when retained evidence cannot prove the report."""


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


def _require(condition: object, message: str) -> None:
    if not condition:
        raise GateEvidenceError(message)


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    _require(
        type(value) is int and value >= minimum,
        f"{name} must be an integer >= {minimum}",
    )
    return int(value)


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{name} must be finite",
    )
    number = float(value)
    if minimum is not None:
        _require(number >= minimum, f"{name} must be >= {minimum}")
    return number


def _hash_valid(value: object) -> bool:
    return isinstance(value, str) and HEX64_RE.fullmatch(value) is not None


def _physical_gpu_identity(value: object) -> list[tuple[object, ...]]:
    _require(
        isinstance(value, list) and len(value) == 8,
        "A9 physical GPU inventory is incomplete",
    )
    identity: list[tuple[object, ...]] = []
    for row in value:
        _require(
            isinstance(row, dict),
            "A9 physical GPU inventory contains a non-object row",
        )
        identity.append(
            (
                row.get("index"),
                row.get("uuid"),
                row.get("pci_bus_id"),
                row.get("name"),
                row.get("memory_total_mib"),
            )
        )
    return identity


def formal_config() -> dict[str, object]:
    return {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "arms": ["dp2-tp4", "tp8"],
        "rates_rps": list(RATES_RPS),
        "repeats": REPEATS,
        "requests_per_cell": REQUESTS_PER_CELL,
        "output_tokens": OUTPUT_TOKENS,
        "tp8_warmup_requests_per_repeat": WARMUP_REQUESTS,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "retry_count": 0,
        "ttft_slo_ns": TTFT_SLO_NS,
        "goodput_definition": (
            "exact successful completions with TTFT <= SLO divided by "
            "first-scheduled-to-last-terminal wall interval"
        ),
        "tpot_method": TPOT_METHOD,
        "tpot_definition": TPOT_DEFINITION,
        "percentile_method": PERCENTILE_METHOD,
        "topology_capacity": {
            "dp2-tp4": {
                "replicas": 2,
                "tensor_parallel_size_per_replica": 4,
                "usable_kv_pages_per_replica": 8192,
                "aggregate_usable_kv_pages": 16384,
                "max_num_seqs_per_replica": 16,
                "aggregate_max_num_seqs": 32,
                "max_num_batched_tokens_per_replica": 1024,
                "aggregate_max_num_batched_tokens": 2048,
                "cuda_graph_max_batch_per_replica": 16,
                "aggregate_cuda_graph_max_batch": 32,
            },
            "tp8": {
                "replicas": 1,
                "tensor_parallel_size_per_replica": 8,
                "usable_kv_pages_per_replica": 8192,
                "aggregate_usable_kv_pages": 8192,
                "max_num_seqs_per_replica": 16,
                "aggregate_max_num_seqs": 16,
                "max_num_batched_tokens_per_replica": 1024,
                "aggregate_max_num_batched_tokens": 1024,
                "cuda_graph_max_batch_per_replica": 16,
                "aggregate_cuda_graph_max_batch": 16,
            },
            "interpretation": (
                "per-engine scheduler and cache limits are matched; DP owns "
                "two independent replicas and therefore twice the aggregate "
                "configured capacity"
            ),
        },
        "verdict": "report-only; no numeric performance threshold",
    }


def _validate_formal_config(value: object) -> None:
    _require(
        value == formal_config(),
        "formal config differs from the fixed G2 A9 contract",
    )


def _run_command(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"{' '.join(arguments)} failed with exit "
            f"{completed.returncode}: {detail}"
        )
    return completed.stdout.strip()


def _expected_runtime_package() -> dict[str, object]:
    completed = subprocess.run(
        ("git", "archive", "--format=tar", A8_SOURCE_COMMIT, "kairyu"),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"cannot archive A8 runtime package: {detail}"
        )
    _require(
        sha256_bytes(completed.stdout) == A8_SOURCE_ARCHIVE_SHA256,
        "A8 source archive differs from its fixed digest",
    )
    files: list[dict[str, object]] = []
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(
                    f"cannot read archived runtime file {member.name}"
                )
            payload = extracted.read()
            files.append(
                {
                    "path": member.name,
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )
    files_sha256 = sha256_bytes(canonical_json(files).encode())
    _require(
        len(files) == A8_RUNTIME_FILE_COUNT
        and files_sha256 == A8_RUNTIME_FILES_SHA256,
        "A8 runtime file tree differs from its fixed rollup",
    )
    return {
        "source_commit": A8_SOURCE_COMMIT,
        "source_archive_sha256": A8_SOURCE_ARCHIVE_SHA256,
        "file_count": len(files),
        "files_sha256": files_sha256,
    }


def _runtime_package_reference(container_id: str) -> dict[str, object]:
    completed = subprocess.run(
        (
            "docker",
            "exec",
            container_id,
            "python",
            "-c",
            RUNTIME_PACKAGE_HASH_SCRIPT,
            "/app/kairyu",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"A9 runtime package attestation failed: {detail}"
        )
    measured = json.loads(completed.stdout)
    expected = _expected_runtime_package()
    if not isinstance(measured, dict) or measured != {
        "file_count": expected["file_count"],
        "files_sha256": expected["files_sha256"],
    }:
        raise RuntimeError(
            "A9 mounted runtime package differs from A8 source"
        )
    return {
        **expected,
        "attested_container_id": container_id,
        "source": f"bind:git:{A8_SOURCE_COMMIT}:kairyu",
    }


def _config_file_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in CONFIG_PATHS:
        if not path.is_file():
            raise RuntimeError(f"formal source file missing: {path}")
        result[path.as_posix()] = sha256_file(path)
    return result


def _git_blob_sha256(commit: str, path: Path) -> str:
    if COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("cannot resolve a file from an invalid commit")
    completed = subprocess.run(
        ("git", "show", f"{commit}:{path.as_posix()}"),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"source commit does not contain {path.as_posix()}: {detail}"
        )
    return sha256_bytes(completed.stdout)


def _a8_helper_reference(source_commit: str) -> dict[str, object]:
    pinned_sha256 = _git_blob_sha256(A8_SOURCE_COMMIT, A8_HELPER_PATH)
    current_sha256 = _git_blob_sha256(source_commit, A8_HELPER_PATH)
    _require(
        pinned_sha256 == A8_HELPER_SHA256
        and current_sha256 == pinned_sha256,
        "A9 delegated A8 helper differs from the measured A8 source",
    )
    return {
        "path": A8_HELPER_PATH.as_posix(),
        "source_commit": A8_SOURCE_COMMIT,
        "sha256": pinned_sha256,
    }


def _committed_config_file_hashes(commit: str) -> dict[str, str]:
    if COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("cannot resolve config files from an invalid commit")
    return {
        path.as_posix(): _git_blob_sha256(commit, path)
        for path in CONFIG_PATHS
    }


def _a8_descriptor(artifact_dir: Path) -> dict[str, object]:
    raw_path = artifact_dir / a8.RAW_NAME
    manifest_path = artifact_dir / a8.MANIFEST_NAME
    placements_path = artifact_dir / "placements.jsonl"
    _require(raw_path.is_file(), f"missing A8 raw artifact: {raw_path}")
    _require(manifest_path.is_file(), f"missing A8 manifest: {manifest_path}")
    _require(
        placements_path.is_file(),
        f"missing A8 placement artifact: {placements_path}",
    )
    raw_sha = sha256_file(raw_path)
    manifest_sha = sha256_file(manifest_path)
    placements_sha = sha256_file(placements_path)
    _require(raw_sha == A8_RAW_SHA256, "A8 raw SHA-256 differs from its pin")
    _require(
        manifest_sha == A8_MANIFEST_SHA256,
        "A8 manifest SHA-256 differs from its pin",
    )
    _require(
        placements_sha == A8_PLACEMENTS_SHA256,
        "A8 placements SHA-256 differs from its pin",
    )
    return {
        "path": A8_ARTIFACT_DIR.as_posix(),
        "raw_sha256": raw_sha,
        "manifest_sha256": manifest_sha,
        "placements_sha256": placements_sha,
        "source_commit": A8_SOURCE_COMMIT,
        "image_id": A8_IMAGE_ID,
    }


def load_a8_dp_evidence(
    artifact_dir: Path = A8_ARTIFACT_DIR,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Verify A8 independently and return its formal DP scaling samples."""

    descriptor = _a8_descriptor(artifact_dir)
    manifest = a8.verify_artifact(artifact_dir, assert_gate=False)
    _require(manifest.get("passed") is False, "A8 verdict unexpectedly changed")
    checks = manifest.get("checks")
    _require(isinstance(checks, dict), "A8 checks are missing")
    failed = {name for name, passed in checks.items() if passed is not True}
    _require(
        failed == {A8_ACCEPTED_FALSE_CHECK},
        "A8 must retain only its accepted 1.9x scaling deviation",
    )
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, dict), "A8 provenance is missing")
    _require(
        provenance.get("source_commit") == A8_SOURCE_COMMIT,
        "A8 source commit differs from its pin",
    )
    _require(
        provenance.get("image_id") == A8_IMAGE_ID,
        "A8 image differs from its pin",
    )
    _require(
        provenance.get("model") == MODEL
        and provenance.get("model_revision") == MODEL_REVISION,
        "A8 model identity differs from A9",
    )
    checkpoint = provenance.get("checkpoint")
    _require(isinstance(checkpoint, dict), "A8 checkpoint evidence is missing")
    _require(
        checkpoint.get("weights_sha256") == a8.CHECKPOINT_WEIGHTS_SHA256
        and checkpoint.get("index_sha256") == a8.CHECKPOINT_INDEX_SHA256
        and checkpoint.get("tokenizer_sha256") == a8.PINNED_TOKENIZER_SHA256,
        "A8 checkpoint identity differs from its pins",
    )
    rows = a8._read_jsonl(artifact_dir / a8.RAW_NAME)
    dp_rows = [
        dict(row)
        for row in rows
        if row.get("type") == "request"
        and row.get("phase") == "scaling"
        and row.get("arm") == "dp"
    ]
    _require(
        len(dp_rows) == REPEATS * len(RATES_RPS) * REQUESTS_PER_CELL,
        "A8 DP scaling matrix is incomplete",
    )
    return dp_rows, manifest, descriptor


def _normalize_api_root(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid HTTP endpoint: {value!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        pass
    elif not path:
        path = "/v1"
    else:
        path = f"{path}/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _service_origin(api_root: str) -> str:
    parsed = urlsplit(api_root)
    path = parsed.path.removesuffix("/v1")
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))


async def _endpoint_metadata(
    client: httpx.AsyncClient,
    api_root: str,
    *,
    expected_role: str,
) -> dict[str, object]:
    origin = _service_origin(api_root)
    ready = await client.get(f"{origin}/readyz")
    if ready.status_code != 200:
        raise RuntimeError(f"{origin}/readyz returned HTTP {ready.status_code}")
    response = await client.get(f"{origin}/backends")
    if response.status_code != 200:
        raise RuntimeError(f"{origin}/backends returned HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("role") != expected_role:
        raise RuntimeError(f"{origin}/backends does not report role={expected_role}")
    engines = payload.get("engines")
    if not isinstance(engines, list) or len(engines) != 1:
        raise RuntimeError(f"{origin}/backends must report one qwen3-32b engine")
    engine = engines[0]
    if not isinstance(engine, dict) or engine.get("model") != MODEL:
        raise RuntimeError(f"{origin}/backends model mismatch")
    if expected_role == "engine-host":
        topology_tp = engine.get("tensor_parallel_size")
        backend = engine.get("engine_backend")
    else:
        via_replica = engine.get("via_replica")
        topology_tp = (
            via_replica.get("tensor_parallel_size")
            if isinstance(via_replica, dict)
            else None
        )
        backend = engine.get("engine_backend")
    if topology_tp != 8:
        raise RuntimeError(f"{origin}/backends does not prove TP8")
    expected_backend = "kairyu" if expected_role == "engine-host" else "replica-pool"
    if backend != expected_backend:
        raise RuntimeError(f"{origin}/backends engine backend mismatch")
    return payload


def _validate_backends_payload(value: object, *, role: str) -> None:
    _require(
        isinstance(value, dict) and value.get("role") == role,
        f"{role} /backends invalid",
    )
    engines = value.get("engines")
    _require(
        isinstance(engines, list) and len(engines) == 1,
        f"{role} engine matrix invalid",
    )
    engine = engines[0]
    _require(
        isinstance(engine, dict) and engine.get("model") == MODEL,
        f"{role} model identity invalid",
    )
    if role == "engine-host":
        _require(
            engine.get("engine_backend") == "kairyu"
            and engine.get("tensor_parallel_size") == 8,
            "engine /backends does not prove Kairyu TP8",
        )
    else:
        via_replica = engine.get("via_replica")
        _require(
            engine.get("engine_backend") == "replica-pool"
            and isinstance(via_replica, dict)
            and via_replica.get("tensor_parallel_size") == 8,
            "gateway /backends does not prove one TP8 replica",
        )


def _expected_normalized_mounts(service: str) -> list[dict[str, object]]:
    config_name = "replica" if service == "tp8" else "gateway"
    mounts: list[dict[str, object]] = [
        {
            "type": "bind",
            "source": f"git:{A8_SOURCE_COMMIT}:kairyu",
            "destination": "/app/kairyu",
            "name": None,
            "rw": False,
        },
        {
            "type": "bind",
            "source": (
                "repo:examples/qwen3-32b-multi-gpu/"
                f"a9-tp8-{config_name}.yaml"
            ),
            "destination": "/etc/kairyu/config.yaml",
            "name": None,
            "rw": False,
        }
    ]
    if service == "tp8":
        mounts.append(
            {
                "type": "volume",
                "source": "volume:kairyu-qwen3-32b_qwen3-32b",
                "destination": "/models",
                "name": "kairyu-qwen3-32b_qwen3-32b",
                "rw": False,
            }
        )
    elif service == "gateway":
        mounts.append(
            {
                "type": "bind",
                "source": "artifact:run-dir",
                "destination": "/evidence",
                "name": None,
                "rw": True,
            }
        )
    else:
        raise RuntimeError(f"unknown A9 service: {service}")
    return sorted(mounts, key=lambda mount: str(mount["destination"]))


def _runtime_source_dir(evidence_dir: Path) -> Path:
    resolved = evidence_dir.resolve()
    return (
        resolved.parent
        / f".{resolved.name}.a8-source-{A8_SOURCE_COMMIT}"
        / "kairyu"
    )


def _docker_stack_provenance(
    image_id: str,
    *,
    evidence_dir: Path,
) -> dict[str, object]:
    compose_path = Path(
        "examples/qwen3-32b-multi-gpu/a9-tp8-compose.yaml"
    ).resolve()
    compose_prefix = (
        "docker",
        "compose",
        "--project-name",
        "kairyu-qwen3-32b-a9-tp8",
        "-f",
        str(compose_path),
    )
    expected_devices = {
        "tp8": [str(index) for index in range(8)],
        "gateway": [],
    }
    expected_cpus = {
        "tp8": (
            "0-14,16-30,32-46,48-62,64-78,80-94,96-110,112-126"
        ),
        "gateway": "15,31,47,63,79,95,111,127",
    }
    expected_ports = {"tp8": "8300", "gateway": "8301"}
    expected_configs = {
        "tp8": str(
            Path(
                "examples/qwen3-32b-multi-gpu/a9-tp8-replica.yaml"
            ).resolve()
        ),
        "gateway": str(
            Path(
                "examples/qwen3-32b-multi-gpu/a9-tp8-gateway.yaml"
            ).resolve()
        ),
    }
    expected_runtime_source = str(_runtime_source_dir(evidence_dir))
    services: dict[str, object] = {}
    for service in ("tp8", "gateway"):
        environment = {
            **os.environ,
            "KAIRYU_A9_RUN_DIR": str(evidence_dir),
            "KAIRYU_A9_RUNTIME_DIR": expected_runtime_source,
        }
        config_hash_result = subprocess.run(
            (*compose_prefix, "config", "--hash", service),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if config_hash_result.returncode != 0:
            detail = (
                config_hash_result.stderr or config_hash_result.stdout
            ).strip()
            raise RuntimeError(
                f"Compose config hash for {service} failed: {detail}"
            )
        fields = config_hash_result.stdout.strip().split()
        if (
            len(fields) != 2
            or fields[0] != service
            or not _hash_valid(fields[1])
        ):
            raise RuntimeError(f"Compose config hash for {service} is invalid")
        compose_config_sha256 = fields[1]
        ps_result = subprocess.run(
            (*compose_prefix, "ps", "-q", service),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if ps_result.returncode != 0:
            detail = (ps_result.stderr or ps_result.stdout).strip()
            raise RuntimeError(
                f"Compose ps for {service} failed: {detail}"
            )
        container_id = ps_result.stdout.strip()
        if not container_id or "\n" in container_id:
            raise RuntimeError(
                f"A9 service {service} does not resolve to one container"
            )
        inspected = json.loads(
            _run_command(("docker", "inspect", container_id))
        )
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise RuntimeError(f"docker inspect for {service} is not singular")
        value = inspected[0]
        if not isinstance(value, dict):
            raise RuntimeError(f"docker inspect for {service} is invalid")
        config = value.get("Config")
        host = value.get("HostConfig")
        state = value.get("State")
        mounts = value.get("Mounts")
        if not all(
            isinstance(item, dict) for item in (config, host, state)
        ) or not isinstance(mounts, list):
            raise RuntimeError(f"docker inspect for {service} is incomplete")
        labels = config.get("Labels")
        if not isinstance(labels, dict):
            raise RuntimeError(f"docker labels for {service} are missing")
        if (
            value.get("Image") != image_id
            or labels.get("com.kairyu.a9.image-id") != image_id
            or labels.get("com.kairyu.a9.role") != service
            or labels.get("com.docker.compose.config-hash")
            != compose_config_sha256
            or state.get("Running") is not True
            or not isinstance(state.get("StartedAt"), str)
            or not state["StartedAt"]
            or host.get("CpusetCpus") != expected_cpus[service]
        ):
            raise RuntimeError(f"A9 runtime identity mismatch for {service}")
        device_requests = host.get("DeviceRequests") or []
        device_ids: list[str] = []
        if service == "tp8":
            if not isinstance(device_requests, list) or len(device_requests) != 1:
                raise RuntimeError("A9 TP8 GPU request is missing")
            request = device_requests[0]
            if (
                not isinstance(request, dict)
                or request.get("Driver") != "nvidia"
                or request.get("Capabilities") != [["gpu"]]
                or request.get("DeviceIDs") != expected_devices[service]
            ):
                raise RuntimeError("A9 TP8 GPU set differs from ordinals 0-7")
            device_ids = list(request["DeviceIDs"])
        elif device_requests:
            raise RuntimeError("A9 gateway unexpectedly owns a GPU")

        retained_mounts: list[dict[str, object]] = []
        for mount in mounts:
            if not isinstance(mount, dict):
                raise RuntimeError(f"A9 mount entry for {service} is invalid")
            destination = mount.get("Destination")
            if destination not in {
                "/app/kairyu",
                "/models",
                "/etc/kairyu/config.yaml",
                "/evidence",
            }:
                continue
            retained_mounts.append(
                {
                    "type": mount.get("Type"),
                    "source": mount.get("Source"),
                    "destination": destination,
                    "name": mount.get("Name"),
                    "rw": mount.get("RW"),
                }
            )
        by_destination = {
            str(mount["destination"]): mount for mount in retained_mounts
        }
        expected_destinations = (
            {"/app/kairyu", "/etc/kairyu/config.yaml", "/models"}
            if service == "tp8"
            else {"/app/kairyu", "/etc/kairyu/config.yaml", "/evidence"}
        )
        if (
            set(by_destination) != expected_destinations
            or len(retained_mounts) != len(expected_destinations)
        ):
            raise RuntimeError(f"A9 mount matrix mismatch for {service}")
        config_mount = by_destination.get("/etc/kairyu/config.yaml")
        if (
            config_mount is None
            or config_mount.get("type") != "bind"
            or config_mount.get("source") != expected_configs[service]
            or config_mount.get("name") is not None
            or config_mount.get("rw") is not False
        ):
            raise RuntimeError(f"A9 read-only config mismatch for {service}")
        source_mount = by_destination.get("/app/kairyu")
        if (
            source_mount is None
            or source_mount.get("type") != "bind"
            or source_mount.get("source") != expected_runtime_source
            or source_mount.get("name") is not None
            or source_mount.get("rw") is not False
        ):
            raise RuntimeError(
                f"A9 pinned A8 source mount mismatch for {service}"
            )
        if service == "tp8":
            model_mount = by_destination.get("/models")
            if (
                model_mount is None
                or model_mount.get("type") != "volume"
                or model_mount.get("name")
                != "kairyu-qwen3-32b_qwen3-32b"
                or model_mount.get("rw") is not False
            ):
                raise RuntimeError("A9 checkpoint volume identity is invalid")
        else:
            evidence_mount = by_destination.get("/evidence")
            if (
                evidence_mount is None
                or evidence_mount.get("type") != "bind"
                or evidence_mount.get("source") != str(evidence_dir)
                or evidence_mount.get("name") is not None
                or evidence_mount.get("rw") is not True
            ):
                raise RuntimeError("A9 gateway evidence mount is invalid")
        ports = host.get("PortBindings")
        binding = ports.get("8000/tcp") if isinstance(ports, dict) else None
        if (
            not isinstance(binding, list)
            or len(binding) != 1
            or binding[0].get("HostIp") != "127.0.0.1"
            or binding[0].get("HostPort") != expected_ports[service]
        ):
            raise RuntimeError(f"A9 port binding mismatch for {service}")
        normalized_sources = {
            "/app/kairyu": f"git:{A8_SOURCE_COMMIT}:kairyu",
            "/etc/kairyu/config.yaml": (
                "repo:examples/qwen3-32b-multi-gpu/"
                f"a9-tp8-{'replica' if service == 'tp8' else 'gateway'}.yaml"
            ),
            "/models": "volume:kairyu-qwen3-32b_qwen3-32b",
            "/evidence": "artifact:run-dir",
        }
        normalized_mounts = [
            {
                **mount,
                "source": normalized_sources[str(mount["destination"])],
            }
            for mount in retained_mounts
        ]
        normalized_mounts = sorted(
            normalized_mounts,
            key=lambda mount: str(mount["destination"]),
        )
        if normalized_mounts != _expected_normalized_mounts(service):
            raise RuntimeError(
                f"A9 normalized mount matrix mismatch for {service}"
            )
        services[service] = {
            "container_id": container_id,
            "image_id": value.get("Image"),
            "running": state.get("Running"),
            "started_at": state.get("StartedAt"),
            "compose_config_sha256": compose_config_sha256,
            "device_ids": device_ids,
            "cpuset_cpus": host.get("CpusetCpus"),
            "port": expected_ports[service],
            "mounts": normalized_mounts,
        }
    return {
        "project": "kairyu-qwen3-32b-a9-tp8",
        "compose_file": (
            "repo:examples/qwen3-32b-multi-gpu/a9-tp8-compose.yaml"
        ),
        "services": services,
    }


def _validate_container_provenance(value: object) -> None:
    _require(isinstance(value, dict), "A9 container provenance missing")
    _require(
        value.get("project") == "kairyu-qwen3-32b-a9-tp8",
        "A9 Compose project mismatch",
    )
    _require(
        value.get("compose_file")
        == "repo:examples/qwen3-32b-multi-gpu/a9-tp8-compose.yaml",
        "A9 Compose file identity mismatch",
    )
    services = value.get("services")
    _require(
        isinstance(services, dict) and set(services) == {"tp8", "gateway"},
        "A9 service matrix mismatch",
    )
    expected_devices = {
        "tp8": [str(index) for index in range(8)],
        "gateway": [],
    }
    expected_cpus = {
        "tp8": (
            "0-14,16-30,32-46,48-62,64-78,80-94,96-110,112-126"
        ),
        "gateway": "15,31,47,63,79,95,111,127",
    }
    expected_ports = {"tp8": "8300", "gateway": "8301"}
    for service in ("tp8", "gateway"):
        descriptor = services[service]
        _require(
            isinstance(descriptor, dict),
            f"A9 {service} descriptor invalid",
        )
        _require(
            isinstance(descriptor.get("container_id"), str)
            and HEX64_RE.fullmatch(str(descriptor["container_id"])) is not None
            and descriptor.get("image_id") == A8_IMAGE_ID
            and descriptor.get("running") is True
            and isinstance(descriptor.get("started_at"), str)
            and bool(descriptor["started_at"])
            and _hash_valid(descriptor.get("compose_config_sha256"))
            and descriptor.get("device_ids") == expected_devices[service]
            and descriptor.get("cpuset_cpus") == expected_cpus[service]
            and descriptor.get("port") == expected_ports[service],
            f"A9 {service} runtime identity mismatch",
        )
        mounts = descriptor.get("mounts")
        _require(
            isinstance(mounts, list)
            and mounts == _expected_normalized_mounts(service),
            f"A9 {service} mount evidence invalid",
        )


def _remap_tp8_row(row: Mapping[str, object]) -> dict[str, object]:
    _require(row.get("arm") == "dp", "TP8 source row arm is invalid")
    return {
        **row,
        "arm": "tp8",
        "namespace_arm": "dp",
    }


def _tp8_as_a8_dp(row: Mapping[str, object]) -> dict[str, object]:
    _require(
        row.get("arm") == "tp8" and row.get("namespace_arm") == "dp",
        "TP8 namespace ownership is invalid",
    )
    value = dict(row)
    value["arm"] = "dp"
    value.pop("namespace_arm", None)
    return value


def _max_in_flight(samples: Sequence[Mapping[str, object]]) -> int:
    events: list[tuple[int, int]] = []
    for row in samples:
        events.append((int(row["started_ns"]), 1))
        events.append((int(row["ended_ns"]), -1))
    # An ending request is no longer in flight when another starts at the same
    # timestamp.
    current = 0
    maximum = 0
    for _timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    _require(current == 0, "in-flight interval accounting is unbalanced")
    return maximum


def _median_absolute_deviation(values: Sequence[float]) -> float:
    _require(bool(values), "MAD requires samples")
    median = float(statistics.median(values))
    return float(statistics.median(abs(value - median) for value in values))


def _cell_metrics(
    samples: Sequence[Mapping[str, object]],
    *,
    arm: str,
    repeat: int,
    rate_rps: int,
) -> dict[str, object]:
    _require(
        len(samples) == REQUESTS_PER_CELL,
        f"{arm} repeat {repeat} rate {rate_rps} request count mismatch",
    )
    by_sequence = {int(row["sequence"]): row for row in samples}
    _require(
        set(by_sequence) == set(range(REQUESTS_PER_CELL)),
        f"{arm} repeat {repeat} rate {rate_rps} sequence matrix mismatch",
    )
    first_scheduled = min(int(row["scheduled_ns"]) for row in samples)
    _require(
        all(
            int(by_sequence[sequence]["scheduled_ns"])
            == first_scheduled
            + round(sequence * 1_000_000_000 / rate_rps)
            for sequence in range(REQUESTS_PER_CELL)
        ),
        f"{arm} repeat {repeat} rate {rate_rps} cadence mismatch",
    )
    last_terminal = max(int(row["ended_ns"]) for row in samples)
    wall_ns = last_terminal - first_scheduled
    _require(wall_ns > 0, "cell wall interval is invalid")
    tpot_ns: list[float] = []
    for row in samples:
        completion_tokens = _exact_int(
            row.get("completion_tokens"),
            "completion_tokens",
            minimum=2,
        )
        _require(
            completion_tokens == OUTPUT_TOKENS,
            "completion length differs from A9",
        )
        first_token_ns = _exact_int(
            row.get("first_token_ns"),
            "first_token_ns",
            minimum=1,
        )
        ended_ns = _exact_int(
            row.get("ended_ns"),
            "ended_ns",
            minimum=first_token_ns,
        )
        tpot_ns.append(
            (ended_ns - first_token_ns) / (completion_tokens - 1)
        )
    slo_successes = sum(
        row.get("status") == "success"
        and int(row["ttft_ns"]) <= TTFT_SLO_NS
        for row in samples
    )
    return {
        "arm": arm,
        "repeat": repeat,
        "rate_rps": rate_rps,
        "request_count": len(samples),
        "slo_successes": slo_successes,
        "wall_ns": wall_ns,
        "goodput_rps": slo_successes * 1_000_000_000 / wall_ns,
        "ttft_p50_ns": nearest_rank_percentile(
            [int(row["ttft_ns"]) for row in samples], 0.50
        ),
        "ttft_p99_ns": nearest_rank_percentile(
            [int(row["ttft_ns"]) for row in samples], 0.99
        ),
        "tpot_mean_ns": statistics.fmean(tpot_ns),
        "tpot_p50_ns": nearest_rank_percentile(tpot_ns, 0.50),
        "tpot_p99_ns": nearest_rank_percentile(tpot_ns, 0.99),
        "observed_max_in_flight": _max_in_flight(samples),
    }


def derive_crossover(
    rate_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Describe every measured DP/TP8 goodput ordering transition."""

    _require(
        len(rate_metrics) == len(RATES_RPS),
        "crossover needs the complete arrival grid",
    )
    ordered = sorted(rate_metrics, key=lambda row: int(row["rate_rps"]))
    _require(
        [int(row["rate_rps"]) for row in ordered] == list(RATES_RPS),
        "crossover arrival grid differs from the fixed grid",
    )
    comparisons: list[int] = []
    observed_concurrency: list[dict[str, float]] = []
    for row in ordered:
        dp = _finite_number(
            row.get("dp_median_goodput_rps"),
            "DP median goodput",
            minimum=0.0,
        )
        tp8 = _finite_number(
            row.get("tp8_median_goodput_rps"),
            "TP8 median goodput",
            minimum=0.0,
        )
        comparisons.append(1 if dp > tp8 else 0 if dp == tp8 else -1)
        observed_concurrency.append(
            {
                "dp_median_max_in_flight": _finite_number(
                    row.get("dp_median_max_in_flight"),
                    "DP median maximum in-flight",
                    minimum=0.0,
                ),
                "tp8_median_max_in_flight": _finite_number(
                    row.get("tp8_median_max_in_flight"),
                    "TP8 median maximum in-flight",
                    minimum=0.0,
                ),
            }
        )
    transitions: list[dict[str, object]] = []
    for index in range(1, len(ordered)):
        if comparisons[index] == comparisons[index - 1]:
            continue
        transitions.append(
            {
                "lower_rate_rps": int(ordered[index - 1]["rate_rps"]),
                "upper_rate_rps": int(ordered[index]["rate_rps"]),
                "lower_order": comparisons[index - 1],
                "upper_order": comparisons[index],
                "lower_observed_concurrency": observed_concurrency[
                    index - 1
                ],
                "upper_observed_concurrency": observed_concurrency[index],
            }
        )
    first_noninferior_index = next(
        (index for index, comparison in enumerate(comparisons) if comparison >= 0),
        None,
    )
    if first_noninferior_index is None:
        first_noninferior_status = "none_in_measured_range"
        first_noninferior_rate = None
        bracket = None
        first_noninferior_concurrency = None
    else:
        first = ordered[first_noninferior_index]
        first_noninferior_rate = int(first["rate_rps"])
        first_noninferior_status = (
            "dp_noninferior_from_min_rate"
            if first_noninferior_index == 0
            else "dp_first_noninferior_within_range"
        )
        bracket = (
            None
            if first_noninferior_index == 0
            else [
                int(ordered[first_noninferior_index - 1]["rate_rps"]),
                first_noninferior_rate,
            ]
        )
        first_noninferior_concurrency = observed_concurrency[
            first_noninferior_index
        ]
    has_order_transition = bool(transitions)
    return {
        "axis": "open_loop_arrival_rate_rps",
        "comparison": "median DP goodput versus median TP8 goodput",
        "order_encoding": {"dp_below": -1, "tie": 0, "dp_above": 1},
        "measured_rates_rps": list(RATES_RPS),
        "status": (
            "order_transition_observed"
            if has_order_transition
            else "no_order_transition_in_measured_range"
        ),
        "has_order_transition": has_order_transition,
        "no_order_transition_in_measured_range": not has_order_transition,
        "first_dp_noninferior_status": first_noninferior_status,
        "first_dp_noninferior_rate_rps": first_noninferior_rate,
        "arrival_rate_bracket_rps": bracket,
        "observed_concurrency_at_first_noninferior_rate": (
            first_noninferior_concurrency
        ),
        "order_by_rate": [
            {
                "rate_rps": int(row["rate_rps"]),
                "order": comparison,
            }
            for row, comparison in zip(ordered, comparisons, strict=True)
        ],
        "all_order_transitions": transitions,
        "interpolation": "none",
    }


def _validate_tp8_request(
    row: Mapping[str, object],
    trace_scaling: Mapping[int, Mapping[str, object]],
) -> None:
    transformed = _tp8_as_a8_dp(row)
    try:
        a8._validate_request_common(transformed, trace_scaling, {})
    except a8.GateEvidenceError as error:
        raise GateEvidenceError(str(error)) from error
    _require(
        isinstance(row.get("response_request_id"), str)
        and bool(row["response_request_id"]),
        "TP8 gateway response X-Request-ID missing",
    )


def _validate_checkpoint_descriptor(
    value: object,
    *,
    a8_checkpoint: object,
    container_id: object,
) -> dict[str, object]:
    _require(
        isinstance(a8_checkpoint, dict),
        "A8 checkpoint descriptor is missing",
    )
    _require(
        isinstance(container_id, str)
        and HEX64_RE.fullmatch(container_id) is not None,
        "A9 checkpoint container identity is invalid",
    )
    expected = {
        **a8_checkpoint,
        "attested_container_id": container_id,
    }
    _require(
        value == expected,
        "A9 checkpoint descriptor differs from A8",
    )
    return dict(expected)


def _validate_provenance(
    value: object,
    *,
    a8_manifest: Mapping[str, object],
    a8_descriptor: Mapping[str, object],
) -> dict[str, object]:
    _require(isinstance(value, dict), "A9 provenance is missing")
    provenance = dict(value)
    _require(
        provenance.get("a8_artifact") == a8_descriptor,
        "A8 artifact provenance differs from the fixed evidence",
    )
    source_commit = provenance.get("source_commit")
    _require(
        isinstance(source_commit, str)
        and COMMIT_RE.fullmatch(source_commit) is not None,
        "A9 source commit is invalid",
    )
    _require(
        provenance.get("tracked_tree_clean") is True,
        "A9 source tree was not clean",
    )
    _require(
        provenance.get("a8_helper")
        == _a8_helper_reference(str(source_commit)),
        "A9 delegated A8 helper differs from the measured A8 source",
    )
    runtime_package = provenance.get("runtime_package")
    expected_runtime_package = _expected_runtime_package()
    _require(
        isinstance(runtime_package, dict)
        and runtime_package.get("source_commit") == A8_SOURCE_COMMIT
        and runtime_package.get("source_archive_sha256")
        == A8_SOURCE_ARCHIVE_SHA256
        and runtime_package.get("file_count")
        == expected_runtime_package["file_count"]
        and runtime_package.get("files_sha256")
        == expected_runtime_package["files_sha256"]
        and runtime_package.get("source")
        == f"bind:git:{A8_SOURCE_COMMIT}:kairyu"
        and isinstance(runtime_package.get("attested_container_id"), str)
        and HEX64_RE.fullmatch(
            str(runtime_package["attested_container_id"])
        )
        is not None,
        "A9 mounted runtime package differs from A8 source",
    )
    _require(
        provenance.get("a8_source_commit") == A8_SOURCE_COMMIT,
        "A9 does not identify the pinned A8 source",
    )
    _require(
        provenance.get("image_id") == A8_IMAGE_ID,
        "A9 image differs from the A8 image",
    )
    _require(
        provenance.get("model") == MODEL
        and provenance.get("model_revision") == MODEL_REVISION,
        "A9 model identity differs from A8",
    )
    _require(
        provenance.get("gpu_count") == 8
        and provenance.get("tp8_gpu_ordinals") == list(range(8)),
        "A9 does not prove one TP8 engine on GPUs 0-7",
    )
    gpus = provenance.get("gpus")
    _require(
        isinstance(gpus, list)
        and len(gpus) == 8
        and [row.get("index") for row in gpus if isinstance(row, dict)]
        == list(range(8)),
        "A9 GPU inventory is incomplete",
    )
    topology = provenance.get("nvidia_smi_topology")
    _require(
        isinstance(topology, list)
        and all(
            any(f"GPU{index}" in str(line) for line in topology)
            for index in range(8)
        ),
        "A9 physical topology does not enumerate all GPUs",
    )
    _validate_container_provenance(provenance.get("containers"))
    _validate_backends_payload(
        provenance.get("engine_backends"),
        role="engine-host",
    )
    _validate_backends_payload(
        provenance.get("gateway_backends"),
        role="gateway",
    )
    _require(
        provenance.get("engine_url") == "http://127.0.0.1:8300/v1"
        and provenance.get("gateway_url") == "http://127.0.0.1:8301/v1",
        "A9 endpoint identity differs from the fixed stack",
    )
    _exact_int(
        provenance.get("placement_log_start_offset"),
        "placement_log_start_offset",
    )
    checkpoint = provenance.get("checkpoint")
    _require(isinstance(checkpoint, dict), "A9 checkpoint evidence is missing")
    a8_provenance = a8_manifest.get("provenance")
    _require(
        isinstance(a8_provenance, dict),
        "A8 provenance disappeared during A9 replay",
    )
    a8_checkpoint = a8_provenance.get("checkpoint")
    a8_gpus = a8_provenance.get("gpus")
    _require(
        isinstance(a8_gpus, list)
        and len(a8_gpus) == len(gpus)
        and [
            (
                row.get("index"),
                row.get("uuid"),
                row.get("pci_bus_id"),
                row.get("name"),
                row.get("memory_total_mib"),
            )
            for row in gpus
            if isinstance(row, dict)
        ]
        == [
            (
                row.get("index"),
                row.get("uuid"),
                row.get("pci_bus_id"),
                row.get("name"),
                row.get("memory_total_mib"),
            )
            for row in a8_gpus
            if isinstance(row, dict)
        ]
        and topology == a8_provenance.get("nvidia_smi_topology"),
        "A9 physical host identity differs from A8",
    )
    containers = provenance["containers"]
    assert isinstance(containers, dict)
    services = containers["services"]
    assert isinstance(services, dict)
    tp8_container = services["tp8"]
    assert isinstance(tp8_container, dict)
    _require(
        runtime_package
        == {
            **expected_runtime_package,
            "attested_container_id": tp8_container.get("container_id"),
            "source": f"bind:git:{A8_SOURCE_COMMIT}:kairyu",
        },
        "A9 runtime package descriptor differs from the pinned source",
    )
    _validate_checkpoint_descriptor(
        checkpoint,
        a8_checkpoint=a8_checkpoint,
        container_id=tp8_container.get("container_id"),
    )
    config_hashes = provenance.get("config_file_sha256")
    _require(
        isinstance(config_hashes, dict)
        and set(config_hashes)
        == {path.as_posix() for path in CONFIG_PATHS}
        and config_hashes
        == _committed_config_file_hashes(str(source_commit)),
        "A9 config files differ from their clean source commit",
    )
    return provenance


def recompute_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_sha256: str,
    a8_artifact_dir: Path = A8_ARTIFACT_DIR,
) -> dict[str, object]:
    """Validate raw TP8 evidence and derive the complete A9 report."""

    _require(_hash_valid(raw_sha256), "A9 raw SHA-256 is invalid")
    _require(len(rows) >= 3, "A9 raw evidence is incomplete")
    start = rows[0]
    end = rows[-1]
    _require(start.get("type") == "run", "A9 raw must start with run")
    _require(end.get("type") == "run_end", "A9 raw must end with run_end")
    _require(start.get("schema_version") == SCHEMA_VERSION, "A9 schema mismatch")
    _require(start.get("gate") == GATE, "A9 gate mismatch")
    _validate_formal_config(start.get("config"))
    started_monotonic_ns = _exact_int(
        start.get("started_monotonic_ns"),
        "run started_monotonic_ns",
        minimum=1,
    )
    ended_monotonic_ns = _exact_int(
        end.get("ended_monotonic_ns"),
        "run ended_monotonic_ns",
        minimum=started_monotonic_ns,
    )
    _exact_int(
        start.get("started_unix_ns"),
        "run started_unix_ns",
        minimum=1,
    )
    _exact_int(
        end.get("ended_unix_ns"),
        "run ended_unix_ns",
        minimum=1,
    )
    a8_rows, a8_manifest, a8_descriptor = load_a8_dp_evidence(
        a8_artifact_dir
    )
    provenance = _validate_provenance(
        start.get("provenance"),
        a8_manifest=a8_manifest,
        a8_descriptor=a8_descriptor,
    )
    trace = start.get("trace")
    _require(isinstance(trace, dict), "A9 trace evidence is missing")
    _require(
        trace.get("bundle_sha256") == a8.TRACE_BUNDLE_SHA256
        and trace.get("bundle_schema_version") == a8.TRACE_SCHEMA
        and trace.get("scaling_selection_sha256")
        == a8.SCALING_SELECTION_SHA256,
        "A9 trace differs from A8",
    )
    scaling_trace = trace.get("scaling")
    _require(
        isinstance(scaling_trace, list)
        and len(scaling_trace) == REQUESTS_PER_CELL,
        "A9 scaling trace descriptor count mismatch",
    )
    trace_scaling = {
        _exact_int(row.get("sequence"), "trace sequence"): row
        for row in scaling_trace
        if isinstance(row, dict)
    }
    _require(
        set(trace_scaling) == set(range(REQUESTS_PER_CELL)),
        "A9 trace sequence matrix mismatch",
    )
    _require(
        sha256_bytes(canonical_json(scaling_trace).encode())
        == a8.SCALING_SELECTION_SHA256,
        "A9 trace descriptor rows differ from their digest",
    )

    requests = [
        row for row in rows[1:-1] if row.get("type") == "request"
    ]
    placements = [
        row for row in rows[1:-1] if row.get("type") == "placement"
    ]
    _require(
        len(requests) + len(placements) == len(rows) - 2,
        "A9 raw contains an unknown row type",
    )
    for row in requests:
        _validate_tp8_request(row, trace_scaling)
    _require(
        all(
            all(
                started_monotonic_ns <= int(row[field]) <= ended_monotonic_ns
                for field in (
                    "scheduled_ns",
                    "started_ns",
                    "first_token_ns",
                    "ended_ns",
                )
            )
            for row in requests
        ),
        "A9 TP8 request timestamps escape the run bounds",
    )
    expected_request_count = REPEATS * (
        WARMUP_REQUESTS + len(RATES_RPS) * REQUESTS_PER_CELL
    )
    _require(
        len(requests) == expected_request_count,
        "A9 TP8 request matrix is incomplete",
    )
    expected_order: list[tuple[object, ...]] = []
    for repeat in range(REPEATS):
        expected_order.extend(
            ("warmup", repeat, sequence)
            for sequence in range(WARMUP_REQUESTS)
        )
        for rate in RATES_RPS:
            expected_order.extend(
                ("scaling", repeat, rate, sequence)
                for sequence in range(REQUESTS_PER_CELL)
            )
    actual_order = [
        (
            (
                "scaling",
                row.get("repeat"),
                row.get("rate_rps"),
                row.get("sequence"),
            )
            if row.get("phase") == "scaling"
            else (
                "warmup",
                row.get("repeat"),
                row.get("sequence"),
            )
        )
        for row in requests
    ]
    _require(
        actual_order == expected_order,
        "A9 TP8 request order differs from the fixed execution plan",
    )

    placement_by_id: dict[str, Mapping[str, object]] = {}
    request_by_id = {
        str(row["response_request_id"]): row for row in requests
    }
    _require(
        len(request_by_id) == len(requests),
        "A9 TP8 response IDs are not unique",
    )
    generations: set[str] = set()
    for row in placements:
        response_id = row.get("response_request_id")
        _require(
            isinstance(response_id, str)
            and response_id in request_by_id
            and response_id not in placement_by_id,
            "A9 placement does not join exactly one TP8 response",
        )
        request = request_by_id[response_id]
        started_ns = _exact_int(
            row.get("placement_started_ns"),
            "placement_started_ns",
            minimum=int(request["started_ns"]),
        )
        selected_ns = _exact_int(
            row.get("selected_at_ns"),
            "selected_at_ns",
            minimum=started_ns,
        )
        latency_ns = _exact_int(
            row.get("placement_latency_ns"),
            "placement_latency_ns",
        )
        _require(
            selected_ns - started_ns == latency_ns
            and selected_ns <= int(request["first_token_ns"]),
            "A9 placement timing is invalid",
        )
        _require(
            row.get("pool_size") == 1
            and row.get("eligible_size") == 1
            and row.get("replica_id") == "0",
            "A9 gateway did not place on its sole TP8 replica",
        )
        generation = row.get("replica_generation")
        _require(
            isinstance(generation, str)
            and re.fullmatch(r"[0-9a-f]{32}", generation) is not None,
            "A9 TP8 replica generation is invalid",
        )
        generations.add(generation)
        _require(
            row.get("phase") == request.get("phase")
            and row.get("repeat") == request.get("repeat")
            and row.get("rate_rps") == request.get("rate_rps")
            and row.get("sequence") == request.get("sequence"),
            "A9 placement/request identity mismatch",
        )
        placement_by_id[response_id] = row
    _require(
        set(placement_by_id) == set(request_by_id),
        "every A9 response must have one placement row",
    )
    _require(
        len(generations) == 1,
        "A9 TP8 replica generation changed during measurement",
    )

    by_cell: dict[
        tuple[str, int, int], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in a8_rows:
        by_cell[("dp2-tp4", int(row["repeat"]), int(row["rate_rps"]))].append(
            row
        )
    warmup_by_repeat: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in requests:
        if row.get("phase") == "scaling":
            by_cell[
                ("tp8", int(row["repeat"]), int(row["rate_rps"]))
            ].append(row)
        else:
            warmup_by_repeat[int(row["repeat"])].append(row)
    expected_cells = {
        (arm, repeat, rate)
        for arm in ("dp2-tp4", "tp8")
        for repeat in range(REPEATS)
        for rate in RATES_RPS
    }
    _require(
        set(by_cell) == expected_cells,
        "A9 DP/TP8 cell matrix is incomplete",
    )
    _require(
        set(warmup_by_repeat) == set(range(REPEATS))
        and all(
            len(samples) == WARMUP_REQUESTS
            for samples in warmup_by_repeat.values()
        ),
        "A9 TP8 warmup matrix is incomplete",
    )
    for repeat, samples in warmup_by_repeat.items():
        _require(
            {int(row["sequence"]) for row in samples}
            == set(range(WARMUP_REQUESTS)),
            f"A9 repeat {repeat} warmup sequence mismatch",
        )
    execution_blocks: list[Sequence[Mapping[str, object]]] = []
    tp8_scaling_by_cell: dict[
        tuple[int, int], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in requests:
        if row.get("phase") == "scaling":
            tp8_scaling_by_cell[
                (int(row["repeat"]), int(row["rate_rps"]))
            ].append(row)
    for repeat in range(REPEATS):
        warmup = warmup_by_repeat[repeat]
        _require(
            all(
                int(warmup[index]["scheduled_ns"])
                >= int(warmup[index - 1]["ended_ns"])
                for index in range(1, len(warmup))
            ),
            f"A9 repeat {repeat} warmup was not serialized",
        )
        execution_blocks.append(warmup)
        execution_blocks.extend(
            tp8_scaling_by_cell[(repeat, rate)] for rate in RATES_RPS
        )
    prior_end_ns = 0
    for block in execution_blocks:
        block_start_ns = min(int(row["scheduled_ns"]) for row in block)
        block_end_ns = max(int(row["ended_ns"]) for row in block)
        _require(
            block_start_ns >= prior_end_ns,
            "A9 TP8 execution blocks overlap or were reordered",
        )
        prior_end_ns = block_end_ns

    cell_metrics: dict[str, dict[str, object]] = {}
    for arm, repeat, rate in sorted(expected_cells):
        key = f"r{repeat}-{arm}-{rate}rps"
        cell_metrics[key] = _cell_metrics(
            by_cell[(arm, repeat, rate)],
            arm=arm,
            repeat=repeat,
            rate_rps=rate,
        )
    rate_metrics: list[dict[str, object]] = []
    for rate in RATES_RPS:
        by_arm: dict[str, list[dict[str, object]]] = {
            arm: [
                cell_metrics[f"r{repeat}-{arm}-{rate}rps"]
                for repeat in range(REPEATS)
            ]
            for arm in ("dp2-tp4", "tp8")
        }
        dp_goodput = [
            float(row["goodput_rps"]) for row in by_arm["dp2-tp4"]
        ]
        tp8_goodput = [
            float(row["goodput_rps"]) for row in by_arm["tp8"]
        ]
        dp_median = float(statistics.median(dp_goodput))
        tp8_median = float(statistics.median(tp8_goodput))
        rate_metrics.append(
            {
                "rate_rps": rate,
                "dp_median_goodput_rps": dp_median,
                "tp8_median_goodput_rps": tp8_median,
                "dp_over_tp8_goodput_ratio": (
                    dp_median / tp8_median if tp8_median > 0 else None
                ),
                "dp_goodput_mad_rps": _median_absolute_deviation(
                    dp_goodput
                ),
                "tp8_goodput_mad_rps": _median_absolute_deviation(
                    tp8_goodput
                ),
                "dp_median_max_in_flight": float(
                    statistics.median(
                        int(row["observed_max_in_flight"])
                        for row in by_arm["dp2-tp4"]
                    )
                ),
                "tp8_median_max_in_flight": float(
                    statistics.median(
                        int(row["observed_max_in_flight"])
                        for row in by_arm["tp8"]
                    )
                ),
                "dp_tpot_p50_ns": float(
                    statistics.median(
                        float(row["tpot_p50_ns"])
                        for row in by_arm["dp2-tp4"]
                    )
                ),
                "tp8_tpot_p50_ns": float(
                    statistics.median(
                        float(row["tpot_p50_ns"])
                        for row in by_arm["tp8"]
                    )
                ),
                "dp_tpot_p99_ns": float(
                    statistics.median(
                        float(row["tpot_p99_ns"])
                        for row in by_arm["dp2-tp4"]
                    )
                ),
                "tp8_tpot_p99_ns": float(
                    statistics.median(
                        float(row["tpot_p99_ns"])
                        for row in by_arm["tp8"]
                    )
                ),
                "dp_ttft_p50_ns": float(
                    statistics.median(
                        int(row["ttft_p50_ns"])
                        for row in by_arm["dp2-tp4"]
                    )
                ),
                "tp8_ttft_p50_ns": float(
                    statistics.median(
                        int(row["ttft_p50_ns"])
                        for row in by_arm["tp8"]
                    )
                ),
                "dp_ttft_p99_ns": float(
                    statistics.median(
                        int(row["ttft_p99_ns"])
                        for row in by_arm["dp2-tp4"]
                    )
                ),
                "tp8_ttft_p99_ns": float(
                    statistics.median(
                        int(row["ttft_p99_ns"])
                        for row in by_arm["tp8"]
                    )
                ),
            }
        )
    crossover = derive_crossover(rate_metrics)

    _require(end.get("completed") is True, "A9 run did not complete")
    _require(
        end.get("source_commit") == provenance.get("source_commit")
        and _physical_gpu_identity(end.get("gpu_inventory"))
        == _physical_gpu_identity(provenance.get("gpus"))
        and end.get("containers") == provenance.get("containers")
        and end.get("checkpoint") == provenance.get("checkpoint")
        and end.get("runtime_package") == provenance.get("runtime_package")
        and end.get("engine_backends")
        == provenance.get("engine_backends")
        and end.get("gateway_backends")
        == provenance.get("gateway_backends")
        and end.get("request_count") == len(requests)
        and end.get("placement_count") == len(placements),
        "A9 runtime identity changed during measurement",
    )
    checks = {
        "a8_raw_manifest_and_placements_match_fixed_sha256": True,
        "a8_only_failure_is_the_accepted_1_9x_deviation": True,
        "runtime_package_tree_matches_a8_source": True,
        "runtime_package_unchanged_after_measurement": True,
        "image_checkpoint_model_and_trace_match_a8": True,
        "checkpoint_unchanged_after_measurement": True,
        "tp8_uses_all_eight_gpus_behind_one_replica_gateway": True,
        "fixed_open_loop_grid_three_repeats_exact": True,
        "all_tp8_requests_retry_free_exact_success": True,
        "all_tp8_request_timestamps_inside_run_interval": True,
        "every_tp8_response_has_one_exact_placement": True,
        "goodput_ttft_and_versioned_terminal_tpot_recomputed": True,
        "crossover_or_no_crossover_recorded_without_interpolation": True,
        "runtime_identity_stable_for_full_measurement": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "passed": all(checks.values()),
        "checks": checks,
        "report_only": True,
        "tpot": {
            "method": TPOT_METHOD,
            "definition": TPOT_DEFINITION,
        },
        "cell_metrics": cell_metrics,
        "rate_metrics": rate_metrics,
        "crossover": crossover,
        "provenance": provenance,
        "a8_accepted_deviation": {
            "artifact_passed": False,
            "sole_false_check": A8_ACCEPTED_FALSE_CHECK,
            "not_an_a9_threshold": True,
        },
        "raw": {
            "tp8_sha256": raw_sha256,
            "tp8_request_count": len(requests),
            "tp8_placement_count": len(placements),
            "a8_dp_scaling_request_count": len(a8_rows),
        },
    }


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    normalized = [
        {**row, "row_index": index} for index, row in enumerate(rows)
    ]
    path.write_text(
        "".join(
            f"{canonical_json(row)}\n" for row in normalized
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            _require(
                raw.endswith(b"\n") and raw.strip(),
                f"invalid A9 JSONL framing at line {line_number}",
            )
            value = json.loads(raw)
            _require(
                isinstance(value, dict),
                f"A9 JSONL row {line_number} is not an object",
            )
            rows.append(value)
    _require(rows, "A9 raw evidence is empty")
    _require(
        all(
            row.get("row_index") == index
            for index, row in enumerate(rows)
        ),
        "A9 raw row indices are not contiguous",
    )
    return rows


def seal_artifact(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    a8_artifact_dir: Path = A8_ARTIFACT_DIR,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    manifest_path = output_dir / MANIFEST_NAME
    _write_jsonl(raw_path, rows)
    stored = _read_jsonl(raw_path)
    manifest = recompute_manifest(
        stored,
        raw_sha256=sha256_file(raw_path),
        a8_artifact_dir=a8_artifact_dir,
    )
    manifest_path.write_text(
        f"{canonical_json(manifest)}\n",
        encoding="utf-8",
    )
    return manifest


def verify_artifact(
    artifact_dir: Path,
    *,
    a8_artifact_dir: Path = A8_ARTIFACT_DIR,
    assert_gate: bool = False,
) -> dict[str, object]:
    raw_path = artifact_dir / RAW_NAME
    manifest_path = artifact_dir / MANIFEST_NAME
    _require(raw_path.is_file(), f"missing A9 raw artifact: {raw_path}")
    _require(
        manifest_path.is_file(),
        f"missing A9 manifest artifact: {manifest_path}",
    )
    rows = _read_jsonl(raw_path)
    recomputed = recompute_manifest(
        rows,
        raw_sha256=sha256_file(raw_path),
        a8_artifact_dir=a8_artifact_dir,
    )
    retained = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        isinstance(retained, dict),
        "retained A9 manifest must be an object",
    )
    _require(
        retained == recomputed,
        "retained A9 manifest differs from raw replay",
    )
    if assert_gate and not recomputed["passed"]:
        failed = [
            name
            for name, passed in recomputed["checks"].items()
            if not passed
        ]
        raise GateEvidenceError(
            f"G2 A9 evidence failed: {', '.join(failed)}"
        )
    return recomputed


def replay_artifact(
    artifact_dir: Path,
    *,
    a8_artifact_dir: Path = A8_ARTIFACT_DIR,
    assert_gate: bool = False,
) -> dict[str, object]:
    """Recompute A9 directly from raw TP8 and pinned raw A8 evidence."""

    raw_path = artifact_dir / RAW_NAME
    _require(raw_path.is_file(), f"missing A9 raw artifact: {raw_path}")
    recomputed = recompute_manifest(
        _read_jsonl(raw_path),
        raw_sha256=sha256_file(raw_path),
        a8_artifact_dir=a8_artifact_dir,
    )
    if assert_gate and not recomputed["passed"]:
        failed = [
            name
            for name, passed in recomputed["checks"].items()
            if not passed
        ]
        raise GateEvidenceError(
            f"G2 A9 raw replay failed: {', '.join(failed)}"
        )
    return recomputed


def _resolve_measurement_paths(
    *,
    gateway_placement_log: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    resolved_output_dir = output_dir.resolve()
    expected_placement_log = resolved_output_dir / "placements.jsonl"
    resolved_placement_log = gateway_placement_log.resolve()
    if resolved_placement_log != expected_placement_log:
        raise ValueError(
            "--gateway-placement-log must resolve to "
            "--output-dir/placements.jsonl"
        )
    return resolved_output_dir, resolved_placement_log


async def measure(
    *,
    engine_url: str,
    gateway_url: str,
    trace_bundle: Path,
    gateway_placement_log: Path,
    output_dir: Path,
    a8_artifact_dir: Path,
    image_id: str,
    expected_commit: str | None,
    timeout_s: float,
    placement_timeout_s: float,
    assert_gate: bool,
) -> dict[str, object]:
    if IMAGE_ID_RE.fullmatch(image_id) is None:
        raise ValueError("--image-id must be sha256:<64 lowercase hex>")
    if image_id != A8_IMAGE_ID:
        raise ValueError("--image-id must be the immutable A8 image")
    if timeout_s <= 0 or placement_timeout_s <= 0:
        raise ValueError("timeouts must be positive")
    engine_api = _normalize_api_root(engine_url)
    gateway_api = _normalize_api_root(gateway_url)
    if engine_api != "http://127.0.0.1:8300/v1":
        raise ValueError("A9 engine URL must be http://127.0.0.1:8300/v1")
    if gateway_api != "http://127.0.0.1:8301/v1":
        raise ValueError(
            "A9 gateway URL must be http://127.0.0.1:8301/v1"
        )
    resolved_output_dir, resolved_placement_log = _resolve_measurement_paths(
        gateway_placement_log=gateway_placement_log,
        output_dir=output_dir,
    )
    trace = a8.load_trace_bundle(trace_bundle)
    _a8_rows, a8_manifest, a8_descriptor = load_a8_dp_evidence(
        a8_artifact_dir
    )
    commit, tracked_clean = a8._git_source(expected_commit)
    if not tracked_clean:
        raise RuntimeError(
            "formal measurement requires a clean tracked/index source tree"
        )
    config_hashes = _config_file_hashes()
    if config_hashes != _committed_config_file_hashes(commit):
        raise RuntimeError(
            "A9 formal config files differ from the clean source commit"
        )
    a8_helper = _a8_helper_reference(commit)
    gpu_inventory = a8._gpu_inventory()
    topology = _run_command(("nvidia-smi", "topo", "-m")).splitlines()
    if not topology or not all(
        any(f"GPU{index}" in line for line in topology)
        for index in range(8)
    ):
        raise RuntimeError(
            "nvidia-smi topology does not enumerate all eight GPUs"
        )
    containers = _docker_stack_provenance(
        image_id,
        evidence_dir=resolved_output_dir,
    )
    services = containers.get("services")
    tp8_service = services.get("tp8") if isinstance(services, dict) else None
    container_id = (
        tp8_service.get("container_id")
        if isinstance(tp8_service, dict)
        else None
    )
    if (
        not isinstance(container_id, str)
        or HEX64_RE.fullmatch(container_id) is None
    ):
        raise RuntimeError("A9 TP8 container identity is missing")
    runtime_package = _runtime_package_reference(container_id)
    checkpoint = a8._checkpoint_reference(container_id)
    a8_provenance = a8_manifest.get("provenance")
    if not isinstance(a8_provenance, dict):
        raise RuntimeError("A8 provenance is incomplete")
    _validate_checkpoint_descriptor(
        checkpoint,
        a8_checkpoint=a8_provenance.get("checkpoint"),
        container_id=container_id,
    )
    placement_offset = a8._placement_file_offset(
        resolved_placement_log
    )
    limits = httpx.Limits(
        max_connections=256,
        max_keepalive_connections=128,
    )
    async with (
        httpx.AsyncClient(timeout=timeout_s, limits=limits) as engine_client,
        httpx.AsyncClient(timeout=timeout_s, limits=limits) as gateway_client,
    ):
        engine_backends = await _endpoint_metadata(
            engine_client,
            engine_api,
            expected_role="engine-host",
        )
        gateway_backends = await _endpoint_metadata(
            gateway_client,
            gateway_api,
            expected_role="gateway",
        )
        provenance = {
            "source_commit": commit,
            "tracked_tree_clean": tracked_clean,
            "a8_source_commit": A8_SOURCE_COMMIT,
            "a8_helper": a8_helper,
            "runtime_package": runtime_package,
            "a8_artifact": a8_descriptor,
            "image_id": image_id,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "gpu_count": len(gpu_inventory),
            "gpus": gpu_inventory,
            "nvidia_smi_topology": topology,
            "tp8_gpu_ordinals": list(range(8)),
            "containers": containers,
            "checkpoint": checkpoint,
            "engine_url": engine_api,
            "gateway_url": gateway_api,
            "engine_backends": engine_backends,
            "gateway_backends": gateway_backends,
            "config_file_sha256": config_hashes,
            "placement_log_start_offset": placement_offset,
        }
        rows: list[dict[str, object]] = [
            {
                "type": "run",
                "schema_version": SCHEMA_VERSION,
                "gate": GATE,
                "config": formal_config(),
                "trace": a8.trace_evidence(trace),
                "provenance": provenance,
                "started_unix_ns": time.time_ns(),
                "started_monotonic_ns": time.perf_counter_ns(),
            }
        ]
        scaling_trace = trace["scaling"]
        assert isinstance(scaling_trace, list)
        for repeat in range(REPEATS):
            rows.extend(
                _remap_tp8_row(row)
                for row in await a8._measure_warmup(
                    gateway_client,
                    api_root=gateway_api,
                    arm="dp",
                    repeat=repeat,
                    trace_rows=scaling_trace,
                )
            )
            for rate_rps in RATES_RPS:
                rows.extend(
                    _remap_tp8_row(row)
                    for row in await a8._measure_scaling_cell(
                        gateway_client,
                        api_root=gateway_api,
                        arm="dp",
                        repeat=repeat,
                        rate_rps=rate_rps,
                        trace_rows=scaling_trace,
                    )
                )
        final_engine_backends = await _endpoint_metadata(
            engine_client,
            engine_api,
            expected_role="engine-host",
        )
        final_gateway_backends = await _endpoint_metadata(
            gateway_client,
            gateway_api,
            expected_role="gateway",
        )
    requests = [
        row for row in rows if row.get("type") == "request"
    ]
    placements = await a8._collect_placements(
        resolved_placement_log,
        offset=placement_offset,
        dp_requests=requests,
        timeout_s=placement_timeout_s,
    )
    rows.extend(placements)
    final_gpu_inventory = a8._gpu_inventory()
    final_containers = _docker_stack_provenance(
        image_id,
        evidence_dir=resolved_output_dir,
    )
    final_checkpoint = a8._checkpoint_reference(container_id)
    if final_checkpoint != checkpoint:
        raise RuntimeError("live A9 checkpoint changed during measurement")
    final_runtime_package = _runtime_package_reference(container_id)
    if final_runtime_package != runtime_package:
        raise RuntimeError("mounted A8 runtime changed during measurement")
    final_commit, final_clean = a8._git_source(commit)
    if not final_clean:
        raise RuntimeError("tracked source changed during A9 measurement")
    rows.append(
        {
            "type": "run_end",
            "source_commit": final_commit,
            "completed": True,
            "request_count": len(requests),
            "placement_count": len(placements),
            "gpu_inventory": final_gpu_inventory,
            "containers": final_containers,
            "checkpoint": final_checkpoint,
            "runtime_package": final_runtime_package,
            "engine_backends": final_engine_backends,
            "gateway_backends": final_gateway_backends,
            "ended_unix_ns": time.time_ns(),
            "ended_monotonic_ns": time.perf_counter_ns(),
        }
    )
    manifest = seal_artifact(
        output_dir,
        rows,
        a8_artifact_dir=a8_artifact_dir,
    )
    if assert_gate and not manifest["passed"]:
        failed = [
            name
            for name, passed in manifest["checks"].items()
            if not passed
        ]
        raise GateEvidenceError(
            f"G2 A9 evidence failed after retention: {', '.join(failed)}"
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser(
        "measure",
        help="run only the missing real Qwen3-32B TP8 arm",
    )
    measure_parser.add_argument("--engine-url", required=True)
    measure_parser.add_argument("--gateway-url", required=True)
    measure_parser.add_argument("--trace-bundle", type=Path, required=True)
    measure_parser.add_argument(
        "--gateway-placement-log",
        type=Path,
        required=True,
    )
    measure_parser.add_argument("--output-dir", type=Path, required=True)
    measure_parser.add_argument(
        "--a8-artifact",
        type=Path,
        default=A8_ARTIFACT_DIR,
    )
    measure_parser.add_argument("--image-id", required=True)
    measure_parser.add_argument("--expected-commit")
    measure_parser.add_argument("--timeout-s", type=float, default=120.0)
    measure_parser.add_argument(
        "--placement-timeout-s",
        type=float,
        default=30.0,
    )
    measure_parser.add_argument("--assert-gate", action="store_true")
    for command in ("verify", "replay"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--artifact", type=Path, required=True)
        command_parser.add_argument(
            "--a8-artifact",
            type=Path,
            default=A8_ARTIFACT_DIR,
        )
        command_parser.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "measure":
            result = asyncio.run(
                measure(
                    engine_url=args.engine_url,
                    gateway_url=args.gateway_url,
                    trace_bundle=args.trace_bundle,
                    gateway_placement_log=args.gateway_placement_log,
                    output_dir=args.output_dir,
                    a8_artifact_dir=args.a8_artifact,
                    image_id=args.image_id,
                    expected_commit=args.expected_commit,
                    timeout_s=args.timeout_s,
                    placement_timeout_s=args.placement_timeout_s,
                    assert_gate=args.assert_gate,
                )
            )
        elif args.command == "verify":
            result = verify_artifact(
                args.artifact,
                a8_artifact_dir=args.a8_artifact,
                assert_gate=args.assert_gate,
            )
        else:
            result = replay_artifact(
                args.artifact,
                a8_artifact_dir=args.a8_artifact,
                assert_gate=args.assert_gate,
            )
    except (
        GateEvidenceError,
        a8.GateEvidenceError,
        RuntimeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        parser = _build_parser()
        parser.error(str(error))
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
