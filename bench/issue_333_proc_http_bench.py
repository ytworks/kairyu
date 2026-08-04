#!/usr/bin/env python3
"""Evidence-grade HTTP diagnostic for issue #333.

This operator compares the current in-process ``kairyu`` engine with the
process-isolated ``kairyu-proc`` engine.  It deliberately reuses the G2 A6
ShareGPT trace, strict streaming request executor, warm-up geometry, and metric
definitions, but it is *not* the formal G2 A6 gate and never emits an A6
pass/fail claim.

The binding schedule is four fresh TP4 servers in ABBA order::

    kairyu, kairyu-proc, kairyu-proc, kairyu

Each server receives four serialized warm-ups, the exact 31-request
B=1/2/4/8/16 CUDA-graph warm-up, and one synchronized 128-request ShareGPT
burst.  Every request row, launch/config/source/checkpoint attestation, and
three ``GET /backends`` observations is retained.  ``assemble`` and ``verify``
replay evidence integrity and diagnostic metrics without a performance gate.

Live execution requires a clean committed checkout and writes only below
``--work-dir`` (which must resolve below ``/tmp``).  It never mutates the source
checkout.  A partial run is intentionally not resumable: use a fresh work
directory so ABBA chronology cannot be silently reconstructed from old cells.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))
# A live run is required to start from a fresh detached checkout.  Prevent the
# operator imports below from populating that checkout with ignored bytecode.
sys.dont_write_bytecode = True

import httpx  # noqa: E402
import yaml  # noqa: E402

from bench import g2_a6_vllm_bench as a6  # noqa: E402
from bench import run_g2_a6_formal as formal  # noqa: E402

SCHEMA_VERSION = "kairyu.issue-333.proc-http-diagnostic.v2"
RUNNER_SCHEMA = "kairyu.issue-333.proc-http-runner.v1"
RAW_NAME = "issue-333-proc-http-raw.jsonl"
MANIFEST_NAME = "issue-333-proc-http-manifest.json"
STATE_NAME = "issue-333-proc-http-state.json"
DIAGNOSTIC = "issue-333-kairyu-proc-http"
TP_SIZE = 4
WORKLOAD = "sharegpt"
ARMS = ("kairyu", "kairyu-proc")
ABBA_ORDER = ("kairyu", "kairyu-proc", "kairyu-proc", "kairyu")
PAIR_COUNT = 2
MATERIAL_TTFT_P99_RATIO_CEILING = Fraction(9, 10)
PYCACHE_PREFIX = "/tmp/kairyu-issue333-empty-pycache"
CLEANUP_QUIESCENCE_TIMEOUT_S = 60.0
CLEANUP_QUIESCENCE_POLL_S = 0.5
QUIESCENCE_STABLE_SAMPLES = 2
CONTAINER_STOP_TIMEOUT_S = 120
DOCKER_STOP_COMMAND_TIMEOUT_S = 135.0
DOCKER_CONTROL_TIMEOUT_S = 30.0
DOCKER_LOG_TIMEOUT_S = 30.0
DOCKER_LAUNCH_TIMEOUT_S = 60.0
DOCKER_ATTEST_TIMEOUT_S = 60.0
NVML_QUERY_TIMEOUT_S = 5.0

DEFAULT_REPO = _REPO_ROOT
DEFAULT_WORK_DIR = Path("/tmp/issue-333-proc-http")
DEFAULT_TRACE_BUNDLE = Path("/tmp/a6-traces/g2-a6-traces.json")
DEFAULT_DATASET = Path("/tmp/ShareGPT_V3_unfiltered_cleaned_split.json")
DEFAULT_TOKENIZER = Path("/tmp/kairyu-a7-qwen3-32b-tokenizer.json")
DEFAULT_ENV_ARTIFACT = DEFAULT_REPO / "bench/results/env-2026-07-30.json"
DEFAULT_KAIRYU_TEMPLATE = DEFAULT_REPO / "examples/qwen3-32b-multi-gpu/g2-a6-kairyu.template.yaml"
DEFAULT_IMAGE = "kairyu-qwen3-32b-kairyu:latest"
DEFAULT_MODEL_VOLUME = "kairyu-qwen3-32b_qwen3-32b"
DEFAULT_PORT = 18080
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

CRITICAL_SOURCE_FILES = (
    "kairyu/entrypoints/server/app.py",
    "kairyu/entrypoints/server/health.py",
    "kairyu/engine/kairyu_backend.py",
    "kairyu/engine/zmq_backend.py",
    "kairyu/engine/core/engine_service.py",
    "kairyu/engine/core/worker.py",
    "kairyu/engine/engine_loop.py",
    "kairyu/engine/config_validation.py",
    "kairyu/engine/registry.py",
)
RUNNER_SOURCE_FILES = (
    "bench/issue_333_proc_http_bench.py",
    "bench/g2_a6_vllm_bench.py",
    "bench/run_g2_a6_formal.py",
)


class DiagnosticRunError(RuntimeError):
    """Fail-closed operator or evidence error."""


@dataclass(frozen=True)
class Cell:
    order_index: int
    repeat: int
    arm: str

    @property
    def scenario_id(self) -> str:
        return f"issue333-tp{TP_SIZE}-{WORKLOAD}-r{self.repeat}-{self.arm}"

    @property
    def key(self) -> str:
        return f"{self.order_index:02d}-{self.scenario_id}"

    def identity(self) -> dict[str, object]:
        return {
            "order_index": self.order_index,
            "repeat": self.repeat,
            "arm": self.arm,
            "tensor_parallel_size": TP_SIZE,
            "workload": WORKLOAD,
            "scenario_id": self.scenario_id,
        }


def diagnostic_plan() -> tuple[Cell, ...]:
    """Return the exact two-pair ABBA fresh-server plan."""

    result = (
        Cell(order_index=0, repeat=0, arm="kairyu"),
        Cell(order_index=1, repeat=0, arm="kairyu-proc"),
        Cell(order_index=2, repeat=1, arm="kairyu-proc"),
        Cell(order_index=3, repeat=1, arm="kairyu"),
    )
    if tuple(item.arm for item in result) != ABBA_ORDER:
        raise DiagnosticRunError("issue #333 plan is not exact ABBA")
    if len({item.scenario_id for item in result}) != len(result):
        raise DiagnosticRunError("issue #333 plan contains duplicate scenarios")
    return result


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


def hashed_descriptor(value: object) -> dict[str, object]:
    return {"value": value, "sha256": sha256_bytes(canonical_json(value).encode())}


def descriptor_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"value", "sha256"}
        and isinstance(value.get("sha256"), str)
        and HEX64.fullmatch(str(value["sha256"])) is not None
        and value["sha256"] == sha256_bytes(canonical_json(value.get("value")).encode())
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    rendered = "".join(
        canonical_json({**row, "row_index": index}) + "\n" for index, row in enumerate(rows)
    )
    atomic_write_text(path, rendered)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSON row at {path}:{line_number}")
        result.append(row)
    if any(row.get("row_index") != index for index, row in enumerate(result)):
        raise ValueError(f"{path} row indices are not contiguous")
    return result


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


_A6_CONTRACT_PIN: dict[str, object] = {
    "serial_warmup_requests": 4,
    "graph_warmup_batch_sizes": [1, 2, 4, 8, 16],
    "graph_warmup_requests": 31,
    "measurement_requests": 128,
    "concurrency": 128,
    "completion_tokens": 128,
    "ttft_slo_ns": 10_000_000_000,
    "pipeline_depth": 5,
    "allocated_cache_blocks": 8193,
    "usable_cache_blocks": 8192,
    "max_num_batched_tokens": 1024,
    "max_num_seqs": 16,
    "max_model_len": 8192,
}


def observed_a6_contract() -> dict[str, object]:
    """Read each borrowed contract value once from the current A6 harness."""

    return {
        "serial_warmup_requests": a6.WARMUP_REQUESTS,
        "graph_warmup_batch_sizes": list(a6.GRAPH_WARMUP_BATCH_SIZES),
        "graph_warmup_requests": a6.GRAPH_WARMUP_REQUESTS,
        "measurement_requests": a6.SHAREGPT_REQUESTS,
        "concurrency": a6.SHAREGPT_CONCURRENCY,
        "completion_tokens": a6.SHAREGPT_MAX_TOKENS,
        "ttft_slo_ns": a6.TTFT_SLO_NS,
        "pipeline_depth": a6.KAIRYU_PIPELINE_DEPTH,
        "allocated_cache_blocks": a6.KAIRYU_ALLOCATED_CACHE_BLOCKS,
        "usable_cache_blocks": a6.USABLE_CACHE_BLOCKS,
        "max_num_batched_tokens": a6.MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": a6.MAX_NUM_SEQS,
        "max_model_len": a6.MAX_MODEL_LEN,
    }


def validate_a6_contract() -> None:
    observed = observed_a6_contract()
    if observed != _A6_CONTRACT_PIN:
        raise DiagnosticRunError(
            f"current G2 A6 helpers differ from the issue #333 diagnostic pin: {observed}"
        )


def render_kairyu_config(template: str, *, arm: str) -> str:
    """Render the formal A6 TP4 YAML, changing only the engine backend."""

    if arm not in ARMS:
        raise DiagnosticRunError(f"unknown issue #333 arm: {arm!r}")
    marker = "__TENSOR_PARALLEL_SIZE__"
    backend_line = "    backend: kairyu\n"
    if template.count(marker) != 1 or template.count(backend_line) != 1:
        raise DiagnosticRunError(
            "A6 Kairyu template must contain one TP marker and one backend line"
        )
    rendered = template.replace(marker, str(TP_SIZE))
    if arm == "kairyu-proc":
        rendered = rendered.replace(backend_line, "    backend: kairyu-proc\n")
    parsed = yaml.safe_load(rendered)
    if parsed != expected_parsed_config(arm):
        raise DiagnosticRunError(f"rendered {arm} YAML differs from the A6 TP4 pin")
    return rendered


def expected_parsed_config(arm: str) -> dict[str, object]:
    expected = copy.deepcopy(a6.expected_kairyu_config(TP_SIZE))
    engines = expected["engines"]
    assert isinstance(engines, dict)
    engine = engines["qwen3-32b"]
    assert isinstance(engine, dict)
    engine["backend"] = arm
    return expected


def configuration_source(text: str, *, arm: str) -> dict[str, object]:
    parsed = yaml.safe_load(text)
    if parsed != expected_parsed_config(arm):
        raise DiagnosticRunError(f"{arm} configuration is not the pinned A6 TP4 shape")
    return hashed_descriptor(
        {
            "format": "yaml",
            "text": text,
            "text_sha256": sha256_bytes(text.encode()),
            "parsed": parsed,
        }
    )


def backend_neutral_configuration(value: object) -> object:
    result = copy.deepcopy(value)
    if not isinstance(result, dict):
        return result
    engines = result.get("engines")
    if not isinstance(engines, dict):
        return result
    engine = engines.get("qwen3-32b")
    if isinstance(engine, dict):
        engine["backend"] = "<diagnostic-arm>"
    return result


def _scenario_from_identity(value: Mapping[str, object]) -> Cell:
    try:
        cell = Cell(
            order_index=int(value["order_index"]),
            repeat=int(value["repeat"]),
            arm=str(value["arm"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid issue #333 scenario identity") from error
    if cell not in diagnostic_plan():
        raise ValueError(f"scenario is outside the exact ABBA plan: {cell}")
    if (
        value.get("scenario_id") != cell.scenario_id
        or value.get("tensor_parallel_size") != TP_SIZE
        or value.get("workload") != WORKLOAD
    ):
        raise ValueError("issue #333 scenario identity is inconsistent")
    return cell


def backend_projection(payload: object) -> dict[str, object]:
    """Retain the exact backend/topology fields needed for arm attestation."""

    if not isinstance(payload, dict):
        raise DiagnosticRunError("GET /backends returned a non-object")
    engines = payload.get("engines")
    if not isinstance(engines, list) or len(engines) != 1:
        raise DiagnosticRunError("GET /backends must report exactly one engine")
    engine = engines[0]
    if not isinstance(engine, dict):
        raise DiagnosticRunError("GET /backends engine entry is not an object")
    return {
        "attestation": "GET /backends",
        "attention_backend": payload.get("attention_backend"),
        "requested_attention_backend": payload.get("requested_attention_backend"),
        "attention_components": payload.get("attention_components"),
        "source": payload.get("source"),
        "role": payload.get("role"),
        "engine_model": engine.get("model"),
        "engine_backend": engine.get("engine_backend"),
        "engine_attention_backend": engine.get("attention_backend"),
        "engine_attention_components": engine.get("attention_components"),
        "engine_decision_status": engine.get("decision_status"),
        "tensor_parallel_size": engine.get("tensor_parallel_size"),
    }


def expected_backend_projection(arm: str) -> dict[str, object]:
    expected = copy.deepcopy(a6.expected_resolved_backend(arm="kairyu", tp=TP_SIZE))
    expected["attestation"] = "GET /backends"
    expected["engine_model"] = "qwen3-32b"
    expected["engine_backend"] = arm
    return expected


def attest_backend(endpoint: str, *, arm: str) -> dict[str, object]:
    try:
        direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with direct.open(f"{endpoint}/backends", timeout=10) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise DiagnosticRunError(f"cannot attest {arm} /backends: {error}") from error
    measured = backend_projection(payload)
    expected = expected_backend_projection(arm)
    if measured != expected:
        raise DiagnosticRunError(
            f"{arm} /backends differs from the TP4 diagnostic pin: "
            f"observed={measured}, expected={expected}"
        )
    return hashed_descriptor(measured)


def validate_backend_descriptor(value: object, *, arm: str) -> bool:
    return (
        descriptor_valid(value)
        and isinstance(value, dict)
        and value.get("value") == expected_backend_projection(arm)
    )


_PROCESS_PROBE_CODE = r"""
import json, os
from pathlib import Path
rows=[]
me=os.getpid()
for item in Path('/proc').iterdir():
    if not item.name.isdigit():
        continue
    try:
        raw=(item/'stat').read_text()
        close=raw.rindex(')')
        pid=int(raw[:raw.index(' ')])
        comm=raw[raw.index('(')+1:close]
        fields=raw[close+2:].split()
        state=fields[0]
        ppid=int(fields[1])
        pgid=int(fields[2])
        cmd=(item/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        continue
    if pid == me or state == 'Z' or not cmd:
        continue
    is_api = pid == 1 and 'kairyu' in cmd
    is_spawn = 'spawn_main' in cmd or '--multiprocessing-fork' in cmd
    if not (is_api or is_spawn):
        continue
    rows.append({'pid':pid,'ppid':ppid,'pgid':pgid,'state':state,'comm':comm,'argv':cmd})
print(json.dumps(sorted(rows,key=lambda row:row['pid']),sort_keys=True))
""".strip()


def _derived_process_tree_shape(
    rows: object,
    *,
    arm: str,
) -> dict[str, object] | None:
    """Derive the TP4 process shape from raw kernel identities only."""

    expected_rows = TP_SIZE + 1 if arm == "kairyu-proc" else TP_SIZE
    if not isinstance(rows, list) or len(rows) != expected_rows:
        return None
    required = {"pid", "ppid", "pgid", "state", "comm", "argv"}
    if any(
        not isinstance(row, dict)
        or set(row) != required
        or type(row.get("pid")) is not int
        or int(row["pid"]) < 1
        or type(row.get("ppid")) is not int
        or int(row["ppid"]) < 0
        or type(row.get("pgid")) is not int
        or int(row["pgid"]) < 1
        or not isinstance(row.get("state"), str)
        or row["state"] in {"Z", "X"}
        or not isinstance(row.get("comm"), str)
        or not row["comm"]
        or not isinstance(row.get("argv"), str)
        or not row["argv"]
        for row in rows
    ):
        return None
    typed_rows = [row for row in rows if isinstance(row, dict)]
    pids = [int(row["pid"]) for row in typed_rows]
    if len(set(pids)) != len(pids) or pids != sorted(pids):
        return None
    api_rows = [row for row in typed_rows if row["pid"] == 1]
    if len(api_rows) != 1 or "kairyu" not in str(api_rows[0]["argv"]):
        return None
    api = api_rows[0]
    spawned = [row for row in typed_rows if row["pid"] != 1]
    if any(
        "spawn_main" not in str(row["argv"]) and "--multiprocessing-fork" not in str(row["argv"])
        for row in spawned
    ):
        return None
    if arm == "kairyu-proc":
        services = [row for row in spawned if row["ppid"] == 1 and row["pgid"] == row["pid"]]
        if len(services) != 1:
            return None
        service = services[0]
        service_pid = int(service["pid"])
        group = [row for row in typed_rows if row["pgid"] == service_pid]
        workers = [row for row in group if row["pid"] != service_pid]
        group_pids = {int(row["pid"]) for row in group}
        spawned_pids = {int(row["pid"]) for row in spawned}
        if (
            len(group) != TP_SIZE
            or len(workers) != TP_SIZE - 1
            or any(row["ppid"] != service_pid for row in workers)
            or group_pids != spawned_pids
        ):
            return None
        return {
            "api_pid": 1,
            "api_pgid": int(api["pgid"]),
            "engine_service_pid": service_pid,
            "owned_process_group_id": service_pid,
            "spawn_worker_pids": sorted(int(row["pid"]) for row in workers),
            "live_spawn_worker_count": len(workers),
            "api_engine_process_split": True,
        }
    if arm != "kairyu":
        return None
    # In-process rank 0 is API PID 1. Its TP_SIZE-1 followers inherit the API
    # group directly; no child may establish a service session or own ranks.
    if (
        len(spawned) != TP_SIZE - 1
        or any(row["ppid"] != 1 for row in spawned)
        or any(row["pgid"] != api["pgid"] for row in spawned)
        or any(row["pgid"] == row["pid"] for row in spawned)
    ):
        return None
    return {
        "api_pid": 1,
        "api_pgid": int(api["pgid"]),
        "engine_service_pid": None,
        "owned_process_group_id": None,
        "spawn_worker_pids": sorted(int(row["pid"]) for row in spawned),
        "live_spawn_worker_count": len(spawned),
        "api_engine_process_split": False,
    }


def process_tree_attestation(name: str, *, arm: str) -> dict[str, object]:
    result = formal.run_command(
        (
            "docker",
            "exec",
            name,
            "/app/.venv/bin/python",
            "-c",
            _PROCESS_PROBE_CODE,
        ),
        timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )
    try:
        rows = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise DiagnosticRunError("cannot parse container process tree") from error
    derived = _derived_process_tree_shape(rows, arm=arm)
    if derived is None:
        raise DiagnosticRunError(f"{arm} process tree does not prove the exact live TP4 topology")
    return hashed_descriptor({"arm": arm, "rows": rows, **derived})


def validate_process_tree_descriptor(value: object, *, arm: str) -> bool:
    if not descriptor_valid(value) or not isinstance(value, dict):
        return False
    raw = value.get("value")
    if not isinstance(raw, dict) or raw.get("arm") != arm:
        return False
    rows = raw.get("rows")
    derived = _derived_process_tree_shape(rows, arm=arm)
    return derived is not None and raw == {"arm": arm, "rows": rows, **derived}


def process_tree_generation_identity(
    value: object,
    *,
    arm: str,
) -> tuple[object, ...] | None:
    """Return every generation-stable field, intentionally ignoring only state."""

    if not validate_process_tree_descriptor(value, arm=arm) or not isinstance(value, dict):
        return None
    raw = value.get("value")
    rows = raw.get("rows") if isinstance(raw, dict) else None
    derived = _derived_process_tree_shape(rows, arm=arm)
    if not isinstance(raw, dict) or not isinstance(rows, list) or derived is None:
        return None
    row_identities = tuple(
        (
            int(row["pid"]),
            int(row["ppid"]),
            int(row["pgid"]),
            str(row["comm"]),
            str(row["argv"]),
        )
        for row in rows
        if isinstance(row, dict)
    )
    return (
        arm,
        row_identities,
        int(derived["api_pid"]),
        int(derived["api_pgid"]),
        derived["engine_service_pid"],
        derived["owned_process_group_id"],
        tuple(derived["spawn_worker_pids"]),
        int(derived["live_spawn_worker_count"]),
        bool(derived["api_engine_process_split"]),
    )


def critical_source_modules() -> dict[str, str]:
    """Map every attested source path to the exact imported Python module."""

    return {
        relative: relative.removesuffix(".py").replace("/", ".")
        for relative in CRITICAL_SOURCE_FILES
    }


def _container_import_hashes(name: str) -> dict[str, dict[str, str]]:
    modules = critical_source_modules()
    code = (
        "import hashlib,importlib,json,sys\n"
        "from pathlib import Path\n"
        f"modules={modules!r}\n"
        "root=Path('/app')\n"
        f"prefix=Path({PYCACHE_PREFIX!r})\n"
        f"if sys.pycache_prefix != {PYCACHE_PREFIX!r}:\n"
        "  raise RuntimeError(f'unexpected pycache prefix: {sys.pycache_prefix!r}')\n"
        "if not sys.dont_write_bytecode:\n"
        "  raise RuntimeError('container may write or consume source-adjacent bytecode')\n"
        "if not prefix.is_dir() or any(prefix.iterdir()):\n"
        "  raise RuntimeError('isolated pycache prefix is absent or non-empty')\n"
        "result={}\n"
        "for relative,name in modules.items():\n"
        "  module=importlib.import_module(name)\n"
        "  actual=Path(module.__file__).resolve()\n"
        "  expected=(root/relative).resolve()\n"
        "  if actual != expected:\n"
        "    raise RuntimeError(f'{name} imported from {actual}, expected {expected}')\n"
        "  result[relative]={'module':name,'path':str(actual),"
        "'sha256':hashlib.sha256(actual.read_bytes()).hexdigest()}\n"
        "if any(prefix.iterdir()):\n"
        "  raise RuntimeError('module imports populated the isolated pycache prefix')\n"
        "print(json.dumps(result,sort_keys=True))"
    )
    result = formal.run_command(
        ("docker", "exec", name, "/app/.venv/bin/python", "-c", code),
        timeout_s=DOCKER_ATTEST_TIMEOUT_S,
    )
    try:
        value = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise DiagnosticRunError("cannot parse container source hashes") from error
    if not isinstance(value, dict) or set(value) != set(modules):
        raise DiagnosticRunError("container source hash attestation is incomplete")
    result: dict[str, dict[str, str]] = {}
    for relative, item in value.items():
        expected = {
            "module": modules[str(relative)],
            "path": str((Path("/app") / str(relative)).resolve()),
        }
        if (
            not isinstance(item, dict)
            or item.get("module") != expected["module"]
            or item.get("path") != expected["path"]
            or not isinstance(item.get("sha256"), str)
            or HEX64.fullmatch(str(item["sha256"])) is None
        ):
            raise DiagnosticRunError(
                f"container import-source attestation is invalid for {relative}"
            )
        result[str(relative)] = {
            "module": str(item["module"]),
            "path": str(item["path"]),
            "sha256": str(item["sha256"]),
        }
    return result


def source_file_hashes(repo: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in CRITICAL_SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise DiagnosticRunError(f"missing critical source file: {path}")
        result[relative] = sha256_file(path)
    return result


def runner_source_hashes(repo: Path) -> dict[str, str]:
    """Bind the live operator helpers to one clean detached source root."""

    resolved_repo = repo.resolve()
    module_paths = {
        "bench/issue_333_proc_http_bench.py": Path(__file__).resolve(),
        "bench/g2_a6_vllm_bench.py": Path(a6.__file__).resolve(),
        "bench/run_g2_a6_formal.py": Path(formal.__file__).resolve(),
    }
    if resolved_repo != _REPO_ROOT.resolve():
        raise DiagnosticRunError("--repo must be the source root executing the issue #333 operator")
    for relative, actual in module_paths.items():
        expected = (resolved_repo / relative).resolve()
        if actual != expected:
            raise DiagnosticRunError(f"running {relative} came from {actual}, expected {expected}")
    symbolic = formal.run_command(
        ("git", "symbolic-ref", "-q", "HEAD"),
        cwd=resolved_repo,
        check=False,
        announce=False,
        timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )
    if symbolic.returncode == 0:
        branch = (symbolic.stdout or "").strip()
        raise DiagnosticRunError(
            f"live diagnostic requires detached HEAD, found {branch or 'attached HEAD'}"
        )
    if symbolic.returncode != 1:
        raise DiagnosticRunError(
            "cannot prove detached HEAD: " + (symbolic.stderr or "unknown git error").strip()
        )
    bytecode = sorted(
        path.relative_to(resolved_repo).as_posix()
        for source_root in (resolved_repo / "bench", resolved_repo / "kairyu")
        for pattern in ("*.pyc", "*.pyo")
        for path in source_root.rglob(pattern)
    )
    if bytecode:
        preview = bytecode[:8]
        raise DiagnosticRunError(
            f"detached source contains ignored Python bytecode; use a fresh checkout: {preview}"
        )
    return {relative: sha256_file(resolved_repo / relative) for relative in RUNNER_SOURCE_FILES}


def container_launch_attestation(
    *,
    args: argparse.Namespace,
    name: str,
    cell: Cell,
    config_path: Path,
    config_descriptor: dict[str, object],
    image_id: str,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    result = formal.run_command(
        ("docker", "inspect", name),
        timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )
    try:
        inspected = json.loads((result.stdout or "").strip())
    except json.JSONDecodeError as error:
        raise DiagnosticRunError("cannot parse Docker inspection") from error
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise DiagnosticRunError("Docker inspection did not return one container")
    container = inspected[0]
    if not isinstance(container, dict):
        raise DiagnosticRunError("Docker inspection container is not an object")
    config = container.get("Config")
    host = container.get("HostConfig")
    mounts = container.get("Mounts")
    if not isinstance(config, dict) or not isinstance(host, dict) or not isinstance(mounts, list):
        raise DiagnosticRunError("Docker inspection omitted Config/HostConfig/Mounts")
    if container.get("Image") != image_id or config.get("Image") != image_id:
        raise DiagnosticRunError("container does not use the immutable image ID")
    if config.get("Entrypoint") != ["/app/.venv/bin/kairyu"]:
        raise DiagnosticRunError("container entrypoint differs from the pin")
    if config.get("Cmd") != ["serve", "/etc/kairyu/config.yaml"]:
        raise DiagnosticRunError("container argv differs from the pin")
    env = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in config.get("Env", [])
        if isinstance(item, str) and "=" in item
    }
    required_env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX,
        "KAIRYU_ATTENTION_BACKEND": "flashinfer",
    }
    if any(env.get(key) != value for key, value in required_env.items()):
        raise DiagnosticRunError("container environment differs from the pin")

    expected_mounts = {
        "/models": ("volume", args.model_volume, None),
        "/workspace": ("bind", None, args.repo.resolve()),
        "/app/kairyu": ("bind", None, (args.repo / "kairyu").resolve()),
        "/etc/kairyu/config.yaml": ("bind", None, config_path.resolve()),
    }
    normalized_mounts: list[dict[str, object]] = []
    seen: set[str] = set()
    for mount in mounts:
        if not isinstance(mount, dict):
            raise DiagnosticRunError("Docker mount is not an object")
        destination = mount.get("Destination")
        if destination not in expected_mounts:
            raise DiagnosticRunError(f"container has an unexpected mount: {destination!r}")
        seen.add(str(destination))
        expected_type, expected_name, expected_source = expected_mounts[str(destination)]
        if mount.get("Type") != expected_type or mount.get("RW") is not False:
            raise DiagnosticRunError(f"mount {destination} type/RW differs from the pin")
        if expected_name is not None and mount.get("Name") != expected_name:
            raise DiagnosticRunError(f"mount {destination} volume differs from the pin")
        if expected_source is not None:
            source = mount.get("Source")
            if not isinstance(source, str) or Path(source).resolve() != expected_source:
                raise DiagnosticRunError(f"mount {destination} source differs from the pin")
        normalized_mounts.append(
            {
                "destination": destination,
                "type": mount.get("Type"),
                "name": mount.get("Name"),
                "rw": mount.get("RW"),
            }
        )
    if seen != set(expected_mounts):
        missing = set(expected_mounts) - seen
        raise DiagnosticRunError(f"container is missing required mounts: {missing}")

    ulimits = host.get("Ulimits")
    memlock = [
        item for item in ulimits or [] if isinstance(item, dict) and item.get("Name") == "memlock"
    ]
    ports = host.get("PortBindings")
    bindings = ports.get("8000/tcp") if isinstance(ports, dict) else None
    requests = host.get("DeviceRequests")
    gpu_requests = [item for item in requests or [] if isinstance(item, dict)]
    expected_devices = [str(index) for index in range(TP_SIZE)]
    if (
        host.get("IpcMode") != "host"
        or host.get("NetworkMode") != "bridge"
        or host.get("Tmpfs") != {PYCACHE_PREFIX: "ro,nosuid,nodev,noexec,size=1m"}
        or len(memlock) != 1
        or memlock[0].get("Soft") != -1
        or memlock[0].get("Hard") != -1
        or not isinstance(bindings, list)
        or len(bindings) != 1
        or set(ports) != {"8000/tcp"}
        or bindings[0].get("HostIp") != "127.0.0.1"
        or bindings[0].get("HostPort") != str(args.port)
        or len(gpu_requests) != 1
        or gpu_requests[0].get("Driver") not in {"", "nvidia"}
        or gpu_requests[0].get("Count") != 0
        or gpu_requests[0].get("DeviceIDs") != expected_devices
        or gpu_requests[0].get("Capabilities") != [["gpu"]]
        or gpu_requests[0].get("Options") != {}
    ):
        raise DiagnosticRunError("container HostConfig differs from the TP4 pin")

    container_imports = _container_import_hashes(name)
    container_hashes = {relative: item["sha256"] for relative, item in container_imports.items()}
    if container_hashes != source_hashes:
        raise DiagnosticRunError("container-executed source bytes differ from clean repo")
    source_raw = config_descriptor.get("value")
    if not isinstance(source_raw, dict):
        raise DiagnosticRunError("configuration descriptor is invalid")
    mounted_config = formal.run_command(
        (
            "docker",
            "exec",
            name,
            "/app/.venv/bin/python",
            "-c",
            (
                "import hashlib,json; from pathlib import Path; "
                "p=Path('/etc/kairyu/config.yaml'); b=p.read_bytes(); "
                "print(json.dumps({'sha256':hashlib.sha256(b).hexdigest(),"
                "'text':b.decode()}))"
            ),
        ),
        timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )
    try:
        mounted = json.loads((mounted_config.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise DiagnosticRunError("cannot parse mounted config attestation") from error
    if (
        not isinstance(mounted, dict)
        or mounted.get("sha256") != source_raw.get("text_sha256")
        or mounted.get("text") != source_raw.get("text")
    ):
        raise DiagnosticRunError("mounted YAML bytes differ from retained config")
    measured = {
        "container_id": container.get("Id"),
        "image_id": container.get("Image"),
        "entrypoint": config.get("Entrypoint"),
        "argv": config.get("Cmd"),
        "required_environment": required_env,
        "mounts": sorted(normalized_mounts, key=lambda item: str(item["destination"])),
        "host_config": {
            "ipc_mode": host.get("IpcMode"),
            "network_mode": host.get("NetworkMode"),
            "tmpfs": host.get("Tmpfs"),
            "memlock_soft": memlock[0].get("Soft"),
            "memlock_hard": memlock[0].get("Hard"),
            "published_host_ip": bindings[0].get("HostIp"),
            "published_host_port": int(bindings[0]["HostPort"]),
            "gpu_driver": gpu_requests[0].get("Driver"),
            "gpu_count": gpu_requests[0].get("Count"),
            "gpu_device_ids": gpu_requests[0].get("DeviceIDs"),
            "gpu_capabilities": gpu_requests[0].get("Capabilities"),
            "gpu_options": gpu_requests[0].get("Options"),
        },
        "source_files": container_hashes,
        "engine_imports": container_imports,
        "python_import_policy": {
            "dont_write_bytecode": True,
            "pycache_prefix": PYCACHE_PREFIX,
            "prefix_empty_before_and_after_probe": True,
        },
        "configuration_text_sha256": source_raw.get("text_sha256"),
        "diagnostic_arm": cell.arm,
    }
    container_id = measured["container_id"]
    if not isinstance(container_id, str) or HEX64.fullmatch(container_id) is None:
        raise DiagnosticRunError("Docker container ID is not a full SHA-256 identity")
    return hashed_descriptor(measured)


def docker_command(
    *,
    args: argparse.Namespace,
    name: str,
    config_path: Path,
    image_id: str,
) -> list[str]:
    devices = ",".join(str(index) for index in range(TP_SIZE))
    return [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        f'"device={devices}"',
        "--network",
        "bridge",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "-p",
        f"127.0.0.1:{args.port}:8000",
        "-v",
        f"{args.model_volume}:/models:ro",
        "-v",
        f"{args.repo.resolve()}:/workspace:ro",
        "-v",
        f"{(args.repo / 'kairyu').resolve()}:/app/kairyu:ro",
        "-v",
        f"{config_path.resolve()}:/etc/kairyu/config.yaml:ro",
        "--tmpfs",
        f"{PYCACHE_PREFIX}:ro,nosuid,nodev,noexec,size=1m",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        f"PYTHONPYCACHEPREFIX={PYCACHE_PREFIX}",
        "-e",
        "KAIRYU_ATTENTION_BACKEND=flashinfer",
        "--entrypoint",
        "/app/.venv/bin/kairyu",
        image_id,
        "serve",
        "/etc/kairyu/config.yaml",
    ]


def _container_name(session_id: str, cell: Cell) -> str:
    session = re.sub(r"[^a-z0-9]+", "-", session_id.lower()).strip("-")[-16:]
    arm = cell.arm.replace("kairyu-", "")
    return f"g2a6-issue333-{session}-{cell.order_index}-{arm}"[:120].rstrip("-")


def _port_available(port: int) -> bool:
    """Return false only for a live loopback listener, not stale TIME_WAIT."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _selected_gpu_compute_applications(
    inventory: Sequence[formal.GPU],
    *,
    timeout_s: float = NVML_QUERY_TIMEOUT_S,
) -> list[dict[str, object]]:
    selected = list(inventory[:TP_SIZE])
    selected_uuids = {item.uuid for item in selected}
    try:
        applications = formal.run_command(
            (
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ),
            check=False,
            timeout_s=timeout_s,
        )
    except formal.FormalRunError as error:
        raise DiagnosticRunError(
            f"cannot query selected-GPU compute applications: {error}"
        ) from error
    if applications.returncode != 0:
        raise DiagnosticRunError(
            "cannot query selected-GPU compute applications: "
            + (applications.stderr or "nvidia-smi failed").strip()
        )
    rows: list[dict[str, object]] = []
    for line in (applications.stdout or "").splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or not all(fields):
            raise DiagnosticRunError(f"invalid nvidia-smi compute application row: {line!r}")
        try:
            pid = int(fields[1])
        except ValueError as error:
            raise DiagnosticRunError(
                f"invalid nvidia-smi compute application PID: {line!r}"
            ) from error
        if pid < 1:
            raise DiagnosticRunError(f"invalid nvidia-smi compute application PID: {line!r}")
        if fields[0] not in selected_uuids:
            continue
        rows.append({"gpu_uuid": fields[0], "pid": pid, "process_name": fields[2]})
    return rows


def _selected_gpu_runtime_states(
    inventory: Sequence[formal.GPU],
    *,
    timeout_s: float = NVML_QUERY_TIMEOUT_S,
) -> list[dict[str, object]]:
    selected = list(inventory[:TP_SIZE])
    selected_uuids = [item.uuid for item in selected]
    try:
        runtime = formal.run_command(
            (
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,utilization.gpu,pstate",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            timeout_s=timeout_s,
        )
    except formal.FormalRunError as error:
        raise DiagnosticRunError(
            f"cannot query selected-GPU runtime state: {error}"
        ) from error
    if runtime.returncode != 0:
        raise DiagnosticRunError(
            "cannot query selected-GPU runtime state: "
            + (runtime.stderr or "nvidia-smi failed").strip()
        )
    observed: dict[str, dict[str, object]] = {}
    for line in (runtime.stdout or "").splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or not all(fields):
            raise DiagnosticRunError(f"invalid nvidia-smi GPU runtime row: {line!r}")
        gpu_uuid = fields[0]
        if gpu_uuid not in selected_uuids:
            continue
        if gpu_uuid in observed:
            raise DiagnosticRunError(f"duplicate nvidia-smi GPU runtime row: {line!r}")
        try:
            memory_used_mib = int(fields[1])
            utilization_percent = int(fields[2])
        except ValueError as error:
            raise DiagnosticRunError(
                f"invalid nvidia-smi GPU runtime counters: {line!r}"
            ) from error
        if memory_used_mib < 0 or not 0 <= utilization_percent <= 100:
            raise DiagnosticRunError(f"invalid nvidia-smi GPU runtime counters: {line!r}")
        observed[gpu_uuid] = {
            "gpu_uuid": gpu_uuid,
            "memory_used_mib": memory_used_mib,
            "utilization_percent": utilization_percent,
            "pstate": fields[3],
        }
    if set(observed) != set(selected_uuids):
        missing = [gpu_uuid for gpu_uuid in selected_uuids if gpu_uuid not in observed]
        raise DiagnosticRunError(f"selected GPUs absent from nvidia-smi runtime state: {missing}")
    return [observed[gpu_uuid] for gpu_uuid in selected_uuids]


def _gpu_memory_baseline(states: Sequence[Mapping[str, object]]) -> dict[str, int]:
    baseline: dict[str, int] = {}
    for state in states:
        gpu_uuid = state.get("gpu_uuid")
        memory_used_mib = state.get("memory_used_mib")
        if (
            not isinstance(gpu_uuid, str)
            or type(memory_used_mib) is not int
            or memory_used_mib < 0
            or gpu_uuid in baseline
        ):
            raise DiagnosticRunError("cannot derive selected-GPU idle memory baseline")
        baseline[gpu_uuid] = memory_used_mib
    return baseline


def _gpu_runtime_is_idle(
    states: Sequence[Mapping[str, object]],
    *,
    memory_baseline_mib: Mapping[str, int] | None,
) -> bool:
    if any(
        type(state.get("utilization_percent")) is not int
        or state.get("utilization_percent") != 0
        for state in states
    ):
        return False
    if memory_baseline_mib is None:
        return True
    try:
        return _gpu_memory_baseline(states) == dict(memory_baseline_mib)
    except DiagnosticRunError:
        return False


def _runtime_state_signature(
    states: Sequence[Mapping[str, object]],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            state.get("gpu_uuid"),
            state.get("memory_used_mib"),
            state.get("utilization_percent"),
            state.get("pstate"),
        )
        for state in states
    )


def _selected_gpu_quiescence(
    inventory: Sequence[formal.GPU],
    *,
    memory_baseline_mib: Mapping[str, int] | None = None,
) -> dict[str, object]:
    selected = list(inventory[:TP_SIZE])
    applications = _selected_gpu_compute_applications(inventory)
    runtime_states = _selected_gpu_runtime_states(inventory)
    if applications or not _gpu_runtime_is_idle(
        runtime_states,
        memory_baseline_mib=memory_baseline_mib,
    ):
        raise DiagnosticRunError(
            "selected GPUs are not compute-idle: "
            f"applications={applications}, runtime_states={runtime_states}"
        )
    return {
        "selected_gpu_uuids": [item.uuid for item in selected],
        "compute_applications": applications,
        "gpu_runtime_states": runtime_states,
        "polls": 1,
        "stable_samples": 1,
    }


def _wait_for_selected_gpu_quiescence(
    inventory: Sequence[formal.GPU],
    *,
    timeout_s: float = CLEANUP_QUIESCENCE_TIMEOUT_S,
    memory_baseline_mib: Mapping[str, int] | None = None,
) -> dict[str, object]:
    if timeout_s <= 0:
        raise DiagnosticRunError("cleanup quiescence timeout must be positive")
    selected = list(inventory[:TP_SIZE])
    deadline = time.monotonic() + timeout_s
    polls = 0
    last_error: DiagnosticRunError | None = None
    consecutive = 0
    last_signature: tuple[tuple[object, ...], ...] | None = None
    while True:
        polls += 1
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiagnosticRunError("selected-GPU quiescence deadline expired")
            applications = _selected_gpu_compute_applications(
                inventory,
                timeout_s=min(NVML_QUERY_TIMEOUT_S, remaining),
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiagnosticRunError("selected-GPU quiescence deadline expired")
            runtime_states = _selected_gpu_runtime_states(
                inventory,
                timeout_s=min(NVML_QUERY_TIMEOUT_S, remaining),
            )
            last_error = None
            if not applications and _gpu_runtime_is_idle(
                runtime_states,
                memory_baseline_mib=memory_baseline_mib,
            ):
                signature = _runtime_state_signature(runtime_states)
                consecutive = consecutive + 1 if signature == last_signature else 1
                last_signature = signature
                if consecutive >= QUIESCENCE_STABLE_SAMPLES:
                    return {
                        "selected_gpu_uuids": [item.uuid for item in selected],
                        "compute_applications": [],
                        "gpu_runtime_states": runtime_states,
                        "polls": polls,
                        "stable_samples": QUIESCENCE_STABLE_SAMPLES,
                    }
            else:
                consecutive = 0
                last_signature = None
            observation = (
                f"applications={applications}, runtime_states={runtime_states}"
            )
        except DiagnosticRunError as error:
            last_error = error
            consecutive = 0
            last_signature = None
            observation = str(error)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = str(last_error) if last_error is not None else observation
            raise DiagnosticRunError(f"selected GPUs did not become quiescent: {detail}")
        time.sleep(min(CLEANUP_QUIESCENCE_POLL_S, remaining))


def _save_container_logs_strict(name: str, path: Path, *, work_dir: Path) -> dict[str, object]:
    result = formal.run_command(
        ("docker", "logs", "--timestamps", name),
        check=False,
        timeout_s=DOCKER_LOG_TIMEOUT_S,
    )
    content = (result.stdout or "") + (result.stderr or "")
    atomic_write_text(path, content)
    if result.returncode != 0:
        raise DiagnosticRunError(f"cannot retain Docker logs for {name}")
    return {
        "path": path.resolve().relative_to(work_dir.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _container_absent(identifier: str) -> bool:
    full_id = HEX64.fullmatch(identifier) is not None
    filter_value = f"id={identifier}" if full_id else f"name=^/{identifier}$"
    listed = formal.run_command(
        ("docker", "ps", "-aq", "--no-trunc", "--filter", filter_value),
        announce=False,
        timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )
    rows = [line.strip() for line in (listed.stdout or "").splitlines() if line.strip()]
    if any(HEX64.fullmatch(row) is None for row in rows):
        raise DiagnosticRunError("Docker absence query returned a non-container identity")
    return identifier not in rows if full_id else not rows


def _stop_container_strict(
    identifier: str,
    *,
    expected_container_id: str | None = None,
) -> dict[str, object]:
    stopped = formal.run_command(
        (
            "docker",
            "stop",
            "--timeout",
            str(CONTAINER_STOP_TIMEOUT_S),
            identifier,
        ),
        check=False,
        timeout_s=DOCKER_STOP_COMMAND_TIMEOUT_S,
    )
    if stopped.returncode != 0:
        raise DiagnosticRunError(
            f"cannot gracefully stop container {identifier}: "
            + ((stopped.stderr or stopped.stdout) or "docker stop failed").strip()
        )
    inspected = formal.run_command(
        ("docker", "inspect", identifier),
        timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )
    try:
        parsed = json.loads(inspected.stdout)
        container = parsed[0]
        state = container["State"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise DiagnosticRunError(
            f"cannot attest post-stop container state for {identifier}"
        ) from error
    container_id = container.get("Id") if isinstance(container, dict) else None
    if (
        not isinstance(container_id, str)
        or HEX64.fullmatch(container_id) is None
        or expected_container_id is not None
        and container_id != expected_container_id
    ):
        raise DiagnosticRunError(
            f"post-stop container identity differs for {identifier}: {container_id!r}"
        )
    post_stop_state = {
        "running": state.get("Running") if isinstance(state, dict) else None,
        "oom_killed": state.get("OOMKilled") if isinstance(state, dict) else None,
        "exit_code": state.get("ExitCode") if isinstance(state, dict) else None,
        "error": state.get("Error") if isinstance(state, dict) else None,
    }
    if not (
        post_stop_state["running"] is False
        and post_stop_state["oom_killed"] is False
        and type(post_stop_state["exit_code"]) is int
        and post_stop_state["exit_code"] == 0
        and type(post_stop_state["error"]) is str
        and post_stop_state["error"] == ""
    ):
        raise DiagnosticRunError(
            f"container {identifier} did not exit gracefully: {post_stop_state}"
        )
    return {
        "docker_stop_returncode": 0,
        "stop_timeout_s": CONTAINER_STOP_TIMEOUT_S,
        "post_stop_container_id": container_id,
        "post_stop_state": post_stop_state,
    }


def _remove_container_strict(name: str) -> dict[str, object]:
    removed = formal.run_command(
        ("docker", "rm", name),
        check=False,
        timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )
    if removed.returncode != 0:
        raise DiagnosticRunError(
            f"cannot remove container {name}: "
            + ((removed.stderr or removed.stdout) or "docker rm failed").strip()
        )
    if not _container_absent(name):
        raise DiagnosticRunError(f"container {name} remains after docker rm")
    return {
        "docker_rm_returncode": 0,
        "forced": False,
        "absent_after_remove": True,
    }


def _force_remove_container_for_cleanup(name: str) -> None:
    removed = formal.run_command(
        ("docker", "rm", "-f", name),
        check=False,
        timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )
    if removed.returncode != 0 and not _container_absent(name):
        raise DiagnosticRunError(
            f"cannot force-remove container {name}: "
            + ((removed.stderr or removed.stdout) or "docker rm -f failed").strip()
        )
    if not _container_absent(name):
        raise DiagnosticRunError(f"container {name} remains after docker rm -f")


def _attempt_container_cleanup(
    *,
    name: str,
    log_path: Path,
    work_dir: Path,
    inventory: Sequence[formal.GPU],
    expected_container_id: str | None = None,
    memory_baseline_mib: Mapping[str, int] | None = None,
) -> tuple[
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
    list[BaseException],
]:
    """Attempt every cleanup stage even when an earlier stage fails."""

    errors: list[BaseException] = []
    log_record: dict[str, object] | None = None
    removal_record: dict[str, object] | None = None
    post_cleanup: dict[str, object] | None = None
    stop_record: dict[str, object] | None = None
    target = expected_container_id or name
    try:
        stop_record = _stop_container_strict(
            target,
            expected_container_id=expected_container_id,
        )
    except BaseException as error:
        errors.append(error)
    try:
        log_record = _save_container_logs_strict(target, log_path, work_dir=work_dir)
    except BaseException as error:
        errors.append(error)
    if stop_record is not None:
        try:
            removed = _remove_container_strict(target)
            removal_record = {**stop_record, **removed}
        except BaseException as error:
            errors.append(error)
            try:
                _force_remove_container_for_cleanup(target)
            except BaseException as fallback_error:
                errors.append(fallback_error)
    else:
        try:
            _force_remove_container_for_cleanup(target)
        except BaseException as error:
            errors.append(error)
    try:
        post_cleanup = _wait_for_selected_gpu_quiescence(
            inventory,
            memory_baseline_mib=memory_baseline_mib,
        )
    except BaseException as error:
        errors.append(error)
    return log_record, removal_record, post_cleanup, errors


def _fetch_container_gpu_attestation(
    name: str,
    selected: Sequence[formal.GPU],
) -> dict[str, list[str]]:
    return formal.container_gpu_attestation(
        name,
        selected,
        command_timeout_s=DOCKER_CONTROL_TIMEOUT_S,
    )


def _fetch_environment_value(
    *,
    repo: Path,
    path: Path,
    parsed: dict[str, Any],
) -> dict[str, object]:
    return formal.environment_value(repo=repo, path=path, parsed=parsed)


async def measure_cell_requests(
    *,
    endpoint: str,
    trace_bundle: Mapping[str, object],
    request_timeout_s: float,
    api_key: str | None,
    probe_backend: Callable[[], dict[str, object]],
    probe_process_tree: Callable[[], dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Reuse A6's strict SSE executor and exact warm-up/burst geometry."""

    traces = trace_bundle.get("traces")
    graph_warmup = trace_bundle.get("graph_warmup")
    if not isinstance(traces, dict) or not isinstance(graph_warmup, dict):
        raise DiagnosticRunError("trace bundle lacks ShareGPT/graph warmup")
    trace = traces.get(WORKLOAD)
    if not isinstance(trace, dict):
        raise DiagnosticRunError("trace bundle lacks ShareGPT trace")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(
        max_connections=a6.SHAREGPT_CONCURRENCY,
        max_keepalive_connections=a6.SHAREGPT_CONCURRENCY,
    )
    async with httpx.AsyncClient(
        timeout=request_timeout_s,
        headers=headers,
        limits=limits,
        trust_env=False,
    ) as client:
        models = await client.get(f"{endpoint}/v1/models")
        if models.status_code != 200:
            raise DiagnosticRunError(f"/v1/models returned HTTP {models.status_code}")
        payload = models.json()
        available = (
            {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
            if isinstance(payload, dict)
            else set()
        )
        if a6.MODEL not in available:
            raise DiagnosticRunError(f"endpoint does not serve {a6.MODEL}")
        warmups = await a6._execute_serial(
            client,
            endpoint=endpoint,
            planned=a6._planned_requests(trace, phase="warmup", workload=WORKLOAD),
            workload=WORKLOAD,
        )
        graph_warmups = await a6._execute_graph_warmups(
            client,
            endpoint=endpoint,
            planned=a6._graph_warmup_requests(graph_warmup),
            workload=WORKLOAD,
        )
        backend_after_warmup = await asyncio.to_thread(probe_backend)
        process_after_warmup = await asyncio.to_thread(probe_process_tree)
        measurements = await a6._execute_burst(
            client,
            endpoint=endpoint,
            planned=a6._planned_requests(
                trace,
                phase="measurement",
                workload=WORKLOAD,
            ),
            workload=WORKLOAD,
        )
    backend_after_measurement = await asyncio.to_thread(probe_backend)
    process_after_measurement = await asyncio.to_thread(probe_process_tree)
    timing = [row["timing_ns"] for row in measurements if isinstance(row.get("timing_ns"), dict)]
    if len(timing) != a6.SHAREGPT_REQUESTS:
        raise DiagnosticRunError("measurement did not retain every timing row")
    return (
        [*warmups, *graph_warmups, *measurements],
        {
            "backend_after_warmup": backend_after_warmup,
            "backend_after_measurement": backend_after_measurement,
            "process_tree_after_warmup": process_after_warmup,
            "process_tree_after_measurement": process_after_measurement,
            "measurement_window_ns": {
                "start": min(int(item["start"]) for item in timing),
                "end": max(int(item["end"]) for item in timing),
            },
        },
    )


def measurement_rows(
    *,
    cell: Cell,
    trace_bundle: dict[str, object],
    provenance: dict[str, object],
    cleanup: dict[str, object],
    backend_ready: dict[str, object],
    process_tree_ready: dict[str, object],
    requests: Sequence[dict[str, object]],
    end_evidence: Mapping[str, object],
) -> list[dict[str, object]]:
    traces = trace_bundle["traces"]
    assert isinstance(traces, dict)
    trace = traces[WORKLOAD]
    graph_warmup = trace_bundle["graph_warmup"]
    identity = cell.identity()
    decorated = [{**row, **identity} for row in requests]
    return [
        {
            "type": "diagnostic_run",
            "schema_version": SCHEMA_VERSION,
            "diagnostic": DIAGNOSTIC,
            "diagnostic_only": True,
            "a6_gate_claimed": False,
            **identity,
            "benchmark": trace_bundle["benchmark"],
            "trace": trace,
            "graph_warmup": graph_warmup,
            "provenance": provenance,
            "backend_ready": backend_ready,
            "process_tree_ready": process_tree_ready,
        },
        *decorated,
        {
            "type": "diagnostic_run_end",
            **identity,
            "warmup_request_count": sum(
                row.get("phase") in {"warmup", "graph_warmup"} for row in requests
            ),
            "serial_warmup_request_count": sum(row.get("phase") == "warmup" for row in requests),
            "graph_warmup_request_count": sum(
                row.get("phase") == "graph_warmup" for row in requests
            ),
            "measurement_request_count": sum(row.get("phase") == "measurement" for row in requests),
            "measurement_window_ns": end_evidence["measurement_window_ns"],
            "backend_after_warmup": end_evidence["backend_after_warmup"],
            "backend_after_measurement": end_evidence["backend_after_measurement"],
            "process_tree_after_warmup": end_evidence["process_tree_after_warmup"],
            "process_tree_after_measurement": end_evidence["process_tree_after_measurement"],
            "provenance": provenance,
            "cleanup": cleanup,
        },
    ]


def _parse_shard_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[Cell, dict[str, object], list[dict[str, object]], dict[str, object]]:
    if len(rows) < 2:
        raise ValueError("issue #333 shard is empty")
    start, end = rows[0], rows[-1]
    if start.get("type") != "diagnostic_run" or end.get("type") != "diagnostic_run_end":
        raise ValueError("issue #333 shard framing is invalid")
    if (
        start.get("schema_version") != SCHEMA_VERSION
        or start.get("diagnostic") != DIAGNOSTIC
        or start.get("diagnostic_only") is not True
        or start.get("a6_gate_claimed") is not False
    ):
        raise ValueError("issue #333 shard schema/diagnostic marker is invalid")
    cell = _scenario_from_identity(start)
    if _scenario_from_identity(end) != cell:
        raise ValueError("issue #333 shard start/end identity differs")
    requests = list(rows[1:-1])
    if any(row.get("type") != "request" for row in requests):
        raise ValueError("issue #333 shard contains an unknown row type")
    for row in requests:
        if _scenario_from_identity(row) != cell:
            raise ValueError("issue #333 request belongs to another scenario")
    return cell, start, requests, end


def write_measurement_shard(
    path: Path,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    _write_jsonl(path, rows)
    stored = _read_jsonl(path)
    cell, start, requests, end = _parse_shard_rows(stored)
    return {
        "schema_version": RUNNER_SCHEMA,
        "identity": cell.identity(),
        "sha256": sha256_file(path),
        "requests": len(requests),
        "provenance_stable": start.get("provenance") == end.get("provenance"),
        "cleanup_sha256": (
            end.get("cleanup", {}).get("sha256") if isinstance(end.get("cleanup"), dict) else None
        ),
    }


def _artifact_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[
    dict[str, object],
    dict[Cell, dict[str, object]],
    dict[Cell, list[dict[str, object]]],
    dict[Cell, dict[str, object]],
    dict[str, object],
]:
    if len(rows) < 2 or rows[0].get("type") != "diagnostic":
        raise ValueError("artifact must start with one diagnostic row")
    if rows[-1].get("type") != "diagnostic_end":
        raise ValueError("artifact must end with one diagnostic_end row")
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise ValueError("artifact row indices are not contiguous")
    run, run_end = rows[0], rows[-1]
    starts: dict[Cell, dict[str, object]] = {}
    ends: dict[Cell, dict[str, object]] = {}
    requests: dict[Cell, list[dict[str, object]]] = {cell: [] for cell in diagnostic_plan()}
    active: Cell | None = None
    for row in rows[1:-1]:
        row_type = row.get("type")
        if row_type == "scenario_start":
            cell = _scenario_from_identity(row)
            if active is not None or cell in starts:
                raise ValueError("artifact scenario framing overlaps or repeats")
            starts[cell] = row
            active = cell
        elif row_type == "request":
            cell = _scenario_from_identity(row)
            if cell != active:
                raise ValueError("artifact request is outside its scenario frame")
            requests[cell].append(row)
        elif row_type == "scenario_end":
            cell = _scenario_from_identity(row)
            if cell != active or cell in ends:
                raise ValueError("artifact scenario end is out of order")
            ends[cell] = row
            active = None
        else:
            raise ValueError(f"unknown artifact row type: {row_type!r}")
    if active is not None:
        raise ValueError("artifact has an unterminated scenario")
    return run, starts, requests, ends, run_end


def assemble_rows(
    paths: Sequence[Path],
    *,
    external_workdir_logs_required: bool = False,
) -> list[dict[str, object]]:
    shards: dict[
        Cell,
        tuple[dict[str, object], list[dict[str, object]], dict[str, object], str],
    ] = {}
    for path in paths:
        rows = _read_jsonl(path)
        cell, start, requests, end = _parse_shard_rows(rows)
        if cell in shards:
            raise ValueError(f"duplicate issue #333 shard: {cell.scenario_id}")
        shards[cell] = (start, requests, end, sha256_file(path))
    plan = diagnostic_plan()
    if set(shards) != set(plan):
        missing = [cell.scenario_id for cell in plan if cell not in shards]
        raise ValueError(f"shards must contain the exact four-cell ABBA plan; missing={missing}")
    first = shards[plan[0]][0]
    benchmark = first.get("benchmark")
    trace = first.get("trace")
    graph_warmup = first.get("graph_warmup")
    if any(
        start.get("benchmark") != benchmark
        or start.get("trace") != trace
        or start.get("graph_warmup") != graph_warmup
        for start, _, _, _ in shards.values()
    ):
        raise ValueError("issue #333 shards do not share exact A6 trace inputs")
    combined: list[dict[str, object]] = [
        {
            "type": "diagnostic",
            "schema_version": SCHEMA_VERSION,
            "diagnostic": DIAGNOSTIC,
            "diagnostic_only": True,
            "a6_gate_claimed": False,
            "external_workdir_logs_required": external_workdir_logs_required,
            "plan": [cell.identity() for cell in plan],
            "benchmark": benchmark,
            "trace": trace,
            "graph_warmup": graph_warmup,
        }
    ]
    for cell in plan:
        start, requests, end, shard_sha = shards[cell]
        combined.append(
            {
                **{
                    key: value
                    for key, value in start.items()
                    if key
                    not in {
                        "type",
                        "row_index",
                        "benchmark",
                        "trace",
                        "graph_warmup",
                    }
                },
                "type": "scenario_start",
                "measurement_raw_sha256": shard_sha,
                "trace_sha256": (trace.get("sha256") if isinstance(trace, dict) else None),
                "graph_warmup_sha256": (
                    graph_warmup.get("sha256") if isinstance(graph_warmup, dict) else None
                ),
            }
        )
        combined.extend(
            {key: value for key, value in row.items() if key != "row_index"} for row in requests
        )
        combined.append(
            {
                **{key: value for key, value in end.items() if key != "row_index"},
                "type": "scenario_end",
            }
        )
    combined.append(
        {
            "type": "diagnostic_end",
            "scenario_count": len(plan),
            "request_count": sum(len(shards[cell][1]) for cell in plan),
        }
    )
    return combined


def _fraction_record(value: Fraction | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _median(values: Sequence[Fraction]) -> Fraction | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _completion_output_hash_agreement_diagnostic(
    plan: Sequence[Cell],
    hashes: Mapping[Cell, Mapping[int, tuple[str, str]]],
) -> dict[str, object]:
    """Report cross-cell output agreement without making it an integrity gate."""

    expected_sequences = set(range(a6.SHAREGPT_REQUESTS))
    pairs: list[dict[str, object]] = []
    for left, right in combinations(plan, 2):
        left_hashes = hashes.get(left, {})
        right_hashes = hashes.get(right, {})
        complete = set(left_hashes) == expected_sequences and set(right_hashes) == (
            expected_sequences
        )
        matching = (
            sum(
                left_hashes[sequence][0] == right_hashes[sequence][0]
                for sequence in range(a6.SHAREGPT_REQUESTS)
            )
            if complete
            else None
        )
        relationship = (
            "same_arm_repeat"
            if left.arm == right.arm
            else "paired_backend_comparison"
            if left.repeat == right.repeat
            else "cross_arm_cross_repeat"
        )
        pairs.append(
            {
                "left": left.identity(),
                "right": right.identity(),
                "relationship": relationship,
                "complete_inputs": complete,
                "request_count": a6.SHAREGPT_REQUESTS,
                "matching_sequences": matching,
                "agreement_fraction": (
                    _fraction_record(Fraction(matching, a6.SHAREGPT_REQUESTS))
                    if matching is not None
                    else None
                ),
            }
        )
    return {
        "field": "output_text_sha256",
        "binding": False,
        "pair_count": len(pairs),
        "reason": (
            "the discarded trial observed different hashes between same-arm repeats; "
            "fresh c128 servers can plausibly admit and batch simultaneous requests in "
            "different orders, so exact cross-cell equality cannot isolate backend semantics"
        ),
        "integrity_scope": (
            "raw rows retain well-formed output and stream digest metadata, but do not "
            "retain decoded bytes from which those digests could be recomputed; strict "
            "completion structure, per-cell response-ID uniqueness, and serialized-warmup "
            "output parity remain binding"
        ),
        "pairs": pairs,
    }


def _mad(values: Sequence[Fraction]) -> Fraction | None:
    center = _median(values)
    if center is None:
        return None
    return _median([abs(value - center) for value in values])


def material_reduction_classification(
    paired_median_p99_ratio: Fraction | None,
    *,
    evidence_valid: bool,
) -> dict[str, object]:
    """Apply the predeclared report-only issue #333 interpretation."""

    material = (
        evidence_valid
        and paired_median_p99_ratio is not None
        and paired_median_p99_ratio <= MATERIAL_TTFT_P99_RATIO_CEILING
    )
    hypothesis_conclusion: str | None
    if not evidence_valid:
        classification = "insufficient_evidence"
        interpretation = "evidence integrity checks failed; no hypothesis conclusion"
        hypothesis_conclusion = None
    elif paired_median_p99_ratio is None:
        classification = "insufficient_evidence"
        interpretation = "paired median TTFT p99 ratio is unavailable"
        hypothesis_conclusion = None
    elif material:
        classification = "material_reduction"
        interpretation = "dominant process/GIL-contention hypothesis supported"
        hypothesis_conclusion = "supported"
    else:
        classification = "no_material_reduction"
        interpretation = "dominant process/GIL-contention hypothesis not supported"
        hypothesis_conclusion = "not_supported"
    return {
        "binding": False,
        "report_only": True,
        "predeclared_before_measurement": True,
        "metric": "paired median kairyu-proc/kairyu TTFT p99 ratio",
        "criterion": "<= 0.90",
        "ceiling": _fraction_record(MATERIAL_TTFT_P99_RATIO_CEILING),
        "observed": _fraction_record(paired_median_p99_ratio),
        "classification": classification,
        "material_reduction": material,
        "interpretation": interpretation,
        "hypothesis_conclusion": hypothesis_conclusion,
        "evidence_integrity_valid": evidence_valid,
        "causal_scope": (
            "net backend swap including process/GIL isolation and "
            "ZMQ/msgpack/delta/lifecycle overhead"
        ),
    }


def _ratio(numerator: object, denominator: object) -> Fraction | None:
    if type(numerator) is int and type(denominator) is int and denominator > 0:
        return Fraction(numerator, denominator)
    return None


def _goodput_ratio(proc: Mapping[str, object], inproc: Mapping[str, object]) -> Fraction | None:
    proc_slo = proc.get("ttft_slo_successes")
    proc_span = proc.get("measurement_span_ns")
    inproc_slo = inproc.get("ttft_slo_successes")
    inproc_span = inproc.get("measurement_span_ns")
    if (
        type(proc_slo) is int
        and type(proc_span) is int
        and type(inproc_slo) is int
        and type(inproc_span) is int
        and proc_slo >= 0
        and proc_span > 0
        and inproc_slo > 0
        and inproc_span > 0
    ):
        return Fraction(proc_slo * inproc_span, proc_span * inproc_slo)
    return None


def _provenance_value(value: object) -> dict[str, object] | None:
    if not descriptor_valid(value) or not isinstance(value, dict):
        return None
    raw = value.get("value")
    return raw if isinstance(raw, dict) else None


def cleanup_attestation(
    *,
    cell: Cell,
    provenance: dict[str, object],
    log: dict[str, object],
    removal: dict[str, object],
    post_cleanup_quiescence: dict[str, object],
) -> dict[str, object]:
    raw_provenance = _provenance_value(provenance)
    if not isinstance(raw_provenance, dict):
        raise DiagnosticRunError("cannot bind cleanup to invalid provenance")
    launch = raw_provenance.get("container_launch")
    launch_raw = launch.get("value") if isinstance(launch, dict) else None
    if not isinstance(launch_raw, dict):
        raise DiagnosticRunError("cannot bind cleanup to absent container launch")
    session_id = raw_provenance.get("session_id")
    if not isinstance(session_id, str):
        raise DiagnosticRunError("cannot bind cleanup without a session ID")
    return hashed_descriptor(
        {
            "schema_version": 1,
            "identity": cell.identity(),
            "container_name": _container_name(session_id, cell),
            "container_id": launch_raw.get("container_id"),
            "log": log,
            "removal": removal,
            "post_cleanup_gpu_quiescence": post_cleanup_quiescence,
        }
    )


def _quiescence_evidence_valid(
    value: object,
    *,
    gpu_uuids: object,
    memory_baseline_mib: Mapping[str, int] | None = None,
) -> bool:
    if not isinstance(value, dict) or not isinstance(gpu_uuids, list):
        return False
    runtime_states = value.get("gpu_runtime_states")
    if (
        set(value)
        != {
            "selected_gpu_uuids",
            "compute_applications",
            "gpu_runtime_states",
            "polls",
            "stable_samples",
        }
        or value.get("selected_gpu_uuids") != gpu_uuids
        or value.get("compute_applications") != []
        or type(value.get("polls")) is not int
        or int(value["polls"]) < QUIESCENCE_STABLE_SAMPLES
        or type(value.get("stable_samples")) is not int
        or value.get("stable_samples") != QUIESCENCE_STABLE_SAMPLES
        or not isinstance(runtime_states, list)
        or len(runtime_states) != len(gpu_uuids)
    ):
        return False
    for gpu_uuid, state in zip(gpu_uuids, runtime_states, strict=True):
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "gpu_uuid",
                "memory_used_mib",
                "utilization_percent",
                "pstate",
            }
            or state.get("gpu_uuid") != gpu_uuid
            or type(state.get("memory_used_mib")) is not int
            or int(state["memory_used_mib"]) < 0
            or type(state.get("utilization_percent")) is not int
            or state.get("utilization_percent") != 0
            or not isinstance(state.get("pstate"), str)
            or not state["pstate"]
        ):
            return False
    if memory_baseline_mib is not None:
        try:
            if _gpu_memory_baseline(runtime_states) != dict(memory_baseline_mib):
                return False
        except DiagnosticRunError:
            return False
    return True


def _quiescence_memory_baseline(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    states = value.get("gpu_runtime_states")
    if not isinstance(states, list):
        return None
    try:
        return _gpu_memory_baseline(states)
    except DiagnosticRunError:
        return None


def _removal_evidence_valid(value: object, *, container_id: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "docker_stop_returncode",
        "stop_timeout_s",
        "post_stop_container_id",
        "post_stop_state",
        "docker_rm_returncode",
        "forced",
        "absent_after_remove",
    }:
        return False
    state = value.get("post_stop_state")
    return (
        type(value.get("docker_stop_returncode")) is int
        and value.get("docker_stop_returncode") == 0
        and type(value.get("stop_timeout_s")) is int
        and value.get("stop_timeout_s") == CONTAINER_STOP_TIMEOUT_S
        and isinstance(container_id, str)
        and value.get("post_stop_container_id") == container_id
        and isinstance(state, dict)
        and set(state) == {"running", "oom_killed", "exit_code", "error"}
        and state.get("running") is False
        and state.get("oom_killed") is False
        and type(state.get("exit_code")) is int
        and state.get("exit_code") == 0
        and type(state.get("error")) is str
        and state.get("error") == ""
        and type(value.get("docker_rm_returncode")) is int
        and value.get("docker_rm_returncode") == 0
        and value.get("forced") is False
        and value.get("absent_after_remove") is True
    )


def validate_cleanup_descriptor(
    value: object,
    *,
    cell: Cell,
    provenance: object,
) -> bool:
    if not descriptor_valid(value) or not isinstance(value, dict):
        return False
    raw = value.get("value")
    raw_provenance = _provenance_value(provenance)
    if not isinstance(raw, dict) or not isinstance(raw_provenance, dict):
        return False
    launch = raw_provenance.get("container_launch")
    launch_raw = launch.get("value") if isinstance(launch, dict) else None
    gpu = raw_provenance.get("gpu")
    host = raw_provenance.get("host")
    session_id = raw_provenance.get("session_id")
    if (
        not isinstance(launch_raw, dict)
        or not isinstance(gpu, dict)
        or not isinstance(host, dict)
    ):
        return False
    baseline = host.get("run_start_gpu_baseline")
    baseline_memory = _quiescence_memory_baseline(baseline)
    log = raw.get("log")
    removal = raw.get("removal")
    quiescence = raw.get("post_cleanup_gpu_quiescence")
    return (
        set(raw)
        == {
            "schema_version",
            "identity",
            "container_name",
            "container_id",
            "log",
            "removal",
            "post_cleanup_gpu_quiescence",
        }
        and type(raw.get("schema_version")) is int
        and raw.get("schema_version") == 1
        and raw.get("identity") == cell.identity()
        and isinstance(session_id, str)
        and raw.get("container_name") == _container_name(session_id, cell)
        and raw.get("container_id") == launch_raw.get("container_id")
        and isinstance(log, dict)
        and set(log) == {"path", "sha256", "bytes"}
        and log.get("path") == f"logs/{cell.key}.log"
        and isinstance(log.get("sha256"), str)
        and HEX64.fullmatch(str(log["sha256"])) is not None
        and type(log.get("bytes")) is int
        and int(log["bytes"]) >= 0
        and _removal_evidence_valid(
            removal,
            container_id=launch_raw.get("container_id"),
        )
        and _quiescence_evidence_valid(baseline, gpu_uuids=gpu.get("uuids"))
        and baseline_memory is not None
        and _quiescence_evidence_valid(
            quiescence,
            gpu_uuids=gpu.get("uuids"),
            memory_baseline_mib=baseline_memory,
        )
    )


def validate_provenance_descriptor(value: object, *, cell: Cell) -> bool:
    """Validate one independently replayable immutable cell provenance."""

    raw = _provenance_value(value)
    if not isinstance(raw, dict):
        return False
    source = raw.get("source")
    host = raw.get("host")
    gpu = raw.get("gpu")
    image = raw.get("image")
    runtime = raw.get("runtime_versions")
    configuration = raw.get("configuration_source")
    launch = raw.get("container_launch")
    if not (
        type(raw.get("schema_version")) is int
        and raw.get("schema_version") == 1
        and raw.get("diagnostic_only") is True
        and isinstance(raw.get("session_id"), str)
        and bool(raw["session_id"])
        and isinstance(raw.get("server_generation"), str)
        and bool(raw["server_generation"])
        and type(raw.get("server_started_ns")) is int
        and int(raw["server_started_ns"]) > 0
        and raw.get("diagnostic_arm") == cell.arm
        and isinstance(source, dict)
        and source.get("source_clean") is True
        and isinstance(source.get("repo_commit"), str)
        and HEX40.fullmatch(str(source["repo_commit"])) is not None
        and isinstance(host, dict)
        and isinstance(host.get("id"), str)
        and bool(host["id"])
        and isinstance(host.get("environment"), dict)
        and isinstance(gpu, dict)
        and isinstance(image, dict)
        and isinstance(runtime, dict)
        and descriptor_valid(configuration)
        and descriptor_valid(launch)
        and a6._checkpoint_attestation_valid(raw.get("checkpoint"))
        and raw.get("model") == a6.model_identity()
    ):
        return False
    source_hashes = source.get("critical_file_sha256")
    runner_hashes = source.get("runner_file_sha256")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != set(CRITICAL_SOURCE_FILES)
        or any(
            not isinstance(item, str) or HEX64.fullmatch(item) is None
            for item in source_hashes.values()
        )
        or not isinstance(runner_hashes, dict)
        or set(runner_hashes) != set(RUNNER_SOURCE_FILES)
        or runner_hashes
        != {relative: sha256_file(_REPO_ROOT / relative) for relative in RUNNER_SOURCE_FILES}
    ):
        return False
    uuids = gpu.get("uuids")
    pci = gpu.get("pci_bus_ids")
    container_uuids = gpu.get("container_visible_uuids")
    container_pci = gpu.get("container_visible_pci_bus_ids")
    quiescence = host.get("pre_start_gpu_quiescence")
    baseline = host.get("run_start_gpu_baseline")
    baseline_memory = _quiescence_memory_baseline(baseline)
    if not (
        isinstance(uuids, list)
        and len(uuids) == TP_SIZE
        and len(set(uuids)) == TP_SIZE
        and all(isinstance(item, str) and bool(item) for item in uuids)
        and isinstance(pci, list)
        and len(pci) == TP_SIZE
        and len(set(pci)) == TP_SIZE
        and all(isinstance(item, str) and bool(item) for item in pci)
        and container_uuids == uuids
        and container_pci == pci
        and isinstance(gpu.get("product_name"), str)
        and bool(gpu["product_name"])
        and isinstance(gpu.get("driver_version"), str)
        and bool(gpu["driver_version"])
        and _quiescence_evidence_valid(baseline, gpu_uuids=uuids)
        and baseline_memory is not None
        and _quiescence_evidence_valid(
            quiescence,
            gpu_uuids=uuids,
            memory_baseline_mib=baseline_memory,
        )
    ):
        return False
    environment = host.get("environment")
    artifact = environment.get("artifact") if isinstance(environment, dict) else None
    fabric = environment.get("fabric") if isinstance(environment, dict) else None
    minimum_bandwidth = (
        fabric.get("minimum_peer_bandwidth_gbs") if isinstance(fabric, dict) else None
    )
    if not (
        isinstance(environment, dict)
        and set(environment)
        == {
            "measurement_session_id",
            "measured_at_ns",
            "artifact_date",
            "artifact",
            "physical_topology_sha256",
            "fabric",
        }
        and isinstance(environment.get("measurement_session_id"), str)
        and bool(environment["measurement_session_id"])
        and type(environment.get("measured_at_ns")) is int
        and int(environment["measured_at_ns"]) > 0
        and isinstance(environment.get("artifact_date"), str)
        and bool(environment["artifact_date"])
        and isinstance(artifact, dict)
        and set(artifact) == {"path", "sha256"}
        and isinstance(artifact.get("path"), str)
        and str(artifact["path"]).startswith("bench/results/env-")
        and str(artifact["path"]).endswith(".json")
        and isinstance(artifact.get("sha256"), str)
        and HEX64.fullmatch(str(artifact["sha256"])) is not None
        and isinstance(environment.get("physical_topology_sha256"), str)
        and HEX64.fullmatch(str(environment["physical_topology_sha256"])) is not None
        and isinstance(fabric, dict)
        and set(fabric) == {"kind", "raw_p2p_matrix_sha256", "minimum_peer_bandwidth_gbs"}
        and isinstance(fabric.get("kind"), str)
        and bool(fabric["kind"])
        and isinstance(fabric.get("raw_p2p_matrix_sha256"), str)
        and HEX64.fullmatch(str(fabric["raw_p2p_matrix_sha256"])) is not None
        and type(minimum_bandwidth) in {int, float}
        and not isinstance(minimum_bandwidth, bool)
        and math.isfinite(float(minimum_bandwidth))
        and float(minimum_bandwidth) > 0
    ):
        return False
    image_id = image.get("id")
    if (
        not isinstance(image.get("ref"), str)
        or not image["ref"]
        or not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or HEX64.fullmatch(image_id.removeprefix("sha256:")) is None
    ):
        return False
    packages = runtime.get("package_versions")
    if (
        not isinstance(runtime.get("cuda_version"), str)
        or not runtime["cuda_version"]
        or not isinstance(runtime.get("nccl_version"), str)
        or not runtime["nccl_version"]
        or not isinstance(packages, dict)
        or set(packages)
        != {
            "torch",
            "flashinfer",
            "uvloop",
            "httptools",
            "uvicorn",
            "kairyu",
        }
        or any(not isinstance(item, str) or not item for item in packages.values())
    ):
        return False
    config_raw = configuration.get("value") if isinstance(configuration, dict) else None
    launch_raw = launch.get("value") if isinstance(launch, dict) else None
    if not isinstance(config_raw, dict) or not isinstance(launch_raw, dict):
        return False
    text = config_raw.get("text")
    text_sha = config_raw.get("text_sha256")
    container_id = launch_raw.get("container_id")
    launch_host = launch_raw.get("host_config")
    required_environment = launch_raw.get("required_environment")
    mounts = launch_raw.get("mounts")
    expected_mounts = {
        "/models": "volume",
        "/workspace": "bind",
        "/app/kairyu": "bind",
        "/etc/kairyu/config.yaml": "bind",
    }
    mounts_valid = isinstance(mounts, list) and len(mounts) == len(expected_mounts)
    mounts_by_destination: dict[str, dict[str, object]] = {}
    if mounts_valid:
        for item in mounts:
            if (
                not isinstance(item, dict)
                or set(item) != {"destination", "type", "name", "rw"}
                or not isinstance(item.get("destination"), str)
                or item["destination"] in mounts_by_destination
            ):
                mounts_valid = False
                break
            mounts_by_destination[str(item["destination"])] = item
    mounts_valid &= set(mounts_by_destination) == set(expected_mounts)
    if mounts_valid:
        for destination, mount_type in expected_mounts.items():
            item = mounts_by_destination[destination]
            mounts_valid &= item.get("type") == mount_type and item.get("rw") is False
            if destination == "/models":
                mounts_valid &= isinstance(item.get("name"), str) and bool(item["name"])
            else:
                mounts_valid &= item.get("name") is None
    engine_imports = launch_raw.get("engine_imports")
    imports_valid = isinstance(engine_imports, dict) and set(engine_imports) == set(
        CRITICAL_SOURCE_FILES
    )
    if imports_valid:
        for relative, module_name in critical_source_modules().items():
            item = engine_imports.get(relative)
            imports_valid &= (
                isinstance(item, dict)
                and item.get("module") == module_name
                and item.get("path") == str((Path("/app") / relative).resolve())
                and item.get("sha256") == source_hashes[relative]
            )
    if not (
        config_raw.get("format") == "yaml"
        and isinstance(text, str)
        and isinstance(text_sha, str)
        and text_sha == sha256_bytes(text.encode())
        and config_raw.get("parsed") == expected_parsed_config(cell.arm)
        and isinstance(container_id, str)
        and HEX64.fullmatch(container_id) is not None
        and launch_raw.get("image_id") == image_id
        and launch_raw.get("entrypoint") == ["/app/.venv/bin/kairyu"]
        and launch_raw.get("argv") == ["serve", "/etc/kairyu/config.yaml"]
        and required_environment
        == {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": PYCACHE_PREFIX,
            "KAIRYU_ATTENTION_BACKEND": "flashinfer",
        }
        and mounts_valid
        and launch_raw.get("configuration_text_sha256") == text_sha
        and launch_raw.get("diagnostic_arm") == cell.arm
        and launch_raw.get("source_files") == source_hashes
        and imports_valid
        and launch_raw.get("python_import_policy")
        == {
            "dont_write_bytecode": True,
            "pycache_prefix": PYCACHE_PREFIX,
            "prefix_empty_before_and_after_probe": True,
        }
        and isinstance(launch_host, dict)
        and launch_host.get("ipc_mode") == "host"
        and launch_host.get("network_mode") == "bridge"
        and launch_host.get("tmpfs") == {PYCACHE_PREFIX: "ro,nosuid,nodev,noexec,size=1m"}
        and launch_host.get("memlock_soft") == -1
        and launch_host.get("memlock_hard") == -1
        and launch_host.get("published_host_ip") == "127.0.0.1"
        and type(launch_host.get("published_host_port")) is int
        and 1 <= int(launch_host["published_host_port"]) <= 65535
        and launch_host.get("gpu_driver") in {"", "nvidia"}
        and launch_host.get("gpu_count") == 0
        and launch_host.get("gpu_device_ids") == [str(index) for index in range(TP_SIZE)]
        and launch_host.get("gpu_capabilities") == [["gpu"]]
        and launch_host.get("gpu_options") == {}
    ):
        return False
    return True


def _container_id(provenance: object) -> object:
    raw = _provenance_value(provenance)
    launch = raw.get("container_launch") if isinstance(raw, dict) else None
    launch_raw = launch.get("value") if isinstance(launch, dict) else None
    return launch_raw.get("container_id") if isinstance(launch_raw, dict) else None


def _normalized_quiescence_identity(value: object) -> object:
    if not isinstance(value, dict):
        return value
    states = value.get("gpu_runtime_states")
    normalized_states: object = states
    if isinstance(states, list):
        normalized_states = [
            {
                "gpu_uuid": state.get("gpu_uuid"),
                "memory_used_mib": state.get("memory_used_mib"),
                "utilization_percent": state.get("utilization_percent"),
            }
            if isinstance(state, dict)
            else state
            for state in states
        ]
    return {
        "selected_gpu_uuids": value.get("selected_gpu_uuids"),
        "compute_applications": value.get("compute_applications"),
        "gpu_runtime_states": normalized_states,
        "stable_samples": value.get("stable_samples"),
    }


def _provenance_static_identity(value: Mapping[str, object]) -> dict[str, object]:
    launch = value.get("container_launch")
    launch_raw = launch.get("value") if isinstance(launch, dict) else None
    normalized_launch = copy.deepcopy(launch_raw)
    if isinstance(normalized_launch, dict):
        normalized_launch.pop("container_id", None)
        normalized_launch["diagnostic_arm"] = "<diagnostic-arm>"
        normalized_launch["configuration_text_sha256"] = "<diagnostic-arm>"
    config_source = value.get("configuration_source")
    source_raw = config_source.get("value") if isinstance(config_source, dict) else None
    parsed = source_raw.get("parsed") if isinstance(source_raw, dict) else None
    host = copy.deepcopy(value.get("host"))
    if isinstance(host, dict):
        host["pre_start_gpu_quiescence"] = _normalized_quiescence_identity(
            host.get("pre_start_gpu_quiescence")
        )
    return {
        "source": value.get("source"),
        "host": host,
        "gpu": value.get("gpu"),
        "model": value.get("model"),
        "checkpoint": value.get("checkpoint"),
        "image": value.get("image"),
        "runtime_versions": value.get("runtime_versions"),
        "container_launch": normalized_launch,
        "configuration": backend_neutral_configuration(parsed),
    }


def evaluate_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[dict[str, bool], dict[str, object], dict[str, object]]:
    run, starts, requests, ends, run_end = _artifact_rows(rows)
    plan = diagnostic_plan()
    exact_plan = (
        run.get("schema_version") == SCHEMA_VERSION
        and run.get("diagnostic") == DIAGNOSTIC
        and run.get("diagnostic_only") is True
        and run.get("a6_gate_claimed") is False
        and type(run.get("external_workdir_logs_required")) is bool
        and run.get("plan") == [cell.identity() for cell in plan]
        and list(starts) == list(plan)
        and list(ends) == list(plan)
        and run_end.get("scenario_count") == len(plan)
    )
    benchmark = run.get("benchmark")
    dataset_sha = (
        benchmark.get("sharegpt_dataset", {}).get("sha256") if isinstance(benchmark, dict) else None
    )
    benchmark_exact = isinstance(dataset_sha, str) and benchmark == a6.benchmark_config(dataset_sha)
    trace = run.get("trace")
    graph_warmup = run.get("graph_warmup")
    trace_exact = (
        isinstance(dataset_sha, str)
        and a6._trace_valid(trace, workload=WORKLOAD, dataset_sha256=dataset_sha)
        and a6._graph_warmup_trace_valid(graph_warmup)
        and a6._graph_warmup_disjoint(graph_warmup, {WORKLOAD: trace})
    )
    trace_raw = trace.get("value") if isinstance(trace, dict) else None
    expected_warmups = trace_raw.get("warmup") if isinstance(trace_raw, dict) else None
    expected_measurements = trace_raw.get("requests") if isinstance(trace_raw, dict) else None
    counts_exact = exact_plan
    request_evidence_exact = exact_plan and trace_exact
    execution_shape_exact = exact_plan and trace_exact
    attestation_exact = exact_plan
    process_generation_stable = exact_plan
    provenance_exact = exact_plan
    cleanup_exact = exact_plan
    measurement_windows_exact = exact_plan
    measurement_hashes: dict[Cell, dict[int, tuple[str, str]]] = {
        cell: {} for cell in plan
    }
    measurement_hashes_retained_exact = exact_plan
    serial_warmup_hashes: dict[Cell, dict[int, str]] = {cell: {} for cell in plan}
    serial_warmup_output_parity = exact_plan
    response_ids_unique = exact_plan
    metrics: dict[Cell, dict[str, object]] = {}
    provenance_values: dict[Cell, dict[str, object]] = {}
    for cell in plan:
        start = starts.get(cell, {})
        end = ends.get(cell, {})
        cell_rows = requests.get(cell, [])
        warmups = [row for row in cell_rows if row.get("phase") == "warmup"]
        graph_rows = [row for row in cell_rows if row.get("phase") == "graph_warmup"]
        measurement = [row for row in cell_rows if row.get("phase") == "measurement"]
        counts_exact &= (
            len(warmups) == a6.WARMUP_REQUESTS
            and len(graph_rows) == a6.GRAPH_WARMUP_REQUESTS
            and len(measurement) == a6.SHAREGPT_REQUESTS
            and len(cell_rows)
            == a6.WARMUP_REQUESTS + a6.GRAPH_WARMUP_REQUESTS + a6.SHAREGPT_REQUESTS
            and end.get("warmup_request_count") == a6.WARMUP_REQUESTS + a6.GRAPH_WARMUP_REQUESTS
            and end.get("serial_warmup_request_count") == a6.WARMUP_REQUESTS
            and end.get("graph_warmup_request_count") == a6.GRAPH_WARMUP_REQUESTS
            and end.get("measurement_request_count") == a6.SHAREGPT_REQUESTS
        )
        response_ids = [row.get("response_id") for row in cell_rows]
        response_ids_unique &= (
            len(response_ids)
            == a6.WARMUP_REQUESTS + a6.GRAPH_WARMUP_REQUESTS + a6.SHAREGPT_REQUESTS
            and all(isinstance(item, str) and item for item in response_ids)
            and len(set(response_ids)) == len(response_ids)
        )
        for row in warmups:
            sequence = row.get("sequence")
            output_hash = row.get("output_text_sha256")
            if not (
                type(sequence) is int
                and 0 <= sequence < a6.WARMUP_REQUESTS
                and isinstance(output_hash, str)
                and HEX64.fullmatch(output_hash) is not None
                and sequence not in serial_warmup_hashes[cell]
            ):
                serial_warmup_output_parity = False
                continue
            serial_warmup_hashes[cell][sequence] = output_hash
        request_evidence_exact &= (
            isinstance(expected_warmups, list)
            and isinstance(expected_measurements, list)
            and len(expected_warmups) == len(warmups)
            and len(expected_measurements) == len(measurement)
        )
        if isinstance(expected_warmups, list) and isinstance(expected_measurements, list):
            for phase, actual, expected in (
                ("warmup", warmups, expected_warmups),
                ("measurement", measurement, expected_measurements),
            ):
                by_sequence = {row.get("sequence"): row for row in actual}
                request_evidence_exact &= len(by_sequence) == len(actual) and set(
                    by_sequence
                ) == set(range(len(expected)))
                for sequence, expected_row in enumerate(expected):
                    actual_row = by_sequence.get(sequence, {})
                    request_evidence_exact &= isinstance(expected_row, dict) and (
                        a6._request_evidence_valid(
                            actual_row,
                            expected=expected_row if isinstance(expected_row, dict) else {},
                            workload=WORKLOAD,
                            phase=phase,
                        )
                    )
        request_evidence_exact &= isinstance(
            graph_warmup, dict
        ) and a6._graph_warmup_evidence_valid(
            graph_rows,
            graph_warmup=graph_warmup,
            workload=WORKLOAD,
        )
        intervals = [a6._request_interval(row) for row in measurement]
        releases = {row.get("burst_release_ns") for row in measurement}
        release_ns = next(iter(releases)) if len(releases) == 1 else None
        warmup_intervals = [a6._request_interval(row) for row in warmups]
        graph_intervals = [a6._request_interval(row) for row in graph_rows]
        valid_measurements = [item for item in intervals if item is not None]
        execution_shape_exact &= (
            a6._serialized_intervals(warmups)
            and all(item is not None for item in graph_intervals)
            and len(valid_measurements) == a6.SHAREGPT_CONCURRENCY
            and type(release_ns) is int
            and release_ns <= min(item[0] for item in valid_measurements)
            and max(item[0] for item in valid_measurements)
            <= min(item[1] for item in valid_measurements)
            and bool(warmup_intervals)
            and bool(graph_intervals)
            and max(item[1] for item in warmup_intervals if item is not None)
            <= min(item[0] for item in graph_intervals if item is not None)
            and max(item[1] for item in graph_intervals if item is not None)
            <= min(item[0] for item in valid_measurements)
        )
        backend_descriptors = (
            start.get("backend_ready"),
            end.get("backend_after_warmup"),
            end.get("backend_after_measurement"),
        )
        process_descriptors = (
            start.get("process_tree_ready"),
            end.get("process_tree_after_warmup"),
            end.get("process_tree_after_measurement"),
        )
        process_identities = tuple(
            process_tree_generation_identity(item, arm=cell.arm) for item in process_descriptors
        )
        attestation_exact &= (
            all(validate_backend_descriptor(item, arm=cell.arm) for item in backend_descriptors)
            and len({canonical_json(item) for item in backend_descriptors}) == 1
            and all(
                validate_process_tree_descriptor(item, arm=cell.arm) for item in process_descriptors
            )
        )
        process_generation_stable &= (
            all(identity is not None for identity in process_identities)
            and len(set(process_identities)) == 1
        )
        provenance = start.get("provenance")
        raw_provenance = _provenance_value(provenance)
        provenance_exact &= (
            validate_provenance_descriptor(provenance, cell=cell)
            and provenance == end.get("provenance")
            and isinstance(raw_provenance, dict)
            and raw_provenance.get("session_id") == run.get("session_id")
        )
        cleanup_exact &= validate_cleanup_descriptor(
            end.get("cleanup"),
            cell=cell,
            provenance=provenance,
        )
        if isinstance(raw_provenance, dict):
            provenance_values[cell] = raw_provenance
        timing = [row.get("timing_ns") for row in measurement]
        expected_window = (
            {
                "start": min(int(item["start"]) for item in timing if isinstance(item, dict)),
                "end": max(int(item["end"]) for item in timing if isinstance(item, dict)),
            }
            if timing and all(isinstance(item, dict) for item in timing)
            else None
        )
        measurement_windows_exact &= end.get("measurement_window_ns") == expected_window
        for row in measurement:
            sequence = row.get("sequence")
            output_hash = row.get("output_text_sha256")
            stream_hash = row.get("stream_payload_sha256")
            if not (
                type(sequence) is int
                and 0 <= sequence < a6.SHAREGPT_REQUESTS
                and isinstance(output_hash, str)
                and HEX64.fullmatch(output_hash) is not None
                and isinstance(stream_hash, str)
                and HEX64.fullmatch(stream_hash) is not None
                and sequence not in measurement_hashes[cell]
            ):
                measurement_hashes_retained_exact = False
                continue
            measurement_hashes[cell][sequence] = (output_hash, stream_hash)
        metrics[cell] = a6._scenario_metric(cell_rows)

    generations = [value.get("server_generation") for value in provenance_values.values()]
    starts_ns = [value.get("server_started_ns") for value in provenance_values.values()]
    container_ids = [
        _container_id(starts[cell].get("provenance")) for cell in plan if cell in starts
    ]
    fresh_order_exact = (
        len(generations) == len(plan)
        and len(set(generations)) == len(plan)
        and all(isinstance(item, str) and item for item in generations)
        and len(container_ids) == len(plan)
        and len(set(container_ids)) == len(plan)
        and all(isinstance(item, str) and HEX64.fullmatch(item) for item in container_ids)
        and len(starts_ns) == len(plan)
        and all(type(item) is int for item in starts_ns)
        and all(int(starts_ns[index]) < int(starts_ns[index + 1]) for index in range(3))
    )
    common_identity_exact = set(provenance_values) == set(plan)
    if set(provenance_values) == set(plan):
        first = _provenance_static_identity(provenance_values[plan[0]])
        common_identity_exact &= all(
            _provenance_static_identity(value) == first for value in provenance_values.values()
        )
        for cell, value in provenance_values.items():
            config = value.get("configuration_source")
            config_raw = config.get("value") if isinstance(config, dict) else None
            common_identity_exact &= isinstance(config_raw, dict) and config_raw.get(
                "parsed"
            ) == expected_parsed_config(cell.arm)
    expected_sequences = set(range(a6.SHAREGPT_REQUESTS))
    measurement_hashes_retained_exact &= all(
        set(cell_hashes) == expected_sequences for cell_hashes in measurement_hashes.values()
    )
    expected_warmup_sequences = set(range(a6.WARMUP_REQUESTS))
    serial_warmup_output_parity &= all(
        set(cell_hashes) == expected_warmup_sequences
        for cell_hashes in serial_warmup_hashes.values()
    ) and all(
        len({serial_warmup_hashes[cell][sequence] for cell in plan}) == 1
        for sequence in range(a6.WARMUP_REQUESTS)
    )
    expected_requests = len(plan) * (
        a6.WARMUP_REQUESTS + a6.GRAPH_WARMUP_REQUESTS + a6.SHAREGPT_REQUESTS
    )
    counts_exact &= run_end.get("request_count") == expected_requests
    checks = {
        "diagnostic_marker_and_fresh_server_abba_plan_exact": exact_plan,
        "borrowed_a6_benchmark_and_trace_exact": benchmark_exact and trace_exact,
        "all_warmup_and_measurement_rows_retained_exactly_once": counts_exact,
        "strict_a6_sse_request_evidence_exact": request_evidence_exact,
        "every_cell_response_id_unique_across_all_requests": response_ids_unique,
        "serialized_sharegpt_warmup_output_hash_parity_across_all_four_cells": (
            serial_warmup_output_parity
        ),
        "serialized_graph_and_synchronized_burst_shape_exact": execution_shape_exact,
        "measurement_windows_replay_exact": measurement_windows_exact,
        "three_backend_and_process_attestations_per_cell_exact": attestation_exact,
        "process_tree_generation_identity_stable_across_three_attestations": (
            process_generation_stable
        ),
        "source_image_model_gpu_config_provenance_exact": provenance_exact,
        "logs_removal_and_post_cleanup_gpu_quiescence_exact": cleanup_exact,
        "fresh_unique_containers_and_chronological_abba_order_exact": fresh_order_exact,
        "backend_only_configuration_and_common_runtime_identity_exact": (common_identity_exact),
        "measurement_output_and_stream_digest_metadata_present_and_well_formed": (
            measurement_hashes_retained_exact
        ),
    }
    evidence_valid = all(checks.values())

    paired: list[dict[str, object]] = []
    goodput_ratios: list[Fraction] = []
    p50_ratios: list[Fraction] = []
    p99_ratios: list[Fraction] = []
    for repeat in range(PAIR_COUNT):
        inproc_cell = next(cell for cell in plan if cell.repeat == repeat and cell.arm == "kairyu")
        proc_cell = next(
            cell for cell in plan if cell.repeat == repeat and cell.arm == "kairyu-proc"
        )
        inproc = metrics.get(inproc_cell, {})
        proc = metrics.get(proc_cell, {})
        goodput = _goodput_ratio(proc, inproc)
        p50 = _ratio(proc.get("ttft_p50_ns"), inproc.get("ttft_p50_ns"))
        p99 = _ratio(proc.get("ttft_p99_ns"), inproc.get("ttft_p99_ns"))
        if goodput is not None:
            goodput_ratios.append(goodput)
        if p50 is not None:
            p50_ratios.append(p50)
        if p99 is not None:
            p99_ratios.append(p99)
        paired.append(
            {
                "repeat": repeat,
                "server_order": [cell.arm for cell in plan if cell.repeat == repeat],
                "kairyu": inproc,
                "kairyu_proc": proc,
                "kairyu_proc_over_kairyu": {
                    "goodput_rps": _fraction_record(goodput),
                    "ttft_p50_ns": _fraction_record(p50),
                    "ttft_p99_ns": _fraction_record(p99),
                },
            }
        )
    median_goodput = _median(goodput_ratios)
    median_p50 = _median(p50_ratios)
    median_p99 = _median(p99_ratios)
    report_only_classification = material_reduction_classification(
        median_p99,
        evidence_valid=evidence_valid,
    )
    comparisons = {
        "direction": "kairyu-proc divided by kairyu",
        "paired_rounds": paired,
        "paired_median": {
            "goodput_rps": _fraction_record(median_goodput),
            "ttft_p50_ns": _fraction_record(median_p50),
            "ttft_p99_ns": _fraction_record(median_p99),
        },
        "paired_mad": {
            "goodput_rps": _fraction_record(_mad(goodput_ratios)),
            "ttft_p50_ns": _fraction_record(_mad(p50_ratios)),
            "ttft_p99_ns": _fraction_record(_mad(p99_ratios)),
        },
        "report_only_material_reduction_classification": (report_only_classification),
        "completion_output_hash_agreement_diagnostic": (
            _completion_output_hash_agreement_diagnostic(plan, measurement_hashes)
        ),
    }
    diagnostics = {
        "scenario_metrics": {cell.scenario_id: metrics.get(cell, {}) for cell in plan},
        "method": {
            "diagnostic_only": True,
            "formal_g2_a6_gate": False,
            "performance_acceptance_threshold": None,
            "report_only_material_reduction_classification": {
                "metric": report_only_classification["metric"],
                "criterion": report_only_classification["criterion"],
                "binding": False,
                "predeclared_before_measurement": True,
                "conditional_on_all_integrity_checks": True,
                "material_interpretation": ("dominant process/GIL-contention hypothesis supported"),
            },
            "warmup_excluded_from_metrics": True,
            "goodput_definition": "TTFT<=10s successes / measurement span",
            "ttft_quantile": "nearest-rank over successful measurement requests",
            "outlier_removal": False,
            "retry": False,
            "causal_scope": (
                "net backend swap: process/GIL isolation plus ZMQ/msgpack/delta "
                "and lifecycle overhead; not a pure GIL attribution"
            ),
        },
    }
    return checks, comparisons, diagnostics


def assemble_manifest(
    rows: Sequence[dict[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    if HEX64.fullmatch(raw_sha256) is None:
        raise ValueError("manifest requires a raw SHA-256")
    checks, comparisons, diagnostics = evaluate_rows(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": DIAGNOSTIC,
        "diagnostic_only": True,
        "a6_gate_claimed": False,
        "external_workdir_logs_required": rows[0].get("external_workdir_logs_required"),
        "performance_acceptance_threshold": None,
        "raw": {
            "name": RAW_NAME,
            "sha256": raw_sha256,
            "row_count": len(rows),
        },
        "checks": checks,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "evidence_valid": all(checks.values()),
    }


def write_artifact(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    _write_jsonl(raw_path, rows)
    stored = _read_jsonl(raw_path)
    if stored[0].get("external_workdir_logs_required") is True:
        validate_external_workdir_logs(stored, work_dir=output_dir.parent)
    manifest = assemble_manifest(stored, raw_sha256=sha256_file(raw_path))
    atomic_write_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def _artifact_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / RAW_NAME, path / MANIFEST_NAME
    if path.name == RAW_NAME:
        return path, path.with_name(MANIFEST_NAME)
    if path.name == MANIFEST_NAME:
        return path.with_name(RAW_NAME), path
    raise ValueError(f"artifact must be a directory, {RAW_NAME}, or {MANIFEST_NAME}")


def validate_external_workdir_logs(
    rows: Sequence[dict[str, object]],
    *,
    work_dir: Path,
) -> None:
    run, _, _, ends, _ = _artifact_rows(rows)
    if run.get("external_workdir_logs_required") is not True:
        return
    resolved_work_dir = work_dir.resolve()
    for cell in diagnostic_plan():
        cleanup = ends.get(cell, {}).get("cleanup")
        raw = cleanup.get("value") if isinstance(cleanup, dict) else None
        log = raw.get("log") if isinstance(raw, dict) else None
        if not isinstance(log, dict) or not isinstance(log.get("path"), str):
            raise ValueError(f"{cell.scenario_id} cleanup log binding is absent")
        log_path = (resolved_work_dir / str(log["path"])).resolve()
        if not log_path.is_relative_to(resolved_work_dir) or not log_path.is_file():
            raise ValueError(f"{cell.scenario_id} retained Docker log is absent")
        if log_path.stat().st_size != log.get("bytes") or sha256_file(log_path) != log.get(
            "sha256"
        ):
            raise ValueError(f"{cell.scenario_id} retained Docker log differs from cleanup binding")


def verify_artifact(
    path: Path,
    *,
    assert_integrity: bool = False,
) -> dict[str, object]:
    raw_path, manifest_path = _artifact_paths(path)
    rows = _read_jsonl(raw_path)
    manifest = _read_json_object(manifest_path)
    raw_sha = sha256_file(raw_path)
    raw = manifest.get("raw")
    if not isinstance(raw, dict) or raw.get("sha256") != raw_sha:
        raise ValueError("raw SHA-256 does not match issue #333 manifest")
    replayed = assemble_manifest(rows, raw_sha256=raw_sha)
    if manifest != replayed:
        raise ValueError("issue #333 manifest differs from independent raw replay")
    if replayed.get("external_workdir_logs_required") is True:
        validate_external_workdir_logs(rows, work_dir=raw_path.parent.parent)
    if assert_integrity and replayed.get("evidence_valid") is not True:
        failed = [key for key, value in replayed["checks"].items() if not value]
        raise RuntimeError(f"issue #333 evidence integrity failed: {failed}")
    return replayed


def _validate_paths(
    args: argparse.Namespace,
    *,
    regenerate_trace: bool,
) -> tuple[dict[str, object], dict[str, Any], dict[str, str]]:
    if not args.repo.is_dir():
        raise DiagnosticRunError(f"missing repository: {args.repo}")
    if not args.work_dir.is_relative_to(Path("/tmp")):
        raise DiagnosticRunError("--work-dir must resolve below /tmp")
    for label, path in (
        ("dataset", args.dataset),
        ("tokenizer", args.tokenizer),
        ("trace bundle", args.trace_bundle),
        ("environment artifact", args.env_artifact),
        ("Kairyu template", args.kairyu_template),
    ):
        if not path.is_file():
            raise DiagnosticRunError(f"missing {label}: {path}")
    if sha256_file(args.dataset) != a6.SHAREGPT_DATASET_SHA256:
        raise DiagnosticRunError("ShareGPT dataset SHA-256 differs from A6 pin")
    tokenizer_file = (
        args.tokenizer if args.tokenizer.is_file() else args.tokenizer / "tokenizer.json"
    )
    if sha256_file(tokenizer_file) != a6.PINNED_TOKENIZER_SHA256:
        raise DiagnosticRunError("tokenizer.json SHA-256 differs from A6 pin")
    trace_bundle = a6.load_trace_bundle(
        args.trace_bundle,
        dataset_sha256=a6.SHAREGPT_DATASET_SHA256,
    )
    if regenerate_trace:
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
            raise DiagnosticRunError("retained trace bundle differs from exact A6 regeneration")
    parsed_env = formal.parse_environment(args.env_artifact)
    # Fail before any live server if the retained environment artifact is not
    # commit-bound under the clean source tree.
    formal.environment_value(
        repo=args.repo,
        path=args.env_artifact,
        parsed=parsed_env,
    )
    template = args.kairyu_template.read_text(encoding="utf-8")
    rendered_configs = {arm: render_kairyu_config(template, arm=arm) for arm in ARMS}
    return trace_bundle, parsed_env, rendered_configs


def _make_provenance(
    *,
    args: argparse.Namespace,
    cell: Cell,
    session_id: str,
    repo_commit: str,
    source_hashes: dict[str, str],
    runner_hashes: dict[str, str],
    parsed_env: dict[str, Any],
    inventory: Sequence[formal.GPU],
    checkpoint: dict[str, object],
    image_id: str,
    runtime_versions: dict[str, object],
    server_generation: str,
    server_started_ns: int,
    config_descriptor: dict[str, object],
    launch_descriptor: dict[str, object],
    container_gpus: dict[str, list[str]],
    gpu_baseline: dict[str, object],
    quiescence: dict[str, object],
) -> dict[str, object]:
    selected = list(inventory[:TP_SIZE])
    return hashed_descriptor(
        {
            "schema_version": 1,
            "diagnostic_only": True,
            "session_id": session_id,
            "server_generation": server_generation,
            "server_started_ns": server_started_ns,
            "source": {
                "repo_commit": repo_commit,
                "source_clean": True,
                "critical_file_sha256": source_hashes,
                "runner_file_sha256": runner_hashes,
            },
            "host": {
                "id": socket.gethostname(),
                "environment": _fetch_environment_value(
                    repo=args.repo,
                    path=args.env_artifact,
                    parsed=parsed_env,
                ),
                "run_start_gpu_baseline": gpu_baseline,
                "pre_start_gpu_quiescence": quiescence,
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
            "checkpoint": checkpoint,
            "image": {"ref": args.kairyu_image, "id": image_id},
            "runtime_versions": runtime_versions,
            "container_launch": launch_descriptor,
            "configuration_source": config_descriptor,
            "diagnostic_arm": cell.arm,
        }
    )


def _run_cell(
    *,
    args: argparse.Namespace,
    cell: Cell,
    session_id: str,
    repo_commit: str,
    source_hashes: dict[str, str],
    runner_hashes: dict[str, str],
    trace_bundle: dict[str, object],
    parsed_env: dict[str, Any],
    inventory: Sequence[formal.GPU],
    checkpoint: dict[str, object],
    image_id: str,
    runtime_versions: dict[str, object],
    rendered_configs: Mapping[str, str],
    gpu_baseline: dict[str, object],
    previous_started_ns: int,
) -> tuple[Path, int]:
    if not _port_available(args.port):
        raise DiagnosticRunError(f"host port {args.port} is already in use")
    memory_baseline = _quiescence_memory_baseline(gpu_baseline)
    if memory_baseline is None:
        raise DiagnosticRunError("run-start GPU memory baseline is invalid")
    quiescence = _wait_for_selected_gpu_quiescence(
        inventory,
        memory_baseline_mib=memory_baseline,
    )
    formal.assert_source_identity(args.repo, repo_commit)
    config_text = rendered_configs[cell.arm]
    config_descriptor = configuration_source(config_text, arm=cell.arm)
    config_path = args.work_dir / "configs" / f"{cell.key}.yaml"
    atomic_write_text(config_path, config_text)
    name = _container_name(session_id, cell)
    if not _container_absent(name):
        raise DiagnosticRunError(f"refusing to reuse or remove pre-existing container {name!r}")
    server_started_ns = max(time.time_ns(), previous_started_ns + 1)
    server_generation = str(uuid.uuid4())
    log_path = args.work_dir / "logs" / f"{cell.key}.log"
    raw_path = args.work_dir / "raw" / f"{cell.key}.jsonl"
    sidecar_path = raw_path.with_suffix(".summary.json")
    command = docker_command(
        args=args,
        name=name,
        config_path=config_path,
        image_id=image_id,
    )
    endpoint = f"http://127.0.0.1:{args.port}"
    started = False
    launch_attempted = False
    primary_error: BaseException | None = None
    provenance: dict[str, object] | None = None
    backend_ready: dict[str, object] | None = None
    process_ready: dict[str, object] | None = None
    requests: list[dict[str, object]] | None = None
    end_evidence: dict[str, object] | None = None
    launched_container_id: str | None = None
    try:
        launch_attempted = True
        launched = formal.run_command(command, timeout_s=DOCKER_LAUNCH_TIMEOUT_S)
        launched_container_id = (launched.stdout or "").strip()
        if HEX64.fullmatch(launched_container_id) is None:
            raise DiagnosticRunError("docker run did not return a full container identity")
        started = True
        formal.wait_for_health(
            name=name,
            endpoint=endpoint,
            model=a6.MODEL,
            timeout_s=args.startup_timeout_s,
            command_timeout_s=DOCKER_CONTROL_TIMEOUT_S,
        )
        launch = container_launch_attestation(
            args=args,
            name=name,
            cell=cell,
            config_path=config_path,
            config_descriptor=config_descriptor,
            image_id=image_id,
            source_hashes=source_hashes,
        )
        launch_raw = launch.get("value")
        if not isinstance(launch_raw, dict) or not isinstance(
            launch_raw.get("container_id"), str
        ):
            raise DiagnosticRunError("container launch omitted its immutable identity")
        if launch_raw["container_id"] != launched_container_id:
            raise DiagnosticRunError("live launch identity differs from docker run identity")
        container_gpus = _fetch_container_gpu_attestation(
            name,
            inventory[:TP_SIZE],
        )
        observed_runtime = formal.container_runtime_versions(
            name,
            "kairyu",
            command_timeout_s=DOCKER_ATTEST_TIMEOUT_S,
        )
        if observed_runtime != runtime_versions:
            raise DiagnosticRunError("container runtime changed between TP4 cells")
        backend_ready = attest_backend(endpoint, arm=cell.arm)
        process_ready = process_tree_attestation(name, arm=cell.arm)
        provenance = _make_provenance(
            args=args,
            cell=cell,
            session_id=session_id,
            repo_commit=repo_commit,
            source_hashes=source_hashes,
            runner_hashes=runner_hashes,
            parsed_env=parsed_env,
            inventory=inventory,
            checkpoint=checkpoint,
            image_id=image_id,
            runtime_versions=runtime_versions,
            server_generation=server_generation,
            server_started_ns=server_started_ns,
            config_descriptor=config_descriptor,
            launch_descriptor=launch,
            container_gpus=container_gpus,
            gpu_baseline=gpu_baseline,
            quiescence=quiescence,
        )
        requests, end_evidence = asyncio.run(
            measure_cell_requests(
                endpoint=endpoint,
                trace_bundle=trace_bundle,
                request_timeout_s=args.request_timeout_s,
                api_key=args.api_key,
                probe_backend=lambda: attest_backend(endpoint, arm=cell.arm),
                probe_process_tree=lambda: process_tree_attestation(name, arm=cell.arm),
            )
        )
    except BaseException as error:
        primary_error = error

    cleanup_errors: list[BaseException] = []
    log_record: dict[str, object] | None = None
    removal_record: dict[str, object] | None = None
    post_cleanup: dict[str, object] | None = None
    if launch_attempted:
        log_record, removal_record, post_cleanup, attempted_errors = _attempt_container_cleanup(
            name=name,
            log_path=log_path,
            work_dir=args.work_dir,
            inventory=inventory,
            expected_container_id=launched_container_id,
            memory_baseline_mib=memory_baseline,
        )
        cleanup_errors.extend(attempted_errors)

    if primary_error is not None or cleanup_errors:
        errors = ([primary_error] if primary_error is not None else []) + cleanup_errors
        if len(errors) == 1:
            raise errors[0]
        if all(isinstance(error, Exception) for error in errors):
            raise ExceptionGroup(
                f"issue #333 cell {cell.scenario_id} failed with cleanup errors",
                [error for error in errors if isinstance(error, Exception)],
            )
        raise BaseExceptionGroup(
            f"issue #333 cell {cell.scenario_id} failed with cleanup errors",
            errors,
        )
    if not (
        started
        and isinstance(provenance, dict)
        and isinstance(backend_ready, dict)
        and isinstance(process_ready, dict)
        and isinstance(requests, list)
        and isinstance(end_evidence, dict)
        and isinstance(log_record, dict)
        and isinstance(removal_record, dict)
        and isinstance(post_cleanup, dict)
    ):
        raise DiagnosticRunError(f"issue #333 cell {cell.scenario_id} retained incomplete evidence")
    formal.assert_source_identity(args.repo, repo_commit)
    if source_file_hashes(args.repo) != source_hashes:
        raise DiagnosticRunError("critical source bytes changed during issue #333 cell")
    if runner_source_hashes(args.repo) != runner_hashes:
        raise DiagnosticRunError("operator source bytes changed during issue #333 cell")
    cleanup = cleanup_attestation(
        cell=cell,
        provenance=provenance,
        log=log_record,
        removal=removal_record,
        post_cleanup_quiescence=post_cleanup,
    )
    rows = measurement_rows(
        cell=cell,
        trace_bundle=trace_bundle,
        provenance=provenance,
        cleanup=cleanup,
        backend_ready=backend_ready,
        process_tree_ready=process_ready,
        requests=requests,
        end_evidence=end_evidence,
    )
    summary = write_measurement_shard(raw_path, rows)
    atomic_write_json(sidecar_path, summary)
    return raw_path, server_started_ns


def _run(args: argparse.Namespace) -> dict[str, object]:
    validate_a6_contract()
    if not args.repo.is_dir():
        raise DiagnosticRunError(f"missing repository: {args.repo}")
    repo_commit, clean = formal.source_identity(args.repo, require_clean=True)
    if not clean or HEX40.fullmatch(repo_commit) is None:
        raise DiagnosticRunError("live diagnostic requires a clean committed repo")
    runner_hashes = runner_source_hashes(args.repo)
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise DiagnosticRunError(
            "--work-dir is non-empty; issue #333 ABBA is intentionally non-resumable"
        )
    trace_bundle, parsed_env, rendered_configs = _validate_paths(
        args,
        regenerate_trace=True,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    inventory = formal.live_gpu_inventory()
    formal.validate_inventory_against_environment(inventory, parsed_env)
    formal.docker_volume_exists(args.model_volume)
    image_id = formal.docker_image_id(args.kairyu_image)
    checkpoint = formal.checkpoint_attestation(
        a6=a6,
        model_volume=args.model_volume,
        image=image_id,
    )
    runtime_versions = formal.image_runtime_versions(image_id, "kairyu")
    formal.validate_runtime_versions(
        observed=runtime_versions,
        expected={
            "cuda_version": a6.KAIRYU_CUDA_VERSION,
            "nccl_version": a6.KAIRYU_NCCL_VERSION,
        },
        arm="kairyu",
    )
    packages = runtime_versions["package_versions"]
    assert isinstance(packages, dict)
    expected_packages = {
        "kairyu": formal.project_version(args.repo),
        "uvloop": a6.UVLOOP_VERSION,
        "httptools": a6.HTTPTOOLS_VERSION,
        "uvicorn": a6.KAIRYU_UVICORN_VERSION,
    }
    observed_packages = {key: packages.get(key) for key in expected_packages}
    if observed_packages != expected_packages:
        raise DiagnosticRunError(
            "Kairyu/HTTP runtime differs from the A6 pin: "
            f"observed={observed_packages}, expected={expected_packages}"
        )
    source_hashes = source_file_hashes(args.repo)
    gpu_baseline = _wait_for_selected_gpu_quiescence(inventory)
    memory_baseline = _quiescence_memory_baseline(gpu_baseline)
    if memory_baseline is None:
        raise DiagnosticRunError("cannot retain the run-start GPU memory baseline")
    session_id = args.session_id or f"issue333-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    state = {
        "schema_version": RUNNER_SCHEMA,
        "session_id": session_id,
        "repo_commit": repo_commit,
        "source_clean": True,
        "plan": [cell.identity() for cell in diagnostic_plan()],
        "trace_bundle": {
            "path": str(args.trace_bundle),
            "sha256": sha256_file(args.trace_bundle),
        },
        "dataset": {"path": str(args.dataset), "sha256": sha256_file(args.dataset)},
        "tokenizer": {
            "path": str(args.tokenizer),
            "sha256": sha256_file(args.tokenizer),
        },
        "environment": {
            "path": str(args.env_artifact.relative_to(args.repo)),
            "sha256": sha256_file(args.env_artifact),
        },
        "image": {"ref": args.kairyu_image, "id": image_id},
        "model_volume": args.model_volume,
        "checkpoint": checkpoint,
        "runtime_versions": runtime_versions,
        "run_start_gpu_baseline": gpu_baseline,
        "source_files": source_hashes,
        "runner_files": runner_hashes,
    }
    atomic_write_json(args.work_dir / STATE_NAME, state)
    raw_paths: list[Path] = []
    previous_started_ns = 0
    for index, cell in enumerate(diagnostic_plan(), 1):
        print(f"[{index}/4] fresh-server {cell.scenario_id}", flush=True)
        raw, previous_started_ns = _run_cell(
            args=args,
            cell=cell,
            session_id=session_id,
            repo_commit=repo_commit,
            source_hashes=source_hashes,
            runner_hashes=runner_hashes,
            trace_bundle=trace_bundle,
            parsed_env=parsed_env,
            inventory=inventory,
            checkpoint=checkpoint,
            image_id=image_id,
            runtime_versions=runtime_versions,
            rendered_configs=rendered_configs,
            gpu_baseline=gpu_baseline,
            previous_started_ns=previous_started_ns,
        )
        raw_paths.append(raw)
    formal.assert_source_identity(args.repo, repo_commit)
    checkpoint_after = formal.checkpoint_attestation(
        a6=a6,
        model_volume=args.model_volume,
        image=image_id,
    )
    if checkpoint_after != checkpoint:
        raise DiagnosticRunError("model checkpoint changed during issue #333 ABBA")
    rows = assemble_rows(raw_paths, external_workdir_logs_required=True)
    rows[0]["session_id"] = session_id
    manifest = write_artifact(args.work_dir / "artifact", rows)
    verify_artifact(
        args.work_dir / "artifact",
        assert_integrity=args.assert_integrity,
    )
    completed = {
        "schema_version": RUNNER_SCHEMA,
        "diagnostic_only": True,
        "session_id": session_id,
        "repo_commit": repo_commit,
        "manifest_sha256": sha256_file(args.work_dir / "artifact" / MANIFEST_NAME),
        "completed_at_ns": time.time_ns(),
        "evidence_valid": manifest["evidence_valid"],
    }
    atomic_write_json(args.work_dir / "completed.json", completed)
    return manifest


def _dry_run(args: argparse.Namespace) -> None:
    validate_a6_contract()
    _, _, rendered_configs = _validate_paths(args, regenerate_trace=False)
    commit, clean = formal.source_identity(args.repo, require_clean=False)
    image_id = "sha256:" + "1" * 64
    session = args.session_id or "dry-run"
    for cell in diagnostic_plan():
        config_path = args.work_dir / "configs" / f"{cell.key}.yaml"
        command = docker_command(
            args=args,
            name=_container_name(session, cell),
            config_path=config_path,
            image_id=image_id,
        )
        config_hash = sha256_bytes(rendered_configs[cell.arm].encode())
        print(
            f"DRY {cell.order_index + 1}/4 {cell.scenario_id}\n"
            f"  config_sha256: {config_hash}\n"
            f"  server: {formal.command_text(command)}"
        )
    print(
        canonical_json(
            {
                "diagnostic_only": True,
                "a6_gate_claimed": False,
                "repo_commit": commit,
                "source_clean": clean,
                "actual_run_requires_clean": True,
                "plan": list(ABBA_ORDER),
            }
        )
    )


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--trace-bundle", type=Path, default=DEFAULT_TRACE_BUNDLE)
    parser.add_argument("--env-artifact", type=Path, default=DEFAULT_ENV_ARTIFACT)
    parser.add_argument("--kairyu-template", type=Path, default=DEFAULT_KAIRYU_TEMPLATE)
    parser.add_argument("--kairyu-image", default=DEFAULT_IMAGE)
    parser.add_argument("--model-volume", default=DEFAULT_MODEL_VOLUME)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--session-id")
    parser.add_argument("--api-key")
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the exact ABBA launch plan without writes/Docker",
    )
    parser.add_argument(
        "--assert-integrity",
        action="store_true",
        help="fail unless all evidence-integrity checks pass; not a performance gate",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run four fresh TP4 servers in ABBA order")
    _add_common_run_arguments(run_parser)
    assemble_parser = subparsers.add_parser(
        "assemble",
        help="assemble the exact four raw shards into diagnostic evidence",
    )
    assemble_parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        required=True,
        help="one issue #333 measurement shard; repeat exactly four times",
    )
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    assemble_parser.add_argument("--session-id", required=True)
    assemble_parser.add_argument("--assert-integrity", action="store_true")
    verify_parser = subparsers.add_parser(
        "verify",
        help="independently replay raw evidence and verify the retained manifest",
    )
    verify_parser.add_argument("--artifact", type=Path, required=True)
    verify_parser.add_argument("--assert-integrity", action="store_true")
    return parser


def _resolve_run_args(args: argparse.Namespace) -> None:
    args.repo = args.repo.resolve()
    args.work_dir = args.work_dir.resolve()
    args.dataset = args.dataset.resolve()
    args.tokenizer = args.tokenizer.resolve()
    args.trace_bundle = args.trace_bundle.resolve()
    args.env_artifact = args.env_artifact.resolve()
    args.kairyu_template = args.kairyu_template.resolve()
    if not 1 <= args.port <= 65535:
        raise DiagnosticRunError("--port must be in 1..65535")
    if args.startup_timeout_s <= 0 or args.request_timeout_s <= 0:
        raise DiagnosticRunError("timeouts must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        _resolve_run_args(args)
        if args.dry_run:
            _dry_run(args)
            return 0
        manifest = _run(args)
        print(canonical_json(manifest))
        return 0
    if args.command == "assemble":
        rows = assemble_rows(tuple(path.resolve() for path in args.raw))
        rows[0]["session_id"] = args.session_id
        manifest = write_artifact(args.output_dir.resolve(), rows)
        print(canonical_json(manifest))
        if args.assert_integrity and manifest["evidence_valid"] is not True:
            failed = [key for key, value in manifest["checks"].items() if not value]
            raise RuntimeError(f"issue #333 evidence integrity failed: {failed}")
        return 0
    result = verify_artifact(
        args.artifact.resolve(),
        assert_integrity=args.assert_integrity,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
