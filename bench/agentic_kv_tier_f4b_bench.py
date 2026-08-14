#!/usr/bin/env python3
"""Formal agentic multi-turn DRAM-KV evidence for G5 F4b / issue #188.

Four fresh TP4 containers run one shard each, sequentially::

    round 0: tier off on cohort A, then tier on on cohort B
    round 1: tier on on cohort A, then tier off on cohort B

Every request is driven through the production ``EngineLoop`` owned by a
``KairyuBackend``.  The benchmark deliberately calls ``step()`` synchronously
instead of consuming ``KairyuBackend.stream()``: the asynchronous pump may run
ahead of its consumer, so consumer-side cache snapshots do not identify token
boundaries.  Here every step (including update-free prefill steps) records the
clock, cumulative output, free HBM pages, and DRAM-tier counters immediately
after that production step on the same thread.

``assemble`` canonicalizes four raw shards and derives a manifest. ``replay``
derives the same manifest from only the combined raw JSONL, while ``verify``
also checks the retained raw digest and manifest.  All evidence checks are
fail-closed; diagnostics never turn a failed gate into a pass.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence import (  # noqa: E402
    artifact_paths as evidence_artifact_paths,
)
from evidence import (  # noqa: E402
    canonical_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
    verify_replayed_manifest,
)
from evidence import (  # noqa: E402
    read_jsonl as evidence_read_jsonl,
)
from evidence import (  # noqa: E402
    read_jsonl_with_sha256 as evidence_read_jsonl_with_sha256,
)
from evidence import (  # noqa: E402
    replay_manifest as evidence_replay_manifest,
)
from evidence import (  # noqa: E402
    write_jsonl as evidence_write_jsonl,
)

SCHEMA_VERSION = "kairyu.g5.f4b.agentic-kv-tier.v1"
VERDICT_REVISION = "kairyu.g5.f4b.agentic-kv-tier-verdict.v2"
PROFILE_SCHEMA_VERSION = "kairyu.dram-kv-crossover.v2"
MEASUREMENT_KIND = "qwen3-32b-agentic-multiturn-dram-kv-tier"
SOURCE_PATH = "bench/agentic_kv_tier_f4b_bench.py"
RAW_NAME = "agentic-kv-tier-f4b-raw.jsonl"
MANIFEST_NAME = "agentic-kv-tier-f4b-manifest.json"
QUALITY_SCHEMA_VERSION = "kairyu.g5.f4b.agentic-kv-tier-quality.v1"
QUALITY_MEASUREMENT_KIND = f"{MEASUREMENT_KIND}-distribution-quality"
QUALITY_RAW_NAME = "agentic-kv-tier-f4b-quality-raw.jsonl"
QUALITY_MANIFEST_NAME = "agentic-kv-tier-f4b-quality-manifest.json"
CONTAINER_METADATA_DIR = "container-metadata"
QUALITY_CONTAINER_METADATA_DIR = "quality-container-metadata"
IMAGE_INSPECT_NAME = "image-inspect.json"
CONTAINER_ID_NAME = "container-id"
CONTAINER_CREATED_NAME = "container-inspect-created.json"
CONTAINER_EXITED_NAME = "container-inspect-exited.json"

MODEL = "qwen3-32b"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
EXPECTED_MODEL_WEIGHTS_ROLLUP = "6aa37b7da4e37d45b277d0bca47b00c2bfb58bb60986ba93e4c3e667df123955"
EXPECTED_MODEL_CONFIG_SHA256 = "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
F4A_PROFILE_FILE_SHA256 = "7b73d4adb2cc6a89ec41d0ad9e36ce788ce761ca18796b56a31e0f73d97b041d"
F4A_PROFILE_EVIDENCE_SHA256 = "c8a15902923a16ce6f38655843083d8c47c1df3afc49846ee9108235968fa992"
F4A_PROFILE_RUNTIME_FINGERPRINT = "4ba5dc9267fa4442d5545b8c889720bd77df35817e3ff3c03e2e4bd470b62dcb"
F4A_CALIBRATED_IMAGE_ID = (
    "sha256:25543ae9cbc9d2e80f1b4be2193d138486adb91c89a02cdbd0be0e62a1cc67be"
)
F4A_CALIBRATED_IMAGE_REVISION = "edd535f7018695fc03c479a86fbd690174cca5ef"
TP_SIZE = 4
PAGE_SIZE = 16
HBM_PAGES = 1024
DRAM_PAGES = 2048
MAX_NUM_BATCHED_TOKENS = 2048
MAX_MODEL_LEN = 8192
SESSIONS = 16
TURNS = 8
SHARED_TOKENS = 2048
TURN_TOKENS = 512
OUTPUT_TOKENS = 32
REQUESTS_PER_SHARD = SESSIONS * TURNS
TPOT_RATIO_LIMIT = 1.10
QUALITY_TOP_LOGPROBS = 64
QUALITY_LOGPROB_ABS_TOLERANCE = 0.25
QUALITY_PLAN = (("off", "A"), ("on", "A"))
QUALITY_SHARD_LABELS = {
    ("off", "A"): "quality-off",
    ("on", "A"): "quality-on",
}
TIER_STATS_FIELDS = (
    "offload_pages",
    "offload_bypassed_pages",
    "restore_pages",
    "restore_attempts",
    "restore_fallbacks",
    "ownership_failures",
)
RUNTIME_ARM_FIELDS = frozenset(
    {
        "arm",
        "dram_kv_tier_enabled",
        "dram_kv_tier_capacity_pages",
        "dram_kv_tier_profile_sha256",
        "dram_kv_tier_min_restore_tokens",
    }
)
RUNTIME_COMMON_FIELDS = frozenset(
    {
        "tensor_parallel_size",
        "page_size",
        "num_pages",
        "max_num_batched_tokens",
        "max_model_len",
        "pipeline_depth",
        "decode_mode",
        "engine_source_sha256",
        "attention_backend_decision",
        "kv_cache_dtype_requested",
        "kv_cache_dtype_resolved",
    }
)
SHARD_PLAN = (
    (0, "off", "A"),
    (0, "on", "B"),
    (1, "on", "A"),
    (1, "off", "B"),
)
SHARD_LABELS = {
    (0, "off", "A"): "round0-off",
    (0, "on", "B"): "round0-on",
    (1, "on", "A"): "round1-on",
    (1, "off", "B"): "round1-off",
}
COHORT_DEVICE_IDS = {
    "A": ("0", "1", "2", "3"),
    "B": ("4", "5", "6", "7"),
}
_DOCKER_ZERO_TIMESTAMP = "0001-01-01T00:00:00Z"
_RFC3339_NS = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?P<offset>Z|[+-]\d{2}:\d{2})$"
)
REQUIRED_SOURCE_PATHS = frozenset(
    {
        SOURCE_PATH,
        "Dockerfile.cuda",
        "pyproject.toml",
        "uv.lock",
        "kairyu/engine/engine_loop.py",
        "kairyu/engine/kairyu_backend.py",
        "kairyu/engine/core/kv_tier.py",
        "kairyu/engine/core/kv_tier_policy.py",
        "kairyu/engine/core/radix_kv.py",
        "kairyu/engine/core/scheduler.py",
        "kairyu/engine/core/worker.py",
    }
)
_HEX = frozenset("0123456789abcdef")


class EvidenceError(ValueError):
    """Raw or retained F4b evidence violates the formal contract."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def hashed_descriptor(value: object) -> dict[str, object]:
    return {"value": value, "sha256": sha256_json(value)}


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


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} is invalid")
    return int(value)


def _descriptor(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} descriptor is not an object")
    result = dict(value)
    _require(set(result) == {"value", "sha256"}, f"{name} descriptor fields differ")
    _require(_valid_sha256(result["sha256"]), f"{name} descriptor digest is invalid")
    _require(result["sha256"] == sha256_json(result["value"]), f"{name} descriptor digest differs")
    return result


def formal_config() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": MEASUREMENT_KIND,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "tensor_parallel_size": TP_SIZE,
        "page_size": PAGE_SIZE,
        "hbm_pages": HBM_PAGES,
        "dram_pages_when_enabled": DRAM_PAGES,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "pipeline_depth": 1,
        "decode_mode": "eager",
        "sessions": SESSIONS,
        "turns_per_session": TURNS,
        "request_order": "turn-major then session-major",
        "shared_prefix_tokens": SHARED_TOKENS,
        "canonical_agent_tool_tokens_per_turn": TURN_TOKENS,
        "prompt_tokens_by_turn": [
            SHARED_TOKENS + (turn + 1) * TURN_TOKENS for turn in range(TURNS)
        ],
        "sampling": {
            "temperature": 0.0,
            "min_tokens": OUTPUT_TOKENS,
            "max_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "seed": 0,
        },
        "primary_cache_metric": "sum(engine cached_tokens) / sum(engine prompt_tokens)",
        "tpot_definition": "(terminal_elapsed_ns - first_content_elapsed_ns) / 31",
        "p99_definition": "nearest-rank ceil(0.99*n)",
        "tpot_noninferiority_ratio_limit": TPOT_RATIO_LIMIT,
        "shard_plan": [
            {"round": round_index, "tier": arm, "cohort": cohort}
            for round_index, arm, cohort in SHARD_PLAN
        ],
    }


def quality_config() -> dict[str, object]:
    config = formal_config()
    return {
        **config,
        "schema_version": QUALITY_SCHEMA_VERSION,
        "measurement_kind": QUALITY_MEASUREMENT_KIND,
        "sampling": {
            **config["sampling"],
            "logprobs": QUALITY_TOP_LOGPROBS,
        },
        "timing_is_binding": False,
        "logprob_abs_tolerance": QUALITY_LOGPROB_ABS_TOLERANCE,
        "quality_plan": [
            {"tier": arm, "cohort": cohort}
            for arm, cohort in QUALITY_PLAN
        ],
    }


def _request_identity(sequence: int) -> tuple[int, int]:
    return sequence % SESSIONS, sequence // SESSIONS


def _trace_rows_from_requests(
    requests: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "sequence": row["sequence"],
            "session": row["session"],
            "turn": row["turn"],
            "prompt_tokens": row["prompt_tokens"],
            "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
        }
        for row in requests
    ]


def _exact_token_block(tokenizer: object, text: str, length: int) -> tuple[int, ...]:
    encoded = tuple(int(token) for token in tokenizer.encode(text))
    _require(encoded and len(encoded) <= length, "canonical trace text exceeds its token block")
    filler = tuple(
        int(token)
        for token in tokenizer.encode(
            " deterministic evidence filler preserving an exact token geometry."
        )
    )
    _require(filler, "tokenizer produced no deterministic filler tokens")
    needed = length - len(encoded)
    return encoded + (filler * ((needed + len(filler) - 1) // len(filler)))[:needed]


def build_trace(
    tokenizer: object, *, tokenizer_sha256: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Create the fixed page-aligned 16-session, eight-turn token trace."""

    shared = _exact_token_block(
        tokenizer,
        (
            "<|im_start|>system\nYou are a production operations agent. Inspect the "
            "service state, choose tools deliberately, verify every mutation, and "
            "summarize evidence. Available tools: inspect_metrics, query_logs, "
            "read_config, apply_change, verify_health.<|im_end|>\n"
        ),
        SHARED_TOKENS,
    )
    segments: dict[tuple[int, int], tuple[int, ...]] = {}
    for session in range(SESSIONS):
        for turn in range(TURNS):
            segments[(session, turn)] = _exact_token_block(
                tokenizer,
                (
                    f"<|im_start|>user\nSession {session}, turn {turn}: diagnose the "
                    "next bounded production condition and cite the observation."
                    "<|im_end|>\n<|im_start|>assistant\n"
                    f'<tool_call>{{"name":"inspect_metrics","arguments":{{"session":{session},"turn":{turn}}}}}</tool_call>'
                    "<|im_end|>\n<|im_start|>tool\n"
                    f'{{"session":{session},"turn":{turn},"healthy":true,"action":"continue"}}'
                    "<|im_end|>\n"
                ),
                TURN_TOKENS,
            )
    requests: list[dict[str, object]] = []
    for sequence in range(REQUESTS_PER_SHARD):
        session, turn = _request_identity(sequence)
        prompt = shared + tuple(
            token for prior_turn in range(turn + 1) for token in segments[(session, prior_turn)]
        )
        requests.append(
            {
                "sequence": sequence,
                "session": session,
                "turn": turn,
                "prompt_tokens": len(prompt),
                "prompt_token_ids": list(prompt),
                "prompt_token_ids_sha256": sha256_json(list(prompt)),
            }
        )
    value = {
        "kind": "canonical-agent-tool-token-trace-v1",
        "tokenizer_sha256": tokenizer_sha256,
        "shared_prefix_tokens": SHARED_TOKENS,
        "turn_tokens": TURN_TOKENS,
        "shared_prefix_sha256": sha256_json(list(shared)),
        "rows": _trace_rows_from_requests(requests),
    }
    return hashed_descriptor(value), requests


def _git_text(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(completed.returncode == 0, f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _source_provenance_live() -> dict[str, object]:
    commit = _git_text("rev-parse", "HEAD")
    dirty = _git_text("status", "--porcelain", "--untracked-files=no")
    _require(_valid_commit(commit), "live source commit is invalid")
    _require(not dirty, "formal run requires a clean tracked worktree")
    python_paths = tuple(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in sorted((_REPO_ROOT / "kairyu").rglob("*.py"))
    )
    bound_paths = tuple(
        dict.fromkeys((SOURCE_PATH, "Dockerfile.cuda", "pyproject.toml", "uv.lock", *python_paths))
    )
    _require(REQUIRED_SOURCE_PATHS <= set(bound_paths), "source inventory omits F4b implementation")
    bound: dict[str, str] = {}
    for relative in bound_paths:
        path = _REPO_ROOT / relative
        _require(path.is_file(), f"bound source is missing: {relative}")
        payload = path.read_bytes()
        committed = subprocess.run(
            ("git", "show", f"{commit}:{relative}"),
            cwd=_REPO_ROOT,
            capture_output=True,
            check=False,
        )
        _require(
            committed.returncode == 0 and committed.stdout == payload,
            f"bound source differs from commit: {relative}",
        )
        bound[relative] = sha256_bytes(payload)
    kairyu_bound = {path: bound[path] for path in python_paths}
    return {
        "commit": commit,
        "dirty": False,
        "source_kind": "git",
        "script_path": SOURCE_PATH,
        "script_sha256": bound[SOURCE_PATH],
        "bound_file_sha256": bound,
        "bound_files_sha256": sha256_json(dict(sorted(bound.items()))),
        "kairyu_python_paths": list(python_paths),
        "engine_source_sha256": sha256_json(dict(sorted(kairyu_bound.items()))),
        "python": sys.version,
        "platform": platform.platform(),
    }


def _checkpoint_provenance_live(model_path: Path) -> dict[str, object]:
    """Full-hash the pinned Qwen3-32B checkpoint used by this shard."""

    _require(model_path.is_dir(), f"model path is not a directory: {model_path}")
    config_path = model_path / "config.json"
    tokenizer_path = model_path / "tokenizer.json"
    _require(config_path.is_file(), "checkpoint config.json is missing")
    _require(tokenizer_path.is_file(), "checkpoint tokenizer.json is missing")
    _require(
        sha256_file(config_path) == EXPECTED_MODEL_CONFIG_SHA256,
        "checkpoint config.json differs from pinned Qwen3-32B",
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvidenceError("checkpoint config.json is invalid") from error
    _require(isinstance(config, dict), "checkpoint config.json is not an object")
    shards = sorted(model_path.glob("*.safetensors"))
    _require(len(shards) == 17, "Qwen3-32B checkpoint must contain 17 weight shards")
    with ThreadPoolExecutor(max_workers=8) as executor:
        digests = list(executor.map(sha256_file, shards))
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
        "checkpoint weights differ from pinned Qwen3-32B",
    )
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
        "architecture": {
            "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "vocab_size": config.get("vocab_size"),
        },
        "weights": weights,
        "weights_sha256": sha256_json(weights),
        "pinned_weights_rollup": parity_rollup,
        "metadata_sha256": metadata,
        "model_config_sha256": sha256_json(config),
        "read_from": str(model_path.resolve()),
    }


def _validate_checkpoint(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} is not an object")
    checkpoint = dict(value)
    _require(
        checkpoint.get("model") == MODEL and checkpoint.get("revision") == MODEL_REVISION,
        f"{name} is not pinned Qwen3-32B",
    )
    _require(
        checkpoint.get("architecture")
        == {
            "model_type": "qwen3",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        },
        f"{name} architecture differs",
    )
    weights = checkpoint.get("weights")
    _require(
        isinstance(weights, list) and len(weights) == 17,
        f"{name} weights differ",
    )
    seen: set[str] = set()
    for index, row in enumerate(weights):
        _require(isinstance(row, dict), f"{name} weight {index} is invalid")
        path = row.get("file")
        _require(
            isinstance(path, str) and path.endswith(".safetensors") and path not in seen,
            f"{name} weight {index} path is invalid",
        )
        seen.add(path)
        _exact_int(row.get("bytes"), f"{name} weight {index} bytes", minimum=1)
        _require(
            _valid_sha256(row.get("sha256")),
            f"{name} weight {index} digest is invalid",
        )
    _require(
        checkpoint.get("weights_sha256") == sha256_json(weights),
        f"{name} weight rollup differs",
    )
    _require(
        checkpoint.get("pinned_weights_rollup") == EXPECTED_MODEL_WEIGHTS_ROLLUP,
        f"{name} pinned weight identity differs",
    )
    metadata = checkpoint.get("metadata_sha256")
    _require(
        isinstance(metadata, dict)
        and {"config.json", "tokenizer.json"} <= set(metadata)
        and all(_valid_sha256(digest) for digest in metadata.values()),
        f"{name} metadata identity is incomplete",
    )
    _require(
        metadata["config.json"] == EXPECTED_MODEL_CONFIG_SHA256,
        f"{name} config identity differs",
    )
    _require(
        _valid_sha256(checkpoint.get("model_config_sha256")),
        f"{name} canonical config digest is invalid",
    )
    _require(
        isinstance(checkpoint.get("read_from"), str)
        and Path(checkpoint["read_from"]).is_absolute(),
        f"{name} read path is invalid",
    )
    return checkpoint


def _run_text(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            tuple(command),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise EvidenceError(f"cannot execute {command[0]!r}") from error
    _require(completed.returncode == 0, f"command failed: {' '.join(command)}")
    return completed.stdout.strip()


def _visible_gpu_rows() -> tuple[dict[str, str], ...]:
    output = _run_text(
        (
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id,name",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        _require(len(parts) == 3, "nvidia-smi GPU identity row is malformed")
        rows.append(
            {
                "device_uuid": parts[0],
                "pci_bus_id": parts[1].lower(),
                "name": parts[2],
            }
        )
    _require(rows, "nvidia-smi returned no GPU identities")
    return tuple(rows)


def _resolve_container_id(
    *,
    container_id: str | None,
    container_id_file: Path | None,
) -> str:
    from_file: str | None = None
    if container_id_file is not None:
        _require(container_id_file.is_file(), "container id file is missing")
        from_file = container_id_file.read_text(encoding="utf-8").strip()
    if container_id is not None and from_file is not None:
        _require(container_id == from_file, "container id argument and file differ")
    resolved = container_id or from_file
    _require(_valid_sha256(resolved), "full 64-hex Docker container id is required")
    return str(resolved)


def _command_options(
    command: object,
    *,
    entrypoint: str | None = None,
) -> dict[str, list[str]]:
    _require(
        isinstance(command, list)
        and len(command) >= 2
        and all(isinstance(part, str) and part for part in command),
        "container command is missing",
    )
    _require(
        command[0].endswith(SOURCE_PATH)
        and command[1] in {"run", "quality-run"}
        and (entrypoint is None or command[1] == entrypoint),
        "container command is not the expected F4b entrypoint",
    )
    options: dict[str, list[str]] = {}
    index = 2
    while index < len(command):
        option = command[index]
        _require(
            option.startswith("--") and index + 1 < len(command),
            "container run command contains a malformed option",
        )
        argument = command[index + 1]
        _require(
            not argument.startswith("--"),
            f"container run option {option} has no argument",
        )
        options.setdefault(option, []).append(argument)
        index += 2
    return options


def _container_provenance_live(
    *,
    container_id: str,
    image_id: str,
    repo_digests: Sequence[str],
    command: Sequence[str],
) -> dict[str, object]:
    _require(
        image_id == F4A_CALIBRATED_IMAGE_ID,
        "container image id differs from the calibrated F4a image",
    )
    _require(
        bool(repo_digests)
        and all(
            isinstance(digest, str)
            and "@sha256:" in digest
            and _valid_sha256(digest.rsplit("@sha256:", 1)[-1])
            for digest in repo_digests
        ),
        "container repository digest is invalid",
    )
    hostname = socket.gethostname().strip()
    _require(
        bool(hostname) and container_id.startswith(hostname),
        "current Docker hostname is not a prefix of supplied container id",
    )
    command_list = list(command)
    options = _command_options(command_list)
    direct = options.get("--container-id", [])
    from_file = options.get("--container-id-file", [])
    _require(
        (len(direct), len(from_file)) in {(1, 0), (0, 1)},
        "container command must identify its container exactly once",
    )
    if direct:
        _require(direct[0] == container_id, "container-id argument differs")
        binding = {
            "option": "--container-id",
            "argument": direct[0],
            "resolved_container_id": container_id,
        }
    else:
        identifier_path = Path(from_file[0])
        _require(identifier_path.is_file(), "container-id-file argument is missing")
        _require(
            identifier_path.read_text(encoding="utf-8").strip() == container_id,
            "container-id-file argument differs",
        )
        binding = {
            "option": "--container-id-file",
            "argument": from_file[0],
            "resolved_container_id": container_id,
        }
    return {
        "container_id": container_id,
        "image_id": image_id,
        "repo_digests": list(repo_digests),
        "command": command_list,
        "container_id_binding": binding,
        "hostname": hostname,
    }


def _validate_container(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "container provenance is not an object")
    container = dict(value)
    _require(
        set(container)
        == {
            "container_id",
            "image_id",
            "repo_digests",
            "command",
            "container_id_binding",
            "hostname",
        },
        "container provenance fields differ",
    )
    image_id = container.get("image_id")
    _require(
        image_id == F4A_CALIBRATED_IMAGE_ID,
        "container image id differs from the calibrated F4a image",
    )
    _require(
        _valid_sha256(container.get("container_id")),
        "container id is invalid",
    )
    options = _command_options(container.get("command"))
    _require(
        isinstance(container.get("repo_digests"), list)
        and bool(container["repo_digests"])
        and all(
            isinstance(item, str)
            and "@sha256:" in item
            and _valid_sha256(item.rsplit("@sha256:", 1)[-1])
            for item in container["repo_digests"]
        ),
        "container repository digest inventory is invalid",
    )
    hostname = container.get("hostname")
    _require(
        isinstance(hostname, str)
        and bool(hostname)
        and container["container_id"].startswith(hostname),
        "container hostname is not a prefix of its full id",
    )
    binding = container.get("container_id_binding")
    _require(
        isinstance(binding, dict)
        and set(binding) == {"option", "argument", "resolved_container_id"}
        and binding.get("option") in {"--container-id", "--container-id-file"}
        and isinstance(binding.get("argument"), str)
        and bool(binding["argument"])
        and binding.get("resolved_container_id") == container["container_id"]
        and options.get(binding["option"]) == [binding["argument"]],
        "container id binding differs from its command and full id",
    )
    other = "--container-id-file" if binding["option"] == "--container-id" else "--container-id"
    _require(other not in options, "container command has two id authorities")
    return container


def _one_command_argument(
    options: Mapping[str, list[str]],
    name: str,
) -> str:
    values = options.get(name)
    _require(values is not None and len(values) == 1, f"container command {name} differs")
    return values[0]


def _validate_run_command(
    header: Mapping[str, object],
    *,
    container: Mapping[str, object],
    checkpoint: Mapping[str, object],
    profile: Mapping[str, object] | None,
) -> None:
    options = _command_options(container["command"], entrypoint="run")
    allowed = {
        "--round",
        "--arm",
        "--cohort",
        "--model-path",
        "--profile",
        "--output",
        "--image-id",
        "--container-id",
        "--container-id-file",
        "--repo-digest",
    }
    _require(set(options) <= allowed, "container run command has unknown options")
    _require(
        _one_command_argument(options, "--round") == str(header["round"])
        and _one_command_argument(options, "--arm") == header["arm"]
        and _one_command_argument(options, "--cohort") == header["cohort"],
        "container command shard identity differs",
    )
    _require(
        _one_command_argument(options, "--model-path") == checkpoint.get("read_from"),
        "container command model path differs from checkpoint provenance",
    )
    _require(
        _one_command_argument(options, "--output") == header.get("output_path"),
        "container command output path differs from raw identity",
    )
    _require(
        _one_command_argument(options, "--image-id") == container["image_id"],
        "container command image id differs from provenance",
    )
    _require(
        options.get("--repo-digest", []) == container["repo_digests"],
        "container command repository digests differ from provenance",
    )
    profile_arguments = options.get("--profile", [])
    if profile is None:
        _require(not profile_arguments, "tier-off container command has a profile")
    else:
        _require(
            len(profile_arguments) == 1 and bool(profile_arguments[0]),
            "tier-on container command lacks exactly one profile",
        )


def _validate_quality_run_command(
    header: Mapping[str, object],
    *,
    container: Mapping[str, object],
    checkpoint: Mapping[str, object],
    profile: Mapping[str, object] | None,
) -> None:
    command = container["command"]
    options = _command_options(command, entrypoint="quality-run")
    allowed = {
        "--arm",
        "--cohort",
        "--performance-raw-sha256",
        "--model-path",
        "--profile",
        "--output",
        "--image-id",
        "--container-id",
        "--container-id-file",
        "--repo-digest",
    }
    _require(set(options) <= allowed, "quality container command has unknown options")
    _require(
        _one_command_argument(options, "--arm") == header["arm"]
        and _one_command_argument(options, "--cohort") == header["cohort"],
        "quality container command shard identity differs",
    )
    _require(
        _one_command_argument(options, "--performance-raw-sha256")
        == header["parent_performance_raw_sha256"],
        "quality container command parent raw digest differs",
    )
    _require(
        _one_command_argument(options, "--model-path") == checkpoint.get("read_from"),
        "quality container command model path differs",
    )
    _require(
        _one_command_argument(options, "--output") == header.get("output_path"),
        "quality container command output path differs",
    )
    _require(
        _one_command_argument(options, "--image-id") == container["image_id"],
        "quality container command image id differs",
    )
    _require(
        options.get("--repo-digest", []) == container["repo_digests"],
        "quality container command repository digests differ",
    )
    profile_arguments = options.get("--profile", [])
    if profile is None:
        _require(
            not profile_arguments,
            "tier-off quality container command has a profile",
        )
    else:
        _require(
            len(profile_arguments) == 1 and bool(profile_arguments[0]),
            "tier-on quality container command lacks exactly one profile",
        )


def _profile_descriptor_live(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read retained TP4 F4a profile: {path}") from error
    _require(isinstance(value, dict), "F4a profile must be an object")
    runtime = value.get("runtime_identity")
    policy = value.get("policy")
    file_sha256 = sha256_bytes(payload)
    _require(
        file_sha256 == F4A_PROFILE_FILE_SHA256,
        "retained TP4 F4a profile file SHA-256 differs",
    )
    _require(
        value.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "F4a profile schema differs",
    )
    _require(value.get("passed") is True, "F4a profile is not passing")
    _require(
        value.get("evidence_sha256") == F4A_PROFILE_EVIDENCE_SHA256,
        "retained TP4 F4a evidence SHA-256 differs",
    )
    _require(
        value.get("runtime_fingerprint") == F4A_PROFILE_RUNTIME_FINGERPRINT,
        "retained TP4 F4a runtime fingerprint differs",
    )
    _require(
        isinstance(runtime, dict) and runtime.get("tensor_parallel_size") == TP_SIZE,
        "F4a profile is not TP4",
    )
    _require(
        isinstance(policy, dict) and policy.get("min_restore_tokens") == 1024,
        "retained TP4 F4a threshold must be 1024 tokens",
    )
    return {
        "file_sha256": file_sha256,
        "value": value,
        "value_sha256": sha256_json(value),
    }


def _hardware_live(cohort: str) -> dict[str, object]:
    import torch

    rows = list(_visible_gpu_rows())
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == TP_SIZE, "run requires exactly four visible GPUs")
    _require(len(rows) == TP_SIZE, "nvidia-smi must expose exactly four GPUs")
    for index, row in enumerate(rows):
        properties = torch.cuda.get_device_properties(index)
        row.update(
            {
                "rank": index,
                "torch_name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            }
        )
    driver = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"),
        capture_output=True,
        text=True,
        check=False,
    )
    _require(driver.returncode == 0, "cannot capture NVIDIA driver version")
    drivers = sorted({line.strip() for line in driver.stdout.splitlines() if line.strip()})
    _require(len(drivers) == 1, "visible GPUs report different driver versions")
    return {
        "cohort": cohort,
        "hostname": socket.gethostname(),
        "gpus": rows,
        "driver_version": drivers[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nccl_version": str(torch.cuda.nccl.version()),
    }


def _runtime_live(backend: object, source: Mapping[str, object], arm: str) -> dict[str, object]:
    decision = getattr(backend, "attention_backend_decision", None)
    if is_dataclass(decision) and not isinstance(decision, type):
        decision_value = asdict(decision)
    elif isinstance(decision, Mapping):
        decision_value = dict(decision)
    else:
        raise EvidenceError("runtime attention backend decision is missing")
    return {
        "tensor_parallel_size": TP_SIZE,
        "page_size": PAGE_SIZE,
        "num_pages": HBM_PAGES,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "pipeline_depth": 1,
        "decode_mode": "eager",
        "engine_source_sha256": source["engine_source_sha256"],
        "attention_backend_decision": decision_value,
        "kv_cache_dtype_requested": backend.kv_cache_dtype_requested,
        "kv_cache_dtype_resolved": backend.kv_cache_dtype_resolved,
        "dram_kv_tier_enabled": backend.dram_kv_tier_enabled,
        "dram_kv_tier_capacity_pages": backend.dram_kv_tier_capacity_pages,
        "dram_kv_tier_profile_sha256": backend.dram_kv_tier_profile_sha256,
        "dram_kv_tier_min_restore_tokens": backend.dram_kv_tier_min_restore_tokens,
        "arm": arm,
    }


def _tier_stats(value: object, name: str) -> dict[str, int]:
    _require(isinstance(value, dict), f"{name} tier stats are not an object")
    _require(set(value) == set(TIER_STATS_FIELDS), f"{name} tier stats fields differ")
    return {field: _exact_int(value[field], f"{name} {field}") for field in TIER_STATS_FIELDS}


def _step_event(
    *,
    step_index: int,
    elapsed_ns: int,
    update: object | None,
    prior_output: tuple[int, ...],
    prior_finished: bool,
    cache: object,
) -> tuple[dict[str, object], tuple[int, ...], bool]:
    if update is None:
        output = prior_output
        finished = prior_finished
        finish_reason = None
        error_type = None
        emitted_update = False
    else:
        output = tuple(int(token) for token in update.outputs)
        finished = bool(update.finished)
        finish_reason = update.finish_reason
        error_type = type(update.error).__name__ if update.error is not None else None
        emitted_update = True
    event = {
        "step_index": step_index,
        "elapsed_ns": elapsed_ns,
        "emitted_update": emitted_update,
        "output_token_ids": list(output),
        "output_count": len(output),
        "finished": finished,
        "finish_reason": finish_reason,
        "error_type": error_type,
        "tier_stats": cache.dram_tier_stats,
        "num_free_pages": cache.num_free_pages,
    }
    return event, output, finished


def _run_one_request(
    backend: object,
    trace_row: Mapping[str, object],
    params: object,
) -> dict[str, object]:
    from kairyu.engine.prompt import TokensPrompt

    request_id = f"f4b-s{trace_row['session']:02d}-t{trace_row['turn']:02d}"
    prompt_ids = tuple(int(token) for token in trace_row["prompt_token_ids"])
    prompt = TokensPrompt(prompt_ids)
    loop = backend._loop
    prepared = loop.prepare_prompt(prompt, params)
    initial_stats = backend._cache.dram_tier_stats
    initial_free = backend._cache.num_free_pages
    started = time.perf_counter_ns()
    loop.submit(request_id, prompt, params, prepared_prompt=prepared)
    events: list[dict[str, object]] = []
    output: tuple[int, ...] = ()
    finished = False
    terminal_update = None
    step_index = 0
    while loop.has_work():
        updates = loop.step()
        elapsed = time.perf_counter_ns() - started
        matching = [update for candidate_id, update in updates if candidate_id == request_id]
        _require(len(matching) <= 1, "one synchronous step emitted duplicate request updates")
        _require(
            all(candidate_id == request_id for candidate_id, _update in updates),
            "sequential trace observed another request",
        )
        update = matching[0] if matching else None
        event, output, finished = _step_event(
            step_index=step_index,
            elapsed_ns=elapsed,
            update=update,
            prior_output=output,
            prior_finished=finished,
            cache=backend._cache,
        )
        events.append(event)
        if update is not None:
            terminal_update = update
            if update.error is not None:
                raise update.error
        step_index += 1
        if finished:
            break
    _require(finished and not loop.has_work(), "request did not finish cleanly in isolation")
    _require(terminal_update is not None, "request produced no stream update")
    usage = {
        "prompt_tokens": terminal_update.num_prompt_tokens,
        "cached_tokens": terminal_update.num_cached_tokens,
        "completion_tokens": len(output),
    }
    result = {
        "row_type": "request",
        "sequence": trace_row["sequence"],
        "session": trace_row["session"],
        "turn": trace_row["turn"],
        "request_id": request_id,
        "status": "success",
        "retry_count": 0,
        "prompt_tokens": len(prompt_ids),
        "prompt_token_ids": list(prompt_ids),
        "prompt_token_ids_sha256": sha256_json(list(prompt_ids)),
        "output_token_ids": list(output),
        "usage": usage,
        "initial_tier_stats": initial_stats,
        "initial_num_free_pages": initial_free,
        "events": events,
        "final_tier_stats": backend._cache.dram_tier_stats,
        "final_num_free_pages": backend._cache.num_free_pages,
    }
    if getattr(params, "logprobs", None) is not None:
        distributions = terminal_update.logprobs
        _require(
            isinstance(distributions, tuple)
            and len(distributions) == len(output),
            "quality request did not retain one logprob distribution per token",
        )
        records = []
        for position, (token_id, distribution) in enumerate(
            zip(output, distributions, strict=True)
        ):
            _require(
                isinstance(distribution, dict)
                and token_id in distribution,
                f"quality token {position} lacks its selected logprob",
            )
            ordered = sorted(
                (
                    {"token_id": int(candidate), "logprob": float(logprob)}
                    for candidate, logprob in distribution.items()
                ),
                key=lambda item: (-item["logprob"], item["token_id"]),
            )
            records.append(
                {
                    "position": position,
                    "selected_token_id": token_id,
                    "selected_logprob": float(distribution[token_id]),
                    "top_logprobs": ordered,
                }
            )
        result["logprob_records"] = records
    return result


async def _run_live_shard(
    *,
    round_index: int,
    arm: str,
    cohort: str,
    model_path: Path,
    profile_path: Path | None,
    output: Path,
    container: Mapping[str, object],
) -> dict[str, object]:
    from kairyu.engine.kairyu_backend import KairyuBackend
    from kairyu.engine.tokenizer import resolve_tokenizer
    from kairyu.sampling_params import SamplingParams

    _require((round_index, arm, cohort) in SHARD_PLAN, "run is outside the fixed shard plan")
    _require(model_path.is_dir(), f"model path is not a directory: {model_path}")
    if arm == "on":
        _require(profile_path is not None, "tier-on run requires retained TP4 F4a profile")
        profile = _profile_descriptor_live(profile_path)
        tier_capacity = DRAM_PAGES
        tier_profile: Path | None = profile_path
    else:
        _require(profile_path is None, "tier-off run must not receive a profile")
        profile = None
        tier_capacity = 0
        tier_profile = None

    source_start = _source_provenance_live()
    checkpoint = _checkpoint_provenance_live(model_path)
    tokenizer = resolve_tokenizer(str(model_path))
    trace, request_specs = build_trace(
        tokenizer,
        tokenizer_sha256=checkpoint["metadata_sha256"]["tokenizer.json"],
    )
    hardware = _hardware_live(cohort)
    params = SamplingParams(
        temperature=0.0,
        min_tokens=OUTPUT_TOKENS,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
        seed=0,
    )
    backend = KairyuBackend(
        model_path=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=TP_SIZE,
        num_pages=HBM_PAGES,
        page_size=PAGE_SIZE,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=1,
        max_model_len=MAX_MODEL_LEN,
        pipeline_depth=1,
        decode_mode="eager",
        dram_kv_tier_capacity_pages=tier_capacity,
        dram_kv_tier_profile=tier_profile,
    )
    runtime = _runtime_live(backend, source_start, arm)
    if arm == "on":
        _require(runtime["dram_kv_tier_enabled"] is True, "tier-on backend did not enable DRAM")
        _require(
            runtime["dram_kv_tier_capacity_pages"] == DRAM_PAGES, "tier-on slab capacity differs"
        )
        _require(
            runtime["dram_kv_tier_profile_sha256"] == profile["file_sha256"],
            "backend accepted a different F4a profile",
        )
    else:
        _require(
            runtime["dram_kv_tier_enabled"] is False, "tier-off backend unexpectedly enabled DRAM"
        )
        _require(
            runtime["dram_kv_tier_capacity_pages"] == 0, "tier-off backend allocated a DRAM slab"
        )
        _require(
            runtime["dram_kv_tier_profile_sha256"] is None, "tier-off backend loaded a profile"
        )

    run_id = f"issue-188-r{round_index}-{arm}-{cohort}-{uuid.uuid4().hex}"
    started_utc = datetime.now(UTC).isoformat()
    started_unix_ns = time.time_ns()
    requests: list[dict[str, object]] = []
    try:
        for spec in request_specs:
            requests.append(_run_one_request(backend, spec, params))
    finally:
        await backend.shutdown()
    ended_unix_ns = time.time_ns()
    ended_utc = datetime.now(UTC).isoformat()
    source_end = _source_provenance_live()
    _require(source_start == source_end, "source changed during the formal shard")
    rows: list[dict[str, object]] = [
        {
            "row_type": "run",
            "schema_version": SCHEMA_VERSION,
            "measurement_kind": MEASUREMENT_KIND,
            "run_id": run_id,
            "output_path": str(output),
            "round": round_index,
            "arm": arm,
            "cohort": cohort,
            "config": formal_config(),
            "trace": trace,
            "source": source_start,
            "checkpoint": checkpoint,
            "profile": profile,
            "container": dict(container),
            "hardware": hardware,
            "runtime": runtime,
            "measurement_started_at": started_utc,
            "measurement_started_unix_ns": started_unix_ns,
        },
        *requests,
        {
            "row_type": "run_end",
            "run_id": run_id,
            "round": round_index,
            "arm": arm,
            "cohort": cohort,
            "status": "complete",
            "errors": [],
            "request_count": len(requests),
            "source": source_end,
            "final_tier_stats": requests[-1]["final_tier_stats"],
            "measurement_ended_at": ended_utc,
            "measurement_ended_unix_ns": ended_unix_ns,
        },
    ]
    write_jsonl(output, rows)
    parsed = parse_shard(read_jsonl(output))
    return {
        "schema_version": SCHEMA_VERSION,
        "shard": list(parsed["identity"]),
        "requests": len(parsed["requests"]),
        "raw_sha256": sha256_file(output),
        "output": str(output),
    }


async def _run_live_quality_shard(
    *,
    arm: str,
    cohort: str,
    performance_raw_sha256: str,
    model_path: Path,
    profile_path: Path | None,
    output: Path,
    container: Mapping[str, object],
) -> dict[str, object]:
    """Record distribution quality without contaminating TPOT measurements."""

    from kairyu.engine.kairyu_backend import KairyuBackend
    from kairyu.engine.tokenizer import resolve_tokenizer
    from kairyu.sampling_params import SamplingParams

    _require((arm, cohort) in QUALITY_PLAN, "quality run is outside the fixed plan")
    _require(
        _valid_sha256(performance_raw_sha256),
        "quality parent performance raw digest is invalid",
    )
    _require(model_path.is_dir(), f"model path is not a directory: {model_path}")
    if arm == "on":
        _require(
            profile_path is not None,
            "tier-on quality run requires retained TP4 F4a profile",
        )
        profile = _profile_descriptor_live(profile_path)
        tier_capacity = DRAM_PAGES
        tier_profile: Path | None = profile_path
    else:
        _require(
            profile_path is None,
            "tier-off quality run must not receive a profile",
        )
        profile = None
        tier_capacity = 0
        tier_profile = None

    source_start = _source_provenance_live()
    checkpoint = _checkpoint_provenance_live(model_path)
    tokenizer = resolve_tokenizer(str(model_path))
    trace, request_specs = build_trace(
        tokenizer,
        tokenizer_sha256=checkpoint["metadata_sha256"]["tokenizer.json"],
    )
    hardware = _hardware_live(cohort)
    params = SamplingParams(
        temperature=0.0,
        min_tokens=OUTPUT_TOKENS,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
        seed=0,
        logprobs=QUALITY_TOP_LOGPROBS,
    )
    backend = KairyuBackend(
        model_path=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=TP_SIZE,
        num_pages=HBM_PAGES,
        page_size=PAGE_SIZE,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=1,
        max_model_len=MAX_MODEL_LEN,
        pipeline_depth=1,
        decode_mode="eager",
        dram_kv_tier_capacity_pages=tier_capacity,
        dram_kv_tier_profile=tier_profile,
    )
    runtime = _runtime_live(backend, source_start, arm)
    if arm == "on":
        _require(
            runtime["dram_kv_tier_enabled"] is True
            and runtime["dram_kv_tier_capacity_pages"] == DRAM_PAGES
            and runtime["dram_kv_tier_profile_sha256"] == profile["file_sha256"],
            "tier-on quality backend did not bind the retained tier profile",
        )
    else:
        _require(
            runtime["dram_kv_tier_enabled"] is False
            and runtime["dram_kv_tier_capacity_pages"] == 0
            and runtime["dram_kv_tier_profile_sha256"] is None,
            "tier-off quality backend unexpectedly enabled DRAM",
        )

    run_id = f"issue-188-quality-{arm}-{cohort}-{uuid.uuid4().hex}"
    started_utc = datetime.now(UTC).isoformat()
    started_unix_ns = time.time_ns()
    requests: list[dict[str, object]] = []
    try:
        for spec in request_specs:
            request = _run_one_request(backend, spec, params)
            requests.append(
                {
                    key: request[key]
                    for key in (
                        "sequence",
                        "session",
                        "turn",
                        "request_id",
                        "status",
                        "retry_count",
                        "prompt_tokens",
                        "prompt_token_ids",
                        "prompt_token_ids_sha256",
                        "output_token_ids",
                        "usage",
                        "initial_tier_stats",
                        "final_tier_stats",
                        "logprob_records",
                    )
                }
            )
    finally:
        await backend.shutdown()
    ended_unix_ns = time.time_ns()
    ended_utc = datetime.now(UTC).isoformat()
    source_end = _source_provenance_live()
    _require(source_start == source_end, "source changed during the quality shard")
    rows: list[dict[str, object]] = [
        {
            "row_type": "quality_run",
            "schema_version": QUALITY_SCHEMA_VERSION,
            "measurement_kind": QUALITY_MEASUREMENT_KIND,
            "run_id": run_id,
            "output_path": str(output),
            "arm": arm,
            "cohort": cohort,
            "parent_performance_raw_sha256": performance_raw_sha256,
            "config": quality_config(),
            "trace": trace,
            "source": source_start,
            "checkpoint": checkpoint,
            "profile": profile,
            "container": dict(container),
            "hardware": hardware,
            "runtime": runtime,
            "measurement_started_at": started_utc,
            "measurement_started_unix_ns": started_unix_ns,
        },
        *({"row_type": "quality_request", **request} for request in requests),
        {
            "row_type": "quality_run_end",
            "run_id": run_id,
            "arm": arm,
            "cohort": cohort,
            "status": "complete",
            "errors": [],
            "request_count": len(requests),
            "source": source_end,
            "final_tier_stats": requests[-1]["final_tier_stats"],
            "measurement_ended_at": ended_utc,
            "measurement_ended_unix_ns": ended_unix_ns,
        },
    ]
    write_jsonl(output, rows)
    parse_quality_shard(read_jsonl(output))
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "shard": [arm, cohort],
        "requests": len(requests),
        "raw_sha256": sha256_file(output),
        "output": str(output),
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    evidence_write_jsonl(path, rows)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return evidence_read_jsonl(path, error_type=EvidenceError)


def _validate_source(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "source provenance is not an object")
    source = dict(value)
    _require(_valid_commit(source.get("commit")), "source commit is invalid")
    _require(
        source.get("dirty") is False and source.get("source_kind") == "git",
        "source is not a clean git commit",
    )
    _require(source.get("script_path") == SOURCE_PATH, "source script path differs")
    bound = source.get("bound_file_sha256")
    paths = source.get("kairyu_python_paths")
    _require(
        isinstance(bound, dict) and REQUIRED_SOURCE_PATHS <= set(bound),
        "source bound files are incomplete",
    )
    _require(
        all(isinstance(path, str) and _valid_sha256(digest) for path, digest in bound.items()),
        "source bound file digest is invalid",
    )
    _require(
        isinstance(paths, list)
        and paths
        and len(paths) == len(set(paths))
        and all(
            isinstance(path, str) and path.startswith("kairyu/") and path.endswith(".py")
            for path in paths
        ),
        "source Kairyu Python inventory is invalid",
    )
    _require(
        set(paths)
        == {path for path in bound if path.startswith("kairyu/") and path.endswith(".py")},
        "source does not bind its recorded full Kairyu Python inventory",
    )
    _require(source.get("script_sha256") == bound.get(SOURCE_PATH), "source script digest differs")
    _require(
        source.get("bound_files_sha256") == sha256_json(dict(sorted(bound.items()))),
        "source rollup differs",
    )
    engine = {path: bound[path] for path in paths}
    _require(
        source.get("engine_source_sha256") == sha256_json(dict(sorted(engine.items()))),
        "engine source rollup differs",
    )
    return source


def _validate_profile(
    value: object, *, arm: str, engine_source_sha256: str
) -> dict[str, object] | None:
    if arm == "off":
        _require(value is None, "tier-off shard must not bind a profile")
        return None
    _require(isinstance(value, dict), "tier-on profile descriptor is missing")
    profile = dict(value)
    _require(
        set(profile) == {"file_sha256", "value", "value_sha256"}, "profile descriptor fields differ"
    )
    _require(
        profile["file_sha256"] == F4A_PROFILE_FILE_SHA256,
        "profile file digest differs from retained F4a TP4",
    )
    _require(
        _valid_sha256(profile["value_sha256"])
        and profile["value_sha256"] == sha256_json(profile["value"]),
        "profile value digest differs",
    )
    raw = profile["value"]
    _require(
        isinstance(raw, dict) and raw.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "profile schema differs",
    )
    _require(raw.get("passed") is True, "profile is not passing evidence")
    _require(
        raw.get("evidence_sha256") == F4A_PROFILE_EVIDENCE_SHA256,
        "profile evidence digest differs from retained F4a TP4",
    )
    _require(
        raw.get("runtime_fingerprint") == F4A_PROFILE_RUNTIME_FINGERPRINT,
        "profile runtime fingerprint differs from retained F4a TP4",
    )
    runtime = raw.get("runtime_identity")
    policy = raw.get("policy")
    _require(
        isinstance(runtime, dict)
        and runtime.get("tensor_parallel_size") == TP_SIZE
        and runtime.get("page_size") == PAGE_SIZE
        and runtime.get("engine_source_sha256") == engine_source_sha256,
        "profile runtime identity differs from this TP4 engine",
    )
    _require(
        isinstance(policy, dict) and policy.get("min_restore_tokens") == 1024,
        "profile threshold differs from retained F4a TP4",
    )
    return profile


def _validate_hardware(value: object, cohort: str) -> dict[str, object]:
    _require(isinstance(value, dict), "hardware provenance is not an object")
    hardware = dict(value)
    _require(
        set(hardware)
        == {
            "cohort",
            "hostname",
            "gpus",
            "driver_version",
            "torch_version",
            "torch_cuda_version",
            "nccl_version",
        },
        "hardware provenance fields differ",
    )
    gpus = hardware.get("gpus")
    _require(hardware.get("cohort") == cohort, "hardware cohort differs")
    _require(
        isinstance(hardware.get("hostname"), str) and bool(hardware["hostname"]),
        "hardware hostname is missing",
    )
    for field in (
        "driver_version",
        "torch_version",
        "torch_cuda_version",
        "nccl_version",
    ):
        _require(
            isinstance(hardware.get(field), str) and bool(hardware[field]),
            f"hardware {field} is missing",
        )
    _require(isinstance(gpus, list) and len(gpus) == TP_SIZE, "hardware must contain four GPUs")
    uuids: list[str] = []
    specifications: list[dict[str, object]] = []
    for rank, row in enumerate(gpus):
        _require(
            isinstance(row, dict)
            and set(row)
            == {
                "rank",
                "device_uuid",
                "pci_bus_id",
                "name",
                "torch_name",
                "compute_capability",
                "total_memory_bytes",
            }
            and row.get("rank") == rank,
            "hardware GPU row fields/rank differ",
        )
        uuid_value = row.get("device_uuid")
        _require(
            isinstance(uuid_value, str) and uuid_value.startswith("GPU-"),
            "hardware GPU UUID is invalid",
        )
        _require(
            isinstance(row.get("pci_bus_id"), str) and bool(row["pci_bus_id"]),
            "hardware GPU PCI identity is missing",
        )
        _require(
            isinstance(row.get("name"), str)
            and bool(row["name"])
            and row.get("torch_name") == row["name"],
            "hardware GPU names differ",
        )
        capability = row.get("compute_capability")
        _require(
            isinstance(capability, list)
            and len(capability) == 2
            and all(type(part) is int and part >= 0 for part in capability),
            "hardware GPU compute capability is invalid",
        )
        _exact_int(row.get("total_memory_bytes"), "hardware GPU memory", minimum=1)
        uuids.append(uuid_value)
        specifications.append(
            {
                "name": row["name"],
                "torch_name": row["torch_name"],
                "compute_capability": capability,
                "total_memory_bytes": row["total_memory_bytes"],
            }
        )
    _require(len(set(uuids)) == TP_SIZE, "hardware GPU UUIDs are not unique")
    _require(
        len({canonical_json(specification) for specification in specifications}) == 1,
        "one TP4 cohort contains mixed GPU specifications",
    )
    return hardware


def _validate_attention_decision(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "runtime attention backend decision is not an object")
    decision = dict(value)
    _require(
        set(decision)
        == {"requested", "resolved", "source", "components", "rationale", "architecture"},
        "runtime attention backend decision fields differ",
    )
    _require(
        decision.get("requested")
        in {"auto", "torch", "flashinfer", "flashattention3", "flashattention4"}
        and decision.get("resolved")
        in {"torch", "flashinfer", "flashattention3", "flashattention4"}
        and decision.get("source") in {"env", "hw_profile"}
        and isinstance(decision.get("rationale"), str)
        and bool(decision["rationale"]),
        "runtime attention backend selection values are invalid",
    )
    components = decision.get("components")
    _require(
        isinstance(components, dict)
        and set(components) == {"prefill", "decode", "kv_mode"}
        and all(isinstance(item, str) and bool(item) for item in components.values()),
        "runtime attention backend components are invalid",
    )
    architecture = decision.get("architecture")
    _require(
        isinstance(architecture, dict)
        and set(architecture) == {"arch", "device_name", "sm", "kernel_tier"}
        and isinstance(architecture.get("arch"), str)
        and bool(architecture["arch"])
        and isinstance(architecture.get("device_name"), str)
        and bool(architecture["device_name"])
        and type(architecture.get("sm")) is int
        and architecture["sm"] > 0
        and isinstance(architecture.get("kernel_tier"), str)
        and bool(architecture["kernel_tier"]),
        "runtime attention backend architecture is invalid",
    )
    return decision


def _validate_runtime(
    value: object,
    *,
    arm: str,
    source: Mapping[str, object],
    profile: Mapping[str, object] | None,
) -> dict[str, object]:
    _require(isinstance(value, dict), "runtime is not an object")
    runtime = dict(value)
    _require(
        set(runtime) == RUNTIME_COMMON_FIELDS | RUNTIME_ARM_FIELDS,
        "runtime fields are incomplete or unknown",
    )
    expected = {
        "tensor_parallel_size": TP_SIZE,
        "page_size": PAGE_SIZE,
        "num_pages": HBM_PAGES,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "pipeline_depth": 1,
        "decode_mode": "eager",
        "engine_source_sha256": source["engine_source_sha256"],
        "arm": arm,
    }
    _require(
        all(runtime.get(key) == expected_value for key, expected_value in expected.items()),
        "runtime fixed configuration differs",
    )
    runtime["attention_backend_decision"] = _validate_attention_decision(
        runtime.get("attention_backend_decision")
    )
    _require(
        runtime.get("kv_cache_dtype_requested") in {"auto", "bfloat16"}
        and runtime.get("kv_cache_dtype_resolved") == "bfloat16",
        "runtime KV dtype identity is invalid for pinned Qwen3-32B",
    )
    if arm == "on":
        assert profile is not None
        _require(
            runtime.get("dram_kv_tier_enabled") is True
            and runtime.get("dram_kv_tier_capacity_pages") == DRAM_PAGES
            and runtime.get("dram_kv_tier_profile_sha256") == profile["file_sha256"]
            and runtime.get("dram_kv_tier_min_restore_tokens") == 1024,
            "tier-on runtime did not bind the retained profile and slab",
        )
    else:
        _require(
            runtime.get("dram_kv_tier_enabled") is False
            and runtime.get("dram_kv_tier_capacity_pages") == 0
            and runtime.get("dram_kv_tier_profile_sha256") is None
            and runtime.get("dram_kv_tier_min_restore_tokens") is None,
            "tier-off runtime loaded a profile or DRAM slab",
        )
    return runtime


def _validate_trace(value: object) -> dict[str, object]:
    trace = _descriptor(value, "trace")
    raw = trace["value"]
    _require(isinstance(raw, dict), "trace value is not an object")
    _require(raw.get("kind") == "canonical-agent-tool-token-trace-v1", "trace kind differs")
    _require(
        raw.get("shared_prefix_tokens") == SHARED_TOKENS and raw.get("turn_tokens") == TURN_TOKENS,
        "trace geometry differs",
    )
    _require(
        _valid_sha256(raw.get("tokenizer_sha256"))
        and _valid_sha256(raw.get("shared_prefix_sha256")),
        "trace identity digest is invalid",
    )
    rows = raw.get("rows")
    _require(
        isinstance(rows, list) and len(rows) == REQUESTS_PER_SHARD, "trace request schedule differs"
    )
    for sequence, row in enumerate(rows):
        session, turn = _request_identity(sequence)
        _require(
            isinstance(row, dict)
            and row.get("sequence") == sequence
            and row.get("session") == session
            and row.get("turn") == turn
            and row.get("prompt_tokens") == SHARED_TOKENS + (turn + 1) * TURN_TOKENS
            and _valid_sha256(row.get("prompt_token_ids_sha256")),
            f"trace row {sequence} differs from fixed order/geometry",
        )
    return trace


def _validate_event_sequence(
    events: object, sequence: int
) -> tuple[list[dict[str, object]], float]:
    _require(isinstance(events, list) and events, f"request {sequence} has no raw steps")
    normalized: list[dict[str, object]] = []
    prior_output: list[int] = []
    prior_elapsed = -1
    terminal_seen = False
    for index, event in enumerate(events):
        _require(isinstance(event, dict), f"request {sequence} step {index} is invalid")
        _require(event.get("step_index") == index, f"request {sequence} step indices differ")
        elapsed = _exact_int(event.get("elapsed_ns"), f"request {sequence} elapsed")
        _require(
            elapsed > prior_elapsed, f"request {sequence} elapsed time is not strictly increasing"
        )
        output = event.get("output_token_ids")
        _require(
            isinstance(output, list)
            and all(type(token) is int and token >= 0 for token in output)
            and event.get("output_count") == len(output)
            and output[: len(prior_output)] == prior_output
            and len(output) - len(prior_output) in {0, 1},
            f"request {sequence} output stream is not cumulative one-token progress",
        )
        emitted = event.get("emitted_update")
        _require(type(emitted) is bool, f"request {sequence} emitted-update flag is invalid")
        if not emitted:
            _require(
                output == prior_output and event.get("finished") is terminal_seen,
                f"request {sequence} empty step changed stream state",
            )
        _require(
            type(event.get("finished")) is bool, f"request {sequence} finished flag is invalid"
        )
        _require(event.get("error_type") is None, f"request {sequence} recorded a step error")
        _tier_stats(event.get("tier_stats"), f"request {sequence} step {index}")
        free = _exact_int(event.get("num_free_pages"), f"request {sequence} free pages")
        _require(free <= HBM_PAGES, f"request {sequence} free pages exceed HBM capacity")
        if terminal_seen:
            raise EvidenceError(f"request {sequence} contains steps after terminal")
        terminal_seen = bool(event["finished"])
        prior_output = list(output)
        prior_elapsed = elapsed
        normalized.append(dict(event))
    _require(
        terminal_seen and len(prior_output) == OUTPUT_TOKENS,
        f"request {sequence} did not terminate at 32 tokens",
    )
    first = next((event for event in normalized if event["output_count"] == 1), None)
    terminal = normalized[-1]
    _require(
        first is not None and terminal["output_count"] == OUTPUT_TOKENS,
        f"request {sequence} lacks TPOT boundaries",
    )
    tpot = (terminal["elapsed_ns"] - first["elapsed_ns"]) / (OUTPUT_TOKENS - 1)
    _require(math.isfinite(tpot) and tpot > 0, f"request {sequence} TPOT is invalid")
    return normalized, tpot


def _validate_request(row: Mapping[str, object], sequence: int) -> dict[str, object]:
    session, turn = _request_identity(sequence)
    _require(
        row.get("row_type") == "request"
        and row.get("sequence") == sequence
        and row.get("session") == session
        and row.get("turn") == turn,
        f"request {sequence} identity/order differs",
    )
    _require(
        row.get("status") == "success" and row.get("retry_count") == 0,
        f"request {sequence} failed or retried",
    )
    prompt = row.get("prompt_token_ids")
    expected_prompt_tokens = SHARED_TOKENS + (turn + 1) * TURN_TOKENS
    _require(
        isinstance(prompt, list)
        and len(prompt) == expected_prompt_tokens
        and all(type(token) is int and token >= 0 for token in prompt)
        and row.get("prompt_tokens") == len(prompt)
        and row.get("prompt_token_ids_sha256") == sha256_json(prompt),
        f"request {sequence} prompt identity differs",
    )
    output = row.get("output_token_ids")
    _require(
        isinstance(output, list)
        and len(output) == OUTPUT_TOKENS
        and all(type(token) is int and token >= 0 for token in output),
        f"request {sequence} output differs",
    )
    usage = row.get("usage")
    _require(
        isinstance(usage, dict)
        and usage.get("prompt_tokens") == expected_prompt_tokens
        and type(usage.get("cached_tokens")) is int
        and 0 <= usage["cached_tokens"] <= usage["prompt_tokens"]
        and usage.get("completion_tokens") == OUTPUT_TOKENS,
        f"request {sequence} engine usage is invalid",
    )
    initial = _tier_stats(row.get("initial_tier_stats"), f"request {sequence} initial")
    final = _tier_stats(row.get("final_tier_stats"), f"request {sequence} final")
    initial_free = _exact_int(row.get("initial_num_free_pages"), f"request {sequence} initial free")
    final_free = _exact_int(row.get("final_num_free_pages"), f"request {sequence} final free")
    _require(
        initial_free <= HBM_PAGES and final_free <= HBM_PAGES,
        f"request {sequence} free-page boundary differs",
    )
    events, tpot = _validate_event_sequence(row.get("events"), sequence)
    _require(events[-1]["tier_stats"] == final, f"request {sequence} final tier stats differ")
    _require(
        events[-1]["num_free_pages"] == final_free, f"request {sequence} final free pages differ"
    )
    return {
        **dict(row),
        "events": events,
        "initial_tier_stats": initial,
        "final_tier_stats": final,
        "tpot_ns": tpot,
    }


def parse_shard(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    _require(len(rows) == REQUESTS_PER_SHARD + 2, "shard row count differs")
    header = dict(rows[0])
    end = dict(rows[-1])
    _require(
        header.get("row_type") == "run" and end.get("row_type") == "run_end",
        "shard framing differs",
    )
    _require(
        header.get("schema_version") == SCHEMA_VERSION
        and header.get("measurement_kind") == MEASUREMENT_KIND,
        "shard schema/kind differs",
    )
    identity = (header.get("round"), header.get("arm"), header.get("cohort"))
    _require(identity in SHARD_PLAN, "shard identity is outside fixed plan")
    round_index, arm, cohort = identity
    _require(header.get("config") == formal_config(), "formal config differs")
    _require(isinstance(header.get("run_id"), str) and bool(header["run_id"]), "run id is missing")
    _require(
        isinstance(header.get("output_path"), str) and Path(header["output_path"]).is_absolute(),
        "raw output path identity is invalid",
    )
    source = _validate_source(header.get("source"))
    checkpoint = _validate_checkpoint(header.get("checkpoint"), "F4b checkpoint")
    container = _validate_container(header.get("container"))
    profile = _validate_profile(
        header.get("profile"), arm=arm, engine_source_sha256=source["engine_source_sha256"]
    )
    _validate_run_command(
        header,
        container=container,
        checkpoint=checkpoint,
        profile=profile,
    )
    hardware = _validate_hardware(header.get("hardware"), cohort)
    runtime = _validate_runtime(header.get("runtime"), arm=arm, source=source, profile=profile)
    trace = _validate_trace(header.get("trace"))
    start_ns = _exact_int(header.get("measurement_started_unix_ns"), "measurement start", minimum=1)
    end_ns = _exact_int(end.get("measurement_ended_unix_ns"), "measurement end", minimum=1)
    _require(end_ns > start_ns, "measurement interval is empty or reversed")
    _require(
        end.get("run_id") == header["run_id"]
        and (end.get("round"), end.get("arm"), end.get("cohort")) == identity
        and end.get("status") == "complete"
        and end.get("errors") == []
        and end.get("request_count") == REQUESTS_PER_SHARD
        and end.get("source") == source,
        "run_end identity/status/provenance differs",
    )
    requests = [_validate_request(row, sequence) for sequence, row in enumerate(rows[1:-1])]
    trace_rows = trace["value"]["rows"]
    _require(
        _trace_rows_from_requests(requests) == trace_rows,
        "request rows differ from bound trace descriptor",
    )
    shared = requests[0]["prompt_token_ids"][:SHARED_TOKENS]
    _require(
        sha256_json(shared) == trace["value"]["shared_prefix_sha256"],
        "raw shared prefix differs from trace",
    )
    for request in requests:
        _require(
            request["prompt_token_ids"][:SHARED_TOKENS] == shared,
            "fleet shared prefix differs between requests",
        )
        if request["turn"]:
            previous = requests[(request["turn"] - 1) * SESSIONS + request["session"]]
            _require(
                request["prompt_token_ids"][: len(previous["prompt_token_ids"])]
                == previous["prompt_token_ids"],
                "session prompt is not an append-only 512-token transcript",
            )
    _require(
        end.get("final_tier_stats") == requests[-1]["final_tier_stats"], "run_end tier stats differ"
    )
    return {
        "identity": identity,
        "header": {
            **header,
            "source": source,
            "checkpoint": checkpoint,
            "container": container,
            "profile": profile,
            "hardware": hardware,
            "runtime": runtime,
            "trace": trace,
        },
        "requests": requests,
        "end": end,
        "interval": (start_ns, end_ns),
    }


def _validate_quality_logprob_records(
    value: object,
    output: Sequence[int],
    *,
    sequence: int,
) -> list[dict[str, object]]:
    _require(
        isinstance(value, list) and len(value) == OUTPUT_TOKENS,
        f"quality request {sequence} logprob record count differs",
    )
    records: list[dict[str, object]] = []
    for position, (record_value, selected_token) in enumerate(
        zip(value, output, strict=True)
    ):
        _require(
            isinstance(record_value, dict),
            f"quality request {sequence} logprob record {position} is invalid",
        )
        record = dict(record_value)
        _require(
            set(record)
            == {
                "position",
                "selected_token_id",
                "selected_logprob",
                "top_logprobs",
            }
            and record.get("position") == position
            and record.get("selected_token_id") == selected_token,
            f"quality request {sequence} selected token record differs",
        )
        selected_logprob = record.get("selected_logprob")
        _require(
            type(selected_logprob) in (int, float)
            and math.isfinite(float(selected_logprob))
            and float(selected_logprob) <= 0,
            f"quality request {sequence} selected logprob is invalid",
        )
        top = record.get("top_logprobs")
        _require(
            isinstance(top, list) and len(top) == QUALITY_TOP_LOGPROBS,
            f"quality request {sequence} top-logprob inventory differs",
        )
        normalized_top: list[dict[str, object]] = []
        for candidate in top:
            _require(
                isinstance(candidate, dict)
                and set(candidate) == {"token_id", "logprob"}
                and type(candidate.get("token_id")) is int
                and candidate["token_id"] >= 0
                and type(candidate.get("logprob")) in (int, float)
                and math.isfinite(float(candidate["logprob"]))
                and float(candidate["logprob"]) <= 0,
                f"quality request {sequence} top-logprob entry is invalid",
            )
            normalized_top.append(
                {
                    "token_id": int(candidate["token_id"]),
                    "logprob": float(candidate["logprob"]),
                }
            )
        _require(
            len({candidate["token_id"] for candidate in normalized_top})
            == QUALITY_TOP_LOGPROBS,
            f"quality request {sequence} top-logprob token ids are duplicated",
        )
        expected_order = sorted(
            normalized_top,
            key=lambda candidate: (
                -candidate["logprob"],
                candidate["token_id"],
            ),
        )
        _require(
            normalized_top == expected_order,
            f"quality request {sequence} top-logprobs are not ordered",
        )
        by_token = {
            candidate["token_id"]: candidate["logprob"]
            for candidate in normalized_top
        }
        _require(
            selected_token in by_token
            and by_token[selected_token] == float(selected_logprob),
            f"quality request {sequence} selected logprob differs from top-64",
        )
        records.append(
            {
                **record,
                "selected_logprob": float(selected_logprob),
                "top_logprobs": normalized_top,
            }
        )
    return records


def _validate_quality_request(
    row: Mapping[str, object],
    sequence: int,
) -> dict[str, object]:
    session, turn = _request_identity(sequence)
    _require(
        row.get("row_type") == "quality_request"
        and row.get("sequence") == sequence
        and row.get("session") == session
        and row.get("turn") == turn,
        f"quality request {sequence} identity/order differs",
    )
    _require(
        row.get("status") == "success" and row.get("retry_count") == 0,
        f"quality request {sequence} failed or retried",
    )
    prompt = row.get("prompt_token_ids")
    expected_prompt_tokens = SHARED_TOKENS + (turn + 1) * TURN_TOKENS
    _require(
        isinstance(prompt, list)
        and len(prompt) == expected_prompt_tokens
        and all(type(token) is int and token >= 0 for token in prompt)
        and row.get("prompt_tokens") == len(prompt)
        and row.get("prompt_token_ids_sha256") == sha256_json(prompt),
        f"quality request {sequence} prompt identity differs",
    )
    output = row.get("output_token_ids")
    _require(
        isinstance(output, list)
        and len(output) == OUTPUT_TOKENS
        and all(type(token) is int and token >= 0 for token in output),
        f"quality request {sequence} output differs",
    )
    usage = row.get("usage")
    _require(
        isinstance(usage, dict)
        and usage.get("prompt_tokens") == expected_prompt_tokens
        and type(usage.get("cached_tokens")) is int
        and 0 <= usage["cached_tokens"] <= expected_prompt_tokens
        and usage.get("completion_tokens") == OUTPUT_TOKENS,
        f"quality request {sequence} engine usage is invalid",
    )
    initial = _tier_stats(
        row.get("initial_tier_stats"),
        f"quality request {sequence} initial",
    )
    final = _tier_stats(
        row.get("final_tier_stats"),
        f"quality request {sequence} final",
    )
    records = _validate_quality_logprob_records(
        row.get("logprob_records"),
        output,
        sequence=sequence,
    )
    return {
        **dict(row),
        "initial_tier_stats": initial,
        "final_tier_stats": final,
        "logprob_records": records,
    }


def parse_quality_shard(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    _require(
        len(rows) == REQUESTS_PER_SHARD + 2,
        "quality shard row count differs",
    )
    header = dict(rows[0])
    end = dict(rows[-1])
    _require(
        header.get("row_type") == "quality_run"
        and end.get("row_type") == "quality_run_end",
        "quality shard framing differs",
    )
    _require(
        header.get("schema_version") == QUALITY_SCHEMA_VERSION
        and header.get("measurement_kind") == QUALITY_MEASUREMENT_KIND
        and header.get("config") == quality_config(),
        "quality shard schema/config differs",
    )
    identity = (header.get("arm"), header.get("cohort"))
    _require(identity in QUALITY_PLAN, "quality shard identity is outside fixed plan")
    arm, cohort = identity
    _require(
        _valid_sha256(header.get("parent_performance_raw_sha256")),
        "quality parent performance raw digest is invalid",
    )
    _require(
        isinstance(header.get("run_id"), str)
        and bool(header["run_id"])
        and isinstance(header.get("output_path"), str)
        and Path(header["output_path"]).is_absolute(),
        "quality raw output/run identity is invalid",
    )
    source = _validate_source(header.get("source"))
    checkpoint = _validate_checkpoint(
        header.get("checkpoint"),
        "F4b quality checkpoint",
    )
    container = _validate_container(header.get("container"))
    profile = _validate_profile(
        header.get("profile"),
        arm=arm,
        engine_source_sha256=source["engine_source_sha256"],
    )
    _validate_quality_run_command(
        header,
        container=container,
        checkpoint=checkpoint,
        profile=profile,
    )
    hardware = _validate_hardware(header.get("hardware"), cohort)
    runtime = _validate_runtime(
        header.get("runtime"),
        arm=arm,
        source=source,
        profile=profile,
    )
    trace = _validate_trace(header.get("trace"))
    start_ns = _exact_int(
        header.get("measurement_started_unix_ns"),
        "quality measurement start",
        minimum=1,
    )
    end_ns = _exact_int(
        end.get("measurement_ended_unix_ns"),
        "quality measurement end",
        minimum=1,
    )
    _require(end_ns > start_ns, "quality measurement interval is empty or reversed")
    _require(
        end.get("run_id") == header["run_id"]
        and (end.get("arm"), end.get("cohort")) == identity
        and end.get("status") == "complete"
        and end.get("errors") == []
        and end.get("request_count") == REQUESTS_PER_SHARD
        and end.get("source") == source,
        "quality run_end identity/status/provenance differs",
    )
    requests = [
        _validate_quality_request(row, sequence)
        for sequence, row in enumerate(rows[1:-1])
    ]
    previous_final: Mapping[str, int] | None = None
    for request in requests:
        initial_stats = request["initial_tier_stats"]
        final_stats = request["final_tier_stats"]
        _require(
            previous_final is None or initial_stats == previous_final,
            "quality cumulative tier stats are discontinuous between requests",
        )
        _require(
            all(final_stats[field] >= initial_stats[field] for field in TIER_STATS_FIELDS),
            "quality cumulative tier stats decreased within a request",
        )
        previous_final = final_stats
    _require(
        _trace_rows_from_requests(requests) == trace["value"]["rows"],
        "quality requests differ from the bound trace",
    )
    shared = requests[0]["prompt_token_ids"][:SHARED_TOKENS]
    _require(
        sha256_json(shared) == trace["value"]["shared_prefix_sha256"],
        "quality shared prefix differs from its trace",
    )
    for request in requests:
        _require(
            request["prompt_token_ids"][:SHARED_TOKENS] == shared,
            "quality shared prefix differs between requests",
        )
        if request["turn"]:
            previous = requests[
                (request["turn"] - 1) * SESSIONS + request["session"]
            ]
            _require(
                request["prompt_token_ids"][
                    : len(previous["prompt_token_ids"])
                ]
                == previous["prompt_token_ids"],
                "quality session prompt is not append-only",
            )
    _require(
        end.get("final_tier_stats") == requests[-1]["final_tier_stats"],
        "quality run_end tier stats differ",
    )
    return {
        "identity": identity,
        "header": {
            **header,
            "source": source,
            "checkpoint": checkpoint,
            "container": container,
            "profile": profile,
            "hardware": hardware,
            "runtime": runtime,
            "trace": trace,
        },
        "requests": requests,
        "end": end,
        "interval": (start_ns, end_ns),
    }


def _metadata_relative_paths(
    *,
    plan: Sequence[tuple[object, ...]] = SHARD_PLAN,
    labels: Mapping[tuple[object, ...], str] = SHARD_LABELS,
) -> tuple[str, ...]:
    paths = [IMAGE_INSPECT_NAME]
    for identity in plan:
        label = labels[identity]
        paths.extend(
            (
                f"{label}/{CONTAINER_ID_NAME}",
                f"{label}/{CONTAINER_CREATED_NAME}",
                f"{label}/{CONTAINER_EXITED_NAME}",
            )
        )
    return tuple(paths)


def _metadata_file_value(path: Path) -> object:
    try:
        payload = path.read_bytes()
        if path.suffix == ".json":
            return json.loads(payload)
        return payload.decode("utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read container metadata: {path}") from error


def _metadata_descriptor(metadata_dir: Path, relative_path: str) -> dict[str, object]:
    path = metadata_dir / relative_path
    _require(
        path.is_file() and not path.is_symlink(),
        f"container metadata is missing: {relative_path}",
    )
    value = _metadata_file_value(path)
    return {
        "relative_path": relative_path,
        "file_sha256": sha256_file(path),
        "value_sha256": sha256_json(value),
        "value": value,
    }


def _load_container_lifecycle(
    metadata_dir: Path,
    *,
    plan: Sequence[tuple[object, ...]] = SHARD_PLAN,
    labels: Mapping[tuple[object, ...], str] = SHARD_LABELS,
) -> dict[str, object]:
    _require(metadata_dir.is_dir(), f"container metadata directory is missing: {metadata_dir}")
    expected_paths = set(_metadata_relative_paths(plan=plan, labels=labels))
    actual_paths = {
        path.relative_to(metadata_dir).as_posix()
        for path in metadata_dir.rglob("*")
        if path.is_file()
    }
    _require(actual_paths == expected_paths, "container metadata fixed layout differs")
    shards = []
    for identity in plan:
        label = labels[identity]
        shards.append(
            {
                "identity": list(identity),
                "label": label,
                "container_id": _metadata_descriptor(metadata_dir, f"{label}/{CONTAINER_ID_NAME}"),
                "created_inspect": _metadata_descriptor(
                    metadata_dir, f"{label}/{CONTAINER_CREATED_NAME}"
                ),
                "exited_inspect": _metadata_descriptor(
                    metadata_dir, f"{label}/{CONTAINER_EXITED_NAME}"
                ),
            }
        )
    return {
        "schema_version": 1,
        "image_inspect": _metadata_descriptor(metadata_dir, IMAGE_INSPECT_NAME),
        "shards": shards,
    }


def _validate_metadata_descriptor(
    value: object,
    *,
    relative_path: str,
) -> dict[str, object]:
    _require(isinstance(value, dict), f"metadata descriptor is missing: {relative_path}")
    descriptor = dict(value)
    _require(
        set(descriptor) == {"relative_path", "file_sha256", "value_sha256", "value"},
        f"metadata descriptor fields differ: {relative_path}",
    )
    _require(descriptor.get("relative_path") == relative_path, "metadata relative path differs")
    _require(_valid_sha256(descriptor.get("file_sha256")), "metadata file SHA-256 is invalid")
    _require(
        _valid_sha256(descriptor.get("value_sha256"))
        and descriptor["value_sha256"] == sha256_json(descriptor.get("value")),
        "metadata value SHA-256 differs",
    )
    return descriptor


def _single_docker_inspect(descriptor: Mapping[str, object], name: str) -> dict[str, object]:
    value = descriptor["value"]
    _require(
        isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict),
        f"{name} must contain exactly one Docker inspect object",
    )
    return dict(value[0])


def _rfc3339_unix_ns(value: object, name: str) -> int:
    _require(isinstance(value, str), f"{name} timestamp is missing")
    match = _RFC3339_NS.fullmatch(value)
    _require(match is not None, f"{name} timestamp is not RFC3339 nanoseconds")
    assert match is not None
    offset = "+00:00" if match.group("offset") == "Z" else match.group("offset")
    try:
        parsed = datetime.fromisoformat(f"{match.group('date')}{offset}").astimezone(UTC)
    except ValueError as error:
        raise EvidenceError(f"{name} timestamp is invalid") from error
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = parsed - epoch
    seconds = delta.days * 86_400 + delta.seconds
    fraction = (match.group("fraction") or "0").ljust(9, "0")
    return seconds * 1_000_000_000 + int(fraction)


def _docker_mapping(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"Docker inspect {name} is not an object")
    return dict(value)


def _normalized_host_config(value: object, name: str) -> dict[str, object]:
    host_config = _docker_mapping(value, name)
    # Docker normalizes this inspect field from false immediately after create
    # to null after start/exit.  Both mean that OOM killing was not disabled.
    if host_config.get("OomKillDisable") is False or host_config.get("OomKillDisable") is None:
        host_config["OomKillDisable"] = None
    return host_config


def _calibrated_execution_image_identity(
    lifecycle: Mapping[str, object],
) -> dict[str, object]:
    image_descriptor = _validate_metadata_descriptor(
        lifecycle.get("image_inspect"), relative_path=IMAGE_INSPECT_NAME
    )
    image = _single_docker_inspect(image_descriptor, "image inspect")
    image_id = image.get("Id")
    _require(
        image_id == F4A_CALIBRATED_IMAGE_ID,
        "Docker image inspect ID differs from the calibrated F4a image",
    )
    image_config = _docker_mapping(image.get("Config"), "image Config")
    labels = _docker_mapping(image_config.get("Labels"), "image labels")
    revision = labels.get("org.opencontainers.image.revision")
    _require(
        _valid_commit(revision) and revision == F4A_CALIBRATED_IMAGE_REVISION,
        "Docker image revision differs from the calibrated F4a image",
    )
    repo_digests = image.get("RepoDigests")
    _require(
        isinstance(repo_digests, list)
        and bool(repo_digests)
        and all(
            isinstance(item, str)
            and "@sha256:" in item
            and _valid_sha256(item.rsplit("@sha256:", 1)[-1])
            for item in repo_digests
        )
        and len(set(repo_digests)) == len(repo_digests),
        "Docker image repository digests are invalid",
    )
    return {
        "image_id": image_id,
        "revision": revision,
        "repo_digests": list(repo_digests),
        "image_inspect_file_sha256": image_descriptor["file_sha256"],
    }


def _container_mount_inventory(value: object, label: str) -> dict[str, dict[str, object]]:
    _require(isinstance(value, list), f"Docker Mounts are invalid: {label}")
    mounts: dict[str, dict[str, object]] = {}
    for item in value:
        mount = _docker_mapping(item, f"{label} Mount")
        destination = mount.get("Destination")
        source = mount.get("Source")
        _require(
            isinstance(destination, str)
            and destination.startswith("/")
            and destination not in mounts
            and isinstance(source, str)
            and source.startswith("/"),
            f"Docker mount identity is invalid: {label}",
        )
        mounts[destination] = {
            "type": mount.get("Type"),
            "source": source,
            "rw": mount.get("RW"),
        }
    expected = {
        "/workspace/kairyu": {"type": "bind", "rw": False},
        "/models/qwen3-32b": {"type": "volume", "rw": False},
        "/evidence": {"type": "bind", "rw": True},
        "/run/kairyu-meta": {"type": "bind", "rw": False},
    }
    _require(set(mounts) == set(expected), f"Docker mount destinations differ: {label}")
    _require(
        all(
            mounts[destination]["type"] == contract["type"]
            and mounts[destination]["rw"] is contract["rw"]
            for destination, contract in expected.items()
        ),
        f"Docker mount type/access differs: {label}",
    )
    return mounts


def _container_env_inventory(value: object, label: str) -> dict[str, str]:
    _require(isinstance(value, list), f"Docker environment is invalid: {label}")
    environment: dict[str, str] = {}
    for item in value:
        _require(
            isinstance(item, str) and "=" in item,
            f"Docker environment entry is invalid: {label}",
        )
        name, entry_value = item.split("=", 1)
        _require(
            bool(name) and name not in environment,
            f"Docker environment name is invalid or duplicated: {label}",
        )
        environment[name] = entry_value
    required = {
        "GLOO_SOCKET_IFNAME": "lo",
        "TRITON_CACHE_DIR": "/evidence/triton-cache",
        "FLASHINFER_WORKSPACE_BASE": "/evidence/flashinfer-workspace",
    }
    _require(
        all(environment.get(name) == expected for name, expected in required.items()),
        f"Docker per-arm cache environment differs: {label}",
    )
    return environment


def _validate_container_lifecycle(
    value: object,
    shards: Mapping[tuple[object, ...], Mapping[str, object]],
    *,
    plan: Sequence[tuple[object, ...]] = SHARD_PLAN,
    labels: Mapping[tuple[object, ...], str] = SHARD_LABELS,
    cohort_index: int = 2,
) -> dict[str, object]:
    _require(isinstance(value, dict), "container lifecycle metadata is missing")
    lifecycle = dict(value)
    _require(
        set(lifecycle) == {"schema_version", "image_inspect", "shards"}
        and lifecycle.get("schema_version") == 1,
        "container lifecycle metadata fields/schema differ",
    )
    execution_image = _calibrated_execution_image_identity(lifecycle)
    image_id = execution_image["image_id"]
    image_repo_digests = execution_image["repo_digests"]

    entries = lifecycle.get("shards")
    _require(
        isinstance(entries, list) and len(entries) == len(plan),
        "container lifecycle shard inventory differs",
    )
    previous_finished_ns: int | None = None
    bound_source_host_path: str | None = None
    model_volume_host_path: str | None = None
    evidence_host_paths: set[str] = set()
    metadata_host_paths: set[str] = set()
    for index, identity in enumerate(plan):
        label = labels[identity]
        entry_value = entries[index]
        _require(isinstance(entry_value, dict), f"container lifecycle entry differs: {label}")
        entry = dict(entry_value)
        _require(
            set(entry) == {"identity", "label", "container_id", "created_inspect", "exited_inspect"}
            and entry.get("identity") == list(identity)
            and entry.get("label") == label,
            f"container lifecycle identity/order differs: {label}",
        )
        identifier_descriptor = _validate_metadata_descriptor(
            entry.get("container_id"), relative_path=f"{label}/{CONTAINER_ID_NAME}"
        )
        created_descriptor = _validate_metadata_descriptor(
            entry.get("created_inspect"), relative_path=f"{label}/{CONTAINER_CREATED_NAME}"
        )
        exited_descriptor = _validate_metadata_descriptor(
            entry.get("exited_inspect"), relative_path=f"{label}/{CONTAINER_EXITED_NAME}"
        )
        identifier_value = identifier_descriptor["value"]
        _require(isinstance(identifier_value, str), f"container id file is not text: {label}")
        container_id = identifier_value.strip()
        created = _single_docker_inspect(created_descriptor, f"{label} created inspect")
        exited = _single_docker_inspect(exited_descriptor, f"{label} exited inspect")
        shard = shards[identity]
        header = shard["header"]
        container = header["container"]
        _require(
            _valid_sha256(container_id)
            and container_id == container["container_id"]
            and created.get("Id") == container_id
            and exited.get("Id") == container_id,
            f"Docker container ID differs: {label}",
        )
        for field in ("Id", "Image", "Created", "Path", "Args", "Config"):
            _require(
                field in created and created[field] == exited.get(field),
                f"Docker static container field differs at exit: {label} {field}",
            )
        _require(
            created.get("Image") == image_id == container["image_id"],
            f"Docker container image differs: {label}",
        )
        _require(
            container["repo_digests"] == image_repo_digests,
            f"Docker repository digests differ: {label}",
        )
        config = _docker_mapping(created.get("Config"), f"{label} Config")
        host_config = _normalized_host_config(created.get("HostConfig"), f"{label} HostConfig")
        exited_host_config = _normalized_host_config(
            exited.get("HostConfig"), f"{label} exited HostConfig"
        )
        _require(
            host_config == exited_host_config,
            f"Docker static container field differs at exit: {label} HostConfig",
        )
        command = container["command"]
        _require(
            created.get("Path") == "/app/.venv/bin/python"
            and created.get("Args") == command
            and config.get("Entrypoint") == ["/app/.venv/bin/python"]
            and config.get("Cmd") == command
            and config.get("Image") == image_id,
            f"Docker command/image configuration differs: {label}",
        )
        _require(
            config.get("Hostname") == container_id[:12]
            and container.get("hostname") == container_id[:12],
            f"Docker hostname differs from full container ID: {label}",
        )
        _require(
            config.get("WorkingDir") == "/workspace/kairyu"
            and isinstance(config.get("User"), str)
            and re.fullmatch(r"[1-9][0-9]*(?::[1-9][0-9]*)?", config["User"]) is not None,
            f"Docker bound-source working directory/non-root user differs: {label}",
        )
        _container_env_inventory(config.get("Env"), label)
        created_mounts = _container_mount_inventory(created.get("Mounts"), f"{label} created")
        exited_mounts = _container_mount_inventory(exited.get("Mounts"), f"{label} exited")
        _require(
            created_mounts == exited_mounts,
            f"Docker mount identity changed during execution: {label}",
        )
        source_host_path = created_mounts["/workspace/kairyu"]["source"]
        model_host_path = created_mounts["/models/qwen3-32b"]["source"]
        evidence_host_paths.add(str(created_mounts["/evidence"]["source"]))
        metadata_host_paths.add(str(created_mounts["/run/kairyu-meta"]["source"]))
        if bound_source_host_path is None:
            bound_source_host_path = str(source_host_path)
            model_volume_host_path = str(model_host_path)
        else:
            _require(
                source_host_path == bound_source_host_path
                and model_host_path == model_volume_host_path,
                f"Docker shared source/model carrier differs across shards: {label}",
            )
        expected_device_request = [
            {
                "Driver": "",
                "Count": 0,
                "DeviceIDs": list(COHORT_DEVICE_IDS[str(identity[cohort_index])]),
                "Capabilities": [["gpu"]],
                "Options": {},
            }
        ]
        _require(
            host_config.get("DeviceRequests") == expected_device_request,
            f"Docker GPU DeviceRequests differ from fixed cohort: {label}",
        )
        network_mode = host_config.get("NetworkMode")
        _require(
            isinstance(network_mode, str) and bool(network_mode) and network_mode != "host",
            f"Docker container uses invalid host networking: {label}",
        )
        created_state = _docker_mapping(created.get("State"), f"{label} created State")
        exited_state = _docker_mapping(exited.get("State"), f"{label} exited State")
        _require(
            created_state.get("Status") == "created"
            and created_state.get("Running") is False
            and created_state.get("ExitCode") == 0
            and created_state.get("OOMKilled") is False
            and created_state.get("Error") == ""
            and created_state.get("StartedAt") == _DOCKER_ZERO_TIMESTAMP
            and created_state.get("FinishedAt") == _DOCKER_ZERO_TIMESTAMP,
            f"Docker created state differs: {label}",
        )
        _require(
            exited_state.get("Status") == "exited"
            and exited_state.get("Running") is False
            and exited_state.get("ExitCode") == 0
            and exited_state.get("OOMKilled") is False
            and exited_state.get("Error") == "",
            f"Docker exited state is not a clean exit: {label}",
        )
        created_ns = _rfc3339_unix_ns(created.get("Created"), f"{label} created")
        started_ns = _rfc3339_unix_ns(exited_state.get("StartedAt"), f"{label} started")
        finished_ns = _rfc3339_unix_ns(exited_state.get("FinishedAt"), f"{label} finished")
        measurement_start_ns, measurement_end_ns = shard["interval"]
        _require(
            created_ns <= started_ns <= measurement_start_ns < measurement_end_ns <= finished_ns,
            f"Docker lifecycle does not contain measurement interval: {label}",
        )
        if previous_finished_ns is not None:
            _require(
                previous_finished_ns <= created_ns <= started_ns,
                f"Docker container lifecycles overlap fixed shard order: {label}",
            )
        previous_finished_ns = finished_ns
    _require(
        len(evidence_host_paths) == len(plan)
        and len(metadata_host_paths) == len(plan),
        "Docker per-arm cache/evidence metadata mounts are not isolated",
    )
    return lifecycle


def _metadata_digest_map(lifecycle: Mapping[str, object]) -> dict[str, str]:
    descriptors = [lifecycle["image_inspect"]]
    for entry in lifecycle["shards"]:
        descriptors.extend(
            (entry["container_id"], entry["created_inspect"], entry["exited_inspect"])
        )
    return {descriptor["relative_path"]: descriptor["file_sha256"] for descriptor in descriptors}


def _container_lifecycle_bounds(
    lifecycle: Mapping[str, object],
) -> tuple[int, int]:
    entries = lifecycle["shards"]
    _require(isinstance(entries, list) and entries, "container lifecycle is empty")
    first = _single_docker_inspect(entries[0]["created_inspect"], "first created inspect")
    last = _single_docker_inspect(entries[-1]["exited_inspect"], "last exited inspect")
    last_state = _docker_mapping(last.get("State"), "last exited State")
    return (
        _rfc3339_unix_ns(first.get("Created"), "first container created"),
        _rfc3339_unix_ns(last_state.get("FinishedAt"), "last container finished"),
    )


def assemble_rows(
    paths: Sequence[Path],
    *,
    metadata_dir: Path,
) -> list[dict[str, object]]:
    _require(len(paths) == len(SHARD_PLAN), "assemble requires exactly four raw shards")
    shards: dict[tuple[object, object, object], tuple[dict[str, object], str]] = {}
    for path in paths:
        parsed = parse_shard(read_jsonl(path))
        identity = tuple(parsed["identity"])
        _require(identity not in shards, f"duplicate shard {identity}")
        shards[identity] = (parsed, sha256_file(path))
    _require(set(shards) == set(SHARD_PLAN), "four-shard crossover plan is incomplete")
    parsed_shards = {identity: parsed for identity, (parsed, _digest) in shards.items()}
    lifecycle = _validate_container_lifecycle(
        _load_container_lifecycle(metadata_dir), parsed_shards
    )
    combined: list[dict[str, object]] = [
        {
            "row_type": "run",
            "schema_version": SCHEMA_VERSION,
            "measurement_kind": MEASUREMENT_KIND,
            "config": formal_config(),
            "container_lifecycle": lifecycle,
        }
    ]
    for identity in SHARD_PLAN:
        parsed, digest = shards[identity]
        header = parsed["header"]
        combined.append(
            {
                **{
                    key: value
                    for key, value in header.items()
                    if key not in {"row_type", "row_index"}
                },
                "row_type": "shard_start",
                "input_raw_sha256": digest,
            }
        )
        combined.extend(
            {key: value for key, value in request.items() if key not in {"row_index", "tpot_ns"}}
            for request in parsed["requests"]
        )
        combined.append(
            {
                **{
                    key: value
                    for key, value in parsed["end"].items()
                    if key not in {"row_type", "row_index"}
                },
                "row_type": "shard_end",
            }
        )
    combined.append(
        {
            "row_type": "run_end",
            "shard_count": len(SHARD_PLAN),
            "request_count": len(SHARD_PLAN) * REQUESTS_PER_SHARD,
        }
    )
    return combined


def _parse_combined(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[int, str, str], dict[str, object]],
    dict[str, object],
]:
    _require(
        len(rows) >= 2
        and rows[0].get("row_type") == "run"
        and rows[-1].get("row_type") == "run_end",
        "combined raw framing differs",
    )
    _require(
        rows[0].get("schema_version") == SCHEMA_VERSION
        and rows[0].get("measurement_kind") == MEASUREMENT_KIND
        and rows[0].get("config") == formal_config(),
        "combined raw header differs",
    )
    _require(
        rows[-1].get("shard_count") == len(SHARD_PLAN)
        and rows[-1].get("request_count") == len(SHARD_PLAN) * REQUESTS_PER_SHARD,
        "combined raw end counts differ",
    )
    shards: dict[tuple[int, str, str], dict[str, object]] = {}
    cursor = 1
    for expected_identity in SHARD_PLAN:
        _require(
            cursor < len(rows) - 1 and rows[cursor].get("row_type") == "shard_start",
            "combined raw shard start is missing",
        )
        start = dict(rows[cursor])
        cursor += 1
        requests = [dict(row) for row in rows[cursor : cursor + REQUESTS_PER_SHARD]]
        cursor += REQUESTS_PER_SHARD
        _require(
            cursor < len(rows) - 1 and rows[cursor].get("row_type") == "shard_end",
            "combined raw shard end is missing",
        )
        end = dict(rows[cursor])
        cursor += 1
        identity = (start.get("round"), start.get("arm"), start.get("cohort"))
        _require(identity == expected_identity, "combined raw shard plan order differs")
        raw_header = {**start, "row_type": "run"}
        raw_header.pop("input_raw_sha256", None)
        raw_rows = [raw_header, *requests, {**end, "row_type": "run_end"}]
        parsed = parse_shard(raw_rows)
        _require(_valid_sha256(start.get("input_raw_sha256")), "input shard digest is invalid")
        shards[expected_identity] = parsed
    _require(cursor == len(rows) - 1, "combined raw contains unexpected rows")
    lifecycle = _validate_container_lifecycle(rows[0].get("container_lifecycle"), shards)
    return shards, lifecycle


def assemble_quality_rows(
    paths: Sequence[Path],
    *,
    parent_performance_raw_sha256: str,
    metadata_dir: Path,
) -> list[dict[str, object]]:
    _require(
        len(paths) == len(QUALITY_PLAN),
        "quality assemble requires exactly two raw shards",
    )
    _require(
        _valid_sha256(parent_performance_raw_sha256),
        "quality parent performance raw digest is invalid",
    )
    shards: dict[
        tuple[object, object],
        tuple[dict[str, object], str],
    ] = {}
    for path in paths:
        parsed = parse_quality_shard(read_jsonl(path))
        identity = tuple(parsed["identity"])
        _require(identity not in shards, f"duplicate quality shard {identity}")
        shards[identity] = (parsed, sha256_file(path))
    _require(
        set(shards) == set(QUALITY_PLAN),
        "two-shard quality plan is incomplete",
    )
    _require(
        all(
            parsed["header"]["parent_performance_raw_sha256"]
            == parent_performance_raw_sha256
            for parsed, _digest in shards.values()
        ),
        "quality shard parent performance raw digest differs",
    )
    parsed_shards = {
        identity: parsed for identity, (parsed, _digest) in shards.items()
    }
    lifecycle = _validate_container_lifecycle(
        _load_container_lifecycle(
            metadata_dir,
            plan=QUALITY_PLAN,
            labels=QUALITY_SHARD_LABELS,
        ),
        parsed_shards,
        plan=QUALITY_PLAN,
        labels=QUALITY_SHARD_LABELS,
        cohort_index=1,
    )
    combined: list[dict[str, object]] = [
        {
            "row_type": "quality_run",
            "schema_version": QUALITY_SCHEMA_VERSION,
            "measurement_kind": QUALITY_MEASUREMENT_KIND,
            "config": quality_config(),
            "parent_performance_raw_sha256": parent_performance_raw_sha256,
            "container_lifecycle": lifecycle,
        }
    ]
    for identity in QUALITY_PLAN:
        parsed, digest = shards[identity]
        header = parsed["header"]
        combined.append(
            {
                **{
                    key: value
                    for key, value in header.items()
                    if key not in {"row_type", "row_index"}
                },
                "row_type": "quality_shard_start",
                "input_raw_sha256": digest,
            }
        )
        combined.extend(
            {
                key: value
                for key, value in request.items()
                if key != "row_index"
            }
            for request in parsed["requests"]
        )
        combined.append(
            {
                **{
                    key: value
                    for key, value in parsed["end"].items()
                    if key not in {"row_type", "row_index"}
                },
                "row_type": "quality_shard_end",
            }
        )
    combined.append(
        {
            "row_type": "quality_run_end",
            "shard_count": len(QUALITY_PLAN),
            "request_count": len(QUALITY_PLAN) * REQUESTS_PER_SHARD,
        }
    )
    return combined


def _parse_quality_combined(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    dict[str, object],
]:
    _require(
        len(rows) >= 2
        and rows[0].get("row_type") == "quality_run"
        and rows[-1].get("row_type") == "quality_run_end",
        "combined quality raw framing differs",
    )
    _require(
        rows[0].get("schema_version") == QUALITY_SCHEMA_VERSION
        and rows[0].get("measurement_kind") == QUALITY_MEASUREMENT_KIND
        and rows[0].get("config") == quality_config()
        and _valid_sha256(rows[0].get("parent_performance_raw_sha256")),
        "combined quality raw header differs",
    )
    _require(
        rows[-1].get("shard_count") == len(QUALITY_PLAN)
        and rows[-1].get("request_count")
        == len(QUALITY_PLAN) * REQUESTS_PER_SHARD,
        "combined quality raw end counts differ",
    )
    parent_digest = rows[0]["parent_performance_raw_sha256"]
    shards: dict[tuple[str, str], dict[str, object]] = {}
    cursor = 1
    for expected_identity in QUALITY_PLAN:
        _require(
            cursor < len(rows) - 1
            and rows[cursor].get("row_type") == "quality_shard_start",
            "combined quality shard start is missing",
        )
        start = dict(rows[cursor])
        cursor += 1
        requests = [
            dict(row)
            for row in rows[cursor : cursor + REQUESTS_PER_SHARD]
        ]
        cursor += REQUESTS_PER_SHARD
        _require(
            cursor < len(rows) - 1
            and rows[cursor].get("row_type") == "quality_shard_end",
            "combined quality shard end is missing",
        )
        end = dict(rows[cursor])
        cursor += 1
        identity = (start.get("arm"), start.get("cohort"))
        _require(
            identity == expected_identity,
            "combined quality shard plan order differs",
        )
        raw_header = {**start, "row_type": "quality_run"}
        raw_header.pop("input_raw_sha256", None)
        raw_rows = [
            raw_header,
            *requests,
            {**end, "row_type": "quality_run_end"},
        ]
        parsed = parse_quality_shard(raw_rows)
        _require(
            _valid_sha256(start.get("input_raw_sha256")),
            "quality input shard digest is invalid",
        )
        _require(
            parsed["header"]["parent_performance_raw_sha256"]
            == parent_digest,
            "quality shard parent raw digest differs from combined header",
        )
        shards[expected_identity] = parsed
    _require(
        cursor == len(rows) - 1,
        "combined quality raw contains unexpected rows",
    )
    lifecycle = _validate_container_lifecycle(
        rows[0].get("container_lifecycle"),
        shards,
        plan=QUALITY_PLAN,
        labels=QUALITY_SHARD_LABELS,
        cohort_index=1,
    )
    return shards, lifecycle


def _quality_distribution_comparison(
    off_requests: Sequence[Mapping[str, object]],
    on_requests: Sequence[Mapping[str, object]],
) -> tuple[dict[str, bool], dict[str, object]]:
    aligned_deltas: list[float] = []
    aligned_failures: list[dict[str, object]] = []
    first_divergence_deltas: list[float] = []
    first_divergences: list[dict[str, object]] = []
    missing_cross_selected: list[dict[str, object]] = []
    first_divergence_failures: list[dict[str, object]] = []
    for off_request, on_request in zip(
        off_requests,
        on_requests,
        strict=True,
    ):
        for position, (
            off_token,
            on_token,
            off_record,
            on_record,
        ) in enumerate(
            zip(
                off_request["output_token_ids"],
                on_request["output_token_ids"],
                off_request["logprob_records"],
                on_request["logprob_records"],
                strict=True,
            )
        ):
            off_distribution = {
                candidate["token_id"]: candidate["logprob"]
                for candidate in off_record["top_logprobs"]
            }
            on_distribution = {
                candidate["token_id"]: candidate["logprob"]
                for candidate in on_record["top_logprobs"]
            }
            if off_token == on_token:
                delta = abs(
                    float(off_record["selected_logprob"])
                    - float(on_record["selected_logprob"])
                )
                aligned_deltas.append(delta)
                if delta > QUALITY_LOGPROB_ABS_TOLERANCE:
                    aligned_failures.append(
                        {
                            "sequence": off_request["sequence"],
                            "position": position,
                            "token_id": off_token,
                            "absolute_delta": delta,
                        }
                    )
                continue

            divergence = {
                "sequence": off_request["sequence"],
                "session": off_request["session"],
                "turn": off_request["turn"],
                "position": position,
                "off_token_id": off_token,
                "on_token_id": on_token,
            }
            first_divergences.append(divergence)
            cross_views = (
                (
                    "off_selected_seen_by_on",
                    off_token,
                    float(off_record["selected_logprob"]),
                    on_distribution.get(off_token),
                ),
                (
                    "on_selected_seen_by_off",
                    on_token,
                    float(on_record["selected_logprob"]),
                    off_distribution.get(on_token),
                ),
            )
            for view, token_id, owner_logprob, other_logprob in cross_views:
                if other_logprob is None:
                    missing_cross_selected.append(
                        {
                            **divergence,
                            "view": view,
                            "token_id": token_id,
                        }
                    )
                    continue
                delta = abs(owner_logprob - float(other_logprob))
                first_divergence_deltas.append(delta)
                if delta > QUALITY_LOGPROB_ABS_TOLERANCE:
                    first_divergence_failures.append(
                        {
                            **divergence,
                            "view": view,
                            "token_id": token_id,
                            "owner_logprob": owner_logprob,
                            "other_arm_logprob": float(other_logprob),
                            "absolute_delta": delta,
                        }
                    )
            # Later positions are conditioned on different generated prefixes.
            break
    checks = {
        "aligned_prefix_selected_logprob_abs_delta_at_most_0_25": (
            not aligned_failures
        ),
        "first_divergence_reciprocal_cross_selected_top64_complete": (
            not missing_cross_selected
        ),
        "first_divergence_reciprocal_cross_selected_abs_delta_at_most_0_25": (
            not missing_cross_selected and not first_divergence_failures
        ),
    }
    diagnostics = {
        "aligned_prefix_position_count": len(aligned_deltas),
        "aligned_prefix_selected_logprob_max_abs_delta": max(
            aligned_deltas,
            default=None,
        ),
        "aligned_prefix_failures": aligned_failures,
        "first_divergence_count": len(first_divergences),
        "first_divergences": first_divergences,
        "first_divergence_cross_selected_comparison_count": len(
            first_divergence_deltas
        ),
        "first_divergence_cross_selected_max_abs_delta": max(
            first_divergence_deltas,
            default=None,
        ),
        "missing_cross_selected": missing_cross_selected,
        "first_divergence_failures": first_divergence_failures,
        "post_divergence_distribution_comparison_performed": False,
        "free_running_output_equality_is_binding": False,
    }
    return checks, diagnostics


def recompute_quality_manifest(
    performance_rows: Sequence[Mapping[str, object]],
    quality_rows: Sequence[Mapping[str, object]],
    *,
    performance_raw_sha256: str,
    quality_raw_sha256: str,
) -> dict[str, object]:
    _require(
        _valid_sha256(performance_raw_sha256),
        "parent performance combined raw digest is invalid",
    )
    _require(
        _valid_sha256(quality_raw_sha256),
        "quality combined raw digest is invalid",
    )
    performance_manifest = recompute_manifest(
        performance_rows,
        raw_sha256=performance_raw_sha256,
    )
    performance_shards, performance_lifecycle = _parse_combined(performance_rows)
    quality_shards, quality_lifecycle = _parse_quality_combined(quality_rows)
    off = quality_shards[("off", "A")]
    on = quality_shards[("on", "A")]
    quality_pair = (off, on)
    performance_by_arm = {
        "off": performance_shards[(0, "off", "A")],
        "on": performance_shards[(1, "on", "A")],
    }

    quality_sources = [shard["header"]["source"] for shard in quality_pair]
    performance_source = performance_shards[SHARD_PLAN[0]]["header"]["source"]
    stable_performance_bound = {
        path: digest
        for path, digest in performance_source["bound_file_sha256"].items()
        if path != SOURCE_PATH
    }
    source_compatible = all(
        source["engine_source_sha256"]
        == performance_source["engine_source_sha256"]
        and {
            path: digest
            for path, digest in source["bound_file_sha256"].items()
            if path != SOURCE_PATH
        }
        == stable_performance_bound
        for source in quality_sources
    )
    quality_source_same = (
        len({canonical_json(source) for source in quality_sources}) == 1
    )
    checkpoint_exact = all(
        shard["header"]["checkpoint"]
        == performance_by_arm[shard["identity"][0]]["header"]["checkpoint"]
        for shard in quality_pair
    )
    trace_exact = all(
        shard["header"]["trace"]
        == performance_by_arm[shard["identity"][0]]["header"]["trace"]
        for shard in quality_pair
    )
    def stable_hardware(
        shard: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            key: value
            for key, value in shard["header"]["hardware"].items()
            if key != "hostname"
        }

    hardware_exact = (
        stable_hardware(off) == stable_hardware(on)
        and all(
            stable_hardware(shard)
            == stable_hardware(
                performance_by_arm[shard["identity"][0]]
            )
            for shard in quality_pair
        )
    )
    runtime_exact = all(
        shard["header"]["runtime"]
        == performance_by_arm[shard["identity"][0]]["header"]["runtime"]
        for shard in quality_pair
    )
    container_exact = all(
        shard["header"]["container"]["image_id"]
        == performance_by_arm[shard["identity"][0]]["header"]["container"][
            "image_id"
        ]
        and shard["header"]["container"]["repo_digests"]
        == performance_by_arm[shard["identity"][0]]["header"]["container"][
            "repo_digests"
        ]
        for shard in quality_pair
    )
    performance_container_ids = {
        shard["header"]["container"]["container_id"]
        for shard in performance_shards.values()
    }
    quality_container_ids = {
        shard["header"]["container"]["container_id"]
        for shard in quality_pair
    }
    fresh_containers = (
        len(quality_container_ids) == len(QUALITY_PLAN)
        and not (quality_container_ids & performance_container_ids)
    )
    performance_lifecycle_bounds = _container_lifecycle_bounds(
        performance_lifecycle
    )
    quality_lifecycle_bounds = _container_lifecycle_bounds(quality_lifecycle)
    quality_intervals_sequential = (
        max(shard["interval"][1] for shard in performance_shards.values())
        < off["interval"][0]
        and off["interval"][1] < on["interval"][0]
        and performance_lifecycle_bounds[1] <= quality_lifecycle_bounds[0]
    )
    parent_digest_exact = all(
        shard["header"]["parent_performance_raw_sha256"]
        == performance_raw_sha256
        for shard in quality_pair
    )

    outputs_cache_and_tier_stats_match = True
    prompts_match = True
    for shard in quality_pair:
        reference = performance_by_arm[shard["identity"][0]]
        for quality_request, performance_request in zip(
            shard["requests"],
            reference["requests"],
            strict=True,
        ):
            prompts_match &= (
                quality_request["prompt_token_ids"]
                == performance_request["prompt_token_ids"]
            )
            outputs_cache_and_tier_stats_match &= (
                quality_request["output_token_ids"]
                == performance_request["output_token_ids"]
                and quality_request["usage"] == performance_request["usage"]
                and quality_request["initial_tier_stats"]
                == performance_request["initial_tier_stats"]
                and quality_request["final_tier_stats"]
                == performance_request["final_tier_stats"]
            )

    off_tier_zero = all(
        all(
            request[boundary][field] == 0
            for boundary in ("initial_tier_stats", "final_tier_stats")
            for field in TIER_STATS_FIELDS
        )
        for request in off["requests"]
    )
    on_final = on["requests"][-1]["final_tier_stats"]
    on_tier_activity = (
        on_final["offload_pages"] > 0
        and on_final["restore_pages"] > 0
        and on_final["restore_attempts"] > 0
        and on_final["restore_fallbacks"] == 0
        and on_final["ownership_failures"] == 0
    )
    distribution_checks, distribution_diagnostics = (
        _quality_distribution_comparison(
            off["requests"],
            on["requests"],
        )
    )
    first_divergence_sequences = sorted(
        {
            int(divergence["sequence"])
            for divergence in distribution_diagnostics["first_divergences"]
        }
    )
    first_divergence_restore_observations = []
    for sequence in first_divergence_sequences:
        request = on["requests"][sequence]
        initial_stats = request["initial_tier_stats"]
        final_stats = request["final_tier_stats"]
        first_divergence_restore_observations.append(
            {
                "sequence": sequence,
                "restore_attempts_delta": (
                    final_stats["restore_attempts"]
                    - initial_stats["restore_attempts"]
                ),
                "restore_pages_delta": (
                    final_stats["restore_pages"] - initial_stats["restore_pages"]
                ),
                "restore_fallbacks_delta": (
                    final_stats["restore_fallbacks"]
                    - initial_stats["restore_fallbacks"]
                ),
                "ownership_failures_delta": (
                    final_stats["ownership_failures"]
                    - initial_stats["ownership_failures"]
                ),
            }
        )
    first_divergence_requests_observe_restore = all(
        observation["restore_attempts_delta"] > 0
        and observation["restore_pages_delta"] > 0
        and observation["restore_fallbacks_delta"] == 0
        and observation["ownership_failures_delta"] == 0
        for observation in first_divergence_restore_observations
    )
    checks = {
        "parent_performance_artifact_passes": performance_manifest["passed"]
        is True,
        "parent_performance_raw_digest_exact": parent_digest_exact,
        "quality_container_lifecycle_metadata_exact_and_sequential": True,
        "quality_two_fresh_sequential_same_cohort_containers": (
            fresh_containers and quality_intervals_sequential and hardware_exact
        ),
        "quality_clean_source_and_engine_equal_parent": (
            quality_source_same and source_compatible
        ),
        "quality_checkpoint_trace_image_and_runtime_equal_parent": (
            checkpoint_exact
            and trace_exact
            and container_exact
            and runtime_exact
        ),
        "quality_prompts_equal_parent_performance_trace": prompts_match,
        "quality_outputs_cache_and_tier_stats_equal_parent_arms": (
            outputs_cache_and_tier_stats_match
        ),
        "quality_tier_on_activity_without_failure_and_tier_off_zero": (
            on_tier_activity and off_tier_zero
        ),
        "first_divergence_requests_observe_tier_on_restore": (
            first_divergence_requests_observe_restore
        ),
        **distribution_checks,
    }
    distribution_diagnostics["first_divergence_tier_on_restore_observations"] = (
        first_divergence_restore_observations
    )
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "measurement_kind": QUALITY_MEASUREMENT_KIND,
        "config": quality_config(),
        "parent_performance_raw": {
            "name": RAW_NAME,
            "sha256": performance_raw_sha256,
            "row_count": len(performance_rows),
        },
        "raw": {
            "name": QUALITY_RAW_NAME,
            "sha256": quality_raw_sha256,
            "row_count": len(quality_rows),
        },
        "container_metadata": {
            "directory": QUALITY_CONTAINER_METADATA_DIR,
            "file_sha256": dict(
                sorted(_metadata_digest_map(quality_lifecycle).items())
            ),
            "files_sha256": sha256_json(
                dict(sorted(_metadata_digest_map(quality_lifecycle).items()))
            ),
            "lifecycle_value_sha256": sha256_json(quality_lifecycle),
        },
        "quality_source": {
            "commit": quality_sources[0]["commit"],
            "bound_files_sha256": quality_sources[0][
                "bound_files_sha256"
            ],
            "engine_source_sha256": quality_sources[0][
                "engine_source_sha256"
            ],
        },
        "checks": checks,
        "diagnostics": distribution_diagnostics,
        "passed": all(checks.values()),
    }


def nearest_rank_percentile(values: Sequence[float], quantile: float) -> float:
    _require(values and 0 < quantile <= 1, "percentile input is invalid")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _critical_path_ok(request: Mapping[str, object]) -> bool:
    events = request["events"]
    first_index = next(
        (index for index, event in enumerate(events) if event["output_count"] == 1), None
    )
    if first_index is None:
        return False
    baseline = events[first_index]["tier_stats"]
    if any(event["tier_stats"] != baseline for event in events[first_index:]):
        return False
    by_count = {event["output_count"]: event for event in events if event["emitted_update"]}
    at_16 = by_count.get(16)
    at_17 = by_count.get(17)
    return (
        request["prompt_tokens"] % PAGE_SIZE == 0
        and at_16 is not None
        and at_17 is not None
        and at_17["num_free_pages"] == at_16["num_free_pages"] - 1
        and at_17["tier_stats"] == at_16["tier_stats"]
    )


def _gpu_uuids(shard: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(row["device_uuid"] for row in shard["header"]["hardware"]["gpus"])


def _gpu_specification(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: row[key]
        for key in (
            "name",
            "torch_name",
            "compute_capability",
            "total_memory_bytes",
        )
    }


def recompute_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    _require(_valid_sha256(raw_sha256), "manifest raw digest is invalid")
    shards, lifecycle = _parse_combined(rows)
    all_shards = [shards[identity] for identity in SHARD_PLAN]
    metadata_digests = _metadata_digest_map(lifecycle)
    metadata_rollup = sha256_json(dict(sorted(metadata_digests.items())))

    source_same = len({canonical_json(shard["header"]["source"]) for shard in all_shards}) == 1
    reference_source = all_shards[0]["header"]["source"]
    bound_source = {
        "commit": reference_source["commit"],
        "bound_files_sha256": reference_source["bound_files_sha256"],
        "engine_source_sha256": reference_source["engine_source_sha256"],
    }
    checkpoint_same = (
        len({canonical_json(shard["header"]["checkpoint"]) for shard in all_shards}) == 1
    )
    calibrated_execution_image = _calibrated_execution_image_identity(lifecycle)
    execution_image_same = (
        len(
            {
                canonical_json(
                    {
                        "image_id": shard["header"]["container"]["image_id"],
                        "repo_digests": shard["header"]["container"]["repo_digests"],
                    }
                )
                for shard in all_shards
            }
        )
        == 1
    )
    containers_fresh = len(
        {shard["header"]["container"]["container_id"] for shard in all_shards}
    ) == len(SHARD_PLAN)
    profiles = [
        shards[identity]["header"]["profile"] for identity in SHARD_PLAN if identity[1] == "on"
    ]
    profile_same = len({canonical_json(profile) for profile in profiles}) == 1
    profile_identity_exact = all(
        profile["file_sha256"] == F4A_PROFILE_FILE_SHA256
        and profile["value"]["evidence_sha256"] == F4A_PROFILE_EVIDENCE_SHA256
        and profile["value"]["runtime_fingerprint"] == F4A_PROFILE_RUNTIME_FINGERPRINT
        for profile in profiles
    )

    runtime_common = [
        {
            key: value
            for key, value in shard["header"]["runtime"].items()
            if key not in RUNTIME_ARM_FIELDS
        }
        for shard in all_shards
    ]
    runtime_identity_same = len({canonical_json(identity) for identity in runtime_common}) == 1

    hardware_environments = [
        {
            key: shard["header"]["hardware"][key]
            for key in (
                "driver_version",
                "torch_version",
                "torch_cuda_version",
                "nccl_version",
            )
        }
        for shard in all_shards
    ]
    hardware_environment_same = (
        len({canonical_json(identity) for identity in hardware_environments}) == 1
    )
    gpu_specifications = [
        _gpu_specification(row)
        for shard in all_shards
        for row in shard["header"]["hardware"]["gpus"]
    ]
    gpu_specification_same = len({canonical_json(identity) for identity in gpu_specifications}) == 1

    intervals = [shards[identity]["interval"] for identity in SHARD_PLAN]
    intervals_nonoverlap = all(
        intervals[index][1] < intervals[index + 1][0] for index in range(len(intervals) - 1)
    )
    cohort_a_shard = shards[(0, "off", "A")]
    cohort_a_swap_shard = shards[(1, "on", "A")]
    cohort_b_shard = shards[(0, "on", "B")]
    cohort_b_swap_shard = shards[(1, "off", "B")]
    cohort_a = _gpu_uuids(cohort_a_shard)
    cohort_b = _gpu_uuids(cohort_b_shard)
    cohort_swap_exact = (
        cohort_a_shard["header"]["hardware"]["gpus"]
        == cohort_a_swap_shard["header"]["hardware"]["gpus"]
        and cohort_b_shard["header"]["hardware"]["gpus"]
        == cohort_b_swap_shard["header"]["hardware"]["gpus"]
        and not (set(cohort_a) & set(cohort_b))
    )
    container_identity_exact = all(
        shard["header"]["container"]["container_id"].startswith(
            shard["header"]["container"]["hostname"]
        )
        and shard["header"]["container"]["container_id_binding"]["resolved_container_id"]
        == shard["header"]["container"]["container_id"]
        for shard in all_shards
    )

    reference = shards[SHARD_PLAN[0]]["requests"]
    prompt_order_identity = True
    for shard in all_shards[1:]:
        prompt_order_identity &= all(
            candidate["sequence"] == base["sequence"]
            and candidate["session"] == base["session"]
            and candidate["turn"] == base["turn"]
            and candidate["prompt_token_ids"] == base["prompt_token_ids"]
            for base, candidate in zip(reference, shard["requests"], strict=True)
        )

    def output_agreement(
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> dict[str, object]:
        left_requests = left["requests"]
        right_requests = right["requests"]
        exact_requests = 0
        matching_token_positions = 0
        first_divergences = []
        for left_request, right_request in zip(
            left_requests,
            right_requests,
            strict=True,
        ):
            left_output = left_request["output_token_ids"]
            right_output = right_request["output_token_ids"]
            exact_requests += left_output == right_output
            matching_token_positions += sum(
                left_token == right_token
                for left_token, right_token in zip(
                    left_output,
                    right_output,
                    strict=True,
                )
            )
            first_divergence = next(
                (
                    position
                    for position, (left_token, right_token) in enumerate(
                        zip(left_output, right_output, strict=True)
                    )
                    if left_token != right_token
                ),
                None,
            )
            if first_divergence is not None:
                first_divergences.append(
                    {
                        "sequence": left_request["sequence"],
                        "session": left_request["session"],
                        "turn": left_request["turn"],
                        "position": first_divergence,
                        "left_token_id": left_output[first_divergence],
                        "right_token_id": right_output[first_divergence],
                    }
                )
        request_total = len(left_requests)
        token_total = request_total * OUTPUT_TOKENS
        return {
            "exact_request_matches": exact_requests,
            "request_total": request_total,
            "exact_request_match_rate": exact_requests / request_total,
            "matching_token_positions": matching_token_positions,
            "token_position_total": token_total,
            "token_position_match_rate": matching_token_positions / token_total,
            "first_divergences": first_divergences,
        }

    output_agreements = {
        "tier_off_cross_cohort": output_agreement(
            shards[(0, "off", "A")],
            shards[(1, "off", "B")],
        ),
        "tier_on_cross_cohort": output_agreement(
            shards[(1, "on", "A")],
            shards[(0, "on", "B")],
        ),
        "tier_off_vs_on_cohort_a": output_agreement(
            shards[(0, "off", "A")],
            shards[(1, "on", "A")],
        ),
        "tier_off_vs_on_cohort_b": output_agreement(
            shards[(1, "off", "B")],
            shards[(0, "on", "B")],
        ),
    }
    all_arm_exact_requests = sum(
        all(
            shard["requests"][sequence]["output_token_ids"]
            == reference[sequence]["output_token_ids"]
            for shard in all_shards[1:]
        )
        for sequence in range(REQUESTS_PER_SHARD)
    )

    metrics: dict[str, dict[str, object]] = {}
    for identity in SHARD_PLAN:
        shard = shards[identity]
        prompt_tokens = sum(request["usage"]["prompt_tokens"] for request in shard["requests"])
        cached_tokens = sum(request["usage"]["cached_tokens"] for request in shard["requests"])
        tpots = [request["tpot_ns"] for request in shard["requests"]]
        key = f"round{identity[0]}-{identity[1]}-cohort{identity[2]}"
        metrics[key] = {
            "round": identity[0],
            "arm": identity[1],
            "cohort": identity[2],
            "request_count": len(shard["requests"]),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "prefix_hit_rate": cached_tokens / prompt_tokens,
            "tpot_p99_ns": nearest_rank_percentile(tpots, 0.99),
            "final_tier_stats": shard["requests"][-1]["final_tier_stats"],
        }

    def requests_for_arm(arm: str) -> list[dict[str, object]]:
        return [
            request
            for identity in SHARD_PLAN
            if identity[1] == arm
            for request in shards[identity]["requests"]
        ]

    arm_metrics: dict[str, dict[str, object]] = {}
    for arm in ("off", "on"):
        requests = requests_for_arm(arm)
        prompt = sum(request["usage"]["prompt_tokens"] for request in requests)
        cached = sum(request["usage"]["cached_tokens"] for request in requests)
        arm_metrics[arm] = {
            "request_count": len(requests),
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "prefix_hit_rate": cached / prompt,
            "tpot_p99_ns": nearest_rank_percentile(
                [request["tpot_ns"] for request in requests], 0.99
            ),
        }
    pooled_tpot_ratio = arm_metrics["on"]["tpot_p99_ns"] / arm_metrics["off"]["tpot_p99_ns"]

    cohort_ratios: dict[str, float] = {}
    cohort_cache_gain: dict[str, float] = {}
    for cohort in ("A", "B"):
        on_identity = next(
            identity for identity in SHARD_PLAN if identity[1] == "on" and identity[2] == cohort
        )
        off_identity = next(
            identity for identity in SHARD_PLAN if identity[1] == "off" and identity[2] == cohort
        )
        on_key = f"round{on_identity[0]}-on-cohort{cohort}"
        off_key = f"round{off_identity[0]}-off-cohort{cohort}"
        cohort_ratios[cohort] = metrics[on_key]["tpot_p99_ns"] / metrics[off_key]["tpot_p99_ns"]
        cohort_cache_gain[cohort] = (
            metrics[on_key]["prefix_hit_rate"] - metrics[off_key]["prefix_hit_rate"]
        )
    cohort_ratio_geomean = math.sqrt(cohort_ratios["A"] * cohort_ratios["B"])

    on_requests = requests_for_arm("on")
    off_requests = requests_for_arm("off")
    tier_on_final = [
        shards[identity]["requests"][-1]["final_tier_stats"]
        for identity in SHARD_PLAN
        if identity[1] == "on"
    ]
    tier_activity = all(
        stats["offload_pages"] > 0
        and stats["restore_pages"] > 0
        and stats["restore_attempts"] > 0
        and stats["restore_fallbacks"] == 0
        and stats["ownership_failures"] == 0
        for stats in tier_on_final
    )
    off_stats_zero = all(
        all(event["tier_stats"][field] == 0 for field in TIER_STATS_FIELDS)
        and all(
            request[boundary][field] == 0
            for boundary in ("initial_tier_stats", "final_tier_stats")
            for field in TIER_STATS_FIELDS
        )
        for request in off_requests
        for event in request["events"]
    )
    critical_path = all(_critical_path_ok(request) for request in on_requests)

    checks = {
        "four_fresh_sequential_shards_in_fixed_crossover_order": intervals_nonoverlap
        and containers_fresh,
        "container_lifecycle_metadata_exact_and_sequential": True,
        "cohort_swap_and_disjoint_gpu_assignment_exact": cohort_swap_exact,
        "runtime_identity_equal_across_all_arms": runtime_identity_same,
        "hardware_environment_and_gpu_specification_equal_across_all_arms": (
            hardware_environment_same and gpu_specification_same
        ),
        "container_hostname_command_and_shard_identity_exact": container_identity_exact,
        "bound_source_identity_exact_across_all_arms": source_same,
        "checkpoint_identity_exact_across_all_arms": checkpoint_same,
        "calibrated_execution_image_identity_exact_across_all_arms": execution_image_same,
        "retained_f4a_tp4_profile_identity_exact": profile_identity_exact and profile_same,
        "all_requests_complete_without_retry_at_exact_32_tokens": all(
            request["status"] == "success"
            and request["retry_count"] == 0
            and request["usage"]["completion_tokens"] == OUTPUT_TOKENS
            for shard in all_shards
            for request in shard["requests"]
        ),
        "prompt_order_identity_exact_across_all_arms": prompt_order_identity,
        "tier_on_prefix_hit_rate_strictly_higher_pooled_and_per_cohort": (
            arm_metrics["on"]["prefix_hit_rate"] > arm_metrics["off"]["prefix_hit_rate"]
            and all(gain > 0 for gain in cohort_cache_gain.values())
        ),
        "tier_activity_nonzero_without_restore_or_ownership_failure": tier_activity,
        "tier_off_has_no_profile_slab_or_tier_activity": off_stats_zero,
        "tier_transfers_absent_after_first_content_with_decode_allocation_proven": critical_path,
        "pooled_tpot_p99_ratio_at_most_1_10": pooled_tpot_ratio <= TPOT_RATIO_LIMIT,
        "cohort_tpot_p99_ratio_geometric_mean_at_most_1_10": cohort_ratio_geomean
        <= TPOT_RATIO_LIMIT,
    }
    diagnostics = {
        "cache_gain_percentage_points_pooled": 100
        * (arm_metrics["on"]["prefix_hit_rate"] - arm_metrics["off"]["prefix_hit_rate"]),
        "cache_gain_percentage_points_by_cohort": {
            cohort: 100 * gain for cohort, gain in cohort_cache_gain.items()
        },
        "tpot_p99_ratio_pooled": pooled_tpot_ratio,
        "tpot_p99_ratio_by_cohort": cohort_ratios,
        "tpot_p99_ratio_cohort_geometric_mean": cohort_ratio_geomean,
        "individual_cohort_ratios_are_diagnostic_not_individual_gates": True,
        "critical_path_scope": "fixed representative serial agentic trace only",
        "free_running_output_equality_is_binding": False,
        "same_arm_cross_cohort_output_reproducibility_is_binding": False,
        "same_arm_cross_cohort_output_reproducibility_scope": (
            "artifact-specific diagnostic"
        ),
        "same_arm_cross_cohort_output_reproducibility_exact": (
            output_agreements["tier_off_cross_cohort"]["exact_request_matches"]
            == REQUESTS_PER_SHARD
            and output_agreements["tier_on_cross_cohort"]["exact_request_matches"]
            == REQUESTS_PER_SHARD
        ),
        "all_arm_exact_free_running_requests": all_arm_exact_requests,
        "all_arm_exact_free_running_request_total": REQUESTS_PER_SHARD,
        "output_agreement": output_agreements,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "verdict_revision": VERDICT_REVISION,
        "measurement_kind": MEASUREMENT_KIND,
        "config": formal_config(),
        "raw": {"name": RAW_NAME, "sha256": raw_sha256, "row_count": len(rows)},
        "bound_source": bound_source,
        "calibrated_execution_image": calibrated_execution_image,
        "container_metadata": {
            "directory": CONTAINER_METADATA_DIR,
            "file_sha256": dict(sorted(metadata_digests.items())),
            "files_sha256": metadata_rollup,
            "lifecycle_value_sha256": sha256_json(lifecycle),
        },
        "shards": metrics,
        "arms": arm_metrics,
        "checks": checks,
        "diagnostics": diagnostics,
        "passed": all(checks.values()),
    }


def replay_raw(path: Path, *, assert_gate: bool = False) -> dict[str, object]:
    manifest = evidence_replay_manifest(
        path,
        recompute=lambda rows, raw_sha256: recompute_manifest(
            rows,
            raw_sha256=raw_sha256,
        ),
        error_type=EvidenceError,
    )
    if assert_gate and manifest["passed"] is not True:
        failed = [name for name, passed in manifest["checks"].items() if not passed]
        raise EvidenceError(f"G5 F4b evidence failed: {', '.join(failed)}")
    return manifest


def assemble_artifact(
    raw_paths: Sequence[Path],
    output_dir: Path,
    *,
    metadata_dir: Path,
    assert_gate: bool = False,
) -> dict[str, object]:
    combined = assemble_rows(raw_paths, metadata_dir=metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retained_metadata = output_dir / CONTAINER_METADATA_DIR
    source_metadata = metadata_dir.resolve()
    destination_metadata = retained_metadata.resolve()
    _require(
        not destination_metadata.is_relative_to(source_metadata),
        "artifact container metadata cannot be nested in its source",
    )
    _require(
        not retained_metadata.exists(),
        f"retained container metadata already exists: {retained_metadata}",
    )
    shutil.copytree(metadata_dir, retained_metadata)
    raw_path = output_dir / RAW_NAME
    write_jsonl(raw_path, combined)
    manifest = replay_raw(raw_path, assert_gate=False)
    (output_dir / MANIFEST_NAME).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    if assert_gate and manifest["passed"] is not True:
        failed = [name for name, passed in manifest["checks"].items() if not passed]
        raise EvidenceError(f"G5 F4b evidence failed: {', '.join(failed)}")
    return manifest


def _artifact_paths(path: Path) -> tuple[Path, Path]:
    return evidence_artifact_paths(
        path,
        raw_name=RAW_NAME,
        manifest_name=MANIFEST_NAME,
        error_type=EvidenceError,
    )


def _verify_artifact_with_rows(
    path: Path,
    *,
    assert_gate: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    raw_path, _manifest_path = _artifact_paths(path)
    replayed_rows: list[dict[str, object]] | None = None

    def recompute_and_capture(
        rows: Sequence[dict[str, object]],
        raw_sha256: str,
    ) -> dict[str, object]:
        nonlocal replayed_rows
        replayed_rows = list(rows)
        return recompute_manifest(rows, raw_sha256=raw_sha256)

    recomputed = verify_replayed_manifest(
        path,
        raw_name=RAW_NAME,
        manifest_name=MANIFEST_NAME,
        recompute=recompute_and_capture,
        error_type=EvidenceError,
    )
    _require(replayed_rows is not None, "performance raw replay returned no rows")
    embedded_lifecycle = replayed_rows[0].get("container_lifecycle")
    retained_lifecycle = _load_container_lifecycle(raw_path.parent / CONTAINER_METADATA_DIR)
    _require(
        retained_lifecycle == embedded_lifecycle,
        "retained container metadata bytes/value differ from combined raw",
    )
    if assert_gate and recomputed["passed"] is not True:
        failed = [name for name, passed in recomputed["checks"].items() if not passed]
        raise EvidenceError(f"G5 F4b evidence failed: {', '.join(failed)}")
    return recomputed, replayed_rows


def verify_artifact(path: Path, *, assert_gate: bool = False) -> dict[str, object]:
    recomputed, _rows = _verify_artifact_with_rows(path, assert_gate=assert_gate)
    return recomputed


def replay_quality_raw(
    performance_raw_path: Path,
    quality_raw_path: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    performance_rows, performance_raw_sha256 = evidence_read_jsonl_with_sha256(
        performance_raw_path,
        error_type=EvidenceError,
    )
    manifest = evidence_replay_manifest(
        quality_raw_path,
        recompute=lambda quality_rows, quality_raw_sha256: recompute_quality_manifest(
            performance_rows,
            quality_rows,
            performance_raw_sha256=performance_raw_sha256,
            quality_raw_sha256=quality_raw_sha256,
        ),
        error_type=EvidenceError,
    )
    if assert_gate and manifest["passed"] is not True:
        failed = [
            name
            for name, passed in manifest["checks"].items()
            if not passed
        ]
        raise EvidenceError(
            f"G5 F4b quality evidence failed: {', '.join(failed)}"
        )
    return manifest


def seal_quality_artifact(
    performance_artifact: Path,
    quality_raw_paths: Sequence[Path],
    output_dir: Path,
    *,
    quality_metadata_dir: Path,
    assert_gate: bool = False,
) -> dict[str, object]:
    performance_manifest = verify_artifact(
        performance_artifact,
        assert_gate=True,
    )
    performance_raw_path, _performance_manifest_path = _artifact_paths(
        performance_artifact
    )
    performance_raw = performance_manifest.get("raw")
    _require(isinstance(performance_raw, dict), "performance raw manifest is invalid")
    performance_raw_sha256 = performance_raw.get("sha256")
    _require(
        _valid_sha256(performance_raw_sha256),
        "performance raw manifest digest is invalid",
    )
    assert isinstance(performance_raw_sha256, str)
    combined_quality = assemble_quality_rows(
        quality_raw_paths,
        parent_performance_raw_sha256=performance_raw_sha256,
        metadata_dir=quality_metadata_dir,
    )
    _require(
        not output_dir.exists(),
        f"quality artifact output already exists: {output_dir}",
    )
    shutil.copytree(performance_raw_path.parent, output_dir)
    retained_quality_metadata = output_dir / QUALITY_CONTAINER_METADATA_DIR
    source_quality_metadata = quality_metadata_dir.resolve()
    destination_quality_metadata = retained_quality_metadata.resolve()
    _require(
        not destination_quality_metadata.is_relative_to(source_quality_metadata),
        "quality artifact container metadata cannot be nested in its source",
    )
    _require(
        not retained_quality_metadata.exists(),
        f"retained quality container metadata already exists: {retained_quality_metadata}",
    )
    shutil.copytree(quality_metadata_dir, retained_quality_metadata)
    retained_performance_raw = output_dir / RAW_NAME
    _require(
        sha256_file(retained_performance_raw) == performance_raw_sha256,
        "sealing quality changed the parent performance raw bytes",
    )
    quality_raw_path = output_dir / QUALITY_RAW_NAME
    write_jsonl(quality_raw_path, combined_quality)
    quality_manifest = replay_quality_raw(
        retained_performance_raw,
        quality_raw_path,
        assert_gate=False,
    )
    (output_dir / QUALITY_MANIFEST_NAME).write_text(
        canonical_json(quality_manifest) + "\n",
        encoding="utf-8",
    )
    if assert_gate and quality_manifest["passed"] is not True:
        failed = [
            name
            for name, passed in quality_manifest["checks"].items()
            if not passed
        ]
        raise EvidenceError(
            f"G5 F4b quality evidence failed: {', '.join(failed)}"
        )
    return {
        "performance": performance_manifest,
        "quality": quality_manifest,
        "passed": (
            performance_manifest["passed"] is True
            and quality_manifest["passed"] is True
        ),
    }


def verify_quality_artifact(
    path: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    _require(path.is_dir(), "quality artifact must be a directory")
    performance_manifest, performance_rows = _verify_artifact_with_rows(
        path,
        assert_gate=False,
    )
    performance_raw = performance_manifest.get("raw")
    _require(isinstance(performance_raw, dict), "performance raw manifest is invalid")
    performance_raw_sha256 = performance_raw.get("sha256")
    _require(
        _valid_sha256(performance_raw_sha256),
        "performance raw manifest digest is invalid",
    )
    assert isinstance(performance_raw_sha256, str)
    quality_rows: list[dict[str, object]] | None = None

    def recompute_and_capture(
        rows: Sequence[dict[str, object]],
        quality_raw_sha256: str,
    ) -> dict[str, object]:
        nonlocal quality_rows
        quality_rows = list(rows)
        return recompute_quality_manifest(
            performance_rows,
            rows,
            performance_raw_sha256=performance_raw_sha256,
            quality_raw_sha256=quality_raw_sha256,
        )

    recomputed = verify_replayed_manifest(
        path,
        raw_name=QUALITY_RAW_NAME,
        manifest_name=QUALITY_MANIFEST_NAME,
        recompute=recompute_and_capture,
        error_type=EvidenceError,
    )
    _require(quality_rows is not None, "quality raw replay returned no rows")
    embedded_lifecycle = quality_rows[0].get("container_lifecycle")
    retained_lifecycle = _load_container_lifecycle(
        path / QUALITY_CONTAINER_METADATA_DIR,
        plan=QUALITY_PLAN,
        labels=QUALITY_SHARD_LABELS,
    )
    _require(
        retained_lifecycle == embedded_lifecycle,
        "retained quality container metadata bytes/value differ from combined raw",
    )
    result = {
        "performance": performance_manifest,
        "quality": recomputed,
        "passed": (
            performance_manifest["passed"] is True
            and recomputed["passed"] is True
        ),
    }
    if assert_gate and result["passed"] is not True:
        failed = [
            *(
                f"performance:{name}"
                for name, passed in performance_manifest["checks"].items()
                if not passed
            ),
            *(
                f"quality:{name}"
                for name, passed in recomputed["checks"].items()
                if not passed
            ),
        ]
        raise EvidenceError(
            f"G5 F4b sealed evidence failed: {', '.join(failed)}"
        )
    return result


def schema_summary() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "verdict_revision": VERDICT_REVISION,
        "measurement_kind": MEASUREMENT_KIND,
        "formal_config": formal_config(),
        "raw_shards": len(SHARD_PLAN),
        "requests_per_shard": REQUESTS_PER_SHARD,
        "commands": [
            "run",
            "assemble",
            "quality-run",
            "seal-quality",
            "verify",
            "verify-quality",
            "replay",
            "replay-quality",
            "schema",
        ],
        "retained_files": [
            RAW_NAME,
            MANIFEST_NAME,
            QUALITY_RAW_NAME,
            QUALITY_MANIFEST_NAME,
            f"{CONTAINER_METADATA_DIR}/",
            f"{QUALITY_CONTAINER_METADATA_DIR}/",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="record one fresh TP4 shard")
    run.add_argument("--round", type=int, choices=(0, 1), required=True)
    run.add_argument("--arm", choices=("off", "on"), required=True)
    run.add_argument("--cohort", choices=("A", "B"), required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--profile", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--image-id", required=True)
    run.add_argument("--container-id")
    run.add_argument("--container-id-file", type=Path)
    run.add_argument("--repo-digest", action="append", default=[])

    quality_run = subparsers.add_parser(
        "quality-run",
        help="record one timing-independent TP4 distribution-quality shard",
    )
    quality_run.add_argument("--arm", choices=("off", "on"), required=True)
    quality_run.add_argument("--cohort", choices=("A",), required=True)
    quality_run.add_argument(
        "--performance-raw-sha256",
        required=True,
    )
    quality_run.add_argument("--model-path", type=Path, required=True)
    quality_run.add_argument("--profile", type=Path)
    quality_run.add_argument("--output", type=Path, required=True)
    quality_run.add_argument("--image-id", required=True)
    quality_run.add_argument("--container-id")
    quality_run.add_argument("--container-id-file", type=Path)
    quality_run.add_argument(
        "--repo-digest",
        action="append",
        default=[],
    )

    assemble = subparsers.add_parser("assemble", help="seal four raw shards")
    assemble.add_argument("--raw", type=Path, action="append", required=True)
    assemble.add_argument("--metadata-dir", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--assert-gate", action="store_true")

    verify = subparsers.add_parser("verify", help="verify retained raw and manifest")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--assert-gate", action="store_true")

    seal_quality = subparsers.add_parser(
        "seal-quality",
        help="add two quality shards to a passing performance artifact",
    )
    seal_quality.add_argument(
        "--performance-artifact",
        type=Path,
        required=True,
    )
    seal_quality.add_argument(
        "--quality-raw",
        type=Path,
        action="append",
        required=True,
    )
    seal_quality.add_argument(
        "--quality-metadata-dir",
        type=Path,
        required=True,
    )
    seal_quality.add_argument("--output-dir", type=Path, required=True)
    seal_quality.add_argument("--assert-gate", action="store_true")

    verify_quality = subparsers.add_parser(
        "verify-quality",
        help="verify the retained performance and quality evidence",
    )
    verify_quality.add_argument("--artifact", type=Path, required=True)
    verify_quality.add_argument("--assert-gate", action="store_true")

    replay = subparsers.add_parser("replay", help="derive manifest from combined raw")
    replay.add_argument("--raw", type=Path, required=True)
    replay.add_argument("--assert-gate", action="store_true")

    replay_quality = subparsers.add_parser(
        "replay-quality",
        help="derive the quality manifest from both combined raw files",
    )
    replay_quality.add_argument(
        "--performance-raw",
        type=Path,
        required=True,
    )
    replay_quality.add_argument("--quality-raw", type=Path, required=True)
    replay_quality.add_argument("--assert-gate", action="store_true")

    subparsers.add_parser("schema", help="print the fixed evidence contract")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        result = asyncio.run(
            _run_live_shard(
                round_index=args.round,
                arm=args.arm,
                cohort=args.cohort,
                model_path=args.model_path.resolve(),
                profile_path=args.profile.resolve() if args.profile else None,
                output=args.output.resolve(),
                container=container,
            )
        )
    elif args.command == "quality-run":
        container_id = _resolve_container_id(
            container_id=args.container_id,
            container_id_file=args.container_id_file,
        )
        container = _container_provenance_live(
            container_id=container_id,
            image_id=args.image_id,
            repo_digests=args.repo_digest,
            command=list(
                sys.argv if argv is None else (SOURCE_PATH, *argv)
            ),
        )
        result = asyncio.run(
            _run_live_quality_shard(
                arm=args.arm,
                cohort=args.cohort,
                performance_raw_sha256=args.performance_raw_sha256,
                model_path=args.model_path.resolve(),
                profile_path=(
                    args.profile.resolve() if args.profile else None
                ),
                output=args.output.resolve(),
                container=container,
            )
        )
    elif args.command == "assemble":
        result = assemble_artifact(
            tuple(args.raw),
            args.output_dir,
            metadata_dir=args.metadata_dir,
            assert_gate=args.assert_gate,
        )
    elif args.command == "verify":
        result = verify_artifact(args.artifact, assert_gate=args.assert_gate)
    elif args.command == "seal-quality":
        result = seal_quality_artifact(
            args.performance_artifact,
            tuple(args.quality_raw),
            args.output_dir,
            quality_metadata_dir=args.quality_metadata_dir,
            assert_gate=args.assert_gate,
        )
    elif args.command == "verify-quality":
        result = verify_quality_artifact(
            args.artifact,
            assert_gate=args.assert_gate,
        )
    elif args.command == "replay":
        result = replay_raw(args.raw, assert_gate=args.assert_gate)
    elif args.command == "replay-quality":
        result = replay_quality_raw(
            args.performance_raw,
            args.quality_raw,
            assert_gate=args.assert_gate,
        )
    else:
        result = schema_summary()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
