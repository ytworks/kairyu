#!/usr/bin/env python3
"""G4 M-A2: prove radix KV reuse is identical on every Qwen3-235B EP4 rank.

This repository-only formal operator lives at
``bench/g4_ma2_qwen3_235b_ep_kv_bench.py``.  ``run`` is intentionally a local
four-GPU operation; ordinary GitHub CI uses only ``verify``/``replay`` and the
synthetic tamper tests.

The real run owns exactly one production ``DistEPLauncher``, one persistent
``RadixKVCache``/``Scheduler``/``EngineLoop``, and submits the fixed A7 trace
strictly one request at a time.  Cache rate is computed only from terminal
``StreamUpdate.num_cached_tokens / num_prompt_tokens``.  The four physical-rank
observations prove that the replicated EP state agrees with those logical
rows; they are never multiplied into the rate.

Usage::

    python bench/g4_ma2_qwen3_235b_ep_kv_bench.py run \
      --model-path /models/Qwen3-235B-A22B-NVFP4 \
      --container-inspect /results/container-inspect.json \
      --image-repo-digest repo/image@sha256:... \
      --image-config-id sha256:... --output-dir /results/g4-ma2 \
      --assert-gate
    python bench/g4_ma2_qwen3_235b_ep_kv_bench.py verify \
      --artifact /results/g4-ma2 --assert-gate
    python bench/g4_ma2_qwen3_235b_ep_kv_bench.py replay \
      --artifact /results/g4-ma2 --assert-gate
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

from bench import g4_ma1_qwen3_235b_nvfp4_capture as ma1_capture  # noqa: E402
from bench import tp_kv_hit_g2_a7_bench as a7  # noqa: E402
from kairyu.bench.reporting import atomic_write_json, atomic_write_text  # noqa: E402

ma1 = ma1_capture.gate

SCHEMA_VERSION = "kairyu.g4.ma2.qwen3-235b-ep-kv.v1"
GATE = "G4-M-A2"
RAW_NAME = "g4-ma2-qwen3-235b-ep-kv-raw.jsonl"
MANIFEST_NAME = "g4-ma2-qwen3-235b-ep-kv-manifest.json"

MODEL = ma1.MODEL
MODEL_REVISION = ma1.MODEL_REVISION
EXPERT_PARALLEL_SIZE = 4
PAGE_SIZE = 16
MAX_TOKENS = 1
CACHE_RATE_FLOOR = 0.80
EXPECTED_REQUESTS = 512
EXPECTED_CACHE_EVENTS = EXPECTED_REQUESTS
ACCOUNTING_OBSERVATION_LIMIT = 4_096
MAX_ENGINE_STEPS_PER_REQUEST = 4

# Every prompt is page aligned.  The retained tree contains one 512-token
# shared prefix plus eight new 128-token pages for each of 64 sessions on each
# of eight turns: 32 + 64 * 8 * 8 = 4,128 pages.  One additional page is a
# deliberate active-request guard for the one-token generation contract.  This
# is 4,129 rather than the unrelated A7 simulator's 8,192-page default.
SHARED_PREFIX_PAGES = a7.SYSTEM_PROMPT_TOKENS // PAGE_SIZE
PAGES_PER_TURN = a7.TURN_TOKENS // PAGE_SIZE
MAX_RETAINED_PAGES = (
    SHARED_PREFIX_PAGES
    + a7.NUM_SESSIONS * a7.TURNS_PER_SESSION * PAGES_PER_TURN
)
ACTIVE_REQUEST_GUARD_PAGES = math.ceil(MAX_TOKENS / PAGE_SIZE)
NUM_PAGES = MAX_RETAINED_PAGES + ACTIVE_REQUEST_GUARD_PAGES

# Qwen3-235B has 94 layers, K+V, four KV heads, head-dim 128, BF16, 16 tokens.
KV_BYTES_PER_PAGE = ma1.NUM_LAYERS * 2 * 4 * 128 * 2 * PAGE_SIZE
KV_CAPACITY_BYTES_PER_RANK = NUM_PAGES * KV_BYTES_PER_PAGE

EXPECTED_PROMPT_TOKENS = 557_056
EXPECTED_THEORETICAL_CACHED_TOKENS = 491_008

BOUND_SOURCE_FILES = tuple(
    sorted(
        {
            *ma1.BOUND_SOURCE_FILES,
            "Dockerfile.cuda",
            "bench/g4_ma2_qwen3_235b_ep_kv_bench.py",
            "bench/multiturn_prefix.py",
            "bench/tp_kv_hit_g2_a7_bench.py",
            "kairyu/engine/config_validation.py",
            "kairyu/engine/core/radix_kv.py",
            "kairyu/engine/core/scheduler.py",
            "kairyu/engine/core/step_input.py",
            "kairyu/engine/engine_loop.py",
            "kairyu/engine/kairyu_backend.py",
            "kairyu/engine/openai_backend.py",
            "kairyu/engine/prompt.py",
            "kairyu/engine/tokenizer.py",
            "kairyu/entrypoints/server/health.py",
            "kairyu/sampling_params.py",
        }
    )
)

_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ARM = "kairyu_ep4"


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
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        cwd=_REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _source_snapshot() -> dict[str, object]:
    files: dict[str, str] = {}
    for relative in BOUND_SOURCE_FILES:
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
            raise ValueError(f"ambiguous tracked source identity: {relative}")
        files[relative] = sha256_file(path)
    value = {
        "commit": _git("rev-parse", "HEAD"),
        "tracked_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=all")
        ),
        "files": files,
    }
    if not _source_valid(value):
        raise ValueError("formal capture requires a clean bound source commit")
    return value


def _source_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"commit", "tracked_dirty", "files"}
        and isinstance(value["commit"], str)
        and _HEX40.fullmatch(value["commit"]) is not None
        and value["tracked_dirty"] is False
        and isinstance(value["files"], dict)
        and set(value["files"]) == set(BOUND_SOURCE_FILES)
        and all(
            isinstance(digest, str) and _HEX64.fullmatch(digest) is not None
            for digest in value["files"].values()
        )
    )


@cache
def _fixed_geometry() -> tuple[dict[str, int], ...]:
    rows = a7._fixed_geometry()
    if len(rows) != EXPECTED_REQUESTS:
        raise RuntimeError("A7 fixed geometry is no longer 512 requests")
    if (
        sum(row["prompt_tokens"] for row in rows) != EXPECTED_PROMPT_TOKENS
        or sum(row["expected_cached_tokens"] for row in rows)
        != EXPECTED_THEORETICAL_CACHED_TOKENS
        or any(row["prompt_tokens"] % PAGE_SIZE for row in rows)
        or any(row["expected_cached_tokens"] % PAGE_SIZE for row in rows)
        or MAX_RETAINED_PAGES != 4_128
        or NUM_PAGES != 4_129
    ):
        raise RuntimeError("A7 geometry or the derived EP KV capacity changed")
    return rows


def formal_config() -> dict[str, object]:
    _fixed_geometry()
    return {
        "gate": GATE,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "topology": {
            "expert_parallel_size": EXPERT_PARALLEL_SIZE,
            "tensor_parallel_size": 1,
            "attention_placement": "replicated",
            "attention_output_placement": "row_parallel",
            "pipeline_depth": 1,
            "decode_mode": "eager",
            "kv_cache_dtype": "bfloat16",
        },
        "workload": {
            "source": "G2-A7-fixed-geometry",
            "construction": "stable-single-token-repeat-v1",
            "seed": a7.WORKLOAD_SEED,
            "sessions": a7.NUM_SESSIONS,
            "turns_per_session": a7.TURNS_PER_SESSION,
            "shared_prefix_tokens": a7.SYSTEM_PROMPT_TOKENS,
            "turn_tokens": a7.TURN_TOKENS,
            "request_count": EXPECTED_REQUESTS,
            "serialized": True,
            "theoretical_prompt_tokens": EXPECTED_PROMPT_TOKENS,
            "theoretical_cached_tokens": (
                EXPECTED_THEORETICAL_CACHED_TOKENS
            ),
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": a7.WORKLOAD_SEED,
            "max_tokens": MAX_TOKENS,
            "ignore_eos": True,
        },
        "kv_policy": {
            "page_size": PAGE_SIZE,
            "shared_prefix_pages": SHARED_PREFIX_PAGES,
            "pages_per_session_turn": PAGES_PER_TURN,
            "max_retained_pages": MAX_RETAINED_PAGES,
            "active_request_guard_pages": ACTIVE_REQUEST_GUARD_PAGES,
            "num_pages": NUM_PAGES,
            "bytes_per_page_per_rank": KV_BYTES_PER_PAGE,
            "capacity_bytes_per_rank": KV_CAPACITY_BYTES_PER_RANK,
        },
        "binding": {
            "cache_rate": {
                "definition": (
                    "sum(terminal StreamUpdate.num_cached_tokens) / "
                    "sum(terminal StreamUpdate.num_prompt_tokens)"
                ),
                "operator": ">",
                "value": CACHE_RATE_FLOOR,
            },
            "rank_rows_are_not_rate_samples": True,
            "required_rank_observations_per_request": EXPERT_PARALLEL_SIZE,
        },
    }


def build_workload(tokenizer) -> tuple[a7.TextWorkloadPrompt, ...]:
    """Use A7's Qwen-safe one-token text replacement without HTTP routing."""

    prompts = a7.build_text_workload(tokenizer, path="direct")
    if len(prompts) != EXPECTED_REQUESTS:
        raise RuntimeError("tokenized M-A2 workload is not 512 requests")
    return prompts


def trace_descriptor(
    prompts: Sequence[a7.TextWorkloadPrompt],
    *,
    tokenizer_sha256: str,
) -> dict[str, object]:
    return a7.trace_descriptor(
        prompts,
        path="direct",
        tokenizer_sha256=tokenizer_sha256,
    )


def _request_id(sequence: int, session: int, turn: int) -> str:
    return f"g4-ma2-q{sequence:03d}-s{session:02d}-t{turn}"


def _expected_parallelism_metadata() -> dict[str, object]:
    return {
        "parallelism": "expert_parallel",
        "expert_parallel_size": EXPERT_PARALLEL_SIZE,
        "attention_placement": "replicated",
        "attention_output_placement": "row_parallel",
        "attention_output_parallel_size": EXPERT_PARALLEL_SIZE,
        "attention_output_partial_dtype": "bfloat16",
        "execution_mode": "replicated-attention-correctness",
        "pipeline_depth": 1,
        "decode_mode": "eager",
        "kv_cache_dtype": "bfloat16",
    }


def _topology_value(
    metadata: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, object]:
    visibility = environment.get("visibility")
    if not isinstance(visibility, Mapping):
        raise ValueError("environment has no GPU visibility evidence")
    return {
        "parallelism": dict(metadata),
        "kv_cache": dict(formal_config()["kv_policy"]),
        "rank_gpu_uuid_order": list(visibility["resolved_rank_uuid_order"]),
    }


def _sampling_ownership_expected() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "control_world_size": EXPERT_PARALLEL_SIZE,
            "control_backend": "gloo",
            "model_world_size": EXPERT_PARALLEL_SIZE,
            "model_backend": "nccl",
            "sampling_owner": rank == 0,
            "sampler_present": rank == 0,
            "device": f"cuda:{rank}",
        }
        for rank in range(EXPERT_PARALLEL_SIZE)
    ]


def _kernel_inventory_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) != EXPERT_PARALLEL_SIZE:
        return False
    experts_per_rank = ma1.NUM_EXPERTS // EXPERT_PARALLEL_SIZE
    routed_count = 3 * ma1.NUM_LAYERS * experts_per_rank
    projection_count = 4 * ma1.NUM_LAYERS + routed_count
    dense_count = projection_count - routed_count
    seen_digests: set[str] = set()
    for rank, row in enumerate(value):
        first_owned = rank * experts_per_rank
        projection_sha = row.get("projection_inventory_sha256") if isinstance(row, dict) else None
        expected = {
            "rank": rank,
            "projection_count": projection_count,
            "projection_inventory_sha256": projection_sha,
            "kernel_counts": {
                "nvfp4:flashinfer_cutlass_fused_moe": routed_count,
                "nvfp4:flashinfer_nvfp4": dense_count,
            },
            "fused_moe": {
                "backend": "flashinfer.cutlass_fused_moe",
                "global_expert_count": ma1.NUM_EXPERTS,
                "local_expert_count": experts_per_rank,
                "local_expert_start": first_owned,
                "weight_order": "up_gate",
                "scale_layout": "128x4",
                "output_dtype": "bfloat16",
                "ep_partial": True,
                "enable_alltoall": False,
                "block_count": ma1.NUM_LAYERS,
                "successful_forward_block_count": ma1.NUM_LAYERS,
            },
            "attention_output": {
                "backend": "flashinfer.mm_fp4",
                "placement": "row_parallel",
                "parallel_size": EXPERT_PARALLEL_SIZE,
                "rank": rank,
                "shard_axis": 1,
                "global_in_features": 8_192,
                "local_in_features": 8_192 // EXPERT_PARALLEL_SIZE,
                "out_features": 4_096,
                "partial_dtype": "bfloat16",
                "block_count": ma1.NUM_LAYERS,
                "successful_forward_block_count": ma1.NUM_LAYERS,
            },
            "meta_count": {"parameters": 0, "buffers": 0, "total": 0},
            "load_info": {
                "ep_rank": rank,
                "ep_size": EXPERT_PARALLEL_SIZE,
                "owned_expert_count": experts_per_rank,
                "owned_expert_indices_sha256": sha256_json(
                    list(range(first_owned, first_owned + experts_per_rank))
                ),
                "first_owned_expert": first_owned,
                "last_owned_expert": first_owned + experts_per_rank - 1,
                "quantization_source": "hf_quant_config.json",
                "quantization_method": "nvfp4",
                "kv_cache_quant_algo": "FP8",
                "producer_name": "modelopt",
                "producer_version": "0.33.0",
                "checkpoint_tensor_count": ma1.EXPECTED_TENSOR_COUNT,
                "rank_loaded_tensor_count": (
                    ma1.LOADED_NON_EXPERT_TENSOR_COUNT
                    + ma1.NUM_LAYERS * experts_per_rank * 12
                ),
                "auxiliary_kv_scale_count": 188,
            },
        }
        if (
            not isinstance(row, dict)
            or not isinstance(projection_sha, str)
            or _HEX64.fullmatch(projection_sha) is None
            or projection_sha in seen_digests
            or row != expected
        ):
            return False
        seen_digests.add(projection_sha)
    return True


def _accounting_start_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) != EXPERT_PARALLEL_SIZE:
        return False
    empty_sha = sha256_json([])
    return all(
        row
        == {
            "rank": rank,
            "world_size": EXPERT_PARALLEL_SIZE,
            "capture_action": "start",
            "observation_limit": ACCOUNTING_OBSERVATION_LIMIT,
            "overflow": False,
            "capture_error": None,
            "request_count": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "observations_sha256": empty_sha,
            "observations": [],
        }
        for rank, row in enumerate(value)
    )


def _trace_valid(value: object, checkpoint: object) -> bool:
    if not a7._trace_valid(value, path="direct") or not isinstance(checkpoint, dict):
        return False
    assert isinstance(value, dict)
    trace = value.get("value")
    return (
        isinstance(trace, dict)
        and trace.get("tokenizer_sha256")
        == checkpoint.get("tokenizer_rollup_sha256")
        and len(trace.get("rows", [])) == EXPECTED_REQUESTS
    )


def _run_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "gate",
        "type",
        "row_index",
        "created_at",
        "config",
        "trace",
        "runtime",
        "source_start",
        "checkpoint_start",
        "checkpoint_view",
        "environment_start",
        "topology",
        "sampling_ownership",
        "attention_backend",
        "accounting_start",
    }:
        return False
    source = value["source_start"]
    checkpoint = value["checkpoint_start"]
    environment = value["environment_start"]
    runtime = value["runtime"]
    topology = value["topology"]
    return (
        value["schema_version"] == SCHEMA_VERSION
        and value["gate"] == GATE
        and value["type"] == "run"
        and value["row_index"] == 0
        and ma1._utc_timestamp(value["created_at"]) is not None
        and value["config"] == formal_config()
        and _source_valid(source)
        and ma1._checkpoint_valid(checkpoint)
        and _trace_valid(value["trace"], checkpoint)
        and ma1._environment_valid(environment, arm=_ARM)
        and isinstance(runtime, dict)
        and ma1._runtime_valid(
            runtime,
            arm=_ARM,
            run_image_repo_digest=runtime.get("image_repo_digest"),
            run_image_config_id=runtime.get("image_config_id"),
            source_commit=source.get("commit") if isinstance(source, dict) else None,
        )
        and ma1._checkpoint_view_valid(
            value["checkpoint_view"], arm=_ARM, checkpoint=checkpoint
        )
        and isinstance(topology, dict)
        and topology
        == {
            "parallelism": _expected_parallelism_metadata(),
            "kv_cache": formal_config()["kv_policy"],
            "rank_gpu_uuid_order": environment["visibility"][
                "resolved_rank_uuid_order"
            ],
        }
        and value["sampling_ownership"] == _sampling_ownership_expected()
        and ma1._attention_backend_valid(value["attention_backend"], arm=_ARM)
        and _accounting_start_valid(value["accounting_start"])
    )


def _run_end_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "gate",
            "type",
            "row_index",
            "finished_at",
            "source_end",
            "checkpoint_end",
            "environment_end",
            "row_counts_before_end",
        }
        and value["schema_version"] == SCHEMA_VERSION
        and value["gate"] == GATE
        and value["type"] == "run_end"
        and ma1._utc_timestamp(value["finished_at"]) is not None
        and _source_valid(value["source_end"])
        and ma1._checkpoint_valid(value["checkpoint_end"])
        and ma1._environment_valid(value["environment_end"], arm=_ARM)
        and isinstance(value["row_counts_before_end"], dict)
        and all(
            isinstance(key, str)
            and bool(key)
            and type(count) is int
            and count >= 0
            for key, count in value["row_counts_before_end"].items()
        )
    )


def _prefix_block_hashes(token_ids: Sequence[int]) -> tuple[str, ...]:
    # Match RadixKVCache's legacy repr(tuple(prefix)) identity in one pass.
    # Re-rendering every growing prefix makes replay quadratic in prompt size.
    prefix = hashlib.sha256()
    prefix.update(b"(")
    hashes: list[str] = []
    for position, token_id in enumerate(token_ids, start=1):
        if position > 1:
            prefix.update(b", ")
        prefix.update(repr(token_id).encode())
        if position % PAGE_SIZE == 0:
            snapshot = prefix.copy()
            if position == 1:
                snapshot.update(b",")
            snapshot.update(b")")
            hashes.append(snapshot.hexdigest()[:16])
    return tuple(hashes)


@cache
def _expected_event_contracts(
    shared_id: int,
    session_ids: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    contracts: list[dict[str, object]] = []
    for geometry in _fixed_geometry():
        sequence = geometry["sequence"]
        session = geometry["session"]
        turn = geometry["turn"]
        prompt = (
            (shared_id,) * a7.SYSTEM_PROMPT_TOKENS
            + (session_ids[session],) * ((turn + 1) * a7.TURN_TOKENS)
        )
        cached = geometry["expected_cached_tokens"]
        hashes = _prefix_block_hashes(prompt)
        cached_pages = cached // PAGE_SIZE
        contracts.append(
            {
                "request_id": _request_id(sequence, session, turn),
                "payload": {
                    "type": "BlockStored",
                    "block_hashes": list(hashes[cached_pages:]),
                    "parent_block_hash": (
                        hashes[cached_pages - 1] if cached_pages else None
                    ),
                    "token_ids": list(prompt[cached:]),
                    "block_size": PAGE_SIZE,
                },
            }
        )
    return tuple(contracts)


def _event_payload_valid(
    row: Mapping[str, object],
    *,
    trace: Mapping[str, object],
) -> bool:
    if set(row) != {
        "schema_version",
        "gate",
        "type",
        "row_index",
        "event_sequence",
        "request_id",
        "payload",
    }:
        return False
    sequence = row["event_sequence"]
    payload = row["payload"]
    trace_value = trace.get("value")
    if (
        not isinstance(trace_value, dict)
        or type(trace_value.get("shared_token_id")) is not int
        or not isinstance(trace_value.get("session_token_ids"), list)
        or any(type(token_id) is not int for token_id in trace_value["session_token_ids"])
        or type(sequence) is not int
        or not 0 <= sequence < EXPECTED_REQUESTS
        or not isinstance(payload, dict)
    ):
        return False
    contract = _expected_event_contracts(
        trace_value["shared_token_id"],
        tuple(trace_value["session_token_ids"]),
    )
    expected_payload = contract[sequence]["payload"]
    assert isinstance(expected_payload, dict)
    expected_hashes = expected_payload["block_hashes"]
    return (
        row["schema_version"] == SCHEMA_VERSION
        and row["gate"] == GATE
        and row["type"] == "cache_event"
        and row["request_id"] == contract[sequence]["request_id"]
        and set(payload)
        == {
            "type",
            "block_hashes",
            "parent_block_hash",
            "token_ids",
            "block_size",
        }
        and payload["type"] == "BlockStored"
        and payload == expected_payload
        and all(_HEX16.fullmatch(item) is not None for item in expected_hashes)
    )


def _request_usage_valid(value: object, *, prompt_tokens: int) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"prompt_tokens", "cached_tokens", "completion_tokens"}
        and type(value["prompt_tokens"]) is int
        and value["prompt_tokens"] == prompt_tokens
        and type(value["cached_tokens"]) is int
        and 0 <= value["cached_tokens"] <= prompt_tokens
        and value["cached_tokens"] % PAGE_SIZE == 0
        and value["completion_tokens"] == MAX_TOKENS
    )


def _request_row_valid(
    row: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> bool:
    sequence = expected["sequence"]
    prompt_tokens = expected["prompt_tokens"]
    return (
        set(row)
        == {
            "schema_version",
            "gate",
            "type",
            "row_index",
            "sequence",
            "request_id",
            "session",
            "turn",
            "prompt_token_ids_sha256",
            "expected_cached_tokens",
            "status",
            "finish_reason",
            "error_type",
            "output_token_ids",
            "usage",
        }
        and row["schema_version"] == SCHEMA_VERSION
        and row["gate"] == GATE
        and row["type"] == "request"
        and row["sequence"] == sequence
        and row["request_id"]
        == _request_id(sequence, expected["session"], expected["turn"])
        and row["session"] == expected["session"]
        and row["turn"] == expected["turn"]
        and row["prompt_token_ids_sha256"]
        == expected["prompt_token_ids_sha256"]
        and row["expected_cached_tokens"]
        == expected["expected_cached_tokens"]
        and row["status"] == "success"
        and row["finish_reason"] == "length"
        and row["error_type"] is None
        and isinstance(row["output_token_ids"], list)
        and len(row["output_token_ids"]) == MAX_TOKENS
        and all(type(token_id) is int for token_id in row["output_token_ids"])
        and _request_usage_valid(row["usage"], prompt_tokens=prompt_tokens)
    )


def _accounting_row_valid(value: object, *, rank: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "rank",
        "world_size",
        "capture_action",
        "observation_limit",
        "overflow",
        "capture_error",
        "request_count",
        "prompt_tokens",
        "cached_tokens",
        "observations_sha256",
        "observations",
    }:
        return False
    observations = value["observations"]
    if not isinstance(observations, list):
        return False
    return (
        value["rank"] == rank
        and value["world_size"] == EXPERT_PARALLEL_SIZE
        and value["capture_action"] == "finish"
        and value["observation_limit"] == ACCOUNTING_OBSERVATION_LIMIT
        and value["overflow"] is False
        and value["capture_error"] is None
        and value["request_count"] == len(observations) == EXPECTED_REQUESTS
        and value["prompt_tokens"]
        == sum(
            item.get("prompt_tokens", -1)
            for item in observations
            if isinstance(item, dict)
        )
        and value["cached_tokens"]
        == sum(
            item.get("cached_tokens", -1)
            for item in observations
            if isinstance(item, dict)
        )
        and isinstance(value["observations_sha256"], str)
        and value["observations_sha256"] == sha256_json(observations)
    )


def _parse_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    if len(rows) < 2 or rows[0].get("type") != "run":
        raise ValueError("raw artifact must start with one run row")
    if rows[-1].get("type") != "run_end":
        raise ValueError("raw artifact must end with one run_end row")
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise ValueError("raw artifact row indices are not contiguous")
    if any(
        row.get("schema_version") != SCHEMA_VERSION or row.get("gate") != GATE
        for row in rows
    ):
        raise ValueError("raw artifact schema/gate mismatch")
    by_type: dict[str, list[dict[str, object]]] = {
        "request": [],
        "cache_event": [],
        "rank_accounting": [],
        "kernel_rank": [],
    }
    for row in rows[1:-1]:
        row_type = row.get("type")
        if row_type not in by_type:
            raise ValueError(f"unknown raw row type {row_type!r}")
        by_type[str(row_type)].append(row)
    return (
        rows[0],
        by_type["request"],
        by_type["cache_event"],
        by_type["rank_accounting"],
        by_type["kernel_rank"],
        rows[-1],
    )


def evaluate_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[dict[str, bool], dict[str, object]]:
    run, requests, events, rank_rows, kernel_rows, run_end = _parse_rows(rows)
    run_valid = _run_valid(run)
    end_valid = _run_end_valid(run_end)
    trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
    trace_value = trace.get("value") if isinstance(trace, dict) else None
    trace_rows = (
        trace_value.get("rows", []) if isinstance(trace_value, dict) else []
    )

    request_by_sequence: dict[int, dict[str, object]] = {}
    request_rows_exact = len(requests) == EXPECTED_REQUESTS
    successes_exact = len(requests) == EXPECTED_REQUESTS
    usages_complete = len(requests) == EXPECTED_REQUESTS
    for row in requests:
        sequence = row.get("sequence")
        if type(sequence) is not int or sequence in request_by_sequence:
            request_rows_exact = successes_exact = usages_complete = False
            continue
        request_by_sequence[sequence] = row
    for sequence in range(EXPECTED_REQUESTS):
        row = request_by_sequence.get(sequence)
        expected = (
            trace_rows[sequence]
            if sequence < len(trace_rows) and isinstance(trace_rows[sequence], dict)
            else None
        )
        if row is None or expected is None:
            request_rows_exact = successes_exact = usages_complete = False
            continue
        valid = _request_row_valid(row, expected=expected)
        request_rows_exact &= valid
        successes_exact &= (
            row.get("status") == "success"
            and row.get("error_type") is None
            and row.get("finish_reason") == "length"
        )
        usages_complete &= _request_usage_valid(
            row.get("usage"), prompt_tokens=expected["prompt_tokens"]
        )

    valid_usages = [
        row["usage"]
        for row in requests
        if isinstance(row.get("usage"), dict)
        and type(row["usage"].get("prompt_tokens")) is int
        and type(row["usage"].get("cached_tokens")) is int
    ]
    logical_prompt_tokens = sum(item["prompt_tokens"] for item in valid_usages)
    logical_cached_tokens = sum(item["cached_tokens"] for item in valid_usages)
    logical_rate = (
        logical_cached_tokens / logical_prompt_tokens
        if logical_prompt_tokens
        else None
    )
    rate_above_floor = (
        usages_complete
        and logical_rate is not None
        and logical_rate > CACHE_RATE_FLOOR
    )

    event_rows_exact = len(events) == EXPECTED_CACHE_EVENTS
    stored_hashes: set[str] = set()
    for expected_sequence, row in enumerate(events):
        payload = row.get("payload")
        hashes = payload.get("block_hashes", []) if isinstance(payload, dict) else []
        parent = payload.get("parent_block_hash") if isinstance(payload, dict) else None
        valid = (
            row.get("event_sequence") == expected_sequence
            and _event_payload_valid(row, trace=trace)
            and isinstance(hashes, list)
            and not any(item in stored_hashes for item in hashes)
            and (parent is None or parent in stored_hashes)
        )
        event_rows_exact &= valid
        if valid:
            stored_hashes.update(hashes)
    event_rows_exact &= len(stored_hashes) == MAX_RETAINED_PAGES

    rank_by_rank: dict[int, dict[str, object]] = {}
    accounting_complete = len(rank_rows) == EXPERT_PARALLEL_SIZE
    observations_match = accounting_complete and request_rows_exact
    page_identity_exact = accounting_complete
    rank_prompt_tokens: dict[int, int] = {}
    rank_cached_tokens: dict[int, int] = {}
    page_identity_by_sequence: dict[int, str] = {}
    first_num_tokens_by_sequence: dict[int, int] = {}
    for wrapper in rank_rows:
        if not isinstance(wrapper, dict) or set(wrapper) != {
            "schema_version",
            "gate",
            "type",
            "row_index",
            "rank",
            "evidence",
        }:
            accounting_complete = observations_match = page_identity_exact = False
            continue
        rank = wrapper.get("rank")
        evidence = wrapper.get("evidence")
        if type(rank) is not int or rank in rank_by_rank:
            accounting_complete = observations_match = page_identity_exact = False
            continue
        rank_by_rank[rank] = wrapper
        valid = _accounting_row_valid(evidence, rank=rank)
        accounting_complete &= valid
        if not valid:
            observations_match = page_identity_exact = False
            continue
        assert isinstance(evidence, dict)
        rank_prompt_tokens[rank] = int(evidence["prompt_tokens"])
        rank_cached_tokens[rank] = int(evidence["cached_tokens"])
        for sequence, observation in enumerate(evidence["observations"]):
            logical = request_by_sequence.get(sequence)
            usage = logical.get("usage") if isinstance(logical, dict) else None
            expected = (
                trace_rows[sequence]
                if sequence < len(trace_rows) and isinstance(trace_rows[sequence], dict)
                else None
            )
            if (
                not isinstance(observation, dict)
                or set(observation)
                != {
                    "sequence",
                    "request_id",
                    "prompt_tokens",
                    "cached_tokens",
                    "prefill_start",
                    "first_num_tokens",
                    "first_is_prefill",
                    "prompt_token_ids_sha256",
                    "page_count",
                    "page_ids_sha256",
                }
                or not isinstance(usage, dict)
                or expected is None
            ):
                observations_match = page_identity_exact = False
                continue
            cached = usage.get("cached_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            observations_match &= (
                observation["sequence"] == sequence
                and observation["request_id"] == logical.get("request_id")
                and observation["prompt_tokens"] == prompt_tokens
                and observation["cached_tokens"] == cached
                and type(cached) is int
                and type(prompt_tokens) is int
                and 0 <= cached <= prompt_tokens
                and observation["prefill_start"]
                == min(cached, prompt_tokens - 1)
                and type(observation["first_num_tokens"]) is int
                and 0
                < observation["first_num_tokens"]
                <= prompt_tokens - observation["prefill_start"]
                and observation["first_is_prefill"] is True
                and observation["prompt_token_ids_sha256"]
                == logical.get("prompt_token_ids_sha256")
                and observation["page_count"]
                == math.ceil(expected["prompt_tokens"] / PAGE_SIZE)
                and isinstance(observation["page_ids_sha256"], str)
                and _HEX64.fullmatch(observation["page_ids_sha256"]) is not None
            )
            previous = page_identity_by_sequence.setdefault(
                sequence, observation.get("page_ids_sha256")
            )
            page_identity_exact &= previous == observation.get("page_ids_sha256")
            previous_first_num = first_num_tokens_by_sequence.setdefault(
                sequence, observation.get("first_num_tokens")
            )
            observations_match &= (
                previous_first_num == observation.get("first_num_tokens")
            )
    accounting_complete &= set(rank_by_rank) == set(range(EXPERT_PARALLEL_SIZE))
    observations_match &= accounting_complete and all(
        rank_prompt_tokens.get(rank) == logical_prompt_tokens
        and rank_cached_tokens.get(rank) == logical_cached_tokens
        for rank in range(EXPERT_PARALLEL_SIZE)
    )
    page_identity_exact &= (
        accounting_complete
        and len(page_identity_by_sequence) == EXPECTED_REQUESTS
    )

    kernel_by_rank: dict[int, object] = {}
    kernel_wrappers_exact = len(kernel_rows) == EXPERT_PARALLEL_SIZE
    for wrapper in kernel_rows:
        if not isinstance(wrapper, dict) or set(wrapper) != {
            "schema_version",
            "gate",
            "type",
            "row_index",
            "rank",
            "evidence",
        }:
            kernel_wrappers_exact = False
            continue
        rank = wrapper.get("rank")
        if type(rank) is not int or rank in kernel_by_rank:
            kernel_wrappers_exact = False
            continue
        kernel_by_rank[rank] = wrapper.get("evidence")
    kernel_inventory = [
        kernel_by_rank.get(rank) for rank in range(EXPERT_PARALLEL_SIZE)
    ]
    topology_kernel_exact = (
        run_valid
        and kernel_wrappers_exact
        and set(kernel_by_rank) == set(range(EXPERT_PARALLEL_SIZE))
        and _kernel_inventory_valid(kernel_inventory)
    )

    expected_counts = dict(
        sorted(Counter(str(row["type"]) for row in rows[:-1]).items())
    )
    end_counts_exact = end_valid and run_end.get("row_counts_before_end") == expected_counts
    provenance_exact = (
        run_valid
        and end_valid
        and run.get("source_start") == run_end.get("source_end")
        and run.get("checkpoint_start") == run_end.get("checkpoint_end")
        and run.get("environment_start") == run_end.get("environment_end")
    )
    checks = {
        "formal_run_metadata_exact": run_valid and end_counts_exact,
        "fixed_a7_512_request_trace_exact": (
            run_valid and len(trace_rows) == EXPECTED_REQUESTS
        ),
        "every_request_present_exactly_once": request_rows_exact,
        "all_512_requests_successful": successes_exact,
        "terminal_streamupdate_usage_complete": usages_complete,
        "logical_engine_cache_rate_strictly_above_0_80": rate_above_floor,
        "raw_radix_blockstored_events_complete_and_exact": event_rows_exact,
        "ep4_rank_accounting_complete": accounting_complete,
        "every_rank_observation_matches_logical_engine_truth": observations_match,
        "rank_page_identity_has_no_drift": page_identity_exact,
        "ep4_topology_sampling_and_kernel_evidence_exact": topology_kernel_exact,
        "source_checkpoint_container_and_gpu_start_end_exact": provenance_exact,
    }
    metrics = {
        "request_count": len(requests),
        "success_count": sum(row.get("status") == "success" for row in requests),
        "logical_prompt_tokens": logical_prompt_tokens,
        "logical_cached_tokens": logical_cached_tokens,
        "logical_engine_cache_rate": logical_rate,
        "cache_rate_definition": formal_config()["binding"]["cache_rate"]["definition"],
        "rank_rows_are_not_aggregated_into_rate": True,
        "rank_count": len(rank_by_rank),
        "rank_prompt_tokens": {
            str(rank): value for rank, value in rank_prompt_tokens.items()
        },
        "rank_cached_tokens": {
            str(rank): value for rank, value in rank_cached_tokens.items()
        },
        "cache_event_count": len(events),
        "retained_block_hash_count": len(stored_hashes),
        "kernel_rank_count": len(kernel_by_rank),
    }
    return checks, metrics


def recompute_manifest(
    rows: Sequence[dict[str, object]], *, raw_sha256: str
) -> dict[str, object]:
    if _HEX64.fullmatch(raw_sha256) is None:
        raise ValueError("manifest requires a lowercase raw SHA-256")
    checks, metrics = evaluate_rows(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "config": formal_config(),
        "raw": {
            "name": RAW_NAME,
            "sha256": raw_sha256,
            "row_count": len(rows),
        },
        "checks": checks,
        "metrics": metrics,
        "passed": all(checks.values()),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _strict_json_loads(value: str) -> object:
    return ma1.strict_json_loads(value)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise ValueError(f"{path}:{line_number}: blank JSONL row")
        value = _strict_json_loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _with_row_indices(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        # Convert tuples returned by rank probes to their retained JSON form
        # before the pre-write replay.  The replay must see exactly what a
        # later process reads from JSONL, not a Python-only container variant.
        copied = _strict_json_loads(canonical_json(dict(row)))
        if not isinstance(copied, dict):
            raise TypeError("raw artifact row did not normalize to an object")
        copied["row_index"] = index
        result.append(copied)
    return result


def write_artifact(
    output_dir: Path, rows: Sequence[dict[str, object]]
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    atomic_write_text(
        raw_path,
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    stored_rows = _read_jsonl(raw_path)
    manifest = recompute_manifest(
        stored_rows, raw_sha256=sha256_file(raw_path)
    )
    atomic_write_json(
        output_dir / MANIFEST_NAME,
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
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
        f"artifact must be a directory, {RAW_NAME}, or {MANIFEST_NAME}"
    )


def verify_artifact(
    path: Path, *, assert_gate: bool = False
) -> dict[str, object]:
    raw_path, manifest_path = _artifact_paths(path)
    rows = _read_jsonl(raw_path)
    manifest = _strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    raw_sha256 = sha256_file(raw_path)
    raw = manifest.get("raw")
    if not isinstance(raw, dict) or raw.get("sha256") != raw_sha256:
        raise ValueError("raw SHA-256 does not match the manifest")
    replayed = recompute_manifest(rows, raw_sha256=raw_sha256)
    if manifest != replayed:
        raise ValueError("manifest differs from independent raw replay")
    if assert_gate and replayed["passed"] is not True:
        raise RuntimeError("G4 M-A2 artifact does not pass every binding check")
    return replayed


def replay_artifact(
    path: Path, *, assert_gate: bool = False
) -> dict[str, object]:
    raw_path, _manifest_path = _artifact_paths(path)
    replayed = recompute_manifest(
        _read_jsonl(raw_path), raw_sha256=sha256_file(raw_path)
    )
    if assert_gate and replayed["passed"] is not True:
        raise RuntimeError("G4 M-A2 raw replay does not pass every binding check")
    return replayed


def _runtime_value(
    *,
    source: Mapping[str, object],
    container: Mapping[str, object],
    image_repo_digest: str,
    image_config_id: str,
) -> dict[str, object]:
    return {
        "name": "kairyu",
        "version": ma1_capture._package_version("kairyu"),
        "image_repo_digest": image_repo_digest,
        "image_config_id": image_config_id,
        "backend": "pytorch",
        "result_api": ma1.KAIRYU_RESULT_API,
        "source_commit": source["commit"],
        "container_observation": dict(container),
    }


def capture_formal_rows(
    *,
    model_path: Path,
    container_inspect: Path,
    image_repo_digest: str,
    image_config_id: str,
    launcher_type: object | None = None,
    tokenizer: object | None = None,
) -> list[dict[str, object]]:
    """Execute the one real EP4 trace and return canonical, replayable rows."""

    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.engine_loop import EngineLoop
    from kairyu.engine.prompt import TokensPrompt
    from kairyu.sampling_params import SamplingParams

    model_path = model_path.resolve()
    created_at = ma1_capture._now()
    source_start = _source_snapshot()
    checkpoint_start = ma1_capture._checkpoint_snapshot(model_path)
    if tokenizer is None:
        from kairyu.engine.tokenizer import resolve_tokenizer

        tokenizer = resolve_tokenizer(str(model_path))
    prompts = build_workload(tokenizer)
    trace = trace_descriptor(
        prompts,
        tokenizer_sha256=str(checkpoint_start["tokenizer_rollup_sha256"]),
    )
    environment_start = ma1_capture._environment(_ARM)
    container = ma1_capture._container_observation(
        container_inspect,
        image_repo_digest=image_repo_digest,
        image_config_id=image_config_id,
    )
    runtime = _runtime_value(
        source=source_start,
        container=container,
        image_repo_digest=image_repo_digest,
        image_config_id=image_config_id,
    )
    if not ma1._runtime_valid(
        runtime,
        arm=_ARM,
        run_image_repo_digest=image_repo_digest,
        run_image_config_id=image_config_id,
        source_commit=source_start["commit"],
    ):
        raise ValueError("container/runtime evidence is not bound to this source")
    checkpoint_view = {
        "kind": "original",
        "model_path": str(model_path),
        "hf_quant_config_sha256": checkpoint_start["hf_quant_config_sha256"],
    }
    if launcher_type is None:
        from kairyu.engine.core.worker import DistEPLauncher

        launcher_type = DistEPLauncher

    events: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []
    current_request_id: str | None = None

    def event_sink(payload: Mapping[str, object]) -> None:
        if current_request_id is None:
            raise RuntimeError("Radix emitted an event outside the serialized request")
        events.append(
            {
                "schema_version": SCHEMA_VERSION,
                "gate": GATE,
                "type": "cache_event",
                "event_sequence": len(events),
                "request_id": current_request_id,
                "payload": copy.deepcopy(dict(payload)),
            }
        )

    launcher = None
    loop = None
    accounting_start: list[dict[str, object]] | None = None
    accounting_finish: list[dict[str, object]] | None = None
    kernel_inventory: list[dict[str, object]] | None = None
    topology: dict[str, object] | None = None
    sampling_ownership: list[dict[str, object]] | None = None
    attention_backend: dict[str, object] | None = None
    try:
        launcher = launcher_type(
            str(model_path),
            EXPERT_PARALLEL_SIZE,
            NUM_PAGES,
            PAGE_SIZE,
            tokenizer.vocab(),
            pipeline_depth=1,
            decode_mode="eager",
            kv_cache_dtype="bfloat16",
        )
        metadata = launcher.parallelism_metadata()
        if metadata != _expected_parallelism_metadata():
            raise RuntimeError(f"unexpected DistEPLauncher topology: {metadata!r}")
        topology = _topology_value(metadata, environment_start)
        observed_sampling = launcher.runner.sampling_ownership_metadata()
        sampling_ownership = [dict(row) for row in observed_sampling]
        if sampling_ownership != _sampling_ownership_expected():
            raise RuntimeError("EP sampling ownership is not rank-0 authoritative")
        start_value = _strict_json_loads(
            canonical_json(list(launcher.runner.start_ep_kv_accounting()))
        )
        if not isinstance(start_value, list):
            raise RuntimeError("EP KV accounting start probe is not a JSON list")
        accounting_start = start_value
        if not _accounting_start_valid(accounting_start):
            raise RuntimeError("EP KV accounting did not arm on all four ranks")

        cache = RadixKVCache(
            num_pages=NUM_PAGES,
            page_size=PAGE_SIZE,
            event_sink=event_sink,
        )
        scheduler = Scheduler(
            cache,
            max_num_batched_tokens=max(
                len(prompt.prompt_token_ids) for prompt in prompts
            ),
            max_num_seqs=1,
            page_size=PAGE_SIZE,
        )
        loop = EngineLoop(
            tokenizer,
            scheduler,
            launcher.runner,
            pipeline_depth=1,
            max_model_len=max(
                len(prompt.prompt_token_ids) for prompt in prompts
            )
            + MAX_TOKENS,
        )
        params = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            seed=a7.WORKLOAD_SEED,
            ignore_eos=True,
            skip_special_tokens=False,
        )
        for prompt in prompts:
            request_id = _request_id(prompt.sequence, prompt.session, prompt.turn)
            current_request_id = request_id
            loop.submit(
                request_id,
                TokensPrompt(prompt.prompt_token_ids, prompt.prompt),
                params,
            )
            terminal = None
            steps = 0
            while loop.has_work():
                steps += 1
                if steps > MAX_ENGINE_STEPS_PER_REQUEST:
                    raise RuntimeError(
                        f"serialized request {request_id} exceeded the step bound"
                    )
                for update_id, update in loop.step():
                    if update_id != request_id:
                        raise RuntimeError("serialized loop emitted another request")
                    if update.error is not None:
                        raise update.error
                    if update.finished:
                        if terminal is not None:
                            raise RuntimeError("request emitted two terminal updates")
                        terminal = update
            if terminal is None:
                raise RuntimeError(f"request {request_id} has no terminal update")
            requests.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "gate": GATE,
                    "type": "request",
                    "sequence": prompt.sequence,
                    "request_id": request_id,
                    "session": prompt.session,
                    "turn": prompt.turn,
                    "prompt_token_ids_sha256": prompt.prompt_token_ids_sha256,
                    "expected_cached_tokens": prompt.expected_cached_tokens,
                    "status": "success",
                    "finish_reason": terminal.finish_reason,
                    "error_type": None,
                    "output_token_ids": list(terminal.outputs),
                    "usage": {
                        "prompt_tokens": terminal.num_prompt_tokens,
                        "cached_tokens": terminal.num_cached_tokens,
                        "completion_tokens": len(terminal.outputs),
                    },
                }
            )
            current_request_id = None

        accounting_finish = [
            dict(row) for row in launcher.runner.finish_ep_kv_accounting()
        ]
        kernel_inventory = [
            dict(row) for row in launcher.ep_kernel_inventory_metadata()
        ]
        decision = launcher.attention_backend_decision
        attention_backend = {
            "requested": decision.requested,
            "resolved": decision.resolved,
            "components": dict(decision.components),
            "identity": launcher.attention_backend_identity,
        }
    finally:
        # Keep the current request identity through loop cleanup: aborting an
        # in-progress prefill can legitimately publish/release Radix evidence.
        # Launcher teardown still runs if loop.close itself reports a failure.
        try:
            if loop is not None:
                loop.close()
        finally:
            current_request_id = None
            if launcher is not None:
                launcher.shutdown()

    if any(
        value is None
        for value in (
            accounting_start,
            accounting_finish,
            kernel_inventory,
            topology,
            sampling_ownership,
            attention_backend,
        )
    ):
        raise RuntimeError("formal EP4 run did not produce complete evidence")

    environment_end = ma1_capture._environment(_ARM)
    checkpoint_end = ma1_capture._checkpoint_snapshot(model_path)
    source_end = _source_snapshot()
    run = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "type": "run",
        "created_at": created_at,
        "config": formal_config(),
        "trace": trace,
        "runtime": runtime,
        "source_start": source_start,
        "checkpoint_start": checkpoint_start,
        "checkpoint_view": checkpoint_view,
        "environment_start": environment_start,
        "topology": topology,
        "sampling_ownership": sampling_ownership,
        "attention_backend": attention_backend,
        "accounting_start": accounting_start,
    }
    raw_rows: list[dict[str, object]] = [run, *requests, *events]
    raw_rows.extend(
        {
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "type": "rank_accounting",
            "rank": rank,
            "evidence": accounting_finish[rank],
        }
        for rank in range(EXPERT_PARALLEL_SIZE)
    )
    raw_rows.extend(
        {
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "type": "kernel_rank",
            "rank": rank,
            "evidence": kernel_inventory[rank],
        }
        for rank in range(EXPERT_PARALLEL_SIZE)
    )
    counts = dict(sorted(Counter(row["type"] for row in raw_rows).items()))
    raw_rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "type": "run_end",
            "finished_at": ma1_capture._now(),
            "source_end": source_end,
            "checkpoint_end": checkpoint_end,
            "environment_end": environment_end,
            "row_counts_before_end": counts,
        }
    )
    indexed = _with_row_indices(raw_rows)
    return indexed


def run_gate(
    *,
    model_path: Path,
    container_inspect: Path,
    image_repo_digest: str,
    image_config_id: str,
    output_dir: Path,
) -> dict[str, object]:
    rows = capture_formal_rows(
        model_path=model_path,
        container_inspect=container_inspect,
        image_repo_digest=image_repo_digest,
        image_config_id=image_config_id,
    )
    return write_artifact(output_dir, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute the real EP4 gate")
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--container-inspect", type=Path, required=True)
    run_parser.add_argument("--image-repo-digest", required=True)
    run_parser.add_argument("--image-config-id", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--assert-gate", action="store_true")

    for command, help_text in (
        ("verify", "compare manifest with an independent raw replay"),
        ("replay", "recompute the verdict from raw JSONL only"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--artifact", type=Path, required=True)
        child.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_gate(
            model_path=args.model_path,
            container_inspect=args.container_inspect,
            image_repo_digest=args.image_repo_digest,
            image_config_id=args.image_config_id,
            output_dir=args.output_dir,
        )
    elif args.command == "verify":
        result = verify_artifact(args.artifact, assert_gate=args.assert_gate)
    else:
        result = replay_artifact(args.artifact, assert_gate=args.assert_gate)
    print(canonical_json(result))
    return 1 if args.assert_gate and result["passed"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
