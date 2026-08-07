#!/usr/bin/env python3
"""G2 A6: replayable Kairyu versus stock-vLLM serving comparison.

The binding experiment is deliberately narrow: one fixed-concurrency-128
ShareGPT burst and one serialized fixed shared-prefix trace, repeated four
times at TP4 and TP8 in the balanced K/V, V/K, V/K, K/V server-start order.
The predeclared 1/2/4/8/16-rps open-loop grid is retained as report-only
saturation evidence on a fresh server per point; no observed rate is selected
afterward, and the grid is not substituted for the acceptance burst.

``measure`` writes one fresh-server scenario shard. ``assemble`` requires the
complete 32-cell binding matrix plus all 20 report-only sweep points, then
writes canonical raw JSONL plus a derived manifest.
``verify`` hashes the raw stream and independently recomputes all metrics and
checks. Warmup rows stay in raw evidence but never enter an aggregate. Requests
are attempted exactly once; a failed request is retained and later planned
requests still run. The HTTP prompt is the same retained UTF-8 text for both
arms so TTFT includes server tokenization; pinned token IDs are the identity and
response-usage oracle, not the request carrier.

Server lifecycle is an explicit operator boundary: this module never launches,
restarts, or stops Docker. For each shard the operator starts one server,
writes its attested provenance JSON, runs ``measure`` or ``measure-sweep``, and
then stops it. Assembly rejects reused server-generation IDs, which makes a
silent warm-cache reuse fail closed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
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
from bench.tp_kv_hit_g2_a7_bench import (  # noqa: E402
    _CHECKPOINT_HASH_SCRIPT,
    MODEL,
    MODEL_REVISION,
    MODEL_WEIGHTS_ROLLUP,
    _expected_checkpoint_identity,
)
from kairyu.engine.tokenizer import HFTokenizer  # noqa: E402

SCHEMA_VERSION = "kairyu.g2.a6.vllm-comparison.v1"
GATE = "G2-A6"
RAW_NAME = "g2-a6-vllm-raw.jsonl"
MANIFEST_NAME = "g2-a6-vllm-manifest.json"
TRACE_BUNDLE_SCHEMA = "kairyu.g2.a6.trace-bundle.v2"

TP_DEGREES = (4, 8)
ARMS = ("kairyu", "vllm-stock")
WORKLOADS = ("sharegpt", "shared-prefix")
REPEATS = 4
PAIR_ORDER = (
    ("kairyu", "vllm-stock"),
    ("vllm-stock", "kairyu"),
    ("vllm-stock", "kairyu"),
    ("kairyu", "vllm-stock"),
)
EXPECTED_SCENARIOS = tuple(
    (tp, workload, repeat, arm)
    for tp in TP_DEGREES
    for workload in WORKLOADS
    for repeat in range(REPEATS)
    for arm in ARMS
)
OPEN_LOOP_RATES_RPS = (1, 2, 4, 8, 16)
EXPECTED_SWEEP_SCENARIOS = tuple(
    (tp, rate_rps, arm) for tp in TP_DEGREES for rate_rps in OPEN_LOOP_RATES_RPS for arm in ARMS
)

PINNED_TOKENIZER_SHA256 = _expected_checkpoint_identity()["tokenizer_sha256"]["tokenizer.json"]
MODEL_DTYPE = "bfloat16"
VLLM_VERSION = "0.26.0+cu129"
VLLM_BUILD_COMMIT = "ffd46bfab2128bb84146050e98b51a617c6575ab"
VLLM_IMAGE_REF = "vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404"
VLLM_IMAGE_ID = "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8"
VLLM_CUDA_VERSION = "12.9"
VLLM_NCCL_VERSION = "2.28.9"
KAIRYU_CUDA_VERSION = "13.0"
KAIRYU_NCCL_VERSION = "2.29.7"
UVLOOP_VERSION = "0.22.1"
HTTPTOOLS_VERSION = "0.8.0"
KAIRYU_UVICORN_VERSION = "0.49.0"
VLLM_UVICORN_VERSION = "0.51.0"

SHAREGPT_DATASET_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_DATASET_REVISION = "745745adf6cd15b84e4f1c4a5a051fb4304f9342"
SHAREGPT_DATASET_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_DATASET_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
SHAREGPT_DATASET_SEED = 0
SHAREGPT_REQUESTS = 128
SHAREGPT_CONCURRENCY = 128
SHAREGPT_MAX_TOKENS = 128
SHAREGPT_MIN_TOKENS = 128
SHAREGPT_MIN_PROMPT_TOKENS = 4
SHAREGPT_MAX_PROMPT_TOKENS = 1024
SHARED_PREFIX_REQUESTS = NUM_SESSIONS * TURNS_PER_SESSION
SHARED_PREFIX_MAX_TOKENS = 1
SHARED_PREFIX_MIN_TOKENS = 1
WARMUP_REQUESTS = 4
WARMUP_PROMPT_TOKENS = 64
WARMUP_MAX_TOKENS = 8
GRAPH_WARMUP_BATCH_SIZES = (1, 2, 4, 8, 16)
GRAPH_WARMUP_REQUESTS = sum(GRAPH_WARMUP_BATCH_SIZES)
GRAPH_WARMUP_PROMPT_TOKENS = 64
GRAPH_WARMUP_COMPLETION_TOKENS = 2
TTFT_SLO_NS = 10_000_000_000
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = -1
SEED = 0
N = 1
IGNORE_EOS = True
MAX_NUM_BATCHED_TOKENS = 1024
MAX_NUM_SEQS = 16
CACHE_BLOCK_SIZE = 16
USABLE_CACHE_BLOCKS = 8192
KAIRYU_ALLOCATED_CACHE_BLOCKS = USABLE_CACHE_BLOCKS + 1
VLLM_ALLOCATED_CACHE_BLOCKS = USABLE_CACHE_BLOCKS + 1
KAIRYU_GRAPH_SCRATCH_BLOCKS = 1
VLLM_GRAPH_SCRATCH_BLOCKS = 0
KAIRYU_NULL_BLOCKS = 0
VLLM_NULL_BLOCKS = 1
CACHE_BLOCKS = USABLE_CACHE_BLOCKS
CACHE_TOKENS = CACHE_BLOCK_SIZE * USABLE_CACHE_BLOCKS
KV_CACHE_DTYPE = "bfloat16"
USABLE_KV_CACHE_BYTES_PER_RANK = {
    4: 8 * 1024**3,
    8: 4 * 1024**3,
}
KV_BLOCK_BYTES_PER_RANK = {
    tp: value // USABLE_CACHE_BLOCKS
    for tp, value in USABLE_KV_CACHE_BYTES_PER_RANK.items()
}
ALLOCATED_KV_CACHE_BYTES_PER_RANK = {
    tp: KV_BLOCK_BYTES_PER_RANK[tp] * KAIRYU_ALLOCATED_CACHE_BLOCKS
    for tp in TP_DEGREES
}
RESERVED_KV_CACHE_BYTES_PER_RANK = dict(KV_BLOCK_BYTES_PER_RANK)
MAX_MODEL_LEN = 8192
ACTIVE_DECODE_GRAPH_CAP = 16
VLLM_CUDA_GRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)
KAIRYU_CUDA_GRAPH_CAPTURE_SIZES = GRAPH_WARMUP_BATCH_SIZES
KAIRYU_PIPELINE_DEPTH = 5
ACCESS_LOG_ENABLED = False
VLLM_ASYNC_SCHEDULING = True
VLLM_ATTENTION_BACKEND = "FLASH_ATTN"
VLLM_FLASH_ATTN_VERSION = 2
VLLM_ATTENTION_BACKEND_LOG_MARKER = (
    "Using AttentionBackendEnum.FLASH_ATTN backend."
)
VLLM_DISTRIBUTED_EXECUTOR_BACKEND = "mp"
VLLM_CUSTOM_ALL_REDUCE = False
VLLM_COMPILATION_MODE = 3
VLLM_CUDAGRAPH_MODE = "FULL_AND_PIECEWISE"
SOURCE_CLEAN_EXCEPTION_PREFIX = ".claude/worktrees/"
CHECKPOINT_ROOT = "/models/qwen3-32b"
CHECKPOINT_HASH_SCRIPT = _CHECKPOINT_HASH_SCRIPT
DEFAULT_TIMEOUT_S = 600.0

GOODPUT_RATIO_FLOOR = 0.95
GOODPUT_RATIO_FLOOR_FRACTION = (95, 100)
SHAREGPT_P99_RATIO_CEILING = 1.0
SHARED_PREFIX_P50_RATIO_CEILING = 0.80
SHARED_PREFIX_P50_RATIO_CEILING_FRACTION = (4, 5)
_HEX = frozenset("0123456789abcdef")


class _Tokenizer(Protocol):
    def encode(self, text: str) -> tuple[int, ...]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def vocab(self) -> list[str]: ...


@dataclass(frozen=True)
class PlannedRequest:
    phase: str
    sequence: int
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    expected_completion_tokens: int
    graph_batch_size: int | None = None
    graph_batch_sequence: int | None = None

    @property
    def prompt_text_sha256(self) -> str:
        return sha256_text(self.prompt_text)

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


def model_identity() -> dict[str, str]:
    return {
        "name": MODEL,
        "revision": MODEL_REVISION,
        "weights_sha256": MODEL_WEIGHTS_ROLLUP,
        "tokenizer_sha256": PINNED_TOKENIZER_SHA256,
        "dtype": MODEL_DTYPE,
    }


def vllm_compilation_config() -> dict[str, object]:
    return {
        "mode": VLLM_COMPILATION_MODE,
        "cudagraph_mode": VLLM_CUDAGRAPH_MODE,
        "cudagraph_capture_sizes": list(VLLM_CUDA_GRAPH_CAPTURE_SIZES),
    }


def expected_engine_argv(*, arm: str, tp: int) -> list[str]:
    if tp not in TP_DEGREES:
        raise ValueError(f"unsupported TP degree {tp}")
    if arm == "kairyu":
        return ["serve", "/etc/kairyu/config.yaml"]
    if arm != "vllm-stock":
        raise ValueError(f"unknown arm {arm!r}")
    return [
        "serve",
        CHECKPOINT_ROOT,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        MODEL,
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        MODEL_DTYPE,
        "--kv-cache-dtype",
        KV_CACHE_DTYPE,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--block-size",
        str(CACHE_BLOCK_SIZE),
        "--num-gpu-blocks-override",
        str(VLLM_ALLOCATED_CACHE_BLOCKS),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--scheduling-policy",
        "fcfs",
        "--seed",
        str(SEED),
        "--attention-backend",
        VLLM_ATTENTION_BACKEND,
        "--distributed-executor-backend",
        VLLM_DISTRIBUTED_EXECUTOR_BACKEND,
        "--disable-custom-all-reduce",
        "--async-scheduling",
        "--disable-uvicorn-access-log",
        "--no-enable-log-requests",
        "--compilation-config",
        canonical_json(vllm_compilation_config()),
    ]


def expected_container_launch(*, arm: str, tp: int) -> dict[str, object]:
    return {
        "entrypoint": (
            ["/app/.venv/bin/kairyu"]
            if arm == "kairyu"
            else ["/usr/local/bin/vllm"]
        ),
        "argv": expected_engine_argv(arm=arm, tp=tp),
        "environment": (
            {
                "KAIRYU_ATTENTION_BACKEND": "flashinfer",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            if arm == "kairyu"
            else {
                "VLLM_USE_V1": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        ),
    }


def expected_kairyu_config(tp: int) -> dict[str, object]:
    if tp not in TP_DEGREES:
        raise ValueError(f"unsupported TP degree {tp}")
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "api_keys_env": None,
            "access_log": False,
            "max_concurrency": SHAREGPT_CONCURRENCY,
            "admission_wait_timeout_s": DEFAULT_TIMEOUT_S,
        },
        "engines": {
            "qwen3-32b": {
                "backend": "kairyu",
                "options": {
                    "model_path": CHECKPOINT_ROOT,
                    "tensor_parallel_size": tp,
                    "num_pages": KAIRYU_ALLOCATED_CACHE_BLOCKS,
                    "max_model_len": MAX_MODEL_LEN,
                    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                    "max_num_seqs": MAX_NUM_SEQS,
                    "pipeline_depth": KAIRYU_PIPELINE_DEPTH,
                    "priority_age_s": 60.0,
                    "decode_mode": "cuda_graph",
                    "cuda_graph_max_batch": ACTIVE_DECODE_GRAPH_CAP,
                    "cuda_graph_max_pages": 512,
                    "cuda_graph_warmup_iters": 3,
                },
            }
        },
    }


def kairyu_configuration_source(text: str, *, tp: int) -> dict[str, object]:
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"Kairyu formal YAML is invalid: {error}") from error
    value = {
        "format": "utf-8-yaml",
        "text": text,
        "text_sha256": sha256_text(text),
        "parsed": parsed,
    }
    descriptor = hashed_descriptor(value)
    if not _configuration_source_valid(descriptor, tp=tp, arm="kairyu"):
        raise ValueError("Kairyu formal YAML does not exactly match the pinned config")
    return descriptor


def _configuration_source_valid(
    value: object,
    *,
    tp: int,
    arm: str,
) -> bool:
    if arm == "vllm-stock":
        return value is None
    if arm != "kairyu" or not _descriptor_valid(value):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    if not isinstance(raw, dict) or set(raw) != {
        "format",
        "text",
        "text_sha256",
        "parsed",
    }:
        return False
    text = raw["text"]
    if (
        raw["format"] != "utf-8-yaml"
        or not isinstance(text, str)
        or not text
        or raw["text_sha256"] != sha256_text(text)
        or raw["parsed"] != expected_kairyu_config(tp)
    ):
        return False
    try:
        return yaml.safe_load(text) == raw["parsed"]
    except yaml.YAMLError:
        return False


def expected_resolved_backend(*, arm: str, tp: int) -> dict[str, object]:
    if arm == "kairyu":
        components = {
            "prefill": "flashinfer",
            "decode": "flashinfer",
            "kv_mode": "paged-direct",
        }
        return {
            "attestation": "GET /backends after readiness",
            "attention_backend": "flashinfer",
            "requested_attention_backend": "flashinfer",
            "attention_components": components,
            "source": "env",
            "role": "engine-host",
            "engine_backend": "kairyu",
            "engine_attention_backend": "flashinfer",
            "engine_attention_components": components,
            "engine_decision_status": "actual",
            "tensor_parallel_size": tp,
        }
    if arm != "vllm-stock":
        raise ValueError(f"unknown arm {arm!r}")
    return {
        "attestation": (
            "retained post-start vLLM V1 startup messages plus inspected argv"
        ),
        "attention_backend": VLLM_ATTENTION_BACKEND,
        "attention_backend_version": VLLM_FLASH_ATTN_VERSION,
        "distributed_executor_backend": VLLM_DISTRIBUTED_EXECUTOR_BACKEND,
        "async_scheduling": VLLM_ASYNC_SCHEDULING,
        "custom_all_reduce": VLLM_CUSTOM_ALL_REDUCE,
        "compilation": vllm_compilation_config(),
        "tensor_parallel_size": tp,
    }


def _vllm_startup_markers(
    config_message: str,
    attention_messages: Sequence[str],
    attention_version_messages: Sequence[str],
    async_scheduling_messages: Sequence[str],
    *,
    tp: int,
) -> dict[str, bool]:
    return {
        "v1_engine": "Initializing a V1 LLM engine" in config_message,
        "tensor_parallel_size": f"tensor_parallel_size={tp}" in config_message,
        "max_model_len": f"max_seq_len={MAX_MODEL_LEN}" in config_message,
        "prefix_caching": "enable_prefix_caching=True" in config_message,
        "chunked_prefill": "enable_chunked_prefill=True" in config_message,
        "custom_all_reduce_disabled": (
            "disable_custom_all_reduce=True" in config_message
        ),
        "async_scheduling": bool(async_scheduling_messages)
        and all(
            "Asynchronous scheduling is enabled." in message
            for message in async_scheduling_messages
        ),
        "compilation_mode_vllm_compile": (
            "VLLM_COMPILE" in config_message
            or "'mode': 3" in config_message
            or '"mode": 3' in config_message
        ),
        "cudagraph_mode_full_and_piecewise": (
            VLLM_CUDAGRAPH_MODE in config_message
        ),
        "capture_sizes": (
            str(list(VLLM_CUDA_GRAPH_CAPTURE_SIZES)) in config_message
        ),
        "attention_backend_flash_attn": bool(attention_messages)
        and all(
            VLLM_ATTENTION_BACKEND_LOG_MARKER in message
            for message in attention_messages
        ),
        "attention_backend_version_2": bool(attention_version_messages)
        and all(
            f"Using FlashAttention version {VLLM_FLASH_ATTN_VERSION}" in message
            for message in attention_version_messages
        ),
    }


def _vllm_startup_log_valid(value: object, *, tp: int) -> bool:
    if not _descriptor_valid(value):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    if not isinstance(raw, dict):
        return False
    config_message = raw.get("engine_config_message")
    attention_messages = raw.get("attention_backend_messages")
    attention_version_messages = raw.get("attention_version_messages")
    async_scheduling_messages = raw.get("async_scheduling_messages")
    if (
        not isinstance(config_message, str)
        or not config_message
        or not isinstance(attention_messages, list)
        or not attention_messages
        or not all(isinstance(message, str) and message for message in attention_messages)
        or not isinstance(attention_version_messages, list)
        or not attention_version_messages
        or not all(
            isinstance(message, str) and message
            for message in attention_version_messages
        )
        or not isinstance(async_scheduling_messages, list)
        or not async_scheduling_messages
        or not all(
            isinstance(message, str) and message
            for message in async_scheduling_messages
        )
    ):
        return False
    markers = _vllm_startup_markers(
        config_message,
        attention_messages,
        attention_version_messages,
        async_scheduling_messages,
        tp=tp,
    )
    return (
        set(raw)
        == {
            "engine_config_message",
            "engine_config_line_sha256",
            "attention_backend_messages",
            "attention_backend_line_sha256",
            "attention_version_messages",
            "attention_version_line_sha256",
            "async_scheduling_messages",
            "async_scheduling_line_sha256",
            "resolved_markers",
        }
        and raw["engine_config_line_sha256"] == sha256_text(config_message)
        and raw["attention_backend_line_sha256"]
        == sha256_json(attention_messages)
        and raw["attention_version_line_sha256"]
        == sha256_json(attention_version_messages)
        and raw["async_scheduling_line_sha256"]
        == sha256_json(async_scheduling_messages)
        and raw["resolved_markers"] == markers
        and all(markers.values())
    )


def request_config(workload: str) -> dict[str, object]:
    if workload == "sharegpt":
        maximum = SHAREGPT_MAX_TOKENS
        minimum = SHAREGPT_MIN_TOKENS
    elif workload == "shared-prefix":
        maximum = SHARED_PREFIX_MAX_TOKENS
        minimum = SHARED_PREFIX_MIN_TOKENS
    else:
        raise ValueError(f"unknown workload {workload!r}")
    return {
        "endpoint": "/v1/completions",
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "n": N,
        "max_tokens": maximum,
        "min_tokens": minimum,
        "ignore_eos": IGNORE_EOS,
    }


def benchmark_config(dataset_sha256: str) -> dict[str, object]:
    if not _valid_sha256(dataset_sha256):
        raise ValueError("ShareGPT dataset SHA-256 must be 64 lowercase hex")
    if dataset_sha256 != SHAREGPT_DATASET_SHA256:
        raise ValueError("ShareGPT dataset does not match the pinned upstream file")
    return {
        "model": model_identity(),
        "sharegpt_dataset": {
            "format": "ShareGPT conversations JSON",
            "repository": SHAREGPT_DATASET_REPO,
            "revision": SHAREGPT_DATASET_REVISION,
            "file": SHAREGPT_DATASET_FILE,
            "sha256": dataset_sha256,
            "selection_seed": SHAREGPT_DATASET_SEED,
        },
        "tensor_parallel_degrees": list(TP_DEGREES),
        "arms": list(ARMS),
        "workloads": list(WORKLOADS),
        "repeats": REPEATS,
        "pair_order": [list(pair) for pair in PAIR_ORDER],
        "warmup": {
            "serial_requests_per_scenario": WARMUP_REQUESTS,
            "graph_capture": {
                "batch_sizes": list(GRAPH_WARMUP_BATCH_SIZES),
                "requests_per_scenario": GRAPH_WARMUP_REQUESTS,
                "prompt_tokens": GRAPH_WARMUP_PROMPT_TOKENS,
                "completion_tokens": GRAPH_WARMUP_COMPLETION_TOKENS,
                "synchronized_within_batch": True,
            },
            "retained_in_raw": True,
            "excluded_from_aggregates": True,
        },
        "matched_serving_configuration": {
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_model_len": MAX_MODEL_LEN,
            "prefix_caching": True,
            "cache_block_size": CACHE_BLOCK_SIZE,
            "usable_cache_blocks": USABLE_CACHE_BLOCKS,
            "cache_tokens": CACHE_TOKENS,
            "kv_cache_dtype": KV_CACHE_DTYPE,
            "allocated_kv_cache_bytes_per_rank": {
                str(tp): ALLOCATED_KV_CACHE_BYTES_PER_RANK[tp]
                for tp in TP_DEGREES
            },
            "reserved_kv_cache_bytes_per_rank": {
                str(tp): RESERVED_KV_CACHE_BYTES_PER_RANK[tp]
                for tp in TP_DEGREES
            },
            "usable_kv_cache_bytes_per_rank": {
                str(tp): USABLE_KV_CACHE_BYTES_PER_RANK[tp]
                for tp in TP_DEGREES
            },
            "chunked_prefill": True,
            "scheduler_policy": "fcfs",
            "active_decode_graph_cap": ACTIVE_DECODE_GRAPH_CAP,
            "access_log_enabled": ACCESS_LOG_ENABLED,
            "http_event_loop": "uvloop",
            "uvloop_version": UVLOOP_VERSION,
            "http_parser": "httptools",
            "httptools_version": HTTPTOOLS_VERSION,
        },
        "arm_specific_serving_configuration": {
            "kairyu": {
                "allocated_cache_blocks": KAIRYU_ALLOCATED_CACHE_BLOCKS,
                "reserved_graph_scratch_blocks": KAIRYU_GRAPH_SCRATCH_BLOCKS,
                "reserved_null_blocks": KAIRYU_NULL_BLOCKS,
                "pipeline_depth": KAIRYU_PIPELINE_DEPTH,
                "attention_backend": "flashinfer",
                "distributed_executor_backend": "torch-distributed-nccl",
                "custom_all_reduce": False,
                "cuda_graph_mode": "decode-only",
                "cuda_graph_capture_sizes": list(KAIRYU_CUDA_GRAPH_CAPTURE_SIZES),
            },
            "vllm-stock": {
                "allocated_cache_blocks": VLLM_ALLOCATED_CACHE_BLOCKS,
                "reserved_graph_scratch_blocks": VLLM_GRAPH_SCRATCH_BLOCKS,
                "reserved_null_blocks": VLLM_NULL_BLOCKS,
                "async_scheduling": VLLM_ASYNC_SCHEDULING,
                "attention_backend": VLLM_ATTENTION_BACKEND,
                "distributed_executor_backend": VLLM_DISTRIBUTED_EXECUTOR_BACKEND,
                "custom_all_reduce": VLLM_CUSTOM_ALL_REDUCE,
                "compilation_mode": VLLM_COMPILATION_MODE,
                "cuda_graph_mode": VLLM_CUDAGRAPH_MODE,
                "cuda_graph_capture_sizes": list(VLLM_CUDA_GRAPH_CAPTURE_SIZES),
            },
        },
        "binding_regime": {
            "sharegpt": {
                "kind": "single synchronized burst",
                "requests": SHAREGPT_REQUESTS,
                "concurrency": SHAREGPT_CONCURRENCY,
                "ttft_slo_ns": TTFT_SLO_NS,
                "request": request_config("sharegpt"),
            },
            "shared-prefix": {
                "kind": "serialized fixed trace",
                "requests": SHARED_PREFIX_REQUESTS,
                "sessions": NUM_SESSIONS,
                "turns_per_session": TURNS_PER_SESSION,
                "shared_prefix_tokens": SYSTEM_PROMPT_TOKENS,
                "turn_tokens": TURN_TOKENS,
                "workload_seed": WORKLOAD_SEED,
                "request": request_config("shared-prefix"),
            },
        },
        "aggregation": {
            "quantile": "nearest-rank over all four raw measurement repeats",
            "goodput": (
                "sum(TTFT-SLO successes) / sum(first-start-to-last-terminal "
                "span for each synchronized burst)"
            ),
            "paired_order_diagnostics": (
                "for each repeat compute exact Kairyu/vLLM Fractions, then "
                "take the exact median of four ratios (average the middle two); "
                "retain median/MAD as non-binding order/variability diagnostics; "
                "only the issue-specified pooled thresholds are performance gates"
            ),
            "warmup_excluded": True,
            "outlier_removal": False,
            "round_before_gate": False,
        },
        "thresholds": {
            "sharegpt_goodput_kairyu_over_vllm_min": GOODPUT_RATIO_FLOOR,
            "sharegpt_ttft_p99_kairyu_over_vllm_max": (SHAREGPT_P99_RATIO_CEILING),
            "shared_prefix_ttft_p50_kairyu_over_vllm_max": (SHARED_PREFIX_P50_RATIO_CEILING),
        },
        "open_loop_arrival_sweep": {
            "binding": False,
            "fresh_server_per_point": True,
            "requests_per_point": SHAREGPT_REQUESTS,
            "maximum_concurrency": SHAREGPT_CONCURRENCY,
            "rates_rps": list(OPEN_LOOP_RATES_RPS),
            "request": request_config("sharegpt"),
            "role": (
                "predeclared report-only saturation diagnostic; no rate may "
                "be selected after seeing results, and it cannot replace the "
                "issue-specified fixed-concurrency-128 burst"
            ),
        },
    }


def _token_row(
    *,
    sequence: int,
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    **extra: object,
) -> dict[str, object]:
    token_ids = list(prompt_token_ids)
    return {
        "sequence": sequence,
        "prompt_text": prompt_text,
        "prompt_text_sha256": sha256_text(prompt_text),
        "prompt_token_ids": token_ids,
        "prompt_tokens": len(token_ids),
        "prompt_token_ids_sha256": sha256_json(token_ids),
        **extra,
    }


def _first_human_text(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    turns = record.get("conversations")
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    turn = turns[0]
    if (
        isinstance(turn, dict)
        and turn.get("from") in {"human", "user"}
        and isinstance(turn.get("value"), str)
        and turn["value"].strip()
    ):
        return turn["value"]
    return None


def build_sharegpt_trace(
    dataset_path: Path,
    tokenizer: _Tokenizer,
    *,
    expected_dataset_sha256: str,
    tokenizer_sha256: str = PINNED_TOKENIZER_SHA256,
) -> dict[str, object]:
    """Tokenize and retain the fixed-seed ShareGPT selection losslessly."""

    actual_sha256 = sha256_file(dataset_path)
    if actual_sha256 != expected_dataset_sha256:
        raise ValueError(
            "ShareGPT dataset SHA-256 mismatch: "
            f"expected {expected_dataset_sha256}, got {actual_sha256}"
        )
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("ShareGPT dataset must contain a JSON list")
    candidates: list[tuple[int, object]] = []
    for source_index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("conversations"), list)
            or len(record["conversations"]) < 2
        ):
            continue
        candidates.append((source_index, record))
    random.Random(SHAREGPT_DATASET_SEED).shuffle(candidates)
    selected: list[tuple[int, str, tuple[int, ...]]] = []
    required = WARMUP_REQUESTS + SHAREGPT_REQUESTS
    for source_index, record in candidates:
        text = _first_human_text(record)
        if text is None:
            continue
        token_ids = tokenizer.encode(text)
        if not (SHAREGPT_MIN_PROMPT_TOKENS <= len(token_ids) <= SHAREGPT_MAX_PROMPT_TOKENS):
            continue
        selected.append((source_index, text, token_ids))
        if len(selected) == required:
            break
    if len(selected) < required:
        raise ValueError(
            f"ShareGPT dataset has only {len(selected)} selected usable prompts, need {required}"
        )

    def rows(
        values: Sequence[tuple[int, str, tuple[int, ...]]],
    ) -> list[dict[str, object]]:
        return [
            _token_row(
                sequence=sequence,
                prompt_text=text,
                prompt_token_ids=token_ids,
                source_index=source_index,
            )
            for sequence, (source_index, text, token_ids) in enumerate(values)
        ]

    value = {
        "schema_version": 1,
        "workload": "sharegpt",
        "tokenizer_sha256": tokenizer_sha256,
        "dataset": {
            "format": "ShareGPT conversations JSON",
            "repository": SHAREGPT_DATASET_REPO,
            "revision": SHAREGPT_DATASET_REVISION,
            "file": SHAREGPT_DATASET_FILE,
            "sha256": actual_sha256,
            "selection_seed": SHAREGPT_DATASET_SEED,
            "candidate_conversation_count": len(candidates),
            "minimum_prompt_tokens": SHAREGPT_MIN_PROMPT_TOKENS,
            "maximum_prompt_tokens": SHAREGPT_MAX_PROMPT_TOKENS,
        },
        # vLLM-compatible seed-0 selection: the first 128 usable shuffled
        # prompts are binding; warmups come from the next four and cannot
        # displace a binding prompt.
        "warmup": rows(selected[SHAREGPT_REQUESTS:]),
        "requests": rows(selected[:SHAREGPT_REQUESTS]),
    }
    return hashed_descriptor(value)


def _stable_token_pieces(
    tokenizer: _Tokenizer,
    *,
    required: int,
) -> tuple[tuple[int, str], ...]:
    """Find context-stable one-token pieces for exact synthetic geometry."""

    result: list[tuple[int, str]] = []
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
        result.append((token_id, text))
        if len(result) == required:
            return tuple(result)
    raise ValueError(f"tokenizer exposes {len(result)} stable token pieces; need {required}")


def build_shared_prefix_trace(
    tokenizer: _Tokenizer,
    *,
    tokenizer_sha256: str = PINNED_TOKENIZER_SHA256,
) -> dict[str, object]:
    """Build the arm-neutral 64-session x 8-turn fixed token-ID trace."""

    pieces = _stable_token_pieces(
        tokenizer,
        required=1 + NUM_SESSIONS + WARMUP_REQUESTS,
    )
    shared_token_id, shared_text = pieces[0]
    session_pieces = pieces[1 : 1 + NUM_SESSIONS]
    warmup_pieces = pieces[1 + NUM_SESSIONS :]
    warmup = [
        _token_row(
            sequence=sequence,
            prompt_text=text * WARMUP_PROMPT_TOKENS,
            prompt_token_ids=(token_id,) * WARMUP_PROMPT_TOKENS,
        )
        for sequence, (token_id, text) in enumerate(warmup_pieces)
    ]
    requests = []
    for row in fixed_workload():
        session_token_id, session_text = session_pieces[row.session]
        prompt = (shared_token_id,) * SYSTEM_PROMPT_TOKENS + (session_token_id,) * (
            (row.turn + 1) * TURN_TOKENS
        )
        prompt_text = shared_text * SYSTEM_PROMPT_TOKENS + session_text * (
            (row.turn + 1) * TURN_TOKENS
        )
        if tokenizer.encode(prompt_text) != prompt:
            raise ValueError(
                f"shared-prefix text lost arm-neutral token identity at sequence {row.sequence}"
            )
        requests.append(
            _token_row(
                sequence=row.sequence,
                prompt_text=prompt_text,
                prompt_token_ids=prompt,
                session=row.session,
                turn=row.turn,
                expected_cached_tokens=row.expected_cached_tokens,
            )
        )
    value = {
        "schema_version": 1,
        "workload": "shared-prefix",
        "construction": "arm-neutral-stable-token-repeat-v1",
        "tokenizer_sha256": tokenizer_sha256,
        "geometry": {
            "seed": WORKLOAD_SEED,
            "sessions": NUM_SESSIONS,
            "turns_per_session": TURNS_PER_SESSION,
            "shared_prefix_tokens": SYSTEM_PROMPT_TOKENS,
            "turn_tokens": TURN_TOKENS,
        },
        "shared_token_id": shared_token_id,
        "session_token_ids": [token_id for token_id, _ in session_pieces],
        "warmup": warmup,
        "requests": requests,
    }
    return hashed_descriptor(value)


def build_graph_warmup_trace(
    tokenizer: _Tokenizer,
    *,
    tokenizer_sha256: str = PINNED_TOKENIZER_SHA256,
) -> dict[str, object]:
    """Build 31 unique prompts for the configured graph-size request geometry."""

    pieces = _stable_token_pieces(
        tokenizer,
        required=1 + NUM_SESSIONS + WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS,
    )
    graph_pieces = pieces[1 + NUM_SESSIONS + WARMUP_REQUESTS :]
    prompts: list[dict[str, object]] = []
    for sequence, (token_id, text) in enumerate(graph_pieces):
        prompt_text = text * GRAPH_WARMUP_PROMPT_TOKENS
        prompt_token_ids = (token_id,) * GRAPH_WARMUP_PROMPT_TOKENS
        if tokenizer.encode(prompt_text) != prompt_token_ids:
            raise ValueError("graph-warmup text lost its pinned token geometry")
        prompts.append(
            _token_row(
                sequence=sequence,
                prompt_text=prompt_text,
                prompt_token_ids=prompt_token_ids,
            )
        )
    return hashed_descriptor(
        {
            "schema_version": 1,
            "construction": "arm-neutral-stable-token-repeat-v1",
            "tokenizer_sha256": tokenizer_sha256,
            "prompts": prompts,
            "batch_sizes": list(GRAPH_WARMUP_BATCH_SIZES),
            "request_count": GRAPH_WARMUP_REQUESTS,
            "completion_tokens": GRAPH_WARMUP_COMPLETION_TOKENS,
        }
    )


def write_trace_bundle(
    output: Path,
    *,
    dataset_path: Path,
    tokenizer_path: Path,
    dataset_sha256: str = SHAREGPT_DATASET_SHA256,
) -> dict[str, object]:
    """Tokenize the 673 MiB source once for reuse by all 52 fresh servers."""

    tokenizer_file = (
        tokenizer_path if tokenizer_path.is_file() else tokenizer_path / "tokenizer.json"
    )
    if sha256_file(tokenizer_file) != PINNED_TOKENIZER_SHA256:
        raise ValueError("tokenizer.json does not match pinned Qwen3-32B")
    tokenizer = HFTokenizer(tokenizer_path)
    traces = {
        "sharegpt": build_sharegpt_trace(
            dataset_path,
            tokenizer,
            expected_dataset_sha256=dataset_sha256,
        ),
        "shared-prefix": build_shared_prefix_trace(tokenizer),
    }
    graph_warmup = build_graph_warmup_trace(tokenizer)
    bundle = {
        "schema_version": TRACE_BUNDLE_SCHEMA,
        "benchmark": benchmark_config(dataset_sha256),
        "graph_warmup": graph_warmup,
        "traces": traces,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(bundle) + "\n", encoding="utf-8")
    return load_trace_bundle(output, dataset_sha256=dataset_sha256)


def load_trace_bundle(
    path: Path,
    *,
    dataset_sha256: str = SHAREGPT_DATASET_SHA256,
) -> dict[str, object]:
    bundle = _read_json_object(path)
    traces = bundle.get("traces")
    if not (
        bundle.get("schema_version") == TRACE_BUNDLE_SCHEMA
        and bundle.get("benchmark") == benchmark_config(dataset_sha256)
        and _graph_warmup_trace_valid(bundle.get("graph_warmup"))
        and _graph_warmup_disjoint(bundle.get("graph_warmup"), traces)
        and isinstance(traces, dict)
        and set(traces) == set(WORKLOADS)
        and all(
            _trace_valid(
                traces.get(workload),
                workload=workload,
                dataset_sha256=dataset_sha256,
            )
            for workload in WORKLOADS
        )
    ):
        raise ValueError("trace bundle does not satisfy the pinned G2 A6 schema")
    return bundle


def _graph_warmup_trace_valid(value: object) -> bool:
    if not _descriptor_valid(value):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    if not isinstance(raw, dict):
        return False
    prompts = raw.get("prompts")
    prompt_token_ids = (
        [
            tuple(row.get("prompt_token_ids", ()))
            for row in prompts
            if isinstance(row, dict)
        ]
        if isinstance(prompts, list)
        else []
    )
    return (
        raw.get("schema_version") == 1
        and raw.get("construction") == "arm-neutral-stable-token-repeat-v1"
        and raw.get("tokenizer_sha256") == PINNED_TOKENIZER_SHA256
        and raw.get("batch_sizes") == list(GRAPH_WARMUP_BATCH_SIZES)
        and raw.get("request_count") == GRAPH_WARMUP_REQUESTS
        and raw.get("completion_tokens") == GRAPH_WARMUP_COMPLETION_TOKENS
        and _token_rows_valid(
            prompts,
            expected_count=GRAPH_WARMUP_REQUESTS,
            require_source=False,
        )
        and len(prompt_token_ids) == GRAPH_WARMUP_REQUESTS
        and len(set(prompt_token_ids)) == GRAPH_WARMUP_REQUESTS
        and all(
            len(token_ids) == GRAPH_WARMUP_PROMPT_TOKENS
            and len(set(token_ids)) == 1
            for token_ids in prompt_token_ids
        )
    )


def _graph_warmup_disjoint(value: object, traces: object) -> bool:
    if not _graph_warmup_trace_valid(value) or not isinstance(traces, dict):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    assert isinstance(raw, dict)
    prompts = raw["prompts"]
    assert isinstance(prompts, list)
    graph_hashes = {
        row.get("prompt_token_ids_sha256")
        for row in prompts
        if isinstance(row, dict)
    }
    retained_hashes: set[object] = set()
    for trace in traces.values():
        if not isinstance(trace, dict) or not isinstance(trace.get("value"), dict):
            return False
        trace_raw = trace["value"]
        for key in ("warmup", "requests"):
            rows = trace_raw.get(key)
            if not isinstance(rows, list):
                return False
            retained_hashes.update(
                row.get("prompt_token_ids_sha256")
                for row in rows
                if isinstance(row, dict)
            )
    return len(graph_hashes) == GRAPH_WARMUP_REQUESTS and not (
        graph_hashes & retained_hashes
    )


def _token_rows_valid(
    rows: object,
    *,
    expected_count: int,
    require_source: bool,
) -> bool:
    if not isinstance(rows, list) or len(rows) != expected_count:
        return False
    source_indices: set[int] = set()
    for sequence, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("sequence") != sequence:
            return False
        token_ids = row.get("prompt_token_ids")
        prompt_text = row.get("prompt_text")
        if (
            not isinstance(prompt_text, str)
            or not prompt_text
            or row.get("prompt_text_sha256") != sha256_text(prompt_text)
            or not isinstance(token_ids, list)
            or not token_ids
            or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
            or row.get("prompt_tokens") != len(token_ids)
            or row.get("prompt_token_ids_sha256") != sha256_json(token_ids)
        ):
            return False
        if require_source:
            source_index = row.get("source_index")
            if type(source_index) is not int or source_index < 0 or source_index in source_indices:
                return False
            source_indices.add(source_index)
    return True


def _trace_valid(value: object, *, workload: str, dataset_sha256: str) -> bool:
    if not _descriptor_valid(value):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    if not isinstance(raw, dict):
        return False
    if (
        raw.get("schema_version") != 1
        or raw.get("workload") != workload
        or raw.get("tokenizer_sha256") != PINNED_TOKENIZER_SHA256
        or not _token_rows_valid(
            raw.get("warmup"),
            expected_count=WARMUP_REQUESTS,
            require_source=workload == "sharegpt",
        )
    ):
        return False
    if workload == "sharegpt":
        dataset = raw.get("dataset")
        requests = raw.get("requests")
        rows = (
            [*raw["warmup"], *requests]
            if isinstance(raw.get("warmup"), list) and isinstance(requests, list)
            else []
        )
        source_indices = [row.get("source_index") for row in rows if isinstance(row, dict)]
        return (
            isinstance(dataset, dict)
            and dataset.get("format") == "ShareGPT conversations JSON"
            and dataset.get("repository") == SHAREGPT_DATASET_REPO
            and dataset.get("revision") == SHAREGPT_DATASET_REVISION
            and dataset.get("file") == SHAREGPT_DATASET_FILE
            and dataset.get("sha256") == dataset_sha256 == SHAREGPT_DATASET_SHA256
            and dataset.get("selection_seed") == SHAREGPT_DATASET_SEED
            and type(dataset.get("candidate_conversation_count")) is int
            and int(dataset["candidate_conversation_count"]) >= SHAREGPT_REQUESTS + WARMUP_REQUESTS
            and dataset.get("maximum_prompt_tokens") == SHAREGPT_MAX_PROMPT_TOKENS
            and dataset.get("minimum_prompt_tokens") == SHAREGPT_MIN_PROMPT_TOKENS
            and _token_rows_valid(
                requests,
                expected_count=SHAREGPT_REQUESTS,
                require_source=True,
            )
            and all(
                SHAREGPT_MIN_PROMPT_TOKENS
                <= int(row["prompt_tokens"])
                <= SHAREGPT_MAX_PROMPT_TOKENS
                for row in rows
            )
            and len(source_indices) == WARMUP_REQUESTS + SHAREGPT_REQUESTS
            and len(set(source_indices)) == len(source_indices)
        )
    if workload != "shared-prefix":
        return False
    geometry = raw.get("geometry")
    requests = raw.get("requests")
    session_token_ids = raw.get("session_token_ids")
    if not (
        raw.get("construction") == "arm-neutral-stable-token-repeat-v1"
        and geometry
        == {
            "seed": WORKLOAD_SEED,
            "sessions": NUM_SESSIONS,
            "turns_per_session": TURNS_PER_SESSION,
            "shared_prefix_tokens": SYSTEM_PROMPT_TOKENS,
            "turn_tokens": TURN_TOKENS,
        }
        and type(raw.get("shared_token_id")) is int
        and isinstance(session_token_ids, list)
        and len(session_token_ids) == NUM_SESSIONS
        and len(set(session_token_ids)) == NUM_SESSIONS
        and _token_rows_valid(
            requests,
            expected_count=SHARED_PREFIX_REQUESTS,
            require_source=False,
        )
    ):
        return False
    expected_geometry = fixed_workload()
    assert isinstance(requests, list)
    shared = raw["shared_token_id"]
    for actual, expected in zip(requests, expected_geometry, strict=True):
        assert isinstance(actual, dict)
        ids = actual["prompt_token_ids"]
        if (
            actual.get("session") != expected.session
            or actual.get("turn") != expected.turn
            or actual.get("expected_cached_tokens") != expected.expected_cached_tokens
            or ids[:SYSTEM_PROMPT_TOKENS] != [shared] * SYSTEM_PROMPT_TOKENS
            or ids[SYSTEM_PROMPT_TOKENS:]
            != [session_token_ids[expected.session]] * ((expected.turn + 1) * TURN_TOKENS)
        ):
            return False
    return True


def _scenario_id(tp: int, workload: str, repeat: int, arm: str) -> str:
    return f"tp{tp}-{workload}-r{repeat}-{arm}"


def _sweep_scenario_id(tp: int, rate_rps: int, arm: str) -> str:
    return f"tp{tp}-sharegpt-open-loop-{rate_rps}rps-{arm}"


def _scenario_valid(tp: object, workload: object, repeat: object, arm: object) -> bool:
    return (
        type(tp) is int
        and tp in TP_DEGREES
        and workload in WORKLOADS
        and type(repeat) is int
        and 0 <= repeat < REPEATS
        and arm in ARMS
    )


def _sweep_scenario_valid(tp: object, rate_rps: object, arm: object) -> bool:
    return (
        type(tp) is int
        and tp in TP_DEGREES
        and type(rate_rps) is int
        and rate_rps in OPEN_LOOP_RATES_RPS
        and arm in ARMS
    )


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _image_id_valid(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _valid_sha256(value.removeprefix("sha256:"))
    )


def _container_id_valid(value: object) -> bool:
    return isinstance(value, str) and _valid_sha256(value)


def _mount_valid(
    value: object,
    *,
    destination: str,
    source_prefix: str,
    mount_type: str,
) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"type", "name", "source", "destination", "rw"}
        and value["type"] == mount_type
        and value["destination"] == destination
        and value["rw"] is False
        and isinstance(value["source"], str)
        and value["source"].startswith(source_prefix)
        and (
            isinstance(value["name"], str) and bool(value["name"])
            if mount_type == "volume"
            else value["name"] in {None, ""}
        )
    )


def _container_launch_valid(
    value: object,
    *,
    arm: str,
    tp: int,
    image_id: object,
    configuration_source: object,
) -> bool:
    if not _descriptor_valid(value) or not _image_id_valid(image_id):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    if not isinstance(raw, dict) or set(raw) != {
        "container_id",
        "image_id",
        "config_image",
        "entrypoint",
        "argv",
        "environment",
        "working_dir",
        "mounts",
        "host_config",
        "engine_import",
    }:
        return False
    expected = expected_container_launch(arm=arm, tp=tp)
    environment = raw["environment"]
    mounts = raw["mounts"]
    host_config = raw["host_config"]
    engine_import = raw["engine_import"]
    if not (
        _container_id_valid(raw["container_id"])
        and raw["image_id"] == image_id
        and raw["config_image"] == image_id
        and raw["entrypoint"] == expected["entrypoint"]
        and raw["argv"] == expected["argv"]
        and isinstance(environment, dict)
        and all(
            isinstance(key, str)
            and bool(key)
            and isinstance(item, str)
            for key, item in environment.items()
        )
        and all(
            environment.get(key) == item
            for key, item in expected["environment"].items()
        )
        and isinstance(raw["working_dir"], str)
        and raw["working_dir"].startswith("/")
        and isinstance(mounts, list)
        and isinstance(host_config, dict)
        and set(host_config)
        == {
            "ipc_mode",
            "memlock_soft",
            "memlock_hard",
            "published_container_port",
            "published_host_ip",
            "published_host_port",
            "gpu_device_ids",
        }
        and host_config["ipc_mode"] == "host"
        and host_config["memlock_soft"] == -1
        and host_config["memlock_hard"] == -1
        and host_config["published_container_port"] == "8000/tcp"
        and host_config["published_host_ip"] == "127.0.0.1"
        and type(host_config["published_host_port"]) is int
        and 1 <= int(host_config["published_host_port"]) <= 65535
        and host_config["gpu_device_ids"] == [str(index) for index in range(tp)]
        and isinstance(engine_import, dict)
    ):
        return False
    by_destination = {
        mount.get("destination"): mount
        for mount in mounts
        if isinstance(mount, dict)
    }
    expected_destinations = {"/models", "/workspace"}
    if arm == "kairyu":
        expected_destinations.update({"/app/kairyu", "/etc/kairyu/config.yaml"})
    if (
        len(by_destination) != len(mounts)
        or set(by_destination) != expected_destinations
        or not _mount_valid(
            by_destination.get("/models"),
            destination="/models",
            source_prefix="volume:",
            mount_type="volume",
        )
        or not _mount_valid(
            by_destination.get("/workspace"),
            destination="/workspace",
            source_prefix="repo:.",
            mount_type="bind",
        )
    ):
        return False
    if arm == "vllm-stock":
        return (
            configuration_source is None
            and set(engine_import) == {"distribution", "package_file", "package_sha256"}
            and engine_import["distribution"] == "vllm"
            and isinstance(engine_import["package_file"], str)
            and engine_import["package_file"].startswith("/")
            and "/vllm/" in engine_import["package_file"]
            and _valid_sha256(engine_import["package_sha256"])
        )
    if arm != "kairyu" or not _configuration_source_valid(
        configuration_source,
        tp=tp,
        arm=arm,
    ):
        return False
    assert isinstance(configuration_source, dict)
    config_value = configuration_source["value"]
    assert isinstance(config_value, dict)
    return (
        _mount_valid(
            by_destination.get("/app/kairyu"),
            destination="/app/kairyu",
            source_prefix="repo:kairyu",
            mount_type="bind",
        )
        and _mount_valid(
            by_destination.get("/etc/kairyu/config.yaml"),
            destination="/etc/kairyu/config.yaml",
            source_prefix=f"generated-config:{config_value['text_sha256']}",
            mount_type="bind",
        )
        and set(engine_import)
        == {
            "distribution",
            "package_file",
            "package_sha256",
            "engine_file",
            "engine_sha256",
        }
        and engine_import["distribution"] == "kairyu"
        and engine_import["package_file"] == "/app/kairyu/__init__.py"
        and engine_import["engine_file"] == "/app/kairyu/engine/engine_loop.py"
        and _valid_sha256(engine_import["package_sha256"])
        and _valid_sha256(engine_import["engine_sha256"])
    )


def _environment_artifact(
    descriptor: object,
) -> tuple[dict[str, object], Path] | None:
    if not (
        isinstance(descriptor, dict)
        and set(descriptor) == {"path", "sha256"}
        and isinstance(descriptor.get("path"), str)
        and _valid_sha256(descriptor.get("sha256"))
    ):
        return None
    relative = Path(descriptor["path"])
    results_root = (_REPO_ROOT / "bench" / "results").resolve()
    if (
        relative.is_absolute()
        or relative.as_posix() != descriptor["path"]
        or relative.parts[:2] != ("bench", "results")
        or ".." in relative.parts
        or not relative.name.startswith("env-")
        or relative.suffix != ".json"
    ):
        return None
    path = (_REPO_ROOT / relative).resolve()
    if not path.is_relative_to(results_root) or not path.is_file():
        return None
    if sha256_file(path) != descriptor["sha256"]:
        return None
    try:
        artifact = _read_json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    return artifact, path


def _inventory_pci_bus_ids(notes: str) -> tuple[str, ...]:
    in_inventory = False
    result: list[str] = []
    for line in notes.splitlines():
        if line.strip() == "inventory:":
            in_inventory = True
            continue
        if not in_inventory or not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3 and fields[0].isdigit() and fields[2]:
            result.append(fields[2])
    return tuple(result)


def _environment_value_valid(
    value: object,
    *,
    selected_pci_bus_ids: Sequence[str],
    gpu_product_name: str,
    driver_version: str,
) -> bool:
    if not (
        isinstance(value, dict)
        and set(value)
        == {
            "measurement_session_id",
            "measured_at_ns",
            "artifact_date",
            "artifact",
            "physical_topology_sha256",
            "fabric",
        }
        and isinstance(value.get("measurement_session_id"), str)
        and bool(value["measurement_session_id"])
        and type(value.get("measured_at_ns")) is int
        and int(value["measured_at_ns"]) > 0
        and isinstance(value.get("artifact_date"), str)
        and bool(value["artifact_date"])
        and _valid_sha256(value.get("physical_topology_sha256"))
        and isinstance(value.get("fabric"), dict)
    ):
        return False
    loaded = _environment_artifact(value.get("artifact"))
    if loaded is None:
        return False
    artifact, _ = loaded
    profile = artifact.get("profile")
    notes = artifact.get("notes")
    fabric = value["fabric"]
    if not (
        artifact.get("date") == value["artifact_date"]
        and artifact.get("measurement_session_id") == value["measurement_session_id"]
        and artifact.get("measured_at_ns") == value["measured_at_ns"]
        and isinstance(profile, dict)
        and isinstance(notes, str)
        and bool(notes)
        and profile.get("device_name") == gpu_product_name
        and artifact.get("driver") == driver_version
        and value["physical_topology_sha256"] == sha256_text(notes)
        and set(fabric)
        == {
            "kind",
            "raw_p2p_matrix_sha256",
            "minimum_peer_bandwidth_gbs",
        }
        and isinstance(profile.get("interconnect"), str)
        and bool(profile["interconnect"])
        and fabric.get("kind") == profile["interconnect"]
        and _valid_sha256(fabric.get("raw_p2p_matrix_sha256"))
        and type(profile.get("device_count")) is int
        and int(profile["device_count"]) >= max(TP_DEGREES)
        and type(profile.get("measured_bandwidth_gbs")) in {int, float}
        and not isinstance(profile.get("measured_bandwidth_gbs"), bool)
        and math.isfinite(float(profile["measured_bandwidth_gbs"]))
        and float(profile["measured_bandwidth_gbs"]) > 0
    ):
        return False
    matrix = profile.get("p2p_matrix")
    device_count = int(profile["device_count"])
    if not (
        isinstance(matrix, list)
        and len(matrix) == device_count
        and all(isinstance(row, list) and len(row) == device_count for row in matrix)
    ):
        return False
    peer_rates: list[float] = []
    for source, row in enumerate(matrix):
        for destination, rate in enumerate(row):
            if (
                type(rate) not in {int, float}
                or isinstance(rate, bool)
                or not math.isfinite(float(rate))
                or float(rate) <= 0
            ):
                return False
            if source != destination:
                peer_rates.append(float(rate))
    inventory = _inventory_pci_bus_ids(notes)
    return (
        len(inventory) == device_count
        and len(set(inventory)) == device_count
        and set(selected_pci_bus_ids).issubset(set(inventory))
        and bool(peer_rates)
        and fabric["raw_p2p_matrix_sha256"] == sha256_json(matrix)
        and type(fabric.get("minimum_peer_bandwidth_gbs")) in {int, float}
        and not isinstance(fabric.get("minimum_peer_bandwidth_gbs"), bool)
        and math.isfinite(float(fabric["minimum_peer_bandwidth_gbs"]))
        and float(fabric["minimum_peer_bandwidth_gbs"]) == min(peer_rates)
    )


def _checkpoint_attestation_valid(value: object) -> bool:
    if not _descriptor_valid(value):
        return False
    assert isinstance(value, dict)
    raw = value["value"]
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "model",
        "model_revision",
        "checkpoint_root",
        "checkpoint",
        "revision_files",
    }:
        return False
    revisions = raw["revision_files"]
    return (
        raw["schema_version"] == 1
        and raw["model"] == MODEL
        and raw["model_revision"] == MODEL_REVISION
        and raw["checkpoint_root"] == CHECKPOINT_ROOT
        and raw["checkpoint"] == _expected_checkpoint_identity()
        and isinstance(revisions, list)
        and bool(revisions)
        and all(
            isinstance(item, dict)
            and set(item) == {"file", "revision"}
            and isinstance(item["file"], str)
            and bool(item["file"])
            and item["revision"] == MODEL_REVISION
            for item in revisions
        )
    )


def _provenance_value_valid(
    value: object,
    *,
    tp: int,
    arm: str,
) -> bool:
    if not isinstance(value, dict):
        return False
    server_generation = value.get("server_generation")
    server_started_ns = value.get("server_started_ns")
    host = value.get("host")
    gpu = value.get("gpu")
    runtime = value.get("runtime")
    configuration = value.get("configuration")
    configuration_source = value.get("configuration_source")
    if not (
        set(value)
        == {
            "schema_version",
            "server_generation",
            "server_started_ns",
            "host",
            "gpu",
            "model",
            "checkpoint",
            "runtime",
            "configuration",
            "configuration_source",
        }
        and value.get("schema_version") == 1
        and isinstance(server_generation, str)
        and bool(server_generation)
        and type(server_started_ns) is int
        and server_started_ns > 0
        and value.get("model") == model_identity()
        and _checkpoint_attestation_valid(value.get("checkpoint"))
        and isinstance(host, dict)
        and isinstance(host.get("id"), str)
        and bool(host["id"])
        and isinstance(gpu, dict)
        and isinstance(runtime, dict)
        and isinstance(configuration, dict)
    ):
        return False
    uuids = gpu.get("uuids")
    pci_bus_ids = gpu.get("pci_bus_ids")
    container_uuids = gpu.get("container_visible_uuids")
    container_pci_bus_ids = gpu.get("container_visible_pci_bus_ids")
    if not (
        isinstance(uuids, list)
        and len(uuids) == tp
        and len(set(uuids)) == tp
        and all(isinstance(item, str) and item for item in uuids)
        and isinstance(pci_bus_ids, list)
        and len(pci_bus_ids) == tp
        and len(set(pci_bus_ids)) == tp
        and all(isinstance(item, str) and item for item in pci_bus_ids)
        and container_uuids == uuids
        and container_pci_bus_ids == pci_bus_ids
        and isinstance(gpu.get("product_name"), str)
        and bool(gpu["product_name"])
        and isinstance(gpu.get("driver_version"), str)
        and bool(gpu["driver_version"])
        and _environment_value_valid(
            host.get("environment"),
            selected_pci_bus_ids=pci_bus_ids,
            gpu_product_name=gpu["product_name"],
            driver_version=gpu["driver_version"],
        )
    ):
        return False
    cuda_graph = configuration.get("cuda_graph")
    package_versions = runtime.get("package_versions")
    container_launch = runtime.get("container_launch")
    resolved_backend = runtime.get("resolved_backend")
    if not (
        runtime.get("arm") == arm
        and isinstance(runtime.get("version"), str)
        and bool(runtime["version"])
        and _valid_git_commit(runtime.get("build_commit"))
        and isinstance(runtime.get("image_ref"), str)
        and bool(runtime["image_ref"])
        and _image_id_valid(runtime.get("image_id"))
        and isinstance(runtime.get("cuda_version"), str)
        and bool(runtime["cuda_version"])
        and isinstance(runtime.get("nccl_version"), str)
        and bool(runtime["nccl_version"])
        and isinstance(package_versions, dict)
        and all(
            isinstance(version, str) and bool(version)
            for version in package_versions.values()
        )
        and _descriptor_valid(container_launch)
        and _container_launch_valid(
            container_launch,
            arm=arm,
            tp=tp,
            image_id=runtime.get("image_id"),
            configuration_source=configuration_source,
        )
        and _descriptor_valid(resolved_backend)
        and isinstance(resolved_backend, dict)
        and resolved_backend.get("value")
        == expected_resolved_backend(arm=arm, tp=tp)
        and configuration.get("served_model_name") == MODEL
        and configuration.get("dtype") == MODEL_DTYPE
        and configuration.get("tensor_parallel_size") == tp
        and configuration.get("max_num_batched_tokens") == MAX_NUM_BATCHED_TOKENS
        and configuration.get("max_num_seqs") == MAX_NUM_SEQS
        and configuration.get("enable_prefix_caching") is True
        and configuration.get("cache_block_size") == CACHE_BLOCK_SIZE
        and configuration.get("usable_cache_blocks") == USABLE_CACHE_BLOCKS
        and type(configuration.get("allocated_cache_blocks")) is int
        and type(configuration.get("reserved_graph_scratch_blocks")) is int
        and type(configuration.get("reserved_null_blocks")) is int
        and configuration["allocated_cache_blocks"]
        - configuration["reserved_graph_scratch_blocks"]
        - configuration["reserved_null_blocks"]
        == configuration["usable_cache_blocks"]
        and configuration.get("cache_tokens") == CACHE_TOKENS
        and configuration.get("kv_cache_dtype") == KV_CACHE_DTYPE
        and configuration.get("allocated_kv_cache_bytes_per_rank")
        == ALLOCATED_KV_CACHE_BYTES_PER_RANK[tp]
        and configuration.get("reserved_kv_cache_bytes_per_rank")
        == RESERVED_KV_CACHE_BYTES_PER_RANK[tp]
        and configuration.get("usable_kv_cache_bytes_per_rank")
        == USABLE_KV_CACHE_BYTES_PER_RANK[tp]
        and configuration["allocated_kv_cache_bytes_per_rank"]
        - configuration["reserved_kv_cache_bytes_per_rank"]
        == configuration["usable_kv_cache_bytes_per_rank"]
        and configuration.get("chunked_prefill_enabled") is True
        and configuration.get("scheduler_policy") == "fcfs"
        and configuration.get("access_log_enabled") is ACCESS_LOG_ENABLED
        and type(configuration.get("max_model_len")) is int
        and int(configuration["max_model_len"]) == MAX_MODEL_LEN
        and isinstance(cuda_graph, dict)
        and cuda_graph.get("enabled") is True
        and isinstance(cuda_graph.get("mode"), str)
        and bool(cuda_graph["mode"])
        and cuda_graph.get("active_decode_graph_cap") == ACTIVE_DECODE_GRAPH_CAP
        and isinstance(cuda_graph.get("capture_sizes"), list)
        and ACTIVE_DECODE_GRAPH_CAP in cuda_graph["capture_sizes"]
        and all(type(size) is int and size > 0 for size in cuda_graph["capture_sizes"])
    ):
        return False
    if arm == "vllm-stock":
        return (
            runtime.get("version") == VLLM_VERSION
            and runtime.get("build_commit") == VLLM_BUILD_COMMIT
            and runtime.get("image_ref") == VLLM_IMAGE_REF
            and runtime.get("image_id") == VLLM_IMAGE_ID
            and runtime.get("v1_engine") is True
            and runtime.get("stock_image") is True
            and _vllm_startup_log_valid(runtime.get("startup_log"), tp=tp)
            and configuration_source is None
            and runtime.get("cuda_version") == VLLM_CUDA_VERSION
            and runtime.get("nccl_version") == VLLM_NCCL_VERSION
            and set(package_versions)
            == {
                "torch",
                "flashinfer",
                "vllm",
                "uvloop",
                "httptools",
                "uvicorn",
            }
            and package_versions.get("vllm") == VLLM_VERSION
            and package_versions.get("uvloop") == UVLOOP_VERSION
            and package_versions.get("httptools") == HTTPTOOLS_VERSION
            and package_versions.get("uvicorn") == VLLM_UVICORN_VERSION
            and configuration.get("allocated_cache_blocks")
            == VLLM_ALLOCATED_CACHE_BLOCKS
            and configuration.get("reserved_graph_scratch_blocks")
            == VLLM_GRAPH_SCRATCH_BLOCKS
            and configuration.get("reserved_null_blocks")
            == VLLM_NULL_BLOCKS
            and configuration.get("pipeline_depth") is None
            and configuration.get("async_scheduling") is VLLM_ASYNC_SCHEDULING
            and configuration.get("attention_backend") == VLLM_ATTENTION_BACKEND
            and configuration.get("distributed_executor_backend")
            == VLLM_DISTRIBUTED_EXECUTOR_BACKEND
            and configuration.get("custom_all_reduce") is VLLM_CUSTOM_ALL_REDUCE
            and configuration.get("compilation") == vllm_compilation_config()
            and cuda_graph.get("mode") == VLLM_CUDAGRAPH_MODE
            and cuda_graph.get("capture_sizes") == list(VLLM_CUDA_GRAPH_CAPTURE_SIZES)
        )
    return (
        arm == "kairyu"
        and runtime.get("source_clean") is True
        and runtime.get("source_status_policy")
        == {
            "format": "porcelain-v1-z-untracked-files-all",
            "only_allowed_exception": f"?? {SOURCE_CLEAN_EXCEPTION_PREFIX}*",
        }
        and runtime.get("stock_image") is False
        and _configuration_source_valid(
            configuration_source,
            tp=tp,
            arm=arm,
        )
        and runtime.get("cuda_version") == KAIRYU_CUDA_VERSION
        and runtime.get("nccl_version") == KAIRYU_NCCL_VERSION
        and set(package_versions)
        == {
            "torch",
            "flashinfer",
            "kairyu",
            "uvloop",
            "httptools",
            "uvicorn",
        }
        and package_versions.get("kairyu") == runtime.get("version")
        and package_versions.get("uvloop") == UVLOOP_VERSION
        and package_versions.get("httptools") == HTTPTOOLS_VERSION
        and package_versions.get("uvicorn") == KAIRYU_UVICORN_VERSION
        and configuration.get("allocated_cache_blocks")
        == KAIRYU_ALLOCATED_CACHE_BLOCKS
        and configuration.get("reserved_graph_scratch_blocks")
        == KAIRYU_GRAPH_SCRATCH_BLOCKS
        and configuration.get("reserved_null_blocks")
        == KAIRYU_NULL_BLOCKS
        and configuration.get("pipeline_depth") == KAIRYU_PIPELINE_DEPTH
        and configuration.get("async_scheduling") is False
        and configuration.get("attention_backend") == "flashinfer"
        and configuration.get("distributed_executor_backend")
        == "torch-distributed-nccl"
        and configuration.get("custom_all_reduce") is False
        and configuration.get("compilation") is None
        and cuda_graph.get("mode") == "decode-only"
        and cuda_graph.get("capture_sizes") == list(KAIRYU_CUDA_GRAPH_CAPTURE_SIZES)
    )


def _provenance_valid(value: object, *, tp: int, arm: str) -> bool:
    return (
        _descriptor_valid(value)
        and isinstance(value, dict)
        and _provenance_value_valid(value["value"], tp=tp, arm=arm)
    )


def load_provenance(path: Path, *, tp: int, arm: str) -> dict[str, object]:
    result = hashed_descriptor(_read_json_object(path))
    if not _provenance_valid(result, tp=tp, arm=arm):
        raise ValueError(f"{path} does not satisfy the G2 A6 provenance schema")
    return result


def _planned_requests(
    trace: Mapping[str, object],
    *,
    phase: str,
    workload: str,
) -> list[PlannedRequest]:
    raw = trace["value"]
    assert isinstance(raw, dict)
    rows = raw["warmup" if phase == "warmup" else "requests"]
    assert isinstance(rows, list)
    completion_tokens = (
        WARMUP_MAX_TOKENS if phase == "warmup" else int(request_config(workload)["max_tokens"])
    )
    return [
        PlannedRequest(
            phase=phase,
            sequence=int(row["sequence"]),
            prompt_text=str(row["prompt_text"]),
            prompt_token_ids=tuple(row["prompt_token_ids"]),
            expected_completion_tokens=completion_tokens,
        )
        for row in rows
        if isinstance(row, dict)
    ]


def _graph_warmup_requests(
    graph_warmup: Mapping[str, object],
) -> list[PlannedRequest]:
    if not _graph_warmup_trace_valid(graph_warmup):
        raise ValueError("graph warmup trace is invalid")
    raw = graph_warmup["value"]
    assert isinstance(raw, dict)
    prompts = raw["prompts"]
    assert isinstance(prompts, list)
    result: list[PlannedRequest] = []
    sequence = 0
    for batch_size in GRAPH_WARMUP_BATCH_SIZES:
        for batch_sequence in range(batch_size):
            prompt = prompts[sequence]
            assert isinstance(prompt, dict)
            result.append(
                PlannedRequest(
                    phase="graph_warmup",
                    sequence=sequence,
                    prompt_text=str(prompt["prompt_text"]),
                    prompt_token_ids=tuple(prompt["prompt_token_ids"]),
                    expected_completion_tokens=GRAPH_WARMUP_COMPLETION_TOKENS,
                    graph_batch_size=batch_size,
                    graph_batch_sequence=batch_sequence,
                )
            )
            sequence += 1
    return result


def _usage_from_payload(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    total_tokens = value.get("total_tokens")
    if (
        type(prompt_tokens) is not int
        or prompt_tokens < 0
        or type(completion_tokens) is not int
        or completion_tokens < 0
        or type(total_tokens) is not int
        or total_tokens != prompt_tokens + completion_tokens
    ):
        return None
    details = value.get("prompt_tokens_details")
    cached_tokens = 0
    if details is not None:
        if not isinstance(details, dict):
            return None
        candidate = details.get("cached_tokens", 0)
        if type(candidate) is not int or not 0 <= candidate <= prompt_tokens:
            return None
        cached_tokens = candidate
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


async def stream_completion_once(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    planned: PlannedRequest,
    workload: str,
) -> dict[str, object]:
    """Execute one strict streaming request exactly once, with no retry path."""

    config = {
        **request_config(workload),
        "max_tokens": planned.expected_completion_tokens,
        "min_tokens": planned.expected_completion_tokens,
    }
    body = {
        "model": MODEL,
        # Text is intentional: the G2 TTFT definition includes server-side
        # tokenization. The pinned token IDs remain the usage/identity oracle.
        "prompt": planned.prompt_text,
        **{key: value for key, value in config.items() if key != "endpoint"},
    }
    start_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    end_ns = start_ns
    http_status: int | None = None
    content_type: str | None = None
    response_request_header: str | None = None
    response_ids: set[str] = set()
    output_parts: list[str] = []
    usage: dict[str, int] | None = None
    usage_events = 0
    saw_usage = False
    done_events = 0
    data_events: list[str] = []
    terminal_seen = False
    terminal_choice_events = 0
    finish_reasons: list[str] = []
    error_type: str | None = None
    try:
        async with client.stream(
            "POST",
            f"{endpoint}/v1/completions",
            json=body,
        ) as response:
            http_status = response.status_code
            content_type = response.headers.get("content-type")
            response_request_header = response.headers.get("x-request-id")
            if response.status_code != 200:
                payload = await response.aread()
                data_events.append(payload.decode(errors="replace"))
                error_type = "HTTPError"
            elif not (content_type or "").lower().startswith("text/event-stream"):
                payload = await response.aread()
                data_events.append(payload.decode(errors="replace"))
                error_type = "UnexpectedContentType"
            else:
                saw_done = False
                async for line in response.aiter_lines():
                    end_ns = time.perf_counter_ns()
                    if line == "":
                        continue
                    if not line.startswith("data: "):
                        error_type = error_type or "InvalidSSEFraming"
                        continue
                    data = line.removeprefix("data: ")
                    data_events.append(data)
                    if saw_done:
                        error_type = error_type or "DataAfterDONE"
                        continue
                    if data == "[DONE]":
                        done_events += 1
                        saw_done = True
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        error_type = error_type or "InvalidSSEJSON"
                        continue
                    if not isinstance(payload, dict):
                        error_type = error_type or "InvalidSSEPayload"
                        continue
                    if payload.get("error") is not None:
                        error_type = error_type or "SSEError"
                    response_id = payload.get("id")
                    if isinstance(response_id, str) and response_id:
                        response_ids.add(response_id)
                    else:
                        error_type = error_type or "MissingResponseID"
                    choices = payload.get("choices")
                    if not isinstance(choices, list):
                        error_type = error_type or "MissingChoices"
                        continue
                    if len(choices) > 1:
                        error_type = error_type or "UnexpectedChoiceCount"
                    for choice in choices:
                        if saw_usage:
                            error_type = error_type or "ChoicesAfterUsage"
                        if terminal_seen:
                            error_type = error_type or "ChoicesAfterTerminal"
                        if not isinstance(choice, dict):
                            error_type = error_type or "InvalidChoice"
                            continue
                        if choice.get("index") != 0:
                            error_type = error_type or "UnexpectedChoiceIndex"
                        text = choice.get("text")
                        if not isinstance(text, str):
                            error_type = error_type or "InvalidChoiceText"
                            continue
                        # Both stock vLLM and the Kairyu compatibility endpoint
                        # suppress empty chunked-prefill events.  The first
                        # remaining choice is therefore the first generated
                        # token event even when that token decodes to "" (and
                        # even when max_tokens=1 makes the same event terminal).
                        if first_token_ns is None:
                            first_token_ns = end_ns
                        if text:
                            output_parts.append(text)
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            terminal_choice_events += 1
                            if not isinstance(finish_reason, str):
                                error_type = error_type or "InvalidFinishReason"
                            else:
                                finish_reasons.append(finish_reason)
                                if finish_reason != "length":
                                    error_type = error_type or "UnexpectedFinishReason"
                            terminal_seen = True
                    candidate_usage = payload.get("usage")
                    if candidate_usage is not None:
                        usage_events += 1
                        if choices:
                            error_type = error_type or "UsageChoicesNotEmpty"
                        if not terminal_seen:
                            error_type = error_type or "UsageBeforeTerminal"
                        saw_usage = True
                        normalized = _usage_from_payload(candidate_usage)
                        if normalized is None:
                            error_type = error_type or "InvalidUsage"
                        elif usage is not None:
                            error_type = error_type or "DuplicateUsage"
                        else:
                            usage = normalized
                if not saw_done:
                    error_type = error_type or "MissingDONE"
    except Exception as error:
        error_type = error_type or type(error).__name__
    end_ns = max(end_ns, time.perf_counter_ns())
    if len(response_ids) != 1:
        error_type = error_type or "ResponseIDMismatch"
    if done_events != 1:
        error_type = error_type or "DONECountMismatch"
    if not terminal_seen:
        error_type = error_type or "MissingTerminalChoice"
    if terminal_choice_events != 1:
        error_type = error_type or "TerminalChoiceCountMismatch"
    if first_token_ns is None:
        error_type = error_type or "MissingFirstToken"
    if usage_events != 1 or usage is None:
        error_type = error_type or "MissingExactUsage"
    elif (
        usage["prompt_tokens"] != len(planned.prompt_token_ids)
        or usage["completion_tokens"] != planned.expected_completion_tokens
    ):
        error_type = error_type or "UsageCountMismatch"
    success = error_type is None
    output_text = "".join(output_parts)
    return {
        "type": "request",
        "phase": planned.phase,
        "sequence": planned.sequence,
        "attempts": 1,
        "prompt_text_sha256": planned.prompt_text_sha256,
        "prompt_tokens": len(planned.prompt_token_ids),
        "prompt_token_ids_sha256": planned.prompt_token_ids_sha256,
        "expected_completion_tokens": planned.expected_completion_tokens,
        "graph_batch_size": planned.graph_batch_size,
        "graph_batch_sequence": planned.graph_batch_sequence,
        "http_status": http_status,
        "content_type": content_type,
        "response_request_header": response_request_header,
        "response_id": next(iter(response_ids)) if len(response_ids) == 1 else None,
        "status": "success" if success else "failure",
        "error_type": error_type,
        "done": done_events == 1,
        "done_events": done_events,
        "terminal_choice": terminal_seen,
        "terminal_choice_events": terminal_choice_events,
        "finish_reasons": finish_reasons,
        "usage_events": usage_events,
        "usage": usage,
        "timing_ns": {
            "start": start_ns,
            "first_token": first_token_ns,
            "end": end_ns,
            "ttft": first_token_ns - start_ns if first_token_ns is not None else None,
            "total": end_ns - start_ns,
        },
        "output_text_sha256": sha256_text(output_text),
        "stream_payload_sha256": sha256_text("\n".join(data_events)),
    }


async def _execute_serial(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    planned: Sequence[PlannedRequest],
    workload: str,
) -> list[dict[str, object]]:
    rows = []
    for request in planned:
        rows.append(
            await stream_completion_once(
                client,
                endpoint=endpoint,
                planned=request,
                workload=workload,
            )
        )
    return rows


async def _execute_burst(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    planned: Sequence[PlannedRequest],
    workload: str,
) -> list[dict[str, object]]:
    if len(planned) != SHAREGPT_CONCURRENCY:
        raise ValueError("binding ShareGPT burst requires exactly 128 simultaneous requests")
    release = asyncio.Event()

    async def one(request: PlannedRequest) -> dict[str, object]:
        await release.wait()
        return await stream_completion_once(
            client,
            endpoint=endpoint,
            planned=request,
            workload=workload,
        )

    tasks = [asyncio.create_task(one(request)) for request in planned]
    await asyncio.sleep(0)
    release_ns = time.perf_counter_ns()
    release.set()
    return [{**row, "burst_release_ns": release_ns} for row in await asyncio.gather(*tasks)]


async def _execute_graph_warmups(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    planned: Sequence[PlannedRequest],
    workload: str,
) -> list[dict[str, object]]:
    """Replay B=1/2/4/8/16 in order, synchronizing every batch release."""

    if len(planned) != GRAPH_WARMUP_REQUESTS:
        raise ValueError("graph warmup requires the exact 31-request plan")
    result: list[dict[str, object]] = []
    offset = 0
    for batch_size in GRAPH_WARMUP_BATCH_SIZES:
        batch = planned[offset : offset + batch_size]
        if (
            len(batch) != batch_size
            or any(item.graph_batch_size != batch_size for item in batch)
            or [item.graph_batch_sequence for item in batch] != list(range(batch_size))
        ):
            raise ValueError("graph warmup plan does not match the pinned batch geometry")
        release = asyncio.Event()

        async def one(
            request: PlannedRequest,
            release_event: asyncio.Event = release,
        ) -> dict[str, object]:
            await release_event.wait()
            return await stream_completion_once(
                client,
                endpoint=endpoint,
                planned=request,
                workload=workload,
            )

        tasks = [asyncio.create_task(one(request)) for request in batch]
        await asyncio.sleep(0)
        release_ns = time.perf_counter_ns()
        release.set()
        rows = await asyncio.gather(*tasks)
        result.extend({**row, "graph_warmup_release_ns": release_ns} for row in rows)
        offset += batch_size
    if offset != len(planned):
        raise ValueError("graph warmup plan has trailing requests")
    return result


async def _execute_open_loop(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    planned: Sequence[PlannedRequest],
    rate_rps: int,
) -> list[dict[str, object]]:
    """Release the fixed 128 prompts at one predeclared constant rate."""

    if rate_rps not in OPEN_LOOP_RATES_RPS:
        raise ValueError(f"rate must be one of {OPEN_LOOP_RATES_RPS}")
    if len(planned) != SHAREGPT_REQUESTS:
        raise ValueError("open-loop point requires the fixed 128 ShareGPT prompts")
    period_ns = 1_000_000_000 // rate_rps
    anchor_ns = time.perf_counter_ns() + 10_000_000

    async def one(request: PlannedRequest) -> dict[str, object]:
        scheduled_ns = anchor_ns + request.sequence * period_ns
        delay_s = (scheduled_ns - time.perf_counter_ns()) / 1_000_000_000
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        row = await stream_completion_once(
            client,
            endpoint=endpoint,
            planned=request,
            workload="sharegpt",
        )
        timing = row.get("timing_ns")
        actual_start = timing.get("start") if isinstance(timing, dict) else None
        return {
            **row,
            "type": "sweep_request",
            "arrival_rate_rps": rate_rps,
            "scheduled_offset_ns": request.sequence * period_ns,
            "scheduled_start_ns": scheduled_ns,
            "release_lag_ns": (actual_start - scheduled_ns if type(actual_start) is int else None),
        }

    tasks = [asyncio.create_task(one(request)) for request in planned]
    return list(await asyncio.gather(*tasks))


def _normalize_endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    return endpoint


async def measure(
    *,
    tensor_parallel_size: int,
    arm: str,
    workload: str,
    repeat: int,
    endpoint: str,
    tokenizer_path: Path,
    dataset_sha256: str,
    provenance_path: Path,
    dataset_path: Path | None = None,
    trace_bundle_path: Path | None = None,
    api_key: str | None = None,
    request_timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[dict[str, object]]:
    """Measure one fresh-server scenario and retain every planned request."""

    if not _scenario_valid(tensor_parallel_size, workload, repeat, arm):
        raise ValueError("invalid G2 A6 scenario")
    if request_timeout_s <= 0:
        raise ValueError("request timeout must be positive")
    endpoint = _normalize_endpoint(endpoint)
    tokenizer_file = (
        tokenizer_path if tokenizer_path.is_file() else tokenizer_path / "tokenizer.json"
    )
    if sha256_file(tokenizer_file) != PINNED_TOKENIZER_SHA256:
        raise ValueError("tokenizer.json does not match pinned Qwen3-32B")
    if trace_bundle_path is not None:
        bundle = load_trace_bundle(
            trace_bundle_path,
            dataset_sha256=dataset_sha256,
        )
        traces = bundle["traces"]
        assert isinstance(traces, dict)
        trace = traces[workload]
        graph_warmup = bundle["graph_warmup"]
        assert isinstance(graph_warmup, dict)
    else:
        tokenizer = HFTokenizer(tokenizer_path)
        graph_warmup = build_graph_warmup_trace(tokenizer)
        if workload == "sharegpt":
            if dataset_path is None:
                raise ValueError("ShareGPT measurement requires --dataset or --trace-bundle")
            trace = build_sharegpt_trace(
                dataset_path,
                tokenizer,
                expected_dataset_sha256=dataset_sha256,
            )
        else:
            trace = build_shared_prefix_trace(tokenizer)
    if not _trace_valid(
        trace,
        workload=workload,
        dataset_sha256=dataset_sha256,
    ) or not _graph_warmup_disjoint(graph_warmup, {workload: trace}):
        raise RuntimeError("constructed workload trace is invalid")

    scenario = (
        tensor_parallel_size,
        workload,
        repeat,
        arm,
    )
    scenario_id = _scenario_id(*scenario)
    provenance_start = load_provenance(
        provenance_path,
        tp=tensor_parallel_size,
        arm=arm,
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(
        max_connections=SHAREGPT_CONCURRENCY,
        max_keepalive_connections=SHAREGPT_CONCURRENCY,
    )
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "scenario_id": scenario_id,
            "tensor_parallel_size": tensor_parallel_size,
            "arm": arm,
            "workload": workload,
            "repeat": repeat,
            "endpoint": endpoint,
            "benchmark": benchmark_config(dataset_sha256),
            "trace": trace,
            "graph_warmup": graph_warmup,
            "provenance_start": provenance_start,
        }
    ]
    async with httpx.AsyncClient(
        timeout=request_timeout_s,
        headers=headers,
        limits=limits,
    ) as client:
        models = await client.get(f"{endpoint}/v1/models")
        if models.status_code != 200:
            raise RuntimeError(f"{scenario_id} /v1/models returned HTTP {models.status_code}")
        payload = models.json()
        available = (
            {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
            if isinstance(payload, dict)
            else set()
        )
        if MODEL not in available:
            raise ValueError(f"{scenario_id} endpoint does not serve {MODEL}")
        warmup_rows = await _execute_serial(
            client,
            endpoint=endpoint,
            planned=_planned_requests(
                trace,
                phase="warmup",
                workload=workload,
            ),
            workload=workload,
        )
        graph_warmup_rows = await _execute_graph_warmups(
            client,
            endpoint=endpoint,
            planned=_graph_warmup_requests(graph_warmup),
            workload=workload,
        )
        planned = _planned_requests(
            trace,
            phase="measurement",
            workload=workload,
        )
        if workload == "sharegpt":
            measurement_rows = await _execute_burst(
                client,
                endpoint=endpoint,
                planned=planned,
                workload=workload,
            )
        else:
            measurement_rows = await _execute_serial(
                client,
                endpoint=endpoint,
                planned=planned,
                workload=workload,
            )
        rows.extend(
            {
                **row,
                "scenario_id": scenario_id,
                "tensor_parallel_size": tensor_parallel_size,
                "arm": arm,
                "workload": workload,
                "repeat": repeat,
            }
            for row in (*warmup_rows, *graph_warmup_rows, *measurement_rows)
        )
    provenance_end = load_provenance(
        provenance_path,
        tp=tensor_parallel_size,
        arm=arm,
    )
    formal_rows = [row for row in rows if row.get("phase") == "measurement"]
    timing_rows = [
        row["timing_ns"] for row in formal_rows if isinstance(row.get("timing_ns"), dict)
    ]
    rows.append(
        {
            "type": "run_end",
            "scenario_id": scenario_id,
            "tensor_parallel_size": tensor_parallel_size,
            "arm": arm,
            "workload": workload,
            "repeat": repeat,
            "warmup_request_count": len(warmup_rows) + len(graph_warmup_rows),
            "serial_warmup_request_count": len(warmup_rows),
            "graph_warmup_request_count": len(graph_warmup_rows),
            "measurement_request_count": len(measurement_rows),
            "measurement_window_ns": {
                "start": min(int(item["start"]) for item in timing_rows),
                "end": max(int(item["end"]) for item in timing_rows),
            },
            "provenance_end": provenance_end,
        }
    )
    return rows


async def measure_sweep(
    *,
    tensor_parallel_size: int,
    arm: str,
    rate_rps: int,
    endpoint: str,
    tokenizer_path: Path,
    dataset_path: Path | None,
    dataset_sha256: str,
    provenance_path: Path,
    trace_bundle_path: Path | None = None,
    api_key: str | None = None,
    request_timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[dict[str, object]]:
    """Measure one fresh-server, report-only open-loop ShareGPT rate."""

    if not _sweep_scenario_valid(tensor_parallel_size, rate_rps, arm):
        raise ValueError("invalid G2 A6 open-loop scenario")
    if request_timeout_s <= 0:
        raise ValueError("request timeout must be positive")
    endpoint = _normalize_endpoint(endpoint)
    tokenizer_file = (
        tokenizer_path if tokenizer_path.is_file() else tokenizer_path / "tokenizer.json"
    )
    if sha256_file(tokenizer_file) != PINNED_TOKENIZER_SHA256:
        raise ValueError("tokenizer.json does not match pinned Qwen3-32B")
    if trace_bundle_path is not None:
        bundle = load_trace_bundle(
            trace_bundle_path,
            dataset_sha256=dataset_sha256,
        )
        traces = bundle["traces"]
        assert isinstance(traces, dict)
        trace = traces["sharegpt"]
        graph_warmup = bundle["graph_warmup"]
        assert isinstance(graph_warmup, dict)
    else:
        if dataset_path is None:
            raise ValueError("open-loop measurement requires --dataset or --trace-bundle")
        tokenizer = HFTokenizer(tokenizer_path)
        graph_warmup = build_graph_warmup_trace(tokenizer)
        trace = build_sharegpt_trace(
            dataset_path,
            tokenizer,
            expected_dataset_sha256=dataset_sha256,
        )
    if not _trace_valid(
        trace,
        workload="sharegpt",
        dataset_sha256=dataset_sha256,
    ) or not _graph_warmup_disjoint(graph_warmup, {"sharegpt": trace}):
        raise RuntimeError("constructed ShareGPT trace is invalid")
    scenario_id = _sweep_scenario_id(tensor_parallel_size, rate_rps, arm)
    provenance_start = load_provenance(
        provenance_path,
        tp=tensor_parallel_size,
        arm=arm,
    )
    rows: list[dict[str, object]] = [
        {
            "type": "sweep_run",
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "scenario_id": scenario_id,
            "tensor_parallel_size": tensor_parallel_size,
            "arm": arm,
            "arrival_rate_rps": rate_rps,
            "endpoint": endpoint,
            "benchmark": benchmark_config(dataset_sha256),
            "trace": trace,
            "graph_warmup": graph_warmup,
            "provenance_start": provenance_start,
        }
    ]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(
        max_connections=SHAREGPT_CONCURRENCY,
        max_keepalive_connections=SHAREGPT_CONCURRENCY,
    )
    async with httpx.AsyncClient(
        timeout=request_timeout_s,
        headers=headers,
        limits=limits,
    ) as client:
        models = await client.get(f"{endpoint}/v1/models")
        if models.status_code != 200:
            raise RuntimeError(f"{scenario_id} /v1/models returned HTTP {models.status_code}")
        payload = models.json()
        available = (
            {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
            if isinstance(payload, dict)
            else set()
        )
        if MODEL not in available:
            raise ValueError(f"{scenario_id} endpoint does not serve {MODEL}")
        warmups = await _execute_serial(
            client,
            endpoint=endpoint,
            planned=_planned_requests(
                trace,
                phase="warmup",
                workload="sharegpt",
            ),
            workload="sharegpt",
        )
        graph_warmups = await _execute_graph_warmups(
            client,
            endpoint=endpoint,
            planned=_graph_warmup_requests(graph_warmup),
            workload="sharegpt",
        )
        planned = _planned_requests(
            trace,
            phase="sweep",
            workload="sharegpt",
        )
        sweep = await _execute_open_loop(
            client,
            endpoint=endpoint,
            planned=planned,
            rate_rps=rate_rps,
        )
        rows.extend(
            {
                **row,
                "type": "sweep_request",
                "scenario_id": scenario_id,
                "tensor_parallel_size": tensor_parallel_size,
                "arm": arm,
                "arrival_rate_rps": rate_rps,
            }
            for row in (*warmups, *graph_warmups, *sweep)
        )
    provenance_end = load_provenance(
        provenance_path,
        tp=tensor_parallel_size,
        arm=arm,
    )
    formal = [row for row in rows if row.get("phase") == "sweep"]
    timing = [row["timing_ns"] for row in formal if isinstance(row.get("timing_ns"), dict)]
    rows.append(
        {
            "type": "sweep_run_end",
            "scenario_id": scenario_id,
            "tensor_parallel_size": tensor_parallel_size,
            "arm": arm,
            "arrival_rate_rps": rate_rps,
            "warmup_request_count": len(warmups) + len(graph_warmups),
            "serial_warmup_request_count": len(warmups),
            "graph_warmup_request_count": len(graph_warmups),
            "sweep_request_count": len(sweep),
            "measurement_window_ns": {
                "start": min(int(item["start"]) for item in timing),
                "end": max(int(item["end"]) for item in timing),
            },
            "arrival_anchor_ns": min(int(row["scheduled_start_ns"]) for row in formal),
            "provenance_end": provenance_end,
        }
    )
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{**row, "row_index": index} for index, row in enumerate(rows)]
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in normalized),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"invalid JSONL framing at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise ValueError(f"non-contiguous row indices in {path}")
    return rows


def _parse_measurement_shard(
    rows: Sequence[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if len(rows) < 2:
        raise ValueError("measurement shard is empty")
    start, end = rows[0], rows[-1]
    if start.get("type") != "run" or end.get("type") != "run_end":
        raise ValueError("measurement shard must be framed by run/run_end")
    if start.get("schema_version") != SCHEMA_VERSION or start.get("gate") != GATE:
        raise ValueError("measurement shard schema/gate mismatch")
    identity = (
        start.get("tensor_parallel_size"),
        start.get("workload"),
        start.get("repeat"),
        start.get("arm"),
    )
    if not _scenario_valid(*identity):
        raise ValueError("measurement shard scenario is invalid")
    scenario_id = _scenario_id(*identity)
    if start.get("scenario_id") != scenario_id or end.get("scenario_id") != scenario_id:
        raise ValueError("measurement shard scenario identity is inconsistent")
    requests = list(rows[1:-1])
    if any(row.get("type") != "request" for row in requests):
        raise ValueError("measurement shard contains an unknown row type")
    if any(row.get("scenario_id") != scenario_id for row in requests):
        raise ValueError("measurement request belongs to another scenario")
    return start, requests, end


def write_measurement_shard(
    path: Path,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    _write_jsonl(path, rows)
    stored = _read_jsonl(path)
    start, requests, end = _parse_measurement_shard(stored)
    return {
        "scenario_id": start["scenario_id"],
        "warmup_requests": sum(row.get("phase") == "warmup" for row in requests),
        "graph_warmup_requests": sum(
            row.get("phase") == "graph_warmup" for row in requests
        ),
        "measurement_requests": sum(row.get("phase") == "measurement" for row in requests),
        "successful_requests": sum(row.get("status") == "success" for row in requests),
        "provenance_stable": (start.get("provenance_start") == end.get("provenance_end")),
        "sha256": sha256_file(path),
    }


def _parse_sweep_shard(
    rows: Sequence[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if len(rows) < 2:
        raise ValueError("open-loop sweep shard is empty")
    start, end = rows[0], rows[-1]
    if start.get("type") != "sweep_run" or end.get("type") != "sweep_run_end":
        raise ValueError("open-loop shard must be framed by sweep_run/sweep_run_end")
    if start.get("schema_version") != SCHEMA_VERSION or start.get("gate") != GATE:
        raise ValueError("open-loop shard schema/gate mismatch")
    identity = (
        start.get("tensor_parallel_size"),
        start.get("arrival_rate_rps"),
        start.get("arm"),
    )
    if not _sweep_scenario_valid(*identity):
        raise ValueError("open-loop shard scenario is invalid")
    scenario_id = _sweep_scenario_id(*identity)
    if start.get("scenario_id") != scenario_id or end.get("scenario_id") != scenario_id:
        raise ValueError("open-loop shard scenario identity is inconsistent")
    requests = list(rows[1:-1])
    if any(row.get("type") != "sweep_request" for row in requests):
        raise ValueError("open-loop shard contains an unknown row type")
    if any(row.get("scenario_id") != scenario_id for row in requests):
        raise ValueError("open-loop request belongs to another scenario")
    return start, requests, end


def write_sweep_shard(
    path: Path,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    _write_jsonl(path, rows)
    stored = _read_jsonl(path)
    start, requests, end = _parse_sweep_shard(stored)
    return {
        "scenario_id": start["scenario_id"],
        "warmup_requests": sum(row.get("phase") == "warmup" for row in requests),
        "graph_warmup_requests": sum(
            row.get("phase") == "graph_warmup" for row in requests
        ),
        "sweep_requests": sum(row.get("phase") == "sweep" for row in requests),
        "successful_requests": sum(row.get("status") == "success" for row in requests),
        "provenance_stable": (start.get("provenance_start") == end.get("provenance_end")),
        "sha256": sha256_file(path),
    }


def assemble_rows(
    paths: Sequence[Path],
    sweep_paths: Sequence[Path] = (),
) -> list[dict[str, object]]:
    """Combine scenario shards into one canonical independently replayable raw."""

    shards: dict[
        tuple[int, str, int, str],
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
        scenario = (
            int(start["tensor_parallel_size"]),
            str(start["workload"]),
            int(start["repeat"]),
            str(start["arm"]),
        )
        if scenario in shards:
            raise ValueError(f"duplicate measurement shard for {scenario}")
        shards[scenario] = (start, requests, end, sha256_file(path))
    if set(shards) != set(EXPECTED_SCENARIOS):
        missing = sorted(set(EXPECTED_SCENARIOS) - set(shards))
        raise ValueError(
            f"measurement shards must contain the exact 32-cell matrix; missing={missing}"
        )

    first = next(iter(shards.values()))[0]
    benchmark = first.get("benchmark")
    graph_warmup = first.get("graph_warmup")
    if not _graph_warmup_trace_valid(graph_warmup) or any(
        start.get("graph_warmup") != graph_warmup
        for start, _, _, _ in shards.values()
    ):
        raise ValueError("binding shards do not share the exact graph-warmup trace")
    traces: dict[str, object] = {}
    for workload in WORKLOADS:
        candidates = [
            start.get("trace")
            for scenario, (start, _, _, _) in shards.items()
            if scenario[1] == workload
        ]
        if candidates:
            traces[workload] = candidates[0]
    combined: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "benchmark": benchmark,
            "graph_warmup": graph_warmup,
            "traces": traces,
        }
    ]
    for scenario in sorted(shards):
        start, requests, end, shard_sha256 = shards[scenario]
        trace = start.get("trace")
        combined.append(
            {
                **{
                    key: value
                    for key, value in start.items()
                    if key not in {"type", "row_index", "trace", "graph_warmup"}
                },
                "type": "scenario_start",
                "trace_sha256": (trace.get("sha256") if isinstance(trace, dict) else None),
                "graph_warmup_sha256": (
                    graph_warmup.get("sha256")
                    if isinstance(graph_warmup, dict)
                    else None
                ),
                "measurement_raw_sha256": shard_sha256,
            }
        )
        combined.extend(
            {key: value for key, value in row.items() if key != "row_index"} for row in requests
        )
        combined.append(
            {
                **{key: value for key, value in end.items() if key not in {"type", "row_index"}},
                "type": "scenario_end",
            }
        )
    sweep_shards: dict[
        tuple[int, int, str],
        tuple[
            dict[str, object],
            list[dict[str, object]],
            dict[str, object],
            str,
        ],
    ] = {}
    for path in sweep_paths:
        sweep_rows = _read_jsonl(path)
        start, requests, end = _parse_sweep_shard(sweep_rows)
        scenario = (
            int(start["tensor_parallel_size"]),
            int(start["arrival_rate_rps"]),
            str(start["arm"]),
        )
        if scenario in sweep_shards:
            raise ValueError(f"duplicate open-loop shard for {scenario}")
        sweep_shards[scenario] = (
            start,
            requests,
            end,
            sha256_file(path),
        )
    if set(sweep_shards) != set(EXPECTED_SWEEP_SCENARIOS):
        missing = sorted(set(EXPECTED_SWEEP_SCENARIOS) - set(sweep_shards))
        raise ValueError(
            f"open-loop shards must contain the exact 20-point grid; missing={missing}"
        )
    if any(
        start.get("graph_warmup") != graph_warmup
        for start, _, _, _ in sweep_shards.values()
    ):
        raise ValueError("open-loop shards do not share the binding graph-warmup trace")
    for scenario in sorted(sweep_shards):
        start, requests, end, shard_sha256 = sweep_shards[scenario]
        trace = start.get("trace")
        combined.append(
            {
                **{
                    key: value
                    for key, value in start.items()
                    if key not in {"type", "row_index", "trace", "graph_warmup"}
                },
                "type": "sweep_scenario_start",
                "trace_sha256": (trace.get("sha256") if isinstance(trace, dict) else None),
                "graph_warmup_sha256": (
                    graph_warmup.get("sha256")
                    if isinstance(graph_warmup, dict)
                    else None
                ),
                "measurement_raw_sha256": shard_sha256,
            }
        )
        combined.extend(
            {
                **{key: value for key, value in request.items() if key != "row_index"},
                "type": "sweep_request",
            }
            for request in requests
        )
        combined.append(
            {
                **{key: value for key, value in end.items() if key not in {"type", "row_index"}},
                "type": "sweep_scenario_end",
            }
        )
    combined.append(
        {
            "type": "run_end",
            "scenario_count": len(shards),
            "sweep_scenario_count": len(sweep_shards),
            "binding_request_count": sum(len(value[1]) for value in shards.values()),
            "sweep_request_count": sum(len(value[1]) for value in sweep_shards.values()),
            "request_count": (
                sum(len(value[1]) for value in shards.values())
                + sum(len(value[1]) for value in sweep_shards.values())
            ),
        }
    )
    return combined


def _parse_artifact_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[
    dict[str, object],
    dict[tuple[int, str, int, str], dict[str, object]],
    dict[tuple[int, str, int, str], list[dict[str, object]]],
    dict[tuple[int, str, int, str], dict[str, object]],
    dict[str, object],
]:
    if len(rows) < 2 or rows[0].get("type") != "run":
        raise ValueError("artifact must start with one run row")
    if rows[-1].get("type") != "run_end":
        raise ValueError("artifact must end with one run_end row")
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise ValueError("artifact row indices are not contiguous")
    run, run_end = rows[0], rows[-1]
    if run.get("schema_version") != SCHEMA_VERSION or run.get("gate") != GATE:
        raise ValueError("artifact schema/gate mismatch")
    starts: dict[tuple[int, str, int, str], dict[str, object]] = {}
    ends: dict[tuple[int, str, int, str], dict[str, object]] = {}
    requests: dict[tuple[int, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows[1:-1]:
        row_type = row.get("type")
        if row_type in {
            "sweep_scenario_start",
            "sweep_request",
            "sweep_scenario_end",
        }:
            continue
        scenario = (
            row.get("tensor_parallel_size"),
            row.get("workload"),
            row.get("repeat"),
            row.get("arm"),
        )
        if not _scenario_valid(*scenario):
            raise ValueError("artifact scenario row has invalid identity")
        typed = (int(scenario[0]), str(scenario[1]), int(scenario[2]), str(scenario[3]))
        if row.get("scenario_id") != _scenario_id(*typed):
            raise ValueError("artifact scenario ID mismatch")
        target = starts if row_type == "scenario_start" else ends
        if row_type in {"scenario_start", "scenario_end"}:
            if typed in target:
                raise ValueError(f"duplicate {row_type} for {typed}")
            target[typed] = row
        elif row_type == "request":
            requests[typed].append(row)
        else:
            raise ValueError(f"unknown artifact row type {row_type!r}")
    return run, starts, requests, ends, run_end


def _parse_sweep_artifact_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[
    dict[tuple[int, int, str], dict[str, object]],
    dict[tuple[int, int, str], list[dict[str, object]]],
    dict[tuple[int, int, str], dict[str, object]],
]:
    starts: dict[tuple[int, int, str], dict[str, object]] = {}
    ends: dict[tuple[int, int, str], dict[str, object]] = {}
    requests: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows[1:-1]:
        row_type = row.get("type")
        if row_type not in {
            "sweep_scenario_start",
            "sweep_request",
            "sweep_scenario_end",
        }:
            continue
        scenario = (
            row.get("tensor_parallel_size"),
            row.get("arrival_rate_rps"),
            row.get("arm"),
        )
        if not _sweep_scenario_valid(*scenario):
            raise ValueError("open-loop artifact row has invalid identity")
        typed = (int(scenario[0]), int(scenario[1]), str(scenario[2]))
        if row.get("scenario_id") != _sweep_scenario_id(*typed):
            raise ValueError("open-loop artifact scenario ID mismatch")
        target = starts if row_type == "sweep_scenario_start" else ends
        if row_type in {"sweep_scenario_start", "sweep_scenario_end"}:
            if typed in target:
                raise ValueError(f"duplicate {row_type} for {typed}")
            target[typed] = row
        else:
            requests[typed].append(row)
    return starts, requests, ends


def nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("nearest-rank quantile requires samples")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def _request_interval(row: Mapping[str, object]) -> tuple[int, int] | None:
    timing = row.get("timing_ns")
    if not isinstance(timing, dict):
        return None
    start = timing.get("start")
    end = timing.get("end")
    if type(start) is not int or type(end) is not int or start > end:
        return None
    return start, end


def _serialized_intervals(rows: Sequence[Mapping[str, object]]) -> bool:
    ordered = sorted(rows, key=lambda row: int(row.get("sequence", -1)))
    if [row.get("sequence") for row in ordered] != list(range(len(rows))):
        return False
    intervals = [_request_interval(row) for row in ordered]
    if any(interval is None for interval in intervals):
        return False
    typed = [interval for interval in intervals if interval is not None]
    return all(typed[index][1] <= typed[index + 1][0] for index in range(len(typed) - 1))


def _expected_completion_tokens(workload: str, phase: str) -> int:
    if phase == "warmup":
        return WARMUP_MAX_TOKENS
    if phase == "graph_warmup":
        return GRAPH_WARMUP_COMPLETION_TOKENS
    return int(request_config(workload)["max_tokens"])


def _graph_warmup_evidence_valid(
    rows: Sequence[Mapping[str, object]],
    *,
    graph_warmup: Mapping[str, object],
    workload: str,
) -> bool:
    if not _graph_warmup_trace_valid(graph_warmup) or len(rows) != GRAPH_WARMUP_REQUESTS:
        return False
    raw = graph_warmup["value"]
    assert isinstance(raw, dict)
    prompts = raw["prompts"]
    assert isinstance(prompts, list)
    by_sequence = {row.get("sequence"): row for row in rows}
    if len(by_sequence) != len(rows) or set(by_sequence) != set(range(GRAPH_WARMUP_REQUESTS)):
        return False
    previous_end: int | None = None
    sequence = 0
    for batch_size in GRAPH_WARMUP_BATCH_SIZES:
        batch: list[Mapping[str, object]] = []
        for batch_sequence in range(batch_size):
            row = by_sequence[sequence]
            expected = prompts[sequence]
            if not isinstance(expected, dict):
                return False
            if not (
                row.get("graph_batch_size") == batch_size
                and row.get("graph_batch_sequence") == batch_sequence
                and _request_evidence_valid(
                    row,
                    expected=expected,
                    workload=workload,
                    phase="graph_warmup",
                )
            ):
                return False
            batch.append(row)
            sequence += 1
        intervals = [_request_interval(row) for row in batch]
        if any(interval is None for interval in intervals):
            return False
        typed = [interval for interval in intervals if interval is not None]
        releases = {row.get("graph_warmup_release_ns") for row in batch}
        release_ns = next(iter(releases)) if len(releases) == 1 else None
        if not (
            type(release_ns) is int
            and release_ns <= min(interval[0] for interval in typed)
            and max(interval[0] for interval in typed) <= min(interval[1] for interval in typed)
            and (previous_end is None or previous_end <= min(interval[0] for interval in typed))
        ):
            return False
        previous_end = max(interval[1] for interval in typed)
    return sequence == GRAPH_WARMUP_REQUESTS


def _request_evidence_valid(
    row: Mapping[str, object],
    *,
    expected: Mapping[str, object],
    workload: str,
    phase: str,
) -> bool:
    usage = row.get("usage")
    timing = row.get("timing_ns")
    expected_completion = _expected_completion_tokens(workload, phase)
    return (
        row.get("phase") == phase
        and row.get("sequence") == expected.get("sequence")
        and row.get("attempts") == 1
        and row.get("prompt_text_sha256") == expected.get("prompt_text_sha256")
        and row.get("prompt_tokens") == expected.get("prompt_tokens")
        and row.get("prompt_token_ids_sha256") == expected.get("prompt_token_ids_sha256")
        and row.get("expected_completion_tokens") == expected_completion
        and row.get("http_status") == 200
        and isinstance(row.get("content_type"), str)
        and str(row["content_type"]).lower().startswith("text/event-stream")
        and isinstance(row.get("response_id"), str)
        and bool(row["response_id"])
        and row.get("status") == "success"
        and row.get("error_type") is None
        and row.get("done") is True
        and row.get("done_events") == 1
        and row.get("terminal_choice") is True
        and row.get("terminal_choice_events") == 1
        and row.get("finish_reasons") == ["length"]
        and row.get("usage_events") == 1
        and isinstance(usage, dict)
        and usage.get("prompt_tokens") == expected.get("prompt_tokens")
        and usage.get("completion_tokens") == expected_completion
        and type(usage.get("cached_tokens")) is int
        and 0 <= int(usage["cached_tokens"]) <= int(usage["prompt_tokens"])
        and isinstance(timing, dict)
        and type(timing.get("start")) is int
        and type(timing.get("first_token")) is int
        and type(timing.get("end")) is int
        and type(timing.get("ttft")) is int
        and type(timing.get("total")) is int
        and timing["start"] <= timing["first_token"] <= timing["end"]
        and timing["ttft"] == timing["first_token"] - timing["start"]
        and timing["total"] == timing["end"] - timing["start"]
        and timing["ttft"] >= 0
        and timing["total"] >= timing["ttft"]
        and _valid_sha256(row.get("output_text_sha256"))
        and _valid_sha256(row.get("stream_payload_sha256"))
    )


def _scenario_metric(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    formal = [row for row in rows if row.get("phase") == "measurement"]
    ttfts = [
        int(row["timing_ns"]["ttft"])
        for row in formal
        if row.get("status") == "success"
        and isinstance(row.get("timing_ns"), dict)
        and type(row["timing_ns"].get("ttft")) is int
    ]
    starts = [
        int(row["timing_ns"]["start"])
        for row in formal
        if isinstance(row.get("timing_ns"), dict) and type(row["timing_ns"].get("start")) is int
    ]
    ends = [
        int(row["timing_ns"]["end"])
        for row in formal
        if isinstance(row.get("timing_ns"), dict) and type(row["timing_ns"].get("end")) is int
    ]
    span_ns = max(ends) - min(starts) if starts and ends else 0
    slo_successes = sum(value <= TTFT_SLO_NS for value in ttfts)
    return {
        "request_count": len(formal),
        "success_count": sum(row.get("status") == "success" for row in formal),
        "ttft_sample_count": len(ttfts),
        "ttft_p50_ns": nearest_rank(ttfts, 0.50) if ttfts else None,
        "ttft_p99_ns": nearest_rank(ttfts, 0.99) if ttfts else None,
        "measurement_span_ns": span_ns,
        "ttft_slo_successes": slo_successes,
        "goodput_rps": (slo_successes * 1_000_000_000 / span_ns if span_ns > 0 else None),
    }


def _pooled_metric(
    scenario_metrics: Sequence[dict[str, object]],
    scenario_rows: Sequence[Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    ttfts = [
        int(row["timing_ns"]["ttft"])
        for rows in scenario_rows
        for row in rows
        if row.get("phase") == "measurement"
        and row.get("status") == "success"
        and isinstance(row.get("timing_ns"), dict)
        and type(row["timing_ns"].get("ttft")) is int
    ]
    spans = sum(int(metric["measurement_span_ns"]) for metric in scenario_metrics)
    slo_successes = sum(int(metric["ttft_slo_successes"]) for metric in scenario_metrics)
    return {
        "repeats": len(scenario_metrics),
        "samples": len(ttfts),
        "ttft_p50_ns": nearest_rank(ttfts, 0.50) if ttfts else None,
        "ttft_p99_ns": nearest_rank(ttfts, 0.99) if ttfts else None,
        "measurement_span_sum_ns": spans,
        "ttft_slo_successes": slo_successes,
        "goodput_rps": (slo_successes * 1_000_000_000 / spans if spans > 0 else None),
    }


def _fraction_record(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("fraction median requires samples")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _mad_fraction(values: Sequence[Fraction]) -> Fraction:
    median = _median_fraction(values)
    return _median_fraction([abs(value - median) for value in values])


def _paired_goodput_ratio(
    kairyu: Mapping[str, object],
    vllm: Mapping[str, object],
) -> Fraction | None:
    k_successes = kairyu.get("ttft_slo_successes")
    v_successes = vllm.get("ttft_slo_successes")
    k_span = kairyu.get("measurement_span_ns")
    v_span = vllm.get("measurement_span_ns")
    if not (
        type(k_successes) is int
        and type(v_successes) is int
        and type(k_span) is int
        and type(v_span) is int
        and k_successes >= 0
        and v_successes > 0
        and k_span > 0
        and v_span > 0
    ):
        return None
    return Fraction(k_successes * v_span, v_successes * k_span)


def _paired_latency_ratio(
    kairyu: Mapping[str, object],
    vllm: Mapping[str, object],
    *,
    metric: str,
) -> Fraction | None:
    k_value = kairyu.get(metric)
    v_value = vllm.get(metric)
    if not (
        type(k_value) is int
        and type(v_value) is int
        and k_value >= 0
        and v_value > 0
    ):
        return None
    return Fraction(k_value, v_value)


def _common_configuration(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.get(key)
        for key in (
            "served_model_name",
            "dtype",
            "tensor_parallel_size",
            "max_num_batched_tokens",
            "max_num_seqs",
            "enable_prefix_caching",
            "cache_block_size",
            "usable_cache_blocks",
            "cache_tokens",
            "kv_cache_dtype",
            "allocated_kv_cache_bytes_per_rank",
            "reserved_kv_cache_bytes_per_rank",
            "usable_kv_cache_bytes_per_rank",
            "chunked_prefill_enabled",
            "scheduler_policy",
            "max_model_len",
            "access_log_enabled",
        )
    }


def _runtime_build_identity(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.get(key)
        for key in (
            "arm",
            "version",
            "build_commit",
            "image_ref",
            "image_id",
            "stock_image",
            "cuda_version",
            "nccl_version",
            "package_versions",
            "source_clean",
            "source_status_policy",
            "v1_engine",
        )
    }


def _container_launch_static_identity(value: object) -> object:
    if not isinstance(value, dict) or not isinstance(value.get("value"), dict):
        return value
    raw = value["value"]
    projected = {
        key: item
        for key, item in raw.items()
        if key != "container_id"
    }
    return hashed_descriptor(projected)


def _startup_log_static_identity(value: object) -> object:
    if not isinstance(value, dict) or not isinstance(value.get("value"), dict):
        return value
    raw = value["value"]
    return raw.get("resolved_markers")


def _runtime_configuration_identity(value: Mapping[str, object]) -> dict[str, object]:
    return {
        **_runtime_build_identity(value),
        "container_launch": _container_launch_static_identity(
            value.get("container_launch")
        ),
        "resolved_backend": value.get("resolved_backend"),
        "startup_log_markers": _startup_log_static_identity(
            value.get("startup_log")
        ),
    }


def _container_id_from_provenance(value: Mapping[str, object]) -> object:
    runtime = value.get("runtime")
    launch = runtime.get("container_launch") if isinstance(runtime, dict) else None
    raw = launch.get("value") if isinstance(launch, dict) else None
    return raw.get("container_id") if isinstance(raw, dict) else None


def _provenance_matrix(
    starts: Mapping[tuple[int, str, int, str], Mapping[str, object]],
    ends: Mapping[tuple[int, str, int, str], Mapping[str, object]],
) -> tuple[dict[str, bool], dict[str, object]]:
    valid = bool(starts) and set(starts) == set(ends)
    values: dict[tuple[int, str, int, str], dict[str, object]] = {}
    stable = valid
    for scenario in set(starts) | set(ends):
        tp, _, _, arm = scenario
        start = starts.get(scenario, {}).get("provenance_start")
        end = ends.get(scenario, {}).get("provenance_end")
        valid &= _provenance_valid(start, tp=tp, arm=arm)
        valid &= _provenance_valid(end, tp=tp, arm=arm)
        stable &= start == end
        if isinstance(start, dict) and isinstance(start.get("value"), dict):
            values[scenario] = start["value"]

    generations = [value.get("server_generation") for value in values.values()]
    generation_unique = len(generations) == len(EXPECTED_SCENARIOS) and len(
        set(generations)
    ) == len(generations)
    container_ids = [_container_id_from_provenance(value) for value in values.values()]
    container_unique = (
        len(container_ids) == len(EXPECTED_SCENARIOS)
        and len(set(container_ids)) == len(container_ids)
        and all(_container_id_valid(item) for item in container_ids)
    )
    pair_order_exact = set(values) == set(EXPECTED_SCENARIOS)
    if pair_order_exact:
        for tp in TP_DEGREES:
            for workload in WORKLOADS:
                ordered_timestamps: list[int] = []
                for repeat, expected_order in enumerate(PAIR_ORDER):
                    timestamps = {
                        arm: values[(tp, workload, repeat, arm)]["server_started_ns"]
                        for arm in ARMS
                    }
                    ordered_timestamps.extend(
                        (
                            timestamps[expected_order[0]],
                            timestamps[expected_order[1]],
                        )
                    )
                pair_order_exact &= all(
                    ordered_timestamps[index] < ordered_timestamps[index + 1]
                    for index in range(len(ordered_timestamps) - 1)
                )

    model_host_exact = set(values) == set(EXPECTED_SCENARIOS)
    gpu_exact = model_host_exact
    runtime_config_exact = model_host_exact
    common_config_exact = model_host_exact
    if values:
        first = next(iter(values.values()))
        model_host_exact &= all(
            value.get("model") == first.get("model") and value.get("host") == first.get("host")
            for value in values.values()
        )
    for tp in TP_DEGREES:
        tp_values = [value for scenario, value in values.items() if scenario[0] == tp]
        if len(tp_values) != len(WORKLOADS) * REPEATS * len(ARMS):
            gpu_exact = False
            common_config_exact = False
            continue
        first_gpu = tp_values[0].get("gpu")
        gpu_exact &= all(value.get("gpu") == first_gpu for value in tp_values)
        by_arm: dict[str, list[dict[str, object]]] = {
            arm: [
                value
                for scenario, value in values.items()
                if scenario[0] == tp and scenario[3] == arm
            ]
            for arm in ARMS
        }
        for arm_values in by_arm.values():
            if len(arm_values) != len(WORKLOADS) * REPEATS:
                runtime_config_exact = False
                continue
            first_runtime = arm_values[0]["runtime"]
            first_config = arm_values[0]["configuration"]
            runtime_config_exact &= all(
                isinstance(value.get("runtime"), dict)
                and _runtime_configuration_identity(value["runtime"])
                == _runtime_configuration_identity(first_runtime)
                and value.get("configuration") == first_config
                and value.get("configuration_source")
                == arm_values[0].get("configuration_source")
                for value in arm_values
            )
        if all(by_arm.values()):
            common_config_exact &= _common_configuration(
                by_arm["kairyu"][0]["configuration"]
            ) == _common_configuration(by_arm["vllm-stock"][0]["configuration"])
    for arm in ARMS:
        arm_values = [value for scenario, value in values.items() if scenario[3] == arm]
        if len(arm_values) != len(TP_DEGREES) * len(WORKLOADS) * REPEATS:
            runtime_config_exact = False
        elif arm_values:
            first_runtime = arm_values[0]["runtime"]
            runtime_config_exact &= all(
                isinstance(value.get("runtime"), dict)
                and _runtime_build_identity(value["runtime"])
                == _runtime_build_identity(first_runtime)
                for value in arm_values
            )
    tp4_assignments = {
        tuple(zip(value["gpu"]["uuids"], value["gpu"]["pci_bus_ids"], strict=True))
        for scenario, value in values.items()
        if scenario[0] == 4
    }
    tp8_assignments = {
        tuple(zip(value["gpu"]["uuids"], value["gpu"]["pci_bus_ids"], strict=True))
        for scenario, value in values.items()
        if scenario[0] == 8
    }
    physical_assignment_exact = (
        len(tp4_assignments) == 1
        and len(tp8_assignments) == 1
        and set(next(iter(tp4_assignments))).issubset(set(next(iter(tp8_assignments))))
    )
    graph_disclosure = {
        _scenario_id(*scenario): value.get("configuration", {}).get("cuda_graph")
        for scenario, value in sorted(values.items())
    }
    return (
        {
            "provenance_schema_exact": valid,
            "provenance_start_end_stable": stable,
            "fresh_server_generation_per_scenario": generation_unique,
            "fresh_container_identity_per_scenario": container_unique,
            "balanced_kv_vk_vk_kv_server_start_order": pair_order_exact,
            "model_and_host_identity_exact": model_host_exact,
            "gpu_identity_exact_with_tp4_subset_of_tp8": (gpu_exact and physical_assignment_exact),
            "runtime_and_full_config_stable_within_arm_tp": (runtime_config_exact),
            "common_serving_configuration_equal_between_arms": (common_config_exact),
        },
        {"cuda_graph_parity_disclosure": graph_disclosure},
    )


def _evaluate_open_loop_sweeps(
    rows: Sequence[dict[str, object]],
    *,
    traces: object,
    graph_warmup: object,
    benchmark: object,
    dataset_sha256: str | None,
    binding_starts: Mapping[
        tuple[int, str, int, str],
        Mapping[str, object],
    ],
) -> tuple[dict[str, bool], dict[str, object], dict[str, object]]:
    starts, requests, ends = _parse_sweep_artifact_rows(rows)
    matrix_exact = set(starts) == set(requests) == set(ends) == set(EXPECTED_SWEEP_SCENARIOS)
    share_trace = traces.get("sharegpt") if isinstance(traces, dict) else None
    trace_exact = isinstance(dataset_sha256, str) and _trace_valid(
        share_trace,
        workload="sharegpt",
        dataset_sha256=dataset_sha256,
    )
    graph_warmup_exact = _graph_warmup_trace_valid(graph_warmup)
    raw_trace = share_trace.get("value") if isinstance(share_trace, dict) else None
    expected_warmups = raw_trace.get("warmup") if isinstance(raw_trace, dict) else []
    expected_requests = raw_trace.get("requests") if isinstance(raw_trace, dict) else []
    counts_exact = matrix_exact
    request_evidence_exact = matrix_exact and trace_exact
    envelopes_exact = matrix_exact
    schedule_exact = matrix_exact
    provenance_exact = matrix_exact
    fresh_generations = matrix_exact
    config_matches_binding = matrix_exact
    generations: list[object] = []
    container_ids: list[object] = []
    metrics: dict[str, object] = {}
    for scenario in sorted(set(starts) | set(requests) | set(ends)):
        tp, rate_rps, arm = scenario
        start = starts.get(scenario, {})
        end = ends.get(scenario, {})
        scenario_rows = requests.get(scenario, [])
        warmups = [row for row in scenario_rows if row.get("phase") == "warmup"]
        graph_warmups = [
            row for row in scenario_rows if row.get("phase") == "graph_warmup"
        ]
        formal = [row for row in scenario_rows if row.get("phase") == "sweep"]
        counts_exact &= (
            len(warmups) == WARMUP_REQUESTS
            and len(graph_warmups) == GRAPH_WARMUP_REQUESTS
            and len(formal) == SHAREGPT_REQUESTS
            and len(scenario_rows)
            == WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS + SHAREGPT_REQUESTS
            and end.get("warmup_request_count")
            == WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS
            and end.get("serial_warmup_request_count") == WARMUP_REQUESTS
            and end.get("graph_warmup_request_count") == GRAPH_WARMUP_REQUESTS
            and end.get("sweep_request_count") == SHAREGPT_REQUESTS
        )
        envelopes_exact &= (
            start.get("schema_version") == SCHEMA_VERSION
            and start.get("gate") == GATE
            and start.get("benchmark") == benchmark
            and _valid_sha256(start.get("measurement_raw_sha256"))
            and isinstance(graph_warmup, dict)
            and start.get("graph_warmup_sha256") == graph_warmup.get("sha256")
        )
        graph_warmup_exact &= (
            isinstance(graph_warmup, dict)
            and _graph_warmup_evidence_valid(
                graph_warmups,
                graph_warmup=graph_warmup,
                workload="sharegpt",
            )
        )
        serial_intervals = [_request_interval(row) for row in warmups]
        graph_intervals = [_request_interval(row) for row in graph_warmups]
        formal_intervals = [_request_interval(row) for row in formal]
        graph_warmup_exact &= (
            _serialized_intervals(warmups)
            and all(interval is not None for interval in graph_intervals)
            and all(interval is not None for interval in formal_intervals)
            and bool(serial_intervals)
            and bool(graph_intervals)
            and bool(formal_intervals)
            and max(interval[1] for interval in serial_intervals if interval is not None)
            <= min(interval[0] for interval in graph_intervals if interval is not None)
            and max(interval[1] for interval in graph_intervals if interval is not None)
            <= min(interval[0] for interval in formal_intervals if interval is not None)
        )
        trace_exact &= isinstance(share_trace, dict) and start.get(
            "trace_sha256"
        ) == share_trace.get("sha256")
        if not (
            isinstance(expected_warmups, list)
            and isinstance(expected_requests, list)
            and len(warmups) == len(expected_warmups)
            and len(formal) == len(expected_requests)
        ):
            request_evidence_exact = False
            schedule_exact = False
        else:
            for phase, actual_rows, expected_rows in (
                ("warmup", warmups, expected_warmups),
                ("sweep", formal, expected_requests),
            ):
                by_sequence = {row.get("sequence"): row for row in actual_rows}
                request_evidence_exact &= len(by_sequence) == len(actual_rows) and set(
                    by_sequence
                ) == set(range(len(expected_rows)))
                for sequence, expected in enumerate(expected_rows):
                    actual = by_sequence.get(sequence, {})
                    if not isinstance(expected, dict):
                        request_evidence_exact = False
                        continue
                    request_evidence_exact &= _request_evidence_valid(
                        actual,
                        expected=expected,
                        workload="sharegpt",
                        phase=phase,
                    )
                    if phase == "sweep":
                        period_ns = 1_000_000_000 // rate_rps
                        schedule_exact &= (
                            actual.get("arrival_rate_rps") == rate_rps
                            and actual.get("scheduled_offset_ns") == sequence * period_ns
                            and type(actual.get("scheduled_start_ns")) is int
                            and type(actual.get("release_lag_ns")) is int
                            and isinstance(actual.get("timing_ns"), dict)
                            and actual["timing_ns"].get("start")
                            == actual["scheduled_start_ns"] + actual["release_lag_ns"]
                        )
        start_provenance = start.get("provenance_start")
        end_provenance = end.get("provenance_end")
        provenance_exact &= (
            _provenance_valid(start_provenance, tp=tp, arm=arm)
            and start_provenance == end_provenance
        )
        if isinstance(start_provenance, dict) and isinstance(start_provenance.get("value"), dict):
            provenance_value = start_provenance["value"]
            generations.append(provenance_value.get("server_generation"))
            container_ids.append(_container_id_from_provenance(provenance_value))
            binding = binding_starts.get((tp, "sharegpt", 0, arm), {}).get("provenance_start")
            binding_value = binding.get("value") if isinstance(binding, dict) else None
            if isinstance(binding_value, dict):
                runtime = provenance_value.get("runtime")
                binding_runtime = binding_value.get("runtime")
                config_matches_binding &= (
                    provenance_value.get("host") == binding_value.get("host")
                    and provenance_value.get("gpu") == binding_value.get("gpu")
                    and provenance_value.get("model") == binding_value.get("model")
                    and isinstance(runtime, dict)
                    and isinstance(binding_runtime, dict)
                    and _runtime_configuration_identity(runtime)
                    == _runtime_configuration_identity(binding_runtime)
                    and provenance_value.get("configuration") == binding_value.get("configuration")
                    and provenance_value.get("configuration_source")
                    == binding_value.get("configuration_source")
                )
            else:
                config_matches_binding = False
        else:
            fresh_generations = False
            config_matches_binding = False
        ttfts = [
            int(row["timing_ns"]["ttft"])
            for row in formal
            if row.get("status") == "success"
            and isinstance(row.get("timing_ns"), dict)
            and type(row["timing_ns"].get("ttft")) is int
        ]
        starts_ns = [
            int(row["timing_ns"]["start"])
            for row in formal
            if isinstance(row.get("timing_ns"), dict) and type(row["timing_ns"].get("start")) is int
        ]
        ends_ns = [
            int(row["timing_ns"]["end"])
            for row in formal
            if isinstance(row.get("timing_ns"), dict) and type(row["timing_ns"].get("end")) is int
        ]
        span_ns = max(ends_ns) - min(starts_ns) if starts_ns and ends_ns else 0
        slo_successes = sum(value <= TTFT_SLO_NS for value in ttfts)
        metrics[_sweep_scenario_id(*scenario)] = {
            "offered_rate_rps": rate_rps,
            "requests": len(formal),
            "successes": sum(row.get("status") == "success" for row in formal),
            "ttft_samples": len(ttfts),
            "ttft_p50_ns": nearest_rank(ttfts, 0.50) if ttfts else None,
            "ttft_p99_ns": nearest_rank(ttfts, 0.99) if ttfts else None,
            "measurement_span_ns": span_ns,
            "ttft_slo_successes": slo_successes,
            "goodput_rps": (slo_successes * 1_000_000_000 / span_ns if span_ns > 0 else None),
        }
        timing = [row.get("timing_ns") for row in formal if isinstance(row.get("timing_ns"), dict)]
        expected_window = (
            {
                "start": min(int(item["start"]) for item in timing),
                "end": max(int(item["end"]) for item in timing),
            }
            if timing
            else None
        )
        ordered_formal = sorted(
            formal,
            key=lambda row: int(row.get("sequence", -1)),
        )
        scheduled = [
            row.get("scheduled_start_ns")
            for row in ordered_formal
            if type(row.get("scheduled_start_ns")) is int
        ]
        schedule_exact &= (
            end.get("measurement_window_ns") == expected_window
            and len(scheduled) == SHAREGPT_REQUESTS
            and end.get("arrival_anchor_ns") == min(scheduled)
            and all(
                int(value) - int(ordered_formal[index]["scheduled_offset_ns"])
                == end.get("arrival_anchor_ns")
                for index, value in enumerate(scheduled)
            )
        )
    binding_generations = {
        provenance["value"].get("server_generation")
        for start in binding_starts.values()
        if isinstance((provenance := start.get("provenance_start")), dict)
        and isinstance(provenance.get("value"), dict)
    }
    binding_container_ids = {
        _container_id_from_provenance(provenance["value"])
        for start in binding_starts.values()
        if isinstance((provenance := start.get("provenance_start")), dict)
        and isinstance(provenance.get("value"), dict)
    }
    fresh_generations &= (
        len(generations) == len(EXPECTED_SWEEP_SCENARIOS)
        and len(set(generations)) == len(generations)
        and not (set(generations) & binding_generations)
        and len(container_ids) == len(EXPECTED_SWEEP_SCENARIOS)
        and len(set(container_ids)) == len(container_ids)
        and all(_container_id_valid(item) for item in container_ids)
        and not (set(container_ids) & binding_container_ids)
    )
    return (
        {
            "report_only_open_loop_rate_grid_exact": matrix_exact,
            "open_loop_trace_and_request_evidence_exact": (
                trace_exact
                and graph_warmup_exact
                and counts_exact
                and request_evidence_exact
            ),
            "open_loop_benchmark_and_source_shard_hashes_exact": (envelopes_exact),
            "open_loop_fixed_arrival_schedule_replays_exact": schedule_exact,
            "open_loop_fresh_server_provenance_exact": (provenance_exact and fresh_generations),
            "open_loop_runtime_config_matches_binding_cells": (config_matches_binding),
        },
        metrics,
        {
            "binding": False,
            "rates_selected_before_measurement": list(OPEN_LOOP_RATES_RPS),
            "best_observed_rate_is_not_a_gate": True,
        },
    )


def evaluate_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[
    dict[str, bool],
    dict[str, object],
    dict[str, object],
]:
    run, starts, requests, ends, run_end = _parse_artifact_rows(rows)
    required_matrix = set(starts) == set(requests) == set(ends) == set(
        EXPECTED_SCENARIOS
    ) and run_end.get("scenario_count") == len(EXPECTED_SCENARIOS)
    benchmark = run.get("benchmark")
    dataset_sha256 = (
        benchmark.get("sharegpt_dataset", {}).get("sha256") if isinstance(benchmark, dict) else None
    )
    benchmark_exact = (
        isinstance(dataset_sha256, str)
        and _valid_sha256(dataset_sha256)
        and benchmark == benchmark_config(dataset_sha256)
    )
    traces = run.get("traces")
    graph_warmup = run.get("graph_warmup")
    traces_valid = (
        isinstance(traces, dict)
        and set(traces) == set(WORKLOADS)
        and isinstance(dataset_sha256, str)
        and all(
            _trace_valid(
                traces.get(workload),
                workload=workload,
                dataset_sha256=dataset_sha256,
            )
            for workload in WORKLOADS
        )
    )
    graph_warmup_valid = _graph_warmup_trace_valid(graph_warmup)

    counts_exact = required_matrix
    request_identity_exact = required_matrix and traces_valid
    strict_success_exact = required_matrix and traces_valid
    trace_hash_exact = required_matrix and traces_valid
    scenario_envelopes_exact = required_matrix and benchmark_exact
    window_exact = required_matrix
    warmup_shape_exact = required_matrix
    graph_warmup_shape_exact = required_matrix and graph_warmup_valid
    sharegpt_burst_shape_exact = required_matrix
    shared_prefix_serial_shape_exact = required_matrix
    scenario_metrics: dict[tuple[int, str, int, str], dict[str, object]] = {}
    for scenario in sorted(set(starts) | set(requests) | set(ends)):
        _, workload, _, _ = scenario
        start = starts.get(scenario, {})
        end = ends.get(scenario, {})
        scenario_rows = requests.get(scenario, [])
        warmups = [row for row in scenario_rows if row.get("phase") == "warmup"]
        graph_warmups = [
            row for row in scenario_rows if row.get("phase") == "graph_warmup"
        ]
        formal = [row for row in scenario_rows if row.get("phase") == "measurement"]
        expected_formal = SHAREGPT_REQUESTS if workload == "sharegpt" else SHARED_PREFIX_REQUESTS
        counts_exact &= (
            len(warmups) == WARMUP_REQUESTS
            and len(graph_warmups) == GRAPH_WARMUP_REQUESTS
            and len(formal) == expected_formal
            and len(scenario_rows)
            == WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS + expected_formal
            and end.get("warmup_request_count")
            == WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS
            and end.get("serial_warmup_request_count") == WARMUP_REQUESTS
            and end.get("graph_warmup_request_count") == GRAPH_WARMUP_REQUESTS
            and end.get("measurement_request_count") == expected_formal
        )
        warmup_intervals = [_request_interval(row) for row in warmups]
        graph_warmup_intervals = [_request_interval(row) for row in graph_warmups]
        formal_intervals = [_request_interval(row) for row in formal]
        warmup_shape_exact &= (
            len(warmups) == WARMUP_REQUESTS
            and _serialized_intervals(warmups)
            and len(formal_intervals) == expected_formal
            and all(interval is not None for interval in formal_intervals)
            and len(graph_warmup_intervals) == GRAPH_WARMUP_REQUESTS
            and all(interval is not None for interval in graph_warmup_intervals)
            and bool(formal_intervals)
            and bool(warmup_intervals)
            and bool(graph_warmup_intervals)
            and max(interval[1] for interval in warmup_intervals if interval is not None)
            <= min(
                interval[0]
                for interval in graph_warmup_intervals
                if interval is not None
            )
        )
        graph_warmup_shape_exact &= (
            isinstance(graph_warmup, dict)
            and _graph_warmup_evidence_valid(
                graph_warmups,
                graph_warmup=graph_warmup,
                workload=workload,
            )
            and len(graph_warmup_intervals) == GRAPH_WARMUP_REQUESTS
            and all(interval is not None for interval in graph_warmup_intervals)
            and max(
                interval[1]
                for interval in graph_warmup_intervals
                if interval is not None
            )
            <= min(interval[0] for interval in formal_intervals if interval is not None)
        )
        if workload == "sharegpt":
            valid_intervals = [interval for interval in formal_intervals if interval is not None]
            release_values = {row.get("burst_release_ns") for row in formal}
            release_ns = next(iter(release_values)) if len(release_values) == 1 else None
            sharegpt_burst_shape_exact &= (
                len(valid_intervals) == SHAREGPT_CONCURRENCY
                and type(release_ns) is int
                and release_ns <= min(interval[0] for interval in valid_intervals)
                and max(interval[0] for interval in valid_intervals)
                <= min(interval[1] for interval in valid_intervals)
            )
        else:
            shared_prefix_serial_shape_exact &= len(
                formal
            ) == SHARED_PREFIX_REQUESTS and _serialized_intervals(formal)
        scenario_envelopes_exact &= (
            start.get("schema_version") == SCHEMA_VERSION
            and start.get("gate") == GATE
            and start.get("benchmark") == benchmark
            and _valid_sha256(start.get("measurement_raw_sha256"))
            and isinstance(graph_warmup, dict)
            and start.get("graph_warmup_sha256") == graph_warmup.get("sha256")
        )
        trace = traces.get(workload) if isinstance(traces, dict) else None
        raw_trace = trace.get("value") if isinstance(trace, dict) else None
        expected_warmup = raw_trace.get("warmup") if isinstance(raw_trace, dict) else []
        expected_requests = raw_trace.get("requests") if isinstance(raw_trace, dict) else []
        trace_hash_exact &= isinstance(trace, dict) and start.get("trace_sha256") == trace.get(
            "sha256"
        )
        if not (
            isinstance(expected_warmup, list)
            and isinstance(expected_requests, list)
            and len(warmups) == len(expected_warmup)
            and len(formal) == len(expected_requests)
        ):
            request_identity_exact = False
            strict_success_exact = False
        else:
            for phase, actual_rows, expected_rows in (
                ("warmup", warmups, expected_warmup),
                ("measurement", formal, expected_requests),
            ):
                actual_by_sequence = {row.get("sequence"): row for row in actual_rows}
                request_identity_exact &= len(actual_by_sequence) == len(actual_rows) and set(
                    actual_by_sequence
                ) == set(range(len(expected_rows)))
                for sequence, expected in enumerate(expected_rows):
                    actual = actual_by_sequence.get(sequence, {})
                    if not isinstance(expected, dict):
                        request_identity_exact = False
                        strict_success_exact = False
                        continue
                    request_identity_exact &= all(
                        actual.get(key) == expected.get(key)
                        for key in (
                            "sequence",
                            "prompt_text_sha256",
                            "prompt_tokens",
                            "prompt_token_ids_sha256",
                        )
                    )
                    strict_success_exact &= _request_evidence_valid(
                        actual,
                        expected=expected,
                        workload=workload,
                        phase=phase,
                    )
        metric = _scenario_metric(scenario_rows)
        scenario_metrics[scenario] = metric
        timing = [row.get("timing_ns") for row in formal if isinstance(row.get("timing_ns"), dict)]
        expected_window = (
            {
                "start": min(int(item["start"]) for item in timing),
                "end": max(int(item["end"]) for item in timing),
            }
            if timing
            else None
        )
        window_exact &= end.get("measurement_window_ns") == expected_window
    expected_binding_requests = len(EXPECTED_SCENARIOS) * (
        WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS
    ) + len(TP_DEGREES) * len(ARMS) * REPEATS * (
        SHAREGPT_REQUESTS + SHARED_PREFIX_REQUESTS
    )
    counts_exact &= run_end.get("binding_request_count") == expected_binding_requests

    pooled: dict[tuple[int, str, str], dict[str, object]] = {}
    for tp in TP_DEGREES:
        for workload in WORKLOADS:
            for arm in ARMS:
                scenarios = [(tp, workload, repeat, arm) for repeat in range(REPEATS)]
                metrics = [
                    scenario_metrics[scenario]
                    for scenario in scenarios
                    if scenario in scenario_metrics
                ]
                raw = [requests[scenario] for scenario in scenarios if scenario in requests]
                pooled[(tp, workload, arm)] = _pooled_metric(metrics, raw)

    comparisons: dict[str, object] = {}
    goodput_gate = required_matrix
    p99_gate = required_matrix
    shared_p50_gate = required_matrix
    for tp in TP_DEGREES:
        share_k = pooled.get((tp, "sharegpt", "kairyu"), {})
        share_v = pooled.get((tp, "sharegpt", "vllm-stock"), {})
        prefix_k = pooled.get((tp, "shared-prefix", "kairyu"), {})
        prefix_v = pooled.get((tp, "shared-prefix", "vllm-stock"), {})
        k_goodput = share_k.get("goodput_rps")
        v_goodput = share_v.get("goodput_rps")
        k_slo_successes = share_k.get("ttft_slo_successes")
        v_slo_successes = share_v.get("ttft_slo_successes")
        k_span_ns = share_k.get("measurement_span_sum_ns")
        v_span_ns = share_v.get("measurement_span_sum_ns")
        k_p99 = share_k.get("ttft_p99_ns")
        v_p99 = share_v.get("ttft_p99_ns")
        k_p50 = prefix_k.get("ttft_p50_ns")
        v_p50 = prefix_v.get("ttft_p50_ns")
        goodput_ratio = (
            float(k_goodput) / float(v_goodput)
            if isinstance(k_goodput, (int, float))
            and isinstance(v_goodput, (int, float))
            and float(v_goodput) > 0
            else None
        )
        p99_ratio = (
            int(k_p99) / int(v_p99)
            if type(k_p99) is int and type(v_p99) is int and v_p99 > 0
            else None
        )
        p50_ratio = (
            int(k_p50) / int(v_p50)
            if type(k_p50) is int and type(v_p50) is int and v_p50 > 0
            else None
        )
        floor_numerator, floor_denominator = GOODPUT_RATIO_FLOOR_FRACTION
        goodput_gate &= (
            type(k_slo_successes) is int
            and type(v_slo_successes) is int
            and type(k_span_ns) is int
            and type(v_span_ns) is int
            and k_slo_successes >= 0
            and v_slo_successes > 0
            and k_span_ns > 0
            and v_span_ns > 0
            and floor_denominator * k_slo_successes * v_span_ns
            >= floor_numerator * v_slo_successes * k_span_ns
        )
        p99_gate &= type(k_p99) is int and type(v_p99) is int and v_p99 > 0 and k_p99 <= v_p99
        ceiling_numerator, ceiling_denominator = SHARED_PREFIX_P50_RATIO_CEILING_FRACTION
        shared_p50_gate &= (
            type(k_p50) is int
            and type(v_p50) is int
            and v_p50 > 0
            and ceiling_denominator * k_p50 <= ceiling_numerator * v_p50
        )
        paired_rounds: list[dict[str, object]] = []
        paired_goodput: list[Fraction] = []
        paired_p99: list[Fraction] = []
        paired_prefix_p50: list[Fraction] = []
        for repeat in range(REPEATS):
            share_k_repeat = scenario_metrics.get(
                (tp, "sharegpt", repeat, "kairyu"),
                {},
            )
            share_v_repeat = scenario_metrics.get(
                (tp, "sharegpt", repeat, "vllm-stock"),
                {},
            )
            prefix_k_repeat = scenario_metrics.get(
                (tp, "shared-prefix", repeat, "kairyu"),
                {},
            )
            prefix_v_repeat = scenario_metrics.get(
                (tp, "shared-prefix", repeat, "vllm-stock"),
                {},
            )
            goodput_fraction = _paired_goodput_ratio(
                share_k_repeat,
                share_v_repeat,
            )
            p99_fraction = _paired_latency_ratio(
                share_k_repeat,
                share_v_repeat,
                metric="ttft_p99_ns",
            )
            prefix_p50_fraction = _paired_latency_ratio(
                prefix_k_repeat,
                prefix_v_repeat,
                metric="ttft_p50_ns",
            )
            if goodput_fraction is not None:
                paired_goodput.append(goodput_fraction)
            if p99_fraction is not None:
                paired_p99.append(p99_fraction)
            if prefix_p50_fraction is not None:
                paired_prefix_p50.append(prefix_p50_fraction)
            paired_rounds.append(
                {
                    "repeat": repeat,
                    "server_start_order": list(PAIR_ORDER[repeat]),
                    "sharegpt_goodput_kairyu_over_vllm": (
                        _fraction_record(goodput_fraction)
                        if goodput_fraction is not None
                        else None
                    ),
                    "sharegpt_ttft_p99_kairyu_over_vllm": (
                        _fraction_record(p99_fraction)
                        if p99_fraction is not None
                        else None
                    ),
                    "shared_prefix_ttft_p50_kairyu_over_vllm": (
                        _fraction_record(prefix_p50_fraction)
                        if prefix_p50_fraction is not None
                        else None
                    ),
                }
            )
        paired_complete = (
            len(paired_goodput)
            == len(paired_p99)
            == len(paired_prefix_p50)
            == REPEATS
        )
        paired_goodput_median = (
            _median_fraction(paired_goodput) if paired_complete else None
        )
        paired_p99_median = (
            _median_fraction(paired_p99) if paired_complete else None
        )
        paired_prefix_p50_median = (
            _median_fraction(paired_prefix_p50) if paired_complete else None
        )
        paired_goodput_mad = (
            _mad_fraction(paired_goodput) if paired_complete else None
        )
        paired_p99_mad = (
            _mad_fraction(paired_p99) if paired_complete else None
        )
        paired_prefix_p50_mad = (
            _mad_fraction(paired_prefix_p50) if paired_complete else None
        )
        comparisons[f"tp{tp}"] = {
            "sharegpt": {
                "kairyu": share_k,
                "vllm-stock": share_v,
                "kairyu_over_vllm_goodput_ratio": goodput_ratio,
                "kairyu_over_vllm_ttft_p99_ratio": p99_ratio,
            },
            "shared-prefix": {
                "kairyu": prefix_k,
                "vllm-stock": prefix_v,
                "kairyu_over_vllm_ttft_p50_ratio": p50_ratio,
            },
            "paired_rounds": paired_rounds,
            "paired_median": {
                "method": (
                    "exact median of four repeat ratios; for even N, "
                    "the middle two exact Fractions are averaged"
                ),
                "sharegpt_goodput_kairyu_over_vllm": (
                    _fraction_record(paired_goodput_median)
                    if paired_goodput_median is not None
                    else None
                ),
                "sharegpt_ttft_p99_kairyu_over_vllm": (
                    _fraction_record(paired_p99_median)
                    if paired_p99_median is not None
                    else None
                ),
                "shared_prefix_ttft_p50_kairyu_over_vllm": (
                    _fraction_record(paired_prefix_p50_median)
                    if paired_prefix_p50_median is not None
                    else None
                ),
                "sharegpt_goodput_kairyu_over_vllm_mad": (
                    _fraction_record(paired_goodput_mad)
                    if paired_goodput_mad is not None
                    else None
                ),
                "sharegpt_ttft_p99_kairyu_over_vllm_mad": (
                    _fraction_record(paired_p99_mad)
                    if paired_p99_mad is not None
                    else None
                ),
                "shared_prefix_ttft_p50_kairyu_over_vllm_mad": (
                    _fraction_record(paired_prefix_p50_mad)
                    if paired_prefix_p50_mad is not None
                    else None
                ),
                "binding": False,
            },
        }
    provenance_checks, provenance_diagnostics = _provenance_matrix(starts, ends)
    sweep_checks, sweep_metrics, sweep_diagnostics = _evaluate_open_loop_sweeps(
        rows,
        traces=traces,
        graph_warmup=graph_warmup,
        benchmark=benchmark,
        dataset_sha256=(dataset_sha256 if isinstance(dataset_sha256, str) else None),
        binding_starts=starts,
    )
    expected_sweep_requests = len(EXPECTED_SWEEP_SCENARIOS) * (
        WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS + SHAREGPT_REQUESTS
    )
    sweep_counts_exact = (
        run_end.get("sweep_scenario_count") == len(EXPECTED_SWEEP_SCENARIOS)
        and run_end.get("sweep_request_count") == expected_sweep_requests
        and run_end.get("request_count") == expected_binding_requests + expected_sweep_requests
    )
    sweep_checks["report_only_open_loop_raw_counts_exact"] = sweep_counts_exact
    comparisons["open_loop_report_only"] = sweep_metrics
    checks = {
        "required_tp_arm_workload_repeat_matrix_exact": required_matrix,
        "fixed_benchmark_configuration_exact": benchmark_exact,
        "scenario_benchmark_and_source_shard_hashes_exact": (scenario_envelopes_exact),
        "arm_neutral_trace_identity_and_geometry_exact": (
            traces_valid and trace_hash_exact and request_identity_exact
        ),
        "every_planned_request_retained_exactly_once": counts_exact,
        "strict_streaming_all_requests_successful_without_retry": (strict_success_exact),
        "measurement_windows_replay_exact": window_exact,
        "fixed_serial_warmup_precedes_measurement_exact": warmup_shape_exact,
        "fixed_synchronized_graph_warmup_precedes_measurement_exact": (
            graph_warmup_shape_exact
        ),
        "sharegpt_synchronized_concurrency_128_shape_exact": (sharegpt_burst_shape_exact),
        "shared_prefix_serial_execution_shape_exact": (shared_prefix_serial_shape_exact),
        **provenance_checks,
        **sweep_checks,
        "sharegpt_goodput_kairyu_at_least_0_95_vllm_each_tp": (goodput_gate),
        "sharegpt_ttft_p99_kairyu_no_worse_than_vllm_each_tp": p99_gate,
        "shared_prefix_ttft_p50_kairyu_at_most_0_80_vllm_each_tp": (shared_p50_gate),
    }
    diagnostics = {
        **provenance_diagnostics,
        "open_loop_report_only": sweep_diagnostics,
        "scenario_metrics": {
            _scenario_id(*scenario): metric for scenario, metric in sorted(scenario_metrics.items())
        },
        "method": {
            "warmup_in_raw_but_excluded": True,
            "outlier_removal": False,
            "gate_uses_unrounded_values": True,
            "paired_order_diagnostics": (
                "four balanced K/V, V/K, V/K, K/V repeats reduce order bias; "
                "exact-Fraction per-repeat ratios and median/MAD disclose "
                "variability and are non-binding"
            ),
            "open_loop_arrival_sweep_is_binding": False,
        },
    }
    return checks, comparisons, diagnostics


def assemble_manifest(
    rows: Sequence[dict[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    if not _valid_sha256(raw_sha256):
        raise ValueError("manifest requires a raw SHA-256")
    checks, comparisons, diagnostics = evaluate_rows(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "raw": {
            "name": RAW_NAME,
            "sha256": raw_sha256,
            "row_count": len(rows),
        },
        "comparisons": comparisons,
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
    stored = _read_jsonl(raw_path)
    manifest = assemble_manifest(stored, raw_sha256=sha256_file(raw_path))
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
    raise ValueError(f"artifact path must be a directory, {RAW_NAME}, or {MANIFEST_NAME}")


def verify_artifact(
    path: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    raw_path, manifest_path = _artifact_paths(path)
    rows = _read_jsonl(raw_path)
    manifest = _read_json_object(manifest_path)
    raw_sha256 = sha256_file(raw_path)
    raw = manifest.get("raw")
    if not isinstance(raw, dict) or raw.get("sha256") != raw_sha256:
        raise ValueError("raw SHA-256 does not match manifest")
    replayed = assemble_manifest(rows, raw_sha256=raw_sha256)
    if manifest != replayed:
        raise ValueError("manifest differs from independent raw replay")
    if assert_gate and replayed["passed"] is not True:
        raise RuntimeError("G2 A6 artifact does not pass every binding check")
    return replayed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-traces",
        help="tokenize the pinned dataset once for reuse by every fresh server",
    )
    prepare_parser.add_argument("--tokenizer", type=Path, required=True)
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument(
        "--dataset-sha256",
        default=SHAREGPT_DATASET_SHA256,
    )
    prepare_parser.add_argument("--output", type=Path, required=True)

    measure_parser = subparsers.add_parser(
        "measure",
        help="measure one fresh-server TP/arm/workload/repeat scenario",
    )
    measure_parser.add_argument("--tp", type=int, choices=TP_DEGREES, required=True)
    measure_parser.add_argument("--arm", choices=ARMS, required=True)
    measure_parser.add_argument("--workload", choices=WORKLOADS, required=True)
    measure_parser.add_argument(
        "--repeat",
        type=int,
        choices=range(REPEATS),
        required=True,
    )
    measure_parser.add_argument("--endpoint", required=True)
    measure_parser.add_argument("--tokenizer", type=Path, required=True)
    measure_parser.add_argument("--dataset", type=Path)
    measure_parser.add_argument("--trace-bundle", type=Path)
    measure_parser.add_argument(
        "--dataset-sha256",
        default=SHAREGPT_DATASET_SHA256,
    )
    measure_parser.add_argument("--provenance", type=Path, required=True)
    measure_parser.add_argument("--api-key")
    measure_parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
    )
    measure_parser.add_argument("--output", type=Path, required=True)

    sweep_parser = subparsers.add_parser(
        "measure-sweep",
        help="measure one fresh-server report-only open-loop rate",
    )
    sweep_parser.add_argument("--tp", type=int, choices=TP_DEGREES, required=True)
    sweep_parser.add_argument("--arm", choices=ARMS, required=True)
    sweep_parser.add_argument(
        "--rate-rps",
        type=int,
        choices=OPEN_LOOP_RATES_RPS,
        required=True,
    )
    sweep_parser.add_argument("--endpoint", required=True)
    sweep_parser.add_argument("--tokenizer", type=Path, required=True)
    sweep_parser.add_argument("--dataset", type=Path)
    sweep_parser.add_argument("--trace-bundle", type=Path)
    sweep_parser.add_argument(
        "--dataset-sha256",
        default=SHAREGPT_DATASET_SHA256,
    )
    sweep_parser.add_argument("--provenance", type=Path, required=True)
    sweep_parser.add_argument("--api-key")
    sweep_parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
    )
    sweep_parser.add_argument("--output", type=Path, required=True)

    assemble_parser = subparsers.add_parser(
        "assemble",
        help="assemble the exact 32-shard matrix into retained evidence",
    )
    assemble_parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        required=True,
        help="one measurement JSONL shard; repeat 32 times",
    )
    assemble_parser.add_argument(
        "--sweep-raw",
        type=Path,
        action="append",
        required=True,
        help="one fresh-server open-loop JSONL shard; repeat 20 times",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-traces":
        bundle = write_trace_bundle(
            args.output,
            dataset_path=args.dataset,
            tokenizer_path=args.tokenizer,
            dataset_sha256=args.dataset_sha256,
        )
        traces = bundle["traces"]
        assert isinstance(traces, dict)
        print(
            canonical_json(
                {
                    "output": str(args.output),
                    "sha256": sha256_file(args.output),
                    "trace_sha256": {
                        workload: traces[workload]["sha256"] for workload in WORKLOADS
                    },
                }
            )
        )
        return 0
    if args.command == "measure":
        if args.workload == "sharegpt" and args.dataset is None and args.trace_bundle is None:
            parser.error("measure --workload sharegpt requires --dataset or --trace-bundle")
        rows = asyncio.run(
            measure(
                tensor_parallel_size=args.tp,
                arm=args.arm,
                workload=args.workload,
                repeat=args.repeat,
                endpoint=args.endpoint,
                tokenizer_path=args.tokenizer,
                dataset_path=args.dataset,
                trace_bundle_path=args.trace_bundle,
                dataset_sha256=args.dataset_sha256,
                provenance_path=args.provenance,
                api_key=args.api_key,
                request_timeout_s=args.request_timeout_s,
            )
        )
        print(canonical_json(write_measurement_shard(args.output, rows)))
        return 0
    if args.command == "measure-sweep":
        if args.dataset is None and args.trace_bundle is None:
            parser.error("measure-sweep requires --dataset or --trace-bundle")
        rows = asyncio.run(
            measure_sweep(
                tensor_parallel_size=args.tp,
                arm=args.arm,
                rate_rps=args.rate_rps,
                endpoint=args.endpoint,
                tokenizer_path=args.tokenizer,
                dataset_path=args.dataset,
                trace_bundle_path=args.trace_bundle,
                dataset_sha256=args.dataset_sha256,
                provenance_path=args.provenance,
                api_key=args.api_key,
                request_timeout_s=args.request_timeout_s,
            )
        )
        print(canonical_json(write_sweep_shard(args.output, rows)))
        return 0
    if args.command == "assemble":
        rows = assemble_rows(tuple(args.raw), tuple(args.sweep_raw))
        manifest = write_artifact(args.output_dir, rows)
        print(canonical_json(manifest))
        if args.assert_gate and manifest["passed"] is not True:
            raise RuntimeError("G2 A6 artifact does not pass every binding check")
        return 0
    result = verify_artifact(args.artifact, assert_gate=args.assert_gate)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
