#!/usr/bin/env python3
"""G4 M-A3: replayable Kairyu versus SGLang v0.5.16 comparison.

The module is deliberately an evidence operator, not a Docker supervisor.  One
``run`` invocation measures one already-started fresh server and writes a raw
JSONL shard.  ``assemble`` requires the complete ten-generation matrix: one
preflight generation per arm followed by eight formal generations.  It proves
that every server generation is unique and sequential, then writes canonical
raw JSONL plus a derived manifest.  ``replay`` ignores the manifest entirely;
``verify`` additionally proves that the retained manifest is byte-equivalent
to a fresh raw replay.

The binding workload is one fixed ShareGPT population of 128 simultaneous
requests, exactly 128 completion tokens each, on the same four physical GPUs.
Four paired rounds use K/S, S/K, S/K, K/S fresh-server order.  Acceptance uses
the exact median of the four per-round ratios: completion tok/s/GPU K/S must be
at least one and TTFT p99 K/S must be at most one.  Failed or retried requests
remain evidence and make the gate fail; they are never removed from an
aggregate.

Before formal capture, one short retained preflight confirms the pinned
production choices. ``assemble --preflight-only`` freezes that selection for
the formal run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402

from bench import g2_a6_vllm_bench as a6  # noqa: E402
from bench import g4_ma1_qwen3_235b_nvfp4_bench as ma1  # noqa: E402
from bench import g4_ma1_qwen3_235b_nvfp4_capture as ma1_capture  # noqa: E402
from kairyu.bench.evidence import (  # noqa: E402
    artifact_paths as evidence_artifact_paths,
)
from kairyu.bench.evidence import (  # noqa: E402
    canonical_json,
    sha256_file,
    sha256_json,
    sha256_text,
    verify_replayed_manifest,
)
from kairyu.bench.evidence import (  # noqa: E402
    read_json_object as evidence_read_json_object,
)
from kairyu.bench.evidence import (  # noqa: E402
    read_jsonl as evidence_read_jsonl,
)
from kairyu.bench.evidence import (  # noqa: E402
    replay_manifest as evidence_replay_manifest,
)
from kairyu.bench.evidence import (  # noqa: E402
    strict_json_loads as evidence_strict_json_loads,
)
from kairyu.bench.evidence import (  # noqa: E402
    write_jsonl as evidence_write_jsonl,
)
from kairyu.bench.reporting import atomic_write_json, atomic_write_text  # noqa: E402
from kairyu.engine.tokenizer import HFTokenizer  # noqa: E402

SCHEMA_VERSION = "kairyu.g4.ma3.sglang-comparison.v2"
TRACE_SCHEMA_VERSION = "kairyu.g4.ma3.sharegpt-trace.v2"
SELECTION_SCHEMA_VERSION = "kairyu.g4.ma3.preflight-selection.v1"
CHECKPOINT_CAPTURE_SCHEMA_VERSION = "kairyu.g4.ma3.checkpoint-capture.v1"
LIVE_OBSERVATION_SCHEMA_VERSION = "kairyu.g4.ma3.live-observation.v1"
HTTP_CLIENT_LIFECYCLE_SCHEMA_VERSION = "kairyu.g4.ma3.http-client-lifecycle.v1"
GATE = "G4-M-A3"
RAW_NAME = "g4-ma3-sglang-raw.jsonl"
MANIFEST_NAME = "g4-ma3-sglang-manifest.json"

MODEL = ma1.MODEL
MODEL_REVISION = ma1.MODEL_REVISION
TOKENIZER_SHA256 = next(
    digest for name, _size, digest in ma1.EXPECTED_TOKENIZER_FILES if name == "tokenizer.json"
)
CHECKPOINT_ROOT = "/models/qwen3-235b-nvfp4"
SERVED_MODEL_NAME = CHECKPOINT_ROOT
MODEL_VOLUME_SUBPATH = "qwen3-235b-nvfp4"
FORMAL_ENDPOINT = "http://127.0.0.1:30000"
CAPTURE_SOURCE_ROOT = "/workspace"
CAPTURE_HELPER = "/workspace/bench/g4_ma3_sglang_bench.py"
CAPTURE_PYTHON = "/app/.venv/bin/python"
_CHECKPOINT_HELPER_COMMAND = "_checkpoint-helper"

SGLANG_VERSION = "0.5.16"
SGLANG_TAG = "v0.5.16"
SGLANG_SOURCE_REPOSITORY = "https://github.com/sgl-project/sglang"
SGLANG_TAG_OBJECT = "d21f3c3a10606ba3c7bf43f981496da0a7d620cd"
SGLANG_SOURCE_COMMIT = "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.16"
SGLANG_IMAGE_REPO_DIGEST = (
    "lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c"
)
SGLANG_AMD64_MANIFEST_DIGEST = (
    "sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c"
)

# Keep the final dispatcher choice in one place.  The CLI can override it only
# while preparing the trace/plan; every later shard is bound to that plan.
DEFAULT_SGLANG_MOE_A2A_BACKEND = "none"
SGLANG_FP4_GEMM_BACKEND = "flashinfer_cutlass"
SGLANG_MOE_RUNNER_BACKEND = "flashinfer_cutlass"
SGLANG_FLASHINFER_VERSION = "0.6.14"
SGLANG_TORCH_VERSION = "2.11.0+cu130"
SGLANG_CUDA_VERSION = "13.0"
SGLANG_NCCL_VERSION = "2.28.9"
KAIRYU_MOE_A2A_BACKEND = "direct_nccl_ctypes"
KAIRYU_MOE_RUNNER_BACKEND = "nvfp4_allgather_reduce_scatter"
KAIRYU_FP4_GEMM_BACKEND = "flashinfer_cutlass"

ARMS = ("kairyu", "sglang")
GPU_COUNT = 4
PHYSICAL_GPU_INDICES = (4, 5, 6, 7)
EXPECTED_CAP_ADD = ("SYS_NICE", "SYS_PTRACE")
REPEATS = 4
PAIR_ORDER = (
    ("kairyu", "sglang"),
    ("sglang", "kairyu"),
    ("sglang", "kairyu"),
    ("kairyu", "sglang"),
)
FORMAL_SEQUENCE = tuple((repeat, arm) for repeat, pair in enumerate(PAIR_ORDER) for arm in pair)

KAIRYU_PREFLIGHT_CANDIDATES = ("depth-5",)
SGLANG_PREFLIGHT_CANDIDATES = ("default",)
PREFLIGHT_CANDIDATES = {
    "kairyu": KAIRYU_PREFLIGHT_CANDIDATES,
    "sglang": SGLANG_PREFLIGHT_CANDIDATES,
}
# The production choices are fixed. One retained smoke-sized preflight catches
# a broken launch without spending 18 fresh model loads re-proving a
# non-binding tuning result.
PREFLIGHT_REPEATS = 1
PREFLIGHT_REQUESTS = 32
PREFLIGHT_OUTPUT_TOKENS = 32

SHAREGPT_REQUESTS = 128
SHAREGPT_CONCURRENCY = 128
SHAREGPT_OUTPUT_TOKENS = 128
SERIAL_WARMUP_REQUESTS = 4
SERIAL_WARMUP_OUTPUT_TOKENS = 8
# Attention-DP assigns requests round-robin over four owners.  These global
# bursts therefore exercise every configured local graph bucket
# (1,2,4,8,16,24,32) before measurement.  Reusing A6's global
# 1/2/4/8/16 geometry would warm owner buckets only through 4 and could charge
# four model forwards of a lazy larger-bucket capture to formal TTFT and
# throughput.
GRAPH_WARMUP_BATCH_SIZES = (4, 8, 16, 32, 64, 96, 128)
GRAPH_WARMUP_REQUESTS = sum(GRAPH_WARMUP_BATCH_SIZES)
GRAPH_WARMUP_PROMPT_TOKENS = 64
# Keep every synchronized burst alive through enough steady-decode steps for
# independently arriving HTTP requests to coalesce in the scheduler.  Two
# tokens provide only one decode opportunity and were observed on the target
# box to capture four of seven buckets despite 348 successful warmup requests.
# Sixteen deterministically captured all seven buckets on both depth candidates;
# the live witness below still fails closed before measurement if that ever
# ceases to hold.
GRAPH_WARMUP_OUTPUT_TOKENS = 16
KAIRYU_CUDA_GRAPH_MAX_BATCH = 32
KAIRYU_CUDA_GRAPH_MAX_PAGES = 128
KAIRYU_CUDA_GRAPH_WARMUP_ITERS = 3
KAIRYU_CUDA_GRAPH_BUCKETS = (1, 2, 4, 8, 16, 24, 32)
OPEN_LOOP_POINTS = 5
OPEN_LOOP_REQUESTS_PER_POINT = 32

TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = -1
SEED = 0
IGNORE_EOS = True
N = 1
PAGE_SIZE = 16
KV_CACHE_DTYPE = "bfloat16"
CONFIGURED_PREFILL_TOKENS = 8192
RESOLVED_PREFILL_TOKENS_PER_OWNER = CONFIGURED_PREFILL_TOKENS // GPU_COUNT
SGLANG_MAX_PREFILL_TOKENS_PER_OWNER = RESOLVED_PREFILL_TOKENS_PER_OWNER
MAX_RUNNING_REQUESTS = 256
# Kairyu owns one global page namespace while SGLang owns one namespace per
# attention-DP rank.  Pin the same aggregate capacity and divide SGLang's
# per-owner ``--max-total-tokens`` value, rather than silently giving it 4x the
# cache used by the Kairyu arm.
MAX_TOTAL_TOKENS = 65_536
MAX_TOTAL_TOKENS_PER_OWNER = MAX_TOTAL_TOKENS // GPU_COUNT
DEFAULT_TIMEOUT_S = 600.0

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPO_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")

_KAIRYU_GRAPH_WITNESS_KEYS = frozenset(
    {
        "model",
        "engine_backend",
        "parallelism",
        "expert_parallel_size",
        "attention_placement",
        "attention_output_placement",
        "attention_output_parallel_size",
        "attention_output_partial_dtype",
        "execution_mode",
        "pipeline_depth",
        "decode_mode",
        "kv_cache_dtype",
        "attention_data_parallel_size",
        "attention_tensor_parallel_size",
        "moe_dispatcher",
        "sampling_ownership",
        "kv_cache_ownership",
        "cuda_graph_decode",
        "cuda_graph_buckets",
        "cuda_graph_captures",
        "cuda_graph_replays",
        "cuda_graph_eager_fallbacks",
        "moe_collective_transport",
    }
)
_KAIRYU_TRANSPORT_WITNESS_KEYS = frozenset(
    {
        "selected_backend",
        "fallback_backend",
        "direct_nccl_active",
        "direct_nccl_library",
        "direct_nccl_version",
        "selection_reason",
    }
)


class GateEvidenceError(ValueError):
    """Raised when structural evidence is absent, ambiguous, or inconsistent."""


class _Tokenizer(Protocol):
    def encode(self, text: str) -> tuple[int, ...]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def vocab(self) -> list[str]: ...


class _TrackedAsyncClient:
    """Own one HTTP connection pool and retain its exact outbound request order."""

    def __init__(self, *, role: str, **kwargs: object) -> None:
        if role not in {"warmup", "measurement"}:
            raise ValueError(f"invalid HTTP client role {role!r}")
        self.role = role
        self.created_ns = time.perf_counter_ns()
        self.closed_ns: int | None = None
        self._request_paths: list[str] = []
        self._client = httpx.AsyncClient(**kwargs)

    @property
    def request_count(self) -> int:
        return len(self._request_paths)

    def _record(self, url: str) -> None:
        if self.closed_ns is not None:
            raise RuntimeError(f"{self.role} HTTP client is already closed")
        self._request_paths.append(httpx.URL(url).path)

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        self._record(url)
        return await self._client.get(url, **kwargs)

    def stream(self, method: str, url: str, **kwargs: object) -> object:
        self._record(url)
        return self._client.stream(method, url, **kwargs)

    async def __aenter__(self) -> _TrackedAsyncClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> object:
        try:
            return await self._client.__aexit__(exc_type, exc_value, traceback)
        finally:
            self.closed_ns = time.perf_counter_ns()

    def evidence(self) -> dict[str, object]:
        if self.closed_ns is None:
            raise GateEvidenceError(f"{self.role} HTTP client was not fully closed")
        counts = {
            path: self._request_paths.count(path) for path in sorted(set(self._request_paths))
        }
        return {
            "role": self.role,
            "created_ns": self.created_ns,
            "closed_ns": self.closed_ns,
            "request_count": self.request_count,
            "request_path_counts": counts,
            "request_path_sequence_sha256": sha256_json(self._request_paths),
        }


@dataclass(frozen=True)
class PlannedRequest:
    phase: str
    sequence: int
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    expected_completion_tokens: int
    preflight_repeat: int | None = None
    graph_batch_size: int | None = None
    graph_batch_sequence: int | None = None
    open_loop_point: int | None = None
    rate_millirps: int | None = None

    @property
    def prompt_text_sha256(self) -> str:
        return sha256_text(self.prompt_text)

    @property
    def prompt_token_ids_sha256(self) -> str:
        return sha256_json(list(self.prompt_token_ids))


def strict_json_loads(value: str) -> object:
    return evidence_strict_json_loads(
        value,
        error_type=GateEvidenceError,
    )


def hashed_descriptor(value: object) -> dict[str, object]:
    return {"value": value, "sha256": sha256_json(value)}


def _descriptor_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"value", "sha256"}
        and isinstance(value["sha256"], str)
        and _HEX64.fullmatch(value["sha256"]) is not None
        and value["sha256"] == sha256_json(value["value"])
    )


def _build_graph_warmup_trace(tokenizer: _Tokenizer) -> dict[str, object]:
    """Build prompts for the exact attention-DP owner buckets under test."""

    pieces = a6._stable_token_pieces(  # noqa: SLF001 - shared evidence primitive
        tokenizer,
        required=GRAPH_WARMUP_REQUESTS,
    )
    prompts: list[dict[str, object]] = []
    for sequence, (token_id, text) in enumerate(pieces):
        prompt_text = text * GRAPH_WARMUP_PROMPT_TOKENS
        prompt_token_ids = (token_id,) * GRAPH_WARMUP_PROMPT_TOKENS
        if tokenizer.encode(prompt_text) != prompt_token_ids:
            raise GateEvidenceError(
                "G4 M-A3 graph-warmup text lost its pinned token geometry"
            )
        prompts.append(
            a6._token_row(  # noqa: SLF001 - shared canonical token row
                sequence=sequence,
                prompt_text=prompt_text,
                prompt_token_ids=prompt_token_ids,
            )
        )
    return hashed_descriptor(
        {
            "schema_version": 1,
            "construction": "arm-neutral-stable-token-repeat-v1",
            "tokenizer_sha256": TOKENIZER_SHA256,
            "prompts": prompts,
            "batch_sizes": list(GRAPH_WARMUP_BATCH_SIZES),
            "request_count": GRAPH_WARMUP_REQUESTS,
            "completion_tokens": GRAPH_WARMUP_OUTPUT_TOKENS,
        }
    )


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
        and raw.get("tokenizer_sha256") == TOKENIZER_SHA256
        and raw.get("batch_sizes") == list(GRAPH_WARMUP_BATCH_SIZES)
        and raw.get("request_count") == GRAPH_WARMUP_REQUESTS
        and raw.get("completion_tokens") == GRAPH_WARMUP_OUTPUT_TOKENS
        and a6._token_rows_valid(  # noqa: SLF001 - shared canonical row validator
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


def _graph_warmup_disjoint(value: object, trace: object) -> bool:
    if (
        not _graph_warmup_trace_valid(value)
        or not isinstance(value, dict)
        or not _descriptor_valid(trace)
        or not isinstance(trace, dict)
    ):
        return False
    graph_raw = value["value"]
    trace_raw = trace["value"]
    if not isinstance(graph_raw, dict) or not isinstance(trace_raw, dict):
        return False
    prompts = graph_raw.get("prompts")
    if not isinstance(prompts, list):
        return False
    graph_hashes = {
        row.get("prompt_token_ids_sha256")
        for row in prompts
        if isinstance(row, dict)
    }
    retained_hashes: set[object] = set()
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


def _fraction_record(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _fraction_from_record(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator", "decimal"}:
        raise GateEvidenceError("invalid exact-fraction record")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise GateEvidenceError("invalid exact-fraction components")
    result = Fraction(numerator, denominator)
    if not isinstance(value["decimal"], (int, float)) or float(value["decimal"]) != float(result):
        raise GateEvidenceError("exact-fraction decimal does not match")
    return result


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise GateEvidenceError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values or not 0 < quantile <= 1:
        raise GateEvidenceError("nearest-rank requires values and 0 < q <= 1")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _read_json_object(path: Path) -> dict[str, object]:
    return evidence_read_json_object(path, error_type=GateEvidenceError)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    evidence_write_jsonl(path, rows)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return evidence_read_jsonl(path, error_type=GateEvidenceError)


def request_config(output_tokens: int) -> dict[str, object]:
    return {
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "n": N,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": IGNORE_EOS,
    }


def expected_checkpoint() -> dict[str, object]:
    """Return the complete immutable Qwen3-235B NVFP4 checkpoint identity."""

    weights = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in ma1.EXPECTED_WEIGHT_FILES
    ]
    tokenizer = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in ma1.EXPECTED_TOKENIZER_FILES
    ]
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "config_sha256": ma1.EXPECTED_CONFIG_SHA256,
        "index_sha256": ma1.EXPECTED_INDEX_SHA256,
        "hf_quant_config_sha256": ma1.EXPECTED_HF_QUANT_CONFIG_SHA256,
        "index_total_size": ma1.EXPECTED_INDEX_TOTAL_SIZE,
        "tensor_count": ma1.EXPECTED_TENSOR_COUNT,
        "weight_shards": weights,
        "weights_rollup_sha256": sha256_json(weights),
        "tokenizer_files": tokenizer,
        "tokenizer_rollup_sha256": sha256_json(tokenizer),
        "architecture": dict(ma1.EXPECTED_ARCHITECTURE),
        "quantization": dict(ma1.EXPECTED_QUANTIZATION),
    }


def _checkpoint_capture_container_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "id",
        "image_repo_digest",
        "image_config_id",
        "source_mount",
        "model_mount",
        "network_mode",
        "gpu_device_requests",
        "resolved_argv",
    }:
        return False
    source_mount = value["source_mount"]
    model_mount = value["model_mount"]
    return (
        isinstance(value["id"], str)
        and _CONTAINER_ID.fullmatch(value["id"]) is not None
        and isinstance(value["image_repo_digest"], str)
        and _REPO_DIGEST.fullmatch(value["image_repo_digest"]) is not None
        and isinstance(value["image_config_id"], str)
        and _IMAGE_ID.fullmatch(value["image_config_id"]) is not None
        and isinstance(source_mount, dict)
        and set(source_mount) == {"source", "destination", "read_only"}
        and isinstance(source_mount["source"], str)
        and bool(source_mount["source"])
        and source_mount["destination"] == CAPTURE_SOURCE_ROOT
        and source_mount["read_only"] is True
        and isinstance(model_mount, dict)
        and set(model_mount) == {"source", "destination", "read_only"}
        and isinstance(model_mount["source"], str)
        and model_mount["source"].startswith("volume:")
        and model_mount["source"].endswith(f"/{MODEL_VOLUME_SUBPATH}")
        and model_mount["destination"] == CHECKPOINT_ROOT
        and model_mount["read_only"] is True
        and value["network_mode"] == "none"
        and value["gpu_device_requests"] == []
        and value["resolved_argv"] == ["sleep", "infinity"]
    )


def _checkpoint_capture_value_valid(value: object, *, phase: str) -> bool:
    if phase not in {"start", "end"} or not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "phase",
        "capture_started_ns",
        "capture_completed_ns",
        "capture_container",
        "source",
        "checkpoint_volume",
        "checkpoint",
    }:
        return False
    container = value["capture_container"]
    source = value["source"]
    volume = value["checkpoint_volume"]
    checkpoint = value["checkpoint"]
    if (
        not _checkpoint_capture_container_valid(container)
        or not _source_valid(source, arm="kairyu")
        or not _checkpoint_volume_valid(volume)
        or not isinstance(container, dict)
        or not isinstance(volume, dict)
    ):
        return False
    model_mount = container["model_mount"]
    return (
        value["schema_version"] == CHECKPOINT_CAPTURE_SCHEMA_VERSION
        and value["type"] == "checkpoint_capture"
        and value["phase"] == phase
        and type(value["capture_started_ns"]) is int
        and type(value["capture_completed_ns"]) is int
        and 0 < value["capture_started_ns"] < value["capture_completed_ns"]
        and model_mount["source"]
        == f"volume:{volume['name']}/{volume['subpath']}"
        and _descriptor_valid(checkpoint)
        and isinstance(checkpoint, dict)
        and checkpoint["value"] == expected_checkpoint()
        and ma1._checkpoint_valid(checkpoint["value"])
    )


def load_checkpoint_capture(path: Path, *, phase: str) -> dict[str, object]:
    descriptor = hashed_descriptor(_read_json_object(path))
    if not _descriptor_valid(descriptor) or not _checkpoint_capture_value_valid(
        descriptor["value"], phase=phase
    ):
        raise GateEvidenceError(
            f"{path} does not satisfy the G4 M-A3 {phase} checkpoint capture schema"
        )
    return descriptor


def capture_checkpoint(
    output: Path,
    *,
    phase: str,
    container_name: str,
    source_root: Path,
) -> dict[str, object]:
    """Hash the observed read-only model volume inside one capture container."""

    if phase not in {"start", "end"}:
        raise GateEvidenceError("checkpoint capture phase must be start or end")
    if not container_name or not source_root.is_dir():
        raise GateEvidenceError("checkpoint capture requires a container and source root")
    capture_started_ns = time.perf_counter_ns()

    container = _capture_single_object(
        ("docker", "inspect", container_name),
        label="checkpoint capture container inspect",
    )
    config = container.get("Config")
    host_config = container.get("HostConfig")
    state = container.get("State")
    mounts = container.get("Mounts")
    if not all(isinstance(item, dict) for item in (config, host_config, state)) or not isinstance(
        mounts, list
    ):
        raise GateEvidenceError("checkpoint capture inspect is incomplete")
    assert isinstance(config, dict) and isinstance(host_config, dict) and isinstance(state, dict)
    if state.get("Running") is not True:
        raise GateEvidenceError("checkpoint capture container is not running")
    if not (
        host_config.get("NetworkMode") == "none"
        and not host_config.get("DeviceRequests")
        and not host_config.get("PortBindings")
        and container.get("Path") == "sleep"
        and container.get("Args") == ["infinity"]
    ):
        raise GateEvidenceError("checkpoint capture container is not dedicated and GPU-less")
    container_id = container.get("Id")
    image_config_id = container.get("Image")
    image_repo_digest = config.get("Image")
    if (
        not isinstance(container_id, str)
        or _CONTAINER_ID.fullmatch(container_id) is None
        or not isinstance(image_config_id, str)
        or _IMAGE_ID.fullmatch(image_config_id) is None
        or not isinstance(image_repo_digest, str)
        or _REPO_DIGEST.fullmatch(image_repo_digest) is None
    ):
        raise GateEvidenceError("checkpoint capture container/image identity is not immutable")
    image = _capture_single_object(
        ("docker", "image", "inspect", image_repo_digest),
        label="checkpoint capture image inspect",
    )
    if image.get("Id") != image_config_id or image_repo_digest not in (
        image.get("RepoDigests") or ()
    ):
        raise GateEvidenceError("checkpoint capture image does not match its RepoDigest")

    resolved_source = source_root.resolve()
    source_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == CAPTURE_SOURCE_ROOT
    ]
    model_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == CHECKPOINT_ROOT
    ]
    if len(source_mounts) != 1 or len(model_mounts) != 1:
        raise GateEvidenceError("checkpoint capture requires singular source/model mounts")
    source_mount = source_mounts[0]
    model_mount = model_mounts[0]
    model_volume = model_mount.get("Name")
    if not (
        source_mount.get("Type") == "bind"
        and isinstance(source_mount.get("Source"), str)
        and Path(str(source_mount["Source"])).resolve() == resolved_source
        and source_mount.get("RW") is False
        and config.get("WorkingDir") == CAPTURE_SOURCE_ROOT
        and model_mount.get("Type") == "volume"
        and isinstance(model_volume, str)
        and bool(model_volume)
        and model_mount.get("RW") is False
        and model_mount.get("Subpath") in (None, MODEL_VOLUME_SUBPATH)
    ):
        raise GateEvidenceError("checkpoint capture source/model live mounts are invalid")
    configured_mounts = host_config.get("Mounts")
    configured_source = [
        mount
        for mount in configured_mounts or ()
        if isinstance(mount, dict) and mount.get("Target") == CAPTURE_SOURCE_ROOT
    ]
    configured_model = [
        mount
        for mount in configured_mounts or ()
        if isinstance(mount, dict) and mount.get("Target") == CHECKPOINT_ROOT
    ]
    if len(configured_source) != 1 or len(configured_model) != 1:
        raise GateEvidenceError("checkpoint capture HostConfig mounts are incomplete")
    source_config = configured_source[0]
    model_config = configured_model[0]
    volume_options = model_config.get("VolumeOptions")
    if not (
        source_config.get("Type") == "bind"
        and Path(str(source_config.get("Source"))).resolve() == resolved_source
        and source_config.get("ReadOnly") is True
        and model_config.get("Type") == "volume"
        and model_config.get("Source") == model_volume
        and model_config.get("ReadOnly") is True
        and isinstance(volume_options, dict)
        and volume_options.get("Subpath") == MODEL_VOLUME_SUBPATH
    ):
        raise GateEvidenceError("checkpoint capture configured mounts are invalid")

    commit = _capture_command(("git", "-C", str(resolved_source), "rev-parse", "HEAD")).strip()
    status = _capture_command(
        (
            "git",
            "-C",
            str(resolved_source),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    )
    tracked = _capture_command(
        (
            "git",
            "-C",
            str(resolved_source),
            "ls-files",
            "--error-unmatch",
            "bench/g4_ma3_sglang_bench.py",
            "bench/g4_ma1_qwen3_235b_nvfp4_capture.py",
        )
    ).splitlines()
    if (
        _HEX40.fullmatch(commit) is None
        or status
        or tracked
        != [
            "bench/g4_ma1_qwen3_235b_nvfp4_capture.py",
            "bench/g4_ma3_sglang_bench.py",
        ]
    ):
        raise GateEvidenceError("checkpoint capture source is not one clean tracked commit")
    checkpoint_volume = _capture_checkpoint_volume(
        container_id=container_id,
        model_volume=model_volume,
    )
    try:
        helper_output = _capture_command(
            (
                "docker",
                "exec",
                container_id,
                CAPTURE_PYTHON,
                CAPTURE_HELPER,
                _CHECKPOINT_HELPER_COMMAND,
            )
        )
        checkpoint = strict_json_loads(helper_output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError, OSError, ValueError) as error:
        raise GateEvidenceError(f"checkpoint byte capture failed: {error}") from error
    capture_completed_ns = time.perf_counter_ns()
    if checkpoint != expected_checkpoint():
        raise GateEvidenceError("mounted checkpoint differs from the pinned M-A1 identity")
    value: dict[str, object] = {
        "schema_version": CHECKPOINT_CAPTURE_SCHEMA_VERSION,
        "type": "checkpoint_capture",
        "phase": phase,
        "capture_started_ns": capture_started_ns,
        "capture_completed_ns": capture_completed_ns,
        "capture_container": {
            "id": container_id,
            "image_repo_digest": image_repo_digest,
            "image_config_id": image_config_id,
            "source_mount": {
                "source": str(resolved_source),
                "destination": CAPTURE_SOURCE_ROOT,
                "read_only": True,
            },
            "model_mount": {
                "source": f"volume:{model_volume}/{MODEL_VOLUME_SUBPATH}",
                "destination": CHECKPOINT_ROOT,
                "read_only": True,
            },
            "network_mode": "none",
            "gpu_device_requests": [],
            "resolved_argv": ["sleep", "infinity"],
        },
        "source": {
            "repository": "https://github.com/ytworks/kairyu",
            "commit": commit,
            "clean": True,
        },
        "checkpoint_volume": checkpoint_volume,
        "checkpoint": hashed_descriptor(checkpoint),
    }
    if not _checkpoint_capture_value_valid(value, phase=phase):
        raise GateEvidenceError("captured checkpoint does not satisfy the strict schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value, indent=2, sort_keys=True, ensure_ascii=False)
    load_checkpoint_capture(output, phase=phase)
    return value


def expected_sglang_argv(
    *,
    overlap_mode: str,
    moe_a2a_backend: str = DEFAULT_SGLANG_MOE_A2A_BACKEND,
) -> list[str]:
    """Return the single pinned v0.5.16 launch contract."""

    if overlap_mode not in SGLANG_PREFLIGHT_CANDIDATES:
        raise ValueError(f"unknown SGLang overlap mode {overlap_mode!r}")
    if moe_a2a_backend != DEFAULT_SGLANG_MOE_A2A_BACKEND:
        raise ValueError(
            f"formal SGLang MoE A2A backend is pinned to {DEFAULT_SGLANG_MOE_A2A_BACKEND!r}"
        )
    result = [
        "sglang",
        "serve",
        "--model-path",
        CHECKPOINT_ROOT,
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--tp-size",
        "4",
        "--dp-size",
        "4",
        "--ep-size",
        "4",
        "--enable-dp-attention",
        "--load-balance-method",
        "round_robin",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        KV_CACHE_DTYPE,
        "--fp4-gemm-backend",
        SGLANG_FP4_GEMM_BACKEND,
        "--moe-a2a-backend",
        moe_a2a_backend,
        "--moe-runner-backend",
        SGLANG_MOE_RUNNER_BACKEND,
        "--page-size",
        str(PAGE_SIZE),
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
        "--max-total-tokens",
        str(MAX_TOTAL_TOKENS_PER_OWNER),
        "--chunked-prefill-size",
        str(CONFIGURED_PREFILL_TOKENS),
        "--max-prefill-tokens",
        str(SGLANG_MAX_PREFILL_TOKENS_PER_OWNER),
        "--schedule-policy",
        "fcfs",
        "--log-level-http",
        "warning",
        "--cuda-graph-max-bs-decode",
        str(KAIRYU_CUDA_GRAPH_MAX_BATCH),
        "--cuda-graph-backend-prefill",
        "disabled",
    ]
    if overlap_mode == "single-batch":
        result.append("--enable-single-batch-overlap")
    elif overlap_mode == "two-batch":
        result.append("--enable-two-batch-overlap")
    return result


def benchmark_config(
    dataset_sha256: str,
    *,
    moe_a2a_backend: str = DEFAULT_SGLANG_MOE_A2A_BACKEND,
) -> dict[str, object]:
    if _HEX64.fullmatch(dataset_sha256) is None:
        raise ValueError("ShareGPT dataset SHA-256 must be lowercase hex")
    if moe_a2a_backend != DEFAULT_SGLANG_MOE_A2A_BACKEND:
        raise ValueError(
            f"formal SGLang MoE A2A backend is pinned to {DEFAULT_SGLANG_MOE_A2A_BACKEND!r}"
        )
    return {
        "gate": GATE,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "served_model_name": SERVED_MODEL_NAME,
        "checkpoint_contract": "all 27 weights and tokenizer/config metadata from G4 M-A1",
        "gpu_count": GPU_COUNT,
        "physical_gpu_indices": list(PHYSICAL_GPU_INDICES),
        "arms": list(ARMS),
        "pair_order": [list(pair) for pair in PAIR_ORDER],
        "sharegpt": {
            "repository": a6.SHAREGPT_DATASET_REPO,
            "revision": a6.SHAREGPT_DATASET_REVISION,
            "file": a6.SHAREGPT_DATASET_FILE,
            "sha256": dataset_sha256,
            "selection_seed": a6.SHAREGPT_DATASET_SEED,
            "requests": SHAREGPT_REQUESTS,
            "concurrency": SHAREGPT_CONCURRENCY,
            "prompt_tokens_min": a6.SHAREGPT_MIN_PROMPT_TOKENS,
            "prompt_tokens_max": a6.SHAREGPT_MAX_PROMPT_TOKENS,
            "completion_tokens": SHAREGPT_OUTPUT_TOKENS,
            "request": request_config(SHAREGPT_OUTPUT_TOKENS),
        },
        "warmup": {
            "serial_requests": SERIAL_WARMUP_REQUESTS,
            "serial_completion_tokens": SERIAL_WARMUP_OUTPUT_TOKENS,
            "graph_batch_sizes": list(GRAPH_WARMUP_BATCH_SIZES),
            "graph_completion_tokens": GRAPH_WARMUP_OUTPUT_TOKENS,
            "retained": True,
            "excluded_from_metrics": True,
        },
        "http_client_lifecycle": {
            "applies_to_arms": list(ARMS),
            "applies_to_phases": ["preflight", "formal"],
            "warmup_client_request_order": [
                "model_probe",
                "serial_warmup",
                "graph_warmup",
                "kairyu_graph_start_witness_if_kairyu",
            ],
            "warmup_client_fully_closed_before_measurement_client_created": True,
            "measurement_client": "distinct fresh connection pool",
            "measurement_client_requests_before_synchronized_group": 0,
            "measurement_client_request_order": [
                "one synchronized preflight_or_formal_group",
                "kairyu_graph_end_witness_if_kairyu",
            ],
            "measurement_client_fully_closed_after_witness": True,
        },
        "matched": {
            "maximum_workload_tokens": a6.SHAREGPT_MAX_PROMPT_TOKENS + SHAREGPT_OUTPUT_TOKENS,
            "configured_chunked_prefill_tokens": CONFIGURED_PREFILL_TOKENS,
            "resolved_chunked_prefill_tokens_per_owner": (RESOLVED_PREFILL_TOKENS_PER_OWNER),
            "kairyu_configured_max_prefill_tokens": CONFIGURED_PREFILL_TOKENS,
            "sglang_configured_max_prefill_tokens_per_owner": (
                SGLANG_MAX_PREFILL_TOKENS_PER_OWNER
            ),
            "resolved_max_prefill_tokens_per_owner": (RESOLVED_PREFILL_TOKENS_PER_OWNER),
            "max_running_requests": MAX_RUNNING_REQUESTS,
            "page_size": PAGE_SIZE,
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "max_total_tokens_per_owner": MAX_TOTAL_TOKENS_PER_OWNER,
            "kv_cache_dtype": KV_CACHE_DTYPE,
            "prefix_cache": True,
            "scheduler_policy": "fcfs",
            "speculative_decoding": False,
            "access_log": False,
        },
        "preflight": {
            "kairyu_candidates": list(KAIRYU_PREFLIGHT_CANDIDATES),
            "sglang_candidates": list(SGLANG_PREFLIGHT_CANDIDATES),
            "repeats_per_candidate": PREFLIGHT_REPEATS,
            "requests_per_repeat": PREFLIGHT_REQUESTS,
            "completion_tokens": PREFLIGHT_OUTPUT_TOKENS,
            "selection": "highest exact median completion tok/s; declared order breaks exact ties",
        },
        "sglang": {
            "version": SGLANG_VERSION,
            "source_repository": SGLANG_SOURCE_REPOSITORY,
            "source_tag": SGLANG_TAG,
            "tag_object": SGLANG_TAG_OBJECT,
            "source_commit": SGLANG_SOURCE_COMMIT,
            "image_tag_diagnostic": SGLANG_IMAGE_TAG,
            "image_repo_digest": SGLANG_IMAGE_REPO_DIGEST,
            "amd64_manifest_digest": SGLANG_AMD64_MANIFEST_DIGEST,
            "moe_a2a_backend": moe_a2a_backend,
            "fp4_gemm_backend": SGLANG_FP4_GEMM_BACKEND,
            "runtime_versions": {
                "cuda": SGLANG_CUDA_VERSION,
                "nccl": SGLANG_NCCL_VERSION,
                "torch": SGLANG_TORCH_VERSION,
                "flashinfer": SGLANG_FLASHINFER_VERSION,
            },
            "argv_by_overlap_mode": {
                mode: expected_sglang_argv(
                    overlap_mode=mode,
                    moe_a2a_backend=moe_a2a_backend,
                )
                for mode in SGLANG_PREFLIGHT_CANDIDATES
            },
            "docker": {
                "gpu_device_indices": list(PHYSICAL_GPU_INDICES),
                "ipc_mode": "host",
                "shm_size_bytes": 32 * 1024**3,
                "cap_add": list(EXPECTED_CAP_ADD),
                "model_mount_destination": CHECKPOINT_ROOT,
                "model_mount_read_only": True,
            },
        },
        "thresholds": {
            "paired_median_completion_tok_s_per_gpu_kairyu_over_sglang_min": 1.0,
            "paired_median_ttft_p99_kairyu_over_sglang_max": 1.0,
        },
        "aggregation": {
            "throughput": (
                "successful completion tokens / first-start-to-last-terminal span / 4 GPUs"
            ),
            "ttft_p99": "nearest-rank over exactly 128 requests per scenario",
            "paired_ratio": "exact Fraction before the exact median of four rounds",
            "outlier_removal": False,
            "round_before_gate": False,
            "failure_or_retry_exclusion": False,
        },
        "m_a1_status": "formal_fail_independent",
    }


def write_trace_bundle(
    output: Path,
    *,
    dataset_path: Path,
    tokenizer_path: Path,
    dataset_sha256: str = a6.SHAREGPT_DATASET_SHA256,
    moe_a2a_backend: str = DEFAULT_SGLANG_MOE_A2A_BACKEND,
) -> dict[str, object]:
    tokenizer_file = (
        tokenizer_path if tokenizer_path.is_file() else tokenizer_path / "tokenizer.json"
    )
    if sha256_file(tokenizer_file) != TOKENIZER_SHA256:
        raise GateEvidenceError("tokenizer.json does not match Qwen3-235B NVFP4")
    tokenizer = HFTokenizer(tokenizer_path)
    trace = a6.build_sharegpt_trace(
        dataset_path,
        tokenizer,
        expected_dataset_sha256=dataset_sha256,
        tokenizer_sha256=TOKENIZER_SHA256,
    )
    graph_warmup = _build_graph_warmup_trace(tokenizer)
    value = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "benchmark": benchmark_config(dataset_sha256, moe_a2a_backend=moe_a2a_backend),
        "trace": trace,
        "graph_warmup": graph_warmup,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, canonical_json(value) + "\n", encoding="utf-8")
    return load_trace_bundle(output, dataset_sha256=dataset_sha256)


def load_trace_bundle(
    path: Path,
    *,
    dataset_sha256: str = a6.SHAREGPT_DATASET_SHA256,
) -> dict[str, object]:
    value = _read_json_object(path)
    benchmark = value.get("benchmark")
    sglang = benchmark.get("sglang") if isinstance(benchmark, dict) else None
    a2a = sglang.get("moe_a2a_backend") if isinstance(sglang, dict) else None
    if not isinstance(a2a, str):
        raise GateEvidenceError("trace bundle does not bind the SGLang A2A backend")
    if not (
        set(value) == {"schema_version", "benchmark", "trace", "graph_warmup"}
        and value["schema_version"] == TRACE_SCHEMA_VERSION
        and benchmark == benchmark_config(dataset_sha256, moe_a2a_backend=a2a)
        and a6._trace_valid(value["trace"], workload="sharegpt", dataset_sha256=dataset_sha256)
        and _graph_warmup_trace_valid(value["graph_warmup"])
        and _graph_warmup_disjoint(value["graph_warmup"], value["trace"])
    ):
        raise GateEvidenceError("trace bundle does not satisfy the G4 M-A3 contract")
    return value


def _gpu_inventory_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) != GPU_COUNT:
        return False
    expected_keys = {
        "visible_index",
        "physical_index",
        "uuid",
        "pci_bus_id",
        "name",
        "total_memory_bytes",
    }
    return (
        all(isinstance(row, dict) and set(row) == expected_keys for row in value)
        and [row["visible_index"] for row in value] == list(range(GPU_COUNT))
        and [row["physical_index"] for row in value] == list(PHYSICAL_GPU_INDICES)
        and all(
            isinstance(row["uuid"], str)
            and row["uuid"].startswith("GPU-")
            and isinstance(row["pci_bus_id"], str)
            and row["pci_bus_id"]
            and isinstance(row["name"], str)
            and row["name"]
            and type(row["total_memory_bytes"]) is int
            and row["total_memory_bytes"] > 0
            for row in value
        )
        and len({row["uuid"] for row in value}) == GPU_COUNT
        and len({row["pci_bus_id"] for row in value}) == GPU_COUNT
    )


def _source_valid(value: object, *, arm: str) -> bool:
    if not isinstance(value, dict):
        return False
    if arm == "sglang":
        return value == {
            "repository": SGLANG_SOURCE_REPOSITORY,
            "version": SGLANG_VERSION,
            "tag": SGLANG_TAG,
            "tag_object": SGLANG_TAG_OBJECT,
            "commit": SGLANG_SOURCE_COMMIT,
            "clean": True,
        }
    return (
        arm == "kairyu"
        and set(value) == {"repository", "commit", "clean"}
        and value["repository"] == "https://github.com/ytworks/kairyu"
        and isinstance(value["commit"], str)
        and _HEX40.fullmatch(value["commit"]) is not None
        and value["clean"] is True
    )


def _container_valid(value: object, *, arm: str) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "id",
        "image_repo_digest",
        "image_platform_digest",
        "image_config_id",
        "model_mount",
        "gpu_device_indices",
        "ipc_mode",
        "shm_size_bytes",
        "cap_add",
    }:
        return False
    mount = value["model_mount"]
    common = (
        isinstance(value["id"], str)
        and _CONTAINER_ID.fullmatch(value["id"]) is not None
        and isinstance(value["image_repo_digest"], str)
        and _REPO_DIGEST.fullmatch(value["image_repo_digest"]) is not None
        and isinstance(value["image_platform_digest"], str)
        and _IMAGE_ID.fullmatch(value["image_platform_digest"]) is not None
        and isinstance(value["image_config_id"], str)
        and _IMAGE_ID.fullmatch(value["image_config_id"]) is not None
        and isinstance(mount, dict)
        and set(mount) == {"source", "destination", "read_only"}
        and isinstance(mount["source"], str)
        and mount["source"]
        and mount["destination"] == CHECKPOINT_ROOT
        and mount["read_only"] is True
        and value["gpu_device_indices"] == list(PHYSICAL_GPU_INDICES)
        and value["ipc_mode"] == "host"
        and value["shm_size_bytes"] == 32 * 1024**3
        and isinstance(value["cap_add"], list)
        and len(value["cap_add"]) == 2
        and set(value["cap_add"]) == set(EXPECTED_CAP_ADD)
    )
    if not common:
        return False
    if arm == "sglang":
        return (
            value["image_repo_digest"] == SGLANG_IMAGE_REPO_DIGEST
            and value["image_platform_digest"] == SGLANG_AMD64_MANIFEST_DIGEST
        )
    return arm == "kairyu"


def _environment_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "host_id",
            "driver_version",
            "cuda_version",
            "nccl_version",
            "torch_version",
            "flashinfer_version",
            "engine_name",
            "engine_version",
            "gpu_jobs_exclusive",
        }
        and all(
            isinstance(value[key], str) and bool(value[key])
            for key in (
                "host_id",
                "driver_version",
                "cuda_version",
                "nccl_version",
                "torch_version",
                "flashinfer_version",
                "engine_name",
                "engine_version",
            )
        )
        and value["gpu_jobs_exclusive"] is True
    )


def _arm_environment_valid(value: object, *, arm: str) -> bool:
    if not _environment_valid(value) or not isinstance(value, dict):
        return False
    if arm == "kairyu":
        return value["engine_name"] == "kairyu"
    return (
        arm == "sglang"
        and value["cuda_version"] == SGLANG_CUDA_VERSION
        and value["nccl_version"] == SGLANG_NCCL_VERSION
        and value["torch_version"] == SGLANG_TORCH_VERSION
        and value["flashinfer_version"] == SGLANG_FLASHINFER_VERSION
        and value["engine_name"] == "sglang"
        and value["engine_version"] == SGLANG_VERSION
    )


def _runtime_valid(
    value: object,
    *,
    arm: str,
    candidate: str,
    moe_a2a_backend: str,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "served_model_name",
        "world_size",
        "tensor_parallel_size",
        "data_parallel_size",
        "expert_parallel_size",
        "attention_data_parallel",
        "request_owned_attention",
        "kv_cache_owner",
        "sampling_owner",
        "moe_a2a_backend",
        "moe_runner_backend",
        "pipeline_depth",
        "overlap_mode",
        "prefill_cuda_graph",
        "decode_cuda_graph",
        "cuda_graph_max_batch",
        "cuda_graph_max_pages",
        "cuda_graph_warmup_iters",
        "cuda_graph_buckets",
        "max_running_requests",
        "configured_chunked_prefill_tokens",
        "resolved_chunked_prefill_tokens_per_owner",
        "configured_max_prefill_tokens",
        "resolved_max_prefill_tokens_per_owner",
        "page_size",
        "max_total_tokens",
        "max_total_tokens_per_owner",
        "kv_cache_dtype",
        "prefix_cache",
        "scheduler_policy",
        "speculative_decoding",
        "access_log",
        "resolved_argv",
    }:
        return False
    common = (
        value["served_model_name"] == SERVED_MODEL_NAME
        and value["world_size"] == GPU_COUNT
        and value["data_parallel_size"] == GPU_COUNT
        and value["expert_parallel_size"] == GPU_COUNT
        and value["attention_data_parallel"] is True
        and value["request_owned_attention"] is True
        and value["kv_cache_owner"] == "request-owner"
        and value["sampling_owner"] == "request-owner"
        and isinstance(value["moe_a2a_backend"], str)
        and bool(value["moe_a2a_backend"])
        and isinstance(value["moe_runner_backend"], str)
        and bool(value["moe_runner_backend"])
        and value["prefill_cuda_graph"] == "disabled"
        and type(value["decode_cuda_graph"]) is bool
        and value["max_running_requests"] == MAX_RUNNING_REQUESTS
        and value["configured_chunked_prefill_tokens"] == CONFIGURED_PREFILL_TOKENS
        and value["resolved_chunked_prefill_tokens_per_owner"] == RESOLVED_PREFILL_TOKENS_PER_OWNER
        and value["resolved_max_prefill_tokens_per_owner"] == RESOLVED_PREFILL_TOKENS_PER_OWNER
        and value["page_size"] == PAGE_SIZE
        and value["max_total_tokens"] == MAX_TOTAL_TOKENS
        and value["max_total_tokens_per_owner"] == MAX_TOTAL_TOKENS_PER_OWNER
        and value["kv_cache_dtype"] == KV_CACHE_DTYPE
        and value["prefix_cache"] is True
        and value["scheduler_policy"] == "fcfs"
        and value["speculative_decoding"] is False
        and value["access_log"] is False
        and isinstance(value["resolved_argv"], list)
        and all(isinstance(item, str) for item in value["resolved_argv"])
    )
    if not common:
        return False
    if arm == "kairyu":
        depth = int(candidate.removeprefix("depth-")) if candidate.startswith("depth-") else None
        return (
            candidate in KAIRYU_PREFLIGHT_CANDIDATES
            and value["tensor_parallel_size"] == 1
            and value["moe_a2a_backend"] == KAIRYU_MOE_A2A_BACKEND
            and value["moe_runner_backend"] == KAIRYU_MOE_RUNNER_BACKEND
            and value["configured_max_prefill_tokens"] == CONFIGURED_PREFILL_TOKENS
            and value["decode_cuda_graph"] is True
            and value["cuda_graph_max_batch"] == KAIRYU_CUDA_GRAPH_MAX_BATCH
            and value["cuda_graph_max_pages"] == KAIRYU_CUDA_GRAPH_MAX_PAGES
            and value["cuda_graph_warmup_iters"] == KAIRYU_CUDA_GRAPH_WARMUP_ITERS
            and value["cuda_graph_buckets"] == list(KAIRYU_CUDA_GRAPH_BUCKETS)
            and value["pipeline_depth"] == depth
            and value["overlap_mode"] is None
            and _kairyu_resolved_argv_valid(value["resolved_argv"], pipeline_depth=depth)
        )
    return (
        arm == "sglang"
        and candidate in SGLANG_PREFLIGHT_CANDIDATES
        and value["tensor_parallel_size"] == GPU_COUNT
        and value["decode_cuda_graph"] is True
        and value["cuda_graph_max_batch"] is None
        and value["cuda_graph_max_pages"] is None
        and value["cuda_graph_warmup_iters"] is None
        and value["cuda_graph_buckets"] is None
        and value["pipeline_depth"] is None
        and value["overlap_mode"] == candidate
        and value["moe_a2a_backend"] == moe_a2a_backend
        and value["moe_runner_backend"] == SGLANG_MOE_RUNNER_BACKEND
        and value["configured_max_prefill_tokens"]
        == SGLANG_MAX_PREFILL_TOKENS_PER_OWNER
        and value["resolved_argv"]
        == expected_sglang_argv(
            overlap_mode=candidate,
            moe_a2a_backend=moe_a2a_backend,
        )
    )


def _kairyu_resolved_argv_valid(value: object, *, pipeline_depth: object) -> bool:
    """Validate the dedicated formal launcher without pinning its install root."""

    if (
        not isinstance(value, list)
        or len(value) != 8
        or type(pipeline_depth) is not int
        or not all(isinstance(item, str) for item in value)
    ):
        return False
    interpreter = Path(value[0]).name
    launcher = value[1]
    return (
        re.fullmatch(r"python(?:3(?:\.\d+)?)?", interpreter) is not None
        and (
            launcher == "bench/g4_ma3_kairyu_server.py"
            or launcher.endswith("/bench/g4_ma3_kairyu_server.py")
        )
        and value[2:]
        == [
            "--pipeline-depth",
            str(pipeline_depth),
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
        ]
    )


def _provenance_checkpoint_binding_valid(value: Mapping[str, object]) -> bool:
    capture = value.get("checkpoint_capture_start")
    container = value.get("container")
    volume = value.get("checkpoint_volume")
    if (
        not _descriptor_valid(capture)
        or not isinstance(capture, dict)
        or not _checkpoint_capture_value_valid(capture.get("value"), phase="start")
        or not isinstance(capture.get("value"), dict)
        or not isinstance(container, dict)
        or not isinstance(container.get("model_mount"), dict)
        or not _checkpoint_volume_valid(volume)
        or not isinstance(volume, dict)
    ):
        return False
    capture_value = capture["value"]
    checkpoint = capture_value["checkpoint"]
    capture_container = capture_value["capture_container"]
    if not isinstance(capture_container, dict):
        return False
    mount = capture_container["model_mount"]
    container_mount = container["model_mount"]
    volume_source = f"volume:{volume['name']}/{volume['subpath']}"
    return (
        isinstance(checkpoint, dict)
        and checkpoint.get("value") == value.get("checkpoint")
        and capture_value["checkpoint_volume"] == volume
        and mount == container_mount
        and mount.get("source") == volume_source
        and container_mount.get("source") == volume_source
        and capture_value["capture_completed_ns"] < value.get("server_started_ns", 0)
    )


def _provenance_value_valid(
    value: object,
    *,
    arm: str,
    candidate: str,
    moe_a2a_backend: str,
) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "arm",
            "server_generation_id",
            "server_started_ns",
            "container",
            "source",
            "checkpoint",
            "checkpoint_capture_start",
            "checkpoint_volume",
            "gpus",
            "environment",
            "runtime",
            "runtime_witness",
        }
        and value["schema_version"] == SCHEMA_VERSION
        and value["arm"] == arm
        and isinstance(value["server_generation_id"], str)
        and bool(value["server_generation_id"])
        and type(value["server_started_ns"]) is int
        and value["server_started_ns"] > 0
        and _container_valid(value["container"], arm=arm)
        and _source_valid(value["source"], arm=arm)
        and ma1._checkpoint_valid(value["checkpoint"])
        and value["checkpoint"] == expected_checkpoint()
        and _provenance_checkpoint_binding_valid(value)
        and _gpu_inventory_valid(value["gpus"])
        and _arm_environment_valid(value["environment"], arm=arm)
        and _runtime_valid(
            value["runtime"],
            arm=arm,
            candidate=candidate,
            moe_a2a_backend=moe_a2a_backend,
        )
        and _runtime_witness_valid(
            value["runtime_witness"],
            arm=arm,
            runtime=value["runtime"],
        )
    )


def load_provenance(
    path: Path,
    *,
    arm: str,
    candidate: str,
    moe_a2a_backend: str,
) -> dict[str, object]:
    result = hashed_descriptor(_read_json_object(path))
    if not _descriptor_valid(result) or not _provenance_value_valid(
        result["value"],
        arm=arm,
        candidate=candidate,
        moe_a2a_backend=moe_a2a_backend,
    ):
        raise GateEvidenceError(f"{path} does not satisfy the G4 M-A3 provenance schema")
    return result


def _capture_command(command: Sequence[str]) -> str:
    """Run one read-only provenance observation without a shell."""

    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise GateEvidenceError(
            f"cannot execute provenance observation {command[0]!r}: {error}"
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise GateEvidenceError(
            f"provenance observation failed ({' '.join(command)}): {detail[:500]}"
        )
    return result.stdout


def _capture_single_object(command: Sequence[str], *, label: str) -> dict[str, object]:
    try:
        value = strict_json_loads(_capture_command(command))
    except json.JSONDecodeError as error:
        raise GateEvidenceError(f"{label} is not valid JSON") from error
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], dict)
    ):
        raise GateEvidenceError(f"{label} must contain exactly one object")
    return value[0]


def _capture_env(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        raise GateEvidenceError("container environment is not a list")
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, str) or "=" not in item:
            raise GateEvidenceError("container environment contains a malformed item")
        key, item_value = item.split("=", 1)
        if key in result:
            raise GateEvidenceError(f"container environment repeats {key!r}")
        result[key] = item_value
    return result


def _canonical_cap_add(value: object) -> list[str]:
    """Normalize Docker's optional ``CAP_`` display prefix, and nothing else."""

    if not isinstance(value, list) or len(value) != len(EXPECTED_CAP_ADD) or not all(
        isinstance(item, str) for item in value
    ):
        raise GateEvidenceError("container capability list is absent or malformed")
    normalized = [item.removeprefix("CAP_") for item in value]
    if len(set(normalized)) != len(normalized) or set(normalized) != set(EXPECTED_CAP_ADD):
        raise GateEvidenceError("container capabilities differ from SYS_NICE/SYS_PTRACE")
    return list(EXPECTED_CAP_ADD)


def _checkpoint_volume_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "name",
            "driver",
            "scope",
            "mountpoint",
            "subpath",
            "rw_consumer_count",
        }
        and isinstance(value["name"], str)
        and bool(value["name"])
        and isinstance(value["driver"], str)
        and bool(value["driver"])
        and isinstance(value["scope"], str)
        and bool(value["scope"])
        and isinstance(value["mountpoint"], str)
        and bool(value["mountpoint"])
        and value["subpath"] == MODEL_VOLUME_SUBPATH
        and value["rw_consumer_count"] == 0
    )


def _capture_checkpoint_volume(
    *,
    container_id: str,
    model_volume: str,
) -> dict[str, object]:
    """Bind Docker's live volume identity and reject every RW consumer."""

    volume = _capture_single_object(
        ("docker", "volume", "inspect", model_volume),
        label="docker model volume inspect",
    )
    identity: dict[str, object] = {
        "name": volume.get("Name"),
        "driver": volume.get("Driver"),
        "scope": volume.get("Scope"),
        "mountpoint": volume.get("Mountpoint"),
        "subpath": MODEL_VOLUME_SUBPATH,
        "rw_consumer_count": 0,
    }
    if not _checkpoint_volume_valid(identity) or identity["name"] != model_volume:
        raise GateEvidenceError("Docker model volume identity is absent or malformed")

    consumer_ids = sorted(
        {
            line.strip()
            for line in _capture_command(
                (
                    "docker",
                    "ps",
                    "--all",
                    "--quiet",
                    "--no-trunc",
                    "--filter",
                    f"volume={model_volume}",
                )
            ).splitlines()
            if line.strip()
        }
    )
    if container_id not in consumer_ids:
        raise GateEvidenceError("running server is absent from Docker volume consumers")
    try:
        consumers = strict_json_loads(
            _capture_command(("docker", "inspect", *consumer_ids))
        )
    except json.JSONDecodeError as error:
        raise GateEvidenceError("Docker volume consumer inspect is not valid JSON") from error
    if (
        not isinstance(consumers, list)
        or len(consumers) != len(consumer_ids)
        or not all(isinstance(consumer, dict) for consumer in consumers)
    ):
        raise GateEvidenceError("Docker volume consumer inspect is incomplete")
    observed_ids = {consumer.get("Id") for consumer in consumers}
    if observed_ids != set(consumer_ids):
        raise GateEvidenceError("Docker volume consumer identities changed during capture")
    for consumer in consumers:
        live_mounts = [
            mount
            for mount in consumer.get("Mounts") or ()
            if isinstance(mount, dict)
            and mount.get("Type") == "volume"
            and mount.get("Name") == model_volume
        ]
        host_config = consumer.get("HostConfig")
        configured_mounts = [
            mount
            for mount in (
                host_config.get("Mounts") if isinstance(host_config, dict) else None
            )
            or ()
            if isinstance(mount, dict)
            and mount.get("Type") == "volume"
            and mount.get("Source") == model_volume
        ]
        if not live_mounts or not configured_mounts:
            raise GateEvidenceError(
                f"Docker volume consumer {consumer.get('Id')} has ambiguous mounts"
            )
        if any(mount.get("RW") is not False for mount in live_mounts) or any(
            mount.get("ReadOnly") is not True for mount in configured_mounts
        ):
            raise GateEvidenceError(
                f"model volume has a read-write consumer {consumer.get('Id')}"
            )
    return identity


def _capture_host_gpus() -> list[dict[str, object]]:
    output = _capture_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[dict[str, object]] = []
    for raw in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(raw) != 6:
            raise GateEvidenceError(f"host GPU inventory has an invalid row: {raw!r}")
        index, uuid, pci, name, driver, memory_mib = (item.strip() for item in raw)
        try:
            parsed_index = int(index)
            parsed_memory = int(memory_mib) * 1024**2
        except ValueError as error:
            raise GateEvidenceError("host GPU inventory has a non-integer field") from error
        if parsed_index in PHYSICAL_GPU_INDICES:
            rows.append(
                {
                    "physical_index": parsed_index,
                    "uuid": uuid,
                    "pci_bus_id": pci,
                    "name": name,
                    "driver_version": driver,
                    "total_memory_bytes": parsed_memory,
                }
            )
    rows.sort(key=lambda row: int(row["physical_index"]))
    if (
        [row["physical_index"] for row in rows] != list(PHYSICAL_GPU_INDICES)
        or len({row["uuid"] for row in rows}) != GPU_COUNT
        or len({row["pci_bus_id"] for row in rows}) != GPU_COUNT
        or len({row["driver_version"] for row in rows}) != 1
    ):
        raise GateEvidenceError("host GPU inventory does not contain fixed GPUs 4-7")
    return rows


def _capture_container_gpus(
    container_name: str,
    host_gpus: Sequence[Mapping[str, object]],
) -> None:
    output = _capture_command(
        (
            "docker",
            "exec",
            container_name,
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    if len(rows) != GPU_COUNT or any(len(row) != 5 for row in rows):
        raise GateEvidenceError("container GPU inventory must contain exactly four rows")
    for visible_index, (raw, host) in enumerate(zip(rows, host_gpus, strict=True)):
        index, uuid, pci, name, memory_mib = (item.strip() for item in raw)
        try:
            parsed_index = int(index)
            parsed_memory = int(memory_mib) * 1024**2
        except ValueError as error:
            raise GateEvidenceError("container GPU inventory has a non-integer field") from error
        if not (
            parsed_index == visible_index
            and uuid == host["uuid"]
            and pci.casefold() == str(host["pci_bus_id"]).casefold()
            and name == host["name"]
            and parsed_memory == host["total_memory_bytes"]
        ):
            raise GateEvidenceError(
                "container-visible GPU order differs from physical GPUs 4-7"
            )


def _capture_gpu_exclusivity(
    container_name: str,
    host_gpus: Sequence[Mapping[str, object]],
) -> None:
    top = _capture_command(("docker", "top", container_name, "-eo", "pid"))
    lines = [line.strip() for line in top.splitlines() if line.strip()]
    if len(lines) < 2 or lines[0].split()[0].casefold() != "pid":
        raise GateEvidenceError("docker top did not return container host PIDs")
    try:
        container_pids = {int(line.split()[0]) for line in lines[1:]}
    except (IndexError, ValueError) as error:
        raise GateEvidenceError("docker top returned an invalid PID") from error

    applications = _capture_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        )
    )
    selected = {str(row["uuid"]) for row in host_gpus}
    observed_uuids: set[str] = set()
    for raw in csv.reader(applications.splitlines(), skipinitialspace=True):
        if len(raw) != 2:
            raise GateEvidenceError(f"GPU process inventory has an invalid row: {raw!r}")
        uuid, pid_value = (item.strip() for item in raw)
        if uuid not in selected:
            continue
        try:
            pid = int(pid_value)
        except ValueError as error:
            raise GateEvidenceError("GPU process inventory has a non-integer PID") from error
        observed_uuids.add(uuid)
        if pid not in container_pids:
            raise GateEvidenceError(
                f"physical GPU cohort has a foreign compute process PID {pid}"
            )
    if observed_uuids != selected:
        raise GateEvidenceError("the running server does not own a process on every selected GPU")


def _normalize_captured_nccl_version(value: object) -> str:
    if isinstance(value, list) and value and all(type(item) is int for item in value):
        return ".".join(str(item) for item in value)
    if type(value) is int:
        return f"{value // 10000}.{(value // 100) % 100}.{value % 100}"
    if isinstance(value, str) and value:
        return value
    raise GateEvidenceError("container NCCL version is absent or malformed")


def _capture_runtime_versions(container_name: str, arm: str) -> dict[str, str]:
    python = "/app/.venv/bin/python" if arm == "kairyu" else "python3"
    engine_distribution = "kairyu" if arm == "kairyu" else "sglang"
    probe = (
        "import importlib.metadata as m,json,torch\n"
        "def version(*names):\n"
        "  for name in names:\n"
        "    try: return m.version(name)\n"
        "    except m.PackageNotFoundError: pass\n"
        "  raise RuntimeError('missing distribution: '+','.join(names))\n"
        f"engine={engine_distribution!r}\n"
        "print(json.dumps({'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version(),"
        "'torch':version('torch'),'flashinfer':version('flashinfer-python','flashinfer'),"
        "'engine':version(engine)}))"
    )
    output = _capture_command(("docker", "exec", container_name, python, "-c", probe))
    try:
        value = strict_json_loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise GateEvidenceError("container runtime-version probe is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "cuda",
        "nccl",
        "torch",
        "flashinfer",
        "engine",
    }:
        raise GateEvidenceError("container runtime-version probe is incomplete")
    strings = {
        "cuda_version": value["cuda"],
        "torch_version": value["torch"],
        "flashinfer_version": value["flashinfer"],
        "engine_version": value["engine"],
    }
    if not all(isinstance(item, str) and item for item in strings.values()):
        raise GateEvidenceError("container runtime-version probe has an empty version")
    return {
        **strings,
        "nccl_version": _normalize_captured_nccl_version(value["nccl"]),
    }


def _captured_runtime(
    *,
    arm: str,
    candidate: str,
    resolved_argv: list[str],
) -> dict[str, object]:
    common: dict[str, object] = {
        "served_model_name": SERVED_MODEL_NAME,
        "world_size": GPU_COUNT,
        "tensor_parallel_size": 1 if arm == "kairyu" else GPU_COUNT,
        "data_parallel_size": GPU_COUNT,
        "expert_parallel_size": GPU_COUNT,
        "attention_data_parallel": True,
        "request_owned_attention": True,
        "kv_cache_owner": "request-owner",
        "sampling_owner": "request-owner",
        "moe_a2a_backend": (
            KAIRYU_MOE_A2A_BACKEND if arm == "kairyu" else DEFAULT_SGLANG_MOE_A2A_BACKEND
        ),
        "moe_runner_backend": (
            KAIRYU_MOE_RUNNER_BACKEND if arm == "kairyu" else SGLANG_MOE_RUNNER_BACKEND
        ),
        "pipeline_depth": 5 if arm == "kairyu" else None,
        "overlap_mode": candidate if arm == "sglang" else None,
        "prefill_cuda_graph": "disabled",
        "decode_cuda_graph": True,
        "cuda_graph_max_batch": KAIRYU_CUDA_GRAPH_MAX_BATCH if arm == "kairyu" else None,
        "cuda_graph_max_pages": KAIRYU_CUDA_GRAPH_MAX_PAGES if arm == "kairyu" else None,
        "cuda_graph_warmup_iters": (
            KAIRYU_CUDA_GRAPH_WARMUP_ITERS if arm == "kairyu" else None
        ),
        "cuda_graph_buckets": list(KAIRYU_CUDA_GRAPH_BUCKETS) if arm == "kairyu" else None,
        "max_running_requests": MAX_RUNNING_REQUESTS,
        "configured_chunked_prefill_tokens": CONFIGURED_PREFILL_TOKENS,
        "resolved_chunked_prefill_tokens_per_owner": RESOLVED_PREFILL_TOKENS_PER_OWNER,
        "configured_max_prefill_tokens": (
            CONFIGURED_PREFILL_TOKENS if arm == "kairyu" else SGLANG_MAX_PREFILL_TOKENS_PER_OWNER
        ),
        "resolved_max_prefill_tokens_per_owner": RESOLVED_PREFILL_TOKENS_PER_OWNER,
        "page_size": PAGE_SIZE,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "max_total_tokens_per_owner": MAX_TOTAL_TOKENS_PER_OWNER,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "prefix_cache": True,
        "scheduler_policy": "fcfs",
        "speculative_decoding": False,
        "access_log": False,
        "resolved_argv": resolved_argv,
    }
    return common


def _capture_runtime_witness(arm: str) -> dict[str, object]:
    endpoint = "/backends" if arm == "kairyu" else "/server_info"
    try:
        response = httpx.get(f"http://127.0.0.1:30000{endpoint}", timeout=30.0)
    except httpx.HTTPError as error:
        raise GateEvidenceError(f"live runtime endpoint {endpoint} failed: {error}") from error
    if response.status_code != 200:
        raise GateEvidenceError(
            f"live runtime endpoint {endpoint} returned HTTP {response.status_code}"
        )
    try:
        payload = strict_json_loads(canonical_json(response.json()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise GateEvidenceError(f"live runtime endpoint {endpoint} is not strict JSON") from error
    if not isinstance(payload, dict):
        raise GateEvidenceError(f"live runtime endpoint {endpoint} did not return an object")
    return {"endpoint": endpoint, "response": payload}


def _sglang_runtime_witness_matches(
    response: Mapping[str, object], runtime: Mapping[str, object]
) -> bool:
    graph = response.get("cuda_graph_config")
    decode = graph.get("decode") if isinstance(graph, dict) else None
    prefill = graph.get("prefill") if isinstance(graph, dict) else None
    internal_states = response.get("internal_states")
    required_rank_values = {
        "served_model_name": SERVED_MODEL_NAME,
        "tp_size": GPU_COUNT,
        "dp_size": GPU_COUNT,
        "ep_size": GPU_COUNT,
        "enable_dp_attention": True,
        "load_balance_method": "round_robin",
        "dtype": "bfloat16",
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "fp4_gemm_runner_backend": SGLANG_FP4_GEMM_BACKEND,
        "moe_a2a_backend": runtime.get("moe_a2a_backend"),
        "moe_runner_backend": runtime.get("moe_runner_backend"),
        "page_size": PAGE_SIZE,
        "max_running_requests": MAX_RUNNING_REQUESTS,
        "max_total_tokens": MAX_TOTAL_TOKENS_PER_OWNER,
        "chunked_prefill_size": RESOLVED_PREFILL_TOKENS_PER_OWNER,
        "max_prefill_tokens": SGLANG_MAX_PREFILL_TOKENS_PER_OWNER,
        "schedule_policy": "fcfs",
        "disable_radix_cache": False,
        "speculative_algorithm": None,
        "log_requests": False,
        "log_level_http": "warning",
        "enable_single_batch_overlap": False,
        "enable_two_batch_overlap": False,
    }
    return (
        response.get("version") == SGLANG_VERSION
        and response.get("model_path") == SERVED_MODEL_NAME
        and all(response.get(key) == expected for key, expected in required_rank_values.items())
        and response.get("max_total_num_tokens") == MAX_TOTAL_TOKENS_PER_OWNER
        and isinstance(decode, dict)
        and decode.get("backend") == "full"
        and decode.get("max_bs") == KAIRYU_CUDA_GRAPH_MAX_BATCH
        and isinstance(prefill, dict)
        and prefill.get("backend") == "disabled"
        and isinstance(internal_states, list)
        and len(internal_states) == GPU_COUNT
        and all(
            isinstance(state, dict)
            and all(state.get(key) == expected for key, expected in required_rank_values.items())
            for state in internal_states
        )
        and runtime.get("world_size") == response.get("tp_size")
        and runtime.get("tensor_parallel_size") == response.get("tp_size")
        and runtime.get("data_parallel_size") == response.get("dp_size")
        and runtime.get("expert_parallel_size") == response.get("ep_size")
        and runtime.get("attention_data_parallel") == response.get("enable_dp_attention")
        and runtime.get("resolved_chunked_prefill_tokens_per_owner")
        == response.get("chunked_prefill_size")
        and runtime.get("resolved_max_prefill_tokens_per_owner")
        == response.get("max_prefill_tokens")
        and runtime.get("page_size") == response.get("page_size")
        and type(response.get("max_total_tokens")) is int
        and runtime.get("max_total_tokens") == response.get("max_total_tokens") * GPU_COUNT
        and runtime.get("max_total_tokens_per_owner") == response.get("max_total_tokens")
        and runtime.get("kv_cache_dtype") == response.get("kv_cache_dtype")
        and runtime.get("prefix_cache") is (not response.get("disable_radix_cache"))
        and runtime.get("scheduler_policy") == response.get("schedule_policy")
        and runtime.get("speculative_decoding")
        is (response.get("speculative_algorithm") is not None)
        and runtime.get("access_log") is False
        and response.get("log_level_http") == "warning"
        and runtime.get("prefill_cuda_graph") == prefill.get("backend")
        and runtime.get("decode_cuda_graph") is True
    )


def _kairyu_runtime_witness_matches(
    response: Mapping[str, object], runtime: Mapping[str, object]
) -> bool:
    engines = response.get("engines")
    if not isinstance(engines, list):
        return False
    selected = [
        engine
        for engine in engines
        if isinstance(engine, dict) and engine.get("model") == SERVED_MODEL_NAME
    ]
    if len(selected) != 1:
        return False
    engine = selected[0]
    transport = engine.get("moe_collective_transport")
    return (
        response.get("role") == "engine-host"
        and engine.get("engine_backend") == "kairyu"
        and engine.get("parallelism") == "expert_parallel"
        and engine.get("expert_parallel_size") == runtime.get("expert_parallel_size")
        and engine.get("attention_data_parallel_size") == runtime.get("data_parallel_size")
        and engine.get("attention_tensor_parallel_size")
        == runtime.get("tensor_parallel_size")
        and engine.get("attention_placement") == "request_owned_data_parallel"
        and engine.get("attention_output_placement") == "replicated"
        and engine.get("execution_mode") == "request-owned-attention-dp"
        and engine.get("pipeline_depth") == runtime.get("pipeline_depth")
        and engine.get("decode_mode") == "cuda_graph"
        and engine.get("kv_cache_dtype") == runtime.get("kv_cache_dtype")
        and engine.get("moe_dispatcher") == runtime.get("moe_runner_backend")
        and engine.get("sampling_ownership") == "request_owner"
        and engine.get("kv_cache_ownership") == "request_owner"
        and engine.get("cuda_graph_decode") is runtime.get("decode_cuda_graph")
        and engine.get("cuda_graph_buckets") == runtime.get("cuda_graph_buckets")
        and type(engine.get("cuda_graph_captures")) is int
        and type(engine.get("cuda_graph_replays")) is int
        and engine.get("cuda_graph_eager_fallbacks") == 0
        and isinstance(transport, dict)
        and transport.get("selected_backend") == runtime.get("moe_a2a_backend")
        and transport.get("direct_nccl_active") is True
    )


def _runtime_witness_valid(
    value: object,
    *,
    arm: str,
    runtime: object,
) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != {"endpoint", "response"}
        or not isinstance(value["response"], dict)
        or not isinstance(runtime, dict)
    ):
        return False
    if arm == "sglang":
        return value["endpoint"] == "/server_info" and _sglang_runtime_witness_matches(
            value["response"], runtime
        )
    return arm == "kairyu" and value["endpoint"] == "/backends" and (
        _kairyu_runtime_witness_matches(value["response"], runtime)
    )


def capture_provenance(
    output: Path,
    *,
    arm: str,
    candidate: str,
    container_name: str,
    server_generation_id: str,
    server_started_ns: int,
    model_volume: str,
    checkpoint_start_path: Path,
    source_root: Path | None,
) -> dict[str, object]:
    """Capture one already-running formal server without supervising it."""

    if candidate not in PREFLIGHT_CANDIDATES.get(arm, ()):
        raise GateEvidenceError(f"candidate {candidate!r} is invalid for {arm!r}")
    if not container_name or not server_generation_id or not model_volume:
        raise GateEvidenceError("container, generation, and model volume must be non-empty")
    if type(server_started_ns) is not int or server_started_ns <= 0:
        raise GateEvidenceError("server_started_ns must be a positive host monotonic timestamp")
    if (arm == "kairyu") != (source_root is not None):
        raise GateEvidenceError("source_root is required only for the Kairyu arm")
    checkpoint_capture_start = load_checkpoint_capture(
        checkpoint_start_path,
        phase="start",
    )
    checkpoint_capture_value = checkpoint_capture_start["value"]
    assert isinstance(checkpoint_capture_value, dict)
    if checkpoint_capture_value["capture_completed_ns"] >= server_started_ns:
        raise GateEvidenceError("server started before the checkpoint start capture completed")

    container = _capture_single_object(
        ("docker", "inspect", container_name),
        label="docker container inspect",
    )
    config = container.get("Config")
    host_config = container.get("HostConfig")
    state = container.get("State")
    mounts = container.get("Mounts")
    if not all(isinstance(item, dict) for item in (config, host_config, state)) or not isinstance(
        mounts, list
    ):
        raise GateEvidenceError("docker inspect omitted Config/HostConfig/State/Mounts")
    assert isinstance(config, dict) and isinstance(host_config, dict) and isinstance(state, dict)
    if state.get("Running") is not True:
        raise GateEvidenceError("provenance requires one already-running container")

    container_id = container.get("Id")
    image_config_id = container.get("Image")
    image_repo_digest = config.get("Image")
    if (
        not isinstance(container_id, str)
        or _CONTAINER_ID.fullmatch(container_id) is None
        or not isinstance(image_config_id, str)
        or _IMAGE_ID.fullmatch(image_config_id) is None
        or not isinstance(image_repo_digest, str)
        or _REPO_DIGEST.fullmatch(image_repo_digest) is None
    ):
        raise GateEvidenceError("container/image identities are not immutable digests")
    image_platform_digest = image_repo_digest.split("@", 1)[1]
    if arm == "sglang" and (
        image_repo_digest != SGLANG_IMAGE_REPO_DIGEST
        or image_platform_digest != SGLANG_AMD64_MANIFEST_DIGEST
    ):
        raise GateEvidenceError("SGLang container does not use the pinned amd64 RepoDigest")
    image = _capture_single_object(
        ("docker", "image", "inspect", image_repo_digest),
        label="docker image inspect",
    )
    repo_digests = image.get("RepoDigests")
    if (
        image.get("Id") != image_config_id
        or image.get("Os") != "linux"
        or image.get("Architecture") != "amd64"
        or not isinstance(repo_digests, list)
        or image_repo_digest not in repo_digests
    ):
        raise GateEvidenceError("container image does not match its local amd64 image record")

    model_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == CHECKPOINT_ROOT
    ]
    if len(model_mounts) != 1:
        raise GateEvidenceError("container must have one model mount at the checkpoint root")
    model_mount = model_mounts[0]
    observed_subpath = model_mount.get("Subpath")
    if not (
        model_mount.get("Type") == "volume"
        and model_mount.get("Name") == model_volume
        and model_mount.get("RW") is False
        and (observed_subpath is None or observed_subpath == MODEL_VOLUME_SUBPATH)
    ):
        raise GateEvidenceError("container model volume/name/subpath/read-only state is invalid")
    host_mounts = host_config.get("Mounts")
    host_model_mounts = [
        mount
        for mount in host_mounts or ()
        if isinstance(mount, dict) and mount.get("Target") == CHECKPOINT_ROOT
    ]
    if len(host_model_mounts) != 1:
        raise GateEvidenceError("HostConfig does not retain one model volume mount")
    host_model_mount = host_model_mounts[0]
    volume_options = host_model_mount.get("VolumeOptions")
    if not (
        host_model_mount.get("Type") == "volume"
        and host_model_mount.get("Source") == model_volume
        and host_model_mount.get("ReadOnly") is True
        and isinstance(volume_options, dict)
        and volume_options.get("Subpath") == MODEL_VOLUME_SUBPATH
    ):
        raise GateEvidenceError("HostConfig model volume-subpath contract is invalid")

    cap_add = _canonical_cap_add(host_config.get("CapAdd"))
    if (
        host_config.get("IpcMode") != "host"
        or host_config.get("ShmSize") != 32 * 1024**3
    ):
        raise GateEvidenceError("container IPC/shared-memory/capability contract is invalid")
    device_requests = host_config.get("DeviceRequests")
    gpu_requests = [
        request
        for request in device_requests or ()
        if isinstance(request, dict)
        and (
            request.get("Driver") == "nvidia"
            or any("gpu" in group for group in request.get("Capabilities") or ())
        )
    ]
    if len(gpu_requests) != 1 or gpu_requests[0].get("DeviceIDs") != ["4", "5", "6", "7"]:
        raise GateEvidenceError("container does not request exactly physical GPUs 4-7")
    port_bindings = host_config.get("PortBindings")
    published = (
        port_bindings.get("30000/tcp") if isinstance(port_bindings, dict) else None
    )
    if published != [{"HostIp": "127.0.0.1", "HostPort": "30000"}]:
        raise GateEvidenceError("container does not publish only the fixed loopback endpoint")
    environment = _capture_env(config.get("Env"))
    if environment.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise GateEvidenceError("container CUDA_VISIBLE_DEVICES is not the local 0-3 mapping")
    if arm == "kairyu" and any(
        environment.get(key) != expected
        for key, expected in {
            "PYTHONPATH": "/workspace",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "/workspace",
        }.items()
    ):
        raise GateEvidenceError("Kairyu source-import/safe-directory environment is invalid")

    path = container.get("Path")
    arguments = container.get("Args")
    if not isinstance(path, str) or not path or not isinstance(arguments, list) or not all(
        isinstance(item, str) for item in arguments
    ):
        raise GateEvidenceError("container runtime argv is absent or malformed")
    resolved_argv = [path, *arguments]

    labels = image.get("Config")
    labels = labels.get("Labels") if isinstance(labels, dict) else None
    if arm == "kairyu":
        assert source_root is not None
        resolved_source = source_root.resolve()
        source_mounts = [
            mount
            for mount in mounts
            if isinstance(mount, dict) and mount.get("Destination") == "/workspace"
        ]
        if len(source_mounts) != 1:
            raise GateEvidenceError("Kairyu container has no singular /workspace source mount")
        source_mount = source_mounts[0]
        source = source_mount.get("Source")
        if not (
            source_mount.get("Type") == "bind"
            and isinstance(source, str)
            and Path(source).resolve() == resolved_source
            and source_mount.get("RW") is False
            and config.get("WorkingDir") == "/workspace"
        ):
            raise GateEvidenceError("Kairyu source mount is not the clean read-only checkout")
        commit = _capture_command(("git", "-C", str(resolved_source), "rev-parse", "HEAD")).strip()
        status = _capture_command(
            (
                "git",
                "-C",
                str(resolved_source),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
        )
        _capture_command(
            (
                "git",
                "-C",
                str(resolved_source),
                "ls-files",
                "--error-unmatch",
                "bench/g4_ma3_kairyu_server.py",
            )
        )
        if _HEX40.fullmatch(commit) is None or status:
            raise GateEvidenceError("Kairyu detached source is not a clean tracked commit")
        if (
            not isinstance(labels, dict)
            or labels.get("org.opencontainers.image.revision") != commit
        ):
            raise GateEvidenceError("Kairyu image revision label differs from mounted source")
        source_value: dict[str, object] = {
            "repository": "https://github.com/ytworks/kairyu",
            "commit": commit,
            "clean": True,
        }
    else:
        source_value = {
            "repository": SGLANG_SOURCE_REPOSITORY,
            "version": SGLANG_VERSION,
            "tag": SGLANG_TAG,
            "tag_object": SGLANG_TAG_OBJECT,
            "commit": SGLANG_SOURCE_COMMIT,
            "clean": True,
        }

    host_gpus = _capture_host_gpus()
    _capture_container_gpus(container_name, host_gpus)
    _capture_gpu_exclusivity(container_name, host_gpus)
    checkpoint_volume = _capture_checkpoint_volume(
        container_id=container_id,
        model_volume=model_volume,
    )
    versions = _capture_runtime_versions(container_name, arm)
    runtime = _captured_runtime(
        arm=arm,
        candidate=candidate,
        resolved_argv=resolved_argv,
    )
    runtime_witness = _capture_runtime_witness(arm)
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "server_generation_id": server_generation_id,
        "server_started_ns": server_started_ns,
        "container": {
            "id": container_id,
            "image_repo_digest": image_repo_digest,
            "image_platform_digest": image_platform_digest,
            "image_config_id": image_config_id,
            "model_mount": {
                "source": f"volume:{model_volume}/{MODEL_VOLUME_SUBPATH}",
                "destination": CHECKPOINT_ROOT,
                "read_only": True,
            },
            "gpu_device_indices": list(PHYSICAL_GPU_INDICES),
            "ipc_mode": "host",
            "shm_size_bytes": 32 * 1024**3,
            "cap_add": cap_add,
        },
        "source": source_value,
        "checkpoint": checkpoint_capture_value["checkpoint"]["value"],
        "checkpoint_capture_start": checkpoint_capture_start,
        "checkpoint_volume": checkpoint_volume,
        "gpus": [
            {
                "visible_index": visible_index,
                "physical_index": host["physical_index"],
                "uuid": host["uuid"],
                "pci_bus_id": host["pci_bus_id"],
                "name": host["name"],
                "total_memory_bytes": host["total_memory_bytes"],
            }
            for visible_index, host in enumerate(host_gpus)
        ],
        "environment": {
            "host_id": socket.gethostname(),
            "driver_version": host_gpus[0]["driver_version"],
            **versions,
            "engine_name": arm,
            "gpu_jobs_exclusive": True,
        },
        "runtime": runtime,
        "runtime_witness": runtime_witness,
    }
    if not _provenance_value_valid(
        value,
        arm=arm,
        candidate=candidate,
        moe_a2a_backend=DEFAULT_SGLANG_MOE_A2A_BACKEND,
    ):
        raise GateEvidenceError("captured runtime does not satisfy the provenance schema")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value, indent=2, sort_keys=True, ensure_ascii=False)
    load_provenance(
        output,
        arm=arm,
        candidate=candidate,
        moe_a2a_backend=DEFAULT_SGLANG_MOE_A2A_BACKEND,
    )
    return value


def _live_observation_value_valid(
    value: object,
    *,
    provenance: Mapping[str, object],
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "capture_started_ns",
        "capture_completed_ns",
        "container",
        "source",
        "checkpoint_volume",
        "gpus",
        "environment",
        "runtime_witness",
    }:
        return False
    provenance_container = provenance.get("container")
    runtime = provenance.get("runtime")
    if not isinstance(provenance_container, dict) or not isinstance(runtime, dict):
        return False
    expected_container = {
        "id": provenance_container.get("id"),
        "image_repo_digest": provenance_container.get("image_repo_digest"),
        "image_platform_digest": provenance_container.get("image_platform_digest"),
        "image_config_id": provenance_container.get("image_config_id"),
        "model_mount": provenance_container.get("model_mount"),
        "resolved_argv": runtime.get("resolved_argv"),
    }
    return (
        value["schema_version"] == LIVE_OBSERVATION_SCHEMA_VERSION
        and value["type"] == "live_observation_end"
        and type(value["capture_started_ns"]) is int
        and type(value["capture_completed_ns"]) is int
        and 0 < value["capture_started_ns"] < value["capture_completed_ns"]
        and value["container"] == expected_container
        and value["source"] == provenance.get("source")
        and value["checkpoint_volume"] == provenance.get("checkpoint_volume")
        and _checkpoint_volume_valid(value["checkpoint_volume"])
        and value["gpus"] == provenance.get("gpus")
        and value["environment"] == provenance.get("environment")
        and isinstance(provenance.get("arm"), str)
        and _runtime_witness_valid(
            value["runtime_witness"],
            arm=str(provenance["arm"]),
            runtime=runtime,
        )
    )


def capture_live_observation(provenance: Mapping[str, object]) -> dict[str, object]:
    """Re-observe the measured container and machine after all shard traffic."""

    arm = provenance.get("arm")
    recorded_container = provenance.get("container")
    recorded_runtime = provenance.get("runtime")
    recorded_volume = provenance.get("checkpoint_volume")
    if (
        arm not in ARMS
        or not isinstance(recorded_container, dict)
        or not isinstance(recorded_runtime, dict)
        or not isinstance(recorded_volume, dict)
    ):
        raise GateEvidenceError("end observation requires valid start provenance")
    container_id = recorded_container.get("id")
    model_volume = recorded_volume.get("name")
    if not isinstance(container_id, str) or not isinstance(model_volume, str):
        raise GateEvidenceError("end observation has no container/volume identity")

    capture_started_ns = time.perf_counter_ns()
    container = _capture_single_object(
        ("docker", "inspect", container_id),
        label="end container inspect",
    )
    config = container.get("Config")
    host_config = container.get("HostConfig")
    state = container.get("State")
    mounts = container.get("Mounts")
    if not all(isinstance(item, dict) for item in (config, host_config, state)) or not isinstance(
        mounts, list
    ):
        raise GateEvidenceError("end container inspect is incomplete")
    assert isinstance(config, dict) and isinstance(host_config, dict) and isinstance(state, dict)
    resolved_argv = recorded_runtime.get("resolved_argv")
    if not isinstance(resolved_argv, list) or not resolved_argv:
        raise GateEvidenceError("start provenance has no resolved argv")
    if not (
        container.get("Id") == container_id
        and state.get("Running") is True
        and container.get("Image") == recorded_container.get("image_config_id")
        and config.get("Image") == recorded_container.get("image_repo_digest")
        and [container.get("Path"), *(container.get("Args") or ())] == resolved_argv
    ):
        raise GateEvidenceError("measured container identity/image/argv changed or stopped")
    image = _capture_single_object(
        ("docker", "image", "inspect", str(recorded_container["image_repo_digest"])),
        label="end image inspect",
    )
    if not (
        image.get("Id") == recorded_container.get("image_config_id")
        and image.get("Os") == "linux"
        and image.get("Architecture") == "amd64"
        and recorded_container.get("image_repo_digest") in (image.get("RepoDigests") or ())
    ):
        raise GateEvidenceError("measured image record changed after traffic")

    expected_mount = recorded_container.get("model_mount")
    if not isinstance(expected_mount, dict):
        raise GateEvidenceError("start provenance has no model mount")
    live_model_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == CHECKPOINT_ROOT
    ]
    configured_model_mounts = [
        mount
        for mount in host_config.get("Mounts") or ()
        if isinstance(mount, dict) and mount.get("Target") == CHECKPOINT_ROOT
    ]
    if len(live_model_mounts) != 1 or len(configured_model_mounts) != 1:
        raise GateEvidenceError("measured model mount disappeared after traffic")
    live_model = live_model_mounts[0]
    configured_model = configured_model_mounts[0]
    volume_options = configured_model.get("VolumeOptions")
    if not (
        live_model.get("Type") == "volume"
        and live_model.get("Name") == model_volume
        and live_model.get("RW") is False
        and live_model.get("Subpath") in (None, MODEL_VOLUME_SUBPATH)
        and configured_model.get("Type") == "volume"
        and configured_model.get("Source") == model_volume
        and configured_model.get("ReadOnly") is True
        and isinstance(volume_options, dict)
        and volume_options.get("Subpath") == MODEL_VOLUME_SUBPATH
        and expected_mount.get("source")
        == f"volume:{model_volume}/{MODEL_VOLUME_SUBPATH}"
    ):
        raise GateEvidenceError("measured model volume/subpath/read-only state changed")
    cap_add = _canonical_cap_add(host_config.get("CapAdd"))
    if (
        host_config.get("IpcMode") != recorded_container.get("ipc_mode")
        or host_config.get("ShmSize") != recorded_container.get("shm_size_bytes")
        or cap_add != recorded_container.get("cap_add")
    ):
        raise GateEvidenceError("measured container resource contract changed")
    device_requests = host_config.get("DeviceRequests")
    gpu_requests = [
        request
        for request in device_requests or ()
        if isinstance(request, dict)
        and (
            request.get("Driver") == "nvidia"
            or any("gpu" in group for group in request.get("Capabilities") or ())
        )
    ]
    if len(gpu_requests) != 1 or gpu_requests[0].get("DeviceIDs") != ["4", "5", "6", "7"]:
        raise GateEvidenceError("measured container GPU request changed")
    port_bindings = host_config.get("PortBindings")
    published = port_bindings.get("30000/tcp") if isinstance(port_bindings, dict) else None
    if published != [{"HostIp": "127.0.0.1", "HostPort": "30000"}]:
        raise GateEvidenceError("measured container loopback port binding changed")
    environment = _capture_env(config.get("Env"))
    if environment.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise GateEvidenceError("measured container GPU visibility changed")

    labels = image.get("Config")
    labels = labels.get("Labels") if isinstance(labels, dict) else None
    if arm == "kairyu":
        source_mounts = [
            mount
            for mount in mounts
            if isinstance(mount, dict) and mount.get("Destination") == CAPTURE_SOURCE_ROOT
        ]
        if len(source_mounts) != 1:
            raise GateEvidenceError("measured Kairyu source mount disappeared")
        source_mount = source_mounts[0]
        source_path = source_mount.get("Source")
        if not (
            source_mount.get("Type") == "bind"
            and isinstance(source_path, str)
            and source_mount.get("RW") is False
            and config.get("WorkingDir") == CAPTURE_SOURCE_ROOT
        ):
            raise GateEvidenceError("measured Kairyu source mount changed")
        commit = _capture_command(("git", "-C", source_path, "rev-parse", "HEAD")).strip()
        status = _capture_command(
            (
                "git",
                "-C",
                source_path,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
        )
        _capture_command(
            (
                "git",
                "-C",
                source_path,
                "ls-files",
                "--error-unmatch",
                "bench/g4_ma3_kairyu_server.py",
            )
        )
        source_value: dict[str, object] = {
            "repository": "https://github.com/ytworks/kairyu",
            "commit": commit,
            "clean": not bool(status),
        }
        if (
            source_value != provenance.get("source")
            or not isinstance(labels, dict)
            or labels.get("org.opencontainers.image.revision") != commit
        ):
            raise GateEvidenceError("measured Kairyu source/image revision changed")
    else:
        source_value = {
            "repository": SGLANG_SOURCE_REPOSITORY,
            "version": SGLANG_VERSION,
            "tag": SGLANG_TAG,
            "tag_object": SGLANG_TAG_OBJECT,
            "commit": SGLANG_SOURCE_COMMIT,
            "clean": True,
        }
        if source_value != provenance.get("source"):
            raise GateEvidenceError("measured SGLang source identity changed")

    host_gpus = _capture_host_gpus()
    _capture_container_gpus(container_id, host_gpus)
    _capture_gpu_exclusivity(container_id, host_gpus)
    checkpoint_volume = _capture_checkpoint_volume(
        container_id=container_id,
        model_volume=model_volume,
    )
    versions = _capture_runtime_versions(container_id, str(arm))
    runtime_witness = _capture_runtime_witness(str(arm))
    gpu_rows = [
        {
            "visible_index": visible_index,
            "physical_index": host["physical_index"],
            "uuid": host["uuid"],
            "pci_bus_id": host["pci_bus_id"],
            "name": host["name"],
            "total_memory_bytes": host["total_memory_bytes"],
        }
        for visible_index, host in enumerate(host_gpus)
    ]
    environment_value = {
        "host_id": socket.gethostname(),
        "driver_version": host_gpus[0]["driver_version"],
        **versions,
        "engine_name": arm,
        "gpu_jobs_exclusive": True,
    }
    capture_completed_ns = time.perf_counter_ns()
    value: dict[str, object] = {
        "schema_version": LIVE_OBSERVATION_SCHEMA_VERSION,
        "type": "live_observation_end",
        "capture_started_ns": capture_started_ns,
        "capture_completed_ns": capture_completed_ns,
        "container": {
            "id": container_id,
            "image_repo_digest": recorded_container["image_repo_digest"],
            "image_platform_digest": recorded_container["image_platform_digest"],
            "image_config_id": recorded_container["image_config_id"],
            "model_mount": expected_mount,
            "resolved_argv": resolved_argv,
        },
        "source": source_value,
        "checkpoint_volume": checkpoint_volume,
        "gpus": gpu_rows,
        "environment": environment_value,
        "runtime_witness": runtime_witness,
    }
    if not _live_observation_value_valid(value, provenance=provenance):
        raise GateEvidenceError("end live observation differs from start provenance")
    return value


def _trace_rows(trace: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    raw = trace.get("value")
    if not isinstance(raw, dict) or not isinstance(raw.get(key), list):
        raise GateEvidenceError(f"trace has no {key!r} rows")
    rows = raw[key]
    if not all(isinstance(row, dict) for row in rows):
        raise GateEvidenceError("trace contains a non-object request")
    return rows


def _planned_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    phase: str,
    output_tokens: int,
    preflight_repeat: int | None = None,
    open_loop_point: int | None = None,
    rate_millirps: int | None = None,
) -> list[PlannedRequest]:
    return [
        PlannedRequest(
            phase=phase,
            sequence=int(row["sequence"]),
            prompt_text=str(row["prompt_text"]),
            prompt_token_ids=tuple(int(token) for token in row["prompt_token_ids"]),
            expected_completion_tokens=output_tokens,
            preflight_repeat=preflight_repeat,
            open_loop_point=open_loop_point,
            rate_millirps=rate_millirps,
        )
        for row in rows
    ]


def _serial_warmup_plan(trace: Mapping[str, object]) -> list[PlannedRequest]:
    rows = _trace_rows(trace, "warmup")
    if len(rows) != SERIAL_WARMUP_REQUESTS:
        raise GateEvidenceError("ShareGPT trace must contain four serial warmups")
    return _planned_from_rows(
        rows,
        phase="serial_warmup",
        output_tokens=SERIAL_WARMUP_OUTPUT_TOKENS,
    )


def _graph_warmup_plan(graph_warmup: Mapping[str, object]) -> list[PlannedRequest]:
    if not _graph_warmup_trace_valid(graph_warmup):
        raise GateEvidenceError("graph warmup trace is invalid")
    raw = graph_warmup["value"]
    assert isinstance(raw, dict)
    prompts = raw["prompts"]
    assert isinstance(prompts, list)
    result: list[PlannedRequest] = []
    offset = 0
    for batch_size in GRAPH_WARMUP_BATCH_SIZES:
        for batch_sequence in range(batch_size):
            row = prompts[offset]
            assert isinstance(row, dict)
            result.append(
                PlannedRequest(
                    phase="graph_warmup",
                    sequence=int(row["sequence"]),
                    prompt_text=str(row["prompt_text"]),
                    prompt_token_ids=tuple(int(token) for token in row["prompt_token_ids"]),
                    expected_completion_tokens=GRAPH_WARMUP_OUTPUT_TOKENS,
                    graph_batch_size=batch_size,
                    graph_batch_sequence=batch_sequence,
                )
            )
            offset += 1
    return result


def _formal_plan(trace: Mapping[str, object]) -> list[PlannedRequest]:
    rows = _trace_rows(trace, "requests")
    if len(rows) != SHAREGPT_REQUESTS:
        raise GateEvidenceError("binding ShareGPT trace must contain exactly 128 requests")
    return _planned_from_rows(rows, phase="formal", output_tokens=SHAREGPT_OUTPUT_TOKENS)


def _preflight_plans(trace: Mapping[str, object]) -> list[list[PlannedRequest]]:
    rows = _trace_rows(trace, "requests")
    required = PREFLIGHT_REPEATS * PREFLIGHT_REQUESTS
    if len(rows) < required:
        raise GateEvidenceError("ShareGPT trace is too short for disjoint preflight repeats")
    return [
        _planned_from_rows(
            rows[repeat * PREFLIGHT_REQUESTS : (repeat + 1) * PREFLIGHT_REQUESTS],
            phase="preflight",
            output_tokens=PREFLIGHT_OUTPUT_TOKENS,
            preflight_repeat=repeat,
        )
        for repeat in range(PREFLIGHT_REPEATS)
    ]


def _open_loop_plans(
    trace: Mapping[str, object], rates_millirps: Sequence[int]
) -> list[list[PlannedRequest]]:
    rows = _trace_rows(trace, "requests")
    if len(rates_millirps) != OPEN_LOOP_POINTS:
        raise GateEvidenceError("open-loop selection must contain five rates")
    result = []
    for point, rate in enumerate(rates_millirps):
        offset = (point * OPEN_LOOP_REQUESTS_PER_POINT) % len(rows)
        selected = list(rows[offset : offset + OPEN_LOOP_REQUESTS_PER_POINT])
        if len(selected) < OPEN_LOOP_REQUESTS_PER_POINT:
            selected.extend(rows[: OPEN_LOOP_REQUESTS_PER_POINT - len(selected)])
        result.append(
            _planned_from_rows(
                selected,
                phase="open_loop",
                output_tokens=SHAREGPT_OUTPUT_TOKENS,
                open_loop_point=point,
                rate_millirps=rate,
            )
        )
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
    client: _TrackedAsyncClient,
    *,
    endpoint: str,
    planned: PlannedRequest,
) -> dict[str, object]:
    """Execute one strict OpenAI-compatible stream exactly once, without retry."""

    body = {
        "model": SERVED_MODEL_NAME,
        "prompt": planned.prompt_text,
        **request_config(planned.expected_completion_tokens),
    }
    http_client_request_ordinal = client.request_count
    start_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    end_ns = start_ns
    http_status: int | None = None
    content_type: str | None = None
    response_ids: set[str] = set()
    response_request_header: str | None = None
    output_parts: list[str] = []
    stream_events: list[str] = []
    usage: dict[str, int] | None = None
    usage_events = 0
    done_events = 0
    terminal_choice_events = 0
    finish_reasons: list[str] = []
    terminal_seen = False
    saw_usage = False
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
                stream_events.append(payload.decode(errors="replace"))
                error_type = "HTTPError"
            elif not (content_type or "").lower().startswith("text/event-stream"):
                payload = await response.aread()
                stream_events.append(payload.decode(errors="replace"))
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
                    stream_events.append(data)
                    if saw_done:
                        error_type = error_type or "DataAfterDONE"
                        continue
                    if data == "[DONE]":
                        done_events += 1
                        saw_done = True
                        continue
                    try:
                        payload = strict_json_loads(data)
                    except (json.JSONDecodeError, GateEvidenceError):
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
                        if not isinstance(choice, dict) or choice.get("index") != 0:
                            error_type = error_type or "InvalidChoice"
                            continue
                        text = choice.get("text")
                        if not isinstance(text, str):
                            error_type = error_type or "InvalidChoiceText"
                            continue
                        if first_token_ns is None:
                            first_token_ns = end_ns
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
    except Exception as error:  # retain one failed attempt as evidence
        error_type = error_type or type(error).__name__
    end_ns = max(end_ns, time.perf_counter_ns())
    if len(response_ids) != 1:
        error_type = error_type or "ResponseIDMismatch"
    if done_events != 1:
        error_type = error_type or "DONECountMismatch"
    if terminal_choice_events != 1 or not terminal_seen:
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
    return {
        "type": "request",
        "http_client_role": client.role,
        "http_client_request_ordinal": http_client_request_ordinal,
        "phase": planned.phase,
        "sequence": planned.sequence,
        "attempts": 1,
        "preflight_repeat": planned.preflight_repeat,
        "graph_batch_size": planned.graph_batch_size,
        "graph_batch_sequence": planned.graph_batch_sequence,
        "open_loop_point": planned.open_loop_point,
        "rate_millirps": planned.rate_millirps,
        "prompt_text_sha256": planned.prompt_text_sha256,
        "prompt_token_ids_sha256": planned.prompt_token_ids_sha256,
        "prompt_tokens": len(planned.prompt_token_ids),
        "expected_completion_tokens": planned.expected_completion_tokens,
        "http_status": http_status,
        "content_type": content_type,
        "response_request_header": response_request_header,
        "response_id": next(iter(response_ids)) if len(response_ids) == 1 else None,
        "status": "success" if success else "failure",
        "error_type": error_type,
        "done_events": done_events,
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
        "output_text_sha256": sha256_text("".join(output_parts)),
        "stream_payload_sha256": sha256_text("\n".join(stream_events)),
    }


async def _execute_serial(
    client: _TrackedAsyncClient,
    *,
    endpoint: str,
    planned: Sequence[PlannedRequest],
) -> list[dict[str, object]]:
    result = []
    for request in planned:
        result.append(await stream_completion_once(client, endpoint=endpoint, planned=request))
    return result


async def _execute_synchronized(
    client: _TrackedAsyncClient,
    *,
    endpoint: str,
    planned: Sequence[PlannedRequest],
    expected_size: int,
    release_field: str,
) -> list[dict[str, object]]:
    if len(planned) != expected_size:
        raise GateEvidenceError(f"synchronized group requires exactly {expected_size} requests")
    release = asyncio.Event()

    async def one(request: PlannedRequest) -> dict[str, object]:
        await release.wait()
        return await stream_completion_once(client, endpoint=endpoint, planned=request)

    tasks = [asyncio.create_task(one(request)) for request in planned]
    await asyncio.sleep(0)
    release_ns = time.perf_counter_ns()
    release.set()
    return [{**row, release_field: release_ns} for row in await asyncio.gather(*tasks)]


async def _execute_graph_warmups(
    client: _TrackedAsyncClient,
    *,
    endpoint: str,
    planned: Sequence[PlannedRequest],
) -> list[dict[str, object]]:
    if len(planned) != GRAPH_WARMUP_REQUESTS:
        raise GateEvidenceError(
            f"graph warmup requires exactly {GRAPH_WARMUP_REQUESTS} requests"
        )
    result: list[dict[str, object]] = []
    offset = 0
    for batch_size in GRAPH_WARMUP_BATCH_SIZES:
        batch = planned[offset : offset + batch_size]
        if [request.graph_batch_sequence for request in batch] != list(range(batch_size)):
            raise GateEvidenceError("graph warmup batch sequence is invalid")
        result.extend(
            await _execute_synchronized(
                client,
                endpoint=endpoint,
                planned=batch,
                expected_size=batch_size,
                release_field="graph_release_ns",
            )
        )
        offset += batch_size
    return result


async def _execute_open_loop(
    client: _TrackedAsyncClient,
    *,
    endpoint: str,
    planned: Sequence[PlannedRequest],
    rate_millirps: int,
) -> list[dict[str, object]]:
    if rate_millirps <= 0 or len(planned) != OPEN_LOOP_REQUESTS_PER_POINT:
        raise GateEvidenceError("invalid open-loop point")
    period_ns = 1_000_000_000_000 // rate_millirps
    anchor_ns = time.perf_counter_ns() + 10_000_000

    async def one(position: int, request: PlannedRequest) -> dict[str, object]:
        scheduled_ns = anchor_ns + position * period_ns
        delay = (scheduled_ns - time.perf_counter_ns()) / 1_000_000_000
        if delay > 0:
            await asyncio.sleep(delay)
        row = await stream_completion_once(client, endpoint=endpoint, planned=request)
        timing = row.get("timing_ns")
        actual_start = timing.get("start") if isinstance(timing, dict) else None
        return {
            **row,
            "scheduled_start_ns": scheduled_ns,
            "scheduled_offset_ns": position * period_ns,
            "release_lag_ns": actual_start - scheduled_ns if type(actual_start) is int else None,
        }

    return list(
        await asyncio.gather(*(one(position, request) for position, request in enumerate(planned)))
    )


def _normalize_endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("endpoint must start with http:// or https://")
    return endpoint


async def _assert_served_model(client: httpx.AsyncClient, endpoint: str) -> None:
    response = await client.get(f"{endpoint}/v1/models")
    if response.status_code != 200:
        raise RuntimeError(f"/v1/models returned HTTP {response.status_code}")
    payload = response.json()
    available = (
        {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
        if isinstance(payload, dict)
        else set()
    )
    if SERVED_MODEL_NAME not in available:
        raise GateEvidenceError(f"endpoint does not serve pinned target {SERVED_MODEL_NAME}")


def _kairyu_graph_witness_shape_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _KAIRYU_GRAPH_WITNESS_KEYS:
        return False
    transport = value["moe_collective_transport"]
    graph_buckets = value["cuda_graph_buckets"]
    string_fields = (
        "model",
        "engine_backend",
        "parallelism",
        "attention_placement",
        "attention_output_placement",
        "execution_mode",
        "decode_mode",
        "kv_cache_dtype",
        "moe_dispatcher",
        "sampling_ownership",
        "kv_cache_ownership",
    )
    integer_fields = (
        "expert_parallel_size",
        "attention_output_parallel_size",
        "pipeline_depth",
        "attention_data_parallel_size",
        "attention_tensor_parallel_size",
        "cuda_graph_captures",
        "cuda_graph_replays",
        "cuda_graph_eager_fallbacks",
    )
    return (
        all(isinstance(value[field], str) and bool(value[field]) for field in string_fields)
        and all(type(value[field]) is int and value[field] >= 0 for field in integer_fields)
        and (
            value["attention_output_partial_dtype"] is None
            or isinstance(value["attention_output_partial_dtype"], str)
        )
        and type(value["cuda_graph_decode"]) is bool
        and isinstance(graph_buckets, list)
        and bool(graph_buckets)
        and all(type(bucket) is int and bucket > 0 for bucket in graph_buckets)
        and graph_buckets == sorted(set(graph_buckets))
        and isinstance(transport, dict)
        and set(transport) == _KAIRYU_TRANSPORT_WITNESS_KEYS
        and isinstance(transport["selected_backend"], str)
        and bool(transport["selected_backend"])
        and isinstance(transport["fallback_backend"], str)
        and bool(transport["fallback_backend"])
        and type(transport["direct_nccl_active"]) is bool
        and (
            transport["direct_nccl_library"] is None
            or (
                isinstance(transport["direct_nccl_library"], str)
                and bool(transport["direct_nccl_library"])
            )
        )
        and (
            transport["direct_nccl_version"] is None
            or (
                isinstance(transport["direct_nccl_version"], str)
                and bool(transport["direct_nccl_version"])
            )
        )
        and isinstance(transport["selection_reason"], str)
        and bool(transport["selection_reason"])
    )


def _kairyu_graph_witness_start_valid(value: object) -> bool:
    if not _kairyu_graph_witness_shape_valid(value) or not isinstance(value, dict):
        return False
    transport = value["moe_collective_transport"]
    assert isinstance(transport, dict)
    return (
        value["model"] == SERVED_MODEL_NAME
        and value["engine_backend"] == "kairyu"
        and value["parallelism"] == "expert_parallel"
        and value["expert_parallel_size"] == GPU_COUNT
        and value["attention_placement"] == "request_owned_data_parallel"
        and value["attention_output_placement"] == "replicated"
        and value["attention_output_parallel_size"] == 1
        and value["attention_output_partial_dtype"] is None
        and value["execution_mode"] == "request-owned-attention-dp"
        and value["pipeline_depth"] == 5
        and value["decode_mode"] == "cuda_graph"
        and value["kv_cache_dtype"] == KV_CACHE_DTYPE
        and value["attention_data_parallel_size"] == GPU_COUNT
        and value["attention_tensor_parallel_size"] == 1
        and value["moe_dispatcher"] == "nvfp4_allgather_reduce_scatter"
        and value["sampling_ownership"] == "request_owner"
        and value["kv_cache_ownership"] == "request_owner"
        and value["cuda_graph_decode"] is True
        and value["cuda_graph_buckets"] == list(KAIRYU_CUDA_GRAPH_BUCKETS)
        and value["cuda_graph_captures"] == len(KAIRYU_CUDA_GRAPH_BUCKETS)
        and value["cuda_graph_eager_fallbacks"] == 0
        and transport["selected_backend"] == "direct_nccl_ctypes"
        and transport["fallback_backend"] == "torch.distributed:nccl"
        and transport["direct_nccl_active"] is True
        and isinstance(transport["direct_nccl_library"], str)
        and bool(transport["direct_nccl_library"])
        and isinstance(transport["direct_nccl_version"], str)
        and bool(transport["direct_nccl_version"])
    )


def _kairyu_graph_witness_pair_valid(start: object, end: object) -> bool:
    if (
        not _kairyu_graph_witness_start_valid(start)
        or not _kairyu_graph_witness_start_valid(end)
        or not isinstance(start, dict)
        or not isinstance(end, dict)
    ):
        return False
    if end["cuda_graph_replays"] <= start["cuda_graph_replays"]:
        return False
    return all(
        start[key] == end[key]
        for key in _KAIRYU_GRAPH_WITNESS_KEYS - {"cuda_graph_replays"}
    )


def _dynamic_graph_witness_shape_valid(arm: object, start: object, end: object) -> bool:
    if arm == "sglang":
        return start is None and end is None
    return (
        arm == "kairyu"
        and _kairyu_graph_witness_shape_valid(start)
        and _kairyu_graph_witness_shape_valid(end)
    )


def _dynamic_graph_witness_valid(arm: object, start: object, end: object) -> bool:
    if arm == "sglang":
        return start is None and end is None
    return arm == "kairyu" and _kairyu_graph_witness_pair_valid(start, end)


async def _fetch_kairyu_graph_witness(
    client: httpx.AsyncClient,
    endpoint: str,
) -> dict[str, object]:
    response = await client.get(f"{endpoint}/backends")
    if response.status_code != 200:
        raise GateEvidenceError(f"/backends returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except (TypeError, ValueError) as error:
        raise GateEvidenceError("/backends returned invalid JSON") from error
    engines = payload.get("engines") if isinstance(payload, dict) else None
    if not isinstance(engines, list):
        raise GateEvidenceError("/backends response has no engine list")
    matches = [
        engine
        for engine in engines
        if isinstance(engine, dict) and engine.get("model") == SERVED_MODEL_NAME
    ]
    if len(matches) != 1:
        raise GateEvidenceError("/backends must contain exactly one pinned target engine")
    engine = matches[0]
    witness = {key: engine.get(key) for key in _KAIRYU_GRAPH_WITNESS_KEYS}
    transport = witness.get("moe_collective_transport")
    if isinstance(transport, dict):
        witness["moe_collective_transport"] = dict(transport)
    buckets = witness.get("cuda_graph_buckets")
    if isinstance(buckets, list):
        witness["cuda_graph_buckets"] = list(buckets)
    if not _kairyu_graph_witness_shape_valid(witness):
        raise GateEvidenceError("pinned target /backends witness is incomplete or malformed")
    return witness


def _selection_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "trace_bundle_sha256",
        "moe_a2a_backend",
        "preflight_raw_sha256",
        "preflight_completed_ns",
        "selected",
        "candidate_metrics",
    }:
        return False
    selected = value["selected"]
    metrics = value["candidate_metrics"]
    hashes = value["preflight_raw_sha256"]
    expected_ids = {
        f"preflight-{arm}-{candidate}" for arm in ARMS for candidate in PREFLIGHT_CANDIDATES[arm]
    }
    if not (
        value["schema_version"] == SELECTION_SCHEMA_VERSION
        and isinstance(value["trace_bundle_sha256"], str)
        and _HEX64.fullmatch(value["trace_bundle_sha256"]) is not None
        and isinstance(value["moe_a2a_backend"], str)
        and bool(value["moe_a2a_backend"])
        and isinstance(hashes, dict)
        and set(hashes) == expected_ids
        and all(
            isinstance(item, str) and _HEX64.fullmatch(item) is not None for item in hashes.values()
        )
        and type(value["preflight_completed_ns"]) is int
        and value["preflight_completed_ns"] > 0
        and isinstance(selected, dict)
        and set(selected) == set(ARMS)
        and selected["kairyu"] in KAIRYU_PREFLIGHT_CANDIDATES
        and selected["sglang"] in SGLANG_PREFLIGHT_CANDIDATES
        and isinstance(metrics, dict)
        and set(metrics) == set(ARMS)
    ):
        return False
    for arm in ARMS:
        arm_metrics = metrics[arm]
        if not isinstance(arm_metrics, dict) or set(arm_metrics) != set(PREFLIGHT_CANDIDATES[arm]):
            return False
        try:
            fractions = {
                candidate: _fraction_from_record(record)
                for candidate, record in arm_metrics.items()
            }
        except GateEvidenceError:
            return False
        expected_selected = max(
            PREFLIGHT_CANDIDATES[arm],
            key=lambda candidate: (
                fractions[candidate],
                -PREFLIGHT_CANDIDATES[arm].index(candidate),
            ),
        )
        if selected[arm] != expected_selected:
            return False
    return True


def load_selection(path: Path) -> dict[str, object]:
    value = _read_json_object(path)
    if not _selection_valid(value):
        raise GateEvidenceError(f"{path} does not satisfy the frozen preflight selection schema")
    return hashed_descriptor(value)


def _scenario_id(
    *,
    phase: str,
    arm: str,
    candidate: str,
    repeat: int | None,
) -> str:
    if arm not in ARMS:
        raise GateEvidenceError(f"unknown arm {arm!r}")
    if phase == "preflight":
        return f"preflight-{arm}-{candidate}"
    if phase == "formal":
        if type(repeat) is not int or not 0 <= repeat < REPEATS:
            raise GateEvidenceError("formal repeat must be 0..3")
        return f"formal-r{repeat}-{arm}"
    if phase == "open-loop":
        return f"open-loop-{arm}"
    raise GateEvidenceError(f"unknown run phase {phase!r}")


def _selected_candidate(selection: Mapping[str, object], arm: str) -> str:
    raw = selection.get("value")
    if not isinstance(raw, dict) or not isinstance(raw.get("selected"), dict):
        raise GateEvidenceError("selection descriptor is invalid")
    candidate = raw["selected"].get(arm)
    if not isinstance(candidate, str):
        raise GateEvidenceError("selection has no candidate for arm")
    return candidate


async def measure_shard(
    *,
    phase: str,
    arm: str,
    endpoint: str,
    trace_bundle_path: Path,
    provenance_path: Path,
    dataset_sha256: str,
    candidate: str | None = None,
    repeat: int | None = None,
    selection_path: Path | None = None,
    api_key: str | None = None,
    request_timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[dict[str, object]]:
    if arm not in ARMS or phase not in {"preflight", "formal"}:
        raise GateEvidenceError("invalid run arm/phase")
    if request_timeout_s <= 0:
        raise GateEvidenceError("request timeout must be positive")
    bundle = load_trace_bundle(trace_bundle_path, dataset_sha256=dataset_sha256)
    benchmark = bundle["benchmark"]
    assert isinstance(benchmark, dict)
    sglang = benchmark["sglang"]
    assert isinstance(sglang, dict)
    moe_a2a_backend = str(sglang["moe_a2a_backend"])
    selection: dict[str, object] | None = None
    if phase == "preflight":
        if (
            candidate not in PREFLIGHT_CANDIDATES[arm]
            or repeat is not None
            or selection_path is not None
        ):
            raise GateEvidenceError(
                "preflight requires one declared candidate and no repeat/selection"
            )
    else:
        if candidate is not None or selection_path is None:
            raise GateEvidenceError("formal requires a frozen selection and no candidate")
        selection = load_selection(selection_path)
        selection_raw = selection["value"]
        assert isinstance(selection_raw, dict)
        if (
            selection_raw["trace_bundle_sha256"] != sha256_file(trace_bundle_path)
            or selection_raw["moe_a2a_backend"] != moe_a2a_backend
        ):
            raise GateEvidenceError("selection does not belong to this trace bundle")
        candidate = _selected_candidate(selection, arm)
    assert candidate is not None
    scenario_id = _scenario_id(phase=phase, arm=arm, candidate=candidate, repeat=repeat)
    endpoint = _normalize_endpoint(endpoint)
    if endpoint != FORMAL_ENDPOINT:
        raise GateEvidenceError(
            f"run endpoint must be the captured loopback server {FORMAL_ENDPOINT}"
        )
    provenance_start = load_provenance(
        provenance_path,
        arm=arm,
        candidate=candidate,
        moe_a2a_backend=moe_a2a_backend,
    )
    provenance_raw = provenance_start["value"]
    assert isinstance(provenance_raw, dict)
    if selection is not None:
        selection_raw = selection["value"]
        assert isinstance(selection_raw, dict)
        if provenance_raw["server_started_ns"] <= selection_raw["preflight_completed_ns"]:
            raise GateEvidenceError(
                "formal server predates the frozen preflight selection"
            )
    capture_start_ns = time.perf_counter_ns()
    rows: list[dict[str, object]] = [
        {
            "type": "shard_start",
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "scenario_id": scenario_id,
            "phase": phase,
            "arm": arm,
            "candidate": candidate,
            "repeat": repeat,
            "endpoint": endpoint,
            "trace_bundle_sha256": sha256_file(trace_bundle_path),
            "benchmark": benchmark,
            "trace": bundle["trace"],
            "graph_warmup": bundle["graph_warmup"],
            "selection": selection,
            "provenance_start": provenance_start,
            "kairyu_graph_witness_start": None,
            "capture_start_ns": capture_start_ns,
        }
    ]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(
        max_connections=SHAREGPT_CONCURRENCY,
        max_keepalive_connections=SHAREGPT_CONCURRENCY,
    )
    measurement_groups: list[dict[str, int]] = []
    graph_witness_start: dict[str, object] | None = None
    graph_witness_end: dict[str, object] | None = None
    warmup_client = _TrackedAsyncClient(
        role="warmup",
        timeout=request_timeout_s,
        headers=headers,
        limits=limits,
    )
    async with warmup_client:
        await _assert_served_model(warmup_client, endpoint)
        warmup_rows = await _execute_serial(
            warmup_client,
            endpoint=endpoint,
            planned=_serial_warmup_plan(bundle["trace"]),
        )
        graph_rows = await _execute_graph_warmups(
            warmup_client,
            endpoint=endpoint,
            planned=_graph_warmup_plan(bundle["graph_warmup"]),
        )
        if arm == "kairyu":
            graph_witness_start = await _fetch_kairyu_graph_witness(warmup_client, endpoint)
            if not _kairyu_graph_witness_start_valid(graph_witness_start):
                raise GateEvidenceError(
                    "Kairyu graph warmup did not establish the pinned graph/direct-NCCL state"
                )
            rows[0]["kairyu_graph_witness_start"] = graph_witness_start
    warmup_client_evidence = warmup_client.evidence()

    measurement_client = _TrackedAsyncClient(
        role="measurement",
        timeout=request_timeout_s,
        headers=headers,
        limits=limits,
    )
    measurement_requests_before_group = measurement_client.request_count
    async with measurement_client:
        measured: list[dict[str, object]] = []
        if phase == "preflight":
            for preflight_repeat, plan in enumerate(_preflight_plans(bundle["trace"])):
                group = await _execute_synchronized(
                    measurement_client,
                    endpoint=endpoint,
                    planned=plan,
                    expected_size=PREFLIGHT_REQUESTS,
                    release_field="burst_release_ns",
                )
                measured.extend(group)
                timings = [
                    row["timing_ns"] for row in group if isinstance(row.get("timing_ns"), dict)
                ]
                measurement_groups.append(
                    {
                        "group": preflight_repeat,
                        "start": min(int(item["start"]) for item in timings),
                        "end": max(int(item["end"]) for item in timings),
                    }
                )
        else:
            group = await _execute_synchronized(
                measurement_client,
                endpoint=endpoint,
                planned=_formal_plan(bundle["trace"]),
                expected_size=SHAREGPT_REQUESTS,
                release_field="burst_release_ns",
            )
            measured.extend(group)
            timings = [row["timing_ns"] for row in group if isinstance(row.get("timing_ns"), dict)]
            measurement_groups.append(
                {
                    "group": 0,
                    "start": min(int(item["start"]) for item in timings),
                    "end": max(int(item["end"]) for item in timings),
                }
            )
        if arm == "kairyu":
            graph_witness_end = await _fetch_kairyu_graph_witness(measurement_client, endpoint)
            if not _kairyu_graph_witness_pair_valid(graph_witness_start, graph_witness_end):
                raise GateEvidenceError(
                    "Kairyu measurement changed graph/direct-NCCL state or replayed no graph"
                )
    measurement_client_evidence = measurement_client.evidence()
    client_lifecycle = {
        "schema_version": HTTP_CLIENT_LIFECYCLE_SCHEMA_VERSION,
        "distinct_client_instances": warmup_client is not measurement_client,
        "measurement_requests_before_group": measurement_requests_before_group,
        "warmup_client": warmup_client_evidence,
        "measurement_client": measurement_client_evidence,
    }
    for row in (*warmup_rows, *graph_rows, *measured):
        rows.append(
            {
                **row,
                "scenario_id": scenario_id,
                "run_phase": phase,
                "arm": arm,
                "candidate": candidate,
                "repeat": repeat,
            }
        )
    provenance_reloaded = load_provenance(
        provenance_path,
        arm=arm,
        candidate=candidate,
        moe_a2a_backend=moe_a2a_backend,
    )
    if provenance_reloaded != provenance_start:
        raise GateEvidenceError("start provenance changed during shard traffic")
    provenance_end = hashed_descriptor(capture_live_observation(provenance_raw))
    capture_end_ns = time.perf_counter_ns()
    rows.append(
        {
            "type": "shard_end",
            "scenario_id": scenario_id,
            "phase": phase,
            "arm": arm,
            "candidate": candidate,
            "repeat": repeat,
            "serial_warmup_request_count": len(warmup_rows),
            "graph_warmup_request_count": len(graph_rows),
            "measurement_request_count": len(measured),
            "measurement_groups": measurement_groups,
            "http_client_lifecycle": client_lifecycle,
            "provenance_end": provenance_end,
            "kairyu_graph_witness_end": graph_witness_end,
            "capture_end_ns": capture_end_ns,
        }
    )
    return rows


def write_shard(path: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, rows)
    stored = _read_jsonl(path)
    start, requests, end = _parse_shard(stored)
    successful = sum(row.get("status") == "success" for row in requests)
    provenance_start = start["provenance_start"]
    provenance_end = end["provenance_end"]
    assert isinstance(provenance_start, dict) and isinstance(provenance_start["value"], dict)
    assert isinstance(provenance_end, dict)
    provenance_stable = _live_observation_value_valid(
        provenance_end["value"], provenance=provenance_start["value"]
    )
    graph_witness_valid = _dynamic_graph_witness_valid(
        start["arm"],
        start["kairyu_graph_witness_start"],
        end["kairyu_graph_witness_end"],
    )
    return {
        "scenario_id": start["scenario_id"],
        "phase": start["phase"],
        "request_count": len(requests),
        "successful_request_count": successful,
        "provenance_stable": provenance_stable,
        "dynamic_graph_witness_valid": graph_witness_valid,
        "passed": successful == len(requests) and provenance_stable and graph_witness_valid,
        "sha256": sha256_file(path),
    }


_REQUEST_BASE_KEYS = {
    "type",
    "http_client_role",
    "http_client_request_ordinal",
    "phase",
    "sequence",
    "attempts",
    "preflight_repeat",
    "graph_batch_size",
    "graph_batch_sequence",
    "open_loop_point",
    "rate_millirps",
    "prompt_text_sha256",
    "prompt_token_ids_sha256",
    "prompt_tokens",
    "expected_completion_tokens",
    "http_status",
    "content_type",
    "response_request_header",
    "response_id",
    "status",
    "error_type",
    "done_events",
    "terminal_choice_events",
    "finish_reasons",
    "usage_events",
    "usage",
    "timing_ns",
    "output_text_sha256",
    "stream_payload_sha256",
    "scenario_id",
    "run_phase",
    "arm",
    "candidate",
    "repeat",
    "row_index",
}
_REQUEST_OPTIONAL_KEYS = {
    "burst_release_ns",
    "graph_release_ns",
    "scheduled_start_ns",
    "scheduled_offset_ns",
    "release_lag_ns",
}
_SHARD_START_KEYS = {
    "type",
    "schema_version",
    "gate",
    "scenario_id",
    "phase",
    "arm",
    "candidate",
    "repeat",
    "endpoint",
    "trace_bundle_sha256",
    "benchmark",
    "trace",
    "graph_warmup",
    "selection",
    "provenance_start",
    "kairyu_graph_witness_start",
    "capture_start_ns",
    "row_index",
}
_SHARD_END_KEYS = {
    "type",
    "scenario_id",
    "phase",
    "arm",
    "candidate",
    "repeat",
    "serial_warmup_request_count",
    "graph_warmup_request_count",
    "measurement_request_count",
    "measurement_groups",
    "http_client_lifecycle",
    "provenance_end",
    "kairyu_graph_witness_end",
    "capture_end_ns",
    "row_index",
}
_REQUEST_CAPTURE_TIMESTAMPS = (
    "burst_release_ns",
    "graph_release_ns",
    "scheduled_start_ns",
)


def _http_client_snapshot_valid(
    value: object,
    *,
    role: str,
    expected_paths: Sequence[str],
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "created_ns",
        "closed_ns",
        "request_count",
        "request_path_counts",
        "request_path_sequence_sha256",
    }:
        return False
    expected_counts = {path: expected_paths.count(path) for path in sorted(set(expected_paths))}
    return (
        value["role"] == role
        and type(value["created_ns"]) is int
        and type(value["closed_ns"]) is int
        and 0 < value["created_ns"] <= value["closed_ns"]
        and value["request_count"] == len(expected_paths)
        and value["request_path_counts"] == expected_counts
        and value["request_path_sequence_sha256"] == sha256_json(list(expected_paths))
    )


def _http_client_lifecycle_valid(
    value: object,
    *,
    arm: str,
    phase: str,
    requests: Sequence[Mapping[str, object]],
    capture_start_ns: int,
    end_observation_started_ns: int,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "distinct_client_instances",
        "measurement_requests_before_group",
        "warmup_client",
        "measurement_client",
    }:
        return False
    measured_phase = {
        "preflight": "preflight",
        "formal": "formal",
        "open-loop": "open_loop",
    }.get(phase)
    if measured_phase is None:
        return False
    serial_rows = [row for row in requests if row.get("phase") == "serial_warmup"]
    graph_rows = [row for row in requests if row.get("phase") == "graph_warmup"]
    warmup_rows = [*serial_rows, *graph_rows]
    measurement_rows = [row for row in requests if row.get("phase") == measured_phase]
    if len(warmup_rows) != SERIAL_WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS:
        return False
    if len(warmup_rows) + len(measurement_rows) != len(requests):
        return False
    witness_paths = ["/backends"] if arm == "kairyu" else []
    warmup_paths = [
        "/v1/models",
        *["/v1/completions"] * len(warmup_rows),
        *witness_paths,
    ]
    measurement_paths = [
        *["/v1/completions"] * len(measurement_rows),
        *witness_paths,
    ]
    warmup_client = value["warmup_client"]
    measurement_client = value["measurement_client"]
    if not (
        value["schema_version"] == HTTP_CLIENT_LIFECYCLE_SCHEMA_VERSION
        and value["distinct_client_instances"] is True
        and value["measurement_requests_before_group"] == 0
        and _http_client_snapshot_valid(
            warmup_client,
            role="warmup",
            expected_paths=warmup_paths,
        )
        and _http_client_snapshot_valid(
            measurement_client,
            role="measurement",
            expected_paths=measurement_paths,
        )
    ):
        return False
    assert isinstance(warmup_client, dict) and isinstance(measurement_client, dict)
    serial_ordinals = [
        row.get("http_client_request_ordinal")
        for row in sorted(
            serial_rows,
            key=lambda row: (
                int(row["timing_ns"]["start"])
                if isinstance(row.get("timing_ns"), dict)
                and type(row["timing_ns"].get("start")) is int
                else -1
            ),
        )
    ]
    graph_ordinals = [row.get("http_client_request_ordinal") for row in graph_rows]
    measurement_ordinals = [
        row.get("http_client_request_ordinal") for row in measurement_rows
    ]
    if any(
        type(ordinal) is not int
        for ordinal in (*serial_ordinals, *graph_ordinals, *measurement_ordinals)
    ):
        return False
    if (
        [row.get("http_client_role") for row in warmup_rows]
        != ["warmup"] * len(warmup_rows)
        or serial_ordinals != list(range(1, 1 + SERIAL_WARMUP_REQUESTS))
    ):
        return False
    next_graph_ordinal = 1 + SERIAL_WARMUP_REQUESTS
    for batch_size in GRAPH_WARMUP_BATCH_SIZES:
        batch_ordinals = [
            row.get("http_client_request_ordinal")
            for row in graph_rows
            if row.get("graph_batch_size") == batch_size
        ]
        expected_ordinals = list(
            range(next_graph_ordinal, next_graph_ordinal + batch_size)
        )
        if sorted(batch_ordinals) != expected_ordinals:
            return False
        next_graph_ordinal += batch_size
    if [row.get("http_client_role") for row in measurement_rows] != [
        "measurement"
    ] * len(measurement_rows) or sorted(measurement_ordinals) != list(
        range(len(measurement_rows))
    ):
        return False
    warmup_timings = [row.get("timing_ns") for row in warmup_rows]
    measurement_timings = [row.get("timing_ns") for row in measurement_rows]
    if not all(isinstance(item, dict) for item in (*warmup_timings, *measurement_timings)):
        return False
    warmup_created = int(warmup_client["created_ns"])
    warmup_closed = int(warmup_client["closed_ns"])
    measurement_created = int(measurement_client["created_ns"])
    measurement_closed = int(measurement_client["closed_ns"])
    return (
        capture_start_ns <= warmup_created
        and all(
            warmup_created <= int(item["start"]) <= int(item["end"]) <= warmup_closed
            for item in warmup_timings
            if isinstance(item, dict)
        )
        and warmup_closed < measurement_created
        and all(
            measurement_created
            <= int(item["start"])
            <= int(item["end"])
            <= measurement_closed
            for item in measurement_timings
            if isinstance(item, dict)
        )
        and measurement_closed <= end_observation_started_ns
    )


def _parse_shard(
    rows: Sequence[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if len(rows) < 2 or rows[0].get("type") != "shard_start" or rows[-1].get("type") != "shard_end":
        raise GateEvidenceError("shard must be framed by shard_start/shard_end")
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise GateEvidenceError("shard row indices are not contiguous")
    start, end = rows[0], rows[-1]
    start_keys = set(start)
    if start_keys != _SHARD_START_KEYS and start_keys != _SHARD_START_KEYS | {"shard_sha256"}:
        raise GateEvidenceError("shard start has unknown or missing fields")
    if set(end) != _SHARD_END_KEYS:
        raise GateEvidenceError("shard end has unknown or missing fields")
    if start.get("schema_version") != SCHEMA_VERSION or start.get("gate") != GATE:
        raise GateEvidenceError("shard schema/gate mismatch")
    if start.get("endpoint") != FORMAL_ENDPOINT:
        raise GateEvidenceError("shard endpoint is not the captured loopback server")
    phase = start.get("phase")
    arm = start.get("arm")
    candidate = start.get("candidate")
    repeat = start.get("repeat")
    if not isinstance(phase, str) or not isinstance(arm, str) or not isinstance(candidate, str):
        raise GateEvidenceError("shard scenario identity has wrong types")
    scenario_id = _scenario_id(phase=phase, arm=arm, candidate=candidate, repeat=repeat)
    if start.get("scenario_id") != scenario_id:
        raise GateEvidenceError("shard scenario ID mismatch")
    for key in ("scenario_id", "phase", "arm", "candidate", "repeat"):
        if end.get(key) != start.get(key):
            raise GateEvidenceError(f"shard end changed {key}")
    if not _dynamic_graph_witness_shape_valid(
        arm,
        start.get("kairyu_graph_witness_start"),
        end.get("kairyu_graph_witness_end"),
    ):
        raise GateEvidenceError("shard graph/direct-NCCL witness has an invalid shape")
    if phase == "preflight":
        if candidate not in PREFLIGHT_CANDIDATES.get(arm, ()) or start.get("selection") is not None:
            raise GateEvidenceError("preflight shard candidate/selection mismatch")
    else:
        selection = start.get("selection")
        if not _descriptor_valid(selection):
            raise GateEvidenceError("formal shard lacks a hashed selection")
        assert isinstance(selection, dict)
        if (
            not _selection_valid(selection["value"])
            or _selected_candidate(selection, arm) != candidate
        ):
            raise GateEvidenceError("shard candidate differs from frozen selection")
    benchmark = start.get("benchmark")
    sglang = benchmark.get("sglang") if isinstance(benchmark, dict) else None
    moe_a2a_backend = sglang.get("moe_a2a_backend") if isinstance(sglang, dict) else None
    sharegpt = benchmark.get("sharegpt") if isinstance(benchmark, dict) else None
    dataset_sha256 = sharegpt.get("sha256") if isinstance(sharegpt, dict) else None
    trace = start.get("trace")
    graph_warmup = start.get("graph_warmup")
    if not isinstance(moe_a2a_backend, str) or not isinstance(dataset_sha256, str):
        raise GateEvidenceError("shard benchmark has no pinned dataset/backend identity")
    try:
        expected_benchmark = benchmark_config(
            dataset_sha256,
            moe_a2a_backend=moe_a2a_backend,
        )
    except ValueError as error:
        raise GateEvidenceError("shard benchmark identity is invalid") from error
    trace_bundle = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "benchmark": benchmark,
        "trace": trace,
        "graph_warmup": graph_warmup,
    }
    if not (
        benchmark == expected_benchmark
        and a6._trace_valid(trace, workload="sharegpt", dataset_sha256=dataset_sha256)
        and _graph_warmup_trace_valid(graph_warmup)
        and _graph_warmup_disjoint(graph_warmup, trace)
        and start.get("trace_bundle_sha256")
        == sha256_text(canonical_json(trace_bundle) + "\n")
    ):
        raise GateEvidenceError("shard benchmark/trace bundle violates the v2 contract")
    provenance_start = start["provenance_start"]
    provenance_end = end["provenance_end"]
    if not _descriptor_valid(provenance_start) or not isinstance(provenance_start, dict):
        raise GateEvidenceError("provenance_start is not a hashed descriptor")
    if not _provenance_value_valid(
        provenance_start["value"],
        arm=arm,
        candidate=candidate,
        moe_a2a_backend=moe_a2a_backend,
    ):
        raise GateEvidenceError("provenance_start violates the provenance contract")
    if not _descriptor_valid(provenance_end) or not isinstance(provenance_end, dict):
        raise GateEvidenceError("provenance_end is not a hashed descriptor")
    start_value = provenance_start["value"]
    assert isinstance(start_value, dict)
    if not _live_observation_value_valid(provenance_end["value"], provenance=start_value):
        raise GateEvidenceError("provenance_end violates the live observation contract")
    requests = list(rows[1:-1])
    for request in requests:
        if request.get("type") != "request":
            raise GateEvidenceError(f"unknown shard row type {request.get('type')!r}")
        unknown = set(request) - _REQUEST_BASE_KEYS - _REQUEST_OPTIONAL_KEYS
        missing = _REQUEST_BASE_KEYS - set(request)
        if unknown or missing:
            raise GateEvidenceError(
                f"request row fields differ: missing={missing}, unknown={unknown}"
            )
        for key in ("scenario_id", "arm", "candidate", "repeat"):
            if request.get(key) != start.get(key):
                raise GateEvidenceError(f"request changed scenario field {key}")
        if request.get("run_phase") != phase:
            raise GateEvidenceError("request changed run phase")
    expected_measurements = {
        "preflight": PREFLIGHT_REPEATS * PREFLIGHT_REQUESTS,
        "formal": SHAREGPT_REQUESTS,
        "open-loop": OPEN_LOOP_POINTS * OPEN_LOOP_REQUESTS_PER_POINT,
    }[phase]
    if (
        end.get("serial_warmup_request_count") != SERIAL_WARMUP_REQUESTS
        or end.get("graph_warmup_request_count") != GRAPH_WARMUP_REQUESTS
        or end.get("measurement_request_count") != expected_measurements
        or len(requests) != SERIAL_WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS + expected_measurements
    ):
        raise GateEvidenceError("shard request counts do not match the declared phase")
    capture_start = start.get("capture_start_ns")
    capture_end = end.get("capture_end_ns")
    server_started = start_value["server_started_ns"]
    end_value = provenance_end["value"]
    assert isinstance(end_value, dict)
    end_observation_started = end_value["capture_started_ns"]
    end_observation_completed = end_value["capture_completed_ns"]
    if not (
        type(capture_start) is int
        and type(capture_end) is int
        and type(server_started) is int
        and type(end_observation_started) is int
        and type(end_observation_completed) is int
        and server_started <= capture_start < end_observation_started
        < end_observation_completed <= capture_end
    ):
        raise GateEvidenceError("shard capture/server times are inconsistent")
    assert isinstance(capture_start, int) and isinstance(capture_end, int)
    for request in requests:
        timing = request.get("timing_ns")
        if (
            not isinstance(timing, dict)
            or set(timing) != {"start", "first_token", "end", "ttft", "total"}
            or any(type(timing.get(key)) is not int for key in timing)
            or not (
                capture_start
                <= timing["start"]
                <= timing["first_token"]
                <= timing["end"]
                <= end_observation_started
            )
        ):
            raise GateEvidenceError("request timing is outside the shard capture interval")
        for key in _REQUEST_CAPTURE_TIMESTAMPS:
            if key in request and (
                type(request[key]) is not int
                or not capture_start <= request[key] <= timing["start"]
            ):
                raise GateEvidenceError(
                    f"request {key} is outside the shard capture interval"
                )
    measurement_groups = end.get("measurement_groups")
    if not isinstance(measurement_groups, list) or any(
        not isinstance(group, dict)
        or set(group) != {"group", "start", "end"}
        or type(group["group"]) is not int
        or type(group["start"]) is not int
        or type(group["end"]) is not int
        or not capture_start
        <= group["start"]
        <= group["end"]
        <= end_observation_started
        for group in measurement_groups
    ):
        raise GateEvidenceError("measurement group is outside the shard capture interval")
    if not _http_client_lifecycle_valid(
        end.get("http_client_lifecycle"),
        arm=arm,
        phase=phase,
        requests=requests,
        capture_start_ns=capture_start,
        end_observation_started_ns=end_observation_started,
    ):
        raise GateEvidenceError("shard HTTP client lifecycle is missing or inconsistent")
    return start, requests, end


def _expected_requests_for_shard(
    start: Mapping[str, object],
) -> dict[tuple[object, ...], PlannedRequest]:
    trace = start["trace"]
    graph_warmup = start["graph_warmup"]
    assert isinstance(trace, dict) and isinstance(graph_warmup, dict)
    expected = [*_serial_warmup_plan(trace), *_graph_warmup_plan(graph_warmup)]
    phase = start["phase"]
    if phase == "preflight":
        expected.extend(request for group in _preflight_plans(trace) for request in group)
    elif phase == "formal":
        expected.extend(_formal_plan(trace))
    else:
        selection = start["selection"]
        assert isinstance(selection, dict) and isinstance(selection["value"], dict)
        rates = selection["value"]["open_loop_rates_millirps"]
        assert isinstance(rates, list)
        expected.extend(request for group in _open_loop_plans(trace, rates) for request in group)
    result: dict[tuple[object, ...], PlannedRequest] = {}
    for request in expected:
        identity = (
            request.phase,
            request.sequence,
            request.preflight_repeat,
            request.graph_batch_size,
            request.graph_batch_sequence,
            request.open_loop_point,
        )
        if identity in result:
            raise GateEvidenceError(f"planned request identity is not unique: {identity}")
        result[identity] = request
    return result


def _request_contract_valid(row: Mapping[str, object], planned: PlannedRequest) -> bool:
    usage = row.get("usage")
    timing = row.get("timing_ns")
    expected_client_role = (
        "warmup" if planned.phase in {"serial_warmup", "graph_warmup"} else "measurement"
    )
    identity = (
        row.get("http_client_role") == expected_client_role
        and type(row.get("http_client_request_ordinal")) is int
        and int(row["http_client_request_ordinal"]) >= 0
        and row.get("phase") == planned.phase
        and row.get("sequence") == planned.sequence
        and row.get("preflight_repeat") == planned.preflight_repeat
        and row.get("graph_batch_size") == planned.graph_batch_size
        and row.get("graph_batch_sequence") == planned.graph_batch_sequence
        and row.get("open_loop_point") == planned.open_loop_point
        and row.get("rate_millirps") == planned.rate_millirps
        and row.get("prompt_text_sha256") == planned.prompt_text_sha256
        and row.get("prompt_token_ids_sha256") == planned.prompt_token_ids_sha256
        and row.get("prompt_tokens") == len(planned.prompt_token_ids)
        and row.get("expected_completion_tokens") == planned.expected_completion_tokens
    )
    strict_stream = (
        row.get("attempts") == 1
        and row.get("status") == "success"
        and row.get("error_type") is None
        and row.get("http_status") == 200
        and isinstance(row.get("content_type"), str)
        and str(row["content_type"]).lower().startswith("text/event-stream")
        and isinstance(row.get("response_id"), str)
        and bool(row["response_id"])
        and row.get("done_events") == 1
        and row.get("terminal_choice_events") == 1
        and row.get("finish_reasons") == ["length"]
        and row.get("usage_events") == 1
        and isinstance(row.get("output_text_sha256"), str)
        and _HEX64.fullmatch(str(row["output_text_sha256"])) is not None
        and isinstance(row.get("stream_payload_sha256"), str)
        and _HEX64.fullmatch(str(row["stream_payload_sha256"])) is not None
    )
    usage_exact = (
        isinstance(usage, dict)
        and usage.get("prompt_tokens") == len(planned.prompt_token_ids)
        and usage.get("completion_tokens") == planned.expected_completion_tokens
        and usage.get("cached_tokens") is not None
        and type(usage.get("cached_tokens")) is int
        and 0 <= int(usage["cached_tokens"]) <= len(planned.prompt_token_ids)
    )
    timing_exact = (
        isinstance(timing, dict)
        and set(timing) == {"start", "first_token", "end", "ttft", "total"}
        and all(type(timing.get(key)) is int for key in timing)
        and timing["start"] <= timing["first_token"] <= timing["end"]
        and timing["ttft"] == timing["first_token"] - timing["start"]
        and timing["total"] == timing["end"] - timing["start"]
        and timing["ttft"] >= 0
        and timing["total"] > 0
    )
    shape = True
    if planned.phase in {"formal", "preflight"}:
        shape = type(row.get("burst_release_ns")) is int
    elif planned.phase == "graph_warmup":
        shape = type(row.get("graph_release_ns")) is int
    elif planned.phase == "open_loop":
        shape = (
            type(row.get("scheduled_start_ns")) is int
            and type(row.get("scheduled_offset_ns")) is int
            and type(row.get("release_lag_ns")) is int
        )
    return identity and strict_stream and usage_exact and timing_exact and shape


def _shard_request_checks(
    start: Mapping[str, object], requests: Sequence[Mapping[str, object]]
) -> tuple[bool, bool]:
    expected = _expected_requests_for_shard(start)
    actual: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in requests:
        identity = (
            row.get("phase"),
            row.get("sequence"),
            row.get("preflight_repeat"),
            row.get("graph_batch_size"),
            row.get("graph_batch_sequence"),
            row.get("open_loop_point"),
        )
        if identity in actual:
            return False, False
        actual[identity] = row
    identity_exact = set(actual) == set(expected)
    evidence_exact = identity_exact and all(
        _request_contract_valid(actual[identity], planned) for identity, planned in expected.items()
    )
    return identity_exact, evidence_exact


def _measurement_metric(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    if not rows or not all(row.get("status") == "success" for row in rows):
        return None
    timings = [row.get("timing_ns") for row in rows]
    usages = [row.get("usage") for row in rows]
    if not all(isinstance(item, dict) for item in timings + usages):
        return None
    starts = [int(item["start"]) for item in timings if isinstance(item, dict)]
    ends = [int(item["end"]) for item in timings if isinstance(item, dict)]
    ttfts = [int(item["ttft"]) for item in timings if isinstance(item, dict)]
    completion_tokens = sum(
        int(item["completion_tokens"]) for item in usages if isinstance(item, dict)
    )
    span_ns = max(ends) - min(starts)
    if span_ns <= 0 or len(ttfts) != len(rows):
        return None
    throughput = Fraction(completion_tokens * 1_000_000_000, span_ns * GPU_COUNT)
    return {
        "request_count": len(rows),
        "completion_tokens": completion_tokens,
        "span_ns": span_ns,
        "completion_tok_s_per_gpu": _fraction_record(throughput),
        "ttft_p99_ns": nearest_rank(ttfts, 0.99),
    }


def _provenance_raw(start: Mapping[str, object]) -> Mapping[str, object]:
    descriptor = start["provenance_start"]
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("value"), dict):
        raise GateEvidenceError("scenario provenance descriptor is invalid")
    return descriptor["value"]


def _common_provenance_valid(
    shards: Sequence[tuple[dict[str, object], list[dict[str, object]], dict[str, object]]],
    *,
    require_phase_order: bool,
) -> bool:
    if not shards:
        return False
    for start, _rows, end in shards:
        provenance_start = start.get("provenance_start")
        provenance_end = end.get("provenance_end")
        if (
            not isinstance(provenance_start, dict)
            or not isinstance(provenance_start.get("value"), dict)
            or not isinstance(provenance_end, dict)
            or not _live_observation_value_valid(
                provenance_end.get("value"), provenance=provenance_start["value"]
            )
        ):
            return False
    provenance = [_provenance_raw(start) for start, _rows, _end in shards]
    first = provenance[0]
    gpu_identity = first["gpus"]
    checkpoint = first["checkpoint"]
    checkpoint_capture_start = first["checkpoint_capture_start"]
    checkpoint_volume_identity = {
        key: first["checkpoint_volume"][key]
        for key in ("name", "driver", "scope", "mountpoint", "subpath")
    }
    host_id = first["environment"]["host_id"]
    driver_version = first["environment"]["driver_version"]
    if any(
        item["gpus"] != gpu_identity
        or item["checkpoint"] != checkpoint
        or item["checkpoint_capture_start"] != checkpoint_capture_start
        or {
            key: item["checkpoint_volume"][key]
            for key in ("name", "driver", "scope", "mountpoint", "subpath")
        }
        != checkpoint_volume_identity
        or item["checkpoint_volume"]["rw_consumer_count"] != 0
        or item["environment"]["host_id"] != host_id
        or item["environment"]["driver_version"] != driver_version
        for item in provenance
    ):
        return False
    generation_ids = [item["server_generation_id"] for item in provenance]
    container_ids = [item["container"]["id"] for item in provenance]
    if len(generation_ids) != len(set(generation_ids)) or len(container_ids) != len(
        set(container_ids)
    ):
        return False
    for arm in ARMS:
        arm_provenance = [item for item in provenance if item["arm"] == arm]
        if not arm_provenance:
            return False
        source = arm_provenance[0]["source"]
        environment = arm_provenance[0]["environment"]
        image = {
            key: arm_provenance[0]["container"][key]
            for key in ("image_repo_digest", "image_platform_digest", "image_config_id")
        }
        if any(
            item["source"] != source
            or item["environment"] != environment
            or {
                key: item["container"][key]
                for key in ("image_repo_digest", "image_platform_digest", "image_config_id")
            }
            != image
            for item in arm_provenance
        ):
            return False
    chronological = sorted(shards, key=lambda item: int(item[0]["capture_start_ns"]))
    for previous, current in zip(chronological, chronological[1:], strict=False):
        previous_end = int(previous[2]["capture_end_ns"])
        current_start = int(current[0]["capture_start_ns"])
        current_server_start = int(_provenance_raw(current[0])["server_started_ns"])
        if not previous_end < current_server_start <= current_start:
            return False
    if require_phase_order:
        phase_rank = {"preflight": 0, "formal": 1}
        ranks = [phase_rank[str(start["phase"])] for start, _rows, _end in chronological]
        if ranks != sorted(ranks):
            return False
    return True


def _checkpoint_matrix_binding_valid(
    shards: Sequence[tuple[dict[str, object], list[dict[str, object]], dict[str, object]]],
    *,
    checkpoint_start: object,
    checkpoint_end: object,
) -> bool:
    if (
        not shards
        or not _descriptor_valid(checkpoint_start)
        or not _descriptor_valid(checkpoint_end)
        or not isinstance(checkpoint_start, dict)
        or not isinstance(checkpoint_end, dict)
        or not _checkpoint_capture_value_valid(checkpoint_start.get("value"), phase="start")
        or not _checkpoint_capture_value_valid(checkpoint_end.get("value"), phase="end")
        or not isinstance(checkpoint_start.get("value"), dict)
        or not isinstance(checkpoint_end.get("value"), dict)
    ):
        return False
    start_value = checkpoint_start["value"]
    end_value = checkpoint_end["value"]
    start_container = start_value["capture_container"]
    end_container = end_value["capture_container"]
    assert isinstance(start_container, dict) and isinstance(end_container, dict)
    if (
        start_value["checkpoint"] != end_value["checkpoint"]
        or start_value["source"] != end_value["source"]
        or start_value["checkpoint_volume"] != end_value["checkpoint_volume"]
        or {
            key: start_container[key]
            for key in (
                "image_repo_digest",
                "image_config_id",
                "source_mount",
                "model_mount",
                "network_mode",
                "gpu_device_requests",
                "resolved_argv",
            )
        }
        != {
            key: end_container[key]
            for key in (
                "image_repo_digest",
                "image_config_id",
                "source_mount",
                "model_mount",
                "network_mode",
                "gpu_device_requests",
                "resolved_argv",
            )
        }
    ):
        return False
    start_mount = start_container["model_mount"]
    provenance = [_provenance_raw(start) for start, _requests, _end in shards]
    if any(
        item.get("checkpoint_capture_start") != checkpoint_start
        or item.get("checkpoint") != start_value["checkpoint"]["value"]
        or item.get("container", {}).get("model_mount") != start_mount
        or item.get("checkpoint_volume") != start_value["checkpoint_volume"]
        or (item.get("arm") == "kairyu" and item.get("source") != start_value["source"])
        or item.get("checkpoint_volume", {}).get("rw_consumer_count") != 0
        for item in provenance
    ):
        return False
    return (
        start_value["capture_completed_ns"]
        < min(int(item["server_started_ns"]) for item in provenance)
        and max(int(end["capture_end_ns"]) for _start, _requests, end in shards)
        < end_value["capture_started_ns"]
    )


def _selection_from_metrics(
    *,
    trace_bundle_sha256: str,
    moe_a2a_backend: str,
    preflight_raw_sha256: Mapping[str, str],
    preflight_completed_ns: int,
    candidate_fractions: Mapping[str, Mapping[str, Fraction]],
) -> dict[str, object]:
    selected: dict[str, str] = {}
    records: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        candidates = PREFLIGHT_CANDIDATES[arm]
        fractions = candidate_fractions[arm]
        if set(fractions) != set(candidates):
            raise GateEvidenceError(f"preflight candidate matrix for {arm} is incomplete")
        selected[arm] = max(
            candidates,
            key=lambda candidate: (fractions[candidate], -candidates.index(candidate)),
        )
        records[arm] = {
            candidate: _fraction_record(fractions[candidate]) for candidate in candidates
        }
    value = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "trace_bundle_sha256": trace_bundle_sha256,
        "moe_a2a_backend": moe_a2a_backend,
        "preflight_raw_sha256": dict(sorted(preflight_raw_sha256.items())),
        "preflight_completed_ns": preflight_completed_ns,
        "selected": selected,
        "candidate_metrics": records,
    }
    if not _selection_valid(value):
        raise GateEvidenceError("derived preflight selection is internally inconsistent")
    return value


def _preflight_runtime_static_identity(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"pipeline_depth", "overlap_mode", "resolved_argv"}
    }


def derive_preflight_selection(paths: Sequence[Path]) -> dict[str, object]:
    expected = {(arm, candidate) for arm in ARMS for candidate in PREFLIGHT_CANDIDATES[arm]}
    shards: dict[
        tuple[str, str],
        tuple[dict[str, object], list[dict[str, object]], dict[str, object], str],
    ] = {}
    for path in paths:
        rows = _read_jsonl(path)
        start, requests, end = _parse_shard(rows)
        if start["phase"] != "preflight":
            raise GateEvidenceError(f"{path} is not a preflight shard")
        identity = (str(start["arm"]), str(start["candidate"]))
        if identity in shards:
            raise GateEvidenceError(f"duplicate preflight shard {identity}")
        shards[identity] = (start, requests, end, sha256_file(path))
    if set(shards) != expected:
        raise GateEvidenceError(
            "preflight requires exactly one fixed candidate shard per arm; "
            f"missing={sorted(expected - set(shards))}"
        )
    parsed = [(start, requests, end) for start, requests, end, _digest in shards.values()]
    if not _common_provenance_valid(parsed, require_phase_order=False):
        raise GateEvidenceError(
            "preflight servers are not fresh, sequential, and physically matched"
        )
    first = next(iter(shards.values()))[0]
    trace_sha256 = first["trace_bundle_sha256"]
    benchmark = first["benchmark"]
    sglang = benchmark["sglang"] if isinstance(benchmark, dict) else None
    if not isinstance(trace_sha256, str) or not isinstance(sglang, dict):
        raise GateEvidenceError("preflight benchmark identity is invalid")
    if any(
        start["trace_bundle_sha256"] != trace_sha256
        or start["benchmark"] != benchmark
        or start["trace"] != first["trace"]
        or start["graph_warmup"] != first["graph_warmup"]
        for start, _requests, _end, _digest in shards.values()
    ):
        raise GateEvidenceError("preflight shards do not share one trace and benchmark")
    for arm in ARMS:
        runtimes = [
            _provenance_raw(shards[(arm, candidate)][0])["runtime"]
            for candidate in PREFLIGHT_CANDIDATES[arm]
        ]
        assert all(isinstance(runtime, dict) for runtime in runtimes)
        if any(
            _preflight_runtime_static_identity(runtime)
            != _preflight_runtime_static_identity(runtimes[0])
            for runtime in runtimes[1:]
        ):
            raise GateEvidenceError(
                f"{arm} preflight candidates changed more than the declared overlap control"
            )
    fractions: dict[str, dict[str, Fraction]] = {arm: {} for arm in ARMS}
    for (arm, candidate), (start, requests, end, _digest) in shards.items():
        identity_exact, evidence_exact = _shard_request_checks(start, requests)
        if (
            not identity_exact
            or not evidence_exact
            or not _scenario_execution_shape_valid(start, requests, end)
            or not _dynamic_graph_witness_valid(
                arm,
                start["kairyu_graph_witness_start"],
                end["kairyu_graph_witness_end"],
            )
        ):
            raise GateEvidenceError(f"preflight {arm}/{candidate} has failed or ambiguous requests")
        repeat_metrics = []
        for repeat in range(PREFLIGHT_REPEATS):
            group = [
                row
                for row in requests
                if row["phase"] == "preflight" and row["preflight_repeat"] == repeat
            ]
            metric = _measurement_metric(group)
            if metric is None or metric["request_count"] != PREFLIGHT_REQUESTS:
                raise GateEvidenceError(
                    f"preflight {arm}/{candidate} repeat {repeat} is incomplete"
                )
            repeat_metrics.append(_fraction_from_record(metric["completion_tok_s_per_gpu"]))
        fractions[arm][candidate] = _median_fraction(repeat_metrics)
    hashes = {
        f"preflight-{arm}-{candidate}": digest
        for (arm, candidate), (_start, _requests, _end, digest) in shards.items()
    }
    completed = max(
        int(end["capture_end_ns"]) for _start, _requests, end, _digest in shards.values()
    )
    return _selection_from_metrics(
        trace_bundle_sha256=trace_sha256,
        moe_a2a_backend=str(sglang["moe_a2a_backend"]),
        preflight_raw_sha256=hashes,
        preflight_completed_ns=completed,
        candidate_fractions=fractions,
    )


def write_preflight_selection(path: Path, paths: Sequence[Path]) -> dict[str, object]:
    value = derive_preflight_selection(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value, indent=2, sort_keys=True, ensure_ascii=False)
    return value


def _load_unique_phase_shards(
    paths: Sequence[Path],
    *,
    phase: str,
) -> dict[
    object,
    tuple[dict[str, object], list[dict[str, object]], dict[str, object], str],
]:
    result: dict[
        object,
        tuple[dict[str, object], list[dict[str, object]], dict[str, object], str],
    ] = {}
    for path in paths:
        start, requests, end = _parse_shard(_read_jsonl(path))
        if start["phase"] != phase:
            raise GateEvidenceError(f"{path} is not a {phase} shard")
        identity: object
        if phase == "preflight":
            identity = (start["arm"], start["candidate"])
        elif phase == "formal":
            identity = (start["repeat"], start["arm"])
        else:
            identity = start["arm"]
        if identity in result:
            raise GateEvidenceError(f"duplicate {phase} shard {identity}")
        result[identity] = (start, requests, end, sha256_file(path))
    return result


def assemble_rows(
    *,
    preflight_paths: Sequence[Path],
    formal_paths: Sequence[Path],
    selection_path: Path,
    checkpoint_start_path: Path,
    checkpoint_end_path: Path,
) -> list[dict[str, object]]:
    checkpoint_start = load_checkpoint_capture(checkpoint_start_path, phase="start")
    checkpoint_end = load_checkpoint_capture(checkpoint_end_path, phase="end")
    expected_selection = derive_preflight_selection(preflight_paths)
    selection = load_selection(selection_path)
    if selection["value"] != expected_selection:
        raise GateEvidenceError("selection differs from an independent preflight raw replay")
    preflight = _load_unique_phase_shards(preflight_paths, phase="preflight")
    formal = _load_unique_phase_shards(formal_paths, phase="formal")
    if set(formal) != set(FORMAL_SEQUENCE):
        raise GateEvidenceError(
            "formal shards require the exact eight-cell matrix; "
            f"missing={sorted(set(FORMAL_SEQUENCE) - set(formal))}"
        )
    all_values = [*preflight.values(), *formal.values()]
    parsed = [(start, requests, end) for start, requests, end, _digest in all_values]
    if not _common_provenance_valid(parsed, require_phase_order=True):
        raise GateEvidenceError(
            "all servers must be fresh, sequential, and use the same host/GPU/checkpoint"
        )
    if not _checkpoint_matrix_binding_valid(
        parsed,
        checkpoint_start=checkpoint_start,
        checkpoint_end=checkpoint_end,
    ):
        raise GateEvidenceError(
            "checkpoint bytes, matrix time window, volume, or per-generation binding changed"
        )
    chronological_formal = sorted(
        formal.values(), key=lambda value: int(value[0]["capture_start_ns"])
    )
    observed_order = tuple(
        (int(start["repeat"]), str(start["arm"])) for start, _r, _e, _d in chronological_formal
    )
    if observed_order != FORMAL_SEQUENCE:
        raise GateEvidenceError(
            f"formal server order must be {FORMAL_SEQUENCE}, got {observed_order}"
        )
    for start, _requests, _end, _digest in formal.values():
        if start["selection"] != selection:
            raise GateEvidenceError("formal shard changed the frozen selection")
        provenance = _provenance_raw(start)
        if provenance["server_started_ns"] <= expected_selection["preflight_completed_ns"]:
            raise GateEvidenceError("formal server started before preflight froze")
    for arm in ARMS:
        candidate = expected_selection["selected"][arm]
        selected_preflight = preflight[(arm, candidate)][0]
        selected_runtime = _provenance_raw(selected_preflight)["runtime"]
        for start, _requests, _end, _digest in [
            *[value for key, value in formal.items() if key[1] == arm],
        ]:
            if _provenance_raw(start)["runtime"] != selected_runtime:
                raise GateEvidenceError(
                    f"{arm} formal runtime differs from selected preflight runtime"
                )
    first = next(iter(preflight.values()))[0]
    if any(
        start["benchmark"] != first["benchmark"]
        or start["trace"] != first["trace"]
        or start["graph_warmup"] != first["graph_warmup"]
        or start["trace_bundle_sha256"] != first["trace_bundle_sha256"]
        for start, _requests, _end, _digest in all_values
    ):
        raise GateEvidenceError("shards changed the benchmark or trace")
    combined: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": SCHEMA_VERSION,
            "gate": GATE,
            "benchmark": first["benchmark"],
            "trace": first["trace"],
            "graph_warmup": first["graph_warmup"],
            "trace_bundle_sha256": first["trace_bundle_sha256"],
            "selection": selection,
            "checkpoint_start": checkpoint_start,
        }
    ]
    chronological = sorted(all_values, key=lambda value: int(value[0]["capture_start_ns"]))
    for start, requests, end, digest in chronological:
        combined.append(
            {
                **{key: value for key, value in start.items() if key not in {"type", "row_index"}},
                "type": "scenario_start",
                "shard_sha256": digest,
            }
        )
        combined.extend(
            {key: value for key, value in request.items() if key != "row_index"}
            for request in requests
        )
        combined.append(
            {
                **{key: value for key, value in end.items() if key not in {"type", "row_index"}},
                "type": "scenario_end",
            }
        )
    combined.append(
        {
            "type": "run_end",
            "preflight_scenario_count": len(preflight),
            "formal_scenario_count": len(formal),
            "scenario_count": len(all_values),
            "request_count": sum(len(requests) for _start, requests, _end, _digest in all_values),
            "checkpoint_end": checkpoint_end,
        }
    )
    return combined


def _parse_artifact_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[
    dict[str, object],
    dict[str, tuple[dict[str, object], list[dict[str, object]], dict[str, object]]],
    dict[str, object],
]:
    if len(rows) < 2 or rows[0].get("type") != "run" or rows[-1].get("type") != "run_end":
        raise GateEvidenceError("artifact must be framed by run/run_end")
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise GateEvidenceError("artifact row indices are not contiguous")
    run, run_end = rows[0], rows[-1]
    if set(run) != {
        "type",
        "schema_version",
        "gate",
        "benchmark",
        "trace",
        "graph_warmup",
        "trace_bundle_sha256",
        "selection",
        "checkpoint_start",
        "row_index",
    }:
        raise GateEvidenceError("artifact run row has unknown or missing fields")
    if run["schema_version"] != SCHEMA_VERSION or run["gate"] != GATE:
        raise GateEvidenceError("artifact schema/gate mismatch")
    scenarios: dict[
        str,
        tuple[dict[str, object], list[dict[str, object]], dict[str, object]],
    ] = {}
    current_start: dict[str, object] | None = None
    current_requests: list[dict[str, object]] = []
    for row in rows[1:-1]:
        row_type = row.get("type")
        if row_type == "scenario_start":
            if current_start is not None:
                raise GateEvidenceError("nested scenario_start row")
            current_start = dict(row)
            current_requests = []
        elif row_type == "request":
            if current_start is None:
                raise GateEvidenceError("request outside a scenario")
            current_requests.append(dict(row))
        elif row_type == "scenario_end":
            if current_start is None:
                raise GateEvidenceError("scenario_end without a start")
            scenario_id = current_start.get("scenario_id")
            if not isinstance(scenario_id, str) or scenario_id in scenarios:
                raise GateEvidenceError("duplicate or invalid artifact scenario")
            pseudo_rows = [
                {**current_start, "type": "shard_start"},
                *current_requests,
                {**row, "type": "shard_end"},
            ]
            for index, pseudo in enumerate(pseudo_rows):
                pseudo["row_index"] = index
            parsed_start, parsed_requests, parsed_end = _parse_shard(pseudo_rows)
            scenarios[scenario_id] = (parsed_start, parsed_requests, parsed_end)
            current_start = None
            current_requests = []
        else:
            raise GateEvidenceError(f"unknown artifact row type {row_type!r}")
    if current_start is not None:
        raise GateEvidenceError("unterminated artifact scenario")
    if set(run_end) != {
        "type",
        "preflight_scenario_count",
        "formal_scenario_count",
        "scenario_count",
        "request_count",
        "checkpoint_end",
        "row_index",
    }:
        raise GateEvidenceError("artifact run_end row has unknown or missing fields")
    return run, scenarios, run_end


def _measurement_groups_valid(
    start: Mapping[str, object],
    requests: Sequence[Mapping[str, object]],
    end: Mapping[str, object],
) -> bool:
    phase = start["phase"]
    measurement_phase = {
        "preflight": "preflight",
        "formal": "formal",
        "open-loop": "open_loop",
    }[str(phase)]
    measured = [row for row in requests if row["phase"] == measurement_phase]
    group_field = "preflight_repeat" if phase == "preflight" else "open_loop_point"
    expected_groups = PREFLIGHT_REPEATS if phase == "preflight" else OPEN_LOOP_POINTS
    if phase == "formal":
        grouped = {0: measured}
        expected_groups = 1
    else:
        grouped = {
            group: [row for row in measured if row[group_field] == group]
            for group in range(expected_groups)
        }
    retained = end.get("measurement_groups")
    if not isinstance(retained, list) or len(retained) != expected_groups:
        return False
    recomputed = []
    for group in range(expected_groups):
        rows = grouped[group]
        timings = [row.get("timing_ns") for row in rows]
        if not rows or not all(isinstance(item, dict) for item in timings):
            return False
        starts = [int(item["start"]) for item in timings if isinstance(item, dict)]
        ends = [int(item["end"]) for item in timings if isinstance(item, dict)]
        recomputed.append({"group": group, "start": min(starts), "end": max(ends)})
    return retained == recomputed


def _scenario_execution_shape_valid(
    start: Mapping[str, object],
    requests: Sequence[Mapping[str, object]],
    end: Mapping[str, object],
) -> bool:
    if not _measurement_groups_valid(start, requests, end):
        return False
    if any(
        not isinstance(row.get("timing_ns"), dict)
        or type(row["timing_ns"].get("start")) is not int
        or type(row["timing_ns"].get("end")) is not int
        for row in requests
    ):
        return False
    serial = sorted(
        (row for row in requests if row["phase"] == "serial_warmup"),
        key=lambda row: int(row["sequence"]),
    )
    serial_timings = [row["timing_ns"] for row in serial]
    if any(
        int(previous["end"]) > int(current["start"])
        for previous, current in zip(serial_timings, serial_timings[1:], strict=False)
    ):
        return False
    serial_end = max(int(timing["end"]) for timing in serial_timings)
    graph = [row for row in requests if row["phase"] == "graph_warmup"]
    previous_end: int | None = None
    for batch_size in GRAPH_WARMUP_BATCH_SIZES:
        batch = [row for row in graph if row["graph_batch_size"] == batch_size]
        releases = {row.get("graph_release_ns") for row in batch}
        if (
            len(batch) != batch_size
            or len(releases) != 1
            or not all(type(row.get("graph_release_ns")) is int for row in batch)
            or not all(
                int(row["timing_ns"]["start"]) >= int(row["graph_release_ns"]) for row in batch
            )
        ):
            return False
        batch_start = min(int(row["timing_ns"]["start"]) for row in batch)
        batch_end = max(int(row["timing_ns"]["end"]) for row in batch)
        if previous_end is not None and previous_end > batch_start:
            return False
        previous_end = batch_end
    if min(int(row["timing_ns"]["start"]) for row in graph) < serial_end:
        return False
    assert previous_end is not None
    graph_end = previous_end
    phase = start["phase"]
    if phase in {"preflight", "formal"}:
        measured_phase = "preflight" if phase == "preflight" else "formal"
        measured = [row for row in requests if row["phase"] == measured_phase]
        groups = PREFLIGHT_REPEATS if phase == "preflight" else 1
        previous_end = None
        for group in range(groups):
            batch = (
                [row for row in measured if row["preflight_repeat"] == group]
                if phase == "preflight"
                else measured
            )
            releases = {row.get("burst_release_ns") for row in batch}
            if (
                len(releases) != 1
                or not all(type(row.get("burst_release_ns")) is int for row in batch)
                or not all(
                    int(row["timing_ns"]["start"]) >= int(row["burst_release_ns"]) for row in batch
                )
            ):
                return False
            batch_start = min(int(row["timing_ns"]["start"]) for row in batch)
            batch_end = max(int(row["timing_ns"]["end"]) for row in batch)
            if previous_end is not None and previous_end > batch_start:
                return False
            previous_end = batch_end
        if min(int(row["timing_ns"]["start"]) for row in measured) < graph_end:
            return False
    else:
        measured = [row for row in requests if row["phase"] == "open_loop"]
        if any(
            type(row.get("scheduled_start_ns")) is not int
            or type(row.get("scheduled_offset_ns")) is not int
            for row in measured
        ):
            return False
        selection = start.get("selection")
        if not isinstance(selection, dict) or not isinstance(selection.get("value"), dict):
            return False
        rates = selection["value"].get("open_loop_rates_millirps")
        if not isinstance(rates, list) or len(rates) != OPEN_LOOP_POINTS:
            return False
        previous_end = None
        for point in range(OPEN_LOOP_POINTS):
            group = sorted(
                (row for row in measured if row["open_loop_point"] == point),
                key=lambda row: int(row["scheduled_start_ns"]),
            )
            if len(group) != OPEN_LOOP_REQUESTS_PER_POINT or any(
                int(previous["scheduled_start_ns"]) >= int(current["scheduled_start_ns"])
                for previous, current in zip(group, group[1:], strict=False)
            ):
                return False
            rate = rates[point]
            if type(rate) is not int or rate <= 0:
                return False
            period_ns = 1_000_000_000_000 // rate
            anchors = {
                int(row["scheduled_start_ns"]) - int(row["scheduled_offset_ns"]) for row in group
            }
            if (
                len(anchors) != 1
                or any(row.get("rate_millirps") != rate for row in group)
                or [int(row["scheduled_offset_ns"]) for row in group]
                != [position * period_ns for position in range(len(group))]
                or any(
                    int(row["timing_ns"]["start"]) < int(row["scheduled_start_ns"]) for row in group
                )
            ):
                return False
            group_start = min(int(row["timing_ns"]["start"]) for row in group)
            group_end = max(int(row["timing_ns"]["end"]) for row in group)
            if previous_end is not None and previous_end > group_start:
                return False
            previous_end = group_end
        if min(int(row["timing_ns"]["start"]) for row in measured) < graph_end:
            return False
    return True


def _selection_from_artifact_preflight(
    run: Mapping[str, object],
    scenarios: Mapping[
        str,
        tuple[dict[str, object], list[dict[str, object]], dict[str, object]],
    ],
) -> dict[str, object]:
    fractions: dict[str, dict[str, Fraction]] = {arm: {} for arm in ARMS}
    hashes: dict[str, str] = {}
    completed = 0
    for arm in ARMS:
        for candidate in PREFLIGHT_CANDIDATES[arm]:
            scenario_id = f"preflight-{arm}-{candidate}"
            start, requests, end = scenarios[scenario_id]
            digest = start.get("shard_sha256")
            if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
                raise GateEvidenceError("artifact preflight shard hash is invalid")
            hashes[scenario_id] = digest
            completed = max(completed, int(end["capture_end_ns"]))
            repeat_values = []
            for repeat in range(PREFLIGHT_REPEATS):
                group = [
                    row
                    for row in requests
                    if row["phase"] == "preflight" and row["preflight_repeat"] == repeat
                ]
                metric = _measurement_metric(group)
                if metric is None:
                    raise GateEvidenceError("preflight raw no longer supports a selection")
                repeat_values.append(_fraction_from_record(metric["completion_tok_s_per_gpu"]))
            fractions[arm][candidate] = _median_fraction(repeat_values)
    benchmark = run["benchmark"]
    assert isinstance(benchmark, dict) and isinstance(benchmark["sglang"], dict)
    return _selection_from_metrics(
        trace_bundle_sha256=str(run["trace_bundle_sha256"]),
        moe_a2a_backend=str(benchmark["sglang"]["moe_a2a_backend"]),
        preflight_raw_sha256=hashes,
        preflight_completed_ns=completed,
        candidate_fractions=fractions,
    )


def _artifact_preflight_controls_isolated(
    scenarios: Mapping[
        str,
        tuple[dict[str, object], list[dict[str, object]], dict[str, object]],
    ],
) -> bool:
    for arm in ARMS:
        runtimes = [
            _provenance_raw(scenarios[f"preflight-{arm}-{candidate}"][0])["runtime"]
            for candidate in PREFLIGHT_CANDIDATES[arm]
        ]
        if not all(isinstance(runtime, dict) for runtime in runtimes):
            return False
        if any(
            _preflight_runtime_static_identity(runtime)
            != _preflight_runtime_static_identity(runtimes[0])
            for runtime in runtimes[1:]
        ):
            return False
    return True


def evaluate_rows(
    rows: Sequence[dict[str, object]],
) -> tuple[dict[str, bool], dict[str, bool], dict[str, object], dict[str, object]]:
    run, scenarios, run_end = _parse_artifact_rows(rows)
    expected_preflight = {
        f"preflight-{arm}-{candidate}" for arm in ARMS for candidate in PREFLIGHT_CANDIDATES[arm]
    }
    expected_formal = {f"formal-r{repeat}-{arm}" for repeat, arm in FORMAL_SEQUENCE}
    expected_scenarios = expected_preflight | expected_formal
    if set(scenarios) != expected_scenarios:
        raise GateEvidenceError(
            "artifact scenario matrix differs; "
            f"missing={sorted(expected_scenarios - set(scenarios))}"
        )
    benchmark = run["benchmark"]
    if not isinstance(benchmark, dict) or not isinstance(benchmark.get("sharegpt"), dict):
        raise GateEvidenceError("artifact benchmark is invalid")
    dataset_sha256 = benchmark["sharegpt"].get("sha256")
    sglang = benchmark.get("sglang")
    if not isinstance(dataset_sha256, str) or not isinstance(sglang, dict):
        raise GateEvidenceError("artifact benchmark pins are invalid")
    config_exact = benchmark == benchmark_config(
        dataset_sha256,
        moe_a2a_backend=str(sglang.get("moe_a2a_backend")),
    )
    trace_exact = (
        a6._trace_valid(run["trace"], workload="sharegpt", dataset_sha256=dataset_sha256)
        and _graph_warmup_trace_valid(run["graph_warmup"])
        and _graph_warmup_disjoint(run["graph_warmup"], run["trace"])
    )
    parsed = list(scenarios.values())
    common_provenance = _common_provenance_valid(parsed, require_phase_order=True)
    checkpoint_window_exact = _checkpoint_matrix_binding_valid(
        parsed,
        checkpoint_start=run["checkpoint_start"],
        checkpoint_end=run_end["checkpoint_end"],
    )
    chronological_formal = sorted(
        (scenarios[scenario_id] for scenario_id in expected_formal),
        key=lambda value: int(value[0]["capture_start_ns"]),
    )
    formal_order_exact = (
        tuple(
            (int(start["repeat"]), str(start["arm"]))
            for start, _requests, _end in chronological_formal
        )
        == FORMAL_SEQUENCE
    )
    scenario_envelopes_exact = all(
        start["benchmark"] == benchmark
        and start["trace"] == run["trace"]
        and start["graph_warmup"] == run["graph_warmup"]
        and start["trace_bundle_sha256"] == run["trace_bundle_sha256"]
        for start, _requests, _end in parsed
    )
    identity_checks: dict[str, bool] = {}
    evidence_checks: dict[str, bool] = {}
    shape_checks: dict[str, bool] = {}
    client_lifecycle_checks: dict[str, bool] = {}
    graph_witness_checks: dict[str, bool] = {}
    for scenario_id, (start, requests, end) in scenarios.items():
        identity_checks[scenario_id], evidence_checks[scenario_id] = _shard_request_checks(
            start, requests
        )
        shape_checks[scenario_id] = _scenario_execution_shape_valid(start, requests, end)
        provenance_end = end["provenance_end"]
        assert isinstance(provenance_end, dict) and isinstance(provenance_end["value"], dict)
        client_lifecycle_checks[scenario_id] = _http_client_lifecycle_valid(
            end["http_client_lifecycle"],
            arm=str(start["arm"]),
            phase=str(start["phase"]),
            requests=requests,
            capture_start_ns=int(start["capture_start_ns"]),
            end_observation_started_ns=int(provenance_end["value"]["capture_started_ns"]),
        )
        graph_witness_checks[scenario_id] = _dynamic_graph_witness_valid(
            start["arm"],
            start["kairyu_graph_witness_start"],
            end["kairyu_graph_witness_end"],
        )
    expected_selection = _selection_from_artifact_preflight(run, scenarios)
    selection = run["selection"]
    selection_exact = (
        _descriptor_valid(selection)
        and isinstance(selection, dict)
        and selection["value"] == expected_selection
        and all(
            start["selection"] is None
            if start["phase"] == "preflight"
            else start["selection"] == selection
            for start, _requests, _end in parsed
        )
    )
    preflight_controls_isolated = _artifact_preflight_controls_isolated(scenarios)
    runtime_selected_exact = selection_exact
    if selection_exact:
        assert isinstance(selection, dict) and isinstance(selection["value"], dict)
        selected = selection["value"]["selected"]
        assert isinstance(selected, dict)
        for arm in ARMS:
            preflight_runtime = _provenance_raw(scenarios[f"preflight-{arm}-{selected[arm]}"][0])[
                "runtime"
            ]
            for scenario_id in [
                *(f"formal-r{repeat}-{arm}" for repeat in range(REPEATS)),
            ]:
                runtime_selected_exact &= (
                    _provenance_raw(scenarios[scenario_id][0])["runtime"] == preflight_runtime
                )
    scenario_metrics: dict[str, dict[str, object] | None] = {}
    paired_rounds: list[dict[str, object]] = []
    throughput_ratios: list[Fraction] = []
    ttft_ratios: list[Fraction] = []
    for repeat in range(REPEATS):
        metrics: dict[str, dict[str, object] | None] = {}
        for arm in ARMS:
            scenario_id = f"formal-r{repeat}-{arm}"
            formal_rows = [row for row in scenarios[scenario_id][1] if row["phase"] == "formal"]
            metric = _measurement_metric(formal_rows)
            metrics[arm] = metric
            scenario_metrics[scenario_id] = metric
        kairyu_metric = metrics["kairyu"]
        sglang_metric = metrics["sglang"]
        throughput_ratio: Fraction | None = None
        ttft_ratio: Fraction | None = None
        if kairyu_metric is not None and sglang_metric is not None:
            kairyu_throughput = _fraction_from_record(kairyu_metric["completion_tok_s_per_gpu"])
            sglang_throughput = _fraction_from_record(sglang_metric["completion_tok_s_per_gpu"])
            if sglang_throughput > 0 and int(sglang_metric["ttft_p99_ns"]) > 0:
                throughput_ratio = kairyu_throughput / sglang_throughput
                ttft_ratio = Fraction(
                    int(kairyu_metric["ttft_p99_ns"]),
                    int(sglang_metric["ttft_p99_ns"]),
                )
                throughput_ratios.append(throughput_ratio)
                ttft_ratios.append(ttft_ratio)
        paired_rounds.append(
            {
                "repeat": repeat,
                "server_start_order": list(PAIR_ORDER[repeat]),
                "kairyu": kairyu_metric,
                "sglang": sglang_metric,
                "completion_tok_s_per_gpu_kairyu_over_sglang": (
                    _fraction_record(throughput_ratio) if throughput_ratio is not None else None
                ),
                "ttft_p99_kairyu_over_sglang": (
                    _fraction_record(ttft_ratio) if ttft_ratio is not None else None
                ),
            }
        )
    paired_complete = len(throughput_ratios) == len(ttft_ratios) == REPEATS
    throughput_median = _median_fraction(throughput_ratios) if paired_complete else None
    ttft_median = _median_fraction(ttft_ratios) if paired_complete else None
    throughput_gate = throughput_median is not None and throughput_median >= 1
    ttft_gate = ttft_median is not None and ttft_median <= 1
    expected_request_count = (
        len(expected_preflight)
        * (SERIAL_WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS + PREFLIGHT_REPEATS * PREFLIGHT_REQUESTS)
        + len(expected_formal)
        * (SERIAL_WARMUP_REQUESTS + GRAPH_WARMUP_REQUESTS + SHAREGPT_REQUESTS)
    )
    counts_exact = run_end == {
        "type": "run_end",
        "preflight_scenario_count": len(expected_preflight),
        "formal_scenario_count": len(expected_formal),
        "scenario_count": len(expected_scenarios),
        "request_count": expected_request_count,
        "checkpoint_end": run_end["checkpoint_end"],
        "row_index": len(rows) - 1,
    }
    binding_ids = expected_preflight | expected_formal
    binding_checks = {
        "fixed_benchmark_configuration_exact": config_exact,
        "fixed_sharegpt_trace_and_warmup_exact": trace_exact,
        "exact_scenario_and_request_counts": counts_exact,
        "scenario_trace_and_benchmark_identity_exact": scenario_envelopes_exact,
        "same_four_gpus_fresh_sequential_servers_and_checkpoint_exact": common_provenance,
        "checkpoint_bytes_hashed_before_and_after_full_matrix_exact": (
            checkpoint_window_exact
        ),
        "formal_server_start_order_k_s_s_k_s_k_k_s_exact": formal_order_exact,
        "preflight_selection_replayed_from_raw_and_frozen_before_formal": selection_exact,
        "preflight_candidates_change_only_declared_overlap_control": (preflight_controls_isolated),
        "formal_runtime_matches_selected_preflight": runtime_selected_exact,
        "every_binding_request_identity_retained_exactly_once": all(
            identity_checks[scenario_id] for scenario_id in binding_ids
        ),
        "strict_streaming_binding_requests_all_successful_exactly_once_no_retry": all(
            evidence_checks[scenario_id] for scenario_id in binding_ids
        ),
        "binding_warmup_burst_and_measurement_windows_exact": all(
            shape_checks[scenario_id] for scenario_id in binding_ids
        ),
        "warmup_pool_closed_before_zero-request_fresh_measurement_pool_exact": all(
            client_lifecycle_checks[scenario_id] for scenario_id in binding_ids
        ),
        "kairyu_request_owned_ep4_direct_nccl_and_cuda_graph_replay_exact": all(
            graph_witness_checks[scenario_id] for scenario_id in binding_ids
        ),
        "paired_median_completion_tok_s_per_gpu_kairyu_at_least_sglang": throughput_gate,
        "paired_median_ttft_p99_kairyu_no_worse_than_sglang": ttft_gate,
        "m_a1_formal_fail_is_explicit_and_independent": (
            benchmark.get("m_a1_status") == "formal_fail_independent"
        ),
    }
    report_checks: dict[str, bool] = {}
    comparisons = {
        "paired_rounds": paired_rounds,
        "paired_median": {
            "method": (
                "exact median of four paired ratios; even median averages exact middle Fractions"
            ),
            "completion_tok_s_per_gpu_kairyu_over_sglang": (
                _fraction_record(throughput_median) if throughput_median is not None else None
            ),
            "ttft_p99_kairyu_over_sglang": (
                _fraction_record(ttft_median) if ttft_median is not None else None
            ),
        },
    }
    diagnostics = {
        "scenario_metrics": scenario_metrics,
        "preflight_selection": expected_selection,
        "report_only_checks": report_checks,
        "method": {
            "raw_jsonl_is_authority": True,
            "failures_or_retries_excluded": False,
            "warmups_excluded_from_metrics": True,
            "outlier_removal": False,
            "round_before_gate": False,
            "m_a1_status": "formal_fail_independent",
        },
        "sglang_limitations": [
            "SM120 uses FlashInfer CUTLASS rather than the SM100-only TRTLLM-gen MoE path",
            "breakable prefill CUDA graph is disabled; decode CUDA graph remains enabled",
            "MTP/speculative decoding is disabled because it belongs to M-A4",
        ],
    }
    return binding_checks, report_checks, comparisons, diagnostics


def recompute_manifest(
    rows: Sequence[dict[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    if _HEX64.fullmatch(raw_sha256) is None:
        raise GateEvidenceError("manifest requires a lowercase raw SHA-256")
    binding_checks, report_checks, comparisons, diagnostics = evaluate_rows(rows)
    passed = all(binding_checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "raw": {
            "name": RAW_NAME,
            "sha256": raw_sha256,
            "row_count": len(rows),
        },
        "checks": {
            "binding": binding_checks,
            "report_only": report_checks,
        },
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


def write_artifact(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    _write_jsonl(raw_path, rows)
    stored = _read_jsonl(raw_path)
    manifest = recompute_manifest(stored, raw_sha256=sha256_file(raw_path))
    atomic_write_json(
        output_dir / MANIFEST_NAME,
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    return manifest


def _artifact_paths(path: Path) -> tuple[Path, Path]:
    return evidence_artifact_paths(
        path,
        raw_name=RAW_NAME,
        manifest_name=MANIFEST_NAME,
        error_type=GateEvidenceError,
    )


def verify_artifact(path: Path, *, assert_gate: bool = False) -> dict[str, object]:
    replayed = verify_replayed_manifest(
        path,
        raw_name=RAW_NAME,
        manifest_name=MANIFEST_NAME,
        recompute=lambda rows, raw_sha256: recompute_manifest(
            rows,
            raw_sha256=raw_sha256,
        ),
        error_type=GateEvidenceError,
    )
    if assert_gate and replayed["passed"] is not True:
        raise RuntimeError("G4 M-A3 artifact does not pass every binding check")
    return replayed


def replay_artifact(path: Path, *, assert_gate: bool = False) -> dict[str, object]:
    raw_path, _manifest_path = _artifact_paths(path)
    replayed = evidence_replay_manifest(
        raw_path,
        recompute=lambda rows, raw_sha256: recompute_manifest(
            rows,
            raw_sha256=raw_sha256,
        ),
        error_type=GateEvidenceError,
    )
    if assert_gate and replayed["passed"] is not True:
        raise RuntimeError("G4 M-A3 raw replay does not pass every binding check")
    return replayed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build the pinned ShareGPT trace bundle")
    prepare.add_argument("--tokenizer", type=Path, required=True)
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--dataset-sha256", default=a6.SHAREGPT_DATASET_SHA256)
    prepare.add_argument(
        "--sglang-moe-a2a-backend",
        default=DEFAULT_SGLANG_MOE_A2A_BACKEND,
        help="one frozen dispatcher backend used by argv, provenance, and replay",
    )
    prepare.add_argument("--output", type=Path, required=True)

    checkpoint = subparsers.add_parser(
        "capture-checkpoint",
        help="hash the complete mounted checkpoint at one matrix boundary",
    )
    checkpoint.add_argument("--phase", choices=("start", "end"), required=True)
    checkpoint.add_argument("--container", required=True)
    checkpoint.add_argument("--source-root", type=Path, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)

    capture = subparsers.add_parser(
        "capture-provenance",
        help="observe one already-running formal container and write strict provenance",
    )
    capture.add_argument("--arm", choices=ARMS, required=True)
    capture.add_argument(
        "--candidate",
        choices=(*KAIRYU_PREFLIGHT_CANDIDATES, *SGLANG_PREFLIGHT_CANDIDATES),
        required=True,
    )
    capture.add_argument("--container", required=True)
    capture.add_argument("--server-generation-id", required=True)
    capture.add_argument("--server-started-ns", type=int, required=True)
    capture.add_argument("--model-volume", required=True)
    capture.add_argument("--checkpoint-start", type=Path, required=True)
    capture.add_argument("--source-root", type=Path)
    capture.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="measure one already-started fresh server")
    run.add_argument("--phase", choices=("preflight", "formal"), required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument(
        "--candidate", choices=(*KAIRYU_PREFLIGHT_CANDIDATES, *SGLANG_PREFLIGHT_CANDIDATES)
    )
    run.add_argument("--repeat", type=int, choices=range(REPEATS))
    run.add_argument("--endpoint", choices=(FORMAL_ENDPOINT,), required=True)
    run.add_argument("--trace-bundle", type=Path, required=True)
    run.add_argument("--selection", type=Path)
    run.add_argument("--provenance", type=Path, required=True)
    run.add_argument("--dataset-sha256", default=a6.SHAREGPT_DATASET_SHA256)
    run.add_argument("--api-key")
    run.add_argument("--request-timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    run.add_argument("--output", type=Path, required=True)

    assemble = subparsers.add_parser(
        "assemble",
        help="derive preflight selection or assemble the complete formal artifact",
    )
    assemble.add_argument("--preflight-raw", type=Path, action="append", required=True)
    assemble.add_argument("--preflight-only", action="store_true")
    assemble.add_argument("--selection-output", type=Path)
    assemble.add_argument("--selection", type=Path)
    assemble.add_argument("--raw", type=Path, action="append")
    assemble.add_argument("--checkpoint-start", type=Path)
    assemble.add_argument("--checkpoint-end", type=Path)
    assemble.add_argument("--output-dir", type=Path)
    assemble.add_argument("--assert-gate", action="store_true")

    for command, help_text in (
        ("verify", "compare the manifest with an independent raw replay"),
        ("replay", "recompute the verdict from raw JSONL only"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--artifact", type=Path, required=True)
        child.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv == [_CHECKPOINT_HELPER_COMMAND]:
        checkpoint = ma1_capture._checkpoint_snapshot(Path(CHECKPOINT_ROOT))  # noqa: SLF001
        print(canonical_json(checkpoint))
        return 0
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.command == "prepare":
        result = write_trace_bundle(
            args.output,
            dataset_path=args.dataset,
            tokenizer_path=args.tokenizer,
            dataset_sha256=args.dataset_sha256,
            moe_a2a_backend=args.sglang_moe_a2a_backend,
        )
        output: dict[str, object] = {
            "output": str(args.output),
            "sha256": sha256_file(args.output),
            "trace_sha256": result["trace"]["sha256"],
        }
    elif args.command == "capture-checkpoint":
        result = capture_checkpoint(
            args.output,
            phase=args.phase,
            container_name=args.container,
            source_root=args.source_root,
        )
        output = {
            "output": str(args.output),
            "sha256": sha256_file(args.output),
            "phase": result["phase"],
            "checkpoint_sha256": result["checkpoint"]["sha256"],
            "passed": True,
        }
    elif args.command == "capture-provenance":
        result = capture_provenance(
            args.output,
            arm=args.arm,
            candidate=args.candidate,
            container_name=args.container,
            server_generation_id=args.server_generation_id,
            server_started_ns=args.server_started_ns,
            model_volume=args.model_volume,
            checkpoint_start_path=args.checkpoint_start,
            source_root=args.source_root,
        )
        output = {
            "output": str(args.output),
            "sha256": sha256_file(args.output),
            "arm": result["arm"],
            "server_generation_id": result["server_generation_id"],
            "passed": True,
        }
    elif args.command == "run":
        rows = asyncio.run(
            measure_shard(
                phase=args.phase,
                arm=args.arm,
                candidate=args.candidate,
                repeat=args.repeat,
                endpoint=args.endpoint,
                trace_bundle_path=args.trace_bundle,
                selection_path=args.selection,
                provenance_path=args.provenance,
                dataset_sha256=args.dataset_sha256,
                api_key=args.api_key,
                request_timeout_s=args.request_timeout_s,
            )
        )
        output = write_shard(args.output, rows)
    elif args.command == "assemble":
        if args.preflight_only:
            if (
                args.selection_output is None
                or args.selection is not None
                or args.raw
                or args.checkpoint_start is not None
                or args.checkpoint_end is not None
                or args.output_dir is not None
                or args.assert_gate
            ):
                parser.error(
                    "assemble --preflight-only requires only --preflight-raw and --selection-output"
                )
            output = write_preflight_selection(args.selection_output, args.preflight_raw)
        else:
            if (
                args.selection is None
                or not args.raw
                or args.checkpoint_start is None
                or args.checkpoint_end is None
                or args.output_dir is None
                or args.selection_output is not None
            ):
                parser.error(
                    "formal assemble requires --selection, --raw, "
                    "--checkpoint-start, --checkpoint-end, and --output-dir"
                )
            rows = assemble_rows(
                preflight_paths=args.preflight_raw,
                formal_paths=args.raw,
                selection_path=args.selection,
                checkpoint_start_path=args.checkpoint_start,
                checkpoint_end_path=args.checkpoint_end,
            )
            output = write_artifact(args.output_dir, rows)
    elif args.command == "verify":
        output = verify_artifact(args.artifact, assert_gate=args.assert_gate)
    else:
        output = replay_artifact(args.artifact, assert_gate=args.assert_gate)
    print(canonical_json(output))
    must_pass = getattr(args, "assert_gate", False) or args.command == "run"
    return 1 if must_pass and output.get("passed") is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
