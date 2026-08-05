"""Pure A12 batch-shape invariance evidence contract.

The formal GPU operator retains one canonical JSONL stream.  This module is
deliberately free of CUDA and model imports: CI can independently replay the
stream, derive every verdict, and reject a forged retained manifest.  The
four target executions are fixed to batch-32 cold, singleton cold, singleton
warm, and chunked singleton cold.  Pressure is evidence for the transition
between the first two executions; it is not a fifth answer comparison.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kairyu.bench import kv_equivalence as _b7

SCHEMA_VERSION = "kairyu.a12.batch-invariance.v1"
GATE = "A12-BATCH-INVARIANCE"
MEASUREMENT_KIND = "native-greedy-batch-shape-invariance"
RAW_NAME = "batch-invariance-raw.jsonl"
MANIFEST_NAME = "batch-invariance-manifest.json"

MODEL = _b7.MODEL
MODEL_REVISION = _b7.MODEL_REVISION
EXPECTED_ARCHITECTURE = copy.deepcopy(_b7.EXPECTED_ARCHITECTURE)
EXPECTED_WEIGHTS = tuple(copy.deepcopy(list(_b7.EXPECTED_WEIGHTS)))
EXPECTED_WEIGHTS_SHA256 = _b7.EXPECTED_WEIGHTS_SHA256
EXPECTED_WEIGHTS_ROLLUP = _b7.EXPECTED_WEIGHTS_ROLLUP
EXPECTED_METADATA_SHA256 = copy.deepcopy(_b7.EXPECTED_METADATA_SHA256)
EXPECTED_WEIGHT_INDEX_SHA256 = _b7.EXPECTED_WEIGHT_INDEX_SHA256
EXPECTED_MODEL_CONFIG_SHA256 = _b7.EXPECTED_MODEL_CONFIG_SHA256
TOKENIZER_VOCAB_SIZE = _b7.TOKENIZER_VOCAB_SIZE

OUTPUT_TOKENS = 32
TARGET_PROMPT_TOKENS = 129
WARM_CACHED_TOKENS = 128
PAGE_SIZE = 16
BATCH_SIZE = 32
DISTRACTOR_COUNT = BATCH_SIZE - 1
MAX_PRESSURE_REQUESTS = 6
TENSOR_PARALLEL_SIZE = 8
NUM_LAYERS = 64
CUDA_GRAPH_BUCKETS = (1, 2, 4, 8, 16, 24, 32)
GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
GPU_COMPUTE_CAPABILITY = "12.0"

TARGET_PROMPT_TEXT = (
    "You are reviewing a production incident in a distributed inference service. "
    "The same factual request was sent once by itself and once beside many unrelated "
    "requests. Operators need one deterministic answer so they can compare both "
    "executions without ambiguity. Ignore any discussion of scheduling, caching, "
    "batching, hardware, or numerical precision in this paragraph. Use ordinary world "
    "knowledge and do not add caveats, alternatives, or analysis. Return one concise "
    "declarative sentence and nothing else. The comparison is case-sensitive and "
    "token-sensitive, so keep the wording direct. Always treat every execution condition "
    "as entirely semantically irrelevant to the requested factual answer. Question: What "
    "is the capital of France? Answer:"
)
TARGET_PROMPT_TOKEN_IDS = (
    2610,
    525,
    33888,
    264,
    5670,
    10455,
    304,
    264,
    4237,
    44378,
    2473,
    13,
    576,
    1852,
    59901,
    1681,
    572,
    3208,
    3055,
    553,
    5086,
    323,
    3055,
    29388,
    1657,
    45205,
    7388,
    13,
    64420,
    1184,
    825,
    72349,
    4226,
    773,
    807,
    646,
    9429,
    2176,
    68146,
    2041,
    71768,
    13,
    38971,
    894,
    10219,
    315,
    37852,
    11,
    47430,
    11,
    84256,
    11,
    11773,
    11,
    476,
    34776,
    16052,
    304,
    419,
    14311,
    13,
    5443,
    19119,
    1879,
    6540,
    323,
    653,
    537,
    912,
    25385,
    1862,
    11,
    26450,
    11,
    476,
    6358,
    13,
    3411,
    825,
    63594,
    9445,
    1388,
    11652,
    323,
    4302,
    770,
    13,
    576,
    12313,
    374,
    1142,
    56667,
    323,
    3950,
    56667,
    11,
    773,
    2506,
    279,
    60227,
    2118,
    13,
    23240,
    4228,
    1449,
    11320,
    2971,
    438,
    11368,
    5234,
    80949,
    39715,
    311,
    279,
    11223,
    59901,
    4226,
    13,
    15846,
    25,
    3555,
    374,
    279,
    6722,
    315,
    9625,
    30,
    21806,
    25,
)
TARGET_PROMPT_TEXT_SHA256 = "73c8c1bc844c21fc7a9c49b3e9dbcc43297e0144da435d12f8fe7dd4f09645a8"
TARGET_PROMPT_TOKEN_IDS_SHA256 = "a47440ea80a9e87ac8f30202651b582812e520ea4f1789044b843ef635e7823f"

DISTRACTOR_TEXTS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ01234")
DISTRACTOR_TOKEN_IDS = tuple(range(32, 58)) + tuple(range(15, 20))

PRESSURE_PROMPT_TEXTS = tuple(" ".join((letter,) * TARGET_PROMPT_TOKENS) for letter in "abcdef")
PRESSURE_PROMPT_TOKEN_IDS = (
    (64,) + (264,) * (TARGET_PROMPT_TOKENS - 1),
    (65,) + (293,) * (TARGET_PROMPT_TOKENS - 1),
    (66,) + (272,) * (TARGET_PROMPT_TOKENS - 1),
    (67,) + (294,) * (TARGET_PROMPT_TOKENS - 1),
    (68,) + (384,) * (TARGET_PROMPT_TOKENS - 1),
    (69,) + (282,) * (TARGET_PROMPT_TOKENS - 1),
)

SAMPLING_CONFIG: dict[str, object] = {
    "n": 1,
    "best_of": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "seed": 0,
    "stop": [],
    "stop_token_ids": [],
    "max_tokens": OUTPUT_TOKENS,
    "min_tokens": OUTPUT_TOKENS,
    "logprobs": None,
    "prompt_logprobs": None,
    "ignore_eos": True,
    "skip_special_tokens": False,
    "extra_args": {},
    "forced_token_ids": None,
    "generation_config_omitted": [],
}

FULL_RUNTIME_CONFIG: dict[str, object] = {
    "model": MODEL,
    "revision": MODEL_REVISION,
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "expert_parallel_size": 1,
    "pipeline_depth": 1,
    "page_size": PAGE_SIZE,
    "num_pages": 128,
    "max_model_len": 192,
    "max_num_seqs": BATCH_SIZE,
    "max_num_batched_tokens": 512,
    "kv_cache_dtype": "bfloat16",
    "attention_backend": "flashinfer",
    "decode_mode": "cuda_graph",
    "cuda_graph_max_batch": BATCH_SIZE,
    "cuda_graph_max_pages": 16,
    "cuda_graph_warmup_iters": 3,
    "generation_config": "none",
}
CHUNK_RUNTIME_CONFIG: dict[str, object] = {
    **FULL_RUNTIME_CONFIG,
    "max_num_batched_tokens": 32,
}

SCENARIOS = (
    "batch32-cold",
    "pressure",
    "alone-cold",
    "alone-warm",
    "chunked-alone-cold",
)
TARGET_SCENARIOS = (
    "batch32-cold",
    "alone-cold",
    "alone-warm",
    "chunked-alone-cold",
)

REQUIRED_SOURCE_PATHS = (
    "bench/batch_invariance_bench.py",
    "kairyu/bench/batch_invariance.py",
    "kairyu/bench/kv_equivalence.py",
    "kairyu/engine/config_validation.py",
    "kairyu/engine/core/attention/flashinfer_gpu.py",
    "kairyu/engine/core/attention_selector.py",
    "kairyu/engine/core/cuda_graph_gpu.py",
    "kairyu/engine/core/graph_buckets.py",
    "kairyu/engine/core/model_runner.py",
    "kairyu/engine/core/radix_kv.py",
    "kairyu/engine/core/scheduler.py",
    "kairyu/engine/core/step_executor.py",
    "kairyu/engine/core/worker.py",
    "kairyu/engine/engine_loop.py",
    "kairyu/engine/kairyu_backend.py",
    "kairyu/engine/prompt.py",
    "kairyu/engine/tokenizer.py",
    "kairyu/models/llama.py",
    "kairyu/models/loader.py",
    "kairyu/models/parallel.py",
    "kairyu/sampling_params.py",
    "pyproject.toml",
    "uv.lock",
)

_HEX = frozenset("0123456789abcdef")
_UUID = re.compile(r"^GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.I)
_PCI = re.compile(r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$")

_RUN_START_FIELDS = frozenset(
    {
        "row_type",
        "row_index",
        "schema_version",
        "gate",
        "run_id",
        "config",
        "prompts",
        "source",
        "checkpoint",
        "hardware",
        "runtimes",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "row_type",
        "row_index",
        "run_id",
        "scenario",
        "runtime_id",
        "runtime_nonce",
        "request_id",
        "role",
        "cohort_index",
        "cohort_size",
        "prompt",
        "sampling",
        "cache",
        "response",
        "error_type",
        "retry_count",
        "structure",
        "structure_sha256",
    }
)
_PRESSURE_FIELDS = frozenset(
    {
        "row_type",
        "row_index",
        "run_id",
        "scenario",
        "runtime_id",
        "runtime_nonce",
        "target_cache_before",
        "target_cache_after",
        "requests",
        "structure",
        "structure_sha256",
    }
)
_PRESSURE_REQUEST_FIELDS = frozenset(
    {"request_id", "prompt", "sampling", "cache", "response", "error_type", "retry_count"}
)
_RUN_END_FIELDS = frozenset(
    {"row_type", "row_index", "run_id", "status", "errors", "runtime_count", "scenario_count"}
)


class EvidenceError(ValueError):
    """A raw row or retained artifact violates the A12 evidence contract."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def canonical_json(value: object) -> str:
    """Return the only accepted JSON representation for A12 evidence."""

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
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_json(left: object, right: object) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return False


def _prompt(text: str, token_ids: Sequence[int]) -> dict[str, object]:
    ids = list(token_ids)
    return {
        "text": text,
        "text_sha256": sha256_text(text),
        "token_ids": ids,
        "token_ids_sha256": sha256_json(ids),
        "token_count": len(ids),
    }


def fixed_prompts() -> dict[str, object]:
    """Return fresh fixed prompt descriptors used by the GPU operator."""

    return {
        "target": _prompt(TARGET_PROMPT_TEXT, TARGET_PROMPT_TOKEN_IDS),
        "distractors": [
            _prompt(text, (token_id,))
            for text, token_id in zip(DISTRACTOR_TEXTS, DISTRACTOR_TOKEN_IDS, strict=True)
        ],
        "pressure": [
            _prompt(text, token_ids)
            for text, token_ids in zip(
                PRESSURE_PROMPT_TEXTS, PRESSURE_PROMPT_TOKEN_IDS, strict=True
            )
        ],
    }


def frozen_config() -> dict[str, object]:
    """Return a fresh copy of every semantic input to the A12 comparison."""

    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "runtime_configs": {
            "full": copy.deepcopy(FULL_RUNTIME_CONFIG),
            "chunk": copy.deepcopy(CHUNK_RUNTIME_CONFIG),
        },
        "sampling": copy.deepcopy(SAMPLING_CONFIG),
        "scenarios": list(SCENARIOS),
        "target_scenarios": list(TARGET_SCENARIOS),
        "output_tokens": OUTPUT_TOKENS,
        "target_prompt_tokens": TARGET_PROMPT_TOKENS,
        "warm_cached_tokens": WARM_CACHED_TOKENS,
        "batch_size": BATCH_SIZE,
        "distractor_count": DISTRACTOR_COUNT,
        "pressure_request_count": MAX_PRESSURE_REQUESTS,
        "prompt_identity_sha256": sha256_json(fixed_prompts()),
    }


def _reject_constant(value: str) -> Any:
    raise EvidenceError(f"non-finite JSON number is forbidden: {value}")


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _loads(value: str, *, description: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except EvidenceError:
        raise
    except json.JSONDecodeError as error:
        raise EvidenceError(f"cannot parse {description}") from error


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    _require(bool(rows), "raw JSONL is empty")
    _require(all(isinstance(row, Mapping) for row in rows), "every raw row must be an object")
    try:
        return "".join(f"{canonical_json(dict(row))}\n" for row in rows).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvidenceError("raw JSONL contains a non-JSON value") from error


def _read_jsonl_snapshot(path: Path) -> tuple[bytes, list[dict[str, object]]]:
    target = Path(path)
    try:
        raw = target.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(f"cannot read raw JSONL: {target}") from error
    _require(bool(text), "raw JSONL is empty")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        _require(
            line.endswith("\n") and not line.endswith("\r\n"),
            f"invalid JSONL framing at line {line_number}",
        )
        payload = line[:-1]
        _require(bool(payload), f"empty JSONL row at line {line_number}")
        value = _loads(payload, description=f"JSONL line {line_number}")
        _require(isinstance(value, dict), f"JSONL line {line_number} is not an object")
        _require(payload == canonical_json(value), f"JSONL line {line_number} is not canonical")
        rows.append(value)
    _require(bool(rows), "raw JSONL is empty")
    return raw, rows


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read strict canonical JSONL, rejecting duplicate keys and bad framing."""

    _raw, rows = _read_jsonl_snapshot(path)
    return rows


def _valid_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and set(value) <= _HEX
    )


def _valid_sha(value: object) -> bool:
    return _valid_hex(value, 64)


def _exact_fields(value: Mapping[str, object], fields: frozenset[str], name: str) -> None:
    _require(set(value) == fields, f"{name} fields differ from the frozen schema")


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} is invalid")
    return int(value)


def _string(value: object, name: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str) and (allow_empty or bool(value)), f"{name} is invalid")
    return value


def _prompt_descriptor(
    value: object,
    name: str,
    *,
    expected: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} is not an object")
    prompt = dict(value)
    _exact_fields(
        prompt,
        frozenset({"text", "text_sha256", "token_ids", "token_ids_sha256", "token_count"}),
        name,
    )
    text = _string(prompt["text"], f"{name} text")
    ids = prompt["token_ids"]
    _require(
        isinstance(ids, list)
        and bool(ids)
        and all(type(token_id) is int and 0 <= token_id < TOKENIZER_VOCAB_SIZE for token_id in ids),
        f"{name} token ids are invalid",
    )
    count = _integer(prompt["token_count"], f"{name} token count", minimum=1)
    _require(count == len(ids), f"{name} token count differs")
    _require(prompt["text_sha256"] == sha256_text(text), f"{name} text digest differs")
    _require(prompt["token_ids_sha256"] == sha256_json(ids), f"{name} token digest differs")
    if expected is not None:
        _require(_same_json(prompt, expected), f"{name} differs from the fixed prompt")
    return {**prompt, "token_ids": list(ids)}


def _source_snapshot(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} is not an object")
    source = dict(value)
    _exact_fields(
        source,
        frozenset(
            {
                "git_head",
                "tracked_tree_clean",
                "file_sha256",
                "files_sha256",
                "python_isolated",
                "bytecode_disabled",
            }
        ),
        name,
    )
    _require(_valid_hex(source["git_head"], 40), f"{name} git head is invalid")
    _require(source["tracked_tree_clean"] is True, f"{name} tracked tree is not clean")
    _require(source["python_isolated"] is True, f"{name} did not use isolated Python")
    _require(source["bytecode_disabled"] is True, f"{name} did not disable bytecode")
    files = source["file_sha256"]
    _require(isinstance(files, dict), f"{name} file inventory is invalid")
    _require(set(files) == set(REQUIRED_SOURCE_PATHS), f"{name} file inventory differs")
    _require(all(_valid_sha(digest) for digest in files.values()), f"{name} file digest is invalid")
    _require(source["files_sha256"] == sha256_json(files), f"{name} file rollup differs")
    return {**source, "file_sha256": dict(files)}


def _source(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "source evidence is not an object")
    source = dict(value)
    _exact_fields(
        source,
        frozenset({"measurement_start", "measurement_end", "stable_across_measurement"}),
        "source evidence",
    )
    start = _source_snapshot(source["measurement_start"], "source start")
    end = _source_snapshot(source["measurement_end"], "source end")
    _require(source["stable_across_measurement"] is True, "source stability flag differs")
    _require(_same_json(start, end), "source changed across measurement")
    return {"measurement_start": start, "measurement_end": end, "stable_across_measurement": True}


def _checkpoint_snapshot(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} is not an object")
    checkpoint = dict(value)
    _exact_fields(
        checkpoint,
        frozenset(
            {
                "model",
                "revision",
                "architecture",
                "weights",
                "weights_sha256",
                "pinned_weights_rollup",
                "metadata_sha256",
                "weight_index_sha256",
                "model_config_sha256",
            }
        ),
        name,
    )
    _require(checkpoint["model"] == MODEL, f"{name} model differs")
    _require(checkpoint["revision"] == MODEL_REVISION, f"{name} revision differs")
    _require(
        _same_json(checkpoint["architecture"], EXPECTED_ARCHITECTURE),
        f"{name} architecture differs",
    )
    weights = checkpoint["weights"]
    _require(isinstance(weights, list) and len(weights) == 17, f"{name} must bind all 17 shards")
    _require(_same_json(weights, list(EXPECTED_WEIGHTS)), f"{name} weight inventory differs")
    _require(
        checkpoint["weights_sha256"] == sha256_json(weights) == EXPECTED_WEIGHTS_SHA256,
        f"{name} canonical weight rollup differs",
    )
    _require(
        checkpoint["pinned_weights_rollup"]
        == _b7.legacy_weights_rollup(weights)
        == EXPECTED_WEIGHTS_ROLLUP,
        f"{name} pinned weight rollup differs",
    )
    _require(
        _same_json(checkpoint["metadata_sha256"], EXPECTED_METADATA_SHA256),
        f"{name} metadata differs",
    )
    _require(
        checkpoint["weight_index_sha256"] == EXPECTED_WEIGHT_INDEX_SHA256, f"{name} index differs"
    )
    _require(
        checkpoint["model_config_sha256"] == EXPECTED_MODEL_CONFIG_SHA256, f"{name} config differs"
    )
    return copy.deepcopy(checkpoint)


def _checkpoint(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "checkpoint evidence is not an object")
    checkpoint = dict(value)
    _exact_fields(
        checkpoint,
        frozenset({"measurement_start", "measurement_end", "stable_across_measurement"}),
        "checkpoint evidence",
    )
    start = _checkpoint_snapshot(checkpoint["measurement_start"], "checkpoint start")
    end = _checkpoint_snapshot(checkpoint["measurement_end"], "checkpoint end")
    _require(checkpoint["stable_across_measurement"] is True, "checkpoint stability flag differs")
    _require(_same_json(start, end), "checkpoint changed across measurement")
    return {"measurement_start": start, "measurement_end": end, "stable_across_measurement": True}


def _hardware(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "hardware evidence is not an object")
    hardware = dict(value)
    _exact_fields(
        hardware, frozenset({"gpu_count", "gpu_exclusive", "gpus", "software"}), "hardware"
    )
    _require(hardware["gpu_count"] == TENSOR_PARALLEL_SIZE, "hardware GPU count differs")
    _require(hardware["gpu_exclusive"] is True, "formal GPU allocation is not exclusive")
    gpus = hardware["gpus"]
    _require(
        isinstance(gpus, list) and len(gpus) == TENSOR_PARALLEL_SIZE, "hardware GPU rows differ"
    )
    normalized: list[dict[str, object]] = []
    for rank, raw in enumerate(gpus):
        _require(isinstance(raw, dict), f"GPU {rank} descriptor is not an object")
        row = dict(raw)
        _exact_fields(
            row,
            frozenset(
                {
                    "rank",
                    "visible_index",
                    "name",
                    "uuid",
                    "pci_bus_id",
                    "compute_capability",
                    "total_memory_bytes",
                }
            ),
            f"GPU {rank}",
        )
        _require(row["rank"] == rank and row["visible_index"] == rank, f"GPU {rank} rank differs")
        _require(row["name"] == GPU_NAME, f"GPU {rank} model differs")
        _require(
            isinstance(row["uuid"], str) and _UUID.fullmatch(row["uuid"]),
            f"GPU {rank} UUID is invalid",
        )
        _require(
            isinstance(row["pci_bus_id"], str) and _PCI.fullmatch(row["pci_bus_id"]),
            f"GPU {rank} PCI id is invalid",
        )
        _require(
            row["compute_capability"] == GPU_COMPUTE_CAPABILITY,
            f"GPU {rank} compute capability differs",
        )
        _integer(
            row["total_memory_bytes"],
            f"GPU {rank} memory",
            minimum=100_000_000_000,
        )
        normalized.append(row)
    _require(
        len({row["uuid"] for row in normalized}) == TENSOR_PARALLEL_SIZE, "GPU UUIDs are not unique"
    )
    _require(
        len({row["pci_bus_id"] for row in normalized}) == TENSOR_PARALLEL_SIZE,
        "GPU PCI ids are not unique",
    )
    software = hardware["software"]
    _require(isinstance(software, dict), "hardware software is not an object")
    _exact_fields(
        software,
        frozenset(
            {
                "python_version",
                "torch_version",
                "cuda_version",
                "driver_version",
                "nccl_version",
                "flashinfer_version",
            }
        ),
        "hardware software",
    )
    for field, version in software.items():
        _string(version, f"hardware software {field}")
    return {**hardware, "gpus": normalized, "software": dict(software)}


def _runtime(value: object, runtime_id: str, hardware: Mapping[str, object]) -> dict[str, object]:
    _require(isinstance(value, dict), f"{runtime_id} runtime is not an object")
    runtime = dict(value)
    _exact_fields(
        runtime,
        frozenset({"runtime_id", "nonce", "engine", "topology", "attention"}),
        f"{runtime_id} runtime",
    )
    _require(runtime["runtime_id"] == runtime_id, f"{runtime_id} runtime id differs")
    _require(_valid_hex(runtime["nonce"], 32), f"{runtime_id} runtime nonce is invalid")
    expected_engine = FULL_RUNTIME_CONFIG if runtime_id == "full" else CHUNK_RUNTIME_CONFIG
    _require(_same_json(runtime["engine"], expected_engine), f"{runtime_id} engine config differs")
    topology = runtime["topology"]
    _require(isinstance(topology, dict), f"{runtime_id} topology is not an object")
    _exact_fields(
        topology,
        frozenset(
            {
                "parallelism",
                "control_backend",
                "model_backend",
                "rank_gpu_uuid_order",
                "sampling_ownership",
            }
        ),
        f"{runtime_id} topology",
    )
    _require(
        topology["parallelism"] == {"tensor": 8, "expert": 1, "pipeline": 1},
        f"{runtime_id} parallelism differs",
    )
    _require(
        topology["control_backend"] == "gloo" and topology["model_backend"] == "nccl",
        f"{runtime_id} communicators differ",
    )
    expected_uuids = [row["uuid"] for row in hardware["gpus"]]
    _require(
        topology["rank_gpu_uuid_order"] == expected_uuids, f"{runtime_id} rank/GPU order differs"
    )
    ownership = topology["sampling_ownership"]
    expected_ownership = [
        {"rank": rank, "world_size": TENSOR_PARALLEL_SIZE, "sampling_owner": rank == 0}
        for rank in range(TENSOR_PARALLEL_SIZE)
    ]
    _require(ownership == expected_ownership, f"{runtime_id} sampling ownership differs")
    attention = runtime["attention"]
    _require(isinstance(attention, dict), f"{runtime_id} attention is not an object")
    _exact_fields(
        attention,
        frozenset(
            {"requested_backend", "selected_backend", "execution_identity", "fallback_reason"}
        ),
        f"{runtime_id} attention",
    )
    _require(
        attention["requested_backend"] == "flashinfer"
        and attention["selected_backend"] == "flashinfer"
        and isinstance(attention["execution_identity"], str)
        and bool(attention["execution_identity"])
        and attention["fallback_reason"] is None,
        f"{runtime_id} did not select exact FlashInfer execution",
    )
    identity_text = str(attention["execution_identity"])
    identity = _loads(
        identity_text,
        description=f"{runtime_id} attention execution identity",
    )
    _require(
        isinstance(identity, dict)
        and set(identity)
        == {
            "requested",
            "resolved",
            "source",
            "components",
            "component_versions",
            "component_builds",
            "architecture",
        },
        f"{runtime_id} attention execution identity fields differ",
    )
    _require(
        identity_text == canonical_json(identity),
        f"{runtime_id} attention execution identity is not canonical",
    )
    _require(
        identity["requested"] == attention["requested_backend"] == "flashinfer"
        and identity["resolved"] == attention["selected_backend"] == "flashinfer"
        and identity["source"] == "env"
        and identity["components"]
        == {
            "prefill": "flashinfer",
            "decode": "flashinfer",
            "kv_mode": "paged-direct",
        },
        f"{runtime_id} attention execution selection differs",
    )
    flashinfer_version = hardware["software"]["flashinfer_version"]
    _require(
        identity["component_versions"]
        == {"prefill": flashinfer_version, "decode": flashinfer_version},
        f"{runtime_id} FlashInfer component versions differ",
    )
    _require(
        identity["architecture"]
        == {
            "arch": "cuda",
            "device_name": GPU_NAME,
            "sm": 120,
            "kernel_tier": "fa2",
        },
        f"{runtime_id} attention architecture differs",
    )
    builds = identity["component_builds"]
    _require(
        isinstance(builds, dict) and set(builds) == {"flashinfer-jit-cache"},
        f"{runtime_id} FlashInfer build identity differs",
    )
    jit_cache = builds["flashinfer-jit-cache"]
    _require(isinstance(jit_cache, dict), f"{runtime_id} JIT-cache identity is invalid")
    installed = jit_cache.get("installed")
    _require(type(installed) is bool, f"{runtime_id} JIT-cache installed flag is invalid")
    if installed:
        _require(
            set(jit_cache) == {"installed", "version", "record_sha256"}
            and jit_cache["version"] == flashinfer_version
            and _valid_sha(jit_cache["record_sha256"]),
            f"{runtime_id} installed JIT-cache identity is invalid",
        )
    else:
        _require(
            jit_cache == {"installed": False},
            f"{runtime_id} absent JIT-cache identity is invalid",
        )
    return copy.deepcopy(runtime)


def _usage(value: object, name: str) -> dict[str, int]:
    _require(isinstance(value, dict), f"{name} is not an object")
    usage = dict(value)
    _exact_fields(usage, frozenset({"prompt_tokens", "completion_tokens", "cached_tokens"}), name)
    return {field: _integer(usage[field], f"{name} {field}") for field in usage}


def _response(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} is not an object")
    response = dict(value)
    _exact_fields(
        response,
        frozenset({"status", "token_ids", "token_pieces", "text", "finish_reason", "usage"}),
        name,
    )
    _require(response["status"] in {"success", "error"}, f"{name} status is invalid")
    ids = response["token_ids"]
    pieces = response["token_pieces"]
    _require(
        isinstance(ids, list)
        and len(ids) <= OUTPUT_TOKENS
        and all(type(token_id) is int and 0 <= token_id < TOKENIZER_VOCAB_SIZE for token_id in ids),
        f"{name} output token ids are invalid",
    )
    _require(
        isinstance(pieces, list)
        and len(pieces) == len(ids)
        and all(isinstance(piece, str) and bool(piece) for piece in pieces),
        f"{name} token pieces are invalid",
    )
    _string(response["text"], f"{name} text", allow_empty=True)
    _require(
        response["finish_reason"] in {None, "length", "stop", "error"},
        f"{name} finish reason is invalid",
    )
    usage = _usage(response["usage"], f"{name} usage")
    _require(usage["completion_tokens"] == len(ids), f"{name} completion usage differs")
    return {**response, "token_ids": list(ids), "token_pieces": list(pieces), "usage": usage}


def _cache(value: object, name: str, *, prompt_tokens: int) -> dict[str, int]:
    _require(isinstance(value, dict), f"{name} is not an object")
    cache = dict(value)
    _exact_fields(cache, frozenset({"probe_before", "native_cached_tokens"}), name)
    probe = _integer(cache["probe_before"], f"{name} probe")
    native = _integer(cache["native_cached_tokens"], f"{name} native usage")
    _require(probe <= prompt_tokens and native <= prompt_tokens, f"{name} exceeds prompt")
    return {"probe_before": probe, "native_cached_tokens": native}


def _scheduler_steps(value: object, name: str) -> list[dict[str, object]]:
    _require(isinstance(value, list), f"{name} is not a list")
    steps: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        _require(isinstance(raw, dict), f"{name} step {index} is not an object")
        step = dict(raw)
        _exact_fields(
            step,
            frozenset({"step_index", "kind", "request_ids", "num_tokens"}),
            f"{name} step {index}",
        )
        _require(step["step_index"] == index, f"{name} step indices are not contiguous")
        _require(step["kind"] in {"prefill", "decode"}, f"{name} step kind is invalid")
        request_ids = step["request_ids"]
        tokens = step["num_tokens"]
        _require(
            isinstance(request_ids, list)
            and bool(request_ids)
            and all(isinstance(item, str) and item for item in request_ids),
            f"{name} request ids are invalid",
        )
        _require(
            isinstance(tokens, list)
            and len(tokens) == len(request_ids)
            and all(type(item) is int and item >= 1 for item in tokens),
            f"{name} token counts are invalid",
        )
        steps.append({**step, "request_ids": list(request_ids), "num_tokens": list(tokens)})
    return steps


def _prefill_rank_rows(value: object, name: str) -> list[dict[str, object]]:
    _require(isinstance(value, list), f"{name} is not a list")
    result: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        _require(isinstance(raw, dict), f"{name} row {index} is not an object")
        row = dict(raw)
        _exact_fields(
            row, frozenset({"rank", "world_size", "device", "stats"}), f"{name} row {index}"
        )
        _integer(row["rank"], f"{name} rank")
        _integer(row["world_size"], f"{name} world size", minimum=1)
        _string(row["device"], f"{name} device")
        stats = row["stats"]
        _require(isinstance(stats, dict), f"{name} stats are not an object")
        _exact_fields(
            stats,
            frozenset(
                {
                    "enabled",
                    "capability_gap",
                    "rows",
                    "model_calls",
                    "batched_groups",
                    "sequential_rows",
                    "backend",
                }
            ),
            f"{name} stats",
        )
        _require(type(stats["enabled"]) is bool, f"{name} enabled flag is invalid")
        _require(
            stats["capability_gap"] is None or isinstance(stats["capability_gap"], str),
            f"{name} capability gap is invalid",
        )
        for field in ("rows", "model_calls", "batched_groups", "sequential_rows"):
            _integer(stats[field], f"{name} {field}")
        backend = stats["backend"]
        _require(isinstance(backend, list), f"{name} backend rows are invalid")
        for backend_row in backend:
            _require(isinstance(backend_row, dict), f"{name} backend row is not an object")
            _exact_fields(backend_row, frozenset({"type", "plans", "runs"}), f"{name} backend row")
            _string(backend_row["type"], f"{name} backend type")
            _integer(backend_row["plans"], f"{name} backend plans")
            _integer(backend_row["runs"], f"{name} backend runs")
        result.append(copy.deepcopy(row))
    return result


def _decode_rank_rows(value: object, name: str) -> list[dict[str, object]]:
    _require(isinstance(value, list), f"{name} is not a list")
    result: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        _require(isinstance(raw, dict), f"{name} row {index} is not an object")
        row = dict(raw)
        _exact_fields(
            row, frozenset({"rank", "world_size", "device", "stats"}), f"{name} row {index}"
        )
        _integer(row["rank"], f"{name} rank")
        _integer(row["world_size"], f"{name} world size", minimum=1)
        _string(row["device"], f"{name} device")
        stats = row["stats"]
        _require(isinstance(stats, dict), f"{name} stats are not an object")
        _exact_fields(
            stats,
            frozenset(
                {
                    "enabled",
                    "capability_gap",
                    "requests",
                    "positions",
                    "model_calls",
                    "batched_groups",
                    "sequential_positions",
                    "graph_groups",
                    "graph",
                    "backend",
                }
            ),
            f"{name} stats",
        )
        _require(type(stats["enabled"]) is bool, f"{name} enabled flag is invalid")
        _require(
            stats["capability_gap"] is None or isinstance(stats["capability_gap"], str),
            f"{name} capability gap is invalid",
        )
        # These are speculative-verification counters.  Ordinary decode does
        # not use them as proof; retain only their scalar type for auditability.
        for field in (
            "requests",
            "positions",
            "model_calls",
            "batched_groups",
            "sequential_positions",
            "graph_groups",
        ):
            _integer(stats[field], f"{name} {field}")
        graph = stats["graph"]
        _require(isinstance(graph, dict), f"{name} graph stats are not an object")
        _exact_fields(
            graph,
            frozenset({"graph_executions", "eager_fallbacks", "captured_buckets"}),
            f"{name} graph stats",
        )
        _integer(graph["graph_executions"], f"{name} graph executions")
        _integer(graph["eager_fallbacks"], f"{name} eager fallbacks")
        buckets = graph["captured_buckets"]
        _require(
            isinstance(buckets, list)
            and all(type(bucket) is int and bucket >= 1 for bucket in buckets),
            f"{name} graph buckets are invalid",
        )
        backend = stats["backend"]
        _require(isinstance(backend, list), f"{name} backend rows are invalid")
        for backend_row in backend:
            _require(isinstance(backend_row, dict), f"{name} backend row is not an object")
            _exact_fields(
                backend_row,
                frozenset(
                    {
                        "type",
                        "plans",
                        "runs",
                        "fast_replay_plans",
                        "stock_replay_fallbacks",
                        "replay_fallback_reason",
                    }
                ),
                f"{name} backend row",
            )
            _string(backend_row["type"], f"{name} backend type")
            for field in ("plans", "runs", "fast_replay_plans", "stock_replay_fallbacks"):
                _integer(backend_row[field], f"{name} backend {field}")
            _require(
                backend_row["replay_fallback_reason"] is None
                or isinstance(backend_row["replay_fallback_reason"], str),
                f"{name} fallback reason is invalid",
            )
        result.append(copy.deepcopy(row))
    return result


def _structure(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} is not an object")
    structure = dict(value)
    _exact_fields(
        structure, frozenset({"scheduler_steps", "prefill_rank_stats", "decode_rank_stats"}), name
    )
    return {
        "scheduler_steps": _scheduler_steps(structure["scheduler_steps"], f"{name} scheduler"),
        "prefill_rank_stats": _prefill_rank_rows(
            structure["prefill_rank_stats"], f"{name} prefill ranks"
        ),
        "decode_rank_stats": _decode_rank_rows(
            structure["decode_rank_stats"], f"{name} decode ranks"
        ),
    }


def _successful_response(
    response: Mapping[str, object], error_type: object, retry_count: object
) -> bool:
    usage = response["usage"]
    return (
        response["status"] == "success"
        and response["finish_reason"] == "length"
        and len(response["token_ids"]) == OUTPUT_TOKENS
        and len(response["token_pieces"]) == OUTPUT_TOKENS
        and isinstance(response["text"], str)
        and bool(response["text"])
        and usage["completion_tokens"] == OUTPUT_TOKENS
        and error_type is None
        and retry_count == 0
    )


def _answer(response: Mapping[str, object]) -> dict[str, object]:
    return {
        "token_ids": response["token_ids"],
        "token_pieces": response["token_pieces"],
        "text": response["text"],
    }


def _expected_scheduler(scenario: str, request_ids: Sequence[str]) -> list[dict[str, object]]:
    ids = list(request_ids)
    steps: list[dict[str, object]] = []
    if scenario == "batch32-cold":
        prefill_chunks = [[TARGET_PROMPT_TOKENS] + [1] * DISTRACTOR_COUNT]
        prefill_ids = [ids]
        decode_groups = [(ids, [1] * BATCH_SIZE)] * (OUTPUT_TOKENS - 1)
    elif scenario == "pressure":
        prefill_chunks = []
        prefill_ids = []
        decode_groups = []
        for request_id in ids:
            prefill_chunks.append([TARGET_PROMPT_TOKENS])
            prefill_ids.append([request_id])
            decode_groups.extend([([request_id], [1])] * (OUTPUT_TOKENS - 1))
    elif scenario == "alone-warm":
        prefill_chunks = [[1]]
        prefill_ids = [ids]
        decode_groups = [(ids, [1])] * (OUTPUT_TOKENS - 1)
    elif scenario == "chunked-alone-cold":
        prefill_chunks = [[32], [32], [32], [32], [1]]
        prefill_ids = [ids] * 5
        decode_groups = [(ids, [1])] * (OUTPUT_TOKENS - 1)
    else:
        prefill_chunks = [[TARGET_PROMPT_TOKENS]]
        prefill_ids = [ids]
        decode_groups = [(ids, [1])] * (OUTPUT_TOKENS - 1)
    for step_ids, chunks in zip(prefill_ids, prefill_chunks, strict=True):
        steps.append(
            {
                "step_index": len(steps),
                "kind": "prefill",
                "request_ids": step_ids,
                "num_tokens": chunks,
            }
        )
        if scenario == "pressure":
            current = step_ids[0]
            for _ in range(OUTPUT_TOKENS - 1):
                steps.append(
                    {
                        "step_index": len(steps),
                        "kind": "decode",
                        "request_ids": [current],
                        "num_tokens": [1],
                    }
                )
    if scenario != "pressure":
        for step_ids, chunks in decode_groups:
            steps.append(
                {
                    "step_index": len(steps),
                    "kind": "decode",
                    "request_ids": step_ids,
                    "num_tokens": chunks,
                }
            )
    return steps


def _stats_contract(
    structure: Mapping[str, object],
    *,
    scenario: str,
    expected_prefill_rows: int,
    expected_prefill_calls: int,
    expected_batched_groups: int,
    expected_sequential_rows: int,
    expected_prefill_backend_calls: int,
    expected_decode_stock_calls: int,
    expected_graph_executions: int,
    expected_bucket: int,
) -> bool:
    prefill = structure["prefill_rank_stats"]
    decode = structure["decode_rank_stats"]
    if not isinstance(prefill, list) or not isinstance(decode, list):
        return False
    ranks = list(range(TENSOR_PARALLEL_SIZE))
    if [row["rank"] for row in prefill] != ranks or [row["rank"] for row in decode] != ranks:
        return False
    for rows in (prefill, decode):
        if any(row["world_size"] != TENSOR_PARALLEL_SIZE for row in rows):
            return False
        if any(row["device"] != f"cuda:{row['rank']}" for row in rows):
            return False
        if len({canonical_json(row["stats"]) for row in rows}) != 1:
            return False
    for row in prefill:
        stats = row["stats"]
        backend = stats["backend"]
        if not (
            stats["enabled"] is True
            and stats["capability_gap"] is None
            and stats["rows"] == expected_prefill_rows
            and stats["model_calls"] == expected_prefill_calls
            and stats["batched_groups"] == expected_batched_groups
            and stats["sequential_rows"] == expected_sequential_rows
            and backend
            == [
                {
                    "type": "FlashInferBackend",
                    "plans": expected_prefill_backend_calls,
                    "runs": expected_prefill_backend_calls * NUM_LAYERS,
                }
            ]
        ):
            return False
    configured = set(CUDA_GRAPH_BUCKETS)
    captures_bucket = scenario in {
        "batch32-cold",
        "pressure",
        "chunked-alone-cold",
    }
    # A new FlashInfer graph bucket performs one explicit stock plan before
    # warmup plus two lazy stock replans between the three non-capturing
    # warmups.  The capture forward reuses the third plan; each real replay
    # contributes one fast plan.
    # A terminal one-token prefill remains a scheduler/model prefill call but
    # Qwen attention deliberately dispatches it through the decode-shaped
    # FlashInfer backend.  Its one stock plan/run is therefore visible in the
    # decode backend counters before graph replay/capture accounting.
    expected_decode_plans = (
        expected_graph_executions
        + expected_decode_stock_calls
        + (3 if captures_bucket else 0)
    )
    expected_decode_runs = (
        (int(FULL_RUNTIME_CONFIG["cuda_graph_warmup_iters"]) + 1) * NUM_LAYERS
        if captures_bucket
        else 0
    ) + expected_decode_stock_calls * NUM_LAYERS
    for row in decode:
        stats = row["stats"]
        graph = stats["graph"]
        backend = stats["backend"]
        buckets = graph["captured_buckets"]
        verification_fields = (
            "requests",
            "positions",
            "model_calls",
            "batched_groups",
            "sequential_positions",
            "graph_groups",
        )
        if not (
            all(stats[field] == 0 for field in verification_fields)
            and stats["enabled"] is True
            and stats["capability_gap"] is None
            and graph["graph_executions"] == expected_graph_executions
            and graph["eager_fallbacks"] == 0
            and expected_bucket in buckets
            and buckets == sorted(set(buckets))
            and set(buckets) <= configured
            and backend
            == [
                {
                    "type": "FlashInferBackend",
                    "plans": expected_decode_plans,
                    "runs": expected_decode_runs,
                    "fast_replay_plans": expected_graph_executions,
                    "stock_replay_fallbacks": 0,
                    "replay_fallback_reason": None,
                }
            ]
        ):
            return False
    return True


def _request(
    value: object,
    *,
    run_id: str,
    scenario: str,
    runtime_id: str,
    runtime_nonce: str,
    request_id: str,
    role: str,
    cohort_index: int,
    cohort_size: int,
    expected_prompt: Mapping[str, object],
    full_structure: bool,
) -> dict[str, object]:
    _require(isinstance(value, dict), f"{request_id} row is not an object")
    row = dict(value)
    _exact_fields(row, _REQUEST_FIELDS, f"{request_id} row")
    _require(row["row_type"] == "request", f"{request_id} row type differs")
    _require(row["run_id"] == run_id, f"{request_id} run id differs")
    _require(row["scenario"] == scenario, f"{request_id} scenario differs")
    _require(
        row["runtime_id"] == runtime_id and row["runtime_nonce"] == runtime_nonce,
        f"{request_id} runtime differs",
    )
    _require(row["request_id"] == request_id, f"{request_id} identity differs")
    _require(row["role"] == role, f"{request_id} role differs")
    _require(
        row["cohort_index"] == cohort_index and row["cohort_size"] == cohort_size,
        f"{request_id} cohort differs",
    )
    prompt = _prompt_descriptor(row["prompt"], f"{request_id} prompt", expected=expected_prompt)
    _require(_same_json(row["sampling"], SAMPLING_CONFIG), f"{request_id} sampling differs")
    cache = _cache(row["cache"], f"{request_id} cache", prompt_tokens=int(prompt["token_count"]))
    response = _response(row["response"], f"{request_id} response")
    _require(
        response["usage"]["prompt_tokens"] == prompt["token_count"],
        f"{request_id} prompt usage differs",
    )
    _require(
        response["usage"]["cached_tokens"] == cache["native_cached_tokens"],
        f"{request_id} cache usage differs",
    )
    _require(
        row["error_type"] is None or isinstance(row["error_type"], str),
        f"{request_id} error type is invalid",
    )
    retry_count = _integer(row["retry_count"], f"{request_id} retry count")
    structure: dict[str, object] | None
    if full_structure:
        structure = _structure(row["structure"], f"{scenario} structure")
        _require(
            row["structure_sha256"] == sha256_json(structure),
            f"{scenario} structure digest differs",
        )
    else:
        _require(row["structure"] is None, f"{request_id} duplicates shared structure")
        _require(_valid_sha(row["structure_sha256"]), f"{request_id} structure digest is invalid")
        structure = None
    return {
        **row,
        "prompt": prompt,
        "cache": cache,
        "response": response,
        "retry_count": retry_count,
        "structure": structure,
    }


def _pressure_request(value: object, index: int) -> dict[str, object]:
    request_id = f"pressure-{index:02d}"
    _require(isinstance(value, dict), f"{request_id} is not an object")
    request = dict(value)
    _exact_fields(request, _PRESSURE_REQUEST_FIELDS, request_id)
    _require(request["request_id"] == request_id, f"{request_id} identity differs")
    prompt = _prompt_descriptor(
        request["prompt"], f"{request_id} prompt", expected=fixed_prompts()["pressure"][index]
    )
    _require(_same_json(request["sampling"], SAMPLING_CONFIG), f"{request_id} sampling differs")
    cache = _cache(request["cache"], f"{request_id} cache", prompt_tokens=TARGET_PROMPT_TOKENS)
    response = _response(request["response"], f"{request_id} response")
    _require(
        response["usage"]["prompt_tokens"] == TARGET_PROMPT_TOKENS,
        f"{request_id} prompt usage differs",
    )
    _require(
        response["usage"]["cached_tokens"] == cache["native_cached_tokens"],
        f"{request_id} cache usage differs",
    )
    _require(
        request["error_type"] is None or isinstance(request["error_type"], str),
        f"{request_id} error type is invalid",
    )
    retry = _integer(request["retry_count"], f"{request_id} retry count")
    return {**request, "prompt": prompt, "cache": cache, "response": response, "retry_count": retry}


def _scenario_structure_checks(
    scenario: str,
    structure: Mapping[str, object],
    request_ids: Sequence[str],
) -> dict[str, bool]:
    expected_shape = _expected_scheduler(scenario, request_ids)
    profiles = {
        "batch32-cold": (32, 1, 1, 0, 1, 0, 31, 32),
        "pressure": (6, 6, 0, 6, 6, 0, 186, 1),
        "alone-cold": (1, 1, 0, 1, 1, 0, 31, 1),
        "alone-warm": (1, 1, 0, 1, 0, 1, 31, 1),
        "chunked-alone-cold": (5, 5, 0, 5, 4, 1, 31, 1),
    }
    (
        prefill_rows,
        calls,
        batched,
        sequential,
        prefill_backend_calls,
        decode_stock_calls,
        graphs,
        bucket,
    ) = profiles[scenario]
    return {
        f"{scenario}_scheduler_shape_exact": _same_json(
            structure["scheduler_steps"], expected_shape
        ),
        f"{scenario}_all_rank_flashinfer_graph_stats_exact": _stats_contract(
            structure,
            scenario=scenario,
            expected_prefill_rows=prefill_rows,
            expected_prefill_calls=calls,
            expected_batched_groups=batched,
            expected_sequential_rows=sequential,
            expected_prefill_backend_calls=prefill_backend_calls,
            expected_decode_stock_calls=decode_stock_calls,
            expected_graph_executions=graphs,
            expected_bucket=bucket,
        ),
    }


def recompute_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    """Strictly parse raw A12 rows and independently derive its verdict."""

    _require(_valid_sha(raw_sha256), "raw SHA-256 is invalid")
    _require(len(rows) == 38, "A12 raw evidence must contain exactly 38 rows")
    _require(all(isinstance(row, Mapping) for row in rows), "every raw row must be an object")
    _require(
        all(
            row.get("row_index") == index and type(row.get("row_index")) is int
            for index, row in enumerate(rows)
        ),
        "raw row indices are not contiguous",
    )

    header = dict(rows[0])
    _exact_fields(header, _RUN_START_FIELDS, "run_start")
    _require(
        header["row_type"] == "run_start" and header["row_index"] == 0,
        "raw does not start with run_start",
    )
    _require(
        header["schema_version"] == SCHEMA_VERSION and header["gate"] == GATE,
        "run identity differs",
    )
    run_id = _string(header["run_id"], "run id")
    _require(_same_json(header["config"], frozen_config()), "frozen A12 config differs")
    expected_prompts = fixed_prompts()
    _require(_same_json(header["prompts"], expected_prompts), "run prompt inventory differs")
    source = _source(header["source"])
    checkpoint = _checkpoint(header["checkpoint"])
    hardware = _hardware(header["hardware"])
    runtimes_value = header["runtimes"]
    _require(
        isinstance(runtimes_value, dict) and set(runtimes_value) == {"full", "chunk"},
        "runtime inventory differs",
    )
    runtimes = {
        runtime_id: _runtime(runtimes_value[runtime_id], runtime_id, hardware)
        for runtime_id in ("full", "chunk")
    }
    _require(
        runtimes["full"]["nonce"] != runtimes["chunk"]["nonce"], "runtime nonces are not fresh"
    )
    _require(
        runtimes["full"]["attention"]["execution_identity"]
        == runtimes["chunk"]["attention"]["execution_identity"],
        "full/chunk attention execution identities differ",
    )

    cursor = 1
    batch: list[dict[str, object]] = []
    batch_ids = ["batch32-target"] + [
        f"batch32-distractor-{index:02d}" for index in range(DISTRACTOR_COUNT)
    ]
    batch_prompts = [expected_prompts["target"], *expected_prompts["distractors"]]
    for index, (request_id, prompt) in enumerate(zip(batch_ids, batch_prompts, strict=True)):
        batch.append(
            _request(
                rows[cursor],
                run_id=run_id,
                scenario="batch32-cold",
                runtime_id="full",
                runtime_nonce=str(runtimes["full"]["nonce"]),
                request_id=request_id,
                role="target" if index == 0 else "distractor",
                cohort_index=index,
                cohort_size=BATCH_SIZE,
                expected_prompt=prompt,
                full_structure=index == 0,
            )
        )
        cursor += 1
    batch_structure = batch[0]["structure"]
    assert isinstance(batch_structure, dict)
    batch_structure_sha = batch[0]["structure_sha256"]
    _require(
        all(row["structure_sha256"] == batch_structure_sha for row in batch),
        "batch rows do not share one structure digest",
    )

    pressure_raw = rows[cursor]
    cursor += 1
    _require(isinstance(pressure_raw, Mapping), "pressure_summary is not an object")
    pressure = dict(pressure_raw)
    _exact_fields(pressure, _PRESSURE_FIELDS, "pressure_summary")
    _require(pressure["row_type"] == "pressure_summary", "pressure row type differs")
    _require(
        pressure["run_id"] == run_id and pressure["scenario"] == "pressure",
        "pressure identity differs",
    )
    _require(
        pressure["runtime_id"] == "full" and pressure["runtime_nonce"] == runtimes["full"]["nonce"],
        "pressure runtime differs",
    )
    target_before = _integer(pressure["target_cache_before"], "pressure target cache before")
    target_after = _integer(pressure["target_cache_after"], "pressure target cache after")
    pressure_requests_raw = pressure["requests"]
    _require(
        isinstance(pressure_requests_raw, list)
        and len(pressure_requests_raw) == MAX_PRESSURE_REQUESTS,
        "pressure request count differs",
    )
    pressure_requests = [
        _pressure_request(value, index) for index, value in enumerate(pressure_requests_raw)
    ]
    pressure_structure = _structure(pressure["structure"], "pressure structure")
    _require(
        pressure["structure_sha256"] == sha256_json(pressure_structure),
        "pressure structure digest differs",
    )

    target_rows: dict[str, dict[str, object]] = {"batch32-cold": batch[0]}
    singleton_specs = (
        ("alone-cold", "full", 0),
        ("alone-warm", "full", WARM_CACHED_TOKENS),
        ("chunked-alone-cold", "chunk", 0),
    )
    for scenario, runtime_id, expected_cache in singleton_specs:
        request_id = f"{scenario}-target"
        target = _request(
            rows[cursor],
            run_id=run_id,
            scenario=scenario,
            runtime_id=runtime_id,
            runtime_nonce=str(runtimes[runtime_id]["nonce"]),
            request_id=request_id,
            role="target",
            cohort_index=0,
            cohort_size=1,
            expected_prompt=expected_prompts["target"],
            full_structure=True,
        )
        target["expected_cache"] = expected_cache
        target_rows[scenario] = target
        cursor += 1

    end = dict(rows[cursor])
    _exact_fields(end, _RUN_END_FIELDS, "run_end")
    _require(end["row_type"] == "run_end" and end["run_id"] == run_id, "run_end identity differs")
    _require(end["status"] in {"complete", "failed"}, "run_end status is invalid")
    _require(
        isinstance(end["errors"], list)
        and all(isinstance(error, str) and error for error in end["errors"]),
        "run_end errors are invalid",
    )
    _require(
        end["runtime_count"] == 2 and end["scenario_count"] == len(SCENARIOS),
        "run_end counts differ",
    )

    checks: dict[str, bool] = {
        "run_completed_without_recorded_errors": end["status"] == "complete"
        and end["errors"] == [],
        "source_checkpoint_and_hardware_contract_exact": True,
        "two_fresh_native_runtime_identities_exact": runtimes["full"]["nonce"]
        != runtimes["chunk"]["nonce"],
        "batch32_all_requests_succeeded_once_at_32_tokens": all(
            _successful_response(row["response"], row["error_type"], row["retry_count"])
            for row in batch
        ),
        "batch32_all_prompts_were_native_cache_cold": all(
            row["cache"] == {"probe_before": 0, "native_cached_tokens": 0} for row in batch
        ),
        "pressure_requests_fixed_distinct_cold_and_successful": (
            len({row["prompt"]["token_ids_sha256"] for row in pressure_requests})
            == MAX_PRESSURE_REQUESTS
            and all(
                row["cache"] == {"probe_before": 0, "native_cached_tokens": 0}
                for row in pressure_requests
            )
            and all(
                _successful_response(row["response"], row["error_type"], row["retry_count"])
                for row in pressure_requests
            )
        ),
        "pressure_proves_target_evicted_before_alone_cold": target_before == WARM_CACHED_TOKENS
        and target_after == 0,
    }
    checks.update(_scenario_structure_checks("batch32-cold", batch_structure, batch_ids))
    checks.update(
        _scenario_structure_checks(
            "pressure", pressure_structure, [row["request_id"] for row in pressure_requests]
        )
    )

    expected_cache_by_scenario = {
        "batch32-cold": 0,
        "alone-cold": 0,
        "alone-warm": WARM_CACHED_TOKENS,
        "chunked-alone-cold": 0,
    }
    for scenario in TARGET_SCENARIOS:
        target = target_rows[scenario]
        expected_cache = expected_cache_by_scenario[scenario]
        checks[f"{scenario}_target_succeeded_once_at_32_tokens"] = _successful_response(
            target["response"], target["error_type"], target["retry_count"]
        )
        checks[f"{scenario}_cache_probe_and_native_usage_exact"] = target["cache"] == {
            "probe_before": expected_cache,
            "native_cached_tokens": expected_cache,
        }
        if scenario != "batch32-cold":
            structure = target["structure"]
            assert isinstance(structure, dict)
            checks.update(
                _scenario_structure_checks(
                    scenario,
                    structure,
                    [str(target["request_id"])],
                )
            )

    answers = {
        scenario: _answer(target_rows[scenario]["response"]) for scenario in TARGET_SCENARIOS
    }
    answer_serializations = {canonical_json(answer) for answer in answers.values()}
    checks["four_target_native_answers_are_exactly_identical"] = len(answer_serializations) == 1
    checks["target_prompt_identity_is_exact_in_all_four_scenarios"] = all(
        _same_json(target_rows[scenario]["prompt"], expected_prompts["target"])
        for scenario in TARGET_SCENARIOS
    )
    checks["chunked_prefill_is_exactly_32_32_32_32_1"] = [
        step["num_tokens"][0]
        for step in target_rows["chunked-alone-cold"]["structure"]["scheduler_steps"]
        if step["kind"] == "prefill"
    ] == [32, 32, 32, 32, 1]

    scenario_metrics = {
        scenario: {
            "answer_sha256": sha256_json(answers[scenario]),
            "output_token_ids_sha256": sha256_json(answers[scenario]["token_ids"]),
            "cached_tokens": target_rows[scenario]["cache"]["native_cached_tokens"],
            "scheduler_step_count": len(target_rows[scenario]["structure"]["scheduler_steps"]),
            "structure_sha256": target_rows[scenario]["structure_sha256"],
        }
        for scenario in TARGET_SCENARIOS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "measurement_kind": MEASUREMENT_KIND,
        "run_id": run_id,
        "config": frozen_config(),
        "raw": {"name": RAW_NAME, "sha256": raw_sha256, "row_count": len(rows)},
        "provenance": {
            "source_sha256": sha256_json(source),
            "checkpoint_sha256": sha256_json(checkpoint),
            "hardware_sha256": sha256_json(hardware),
            "runtime_sha256": {key: sha256_json(value) for key, value in runtimes.items()},
        },
        "scenarios": scenario_metrics,
        "answer_equality": {
            "compared": list(TARGET_SCENARIOS),
            "exact_native_answer_count": len(answer_serializations),
            "common_answer_sha256": (
                sha256_json(next(iter(answers.values())))
                if len(answer_serializations) == 1
                else None
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _assert_gate_manifest(manifest: Mapping[str, object]) -> None:
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "gate manifest schema differs")
    _require(manifest.get("gate") == GATE, "gate manifest identity differs")
    checks = manifest.get("checks")
    _require(isinstance(checks, Mapping) and bool(checks), "gate manifest checks are missing")
    _require(all(type(value) is bool for value in checks.values()), "gate checks are not booleans")
    passed = all(checks.values())
    _require(manifest.get("passed") is passed, "gate manifest passed field is inconsistent")
    if not passed:
        failed = [str(name) for name, value in checks.items() if value is not True]
        raise EvidenceError(f"{GATE} failed: {', '.join(failed)}")


def assert_gate(manifest: Mapping[str, object]) -> None:
    """Raise unless a recomputed A12 manifest passes every derived check."""

    _require(isinstance(manifest, Mapping), "gate manifest is not an object")
    _assert_gate_manifest(manifest)


def replay_raw(path: Path, *, assert_gate: bool = False) -> dict[str, object]:
    """Recompute the complete manifest using raw JSONL only."""

    raw, rows = _read_jsonl_snapshot(Path(path))
    manifest = recompute_manifest(rows, raw_sha256=sha256_bytes(raw))
    if assert_gate:
        _assert_gate_manifest(manifest)
    return manifest


def _resolve_artifact_paths(raw_path: Path, manifest_path: Path | None) -> tuple[Path, Path]:
    raw = Path(raw_path)
    if manifest_path is not None:
        return raw, Path(manifest_path)
    if raw.is_dir():
        return raw / RAW_NAME, raw / MANIFEST_NAME
    if raw.name == RAW_NAME:
        return raw, raw.with_name(MANIFEST_NAME)
    if raw.name == MANIFEST_NAME:
        return raw.with_name(RAW_NAME), raw
    raise EvidenceError(
        f"artifact must be a directory, {RAW_NAME}, {MANIFEST_NAME}, or two explicit paths"
    )


def verify_artifact(
    raw_path: Path,
    manifest_path: Path | None = None,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    """Hash raw bytes and compare a retained manifest with independent replay."""

    raw_target, manifest_target = _resolve_artifact_paths(raw_path, manifest_path)
    replayed = replay_raw(raw_target, assert_gate=False)
    try:
        text = manifest_target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(f"cannot read retained manifest: {manifest_target}") from error
    _require(text.endswith("\n") and not text.endswith("\n\n"), "retained manifest framing differs")
    retained = _loads(text[:-1], description="retained manifest")
    _require(isinstance(retained, dict), "retained manifest is not an object")
    _require(text == canonical_json(retained) + "\n", "retained manifest is not canonical")
    _require(
        _same_json(retained, replayed), "retained manifest differs from independent raw replay"
    )
    if assert_gate:
        _assert_gate_manifest(replayed)
    return replayed


def write_artifact(
    raw_path: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    """Exclusively create, re-read, and independently verify one A12 artifact."""

    raw_target = Path(raw_path)
    manifest_target = Path(manifest_path)
    _require(raw_target.resolve() != manifest_target.resolve(), "raw and manifest paths collide")
    _require(
        not raw_target.exists() and not manifest_target.exists(),
        "refusing to overwrite retained A12 artifact",
    )
    raw_bytes = _jsonl_bytes(rows)
    manifest = recompute_manifest(rows, raw_sha256=sha256_bytes(raw_bytes))
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with raw_target.open("xb") as handle:
            handle.write(raw_bytes)
        with manifest_target.open("xb") as handle:
            handle.write(manifest_bytes)
    except FileExistsError as error:
        raise EvidenceError("refusing to overwrite retained A12 artifact") from error
    except OSError as error:
        raise EvidenceError("cannot create retained A12 artifact") from error
    # A successful return always describes bytes independently re-read from
    # disk.  This catches short writes and post-write mutation before handoff.
    verified = verify_artifact(raw_target, manifest_target, assert_gate=False)
    if assert_gate:
        _assert_gate_manifest(verified)
    return verified


__all__ = [
    "BATCH_SIZE",
    "CUDA_GRAPH_BUCKETS",
    "CHUNK_RUNTIME_CONFIG",
    "DISTRACTOR_COUNT",
    "DISTRACTOR_TEXTS",
    "DISTRACTOR_TOKEN_IDS",
    "EXPECTED_ARCHITECTURE",
    "EXPECTED_METADATA_SHA256",
    "EXPECTED_MODEL_CONFIG_SHA256",
    "EXPECTED_WEIGHTS",
    "EXPECTED_WEIGHTS_ROLLUP",
    "EXPECTED_WEIGHTS_SHA256",
    "EXPECTED_WEIGHT_INDEX_SHA256",
    "EvidenceError",
    "FULL_RUNTIME_CONFIG",
    "GATE",
    "GPU_COMPUTE_CAPABILITY",
    "GPU_NAME",
    "MANIFEST_NAME",
    "MAX_PRESSURE_REQUESTS",
    "MEASUREMENT_KIND",
    "MODEL",
    "MODEL_REVISION",
    "OUTPUT_TOKENS",
    "PAGE_SIZE",
    "PRESSURE_PROMPT_TEXTS",
    "PRESSURE_PROMPT_TOKEN_IDS",
    "RAW_NAME",
    "REQUIRED_SOURCE_PATHS",
    "SAMPLING_CONFIG",
    "SCENARIOS",
    "SCHEMA_VERSION",
    "TARGET_PROMPT_TEXT",
    "TARGET_PROMPT_TEXT_SHA256",
    "TARGET_PROMPT_TOKEN_IDS",
    "TARGET_PROMPT_TOKEN_IDS_SHA256",
    "TARGET_PROMPT_TOKENS",
    "TARGET_SCENARIOS",
    "TENSOR_PARALLEL_SIZE",
    "TOKENIZER_VOCAB_SIZE",
    "WARM_CACHED_TOKENS",
    "assert_gate",
    "canonical_json",
    "fixed_prompts",
    "frozen_config",
    "read_jsonl",
    "recompute_manifest",
    "replay_raw",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "verify_artifact",
    "write_artifact",
]
