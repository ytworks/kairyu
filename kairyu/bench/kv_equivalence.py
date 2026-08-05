"""Pure evidence contract for cache-cold/cache-warm answer equivalence.

The gate is an additive sidecar to F2c, F2d, F4a, and F4b.  It binds five
fixed production-topology cells, their reviewed parent evidence, the complete
Qwen3-32B checkpoint, and a clean source snapshot before accepting exact
native cold/warm greedy-answer equality.  Existing parent schemas and verdicts
remain untouched.

All verdicts are derived from canonical raw JSONL.  A retained ``passed``
field is never accepted without hashing and independently replaying that raw.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "kairyu.g5.kv-answer-equivalence.v1"
GATE = "G5-KV-ANSWER-EQUIVALENCE"
MEASUREMENT_KIND = "cache-cold-warm-greedy-answer-equivalence"
RAW_NAME = "kv-answer-equivalence-raw.jsonl"
MANIFEST_NAME = "kv-answer-equivalence-manifest.json"

OUTPUT_TOKENS = 32
PAGE_SIZE = 16
COMPARISON_PROMPT_TOKENS = 1024
MIN_DRAM_PROMPT_TOKENS = 1024
MAX_PRESSURE_REQUESTS = 128
FEATURES = ("f2c", "f2d", "f4a", "f4b")
CELL_IDS = ("f2c-tp2", "f2d-tp2", "f4a-tp4", "f4a-tp8", "f4b-tp4")

MODEL = "qwen3-32b"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
EXPECTED_WEIGHTS_SHA256 = "3c977c1b109feee4395ef000fea4bf8860ab251326a8fb7c75ee3050e439a387"
EXPECTED_WEIGHTS_ROLLUP = "6aa37b7da4e37d45b277d0bca47b00c2bfb58bb60986ba93e4c3e667df123955"
EXPECTED_MODEL_CONFIG_SHA256 = "f7b15097c58db24a59030f0e57775b2b49b33e2ee294fb85ff0ade5b902c489b"
EXPECTED_METADATA_SHA256 = {
    "config.json": "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb",
    "generation_config.json": ("2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": ("d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
EXPECTED_WEIGHT_INDEX_SHA256 = "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
EXPECTED_ARCHITECTURE = {
    "model_type": "qwen3",
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}
# Qwen3 pads the model's logit dimension beyond the tokenizer-owned ID space.
# Native prompt/output evidence can only use IDs exposed by tokenizer.json.
TOKENIZER_VOCAB_SIZE = 151669
EXPECTED_WEIGHTS: tuple[dict[str, object], ...] = (
    {
        "file": "model-00001-of-00017.safetensors",
        "bytes": 3957109648,
        "sha256": "52562b2ff97b61764260273e71bf5b4cf8a66f569399398f26dec0300fcf1316",
    },
    {
        "file": "model-00002-of-00017.safetensors",
        "bytes": 3900791760,
        "sha256": "e26764b2c6878e3fb7198895fa833ec62838d84a19665e6abfbae43c6daf02b3",
    },
    {
        "file": "model-00003-of-00017.safetensors",
        "bytes": 3900791760,
        "sha256": "6c5ba7bed9c52bc121e75cbe8a7be46936d0006cc80f42a6d5886ed40b4c2a62",
    },
    {
        "file": "model-00004-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "f736f6ac4d8c30866107fb1185a05b3c3cfce9717720082f466fa44e691bcec8",
    },
    {
        "file": "model-00005-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "a52ed375c083209c54d42ac510afeb1fbb5af4f193be2dc7d103f665a0f212d3",
    },
    {
        "file": "model-00006-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "37fae28990b0e4a70228549d040c0393e87bee3820d59e58e47844974d8dff5b",
    },
    {
        "file": "model-00007-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "37776006aeaba29eca8bc73b2b963fe3477e1c2e3f6a27cb9527be75b905e1bf",
    },
    {
        "file": "model-00008-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "73e74e9129674fe330948005075d70ab4fa0b92b68fb220c5a693f9cea553730",
    },
    {
        "file": "model-00009-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "a044b3602a01bd8ea62ff51badf9cc038ab1d73d97399480e6a55b4c86fa7fa6",
    },
    {
        "file": "model-00010-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "9966612ba7ecfc2cd2e592fb95224b86d743271ff88e172cc272a4b26382aa75",
    },
    {
        "file": "model-00011-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "e2a058a0ac7d4b992b731c29221ecfb4b76b8a48d9004d0e8a62ba44f699845c",
    },
    {
        "file": "model-00012-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "58a1aa89093fea07325f787072a468e3482a470ff4b7fe5ead5f749683907c40",
    },
    {
        "file": "model-00013-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "35f3381bab31a23370c37d922290aeecdf603418336058fb86fe42d8f51ac40c",
    },
    {
        "file": "model-00014-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "8713b062ddc178acf5917610b7f4b64eede833b2ea4aa37bd562dcf2f3a3339d",
    },
    {
        "file": "model-00015-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "bec439d23931821a236d8f62fa79deecf5551bd25602278aa2ae0ce432b378cf",
    },
    {
        "file": "model-00016-of-00017.safetensors",
        "bytes": 3900791800,
        "sha256": "e569139fadd61fe7c8f9eb1c976d9a627cae48c57ddf228cfbd0593c59c64ff7",
    },
    {
        "file": "model-00017-of-00017.safetensors",
        "bytes": 3055341992,
        "sha256": "1f47c318fcd7797c0f85b4233cb754438b10e795b8bc874889090c416a94bd38",
    },
)

REQUIRED_SOURCE_PATHS = (
    "bench/kv_answer_equivalence_bench.py",
    "kairyu/bench/kv_equivalence.py",
    "kairyu/engine/backend.py",
    "kairyu/engine/kairyu_backend.py",
    "kairyu/engine/prompt.py",
    "kairyu/engine/tokenizer.py",
    "kairyu/engine/core/kv_tier.py",
    "kairyu/engine/core/kv_tier_policy.py",
    "kairyu/engine/core/model_runner.py",
    "kairyu/engine/core/radix_kv.py",
    "kairyu/engine/core/scheduler.py",
    "kairyu/sampling_params.py",
    "pyproject.toml",
    "uv.lock",
)

FEATURE_SPECS: dict[str, dict[str, str]] = {
    "f2c": {
        "transition": "radix",
        "parent_gate": "G5-F2c",
        "parent_schema_version": "kairyu.f2c.kv-aware-ttft.v1",
        "proof_kind": "radix_reuse",
    },
    "f2d": {
        "transition": "radix",
        "parent_gate": "G5-F2d",
        "parent_schema_version": "kairyu.f2d.prefix-weight-replay.v1",
        "proof_kind": "radix_reuse",
    },
    "f4a": {
        "transition": "dram_restore",
        "parent_gate": "G5-F4a",
        "parent_schema_version": "kairyu.g5.f4a.dram-kv-tier.v2",
        "proof_kind": "dram_restore",
    },
    "f4b": {
        "transition": "dram_restore",
        "parent_gate": "G5-F4b",
        "parent_schema_version": "kairyu.g5.f4b.agentic-kv-tier.v1",
        "proof_kind": "dram_restore",
    },
}


def _cell_spec(
    feature_id: str,
    tensor_parallel_size: int,
    parent_binding_kind: str,
    parent_topology: Mapping[str, object],
    *,
    num_pages: int,
    dram_capacity_pages: int,
    decode_mode: str,
    max_num_batched_tokens: int,
    max_model_len: int,
    dram_profile_sha256: str | None,
    dram_min_restore_tokens: int,
) -> dict[str, object]:
    feature = FEATURE_SPECS[feature_id]
    return {
        "feature_id": feature_id,
        "transition": feature["transition"],
        "proof_kind": feature["proof_kind"],
        "tensor_parallel_size": tensor_parallel_size,
        "page_size": PAGE_SIZE,
        "num_pages": num_pages,
        "dram_capacity_pages": dram_capacity_pages,
        "dram_profile_sha256": dram_profile_sha256,
        "dram_min_restore_tokens": dram_min_restore_tokens,
        "decode_mode": decode_mode,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_model_len": max_model_len,
        "parent_gate": feature["parent_gate"],
        "parent_schema_version": feature["parent_schema_version"],
        "parent_binding_kind": parent_binding_kind,
        "parent_topology": dict(parent_topology),
        "dram_required": dram_capacity_pages > 0,
    }


CELL_SPECS: dict[str, dict[str, object]] = {
    "f2c-tp2": _cell_spec(
        "f2c",
        2,
        "direct-partial",
        {
            "kind": "direct",
            "tensor_parallel_size": 2,
            "service_count": 4,
            "service_device_ids": {
                "replica-a0": ["0", "1"],
                "replica-a1": ["2", "3"],
                "replica-b0": ["4", "5"],
                "replica-b1": ["6", "7"],
            },
            "unique_device_count": 8,
        },
        num_pages=8192,
        dram_capacity_pages=0,
        decode_mode="cuda_graph",
        max_num_batched_tokens=1024,
        max_model_len=8192,
        dram_profile_sha256=None,
        dram_min_restore_tokens=0,
    ),
    "f2d-tp2": _cell_spec(
        "f2d",
        2,
        "aggregate-common-f2c",
        {"kind": "aggregate-inherited", "from_cell_id": "f2c-tp2"},
        num_pages=8192,
        dram_capacity_pages=0,
        decode_mode="cuda_graph",
        max_num_batched_tokens=1024,
        max_model_len=8192,
        dram_profile_sha256=None,
        dram_min_restore_tokens=0,
    ),
    "f4a-tp4": _cell_spec(
        "f4a",
        4,
        "direct-full",
        {"kind": "direct", "tensor_parallel_size": 4},
        num_pages=1024,
        dram_capacity_pages=512,
        decode_mode="eager",
        max_num_batched_tokens=2048,
        max_model_len=8193,
        dram_profile_sha256=("7b73d4adb2cc6a89ec41d0ad9e36ce788ce761ca18796b56a31e0f73d97b041d"),
        dram_min_restore_tokens=1024,
    ),
    "f4a-tp8": _cell_spec(
        "f4a",
        8,
        "direct-full",
        {"kind": "direct", "tensor_parallel_size": 8},
        num_pages=1024,
        dram_capacity_pages=512,
        decode_mode="eager",
        max_num_batched_tokens=2048,
        max_model_len=8193,
        dram_profile_sha256=("e868482eba73efc56c5584224a7edd973efc989137c94187393db95c093511e7"),
        dram_min_restore_tokens=16,
    ),
    "f4b-tp4": _cell_spec(
        "f4b",
        4,
        "direct-full",
        {"kind": "direct", "tensor_parallel_size": 4},
        num_pages=1024,
        dram_capacity_pages=2048,
        decode_mode="eager",
        max_num_batched_tokens=2048,
        max_model_len=8192,
        dram_profile_sha256=("7b73d4adb2cc6a89ec41d0ad9e36ce788ce761ca18796b56a31e0f73d97b041d"),
        dram_min_restore_tokens=1024,
    ),
}

TIER_STATS_FIELDS = (
    "offload_pages",
    "offload_bypassed_pages",
    "restore_pages",
    "restore_attempts",
    "restore_fallbacks",
    "ownership_failures",
)

_HEX = frozenset("0123456789abcdef")
_RUN_FIELDS = frozenset(
    {"row_type", "schema_version", "gate", "run_id", "required_cells", "config"}
)
_CELL_START_FIELDS = frozenset(
    {
        "row_type",
        "run_id",
        "cell_id",
        "feature_id",
        "transition",
        "prompt",
        "sampling",
        "parent_evidence",
        "runtime",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "row_type",
        "run_id",
        "cell_id",
        "feature_id",
        "phase",
        "runtime_nonce",
        "prompt",
        "response",
        "error_type",
        "retry_count",
    }
)
_CELL_END_FIELDS = frozenset({"row_type", "run_id", "cell_id", "feature_id", "feature_proof"})
_RUN_END_FIELDS = frozenset({"row_type", "run_id", "status", "errors", "cell_count"})


class EvidenceError(ValueError):
    """Raw or retained equivalence evidence violates the sidecar contract."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def canonical_json(value: object) -> str:
    """Return the sole JSON serialization used by this evidence contract."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _same_json_value(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int/float equality coercions."""

    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return False


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_weights_rollup(weights: Sequence[Mapping[str, object]]) -> str:
    """Reproduce the retained F4a/F4b legacy weight-inventory digest."""

    return sha256_bytes(json.dumps(list(weights), sort_keys=True).encode("utf-8"))


def frozen_config() -> dict[str, object]:
    """Return a fresh copy of the complete frozen sidecar configuration."""

    return {
        "required_cells": list(CELL_IDS),
        "cell_topology": {
            cell_id: {
                "feature_id": CELL_SPECS[cell_id]["feature_id"],
                "tensor_parallel_size": CELL_SPECS[cell_id]["tensor_parallel_size"],
                "page_size": CELL_SPECS[cell_id]["page_size"],
                "num_pages": CELL_SPECS[cell_id]["num_pages"],
                "dram_capacity_pages": CELL_SPECS[cell_id]["dram_capacity_pages"],
                "dram_profile_sha256": CELL_SPECS[cell_id]["dram_profile_sha256"],
                "dram_min_restore_tokens": CELL_SPECS[cell_id]["dram_min_restore_tokens"],
                "decode_mode": CELL_SPECS[cell_id]["decode_mode"],
                "max_num_batched_tokens": CELL_SPECS[cell_id]["max_num_batched_tokens"],
                "max_model_len": CELL_SPECS[cell_id]["max_model_len"],
            }
            for cell_id in CELL_IDS
        },
        "sampling": {
            "temperature": 0.0,
            "min_tokens": OUTPUT_TOKENS,
            "max_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "seed": 0,
            "skip_special_tokens": False,
        },
        "output_tokens": OUTPUT_TOKENS,
        "comparison_prompt_tokens": COMPARISON_PROMPT_TOKENS,
        "request_order": {
            "radix": ["cold", "warm"],
            "dram_restore": ["cold", "pressure+", "warm"],
        },
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    _require(bool(rows), "raw JSONL is empty")
    return "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_jsonl_bytes(rows))


def _reject_constant(value: str) -> Any:
    raise EvidenceError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _loads_json(value: str, *, description: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except EvidenceError:
        raise
    except json.JSONDecodeError as error:
        raise EvidenceError(f"cannot parse {description}") from error


def _parse_jsonl_bytes(raw: bytes, *, target: Path) -> list[dict[str, object]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"cannot read raw JSONL: {target}") from error
    _require(bool(text), "raw JSONL is empty")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        _require(
            line.endswith("\n") and not line.endswith("\r\n"),
            f"invalid JSONL framing at line {line_number}",
        )
        payload = line[:-1]
        _require(bool(payload), f"empty JSONL row at line {line_number}")
        value = _loads_json(payload, description=f"JSONL line {line_number}")
        _require(isinstance(value, dict), f"JSONL line {line_number} is not an object")
        _require(payload == canonical_json(value), f"JSONL line {line_number} is not canonical")
        rows.append(value)
    _require(bool(rows), "raw JSONL is empty")
    return rows


def _read_jsonl_snapshot(path: Path) -> tuple[bytes, list[dict[str, object]]]:
    """Read once so replayed rows and their digest describe identical bytes."""

    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as error:
        raise EvidenceError(f"cannot read raw JSONL: {target}") from error
    return raw, _parse_jsonl_bytes(raw, target=target)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    _raw, rows = _read_jsonl_snapshot(path)
    return rows


def _valid_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and set(value) <= _HEX
    )


def _valid_sha256(value: object) -> bool:
    return _valid_hex(value, 64)


def _exact_fields(value: Mapping[str, object], fields: frozenset[str], name: str) -> None:
    _require(set(value) == fields, f"{name} fields differ from the frozen schema")


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{name} is invalid")
    return int(value)


def _nonempty_string(value: object, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} is invalid")
    return value


def _prompt_descriptor(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} is not an object")
    prompt = dict(value)
    _exact_fields(
        prompt,
        frozenset({"token_ids", "token_ids_sha256", "token_count"}),
        name,
    )
    token_ids = prompt["token_ids"]
    _require(
        isinstance(token_ids, list)
        and bool(token_ids)
        and all(type(token) is int and 0 <= token < TOKENIZER_VOCAB_SIZE for token in token_ids),
        f"{name} token ids are invalid",
    )
    _require(_valid_sha256(prompt["token_ids_sha256"]), f"{name} digest is invalid")
    token_count = _exact_int(prompt["token_count"], f"{name} token count", minimum=1)
    _require(token_count == len(token_ids), f"{name} token count differs from raw ids")
    _require(
        prompt["token_ids_sha256"] == sha256_json(token_ids),
        f"{name} digest differs from raw ids",
    )
    return {**prompt, "token_ids": list(token_ids)}


def _source_descriptor(value: object, cell_id: str) -> dict[str, object]:
    name = f"{cell_id} source"
    _require(isinstance(value, dict), f"{name} is not an object")
    source = dict(value)
    _exact_fields(
        source,
        frozenset({"git_head", "tracked_tree_clean", "file_sha256", "files_sha256"}),
        name,
    )
    _require(_valid_hex(source["git_head"], 40), f"{name} git head is invalid")
    _require(source["tracked_tree_clean"] is True, f"{name} is not a clean tracked tree")
    files = source["file_sha256"]
    _require(isinstance(files, dict), f"{name} file inventory is invalid")
    _require(set(files) == set(REQUIRED_SOURCE_PATHS), f"{name} file inventory differs")
    _require(all(_valid_sha256(digest) for digest in files.values()), f"{name} digest is invalid")
    _require(
        source["files_sha256"] == sha256_json(files),
        f"{name} file rollup differs",
    )
    return {**source, "file_sha256": dict(files)}


def _weight_inventory(value: object, name: str) -> list[dict[str, object]]:
    _require(isinstance(value, list), f"{name} is not a list")
    _require(len(value) == len(EXPECTED_WEIGHTS), f"{name} must bind all 17 shards")
    weights = [dict(row) if isinstance(row, dict) else None for row in value]
    _require(all(row is not None for row in weights), f"{name} contains a non-object")
    expected_files = [str(row["file"]) for row in EXPECTED_WEIGHTS]
    normalized: list[dict[str, object]] = []
    for index, row_value in enumerate(weights):
        assert row_value is not None
        row = dict(row_value)
        _exact_fields(row, frozenset({"file", "bytes", "sha256"}), f"{name} row {index}")
        _require(row["file"] == expected_files[index], f"{name} file order differs")
        _exact_int(row["bytes"], f"{name} row {index} bytes", minimum=1)
        _require(_valid_sha256(row["sha256"]), f"{name} row {index} digest is invalid")
        normalized.append(row)
    return normalized


def _checkpoint_descriptor(value: object, cell_id: str) -> dict[str, object]:
    name = f"{cell_id} checkpoint"
    _require(isinstance(value, dict), f"{name} is not an object")
    checkpoint = dict(value)
    fields = frozenset(
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
    )
    _exact_fields(checkpoint, fields, name)
    _require(checkpoint["model"] == MODEL, f"{name} model differs")
    _require(checkpoint["revision"] == MODEL_REVISION, f"{name} revision differs")
    _require(
        _same_json_value(checkpoint["architecture"], EXPECTED_ARCHITECTURE),
        f"{name} shape differs",
    )
    weights = _weight_inventory(checkpoint["weights"], f"{name} weights")
    _require(
        _same_json_value(weights, list(EXPECTED_WEIGHTS)),
        f"{name} weight inventory differs",
    )
    _require(
        checkpoint["weights_sha256"] == sha256_json(weights) == EXPECTED_WEIGHTS_SHA256,
        f"{name} canonical weight rollup differs",
    )
    _require(
        checkpoint["pinned_weights_rollup"]
        == legacy_weights_rollup(weights)
        == EXPECTED_WEIGHTS_ROLLUP,
        f"{name} legacy weight rollup differs",
    )
    metadata = checkpoint["metadata_sha256"]
    _require(
        _same_json_value(metadata, EXPECTED_METADATA_SHA256),
        f"{name} metadata identity differs",
    )
    _require(
        checkpoint["weight_index_sha256"] == EXPECTED_WEIGHT_INDEX_SHA256,
        f"{name} weight index identity differs",
    )
    _require(
        checkpoint["model_config_sha256"] == EXPECTED_MODEL_CONFIG_SHA256,
        f"{name} canonical config identity differs",
    )
    return {**checkpoint, "weights": weights, "metadata_sha256": dict(metadata)}


def _direct_parent_constraint(value: object, cell_id: str) -> dict[str, object]:
    name = f"{cell_id} checkpoint constraint"
    _require(isinstance(value, dict), f"{name} is not an object")
    constraint = dict(value)
    fields = frozenset(
        {
            "binding_kind",
            "model",
            "revision",
            "architecture",
            "weights",
            "weights_sha256",
            "pinned_weights_rollup",
            "metadata_sha256",
            "model_config_sha256",
        }
    )
    _exact_fields(constraint, fields, name)
    _require(constraint["binding_kind"] == "direct-full", f"{name} kind differs")
    _require(constraint["model"] == MODEL, f"{name} model differs")
    _require(constraint["revision"] == MODEL_REVISION, f"{name} revision differs")
    _require(
        _same_json_value(constraint["architecture"], EXPECTED_ARCHITECTURE),
        f"{name} shape differs",
    )
    weights = _weight_inventory(constraint["weights"], f"{name} weights")
    _require(
        _same_json_value(weights, list(EXPECTED_WEIGHTS)),
        f"{name} weight inventory differs",
    )
    _require(
        constraint["weights_sha256"] == sha256_json(weights) == EXPECTED_WEIGHTS_SHA256,
        f"{name} canonical weight rollup differs",
    )
    _require(
        constraint["pinned_weights_rollup"]
        == legacy_weights_rollup(weights)
        == EXPECTED_WEIGHTS_ROLLUP,
        f"{name} legacy weight rollup differs",
    )
    _require(
        _same_json_value(constraint["metadata_sha256"], EXPECTED_METADATA_SHA256),
        f"{name} metadata identity differs",
    )
    _require(
        constraint["model_config_sha256"] == EXPECTED_MODEL_CONFIG_SHA256,
        f"{name} canonical config identity differs",
    )
    return {**constraint, "weights": weights}


def _parent_descriptor(value: object, cell_id: str) -> dict[str, object]:
    name = f"{cell_id} parent evidence"
    spec = CELL_SPECS[cell_id]
    _require(isinstance(value, dict), f"{name} is not an object")
    parent = dict(value)
    _exact_fields(
        parent,
        frozenset(
            {
                "gate",
                "schema_version",
                "manifest_sha256",
                "raw_sha256",
                "quality_manifest_sha256",
                "quality_raw_sha256",
                "parent_topology",
                "checkpoint_binding",
            }
        ),
        name,
    )
    _require(parent["gate"] == spec["parent_gate"], f"{name} gate differs")
    _require(
        parent["schema_version"] == spec["parent_schema_version"],
        f"{name} schema differs",
    )
    _require(_valid_sha256(parent["manifest_sha256"]), f"{name} manifest digest is invalid")
    _require(_valid_sha256(parent["raw_sha256"]), f"{name} raw digest is invalid")
    if cell_id == "f4b-tp4":
        _require(
            _valid_sha256(parent["quality_manifest_sha256"]),
            f"{name} quality manifest digest is invalid",
        )
        _require(
            _valid_sha256(parent["quality_raw_sha256"]),
            f"{name} quality raw digest is invalid",
        )
    else:
        _require(
            parent["quality_manifest_sha256"] is None and parent["quality_raw_sha256"] is None,
            f"{name} has unexpected quality evidence",
        )
    _require(
        _same_json_value(parent["parent_topology"], spec["parent_topology"]),
        f"{name} topology differs",
    )
    kind = spec["parent_binding_kind"]
    constraint_value = parent["checkpoint_binding"]
    if kind == "direct-partial":
        _require(isinstance(constraint_value, dict), f"{name} constraint is not an object")
        constraint = dict(constraint_value)
        _exact_fields(
            constraint,
            frozenset({"binding_kind", "model", "revision", "pinned_weights_rollup"}),
            f"{name} constraint",
        )
        _require(constraint["binding_kind"] == kind, f"{name} constraint kind differs")
        _require(constraint["model"] == MODEL, f"{name} constraint model differs")
        _require(constraint["revision"] == MODEL_REVISION, f"{name} constraint revision differs")
        _require(
            constraint["pinned_weights_rollup"] == EXPECTED_WEIGHTS_ROLLUP,
            f"{name} constraint weight identity differs",
        )
    elif kind == "aggregate-common-f2c":
        _require(
            constraint_value == {"binding_kind": kind},
            f"{name} aggregate constraint differs",
        )
        constraint = dict(constraint_value)
    else:
        constraint = _direct_parent_constraint(constraint_value, cell_id)
    return {
        **parent,
        "parent_topology": dict(parent["parent_topology"]),
        "checkpoint_binding": constraint,
    }


def _engine_descriptor(value: object, cell_id: str) -> dict[str, object]:
    name = f"{cell_id} engine"
    spec = CELL_SPECS[cell_id]
    _require(isinstance(value, dict), f"{name} is not an object")
    engine = dict(value)
    _exact_fields(
        engine,
        frozenset(
            {
                "cell_id",
                "tensor_parallel_size",
                "page_size",
                "num_pages",
                "dram_profile_sha256",
                "dram_capacity_pages",
                "dram_min_restore_tokens",
                "decode_mode",
                "max_num_batched_tokens",
                "max_model_len",
            }
        ),
        name,
    )
    _require(engine["cell_id"] == cell_id, f"{name} cell identity differs")
    for field in (
        "tensor_parallel_size",
        "page_size",
        "num_pages",
        "dram_capacity_pages",
        "dram_min_restore_tokens",
        "decode_mode",
        "max_num_batched_tokens",
        "max_model_len",
    ):
        _require(
            _same_json_value(engine[field], spec[field]),
            f"{name} {field} differs",
        )
    _require(
        _same_json_value(engine["dram_profile_sha256"], spec["dram_profile_sha256"]),
        f"{name} profile differs",
    )
    if spec["dram_required"]:
        _require(_valid_sha256(engine["dram_profile_sha256"]), f"{name} profile is invalid")
    else:
        _require(engine["dram_profile_sha256"] is None, f"{name} has an unexpected profile")
    return engine


def _runtime_descriptor(value: object, cell_id: str) -> dict[str, object]:
    name = f"{cell_id} runtime"
    _require(isinstance(value, dict), f"{name} is not an object")
    runtime = dict(value)
    _exact_fields(runtime, frozenset({"run_nonce", "source", "checkpoint", "engine"}), name)
    _require(_valid_hex(runtime["run_nonce"], 32), f"{name} nonce is invalid")
    return {
        "run_nonce": runtime["run_nonce"],
        "source": _source_descriptor(runtime["source"], cell_id),
        "checkpoint": _checkpoint_descriptor(runtime["checkpoint"], cell_id),
        "engine": _engine_descriptor(runtime["engine"], cell_id),
    }


def _usage(value: object, name: str) -> dict[str, int]:
    _require(isinstance(value, dict), f"{name} is not an object")
    usage = dict(value)
    _exact_fields(
        usage,
        frozenset({"prompt_tokens", "completion_tokens", "cached_tokens"}),
        name,
    )
    return {
        field: _exact_int(usage[field], f"{name} {field}")
        for field in ("prompt_tokens", "completion_tokens", "cached_tokens")
    }


def _response(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} is not an object")
    response = dict(value)
    _exact_fields(
        response,
        frozenset({"status", "token_ids", "token_pieces", "text", "finish_reason", "usage"}),
        name,
    )
    _require(response["status"] in {"success", "error"}, f"{name} status is invalid")
    token_ids = response["token_ids"]
    _require(
        isinstance(token_ids, list)
        and all(type(token) is int and 0 <= token < TOKENIZER_VOCAB_SIZE for token in token_ids),
        f"{name} token ids are invalid",
    )
    token_pieces = response["token_pieces"]
    _require(
        isinstance(token_pieces, list)
        and all(isinstance(piece, str) and bool(piece) for piece in token_pieces),
        f"{name} token pieces are invalid",
    )
    _require(isinstance(response["text"], str), f"{name} text is invalid")
    _require(
        response["finish_reason"] is None or isinstance(response["finish_reason"], str),
        f"{name} finish reason is invalid",
    )
    return {
        **response,
        "token_ids": list(token_ids),
        "token_pieces": list(token_pieces),
        "usage": _usage(response["usage"], f"{name} usage"),
    }


def _request(
    value: Mapping[str, object],
    *,
    run_id: str,
    cell_id: str,
    feature: str,
    runtime_nonce: str,
    phase: str,
    ordinal: int,
) -> dict[str, object]:
    name = f"{cell_id} {phase} request {ordinal}"
    row = dict(value)
    _exact_fields(row, _REQUEST_FIELDS, name)
    _require(row["row_type"] == "request", f"{name} row type differs")
    _require(row["run_id"] == run_id, f"{name} run id differs")
    _require(row["cell_id"] == cell_id, f"{name} cell differs")
    _require(row["feature_id"] == feature, f"{name} feature differs")
    _require(row["phase"] == phase, f"{name} phase differs")
    _require(row["runtime_nonce"] == runtime_nonce, f"{name} runtime nonce differs")
    _require(
        row["error_type"] is None or isinstance(row["error_type"], str),
        f"{name} error type is invalid",
    )
    _exact_int(row["retry_count"], f"{name} retry count")
    return {
        **row,
        "prompt": _prompt_descriptor(row["prompt"], f"{name} prompt"),
        "response": _response(row["response"], f"{name} response"),
    }


def _tier_stats(value: object, name: str) -> dict[str, int]:
    _require(isinstance(value, dict), f"{name} is not an object")
    stats = dict(value)
    _exact_fields(stats, frozenset(TIER_STATS_FIELDS), name)
    return {field: _exact_int(stats[field], f"{name} {field}") for field in TIER_STATS_FIELDS}


def _feature_proof(value: object, cell_id: str) -> dict[str, object]:
    name = f"{cell_id} feature proof"
    proof_kind = CELL_SPECS[cell_id]["proof_kind"]
    _require(isinstance(value, dict), f"{name} is not an object")
    proof = dict(value)
    if proof_kind == "radix_reuse":
        _exact_fields(
            proof,
            frozenset({"kind", "same_runtime", "cache_hit_tokens"}),
            name,
        )
        _require(proof["kind"] == proof_kind, f"{name} kind differs")
        _require(type(proof["same_runtime"]) is bool, f"{name} same-runtime flag is invalid")
        _exact_int(proof["cache_hit_tokens"], f"{name} cache-hit tokens")
        return proof
    _exact_fields(
        proof,
        frozenset(
            {
                "kind",
                "before",
                "pre_warm",
                "after",
                "pressure_delta",
                "warm_delta",
                "target_cached_tokens_pre_warm",
            }
        ),
        name,
    )
    _require(proof["kind"] == proof_kind, f"{name} kind differs")
    return {
        "kind": proof_kind,
        "before": _tier_stats(proof["before"], f"{name} before"),
        "pre_warm": _tier_stats(proof["pre_warm"], f"{name} pre-warm"),
        "after": _tier_stats(proof["after"], f"{name} after"),
        "pressure_delta": _tier_stats(proof["pressure_delta"], f"{name} pressure delta"),
        "warm_delta": _tier_stats(proof["warm_delta"], f"{name} warm delta"),
        "target_cached_tokens_pre_warm": _exact_int(
            proof["target_cached_tokens_pre_warm"],
            f"{name} target cached tokens before warm",
        ),
    }


def _successful_fixed_response(request: Mapping[str, object]) -> bool:
    response = request["response"]
    prompt = request["prompt"]
    usage = response["usage"]
    return (
        response["status"] == "success"
        and request["error_type"] is None
        and request["retry_count"] == 0
        and len(response["token_ids"]) == OUTPUT_TOKENS
        and len(response["token_pieces"]) == OUTPUT_TOKENS
        and response["finish_reason"] == "length"
        and usage["prompt_tokens"] == prompt["token_count"]
        and usage["completion_tokens"] == OUTPUT_TOKENS
        and usage["cached_tokens"] <= usage["prompt_tokens"]
    )


def _answer_value(request: Mapping[str, object]) -> dict[str, object]:
    response = request["response"]
    return {
        "token_ids": response["token_ids"],
        "token_pieces": response["token_pieces"],
        "text": response["text"],
    }


def _runtime_matches_parent(
    runtime_checkpoint: Mapping[str, object],
    parent: Mapping[str, object],
) -> bool:
    constraint = parent["checkpoint_binding"]
    kind = constraint["binding_kind"]
    if kind == "aggregate-common-f2c":
        return True
    if kind == "direct-partial":
        return (
            runtime_checkpoint["model"] == constraint["model"]
            and runtime_checkpoint["revision"] == constraint["revision"]
            and runtime_checkpoint["pinned_weights_rollup"] == constraint["pinned_weights_rollup"]
        )
    return all(
        runtime_checkpoint[field] == constraint[field]
        for field in (
            "model",
            "revision",
            "architecture",
            "weights",
            "weights_sha256",
            "pinned_weights_rollup",
            "model_config_sha256",
        )
    ) and all(
        runtime_checkpoint["metadata_sha256"][name] == digest
        for name, digest in constraint["metadata_sha256"].items()
    )


def _cell_checks(cell: Mapping[str, object]) -> dict[str, bool]:
    cell_id = str(cell["cell_id"])
    transition = str(cell["transition"])
    cold = cell["cold"]
    warm = cell["warm"]
    pressure = cell["pressure"]
    proof = cell["feature_proof"]
    cold_usage = cold["response"]["usage"]
    warm_usage = warm["response"]["usage"]
    checks = {
        f"{cell_id}_runtime_checkpoint_matches_parent_constraint": _runtime_matches_parent(
            cell["runtime"]["checkpoint"], cell["parent_evidence"]
        ),
        f"{cell_id}_all_requests_succeeded_once_at_exact_32_tokens": all(
            _successful_fixed_response(request) for request in (cold, *pressure, warm)
        ),
        f"{cell_id}_cold_and_warm_prompt_identity_exact": (
            cell["prompt"] == cold["prompt"] == warm["prompt"]
        ),
        f"{cell_id}_cold_and_warm_native_answer_exact": (
            _answer_value(cold) == _answer_value(warm)
        ),
        f"{cell_id}_cold_miss_and_full_warm_hit_usage_exact": (
            cold_usage["cached_tokens"] == 0
            and warm_usage["cached_tokens"] == cell["prompt"]["token_count"]
        ),
    }
    if transition == "radix":
        checks[f"{cell_id}_radix_hit_proof_exact"] = (
            not pressure
            and proof["same_runtime"] is True
            and proof["cache_hit_tokens"] == warm_usage["cached_tokens"]
            and proof["cache_hit_tokens"] == cell["prompt"]["token_count"]
        )
    else:
        before = proof["before"]
        pre_warm = proof["pre_warm"]
        after = proof["after"]
        pressure_delta = proof["pressure_delta"]
        warm_delta = proof["warm_delta"]
        pressure_delta_exact = all(
            pre_warm[field] - before[field] == pressure_delta[field] for field in TIER_STATS_FIELDS
        )
        warm_delta_exact = all(
            after[field] - pre_warm[field] == warm_delta[field] for field in TIER_STATS_FIELDS
        )
        failure_counters_zero = all(
            stats[field] == 0
            for stats in (before, pre_warm, after, pressure_delta, warm_delta)
            for field in ("restore_fallbacks", "ownership_failures")
        )
        pressure_digests = [request["prompt"]["token_ids_sha256"] for request in pressure]
        checks[f"{cell_id}_pressure_prompts_are_distinct_uncached_inputs"] = (
            1 <= len(pressure) <= MAX_PRESSURE_REQUESTS
            and len(set(pressure_digests)) == len(pressure_digests)
            and cell["prompt"]["token_ids_sha256"] not in pressure_digests
            and all(
                request["prompt"]["token_count"] == cell["prompt"]["token_count"]
                for request in pressure
            )
            and all(request["response"]["usage"]["cached_tokens"] == 0 for request in pressure)
        )
        checks[f"{cell_id}_dram_offload_eviction_and_warm_restore_proof_exact"] = (
            pressure_delta_exact
            and warm_delta_exact
            and pressure_delta["offload_pages"] > 0
            and pressure_delta["restore_pages"] == 0
            and pressure_delta["restore_attempts"] == 0
            and proof["target_cached_tokens_pre_warm"] == 0
            and warm_delta["restore_pages"] > 0
            and warm_delta["restore_pages"] * CELL_SPECS[cell_id]["page_size"]
            >= CELL_SPECS[cell_id]["dram_min_restore_tokens"]
            and warm_delta["restore_attempts"] > 0
            and failure_counters_zero
        )
    return checks


def _parse_single_cell(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_cell_id: str | None,
) -> dict[str, object]:
    _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)), "cell rows differ")
    _require(len(rows) >= 4, "cell row count is too small")
    _require(all(isinstance(row, Mapping) for row in rows), "every cell row must be an object")
    start = dict(rows[0])
    _exact_fields(start, _CELL_START_FIELDS, "cell_start")
    _require(start["row_type"] == "cell_start", "cell does not start with cell_start")
    cell_id = _nonempty_string(start["cell_id"], "cell id")
    _require(cell_id in CELL_SPECS, "cell id is outside the fixed topology")
    if expected_cell_id is not None:
        _require(cell_id == expected_cell_id, f"expected {expected_cell_id}, got {cell_id}")
    spec = CELL_SPECS[cell_id]
    feature = str(spec["feature_id"])
    run_id = _nonempty_string(start["run_id"], f"{cell_id} run id")
    _require(start["feature_id"] == feature, f"{cell_id} feature differs")
    _require(start["transition"] == spec["transition"], f"{cell_id} transition differs")
    _require(
        _same_json_value(start["sampling"], frozen_config()["sampling"]),
        f"{cell_id} sampling differs",
    )
    prompt = _prompt_descriptor(start["prompt"], f"{cell_id} cell prompt")
    _require(
        prompt["token_count"] == COMPARISON_PROMPT_TOKENS,
        f"{cell_id} comparison prompt must contain exactly {COMPARISON_PROMPT_TOKENS} tokens",
    )
    _require(
        prompt["token_count"] + OUTPUT_TOKENS <= spec["max_model_len"],
        f"{cell_id} prompt exceeds the frozen model length",
    )
    if spec["dram_required"]:
        _require(
            prompt["token_count"] >= max(MIN_DRAM_PROMPT_TOKENS, spec["dram_min_restore_tokens"]),
            f"{cell_id} DRAM prompt is below the eviction floor",
        )
    parent = _parent_descriptor(start["parent_evidence"], cell_id)
    runtime = _runtime_descriptor(start["runtime"], cell_id)

    cursor = 1
    cold = _request(
        rows[cursor],
        run_id=run_id,
        cell_id=cell_id,
        feature=feature,
        runtime_nonce=str(runtime["run_nonce"]),
        phase="cold",
        ordinal=0,
    )
    cursor += 1
    pressure: list[dict[str, object]] = []
    if spec["transition"] == "dram_restore":
        while (
            cursor < len(rows) - 1
            and rows[cursor].get("row_type") == "request"
            and rows[cursor].get("phase") == "pressure"
        ):
            pressure.append(
                _request(
                    rows[cursor],
                    run_id=run_id,
                    cell_id=cell_id,
                    feature=feature,
                    runtime_nonce=str(runtime["run_nonce"]),
                    phase="pressure",
                    ordinal=len(pressure),
                )
            )
            cursor += 1
    _require(cursor < len(rows) - 1, f"{cell_id} warm request is missing")
    warm = _request(
        rows[cursor],
        run_id=run_id,
        cell_id=cell_id,
        feature=feature,
        runtime_nonce=str(runtime["run_nonce"]),
        phase="warm",
        ordinal=0,
    )
    cursor += 1
    _require(cursor == len(rows) - 1, f"{cell_id} has extra or reordered requests")
    end = dict(rows[cursor])
    _exact_fields(end, _CELL_END_FIELDS, f"{cell_id} cell_end")
    _require(end["row_type"] == "cell_end", f"{cell_id} does not end with cell_end")
    _require(end["run_id"] == run_id, f"{cell_id} cell_end run id differs")
    _require(end["cell_id"] == cell_id, f"{cell_id} cell_end identity differs")
    _require(end["feature_id"] == feature, f"{cell_id} cell_end feature differs")
    return {
        "cell_id": cell_id,
        "feature_id": feature,
        "transition": spec["transition"],
        "run_id": run_id,
        "prompt": prompt,
        "parent_evidence": parent,
        "runtime": runtime,
        "cold": cold,
        "pressure": pressure,
        "warm": warm,
        "feature_proof": _feature_proof(end["feature_proof"], cell_id),
    }


def validate_single_cell(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_cell_id: str | None = None,
) -> dict[str, object]:
    """Purely validate one child slice and derive the exact child verdict."""

    cell = _parse_single_cell(rows, expected_cell_id=expected_cell_id)
    checks = _cell_checks(cell)
    cold = cell["cold"]
    warm = cell["warm"]
    return {
        "cell_id": cell["cell_id"],
        "feature_id": cell["feature_id"],
        "transition": cell["transition"],
        "run_id": cell["run_id"],
        "prompt": cell["prompt"],
        "parent_evidence": cell["parent_evidence"],
        "runtime": cell["runtime"],
        "request_counts": {
            "cold": 1,
            "pressure": len(cell["pressure"]),
            "warm": 1,
        },
        "cold_cached_tokens": cold["response"]["usage"]["cached_tokens"],
        "warm_cached_tokens": warm["response"]["usage"]["cached_tokens"],
        "cold_answer_sha256": sha256_json(_answer_value(cold)),
        "warm_answer_sha256": sha256_json(_answer_value(warm)),
        "feature_proof": cell["feature_proof"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _cell_slice_end(rows: Sequence[Mapping[str, object]], start: int, cell_id: str) -> int:
    for index in range(start + 1, len(rows)):
        row = rows[index]
        if row.get("row_type") == "cell_end":
            return index
        _require(
            row.get("row_type") == "request",
            f"{cell_id} contains invalid framing before cell_end",
        )
    raise EvidenceError(f"{cell_id} cell_end is missing")


def recompute_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    """Derive the complete five-cell verdict from raw rows and their digest."""

    _require(_valid_sha256(raw_sha256), "raw SHA-256 is invalid")
    _require(
        all(isinstance(row, Mapping) for row in rows),
        "every raw row must be an object",
    )
    _require(len(rows) >= 22, "raw row count is too small for five complete cells")
    header = dict(rows[0])
    _exact_fields(header, _RUN_FIELDS, "run header")
    _require(header["row_type"] == "run", "raw evidence does not start with run")
    _require(header["schema_version"] == SCHEMA_VERSION, "run schema differs")
    _require(header["gate"] == GATE, "run gate differs")
    run_id = _nonempty_string(header["run_id"], "run id")
    _require(header["required_cells"] == list(CELL_IDS), "required cell order differs")
    _require(
        _same_json_value(header["config"], frozen_config()),
        "frozen configuration differs",
    )

    cursor = 1
    cells: list[dict[str, object]] = []
    for cell_id in CELL_IDS:
        _require(cursor < len(rows), f"{cell_id} is missing")
        end = _cell_slice_end(rows, cursor, cell_id)
        cell = validate_single_cell(rows[cursor : end + 1], expected_cell_id=cell_id)
        _require(cell["run_id"] == run_id, f"{cell_id} aggregate run id differs")
        cells.append(cell)
        cursor = end + 1
    _require(cursor == len(rows) - 1, "raw evidence has extra, missing, or reordered cells")
    end = dict(rows[cursor])
    _exact_fields(end, _RUN_END_FIELDS, "run_end")
    _require(end["row_type"] == "run_end", "raw evidence does not end with run_end")
    _require(end["run_id"] == run_id, "run_end run id differs")
    _require(end["status"] == "complete", "run_end status differs")
    _require(end["errors"] == [], "run_end records errors")
    _require(
        _exact_int(end["cell_count"], "run_end cell count") == len(CELL_IDS),
        "run_end cell count differs",
    )

    by_id = {str(cell["cell_id"]): cell for cell in cells}
    checkpoint_values = [cell["runtime"]["checkpoint"] for cell in cells]
    source_values = [cell["runtime"]["source"] for cell in cells]
    nonces = [cell["runtime"]["run_nonce"] for cell in cells]
    f4a_tp4 = by_id["f4a-tp4"]["parent_evidence"]
    f4a_tp8 = by_id["f4a-tp8"]["parent_evidence"]
    global_checks = {
        "five_cells_exact_order_and_production_topology": True,
        "all_runtime_checkpoint_identities_exact": len(
            {canonical_json(value) for value in checkpoint_values}
        )
        == 1,
        "all_cells_use_one_clean_source_identity": len(
            {canonical_json(value) for value in source_values}
        )
        == 1,
        "f2d_runtime_checkpoint_equals_f2c": (
            by_id["f2d-tp2"]["runtime"]["checkpoint"] == by_id["f2c-tp2"]["runtime"]["checkpoint"]
        ),
        "f4a_tp4_tp8_share_manifest_and_use_distinct_raw_shards": (
            f4a_tp4["manifest_sha256"] == f4a_tp8["manifest_sha256"]
            and f4a_tp4["raw_sha256"] != f4a_tp8["raw_sha256"]
            and f4a_tp4["checkpoint_binding"] == f4a_tp8["checkpoint_binding"]
        ),
        "every_cell_uses_a_unique_runtime_nonce": len(set(nonces)) == len(CELL_IDS),
    }
    checks = dict(global_checks)
    for cell in cells:
        checks.update(cell["checks"])
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "measurement_kind": MEASUREMENT_KIND,
        "run_id": run_id,
        "config": frozen_config(),
        "raw": {"name": RAW_NAME, "sha256": raw_sha256, "row_count": len(rows)},
        "parent_evidence": {str(cell["cell_id"]): cell["parent_evidence"] for cell in cells},
        "cells": cells,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _assert_gate_manifest(manifest: Mapping[str, object]) -> None:
    _require(isinstance(manifest, Mapping), "gate manifest is not an object")
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
    """Raise unless a recomputed manifest is internally consistent and passes."""

    _assert_gate_manifest(manifest)


def replay_raw(path: Path, *, assert_gate: bool = False) -> dict[str, object]:
    raw_path = Path(path)
    raw_bytes, rows = _read_jsonl_snapshot(raw_path)
    manifest = recompute_manifest(rows, raw_sha256=sha256_bytes(raw_bytes))
    if assert_gate:
        _assert_gate_manifest(manifest)
    return manifest


def write_artifact(
    raw_path: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    raw_target = Path(raw_path)
    manifest_target = Path(manifest_path)
    _require(raw_target.resolve() != manifest_target.resolve(), "raw and manifest paths collide")
    _require(
        not raw_target.exists() and not manifest_target.exists(),
        "refusing to overwrite retained B7 artifact",
    )
    raw_bytes = _jsonl_bytes(rows)
    manifest = recompute_manifest(rows, raw_sha256=sha256_bytes(raw_bytes))
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    try:
        with raw_target.open("xb") as handle:
            handle.write(raw_bytes)
        with manifest_target.open("xb") as handle:
            handle.write(manifest_bytes)
    except FileExistsError as error:
        raise EvidenceError("refusing to overwrite retained B7 artifact") from error
    except OSError as error:
        raise EvidenceError("cannot create retained B7 artifact") from error
    try:
        _require(
            raw_target.read_bytes() == raw_bytes and manifest_target.read_bytes() == manifest_bytes,
            "retained B7 artifact changed while writing",
        )
    except OSError as error:
        raise EvidenceError("cannot re-read retained B7 artifact") from error
    if assert_gate:
        _assert_gate_manifest(manifest)
    return manifest


def _resolve_artifact_paths(
    raw_path: Path,
    manifest_path: Path | None,
) -> tuple[Path, Path]:
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
    raw_target, manifest_target = _resolve_artifact_paths(raw_path, manifest_path)
    recomputed = replay_raw(raw_target, assert_gate=False)
    try:
        manifest_text = manifest_target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(f"cannot read retained manifest: {manifest_target}") from error
    _require(
        manifest_text.endswith("\n") and not manifest_text.endswith("\n\n"),
        "retained manifest framing differs",
    )
    retained = _loads_json(manifest_text[:-1], description="retained manifest")
    _require(isinstance(retained, dict), "retained manifest is not an object")
    _require(manifest_text == canonical_json(retained) + "\n", "retained manifest is not canonical")
    _require(
        _same_json_value(retained, recomputed),
        "retained manifest differs from independent raw replay",
    )
    if assert_gate:
        _assert_gate_manifest(recomputed)
    return recomputed


__all__ = [
    "CELL_IDS",
    "CELL_SPECS",
    "COMPARISON_PROMPT_TOKENS",
    "EXPECTED_ARCHITECTURE",
    "EXPECTED_METADATA_SHA256",
    "EXPECTED_MODEL_CONFIG_SHA256",
    "EXPECTED_WEIGHTS",
    "EXPECTED_WEIGHTS_ROLLUP",
    "EXPECTED_WEIGHTS_SHA256",
    "EXPECTED_WEIGHT_INDEX_SHA256",
    "EvidenceError",
    "FEATURES",
    "FEATURE_SPECS",
    "GATE",
    "MANIFEST_NAME",
    "MAX_PRESSURE_REQUESTS",
    "MEASUREMENT_KIND",
    "MODEL",
    "MODEL_REVISION",
    "MIN_DRAM_PROMPT_TOKENS",
    "OUTPUT_TOKENS",
    "PAGE_SIZE",
    "RAW_NAME",
    "REQUIRED_SOURCE_PATHS",
    "SCHEMA_VERSION",
    "TIER_STATS_FIELDS",
    "TOKENIZER_VOCAB_SIZE",
    "assert_gate",
    "canonical_json",
    "frozen_config",
    "legacy_weights_rollup",
    "read_jsonl",
    "recompute_manifest",
    "replay_raw",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "validate_single_cell",
    "verify_artifact",
    "write_artifact",
    "write_jsonl",
]
