#!/usr/bin/env python3
"""Replayable G4 M-A1 Qwen3-235B NVFP4 correctness gate.

The real engines are deliberately outside this file.  ``run`` consumes one
completed engine-capture JSONL stream and turns it into the retained raw plus a
manifest recomputed solely from that raw stream.  This keeps normal GitHub CI
on deterministic replay/tamper tests instead of pretending that a hosted
runner can execute the 134 GB checkpoint.

The capture adapter contract is strict:

* one ``run`` row, the exact 64 ``bench.parity_tp._TEXT_PROMPTS`` rows, four
  non-overlapping ``session`` rows in reference-EP2/Kairyu-EP2/reference-EP4/
  Kairyu-EP4 order, Kairyu rank-ownership rows, all free-running outputs, all
  teacher-forced positions, and one final ``run_end`` row;
* sixteen greedy tokens per prompt, seed 20260702, temperature 0, top-p 1,
  top-k -1, EOS ignored, and token-ID top-64 logprobs;
* reference-w2 and Kairyu-EP2 score the exact reference-w2 free-running prefix;
  reference-w4 and Kairyu-EP4 independently score the exact reference-w4
  prefix.  Free-running equality and cross-world equality are diagnostic only;
* reference rows come from the pinned TensorRT-LLM Python ``LLM`` API image.
  Kairyu rows come directly from ``SampledToken``.  HTTP/text top-k adapters are
  not valid evidence;
* every session retains start/end checkpoint, runtime, environment, kernel and
  physical-rank identity.  Kairyu additionally retains the exact contiguous
  expert partition and proves zero remote tensor reads or unresolved meta
  tensors.

Unknown schemas/types/fields, JSON duplicate keys, NaN/Infinity, missing or
duplicate evidence, metadata drift, fallback kernels, and invalid ownership
all fail closed.  A manifest verdict is never trusted: ``verify`` compares it
with an independent raw replay, while ``replay`` ignores it entirely.

Usage::

    python bench/g4_ma1_qwen3_235b_nvfp4_bench.py run \
      --capture /results/engine-capture.jsonl \
      --output-dir /results/g4-ma1 --assert-gate
    python bench/g4_ma1_qwen3_235b_nvfp4_bench.py verify \
      --artifact /results/g4-ma1 --assert-gate
    python bench/g4_ma1_qwen3_235b_nvfp4_bench.py replay \
      --artifact /results/g4-ma1 --assert-gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.parity_tp import _TEXT_PROMPTS  # noqa: E402
from kairyu.bench.reporting import atomic_write_json, atomic_write_text  # noqa: E402

SCHEMA_VERSION = "kairyu.g4.ma1.qwen3-235b-nvfp4.v1"
GATE = "G4-M-A1"
RAW_NAME = "g4-ma1-qwen3-235b-nvfp4-raw.jsonl"
MANIFEST_NAME = "g4-ma1-qwen3-235b-nvfp4-manifest.json"

MODEL = "nvidia/Qwen3-235B-A22B-NVFP4"
MODEL_REVISION = "21cfa2c9e152032eb60647ee7b46a2bbcd8d76d2"
EXPECTED_CONFIG_SHA256 = (
    "2d002795e3d79a92e03114cceeccf08cf9f4ea02ac5cd594ca9bea1faeb2c0b8"
)
EXPECTED_INDEX_SHA256 = (
    "593af4d5bae0b67f7edcd16195a27131c82cf5ae1f775b3ee68872be06bccfac"
)
EXPECTED_HF_QUANT_CONFIG_SHA256 = (
    "6423e9dd81773597a783e1109ffeb8d8b6a30a85c0eb4e503524a4fdf5f7f98b"
)
EXPECTED_TRT_RUNTIME_HF_QUANT_CONFIG_SHA256 = (
    "fdd091619a68ad06022f829d8e634684650da41c62126464622984babc26352e"
)
EXPECTED_INDEX_TOTAL_SIZE = 134_101_652_144
EXPECTED_TENSOR_COUNT = 146_549
EXPECTED_WEIGHT_FILES: tuple[tuple[str, int, str], ...] = (
    (
        "model-00001-of-00027.safetensors",
        4_999_619_880,
        "2749205e6aab3ed45e20b3c12fc8d0538655b1c407c9a96395607c09b4bd806e",
    ),
    (
        "model-00002-of-00027.safetensors",
        4_999_554_200,
        "8777a44b02ce7b95f21a1012e341763924ca7e38252b3ce6bd5c493b5d821f68",
    ),
    (
        "model-00003-of-00027.safetensors",
        5_000_457_620,
        "ff38c8485509442baf2c0e0e6216574f32d253a9dffd1a69bfee4ccc33c7d7dc",
    ),
    (
        "model-00004-of-00027.safetensors",
        4_999_952_884,
        "5b7ef0335b4be1cf5fa7c07223c7fcaf1cfab2c842d2cfd9714cfb085d97a73a",
    ),
    (
        "model-00005-of-00027.safetensors",
        5_000_463_428,
        "7f6703f077fa2211d0b9517f3c77c36add27b1faf9ca32d236a61cf2b99f80f6",
    ),
    (
        "model-00006-of-00027.safetensors",
        4_999_952_932,
        "4ce5dbd0c6f5e4ef609e1ea7d0af2dd62a61874b922af2800bf6ea5ff6064811",
    ),
    (
        "model-00007-of-00027.safetensors",
        4_999_559_752,
        "11ca3ebd307b2ba7eddaf4cc38dac2692b557af60b19445f615fde1ff4d0a3e6",
    ),
    (
        "model-00008-of-00027.safetensors",
        5_000_463_148,
        "02e4d44f0d75230ff0e842b9243e778d046d9e175cd70e169db85bd0d4cf27b3",
    ),
    (
        "model-00009-of-00027.safetensors",
        4_999_953_212,
        "3ec87a5e875152d8c5f86661e49c88dbfd57aa3638d035d8d4c7f4b906f05dab",
    ),
    (
        "model-00010-of-00027.safetensors",
        5_000_463_196,
        "b263380ff3013d9a8850b47e1d017bb9c67eb24a0ae8b1e8a6cde6613ef6c700",
    ),
    (
        "model-00011-of-00027.safetensors",
        4_999_953_172,
        "5a90cc578b09f1c1cc6f08d0bdeea6ef86c683109a7a805ba0e017b525683d4b",
    ),
    (
        "model-00012-of-00027.safetensors",
        5_000_463_412,
        "d54a80aebea95b0a1ced1c6776b608f4b68df26498cd30624b9f5e04b13ac725",
    ),
    (
        "model-00013-of-00027.safetensors",
        4_999_952_940,
        "cb2b08700594c0f9906c2400d56b3f61373248ac2c83ac04e8c47138ebbaf8bb",
    ),
    (
        "model-00014-of-00027.safetensors",
        4_999_559_760,
        "46fb33a57f93ed222e9b6baf002d32b863d9c6b2c5fc57def9fc6cd1fb7089d9",
    ),
    (
        "model-00015-of-00027.safetensors",
        5_000_463_132,
        "3005ec4b5864ddf39fa74e37f1883854730af5820df9f4136bd347964930f98e",
    ),
    (
        "model-00016-of-00027.safetensors",
        4_999_953_212,
        "dd6c03c02283b3b29d27f8703a00d5c707484d2b8a480ba4ee99c6d902247e51",
    ),
    (
        "model-00017-of-00027.safetensors",
        5_000_463_188,
        "d5168ae13255fa5f2c9af147304dd8759e12fcca4d0270cf011b06c5a2d99644",
    ),
    (
        "model-00018-of-00027.safetensors",
        4_999_953_172,
        "974d3120b0c702ec3cbede30bbc79d2c8169ee7dadcbb8c872cb6b0a69eee652",
    ),
    (
        "model-00019-of-00027.safetensors",
        5_000_463_412,
        "3818e38e033a384964f85e59352f9254f79615b8f88c98ae9d516d2097059297",
    ),
    (
        "model-00020-of-00027.safetensors",
        4_999_952_956,
        "005a583b0d8ebc069d1b8013ffaecfbae0144b404312cd02b370528b72df05a4",
    ),
    (
        "model-00021-of-00027.safetensors",
        4_999_559_760,
        "e57e4e8e84f9f351344799d2a2b8eb9a2c278aa9473c8613586635092568fe76",
    ),
    (
        "model-00022-of-00027.safetensors",
        5_000_463_132,
        "0407d2873efc8b2936788c41c75e9da53c300b8fa6cd16599d0e74d672661afe",
    ),
    (
        "model-00023-of-00027.safetensors",
        4_999_953_220,
        "4a87113d6f9605b2e1ce43db2a6c53978c7b43fe7267df22d116c5079045f67d",
    ),
    (
        "model-00024-of-00027.safetensors",
        5_000_463_172,
        "c4932031bcf1455658a06b954436d1c2dbf88f6041efedff5242e99d0a6bfca4",
    ),
    (
        "model-00025-of-00027.safetensors",
        4_999_953_180,
        "55ae45e3cbbc97ef31ce2758538a8e1c717e2d73a6fb8373ce219c413bd2252a",
    ),
    (
        "model-00026-of-00027.safetensors",
        5_000_463_404,
        "e3e86a69dbf6202799d3d5b1a35eea4de3ef892d65302abea73d39ceedb1d87e",
    ),
    (
        "model-00027-of-00027.safetensors",
        4_116_499_340,
        "dbcccd9f01f34c2840b73f0cce9c4f786b43c62f53d56793722b30821979df4d",
    ),
)
EXPECTED_TOKENIZER_FILES: tuple[tuple[str, int, str], ...] = (
    (
        "added_tokens.json",
        707,
        "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
    ),
    (
        "merges.txt",
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    (
        "special_tokens_map.json",
        388,
        "a4c29a7f63146cb43eef20b92caab0e2a5000d13d4dc9cd29888bbfad8059466",
    ),
    (
        "tokenizer.json",
        11_422_654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    (
        "tokenizer_config.json",
        9_709,
        "9a3a6f623ddaad552d27957966270df1c7c6b8777ddaebeb4753abd4ed635edd",
    ),
    (
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)
EXPECTED_WEIGHT_NAMES = tuple(row[0] for row in EXPECTED_WEIGHT_FILES)

PROMPTS = tuple(_TEXT_PROMPTS)
PROMPT_COUNT = 64
POSITIONS = 16
EXPECTED_POSITIONS = PROMPT_COUNT * POSITIONS
PROMPTS_SHA256 = (
    "a3acedd9a419e440b4a2ddb2bcd96e0a5f6151fcce48c8982a1608c36938633c"
)
SEED = 20_260_702
TOP_LOGPROBS = 64
WORLD_SIZES = (2, 4)
AGREEMENT_RATE_MIN = 0.99
AGREEMENT_COUNT_MIN = math.ceil(AGREEMENT_RATE_MIN * EXPECTED_POSITIONS)
TIE_GAP_NATS = 0.125
LOGPROB_ABS_DELTA_MAX = 0.25
VOCAB_SIZE = 151_936
NUM_LAYERS = 94
NUM_EXPERTS = 128
KV_CACHE_CAPACITY_TOKENS = 4_096
KAIRYU_KV_PAGE_SIZE = 16
KAIRYU_KV_NUM_PAGES = KV_CACHE_CAPACITY_TOKENS // KAIRYU_KV_PAGE_SIZE
# TensorRT-LLM uses the smaller of this ceiling and ``max_tokens`` and rejects
# 1.0 during standard full-attention engine construction.  The measured EP2
# reference leaves about 28 GiB free after weights, so 5% still exceeds the
# roughly 376 MiB TP2 BF16 capacity for 4,096 tokens while preserving ample
# allocator/workspace headroom.  EP4 has more headroom.
REFERENCE_KV_FREE_GPU_MEMORY_FRACTION = 0.05

EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
EXPECTED_SM = 120
MIN_GPU_MEMORY_BYTES = 90 * 1024**3

# Keep the external reference identity in one replacement-safe constant.  The
# image is immutable and the implementation/API choices are all binding.
REFERENCE_RUNTIME: Mapping[str, object] = {
    "name": "tensorrt_llm",
    "version": "1.2.1",
    "image_repo_digest": (
        "nvcr.io/nvidia/tensorrt-llm/release@sha256:"
        "cb4d8af81c586a90235ae3739b6d4ddc5d8336f2174c8a1c6b573d2e13faf5d7"
    ),
    "image_config_id": (
        "sha256:cb4d8af81c586a90235ae3739b6d4ddc5d8336f2174c8a1c6b573d2e13faf5d7"
    ),
    "backend": "pytorch",
    "result_api": (
        "tensorrt_llm.LLM.RequestOutput.token_ids+logprobs"
    ),
}
KAIRYU_RESULT_API = "kairyu.engine.core.sampling_types.SampledToken"

ARMS = (
    "reference_ep2",
    "kairyu_ep2",
    "reference_ep4",
    "kairyu_ep4",
)
REFERENCE_ARM = {2: "reference_ep2", 4: "reference_ep4"}
KAIRYU_ARM = {2: "kairyu_ep2", 4: "kairyu_ep4"}

BOUND_SOURCE_FILES = (
    "bench/g4_ma1_qwen3_235b_nvfp4_bench.py",
    "bench/g4_ma1_qwen3_235b_nvfp4_capture.py",
    "bench/parity_tp.py",
    "kairyu/engine/core/quant_config.py",
    "kairyu/engine/core/weights.py",
    "kairyu/engine/core/worker.py",
    "kairyu/kernels/quant_gemm_gpu.py",
    "kairyu/models/loader.py",
    "kairyu/models/moe_parallel.py",
    "kairyu/models/parallel.py",
    "pyproject.toml",
    "uv.lock",
)

EXPECTED_ARCHITECTURE: Mapping[str, object] = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 4096,
    "intermediate_size": 12_288,
    "moe_intermediate_size": 1536,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_experts": NUM_EXPERTS,
    "num_experts_per_tok": 8,
    "vocab_size": VOCAB_SIZE,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
}
EXPECTED_QUANTIZATION: Mapping[str, object] = {
    "source": "hf_quant_config.json",
    "producer_name": "modelopt",
    "producer_version": "0.33.0",
    "quant_algo": "NVFP4",
    "group_size": 16,
    "checkpoint_kv_cache_quant_algo": "FP8",
    "runtime_kv_cache_dtype": "bfloat16",
    "excluded_module_count": 95,
    "auxiliary_kv_scale_tensor_count": 188,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPO_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


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


if len(PROMPTS) != PROMPT_COUNT or sha256_json(list(PROMPTS)) != PROMPTS_SHA256:
    raise RuntimeError(
        "bench.parity_tp._TEXT_PROMPTS changed; amend the formal M-A1 contract "
        "instead of silently changing the fixed prompt population"
    )


def _arm_stack(arm: str) -> str:
    return "reference" if arm.startswith("reference_") else "kairyu"


def _arm_world_size(arm: str) -> int:
    return int(arm[-1])


def _result_api(arm: str) -> str:
    if _arm_stack(arm) == "reference":
        return str(REFERENCE_RUNTIME["result_api"])
    return KAIRYU_RESULT_API


def _session_config(arm: str) -> dict[str, object]:
    world_size = _arm_world_size(arm)
    reference = _arm_stack(arm) == "reference"
    return {
        "stack": "tensorrt_llm" if reference else "kairyu",
        "world_size": world_size,
        "tensor_parallel_size": world_size if reference else 1,
        "expert_parallel_size": world_size,
        "attention_mode": "tensor_parallel" if reference else "replicated",
        "attention_data_parallel": False,
        "overlap": False,
        "cuda_graph": False,
        "radix_prefix_cache": False,
        "block_reuse": False,
        "compute_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "kv_cache_capacity_tokens": KV_CACHE_CAPACITY_TOKENS,
        "kv_cache_policy": (
            {
                "kind": "max_tokens",
                "max_tokens": KV_CACHE_CAPACITY_TOKENS,
                "free_gpu_memory_fraction_ceiling": (
                    REFERENCE_KV_FREE_GPU_MEMORY_FRACTION
                ),
            }
            if reference
            else {
                "kind": "fixed_pages",
                "page_size": KAIRYU_KV_PAGE_SIZE,
                "num_pages": KAIRYU_KV_NUM_PAGES,
            }
        ),
        "moe_backend": "CUTLASS" if reference else "kairyu_ep_all_to_all",
        "sampling_arguments": {
            "seed": SEED,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0 if reference else -1,
            "top_k_normalized": "disabled",
            "ignore_eos": True,
            "max_tokens": POSITIONS,
            "logprobs": TOP_LOGPROBS,
        },
    }


def formal_config() -> dict[str, object]:
    return {
        "gate": GATE,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "prompts": {
            "source": "bench.parity_tp._TEXT_PROMPTS",
            "count": PROMPT_COUNT,
            "sha256": PROMPTS_SHA256,
        },
        "sampling": {
            "seed": SEED,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": {
                "normalized": "disabled",
                "kairyu_argument": -1,
                "tensorrt_llm_argument": 0,
            },
            "ignore_eos": True,
            "output_tokens": POSITIONS,
            "top_logprobs": TOP_LOGPROBS,
            "identity": "token-id",
        },
        "session_order": list(ARMS),
        "sessions": {arm: _session_config(arm) for arm in ARMS},
        "teacher_prefix": {
            "ep2": "reference_ep2_free_run",
            "ep4": "reference_ep4_free_run",
        },
        "binding": {
            "positions_per_cell": EXPECTED_POSITIONS,
            "reference_canonical_reproduction": EXPECTED_POSITIONS,
            "agreement_rate_min": AGREEMENT_RATE_MIN,
            "agreement_count_min": AGREEMENT_COUNT_MIN,
            "tie_gap_nats": TIE_GAP_NATS,
            "selected_logprob_abs_delta_max": LOGPROB_ABS_DELTA_MAX,
            "reciprocal_cross_selected_abs_delta_max": LOGPROB_ABS_DELTA_MAX,
        },
        "reference": dict(REFERENCE_RUNTIME),
        "free_running": "diagnostic-only",
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def strict_json_loads(value: str) -> object:
    return json.loads(
        value,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )


def _utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _repo_digest_valid(value: object) -> bool:
    return isinstance(value, str) and _REPO_DIGEST.fullmatch(value) is not None


def _source_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"commit", "tracked_dirty", "files"}
        and isinstance(value["commit"], str)
        and _COMMIT.fullmatch(value["commit"]) is not None
        and value["tracked_dirty"] is False
        and isinstance(value["files"], dict)
        and set(value["files"]) == set(BOUND_SOURCE_FILES)
        and all(
            isinstance(digest, str) and _HEX64.fullmatch(digest) is not None
            for digest in value["files"].values()
        )
    )


def _file_rows_valid(
    value: object,
    *,
    expected_names: set[str] | None = None,
    required_names: set[str] | None = None,
) -> bool:
    if not isinstance(value, list) or not value:
        return False
    names: list[str] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"name", "size", "sha256"}:
            return False
        if (
            not isinstance(row["name"], str)
            or not row["name"]
            or Path(row["name"]).name != row["name"]
            or type(row["size"]) is not int
            or row["size"] <= 0
            or not isinstance(row["sha256"], str)
            or _HEX64.fullmatch(row["sha256"]) is None
        ):
            return False
        names.append(row["name"])
    if len(names) != len(set(names)) or names != sorted(names):
        return False
    if expected_names is not None and set(names) != expected_names:
        return False
    if required_names is not None and not required_names <= set(names):
        return False
    return True


def _checkpoint_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "model",
        "revision",
        "config_sha256",
        "index_sha256",
        "hf_quant_config_sha256",
        "index_total_size",
        "tensor_count",
        "weight_shards",
        "weights_rollup_sha256",
        "tokenizer_files",
        "tokenizer_rollup_sha256",
        "architecture",
        "quantization",
    }:
        return False
    weights = value["weight_shards"]
    tokenizer = value["tokenizer_files"]
    expected_weights = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in EXPECTED_WEIGHT_FILES
    ]
    expected_tokenizer = [
        {"name": name, "size": size, "sha256": digest}
        for name, size, digest in EXPECTED_TOKENIZER_FILES
    ]
    return (
        value["model"] == MODEL
        and value["revision"] == MODEL_REVISION
        and value["config_sha256"] == EXPECTED_CONFIG_SHA256
        and value["index_sha256"] == EXPECTED_INDEX_SHA256
        and value["hf_quant_config_sha256"] == EXPECTED_HF_QUANT_CONFIG_SHA256
        and value["index_total_size"] == EXPECTED_INDEX_TOTAL_SIZE
        and value["tensor_count"] == EXPECTED_TENSOR_COUNT
        and _file_rows_valid(weights, expected_names=set(EXPECTED_WEIGHT_NAMES))
        and weights == expected_weights
        and value["weights_rollup_sha256"] == sha256_json(weights)
        and _file_rows_valid(
            tokenizer,
            expected_names={row[0] for row in EXPECTED_TOKENIZER_FILES},
        )
        and tokenizer == expected_tokenizer
        and value["tokenizer_rollup_sha256"] == sha256_json(tokenizer)
        and value["architecture"] == EXPECTED_ARCHITECTURE
        and value["quantization"] == EXPECTED_QUANTIZATION
    )


def _gpu_valid(value: object, *, rank: int) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "rank",
            "local_device_index",
            "uuid",
            "pci_bus_id",
            "name",
            "sm",
            "total_memory_bytes",
            "numa_node",
        }
        and value["rank"] == rank
        and value["local_device_index"] == rank
        and isinstance(value["uuid"], str)
        and value["uuid"].startswith("GPU-")
        and isinstance(value["pci_bus_id"], str)
        and bool(value["pci_bus_id"])
        and value["name"] == EXPECTED_GPU_NAME
        and value["sm"] == EXPECTED_SM
        and type(value["total_memory_bytes"]) is int
        and value["total_memory_bytes"] > MIN_GPU_MEMORY_BYTES
        and type(value["numa_node"]) is int
        and value["numa_node"] >= 0
    )


def _environment_valid(value: object, *, arm: str) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "packages",
        "cuda",
        "visibility",
        "gpus",
        "topology",
    }:
        return False
    packages = value["packages"]
    cuda = value["cuda"]
    visibility = value["visibility"]
    gpus = value["gpus"]
    topology = value["topology"]
    required_packages = (
        {"python", "torch", "tensorrt_llm"}
        if _arm_stack(arm) == "reference"
        else {"python", "torch", "kairyu", "flashinfer"}
    )
    return (
        isinstance(packages, dict)
        and required_packages <= set(packages)
        and all(
            isinstance(name, str)
            and bool(name)
            and isinstance(version, str)
            and bool(version)
            for name, version in packages.items()
        )
        and (
            _arm_stack(arm) != "reference"
            or packages.get("tensorrt_llm") == REFERENCE_RUNTIME["version"]
        )
        and isinstance(cuda, dict)
        and set(cuda) == {"runtime", "driver", "nccl"}
        and all(isinstance(item, str) and bool(item) for item in cuda.values())
        and isinstance(visibility, dict)
        and set(visibility)
        == {
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "resolved_rank_uuid_order",
        }
        and isinstance(visibility["CUDA_VISIBLE_DEVICES"], str)
        and bool(visibility["CUDA_VISIBLE_DEVICES"])
        and isinstance(visibility["NVIDIA_VISIBLE_DEVICES"], str)
        and bool(visibility["NVIDIA_VISIBLE_DEVICES"])
        and visibility["resolved_rank_uuid_order"]
        == [gpu["uuid"] for gpu in gpus]
        and isinstance(gpus, list)
        and len(gpus) == _arm_world_size(arm)
        and all(_gpu_valid(gpu, rank=rank) for rank, gpu in enumerate(gpus))
        and len({gpu["uuid"] for gpu in gpus}) == len(gpus)
        and len({gpu["pci_bus_id"] for gpu in gpus}) == len(gpus)
        and isinstance(topology, dict)
        and set(topology)
        == {
            "gpu_uuid_order",
            "gpu_pci_bus_id_order",
            "gpu_numa_node_order",
            "nvidia_smi_topology_raw",
            "nvidia_smi_topology_sha256",
        }
        and topology["gpu_uuid_order"] == [gpu["uuid"] for gpu in gpus]
        and topology["gpu_pci_bus_id_order"]
        == [gpu["pci_bus_id"] for gpu in gpus]
        and topology["gpu_numa_node_order"]
        == [gpu["numa_node"] for gpu in gpus]
        and isinstance(topology["nvidia_smi_topology_raw"], str)
        and bool(topology["nvidia_smi_topology_raw"])
        and topology["nvidia_smi_topology_sha256"]
        == sha256_bytes(topology["nvidia_smi_topology_raw"].encode())
    )


def _runtime_valid(
    value: object,
    *,
    arm: str,
    run_image_repo_digest: object,
    run_image_config_id: object,
    source_commit: object,
) -> bool:
    if not isinstance(value, dict):
        return False
    observation = value.get("container_observation")
    if not _container_observation_valid(observation):
        return False
    if _arm_stack(arm) == "reference":
        return (
            set(value) == {*REFERENCE_RUNTIME, "container_observation"}
            and all(value.get(key) == expected for key, expected in REFERENCE_RUNTIME.items())
            and observation["image_repo_digest"]
            == REFERENCE_RUNTIME["image_repo_digest"]
            and observation["image_config_id"]
            == REFERENCE_RUNTIME["image_config_id"]
        )
    if set(value) != {
        "name",
        "version",
        "image_repo_digest",
        "image_config_id",
        "backend",
        "result_api",
        "source_commit",
        "container_observation",
    }:
        return False
    image_repo_digest = value["image_repo_digest"]
    image_config_id = value["image_config_id"]
    return (
        value["name"] == "kairyu"
        and isinstance(value["version"], str)
        and bool(value["version"])
        and _repo_digest_valid(image_repo_digest)
        and image_repo_digest == run_image_repo_digest
        and isinstance(image_config_id, str)
        and _IMAGE_ID.fullmatch(image_config_id) is not None
        and image_config_id == run_image_config_id
        and observation["image_repo_digest"] == image_repo_digest
        and observation["image_config_id"] == image_config_id
        and observation["source_revision"] == source_commit
        and value["backend"] == "pytorch"
        and value["result_api"] == KAIRYU_RESULT_API
        and value["source_commit"] == source_commit
    )


def _container_observation_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "source",
            "container_id",
            "image_repo_digest",
            "image_config_id",
            "running",
            "started_at",
            "inspect_sha256",
            "source_revision",
        }
        and value["source"] == "docker-container-inspect"
        and isinstance(value["container_id"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["container_id"]) is not None
        and _repo_digest_valid(value["image_repo_digest"])
        and isinstance(value["image_config_id"], str)
        and _IMAGE_ID.fullmatch(value["image_config_id"]) is not None
        and value["running"] is True
        and _utc_timestamp(value["started_at"]) is not None
        and isinstance(value["inspect_sha256"], str)
        and _HEX64.fullmatch(value["inspect_sha256"]) is not None
        and (
            value["source_revision"] is None
            or (
                isinstance(value["source_revision"], str)
                and _COMMIT.fullmatch(value["source_revision"]) is not None
            )
        )
    )


def _kernel_rank_valid(value: object, *, arm: str, rank: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "rank",
        "sm",
        "requested",
        "resolved",
        "provider",
        "provider_version",
        "checkpoint_layout",
        "weight_dtype",
        "scale_dtype",
        "global_scale_dtype",
        "activation_dtype",
        "compute_dtype",
        "kv_cache_dtype",
        "quantized_projection_count",
        "native_nvfp4_projection_count",
        "fallback_projection_count",
        "projection_inventory_sha256",
        "fallback",
        "execution_evidence",
    }:
        return False
    reference = _arm_stack(arm) == "reference"
    experts_per_rank = NUM_EXPERTS // _arm_world_size(arm)
    expected_projection_count = 4 * NUM_LAYERS + (
        3 * NUM_LAYERS * experts_per_rank
    )
    evidence = value["execution_evidence"]
    if reference:
        evidence_valid = evidence == {
            "kind": "singleton-backend-successful-forward",
            "projection_inventory_source": "checkpoint-index",
            "nvfp4_allowed_backends": ["cutlass"],
            "moe_backend": "CUTLASS",
            "successful_forward": True,
        }
    else:
        owned_count = NUM_EXPERTS // _arm_world_size(arm)
        first_owned = rank * owned_count
        last_owned = first_owned + owned_count - 1
        expected_load_info = {
            "ep_rank": rank,
            "ep_size": _arm_world_size(arm),
            "owned_expert_count": owned_count,
            "owned_expert_indices_sha256": sha256_json(
                list(range(first_owned, last_owned + 1))
            ),
            "first_owned_expert": first_owned,
            "last_owned_expert": last_owned,
            "quantization_source": "hf_quant_config.json",
            "quantization_method": "nvfp4",
            "kv_cache_quant_algo": "FP8",
            "producer_name": "modelopt",
            "producer_version": "0.33.0",
            "checkpoint_tensor_count": EXPECTED_TENSOR_COUNT,
            "rank_loaded_tensor_count": (
                1977 + NUM_LAYERS * owned_count * 12
            ),
            "auxiliary_kv_scale_count": 188,
        }
        observed = evidence.get("observed") if isinstance(evidence, dict) else None
        evidence_valid = (
            isinstance(observed, dict)
            and observed
            == {
                "rank": rank,
                "projection_count": expected_projection_count,
                "projection_inventory_sha256": value[
                    "projection_inventory_sha256"
                ],
                "kernel_counts": {
                    "nvfp4:flashinfer_nvfp4": expected_projection_count
                },
                "meta_count": {
                    "parameters": 0,
                    "buffers": 0,
                    "total": 0,
                },
                "load_info": expected_load_info,
            }
            and evidence
            == {
                "kind": "all-rank-runtime-module-probe",
                "projection_inventory_source": "rank-local-model.named_modules",
                "probe": "DistEPModelRunner.ep_kernel_inventory_metadata",
                "successful_forward": True,
                "observed": observed,
            }
        )
    return (
        value["rank"] == rank
        and value["sm"] == EXPECTED_SM
        and value["requested"] == "nvfp4"
        and value["resolved"]
        == (
            "tensorrt_llm.pytorch.MoeConfig.CUTLASS"
            if reference
            else "flashinfer.mm_fp4"
        )
        and value["provider"] == ("cutlass" if reference else "flashinfer")
        and isinstance(value["provider_version"], str)
        and bool(value["provider_version"])
        and value["checkpoint_layout"] == "modelopt-nvfp4-group16"
        and value["weight_dtype"] == "uint8-packed-e2m1"
        and value["scale_dtype"] == "float8_e4m3fn"
        and value["global_scale_dtype"] == "float32"
        and value["activation_dtype"] == "nvfp4-e2m1"
        and value["compute_dtype"] == "bfloat16"
        and value["kv_cache_dtype"] == "bfloat16"
        and value["quantized_projection_count"] == expected_projection_count
        and value["native_nvfp4_projection_count"]
        == value["quantized_projection_count"]
        and value["fallback_projection_count"] == 0
        and isinstance(value["projection_inventory_sha256"], str)
        and _HEX64.fullmatch(value["projection_inventory_sha256"])
        is not None
        and value["fallback"] is None
        and evidence_valid
    )


def _run_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "type",
            "created_at",
            "config",
            "image_repo_digest",
            "image_config_id",
            "source_start",
            "checkpoint_start",
        }
        and value["schema_version"] == SCHEMA_VERSION
        and value["type"] == "run"
        and _utc_timestamp(value["created_at"]) is not None
        and value["config"] == formal_config()
        and _repo_digest_valid(value["image_repo_digest"])
        and isinstance(value["image_config_id"], str)
        and _IMAGE_ID.fullmatch(value["image_config_id"]) is not None
        and _source_valid(value["source_start"])
        and _checkpoint_valid(value["checkpoint_start"])
    )


def _run_end_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "type",
            "source_end",
            "checkpoint_end",
            "row_counts_before_end",
        }
        and value["schema_version"] == SCHEMA_VERSION
        and value["type"] == "run_end"
        and _source_valid(value["source_end"])
        and _checkpoint_valid(value["checkpoint_end"])
        and isinstance(value["row_counts_before_end"], dict)
        and all(
            isinstance(kind, str)
            and bool(kind)
            and type(count) is int
            and count >= 0
            for kind, count in value["row_counts_before_end"].items()
        )
    )


def _attention_backend_valid(value: object, *, arm: str) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "requested",
        "resolved",
        "components",
        "identity",
    }:
        return False
    reference = _arm_stack(arm) == "reference"
    expected = "tensorrt_llm" if reference else "flashinfer"
    components = value["components"]
    return (
        value["requested"] == ("tensorrt_llm" if reference else "auto")
        and value["resolved"] == expected
        and isinstance(components, dict)
        and set(components) == {"prefill", "decode", "kv_mode"}
        and components["prefill"] == expected
        and components["decode"] == expected
        and components["kv_mode"] == "paged-direct"
        and isinstance(value["identity"], str)
        and bool(value["identity"])
    )


def _checkpoint_view_valid(
    value: object,
    *,
    arm: str,
    checkpoint: object,
) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    if _arm_stack(arm) == "kairyu":
        return (
            isinstance(value, dict)
            and set(value) == {"kind", "model_path", "hf_quant_config_sha256"}
            and value["kind"] == "original"
            and isinstance(value["model_path"], str)
            and Path(value["model_path"]).is_absolute()
            and value["hf_quant_config_sha256"]
            == checkpoint.get("hf_quant_config_sha256")
        )
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "source_model_path",
        "runtime_model_path",
        "source_hf_quant_config_sha256",
        "runtime_hf_quant_config_sha256",
        "removed_json_path",
        "removed_value",
        "kv_cache_config_dtype_argument",
        "effective_kv_cache_dtype",
        "weight_links",
    }:
        return False
    links = value["weight_links"]
    if not isinstance(links, list) or len(links) != len(EXPECTED_WEIGHT_NAMES):
        return False
    expected_sizes = {
        row["name"]: row["size"]
        for row in checkpoint.get("weight_shards", [])
        if isinstance(row, dict) and set(row) >= {"name", "size"}
    }
    names: list[str] = []
    for link in links:
        if not isinstance(link, dict) or set(link) != {
            "name",
            "symlink_target",
            "source_device",
            "source_inode",
            "runtime_device",
            "runtime_inode",
            "size_bytes",
        }:
            return False
        name = link["name"]
        if (
            not isinstance(name, str)
            or name not in expected_sizes
            or not isinstance(link["symlink_target"], str)
            or not link["symlink_target"]
            or type(link["source_device"]) is not int
            or type(link["source_inode"]) is not int
            or type(link["runtime_device"]) is not int
            or type(link["runtime_inode"]) is not int
            or link["source_device"] != link["runtime_device"]
            or link["source_inode"] != link["runtime_inode"]
            or link["size_bytes"] != expected_sizes[name]
        ):
            return False
        names.append(name)
    source_path = value["source_model_path"]
    runtime_path = value["runtime_model_path"]
    return (
        names == list(EXPECTED_WEIGHT_NAMES)
        and isinstance(source_path, str)
        and isinstance(runtime_path, str)
        and Path(source_path).is_absolute()
        and Path(runtime_path).is_absolute()
        and source_path != runtime_path
        and value["kind"] == "bf16-kv-metadata-overlay"
        and value["source_hf_quant_config_sha256"]
        == checkpoint.get("hf_quant_config_sha256")
        and value["runtime_hf_quant_config_sha256"]
        == EXPECTED_TRT_RUNTIME_HF_QUANT_CONFIG_SHA256
        and value["removed_json_path"] == "quantization.kv_cache_quant_algo"
        and value["removed_value"] == "FP8"
        and value["kv_cache_config_dtype_argument"] == "auto"
        and value["effective_kv_cache_dtype"] == "bfloat16"
    )


def _sampling_ownership_valid(value: object, *, arm: str) -> bool:
    if _arm_stack(arm) == "reference":
        return value == []
    world_size = _arm_world_size(arm)
    return (
        isinstance(value, list)
        and len(value) == world_size
        and all(
            isinstance(row, dict)
            and row
            == {
                "rank": rank,
                "control_world_size": world_size,
                "control_backend": "gloo",
                "model_world_size": world_size,
                "model_backend": "nccl",
                "sampling_owner": rank == 0,
                "sampler_present": rank == 0,
                "device": f"cuda:{rank}",
            }
            for rank, row in enumerate(value)
        )
    )


def _session_valid(value: object, *, arm: str, run: Mapping[str, object]) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "arm",
        "sequence_index",
        "started_at",
        "finished_at",
        "config",
        "runtime",
        "environment_start",
        "environment_end",
        "checkpoint_start",
        "checkpoint_end",
        "kernel_ranks",
        "sampling_ownership",
        "attention_backend",
        "checkpoint_view",
    }:
        return False
    started = _utc_timestamp(value["started_at"])
    finished = _utc_timestamp(value["finished_at"])
    kernels = value["kernel_ranks"]
    source = run.get("source_start")
    source_commit = source.get("commit") if isinstance(source, dict) else None
    return (
        value["schema_version"] == SCHEMA_VERSION
        and value["type"] == "session"
        and value["arm"] == arm
        and value["sequence_index"] == ARMS.index(arm)
        and started is not None
        and finished is not None
        and started < finished
        and value["config"] == _session_config(arm)
        and _runtime_valid(
            value["runtime"],
            arm=arm,
            run_image_repo_digest=run.get("image_repo_digest"),
            run_image_config_id=run.get("image_config_id"),
            source_commit=source_commit,
        )
        and _environment_valid(value["environment_start"], arm=arm)
        and value["environment_start"] == value["environment_end"]
        and _checkpoint_valid(value["checkpoint_start"])
        and value["checkpoint_start"] == value["checkpoint_end"]
        and value["checkpoint_start"] == run.get("checkpoint_start")
        and isinstance(kernels, list)
        and len(kernels) == _arm_world_size(arm)
        and all(
            _kernel_rank_valid(kernel, arm=arm, rank=rank)
            for rank, kernel in enumerate(kernels)
        )
        and _sampling_ownership_valid(value["sampling_ownership"], arm=arm)
        and _attention_backend_valid(value["attention_backend"], arm=arm)
        and _checkpoint_view_valid(
            value["checkpoint_view"],
            arm=arm,
            checkpoint=value["checkpoint_start"],
        )
    )


def _prompt_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "prompt_id",
        "prompt_index",
        "text",
        "text_sha256",
        "token_ids",
        "token_ids_sha256",
        "tokenizer_rollup_sha256",
    }:
        return False
    index = value["prompt_index"]
    tokens = value["token_ids"]
    return (
        value["schema_version"] == SCHEMA_VERSION
        and value["type"] == "prompt"
        and type(index) is int
        and 0 <= index < PROMPT_COUNT
        and value["prompt_id"] == f"p{index}"
        and value["text"] == PROMPTS[index]
        and value["text_sha256"] == sha256_bytes(PROMPTS[index].encode())
        and isinstance(tokens, list)
        and bool(tokens)
        and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in tokens)
        and value["token_ids_sha256"] == sha256_json(tokens)
        and isinstance(value["tokenizer_rollup_sha256"], str)
        and _HEX64.fullmatch(value["tokenizer_rollup_sha256"]) is not None
    )


def _free_run_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "arm",
        "prompt_id",
        "output_token_ids",
        "output_token_ids_sha256",
        "selected_logprobs",
        "finish_reason",
        "source_api",
    }:
        return False
    arm = value["arm"]
    prompt_id = value["prompt_id"]
    tokens = value["output_token_ids"]
    logprobs = value["selected_logprobs"]
    return (
        value["schema_version"] == SCHEMA_VERSION
        and value["type"] == "free_run"
        and arm in ARMS
        and isinstance(prompt_id, str)
        and prompt_id in {f"p{index}" for index in range(PROMPT_COUNT)}
        and isinstance(tokens, list)
        and len(tokens) == POSITIONS
        and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in tokens)
        and value["output_token_ids_sha256"] == sha256_json(tokens)
        and isinstance(logprobs, list)
        and len(logprobs) == POSITIONS
        and all(_finite_float(logprob) for logprob in logprobs)
        and value["finish_reason"] == "length"
        and value["source_api"] == _result_api(arm)
    )


def _top_logprobs_valid(
    value: object,
    *,
    selected_token_id: object,
    selected_logprob: object,
) -> bool:
    if not isinstance(value, list) or len(value) != TOP_LOGPROBS:
        return False
    token_ids: list[int] = []
    logprobs: list[float] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"token_id", "logprob"}:
            return False
        if (
            type(row["token_id"]) is not int
            or not 0 <= row["token_id"] < VOCAB_SIZE
            or not _finite_float(row["logprob"])
        ):
            return False
        token_ids.append(row["token_id"])
        logprobs.append(row["logprob"])
    return (
        len(token_ids) == len(set(token_ids))
        and all(
            left >= right
            for left, right in zip(logprobs, logprobs[1:], strict=False)
        )
        and token_ids[0] == selected_token_id
        and logprobs[0] == selected_logprob
    )


def _teacher_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "arm",
        "prompt_id",
        "position",
        "prefix_token_count",
        "prefix_token_ids_sha256",
        "canonical_token_id",
        "selected_token_id",
        "selected_logprob",
        "top_logprobs",
        "source_api",
    }:
        return False
    arm = value["arm"]
    position = value["position"]
    return (
        value["schema_version"] == SCHEMA_VERSION
        and value["type"] == "teacher"
        and arm in ARMS
        and value["prompt_id"] in {f"p{index}" for index in range(PROMPT_COUNT)}
        and type(position) is int
        and 0 <= position < POSITIONS
        and type(value["prefix_token_count"]) is int
        and value["prefix_token_count"] > position
        and isinstance(value["prefix_token_ids_sha256"], str)
        and _HEX64.fullmatch(value["prefix_token_ids_sha256"]) is not None
        and type(value["canonical_token_id"]) is int
        and 0 <= value["canonical_token_id"] < VOCAB_SIZE
        and type(value["selected_token_id"]) is int
        and 0 <= value["selected_token_id"] < VOCAB_SIZE
        and _finite_float(value["selected_logprob"])
        and _top_logprobs_valid(
            value["top_logprobs"],
            selected_token_id=value["selected_token_id"],
            selected_logprob=value["selected_logprob"],
        )
        and value["source_api"] == _result_api(arm)
    )


def _ownership_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "type",
        "arm",
        "rank",
        "world_size",
        "layers",
        "num_experts",
        "owned_expert_indices",
        "owned_expert_tensor_count",
        "owned_expert_tensor_names_sha256",
        "rank_loaded_tensor_count",
        "rank_loaded_bytes",
        "loaded_tensor_names_sha256",
        "dense_replication_sha256",
        "router_replication_sha256",
        "shared_expert_replication_sha256",
        "remote_expert_tensor_reads",
        "unresolved_meta_tensors",
        "load_contract",
        "evidence_source",
    }:
        return False
    arm = value["arm"]
    if arm not in KAIRYU_ARM.values():
        return False
    world_size = _arm_world_size(arm)
    rank = value["rank"]
    if type(rank) is not int or not 0 <= rank < world_size:
        return False
    experts_per_rank = NUM_EXPERTS // world_size
    expected = list(
        range(rank * experts_per_rank, (rank + 1) * experts_per_rank)
    )
    expected_owned_tensor_count = NUM_LAYERS * experts_per_rank * 12
    expected_rank_loaded_tensor_count = 1977 + expected_owned_tensor_count
    load_contract = value["load_contract"]
    return (
        value["schema_version"] == SCHEMA_VERSION
        and value["type"] == "ownership"
        and value["world_size"] == world_size
        and value["layers"] == NUM_LAYERS
        and value["num_experts"] == NUM_EXPERTS
        and value["owned_expert_indices"] == expected
        and value["owned_expert_tensor_count"] == expected_owned_tensor_count
        and value["rank_loaded_tensor_count"]
        == expected_rank_loaded_tensor_count
        and type(value["rank_loaded_bytes"]) is int
        and value["rank_loaded_bytes"] > 0
        and all(
            isinstance(value[name], str) and _HEX64.fullmatch(value[name]) is not None
            for name in (
                "owned_expert_tensor_names_sha256",
                "loaded_tensor_names_sha256",
                "dense_replication_sha256",
                "router_replication_sha256",
                "shared_expert_replication_sha256",
            )
        )
        and value["remote_expert_tensor_reads"] == 0
        and value["unresolved_meta_tensors"] == 0
        and value["evidence_source"]
        == "checkpoint-index-and-build-contract"
        and isinstance(load_contract, dict)
        and load_contract
        == {
            "quantization_source": "hf_quant_config.json",
            "quantization_method": "nvfp4",
            "checkpoint_kv_cache_quant_algo": "FP8",
            "runtime_kv_cache_dtype": "bfloat16",
            "producer_name": "modelopt",
            "producer_version": "0.33.0",
            "checkpoint_tensor_count": EXPECTED_TENSOR_COUNT,
            "auxiliary_kv_scale_count": 188,
            "ownership": "contiguous-global-expert-blocks",
        }
    )


def _failure_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {"schema_version", "type", "arm", "error_type", "message"}
        and value["schema_version"] == SCHEMA_VERSION
        and value["type"] == "failure"
        and value["arm"] in ARMS
        and isinstance(value["error_type"], str)
        and bool(value["error_type"])
        and isinstance(value["message"], str)
        and bool(value["message"])
    )


def _row_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("type")) for row in rows).items()))


def _keyed_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    keys: Sequence[str],
) -> tuple[dict[tuple[object, ...], Mapping[str, object]], bool]:
    result: dict[tuple[object, ...], Mapping[str, object]] = {}
    duplicate = False
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        duplicate |= key in result
        result[key] = row
    return result, duplicate


def _top_map(row: Mapping[str, object]) -> dict[int, float]:
    value = row.get("top_logprobs")
    if not isinstance(value, list):
        return {}
    result: dict[int, float] = {}
    for item in value:
        if (
            isinstance(item, dict)
            and type(item.get("token_id")) is int
            and _finite_float(item.get("logprob"))
        ):
            result[item["token_id"]] = item["logprob"]
    return result


def _score_cell(
    reference: Mapping[tuple[object, ...], Mapping[str, object]],
    candidate: Mapping[tuple[object, ...], Mapping[str, object]],
) -> dict[str, object]:
    expected = {
        (f"p{prompt}", position)
        for prompt in range(PROMPT_COUNT)
        for position in range(POSITIONS)
    }
    complete = set(reference) == expected and set(candidate) == expected
    agreed = 0
    selected_deltas: list[float] = []
    reciprocal_deltas: list[float] = []
    missing_reciprocal: list[dict[str, object]] = []
    disagreements: list[dict[str, object]] = []
    substantive = 0
    if complete:
        for prompt_id, position in sorted(expected):
            ref = reference[(prompt_id, position)]
            cand = candidate[(prompt_id, position)]
            ref_token = ref["selected_token_id"]
            cand_token = cand["selected_token_id"]
            ref_top = _top_map(ref)
            cand_top = _top_map(cand)
            if ref_token == cand_token:
                agreed += 1
                selected_deltas.append(
                    abs(float(ref["selected_logprob"]) - float(cand["selected_logprob"]))
                )
                continue
            missing = ref_token not in cand_top or cand_token not in ref_top
            ref_gap: float | None = None
            cand_gap: float | None = None
            cross_deltas: list[float] = []
            if missing:
                missing_reciprocal.append(
                    {"prompt_id": prompt_id, "position": position}
                )
            else:
                ref_gap = float(ref["selected_logprob"]) - ref_top[cand_token]
                cand_gap = float(cand["selected_logprob"]) - cand_top[ref_token]
                cross_deltas = [
                    abs(ref_top[ref_token] - cand_top[ref_token]),
                    abs(ref_top[cand_token] - cand_top[cand_token]),
                ]
                reciprocal_deltas.extend(cross_deltas)
            is_substantive = (
                missing
                or ref_gap is None
                or ref_gap > TIE_GAP_NATS
            )
            substantive += int(is_substantive)
            disagreements.append(
                {
                    "prompt_id": prompt_id,
                    "position": position,
                    "reference_token_id": ref_token,
                    "candidate_token_id": cand_token,
                    "reference_tie_gap_nats": ref_gap,
                    "candidate_tie_gap_nats": cand_gap,
                    "reciprocal_cross_selected_abs_deltas": cross_deltas,
                    "substantive": is_substantive,
                }
            )
    return {
        "positions": EXPECTED_POSITIONS,
        "complete": complete,
        "agreed": agreed,
        "agreement_rate": agreed / EXPECTED_POSITIONS,
        "agreement_count_min": AGREEMENT_COUNT_MIN,
        "agreement_rate_min": AGREEMENT_RATE_MIN,
        "disagreements": disagreements,
        "substantive_disagreements": substantive,
        "missing_reciprocal_top64": missing_reciprocal,
        "max_selected_logprob_abs_delta": (
            max(selected_deltas) if selected_deltas else None
        ),
        "max_reciprocal_cross_selected_abs_delta": (
            max(reciprocal_deltas) if reciprocal_deltas else 0.0
        ),
        "tie_gap_nats": TIE_GAP_NATS,
        "logprob_abs_delta_max": LOGPROB_ABS_DELTA_MAX,
    }


def _free_run_diagnostics(
    free_by_key: Mapping[tuple[object, ...], Mapping[str, object]],
) -> list[dict[str, object]]:
    pairs = (
        ("reference_ep2", "kairyu_ep2", "world2"),
        ("reference_ep4", "kairyu_ep4", "world4"),
        ("reference_ep2", "reference_ep4", "reference_cross_world"),
        ("kairyu_ep2", "kairyu_ep4", "kairyu_cross_world"),
    )
    diagnostics: list[dict[str, object]] = []
    for left_arm, right_arm, name in pairs:
        sequences_equal = 0
        aligned_equal = 0
        common_prefix = 0
        common_prefix_logprob_deltas: list[float] = []
        for prompt in range(PROMPT_COUNT):
            left = free_by_key.get((left_arm, f"p{prompt}"), {})
            right = free_by_key.get((right_arm, f"p{prompt}"), {})
            left_tokens = left.get("output_token_ids")
            right_tokens = right.get("output_token_ids")
            left_lps = left.get("selected_logprobs")
            right_lps = right.get("selected_logprobs")
            if not all(isinstance(value, list) for value in (left_tokens, right_tokens)):
                continue
            sequences_equal += int(left_tokens == right_tokens)
            aligned_equal += sum(
                a == b for a, b in zip(left_tokens, right_tokens, strict=False)
            )
            prefix = 0
            for a, b in zip(left_tokens, right_tokens, strict=False):
                if a != b:
                    break
                prefix += 1
            common_prefix += prefix
            if isinstance(left_lps, list) and isinstance(right_lps, list):
                common_prefix_logprob_deltas.extend(
                    abs(float(a) - float(b))
                    for a, b in zip(
                        left_lps[:prefix], right_lps[:prefix], strict=False
                    )
                )
        diagnostics.append(
            {
                "name": name,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "prompts": PROMPT_COUNT,
                "exact_sequences": sequences_equal,
                "aligned_equal_positions": aligned_equal,
                "common_prefix_positions": common_prefix,
                "max_common_prefix_selected_logprob_abs_delta": (
                    max(common_prefix_logprob_deltas)
                    if common_prefix_logprob_deltas
                    else None
                ),
                "binding": False,
            }
        )
    return diagnostics


def recompute_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    if _HEX64.fullmatch(raw_sha256) is None:
        raise ValueError("raw_sha256 must be 64 lowercase hex")
    allowed_types = {
        "run",
        "prompt",
        "session",
        "ownership",
        "free_run",
        "teacher",
        "failure",
        "run_end",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"raw row {index} is not an object")
        try:
            canonical_json(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"raw row {index} is not finite canonical JSON"
            ) from exc
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"raw row {index} has the wrong schema")
        if row.get("type") not in allowed_types:
            raise ValueError(f"raw row {index} has unknown type")

    by_type: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["type"])].append(row)
    run = by_type["run"][0] if len(by_type["run"]) == 1 else {}
    run_end = by_type["run_end"][0] if len(by_type["run_end"]) == 1 else {}

    checks: dict[str, bool] = {}
    checks["one_run_and_one_run_end"] = (
        len(by_type["run"]) == 1
        and len(by_type["run_end"]) == 1
        and _run_valid(run)
        and _run_end_valid(run_end)
    )
    checks["run_boundaries_ordered"] = (
        bool(rows)
        and rows[0].get("type") == "run"
        and rows[-1].get("type") == "run_end"
    )
    checks["formal_config_exact"] = isinstance(run, dict) and run.get(
        "config"
    ) == formal_config()
    checks["clean_source_stable"] = (
        _source_valid(run.get("source_start"))
        and _source_valid(run_end.get("source_end"))
        and run.get("source_start") == run_end.get("source_end")
    )
    checks["full_checkpoint_identity_stable"] = (
        _checkpoint_valid(run.get("checkpoint_start"))
        and _checkpoint_valid(run_end.get("checkpoint_end"))
        and run.get("checkpoint_start") == run_end.get("checkpoint_end")
    )
    checks["no_runtime_failure"] = (
        len(by_type["failure"]) == 0
        and all(_failure_valid(row) for row in by_type["failure"])
    )
    checks["run_end_counts_exact"] = (
        isinstance(run_end, dict)
        and run_end.get("row_counts_before_end")
        == _row_counts([row for row in rows if row.get("type") != "run_end"])
    )

    prompt_by_key, duplicate_prompts = _keyed_rows(
        by_type["prompt"], keys=("prompt_id",)
    )
    checkpoint = run.get("checkpoint_start")
    tokenizer_rollup = (
        checkpoint.get("tokenizer_rollup_sha256")
        if isinstance(checkpoint, dict)
        else None
    )
    expected_prompt_keys = {(f"p{index}",) for index in range(PROMPT_COUNT)}
    checks["exact_fixed_tokenized_prompts"] = (
        not duplicate_prompts
        and len(by_type["prompt"]) == PROMPT_COUNT
        and set(prompt_by_key) == expected_prompt_keys
        and all(_prompt_valid(row) for row in by_type["prompt"])
        and all(
            row.get("tokenizer_rollup_sha256") == tokenizer_rollup
            for row in by_type["prompt"]
        )
    )

    session_by_key, duplicate_sessions = _keyed_rows(
        by_type["session"], keys=("arm",)
    )
    expected_session_keys = {(arm,) for arm in ARMS}
    complete_sessions = (
        not duplicate_sessions
        and len(by_type["session"]) == len(ARMS)
        and set(session_by_key) == expected_session_keys
        and all(
            _session_valid(session_by_key[(arm,)], arm=arm, run=run)
            for arm in ARMS
        )
    )
    checks["complete_session_provenance"] = complete_sessions
    sequential = complete_sessions
    if complete_sessions:
        ordered_sessions = [session_by_key[(arm,)] for arm in ARMS]
        sequential = all(
            _utc_timestamp(left["finished_at"])
            <= _utc_timestamp(right["started_at"])
            for left, right in zip(
                ordered_sessions, ordered_sessions[1:], strict=False
            )
        )
    checks["sessions_sequential_not_parallel"] = bool(sequential)
    checks["pinned_reference_runtime_and_python_api"] = complete_sessions and all(
        _runtime_valid(
            session_by_key[(REFERENCE_ARM[world_size],)].get("runtime"),
            arm=REFERENCE_ARM[world_size],
            run_image_repo_digest=run.get("image_repo_digest"),
            run_image_config_id=run.get("image_config_id"),
            source_commit=run.get("source_start", {}).get("commit"),
        )
        for world_size in WORLD_SIZES
    )
    checks["bf16_kv_all_sessions"] = complete_sessions and all(
        session_by_key[(arm,)].get("config", {}).get("kv_cache_dtype")
        == "bfloat16"
        for arm in ARMS
    )
    checks["native_nvfp4_kernels_no_fallback"] = complete_sessions and all(
        all(
            _kernel_rank_valid(kernel, arm=arm, rank=rank)
            for rank, kernel in enumerate(
                session_by_key[(arm,)].get("kernel_ranks", [])
            )
        )
        for arm in ARMS
    )

    topology_matched = complete_sessions
    if complete_sessions:
        physical: dict[str, list[tuple[object, object]]] = {}
        for arm in ARMS:
            gpus = session_by_key[(arm,)]["environment_start"]["gpus"]
            physical[arm] = [(gpu["uuid"], gpu["pci_bus_id"]) for gpu in gpus]
        topology_matched = (
            physical["reference_ep2"] == physical["kairyu_ep2"]
            and physical["reference_ep4"] == physical["kairyu_ep4"]
            and physical["reference_ep2"] == physical["reference_ep4"][:2]
        )
    checks["matched_physical_gpu_topology"] = bool(topology_matched)

    ownership_by_key, duplicate_ownership = _keyed_rows(
        by_type["ownership"], keys=("arm", "rank")
    )
    expected_ownership_keys = {
        (KAIRYU_ARM[world_size], rank)
        for world_size in WORLD_SIZES
        for rank in range(world_size)
    }
    ownership_complete = (
        not duplicate_ownership
        and len(by_type["ownership"]) == len(expected_ownership_keys)
        and set(ownership_by_key) == expected_ownership_keys
        and all(_ownership_valid(row) for row in by_type["ownership"])
    )
    checks["complete_contiguous_expert_ownership"] = ownership_complete
    no_remote_or_meta = ownership_complete and all(
        row["remote_expert_tensor_reads"] == 0
        and row["unresolved_meta_tensors"] == 0
        for row in by_type["ownership"]
    )
    checks["no_remote_tensor_reads_or_unresolved_meta"] = no_remote_or_meta
    replicated_dense = ownership_complete
    if ownership_complete:
        for world_size in WORLD_SIZES:
            rows_for_world = [
                ownership_by_key[(KAIRYU_ARM[world_size], rank)]
                for rank in range(world_size)
            ]
            for field in (
                "dense_replication_sha256",
                "router_replication_sha256",
                "shared_expert_replication_sha256",
            ):
                replicated_dense &= len({row[field] for row in rows_for_world}) == 1
            replicated_dense &= len(
                {row["rank_loaded_tensor_count"] for row in rows_for_world}
            ) == 1
            replicated_dense &= len(
                {row["rank_loaded_bytes"] for row in rows_for_world}
            ) == 1
        for field in (
            "dense_replication_sha256",
            "router_replication_sha256",
            "shared_expert_replication_sha256",
        ):
            replicated_dense &= len(
                {row[field] for row in by_type["ownership"]}
            ) == 1
    checks["rank_invariant_replicated_and_load_geometry"] = bool(replicated_dense)

    free_by_key, duplicate_free = _keyed_rows(
        by_type["free_run"], keys=("arm", "prompt_id")
    )
    expected_free_keys = {
        (arm, f"p{prompt}") for arm in ARMS for prompt in range(PROMPT_COUNT)
    }
    complete_free = (
        not duplicate_free
        and len(by_type["free_run"]) == len(expected_free_keys)
        and set(free_by_key) == expected_free_keys
        and all(_free_run_valid(row) for row in by_type["free_run"])
    )
    checks["complete_finite_free_running_diagnostics"] = complete_free

    teacher_by_key, duplicate_teacher = _keyed_rows(
        by_type["teacher"], keys=("arm", "prompt_id", "position")
    )
    expected_teacher_keys = {
        (arm, f"p{prompt}", position)
        for arm in ARMS
        for prompt in range(PROMPT_COUNT)
        for position in range(POSITIONS)
    }
    complete_teacher = (
        not duplicate_teacher
        and len(by_type["teacher"]) == len(expected_teacher_keys)
        and set(teacher_by_key) == expected_teacher_keys
        and all(_teacher_valid(row) for row in by_type["teacher"])
    )
    checks["complete_finite_token_id_top64_teacher_rows"] = complete_teacher

    prefixes_exact = complete_teacher and complete_free and checks[
        "exact_fixed_tokenized_prompts"
    ]
    reference_canonical_exact = prefixes_exact
    if prefixes_exact:
        for world_size in WORLD_SIZES:
            ref_arm = REFERENCE_ARM[world_size]
            for prompt in range(PROMPT_COUNT):
                prompt_id = f"p{prompt}"
                prompt_tokens = prompt_by_key[(prompt_id,)]["token_ids"]
                canonical = free_by_key[(ref_arm, prompt_id)]["output_token_ids"]
                for position in range(POSITIONS):
                    expected_prefix = prompt_tokens + canonical[:position]
                    expected_digest = sha256_json(expected_prefix)
                    expected_token = canonical[position]
                    for arm in (ref_arm, KAIRYU_ARM[world_size]):
                        row = teacher_by_key[(arm, prompt_id, position)]
                        prefixes_exact &= (
                            row["prefix_token_count"] == len(expected_prefix)
                            and row["prefix_token_ids_sha256"] == expected_digest
                            and row["canonical_token_id"] == expected_token
                        )
                    reference_canonical_exact &= (
                        teacher_by_key[(ref_arm, prompt_id, position)][
                            "selected_token_id"
                        ]
                        == expected_token
                    )
    checks["world_matched_teacher_prefixes_exact"] = bool(prefixes_exact)
    checks["reference_teacher_reproduces_all_canonical_tokens"] = bool(
        reference_canonical_exact
    )

    binding: dict[str, dict[str, object]] = {}
    for world_size in WORLD_SIZES:
        ref_arm = REFERENCE_ARM[world_size]
        candidate_arm = KAIRYU_ARM[world_size]
        reference_rows = (
            {
                (key[1], key[2]): row
                for key, row in teacher_by_key.items()
                if key[0] == ref_arm
            }
            if complete_teacher
            else {}
        )
        candidate_rows = (
            {
                (key[1], key[2]): row
                for key, row in teacher_by_key.items()
                if key[0] == candidate_arm
            }
            if complete_teacher
            else {}
        )
        score = _score_cell(reference_rows, candidate_rows)
        binding[f"ep{world_size}"] = score
        checks[f"ep{world_size}_teacher_agreement_at_least_99_percent"] = (
            complete_teacher
            and prefixes_exact
            and score["complete"] is True
            and score["agreed"] >= AGREEMENT_COUNT_MIN
        )
        checks[f"ep{world_size}_zero_substantive_disagreements"] = (
            complete_teacher
            and score["substantive_disagreements"] == 0
            and not score["missing_reciprocal_top64"]
        )
        selected_delta = score["max_selected_logprob_abs_delta"]
        checks[f"ep{world_size}_selected_logprob_delta_at_most_0_25"] = (
            complete_teacher
            and selected_delta is not None
            and selected_delta <= LOGPROB_ABS_DELTA_MAX
        )
        reciprocal_delta = score["max_reciprocal_cross_selected_abs_delta"]
        checks[f"ep{world_size}_reciprocal_cross_selected_delta_at_most_0_25"] = (
            complete_teacher
            and reciprocal_delta <= LOGPROB_ABS_DELTA_MAX
            and not score["missing_reciprocal_top64"]
        )

    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "config": formal_config(),
        "raw": {"name": RAW_NAME, "sha256": raw_sha256, "rows": len(rows)},
        "checks": checks,
        "metrics": {
            "binding_teacher_forced": binding,
            "diagnostic_free_running": (
                _free_run_diagnostics(free_by_key) if complete_free else []
            ),
            "sessions": [
                {
                    "arm": arm,
                    "started_at": session_by_key.get((arm,), {}).get("started_at"),
                    "finished_at": session_by_key.get((arm,), {}).get("finished_at"),
                    "runtime": session_by_key.get((arm,), {}).get("runtime"),
                    "config": session_by_key.get((arm,), {}).get("config"),
                }
                for arm in ARMS
            ],
        },
        "failures": by_type["failure"],
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
    }


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


def _read_raw(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            raise ValueError(f"{path}:{line_number}: blank JSONL row")
        value = strict_json_loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def write_artifact(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    atomic_write_text(
        raw_path,
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = recompute_manifest(rows, raw_sha256=sha256_file(raw_path))
    atomic_write_json(
        output_dir / MANIFEST_NAME,
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    return manifest


def run_capture(capture: Path, output_dir: Path) -> dict[str, object]:
    """Validate a completed engine capture and retain canonical raw evidence."""

    return write_artifact(output_dir, _read_raw(capture))


def verify_artifact(path: Path) -> dict[str, object]:
    raw_path, manifest_path = _artifact_paths(path)
    rows = _read_raw(raw_path)
    manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not an object")
    replayed = recompute_manifest(rows, raw_sha256=sha256_file(raw_path))
    if manifest != replayed:
        raise ValueError("manifest differs from independent raw replay")
    return replayed


def replay_artifact(path: Path) -> dict[str, object]:
    raw_path, _manifest_path = _artifact_paths(path)
    rows = _read_raw(raw_path)
    return recompute_manifest(rows, raw_sha256=sha256_file(raw_path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="ingest a completed strict engine-capture JSONL"
    )
    run_parser.add_argument("--capture", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--assert-gate", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="compare a manifest with independent raw replay"
    )
    verify_parser.add_argument("--artifact", type=Path, required=True)
    verify_parser.add_argument("--assert-gate", action="store_true")

    replay_parser = subparsers.add_parser(
        "replay", help="recompute the verdict from raw JSONL only"
    )
    replay_parser.add_argument("--artifact", type=Path, required=True)
    replay_parser.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        result = run_capture(args.capture.resolve(), args.output_dir.resolve())
    elif args.command == "verify":
        result = verify_artifact(args.artifact.resolve())
    else:
        result = replay_artifact(args.artifact.resolve())
    print(canonical_json(result))
    return 1 if args.assert_gate and result["passed"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
