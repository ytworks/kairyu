"""Formal G2 A8 DP-scaling, placement-latency, and affinity gate.

The live ``measure`` command drives one Qwen3-32B TP4 replica directly and the
same replica plus a second TP4 replica through the production L2
``ReplicaPool``.  It consumes the already pinned G2 A6 trace bundle rather than
downloading or regenerating prompts.  ``verify`` and ``replay`` never contact a
service or GPU: they hash the raw JSONL and independently recompute every
binding verdict.

CPU tests exercise only the fail-closed verifier.  They cannot produce a G2 A8
PASS; a passing artifact requires the exact real eight-GPU topology recorded by
``measure``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from kairyu.bench.reporting import nearest_rank_percentile

SCHEMA_VERSION = "kairyu.g2.a8.dp-scaling.v1"
GATE = "G2-A8"
MODEL = "qwen3-32b"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
RAW_NAME = "g2-a8-dp-scaling-raw.jsonl"
MANIFEST_NAME = "g2-a8-dp-scaling-manifest.json"

RATES_RPS = (4, 8, 16, 32, 64)
REPEATS = 3
SCALING_REQUESTS = 64
SCALING_OUTPUT_TOKENS = 32
WARMUP_REQUESTS = 8
AFFINITY_SESSIONS = 64
AFFINITY_TURNS = 8
AFFINITY_REQUESTS = AFFINITY_SESSIONS * AFFINITY_TURNS
AFFINITY_OUTPUT_TOKENS = 1
TTFT_SLO_NS = 10_000_000_000
GOODPUT_RATIO_FLOOR = 1.9
PLACEMENT_P99_CEILING_NS = 10_000_000
CACHE_RETENTION_FLOOR = 0.90
ARM_ORDERS = (("single", "dp"), ("dp", "single"), ("single", "dp"))
PERCENTILE_METHOD = "nearest-rank-v1"
TRACE_SCHEMA = "kairyu.g2.a6.trace-bundle.v2"
TRACE_BUNDLE_SHA256 = "93fafd75917980b220a418608635b478868b999861e45c33aff8026234578c6d"
SHAREGPT_DESCRIPTOR_SHA256 = (
    "dac8e527d50996f0477c3eaa9279eafb3eb930441e6e34f99c85d04d328dcf9c"
)
AFFINITY_DESCRIPTOR_SHA256 = (
    "a499b93a04c5d4c7bbd982bcb59db8cb5fbb69907ac2786c030b7eb3a8ccb0ed"
)
SCALING_SELECTION_SHA256 = (
    "2cd9408f293edca942d39f69e03eba10795c0876976d147c059e67e9bdef8247"
)
AFFINITY_SELECTION_SHA256 = (
    "6a9a364d2974b16d8196bbf5d9f56447a438f5dd4f0d7403c06f5d38ea782314"
)
CHECKPOINT_WEIGHTS_SHA256 = (
    "6aa37b7da4e37d45b277d0bca47b00c2bfb58bb60986ba93e4c3e667df123955"
)
CHECKPOINT_INDEX_SHA256 = (
    "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
)
CHECKPOINT_CONFIG_SHA256 = (
    "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
)
CHECKPOINT_EVIDENCE_SHA256 = (
    "989c3f8caee5423ff081a3f4856bd2797cf53b3ca836147e048c5faae57ba27b"
)
A7_ARTIFACT_DIR = Path(
    "bench/results/g2-a7-kv-hit-qwen3-32b-rtxpro6000-2026-07-29"
)
A7_RAW_SHA256 = "a8049a5211bc703e336cac5fb6b6b283078ac9e55688225021d18dc82e84de0c"
A7_MANIFEST_SHA256 = (
    "43f65cd5ea0fee8b0a205458a0e6f42c2c0ebb9f38fc100e9f761051bc4e9a27"
)
PINNED_TOKENIZER_SHA256 = (
    "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
)
# Each piece round-trips to exactly the paired one-token ID under the pinned
# Qwen3-32B tokenizer.  Distinct system-message pieces isolate every measured
# cache namespace without changing token count across arms/cells.
NAMESPACE_PIECES = (
    (264, " a"),
    (270, " th"),
    (272, " c"),
    (274, " s"),
    (279, " the"),
    (281, " p"),
    (282, " f"),
    (284, " ="),
    (289, " w"),
    (293, " b"),
    (294, " d"),
    (296, " m"),
    (297, " o"),
    (304, " in"),
    (305, " h"),
    (308, " n"),
    (311, " to"),
    (312, " re"),
    (314, " {"),
    (315, " of"),
    (320, " ("),
    (323, " and"),
    (326, " l"),
    (328, " S"),
    (330, ' "'),
    (335, " }"),
    (342, " g"),
    (348, " v"),
    (350, " T"),
    (353, " *"),
    (356, " C"),
    (357, " st"),
    (358, " I"),
    (362, " A"),
    (364, " '"),
    (366, " <"),
    (369, " for"),
    (374, " is"),
)
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SSE_PREFIX = "data: "
CONFIG_PATHS = (
    Path("bench/dp_scaling_g2_a8_bench.py"),
    Path("examples/qwen3-32b-multi-gpu/a8-compose.yaml"),
    Path("examples/qwen3-32b-multi-gpu/a8-replica.yaml"),
    Path("examples/qwen3-32b-multi-gpu/a8-gateway.yaml"),
    Path("examples/qwen3-32b-multi-gpu/a8-stack.sh"),
)
CHECKPOINT_HASH_SCRIPT = r"""
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

weight_paths = sorted(root.glob("model-*.safetensors"))
with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(weight_paths))) as pool:
    weight_hashes = list(pool.map(sha256, weight_paths))
weights = [
    {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}
    for path, digest in zip(weight_paths, weight_hashes, strict=True)
]
result = {
    "weights": weights,
    "weights_sha256": hashlib.sha256(
        json.dumps(weights, sort_keys=True).encode()
    ).hexdigest(),
    "index_sha256": sha256(root / "model.safetensors.index.json"),
    "tokenizer_sha256": sha256(root / "tokenizer.json"),
    "config_sha256": sha256(root / "config.json"),
}
print(json.dumps(result, sort_keys=True))
"""


class GateEvidenceError(ValueError):
    """Raised when retained evidence cannot prove the gate."""


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
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_number(value: object, name: str, *, minimum: float | None = None) -> float:
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


def formal_config() -> dict[str, object]:
    return {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "rates_rps": list(RATES_RPS),
        "repeats": REPEATS,
        "scaling_requests_per_cell": SCALING_REQUESTS,
        "scaling_output_tokens": SCALING_OUTPUT_TOKENS,
        "warmup_requests_per_arm_repeat": WARMUP_REQUESTS,
        "arm_orders": [list(order) for order in ARM_ORDERS],
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "retry_count": 0,
        "ttft_slo_ns": TTFT_SLO_NS,
        "affinity_order": ["single", "dp"],
        "affinity_sessions": AFFINITY_SESSIONS,
        "affinity_turns": AFFINITY_TURNS,
        "affinity_requests_per_arm": AFFINITY_REQUESTS,
        "affinity_output_tokens": AFFINITY_OUTPUT_TOKENS,
        "percentile_method": PERCENTILE_METHOD,
        "thresholds": {
            "median_peak_goodput_dp_over_single_min": GOODPUT_RATIO_FLOOR,
            "placement_latency_p99_ns_exclusive_max": PLACEMENT_P99_CEILING_NS,
            "affinity_cache_rate_dp_over_single_min": CACHE_RETENTION_FLOOR,
        },
    }


def _validate_formal_config(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "formal config must be an object")
    _require(value == formal_config(), "formal config differs from the fixed G2 A8 contract")
    return dict(value)


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


def _canonical_digest(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode())


def _descriptor_value(value: object, name: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} descriptor must be an object")
    payload = value.get("value")
    digest = value.get("sha256")
    _require(isinstance(payload, dict), f"{name} descriptor value must be an object")
    _require(_hash_valid(digest), f"{name} descriptor SHA-256 is invalid")
    _require(_canonical_digest(payload) == digest, f"{name} descriptor SHA-256 mismatch")
    return payload


def load_trace_bundle(path: Path) -> dict[str, object]:
    """Load and validate the retained A6 trace bundle used by both A8 phases."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        sha256_file(path) == TRACE_BUNDLE_SHA256,
        "trace bundle differs from the retained G2 A6 pin",
    )
    _require(isinstance(payload, dict), "trace bundle must be an object")
    _require(payload.get("schema_version") == TRACE_SCHEMA, "trace bundle schema mismatch")
    traces = payload.get("traces")
    _require(isinstance(traces, dict), "trace bundle traces missing")
    _require(
        isinstance(traces.get("sharegpt"), dict)
        and traces["sharegpt"].get("sha256") == SHAREGPT_DESCRIPTOR_SHA256,
        "ShareGPT descriptor differs from the G2 A6 pin",
    )
    _require(
        isinstance(traces.get("shared-prefix"), dict)
        and traces["shared-prefix"].get("sha256") == AFFINITY_DESCRIPTOR_SHA256,
        "affinity descriptor differs from the G2 A6 pin",
    )
    share = _descriptor_value(traces.get("sharegpt"), "ShareGPT")
    affinity = _descriptor_value(traces.get("shared-prefix"), "shared-prefix")
    share_rows = share.get("requests")
    affinity_rows = affinity.get("requests")
    _require(
        isinstance(share_rows, list) and len(share_rows) >= SCALING_REQUESTS,
        "trace bundle needs at least 64 ShareGPT requests",
    )
    _require(
        isinstance(affinity_rows, list) and len(affinity_rows) == AFFINITY_REQUESTS,
        "trace bundle needs the exact 512-request affinity trace",
    )
    _require(
        affinity.get("tokenizer_sha256") == PINNED_TOKENIZER_SHA256,
        "affinity trace tokenizer differs from the Qwen3-32B pin",
    )

    selected_share: list[dict[str, object]] = []
    for sequence, raw in enumerate(share_rows[:SCALING_REQUESTS]):
        _require(isinstance(raw, dict), "ShareGPT trace row must be an object")
        prompt = raw.get("prompt_text")
        prompt_hash = raw.get("prompt_text_sha256")
        _require(isinstance(prompt, str) and prompt, "ShareGPT prompt text missing")
        _require(
            _hash_valid(prompt_hash)
            and sha256_bytes(prompt.encode()) == prompt_hash,
            "ShareGPT prompt hash mismatch",
        )
        _require(raw.get("sequence") == sequence, "ShareGPT sequence is not contiguous")
        token_ids = raw.get("prompt_token_ids")
        _require(
            isinstance(token_ids, list)
            and token_ids
            and all(type(token_id) is int and token_id >= 0 for token_id in token_ids),
            "ShareGPT token IDs are invalid",
        )
        _require(
            raw.get("prompt_tokens") == len(token_ids)
            and raw.get("prompt_token_ids_sha256") == _canonical_digest(token_ids),
            "ShareGPT token identity mismatch",
        )
        selected_share.append(dict(raw))

    selected_affinity: list[dict[str, object]] = []
    session_turns: set[tuple[int, int]] = set()
    for sequence, raw in enumerate(affinity_rows):
        _require(isinstance(raw, dict), "affinity trace row must be an object")
        prompt = raw.get("prompt_text")
        prompt_hash = raw.get("prompt_text_sha256")
        session = _exact_int(raw.get("session"), "affinity session")
        turn = _exact_int(raw.get("turn"), "affinity turn")
        _require(session < AFFINITY_SESSIONS, "affinity session out of range")
        _require(turn < AFFINITY_TURNS, "affinity turn out of range")
        _require((session, turn) not in session_turns, "duplicate affinity session/turn")
        session_turns.add((session, turn))
        _require(isinstance(prompt, str) and prompt, "affinity prompt text missing")
        _require(
            _hash_valid(prompt_hash)
            and sha256_bytes(prompt.encode()) == prompt_hash,
            "affinity prompt hash mismatch",
        )
        _require(raw.get("sequence") == sequence, "affinity sequence is not contiguous")
        token_ids = raw.get("prompt_token_ids")
        expected_tokens = 512 + 128 * (turn + 1)
        _require(
            isinstance(token_ids, list)
            and len(token_ids) == expected_tokens
            and all(type(token_id) is int and token_id >= 0 for token_id in token_ids),
            "affinity token geometry is invalid",
        )
        _require(
            raw.get("prompt_token_ids_sha256") == _canonical_digest(token_ids),
            "affinity token identity mismatch",
        )
        selected_affinity.append(dict(raw))
    _require(
        session_turns
        == {
            (session, turn)
            for session in range(AFFINITY_SESSIONS)
            for turn in range(AFFINITY_TURNS)
        },
        "affinity trace matrix is incomplete",
    )

    return {
        "path": str(path),
        "sha256": TRACE_BUNDLE_SHA256,
        "schema_version": TRACE_SCHEMA,
        "sharegpt_descriptor_sha256": traces["sharegpt"]["sha256"],
        "shared_prefix_descriptor_sha256": traces["shared-prefix"]["sha256"],
        "scaling": selected_share,
        "affinity": selected_affinity,
    }


def trace_evidence(trace: Mapping[str, object]) -> dict[str, object]:
    scaling = trace["scaling"]
    affinity = trace["affinity"]
    assert isinstance(scaling, list)
    assert isinstance(affinity, list)
    scaling_evidence = [
        {
            "sequence": row["sequence"],
            "source_index": row.get("source_index"),
            "prompt_sha256": row["prompt_text_sha256"],
            "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
            "prompt_tokens": row.get("prompt_tokens"),
        }
        for row in scaling
    ]
    affinity_evidence = [
        {
            "sequence": row["sequence"],
            "session": row["session"],
            "turn": row["turn"],
            "prompt_sha256": row["prompt_text_sha256"],
            "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
            "prompt_tokens": len(row.get("prompt_token_ids", ())),
        }
        for row in affinity
    ]
    return {
        "bundle_sha256": trace["sha256"],
        "bundle_schema_version": trace["schema_version"],
        "sharegpt_descriptor_sha256": trace["sharegpt_descriptor_sha256"],
        "shared_prefix_descriptor_sha256": trace["shared_prefix_descriptor_sha256"],
        "scaling_selection_sha256": _canonical_digest(scaling_evidence),
        "affinity_selection_sha256": _canonical_digest(affinity_evidence),
        "scaling": scaling_evidence,
        "affinity": affinity_evidence,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            _require(
                raw.endswith(b"\n") and raw.strip(),
                f"invalid JSONL framing at line {line_number}",
            )
            value = json.loads(raw)
            _require(isinstance(value, dict), f"JSONL row {line_number} is not an object")
            rows.append(value)
    _require(rows, "raw evidence is empty")
    _require(
        all(row.get("row_index") == index for index, row in enumerate(rows)),
        "raw row indices are not contiguous",
    )
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    normalized = [{**row, "row_index": index} for index, row in enumerate(rows)]
    path.write_text(
        "".join(f"{canonical_json(row)}\n" for row in normalized),
        encoding="utf-8",
    )


def _request_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    phase = row.get("phase")
    if phase == "scaling":
        return (
            phase,
            row.get("repeat"),
            row.get("arm"),
            row.get("rate_rps"),
            row.get("sequence"),
        )
    if phase == "warmup":
        return (phase, row.get("repeat"), row.get("arm"), row.get("sequence"))
    if phase == "affinity":
        return (phase, row.get("arm"), row.get("sequence"))
    return (phase, row.get("row_index"))


def _cache_namespace(
    *,
    phase: str,
    arm: str,
    repeat: int | None = None,
    rate_rps: int | None = None,
) -> str:
    """Return an arm-disjoint, exactly one-token cache namespace."""

    return NAMESPACE_PIECES[
        _cache_namespace_slot(
            phase=phase,
            arm=arm,
            repeat=repeat,
            rate_rps=rate_rps,
        )
    ][1]


def _cache_namespace_slot(
    *,
    phase: str,
    arm: str,
    repeat: int | None,
    rate_rps: int | None,
) -> int:
    arm_index = {"single": 0, "dp": 1}.get(arm)
    if arm_index is None:
        raise GateEvidenceError("cache namespace arm is invalid")
    if phase == "warmup":
        _require(type(repeat) is int and 0 <= repeat < REPEATS, "warmup namespace repeat invalid")
        _require(rate_rps is None, "warmup namespace must not carry a rate")
        return int(repeat) * 2 + arm_index
    if phase == "scaling":
        _require(type(repeat) is int and 0 <= repeat < REPEATS, "scaling namespace repeat invalid")
        _require(rate_rps in RATES_RPS, "scaling namespace rate invalid")
        return 6 + int(repeat) * 10 + arm_index * 5 + RATES_RPS.index(int(rate_rps))
    if phase == "affinity":
        _require(repeat is None and rate_rps is None, "affinity namespace shape invalid")
        return 36 + arm_index
    raise GateEvidenceError("cache namespace phase is invalid")


def _expected_namespace_token_id(row: Mapping[str, object]) -> int:
    slot = _cache_namespace_slot(
        phase=str(row.get("phase")),
        arm=str(row.get("arm")),
        repeat=int(row["repeat"]) if type(row.get("repeat")) is int else None,
        rate_rps=(
            int(row["rate_rps"]) if type(row.get("rate_rps")) is int else None
        ),
    )
    return NAMESPACE_PIECES[slot][0]


def _expected_namespace_sha256(row: Mapping[str, object]) -> str:
    return sha256_bytes(
        _cache_namespace(
            phase=str(row.get("phase")),
            arm=str(row.get("arm")),
            repeat=(
                int(row["repeat"])
                if type(row.get("repeat")) is int
                else None
            ),
            rate_rps=(
                int(row["rate_rps"])
                if type(row.get("rate_rps")) is int
                else None
            ),
        ).encode()
    )


def _validate_request_common(
    row: Mapping[str, object],
    trace_scaling: Mapping[int, Mapping[str, object]],
    trace_affinity: Mapping[int, Mapping[str, object]],
) -> None:
    phase = row.get("phase")
    arm = row.get("arm")
    _require(phase in {"warmup", "scaling", "affinity"}, "request phase is invalid")
    _require(arm in {"single", "dp"}, "request arm is invalid")
    _require(row.get("retry_count") == 0, "request retry_count must be zero")
    _require(
        row.get("namespace_sha256") == _expected_namespace_sha256(row),
        "request cache namespace mismatch",
    )
    _require(
        row.get("namespace_token_id") == _expected_namespace_token_id(row),
        "request cache namespace token ID mismatch",
    )
    _require(row.get("status") == "success", "every formal request must succeed")
    _require(row.get("http_status") == 200, "every formal request must return HTTP 200")
    _require(row.get("error_type") is None, "successful request contains an error")
    _require(
        row.get("saw_done") is True and row.get("finish_reason") == "length",
        "request stream did not terminate exactly at the token limit",
    )
    sequence = _exact_int(row.get("sequence"), "request sequence")
    scheduled_ns = _exact_int(row.get("scheduled_ns"), "scheduled_ns", minimum=1)
    started_ns = _exact_int(row.get("started_ns"), "started_ns", minimum=scheduled_ns)
    first_token_ns = _exact_int(
        row.get("first_token_ns"), "first_token_ns", minimum=started_ns
    )
    ended_ns = _exact_int(row.get("ended_ns"), "ended_ns", minimum=first_token_ns)
    _require(
        row.get("ttft_ns") == first_token_ns - started_ns,
        "TTFT does not match raw timestamps",
    )
    _require(
        row.get("total_ns") == ended_ns - started_ns,
        "total latency does not match raw timestamps",
    )
    _require(_hash_valid(row.get("output_sha256")), "output digest is invalid")
    response_request_id = row.get("response_request_id")
    if arm == "dp":
        _require(
            isinstance(response_request_id, str) and response_request_id,
            "DP response X-Request-ID missing",
        )
    if phase in {"warmup", "scaling"}:
        expected = trace_scaling.get(int(row.get("prompt_index", -1)))
        _require(expected is not None, "scaling prompt index is invalid")
        _require(
            row.get("prompt_sha256") == expected.get("prompt_sha256"),
            "scaling prompt digest mismatch",
        )
        _require(
            row.get("completion_tokens") == SCALING_OUTPUT_TOKENS,
            "scaling request did not return exactly 32 tokens",
        )
        if phase == "scaling":
            repeat = _exact_int(row.get("repeat"), "scaling repeat")
            _require(
                row.get("prompt_index") == _scaling_prompt_index(repeat, sequence),
                "scaling prompt selection differs from the fixed permutation",
            )
        else:
            _require(
                row.get("prompt_index") == sequence,
                "warmup prompt selection differs from the fixed prefix",
            )
    else:
        expected = trace_affinity.get(sequence)
        _require(expected is not None, "affinity sequence is invalid")
        _require(
            row.get("prompt_sha256") == expected.get("prompt_sha256"),
            "affinity prompt digest mismatch",
        )
        _require(row.get("session") == expected.get("session"), "affinity session mismatch")
        _require(row.get("turn") == expected.get("turn"), "affinity turn mismatch")
        _require(
            expected.get("prompt_tokens")
            == 512 + 128 * (int(row["turn"]) + 1),
            "affinity trace token geometry mismatch",
        )
        _require(
            row.get("completion_tokens") == AFFINITY_OUTPUT_TOKENS,
            "affinity request did not return exactly one token",
        )
        prompt_tokens = _exact_int(row.get("prompt_tokens"), "usage.prompt_tokens", minimum=1)
        cached_tokens = _exact_int(row.get("cached_tokens"), "usage.cached_tokens")
        _require(cached_tokens <= prompt_tokens, "cached_tokens exceeds prompt_tokens")


def _median(values: Sequence[float]) -> float:
    _require(bool(values), "median requires samples")
    return float(statistics.median(values))


def _validate_backends_payload(
    value: object,
    *,
    role: str,
) -> None:
    _require(isinstance(value, dict) and value.get("role") == role, f"{role} /backends invalid")
    engines = value.get("engines")
    _require(isinstance(engines, list) and len(engines) == 1, f"{role} engine matrix invalid")
    engine = engines[0]
    _require(
        isinstance(engine, dict) and engine.get("model") == MODEL,
        f"{role} model identity invalid",
    )
    if role == "engine-host":
        _require(
            engine.get("engine_backend") == "kairyu"
            and engine.get("tensor_parallel_size") == 4,
            "single /backends does not prove one Kairyu TP4 engine",
        )
    else:
        via_replica = engine.get("via_replica")
        _require(
            engine.get("engine_backend") == "replica-pool"
            and isinstance(via_replica, dict)
            and via_replica.get("tensor_parallel_size") == 4,
            "gateway /backends does not prove TP4 replicas",
        )


def _validate_container_provenance(value: object, image_id: str) -> None:
    _require(isinstance(value, dict), "container provenance missing")
    _require(value.get("project") == "kairyu-qwen3-32b-a8", "compose project mismatch")
    compose_file = value.get("compose_file")
    _require(
        compose_file == "repo:examples/qwen3-32b-multi-gpu/a8-compose.yaml",
        "compose file identity mismatch",
    )
    services = value.get("services")
    _require(
        isinstance(services, dict)
        and set(services) == {"replica0", "replica1", "gateway"},
        "A8 service matrix mismatch",
    )
    expected_devices = {
        "replica0": ["0", "1", "2", "3"],
        "replica1": ["4", "5", "6", "7"],
        "gateway": [],
    }
    expected_cpus = {
        "replica0": "0-14,16-30,64-78,80-94",
        "replica1": "32-46,48-62,96-110,112-126",
        "gateway": "15,31,47,63,79,95,111,127",
    }
    expected_ports = {"replica0": "8200", "replica1": "8201", "gateway": "8202"}
    expected_source = "repo:kairyu"
    expected_configs = {
        "replica0": "repo:examples/qwen3-32b-multi-gpu/a8-replica.yaml",
        "replica1": "repo:examples/qwen3-32b-multi-gpu/a8-replica.yaml",
        "gateway": "repo:examples/qwen3-32b-multi-gpu/a8-gateway.yaml",
    }
    for service, expected_gpu_ids in expected_devices.items():
        descriptor = services[service]
        _require(isinstance(descriptor, dict), f"{service} descriptor invalid")
        _require(
            isinstance(descriptor.get("container_id"), str)
            and HEX64_RE.fullmatch(str(descriptor["container_id"])) is not None,
            f"{service} container ID invalid",
        )
        _require(
            descriptor.get("image_id") == image_id
            and descriptor.get("running") is True
            and isinstance(descriptor.get("started_at"), str)
            and bool(descriptor["started_at"])
            and _hash_valid(descriptor.get("compose_config_sha256"))
            and descriptor.get("device_ids") == expected_gpu_ids
            and descriptor.get("cpuset_cpus") == expected_cpus[service]
            and descriptor.get("port") == expected_ports[service],
            f"{service} runtime identity mismatch",
        )
        mounts = descriptor.get("mounts")
        _require(isinstance(mounts, list), f"{service} mount evidence invalid")
        by_destination = {
            mount.get("destination"): mount
            for mount in mounts
            if isinstance(mount, dict)
        }
        _require(
            isinstance(by_destination.get("/app/kairyu"), dict)
            and by_destination["/app/kairyu"].get("type") == "bind"
            and by_destination["/app/kairyu"].get("rw") is False
            and by_destination["/app/kairyu"].get("source") == expected_source
            and isinstance(by_destination.get("/etc/kairyu/config.yaml"), dict)
            and by_destination["/etc/kairyu/config.yaml"].get("type") == "bind"
            and by_destination["/etc/kairyu/config.yaml"].get("rw") is False
            and by_destination["/etc/kairyu/config.yaml"].get("source")
            == expected_configs[service],
            f"{service} source/config mounts are not read-only",
        )
        if service != "gateway":
            model_mount = by_destination.get("/models")
            _require(
                isinstance(model_mount, dict)
                and model_mount.get("type") == "volume"
                and model_mount.get("name") == "kairyu-qwen3-32b_qwen3-32b"
                and model_mount.get("source")
                == "volume:kairyu-qwen3-32b_qwen3-32b"
                and model_mount.get("rw") is False,
                f"{service} model volume identity mismatch",
            )
        else:
            evidence_mount = by_destination.get("/evidence")
            _require(
                isinstance(evidence_mount, dict)
                and evidence_mount.get("type") == "bind"
                and evidence_mount.get("source") == "artifact:run-dir"
                and evidence_mount.get("rw") is True,
                "gateway evidence mount identity mismatch",
            )


def _validate_provenance(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "run provenance missing")
    provenance = dict(value)
    _require(
        provenance.get("model") == MODEL
        and provenance.get("model_revision") == MODEL_REVISION,
        "model identity mismatch",
    )
    image_id = provenance.get("image_id")
    _require(
        isinstance(image_id, str) and IMAGE_ID_RE.fullmatch(image_id) is not None,
        "image ID invalid",
    )
    _require(
        provenance.get("source_commit")
        and COMMIT_RE.fullmatch(str(provenance["source_commit"])) is not None
        and provenance.get("tracked_tree_clean") is True,
        "source identity is invalid or dirty",
    )
    _require(
        provenance.get("single_url") == "http://127.0.0.1:8200/v1"
        and provenance.get("dp_url") == "http://127.0.0.1:8202/v1",
        "formal endpoint identity mismatch",
    )
    _validate_backends_payload(provenance.get("single_backends"), role="engine-host")
    _validate_backends_payload(provenance.get("dp_backends"), role="gateway")

    gpus = provenance.get("gpus")
    _require(isinstance(gpus, list) and len(gpus) == 8, "eight-GPU inventory missing")
    _require(
        [gpu.get("index") for gpu in gpus if isinstance(gpu, dict)] == list(range(8)),
        "GPU ordinal inventory mismatch",
    )
    uuids: list[str] = []
    bus_ids: list[str] = []
    memory_used: list[int] = []
    for gpu in gpus:
        _require(isinstance(gpu, dict), "GPU inventory row invalid")
        uuid = gpu.get("uuid")
        bus_id = gpu.get("pci_bus_id")
        name = gpu.get("name")
        total = gpu.get("memory_total_mib")
        used = gpu.get("memory_used_mib")
        _require(
            isinstance(uuid, str)
            and uuid.startswith("GPU-")
            and isinstance(bus_id, str)
            and bus_id
            and isinstance(name, str)
            and "RTX PRO 6000 Blackwell" in name
            and type(total) is int
            and total >= 90_000
            and type(used) is int
            and used >= 0,
            "GPU physical identity row invalid",
        )
        uuids.append(uuid)
        bus_ids.append(bus_id)
        memory_used.append(used)
    _require(len(set(uuids)) == len(set(bus_ids)) == 8, "GPU UUID/PCI identity is not unique")
    _require(max(memory_used) - min(memory_used) <= 4096, "GPU memory ownership is asymmetric")
    topology = provenance.get("nvidia_smi_topology")
    _require(
        isinstance(topology, list)
        and topology
        and all(isinstance(line, str) for line in topology)
        and all(any(f"GPU{index}" in line for line in topology) for index in range(8)),
        "physical GPU topology evidence is incomplete",
    )
    _require(provenance.get("gpu_count") == 8, "formal evidence requires exactly eight GPUs")
    _require(provenance.get("single_gpu_ordinals") == [0, 1, 2, 3], "single topology mismatch")
    _require(
        provenance.get("dp_gpu_ordinals") == list(range(8)),
        "DP topology must cover GPU ordinals 0-7",
    )
    containers = provenance.get("containers")
    _validate_container_provenance(containers, image_id)
    assert isinstance(containers, dict)
    services = containers["services"]
    assert isinstance(services, dict)
    replica0 = services["replica0"]
    assert isinstance(replica0, dict)

    checkpoint = provenance.get("checkpoint")
    _require(
        isinstance(checkpoint, dict),
        "checkpoint provenance is missing",
    )
    _require(
        all(
            checkpoint.get(key) == expected
            for key, expected in {
                "model": MODEL,
                "revision": MODEL_REVISION,
                "weights_sha256": CHECKPOINT_WEIGHTS_SHA256,
                "index_sha256": CHECKPOINT_INDEX_SHA256,
                "checkpoint_evidence_sha256": CHECKPOINT_EVIDENCE_SHA256,
                "tokenizer_sha256": PINNED_TOKENIZER_SHA256,
                "a7_raw_sha256": A7_RAW_SHA256,
                "a7_manifest_sha256": A7_MANIFEST_SHA256,
            }.items()
        )
        and checkpoint.get("attested_container_id")
        == replica0.get("container_id"),
        "checkpoint provenance differs from the retained A7 pin",
    )
    live_checkpoint = checkpoint.get("live_volume_attestation")
    _require(
        isinstance(live_checkpoint, dict),
        "live model-volume attestation is missing",
    )
    weights = live_checkpoint.get("weights")
    _require(
        isinstance(weights, list)
        and len(weights) == 17
        and all(
            isinstance(row, dict)
            and isinstance(row.get("file"), str)
            and re.fullmatch(
                r"model-[0-9]{5}-of-00017\.safetensors",
                str(row["file"]),
            )
            is not None
            and type(row.get("bytes")) is int
            and int(row["bytes"]) > 0
            and _hash_valid(row.get("sha256"))
            for row in weights
        )
        and len({str(row["file"]) for row in weights}) == 17,
        "live checkpoint shard evidence is invalid",
    )
    _require(
        live_checkpoint.get("weights_sha256") == CHECKPOINT_WEIGHTS_SHA256
        and sha256_bytes(json.dumps(weights, sort_keys=True).encode())
        == CHECKPOINT_WEIGHTS_SHA256
        and live_checkpoint.get("index_sha256") == CHECKPOINT_INDEX_SHA256
        and live_checkpoint.get("tokenizer_sha256")
        == PINNED_TOKENIZER_SHA256
        and live_checkpoint.get("config_sha256")
        == CHECKPOINT_CONFIG_SHA256,
        "live model-volume bytes differ from the pinned checkpoint",
    )
    config_hashes = provenance.get("config_file_sha256")
    expected_config_paths = {
        "bench/dp_scaling_g2_a8_bench.py",
        "examples/qwen3-32b-multi-gpu/a8-compose.yaml",
        "examples/qwen3-32b-multi-gpu/a8-replica.yaml",
        "examples/qwen3-32b-multi-gpu/a8-gateway.yaml",
        "examples/qwen3-32b-multi-gpu/a8-stack.sh",
    }
    _require(
        isinstance(config_hashes, dict)
        and set(config_hashes) == expected_config_paths
        and config_hashes
        == _committed_config_file_hashes(str(provenance["source_commit"])),
        "formal config source hashes are incomplete or differ from the source commit",
    )
    return provenance


def recompute_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_sha256: str,
) -> dict[str, object]:
    """Validate raw evidence and derive the complete binding manifest."""

    _require(_hash_valid(raw_sha256), "raw SHA-256 is invalid")
    _require(len(rows) >= 3, "raw evidence is incomplete")
    start = rows[0]
    end = rows[-1]
    _require(start.get("type") == "run", "raw evidence must start with run")
    _require(end.get("type") == "run_end", "raw evidence must end with run_end")
    _require(start.get("schema_version") == SCHEMA_VERSION, "raw schema mismatch")
    _require(start.get("gate") == GATE, "raw gate mismatch")
    _validate_formal_config(start.get("config"))
    trace = start.get("trace")
    _require(isinstance(trace, dict), "trace evidence missing")
    _require(
        trace.get("bundle_sha256") == TRACE_BUNDLE_SHA256
        and trace.get("bundle_schema_version") == TRACE_SCHEMA
        and trace.get("sharegpt_descriptor_sha256")
        == SHAREGPT_DESCRIPTOR_SHA256
        and trace.get("shared_prefix_descriptor_sha256")
        == AFFINITY_DESCRIPTOR_SHA256,
        "trace evidence differs from the pinned A6 bundle",
    )
    _require(
        trace.get("scaling_selection_sha256") == SCALING_SELECTION_SHA256
        and trace.get("affinity_selection_sha256")
        == AFFINITY_SELECTION_SHA256,
        "trace selection digests differ from the pinned A6 selections",
    )
    scaling_trace = trace.get("scaling")
    affinity_trace = trace.get("affinity")
    _require(
        isinstance(scaling_trace, list) and len(scaling_trace) == SCALING_REQUESTS,
        "scaling trace descriptor count mismatch",
    )
    _require(
        isinstance(affinity_trace, list) and len(affinity_trace) == AFFINITY_REQUESTS,
        "affinity trace descriptor count mismatch",
    )
    trace_scaling = {
        _exact_int(row.get("sequence"), "scaling trace sequence"): row
        for row in scaling_trace
        if isinstance(row, dict)
    }
    trace_affinity = {
        _exact_int(row.get("sequence"), "affinity trace sequence"): row
        for row in affinity_trace
        if isinstance(row, dict)
    }
    _require(
        set(trace_scaling) == set(range(SCALING_REQUESTS)),
        "scaling trace sequence matrix mismatch",
    )
    _require(
        set(trace_affinity) == set(range(AFFINITY_REQUESTS)),
        "affinity trace sequence matrix mismatch",
    )
    _require(
        all(
            row.get("prompt_tokens")
            == 512 + 128 * (int(row.get("turn", -1)) + 1)
            and _hash_valid(row.get("prompt_sha256"))
            and _hash_valid(row.get("prompt_token_ids_sha256"))
            for row in trace_affinity.values()
        ),
        "affinity trace geometry or identity is invalid",
    )
    _require(
        all(
            _hash_valid(row.get("prompt_sha256"))
            and _hash_valid(row.get("prompt_token_ids_sha256"))
            and type(row.get("prompt_tokens")) is int
            and int(row["prompt_tokens"]) > 0
            for row in trace_scaling.values()
        ),
        "scaling trace identity is invalid",
    )
    _require(
        _canonical_digest(scaling_trace)
        == trace.get("scaling_selection_sha256")
        and _canonical_digest(affinity_trace)
        == trace.get("affinity_selection_sha256"),
        "trace descriptor rows do not match their selection digests",
    )

    requests = [row for row in rows[1:-1] if row.get("type") == "request"]
    placements = [row for row in rows[1:-1] if row.get("type") == "placement"]
    _require(
        len(requests) + len(placements) == len(rows) - 2,
        "raw evidence contains an unknown row type",
    )
    identities = [_request_identity(row) for row in requests]
    _require(len(set(identities)) == len(identities), "duplicate formal request identity")
    for row in requests:
        _validate_request_common(row, trace_scaling, trace_affinity)

    scaling_by_cell: dict[tuple[int, str, int], list[Mapping[str, object]]] = defaultdict(list)
    warmup_by_arm: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    affinity_by_arm: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    observed_arm_orders: dict[int, list[str]] = defaultdict(list)
    prior_cell: tuple[int, str, int] | None = None
    for row in requests:
        phase = row["phase"]
        arm = str(row["arm"])
        if phase == "scaling":
            repeat = _exact_int(row.get("repeat"), "scaling repeat")
            rate = _exact_int(row.get("rate_rps"), "rate_rps", minimum=1)
            _require(repeat < REPEATS and rate in RATES_RPS, "scaling cell is outside the grid")
            cell = (repeat, arm, rate)
            scaling_by_cell[cell].append(row)
            if prior_cell != cell and arm not in observed_arm_orders[repeat]:
                observed_arm_orders[repeat].append(arm)
            prior_cell = cell
        elif phase == "warmup":
            repeat = _exact_int(row.get("repeat"), "warmup repeat")
            _require(repeat < REPEATS, "warmup repeat out of range")
            warmup_by_arm[(repeat, arm)].append(row)
            prior_cell = None
        else:
            affinity_by_arm[arm].append(row)
            prior_cell = None

    expected_cells = {
        (repeat, arm, rate)
        for repeat in range(REPEATS)
        for arm in ("single", "dp")
        for rate in RATES_RPS
    }
    _require(set(scaling_by_cell) == expected_cells, "scaling sweep matrix is incomplete")
    _require(
        set(warmup_by_arm)
        == {(repeat, arm) for repeat in range(REPEATS) for arm in ("single", "dp")},
        "warmup matrix is incomplete",
    )
    _require(set(affinity_by_arm) == {"single", "dp"}, "affinity arm matrix is incomplete")
    _require(
        [observed_arm_orders[index] for index in range(REPEATS)]
        == [list(order) for order in ARM_ORDERS],
        "paired arm order differs from the fixed contract",
    )
    expected_request_order: list[tuple[object, ...]] = []
    for repeat, order in enumerate(ARM_ORDERS):
        for arm in order:
            expected_request_order.extend(
                ("warmup", repeat, arm, sequence)
                for sequence in range(WARMUP_REQUESTS)
            )
            for rate in RATES_RPS:
                expected_request_order.extend(
                    ("scaling", repeat, arm, rate, sequence)
                    for sequence in range(SCALING_REQUESTS)
                )
    expected_request_order.extend(
        ("affinity", "single", sequence)
        for sequence in range(AFFINITY_REQUESTS)
    )
    expected_request_order.extend(
        ("affinity", "dp", sequence)
        for sequence in range(AFFINITY_REQUESTS)
    )
    _require(
        identities == expected_request_order,
        "formal request row order differs from the fixed execution plan",
    )

    cell_metrics: dict[str, dict[str, object]] = {}
    repeat_peaks: dict[int, dict[str, float]] = defaultdict(dict)
    for cell in sorted(expected_cells):
        repeat, arm, rate = cell
        samples = scaling_by_cell[cell]
        _require(len(samples) == SCALING_REQUESTS, f"cell {cell} request count mismatch")
        _require(
            {int(row["sequence"]) for row in samples} == set(range(SCALING_REQUESTS)),
            f"cell {cell} sequence matrix mismatch",
        )
        first_scheduled = min(int(row["scheduled_ns"]) for row in samples)
        by_sequence = {int(row["sequence"]): row for row in samples}
        _require(
            all(
                int(by_sequence[sequence]["scheduled_ns"])
                == first_scheduled + round(sequence * 1_000_000_000 / rate)
                for sequence in range(SCALING_REQUESTS)
            ),
            f"cell {cell} does not follow the fixed open-loop cadence",
        )
        last_terminal = max(int(row["ended_ns"]) for row in samples)
        wall_ns = last_terminal - first_scheduled
        _require(wall_ns > 0, f"cell {cell} wall interval is invalid")
        slo_successes = sum(int(row["ttft_ns"]) <= TTFT_SLO_NS for row in samples)
        goodput = slo_successes * 1_000_000_000 / wall_ns
        key = f"r{repeat}-{arm}-{rate}rps"
        cell_metrics[key] = {
            "repeat": repeat,
            "arm": arm,
            "rate_rps": rate,
            "request_count": len(samples),
            "slo_successes": slo_successes,
            "wall_ns": wall_ns,
            "goodput_rps": goodput,
            "ttft_p50_ns": nearest_rank_percentile(
                [int(row["ttft_ns"]) for row in samples], 0.50
            ),
            "ttft_p99_ns": nearest_rank_percentile(
                [int(row["ttft_ns"]) for row in samples], 0.99
            ),
        }
        repeat_peaks[repeat][arm] = max(
            repeat_peaks[repeat].get(arm, 0.0),
            goodput,
        )

    for key, samples in warmup_by_arm.items():
        _require(len(samples) == WARMUP_REQUESTS, f"warmup {key} count mismatch")
        _require(
            {int(row["sequence"]) for row in samples} == set(range(WARMUP_REQUESTS)),
            f"warmup {key} sequence mismatch",
        )
    for arm, samples in affinity_by_arm.items():
        _require(len(samples) == AFFINITY_REQUESTS, f"affinity {arm} count mismatch")
        _require(
            {int(row["sequence"]) for row in samples} == set(range(AFFINITY_REQUESTS)),
            f"affinity {arm} sequence mismatch",
        )
    execution_blocks: list[Sequence[Mapping[str, object]]] = []
    for repeat, order in enumerate(ARM_ORDERS):
        for arm in order:
            warmup_samples = warmup_by_arm[(repeat, arm)]
            _require(
                all(
                    int(warmup_samples[index]["scheduled_ns"])
                    >= int(warmup_samples[index - 1]["ended_ns"])
                    for index in range(1, len(warmup_samples))
                ),
                f"warmup {(repeat, arm)} was not serialized",
            )
            execution_blocks.append(warmup_samples)
            execution_blocks.extend(
                scaling_by_cell[(repeat, arm, rate)] for rate in RATES_RPS
            )
    execution_blocks.extend(
        (affinity_by_arm["single"], affinity_by_arm["dp"])
    )
    prior_end_ns = 0
    for block in execution_blocks:
        block_start_ns = min(int(row["scheduled_ns"]) for row in block)
        block_end_ns = max(int(row["ended_ns"]) for row in block)
        _require(
            block_start_ns >= prior_end_ns,
            "formal execution blocks overlap or were reordered",
        )
        prior_end_ns = block_end_ns

    single_goodput_positive = all(
        repeat_peaks[repeat]["single"] > 0.0 for repeat in range(REPEATS)
    )
    round_ratios = [
        (
            repeat_peaks[repeat]["dp"] / repeat_peaks[repeat]["single"]
            if repeat_peaks[repeat]["single"] > 0.0
            else 0.0
        )
        for repeat in range(REPEATS)
    ]
    median_goodput_ratio = _median(round_ratios)

    dp_request_by_id: dict[str, Mapping[str, object]] = {}
    for row in requests:
        if row.get("arm") != "dp":
            continue
        response_id = str(row["response_request_id"])
        _require(response_id not in dp_request_by_id, "duplicate DP response X-Request-ID")
        dp_request_by_id[response_id] = row
    placement_by_id: dict[str, Mapping[str, object]] = {}
    generations_by_replica: dict[str, set[str]] = defaultdict(set)
    for row in placements:
        response_id = row.get("response_request_id")
        _require(
            isinstance(response_id, str) and response_id in dp_request_by_id,
            "placement row does not join a DP response",
        )
        _require(response_id not in placement_by_id, "duplicate placement row")
        started_ns = _exact_int(row.get("placement_started_ns"), "placement_started_ns", minimum=1)
        selected_ns = _exact_int(row.get("selected_at_ns"), "selected_at_ns", minimum=started_ns)
        latency_ns = _exact_int(row.get("placement_latency_ns"), "placement_latency_ns")
        _require(latency_ns == selected_ns - started_ns, "placement latency arithmetic mismatch")
        _require(row.get("pool_size") == 2, "placement pool_size must be two")
        _require(row.get("eligible_size") == 2, "placement eligible_size must be two")
        _require(row.get("replica_id") in {"0", "1"}, "placement replica_id is invalid")
        request = dp_request_by_id[response_id]
        _require(
            int(request["started_ns"]) <= started_ns
            and selected_ns <= int(request["first_token_ns"]),
            "placement timestamps escape the joined client request bounds",
        )
        generation = row.get("replica_generation")
        _require(
            isinstance(generation, str)
            and re.fullmatch(r"[0-9a-f]{32}", generation) is not None,
            "placement replica generation is invalid",
        )
        generations_by_replica[str(row["replica_id"])].add(generation)
        _require(row.get("phase") == request.get("phase"), "placement phase join mismatch")
        placement_by_id[response_id] = row
    _require(
        set(placement_by_id) == set(dp_request_by_id),
        "every DP response must have exactly one placement row",
    )
    _require(
        set(generations_by_replica) == {"0", "1"}
        and all(len(generations) == 1 for generations in generations_by_replica.values()),
        "replica generation changed during the formal measurement",
    )

    scaling_placements = [
        placement_by_id[str(row["response_request_id"])]
        for row in requests
        if row.get("phase") == "scaling" and row.get("arm") == "dp"
    ]
    placement_p99_ns = nearest_rank_percentile(
        [int(row["placement_latency_ns"]) for row in scaling_placements],
        0.99,
    )
    _require(
        {str(row["replica_id"]) for row in scaling_placements} == {"0", "1"},
        "both replicas must serve the scaling sweep",
    )

    affinity_metrics: dict[str, dict[str, object]] = {}
    for arm in ("dp", "single"):
        samples = affinity_by_arm[arm]
        prompt_tokens = sum(int(row["prompt_tokens"]) for row in samples)
        cached_tokens = sum(int(row["cached_tokens"]) for row in samples)
        _require(prompt_tokens > 0, f"affinity {arm} prompt tokens are zero")
        affinity_metrics[arm] = {
            "request_count": len(samples),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "cache_rate": cached_tokens / prompt_tokens,
        }
    single_cache_rate = float(affinity_metrics["single"]["cache_rate"])
    single_cache_rate_positive = single_cache_rate > 0.0
    cache_retention = (
        float(affinity_metrics["dp"]["cache_rate"]) / single_cache_rate
        if single_cache_rate_positive
        else 0.0
    )
    dp_affinity_placements = [
        placement_by_id[str(row["response_request_id"])]
        for row in affinity_by_arm["dp"]
    ]
    _require(
        all(row.get("reason") == "session_affinity" for row in dp_affinity_placements),
        "every affinity request must use session_affinity",
    )
    affinity_replica_by_session: dict[int, set[str]] = defaultdict(set)
    for request in affinity_by_arm["dp"]:
        placement = placement_by_id[str(request["response_request_id"])]
        expected_session_hash = sha256_bytes(
            f"g2-a8-{int(request['session']):02d}".encode()
        )
        _require(
            placement.get("session_sha256") == expected_session_hash,
            "placement session hash does not match X-Session-ID",
        )
        affinity_replica_by_session[int(request["session"])].add(
            str(placement["replica_id"])
        )
    _require(
        set(affinity_replica_by_session) == set(range(AFFINITY_SESSIONS))
        and all(len(replicas) == 1 for replicas in affinity_replica_by_session.values()),
        "session affinity migrated a conversation",
    )
    _require(
        set().union(*affinity_replica_by_session.values()) == {"0", "1"},
        "affinity trace must use both replicas",
    )

    provenance = _validate_provenance(start.get("provenance"))
    _require(
        end.get("source_commit") == provenance.get("source_commit"),
        "source changed during run",
    )
    _require(end.get("completed") is True, "run did not reach a complete terminal record")
    _require(end.get("request_count") == len(requests), "run_end request count mismatch")
    _require(end.get("placement_count") == len(placements), "run_end placement count mismatch")
    _require(
        end.get("containers") == provenance.get("containers"),
        "container runtime changed during measurement",
    )
    _require(
        end.get("single_backends") == provenance.get("single_backends")
        and end.get("dp_backends") == provenance.get("dp_backends"),
        "/backends topology changed during measurement",
    )
    final_gpus = end.get("gpu_inventory")
    initial_gpus = provenance.get("gpus")
    _require(
        isinstance(final_gpus, list)
        and isinstance(initial_gpus, list)
        and [
            (gpu.get("index"), gpu.get("uuid"), gpu.get("pci_bus_id"))
            for gpu in final_gpus
            if isinstance(gpu, dict)
        ]
        == [
            (gpu.get("index"), gpu.get("uuid"), gpu.get("pci_bus_id"))
            for gpu in initial_gpus
            if isinstance(gpu, dict)
        ],
        "physical GPU inventory changed during measurement",
    )
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
    _exact_int(start.get("started_unix_ns"), "run started_unix_ns", minimum=1)
    _exact_int(end.get("ended_unix_ns"), "run ended_unix_ns", minimum=1)
    _require(
        all(
            started_monotonic_ns <= int(row["scheduled_ns"])
            and int(row["ended_ns"]) <= ended_monotonic_ns
            for row in requests
        ),
        "request timestamps escape the run bounds",
    )

    checks = {
        "fixed_open_loop_grid_three_paired_repeats_exact": True,
        "warmup_complete_and_excluded": True,
        "all_requests_retry_free_exact_success": True,
        "raw_timestamps_and_goodput_recomputed": True,
        "every_dp_response_has_one_exact_placement": True,
        "replica_generation_stable_for_full_run": True,
        "two_tp4_partitions_cover_same_eight_gpu_host": True,
        "both_dp_replicas_serve_scaling_and_affinity": True,
        "session_affinity_is_stable_for_all_64_sessions": True,
        "single_peak_goodput_positive_each_repeat": single_goodput_positive,
        "single_affinity_cache_rate_positive": single_cache_rate_positive,
        "median_peak_goodput_ratio_at_least_1_9": (
            median_goodput_ratio >= GOODPUT_RATIO_FLOOR
        ),
        "scaling_placement_latency_p99_below_10ms": (
            placement_p99_ns < PLACEMENT_P99_CEILING_NS
        ),
        "affinity_cache_rate_retention_at_least_0_90": (
            cache_retention >= CACHE_RETENTION_FLOOR
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "passed": all(checks.values()),
        "checks": checks,
        "raw": {
            "name": RAW_NAME,
            "sha256": raw_sha256,
            "row_count": len(rows),
            "request_rows": len(requests),
            "placement_rows": len(placements),
        },
        "goodput": {
            "cell_metrics": cell_metrics,
            "repeat_peak_rps": {
                str(repeat): repeat_peaks[repeat] for repeat in range(REPEATS)
            },
            "round_dp_over_single_ratios": round_ratios,
            "median_dp_over_single_ratio": median_goodput_ratio,
        },
        "router": {
            "definition": "outer HTTP ingress to ReplicaPool selection",
            "percentile_method": PERCENTILE_METHOD,
            "scaling_sample_count": len(scaling_placements),
            "placement_latency_p99_ns": placement_p99_ns,
            "replica_counts": dict(
                sorted(Counter(str(row["replica_id"]) for row in scaling_placements).items())
            ),
            "replica_generations": {
                replica_id: next(iter(generations_by_replica[replica_id]))
                for replica_id in sorted(generations_by_replica)
            },
        },
        "affinity": {
            **affinity_metrics,
            "dp_over_single_cache_rate": cache_retention,
            "stable_sessions": len(affinity_replica_by_session),
            "replicas_used": sorted(set().union(*affinity_replica_by_session.values())),
        },
        "provenance": provenance,
        "trace": {
            key: trace[key]
            for key in (
                "bundle_sha256",
                "bundle_schema_version",
                "sharegpt_descriptor_sha256",
                "shared_prefix_descriptor_sha256",
                "scaling_selection_sha256",
                "affinity_selection_sha256",
            )
        },
    }


def seal_artifact(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_NAME
    manifest_path = output_dir / MANIFEST_NAME
    _write_jsonl(raw_path, rows)
    stored = _read_jsonl(raw_path)
    manifest = recompute_manifest(stored, raw_sha256=sha256_file(raw_path))
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    return manifest


def verify_artifact(
    artifact_dir: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    raw_path = artifact_dir / RAW_NAME
    manifest_path = artifact_dir / MANIFEST_NAME
    _require(raw_path.is_file(), f"missing raw artifact: {raw_path}")
    _require(manifest_path.is_file(), f"missing manifest artifact: {manifest_path}")
    rows = _read_jsonl(raw_path)
    recomputed = recompute_manifest(rows, raw_sha256=sha256_file(raw_path))
    retained = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(retained, dict), "retained manifest must be an object")
    _require(retained == recomputed, "retained manifest differs from raw replay")
    if assert_gate and not recomputed["passed"]:
        failed = [name for name, passed in recomputed["checks"].items() if not passed]
        raise GateEvidenceError(f"G2 A8 gate failed: {', '.join(failed)}")
    return recomputed


def replay_artifact(
    artifact_dir: Path,
    *,
    assert_gate: bool = False,
) -> dict[str, object]:
    """Recompute the verdict from raw JSONL without reading the manifest."""

    raw_path = artifact_dir / RAW_NAME
    _require(raw_path.is_file(), f"missing raw artifact: {raw_path}")
    rows = _read_jsonl(raw_path)
    recomputed = recompute_manifest(rows, raw_sha256=sha256_file(raw_path))
    if assert_gate and not recomputed["passed"]:
        failed = [name for name, passed in recomputed["checks"].items() if not passed]
        raise GateEvidenceError(f"G2 A8 replay failed: {', '.join(failed)}")
    return recomputed


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
            f"{' '.join(arguments)} failed with exit {completed.returncode}: {detail}"
        )
    return completed.stdout.strip()


def _git_source(expected_commit: str | None) -> tuple[str, bool]:
    commit = _run_command(("git", "rev-parse", "HEAD"))
    if COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError(f"git returned an invalid commit: {commit!r}")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"source commit {commit} differs from --expected-commit {expected_commit}"
        )
    # Untracked operator-owned output and the user's .claude/worktrees are not
    # source mutations. Tracked/index changes always fail the formal preflight.
    tracked_status = _run_command(
        ("git", "status", "--porcelain", "--untracked-files=no")
    )
    return commit, not bool(tracked_status)


def _gpu_inventory() -> list[dict[str, object]]:
    text = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        index, uuid, bus_id, name, memory_total, memory_used = fields
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "pci_bus_id": bus_id,
                "name": name,
                "memory_total_mib": int(memory_total),
                "memory_used_mib": int(memory_used),
            }
        )
    if [row["index"] for row in rows] != list(range(8)):
        raise RuntimeError("formal G2 A8 run requires exact GPU ordinals 0-7")
    if len({str(row["uuid"]) for row in rows}) != 8:
        raise RuntimeError("nvidia-smi returned duplicate GPU UUIDs")
    memory_used = [int(row["memory_used_mib"]) for row in rows]
    if max(memory_used) - min(memory_used) > 4096:
        raise RuntimeError(
            "GPU memory is asymmetric by more than 4096 MiB; stop unrelated "
            "GPU owners before formal measurement"
        )
    return rows


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
    if topology_tp != 4:
        raise RuntimeError(f"{origin}/backends does not prove TP4 replicas")
    if backend != ("kairyu" if expected_role == "engine-host" else "replica-pool"):
        raise RuntimeError(f"{origin}/backends engine backend mismatch")
    return payload


def _usage_from_wire(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, dict):
        return None
    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    details = value.get("prompt_tokens_details")
    # Kairyu omits prompt_tokens_details when the exact cached count is zero.
    # /backends is pinned to Kairyu before measurement, so absence has the
    # documented exact meaning 0 rather than "unknown upstream field".
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else 0
    if (
        type(prompt_tokens) is not int
        or prompt_tokens <= 0
        or type(completion_tokens) is not int
        or completion_tokens < 0
        or type(cached_tokens) is not int
        or cached_tokens < 0
        or cached_tokens > prompt_tokens
    ):
        return None
    return prompt_tokens, completion_tokens, cached_tokens


async def _stream_request(
    client: httpx.AsyncClient,
    *,
    api_root: str,
    phase: str,
    arm: str,
    sequence: int,
    prompt_index: int | None,
    prompt: str,
    prompt_sha256: str,
    max_tokens: int,
    scheduled_ns: int,
    repeat: int | None = None,
    rate_rps: int | None = None,
    session: int | None = None,
    turn: int | None = None,
) -> dict[str, object]:
    namespace = _cache_namespace(
        phase=phase,
        arm=arm,
        repeat=repeat,
        rate_rps=rate_rps,
    )
    namespace_token_id = _expected_namespace_token_id(
        {
            "phase": phase,
            "arm": arm,
            "repeat": repeat,
            "rate_rps": rate_rps,
        }
    )
    now_ns = time.perf_counter_ns()
    if now_ns < scheduled_ns:
        await asyncio.sleep((scheduled_ns - now_ns) / 1_000_000_000)
    started_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    ended_ns = started_ns
    http_status: int | None = None
    response_request_id: str | None = None
    error_type: str | None = None
    usage: tuple[int, int, int] | None = None
    output_parts: list[str] = []
    saw_done = False
    finish_reasons: list[str] = []
    headers = (
        {"x-session-id": f"g2-a8-{session:02d}"}
        if session is not None
        else None
    )
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": namespace},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with client.stream(
            "POST",
            f"{api_root}/chat/completions",
            json=body,
            headers=headers,
        ) as response:
            http_status = response.status_code
            response_request_id = response.headers.get("x-request-id")
            if response.status_code != 200:
                error_type = "HTTPError"
                await response.aread()
            else:
                async for line in response.aiter_lines():
                    if line == f"{SSE_PREFIX}[DONE]":
                        saw_done = True
                        continue
                    if not line.startswith(SSE_PREFIX):
                        continue
                    chunk = json.loads(line[len(SSE_PREFIX) :])
                    parsed_usage = _usage_from_wire(chunk.get("usage"))
                    if parsed_usage is not None:
                        usage = parsed_usage
                    for choice in chunk.get("choices", ()):
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        finish_reason = choice.get("finish_reason")
                        if isinstance(finish_reason, str):
                            finish_reasons.append(finish_reason)
                        if not isinstance(delta, dict):
                            continue
                        for key in ("reasoning_content", "content"):
                            content = delta.get(key)
                            if isinstance(content, str) and content:
                                if first_token_ns is None:
                                    first_token_ns = time.perf_counter_ns()
                                output_parts.append(content)
    except Exception as error:
        error_type = type(error).__name__
    ended_ns = time.perf_counter_ns()
    if first_token_ns is None:
        first_token_ns = ended_ns
        if error_type is None:
            error_type = "MissingStreamToken"
    if usage is None and error_type is None:
        error_type = "IncompleteEngineUsage"
    prompt_tokens, completion_tokens, cached_tokens = usage or (None, None, None)
    if completion_tokens != max_tokens and error_type is None:
        error_type = "UnexpectedCompletionLength"
    if (not saw_done or finish_reasons != ["length"]) and error_type is None:
        error_type = "IncompleteStreamTermination"
    success = (
        error_type is None
        and http_status == 200
        and completion_tokens == max_tokens
        and saw_done
        and finish_reasons == ["length"]
    )
    return {
        "type": "request",
        "phase": phase,
        "arm": arm,
        "repeat": repeat,
        "rate_rps": rate_rps,
        "sequence": sequence,
        "prompt_index": prompt_index,
        "session": session,
        "turn": turn,
        "prompt_sha256": prompt_sha256,
        "namespace_sha256": sha256_bytes(namespace.encode()),
        "namespace_token_id": namespace_token_id,
        "scheduled_ns": scheduled_ns,
        "started_ns": started_ns,
        "first_token_ns": first_token_ns,
        "ended_ns": ended_ns,
        "ttft_ns": first_token_ns - started_ns,
        "total_ns": ended_ns - started_ns,
        "http_status": http_status,
        "response_request_id": response_request_id,
        "retry_count": 0,
        "status": "success" if success else "failure",
        "error_type": error_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "output_sha256": sha256_bytes("".join(output_parts).encode()),
        "saw_done": saw_done,
        "finish_reason": finish_reasons[0] if len(finish_reasons) == 1 else None,
    }


def _scaling_prompt_index(repeat: int, sequence: int) -> int:
    return (sequence + repeat * 17) % SCALING_REQUESTS


async def _measure_scaling_cell(
    client: httpx.AsyncClient,
    *,
    api_root: str,
    arm: str,
    repeat: int,
    rate_rps: int,
    trace_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    origin_ns = time.perf_counter_ns() + 100_000_000
    tasks = []
    for sequence in range(SCALING_REQUESTS):
        prompt_index = _scaling_prompt_index(repeat, sequence)
        trace = trace_rows[prompt_index]
        tasks.append(
            asyncio.create_task(
                _stream_request(
                    client,
                    api_root=api_root,
                    phase="scaling",
                    arm=arm,
                    repeat=repeat,
                    rate_rps=rate_rps,
                    sequence=sequence,
                    prompt_index=prompt_index,
                    prompt=str(trace["prompt_text"]),
                    prompt_sha256=str(trace["prompt_text_sha256"]),
                    max_tokens=SCALING_OUTPUT_TOKENS,
                    scheduled_ns=origin_ns
                    + round(sequence * 1_000_000_000 / rate_rps),
                )
            )
        )
    return list(await asyncio.gather(*tasks))


async def _measure_warmup(
    client: httpx.AsyncClient,
    *,
    api_root: str,
    arm: str,
    repeat: int,
    trace_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence in range(WARMUP_REQUESTS):
        trace = trace_rows[sequence]
        scheduled_ns = time.perf_counter_ns()
        rows.append(
            await _stream_request(
                client,
                api_root=api_root,
                phase="warmup",
                arm=arm,
                repeat=repeat,
                rate_rps=None,
                sequence=sequence,
                prompt_index=sequence,
                prompt=str(trace["prompt_text"]),
                prompt_sha256=str(trace["prompt_text_sha256"]),
                max_tokens=SCALING_OUTPUT_TOKENS,
                scheduled_ns=scheduled_ns,
            )
        )
    return rows


async def _measure_affinity(
    client: httpx.AsyncClient,
    *,
    api_root: str,
    arm: str,
    trace_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence, trace in enumerate(trace_rows):
        rows.append(
            await _stream_request(
                client,
                api_root=api_root,
                phase="affinity",
                arm=arm,
                repeat=None,
                rate_rps=None,
                sequence=sequence,
                prompt_index=None,
                prompt=str(trace["prompt_text"]),
                prompt_sha256=str(trace["prompt_text_sha256"]),
                max_tokens=AFFINITY_OUTPUT_TOKENS,
                scheduled_ns=time.perf_counter_ns(),
                session=int(trace["session"]),
                turn=int(trace["turn"]),
            )
        )
    return rows


def _placement_file_offset(path: Path) -> int:
    if not path.exists():
        return 0
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("gateway placement log must be a regular file")
    with path.open("rb") as handle:
        handle.seek(0, 2)
        return handle.tell()


def _read_placement_segment(path: Path, offset: int) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read()
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        return []
    rows: list[dict[str, object]] = []
    for raw in payload.splitlines():
        value = json.loads(raw)
        if isinstance(value, dict) and value.get("kind") == "replica":
            rows.append(value)
    return rows


async def _collect_placements(
    path: Path,
    *,
    offset: int,
    dp_requests: Sequence[Mapping[str, object]],
    timeout_s: float,
) -> list[dict[str, object]]:
    request_by_id: dict[str, Mapping[str, object]] = {}
    for request in dp_requests:
        response_id = request.get("response_request_id")
        if not isinstance(response_id, str) or not response_id:
            raise RuntimeError("a DP request did not return X-Request-ID")
        if response_id in request_by_id:
            raise RuntimeError("duplicate DP X-Request-ID")
        request_by_id[response_id] = request
    deadline = time.monotonic() + timeout_s
    latest: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        latest = _read_placement_segment(path, offset)
        observed = [
            row.get("request_id")
            for row in latest
            if isinstance(row.get("request_id"), str)
        ]
        if len(observed) == len(request_by_id) and set(observed) == set(request_by_id):
            break
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError(
            f"placement log did not expose all {len(request_by_id)} decisions "
            f"within {timeout_s}s"
        )
    if len(latest) != len(request_by_id):
        raise RuntimeError("placement segment contains unrelated or duplicate decisions")
    placements: list[dict[str, object]] = []
    for raw in latest:
        response_id = str(raw["request_id"])
        request = request_by_id[response_id]
        placements.append(
            {
                "type": "placement",
                "response_request_id": response_id,
                "phase": request["phase"],
                "repeat": request.get("repeat"),
                "rate_rps": request.get("rate_rps"),
                "sequence": request["sequence"],
                "session": request.get("session"),
                "turn": request.get("turn"),
                "replica_id": raw.get("replica_id"),
                "replica_generation": raw.get("replica_generation"),
                "reason": raw.get("reason"),
                "pool_size": raw.get("pool_size"),
                "eligible_size": raw.get("eligible_size"),
                "placement_started_ns": raw.get("placement_started_ns"),
                "selected_at_ns": raw.get("selected_at_ns"),
                "placement_latency_ns": raw.get("placement_latency_ns"),
                "session_sha256": raw.get("session_sha256"),
            }
        )
    return placements


def _config_file_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in CONFIG_PATHS:
        if not path.is_file():
            raise RuntimeError(f"formal source file missing: {path}")
        result[path.as_posix()] = sha256_file(path)
    return result


def _committed_config_file_hashes(commit: str) -> dict[str, str]:
    if COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("cannot resolve config files from an invalid source commit")
    result: dict[str, str] = {}
    for path in CONFIG_PATHS:
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
        result[path.as_posix()] = sha256_bytes(completed.stdout)
    return result


def _checkpoint_reference(container_id: str) -> dict[str, object]:
    raw_path = A7_ARTIFACT_DIR / "tp-kv-hit-g2-a7-raw.jsonl"
    manifest_path = A7_ARTIFACT_DIR / "tp-kv-hit-g2-a7-manifest.json"
    if (
        not raw_path.is_file()
        or not manifest_path.is_file()
        or sha256_file(raw_path) != A7_RAW_SHA256
        or sha256_file(manifest_path) != A7_MANIFEST_SHA256
    ):
        raise RuntimeError("retained A7 checkpoint reference artifact is missing or changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("passed") is not True
        or not isinstance(manifest.get("raw"), dict)
        or manifest["raw"].get("sha256") != A7_RAW_SHA256
    ):
        raise RuntimeError("retained A7 checkpoint reference does not replay as accepted")
    with raw_path.open("rb") as handle:
        next(handle)
        scenario = json.loads(next(handle))
    try:
        retained = scenario["provenance_start"]["configuration"]["value"][
            "checkpoint_evidence"
        ]["value"]["checkpoint"]
        tokenizer_digests = retained["tokenizer_sha256"]
        metadata_digests = retained["metadata_sha256"]
        expected_live = {
            "weights": retained["weights"],
            "weights_sha256": retained["weights_sha256"],
            "index_sha256": retained["index_sha256"],
            "tokenizer_sha256": tokenizer_digests["tokenizer.json"],
            "config_sha256": metadata_digests["config.json"],
        }
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "retained A7 checkpoint identity is incomplete"
        ) from error
    if (
        not isinstance(expected_live["weights"], list)
        or len(expected_live["weights"]) != 17
        or expected_live["weights_sha256"] != CHECKPOINT_WEIGHTS_SHA256
        or expected_live["index_sha256"] != CHECKPOINT_INDEX_SHA256
        or expected_live["tokenizer_sha256"] != PINNED_TOKENIZER_SHA256
        or not _hash_valid(expected_live["config_sha256"])
    ):
        raise RuntimeError("retained A7 checkpoint identity differs from its pins")
    completed = subprocess.run(
        (
            "docker",
            "exec",
            container_id,
            "python",
            "-c",
            CHECKPOINT_HASH_SCRIPT,
            "/models/qwen3-32b",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"live checkpoint attestation failed with exit "
            f"{completed.returncode}: {detail}"
        )
    measured = json.loads(completed.stdout)
    if measured != expected_live:
        raise RuntimeError(
            "live A8 model-volume bytes differ from pinned A7 checkpoint evidence"
        )
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "weights_sha256": CHECKPOINT_WEIGHTS_SHA256,
        "index_sha256": CHECKPOINT_INDEX_SHA256,
        "checkpoint_evidence_sha256": CHECKPOINT_EVIDENCE_SHA256,
        "tokenizer_sha256": PINNED_TOKENIZER_SHA256,
        "a7_raw_sha256": A7_RAW_SHA256,
        "a7_manifest_sha256": A7_MANIFEST_SHA256,
        "attested_container_id": container_id,
        "live_volume_attestation": measured,
    }


def _docker_stack_provenance(
    image_id: str,
    *,
    evidence_dir: Path,
) -> dict[str, object]:
    compose_path = Path(
        "examples/qwen3-32b-multi-gpu/a8-compose.yaml"
    ).resolve()
    compose_prefix = (
        "docker",
        "compose",
        "--project-name",
        "kairyu-qwen3-32b-a8",
        "-f",
        str(compose_path),
    )
    expected_devices = {
        "replica0": ["0", "1", "2", "3"],
        "replica1": ["4", "5", "6", "7"],
        "gateway": [],
    }
    expected_cpus = {
        "replica0": "0-14,16-30,64-78,80-94",
        "replica1": "32-46,48-62,96-110,112-126",
        "gateway": "15,31,47,63,79,95,111,127",
    }
    expected_ports = {
        "replica0": "8200",
        "replica1": "8201",
        "gateway": "8202",
    }
    expected_source = str(Path("kairyu").resolve())
    expected_configs = {
        "replica0": str(
            Path("examples/qwen3-32b-multi-gpu/a8-replica.yaml").resolve()
        ),
        "replica1": str(
            Path("examples/qwen3-32b-multi-gpu/a8-replica.yaml").resolve()
        ),
        "gateway": str(
            Path("examples/qwen3-32b-multi-gpu/a8-gateway.yaml").resolve()
        ),
    }
    services: dict[str, object] = {}
    for service in ("replica0", "replica1", "gateway"):
        config_hash_result = subprocess.run(
            (*compose_prefix, "config", "--hash", service),
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "KAIRYU_A8_RUN_DIR": str(evidence_dir),
            },
        )
        if config_hash_result.returncode != 0:
            detail = (
                config_hash_result.stderr or config_hash_result.stdout
            ).strip()
            raise RuntimeError(
                f"Compose config hash for {service} failed: {detail}"
            )
        config_hash_line = config_hash_result.stdout.strip()
        config_hash_fields = config_hash_line.split()
        if (
            len(config_hash_fields) != 2
            or config_hash_fields[0] != service
            or not _hash_valid(config_hash_fields[1])
        ):
            raise RuntimeError(f"Compose config hash for {service} is invalid")
        compose_config_sha256 = config_hash_fields[1]
        container_id = _run_command((*compose_prefix, "ps", "-q", service))
        if not container_id or "\n" in container_id:
            raise RuntimeError(f"A8 service {service} does not resolve to one container")
        inspected = json.loads(_run_command(("docker", "inspect", container_id)))
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise RuntimeError(f"docker inspect for {service} is not singular")
        value = inspected[0]
        if not isinstance(value, dict):
            raise RuntimeError(f"docker inspect for {service} is invalid")
        config = value.get("Config")
        host = value.get("HostConfig")
        state = value.get("State")
        mounts = value.get("Mounts")
        network = value.get("NetworkSettings")
        if not all(
            isinstance(item, dict)
            for item in (config, host, state, network)
        ) or not isinstance(mounts, list):
            raise RuntimeError(f"docker inspect for {service} is incomplete")
        labels = config.get("Labels")
        if not isinstance(labels, dict):
            raise RuntimeError(f"docker labels for {service} are missing")
        if (
            value.get("Image") != image_id
            or labels.get("com.kairyu.a8.image-id") != image_id
            or labels.get("com.kairyu.a8.role") != service
            or labels.get("com.docker.compose.config-hash")
            != compose_config_sha256
            or state.get("Running") is not True
            or not isinstance(state.get("StartedAt"), str)
            or not state["StartedAt"]
            or host.get("CpusetCpus") != expected_cpus[service]
        ):
            raise RuntimeError(f"A8 runtime identity mismatch for {service}")
        device_requests = host.get("DeviceRequests") or []
        device_ids: list[str] = []
        if service != "gateway":
            if not isinstance(device_requests, list) or len(device_requests) != 1:
                raise RuntimeError(f"A8 GPU request missing for {service}")
            request = device_requests[0]
            if (
                not isinstance(request, dict)
                or request.get("Driver") != "nvidia"
                or request.get("Capabilities") != [["gpu"]]
                or request.get("DeviceIDs") != expected_devices[service]
            ):
                raise RuntimeError(f"A8 GPU partition mismatch for {service}")
            device_ids = list(request["DeviceIDs"])
        elif device_requests:
            raise RuntimeError("A8 gateway unexpectedly owns a GPU")

        retained_mounts: list[dict[str, object]] = []
        for mount in mounts:
            if not isinstance(mount, dict):
                raise RuntimeError(f"A8 mount entry for {service} is invalid")
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
        source_mount = by_destination.get("/app/kairyu")
        config_mount = by_destination.get("/etc/kairyu/config.yaml")
        if (
            source_mount is None
            or source_mount.get("source") != expected_source
            or source_mount.get("rw") is not False
            or config_mount is None
            or config_mount.get("source") != expected_configs[service]
            or config_mount.get("rw") is not False
        ):
            raise RuntimeError(f"A8 read-only source/config mount mismatch for {service}")
        if service != "gateway":
            model_mount = by_destination.get("/models")
            if (
                model_mount is None
                or model_mount.get("type") != "volume"
                or model_mount.get("name") != "kairyu-qwen3-32b_qwen3-32b"
                or model_mount.get("rw") is not False
            ):
                raise RuntimeError(f"A8 model mount mismatch for {service}")
        else:
            evidence_mount = by_destination.get("/evidence")
            if (
                evidence_mount is None
                or evidence_mount.get("type") != "bind"
                or evidence_mount.get("source") != str(evidence_dir)
                or evidence_mount.get("rw") is not True
            ):
                raise RuntimeError("A8 gateway evidence mount mismatch")
        ports = host.get("PortBindings")
        binding = ports.get("8000/tcp") if isinstance(ports, dict) else None
        if (
            not isinstance(binding, list)
            or len(binding) != 1
            or binding[0].get("HostIp") != "127.0.0.1"
            or binding[0].get("HostPort") != expected_ports[service]
        ):
            raise RuntimeError(f"A8 port binding mismatch for {service}")
        normalized_sources = {
            "/app/kairyu": "repo:kairyu",
            "/etc/kairyu/config.yaml": (
                "repo:examples/qwen3-32b-multi-gpu/a8-gateway.yaml"
                if service == "gateway"
                else "repo:examples/qwen3-32b-multi-gpu/a8-replica.yaml"
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
        services[service] = {
            "container_id": container_id,
            "image_id": value.get("Image"),
            "running": state.get("Running"),
            "started_at": state.get("StartedAt"),
            "compose_config_sha256": compose_config_sha256,
            "device_ids": device_ids,
            "cpuset_cpus": host.get("CpusetCpus"),
            "port": expected_ports[service],
            "mounts": sorted(
                normalized_mounts,
                key=lambda mount: str(mount["destination"]),
            ),
        }
    return {
        "project": "kairyu-qwen3-32b-a8",
        "compose_file": "repo:examples/qwen3-32b-multi-gpu/a8-compose.yaml",
        "services": services,
    }


async def measure(
    *,
    single_url: str,
    dp_url: str,
    trace_bundle: Path,
    gateway_placement_log: Path,
    output_dir: Path,
    image_id: str,
    expected_commit: str | None,
    timeout_s: float,
    placement_timeout_s: float,
    assert_gate: bool,
) -> dict[str, object]:
    if IMAGE_ID_RE.fullmatch(image_id) is None:
        raise ValueError("--image-id must be sha256:<64 lowercase hex>")
    if timeout_s <= 0 or placement_timeout_s <= 0:
        raise ValueError("timeouts must be positive")
    single_api = _normalize_api_root(single_url)
    dp_api = _normalize_api_root(dp_url)
    resolved_output_dir = output_dir.resolve()
    if gateway_placement_log.resolve().parent != resolved_output_dir:
        raise ValueError(
            "--gateway-placement-log must be directly inside --output-dir"
        )
    trace = load_trace_bundle(trace_bundle)
    commit, tracked_clean = _git_source(expected_commit)
    if not tracked_clean:
        raise RuntimeError("formal measurement requires a clean tracked/index source tree")
    config_file_hashes = _config_file_hashes()
    if config_file_hashes != _committed_config_file_hashes(commit):
        raise RuntimeError("formal config files differ from the clean source commit")
    gpu_inventory = _gpu_inventory()
    topology = _run_command(("nvidia-smi", "topo", "-m")).splitlines()
    if not topology or not all(
        any(f"GPU{index}" in line for line in topology) for index in range(8)
    ):
        raise RuntimeError("nvidia-smi topology does not enumerate all eight GPUs")
    containers = _docker_stack_provenance(
        image_id,
        evidence_dir=resolved_output_dir,
    )
    services = containers.get("services")
    replica0 = services.get("replica0") if isinstance(services, dict) else None
    container_id = (
        replica0.get("container_id") if isinstance(replica0, dict) else None
    )
    if (
        not isinstance(container_id, str)
        or HEX64_RE.fullmatch(container_id) is None
    ):
        raise RuntimeError("A8 replica0 container identity is missing")
    checkpoint = _checkpoint_reference(container_id)
    placement_offset = _placement_file_offset(gateway_placement_log)
    limits = httpx.Limits(
        max_connections=256,
        max_keepalive_connections=128,
    )
    async with (
        httpx.AsyncClient(timeout=timeout_s, limits=limits) as single_client,
        httpx.AsyncClient(timeout=timeout_s, limits=limits) as dp_client,
    ):
        single_backends = await _endpoint_metadata(
            single_client,
            single_api,
            expected_role="engine-host",
        )
        dp_backends = await _endpoint_metadata(
            dp_client,
            dp_api,
            expected_role="gateway",
        )
        provenance = {
            "source_commit": commit,
            "tracked_tree_clean": tracked_clean,
            "image_id": image_id,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "gpu_count": len(gpu_inventory),
            "gpus": gpu_inventory,
            "nvidia_smi_topology": topology,
            "single_gpu_ordinals": [0, 1, 2, 3],
            "dp_gpu_ordinals": list(range(8)),
            "containers": containers,
            "checkpoint": checkpoint,
            "single_url": single_api,
            "dp_url": dp_api,
            "single_backends": single_backends,
            "dp_backends": dp_backends,
            "config_file_sha256": config_file_hashes,
            "placement_log_start_offset": placement_offset,
        }
        rows: list[dict[str, object]] = [
            {
                "type": "run",
                "schema_version": SCHEMA_VERSION,
                "gate": GATE,
                "config": formal_config(),
                "trace": trace_evidence(trace),
                "provenance": provenance,
                "started_unix_ns": time.time_ns(),
                "started_monotonic_ns": time.perf_counter_ns(),
            }
        ]
        clients = {"single": single_client, "dp": dp_client}
        api_roots = {"single": single_api, "dp": dp_api}
        scaling_trace = trace["scaling"]
        affinity_trace = trace["affinity"]
        assert isinstance(scaling_trace, list)
        assert isinstance(affinity_trace, list)
        for repeat, order in enumerate(ARM_ORDERS):
            for arm in order:
                rows.extend(
                    await _measure_warmup(
                        clients[arm],
                        api_root=api_roots[arm],
                        arm=arm,
                        repeat=repeat,
                        trace_rows=scaling_trace,
                    )
                )
                for rate_rps in RATES_RPS:
                    rows.extend(
                        await _measure_scaling_cell(
                            clients[arm],
                            api_root=api_roots[arm],
                            arm=arm,
                            repeat=repeat,
                            rate_rps=rate_rps,
                            trace_rows=scaling_trace,
                        )
                    )

        # The namespaces are arm-disjoint, so the documented single-first order
        # cannot transfer reusable prefix tokens into the later DP candidate.
        rows.extend(
            await _measure_affinity(
                single_client,
                api_root=single_api,
                arm="single",
                trace_rows=affinity_trace,
            )
        )
        rows.extend(
            await _measure_affinity(
                dp_client,
                api_root=dp_api,
                arm="dp",
                trace_rows=affinity_trace,
            )
        )
        final_single_backends = await _endpoint_metadata(
            single_client,
            single_api,
            expected_role="engine-host",
        )
        final_dp_backends = await _endpoint_metadata(
            dp_client,
            dp_api,
            expected_role="gateway",
        )

    dp_requests = [
        row
        for row in rows
        if row.get("type") == "request" and row.get("arm") == "dp"
    ]
    placements = await _collect_placements(
        gateway_placement_log,
        offset=placement_offset,
        dp_requests=dp_requests,
        timeout_s=placement_timeout_s,
    )
    rows.extend(placements)
    final_gpu_inventory = _gpu_inventory()
    final_containers = _docker_stack_provenance(
        image_id,
        evidence_dir=resolved_output_dir,
    )
    final_commit, final_clean = _git_source(commit)
    if not final_clean:
        raise RuntimeError("tracked source changed during measurement")
    rows.append(
        {
            "type": "run_end",
            "source_commit": final_commit,
            "completed": True,
            "request_count": sum(row.get("type") == "request" for row in rows),
            "placement_count": len(placements),
            "gpu_inventory": final_gpu_inventory,
            "containers": final_containers,
            "single_backends": final_single_backends,
            "dp_backends": final_dp_backends,
            "ended_unix_ns": time.time_ns(),
            "ended_monotonic_ns": time.perf_counter_ns(),
        }
    )
    manifest = seal_artifact(output_dir, rows)
    if assert_gate and not manifest["passed"]:
        failed = [name for name, passed in manifest["checks"].items() if not passed]
        raise GateEvidenceError(
            f"G2 A8 gate failed after retaining evidence: {', '.join(failed)}"
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_parser = subparsers.add_parser(
        "measure",
        help="run the fixed real Qwen3-32B eight-GPU measurement",
    )
    measure_parser.add_argument("--single-url", required=True)
    measure_parser.add_argument("--dp-url", required=True)
    measure_parser.add_argument("--trace-bundle", type=Path, required=True)
    measure_parser.add_argument("--gateway-placement-log", type=Path, required=True)
    measure_parser.add_argument("--output-dir", type=Path, required=True)
    measure_parser.add_argument("--image-id", required=True)
    measure_parser.add_argument("--expected-commit")
    measure_parser.add_argument("--timeout-s", type=float, default=120.0)
    measure_parser.add_argument("--placement-timeout-s", type=float, default=30.0)
    measure_parser.add_argument("--assert-gate", action="store_true")

    for command in ("verify", "replay"):
        command_parser = subparsers.add_parser(
            command,
            help=(
                "verify retained manifest against raw evidence"
                if command == "verify"
                else "independently replay raw evidence"
            ),
        )
        command_parser.add_argument("--artifact", type=Path, required=True)
        command_parser.add_argument("--assert-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "measure":
            result = asyncio.run(
                measure(
                    single_url=args.single_url,
                    dp_url=args.dp_url,
                    trace_bundle=args.trace_bundle,
                    gateway_placement_log=args.gateway_placement_log,
                    output_dir=args.output_dir,
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
                assert_gate=args.assert_gate,
            )
        else:
            result = replay_artifact(
                args.artifact,
                assert_gate=args.assert_gate,
            )
    except (GateEvidenceError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        parser = _build_parser()
        parser.error(str(error))
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
