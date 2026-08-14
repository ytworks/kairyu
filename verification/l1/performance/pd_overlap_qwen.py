"""Formal Qwen3-32B blocking/deferred P-D handoff evidence for issue #223.

This harness exercises the production ``KairyuBackend`` twice on the same
cross-device prefill/decode pair:

* ``blocking`` waits on every KV transfer before publication; and
* ``deferred`` records transfer completion and lets the next prefill step
  overlap it before publication.

Both conditions receive the same fixed natural-language prompts, greedy
sampling parameters, request IDs, and round order.  Passing requires the real
Qwen3-32B checkpoint, at least two distinct CUDA GPUs, the production
``StreamCopyKVHandoff`` topology, complete output records, and exact token/text
parity between the two conditions.  The same artifact also binds a
Qwen3-32B-layout P2P probe: dedicated source/destination streams, CUDA-event
interval overlap for deferred transfer (and serialization for the blocking
control), publication ordering, and exact source/blocking/deferred raw KV-byte
digests.  There is no synthetic or skipped mode.

Wall time, TTFT, completion latency, and throughput are recorded as diagnostic
before/after evidence.  They are deliberately non-binding: a short run on a
shared host cannot turn OS/runtime jitter into a trustworthy regression gate.

Examples:

    uv run python verification/l1/performance/pd_overlap_qwen.py \
      --model-path /var/lib/docker/volumes/kairyu-qwen3-32b_qwen3-32b/_data/qwen3-32b \
      --prefill-device 0 --decode-device 1 \
      --output bench/results/issue-223-pd-overlap-qwen3-32b.json \
      --assert-gate

    uv run python verification/l1/performance/pd_overlap_qwen.py \
      --verify bench/results/issue-223-pd-overlap-qwen3-32b.json \
      --assert-gate
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = 1
MEASUREMENT_KIND = "real-qwen3-32b-cross-device-pd"
CONDITIONS = ("blocking", "deferred")
DEFAULT_ROUNDS = 2
MINIMUM_VISIBLE_GPUS = 2
DEFAULT_NUM_PAGES = 256
DEFAULT_PAGE_SIZE = 16
MAX_NUM_BATCHED_TOKENS = 512
MAX_NUM_SEQS = 4
MAX_NEW_TOKENS = 16
WARMUP_NEW_TOKENS = 2
P2P_PROBE_NUM_LAYERS = 64
P2P_PROBE_NUM_PAGES = 32
P2P_PROBE_PAGE_SIZE = 16
P2P_PROBE_NUM_KV_HEADS = 8
P2P_PROBE_HEAD_DIM = 128
P2P_PROBE_COPY_SLEEP_CYCLES = 1_000_000_000
P2P_PROBE_ROLE_WORK_SLEEP_CYCLES = 500_000_000
P2P_PROBE_RAW_BYTES = (
    P2P_PROBE_NUM_LAYERS
    * P2P_PROBE_NUM_PAGES
    * P2P_PROBE_PAGE_SIZE
    * P2P_PROBE_NUM_KV_HEADS
    * P2P_PROBE_HEAD_DIM
    * 2  # bfloat16
    * 2  # K and V
)
PERFORMANCE_POLICY = (
    "diagnostic-only: no latency or throughput threshold is binding because "
    "OS, thermal, clock, and shared-host jitter are not controlled"
)
SOURCE_PATH = "verification/l1/performance/pd_overlap_qwen.py"
_KAIRYU_SOURCE_PATHS = tuple(
    path.relative_to(_REPO_ROOT).as_posix()
    for path in sorted((_REPO_ROOT / "kairyu").rglob("*.py"))
)
SOURCE_BOUND_PATHS = (
    SOURCE_PATH,
    "verification/l1/correctness/parity_tp.py",
    "pyproject.toml",
    "uv.lock",
    *_KAIRYU_SOURCE_PATHS,
)
# Bound to the fixed request matrix after tokenization by the reviewed
# Qwen3-32B tokenizer.  Filled from that checkpoint, not from a toy tokenizer.
EXPECTED_PROMPT_TOKEN_IDS_SHA256 = (
    "9f5644941f40c1a02c68b1ed601809dc6c49c1264ed785ae127aa0097ec82dea"
)


def _ensure_python_bin_on_path() -> None:
    """Expose build helpers installed beside the active Python executable."""
    path = os.environ.get("PATH", "")
    if shutil.which("ninja", path=path) is not None:
        return
    python_bin = Path(sys.executable).parent
    ninja = python_bin / "ninja"
    if os.access(ninja, os.X_OK):
        os.environ["PATH"] = os.pathsep.join(
            part for part in (str(python_bin), path) if part
        )


@dataclass(frozen=True)
class _RequestCase:
    case_id: str
    paragraph: str
    suffix: str
    repetitions: int
    seed: int


_REQUEST_CASES = (
    _RequestCase(
        case_id="compiler",
        paragraph=(
            "A compiler transforms source code through parsing, semantic analysis, "
            "optimization, and code generation while preserving the observable "
            "behavior of the program. "
        ),
        suffix="Explain which invariants make each transformation safe.",
        repetitions=16,
        seed=223_001,
    ),
    _RequestCase(
        case_id="database",
        paragraph=(
            "A database query planner compares access paths, join orders, "
            "cardinality estimates, and memory costs before selecting an execution "
            "strategy. "
        ),
        suffix="Describe how estimation errors can affect tail latency.",
        repetitions=16,
        seed=223_002,
    ),
    _RequestCase(
        case_id="network",
        paragraph=(
            "A distributed service uses retries, deadlines, idempotency keys, "
            "backpressure, and tracing to remain understandable during partial "
            "failures. "
        ),
        suffix="Give a concise failure analysis for an overloaded dependency.",
        repetitions=16,
        seed=223_003,
    ),
    _RequestCase(
        case_id="numerics",
        paragraph=(
            "Numerical software balances stability, precision, vectorization, "
            "memory locality, and reproducibility when implementing iterative "
            "algorithms. "
        ),
        suffix="Explain how you would validate a parallel reduction.",
        repetitions=16,
        seed=223_004,
    ),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode())


def _canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _checkpoint_weights_sha256(weights: object) -> str:
    """Match ``verification.l1.correctness.parity_tp._checkpoint_provenance`` byte-for-byte."""
    return _sha256_text(json.dumps(weights, sort_keys=True))


def _git_show(commit: str, relative: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_provenance() -> dict[str, Any]:
    from verification.l1.correctness.parity_tp import _code_provenance

    code = _code_provenance()
    current = Path(__file__).read_bytes()
    committed = (
        _git_show(str(code["commit"]), SOURCE_PATH)
        if code.get("commit") is not None
        else None
    )
    commit = code.get("commit")
    bound_file_sha256: dict[str, str] = {}
    bound_files_match_commit = isinstance(commit, str)
    if isinstance(commit, str):
        for relative in SOURCE_BOUND_PATHS:
            committed_blob = _git_show(commit, relative)
            working_path = _REPO_ROOT / relative
            if committed_blob is None or not working_path.is_file():
                bound_files_match_commit = False
                continue
            working_blob = working_path.read_bytes()
            bound_file_sha256[relative] = _sha256_bytes(committed_blob)
            bound_files_match_commit &= working_blob == committed_blob
    return {
        **code,
        "script_path": SOURCE_PATH,
        "script_sha256": _sha256_bytes(current),
        "committed_script_sha256": (
            _sha256_bytes(committed) if committed is not None else None
        ),
        "script_matches_commit": committed == current,
        "bound_file_sha256": bound_file_sha256,
        "bound_files_match_commit": (
            bound_files_match_commit
            and set(bound_file_sha256) == set(SOURCE_BOUND_PATHS)
        ),
    }


def _source_snapshot_is_clean_and_bound(source: dict[str, Any]) -> bool:
    commit = source.get("commit")
    recorded = source.get("bound_file_sha256")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or source.get("source") not in {"git", "environment"}
        or source.get("dirty") is not False
        or source.get("script_path") != SOURCE_PATH
        or source.get("script_matches_commit") is not True
        or not isinstance(recorded, dict)
        or set(recorded) != set(SOURCE_BOUND_PATHS)
        or source.get("bound_files_match_commit") is not True
    ):
        return False
    for relative in SOURCE_BOUND_PATHS:
        committed = _git_show(commit, relative)
        working_path = _REPO_ROOT / relative
        if committed is None or not working_path.is_file():
            return False
        current = working_path.read_bytes()
        committed_sha256 = _sha256_bytes(committed)
        if recorded.get(relative) != committed_sha256 or current != committed:
            return False
    script_sha256 = recorded[SOURCE_PATH]
    return (
        source.get("script_sha256") == script_sha256
        and source.get("committed_script_sha256") == script_sha256
    )


def _source_is_clean_and_bound(source: dict[str, Any]) -> bool:
    end = source.get("measurement_end")
    if not isinstance(end, dict):
        return False
    start = {
        key: value
        for key, value in source.items()
        if key not in {"measurement_end", "stable_across_measurement"}
    }
    return (
        source.get("stable_across_measurement") is True
        and start == end
        and _source_snapshot_is_clean_and_bound(start)
        and _source_snapshot_is_clean_and_bound(end)
    )


def _hardware_provenance() -> dict[str, Any]:
    import torch

    from kairyu.engine.core.hw_profile import probe
    from verification.l1.correctness.parity_tp import _gpu_runtime_provenance

    profile = probe()
    visible = torch.cuda.device_count()
    devices = []
    for index in range(visible):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "uuid": str(properties.uuid),
                "pci": (
                    f"{properties.pci_domain_id:04x}:"
                    f"{properties.pci_bus_id:02x}:"
                    f"{properties.pci_device_id:02x}"
                ),
                "total_memory_bytes": properties.total_memory,
            }
        )
    peer_access_matrix = [
        [
            (
                None
                if source == destination
                else bool(
                    torch.cuda.can_device_access_peer(source, destination)
                )
            )
            for destination in range(visible)
        ]
        for source in range(visible)
    ]
    return {
        "arch": profile.arch,
        "profile_device_name": profile.device_name,
        "profile_device_count": profile.device_count,
        "profile_sm": profile.sm,
        "cuda_available": torch.cuda.is_available(),
        "visible_device_count": visible,
        "devices": devices,
        "peer_access_matrix": peer_access_matrix,
        "runtime": _gpu_runtime_provenance(),
    }


def _hardware_is_real_selected_pair(
    hardware: dict[str, Any], selected: dict[str, Any]
) -> bool:
    devices = hardware.get("devices")
    peer_access_matrix = hardware.get("peer_access_matrix")
    runtime = hardware.get("runtime")
    visible = hardware.get("visible_device_count")
    if (
        not isinstance(devices, list)
        or not isinstance(runtime, dict)
        or not isinstance(visible, int)
        or visible < MINIMUM_VISIBLE_GPUS
        or not isinstance(peer_access_matrix, list)
        or len(peer_access_matrix) != visible
        or any(
            not isinstance(row, list) or len(row) != visible
            for row in peer_access_matrix
        )
        or set(selected) != {"prefill", "decode"}
    ):
        return False
    prefill = selected.get("prefill")
    decode = selected.get("decode")
    if (
        not isinstance(prefill, int)
        or isinstance(prefill, bool)
        or not isinstance(decode, int)
        or isinstance(decode, bool)
        or prefill == decode
        or prefill not in range(visible)
        or decode not in range(visible)
    ):
        return False
    capabilities = runtime.get("device_capabilities")
    nvidia_smi = runtime.get("nvidia_smi")
    topology = runtime.get("nvidia_smi_topology")
    return (
        hardware.get("arch") == "cuda"
        and hardware.get("cuda_available") is True
        and hardware.get("profile_device_count") == visible
        and len(devices) == visible
        and [device.get("index") for device in devices] == list(range(visible))
        and len({device.get("uuid") for device in devices}) == visible
        and len({device.get("pci") for device in devices}) == visible
        and all(
            peer_access_matrix[source][destination] is None
            if source == destination
            else isinstance(
                peer_access_matrix[source][destination], bool
            )
            for source in range(visible)
            for destination in range(visible)
        )
        and peer_access_matrix[prefill][decode] is True
        and peer_access_matrix[decode][prefill] is True
        and all(
            isinstance(device.get("name"), str)
            and bool(device["name"])
            and isinstance(device.get("capability"), list)
            and len(device["capability"]) == 2
            and isinstance(device.get("uuid"), str)
            and bool(device["uuid"])
            and isinstance(device.get("pci"), str)
            and bool(device["pci"])
            and isinstance(device.get("total_memory_bytes"), int)
            and device["total_memory_bytes"] > 0
            for device in devices
        )
        and isinstance(runtime.get("torch_cuda"), str)
        and bool(runtime["torch_cuda"])
        and isinstance(capabilities, list)
        and len(capabilities) == visible
        and isinstance(nvidia_smi, list)
        and len(nvidia_smi) >= visible
        and isinstance(topology, list)
        and bool(topology)
    )


def _checkpoint_snapshot_is_qwen3_32b(
    checkpoint: dict[str, Any],
) -> bool:
    architecture = checkpoint.get("architecture")
    weights = checkpoint.get("weights")
    if not isinstance(architecture, dict) or not isinstance(weights, list):
        return False
    expected_architecture = {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
    }
    if any(
        architecture.get(name) != value
        for name, value in expected_architecture.items()
    ):
        return False
    if not weights:
        return False
    filenames: set[str] = set()
    for row in weights:
        if not isinstance(row, dict):
            return False
        filename = row.get("file")
        digest = row.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or filename in filenames
            or not isinstance(row.get("bytes"), int)
            or isinstance(row["bytes"], bool)
            or row["bytes"] <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
        filenames.add(filename)
    rollup = _checkpoint_weights_sha256(weights)
    metadata = checkpoint.get("metadata_sha256")
    tokenizer = checkpoint.get("tokenizer_sha256")
    return (
        checkpoint.get("weights_sha256") == rollup
        and isinstance(metadata, dict)
        and isinstance(metadata.get("config.json"), str)
        and len(metadata["config.json"]) == 64
        and isinstance(tokenizer, dict)
        and bool(tokenizer)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (*metadata.values(), *tokenizer.values())
        )
    )


def _checkpoint_is_qwen3_32b(checkpoint: dict[str, Any]) -> bool:
    end = checkpoint.get("measurement_end")
    if not isinstance(end, dict):
        return False
    start = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"measurement_end", "stable_across_measurement"}
    }
    return (
        checkpoint.get("stable_across_measurement") is True
        and start == end
        and _checkpoint_snapshot_is_qwen3_32b(start)
        and _checkpoint_snapshot_is_qwen3_32b(end)
    )


def _sampling_config(case: _RequestCase, round_index: int) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_k": -1,
        "top_p": 1.0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": case.seed + round_index,
        "max_tokens": MAX_NEW_TOKENS,
        "ignore_eos": True,
    }


def _expected_request_config(rounds: int = DEFAULT_ROUNDS) -> list[dict[str, Any]]:
    requests = []
    for round_index in range(rounds):
        for case in _REQUEST_CASES:
            text = (
                f"Issue 223 evidence round {round_index}, case {case.case_id}. "
                f"{case.paragraph * case.repetitions}{case.suffix}"
            )
            requests.append(
                {
                    "round": round_index,
                    "case_id": case.case_id,
                    "request_id": (
                        f"issue223-r{round_index:02d}-{case.case_id}"
                    ),
                    "text": text,
                    "sampling": _sampling_config(case, round_index),
                }
            )
    return requests


def _prompt_token_ids_sha256(requests: list[dict[str, Any]]) -> str:
    payload = [
        {
            "round": request["round"],
            "case_id": request["case_id"],
            "request_id": request["request_id"],
            "text": request["text"],
            "prompt_token_ids": request["prompt_token_ids"],
        }
        for request in requests
    ]
    return _canonical_json_sha256(payload)


def _build_request_evidence(model_path: str) -> list[dict[str, Any]]:
    from kairyu.engine.tokenizer import resolve_tokenizer

    tokenizer = resolve_tokenizer(model_path)
    requests = []
    for expected in _expected_request_config():
        token_ids = list(tokenizer.encode(expected["text"]))
        if not 1 <= len(token_ids) <= MAX_NUM_BATCHED_TOKENS:
            raise RuntimeError(
                f"{expected['request_id']} has {len(token_ids)} prompt tokens; "
                f"formal range is 1..{MAX_NUM_BATCHED_TOKENS}"
            )
        requests.append(
            {
                **expected,
                "prompt_token_ids": token_ids,
                "prompt_tokens": len(token_ids),
                "text_sha256": _sha256_text(expected["text"]),
                "prompt_token_ids_sha256": _canonical_json_sha256(token_ids),
            }
        )
    digest = _prompt_token_ids_sha256(requests)
    if digest != EXPECTED_PROMPT_TOKEN_IDS_SHA256:
        raise RuntimeError(
            "fixed prompt tokenization does not match the reviewed Qwen3-32B "
            f"request matrix: {digest}"
        )
    return requests


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _close_float(actual: object, expected: float) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
    )


def _request_shape_ok(
    requests: object,
    *,
    rounds: int,
    vocab_size: int,
) -> bool:
    if not isinstance(requests, list):
        return False
    expected = _expected_request_config(rounds)
    if len(requests) != len(expected):
        return False
    for actual, fixed in zip(requests, expected, strict=True):
        if not isinstance(actual, dict):
            return False
        if any(actual.get(key) != value for key, value in fixed.items()):
            return False
        token_ids = actual.get("prompt_token_ids")
        text = actual.get("text")
        if (
            not isinstance(token_ids, list)
            or not 1 <= len(token_ids) <= MAX_NUM_BATCHED_TOKENS
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < vocab_size
                for token_id in token_ids
            )
            or actual.get("prompt_tokens") != len(token_ids)
            or not isinstance(text, str)
            or actual.get("text_sha256") != _sha256_text(text)
            or actual.get("prompt_token_ids_sha256")
            != _canonical_json_sha256(token_ids)
        ):
            return False
    return _prompt_token_ids_sha256(requests) == EXPECTED_PROMPT_TOKEN_IDS_SHA256


def _expected_topology(
    condition: str, selected: dict[str, int]
) -> dict[str, Any]:
    prefill = f"cuda:{selected['prefill']}"
    decode = f"cuda:{selected['decode']}"
    return {
        "backend_class": "KairyuBackend",
        "coordinator_class": "PDCoordinator",
        "adapter_class": "PDLoopAdapter",
        "prefill_runner_class": "PagedModelRunner",
        "decode_runner_class": "PagedModelRunner",
        "handoff_class": "StreamCopyKVHandoff",
        "inner_handoff_class": "LocalCopyKVHandoff",
        "stream_provider_class": "CudaStreamProvider",
        "prefill_runner_device": prefill,
        "decode_runner_device": decode,
        "source_pool_device": prefill,
        "destination_pool_device": decode,
        "provider_source_device": prefill,
        "provider_transfer_device": decode,
        "provider_source_copy_stream_device": prefill,
        "provider_destination_dependency_stream_device": decode,
        "provider_gate_devices": [prefill, decode],
        "provider_source_copy_stream_is_compute_stream": False,
        "provider_destination_dependency_stream_is_compute_stream": False,
        "provider_copy_streams_are_distinct": True,
        "defers": condition == "deferred",
        "inner_defer_publish": True,
        "cross_device": True,
    }


def _capture_topology(backend: object) -> dict[str, Any]:
    loop = getattr(backend, "_loop", None)
    coordinator = getattr(loop, "pd_coordinator", None)
    adapter = getattr(backend, "_scheduler", None)
    handoff = getattr(coordinator, "_handoff", None)
    inner = getattr(handoff, "_inner", None)
    provider = getattr(handoff, "_provider", None)
    prefill_runner = getattr(coordinator, "_prefill_runner", None)
    decode_runner = getattr(coordinator, "_decode_runner", None)
    source_pool = getattr(inner, "_source_pool", None)
    destination_pool = getattr(inner, "_dest_pool", None)
    stream = getattr(provider, "_stream", None)
    source_stream = getattr(provider, "_source_stream", None)
    gate_devices = getattr(provider, "_gate_devices", ())
    required = (
        coordinator,
        adapter,
        handoff,
        inner,
        provider,
        prefill_runner,
        decode_runner,
        source_pool,
        destination_pool,
        stream,
        source_stream,
    )
    if any(item is None for item in required):
        raise RuntimeError("production P-D backend does not expose the expected topology")
    import torch

    source_compute_stream = torch.cuda.current_stream(
        device=provider._source_device
    )
    destination_compute_stream = torch.cuda.current_stream(
        device=stream.device
    )
    return {
        "backend_class": type(backend).__name__,
        "coordinator_class": type(coordinator).__name__,
        "adapter_class": type(adapter).__name__,
        "prefill_runner_class": type(prefill_runner).__name__,
        "decode_runner_class": type(decode_runner).__name__,
        "handoff_class": type(handoff).__name__,
        "inner_handoff_class": type(inner).__name__,
        "stream_provider_class": type(provider).__name__,
        "prefill_runner_device": str(getattr(prefill_runner, "_device", None)),
        "decode_runner_device": str(getattr(decode_runner, "_device", None)),
        "source_pool_device": str(source_pool.k.device),
        "destination_pool_device": str(destination_pool.k.device),
        "provider_source_device": str(
            getattr(provider, "_source_device", None)
        ),
        "provider_transfer_device": str(stream.device),
        "provider_source_copy_stream_device": str(source_stream.device),
        "provider_destination_dependency_stream_device": str(stream.device),
        "provider_gate_devices": [str(device) for device in gate_devices],
        "provider_source_copy_stream_is_compute_stream": (
            source_stream == source_compute_stream
        ),
        "provider_destination_dependency_stream_is_compute_stream": (
            stream == destination_compute_stream
        ),
        "provider_copy_streams_are_distinct": source_stream != stream,
        "defers": getattr(handoff, "defers", None),
        "inner_defer_publish": getattr(inner, "_defer_publish", None),
        "cross_device": source_pool.k.device != destination_pool.k.device,
    }


def _synchronize_devices(selected: dict[str, int]) -> None:
    import torch

    for device in (selected["prefill"], selected["decode"]):
        torch.cuda.synchronize(device)


def _raw_kv_digest(pool: object, pages: tuple[int, ...]) -> dict[str, Any]:
    """Hash logical K then V bytes in allocation-page order."""
    import torch

    page_index = torch.tensor(
        pages,
        dtype=torch.long,
        device=pool.k.device,
    )
    digest = hashlib.sha256()
    raw_bytes = 0
    for tensor in (pool.k, pool.v):
        logical = tensor.index_select(1, page_index).contiguous()
        raw = logical.view(torch.uint8).cpu()
        digest.update(memoryview(raw.numpy()))
        raw_bytes += raw.numel()
    return {"sha256": digest.hexdigest(), "bytes": raw_bytes}


def _event_interval_ms(
    base: object,
    start: object,
    end: object,
) -> list[float]:
    return [
        float(base.elapsed_time(start)),
        float(base.elapsed_time(end)),
    ]


def _intervals_overlap(left: list[float], right: list[float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _stream_record(stream: object) -> dict[str, Any]:
    return {
        "device": str(stream.device),
        "cuda_stream": int(stream.cuda_stream),
    }


def _probe_stream_topology(
    provider: object,
    selected: dict[str, int],
) -> dict[str, Any]:
    import torch

    source_copy = provider._source_stream
    destination_dependency = provider._stream
    source_compute = torch.cuda.current_stream(selected["prefill"])
    destination_compute = torch.cuda.current_stream(selected["decode"])
    return {
        "source_copy_stream": _stream_record(source_copy),
        "source_compute_stream": _stream_record(source_compute),
        "destination_dependency_stream": _stream_record(
            destination_dependency
        ),
        "destination_compute_stream": _stream_record(destination_compute),
        "gate_devices": [
            str(device) for device in provider._gate_devices
        ],
        "source_copy_stream_is_compute_stream": source_copy == source_compute,
        "destination_dependency_stream_is_compute_stream": (
            destination_dependency == destination_compute
        ),
        "copy_streams_are_distinct": (
            source_copy != destination_dependency
        ),
    }


def _run_p2p_probe_condition(
    *,
    condition: str,
    selected: dict[str, int],
    tokens: tuple[int, ...],
    source_pages: tuple[int, ...],
    source_pool: object,
) -> dict[str, Any]:
    import torch

    from kairyu.engine.core.handoff_stream import (
        CudaStreamProvider,
        StreamCopyKVHandoff,
    )
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd import LocalCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    class _TimedP2PProvider(CudaStreamProvider):
        def __init__(self) -> None:
            super().__init__(
                device=f"cuda:{selected['prefill']}",
                transfer_device=f"cuda:{selected['decode']}",
                gate_devices=(
                    f"cuda:{selected['prefill']}",
                    f"cuda:{selected['decode']}",
                ),
                require_peer_access=True,
            )
            self.source_start = torch.cuda.Event(enable_timing=True)
            self.source_done = torch.cuda.Event(enable_timing=True)
            self.destination_start = torch.cuda.Event(enable_timing=True)
            self.destination_done = torch.cuda.Event(enable_timing=True)

        def begin(self) -> None:
            super().begin()
            self.source_start.record(self._source_stream)
            self.destination_start.record(self._stream)
            # A long, low-occupancy device wait makes ordering visible to CUDA
            # events without turning elapsed wall time into a pass criterion.
            with torch.cuda.device(self._source_device):
                torch.cuda._sleep(P2P_PROBE_COPY_SLEEP_CYCLES)

        def _record_tails(self) -> None:
            self.source_done.record(self._source_stream)
            if self._source_stream is not self._stream:
                self.source_done.wait(self._stream)
            self.destination_done.record(self._stream)

        def synchronize(self) -> None:
            self._record_tails()
            super().synchronize()

        def record(self) -> object:
            self.source_done.record(self._source_stream)
            event = super().record()
            self.destination_done.record(self._stream)
            return event

    destination_pool = PagedKVPool(
        num_layers=P2P_PROBE_NUM_LAYERS,
        num_pages=P2P_PROBE_NUM_PAGES,
        page_size=P2P_PROBE_PAGE_SIZE,
        num_kv_heads=P2P_PROBE_NUM_KV_HEADS,
        head_dim=P2P_PROBE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=f"cuda:{selected['decode']}",
    )
    publication_events: list[dict[str, Any]] = []
    destination_cache = RadixKVCache(
        num_pages=P2P_PROBE_NUM_PAGES,
        page_size=P2P_PROBE_PAGE_SIZE,
        event_sink=publication_events.append,
    )
    provider = _TimedP2PProvider()
    topology = _probe_stream_topology(provider, selected)
    handoff = StreamCopyKVHandoff(
        LocalCopyKVHandoff(
            destination_cache,
            source_pool,
            destination_pool,
        ),
        provider,
        defer=condition == "deferred",
    )

    source_base = torch.cuda.Event(enable_timing=True)
    destination_base = torch.cuda.Event(enable_timing=True)
    source_base.record(torch.cuda.current_stream(selected["prefill"]))
    destination_base.record(torch.cuda.current_stream(selected["decode"]))
    source_base.synchronize()
    destination_base.synchronize()

    allocation = handoff.transfer(
        tokens,
        first_token=tokens[-1],
        pages=source_pages,
    )
    if condition == "deferred":
        completion = handoff.completion_for(allocation)
        completion_ready_at_return = handoff.completion_ready(completion)
        completion_token_created = True
    else:
        completion = None
        completion_ready_at_return = bool(
            provider.source_done.query()
            and provider.destination_done.query()
        )
        completion_token_created = False
    publication_at_return = list(publication_events)
    pending_at_return = len(handoff.pending_events)

    source_work_start = torch.cuda.Event(enable_timing=True)
    source_work_done = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(selected["prefill"]):
        source_work_start.record()
        torch.cuda._sleep(P2P_PROBE_ROLE_WORK_SLEEP_CYCLES)
        source_work_done.record()
    destination_work_start = torch.cuda.Event(enable_timing=True)
    destination_work_done = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(selected["decode"]):
        destination_work_start.record()
        torch.cuda._sleep(P2P_PROBE_ROLE_WORK_SLEEP_CYCLES)
        destination_work_done.record()

    if completion is not None:
        deadline = time.perf_counter() + 15.0
        while not handoff.completion_ready(completion):
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"{condition} P2P completion did not become ready"
                )
            time.sleep(0)
        if not handoff.finalize_completion(completion):
            raise RuntimeError(
                f"{condition} P2P completion became unready before finalize"
            )
        handoff.acknowledge_completion(completion)
    _synchronize_devices(selected)

    source_copy = _event_interval_ms(
        source_base,
        provider.source_start,
        provider.source_done,
    )
    destination_dependency = _event_interval_ms(
        destination_base,
        provider.destination_start,
        provider.destination_done,
    )
    source_role_work = _event_interval_ms(
        source_base,
        source_work_start,
        source_work_done,
    )
    destination_role_work = _event_interval_ms(
        destination_base,
        destination_work_start,
        destination_work_done,
    )
    destination_digest = _raw_kv_digest(
        destination_pool,
        tuple(allocation.pages),
    )
    return {
        "condition": condition,
        "status": "complete",
        "topology": topology,
        "completion_token_created": completion_token_created,
        "completion_ready_at_return": completion_ready_at_return,
        "pending_completions_at_return": pending_at_return,
        "pending_completions_final": len(handoff.pending_events),
        "settled_completion_outcomes_final": len(handoff._settled),
        "publication_events_at_return": publication_at_return,
        "publication_events_final": list(publication_events),
        "source_copy_interval_ms": source_copy,
        "destination_dependency_interval_ms": destination_dependency,
        "source_role_work_interval_ms": source_role_work,
        "destination_role_work_interval_ms": destination_role_work,
        "source_role_overlap": _intervals_overlap(
            source_copy,
            source_role_work,
        ),
        "destination_role_overlap": _intervals_overlap(
            destination_dependency,
            destination_role_work,
        ),
        "destination_kv": destination_digest,
    }


def _run_p2p_probe(
    selected: dict[str, int],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.radix_kv import RadixKVCache

    matrix = hardware["peer_access_matrix"]
    if (
        matrix[selected["prefill"]][selected["decode"]] is not True
        or matrix[selected["decode"]][selected["prefill"]] is not True
    ):
        raise RuntimeError(
            "formal P2P evidence requires peer access in both selected directions"
        )

    # Exclude one-time CUDA peer setup from both formal event timelines.
    source_warmup = torch.ones(
        1,
        device=f"cuda:{selected['prefill']}",
    )
    destination_warmup = torch.empty(
        1,
        device=f"cuda:{selected['decode']}",
    )
    destination_warmup.copy_(source_warmup, non_blocking=True)
    _synchronize_devices(selected)

    source_pool = PagedKVPool(
        num_layers=P2P_PROBE_NUM_LAYERS,
        num_pages=P2P_PROBE_NUM_PAGES,
        page_size=P2P_PROBE_PAGE_SIZE,
        num_kv_heads=P2P_PROBE_NUM_KV_HEADS,
        head_dim=P2P_PROBE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=f"cuda:{selected['prefill']}",
    )
    source_cache = RadixKVCache(
        num_pages=P2P_PROBE_NUM_PAGES,
        page_size=P2P_PROBE_PAGE_SIZE,
    )
    tokens = tuple(
        range(1, P2P_PROBE_NUM_PAGES * P2P_PROBE_PAGE_SIZE + 1)
    )
    source_allocation = source_cache.allocate(tokens)
    source_cache.mark_computed(source_allocation)
    generator = torch.Generator(
        device=f"cuda:{selected['prefill']}"
    ).manual_seed(223)
    source_pool.k.uniform_(-1.0, 1.0, generator=generator)
    source_pool.v.uniform_(-1.0, 1.0, generator=generator)
    _synchronize_devices(selected)
    source_before = _raw_kv_digest(
        source_pool,
        tuple(source_allocation.pages),
    )

    conditions = {}
    for condition in CONDITIONS:
        conditions[condition] = _run_p2p_probe_condition(
            condition=condition,
            selected=selected,
            tokens=tokens,
            source_pages=tuple(source_allocation.pages),
            source_pool=source_pool,
        )
        gc.collect()
        with torch.cuda.device(selected["decode"]):
            torch.cuda.empty_cache()
    source_after = _raw_kv_digest(
        source_pool,
        tuple(source_allocation.pages),
    )
    probe = {
        "measurement_kind": "real-qwen3-32b-layout-p2p-event-probe",
        "selected_devices": dict(selected),
        "peer_access": {
            "prefill_to_decode": (
                matrix[selected["prefill"]][selected["decode"]]
            ),
            "decode_to_prefill": (
                matrix[selected["decode"]][selected["prefill"]]
            ),
        },
        "geometry": {
            "num_layers": P2P_PROBE_NUM_LAYERS,
            "num_pages": P2P_PROBE_NUM_PAGES,
            "page_size": P2P_PROBE_PAGE_SIZE,
            "num_kv_heads": P2P_PROBE_NUM_KV_HEADS,
            "head_dim": P2P_PROBE_HEAD_DIM,
            "dtype": "torch.bfloat16",
            "logical_tokens": len(tokens),
            "raw_bytes": P2P_PROBE_RAW_BYTES,
            "digest_order": "K then V, allocation page order",
            "token_ids_sha256": _canonical_json_sha256(tokens),
        },
        "event_policy": (
            "binding source-copy and destination-dependency CUDA-event "
            "interval intersection; no elapsed-time or throughput threshold"
        ),
        "source_kv_before": source_before,
        "source_kv_after": source_after,
        "conditions": conditions,
    }
    del source_pool, source_cache, source_allocation
    gc.collect()
    for device in (selected["prefill"], selected["decode"]):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return probe


def _expected_p2p_probe_config() -> dict[str, Any]:
    return {
        "num_layers": P2P_PROBE_NUM_LAYERS,
        "num_pages": P2P_PROBE_NUM_PAGES,
        "page_size": P2P_PROBE_PAGE_SIZE,
        "num_kv_heads": P2P_PROBE_NUM_KV_HEADS,
        "head_dim": P2P_PROBE_HEAD_DIM,
        "dtype": "torch.bfloat16",
        "copy_sleep_cycles": P2P_PROBE_COPY_SLEEP_CYCLES,
        "role_work_sleep_cycles": P2P_PROBE_ROLE_WORK_SLEEP_CYCLES,
        "raw_bytes": P2P_PROBE_RAW_BYTES,
        "event_overlap_binding": True,
        "raw_kv_digest_binding": True,
        "performance_threshold_binding": False,
    }


def _valid_interval(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and _finite_nonnegative(value[0])
        and _finite_positive(value[1])
        and value[0] < value[1]
    )


def _valid_digest(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    digest = value.get("sha256")
    return (
        value.get("bytes") == P2P_PROBE_RAW_BYTES
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _probe_publication_event_ok(event: object) -> bool:
    if not isinstance(event, dict):
        return False
    hashes = event.get("block_hashes")
    return (
        event.get("type") == "BlockStored"
        and event.get("parent_block_hash") is None
        and event.get("block_size") == P2P_PROBE_PAGE_SIZE
        and event.get("token_ids")
        == list(
            range(
                1,
                P2P_PROBE_NUM_PAGES * P2P_PROBE_PAGE_SIZE + 1,
            )
        )
        and hashes == _expected_probe_block_hashes()
    )


def _expected_probe_block_hashes() -> list[str]:
    tokens = tuple(
        range(1, P2P_PROBE_NUM_PAGES * P2P_PROBE_PAGE_SIZE + 1)
    )
    return [
        _sha256_text(repr(tokens[:token_count]))[:16]
        for token_count in range(
            P2P_PROBE_PAGE_SIZE,
            len(tokens) + 1,
            P2P_PROBE_PAGE_SIZE,
        )
    ]


def _probe_topology_ok(
    topology: object,
    selected: dict[str, int],
) -> bool:
    if not isinstance(topology, dict):
        return False
    expected_devices = {
        "source_copy_stream": f"cuda:{selected['prefill']}",
        "source_compute_stream": f"cuda:{selected['prefill']}",
        "destination_dependency_stream": f"cuda:{selected['decode']}",
        "destination_compute_stream": f"cuda:{selected['decode']}",
    }
    identities: dict[str, tuple[str, int]] = {}
    for name, device in expected_devices.items():
        record = topology.get(name)
        if (
            not isinstance(record, dict)
            or record.get("device") != device
            or not isinstance(record.get("cuda_stream"), int)
            or isinstance(record["cuda_stream"], bool)
            or record["cuda_stream"] < 0
        ):
            return False
        identities[name] = (record["device"], record["cuda_stream"])
    return (
        topology.get("gate_devices")
        == [
            f"cuda:{selected['prefill']}",
            f"cuda:{selected['decode']}",
        ]
        and topology.get("source_copy_stream_is_compute_stream") is False
        and topology.get(
            "destination_dependency_stream_is_compute_stream"
        )
        is False
        and topology.get("copy_streams_are_distinct") is True
        and identities["source_copy_stream"]
        != identities["source_compute_stream"]
        and identities["destination_dependency_stream"]
        != identities["destination_compute_stream"]
        and identities["source_copy_stream"]
        != identities["destination_dependency_stream"]
    )


def _p2p_probe_ok(
    probe: object,
    selected: dict[str, int],
) -> bool:
    if (
        not isinstance(probe, dict)
        or not isinstance(selected, dict)
        or set(selected) != {"prefill", "decode"}
        or any(
            not isinstance(selected[role], int)
            or isinstance(selected[role], bool)
            for role in ("prefill", "decode")
        )
        or selected["prefill"] == selected["decode"]
    ):
        return False
    geometry = probe.get("geometry")
    peer_access = probe.get("peer_access")
    conditions = probe.get("conditions")
    source_before = probe.get("source_kv_before")
    source_after = probe.get("source_kv_after")
    expected_geometry = {
        "num_layers": P2P_PROBE_NUM_LAYERS,
        "num_pages": P2P_PROBE_NUM_PAGES,
        "page_size": P2P_PROBE_PAGE_SIZE,
        "num_kv_heads": P2P_PROBE_NUM_KV_HEADS,
        "head_dim": P2P_PROBE_HEAD_DIM,
        "dtype": "torch.bfloat16",
        "logical_tokens": P2P_PROBE_NUM_PAGES * P2P_PROBE_PAGE_SIZE,
        "raw_bytes": P2P_PROBE_RAW_BYTES,
        "digest_order": "K then V, allocation page order",
        "token_ids_sha256": _canonical_json_sha256(
            tuple(
                range(
                    1,
                    P2P_PROBE_NUM_PAGES * P2P_PROBE_PAGE_SIZE + 1,
                )
            )
        ),
    }
    if (
        probe.get("measurement_kind")
        != "real-qwen3-32b-layout-p2p-event-probe"
        or probe.get("selected_devices") != selected
        or geometry != expected_geometry
        or peer_access
        != {
            "prefill_to_decode": True,
            "decode_to_prefill": True,
        }
        or probe.get("event_policy")
        != (
            "binding source-copy and destination-dependency CUDA-event "
            "interval intersection; no elapsed-time or throughput threshold"
        )
        or not isinstance(conditions, dict)
        or set(conditions) != set(CONDITIONS)
        or not _valid_digest(source_before)
        or not _valid_digest(source_after)
        or source_before != source_after
    ):
        return False
    for condition in CONDITIONS:
        row = conditions.get(condition)
        if not isinstance(row, dict):
            return False
        intervals = {
            name: row.get(name)
            for name in (
                "source_copy_interval_ms",
                "destination_dependency_interval_ms",
                "source_role_work_interval_ms",
                "destination_role_work_interval_ms",
            )
        }
        if (
            row.get("condition") != condition
            or row.get("status") != "complete"
            or not _probe_topology_ok(row.get("topology"), selected)
            or not all(_valid_interval(value) for value in intervals.values())
            or not _valid_digest(row.get("destination_kv"))
            or row["destination_kv"] != source_before
            or row.get("pending_completions_final") != 0
            or row.get("settled_completion_outcomes_final") != 0
        ):
            return False
        source_overlap = _intervals_overlap(
            intervals["source_copy_interval_ms"],
            intervals["source_role_work_interval_ms"],
        )
        destination_overlap = _intervals_overlap(
            intervals["destination_dependency_interval_ms"],
            intervals["destination_role_work_interval_ms"],
        )
        source_ordered_after_transfer = (
            intervals["source_copy_interval_ms"][0]
            <= intervals["source_role_work_interval_ms"][0]
        )
        destination_ordered_after_transfer = (
            intervals["destination_dependency_interval_ms"][0]
            <= intervals["destination_role_work_interval_ms"][0]
        )
        expected_overlap = condition == "deferred"
        final_publication = row.get("publication_events_final")
        if (
            row.get("source_role_overlap") is not source_overlap
            or row.get("destination_role_overlap") is not destination_overlap
            or not source_ordered_after_transfer
            or not destination_ordered_after_transfer
            or source_overlap is not expected_overlap
            or destination_overlap is not expected_overlap
            or not isinstance(final_publication, list)
            or len(final_publication) != 1
            or not _probe_publication_event_ok(final_publication[0])
        ):
            return False
        if condition == "blocking":
            if (
                intervals["source_copy_interval_ms"][1]
                > intervals["source_role_work_interval_ms"][0]
                or intervals["destination_dependency_interval_ms"][1]
                > intervals["destination_role_work_interval_ms"][0]
                or row.get("completion_token_created") is not False
                or row.get("completion_ready_at_return") is not True
                or row.get("pending_completions_at_return") != 0
                or row.get("publication_events_at_return")
                != final_publication
            ):
                return False
        elif (
            row.get("completion_token_created") is not True
            or row.get("completion_ready_at_return") is not False
            or row.get("pending_completions_at_return") != 1
            or row.get("publication_events_at_return") != []
        ):
            return False
    return True


async def _capture_request(backend: object, request: object) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    final = None
    updates = 0
    async for result in backend.stream(request):
        updates += 1
        completion = result.completions[0] if result.completions else None
        if (
            first_token_ns is None
            and completion is not None
            and completion.token_ids
        ):
            first_token_ns = time.perf_counter_ns()
        final = result
    ended_ns = time.perf_counter_ns()
    if (
        final is None
        or not final.finished
        or len(final.completions) != 1
        or final.usage is None
        or first_token_ns is None
    ):
        raise RuntimeError(f"incomplete stream for {request.request_id}")
    completion = final.completions[0]
    token_ids = list(completion.token_ids)
    return {
        "request_id": request.request_id,
        "status": "complete",
        "finished": True,
        "stream_updates": updates,
        "started_ns": started_ns,
        "first_token_ns": first_token_ns,
        "ended_ns": ended_ns,
        "ttft_s": (first_token_ns - started_ns) / 1e9,
        "completion_latency_s": (ended_ns - started_ns) / 1e9,
        "token_ids": token_ids,
        "text": completion.text,
        "text_sha256": _sha256_text(completion.text),
        "finish_reason": completion.finish_reason,
        "usage": {
            "prompt_tokens": final.usage.prompt_tokens,
            "completion_tokens": final.usage.completion_tokens,
            "cached_tokens": final.usage.cached_tokens,
        },
    }


def _make_generation_request(evidence: dict[str, Any]) -> object:
    from kairyu.engine.backend import GenerationRequest
    from kairyu.sampling_params import SamplingParams

    return GenerationRequest(
        request_id=evidence["request_id"],
        prompt=evidence["text"],
        sampling_params=SamplingParams(**evidence["sampling"]),
    )


async def _run_batch(
    backend: object,
    *,
    condition: str,
    round_index: int,
    request_evidence: list[dict[str, Any]],
    selected: dict[str, int],
) -> dict[str, Any]:
    requests = [_make_generation_request(row) for row in request_evidence]
    _synchronize_devices(selected)
    batch_started_ns = time.perf_counter_ns()
    rows = await asyncio.gather(
        *(_capture_request(backend, request) for request in requests)
    )
    _synchronize_devices(selected)
    batch_ended_ns = time.perf_counter_ns()
    wall_s = (batch_ended_ns - batch_started_ns) / 1e9
    for row in rows:
        row["start_offset_s"] = (
            row.pop("started_ns") - batch_started_ns
        ) / 1e9
        row.pop("first_token_ns")
        row.pop("ended_ns")
    total_completion_tokens = sum(
        row["usage"]["completion_tokens"] for row in rows
    )
    return {
        "condition": condition,
        "round": round_index,
        "status": "complete",
        "request_ids": [request.request_id for request in requests],
        "wall_s": wall_s,
        "completed_requests_per_s": len(rows) / wall_s,
        "output_tokens_per_s": total_completion_tokens / wall_s,
        "total_completion_tokens": total_completion_tokens,
        "median_ttft_s": statistics.median(row["ttft_s"] for row in rows),
        "median_completion_latency_s": statistics.median(
            row["completion_latency_s"] for row in rows
        ),
        "requests": rows,
    }


async def _run_condition(
    *,
    condition: str,
    model_path: str,
    requests: list[dict[str, Any]],
    selected: dict[str, int],
) -> dict[str, Any]:
    import torch

    from kairyu.engine.kairyu_backend import KairyuBackend

    backend = None
    try:
        load_started = time.perf_counter_ns()
        backend = KairyuBackend(
            model_path=model_path,
            pd_separation=True,
            pd_prefill_device=f"cuda:{selected['prefill']}",
            pd_decode_device=f"cuda:{selected['decode']}",
            pd_defer_handoff=condition == "deferred",
            num_pages=DEFAULT_NUM_PAGES,
            page_size=DEFAULT_PAGE_SIZE,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
            pipeline_depth=1,
            decode_mode="eager",
        )
        _synchronize_devices(selected)
        load_s = (time.perf_counter_ns() - load_started) / 1e9
        topology = _capture_topology(backend)

        from kairyu.engine.backend import GenerationRequest
        from kairyu.sampling_params import SamplingParams

        warmup_started = time.perf_counter_ns()
        warmup = await _capture_request(
            backend,
            GenerationRequest(
                request_id=f"issue223-{condition}-warmup",
                prompt=(
                    "Warm the Qwen3-32B P-D serving path with a prompt that "
                    "shares no prefix with the measured requests."
                ),
                sampling_params=SamplingParams(
                    temperature=0.0,
                    seed=223_999,
                    max_tokens=WARMUP_NEW_TOKENS,
                    ignore_eos=True,
                ),
            ),
        )
        _synchronize_devices(selected)
        warmup["wall_s"] = (time.perf_counter_ns() - warmup_started) / 1e9
        for transient in ("started_ns", "first_token_ns", "ended_ns"):
            warmup.pop(transient)

        measured_rounds = []
        for round_index in range(DEFAULT_ROUNDS):
            rows = [
                row for row in requests if row["round"] == round_index
            ]
            measured_rounds.append(
                await _run_batch(
                    backend,
                    condition=condition,
                    round_index=round_index,
                    request_evidence=rows,
                    selected=selected,
                )
            )
        return {
            "condition": condition,
            "measurement_kind": MEASUREMENT_KIND,
            "status": "complete",
            "load_s_excluded_from_measurement": load_s,
            "topology": topology,
            "warmup": warmup,
            "rounds": measured_rounds,
        }
    finally:
        if backend is not None:
            await backend.shutdown()
        backend = None
        gc.collect()
        for device in (selected["prefill"], selected["decode"]):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def _run_request_ok(
    row: object,
    evidence: dict[str, Any],
    *,
    vocab_size: int,
    wall_s: float,
) -> bool:
    if not isinstance(row, dict):
        return False
    token_ids = row.get("token_ids")
    text = row.get("text")
    usage = row.get("usage")
    return (
        row.get("request_id") == evidence["request_id"]
        and row.get("status") == "complete"
        and row.get("finished") is True
        and "skipped" not in row
        and "error" not in row
        and isinstance(row.get("stream_updates"), int)
        and not isinstance(row["stream_updates"], bool)
        and row["stream_updates"] >= 1
        and _finite_nonnegative(row.get("start_offset_s"))
        and row["start_offset_s"] <= wall_s
        and _finite_positive(row.get("ttft_s"))
        and _finite_positive(row.get("completion_latency_s"))
        and row["ttft_s"] <= row["completion_latency_s"]
        and row["completion_latency_s"] <= wall_s
        and isinstance(token_ids, list)
        and len(token_ids) == MAX_NEW_TOKENS
        and all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id < vocab_size
            for token_id in token_ids
        )
        and isinstance(text, str)
        and bool(text)
        and row.get("text_sha256") == _sha256_text(text)
        and row.get("finish_reason") == "length"
        and isinstance(usage, dict)
        and usage.get("prompt_tokens") == evidence["prompt_tokens"]
        and usage.get("completion_tokens") == len(token_ids)
        and isinstance(usage.get("cached_tokens"), int)
        and not isinstance(usage["cached_tokens"], bool)
        and 0 <= usage["cached_tokens"] <= evidence["prompt_tokens"]
    )


def _run_shape_ok(
    run: object,
    *,
    condition: str,
    selected: dict[str, int],
    requests: list[dict[str, Any]],
    rounds: int,
    vocab_size: int,
) -> bool:
    if not isinstance(run, dict):
        return False
    topology = run.get("topology")
    warmup = run.get("warmup")
    measured = run.get("rounds")
    if (
        run.get("condition") != condition
        or run.get("measurement_kind") != MEASUREMENT_KIND
        or run.get("status") != "complete"
        or "skipped" in run
        or "error" in run
        or not _finite_positive(run.get("load_s_excluded_from_measurement"))
        or topology != _expected_topology(condition, selected)
        or not isinstance(warmup, dict)
        or warmup.get("status") != "complete"
        or warmup.get("finished") is not True
        or len(warmup.get("token_ids", ())) != WARMUP_NEW_TOKENS
        or not _finite_positive(warmup.get("wall_s"))
        or not isinstance(measured, list)
        or len(measured) != rounds
    ):
        return False
    for round_index, batch in enumerate(measured):
        evidence_rows = [
            row for row in requests if row["round"] == round_index
        ]
        if not isinstance(batch, dict):
            return False
        rows = batch.get("requests")
        wall_s = batch.get("wall_s")
        if (
            batch.get("condition") != condition
            or batch.get("round") != round_index
            or batch.get("status") != "complete"
            or "skipped" in batch
            or "error" in batch
            or batch.get("request_ids")
            != [row["request_id"] for row in evidence_rows]
            or not _finite_positive(wall_s)
            or not isinstance(rows, list)
            or len(rows) != len(evidence_rows)
            or any(
                not _run_request_ok(
                    row,
                    evidence,
                    vocab_size=vocab_size,
                    wall_s=float(wall_s),
                )
                for row, evidence in zip(rows, evidence_rows, strict=True)
            )
        ):
            return False
        completion_tokens = sum(
            row["usage"]["completion_tokens"] for row in rows
        )
        if (
            batch.get("total_completion_tokens") != completion_tokens
            or not _close_float(
                batch.get("completed_requests_per_s"),
                len(rows) / float(wall_s),
            )
            or not _close_float(
                batch.get("output_tokens_per_s"),
                completion_tokens / float(wall_s),
            )
            or not _close_float(
                batch.get("median_ttft_s"),
                statistics.median(row["ttft_s"] for row in rows),
            )
            or not _close_float(
                batch.get("median_completion_latency_s"),
                statistics.median(
                    row["completion_latency_s"] for row in rows
                ),
            )
        ):
            return False
    return True


def _output_parity(
    runs: dict[str, Any], requests: list[dict[str, Any]]
) -> tuple[bool, int, list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    compared = 0
    try:
        blocking_rounds = runs["blocking"]["rounds"]
        deferred_rounds = runs["deferred"]["rounds"]
        expected_by_round = {
            round_index: [
                row["request_id"]
                for row in requests
                if row["round"] == round_index
            ]
            for round_index in range(DEFAULT_ROUNDS)
        }
        for round_index in range(DEFAULT_ROUNDS):
            blocking = {
                row["request_id"]: row
                for row in blocking_rounds[round_index]["requests"]
            }
            deferred = {
                row["request_id"]: row
                for row in deferred_rounds[round_index]["requests"]
            }
            for request_id in expected_by_round[round_index]:
                before = blocking[request_id]
                after = deferred[request_id]
                compared += 1
                fields = {
                    "token_ids": (
                        before.get("token_ids"),
                        after.get("token_ids"),
                    ),
                    "text": (before.get("text"), after.get("text")),
                    "prompt_tokens": (
                        before.get("usage", {}).get("prompt_tokens"),
                        after.get("usage", {}).get("prompt_tokens"),
                    ),
                    "completion_tokens": (
                        before.get("usage", {}).get("completion_tokens"),
                        after.get("usage", {}).get("completion_tokens"),
                    ),
                }
                differing = [
                    field
                    for field, (left, right) in fields.items()
                    if left != right
                ]
                if differing:
                    mismatches.append(
                        {
                            "round": round_index,
                            "request_id": request_id,
                            "differing_fields": differing,
                        }
                    )
    except (KeyError, TypeError, IndexError):
        return False, compared, [{"error": "raw run shape is invalid"}]
    return not mismatches, compared, mismatches


def _performance_diagnostics(
    runs: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics = []
    try:
        for round_index in range(DEFAULT_ROUNDS):
            blocking = runs["blocking"]["rounds"][round_index]
            deferred = runs["deferred"]["rounds"][round_index]
            diagnostics.append(
                {
                    "round": round_index,
                    "binding": False,
                    "blocking": {
                        "wall_s": blocking["wall_s"],
                        "median_ttft_s": blocking["median_ttft_s"],
                        "median_completion_latency_s": (
                            blocking["median_completion_latency_s"]
                        ),
                        "output_tokens_per_s": blocking[
                            "output_tokens_per_s"
                        ],
                    },
                    "deferred": {
                        "wall_s": deferred["wall_s"],
                        "median_ttft_s": deferred["median_ttft_s"],
                        "median_completion_latency_s": (
                            deferred["median_completion_latency_s"]
                        ),
                        "output_tokens_per_s": deferred[
                            "output_tokens_per_s"
                        ],
                    },
                    "deferred_over_blocking": {
                        "wall": deferred["wall_s"] / blocking["wall_s"],
                        "median_ttft": (
                            deferred["median_ttft_s"]
                            / blocking["median_ttft_s"]
                        ),
                        "median_completion_latency": (
                            deferred["median_completion_latency_s"]
                            / blocking["median_completion_latency_s"]
                        ),
                        "output_tokens_per_s": (
                            deferred["output_tokens_per_s"]
                            / blocking["output_tokens_per_s"]
                        ),
                    },
                }
            )
    except (KeyError, TypeError, IndexError, ZeroDivisionError):
        return [{"binding": False, "error": "performance records are invalid"}]
    ratios = [
        row["deferred_over_blocking"]
        for row in diagnostics
    ]
    diagnostics.append(
        {
            "aggregate": "median_of_round_ratios",
            "binding": False,
            "deferred_over_blocking": {
                metric: statistics.median(row[metric] for row in ratios)
                for metric in (
                    "wall",
                    "median_ttft",
                    "median_completion_latency",
                    "output_tokens_per_s",
                )
            },
            "policy": PERFORMANCE_POLICY,
        }
    )
    return diagnostics


def verify_evidence(
    artifact: dict[str, Any],
    *,
    verify_stored: bool = False,
) -> tuple[dict[str, bool], dict[str, Any], list[dict[str, Any]]]:
    """Rebuild every binding verdict and diagnostic from raw records.

    Verification performs no model load, CUDA work, network request, or
    benchmark execution.  It only needs the repository to re-bind the recorded
    clean commit.
    """
    config = artifact.get("config")
    source = artifact.get("source")
    hardware = artifact.get("hardware")
    checkpoint = artifact.get("checkpoint")
    p2p_probe = artifact.get("p2p_probe")
    requests = artifact.get("requests")
    runs = artifact.get("runs")
    required_dicts = (
        config,
        source,
        hardware,
        checkpoint,
        p2p_probe,
        runs,
    )
    if not all(isinstance(value, dict) for value in required_dicts):
        raise ValueError("artifact is missing a required object")
    architecture = checkpoint.get("architecture")
    vocab_size = (
        architecture.get("vocab_size")
        if isinstance(architecture, dict)
        else None
    )
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("checkpoint architecture has no valid vocab_size")

    selected = config.get("selected_devices")
    rounds = config.get("rounds")
    selected_ok = (
        isinstance(selected, dict)
        and set(selected) == {"prefill", "decode"}
        and all(
            isinstance(selected[role], int)
            and not isinstance(selected[role], bool)
            for role in ("prefill", "decode")
        )
        and selected["prefill"] != selected["decode"]
    )
    config_ok = (
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("measurement_kind") == MEASUREMENT_KIND
        and config.get("conditions") == list(CONDITIONS)
        and config.get("condition_execution_order") == list(CONDITIONS)
        and rounds == DEFAULT_ROUNDS
        and config.get("minimum_visible_cuda_devices")
        == MINIMUM_VISIBLE_GPUS
        and config.get("num_pages") == DEFAULT_NUM_PAGES
        and config.get("page_size") == DEFAULT_PAGE_SIZE
        and config.get("max_num_batched_tokens")
        == MAX_NUM_BATCHED_TOKENS
        and config.get("max_num_seqs") == MAX_NUM_SEQS
        and config.get("max_new_tokens") == MAX_NEW_TOKENS
        and config.get("warmup_new_tokens") == WARMUP_NEW_TOKENS
        and config.get("pd_separation") is True
        and config.get("tensor_parallel_size") == 1
        and config.get("decode_mode") == "eager"
        and config.get("latency_clock") == "time.perf_counter_ns"
        and config.get("cuda_boundary_synchronization") is True
        and config.get("p2p_probe") == _expected_p2p_probe_config()
        and config.get("performance_gate") is False
        and config.get("performance_policy") == PERFORMANCE_POLICY
        and config.get("prompt_token_ids_sha256")
        == EXPECTED_PROMPT_TOKEN_IDS_SHA256
        and config.get("measurement_source")
        == "production KairyuBackend with real checkpoint"
        and selected_ok
    )
    requests_ok = (
        isinstance(rounds, int)
        and not isinstance(rounds, bool)
        and _request_shape_ok(
            requests,
            rounds=rounds,
            vocab_size=vocab_size,
        )
    )
    run_keys_ok = isinstance(runs, dict) and set(runs) == set(CONDITIONS)
    selected_for_run = selected if selected_ok else {}
    p2p_probe_ok = (
        selected_ok
        and _p2p_probe_ok(p2p_probe, selected)
    )
    blocking_ok = (
        requests_ok
        and run_keys_ok
        and selected_ok
        and _run_shape_ok(
            runs["blocking"],
            condition="blocking",
            selected=selected_for_run,
            requests=requests,
            rounds=rounds,
            vocab_size=vocab_size,
        )
    )
    deferred_ok = (
        requests_ok
        and run_keys_ok
        and selected_ok
        and _run_shape_ok(
            runs["deferred"],
            condition="deferred",
            selected=selected_for_run,
            requests=requests,
            rounds=rounds,
            vocab_size=vocab_size,
        )
    )
    parity_ok, compared, mismatches = (
        _output_parity(runs, requests)
        if blocking_ok and deferred_ok
        else (False, 0, [{"error": "raw run shape is invalid"}])
    )
    diagnostics = (
        _performance_diagnostics(runs)
        if blocking_ok and deferred_ok
        else [{"binding": False, "error": "performance records are invalid"}]
    )
    summary = {
        "rounds": rounds if isinstance(rounds, int) else None,
        "requests_per_condition": (
            len(requests) if isinstance(requests, list) else 0
        ),
        "outputs_compared": compared,
        "exact_output_parity": parity_ok,
        "output_mismatches": mismatches,
        "p2p_event_overlap_and_raw_kv_parity": p2p_probe_ok,
        "performance_is_binding": False,
    }
    checks = {
        "fixed_real_measurement_configuration": config_ok,
        "clean_committed_source_and_script_binding": (
            _source_is_clean_and_bound(source)
        ),
        "full_qwen3_32b_checkpoint_provenance": (
            _checkpoint_is_qwen3_32b(checkpoint)
        ),
        "real_distinct_cuda_device_pair": (
            selected_ok
            and _hardware_is_real_selected_pair(hardware, selected)
        ),
        "real_p2p_event_overlap_and_raw_kv_parity": p2p_probe_ok,
        "fixed_natural_requests_and_qwen_tokenization": requests_ok,
        "blocking_production_run_complete": blocking_ok,
        "deferred_production_run_complete": deferred_ok,
        "exact_blocking_deferred_output_parity": parity_ok,
        "performance_evidence_explicitly_nonbinding": (
            config.get("performance_gate") is False
            and config.get("performance_policy") == PERFORMANCE_POLICY
            and all(row.get("binding") is False for row in diagnostics)
        ),
    }
    if verify_stored:
        stored_matches = (
            artifact.get("checks") == checks
            and artifact.get("summary") == summary
            and artifact.get("diagnostics") == diagnostics
            and artifact.get("gate_passed") == all(checks.values())
        )
        checks["stored_replay_matches"] = stored_matches
    return checks, summary, diagnostics


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    from verification.l1.correctness.parity_tp import _checkpoint_provenance

    _ensure_python_bin_on_path()
    selected = {
        "prefill": args.prefill_device,
        "decode": args.decode_device,
    }
    source_start = _source_provenance()
    hardware = _hardware_provenance()
    if not _hardware_is_real_selected_pair(hardware, selected):
        raise RuntimeError(
            "formal P-D evidence requires two distinct selected CUDA GPUs"
        )
    if args.assert_gate and not _source_snapshot_is_clean_and_bound(source_start):
        raise RuntimeError(
            "formal evidence requires a clean commit containing every bound source"
        )
    checkpoint_start = _checkpoint_provenance(args.model_path)
    if not _checkpoint_snapshot_is_qwen3_32b(checkpoint_start):
        raise RuntimeError(
            "--model-path is not the fully identified Qwen3-32B checkpoint"
        )
    requests = _build_request_evidence(args.model_path)
    p2p_probe = _run_p2p_probe(selected, hardware)

    runs = {}
    for condition in CONDITIONS:
        runs[condition] = asyncio.run(
            _run_condition(
                condition=condition,
                model_path=args.model_path,
                requests=requests,
                selected=selected,
            )
        )

    checkpoint_end = _checkpoint_provenance(args.model_path)
    checkpoint = {
        **checkpoint_start,
        "measurement_end": checkpoint_end,
        "stable_across_measurement": checkpoint_start == checkpoint_end,
    }
    if args.assert_gate and not _checkpoint_is_qwen3_32b(checkpoint):
        raise RuntimeError(
            "Qwen3-32B checkpoint changed while formal evidence was running"
        )
    source_end = _source_provenance()
    source = {
        **source_start,
        "measurement_end": source_end,
        "stable_across_measurement": source_start == source_end,
    }
    if args.assert_gate and not _source_is_clean_and_bound(source):
        raise RuntimeError(
            "bound source or commit changed while formal evidence was running"
        )

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": MEASUREMENT_KIND,
        "created_at": datetime.now(UTC).isoformat(),
        "source": source,
        "hardware": hardware,
        "checkpoint": checkpoint,
        "p2p_probe": p2p_probe,
        "config": {
            "conditions": list(CONDITIONS),
            "condition_execution_order": list(CONDITIONS),
            "selected_devices": selected,
            "minimum_visible_cuda_devices": MINIMUM_VISIBLE_GPUS,
            "rounds": DEFAULT_ROUNDS,
            "num_pages": DEFAULT_NUM_PAGES,
            "page_size": DEFAULT_PAGE_SIZE,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "warmup_new_tokens": WARMUP_NEW_TOKENS,
            "pd_separation": True,
            "tensor_parallel_size": 1,
            "decode_mode": "eager",
            "latency_clock": "time.perf_counter_ns",
            "cuda_boundary_synchronization": True,
            "p2p_probe": _expected_p2p_probe_config(),
            "performance_gate": False,
            "performance_policy": PERFORMANCE_POLICY,
            "prompt_token_ids_sha256": (
                EXPECTED_PROMPT_TOKEN_IDS_SHA256
            ),
            "measurement_source": (
                "production KairyuBackend with real checkpoint"
            ),
            "model_path_for_debugging_only": args.model_path,
        },
        "requests": requests,
        "runs": runs,
    }
    checks, summary, diagnostics = verify_evidence(artifact)
    artifact["checks"] = checks
    artifact["summary"] = summary
    artifact["diagnostics"] = diagnostics
    artifact["gate_passed"] = all(checks.values())
    return artifact


def _print_verdict(
    checks: dict[str, bool],
    summary: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> None:
    print(
        json.dumps(
            {
                "checks": checks,
                "summary": summary,
                "diagnostics": diagnostics,
                "gate_passed": all(checks.values()),
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--prefill-device", type=int, default=0)
    parser.add_argument("--decode-device", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--num-pages", type=int, default=DEFAULT_NUM_PAGES)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--assert-gate",
        action="store_true",
        help="exit non-zero unless every binding correctness check passes",
    )
    args = parser.parse_args()

    if args.verify is not None:
        measurement_args = (
            args.model_path,
            args.output,
        )
        if any(value is not None for value in measurement_args):
            parser.error(
                "--verify cannot be combined with --model-path or --output"
            )
        artifact = json.loads(args.verify.read_text())
        checks, summary, diagnostics = verify_evidence(
            artifact, verify_stored=True
        )
        _print_verdict(checks, summary, diagnostics)
        return 1 if args.assert_gate and not all(checks.values()) else 0

    if args.model_path is None or args.output is None:
        parser.error("measurement requires both --model-path and --output")
    if args.prefill_device == args.decode_device:
        parser.error("--prefill-device and --decode-device must be distinct")
    if args.prefill_device < 0 or args.decode_device < 0:
        parser.error("CUDA device indices must be non-negative")
    if args.rounds != DEFAULT_ROUNDS:
        parser.error(
            f"formal gate fixes --rounds at {DEFAULT_ROUNDS}, got {args.rounds}"
        )
    if args.num_pages != DEFAULT_NUM_PAGES:
        parser.error(
            "formal gate fixes --num-pages at "
            f"{DEFAULT_NUM_PAGES}, got {args.num_pages}"
        )
    if args.page_size != DEFAULT_PAGE_SIZE:
        parser.error(
            "formal gate fixes --page-size at "
            f"{DEFAULT_PAGE_SIZE}, got {args.page_size}"
        )

    artifact = _measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    _print_verdict(
        artifact["checks"],
        artifact["summary"],
        artifact["diagnostics"],
    )
    print(f"written: {args.output}")
    return 1 if args.assert_gate and not artifact["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
