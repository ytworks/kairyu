"""Formal Qwen3-32B TP=1/TP=8 sampling-ownership evidence gate (#225).

The gate runs the deployment model runners, once at TP=1 and once at TP=8,
over the same fixed natural prompts, request IDs, and sampling configuration.
It intentionally mixes completion lengths and covers:

* exact greedy sampling;
* explicitly seeded temperature/top-k/top-p/min-p sampling;
* presence, frequency, and repetition penalties; and
* selected-token plus top-token logprobs.

Every public ``SampledToken`` is retained as raw evidence. Passing requires
complete finite continuations, the real TP topology and rank-0-only ownership,
and compatible distributions while both runs still have the same prefix.
At the first free-running divergence, each selected token is scored directly
under the other run's full raw distribution and must have bounded cross-run
logprob drift. Later tokens have
different prefixes and exact TP=1/TP=8 equality is therefore diagnostic only.
This is a correctness gate, not a throughput benchmark, so wall time and OS
jitter are not measured.

Examples:

    uv run python bench/tp_sampling_owner_qwen.py \
      --model-path /models/qwen3-32b \
      --output bench/results/issue-225-qwen3-32b-tp1-tp8.json \
      --assert-gate

    uv run python bench/tp_sampling_owner_qwen.py \
      --verify bench/results/issue-225-qwen3-32b-tp1-tp8.json \
      --assert-gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = 1
REQUIRED_TP_DEGREES = (1, 8)
REQUIRED_VISIBLE_GPUS = 8
TOP_LOGPROBS = 64
# This is the same bound used by the established real-model distribution gate:
# it tolerates BF16 TP reduction-order drift, while a materially different
# distribution still fails independently of exact token equality.
LOGPROB_ABS_TOLERANCE = 0.25
PROBE_OWN_LOGPROB_ABS_TOLERANCE = 1e-5
# A top-k boundary can move under harmless BF16 near-ties. Shared top-token
# membership is reported for aligned prefixes, never used as a pass/fail
# condition. Sixty-of-sixty-four is an orientation threshold only.
DIAGNOSTIC_TOP_LOGPROB_OVERLAP = 60
# The fixed natural-language text is not sufficient provenance on its own:
# bind it to the exact tokenizer output from the reviewed Qwen3-32B checkpoint.
EXPECTED_PROMPT_TOKEN_IDS_SHA256 = (
    "5db4cd58e794d5bbfadb8d12e56d59d41d096335f705f3a65bb0aef389572cab"
)
# Six short requests consume far below 256 x 16 token slots. Keeping the formal
# pool at 256 avoids reserving roughly 16 GiB of TP=1 Qwen KV for a 43-token
# continuation gate while retaining ample headroom for all fixed prompts.
DEFAULT_NUM_PAGES = 256
DEFAULT_PAGE_SIZE = 16
MAX_NUM_BATCHED_TOKENS = 2048
SOURCE_PATH = "bench/tp_sampling_owner_qwen.py"
SOURCE_BOUND_PATHS = (
    SOURCE_PATH,
    "bench/parity_tp.py",
    "kairyu/engine/core/comm.py",
    "kairyu/engine/core/dist_comm.py",
    "kairyu/engine/core/engine_core.py",
    "kairyu/engine/core/model_runner.py",
    "kairyu/engine/core/sampler.py",
    "kairyu/engine/core/spec_runner.py",
    "kairyu/engine/core/step_input.py",
    "kairyu/engine/core/tp_runner.py",
    "kairyu/engine/core/worker.py",
    "kairyu/engine/kairyu_backend.py",
    "kairyu/models/parallel.py",
)


def _ensure_python_bin_on_path() -> None:
    """Expose build helpers installed beside the active Python executable."""
    path = os.environ.get("PATH", "")
    if shutil.which("ninja", path=path) is not None:
        return
    python_bin = Path(sys.executable).parent
    ninja = python_bin / "ninja"
    if os.access(ninja, os.X_OK):
        os.environ["PATH"] = os.pathsep.join(part for part in (str(python_bin), path) if part)


@dataclass(frozen=True)
class _RequestCase:
    request_id: str
    text: str
    max_new_tokens: int
    sampling: dict[str, int | float | None]


_REQUEST_CASES: tuple[_RequestCase, ...] = (
    _RequestCase(
        request_id="greedy-fact-short",
        text="The capital of France is",
        max_new_tokens=4,
        sampling={
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": None,
            "logprobs": TOP_LOGPROBS,
        },
    ),
    _RequestCase(
        request_id="greedy-code-long",
        text="def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        max_new_tokens=9,
        sampling={
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": None,
            "logprobs": TOP_LOGPROBS,
        },
    ),
    _RequestCase(
        request_id="seeded-filtered-short",
        text="The three primary colors are red, blue, and",
        max_new_tokens=5,
        sampling={
            "temperature": 0.75,
            "top_k": 40,
            "top_p": 0.92,
            "min_p": 0.02,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": 225_001,
            "logprobs": TOP_LOGPROBS,
        },
    ),
    _RequestCase(
        request_id="seeded-filtered-long",
        text="In object-oriented programming, inheritance allows a class to",
        max_new_tokens=8,
        sampling={
            "temperature": 0.8,
            "top_k": 32,
            "top_p": 0.9,
            "min_p": 0.01,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": 225_002,
            "logprobs": TOP_LOGPROBS,
        },
    ),
    _RequestCase(
        request_id="seeded-penalties-medium",
        text="The first ten prime numbers are 2, 3, 5, 7, 11,",
        max_new_tokens=7,
        sampling={
            "temperature": 0.7,
            "top_k": 48,
            "top_p": 0.94,
            "min_p": 0.01,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.1,
            "seed": 225_003,
            "logprobs": TOP_LOGPROBS,
        },
    ),
    _RequestCase(
        request_id="seeded-penalties-long",
        text="To sort a list in reverse order in Python, pass the argument",
        max_new_tokens=10,
        sampling={
            "temperature": 0.85,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.015,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.15,
            "repetition_penalty": 1.05,
            "seed": 225_004,
            "logprobs": TOP_LOGPROBS,
        },
    ),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    from bench.parity_tp import _code_provenance

    code = _code_provenance()
    current = Path(__file__).read_bytes()
    committed = (
        _git_show(str(code["commit"]), SOURCE_PATH) if code.get("commit") is not None else None
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
            bound_file_sha256[relative] = _sha256(committed_blob)
            bound_files_match_commit &= working_blob == committed_blob
    return {
        **code,
        "script_path": SOURCE_PATH,
        "script_sha256": _sha256(current),
        "committed_script_sha256": (_sha256(committed) if committed is not None else None),
        "script_matches_commit": committed == current,
        "bound_file_sha256": bound_file_sha256,
        "bound_files_match_commit": (
            bound_files_match_commit and set(bound_file_sha256) == set(SOURCE_BOUND_PATHS)
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
        committed_sha256 = _sha256(committed)
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

    from bench.parity_tp import _gpu_runtime_provenance
    from kairyu.engine.core.hw_profile import probe

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
    return {
        "arch": profile.arch,
        "profile_device_name": profile.device_name,
        "profile_device_count": profile.device_count,
        "profile_sm": profile.sm,
        "cuda_available": torch.cuda.is_available(),
        "visible_device_count": visible,
        "devices": devices,
        "runtime": _gpu_runtime_provenance(),
    }


def _checkpoint_is_qwen3_32b(checkpoint: dict[str, Any]) -> bool:
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
    if any(architecture.get(name) != value for name, value in expected_architecture.items()):
        return False
    if not weights:
        return False
    filenames: set[str] = set()
    for row in weights:
        filename = row.get("file")
        digest = row.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or filename in filenames
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
        filenames.add(filename)
    rollup = _sha256(json.dumps(weights, sort_keys=True).encode())
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


def _hardware_is_exact_eight_gpu(hardware: dict[str, Any]) -> bool:
    devices = hardware.get("devices")
    runtime = hardware.get("runtime")
    if not isinstance(devices, list) or not isinstance(runtime, dict):
        return False
    capabilities = runtime.get("device_capabilities")
    nvidia_smi = runtime.get("nvidia_smi")
    return (
        hardware.get("arch") == "cuda"
        and hardware.get("cuda_available") is True
        and hardware.get("profile_device_count") == REQUIRED_VISIBLE_GPUS
        and hardware.get("visible_device_count") == REQUIRED_VISIBLE_GPUS
        and len(devices) == REQUIRED_VISIBLE_GPUS
        and [device.get("index") for device in devices] == list(range(REQUIRED_VISIBLE_GPUS))
        and len({device.get("uuid") for device in devices}) == REQUIRED_VISIBLE_GPUS
        and len({device.get("pci") for device in devices}) == REQUIRED_VISIBLE_GPUS
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
        and isinstance(capabilities, list)
        and len(capabilities) == REQUIRED_VISIBLE_GPUS
        and isinstance(nvidia_smi, list)
        and len(nvidia_smi) == REQUIRED_VISIBLE_GPUS
        and runtime.get("nccl") is not None
        and runtime.get("torch_cuda") is not None
    )


def _expected_request_config() -> list[dict[str, Any]]:
    return [
        {
            "request_id": case.request_id,
            "text": case.text,
            "max_new_tokens": case.max_new_tokens,
            "sampling": dict(case.sampling),
            "ignore_eos": True,
        }
        for case in _REQUEST_CASES
    ]


def _prompt_token_ids_sha256(requests: list[dict[str, Any]]) -> str:
    payload = [
        {
            "request_id": request["request_id"],
            "text": request["text"],
            "prompt_token_ids": request["prompt_token_ids"],
        }
        for request in requests
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256(encoded)


def _expected_sampling_ownership(tp: int) -> list[dict[str, Any]]:
    if tp == 1:
        return [
            {
                "rank": 0,
                "control_world_size": 1,
                "control_backend": "local",
                "model_world_size": 1,
                "model_backend": "local",
                "sampling_owner": True,
                "sampler_present": True,
                "device": "cuda:0",
            }
        ]
    return [
        {
            "rank": rank,
            "control_world_size": tp,
            "control_backend": "gloo",
            "model_world_size": tp,
            "model_backend": "nccl",
            "sampling_owner": rank == 0,
            "sampler_present": rank == 0,
            "device": f"cuda:{rank}",
        }
        for rank in range(tp)
    ]


def _local_sampling_ownership(runner: Any) -> list[dict[str, Any]]:
    sampling_owner = getattr(runner, "sampling_owner", None)
    sampler = getattr(runner, "sampler", None)
    device = getattr(runner, "_device", None)
    if sampling_owner is not True or sampler is None or device is None:
        raise RuntimeError("TP=1 runner does not expose the expected owner, sampler, and device")
    return [
        {
            "rank": 0,
            "control_world_size": 1,
            "control_backend": "local",
            "model_world_size": 1,
            "model_backend": "local",
            "sampling_owner": sampling_owner,
            "sampler_present": sampler is not None,
            "device": str(device),
        }
    ]


def _build_requests(
    model_path: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    from bench.parity_tp import _check_vocab_agreement
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest
    from kairyu.engine.tokenizer import resolve_tokenizer

    tokenizer = resolve_tokenizer(model_path)
    vocab_size = len(tokenizer.vocab())
    _check_vocab_agreement(model_path, vocab_size)
    requests = []
    evidence = []
    for case in _REQUEST_CASES:
        prompt_token_ids = tuple(tokenizer.encode(case.text))
        if not prompt_token_ids:
            raise RuntimeError(f"tokenizer produced an empty prompt for {case.request_id}")
        requests.append(
            EngineRequest(
                request_id=case.request_id,
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=case.max_new_tokens,
                ignore_eos=True,
                sampling=EngineSampling(**case.sampling),
            )
        )
        evidence.append(
            {
                **asdict(case),
                "ignore_eos": True,
                "prompt_token_ids": list(prompt_token_ids),
            }
        )
    return requests, evidence


def _sample_record(position: int, sample: Any) -> dict[str, Any]:
    return {
        "position": position,
        "token_id": int(sample.token_id),
        "logprob": (None if sample.logprob is None else float(sample.logprob)),
        "top_logprobs": (
            None
            if sample.top_logprobs is None
            else [
                {"token_id": int(token_id), "logprob": float(logprob)}
                for token_id, logprob in sample.top_logprobs
            ]
        ),
        "grammar_terminated": bool(sample.grammar_terminated),
    }


class _RecordingRunner:
    """Resolve and retain every public sample without changing model work."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.records: dict[str, list[dict[str, Any]]] = {
            case.request_id: [] for case in _REQUEST_CASES
        }
        self.steps = 0

    def execute(self, scheduled: Any, states: Any) -> dict[str, tuple[Any, ...]]:
        sampled = self._inner.execute(scheduled, states)
        # Resolving here materializes the public result that the scheduler must
        # consume anyway. The underlying forward, owner-only sampling, packet
        # broadcast, and follower adoption have already completed.
        resolved = {request_id: tuple(tokens) for request_id, tokens in (sampled or {}).items()}
        for request_id, tokens in resolved.items():
            if request_id not in self.records:
                raise RuntimeError(f"runner emitted an unknown request ID {request_id!r}")
            for token in tokens:
                position = len(self.records[request_id])
                self.records[request_id].append(_sample_record(position, token))
        self.steps += 1
        return resolved

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _install_raw_logprob_recorder(
    inner: Any,
) -> dict[str, dict[int, Any]]:
    """Retain raw distributions in host memory until both TP runs finish.

    Public ``top_logprobs`` are intentionally pre-penalty top-N values. A
    penalty can promote a selected token from outside that set, so reciprocal
    first-divergence evidence must gather the two exact token indices from the
    full raw log-softmax rather than treating top-N membership as support.
    """
    import torch

    local = getattr(inner, "_local", inner)
    sampler = getattr(local, "sampler", None)
    if sampler is None:
        raise RuntimeError("formal Qwen runner has no rank-0 sampler to observe")
    records: dict[str, dict[int, Any]] = {}

    def retain(request_id: str, position: int, logits: Any) -> None:
        by_position = records.setdefault(request_id, {})
        if position in by_position:
            raise RuntimeError(
                "formal raw-logprob recorder observed duplicate position "
                f"{position} for {request_id!r}"
            )
        by_position[position] = torch.log_softmax(
            logits.detach().to(dtype=torch.float32),
            dim=-1,
        ).cpu()

    original_device = sampler.sample_device

    def observed_device(
        request_id: str,
        sampling: Any,
        position: int,
        logits: Any,
        **kwargs: Any,
    ) -> Any:
        retain(request_id, position, logits)
        return original_device(
            request_id,
            sampling,
            position,
            logits,
            **kwargs,
        )

    original_host = sampler.sample

    def observed_host(
        request_id: str,
        sampling: Any,
        position: int,
        logits: Any,
        **kwargs: Any,
    ) -> Any:
        retain(request_id, position, logits)
        return original_host(
            request_id,
            sampling,
            position,
            logits,
            **kwargs,
        )

    sampler.sample_device = observed_device
    sampler.sample = observed_host
    return records


def _run_degree(
    model_path: str,
    tp: int,
    requests: list[Any],
    num_pages: int,
    page_size: int,
) -> tuple[dict[str, Any], dict[str, dict[int, Any]]]:
    from bench.parity_tp import _real_runner
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler

    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        page_size=page_size,
    )
    inner, teardown = _real_runner(model_path, tp, num_pages, page_size)
    raw_logprobs = _install_raw_logprob_recorder(inner)
    recorder = _RecordingRunner(inner)
    core = None
    try:
        core = EngineCore(scheduler, recorder)
        for request in requests:
            core.add_request(request)
        output_tokens = core.run_to_completion()
        raw_records = recorder.records
        steps = recorder.steps
        sampling_ownership = (
            _local_sampling_ownership(inner)
            if tp == 1
            else [dict(row) for row in inner.sampling_ownership_metadata()]
        )
    finally:
        # Drop caller references before the helper releases weights; otherwise
        # TP=1 can retain a full checkpoint on GPU 0 while TP=8 starts.
        core = None
        recorder = None
        inner = None
        scheduler = None
        cache = None
        teardown()

    return (
        {
            "tp_degree": tp,
            "steps": steps,
            "ownership_probe_phase": "after_generation_completed",
            "sampling_ownership": sampling_ownership,
            "output_tokens": {
                request_id: list(tokens) for request_id, tokens in sorted(output_tokens.items())
            },
            "raw_records": {
                request_id: records for request_id, records in sorted(raw_records.items())
            },
        },
        raw_logprobs,
    )


def _request_shape_ok(requests: Any, vocab_size: int) -> bool:
    if not isinstance(requests, list) or len(requests) != len(_REQUEST_CASES):
        return False
    expected = _expected_request_config()
    for actual, fixed in zip(requests, expected, strict=True):
        if any(actual.get(key) != value for key, value in fixed.items()):
            return False
        prompt = actual.get("prompt_token_ids")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < vocab_size
                for token_id in prompt
            )
        ):
            return False
    return (
        len({row["request_id"] for row in requests}) == len(requests)
        and _prompt_token_ids_sha256(requests) == EXPECTED_PROMPT_TOKEN_IDS_SHA256
    )


def _validate_record(
    record: Any,
    *,
    position: int,
    vocab_size: int,
) -> bool:
    if not isinstance(record, dict):
        return False
    token_id = record.get("token_id")
    logprob = record.get("logprob")
    top = record.get("top_logprobs")
    if (
        record.get("position") != position
        or not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or not 0 <= token_id < vocab_size
        or not isinstance(logprob, (int, float))
        or isinstance(logprob, bool)
        or not math.isfinite(float(logprob))
        or float(logprob) > 1e-3
        or record.get("grammar_terminated") is not False
        or not isinstance(top, list)
        or len(top) != TOP_LOGPROBS
    ):
        return False
    seen: set[int] = set()
    previous = math.inf
    selected_top_logprob: float | None = None
    for entry in top:
        if not isinstance(entry, dict):
            return False
        top_token = entry.get("token_id")
        top_logprob = entry.get("logprob")
        if (
            not isinstance(top_token, int)
            or isinstance(top_token, bool)
            or not 0 <= top_token < vocab_size
            or top_token in seen
            or not isinstance(top_logprob, (int, float))
            or isinstance(top_logprob, bool)
            or not math.isfinite(float(top_logprob))
            or float(top_logprob) > 1e-3
            or float(top_logprob) > previous + 1e-6
        ):
            return False
        seen.add(top_token)
        previous = float(top_logprob)
        if top_token == token_id:
            selected_top_logprob = float(top_logprob)
    return selected_top_logprob is None or abs(selected_top_logprob - float(logprob)) <= 1e-6


def _run_shape_ok(
    run: Any,
    *,
    tp: int,
    requests: list[dict[str, Any]],
    vocab_size: int,
) -> bool:
    if not isinstance(run, dict) or run.get("tp_degree") != tp:
        return False
    if not isinstance(run.get("steps"), int) or run["steps"] <= 0:
        return False
    output_tokens = run.get("output_tokens")
    raw_records = run.get("raw_records")
    expected_ids = {request["request_id"] for request in requests}
    if (
        not isinstance(output_tokens, dict)
        or not isinstance(raw_records, dict)
        or set(output_tokens) != expected_ids
        or set(raw_records) != expected_ids
    ):
        return False
    for request in requests:
        request_id = request["request_id"]
        expected_length = request["max_new_tokens"]
        tokens = output_tokens[request_id]
        records = raw_records[request_id]
        if (
            not isinstance(tokens, list)
            or not isinstance(records, list)
            or len(tokens) != expected_length
            or len(records) != expected_length
        ):
            return False
        if any(
            not _validate_record(
                record,
                position=position,
                vocab_size=vocab_size,
            )
            for position, record in enumerate(records)
        ):
            return False
        if tokens != [record["token_id"] for record in records]:
            return False
    return True


def _sampling_ownership_ok(run: Any, *, tp: int) -> bool:
    return (
        isinstance(run, dict)
        and run.get("tp_degree") == tp
        and run.get("ownership_probe_phase") == "after_generation_completed"
        and run.get("sampling_ownership") == _expected_sampling_ownership(tp)
    )


_CROSS_PROBE_FIELDS = frozenset(
    {
        "request_id",
        "position",
        "tp1_token_id",
        "tp8_token_id",
        "tp1_selected_under_tp1",
        "tp1_selected_under_tp8",
        "tp8_selected_under_tp1",
        "tp8_selected_under_tp8",
    }
)


def _first_divergences(
    tp1: dict[str, Any],
    tp8: dict[str, Any],
    requests: list[dict[str, Any]],
) -> list[tuple[str, int, dict[str, Any], dict[str, Any]]]:
    rows = []
    for request in requests:
        request_id = request["request_id"]
        for position, (tp1_record, tp8_record) in enumerate(
            zip(
                tp1["raw_records"][request_id],
                tp8["raw_records"][request_id],
                strict=True,
            )
        ):
            if tp1_record["token_id"] != tp8_record["token_id"]:
                rows.append((request_id, position, tp1_record, tp8_record))
                break
    return rows


def _build_first_divergence_probes(
    runs: dict[str, dict[str, Any]],
    raw_logprobs: dict[str, dict[str, dict[int, Any]]],
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    probes = []
    for request_id, position, tp1_record, tp8_record in _first_divergences(
        runs["tp1"], runs["tp8"], requests
    ):
        tp1_distribution = raw_logprobs["tp1"].get(request_id, {}).get(position)
        tp8_distribution = raw_logprobs["tp8"].get(request_id, {}).get(position)
        if tp1_distribution is None or tp8_distribution is None:
            raise RuntimeError(
                "formal raw-logprob recorder is missing first-divergence "
                f"evidence for {request_id!r} position {position}"
            )
        if tp1_distribution.ndim != 1 or tp8_distribution.ndim != 1:
            raise RuntimeError(
                "formal raw-logprob recorder produced a non-vector "
                f"distribution for {request_id!r} position {position}"
            )
        tp1_token = tp1_record["token_id"]
        tp8_token = tp8_record["token_id"]
        if max(tp1_token, tp8_token) >= min(tp1_distribution.numel(), tp8_distribution.numel()):
            raise RuntimeError(
                "first-divergence token is outside the retained raw "
                f"distribution for {request_id!r} position {position}"
            )
        probes.append(
            {
                "request_id": request_id,
                "position": position,
                "tp1_token_id": tp1_token,
                "tp8_token_id": tp8_token,
                "tp1_selected_under_tp1": float(tp1_distribution[tp1_token]),
                "tp1_selected_under_tp8": float(tp8_distribution[tp1_token]),
                "tp8_selected_under_tp1": float(tp1_distribution[tp8_token]),
                "tp8_selected_under_tp8": float(tp8_distribution[tp8_token]),
            }
        )
    return probes


def _validate_first_divergence_probes(
    tp1: dict[str, Any],
    tp8: dict[str, Any],
    requests: list[dict[str, Any]],
    probes: Any,
) -> tuple[bool, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    expected = _first_divergences(tp1, tp8, requests)
    if not isinstance(probes, list) or len(probes) != len(expected):
        return (
            False,
            {},
            [
                {
                    "kind": "first_divergence_probe_count_failure",
                    "expected": len(expected),
                    "actual": (len(probes) if isinstance(probes, list) else None),
                }
            ],
        )

    by_request: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for probe, (
        request_id,
        position,
        tp1_record,
        tp8_record,
    ) in zip(probes, expected, strict=True):
        if not isinstance(probe, dict) or set(probe) != _CROSS_PROBE_FIELDS:
            diagnostics.append(
                {
                    "kind": "first_divergence_probe_shape_failure",
                    "request_id": request_id,
                    "position": position,
                }
            )
            continue
        fixed = {
            "request_id": request_id,
            "position": position,
            "tp1_token_id": tp1_record["token_id"],
            "tp8_token_id": tp8_record["token_id"],
        }
        if any(probe.get(name) != value for name, value in fixed.items()):
            diagnostics.append(
                {
                    "kind": "first_divergence_probe_identity_failure",
                    "request_id": request_id,
                    "position": position,
                }
            )
            continue
        value_names = (
            "tp1_selected_under_tp1",
            "tp1_selected_under_tp8",
            "tp8_selected_under_tp1",
            "tp8_selected_under_tp8",
        )
        if any(
            not isinstance(probe[name], (int, float))
            or isinstance(probe[name], bool)
            or not math.isfinite(float(probe[name]))
            or float(probe[name]) > 1e-3
            for name in value_names
        ):
            diagnostics.append(
                {
                    "kind": "first_divergence_probe_value_failure",
                    "request_id": request_id,
                    "position": position,
                }
            )
            continue
        public_deltas = {
            "tp1_selected_under_tp1": abs(
                float(probe["tp1_selected_under_tp1"]) - float(tp1_record["logprob"])
            ),
            "tp8_selected_under_tp8": abs(
                float(probe["tp8_selected_under_tp8"]) - float(tp8_record["logprob"])
            ),
        }
        tp1_top = {row["token_id"]: float(row["logprob"]) for row in tp1_record["top_logprobs"]}
        tp8_top = {row["token_id"]: float(row["logprob"]) for row in tp8_record["top_logprobs"]}
        if tp1_record["token_id"] in tp8_top:
            public_deltas["tp1_selected_under_tp8_topn"] = abs(
                float(probe["tp1_selected_under_tp8"]) - tp8_top[tp1_record["token_id"]]
            )
        if tp8_record["token_id"] in tp1_top:
            public_deltas["tp8_selected_under_tp1_topn"] = abs(
                float(probe["tp8_selected_under_tp1"]) - tp1_top[tp8_record["token_id"]]
            )
        if any(delta > PROBE_OWN_LOGPROB_ABS_TOLERANCE for delta in public_deltas.values()):
            diagnostics.append(
                {
                    "kind": "first_divergence_probe_public_logprob_failure",
                    "request_id": request_id,
                    "position": position,
                    "absolute_deltas": public_deltas,
                }
            )
            continue
        by_request[request_id] = probe
    return not diagnostics, by_request, diagnostics


def _compare_runs(
    tp1: dict[str, Any],
    tp8: dict[str, Any],
    requests: list[dict[str, Any]],
    first_divergence_probes: Any,
) -> tuple[dict[str, bool], dict[str, Any], list[dict[str, Any]]]:
    token_mismatches: list[dict[str, Any]] = []
    aligned_selected_deltas: list[float] = []
    aligned_selected_delta_failures: list[dict[str, Any]] = []
    first_divergence_deltas: list[float] = []
    first_divergence_delta_failures: list[dict[str, Any]] = []
    top_deltas: list[float] = []
    top_overlaps: list[int] = []
    top_overlap_diagnostics: list[dict[str, Any]] = []
    top_delta_diagnostics: list[dict[str, Any]] = []
    first_divergence_diagnostics: list[dict[str, Any]] = []
    common_prefix_tokens = 0
    requests_with_divergence = 0
    (
        first_divergence_probes_ok,
        probes_by_request,
        first_divergence_probe_diagnostics,
    ) = _validate_first_divergence_probes(
        tp1,
        tp8,
        requests,
        first_divergence_probes,
    )

    for request in requests:
        request_id = request["request_id"]
        left = tp1["raw_records"][request_id]
        right = tp8["raw_records"][request_id]
        diverged = False
        for position, (tp1_record, tp8_record) in enumerate(zip(left, right, strict=True)):
            tp1_token = tp1_record["token_id"]
            tp8_token = tp8_record["token_id"]
            tokens_match = tp1_token == tp8_token
            if not tokens_match:
                token_mismatches.append(
                    {
                        "kind": "free_running_token_mismatch_diagnostic",
                        "request_id": request_id,
                        "position": position,
                        "tp1": tp1_token,
                        "tp8": tp8_token,
                        "after_first_divergence": diverged,
                    }
                )

            # Once the generated tokens differ, all later distributions are
            # conditioned on different inputs and cannot be compared.
            if diverged:
                continue

            tp1_top = {
                entry["token_id"]: float(entry["logprob"]) for entry in tp1_record["top_logprobs"]
            }
            tp8_top = {
                entry["token_id"]: float(entry["logprob"]) for entry in tp8_record["top_logprobs"]
            }
            shared = sorted(set(tp1_top) & set(tp8_top))
            top_overlaps.append(len(shared))
            if len(shared) < DIAGNOSTIC_TOP_LOGPROB_OVERLAP:
                top_overlap_diagnostics.append(
                    {
                        "kind": "aligned_top_logprob_membership_diagnostic",
                        "request_id": request_id,
                        "position": position,
                        "tp1_token_ids": sorted(tp1_top),
                        "tp8_token_ids": sorted(tp8_top),
                        "overlap": len(shared),
                    }
                )
            for token_id in shared:
                delta = abs(tp1_top[token_id] - tp8_top[token_id])
                top_deltas.append(delta)
                if delta > LOGPROB_ABS_TOLERANCE:
                    top_delta_diagnostics.append(
                        {
                            "kind": "aligned_shared_top_logprob_delta_diagnostic",
                            "request_id": request_id,
                            "position": position,
                            "token_id": token_id,
                            "absolute_delta": delta,
                        }
                    )

            if tokens_match:
                common_prefix_tokens += 1
                selected_delta = abs(float(tp1_record["logprob"]) - float(tp8_record["logprob"]))
                aligned_selected_deltas.append(selected_delta)
                if selected_delta > LOGPROB_ABS_TOLERANCE:
                    aligned_selected_delta_failures.append(
                        {
                            "kind": "aligned_selected_logprob_delta_failure",
                            "request_id": request_id,
                            "position": position,
                            "token_id": tp1_token,
                            "tp1_logprob": tp1_record["logprob"],
                            "tp8_logprob": tp8_record["logprob"],
                            "absolute_delta": selected_delta,
                        }
                    )
                continue

            diverged = True
            requests_with_divergence += 1
            tp1_supported_by_tp8 = tp1_token in tp8_top
            tp8_supported_by_tp1 = tp8_token in tp1_top
            first_divergence_diagnostics.append(
                {
                    "kind": "first_free_running_divergence",
                    "request_id": request_id,
                    "position": position,
                    "tp1_token_id": tp1_token,
                    "tp8_token_id": tp8_token,
                    "tp1_selected_in_tp8_raw_top64_diagnostic": (tp1_supported_by_tp8),
                    "tp8_selected_in_tp1_raw_top64_diagnostic": (tp8_supported_by_tp1),
                    "direct_cross_logprob_probe_present": (request_id in probes_by_request),
                }
            )
            probe = probes_by_request.get(request_id)
            if probe is None:
                continue

            cross_views = (
                (
                    "tp1_selected_seen_by_tp8",
                    tp1_token,
                    float(probe["tp1_selected_under_tp1"]),
                    float(probe["tp1_selected_under_tp8"]),
                ),
                (
                    "tp8_selected_seen_by_tp1",
                    tp8_token,
                    float(probe["tp8_selected_under_tp8"]),
                    float(probe["tp8_selected_under_tp1"]),
                ),
            )
            for view, token_id, owner_logprob, other_logprob in cross_views:
                delta = abs(owner_logprob - other_logprob)
                first_divergence_deltas.append(delta)
                if delta > LOGPROB_ABS_TOLERANCE:
                    first_divergence_delta_failures.append(
                        {
                            "kind": "first_divergence_selected_logprob_failure",
                            "request_id": request_id,
                            "position": position,
                            "view": view,
                            "token_id": token_id,
                            "owner_logprob": owner_logprob,
                            "other_run_logprob": other_logprob,
                            "absolute_delta": delta,
                        }
                    )

    total_tokens = sum(request["max_new_tokens"] for request in requests)
    exact_matches = total_tokens - len(token_mismatches)
    selected_max = max(aligned_selected_deltas, default=None)
    first_divergence_max = max(first_divergence_deltas, default=None)
    top_max = max(top_deltas, default=None)
    checks = {
        "aligned_prefix_selected_logprob_tolerance": (not aligned_selected_delta_failures),
        "first_divergence_direct_cross_logprobs_complete": (first_divergence_probes_ok),
        "first_divergence_selected_logprob_tolerance": (
            first_divergence_probes_ok and not first_divergence_delta_failures
        ),
    }
    summary = {
        "requests": len(requests),
        "tokens_per_degree": total_tokens,
        "exact_free_running_token_parity_diagnostic": not token_mismatches,
        "exact_free_running_token_matches": exact_matches,
        "exact_free_running_token_match_rate": exact_matches / total_tokens,
        "token_mismatches": len(token_mismatches),
        "requests_with_first_divergence": requests_with_divergence,
        "first_divergence_direct_cross_logprob_probes": len(probes_by_request),
        "common_prefix_matching_tokens": common_prefix_tokens,
        "aligned_selected_logprob_max_abs_delta": selected_max,
        "first_divergence_selected_logprob_max_abs_delta": (first_divergence_max),
        "aligned_shared_top_logprob_max_abs_delta_diagnostic": top_max,
        "aligned_minimum_top_logprob_token_overlap_diagnostic": (min(top_overlaps, default=None)),
        "free_running_exact_equality_is_binding": False,
        "post_divergence_distribution_comparison_performed": False,
        "shared_top_logprob_comparison_is_binding": False,
    }
    diagnostics = [
        *token_mismatches,
        *aligned_selected_delta_failures,
        *first_divergence_diagnostics,
        *first_divergence_probe_diagnostics,
        *first_divergence_delta_failures,
        *top_overlap_diagnostics,
        *top_delta_diagnostics,
    ]
    return checks, summary, diagnostics


def verify_evidence(
    artifact: dict[str, Any],
    *,
    verify_stored: bool = False,
) -> tuple[dict[str, bool], dict[str, Any], list[dict[str, Any]]]:
    """Rebuild every verdict from the raw request and sample records."""
    config = artifact.get("config")
    source = artifact.get("source")
    hardware = artifact.get("hardware")
    checkpoint = artifact.get("checkpoint")
    requests = artifact.get("requests")
    runs = artifact.get("runs")
    first_divergence_probes = artifact.get("first_divergence_cross_logprobs")
    if not all(isinstance(value, dict) for value in (config, source, hardware, checkpoint, runs)):
        raise ValueError("artifact is missing a required object")
    architecture = checkpoint.get("architecture")
    vocab_size = architecture.get("vocab_size") if isinstance(architecture, dict) else None
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("checkpoint architecture has no valid vocab_size")

    config_ok = (
        artifact.get("schema_version") == SCHEMA_VERSION
        and config.get("tp_degrees") == list(REQUIRED_TP_DEGREES)
        and config.get("visible_gpu_gate") == REQUIRED_VISIBLE_GPUS
        and config.get("num_pages") == DEFAULT_NUM_PAGES
        and config.get("page_size") == DEFAULT_PAGE_SIZE
        and config.get("max_num_batched_tokens") == MAX_NUM_BATCHED_TOKENS
        and config.get("top_logprobs") == TOP_LOGPROBS
        and config.get("selected_logprob_abs_tolerance") == LOGPROB_ABS_TOLERANCE
        and config.get("probe_own_logprob_abs_tolerance") == PROBE_OWN_LOGPROB_ABS_TOLERANCE
        and config.get("diagnostic_top_logprob_overlap_reference") == DIAGNOSTIC_TOP_LOGPROB_OVERLAP
        and config.get("prompt_token_ids_sha256") == EXPECTED_PROMPT_TOKEN_IDS_SHA256
        and config.get("free_running_exact_equality_is_binding") is False
        and config.get("post_divergence_distribution_comparison") is False
        and config.get("shared_top_logprob_comparison_is_binding") is False
        and config.get("first_divergence_cross_logprob_source")
        == "full_raw_logsoftmax_sparse_token_extract"
        and config.get("throughput_or_latency_gate") is False
    )
    requests_ok = _request_shape_ok(requests, vocab_size)
    run_keys_ok = isinstance(runs, dict) and set(runs) == {"tp1", "tp8"}
    tp1_shape_ok = (
        requests_ok
        and run_keys_ok
        and _run_shape_ok(
            runs["tp1"],
            tp=1,
            requests=requests,
            vocab_size=vocab_size,
        )
    )
    tp8_shape_ok = (
        requests_ok
        and run_keys_ok
        and _run_shape_ok(
            runs["tp8"],
            tp=8,
            requests=requests,
            vocab_size=vocab_size,
        )
    )
    tp1_ownership_ok = run_keys_ok and _sampling_ownership_ok(runs["tp1"], tp=1)
    tp8_ownership_ok = run_keys_ok and _sampling_ownership_ok(runs["tp8"], tp=8)
    if tp1_shape_ok and tp8_shape_ok:
        comparison_checks, summary, diagnostics = _compare_runs(
            runs["tp1"],
            runs["tp8"],
            requests,
            first_divergence_probes,
        )
    else:
        comparison_checks = {
            "aligned_prefix_selected_logprob_tolerance": False,
            "first_divergence_direct_cross_logprobs_complete": False,
            "first_divergence_selected_logprob_tolerance": False,
        }
        summary = {
            "requests": len(requests) if isinstance(requests, list) else 0,
            "tokens_per_degree": 0,
            "exact_free_running_token_parity_diagnostic": None,
            "exact_free_running_token_matches": None,
            "exact_free_running_token_match_rate": None,
            "token_mismatches": None,
            "requests_with_first_divergence": None,
            "first_divergence_direct_cross_logprob_probes": None,
            "common_prefix_matching_tokens": None,
            "aligned_selected_logprob_max_abs_delta": None,
            "first_divergence_selected_logprob_max_abs_delta": None,
            "aligned_shared_top_logprob_max_abs_delta_diagnostic": None,
            "aligned_minimum_top_logprob_token_overlap_diagnostic": None,
            "free_running_exact_equality_is_binding": False,
            "post_divergence_distribution_comparison_performed": False,
            "shared_top_logprob_comparison_is_binding": False,
        }
        diagnostics = [{"error": "raw run shape is invalid"}]

    checks = {
        "fixed_gate_configuration": config_ok,
        "clean_committed_source_and_script_binding": (_source_is_clean_and_bound(source)),
        "exact_eight_gpu_environment": _hardware_is_exact_eight_gpu(hardware),
        "full_qwen3_32b_checkpoint_provenance": (_checkpoint_is_qwen3_32b(checkpoint)),
        "fixed_natural_requests_and_sampling_matrix": requests_ok,
        "tp1_complete_raw_records": tp1_shape_ok,
        "tp8_complete_raw_records": tp8_shape_ok,
        "tp1_local_sampling_owner_topology": tp1_ownership_ok,
        "tp8_rank0_sampling_owner_gloo_nccl_topology": tp8_ownership_ok,
        **comparison_checks,
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
    from bench.parity_tp import _checkpoint_provenance

    _ensure_python_bin_on_path()
    source_start = _source_provenance()
    hardware = _hardware_provenance()
    if not _hardware_is_exact_eight_gpu(hardware):
        raise RuntimeError("formal ownership parity requires exactly eight visible CUDA GPUs")
    if args.assert_gate and not _source_snapshot_is_clean_and_bound(source_start):
        raise RuntimeError("formal evidence requires a clean commit containing every bound source")

    checkpoint = _checkpoint_provenance(args.model_path)
    if not _checkpoint_is_qwen3_32b(checkpoint):
        raise RuntimeError("--model-path is not the fully identified Qwen3-32B checkpoint")
    requests, request_evidence = _build_requests(args.model_path)

    runs = {}
    raw_logprobs = {}
    for tp in REQUIRED_TP_DEGREES:
        run, raw = _run_degree(
            args.model_path,
            tp,
            requests,
            args.num_pages,
            args.page_size,
        )
        runs[f"tp{tp}"] = run
        raw_logprobs[f"tp{tp}"] = raw
    first_divergence_probes = _build_first_divergence_probes(
        runs,
        raw_logprobs,
        request_evidence,
    )
    source_end = _source_provenance()
    source = {
        **source_start,
        "measurement_end": source_end,
        "stable_across_measurement": source_start == source_end,
    }
    if args.assert_gate and not _source_is_clean_and_bound(source):
        raise RuntimeError("bound source or commit changed while formal evidence was running")

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": source,
        "hardware": hardware,
        "checkpoint": checkpoint,
        "config": {
            "tp_degrees": list(REQUIRED_TP_DEGREES),
            "visible_gpu_gate": REQUIRED_VISIBLE_GPUS,
            "num_pages": args.num_pages,
            "page_size": args.page_size,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "top_logprobs": TOP_LOGPROBS,
            "selected_logprob_abs_tolerance": LOGPROB_ABS_TOLERANCE,
            "probe_own_logprob_abs_tolerance": (PROBE_OWN_LOGPROB_ABS_TOLERANCE),
            "diagnostic_top_logprob_overlap_reference": (DIAGNOSTIC_TOP_LOGPROB_OVERLAP),
            "prompt_token_ids_sha256": EXPECTED_PROMPT_TOKEN_IDS_SHA256,
            "free_running_exact_equality_is_binding": False,
            "post_divergence_distribution_comparison": False,
            "shared_top_logprob_comparison_is_binding": False,
            "first_divergence_cross_logprob_source": ("full_raw_logsoftmax_sparse_token_extract"),
            "throughput_or_latency_gate": False,
            "model_path_for_debugging_only": args.model_path,
        },
        "requests": request_evidence,
        "runs": runs,
        "first_divergence_cross_logprobs": first_divergence_probes,
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
                "diagnostics": diagnostics[:20],
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
    parser.add_argument("--num-pages", type=int, default=DEFAULT_NUM_PAGES)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--assert-gate",
        action="store_true",
        help="exit non-zero unless every binding check passes",
    )
    args = parser.parse_args()

    if args.verify is not None:
        if args.model_path is not None or args.output is not None:
            parser.error("--verify cannot be combined with --model-path or --output")
        artifact = json.loads(args.verify.read_text())
        checks, summary, diagnostics = verify_evidence(artifact, verify_stored=True)
        _print_verdict(checks, summary, diagnostics)
        return 1 if args.assert_gate and not all(checks.values()) else 0

    if args.model_path is None or args.output is None:
        parser.error("measurement requires both --model-path and --output")
    if args.num_pages != DEFAULT_NUM_PAGES:
        parser.error(f"formal gate fixes --num-pages at {DEFAULT_NUM_PAGES}, got {args.num_pages}")
    if args.page_size != DEFAULT_PAGE_SIZE:
        parser.error(f"formal gate fixes --page-size at {DEFAULT_PAGE_SIZE}, got {args.page_size}")

    artifact = _measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    _print_verdict(artifact["checks"], artifact["summary"], artifact["diagnostics"])
    print(f"written: {args.output}")
    return 1 if args.assert_gate and not artifact["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
