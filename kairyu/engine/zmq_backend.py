"""Process-split engine backend ("kairyu-proc", design m8 D6).

The engine core runs in a spawned child process (see
``kairyu.engine.core.engine_service``); this EngineBackend talks to it over a
``zmq.asyncio`` DEALER with msgpack framing. The socket and receiver task are
created lazily on first submit — ``build_app_from_spec`` constructs backends
before any event loop exists.

Lifecycle: ``shutdown()`` escalates — shutdown op → ``join(timeout)`` →
``terminate()`` → ``kill()`` — and an atexit guard covers non-lifespan
construction. A terminated child loses its coverage data, so tests must end
via the clean shutdown op.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import hashlib
import logging
import math
import os
import signal
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import TypeVar

from kairyu.async_thread import run_prompt_work, run_serialized_prompt_work
from kairyu.engine.backend import (
    EngineReadiness,
    GenerationRequest,
    GenerationResult,
    GenerationStageMetric,
    GenerationUsage,
    native_sampling_params,
    prompt_with_tool_intent,
    validate_backend_request_before_prepare,
    validate_native_request_surface,
    validate_native_request_surface_before_prepare,
)
from kairyu.engine.core.attention_selector import AttentionBackendDecision
from kairyu.engine.core.engine_service import (
    LEGACY_WIRE_VERSION,
    STARTUP_WIRE_VERSION,
    WIRE_VERSION,
    run_engine_service,
    sampling_params_to_wire,
)
from kairyu.engine.core.kv_cache_dtype import validate_kv_cache_dtype
from kairyu.engine.core.sampling_types import stable_request_seed
from kairyu.engine.engine_loop import (
    _validate_max_model_len,
    engine_sampling_from,
    validate_structured_sampling,
)
from kairyu.engine.prompt import (
    PromptInput,
    TemplatedPrompt,
    TokensPrompt,
    prompt_kind,
    prompt_text,
    prompt_to_wire,
    supplied_prompt_token_ids,
)
from kairyu.engine.registry import register_backend
from kairyu.engine.tokenizer import (
    Tokenizer,
    resolve_tokenizer,
    tokenize_loglikelihood_continuation,
    tokenizer_encode_is_concurrent_safe,
)
from kairyu.models.generation import (
    GenerationConfigMode,
    GenerationDefaults,
    validate_generation_config_mode,
)
from kairyu.outputs import CompletionOutput, TokenLogprob, _lazy_text
from kairyu.sampling_params import SamplingParams

_SPAWN_TIMEOUT_S = 30.0
_SHUTDOWN_TIMEOUT_S = 5.0
_FORCED_REAP_TIMEOUT_S = 5.0
_RECV_TICK_S = 1.0
_CONTROL_SEND_TIMEOUT_S = 1.0
_PROMPT_WIRE_VERSION = 1
_SPAWN_POLL_S = 0.05
_PR_SET_CHILD_SUBREAPER = 36
_SUBREAPER_PID: int | None = None
_T = TypeVar("_T")


def _decode_token_logprob(raw: list) -> TokenLogprob:
    token, token_id, logprob, bytes_, top = raw
    return TokenLogprob(
        token=token,
        token_id=token_id,
        logprob=logprob,
        bytes_=tuple(bytes_) if bytes_ is not None else None,
        top=tuple(_decode_token_logprob(entry) for entry in top),
    )


_STAGE_METRIC_WIRE_KEYS = frozenset(
    {"stage", "duration_ns", "occurrences"}
)


def _decode_stage_metrics(raw: object) -> tuple[GenerationStageMetric, ...]:
    """Validate the complete child-owned stage metric boundary."""

    if raw is None:
        return ()
    if isinstance(raw, tuple) and all(
        type(metric) is GenerationStageMetric for metric in raw
    ):
        return raw
    if not isinstance(raw, list):
        raise EngineServiceError("engine wire stage_metrics must be a list")
    metrics: list[GenerationStageMetric] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != _STAGE_METRIC_WIRE_KEYS:
            raise EngineServiceError(
                "engine wire stage metric must contain stage, duration_ns, "
                "and occurrences exactly"
            )
        try:
            metric = GenerationStageMetric(
                stage=item["stage"],
                duration_ns=item["duration_ns"],
                occurrences=item["occurrences"],
            )
        except (TypeError, ValueError) as error:
            raise EngineServiceError(
                "engine wire stage metric is invalid"
            ) from error
        if metric.stage in seen:
            raise EngineServiceError(
                f"engine wire stage_metrics contain duplicate stage {metric.stage!r}"
            )
        seen.add(metric.stage)
        metrics.append(metric)
    return tuple(metrics)


class EngineServiceError(RuntimeError):
    """The engine service process died or became unreachable."""


async def _send_control(socket, payload: bytes, *, timeout_s: float | None = None) -> None:
    """Bound a local DEALER write so failure and cleanup can always progress."""

    timeout = _CONTROL_SEND_TIMEOUT_S if timeout_s is None else timeout_s
    await asyncio.wait_for(socket.send(payload), timeout=timeout)


def _ensure_child_subreaper() -> None:
    """Make the API process the reaper for forcibly orphaned TP ranks."""

    global _SUBREAPER_PID
    current_pid = os.getpid()
    if (
        _SUBREAPER_PID == current_pid
        or os.name != "posix"
        or os.uname().sysname != "Linux"
        or not Path("/proc").is_dir()
    ):
        return
    # Linux containers commonly run the API as PID 1, but tests and
    # non-container deployments do not. A subreaper gives both layouts the
    # same ownership contract when rank 0 must be SIGKILLed. prctl is
    # idempotent; a PID marker (rather than a bool) also handles a later fork.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    if result != 0:
        error_number = ctypes.get_errno()
        raise EngineServiceError(
            "cannot establish engine-service child reaper: "
            f"{os.strerror(error_number)}"
        )
    _SUBREAPER_PID = current_pid


def _watch_parent_lifetime(lifetime_pipe) -> None:
    """Kill the complete service group when its API parent disappears."""

    try:
        lifetime_pipe.recv_bytes()
    except (EOFError, OSError):
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgrp(), signal.SIGKILL)
        else:  # pragma: no cover - production GPU deployments are POSIX
            os.kill(os.getpid(), signal.SIGKILL)


def _run_engine_service_in_owned_process_group(
    service,
    lifetime_pipe,
    *args,
) -> None:
    """Make one service and all of its distributed ranks one leased tree."""

    if os.name == "posix":
        os.setsid()
    threading.Thread(
        target=_watch_parent_lifetime,
        args=(lifetime_pipe,),
        name="kairyu-proc-parent-lease",
        daemon=True,
    ).start()
    service(*args)


def _signal_process_tree(process, sig: signal.Signals) -> None:
    """Signal the owned service process group, with a pre-setsid fallback."""

    pid = process.pid
    if os.name == "posix" and type(pid) is int and pid > 0:
        try:
            os.killpg(pid, sig)
            return
        except ProcessLookupError:
            # The child may not have reached setsid yet. It cannot have spawned
            # engine ranks before reporting its port, so PID-only fallback is
            # complete in that narrow startup window.
            pass
    if process.is_alive():
        with contextlib.suppress(ProcessLookupError):
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()


def _process_group_member_states(process) -> tuple[tuple[int, str], ...]:
    """Return every observable member of the owned group, including zombies."""

    pid = process.pid
    if os.name != "posix" or type(pid) is not int or pid < 1:
        return ((pid, "?"),) if process.is_alive() and type(pid) is int else ()
    proc = Path("/proc")
    if not proc.is_dir():  # pragma: no cover - POSIX without procfs
        return ((pid, "?"),) if process.is_alive() else ()
    members: list[tuple[int, str]] = []
    for stat_path in proc.glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text()
            suffix = raw[raw.rindex(")") + 2 :].split()
            state = suffix[0]
            process_group = int(suffix[2])
            member_pid = int(stat_path.parent.name)
        except (FileNotFoundError, IndexError, ProcessLookupError, ValueError):
            continue
        if process_group == pid:
            members.append((member_pid, state))
    return tuple(sorted(members))


def _live_process_group_members(process) -> tuple[int, ...]:
    """Return non-zombie members of the owned POSIX group when observable."""

    return tuple(
        pid
        for pid, state in _process_group_member_states(process)
        if state not in {"Z", "X"}
    )


def _reap_adopted_process_group_zombies(process) -> None:
    """Reap only orphaned descendants from this backend's private group."""

    service_pid = process.pid
    if os.name != "posix" or type(service_pid) is not int:
        return
    for member_pid, state in _process_group_member_states(process):
        if member_pid == service_pid or state != "Z":
            continue
        try:
            os.waitpid(member_pid, os.WNOHANG)
        except ChildProcessError:
            # Reparenting may still be racing the group signal. The bounded
            # caller retries and fails closed if the zombie never becomes ours.
            continue


def _wait_process_tree_exit(process, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        process.join(timeout=0)
        _reap_adopted_process_group_zombies(process)
        if not process.is_alive() and not _process_group_member_states(process):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        process.join(timeout=min(remaining, 0.05))


def _kill_and_reap_process(process, timeout_s: float) -> bool:
    """Kill one owned service tree and confirm every descendant is reaped."""

    _signal_process_tree(process, signal.SIGKILL)
    return _wait_process_tree_exit(process, timeout_s)


def _is_startup_value(value) -> bool:
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_startup_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and bool(key) and _is_startup_value(item)
            for key, item in value.items()
        )
    return False


def _attention_backend_decision_from_wire(raw) -> AttentionBackendDecision | None:
    """Validate startup metadata before exposing it through ``/backends``."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise EngineServiceError("engine startup attention_backend_decision must be a map or null")
    values = {}
    for field_name in ("requested", "resolved", "source", "rationale"):
        value = raw.get(field_name)
        if not isinstance(value, str) or not value:
            raise EngineServiceError(
                f"engine startup attention_backend_decision {field_name} must be a non-empty string"
            )
        values[field_name] = value
    components = raw.get("components")
    if not isinstance(components, dict) or not all(
        isinstance(key, str) and bool(key) and isinstance(value, str) and bool(value)
        for key, value in components.items()
    ):
        raise EngineServiceError(
            "engine startup attention_backend_decision components must be a string-to-string map"
        )
    architecture = raw.get("architecture", {})
    if not isinstance(architecture, dict) or not _is_startup_value(architecture):
        raise EngineServiceError(
            "engine startup attention_backend_decision architecture must be "
            "a JSON-compatible string-keyed map"
        )
    return AttentionBackendDecision(
        requested=values["requested"],
        resolved=values["resolved"],
        source=values["source"],
        components=dict(components),
        rationale=values["rationale"],
        architecture=dict(architecture),
    )


def _decision_from_startup_frame(frame) -> AttentionBackendDecision | None:
    if not isinstance(frame, dict):
        raise EngineServiceError("engine service startup metadata must be a map")
    version = frame.get("startup_wire_version")
    if type(version) is not int or version != STARTUP_WIRE_VERSION:
        raise EngineServiceError(
            f"unsupported engine startup wire version {version!r}; "
            f"supported version is {STARTUP_WIRE_VERSION}"
        )
    status = frame.get("status")
    if status == "error":
        error_type = frame.get("error_type")
        error = frame.get("error")
        if not isinstance(error_type, str) or not error_type:
            raise EngineServiceError("engine service startup error_type must be a non-empty string")
        if not isinstance(error, str):
            raise EngineServiceError("engine service startup error must be a string")
        raise EngineServiceError(f"engine service startup failed with {error_type}: {error}")
    if status != "ready":
        raise EngineServiceError(f"unknown engine service startup status {status!r}")
    return _attention_backend_decision_from_wire(frame.get("attention_backend_decision"))


def _kv_cache_dtype_from_startup_frame(
    frame,
) -> tuple[str | None, str | None]:
    """Validate optional KV metadata added compatibly to startup wire v1."""

    requested = frame.get("kv_cache_dtype_requested")
    resolved = frame.get("kv_cache_dtype_resolved")
    if requested is None and resolved is None:
        return None, None
    if not isinstance(requested, str) or not requested:
        raise EngineServiceError(
            "engine startup kv_cache_dtype_requested must be a non-empty string"
        )
    if not isinstance(resolved, str) or not resolved:
        raise EngineServiceError(
            "engine startup kv_cache_dtype_resolved must be a non-empty string"
        )
    try:
        validate_kv_cache_dtype(requested)
    except (ValueError, RuntimeError) as error:
        raise EngineServiceError(
            "engine startup kv_cache_dtype_requested is unavailable: "
            f"{error}"
        ) from error
    return requested, resolved


def _tensor_parallel_size_from_startup_frame(frame) -> int | None:
    """Read the optional runtime TP identity added compatibly to startup v1."""

    value = frame.get("tensor_parallel_size")
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise EngineServiceError(
            "engine startup tensor_parallel_size must be a positive integer"
        )
    return value


def _decode_graph_metadata_from_wire(raw) -> dict[str, object] | None:
    """Validate optional startup/live graph counters from the child."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise EngineServiceError("engine decode-graph metadata must be a map or null")
    decode_mode = raw.get("decode_mode")
    graph_decode = raw.get("cuda_graph_decode")
    buckets = raw.get("cuda_graph_buckets")
    captures = raw.get("cuda_graph_captures")
    replays = raw.get("cuda_graph_replays")
    fallbacks = raw.get("cuda_graph_eager_fallbacks")
    if (
        decode_mode not in {"eager", "cuda_graph"}
        or type(graph_decode) is not bool
        or graph_decode != (decode_mode == "cuda_graph")
        or not isinstance(buckets, (tuple, list))
        or any(type(bucket) is not int or bucket < 1 for bucket in buckets)
        or tuple(buckets) != tuple(sorted(set(buckets)))
        or any(
            type(count) is not int or count < 0
            for count in (captures, replays, fallbacks)
        )
        or (graph_decode and (not buckets or captures != len(buckets)))
        or (
            not graph_decode
            and (bool(buckets) or captures != 0 or replays != 0 or fallbacks != 0)
        )
    ):
        raise EngineServiceError("engine decode-graph metadata is malformed")
    return {
        "decode_mode": decode_mode,
        "cuda_graph_decode": graph_decode,
        "cuda_graph_buckets": tuple(buckets),
        "cuda_graph_captures": captures,
        "cuda_graph_replays": replays,
        "cuda_graph_eager_fallbacks": fallbacks,
    }


def _validate_startup_tensor_parallel_identity(
    configured: int,
    actual: int | None,
) -> None:
    """Fail closed before advertising a process-split distributed topology."""

    if actual is None:
        # A rolling-upgrade child predating this optional startup field can
        # only be trusted for the historical TP1 contract.  TP>1 must be
        # positively attested by the child that constructed the ranks.
        if configured != 1:
            raise EngineServiceError(
                "engine service did not report tensor-parallel startup metadata "
                f"for configured tensor_parallel_size={configured}"
            )
        return
    if actual != configured:
        raise EngineServiceError(
            "engine startup tensor-parallel request mismatch: "
            f"parent={configured}, child={actual}"
        )


def _generation_defaults_from_startup_frame(
    frame,
) -> GenerationDefaults | None:
    raw = frame.get("generation_defaults")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise EngineServiceError(
            "engine startup generation_defaults must be a map or null"
        )
    sampling = raw.get("sampling")
    stop_token_ids = raw.get("stop_token_ids")
    if not isinstance(sampling, dict) or set(sampling) != {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
    }:
        raise EngineServiceError(
            "engine startup generation sampling defaults are malformed"
        )
    if not isinstance(stop_token_ids, list) or any(
        type(token_id) is not int for token_id in stop_token_ids
    ):
        raise EngineServiceError(
            "engine startup generation stop_token_ids must be an integer list"
        )
    eos_token_id = raw.get("eos_token_id")
    if eos_token_id is not None and type(eos_token_id) is not int:
        raise EngineServiceError(
            "engine startup generation eos_token_id must be an integer or null"
        )
    try:
        mode = validate_generation_config_mode(raw.get("mode"))
        source = raw.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("generation source must be a non-empty string")
        defaults = GenerationDefaults(
            eos_token_id=eos_token_id,
            stop_token_ids=tuple(stop_token_ids),
            mode=mode,
            source=source,
            **sampling,
        )
        # Reuse the SamplingParams validation reached through apply.
        defaults.apply(
            SamplingParams().with_generation_config_omitted(sampling)
        )
    except (TypeError, ValueError) as error:
        raise EngineServiceError(
            "engine startup generation defaults are invalid"
        ) from error
    return defaults


def _validate_startup_kv_cache_dtype_identity(
    configured: str,
    requested: str | None,
    resolved: str | None,
) -> None:
    """Reject a child that did not construct the parent's requested KV policy."""

    if requested is None:
        if configured != "auto":
            raise EngineServiceError(
                "engine service did not report KV cache dtype metadata for "
                f"explicit request {configured!r}"
            )
        return
    if requested != configured:
        raise EngineServiceError(
            "engine startup KV cache dtype request mismatch: "
            f"parent={configured!r}, child={requested!r}"
        )
    allowed_resolutions = {
        "auto": frozenset(
            {
                "float16",
                "float32",
                "bfloat16",
                "not-applicable",
                "role-specific",
            }
        ),
        "bfloat16": frozenset({"bfloat16"}),
        "fp8_e4m3": frozenset({"fp8_e4m3"}),
    }
    if resolved not in allowed_resolutions[configured]:
        raise EngineServiceError(
            "engine startup KV cache dtype resolution is inconsistent: "
            f"requested={configured!r}, resolved={resolved!r}"
        )


def _dram_kv_tier_from_startup_frame(frame) -> dict[str, object] | None:
    raw = frame.get("dram_kv_tier")
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {
        "enabled",
        "capacity_pages",
        "profile_sha256",
        "min_restore_tokens",
    }:
        raise EngineServiceError("engine startup DRAM KV tier metadata is invalid")
    enabled = raw["enabled"]
    capacity = raw["capacity_pages"]
    profile_sha = raw["profile_sha256"]
    threshold = raw["min_restore_tokens"]
    if type(enabled) is not bool or type(capacity) is not int or capacity < 0:
        raise EngineServiceError("engine startup DRAM KV tier state is invalid")
    if enabled:
        if (
            capacity < 1
            or not isinstance(profile_sha, str)
            or len(profile_sha) != 64
            or any(character not in "0123456789abcdef" for character in profile_sha)
            or type(threshold) is not int
            or threshold < 1
        ):
            raise EngineServiceError(
                "engine startup enabled DRAM KV tier identity is incomplete"
            )
    elif capacity != 0 or profile_sha is not None or threshold is not None:
        raise EngineServiceError(
            "engine startup disabled DRAM KV tier reported active metadata"
        )
    return dict(raw)


def _validate_startup_dram_kv_tier_identity(
    configured_capacity_pages: int,
    actual: dict[str, object] | None,
    expected_profile_sha256: str | None = None,
) -> None:
    if actual is None:
        if configured_capacity_pages:
            raise EngineServiceError(
                "engine service did not report DRAM KV tier metadata for "
                "an enabled tier"
            )
        return
    expected_enabled = configured_capacity_pages > 0
    if (
        actual["enabled"] is not expected_enabled
        or actual["capacity_pages"] != configured_capacity_pages
    ):
        raise EngineServiceError(
            "engine startup DRAM KV tier request mismatch: "
            f"parent capacity={configured_capacity_pages}, child={actual}"
        )
    if (
        expected_profile_sha256 is not None
        and actual["profile_sha256"] != expected_profile_sha256
    ):
        raise EngineServiceError(
            "engine startup DRAM KV tier profile mismatch: "
            f"parent sha256={expected_profile_sha256}, "
            f"child sha256={actual['profile_sha256']}"
        )


def _recv_optional_startup_frame(
    parent_pipe,
    process,
) -> tuple[
    AttentionBackendDecision | None,
    str | None,
    str | None,
    dict[str, object] | None,
    GenerationDefaults | None,
    int | None,
    dict[str, object] | None,
]:
    """Receive a new child's second frame; EOF from a live child means legacy."""

    try:
        frame = parent_pipe.recv()
    except EOFError:
        if process.is_alive():
            # Old services close the Pipe immediately after the integer port.
            return None, None, None, None, None, None, None
        raise EngineServiceError(
            "engine service exited before reporting startup metadata"
        ) from None
    decision = _decision_from_startup_frame(frame)
    requested, resolved = _kv_cache_dtype_from_startup_frame(frame)
    dram_kv_tier = _dram_kv_tier_from_startup_frame(frame)
    generation_defaults = _generation_defaults_from_startup_frame(frame)
    tensor_parallel_size = _tensor_parallel_size_from_startup_frame(frame)
    graph_metadata = _decode_graph_metadata_from_wire(
        frame.get("decode_graph_metadata")
    )
    return (
        decision,
        requested,
        resolved,
        dram_kv_tier,
        generation_defaults,
        tensor_parallel_size,
        graph_metadata,
    )


def _stream_event_needs_materialization(event: Mapping[str, object]) -> bool:
    """Whether one wire event can change a public streaming result."""

    return (
        event.get("wire_version", LEGACY_WIRE_VERSION) != WIRE_VERSION
        or event.get("event") == "snapshot"
        or bool(event.get("new_token_ids"))
        or event.get("finished") is True
    )


@dataclass
class _WireAccumulator:
    """Reconstruct the public cumulative result from legacy or v2 events."""

    started: bool = False
    next_sequence: int = 0
    outputs: list[int] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    text_length: int = 0
    logprobs: list[dict] | None = None
    logprob_content: list[list] | None = None
    num_prompt_tokens: int = 0
    num_cached_tokens: int = 0
    cumulative_logprob: float = 0.0
    stage_metrics: tuple[GenerationStageMetric, ...] = ()
    wire_version: int | None = None

    @staticmethod
    def _require_offset(event: dict, name: str, expected: int) -> None:
        actual = event.get(name)
        if type(actual) is not int or actual != expected:
            raise EngineServiceError(
                f"engine wire delta {name} mismatch: expected {expected}, got {actual!r}"
            )

    def _snapshot(self, event: dict) -> None:
        if self.started:
            raise EngineServiceError("engine wire sent a duplicate snapshot")
        sequence = event.get("sequence")
        if type(sequence) is not int or sequence != 0:
            raise EngineServiceError(f"engine wire snapshot sequence must be 0, got {sequence!r}")
        outputs = event.get("outputs")
        text = event.get("text")
        if not isinstance(outputs, list) or not isinstance(text, str):
            raise EngineServiceError("engine wire snapshot has invalid outputs or text")
        self.outputs.extend(outputs)
        self.text_parts.append(text)
        self.text_length = len(text)
        if "logprobs" in event:
            if not isinstance(event["logprobs"], list):
                raise EngineServiceError("engine wire snapshot has invalid logprobs")
            self.logprobs = list(event["logprobs"])
        if "logprob_content" in event:
            if not isinstance(event["logprob_content"], list):
                raise EngineServiceError("engine wire snapshot has invalid logprob content")
            self.logprob_content = list(event["logprob_content"])
        self.num_prompt_tokens = event.get("num_prompt_tokens", 0)
        self.started = True
        self.next_sequence = 1

    def _delta(self, event: dict) -> None:
        if not self.started:
            raise EngineServiceError("engine wire sent a delta before its snapshot")
        sequence = event.get("sequence")
        if type(sequence) is not int or sequence != self.next_sequence:
            raise EngineServiceError(
                f"engine wire sequence mismatch: expected {self.next_sequence}, got {sequence!r}"
            )
        new_token_ids = event.get("new_token_ids")
        text_delta = event.get("text_delta")
        if not isinstance(new_token_ids, list) or not isinstance(text_delta, str):
            raise EngineServiceError("engine wire delta has invalid token IDs or text")
        self._require_offset(event, "output_offset", len(self.outputs))
        self.outputs.extend(new_token_ids)
        text_offset = event.get("text_offset")
        if type(text_offset) is not int or text_offset < 0 or text_offset > self.text_length:
            raise EngineServiceError(
                f"engine wire delta text_offset mismatch: expected a value in "
                f"[0, {self.text_length}], got {text_offset!r}"
            )
        if text_offset != self.text_length and event.get("finished") is not True:
            raise EngineServiceError("engine wire attempted a non-terminal visible-text rewrite")
        if text_offset == self.text_length:
            self.text_parts.append(text_delta)
        else:
            current = "".join(self.text_parts)
            self.text_parts = [current[:text_offset] + text_delta]
        self.text_length = text_offset + len(text_delta)

        has_new_logprobs = "new_logprobs" in event
        has_logprob_offset = "logprob_offset" in event
        if has_new_logprobs != has_logprob_offset:
            raise EngineServiceError("engine wire delta logprobs and offset must appear together")
        if self.logprobs is not None and not has_new_logprobs:
            raise EngineServiceError("engine wire delta omitted active logprob metadata")
        if has_new_logprobs:
            new_logprobs = event["new_logprobs"]
            if not isinstance(new_logprobs, list):
                raise EngineServiceError("engine wire delta has invalid logprobs")
            expected = len(self.logprobs or ())
            self._require_offset(event, "logprob_offset", expected)
            if self.logprobs is None:
                self.logprobs = []
            self.logprobs.extend(new_logprobs)
        has_new_content = "new_logprob_content" in event
        has_content_offset = "logprob_content_offset" in event
        if has_new_content != has_content_offset:
            raise EngineServiceError(
                "engine wire delta logprob content and offset must appear together"
            )
        if self.logprob_content is not None and not has_new_content:
            raise EngineServiceError("engine wire delta omitted active rich logprob metadata")
        if has_new_content:
            new_content = event["new_logprob_content"]
            if not isinstance(new_content, list):
                raise EngineServiceError("engine wire delta has invalid logprob content")
            expected = len(self.logprob_content or ())
            self._require_offset(event, "logprob_content_offset", expected)
            if self.logprob_content is None:
                self.logprob_content = []
            self.logprob_content.extend(new_content)
        self.next_sequence += 1

    def apply(self, event: dict, *, materialize: bool = True) -> dict:
        """Return one normalized cumulative event.

        A missing version (or explicit v1) is a response from an older service
        and remains byte-compatible. Version 2 is reconstructed and validated
        here; request-local state is discarded on cancellation/error.
        """

        wire_version = event.get("wire_version", LEGACY_WIRE_VERSION)
        if type(wire_version) is not int:
            raise EngineServiceError(f"invalid engine wire version {wire_version!r}")
        if self.wire_version is None:
            self.wire_version = wire_version
        elif wire_version != self.wire_version:
            raise EngineServiceError(
                f"engine wire changed version within one request: "
                f"{self.wire_version!r} -> {wire_version!r}"
            )
        if wire_version == LEGACY_WIRE_VERSION:
            return event
        if wire_version != WIRE_VERSION:
            raise EngineServiceError(f"unsupported engine wire version {wire_version!r}")
        event_type = event.get("event")
        if event_type == "snapshot":
            self._snapshot(event)
        elif event_type == "delta":
            self._delta(event)
        else:
            raise EngineServiceError(f"unknown engine wire event type {event_type!r}")

        self.num_cached_tokens = event.get(
            "num_cached_tokens",
            self.num_cached_tokens,
        )
        self.cumulative_logprob = event.get(
            "cumulative_logprob",
            self.cumulative_logprob,
        )
        if "stage_metrics" in event:
            # Metrics are cumulative snapshots. Omission by an older service
            # leaves the latest value unchanged; a request that never receives
            # the optional field therefore remains the required empty tuple.
            stage_metrics = _decode_stage_metrics(event["stage_metrics"])
            previous = {metric.stage: metric for metric in self.stage_metrics}
            current = {metric.stage: metric for metric in stage_metrics}
            for stage, old in previous.items():
                new = current.get(stage)
                if new is None:
                    raise EngineServiceError(
                        f"engine wire stage metric {stage!r} disappeared"
                    )
                if (
                    new.duration_ns < old.duration_ns
                    or new.occurrences < old.occurrences
                ):
                    raise EngineServiceError(
                        f"engine wire stage metric {stage!r} decreased"
                    )
            self.stage_metrics = stage_metrics
        normalized = {
            "request_id": event["request_id"],
            "outputs": self.outputs,
            "finished": event["finished"],
            "finish_reason": event.get("finish_reason"),
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "cumulative_logprob": self.cumulative_logprob,
            "stage_metrics": self.stage_metrics,
        }
        if event_type == "snapshot":
            normalized["text_offset"] = 0
            normalized["text_delta"] = event["text"]
        else:
            normalized["text_offset"] = event["text_offset"]
            normalized["text_delta"] = event["text_delta"]
        if materialize:
            normalized["text"] = "".join(self.text_parts)
        else:
            normalized["text"] = _lazy_text(self.text_parts)
        if self.logprobs is not None:
            normalized["logprobs"] = self.logprobs
        if self.logprob_content is not None:
            normalized["logprob_content"] = self.logprob_content
        return normalized


@dataclass(frozen=True)
class _PreparedProcPrompt:
    prompt: PromptInput
    tokenize_duration_ns: int | None = None


@dataclass
class _ProcPromptPreparationFlight:
    """One pure parent-preflight computation shared by async callers."""

    request_ref: weakref.ReferenceType[GenerationRequest]
    task: asyncio.Task[_PreparedProcPrompt]
    waiters: int = 0


def _import_deps():
    try:
        import msgpack
        import zmq
        import zmq.asyncio
    except ImportError as error:  # pragma: no cover - exercised only without deps
        raise RuntimeError(
            "the kairyu-proc backend requires pyzmq and msgpack (uv sync --extra fleet)"
        ) from error
    return zmq, msgpack


class ZmqEngineBackend:
    """EngineBackend over a spawned engine-service process.

    ``supports_n = False``: the server validates n>1 per backend and returns
    400 (m9 D3 review — a backend exception would surface as 502).

    ``tokenizer`` must be a string ("toy" or a tokenizer path): the config
    crosses a process boundary. Custom runner objects cannot cross either —
    the service builds its own (real model runners arrive with M12 configs).
    """

    supports_n = False  # revisited in M11

    def __init__(
        self,
        num_pages: int = 4096,
        page_size: int = 16,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 256,
        priority_age_s: float | None = 60.0,
        tokenizer: str | None = None,
        speculative: str | None = None,
        speculative_tokens: int = 4,
        death_timeout_s: float = 120.0,
        model_path: str | None = None,
        pipeline_depth: int = 1,
        decode_mode: str | None = None,
        cuda_graph_max_batch: int | None = None,
        cuda_graph_max_pages: int | None = None,
        cuda_graph_warmup_iters: int = 3,
        max_model_len: int | None = None,
        kv_cache_dtype: str = "auto",
        dram_kv_tier_capacity_pages: int = 0,
        dram_kv_tier_profile: str | PathLike[str] | None = None,
        generation_config: GenerationConfigMode = "auto",
        tensor_parallel_size: int = 1,
        max_num_partial_prefills: int = 2,
        draft_model_path: str | PathLike[str] | None = None,
        nvfp4_accuracy_profile=None,
    ) -> None:
        if tokenizer is not None and not isinstance(tokenizer, str):
            raise ValueError("kairyu-proc requires a string tokenizer (name or path)")
        if draft_model_path is not None and (
            not isinstance(draft_model_path, (str, PathLike))
            or not str(draft_model_path)
        ):
            raise ValueError(
                "kairyu-proc draft_model_path must be a non-empty local path or null"
            )
        if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be a positive integer")
        if model_path is None:
            from kairyu.engine.core.tp_runner import validate_tp_degree

            validate_tp_degree(tensor_parallel_size)
        if (
            isinstance(death_timeout_s, bool)
            or not isinstance(death_timeout_s, (int, float))
            or not math.isfinite(death_timeout_s)
            or death_timeout_s <= 0
        ):
            raise ValueError("death_timeout_s must be a positive finite number")
        _validate_max_model_len(max_model_len)
        generation_config = validate_generation_config_mode(generation_config)
        from kairyu.engine.core.nvfp4_accuracy import NvFp4AccuracyProfile

        accuracy_profile = NvFp4AccuracyProfile.parse(nvfp4_accuracy_profile)
        if accuracy_profile.active and model_path is None:
            raise ValueError(
                "kairyu-proc NVFP4 accuracy profile requires a real model_path"
            )
        kv_cache_dtype = validate_kv_cache_dtype(kv_cache_dtype)
        if kv_cache_dtype != "auto" and model_path is None:
            raise ValueError(
                "kairyu-proc explicit KV cache dtype requires a real model_path"
            )
        if (
            type(dram_kv_tier_capacity_pages) is not int
            or dram_kv_tier_capacity_pages < 0
        ):
            raise ValueError(
                "dram_kv_tier_capacity_pages must be a non-negative integer"
            )
        if dram_kv_tier_profile is not None and (
            not isinstance(dram_kv_tier_profile, (str, PathLike))
            or not str(dram_kv_tier_profile)
        ):
            raise ValueError(
                "dram_kv_tier_profile must be a non-empty local path or null"
            )
        if (dram_kv_tier_capacity_pages > 0) != (
            dram_kv_tier_profile is not None
        ):
            raise ValueError(
                "DRAM KV tier requires both a positive "
                "dram_kv_tier_capacity_pages and dram_kv_tier_profile"
            )
        if dram_kv_tier_capacity_pages and model_path is None:
            raise ValueError("DRAM KV tier requires a real model_path")
        dram_kv_tier_profile_bytes: bytes | None = None
        dram_kv_tier_expected_profile_sha256: str | None = None
        if dram_kv_tier_profile is not None:
            profile_path = Path(dram_kv_tier_profile)
            try:
                dram_kv_tier_profile_bytes = profile_path.read_bytes()
            except OSError as error:
                raise ValueError(
                    f"cannot read DRAM KV crossover profile: {profile_path}"
                ) from error
            if not dram_kv_tier_profile_bytes:
                raise ValueError(
                    f"DRAM KV crossover profile is empty: {profile_path}"
                )
            dram_kv_tier_expected_profile_sha256 = hashlib.sha256(
                dram_kv_tier_profile_bytes
            ).hexdigest()
        self._config = {
            "num_pages": num_pages,
            "page_size": page_size,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_partial_prefills": max_num_partial_prefills,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "priority_age_s": priority_age_s,
            "tensor_parallel_size": tensor_parallel_size,
            "tokenizer": tokenizer,
            "speculative": speculative,
            "speculative_tokens": speculative_tokens,
            "draft_model_path": (
                str(draft_model_path) if draft_model_path is not None else None
            ),
            "model_path": model_path,
            "pipeline_depth": pipeline_depth,
            "decode_mode": decode_mode,
            "cuda_graph_max_batch": cuda_graph_max_batch,
            "cuda_graph_max_pages": cuda_graph_max_pages,
            "cuda_graph_warmup_iters": cuda_graph_warmup_iters,
            "kv_cache_dtype": kv_cache_dtype,
            "dram_kv_tier_capacity_pages": dram_kv_tier_capacity_pages,
            "dram_kv_tier_profile": (
                str(dram_kv_tier_profile)
                if dram_kv_tier_profile is not None
                else None
            ),
            "generation_config": generation_config,
            "nvfp4_accuracy_profile": (
                accuracy_profile.as_dict() if accuracy_profile.active else None
            ),
        }
        self._sequence_budget = max_num_seqs
        self._death_timeout_s = death_timeout_s
        self._configured_tensor_parallel_size = tensor_parallel_size
        # Public topology is populated only after the child reports a matching
        # runtime identity. A configured value alone is not an attestation.
        self.tensor_parallel_size: int | None = None
        self._process = None
        self._lifetime_pipe = None
        self._socket = None
        self._context = None
        self._receiver: asyncio.Task | None = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._stream_ids: dict[str, str] = {}
        self._wire_request_ids: dict[str, str] = {}
        self._public_request_ids: dict[str, str] = {}
        self._queue_generations: dict[str, int] = {}
        self._parent_tokenize_durations_ns: dict[str, int] = {}
        self._failed_generations: set[int] = set()
        self._generation_counter = 0
        self._live_generation: int | None = None
        self._service_failure_type: str | None = None
        self._wire_version = WIRE_VERSION
        self._active_request_ids: set[str] = set()
        self.attention_backend_decision: AttentionBackendDecision | None = None
        self._decode_graph_metadata: dict[str, object] | None = None
        self._configured_generation_defaults: GenerationDefaults | None = (
            GenerationDefaults(
                mode=generation_config,
                source=(
                    "builtin"
                    if generation_config == "auto"
                    else "disabled"
                    if generation_config == "none"
                    else "vllm"
                ),
            )
            if model_path is None
            else None
        )
        self.generation_defaults: GenerationDefaults | None = (
            self._configured_generation_defaults
        )
        self.kv_cache_dtype_requested = kv_cache_dtype
        self.kv_cache_dtype_resolved: str | None = None
        self.dram_kv_tier_enabled = dram_kv_tier_capacity_pages > 0
        self.dram_kv_tier_capacity_pages = dram_kv_tier_capacity_pages
        self.dram_kv_tier_profile = (
            str(dram_kv_tier_profile)
            if dram_kv_tier_profile is not None
            else None
        )
        self.dram_kv_tier_profile_sha256: str | None = None
        self.dram_kv_tier_min_restore_tokens: int | None = None
        # The path is operator-facing configuration, but it is not a stable
        # process-boundary payload.  Pin its bytes at backend construction so
        # every child generation validates exactly the same measured policy.
        self._dram_kv_tier_profile_bytes = dram_kv_tier_profile_bytes
        self._dram_kv_tier_expected_profile_sha256 = (
            dram_kv_tier_expected_profile_sha256
        )
        self._start_lock = asyncio.Lock()
        self._startup_task: asyncio.Task[int] | None = None
        self._startup_abandoned: threading.Event | None = None
        self._process_reap_owner = None
        self._process_reap_task: asyncio.Task[bool] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._closed = False
        self._atexit_registered = False
        self._max_model_len = max_model_len
        # A configured context limit must be enforceable by the HTTP-facing
        # parent before a StreamingResponse commits SSE headers. All text is
        # also resolved here before crossing the process boundary, even when
        # no limit is configured. Lazily load the exact child tokenizer source
        # so construction and config-only tooling stay side-effect compatible.
        self._preflight_tokenizer_source = (
            tokenizer
            if tokenizer is not None
            else model_path
            if model_path is not None
            else "toy"
        )
        self._request_validation_contract = (
            model_path,
            self._preflight_tokenizer_source,
            max_model_len,
        )
        self._preflight_tokenizer: Tokenizer | None = None
        self._preflight_tokenizer_lock = threading.Lock()
        # Validation may be called more than once by wrapping orchestration
        # layers. Retain only this exact immutable request object, consume the
        # prepared token prompt at submit, and let an abandoned request's weak
        # reference reclaim the entry.
        self._prepared_requests: dict[
            int,
            tuple[weakref.ReferenceType[GenerationRequest], _PreparedProcPrompt],
        ] = {}
        self._prepared_requests_lock = threading.Lock()
        # Unknown compatibility tokenizers queue on the event loop, leaving
        # prompt executor threads available to other backends while they wait.
        self._prompt_tokenizer_gate = asyncio.Lock()
        # Event-loop-owned single flights.  Workers only compute; a live
        # waiter publishes into `_prepared_requests` after a successful await.
        self._preparing_requests: dict[int, _ProcPromptPreparationFlight] = {}

    @property
    def sequence_budget(self) -> int:
        """Construction-time active sequence capacity for HTTP admission."""

        return self._sequence_budget

    @property
    def request_validation_key(
        self,
    ) -> tuple[str | None, str, int | None] | None:
        """Identity for process replicas with equal parent validation."""

        if type(self) is not ZmqEngineBackend:
            return None
        return self._request_validation_contract

    # -- lifecycle ---------------------------------------------------------

    def _validate_dram_kv_profile_unchanged(self) -> None:
        """Refuse a respawn after the configured profile path changes."""

        expected = self._dram_kv_tier_profile_bytes
        if expected is None:
            return
        profile_path = self.dram_kv_tier_profile
        assert profile_path is not None
        try:
            current = Path(profile_path).read_bytes()
        except OSError as error:
            raise EngineServiceError(
                "configured DRAM KV profile became unreadable after backend "
                f"construction: {profile_path}"
            ) from error
        if current != expected:
            raise EngineServiceError(
                "configured DRAM KV profile changed after backend construction; "
                "recreate the backend to adopt a new measured policy"
            )

    def _spawn(self, abandoned: threading.Event | None = None) -> int:
        import multiprocessing

        if abandoned is not None and abandoned.is_set():
            raise EngineServiceError("engine service startup was cancelled")
        self._validate_dram_kv_profile_unchanged()
        # Never let a replacement child inherit sampling/audit state from the
        # prior process. A versioned startup record replaces this placeholder
        # before the new generation becomes live.
        self.generation_defaults = self._configured_generation_defaults
        if self._configured_tensor_parallel_size > 1:
            _ensure_child_subreaper()
        spawn = multiprocessing.get_context("spawn")
        parent_pipe, child_pipe = spawn.Pipe()
        child_lifetime_pipe, parent_lifetime_pipe = spawn.Pipe(duplex=False)
        service_args = (child_pipe, self._config)
        if self._dram_kv_tier_profile_bytes is not None:
            service_args = (*service_args, self._dram_kv_tier_profile_bytes)
        process = spawn.Process(
            target=_run_engine_service_in_owned_process_group,
            args=(run_engine_service, child_lifetime_pipe, *service_args),
            # A real TP service is itself the parent of rank workers. Python
            # forbids a daemon process from creating children, so distributed
            # process isolation must use an explicitly owned non-daemon
            # service. TP1 retains the historical daemon behavior.
            daemon=self._configured_tensor_parallel_size == 1,
        )
        try:
            process.start()
        except BaseException:
            child_lifetime_pipe.close()
            parent_lifetime_pipe.close()
            parent_pipe.close()
            child_pipe.close()
            raise
        # Register ownership before either Pipe wait.  If the coroutine awaiting
        # this worker thread is cancelled, its cleanup can now find and kill the
        # child instead of leaving an unowned model load behind.
        self._process = process
        self._lifetime_pipe = parent_lifetime_pipe
        if not self._atexit_registered:
            # Register as soon as the PID/group is owned. Model loading and TP
            # startup happen after this point and may outlive the initiating
            # coroutine or interpreter shutdown.
            atexit.register(self._kill_process)
            self._atexit_registered = True
        child_pipe.close()
        child_lifetime_pipe.close()
        try:
            port_deadline = time.monotonic() + _SPAWN_TIMEOUT_S
            while not parent_pipe.poll(_SPAWN_POLL_S):
                if abandoned is not None and abandoned.is_set():
                    raise EngineServiceError("engine service startup was cancelled")
                if not process.is_alive():
                    raise EngineServiceError("engine service exited before reporting its port")
                if time.monotonic() >= port_deadline:
                    raise EngineServiceError("engine service did not report its port in time")
            try:
                port = parent_pipe.recv()
            except EOFError:
                raise EngineServiceError(
                    "engine service exited before reporting its port"
                ) from None
            if type(port) is not int or not 1 <= port <= 65535:
                raise EngineServiceError(f"engine service reported an invalid port {port!r}")
            # Model load is deliberately outside the 30-second port timeout.
            # Poll rather than blocking in recv() so cancellation can reclaim a
            # Qwen-sized load promptly and cannot later publish stale metadata.
            while not parent_pipe.poll(_SPAWN_POLL_S):
                if abandoned is not None and abandoned.is_set():
                    raise EngineServiceError("engine service startup was cancelled")
                if not process.is_alive():
                    raise EngineServiceError(
                        "engine service exited before reporting startup metadata"
                    )
            (
                decision,
                requested_kv_dtype,
                resolved_kv_dtype,
                dram_kv_tier,
                generation_defaults,
                actual_tensor_parallel_size,
                graph_metadata,
            ) = (
                _recv_optional_startup_frame(parent_pipe, process)
            )
            _validate_startup_tensor_parallel_identity(
                self._configured_tensor_parallel_size,
                actual_tensor_parallel_size,
            )
            _validate_startup_kv_cache_dtype_identity(
                self.kv_cache_dtype_requested,
                requested_kv_dtype,
                resolved_kv_dtype,
            )
            _validate_startup_dram_kv_tier_identity(
                self.dram_kv_tier_capacity_pages,
                dram_kv_tier,
                self._dram_kv_tier_expected_profile_sha256,
            )
            if (
                generation_defaults is not None
                and generation_defaults.mode
                != self._config["generation_config"]
            ):
                raise EngineServiceError(
                    "engine startup generation_config request mismatch"
                )
            if (
                generation_defaults is None
                and self._config.get("model_path") is not None
            ):
                raise EngineServiceError(
                    "engine startup omitted generation defaults for a configured model"
                )
            # Catch an operator rewrite that raced this respawn.  The child
            # still consumed the pinned bytes, so this is detection rather
            # than a data race; rejecting readiness makes the config change
            # explicit instead of silently keeping the old policy alive.
            self._validate_dram_kv_profile_unchanged()
            if abandoned is not None and abandoned.is_set():
                raise EngineServiceError("engine service startup was cancelled")
            if not process.is_alive():
                raise EngineServiceError("engine service exited after reporting startup readiness")
            if self._process is not process:
                raise EngineServiceError("engine service startup ownership was lost")
        except BaseException as error:
            self.attention_backend_decision = None
            self._decode_graph_metadata = None
            self.tensor_parallel_size = None
            self.kv_cache_dtype_resolved = None
            self.dram_kv_tier_profile_sha256 = None
            self.dram_kv_tier_min_restore_tokens = None
            if self._lifetime_pipe is parent_lifetime_pipe:
                self._lifetime_pipe.close()
                self._lifetime_pipe = None
            reaped = _kill_and_reap_process(
                process,
                timeout_s=_FORCED_REAP_TIMEOUT_S,
            )
            if self._process is process and reaped:
                self._process = None
            if not reaped:
                raise EngineServiceError(
                    "cancelled engine service startup left a live child; "
                    "refusing to release process ownership"
                ) from error
            raise
        finally:
            parent_pipe.close()
        self.attention_backend_decision = decision
        self._decode_graph_metadata = graph_metadata
        self.tensor_parallel_size = (
            actual_tensor_parallel_size
            if actual_tensor_parallel_size is not None
            else self._configured_tensor_parallel_size
        )
        if generation_defaults is not None:
            self.generation_defaults = generation_defaults
        if requested_kv_dtype is not None:
            self.kv_cache_dtype_requested = requested_kv_dtype
        self.kv_cache_dtype_resolved = resolved_kv_dtype
        if dram_kv_tier is not None:
            profile_sha = dram_kv_tier["profile_sha256"]
            threshold = dram_kv_tier["min_restore_tokens"]
            self.dram_kv_tier_profile_sha256 = (
                profile_sha if isinstance(profile_sha, str) else None
            )
            self.dram_kv_tier_min_restore_tokens = (
                threshold if type(threshold) is int else None
            )
        return port

    def _is_healthy(self) -> bool:
        return (
            self._service_failure_type is None
            and not self._closed
            and self._socket is not None
            and self._receiver is not None
            and not self._receiver.done()
            and self._process is not None
            and self._process.is_alive()
            and self._live_generation is not None
        )

    def _retire_unreliable_transport(
        self,
        expected_generation: int | None = None,
        expected_socket: object | None = None,
    ) -> None:
        """Stop a generation after a data-plane send has unknown delivery."""

        if self._closed:
            return
        if (
            expected_generation is not None
            and self._live_generation != expected_generation
        ) or (
            expected_socket is not None
            and self._socket is not expected_socket
        ):
            # A replacement generation must never be killed for an uncertain
            # send owned by its already-retired predecessor.
            return
        self._service_failure_type = "EngineServiceError"
        process = self._process
        if process is not None:
            _signal_process_tree(process, signal.SIGKILL)

    def readiness(self) -> EngineReadiness:
        """Report a known-dead service generation without running a probe."""

        if self._closed:
            return EngineReadiness(False, "engine service is shut down")
        if self._service_failure_type is not None:
            return EngineReadiness(
                False,
                f"engine service failed: {self._service_failure_type}",
                fatal=True,
            )
        process = self._process
        if process is not None and not process.is_alive():
            return EngineReadiness(
                False,
                "engine service process is not running",
                fatal=True,
            )
        receiver = self._receiver
        if self._live_generation is not None and (
            receiver is None or receiver.done()
        ):
            return EngineReadiness(
                False,
                "engine service receiver is not running",
                fatal=True,
            )
        return EngineReadiness(True)

    def decode_graph_metadata_snapshot(self) -> dict[str, object] | None:
        """Return the latest graph counters reported by the live child."""

        metadata = self._decode_graph_metadata
        return dict(metadata) if isinstance(metadata, Mapping) else None

    def decode_graph_metric_generation_snapshot(self) -> int | None:
        """Identify the child-service generation that owns graph counters."""

        return self._live_generation

    def _fail_generation_once(
        self,
        generation: int | None,
        error: BaseException,
    ) -> None:
        """Wake every request owned by one failed service generation exactly once."""

        if generation is None or generation in self._failed_generations:
            return
        targets = [
            queue
            for request_id, queue in tuple(self._queues.items())
            if self._queue_generations.get(request_id) == generation
        ]
        if not targets:
            # live_generation is cleared before every caller reaches here, so
            # no request can subsequently join this retired generation.
            return
        self._failed_generations.add(generation)
        event = {"error": repr(error)}
        for queue in targets:
            queue.put_nowait(dict(event))

    @staticmethod
    async def _drain_owned_task(
        task: asyncio.Task[_T],
    ) -> tuple[_T, asyncio.CancelledError | None]:
        """Wait through caller cancellation and return it after owned cleanup."""

        if task.done():
            return task.result(), None
        cancelled: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if cancelled is None:
                    cancelled = error
                continue
            except BaseException:
                # ``result()`` below preserves the cleanup exception and chains
                # it behind caller cancellation when both happened.
                break
        try:
            result = task.result()
        except BaseException as cleanup_error:
            if cancelled is not None:
                raise cancelled from cleanup_error
            raise
        return result, cancelled

    @classmethod
    async def _await_owned_task(cls, task: asyncio.Task[_T]) -> _T:
        result, cancelled = await cls._drain_owned_task(task)
        if cancelled is not None:
            raise cancelled
        return result

    async def _run_process_reap(self, process, timeout_s: float) -> bool:
        reaped = await asyncio.to_thread(
            _wait_process_tree_exit,
            process,
            timeout_s,
        )
        if reaped and self._process is process:
            # Clear ownership in the task that actually proved the complete
            # group gone. Callers can be cancelled without leaving a stale PID
            # handle that a later reset might signal after PID/PGID reuse.
            self._process = None
        return reaped

    def _shared_process_reap_task(
        self,
        process,
        timeout_s: float,
    ) -> asyncio.Task[bool]:
        task = self._process_reap_task
        owner = self._process_reap_owner
        if task is not None and not task.done():
            if owner is not process:
                raise EngineServiceError(
                    "cannot reap a new engine service while prior cleanup is active"
                )
            return task
        task = asyncio.create_task(self._run_process_reap(process, timeout_s))
        self._process_reap_owner = process
        self._process_reap_task = task
        return task

    async def _wait_for_process_reap(
        self,
        process,
        timeout_s: float,
    ) -> bool:
        return await self._await_owned_task(
            self._shared_process_reap_task(process, timeout_s)
        )

    async def _cancel_receiver_locked(self) -> asyncio.CancelledError | None:
        """Cancel one receiver, but drain its owned cleanup before returning."""

        receiver = self._receiver
        self._receiver = None
        if receiver is None:
            return None
        if not receiver.done():
            receiver.cancel()
        caller = asyncio.current_task()
        cancelled: asyncio.CancelledError | None = None
        while not receiver.done():
            try:
                await asyncio.shield(receiver)
            except asyncio.CancelledError as error:
                if caller is not None and caller.cancelling() and cancelled is None:
                    cancelled = error
                continue
            except Exception:
                break
        with contextlib.suppress(asyncio.CancelledError, Exception):
            receiver.result()
        return cancelled

    def _close_transport_locked(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        if self._lifetime_pipe is not None:
            self._lifetime_pipe.close()
            self._lifetime_pipe = None

    async def _reset_dead_locked(self) -> None:
        """Tear down a crashed child's stale socket/context/process (E1).

        A receiver that exited (child death or fatal frame) leaves ``_socket``
        set, so the old ``_ensure_started`` returned early and every subsequent
        request awaited a queue nothing would ever fill. Clearing the dead refs
        lets the next request spawn a fresh service.
        """
        generation = self._live_generation
        self._live_generation = None
        self.attention_backend_decision = None
        self._decode_graph_metadata = None
        self.tensor_parallel_size = None
        self.kv_cache_dtype_resolved = None
        self.dram_kv_tier_profile_sha256 = None
        self.dram_kv_tier_min_restore_tokens = None
        self._fail_generation_once(
            generation,
            EngineServiceError(
                "engine service became unavailable and was reset before completing the request"
            ),
        )
        cancelled = await self._cancel_receiver_locked()
        self._close_transport_locked()
        process = self._process
        if process is not None:
            _signal_process_tree(process, signal.SIGKILL)
            task = self._shared_process_reap_task(
                process,
                _FORCED_REAP_TIMEOUT_S,
            )
            reaped, reap_cancelled = await self._drain_owned_task(task)
            if cancelled is None:
                cancelled = reap_cancelled
            if not reaped:
                # Retaining the handle prevents a second GPU-owning child from
                # being spawned over an uninterruptible old process.
                cleanup_error = EngineServiceError(
                    "engine service child did not exit after kill; refusing to respawn"
                )
                if cancelled is not None:
                    raise cancelled from cleanup_error
                raise cleanup_error
        if cancelled is not None:
            raise cancelled

    async def _ensure_started(self) -> None:
        if self._closed:
            raise EngineServiceError("kairyu-proc backend is shut down")
        if self._is_healthy():
            return
        async with self._start_lock:
            if self._closed:
                raise EngineServiceError("kairyu-proc backend is shut down")
            if self._is_healthy():
                return
            await self._reset_dead_locked()  # respawn over a crashed child (E1)
            if self._closed:
                raise EngineServiceError("kairyu-proc backend is shut down")
            try:
                zmq, _ = _import_deps()
                abandoned = threading.Event()
                spawn_task = asyncio.create_task(asyncio.to_thread(self._spawn, abandoned))
                self._startup_abandoned = abandoned
                self._startup_task = spawn_task
                try:
                    # Shield is essential: cancelling this coroutine must signal
                    # and drain the worker, not mark its asyncio wrapper done while
                    # the underlying thread later publishes an orphan process.
                    port = await asyncio.shield(spawn_task)
                except asyncio.CancelledError:
                    abandoned.set()
                    while not spawn_task.done():
                        try:
                            await asyncio.shield(spawn_task)
                        except asyncio.CancelledError:
                            # Repeated cancellation still cannot bypass ownership
                            # cleanup; the poll loop observes the same event.
                            abandoned.set()
                        except BaseException:
                            # The expected worker-side cancellation error is
                            # consumed below; the caller still observes its
                            # original asyncio cancellation.
                            break
                    try:
                        spawn_task.result()
                    except BaseException:
                        pass
                    raise
                finally:
                    if self._startup_task is spawn_task:
                        self._startup_task = None
                    if self._startup_abandoned is abandoned:
                        self._startup_abandoned = None
                if self._closed:
                    raise EngineServiceError("kairyu-proc backend was shut down during startup")
                self._context = zmq.asyncio.Context()
                socket = self._context.socket(zmq.DEALER)
                socket.connect(f"tcp://127.0.0.1:{port}")
                self._socket = socket
                self._generation_counter += 1
                generation = self._generation_counter
                self._live_generation = generation
                process = self._process
                assert process is not None
                self._service_failure_type = None
                self._receiver = asyncio.get_running_loop().create_task(
                    self._receive_loop(generation, socket, process)
                )
            except BaseException as error:
                if not self._closed and not isinstance(error, asyncio.CancelledError):
                    self._service_failure_type = type(error).__name__
                await self._reset_dead_locked()
                raise

    async def startup(self) -> None:
        """Finish child construction before an owning app begins serving."""

        await self._ensure_started()

    def _kill_process(self) -> None:
        process = self._process
        if process is not None:  # pragma: no cover - interpreter crash path
            if self._lifetime_pipe is not None:
                self._lifetime_pipe.close()
            _signal_process_tree(process, signal.SIGKILL)
            _wait_process_tree_exit(process, _FORCED_REAP_TIMEOUT_S)

    async def _shutdown_impl(self) -> None:
        preparation_tasks = tuple(
            flight.task for flight in self._preparing_requests.values()
        )
        if preparation_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in preparation_tasks),
                return_exceptions=True,
            )
        self._preparing_requests.clear()
        async with self._start_lock:
            generation = self._live_generation
            self._live_generation = None
            self.attention_backend_decision = None
            self._decode_graph_metadata = None
            self.tensor_parallel_size = None
            self.kv_cache_dtype_resolved = None
            self.dram_kv_tier_profile_sha256 = None
            self.dram_kv_tier_min_restore_tokens = None
            self._fail_generation_once(
                generation,
                EngineServiceError("engine service shut down before completing the request"),
            )
            receiver_cancelled = await self._cancel_receiver_locked()

            process = self._process
            cleanup_error: EngineServiceError | None = None
            try:
                if process is not None and self._socket is not None:
                    _, msgpack = _import_deps()
                    try:
                        await _send_control(
                            self._socket,
                            msgpack.packb({"op": "shutdown"}),
                        )
                    except Exception:
                        # A wedged ROUTER must not prevent process-group
                        # escalation below.
                        pass
                if process is not None:
                    tree_exited = await self._wait_for_process_reap(
                        process,
                        _SHUTDOWN_TIMEOUT_S,
                    )
                    if not tree_exited:
                        _signal_process_tree(process, signal.SIGTERM)
                        tree_exited = await self._wait_for_process_reap(
                            process,
                            2.0,
                        )
                    if not tree_exited:
                        _signal_process_tree(process, signal.SIGKILL)
                        tree_exited = await self._wait_for_process_reap(
                            process,
                            2.0,
                        )
                    if not tree_exited:
                        cleanup_error = EngineServiceError(
                            "engine service process group was not completely "
                            "reaped after shutdown escalation"
                        )
            finally:
                self._close_transport_locked()
                self.attention_backend_decision = None
                self._decode_graph_metadata = None
                self.tensor_parallel_size = None
                self.kv_cache_dtype_resolved = None
                self.dram_kv_tier_profile_sha256 = None
                self.dram_kv_tier_min_restore_tokens = None
            if cleanup_error is not None:
                raise cleanup_error
            if receiver_cancelled is not None:
                raise receiver_cancelled

    async def shutdown(self) -> None:
        # Terminal before the first await. If a model load is in flight, its
        # Pipe poll sees ``abandoned`` while every caller shares one cleanup.
        self._closed = True
        with self._prepared_requests_lock:
            self._prepared_requests.clear()
        abandoned = self._startup_abandoned
        if abandoned is not None:
            abandoned.set()
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(self._shutdown_impl())
            self._shutdown_task = task
        try:
            await self._await_owned_task(task)
        except BaseException:
            # A successful cleanup is the permanent idempotency record. A
            # failed/cancelled cleanup must not poison shutdown forever: the
            # retained process handle is precisely what a later call needs in
            # order to retry escalation and reap the service tree.
            failed = task.done() and (
                task.cancelled() or task.exception() is not None
            )
            if failed and self._shutdown_task is task:
                self._shutdown_task = None
            raise

    # -- request plumbing ----------------------------------------------------

    def _peek_prepared_request(
        self,
        request: GenerationRequest,
    ) -> _PreparedProcPrompt | None:
        key = id(request)
        with self._prepared_requests_lock:
            cached = self._prepared_requests.get(key)
            if cached is None:
                return None
            if cached[0]() is request:
                return cached[1]
            # Defensive against delayed weakref callbacks and object-ID reuse.
            self._prepared_requests.pop(key, None)
        return None

    def _retain_prepared_request(
        self,
        request: GenerationRequest,
        prepared: _PreparedProcPrompt,
    ) -> _PreparedProcPrompt:
        key = id(request)
        backend_ref = weakref.ref(self)

        def discard(request_ref: weakref.ReferenceType[GenerationRequest]) -> None:
            backend = backend_ref()
            if backend is None:
                return
            with backend._prepared_requests_lock:
                current = backend._prepared_requests.get(key)
                if current is not None and current[0] is request_ref:
                    backend._prepared_requests.pop(key, None)

        request_ref = weakref.ref(request, discard)
        with self._prepared_requests_lock:
            cached = self._prepared_requests.get(key)
            if cached is not None and cached[0]() is request:
                return cached[1]
            self._prepared_requests[key] = (request_ref, prepared)
        return prepared

    def _take_prepared_request(
        self,
        request: GenerationRequest,
    ) -> _PreparedProcPrompt | None:
        key = id(request)
        with self._prepared_requests_lock:
            cached = self._prepared_requests.get(key)
            if cached is None:
                return None
            self._prepared_requests.pop(key, None)
            if cached[0]() is request:
                return cached[1]
        return None

    def _prepare_request(
        self,
        request: GenerationRequest,
        *,
        validate_surface: bool = True,
        prompt: PromptInput | None = None,
    ) -> _PreparedProcPrompt:
        if validate_surface:
            validate_native_request_surface(request)
        native_params = native_sampling_params(request)
        if engine_sampling_from(native_params).needs_grammar:
            tokenizer = self._get_preflight_tokenizer()
            validate_structured_sampling(
                native_params,
                tokenizer,
                tokenizer.eos_token_id,
            )
        if prompt is None:
            prompt = prompt_with_tool_intent(request)
        max_model_len = self._max_model_len
        tokenize_started_ns = (
            time.perf_counter_ns() if request.trace_requested else None
        )

        prompt_token_ids = supplied_prompt_token_ids(prompt)
        if prompt_token_ids is None:
            text = prompt_text(prompt)
            assert text is not None
            tokenizer = self._get_preflight_tokenizer()
            prompt_token_ids = tuple(tokenizer.encode(text))
            if not prompt_token_ids:
                raise ValueError("prompt must tokenize to at least one token")
            # The child sees the exact parent-resolved IDs and therefore does
            # not repeat text tokenization. Display text is retained only for
            # diagnostics; the caller's public request remains untouched.
            prompt = TokensPrompt(prompt_token_ids, prompt=text)

        if max_model_len is not None:
            max_new_tokens = (
                request.sampling_params.max_tokens
                if request.sampling_params.max_tokens is not None
                else 16
            )
            requested_length = len(prompt_token_ids) + max_new_tokens
            if requested_length > max_model_len:
                raise ValueError(
                    f"prompt tokens ({len(prompt_token_ids)}) plus max_tokens "
                    f"({max_new_tokens}) exceed max_model_len ({max_model_len})"
                )
        tokenize_duration_ns = (
            max(0, time.perf_counter_ns() - tokenize_started_ns)
            if tokenize_started_ns is not None
            else None
        )
        return _PreparedProcPrompt(
            prompt,
            tokenize_duration_ns=tokenize_duration_ns,
        )

    def _discard_finished_preparation(
        self,
        key: int,
        flight: _ProcPromptPreparationFlight,
        task: asyncio.Task[_PreparedProcPrompt],
    ) -> None:
        if not task.cancelled():
            task.exception()
        if (
            flight.waiters == 0
            and self._preparing_requests.get(key) is flight
        ):
            self._preparing_requests.pop(key, None)

    async def prepare_request(self, request: GenerationRequest) -> None:
        """Run parent tokenization off-loop and publish only to a live caller."""

        if self._closed:
            raise EngineServiceError("kairyu-proc backend is shut down")
        validate_backend_request_before_prepare(self, request)
        if self._peek_prepared_request(request) is not None:
            return

        key = id(request)
        flight = self._preparing_requests.get(key)
        if flight is not None and flight.request_ref() is not request:
            self._preparing_requests.pop(key, None)
            flight = None
        if flight is None:
            task = asyncio.create_task(
                self._run_tokenizer_work(self._prepare_request, request)
            )
            flight = _ProcPromptPreparationFlight(weakref.ref(request), task)
            self._preparing_requests[key] = flight
            task.add_done_callback(
                lambda done, key=key, flight=flight: self._discard_finished_preparation(
                    key,
                    flight,
                    done,
                )
            )

        flight.waiters += 1
        try:
            prepared = await asyncio.shield(flight.task)
            if self._closed:
                raise EngineServiceError("kairyu-proc backend is shut down")
            self._retain_prepared_request(request, prepared)
        finally:
            flight.waiters -= 1
            if (
                flight.task.done()
                and flight.waiters == 0
                and self._preparing_requests.get(key) is flight
            ):
                self._preparing_requests.pop(key, None)

    def _preflight_tokenizer_concurrent_encode_safe(self) -> bool:
        tokenizer = self._preflight_tokenizer
        return tokenizer is not None and tokenizer_encode_is_concurrent_safe(tokenizer)

    async def _run_tokenizer_work(
        self,
        function: Callable[..., _T],
        /,
        *args: object,
    ) -> _T:
        """Run tokenizer work with unknown implementations serialized off-lane."""

        if self._preflight_tokenizer_concurrent_encode_safe():
            return await run_prompt_work(function, *args)
        return await run_serialized_prompt_work(
            self._prompt_tokenizer_gate,
            lambda: not self._preflight_tokenizer_concurrent_encode_safe(),
            function,
            *args,
        )

    def _get_preflight_tokenizer(self) -> Tokenizer:
        tokenizer = self._preflight_tokenizer
        if tokenizer is not None:
            return tokenizer
        with self._preflight_tokenizer_lock:
            tokenizer = self._preflight_tokenizer
            if tokenizer is None:
                tokenizer = resolve_tokenizer(self._preflight_tokenizer_source)
                self._preflight_tokenizer = tokenizer
        return tokenizer

    def tokenize_loglikelihood(
        self,
        context: str,
        continuation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Resolve scoring IDs in the parent with the child's tokenizer source."""

        return tokenize_loglikelihood_continuation(
            self._get_preflight_tokenizer(),
            context,
            continuation,
        )

    async def tokenize_loglikelihood_async(
        self,
        context: str,
        continuation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Tokenize scoring input through the shared async tokenizer gate."""

        return await self._run_tokenizer_work(
            self.tokenize_loglikelihood,
            context,
            continuation,
        )

    def validate_request_before_prepare(self, request: GenerationRequest) -> None:
        """Validate native surface/shape without parent tokenizer CPU work."""

        validate_native_request_surface_before_prepare(request)
        kind = prompt_kind(request.prompt)
        if (
            request.tools
            and not request.tools_in_prompt
            and request.tool_choice != "none"
        ):
            if isinstance(request.prompt, TemplatedPrompt):
                raise ValueError(
                    "templated prompts cannot receive an implicit "
                    "tool-instruction suffix; render tools inside the chat "
                    "template and set tools_in_prompt=true"
                )
            if kind != "text":
                raise ValueError(
                    f"{kind} prompts cannot receive an implicit "
                    "tool-instruction suffix; render tools before tokenization "
                    "and set tools_in_prompt=true"
                )
        if kind == "multimodal":
            raise ValueError(
                "kairyu-proc does not support multimodal prompts; "
                "no modality processor is configured"
            )

    def validate_request(self, request: GenerationRequest) -> None:
        """Preserve synchronous parent preflight and exact prepared cache."""

        ZmqEngineBackend.validate_request_before_prepare(self, request)
        validate_native_request_surface(request)
        prompt = prompt_with_tool_intent(request)
        if (
            self._max_model_len is not None
            and self._peek_prepared_request(request) is None
        ):
            prepared = self._prepare_request(
                request,
                validate_surface=False,
                prompt=prompt,
            )
            self._retain_prepared_request(request, prepared)

    def _reserve_request_id(self, request_id: str) -> None:
        if request_id in self._active_request_ids:
            raise ValueError(f"duplicate request_id {request_id!r}")
        self._active_request_ids.add(request_id)

    async def _receive_loop(self, generation: int, socket, process) -> None:
        _, msgpack = _import_deps()
        loop = asyncio.get_running_loop()
        last_activity = loop.time()
        last_ping = last_activity
        receive_tick = min(_RECV_TICK_S, self._death_timeout_s / 3.0)
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=receive_tick)
                except TimeoutError:
                    if not process.is_alive():
                        raise EngineServiceError("engine service process died") from None
                    now = loop.time()
                    silent_for = now - last_activity
                    if silent_for >= self._death_timeout_s:
                        raise EngineServiceError(
                            "engine service heartbeat timed out"
                        ) from None
                    if now - last_ping >= receive_tick:
                        remaining = self._death_timeout_s - silent_for
                        try:
                            await _send_control(
                                socket,
                                msgpack.packb({"op": "ping"}),
                                timeout_s=min(_CONTROL_SEND_TIMEOUT_S, remaining),
                            )
                        except Exception:
                            # Silence is judged by the single monotonic deadline;
                            # a blocked control send cannot extend it.
                            pass
                        last_ping = loop.time()
                    continue
                try:
                    event = msgpack.unpackb(raw)
                    if not isinstance(event, dict):
                        raise TypeError("engine event must be a map")
                except Exception as error:
                    # a single corrupt/malformed frame must not kill the receiver
                    # and hang every request (E1); drop it and keep reading
                    logging.warning("kairyu-proc dropped a malformed engine event: %r", error)
                    continue
                last_activity = loop.time()
                op = event.get("op")
                if op in ("pong", "bye"):
                    continue
                if op == "fatal":
                    error_type = event.get("error_type")
                    dead_ranks = event.get("dead_ranks")
                    if not isinstance(error_type, str) or not error_type:
                        error_type = "EngineServiceError"
                    ranks = (
                        sorted(rank for rank in dead_ranks if type(rank) is int)
                        if isinstance(dead_ranks, (list, tuple))
                        else []
                    )
                    detail = (
                        f"engine service reported fatal {error_type}"
                        f" for ranks {ranks}"
                    )
                    raise EngineServiceError(detail)
                if "decode_graph_metadata" in event:
                    try:
                        graph_metadata = _decode_graph_metadata_from_wire(
                            event["decode_graph_metadata"]
                        )
                    except EngineServiceError as error:
                        # Metrics are diagnostic. A malformed additive field
                        # must not discard an otherwise valid request event.
                        logging.warning(
                            "kairyu-proc ignored malformed decode-graph "
                            "metadata: %r",
                            error,
                        )
                    else:
                        if graph_metadata is not None:
                            self._decode_graph_metadata = graph_metadata
                try:
                    self._deliver_event(event)
                except Exception as error:
                    # a malformed request event is isolated exactly like a bad
                    # msgpack frame; heartbeats remain able to retire the child.
                    logging.warning("kairyu-proc dropped a malformed engine event: %r", error)
        except asyncio.CancelledError:  # pragma: no cover - clean shutdown
            raise
        except Exception as caught:
            failure: BaseException = caught
            owns_generation = self._live_generation == generation
            if owns_generation:
                # Close admission before the first cleanup await. The process
                # can remain observable/alive while group reaping blocks, but
                # no new request may be routed into this failed generation.
                self._live_generation = None
                self._service_failure_type = type(failure).__name__
                self.attention_backend_decision = None
                self._decode_graph_metadata = None
                self.tensor_parallel_size = None
                self.kv_cache_dtype_resolved = None
                self.dram_kv_tier_profile_sha256 = None
                self.dram_kv_tier_min_restore_tokens = None
            self._fail_generation_once(generation, failure)
            # Rank 0 can die from OOM/SIGKILL while its distributed followers
            # remain alive. Clean the owned process group immediately; waiting
            # for a later request/reset would strand their GPU allocations.
            _signal_process_tree(process, signal.SIGKILL)
            reaped = await self._wait_for_process_reap(
                process,
                _FORCED_REAP_TIMEOUT_S,
            )
            if not reaped:
                failure = EngineServiceError(
                    "engine service died and its process group did not exit"
                )
            if owns_generation:
                self._service_failure_type = type(failure).__name__

    def _deliver_event(self, event: dict) -> None:
        """Route one decoded event without crossing request generations."""

        wire_request_id = event["request_id"]
        request_id = self._public_request_ids.get(wire_request_id)
        if request_id is None:
            # The public ID was cancelled/reused and this event belongs to its
            # retired wire generation (v1 and v2 both echo request_id).
            return
        queue = self._queues.get(request_id)
        if queue is None:
            return
        if event.get("wire_version") == WIRE_VERSION:
            stream_id = event.get("stream_id")
            if not isinstance(stream_id, str):
                queue.put_nowait(
                    {
                        "error": repr(
                            EngineServiceError("engine wire v2 event omitted its stream_id")
                        )
                    }
                )
                return
            expected_stream_id = self._stream_ids.get(request_id)
            if stream_id != expected_stream_id:
                # A retired generation has a different wire request ID and was
                # dropped above. A mismatch on the current route is therefore
                # malformed; fail this request instead of waiting forever.
                queue.put_nowait(
                    {
                        "error": repr(
                            EngineServiceError(
                                "engine wire v2 stream_id mismatch: "
                                f"expected {expected_stream_id!r}, got {stream_id!r}"
                            )
                        )
                    }
                )
                return
        queue.put_nowait(event)

    @staticmethod
    def _pack_add_message(
        request: GenerationRequest,
        prepared_prompt: PromptInput | None,
        generation_defaults: GenerationDefaults,
        wire_request_id: str,
        stream_id: str | None,
        wire_version: int,
    ) -> bytes:
        """Build the request-sized process frame on the prompt CPU lane."""

        _, msgpack = _import_deps()
        sampling = sampling_params_to_wire(
            generation_defaults.apply(native_sampling_params(request))
        )
        if sampling["seed"] is None:
            # The child schedules by the generation-unique wire ID. Make the
            # historical public-ID default seed explicit so process splitting
            # and request retries remain output-identical.
            sampling["seed"] = stable_request_seed(request.request_id)
        prompt = (
            prepared_prompt
            if prepared_prompt is not None
            else prompt_with_tool_intent(request)
        )
        message = {
            "op": "add",
            "request_id": wire_request_id,
            "prompt": prompt_to_wire(prompt),
            "sampling": sampling,
            "priority": request.priority,
            "scheduling_class": request.scheduling_class,
        }
        if type(prompt) is not str:
            message["prompt_wire_version"] = _PROMPT_WIRE_VERSION
        if wire_version == WIRE_VERSION:
            # Per-request negotiation keeps rolling upgrades bidirectionally
            # compatible: an old service ignores these additive fields.
            message["wire_version"] = WIRE_VERSION
            message["stream_id"] = stream_id
            if request.trace_requested:
                message["trace_requested"] = True
        return msgpack.packb(message)

    async def _submit(self, request: GenerationRequest) -> asyncio.Queue:
        self.validate_request_before_prepare(request)
        await self.prepare_request(request)
        await self._ensure_started()
        assert self._socket is not None
        socket = self._socket
        service_generation = self._live_generation
        if service_generation is None:
            raise EngineServiceError("engine service stopped while submitting the request")
        queue: asyncio.Queue = asyncio.Queue()
        stream_generation = uuid.uuid4().hex
        wire_request_id = f"wire-{stream_generation}"
        stream_id = stream_generation if self._wire_version == WIRE_VERSION else None
        self._queues[request.request_id] = queue
        self._queue_generations[request.request_id] = service_generation
        self._wire_request_ids[request.request_id] = wire_request_id
        self._public_request_ids[wire_request_id] = request.request_id
        if stream_id is not None:
            self._stream_ids[request.request_id] = stream_id
        try:
            generation_defaults = self.generation_defaults
            if not isinstance(generation_defaults, GenerationDefaults):
                raise EngineServiceError(
                    "engine generation defaults are unavailable after startup"
                )
            prepared = self._take_prepared_request(request)
            if prepared is not None:
                if prepared.tokenize_duration_ns is not None:
                    self._parent_tokenize_durations_ns[request.request_id] = (
                        prepared.tokenize_duration_ns
                    )
            packed = await run_prompt_work(
                self._pack_add_message,
                request,
                prepared.prompt if prepared is not None else None,
                generation_defaults,
                wire_request_id,
                stream_id,
                self._wire_version,
            )
            if (
                self._socket is not socket
                or self._live_generation != service_generation
                or self._closed
                or self._service_failure_type is not None
                or (self._receiver is not None and self._receiver.done())
                or (
                    self._process is not None
                    and not self._process.is_alive()
                )
            ):
                raise EngineServiceError(
                    "engine service changed while packing the request"
                )
            try:
                await _send_control(socket, packed)
            except BaseException as error:
                # Cancellation of a ZMQ send cannot prove whether the add
                # reached the child. Retire the generation so an unowned ghost
                # request cannot keep consuming GPU work.
                self._retire_unreliable_transport(service_generation, socket)
                if isinstance(error, TimeoutError):
                    raise EngineServiceError(
                        "engine service request send timed out"
                    ) from None
                raise
        except BaseException:
            if self._queues.get(request.request_id) is queue:
                self._queues.pop(request.request_id, None)
                self._release_wire_route(request.request_id)
            raise
        return queue

    async def _abort(self, request_id: str) -> None:
        if self._socket is None:
            return
        socket = self._socket
        service_generation = self._queue_generations.get(request_id)
        wire_request_id = self._wire_request_ids.get(request_id)
        if wire_request_id is None:
            return
        _, msgpack = _import_deps()
        try:
            await _send_control(
                socket,
                msgpack.packb(
                    {
                        "op": "abort",
                        "request_id": wire_request_id,
                        "stream_id": self._stream_ids.get(request_id),
                    }
                ),
            )
        except asyncio.CancelledError:
            self._retire_unreliable_transport(service_generation, socket)
            raise
        except Exception:
            # Stream finalization must release its route even when the service
            # transport is wedged. Abort delivery is then unknown, so retire
            # the generation instead of leaving an unowned request running.
            self._retire_unreliable_transport(service_generation, socket)
            pass

    def _release_wire_route(self, request_id: str) -> None:
        wire_request_id = self._wire_request_ids.pop(request_id, None)
        if wire_request_id is not None:
            self._public_request_ids.pop(wire_request_id, None)
        self._stream_ids.pop(request_id, None)
        self._parent_tokenize_durations_ns.pop(request_id, None)
        generation = self._queue_generations.pop(request_id, None)
        if generation is not None and generation not in self._queue_generations.values():
            self._failed_generations.discard(generation)

    def _result(
        self,
        request: GenerationRequest,
        event: dict,
        *,
        delta_native: bool = False,
    ) -> GenerationResult:
        logprobs = None
        if event.get("logprobs") is not None:
            logprobs = tuple(
                {int(token_id): logprob for token_id, logprob in entry.items()}
                for entry in event["logprobs"]
            )
        content = None
        if event.get("logprob_content") is not None:
            content = tuple(_decode_token_logprob(raw) for raw in event["logprob_content"])
        completion = CompletionOutput(
            index=0,
            text=event.get("text", ""),
            token_ids=tuple(event["outputs"]),
            cumulative_logprob=event.get("cumulative_logprob", 0.0),
            logprobs=logprobs,
            finish_reason=event.get("finish_reason"),
            logprob_content=content,
            text_delta=event.get("text_delta") if delta_native else None,
            text_offset=event.get("text_offset") if delta_native else None,
        )
        if request.trace_requested:
            decoded_stage_metrics = list(
                _decode_stage_metrics(event.get("stage_metrics"))
            )
            parent_tokenize_ns = self._parent_tokenize_durations_ns.get(
                request.request_id
            )
            if parent_tokenize_ns is not None:
                for index, metric in enumerate(decoded_stage_metrics):
                    if metric.stage == "tokenize":
                        decoded_stage_metrics[index] = GenerationStageMetric(
                            stage="tokenize",
                            duration_ns=metric.duration_ns + parent_tokenize_ns,
                            occurrences=metric.occurrences + 1,
                        )
                        break
                else:
                    decoded_stage_metrics.insert(
                        0,
                        GenerationStageMetric(
                            stage="tokenize",
                            duration_ns=parent_tokenize_ns,
                        ),
                    )
            stage_metrics = tuple(decoded_stage_metrics)
        else:
            stage_metrics = ()
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(completion,),
            finished=event["finished"],
            usage=GenerationUsage(
                prompt_tokens=event.get("num_prompt_tokens", 0),
                completion_tokens=len(event["outputs"]),
                cached_tokens=event.get("num_cached_tokens", 0),
            ),
            prompt_token_ids=supplied_prompt_token_ids(request.prompt) or (),
            stage_metrics=stage_metrics,
        )

    @staticmethod
    def _raise_on_error(event: dict) -> None:
        if "error" in event:
            raise EngineServiceError(event["error"])

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self._reserve_request_id(request.request_id)
        queue = None
        accumulator = _WireAccumulator()
        finished_cleanly = False
        try:
            queue = await self._submit(request)
            while True:
                event = await queue.get()
                self._raise_on_error(event)
                event = accumulator.apply(
                    event,
                    materialize=bool(event.get("finished")),
                )
                if event["finished"]:
                    finished_cleanly = True
                    return self._result(request, event)
        finally:
            try:
                owns_queue = queue is not None and self._queues.get(request.request_id) is queue
                if owns_queue:
                    self._queues.pop(request.request_id, None)
                try:
                    if queue is not None and not finished_cleanly:
                        # Keep the old stream generation available until its
                        # abort frame is sent. Only then may the public request
                        # ID be reused by a new generation.
                        await self._abort(request.request_id)
                finally:
                    if owns_queue:
                        self._release_wire_route(request.request_id)
            finally:
                self._active_request_ids.discard(request.request_id)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        self._reserve_request_id(request.request_id)
        queue = None
        accumulator = _WireAccumulator()
        emitted = -1
        finished_cleanly = False
        try:
            queue = await self._submit(request)
            while True:
                event = await queue.get()
                self._raise_on_error(event)
                emit = _stream_event_needs_materialization(event)
                event = accumulator.apply(
                    event,
                    materialize=bool(event.get("finished")),
                )
                if not emit:
                    continue
                if len(event["outputs"]) > emitted or event["finished"]:
                    emitted = len(event["outputs"])
                    yield self._result(request, event, delta_native=True)
                if event["finished"]:
                    finished_cleanly = True
                    return
        finally:
            try:
                owns_queue = queue is not None and self._queues.get(request.request_id) is queue
                if owns_queue:
                    self._queues.pop(request.request_id, None)
                try:
                    if queue is not None and not finished_cleanly:
                        await self._abort(request.request_id)
                finally:
                    if owns_queue:
                        self._release_wire_route(request.request_id)
            finally:
                self._active_request_ids.discard(request.request_id)


register_backend("kairyu-proc", ZmqEngineBackend)
