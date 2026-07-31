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
import logging
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    prompt_with_tool_intent,
    validate_native_request_surface,
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
from kairyu.engine.engine_loop import _validate_max_model_len
from kairyu.engine.prompt import (
    PromptInput,
    TokensPrompt,
    prompt_kind,
    prompt_text,
    prompt_to_wire,
    supplied_prompt_token_ids,
)
from kairyu.engine.registry import register_backend
from kairyu.engine.tokenizer import Tokenizer, resolve_tokenizer
from kairyu.outputs import CompletionOutput, TokenLogprob

_SPAWN_TIMEOUT_S = 30.0
_SHUTDOWN_TIMEOUT_S = 5.0
_RECV_TICK_S = 1.0
_PROMPT_WIRE_VERSION = 1
_SPAWN_POLL_S = 0.05


def _decode_token_logprob(raw: list) -> TokenLogprob:
    token, token_id, logprob, bytes_, top = raw
    return TokenLogprob(
        token=token,
        token_id=token_id,
        logprob=logprob,
        bytes_=tuple(bytes_) if bytes_ is not None else None,
        top=tuple(_decode_token_logprob(entry) for entry in top),
    )


class EngineServiceError(RuntimeError):
    """The engine service process died or became unreachable."""


def _kill_and_reap_process(process, timeout_s: float) -> bool:
    """Kill one owned child and report whether it was actually reaped."""

    if process.is_alive():
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    process.join(timeout=timeout_s)
    return not process.is_alive()


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


def _recv_optional_startup_frame(
    parent_pipe,
    process,
) -> tuple[AttentionBackendDecision | None, str | None, str | None]:
    """Receive a new child's second frame; EOF from a live child means legacy."""

    try:
        frame = parent_pipe.recv()
    except EOFError:
        if process.is_alive():
            # Old services close the Pipe immediately after the integer port.
            return None, None, None
        raise EngineServiceError(
            "engine service exited before reporting startup metadata"
        ) from None
    decision = _decision_from_startup_frame(frame)
    requested, resolved = _kv_cache_dtype_from_startup_frame(frame)
    return decision, requested, resolved


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
        if not materialize:
            return {"finished": event["finished"]}
        normalized = {
            "request_id": event["request_id"],
            "outputs": self.outputs,
            "text": "".join(self.text_parts),
            "finished": event["finished"],
            "finish_reason": event.get("finish_reason"),
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "cumulative_logprob": self.cumulative_logprob,
        }
        if self.logprobs is not None:
            normalized["logprobs"] = self.logprobs
        if self.logprob_content is not None:
            normalized["logprob_content"] = self.logprob_content
        return normalized


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
        death_timeout_s: float = 10.0,
        model_path: str | None = None,
        pipeline_depth: int = 1,
        decode_mode: str = "eager",
        cuda_graph_max_batch: int = 8,
        cuda_graph_max_pages: int = 512,
        cuda_graph_warmup_iters: int = 3,
        max_model_len: int | None = None,
        kv_cache_dtype: str = "auto",
    ) -> None:
        if tokenizer is not None and not isinstance(tokenizer, str):
            raise ValueError("kairyu-proc requires a string tokenizer (name or path)")
        _validate_max_model_len(max_model_len)
        kv_cache_dtype = validate_kv_cache_dtype(kv_cache_dtype)
        if kv_cache_dtype != "auto" and model_path is None:
            raise ValueError(
                "kairyu-proc explicit KV cache dtype requires a real model_path"
            )
        self._config = {
            "num_pages": num_pages,
            "page_size": page_size,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "priority_age_s": priority_age_s,
            "tokenizer": tokenizer,
            "speculative": speculative,
            "speculative_tokens": speculative_tokens,
            "model_path": model_path,
            "pipeline_depth": pipeline_depth,
            "decode_mode": decode_mode,
            "cuda_graph_max_batch": cuda_graph_max_batch,
            "cuda_graph_max_pages": cuda_graph_max_pages,
            "cuda_graph_warmup_iters": cuda_graph_warmup_iters,
            "kv_cache_dtype": kv_cache_dtype,
        }
        self._death_timeout_s = death_timeout_s
        self._process = None
        self._socket = None
        self._context = None
        self._receiver: asyncio.Task | None = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._stream_ids: dict[str, str] = {}
        self._wire_request_ids: dict[str, str] = {}
        self._public_request_ids: dict[str, str] = {}
        self._queue_generations: dict[str, int] = {}
        self._failed_generations: set[int] = set()
        self._generation_counter = 0
        self._live_generation: int | None = None
        self._wire_version = WIRE_VERSION
        self._active_request_ids: set[str] = set()
        self.attention_backend_decision: AttentionBackendDecision | None = None
        self.kv_cache_dtype_requested = kv_cache_dtype
        self.kv_cache_dtype_resolved: str | None = None
        self._start_lock = asyncio.Lock()
        self._startup_task: asyncio.Task[int] | None = None
        self._startup_abandoned: threading.Event | None = None
        self._closed = False
        self._atexit_registered = False
        self._max_model_len = max_model_len
        # A configured context limit must be enforceable by the HTTP-facing
        # parent before a StreamingResponse commits SSE headers. Lazily resolve
        # the exact tokenizer source that the child uses so construction stays
        # side-effect compatible and config-only tooling need not mount a
        # checkpoint. With no limit, retain the historical one-tokenizer-in-the-
        # child process layout.
        self._preflight_tokenizer_source = (
            tokenizer
            if tokenizer is not None
            else model_path
            if model_path is not None
            else "toy"
        )
        self._preflight_tokenizer: Tokenizer | None = None
        self._preflight_tokenizer_lock = threading.Lock()
        # Validation may be called more than once by wrapping orchestration
        # layers. Retain only this exact immutable request object, consume the
        # prepared token prompt at submit, and let an abandoned request's weak
        # reference reclaim the entry.
        self._prepared_requests: dict[
            int,
            tuple[weakref.ReferenceType[GenerationRequest], PromptInput],
        ] = {}
        self._prepared_requests_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _spawn(self, abandoned: threading.Event | None = None) -> int:
        import multiprocessing

        if abandoned is not None and abandoned.is_set():
            raise EngineServiceError("engine service startup was cancelled")
        spawn = multiprocessing.get_context("spawn")
        parent_pipe, child_pipe = spawn.Pipe()
        process = spawn.Process(
            target=run_engine_service, args=(child_pipe, self._config), daemon=True
        )
        process.start()
        # Register ownership before either Pipe wait.  If the coroutine awaiting
        # this worker thread is cancelled, its cleanup can now find and kill the
        # child instead of leaving an unowned model load behind.
        self._process = process
        child_pipe.close()
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
            decision, requested_kv_dtype, resolved_kv_dtype = (
                _recv_optional_startup_frame(parent_pipe, process)
            )
            _validate_startup_kv_cache_dtype_identity(
                self.kv_cache_dtype_requested,
                requested_kv_dtype,
                resolved_kv_dtype,
            )
            if abandoned is not None and abandoned.is_set():
                raise EngineServiceError("engine service startup was cancelled")
            if not process.is_alive():
                raise EngineServiceError("engine service exited after reporting startup readiness")
            if self._process is not process:
                raise EngineServiceError("engine service startup ownership was lost")
        except BaseException as error:
            self.attention_backend_decision = None
            self.kv_cache_dtype_resolved = None
            reaped = _kill_and_reap_process(process, timeout_s=1.0)
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
        if requested_kv_dtype is not None:
            self.kv_cache_dtype_requested = requested_kv_dtype
        self.kv_cache_dtype_resolved = resolved_kv_dtype
        if not self._atexit_registered:
            atexit.register(self._kill_process)
            self._atexit_registered = True
        return port

    def _is_healthy(self) -> bool:
        return (
            self._socket is not None
            and self._receiver is not None
            and not self._receiver.done()
            and self._process is not None
            and self._process.is_alive()
            and self._live_generation is not None
        )

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

    async def _cancel_receiver_locked(self) -> None:
        receiver = self._receiver
        self._receiver = None
        if receiver is None:
            return
        if not receiver.done():
            receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await receiver

    def _close_transport_locked(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

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
        self.kv_cache_dtype_resolved = None
        self._fail_generation_once(
            generation,
            EngineServiceError(
                "engine service became unavailable and was reset before completing the request"
            ),
        )
        await self._cancel_receiver_locked()
        self._close_transport_locked()
        process = self._process
        if process is not None:
            reaped = await asyncio.to_thread(
                _kill_and_reap_process,
                process,
                1.0,
            )
            if not reaped:
                # Retaining the handle prevents a second GPU-owning child from
                # being spawned over an uninterruptible old process.
                raise EngineServiceError(
                    "engine service child did not exit after kill; refusing to respawn"
                )
            if self._process is process:
                self._process = None

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
                self._receiver = asyncio.get_running_loop().create_task(
                    self._receive_loop(generation, socket, process)
                )
            except BaseException:
                await self._reset_dead_locked()
                raise

    async def startup(self) -> None:
        """Finish child construction before an owning app begins serving."""

        await self._ensure_started()

    def _kill_process(self) -> None:
        process = self._process
        if process is not None and process.is_alive():  # pragma: no cover - crash path
            process.kill()

    async def shutdown(self) -> None:
        # Close admission before waiting for the start lock. If a model load is
        # in flight, its 50 ms Pipe poll observes this event; _ensure_started
        # drains the worker before releasing the same lock to us.
        self._closed = True
        with self._prepared_requests_lock:
            self._prepared_requests.clear()
        abandoned = self._startup_abandoned
        if abandoned is not None:
            abandoned.set()

        async with self._start_lock:
            generation = self._live_generation
            self._live_generation = None
            self.attention_backend_decision = None
            self.kv_cache_dtype_resolved = None
            self._fail_generation_once(
                generation,
                EngineServiceError("engine service shut down before completing the request"),
            )
            await self._cancel_receiver_locked()

            process = self._process
            cleanup_error: EngineServiceError | None = None
            try:
                if process is not None and self._socket is not None:
                    _, msgpack = _import_deps()
                    try:
                        await self._socket.send(msgpack.packb({"op": "shutdown"}))
                    except Exception:  # pragma: no cover - socket already dead
                        pass
                if process is not None:
                    await asyncio.to_thread(process.join, _SHUTDOWN_TIMEOUT_S)
                    if process.is_alive():  # pragma: no cover - hung child
                        with contextlib.suppress(ProcessLookupError):
                            process.terminate()
                        await asyncio.to_thread(process.join, 2.0)
                    if process.is_alive():  # pragma: no cover - wedged child
                        with contextlib.suppress(ProcessLookupError):
                            process.kill()
                        await asyncio.to_thread(process.join, 2.0)
                    if process.is_alive():
                        cleanup_error = EngineServiceError(
                            "engine service child remained alive after shutdown escalation"
                        )
                    elif self._process is process:
                        self._process = None
            finally:
                self._close_transport_locked()
                self.attention_backend_decision = None
                self.kv_cache_dtype_resolved = None
            if cleanup_error is not None:
                # Keep the Process object so a later shutdown attempt can reap
                # it and no future code can mistake the backend for child-free.
                raise cleanup_error

    # -- request plumbing ----------------------------------------------------

    def _peek_prepared_request(
        self,
        request: GenerationRequest,
    ) -> PromptInput | None:
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
        prompt: PromptInput,
    ) -> PromptInput:
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
            self._prepared_requests[key] = (request_ref, prompt)
        return prompt

    def _take_prepared_request(
        self,
        request: GenerationRequest,
    ) -> PromptInput | None:
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
        prompt: PromptInput,
    ) -> PromptInput:
        max_model_len = self._max_model_len
        if max_model_len is None:
            return prompt
        tokenizer = self._get_preflight_tokenizer()

        prompt_token_ids = supplied_prompt_token_ids(prompt)
        if prompt_token_ids is None:
            text = prompt_text(prompt)
            assert text is not None
            prompt_token_ids = tuple(tokenizer.encode(text))
            if not prompt_token_ids:
                raise ValueError("prompt must tokenize to at least one token")
            # The child sees the exact parent-resolved IDs and therefore does
            # not repeat text tokenization. Display text is retained only for
            # diagnostics; the caller's public request remains untouched.
            prompt = TokensPrompt(prompt_token_ids, prompt=text)

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
        return prompt

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

    def validate_request(self, request: GenerationRequest) -> None:
        validate_native_request_surface(request)
        prompt = prompt_with_tool_intent(request)
        if prompt_kind(prompt) == "multimodal":
            raise ValueError(
                "kairyu-proc does not support multimodal prompts; "
                "no modality processor is configured"
            )
        if (
            self._max_model_len is not None
            and self._peek_prepared_request(request) is None
        ):
            prepared = self._prepare_request(request, prompt)
            self._retain_prepared_request(request, prepared)

    def _reserve_request_id(self, request_id: str) -> None:
        if request_id in self._active_request_ids:
            raise ValueError(f"duplicate request_id {request_id!r}")
        self._active_request_ids.add(request_id)

    async def _receive_loop(self, generation: int, socket, process) -> None:
        _, msgpack = _import_deps()
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=_RECV_TICK_S)
                except TimeoutError:
                    if not process.is_alive():
                        raise EngineServiceError("engine service process died") from None
                    continue
                try:
                    event = msgpack.unpackb(raw)
                    if event.get("op") in ("pong", "bye"):
                        continue
                    self._deliver_event(event)
                except Exception as error:
                    # a single corrupt/malformed frame must not kill the receiver
                    # and hang every request (E1); drop it and keep reading
                    logging.warning("kairyu-proc dropped a malformed engine event: %r", error)
                    continue
        except asyncio.CancelledError:  # pragma: no cover - clean shutdown
            raise
        except Exception as error:
            if self._live_generation == generation:
                self._live_generation = None
                self.attention_backend_decision = None
                self.kv_cache_dtype_resolved = None
            self._fail_generation_once(generation, error)

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

    async def _submit(self, request: GenerationRequest) -> asyncio.Queue:
        self.validate_request(request)
        await self._ensure_started()
        assert self._socket is not None
        service_generation = self._live_generation
        if service_generation is None:
            raise EngineServiceError("engine service stopped while submitting the request")
        _, msgpack = _import_deps()
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
            sampling = sampling_params_to_wire(request.sampling_params)
            if sampling["seed"] is None:
                # The child schedules by the generation-unique wire ID. Make
                # the historical public-ID default seed explicit so process
                # splitting and request retries remain output-identical.
                sampling["seed"] = stable_request_seed(request.request_id)
            prompt = self._take_prepared_request(request)
            if prompt is None:
                prompt = prompt_with_tool_intent(request)
            message = {
                "op": "add",
                "request_id": wire_request_id,
                "prompt": prompt_to_wire(prompt),
                "sampling": sampling,
                "priority": request.priority,
                "scheduling_class": request.scheduling_class,
            }
            if not isinstance(prompt, str):
                message["prompt_wire_version"] = _PROMPT_WIRE_VERSION
            if self._wire_version == WIRE_VERSION:
                # Per-request negotiation keeps rolling upgrades
                # bidirectionally compatible: an old service ignores these
                # fields; a new service defaults absent fields to v1.
                message["wire_version"] = WIRE_VERSION
                message["stream_id"] = stream_id
            await self._socket.send(msgpack.packb(message))
        except BaseException:
            if self._queues.get(request.request_id) is queue:
                self._queues.pop(request.request_id, None)
                self._release_wire_route(request.request_id)
            raise
        return queue

    async def _abort(self, request_id: str) -> None:
        if self._socket is None:
            return
        wire_request_id = self._wire_request_ids.get(request_id)
        if wire_request_id is None:
            return
        _, msgpack = _import_deps()
        try:
            await self._socket.send(
                msgpack.packb(
                    {
                        "op": "abort",
                        "request_id": wire_request_id,
                        "stream_id": self._stream_ids.get(request_id),
                    }
                )
            )
        except Exception:  # pragma: no cover - shutdown race
            pass

    def _release_wire_route(self, request_id: str) -> None:
        wire_request_id = self._wire_request_ids.pop(request_id, None)
        if wire_request_id is not None:
            self._public_request_ids.pop(wire_request_id, None)
        self._stream_ids.pop(request_id, None)
        generation = self._queue_generations.pop(request_id, None)
        if generation is not None and generation not in self._queue_generations.values():
            self._failed_generations.discard(generation)

    def _result(self, request: GenerationRequest, event: dict) -> GenerationResult:
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
            text=event["text"],
            token_ids=tuple(event["outputs"]),
            cumulative_logprob=event.get("cumulative_logprob", 0.0),
            logprobs=logprobs,
            finish_reason=event.get("finish_reason"),
            logprob_content=content,
        )
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
                event = accumulator.apply(event)
                if len(event["outputs"]) > emitted or event["finished"]:
                    emitted = len(event["outputs"])
                    yield self._result(request, event)
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
