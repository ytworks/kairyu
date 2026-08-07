"""Engine-core service process: ZMQ ROUTER + msgpack over one EngineLoop (m8 D6).

The service owns tokenizer + sampler + scheduler + runner — the deploy-day
process layout. Its socket thread drains ops, polls control traffic, and sends
already-packed events. One serial step executor advances the engine, while the
loop's private serial output lane detokenizes and encodes wire events. The
socket keeps polling during both phases; the death-detection timeout still must
allow for transport and process scheduling delays.

Wire protocol (msgpack maps):
  client → service: {"op": "add", "request_id", "prompt", "sampling": {...},
                     "priority": int, "scheduling_class": str,
                     "wire_version": 2, optional "trace_requested": bool}
                    Typed text/token prompts add ``prompt_wire_version: 1`` and
                    a strict tagged ``prompt`` map; legacy strings stay raw.
                    {"op": "abort", "request_id"} | {"op": "ping"} | {"op": "shutdown"}
  service → v2 client: one cumulative ``snapshot`` followed by sequenced
                    ``delta`` events carrying only new token IDs, visible text,
                    and logprob metadata. Trace-enabled events additionally
                    carry one sanitized cumulative ``stage_metrics`` list. The
                    first event carries ``num_prompt_tokens``. Native runners
                    may add live ``decode_graph_metadata`` counters.
  service → legacy client: the historical cumulative per-step event.
                    New clients accept both forms, so a rolling upgrade works
                    in either order.
                    Control replies are {"op": "pong"} | {"op": "bye"} or a
                    sanitized {"op": "fatal", "error_type", "dead_ranks"}.

The child entrypoint is a top-level function (spawn pickles it).  The
multiprocessing Pipe keeps its historical first frame — the raw ephemeral port
integer — and a new service optionally follows it with a versioned startup
frame after the engine is fully constructed.  Keeping the first frame intact
lets old parents start new children; the optional send tolerates the old parent
closing its Pipe end immediately.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kairyu.engine.prompt import prompt_from_wire
from kairyu.sampling_params import SamplingParams

if TYPE_CHECKING:  # pragma: no cover
    from kairyu.engine.engine_loop import StreamUpdate

_POLL_IDLE_MS = 50
_POLL_ACTIVE_MS = 1
LEGACY_WIRE_VERSION = 1
WIRE_VERSION = 2
_SUPPORTED_WIRE_VERSIONS = frozenset({LEGACY_WIRE_VERSION, WIRE_VERSION})
_PROMPT_WIRE_VERSION = 1
STARTUP_WIRE_VERSION = 1


@dataclass
class WireEventCursor:
    """Per-request v2 cursor used to encode cumulative engine updates as deltas."""

    sequence: int = 0
    output_count: int = 0
    text: str = ""
    logprob_count: int = 0
    logprob_content_count: int = 0
    has_logprobs: bool = False
    has_logprob_content: bool = False
    stage_metrics: tuple[tuple[str, int, int], ...] = ()


@dataclass
class _RequestOwner:
    identity: bytes
    wire_version: int
    stream_id: str | None
    trace_requested: bool
    cursor: WireEventCursor


def sampling_params_from_wire(payload: dict) -> SamplingParams:
    """Rebuild SamplingParams from its wire dict (unknown keys rejected loudly)."""
    return SamplingParams(**payload)


def sampling_params_to_wire(params: SamplingParams) -> dict:
    payload = {
        "n": params.n,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
        "repetition_penalty": params.repetition_penalty,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "min_p": params.min_p,
        "seed": params.seed,
        "stop": list(params.stop or ()),
        "stop_token_ids": list(params.stop_token_ids or ()),
        "max_tokens": params.max_tokens,
        "min_tokens": params.min_tokens,
        "logprobs": params.logprobs,
        "ignore_eos": params.ignore_eos,
        "skip_special_tokens": params.skip_special_tokens,
        "extra_args": params.extra_args or {},
    }
    # Preserve the exact legacy/default wire bytes.  An active native-only
    # continuation is explicit and fails loudly against an older service,
    # while ordinary rolling-upgrade traffic remains unchanged.
    if params.forced_token_ids is not None:
        payload["forced_token_ids"] = list(params.forced_token_ids)
    return payload


def _trace_requested_from_wire(message: dict) -> bool:
    """Decode the optional add-frame trace flag without truthy coercion."""

    trace_requested = message.get("trace_requested", False)
    if type(trace_requested) is not bool:
        raise ValueError("engine add trace_requested must be a boolean")
    return trace_requested


def _stage_metrics_to_wire(metrics: object) -> tuple[
    tuple[tuple[str, int, int], ...],
    list[dict[str, str | int]],
]:
    """Reduce internal metric values to the reviewed scalar wire schema."""

    if not isinstance(metrics, tuple):
        raise ValueError("engine stage metrics must be a tuple")
    rows: list[tuple[str, int, int]] = []
    payload: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for metric in metrics:
        stage = getattr(metric, "stage", None)
        duration_ns = getattr(metric, "duration_ns", None)
        occurrences = getattr(metric, "occurrences", None)
        if type(stage) is not str or not stage.strip():
            raise ValueError("engine stage metric stage must be a non-empty string")
        if type(duration_ns) is not int or duration_ns < 0:
            raise ValueError("engine stage metric duration_ns must be a non-negative integer")
        if type(occurrences) is not int or occurrences < 1:
            raise ValueError("engine stage metric occurrences must be a positive integer")
        if stage in seen:
            raise ValueError(f"engine stage metrics contain duplicate stage {stage!r}")
        seen.add(stage)
        rows.append((stage, duration_ns, occurrences))
        payload.append(
            {
                "stage": stage,
                "duration_ns": duration_ns,
                "occurrences": occurrences,
            }
        )
    return tuple(rows), payload


def _legacy_event_from_update(request_id: str, update: StreamUpdate) -> dict:
    event: dict = {
        "request_id": request_id,
        "outputs": list(update.outputs),
        "text": update.text,
        "finished": update.finished,
        "finish_reason": update.finish_reason,
        "num_prompt_tokens": update.num_prompt_tokens,
        "num_cached_tokens": update.num_cached_tokens,
        "cumulative_logprob": update.cumulative_logprob,
    }
    if update.logprobs is not None:
        event["logprobs"] = [
            {str(token_id): logprob for token_id, logprob in entry.items()}
            for entry in update.logprobs
        ]
    if update.logprob_content is not None:
        event["logprob_content"] = [_encode_token_logprob(t) for t in update.logprob_content]
    return event


def _validate_monotonic_length(name: str, current: int, previous: int) -> None:
    if current < previous:
        raise ValueError(
            f"engine {name} retracted from {previous} to {current}; "
            "delta wire requires cumulative updates"
        )


def _v2_event_from_update(
    request_id: str,
    update: StreamUpdate,
    cursor: WireEventCursor,
    *,
    trace_requested: bool = False,
) -> dict | None:
    """Encode one cumulative ``StreamUpdate`` without retransmitting its prefix."""

    output_count = len(update.outputs)
    text_length = len(update.text)
    logprob_count = len(update.logprobs) if update.logprobs is not None else 0
    content_count = len(update.logprob_content) if update.logprob_content is not None else 0
    stage_metric_rows: tuple[tuple[str, int, int], ...] = ()
    stage_metric_payload: list[dict[str, str | int]] | None = None
    if trace_requested:
        stage_metric_rows, stage_metric_payload = _stage_metrics_to_wire(
            getattr(update, "stage_metrics", ())
        )
    _validate_monotonic_length("token output", output_count, cursor.output_count)
    _validate_monotonic_length("logprobs", logprob_count, cursor.logprob_count)
    _validate_monotonic_length(
        "logprob content",
        content_count,
        cursor.logprob_content_count,
    )
    if cursor.has_logprobs and update.logprobs is None:
        raise ValueError("engine logprobs disappeared from a cumulative update")
    if cursor.has_logprob_content and update.logprob_content is None:
        raise ValueError("engine logprob content disappeared from a cumulative update")
    if cursor.sequence > 0 and not update.finished:
        _validate_monotonic_length(
            "visible text",
            text_length,
            len(cursor.text),
        )
        if not update.text.startswith(cursor.text):
            raise ValueError("engine visible text changed an already-emitted non-terminal prefix")
    if (
        cursor.sequence > 0
        and output_count == cursor.output_count
        and text_length == len(cursor.text)
        and logprob_count == cursor.logprob_count
        and content_count == cursor.logprob_content_count
        and (update.logprobs is not None) == cursor.has_logprobs
        and (update.logprob_content is not None) == cursor.has_logprob_content
        and (not trace_requested or stage_metric_rows == cursor.stage_metrics)
        and not update.finished
    ):
        # A request can receive a cumulative update while another request made
        # the actual engine progress. Suppressing an empty non-terminal event
        # keeps wire volume proportional to this request's output, not to
        # unrelated scheduler steps.
        return None

    common: dict = {
        "wire_version": WIRE_VERSION,
        "request_id": request_id,
        "sequence": cursor.sequence,
        "finished": update.finished,
        "finish_reason": update.finish_reason,
        "num_cached_tokens": update.num_cached_tokens,
        "cumulative_logprob": update.cumulative_logprob,
    }
    if stage_metric_payload is not None:
        common["stage_metrics"] = stage_metric_payload
    if cursor.sequence == 0:
        event = {
            **common,
            "event": "snapshot",
            "outputs": list(update.outputs),
            "text": update.text,
            "num_prompt_tokens": update.num_prompt_tokens,
        }
        if update.logprobs is not None:
            event["logprobs"] = [
                {str(token_id): logprob for token_id, logprob in entry.items()}
                for entry in update.logprobs
            ]
        if update.logprob_content is not None:
            event["logprob_content"] = [
                _encode_token_logprob(entry) for entry in update.logprob_content
            ]
    else:
        if update.finished:
            # Final exact detokenization may rewrite the last unstable
            # characters. Compute one terminal longest-common-prefix so the
            # client can replace the tail exactly; this O(L) scan happens once.
            text_offset = 0
            for previous, current in zip(cursor.text, update.text, strict=False):
                if previous != current:
                    break
                text_offset += 1
        else:
            text_offset = len(cursor.text)
        event = {
            **common,
            "event": "delta",
            "output_offset": cursor.output_count,
            "new_token_ids": list(update.outputs[cursor.output_count :]),
            "text_offset": text_offset,
            "text_delta": update.text[text_offset:],
        }
        if update.logprobs is not None:
            event["logprob_offset"] = cursor.logprob_count
            event["new_logprobs"] = [
                {str(token_id): logprob for token_id, logprob in entry.items()}
                for entry in update.logprobs[cursor.logprob_count :]
            ]
        if update.logprob_content is not None:
            event["logprob_content_offset"] = cursor.logprob_content_count
            event["new_logprob_content"] = [
                _encode_token_logprob(entry)
                for entry in update.logprob_content[cursor.logprob_content_count :]
            ]

    cursor.sequence += 1
    cursor.output_count = output_count
    cursor.text = update.text
    cursor.logprob_count = logprob_count
    cursor.logprob_content_count = content_count
    cursor.has_logprobs = update.logprobs is not None
    cursor.has_logprob_content = update.logprob_content is not None
    if trace_requested:
        cursor.stage_metrics = stage_metric_rows
    return event


def event_from_update(
    request_id: str,
    update: StreamUpdate,
    *,
    wire_version: int = LEGACY_WIRE_VERSION,
    cursor: WireEventCursor | None = None,
    trace_requested: bool = False,
) -> dict | None:
    """Encode one event for a negotiated protocol version.

    Version 1 deliberately retains the historical map for rolling upgrades.
    Version 2 requires a request-local cursor and emits one snapshot followed
    by deltas.
    """

    if wire_version == LEGACY_WIRE_VERSION:
        return _legacy_event_from_update(request_id, update)
    if wire_version == WIRE_VERSION:
        if cursor is None:
            raise ValueError("wire v2 event encoding requires a cursor")
        return _v2_event_from_update(
            request_id,
            update,
            cursor,
            trace_requested=trace_requested,
        )
    raise ValueError(f"unsupported engine wire version {wire_version}")


def _encode_token_logprob(entry) -> list:
    return [
        entry.token,
        entry.token_id,
        entry.logprob,
        list(entry.bytes_) if entry.bytes_ is not None else None,
        [_encode_token_logprob(t) for t in entry.top],
    ]


def _attention_backend_decision_to_wire(decision) -> dict | None:
    """Encode the child's constructed decision as plain startup metadata."""

    if decision is None:
        return None
    return {
        "requested": decision.requested,
        "resolved": decision.resolved,
        "source": decision.source,
        "components": dict(decision.components),
        "rationale": decision.rationale,
        "architecture": dict(decision.architecture),
    }


def _decode_graph_metadata_to_wire(engine_loop) -> dict | None:
    """Best-effort live graph counters for startup and request event frames."""

    runner = getattr(engine_loop, "_runner", None)
    getter = getattr(runner, "decode_graph_metadata", None)
    if not callable(getter):
        return None
    try:
        metadata = getter()
    except Exception:
        return None
    if not isinstance(metadata, Mapping):
        return None
    return {
        "decode_mode": metadata.get("decode_mode"),
        "cuda_graph_decode": metadata.get("cuda_graph_decode"),
        "cuda_graph_buckets": list(metadata.get("cuda_graph_buckets", ())),
        "cuda_graph_captures": metadata.get("cuda_graph_captures"),
        "cuda_graph_replays": metadata.get("cuda_graph_replays"),
        "cuda_graph_eager_fallbacks": metadata.get(
            "cuda_graph_eager_fallbacks"
        ),
    }


def _encode_update_batch(
    owners: Mapping[str, _RequestOwner],
    updates: list[tuple[str, StreamUpdate]],
    msgpack,
    graph_metadata: Mapping[str, object] | None,
) -> list[tuple[str, _RequestOwner, bool, bytes]]:
    """Build and pack one FIFO update batch away from the socket thread."""

    encoded: list[tuple[str, _RequestOwner, bool, bytes]] = []
    for request_id, update in updates:
        owner = owners.get(request_id)
        if owner is None:
            continue
        event = event_from_update(
            request_id,
            update,
            wire_version=owner.wire_version,
            cursor=owner.cursor,
            trace_requested=owner.trace_requested,
        )
        if event is None:
            continue
        if owner.stream_id is not None:
            event["stream_id"] = owner.stream_id
        if graph_metadata is not None:
            event["decode_graph_metadata"] = dict(graph_metadata)
        encoded.append(
            (request_id, owner, update.finished, msgpack.packb(event))
        )
    return encoded


def _parallel_failure_frame(engine_loop) -> dict | None:
    """Return one sanitized fatal frame for an unusable parallel group."""

    launcher = getattr(engine_loop, "parallel_launcher", None)
    if launcher is None:
        launcher = getattr(engine_loop, "tp_launcher", None)
    if launcher is None:
        return None
    try:
        probe_failure = getattr(launcher, "failure_type", None)
        failure_type = probe_failure() if probe_failure is not None else None
        dead_ranks = tuple(launcher.dead_ranks())
    except Exception as error:
        # A health probe that cannot inspect the group is itself fatal. Only
        # the exception class crosses the local control channel.
        failure_type = type(error).__name__
        dead_ranks = ()
    if failure_type is None and not dead_ranks:
        return None
    return {
        "op": "fatal",
        "error_type": failure_type or "RankProcessExited",
        "dead_ranks": sorted(dead_ranks),
    }


def _startup_ready_frame(engine_loop) -> dict:
    generation = getattr(engine_loop, "generation_defaults", None)
    parallel_failure = _parallel_failure_frame(engine_loop)
    if parallel_failure is not None:
        raise RuntimeError(
            "parallel ranks failed before startup readiness: "
            f"{parallel_failure['error_type']} "
            f"{parallel_failure['dead_ranks']}"
        )
    tensor_parallel_size = getattr(engine_loop, "tensor_parallel_size", 1)
    return {
        "startup_wire_version": STARTUP_WIRE_VERSION,
        "status": "ready",
        "tensor_parallel_size": tensor_parallel_size,
        "attention_backend_decision": _attention_backend_decision_to_wire(
            getattr(engine_loop, "attention_backend_decision", None)
        ),
        "decode_graph_metadata": _decode_graph_metadata_to_wire(engine_loop),
        "kv_cache_dtype_requested": getattr(
            engine_loop, "kv_cache_dtype_requested", None
        ),
        "kv_cache_dtype_resolved": getattr(
            engine_loop, "kv_cache_dtype_resolved", None
        ),
        "dram_kv_tier": {
            "enabled": getattr(engine_loop, "dram_kv_tier_enabled", False),
            "capacity_pages": getattr(
                engine_loop, "dram_kv_tier_capacity_pages", 0
            ),
            "profile_sha256": getattr(
                engine_loop, "dram_kv_tier_profile_sha256", None
            ),
            "min_restore_tokens": getattr(
                engine_loop, "dram_kv_tier_min_restore_tokens", None
            ),
        },
        "generation_defaults": (
            {
                "eos_token_id": generation.eos_token_id,
                "stop_token_ids": list(generation.stop_token_ids),
                "sampling": generation.sampling_defaults(),
                "mode": generation.mode,
                "source": generation.source,
            }
            if generation is not None
            else None
        ),
    }


def _startup_error_frame(error: BaseException) -> dict:
    return {
        "startup_wire_version": STARTUP_WIRE_VERSION,
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _send_optional_startup_frame(port_pipe, frame: dict) -> bool:
    """Best-effort second Pipe frame, absent from the legacy parent contract."""

    try:
        port_pipe.send(frame)
    except (BrokenPipeError, EOFError, OSError):
        # A legacy parent closes its endpoint immediately after receiving the
        # integer port.  That is a supported rolling-upgrade direction, not a
        # reason to take down a successfully constructed engine.
        return False
    return True


def _close_engine_service_loop(engine_loop) -> None:
    """Settle the loop and always release any distributed rank launcher."""

    try:
        engine_loop.close()
    finally:
        launcher = getattr(engine_loop, "parallel_launcher", None)
        if launcher is None:
            launcher = getattr(engine_loop, "tp_launcher", None)
        if launcher is not None:
            launcher.shutdown()


def _cleanup_engine_service(engine_loop, socket, context) -> None:
    """Always release distributed ranks, transport, and the ZMQ context."""

    try:
        _close_engine_service_loop(engine_loop)
    finally:
        try:
            socket.close(linger=0)
        finally:
            context.term()


@contextmanager
def _engine_config_with_pinned_dram_profile(
    config: dict,
    profile_bytes: bytes | None,
) -> Iterator[dict]:
    """Materialize the parent's immutable profile bytes for one child load."""

    if profile_bytes is None:
        yield config
        return
    if not isinstance(profile_bytes, bytes) or not profile_bytes:
        raise ValueError("pinned DRAM KV profile bytes must be non-empty bytes")
    if not config.get("dram_kv_tier_profile"):
        raise ValueError("pinned DRAM KV profile bytes require an enabled profile")

    descriptor, snapshot_path = tempfile.mkstemp(
        prefix="kairyu-dram-kv-profile-",
        suffix=".json",
    )
    try:
        snapshot = os.fdopen(descriptor, "wb")
        descriptor = -1
        with snapshot:
            snapshot.write(profile_bytes)
        child_config = dict(config)
        child_config["dram_kv_tier_profile"] = snapshot_path
        yield child_config
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(snapshot_path)


def run_engine_service(
    port_pipe,
    config: dict,
    dram_kv_tier_profile_bytes: bytes | None = None,
) -> None:
    """Child-process main: bind, report the port, serve until shutdown."""
    import msgpack
    import zmq

    from kairyu.engine.kairyu_backend import build_engine_loop

    # bind + report BEFORE building the loop: model load must not eat into
    # the client's spawn timeout (m12 D5 amendment)
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    port_pipe.send(port)
    try:
        with _engine_config_with_pinned_dram_profile(
            config,
            dram_kv_tier_profile_bytes,
        ) as child_config:
            engine_loop, _, _ = build_engine_loop(**child_config)
    except BaseException as error:
        _send_optional_startup_frame(port_pipe, _startup_error_frame(error))
        port_pipe.close()
        socket.close(linger=0)
        context.term()
        raise
    try:
        ready_frame = _startup_ready_frame(engine_loop)
    except BaseException as error:
        _send_optional_startup_frame(port_pipe, _startup_error_frame(error))
        port_pipe.close()
        _cleanup_engine_service(engine_loop, socket, context)
        raise
    _send_optional_startup_frame(port_pipe, ready_frame)
    port_pipe.close()

    owners: dict[str, _RequestOwner] = {}
    step_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="kairyu-step",
    )
    step_future: Future | None = None
    step_clears_socket_boundary = False
    ready_batch = None
    pending_output = None
    output_ack_future: Future | None = None
    step_error: BaseException | None = None
    socket_boundary_pending = False
    running = True
    try:
        while running:
            active = step_error is not None or any(
                item is not None
                for item in (
                    step_future,
                    ready_batch,
                    pending_output,
                    output_ack_future,
                )
            )
            timeout = (
                _POLL_ACTIVE_MS
                if active
                else (0 if engine_loop.has_work() else _POLL_IDLE_MS)
            )
            socket.poll(timeout)
            while not socket_boundary_pending and socket.poll(0):
                identity, raw = socket.recv_multipart()
                # Per-message fault isolation: a malformed frame, a bad sampling
                # payload, or a duplicate request_id must fail only the offending
                # client, never take down the shared engine for everyone else.
                message = None
                try:
                    message = msgpack.unpackb(raw)
                    op = message.get("op")
                    if op == "add":
                        request_id = message["request_id"]
                        wire_version = message.get(
                            "wire_version",
                            LEGACY_WIRE_VERSION,
                        )
                        if (
                            type(wire_version) is not int
                            or wire_version not in _SUPPORTED_WIRE_VERSIONS
                        ):
                            raise ValueError(
                                f"unsupported engine wire version {wire_version!r}; "
                                f"supported versions are "
                                f"{sorted(_SUPPORTED_WIRE_VERSIONS)}"
                            )
                        stream_id = message.get("stream_id")
                        if wire_version == WIRE_VERSION and (
                            not isinstance(stream_id, str) or not stream_id
                        ):
                            raise ValueError("engine wire v2 add requires a non-empty stream_id")
                        trace_requested = _trace_requested_from_wire(message)
                        prompt_wire_version = message.get("prompt_wire_version")
                        if prompt_wire_version is None:
                            prompt = message["prompt"]
                            if not isinstance(prompt, str):
                                raise ValueError(
                                    "legacy engine prompt wire requires a string; "
                                    "typed prompts require prompt_wire_version=1"
                                )
                        elif prompt_wire_version == _PROMPT_WIRE_VERSION:
                            prompt = prompt_from_wire(message["prompt"])
                        else:
                            raise ValueError(
                                "unsupported engine prompt wire version "
                                f"{prompt_wire_version!r}; supported version is "
                                f"{_PROMPT_WIRE_VERSION}"
                            )
                        engine_loop.submit(
                            request_id,
                            prompt,
                            sampling_params_from_wire(message["sampling"]),
                            priority=message.get("priority", 0),
                            scheduling_class=message.get("scheduling_class", "interactive"),
                            trace_requested=trace_requested,
                        )
                        # Install ownership only after a clean submit; a rejected
                        # duplicate must not replace the original request's route.
                        owners[request_id] = _RequestOwner(
                            identity=identity,
                            wire_version=wire_version,
                            stream_id=stream_id,
                            trace_requested=trace_requested,
                            cursor=WireEventCursor(),
                        )
                    elif op == "abort":
                        request_id = message["request_id"]
                        owner = owners.get(request_id)
                        # A cancelled stream can be followed immediately by a
                        # new stream reusing the public request ID. Never let a
                        # delayed abort from the old generation kill the new one.
                        if owner is None or owner.stream_id == message.get("stream_id"):
                            engine_loop.abort(request_id)
                            owners.pop(request_id, None)
                            socket_boundary_pending = True
                        # Apply abort/_forget before accepting a queued add with
                        # the same ID; otherwise one socket-drain batch sees it
                        # as a duplicate even though the client awaited abort send.
                        break
                    elif op == "ping":
                        parallel_failure = _parallel_failure_frame(engine_loop)
                        if parallel_failure is None:
                            reply = {"op": "pong"}
                        else:
                            reply = parallel_failure
                            running = False
                        socket.send_multipart([identity, msgpack.packb(reply)])
                        if not running:
                            break
                    elif op == "shutdown":
                        socket.send_multipart([identity, msgpack.packb({"op": "bye"})])
                        running = False
                        break
                except Exception as error:
                    logging.warning("kairyu engine service rejected a message: %r", error)
                    request_id = message.get("request_id") if isinstance(message, dict) else None
                    socket.send_multipart(
                        [
                            identity,
                            msgpack.packb(
                                {
                                    "wire_version": (
                                        message.get(
                                            "wire_version",
                                            LEGACY_WIRE_VERSION,
                                        )
                                        if isinstance(message, dict)
                                        else LEGACY_WIRE_VERSION
                                    ),
                                    "request_id": request_id or "",
                                    "stream_id": (
                                        message.get("stream_id")
                                        if isinstance(message, dict)
                                        else None
                                    ),
                                    "error": repr(error),
                                    "finished": True,
                                }
                            ),
                        ]
                    )
            if not running:
                break

            if output_ack_future is not None and output_ack_future.done():
                encoded = output_ack_future.result()
                output_ack_future = None
                for request_id, owner, finished, payload in encoded:
                    if owners.get(request_id) is not owner:
                        continue
                    socket.send_multipart([owner.identity, payload])
                    if finished:
                        del owners[request_id]

            if pending_output is not None and pending_output.future.done():
                if pending_output.requires_barrier:
                    # Stop-string feedback must return to the scheduler owner
                    # before another step. The core executor is idle whenever
                    # a barrier output is active.
                    output_ack_future = step_executor.submit(
                        engine_loop.finish_output,
                        pending_output,
                    )
                else:
                    encoded = engine_loop.finish_output(pending_output)
                    for request_id, owner, finished, payload in encoded:
                        if owners.get(request_id) is not owner:
                            continue
                        socket.send_multipart([owner.identity, payload])
                        if finished:
                            del owners[request_id]
                pending_output = None

            if step_future is not None and step_future.done():
                try:
                    ready_batch = step_future.result()
                except BaseException as error:
                    step_error = error
                step_future = None
                if step_clears_socket_boundary:
                    socket_boundary_pending = False
                    step_clears_socket_boundary = False

            # An abort for an already-finished or unknown request queues no
            # engine op. There is then no step boundary to wait for, so do not
            # leave socket draining permanently gated.
            if (
                socket_boundary_pending
                and step_future is None
                and ready_batch is None
                and pending_output is None
                and output_ack_future is None
                and step_error is None
                and not engine_loop.has_work()
            ):
                socket_boundary_pending = False

            if (
                ready_batch is not None
                and pending_output is None
                and output_ack_future is None
            ):
                batch, ready_batch = ready_batch, None
                if not batch.inputs:
                    continue
                owner_snapshot = dict(owners)
                graph_metadata = _decode_graph_metadata_to_wire(engine_loop)
                pending_output = engine_loop.start_output(
                    batch,
                    lambda updates,
                    owner_snapshot=owner_snapshot,
                    graph_metadata=graph_metadata: _encode_update_batch(
                        owner_snapshot,
                        updates,
                        msgpack,
                        graph_metadata,
                    ),
                )

            if (
                step_error is not None
                and pending_output is None
                and output_ack_future is None
                and ready_batch is None
            ):
                raise step_error

            can_advance = (
                step_future is None
                and ready_batch is None
                and output_ack_future is None
                and step_error is None
                and (
                    pending_output is None
                    or not pending_output.requires_barrier
                )
            )
            if can_advance and engine_loop.has_work():
                step_clears_socket_boundary = socket_boundary_pending
                step_future = step_executor.submit(engine_loop.advance_output)
    finally:
        step_executor.shutdown(wait=True, cancel_futures=True)
        _cleanup_engine_service(engine_loop, socket, context)
