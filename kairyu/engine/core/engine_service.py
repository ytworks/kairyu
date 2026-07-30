"""Engine-core service process: ZMQ ROUTER + msgpack over one EngineLoop (m8 D6).

The service owns tokenizer + sampler + scheduler + runner — the deploy-day
process layout. Single-threaded loop: drain socket ops → one engine step →
send events; the m8 D1 op discipline holds by construction. Heartbeats are
answered between steps, so the client's death-detection timeout must exceed
the worst-case step time.

Wire protocol (msgpack maps):
  client → service: {"op": "add", "request_id", "prompt", "sampling": {...},
                     "priority": int, "scheduling_class": str,
                     "wire_version": 2}
                    Typed text/token prompts add ``prompt_wire_version: 1`` and
                    a strict tagged ``prompt`` map; legacy strings stay raw.
                    {"op": "abort", "request_id"} | {"op": "ping"} | {"op": "shutdown"}
  service → v2 client: one cumulative ``snapshot`` followed by sequenced
                    ``delta`` events carrying only new token IDs, visible text,
                    and logprob metadata. The first event carries
                    ``num_prompt_tokens``.
  service → legacy client: the historical cumulative per-step event.
                    New clients accept both forms, so a rolling upgrade works
                    in either order.
                    Control replies remain {"op": "pong"} | {"op": "bye"}.

The child entrypoint is a top-level function (spawn pickles it); the ephemeral
port travels back over a multiprocessing Pipe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kairyu.engine.prompt import prompt_from_wire
from kairyu.sampling_params import SamplingParams

if TYPE_CHECKING:  # pragma: no cover
    from kairyu.engine.engine_loop import StreamUpdate

_POLL_IDLE_MS = 50
LEGACY_WIRE_VERSION = 1
WIRE_VERSION = 2
_SUPPORTED_WIRE_VERSIONS = frozenset({LEGACY_WIRE_VERSION, WIRE_VERSION})
_PROMPT_WIRE_VERSION = 1


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


@dataclass
class _RequestOwner:
    identity: bytes
    wire_version: int
    stream_id: str | None
    cursor: WireEventCursor


def sampling_params_from_wire(payload: dict) -> SamplingParams:
    """Rebuild SamplingParams from its wire dict (unknown keys rejected loudly)."""
    return SamplingParams(**payload)


def sampling_params_to_wire(params: SamplingParams) -> dict:
    return {
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
        "extra_args": params.extra_args or {},
    }


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
) -> dict | None:
    """Encode one cumulative ``StreamUpdate`` without retransmitting its prefix."""

    output_count = len(update.outputs)
    text_length = len(update.text)
    logprob_count = len(update.logprobs) if update.logprobs is not None else 0
    content_count = (
        len(update.logprob_content) if update.logprob_content is not None else 0
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
        raise ValueError(
            "engine logprob content disappeared from a cumulative update"
        )
    if cursor.sequence > 0 and not update.finished:
        _validate_monotonic_length(
            "visible text",
            text_length,
            len(cursor.text),
        )
        if not update.text.startswith(cursor.text):
            raise ValueError(
                "engine visible text changed an already-emitted non-terminal prefix"
            )
    if (
        cursor.sequence > 0
        and output_count == cursor.output_count
        and text_length == len(cursor.text)
        and logprob_count == cursor.logprob_count
        and content_count == cursor.logprob_content_count
        and (update.logprobs is not None) == cursor.has_logprobs
        and (update.logprob_content is not None) == cursor.has_logprob_content
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
    return event


def event_from_update(
    request_id: str,
    update: StreamUpdate,
    *,
    wire_version: int = LEGACY_WIRE_VERSION,
    cursor: WireEventCursor | None = None,
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
        return _v2_event_from_update(request_id, update, cursor)
    raise ValueError(f"unsupported engine wire version {wire_version}")


def _encode_token_logprob(entry) -> list:
    return [
        entry.token,
        entry.token_id,
        entry.logprob,
        list(entry.bytes_) if entry.bytes_ is not None else None,
        [_encode_token_logprob(t) for t in entry.top],
    ]


def run_engine_service(port_pipe, config: dict) -> None:
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
    port_pipe.close()
    engine_loop, _, _ = build_engine_loop(**config)

    owners: dict[str, _RequestOwner] = {}
    running = True
    try:
        while running:
            timeout = 0 if engine_loop.has_work() else _POLL_IDLE_MS
            socket.poll(timeout)
            while socket.poll(0):
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
                            raise ValueError(
                                "engine wire v2 add requires a non-empty stream_id"
                            )
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
                            scheduling_class=message.get(
                                "scheduling_class", "interactive"
                            ),
                        )
                        # Install ownership only after a clean submit; a rejected
                        # duplicate must not replace the original request's route.
                        owners[request_id] = _RequestOwner(
                            identity=identity,
                            wire_version=wire_version,
                            stream_id=stream_id,
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
                        # Apply abort/_forget before accepting a queued add with
                        # the same ID; otherwise one socket-drain batch sees it
                        # as a duplicate even though the client awaited abort send.
                        break
                    elif op == "ping":
                        socket.send_multipart([identity, msgpack.packb({"op": "pong"})])
                    elif op == "shutdown":
                        socket.send_multipart([identity, msgpack.packb({"op": "bye"})])
                        running = False
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
            if engine_loop.has_work():
                for request_id, update in engine_loop.step():
                    owner = owners.get(request_id)
                    if owner is None:
                        continue
                    event = event_from_update(
                        request_id,
                        update,
                        wire_version=owner.wire_version,
                        cursor=owner.cursor,
                    )
                    if event is None:
                        continue
                    if owner.stream_id is not None:
                        event["stream_id"] = owner.stream_id
                    socket.send_multipart([owner.identity, msgpack.packb(event)])
                    if update.finished:
                        del owners[request_id]
    finally:
        engine_loop.close()
        socket.close(linger=0)
        context.term()
