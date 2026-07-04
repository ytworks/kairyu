"""Engine-core service process: ZMQ ROUTER + msgpack over one EngineLoop (m8 D6).

The service owns tokenizer + sampler + scheduler + runner — the deploy-day
process layout. Single-threaded loop: drain socket ops → one engine step →
send events; the m8 D1 op discipline holds by construction. Heartbeats are
answered between steps, so the client's death-detection timeout must exceed
the worst-case step time.

Wire protocol (msgpack maps):
  client → service: {"op": "add", "request_id", "prompt", "sampling": {...}}
                    {"op": "abort", "request_id"} | {"op": "ping"} | {"op": "shutdown"}
  service → client: per-step events {"request_id", "new_token_ids", "text",
                    "finished", "finish_reason", "num_cached_tokens",
                    "num_prompt_tokens", "logprobs"?, "cumulative_logprob"}
                    (first event for a request carries num_prompt_tokens; text
                    is cumulative visible text) | {"op": "pong"} | {"op": "bye"}

The child entrypoint is a top-level function (spawn pickles it); the ephemeral
port travels back over a multiprocessing Pipe.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kairyu.sampling_params import SamplingParams

if TYPE_CHECKING:  # pragma: no cover
    from kairyu.engine.engine_loop import StreamUpdate

_POLL_IDLE_MS = 50


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


def _event_from_update(request_id: str, update: StreamUpdate) -> dict:
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

    owners: dict[str, bytes] = {}
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
                        engine_loop.submit(
                            request_id,
                            message["prompt"],
                            sampling_params_from_wire(message["sampling"]),
                        )
                        owners[request_id] = identity  # only after a clean submit
                    elif op == "abort":
                        engine_loop.abort(message["request_id"])
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
                                    "request_id": request_id or "",
                                    "error": repr(error),
                                    "finished": True,
                                }
                            ),
                        ]
                    )
            if engine_loop.has_work():
                for request_id, update in engine_loop.step():
                    identity = owners.get(request_id)
                    if identity is None:
                        continue
                    event = _event_from_update(request_id, update)
                    socket.send_multipart([identity, msgpack.packb(event)])
                    if update.finished:
                        del owners[request_id]
    finally:
        socket.close(linger=0)
        context.term()
