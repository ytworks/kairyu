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
from collections.abc import AsyncIterator

from kairyu.engine.backend import GenerationRequest, GenerationResult, GenerationUsage
from kairyu.engine.core.engine_service import run_engine_service, sampling_params_to_wire
from kairyu.engine.registry import register_backend
from kairyu.outputs import CompletionOutput, TokenLogprob

_SPAWN_TIMEOUT_S = 30.0
_SHUTDOWN_TIMEOUT_S = 5.0
_RECV_TICK_S = 1.0


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
        tokenizer: str = "toy",
        speculative: str | None = None,
        speculative_tokens: int = 4,
        death_timeout_s: float = 10.0,
    ) -> None:
        if not isinstance(tokenizer, str):
            raise ValueError("kairyu-proc requires a string tokenizer (name or path)")
        self._config = {
            "num_pages": num_pages,
            "page_size": page_size,
            "max_num_batched_tokens": max_num_batched_tokens,
            "tokenizer": tokenizer,
            "speculative": speculative,
            "speculative_tokens": speculative_tokens,
        }
        self._death_timeout_s = death_timeout_s
        self._process = None
        self._socket = None
        self._context = None
        self._receiver: asyncio.Task | None = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._start_lock = asyncio.Lock()
        self._atexit_registered = False

    # -- lifecycle ---------------------------------------------------------

    def _spawn(self) -> int:
        import multiprocessing

        spawn = multiprocessing.get_context("spawn")
        parent_pipe, child_pipe = spawn.Pipe()
        process = spawn.Process(
            target=run_engine_service, args=(child_pipe, self._config), daemon=True
        )
        process.start()
        child_pipe.close()
        if not parent_pipe.poll(_SPAWN_TIMEOUT_S):
            process.kill()
            raise EngineServiceError("engine service did not report its port in time")
        port = parent_pipe.recv()
        parent_pipe.close()
        self._process = process
        if not self._atexit_registered:
            atexit.register(self._kill_process)
            self._atexit_registered = True
        return port

    async def _ensure_started(self) -> None:
        if self._socket is not None:
            return
        async with self._start_lock:
            if self._socket is not None:
                return
            zmq, _ = _import_deps()
            port = await asyncio.to_thread(self._spawn)
            self._context = zmq.asyncio.Context()
            socket = self._context.socket(zmq.DEALER)
            socket.connect(f"tcp://127.0.0.1:{port}")
            self._socket = socket
            self._receiver = asyncio.get_running_loop().create_task(self._receive_loop())

    def _kill_process(self) -> None:
        process = self._process
        if process is not None and process.is_alive():  # pragma: no cover - crash path
            process.kill()

    async def shutdown(self) -> None:
        if self._receiver is not None:
            self._receiver.cancel()
            self._receiver = None
        process = self._process
        if process is None:
            return
        _, msgpack = _import_deps()
        if self._socket is not None:
            try:
                await self._socket.send(msgpack.packb({"op": "shutdown"}))
            except Exception:  # pragma: no cover - socket already dead
                pass
        await asyncio.to_thread(process.join, _SHUTDOWN_TIMEOUT_S)
        if process.is_alive():  # pragma: no cover - hung child
            process.terminate()
            await asyncio.to_thread(process.join, 2.0)
            if process.is_alive():
                process.kill()
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._process = None

    # -- request plumbing ----------------------------------------------------

    async def _receive_loop(self) -> None:
        assert self._socket is not None
        _, msgpack = _import_deps()
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(self._socket.recv(), timeout=_RECV_TICK_S)
                except TimeoutError:
                    if self._queues and not (self._process and self._process.is_alive()):
                        raise EngineServiceError("engine service process died") from None
                    continue
                event = msgpack.unpackb(raw)
                if event.get("op") in ("pong", "bye"):
                    continue
                queue = self._queues.get(event["request_id"])
                if queue is not None:
                    queue.put_nowait(event)
        except asyncio.CancelledError:  # pragma: no cover - clean shutdown
            raise
        except Exception as error:
            for queue in self._queues.values():
                queue.put_nowait({"error": repr(error)})

    async def _submit(self, request: GenerationRequest) -> asyncio.Queue:
        await self._ensure_started()
        assert self._socket is not None
        _, msgpack = _import_deps()
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[request.request_id] = queue
        await self._socket.send(
            msgpack.packb(
                {
                    "op": "add",
                    "request_id": request.request_id,
                    "prompt": request.prompt,
                    "sampling": sampling_params_to_wire(request.sampling_params),
                }
            )
        )
        return queue

    async def _abort(self, request_id: str) -> None:
        if self._socket is None:
            return
        _, msgpack = _import_deps()
        try:
            await self._socket.send(msgpack.packb({"op": "abort", "request_id": request_id}))
        except Exception:  # pragma: no cover - shutdown race
            pass

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
        )

    @staticmethod
    def _raise_on_error(event: dict) -> None:
        if "error" in event:
            raise EngineServiceError(event["error"])

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        queue = await self._submit(request)
        try:
            while True:
                event = await queue.get()
                self._raise_on_error(event)
                if event["finished"]:
                    return self._result(request, event)
        finally:
            self._queues.pop(request.request_id, None)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        queue = await self._submit(request)
        emitted = -1
        finished_cleanly = False
        try:
            while True:
                event = await queue.get()
                self._raise_on_error(event)
                if len(event["outputs"]) > emitted or event["finished"]:
                    emitted = len(event["outputs"])
                    yield self._result(request, event)
                if event["finished"]:
                    finished_cleanly = True
                    return
        finally:
            self._queues.pop(request.request_id, None)
            if not finished_cleanly:
                await self._abort(request.request_id)


register_backend("kairyu-proc", ZmqEngineBackend)
