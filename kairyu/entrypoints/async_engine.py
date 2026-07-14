"""vLLM AsyncLLMEngine-signature-compatible async entrypoint (design doc D2).

Wraps any EngineBackend; ``generate`` is an async generator of incremental
``RequestOutput`` snapshots, finishing with ``finished=True``, matching vLLM's
streaming contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from kairyu.engine.backend import EngineBackend, GenerationRequest
from kairyu.outputs import RequestOutput
from kairyu.sampling_params import SamplingParams


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


@dataclass(frozen=True)
class AsyncEngineArgs:
    """Subset of vLLM's AsyncEngineArgs; unknown extras go in ``extra_args``."""

    model: str
    tokenizer: str | None = None
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    seed: int | None = 0
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool | None = None
    trust_remote_code: bool = False
    extra_args: dict = field(default_factory=dict, compare=False)


def _default_backend(args: AsyncEngineArgs) -> EngineBackend:
    if importlib.util.find_spec("vllm") is not None:
        from kairyu.engine.vllm_backend import VLLMBackend

        return VLLMBackend(
            model=args.model,
            enable_prefix_caching=args.enable_prefix_caching,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    from kairyu.engine.mock import MockBackend

    return MockBackend(tensor_parallel_size=args.tensor_parallel_size)


class AsyncLLMEngine:
    def __init__(self, backend: EngineBackend, model: str = "") -> None:
        self._backend = backend
        self.model = model
        self._active: dict[str, asyncio.Event] = {}

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs) -> AsyncLLMEngine:
        return cls(backend=_default_backend(engine_args), model=engine_args.model)

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncIterator[RequestOutput]:
        if request_id in self._active:
            raise ValueError(f"request ID {request_id!r} is already active")
        abort_event = asyncio.Event()
        self._active[request_id] = abort_event
        request = GenerationRequest(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        )
        stream: AsyncIterator | None = None
        next_task: asyncio.Task | None = None
        abort_task: asyncio.Task | None = None
        try:
            stream = aiter(self._backend.stream(request))
            while True:
                next_task = asyncio.create_task(anext(stream))
                abort_task = asyncio.create_task(abort_event.wait())
                done, _ = await asyncio.wait(
                    {next_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if abort_task in done:
                    await _cancel_task(next_task)
                    next_task = None
                    return

                await _cancel_task(abort_task)
                abort_task = None
                try:
                    partial = await next_task
                except StopAsyncIteration:
                    return
                finally:
                    next_task = None
                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=(),
                    outputs=partial.completions,
                    finished=partial.finished,
                )
        finally:
            self._active.pop(request_id, None)
            await _cancel_task(next_task)
            await _cancel_task(abort_task)
            close = getattr(stream, "aclose", None) if stream is not None else None
            if close is not None:
                await close()

    async def abort(self, request_id: str) -> None:
        """Stop streaming a request; unknown ids are ignored (vLLM behavior)."""
        event = self._active.get(request_id)
        if event is not None:
            event.set()

    async def shutdown(self) -> None:
        await self._backend.shutdown()
