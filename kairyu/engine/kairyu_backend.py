"""Kairyu engine core exposed as an EngineBackend (design m2 §1, backend name "kairyu").

Full-stack integration on CPU: OpenAI server / LLM API → this backend →
``EngineLoop`` (tokenizer + Scheduler + RadixKVCache + runner). The model
forward is a deterministic toy runner on CPU; the GPU phase swaps in the real
ModelRunner behind the same protocol — nothing above it changes.

Threading discipline (m8 D1): all scheduler mutations happen inside
``EngineLoop.step()`` on the step thread; the event loop only enqueues ops and
reads queues. The ZMQ process-split backend ("kairyu-proc", m8 D6) drives the
same ``EngineLoop`` from a child process.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

from kairyu.engine.backend import GenerationRequest, GenerationResult, GenerationUsage
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk, Scheduler
from kairyu.engine.core.spec_runner import SpeculativeRunner
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree
from kairyu.engine.engine_loop import EngineLoop, StreamUpdate
from kairyu.engine.registry import register_backend
from kairyu.engine.tokenizer import Tokenizer, resolve_tokenizer
from kairyu.outputs import CompletionOutput

_VOCAB_SIZE = 50_000


class _ToyRunner:
    """Deterministic CPU stand-in for the GPU model forward (greedy only —
    sampling params take effect with a Sampler-equipped runner, m8 D2)."""

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids) if state.request.prompt_token_ids else 0
                token_id = (seed + 31 * chunk.position) % _VOCAB_SIZE
                sampled[chunk.request_id] = (SampledToken(token_id),)
        return sampled


def build_engine_loop(
    *,
    num_pages: int = 4096,
    page_size: int = 16,
    max_num_batched_tokens: int = 2048,
    runner: object | None = None,
    tensor_parallel_size: int = 1,
    tokenizer: str | Tokenizer = "toy",
    speculative: str | None = None,
    speculative_tokens: int = 4,
) -> tuple[EngineLoop, RadixKVCache, Scheduler]:
    """Assemble the engine stack; shared by KairyuBackend and the ZMQ service."""
    validate_tp_degree(tensor_parallel_size)
    if speculative is not None and speculative != "ngram":
        raise ValueError(f"unknown speculative mode {speculative!r} (only 'ngram')")
    if speculative is not None and tensor_parallel_size > 1:
        raise ValueError("speculative decoding with tensor_parallel_size > 1 is not supported")
    resolved = resolve_tokenizer(tokenizer)
    cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=max_num_batched_tokens,
        page_size=page_size,
        speculative_tokens=speculative_tokens if speculative else 0,
    )
    if tensor_parallel_size > 1:
        # CPU-testable TP path (design m5 D1/D3): deterministic rank runners
        # over a FakeCommunicator group; outputs are identical to TP=1.
        active: object = TPModelRunner(
            rank_runners=tuple(
                (runner if runner is not None else _ToyRunner())
                for _ in range(tensor_parallel_size)
            ),
            comms=FakeCommunicator.create_group(tensor_parallel_size),
        )
    else:
        active = runner or _ToyRunner()
    if speculative == "ngram":
        active = SpeculativeRunner(active)
    return EngineLoop(resolved, scheduler, active), cache, scheduler


class KairyuBackend:
    def __init__(
        self,
        num_pages: int = 4096,
        page_size: int = 16,
        max_num_batched_tokens: int = 2048,
        runner: object | None = None,
        tensor_parallel_size: int = 1,
        tokenizer: str | Tokenizer = "toy",
        speculative: str | None = None,
        speculative_tokens: int = 4,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self._loop, self._cache, self._scheduler = build_engine_loop(
            num_pages=num_pages,
            page_size=page_size,
            max_num_batched_tokens=max_num_batched_tokens,
            runner=runner,
            tensor_parallel_size=tensor_parallel_size,
            tokenizer=tokenizer,
            speculative=speculative,
            speculative_tokens=speculative_tokens,
        )
        self._queues: dict[str, asyncio.Queue] = {}  # event-loop thread only
        self._pump_task: asyncio.Task | None = None

    async def _pump(self) -> None:
        try:
            while self._loop.has_work():
                updates = await asyncio.to_thread(self._loop.step)
                for request_id, update in updates:
                    queue = self._queues.get(request_id)
                    if queue is not None:
                        queue.put_nowait(update)
        except Exception as error:
            failure = StreamUpdate((), "", True, None, error)
            for queue in self._queues.values():
                queue.put_nowait(failure)
        finally:
            self._pump_task = None

    def _submit(self, request: GenerationRequest) -> asyncio.Queue:
        self._loop.submit(request.request_id, request.prompt, request.sampling_params)
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[request.request_id] = queue
        if self._pump_task is None:
            self._pump_task = asyncio.get_running_loop().create_task(self._pump())
        return queue

    def _result(self, request: GenerationRequest, update: StreamUpdate) -> GenerationResult:
        completion = CompletionOutput(
            index=0,
            text=update.text,
            token_ids=update.outputs,
            cumulative_logprob=update.cumulative_logprob,
            logprobs=update.logprobs,
            finish_reason=update.finish_reason,
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(completion,),
            finished=update.finished,
            usage=GenerationUsage(
                prompt_tokens=update.num_prompt_tokens,
                completion_tokens=len(update.outputs),
                cached_tokens=update.num_cached_tokens,
            ),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        queue = self._submit(request)
        try:
            while True:
                update: StreamUpdate = await queue.get()
                if update.error is not None:
                    raise update.error
                if update.finished:
                    return self._result(request, update)
        finally:
            self._queues.pop(request.request_id, None)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        queue = self._submit(request)
        emitted = -1
        try:
            while True:
                update: StreamUpdate = await queue.get()
                if update.error is not None:
                    raise update.error
                if len(update.outputs) > emitted or update.finished:
                    emitted = len(update.outputs)
                    yield self._result(request, update)
                if update.finished:
                    return
        finally:
            self._queues.pop(request.request_id, None)

    async def shutdown(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()


register_backend("kairyu", KairyuBackend)
