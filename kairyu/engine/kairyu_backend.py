"""Kairyu engine core exposed as an EngineBackend (design m2 §1, backend name "kairyu").

Full-stack integration on CPU: OpenAI server / LLM API → this backend →
Scheduler + RadixKVCache + EngineCore step loop. The model forward is a
deterministic toy runner on CPU; the GPU phase swaps in the FlashInfer
ModelRunner behind the same ModelRunner protocol — nothing above it changes.

Tokenization is a placeholder word-hash (the real tokenizer/detokenizer
component arrives with the GPU runner, m2 §5 item 3); completion text is a
readable rendering of toy token ids, clearly not model output.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.scheduler import EngineRequest, ScheduledChunk, Scheduler
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree
from kairyu.engine.registry import register_backend
from kairyu.outputs import CompletionOutput

_VOCAB_SIZE = 50_000
_DEFAULT_MAX_NEW_TOKENS = 16


class _ToyRunner:
    """Deterministic CPU stand-in for the GPU model forward."""

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> dict[str, int]:
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids) if state.request.prompt_token_ids else 0
                sampled[chunk.request_id] = (seed + 31 * chunk.position) % _VOCAB_SIZE
        return sampled


def _tokenize(prompt: str) -> tuple[int, ...]:
    words = prompt.split()
    if not words:
        return (0,)
    return tuple(hash(word) % _VOCAB_SIZE for word in words)


def _render(token_ids: tuple[int, ...]) -> str:
    return " ".join(f"tok{token_id}" for token_id in token_ids)


class KairyuBackend:
    def __init__(
        self,
        num_pages: int = 4096,
        page_size: int = 16,
        max_num_batched_tokens: int = 2048,
        runner: object | None = None,
        tensor_parallel_size: int = 1,
    ) -> None:
        validate_tp_degree(tensor_parallel_size)
        self.tensor_parallel_size = tensor_parallel_size
        self._cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
        self._scheduler = Scheduler(
            self._cache, max_num_batched_tokens=max_num_batched_tokens, page_size=page_size
        )
        if tensor_parallel_size > 1:
            # CPU-testable TP path (design m5 D1/D3): deterministic rank runners
            # over a FakeCommunicator group; outputs are identical to TP=1.
            self._runner = TPModelRunner(
                rank_runners=tuple(
                    (runner if runner is not None else _ToyRunner())
                    for _ in range(tensor_parallel_size)
                ),
                comms=FakeCommunicator.create_group(tensor_parallel_size),
            )
        else:
            self._runner = runner or _ToyRunner()
        self._queues: dict[str, asyncio.Queue] = {}
        self._pump_task: asyncio.Task | None = None

    def _step(self) -> None:
        plan = self._scheduler.schedule()
        if not plan.scheduled:
            if self._scheduler.has_unfinished():
                raise RuntimeError("engine stall: nothing schedulable")
            return
        sampled = self._runner.execute(plan.scheduled, self._scheduler.states)
        if sampled:
            self._scheduler.update(sampled)

    async def _pump(self) -> None:
        try:
            while self._scheduler.has_unfinished():
                await asyncio.to_thread(self._step)
                for request_id, queue in list(self._queues.items()):
                    state = self._scheduler.states.get(request_id)
                    if state is None:
                        continue
                    outputs = self._scheduler.output_tokens(request_id)
                    finished = state.status.value == "finished"
                    queue.put_nowait((outputs, finished, None))
        except Exception as error:
            for queue in self._queues.values():
                queue.put_nowait(((), True, error))
        finally:
            self._pump_task = None

    def _submit(self, request: GenerationRequest) -> asyncio.Queue:
        max_new = request.sampling_params.max_tokens or _DEFAULT_MAX_NEW_TOKENS
        self._scheduler.add_request(
            EngineRequest(
                request_id=request.request_id,
                prompt_token_ids=_tokenize(request.prompt),
                max_new_tokens=max_new,
            )
        )
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[request.request_id] = queue
        if self._pump_task is None:
            self._pump_task = asyncio.get_running_loop().create_task(self._pump())
        return queue

    def _result(
        self, request: GenerationRequest, outputs: tuple[int, ...], finished: bool
    ) -> GenerationResult:
        completion = CompletionOutput(
            index=0,
            text=_render(outputs),
            token_ids=outputs,
            cumulative_logprob=0.0,
            finish_reason="length" if finished else None,
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(completion,),
            finished=finished,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        queue = self._submit(request)
        try:
            while True:
                outputs, finished, error = await queue.get()
                if error is not None:
                    raise error
                if finished:
                    return self._result(request, outputs, finished=True)
        finally:
            self._queues.pop(request.request_id, None)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        queue = self._submit(request)
        emitted = -1
        try:
            while True:
                outputs, finished, error = await queue.get()
                if error is not None:
                    raise error
                if len(outputs) > emitted or finished:
                    emitted = len(outputs)
                    yield self._result(request, outputs, finished=finished)
                if finished:
                    return
        finally:
            self._queues.pop(request.request_id, None)

    async def shutdown(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()


register_backend("kairyu", KairyuBackend)
