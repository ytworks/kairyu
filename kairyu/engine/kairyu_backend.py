"""Kairyu engine core exposed as an EngineBackend (design m2 §1, backend name "kairyu").

Full-stack integration on CPU: OpenAI server / LLM API → this backend →
Scheduler + RadixKVCache + EngineCore step loop. The model forward is a
deterministic toy runner on CPU; the GPU phase swaps in the real ModelRunner
behind the same protocol — nothing above it changes.

Threading discipline (m8 D1): all scheduler mutations — add_request and
stop-string ``finish_early`` — happen on the step thread, between ``update()``
and the next ``schedule()``. The event loop only appends ops and reads queues.
Stop strings are enforced with SSE-safe holdback: text that could still become
part of a stop string is withheld from the stream, so a stop spanning two
deltas never leaks its prefix to a client.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.engine_core import grammar_finished, token_ids
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
from kairyu.engine.core.scheduler import EngineRequest, ScheduledChunk, Scheduler
from kairyu.engine.core.spec_runner import SpeculativeRunner
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree
from kairyu.engine.registry import register_backend
from kairyu.engine.tokenizer import IncrementalDetokenizer, Tokenizer, resolve_tokenizer
from kairyu.outputs import CompletionOutput

_VOCAB_SIZE = 50_000
_DEFAULT_MAX_NEW_TOKENS = 16


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


@dataclass(frozen=True)
class _StreamUpdate:
    outputs: tuple[int, ...]
    text: str
    finished: bool
    finish_reason: str | None
    error: Exception | None = None
    logprobs: tuple[dict[int, float], ...] | None = None
    cumulative_logprob: float = 0.0


def _engine_sampling(params) -> EngineSampling:
    """Map API SamplingParams (+ response_format in extra_args) to the engine
    subset (m8 D2): {"type": "json_object"} -> builtin JSON grammar;
    {"type": "json_schema", "json_schema": {"schema": ...}} -> schema."""
    response_format = (params.extra_args or {}).get("response_format") or {}
    kind = response_format.get("type")
    json_schema = None
    json_mode = kind == "json_object"
    if kind == "json_schema":
        json_schema = (response_format.get("json_schema") or {}).get("schema") or {}
    return EngineSampling(
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        min_p=params.min_p,
        presence_penalty=params.presence_penalty,
        frequency_penalty=params.frequency_penalty,
        repetition_penalty=params.repetition_penalty,
        seed=params.seed,
        logprobs=params.logprobs,
        json_schema=json_schema,
        json_mode=json_mode,
    )


def _logprob_fields(
    meta: list[SampledToken],
) -> tuple[tuple[dict[int, float], ...] | None, float]:
    if not any(token.logprob is not None for token in meta):
        return None, 0.0
    entries = []
    cumulative = 0.0
    for token in meta:
        if token.logprob is None:
            continue
        cumulative += token.logprob
        entry = {token.token_id: token.logprob}
        for top_id, top_lp in token.top_logprobs or ():
            entry.setdefault(top_id, top_lp)
        entries.append(entry)
    return tuple(entries), cumulative


class _RequestTrack:
    """Step-thread-only streaming state for one request."""

    __slots__ = ("detok", "stops", "holdback", "consumed", "stable", "meta", "pending")

    def __init__(self, detok: IncrementalDetokenizer, stops: tuple[str, ...]) -> None:
        self.detok = detok
        self.stops = stops
        self.holdback = max((len(stop) for stop in stops), default=1) - 1
        self.consumed = 0
        self.stable = ""
        self.meta: list[SampledToken] = []  # committed tokens' logprob metadata
        self.pending: list[SampledToken] = []

    def find_stop(self, text: str) -> int | None:
        indices = [index for stop in self.stops if (index := text.find(stop)) != -1]
        return min(indices) if indices else None


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
        validate_tp_degree(tensor_parallel_size)
        if speculative is not None and speculative != "ngram":
            raise ValueError(f"unknown speculative mode {speculative!r} (only 'ngram')")
        if speculative is not None and tensor_parallel_size > 1:
            raise ValueError("speculative decoding with tensor_parallel_size > 1 is not supported")
        self.tensor_parallel_size = tensor_parallel_size
        self._tokenizer = resolve_tokenizer(tokenizer)
        self._cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
        self._scheduler = Scheduler(
            self._cache,
            max_num_batched_tokens=max_num_batched_tokens,
            page_size=page_size,
            speculative_tokens=speculative_tokens if speculative else 0,
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
        if speculative == "ngram":
            self._runner = SpeculativeRunner(self._runner)
        self._ops: deque[tuple[EngineRequest, _RequestTrack]] = deque()
        self._tracked: dict[str, _RequestTrack] = {}  # step-thread only
        self._queues: dict[str, asyncio.Queue] = {}  # event-loop thread only
        self._pump_task: asyncio.Task | None = None

    def _drain_ops(self) -> None:
        while self._ops:
            engine_request, track = self._ops.popleft()
            self._scheduler.add_request(engine_request)
            self._tracked[engine_request.request_id] = track

    def _step(self) -> list[tuple[str, _StreamUpdate]]:
        """One engine step on the step thread; returns per-request updates."""
        self._drain_ops()
        if self._scheduler.has_unfinished():
            plan = self._scheduler.schedule()
            if not plan.scheduled:
                raise RuntimeError("engine stall: nothing schedulable")
            sampled = self._runner.execute(plan.scheduled, self._scheduler.states)
            if sampled:
                finished = self._scheduler.update(token_ids(sampled))
                for request_id in grammar_finished(sampled, finished):
                    # between update() and the next schedule(): safe finish point
                    self._scheduler.finish_early(request_id)
                for request_id, tokens in sampled.items():
                    track = self._tracked.get(request_id)
                    if track is not None:
                        track.pending.extend(tokens)
        updates: list[tuple[str, _StreamUpdate]] = []
        for request_id, track in list(self._tracked.items()):
            update = self._track_update(request_id, track)
            if update is None:
                continue
            updates.append((request_id, update))
            if update.finished:
                del self._tracked[request_id]
        return updates

    def _track_update(self, request_id: str, track: _RequestTrack) -> _StreamUpdate | None:
        state = self._scheduler.states.get(request_id)
        if state is None:
            return None
        outputs = self._scheduler.output_tokens(request_id)
        new_ids = outputs[track.consumed :]
        track.consumed = len(outputs)
        # committed tokens are the prefix of this step's pending metadata;
        # discarded (post-terminal) tokens drop with the clear
        track.meta.extend(track.pending[: len(new_ids)])
        track.pending.clear()
        if new_ids:
            track.stable = track.detok.push(new_ids)
        logprobs, cumulative = _logprob_fields(track.meta)
        if state.status.value == "finished":
            full = track.detok.finalize()
            stop_at = track.find_stop(full)
            reason = self._scheduler.finish_reason(request_id) or "length"
            if stop_at is not None:
                return _StreamUpdate(
                    outputs, full[:stop_at], True, "stop",
                    logprobs=logprobs, cumulative_logprob=cumulative,
                )
            return _StreamUpdate(
                outputs, full, True, reason,
                logprobs=logprobs, cumulative_logprob=cumulative,
            )
        stop_at = track.find_stop(track.stable)
        if stop_at is not None:
            # between update() and the next schedule(): the safe finish point
            self._scheduler.finish_early(request_id)
            return _StreamUpdate(
                outputs, track.stable[:stop_at], True, "stop",
                logprobs=logprobs, cumulative_logprob=cumulative,
            )
        visible_end = max(0, len(track.stable) - track.holdback)
        return _StreamUpdate(
            outputs, track.stable[:visible_end], False, None,
            logprobs=logprobs, cumulative_logprob=cumulative,
        )

    async def _pump(self) -> None:
        try:
            while self._ops or self._scheduler.has_unfinished():
                updates = await asyncio.to_thread(self._step)
                for request_id, update in updates:
                    queue = self._queues.get(request_id)
                    if queue is not None:
                        queue.put_nowait(update)
        except Exception as error:
            failure = _StreamUpdate((), "", True, None, error)
            for queue in self._queues.values():
                queue.put_nowait(failure)
        finally:
            self._pump_task = None

    def _submit(self, request: GenerationRequest) -> asyncio.Queue:
        params = request.sampling_params
        engine_request = EngineRequest(
            request_id=request.request_id,
            prompt_token_ids=self._tokenizer.encode(request.prompt),
            max_new_tokens=params.max_tokens or _DEFAULT_MAX_NEW_TOKENS,
            eos_token_id=self._tokenizer.eos_token_id,
            stop_token_ids=tuple(params.stop_token_ids or ()),
            min_tokens=params.min_tokens,
            ignore_eos=params.ignore_eos,
            sampling=_engine_sampling(params),
        )
        track = _RequestTrack(
            detok=IncrementalDetokenizer(self._tokenizer), stops=tuple(params.stop or ())
        )
        self._ops.append((engine_request, track))
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[request.request_id] = queue
        if self._pump_task is None:
            self._pump_task = asyncio.get_running_loop().create_task(self._pump())
        return queue

    def _result(self, request: GenerationRequest, update: _StreamUpdate) -> GenerationResult:
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
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        queue = self._submit(request)
        try:
            while True:
                update: _StreamUpdate = await queue.get()
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
                update: _StreamUpdate = await queue.get()
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
