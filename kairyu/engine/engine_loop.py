"""Shared production engine loop: ops → schedule-ahead → execute → commit → stream.

One thread-agnostic core used by both process layouts:
``KairyuBackend`` drives it from an asyncio pump (``asyncio.to_thread``), the
ZMQ ``engine_service`` drives it from its socket loop. All scheduler mutations
(submit/abort/stop-string ``finish_early``) happen inside the serialized step
boundary, while production detokenization and output encoding run on one
private serial output lane.

``pipeline_depth=1`` is the historical synchronous behavior. Larger depths
freeze each scheduled step and let the device worker run step N while this
thread schedules N+1. Streaming, stop holdback, grammar termination,
speculation, preemption and P-D carried tokens still commit through this one
loop; there is no alternate production overlap core.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import perf_counter_ns
from typing import Generic, Protocol, TypeVar

from kairyu.engine.backend import GenerationStageMetric
from kairyu.engine.core.engine_core import grammar_finished, token_ids
from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.step_input import snapshot_step
from kairyu.engine.prompt import (
    PromptInput,
    prompt_kind,
    prompt_text,
    supplied_prompt_token_ids,
)
from kairyu.engine.tokenizer import (
    IncrementalDetokenizer,
    Tokenizer,
    tokenize_loglikelihood_continuation,
    tokenizer_encode_is_concurrent_safe,
    tokenizer_prompt_preparation_is_concurrent_safe,
)
from kairyu.models.generation import GenerationDefaults
from kairyu.outputs import TokenLogprob
from kairyu.sampling_params import RESPONSE_FORMAT_EXTRA_ARG, SamplingParams

_DEFAULT_MAX_NEW_TOKENS = 16
_DEFAULT_PIPELINE_DEPTH = 1
_OutputT = TypeVar("_OutputT")
_TRACE_STAGE_ORDER = (
    "tokenize",
    "queue_wait",
    "schedule",
    "prefill",
    "decode_step",
    "detokenize",
)


def _validate_max_model_len(max_model_len: int | None) -> None:
    if max_model_len is not None and (
        type(max_model_len) is not int or max_model_len < 1
    ):
        raise ValueError(
            "max_model_len must be a positive integer or None, "
            f"got {max_model_len!r}"
        )


class _StepHandle(Protocol):
    def result(self) -> dict[str, tuple[SampledToken, ...]]: ...


class _ResolvedHandle:
    """Depth-1 synchronous execution expressed through the common handle seam."""

    def __init__(self, sampled: dict[str, tuple[SampledToken, ...]]) -> None:
        self._sampled = sampled

    def result(self) -> dict[str, tuple[SampledToken, ...]]:
        return self._sampled


@dataclass(frozen=True)
class _PendingStep:
    handle: _StepHandle
    request_ids: tuple[str, ...]
    contains_prefill: bool
    trace_started_ns: int | None = None
    traced_prefill_ids: tuple[str, ...] = ()
    traced_decode_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamUpdate:
    """Cumulative per-request snapshot emitted after each engine step."""

    outputs: tuple[int, ...]
    text: str
    finished: bool
    finish_reason: str | None
    error: Exception | None = None
    logprobs: tuple[dict[int, float], ...] | None = None
    cumulative_logprob: float = 0.0
    num_prompt_tokens: int = 0
    num_cached_tokens: int = 0
    logprob_content: tuple[TokenLogprob, ...] | None = None
    stage_metrics: tuple[GenerationStageMetric, ...] = ()


@dataclass(frozen=True)
class PreparedPrompt:
    """One loop-owned prompt resolution safe to reuse at admission.

    Native HTTP preflight must tokenize text to reject an oversized context
    before response headers are sent.  Carrying this opaque value into
    ``submit`` avoids resolving that same prompt again without creating a
    cross-request prompt cache.
    """

    prompt: PromptInput
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    _owner: object
    tokenize_duration_ns: int | None = None


def engine_sampling_from(params: SamplingParams) -> EngineSampling:
    """Map API SamplingParams (+ response_format in extra_args) to the engine
    subset (m8 D2): {"type": "json_object"} -> builtin JSON grammar;
    {"type": "json_schema", "json_schema": {"schema": ...}} -> schema."""
    response_format = (params.extra_args or {}).get(RESPONSE_FORMAT_EXTRA_ARG) or {}
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
        forced_token_ids=params.forced_token_ids,
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


def _token_logprob(
    decode: Callable[[tuple[int, ...]], str],
    vocabulary: tuple[str, ...],
    token_id: int,
    logprob: float,
) -> TokenLogprob:
    decoded = decode((token_id,))
    token = (
        vocabulary[token_id]
        if type(token_id) is int and 0 <= token_id < len(vocabulary)
        else decoded
    )
    return TokenLogprob(
        token=token,
        token_id=token_id,
        logprob=logprob,
        # Preserve the existing decoded bytes contract independently of the
        # raw vocabulary piece used for the public token string.
        bytes_=tuple(decoded.encode("utf-8")),
    )


def _logprob_content(
    decode: Callable[[tuple[int, ...]], str],
    vocabulary: tuple[str, ...],
    meta: list[SampledToken],
) -> tuple[TokenLogprob, ...] | None:
    if not any(token.logprob is not None for token in meta):
        return None
    entries = []
    for token in meta:
        if token.logprob is None:
            continue
        top = tuple(
            _token_logprob(decode, vocabulary, top_id, top_lp)
            for top_id, top_lp in token.top_logprobs or ()
        )
        base = _token_logprob(
            decode,
            vocabulary,
            token.token_id,
            token.logprob,
        )
        entries.append(
            TokenLogprob(
                token=base.token,
                token_id=base.token_id,
                logprob=base.logprob,
                bytes_=base.bytes_,
                top=top,
            )
        )
    return tuple(entries)


class _IncrementalStopMatcher:
    """Per-request bounded-overlap matcher for cumulative stable text."""

    __slots__ = (
        "_match",
        "_overlap",
        "_searched_length",
        "_stops",
    )

    def __init__(self, stops: tuple[str, ...]) -> None:
        self._stops = stops
        self._overlap = max(
            0, max((len(stop) for stop in stops), default=1) - 1
        )
        self._searched_length = -1
        self._match: int | None = None

    @property
    def overlap(self) -> int:
        return self._overlap

    def find(self, text: str) -> int | None:
        if self._match is not None:
            return self._match
        if len(text) < self._searched_length:
            raise ValueError("stop matcher text must never retract")
        if len(text) == self._searched_length:
            return None

        # A newly completed stop can begin at most max_stop_length - 1
        # characters before the previous tail. Everything earlier was already
        # proven match-free.
        start = (
            0
            if self._searched_length < 0
            else max(0, self._searched_length - self._overlap)
        )
        match: int | None = None
        for stop in self._stops:
            index = text.find(stop, start)
            if index != -1 and (match is None or index < match):
                match = index
        self._searched_length = len(text)
        self._match = match
        return self._match


class _RequestTrack:
    """Step-side streaming state for one request."""

    __slots__ = (
        "detok",
        "stop_matcher",
        "holdback",
        "consumed",
        "stable",
        "meta",
        "pending",
        "num_prompt_tokens",
        "num_cached_tokens",
        "trace_requested",
        "submitted_ns",
        "first_schedule_observed",
        "stage_durations_ns",
        "stage_occurrences",
        "output_barrier",
    )

    def __init__(
        self,
        detok: IncrementalDetokenizer,
        stops: tuple[str, ...],
        num_prompt_tokens: int,
        *,
        trace_requested: bool = False,
        tokenize_duration_ns: int | None = None,
        submitted_ns: int | None = None,
    ) -> None:
        self.detok = detok
        self.stop_matcher = _IncrementalStopMatcher(stops)
        self.holdback = self.stop_matcher.overlap
        self.consumed = 0
        self.stable = ""
        self.meta: list[SampledToken] = []  # committed tokens' logprob metadata
        self.pending: list[SampledToken] = []
        self.num_prompt_tokens = num_prompt_tokens
        self.num_cached_tokens = 0
        self.trace_requested = trace_requested
        self.submitted_ns = submitted_ns
        self.first_schedule_observed = False
        self.stage_durations_ns: dict[str, int] | None = (
            {} if trace_requested else None
        )
        self.stage_occurrences: dict[str, int] | None = (
            {} if trace_requested else None
        )
        # Stop matching can feed ``finish_early`` back into the scheduler, and
        # trace accumulation currently shares one request-local metrics map.
        # Either condition must finish presentation before the next schedule.
        self.output_barrier = bool(stops) or trace_requested
        if tokenize_duration_ns is not None:
            self.record_stage("tokenize", tokenize_duration_ns)

    def find_stop(self, text: str) -> int | None:
        return self.stop_matcher.find(text)

    def record_stage(self, stage: str, duration_ns: int) -> None:
        durations = self.stage_durations_ns
        occurrences = self.stage_occurrences
        if durations is None or occurrences is None:
            return
        durations[stage] = durations.get(stage, 0) + max(0, duration_ns)
        occurrences[stage] = occurrences.get(stage, 0) + 1

    def metrics(self) -> tuple[GenerationStageMetric, ...]:
        durations = self.stage_durations_ns
        occurrences = self.stage_occurrences
        if durations is None or occurrences is None:
            return ()
        return tuple(
            GenerationStageMetric(
                stage=stage,
                duration_ns=durations[stage],
                occurrences=occurrences[stage],
            )
            for stage in _TRACE_STAGE_ORDER
            if stage in durations
        )


@dataclass(frozen=True)
class _TrackUpdateInput:
    """Scheduler-owned facts copied before presentation leaves the step thread."""

    request_id: str
    track: _RequestTrack
    outputs: tuple[int, ...]
    new_ids: tuple[int, ...]
    new_meta: tuple[SampledToken, ...]
    finished: bool
    state_finish_reason: str | None
    request: EngineRequest
    num_cached_tokens: int
    finish_reason: str | None


@dataclass(frozen=True)
class _TrackUpdateResult:
    request_id: str
    update: StreamUpdate
    finish_early: bool = False


@dataclass(frozen=True)
class _StepOutputBatch:
    inputs: tuple[_TrackUpdateInput, ...]
    requires_barrier: bool


@dataclass(frozen=True)
class _PendingOutput(Generic[_OutputT]):
    future: Future[tuple[list[_TrackUpdateResult], _OutputT]]
    requires_barrier: bool


@dataclass
class _AddBatch:
    requests: list[EngineRequest]
    tracks: list[_RequestTrack]


@dataclass
class _AbortBatch:
    request_ids: list[str]


_OpBatch = _AddBatch | _AbortBatch


class EngineLoop:
    """Owns tokenizer + scheduler + runner; drains ops and produces updates."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        scheduler: Scheduler,
        runner: object,
        default_eos_token_id: int | None = None,
        default_stop_token_ids: tuple[int, ...] = (),
        pipeline_depth: int = _DEFAULT_PIPELINE_DEPTH,
        max_model_len: int | None = None,
        generation_defaults: GenerationDefaults | None = None,
    ) -> None:
        if pipeline_depth < 1:
            raise ValueError(f"pipeline_depth must be >= 1, got {pipeline_depth}")
        _validate_max_model_len(max_model_len)
        self._tokenizer = tokenizer
        self._prompt_vocab_size: int | None = None
        self._logprob_vocab: tuple[str, ...] | None = None
        self._scheduler = scheduler
        self._runner = runner
        self._pipeline_depth = pipeline_depth
        self._max_model_len = max_model_len
        self.generation_defaults = generation_defaults or GenerationDefaults(
            mode="none",
            source="disabled",
        )
        self._prepared_prompt_owner = object()
        self._step_lock = Lock()
        self._step_index = 0
        self._pending_steps: deque[_PendingStep] = deque()
        self._pending_by_request: dict[str, int] = {}
        self._deferred_forget: set[str] = set()
        # A synchronous runner needs a dedicated serial device lane only when
        # overlap is requested. Native async/PP runners provide submit() and do
        # not need this executor; depth 1 remains allocation-free and immediate.
        self._device_executor: ThreadPoolExecutor | None = None
        if pipeline_depth > 1 and not callable(getattr(runner, "submit", None)):
            self._device_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kairyu-device"
            )
        # Created only by production drivers. Direct ``step()`` callers retain
        # the allocation-free synchronous compatibility path.
        self._output_executor: ThreadPoolExecutor | None = None
        # generation_config.json may carry an eos LIST (m12 D5): first entry
        # is eos, the rest are stop tokens; falls back to the tokenizer's eos
        self._default_eos = (
            default_eos_token_id
            if default_eos_token_id is not None
            else tokenizer.eos_token_id
        )
        self._default_stop_ids = default_stop_token_ids
        # Producers can submit/abort from arbitrary threads while the step
        # thread drains a frozen batch snapshot. The lock also makes duplicate
        # request reservation atomic instead of relying on individual GIL ops.
        self._ops_lock = Lock()
        self._ops: deque[_OpBatch] = deque()
        self._abort_requested: set[str] = set()
        self._closed = False
        self._tracked: dict[str, _RequestTrack] = {}  # step-side only
        self._traced_requests = 0
        self._active_request_ids: set[str] = set()

    def _activate_track(self, request_id: str, track: _RequestTrack) -> None:
        self._tracked[request_id] = track
        if track.trace_requested:
            self._traced_requests += 1

    @property
    def tokenizer_concurrent_encode_safe(self) -> bool:
        """Whether request-local encode calls may overlap across workers."""

        return tokenizer_encode_is_concurrent_safe(self._tokenizer)

    @property
    def tokenizer_concurrent_prompt_preparation_safe(self) -> bool:
        """Whether complete request preparation may overlap across workers."""

        return tokenizer_prompt_preparation_is_concurrent_safe(self._tokenizer)

    def _remove_track(self, request_id: str) -> _RequestTrack | None:
        track = self._tracked.pop(request_id, None)
        if track is not None and track.trace_requested:
            self._traced_requests -= 1
        return track

    def tokenize_prompt(self, prompt: str) -> tuple[int, ...]:
        prompt_token_ids = self._tokenizer.encode(prompt)
        if not prompt_token_ids:
            raise ValueError("prompt must tokenize to at least one token")
        return prompt_token_ids

    def tokenize_loglikelihood(
        self,
        context: str,
        continuation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Resolve one exact continuation boundary with the engine tokenizer."""

        return tokenize_loglikelihood_continuation(
            self._tokenizer,
            context,
            continuation,
        )

    def resolve_prompt_token_ids(self, prompt: PromptInput) -> tuple[int, ...]:
        """Resolve public input while preserving caller-owned token IDs."""

        if prompt_kind(prompt) == "multimodal":
            raise ValueError(
                "Kairyu backend does not support multimodal prompts; "
                "no modality processor is configured"
            )
        prompt_token_ids = supplied_prompt_token_ids(prompt)
        if prompt_token_ids is not None:
            if self._prompt_vocab_size is None:
                self._prompt_vocab_size = len(self._tokenizer.vocab())
                if self._prompt_vocab_size < 1:
                    raise ValueError("tokenizer vocabulary must not be empty")
            for index, token_id in enumerate(prompt_token_ids):
                if token_id >= self._prompt_vocab_size:
                    raise ValueError(
                        f"prompt_token_ids[{index}]={token_id} is outside the "
                        f"backend tokenizer vocabulary of "
                        f"{self._prompt_vocab_size} entries"
                    )
            return prompt_token_ids
        text = prompt_text(prompt)
        assert text is not None
        return self.tokenize_prompt(text)

    @staticmethod
    def _max_new_tokens(params: SamplingParams) -> int:
        return (
            params.max_tokens
            if params.max_tokens is not None
            else _DEFAULT_MAX_NEW_TOKENS
        )

    def _validate_context_length(
        self,
        prompt_token_ids: tuple[int, ...],
        max_new_tokens: int,
    ) -> None:
        requested_length = len(prompt_token_ids) + max_new_tokens
        if (
            self._max_model_len is not None
            and requested_length > self._max_model_len
        ):
            raise ValueError(
                f"prompt tokens ({len(prompt_token_ids)}) plus max_tokens "
                f"({max_new_tokens}) exceed max_model_len "
                f"({self._max_model_len})"
            )

    def prepare_prompt(
        self,
        prompt: PromptInput,
        params: SamplingParams,
        *,
        trace_requested: bool = False,
    ) -> PreparedPrompt:
        """Resolve and validate a prompt once without reserving a request ID."""

        if type(trace_requested) is not bool:
            raise ValueError("trace_requested must be a boolean")
        tokenize_started_ns = perf_counter_ns() if trace_requested else None
        prompt_token_ids = self.resolve_prompt_token_ids(prompt)
        max_new_tokens = self._max_new_tokens(params)
        self._validate_context_length(prompt_token_ids, max_new_tokens)
        tokenize_duration_ns = (
            max(0, perf_counter_ns() - tokenize_started_ns)
            if tokenize_started_ns is not None
            else None
        )
        return PreparedPrompt(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            _owner=self._prepared_prompt_owner,
            tokenize_duration_ns=tokenize_duration_ns,
        )

    def validate_prompt_length(
        self,
        prompt: PromptInput,
        params: SamplingParams,
    ) -> None:
        """Validate the tokenized request context without reserving its ID."""

        self.prepare_prompt(prompt, params)

    def submit(
        self,
        request_id: str,
        prompt: PromptInput,
        params: SamplingParams,
        priority: int = 0,
        scheduling_class: str = "interactive",
        prepared_prompt: PreparedPrompt | None = None,
        trace_requested: bool = False,
    ) -> None:
        if type(trace_requested) is not bool:
            raise ValueError("trace_requested must be a boolean")
        params = self.generation_defaults.apply(params)
        # Advisory fast rejection avoids tokenization and lock traffic for the
        # common duplicate case. The lock-protected check below remains the
        # authority when producers race.
        if request_id in self._active_request_ids:
            raise ValueError(f"duplicate request_id {request_id!r}")
        if prepared_prompt is None:
            prepared_prompt = self.prepare_prompt(
                prompt,
                params,
                trace_requested=trace_requested,
            )
        else:
            if prepared_prompt._owner is not self._prepared_prompt_owner:
                raise ValueError("prepared prompt belongs to a different engine loop")
            if prepared_prompt.prompt != prompt:
                raise ValueError("prepared prompt does not match the submitted prompt")
            max_new_tokens = self._max_new_tokens(params)
            if prepared_prompt.max_new_tokens != max_new_tokens:
                raise ValueError(
                    "prepared prompt max_tokens does not match the submitted request"
                )
            self._validate_context_length(
                prepared_prompt.prompt_token_ids,
                max_new_tokens,
            )
        prompt_token_ids = prepared_prompt.prompt_token_ids
        max_new_tokens = prepared_prompt.max_new_tokens
        forced_continuation = params.forced_token_ids is not None
        engine_request = EngineRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self._default_eos,
            stop_token_ids=(
                ()
                if forced_continuation
                else tuple(params.stop_token_ids or ()) + self._default_stop_ids
            ),
            min_tokens=0 if forced_continuation else params.min_tokens,
            ignore_eos=True if forced_continuation else params.ignore_eos,
            priority=priority,
            scheduling_class=scheduling_class,
            sampling=engine_sampling_from(params),
        )
        track = _RequestTrack(
            detok=IncrementalDetokenizer(
                self._tokenizer,
                skip_special_tokens=params.skip_special_tokens,
            ),
            stops=() if forced_continuation else tuple(params.stop or ()),
            num_prompt_tokens=len(engine_request.prompt_token_ids),
            trace_requested=trace_requested,
            tokenize_duration_ns=(
                prepared_prompt.tokenize_duration_ns if trace_requested else None
            ),
            submitted_ns=(perf_counter_ns() if trace_requested else None),
        )
        with self._ops_lock:
            if self._closed:
                raise RuntimeError("engine loop is closed")
            if request_id in self._active_request_ids:
                raise ValueError(f"duplicate request_id {request_id!r}")
            self._active_request_ids.add(request_id)
            if self._ops and isinstance(self._ops[-1], _AddBatch):
                batch = self._ops[-1]
                batch.requests.append(engine_request)
                batch.tracks.append(track)
            else:
                self._ops.append(_AddBatch([engine_request], [track]))

    def abort(self, request_id: str) -> None:
        # Read-only optimistic checks are safe under CPython's GIL and only
        # reject operations that already have a definitive lifecycle result.
        # A possible false negative enters the lock and is checked again.
        if (
            request_id not in self._active_request_ids
            or request_id in self._abort_requested
        ):
            return
        with self._ops_lock:
            if (
                self._closed
                or request_id not in self._active_request_ids
                or request_id in self._abort_requested
            ):
                return
            self._abort_requested.add(request_id)
            if self._ops and isinstance(self._ops[-1], _AbortBatch):
                self._ops[-1].request_ids.append(request_id)
            else:
                self._ops.append(_AbortBatch([request_id]))

    def purge(self, request_ids: tuple[str, ...]) -> None:
        """Abort and forget requests after a fatal runner step.

        Called only when no ``step()`` call is active. Pending adds and aborts
        for the purged ids are removed before scheduler and runner state is
        reclaimed.
        """
        # A depth>1 step may fail while later immutable snapshots are already
        # queued on the serial device lane. Let those runner calls settle before
        # releasing sampler/device state; their scheduler results are discarded
        # and abort() below releases every outstanding reservation.
        pending_ids = self._discard_pending_steps()
        # A fatal runner step invalidates every already-submitted snapshot, not
        # only the public queues the caller happened to name. Discarding a
        # pending result without aborting its request would strand in_flight
        # reservations forever.
        ids = set(request_ids) | pending_ids
        # Filter queued batches under the producer lock. Concurrent producers
        # append only after this snapshot is restored, so purge cannot lose an
        # unrelated add/abort.
        with self._ops_lock:
            retained: deque[_OpBatch] = deque()
            for batch in self._ops:
                if isinstance(batch, _AddBatch):
                    kept = [
                        (request, track)
                        for request, track in zip(
                            batch.requests, batch.tracks, strict=True
                        )
                        if request.request_id not in ids
                    ]
                    if kept:
                        retained.append(
                            _AddBatch(
                                [request for request, _track in kept],
                                [track for _request, track in kept],
                            )
                        )
                else:
                    request_ids = [
                        request_id
                        for request_id in batch.request_ids
                        if request_id not in ids
                    ]
                    if request_ids:
                        retained.append(_AbortBatch(request_ids))
            self._ops.clear()
            self._ops.extend(retained)
            self._abort_requested.difference_update(ids)
            active = tuple(ids & self._active_request_ids)
        for request_id in active:
            self._scheduler.abort(request_id)
            self._remove_track(request_id)
            self._forget(request_id)

    def has_work(self) -> bool:
        return (
            self._has_queued_ops()
            or bool(self._pending_steps)
            or self._scheduler.has_unfinished()
        )

    @property
    def pipeline_depth(self) -> int:
        return self._pipeline_depth

    @property
    def max_model_len(self) -> int | None:
        return self._max_model_len

    def _take_ops(self) -> tuple[_OpBatch, ...]:
        """Freeze one step-boundary snapshot; later producers target a new queue."""
        with self._ops_lock:
            batches = tuple(self._ops)
            self._ops.clear()
        return batches

    def _has_queued_ops(self) -> bool:
        """Whether a producer wrote after the current step boundary."""

        with self._ops_lock:
            return bool(self._ops)

    def _restore_ops(self, batches: list[_OpBatch]) -> None:
        """Put unprocessed older work ahead of concurrent producer batches."""
        if not batches:
            return
        with self._ops_lock:
            for batch in reversed(batches):
                self._ops.appendleft(batch)

    def _drain_ops(self) -> None:
        batches = self._take_ops()
        for batch_index, batch in enumerate(batches):
            if isinstance(batch, _AddBatch):
                bulk_add = getattr(self._scheduler, "add_requests_atomic", None)
                if len(batch.requests) > 1 and callable(bulk_add):
                    try:
                        bulk_add(batch.requests)
                    except Exception:
                        # The explicit contract is no mutation on failure. Fall
                        # through one-by-one to preserve the historical failing
                        # request and partial-progress semantics.
                        pass
                    else:
                        for request, track in zip(
                            batch.requests, batch.tracks, strict=True
                        ):
                            self._activate_track(request.request_id, track)
                        continue

                for request_index, (engine_request, track) in enumerate(
                    zip(batch.requests, batch.tracks, strict=True)
                ):
                    try:
                        self._scheduler.add_request(engine_request)
                    except Exception:
                        with self._ops_lock:
                            self._active_request_ids.discard(
                                engine_request.request_id
                            )
                            self._abort_requested.discard(engine_request.request_id)
                        remaining: list[_OpBatch] = []
                        if request_index + 1 < len(batch.requests):
                            remaining.append(
                                _AddBatch(
                                    batch.requests[request_index + 1 :],
                                    batch.tracks[request_index + 1 :],
                                )
                            )
                        remaining.extend(batches[batch_index + 1 :])
                        self._restore_ops(remaining)
                        raise
                    self._activate_track(engine_request.request_id, track)
                continue

            for abort_index, request_id in enumerate(batch.request_ids):
                try:
                    if request_id in self._tracked:
                        self._scheduler.abort(request_id)
                except Exception:
                    with self._ops_lock:
                        self._abort_requested.discard(request_id)
                    remaining = []
                    if abort_index + 1 < len(batch.request_ids):
                        remaining.append(
                            _AbortBatch(batch.request_ids[abort_index + 1 :])
                        )
                    remaining.extend(batches[batch_index + 1 :])
                    self._restore_ops(remaining)
                    raise

    def step(self) -> list[tuple[str, StreamUpdate]]:
        """Serialize public step calls around the unified pipeline state."""
        with self._step_lock:
            batch = self._step_once()
            results = self._materialize_updates(batch.inputs)
            return self._apply_update_results(results)

    def advance_output(self) -> _StepOutputBatch:
        """Advance one core step and return immutable presentation inputs.

        Production drivers may hold at most this one raw batch while one prior
        output is outstanding; ``step()`` remains the unrestricted public
        compatibility entry point.
        """

        with self._step_lock:
            return self._step_once()

    def start_output(
        self,
        batch: _StepOutputBatch,
        transform: Callable[[list[tuple[str, StreamUpdate]]], _OutputT],
    ) -> _PendingOutput[_OutputT]:
        """Submit one batch to the serial output lane.

        Production drivers call this only after the prior pending output has
        been finished. That driver precondition bounds the executor to one
        outstanding item without adding a second queue protocol here.
        """

        with self._step_lock:
            executor = self._output_executor
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="kairyu-output",
                )
                self._output_executor = executor
            future = executor.submit(
                self._materialize_and_transform,
                batch.inputs,
                transform,
            )
        return _PendingOutput(future, batch.requires_barrier)

    def finish_output(self, pending: _PendingOutput[_OutputT]) -> _OutputT:
        """Resolve one output batch and apply any stop-string feedback."""

        results, output = pending.future.result()
        has_feedback = any(result.finish_early for result in results)
        if has_feedback and not pending.requires_barrier:
            raise RuntimeError("output feedback escaped its scheduler barrier")
        if has_feedback:
            # Production drivers never advance past a barrier batch, so this
            # lock reacquires the same serialized scheduler safe point.
            with self._step_lock:
                self._apply_update_results(results)
        return output

    def _step_once(self) -> _StepOutputBatch:
        """Advance the unified pipeline and snapshot presentation inputs.

        At most one submitted device step is committed per call. Before that
        commit, the scheduler fills the configured pipeline depth with frozen
        snapshots. This preserves the historical streaming cadence while
        allowing scheduling and the oldest device execution to overlap.
        """
        self._drain_ops()
        while self._scheduler.has_unfinished() and (
            len(self._pending_steps) < self._schedule_ahead_limit()
        ):
            if self._has_queued_ops():
                # A producer arrived after this step's frozen op snapshot. Do
                # not bury that admission/abort behind the remaining decode
                # horizon: commit the oldest already-submitted handle, then
                # drain the producer queue at the next public step boundary.
                break
            drain_before_admission = getattr(
                self._scheduler,
                "should_drain_before_admission",
                None,
            )
            if (
                self._device_executor is not None
                and
                self._pending_steps
                and callable(drain_before_admission)
                and drain_before_admission()
            ):
                # Every live request has its length-terminal token in an older
                # immutable handle. Commit that bounded work before filling a
                # partially empty sequence window, then re-evaluate admission
                # with the newly released slots on the next public step.
                break
            schedule_started_ns = (
                perf_counter_ns() if self._traced_requests else None
            )
            plan = self._scheduler.schedule()
            if schedule_started_ns is not None:
                schedule_duration_ns = max(
                    0,
                    perf_counter_ns() - schedule_started_ns,
                )
                traced_participants: set[str] = set()
                for chunk in plan.scheduled:
                    track = self._tracked.get(chunk.request_id)
                    if track is None or not track.trace_requested:
                        continue
                    traced_participants.add(chunk.request_id)
                for request_id in traced_participants:
                    track = self._tracked[request_id]
                    if not track.first_schedule_observed:
                        submitted_ns = track.submitted_ns
                        if submitted_ns is not None:
                            track.record_stage(
                                "queue_wait",
                                schedule_started_ns - submitted_ns,
                            )
                        track.first_schedule_observed = True
                    track.record_stage("schedule", schedule_duration_ns)
            # prompts too large to ever fit are rejected in schedule() (C2);
            # their tracks surface as finished via _track_update below
            self._scheduler.drain_rejected()
            # P-D adoption commits token 0 during schedule(), outside execute().
            # Record its metadata immediately; a decode sampled by this plan is
            # appended later at commit, preserving per-request token order.
            for request_id, token in self._drain_carried().items():
                track = self._tracked.get(request_id)
                if track is not None:
                    track.pending.append(token)
            if not plan.scheduled:
                # This is either one unadmittable waiting head or a violated
                # scheduler capacity invariant, unless an adapter explicitly
                # reports a control-only transition such as P-D prefill/KV
                # handoff. An empty plan is also expected when every remaining
                # token is already represented by an older in-flight handle.
                progress_hook = getattr(
                    self._scheduler, "made_control_progress", None
                )
                made_control_progress = (
                    bool(progress_hook()) if callable(progress_hook) else False
                )
                # A waiting head can be temporarily blocked by pages owned by a
                # decode whose result is already in flight. Reject only when no
                # pending device step can commit/finish and free that capacity.
                rejected = None
                if (
                    not self._pending_steps
                    and not made_control_progress
                    and self._scheduler.has_unfinished()
                ):
                    rejected = self._scheduler.reject_waiting_head()
                if (
                    rejected is None
                    and self._scheduler.has_unfinished()
                    and not made_control_progress
                    and not self._pending_steps
                ):
                    raise RuntimeError(
                        "scheduler made no progress with running requests"
                    )
                # One control-only/rejection transition per public step keeps
                # P-D chunked-prefill streaming responsive and avoids a busy
                # loop. With an older device step pending, commit it now.
                break
            self._submit_step(plan.scheduled)
            if self._needs_commit_barrier(plan.scheduled):
                break

        if self._pending_steps:
            self._commit_oldest()

        inputs: list[_TrackUpdateInput] = []
        requires_barrier = False
        terminal_request_ids: list[str] = []
        for request_id, track in list(self._tracked.items()):
            state = self._scheduler.states.get(request_id)
            if state is None:
                requires_barrier = requires_barrier or track.output_barrier
                continue
            outputs = tuple(self._scheduler.output_tokens(request_id))
            new_ids = outputs[track.consumed :]
            track.consumed = len(outputs)
            # Committed tokens are the prefix of this step's pending metadata;
            # discarded post-terminal tokens drop with the clear.
            new_meta = tuple(track.pending[: len(new_ids)])
            track.pending.clear()
            finished = state.status.value == "finished"
            if finished:
                terminal_request_ids.append(request_id)
            elif track.output_barrier:
                requires_barrier = True
            inputs.append(
                _TrackUpdateInput(
                    request_id=request_id,
                    track=track,
                    outputs=outputs,
                    new_ids=new_ids,
                    new_meta=new_meta,
                    finished=finished,
                    state_finish_reason=state.finish_reason,
                    request=state.request,
                    num_cached_tokens=self._scheduler.num_cached_tokens(
                        request_id
                    ),
                    finish_reason=(
                        self._scheduler.finish_reason(request_id)
                        if finished
                        else None
                    ),
                )
            )
        # A scheduler-known terminal can be reclaimed before presentation: the
        # output worker owns the captured track object, while future core steps
        # must neither resnapshot it nor delay unrelated requests. Stop-string
        # terminals are not scheduler-known and remain behind the barrier.
        for request_id in terminal_request_ids:
            self._remove_track(request_id)
            if self._pending_by_request.get(request_id, 0):
                self._deferred_forget.add(request_id)
            else:
                self._forget(request_id)
        return _StepOutputBatch(tuple(inputs), requires_barrier)

    def _submit_step(self, scheduled: tuple) -> None:
        """Freeze and submit one scheduler plan to the common handle contract."""
        step = snapshot_step(scheduled, self._scheduler.states)
        traced_prefill_ids: tuple[str, ...] = ()
        traced_decode_ids: tuple[str, ...] = ()
        if self._traced_requests:
            traced_prefill_ids = tuple(
                dict.fromkeys(
                    chunk.request_id
                    for chunk in step.chunks
                    if chunk.is_prefill
                    and (track := self._tracked.get(chunk.request_id)) is not None
                    and track.trace_requested
                )
            )
            traced_decode_ids = tuple(
                dict.fromkeys(
                    chunk.request_id
                    for chunk in step.chunks
                    if not chunk.is_prefill
                    and (track := self._tracked.get(chunk.request_id)) is not None
                    and track.trace_requested
                )
            )
        trace_started_ns = (
            perf_counter_ns()
            if traced_prefill_ids or traced_decode_ids
            else None
        )
        submit = getattr(self._runner, "submit", None)
        if callable(submit):
            handle = submit(self._step_index, step.chunks, step.states_view())
        elif self._device_executor is not None:
            handle = self._device_executor.submit(
                self._runner.execute, step.chunks, step.states_view()
            )
        else:
            handle = _ResolvedHandle(
                self._runner.execute(step.chunks, step.states_view())
            )
        request_ids = tuple(dict.fromkeys(chunk.request_id for chunk in step.chunks))
        self._pending_steps.append(
            _PendingStep(
                handle,
                request_ids,
                contains_prefill=any(chunk.is_prefill for chunk in step.chunks),
                trace_started_ns=trace_started_ns,
                traced_prefill_ids=traced_prefill_ids,
                traced_decode_ids=traced_decode_ids,
            )
        )
        for request_id in request_ids:
            self._pending_by_request[request_id] = (
                self._pending_by_request.get(request_id, 0) + 1
            )
        self._step_index += 1

    def _schedule_ahead_limit(self) -> int:
        """Bound unresolved admission work while retaining deep decode overlap.

        SGLang's overlap scheduler and vLLM's async scheduler both keep at most
        the previous and current batches unresolved while incoming/prefill work
        can still change the cohort.  Kairyu may use its deeper configured
        horizon once every outstanding snapshot and scheduler request is pure
        decode.  A scheduler adapter without the read-only prefill signal cannot
        prove that condition, so it stays on the fail-safe two-step horizon.
        """

        admission_depth = min(self._pipeline_depth, 2)
        if any(step.contains_prefill for step in self._pending_steps):
            return admission_depth
        has_prefill_work = getattr(self._scheduler, "has_prefill_work", None)
        if not callable(has_prefill_work) or has_prefill_work():
            return admission_depth
        return self._pipeline_depth

    def _needs_commit_barrier(self, scheduled: tuple) -> bool:
        """Whether the next plan depends on this step's variable-length result.

        A speculative chunk can accept anywhere from one to ``k+1`` tokens, so
        its successor position cannot be snapshotted before verification. The
        prompt-completing sample is also committed before the first speculative
        decode; otherwise a depth>1 loop permanently keeps one plain token in
        flight and the scheduler's safe ``in_flight == 0`` speculation gate is
        never reached.
        """
        if any(not chunk.is_prefill and chunk.num_tokens > 1 for chunk in scheduled):
            return True
        if not getattr(self._scheduler, "speculative_tokens", 0):
            return False
        states = self._scheduler.states
        return any(
            chunk.is_prefill and states[chunk.request_id].prefill_done
            for chunk in scheduled
        )

    def _commit_oldest(self) -> None:
        pending = self._pending_steps.popleft()
        try:
            try:
                sampled = pending.handle.result()
            finally:
                self._record_pending_observation(pending)
            if sampled:
                finished = self._scheduler.update(token_ids(sampled))
                for request_id in grammar_finished(sampled, finished):
                    # between update() and the next schedule(): safe finish point
                    self._scheduler.finish_early(request_id)
                for request_id, tokens in sampled.items():
                    track = self._tracked.get(request_id)
                    if track is not None:
                        track.pending.extend(tokens)
        finally:
            self._release_pending_counts(pending.request_ids)

    def _record_pending_observation(self, pending: _PendingStep) -> None:
        started_ns = pending.trace_started_ns
        if started_ns is None:
            return
        duration_ns = max(0, perf_counter_ns() - started_ns)
        for request_id in pending.traced_prefill_ids:
            track = self._tracked.get(request_id)
            if track is not None:
                track.record_stage("prefill", duration_ns)
        for request_id in pending.traced_decode_ids:
            track = self._tracked.get(request_id)
            if track is not None:
                track.record_stage("decode_step", duration_ns)

    def _release_pending_counts(self, request_ids: tuple[str, ...]) -> None:
        for request_id in request_ids:
            remaining = self._pending_by_request[request_id] - 1
            if remaining:
                self._pending_by_request[request_id] = remaining
                continue
            del self._pending_by_request[request_id]
            if request_id in self._deferred_forget:
                self._deferred_forget.remove(request_id)
                self._forget(request_id)

    def _discard_pending_steps(self) -> set[str]:
        discarded_ids: set[str] = set()
        while self._pending_steps:
            pending = self._pending_steps.popleft()
            discarded_ids.update(pending.request_ids)
            try:
                pending.handle.result()
            except Exception:
                pass
            finally:
                self._release_pending_counts(pending.request_ids)
        return discarded_ids

    def close(self) -> None:
        """Settle outstanding work and release private execution lanes."""
        # asyncio cancellation cannot stop an already-running to_thread(step).
        # Waiting on the same lock prevents shutdown from releasing runner state
        # underneath that call.
        with self._step_lock:
            if self._output_executor is not None:
                self._output_executor.shutdown(wait=True, cancel_futures=True)
                self._output_executor = None
            self._discard_pending_steps()
            # Reclaim live requests too: shutdown may interrupt a long stream
            # after its current device call. Pending adds have no scheduler
            # state yet, but _forget still releases any runner-local residue.
            with self._ops_lock:
                self._closed = True
                active = tuple(self._active_request_ids)
                self._ops.clear()
                self._abort_requested.clear()
            for request_id in active:
                self._scheduler.abort(request_id)
                self._remove_track(request_id)
                self._forget(request_id)
            self._deferred_forget.clear()
            if self._device_executor is not None:
                self._device_executor.shutdown(wait=True, cancel_futures=True)
                self._device_executor = None

    def _drain_carried(self) -> dict[str, SampledToken]:
        """Tokens a scheduler committed without a runner reporting them.

        Only the P-D adapter has any: ``resume_with_kv`` writes token 0 straight
        into the decode request's outputs, so the loop would otherwise pair the
        completion's SECOND token's metadata with its first — or, at
        ``max_tokens=1``, report no logprobs at all because no decode step ran.
        """
        drain = getattr(self._scheduler, "drain_carried_tokens", None)
        return drain() if drain is not None else {}

    def _forget(self, request_id: str) -> None:
        """Reclaim finished per-request state in the scheduler and runner (E2)."""
        try:
            self._scheduler.forget(request_id)
            release = getattr(self._runner, "release", None)
            if release is not None:
                release(request_id)
        finally:
            with self._ops_lock:
                self._active_request_ids.discard(request_id)
                self._abort_requested.discard(request_id)

    def _materialize_updates(
        self,
        inputs: list[_TrackUpdateInput],
    ) -> list[_TrackUpdateResult]:
        return [self._materialize_update(item) for item in inputs]

    def _materialize_and_transform(
        self,
        inputs: list[_TrackUpdateInput],
        transform: Callable[[list[tuple[str, StreamUpdate]]], _OutputT],
    ) -> tuple[list[_TrackUpdateResult], _OutputT]:
        results = self._materialize_updates(inputs)
        updates = [(result.request_id, result.update) for result in results]
        return results, transform(updates)

    def _apply_update_results(
        self,
        results: list[_TrackUpdateResult],
    ) -> list[tuple[str, StreamUpdate]]:
        updates: list[tuple[str, StreamUpdate]] = []
        for result in results:
            request_id = result.request_id
            update = result.update
            if result.finish_early:
                # The output lane can discover a stop string, but the scheduler
                # mutation remains on the serialized step thread.
                self._scheduler.finish_early(request_id)
                self._remove_track(request_id)
                if self._pending_by_request.get(request_id, 0):
                    self._deferred_forget.add(request_id)
                else:
                    self._forget(request_id)
            updates.append((request_id, update))
        return updates

    def _materialize_update(self, item: _TrackUpdateInput) -> _TrackUpdateResult:
        track = item.track
        track.meta.extend(item.new_meta)
        visible_new_ids = item.new_ids
        if (
            item.new_ids
            and item.finished
            and item.state_finish_reason == "stop"
        ):
            terminal_id = item.new_ids[-1]
            request = item.request
            terminal_is_eos = (
                not request.ignore_eos
                and request.eos_token_id is not None
                and terminal_id == request.eos_token_id
            )
            if terminal_is_eos or terminal_id in request.stop_token_ids:
                # Retain the sampled stop token in token IDs, logprobs, usage,
                # KV/radix history, and scheduler accounting.  It alone is not
                # visible text, and a native incremental decoder cannot retract
                # it after it has been pushed.
                visible_new_ids = item.new_ids[:-1]
        if visible_new_ids:
            if track.trace_requested:
                detokenize_started_ns = perf_counter_ns()
                track.stable = track.detok.push(visible_new_ids)
                track.record_stage(
                    "detokenize",
                    perf_counter_ns() - detokenize_started_ns,
                )
            else:
                track.stable = track.detok.push(visible_new_ids)
        track.num_cached_tokens = max(
            track.num_cached_tokens,
            item.num_cached_tokens,
        )
        logprobs, cumulative = _logprob_fields(track.meta)
        content = None
        if any(token.logprob is not None for token in track.meta):
            if self._logprob_vocab is None:
                self._logprob_vocab = tuple(self._tokenizer.vocab())
            content = _logprob_content(
                track.detok.decode,
                self._logprob_vocab,
                track.meta,
            )

        def _update(text: str, finished: bool, reason: str | None) -> StreamUpdate:
            return StreamUpdate(
                item.outputs,
                text,
                finished,
                reason,
                logprobs=logprobs,
                cumulative_logprob=cumulative,
                num_prompt_tokens=track.num_prompt_tokens,
                num_cached_tokens=track.num_cached_tokens,
                logprob_content=content,
                stage_metrics=track.metrics(),
            )

        if item.finished:
            if track.trace_requested:
                detokenize_started_ns = perf_counter_ns()
                terminal = track.detok.finalize()
                track.record_stage(
                    "detokenize",
                    perf_counter_ns() - detokenize_started_ns,
                )
            else:
                terminal = track.detok.finalize()
            # ``stable`` is the never-retract decoding authority; all but its
            # configured stop holdback may already have crossed the streaming
            # boundary. A defensive tokenizer implementation may nevertheless
            # return a shorter or rewritten full decode here.
            # Only accept a terminal flush that extends the stable authority;
            # otherwise keep it intact so cumulative SSE consumers and the
            # incremental stop matcher's monotonic-input invariant both hold.
            if terminal.startswith(track.stable):
                track.stable = terminal
            stop_at = track.find_stop(track.stable)
            if stop_at is not None:
                return _TrackUpdateResult(
                    item.request_id,
                    _update(track.stable[:stop_at], True, "stop"),
                )
            return _TrackUpdateResult(
                item.request_id,
                _update(
                    track.stable,
                    True,
                    item.finish_reason or "length",
                ),
            )
        stop_at = track.find_stop(track.stable)
        if stop_at is not None:
            return _TrackUpdateResult(
                item.request_id,
                _update(track.stable[:stop_at], True, "stop"),
                finish_early=True,
            )
        visible_end = max(0, len(track.stable) - track.holdback)
        return _TrackUpdateResult(
            item.request_id,
            _update(track.stable[:visible_end], False, None),
        )
