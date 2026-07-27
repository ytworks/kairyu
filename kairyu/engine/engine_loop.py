"""Shared production engine loop: ops → schedule-ahead → execute → commit → stream.

One thread-agnostic core used by both process layouts:
``KairyuBackend`` drives it from an asyncio pump (``asyncio.to_thread``), the
ZMQ ``engine_service`` drives it from its single-threaded socket loop. All
scheduler mutations (submit/abort/stop-string ``finish_early``) happen inside
``step()`` — the m8 D1 op discipline holds by construction.

``pipeline_depth=1`` is the historical synchronous behavior. Larger depths
freeze each scheduled step and let the device worker run step N while this
thread schedules N+1. Streaming, stop holdback, grammar termination,
speculation, preemption and P-D carried tokens still commit through this one
loop; there is no alternate production overlap core.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from kairyu.engine.core.engine_core import grammar_finished, token_ids
from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.step_input import snapshot_step
from kairyu.engine.tokenizer import IncrementalDetokenizer, Tokenizer
from kairyu.outputs import TokenLogprob
from kairyu.sampling_params import SamplingParams

_DEFAULT_MAX_NEW_TOKENS = 16
_DEFAULT_PIPELINE_DEPTH = 1


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


def engine_sampling_from(params: SamplingParams) -> EngineSampling:
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


def _token_logprob(tokenizer: Tokenizer, token_id: int, logprob: float) -> TokenLogprob:
    token = tokenizer.decode((token_id,))
    return TokenLogprob(
        token=token,
        token_id=token_id,
        logprob=logprob,
        # bytes_ is the lossless form: byte-level BPE fragments decode to U+FFFD
        bytes_=tuple(token.encode("utf-8")),
    )


def _logprob_content(
    tokenizer: Tokenizer, meta: list[SampledToken]
) -> tuple[TokenLogprob, ...] | None:
    if not any(token.logprob is not None for token in meta):
        return None
    entries = []
    for token in meta:
        if token.logprob is None:
            continue
        top = tuple(
            _token_logprob(tokenizer, top_id, top_lp)
            for top_id, top_lp in token.top_logprobs or ()
        )
        base = _token_logprob(tokenizer, token.token_id, token.logprob)
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
    )

    def __init__(
        self, detok: IncrementalDetokenizer, stops: tuple[str, ...], num_prompt_tokens: int
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

    def find_stop(self, text: str) -> int | None:
        return self.stop_matcher.find(text)


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
    ) -> None:
        if pipeline_depth < 1:
            raise ValueError(f"pipeline_depth must be >= 1, got {pipeline_depth}")
        self._tokenizer = tokenizer
        self._scheduler = scheduler
        self._runner = runner
        self._pipeline_depth = pipeline_depth
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
        self._active_request_ids: set[str] = set()

    def tokenize_prompt(self, prompt: str) -> tuple[int, ...]:
        prompt_token_ids = self._tokenizer.encode(prompt)
        if not prompt_token_ids:
            raise ValueError("prompt must tokenize to at least one token")
        return prompt_token_ids

    def submit(self, request_id: str, prompt: str, params: SamplingParams) -> None:
        # Advisory fast rejection avoids tokenization and lock traffic for the
        # common duplicate case. The lock-protected check below remains the
        # authority when producers race.
        if request_id in self._active_request_ids:
            raise ValueError(f"duplicate request_id {request_id!r}")
        engine_request = EngineRequest(
            request_id=request_id,
            prompt_token_ids=self.tokenize_prompt(prompt),
            max_new_tokens=params.max_tokens or _DEFAULT_MAX_NEW_TOKENS,
            eos_token_id=self._default_eos,
            stop_token_ids=tuple(params.stop_token_ids or ()) + self._default_stop_ids,
            min_tokens=params.min_tokens,
            ignore_eos=params.ignore_eos,
            sampling=engine_sampling_from(params),
        )
        track = _RequestTrack(
            detok=IncrementalDetokenizer(self._tokenizer),
            stops=tuple(params.stop or ()),
            num_prompt_tokens=len(engine_request.prompt_token_ids),
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
            self._tracked.pop(request_id, None)
            self._forget(request_id)

    def has_work(self) -> bool:
        with self._ops_lock:
            has_ops = bool(self._ops)
        return (
            has_ops
            or bool(self._pending_steps)
            or self._scheduler.has_unfinished()
        )

    @property
    def pipeline_depth(self) -> int:
        return self._pipeline_depth

    def _take_ops(self) -> tuple[_OpBatch, ...]:
        """Freeze one step-boundary snapshot; later producers target a new queue."""
        with self._ops_lock:
            batches = tuple(self._ops)
            self._ops.clear()
        return batches

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
                            self._tracked[request.request_id] = track
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
                    self._tracked[engine_request.request_id] = track
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
            return self._step_once()

    def _step_once(self) -> list[tuple[str, StreamUpdate]]:
        """Advance the unified pipeline and return cumulative stream updates.

        At most one submitted device step is committed per call. Before that
        commit, the scheduler fills the configured pipeline depth with frozen
        snapshots. This preserves the historical streaming cadence while
        allowing scheduling and the oldest device execution to overlap.
        """
        self._drain_ops()
        while (
            self._scheduler.has_unfinished()
            and len(self._pending_steps) < self._pipeline_depth
        ):
            plan = self._scheduler.schedule()
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

        updates: list[tuple[str, StreamUpdate]] = []
        for request_id, track in list(self._tracked.items()):
            update = self._track_update(request_id, track)
            if update is None:
                continue
            updates.append((request_id, update))
            if update.finished:
                del self._tracked[request_id]
                if self._pending_by_request.get(request_id, 0):
                    # A later scheduled-ahead step can still return this id.
                    # Keep scheduler + runner state until its surplus token is
                    # trimmed, but emit the terminal stream update immediately.
                    self._deferred_forget.add(request_id)
                else:
                    self._forget(request_id)
        return updates

    def _submit_step(self, scheduled: tuple) -> None:
        """Freeze and submit one scheduler plan to the common handle contract."""
        step = snapshot_step(scheduled, self._scheduler.states)
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
        self._pending_steps.append(_PendingStep(handle, request_ids))
        for request_id in request_ids:
            self._pending_by_request[request_id] = (
                self._pending_by_request.get(request_id, 0) + 1
            )
        self._step_index += 1

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
            sampled = pending.handle.result()
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
        """Settle outstanding device work and release the private device lane."""
        # asyncio cancellation cannot stop an already-running to_thread(step).
        # Waiting on the same lock prevents shutdown from releasing runner state
        # underneath that call.
        with self._step_lock:
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
                self._tracked.pop(request_id, None)
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

    def _track_update(self, request_id: str, track: _RequestTrack) -> StreamUpdate | None:
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
        track.num_cached_tokens = max(
            track.num_cached_tokens, self._scheduler.num_cached_tokens(request_id)
        )
        logprobs, cumulative = _logprob_fields(track.meta)
        content = _logprob_content(self._tokenizer, track.meta)

        def _update(text: str, finished: bool, reason: str | None) -> StreamUpdate:
            return StreamUpdate(
                outputs,
                text,
                finished,
                reason,
                logprobs=logprobs,
                cumulative_logprob=cumulative,
                num_prompt_tokens=track.num_prompt_tokens,
                num_cached_tokens=track.num_cached_tokens,
                logprob_content=content,
            )

        if state.status.value == "finished":
            full = track.detok.finalize()
            stop_at = track.find_stop(full)
            if stop_at is not None:
                return _update(full[:stop_at], True, "stop")
            return _update(full, True, self._scheduler.finish_reason(request_id) or "length")
        stop_at = track.find_stop(track.stable)
        if stop_at is not None:
            # between update() and the next schedule(): the safe finish point
            self._scheduler.finish_early(request_id)
            return _update(track.stable[:stop_at], True, "stop")
        visible_end = max(0, len(track.stable) - track.holdback)
        return _update(track.stable[:visible_end], False, None)
