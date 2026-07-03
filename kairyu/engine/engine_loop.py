"""Shared engine step loop: ops → schedule → execute → update → updates (m8 D6).

One synchronous, thread-agnostic core used by both process layouts:
``KairyuBackend`` drives it from an asyncio pump (``asyncio.to_thread``), the
ZMQ ``engine_service`` drives it from its single-threaded socket loop. All
scheduler mutations (submit/abort/stop-string ``finish_early``) happen inside
``step()`` — the m8 D1 op discipline holds by construction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from kairyu.engine.core.engine_core import grammar_finished, token_ids
from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.tokenizer import IncrementalDetokenizer, Tokenizer
from kairyu.outputs import TokenLogprob
from kairyu.sampling_params import SamplingParams

_DEFAULT_MAX_NEW_TOKENS = 16


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


class _RequestTrack:
    """Step-side streaming state for one request."""

    __slots__ = (
        "detok",
        "stops",
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
        self.stops = stops
        self.holdback = max((len(stop) for stop in stops), default=1) - 1
        self.consumed = 0
        self.stable = ""
        self.meta: list[SampledToken] = []  # committed tokens' logprob metadata
        self.pending: list[SampledToken] = []
        self.num_prompt_tokens = num_prompt_tokens
        self.num_cached_tokens = 0

    def find_stop(self, text: str) -> int | None:
        indices = [index for stop in self.stops if (index := text.find(stop)) != -1]
        return min(indices) if indices else None


class EngineLoop:
    """Owns tokenizer + scheduler + runner; drains ops and produces updates."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        scheduler: Scheduler,
        runner: object,
        default_eos_token_id: int | None = None,
        default_stop_token_ids: tuple[int, ...] = (),
    ) -> None:
        self._tokenizer = tokenizer
        self._scheduler = scheduler
        self._runner = runner
        # generation_config.json may carry an eos LIST (m12 D5): first entry
        # is eos, the rest are stop tokens; falls back to the tokenizer's eos
        self._default_eos = (
            default_eos_token_id
            if default_eos_token_id is not None
            else tokenizer.eos_token_id
        )
        self._default_stop_ids = default_stop_token_ids
        # deque appends are atomic: producers may enqueue from another thread
        self._ops: deque[tuple[str, object]] = deque()
        self._tracked: dict[str, _RequestTrack] = {}  # step-side only

    def submit(self, request_id: str, prompt: str, params: SamplingParams) -> None:
        engine_request = EngineRequest(
            request_id=request_id,
            prompt_token_ids=self._tokenizer.encode(prompt),
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
        self._ops.append(("add", (engine_request, track)))

    def abort(self, request_id: str) -> None:
        self._ops.append(("abort", request_id))

    def has_work(self) -> bool:
        return bool(self._ops) or self._scheduler.has_unfinished()

    def _drain_ops(self) -> None:
        while self._ops:
            op, payload = self._ops.popleft()
            if op == "add":
                engine_request, track = payload
                self._scheduler.add_request(engine_request)
                self._tracked[engine_request.request_id] = track
            elif op == "abort":
                request_id = payload
                if request_id in self._tracked:
                    self._scheduler.abort(request_id)

    def step(self) -> list[tuple[str, StreamUpdate]]:
        """One engine step; returns cumulative per-request updates."""
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
        updates: list[tuple[str, StreamUpdate]] = []
        for request_id, track in list(self._tracked.items()):
            update = self._track_update(request_id, track)
            if update is None:
                continue
            updates.append((request_id, update))
            if update.finished:
                del self._tracked[request_id]
        return updates

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
