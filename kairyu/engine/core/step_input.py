"""Typed immutable per-step snapshot for ModelRunners (design m5 D2; m2 §5 item 3).

``snapshot_step`` copies everything a rank needs for one engine step out of
live ``Scheduler`` state into frozen dataclasses. The whole point is a
torn-free snapshot: after it is built, no scheduler mutation (overlap thread,
``update()`` commits, preemption) can change what a rank sees, so it is safe
to broadcast across ranks/processes.

``RequestSnapshot`` also exposes the ``_RequestState`` surface that existing
CPU runners use (``state.request.prompt_token_ids``, ``state.prefill_done``),
so _ToyRunner-style runners execute unchanged on snapshots.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kairyu.engine.core.frozen_prefix import FrozenPrefix
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import ScheduledChunk, SchedulerOutput


@dataclass(frozen=True)
class RequestSnapshot:
    """Frozen per-request view of ``_RequestState`` at snapshot time.

    m16 A1 extension (mandated by the m12 review): carries ``outputs`` values,
    ``sampling`` and ``num_cached_tokens`` plus allocation-shaped aliases, so
    ``PagedModelRunner``'s canonical state-access contract works on broadcast
    snapshots — this is also how TP workers receive rank-0's committed tokens
    (the NEXT step's snapshot outputs).
    """

    request_id: str
    prompt_token_ids: tuple[int, ...]
    computed_prompt: int
    outputs: Sequence[int]
    in_flight: int
    page_ids: tuple[int, ...]
    decode_page_ids: tuple[int, ...]
    eos_token_id: int | None
    max_new_tokens: int
    num_cached_tokens: int = 0
    sampling: EngineSampling = field(default_factory=EngineSampling)
    output_epoch: int = 0
    # Speculative target scoring replaces the scheduler completion with a draft
    # prefix.  Runners must prefer that explicit host prefix over any retained
    # future device token at the same position.
    outputs_override: bool = False
    # P-D runs prefill under an internal clone id while decode uses the public
    # one, so the id a sampler keys state under is NOT always `request_id`
    # (m5 D5). The snapshot has to carry it: the overlap path and the TP workers
    # sample from a snapshot, never from the EngineRequest, and keying those on
    # `request_id` would put the clone's tokens on a different RNG stream.
    sampling_id: str | None = None
    # Termination metadata is immutable per request but must cross every
    # overlap/TP/EP snapshot so selection cannot bypass the min-token mask.
    stop_token_ids: tuple[int, ...] = ()
    min_tokens: int = 0
    ignore_eos: bool = False

    @property
    def sampling_identity(self) -> str:
        """The id every sampler must key this request's state under."""
        return self.sampling_id or self.request_id

    @property
    def output_len(self) -> int:
        return len(self.outputs)

    @property
    def request(self) -> RequestSnapshot:
        """Runner-compatible alias: ``state.request.prompt_token_ids`` works here."""
        return self

    @property
    def allocation(self) -> RequestSnapshot:
        """Runner-compatible alias: ``state.allocation.pages`` etc. (truthy)."""
        return self

    @property
    def pages(self) -> tuple[int, ...]:
        return self.page_ids

    @property
    def decode_pages(self) -> tuple[int, ...]:
        return self.decode_page_ids

    @property
    def prefill_done(self) -> bool:
        return self.computed_prompt >= len(self.prompt_token_ids)


@dataclass(frozen=True)
class StepInput:
    """Everything one engine step needs, with no live scheduler state attached."""

    chunks: tuple[ScheduledChunk, ...]
    requests: tuple[RequestSnapshot, ...]

    def states_view(self) -> dict[str, RequestSnapshot]:
        """Mapping shaped like the ModelRunner ``states`` argument."""
        return {snapshot.request_id: snapshot for snapshot in self.requests}


def _snapshot_state(state: object) -> RequestSnapshot:
    """Copy one live ``_RequestState`` (duck-typed) into a frozen snapshot."""
    request = state.request  # type: ignore[attr-defined]
    allocation = state.allocation  # type: ignore[attr-defined]
    return RequestSnapshot(
        request_id=request.request_id,
        prompt_token_ids=tuple(request.prompt_token_ids),
        computed_prompt=state.computed_prompt,  # type: ignore[attr-defined]
        outputs=FrozenPrefix(state.outputs),  # type: ignore[attr-defined]
        in_flight=state.in_flight,  # type: ignore[attr-defined]
        page_ids=tuple(allocation.pages) if allocation is not None else (),
        decode_page_ids=(
            state.decode_pages_snapshot  # type: ignore[attr-defined]
            if hasattr(state, "decode_pages_snapshot")
            else tuple(state.decode_pages)  # type: ignore[attr-defined]
        ),
        eos_token_id=request.eos_token_id,
        max_new_tokens=request.max_new_tokens,
        stop_token_ids=tuple(getattr(request, "stop_token_ids", ())),
        min_tokens=getattr(request, "min_tokens", 0),
        ignore_eos=getattr(request, "ignore_eos", False),
        num_cached_tokens=allocation.num_cached_tokens if allocation is not None else 0,
        sampling=getattr(request, "sampling", EngineSampling()),
        sampling_id=getattr(request, "sampling_id", None),
        output_epoch=getattr(state, "output_epoch", 0),
        outputs_override=getattr(state, "outputs_override", False),
    )


@dataclass(frozen=True)
class RequestDelta:
    """Only the MUTABLE fields of a request that already crossed the wire (F4).

    prompt_token_ids / page_ids / sampling / eos / max_new_tokens are immutable
    after admission, so a re-seen request sends just its changed fields plus the
    NEW output tokens (the tail beyond what the peer already holds)."""

    request_id: str
    computed_prompt: int
    new_outputs: tuple[int, ...]
    in_flight: int
    new_decode_page_ids: tuple[int, ...]
    num_cached_tokens: int


@dataclass(frozen=True)
class StepDelta:
    """Delta-broadcast step (F4): full snapshots only for first-seen (or
    re-allocated) requests; small field deltas for the rest; dropped ids leave
    the peer's step state. Finished-request cleanup is batched into the next
    step so every TP rank releases it at the same protocol boundary without a
    separate control collective. Reconstructs snapshot_step()'s StepInput
    exactly."""

    chunks: tuple[ScheduledChunk, ...]
    new: tuple[RequestSnapshot, ...]
    updates: tuple[RequestDelta, ...]
    dropped: tuple[str, ...]
    released_ids: tuple[str, ...] = ()


_STEP_DELTA_TENSOR_MAGIC = 0x4B535444  # "KSTD"
_STEP_DELTA_TENSOR_VERSION = 1
_STEP_DELTA_STRING_BYTES_PER_WORD = 7
_STEP_DELTA_INT64_MIN = -(1 << 63)
_STEP_DELTA_INT64_MAX = (1 << 63) - 1
_STEP_DELTA_UINT64_MODULUS = 1 << 64


class _IntWriter:
    def __init__(self) -> None:
        self.values: list[int] = []

    def integer(self, value: int) -> None:
        if type(value) is not int:
            raise TypeError(f"step tensor integer must be int, got {type(value).__name__}")
        if not _STEP_DELTA_INT64_MIN <= value <= _STEP_DELTA_INT64_MAX:
            raise ValueError(f"step tensor integer is outside int64 range: {value}")
        self.values.append(value)

    def boolean(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("step tensor boolean must be bool")
        self.values.append(int(value))

    def optional_integer(self, value: int | None) -> None:
        self.boolean(value is not None)
        if value is not None:
            self.integer(value)

    def optional_seed(self, value: int | None) -> None:
        self.boolean(value is not None)
        if value is None:
            return
        if type(value) is not int or not (
            _STEP_DELTA_INT64_MIN <= value < _STEP_DELTA_UINT64_MODULUS
        ):
            raise ValueError(f"step tensor seed is outside 64-bit range: {value}")
        unsigned_high = value > _STEP_DELTA_INT64_MAX
        self.boolean(unsigned_high)
        self.integer(value - _STEP_DELTA_UINT64_MODULUS if unsigned_high else value)

    def floating(self, value: float) -> None:
        self.integer(struct.unpack("!q", struct.pack("!d", float(value)))[0])

    def string(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("step tensor string must be str")
        encoded = value.encode("utf-8")
        self.integer(len(encoded))
        width = _STEP_DELTA_STRING_BYTES_PER_WORD
        self.values.extend(
            int.from_bytes(encoded[start : start + width], "little")
            for start in range(0, len(encoded), width)
        )

    def optional_string(self, value: str | None) -> None:
        self.boolean(value is not None)
        if value is not None:
            self.string(value)

    def integers(self, values: Sequence[int]) -> None:
        self.integer(len(values))
        for value in values:
            self.integer(value)

    def strings(self, values: Sequence[str]) -> None:
        self.integer(len(values))
        for value in values:
            self.string(value)

    def optional_json_object(self, value: dict | None) -> None:
        self.boolean(value is not None)
        if value is not None:
            self.string(
                json.dumps(
                    value,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )


class _IntReader:
    def __init__(self, values: Sequence[int]) -> None:
        self._values = values
        self._position = 0

    def integer(self) -> int:
        if self._position >= len(self._values):
            raise ValueError("truncated step tensor payload")
        value = self._values[self._position]
        self._position += 1
        if type(value) is not int:
            raise TypeError("step tensor payload values must be integers")
        return value

    def count(self, label: str) -> int:
        value = self.integer()
        if value < 0 or value > len(self._values) - self._position:
            raise ValueError(f"invalid step tensor {label} length {value}")
        return value

    def boolean(self) -> bool:
        value = self.integer()
        if value not in (0, 1):
            raise ValueError(f"invalid step tensor boolean {value}")
        return bool(value)

    def optional_integer(self) -> int | None:
        return self.integer() if self.boolean() else None

    def optional_seed(self) -> int | None:
        if not self.boolean():
            return None
        unsigned_high = self.boolean()
        value = self.integer()
        if unsigned_high:
            if value >= 0:
                raise ValueError("invalid unsigned-high step tensor seed")
            return value + _STEP_DELTA_UINT64_MODULUS
        return value

    def floating(self) -> float:
        return struct.unpack("!d", struct.pack("!q", self.integer()))[0]

    def string(self) -> str:
        size = self.integer()
        if size < 0:
            raise ValueError(f"invalid step tensor string length {size}")
        width = _STEP_DELTA_STRING_BYTES_PER_WORD
        words = (size + width - 1) // width
        if words > len(self._values) - self._position:
            raise ValueError(f"invalid step tensor string length {size}")
        chunks = []
        for _ in range(words):
            value = self.integer()
            if not 0 <= value < (1 << (8 * width)):
                raise ValueError(f"invalid packed step tensor string word {value}")
            chunks.append(value.to_bytes(width, "little"))
        encoded = b"".join(chunks)[:size]
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("step tensor string is not valid UTF-8") from error

    def optional_string(self) -> str | None:
        return self.string() if self.boolean() else None

    def integers(self) -> tuple[int, ...]:
        return tuple(self.integer() for _ in range(self.count("integer tuple")))

    def strings(self) -> tuple[str, ...]:
        return tuple(self.string() for _ in range(self.count("string tuple")))

    def optional_json_object(self) -> dict | None:
        if not self.boolean():
            return None
        try:
            value = json.loads(self.string())
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("step tensor JSON object is malformed") from error
        if not isinstance(value, dict):
            raise ValueError("step tensor JSON value must be an object")
        return value

    def finish(self) -> None:
        if self._position != len(self._values):
            raise ValueError(
                f"step tensor payload has {len(self._values) - self._position} "
                "trailing values"
            )


def _write_sampling(writer: _IntWriter, sampling: EngineSampling) -> None:
    writer.floating(sampling.temperature)
    writer.integer(sampling.top_k)
    writer.floating(sampling.top_p)
    writer.floating(sampling.min_p)
    writer.floating(sampling.presence_penalty)
    writer.floating(sampling.frequency_penalty)
    writer.floating(sampling.repetition_penalty)
    writer.optional_seed(sampling.seed)
    writer.optional_integer(sampling.logprobs)
    writer.optional_json_object(sampling.json_schema)
    writer.boolean(sampling.json_mode)
    writer.optional_string(sampling.regex)
    writer.optional_string(sampling.grammar)
    writer.optional_json_object(sampling.structural_tag)
    writer.boolean(sampling.forced_token_ids is not None)
    if sampling.forced_token_ids is not None:
        writer.integers(sampling.forced_token_ids)


def _read_sampling(reader: _IntReader) -> EngineSampling:
    return EngineSampling(
        temperature=reader.floating(),
        top_k=reader.integer(),
        top_p=reader.floating(),
        min_p=reader.floating(),
        presence_penalty=reader.floating(),
        frequency_penalty=reader.floating(),
        repetition_penalty=reader.floating(),
        seed=reader.optional_seed(),
        logprobs=reader.optional_integer(),
        json_schema=reader.optional_json_object(),
        json_mode=reader.boolean(),
        regex=reader.optional_string(),
        grammar=reader.optional_string(),
        structural_tag=reader.optional_json_object(),
        forced_token_ids=reader.integers() if reader.boolean() else None,
    )


def _write_snapshot(writer: _IntWriter, snapshot: RequestSnapshot) -> None:
    writer.string(snapshot.request_id)
    writer.integers(snapshot.prompt_token_ids)
    writer.integer(snapshot.computed_prompt)
    writer.integers(snapshot.outputs)
    writer.integer(snapshot.in_flight)
    writer.integers(snapshot.page_ids)
    writer.integers(snapshot.decode_page_ids)
    writer.optional_integer(snapshot.eos_token_id)
    writer.integer(snapshot.max_new_tokens)
    writer.integer(snapshot.num_cached_tokens)
    _write_sampling(writer, snapshot.sampling)
    writer.integer(snapshot.output_epoch)
    writer.boolean(snapshot.outputs_override)
    writer.optional_string(snapshot.sampling_id)
    writer.integers(snapshot.stop_token_ids)
    writer.integer(snapshot.min_tokens)
    writer.boolean(snapshot.ignore_eos)


def _read_snapshot(reader: _IntReader) -> RequestSnapshot:
    return RequestSnapshot(
        request_id=reader.string(),
        prompt_token_ids=reader.integers(),
        computed_prompt=reader.integer(),
        outputs=reader.integers(),
        in_flight=reader.integer(),
        page_ids=reader.integers(),
        decode_page_ids=reader.integers(),
        eos_token_id=reader.optional_integer(),
        max_new_tokens=reader.integer(),
        num_cached_tokens=reader.integer(),
        sampling=_read_sampling(reader),
        output_epoch=reader.integer(),
        outputs_override=reader.boolean(),
        sampling_id=reader.optional_string(),
        stop_token_ids=reader.integers(),
        min_tokens=reader.integer(),
        ignore_eos=reader.boolean(),
    )


def encode_step_delta(delta: StepDelta) -> tuple[int, ...]:
    """Encode the hot TP step control payload without pickle or object collectives."""

    if not isinstance(delta, StepDelta):
        raise TypeError("step tensor payload must be StepDelta")
    writer = _IntWriter()
    writer.integer(_STEP_DELTA_TENSOR_MAGIC)
    writer.integer(_STEP_DELTA_TENSOR_VERSION)
    writer.integer(len(delta.chunks))
    for chunk in delta.chunks:
        writer.string(chunk.request_id)
        writer.integer(chunk.num_tokens)
        writer.boolean(chunk.is_prefill)
        writer.integer(chunk.position)
    writer.integer(len(delta.new))
    for snapshot in delta.new:
        _write_snapshot(writer, snapshot)
    writer.integer(len(delta.updates))
    for update in delta.updates:
        writer.string(update.request_id)
        writer.integer(update.computed_prompt)
        writer.integers(update.new_outputs)
        writer.integer(update.in_flight)
        writer.integers(update.new_decode_page_ids)
        writer.integer(update.num_cached_tokens)
    writer.strings(delta.dropped)
    writer.strings(delta.released_ids)
    return tuple(writer.values)


def decode_step_delta(values: Sequence[int]) -> StepDelta:
    """Decode and validate one tensor-transported TP step delta."""

    reader = _IntReader(values)
    if reader.integer() != _STEP_DELTA_TENSOR_MAGIC:
        raise ValueError("step tensor payload has invalid magic")
    version = reader.integer()
    if version != _STEP_DELTA_TENSOR_VERSION:
        raise ValueError(f"unsupported step tensor version {version}")
    chunks = tuple(
        ScheduledChunk(
            reader.string(),
            reader.integer(),
            reader.boolean(),
            reader.integer(),
        )
        for _ in range(reader.count("chunks"))
    )
    new = tuple(_read_snapshot(reader) for _ in range(reader.count("new snapshots")))
    updates = tuple(
        RequestDelta(
            request_id=reader.string(),
            computed_prompt=reader.integer(),
            new_outputs=reader.integers(),
            in_flight=reader.integer(),
            new_decode_page_ids=reader.integers(),
            num_cached_tokens=reader.integer(),
        )
        for _ in range(reader.count("updates"))
    )
    delta = StepDelta(
        chunks=chunks,
        new=new,
        updates=updates,
        dropped=reader.strings(),
        released_ids=reader.strings(),
    )
    reader.finish()
    return delta


class StateSync:
    """Incremental peer state: rank 0 diffs live scheduler state into a StepDelta;
    every rank applies it to reconstruct the SAME full RequestSnapshots without
    re-sending immutable/accumulated fields each step."""

    def __init__(self) -> None:
        self._states: dict[str, RequestSnapshot] = {}
        self._outputs: dict[str, list[int]] = {}

    def discard(self, request_ids: tuple[str, ...]) -> None:
        """Forget completed request identities before a possible immediate reuse."""
        for request_id in request_ids:
            self._states.pop(request_id, None)
            self._outputs.pop(request_id, None)

    def diff(
        self,
        chunks: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
        *,
        released_ids: tuple[str, ...] = (),
    ) -> StepDelta:
        # A request id may be reused immediately after release. Compare it as
        # first-seen so an identical prompt/allocation still crosses as a full
        # snapshot, but do not mutate driver state until the control broadcast
        # succeeds and apply() commits the delta on every rank.
        released = set(released_ids)
        active: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.request_id not in seen:
                seen.add(chunk.request_id)
                active.append(chunk.request_id)
        new: list[RequestSnapshot] = []
        updates: list[RequestDelta] = []
        for rid in active:
            snap = _snapshot_state(states[rid])
            prev = None if rid in released else self._states.get(rid)
            # Production scheduler outputs and private decode pages are append-
            # only for one allocation epoch.  Reallocation, speculative overlay,
            # or an owner-declared output rewrite takes the full-snapshot path.
            if (
                prev is None
                or prev.page_ids != snap.page_ids
                or prev.prompt_token_ids != snap.prompt_token_ids
                or prev.output_epoch != snap.output_epoch
                or prev.outputs_override != snap.outputs_override
                or len(snap.outputs) < len(prev.outputs)
                or len(snap.decode_page_ids) < len(prev.decode_page_ids)
            ):
                new.append(
                    dataclasses.replace(
                        snap,
                        outputs=tuple(snap.outputs),
                        decode_page_ids=tuple(snap.decode_page_ids),
                    )
                )
            else:
                updates.append(
                    RequestDelta(
                        request_id=rid,
                        computed_prompt=snap.computed_prompt,
                        new_outputs=snap.outputs[len(prev.outputs):],
                        in_flight=snap.in_flight,
                        new_decode_page_ids=tuple(
                            snap.decode_page_ids[len(prev.decode_page_ids) :]
                        ),
                        num_cached_tokens=snap.num_cached_tokens,
                    )
                )
        dropped = tuple(rid for rid in self._states if rid not in seen)
        return StepDelta(
            chunks=chunks,
            new=tuple(new),
            updates=tuple(updates),
            dropped=dropped,
            released_ids=released_ids,
        )

    def apply(self, delta: StepDelta) -> dict[str, RequestSnapshot]:
        self.discard(delta.released_ids)
        for rid in delta.dropped:
            self._states.pop(rid, None)
            self._outputs.pop(rid, None)
        for snap in delta.new:
            outputs = list(snap.outputs)
            self._outputs[snap.request_id] = outputs
            self._states[snap.request_id] = dataclasses.replace(
                snap,
                outputs=FrozenPrefix(outputs),
            )
        for update in delta.updates:
            prev = self._states[update.request_id]
            outputs = self._outputs[update.request_id]
            outputs.extend(update.new_outputs)
            self._states[update.request_id] = dataclasses.replace(
                prev,
                computed_prompt=update.computed_prompt,
                outputs=FrozenPrefix(outputs),
                in_flight=update.in_flight,
                decode_page_ids=(
                    prev.decode_page_ids + update.new_decode_page_ids
                ),
                num_cached_tokens=update.num_cached_tokens,
            )
        active = {chunk.request_id for chunk in delta.chunks}
        return {rid: self._states[rid] for rid in active}


def snapshot_step(
    scheduled: SchedulerOutput | tuple[ScheduledChunk, ...],
    states: Mapping[str, object],
) -> StepInput:
    """Build a StepInput from ``Scheduler.schedule()`` output and ``Scheduler.states``.

    Accepts either the ``SchedulerOutput`` or its ``scheduled`` chunk tuple
    (the shape ``ModelRunner.execute`` receives). Only requests referenced by
    the scheduled chunks are snapshotted.
    """
    chunks = scheduled.scheduled if isinstance(scheduled, SchedulerOutput) else tuple(scheduled)
    snapshots: list[RequestSnapshot] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.request_id in seen:
            continue
        state = states.get(chunk.request_id)
        if state is None:
            raise ValueError(
                f"scheduled chunk references unknown request {chunk.request_id!r}"
            )
        seen.add(chunk.request_id)
        snapshots.append(_snapshot_state(state))
    return StepInput(chunks=chunks, requests=tuple(snapshots))
