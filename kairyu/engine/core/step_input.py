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

from collections.abc import Mapping
from dataclasses import dataclass, field

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
    outputs: tuple[int, ...]
    in_flight: int
    page_ids: tuple[int, ...]
    decode_page_ids: tuple[int, ...]
    eos_token_id: int | None
    max_new_tokens: int
    num_cached_tokens: int = 0
    sampling: EngineSampling = field(default_factory=EngineSampling)

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
        outputs=tuple(state.outputs),  # type: ignore[attr-defined]
        in_flight=state.in_flight,  # type: ignore[attr-defined]
        page_ids=tuple(allocation.pages) if allocation is not None else (),
        decode_page_ids=tuple(state.decode_pages),  # type: ignore[attr-defined]
        eos_token_id=request.eos_token_id,
        max_new_tokens=request.max_new_tokens,
        num_cached_tokens=allocation.num_cached_tokens if allocation is not None else 0,
        sampling=getattr(request, "sampling", EngineSampling()),
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
    decode_page_ids: tuple[int, ...]
    num_cached_tokens: int


@dataclass(frozen=True)
class StepDelta:
    """Delta-broadcast step (F4): full snapshots only for first-seen (or
    re-allocated) requests; small field deltas for the rest; dropped ids leave
    the peer's state. Reconstructs snapshot_step()'s StepInput exactly."""

    chunks: tuple[ScheduledChunk, ...]
    new: tuple[RequestSnapshot, ...]
    updates: tuple[RequestDelta, ...]
    dropped: tuple[str, ...]


class StateSync:
    """Incremental peer state: rank 0 diffs live scheduler state into a StepDelta;
    every rank applies it to reconstruct the SAME full RequestSnapshots without
    re-sending immutable/accumulated fields each step."""

    def __init__(self) -> None:
        self._states: dict[str, RequestSnapshot] = {}

    def diff(
        self,
        chunks: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> StepDelta:
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
            prev = self._states.get(rid)
            # a preempted+re-admitted request may change pages/prompt: re-send full
            if (
                prev is None
                or prev.page_ids != snap.page_ids
                or prev.prompt_token_ids != snap.prompt_token_ids
                or len(prev.outputs) > len(snap.outputs)
            ):
                new.append(snap)
            else:
                updates.append(
                    RequestDelta(
                        request_id=rid,
                        computed_prompt=snap.computed_prompt,
                        new_outputs=snap.outputs[len(prev.outputs):],
                        in_flight=snap.in_flight,
                        decode_page_ids=snap.decode_page_ids,
                        num_cached_tokens=snap.num_cached_tokens,
                    )
                )
        dropped = tuple(rid for rid in self._states if rid not in seen)
        return StepDelta(
            chunks=chunks, new=tuple(new), updates=tuple(updates), dropped=dropped
        )

    def apply(self, delta: StepDelta) -> dict[str, RequestSnapshot]:
        for rid in delta.dropped:
            self._states.pop(rid, None)
        for snap in delta.new:
            self._states[snap.request_id] = snap
        for update in delta.updates:
            import dataclasses

            prev = self._states[update.request_id]
            self._states[update.request_id] = dataclasses.replace(
                prev,
                computed_prompt=update.computed_prompt,
                outputs=prev.outputs + update.new_outputs,
                in_flight=update.in_flight,
                decode_page_ids=update.decode_page_ids,
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
