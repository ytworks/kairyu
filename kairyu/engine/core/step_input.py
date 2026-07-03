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
