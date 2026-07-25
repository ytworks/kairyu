"""Drive a ``PDCoordinator`` from ``EngineLoop`` (m18 D3, G2 stage 5.3).

``EngineLoop`` owns ONE scheduler and ONE runner; a ``PDCoordinator`` owns two
of each and a KV handoff between them. Handing the loop the coordinator's decode
scheduler does not connect them — ``EngineLoop._drain_ops`` adds submissions
straight to the scheduler it was given, so requests would enter at DECODE with
no prompt KV, and ``PDCoordinator`` has no ``execute`` for the loop to call.

``PDLoopAdapter`` is that missing seam, presenting both surfaces the loop needs:

- **scheduler** — ``add_request`` enters the PREFILL scheduler via the
  coordinator (so the handoff runs at all); ``schedule()`` drives one prefill
  step, which prefills, transfers the KV and commits, and only then plans
  decode; every per-request query routes to whichever half holds the request
  right now.
- **runner** — ``execute`` runs the DECODE runner on the plan it just returned.

The request-facing state view is keyed by PUBLIC request ids and prefers decode,
falling back to the prefill clone. That fallback is what lets a request that
never reaches decode — an unadmittable prompt, a transfer that exhausted its
retries — still surface as finished to the loop instead of hanging forever.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from kairyu.engine.core.pd import PDCoordinator
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler, SchedulerOutput


class _PDStates(Mapping):
    """Merged read view of both halves' request states, keyed by public id."""

    def __init__(self, adapter: PDLoopAdapter) -> None:
        self._adapter = adapter

    def __getitem__(self, request_id: str) -> object:
        scheduler, resolved = self._adapter.route(request_id)
        if scheduler is None:
            raise KeyError(request_id)
        return scheduler.states[resolved]

    def __iter__(self) -> Iterator[str]:
        coordinator = self._adapter.coordinator
        decode_states = coordinator.decode_scheduler.states
        yield from decode_states
        prefill_states = coordinator.prefill_scheduler.states
        for public_id, internal_id in coordinator.internal_ids.items():
            if public_id not in decode_states and internal_id in prefill_states:
                yield public_id

    def __len__(self) -> int:
        return sum(1 for _ in self)


class PDLoopAdapter:
    """One object standing in for the scheduler AND the runner of an EngineLoop.

    Both roles are the same object because both are the same coordinator: the
    loop's "schedule then execute" pair is prefill-step-then-decode-plan and the
    decode runner, and splitting them would only hide that they must interleave
    in that order.
    """

    def __init__(self, coordinator: PDCoordinator) -> None:
        self._coordinator = coordinator
        self._states = _PDStates(self)

    @property
    def coordinator(self) -> PDCoordinator:
        return self._coordinator

    @property
    def kv_cache(self) -> RadixKVCache:
        """The cache a served request's KV ends up in (decode's)."""
        return self._coordinator.decode_cache

    def route(self, request_id: str) -> tuple[Scheduler | None, str]:
        """Which half holds ``request_id``, and under which id there."""
        decode = self._coordinator.decode_scheduler
        if request_id in decode.states:
            return decode, request_id
        prefill = self._coordinator.prefill_scheduler
        internal_id = self._coordinator.internal_id_for(request_id)
        if internal_id is not None and internal_id in prefill.states:
            return prefill, internal_id
        return None, request_id

    # --- scheduler surface ----------------------------------------------------

    def add_request(self, request: EngineRequest) -> None:
        # NOT decode.add_request: a request with no prompt KV admitted straight
        # into decode is exactly the bug this adapter exists to prevent
        self._coordinator.add_request(request)

    def abort(self, request_id: str) -> None:
        self._coordinator.abort(request_id)

    def forget(self, request_id: str) -> None:
        self._coordinator.forget(request_id)

    def has_unfinished(self) -> bool:
        return (
            self._coordinator.prefill_scheduler.has_unfinished()
            or self._coordinator.decode_scheduler.has_unfinished()
        )

    @property
    def states(self) -> Mapping[str, object]:
        return self._states

    def schedule(self) -> SchedulerOutput:
        """Advance prefill (including the KV handoff), then plan decode.

        Ordering is the m5 D5 protocol: a prompt that finishes prefilling in
        this step is transferred and resumed under decode before decode plans,
        so it starts generating in the same engine step.
        """
        if self._coordinator.prefill_scheduler.has_unfinished():
            self._coordinator.step_prefill(reject_on_stall=True)
        return self._coordinator.decode_scheduler.schedule()

    def drain_rejected(self) -> tuple[str, ...]:
        prefill_rejected = self._coordinator.prefill_scheduler.drain_rejected()
        decode_rejected = self._coordinator.decode_scheduler.drain_rejected()
        # prefill ids are clone ids; the loop only needs them cleared, and the
        # public request surfaces as finished through the merged state view
        return prefill_rejected + decode_rejected

    def reject_waiting_head(self) -> str | None:
        if self._coordinator.prefill_scheduler.has_unfinished():
            # an empty decode plan is normal while a prompt is still prefilling
            # (chunked prefill spans steps) — nothing is stuck
            return None
        return self._coordinator.decode_scheduler.reject_waiting_head()

    def update(self, sampled: Mapping[str, int | Sequence[int]]) -> tuple[str, ...]:
        # only decode chunks are ever handed back to the loop; prefill commits
        # inside step_prefill, after the handoff
        return self._coordinator.decode_scheduler.update(sampled)

    def finish_early(self, request_id: str) -> None:
        scheduler, resolved = self.route(request_id)
        if scheduler is not None:
            scheduler.finish_early(resolved)

    def finish_reason(self, request_id: str) -> str | None:
        scheduler, resolved = self.route(request_id)
        return None if scheduler is None else scheduler.finish_reason(resolved)

    def output_tokens(self, request_id: str) -> tuple[int, ...]:
        scheduler, resolved = self.route(request_id)
        return () if scheduler is None else scheduler.output_tokens(resolved)

    def num_cached_tokens(self, request_id: str) -> int:
        scheduler, resolved = self.route(request_id)
        return 0 if scheduler is None else scheduler.num_cached_tokens(resolved)

    # --- runner surface -------------------------------------------------------

    def execute(
        self, scheduled: tuple, states: Mapping[str, object]
    ) -> dict[str, tuple[SampledToken, ...]]:
        return self._coordinator.decode_runner.execute(scheduled, states)

    def release(self, request_id: str) -> None:
        release = getattr(self._coordinator.decode_runner, "release", None)
        if release is not None:
            release(request_id)
