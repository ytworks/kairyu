"""Side-stream KV handoff (m18 D3): extraction on a stream of its own.

``StreamProvider`` is the CUDA seam, with ``CpuNoopStream`` and
``CudaStreamProvider`` as its two implementations. ``StreamCopyKVHandoff`` pins
the ordering: enter stream → inner.transfer (which extracts+copies) →
synchronize → return. A recording fake tests the order.

With ``defer=True`` the handoff returns while the copy is still running and
records its completion event instead. That is only safe for a consumer that
takes on the whole m6 D4 ordering rule, which has two halves and not one:

1. nothing may READ the destination pages before the event, and
2. nothing may REUSE the source pages before the event either — the copy is
   still reading them, and releasing them lets the next step allocate the same
   page and overwrite it on the caller's stream.

``PDCoordinator`` is the consumer that implements both, by gating a whole
prefill step's settlement — releases and decode-side adoption alike — on
``gate_pending()``. WHERE it applies that gate is what turns a non-blocking host
into actual overlap: one producer step later, so the next prefill forward is
already queued in front of it. ``pd_factory`` is what wires it.
"""

from __future__ import annotations

from typing import Protocol

from kairyu.engine.core.radix_kv import KVAllocation


class StreamProvider(Protocol):
    def begin(self) -> None: ...

    def synchronize(self) -> None: ...


class CpuNoopEvent:
    """What ``CpuNoopStream`` hands back: waiting on it is a no-op."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.waited = False

    def wait(self, stream=None) -> None:  # noqa: ARG002 - mirrors torch.cuda.Event
        self.waited = True
        self._events.append("wait")

    def synchronize(self) -> None:
        self.waited = True
        self._events.append("wait")


class CpuNoopStream:
    """CPU: no streams; records call order for the contract tests."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def begin(self) -> None:
        self.events.append("begin")

    def synchronize(self) -> None:
        self.events.append("synchronize")

    def record(self) -> CpuNoopEvent:
        self.events.append("record")
        return CpuNoopEvent(self.events)

    def gate(self, event: CpuNoopEvent) -> None:
        event.wait()


class CudaStreamProvider:
    """The GPU half of the seam (m18 D3): extraction runs on a side stream.

    ``begin()`` makes the side stream wait for whatever the caller's stream has
    already queued — otherwise the copy could read pages the forward has not
    finished writing — and then makes it current. ``synchronize()`` leaves that
    window and blocks until the copy has actually completed.

    A full ``stream.synchronize()`` is the BLOCKING form's deliberate choice: the
    handoff's commit point publishes the allocation to other threads, and the m18
    D3 ordering rule is that it must never run ahead of the copy. ``record()`` +
    ``gate()`` are the deferred form of the same rule, expressed in stream order
    rather than by stopping the host.

    What this buys, stated precisely because two earlier versions of this
    docstring got it wrong. ``begin()`` waits on everything ALREADY queued on the
    caller's stream — including work unrelated to the pages, which the copy
    therefore still follows. What it gains is the other direction: work the
    caller queues AFTER that point runs independently of the copy, because the
    two are on separate streams.
    """

    def __init__(self, device: object | None = None) -> None:
        import torch

        if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
            raise RuntimeError("CudaStreamProvider requires CUDA")
        self._torch = torch
        self._stream = torch.cuda.Stream(device=device)
        self._window: object | None = None

    def begin(self) -> None:
        if self._window is not None:  # pragma: no cover - misuse guard
            raise RuntimeError("CudaStreamProvider.begin() is already open")
        # the caller's stream ON THIS PROVIDER'S DEVICE: without the argument a
        # thread whose current device is 0 would wait on cuda:0 while the pages
        # were written on the cuda:1 stream this provider was built for
        self._stream.wait_stream(
            self._torch.cuda.current_stream(device=self._stream.device)
        )
        window = self._torch.cuda.stream(self._stream)
        window.__enter__()
        self._window = window

    def _close_window(self) -> None:
        window, self._window = self._window, None
        if window is not None:
            # runs from StreamCopyKVHandoff's finally, so it must close the
            # window even when the transfer raised — a leaked stream context
            # would silently redirect every later op on this thread
            window.__exit__(None, None, None)

    def synchronize(self) -> None:
        self._close_window()
        self._stream.synchronize()

    def record(self) -> object:
        """Close the window and return an EVENT instead of blocking the host.

        This is what makes overlap real: the producer can queue its next step
        immediately, and whoever consumes the copied pages waits on the event.
        ``synchronize()`` remains for callers that want the simple ordering.
        """
        self._close_window()
        event = self._torch.cuda.Event()
        event.record(self._stream)
        return event

    def gate(self, event) -> None:
        """Order everything the caller queues NEXT after ``event``.

        The host is not stopped: the dependency is expressed on the caller's
        stream, so both halves of the m6 D4 rule hold for stream-ordered work —
        reads of the destination pages and rewrites of the released source pages
        alike are queued behind the copy.
        """
        event.wait(self._torch.cuda.current_stream(device=self._stream.device))


class StreamCopyKVHandoff:
    """Wraps any KVHandoff: copy work happens inside the stream window.

    ``defer=True`` returns without blocking the host and records the copy's
    completion event. The producer can then queue its next step while the copy is
    still running — the overlap this seam is named for — provided the consumer
    closes the window it opens, with either:

    - ``gate_pending()``: order the caller's stream after the copy without
      stopping the host. This is what ``PDCoordinator`` uses, and it covers BOTH
      halves of m6 D4 — the decode read of the destination pages, and the reuse
      of the prefill-side source pages the release hands back to the pool. Note
      that not stopping the host is necessary but not sufficient: a gate applied
      before the producer has queued anything is a fence, and the copy stays on
      the critical path. The consumer picks the position, not this class.
    - ``wait_for_pending()``: block the host until the copy has landed. Simpler,
      and what a caller doing non-stream-ordered work with the pages needs.

    Default stays ``defer=False``: block before returning, so the commit point
    cannot run ahead of the copy. Deferring moves that responsibility to the
    caller, and a caller that forgets it reads half-written KV — or lets the next
    forward overwrite the pages the copy is reading.

    Events accumulate: a prefill step transfers every prompt that finished in it,
    so one deferred event per transfer is pending until the consumer settles them
    together. Keeping only the last would silently drop the ordering for every
    earlier copy in the step.
    """

    def __init__(self, inner, provider: StreamProvider, *, defer: bool = False) -> None:
        self._inner = inner
        self._provider = provider
        self._defer = defer
        self._pending: list[object] = []

    @property
    def defers(self) -> bool:
        """True when ``transfer()`` returns before the copy has completed.

        The consumer contract switch: a deferring handoff obliges its caller to
        settle ``gate_pending()``/``wait_for_pending()`` before reading the
        destination or releasing the source.
        """
        return self._defer

    @property
    def pending_event(self) -> object | None:
        """The most recent unsettled completion event, or None."""
        return self._pending[-1] if self._pending else None

    @property
    def pending_events(self) -> tuple[object, ...]:
        return tuple(self._pending)

    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        self._provider.begin()
        try:
            allocation = self._inner.transfer(tokens, first_token, pages)
        finally:
            if self._defer:
                record = getattr(self._provider, "record", None)
                # a raising transfer may already have queued part of the copy, so
                # its event is recorded too: the caller still has to gate before
                # aborting and releasing the source pages
                event = record() if record is not None else self._provider.synchronize()
                if event is not None:
                    self._pending.append(event)
            else:
                # the commit point must never run ahead of the copy (m6 D4)
                self._provider.synchronize()
        return allocation

    def _take_pending(self) -> tuple[object, ...]:
        events, self._pending = tuple(self._pending), []
        return events

    def wait_for_pending(self) -> None:
        """Block until every deferred copy has landed; a no-op when none is."""
        for event in self._take_pending():
            event.synchronize()

    def gate_pending(self) -> None:
        """Order the caller's subsequent stream work after every deferred copy.

        Unlike ``wait_for_pending()`` this does not stop the host, so work the
        producer queued BEFORE this call keeps running against the copy — which
        is why the caller is expected to have queued some. Providers without a
        ``gate`` fall back to the host wait rather than skipping the ordering.
        """
        gate = getattr(self._provider, "gate", None)
        for event in self._take_pending():
            if gate is not None:
                gate(event)
            else:  # pragma: no cover - every shipped provider has gate()
                event.synchronize()
