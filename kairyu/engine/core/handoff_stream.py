"""Side-stream KV handoff (m18 D3): extraction on a stream of its own.

``StreamProvider`` is the CUDA seam, with ``CpuNoopStream`` and
``CudaStreamProvider`` as its two implementations. ``StreamCopyKVHandoff`` pins
the ordering: enter stream → inner.transfer (which extracts+copies) →
synchronize → return. A recording fake tests the order.

Scope, because the name invites a bigger reading: this runs the copy on its own
stream. It does NOT overlap the copy with the next forward. ``transfer()`` blocks
the host before returning and ``PDCoordinator`` commits before stepping decode,
so nothing is queued alongside it, and no production path constructs a
``CudaStreamProvider`` yet. Both need the consumer to take a completion EVENT in
place of the host-wide wait.
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


class CudaStreamProvider:
    """The GPU half of the seam (m18 D3): extraction runs on a side stream.

    ``begin()`` makes the side stream wait for whatever the caller's stream has
    already queued — otherwise the copy could read pages the forward has not
    finished writing — and then makes it current. ``synchronize()`` leaves that
    window and blocks until the copy has actually completed.

    A full ``stream.synchronize()`` is deliberate rather than an event wait: the
    handoff's commit point publishes the allocation to other threads, and the m18
    D3 ordering rule is that it must never run ahead of the copy.

    What this buys, stated precisely because two earlier versions of this
    docstring got it wrong. ``begin()`` waits on everything ALREADY queued on the
    caller's stream — including work unrelated to the pages, which the copy
    therefore still follows. What it gains is the other direction: work the
    caller queues AFTER that point runs independently of the copy, because the
    two are on separate streams.

    It does NOT overlap the copy with the next forward. ``StreamCopyKVHandoff``
    blocks the host before returning and ``PDCoordinator`` commits before
    stepping decode, so nothing is queued alongside it. That needs the consumer
    to take a completion EVENT instead of the host-wide wait, which is not in
    this seam.
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

    def synchronize(self) -> None:
        window, self._window = self._window, None
        if window is not None:
            # runs from StreamCopyKVHandoff's finally, so it must close the
            # window even when the transfer raised — a leaked stream context
            # would silently redirect every later op on this thread
            window.__exit__(None, None, None)
        self._stream.synchronize()

    def record(self) -> object:
        """Close the window and return an EVENT instead of blocking the host.

        This is what makes overlap real: the producer can queue its next step
        immediately, and whoever consumes the copied pages waits on the event.
        ``synchronize()`` remains for callers that want the simple ordering.
        """
        window, self._window = self._window, None
        if window is not None:
            window.__exit__(None, None, None)
        event = self._torch.cuda.Event()
        event.record(self._stream)
        return event


class StreamCopyKVHandoff:
    """Wraps any KVHandoff: copy work happens inside the stream window.

    ``defer=True`` returns without blocking the host and exposes the copy's
    completion event on ``pending_event``. The producer can then queue its next
    step while the copy is still running — the overlap this seam is named for —
    provided the CONSUMER waits on the event before reading the pages.

    Default stays ``defer=False``: block before returning, so the commit point
    cannot run ahead of the copy (m6 D4). Deferring moves that responsibility to
    the caller, and a caller that forgets it reads half-written KV.
    """

    def __init__(self, inner, provider: StreamProvider, *, defer: bool = False) -> None:
        self._inner = inner
        self._provider = provider
        self._defer = defer
        self.pending_event: object | None = None

    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        self._provider.begin()
        try:
            allocation = self._inner.transfer(tokens, first_token, pages)
        finally:
            if self._defer:
                record = getattr(self._provider, "record", None)
                self.pending_event = (
                    record() if record is not None else self._provider.synchronize()
                )
            else:
                # the commit point must never run ahead of the copy (m6 D4)
                self._provider.synchronize()
        return allocation

    def wait_for_pending(self) -> None:
        """Block until the deferred copy has landed; a no-op when nothing is."""
        event, self.pending_event = self.pending_event, None
        if event is not None:
            event.synchronize()
