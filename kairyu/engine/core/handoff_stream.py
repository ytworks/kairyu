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


class CpuNoopStream:
    """CPU: no streams; records call order for the contract tests."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def begin(self) -> None:
        self.events.append("begin")

    def synchronize(self) -> None:
        self.events.append("synchronize")


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


class StreamCopyKVHandoff:
    """Wraps any KVHandoff: copy work happens inside the stream window."""

    def __init__(self, inner, provider: StreamProvider) -> None:
        self._inner = inner
        self._provider = provider

    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        self._provider.begin()
        try:
            allocation = self._inner.transfer(tokens, first_token, pages)
        finally:
            # the commit point must never run ahead of the copy (m6 D4)
            self._provider.synchronize()
        return allocation
