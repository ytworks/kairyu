"""Stream-overlapped KV handoff (m18 D3): extraction on a side stream.

``StreamProvider`` is the CUDA seam: ``CpuNoopStream`` here,
``CudaStreamProvider`` on deploy day (a context manager over
``torch.cuda.stream`` + ``synchronize``). ``StreamCopyKVHandoff`` pins the
ordering: enter stream → inner.transfer (which extracts+copies) →
synchronize → return. A recording fake tests the order.
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

    A full ``stream.synchronize()`` is deliberate rather than an event wait. The
    handoff's commit point publishes the allocation to other threads, and the
    m18 D3 ordering rule is that it must never run ahead of the copy; a host-side
    block is the version of that which cannot be got subtly wrong. The overlap
    being bought is against the *next* forward, not against this caller.
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
        self._stream.wait_stream(self._torch.cuda.current_stream())
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
