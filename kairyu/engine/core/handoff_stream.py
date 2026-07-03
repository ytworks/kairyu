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
