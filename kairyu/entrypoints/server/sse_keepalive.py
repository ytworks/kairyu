"""Shared SSE stream helpers for the Messages and Responses adapters."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterable, AsyncIterator


async def iter_with_idle_markers(
    stream: AsyncIterable[object], idle_seconds: float
) -> AsyncIterator[object]:
    """Yield stream items, interleaving ``None`` markers during silence.

    Each marker means ``idle_seconds`` passed without a new item, so callers can
    write a protocol-level liveness frame while a backend is still thinking.
    The pending ``anext`` is never cancelled by a marker, only on close, and
    closing the wrapper closes ``stream`` (a consumer that stops early stops
    the generation behind it).
    """

    iterator = stream.__aiter__()
    try:
        while True:
            task = asyncio.ensure_future(anext(iterator))
            try:
                while True:
                    done, _pending = await asyncio.wait({task}, timeout=idle_seconds)
                    if done:
                        break
                    yield None
            except BaseException:
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
                raise
            try:
                item = task.result()
            except StopAsyncIteration:
                return
            yield item
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


async def sse_frames(upstream: AsyncIterable[str | bytes]) -> AsyncIterator[str]:
    """Split an in-process SSE byte stream into frames."""

    buffer = ""
    async for chunk in upstream:
        buffer += chunk.decode() if isinstance(chunk, bytes) else chunk
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            if frame:
                yield frame
    if buffer:
        yield buffer
