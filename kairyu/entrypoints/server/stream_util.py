"""Shared SSE stream helpers for the Messages and Responses adapters."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterable, AsyncIterator, Mapping

from kairyu.sse import escape_json_line_separators

# Yielded by ``iter_with_idle_markers`` when the wrapped stream stays silent.
IDLE_MARKER = None
# Items the pump may read ahead of the consumer. A queue that rarely runs dry
# keeps the per-item cost at one queue hand-off instead of a task switch.
_READ_AHEAD_ITEMS = 64
_END = object()


class _Raised:
    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


async def iter_with_idle_markers(
    stream: AsyncIterable[object],
    *,
    idle_seconds: float,
) -> AsyncIterator[object]:
    """Yield stream items, interleaving ``IDLE_MARKER`` during silence.

    Each marker means ``idle_seconds`` passed without the consumer receiving
    an item, so callers can write a protocol-level liveness frame while a
    backend is still thinking. One pump task reads the stream (at most
    ``_READ_AHEAD_ITEMS`` ahead) and one ticker task raises markers, so a fast
    stream pays no per-item timer or task. Callers must close the iterator
    (``aclose``/``contextlib.aclosing``) when they stop early: closing cancels
    the pump and closes ``stream`` before control returns.
    """

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue(maxsize=_READ_AHEAD_ITEMS)
    last_item_at = loop.time()

    async def pump() -> None:
        iterator = aiter(stream)
        outcome: object = _END
        try:
            async for item in iterator:
                await queue.put(item)
        except asyncio.CancelledError as error:
            if asyncio.current_task().cancelling():
                raise  # the consumer closed the iteration
            outcome = _Raised(error)  # the stream itself was cancelled
        except Exception as error:
            outcome = _Raised(error)
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()
        await queue.put(outcome)

    async def tick() -> None:
        while True:
            delay = last_item_at + idle_seconds - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
                continue
            if queue.empty():
                queue.put_nowait(IDLE_MARKER)
            await asyncio.sleep(idle_seconds)

    tasks = (asyncio.ensure_future(pump()), asyncio.ensure_future(tick()))
    try:
        while True:
            item = queue.get_nowait() if not queue.empty() else await queue.get()
            last_item_at = loop.time()
            if item is _END:
                return
            if isinstance(item, _Raised):
                raise item.error
            yield item
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(BaseException):
                await task


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


def sse_event(event_type: str, payload: Mapping) -> str:
    """One named SSE event whose data is compact JSON on a single line."""

    serialized = escape_json_line_separators(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return f"event: {event_type}\ndata: {serialized}\n\n"


class DataHeartbeat:
    """Single-owner clock of the last data event a stream wrote.

    Upstream items that produce no client-visible event (thinking, prefill
    progress, orchestrator status comments) must not count as liveness for
    clients that reset their idle timers only on data events.
    """

    def __init__(self, interval_s: float) -> None:
        self._interval_s = interval_s
        self._last = time.monotonic()

    def mark(self) -> None:
        self._last = time.monotonic()

    def due(self) -> bool:
        return time.monotonic() - self._last >= self._interval_s


def primary_chat_delta(chunk_payload: Mapping) -> tuple[str, str | None]:
    """Text delta and finish reason of choice 0 in one chat-chunk payload."""

    for choice in chunk_payload.get("choices") or ():
        if choice.get("index", 0) == 0:
            content = (choice.get("delta") or {}).get("content")
            text = content if isinstance(content, str) else ""
            return text, choice.get("finish_reason")
    return "", None
