"""Concurrency contract of the shared idle-marker stream wrapper."""

from __future__ import annotations

import asyncio

from kairyu.entrypoints.server.stream_util import IDLE_MARKER, iter_with_idle_markers


async def test_idle_markers_keep_the_pending_item_alive():
    # A backend that thinks before its first partial must surface liveness
    # markers while that same partial keeps running, not restart or drop it.
    async def thinking_source():
        await asyncio.sleep(0.2)
        yield "first"
        yield "second"

    items = [
        item
        async for item in iter_with_idle_markers(thinking_source(), idle_seconds=0.03)
    ]

    assert items.count(IDLE_MARKER) >= 2
    assert [item for item in items if item is not IDLE_MARKER] == ["first", "second"]


async def test_closing_early_stops_reading_and_closes_the_source():
    # A client that leaves mid-stream must stop generation: closing the
    # wrapper cancels its pump and closes the source before returning, so a
    # caller can safely close the upstream generator next.
    produced = []
    closed = asyncio.Event()

    async def endless_source():
        try:
            index = 0
            while True:
                produced.append(index)
                yield index
                index += 1
                await asyncio.sleep(0.001)
        finally:
            closed.set()

    stream = iter_with_idle_markers(endless_source(), idle_seconds=10.0)
    assert await anext(stream) == 0
    await stream.aclose()
    produced_at_close = len(produced)
    await asyncio.sleep(0.05)

    assert closed.is_set()
    assert len(produced) == produced_at_close
