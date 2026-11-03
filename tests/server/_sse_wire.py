"""Let a fake backend wait for SSE events the server has already sent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

# Hang bound only: an awaited wire event is expected within milliseconds.
HANG_S = 10.0


class SseWire:
    """ASGI wrapper recording the ``event:`` name of every SSE frame sent."""

    def __init__(self, app) -> None:
        self.app = app
        self.events: list[str] = []
        self._changed: asyncio.Event | None = None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def recording_send(message) -> None:
            await send(message)
            if message["type"] != "http.response.body":
                return
            body = message.get("body", b"").decode(errors="replace")
            for line in body.splitlines():
                if line.startswith("event: "):
                    self.events.append(line.removeprefix("event: "))
            if self._changed is not None:
                self._changed.set()

        await self.app(scope, receive, recording_send)

    async def wait_until(self, predicate: Callable[[list[str]], bool]) -> None:
        """Wait until ``predicate(events)`` holds; the bound only reports a hang."""

        async def wait() -> None:
            while not predicate(self.events):
                self._changed = asyncio.Event()
                await self._changed.wait()

        await asyncio.wait_for(wait(), timeout=HANG_S)


def event_after(first: str, then: str) -> Callable[[list[str]], bool]:
    """Predicate: ``then`` was sent after the first ``first`` event."""

    def sent(events: list[str]) -> bool:
        return first in events and then in events[events.index(first) + 1 :]

    return sent
