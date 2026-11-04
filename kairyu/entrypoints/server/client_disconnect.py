"""Cancel a request's in-flight work when its HTTP client disconnects (issue #495).

Shared by every unary route that awaits generation before it answers: the
chat completions orchestration and direct-engine paths, and the direct-engine
Messages and Responses paths. Streaming responses need no watcher; the
streaming response cancels its body when the client leaves.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable
from typing import Any

from fastapi import Request
from fastapi.responses import Response

logger = logging.getLogger(__name__)


async def _watch_client_disconnect(http_request: Request) -> None:
    """Resolve when the HTTP client drops the connection."""

    while True:
        message = await http_request.receive()
        if message["type"] == "http.disconnect":
            return


async def _run_until_client_disconnect(
    http_request: Request,
    work: Awaitable[Any],
) -> tuple[Any, Response | None]:
    """Await ``work`` unless the client disconnects first (issue #495).

    A dropped client can no longer read the response, so the in-flight
    work is cancelled to release replicas, the admission slot, and the
    tenant reservation immediately instead of running to completion.
    Returns ``(result, None)`` normally, or ``(None, response)`` after a
    disconnect-triggered cancellation; ``work``'s exceptions propagate.
    """

    work_task = asyncio.ensure_future(work)
    disconnect_task = asyncio.ensure_future(_watch_client_disconnect(http_request))
    try:
        done, _ = await asyncio.wait(
            (work_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if work_task not in done:
            work_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work_task
            logger.info("client disconnected; cancelled in-flight work")
            return None, Response(status_code=499)
    finally:
        disconnect_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_task
    return work_task.result(), None
