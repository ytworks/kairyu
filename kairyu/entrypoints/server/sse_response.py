"""Shared response policy for public Server-Sent Events endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterable

import anyio
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

_SSE_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


class _OwnedStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Disconnect can interrupt send() while the iterator is suspended at
            # yield. Starlette does not close that iterator; explicitly release
            # its outstanding upstream work before returning to the server.
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                with anyio.CancelScope(shield=True):
                    await close()


def sse_response(content: AsyncIterable[str | bytes]) -> StreamingResponse:
    """Return SSE with cache revalidation and NGINX buffering controls."""

    # HTTP/1.1 connections persist by default.  A ``Connection: keep-alive``
    # header adds no streaming guarantee and is forbidden by RFC 9113 §8.2.2.
    return _OwnedStreamingResponse(
        content,
        media_type="text/event-stream",
        headers=_SSE_RESPONSE_HEADERS,
    )
