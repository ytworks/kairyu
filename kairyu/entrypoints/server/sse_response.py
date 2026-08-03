"""Shared response policy for public Server-Sent Events endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterable

from fastapi.responses import StreamingResponse

_SSE_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def sse_response(content: AsyncIterable[str | bytes]) -> StreamingResponse:
    """Return SSE with cache revalidation and NGINX buffering controls."""

    # HTTP/1.1 connections persist by default.  A ``Connection: keep-alive``
    # header adds no streaming guarantee and is forbidden by RFC 9113 §8.2.2.
    return StreamingResponse(
        content,
        media_type="text/event-stream",
        headers=_SSE_RESPONSE_HEADERS,
    )
