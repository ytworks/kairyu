"""Small, spec-oriented helpers for Server-Sent Events."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

_JSON_LINE_SEPARATOR_ESCAPES = str.maketrans(
    {
        "\u0085": "\\u0085",
        "\u2028": "\\u2028",
        "\u2029": "\\u2029",
    }
)


def escape_json_line_separators(serialized: str) -> str:
    """Keep valid JSON text on one physical SSE line for legacy clients."""

    if (
        "\u0085" not in serialized
        and "\u2028" not in serialized
        and "\u2029" not in serialized
    ):
        return serialized
    return serialized.translate(_JSON_LINE_SEPARATOR_ESCAPES)


def _pop_line(buffer: bytearray, *, eof: bool) -> bytes | None:
    """Pop one SSE line without treating Unicode separators as framing."""

    for index, value in enumerate(buffer):
        if value == 0x0A:  # LF
            line = bytes(buffer[:index])
            del buffer[: index + 1]
            return line
        if value != 0x0D:  # CR
            continue
        if index + 1 == len(buffer) and not eof:
            return None
        terminator_bytes = (
            2
            if index + 1 < len(buffer) and buffer[index + 1] == 0x0A
            else 1
        )
        line = bytes(buffer[:index])
        del buffer[: index + terminator_bytes]
        return line
    if eof and buffer:
        line = bytes(buffer)
        buffer.clear()
        return line
    return None


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield complete SSE data fields using CR/LF framing from raw bytes.

    ``httpx.Response.aiter_lines`` intentionally follows ``str.splitlines`` and
    therefore treats U+0085, U+2028, and U+2029 as line endings. Those code
    points are legal, unescaped JSON string content. SSE framing only uses CR
    and LF, so parse it before UTF-8 decoding and join repeated ``data:`` fields
    at the blank event boundary.
    """

    buffer = bytearray()
    data_lines: list[bytes] = []
    first_line = True

    async def consume(*, eof: bool) -> AsyncIterator[str]:
        nonlocal first_line
        while (line := _pop_line(buffer, eof=eof)) is not None:
            if first_line:
                line = line.removeprefix(b"\xef\xbb\xbf")
                first_line = False
            if not line:
                if data_lines:
                    yield b"\n".join(data_lines).decode("utf-8")
                    data_lines.clear()
                continue
            if line.startswith(b":"):
                continue
            field, separator, value = line.partition(b":")
            if separator and value.startswith(b" "):
                value = value[1:]
            if field == b"data":
                data_lines.append(value)

    async for chunk in response.aiter_bytes():
        buffer.extend(chunk)
        async for data in consume(eof=False):
            yield data
    async for data in consume(eof=True):
        yield data
