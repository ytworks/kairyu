from __future__ import annotations

import re
from collections.abc import AsyncIterator

import httpx
import pytest

import kairyu.sse as sse_module
from kairyu.sse import _pop_line, iter_sse_data


class _ChunkedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _RecordingBuffer(bytearray):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)

    def __iter__(self):
        raise AssertionError("SSE framing must not iterate over individual bytes")

    def find(self, *args, **kwargs):
        raise AssertionError("SSE framing must use one CR-or-LF search")


class _RecordingPattern:
    def __init__(self, pattern: re.Pattern[bytes]) -> None:
        self._pattern = pattern
        self.search_starts: list[int] = []

    def search(self, buffer: bytearray, start: int = 0):
        self.search_starts.append(start)
        return self._pattern.search(buffer, start)


@pytest.mark.parametrize(
    ("wire", "expected", "remaining"),
    [
        (b"first\nrest", b"first", b"rest"),
        (b"first\r\nrest", b"first", b"rest"),
        (b"first\rrest", b"first", b"rest"),
        (b"\nrest", b"", b"rest"),
        (b"first\nsecond\r", b"first", b"second\r"),
        (b"first\rsecond\n", b"first", b"second\n"),
    ],
)
def test_pop_line_preserves_cr_lf_framing(
    wire: bytes,
    expected: bytes,
    remaining: bytes,
) -> None:
    buffer = bytearray(wire)

    line, scan_from = _pop_line(buffer, eof=False, scan_from=0)

    assert line == expected
    assert bytes(buffer) == remaining
    assert scan_from == 0


def test_pop_line_resumes_after_scanned_prefix_and_rechecks_trailing_cr(
    monkeypatch,
) -> None:
    pattern = _RecordingPattern(sse_module._SSE_LINE_END_RE)
    monkeypatch.setattr(sse_module, "_SSE_LINE_END_RE", pattern)
    buffer = _RecordingBuffer(b"fragment")

    line, scan_from = _pop_line(buffer, eof=False, scan_from=0)

    assert line is None
    assert scan_from == len(b"fragment")
    assert pattern.search_starts == [0]

    pattern.search_starts.clear()
    buffer.extend(b"ed")
    line, scan_from = _pop_line(buffer, eof=False, scan_from=scan_from)

    assert line is None
    assert scan_from == len(b"fragmented")
    assert pattern.search_starts == [len(b"fragment")]

    pattern.search_starts.clear()
    buffer.extend(b"\r")
    line, scan_from = _pop_line(buffer, eof=False, scan_from=scan_from)

    assert line is None
    assert scan_from == len(b"fragmented")
    assert pattern.search_starts == [len(b"fragmented")]

    pattern.search_starts.clear()
    buffer.extend(b"\nnext")
    line, scan_from = _pop_line(buffer, eof=False, scan_from=scan_from)

    assert line == b"fragmented"
    assert bytes(buffer) == b"next"
    assert scan_from == 0
    assert pattern.search_starts == [len(b"fragmented")]


def test_pop_line_uses_one_search_for_cr_only_lines(monkeypatch) -> None:
    pattern = _RecordingPattern(sse_module._SSE_LINE_END_RE)
    monkeypatch.setattr(sse_module, "_SSE_LINE_END_RE", pattern)
    line_count = 4_096
    buffer = _RecordingBuffer(b"value\r" * line_count + b"tail")

    for _ in range(line_count):
        line, scan_from = _pop_line(buffer, eof=False, scan_from=0)
        assert line == b"value"
        assert scan_from == 0

    assert bytes(buffer) == b"tail"
    assert pattern.search_starts == [0] * line_count


@pytest.mark.parametrize(
    ("wire", "expected", "remaining"),
    [
        (b"tail", b"tail", b""),
        (b"tail\r", b"tail", b""),
        (b"\r", b"", b""),
        (b"", None, b""),
    ],
)
def test_pop_line_preserves_eof_behavior(
    wire: bytes,
    expected: bytes | None,
    remaining: bytes,
) -> None:
    buffer = bytearray(wire)

    line, scan_from = _pop_line(buffer, eof=True, scan_from=0)

    assert line == expected
    assert bytes(buffer) == remaining
    assert scan_from == 0


async def test_iter_sse_data_preserves_fields_bom_and_unicode_separators() -> None:
    separators = "before\u0085middle\u2028after\u2029"
    wire = (
        b"\xef\xbb\xbf: keepalive\r"
        b"event: ignored\n"
        + f"data: {separators}\r\n".encode()
        + b"data: second\r"
        b"\r"
    )
    response = httpx.Response(
        200,
        stream=_ChunkedAsyncStream(
            wire[:1],
            wire[1:19],
            wire[19:37],
            wire[37:-1],
            wire[-1:],
        ),
    )

    events = [event async for event in iter_sse_data(response)]

    assert events == [f"{separators}\nsecond"]
