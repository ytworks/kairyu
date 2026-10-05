"""Content-Encoding request bodies (M20 WP-41, m20 D21, G-transport-3).

gzip is decoded in core and zstd with the ``kairyu[zstd]`` extra (in the dev
group, so the Codex ``harbor-zstd`` fixture replays a zstd body end to end).
Without the extra a zstd body is 415 ``unsupported_content_encoding``. A
decompression bomb stops at the default 64 MiB output cap with 413: each codec
bounds every decoding step its own way (zlib output steps, zstd input slices),
so each has its own bomb. The same cap bounds the compressed bytes: padding
that decodes to nothing (empty gzip members) must not stream past every body
limit.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from tests.server._legacy_chat import create_legacy_app

_TURN = json.dumps({"model": "m", "input": "hello"}).encode()
_CAP = ServerSettings().max_decompressed_bytes
_GZIP_MEMBER_BYTES = 16 << 20
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
# Frame header: no content size, not single-segment; window 2^(10+7) = 128 KiB.
_ZSTD_FRAME_HEADER = b"\x00\x38"
_ZSTD_RLE_BLOCK_BYTES = 128 << 10
_ZSTD_RLE = 1
_PADDING_CAP = 64 << 10


def _gzip_bomb() -> bytes:
    """Concatenated gzip members (RFC 1952) that inflate past the cap."""

    member = gzip.compress(bytes(_GZIP_MEMBER_BYTES), compresslevel=9)
    return member * (_CAP // _GZIP_MEMBER_BYTES + 1)


def _gzip_padding() -> bytes:
    """Empty gzip members past a small cap, then a valid turn: decodes to the turn."""

    empty_member = gzip.compress(b"")
    return empty_member * (2 * _PADDING_CAP // len(empty_member)) + gzip.compress(_TURN)


def _zstd_zeros(size: int) -> bytes:
    """A zstd frame of RLE blocks: 4 bytes on the wire per 128 KiB of zeros."""

    blocks = -(-size // _ZSTD_RLE_BLOCK_BYTES)
    frame = bytearray(_ZSTD_MAGIC + _ZSTD_FRAME_HEADER)
    for index in range(blocks):
        last = index == blocks - 1
        header = (_ZSTD_RLE_BLOCK_BYTES << 3) | (_ZSTD_RLE << 1) | int(last)
        frame += header.to_bytes(3, "little") + b"\x00"
    return bytes(frame)


@dataclass(frozen=True)
class _Case:
    encoding: str
    body: Callable[[], bytes]
    status: int
    code: str | None = None
    zstd_installed: bool = True
    accept_encoding: str | None = None
    max_decompressed_bytes: int = _CAP


_CASES = {
    "gzip": _Case("gzip", lambda: gzip.compress(_TURN), 200),
    "gzip-bomb": _Case("gzip", _gzip_bomb, 413, "request_too_large"),
    "zstd-bomb": _Case("zstd", lambda: _zstd_zeros(16 * _CAP), 413, "request_too_large"),
    "gzip-padding": _Case(
        "gzip",
        _gzip_padding,
        413,
        "request_too_large",
        max_decompressed_bytes=_PADDING_CAP,
    ),
    "zstd-without-extra": _Case(
        "zstd",
        lambda: _zstd_zeros(_ZSTD_RLE_BLOCK_BYTES),
        415,
        "unsupported_content_encoding",
        zstd_installed=False,
        accept_encoding="gzip",
    ),
}


@pytest.mark.parametrize("case", list(_CASES.values()), ids=list(_CASES))
async def test_compressed_request_bodies(case: _Case, monkeypatch) -> None:
    if not case.zstd_installed:
        monkeypatch.setitem(sys.modules, "zstandard", None)  # import fails
    settings = ServerSettings(max_decompressed_bytes=case.max_decompressed_bytes)
    app = create_legacy_app({"m": MockBackend()}, settings=settings)
    headers = {"content-type": "application/json", "content-encoding": case.encoding}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/responses", content=case.body(), headers=headers)

    assert response.status_code == case.status, response.text[:300]
    body = response.json()
    if case.status == 200:
        assert body["status"] == "completed"
        assert body["output"][0]["content"][0]["text"]
        return
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == case.code
    assert response.headers.get("accept-encoding") == case.accept_encoding
