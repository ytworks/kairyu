"""Decode ``Content-Encoding`` request bodies under an output cap (M20 WP-41, m20 D21).

gzip (and its ``x-gzip`` alias, RFC 9110) is decoded with the stdlib; zstd
needs the optional ``kairyu[zstd]`` extra (``zstandard``). Codex sends zstd
bodies when it uses ChatGPT auth against an OpenAI base URL. Any other coding,
a list of codings, or zstd without the extra is refused with 415
``unsupported_content_encoding`` in the route's error dialect, and the answer
names what is accepted in ``Accept-Encoding`` (RFC 7694).

The body is decoded as the app reads it, chunk by chunk, on the request-body
worker (``run_request_body_work``) so a large body never stalls the event
loop. Every decoding step is bounded -- zlib by its output step, zstd by
feeding at most ``_ZSTD_SLICE_BYTES`` of input per call -- so a decompression
bomb is refused with 413 once its output passes ``max_decompressed_bytes``,
never inflated in full. zstd frames may need at most an 8 MiB window, the
HTTP limit of RFC 9659. Corrupt or truncated data is a 400.

The app sees the decoded body without ``content-encoding`` and
``content-length``, so the body limits inside this middleware bound the
decoded JSON.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Awaitable, Callable, Iterator, Mapping
from functools import partial
from types import MappingProxyType, ModuleType
from typing import Protocol

from starlette.requests import ClientDisconnect

from kairyu.async_thread import run_request_body_work
from kairyu.entrypoints.server.error_classifier import ClassifiedError, pre_stream_error
from kairyu.entrypoints.server.middleware import send_error

logger = logging.getLogger(__name__)

_ASGIApp = Callable[..., Awaitable[None]]
_INVALID = "invalid_request_error"
_STRIPPED_HEADERS = frozenset({b"content-encoding", b"content-length"})
_IDENTITY = "identity"
# gzip header and trailer only (no zlib or raw-deflate fallback).
_GZIP_WBITS = 16 + zlib.MAX_WBITS
# The most zlib inflates in one step before the cap is checked again.
_GZIP_STEP_BYTES = 1 << 20
# A zstd block regenerates at most 128 KiB from at least 4 input bytes, so a
# 64-byte slice yields at most 2 MiB however the frame is crafted.
_ZSTD_SLICE_BYTES = 64
_ZSTD_MAX_WINDOW_BYTES = 8 << 20


class _MalformedBody(Exception):
    """The compressed data is corrupt or ends inside a member or frame."""


class _OverCap(Exception):
    """The decoded body grew past the configured cap."""


class _CodecStream(Protocol):
    """One request body's decoder; each yielded piece is bounded.

    ``pieces`` raises ``_MalformedBody`` for corrupt input.
    """

    def pieces(self, data: bytes) -> Iterator[bytes]: ...

    def at_frame_end(self) -> bool: ...


class _GzipStream:
    """Concatenated gzip members (RFC 1952), at most one zlib step per piece."""

    def __init__(self) -> None:
        self._member = zlib.decompressobj(_GZIP_WBITS)

    def pieces(self, data: bytes) -> Iterator[bytes]:
        pending = data
        while True:
            try:
                piece = self._member.decompress(pending, _GZIP_STEP_BYTES)
            except zlib.error as error:
                raise _MalformedBody from error
            pending = self._member.unconsumed_tail
            if piece:
                yield piece
            if self._member.eof and self._member.unused_data:
                pending = self._member.unused_data  # the next member
                self._member = zlib.decompressobj(_GZIP_WBITS)
            elif not pending and len(piece) < _GZIP_STEP_BYTES:
                return  # input consumed and no output left inside zlib

    def at_frame_end(self) -> bool:
        return self._member.eof


class _ZstdStream:
    """Concatenated zstd frames, fed in slices that bound each call's output."""

    def __init__(self, zstd: ModuleType) -> None:
        self._error = zstd.ZstdError
        self._decompressor = zstd.ZstdDecompressor(max_window_size=_ZSTD_MAX_WINDOW_BYTES)
        self._frame = self._decompressor.decompressobj()

    def pieces(self, data: bytes) -> Iterator[bytes]:
        view = memoryview(data)
        for start in range(0, len(view), _ZSTD_SLICE_BYTES):
            pending = view[start : start + _ZSTD_SLICE_BYTES]
            while pending:
                if self._frame.eof:
                    self._frame = self._decompressor.decompressobj()  # the next frame
                try:
                    piece = self._frame.decompress(pending)
                except self._error as error:
                    raise _MalformedBody from error
                pending = self._frame.unused_data if self._frame.eof else b""
                if piece:
                    yield piece

    def at_frame_end(self) -> bool:
        return self._frame.eof


class _CappedDecoder:
    """Single-owner decoding state of one request body, one chunk at a time."""

    def __init__(self, stream: _CodecStream, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self._produced = 0
        self._saw_input = False

    def decode(self, data: bytes) -> bytes:
        """Decode one received chunk; runs on the request-body worker."""

        decoded: list[bytes] = []
        for piece in self._stream.pieces(data):
            self._produced += len(piece)
            if self._produced > self._limit:
                raise _OverCap
            decoded.append(piece)
        self._saw_input = self._saw_input or bool(data)
        return b"".join(decoded)

    def finish(self) -> None:
        if self._saw_input and not self._stream.at_frame_end():
            raise _MalformedBody


def _optional_zstd() -> ModuleType | None:
    try:
        import zstandard
    except ImportError:
        return None
    return zstandard


def _content_codings(scope: dict) -> tuple[str, ...]:
    values = (
        value.decode("latin-1") for name, value in scope["headers"] if name == b"content-encoding"
    )
    tokens = (token.strip().lower() for value in values for token in value.split(","))
    return tuple(token for token in tokens if token and token != _IDENTITY)


def _unsupported(codings: tuple[str, ...], accepted: str) -> ClassifiedError:
    named = ", ".join(codings)
    return pre_stream_error(
        415,
        _INVALID,
        "unsupported_content_encoding",
        f"Content-Encoding '{named}' is not supported; this server accepts {accepted}.",
    )


def _too_large(limit: int) -> ClassifiedError:
    return pre_stream_error(
        413,
        _INVALID,
        "request_too_large",
        f"decompressed request body exceeds the configured {limit}-byte limit",
    )


def _malformed(coding: str) -> ClassifiedError:
    return pre_stream_error(
        400,
        _INVALID,
        "invalid_content_encoding",
        f"The request body is not valid {coding} data.",
    )


class DecompressionMiddleware:
    """Decode gzip and (with the extra) zstd request bodies, capped (m20 D21)."""

    def __init__(self, app: _ASGIApp, *, max_decompressed_bytes: int) -> None:
        if max_decompressed_bytes < 1:
            raise ValueError("max_decompressed_bytes must be positive")
        self.app = app
        self._limit = max_decompressed_bytes
        zstd = _optional_zstd()
        codecs: dict[str, Callable[[], _CodecStream]] = {"gzip": _GzipStream}
        if zstd is not None:
            codecs["zstd"] = partial(_ZstdStream, zstd)
        self._accept_encoding = ", ".join(codecs)
        codecs["x-gzip"] = _GzipStream
        self._codecs: Mapping[str, Callable[[], _CodecStream]] = MappingProxyType(codecs)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        codings = _content_codings(scope)
        if not codings:
            await self.app(scope, receive, send)
            return
        codec = self._codecs.get(codings[0]) if len(codings) == 1 else None
        if codec is None:
            error = _unsupported(codings, self._accept_encoding)
            await send_error(send, scope, error, {"accept-encoding": self._accept_encoding})
            return
        decoder = _CappedDecoder(codec(), self._limit)
        exchange = _DecodedExchange(scope, receive, send, decoder, codings[0], self._limit)
        await exchange.run(self.app)


class _DecodedExchange:
    """One request served with a decoded body; a refusal answers in place of the app."""

    def __init__(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
        decoder: _CappedDecoder,
        coding: str,
        limit: int,
    ) -> None:
        self._scope = scope
        self._receive = receive
        self._send = send
        self._decoder = decoder
        self._coding = coding
        self._limit = limit
        self._refused = False
        self._started = False

    async def run(self, app: _ASGIApp) -> None:
        self._scope.setdefault("state", {})  # shared with the decoded scope below
        headers = [(k, v) for k, v in self._scope["headers"] if k not in _STRIPPED_HEADERS]
        try:
            await app({**self._scope, "headers": headers}, self._receive_decoded, self._send_app)
        except ClientDisconnect:
            if not self._refused:
                raise

    async def _receive_decoded(self) -> dict:
        if self._refused:
            return {"type": "http.disconnect"}
        message = await self._receive()
        if message["type"] != "http.request":
            return message
        more_body = message.get("more_body", False)
        body = message.get("body", b"")
        try:
            decoded = await run_request_body_work(self._decoder.decode, body) if body else b""
            if not more_body:
                self._decoder.finish()
        except _OverCap:
            logger.warning(
                "refused a %s request body over the %d-byte decompression cap (%s %s)",
                self._coding,
                self._limit,
                self._scope.get("method"),
                self._scope.get("path"),
            )
            return await self._refuse(_too_large(self._limit))
        except _MalformedBody:
            return await self._refuse(_malformed(self._coding))
        return {"type": "http.request", "body": decoded, "more_body": more_body}

    async def _refuse(self, error: ClassifiedError) -> dict:
        self._refused = True
        if not self._started:
            await send_error(self._send, self._scope, error)
        return {"type": "http.disconnect"}

    async def _send_app(self, message: dict) -> None:
        if self._refused:
            return
        if message["type"] == "http.response.start":
            self._started = True
        await self._send(message)
