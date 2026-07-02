"""Page-granular KV transfer plane (design m6 D3).

The protocol carries whole pages as `PageFrame`s of per-layer/per-rank
*fragments* — the real sharded layout (a logical page spans layers and TP
shards), because fragment aggregation is part of the protocol, not the
transport's problem. `register()` is called once at startup (the GPU-phase
RDMA transport pins the whole pool there; re-registering is an error by
contract). Implementations here: `LocalFabric` (in-process, unit tests and the
M5 intra-node degenerate case) and `TcpLoopbackTransport` (real serialization
and framing over asyncio TCP — what `bench/kv_transfer_bench.py` measures on
CPU; the GPU phase swaps in NCCL p2p / RDMA behind the same seam).

Framing (TCP): 4-byte big-endian header length, JSON header (page ids,
fragment sizes, sequence meta), then all fragments concatenated in one
aggregated body write — the staging-ring analog.
"""

from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass, field
from typing import Protocol

_HEADER_LEN_BYTES = 4


class KVTransportError(RuntimeError):
    """Transfer-plane failure (unregistered endpoint, bad frame, dead peer)."""


@dataclass(frozen=True)
class SequenceMeta:
    """Rides with the pages so the receiver can fold them into its radix tree."""

    token_ids: tuple[int, ...]
    first_token: int | None = None


@dataclass(frozen=True)
class PageFrame:
    """One logical page as its physical fragments (per layer x per TP shard)."""

    page_id: int
    fragments: tuple[bytes, ...]


class KVTransport(Protocol):
    def register(self, num_pages: int) -> None: ...

    async def send(
        self, dst: str, frames: tuple[PageFrame, ...], meta: SequenceMeta
    ) -> None: ...

    async def recv(self, src: str) -> tuple[tuple[PageFrame, ...], SequenceMeta]: ...


def _validate_send(registered: bool, frames: tuple[PageFrame, ...]) -> None:
    if not registered:
        raise KVTransportError("transport is not registered; call register() at startup")
    if not frames:
        raise KVTransportError("send requires at least one page frame")
    page_ids = [frame.page_id for frame in frames]
    if len(set(page_ids)) != len(page_ids):
        raise KVTransportError(f"duplicate page ids in one send: {sorted(page_ids)}")


def _validate_register(registered: bool, num_pages: int) -> None:
    if registered:
        raise KVTransportError("transport already registered (register once, at startup)")
    if num_pages < 1:
        raise KVTransportError(f"num_pages must be >= 1, got {num_pages}")


@dataclass
class _LocalEndpoint:
    """One named endpoint on an in-process fabric."""

    name: str
    fabric: LocalFabric
    registered: bool = False

    def register(self, num_pages: int) -> None:
        _validate_register(self.registered, num_pages)
        self.registered = True

    async def send(
        self, dst: str, frames: tuple[PageFrame, ...], meta: SequenceMeta
    ) -> None:
        _validate_send(self.registered, frames)
        await self.fabric.channel(self.name, dst).put((frames, meta))

    async def recv(self, src: str) -> tuple[tuple[PageFrame, ...], SequenceMeta]:
        return await self.fabric.channel(src, self.name).get()


@dataclass
class LocalFabric:
    """In-process fabric: FIFO channel per (src, dst) endpoint pair."""

    _channels: dict[tuple[str, str], asyncio.Queue] = field(default_factory=dict)
    _endpoints: dict[str, _LocalEndpoint] = field(default_factory=dict)

    def endpoint(self, name: str) -> _LocalEndpoint:
        if name not in self._endpoints:
            self._endpoints[name] = _LocalEndpoint(name=name, fabric=self)
        return self._endpoints[name]

    def channel(self, src: str, dst: str) -> asyncio.Queue:
        if dst not in self._endpoints:
            raise KVTransportError(f"unknown destination endpoint {dst!r}")
        return self._channels.setdefault((src, dst), asyncio.Queue())


def _encode(frames: tuple[PageFrame, ...], meta: SequenceMeta) -> bytes:
    header = json.dumps(
        {
            "page_ids": [frame.page_id for frame in frames],
            "fragment_sizes": [
                [len(fragment) for fragment in frame.fragments] for frame in frames
            ],
            "meta": {"token_ids": list(meta.token_ids), "first_token": meta.first_token},
        }
    ).encode()
    body = b"".join(fragment for frame in frames for fragment in frame.fragments)
    return struct.pack(">I", len(header)) + header + body


async def _decode(reader: asyncio.StreamReader) -> tuple[tuple[PageFrame, ...], SequenceMeta]:
    header_len = struct.unpack(">I", await reader.readexactly(_HEADER_LEN_BYTES))[0]
    header = json.loads(await reader.readexactly(header_len))
    body = await reader.readexactly(
        sum(size for sizes in header["fragment_sizes"] for size in sizes)
    )
    frames: list[PageFrame] = []
    offset = 0
    for page_id, sizes in zip(header["page_ids"], header["fragment_sizes"], strict=True):
        fragments: list[bytes] = []
        for size in sizes:
            fragments.append(body[offset : offset + size])
            offset += size
        frames.append(PageFrame(page_id=page_id, fragments=tuple(fragments)))
    meta = SequenceMeta(
        token_ids=tuple(header["meta"]["token_ids"]),
        first_token=header["meta"]["first_token"],
    )
    return tuple(frames), meta


class TcpLoopbackTransport:
    """KVTransport over asyncio TCP with persistent per-destination connections."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._registered = False
        self._server: asyncio.AbstractServer | None = None
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._connections: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self.connections_accepted = 0

    def register(self, num_pages: int) -> None:
        _validate_register(self._registered, num_pages)
        self._registered = True

    async def start_server(self, host: str = "127.0.0.1") -> str:
        self._server = await asyncio.start_server(self._serve_connection, host, 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"{host}:{port}"

    async def _serve_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections_accepted += 1
        try:
            while True:
                await self._inbox.put(await _decode(reader))
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass  # peer closed; queued batches remain consumable
        finally:
            writer.close()

    async def send(
        self, dst: str, frames: tuple[PageFrame, ...], meta: SequenceMeta
    ) -> None:
        _validate_send(self._registered, frames)
        if dst not in self._connections:
            host, _, port = dst.rpartition(":")
            try:
                self._connections[dst] = await asyncio.open_connection(host, int(port))
            except (OSError, ValueError) as error:
                raise KVTransportError(f"cannot connect to {dst!r}: {error}") from error
        _, writer = self._connections[dst]
        writer.write(_encode(frames, meta))
        await writer.drain()

    async def recv(self, src: str) -> tuple[tuple[PageFrame, ...], SequenceMeta]:
        if not self._registered:
            raise KVTransportError("transport is not registered; call register() at startup")
        return await self._inbox.get()

    async def close(self) -> None:
        for _, writer in self._connections.values():
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass
        self._connections.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
