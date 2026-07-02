"""KV transfer plane: LocalFabric and TCP-loopback transports (design m6 D3)."""

from __future__ import annotations

import asyncio

import pytest

from kairyu.engine.core.kv_transport import (
    KVTransportError,
    LocalFabric,
    PageFrame,
    SequenceMeta,
    TcpLoopbackTransport,
)


def _frames(num_pages: int, fragments_per_page: int = 4, fragment_bytes: int = 32):
    return tuple(
        PageFrame(
            page_id=page,
            fragments=tuple(
                bytes([page % 251]) * fragment_bytes for _ in range(fragments_per_page)
            ),
        )
        for page in range(num_pages)
    )


_META = SequenceMeta(token_ids=(1, 2, 3, 4), first_token=42)


def test_local_fabric_roundtrip_preserves_frames_and_meta() -> None:
    fabric = LocalFabric()
    sender = fabric.endpoint("prefill")
    receiver = fabric.endpoint("decode")
    sender.register(num_pages=16)
    receiver.register(num_pages=16)
    frames = _frames(3)

    async def roundtrip():
        await sender.send("decode", frames, _META)
        return await receiver.recv("prefill")

    got_frames, got_meta = asyncio.run(roundtrip())
    assert got_frames == frames
    assert got_meta == _META


def test_send_requires_registration_and_frames() -> None:
    fabric = LocalFabric()
    sender = fabric.endpoint("a")
    fabric.endpoint("b")

    async def send_unregistered():
        await sender.send("b", _frames(1), _META)

    with pytest.raises(KVTransportError, match="register"):
        asyncio.run(send_unregistered())

    sender.register(num_pages=4)
    with pytest.raises(KVTransportError, match="register"):
        sender.register(num_pages=4)  # register once, at startup (design m6 D3)

    async def send_empty():
        await sender.send("b", (), _META)

    with pytest.raises(KVTransportError, match="frame"):
        asyncio.run(send_empty())


def test_tcp_loopback_roundtrip_with_real_serialization() -> None:
    async def roundtrip():
        receiver = TcpLoopbackTransport("decode")
        receiver.register(num_pages=128)
        address = await receiver.start_server()
        sender = TcpLoopbackTransport("prefill")
        sender.register(num_pages=128)
        frames = _frames(8, fragments_per_page=10, fragment_bytes=512)
        try:
            await sender.send(address, frames, _META)
            got_frames, got_meta = await receiver.recv(address)
        finally:
            await sender.close()
            await receiver.close()
        return frames, got_frames, got_meta

    frames, got_frames, got_meta = asyncio.run(roundtrip())
    assert got_frames == frames
    assert got_meta == _META


def test_tcp_loopback_reuses_one_connection_for_many_sends() -> None:
    async def run():
        receiver = TcpLoopbackTransport("decode")
        receiver.register(num_pages=64)
        address = await receiver.start_server()
        sender = TcpLoopbackTransport("prefill")
        sender.register(num_pages=64)
        try:
            for _ in range(5):
                await sender.send(address, _frames(2), _META)
            batches = [await receiver.recv(address) for _ in range(5)]
            connections = receiver.connections_accepted
        finally:
            await sender.close()
            await receiver.close()
        return batches, connections

    batches, connections = asyncio.run(run())
    assert len(batches) == 5
    assert all(meta == _META for _, meta in batches)
    assert connections == 1  # persistent connection: no per-send handshake


def test_rejects_duplicate_page_ids_in_one_send() -> None:
    fabric = LocalFabric()
    sender = fabric.endpoint("a")
    fabric.endpoint("b")
    sender.register(num_pages=4)
    duplicated = (_frames(1)[0], _frames(1)[0])

    async def send_duplicated():
        await sender.send("b", duplicated, _META)

    with pytest.raises(KVTransportError, match="duplicate"):
        asyncio.run(send_duplicated())
