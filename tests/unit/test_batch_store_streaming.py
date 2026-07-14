import asyncio

import pytest

import kairyu.batch.store as batch_store

BatchStore = batch_store.BatchStore


async def _chunks(*chunks: bytes):
    for chunk in chunks:
        yield chunk


async def test_streaming_save_publishes_exact_owner_scoped_file(tmp_path):
    store = BatchStore(tmp_path)

    file = await store.save_file_streaming(
        _chunks(b"first-", b"second", b"-third"),
        filename="input.jsonl",
        purpose="batch",
        owner="tenant-a",
        max_bytes=1024,
    )

    expected = b"first-second-third"
    assert file.bytes == len(expected)
    assert file.filename == "input.jsonl"
    assert file.purpose == "batch"
    assert file.owner == "tenant-a"
    assert store.get_file(file.id, owner="tenant-a") == file
    assert store.read_file_content(file.id, owner="tenant-a") == expected
    assert len(list((tmp_path / "files").glob("*.bin"))) == 1
    assert len(list((tmp_path / "files").glob("*.json"))) == 1
    assert list((tmp_path / "files").glob("*.tmp")) == []


async def test_streaming_save_accepts_exact_limit_and_rejects_next_byte(tmp_path):
    exact_store = BatchStore(tmp_path / "exact")
    exact = await exact_store.save_file_streaming(
        _chunks(b"12", b"345"), "exact", "batch", max_bytes=5
    )
    assert exact_store.read_file_content(exact.id) == b"12345"

    over_store = BatchStore(tmp_path / "over")
    with pytest.raises(batch_store.FileTooLargeError):
        await over_store.save_file_streaming(
            _chunks(b"12", b"345", b"6"), "over", "batch", max_bytes=5
        )
    assert list((tmp_path / "over" / "files").iterdir()) == []


async def test_streaming_save_discards_partial_file_on_iterator_failure(tmp_path):
    store = BatchStore(tmp_path)

    async def broken_chunks():
        yield b"partial"
        raise RuntimeError("stream failed")

    with pytest.raises(RuntimeError, match="stream failed"):
        await store.save_file_streaming(
            broken_chunks(), "broken", "batch", max_bytes=1024
        )

    assert list((tmp_path / "files").iterdir()) == []


async def test_streaming_save_discards_partial_file_on_cancellation(tmp_path):
    store = BatchStore(tmp_path)
    waiting = asyncio.Event()

    async def stalled_chunks():
        yield b"partial"
        waiting.set()
        await asyncio.Event().wait()
        yield b"unreachable"

    task = asyncio.create_task(
        store.save_file_streaming(
            stalled_chunks(), "cancelled", "batch", max_bytes=1024
        )
    )
    await asyncio.wait_for(waiting.wait(), timeout=1)
    assert len(list((tmp_path / "files").glob("*.tmp"))) == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert list((tmp_path / "files").iterdir()) == []


async def test_streaming_save_publishes_empty_file(tmp_path):
    store = BatchStore(tmp_path)

    file = await store.save_file_streaming(
        _chunks(), "empty.jsonl", "batch", max_bytes=0
    )

    assert file.bytes == 0
    assert store.read_file_content(file.id) == b""
    assert list((tmp_path / "files").glob("*.tmp")) == []
