from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO

import pytest

from kairyu.audit_io import (
    AuditQueueFull,
    AuditWriteError,
    BoundedJsonlWriter,
)


class _WrappedFile:
    def __init__(
        self,
        handle: IO[str],
        *,
        write_started: threading.Event | None = None,
        release_write: threading.Event | None = None,
        fail_write: bool = False,
    ) -> None:
        self._handle = handle
        self._write_started = write_started
        self._release_write = release_write
        self._fail_write = fail_write
        self.write_calls = 0

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def write(self, value: str) -> int:
        self.write_calls += 1
        if self._write_started is not None:
            self._write_started.set()
        if self._release_write is not None:
            assert self._release_write.wait(timeout=5)
        if self._fail_write:
            raise OSError("simulated disk failure")
        return self._handle.write(value)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class _OpenFile:
    def __init__(
        self,
        *,
        write_started: threading.Event | None = None,
        release_write: threading.Event | None = None,
        fail_write: bool = False,
    ) -> None:
        self._write_started = write_started
        self._release_write = release_write
        self._fail_write = fail_write
        self.handles: list[_WrappedFile] = []

    def __call__(
        self,
        path: Path,
        mode: str,
        *,
        encoding: str,
    ) -> IO[str]:
        wrapped = _WrappedFile(
            open(path, mode, encoding=encoding),  # noqa: SIM115
            write_started=self._write_started,
            release_write=self._release_write,
            fail_write=self._fail_write,
        )
        self.handles.append(wrapped)
        return wrapped  # type: ignore[return-value]


def test_slow_filesystem_does_not_block_record_admission(tmp_path):
    write_started = threading.Event()
    release_write = threading.Event()
    writer = BoundedJsonlWriter(
        tmp_path / "audit.jsonl",
        max_pending=16,
        batch_size=4,
        open_file=_OpenFile(
            write_started=write_started,
            release_write=release_write,
        ),
    )

    writer.append('{"record":0}')
    assert write_started.wait(timeout=5)
    started = time.perf_counter()
    for index in range(1, 9):
        writer.append(f'{{"record":{index}}}')
    admission_seconds = time.perf_counter() - started

    assert admission_seconds < 0.05
    release_write.set()
    writer.close()
    assert (tmp_path / "audit.jsonl").read_text().splitlines() == [
        f'{{"record":{index}}}' for index in range(9)
    ]


def test_full_queue_fails_immediately_without_silent_drop(tmp_path):
    write_started = threading.Event()
    release_write = threading.Event()
    writer = BoundedJsonlWriter(
        tmp_path / "audit.jsonl",
        max_pending=1,
        batch_size=1,
        open_file=_OpenFile(
            write_started=write_started,
            release_write=release_write,
        ),
    )
    writer.append('{"record":0}')
    assert write_started.wait(timeout=5)
    writer.append('{"record":1}')

    started = time.perf_counter()
    with pytest.raises(AuditQueueFull, match="queue is full"):
        writer.append('{"record":2}')
    assert time.perf_counter() - started < 0.05

    release_write.set()
    writer.close()
    assert (tmp_path / "audit.jsonl").read_text().splitlines() == [
        '{"record":0}',
        '{"record":1}',
    ]


def test_close_drains_batches_and_writer_can_restart(tmp_path):
    write_started = threading.Event()
    release_write = threading.Event()
    open_file = _OpenFile(
        write_started=write_started,
        release_write=release_write,
    )
    path = tmp_path / "audit.jsonl"
    writer = BoundedJsonlWriter(
        path,
        max_pending=16,
        batch_size=4,
        open_file=open_file,
    )
    writer.append('{"record":0}')
    assert write_started.wait(timeout=5)
    for index in range(1, 9):
        writer.append(f'{{"record":{index}}}')
    release_write.set()
    writer.close()

    assert open_file.handles[0].closed
    assert open_file.handles[0].write_calls == 3
    writer.append('{"record":9}')
    writer.flush()
    assert len(open_file.handles) == 2
    assert path.read_text().splitlines()[-1] == '{"record":9}'
    writer.close()
    assert path.read_text().splitlines() == [
        f'{{"record":{index}}}' for index in range(10)
    ]


def test_deferred_encoding_batches_are_bounded_by_utf8_bytes(tmp_path):
    open_file = _OpenFile()
    writer = BoundedJsonlWriter(
        tmp_path / "audit.jsonl",
        batch_size=128,
        max_batch_bytes=16,
        open_file=open_file,
    )

    for index in range(3):
        writer.append_deferred(
            lambda index=index: f'{{"record":{index},"padding":"xxxxxxxx"}}'
        )
    writer.close()

    assert open_file.handles[0].write_calls == 3


def test_flush_surfaces_sticky_disk_error(tmp_path):
    writer = BoundedJsonlWriter(
        tmp_path / "audit.jsonl",
        open_file=_OpenFile(fail_write=True),
    )
    writer.append('{"record":0}')

    with pytest.raises(AuditWriteError, match="OSError"):
        writer.flush()
    with pytest.raises(AuditWriteError, match="OSError"):
        writer.append('{"record":1}')
    with pytest.raises(AuditWriteError, match="OSError"):
        writer.close()


def test_restart_separates_new_rows_from_a_truncated_tail(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"truncated":', encoding="utf-8")
    writer = BoundedJsonlWriter(path)

    writer.append('{"record":1}')
    writer.close()

    assert path.read_text().splitlines() == [
        '{"truncated":',
        '{"record":1}',
    ]


def test_concurrent_producers_preserve_complete_jsonl_records(tmp_path):
    path = tmp_path / "audit.jsonl"
    writer = BoundedJsonlWriter(path, max_pending=1_000, batch_size=32)

    def append_range(producer: int) -> None:
        for index in range(100):
            writer.append(f'{{"producer":{producer},"record":{index}}}')

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append_range, range(4)))
    writer.close()

    records = set(path.read_text().splitlines())
    assert len(records) == 400
    assert records == {
        f'{{"producer":{producer},"record":{index}}}'
        for producer in range(4)
        for index in range(100)
    }


def test_deferred_renderer_runs_on_writer_thread(tmp_path):
    path = tmp_path / "audit.jsonl"
    writer = BoundedJsonlWriter(path)
    producer_thread = threading.get_ident()
    renderer_threads: list[int] = []

    def render() -> str:
        renderer_threads.append(threading.get_ident())
        return '{"record":0}'

    writer.append_deferred(render)
    writer.close()

    assert renderer_threads
    assert renderer_threads[0] != producer_thread
    assert path.read_text().splitlines() == ['{"record":0}']


@pytest.mark.parametrize(
    "render",
    [
        lambda: 1,
        lambda: '{"record":0}\n',
        lambda: (_ for _ in ()).throw(ValueError("encode failed")),
    ],
)
def test_deferred_renderer_failure_is_sticky(tmp_path, render):
    writer = BoundedJsonlWriter(tmp_path / "audit.jsonl")
    writer.append_deferred(render)

    with pytest.raises(AuditWriteError, match="encode failed"):
        writer.flush()
    with pytest.raises(AuditWriteError, match="encode failed"):
        writer.append('{"record":1}')
    with pytest.raises(AuditWriteError, match="encode failed"):
        writer.close()
