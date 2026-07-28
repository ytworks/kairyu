"""Bounded lifecycle-owned JSONL writing outside request/event-loop threads."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

_DEFAULT_MAX_PENDING = 4_096
_DEFAULT_BATCH_SIZE = 128
_DEFAULT_MAX_BATCH_BYTES = 64 * 1_024


class AuditQueueFull(RuntimeError):
    """The bounded writer cannot accept another record without blocking."""


class AuditWriteError(RuntimeError):
    """A background append/flush/close operation failed."""


@dataclass
class _Barrier:
    stop: bool = False
    event: threading.Event = field(default_factory=threading.Event)
    error: AuditWriteError | None = None


@dataclass(frozen=True)
class _DeferredLine:
    render: Callable[[], str]


_WorkItem = str | _DeferredLine | _Barrier


class BoundedJsonlWriter:
    """Single background append owner with bounded admission and batch flush.

    ``append`` never waits for filesystem work. A full queue raises
    ``AuditQueueFull`` immediately, so records are never silently dropped and a
    request/event-loop thread never becomes the disk backpressure worker.
    ``flush`` is an ordered reader-visibility barrier (buffer flush, not
    ``fsync``), and ``close`` drains every accepted record before closing the
    handle. Background disk failures are sticky for the current lifecycle and
    surface through append/flush/close.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_pending: int = _DEFAULT_MAX_PENDING,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
        open_file: Callable[..., IO[str]] = open,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_batch_bytes < 1:
            raise ValueError("max_batch_bytes must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_pending = max_pending
        self.batch_size = batch_size
        self.max_batch_bytes = max_batch_bytes
        self._open_file = open_file
        self._lock = threading.Lock()
        self._queue: queue.Queue[_WorkItem] | None = None
        self._thread: threading.Thread | None = None
        self._closing = False
        self._error: AuditWriteError | None = None
        self._handle: IO[str] | None = None

    @property
    def handle(self) -> IO[str] | None:
        """Current or most recently closed append handle (diagnostics/tests)."""
        return self._handle

    def _start_locked(self) -> queue.Queue[_WorkItem]:
        thread = self._thread
        if thread is not None and thread.is_alive():
            assert self._queue is not None
            return self._queue
        self._error = None
        work: queue.Queue[_WorkItem] = queue.Queue(
            maxsize=self.max_pending
        )
        thread = threading.Thread(
            target=self._run,
            args=(work,),
            name=f"kairyu-jsonl-{self.path.name}",
            daemon=True,
        )
        self._queue = work
        self._thread = thread
        thread.start()
        return work

    def _raise_error_locked(self) -> None:
        if self._error is not None:
            raise self._error

    def _admit(self, item: str | _DeferredLine) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("audit writer is closing")
            self._raise_error_locked()
            work = self._start_locked()
            try:
                work.put_nowait(item)
            except queue.Full as error:
                raise AuditQueueFull(
                    f"audit writer queue is full for {self.path}"
                ) from error

    def append(self, line: str) -> None:
        """Accept one newline-free JSON record without waiting for disk I/O."""
        if "\n" in line or "\r" in line:
            raise ValueError("JSONL record must not contain a newline")
        self._admit(line)

    def append_deferred(self, render: Callable[[], str]) -> None:
        """Queue CPU-side record encoding on the writer thread.

        The callable must return one complete newline-free record. Encoding
        failures become sticky ``AuditWriteError`` values surfaced by the next
        append, flush, or close, just like filesystem failures.
        """

        if not callable(render):
            raise TypeError("render must be callable")
        self._admit(_DeferredLine(render))

    def _barrier(self, *, stop: bool) -> _Barrier | None:
        with self._lock:
            if self._thread is None:
                self._raise_error_locked()
                return None
            if stop:
                if self._closing:
                    raise RuntimeError("audit writer is already closing")
                self._closing = True
            else:
                if self._closing:
                    raise RuntimeError("audit writer is closing")
                self._raise_error_locked()
            assert self._queue is not None
            barrier = _Barrier(stop=stop)
            # flush/close are lifecycle/read operations, not request admission.
            # They deliberately wait for queue capacity and all earlier writes.
            self._queue.put(barrier)
        barrier.event.wait()
        return barrier

    def flush(self) -> None:
        """Flush accepted rows to the OS; this is not power-loss durability."""
        barrier = self._barrier(stop=False)
        if barrier is not None and barrier.error is not None:
            raise barrier.error

    def close(self) -> None:
        """Drain accepted records, close the handle, and allow a later restart."""
        barrier = self._barrier(stop=True)
        if barrier is None:
            return
        thread = self._thread
        assert thread is not None
        thread.join()
        error = barrier.error
        with self._lock:
            self._thread = None
            self._queue = None
            self._closing = False
            self._error = None
        if error is not None:
            raise error

    def _set_error(self, operation: str, error: BaseException) -> None:
        if self._error is None:
            self._error = AuditWriteError(
                f"audit writer {operation} failed for {self.path}: "
                f"{type(error).__name__}"
            )

    def _open(self) -> IO[str]:
        needs_separator = False
        if self.path.is_file():
            with self.path.open("rb") as existing:
                existing.seek(0, 2)
                if existing.tell() > 0:
                    existing.seek(-1, 2)
                    needs_separator = existing.read(1) != b"\n"
        handle = self._open_file(
            self.path,
            "a",
            encoding="utf-8",
        )
        self._handle = handle
        # A prior crash can leave one truncated JSON tail. Preserve that
        # forensic row, but terminate it before accepting new complete rows.
        if needs_separator:
            handle.write("\n")
            handle.flush()
        return handle

    def _write_batch(
        self,
        handle: IO[str] | None,
        records: list[str],
    ) -> IO[str] | None:
        if self._error is not None:
            return handle
        try:
            if handle is None or handle.closed:
                handle = self._open()
            handle.write("".join(f"{record}\n" for record in records))
            handle.flush()
        except BaseException as error:
            self._set_error("append/flush", error)
        return handle

    def _finish_barrier(
        self,
        handle: IO[str] | None,
        barrier: _Barrier,
    ) -> bool:
        if self._error is None and handle is not None and not handle.closed:
            try:
                handle.flush()
            except BaseException as error:
                self._set_error("flush", error)
        if barrier.stop and handle is not None and not handle.closed:
            try:
                handle.close()
            except BaseException as error:
                self._set_error("close", error)
        barrier.error = self._error
        barrier.event.set()
        return barrier.stop

    def _render_line(self, item: str | _DeferredLine) -> str | None:
        if isinstance(item, str):
            return item
        try:
            line = item.render()
            if not isinstance(line, str):
                raise TypeError("deferred audit renderer must return str")
            if "\n" in line or "\r" in line:
                raise ValueError("JSONL record must not contain a newline")
        except BaseException as error:
            self._set_error("encode", error)
            return None
        return line

    @staticmethod
    def _encoded_size(line: str) -> int:
        return len(line.encode("utf-8")) + 1

    def _run(self, work: queue.Queue[_WorkItem]) -> None:
        handle: IO[str] | None = None
        deferred: _Barrier | None = None
        while True:
            item: _WorkItem
            if deferred is not None:
                item = deferred
                deferred = None
            else:
                item = work.get()
            if isinstance(item, _Barrier):
                if self._finish_barrier(handle, item):
                    return
                continue

            first = self._render_line(item)
            records = [] if first is None else [first]
            batch_bytes = 0 if first is None else self._encoded_size(first)
            while (
                len(records) < self.batch_size
                and batch_bytes < self.max_batch_bytes
            ):
                try:
                    following = work.get_nowait()
                except queue.Empty:
                    break
                if isinstance(following, _Barrier):
                    deferred = following
                    break
                rendered = self._render_line(following)
                if rendered is not None:
                    records.append(rendered)
                    batch_bytes += self._encoded_size(rendered)
            if records:
                handle = self._write_batch(handle, records)
