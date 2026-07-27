#!/usr/bin/env python3
"""Reproducible request-path audit I/O A/B benchmark (issue #213).

The injected delay represents a slow filesystem call. The legacy path performs
one write+flush per record on the producer thread; the bounded path measures
producer admission separately from lifecycle drain and batches filesystem work.

Run:
    uv run python bench/audit_io_bench.py --records 10000 --delay-ms 0.1
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import IO

from kairyu.audit_io import BoundedJsonlWriter


class _DelayedFile:
    def __init__(self, handle: IO[str], delay_seconds: float) -> None:
        self._handle = handle
        self._delay_seconds = delay_seconds
        self.write_calls = 0
        self.flush_calls = 0

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def write(self, value: str) -> int:
        self.write_calls += 1
        time.sleep(self._delay_seconds)
        return self._handle.write(value)

    def flush(self) -> None:
        self.flush_calls += 1
        time.sleep(self._delay_seconds)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class _DelayedOpen:
    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds
        self.handles: list[_DelayedFile] = []

    def __call__(
        self,
        path: Path,
        mode: str,
        *,
        encoding: str,
    ) -> IO[str]:
        handle = _DelayedFile(
            open(path, mode, encoding=encoding),  # noqa: SIM115
            self._delay_seconds,
        )
        self.handles.append(handle)
        return handle  # type: ignore[return-value]


def _legacy(path: Path, records: list[str], delay_seconds: float) -> dict:
    delayed = _DelayedOpen(delay_seconds)
    handle = delayed(path, "a", encoding="utf-8")
    started = time.perf_counter()
    for record in records:
        handle.write(record + "\n")
        handle.flush()
    elapsed = time.perf_counter() - started
    handle.close()
    return {
        "producer_seconds": elapsed,
        "flush_complete_seconds": elapsed,
        "write_calls": delayed.handles[0].write_calls,
        "flush_calls": delayed.handles[0].flush_calls,
    }


def _bounded(
    path: Path,
    records: list[str],
    delay_seconds: float,
    batch_size: int,
) -> dict:
    delayed = _DelayedOpen(delay_seconds)
    writer = BoundedJsonlWriter(
        path,
        max_pending=len(records),
        batch_size=batch_size,
        open_file=delayed,
    )
    started = time.perf_counter()
    for record in records:
        writer.append(record)
    admitted = time.perf_counter()
    writer.close()
    flush_complete = time.perf_counter()
    return {
        "producer_seconds": admitted - started,
        "flush_complete_seconds": flush_complete - started,
        "write_calls": delayed.handles[0].write_calls,
        "flush_calls": delayed.handles[0].flush_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--delay-ms", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.records < 1 or args.batch_size < 1:
        parser.error("records and batch-size must be positive")
    if args.delay_ms < 0:
        parser.error("delay-ms must be non-negative")

    records = [
        json.dumps({"request": index, "prompt_tokens": 32})
        for index in range(args.records)
    ]
    with tempfile.TemporaryDirectory(prefix="kairyu-audit-io-") as directory:
        root = Path(directory)
        legacy = _legacy(
            root / "legacy.jsonl",
            records,
            args.delay_ms / 1_000,
        )
        bounded = _bounded(
            root / "bounded.jsonl",
            records,
            args.delay_ms / 1_000,
            args.batch_size,
        )
        assert (root / "legacy.jsonl").read_text() == (
            root / "bounded.jsonl"
        ).read_text()

    result = {
        "records": args.records,
        "delay_ms": args.delay_ms,
        "batch_size": args.batch_size,
        "legacy": legacy,
        "bounded": bounded,
        "producer_speedup": (
            legacy["producer_seconds"] / bounded["producer_seconds"]
        ),
        "flush_complete_speedup": (
            legacy["flush_complete_seconds"]
            / bounded["flush_complete_seconds"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
