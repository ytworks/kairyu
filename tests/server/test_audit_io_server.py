from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import IO

import httpx

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server import tenancy as tenancy_module
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.tenancy import UsageLedger


class _SlowFile:
    def __init__(
        self,
        handle: IO[str],
        write_started: threading.Event,
        release_write: threading.Event,
    ) -> None:
        self._handle = handle
        self._write_started = write_started
        self._release_write = release_write

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def write(self, value: str) -> int:
        self._write_started.set()
        assert self._release_write.wait(timeout=5)
        return self._handle.write(value)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def test_usage_ledger_defers_json_encoding_to_writer_thread(
    tmp_path,
    monkeypatch,
) -> None:
    producer_thread = threading.get_ident()
    encoder_threads: list[int] = []
    original_dumps = tenancy_module.json.dumps

    def tracked_dumps(value, *args, **kwargs):
        encoder_threads.append(threading.get_ident())
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(tenancy_module.json, "dumps", tracked_dumps)
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    ledger.record(
        "tenant-a",
        "model-a",
        prompt_tokens=7,
        completion_tokens=3,
        cached_tokens=2,
    )
    ledger.close()

    assert len(encoder_threads) == 1
    assert encoder_threads[0] != producer_thread
    payload = tenancy_module.json.loads(
        (tmp_path / "usage.jsonl").read_text(encoding="utf-8")
    )
    timestamp = payload.pop("ts")
    assert type(timestamp) is float
    assert payload == {
        "tenant": "tenant-a",
        "model": "model-a",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cached_tokens": 2,
        "uncached_tokens": 5,
    }


async def test_slow_ledger_filesystem_does_not_delay_stream_completion(tmp_path):
    write_started = threading.Event()
    release_write = threading.Event()

    def slow_open(path: Path, mode: str, *, encoding: str) -> IO[str]:
        handle = open(path, mode, encoding=encoding)  # noqa: SIM115
        return _SlowFile(  # type: ignore[return-value]
            handle,
            write_started,
            release_write,
        )

    app = create_app({"m": MockBackend()})
    ledger = UsageLedger(tmp_path / "usage.jsonl", _open_file=slow_open)
    app.state.usage_ledger = ledger
    transport = httpx.ASGITransport(app=app)

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await asyncio.wait_for(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                ),
                timeout=1,
            )

        assert response.status_code == 200
        assert "data: [DONE]" in response.text
        assert await asyncio.to_thread(write_started.wait, 1)
    finally:
        release_write.set()
        await asyncio.to_thread(ledger.close)

    assert ledger.totals()["default"]["requests"] == 1
