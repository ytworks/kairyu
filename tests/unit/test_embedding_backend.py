"""Production embedding backend lifecycle, integrity, and concurrency."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
import threading
import time
import types

import pytest

from kairyu.engine.embedding import (
    EmbeddingCapacityError,
    FastEmbedEmbeddingBackend,
)

REVISION = "a" * 40


def _model_bundle(tmp_path, *, model_bytes: bytes = b"onnx-model"):
    model_path = tmp_path / "model"
    model_path.mkdir()
    model_file = model_path / "model.onnx"
    model_file.write_bytes(model_bytes)
    tokenizer_file = model_path / "tokenizer.json"
    tokenizer_file.write_bytes(b'{"tokenizer":"pinned"}')
    sha256 = hashlib.sha256(model_bytes).hexdigest()
    payload = {
        "schema_version": 1,
        "repository": "Qdrant/test-onnx",
        "revision": REVISION,
        "license": "apache-2.0",
        "files": {
            "model.onnx": {
                "sha256": sha256,
                "size": len(model_bytes),
            },
            "tokenizer.json": {
                "sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
                "size": tokenizer_file.stat().st_size,
            },
        },
    }
    provenance = model_path / "MODEL_PROVENANCE.json"
    provenance.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance_sha256 = hashlib.sha256(provenance.read_bytes()).hexdigest()
    return model_path, sha256, provenance_sha256


class _FakeTextEmbedding:
    embedding_size = 4
    constructor_calls = []
    thread_names = []
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    delay_s = 0.0

    @classmethod
    def list_supported_models(cls):
        return [
            {
                "model": "test/model",
                "sources": {"hf": "Qdrant/test-onnx"},
                "model_file": "model.onnx",
                "dim": 4,
            }
        ]

    def __init__(self, **kwargs):
        type(self).constructor_calls.append(kwargs)

    def token_count(self, texts, *, batch_size):
        type(self).thread_names.append(threading.current_thread().name)
        return sum(len(text.split()) + 2 for text in texts)

    def embed(self, texts, *, batch_size, parallel):
        assert parallel is None
        with type(self).active_lock:
            type(self).active += 1
            type(self).max_active = max(
                type(self).max_active,
                type(self).active,
            )
        try:
            if type(self).delay_s:
                time.sleep(type(self).delay_s)
            for index, _text in enumerate(texts):
                vector = [0.0] * 4
                vector[index % 4] = 1.0
                yield vector
        finally:
            with type(self).active_lock:
                type(self).active -= 1


@pytest.fixture(autouse=True)
def _fake_fastembed(monkeypatch):
    _FakeTextEmbedding.constructor_calls = []
    _FakeTextEmbedding.thread_names = []
    _FakeTextEmbedding.active = 0
    _FakeTextEmbedding.max_active = 0
    _FakeTextEmbedding.delay_s = 0.0
    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding),
    )


def _backend(model_path, sha256, provenance_sha256, **overrides):
    return FastEmbedEmbeddingBackend(
        dimensions=4,
        model="test/model",
        model_path=model_path,
        revision=REVISION,
        model_sha256=sha256,
        provenance_sha256=provenance_sha256,
        batch_size=2,
        threads=3,
        max_concurrency=1,
        **overrides,
    )


async def test_backend_loads_offline_warms_and_reports_exact_ordered_usage(tmp_path):
    model_path, sha256, provenance_sha256 = _model_bundle(tmp_path)
    backend = _backend(model_path, sha256, provenance_sha256)

    assert not backend.readiness().ready
    await backend.startup()
    assert backend.readiness().ready
    assert _FakeTextEmbedding.constructor_calls == [
        {
            "model_name": "test/model",
            "specific_model_path": str(model_path.resolve()),
            "local_files_only": True,
            "providers": ["CPUExecutionProvider"],
            "threads": 3,
        }
    ]

    result = await backend.embed(["two words", "three word input"])

    assert result.prompt_tokens == 9
    assert result.usage_exact is True
    assert result.vectors == (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
    )
    assert all(
        math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)
        for vector in result.vectors
    )
    assert _FakeTextEmbedding.thread_names
    assert all(
        name.startswith("kairyu-embedding")
        for name in _FakeTextEmbedding.thread_names
    )

    await backend.shutdown()
    assert backend.readiness().detail == "embedding backend is stopped"
    with pytest.raises(RuntimeError, match="not ready"):
        await backend.embed(["after shutdown"])
    await backend.shutdown()


async def test_backend_executor_enforces_configured_concurrency(tmp_path):
    model_path, sha256, provenance_sha256 = _model_bundle(tmp_path)
    backend = _backend(model_path, sha256, provenance_sha256)
    await backend.startup()
    _FakeTextEmbedding.delay_s = 0.03

    accepted = backend.embed(["accepted"])
    with pytest.raises(EmbeddingCapacityError, match="max concurrency"):
        backend.embed(["rejected"])
    await accepted

    assert _FakeTextEmbedding.max_active == 1
    await backend.shutdown()


async def test_integrity_failure_is_fatal_and_never_becomes_ready(tmp_path):
    model_path, _sha256, provenance_sha256 = _model_bundle(tmp_path)
    backend = _backend(model_path, "0" * 64, provenance_sha256)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        await backend.startup()

    status = backend.readiness()
    assert status.ready is False
    assert status.fatal is True
    assert status.detail == "embedding startup failed (ValueError)"
    assert _FakeTextEmbedding.constructor_calls == []
    await backend.shutdown()


async def test_catalog_dimension_mismatch_fails_before_model_construction(tmp_path):
    model_path, sha256, provenance_sha256 = _model_bundle(tmp_path)
    backend = FastEmbedEmbeddingBackend(
        dimensions=5,
        model="test/model",
        model_path=model_path,
        revision=REVISION,
        model_sha256=sha256,
        provenance_sha256=provenance_sha256,
    )

    with pytest.raises(ValueError, match="catalog dimensions"):
        await backend.startup()

    assert _FakeTextEmbedding.constructor_calls == []


async def test_tokenizer_mutation_fails_bundle_integrity_before_construction(tmp_path):
    model_path, sha256, provenance_sha256 = _model_bundle(tmp_path)
    (model_path / "tokenizer.json").write_text(
        '{"tokenizer":"mutated"}',
        encoding="utf-8",
    )
    backend = _backend(model_path, sha256, provenance_sha256)

    with pytest.raises(ValueError, match="file (size|hash) mismatch"):
        await backend.startup()

    assert backend.readiness().fatal
    assert _FakeTextEmbedding.constructor_calls == []


async def test_startup_cancellation_drains_owned_executor(tmp_path, monkeypatch):
    model_path, sha256, provenance_sha256 = _model_bundle(tmp_path)
    backend = _backend(model_path, sha256, provenance_sha256)
    started = threading.Event()
    release = threading.Event()
    original_load = backend._load_sync

    def blocked_load():
        started.set()
        assert release.wait(timeout=5)
        return original_load()

    monkeypatch.setattr(backend, "_load_sync", blocked_load)
    startup = asyncio.create_task(backend.startup())
    while not started.is_set():
        await asyncio.sleep(0)

    startup.cancel()
    await asyncio.sleep(0)
    assert not startup.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await startup

    await backend.shutdown()
    assert backend.readiness().detail == "embedding backend is stopped"
    assert not any(
        thread.name.startswith("kairyu-embedding")
        for thread in threading.enumerate()
    )


async def test_cancelled_shutdown_still_drains_work_and_stops(tmp_path, monkeypatch):
    model_path, sha256, provenance_sha256 = _model_bundle(tmp_path)
    backend = _backend(model_path, sha256, provenance_sha256)
    await backend.startup()
    started = threading.Event()
    release = threading.Event()
    original_embed = backend._embed_sync

    def blocked_embed(model, texts):
        started.set()
        assert release.wait(timeout=5)
        return original_embed(model, texts)

    monkeypatch.setattr(backend, "_embed_sync", blocked_embed)
    inference = asyncio.create_task(backend.embed(["accepted"]))
    while not started.is_set():
        await asyncio.sleep(0)

    shutdown = asyncio.create_task(backend.shutdown())
    while backend.readiness().detail != "embedding backend is stopping":
        await asyncio.sleep(0)
    shutdown.cancel()
    await asyncio.sleep(0)
    assert not shutdown.done()
    release.set()

    await inference
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    assert backend.readiness().detail == "embedding backend is stopped"
    assert not any(
        thread.name.startswith("kairyu-embedding")
        for thread in threading.enumerate()
    )
