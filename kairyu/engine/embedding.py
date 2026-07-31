"""Embedding engine contracts and production CPU inference backends."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import math
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

from kairyu.engine.backend import EngineReadiness

_PROVENANCE_FILENAME = "MODEL_PROVENANCE.json"


@dataclass(frozen=True)
class EmbeddingResult:
    """One ordered embedding batch plus its authoritative input usage."""

    vectors: tuple[tuple[float, ...], ...]
    prompt_tokens: int
    usage_exact: bool


class EmbeddingCapacityError(RuntimeError):
    """The backend has no execution slot and did not accept the request."""


class EmbeddingBackend(Protocol):
    dimensions: int
    reports_exact_usage: bool

    def embed(self, texts: list[str]) -> Awaitable[EmbeddingResult]: ...

    async def startup(self) -> None: ...

    def readiness(self) -> EngineReadiness: ...

    async def shutdown(self) -> None: ...


class MockEmbeddingBackend:
    """Deterministic hash-based unit vectors (CPU tests, wire-format truth)."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions
        self.reports_exact_usage = False

    async def startup(self) -> None:
        return None

    def readiness(self) -> EngineReadiness:
        return EngineReadiness(True)

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            values = []
            counter = 0
            while len(values) < self.dimensions:
                digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
                values.extend(b / 255.0 - 0.5 for b in digest)
                counter += 1
            norm = sum(v * v for v in values[: self.dimensions]) ** 0.5 or 1.0
            vectors.append(
                tuple(v / norm for v in values[: self.dimensions])
            )
        prompt_tokens = sum(len(text.split()) for text in texts)
        return EmbeddingResult(
            vectors=tuple(vectors),
            prompt_tokens=prompt_tokens,
            # The mock's whitespace counter preserves the historical wire
            # contract, but it is not a production tokenizer.
            usage_exact=False,
        )

    async def shutdown(self) -> None:
        return None


class FastEmbedEmbeddingBackend:
    """Pinned, offline FastEmbed/ONNX encoder with owned lifecycle.

    Construction performs no file or model I/O. ``startup`` validates an
    immutable, prefetched model bundle, creates one long-lived ONNX session and
    warms it before readiness can become true. A dedicated executor bounds
    concurrent synchronous ORT calls and lets shutdown drain them explicitly.
    """

    def __init__(
        self,
        *,
        dimensions: int,
        model: str,
        model_path: str | Path,
        revision: str,
        model_sha256: str,
        provenance_sha256: str,
        batch_size: int = 64,
        threads: int = 2,
        max_concurrency: int = 2,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if threads < 1:
            raise ValueError("threads must be >= 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.dimensions = dimensions
        self.reports_exact_usage = True
        self.model_name = model
        self.model_path = Path(model_path)
        self.revision = revision
        self.model_sha256 = model_sha256
        self.provenance_sha256 = provenance_sha256
        self.batch_size = batch_size
        self.threads = threads
        self.max_concurrency = max_concurrency

        self._state = "created"
        self._failure_type: str | None = None
        self._model = None
        self._executor: ThreadPoolExecutor | None = None
        self._work: set[asyncio.Future[EmbeddingResult]] = set()
        self._lifecycle_lock = asyncio.Lock()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validated_model_description(self, text_embedding_type):
        descriptions = text_embedding_type.list_supported_models()
        description = next(
            (
                item
                for item in descriptions
                if item.get("model", "").lower() == self.model_name.lower()
            ),
            None,
        )
        if description is None:
            raise ValueError(f"unsupported FastEmbed model {self.model_name!r}")
        if int(description.get("dim", -1)) != self.dimensions:
            raise ValueError(
                f"configured dimensions {self.dimensions} do not match "
                f"FastEmbed catalog dimensions {description.get('dim')}"
            )
        return description

    def _validate_bundle(self, description: dict) -> Path:
        root = self.model_path.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("embedding model_path must be a directory")
        provenance_path = root / _PROVENANCE_FILENAME
        if self._sha256(provenance_path) != self.provenance_sha256:
            raise ValueError("embedding model provenance SHA-256 mismatch")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        source = description.get("sources", {}).get("hf")
        if provenance.get("schema_version") != 1:
            raise ValueError("unsupported embedding model provenance schema")
        if str(provenance.get("repository", "")).lower() != str(source).lower():
            raise ValueError("embedding model provenance repository mismatch")
        if provenance.get("revision") != self.revision:
            raise ValueError("embedding model provenance revision mismatch")

        files = provenance.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("embedding model provenance has no files")
        validated_hashes: dict[str, str] = {}
        for relative, record in files.items():
            if (
                not isinstance(relative, str)
                or not relative
                or not isinstance(record, dict)
            ):
                raise ValueError("embedding model provenance file entry is invalid")
            candidate = (root / relative).resolve(strict=True)
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise ValueError("embedding model provenance file escapes model_path")
            expected_size = record.get("size")
            expected_sha256 = record.get("sha256")
            if (
                type(expected_size) is not int
                or expected_size < 0
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
            ):
                raise ValueError("embedding model provenance file metadata is invalid")
            if candidate.stat().st_size != expected_size:
                raise ValueError("embedding model provenance file size mismatch")
            actual_sha256 = self._sha256(candidate)
            if actual_sha256 != expected_sha256:
                raise ValueError("embedding model provenance file hash mismatch")
            validated_hashes[relative] = actual_sha256

        model_file = description.get("model_file")
        if not isinstance(model_file, str) or not model_file:
            raise ValueError("FastEmbed catalog did not declare a model file")
        resolved_model_file = (root / model_file).resolve(strict=True)
        if not resolved_model_file.is_relative_to(root):
            raise ValueError("embedding model file escapes model_path")
        actual_sha256 = validated_hashes.get(model_file)
        if actual_sha256 is None:
            raise ValueError("embedding model file is absent from provenance")
        if actual_sha256 != self.model_sha256:
            raise ValueError("embedding model SHA-256 mismatch")
        return root

    def _validate_vectors(
        self,
        vectors,
        *,
        expected_count: int,
    ) -> tuple[tuple[float, ...], ...]:
        converted = tuple(
            tuple(float(value) for value in vector)
            for vector in vectors
        )
        if len(converted) != expected_count:
            raise RuntimeError(
                f"embedding backend returned {len(converted)} vectors "
                f"for {expected_count} inputs"
            )
        for vector in converted:
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"embedding backend returned dimension {len(vector)}, "
                    f"expected {self.dimensions}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError("embedding backend returned a non-finite value")
        return converted

    def _load_sync(self):
        from fastembed import TextEmbedding

        description = self._validated_model_description(TextEmbedding)
        root = self._validate_bundle(description)
        model = TextEmbedding(
            model_name=self.model_name,
            specific_model_path=str(root),
            local_files_only=True,
            providers=["CPUExecutionProvider"],
            threads=self.threads,
        )
        if int(model.embedding_size) != self.dimensions:
            raise ValueError(
                f"loaded embedding dimensions {model.embedding_size} "
                f"do not match configured dimensions {self.dimensions}"
            )
        warmup = self._validate_vectors(
            list(
                model.embed(
                    ["kairyu embedding readiness"],
                    batch_size=1,
                    parallel=None,
                )
            ),
            expected_count=1,
        )
        norm = math.sqrt(sum(value * value for value in warmup[0]))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("embedding warmup vector is not L2-normalized")
        return model

    def _embed_sync(self, model, texts: tuple[str, ...]) -> EmbeddingResult:
        prompt_tokens = int(
            model.token_count(texts, batch_size=self.batch_size)
        )
        vectors = self._validate_vectors(
            list(
                model.embed(
                    texts,
                    batch_size=self.batch_size,
                    parallel=None,
                )
            ),
            expected_count=len(texts),
        )
        return EmbeddingResult(
            vectors=vectors,
            prompt_tokens=prompt_tokens,
            usage_exact=True,
        )

    @staticmethod
    async def _drain_ignoring_cancellation(future: asyncio.Future) -> None:
        """Wait for owned blocking work even if the caller is cancelled again."""

        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            future.exception()
        except asyncio.CancelledError:
            pass

    @classmethod
    async def _await_work(
        cls,
        future: asyncio.Future[EmbeddingResult],
    ) -> EmbeddingResult:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await cls._drain_ignoring_cancellation(future)
            raise

    async def _shutdown_executor(self, executor: ThreadPoolExecutor) -> None:
        shutdown = asyncio.create_task(
            asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=False,
            )
        )
        try:
            await asyncio.shield(shutdown)
        except asyncio.CancelledError:
            await self._drain_ignoring_cancellation(shutdown)
            raise

    async def startup(self) -> None:
        async with self._lifecycle_lock:
            if self._state == "ready":
                return
            if self._state == "failed":
                raise RuntimeError("embedding backend startup previously failed")
            if self._state not in {"created", "stopped"}:
                raise RuntimeError(f"embedding backend cannot start from {self._state}")

            self._state = "starting"
            executor = ThreadPoolExecutor(
                max_workers=self.max_concurrency,
                thread_name_prefix="kairyu-embedding",
            )
            self._executor = executor
            load: asyncio.Future | None = None
            try:
                loop = asyncio.get_running_loop()
                load = loop.run_in_executor(executor, self._load_sync)
                model = await asyncio.shield(load)
            except asyncio.CancelledError:
                if load is not None:
                    await self._drain_ignoring_cancellation(load)
                executor.shutdown(wait=True, cancel_futures=False)
                self._executor = None
                self._model = None
                self._failure_type = None
                self._state = "stopped"
                raise
            except Exception as error:
                self._failure_type = type(error).__name__
                self._state = "failed"
                executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
                raise
            self._model = model
            self._failure_type = None
            self._state = "ready"

    def readiness(self) -> EngineReadiness:
        if self._state == "ready":
            return EngineReadiness(True)
        if self._state == "failed":
            failure = self._failure_type or "Exception"
            return EngineReadiness(
                False,
                f"embedding startup failed ({failure})",
                fatal=True,
            )
        return EngineReadiness(False, f"embedding backend is {self._state}")

    def embed(self, texts: list[str]) -> Awaitable[EmbeddingResult]:
        if self._state != "ready" or self._executor is None or self._model is None:
            raise RuntimeError("embedding backend is not ready")
        if len(self._work) >= self.max_concurrency:
            raise EmbeddingCapacityError(
                f"embedding backend is at max concurrency ({self.max_concurrency})"
            )
        frozen_texts = tuple(texts)
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor,
            partial(self._embed_sync, self._model, frozen_texts),
        )
        self._work.add(future)
        future.add_done_callback(self._work.discard)
        return self._await_work(future)

    async def shutdown(self) -> None:
        async with self._lifecycle_lock:
            if self._state in {"created", "stopped"}:
                self._state = "stopped"
                return
            executor = self._executor
            self._state = "stopping"
            cancelled: asyncio.CancelledError | None = None
            try:
                if executor is not None:
                    await self._shutdown_executor(executor)
            except asyncio.CancelledError as error:
                cancelled = error
            finally:
                self._work.clear()
                self._executor = None
                self._model = None
                gc.collect()
                self._state = "stopped"
            if cancelled is not None:
                raise cancelled
