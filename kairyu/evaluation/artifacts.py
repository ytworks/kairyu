"""Safe, durable filesystem storage for benchmark run artifacts.

The store owns one directory per run under ``benchmark_runs``.  Callers only
provide a single-component run ID and a relative artifact path; absolute paths,
traversal components, and symlinks anywhere below (or leading to) the store
root are rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kairyu.evaluation.control_store import (
    LeaseConflictError,
    PublicationToken,
    validate_artifact_publication,
)
from kairyu.evaluation.safety import (
    SecretValueRegistry,
    ensure_secret_free_bytes,
    ensure_secret_free_serialized_json,
)
from kairyu.evaluation.schemas import thaw_json_value

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

PublicationGuardFactory = Callable[[PublicationToken], AbstractContextManager[None]]

RUN_ARTIFACT_FILES = (
    "manifest.json",
    "protocol.json",
    "events.jsonl",
    "predictions.jsonl",
    "item_results.jsonl",
    "metrics.json",
    "errors.jsonl",
    "usage.json",
    "references.json",
    "report.json",
    "report.md",
    "report.html",
)
RUN_ARTIFACT_DIRECTORIES = ("logs", "upstream")


class UnsafeArtifactPath(ValueError):
    """An artifact path could escape the store or traverse a symlink."""


class ArtifactConflictError(RuntimeError):
    """An artifact destination already contains different immutable bytes."""


@dataclass(frozen=True, slots=True)
class ArtifactWrite:
    """Metadata produced by one durable artifact publication."""

    run_id: str
    relative_path: str
    sha256: str
    size_bytes: int


class ArtifactStore:
    """Filesystem-backed artifact store rooted at ``benchmark_runs``.

    ``root`` is the benchmark-runs directory itself.  It defaults to the exact
    repository-local layout named in the evaluation contract.
    """

    def __init__(
        self,
        root: str | Path = "benchmark_runs",
        *,
        publication_guard: PublicationGuardFactory,
        secret_registry: SecretValueRegistry | None = None,
    ) -> None:
        if not callable(publication_guard):
            raise TypeError("publication_guard must be callable")
        candidate = Path(root).expanduser().absolute()
        _assert_no_symlink(candidate, include_missing_tail=False)
        candidate.mkdir(parents=True, exist_ok=True)
        _assert_directory(candidate)
        self._root = candidate
        self._publication_guard = publication_guard
        self._secret_registry = secret_registry

    @property
    def root(self) -> Path:
        return self._root

    def create_run(self, run_id: str) -> Path:
        """Create and return a run directory, including its fixed subdirectories."""
        self._scan_text(run_id)
        run_directory = self._run_path(run_id)
        _assert_directory(self._root)
        if run_directory.is_symlink():
            raise UnsafeArtifactPath(f"run directory is a symlink: {run_id!r}")
        run_directory.mkdir(mode=0o700, exist_ok=True)
        _assert_directory(run_directory)
        for name in RUN_ARTIFACT_DIRECTORIES:
            child = run_directory / name
            if child.is_symlink():
                raise UnsafeArtifactPath(f"run artifact directory is a symlink: {name!r}")
            child.mkdir(mode=0o700, exist_ok=True)
            _assert_directory(child)
        return run_directory

    def run_dir(self, run_id: str) -> Path:
        """Return an existing safe run directory."""
        self._scan_text(run_id)
        path = self._run_path(run_id)
        _assert_directory(self._root)
        _assert_directory(path)
        return path

    def path_for(self, run_id: str, relative_path: str | Path) -> Path:
        """Resolve an existing or prospective artifact path without following links."""
        run_directory = self.run_dir(run_id)
        self._scan_text(os.fspath(relative_path))
        parts = _validate_relative_path(relative_path)
        candidate = run_directory.joinpath(*parts)
        _assert_contained(run_directory, candidate)
        _assert_no_symlink(candidate, stop_at=run_directory, include_missing_tail=False)
        return candidate

    def write_bytes(
        self,
        run_id: str,
        relative_path: str | Path,
        content: bytes,
        *,
        publication_token: PublicationToken,
    ) -> ArtifactWrite:
        """Publish immutable bytes while holding an active publication fence."""
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        if not isinstance(publication_token, PublicationToken):
            raise TypeError("publication_token must be a PublicationToken")
        if publication_token.run_id != run_id:
            raise LeaseConflictError("artifact publication token does not match run")
        self._scan_text(run_id)
        self._scan_text(os.fspath(relative_path))
        normalized_relative_path = "/".join(_validate_relative_path(relative_path))
        validate_artifact_publication(publication_token, normalized_relative_path)
        ensure_secret_free_bytes(
            content,
            secret_registry=self._secret_registry,
        )
        content_sha256 = hashlib.sha256(content).hexdigest()

        with self._publication_guard(publication_token):
            destination = self.path_for(run_id, relative_path)
            self._create_safe_parents(self.run_dir(run_id), destination.parent)
            _reject_symlink_or_directory(destination)
            file_descriptor, raw_temporary_path = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary_path: Path | None = Path(raw_temporary_path)
            try:
                with os.fdopen(file_descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                _reject_symlink_or_directory(destination)
                _assert_directory(destination.parent)
                try:
                    os.link(
                        temporary_path,
                        destination,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    existing_sha256, existing_size = _hash_regular_file(destination)
                    if existing_sha256 != content_sha256 or existing_size != len(content):
                        raise ArtifactConflictError(
                            "artifact destination already contains different bytes"
                        ) from None
                temporary_path.unlink()
                temporary_path = None
                _fsync_directory(destination.parent)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        return ArtifactWrite(
            run_id=run_id,
            relative_path=normalized_relative_path,
            sha256=content_sha256,
            size_bytes=len(content),
        )

    def write_json(
        self,
        run_id: str,
        relative_path: str | Path,
        payload: Mapping[str, Any] | list[Any],
        *,
        publication_token: PublicationToken,
    ) -> ArtifactWrite:
        """Serialize canonical UTF-8 JSON and publish it atomically."""
        encoded = json.dumps(
            thaw_json_value(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        ensure_secret_free_serialized_json(
            encoded,
            secret_registry=self._secret_registry,
        )
        return self.write_bytes(
            run_id,
            relative_path,
            encoded,
            publication_token=publication_token,
        )

    def write_jsonl(
        self,
        run_id: str,
        relative_path: str | Path,
        records: Iterable[Mapping[str, Any]],
        *,
        publication_token: PublicationToken,
    ) -> ArtifactWrite:
        """Publish newline-delimited canonical JSON after scanning each record."""
        encoded_records: list[bytes] = []
        for record in records:
            encoded = json.dumps(
                thaw_json_value(record),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            ensure_secret_free_serialized_json(
                encoded,
                secret_registry=self._secret_registry,
            )
            encoded_records.append(encoded)
        content = b"\n".join(encoded_records)
        if content:
            content += b"\n"
        return self.write_bytes(
            run_id,
            relative_path,
            content,
            publication_token=publication_token,
        )

    def write_text(
        self,
        run_id: str,
        relative_path: str | Path,
        content: str,
        *,
        publication_token: PublicationToken,
    ) -> ArtifactWrite:
        """Publish UTF-8 report or log text only after credential scanning."""
        if not isinstance(content, str):
            raise TypeError("artifact text content must be a string")
        return self.write_bytes(
            run_id,
            relative_path,
            content.encode("utf-8"),
            publication_token=publication_token,
        )

    def read_bytes(self, run_id: str, relative_path: str | Path) -> bytes:
        """Read one artifact without following a symlink at the final component."""
        path = self.path_for(run_id, relative_path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(path, flags)
        except OSError as exc:
            if path.is_symlink():
                raise UnsafeArtifactPath(f"artifact is a symlink: {relative_path!s}") from exc
            raise
        try:
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise UnsafeArtifactPath(f"artifact is not a regular file: {relative_path!s}")
            with os.fdopen(file_descriptor, "rb") as handle:
                file_descriptor = -1
                return handle.read()
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)

    def read_json(self, run_id: str, relative_path: str | Path) -> Any:
        return json.loads(self.read_bytes(run_id, relative_path))

    def _scan_text(self, value: str) -> None:
        ensure_secret_free_bytes(
            value.encode("utf-8"),
            secret_registry=self._secret_registry,
        )

    def _run_path(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        path = self._root / run_id
        _assert_contained(self._root, path)
        return path

    @staticmethod
    def _create_safe_parents(run_directory: Path, parent: Path) -> None:
        _assert_contained(run_directory, parent)
        relative_parts = parent.relative_to(run_directory).parts
        current = run_directory
        for part in relative_parts:
            current /= part
            if current.is_symlink():
                raise UnsafeArtifactPath(f"artifact parent is a symlink: {current}")
            current.mkdir(mode=0o700, exist_ok=True)
            _assert_directory(current)


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _SAFE_COMPONENT.fullmatch(run_id):
        raise UnsafeArtifactPath("run_id must be one safe path component of 1-128 ASCII characters")


def _validate_relative_path(relative_path: str | Path) -> tuple[str, ...]:
    raw = os.fspath(relative_path)
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise UnsafeArtifactPath("artifact path must be a non-empty portable relative path")
    path = Path(raw)
    if path.is_absolute():
        raise UnsafeArtifactPath("absolute artifact paths are not allowed")
    if path.as_posix() != raw:
        raise UnsafeArtifactPath("artifact path must use canonical portable spelling")
    parts = path.parts
    if not parts or any(
        part in {"", ".", ".."} or not _SAFE_COMPONENT.fullmatch(part) for part in parts
    ):
        raise UnsafeArtifactPath(f"unsafe artifact path: {raw!r}")
    return parts


def _assert_contained(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise UnsafeArtifactPath(f"path escapes artifact root: {path}") from exc


def _assert_no_symlink(
    path: Path,
    *,
    stop_at: Path | None = None,
    include_missing_tail: bool = True,
) -> None:
    """Reject symlinks in existing components from ``stop_at`` (or anchor)."""
    absolute = path.absolute()
    if stop_at is None:
        current = Path(absolute.anchor)
        parts = absolute.parts[1:]
    else:
        stop = stop_at.absolute()
        _assert_contained(stop, absolute)
        _assert_no_symlink(stop, include_missing_tail=include_missing_tail)
        current = stop
        parts = absolute.relative_to(stop).parts

    for part in parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if include_missing_tail:
                continue
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise UnsafeArtifactPath(f"symlink is not allowed in artifact path: {current}")


def _assert_directory(path: Path) -> None:
    _assert_no_symlink(path, include_missing_tail=False)
    try:
        metadata = path.stat()
    except FileNotFoundError:
        raise
    if not stat.S_ISDIR(metadata.st_mode):
        raise UnsafeArtifactPath(f"artifact path is not a directory: {path}")


def _reject_symlink_or_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise UnsafeArtifactPath(f"artifact destination is a symlink: {path}")
    if stat.S_ISDIR(metadata.st_mode):
        raise UnsafeArtifactPath(f"artifact destination is a directory: {path}")


def _hash_regular_file(path: Path) -> tuple[str, int]:
    _reject_symlink_or_directory(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise UnsafeArtifactPath(f"artifact is a symlink: {path}") from exc
        raise
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsafeArtifactPath(f"artifact is not a regular file: {path}")
        with os.fdopen(file_descriptor, "rb") as handle:
            file_descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest(), metadata.st_size
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    file_descriptor = os.open(path, flags)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
