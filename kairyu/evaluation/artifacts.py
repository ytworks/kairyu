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
import secrets
import stat
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
DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

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


class ArtifactSizeLimitError(ValueError):
    """An artifact exceeded this store instance’s configured byte limit."""


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
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    ) -> None:
        if not callable(publication_guard):
            raise TypeError("publication_guard must be callable")
        if (
            isinstance(max_artifact_bytes, bool)
            or not isinstance(max_artifact_bytes, int)
            or max_artifact_bytes <= 0
        ):
            raise ValueError("max_artifact_bytes must be a positive integer")
        candidate = Path(root).expanduser().absolute()
        _assert_no_symlink(candidate, include_missing_tail=False)
        candidate.mkdir(parents=True, exist_ok=True)
        _assert_directory(candidate)
        self._root = candidate
        self._publication_guard = publication_guard
        self._secret_registry = secret_registry
        self._max_artifact_bytes = max_artifact_bytes

    @property
    def root(self) -> Path:
        return self._root

    @property
    def max_artifact_bytes(self) -> int:
        return self._max_artifact_bytes

    def create_run(self, run_id: str) -> Path:
        """Create and return a run directory, including its fixed subdirectories."""
        self._scan_text(run_id)
        run_directory = self._run_path(run_id)
        root_descriptor = _open_directory_path(self._root)
        try:
            run_descriptor = _open_or_create_directory_at(
                root_descriptor,
                run_id,
                display_path=run_directory,
            )
            try:
                for name in RUN_ARTIFACT_DIRECTORIES:
                    child_descriptor = _open_or_create_directory_at(
                        run_descriptor,
                        name,
                        display_path=run_directory / name,
                    )
                    os.close(child_descriptor)
            finally:
                os.close(run_descriptor)
        finally:
            os.close(root_descriptor)
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
        self._ensure_size(len(content))
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
            parts = _validate_relative_path(relative_path)
            parent_descriptor = _open_artifact_parent(
                self._root,
                run_id,
                parts,
                create=True,
            )
            temporary_name: str | None = None
            file_descriptor = -1
            try:
                _reject_symlink_or_directory_at(
                    parent_descriptor,
                    parts[-1],
                    display_path=destination,
                )
                temporary_name, file_descriptor = _create_temporary_file_at(
                    parent_descriptor,
                    destination_name=parts[-1],
                )
                with os.fdopen(file_descriptor, "wb") as handle:
                    file_descriptor = -1
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                _reject_symlink_or_directory_at(
                    parent_descriptor,
                    parts[-1],
                    display_path=destination,
                )
                try:
                    os.link(
                        temporary_name,
                        parts[-1],
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    existing_sha256, existing_size = _hash_regular_file_at(
                        parent_descriptor,
                        parts[-1],
                        display_path=destination,
                        max_bytes=self._max_artifact_bytes,
                    )
                    if existing_sha256 != content_sha256 or existing_size != len(content):
                        raise ArtifactConflictError(
                            "artifact destination already contains different bytes"
                        ) from None
                _unlink_at(parent_descriptor, temporary_name)
                temporary_name = None
                os.fsync(parent_descriptor)
            finally:
                if file_descriptor >= 0:
                    os.close(file_descriptor)
                if temporary_name is not None:
                    _unlink_at(parent_descriptor, temporary_name, missing_ok=True)
                os.close(parent_descriptor)

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
        self._ensure_size(len(encoded))
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
        encoded_size = 0
        for record in records:
            encoded = json.dumps(
                thaw_json_value(record),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            encoded_size += len(encoded) + 1
            self._ensure_size(encoded_size)
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
        """Read one artifact through a pinned, symlink-safe parent descriptor."""
        path = self.path_for(run_id, relative_path)
        parts = _validate_relative_path(relative_path)
        parent_descriptor = _open_artifact_parent(
            self._root,
            run_id,
            parts,
            create=False,
        )
        try:
            file_descriptor = _open_regular_file_at(
                parent_descriptor,
                parts[-1],
                display_path=path,
            )
            try:
                metadata = os.fstat(file_descriptor)
                self._ensure_size(metadata.st_size)
                with os.fdopen(file_descriptor, "rb") as handle:
                    file_descriptor = -1
                    content = handle.read(self._max_artifact_bytes + 1)
                self._ensure_size(len(content))
                return content
            finally:
                if file_descriptor >= 0:
                    os.close(file_descriptor)
        finally:
            os.close(parent_descriptor)

    def read_json(self, run_id: str, relative_path: str | Path) -> Any:
        return json.loads(self.read_bytes(run_id, relative_path))

    def _ensure_size(self, size_bytes: int) -> None:
        if size_bytes > self._max_artifact_bytes:
            raise ArtifactSizeLimitError(
                "artifact exceeds configured byte limit "
                f"({size_bytes} > {self._max_artifact_bytes})"
            )

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


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_path(path: Path) -> int:
    try:
        descriptor = os.open(path, _directory_open_flags())
    except OSError as exc:
        if path.is_symlink():
            raise UnsafeArtifactPath(f"artifact directory is a symlink: {path}") from exc
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise UnsafeArtifactPath(f"artifact path is not a directory: {path}")
    return descriptor


def _open_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
) -> int:
    try:
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        if _is_symlink_at(parent_descriptor, name):
            raise UnsafeArtifactPath(
                f"symlink is not allowed in artifact path: {display_path}"
            ) from exc
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise UnsafeArtifactPath(f"artifact path is not a directory: {display_path}")
    return descriptor


def _open_or_create_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    try:
        return _open_directory_at(
            parent_descriptor,
            name,
            display_path=display_path,
        )
    except NotADirectoryError as exc:
        # Preserve Path.mkdir(exist_ok=True) semantics for callers that use a
        # file where a run or artifact directory must be created.
        raise FileExistsError(display_path) from exc


def _open_artifact_parent(
    root_directory: Path,
    run_id: str,
    parts: tuple[str, ...],
    *,
    create: bool,
) -> int:
    """Open an artifact parent below pinned root and run directory FDs."""
    current_descriptor = _open_directory_path(root_directory)
    try:
        current_path = root_directory / run_id
        next_descriptor = _open_directory_at(
            current_descriptor,
            run_id,
            display_path=current_path,
        )
        os.close(current_descriptor)
        current_descriptor = next_descriptor
        for part in parts[:-1]:
            current_path /= part
            if create:
                next_descriptor = _open_or_create_directory_at(
                    current_descriptor,
                    part,
                    display_path=current_path,
                )
            else:
                next_descriptor = _open_directory_at(
                    current_descriptor,
                    part,
                    display_path=current_path,
                )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        result = current_descriptor
        current_descriptor = -1
        return result
    finally:
        if current_descriptor >= 0:
            os.close(current_descriptor)


def _is_symlink_at(parent_descriptor: int, name: str) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISLNK(metadata.st_mode)


def _reject_symlink_or_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise UnsafeArtifactPath(f"artifact destination is a symlink: {display_path}")
    if stat.S_ISDIR(metadata.st_mode):
        raise UnsafeArtifactPath(f"artifact destination is a directory: {display_path}")


def _create_temporary_file_at(
    parent_descriptor: int,
    *,
    destination_name: str,
) -> tuple[str, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        name = f".{destination_name}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                mode=0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        return name, descriptor
    raise FileExistsError("could not allocate a unique artifact temporary file")


def _open_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        if _is_symlink_at(parent_descriptor, name):
            raise UnsafeArtifactPath(f"artifact is a symlink: {display_path}") from exc
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise UnsafeArtifactPath(f"artifact is not a regular file: {display_path}")
    return descriptor


def _hash_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    display_path: Path,
    max_bytes: int,
) -> tuple[str, int]:
    file_descriptor = _open_regular_file_at(
        parent_descriptor,
        name,
        display_path=display_path,
    )
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(file_descriptor)
        if metadata.st_size > max_bytes:
            raise ArtifactSizeLimitError(
                f"artifact exceeds configured byte limit ({metadata.st_size} > {max_bytes})"
            )
        observed_size = 0
        with os.fdopen(file_descriptor, "rb") as handle:
            file_descriptor = -1
            while chunk := handle.read(min(1024 * 1024, max_bytes + 1)):
                observed_size += len(chunk)
                if observed_size > max_bytes:
                    raise ArtifactSizeLimitError(
                        "artifact exceeds configured byte limit during read"
                    )
                digest.update(chunk)
        return digest.hexdigest(), observed_size
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _unlink_at(
    parent_descriptor: int,
    name: str,
    *,
    missing_ok: bool = False,
) -> None:
    try:
        os.unlink(name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        if not missing_ok:
            raise
