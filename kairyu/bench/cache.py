"""Benchmark dataset cache: downloaded once, normalized to JSONL, never committed.

Layout: <root>/<adapter>/{data.jsonl, assets/, manifest.json}. The manifest
makes downloads idempotent (`is_ready`) and records provenance (dataset id,
revision, row count, sha256) for the methodology block of every result.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

_ENV_VAR = "KAIRYU_BENCH_CACHE"
_DEFAULT = "~/.cache/kairyu/benchmarks"
_SHA256_HEX = frozenset("0123456789abcdef")
_FileIdentity = tuple[int, int, int, int, int]


def _file_identity(path: Path) -> _FileIdentity:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _referenced_assets(value: object) -> tuple[str, ...]:
    references: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, str) or not item.startswith("assets/"):
            return
        relative = PurePosixPath(item)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
            raise ValueError(f"unsafe benchmark asset reference {item!r}")
        references.add(relative.as_posix())

    visit(value)
    return tuple(sorted(references))


def _asset_identities(adapter_dir: Path, references: Sequence[str]) -> list[dict[str, str]]:
    identities: list[dict[str, str]] = []
    for reference in references:
        relative = PurePosixPath(reference)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 2
            or relative.parts[0] != "assets"
        ):
            raise ValueError(f"unsafe benchmark asset reference {reference!r}")
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(adapter_dir, directory_flags)
        except OSError as error:
            raise OSError(f"benchmark asset {reference!r} is missing or unsafe") from error
        try:
            for component in relative.parts[:-1]:
                try:
                    child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                except OSError as error:
                    raise OSError(
                        f"benchmark asset {reference!r} is missing or unsafe"
                    ) from error
                os.close(directory_fd)
                directory_fd = child_fd

            file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            file_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            try:
                file_fd = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
            except OSError as error:
                raise OSError(
                    f"benchmark asset {reference!r} is missing or unsafe"
                ) from error
            try:
                if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                    raise OSError(f"benchmark asset {reference!r} is missing or unsafe")
                digest = hashlib.sha256()
                while chunk := os.read(file_fd, 1024 * 1024):
                    digest.update(chunk)
            finally:
                os.close(file_fd)
        finally:
            os.close(directory_fd)
        identities.append({"path": reference, "sha256": digest.hexdigest()})
    return identities


def resolve_cache_root(flag: str | None = None) -> Path:
    """--cache-dir flag > $KAIRYU_BENCH_CACHE > ~/.cache/kairyu/benchmarks."""
    raw = flag or os.environ.get(_ENV_VAR) or _DEFAULT
    return Path(raw).expanduser()


class BenchCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        # A suite run asks the same readiness question while resolving
        # downloads, identities, and runtime preconditions. Keep the expensive
        # data digest/JSON scan only while both files retain their exact stat
        # identity; adapter-owned assets are still hashed on every check.
        self._verified_content: dict[
            str,
            tuple[_FileIdentity, _FileIdentity, dict, tuple[str, ...]],
        ] = {}

    def adapter_dir(self, adapter: str) -> Path:
        return self.root / adapter

    def data_path(self, adapter: str) -> Path:
        return self.adapter_dir(adapter) / "data.jsonl"

    def assets_dir(self, adapter: str) -> Path:
        return self.adapter_dir(adapter) / "assets"

    def manifest_path(self, adapter: str) -> Path:
        return self.adapter_dir(adapter) / "manifest.json"

    def is_ready(
        self,
        adapter: str,
        dataset: str | None = None,
        revision: str | None = None,
        sources: Sequence[Sequence[str]] | None = None,
    ) -> bool:
        """Return whether the cached bytes and optional dataset pins still match.

        `sources` are the adapter's secondary pins (testcase archives, golden
        data). They determine the tests and expected answers, so a cache built
        under different ones is not ready even when `data.jsonl` still hashes
        correctly.
        """
        if not (self.manifest_path(adapter).exists() and self.data_path(adapter).exists()):
            return False
        try:
            data_identity = _file_identity(self.data_path(adapter))
            manifest_identity = _file_identity(self.manifest_path(adapter))
            verified = self._verified_content.get(adapter)
            if (
                verified is not None
                and verified[0] == data_identity
                and verified[1] == manifest_identity
            ):
                manifest, references = verified[2], verified[3]
            else:
                manifest = self.read_manifest(adapter)
                if not isinstance(manifest, dict):
                    return False
                expected_digest = manifest.get("sha256")
                if (
                    not isinstance(expected_digest, str)
                    or len(expected_digest) != 64
                    or any(character not in _SHA256_HEX for character in expected_digest)
                ):
                    return False
                if _sha256_file(self.data_path(adapter)) != expected_digest:
                    return False
                references = _referenced_assets(self.read_rows(adapter))
                self._verified_content[adapter] = (
                    data_identity,
                    manifest_identity,
                    manifest,
                    references,
                )
            recorded_assets = manifest.get("assets")
            if not isinstance(recorded_assets, list):
                return False
            if _asset_identities(self.adapter_dir(adapter), references) != recorded_assets:
                return False
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return False
        if dataset is not None and manifest.get("dataset") != dataset:
            return False
        if revision is not None and manifest.get("revision") != revision:
            return False
        if sources:
            recorded = manifest.get("sources")
            if not isinstance(recorded, list):
                return False  # pre-`sources` cache: rebuild rather than trust it
            if [list(entry) for entry in recorded] != [list(entry) for entry in sources]:
                return False
        return True

    def read_manifest(self, adapter: str) -> dict:
        return json.loads(self.manifest_path(adapter).read_text(encoding="utf-8"))

    def read_rows(self, adapter: str) -> list[dict]:
        rows = []
        with self.data_path(adapter).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def write_rows(self, adapter: str, rows: list[dict], manifest: dict) -> None:
        """Materialize normalized rows + provenance manifest (atomic-enough:
        manifest is written last, and `is_ready` requires both files)."""
        directory = self.adapter_dir(adapter)
        directory.mkdir(parents=True, exist_ok=True)
        data_path = self.data_path(adapter)
        with data_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        digest = _sha256_file(data_path)
        assets = _asset_identities(self.adapter_dir(adapter), _referenced_assets(rows))
        full_manifest = {
            **manifest,
            "rows": len(rows),
            "sha256": digest,
            "assets": assets,
        }
        self.manifest_path(adapter).write_text(
            json.dumps(full_manifest, indent=2), encoding="utf-8"
        )
        self._verified_content.pop(adapter, None)

    def clear(self, adapter: str) -> None:
        self._verified_content.pop(adapter, None)
        manifest = self.manifest_path(adapter)
        data = self.data_path(adapter)
        for path in (manifest, data):
            if path.exists():
                path.unlink()
