"""Benchmark dataset cache: downloaded once, normalized to JSONL, never committed.

Layout: <root>/<adapter>/{data.jsonl, assets/, manifest.json}. The manifest
makes downloads idempotent (`is_ready`) and records provenance (dataset id,
revision, row count, sha256) for the methodology block of every result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_ENV_VAR = "KAIRYU_BENCH_CACHE"
_DEFAULT = "~/.cache/kairyu/benchmarks"


def resolve_cache_root(flag: str | None = None) -> Path:
    """--cache-dir flag > $KAIRYU_BENCH_CACHE > ~/.cache/kairyu/benchmarks."""
    raw = flag or os.environ.get(_ENV_VAR) or _DEFAULT
    return Path(raw).expanduser()


class BenchCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def adapter_dir(self, adapter: str) -> Path:
        return self.root / adapter

    def data_path(self, adapter: str) -> Path:
        return self.adapter_dir(adapter) / "data.jsonl"

    def assets_dir(self, adapter: str) -> Path:
        return self.adapter_dir(adapter) / "assets"

    def manifest_path(self, adapter: str) -> Path:
        return self.adapter_dir(adapter) / "manifest.json"

    def is_ready(self, adapter: str) -> bool:
        return self.manifest_path(adapter).exists() and self.data_path(adapter).exists()

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
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        full_manifest = {**manifest, "rows": len(rows), "sha256": digest}
        self.manifest_path(adapter).write_text(
            json.dumps(full_manifest, indent=2), encoding="utf-8"
        )

    def clear(self, adapter: str) -> None:
        manifest = self.manifest_path(adapter)
        data = self.data_path(adapter)
        for path in (manifest, data):
            if path.exists():
                path.unlink()
