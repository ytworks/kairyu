"""Result store: bench/results/fugu/<run_id>/ with atomic per-pair JSON.

Resume contract: a pair JSON whose status is not "failed" is reused on the
next run with the same run_id; failed pairs are always retried; --rerun
ignores everything. Filenames sanitize model names (they may contain "/").
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PureWindowsPath

from kairyu.bench.types import PairResult


def _safe(name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "__", name)[:96] or "value"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}--{digest}"


def _validate_run_id(run_id: str) -> None:
    windows_drive = PureWindowsPath(run_id).drive if isinstance(run_id, str) else ""
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or Path(run_id).is_absolute()
        or windows_drive
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError(
            f"invalid run id {run_id!r}: expected one non-dot path component"
        )


class ResultStore:
    def __init__(self, results_dir: str | Path, run_id: str) -> None:
        _validate_run_id(run_id)
        self.results_dir = Path(results_dir)
        self.run_dir = self.results_dir / run_id
        self.run_id = run_id

    def ensure(self) -> None:
        self._require_contained_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._require_contained_run_dir()

    def write_run_config(self, config: dict) -> None:
        self.ensure()
        self._atomic_write(self.run_dir / "run.json", json.dumps(config, indent=2))

    def initialize_run(self, metadata: dict) -> None:
        """Create run metadata once, or validate an existing run identity."""
        fingerprint = metadata.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(
                f"run id {self.run_id!r} requires a non-empty fingerprint"
            )

        text = json.dumps(metadata, indent=2)
        self._require_contained_run_dir()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._require_contained_run_dir()
        try:
            self.run_dir.mkdir()
        except FileExistsError:
            self._require_contained_run_dir()
            self._require_matching_fingerprint(fingerprint)
            return

        try:
            self._atomic_write(self.run_dir / "run.json", text)
        except Exception:
            self._remove_partial_legacy_tmp(self.run_dir / "run.json")
            try:
                self.run_dir.rmdir()
            except OSError:
                pass
            raise

    def _require_contained_run_dir(self) -> None:
        try:
            resolved_results_dir = self.results_dir.resolve(strict=False)
            resolved_run_dir = self.run_dir.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"fingerprint-bound run id {self.run_id!r} cannot be resolved "
                "within the results directory"
            ) from error
        if self.run_dir.is_symlink() or resolved_run_dir.parent != resolved_results_dir:
            raise ValueError(
                f"fingerprint-bound run id {self.run_id!r} resolves outside "
                "the results directory"
            )

    def _require_contained_artifact(self, path: Path) -> Path:
        """Refuse symlinks or resolved escapes anywhere below this run root."""
        self._require_contained_run_dir()
        try:
            relative = path.relative_to(self.run_dir)
            resolved_run_dir = self.run_dir.resolve(strict=False)
            resolved_path = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"fingerprint-bound run id {self.run_id!r} has an unsafe artifact path"
            ) from error

        current = self.run_dir
        try:
            for component in relative.parts:
                current /= component
                if current.is_symlink():
                    raise ValueError(
                        f"fingerprint-bound run id {self.run_id!r} refuses "
                        f"symlink artifact {path}"
                    )
        except OSError as error:
            raise ValueError(
                f"fingerprint-bound run id {self.run_id!r} has an unsafe artifact path"
            ) from error

        if resolved_run_dir not in resolved_path.parents:
            raise ValueError(
                f"fingerprint-bound run id {self.run_id!r} resolves artifact "
                "outside the run directory"
            )
        return path

    def _preflight_atomic_write(self, path: Path) -> None:
        self._require_contained_artifact(path)
        self._require_contained_artifact(
            path.with_suffix(path.suffix + ".tmp")
        )

    def _remove_partial_legacy_tmp(self, path: Path) -> None:
        legacy_tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            self._require_contained_artifact(legacy_tmp)
        except ValueError:
            return
        legacy_tmp.unlink(missing_ok=True)

    def _require_matching_fingerprint(self, expected: str) -> None:
        run_config = self._require_contained_artifact(self.run_dir / "run.json")
        try:
            existing = json.loads(run_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"run id {self.run_id!r} has no matching fingerprint"
            ) from error

        if not isinstance(existing, dict) or existing.get("fingerprint") != expected:
            raise ValueError(
                f"run id {self.run_id!r} has no matching fingerprint"
            )

    def pair_path(self, benchmark: str, target: str) -> Path:
        return self._require_contained_artifact(
            self.run_dir / _safe(benchmark) / f"{_safe(target)}.json"
        )

    def load_pair(
        self,
        benchmark: str,
        target: str,
        *,
        expected_fingerprint: str | None = None,
    ) -> PairResult | None:
        path = self.pair_path(benchmark, target)
        if not path.exists():
            return None
        result = PairResult.model_validate_json(path.read_text(encoding="utf-8"))
        if (
            expected_fingerprint is not None
            and result.run_fingerprint != expected_fingerprint
        ):
            return None
        return result

    def save_pair(self, result: PairResult) -> Path:
        path = self.pair_path(result.benchmark, result.target)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._require_contained_artifact(path)
        self._atomic_write(path, result.model_dump_json(indent=2))
        return path

    def save_scoreboard(self, scoreboard: dict, markdown: str) -> Path:
        self.ensure()
        json_path = self.run_dir / "scoreboard.json"
        markdown_path = self.run_dir / "scoreboard.md"
        for path in (json_path, markdown_path):
            self._preflight_atomic_write(path)
        self._atomic_write(json_path, json.dumps(scoreboard, indent=2))
        self._atomic_write(markdown_path, markdown)
        return markdown_path

    def _atomic_write(self, path: Path, text: str) -> None:
        self._preflight_atomic_write(path)
        fd: int | None = None
        tmp: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                text=True,
            )
            tmp = self._require_contained_artifact(path.parent / Path(tmp_name).name)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                fd = None
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            self._preflight_atomic_write(path)
            os.replace(tmp, path)
            tmp = None
        finally:
            if fd is not None:
                os.close(fd)
            if tmp is not None:
                tmp.unlink(missing_ok=True)
