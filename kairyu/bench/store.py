"""Result store: bench/results/fugu/<run_id>/ with atomic per-pair JSON.

Resume contract: a pair JSON whose status is not "failed" is reused on the
next run with the same run_id; failed pairs are always retried; --rerun
ignores everything. Filenames sanitize model names (they may contain "/").
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from kairyu.bench.types import PairResult


def _safe(name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "__", name)[:96] or "value"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}--{digest}"


def _validate_run_id(run_id: str) -> None:
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or Path(run_id).is_absolute()
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
            (self.run_dir / "run.json.tmp").unlink(missing_ok=True)
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
        if resolved_run_dir.parent != resolved_results_dir:
            raise ValueError(
                f"fingerprint-bound run id {self.run_id!r} resolves outside "
                "the results directory"
            )

    def _require_matching_fingerprint(self, expected: str) -> None:
        run_config = self.run_dir / "run.json"
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
        self._require_contained_run_dir()
        return self.run_dir / _safe(benchmark) / f"{_safe(target)}.json"

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
        self._atomic_write(path, result.model_dump_json(indent=2))
        return path

    def save_scoreboard(self, scoreboard: dict, markdown: str) -> Path:
        self.ensure()
        self._atomic_write(
            self.run_dir / "scoreboard.json", json.dumps(scoreboard, indent=2)
        )
        self._atomic_write(self.run_dir / "scoreboard.md", markdown)
        return self.run_dir / "scoreboard.md"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
