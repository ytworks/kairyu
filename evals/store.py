"""Result store: bench/results/<suite>/<run_id>/ with atomic per-pair JSON.

Resume contract: a pair JSON whose status is not "failed" is reused on the
next run with the same run_id; failed pairs are always retried; --rerun and a
persisted evaluator-drift recovery marker ignore every pair. Filenames sanitize
model names (they may contain "/").
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from pathlib import Path, PureWindowsPath

from evals import history as scoreboard_history
from evals.types import PairResult
from evidence.reporting import atomic_write_text

_SOURCE_TAINT_NAME = "source-taint.json"
_EVALUATOR_TAINT_NAME = "evaluator-taint.json"


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
        raise ValueError(f"invalid run id {run_id!r}: expected one non-dot path component")


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

    def initialize_run(self, metadata: dict) -> dict:
        """Create run metadata once, or return its persisted authority.

        Resume tolerates a new observation time and momentary execution-runner
        availability, but never a different score-bearing fingerprint or stable
        source/runtime environment.
        """
        fingerprint = metadata.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"run id {self.run_id!r} requires a non-empty fingerprint")
        if metadata.get("run_id") != self.run_id:
            raise ValueError(f"run id {self.run_id!r} does not match persisted run metadata")

        text = json.dumps(metadata, indent=2)
        self._require_contained_run_dir()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._require_contained_run_dir()
        try:
            self.run_dir.mkdir()
        except FileExistsError:
            self._require_contained_run_dir()
            return self._require_matching_metadata(metadata)

        try:
            self._atomic_write(self.run_dir / "run.json", text)
        except Exception:
            self._remove_partial_legacy_tmp(self.run_dir / "run.json")
            try:
                self.run_dir.rmdir()
            except OSError:
                pass
            raise
        return json.loads(text)

    def require_resumable_environment(self, environment: dict) -> None:
        """Preflight an existing run's stable environment before external work.

        A new run has nothing to validate.  An existing run must disclose a
        readable ``run.json`` whose stable source/runtime identity matches the
        candidate environment; containment checks happen before any read.
        """
        self._require_contained_run_dir()
        if not self.run_dir.exists():
            return
        self.require_source_untainted()
        run_config = self._require_contained_artifact(self.run_dir / "run.json")
        try:
            existing = json.loads(run_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"run id {self.run_id!r} has no matching fingerprint or stable environment identity"
            ) from error
        if not isinstance(
            existing, dict
        ) or not scoreboard_history.resumable_environment_identity_equal(
            existing.get("environment"), environment
        ):
            raise ValueError(
                f"run id {self.run_id!r} has no matching fingerprint or stable environment identity"
            )

    def require_source_untainted(self) -> None:
        """Permanently reject a run id that observed local source drift."""
        self._require_contained_run_dir()
        if not self.run_dir.exists():
            return
        marker = self._require_contained_artifact(self.run_dir / _SOURCE_TAINT_NAME)
        if marker.exists():
            raise ValueError(
                f"run id {self.run_id!r} is permanently source-tainted; use a new run id"
            )

    def mark_source_tainted(self, *, expected: dict, observed: dict) -> Path:
        """Persist the first source-drift observation before the run can resume."""
        self.ensure()
        marker = self._require_contained_artifact(self.run_dir / _SOURCE_TAINT_NAME)
        if marker.exists():
            return marker
        payload = {
            "schema_version": 1,
            "reason": "local benchmark source identity changed while the suite ran",
            "expected": expected,
            "observed": observed,
        }
        self._atomic_write(marker, json.dumps(payload, indent=2))
        return marker

    def has_evaluator_taint(self) -> bool:
        """Return whether a prior attempt observed score-bearing identity drift."""
        self._require_contained_run_dir()
        if not self.run_dir.exists():
            return False
        marker = self._require_contained_artifact(self.run_dir / _EVALUATOR_TAINT_NAME)
        return marker.exists()

    def mark_evaluator_tainted(self, *, reason: str, adapters: Sequence[str]) -> Path:
        """Persist evaluator drift before any previously completed pair can survive it."""
        self.ensure()
        marker = self._require_contained_artifact(self.run_dir / _EVALUATOR_TAINT_NAME)
        if marker.exists():
            return marker
        payload = {
            "schema_version": 1,
            "reason": reason,
            "adapters": sorted(set(adapters)),
        }
        self._atomic_write(marker, json.dumps(payload, indent=2))
        return marker

    def clear_evaluator_taint(self) -> None:
        """Clear the recovery marker only after every pair was safely re-evaluated."""
        self._require_contained_run_dir()
        if not self.run_dir.exists():
            return
        marker = self._require_contained_artifact(self.run_dir / _EVALUATOR_TAINT_NAME)
        marker.unlink(missing_ok=True)

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
                f"fingerprint-bound run id {self.run_id!r} resolves outside the results directory"
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
                        f"fingerprint-bound run id {self.run_id!r} refuses symlink artifact {path}"
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
        self._require_contained_artifact(path.with_suffix(path.suffix + ".tmp"))

    def _remove_partial_legacy_tmp(self, path: Path) -> None:
        legacy_tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            self._require_contained_artifact(legacy_tmp)
        except ValueError:
            return
        legacy_tmp.unlink(missing_ok=True)

    def _require_matching_metadata(self, expected: dict) -> dict:
        run_config = self._require_contained_artifact(self.run_dir / "run.json")
        try:
            existing = json.loads(run_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"run id {self.run_id!r} has no matching fingerprint") from error

        if (
            not isinstance(existing, dict)
            or existing.get("fingerprint") != expected.get("fingerprint")
            or existing.get("run_id") != expected.get("run_id")
            or not scoreboard_history.resumable_environment_identity_equal(
                existing.get("environment"), expected.get("environment")
            )
        ):
            raise ValueError(
                f"run id {self.run_id!r} has no matching fingerprint and "
                "stable environment identity"
            )
        return existing

    def append_scoreboard_index(
        self,
        scoreboard: dict,
        run_metadata: dict,
        pairs: Sequence[object] = (),
    ) -> Path:
        """Append one source-bound scoreboard snapshot to the suite history."""
        self._require_metadata_run_id(run_metadata)
        return scoreboard_history.append_scoreboard_index(
            self.results_dir,
            scoreboard,
            run_metadata,
            pairs=pairs,
        )

    def load_scoreboard_index(self) -> tuple[dict, ...]:
        """Load and fully validate this results root's scoreboard history."""
        return scoreboard_history.load_scoreboard_index(self.results_dir)

    def find_scoreboard_entry(self, run_id: str) -> dict | None:
        """Find one validated history entry by its unique run id."""
        return scoreboard_history.find_scoreboard_entry(self.results_dir, run_id)

    def ensure_scoreboard_key_available(
        self,
        run_metadata: dict,
        rerun: bool = False,
    ) -> None:
        """Fail before work if this immutable commit/fingerprint key is occupied."""
        self._require_metadata_run_id(run_metadata)
        scoreboard_history.ensure_scoreboard_key_available(
            self.results_dir,
            run_metadata,
            rerun=rerun,
        )

    def _require_metadata_run_id(self, metadata: object) -> None:
        if not isinstance(metadata, dict) or metadata.get("run_id") != self.run_id:
            raise ValueError(f"run id {self.run_id!r} does not match scoreboard run metadata")

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
        expected_source_identity: dict | None = None,
    ) -> PairResult | None:
        path = self.pair_path(benchmark, target)
        if not path.exists():
            return None
        result = PairResult.model_validate_json(path.read_text(encoding="utf-8"))
        if expected_fingerprint is not None and result.run_fingerprint != expected_fingerprint:
            return None
        if (
            expected_source_identity is not None
            and result.source_identity != expected_source_identity
        ):
            return None
        return result

    def load_indexed_pair(
        self,
        entry: dict,
        benchmark: str,
        target: str,
    ) -> PairResult:
        """Load one raw pair only when it exactly matches its history binding.

        The supplied row must still be the authoritative record in this
        results root's fully validated index. Missing, malformed, replaced, or
        symlinked evidence is an error. This is intentionally stricter than
        :meth:`load_pair`, whose ``None`` return is part of resume behavior.
        """
        if not isinstance(entry, dict) or entry.get("run_id") != self.run_id:
            raise ValueError(
                f"run id {self.run_id!r} does not match indexed scoreboard entry"
            )
        for field, value in (("benchmark", benchmark), ("target", target)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"indexed pair {field} must be a non-empty string")

        binding = scoreboard_history.indexed_pair_binding(
            entry,
            run_id=self.run_id,
            benchmark=benchmark,
            target=target,
        )
        authoritative = self.find_scoreboard_entry(self.run_id)
        if (
            authoritative is None
            or authoritative["record_sha256"] != entry["record_sha256"]
        ):
            raise ValueError(
                f"run id {self.run_id!r} entry is not the authoritative current "
                "scoreboard-index record"
            )
        authoritative_binding = scoreboard_history.indexed_pair_binding(
            authoritative,
            run_id=self.run_id,
            benchmark=benchmark,
            target=target,
        )
        if authoritative_binding != binding:
            raise ValueError(
                f"run id {self.run_id!r} pair binding differs from the authoritative index"
            )

        raw = self._read_regular_indexed_pair(benchmark=benchmark, target=target)
        return scoreboard_history.validate_pair_binding(
            raw,
            binding,
            benchmark=benchmark,
            target=target,
        )

    def _read_regular_indexed_pair(
        self,
        *,
        benchmark: str,
        target: str,
    ) -> bytes:
        """Read through pinned root/run/benchmark descriptors without symlinks."""
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NONBLOCK", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        benchmark_name = _safe(benchmark)
        pair_name = f"{_safe(target)}.json"
        descriptors: list[int] = []

        def require_same_entry(
            parent_fd: int,
            name: str,
            opened: os.stat_result,
            *,
            directory: bool,
        ) -> None:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
            if (
                not expected_kind(opened.st_mode)
                or current.st_dev != opened.st_dev
                or current.st_ino != opened.st_ino
            ):
                raise ValueError(
                    f"indexed pair {benchmark!r}/{target!r} path changed while reading"
                )

        def require_same_root(root_fd: int, opened: os.stat_result) -> None:
            pinned = os.fstat(root_fd)
            current = self.results_dir.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or pinned.st_dev != opened.st_dev
                or pinned.st_ino != opened.st_ino
                or current.st_dev != opened.st_dev
                or current.st_ino != opened.st_ino
            ):
                raise ValueError(
                    f"indexed pair {benchmark!r}/{target!r} results root changed while reading"
                )

        try:
            try:
                root_fd = os.open(self.results_dir, directory_flags)
                descriptors.append(root_fd)
                root_opened = os.fstat(root_fd)
                require_same_root(root_fd, root_opened)

                run_fd = os.open(self.run_id, directory_flags, dir_fd=root_fd)
                descriptors.append(run_fd)
                run_opened = os.fstat(run_fd)
                require_same_entry(
                    root_fd,
                    self.run_id,
                    run_opened,
                    directory=True,
                )

                benchmark_fd = os.open(
                    benchmark_name,
                    directory_flags,
                    dir_fd=run_fd,
                )
                descriptors.append(benchmark_fd)
                benchmark_opened = os.fstat(benchmark_fd)
                require_same_entry(
                    run_fd,
                    benchmark_name,
                    benchmark_opened,
                    directory=True,
                )

                pair_fd = os.open(pair_name, file_flags, dir_fd=benchmark_fd)
                descriptors.append(pair_fd)
                pair_opened = os.fstat(pair_fd)
                require_same_entry(
                    benchmark_fd,
                    pair_name,
                    pair_opened,
                    directory=False,
                )
            except OSError as error:
                if error.errno == errno.ENOENT:
                    raise ValueError(
                        f"indexed pair {benchmark!r}/{target!r} is missing for "
                        f"run id {self.run_id!r}"
                    ) from error
                raise ValueError(
                    f"indexed pair {benchmark!r}/{target!r} cannot be opened safely "
                    f"for run id {self.run_id!r}"
                ) from error

            chunks: list[bytes] = []
            remaining = pair_opened.st_size
            while remaining:
                chunk = os.read(pair_fd, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(
                        f"indexed pair {benchmark!r}/{target!r} changed while reading"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(pair_fd, 1):
                raise ValueError(
                    f"indexed pair {benchmark!r}/{target!r} changed while reading"
                )

            pair_after = os.fstat(pair_fd)
            if (
                pair_after.st_dev != pair_opened.st_dev
                or pair_after.st_ino != pair_opened.st_ino
                or pair_after.st_size != pair_opened.st_size
            ):
                raise ValueError(
                    f"indexed pair {benchmark!r}/{target!r} changed while reading"
                )
            require_same_entry(
                benchmark_fd,
                pair_name,
                pair_opened,
                directory=False,
            )
            require_same_entry(
                run_fd,
                benchmark_name,
                benchmark_opened,
                directory=True,
            )
            require_same_entry(
                root_fd,
                self.run_id,
                run_opened,
                directory=True,
            )
            require_same_root(root_fd, root_opened)
            return b"".join(chunks)
        except ValueError:
            raise
        except OSError as error:
            raise ValueError(
                f"indexed pair {benchmark!r}/{target!r} cannot be read safely "
                f"for run id {self.run_id!r}"
            ) from error
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

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

    def save_config_comparison(self, comparison: dict, markdown: str) -> Path:
        """Configuration A/B report for immutable eval runs."""
        self.ensure()
        json_path = self.run_dir / "config-comparison.json"
        markdown_path = self.run_dir / "config-comparison.md"
        for path in (json_path, markdown_path):
            self._preflight_atomic_write(path)
        self._atomic_write(
            json_path,
            json.dumps(comparison, indent=2, allow_nan=False),
        )
        self._atomic_write(markdown_path, markdown)
        return markdown_path

    def save_quantization_sweep(self, sweep: dict, markdown: str) -> Path:
        """Retained quantization table without replacing other reports."""
        self.ensure()
        json_path = self.run_dir / "quantization-sweep.json"
        markdown_path = self.run_dir / "quantization-sweep.md"
        for path in (json_path, markdown_path):
            self._preflight_atomic_write(path)
        self._atomic_write(
            json_path,
            json.dumps(sweep, indent=2, allow_nan=False),
        )
        self._atomic_write(markdown_path, markdown)
        return markdown_path

    def _atomic_write(self, path: Path, text: str) -> None:
        self._preflight_atomic_write(path)
        atomic_write_text(
            path,
            text,
            validate_path=self._require_contained_artifact,
        )
