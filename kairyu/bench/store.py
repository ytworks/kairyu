"""Result store: bench/results/fugu/<run_id>/ with atomic per-pair JSON.

Resume contract: a pair JSON whose status is not "failed" is reused on the
next run with the same run_id; failed pairs are always retried; --rerun
ignores everything. Filenames sanitize model names (they may contain "/").
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kairyu.bench.types import PairResult


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


class ResultStore:
    def __init__(self, results_dir: str | Path, run_id: str) -> None:
        self.run_dir = Path(results_dir) / run_id
        self.run_id = run_id

    def ensure(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_run_config(self, config: dict) -> None:
        self.ensure()
        self._atomic_write(self.run_dir / "run.json", json.dumps(config, indent=2))

    def pair_path(self, benchmark: str, target: str) -> Path:
        return self.run_dir / _safe(benchmark) / f"{_safe(target)}.json"

    def load_pair(self, benchmark: str, target: str) -> PairResult | None:
        path = self.pair_path(benchmark, target)
        if not path.exists():
            return None
        return PairResult.model_validate_json(path.read_text(encoding="utf-8"))

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
