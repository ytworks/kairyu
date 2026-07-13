"""Scoreboard aggregation: pair results -> scoreboard.json + Fugu-layout markdown.

Layout mirrors the Fugu release table: rows = benchmarks in FUGU_ROW_ORDER,
columns = targets. Cells carry footnote markers for annotations and
partial/skip reasons so a degraded run is still an honest artifact.
"""

from __future__ import annotations

from collections.abc import Sequence

from kairyu.bench.adapters import FUGU_ROW_ORDER, all_adapters
from kairyu.bench.types import (
    SCHEMA_VERSION,
    BenchTarget,
    JudgeConfig,
    PairResult,
)


def _resolved_identity(base_url: str, model: str) -> tuple[str, str]:
    return base_url.rstrip("/"), model


def build_scoreboard(
    *,
    run_id: str,
    suite: str,
    config: dict,
    environment: dict,
    pairs: list[PairResult],
    targets: list[str],
    target_configs: Sequence[BenchTarget] | None = None,
    judge: JudgeConfig | None = None,
) -> dict:
    display_names = {
        name: adapter.info.display_name for name, adapter in all_adapters().items()
    }
    by_key = {(pair.benchmark, pair.target): pair for pair in pairs}
    benchmarks = [name for name in FUGU_ROW_ORDER if any(p.benchmark == name for p in pairs)]

    footnotes: list[str] = []

    def footnote(text: str) -> int:
        if text not in footnotes:
            footnotes.append(text)
        return footnotes.index(text) + 1

    # Self-judging is an endpoint/model identity question, never a display-label
    # comparison. Legacy artifacts without enough identity data fail closed as
    # "independence unknown" instead of being declared independent.
    configured_by_label = {
        target.label(): target for target in (target_configs or ())
    }
    judge_requested = judge is not None and (
        judge.base_url is not None or judge.model is not None
    )
    judge_identity = (
        _resolved_identity(judge.base_url, judge.model)
        if judge is not None and judge.enabled
        else None
    )
    self_judged: list[str] = []
    identity_unknown: list[str] = []
    if judge_requested:
        for label in targets:
            target = configured_by_label.get(label)
            if judge_identity is None or target is None:
                identity_unknown.append(label)
            elif _resolved_identity(target.base_url, target.model) == judge_identity:
                self_judged.append(label)

    cells: dict[str, dict[str, dict]] = {}
    for benchmark in benchmarks:
        cells[benchmark] = {}
        for target in targets:
            pair = by_key.get((benchmark, target))
            if pair is None:
                cells[benchmark][target] = {
                    "status": "skipped",
                    "score": None,
                    "n": 0,
                    "reason": "not run",
                    "footnotes": [footnote(f"{benchmark}/{target}: not run")],
                }
                continue
            notes = [footnote(f"{benchmark}: {text}") for text in pair.annotations]
            if pair.status in ("skipped", "partial", "failed") and pair.reason:
                notes.append(footnote(f"{benchmark}/{target}: {pair.status} — {pair.reason}"))
            if target in self_judged:
                notes.append(
                    footnote(
                        f"{target}: self-judged "
                        "(resolved judge endpoint/model == target)"
                    )
                )
            if target in identity_unknown:
                notes.append(
                    footnote(
                        f"{target}: judge independence unknown "
                        "(resolved target or judge identity unavailable)"
                    )
                )
            cells[benchmark][target] = {
                "status": pair.status,
                "score": pair.score,
                "n": pair.metrics.get("n_total"),
                "reason": pair.reason,
                "footnotes": notes,
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "suite": suite,
        "environment": environment,
        "config": config,
        "benchmarks": benchmarks,
        "display_names": {name: display_names.get(name, name) for name in benchmarks},
        "targets": targets,
        "self_judged_targets": self_judged,
        "judge_independence_unknown_targets": identity_unknown,
        "cells": cells,
        "footnotes": footnotes,
    }


def _cell_text(cell: dict) -> str:
    marks = "".join(f"[^{n}]" for n in cell["footnotes"])
    if cell["status"] == "skipped":
        return f"—{marks}"
    if cell["score"] is None:
        return f"n/a{marks}"
    text = f"{cell['score'] * 100:.1f}"
    if cell["status"] in ("partial", "failed"):
        text += "*"
    return f"{text}{marks}"


def render_markdown(scoreboard: dict) -> str:
    targets = scoreboard["targets"]
    lines = [
        f"# Fugu benchmark scoreboard — run {scoreboard['run_id']}",
        "",
        "Scores are percentages; — = skipped, * = partial/failed (see footnotes).",
        "",
        "| Benchmark | " + " | ".join(targets) + " |",
        "|---" * (len(targets) + 1) + "|",
    ]
    for benchmark in scoreboard["benchmarks"]:
        display = scoreboard["display_names"].get(benchmark, benchmark)
        row = [display] + [
            _cell_text(scoreboard["cells"][benchmark][target]) for target in targets
        ]
        lines.append("| " + " | ".join(row) + " |")
    if scoreboard["footnotes"]:
        lines.append("")
        for index, note in enumerate(scoreboard["footnotes"], start=1):
            lines.append(f"[^{index}]: {note}")
    lines.append("")
    return "\n".join(lines)
