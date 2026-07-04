"""Scoreboard aggregation: pair results -> scoreboard.json + Fugu-layout markdown.

Layout mirrors the Fugu release table: rows = benchmarks in FUGU_ROW_ORDER,
columns = targets. Cells carry footnote markers for annotations and
partial/skip reasons so a degraded run is still an honest artifact.
"""

from __future__ import annotations

from kairyu.bench.adapters import FUGU_ROW_ORDER, all_adapters
from kairyu.bench.types import SCHEMA_VERSION, PairResult


def build_scoreboard(
    *,
    run_id: str,
    suite: str,
    config: dict,
    environment: dict,
    pairs: list[PairResult],
    targets: list[str],
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

    # self-judging: a target graded by an LLM judge that IS that target biases the
    # score in its own favor — flag those columns so the number is not read as
    # independent (judge==target detection)
    judge_model = (config.get("judge") or {}).get("model")
    self_judged = [target for target in targets if judge_model and target == judge_model]

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
                notes.append(footnote(f"{target}: self-judged (judge model == target)"))
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
        "self_judged_targets": self_judged,  # judge model == target (biased) — flag it
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
