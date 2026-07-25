"""Accuracy report: a finished run next to the published Fugu-release scores.

The scoreboard says what kairyu measured. This says how that sits against the
numbers Sakana published — with the deltas, and with every reason the delta may
not mean what it looks like: a skipped or partial cell, a substituted dataset,
an uncompiled checker, a single attempt where the published figure used four.

Rules the renderer keeps:

- a missing measurement is `—`, never zero and never omitted;
- a `partial` cell is marked, because its denominator is not the full set;
- a row kairyu measures differently is marked NOT COMPARABLE and its delta is
  withheld rather than printed with a caveat somewhere further down;
- the baselines are labelled provider-reported, because the page says they are.
"""

from __future__ import annotations

from kairyu.bench.reference import (
    NOT_COMPARABLE,
    PROVIDER_REPORTED,
    PUBLISHED_VARIANTS,
    published,
    published_models,
    reference_metadata,
)

#: Published column used as the delta baseline (the model the suite is named for).
DELTA_AGAINST = "Fugu"


def build_comparison(scoreboard: dict) -> dict:
    """Scoreboard + published values -> a comparison document."""
    targets = list(scoreboard.get("targets") or [])
    rows = []
    for benchmark in scoreboard.get("benchmarks") or []:
        cells = (scoreboard.get("cells") or {}).get(benchmark, {})
        reference = published(benchmark)
        measured = {}
        for target in targets:
            cell = cells.get(target) or {}
            score = cell.get("score")
            measured[target] = {
                "score": None if score is None else round(score * 100, 1),
                "status": cell.get("status", "skipped"),
                "n": cell.get("n"),
                "reason": cell.get("reason"),
                "comparable": cell.get("status") == "completed"
                and benchmark not in NOT_COMPARABLE,
            }
        rows.append(
            {
                "benchmark": benchmark,
                "display_name": (scoreboard.get("display_names") or {}).get(
                    benchmark, benchmark
                ),
                "measured": measured,
                "published": reference,
                "published_models": list(published_models(benchmark)),
                "variant": PUBLISHED_VARIANTS.get(benchmark),
                "not_comparable": NOT_COMPARABLE.get(benchmark),
                "deltas": {
                    target: _delta(values, reference, benchmark)
                    for target, values in measured.items()
                },
            }
        )
    return {
        "run_id": scoreboard.get("run_id"),
        "suite": scoreboard.get("suite"),
        "targets": targets,
        "delta_against": DELTA_AGAINST,
        "reference": reference_metadata(),
        "rows": rows,
        # the scoreboard's own footnotes: substituted datasets, uncompiled
        # checkers, self-judging, degraded cells
        "methodology_notes": list(scoreboard.get("footnotes") or []),
    }


def _delta(values: dict, reference: dict[str, float], benchmark: str) -> float | None:
    """Measured minus published `Fugu`, or None when the comparison is unsound."""
    baseline = reference.get(DELTA_AGAINST)
    if values["score"] is None or baseline is None:
        return None
    if benchmark in NOT_COMPARABLE:
        return None
    return round(values["score"] - baseline, 1)


def _measured_text(values: dict) -> str:
    if values["score"] is None:
        return "—"
    text = f"{values['score']:.1f}"
    if values["status"] == "partial":
        text += "*"
    elif values["status"] == "failed":
        text += "!"
    return text


def _delta_text(delta: float | None, values: dict, not_comparable: str | None) -> str:
    if not_comparable is not None:
        return "n/c"
    if delta is None:
        return "—"
    sign = "+" if delta > 0 else ""
    text = f"{sign}{delta:.1f}"
    return text if values["status"] == "completed" else f"{text}*"


def render_comparison_markdown(comparison: dict) -> str:
    reference = comparison["reference"]
    targets = comparison["targets"]
    lines = [
        f"# Accuracy vs published Fugu scores — run {comparison['run_id']}",
        "",
        f"Published values transcribed from {reference['source_url']} on "
        f"{reference['retrieved_on']} (the page publishes them as images, so they "
        "cannot be fetched programmatically).",
        "",
        f"`Δ` is measured minus published **{comparison['delta_against']}**. "
        "`—` = not measured, `*` = partial cell or partial denominator, "
        "`!` = failed, `n/c` = kairyu measures this row differently.",
        "",
    ]

    header = ["Benchmark"] + list(targets)
    published_columns: list[str] = []
    for row in comparison["rows"]:
        for model in row["published_models"]:
            if model not in published_columns:
                published_columns.append(model)
    header += published_columns + [f"Δ {target}" for target in targets]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" * len(header) + "|")

    for row in comparison["rows"]:
        cells = [row["display_name"]]
        cells += [_measured_text(row["measured"][target]) for target in targets]
        for model in published_columns:
            value = row["published"].get(model)
            cells.append("—" if value is None else f"{value:.1f}")
        cells += [
            _delta_text(row["deltas"][target], row["measured"][target], row["not_comparable"])
            for target in targets
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Reading this table", ""]
    for note in reference["footnotes"]:
        lines.append(f"- {note}")
    lines.append(
        "- Baselines "
        f"({', '.join(sorted(PROVIDER_REPORTED))}) are provider-reported under "
        "unknown conditions; treat them as orientation, not as a measurement "
        "made under this harness."
    )

    caveats = _caveats(comparison)
    if caveats:
        lines += ["", "## Why a delta may not mean parity", ""]
        lines += [f"- {text}" for text in caveats]

    notes = comparison.get("methodology_notes") or []
    if notes:
        lines += ["", "## Methodology notes from this run", ""]
        lines += [f"- {note}" for note in notes]
    lines.append("")
    return "\n".join(lines)


def _caveats(comparison: dict) -> list[str]:
    """Per-row reasons a delta is not a like-for-like comparison."""
    caveats: list[str] = []
    for row in comparison["rows"]:
        name = row["display_name"]
        if row["not_comparable"]:
            caveats.append(f"**{name}**: NOT COMPARABLE — {row['not_comparable']}.")
        variant = row["variant"]
        if variant:
            published_variant = ", ".join(
                f"{model} {score:.1f}" for model, score in variant["scores"].items()
            )
            caveats.append(
                f"**{name}**: the release also publishes a {variant['label']} "
                f"({published_variant}); the row above uses the headline table."
            )
        for target, values in row["measured"].items():
            if values["status"] in ("partial", "failed") and values["reason"]:
                caveats.append(
                    f"**{name}** × {target}: {values['status']} — {values['reason']}."
                )
            elif values["status"] == "skipped" and values["reason"]:
                caveats.append(f"**{name}** × {target}: not measured — {values['reason']}.")
    return caveats
