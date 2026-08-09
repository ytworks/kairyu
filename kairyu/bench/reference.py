"""Published frontier benchmark scores, for comparison against a local run.

## Provenance

The legacy Fugu table in ``PUBLISHED_SCORES`` was transcribed from the two
figures on <https://sakana.ai/fugu-release/> on **2026-07-25**:

- `/assets/fugu-release/benchmark-table.png` — the headline table (Fugu, Fugu
  Ultra, Opus 4.8, Gemini 3.1 Pro, GPT 5.5),
- `/assets/fugu-release/benchmark-fugu-grid.png` — the per-benchmark figure,
  which adds Fable 5 and Mythos Preview columns and reports HLE on its
  **text-only** subset.

The page publishes these values **as images**, so there is no machine-readable
source to fetch: the table cannot be scraped, and these constants are the
transcription. To refresh, re-read both images and update `RETRIEVED_ON`.

The six-model additions in ``FRONTIER_SCORE_RECORDS`` come from the provider
pages and technical report listed in ``REFERENCE_SOURCES``; each source records
its own publication and retrieval dates.

## What the published numbers are, and are not

The page states: *"All scores other than Fugu's are reported by the model
providers."* The baseline columns are therefore **self-reported by each
provider**, not re-measured by Sakana under one harness — the page marks them
`†`. Fugu's own SWE-Bench Pro row is marked `*`: *"We use the mini-swe-agent as
the scaffolding for this task."*

Comparing a local kairyu score against these is useful for orientation, never a
like-for-like measurement: the evaluation conditions behind a
provider-reported number are unknown, and several kairyu slots knowingly deviate
(substituted dataset, uncompiled checker, single attempt). Those deviations
travel with each cell as annotations and are reprinted in the comparison report.
"""

from __future__ import annotations

import hashlib
import json

SOURCE_URL = "https://sakana.ai/fugu-release/"
SOURCE_ASSETS = (
    "/assets/fugu-release/benchmark-table.png",
    "/assets/fugu-release/benchmark-fugu-grid.png",
)
RETRIEVED_ON = "2026-07-25"

#: Column order as published in the headline table.
HEADLINE_MODELS = ("Fugu", "Fugu Ultra", "Opus 4.8", "Gemini 3.1 Pro", "GPT 5.5")
#: Extra columns that appear only in the per-benchmark figure.
FIGURE_MODELS = ("Fable 5", "Mythos Preview")
#: Models whose published numbers the page attributes to the provider, not to
#: Sakana's own measurement.
PROVIDER_REPORTED = frozenset(
    HEADLINE_MODELS[2:]
    + FIGURE_MODELS
    + (
        "GPT-5.6 Sol",
        "DeepSeek-V4-Flash-0731",
        "Qwen3.8 MAX",
        "Kimi K3",
    )
)

#: Stable columns requested for the frontier-quality comparison.  A model stays
#: in the report even when no like-for-like public value exists for a row; that
#: absence is rendered as ``—`` and is never filled from an older model.
COMPARISON_MODELS = (
    "Fable 5",
    "GPT-5.6 Sol",
    "DeepSeek-V4-Flash-0731",
    "Qwen3.8 MAX",
    "Kimi K3",
    "Fugu",
)

# Source records are committed rather than fetched at report time.  This keeps
# old reports reproducible when a provider edits a launch page.  ``tier`` is
# deliberately visible: provider pages/papers are primary, while an article
# transcribing a provider image is secondary.
REFERENCE_SOURCES: dict[str, dict[str, object]] = {
    "fugu-release": {
        "title": "Fugu release",
        "url": SOURCE_URL,
        "publisher": "Sakana AI",
        "published_on": "2026-07-23",
        "retrieved_on": RETRIEVED_ON,
        "tier": "primary",
    },
    "openai-gpt-5-6": {
        "title": "Introducing GPT-5.6",
        "url": "https://openai.com/index/gpt-5-6/",
        "publisher": "OpenAI",
        "published_on": "2026-07-28",
        "retrieved_on": "2026-08-09",
        "tier": "primary",
    },
    "deepseek-updates": {
        "title": "DeepSeek API updates — DeepSeek-V4-Flash-0731",
        "url": "https://api-docs.deepseek.com/updates/",
        "publisher": "DeepSeek",
        "published_on": "2026-07-31",
        "retrieved_on": "2026-08-09",
        "tier": "primary",
    },
    "qwen-3-8-launch": {
        "title": "Qwen3.8 launch benchmark table",
        "url": "https://qwen.ai/blog?id=qwen3.8",
        "publisher": "Qwen",
        "published_on": "2026-08-03",
        "retrieved_on": "2026-08-09",
        "tier": "primary-image-transcription",
    },
    "kimi-k3-report": {
        "title": "Kimi K3: Technical Report",
        "url": "https://arxiv.org/abs/2607.24653",
        "publisher": "Moonshot AI",
        "published_on": "2026-07-29",
        "retrieved_on": "2026-08-09",
        "tier": "primary",
    },
}


def _record(
    model: str,
    score: float,
    source: str,
    *,
    condition: str,
    metric: str = "score_percent",
    comparable: bool = True,
    notes: str | None = None,
) -> dict[str, object]:
    return {
        "model": model,
        "score": score,
        "metric": metric,
        "condition": condition,
        "source": source,
        "comparable": comparable,
        "notes": notes,
    }

#: Verbatim from the release page. The page's own `*`/`†` markers are spelled
#: out here so they cannot be confused with this report's `*` (partial cell).
PAGE_FOOTNOTES = (
    "Release page: all scores other than Fugu's are reported by the model providers.",
    "Release page (*): Fugu uses mini-swe-agent as the scaffolding for SWE-Bench Pro.",
    "Release page (†): baseline scores are model provider-reported.",
    "Release page: Fable 5 and Mythos Preview are not in Fugu's agent pool; where "
    "both publish a score, the page reports the higher of the two.",
)

#: adapter name -> {published model -> score (percent)}
#: Headline-table values, plus figure-only columns where the figure has them.
PUBLISHED_SCORES: dict[str, dict[str, float]] = {
    "swe-bench-pro": {
        "Fugu": 59.0,
        "Fugu Ultra": 73.7,
        "Opus 4.8": 69.2,
        "Gemini 3.1 Pro": 54.2,
        "GPT 5.5": 58.6,
        "Fable 5": 80.0,
    },
    "terminal-bench": {
        "Fugu": 80.2,
        "Fugu Ultra": 82.1,
        "Opus 4.8": 74.6,
        "Gemini 3.1 Pro": 70.3,
        "GPT 5.5": 78.2,
        "Fable 5": 80.4,
    },
    "livecodebench": {
        "Fugu": 92.9,
        "Fugu Ultra": 93.2,
        "Opus 4.8": 87.8,
        "Gemini 3.1 Pro": 88.5,
        "GPT 5.5": 85.3,
        "Fable 5": 89.8,
    },
    "livecodebench-pro": {
        "Fugu": 87.8,
        "Fugu Ultra": 90.8,
        "Opus 4.8": 84.8,
        "Gemini 3.1 Pro": 82.9,
        "GPT 5.5": 88.4,
    },
    "hle": {
        "Fugu": 47.2,
        "Fugu Ultra": 50.0,
        "Opus 4.8": 49.8,
        "Gemini 3.1 Pro": 44.4,
        "GPT 5.5": 41.4,
    },
    "charxiv-reasoning": {
        "Fugu": 85.1,
        "Fugu Ultra": 86.6,
        "Opus 4.8": 84.2,
        "Gemini 3.1 Pro": 83.3,
        "GPT 5.5": 84.1,
        "Mythos Preview": 86.1,
    },
    "gpqa-diamond": {
        "Fugu": 95.5,
        "Fugu Ultra": 95.5,
        "Opus 4.8": 92.0,
        "Gemini 3.1 Pro": 94.3,
        "GPT 5.5": 93.6,
        "Mythos Preview": 94.6,
    },
    "scicode": {
        "Fugu": 60.1,
        "Fugu Ultra": 58.7,
        "Opus 4.8": 53.5,
        "Gemini 3.1 Pro": 58.9,
        "GPT 5.5": 56.1,
        "Fable 5": 60.2,
    },
    "tau-bench-banking": {
        "Fugu": 21.7,
        "Fugu Ultra": 20.6,
        "Opus 4.8": 20.6,
        "Gemini 3.1 Pro": 8.4,
        "GPT 5.5": 20.6,
    },
    "long-context-reasoning": {
        "Fugu": 74.7,
        "Fugu Ultra": 73.3,
        "Opus 4.8": 67.7,
        "Gemini 3.1 Pro": 72.7,
        "GPT 5.5": 74.3,
    },
    "mrcr-v2": {
        "Fugu": 86.6,
        "Fugu Ultra": 93.6,
        "Opus 4.8": 87.9,
        "Gemini 3.1 Pro": 84.9,
        "GPT 5.5": 94.8,
    },
}

#: Where the figure reports a different population than the headline table, the
#: variant is recorded rather than silently merged into one number.
PUBLISHED_VARIANTS: dict[str, dict[str, object]] = {
    "hle": {
        "label": "text-only subset (per-benchmark figure)",
        "scores": {
            "Fugu": 48.5,
            "Fugu Ultra": 50.0,
            "Opus 4.8": 45.7,
            "Gemini 3.1 Pro": 44.7,
            "GPT 5.5": 44.3,
            "Fable 5": 53.3,
        },
    },
}

# Compact rows expand into artifact-friendly records below. Equal benchmark names
# do not imply equal tools, attempts, or populations, so conditions stay explicit.
_FRONTIER_ROWS: dict[str, tuple[tuple[str, float, str, str], ...]] = {
    "swe-bench-pro": (
        ("Fable 5", 80.0, "openai-gpt-5-6", "provider baseline in GPT-5.6 table"),
        ("GPT-5.6 Sol", 64.6, "openai-gpt-5-6", "launch-table condition"),
        ("Qwen3.8 MAX", 67.7, "qwen-3-8-launch", "launch-table condition"),
    ),
    "terminal-bench": (
        ("Fable 5", 83.1, "openai-gpt-5-6", "Terminal-Bench 2.1 provider baseline"),
        ("GPT-5.6 Sol", 88.8, "openai-gpt-5-6", "Terminal-Bench 2.1"),
        (
            "DeepSeek-V4-Flash-0731",
            82.7,
            "deepseek-updates",
            "Terminal-Bench 2.1; max effort; temperature=1; top_p=.95",
        ),
        ("Qwen3.8 MAX", 86.6, "qwen-3-8-launch", "Terminal-Bench 2.1"),
        ("Kimi K3", 88.3, "kimi-k3-report", "Terminal-Bench 2.1; max; top_p=1"),
    ),
    "livecodebench": (
        ("Fable 5", 89.8, "fugu-release", "provider baseline in Fugu figure"),
    ),
    "hle": (
        ("Fable 5", 53.3, "kimi-k3-report", "HLE-Full; no tools"),
        ("GPT-5.6 Sol", 44.5, "kimi-k3-report", "HLE-Full; no tools"),
        ("Qwen3.8 MAX", 43.6, "qwen-3-8-launch", "launch-table condition"),
        ("Kimi K3", 43.5, "kimi-k3-report", "HLE-Full; no tools; max; temperature=1"),
    ),
    "charxiv-reasoning": (
        ("Fable 5", 88.9, "kimi-k3-report", "CharXiv RQ; no tools"),
        ("GPT-5.6 Sol", 84.6, "kimi-k3-report", "CharXiv RQ; no tools"),
        ("Kimi K3", 84.8, "kimi-k3-report", "CharXiv RQ; no tools; max"),
    ),
    "gpqa-diamond": (
        ("Fable 5", 92.6, "openai-gpt-5-6", "provider baseline in GPT-5.6 table"),
        ("GPT-5.6 Sol", 94.6, "openai-gpt-5-6", "launch-table condition"),
        ("Kimi K3", 93.5, "kimi-k3-report", "max; temperature=1; top_p=.95"),
    ),
    "scicode": (
        ("Fable 5", 60.2, "kimi-k3-report", "SciCode"),
        ("GPT-5.6 Sol", 56.1, "kimi-k3-report", "SciCode"),
        ("Kimi K3", 58.7, "kimi-k3-report", "SciCode; max; temperature=1; top_p=.95"),
    ),
    "tau-bench-banking": (
        ("Fable 5", 26.8, "kimi-k3-report", "tau3-bench Banking"),
        ("GPT-5.6 Sol", 33.0, "kimi-k3-report", "tau3-bench Banking"),
        ("Kimi K3", 33.4, "kimi-k3-report", "tau3-bench Banking; max; top_p=1"),
    ),
}

FRONTIER_SCORE_RECORDS: dict[str, tuple[dict[str, object], ...]] = {
    benchmark: (
        _record(
            "Fugu",
            scores["Fugu"],
            "fugu-release",
            condition="Sakana release-table condition; no common provider harness",
            notes="mini-swe-agent scaffolding" if benchmark == "swe-bench-pro" else None,
        ),
        *(
            _record(model, score, source, condition=condition)
            for model, score, source, condition in _FRONTIER_ROWS.get(benchmark, ())
        ),
    )
    for benchmark, scores in PUBLISHED_SCORES.items()
}

_VARIANT_ROWS: dict[str, tuple[tuple[str, float, str, str], ...]] = {
    "hle": (
        ("Fable 5", 63.0, "kimi-k3-report", "HLE-Full; with tools"),
        ("GPT-5.6 Sol", 58.0, "kimi-k3-report", "HLE-Full; with tools"),
        ("Kimi K3", 56.0, "kimi-k3-report", "HLE-Full; with tools"),
    ),
    "charxiv-reasoning": (
        ("Fable 5", 93.5, "kimi-k3-report", "CharXiv RQ; with tools"),
        ("GPT-5.6 Sol", 89.1, "kimi-k3-report", "CharXiv RQ; with tools"),
        ("Kimi K3", 91.3, "kimi-k3-report", "CharXiv RQ; with tools"),
    ),
    "gpqa-diamond": (
        ("GPT-5.6 Sol", 94.1, "kimi-k3-report", "value reproduced in Kimi K3 Table 2"),
    ),
    "mrcr-v2": (
        ("GPT-5.6 Sol", 91.5, "openai-gpt-5-6", "MRCR v2 256K–512K bucket"),
        ("GPT-5.6 Sol", 73.8, "openai-gpt-5-6", "MRCR v2 512K–1M bucket"),
    ),
}
FRONTIER_SCORE_VARIANTS = {
    benchmark: tuple(
        _record(model, score, source, condition=condition, comparable=False)
        for model, score, source, condition in rows
    )
    for benchmark, rows in _VARIANT_ROWS.items()
}

#: Published rows that kairyu deliberately measures differently. The per-cell
#: annotations carry the detail; this is the "do not read the delta as parity"
#: flag for the row as a whole.
NOT_COMPARABLE: dict[str, str] = {
    "long-context-reasoning": (
        "kairyu substitutes LongBench v2; Fugu's Long Context Reasoning suite "
        "is unpublished"
    ),
}


def published(benchmark: str) -> dict[str, float]:
    return dict(PUBLISHED_SCORES.get(benchmark, {}))


def published_models(benchmark: str) -> tuple[str, ...]:
    """Published columns for this row, headline order first."""
    scores = PUBLISHED_SCORES.get(benchmark, {})
    ordered = [model for model in HEADLINE_MODELS + FIGURE_MODELS if model in scores]
    return tuple(ordered)


def comparison_published(benchmark: str) -> dict[str, float]:
    """Return only the six requested frontier columns for one row."""

    records = FRONTIER_SCORE_RECORDS.get(benchmark, ())
    return {
        str(record["model"]): float(record["score"])
        for record in records
        if record.get("model") in COMPARISON_MODELS
    }


def comparison_records(benchmark: str) -> list[dict[str, object]]:
    """Return detached primary and variant records for artifact embedding."""

    return [
        {**record, "variant": False}
        for record in FRONTIER_SCORE_RECORDS.get(benchmark, ())
    ] + [
        {**record, "variant": True}
        for record in FRONTIER_SCORE_VARIANTS.get(benchmark, ())
    ]


def _catalog_payload() -> dict[str, object]:
    return {
        "models": list(COMPARISON_MODELS),
        "sources": REFERENCE_SOURCES,
        "scores": FRONTIER_SCORE_RECORDS,
        "variants": FRONTIER_SCORE_VARIANTS,
    }


def catalog_sha256() -> str:
    encoded = json.dumps(
        _catalog_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reference_metadata() -> dict:
    return {
        "source_url": SOURCE_URL,
        "source_assets": list(SOURCE_ASSETS),
        "retrieved_on": RETRIEVED_ON,
        "transcribed_from_images": True,
        "provider_reported_models": sorted(PROVIDER_REPORTED),
        "footnotes": list(PAGE_FOOTNOTES),
        "comparison_models": list(COMPARISON_MODELS),
        "sources": REFERENCE_SOURCES,
        "catalog_sha256": catalog_sha256(),
        "catalog_policy": (
            "committed primary/provider records; missing stays missing; variants are not "
            "silently promoted"
        ),
    }
