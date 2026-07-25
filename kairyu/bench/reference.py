"""Published Fugu-release scores, for comparison against a local run.

## Provenance

Every number below was transcribed from the two figures on
<https://sakana.ai/fugu-release/> on **2026-07-25**:

- `/assets/fugu-release/benchmark-table.png` — the headline table (Fugu, Fugu
  Ultra, Opus 4.8, Gemini 3.1 Pro, GPT 5.5),
- `/assets/fugu-release/benchmark-fugu-grid.png` — the per-benchmark figure,
  which adds Fable 5 and Mythos Preview columns and reports HLE on its
  **text-only** subset.

The page publishes these values **as images**, so there is no machine-readable
source to fetch: the table cannot be scraped, and these constants are the
transcription. To refresh, re-read both images and update `RETRIEVED_ON`.

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
PROVIDER_REPORTED = frozenset(HEADLINE_MODELS[2:] + FIGURE_MODELS)

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


def reference_metadata() -> dict:
    return {
        "source_url": SOURCE_URL,
        "source_assets": list(SOURCE_ASSETS),
        "retrieved_on": RETRIEVED_ON,
        "transcribed_from_images": True,
        "provider_reported_models": sorted(PROVIDER_REPORTED),
        "footnotes": list(PAGE_FOOTNOTES),
    }
