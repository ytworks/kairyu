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

The eight-model comparison in ``FRONTIER_SCORE_RECORDS`` was refreshed from the
provider pages, model cards, technical report, and third-party leaderboards in
``REFERENCE_SOURCES`` on **2026-08-12**. Each selected value keeps its source,
source class, condition, and notes; alternate conditions are retained separately.

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
        "GLM-5.2",
        "Kimi K3",
    )
)

#: Stable columns requested for the frontier-quality comparison.  A model stays
#: in the report even when no like-for-like public value exists for a row; that
#: absence is rendered as ``—`` and is never filled from an older model.
COMPARISON_MODELS = (
    "Fugu",
    "Fugu Ultra",
    "Fable 5",
    "GPT-5.6 Sol",
    "DeepSeek-V4-Flash-0731",
    "Qwen3.8 MAX",
    "GLM-5.2",
    "Kimi K3",
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
        "retrieved_on": "2026-08-11",
        "tier": "primary",
    },
    "anthropic-fable-5": {
        "title": "Claude Fable 5 and Claude Mythos 5",
        "url": "https://www.anthropic.com/news/claude-fable-5-mythos-5",
        "publisher": "Anthropic",
        "published_on": "2026-06-09",
        "retrieved_on": "2026-08-11",
        "tier": "primary",
    },
    "anthropic-fable-5-system-card": {
        "title": "Claude Fable 5 & Claude Mythos 5 System Card",
        "url": (
            "https://www-cdn.anthropic.com/"
            "2f9323abbcc4abe219577539efe19a623c9ca2bd/"
            "Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf"
        ),
        "publisher": "Anthropic",
        "published_on": "2026-06-09",
        "retrieved_on": "2026-08-12",
        "tier": "primary",
    },
    "openai-swe-bench-verified-retirement": {
        "title": "Why SWE-bench Verified no longer measures frontier coding capabilities",
        "url": "https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/",
        "publisher": "OpenAI",
        "published_on": "2026-02-23",
        "retrieved_on": "2026-08-12",
        "tier": "primary",
    },
    "openai-gpt-5-6": {
        "title": "Introducing GPT-5.6",
        "url": "https://openai.com/index/gpt-5-6/",
        "publisher": "OpenAI",
        "published_on": "2026-07-09",
        "retrieved_on": "2026-08-11",
        "tier": "primary",
    },
    "deepseek-v4-flash-0731": {
        "title": "DeepSeek-V4-Flash-0731 model card",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
        "publisher": "DeepSeek",
        "published_on": "2026-07-31",
        "retrieved_on": "2026-08-11",
        "tier": "primary",
    },
    "qwen-3-8-launch": {
        "title": "Qwen3.8 launch benchmark table",
        "url": "https://qwen.ai/blog?id=qwen3.8",
        "publisher": "Qwen",
        "published_on": "2026-08-03",
        "retrieved_on": "2026-08-11",
        "tier": "primary-image-transcription",
    },
    "glm-5-2-card": {
        "title": "GLM-5.2 model card",
        "url": "https://huggingface.co/zai-org/GLM-5.2",
        "publisher": "Z.ai",
        "published_on": "2026-06-16",
        "retrieved_on": "2026-08-11",
        "tier": "primary",
    },
    "kimi-k3-card": {
        "title": "Kimi K3 model card",
        "url": "https://huggingface.co/moonshotai/Kimi-K3",
        "publisher": "Moonshot AI",
        "published_on": "2026-07-29",
        "retrieved_on": "2026-08-11",
        "tier": "primary",
    },
    "vals-fable-5": {
        "title": "Claude Fable 5 independent evaluation",
        "url": "https://www.vals.ai/models/anthropic_claude-fable-5",
        "publisher": "Vals AI",
        "published_on": "rolling leaderboard",
        "retrieved_on": "2026-08-11",
        "tier": "third-party",
    },
    "artificial-analysis-hle": {
        "title": "Humanity's Last Exam leaderboard",
        "url": "https://artificialanalysis.ai/evaluations/humanitys-last-exam",
        "publisher": "Artificial Analysis",
        "published_on": "rolling leaderboard",
        "retrieved_on": "2026-08-11",
        "tier": "third-party",
    },
    "artificial-analysis-scicode": {
        "title": "SciCode leaderboard",
        "url": "https://artificialanalysis.ai/evaluations/scicode",
        "publisher": "Artificial Analysis",
        "published_on": "rolling leaderboard",
        "retrieved_on": "2026-08-11",
        "tier": "third-party",
    },
    "artificial-analysis-tau3": {
        "title": "Tau3 Banking leaderboard",
        "url": "https://artificialanalysis.ai/evaluations/tau3-banking",
        "publisher": "Artificial Analysis",
        "published_on": "rolling leaderboard",
        "retrieved_on": "2026-08-11",
        "tier": "third-party",
    },
    "artificial-analysis-lcr": {
        "title": "Artificial Analysis Long Context Reasoning leaderboard",
        "url": "https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning",
        "publisher": "Artificial Analysis",
        "published_on": "rolling leaderboard",
        "retrieved_on": "2026-08-11",
        "tier": "third-party",
    },
    "artificial-analysis-glm-5-2": {
        "title": "GLM-5.2 intelligence analysis",
        "url": "https://artificialanalysis.ai/models/glm-5-2",
        "publisher": "Artificial Analysis",
        "published_on": "2026-06-16",
        "retrieved_on": "2026-08-11",
        "tier": "third-party",
    },
}


def _record(
    model: str,
    score: float,
    source: str,
    *,
    condition: str,
    source_class: str,
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
        "source_class": source_class,
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
    "swe-bench-verified": {"Fable 5": 95.0},
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

# Equal benchmark names do not imply equal tools, attempts, or populations, so
# the selected matrix keeps conditions explicit and variants stay separate.
_FRONTIER_ROWS: dict[str, tuple[dict[str, object], ...]] = {
    "swe-bench-pro": (
        _record(
            "Fable 5",
            80.3,
            "anthropic-fable-5",
            condition="provider launch evaluation; maximum reasoning",
            source_class="provider",
            notes="the published value has been disputed publicly",
        ),
        _record(
            "GPT-5.6 Sol",
            64.6,
            "openai-gpt-5-6",
            condition="provider launch evaluation; maximum reasoning",
            source_class="provider",
        ),
        _record(
            "Qwen3.8 MAX",
            67.7,
            "qwen-3-8-launch",
            condition="vendor-run launch-table evaluation",
            source_class="provider",
        ),
        _record(
            "GLM-5.2",
            62.1,
            "glm-5-2-card",
            condition="OpenHands; tailored prompt; temperature=1; top_p=1",
            source_class="provider",
        ),
    ),
    "swe-bench-verified": (
        _record(
            "Fable 5",
            95.0,
            "anthropic-fable-5-system-card",
            condition=(
                "SWE-bench Verified; standard configuration; mean of five trials; "
                "thinking blocks included"
            ),
            source_class="provider",
            notes=(
                "local kairyu runs use one trial; OpenAI retired this benchmark for "
                "frontier launch comparisons because of flawed tests and contamination"
            ),
        ),
    ),
    "terminal-bench": (
        _record(
            "Fable 5",
            88.0,
            "kimi-k3-card",
            condition="Terminal-Bench 2.1 provider baseline in the Kimi K3 table",
            source_class="provider",
        ),
        _record(
            "GPT-5.6 Sol",
            88.8,
            "openai-gpt-5-6",
            condition="Terminal-Bench 2.1; maximum reasoning",
            source_class="provider",
        ),
        _record(
            "DeepSeek-V4-Flash-0731",
            82.7,
            "deepseek-v4-flash-0731",
            condition="Terminal-Bench 2.1; max effort; temperature=1; top_p=.95",
            source_class="provider",
        ),
        _record(
            "Qwen3.8 MAX",
            86.6,
            "qwen-3-8-launch",
            condition="Terminal-Bench 2.1; provider launch evaluation",
            source_class="provider",
        ),
        _record(
            "GLM-5.2",
            81.0,
            "glm-5-2-card",
            condition="Terminal-Bench 2.1; Terminus-2 harness",
            source_class="provider",
            notes="the same model card reports 82.7 with the best reported harness",
        ),
        _record(
            "Kimi K3",
            88.3,
            "kimi-k3-card",
            condition="Terminal-Bench 2.1; Kimi Code harness; max; top_p=1",
            source_class="provider",
        ),
    ),
    "hle": (
        _record(
            "Fable 5",
            59.0,
            "anthropic-fable-5",
            condition="HLE-Full; no tools; provider launch evaluation",
            source_class="provider",
        ),
        _record(
            "GPT-5.6 Sol",
            49.5,
            "artificial-analysis-hle",
            condition="Artificial Analysis HLE evaluation; no tools",
            source_class="third_party",
            notes="no provider-published value was found for this condition",
        ),
        _record(
            "Qwen3.8 MAX",
            43.6,
            "qwen-3-8-launch",
            condition="HLE; no tools; provider launch evaluation",
            source_class="provider",
        ),
        _record(
            "GLM-5.2",
            40.5,
            "glm-5-2-card",
            condition="HLE; no tools; provider launch evaluation",
            source_class="provider",
        ),
        _record(
            "Kimi K3",
            43.5,
            "kimi-k3-card",
            condition="HLE-Full; no tools; max; temperature=1; top_p=.95",
            source_class="provider",
        ),
    ),
    "charxiv-reasoning": (
        _record(
            "Qwen3.8 MAX",
            93.5,
            "qwen-3-8-launch",
            condition="CharXiv Reasoning; provider launch evaluation",
            source_class="provider",
            notes="the launch table is ambiguous; 88.4 may be the matching condition",
        ),
        _record(
            "Kimi K3",
            84.8,
            "kimi-k3-card",
            condition="CharXiv RQ; no tools; max; three-run average",
            source_class="provider",
        ),
    ),
    "gpqa-diamond": (
        _record(
            "Fable 5",
            91.3,
            "anthropic-fable-5",
            condition="GPQA Diamond; provider launch evaluation",
            source_class="provider",
            notes="fallback handling materially changes reported results",
        ),
        _record(
            "GPT-5.6 Sol",
            94.6,
            "openai-gpt-5-6",
            condition="GPQA Diamond; maximum reasoning",
            source_class="provider",
        ),
        _record(
            "Qwen3.8 MAX",
            92.6,
            "qwen-3-8-launch",
            condition="GPQA Diamond; provider launch evaluation",
            source_class="provider",
        ),
        _record(
            "GLM-5.2",
            91.2,
            "glm-5-2-card",
            condition="GPQA Diamond; provider launch evaluation",
            source_class="provider",
            notes="Artificial Analysis independently reported 89.0",
        ),
        _record(
            "Kimi K3",
            93.5,
            "kimi-k3-card",
            condition="GPQA Diamond; max; temperature=1; top_p=.95",
            source_class="provider",
        ),
    ),
    "scicode": (
        _record(
            "Fable 5",
            60.2,
            "artificial-analysis-scicode",
            condition="Artificial Analysis SciCode evaluation",
            source_class="third_party",
        ),
        _record(
            "GLM-5.2",
            50.0,
            "artificial-analysis-glm-5-2",
            condition="Artificial Analysis Intelligence Index v4.1",
            source_class="third_party",
        ),
        _record(
            "Kimi K3",
            58.7,
            "kimi-k3-card",
            condition="Artificial Analysis snapshot as of 2026-07-23",
            source_class="provider",
            notes="AA snapshot reproduced by provider",
        ),
    ),
    "tau-bench-banking": (
        _record(
            "Qwen3.8 MAX",
            51.3,
            "artificial-analysis-tau3",
            condition="Artificial Analysis Tau3 Banking evaluation",
            source_class="third_party",
        ),
        _record(
            "GLM-5.2",
            27.0,
            "artificial-analysis-glm-5-2",
            condition="Artificial Analysis Intelligence Index v4.1",
            source_class="third_party",
        ),
        _record(
            "Kimi K3",
            33.4,
            "kimi-k3-card",
            condition="Artificial Analysis snapshot as of 2026-07-23",
            source_class="provider",
            notes="AA snapshot reproduced by provider; current AA value is 46.0",
        ),
    ),
    "long-context-reasoning": (
        _record(
            "GLM-5.2",
            71.0,
            "artificial-analysis-glm-5-2",
            condition="Artificial Analysis Long Context Reasoning",
            source_class="third_party",
        ),
        _record(
            "Kimi K3",
            74.7,
            "kimi-k3-card",
            condition="Artificial Analysis snapshot as of 2026-07-23",
            source_class="provider",
            notes="AA snapshot reproduced by provider; current AA value is 82.7",
        ),
    ),
    "mrcr-v2": (
        _record(
            "GPT-5.6 Sol",
            91.5,
            "openai-gpt-5-6",
            condition="MRCR v2; 256K-512K; eight needles",
            source_class="provider",
        ),
        _record(
            "Qwen3.8 MAX",
            92.9,
            "qwen-3-8-launch",
            condition="MRCR v2; 256K; eight needles",
            source_class="provider",
        ),
    ),
}

def _fugu_records(
    benchmark: str, scores: dict[str, float]
) -> tuple[dict[str, object], ...]:
    records = []
    for model in ("Fugu", "Fugu Ultra"):
        score = scores.get(model)
        if score is None:
            continue
        records.append(
            _record(
                model,
                score,
                "fugu-release",
                condition="Sakana release-table condition; no common provider harness",
                source_class="provider",
                notes=(
                    "mini-swe-agent scaffolding"
                    if benchmark == "swe-bench-pro"
                    else None
                ),
            )
        )
    return tuple(records)


FRONTIER_SCORE_RECORDS: dict[str, tuple[dict[str, object], ...]] = {
    benchmark: (*_fugu_records(benchmark, scores), *_FRONTIER_ROWS.get(benchmark, ()))
    for benchmark, scores in PUBLISHED_SCORES.items()
}

_VARIANT_ROWS: dict[str, tuple[dict[str, object], ...]] = {
    "terminal-bench": (
        _record(
            "Fable 5",
            84.6,
            "vals-fable-5",
            condition="Vals AI Terminal-Bench 2.1 evaluation",
            source_class="third_party",
            comparable=False,
        ),
        _record(
            "GPT-5.6 Sol",
            91.9,
            "openai-gpt-5-6",
            condition="Terminal-Bench 2.1; ultra; four parallel agents",
            source_class="provider",
            comparable=False,
        ),
        _record(
            "GLM-5.2",
            82.7,
            "glm-5-2-card",
            condition="Terminal-Bench 2.1; best reported harness",
            source_class="provider",
            comparable=False,
        ),
    ),
    "livecodebench": (
        _record(
            "Fable 5",
            89.8,
            "fugu-release",
            condition="aggregated value without a verified Fable publication",
            source_class="provider",
            comparable=False,
            notes="excluded from the selected matrix as unverified",
        ),
    ),
    "hle": (
        _record(
            "Fable 5",
            64.5,
            "anthropic-fable-5",
            condition="HLE-Full; with tools",
            source_class="provider",
            comparable=False,
        ),
        _record(
            "Fable 5",
            55.5,
            "artificial-analysis-hle",
            condition="Artificial Analysis HLE evaluation",
            source_class="third_party",
            comparable=False,
        ),
        _record(
            "Qwen3.8 MAX",
            56.2,
            "qwen-3-8-launch",
            condition="HLE; with tools",
            source_class="provider",
            comparable=False,
        ),
        _record(
            "GLM-5.2",
            54.7,
            "glm-5-2-card",
            condition="HLE; with tools",
            source_class="provider",
            comparable=False,
        ),
        _record(
            "Kimi K3",
            56.0,
            "kimi-k3-card",
            condition="HLE-Full; with tools; max; temperature=1",
            source_class="provider",
            comparable=False,
        ),
    ),
    "charxiv-reasoning": (
        _record(
            "Qwen3.8 MAX",
            88.4,
            "qwen-3-8-launch",
            condition="alternate interpretation of the launch-table CharXiv value",
            source_class="provider",
            comparable=False,
        ),
        _record(
            "Kimi K3",
            91.3,
            "kimi-k3-card",
            condition="CharXiv RQ; with Python tools; three-run average",
            source_class="provider",
            comparable=False,
        ),
    ),
    "gpqa-diamond": (
        _record(
            "Fable 5",
            93.2,
            "vals-fable-5",
            condition="Vals AI score including fallback responses",
            source_class="third_party",
            comparable=False,
            notes="Vals reports a 41.92% refusal rate",
        ),
    ),
    "tau-bench-banking": (
        _record(
            "Kimi K3",
            46.0,
            "artificial-analysis-tau3",
            condition="current Artificial Analysis Tau3 Banking evaluation",
            source_class="third_party",
            comparable=False,
        ),
    ),
    "long-context-reasoning": (
        _record(
            "Kimi K3",
            82.7,
            "artificial-analysis-lcr",
            condition="current Artificial Analysis Long Context Reasoning evaluation",
            source_class="third_party",
            comparable=False,
        ),
    ),
    "mrcr-v2": (
        _record(
            "GPT-5.6 Sol",
            73.8,
            "openai-gpt-5-6",
            condition="MRCR v2; 512K-1M; eight needles",
            source_class="provider",
            comparable=False,
        ),
    ),
}
FRONTIER_SCORE_VARIANTS = {
    benchmark: tuple(rows)
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
    """Return only the eight requested frontier columns for one row."""

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
