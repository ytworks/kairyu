"""Pinned dataset revisions for the Fugu suite.

A benchmark whose dataset tracks `main` is not reproducible: the published sets
DO move. `openai/mrcr` was corrected in December 2025, HLE's item count has
shifted since release, and SWE-Bench Pro received post-release test fixes. A
score recorded against "whatever main was that day" cannot be compared with
Fugu's number or with an earlier kairyu run.

Pins live here rather than in each `AdapterInfo` so that one file answers "what
data produced this scoreboard", and so an adapter that needs a specific revision
for its own reasons (e.g. a shard set that only exists at one commit) keeps
ownership of it — a pin here only fills in what the adapter left unset.

## Refreshing a pin

Pins are deliberately manual. To move one:

    curl -s https://huggingface.co/api/datasets/<dataset> | python -c \\
        'import json,sys; print(json.load(sys.stdin)["sha"])'

then update the entry below and note it in `PROGRESS.md`. Changing a pin changes
the run fingerprint, so stored runs are never silently reinterpreted; they are
refused for resume against the new pin.
"""

from __future__ import annotations

from dataclasses import replace

# adapter name -> (expected dataset id, commit sha)
# The dataset id is part of the record so a pin can never silently attach to a
# different dataset after a source swap.
DATASET_PINS: dict[str, tuple[str, str]] = {
    # 2,500 test items as of this commit — the count Fugu reports.
    "hle": ("cais/hle", "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"),
    "gpqa-diamond": ("Idavidrein/gpqa", "633f5ee89ab8ad4522a9f850766b73f62147ffdd"),
    "charxiv-reasoning": (
        "princeton-nlp/CharXiv",
        "f441eb632fc62f6f777830a0f47619e6e86459b0",
    ),
    "scicode": ("SciCode1/SciCode", "4510f6a6aa27c43fad7b43da2c59602a86e88480"),
    "livecodebench-pro": (
        "QAQAQAQAQ/LiveCodeBench-Pro",
        "adebffce047dddb7768a86bace6aea4f7425e3bc",
    ),
    # Corrected upstream in December 2025; unpinned runs straddle that change.
    "mrcr-v2": ("openai/mrcr", "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"),
    "long-context-reasoning": (
        "THUDM/LongBench-v2",
        "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
    ),
}


def pinned_revision(adapter_name: str, dataset: str | None) -> str | None:
    """The pin for this adapter, or None when it has none / the dataset differs."""
    entry = DATASET_PINS.get(adapter_name)
    if entry is None or dataset is None:
        return None
    expected_dataset, revision = entry
    return revision if expected_dataset == dataset else None


def apply_pins(adapters: list) -> list:
    """Fill each adapter's unset `hf_revision` from the pin table.

    An adapter that already declares a revision keeps it: it knows something the
    table does not (which shards exist at which commit, for instance).
    """
    for adapter in adapters:
        info = adapter.info
        if info.hf_revision is not None:
            continue
        revision = pinned_revision(info.name, info.hf_dataset)
        if revision is not None:
            adapter.info = replace(info, hf_revision=revision)
    return adapters
