"""Pinned dataset revisions for the packaged benchmark suites.

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

import re
from dataclasses import replace

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# adapter name -> (expected dataset id, commit sha)
# The dataset id is part of the record so a pin can never silently attach to a
# different dataset after a source swap.
DATASET_PINS: dict[str, tuple[str, str]] = {
    # Cheap deterministic core suite. Counts at these revisions: 1,319 test
    # rows (GSM8K), 14,042 test rows across 57 subjects (MMLU), and 541 prompts
    # carrying 834 verifiable instructions (IFEval).
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84"),
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
    # `release_v6` is this repo's CONFIG name, not a git ref: its refs endpoint
    # lists only `main` and /revision/release_v6 is a 404. The config name goes
    # to `name=`; the revision must be a commit.
    "livecodebench": (
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
    ),
    # Corrected upstream in December 2025; unpinned runs straddle that change.
    "mrcr-v2": ("openai/mrcr", "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"),
    "long-context-reasoning": (
        "THUDM/LongBench-v2",
        "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
    ),
}


#: Secondary artifacts that decide a slot's tests or expected answers. The
#: adapters own the fetching (and, for a raw file, its content hash); this is the
#: single place that answers "what data produced this scoreboard", and the
#: adapters' `extra_sources` must agree with it.
SECONDARY_PINS: dict[str, tuple[tuple[str, str], ...]] = {
    # The data alone does not define IFEval: the 25 deterministic checkers and
    # the English Punkt parameters are also score-bearing inputs. The adapter
    # records and verifies the Punkt archive's content digest in addition to
    # this immutable repository commit.
    "ifeval": (
        (
            "google-research/google-research:instruction_following_eval",
            "066e1eda43f4785922e3994e95429e496080231f",
        ),
        (
            "nltk/nltk_data:packages/tokenizers/punkt_tab.zip",
            "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a",
        ),
    ),
    "livecodebench-pro": (
        ("QAQAQAQAQ/LiveCodeBench-Pro-Testcase", "5257736c0a4e30ba0949d41c56a257c323d9c600"),
    ),
}


def is_commit_sha(revision: str | None) -> bool:
    return bool(revision and _COMMIT_SHA.match(revision))


def pinned_revision(adapter_name: str, dataset: str | None) -> str | None:
    """The pin for this adapter, or None when it has none / the dataset differs."""
    entry = DATASET_PINS.get(adapter_name)
    if entry is None or dataset is None:
        return None
    expected_dataset, revision = entry
    return revision if expected_dataset == dataset else None


def apply_pins(adapters: list) -> list:
    """Fill each adapter's `hf_revision` from the pin table where it is not a commit.

    An adapter that declares a real commit keeps it: it may know which shard set
    exists at which revision. A declared value that is NOT a commit sha (a config
    name, a branch) is replaced, because `revision` is passed to the Hub as a git
    ref and a non-ref there cannot be fetched.
    """
    for adapter in adapters:
        info = adapter.info
        if is_commit_sha(info.hf_revision):
            continue
        revision = pinned_revision(info.name, info.hf_dataset)
        if revision is not None:
            adapter.info = replace(info, hf_revision=revision)
    return adapters
