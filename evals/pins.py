"""Pinned dataset revisions for the retained checkout-only eval suites.

Pins are deliberately manual. Updating one changes the run fingerprint, so a
stored run can never be silently resumed against different source data.
"""

from __future__ import annotations

import re
from dataclasses import replace

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONTENT_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")

# adapter name -> (expected dataset id, commit sha)
# The dataset id is part of the record so a pin can never silently attach to a
# different dataset after a source swap.
DATASET_PINS: dict[str, tuple[str, str]] = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84"),
    "gpqa-diamond": (
        "Idavidrein/gpqa",
        "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    ),
}


#: Secondary artifacts that decide a slot's tests or expected answers. The
#: adapters own the fetching (and, for a raw file, its content hash); this is the
#: single place that answers "what data produced this scoreboard", and the
#: adapters' `extra_sources` must agree with it.
SECONDARY_PINS: dict[str, tuple[tuple[str, str], ...]] = {
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
}


def is_commit_sha(revision: str | None) -> bool:
    return bool(revision and _COMMIT_SHA.match(revision))


def is_pinned_revision(dataset: str, revision: str | None) -> bool:
    if dataset.startswith("package:"):
        return bool(revision and _CONTENT_SHA.fullmatch(revision))
    return is_commit_sha(revision)


def _assert_pin_tables() -> None:
    for name, (dataset, revision) in DATASET_PINS.items():
        assert name
        assert "/" in dataset
        assert is_commit_sha(revision)
    for name, sources in SECONDARY_PINS.items():
        assert name and sources
        for source, revision in sources:
            assert source
            assert is_commit_sha(revision)


_assert_pin_tables()


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
