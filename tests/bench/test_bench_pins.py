"""Pinned dataset revisions: reproducibility of every scoreboard cell.

Unpinned datasets moved under the suite: `openai/mrcr` was corrected in December
2025 and HLE's item count has shifted since release, so a score recorded against
"whatever main was that day" is not comparable with Fugu's number or with an
earlier kairyu run.
"""

import re

import pytest

from kairyu.bench.adapters import all_adapters, suite_adapters
from kairyu.bench.pins import DATASET_PINS, apply_pins, pinned_revision

_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_every_pin_is_a_full_commit_sha():
    for name, (dataset, revision) in DATASET_PINS.items():
        assert _SHA.match(revision), f"{name}: {revision!r} is not a commit sha"
        assert "/" in dataset, f"{name}: {dataset!r} is not a dataset id"


def test_pin_requires_the_dataset_to_match():
    """A source swap must not silently inherit the old dataset's pin."""
    assert pinned_revision("mrcr-v2", "openai/mrcr") == DATASET_PINS["mrcr-v2"][1]
    assert pinned_revision("mrcr-v2", "someone-else/mrcr-mirror") is None
    assert pinned_revision("mrcr-v2", None) is None
    assert pinned_revision("not-a-slot", "openai/mrcr") is None


def test_registry_adapters_are_pinned():
    for name, (_, revision) in DATASET_PINS.items():
        assert all_adapters()[name].info.hf_revision == revision


def test_every_cache_backed_slot_has_a_revision():
    """No slot whose data kairyu downloads may track a moving `main`.

    Agentic slots are exempt because their datasets are fetched by the external
    harness (mini-swe-agent / Harbor / τ), which exposes no revision knob — a
    limitation recorded in docs/benchmarks.md rather than papered over.
    """
    unpinned = [
        adapter.info.name
        for adapter in suite_adapters("fugu")
        if adapter.info.hf_dataset is not None
        and adapter.info.hf_revision is None
        and not adapter.info.agentic
    ]
    assert unpinned == []


def test_adapter_declared_revisions_win():
    """An adapter that knows which commit holds its shards keeps its own pin."""

    class Fake:
        class info:  # noqa: N801 - stand-in for AdapterInfo
            name = "mrcr-v2"
            hf_dataset = "openai/mrcr"
            hf_revision = "0" * 40

    adapter = Fake()
    apply_pins([adapter])
    assert adapter.info.hf_revision == "0" * 40


def test_pins_do_not_mutate_the_class_attribute():
    """apply_pins() shadows per instance; a fresh class stays untouched."""
    from kairyu.bench.adapters.mrcr import MrcrAdapter

    assert all_adapters()["mrcr-v2"].info.hf_revision is not None
    assert MrcrAdapter.info.hf_revision is None


@pytest.mark.parametrize("name", sorted(DATASET_PINS))
def test_pinned_slots_exist_in_the_suite(name):
    assert name in {adapter.info.name for adapter in suite_adapters("fugu")}


def test_pin_change_moves_the_run_fingerprint(tmp_path, monkeypatch):
    """Repinning must refuse resume rather than reinterpret stored evidence."""
    from conftest import make_config

    from kairyu.bench.cache import BenchCache
    from kairyu.bench.runner import _adapter_identity, _run_fingerprint, _run_identity

    config = make_config(tmp_path, models=("m",), only=("mrcr-v2",))
    cache = BenchCache(tmp_path / "cache")

    def fingerprint() -> str:
        adapters = suite_adapters("fugu", only=("mrcr-v2",))
        identities = [
            _adapter_identity(adapter, cache, offline_fixtures=False)
            for adapter in adapters
        ]
        return _run_fingerprint(_run_identity(config, identities))

    before = fingerprint()
    monkeypatch.setitem(DATASET_PINS, "mrcr-v2", ("openai/mrcr", "b" * 40))
    assert fingerprint() != before
