"""Pinned dataset revisions for retained checkout-only eval suites."""

import re

import pytest

from evals.adapters import all_adapters, suite_adapters
from evals.pins import (
    DATASET_PINS,
    SECONDARY_PINS,
    apply_pins,
    pinned_revision,
)

_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONTENT_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_every_pin_is_a_full_commit_sha():
    for name, (dataset, revision) in DATASET_PINS.items():
        assert _SHA.match(revision), f"{name}: {revision!r} is not a commit sha"
        assert "/" in dataset, f"{name}: {dataset!r} is not a dataset id"


def test_pin_requires_the_dataset_to_match():
    """A source swap must not silently inherit the old dataset's pin."""
    assert pinned_revision("gpqa-diamond", "Idavidrein/gpqa") == DATASET_PINS["gpqa-diamond"][1]
    assert pinned_revision("gpqa-diamond", "someone-else/gpqa") is None
    assert pinned_revision("gpqa-diamond", None) is None
    assert pinned_revision("not-a-slot", "Idavidrein/gpqa") is None


def test_registry_adapters_are_pinned():
    for name, (_, revision) in DATASET_PINS.items():
        assert all_adapters()[name].info.hf_revision == revision


def test_list_checks_cache_readiness_against_all_declared_pins(monkeypatch, capsys):
    from argparse import Namespace

    from evals.cache import BenchCache
    from evals.cli import _handle_list

    seen: dict[str, dict] = {}

    def record_ready(self, adapter, dataset=None, revision=None, sources=None):
        seen[adapter] = {
            "dataset": dataset,
            "revision": revision,
            "sources": sources,
        }
        return False

    monkeypatch.setattr(BenchCache, "is_ready", record_ready)

    assert _handle_list(Namespace(suite="core")) == 0
    assert set(seen) == {"gsm8k", "mmlu", "ifeval"}
    registry = all_adapters()
    for name, pins in seen.items():
        info = registry[name].info
        assert pins == {
            "dataset": info.hf_dataset,
            "revision": info.hf_revision,
            "sources": [list(source) for source in info.extra_sources],
        }
    assert "suite core (3 slots)" in capsys.readouterr().out


def test_every_pinned_slot_declares_a_source_appropriate_revision():
    """Hub sources use Git commits; package-owned corpora use content digests."""
    for adapter in all_adapters().values():
        revision = adapter.info.hf_revision
        if revision is None:
            continue
        if (adapter.info.hf_dataset or "").startswith("package:"):
            assert _CONTENT_SHA.match(revision), f"{adapter.info.name}: {revision!r}"
        else:
            assert _SHA.match(revision), f"{adapter.info.name}: {revision!r}"


def test_secondary_pins_match_the_adapters():
    """The registry answers "what data produced this scoreboard" completely."""
    registry = all_adapters()
    for name, sources in SECONDARY_PINS.items():
        declared = registry[name].info.extra_sources
        assert tuple(tuple(source) for source in declared) == sources, name
        for _, revision in sources:
            assert _SHA.match(revision)


def test_every_secondary_source_an_adapter_uses_is_registered():
    for adapter in all_adapters().values():
        for source in adapter.info.extra_sources:
            assert source in SECONDARY_PINS.get(adapter.info.name, ()), (
                f"{adapter.info.name} fetches {source[0]} without a registry entry"
            )


# -- the pin must reach the actual fetch ---------------------------------------


@pytest.mark.parametrize(
    ("name", "split_kwarg"),
    [("gpqa-diamond", "train")],
)
def test_normalize_passes_the_pinned_revision_to_the_hub(
    name, split_kwarg, tmp_path, monkeypatch
):
    """Recording a pin in the manifest while fetching `main` attests a lie."""
    import evals.hub as hub
    from evals.adapters.base import DownloadContext
    from evals.cache import BenchCache

    seen: list[dict] = []

    def fake_rows(dataset, *, name=None, split, revision=None, gated=False):
        seen.append({"dataset": dataset, "split": split, "revision": revision})
        raise RuntimeError("stop after recording the call")

    monkeypatch.setattr(hub, "load_hf_rows", fake_rows)
    monkeypatch.setattr(hub, "download_file", lambda *a, **k: None)
    monkeypatch.setenv("HF_TOKEN", "test-token")  # gated slots

    adapter = all_adapters()[name]
    with pytest.raises(RuntimeError, match="stop after recording"):
        adapter.normalize(DownloadContext(cache=BenchCache(tmp_path / "cache")))

    assert seen, f"{name} never called the hub"
    assert seen[0]["dataset"] == adapter.info.hf_dataset
    assert seen[0]["revision"] == adapter.info.hf_revision
    assert _SHA.match(seen[0]["revision"])


def test_every_cache_backed_slot_has_a_revision():
    """No retained downloaded dataset may track a moving branch."""
    unpinned = [
        adapter.info.name
        for adapter in all_adapters().values()
        if adapter.info.hf_dataset is not None
        and adapter.info.hf_revision is None
    ]
    assert unpinned == []


def test_adapter_declared_revisions_win():
    """An adapter that knows which commit holds its shards keeps its own pin."""

    class Fake:
        class info:  # noqa: N801 - stand-in for AdapterInfo
            name = "gpqa-diamond"
            hf_dataset = "Idavidrein/gpqa"
            hf_revision = "0" * 40

    adapter = Fake()
    apply_pins([adapter])
    assert adapter.info.hf_revision == "0" * 40


def test_pins_do_not_mutate_the_class_attribute():
    """apply_pins() shadows per instance; a fresh class stays untouched."""
    from evals.adapters.gpqa import GpqaDiamondAdapter

    assert all_adapters()["gpqa-diamond"].info.hf_revision is not None
    assert GpqaDiamondAdapter.info.hf_revision is None


@pytest.mark.parametrize("name", sorted(DATASET_PINS))
def test_pinned_slots_exist_in_a_suite(name):
    suite_slots = {
        adapter.info.name
        for suite in ("core", "quantization")
        for adapter in suite_adapters(suite)
    }
    assert name in suite_slots


def test_pin_change_moves_the_run_fingerprint(tmp_path, monkeypatch):
    """Repinning must refuse resume rather than reinterpret stored evidence."""
    from conftest import make_config

    from evals.cache import BenchCache
    from evals.runner import _adapter_identity, _run_fingerprint, _run_identity

    config = make_config(tmp_path, models=("m",), only=("gpqa-diamond",))
    cache = BenchCache(tmp_path / "cache")

    def fingerprint() -> str:
        adapters = suite_adapters("quantization", only=("gpqa-diamond",))
        identities = [
            _adapter_identity(adapter, cache, offline_fixtures=False)
            for adapter in adapters
        ]
        return _run_fingerprint(_run_identity(config, identities))

    before = fingerprint()
    monkeypatch.setitem(DATASET_PINS, "gpqa-diamond", ("Idavidrein/gpqa", "b" * 40))
    assert fingerprint() != before
