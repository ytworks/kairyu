"""Pinned dataset revisions: reproducibility of every scoreboard cell.

Published datasets move: `openai/mrcr` was corrected in December 2025 and HLE's
item count has shifted since release, so a score recorded against "whatever main
was that day" is not comparable with a published number or an earlier run.
"""

import re

import pytest

from evals.adapters import all_adapters, suite_adapters
from evals.pins import (
    DATASET_PINS,
    SECONDARY_PINS,
    apply_pins,
    is_commit_sha,
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
    assert pinned_revision("mrcr-v2", "openai/mrcr") == DATASET_PINS["mrcr-v2"][1]
    assert pinned_revision("mrcr-v2", "someone-else/mrcr-mirror") is None
    assert pinned_revision("mrcr-v2", None) is None
    assert pinned_revision("not-a-slot", "openai/mrcr") is None


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


def test_non_commit_declarations_are_replaced_by_the_pin():
    """`release_v6` is a config name; the Hub 404s on it as a revision."""
    from evals.adapters.base import AdapterInfo

    class Fake:
        info = AdapterInfo(
            name="livecodebench",
            display_name="LiveCodeBench",
            metric="pass@1",
            hf_dataset="livecodebench/code_generation_lite",
            hf_revision="release_v6",
        )

    adapter = Fake()
    apply_pins([adapter])
    assert adapter.info.hf_revision == DATASET_PINS["livecodebench"][1]
    assert is_commit_sha(adapter.info.hf_revision)
    assert not is_commit_sha("release_v6")


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
    [
        ("hle", "test"),
        ("gpqa-diamond", "train"),
        ("scicode", "test"),
        ("mrcr-v2", "train"),
        ("long-context-reasoning", "train"),
    ],
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


def test_charxiv_passes_the_pin_on_both_split_attempts(tmp_path, monkeypatch):
    """The val/validation fallback must not drop the pin."""
    import evals.hub as hub
    from evals.adapters.base import DownloadContext
    from evals.cache import BenchCache
    from evals.types import DatasetUnavailable

    seen: list[tuple[str, str | None]] = []

    def fake_rows(dataset, *, name=None, split, revision=None, gated=False):
        seen.append((split, revision))
        raise DatasetUnavailable("no such split")

    monkeypatch.setattr(hub, "load_hf_rows", fake_rows)
    adapter = all_adapters()["charxiv-reasoning"]
    with pytest.raises(DatasetUnavailable):
        adapter.normalize(DownloadContext(cache=BenchCache(tmp_path / "cache")))

    assert [split for split, _ in seen] == ["validation", "val"]
    assert {revision for _, revision in seen} == {adapter.info.hf_revision}


def test_scicode_golden_data_fetch_carries_the_pin(tmp_path, monkeypatch):
    import evals.hub as hub
    from evals.adapters.base import DownloadContext
    from evals.adapters.scicode import SciCodeAdapter
    from evals.cache import BenchCache

    seen: list[dict] = []

    def fake_download(repo_id, filename, dest, *, repo_type="dataset", revision=None):
        seen.append({"repo": repo_id, "file": filename, "revision": revision})
        return None

    monkeypatch.setattr(hub, "download_file", fake_download)
    monkeypatch.setattr(hub, "load_hf_rows", lambda *a, **k: [])
    monkeypatch.setattr(SciCodeAdapter, "_fetch_provided_steps", lambda self: {})
    adapter = all_adapters()["scicode"]
    # an empty split fails closed on the sub-step count; the golden-data fetch
    # has already happened by then, which is what this test is about
    with pytest.raises(Exception, match="expected 291"):
        adapter.normalize(DownloadContext(cache=BenchCache(tmp_path / "cache")))

    assert seen and seen[0]["file"] == "test_data.h5"
    assert seen[0]["revision"] == adapter.info.hf_revision


def test_livecodebench_shard_fetch_carries_the_pin(tmp_path, monkeypatch):
    import evals.hub as hub
    from evals.adapters.base import DownloadContext
    from evals.cache import BenchCache

    seen: list[dict] = []

    def fake_jsonl(repo_id, filenames, *, revision=None, gated=False):
        seen.append({"repo": repo_id, "revision": revision})
        return []

    monkeypatch.setattr(hub, "load_jsonl_files", fake_jsonl)
    adapter = all_adapters()["livecodebench"]
    with pytest.raises(Exception, match="expected 1055"):
        adapter.normalize(DownloadContext(cache=BenchCache(tmp_path / "cache")))
    assert seen[0]["revision"] == adapter.info.hf_revision
    assert _SHA.match(seen[0]["revision"])


def test_every_cache_backed_slot_has_a_revision():
    """No slot whose data kairyu downloads may track a moving `main`.

    Agentic slots are exempt because their datasets are fetched by the external
    harness (mini-swe-agent / Harbor / τ), which exposes no revision knob — a
    limitation recorded in docs/benchmarks.md rather than papered over.
    """
    unpinned = [
        adapter.info.name
        for adapter in all_adapters().values()
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
    from evals.adapters.mrcr import MrcrAdapter

    assert all_adapters()["mrcr-v2"].info.hf_revision is not None
    assert MrcrAdapter.info.hf_revision is None


@pytest.mark.parametrize("name", sorted(DATASET_PINS))
def test_pinned_slots_exist_in_a_suite(name):
    suite_slots = {
        adapter.info.name
        for suite in ("accuracy", "core", "quantization")
        for adapter in suite_adapters(suite)
    }
    assert name in suite_slots


def test_pin_change_moves_the_run_fingerprint(tmp_path, monkeypatch):
    """Repinning must refuse resume rather than reinterpret stored evidence."""
    from conftest import make_config

    from evals.cache import BenchCache
    from evals.runner import _adapter_identity, _run_fingerprint, _run_identity

    config = make_config(tmp_path, models=("m",), only=("mrcr-v2",))
    cache = BenchCache(tmp_path / "cache")

    def fingerprint() -> str:
        adapters = suite_adapters("accuracy", only=("mrcr-v2",))
        identities = [
            _adapter_identity(adapter, cache, offline_fixtures=False)
            for adapter in adapters
        ]
        return _run_fingerprint(_run_identity(config, identities))

    before = fingerprint()
    monkeypatch.setitem(DATASET_PINS, "mrcr-v2", ("openai/mrcr", "b" * 40))
    assert fingerprint() != before
