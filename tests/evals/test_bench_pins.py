"""Pinned revisions reach cache checks and external fetches."""

import pytest

from evals.adapters import all_adapters


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
