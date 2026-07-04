"""Opt-in networked download tests (`pytest -m hf_hub tests/bench`).

Excluded from the default suite; they verify the real HF paths: gated
datasets produce the typed error without a token, an open dataset
normalizes into the cache, and re-download is an idempotent no-op.
"""

import pytest

from kairyu.bench.adapters.base import DownloadContext
from kairyu.bench.adapters.gpqa import GpqaDiamondAdapter
from kairyu.bench.adapters.scicode import SciCodeAdapter
from kairyu.bench.cache import BenchCache

pytestmark = pytest.mark.hf_hub


def test_gpqa_without_token_reports_gated(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ctx = DownloadContext(cache=BenchCache(tmp_path / "cache"))
    report = GpqaDiamondAdapter().download(ctx)
    assert report.status == "gated"
    assert "huggingface.co/datasets/Idavidrein/gpqa" in report.detail


def test_scicode_downloads_and_is_idempotent(tmp_path):
    cache = BenchCache(tmp_path / "cache")
    ctx = DownloadContext(cache=cache)
    adapter = SciCodeAdapter()

    report = adapter.download(ctx)
    assert report.status == "ok", report.detail
    assert cache.is_ready("scicode")
    rows = cache.read_rows("scicode")
    assert rows, "normalized rows expected"
    first = rows[0]
    assert {"id", "step_id", "function_header", "test_cases"} <= set(first)
    manifest = cache.read_manifest("scicode")
    assert manifest["dataset"] == "SciCode1/SciCode"
    assert manifest["rows"] == len(rows)

    again = adapter.download(ctx)
    assert again.status == "cached"
