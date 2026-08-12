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


def test_mrcr_selects_exactly_the_corrected_official_slice(tmp_path):
    """The corrected real slice must retain 500 exact-token-selected rows.

    Repairs in the pinned December 2025 source moved a few rows across adjacent
    token boundaries, so the original 100-per-bin distribution is no longer an
    invariant.  Exact `o200k_base` selection and the total denominator remain.
    """
    from kairyu.bench.adapters.mrcr import MrcrAdapter, expected_rows

    cache = BenchCache(tmp_path / "cache")
    adapter = MrcrAdapter()
    report = adapter.download(DownloadContext(cache=cache))
    assert report.status == "ok", report.detail

    rows = cache.read_rows("mrcr-v2")
    assert len(rows) == expected_rows()
    assert {row["n_needles"] for row in rows} == {8}
    per_bin: dict[int, int] = {}
    for row in rows:
        per_bin[row["token_bin"]] = per_bin.get(row["token_bin"], 0) + 1
        assert row["total_tokens"] <= 131_072
        assert row["total_tokens"] >= row["prompt_tokens"]
    assert per_bin == {8192: 106, 16384: 96, 32768: 98, 65536: 100, 131072: 100}
