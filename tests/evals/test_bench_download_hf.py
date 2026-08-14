"""Opt-in networked download test for the retained gated GPQA dataset."""

import pytest

from evals.adapters.base import DownloadContext
from evals.adapters.gpqa import GpqaDiamondAdapter
from evals.cache import BenchCache

pytestmark = pytest.mark.hf_hub


def test_gpqa_without_token_reports_gated(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ctx = DownloadContext(cache=BenchCache(tmp_path / "cache"))
    report = GpqaDiamondAdapter().download(ctx)
    assert report.status == "gated"
    assert "huggingface.co/datasets/Idavidrein/gpqa" in report.detail
