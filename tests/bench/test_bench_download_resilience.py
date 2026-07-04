"""B2: an un-typed normalize() error degrades one dataset, not the whole run."""

from kairyu.bench.adapters.base import AdapterInfo, DownloadContext, GenerativeAdapter
from kairyu.bench.cache import BenchCache


class _DriftedAdapter(GenerativeAdapter):
    info = AdapterInfo(name="drifted", display_name="Drifted", metric="accuracy")

    def normalize(self, ctx):
        raise KeyError("Question")  # schema drift: a renamed upstream column

    def build_request(self, item, target, ctx):  # pragma: no cover - not reached
        raise NotImplementedError

    async def score(self, item, response_text, ctx):  # pragma: no cover
        raise NotImplementedError


def test_untyped_normalize_error_degrades_to_unavailable(tmp_path):
    ctx = DownloadContext(cache=BenchCache(tmp_path / "cache"))
    report = _DriftedAdapter().download(ctx)  # must not raise
    assert report.status == "unavailable"
    assert "KeyError" in (report.detail or "")


def test_cache_is_invalidated_by_revision_change(tmp_path):
    # M6: a cache written at revision v1 must NOT be considered ready for v2, so
    # bumping the pin re-downloads instead of scoring stale rows.
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(
        "ds", [{"id": "1"}], {"dataset": "org/ds", "revision": "v1"}
    )
    assert cache.is_ready("ds")  # existence-only check (no pin) still true
    assert cache.is_ready("ds", "org/ds", "v1")  # matching pin
    assert not cache.is_ready("ds", "org/ds", "v2")  # bumped revision -> stale
    assert not cache.is_ready("ds", "org/other", "v1")  # different dataset
