"""Dataset download failures and cache-identity checks degrade locally."""

import json
from pathlib import Path

import httpx
import pytest

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
)
from kairyu.bench.cache import BenchCache
from kairyu.bench.types import (
    BenchTarget,
    ChatRequestSpec,
    ItemResult,
)


class _DriftedAdapter(GenerativeAdapter):
    info = AdapterInfo(name="drifted", display_name="Drifted", metric="accuracy")

    def normalize(self, ctx):
        raise KeyError("Question")  # schema drift: a renamed upstream column

    def build_request(self, item, target, ctx):  # pragma: no cover - not reached
        raise NotImplementedError

    async def score(self, item, response_text, ctx):  # pragma: no cover
        raise NotImplementedError


class _PinnedAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="pinned",
        display_name="Pinned",
        metric="accuracy",
        hf_dataset="org/pinned",
        hf_revision="rev-1",
    )

    def normalize(self, ctx):
        return [{"id": "1", "answer": "original"}]

    def build_request(self, item, target, ctx):
        return ChatRequestSpec(messages=({"role": "user", "content": item.id},))

    async def score(self, item, response_text, ctx):
        return ItemResult(item_id=item.id, status="completed", score=1.0)


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
    assert cache.is_ready("ds")  # no-pin compatibility still verifies the digest
    assert cache.is_ready("ds", "org/ds", "v1")  # matching pin
    assert not cache.is_ready("ds", "org/ds", "v2")  # bumped revision -> stale
    assert not cache.is_ready("ds", "org/other", "v1")  # different dataset


def test_existence_only_readiness_rejects_mutated_data_without_modifying_cache(tmp_path):
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(
        "ds", [{"id": "1"}], {"dataset": "org/ds", "revision": "v1"}
    )
    manifest_before = cache.manifest_path("ds").read_bytes()
    cache.data_path("ds").write_text('{"id": "mutated"}\n', encoding="utf-8")
    data_before = cache.data_path("ds").read_bytes()

    assert not cache.is_ready("ds")
    assert cache.manifest_path("ds").read_bytes() == manifest_before
    assert cache.data_path("ds").read_bytes() == data_before


@pytest.mark.parametrize(
    "digest",
    [
        pytest.param(None, id="missing"),
        pytest.param(123, id="not-a-string"),
        pytest.param("too-short", id="wrong-length"),
        pytest.param("g" * 64, id="non-hex"),
    ],
)
def test_existence_only_readiness_rejects_missing_or_malformed_digest(
    tmp_path, digest
):
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(
        "ds", [{"id": "1"}], {"dataset": "org/ds", "revision": "v1"}
    )
    manifest = json.loads(cache.manifest_path("ds").read_text(encoding="utf-8"))
    if digest is None:
        manifest.pop("sha256")
    else:
        manifest["sha256"] = digest
    cache.manifest_path("ds").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_before = cache.manifest_path("ds").read_bytes()
    data_before = cache.data_path("ds").read_bytes()

    assert not cache.is_ready("ds")
    assert cache.manifest_path("ds").read_bytes() == manifest_before
    assert cache.data_path("ds").read_bytes() == data_before


def test_readiness_fails_closed_for_invalid_manifest_json_without_rewriting(tmp_path):
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(
        "ds", [{"id": "1"}], {"dataset": "org/ds", "revision": "v1"}
    )
    cache.manifest_path("ds").write_text("{invalid", encoding="utf-8")
    manifest_before = cache.manifest_path("ds").read_bytes()
    data_before = cache.data_path("ds").read_bytes()

    assert not cache.is_ready("ds")
    assert cache.manifest_path("ds").read_bytes() == manifest_before
    assert cache.data_path("ds").read_bytes() == data_before


@pytest.mark.parametrize("blocked_file", ["manifest", "data"])
def test_readiness_fails_closed_for_unreadable_files(
    tmp_path, monkeypatch, blocked_file
):
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(
        "ds", [{"id": "1"}], {"dataset": "org/ds", "revision": "v1"}
    )
    blocked_path = (
        cache.manifest_path("ds") if blocked_file == "manifest" else cache.data_path("ds")
    )
    original_open = Path.open

    def fail_open(path, *args, **kwargs):
        if path == blocked_path:
            raise OSError("unreadable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)

    assert not cache.is_ready("ds")


def test_adapter_cache_checks_pass_pins_and_do_not_read_failed_manifest():
    class FailClosedCache:
        def __init__(self):
            self.calls = []

        def is_ready(self, *args):
            self.calls.append(args)
            return False

        def read_manifest(self, adapter):
            raise AssertionError("failed readiness must prevent manifest reads")

    cache = FailClosedCache()
    ctx = RunContext(cache=cache, http_factory=lambda: None)
    adapter = _PinnedAdapter()
    target = BenchTarget(base_url="http://example.test/v1", model="model")

    assert "dataset not in cache" in adapter.check_preconditions(target, ctx)
    assert "manifest" not in adapter.methodology(ctx)
    assert cache.calls == [
        ("pinned", "org/pinned", "rev-1"),
        ("pinned", "org/pinned", "rev-1"),
    ]


async def test_adapter_skips_cache_mutated_between_download_and_execution(tmp_path):
    cache = BenchCache(tmp_path / "cache")
    adapter = _PinnedAdapter()
    report = adapter.download(DownloadContext(cache=cache))
    assert report.status == "ok"
    cache.data_path("pinned").write_text(
        json.dumps({"id": "1", "answer": "mutated"}) + "\n", encoding="utf-8"
    )
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "answer"}}]}
        )

    ctx = RunContext(
        cache=cache,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    target = BenchTarget(base_url="http://example.test/v1", model="model")

    pair = await adapter.run(target, ctx)

    assert pair.status == "skipped"
    assert "dataset not in cache" in (pair.reason or "")
    assert requests == []
