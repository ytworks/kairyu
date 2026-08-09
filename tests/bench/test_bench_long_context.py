"""RULER-style fixed-length NIAH curve."""

from __future__ import annotations

import json

import pytest
from conftest import make_config, make_target

from kairyu.bench.adapters import LONG_CONTEXT_ROW_ORDER, suite_adapters
from kairyu.bench.adapters.base import DownloadContext, RunContext
from kairyu.bench.adapters.ruler_niah import (
    CONTEXT_LENGTHS,
    SAMPLES_PER_LENGTH,
    RulerNiahAdapter,
    ruler_niah_grade,
)
from kairyu.bench.cache import BenchCache
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore
from kairyu.bench.types import BenchItem, ChatRequestSpec, SkipItem


class _WhitespaceEncoder:
    def encode(self, text: str) -> list[str]:
        return text.split()


def _ctx(tmp_path) -> RunContext:
    import httpx

    return RunContext(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        offline_fixtures=True,
    )


def _item(context_tokens: int = 4_096) -> BenchItem:
    return BenchItem(
        id="niah",
        payload={
            "messages": [{"role": "user", "content": "key-a = value-b"}],
            "answer": "value-b",
            "context_tokens": context_tokens,
            "needle_position": 0.5,
        },
    )


def test_long_context_suite_is_the_ordered_4k_to_1m_curve():
    assert CONTEXT_LENGTHS == (
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,
    )
    assert [adapter.info.name for adapter in suite_adapters("long-context")] == list(
        LONG_CONTEXT_ROW_ORDER
    )


def test_normalization_is_deterministic_exact_length_and_position_swept(
    tmp_path, monkeypatch
):
    encoder = _WhitespaceEncoder()
    monkeypatch.setattr(
        "kairyu.bench.adapters.ruler_niah._encoder", lambda: encoder
    )
    adapter = RulerNiahAdapter(4_096)

    first = adapter.normalize(DownloadContext(cache=BenchCache(tmp_path / "cache")))
    second = adapter.normalize(DownloadContext(cache=BenchCache(tmp_path / "other")))

    assert first == second
    assert len(first) == SAMPLES_PER_LENGTH
    assert [row["needle_position"] for row in first] == sorted(
        row["needle_position"] for row in first
    )
    assert first[0]["needle_position"] < 0.05
    assert first[-1]["needle_position"] > 0.95
    assert all(
        len(encoder.encode(row["messages"][0]["content"])) == 4_096
        for row in first
    )
    assert len({row["answer"] for row in first}) == SAMPLES_PER_LENGTH


@pytest.mark.parametrize("context_tokens", [4_096, 262_144])
def test_real_o200k_generator_covers_curve_endpoints(tmp_path, context_tokens):
    import tiktoken

    encoder = tiktoken.get_encoding("o200k_base")
    rows = RulerNiahAdapter(context_tokens).normalize(
        DownloadContext(cache=BenchCache(tmp_path / "cache"))
    )

    assert len(rows) == SAMPLES_PER_LENGTH
    assert all(
        len(encoder.encode(row["messages"][0]["content"])) == context_tokens
        for row in rows
    )


def test_context_limit_skips_whole_curve_point_without_truncation(tmp_path):
    adapter = RulerNiahAdapter(8_192)
    ctx = _ctx(tmp_path)

    skipped = adapter.build_request(
        _item(8_192), make_target(max_context_tokens=4_096), ctx
    )
    request = adapter.build_request(
        _item(8_192),
        make_target(max_context_tokens=8_192, max_output_tokens=100),
        ctx,
    )

    assert isinstance(skipped, SkipItem)
    assert "8192 tokens > target limit 4096" in skipped.reason
    assert isinstance(request, ChatRequestSpec)
    assert request.est_prompt_tokens == 8_192
    assert request.max_tokens == 32


def test_invalid_cached_context_count_is_skipped(tmp_path):
    adapter = RulerNiahAdapter(4_096)
    broken = _item().model_copy(
        update={"payload": {**_item().payload, "context_tokens": "4096"}}
    )

    verdict = adapter.build_request(broken, make_target(), _ctx(tmp_path))

    assert isinstance(verdict, SkipItem)
    assert "no valid context token count" in verdict.reason


async def test_strict_retrieval_scorer_and_methodology(tmp_path):
    adapter = RulerNiahAdapter(4_096)
    ctx = _ctx(tmp_path)

    assert ruler_niah_grade(" value-b\n", "value-b") == 1.0
    assert ruler_niah_grade("The value is value-b", "value-b") == 0.0
    result = await adapter.score(_item(), "value-b", ctx)
    methodology = adapter.methodology(ctx)

    assert result.score == 1.0
    assert methodology["context_tokens"] == 4_096
    assert methodology["official_ruler_comparable"] is False
    assert methodology["token_scope"].startswith("user-message content")


def test_every_curve_point_has_an_offline_smoke_fixture(tmp_path):
    ctx = _ctx(tmp_path)
    for adapter in suite_adapters("long-context"):
        items = adapter.load_items(ctx)
        assert len(items) == 1
        assert items[0].payload["answer"].startswith("value-")


async def test_offline_curve_run_uses_the_standard_scoreboard_path(
    tmp_path, http_factory
):
    config = make_config(tmp_path, models=("m",), suite="long-context")
    runner = SuiteRunner(
        config,
        http_factory=http_factory,
        probe_docker=lambda: (False, "not needed"),
    )

    assert await runner.run() == 0
    store = ResultStore(tmp_path / "results", "test-run")
    scoreboard = json.loads((store.run_dir / "scoreboard.json").read_text())
    assert scoreboard["benchmarks"] == list(LONG_CONTEXT_ROW_ORDER)
    for benchmark in LONG_CONTEXT_ROW_ORDER:
        pair = store.load_pair(benchmark, "m")
        assert pair.status == "completed"
        assert pair.metrics["n_total"] == 1


def test_invalid_curve_length_is_rejected():
    with pytest.raises(ValueError, match="unsupported NIAH context length"):
        RulerNiahAdapter(12_345)
