"""HLE / LongBench v2 / MRCR adapters: scoring, context gating, degradation."""

import json

import pytest
from conftest import make_config, make_target

from kairyu.bench.adapters.base import DownloadContext, RunContext
from kairyu.bench.adapters.hle import HleAdapter
from kairyu.bench.adapters.longbench_v2 import LongBenchV2Adapter
from kairyu.bench.adapters.mrcr import MrcrAdapter, mrcr_grade
from kairyu.bench.cache import BenchCache
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore
from kairyu.bench.types import (
    BenchItem,
    ChatRequestSpec,
    DatasetUnavailable,
    SkipItem,
)


def _ctx(tmp_path, **overrides) -> RunContext:
    import httpx

    defaults = dict(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        offline_fixtures=True,
    )
    defaults.update(overrides)
    return RunContext(**defaults)


# --- HLE ------------------------------------------------------------------------


async def test_hle_mcq_exact_match(tmp_path):
    adapter = HleAdapter()
    ctx = _ctx(tmp_path)
    item = BenchItem(
        id="x",
        payload={"question": "Q", "answer": "B", "answer_type": "multipleChoice", "image": None},
    )
    right = await adapter.score(item, "reasoning... Answer: B", ctx)
    wrong = await adapter.score(item, "Answer: A", ctx)
    assert right.score == 1.0 and right.status == "completed"
    assert wrong.score == 0.0


async def test_hle_freeform_without_judge_is_unjudged(tmp_path):
    adapter = HleAdapter()
    ctx = _ctx(tmp_path, judge=None)
    item = BenchItem(
        id="x",
        payload={"question": "Q", "answer": "photosynthesis", "answer_type": "exactMatch",
                 "image": None},
    )
    result = await adapter.score(item, "photosynthesis", ctx)
    assert result.status == "unjudged"
    assert "judge" in result.error


def test_hle_image_item_skipped_on_non_vision_target(tmp_path):
    adapter = HleAdapter()
    ctx = _ctx(tmp_path)
    item = BenchItem(
        id="x",
        payload={
            "question": "Q",
            "answer": "A",
            "answer_type": "multipleChoice",
            "image": "data:image/png;base64,AAAA",
        },
    )
    verdict = adapter.build_request(item, make_target(supports_vision=False), ctx)
    assert isinstance(verdict, SkipItem)
    request = adapter.build_request(item, make_target(supports_vision=True), ctx)
    assert isinstance(request, ChatRequestSpec)
    content = request.messages[0]["content"]
    assert content[1]["type"] == "image_url"


async def test_hle_pair_is_partial_without_judge(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("hle",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    exit_code = await runner.run()
    assert exit_code == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("hle", "m")
    assert pair.status == "partial"  # 2 MCQ scored, 1 free-form unjudged
    assert pair.metrics["n_unjudged"] == 1
    assert "unjudgeable" in pair.reason


# --- LongBench v2 ----------------------------------------------------------------


def test_longbench_context_gate_skips_items(tmp_path):
    adapter = LongBenchV2Adapter()
    ctx = _ctx(tmp_path)
    item = BenchItem(
        id="x",
        payload={
            "question": "Q",
            "choices": ["a", "b", "c", "d"],
            "answer": "A",
            "context": "words " * 4000,  # ~5k estimated tokens
        },
    )
    small = adapter.build_request(item, make_target(max_context_tokens=1000), ctx)
    assert isinstance(small, SkipItem)
    assert "prompt tokens > target limit" in small.reason
    unlimited = adapter.build_request(item, make_target(), ctx)
    assert isinstance(unlimited, ChatRequestSpec)
    assert unlimited.est_prompt_tokens > 1000


async def test_longbench_scores_and_annotates(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("long-context-reasoning",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    await runner.run()
    pair = ResultStore(tmp_path / "results", "test-run").load_pair(
        "long-context-reasoning", "m"
    )
    assert pair.status == "completed"
    assert any("not directly comparable" in note.lower() for note in pair.annotations)
    assert "truncation_policy" in pair.methodology


# --- MRCR -------------------------------------------------------------------------


def test_mrcr_grade_official_semantics():
    assert mrcr_grade("PREFIXhello", "PREFIXhello", "PREFIX") == 1.0
    assert mrcr_grade("hello", "PREFIXhello", "PREFIX") == 0.0  # missing prepend
    partial = mrcr_grade("PREFIXhelo", "PREFIXhello", "PREFIX")
    assert 0.0 < partial < 1.0


def test_mrcr_context_gate(tmp_path):
    adapter = MrcrAdapter()
    ctx = _ctx(tmp_path)
    item = BenchItem(
        id="x",
        payload={
            "messages": [{"role": "user", "content": "words " * 4000}],
            "answer": "PREFIXok",
            "prepend": "PREFIX",
        },
    )
    gated = adapter.build_request(item, make_target(max_context_tokens=100), ctx)
    assert isinstance(gated, SkipItem)


class _FakeEncoder:
    """Stand-in for o200k_base: one token per character, exactly counted."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)


def _mrcr_row(needles: int, *, tokens: int = 5000, answer_tokens: int = 96) -> dict:
    """A row whose exact prompt+answer token count is `tokens + answer_tokens`."""
    return {
        "prompt": json.dumps([{"role": "user", "content": "x" * tokens}]),
        "answer": "P" * answer_tokens,
        "random_string_to_prepend": "P",
        "n_needles": needles,
        "n_chars": tokens,
    }


def _patch_mrcr(monkeypatch, rows):
    import kairyu.bench.adapters.mrcr as mrcr_mod
    import kairyu.bench.hub as hub

    monkeypatch.setattr(hub, "load_hf_rows", lambda *a, **k: rows)
    monkeypatch.setattr(mrcr_mod, "_encoder", _FakeEncoder)


def _official_slice() -> list[dict]:
    """100 8-needle rows in each official bin at or below 128K."""
    from kairyu.bench.adapters.mrcr import selected_bins

    rows = []
    for bound in selected_bins():
        for _ in range(100):
            rows.append(_mrcr_row(8, tokens=bound - 100, answer_tokens=100))
    return rows


def test_token_bin_matches_the_published_boundaries():
    from kairyu.bench.adapters.mrcr import expected_rows, selected_bins, token_bin

    assert token_bin(4095) is None  # below the first bin
    assert token_bin(4096) == 8192
    assert token_bin(8192) == 8192  # boundaries are inclusive upper bounds
    assert token_bin(8193) == 16384
    assert token_bin(131_072) == 131_072
    assert token_bin(131_073) == 262_144
    assert token_bin(2_000_000) is None
    assert selected_bins() == (8192, 16384, 32768, 65536, 131_072)
    assert expected_rows() == 500  # 100 samples per bin, per the dataset card


def test_count_tokens_includes_the_answer():
    from kairyu.bench.adapters.mrcr import count_tokens

    messages = [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "de"}]
    assert count_tokens(_FakeEncoder(), messages) == 5
    assert count_tokens(_FakeEncoder(), messages, "fgh") == 8


def test_mrcr_normalize_selects_the_official_bins(tmp_path, monkeypatch, capsys):
    """Fugu reports 8-needle up to 128K: the five bins at or below 131,072."""
    rows = _official_slice()
    rows += [_mrcr_row(2), _mrcr_row(4)]  # wrong needle counts
    rows += [_mrcr_row(8, tokens=200_000)]  # a longer official bin
    rows += [_mrcr_row(8, tokens=100, answer_tokens=10)]  # below the first bin
    _patch_mrcr(monkeypatch, rows)

    normalized = MrcrAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))

    assert len(normalized) == 500
    assert {row["n_needles"] for row in normalized} == {8}
    assert {row["token_bin"] for row in normalized} == {8192, 16384, 32768, 65536, 131_072}
    assert all(row["total_tokens"] <= 131_072 for row in normalized)
    out = capsys.readouterr().out
    assert "kept 500/" in out and "per-bin counts" in out


def test_mrcr_normalize_fails_closed_when_the_denominator_does_not_match(
    tmp_path, monkeypatch
):
    """A missing source row must not silently shrink the corrected slice."""
    _patch_mrcr(monkeypatch, _official_slice()[:-1])
    with pytest.raises(DatasetUnavailable, match="expected 500 rows"):
        MrcrAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))


def test_mrcr_accepts_corrected_rows_that_cross_adjacent_bins(
    tmp_path, monkeypatch, capsys
):
    """Source repairs may move rows between exact bins without changing the slice."""
    from kairyu.bench.adapters.mrcr import selected_bins

    bins = selected_bins()
    rows = []
    for bound in bins:
        count = {bins[0]: 99, bins[1]: 101}.get(bound, 100)
        for _ in range(count):
            rows.append(_mrcr_row(8, tokens=bound - 100, answer_tokens=100))
    assert len(rows) == 500  # the total alone would have passed
    _patch_mrcr(monkeypatch, rows)
    normalized = MrcrAdapter().normalize(
        DownloadContext(cache=BenchCache(tmp_path / "c"))
    )

    assert len(normalized) == 500
    assert "8192: 99" in capsys.readouterr().out


def test_mrcr_context_gate_uses_the_exact_prompt_count(tmp_path):
    """The official runner gates on exact o200k message tokens, not chars/4."""
    adapter = MrcrAdapter()
    ctx = _ctx(tmp_path)
    # chars/4 would estimate ~1 token; the exact count recorded is 400
    item = BenchItem(
        id="x",
        payload={
            "messages": [{"role": "user", "content": "hi"}],
            "answer": "PREFIXok",
            "prepend": "PREFIX",
            "prompt_tokens": 400,
            "est_prompt_tokens": 1,
        },
    )
    gated = adapter.build_request(item, make_target(max_context_tokens=100), ctx)
    assert isinstance(gated, SkipItem)
    assert "400" in gated.reason

    fits = adapter.build_request(item, make_target(max_context_tokens=500), ctx)
    assert isinstance(fits, ChatRequestSpec)
    assert fits.est_prompt_tokens == 400


def test_mrcr_context_gate_falls_back_for_legacy_rows(tmp_path):
    adapter = MrcrAdapter()
    item = BenchItem(
        id="x",
        payload={
            "messages": [{"role": "user", "content": "words " * 4000}],
            "answer": "PREFIXok",
            "prepend": "PREFIX",
        },
    )
    gated = adapter.build_request(item, make_target(max_context_tokens=100), _ctx(tmp_path))
    assert isinstance(gated, SkipItem)


def test_mrcr_normalize_requires_the_official_tokenizer(tmp_path, monkeypatch):
    """Approximating would select a different population than the bins define."""
    import kairyu.bench.adapters.mrcr as mrcr_mod
    import kairyu.bench.hub as hub

    monkeypatch.setattr(hub, "load_hf_rows", lambda *a, **k: _official_slice())
    monkeypatch.setattr(
        mrcr_mod,
        "_encoder",
        lambda: (_ for _ in ()).throw(DatasetUnavailable("needs tiktoken")),
    )
    with pytest.raises(DatasetUnavailable, match="tiktoken"):
        MrcrAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))


def test_mrcr_methodology_discloses_the_slice(tmp_path):
    methodology = MrcrAdapter().methodology(_ctx(tmp_path))
    assert methodology["needles"] == 8
    assert methodology["max_context_tokens"] == 131_072
    assert methodology["tokenizer"] == "o200k_base"
    assert methodology["selected_bins"] == [8192, 16384, 32768, 65536, 131_072]
    assert methodology["expected_rows"] == 500
    assert "prompt+answer" in methodology["selection"]
    assert "corrected pinned source" in methodology["selection"]
    assert any("8-needle" in note for note in MrcrAdapter().info.annotations)


async def test_mrcr_runs_end_to_end(tmp_path, http_factory):
    config = make_config(tmp_path, models=("m",), only=("mrcr-v2",))
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    exit_code = await runner.run()
    assert exit_code == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("mrcr-v2", "m")
    assert pair.status == "completed"
    # mock responses never carry the prepend string -> official grade is 0
    assert pair.score == 0.0


async def test_all_context_gated_pair_is_skipped(tmp_path, http_factory):
    config = make_config(
        tmp_path,
        models=(),
        targets=(make_target(model="m", max_context_tokens=1),),
        only=("mrcr-v2",),
    )
    runner = SuiteRunner(config, http_factory=http_factory, probe_docker=lambda: (False, "t"))
    exit_code = await runner.run()
    assert exit_code == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("mrcr-v2", "m")
    assert pair.status == "skipped"
    assert "skipped" in pair.reason
