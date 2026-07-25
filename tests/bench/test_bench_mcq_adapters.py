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


def _mrcr_row(needles: int, *, content: str = "hi", n_chars: int | None = None) -> dict:
    row = {
        "prompt": json.dumps([{"role": "user", "content": content}]),
        "answer": "PREFIXok",
        "random_string_to_prepend": "PREFIX",
        "n_needles": needles,
    }
    if n_chars is not None:
        row["n_chars"] = n_chars
    return row


def _patch_mrcr_rows(monkeypatch, rows):
    import kairyu.bench.hub as hub

    monkeypatch.setattr(hub, "load_hf_rows", lambda *a, **k: rows)


def test_mrcr_normalize_keeps_only_the_fugu_slice(tmp_path, monkeypatch, capsys):
    """Fugu reports 8-needle at up to 128K; the split also ships 2/4-needle."""
    from kairyu.bench.adapters.mrcr import _MAX_CONTEXT_TOKENS

    over_budget = "x" * ((_MAX_CONTEXT_TOKENS + 10) * 4)
    _patch_mrcr_rows(
        monkeypatch,
        [
            _mrcr_row(2),  # wrong needle count
            _mrcr_row(4),  # wrong needle count
            _mrcr_row(8, content="keep me"),  # kept
            _mrcr_row(8, content=over_budget),  # over 128K
        ],
    )
    rows = MrcrAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))

    assert len(rows) == 1
    assert rows[0]["n_needles"] == 8
    assert rows[0]["messages"][0]["content"] == "keep me"
    assert rows[0]["est_prompt_tokens"] > 0
    # the excluded counts are reported, never silently dropped
    out = capsys.readouterr().out
    assert "kept 1/4" in out and "not 8-needle" in out


def test_mrcr_normalize_prefers_dataset_n_chars(tmp_path, monkeypatch):
    """A short-looking row whose upstream n_chars exceeds 128K is excluded."""
    from kairyu.bench.adapters.mrcr import _MAX_CONTEXT_TOKENS

    _patch_mrcr_rows(
        monkeypatch,
        [
            _mrcr_row(8, content="tiny", n_chars=(_MAX_CONTEXT_TOKENS + 100) * 4),
            _mrcr_row(8, content="tiny", n_chars=4000),
        ],
    )
    rows = MrcrAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))
    assert [row["est_prompt_tokens"] for row in rows] == [1001]


def test_mrcr_normalize_fails_closed_on_empty_slice(tmp_path, monkeypatch):
    _patch_mrcr_rows(monkeypatch, [_mrcr_row(2), _mrcr_row(4)])
    with pytest.raises(DatasetUnavailable, match="8-needle"):
        MrcrAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))


def test_mrcr_methodology_discloses_the_slice(tmp_path):
    methodology = MrcrAdapter().methodology(_ctx(tmp_path))
    assert methodology["needles"] == 8
    assert methodology["max_context_tokens"] == 131_072
    assert "n_needles == 8" in methodology["selection"]
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
