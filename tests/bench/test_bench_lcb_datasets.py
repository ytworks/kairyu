"""LiveCodeBench(+Pro) dataset acquisition: pinned files, splits, ZIP testcases.

These paths were unreachable before: LCB passed a config name ("release_v6") as
a git revision on a repo that has only `main`, and LCB-Pro asked for a `train`
split and a tabular testcase repo that do not exist. Both slots therefore always
degraded to "dataset unavailable" and the scoreboard cell was permanently blank.
"""

import io
import json
import zipfile

import pytest

from kairyu.bench.adapters.base import DownloadContext
from kairyu.bench.adapters.livecodebench import (
    LiveCodeBenchAdapter,
    grade_code,
    normalize_output,
)
from kairyu.bench.adapters.livecodebench_pro import (
    LiveCodeBenchProAdapter,
    parse_testcase_zip,
)
from kairyu.bench.cache import BenchCache
from kairyu.bench.types import DatasetGated, DatasetUnavailable


def _ctx(tmp_path) -> DownloadContext:
    return DownloadContext(cache=BenchCache(tmp_path / "cache"))


def _lcb_row(question_id: str = "q1") -> dict:
    return {
        "question_id": question_id,
        "question_content": "add two ints",
        "starter_code": "",
        "metadata": json.dumps({}),
        "public_test_cases": json.dumps(
            [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]
        ),
        "private_test_cases": json.dumps([]),
        "contest_date": "2024-01-01",
    }


def _zip_bytes(entries: dict[str, str], config: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if config is not None:
            archive.writestr("config.yaml", config)
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


# -- output normalization ------------------------------------------------------


def test_normalize_output_strips_lines_and_trailing_blanks():
    assert normalize_output("1 \n2\t\n\n\n") == ["1", "2"]
    assert normalize_output("1\r\n2\r\n") == ["1", "2"]
    # line structure stays significant
    assert normalize_output("1 2") != normalize_output("1\n2")


def test_grade_code_accepts_trailing_whitespace_in_output():
    tests = [{"input": "2 3\n", "output": "5", "testtype": "stdin"}]
    solution = "a, b = map(int, input().split())\nprint(f'{a + b}  ')"
    passed, detail = grade_code(solution, tests, None)
    assert passed, detail


# -- LiveCodeBench: pinned shard files, counted rows ---------------------------


def test_livecodebench_normalize_reads_pinned_release_files(tmp_path, monkeypatch):
    import kairyu.bench.hub as hub
    from kairyu.bench.adapters import livecodebench

    seen: dict = {}

    def fake_load(repo_id, filenames, *, revision=None, gated=False):
        seen.update(repo_id=repo_id, filenames=tuple(filenames), revision=revision)
        return [_lcb_row("q1"), _lcb_row("q2")]

    monkeypatch.setattr(hub, "load_jsonl_files", fake_load)
    monkeypatch.setitem(livecodebench._RELEASE_ROWS, "release_v6", 2)

    rows = LiveCodeBenchAdapter().normalize(_ctx(tmp_path))

    assert seen["repo_id"] == "livecodebench/code_generation_lite"
    assert seen["filenames"] == (
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    )
    # a real commit, not the "release_v6" config name that has no git ref
    assert seen["revision"] == livecodebench._LCB_REVISION
    assert len(seen["revision"]) == 40
    assert [row["id"] for row in rows] == ["q1", "q2"]
    assert rows[0]["tests"] == [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]


def test_livecodebench_normalize_rejects_row_count_drift(tmp_path, monkeypatch):
    import kairyu.bench.hub as hub

    monkeypatch.setattr(
        hub, "load_jsonl_files", lambda *a, **k: [_lcb_row("only-one")]
    )
    with pytest.raises(DatasetUnavailable, match="expected 1055"):
        LiveCodeBenchAdapter().normalize(_ctx(tmp_path))


def test_livecodebench_download_degrades_on_count_drift(tmp_path, monkeypatch):
    """A count mismatch is data (a skipped cell), never a crashed suite run."""
    import kairyu.bench.hub as hub

    monkeypatch.setattr(hub, "load_jsonl_files", lambda *a, **k: [])
    report = LiveCodeBenchAdapter().download(_ctx(tmp_path))
    assert report.status == "unavailable"
    assert "expected 1055" in report.detail


# -- LiveCodeBench Pro: ZIP testcases ------------------------------------------


def test_parse_testcase_zip_orders_numeric_cases():
    data = _zip_bytes(
        {
            "testdata/2.in": "b\n",
            "testdata/2.ans": "B\n",
            "testdata/10.in": "c\n",
            "testdata/10.ans": "C\n",
            "testdata/1.in": "a\n",
            "testdata/1.ans": "A\n",
            "checker.cpp": "// testlib",
        },
        config="type: default\ninput_suffix: .in\noutput_suffix: .ans\n",
    )
    tests = parse_testcase_zip(data)
    # numeric order, not lexicographic ("10" must come last)
    assert [test["input"] for test in tests] == ["a\n", "b\n", "c\n"]
    assert [test["output"] for test in tests] == ["A\n", "B\n", "C\n"]
    assert {test["testtype"] for test in tests} == {"stdin"}


def test_parse_testcase_zip_honours_config_suffixes():
    data = _zip_bytes(
        {"testdata/1.input": "a\n", "testdata/1.output": "A\n"},
        config="input_suffix: .input\noutput_suffix: .output\n",
    )
    assert parse_testcase_zip(data) == [
        {"input": "a\n", "output": "A\n", "testtype": "stdin"}
    ]


def test_parse_testcase_zip_skips_unpaired_and_configless_archives():
    # no config.yaml -> observed defaults; an input without its .ans is dropped
    data = _zip_bytes({"testdata/1.in": "a\n", "testdata/2.in": "b\n", "testdata/1.ans": "A\n"})
    assert parse_testcase_zip(data) == [
        {"input": "a\n", "output": "A\n", "testtype": "stdin"}
    ]
    assert parse_testcase_zip(_zip_bytes({"README": "x"})) == []


def _pro_row(problem_id: str) -> dict:
    return {
        "problem_id": problem_id,
        "problem_title": "t",
        "difficulty": "easy",
        "problem_statement": f"solve {problem_id}",
        "platform": "codeforces",
        "link": "http://example.invalid",
        "time_limit": "2s",
        "memory_limit": "512m",
    }


def _patch_pro_hub(monkeypatch, rows, archives: dict[str, bytes]):
    import kairyu.bench.hub as hub

    seen: dict = {}

    def fake_rows(dataset, *, name=None, split, revision=None, gated=False):
        seen.update(dataset=dataset, split=split, revision=revision, gated=gated)
        return rows

    def fake_download(repo_id, filename, dest, *, repo_type="dataset", revision=None):
        seen.setdefault("testcase_revision", revision)
        data = archives.get(filename)
        if data is None:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    monkeypatch.setattr(hub, "load_hf_rows", fake_rows)
    monkeypatch.setattr(hub, "download_file", fake_download)
    return seen


def test_livecodebench_pro_normalize_joins_zip_testcases(tmp_path, monkeypatch):
    archive = _zip_bytes({"testdata/1.in": "2 3\n", "testdata/1.ans": "5\n"})
    seen = _patch_pro_hub(
        monkeypatch,
        [_pro_row("1983A"), _pro_row("1983B")],
        {"1983A.zip": archive},  # 1983B has no archive -> excluded, not fatal
    )

    ctx = _ctx(tmp_path)
    rows = LiveCodeBenchProAdapter().normalize(ctx)

    assert seen["split"] == "quater_2025_4_6"  # Fugu's 2025 Q2 slice
    assert seen["gated"] is True
    assert len(seen["revision"]) == 40 and len(seen["testcase_revision"]) == 40
    assert [row["id"] for row in rows] == ["lcb-pro-1983A"]
    assert rows[0]["question"] == "solve 1983A"
    assert rows[0]["tests"] == [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]
    assert rows[0]["fn_name"] is None
    # the downloaded archive is not left behind in the cache
    assert not list((ctx.cache.assets_dir("livecodebench-pro")).glob("*.zip"))


def test_livecodebench_pro_normalize_fails_closed_without_any_join(tmp_path, monkeypatch):
    _patch_pro_hub(monkeypatch, [_pro_row("1983A")], {})
    with pytest.raises(DatasetUnavailable, match="could be joined"):
        LiveCodeBenchProAdapter().normalize(_ctx(tmp_path))


def test_livecodebench_pro_normalize_reports_schema_drift(tmp_path, monkeypatch):
    _patch_pro_hub(monkeypatch, [{"question_id": "x", "problem_statement": "y"}], {})
    with pytest.raises(DatasetUnavailable, match="format drift"):
        LiveCodeBenchProAdapter().normalize(_ctx(tmp_path))


def test_livecodebench_pro_is_declared_gated(tmp_path, monkeypatch):
    """The Pro problem repo needs license acceptance + HF_TOKEN."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    adapter = LiveCodeBenchProAdapter()
    assert adapter.info.gated is True

    import kairyu.bench.hub as hub

    def refuse(dataset, *, name=None, split, revision=None, gated=False):
        if gated:
            raise DatasetGated(dataset)
        raise AssertionError("gated flag must reach the loader")

    monkeypatch.setattr(hub, "load_hf_rows", refuse)
    report = adapter.download(_ctx(tmp_path))
    assert report.status == "gated"
    assert "LiveCodeBench-Pro" in report.detail
