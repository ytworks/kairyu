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


def _complete_archive(cases: int = 1) -> bytes:
    entries = {}
    for index in range(1, cases + 1):
        entries[f"testdata/{index}.in"] = f"{index} {index}\n"
        entries[f"testdata/{index}.ans"] = f"{index * 2}\n"
    config = (
        "type: default\ninput_suffix: .in\noutput_suffix: .ans\n"
        f"subtasks:\n  - score: 100\n    n_cases: {cases}\n"
    )
    return _zip_bytes(entries, config=config)


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


def _full_split(count: int = 167) -> list[dict]:
    return [_pro_row(f"p{index:04d}") for index in range(count)]


def test_livecodebench_pro_normalize_joins_zip_testcases(tmp_path, monkeypatch):
    problems = _full_split()
    archives = {f"{row['problem_id']}.zip": _complete_archive() for row in problems}
    seen = _patch_pro_hub(monkeypatch, problems, archives)

    ctx = _ctx(tmp_path)
    rows = LiveCodeBenchProAdapter().normalize(ctx)

    assert seen["split"] == "quater_2025_4_6"  # Fugu's 2025 Q2 slice
    assert seen["gated"] is True
    assert len(seen["revision"]) == 40 and len(seen["testcase_revision"]) == 40
    assert len(rows) == 167  # the full published slice, never a subset
    assert rows[0]["id"] == "lcb-pro-p0000"
    assert rows[0]["tests"] == [{"input": "1 1\n", "output": "2\n", "testtype": "stdin"}]
    assert rows[0]["declared_cases"] == 1
    assert rows[0]["fn_name"] is None
    # the downloaded archives are not left behind in the cache
    assert not list((ctx.cache.assets_dir("livecodebench-pro")).glob("*.zip"))


def test_livecodebench_pro_fails_closed_on_a_short_split(tmp_path, monkeypatch):
    problems = _full_split(160)
    _patch_pro_hub(
        monkeypatch,
        problems,
        {f"{row['problem_id']}.zip": _complete_archive() for row in problems},
    )
    with pytest.raises(DatasetUnavailable, match="expected 167"):
        LiveCodeBenchProAdapter().normalize(_ctx(tmp_path))


def test_livecodebench_pro_fails_closed_on_a_missing_archive(tmp_path, monkeypatch):
    """download_file() maps a timeout, a 401 and a 404 all to None."""
    problems = _full_split()
    archives = {f"{row['problem_id']}.zip": _complete_archive() for row in problems}
    archives.pop("p0100.zip")
    _patch_pro_hub(monkeypatch, problems, archives)
    with pytest.raises(DatasetUnavailable, match="no usable archive for problem p0100"):
        LiveCodeBenchProAdapter().normalize(_ctx(tmp_path))


def test_livecodebench_pro_fails_closed_on_an_incomplete_archive(tmp_path, monkeypatch):
    """config.yaml declares n_cases; fewer usable cases means a partial fetch."""
    problems = _full_split()
    archives = {f"{row['problem_id']}.zip": _complete_archive() for row in problems}
    archives["p0007.zip"] = _zip_bytes(
        {"testdata/1.in": "1\n", "testdata/1.ans": "1\n"},
        config=(
            "input_suffix: .in\noutput_suffix: .ans\n"
            "subtasks:\n  - score: 100\n    n_cases: 60\n"
        ),
    )
    _patch_pro_hub(monkeypatch, problems, archives)
    with pytest.raises(DatasetUnavailable, match="incomplete: 1 usable cases, 60 declared"):
        LiveCodeBenchProAdapter().normalize(_ctx(tmp_path))


def test_livecodebench_pro_fails_closed_on_unpaired_cases(tmp_path, monkeypatch):
    problems = _full_split()
    archives = {f"{row['problem_id']}.zip": _complete_archive() for row in problems}
    archives["p0003.zip"] = _zip_bytes(
        {"testdata/1.in": "1\n", "testdata/1.ans": "1\n", "testdata/2.in": "2\n"},
        config=(
            "input_suffix: .in\noutput_suffix: .ans\n"
            "subtasks:\n  - score: 100\n    n_cases: 1\n"
        ),
    )
    _patch_pro_hub(monkeypatch, problems, archives)
    with pytest.raises(DatasetUnavailable, match="unpaired"):
        LiveCodeBenchProAdapter().normalize(_ctx(tmp_path))


def test_livecodebench_pro_normalize_reports_schema_drift(tmp_path, monkeypatch):
    rows = _full_split()
    rows[0] = {"question_id": "x", "problem_statement": "y"}
    _patch_pro_hub(monkeypatch, rows, {})
    with pytest.raises(DatasetUnavailable, match="format drift"):
        LiveCodeBenchProAdapter().normalize(_ctx(tmp_path))


def test_read_testcase_archive_reports_declared_cases():
    from kairyu.bench.adapters.livecodebench_pro import read_testcase_archive

    parsed = read_testcase_archive(_complete_archive(cases=3))
    assert parsed.declared_cases == 3
    assert len(parsed.tests) == 3
    assert parsed.unpaired == ()
    assert parsed.ignored_extra_cases == ()
    assert parsed.complete
    assert not read_testcase_archive(_zip_bytes({"README": "x"})).complete


def test_archive_uses_declared_numbered_cases_and_records_paired_extras():
    """Pinned archives may retain sample files above the judge denominator."""
    from kairyu.bench.adapters.livecodebench_pro import read_testcase_archive

    parsed = read_testcase_archive(
        _zip_bytes(
            {
                "testdata/1.in": "official\n",
                "testdata/1.ans": "OFFICIAL\n",
                "testdata/2.in": "sample\n",
                "testdata/2.ans": "SAMPLE\n",
            },
            config=(
                "input_suffix: .in\noutput_suffix: .ans\n"
                "subtasks:\n  - score: 100\n    n_cases: 1\n"
            ),
        )
    )

    assert parsed.tests == [
        {"input": "official\n", "output": "OFFICIAL\n", "testtype": "stdin"}
    ]
    assert parsed.ignored_extra_cases == ("2",)
    assert parsed.complete


def test_archive_does_not_substitute_an_extra_for_a_missing_declared_case():
    from kairyu.bench.adapters.livecodebench_pro import read_testcase_archive

    parsed = read_testcase_archive(
        _zip_bytes(
            {
                "testdata/1.in": "one\n",
                "testdata/1.ans": "ONE\n",
                "testdata/3.in": "extra\n",
                "testdata/3.ans": "EXTRA\n",
            },
            config=(
                "input_suffix: .in\noutput_suffix: .ans\n"
                "subtasks:\n  - score: 100\n    n_cases: 2\n"
            ),
        )
    )

    assert len(parsed.tests) == 1
    assert parsed.ignored_extra_cases == ("3",)
    assert not parsed.complete


@pytest.mark.parametrize(
    "config",
    [
        None,  # no config.yaml at all
        "type: default\ninput_suffix: .in\noutput_suffix: .ans\n",  # no subtasks
        "subtasks:\n  - score: 100\n",  # subtasks without n_cases
        "subtasks: not-a-list\n",  # malformed
    ],
)
def test_archive_without_a_declared_count_is_never_complete(config):
    """`sum(subtasks[].n_cases)` is the only denominator evidence there is.

    Treating "as many cases as arrived" as complete would let a truncated or
    schema-drifted archive pass as a valid benchmark.
    """
    from kairyu.bench.adapters.livecodebench_pro import read_testcase_archive

    data = _zip_bytes({"testdata/1.in": "a\n", "testdata/1.ans": "A\n"}, config=config)
    parsed = read_testcase_archive(data)
    assert parsed.tests  # the case parsed fine...
    assert parsed.declared_cases is None
    assert not parsed.complete  # ...but nothing vouches for the count


def test_output_without_an_input_is_unpaired_too():
    """A case that exists but can never be run is drift in the other direction."""
    from kairyu.bench.adapters.livecodebench_pro import read_testcase_archive

    parsed = read_testcase_archive(
        _zip_bytes(
            {"testdata/1.in": "a\n", "testdata/1.ans": "A\n", "testdata/2.ans": "B\n"},
            config=(
                "input_suffix: .in\noutput_suffix: .ans\n"
                "subtasks:\n  - score: 100\n    n_cases: 1\n"
            ),
        )
    )
    assert parsed.unpaired == ("testdata/2.ans",)
    assert not parsed.complete


def test_livecodebench_pro_fails_closed_on_an_undeclared_archive(tmp_path, monkeypatch):
    problems = _full_split()
    archives = {f"{row['problem_id']}.zip": _complete_archive() for row in problems}
    archives["p0042.zip"] = _zip_bytes(
        {"testdata/1.in": "1\n", "testdata/1.ans": "1\n"}
    )  # no config.yaml
    _patch_pro_hub(monkeypatch, problems, archives)
    with pytest.raises(DatasetUnavailable, match="no n_cases declared"):
        LiveCodeBenchProAdapter().normalize(_ctx(tmp_path))


def test_livecodebench_pro_records_the_testcase_pin_in_cache_identity():
    """Repinning the archives must invalidate the cache, not just the docs."""
    from kairyu.bench.adapters.base import cache_pins
    from kairyu.bench.adapters.livecodebench_pro import _TESTCASE_REVISION

    info = LiveCodeBenchProAdapter().info
    pins = cache_pins(info)
    assert pins["sources"] == [
        ["QAQAQAQAQ/LiveCodeBench-Pro-Testcase", _TESTCASE_REVISION]
    ]
    assert len(_TESTCASE_REVISION) == 40


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
