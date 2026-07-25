"""SciCode: sequential sub-steps, background prompts, golden-data provenance.

The published dataset ships no reference code (`ground_truth_code` and
`general_solution` are null for every row), so the previous behaviour — grade
each sub-step in isolation with an empty prior-code slot — could only raise
NameError on helpers an earlier step was meant to define. These tests pin the
sequential contract instead.
"""

import httpx
import pytest
from conftest import make_config, make_target

from kairyu.bench.adapters.base import DownloadContext, RunContext
from kairyu.bench.adapters.scicode import (
    SciCodeAdapter,
    group_by_problem,
    select_problem_groups,
)
from kairyu.bench.cache import BenchCache
from kairyu.bench.runner import SuiteRunner
from kairyu.bench.store import ResultStore
from kairyu.bench.types import BenchItem


def _ctx(tmp_path, **overrides) -> RunContext:
    defaults = dict(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        offline_fixtures=True,
    )
    defaults.update(overrides)
    return RunContext(**defaults)


def _item(problem: str, index: int) -> BenchItem:
    return BenchItem(
        id=f"{problem}.{index}",
        payload={
            "problem_id": problem,
            "step_index": index,
            "step_id": f"{problem}.{index}",
            "main_description": "d",
            "main_background": "",
            "dependencies": "",
            "step_description": "s",
            "step_background": "",
            "function_header": "def f():",
            "return_line": "",
            "test_cases": [],
        },
    )


# -- grouping and whole-problem limits -----------------------------------------


def test_group_by_problem_orders_steps():
    items = [_item("a", 2), _item("b", 0), _item("a", 0), _item("a", 1)]
    groups = {g[0].payload["problem_id"]: g for g in group_by_problem(items)}
    assert [i.payload["step_index"] for i in groups["a"]] == [0, 1, 2]
    assert len(groups["b"]) == 1


def test_select_problem_groups_keeps_whole_problems():
    groups = [[_item("a", i) for i in range(3)], [_item("b", i) for i in range(3)]]
    picked = select_problem_groups(groups, limit=4, seed=0)
    # 3 fits, 6 does not: one whole problem, never a truncated chain
    assert [len(group) for group in picked] == [3]
    assert select_problem_groups(groups, limit=None, seed=0) == groups
    assert select_problem_groups(groups, limit=99, seed=0) == groups


def test_select_problem_groups_always_yields_one_problem():
    """A limit smaller than the first problem still runs that problem."""
    groups = [[_item("a", i) for i in range(5)]]
    assert select_problem_groups(groups, limit=2, seed=0) == groups


def test_select_problem_groups_is_seed_deterministic():
    groups = [[_item(name, 0)] for name in "abcdef"]
    first = select_problem_groups(groups, limit=3, seed=7)
    again = select_problem_groups(groups, limit=3, seed=7)
    assert [g[0].id for g in first] == [g[0].id for g in again]


# -- prompt content ------------------------------------------------------------


def test_build_request_includes_background_and_prior_code(tmp_path):
    adapter = SciCodeAdapter()
    item = BenchItem(
        id="x",
        payload={
            "main_description": "compute things",
            "main_background": "the physics of things",
            "dependencies": "import numpy as np",
            "prior_code": "def helper():\n    return 1",
            "step_description": "write step two",
            "step_background": "Background: step two builds on helper",
            "function_header": "def step_two():",
            "return_line": "int",
        },
    )
    prompt = adapter.build_request(item, make_target(), _ctx(tmp_path)).messages[0]["content"]
    assert "the physics of things" in prompt  # Fugu's with-background condition
    assert "Background: step two builds on helper" in prompt
    assert "def helper():" in prompt  # the model's own earlier code
    assert "import numpy as np" in prompt


def test_build_request_omits_empty_background_sections(tmp_path):
    adapter = SciCodeAdapter()
    item = BenchItem(
        id="x",
        payload={
            "main_description": "d",
            "main_background": "",
            "dependencies": "",
            "step_description": "s",
            "step_background": "   ",
            "function_header": "def f():",
            "return_line": "",
        },
    )
    prompt = adapter.build_request(item, make_target(), _ctx(tmp_path)).messages[0]["content"]
    assert "BACKGROUND" not in prompt
    assert "# (none)" in prompt


# -- sequential execution ------------------------------------------------------


async def test_later_steps_see_earlier_generated_code(tmp_path):
    """Step 2 of the fixture calls step 1's helper; only a chained run can pass."""
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        prompt = _json.loads(request.content)["messages"][0]["content"]
        prompts.append(prompt)
        if "vector_norm(v)" in prompt and "normalize" not in prompt.split("Now implement")[1]:
            code = "def vector_norm(v):\n    return float(np.sqrt(np.sum(np.asarray(v) ** 2)))"
        else:
            code = "def normalize(v):\n    return np.asarray(v) / vector_norm(v)"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"```python\n{code}\n```"},
                    }
                ]
            },
        )

    config = make_config(tmp_path, models=("m",), only=("scicode",))
    runner = SuiteRunner(
        config,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        probe_docker=lambda: (False, "t"),
    )
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("scicode", "m")
    assert pair.status == "completed"
    assert pair.score == 1.0, [item.error for item in pair.items]
    # the second prompt carried the first step's generated implementation
    assert "def vector_norm(v):" in prompts[1]
    assert pair.methodology["limit_granularity"] == "whole problems"
    assert "sequential per problem" in pair.methodology["evaluation"]


async def test_failed_step_does_not_abort_the_problem(tmp_path):
    """A step whose reply has no code block scores 0; later steps still run."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        code_block = "```python\ndef normalize(v):\n    return v\n```"
        content = "sorry, no code" if calls["n"] == 1 else code_block
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}}
                ]
            },
        )

    config = make_config(tmp_path, models=("m",), only=("scicode",))
    runner = SuiteRunner(
        config,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        probe_docker=lambda: (False, "t"),
    )
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("scicode", "m")
    assert calls["n"] == 2  # both sub-steps were attempted
    assert pair.metrics["n_total"] == 2


# -- golden data provenance ----------------------------------------------------


def test_fetch_golden_rejects_non_hdf5_payloads(tmp_path, monkeypatch):
    """An LFS pointer or HTML error page must not be accepted as golden data."""
    import kairyu.bench.hub as hub

    def fake_download(repo_id, filename, dest, *, repo_type="dataset", revision=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        return dest

    monkeypatch.setattr(hub, "download_file", fake_download)
    ctx = DownloadContext(cache=BenchCache(tmp_path / "cache"))
    assert SciCodeAdapter()._fetch_golden(ctx) is None
    assert not (ctx.cache.assets_dir("scicode") / "test_data.h5").exists()


def test_fetch_golden_accepts_hdf5_magic(tmp_path, monkeypatch):
    import kairyu.bench.hub as hub

    tried: list[tuple[str, str | None]] = []

    def fake_download(repo_id, filename, dest, *, repo_type="dataset", revision=None):
        tried.append((repo_id, revision))
        if repo_id == "SciCode1/SciCode":
            return None  # the HF export does not ship it
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 16)
        return dest

    monkeypatch.setattr(hub, "download_file", fake_download)
    ctx = DownloadContext(cache=BenchCache(tmp_path / "cache"))
    path = SciCodeAdapter()._fetch_golden(ctx)
    assert path is not None and path.name == "test_data.h5"
    # upstream is tried first, then the pinned mirror
    assert [repo for repo, _ in tried] == [
        "SciCode1/SciCode",
        "Srimadh/Scicode-test-data-h5",
    ]
    assert tried[1][1] is not None and len(tried[1][1]) == 40


def test_methodology_records_denominator_and_sources(tmp_path):
    methodology = SciCodeAdapter().methodology(_ctx(tmp_path))
    assert methodology["test_split_substeps"] == 291
    assert methodology["golden_backed_substeps"] == 288  # Fugu's 288
    assert any("Srimadh" in source for source in methodology["golden_data_sources"])
    assert methodology["background"].startswith("problem-level and step-level")


@pytest.mark.parametrize("note", ["sequential", "test_data.h5", "background"])
def test_annotations_disclose_the_deviations(note):
    assert any(note in text for text in SciCodeAdapter().info.annotations)
