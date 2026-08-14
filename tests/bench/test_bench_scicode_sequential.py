"""SciCode: sequential sub-steps, background prompts, golden-data provenance.

The published dataset ships no reference code (`ground_truth_code` and
`general_solution` are null for every row), so the previous behaviour — grade
each sub-step in isolation with an empty prior-code slot — could only raise
NameError on helpers an earlier step was meant to define. These tests pin the
sequential contract instead.
"""

import hashlib

import httpx
import pytest
from conftest import make_config, make_target

from evals.adapters.base import DownloadContext, RunContext
from evals.adapters.scicode import (
    SciCodeAdapter,
    group_by_problem,
    select_problem_groups,
)
from evals.cache import BenchCache
from evals.runner import SuiteRunner
from evals.store import ResultStore
from evals.types import BenchItem


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


async def test_sampling_seeds_keep_independent_problem_histories(tmp_path):
    """Each seed reruns the whole dependency chain; histories never cross."""
    later_prompts: dict[int, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        prompt = body["messages"][0]["content"]
        seed = body["seed"]
        if "vector_norm(v)" in prompt and "normalize" not in prompt.split(
            "Now implement"
        )[1]:
            code = (
                "def vector_norm(v):\n"
                "    return float(np.sqrt(np.sum(np.asarray(v) ** 2)))\n"
                f"# sampling-seed-{seed}"
            )
        else:
            later_prompts[seed] = prompt
            code = "def normalize(v):\n    return np.asarray(v) / vector_norm(v)"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": f"```python\n{code}\n```",
                        }
                    }
                ]
            },
        )

    target = make_target(sampling_mode="recommended")
    config = make_config(
        tmp_path,
        targets=(target,),
        only=("scicode",),
        attempts=2,
    )
    runner = SuiteRunner(
        config,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        probe_docker=lambda: (False, "t"),
    )
    assert await runner.run() == 0
    pair = ResultStore(tmp_path / "results", "test-run").load_pair("scicode", "m")
    assert pair.status == "completed"
    assert pair.metrics["sampling_complete"] == 1
    assert set(later_prompts) == {0, 1}
    assert "sampling-seed-0" in later_prompts[0]
    assert "sampling-seed-1" not in later_prompts[0]
    assert "sampling-seed-1" in later_prompts[1]
    assert "sampling-seed-0" not in later_prompts[1]


# -- golden data provenance ----------------------------------------------------


def test_fetch_golden_rejects_non_hdf5_payloads(tmp_path, monkeypatch):
    """An LFS pointer or HTML error page must not be accepted as golden data."""
    import evals.hub as hub

    def fake_download(repo_id, filename, dest, *, repo_type="dataset", revision=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        return dest

    monkeypatch.setattr(hub, "download_file", fake_download)
    ctx = DownloadContext(cache=BenchCache(tmp_path / "cache"))
    assert SciCodeAdapter()._fetch_golden(ctx) is None
    assert not (ctx.cache.assets_dir("scicode") / "test_data.h5").exists()


def test_fetch_golden_requires_the_pinned_content_hash(tmp_path, monkeypatch):
    """Magic bytes prove only the format; a different valid HDF5 must be rejected."""
    import evals.adapters.scicode as scicode_mod
    import evals.hub as hub

    tried: list[tuple[str, str | None]] = []
    authentic = b"\x89HDF\r\n\x1a\n" + b"\x01" * 16

    def fake_download(repo_id, filename, dest, *, repo_type="dataset", revision=None):
        tried.append((repo_id, revision))
        if repo_id == "SciCode1/SciCode":
            return None  # the HF export does not ship it
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(authentic)
        return dest

    monkeypatch.setattr(hub, "download_file", fake_download)
    monkeypatch.setattr(scicode_mod, "_H5_BYTES", len(authentic))
    monkeypatch.setattr(
        scicode_mod, "_H5_SHA256", hashlib.sha256(authentic).hexdigest()
    )
    ctx = DownloadContext(cache=BenchCache(tmp_path / "cache"))
    path = SciCodeAdapter()._fetch_golden(ctx)
    assert path is not None and path.name == "test_data.h5"
    # upstream is tried first, then the pinned mirror
    assert [repo for repo, _ in tried] == [
        "SciCode1/SciCode",
        "Srimadh/Scicode-test-data-h5",
    ]
    assert tried[1][1] is not None and len(tried[1][1]) == 40

    # a valid HDF5 file with different contents is not the pinned artifact
    monkeypatch.setattr(
        scicode_mod, "_H5_SHA256", hashlib.sha256(b"something else").hexdigest()
    )
    assert SciCodeAdapter()._fetch_golden(ctx) is None
    assert not (ctx.cache.assets_dir("scicode") / "test_data.h5").exists()


def test_golden_data_pin_is_a_sha256_and_size(tmp_path):
    from evals.adapters.scicode import (
        _H5_BYTES,
        _H5_SHA256,
        golden_data_is_authentic,
    )

    assert len(_H5_SHA256) == 64 and _H5_BYTES > 0
    # a right-sized non-HDF5 blob still fails
    decoy = tmp_path / "decoy.h5"
    decoy.write_bytes(b"\x00" * 8)
    assert not golden_data_is_authentic(decoy)
    assert not golden_data_is_authentic(tmp_path / "missing.h5")


# -- the official 288-sub-step population --------------------------------------


def _substep(step_number: str, *, background: str = "") -> dict:
    return {
        "step_number": step_number,
        "step_description_prompt": f"do {step_number}",
        "step_background": background,
        "function_header": f"def f{step_number.replace('.', '_')}():",
        "return_line": "",
        "test_cases": ["assert True"],
    }


def _problem(problem_id: str, steps: int) -> dict:
    return {
        "problem_id": problem_id,
        "problem_description_main": f"problem {problem_id}",
        "problem_background_main": "",
        "required_dependencies": "import numpy as np",
        "sub_steps": [_substep(f"{problem_id}.{i + 1}") for i in range(steps)],
    }


def _official_shaped_rows() -> list[dict]:
    """65 problems / 291 sub-steps, including the three the evaluator skips."""
    rows = [_problem("13", 6), _problem("62", 1), _problem("76", 3)]
    remaining = 291 - (6 + 1 + 3)
    for index in range(remaining):
        rows.append(_problem(f"p{index}", 1))
    return rows


def _patch_scicode(monkeypatch, rows, *, provided=None):
    import evals.adapters.scicode as scicode_mod
    import evals.hub as hub

    monkeypatch.setattr(hub, "load_hf_rows", lambda *a, **k: rows)
    monkeypatch.setattr(SciCodeAdapter, "_fetch_golden", lambda self, ctx: None)
    monkeypatch.setattr(
        SciCodeAdapter,
        "_fetch_provided_steps",
        lambda self: (
            provided
            if provided is not None
            else dict.fromkeys(scicode_mod._PROVIDED_STEPS, "def provided():\n    return 1")
        ),
    )


def test_normalize_marks_the_three_steps_the_evaluator_skips(tmp_path, monkeypatch):
    """Fugu's denominator is 288: the official evaluator `continue`s past 3 steps."""
    from evals.adapters.scicode import _EXCLUDED_STEPS

    _patch_scicode(monkeypatch, _official_shaped_rows())
    rows = SciCodeAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))

    assert len(rows) == 291
    excluded = [row for row in rows if row["excluded"]]
    assert {(row["problem_id"], row["step_index"]) for row in excluded} == set(
        _EXCLUDED_STEPS
    )
    assert len(rows) - len(excluded) == 288
    # the excluded steps carry the implementation later steps depend on
    assert all(row["provided_code"] for row in excluded)


def test_normalize_fails_closed_without_the_provided_implementations(tmp_path, monkeypatch):
    """Later steps of 13/62/76 call helpers those steps define.

    Scoring them without the official implementation would charge the model for
    the harness's missing file and pollute the 288-step denominator.
    """
    _patch_scicode(monkeypatch, _official_shaped_rows(), provided={})
    with pytest.raises(Exception, match="provided implementations are unavailable"):
        SciCodeAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))


def test_normalize_fails_closed_on_a_partially_provided_mapping(tmp_path, monkeypatch):
    _patch_scicode(
        monkeypatch,
        _official_shaped_rows(),
        provided={("13", 5): "def provided():\n    return 1"},
    )
    with pytest.raises(Exception, match="problem 62 step 1") as error:
        SciCodeAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))
    assert "problem 76 step 3" in str(error.value)


async def test_cached_golden_data_is_revalidated_on_reuse(tmp_path):
    """A replaced or truncated asset must not become the expected-answer source."""
    adapter = SciCodeAdapter()
    ctx = _ctx(tmp_path, offline_fixtures=False)
    assets = ctx.cache.assets_dir("scicode")
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "test_data.h5").write_bytes(b"corrupt")

    item = BenchItem(
        id="scicode-1.1",
        payload={
            "step_id": "1.1",
            "dependencies": "",
            "prior_code": "",
            "test_cases": ["assert f(1) == target"],
        },
    )
    result = await adapter.score(item, "```python\ndef f(x):\n    return x\n```", ctx)
    assert result.status == "unjudged"
    assert "pinned hash" in result.error


def test_normalize_fails_closed_if_the_population_moved(tmp_path, monkeypatch):
    rows = _official_shaped_rows()
    rows.append(_problem("extra", 2))
    _patch_scicode(monkeypatch, rows)
    with pytest.raises(Exception, match="expected 291"):
        SciCodeAdapter().normalize(DownloadContext(cache=BenchCache(tmp_path / "c")))


async def test_excluded_steps_are_not_scored_but_are_carried_forward(tmp_path):
    """The skipped step never reaches the model, yet its helper is in scope."""
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        prompts.append(_json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "```python\ndef later():\n    return provided()\n```",
                        },
                    }
                ]
            },
        )

    adapter = SciCodeAdapter()
    items = [
        BenchItem(
            id="scicode-62.1",
            payload={
                **_item("62", 0).payload,
                "excluded": True,
                "provided_code": "def provided():\n    return 7",
                "test_cases": [],
            },
        ),
        BenchItem(
            id="scicode-62.2",
            payload={**_item("62", 1).payload, "excluded": False, "test_cases": []},
        ),
    ]
    ctx = _ctx(
        tmp_path,
        http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    from unittest.mock import patch

    with patch.object(SciCodeAdapter, "load_items", return_value=items):
        pair = await adapter.run(make_target(), ctx)

    assert pair.metrics["n_total"] == 1  # only the scoreable step
    assert len(prompts) == 1
    assert "def provided():" in prompts[0]  # the provided helper is in the context
    assert "scicode-62.1" not in {item.item_id for item in pair.items}


def test_previous_steps_carry_their_description_and_background(tmp_path):
    """Official `process_problem_steps` passes description + background + code."""
    adapter = SciCodeAdapter()
    item = BenchItem(
        id="x",
        payload={
            "main_description": "d",
            "main_background": "",
            "dependencies": "import numpy as np",
            "step_description": "write step three",
            "step_background": "",
            "function_header": "def three():",
            "return_line": "",
            "prior_code": "def one():\n    return 1\n\ndef two():\n    return 2",
            "prior_steps": [
                {
                    "step_description": "write step one",
                    "step_background": "Background: one is the base case",
                    "code": "def one():\n    return 1",
                },
                {
                    "step_description": "write step two",
                    "step_background": "",
                    "code": "def two():\n    return 2",
                },
            ],
        },
    )
    prompt = adapter.build_request(item, make_target(), _ctx(tmp_path)).messages[0]["content"]
    assert "write step one" in prompt
    assert "Background: one is the base case" in prompt
    assert "write step two" in prompt
    assert "def one():" in prompt and "def two():" in prompt
    assert "------" in prompt  # the official separator between prior steps


def test_methodology_records_denominator_and_sources(tmp_path):
    methodology = SciCodeAdapter().methodology(_ctx(tmp_path))
    assert methodology["test_split_substeps"] == 291
    assert methodology["scored_substeps"] == 288  # Fugu's 288
    assert methodology["excluded_substeps"] == [
        "problem 13 step 6",
        "problem 62 step 1",
        "problem 76 step 3",
    ]
    assert any("Srimadh" in source for source in methodology["golden_data_sources"])
    assert len(methodology["golden_data_sha256"]) == 64
    assert "NOT independently cross-checked" in methodology["golden_data_provenance"]
    assert "process_problem_steps" in methodology["previous_step_prompt"]
    assert methodology["background"].startswith("problem-level and step-level")


@pytest.mark.parametrize("note", ["sequential", "test_data.h5", "background"])
def test_annotations_disclose_the_deviations(note):
    assert any(note in text for text in SciCodeAdapter().info.annotations)
