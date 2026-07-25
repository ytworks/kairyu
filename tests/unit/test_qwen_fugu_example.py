"""Static contract for the Qwen3-32B Fugu-suite example scripts.

The scripts cannot be executed here (they need GPUs and a served model), so the
properties that would silently produce a wrong or misleading number are pinned
statically: the model is preflighted by exact id, a subset run announces itself,
and the accuracy report is what the operator is pointed at.
"""

import re
import stat
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "qwen3-32b-multi-gpu"


def _run_args(**overrides):
    """Minimal `kairyu bench run` namespace for config assembly."""
    import argparse

    defaults = dict(
        config=None,
        base_url="http://gw/v1",
        model=["qwen3-32b"],
        target=None,
        api_key_env=None,
        no_vision=False,
        reasoning_effort=None,
        top_p=None,
        sampling_seed=None,
        extra_body=None,
        suite=None,
        only=None,
        exclude=None,
        limit=None,
        attempts=None,
        smoke=False,
        offline_fixtures=False,
        seed=None,
        judge_base_url=None,
        judge_model=None,
        judge_api_key_env=None,
        judge_reasoning_effort=None,
        judge_extra_body=None,
        concurrency=None,
        results_dir=None,
        run_id=None,
        rerun=False,
        cache_dir=None,
        no_download=False,
        no_progress=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
FUGU = EXAMPLE / "fugu-benchmark.sh"
RUN_FUGU = EXAMPLE / "run-fugu-benchmark.sh"


@pytest.fixture(scope="module")
def fugu_text() -> str:
    return FUGU.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def run_fugu_text() -> str:
    return RUN_FUGU.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", [FUGU, RUN_FUGU])
def test_scripts_are_executable_posix_sh(script):
    assert script.is_file(), script
    assert script.read_text(encoding="utf-8").startswith("#!/bin/sh")
    assert stat.S_IMODE(script.stat().st_mode) & stat.S_IXUSR
    # fail fast on unset variables and errors, like the sibling scripts
    assert "set -eu" in script.read_text(encoding="utf-8")


def test_readiness_is_waited_for_before_benchmarking(fugu_text, run_fugu_text):
    for text in (fugu_text, run_fugu_text):
        assert "/readyz" in text


def test_served_model_is_preflighted_by_exact_id(fugu_text):
    """A healthy gateway can pass readyz while serving a different model."""
    assert "/models" in fugu_text
    # grepped as a JSON id field, so a substring match on another model's name
    # cannot pass the preflight
    assert '\\"id\\"' in fugu_text
    assert "${model}" in fugu_text
    # and the run stops rather than benchmarking the wrong deployment
    assert re.search(r"is not served at.*\n.*exit 1", fugu_text, re.S)


def test_judge_defaults_to_the_same_gateway(fugu_text):
    """The tau user simulator must share one OPENAI_BASE_URL with the target."""
    assert "--judge-base-url" in fugu_text
    assert 'judge_model="${JUDGE_MODEL:-$model}"' in fugu_text


def test_subset_and_full_runs_are_both_announced(fugu_text):
    """A capped run must never be mistaken for a full-suite number."""
    assert "SUBSET RUN" in fugu_text
    assert "FULL RUN" in fugu_text
    assert 'bench_limit="${BENCH_LIMIT:-20}"' in fugu_text
    assert "--limit" in fugu_text


def test_fixture_mode_says_its_scores_are_meaningless(fugu_text):
    assert "OFFLINE FIXTURES" in fugu_text
    assert "not meaningful" in fugu_text
    assert "--offline-fixtures" in fugu_text


def test_fugu_conditions_are_reachable_from_the_environment(fugu_text):
    for flag in (
        "--reasoning-effort",
        "--judge-reasoning-effort",
        "--extra-body",
        "--attempts",
        "--only",
        "--exclude",
    ):
        assert flag in fugu_text, flag


def test_operator_is_pointed_at_both_artifacts(fugu_text):
    assert "scoreboard.md" in fugu_text
    assert "comparison.md" in fugu_text


def test_dataset_extra_is_installed_for_the_run(fugu_text):
    """The serving image has no dataset deps; the suite runs on the host."""
    assert "uv run --extra bench kairyu bench run" in fugu_text
    assert "command -v uv" in fugu_text


def test_one_command_entry_point_starts_then_benchmarks(run_fugu_text):
    assert "./run.sh --detach" in run_fugu_text
    assert "exec ./fugu-benchmark.sh" in run_fugu_text
    # an already-running service is reused rather than restarted
    assert "already ready" in run_fugu_text


def test_text_only_target_is_declared_text_only(fugu_text):
    """Qwen3-32B is a causal LM; the vision family is the separate Qwen3-VL.

    Declaring it vision-capable would let CharXiv and HLE image rows be measured
    on prompts whose image parts the text-only chat template drops.
    """
    assert "--no-vision" in fugu_text
    assert 'vision_flag="--no-vision"' in fugu_text
    assert "VISION" in fugu_text  # an opt-in for a genuinely multimodal deployment
    assert "$vision_flag" in fugu_text


def test_no_vision_flag_narrows_the_target():
    from kairyu.bench.config import build_config

    assert build_config(_run_args(no_vision=True)).targets[0].supports_vision is False
    assert build_config(_run_args()).targets[0].supports_vision is True


def test_vision_slots_skip_on_a_text_only_target(tmp_path):
    """The honest outcome: skipped, not a score from an image-free prompt."""
    import httpx

    from kairyu.bench.adapters.base import RunContext
    from kairyu.bench.adapters.charxiv import CharXivAdapter
    from kairyu.bench.adapters.hle import HleAdapter
    from kairyu.bench.cache import BenchCache
    from kairyu.bench.judge import JudgeClient
    from kairyu.bench.types import BenchItem, BenchTarget, JudgeConfig, SkipItem

    target = BenchTarget(base_url="http://gw/v1", model="qwen3-32b", supports_vision=False)
    ctx = RunContext(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        offline_fixtures=True,
        # a judge is configured, so vision is the only unmet precondition left
        judge=JudgeClient(
            JudgeConfig(base_url="http://gw/v1", model="j"),
            http_factory=lambda: httpx.AsyncClient(),
        ),
    )
    assert "vision" in CharXivAdapter().check_preconditions(target, ctx)

    image_item = BenchItem(
        id="x",
        payload={
            "question": "Q",
            "answer": "A",
            "answer_type": "multipleChoice",
            "image": "data:image/png;base64,AAAA",
        },
    )
    assert isinstance(HleAdapter().build_request(image_item, target, ctx), SkipItem)


def test_subset_warning_is_said_to_survive_into_the_artifacts(fugu_text):
    assert "scoreboard.md and comparison.md" in fugu_text
    assert "withhold every delta" in fugu_text


def test_port_reaches_the_compose_mapping(run_fugu_text):
    """PORT=9000 must not start a healthy service on 8001 and then time out."""
    compose = (EXAMPLE / "compose.yaml").read_text(encoding="utf-8")
    assert '"127.0.0.1:${PORT:-8001}:8000"' in compose
    assert 'export PORT="$port"' in run_fugu_text
    assert "export PORT" in (EXAMPLE / "run.sh").read_text(encoding="utf-8")


def test_readme_documents_the_quality_suite():
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    assert "run-fugu-benchmark.sh" in readme
    assert "BENCH_LIMIT" in readme
    assert "comparison.md" in readme
