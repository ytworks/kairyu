"""Agentic harness invocations: Fugu's turn/trial conditions and real CLI shapes.

Every command here was verified against the installed harnesses' option
definitions. Three were unrunnable as written: `harbor run` has no
`--output-dir` (that flag belongs to `harbor jobs download`), the τ harness has
no `--output` and no `banking` domain, and neither SWE-Bench Pro nor
Terminal-Bench raised its turn budget to the published limit.
"""

import json
from pathlib import Path

import httpx
import pytest

from kairyu.bench.adapters.base import RunContext
from kairyu.bench.adapters.swebench_pro import SweBenchProAdapter
from kairyu.bench.adapters.swebench_verified import SweBenchVerifiedAdapter
from kairyu.bench.adapters.tau_bench import (
    TauBenchBankingAdapter,
    data_dir_candidates,
    find_results,
)
from kairyu.bench.adapters.terminal_bench import TerminalBenchAdapter
from kairyu.bench.cache import BenchCache
from kairyu.bench.judge import JudgeClient
from kairyu.bench.types import BenchTarget, JudgeConfig


def _ctx(tmp_path, **overrides) -> RunContext:
    defaults = dict(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=lambda: httpx.AsyncClient(),
        docker=(True, "docker available"),
    )
    defaults.update(overrides)
    return RunContext(**defaults)


def _target(**kwargs) -> BenchTarget:
    return BenchTarget(base_url="http://gw/v1", model="m", **kwargs)


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _harbor_agent(command: list[str]) -> dict:
    config = json.loads(Path(_flag_value(command, "--config")).read_text())
    return config["agents"][0]


# -- SWE-Bench -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "subset", "dataset", "steps"),
    [
        pytest.param(
            SweBenchProAdapter(),
            "ScaleAI/SWE-bench_Pro",
            "ScaleAI/SWE-bench_Pro",
            1000,
            id="pro",
        ),
        pytest.param(
            SweBenchVerifiedAdapter(),
            "verified",
            "princeton-nlp/SWE-bench_Verified",
            250,
            id="verified",
        ),
    ],
)
def test_swebench_variants_keep_their_official_conditions(
    tmp_path, adapter, subset, dataset, steps
):
    command = adapter._generate_command(_target(), _ctx(tmp_path), Path("mini-output"))
    configs = [command[i + 1] for i, value in enumerate(command) if value == "--config"]
    assert configs[:2] == ["swebench.yaml", f"agent.step_limit={steps}"]
    assert _flag_value(command, "--subset") == subset
    assert _flag_value(command, "--split") == "test"

    evaluation = adapter._evaluate_command(
        Path("preds.json"), ("a", "b"), "run", Path("reports"), 3
    )
    assert _flag_value(evaluation, "--dataset_name") == dataset
    assert _flag_value(evaluation, "--split") == "test"
    assert _flag_value(evaluation, "--max_workers") == "3"
    assert _flag_value(evaluation, "--run_id") == "run"
    assert _flag_value(evaluation, "--report_dir") == "reports"
    assert evaluation[evaluation.index("--instance_ids") + 1 :] == ["a", "b"]


def test_swebench_sets_fugu_step_limit(tmp_path):
    command = SweBenchProAdapter()._generate_command(
        _target(), _ctx(tmp_path), Path("mini-output")
    )
    configs = [command[i + 1] for i, part in enumerate(command) if part == "--config"]
    # the harness drops its default config as soon as -c is given, so the
    # default file must be restated alongside the override
    assert configs == ["swebench.yaml", "agent.step_limit=1000"]
    assert _flag_value(command, "--subset") == "ScaleAI/SWE-bench_Pro"
    assert _flag_value(command, "--split") == "test"
    assert _flag_value(command, "--output") == "mini-output"


def test_swebench_step_limit_is_disclosed():
    adapter = SweBenchProAdapter()
    assert any("step_limit=1000" in note for note in adapter.info.annotations)


async def test_swebench_rejects_unmapped_attempt_budget(tmp_path):
    pair = await SweBenchProAdapter().run(
        _target(),
        _ctx(tmp_path, attempts=2),
    )
    assert pair.status == "skipped"
    assert pair.reason is not None
    assert "requires attempts=1" in pair.reason


# -- Terminal-Bench ------------------------------------------------------------


def test_terminal_bench_uses_harbor_flags_that_exist(tmp_path):
    command = TerminalBenchAdapter()._command(_target(), _ctx(tmp_path), tmp_path / "jobs")
    # `harbor run` writes to --jobs-dir; --output-dir is a `jobs download` flag
    assert "--output-dir" not in command
    assert _flag_value(command, "--jobs-dir") == str(tmp_path / "jobs")
    # Terminal-Bench 2.1 is a Harbor Hub organization/package, not the legacy
    # registry's `name@version` entry.
    assert _flag_value(command, "-d") == "terminal-bench/terminal-bench-2-1"
    agent = _harbor_agent(command)
    assert agent["name"] == "terminus-2"
    assert agent["model_name"] == "openai/m"


def test_terminal_bench_sets_fugu_turn_budget(tmp_path):
    command = TerminalBenchAdapter()._command(_target(), _ctx(tmp_path), tmp_path)
    assert _harbor_agent(command)["kwargs"]["max_turns"] == 500


def test_terminal_bench_caps_output_limit_via_terminus_llm_kwargs(tmp_path):
    command = TerminalBenchAdapter()._command(
        _target(max_output_tokens=32768), _ctx(tmp_path), tmp_path
    )
    kwargs = _harbor_agent(command)["kwargs"]
    assert kwargs["max_turns"] == 500
    assert kwargs["llm_call_kwargs"]["max_tokens"] == 4096
    response_format = kwargs["llm_call_kwargs"]["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["commands"]["maxItems"] == 3
    assert (
        schema["properties"]["commands"]["items"]["properties"]["keystrokes"][
            "maxLength"
        ]
        == 2048
    )


def test_terminal_bench_preserves_smaller_output_limit(tmp_path):
    command = TerminalBenchAdapter()._command(
        _target(max_output_tokens=2048), _ctx(tmp_path), tmp_path
    )
    llm_kwargs = _harbor_agent(command)["kwargs"]["llm_call_kwargs"]
    assert llm_kwargs["max_tokens"] == 2048
    assert llm_kwargs["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize("attempts", [1, 5])
def test_terminal_bench_passes_attempts(tmp_path, attempts):
    command = TerminalBenchAdapter()._command(
        _target(), _ctx(tmp_path, attempts=attempts), tmp_path
    )
    assert _flag_value(command, "-k") == str(attempts)


def test_terminal_bench_passes_concurrency(tmp_path):
    command = TerminalBenchAdapter()._command(
        _target(), _ctx(tmp_path, concurrency=4), tmp_path
    )
    assert _flag_value(command, "--n-concurrent") == "4"


def test_terminal_bench_expands_only_the_agent_execution_timeout(tmp_path):
    command = TerminalBenchAdapter()._command(_target(), _ctx(tmp_path), tmp_path)
    assert _flag_value(command, "--agent-timeout-multiplier") == "8.0"
    assert _harbor_agent(command)["max_timeout_sec"] == 900.0
    assert "--timeout-multiplier" not in command
    assert "--verifier-timeout-multiplier" not in command


def test_terminal_bench_limit_is_a_task_cap(tmp_path):
    command = TerminalBenchAdapter()._command(_target(), _ctx(tmp_path, limit=3), tmp_path)
    assert _flag_value(command, "--n-tasks") == "3"


def test_terminal_bench_can_use_pinned_local_dataset_and_task_filter(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "terminal-bench-2-1"
    dataset.mkdir()
    monkeypatch.setenv("KAIRYU_TERMINAL_BENCH_PATH", str(dataset))
    monkeypatch.setenv("KAIRYU_TERMINAL_BENCH_TASKS", "write-compressor, fix-git")
    monkeypatch.setattr("kairyu.bench.adapters.terminal_bench.shutil.which", lambda _: "harbor")

    adapter = TerminalBenchAdapter()
    command = adapter._command(_target(), _ctx(tmp_path), tmp_path / "jobs")

    assert "-d" not in command
    assert _flag_value(command, "--path") == str(dataset)
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--include-task-name"
    ] == ["write-compressor", "fix-git"]
    assert adapter._preconditions(_target(), _ctx(tmp_path)) is None


def test_terminal_bench_rejects_missing_local_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "KAIRYU_TERMINAL_BENCH_PATH", str(tmp_path / "missing-terminal-bench")
    )
    monkeypatch.setattr("kairyu.bench.adapters.terminal_bench.shutil.which", lambda _: "harbor")

    reason = TerminalBenchAdapter()._preconditions(_target(), _ctx(tmp_path))

    assert reason is not None
    assert "existing absolute directory" in reason


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(_target(temperature=0.6), id="temperature"),
        pytest.param(_target(sampling_mode="recommended"), id="recommended"),
    ],
)
@pytest.mark.parametrize(
    "adapter",
    [SweBenchProAdapter(), TerminalBenchAdapter(), TauBenchBankingAdapter()],
)
async def test_agentic_harnesses_reject_chat_only_sampling_policy(
    tmp_path,
    adapter,
    target,
):
    ctx = _ctx(tmp_path, judge=_judge())
    pair = await adapter.run(target, ctx)
    assert pair.status == "skipped"
    assert pair.reason is not None
    assert "supported only by chat adapters" in pair.reason


# -- Harbor result schema ------------------------------------------------------

# Embedded JobResult shape retained by older Harbor versions. Harbor 0.17's
# final job-level file excludes trial_results and is covered separately below.
HARBOR_JOB_RESULT = {
    "id": "3f1b0e6a-0000-4000-8000-000000000000",
    "started_at": "2026-07-25T00:00:00",
    "n_total_trials": 4,
    "stats": {"n_completed_trials": 3, "n_errored_trials": 1},
    "trial_results": [
        {
            "trial_name": "build-tool.1",
            "task_id": {"name": "build-tool"},
            "verifier_result": {"rewards": {"reward": 1}},
        },
        {
            "trial_name": "fix-perms.1",
            "task_id": {"name": "fix-perms"},
            "verifier_result": {"rewards": {"reward": 0}},
        },
        {
            "trial_name": "partial-credit.1",
            "task_id": {"name": "partial-credit"},
            "verifier_result": {"rewards": {"accuracy": 0.5}},
        },
        {
            "trial_name": "crashed.1",
            "task_id": {"name": "crashed"},
            "exception_info": {"exception_type": "TimeoutError"},
        },
    ],
}


def test_parse_harbor_job_result_schema():
    from kairyu.bench.adapters.terminal_bench import parse_harbor_results

    items = {item.item_id: item for item in parse_harbor_results(HARBOR_JOB_RESULT)}
    assert items["build-tool.1"].score == 1.0
    assert items["fix-perms.1"].score == 0.0
    assert items["partial-credit.1"].score == 0.5
    assert items["crashed.1"].status == "failed"
    assert "TimeoutError" in items["crashed.1"].error


def test_ambiguous_reward_dict_is_failed_not_guessed():
    from kairyu.bench.adapters.terminal_bench import parse_harbor_results, trial_reward

    assert trial_reward({"tests_passed": 3, "tests_total": 4}) is None
    assert trial_reward({"anything": 1}) == 1.0  # a single key is unambiguous
    assert trial_reward({"resolved": True}) == 1.0

    items = parse_harbor_results(
        {
            "trial_results": [
                {
                    "trial_name": "t",
                    "verifier_result": {"rewards": {"a": 1, "b": 2}},
                }
            ]
        }
    )
    assert items[0].status == "failed"
    assert "ambiguous reward keys" in items[0].error


def test_parse_harbor_results_rejects_string_entries_without_crashing():
    from kairyu.bench.adapters.terminal_bench import parse_harbor_results

    items = parse_harbor_results({"results": ["not-a-trial"]})

    assert len(items) == 1
    assert items[0].status == "failed"
    assert "not an object" in items[0].error


def test_parse_harbor_results_counts_declared_but_missing_trials():
    from kairyu.bench.adapters.terminal_bench import parse_harbor_results

    items = parse_harbor_results(
        {
            "n_total_trials": 2,
            "trial_results": [HARBOR_JOB_RESULT["trial_results"][0]],
        }
    )

    assert len(items) == 2
    assert items[0].score == 1.0
    assert items[1].status == "failed"
    assert "omitted" in items[1].error


async def test_terminal_bench_reads_the_real_harbor_output(tmp_path, monkeypatch):
    """A successful `harbor run` must not report 'no harbor results file found'."""
    import json as _json

    import kairyu.bench.adapters.terminal_bench as tb

    monkeypatch.setattr(tb.shutil, "which", lambda name: "/usr/bin/harbor")

    def fake_run(command, capture_output, timeout, env, check):
        jobs_dir = Path(command[command.index("--jobs-dir") + 1])
        job = jobs_dir / "job-1"
        (job / "trials" / "build-tool.1").mkdir(parents=True)
        # a per-trial artifact exists too; the job-level file must win
        (job / "trials" / "build-tool.1" / "result.json").write_text(
            _json.dumps(HARBOR_JOB_RESULT["trial_results"][0]), encoding="utf-8"
        )
        (job / "result.json").write_text(_json.dumps(HARBOR_JOB_RESULT), encoding="utf-8")

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(tb.subprocess, "run", fake_run)
    pair = await TerminalBenchAdapter().run(_target(), _ctx(tmp_path))
    assert pair.status == "partial"  # the crashed trial keeps it honest
    assert pair.metrics["n_total"] == 4
    assert pair.metrics["n_failed"] == 1
    # Harbor's own Mean maps a missing reward to zero, so the errored trial is a
    # zero in the denominator -- not an exclusion that would inflate the score
    assert pair.score == pytest.approx((1.0 + 0.0 + 0.5 + 0.0) / 4)
    assert "errored ones as zero" in pair.methodology["denominator"]


async def test_terminal_bench_reads_harbor_017_per_trial_results(tmp_path, monkeypatch):
    """Harbor 0.17 omits trial_results from the final job-level result."""
    import json as _json

    import kairyu.bench.adapters.terminal_bench as tb

    monkeypatch.setattr(tb.shutil, "which", lambda name: "/usr/bin/harbor")

    def fake_run(command, capture_output, timeout, env, check):
        jobs_dir = Path(command[command.index("--jobs-dir") + 1])
        job = jobs_dir / "2026-08-11__00-00-00"
        job.mkdir(parents=True)
        (job / "result.json").write_text(
            _json.dumps(
                {
                    "id": "job",
                    "n_total_trials": 4,
                    "stats": {"n_completed_trials": 4},
                }
            ),
            encoding="utf-8",
        )
        for entry in HARBOR_JOB_RESULT["trial_results"]:
            trial = job / entry["trial_name"]
            trial.mkdir()
            (trial / "result.json").write_text(_json.dumps(entry), encoding="utf-8")
        # This resembles an agent artifact that triggered the production crash.
        (job / "agent-artifact.json").write_text(
            _json.dumps({"results": ["stdout", "stderr"]}), encoding="utf-8"
        )

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(tb.subprocess, "run", fake_run)
    pair = await TerminalBenchAdapter().run(_target(), _ctx(tmp_path))

    assert pair.status == "partial"
    assert pair.metrics["n_total"] == 4
    assert pair.metrics["n_failed"] == 1
    assert pair.score == pytest.approx((1.0 + 0.0 + 0.5 + 0.0) / 4)


def test_harbor_mean_counts_every_trial():
    from kairyu.bench.adapters.terminal_bench import harbor_mean
    from kairyu.bench.types import ItemResult

    items = [
        ItemResult(item_id="a", status="completed", score=1.0),
        ItemResult(item_id="b", status="completed", score=0.0),
        ItemResult(item_id="c", status="completed", score=0.5),
        ItemResult(item_id="d", status="failed", error="timeout"),
    ]
    assert harbor_mean(items) == pytest.approx(0.375)
    # excluding the error would report a crashed run as a better score
    assert harbor_mean(items) < (1.0 + 0.0 + 0.5) / 3
    assert harbor_mean([]) is None


# -- τ³ Banking ----------------------------------------------------------------


def _judge(**kwargs) -> JudgeClient:
    config = JudgeConfig(base_url="http://gw/v1", model="j", **kwargs)
    return JudgeClient(config, http_factory=lambda: httpx.AsyncClient())


def test_tau_uses_real_domain_and_single_public_model_retrieval_config(tmp_path):
    ctx = _ctx(tmp_path, judge=_judge())
    command = TauBenchBankingAdapter()._command("tau2", _target(), ctx, "run-name")
    # there is no `banking` domain in the harness, only banking_knowledge
    assert _flag_value(command, "--domain") == "banking_knowledge"
    assert _flag_value(command, "--retrieval-config") == "full_kb"


def test_tau_addresses_results_by_name_not_path(tmp_path):
    ctx = _ctx(tmp_path, judge=_judge())
    command = TauBenchBankingAdapter()._command("tau2", _target(), ctx, "run-name")
    # --output does not exist; --save-to is a name under the harness data dir
    assert "--output" not in command
    assert _flag_value(command, "--save-to") == "run-name"


@pytest.mark.parametrize("attempts", [1, 4])
def test_tau_passes_num_trials(tmp_path, attempts):
    ctx = _ctx(tmp_path, judge=_judge(), attempts=attempts)
    command = TauBenchBankingAdapter()._command("tau2", _target(), ctx, "n")
    assert _flag_value(command, "--num-trials") == str(attempts)


def test_tau_forwards_user_simulator_reasoning_effort(tmp_path):
    """Fugu ran the user simulator at low effort."""
    ctx = _ctx(tmp_path, judge=_judge(reasoning_effort="low"))
    command = TauBenchBankingAdapter()._command(
        "tau2", _target(reasoning_effort="high"), ctx, "n"
    )
    assert json.loads(_flag_value(command, "--user-llm-args")) == {"reasoning_effort": "low"}
    assert json.loads(_flag_value(command, "--agent-llm-args")) == {"reasoning_effort": "high"}


def test_tau_omits_llm_args_when_no_sampling_is_set(tmp_path):
    ctx = _ctx(tmp_path, judge=_judge())
    command = TauBenchBankingAdapter()._command("tau2", _target(), ctx, "n")
    assert "--user-llm-args" not in command
    assert "--agent-llm-args" not in command


def test_find_results_prefers_the_env_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TAU2_DATA_DIR", str(tmp_path / "data"))
    results = tmp_path / "data" / "simulations" / "run-name" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text("{}", encoding="utf-8")
    assert find_results("tau2", "run-name") == results
    assert find_results("tau2", "other-name") is None


def test_data_dir_candidates_include_env_and_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("TAU2_DATA_DIR", str(tmp_path / "explicit"))
    candidates = data_dir_candidates("tau2")
    assert candidates[0] == tmp_path / "explicit"
    assert Path.cwd() / "data" in candidates
    assert len(candidates) == len(set(candidates))  # deduplicated


def test_tau_annotations_disclose_conditions():
    notes = TauBenchBankingAdapter().info.annotations
    assert any("banking_knowledge" in note for note in notes)
    assert any("pass@4" in note for note in notes)
    assert any("not grouped chat seed-sweep" in note for note in notes)


def test_tau_save_to_is_unique_per_invocation(tmp_path):
    """A fixed name makes the harness prompt to resume, or resume stale work."""
    adapter = TauBenchBankingAdapter()
    ctx = _ctx(tmp_path, judge=_judge(), run_id="run-a")
    names = set()
    for _ in range(3):
        # the name is built inside run(); rebuild it the same way the adapter does
        names.add(adapter._save_to_name(_target(), ctx))
    assert len(names) == 3
    assert all(name.startswith("kairyu-tau-bench-banking-m-run-a-") for name in names)


def test_tau_data_dir_prefers_the_harness_resolution(tmp_path, monkeypatch):
    """The harness computes DATA_DIR beside site-packages, not inside it."""
    import kairyu.bench.adapters.tau_bench as tau

    monkeypatch.setenv("TAU2_DATA_DIR", str(tmp_path / "from-env"))
    monkeypatch.setattr(tau, "harness_data_dir", lambda flavor: tmp_path / "from-harness")
    candidates = tau.data_dir_candidates("tau2")
    assert candidates[0] == tmp_path / "from-harness"
    assert tmp_path / "from-env" in candidates


def test_tau_data_dir_falls_back_when_harness_import_fails(tmp_path, monkeypatch):
    import kairyu.bench.adapters.tau_bench as tau

    monkeypatch.delenv("TAU2_DATA_DIR", raising=False)
    monkeypatch.setattr(tau, "harness_data_dir", lambda flavor: None)
    candidates = tau.data_dir_candidates("tau2")
    assert candidates  # still offers the package/cwd layouts
    assert Path.cwd() / "data" in candidates


def test_harness_data_dir_reads_the_module_attribute(monkeypatch, tmp_path):
    import sys
    import types

    import kairyu.bench.adapters.tau_bench as tau

    module = types.ModuleType("tau2.utils.utils")
    module.DATA_DIR = tmp_path / "installed" / "data"
    monkeypatch.setitem(sys.modules, "tau2.utils.utils", module)
    assert tau.harness_data_dir("tau2") == tmp_path / "installed" / "data"


# -- SWE-Bench Pro sampling ----------------------------------------------------


def test_swebench_forwards_named_sampling_to_model_kwargs(tmp_path):
    command = SweBenchProAdapter()._generate_command(
        _target(reasoning_effort="high", top_p=0.9, seed=7),
        _ctx(tmp_path),
        Path("preds.json"),
    )
    configs = [command[i + 1] for i, part in enumerate(command) if part == "--config"]
    assert "model.model_kwargs.reasoning_effort=high" in configs
    assert "model.model_kwargs.top_p=0.9" in configs
    assert "model.model_kwargs.seed=7" in configs


def test_swebench_omits_model_kwargs_when_unset(tmp_path):
    command = SweBenchProAdapter()._generate_command(
        _target(), _ctx(tmp_path), Path("preds.json")
    )
    assert not any("model_kwargs" in part for part in command)


def test_swebench_annotates_what_it_cannot_forward():
    notes = SweBenchProAdapter().info.annotations
    assert any("extra_body" in note and "NOT forwarded" in note for note in notes)
    assert any("temperature" in note and "rejected" in note for note in notes)


def test_terminal_bench_annotates_that_sampling_is_not_forwarded():
    notes = TerminalBenchAdapter().info.annotations
    assert any("NOT" in note and "sampling" in note for note in notes)
    assert any("temperature" in note and "rejected" in note for note in notes)
