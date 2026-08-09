"""Source-, configuration-, and item-bound A/B quality gates."""

from __future__ import annotations

import hashlib
import json
import platform
from copy import deepcopy

import pytest

from kairyu.bench import history
from kairyu.bench.aggregate import build_scoreboard
from kairyu.bench.cli import handle
from kairyu.bench.config_ab import (
    ConfigComparisonError,
    _paired_cluster_ids,
    build_config_comparison,
    render_config_comparison_markdown,
)
from kairyu.bench.history import build_entry
from kairyu.bench.runner import _recordable_config, _run_fingerprint, _run_identity
from kairyu.bench.store import ResultStore
from kairyu.bench.types import (
    BenchConfig,
    BenchTarget,
    ExecutionConfig,
    ItemResult,
    PairResult,
    ServedConfigIdentity,
)
from kairyu.entrypoints import cli


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _adapter_identity(name: str) -> dict:
    requires_execution = name == "scicode"
    resources = [
        ("kairyu.bench.adapters", f"{name.replace('-', '_')}.py"),
        ("kairyu.bench.adapters", "base.py"),
        ("kairyu.bench", "aggregate.py"),
        ("kairyu.bench", "cache.py"),
        ("kairyu.bench", "runner.py"),
        ("kairyu.bench", "sampling.py"),
        ("kairyu.bench", "targets.py"),
        ("kairyu.bench", "types.py"),
    ]
    return {
        "name": name,
        "binary_outcomes": name in {"gsm8k", "scicode"},
        "paired_cluster_key": (
            "item_id_before_final_dot_v1" if name == "scicode" else None
        ),
        "uses_judge_template": False,
        "dataset": f"fixture/{name}",
        "revision": "revision-1",
        "sha256": _digest(f"dataset:{name}"),
        "assets": [],
        "history_provenance": {
            "complete": True,
            "reason": None,
            "requires_content_addressed_execution": requires_execution,
        },
        "evaluation_protocol": {
            "resources": [
                {"package": package, "resource": resource, "sha256": _digest(resource)}
                for package, resource in resources
            ],
            "distributions": [],
            "executables": [],
        },
    }


def _run(
    run_id: str,
    target_name: str,
    served_label: str,
    served_digest: str,
    scores: tuple[float, ...],
    *,
    benchmark: str = "gsm8k",
    item_ids: tuple[str, ...] | None = None,
    model: str = "model-a",
    max_output_tokens: int = 8192,
    attempts: int = 1,
    comparable: bool = True,
    incomparable_reasons: tuple[str, ...] = (),
) -> tuple[dict, PairResult]:
    target = BenchTarget(
        name=target_name,
        base_url=f"http://{target_name}.example/v1",
        model=model,
        max_output_tokens=max_output_tokens,
        served_config=ServedConfigIdentity(label=served_label, sha256=served_digest),
    )
    execution = (
        ExecutionConfig(runner="docker", image="sha256:" + "d" * 64)
        if benchmark == "scicode"
        else ExecutionConfig()
    )
    config = BenchConfig(
        suite="core" if benchmark in {"gsm8k", "mmlu"} else "accuracy",
        targets=(target,),
        only=(benchmark,),
        run_id=run_id,
        attempts=attempts,
        download=False,
        progress=False,
        execution=execution,
    )
    adapter_identity = _adapter_identity(benchmark)
    identity = _run_identity(config, [adapter_identity])
    fingerprint = _run_fingerprint(identity)
    environment = {
        "git_commit": "a" * 40,
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "git-checkout",
        "source_tree_clean": True,
        "source_tree_sha256": "b" * 64,
        "kairyu_version": "0.1.0",
        "python": "3.12.3",
        "created_at": f"2026-08-06T00:00:0{0 if run_id == 'base' else 1}+00:00",
        "execution": (
            {
                "runner": "docker",
                "image": "sha256:" + "d" * 64,
                "cpus": 1.0,
                "pids_limit": 64,
                "disk_mb": 256,
                "pull_policy": "never",
                "available": True,
                "availability_detail": "available",
                "resolved_image_id": "sha256:" + "d" * 64,
                "image_os": "linux",
                "image_architecture": "amd64",
                "image_variant": None,
            }
            if benchmark == "scicode"
            else {
                "runner": "local",
                "available": True,
                "availability_detail": "available",
            }
        ),
    }
    metadata = {
        "fingerprint": fingerprint,
        "identity": identity,
        "config": _recordable_config(config),
        "environment": environment,
        "run_id": run_id,
    }
    ids = item_ids or tuple(f"item-{index}" for index in range(len(scores)))
    items = tuple(
        ItemResult(item_id=item_id, status="completed", score=score)
        for item_id, score in zip(ids, scores, strict=True)
    )
    pair = PairResult(
        benchmark=benchmark,
        target=target_name,
        run_fingerprint=fingerprint,
        source_identity=history.source_identity(environment),
        status="completed",
        metrics={
            "score": sum(scores) / len(scores),
            "n_total": len(scores),
            "n_scored": len(scores),
        },
        items=items,
        methodology={"protocol": "fixed"},
        comparable=comparable,
        incomparable_reasons=incomparable_reasons,
    )
    board = build_scoreboard(
        run_id=run_id,
        suite=config.suite,
        config=metadata["config"],
        environment=environment,
        fingerprint=fingerprint,
        pairs=[pair],
        targets=[target_name],
        target_configs=[target],
        judge=config.judge,
    )
    return build_entry(board, metadata, pairs=[pair], previous_record_sha256=None), pair


def _binary_pair(**candidate_kwargs) -> tuple[dict, PairResult, dict, PairResult]:
    baseline_entry, baseline_pair = _run(
        "base",
        "baseline-arm",
        "bf16",
        "1" * 64,
        (1, 1, 1, 0, 0, 0),
    )
    candidate_entry, candidate_pair = _run(
        "candidate",
        "candidate-arm",
        "fp8",
        "2" * 64,
        candidate_kwargs.pop("scores", (1, 1, 0, 1, 0, 0)),
        **candidate_kwargs,
    )
    return baseline_entry, baseline_pair, candidate_entry, candidate_pair


def _rewrite_adapter_identity(
    entry: dict,
    pair: PairResult,
    mutate_adapter,
    *,
    self_judged: bool = False,
) -> tuple[dict, PairResult]:
    run = deepcopy(entry["run"])
    mutate_adapter(run["identity"]["adapters"][0])
    fingerprint = _run_fingerprint(run["identity"])
    run["fingerprint"] = fingerprint
    scoreboard = deepcopy(entry["scoreboard"])
    scoreboard["fingerprint"] = fingerprint
    if self_judged:
        scoreboard["self_judged_targets"] = [pair.target]
    rewritten_pair = pair.model_copy(update={"run_fingerprint": fingerprint})
    return (
        build_entry(
            scoreboard,
            run,
            pairs=[rewritten_pair],
            previous_record_sha256=None,
        ),
        rewritten_pair,
    )


def _compare(
    baseline_entry: dict,
    baseline_pair: PairResult,
    candidate_entry: dict,
    candidate_pair: PairResult,
    *,
    tolerance: float = 1.0,
) -> dict:
    return build_config_comparison(
        baseline_entry,
        candidate_entry,
        baseline_pairs={baseline_pair.benchmark: baseline_pair},
        candidate_pairs={candidate_pair.benchmark: candidate_pair},
        tolerances={baseline_pair.benchmark: tolerance},
    )


def test_different_target_fingerprints_form_one_item_bound_gate():
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = _binary_pair()

    comparison = _compare(
        baseline_entry, baseline_pair, candidate_entry, candidate_pair, tolerance=0.5
    )

    assert baseline_entry["fingerprint"] != candidate_entry["fingerprint"]
    assert comparison["comparison_type"] == "config_ab"
    assert comparison["baseline"]["served_config"] == {
        "label": "bf16",
        "sha256": "1" * 64,
    }
    assert comparison["candidate"]["served_config"]["label"] == "fp8"
    assert comparison["benchmarks"][0]["statistics"]["counts"] == {
        "both_pass": 2,
        "candidate_win": 1,
        "candidate_loss": 1,
        "both_fail": 2,
    }
    assert comparison["overall_verdict"] == "pass"
    assert comparison["comparison_sha256"]
    assert {
        resource["resource"]
        for resource in comparison["comparison_protocol"]["resources"]
    } == {
        "aggregate.py",
        "config_ab.py",
        "config_compare.py",
        "history.py",
        "store.py",
        "types.py",
    }
    assert comparison["comparison_runtime"]["measurement_python"] == "3.12.3"
    assert comparison["comparison_runtime"]["comparator_python"] == (
        platform.python_version()
    )


def test_renderer_states_noninferiority_and_operator_declared_boundary():
    values = _binary_pair()
    comparison = _compare(*values, tolerance=0.5)

    markdown = render_config_comparison_markdown(comparison)

    assert markdown.startswith("# Configuration A/B quality gate")
    assert "candidate `candidate` minus baseline `base`" in markdown
    assert "one-sided 95% lower confidence bound" in markdown
    assert "Served configuration identity is operator-declared" in markdown
    assert "newcombe_wilson_paired_method_10" in markdown


def test_real_index_pair_reload_cli_and_candidate_artifact_round_trip(tmp_path, capsys):
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = _binary_pair()
    for entry, pair in (
        (baseline_entry, baseline_pair),
        (candidate_entry, candidate_pair),
    ):
        store = ResultStore(tmp_path, entry["run_id"])
        store.save_pair(pair)
        store.append_scoreboard_index(
            entry["scoreboard"], entry["run"], pairs=[pair]
        )
    args = cli._build_parser().parse_args(
        [
            "bench",
            "compare",
            "--baseline",
            "base",
            "--candidate",
            "candidate",
            "--suite",
            "core",
            "--results-dir",
            str(tmp_path),
            "--tolerance",
            "gsm8k=50",
        ]
    )

    assert handle(args) == 0

    artifact = ResultStore(tmp_path, "candidate").run_dir / "config-comparison.json"
    stored = json.loads(artifact.read_text(encoding="utf-8"))
    assert stored["overall_verdict"] == "pass"
    assert stored["benchmarks"][0]["evidence"]["baseline_pair_sha256"]
    assert (artifact.parent / "config-comparison.md").exists()
    assert "Configuration A/B quality gate" in capsys.readouterr().out


def test_gate_uses_unrounded_one_sided_lower_bound():
    values = _binary_pair()
    failed = _compare(*values, tolerance=0.01)
    boundary = -failed["benchmarks"][0]["statistics"]["one_sided"]["lower"]

    passed = _compare(*values, tolerance=boundary)

    assert failed["overall_verdict"] == "fail"
    assert failed["benchmarks"][0]["gate"]["reason"] == (
        "insufficient evidence of non-inferiority"
    )
    assert passed["overall_verdict"] == "pass"


def test_artifact_builder_rejects_pair_not_named_by_index():
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = _binary_pair()
    tampered_items = tuple(
        item.model_copy(update={"score": 0.0}) for item in baseline_pair.items
    )
    tampered_pair = baseline_pair.model_copy(
        update={
            "items": tampered_items,
            "metrics": {
                "score": 0.0,
                "n_total": len(tampered_items),
                "n_scored": len(tampered_items),
            },
        }
    )

    with pytest.raises(
        ConfigComparisonError,
        match="baseline pair does not match its complete indexed binding",
    ):
        _compare(
            baseline_entry,
            tampered_pair,
            candidate_entry,
            candidate_pair,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param("same-served", "digests are identical", id="same-served-config"),
        pytest.param("missing-served", "served_config must be an object", id="anonymous"),
        pytest.param("model", "target request policy differs", id="request-model"),
        pytest.param("methodology", "methodology differs", id="run-methodology"),
        pytest.param("source", "source_tree_sha256", id="source-tree"),
    ],
)
def test_configuration_and_methodology_provenance_fail_closed(mutation, message):
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = _binary_pair()
    if mutation == "same-served":
        target = candidate_entry["run"]["identity"]["config"]["targets"][0]
        target["served_config"]["sha256"] = "1" * 64
        candidate_entry["run"]["config"]["targets"][0]["served_config"]["sha256"] = (
            "1" * 64
        )
        candidate_entry["scoreboard"]["config"] = deepcopy(candidate_entry["run"]["config"])
        candidate_entry["run"]["fingerprint"] = _run_fingerprint(
            candidate_entry["run"]["identity"]
        )
        candidate_entry["fingerprint"] = candidate_entry["run"]["fingerprint"]
        candidate_entry["scoreboard"]["fingerprint"] = candidate_entry["run"]["fingerprint"]
        candidate_pair = candidate_pair.model_copy(
            update={"run_fingerprint": candidate_entry["run"]["fingerprint"]}
        )
        candidate_entry = build_entry(
            candidate_entry["scoreboard"],
            candidate_entry["run"],
            pairs=[candidate_pair],
            previous_record_sha256=None,
        )
    elif mutation == "missing-served":
        candidate_entry["run"]["identity"]["config"]["targets"][0]["served_config"] = None
        candidate_entry["run"]["config"]["targets"][0]["served_config"] = None
        candidate_entry["scoreboard"]["config"] = deepcopy(candidate_entry["run"]["config"])
        candidate_entry["run"]["fingerprint"] = _run_fingerprint(
            candidate_entry["run"]["identity"]
        )
        candidate_entry["fingerprint"] = candidate_entry["run"]["fingerprint"]
        candidate_entry["scoreboard"]["fingerprint"] = candidate_entry["run"]["fingerprint"]
        candidate_pair = candidate_pair.model_copy(
            update={"run_fingerprint": candidate_entry["run"]["fingerprint"]}
        )
        candidate_entry = build_entry(
            candidate_entry["scoreboard"],
            candidate_entry["run"],
            pairs=[candidate_pair],
            previous_record_sha256=None,
        )
    elif mutation == "model":
        # Valid independently, but no longer an isolated deployment-config variable.
        candidate_entry, candidate_pair = _run(
            "candidate",
            "candidate-arm",
            "fp8",
            "2" * 64,
            (1, 1, 0, 1, 0, 0),
            model="other-model",
        )
    elif mutation == "methodology":
        candidate_entry["run"]["identity"]["config"]["seed"] = 7
        candidate_entry["run"]["config"]["seed"] = 7
        candidate_entry["scoreboard"]["config"]["seed"] = 7
        candidate_entry["run"]["fingerprint"] = _run_fingerprint(
            candidate_entry["run"]["identity"]
        )
        candidate_entry["fingerprint"] = candidate_entry["run"]["fingerprint"]
        candidate_entry["scoreboard"]["fingerprint"] = candidate_entry["run"]["fingerprint"]
        candidate_pair = candidate_pair.model_copy(
            update={"run_fingerprint": candidate_entry["run"]["fingerprint"]}
        )
        candidate_entry = build_entry(
            candidate_entry["scoreboard"],
            candidate_entry["run"],
            pairs=[candidate_pair],
            previous_record_sha256=None,
        )
    else:
        candidate_entry["run"]["environment"]["source_tree_sha256"] = "c" * 64
        candidate_entry["scoreboard"]["environment"]["source_tree_sha256"] = "c" * 64
        candidate_pair = candidate_pair.model_copy(
            update={
                "source_identity": history.source_identity(
                    candidate_entry["run"]["environment"]
                )
            }
        )
        candidate_entry = build_entry(
            candidate_entry["scoreboard"],
            candidate_entry["run"],
            pairs=[candidate_pair],
            previous_record_sha256=None,
        )

    with pytest.raises(ConfigComparisonError, match=message):
        _compare(baseline_entry, baseline_pair, candidate_entry, candidate_pair)


def test_multi_attempt_sampling_evidence_is_not_accepted_as_config_ab():
    baseline_entry, baseline_pair = _run(
        "base",
        "baseline-arm",
        "bf16",
        "1" * 64,
        (1, 1, 1, 0, 0, 0),
        attempts=2,
        benchmark="mmlu",
    )
    candidate_entry, candidate_pair = _run(
        "candidate",
        "candidate-arm",
        "fp8",
        "2" * 64,
        (1, 1, 0, 1, 0, 0),
        attempts=2,
        benchmark="mmlu",
    )

    with pytest.raises(
        ConfigComparisonError,
        match=r"baseline configuration A/B requires attempts=1; got 2",
    ):
        _compare(baseline_entry, baseline_pair, candidate_entry, candidate_pair)


def test_item_ids_align_by_identity_not_order():
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = _binary_pair()
    candidate_pair = candidate_pair.model_copy(
        update={"items": tuple(reversed(candidate_pair.items))}
    )
    # Rebuild the immutable candidate binding around the order-changed raw pair.
    candidate_entry = build_entry(
        candidate_entry["scoreboard"],
        candidate_entry["run"],
        pairs=[candidate_pair],
        previous_record_sha256=None,
    )

    comparison = _compare(
        baseline_entry, baseline_pair, candidate_entry, candidate_pair, tolerance=0.5
    )

    assert comparison["benchmarks"][0]["n_pairs"] == 6


def test_same_denominator_with_different_item_ids_is_an_error():
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = _binary_pair()
    changed = candidate_pair.items[-1].model_copy(update={"item_id": "other-item"})
    candidate_pair = candidate_pair.model_copy(
        update={"items": (*candidate_pair.items[:-1], changed)}
    )
    candidate_entry = build_entry(
        candidate_entry["scoreboard"],
        candidate_entry["run"],
        pairs=[candidate_pair],
        previous_record_sha256=None,
    )

    with pytest.raises(ConfigComparisonError, match="item-id sets differ"):
        _compare(baseline_entry, baseline_pair, candidate_entry, candidate_pair)


def test_duplicate_item_id_and_fractional_binary_evidence_are_errors():
    values = _binary_pair()
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = values
    duplicate = candidate_pair.items[1].model_copy(
        update={"item_id": candidate_pair.items[0].item_id}
    )
    duplicate_pair = candidate_pair.model_copy(
        update={"items": (candidate_pair.items[0], duplicate, *candidate_pair.items[2:])}
    )
    with pytest.raises(ConfigComparisonError, match="complete indexed binding"):
        _compare(baseline_entry, baseline_pair, candidate_entry, duplicate_pair)

    fractional_entry, fractional_pair = _run(
        "candidate",
        "candidate-arm",
        "fp8",
        "2" * 64,
        (0.5, 1, 0, 1, 0, 0),
    )
    with pytest.raises(ConfigComparisonError, match="fractional item evidence"):
        _compare(baseline_entry, baseline_pair, fractional_entry, fractional_pair)


def test_mrcr_uses_paired_bounded_bootstrap_not_binomial_wilson():
    baseline_entry, baseline_pair = _run(
        "base",
        "baseline-arm",
        "fp16-kv",
        "1" * 64,
        (0.2, 0.4, 0.8, 1.0),
        benchmark="mrcr-v2",
        item_ids=("d", "b", "a", "c"),
    )
    candidate_entry, candidate_pair = _run(
        "candidate",
        "candidate-arm",
        "fp8-kv",
        "2" * 64,
        (0.3, 0.5, 0.7, 1.0),
        benchmark="mrcr-v2",
        item_ids=("d", "b", "a", "c"),
    )

    comparison = _compare(
        baseline_entry, baseline_pair, candidate_entry, candidate_pair, tolerance=0.5
    )
    row = comparison["benchmarks"][0]

    assert row["metric_kind"] == "bounded_score"
    assert row["statistics"]["method"] == "paired_percentile_bootstrap_v1"
    assert row["statistics"]["resamples"] == 20_000
    assert row["statistics"]["item_ids_sha256"] == row["item_ids_sha256"]


def test_scicode_resamples_dependent_problem_clusters_not_substeps():
    item_ids = (
        "scicode-1.1",
        "scicode-1.2",
        "scicode-2.1",
        "scicode-3.1",
        "scicode-3.2",
        "scicode-3.3",
    )
    baseline_entry, baseline_pair = _run(
        "base",
        "baseline-arm",
        "bf16",
        "1" * 64,
        (1, 1, 0, 0, 0, 0),
        benchmark="scicode",
        item_ids=item_ids,
    )
    candidate_entry, candidate_pair = _run(
        "candidate",
        "candidate-arm",
        "nvfp4",
        "2" * 64,
        (0, 0, 1, 1, 1, 1),
        benchmark="scicode",
        item_ids=item_ids,
    )

    comparison = _compare(
        baseline_entry,
        baseline_pair,
        candidate_entry,
        candidate_pair,
        tolerance=1.0,
    )
    row = comparison["benchmarks"][0]

    assert row["metric_kind"] == "clustered_binary"
    assert row["paired_cluster_key"] == "item_id_before_final_dot_v1"
    assert row["statistics"]["method"] == "paired_cluster_percentile_bootstrap_v1"
    assert row["statistics"]["n_pairs"] == 6
    assert row["statistics"]["n_clusters"] == 3


def test_cluster_identity_rule_fails_closed_for_unknown_method_or_item_shape():
    with pytest.raises(ConfigComparisonError, match="unsupported paired cluster key"):
        _paired_cluster_ids(("scicode-1.1",), "future-cluster-rule")
    with pytest.raises(ConfigComparisonError, match="does not satisfy paired cluster key"):
        _paired_cluster_ids(("scicode-no-step",), "item_id_before_final_dot_v1")


def test_comparison_does_not_consult_later_adapter_registry(monkeypatch):
    values = _binary_pair()

    monkeypatch.setattr(
        "kairyu.bench.adapters.all_adapters",
        lambda: (_ for _ in ()).throw(AssertionError("current registry consulted")),
    )

    comparison = _compare(*values, tolerance=0.5)
    assert comparison["benchmarks"][0]["metric_kind"] == "binary"


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("paired_cluster_key", "lacks a cluster declaration"),
        ("uses_judge_template", "lacks a judge-use declaration"),
    ],
)
def test_historical_statistical_declarations_must_be_explicit(field, message):
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = _binary_pair()

    def remove_field(adapter):
        del adapter[field]

    baseline_entry, baseline_pair = _rewrite_adapter_identity(
        baseline_entry,
        baseline_pair,
        remove_field,
    )

    with pytest.raises(ConfigComparisonError, match=message):
        _compare(baseline_entry, baseline_pair, candidate_entry, candidate_pair)


def test_historical_judge_use_declaration_enforces_self_judging_boundary():
    baseline_entry, baseline_pair, candidate_entry, candidate_pair = _binary_pair()

    def declare_judge(adapter):
        adapter["uses_judge_template"] = True
        adapter["judge_template"] = {
            "name": "fixture-judge",
            "variant": "test",
            "sha256": "f" * 64,
        }

    baseline_entry, baseline_pair = _rewrite_adapter_identity(
        baseline_entry,
        baseline_pair,
        declare_judge,
        self_judged=True,
    )
    candidate_entry, candidate_pair = _rewrite_adapter_identity(
        candidate_entry,
        candidate_pair,
        declare_judge,
    )

    with pytest.raises(ConfigComparisonError, match="baseline target is self-judged"):
        _compare(baseline_entry, baseline_pair, candidate_entry, candidate_pair)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), 1.1, True])
def test_tolerance_is_a_finite_score_fraction(bad):
    values = _binary_pair()

    with pytest.raises(ConfigComparisonError, match="tolerance"):
        _compare(*values, tolerance=bad)
