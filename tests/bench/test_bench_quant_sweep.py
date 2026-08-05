"""Complete, source-bound quantization x task-accuracy sweep artifacts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from kairyu.bench import history
from kairyu.bench.adapters import QUANTIZATION_ROW_ORDER
from kairyu.bench.aggregate import build_scoreboard
from kairyu.bench.cli import handle
from kairyu.bench.history import build_entry
from kairyu.bench.quant_sweep import (
    QuantizationSweepError,
    build_quantization_sweep,
    render_quantization_sweep_markdown,
    required_quantization_profiles,
)
from kairyu.bench.runner import _recordable_config, _run_fingerprint, _run_identity
from kairyu.bench.store import ResultStore
from kairyu.bench.types import (
    BenchConfig,
    BenchTarget,
    ItemResult,
    PairResult,
    QuantizationProfile,
    ServedConfigIdentity,
)
from kairyu.entrypoints import cli


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _adapter_identity(name: str) -> dict:
    resources = [
        ("kairyu.bench.adapters", f"{name.replace('-', '_')}.py"),
        ("kairyu.bench.adapters", "base.py"),
        ("kairyu.bench", "aggregate.py"),
        ("kairyu.bench", "cache.py"),
        ("kairyu.bench", "runner.py"),
        ("kairyu.bench", "targets.py"),
        ("kairyu.bench", "types.py"),
    ]
    return {
        "name": name,
        "binary_outcomes": True,
        "paired_cluster_key": None,
        "uses_judge_template": False,
        "dataset": f"fixture/{name}",
        "revision": "1" * 40,
        "sha256": _digest(f"dataset:{name}"),
        "assets": [],
        "history_provenance": {
            "complete": True,
            "reason": None,
            "requires_content_addressed_execution": False,
        },
        "evaluation_protocol": {
            "resources": [
                {
                    "package": package,
                    "resource": resource,
                    "sha256": _digest(f"{package}:{resource}"),
                }
                for package, resource in resources
            ],
            "distributions": [],
            "executables": [],
        },
    }


def _environment() -> dict:
    return {
        "git_commit": "a" * 40,
        "git_commit_role": "local_benchmark_harness",
        "source_kind": "git-checkout",
        "source_tree_clean": True,
        "source_tree_sha256": "b" * 64,
        "kairyu_version": "0.1.0",
        "python": "3.12.3",
        "created_at": "2026-08-06T00:00:00+00:00",
        "execution": {
            "runner": "local",
            "available": True,
            "availability_detail": "available",
        },
    }


def _make_sweep_evidence(
    *,
    run_id: str = "quant-run",
    profiles: tuple[dict[str, object], ...] | None = None,
    duplicate_manifest: bool = False,
    model_overrides: dict[str, str] | None = None,
    score_overrides: dict[tuple[str, str], tuple[float, ...]] | None = None,
    status_overrides: dict[tuple[str, str], str] | None = None,
) -> tuple[dict, dict[str, dict[str, PairResult]]]:
    profiles = profiles or required_quantization_profiles()
    model_overrides = model_overrides or {}
    score_overrides = score_overrides or {}
    status_overrides = status_overrides or {}
    targets: list[BenchTarget] = []
    for declaration in profiles:
        key = declaration["key"]
        profile = QuantizationProfile.model_validate(declaration["profile"])
        digest = _digest("manifest:shared" if duplicate_manifest else f"manifest:{key}")
        targets.append(
            BenchTarget(
                name=str(key),
                base_url=f"http://{key}.example/v1",
                model=model_overrides.get(str(key), "base-model"),
                served_config=ServedConfigIdentity(
                    label=f"{key}-deployment",
                    sha256=digest,
                ),
                quantization=profile,
            )
        )

    config = BenchConfig(
        suite="quantization",
        targets=tuple(targets),
        run_id=run_id,
        results_dir="bench/results/quantization",
        download=False,
        progress=False,
    )
    adapter_identities = [_adapter_identity(name) for name in QUANTIZATION_ROW_ORDER]
    identity = _run_identity(config, adapter_identities)
    fingerprint = _run_fingerprint(identity)
    environment = _environment()
    metadata = {
        "fingerprint": fingerprint,
        "identity": identity,
        "config": _recordable_config(config),
        "environment": environment,
        "run_id": run_id,
    }
    default_scores = (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
    pairs: list[PairResult] = []
    matrix: dict[str, dict[str, PairResult]] = {target.label(): {} for target in targets}
    for benchmark in QUANTIZATION_ROW_ORDER:
        for target in targets:
            key = (target.label(), benchmark)
            status = status_overrides.get(key, "completed")
            if status == "completed":
                scores = score_overrides.get(key, default_scores)
                items = tuple(
                    ItemResult(
                        item_id=f"{benchmark}-item-{index}",
                        status="completed",
                        score=score,
                    )
                    for index, score in enumerate(scores)
                )
                metrics = {
                    "score": sum(scores) / len(scores),
                    "n_total": len(scores),
                    "n_scored": len(scores),
                }
                reason = None
            else:
                items = ()
                metrics = {"score": None, "n_total": 0, "n_scored": 0}
                reason = "fixture endpoint cannot evaluate this task"
            pair = PairResult(
                benchmark=benchmark,
                target=target.label(),
                run_fingerprint=fingerprint,
                source_identity=history.source_identity(environment),
                status=status,
                reason=reason,
                metrics=metrics,
                items=items,
                methodology={"protocol": f"{benchmark}-v1"},
                comparable=benchmark != "mmlu",
                incomparable_reasons=(
                    ()
                    if benchmark != "mmlu"
                    else ("zero-shot MMLU is not the canonical five-shot protocol",)
                ),
            )
            pairs.append(pair)
            matrix[target.label()][benchmark] = pair
    board = build_scoreboard(
        run_id=run_id,
        suite="quantization",
        config=metadata["config"],
        environment=environment,
        fingerprint=fingerprint,
        pairs=pairs,
        targets=[target.label() for target in targets],
        target_configs=targets,
        judge=config.judge,
    )
    entry = build_entry(
        board,
        metadata,
        pairs=pairs,
        previous_record_sha256=None,
    )
    return entry, matrix


def _tolerances(value: float = 1.0) -> dict[str, float]:
    return {benchmark: value for benchmark in QUANTIZATION_ROW_ORDER}


def test_complete_sweep_reuses_six_exact_config_comparisons():
    entry, pairs = _make_sweep_evidence()

    sweep = build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances())

    assert sweep["sweep_type"] == "quantization_task_accuracy"
    assert sweep["coverage"]["complete"] is True
    assert [arm["key"] for arm in sweep["arms"]] == [
        "bf16",
        "fp8",
        "int8",
        "awq",
        "gptq",
        "nvfp4",
        "fp8-kv",
    ]
    assert sweep["arms"][0]["profile"] == {
        "weight_method": "none",
        "compute_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
    }
    assert len(sweep["config_comparisons"]) == 6
    assert all(
        list(arm["tasks"]) == list(QUANTIZATION_ROW_ORDER)
        for arm in sweep["arms"]
    )
    assert sweep["overall_verdict"] == "pass"
    assert sweep["identity_evidence"] == "operator_declared_not_remotely_attested"
    assert sweep["native_support"]["fp8_e4m3_kv"]["status"] == (
        "disabled_in_this_kairyu_source"
    )
    assert sweep["sweep_sha256"]
    assert {
        resource["resource"] for resource in sweep["sweep_protocol"]["resources"]
    } >= {
        "kairyu/bench/quant_sweep.py",
        "kairyu/bench/config_ab.py",
        "kairyu/engine/core/quant_config.py",
        "kairyu/engine/core/kv_cache_dtype.py",
    }


def test_renderer_has_one_accuracy_row_per_profile_and_all_gate_cells():
    entry, pairs = _make_sweep_evidence()
    sweep = build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances())

    markdown = render_quantization_sweep_markdown(sweep)

    assert markdown.startswith("# Quantization x task-accuracy sweep")
    assert "Deployment and quantization identities are operator-declared" in markdown
    assert "does not claim native support" in markdown
    assert "Tasks and schemes are never averaged" in markdown
    assert "| BF16 | `none` | `bfloat16` | `bfloat16` |" in markdown
    assert "| BF16 + FP8 KV | `none` | `bfloat16` | `fp8_e4m3` |" in markdown
    assert markdown.count("| FP8 | GSM8K |") == 1
    assert markdown.count("| NVFP4 | GPQA Diamond |") == 1
    assert sweep["sweep_sha256"] in markdown


@pytest.mark.parametrize("run_id", ["unsafe`run", "unsafe\n# forged heading"])
def test_report_unsafe_run_id_is_rejected(run_id):
    entry, pairs = _make_sweep_evidence(run_id=run_id)

    with pytest.raises(QuantizationSweepError, match="run id must not contain"):
        build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances())

    safe_entry, safe_pairs = _make_sweep_evidence()
    sweep = build_quantization_sweep(
        safe_entry, pairs=safe_pairs, tolerances=_tolerances()
    )
    sweep["run"]["run_id"] = run_id
    with pytest.raises(QuantizationSweepError, match="run id must not contain"):
        render_quantization_sweep_markdown(sweep)


def test_profile_order_in_run_does_not_change_canonical_sweep_rows():
    reversed_profiles = tuple(reversed(required_quantization_profiles()))
    entry, pairs = _make_sweep_evidence(profiles=reversed_profiles)

    sweep = build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances())

    assert [arm["key"] for arm in sweep["arms"]] == [
        declaration["key"] for declaration in required_quantization_profiles()
    ]


def test_missing_profile_and_duplicate_manifest_fail_closed():
    incomplete = required_quantization_profiles()[:-1]
    entry, pairs = _make_sweep_evidence(profiles=incomplete)
    with pytest.raises(QuantizationSweepError, match="coverage is incomplete"):
        build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances())

    entry, pairs = _make_sweep_evidence(duplicate_manifest=True)
    with pytest.raises(QuantizationSweepError, match="distinct served_config digest"):
        build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances())


def test_task_policy_must_be_complete_and_has_no_scheme_specific_margin():
    entry, pairs = _make_sweep_evidence()
    missing = _tolerances()
    missing.pop("gpqa-diamond")
    with pytest.raises(QuantizationSweepError, match="exactly cover"):
        build_quantization_sweep(entry, pairs=pairs, tolerances=missing)

    extra = {**_tolerances(), "other": 0.1}
    with pytest.raises(QuantizationSweepError, match="exactly cover"):
        build_quantization_sweep(entry, pairs=pairs, tolerances=extra)


def test_request_policy_and_incomplete_pairs_are_rejected_by_config_ab_core():
    entry, pairs = _make_sweep_evidence(model_overrides={"fp8": "other-model"})
    with pytest.raises(ValueError, match="target request policy differs"):
        build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances())

    entry, pairs = _make_sweep_evidence(
        status_overrides={("fp8", "gpqa-diamond"): "skipped"}
    )
    with pytest.raises(ValueError, match="candidate pair is not a complete measurement"):
        build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances())


def test_pair_not_named_by_index_is_rejected():
    entry, pairs = _make_sweep_evidence()
    tampered = pairs["fp8"]["gsm8k"].model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"score": 0.0})
                for item in pairs["fp8"]["gsm8k"].items
            ),
            "metrics": {"score": 0.0, "n_total": 8, "n_scored": 8},
        }
    )
    changed = deepcopy(pairs)
    changed["fp8"]["gsm8k"] = tampered

    with pytest.raises(ValueError, match="complete indexed binding"):
        build_quantization_sweep(entry, pairs=changed, tolerances=_tolerances())


def test_valid_quality_failure_is_retained_as_fail_not_input_error():
    overrides = {
        ("fp8", benchmark): (0.0,) * 8 for benchmark in QUANTIZATION_ROW_ORDER
    }
    entry, pairs = _make_sweep_evidence(score_overrides=overrides)

    sweep = build_quantization_sweep(entry, pairs=pairs, tolerances=_tolerances(0.0))

    fp8 = next(arm for arm in sweep["arms"] if arm["key"] == "fp8")
    assert fp8["overall_verdict"] == "fail"
    assert all(task["gate"]["verdict"] == "fail" for task in fp8["tasks"].values())
    assert sweep["overall_verdict"] == "fail"


def test_real_index_raw_pair_reload_and_cli_artifact_round_trip(tmp_path, capsys):
    entry, matrix = _make_sweep_evidence()
    store = ResultStore(tmp_path, "quant-run")
    all_pairs = [
        matrix[target][benchmark]
        for benchmark in QUANTIZATION_ROW_ORDER
        for target in entry["scoreboard"]["targets"]
    ]
    for pair in all_pairs:
        store.save_pair(pair)
    store.append_scoreboard_index(entry["scoreboard"], entry["run"], all_pairs)
    args = cli._build_parser().parse_args(
        [
            "bench",
            "quant-sweep",
            "--run",
            "quant-run",
            "--results-dir",
            str(tmp_path),
            "--tolerance",
            "gsm8k=100",
            "--tolerance",
            "mmlu=100",
            "--tolerance",
            "ifeval=100",
            "--tolerance",
            "gpqa-diamond=100",
        ]
    )

    assert handle(args) == 0

    artifact = store.run_dir / "quantization-sweep.json"
    stored = json.loads(artifact.read_text(encoding="utf-8"))
    assert stored["overall_verdict"] == "pass"
    assert stored["run"]["record_sha256"]
    assert (store.run_dir / "quantization-sweep.md").exists()
    assert "Quantization x task-accuracy sweep" in capsys.readouterr().out
