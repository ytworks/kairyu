"""Published-gold judge agreement, order bias, and provenance evidence."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from kairyu.bench.calibration import (
    CalibrationCase,
    _calibration_inputs_from_args,
    load_calibration_cases,
    run_calibration_cli,
    run_judge_calibration,
)
from kairyu.bench.judge import judge_protocol_identity
from kairyu.bench.judge_prompts import judge_template_identity
from kairyu.bench.runner import _run_fingerprint, _run_identity
from kairyu.bench.types import (
    BenchConfig,
    BenchTarget,
    JudgeConfig,
    JudgeEndpointConfig,
)


def _write_cases(tmp_path, rows):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _small_rows():
    return [
        {
            "id": "hle-right",
            "template": "hle",
            "question": "Q1",
            "expected": "EXPECTED",
            "response": "Final answer: RIGHT",
            "human_correct": True,
        },
        {
            "id": "hle-wrong",
            "template": "hle",
            "question": "Q1",
            "expected": "EXPECTED",
            "response": "Final answer: WRONG",
            "human_correct": False,
        },
        {
            "id": "char-right",
            "template": "charxiv-reasoning",
            "question": "Q2",
            "expected": "EXPECTED",
            "response": "Final answer: RIGHT",
            "human_correct": True,
        },
        {
            "id": "char-wrong",
            "template": "charxiv-reasoning",
            "question": "Q2",
            "expected": "EXPECTED",
            "response": "Final answer: WRONG",
            "human_correct": False,
        },
    ]


def _label_attestation(record: str) -> dict[str, str]:
    return {
        "label_provenance": "human-reviewed",
        "label_source": "kairyu calibration review",
        "label_source_revision": "2026-08-04",
        "label_source_record": record,
        "label_source_license": "CC0-1.0",
    }


def _headline_rows(*, response_provenance: bool = False) -> list[dict]:
    rows = []
    for template, prefix in (("hle", "h"), ("charxiv-reasoning", "c")):
        for pair_index in range(6):
            for human_correct in (True, False):
                source_is_self = (
                    pair_index % 2 == 0
                    if human_correct
                    else pair_index % 2 == 1
                )
                source_name = "SELF" if source_is_self else "OTHER"
                case_id = (
                    f"{prefix}-{pair_index}-"
                    f"{'right' if human_correct else 'wrong'}"
                )
                row = {
                    "id": case_id,
                    "template": template,
                    "question": f"Question {prefix}-{pair_index}",
                    "expected": "EXPECTED",
                    "response": (
                        f"{source_name} Final answer: "
                        f"{'RIGHT' if human_correct else 'WRONG'}"
                    ),
                    "human_correct": human_correct,
                    **_label_attestation(case_id),
                }
                if response_provenance:
                    row.update(
                        {
                            "response_base_url": (
                                "http://judge/v1"
                                if source_is_self
                                else "http://other/v1"
                            ),
                            "response_model": (
                                "judge-model" if source_is_self else "other-model"
                            ),
                        }
                    )
                rows.append(row)
    return rows


def _target(*, base_url: str = "http://target/v1", model: str = "target-model"):
    return BenchTarget(base_url=base_url, model=model)


def _run_judge_adapter_identities():
    return tuple(
        {
            "name": name,
            "judge_template": judge_template_identity(name),
            "judge_protocol": judge_protocol_identity(),
        }
        for name in ("hle", "charxiv-reasoning")
    )


def _factory(*, order_sensitive=False, unparseable_wrong=False):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        correct = "Final answer: RIGHT" in prompt
        if unparseable_wrong and not correct:
            reply = "I cannot decide"
        else:
            production_order = (
                prompt.index("[response]:") < prompt.index("[correct_answer]:")
                if "[response]:" in prompt
                else prompt.index("Ground-truth answer:")
                < prompt.index("Model response:")
            )
            if order_sensitive and not production_order:
                correct = not correct
            reply = "correct: yes" if correct else "correct: no"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": reply}}
                ]
            },
        )

    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _published_gold_factory():
    cases, _, _ = load_calibration_cases()
    rejected = tuple(case.response for case in cases if not case.human_correct)

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        correct = not any(response in prompt for response in rejected)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"correct: {'yes' if correct else 'no'}",
                        },
                    }
                ]
            },
        )

    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _write_bound_run(tmp_path, *, self_judged: bool = False):
    judge = JudgeConfig(
        base_url="http://judge/v1", model="judge-model", max_retries=0
    )
    target = BenchTarget(
        base_url="http://judge/v1" if self_judged else "http://target/v1",
        model="judge-model" if self_judged else "target-model",
    )
    config = BenchConfig(targets=(target,), judge=judge)
    identity = _run_identity(config, list(_run_judge_adapter_identities()))
    fingerprint = _run_fingerprint(identity)
    run_dir = tmp_path / ("self-run" if self_judged else "independent-run")
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "identity": identity,
                "config": config.model_dump(mode="json"),
                "run_id": run_dir.name,
            }
        ),
        encoding="utf-8",
    )
    return run_dir, judge, target, fingerprint


def _bound_cli_args(run_dir, output, **overrides):
    values = {
        "run": str(run_dir),
        "results_dir": str(run_dir.parent),
        "config": None,
        "judge_base_url": None,
        "judge_model": None,
        "judge_api_key_env": None,
        "judge_reasoning_effort": None,
        "judge_extra_body": None,
        "judge_secondary": None,
        "calibration_set": None,
        "output": str(output),
        "min_agreement": 11 / 12,
        "min_coverage": 1.0,
        "max_position_flip": 0.0,
        "max_self_preference_gap": 0.2,
        "require_self_preference": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_packaged_calibration_set_is_balanced_published_gold_evidence():
    cases, raw, source = load_calibration_cases()
    assert len(cases) == 24
    assert source.startswith("package:")
    assert hashlib.sha256(raw).hexdigest() == (
        "626ce9b255ea484a50fdd4653ad0968a65cc4da525bf11ea46b7e17ea244bf77"
    )
    assert len({case.id for case in cases}) == 24
    assert all(case.label_provenance == "published-gold" for case in cases)
    assert all(case.label_attested for case in cases)
    assert all(
        case.label_source
        and case.label_source_revision
        and case.label_source_record
        and case.label_source_license
        for case in cases
    )
    assert {case.label_source for case in cases} == {
        "princeton-nlp/LLMBar:Dataset/LLMBar/Natural/dataset.json"
    }
    assert {case.label_source_revision for case in cases} == {
        "900616bff90b6c6c8e1681f7d079250637c55992"
    }
    assert {case.label_source_license for case in cases} == {"MIT"}
    assert all(":label=" in (case.label_source_record or "") for case in cases)
    for template in ("hle", "charxiv-reasoning"):
        selected = [case for case in cases if case.template == template]
        assert len(selected) == 12
        assert sum(case.human_correct for case in selected) == 6
        assert len({(case.question, case.expected) for case in selected}) == 6
    assert all(case.response_identity() is None for case in cases)


def test_attested_label_provenance_requires_a_complete_source_record():
    base = {
        "id": "case",
        "template": "hle",
        "question": "Q",
        "expected": "A",
        "response": "A",
        "human_correct": True,
        "label_provenance": "human-reviewed",
    }
    with pytest.raises(ValueError, match="attested label provenance requires"):
        CalibrationCase.model_validate(base)

    case = CalibrationCase.model_validate(
        {**base, **_label_attestation("review-record")}
    )
    assert case.label_attested is True


@pytest.mark.parametrize("response_model", ["", "   ", " judge-model "])
def test_response_producer_identity_rejects_empty_or_padded_model(response_model):
    with pytest.raises(ValueError, match="response_model must be non-empty"):
        CalibrationCase.model_validate(
            {
                "id": "case",
                "template": "hle",
                "question": "Q",
                "expected": "A",
                "response": "A",
                "human_correct": True,
                "response_base_url": "http://producer/v1",
                "response_model": response_model,
            }
        )


async def test_small_legacy_calibration_can_pass_but_is_not_headline_eligible(
    tmp_path, monkeypatch
):
    path = _write_cases(tmp_path, _small_rows())
    monkeypatch.setenv("CALIBRATION_SECRET", "must-not-appear")
    config = JudgeConfig(
        base_url="http://judge/v1",
        model="judge-model",
        api_key_env="CALIBRATION_SECRET",
        max_retries=0,
    )
    report = await run_judge_calibration(
        config,
        calibration_set=path,
        min_agreement=1.0,
        http_factory=_factory(),
    )
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["headline_eligible"] is False
    assert report["headline_checks"]["run_bound"] is False
    assert report["headline_checks"]["calibration_set_headline_shape"] is False
    assert report["calibration_set"]["labels_attested"] is False
    assert report["agreement"]["production"]["agreement"] == 1.0
    assert report["agreement"]["counterbalanced"]["agreement"] == 1.0
    assert report["position_bias"] == {
        "both_parseable": 4,
        "flips": 0,
        "flip_rate": 0.0,
        "production_agreement": 1.0,
        "counterbalanced_agreement": 1.0,
        "agreement_gap": 0.0,
    }
    assert report["members"][0]["self_preference"]["status"] == "unavailable"
    serialized = json.dumps(report)
    assert "CALIBRATION_SECRET" in serialized
    assert "must-not-appear" not in serialized


async def test_attested_bound_calibration_with_independent_judge_is_headline_eligible(
    tmp_path,
):
    path = _write_cases(tmp_path, _headline_rows())
    report = await run_judge_calibration(
        JudgeConfig(
            base_url="http://judge/v1", model="judge-model", max_retries=0
        ),
        calibration_set=path,
        run_fingerprint="a" * 64,
        target_configs=[_target()],
        run_adapter_identities=_run_judge_adapter_identities(),
        http_factory=_factory(),
    )

    assert report["passed"] is True
    assert report["headline_eligible"] is True
    assert all(report["headline_checks"].values())
    assert report["calibration_set"]["headline_shape"] is True
    for summary in report["calibration_set"]["by_template"].values():
        assert summary["responses"] == 12
        assert summary["prompt_pairs"] == 6
        assert summary["headline_shape"] is True
    assert report["members"][0]["target_relationship"] == "nonself"
    assert report["members"][0]["self_preference"]["status"] == "not_applicable"
    assert report["run_binding"] == report["identity"]["run_binding"]
    assert report["run_binding"]["bound"] is True
    assert report["run_binding"]["run_fingerprint"] == "a" * 64
    assert report["run_binding"]["target_identities"] == [
        {
            "label": "target-model",
            "base_url": "http://target/v1",
            "model": "target-model",
        }
    ]


async def test_weaker_thresholds_can_pass_but_never_enable_headline_use(tmp_path):
    path = _write_cases(tmp_path, _headline_rows())
    report = await run_judge_calibration(
        JudgeConfig(
            base_url="http://judge/v1", model="judge-model", max_retries=0
        ),
        calibration_set=path,
        min_agreement=0.0,
        min_coverage=0.0,
        max_position_flip=1.0,
        max_self_preference_gap=1.0,
        run_fingerprint="b" * 64,
        target_configs=[_target()],
        run_adapter_identities=_run_judge_adapter_identities(),
        http_factory=_factory(),
    )

    assert report["passed"] is True
    assert report["headline_eligible"] is False
    assert (
        report["headline_checks"]["thresholds_not_weaker_than_standard"]
        is False
    )
    assert all(
        value
        for name, value in report["headline_checks"].items()
        if name != "thresholds_not_weaker_than_standard"
    )


async def test_matching_target_requires_measured_self_preference_for_headline(
    tmp_path,
):
    path = _write_cases(tmp_path, _headline_rows())
    report = await run_judge_calibration(
        JudgeConfig(
            base_url="http://judge/v1", model="judge-model", max_retries=0
        ),
        calibration_set=path,
        run_fingerprint="c" * 64,
        target_configs=[
            _target(base_url="http://judge/v1", model="judge-model")
        ],
        run_adapter_identities=_run_judge_adapter_identities(),
        http_factory=_factory(),
    )

    preference = report["members"][0]["self_preference"]
    assert report["passed"] is True
    assert report["headline_eligible"] is False
    assert report["members"][0]["target_relationship"] == "self"
    assert preference["status"] == "unavailable"
    assert preference["n_unknown_provenance"] == 24
    assert report["headline_checks"]["member_0_self_preference"] is False


async def test_panel_secondary_matching_target_requires_its_own_bias_evidence(
    tmp_path,
):
    report = await run_judge_calibration(
        JudgeConfig(
            base_url="http://primary/v1",
            model="primary",
            max_retries=0,
            additional_judges=(
                JudgeEndpointConfig(
                    base_url="http://target/v1",
                    model="target-model",
                    max_retries=0,
                ),
            ),
        ),
        calibration_set=_write_cases(tmp_path, _headline_rows()),
        run_fingerprint="3" * 64,
        target_configs=[_target()],
        run_adapter_identities=_run_judge_adapter_identities(),
        http_factory=_factory(),
    )

    assert report["passed"] is True
    assert report["headline_eligible"] is False
    assert [member["target_relationship"] for member in report["members"]] == [
        "nonself",
        "self",
    ]
    assert report["members"][0]["self_preference"]["status"] == "not_applicable"
    assert report["members"][1]["self_preference"]["status"] == "unavailable"
    assert report["headline_checks"]["member_0_self_preference"] is True
    assert report["headline_checks"]["member_1_self_preference"] is False


async def test_run_binding_validates_fingerprint_and_declared_target_identity(
    tmp_path,
):
    path = _write_cases(tmp_path, _small_rows())
    config = JudgeConfig(
        base_url="http://judge/v1", model="judge-model", max_retries=0
    )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        await run_judge_calibration(
            config,
            calibration_set=path,
            run_fingerprint="A" * 64,
            target_configs=[_target()],
            run_adapter_identities=_run_judge_adapter_identities(),
            http_factory=_factory(),
        )
    with pytest.raises(ValueError, match="do not match"):
        await run_judge_calibration(
            config,
            calibration_set=path,
            run_fingerprint="d" * 64,
            target_configs=[_target()],
            target_identities=[("http://different/v1", "different-model")],
            run_adapter_identities=_run_judge_adapter_identities(),
            http_factory=_factory(),
        )


async def test_unparseable_or_position_dependent_judge_fails_closed(tmp_path):
    path = _write_cases(tmp_path, _small_rows())
    config = JudgeConfig(
        base_url="http://judge/v1", model="judge-model", max_retries=0
    )
    unparseable = await run_judge_calibration(
        config,
        calibration_set=path,
        min_agreement=0.0,
        http_factory=_factory(unparseable_wrong=True),
    )
    assert unparseable["passed"] is False
    assert unparseable["agreement"]["production"]["unparseable"] == 2
    assert unparseable["checks"]["production_coverage"] is False

    positional = await run_judge_calibration(
        config,
        calibration_set=path,
        min_agreement=0.0,
        http_factory=_factory(order_sensitive=True),
    )
    assert positional["passed"] is False
    assert positional["position_bias"]["flips"] == 4
    assert positional["checks"]["position_flip_rate"] is False


async def test_calibration_cli_writes_failure_artifact_and_exits_nonzero(
    tmp_path, capsys
):
    cases = _write_cases(tmp_path, _small_rows())
    output = tmp_path / "evidence" / "calibration.json"
    args = SimpleNamespace(
        config=None,
        judge_base_url="http://judge/v1",
        judge_model="judge-model",
        judge_api_key_env=None,
        judge_reasoning_effort=None,
        judge_extra_body=None,
        judge_secondary=None,
        calibration_set=str(cases),
        output=str(output),
        min_agreement=0.0,
        min_coverage=1.0,
        max_position_flip=0.0,
        max_self_preference_gap=0.2,
        require_self_preference=False,
    )
    code = await run_calibration_cli(
        args, http_factory=_factory(unparseable_wrong=True)
    )
    assert code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
    assert '"passed": false' in capsys.readouterr().out


async def test_self_preference_requires_real_balanced_producer_provenance(tmp_path):
    path = _write_cases(tmp_path, _headline_rows(response_provenance=True))

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        production_order = (
            prompt.index("[response]:") < prompt.index("[correct_answer]:")
            if "[response]:" in prompt
            else prompt.index("Ground-truth answer:")
            < prompt.index("Model response:")
        )
        # Only the counterbalanced prompt accepts the judge's own wrong
        # responses. The worst-order gate therefore cannot hide the bias.
        predicted = "Final answer: RIGHT" in prompt or (
            "SELF Final answer: WRONG" in prompt and not production_order
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"correct: {'yes' if predicted else 'no'}",
                        },
                    }
                ]
            },
        )

    report = await run_judge_calibration(
        JudgeConfig(
            base_url="http://judge/v1", model="judge-model", max_retries=0
        ),
        calibration_set=path,
        min_agreement=0.0,
        max_position_flip=1.0,
        require_self_preference=True,
        run_fingerprint="e" * 64,
        target_configs=[
            _target(base_url="http://judge/v1", model="judge-model")
        ],
        run_adapter_identities=_run_judge_adapter_identities(),
        http_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
    )
    preference = report["members"][0]["self_preference"]
    assert preference["status"] == "measured"
    assert preference["n_self"] == 12
    assert preference["n_nonself"] == 12
    assert preference["n_unknown_provenance"] == 0
    assert all(
        count >= 2
        for template in preference["strata"].values()
        for label in template.values()
        for count in label.values()
    )
    assert preference["production"]["false_positive_gap"] == 0.0
    assert preference["counterbalanced"]["self_false_positive_rate"] == 1.0
    assert preference["counterbalanced"]["nonself_false_positive_rate"] == 0.0
    assert preference["counterbalanced"]["false_positive_gap"] == 1.0
    assert preference["max_self_favoring_gap"] == 1.0
    assert report["checks"]["member_0_self_preference"] is False
    assert report["headline_checks"]["member_0_self_preference"] is False
    assert report["passed"] is False


async def test_explicit_self_preference_gate_measures_an_independent_judge(tmp_path):
    report = await run_judge_calibration(
        JudgeConfig(
            base_url="http://judge/v1", model="judge-model", max_retries=0
        ),
        calibration_set=_write_cases(
            tmp_path, _headline_rows(response_provenance=True)
        ),
        require_self_preference=True,
        run_fingerprint="2" * 64,
        target_configs=[_target()],
        run_adapter_identities=_run_judge_adapter_identities(),
        http_factory=_factory(),
    )

    preference = report["members"][0]["self_preference"]
    assert report["members"][0]["target_relationship"] == "nonself"
    assert preference["status"] == "measured"
    assert preference["required_for_headline"] is True
    assert preference["max_self_favoring_gap"] == 0.0
    assert report["checks"]["member_0_self_preference"] is True
    assert report["headline_checks"]["member_0_self_preference"] is True
    assert report["headline_eligible"] is True


async def test_self_preference_measurement_rejects_unknown_or_thin_strata(tmp_path):
    config = JudgeConfig(
        base_url="http://judge/v1", model="judge-model", max_retries=0
    )

    unknown_rows = _headline_rows(response_provenance=True)
    unknown_rows[0].pop("response_base_url")
    unknown_rows[0].pop("response_model")
    unknown = await run_judge_calibration(
        config,
        calibration_set=_write_cases(tmp_path, unknown_rows),
        run_fingerprint="f" * 64,
        target_configs=[
            _target(base_url="http://judge/v1", model="judge-model")
        ],
        run_adapter_identities=_run_judge_adapter_identities(),
        http_factory=_factory(),
    )
    unknown_preference = unknown["members"][0]["self_preference"]
    assert unknown_preference["status"] == "unavailable"
    assert unknown_preference["n_unknown_provenance"] == 1

    thin_rows = _headline_rows(response_provenance=True)
    changed = 0
    for row in thin_rows:
        if (
            row["template"] == "hle"
            and row["human_correct"] is True
            and row["response_model"] == "judge-model"
            and changed < 2
        ):
            row["response_base_url"] = "http://other/v1"
            row["response_model"] = "other-model"
            changed += 1
    thin = await run_judge_calibration(
        config,
        calibration_set=_write_cases(tmp_path, thin_rows),
        run_fingerprint="1" * 64,
        target_configs=[
            _target(base_url="http://judge/v1", model="judge-model")
        ],
        run_adapter_identities=_run_judge_adapter_identities(),
        http_factory=_factory(),
    )
    thin_preference = thin["members"][0]["self_preference"]
    assert thin_preference["status"] == "unavailable"
    assert thin_preference["n_unknown_provenance"] == 0
    assert thin_preference["strata"]["hle"]["human_correct"]["self"] == 1
    assert "requires at least two responses" in thin_preference["reason"]


def test_bound_run_loader_verifies_fingerprint_and_fingerprinted_config(tmp_path):
    run_dir, judge, target, fingerprint = _write_bound_run(tmp_path)
    loaded_judge, loaded_fingerprint, loaded_targets, loaded_adapters = (
        _calibration_inputs_from_args(
            _bound_cli_args(run_dir, tmp_path / "calibration.json")
        )
    )
    assert loaded_judge == judge
    assert loaded_fingerprint == fingerprint
    assert tuple(loaded_targets or ()) == (target,)
    assert tuple(loaded_adapters or ()) == _run_judge_adapter_identities()

    by_id = _bound_cli_args(run_dir, tmp_path / "by-id.json")
    by_id.run = run_dir.name
    assert _calibration_inputs_from_args(by_id)[1] == fingerprint

    metadata_path = run_dir / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"]["judge"]["model"] = "tampered-model"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="identity does not match"):
        _calibration_inputs_from_args(
            _bound_cli_args(run_dir, tmp_path / "tampered.json")
        )

    metadata["config"] = metadata["identity"]["config"]
    metadata["fingerprint"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint does not match"):
        _calibration_inputs_from_args(
            _bound_cli_args(run_dir, tmp_path / "bad-fingerprint.json")
        )


@pytest.mark.parametrize("missing_key", ["judge", "targets"])
def test_bound_run_rejects_config_fields_omitted_from_immutable_identity(
    tmp_path, missing_key
):
    run_dir, _, _, _ = _write_bound_run(tmp_path)
    metadata_path = run_dir / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["identity"]["config"].pop(missing_key)
    metadata["fingerprint"] = _run_fingerprint(metadata["identity"])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="identity does not match"):
        _calibration_inputs_from_args(
            _bound_cli_args(run_dir, tmp_path / "calibration.json")
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-template", "lacks judge template/protocol"),
        ("stale-template", "does not match the current calibration template"),
        ("stale-protocol", "does not match the current calibration protocol"),
        ("duplicate-adapter", "duplicate adapter identity"),
        ("no-judge-adapter", "selected no judge-template adapter"),
    ],
)
def test_bound_run_rejects_unproven_or_stale_judge_identity(
    tmp_path, mutation, message
):
    run_dir, _, _, _ = _write_bound_run(tmp_path)
    metadata_path = run_dir / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapters = metadata["identity"]["adapters"]
    if mutation == "missing-template":
        adapters[0].pop("judge_template")
    elif mutation == "stale-template":
        adapters[0]["judge_template"]["sha256"] = "0" * 64
    elif mutation == "stale-protocol":
        adapters[0]["judge_protocol"]["version"] += 1
    elif mutation == "duplicate-adapter":
        adapters.append(dict(adapters[0]))
    else:
        metadata["identity"]["adapters"] = []
    metadata["fingerprint"] = _run_fingerprint(metadata["identity"])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _calibration_inputs_from_args(
            _bound_cli_args(run_dir, tmp_path / "calibration.json")
        )


def test_bound_run_rejects_judge_or_config_overrides(tmp_path):
    run_dir, _, _, _ = _write_bound_run(tmp_path)
    with pytest.raises(ValueError, match="cannot be combined"):
        _calibration_inputs_from_args(
            _bound_cli_args(
                run_dir,
                tmp_path / "calibration.json",
                judge_model="different-judge",
            )
        )


@pytest.mark.parametrize(
    ("self_judged", "expected_code", "expected_headline"),
    [(False, 0, True), (True, 1, False)],
)
async def test_bound_cli_exits_on_headline_gate_and_records_verified_run(
    tmp_path,
    capsys,
    self_judged,
    expected_code,
    expected_headline,
):
    run_dir, _, _, fingerprint = _write_bound_run(
        tmp_path, self_judged=self_judged
    )
    output = tmp_path / f"calibration-{self_judged}.json"
    code = await run_calibration_cli(
        _bound_cli_args(run_dir, output),
        http_factory=_published_gold_factory(),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == expected_code
    assert report["passed"] is True
    assert report["headline_eligible"] is expected_headline
    assert report["run_binding"]["run_fingerprint"] == fingerprint
    assert report["run_binding"]["bound"] is True
    assert json.loads(capsys.readouterr().out)["fingerprint"] == report["fingerprint"]
