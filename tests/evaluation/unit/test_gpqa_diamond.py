import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import kairyu.evaluation.adapters.gpqa_diamond as gpqa_module
from kairyu.evaluation.adapters.base import ItemResult, RunSelection
from kairyu.evaluation.adapters.gpqa_diamond import GPQADiamondAdapter
from kairyu.evaluation.connectors import (
    ConnectorResponse,
    ConnectorResult,
    ConnectorUsage,
    FakeOpenAIConnector,
)
from kairyu.evaluation.guards import RunGuardError
from kairyu.evaluation.profiles import get_profile_lock, load_profile_resource
from kairyu.evaluation.protocol import protocol_hash
from kairyu.evaluation.safety import SecretSafetyError
from kairyu.evaluation.schemas import RunMode
from kairyu.evaluation.service import (
    BenchmarkService,
    ConnectorConfig,
    EvaluationRuntime,
    bind_connector_to_plan,
    rebuild_plan_from_job,
)

ROOT = Path(__file__).parents[3]
SMOKE_FIXTURE = (
    ROOT / "kairyu" / "evaluation" / "resources" / "fixtures" / "gpqa-diamond-smoke.jsonl"
)
FIXTURE_SHA256 = "d00d5ff92cf97f99b66af968abb7b247494d2f8f79a434a91e1ce45172683eed"
HARNESS_COMMIT = "fce1d21391dc2d7b45c9cf0edb9b9e40d526aed3"
OFFICIAL_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
EVALSCOPE_WHEEL_SHA256 = "0bbbd2302d038b2c3a95cad2c4b20b93cc4ca911dc7b0033060c551a22cbab8f"
COMPATIBILITY_LAYER_SHA256 = "533999762a93d5a75694ea785212e131dadbf725b1862d9c5fc2f453c4774556"


def _selection(**changes):
    values = {
        "profile": "smoke",
        "mode": RunMode.SMOKE,
        "target_model": "offline-model",
    }
    values.update(changes)
    return RunSelection(**values)


def _derived_198_item_snapshot(tmp_path):
    source_rows = [
        json.loads(line)
        for line in SMOKE_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for ordinal in range(198):
        row = dict(source_rows[ordinal % len(source_rows)])
        row["Record ID"] = f"{row['Record ID']}-derived-{ordinal:03d}"
        rows.append(row)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    path = tmp_path / "approved-synthetic-gpqa.jsonl"
    path.write_text(payload, encoding="utf-8")
    return path, hashlib.sha256(payload.encode()).hexdigest(), tuple(rows)


def _hostile_198_item_snapshot(tmp_path):
    path, _, rows = _derived_198_item_snapshot(tmp_path)
    hostile_record_id = "private-record-</script>-DROP TABLE runs"
    hostile_question = (
        "PRIVATE QUESTION TEXT" + chr(10) + "ANSWER: A" + chr(10) + "<script>alert(1)</script>"
    )
    hostile_rows = [dict(row) for row in rows]
    hostile_rows[0]["Record ID"] = hostile_record_id
    hostile_rows[0]["Question"] = hostile_question
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + chr(10) for row in hostile_rows
    )
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return path, digest, hostile_record_id, hostile_question


def _success(request_id, content):
    return ConnectorResult(
        response=ConnectorResponse(
            request_id=request_id,
            content=content,
            finish_reason="stop",
            provider_request_id=f"provider-{request_id}",
            usage=ConnectorUsage(
                prompt_tokens=20,
                completion_tokens=5,
                total_tokens=25,
            ),
            latency_seconds=0.1,
        )
    )


def test_profile_locks_pin_dataset_harness_dependencies_and_protocol():
    resource = load_profile_resource("gpqa-diamond")

    compatibility_path = ROOT / "kairyu" / "evaluation" / "adapters" / "gpqa_v181.py"
    assert hashlib.sha256(compatibility_path.read_bytes()).hexdigest() == (
        COMPATIBILITY_LAYER_SHA256
    )
    assert resource.schema_version == 1
    assert tuple(profile.name for profile in resource.profiles) == (
        "smoke",
        "fugu-2026",
        "official-latest",
    )
    assert all(profile.expected_full_count == 198 for profile in resource.profiles)
    assert all(profile.harness_commit == HARNESS_COMMIT for profile in resource.profiles)
    assert all(
        profile.harness_repository == "https://github.com/modelscope/evalscope"
        for profile in resource.profiles
    )
    assert all(
        profile.evalscope_wheel_sha256 == EVALSCOPE_WHEEL_SHA256 for profile in resource.profiles
    )
    assert all(
        profile.compatibility_layer_sha256 == COMPATIBILITY_LAYER_SHA256
        for profile in resource.profiles
    )
    lock_sha256 = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    assert all(profile.dependency_lock_sha256 == lock_sha256 for profile in resource.profiles)

    smoke = get_profile_lock("gpqa-diamond", "smoke")
    fugu = get_profile_lock("gpqa-diamond", "fugu-2026")
    official = get_profile_lock("gpqa-diamond", "official-latest")
    assert smoke.dataset_id == "kairyu.synthetic.gpqa-diamond"
    assert smoke.dataset_revision == smoke.dataset_sha256 == FIXTURE_SHA256
    assert smoke.gated is False
    assert fugu.dataset_revision == "unresolved-fugu-2026-dataset-snapshot"
    assert fugu.unresolved_fields == ("dataset_revision",)
    assert official.dataset_revision == OFFICIAL_REVISION
    assert official.dataset_sha256 is None
    assert official.unresolved_fields == ("dataset_revision",)
    assert official.gated is True

    for profile in resource.profiles:
        protocol = profile.to_profile("gpqa-diamond").protocol
        assert protocol.harness_version == "1.8.1+compat.1"
        assert protocol.harness_commit == HARNESS_COMMIT
        assert (
            protocol.adapter_configuration["upstream_repository"]
            == "https://github.com/modelscope/evalscope"
        )
        assert protocol.web_access is False
        assert protocol.retries == 0
        assert protocol.generation_parameters == {
            "temperature": 0.0,
            "repeats": 1,
            "seed": 42,
        }
        assert protocol.adapter_configuration["compatibility_layer_sha256"] == (
            COMPATIBILITY_LAYER_SHA256
        )
        assert protocol.dependency_compatibility_patches == (
            f"kairyu.evaluation.adapters.gpqa_v181@sha256:{COMPATIBILITY_LAYER_SHA256}",
        )
        assert all(url.startswith("https://") for url in profile.source_urls)


def test_run_selection_generation_parameters_are_deeply_immutable():
    selection = _selection(
        generation_parameters={
            "temperature": 0.0,
            "provider_options": {"stop": ["synthetic-stop"]},
        }
    )

    with pytest.raises(TypeError, match="immutable"):
        selection.generation_parameters["temperature"] = 1.0
    with pytest.raises(TypeError, match="immutable"):
        selection.generation_parameters["provider_options"]["stop"].append("changed")


def test_smoke_doctor_prepare_and_plan_are_offline_and_match_golden():
    adapter = GPQADiamondAdapter()

    doctor = adapter.doctor("smoke")
    dry_run = adapter.prepare("smoke", dry_run=True)
    prepared = adapter.prepare("smoke", dry_run=False)
    plan = adapter.build_run_plan(_selection(), environ={})

    assert doctor.runnable
    assert {check.check_id for check in doctor.checks} == {
        "compatibility-layer",
        "cpu",
        "disk",
        "docker",
        "harness-pin",
        "memory",
        "model-capability",
        "python",
        "synthetic-fixture",
    }
    checks = {check.check_id: check for check in doctor.checks}
    assert checks["docker"].status.value == "pass"
    assert checks["model-capability"].status.value == "warn"
    assert "not measurable" in checks["model-capability"].summary
    assert dry_run.dry_run and dry_run.item_count == 2
    assert prepared.item_count == 2
    assert prepared.dataset_sha256 == FIXTURE_SHA256
    assert plan.official_eligible is False
    assert plan.expected_full_count == 198
    assert plan.estimated_model_calls == 2
    assert plan.estimate.selected_item_count == 2
    assert plan.estimate.model_calls == 2
    assert plan.estimate.maximum_model_calls == 2
    assert plan.estimate.estimated_input_tokens > 0
    assert plan.estimate.maximum_output_tokens == 2_048
    assert plan.estimate.estimated_duration_seconds is None
    assert plan.estimate.maximum_duration_seconds == 240.0
    assert plan.estimate.estimated_cost_usd is None
    assert plan.resources.cpu_cores == 1
    assert plan.resources.ram_bytes == 512 * 1024 * 1024
    assert plan.resources.disk_bytes >= 64 * 1024 * 1024
    assert plan.resources.docker_required is False
    assert plan.execution.command == ("kairyu", "benchmark", "worker", "--once")
    assert plan.item_input_manifest_sha256 == (
        "396b978e4af4fb42eed17d5e5213f2ba00b768edfdc8c659a5cb7bc480a2ef9a"
    )
    assert [item.item_id for item in plan.items] == [
        "gpqa-7da3fb03c0e6d07bef69a6cb13a7c81f",
        "gpqa-ddb2ff5cd527f270ba7bebd3f2f9ecdb",
    ]
    assert plan.protocol.adapter_configuration["all_item_permutations_sha256"] == (
        "bd77002cb98f929b69dcc3487e6f0acd9e294adada2ae1b8b272625281e56b22"
    )
    assert plan.protocol_hash == (
        "d6dfd21a3000a3cf39131af24fb634f15672af4786b5399f18d803b363203c76"
    )
    assert plan.protocol.generation_parameters == {
        "max_tokens": 1_024,
        "repeats": 1,
        "seed": 42,
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "top_p": 1.0,
    }
    assert plan.selection.generation_parameters == plan.protocol.generation_parameters
    assert plan.protocol.adapter_configuration["model_request_policy"] == (
        "one-chat-completion-per-item-v1"
    )
    assert plan.protocol.adapter_configuration["request_concurrency"] == 1
    assert plan.protocol.adapter_configuration["target_model"] == plan.target_model
    evidence = plan.protocol.adapter_configuration["reproducibility_evidence"]
    assert set(evidence) == {
        "hardware_conditions",
        "provider_api_version",
        "runtime_dependency_environment",
        "source_retrieval_date",
    }
    assert {item["status"] for item in evidence.values()} == {"unresolved"}
    assert all(item["reason"] for item in evidence.values())
    assert plan.protocol.adapter_configuration["reproducibility_evidence_complete"] is False
    assert [
        (item.ordinal, item.choice_permutation, item.target, item.input_sha256)
        for item in plan.items
    ] == [
        (
            0,
            (2, 1, 3, 0),
            "C",
            "fdf314bdf3e9bb304402c8a71bcaef3ba8c6613b32a99fff685956fb16da1a0e",
        ),
        (
            1,
            (3, 2, 0, 1),
            "A",
            "9458782398a62aaa34208347311efb0577d93f31949997b4ccc1f423f81e9100",
        ),
    ]
    assert plan.protocol.dataset_revision == f"{FIXTURE_SHA256}:sha256:{FIXTURE_SHA256}"
    assert plan.protocol.sample_filter == {
        "limit": None,
        "mode": "smoke",
        "sample_ids": [],
    }
    assert plan.protocol.adapter_configuration["evalscope_wheel_sha256"] == (EVALSCOPE_WHEEL_SHA256)
    assert plan.protocol.adapter_configuration["compatibility_layer_sha256"] == (
        COMPATIBILITY_LAYER_SHA256
    )


def test_target_model_is_part_of_the_protocol_hash():
    first = GPQADiamondAdapter().build_run_plan(
        _selection(target_model="synthetic-model-a"),
        environ={},
    )
    second = GPQADiamondAdapter().build_run_plan(
        _selection(target_model="synthetic-model-b"),
        environ={},
    )

    assert first.protocol.adapter_configuration["target_model"] == "synthetic-model-a"
    assert second.protocol.adapter_configuration["target_model"] == "synthetic-model-b"
    assert first.protocol_hash != second.protocol_hash
    assert first.item_input_manifest_sha256 == second.item_input_manifest_sha256


def test_connector_identity_and_retries_are_protocol_bound_and_fail_closed():
    plan = GPQADiamondAdapter().build_run_plan(_selection(), environ={})
    reviewed = ConnectorConfig(
        kind="openai",
        endpoint="http://127.0.0.1:18080",
        max_retries=0,
    )
    protocol_payload = plan.protocol.model_dump(mode="json")
    adapter_configuration = {
        **protocol_payload["adapter_configuration"],
        "reviewed_model_connector": reviewed.model_dump(mode="json"),
    }
    reviewed_protocol = plan.protocol.model_copy(
        update={"adapter_configuration": adapter_configuration}
    )
    eligible_plan = replace(
        plan,
        protocol=reviewed_protocol,
        protocol_hash=protocol_hash(reviewed_protocol),
        official_eligible=True,
    )

    exact = bind_connector_to_plan(eligible_plan, reviewed)
    different_endpoint = bind_connector_to_plan(
        eligible_plan,
        ConnectorConfig(
            kind="openai",
            endpoint="http://127.0.0.1:18081",
            max_retries=0,
        ),
    )
    different_retries = bind_connector_to_plan(
        eligible_plan,
        ConnectorConfig(
            kind="openai",
            endpoint="http://127.0.0.1:18080",
            max_retries=1,
        ),
    )

    fake = bind_connector_to_plan(
        plan,
        ConnectorConfig(kind="fake", max_retries=2),
    )

    assert exact.protocol.retries == 0
    assert exact.estimate.maximum_model_calls == 2
    assert exact.estimate.maximum_duration_seconds == 240.0
    assert exact.official_eligible is True
    assert exact.protocol.adapter_configuration["model_connector_review"]["status"] == ("verified")
    planning_evidence = exact.protocol.adapter_configuration["planning_evidence"]
    assert planning_evidence == {
        "estimate": exact.estimate.model_dump(mode="json"),
        "resources": exact.resources.model_dump(mode="json"),
        "execution": exact.execution.model_dump(mode="json"),
    }
    assert different_endpoint.official_eligible is False
    assert different_retries.protocol.retries == 1
    assert different_retries.estimate.maximum_model_calls == 4
    assert different_retries.estimate.maximum_duration_seconds == 481.0
    assert different_retries.official_eligible is False
    assert fake.protocol.retries == 0
    assert fake.estimate.maximum_model_calls == 2
    assert fake.estimate.maximum_duration_seconds == 240.0


def test_doctor_reports_unmeasurable_resources_and_model_capability_as_warnings(
    monkeypatch,
):
    monkeypatch.setattr(gpqa_module.os, "cpu_count", lambda: None)
    monkeypatch.setattr(gpqa_module, "_total_memory_bytes", lambda: None)
    monkeypatch.setattr(gpqa_module, "_disk_free_bytes", lambda _path: None)

    report = GPQADiamondAdapter().doctor("smoke")
    checks = {check.check_id: check for check in report.checks}

    assert report.runnable
    for check_id in ("cpu", "memory", "disk", "model-capability"):
        assert checks[check_id].status.value == "warn"
        assert checks[check_id].action


def test_doctor_fails_when_the_prompt_pin_does_not_match(monkeypatch):
    monkeypatch.setattr(gpqa_module, "PROMPT_SHA256", "0" * 64)

    report = GPQADiamondAdapter().doctor("smoke")
    checks = {check.check_id: check for check in report.checks}

    assert report.runnable is False
    assert checks["harness-pin"].status.value == "fail"


def test_compatibility_layer_mismatch_fails_doctor_and_precedes_dataset_read(
    monkeypatch,
):
    monkeypatch.setattr(gpqa_module, "_compatibility_layer_sha256", lambda: "0" * 64)

    report = GPQADiamondAdapter().doctor("smoke")
    checks = {check.check_id: check for check in report.checks}
    assert report.runnable is False
    assert checks["compatibility-layer"].status.value == "fail"

    monkeypatch.setattr(
        gpqa_module,
        "load_records_with_sha256",
        lambda _path: pytest.fail("compatibility check must precede dataset reads"),
    )
    with pytest.raises(ValueError, match="compatibility module checksum"):
        GPQADiamondAdapter().build_run_plan(_selection(), environ={})


def test_run_revalidates_compatibility_layer_before_connector_use(monkeypatch):
    adapter = GPQADiamondAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    monkeypatch.setattr(gpqa_module, "_compatibility_layer_sha256", lambda: "0" * 64)

    connector = FakeOpenAIConnector({})
    with pytest.raises(ValueError, match="compatibility module checksum"):
        adapter.run(plan, plan.items[0], connector, cancel_check=lambda: False)


def test_scope_guards_run_before_profile_or_dataset_access(monkeypatch):
    def unexpected_profile_access(*_args, **_kwargs):
        pytest.fail("full-run guard must execute before profile access")

    monkeypatch.setattr(gpqa_module, "get_profile_lock", unexpected_profile_access)
    adapter = GPQADiamondAdapter()

    with pytest.raises(RunGuardError) as raised:
        adapter.build_run_plan(
            _selection(profile="official-latest", mode=RunMode.FULL),
            environ={},
        )

    assert raised.value.code == "full_confirmation_required"


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            _selection(profile="official-latest"),
            "smoke mode requires the smoke profile",
        ),
        (
            _selection(mode=RunMode.SAMPLE, limit=1),
            "synthetic smoke profile cannot run real-data modes",
        ),
    ],
)
def test_smoke_and_real_data_profiles_cannot_be_crossed(selection, message):
    with pytest.raises(ValueError, match=message):
        GPQADiamondAdapter().build_run_plan(selection, environ={})


def test_gated_profile_requires_access_path_digest_and_exactly_198_items(tmp_path):
    adapter = GPQADiamondAdapter()
    missing = adapter.prepare("official-latest", dry_run=False)

    assert missing.status.value == "needs_user_action"
    assert "manually accept" in " ".join(missing.actions)

    fixture_digest = hashlib.sha256(SMOKE_FIXTURE.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exactly 198"):
        adapter.prepare(
            "official-latest",
            dry_run=False,
            dataset_path=SMOKE_FIXTURE,
            dataset_sha256=fixture_digest,
            accepted_access=True,
        )

    path, digest, _ = _derived_198_item_snapshot(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        adapter.prepare(
            "official-latest",
            dry_run=False,
            dataset_path=path,
            dataset_sha256="0" * 64,
            accepted_access=True,
        )

    prepared = adapter.prepare(
        "official-latest",
        dry_run=False,
        dataset_path=path,
        dataset_sha256=digest,
        accepted_access=True,
    )
    assert prepared.status.value == "ready"
    assert prepared.item_count == 198
    assert prepared.dataset_sha256 == digest
    assert prepared.dataset_revision == f"{OFFICIAL_REVISION}:sha256:{digest}"


def test_sample_and_full_plans_revalidate_gated_snapshot_and_official_eligibility(tmp_path):
    path, digest, _ = _derived_198_item_snapshot(tmp_path)
    adapter = GPQADiamondAdapter()

    sample = adapter.build_run_plan(
        _selection(
            profile="official-latest",
            mode=RunMode.SAMPLE,
            limit=2,
            dataset_path=str(path),
            dataset_sha256=digest,
            accepted_access=True,
        ),
        environ={},
    )
    full = adapter.build_run_plan(
        _selection(
            profile="official-latest",
            mode=RunMode.FULL,
            confirm_full_run=True,
            dataset_path=str(path),
            dataset_sha256=digest,
            accepted_access=True,
        ),
        environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
    )
    unresolved_fugu = adapter.build_run_plan(
        _selection(
            profile="fugu-2026",
            mode=RunMode.FULL,
            confirm_full_run=True,
            dataset_path=str(path),
            dataset_sha256=digest,
            accepted_access=True,
        ),
        environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
    )

    assert len(sample.items) == 2
    assert sample.estimated_model_calls == 2
    assert sample.official_eligible is False
    assert len(full.items) == 198
    assert full.estimated_model_calls == 198
    assert full.official_eligible is False
    assert full.protocol.adapter_configuration["dataset_snapshot_reviewed"] is False
    assert full.protocol.dataset_revision == f"{OFFICIAL_REVISION}:sha256:{digest}"
    assert full.protocol.sample_filter == {
        "limit": None,
        "mode": "full",
        "sample_ids": [],
    }
    assert str(path) not in full.protocol.model_dump_json()
    assert unresolved_fugu.protocol.unresolved_fields == ("dataset_revision",)
    assert unresolved_fugu.official_eligible is False


def test_protocol_and_report_exclude_raw_fixture_content_and_credentials():
    secret = "sk-synthetic-secret-value-123456"
    with pytest.raises(SecretSafetyError) as raised:
        _selection(generation_parameters={"api_key": secret})
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)

    adapter = GPQADiamondAdapter()
    plan = adapter.build_run_plan(_selection(), environ={})
    response_by_request = {}
    for item, content in zip(
        plan.items,
        (
            "private synthetic scratch one\nANSWER: C",
            "private synthetic scratch two\nANSWER: B",
        ),
        strict=True,
    ):
        request_id = f"gpqa-{item.ordinal}-{item.input_sha256[:16]}"
        response_by_request[request_id] = _success(request_id, content)
    connector = FakeOpenAIConnector(response_by_request)

    results = tuple(
        adapter.run(plan, item, connector, cancel_check=lambda: False) for item in plan.items
    )
    collected = adapter.collect("gpqa-smoke-run", plan, results)
    report_data = adapter.render_report_data(collected)

    assert [result.correct for result in results] == [True, False]
    assert collected.metrics[0].value == 50.0
    assert collected.metrics[0].denominator == 2
    assert report_data == {
        "benchmark_metric": "Accuracy",
        "error_counts": {},
        "metric_value": 50.0,
        "metric_scale": 100.0,
    }
    persisted_text = plan.protocol.model_dump_json() + json.dumps(report_data, sort_keys=True)
    for raw_fragment in (
        "fictional moon",
        "One orbit",
        "made-up laboratory",
        "private synthetic scratch",
        secret,
    ):
        assert raw_fragment not in persisted_text


def test_official_eligibility_requires_a_reviewed_snapshot_digest(
    tmp_path,
    monkeypatch,
):
    path, digest, _ = _derived_198_item_snapshot(tmp_path)
    unreviewed_lock = get_profile_lock(
        "gpqa-diamond",
        "official-latest",
    ).model_copy(update={"unresolved_fields": ()})
    monkeypatch.setattr(
        gpqa_module,
        "get_profile_lock",
        lambda _benchmark_id, _profile: unreviewed_lock,
    )

    plan = GPQADiamondAdapter().build_run_plan(
        _selection(
            profile="official-latest",
            mode=RunMode.FULL,
            confirm_full_run=True,
            dataset_path=str(path),
            dataset_sha256=digest,
            accepted_access=True,
        ),
        environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
    )

    assert plan.protocol.unresolved_fields == ()
    assert plan.protocol.adapter_configuration["dataset_snapshot_reviewed"] is False
    assert plan.official_eligible is False


def test_hostile_source_fields_never_become_protocol_or_report_facing_ids(tmp_path):
    path, digest, hostile_record_id, hostile_question = _hostile_198_item_snapshot(tmp_path)

    plan = GPQADiamondAdapter().build_run_plan(
        _selection(
            profile="official-latest",
            mode=RunMode.SAMPLE,
            sample_ids=(hostile_record_id,),
            dataset_path=str(path),
            dataset_sha256=digest,
            accepted_access=True,
        ),
        environ={},
    )
    expected_public_id = f"gpqa-{hashlib.sha256(hostile_record_id.encode()).hexdigest()[:32]}"
    item = plan.items[0]
    result = ItemResult(
        item_id=item.item_id,
        ordinal=item.ordinal,
        input_sha256=item.input_sha256,
        response_text="ANSWER: A",
        extracted_answer="A",
        target=item.target,
        correct=item.target == "A",
        provider_model=plan.target_model,
    )
    report_data = GPQADiamondAdapter().render_report_data(
        GPQADiamondAdapter().collect("hostile-run", plan, (result,))
    )
    report_facing = json.dumps(
        {
            "item_id": item.item_id,
            "item_input_manifest_sha256": plan.item_input_manifest_sha256,
            "protocol": plan.protocol.model_dump(mode="json"),
            "report": report_data,
            "selection": plan.selection.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    assert hostile_question in item.prompt
    assert item.item_id == expected_public_id
    assert plan.selection.sample_ids == (expected_public_id,)
    assert plan.protocol.model_dump(mode="json")["sample_filter"]["sample_ids"] == [
        expected_public_id
    ]
    assert hostile_record_id not in report_facing
    assert hostile_question not in report_facing


def test_service_persists_only_public_sample_ids_and_rebuilds_them(tmp_path):
    path, digest, hostile_record_id, hostile_question = _hostile_198_item_snapshot(tmp_path)
    expected_public_id = f"gpqa-{hashlib.sha256(hostile_record_id.encode()).hexdigest()[:32]}"
    runtime = EvaluationRuntime(tmp_path / "state")
    submitted = BenchmarkService(runtime).submit(
        "gpqa-diamond",
        _selection(
            profile="official-latest",
            mode=RunMode.SAMPLE,
            sample_ids=(hostile_record_id,),
            dataset_path=str(path),
            dataset_sha256=digest,
            accepted_access=True,
        ),
        ConnectorConfig(kind="openai", endpoint="http://127.0.0.1:18080"),
        run_id="run-hostile-public-id",
    )
    payload = runtime.store.get_job(submitted.job_id).payload
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert payload["selection"]["sample_ids"] == [expected_public_id]
    assert payload["protocol"]["retries"] == 2
    assert submitted.estimate.model_calls == 1
    assert submitted.estimate.maximum_model_calls == 3
    assert submitted.estimate.maximum_duration_seconds == 361.5
    assert payload["protocol"]["adapter_configuration"]["planning_evidence"] == {
        "estimate": submitted.estimate.model_dump(mode="json"),
        "resources": submitted.resources.model_dump(mode="json"),
        "execution": submitted.execution.model_dump(mode="json"),
    }
    assert hostile_record_id not in serialized
    assert hostile_question not in serialized

    rebuilt = rebuild_plan_from_job(payload)
    assert rebuilt.selection.sample_ids == (expected_public_id,)
    assert [item.item_id for item in rebuilt.items] == [expected_public_id]

    tampered = json.loads(serialized)
    tampered["selection"]["sample_ids"] = [hostile_record_id]
    with pytest.raises(ValueError, match="selection snapshot is not canonical"):
        rebuild_plan_from_job(tampered)


def test_sample_ids_are_canonicalized_to_dataset_execution_order(tmp_path):
    path, digest, rows = _derived_198_item_snapshot(tmp_path)
    expected_public_ids = tuple(
        f"gpqa-{hashlib.sha256(rows[ordinal]['Record ID'].encode()).hexdigest()[:32]}"
        for ordinal in (2, 7)
    )
    common = {
        "profile": "official-latest",
        "mode": RunMode.SAMPLE,
        "dataset_path": str(path),
        "dataset_sha256": digest,
        "accepted_access": True,
    }
    source_id_plan = GPQADiamondAdapter().build_run_plan(
        _selection(
            **common,
            sample_ids=(rows[7]["Record ID"], rows[2]["Record ID"]),
        ),
        environ={},
    )
    public_id_plan = GPQADiamondAdapter().build_run_plan(
        _selection(
            **common,
            sample_ids=tuple(reversed(expected_public_ids)),
        ),
        environ={},
    )

    assert source_id_plan.selection.sample_ids == expected_public_ids
    assert public_id_plan.selection.sample_ids == expected_public_ids
    assert tuple(item.item_id for item in source_id_plan.items) == expected_public_ids
    assert source_id_plan.protocol == public_id_plan.protocol
    assert source_id_plan.protocol_hash == public_id_plan.protocol_hash
    assert source_id_plan.item_input_manifest_sha256 == public_id_plan.item_input_manifest_sha256


def test_gpqa_rejects_unsupported_repeats_before_profile_or_snapshot_access(
    monkeypatch,
):
    def unexpected_access(*_args, **_kwargs):
        pytest.fail("generation validation must precede profile and snapshot access")

    monkeypatch.setattr(gpqa_module, "get_profile_lock", unexpected_access)
    monkeypatch.setattr(gpqa_module, "load_records_with_sha256", unexpected_access)

    with pytest.raises(ValueError, match="repeats must be exactly 1"):
        GPQADiamondAdapter().build_run_plan(
            _selection(generation_parameters={"temperature": 0.0, "repeats": 2}),
            environ={},
        )


@pytest.mark.parametrize(
    "generation_parameters",
    (
        {"temperature": -0.001},
        {"temperature": 2.001},
        {"temperature": "not-a-number"},
        {"top_p": 0.0},
        {"top_p": 1.001},
        {"max_tokens": 0},
        {"max_tokens": 10_000_001},
        {"max_tokens": 1.5},
        {"timeout_seconds": 0.0},
        {"timeout_seconds": 86_401.0},
    ),
)
def test_gpqa_rejects_generation_values_outside_model_request_contract(
    generation_parameters,
):
    with pytest.raises(ValueError, match="ModelRequest constraints"):
        GPQADiamondAdapter().build_run_plan(
            _selection(generation_parameters=generation_parameters),
            environ={},
        )


def test_gpqa_canonicalizes_effective_generation_values_and_preserves_unknown_keys():
    plan = GPQADiamondAdapter().build_run_plan(
        _selection(
            generation_parameters={
                "temperature": 0,
                "top_p": 0.75,
                "max_tokens": 256,
                "timeout_seconds": 30,
                "repeats": 1,
                "provider_option": {"synthetic": True},
            }
        ),
        environ={},
    )
    expected = {
        "max_tokens": 256,
        "provider_option": {"synthetic": True},
        "repeats": 1,
        "seed": 42,
        "temperature": 0.0,
        "timeout_seconds": 30.0,
        "top_p": 0.75,
    }

    assert plan.protocol.generation_parameters == expected
    assert plan.selection.generation_parameters == expected


def test_gpqa_rejects_generation_seed_that_conflicts_with_selection():
    with pytest.raises(ValueError, match="seed must match the run selection seed"):
        GPQADiamondAdapter().build_run_plan(
            _selection(
                seed=42,
                generation_parameters={
                    "temperature": 0.0,
                    "repeats": 1,
                    "seed": 7,
                },
            ),
            environ={},
        )


def test_official_eligibility_remains_fail_closed_with_unattested_runtime(
    tmp_path,
    monkeypatch,
):
    path, digest, _ = _derived_198_item_snapshot(tmp_path)
    reviewed_lock = get_profile_lock(
        "gpqa-diamond",
        "official-latest",
    ).model_copy(
        update={
            "dataset_sha256": digest,
            "unresolved_fields": (),
        }
    )
    monkeypatch.setattr(
        gpqa_module,
        "get_profile_lock",
        lambda _benchmark_id, _profile: reviewed_lock,
    )
    common = {
        "profile": "official-latest",
        "mode": RunMode.FULL,
        "confirm_full_run": True,
        "dataset_path": str(path),
        "dataset_sha256": digest,
        "accepted_access": True,
    }
    adapter = GPQADiamondAdapter()

    default_plan = adapter.build_run_plan(
        _selection(**common),
        environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
    )
    changed_known = adapter.build_run_plan(
        _selection(
            **common,
            generation_parameters={
                "temperature": 0.0,
                "top_p": 0.9,
                "repeats": 1,
            },
        ),
        environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
    )
    unknown_key = adapter.build_run_plan(
        _selection(
            **common,
            generation_parameters={
                "temperature": 0.0,
                "repeats": 1,
                "provider_option": {"synthetic": True},
            },
        ),
        environ={"BENCHMARK_ALLOW_FULL_RUN": "1"},
    )

    assert default_plan.official_eligible is False
    assert default_plan.protocol.adapter_configuration["reproducibility_evidence_complete"] is False
    assert changed_known.official_eligible is False
    assert unknown_key.official_eligible is False
    assert default_plan.protocol.adapter_configuration["model_request_policy"] == (
        "one-chat-completion-per-item-v1"
    )
    assert default_plan.protocol.adapter_configuration["request_concurrency"] == 1
