"""Focused tests for the replayable F2c real-engine evidence gate."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from bench import kv_aware_ttft_f2c_bench as f2c


def test_trace_is_deterministic_and_pairs_opposite_session_hrw_targets() -> None:
    config = f2c.profile_config("smoke")
    first = f2c.build_trace(config)
    second = f2c.build_trace(config)

    assert first == second
    assert len(first) == config.rounds * config.families_per_round
    assert f2c.trace_descriptor(config, first) == f2c.trace_descriptor(
        config,
        second,
    )
    assert len({family.prefix_fingerprint for family in first}) == len(first)
    for family in first:
        assert f2c.frozen_hrw(family.seed_session_id) == family.seed_replica_id
        assert (
            f2c.frozen_hrw(family.measured_session_id)
            == family.measured_replica_id
        )
        assert family.seed_replica_id != family.measured_replica_id
        assert family.prefix_fingerprint
        assert family.shared_prefix not in f2c.canonical_json(
            f2c.trace_descriptor(config, first)
        )


def test_nearest_rank_p95_and_geometric_mean_are_exactly_frozen() -> None:
    assert f2c.nearest_rank_percentile(list(range(1, 21)), 0.95) == 19
    assert f2c.nearest_rank_percentile(list(range(1, 65)), 0.95) == 61
    assert f2c.nearest_rank_percentile([5, 1, 4, 2, 3], 1.0) == 5
    assert f2c.geometric_mean([0.5, 0.5, 2.0, 2.0]) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        f2c.nearest_rank_percentile([], 0.95)
    with pytest.raises(ValueError):
        f2c.nearest_rank_percentile([1.0], 0.0)
    with pytest.raises(ValueError):
        f2c.geometric_mean([1.0, 0.0])


def test_eight_round_exact_order_statistics_have_one_sided_95pct_coverage() -> None:
    ratios = [0.51, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70, 1.20]
    upper_rank = 7
    lower_rank = len(ratios) - upper_rank + 1
    achieved_coverage = sum(
        math.comb(len(ratios), count)
        for count in range(upper_rank)
    ) / 2 ** len(ratios)

    assert achieved_coverage == 0.96484375
    assert sorted(ratios)[upper_rank - 1] == 0.70
    assert sorted(ratios)[lower_rank - 1] == 0.55
    assert sum(value <= 0.70 for value in ratios) == 7


def _passing_request_rows(
    config: f2c.WorkloadConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_index in range(config.rounds):
        for family_index in range(config.families_per_round):
            for policy in f2c.POLICIES:
                for turn in (1, 2):
                    receipt_ns = (
                        1_000_000_000
                        + round_index * 100_000_000_000
                        + family_index * 10_000_000_000
                        + turn * 1_000_000_000
                    )
                    candidate = policy == f2c.CANDIDATE_POLICY
                    rows.append(
                        {
                            "type": "request",
                            "binding": True,
                            "round_index": round_index,
                            "family_index": family_index,
                            "policy": policy,
                            "status": "success",
                            "receipt_ns": receipt_ns,
                            "done_ns": receipt_ns + 2_000_000_000,
                            "ttft_ns": (
                                600_000_000 if candidate else 1_000_000_000
                            ),
                            # Deliberately huge values prove these are
                            # diagnostics and never change the binding gates.
                            "paired_receipt_skew_ns": 9_000_000_000,
                            "schedule_lateness_ns": 8_000_000_000,
                            "usage": {
                                "prompt_tokens": 2_000,
                                "cached_tokens": 1_000 if candidate else 500,
                                "completion_tokens": config.max_tokens,
                            },
                        }
                    )
    return rows


def test_metric_recomputation_binds_stats_but_not_jitter_diagnostics() -> None:
    config = f2c.profile_config("formal", trace_namespace="metrics-test")
    metrics = f2c.derive_metrics(_passing_request_rows(config), config)
    checks = f2c.threshold_checks(metrics, config)

    assert metrics["computable"]
    assert metrics["request_count"] == (
        config.rounds * config.families_per_round * 2 * 2
    )
    assert metrics["ttft"]["upper_order_rank"] == 7
    assert metrics["ttft"]["upper_order_ratio"] == pytest.approx(0.60)
    assert metrics["ttft"]["geometric_mean_ratio"] == pytest.approx(0.60)
    assert metrics["ttft"]["pooled_ratio"] == pytest.approx(0.60)
    assert metrics["goodput"]["lower_order_rank"] == 2
    assert metrics["goodput"]["pooled_ratio"] == pytest.approx(1.0)
    assert metrics["cache"]["candidate_token_weighted_rate"] == 0.5
    assert metrics["cache"]["baseline_token_weighted_rate"] == 0.25
    assert metrics["diagnostics"]["max_paired_receipt_skew_ns"] == 9_000_000_000
    assert metrics["diagnostics"]["max_schedule_lateness_ns"] == 8_000_000_000
    assert all(checks.values())


def _passing_artifact_rows(
    config: f2c.WorkloadConfig,
) -> list[dict[str, object]]:
    source_value = {
        "commit": "1" * 40,
        "expected_commit": None,
        "expected_commit_matches": True,
        "git_clean": True,
        "git_status": [],
        "files": [
            {"path": path, "sha256": "a" * 64}
            for path in f2c._SOURCE_PATHS
        ],
    }
    configuration_value = {
        "workload": asdict(config),
        "cohorts": {
            "a": list(f2c.DEFAULT_COHORT_A_ENDPOINTS),
            "b": list(f2c.DEFAULT_COHORT_B_ENDPOINTS),
        },
        "model": "fixture-model",
        "model_revision": None,
        "model_digests": {},
        "runtime_image": None,
        "runtime_attestation": None,
        "files": [
            {
                "path": "fixture/f2c-compose.yaml",
                "sha256": "b" * 64,
            }
        ],
    }
    environment_value = {
        "platform": "fixture-platform",
        "python": "fixture-python",
        "python_executable": "/fixture/python",
        "cpu_count": 1,
        "sched_affinity": [0],
        "cuda_visible_devices": None,
        "github": {
            "GITHUB_RUN_ID": None,
            "GITHUB_RUN_ATTEMPT": None,
            "GITHUB_EVENT_NAME": None,
            "GITHUB_REPOSITORY": None,
            "GITHUB_SHA": None,
        },
        "nvidia_smi_gpus": [],
        "nvidia_smi_topology": [],
        "runtime_attestation": None,
    }
    provenance = {
        "source": f2c.hashed_descriptor(source_value),
        "configuration": f2c.hashed_descriptor(configuration_value),
        "environment": f2c.hashed_descriptor(environment_value),
    }
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": f2c.SCHEMA_VERSION,
            "gate": f2c.GATE,
            "config": asdict(config),
            "trace": f2c.trace_descriptor(config),
            "provenance_start": provenance,
            "model": "fixture-model",
            "model_revision": "",
            "model_digests": {},
            "runtime_image": None,
            "topology": {},
        }
    ]
    for family in f2c.build_trace(config):
        for turn in (0, 1, 2):
            output = f"{family.family_id}:turn-{turn}:identical-output"
            if turn == 0:
                prompt = family.seed_prompt
            elif turn == 1:
                prompt = family.turn1_prompt
            else:
                prompt = f2c.turn2_prompt(
                    family,
                    f"{family.family_id}:turn-1:identical-output",
                )
            for policy in f2c.POLICIES:
                candidate = policy == f2c.CANDIDATE_POLICY
                planned_ns = (
                    1_000_000_000
                    + family.round_index * 100_000_000_000
                    + family.family_index * 10_000_000_000
                    + turn * 1_000_000_000
                )
                schedule_lateness_ns = (
                    8_000_000_000 if candidate else 16_000_000_000
                )
                receipt_ns = planned_ns + schedule_lateness_ns
                ttft_ns = 600_000_000 if candidate else 1_000_000_000
                selected_replica_id = (
                    family.seed_replica_id
                    if turn == 0 or candidate
                    else family.measured_replica_id
                )
                replica_ordinal = f2c.REPLICA_IDS.index(selected_replica_id)
                reason = (
                    "prefix_match"
                    if turn > 0 and candidate
                    else "session_affinity"
                )
                session_sha256 = (
                    family.seed_session_sha256
                    if turn == 0
                    else family.measured_session_sha256
                )
                request_id = f2c.request_id_for(
                    family,
                    policy,
                    turn,
                )
                selected_at_ns = receipt_ns + 10_000
                rows.append(
                    {
                        "type": "request",
                        "request_id": request_id,
                        "round_index": family.round_index,
                        "family_index": family.family_index,
                        "family_id": family.family_id,
                        "policy": policy,
                        "cohort": f2c.cohort_for(
                            family.round_index,
                            policy,
                        ),
                        "turn": turn,
                        "phase": "seed" if turn == 0 else "measure",
                        "binding": turn > 0,
                        "session_sha256": session_sha256,
                        "prefix_fingerprint": family.prefix_fingerprint,
                        "prompt_sha256": f2c.sha256_text(prompt),
                        "prompt_chars": len(prompt),
                        "output_sha256": f2c.sha256_text(output),
                        "selected_replica_id": selected_replica_id,
                        "replica_ordinal": replica_ordinal,
                        "replica_generation": "c" * 32,
                        "reason": reason,
                        "planned_ns": planned_ns,
                        "receipt_ns": receipt_ns,
                        "selected_at_ns": selected_at_ns,
                        "first_content_ns": receipt_ns + ttft_ns,
                        "done_ns": receipt_ns + 2_000_000_000,
                        "ttft_ns": ttft_ns,
                        "schedule_lateness_ns": schedule_lateness_ns,
                        "paired_receipt_skew_ns": 8_000_000_000,
                        "terminal": True,
                        "status": "success",
                        "error_type": None,
                        "finish_reason": "length",
                        "usage": {
                            "prompt_tokens": 100,
                            "cached_tokens": 60 if candidate else 30,
                            "completion_tokens": config.max_tokens,
                        },
                        "router_decision": {
                            "kind": "replica",
                            "session_sha256": session_sha256,
                            "replica": replica_ordinal,
                            "reason": reason,
                            "replica_id": selected_replica_id,
                            "replica_generation": "c" * 32,
                            "placement_latency_ns": 10_000,
                            "request_id": request_id,
                            "placement_started_ns": receipt_ns,
                            "selected_at_ns": selected_at_ns,
                            "pool_size": 2,
                            "eligible_size": 2,
                        },
                    }
                )
    rows.append(
        {
            "type": "run_end",
            "request_count": len(rows) - 1,
            "provenance_end": provenance,
        }
    )
    return rows


def _write_passing_artifact(path: Path) -> dict[str, object]:
    return f2c.write_artifact(
        path,
        _passing_artifact_rows(f2c.profile_config("smoke")),
    )


def test_compact_artifact_replays_and_accepts_every_gate(tmp_path: Path) -> None:
    manifest = _write_passing_artifact(tmp_path)
    replayed = f2c.verify_artifact(tmp_path, assert_gate=True)

    assert manifest == replayed
    assert replayed["passed"]
    assert all(replayed["checks"].values())
    assert replayed["metrics"]["diagnostics"] == {
        "max_paired_receipt_skew_ns": 8_000_000_000,
        "max_schedule_lateness_ns": 16_000_000_000,
        "timing_fields_are_diagnostic_only": True,
    }


def test_raw_bytes_are_bound_by_manifest_sha256(tmp_path: Path) -> None:
    _write_passing_artifact(tmp_path)
    raw_path = tmp_path / f2c.RAW_NAME
    raw = raw_path.read_text(encoding="utf-8")
    raw_path.write_text(raw.replace("\n", " \n", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="raw SHA-256"):
        f2c.verify_artifact(tmp_path)


@pytest.mark.parametrize(
    "tamper",
    (
        "reason",
        "ttft",
        "schedule",
        "source",
        "nonterminal",
    ),
)
def test_semantic_tampering_regenerates_only_failing_evidence(
    tmp_path: Path,
    tamper: str,
) -> None:
    rows = _passing_artifact_rows(f2c.profile_config("smoke"))
    requests = [row for row in rows if row.get("type") == "request"]
    binding = next(row for row in requests if row["binding"])
    if tamper == "reason":
        binding["reason"] = "session_affinity"
    elif tamper == "ttft":
        binding["ttft_ns"] = int(binding["ttft_ns"]) + 1
    elif tamper == "schedule":
        binding["schedule_lateness_ns"] = (
            int(binding["schedule_lateness_ns"]) + 1
        )
    elif tamper == "source":
        changed_source = {
            **rows[-1]["provenance_end"]["source"]["value"],
            "commit": "2" * 40,
        }
        rows[-1]["provenance_end"] = {
            **rows[-1]["provenance_end"],
            "source": f2c.hashed_descriptor(changed_source),
        }
    elif tamper == "nonterminal":
        binding["terminal"] = False
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(tamper)

    manifest = f2c.write_artifact(tmp_path, rows)
    assert not manifest["passed"]
    with pytest.raises(RuntimeError, match="does not pass every gate"):
        f2c.verify_artifact(tmp_path, assert_gate=True)


def test_smoke_performance_is_diagnostic_but_same_formal_metrics_fail(
    tmp_path: Path,
) -> None:
    smoke = f2c.profile_config("smoke")
    rows = _passing_artifact_rows(smoke)
    slowed_rounds: set[int] = set()
    for row in rows:
        if (
            row.get("type") == "request"
            and row.get("binding") is True
            and row.get("policy") == f2c.CANDIDATE_POLICY
        ):
            row["usage"]["cached_tokens"] = 0
            round_index = int(row["round_index"])
            if round_index in slowed_rounds:
                continue
            slowed_rounds.add(round_index)
            receipt_ns = int(row["receipt_ns"])
            row["ttft_ns"] = 61_000_000_000
            row["first_content_ns"] = receipt_ns + int(row["ttft_ns"])
            row["done_ns"] = receipt_ns + 62_000_000_000

    manifest = f2c.write_artifact(tmp_path, rows)
    metrics = manifest["metrics"]
    assert manifest["passed"]
    assert f2c.threshold_checks(metrics, smoke) == {}
    assert float(metrics["ttft"]["pooled_ratio"]) > smoke.ttft_ratio_limit
    assert (
        float(metrics["goodput"]["pooled_ratio"])
        < smoke.goodput_ratio_floor
    )
    assert not metrics["cache"]["candidate_strictly_above_baseline"]

    formal = f2c.profile_config("formal", trace_namespace="same-metrics")
    formal_checks = f2c.threshold_checks(metrics, formal)
    assert not all(formal_checks.values())
    assert not formal_checks["ttft_pooled_at_most_0_70"]
    assert not formal_checks["goodput_pooled_at_least_0_99"]
    assert not formal_checks["engine_cache_rate_strictly_improved"]


def test_trace_or_request_set_tampering_is_rejected_before_manifest(
    tmp_path: Path,
) -> None:
    rows = _passing_artifact_rows(f2c.profile_config("smoke"))
    del rows[1]
    rows[-1]["request_count"] -= 1
    with pytest.raises(ValueError, match="request set mismatch"):
        f2c.write_artifact(tmp_path / "missing", rows)

    rows = _passing_artifact_rows(f2c.profile_config("smoke"))
    rows[1]["session_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="trace mismatch"):
        f2c.write_artifact(tmp_path / "trace", rows)


def test_manifest_config_tampering_is_rejected(tmp_path: Path) -> None:
    _write_passing_artifact(tmp_path)
    manifest_path = tmp_path / f2c.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["ttft_slo_s"] = 59.0
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest differs"):
        f2c.verify_artifact(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("request_id", "wrong-request"),
        ("session_sha256", "d" * 64),
        ("replica_id", "r9"),
        ("replica", 9),
        ("reason", "least_outstanding"),
        ("placement_started_ns", 1),
        ("selected_at_ns", 2),
        ("placement_latency_ns", 3),
        ("pool_size", 3),
        ("eligible_size", 1),
        ("replica_generation", ""),
        ("unexpected", True),
    ),
)
def test_embedded_router_decision_is_an_exact_causal_join(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    rows = _passing_artifact_rows(f2c.profile_config("smoke"))
    request = next(row for row in rows if row.get("binding") is True)
    request["router_decision"][field] = value

    manifest = f2c.write_artifact(tmp_path, rows)
    assert not manifest["checks"]["embedded_router_decisions_exact"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("error_type", "RuntimeError"), ("finish_reason", "stop")),
)
def test_success_rows_require_clear_error_and_length_finish(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    rows = _passing_artifact_rows(f2c.profile_config("smoke"))
    request = next(row for row in rows if row.get("binding") is True)
    request[field] = value

    manifest = f2c.write_artifact(tmp_path, rows)
    assert not manifest["passed"]


@pytest.mark.parametrize(
    "tamper",
    ("prompt_sha256", "prompt_chars", "prompt_tokens", "completion_tokens", "output"),
)
def test_paired_prompt_output_and_exact_work_are_bound(
    tmp_path: Path,
    tamper: str,
) -> None:
    rows = _passing_artifact_rows(f2c.profile_config("smoke"))
    baseline_turn2 = next(
        row
        for row in rows
        if row.get("policy") == f2c.BASELINE_POLICY and row.get("turn") == 2
    )
    if tamper == "prompt_sha256":
        baseline_turn2["prompt_sha256"] = "d" * 64
    elif tamper == "prompt_chars":
        baseline_turn2["prompt_chars"] += 1
    elif tamper == "prompt_tokens":
        baseline_turn2["usage"]["prompt_tokens"] += 1
    elif tamper == "completion_tokens":
        baseline_turn2["usage"]["completion_tokens"] -= 1
    else:
        baseline_turn2["output_sha256"] = "e" * 64

    manifest = f2c.write_artifact(tmp_path, rows)
    assert not manifest["checks"]["paired_prompts_and_exact_work_identical"] or (
        tamper == "output" and not manifest["checks"]["paired_outputs_identical"]
    )


def test_run_end_count_and_source_file_set_are_structural(tmp_path: Path) -> None:
    rows = _passing_artifact_rows(f2c.profile_config("smoke"))
    rows[-1]["request_count"] -= 1
    with pytest.raises(ValueError, match="request_count"):
        f2c.write_artifact(tmp_path / "count", rows)

    rows = _passing_artifact_rows(f2c.profile_config("smoke"))
    incomplete = {
        **rows[0]["provenance_start"]["source"]["value"],
        "files": [],
    }
    descriptor = f2c.hashed_descriptor(incomplete)
    rows[0]["provenance_start"]["source"] = descriptor
    rows[-1]["provenance_end"]["source"] = descriptor
    with pytest.raises(ValueError, match="frozen file set"):
        f2c.write_artifact(tmp_path / "source", rows)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", 1),
        ("request_timeout_s", 181.0),
        ("ttft_slo_s", 61.0),
        ("ttft_ratio_limit", 0.71),
        ("goodput_ratio_floor", 0.98),
        ("prompt_token_min", 1_499),
        ("prompt_token_max", 3_001),
    ),
)
def test_formal_identity_thresholds_cannot_be_self_consistently_changed(
    field: str,
    value: object,
) -> None:
    raw = asdict(
        f2c.profile_config("formal", trace_namespace="identity-test")
    )
    raw[field] = value
    with pytest.raises(ValueError, match="complete workload and gate identity"):
        f2c.WorkloadConfig(**raw).validate()


def test_profiles_and_namespaces_produce_disjoint_roots() -> None:
    smoke = f2c.build_trace(
        f2c.profile_config("smoke", trace_namespace="same-name")
    )
    formal = f2c.build_trace(
        f2c.profile_config("formal", trace_namespace="same-name")
    )
    assert {row.prefix_fingerprint for row in smoke}.isdisjoint(
        row.prefix_fingerprint for row in formal
    )
    with pytest.raises(ValueError, match="fresh trace_namespace"):
        f2c.profile_config("formal")
    with pytest.raises(ValueError, match="lower-case safe token"):
        f2c.profile_config("smoke", trace_namespace="Unsafe_Name")


def _valid_runtime_attestation() -> tuple[str, str, dict[str, object]]:
    runtime_image = "registry.example/kairyu@sha256:" + "a" * 64
    expected_commit = "1" * 40
    image_id = "sha256:" + "b" * 64
    services = [
        {
            "service": service,
            "container_id": f"{index + 1:064x}",
            "config_image": runtime_image,
            "image_id": image_id,
            "status": "running",
            "health": "healthy",
            "device_ids": list(f2c.F2C_SERVICE_GPU_IDS[service]),
            "cpuset_cpus": f2c.F2C_SERVICE_CPUSETS[service],
            "port": {
                "container_port": "8000/tcp",
                "host_ip": "127.0.0.1",
                "host_port": f2c.F2C_SERVICE_HOST_PORTS[service],
            },
        }
        for index, service in enumerate(f2c.F2C_SERVICES)
    ]
    attestation = {
        "schema_version": 1,
        "compose_file": f2c.DEFAULT_CONFIG_PATHS[0],
        "runtime_image": runtime_image,
        "services": services,
        "image": {
            "id": image_id,
            "ref": runtime_image,
            "id_inspect_id": image_id,
            "ref_inspect_id": image_id,
            "revision": expected_commit,
        },
    }
    return runtime_image, expected_commit, attestation


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("services", 0, "health"), "starting"),
        (("services", 1, "device_ids"), ["0", "1"]),
        (("services", 2, "cpuset_cpus"), ""),
        (("services", 3, "port", "host_port"), "9999"),
        (("image", "revision"), "2" * 40),
    ),
)
def test_runtime_attestation_binds_live_topology(
    path: tuple[object, ...],
    value: object,
) -> None:
    runtime_image, expected_commit, original = _valid_runtime_attestation()
    assert f2c.valid_runtime_attestation(
        original,
        runtime_image=runtime_image,
        expected_commit=expected_commit,
    )
    changed = json.loads(json.dumps(original))
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not f2c.valid_runtime_attestation(
        changed,
        runtime_image=runtime_image,
        expected_commit=expected_commit,
    )


def test_formal_topology_and_preflight_fail_before_runtime_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = f2c.profile_config("formal", trace_namespace="preflight-test")
    topology = f2c.EndpointTopology(
        f2c.DEFAULT_COHORT_A_ENDPOINTS,
        f2c.DEFAULT_COHORT_B_ENDPOINTS,
    )
    config_paths = tuple(
        f2c._REPO_ROOT / path for path in f2c.DEFAULT_CONFIG_PATHS
    )
    with pytest.raises(ValueError, match="expected-commit"):
        f2c.formal_preflight(
            config,
            topology,
            model="qwen3-32b",
            model_revision="2" * 40,
            model_digests={"model": "a" * 64},
            runtime_image="registry/kairyu@sha256:" + "b" * 64,
            config_paths=config_paths,
            expected_commit=None,
        )
    remote = f2c.EndpointTopology(
        ("http://host:8100/v1", "http://host:8101/v1"),
        f2c.DEFAULT_COHORT_B_ENDPOINTS,
    )
    with pytest.raises(ValueError, match="localhost ports"):
        remote.validate(formal=True)
    with pytest.raises(ValueError, match="model qwen3-32b"):
        f2c.formal_preflight(
            config,
            topology,
            model="other-model",
            model_revision="2" * 40,
            model_digests={"model": "a" * 64},
            runtime_image="registry/kairyu@sha256:" + "b" * 64,
            config_paths=config_paths,
            expected_commit="1" * 40,
        )

    runtime_image, expected_commit, attestation = _valid_runtime_attestation()
    source = {
        "commit": expected_commit,
        "expected_commit": expected_commit,
        "expected_commit_matches": True,
        "git_clean": True,
        "git_status": [],
        "files": [
            {"path": path, "sha256": "a" * 64}
            for path in f2c._SOURCE_PATHS
        ],
    }
    monkeypatch.setattr(f2c, "_source_snapshot", lambda _expected: source)
    monkeypatch.setattr(
        f2c,
        "capture_runtime_attestation",
        lambda **_kwargs: attestation,
    )
    for invalid_revision in ("2" * 39, "A" * 40):
        with pytest.raises(ValueError, match="model revision"):
            f2c.formal_preflight(
                config,
                topology,
                model="qwen3-32b",
                model_revision=invalid_revision,
                model_digests={"model": "a" * 64},
                runtime_image=runtime_image,
                config_paths=config_paths,
                expected_commit=expected_commit,
            )
    assert (
        f2c.formal_preflight(
            config,
            topology,
            model="qwen3-32b",
            model_revision="2" * 40,
            model_digests={"model": "a" * 64},
            runtime_image=runtime_image,
            config_paths=config_paths,
            expected_commit=expected_commit,
        )
        == attestation
    )


def test_formal_configuration_paths_and_model_are_exact_and_portable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = f2c.profile_config("formal", trace_namespace="config-test")
    topology = f2c.EndpointTopology(
        f2c.DEFAULT_COHORT_A_ENDPOINTS,
        f2c.DEFAULT_COHORT_B_ENDPOINTS,
    )
    runtime_image, expected_commit, attestation = _valid_runtime_attestation()
    value = f2c._configuration_snapshot(
        config,
        topology,
        model="qwen3-32b",
        model_revision=expected_commit,
        model_digests={"model": "a" * 64},
        runtime_image=runtime_image,
        runtime_attestation=attestation,
        config_paths=tuple(
            f2c._REPO_ROOT / path for path in f2c.DEFAULT_CONFIG_PATHS
        ),
    )
    assert [item["path"] for item in value["files"]] == list(
        f2c.DEFAULT_CONFIG_PATHS
    )
    assert not Path(attestation["compose_file"]).is_absolute()
    assert f2c._valid_configuration_value(value, config)
    monkeypatch.setattr(f2c, "_REPO_ROOT", tmp_path)
    assert f2c._valid_configuration_value(value, config)
    assert f2c.valid_runtime_attestation(
        attestation,
        runtime_image=runtime_image,
        expected_commit=expected_commit,
    )
    changed = json.loads(json.dumps(value))
    changed["files"].reverse()
    assert not f2c._valid_configuration_value(changed, config)
    changed = json.loads(json.dumps(value))
    changed["model"] = "other-model"
    assert not f2c._valid_configuration_value(changed, config)
    changed = json.loads(json.dumps(value))
    changed["model_revision"] = "A" * 40
    assert not f2c._valid_configuration_value(changed, config)
    changed = json.loads(json.dumps(value))
    changed["files"][0]["path"] = "/checkout/" + f2c.DEFAULT_CONFIG_PATHS[0]
    assert not f2c._valid_configuration_value(changed, config)


def test_formal_failed_artifact_exits_nonzero_without_assert_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_measure(*_args, **_kwargs):
        return []

    monkeypatch.setattr(f2c, "measure", fake_measure)
    monkeypatch.setattr(
        f2c,
        "write_artifact",
        lambda *_args, **_kwargs: {"passed": False},
    )
    assert (
        f2c.main(
            [
                "--profile",
                "formal",
                "--trace-namespace",
                "exit-test",
                "--output-dir",
                str(tmp_path),
                "--model-revision",
                "2" * 40,
                "--model-digest",
                "model=" + "a" * 64,
                "--runtime-image",
                "registry/kairyu@sha256:" + "b" * 64,
                "--expected-commit",
                "1" * 40,
            ]
        )
        == 1
    )
