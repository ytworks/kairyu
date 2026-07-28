import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

import bench.fleet_churn_bench as fleet_bench
from bench.fleet_churn_bench import (
    FORMAL_CONFIG,
    SCHEMA_VERSION,
    SMOKE_CONFIG,
    SurfaceContract,
    _endpoint_uids,
    _one_request,
    _parser,
    _sidecar_descriptor,
    churn_plan,
    evaluate_gate,
    percentile,
    resolve_config,
    source_environment,
    verify_artifact,
)


def _passing_evidence():
    config = SMOKE_CONFIG
    plan = churn_plan(config)
    measurement_origin_ns = 100_000_000_000
    mock_image = "kairyu-f1a-mock:dev"
    mock_digest = "sha256:" + "e" * 64
    mock_runtime_digest = "sha256:" + "f" * 64
    runtime_image_id = "docker-pullable://" + mock_runtime_digest
    gateway_image = "kairyu:dev"
    gateway_digest = "sha256:" + "1" * 64
    gateway_status_digest = "sha256:" + "3" * 64
    gateway_runtime_digest = "sha256:" + "4" * 64
    gateway_runtime_image_id = "docker-pullable://" + gateway_runtime_digest
    old_by_ordinal = {
        ordinal: f"old-{ordinal}" for ordinal in range(config.replica_count)
    }
    new_by_ordinal = {
        ordinal: f"new-{ordinal}" for ordinal in range(config.replica_count)
    }
    current_uids = set(old_by_ordinal.values())
    endpoint_transitions = []
    membership_snapshots = [
        {
            "observed_ns": measurement_origin_ns - 2_000_000_000,
            "replica_ids": set(current_uids),
            "healthy_ids": set(current_uids),
            "eligible_ids": set(current_uids),
        }
    ]
    churn = []
    for entry in plan:
        endpoint_state_before = set(current_uids)
        old = []
        new = []
        old_epoch_uids = set()
        new_epoch_uids = set()
        for ordinal in entry["ordinals"]:
            name = f"f1a-replica-{ordinal}"
            old_uid = old_by_ordinal[ordinal]
            new_uid = new_by_ordinal[ordinal]
            old.append({"name": name, "uid": old_uid})
            new.append({"name": name, "uid": new_uid})
            old_epoch_uids.add(old_uid)
            new_epoch_uids.add(new_uid)
        scheduled_ns = measurement_origin_ns + int(
            entry["deadline_offset_s"] * 1_000_000_000
        )
        api_started_ns = scheduled_ns + 250_000
        api_completed_ns = api_started_ns + 10_000_000
        membership_snapshots.append(
            {
                "observed_ns": api_started_ns,
                "replica_ids": set(current_uids),
                "healthy_ids": set(current_uids),
                "eligible_ids": current_uids - old_epoch_uids,
            }
        )
        for old_uid, new_uid in zip(
            sorted(old_epoch_uids),
            sorted(new_epoch_uids),
            strict=True,
        ):
            current_uids.remove(old_uid)
            current_uids.add(new_uid)
        endpoint_transitions.append(
            (endpoint_state_before, set(current_uids))
        )
        membership_snapshots.append(
            {
                "observed_ns": api_started_ns + 500_000_000,
                "replica_ids": set(current_uids),
                "healthy_ids": set(current_uids),
                "eligible_ids": set(current_uids),
            }
        )
        churn.append(
            {
                "epoch": entry["epoch"],
                "deadline_offset_s": entry["deadline_offset_s"],
                "ordinals": entry["ordinals"],
                "scheduled_ns": scheduled_ns,
                "api_started_ns": api_started_ns,
                "api_completed_ns": api_completed_ns,
                "delete_returncode": 0,
                "error": None,
                "recovery_seconds": 0.5,
                "schedule_lateness_ms": 0.25,
                "old_withdrawn_ns": api_started_ns + 250_000_000,
                "old_withdrawal_seconds": 0.25,
                "old_withdrawal_margin_seconds": 4.75,
                "new_ready_ns": api_started_ns + 500_000_000,
                "new_ready_seconds": 0.5,
                "recovered_ns": api_started_ns + 500_000_000,
                "old": old,
                "new": new,
                "endpoint_uids_at_recovery": sorted(current_uids),
                "ready_endpoint_count_at_recovery": config.replica_count,
            }
        )

    source = "source-id"
    memberships = []
    for sequence, snapshot in enumerate(membership_snapshots, 1):
        replica_ids = snapshot["replica_ids"]
        memberships.append(
            {
                "kind": "membership",
                "observed_ns": snapshot["observed_ns"],
                "event_source_id": source,
                "sequence": sequence,
                "event_id": f"{source}:{sequence:020d}",
                "replica_ids": sorted(replica_ids),
                "healthy_ids": sorted(snapshot["healthy_ids"]),
                "eligible_ids": sorted(snapshot["eligible_ids"]),
                "generation_by_id": {
                    uid: f"generation-{uid}" for uid in replica_ids
                },
            }
        )

    requests = []
    replica_placements = []
    interval_ns = int(1_000_000_000 / config.request_rate)
    traffic_origin_ns = measurement_origin_ns - int(
        config.warmup_seconds * 1_000_000_000
    )
    for sequence in range(config.expected_requests):
        response_id = f"server-{sequence}"
        send_ns = traffic_origin_ns + sequence * interval_ns
        selected_at_ns = send_ns + 1_000_000
        latest_membership = max(
            (
                record
                for record in memberships
                if record["observed_ns"] <= selected_at_ns
            ),
            key=lambda record: record["observed_ns"],
        )
        replica_id = latest_membership["eligible_ids"][0]
        replica_generation = latest_membership["generation_by_id"][replica_id]
        ordinal = replica_id.split("-", 1)[1]
        requests.append(
            {
                "sequence": sequence,
                "scheduled_ns": send_ns,
                "measurement_offset_s": (
                    send_ns - measurement_origin_ns
                )
                / 1_000_000_000,
                "send_ns": send_ns,
                "end_ns": send_ns + 2_000_000,
                "pacing_error_ms": 0.0,
                "not_sent": False,
                "transport_error": None,
                "status_code": 200,
                "request_id_echo": response_id,
                "response_valid": True,
                "pod_name": f"f1a-replica-{ordinal}",
                "pod_uid": replica_id,
            }
        )
        replica_placements.append(
            {
                "kind": "replica",
                "request_id": response_id,
                "replica_id": replica_id,
                "replica_generation": replica_generation,
                "placement_latency_ns": 1_000_000,
                "placement_started_ns": send_ns,
                "selected_at_ns": selected_at_ns,
                "pool_size": len(latest_membership["replica_ids"]),
                "eligible_size": len(latest_membership["eligible_ids"]),
            }
        )

    def endpoint_record(
        endpoint_state,
        observed_ns,
        *,
        capture=None,
        epoch=None,
    ):
        record = {
            "error": None,
            "observed_ns": observed_ns,
            "payload": {
                "items": [
                    {
                        "endpoints": [
                            {
                                "conditions": {
                                    "ready": True,
                                    "terminating": False,
                                },
                                "targetRef": {"uid": uid},
                            }
                            for uid in sorted(endpoint_state)
                        ]
                    }
                ]
            },
        }
        if capture is not None:
            record["capture"] = capture
        if epoch is not None:
            record["epoch"] = epoch
        return record

    endpoints = [
        endpoint_record(
            set(old_by_ordinal.values()),
            measurement_origin_ns - 1_000_000_000,
        )
    ]
    for epoch, (endpoint_before, endpoint_after) in enumerate(
        endpoint_transitions
    ):
        churn_record = churn[epoch]
        endpoints.extend(
            [
                endpoint_record(
                    endpoint_before,
                    churn_record["api_started_ns"] - 100_000_000,
                    capture="churn_predelete",
                    epoch=epoch,
                ),
                endpoint_record(
                    endpoint_before,
                    churn_record["api_started_ns"] + 100_000_000,
                    capture="churn_recovery",
                    epoch=epoch,
                ),
                endpoint_record(
                    endpoint_after,
                    churn_record["old_withdrawn_ns"],
                    capture="churn_recovery",
                    epoch=epoch,
                ),
                endpoint_record(
                    endpoint_after,
                    churn_record["recovered_ns"],
                ),
            ]
        )
    placements = [*memberships, *replica_placements]

    def pod_identity(uid):
        ordinal = uid.split("-", 1)[1]
        return {
            "name": f"f1a-replica-{ordinal}",
            "uid": uid,
            "containers": [
                {
                    "name": "mock",
                    "spec_image": mock_image,
                    "runtime_image": (
                        f"docker.io/library/{mock_image}"
                        if uid.startswith("new-")
                        else mock_image
                    ),
                    "image_id": runtime_image_id,
                }
            ],
        }

    pods = [
        {
            "error": None,
            "payload": [
                pod_identity(uid) for uid in sorted(old_by_ordinal.values())
            ],
        },
        {
            "error": None,
            "payload": [
                pod_identity(uid) for uid in sorted(new_by_ordinal.values())
            ],
        },
        {
            "error": None,
            "payload": [
                {
                    "name": "f1a-gateway",
                    "uid": "gateway-uid",
                    "containers": [
                        {
                            "name": "gateway",
                            "spec_image": gateway_image,
                            "runtime_image": (
                                f"docker.io/library/{gateway_image}"
                            ),
                            "image_id": gateway_runtime_image_id,
                        }
                    ],
                }
            ],
        },
    ]
    resources = [{"error": None, "payload": {"node": {}, "pods": []}}]
    environment = {
        "git_commit": "a" * 40,
        "expected_git_commit": "a" * 40,
        "git_commit_matches_expected": True,
        "tracked_dirty": False,
        "tracked_diff_sha256": "0" * 64,
        "source_dirty": False,
        "source_end_match": True,
        "untracked_gate_input_dirty": False,
        "untracked_gate_inputs": [],
        "benchmark_sha256": "b" * 64,
        "driver_sha256": "c" * 64,
        "gateway_image_digest": gateway_digest,
        "mock_image_digest": mock_digest,
        "gateway_containerd_image_digest": gateway_digest,
        "mock_containerd_image_digest": mock_digest,
        "gateway_cri_image_metadata": {
            "path": "gateway-cri-image.json",
            "sha256": "5" * 64,
            "status_id": gateway_status_digest,
            "repo_digests": [
                "docker.io/library/import@"
                + gateway_runtime_digest
            ],
            "repo_tags": [f"docker.io/library/{gateway_image}"],
            "runtime_digests": [
                gateway_status_digest,
                gateway_runtime_digest,
            ],
            "revision": "a" * 40,
        },
        "mock_cri_image_metadata": {
            "path": "mock-cri-image.json",
            "sha256": "6" * 64,
            "status_id": mock_runtime_digest,
            "repo_digests": [],
            "repo_tags": [f"docker.io/library/{mock_image}"],
            "runtime_digests": [mock_runtime_digest],
            "revision": "a" * 40,
        },
        "kind_node_image": "kindest/node@sha256:" + "2" * 64,
        "tool_versions": {
            "kind": "kind v0.29.0",
            "kubectl": "v1.33.0",
            "docker": "29.6.1",
        },
        "frozen_inputs": [{"path": "manifest.yaml", "sha256": "d" * 64}],
        "frozen_sidecars": [
            {
                "path": "rendered-kind-config.yaml",
                "sha256": "7" * 64,
            }
        ],
        "statefulset": "f1a-replica",
        "mock_container_name": "mock",
        "expected_mock_image": mock_image,
        "runtime_mock_image": {
            "container_name": "mock",
            "spec_images": [mock_image],
            "runtime_images": [
                f"docker.io/library/{mock_image}",
                mock_image,
            ],
            "image_ids": [runtime_image_id],
        },
        "gateway_container_name": "gateway",
        "expected_gateway_image": gateway_image,
        "runtime_gateway_image": {
            "container_name": "gateway",
            "spec_images": [gateway_image],
            "runtime_images": [f"docker.io/library/{gateway_image}"],
            "image_ids": [gateway_runtime_image_id],
        },
    }
    return {
        "config": config,
        "requests": requests,
        "placements": placements,
        "churn": churn,
        "endpoints": endpoints,
        "pods": pods,
        "resources": resources,
        "environment": environment,
    }


def test_formal_schedule_is_fixed_disjoint_and_complete() -> None:
    first = churn_plan(FORMAL_CONFIG)
    second = churn_plan(FORMAL_CONFIG)
    flattened = [ordinal for entry in first for ordinal in entry["ordinals"]]

    assert first == second
    assert len(first) == 10
    assert [entry["deadline_offset_s"] for entry in first] == [
        float(value) for value in range(0, 600, 60)
    ]
    assert all(len(entry["ordinals"]) == 20 for entry in first)
    assert len(flattened) == len(set(flattened)) == 200
    assert sorted(flattened) == list(range(200))


def test_smoke_epoch_exceeds_mock_prestop_grace() -> None:
    assert SMOKE_CONFIG.epoch_seconds == 20.0
    assert SMOKE_CONFIG.recovery_timeout_seconds == 20.0
    assert (
        SMOKE_CONFIG.epoch_seconds
        > SMOKE_CONFIG.endpoint_withdrawal_limit_seconds
    )


def test_formal_profile_refuses_configuration_drift() -> None:
    parser = _parser()

    assert resolve_config(parser.parse_args(["--profile", "formal"])) == FORMAL_CONFIG
    with pytest.raises(ValueError, match="request_rate is frozen"):
        resolve_config(
            parser.parse_args(
                ["--profile", "formal", "--request-rate", "49"]
            )
        )


def test_percentile_uses_raw_nearest_rank_samples() -> None:
    assert percentile([float(index) for index in range(1, 101)], 0.99) == 99.0


def test_endpoint_uids_only_counts_ready_nonterminating_targets() -> None:
    payload = {
        "items": [
            {
                "endpoints": [
                    {
                        "conditions": {
                            "ready": True,
                            "terminating": False,
                        },
                        "targetRef": {"uid": "eligible"},
                    },
                    {
                        "conditions": {
                            "ready": False,
                            "terminating": False,
                        },
                        "targetRef": {"uid": "not-ready"},
                    },
                    {
                        "conditions": {
                            "ready": True,
                            "terminating": True,
                        },
                        "targetRef": {"uid": "terminating"},
                    },
                    {
                        "conditions": {},
                        "targetRef": {"uid": "unknown"},
                    },
                ]
            }
        ]
    }

    assert _endpoint_uids(payload) == {"eligible"}


async def test_request_extracts_server_id_and_mock_pod_identity() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Request-ID"].startswith("f1a-00000007-")
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"X-Request-ID": "server-assigned-id"},
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "kairyu-f1a-mock pod_name=f1a-replica-3 "
                                "pod_uid=pod-uid-3"
                            )
                        }
                    }
                ]
            },
        )

    now_ns = time.monotonic_ns()
    async with httpx.AsyncClient(
        base_url="http://gateway",
        transport=httpx.MockTransport(handler),
    ) as client:
        record = await _one_request(
            client,
            sequence=7,
            target_ns=now_ns,
            measurement_origin_ns=now_ns,
            model="f1a",
            contract=SurfaceContract(request_id_header="X-Request-ID"),
        )

    assert record["request_id_echo"] == "server-assigned-id"
    assert record["request_id"] != record["request_id_echo"]
    assert record["pod_name"] == "f1a-replica-3"
    assert record["pod_uid"] == "pod-uid-3"
    assert record["response_valid"]


async def test_request_does_not_trust_top_level_pod_identity() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-Request-ID": "server-assigned-id"},
            json={
                "pod_name": "forged",
                "pod_uid": "forged",
                "choices": [{"message": {"content": "ordinary text"}}],
            },
        )

    now_ns = time.monotonic_ns()
    async with httpx.AsyncClient(
        base_url="http://gateway",
        transport=httpx.MockTransport(handler),
    ) as client:
        record = await _one_request(
            client,
            sequence=8,
            target_ns=now_ns,
            measurement_origin_ns=now_ns,
            model="f1a",
            contract=SurfaceContract(request_id_header="X-Request-ID"),
        )

    assert record["response_valid"]
    assert record["pod_name"] is None
    assert record["pod_uid"] is None


def test_gate_accepts_complete_smoke_evidence() -> None:
    evidence = _passing_evidence()

    result = evaluate_gate(**evidence)

    assert result["passed"]
    assert all(result["checks"].values())
    assert len(result["placement"]["epochs"]) == SMOKE_CONFIG.churn_epochs
    assert len(result["placement"]["churn_windows"]) == SMOKE_CONFIG.churn_epochs


def test_gate_canonicalizes_out_of_order_request_completion_rows() -> None:
    evidence = _passing_evidence()
    evidence["requests"].reverse()

    result = evaluate_gate(**evidence)

    assert result["passed"]
    assert result["checks"]["request_raw_timeline_consistent"]


def test_placement_p99_excludes_warmup_and_cooldown_samples() -> None:
    evidence = _passing_evidence()
    by_request = {
        row.get("request_id"): row for row in evidence["placements"]
    }
    requests = {
        row["request_id_echo"]: row for row in evidence["requests"]
    }
    for request_id in (
        "server-0",
        f"server-{SMOKE_CONFIG.expected_requests - 1}",
    ):
        placement = by_request[request_id]
        request = requests[request_id]
        placement["placement_latency_ns"] = 1_000_000_000
        placement["selected_at_ns"] = request["send_ns"] + 1_000_000_000
        request["end_ns"] = placement["selected_at_ns"] + 1_000_000

    result = evaluate_gate(**evidence)

    assert result["passed"]
    assert result["placement"]["overall"]["p99_ms"] == 1.0


@pytest.mark.parametrize(
    "forged_latency_ns",
    [-1, 0, True, float("inf"), 1_000_001],
)
def test_gate_rejects_invalid_raw_placement_latency(
    forged_latency_ns,
) -> None:
    evidence = _passing_evidence()
    next(
        row
        for row in evidence["placements"]
        if row.get("request_id") == "server-10"
    )["placement_latency_ns"] = forged_latency_ns

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["placement_latency_raw_consistent"]
    assert not result["checks"]["placement_samples_complete"]


def test_gate_requires_exact_echoed_request_to_placement_join() -> None:
    evidence = _passing_evidence()
    next(
        row
        for row in evidence["placements"]
        if row.get("request_id") == "server-10"
    )["request_id"] = "unmatched"

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["request_id_correlation_complete"]
    assert not result["checks"]["placement_samples_complete"]


def test_gate_rejects_unrelated_placement_rows() -> None:
    evidence = _passing_evidence()
    evidence["placements"].append(
        {
            "kind": "replica",
            "request_id": "manual-preflight",
            "replica_id": "old-0",
            "placement_latency_ns": 999_000_000,
        }
    )

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["request_id_correlation_complete"]
    assert result["placement"]["unrelated_replica_rows"] == 1
    assert result["placement"]["overall"]["p99_ms"] == 1.0


def test_gate_rejects_endpoint_withdrawal_at_prestop_limit() -> None:
    evidence = _passing_evidence()
    evidence["churn"][0]["old_withdrawal_seconds"] = (
        SMOKE_CONFIG.endpoint_withdrawal_limit_seconds
    )
    evidence["churn"][0]["old_withdrawal_margin_seconds"] = 0.0

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["old_endpoints_withdrawn_before_grace"]


def test_gate_rejects_recovery_with_inexact_ready_endpoint_count() -> None:
    evidence = _passing_evidence()
    evidence["churn"][0]["endpoint_uids_at_recovery"].pop()
    evidence["churn"][0]["ready_endpoint_count_at_recovery"] -= 1

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["new_endpoints_ready_with_exact_fleet_size"]


def test_gate_rejects_rogue_uid_with_preserved_endpoint_count() -> None:
    evidence = _passing_evidence()
    observed = evidence["churn"][0]["endpoint_uids_at_recovery"]
    observed[-1] = "rogue-uid"

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["new_endpoints_ready_with_exact_fleet_size"]
    assert not result["churn"]["endpoint_epoch_recoveries"][0]["exact"]


def test_gate_requires_periodic_exact_endpoint_snapshot_per_epoch() -> None:
    evidence = _passing_evidence()
    first_recovered_ns = evidence["churn"][0]["recovered_ns"]
    snapshot = next(
        row
        for row in evidence["endpoints"]
        if row["observed_ns"] == first_recovered_ns
    )
    endpoints = snapshot["payload"]["items"][0]["endpoints"]
    endpoints[-1]["targetRef"]["uid"] = "rogue-periodic-uid"

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["new_endpoints_ready_with_exact_fleet_size"]


def test_periodic_endpoint_fetch_started_inside_window_may_finish_after_it() -> None:
    evidence = _passing_evidence()
    churn = evidence["churn"][0]
    snapshot = next(
        row
        for row in evidence["endpoints"]
        if row.get("capture") is None
        and row["observed_ns"] == churn["recovered_ns"]
    )
    interval_ns = int(
        SMOKE_CONFIG.evidence_interval_seconds * 1_000_000_000
    )
    snapshot["fetch_started_ns"] = churn["recovered_ns"] + interval_ns - 1
    snapshot["observed_ns"] = churn["recovered_ns"] + interval_ns + 1

    result = evaluate_gate(**evidence)

    assert result["passed"]
    assert result["checks"]["new_endpoints_ready_with_exact_fleet_size"]


def test_gate_rejects_inexact_predelete_endpoint_snapshot() -> None:
    evidence = _passing_evidence()
    snapshot = next(
        row
        for row in evidence["endpoints"]
        if row.get("capture") == "churn_predelete"
        and row.get("epoch") == 0
    )
    snapshot["payload"]["items"][0]["endpoints"].pop()

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["endpoint_predelete_snapshots_exact"]


def test_gate_rejects_old_endpoint_reappearance_after_withdrawal() -> None:
    evidence = _passing_evidence()
    old_snapshot = next(
        row
        for row in evidence["endpoints"]
        if row.get("capture") == "churn_predelete"
        and row.get("epoch") == 0
    )
    reappeared = json.loads(json.dumps(old_snapshot))
    reappeared["capture"] = "periodic-forged"
    reappeared["observed_ns"] = (
        evidence["churn"][0]["old_withdrawn_ns"] + 100_000_000
    )
    evidence["endpoints"].append(reappeared)

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["endpoint_raw_withdrawal_causal"]


def test_gate_rejects_rogue_uid_in_cumulative_membership_recovery() -> None:
    evidence = _passing_evidence()
    membership = [
        row for row in evidence["placements"] if row.get("kind") == "membership"
    ][2]
    removed = membership["replica_ids"][-1]
    for key in ("replica_ids", "healthy_ids", "eligible_ids"):
        membership[key][-1] = "rogue-membership-uid"
    generation = membership["generation_by_id"].pop(removed)
    membership["generation_by_id"]["rogue-membership-uid"] = generation

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["gateway_membership_recovers_each_epoch"]


def test_gate_rejects_churn_name_to_ordinal_swap() -> None:
    evidence = _passing_evidence()
    first, second = evidence["churn"][0]["old"]
    first["name"], second["name"] = second["name"], first["name"]

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["pod_uid_replaced_exactly_once"]


def test_gate_requires_final_gateway_membership_to_be_fully_eligible() -> None:
    evidence = _passing_evidence()
    memberships = [
        row for row in evidence["placements"] if row.get("kind") == "membership"
    ]
    memberships[-1]["eligible_ids"].pop()

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"][
        "gateway_final_membership_all_replacements_eligible"
    ]


def test_gate_requires_initial_gateway_membership_to_be_fully_eligible() -> None:
    evidence = _passing_evidence()
    memberships = [
        row for row in evidence["placements"] if row.get("kind") == "membership"
    ]
    memberships[0]["healthy_ids"].pop()

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["gateway_initial_membership_all_eligible"]


def test_gate_requires_gateway_withdrawal_by_endpoint_deadline() -> None:
    evidence = _passing_evidence()
    memberships = [
        row for row in evidence["placements"] if row.get("kind") == "membership"
    ]
    deadline_ns = (
        evidence["churn"][0]["old_withdrawn_ns"]
        + int(SMOKE_CONFIG.evidence_interval_seconds * 1_000_000_000)
    )
    memberships[1]["observed_ns"] = deadline_ns + 100_000_000
    memberships[2]["observed_ns"] = deadline_ns + 200_000_000

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["gateway_membership_withdrawal_causal"]


def test_gate_rejects_membership_sequence_gap_or_duplicate() -> None:
    evidence = _passing_evidence()
    memberships = [
        row for row in evidence["placements"] if row.get("kind") == "membership"
    ]
    memberships[1]["sequence"] = memberships[0]["sequence"]
    memberships[1]["event_id"] = memberships[0]["event_id"]

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["membership_event_stream_consistent"]


def test_gate_rejects_nonmonotonic_membership_capture_time() -> None:
    evidence = _passing_evidence()
    memberships = [
        row for row in evidence["placements"] if row.get("kind") == "membership"
    ]
    memberships[2]["observed_ns"] = memberships[1]["observed_ns"] - 1

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["membership_event_stream_consistent"]


def test_gate_rejects_duplicate_membership_identity() -> None:
    evidence = _passing_evidence()
    membership = next(
        row for row in evidence["placements"] if row.get("kind") == "membership"
    )
    membership["eligible_ids"].append(membership["eligible_ids"][0])

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["membership_identity_lists_unique"]


@pytest.mark.parametrize("set_name", ["healthy_ids", "eligible_ids"])
def test_gate_rejects_membership_identity_outside_replica_set(
    set_name,
) -> None:
    evidence = _passing_evidence()
    membership = next(
        row
        for row in evidence["placements"]
        if row.get("kind") == "membership"
    )
    membership[set_name].append("rogue-not-a-replica")

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["membership_set_invariants_valid"]


def test_gate_requires_full_gateway_recovery_in_each_epoch_window() -> None:
    evidence = _passing_evidence()
    memberships = [
        row for row in evidence["placements"] if row.get("kind") == "membership"
    ]
    first_churn_ns = evidence["churn"][0]["api_started_ns"]
    memberships[2]["observed_ns"] = first_churn_ns + int(
        (SMOKE_CONFIG.recovery_timeout_seconds + 0.1) * 1_000_000_000
    )

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["gateway_membership_recovers_each_epoch"]
    assert result["membership"]["epoch_recoveries"][0]["event_id"] is None


def test_gate_rejects_placement_on_not_yet_created_uid() -> None:
    evidence = _passing_evidence()
    request = evidence["requests"][0]
    placement = next(
        row
        for row in evidence["placements"]
        if row.get("request_id") == request["request_id_echo"]
    )
    request.update(
        pod_name="f1a-replica-0",
        pod_uid="new-0",
    )
    placement.update(
        replica_id="new-0",
        replica_generation="generation-new-0",
    )

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["placement_membership_temporal_join_complete"]


def test_gate_rejects_placement_on_deleted_uid() -> None:
    evidence = _passing_evidence()
    request = next(
        row
        for row in evidence["requests"]
        if 1.0 <= row["measurement_offset_s"] < 2.0
    )
    deleted = evidence["churn"][0]["old"][0]
    placement = next(
        row
        for row in evidence["placements"]
        if row.get("request_id") == request["request_id_echo"]
    )
    request.update(
        pod_name=deleted["name"],
        pod_uid=deleted["uid"],
    )
    placement.update(
        replica_id=deleted["uid"],
        replica_generation=f"generation-{deleted['uid']}",
    )

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["placement_membership_temporal_join_complete"]


def test_gate_rejects_placement_generation_or_timestamp_mismatch() -> None:
    evidence = _passing_evidence()
    request = evidence["requests"][0]
    placement = next(
        row
        for row in evidence["placements"]
        if row.get("request_id") == request["request_id_echo"]
    )
    placement["replica_generation"] = "wrong-generation"
    placement["selected_at_ns"] = request["send_ns"] - 1

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["placement_membership_temporal_join_complete"]


def test_gate_rejects_runtime_mock_image_id_drift() -> None:
    evidence = _passing_evidence()
    mock_container = next(
        container
        for record in evidence["pods"]
        for identity in record["payload"]
        for container in identity["containers"]
        if container["name"] == "mock"
    )
    mock_container["image_id"] = "docker-pullable://sha256:" + "a" * 64

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["runtime_mock_image_pinned"]


def test_gate_rejects_containerd_target_not_matching_built_image() -> None:
    evidence = _passing_evidence()
    evidence["environment"]["mock_image_digest"] = "sha256:" + "a" * 64

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["provenance_pinned"]


def test_gate_rejects_runtime_digest_not_allowed_by_cri_metadata() -> None:
    evidence = _passing_evidence()
    metadata = evidence["environment"]["mock_cri_image_metadata"]
    metadata["status_id"] = "sha256:" + "a" * 64
    metadata["runtime_digests"] = [metadata["status_id"]]

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["runtime_mock_image_pinned"]


def test_gate_rejects_cri_revision_not_matching_source() -> None:
    evidence = _passing_evidence()
    evidence["environment"]["gateway_cri_image_metadata"]["revision"] = (
        "b" * 40
    )

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["provenance_pinned"]


def test_gate_rejects_gateway_runtime_image_id_drift() -> None:
    evidence = _passing_evidence()
    gateway_container = next(
        container
        for record in evidence["pods"]
        for identity in record["payload"]
        for container in identity["containers"]
        if container["name"] == "gateway"
    )
    gateway_container["image_id"] = (
        "docker-pullable://sha256:" + "a" * 64
    )

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["runtime_gateway_image_pinned"]


def test_gate_rejects_response_pod_name_not_matching_uid_evidence() -> None:
    evidence = _passing_evidence()
    evidence["requests"][10]["pod_name"] = "forged-name"

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["response_and_identity_complete"]


def test_gate_rejects_one_uid_observed_under_multiple_pod_names() -> None:
    evidence = _passing_evidence()
    identity = json.loads(json.dumps(evidence["pods"][0]["payload"][0]))
    identity["name"] = "forged-second-name"
    evidence["pods"][1]["payload"].append(identity)

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["pod_uid_name_mapping_consistent"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence["requests"][0].update(
            measurement_offset_s=123.0
        ),
        lambda evidence: evidence["requests"][1].update(sequence=0),
        lambda evidence: evidence["churn"][0].update(recovery_seconds=0.01),
        lambda evidence: evidence["churn"][0].update(
            recovered_ns=evidence["churn"][0]["api_started_ns"] - 1
        ),
        lambda evidence: evidence["churn"][1].update(
            deadline_offset_s=9.0
        ),
    ],
)
def test_gate_rejects_forged_raw_timeline_or_derived_scalar(mutation) -> None:
    evidence = _passing_evidence()
    mutation(evidence)

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not (
        result["checks"]["request_raw_timeline_consistent"]
        and result["checks"]["churn_raw_timeline_consistent"]
    )


def test_gate_rejects_missing_request_sequence() -> None:
    evidence = _passing_evidence()
    evidence["requests"].pop(10)
    evidence["placements"] = [
        row
        for row in evidence["placements"]
        if row.get("request_id") != "server-10"
    ]

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["request_count_matches_schedule"]
    assert not result["checks"]["request_raw_timeline_consistent"]


def test_gate_requires_strict_provenance_and_stable_source() -> None:
    evidence = _passing_evidence()
    evidence["environment"]["tool_versions"].pop("docker")
    evidence["environment"]["source_end_match"] = False

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"]["provenance_pinned"]
    assert not result["checks"]["source_stable_during_run"]


def _make_first_churn_window_placement_hit_limit(evidence) -> None:
    request = evidence["requests"][10]
    placement = next(
        row
        for row in evidence["placements"]
        if row.get("request_id") == request["request_id_echo"]
    )
    placement.update(
        placement_started_ns=request["send_ns"],
        selected_at_ns=request["send_ns"] + 10_000_000,
        placement_latency_ns=10_000_000,
    )
    request["end_ns"] = placement["selected_at_ns"] + 1_000_000


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda evidence: evidence["requests"][10].update(status_code=429),
            "all_requests_valid_2xx",
        ),
        (
            lambda evidence: evidence["requests"][10].update(
                transport_error="timeout"
            ),
            "zero_transport_or_not_sent",
        ),
        (
            lambda evidence: evidence["requests"][10].update(
                not_sent=True,
                status_code=None,
            ),
            "zero_transport_or_not_sent",
        ),
        (
            _make_first_churn_window_placement_hit_limit,
            "placement_p99_all_windows",
        ),
    ],
)
def test_gate_counts_each_request_failure_class(mutation, failed_check) -> None:
    evidence = _passing_evidence()
    mutation(evidence)

    result = evaluate_gate(**evidence)

    assert not result["passed"]
    assert not result["checks"][failed_check]


def test_source_environment_rejects_untracked_gate_input(
    monkeypatch,
    tmp_path,
) -> None:
    driver = tmp_path / "driver.sh"
    driver.write_text("#!/bin/sh\n")
    frozen = tmp_path / "manifest.yaml"
    frozen.write_text("kind: List\n")
    gateway_cri = tmp_path / "gateway-cri-image.json"
    gateway_cri.write_text(
        json.dumps(
            {
                "status": {
                    "id": "sha256:" + "4" * 64,
                    "repoDigests": [
                        "docker.io/library/import@sha256:" + "5" * 64
                    ],
                    "repoTags": ["docker.io/library/kairyu:dev"],
                },
                "info": {
                    "imageSpec": {
                        "config": {
                            "Labels": {
                                "org.opencontainers.image.revision": "a" * 40
                            }
                        }
                    }
                },
            }
        )
    )
    mock_cri = tmp_path / "mock-cri-image.json"
    mock_cri.write_text(
        json.dumps(
            {
                "status": {
                    "id": "sha256:" + "6" * 64,
                    "repoDigests": [],
                    "repoTags": [
                        "docker.io/library/kairyu-f1a-mock:dev"
                    ],
                },
                "info": {
                    "imageSpec": {
                        "config": {
                            "Labels": {
                                "org.opencontainers.image.revision": "a" * 40
                            }
                        }
                    }
                },
            }
        )
    )
    repository = str(fleet_bench.Path.cwd())

    def fake_git_output(*command, binary=False):
        if command == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if command == ("diff", "--binary", "HEAD"):
            assert binary
            return b""
        if command == ("rev-parse", "--show-toplevel"):
            return repository + "\n"
        if command == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if command[:3] == (
            "ls-files",
            "--others",
            "--exclude-standard",
        ):
            assert "kairyu" in command
            assert "deploy/kind/f1a" in command
            assert ".dockerignore" in command
            return "kairyu/untracked_kernel.py\n"
        raise AssertionError(command)

    monkeypatch.setattr(fleet_bench, "_git_output", fake_git_output)
    args = SimpleNamespace(
        driver_path=str(driver),
        frozen_input=[str(frozen)],
        frozen_sidecar=[str(frozen)],
        expected_git_commit="a" * 40,
        gateway_image_digest="sha256:" + "1" * 64,
        mock_image_digest="sha256:" + "2" * 64,
        gateway_containerd_image_digest="sha256:" + "1" * 64,
        mock_containerd_image_digest="sha256:" + "2" * 64,
        gateway_cri_image_metadata=str(gateway_cri),
        mock_cri_image_metadata=str(mock_cri),
        kind_node_image="kindest/node@sha256:" + "3" * 64,
        kind_version="kind v0",
        kubectl_version="v1",
        docker_version="29",
        output=tmp_path / "manifest.json",
    )

    environment = source_environment(args)

    assert not environment["tracked_dirty"]
    assert environment["untracked_gate_input_dirty"]
    assert environment["source_dirty"]
    assert environment["untracked_gate_inputs"] == [
        "kairyu/untracked_kernel.py"
    ]
    assert environment["frozen_sidecars"] == [
        {
            "path": "manifest.yaml",
            "sha256": fleet_bench._sha256_file(frozen),
        }
    ]

    evidence = _passing_evidence()
    evidence["environment"].update(environment)
    evidence["environment"]["source_end_match"] = True
    result = evaluate_gate(**evidence)
    assert not result["checks"]["clean_source"]


async def test_cancelled_command_kills_term_ignoring_child(
    monkeypatch,
) -> None:
    class TermIgnoringProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.communicate_started = asyncio.Event()
            self.killed = asyncio.Event()
            self.terminated = False
            self.kill_called = False

        async def communicate(self):
            self.communicate_started.set()
            await self.killed.wait()
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.kill_called = True
            self.returncode = -9
            self.killed.set()

    process = TermIgnoringProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(
        fleet_bench.asyncio,
        "create_subprocess_exec",
        create_process,
    )
    monkeypatch.setattr(
        fleet_bench,
        "_COMMAND_TERMINATE_GRACE_SECONDS",
        0.01,
    )
    task = asyncio.create_task(fleet_bench._run_command("ignored"))
    await asyncio.wait_for(process.communicate_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert process.terminated is True
    assert process.kill_called is True


async def test_run_cancels_owned_tasks_before_closing_sinks(
    monkeypatch,
    tmp_path,
) -> None:
    sinks = []
    finalized = []

    class RecordingSink:
        def __init__(self, path):
            self.path = path
            self.closed = False
            sinks.append(self)

        def write(self, value):
            assert not self.closed

        def close(self):
            self.closed = True

    async def fake_inventory(_args, _config):
        return [
            {
                "name": f"f1a-replica-{index}",
                "uid": f"old-{index}",
                "node": "node",
            }
            for index in range(SMOKE_CONFIG.replica_count)
        ]

    async def fake_snapshot(*, kind, sink, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            sink.write({"kind": kind})
            finalized.append(kind)

    async def fake_traffic(*_args):
        await asyncio.sleep(0)
        raise RuntimeError("traffic failed")

    async def fake_churn(*args):
        sink = args[-1]
        try:
            await asyncio.Event().wait()
        finally:
            sink.write({"kind": "churn"})
            finalized.append("churn")

    monkeypatch.setattr(fleet_bench, "JsonlSink", RecordingSink)
    monkeypatch.setattr(
        fleet_bench,
        "source_environment",
        lambda _args: {"stable": True},
    )
    monkeypatch.setattr(fleet_bench, "_initial_inventory", fake_inventory)
    monkeypatch.setattr(fleet_bench, "_snapshot_loop", fake_snapshot)
    monkeypatch.setattr(fleet_bench, "_traffic_loop", fake_traffic)
    monkeypatch.setattr(fleet_bench, "_churn_loop", fake_churn)
    args = SimpleNamespace(
        output=tmp_path / "manifest.json",
        statefulset="f1a-replica",
        namespace="kairyu-f1a",
        kubectl="kubectl",
        endpoint_service="f1a-replicas",
        pod_selector="app=f1a",
    )

    with pytest.raises(RuntimeError, match="traffic failed"):
        await fleet_bench.run(args, SMOKE_CONFIG)

    assert set(finalized) == {
        "endpointslices",
        "pods",
        "resources",
        "churn",
    }
    assert sinks and all(sink.closed for sink in sinks)


def test_artifact_verifier_rehashes_and_replays_sidecars(tmp_path) -> None:
    evidence = _passing_evidence()
    rendered_kind_config = tmp_path / "rendered-kind-config.yaml"
    rendered_kind_config.write_text(
        "kind: Cluster\nnodes:\n  - role: control-plane\n",
        encoding="utf-8",
    )
    evidence["environment"]["frozen_sidecars"] = [
        {
            "path": rendered_kind_config.name,
            "sha256": fleet_bench._sha256_file(rendered_kind_config),
        }
    ]
    for role in ("gateway", "mock"):
        key = f"{role}_cri_image_metadata"
        metadata = evidence["environment"][key]
        path = tmp_path / metadata["path"]
        path.write_text(
            json.dumps(
                {
                    "status": {
                        "id": metadata["status_id"],
                        "repoDigests": metadata["repo_digests"],
                        "repoTags": metadata["repo_tags"],
                    },
                    "info": {
                        "imageSpec": {
                            "config": {
                                "Labels": {
                                    "org.opencontainers.image.revision": (
                                        metadata["revision"]
                                    )
                                }
                            }
                        }
                    },
                }
            )
        )
        evidence["environment"][key] = fleet_bench._cri_image_metadata(
            path,
            tmp_path,
        )
    replay = evaluate_gate(**evidence)
    sidecar_rows = {
        "requests": evidence["requests"],
        "placements": evidence["placements"],
        "churn": evidence["churn"],
        "endpointslices": evidence["endpoints"],
        "pods": evidence["pods"],
        "resources": evidence["resources"],
    }
    descriptors = {}
    for name, rows in sidecar_rows.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        descriptors[name] = _sidecar_descriptor(path, rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gate": "G5-F1a",
        "configuration": evidence["config"].__dict__,
        "schedule": churn_plan(evidence["config"]),
        "environment": evidence["environment"],
        "sidecars": descriptors,
        **{
            key: replay[key]
            for key in (
                "checks",
                "passed",
                "traffic",
                "placement",
                "churn",
                    "membership",
                    "runtime_mock_image",
                    "runtime_gateway_image",
                )
            },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    assert verify_artifact(manifest_path)["verified"]

    tampered_schedule = json.loads(json.dumps(manifest))
    tampered_schedule["schedule"][0]["ordinals"].reverse()
    manifest_path.write_text(json.dumps(tampered_schedule))
    schedule_result = verify_artifact(manifest_path)
    assert not schedule_result["verified"]
    assert not schedule_result["checks"][
        "published_schedule_matches_frozen_plan"
    ]

    tampered_p99 = json.loads(json.dumps(manifest))
    tampered_p99["placement"]["overall"]["p99_ms"] = 0.0
    manifest_path.write_text(json.dumps(tampered_p99))
    p99_result = verify_artifact(manifest_path)
    assert not p99_result["verified"]
    assert not p99_result["checks"]["published_placement_matches_replay"]

    manifest_path.write_text(json.dumps(manifest))
    gateway_cri_path = tmp_path / "gateway-cri-image.json"
    gateway_cri_raw = gateway_cri_path.read_text()
    gateway_cri_path.write_text("{}")
    cri_result = verify_artifact(manifest_path)
    assert not cri_result["verified"]
    assert not cri_result["checks"]["gateway_cri_image_metadata_raw"]
    gateway_cri_path.write_text(gateway_cri_raw)

    rendered_kind_config_raw = rendered_kind_config.read_text(encoding="utf-8")
    rendered_kind_config.write_text(
        rendered_kind_config_raw + "  # tampered\n",
        encoding="utf-8",
    )
    frozen_sidecar_result = verify_artifact(manifest_path)
    assert not frozen_sidecar_result["verified"]
    assert not frozen_sidecar_result["checks"]["frozen_sidecar:0:hash"]
    rendered_kind_config.write_text(
        rendered_kind_config_raw,
        encoding="utf-8",
    )

    with (tmp_path / "requests.jsonl").open("a") as handle:
        handle.write("{}\n")
    assert not verify_artifact(manifest_path)["verified"]
