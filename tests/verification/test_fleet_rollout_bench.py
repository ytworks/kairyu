import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import verification.fleet.resilience.fleet_rollout_bench as rollout_bench

_LIFECYCLE_IMAGES_VALID = rollout_bench._all_replica_runtime_images_valid
_NAMESPACE = "kairyu-f1a"
_STATEFULSET = "f1a-replica"
_MOCK_IMAGE = "kairyu-f1a-mock:dev"
_GATEWAY_IMAGE = "kairyu:dev"
_OLD_REVISION = "revision-old"
_NEW_REVISION = "revision-new"


def _container(name: str, image: str, digest: str) -> dict:
    return {
        "name": name,
        "spec_image": image,
        "runtime_image": image,
        "image_id": digest,
    }


def _replica_identity(
    ordinal: int,
    uid: str,
    revision: str,
    *,
    ready: bool = True,
) -> dict:
    return {
        "name": f"{_STATEFULSET}-{ordinal}",
        "uid": uid,
        "ready": ready,
        "phase": "Running",
        "restart_count": 0,
        "revision": revision,
        "containers": [_container("mock", _MOCK_IMAGE, "sha256:" + "1" * 64)],
    }


def _gateway_identity() -> dict:
    return {
        "name": "f1a-gateway-test",
        "uid": "gateway-uid",
        "ready": True,
        "phase": "Running",
        "restart_count": 0,
        "revision": None,
        "containers": [_container("gateway", _GATEWAY_IMAGE, "sha256:" + "2" * 64)],
    }


def _pod_row(
    capture: str,
    observed_ns: int,
    identities: list[dict],
    *,
    step: int | None = None,
    ordinal: int | None = None,
) -> dict:
    return {
        "schema_version": rollout_bench.SCHEMA_VERSION,
        "kind": "pods",
        "capture": capture,
        "step": step,
        "ordinal": ordinal,
        "observed_ns": observed_ns,
        "error": None,
        "payload": identities,
    }


def _endpoint_snapshot(uid_by_ordinal: dict[int, str]) -> dict:
    return {
        "items": [
            {
                "endpoints": [
                    {
                        "conditions": {
                            "ready": True,
                            "terminating": False,
                        },
                        "targetRef": {
                            "kind": "Pod",
                            "namespace": _NAMESPACE,
                            "name": f"{_STATEFULSET}-{ordinal}",
                            "uid": uid,
                        },
                    }
                    for ordinal, uid in sorted(uid_by_ordinal.items())
                ]
            }
        ]
    }


def _endpoint_row(
    capture: str,
    observed_ns: int,
    uid_by_ordinal: dict[int, str],
    *,
    step: int | None = None,
    ordinal: int | None = None,
) -> dict:
    return {
        "schema_version": rollout_bench.SCHEMA_VERSION,
        "kind": "endpointslices",
        "capture": capture,
        "step": step,
        "ordinal": ordinal,
        "observed_ns": observed_ns,
        "error": None,
        "payload": _endpoint_snapshot(uid_by_ordinal),
    }


def _membership_row(
    sequence: int,
    observed_ns: int,
    uid_by_ordinal: dict[int, str],
    *,
    reason: str,
    replica_id: str | None = None,
    ineligible: tuple[str, ...] = (),
) -> dict:
    replica_ids = [uid for _, uid in sorted(uid_by_ordinal.items())]
    eligible_ids = [uid for uid in replica_ids if uid not in ineligible]
    row = {
        "schema_version": rollout_bench.SCHEMA_VERSION,
        "kind": "membership",
        "observed_ns": observed_ns,
        "event_source_id": "synthetic-source",
        "sequence": sequence,
        "event_id": f"membership-{sequence}",
        "reason": reason,
        "replica_id": replica_id,
        "replica_ids": replica_ids,
        "healthy_ids": replica_ids,
        "eligible_ids": eligible_ids,
        "generation_by_id": {uid: f"generation-{uid}" for uid in replica_ids},
        "ineligible": list(ineligible),
    }
    if replica_id is not None:
        row["replica_generation"] = row["generation_by_id"][replica_id]
    return row


def _statefulset_state(
    *,
    partition: int,
    current_revision: str,
    update_revision: str,
    updated_replicas: int,
) -> dict:
    replicas = rollout_bench.SMOKE_CONFIG.replica_count
    return {
        "strategy": "RollingUpdate",
        "partition": partition,
        "replicas": replicas,
        "status_replicas": replicas,
        "ready_replicas": replicas,
        "current_replicas": replicas - updated_replicas,
        "updated_replicas": updated_replicas,
        "available_replicas": replicas,
        "current_revision": current_revision,
        "update_revision": update_revision,
        "generation": updated_replicas + 1,
        "observed_generation": updated_replicas + 1,
    }


def _gateway_metrics(
    uid_by_ordinal: dict[int, str],
    *,
    configured: int | None = None,
    draining: int = 0,
) -> dict:
    uids = [uid for _, uid in sorted(uid_by_ordinal.items())]
    return {
        "configured": len(uids) if configured is None else configured,
        "healthy": len(uids),
        "draining": draining,
        "outstanding_by_uid": {uid: 0 for uid in uids},
        "healthy_by_uid": {uid: True for uid in uids},
        "uids": uids,
        "parse_errors": 0,
    }


def _latest_membership(
    memberships: list[dict],
    selected_at_ns: int,
) -> dict:
    return max(
        (row for row in memberships if row["observed_ns"] <= selected_at_ns),
        key=lambda row: row["observed_ns"],
    )


def _passing_evidence() -> dict:
    config = rollout_bench.SMOKE_CONFIG
    old_uids = {ordinal: f"old-uid-{ordinal}" for ordinal in range(config.replica_count)}
    new_uids = {ordinal: f"new-uid-{ordinal}" for ordinal in range(config.replica_count)}
    current = dict(old_uids)

    traffic_started_ns = 1_000_000_000
    rollout_started_ns = 3_000_000_000
    rollout_completed_ns = 7_200_000_000
    traffic_stop_requested_ns = 9_200_000_001
    traffic_stopped_ns = 9_200_000_002

    rollout = [
        {
            "schema_version": rollout_bench.SCHEMA_VERSION,
            "kind": "initial",
            "observed_ns": 900_000_000,
            "traffic_started_ns": traffic_started_ns,
            "statefulset": _statefulset_state(
                partition=config.replica_count,
                current_revision=_OLD_REVISION,
                update_revision=_OLD_REVISION,
                updated_replicas=0,
            ),
            "gateway": _gateway_metrics(old_uids),
            "old_uid_by_ordinal": {str(ordinal): uid for ordinal, uid in sorted(old_uids.items())},
        },
        {
            "schema_version": rollout_bench.SCHEMA_VERSION,
            "kind": "restart",
            "started_ns": rollout_started_ns,
            "completed_ns": rollout_started_ns + 50_000_000,
            "observed_ns": rollout_started_ns + 60_000_000,
            "command": [
                "kubectl",
                "-n",
                _NAMESPACE,
                "rollout",
                "restart",
                f"statefulset/{_STATEFULSET}",
            ],
            "returncode": 0,
            "stdout": "restarted",
            "stderr": "",
            "initial_revision": _OLD_REVISION,
            "update_revision": _NEW_REVISION,
            "statefulset": _statefulset_state(
                partition=config.replica_count,
                current_revision=_OLD_REVISION,
                update_revision=_NEW_REVISION,
                updated_replicas=0,
            ),
        },
    ]
    pods = [
        _pod_row(
            "initial",
            900_000_000,
            [
                _replica_identity(ordinal, uid, _OLD_REVISION)
                for ordinal, uid in sorted(old_uids.items())
            ],
        )
    ]
    endpoints = [_endpoint_row("initial", 900_000_000, old_uids)]
    memberships = [
        _membership_row(
            1,
            900_000_000,
            current,
            reason="initial",
        )
    ]
    membership_sequence = 1

    for planned in rollout_bench.rollout_plan(config):
        step = planned["step"]
        ordinal = planned["ordinal"]
        old_uid = current[ordinal]
        drain_started_ns = 3_100_000_000 + step * 1_000_000_000
        drain_completed_ns = drain_started_ns + 100_000_000
        membership_withdrawn_ns = drain_started_ns + 200_000_000
        withdrawn_observed_ns = drain_started_ns + 300_000_000
        partition_started_ns = drain_started_ns + 400_000_000
        partition_completed_ns = drain_started_ns + 500_000_000
        replacement_observed_ns = drain_started_ns + 900_000_000

        membership_sequence += 1
        memberships.append(
            _membership_row(
                membership_sequence,
                membership_withdrawn_ns,
                current,
                reason="reconciler_drain",
                replica_id=old_uid,
                ineligible=(old_uid,),
            )
        )
        withdrawn_identities = [
            _replica_identity(
                current_ordinal,
                uid,
                (_NEW_REVISION if uid.startswith("new-") else _OLD_REVISION),
                ready=current_ordinal != ordinal,
            )
            for current_ordinal, uid in sorted(current.items())
        ]
        pods.append(
            _pod_row(
                "withdrawal_poll",
                withdrawn_observed_ns,
                withdrawn_identities,
                step=step,
                ordinal=ordinal,
            )
        )
        withdrawn_endpoints = {
            current_ordinal: uid
            for current_ordinal, uid in current.items()
            if current_ordinal != ordinal
        }
        endpoints.append(
            _endpoint_row(
                "withdrawal_poll",
                withdrawn_observed_ns,
                withdrawn_endpoints,
                step=step,
                ordinal=ordinal,
            )
        )

        new_uid = new_uids[ordinal]
        current[ordinal] = new_uid
        membership_sequence += 1
        memberships.append(
            _membership_row(
                membership_sequence,
                replacement_observed_ns,
                current,
                reason="replica_added",
                replica_id=new_uid,
            )
        )
        replacement_identities = [
            _replica_identity(
                current_ordinal,
                uid,
                (_NEW_REVISION if uid.startswith("new-") else _OLD_REVISION),
            )
            for current_ordinal, uid in sorted(current.items())
        ]
        pods.append(
            _pod_row(
                "replacement_poll",
                replacement_observed_ns,
                replacement_identities,
                step=step,
                ordinal=ordinal,
            )
        )
        endpoints.append(
            _endpoint_row(
                "replacement_poll",
                replacement_observed_ns,
                current,
                step=step,
                ordinal=ordinal,
            )
        )
        rollout.append(
            {
                "schema_version": rollout_bench.SCHEMA_VERSION,
                "kind": "step",
                "step": step,
                "ordinal": ordinal,
                "old_uid": old_uid,
                "new_uid": new_uid,
                "drain_started_ns": drain_started_ns,
                "drain_completed_ns": drain_completed_ns,
                "drain_status_code": 200,
                "drain_body_uid": old_uid,
                "drain_error": None,
                "withdrawal_snapshot_observed_ns": withdrawn_observed_ns,
                "withdrawn_observed_ns": withdrawn_observed_ns,
                "pod_ready_false": True,
                "endpoint_absent": True,
                "gateway_withdrawn": True,
                "gateway_at_withdrawal": _gateway_metrics(
                    withdrawn_endpoints,
                    configured=config.replica_count - 1,
                ),
                "partition_patch_started_ns": partition_started_ns,
                "partition_patch_completed_ns": partition_completed_ns,
                "partition": ordinal,
                "replacement_snapshot_observed_ns": (replacement_observed_ns),
                "replacement_observed_ns": replacement_observed_ns,
                "pod_replaced": True,
                "all_pods_ready": True,
                "endpoints_exact": True,
                "gateway_exact": True,
                "statefulset_advanced": True,
                "gateway_at_replacement": _gateway_metrics(current),
                "statefulset": _statefulset_state(
                    partition=ordinal,
                    current_revision=(_NEW_REVISION if ordinal == 0 else _OLD_REVISION),
                    update_revision=_NEW_REVISION,
                    updated_replicas=config.replica_count - ordinal,
                ),
            }
        )

    pods.extend(
        [
            _pod_row(
                "final",
                9_100_000_000,
                [
                    _replica_identity(ordinal, uid, _NEW_REVISION)
                    for ordinal, uid in sorted(new_uids.items())
                ],
            ),
            _pod_row("gateway", 9_100_000_001, [_gateway_identity()]),
        ]
    )
    endpoints.append(_endpoint_row("final", 9_100_000_000, new_uids))
    rollout.append(
        {
            "schema_version": rollout_bench.SCHEMA_VERSION,
            "kind": "final",
            "observed_ns": 9_100_000_000,
            "rollout_completed_ns": rollout_completed_ns,
            "traffic_stop_requested_ns": traffic_stop_requested_ns,
            "traffic_stopped_ns": traffic_stopped_ns,
            "rollout_status_command": [
                "kubectl",
                "-n",
                _NAMESPACE,
                "rollout",
                "status",
                f"statefulset/{_STATEFULSET}",
                "--timeout=30s",
            ],
            "rollout_status_returncode": 0,
            "rollout_status_stdout": "complete",
            "rollout_status_stderr": "",
            "statefulset": _statefulset_state(
                partition=0,
                current_revision=_NEW_REVISION,
                update_revision=_NEW_REVISION,
                updated_replicas=config.replica_count,
            ),
            "gateway": _gateway_metrics(new_uids),
            "new_uid_by_ordinal": {str(ordinal): uid for ordinal, uid in sorted(new_uids.items())},
        }
    )

    requests = []
    replica_placements = []
    first_request_ns = traffic_started_ns
    interval_ns = int(1_000_000_000 / config.request_rate)
    request_count = (
        traffic_stop_requested_ns - traffic_started_ns + interval_ns - 1
    ) // interval_ns
    for sequence in range(request_count):
        send_ns = first_request_ns + sequence * interval_ns
        selected_at_ns = send_ns + 1_000_000
        membership = _latest_membership(memberships, selected_at_ns)
        selected_uid = sorted(membership["eligible_ids"])[0]
        ordinal = int(selected_uid.rsplit("-", 1)[1])
        request_id = f"request-{sequence}"
        request_id_echo = f"gateway-request-{sequence}"
        requests.append(
            {
                "schema_version": rollout_bench.SCHEMA_VERSION,
                "sequence": sequence,
                "request_id": request_id,
                "scheduled_ns": send_ns,
                "send_ns": send_ns,
                "end_ns": selected_at_ns + 1_000_000,
                "pacing_error_ms": 0.0,
                "not_sent": False,
                "status_code": 200,
                "transport_error": None,
                "response_valid": True,
                "request_id_echo": request_id_echo,
                "pod_name": f"{_STATEFULSET}-{ordinal}",
                "pod_uid": selected_uid,
            }
        )
        replica_placements.append(
            {
                "schema_version": rollout_bench.SCHEMA_VERSION,
                "kind": "replica",
                "request_id": request_id_echo,
                "replica_id": selected_uid,
                "replica_generation": membership["generation_by_id"][selected_uid],
                "placement_started_ns": send_ns,
                "selected_at_ns": selected_at_ns,
            }
        )

    return {
        "config": config,
        "requests": requests,
        "placements": [*memberships, *replica_placements],
        "rollout": rollout,
        "pods": pods,
        "endpoints": endpoints,
        "environment": {
            "source_dirty": False,
            "tracked_dirty": False,
            "untracked_gate_input_dirty": False,
            "source_end_match": True,
            "frozen_sidecars": [
                {"path": "rendered-kind-config.yaml"},
                {"path": "rendered-manifest.yaml"},
            ],
            "frozen_inputs": [
                {"path": path}
                for path in (
                    "verification/fleet/resilience/fleet_churn_bench.py",
                    "verification/fleet/resilience/fleet_rollout_bench.py",
                    "scripts/kind_rollout_gate.sh",
                    ".github/workflows/f1b-rollout.yml",
                    "deploy/kind/f1a/kind-config.yaml",
                    "deploy/kind/f1a/base/replicas.yaml",
                    "deploy/kind/f1a/base/gateway.yaml",
                    "deploy/kind/f1a/mock/main.go",
                    "deploy/kind/f1b/overlays/formal/kustomization.yaml",
                    "deploy/kind/f1b/overlays/formal/replicas-patch.yaml",
                    "deploy/kind/f1b/overlays/smoke/kustomization.yaml",
                    "deploy/kind/f1b/overlays/smoke/replicas-patch.yaml",
                )
            ],
        },
    }


def _evaluate(evidence: dict) -> dict:
    return rollout_bench.evaluate_gate(
        config=evidence["config"],
        requests=evidence["requests"],
        placements=evidence["placements"],
        rollout=evidence["rollout"],
        pods=evidence["pods"],
        endpoints=evidence["endpoints"],
        environment=evidence["environment"],
        statefulset=_STATEFULSET,
        namespace=_NAMESPACE,
        mock_container="mock",
        mock_image=_MOCK_IMAGE,
        gateway_container="gateway",
        gateway_image=_GATEWAY_IMAGE,
        kubectl="kubectl",
    )


@pytest.fixture(autouse=True)
def _trust_external_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        rollout_bench,
        "_provenance_valid",
        lambda _environment: True,
    )
    monkeypatch.setattr(
        rollout_bench,
        "_runtime_images_valid",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        rollout_bench,
        "_all_replica_runtime_images_valid",
        lambda *_args, **_kwargs: True,
    )


@pytest.fixture
def passing_evidence() -> dict:
    return _passing_evidence()


def test_formal_and_smoke_configs_have_frozen_controller_plans() -> None:
    assert (
        rollout_bench.resolve_config(SimpleNamespace(profile="formal"))
        == rollout_bench.FORMAL_CONFIG
    )
    assert (
        rollout_bench.resolve_config(SimpleNamespace(profile="smoke")) == rollout_bench.SMOKE_CONFIG
    )

    formal_plan = rollout_bench.rollout_plan(rollout_bench.FORMAL_CONFIG)
    smoke_plan = rollout_bench.rollout_plan(rollout_bench.SMOKE_CONFIG)
    assert len(formal_plan) == 100
    assert formal_plan[0] == {"step": 0, "ordinal": 99, "partition": 99}
    assert formal_plan[-1] == {"step": 99, "ordinal": 0, "partition": 0}
    assert smoke_plan == [
        {"step": 0, "ordinal": 3, "partition": 3},
        {"step": 1, "ordinal": 2, "partition": 2},
        {"step": 2, "ordinal": 1, "partition": 1},
        {"step": 3, "ordinal": 0, "partition": 0},
    ]

    with pytest.raises(ValueError, match="formal setting replica_count"):
        rollout_bench.resolve_config(SimpleNamespace(profile="formal", replica_count=99))


def test_gateway_prometheus_parser_filters_pool_and_rejects_bad_labels() -> None:
    snapshot = rollout_bench._gateway_metrics_snapshot(
        "\n".join(
            (
                "# HELP ignored comment",
                'kairyu_pool_replicas{pool="f1a",state="configured"} 4',
                'kairyu_pool_replicas{pool="f1a",state="healthy"} 3',
                'kairyu_pool_replicas{pool="f1a",state="draining"} 1',
                ('kairyu_replica_outstanding{pool="f1a",replica="uid-a"} 2'),
                'kairyu_replica_healthy{pool="f1a",replica="uid-a"} 1',
                'kairyu_replica_healthy{pool="other",replica="foreign"} 1',
                ('kairyu_replica_healthy{pool="f1a",pool="duplicate",replica="bad"} 1'),
                'unrelated_metric{pool="f1a"} NaN',
            )
        ),
        pool="f1a",
    )

    assert snapshot == {
        "configured": 4,
        "healthy": 3,
        "draining": 1,
        "outstanding_by_uid": {"uid-a": 2},
        "healthy_by_uid": {"uid-a": True},
        "uids": ["uid-a"],
        "parse_errors": 1,
    }
    assert rollout_bench._parse_prometheus_labels('pool="f1a",replica="uid-a"') == {
        "pool": "f1a",
        "replica": "uid-a",
    }
    assert rollout_bench._parse_prometheus_labels('pool="f1a",pool="duplicate"') is None


def test_synthetic_raw_evidence_passes_full_replay(
    passing_evidence,
) -> None:
    result = _evaluate(passing_evidence)

    assert result["passed"], [name for name, passed in result["checks"].items() if not passed]
    assert all(result["checks"].values())
    assert result["traffic"] == {
        "requests": 83,
        "responses_2xx": 83,
        "http_429": 0,
        "http_5xx": 0,
        "other_http": 0,
        "transport_errors": 0,
        "unsent": 0,
        "pacing_success_fraction": 1.0,
    }
    assert result["rollout"]["replicas"] == 4
    assert result["rollout"]["steps"] == 4
    assert result["rollout"]["minimum_ready_pods_observed"] == 3
    assert result["rollout"]["minimum_ready_endpoints_observed"] == 3


def test_lifecycle_image_allows_only_eventually_pinned_pending_uid(
    passing_evidence,
) -> None:
    pods = copy.deepcopy(passing_evidence["pods"])
    pending = copy.deepcopy(next(row for row in pods if row["capture"] == "final")["payload"][0])
    pending["ready"] = False
    pending["phase"] = "Pending"
    pending["containers"][0]["image_id"] = ""
    pods.append(_pod_row("replacement_poll", 9_050_000_000, [pending]))
    environment = {
        **passing_evidence["environment"],
        "mock_cri_image_metadata": {
            "runtime_digests": ["sha256:" + "1" * 64],
        },
    }

    assert _LIFECYCLE_IMAGES_VALID(
        pods,
        environment=environment,
        container_name="mock",
        expected_image=_MOCK_IMAGE,
    )

    pending["containers"] = []
    assert _LIFECYCLE_IMAGES_VALID(
        pods,
        environment=environment,
        container_name="mock",
        expected_image=_MOCK_IMAGE,
    )

    pending["containers"] = [_container("mock", _MOCK_IMAGE, "sha256:" + "f" * 64)]
    assert not _LIFECYCLE_IMAGES_VALID(
        pods,
        environment=environment,
        container_name="mock",
        expected_image=_MOCK_IMAGE,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("drain_status_code", 503),
        ("drain_body_uid", "wrong-uid"),
    ),
)
def test_step_rejects_failed_or_wrong_identity_drain(
    passing_evidence,
    field,
    value,
) -> None:
    evidence = copy.deepcopy(passing_evidence)
    step = next(row for row in evidence["rollout"] if row.get("kind") == "step")
    step[field] = value

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["step_lifecycle_causal"]


def test_step_rejects_partition_before_withdrawal(
    passing_evidence,
) -> None:
    evidence = copy.deepcopy(passing_evidence)
    step = next(row for row in evidence["rollout"] if row.get("kind") == "step")
    step["partition_patch_started_ns"] = step["withdrawn_observed_ns"] - 1

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["step_lifecycle_causal"]


def test_replay_rejects_old_uid_placement_after_withdrawal_and_partition(
    passing_evidence,
) -> None:
    evidence = copy.deepcopy(passing_evidence)
    step = next(row for row in evidence["rollout"] if row.get("kind") == "step")
    old_uid = step["old_uid"]
    placement = next(
        row
        for row in evidence["placements"]
        if row.get("kind") == "replica"
        and row["selected_at_ns"] >= step["partition_patch_started_ns"]
        and row["selected_at_ns"] < step["replacement_observed_ns"]
    )
    placement["replica_id"] = old_uid
    placement["replica_generation"] = f"generation-{old_uid}"
    request = next(
        row for row in evidence["requests"] if row["request_id_echo"] == placement["request_id"]
    )
    request["pod_uid"] = old_uid
    request["pod_name"] = f"{_STATEFULSET}-{step['ordinal']}"

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["no_old_placement_after_gateway_withdrawal"]
    assert not result["checks"]["no_old_placement_after_partition_patch"]
    assert not result["checks"]["placement_membership_temporal_join_complete"]


def test_rollout_plan_rejects_missing_ordinal(passing_evidence) -> None:
    evidence = copy.deepcopy(passing_evidence)
    step_index = next(
        index for index, row in enumerate(evidence["rollout"]) if row.get("kind") == "step"
    )
    evidence["rollout"].pop(step_index)

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["rollout_rows_complete"]
    assert not result["checks"]["step_order_exact"]


def test_rollout_plan_rejects_duplicate_ordinal(passing_evidence) -> None:
    evidence = copy.deepcopy(passing_evidence)
    steps = [row for row in evidence["rollout"] if row.get("kind") == "step"]
    steps[1]["ordinal"] = steps[0]["ordinal"]
    steps[1]["partition"] = steps[0]["partition"]

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["step_order_exact"]
    assert not result["checks"]["step_lifecycle_causal"]


def test_replay_rejects_unchanged_pod_uid(passing_evidence) -> None:
    evidence = copy.deepcopy(passing_evidence)
    first_step = next(row for row in evidence["rollout"] if row.get("kind") == "step")
    final_pods = next(row for row in evidence["pods"] if row.get("capture") == "final")
    identity = next(
        item
        for item in final_pods["payload"]
        if item["name"] == f"{_STATEFULSET}-{first_step['ordinal']}"
    )
    identity["uid"] = first_step["old_uid"]

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["step_lifecycle_causal"]
    assert not result["checks"]["final_exact_replacement_fleet"]


def test_replay_requires_raw_step_snapshots(passing_evidence) -> None:
    evidence = copy.deepcopy(passing_evidence)
    first_step = next(row for row in evidence["rollout"] if row.get("kind") == "step")
    evidence["pods"] = [row for row in evidence["pods"] if row.get("step") != first_step["step"]]
    evidence["endpoints"] = [
        row for row in evidence["endpoints"] if row.get("step") != first_step["step"]
    ]

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["step_lifecycle_causal"]


def test_replay_rejects_deleted_request_suffix(passing_evidence) -> None:
    evidence = copy.deepcopy(passing_evidence)
    removed = evidence["requests"].pop()
    evidence["placements"] = [
        row for row in evidence["placements"] if row.get("request_id") != removed["request_id"]
    ]

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["traffic_sequence_and_schedule_exact"]


def test_replay_derives_pacing_and_response_pod_name(
    passing_evidence,
) -> None:
    evidence = copy.deepcopy(passing_evidence)
    evidence["requests"][0]["pacing_error_ms"] = -123.0
    evidence["requests"][1]["pod_name"] = "wrong-pod-name"

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["traffic_sequence_and_schedule_exact"]
    assert not result["checks"]["traffic_retry_free_zero_failures"]


def test_replay_rejects_extra_uid_or_container_restart(
    passing_evidence,
) -> None:
    evidence = copy.deepcopy(passing_evidence)
    replacement = next(row for row in evidence["pods"] if row.get("capture") == "replacement_poll")
    replacement["payload"][0]["uid"] = "third-unplanned-uid"
    replacement["payload"][1]["restart_count"] = 1

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["pod_uid_lineage_exactly_one_replacement_each"]
    assert not result["checks"]["replica_container_restart_count_zero"]


@pytest.mark.parametrize(
    ("mutation", "summary_field"),
    (
        (
            {
                "status_code": 500,
                "response_valid": False,
            },
            "http_5xx",
        ),
        (
            {
                "status_code": None,
                "transport_error": "ReadTimeout",
                "response_valid": False,
            },
            "transport_errors",
        ),
        (
            {
                "not_sent": True,
                "status_code": None,
                "transport_error": "max_concurrency",
                "response_valid": False,
            },
            "unsent",
        ),
    ),
)
def test_retry_free_traffic_rejects_every_failure_class(
    passing_evidence,
    mutation,
    summary_field,
) -> None:
    evidence = copy.deepcopy(passing_evidence)
    evidence["requests"][0].update(mutation)

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["traffic_retry_free_zero_failures"]
    assert result["traffic"][summary_field] == 1


def test_replay_rejects_final_revision_mismatch(passing_evidence) -> None:
    evidence = copy.deepcopy(passing_evidence)
    final = next(row for row in evidence["rollout"] if row.get("kind") == "final")
    final["statefulset"]["current_revision"] = "unexpected-revision"

    result = _evaluate(evidence)

    assert not result["passed"]
    assert not result["checks"]["final_statefulset_revision_complete"]


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_artifact_verifier_detects_sidecar_hash_tamper(
    passing_evidence,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        rollout_bench,
        "_containerd_metadata_file_matches",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        rollout_bench,
        "_cri_metadata_file_matches",
        lambda *_args, **_kwargs: True,
    )
    frozen = tmp_path / "rendered-kind-config.yaml"
    frozen.write_text("kind: Cluster\n", encoding="utf-8")
    rendered_manifest = tmp_path / "rendered-manifest.yaml"
    rendered_manifest.write_text("kind: List\n", encoding="utf-8")
    environment = {
        **passing_evidence["environment"],
        "frozen_sidecars": [
            {
                "path": frozen.name,
                "sha256": rollout_bench._sha256_file(frozen),
            },
            {
                "path": rendered_manifest.name,
                "sha256": rollout_bench._sha256_file(rendered_manifest),
            },
        ],
    }
    evidence = copy.deepcopy(passing_evidence)
    evidence["environment"] = environment
    replay = _evaluate(evidence)
    assert replay["passed"], [name for name, passed in replay["checks"].items() if not passed]

    rows_by_sidecar = {
        "requests": evidence["requests"],
        "placements": evidence["placements"],
        "rollout": evidence["rollout"],
        "pods": evidence["pods"],
        "endpointslices": evidence["endpoints"],
    }
    sidecars = {}
    paths = {}
    for name, rows in rows_by_sidecar.items():
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, rows)
        paths[name] = path
        sidecars[name] = rollout_bench._sidecar_descriptor(path, rows)

    manifest = {
        "schema_version": rollout_bench.SCHEMA_VERSION,
        "gate": rollout_bench.GATE,
        "assumptions": {
            "statefulset": _STATEFULSET,
            "namespace": _NAMESPACE,
            "mock_container": "mock",
            "mock_image": _MOCK_IMAGE,
            "gateway_container": "gateway",
            "gateway_image": _GATEWAY_IMAGE,
            "kubectl": "kubectl",
        },
        "configuration": asdict(rollout_bench.SMOKE_CONFIG),
        "schedule": rollout_bench.rollout_plan(rollout_bench.SMOKE_CONFIG),
        "environment": environment,
        "sidecars": sidecars,
        **replay,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verified = rollout_bench.verify_artifact(manifest_path)
    assert verified["verified"]

    paths["requests"].write_text(
        paths["requests"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    tampered = rollout_bench.verify_artifact(manifest_path)

    assert not tampered["verified"]
    assert not tampered["checks"]["sidecar:requests:hash"]
    assert tampered["checks"]["sidecar:requests:rows"]
    assert tampered["checks"]["replayed_gate_passed"]
