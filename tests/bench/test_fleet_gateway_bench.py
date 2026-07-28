import hashlib
import json
import runpy
from pathlib import Path

import bench.fleet_gateway_bench as gateway_bench


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_cri_metadata(
    path: Path,
    *,
    image: str,
    status_digest: str,
    repo_digest: str,
    revision: str | None,
) -> None:
    labels = (
        {"org.opencontainers.image.revision": revision}
        if revision is not None
        else {}
    )
    path.write_text(
        json.dumps(
            {
                "status": {
                    "id": status_digest,
                    "repoDigests": [f"{image.split('@', 1)[0]}@{repo_digest}"],
                    "repoTags": [image.split("@", 1)[0]],
                },
                "info": {"imageSpec": {"config": {"Labels": labels}}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_docker_metadata(
    path: Path,
    *,
    image: str,
    config_id: str,
) -> None:
    display_image, separator, digest = image.rpartition("@")
    assert separator
    repository = display_image.rpartition(":")[0]
    path.write_text(
        json.dumps(
            [
                {
                    "Id": config_id,
                    "RepoDigests": [f"{repository}@{digest}"],
                    "RepoTags": [display_image],
                }
            ],
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _affinity_row(
    *,
    phase: str,
    sequence: int,
    session: str,
    gateway_id: str,
    replica_uid: str,
) -> dict:
    pod_name = f"replica-{replica_uid}"
    started_unix_ns = 1_000_000 + sequence * 100
    request_json = {
        "model": "f1a",
        "messages": [
            {
                "role": "user",
                "content": f"F1c affinity {phase} {sequence}",
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
    }
    request_body = gateway_bench._canonical_json(request_json).encode()
    return {
        "schema_version": gateway_bench.SCHEMA_VERSION,
        "kind": "affinity",
        "operation": f"affinity_{phase}",
        "phase": phase,
        "sequence": sequence,
        "request_id": f"request-{sequence}",
        "lb_request_id": f"lb-{sequence}",
        "gateway_request_id": f"gateway-request-{sequence}",
        "session": session,
        "session_sha256": hashlib.sha256(session.encode()).hexdigest(),
        "status_code": 200,
        "transport_error": None,
        "gateway_id": gateway_id,
        "method": "POST",
        "path": "/v1/chat/completions",
        "started_unix_ns": started_unix_ns,
        "ended_unix_ns": started_unix_ns + 10,
        "request_body_bytes": len(request_body),
        "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
        "request_json": request_json,
        "response_json": {
            "choices": [
                {
                    "message": {
                        "content": (
                            "kairyu-f1a-mock "
                            f"pod_name={pod_name} pod_uid={replica_uid}"
                        )
                    }
                }
            ]
        },
        "pod_name": pod_name,
        "pod_uid": replica_uid,
        "response_valid": True,
    }


def _http_row(
    operation: str,
    gateway_id: str,
    sequence: int,
    *,
    method: str,
    path: str,
    request_json: dict | None = None,
    response_json: dict | None = None,
) -> dict:
    session = gateway_bench.session_for_gateway(gateway_id)
    started_unix_ns = 2_000_000 + sequence * 100
    if request_json is not None:
        request_body = gateway_bench._canonical_json(request_json).encode()
    elif method == "GET":
        request_body = b""
    else:
        request_body = f"multipart-body-{sequence}".encode()
    return {
        "schema_version": gateway_bench.SCHEMA_VERSION,
        "kind": "http",
        "operation": operation,
        "request_id": f"batch-http-{sequence}",
        "lb_request_id": f"batch-lb-{sequence}",
        "gateway_request_id": f"batch-gateway-{sequence}",
        "session": session,
        "session_sha256": hashlib.sha256(session.encode()).hexdigest(),
        "status_code": 200,
        "transport_error": None,
        "gateway_id": gateway_id,
        "method": method,
        "path": path,
        "request_json": request_json,
        "response_json": response_json,
        "started_unix_ns": started_unix_ns,
        "ended_unix_ns": started_unix_ns + 10,
        "request_body_bytes": len(request_body),
        "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
    }


def _refresh_artifact(manifest_path: Path, artifact: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = manifest_path.parent / artifact
    metadata = {
        "sha256": gateway_bench._sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(gateway_bench._read_jsonl(path)),
    }
    manifest["artifacts"][artifact] = metadata
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _passing_evidence(tmp_path: Path) -> Path:
    replica_uids = tuple(
        f"replica-uid-{index:02d}"
        for index in range(gateway_bench.DEFAULT_REPLICA_COUNT)
    )
    old_gateway_uids = {
        "a": "gateway-a-old",
        "b": "gateway-b-stable",
        "c": "gateway-c-stable",
    }
    new_gateway_uids = {
        **old_gateway_uids,
        "a": "gateway-a-new",
    }

    gateway_rows = []
    for evidence_phase, uids in (
        ("initial", old_gateway_uids),
        ("final", new_gateway_uids),
    ):
        for gateway_id in gateway_bench.GATEWAY_IDS:
            gateway_rows.append(
                {
                    "schema_version": gateway_bench.SCHEMA_VERSION,
                    "kind": "gateway_pod",
                    "evidence_phase": evidence_phase,
                    "gateway_id": gateway_id,
                    "name": f"f1c-gateway-{gateway_id}-pod",
                    "uid": uids[gateway_id],
                    "ready": True,
                    "pod_phase": "Running",
                    "images": [
                        {
                            "name": "gateway",
                            "image": "kairyu:dev",
                            "image_id": "docker://sha256:" + "1" * 64,
                        }
                    ],
                }
            )
    _write_jsonl(tmp_path / "gateway-pods.jsonl", gateway_rows)

    replica_rows = []
    for evidence_phase in ("initial", "final"):
        for index, uid in enumerate(replica_uids):
            replica_rows.append(
                {
                    "schema_version": gateway_bench.SCHEMA_VERSION,
                    "kind": "replica_pod",
                    "evidence_phase": evidence_phase,
                    "name": f"f1c-replica-{index}",
                    "uid": uid,
                    "ready": True,
                    "pod_phase": "Running",
                    "images": [
                        {
                            "name": "mock",
                            "image": "kairyu-f1a-mock:dev",
                            "image_id": "docker://sha256:" + "2" * 64,
                        }
                    ],
                }
            )
    _write_jsonl(tmp_path / "replica-pods.jsonl", replica_rows)
    postgres_image = "postgres:17.6-bookworm@sha256:" + "3" * 64
    _write_jsonl(
        tmp_path / "postgres-pods.jsonl",
        [
            {
                "schema_version": gateway_bench.SCHEMA_VERSION,
                "kind": "postgres_pod",
                "evidence_phase": phase,
                "name": "f1c-postgres-pod",
                "uid": "postgres-uid-stable",
                "ready": True,
                "pod_phase": "Running",
                "images": [
                    {
                        "name": "postgres",
                        # Kubernetes status.image keeps the display tag while
                        # status.imageID carries the digest-pinned identity.
                        "image": postgres_image.rpartition("@")[0],
                        "image_id": "docker-pullable://postgres@sha256:"
                        + "3" * 64,
                    }
                ],
            }
            for phase in ("initial", "final")
        ],
    )

    store_rows = []
    for phase, uids in ((None, old_gateway_uids), ("final", new_gateway_uids)):
        for gateway_id in gateway_bench.GATEWAY_IDS:
            row = {
                "schema_version": gateway_bench.SCHEMA_VERSION,
                "kind": "store_identity",
                "gateway_id": gateway_id,
                "gateway_uid": uids[gateway_id],
                "dsn_sha256": "d" * 64,
                "postgres_system_identifier": "765432109876543210",
            }
            if phase is not None:
                row["phase"] = phase
            store_rows.append(row)
    _write_jsonl(tmp_path / "store-identities.jsonl", store_rows)

    sessions = {
        gateway_id: [
            gateway_bench.session_for_gateway(gateway_id, ordinal=ordinal)
            for ordinal in range(gateway_bench.SESSIONS_PER_GATEWAY)
        ]
        for gateway_id in gateway_bench.GATEWAY_IDS
    }
    traffic = []
    sequence = 0
    for _round in range(gateway_bench.BASELINE_ROUNDS):
        for gateway_id in gateway_bench.GATEWAY_IDS:
            for session in sessions[gateway_id]:
                traffic.append(
                    _affinity_row(
                        phase="baseline",
                        sequence=sequence,
                        session=session,
                        gateway_id=gateway_id,
                        replica_uid=gateway_bench.replica_winner(
                            session, replica_uids
                        ),
                    )
                )
                sequence += 1

    failover_session = sessions["a"][1]
    surviving_gateway = gateway_bench.rendezvous_order(
        failover_session, ("b", "c")
    )[0]
    failover_replica = gateway_bench.replica_winner(
        failover_session, replica_uids
    )
    traffic.append(
        _affinity_row(
            phase="failover",
            sequence=sequence,
            session=failover_session,
            gateway_id=surviving_gateway,
            replica_uid=failover_replica,
        )
    )
    sequence += 1
    traffic.append(
        _affinity_row(
            phase="recovered",
            sequence=sequence,
            session=failover_session,
            gateway_id="a",
            replica_uid=failover_replica,
        )
    )
    _write_jsonl(tmp_path / "traffic.jsonl", traffic)

    placements = {gateway_id: [] for gateway_id in gateway_bench.GATEWAY_IDS}
    lb_rows = []
    for row in traffic:
        placements[row["gateway_id"]].append(
            {
                "kind": "replica",
                "session_sha256": row["session_sha256"],
                "replica_id": row["pod_uid"],
                "reason": "session_affinity",
                "request_id": row["gateway_request_id"],
            }
        )
        attempts = []
        if row["phase"] == "failover":
            attempts.append(
                {
                    "gateway_id": "a",
                    "error_type": "ConnectionRefusedError",
                    "outcome": "transport_error",
                    "status": None,
                    "upstream_request_id": None,
                }
            )
        attempts.append(
            {
                "gateway_id": row["gateway_id"],
                "outcome": "response",
                "status": 200,
                "upstream_request_id": row["gateway_request_id"],
            }
        )
        lb_rows.append(
            {
                "kind": "f1c_lb_decision",
                "request_id": row["lb_request_id"],
                "session_sha256": row["session_sha256"],
                "selection_reason": "session_hrw",
                "candidate_gateway_ids": list(
                    gateway_bench.rendezvous_order(
                        row["session"], gateway_bench.GATEWAY_IDS
                    )
                ),
                "attempts": attempts,
                "selected_gateway_id": row["gateway_id"],
                "final_status": 200,
                "method": row["method"],
                "path": row["path"],
                "started_ns": row["started_unix_ns"] + 1,
                "finished_ns": row["ended_unix_ns"] - 1,
                "body_bytes": row["request_body_bytes"],
                "body_sha256": row["request_body_sha256"],
            }
        )
    for gateway_id, rows in placements.items():
        rows.append(
            {
                "kind": "membership",
                "observed_ns": 900_000 + ord(gateway_id),
                "replica_ids": list(replica_uids),
                "healthy_ids": list(replica_uids),
                "eligible_ids": list(replica_uids),
            }
        )
        _write_jsonl(tmp_path / f"placements-{gateway_id}.jsonl", rows)
    batch_id = "batch-test"
    input_file_id = "file-input"
    output_file_id = "file-output"
    request_counts = {
        "total": gateway_bench.DEFAULT_BATCH_LINES,
        "completed": gateway_bench.DEFAULT_BATCH_LINES,
        "failed": 0,
    }
    batch_http = [
        _http_row(
            "upload_input_a",
            "a",
            0,
            method="POST",
            path="/v1/files",
            response_json={"id": input_file_id, "purpose": "batch"},
        ),
        _http_row(
            "read_input_b",
            "b",
            1,
            method="GET",
            path=f"/v1/files/{input_file_id}",
            response_json={"id": input_file_id, "purpose": "batch"},
        ),
        _http_row(
            "create_batch_b",
            "b",
            2,
            method="POST",
            path="/v1/batches",
            request_json={
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            response_json={"id": batch_id, "input_file_id": input_file_id},
        ),
        _http_row(
            "poll_batch_c",
            "c",
            3,
            method="GET",
            path=f"/v1/batches/{batch_id}",
            response_json={
                "id": batch_id,
                "status": "in_progress",
                "output_file_id": None,
                "request_counts": {
                    "total": gateway_bench.DEFAULT_BATCH_LINES,
                    "completed": 0,
                    "failed": 0,
                },
            },
        ),
        _http_row(
            "poll_batch_c",
            "c",
            4,
            method="GET",
            path=f"/v1/batches/{batch_id}",
            response_json={
                "id": batch_id,
                "status": "completed",
                "output_file_id": output_file_id,
                "request_counts": request_counts,
            },
        ),
        *[
            _http_row(
                f"read_output_{gateway_id}",
                gateway_id,
                5 + index,
                method="GET",
                path=f"/v1/files/{output_file_id}/content",
            )
            for index, gateway_id in enumerate(gateway_bench.GATEWAY_IDS)
        ],
    ]
    for row in batch_http:
        lb_rows.append(
            {
                "kind": "f1c_lb_decision",
                "request_id": row["lb_request_id"],
                "session_sha256": row["session_sha256"],
                "selection_reason": "session_hrw",
                "candidate_gateway_ids": list(
                    gateway_bench.rendezvous_order(
                        row["session"], gateway_bench.GATEWAY_IDS
                    )
                ),
                "attempts": [
                    {
                        "gateway_id": row["gateway_id"],
                        "outcome": "response",
                        "status": 200,
                        "upstream_request_id": row["gateway_request_id"],
                    }
                ],
                "selected_gateway_id": row["gateway_id"],
                "final_status": 200,
                "method": row["method"],
                "path": row["path"],
                "started_ns": row["started_unix_ns"] + 1,
                "finished_ns": row["ended_unix_ns"] - 1,
                "body_bytes": row["request_body_bytes"],
                "body_sha256": row["request_body_sha256"],
            }
        )
    _write_jsonl(tmp_path / "lb-decisions.jsonl", lb_rows)
    _write_jsonl(tmp_path / "batch-http.jsonl", batch_http)

    old_claim = {
        "sequence": 1,
        "batch_id": batch_id,
        "event": "claimed",
        "worker_id": old_gateway_uids["a"],
        "fencing_token": 1,
        "at_unix_ns": 10,
        "lease_until_unix_ns": 18,
        "previous_lease_until_unix_ns": None,
    }
    new_claim = {
        "sequence": 3,
        "batch_id": batch_id,
        "event": "reclaimed",
        "worker_id": old_gateway_uids["b"],
        "fencing_token": 2,
        "at_unix_ns": 20,
        "lease_until_unix_ns": 30,
        "previous_lease_until_unix_ns": 18,
    }
    lifecycle = [
        {
            "schema_version": gateway_bench.SCHEMA_VERSION,
            "kind": "created",
            "batch_id": batch_id,
            "input_file_id": input_file_id,
            "input_sha256": gateway_bench._sha256_bytes(
                gateway_bench._batch_payload(gateway_bench.DEFAULT_BATCH_LINES)
            ),
            "line_count": gateway_bench.DEFAULT_BATCH_LINES,
        },
        {
            "schema_version": gateway_bench.SCHEMA_VERSION,
            "kind": "owner_claim",
            "gateway_id": "a",
            **old_claim,
        },
        {
            "schema_version": gateway_bench.SCHEMA_VERSION,
            "kind": "owner_kill_requested",
            "gateway_id": "a",
            "deployment": "f1c-gateway-a",
            "batch_id": batch_id,
            "worker_id": old_claim["worker_id"],
            "fencing_token": old_claim["fencing_token"],
            "at_unix_ns": 12,
            "clock": "postgres_clock_timestamp",
        },
        {
            "schema_version": gateway_bench.SCHEMA_VERSION,
            "kind": "owner_killed",
            "gateway_id": "a",
            "deployment": "f1c-gateway-a",
            "batch_id": batch_id,
            "worker_id": old_claim["worker_id"],
            "fencing_token": old_claim["fencing_token"],
            "at_unix_ns": 13,
            "clock": "postgres_clock_timestamp",
        },
        {
            "schema_version": gateway_bench.SCHEMA_VERSION,
            "kind": "reclaimed",
            "gateway_id": "b",
            **new_claim,
        },
        {
            "schema_version": gateway_bench.SCHEMA_VERSION,
            "kind": "completed",
            "batch_id": batch_id,
            "status": "completed",
            "output_file_id": output_file_id,
            "error_file_id": None,
            "errors": None,
            "request_counts": request_counts,
            "at_unix_ns": 26,
            "clock": "postgres_clock_timestamp",
        },
    ]
    _write_jsonl(tmp_path / "batch-lifecycle.jsonl", lifecycle)
    _write_jsonl(
        tmp_path / "batch-claims.jsonl",
        [
            {
                "schema_version": gateway_bench.SCHEMA_VERSION,
                **old_claim,
            },
            {
                "schema_version": gateway_bench.SCHEMA_VERSION,
                "sequence": 2,
                "batch_id": batch_id,
                "event": "renewed",
                "worker_id": old_claim["worker_id"],
                "fencing_token": old_claim["fencing_token"],
                "at_unix_ns": 11,
                "lease_until_unix_ns": 18,
                "previous_lease_until_unix_ns": None,
            },
            {
                "schema_version": gateway_bench.SCHEMA_VERSION,
                **new_claim,
            },
            {
                "schema_version": gateway_bench.SCHEMA_VERSION,
                "sequence": 4,
                "batch_id": batch_id,
                "event": "renewed",
                "worker_id": new_claim["worker_id"],
                "fencing_token": new_claim["fencing_token"],
                "at_unix_ns": 21,
                "lease_until_unix_ns": 30,
                "previous_lease_until_unix_ns": None,
            },
            {
                "schema_version": gateway_bench.SCHEMA_VERSION,
                "sequence": 5,
                "event": "terminal_committed",
                "batch_id": batch_id,
                "worker_id": new_claim["worker_id"],
                "fencing_token": new_claim["fencing_token"],
                "at_unix_ns": 25,
                "lease_until_unix_ns": None,
                "previous_lease_until_unix_ns": None,
            },
        ],
    )

    output_rows = [
        {
            "id": f"batch-result-{index}",
            "custom_id": f"f1c-{index:06d}",
            "response": {"status_code": 200, "body": {"index": index}},
            "error": None,
        }
        for index in range(gateway_bench.DEFAULT_BATCH_LINES)
    ]
    for gateway_id in gateway_bench.GATEWAY_IDS:
        _write_jsonl(tmp_path / f"batch-output-{gateway_id}.jsonl", output_rows)
    for row in batch_http:
        if row["operation"].startswith("read_output_"):
            gateway_id = row["operation"].removeprefix("read_output_")
            row["content_sha256"] = gateway_bench._sha256_file(
                tmp_path / f"batch-output-{gateway_id}.jsonl"
            )
    _write_jsonl(tmp_path / "batch-http.jsonl", batch_http)

    artifacts = {}
    for name in gateway_bench.RAW_ARTIFACTS:
        path = tmp_path / name
        artifacts[name] = {
            "sha256": gateway_bench._sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(gateway_bench._read_jsonl(path)),
        }
    manifest = {
        "schema_version": gateway_bench.SCHEMA_VERSION,
        "gate": "g5-f1c-three-gateway-shared-batchstore",
        "passed": False,
        "config": {
            "namespace": gateway_bench.DEFAULT_NAMESPACE,
            "gateway_ids": list(gateway_bench.GATEWAY_IDS),
            "replica_count": gateway_bench.DEFAULT_REPLICA_COUNT,
            "batch_lines": gateway_bench.DEFAULT_BATCH_LINES,
            "baseline_rounds": gateway_bench.BASELINE_ROUNDS,
            "sessions_per_gateway": gateway_bench.SESSIONS_PER_GATEWAY,
            "timeout_seconds": gateway_bench.DEFAULT_TIMEOUT_SECONDS,
            "load_balancer_hash": "sha256-hrw-x-session-id",
            "batch_delivery": (
                "at-least-once execution with one fenced terminal publication"
            ),
        },
        "source": {
            "commit": "a" * 40,
            "tracked_clean": True,
            "gateway_image_digest": "sha256:" + "1" * 64,
            "mock_image_digest": "sha256:" + "2" * 64,
            "postgres_image": postgres_image,
        },
        "batch_id": batch_id,
        "artifacts": artifacts,
        "checks": {},
    }
    _write_cri_metadata(
        tmp_path / "gateway-cri-image.json",
        image="kairyu:dev",
        status_digest=manifest["source"]["gateway_image_digest"],
        repo_digest="sha256:" + "5" * 64,
        revision=manifest["source"]["commit"],
    )
    _write_cri_metadata(
        tmp_path / "mock-cri-image.json",
        image="kairyu-f1a-mock:dev",
        status_digest=manifest["source"]["mock_image_digest"],
        repo_digest="sha256:" + "6" * 64,
        revision=manifest["source"]["commit"],
    )
    _write_cri_metadata(
        tmp_path / "postgres-cri-image.json",
        image=postgres_image,
        status_digest="sha256:" + "4" * 64,
        repo_digest="sha256:" + "7" * 64,
        revision=None,
    )
    _write_docker_metadata(
        tmp_path / "postgres-docker-image.json",
        image=postgres_image,
        config_id="sha256:" + "4" * 64,
    )
    rendered_manifest_path = tmp_path / "rendered-manifest.yaml"
    rendered_manifest_path.write_text(
        "apiVersion: v1\nkind: List\nitems: []\n",
        encoding="utf-8",
    )
    manifest["source"].update(
        {
            "gateway_cri_image_metadata": gateway_bench._cri_image_metadata(
                tmp_path / "gateway-cri-image.json", tmp_path
            ),
            "mock_cri_image_metadata": gateway_bench._cri_image_metadata(
                tmp_path / "mock-cri-image.json", tmp_path
            ),
            "postgres_cri_image_metadata": gateway_bench._cri_image_metadata(
                tmp_path / "postgres-cri-image.json", tmp_path
            ),
            "postgres_docker_image_metadata": gateway_bench._docker_image_metadata(
                tmp_path / "postgres-docker-image.json", tmp_path
            ),
            "rendered_manifest": gateway_bench._adjacent_file_metadata(
                rendered_manifest_path,
                tmp_path,
            ),
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_rendezvous_order_is_stable_and_sessions_cover_every_gateway() -> None:
    candidates = gateway_bench.GATEWAY_IDS
    sessions = {
        gateway_id: gateway_bench.session_for_gateway(gateway_id)
        for gateway_id in candidates
    }
    assert {
        gateway_bench.rendezvous_order(session, candidates)[0]
        for session in sessions.values()
    } == set(candidates)
    for session in sessions.values():
        expected = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    hashlib.sha256(f"{session}:{candidate}".encode()).digest(),
                    candidate,
                ),
                reverse=True,
            )
        )
        assert gateway_bench.rendezvous_order(session, candidates) == expected


def test_disposable_lb_uses_the_same_hrw_order_as_the_verifier() -> None:
    lb = runpy.run_path("deploy/kind/f1c/lb.py")
    gateways = lb["_parse_gateways"](
        "a=http://a.test:8000,b=http://b.test:8000,c=http://c.test:8000"
    )
    for gateway_id in gateway_bench.GATEWAY_IDS:
        session = gateway_bench.session_for_gateway(gateway_id, ordinal=3)
        actual = tuple(
            gateway.gateway_id
            for gateway in lb["_rank_gateways"](session, gateways)
        )
        assert actual == gateway_bench.rendezvous_order(
            session, gateway_bench.GATEWAY_IDS
        )


def test_offline_verifier_reconstructs_the_complete_gate(tmp_path: Path) -> None:
    _passing_evidence(tmp_path)
    result = gateway_bench.verify_evidence(tmp_path)
    assert result["passed"]
    assert all(result["checks"].values())
    assert result["summary"]["affinity_requests"] == 14
    assert result["summary"]["batch_lines"] == gateway_bench.DEFAULT_BATCH_LINES


def test_verifier_accepts_delayed_owner_absence_observation(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    lifecycle = gateway_bench._read_jsonl(tmp_path / "batch-lifecycle.jsonl")
    owner_killed = next(row for row in lifecycle if row["kind"] == "owner_killed")
    reclaimed = next(row for row in lifecycle if row["kind"] == "reclaimed")
    owner_killed["at_unix_ns"] = reclaimed["at_unix_ns"] + 1
    _write_jsonl(tmp_path / "batch-lifecycle.jsonl", lifecycle)
    _refresh_artifact(manifest_path, "batch-lifecycle.jsonl")

    result = gateway_bench.verify_evidence(tmp_path)

    assert result["passed"]
    assert result["checks"][
        "durable_claim_audit_has_one_new-fence_terminal_commit"
    ]


def test_verifier_rejects_a_postgres_runtime_digest_different_from_the_pin(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "postgres-pods.jsonl")
    for row in rows:
        row["images"][0]["image_id"] = "docker-pullable://postgres@sha256:" + "9" * 64
    _write_jsonl(tmp_path / "postgres-pods.jsonl", rows)
    _refresh_artifact(manifest_path, "postgres-pods.jsonl")

    result = gateway_bench.verify_evidence(tmp_path)

    assert not result["passed"]
    assert not result["checks"][
        "runtime_gateway_mock_and_postgres_images_match_sources"
    ]


def test_verifier_accepts_a_postgres_kind_import_digest_attested_by_cri(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "postgres-pods.jsonl")
    for row in rows:
        row["images"][0]["image_id"] = "docker-pullable://postgres@sha256:" + "7" * 64
    _write_jsonl(tmp_path / "postgres-pods.jsonl", rows)
    _refresh_artifact(manifest_path, "postgres-pods.jsonl")

    result = gateway_bench.verify_evidence(tmp_path)

    assert result["passed"]
    assert result["checks"][
        "runtime_gateway_mock_and_postgres_images_match_sources"
    ]


def test_verifier_rejects_docker_metadata_unlinked_from_the_postgres_pin(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    metadata_path = tmp_path / "postgres-docker-image.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload[0]["RepoDigests"] = ["postgres@sha256:" + "9" * 64]
    metadata_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"][
        "postgres_docker_image_metadata"
    ] = gateway_bench._docker_image_metadata(metadata_path, tmp_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = gateway_bench.verify_evidence(tmp_path)

    assert not result["passed"]
    assert not result["checks"]["clean_source_and_image_references_attested"]


def test_verifier_rejects_a_tampered_sidecar_digest(tmp_path: Path) -> None:
    _passing_evidence(tmp_path)
    with (tmp_path / "traffic.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["artifact_paths_and_digests_replay"]


def test_verifier_recomputes_lb_hash_instead_of_trusting_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "traffic.jsonl")
    rows[0]["gateway_id"] = "c" if rows[0]["gateway_id"] != "c" else "b"
    _write_jsonl(tmp_path / "traffic.jsonl", rows)
    _refresh_artifact(manifest_path, "traffic.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["lb_sha256_hrw_matches_every_baseline_request"]


def test_verifier_requires_the_frozen_baseline_cardinality(tmp_path: Path) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "traffic.jsonl")
    rows.pop(0)
    _write_jsonl(tmp_path / "traffic.jsonl", rows)
    _refresh_artifact(manifest_path, "traffic.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["baseline_sessions_remain_sticky"]


def test_verifier_recomputes_affinity_identity_from_response_json(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "traffic.jsonl")
    rows[0]["response_json"]["choices"][0]["message"]["content"] = (
        "kairyu-f1a-mock pod_name=forged-pod pod_uid=forged-uid"
    )
    _write_jsonl(tmp_path / "traffic.jsonl", rows)
    _refresh_artifact(manifest_path, "traffic.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "affinity_response_json_recomputes_exact_pod_identity"
    ]


def test_verifier_rejects_a_broken_batch_object_id_chain(tmp_path: Path) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "batch-http.jsonl")
    read_input = next(
        row for row in rows if row["operation"] == "read_input_b"
    )
    read_input["path"] = "/v1/files/forged-input"
    _write_jsonl(tmp_path / "batch-http.jsonl", rows)
    _refresh_artifact(manifest_path, "batch-http.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "batch_http_ids_paths_and_lifecycle_form_one_exact_graph"
    ]


def test_verifier_rejects_a_terminal_status_before_completed_poll(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "batch-http.jsonl")
    polls = [row for row in rows if row["operation"] == "poll_batch_c"]
    polls[0]["response_json"]["status"] = "failed"
    _write_jsonl(tmp_path / "batch-http.jsonl", rows)
    _refresh_artifact(manifest_path, "batch-http.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "batch_http_ids_paths_and_lifecycle_form_one_exact_graph"
    ]


def test_verifier_rejects_a_nonconverged_gateway_membership(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "placements-c.jsonl")
    membership = next(
        row for row in reversed(rows) if row["kind"] == "membership"
    )
    membership["eligible_ids"].pop()
    _write_jsonl(tmp_path / "placements-c.jsonl", rows)
    _refresh_artifact(manifest_path, "placements-c.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "all_gateway_membership_logs_converge_to_raw_replica_uids"
    ]


def test_verifier_rejects_extra_attempt_on_a_normal_lb_request(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "lb-decisions.jsonl")
    candidates = rows[0]["candidate_gateway_ids"]
    rows[0]["attempts"].append(
        {
            "gateway_id": candidates[1],
            "outcome": "response",
            "status": 200,
            "upstream_request_id": "forged-upstream-request",
        }
    )
    _write_jsonl(tmp_path / "lb-decisions.jsonl", rows)
    _refresh_artifact(manifest_path, "lb-decisions.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["lb_decision_log_joins_raw_responses"]


def test_verifier_requires_reclaim_after_the_last_owner_lease(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    claims = gateway_bench._read_jsonl(tmp_path / "batch-claims.jsonl")
    claims[2]["at_unix_ns"] = 17
    _write_jsonl(tmp_path / "batch-claims.jsonl", claims)
    _refresh_artifact(manifest_path, "batch-claims.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "durable_claim_audit_has_one_new-fence_terminal_commit"
    ]


def test_verifier_rejects_a_stale_owner_renewal_after_reclaim(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    claims = gateway_bench._read_jsonl(tmp_path / "batch-claims.jsonl")
    claims[3]["worker_id"] = claims[0]["worker_id"]
    claims[3]["fencing_token"] = claims[0]["fencing_token"]
    _write_jsonl(tmp_path / "batch-claims.jsonl", claims)
    _refresh_artifact(manifest_path, "batch-claims.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "durable_claim_audit_has_one_new-fence_terminal_commit"
    ]


def test_verifier_rejects_any_claim_event_after_terminal_publication(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    claims = gateway_bench._read_jsonl(tmp_path / "batch-claims.jsonl")
    claims.append(
        {
            "schema_version": gateway_bench.SCHEMA_VERSION,
            "sequence": 6,
            "batch_id": claims[-1]["batch_id"],
            "event": "renewed",
            "worker_id": claims[-1]["worker_id"],
            "fencing_token": claims[-1]["fencing_token"],
            "at_unix_ns": 26,
            "lease_until_unix_ns": 35,
            "previous_lease_until_unix_ns": None,
        }
    )
    _write_jsonl(tmp_path / "batch-claims.jsonl", claims)
    _refresh_artifact(manifest_path, "batch-claims.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "durable_claim_audit_has_one_new-fence_terminal_commit"
    ]


def test_verifier_rejects_terminal_publication_by_the_stale_fence(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    claims = gateway_bench._read_jsonl(tmp_path / "batch-claims.jsonl")
    claims[-1]["worker_id"] = "gateway-a-old"
    claims[-1]["fencing_token"] = 1
    _write_jsonl(tmp_path / "batch-claims.jsonl", claims)
    _refresh_artifact(manifest_path, "batch-claims.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "durable_claim_audit_has_one_new-fence_terminal_commit"
    ]


def test_verifier_rejects_output_that_differs_through_one_gateway(
    tmp_path: Path,
) -> None:
    manifest_path = _passing_evidence(tmp_path)
    rows = gateway_bench._read_jsonl(tmp_path / "batch-output-c.jsonl")
    rows[0]["response"]["body"]["index"] = 99
    _write_jsonl(tmp_path / "batch-output-c.jsonl", rows)
    _refresh_artifact(manifest_path, "batch-output-c.jsonl")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"][
        "output_is_byte_identical_and_complete_through_all_gateways"
    ]


def test_verifier_replays_the_rendered_manifest_digest(tmp_path: Path) -> None:
    _passing_evidence(tmp_path)
    with (tmp_path / "rendered-manifest.yaml").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("# tampered\n")
    result = gateway_bench.verify_evidence(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["clean_source_and_image_references_attested"]
