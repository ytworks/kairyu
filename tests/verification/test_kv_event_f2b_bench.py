"""Focused tests for the replayable F2b KV-event chaos evidence gate."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from verification.fleet.correctness import kv_event_f2b_bench as f2b
from verification.fleet.resilience.fleet_churn_bench import FORMAL_CONFIG, churn_plan


def _source() -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "git_clean_at_start": True,
        "files": [{"path": path, "sha256": "b" * 64} for path in f2b._SOURCE_PATHS],
    }


def _source_end(source: dict[str, object]) -> dict[str, object]:
    return {
        key: source[key]
        for key in (
            "commit",
            "files",
        )
    }


def _github_context() -> dict[str, str]:
    commit = "a" * 40
    return {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": "ytworks/kairyu",
        "GITHUB_WORKFLOW": "F2b KV-event freshness and chaos",
        "GITHUB_WORKFLOW_REF": (
            "ytworks/kairyu/.github/workflows/f2b-kv-event-chaos.yml@refs/heads/main"
        ),
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": commit,
    }


def _environment(
    github: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "platform": "Linux-test",
        "python": "3.12.0 test",
        "cpu_count": 8,
        "sched_affinity": list(range(8)),
        "github": {} if github is None else github,
    }


def _logical_apply(
    rows: list[dict[str, object]],
    *,
    replica_id: str,
    generation: str,
    event: dict[str, object],
    emitted_ns: int,
    epoch: int | None,
    purpose: str,
) -> None:
    stream_epoch = f"logical:{replica_id}:{generation}"
    sequence = sum(
        row.get("type") == "logical_apply"
        and row.get("replica_id") == replica_id
        and row.get("replica_generation") == generation
        for row in rows
    )
    rows.append(
        {
            "type": "logical_apply",
            "channel": "in_process",
            "replica_id": replica_id,
            "replica_generation": generation,
            "stream_epoch": stream_epoch,
            "sequence": sequence,
            "result": "applied",
            "synchronized": True,
            "epoch": epoch,
            "purpose": purpose,
            "event_kind": event["type"],
            "event": event,
            "emitted_ns": emitted_ns,
            "applied_ns": emitted_ns + 1_000,
        }
    )


def _wire_delivery(
    rows: list[dict[str, object]],
    emit: dict[str, object],
    *,
    generation: str,
    received_ns: int,
    applied_ns: int,
    replayed: bool,
) -> None:
    payload = {
        "channel": "representative_zmq",
        "replica_id": emit["replica_id"],
        "replica_generation": generation,
        "stream_epoch": emit["stream_epoch"],
        "sequence": emit["sequence"],
        "emitted_ns": emit["emitted_ns"],
        "received_ns": received_ns,
        "applied_ns": applied_ns,
        "event_kind": emit["event_kind"],
        "event": emit["event"],
        "replayed": replayed,
        "result": "applied",
    }
    rows.append({"type": "event_receive", **payload})
    rows.append({"type": "event_apply", **payload})


def _passing_rows() -> list[dict[str, object]]:
    config = f2b.profile_config("smoke")
    plan = f2b.compressed_churn_plan(config)
    source = _source()
    environment = _environment()
    origin = 1_000_000_000
    topology = {
        **f2b.topology_descriptor(config),
        "zmq_transport": "inproc",
    }
    rows: list[dict[str, object]] = [
        {
            "type": "run",
            "schema_version": f2b.SCHEMA_VERSION,
            "config": asdict(config),
            "topology": topology,
            "schedule_digest": f2b._sha256_json(plan),
            "source": source,
            "source_end": _source_end(source),
            "environment": environment,
            "reused_evidence": f2b.reused_evidence_descriptor(),
            "measurement_origin_ns": origin,
        }
    ]
    rows.extend({"type": "schedule", **row} for row in plan)

    replica_ids = [f2b._replica_id(ordinal) for ordinal in range(config.replicas)]
    representative = f2b.representative_replica_id(config)
    old_generations = {replica_id: f"old-{replica_id}" for replica_id in replica_ids}
    new_generations = {replica_id: f"new-{replica_id}" for replica_id in replica_ids}
    clear = {"type": "AllBlocksCleared", "block_hashes": []}
    for ordinal, replica_id in enumerate(replica_ids):
        if replica_id != representative:
            _logical_apply(
                rows,
                replica_id=replica_id,
                generation=old_generations[replica_id],
                event=clear,
                emitted_ns=origin - 250_000_000 + ordinal * 10_000,
                epoch=None,
                purpose="initialization",
            )

    stream_epoch = "stream-epoch-1"
    replacement_stream_epoch = "stream-epoch-2"
    emitted_events: list[dict[str, object]] = [
        {
            "type": "event_emit",
            "channel": "representative_zmq",
            "replica_id": representative,
            "replica_generation": old_generations[representative],
            "stream_epoch": stream_epoch,
            "sequence": 0,
            "epoch": None,
            "purpose": "initialization",
            "event_kind": "AllBlocksCleared",
            "event": clear,
            "enqueued_ns": origin - 240_001_000,
            "emitted_ns": origin - 240_000_000,
        }
    ]

    membership_rows: list[dict[str, object]] = []
    churn_rows: list[dict[str, object]] = []
    stream_rotation_rows: list[dict[str, object]] = []
    for epoch, plan_row in enumerate(plan):
        exact_target, approximate_target = f2b._epoch_targets(config, epoch)
        prompt = f2b._epoch_prompt(epoch)
        rows.append(
            {
                "type": "prefix_seed",
                "epoch": epoch,
                "replica_id": approximate_target,
                "observed_ns": origin
                + int(epoch * config.epoch_seconds * f2b._NS_PER_SECOND)
                + 100_000,
                "prompt_sha256": f2b.hashlib.sha256(prompt.encode()).hexdigest(),
                "prefix_fingerprint": f2b.prefix_root_fingerprint(prompt),
            }
        )
        stored = {
            "type": "BlockStored",
            "block_hashes": list(f2b._epoch_block_hashes(epoch)),
        }
        epoch_ns = origin + int(epoch * config.epoch_seconds * f2b._NS_PER_SECOND)

        ordinals = plan_row["ordinals"]
        assert isinstance(ordinals, list)
        spacing_ns = int(
            config.churn_window_seconds * f2b._NS_PER_SECOND / max(1, len(ordinals) - 1)
        )
        for batch_index, ordinal in enumerate(ordinals):
            assert isinstance(ordinal, int)
            replica_id = f2b._replica_id(ordinal)
            scheduled_ns = epoch_ns + batch_index * spacing_ns
            started_ns = scheduled_ns + 1_000_000
            completed_ns = scheduled_ns + 10_000_000
            churn_rows.append(
                {
                    "type": "churn",
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "ordinal": ordinal,
                    "replica_id": replica_id,
                    "scheduled_ns": scheduled_ns,
                    "started_ns": started_ns,
                    "completed_ns": completed_ns,
                    "old_generation": old_generations[replica_id],
                    "new_generation": new_generations[replica_id],
                }
            )
            for offset, reason in enumerate(
                ("manual_drain", "replica_removed", "replica_added"),
                2,
            ):
                membership_rows.append(
                    {
                        "type": "membership",
                        "epoch": epoch,
                        "ordinal": ordinal,
                        "scheduled_ns": scheduled_ns,
                        "observed_ns": scheduled_ns + offset * 1_000_000,
                        "reason": reason,
                        "replica_id": replica_id,
                        "replica_generation": (
                            new_generations[replica_id]
                            if reason == "replica_added"
                            else old_generations[replica_id]
                        ),
                        "added": ([replica_id] if reason == "replica_added" else []),
                        "draining": ([replica_id] if reason == "manual_drain" else []),
                        "undraining": [],
                        "removed": ([replica_id] if reason == "replica_removed" else []),
                        "eligible": ([replica_id] if reason == "replica_added" else []),
                        "ineligible": ([replica_id] if reason == "manual_drain" else []),
                    }
                )
            if replica_id == representative:
                stream_rotation_rows.append(
                    {
                        "type": "stream",
                        "action": "rotate",
                        "epoch": epoch,
                        "ordinal": ordinal,
                        "replica_id": representative,
                        "observed_ns": scheduled_ns + 5_000_000,
                        "previous_stream_epoch": stream_epoch,
                        "stream_epoch": replacement_stream_epoch,
                    }
                )
                emitted_events.append(
                    {
                        "type": "event_emit",
                        "channel": "representative_zmq",
                        "replica_id": representative,
                        "replica_generation": new_generations[representative],
                        "stream_epoch": replacement_stream_epoch,
                        "sequence": 0,
                        "epoch": epoch,
                        "purpose": "replacement_clear",
                        "event_kind": "AllBlocksCleared",
                        "event": clear,
                        "enqueued_ns": scheduled_ns + 5_999_000,
                        "emitted_ns": scheduled_ns + 6_000_000,
                    }
                )
            else:
                _logical_apply(
                    rows,
                    replica_id=replica_id,
                    generation=new_generations[replica_id],
                    event=clear,
                    emitted_ns=scheduled_ns + 6_000_000,
                    epoch=epoch,
                    purpose="replacement_clear",
                )
            if replica_id == exact_target:
                if replica_id == representative:
                    emitted_events.append(
                        {
                            "type": "event_emit",
                            "channel": "representative_zmq",
                            "replica_id": representative,
                            "replica_generation": new_generations[representative],
                            "stream_epoch": replacement_stream_epoch,
                            "sequence": 1,
                            "epoch": epoch,
                            "purpose": "replacement_exact_seed",
                            "event_kind": "BlockStored",
                            "event": stored,
                            "enqueued_ns": scheduled_ns + 6_999_000,
                            "emitted_ns": scheduled_ns + 7_000_000,
                        }
                    )
                else:
                    _logical_apply(
                        rows,
                        replica_id=replica_id,
                        generation=new_generations[replica_id],
                        event=stored,
                        emitted_ns=scheduled_ns + 7_000_000,
                        epoch=epoch,
                        purpose="replacement_exact_seed",
                    )
    rows.extend(membership_rows)
    rows.extend(churn_rows)
    rows.extend(stream_rotation_rows)
    rows.extend(emitted_events)

    old_stream_events = [event for event in emitted_events if event["stream_epoch"] == stream_epoch]
    replacement_stream_events = [
        event for event in emitted_events if event["stream_epoch"] == replacement_stream_epoch
    ]
    for offset, emit in enumerate(old_stream_events):
        _wire_delivery(
            rows,
            emit,
            generation=old_generations[representative],
            received_ns=origin - 230_000_000 + offset * 1_000_000,
            applied_ns=origin - 229_000_000 + offset * 1_000_000,
            replayed=False,
        )
    replay_received_ns = origin + 734_000_000
    for offset, emit in enumerate(replacement_stream_events):
        _wire_delivery(
            rows,
            emit,
            generation=new_generations[representative],
            received_ns=replay_received_ns,
            applied_ns=replay_received_ns + (offset + 1) * 100_000,
            replayed=True,
        )

    direct_ids = [replica_id for replica_id in replica_ids if replica_id != representative]
    heartbeat_sequence = 0
    for applied_ns in range(
        origin - 20_000_000,
        origin + 1_100_000_000,
        20_000_000,
    ):
        stream_high_water = {}
        for replica_id in direct_ids:
            latest = max(
                (
                    row
                    for row in rows
                    if row.get("type") == "logical_apply"
                    and row.get("replica_id") == replica_id
                    and int(row["applied_ns"]) <= applied_ns
                ),
                key=lambda row: int(row["applied_ns"]),
            )
            stream_high_water[replica_id] = {
                "stream_epoch": latest["stream_epoch"],
                "high_sequence": latest["sequence"],
            }
        rows.append(
            {
                "type": "logical_heartbeat",
                "sequence": heartbeat_sequence,
                "scheduled_ns": applied_ns - 1_000_000,
                "started_ns": applied_ns - 500_000,
                "finished_ns": applied_ns,
                "replica_ids": direct_ids,
                "replica_count": len(direct_ids),
                "stream_high_water": stream_high_water,
                "all_refreshed": True,
            }
        )
        heartbeat_sequence += 1
    wire_heartbeat_times = list(
        range(
            origin - 20_000_000,
            origin + 281_000_000,
            20_000_000,
        )
    )
    wire_heartbeat_times.extend(
        range(
            origin + 741_000_000,
            origin + 1_100_000_000,
            20_000_000,
        )
    )
    for applied_ns in wire_heartbeat_times:
        rows.append(
            {
                "type": "heartbeat",
                "channel": "representative_zmq",
                "replica_id": representative,
                "replica_generation": (
                    old_generations[representative]
                    if applied_ns < origin + 500_000_000
                    else new_generations[representative]
                ),
                "stream_epoch": (
                    stream_epoch if applied_ns < origin + 500_000_000 else replacement_stream_epoch
                ),
                "sequence": (
                    len(old_stream_events) - 1
                    if applied_ns < origin + 500_000_000
                    else len(replacement_stream_events) - 1
                ),
                "emitted_ns": applied_ns - 5_000_000,
                "received_ns": applied_ns - 1_000_000,
                "applied_ns": applied_ns,
                "event_kind": "heartbeat",
                "event": None,
                "replayed": False,
                "result": ("recovered" if applied_ns == origin + 741_000_000 else "refreshed"),
            }
        )

    pause = {
        "type": "kill",
        "action": "pause",
        "fault": "subscriber_live_socket_close",
        "scheduled_ns": origin + 300_000_000,
        "requested_ns": origin + 301_000_000,
        "observed_ns": origin + 302_000_000,
        "subscriber_socket_id": 10,
    }
    resume = {
        "type": "kill",
        "action": "resume",
        "fault": "subscriber_live_socket_reconnect",
        "scheduled_ns": origin + 720_000_000,
        "requested_ns": origin + 721_000_000,
        "observed_ns": origin + 722_000_000,
        "subscriber_socket_id": 11,
    }
    rows.extend((pause, resume))
    identity = {
        "pid": 100,
        "pool_id": 101,
        "routing_id": 102,
        "index_id": 103,
        "publisher_id": 104,
        "publisher_thread_id": 105,
        "subscriber_id": 106,
    }
    rows.extend(
        (
            {
                "type": "process_identity",
                "phase": "ready",
                "observed_ns": origin - 200_000_000,
                "subscriber_socket_id": 10,
                **identity,
            },
            {
                "type": "process_identity",
                "phase": "outage",
                "observed_ns": pause["observed_ns"],
                "subscriber_socket_id": 10,
                **identity,
            },
            {
                "type": "process_identity",
                "phase": "restored",
                "observed_ns": origin + 742_000_000,
                "subscriber_socket_id": 11,
                **identity,
            },
        )
    )
    rows.extend(
        (
            {
                "type": "stream",
                "action": "ready",
                "observed_ns": origin - 200_000_000,
                "stream_epoch": stream_epoch,
                "high_sequence": 0,
            },
            {
                "type": "stream",
                "action": "recovered",
                "observed_ns": origin + 742_000_000,
                "stream_epoch": replacement_stream_epoch,
                "high_sequence": len(replacement_stream_events) - 1,
            },
        )
    )

    interval_ns = int(f2b._NS_PER_SECOND / config.route_rate)
    for sequence in range(config.expected_measurement_routes):
        scheduled_ns = origin + sequence * interval_ns
        boundary = sequence % int(config.epoch_seconds * config.route_rate) == 0
        selected_ns = scheduled_ns + (5_000_000 if boundary else 1_000_000)
        epoch = min(
            config.churn_epochs - 1,
            int((sequence / config.route_rate) // config.epoch_seconds),
        )
        exact_target, approximate_target = f2b._epoch_targets(config, epoch)
        if selected_ns < origin + 380_000_000:
            reason = "kv_event_match"
            mode = "exact"
            selected = exact_target
        elif selected_ns < origin + 740_000_000:
            reason = "prefix_match"
            mode = "approximate"
            selected = approximate_target
        else:
            reason = "kv_event_match"
            mode = "exact"
            selected = exact_target
        prompt = f2b._epoch_prompt(epoch)
        rows.append(
            {
                "type": "route",
                "sequence": sequence,
                "epoch": epoch,
                "scheduled_ns": scheduled_ns,
                "selected_ns": selected_ns,
                "completed_ns": selected_ns + 100_000,
                "request_id": f"f2b-route-{sequence:05d}",
                "prompt_sha256": f2b.hashlib.sha256(prompt.encode()).hexdigest(),
                "prefix_fingerprint": f2b.prefix_root_fingerprint(prompt),
                "session_id": f2b._session_for_approximate(epoch, exact_target, approximate_target),
                "selected_replica": selected,
                "backend_replica": selected,
                "reason": reason,
                "mode": mode,
                "expected_exact_replica": exact_target,
                "expected_approximate_replica": approximate_target,
                "pool_size": config.replicas,
                "eligible_size": config.replicas,
                "representative_view": {},
            }
        )

    rows.append(
        {
            "type": "replay",
            "replica_id": representative,
            "replica_generation": new_generations[representative],
            "mode": "events",
            "stream_epoch": replacement_stream_epoch,
            "requested_from": 0,
            "first_available": 0,
            "last_available": len(replacement_stream_events) - 1,
            "high_sequence": len(replacement_stream_events) - 1,
            "event_count": len(replacement_stream_events),
            "reason": None,
            "requested_ns": origin + 725_000_000,
            "response_received_ns": replay_received_ns,
            "completed_ns": origin + 740_000_000,
        }
    )
    recovery_hashes = f2b._rebuild_rotated_stream(
        rows,
        replica_id=representative,
        replica_generation=new_generations[representative],
        stream_epoch=replacement_stream_epoch,
        high_sequence=len(replacement_stream_events) - 1,
    )
    assert recovery_hashes is not None
    recovery_digest = f2b._sha256_json(recovery_hashes)
    rows.append(
        {
            "type": "state_snapshot",
            "phase": "recovery",
            "replica_id": representative,
            "replica_generation": new_generations[representative],
            "stream_epoch": replacement_stream_epoch,
            "high_sequence": len(replacement_stream_events) - 1,
            "replay_completed_ns": origin + 740_000_000,
            "observed_ns": origin + 741_000_000,
            "index_view": {
                "replica_id": representative,
                "state": "live",
                "stream_epoch": replacement_stream_epoch,
                "last_sequence": len(replacement_stream_events) - 1,
                "trusted": True,
                "synchronized": True,
                "age_s": 0.001,
                "block_hashes": copy.deepcopy(recovery_hashes),
            },
            "authoritative_hashes": copy.deepcopy(recovery_hashes),
            "index_hashes": copy.deepcopy(recovery_hashes),
            "authoritative_digest": recovery_digest,
            "index_digest": recovery_digest,
        }
    )
    final_generations = {replica_id: new_generations[replica_id] for replica_id in replica_ids}
    rows.append(
        {
            "type": "fleet_snapshot",
            "phase": "final",
            "epoch": None,
            "observed_ns": origin + 1_100_000_000,
            "replica_ids": replica_ids,
            "healthy_ids": replica_ids,
            "eligible_ids": replica_ids,
            "generation_by_id": final_generations,
            "replica_digest": f2b._sha256_json(replica_ids),
            "eligible_digest": f2b._sha256_json(replica_ids),
        }
    )
    rebuilt = f2b._rebuild_authoritative_state(rows, config)
    assert rebuilt is not None
    digest = f2b._sha256_json(rebuilt)
    rows.append(
        {
            "type": "state_snapshot",
            "phase": "final",
            "observed_ns": origin + 1_100_000_000,
            "authoritative_hashes_by_replica": rebuilt,
            "index_hashes_by_replica": copy.deepcopy(rebuilt),
            "authoritative_digest": digest,
            "index_digest": digest,
        }
    )
    rows.append(
        {
            "type": "transport_summary",
            "publisher_stream_epoch": replacement_stream_epoch,
            "publisher_high_sequence": len(replacement_stream_events) - 1,
            "publisher_dropped_live_events": 0,
            "delivery_count": len([row for row in rows if row.get("type") == "event_receive"]),
            "recovery_count": 1,
        }
    )
    for row_index, row in enumerate(rows):
        row["row_index"] = row_index
    assert all(f2b.threshold_checks(rows, config).values())
    assert all(f2b.diagnostic_checks(rows, config).values())
    return rows


def _write_artifact(
    output_dir: Path,
    rows: list[dict[str, object]],
    *,
    update_manifest: bool = True,
) -> Path:
    config = f2b.profile_config("smoke")
    source = rows[0]["source"]
    source_end = rows[0]["source_end"]
    environment = rows[0]["environment"]
    output_dir.mkdir(parents=True)
    raw_path = output_dir / f2b.RAW_NAME
    f2b._write_jsonl(raw_path, rows)
    gates = f2b.threshold_checks(rows, config)
    diagnostics = f2b.diagnostic_checks(rows, config)
    metrics = f2b.derive_metrics(rows, config)
    manifest = {
        "schema_version": f2b.SCHEMA_VERSION,
        "gate": f2b.GATE,
        "config": asdict(config),
        "schedule": f2b.compressed_churn_plan(config),
        "topology": rows[0]["topology"],
        "reused_evidence": f2b.reused_evidence_descriptor(),
        "source": source,
        "source_end": source_end,
        "environment": environment,
        "raw": {
            "path": f2b.RAW_NAME,
            "sha256": f2b._sha256_file(raw_path),
            "rows": len(rows),
        },
        "metrics": metrics,
        "gate_checks": gates,
        "diagnostic_checks": diagnostics,
        "passed": all(gates.values()),
    }
    if not update_manifest:
        manifest["metrics"] = {"attacker": "did not recompute"}
    manifest_path = output_dir / f2b.MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


@pytest.fixture
def source_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        f2b,
        "_verify_source_descriptor",
        lambda _source: {
            "source_commit_is_recorded": True,
            "source_commit_is_ancestor": True,
            "source_files_are_exact": True,
            "source_file_hashes_match_commit": True,
            "source_clean_at_start": True,
        },
    )


def test_formal_schedule_reuses_f1a_exactly_without_rerunning_it() -> None:
    config = f2b.profile_config("formal")
    f1a = churn_plan(FORMAL_CONFIG)
    f2b_plan = f2b.compressed_churn_plan(config)

    assert (config.replicas, config.churn_batch_size, config.churn_epochs) == (
        200,
        20,
        10,
    )
    assert config.seed == 175
    assert config.churn_acceleration == 60.0
    assert config.churn_window_seconds == pytest.approx(1 / 6)
    assert [row["ordinals"] for row in f2b_plan] == [row["ordinals"] for row in f1a]
    assert [row["source_deadline_offset_s"] for row in f2b_plan] == [
        row["deadline_offset_s"] for row in f1a
    ]
    assert sorted(ordinal for row in f2b_plan for ordinal in row["ordinals"]) == list(range(200))


def test_freshness_gate_has_two_times_stale_lease_headroom() -> None:
    formal = f2b.profile_config("formal")

    assert formal.stale_after_seconds == 0.25
    assert formal.freshness_limit_seconds == 0.5
    assert formal.heartbeat_seconds <= formal.stale_after_seconds / 5
    assert formal.expected_measurement_routes == 500
    reused = f2b.reused_evidence_descriptor()
    assert reused["f1a"]["actions_run_id"] == "30374404150"
    assert reused["f1a"]["evidence_status"] == "external_context_only"
    assert reused["f2a"]["actions_run_id"] == "30411111758"
    assert reused["f2a"]["raw_sha256"] == f2b.F2A_RAW_SHA256


def test_formal_context_binds_source_and_local_environment_not_github() -> None:
    config = f2b.profile_config("formal")
    source = _source()
    environment = _environment(_github_context())

    assert f2b._formal_source_and_environment_complete(
        config,
        source,
        environment,
    )
    assert f2b._formal_source_and_environment_complete(
        config,
        source,
        {**environment, "github": {"GITHUB_RUN_ID": "different"}},
    )
    assert not f2b._formal_source_and_environment_complete(
        config,
        {**source, "git_clean_at_start": False},
        environment,
    )
    assert not f2b._formal_source_and_environment_complete(
        config,
        source,
        {**environment, "cpu_count": 0},
    )


def test_replay_accepts_complete_raw_evidence(
    tmp_path: Path,
    source_verification: None,
) -> None:
    manifest = _write_artifact(tmp_path / "passing", _passing_rows())

    result = f2b.verify_artifact(manifest)

    assert result["passed"], result["errors"]
    assert all(result["checks"].values())
    assert all(result["diagnostics"].values())
    assert result["metrics"]["chaos"]["recovery_convergence_ns"] == 18_000_000
    assert result["metrics"]["routes"]["stale"] > 0
    assert result["metrics"]["routes"]["max_exact_truth_age_ns"] < 100_000_000


def test_wall_clock_deadlines_are_recorded_but_do_not_bind_pass_fail(
    tmp_path: Path,
    source_verification: None,
) -> None:
    rows = copy.deepcopy(_passing_rows())
    replay = next(row for row in rows if row.get("type") == "replay")
    recovery_snapshot = next(
        row
        for row in rows
        if row.get("type") == "state_snapshot" and row.get("phase") == "recovery"
    )
    recovery_snapshot["observed_ns"] = int(replay["completed_ns"]) + 500_000_000
    manifest = _write_artifact(tmp_path / "slow-recovery-observation", rows)

    result = f2b.verify_artifact(manifest)

    assert result["passed"], result["errors"]
    assert set(result["diagnostics"]) == set(f2b._DIAGNOSTIC_CHECK_NAMES)
    assert not result["diagnostics"][
        "recovery_snapshot_observed_strictly_below_500ms_after_replay"
    ]
    assert all(result["checks"].values())


def test_manifest_cannot_lie_about_nonbinding_diagnostics(
    tmp_path: Path,
    source_verification: None,
) -> None:
    manifest_path = _write_artifact(tmp_path / "diagnostic-integrity", _passing_rows())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["diagnostic_checks"][
        "recovery_snapshot_observed_strictly_below_500ms_after_replay"
    ] = False
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = f2b.verify_artifact(manifest_path)

    assert not result["passed"]
    assert not result["checks"]["manifest_diagnostic_checks_recomputed_exactly"]


@pytest.mark.parametrize(
    ("name", "mutate", "expected_check"),
    [
        pytest.param(
            "stale-exact-route",
            lambda rows: next(
                row
                for row in rows
                if row.get("type") == "route" and row.get("reason") == "prefix_match"
            ).update(
                mode="exact",
                reason="kv_event_match",
                selected_replica="replica-wrong",
            ),
            "every_route_after_actual_lease_expiry_uses_approximate_oracle",
            id="stale-exact-route",
        ),
        pytest.param(
            "approximate-oracle",
            lambda rows: next(
                row
                for row in rows
                if row.get("type") == "route" and row.get("reason") == "prefix_match"
            ).update(selected_replica="replica-wrong"),
            "every_route_after_actual_lease_expiry_uses_approximate_oracle",
            id="approximate-oracle",
        ),
        pytest.param(
            "self-attested-oracle",
            lambda rows: next(row for row in rows if row.get("type") == "route").update(
                expected_exact_replica="replica-wrong",
                selected_replica="replica-wrong",
                backend_replica="replica-wrong",
            ),
            "route_oracle_inputs_recompute_from_the_frozen_trace",
            id="self-attested-oracle",
        ),
        pytest.param(
            "replay-range",
            lambda rows: next(row for row in rows if row.get("type") == "replay").update(
                event_count=1
            ),
            "binding_replay_is_contiguous_complete_and_causally_applied",
            id="replay-range",
        ),
        pytest.param(
            "replay-generation",
            lambda rows: next(row for row in rows if row.get("type") == "replay").update(
                replica_generation="wrong-generation"
            ),
            "binding_replay_is_contiguous_complete_and_causally_applied",
            id="replay-generation",
        ),
        pytest.param(
            "replay-stream-epoch",
            lambda rows: next(row for row in rows if row.get("type") == "replay").update(
                stream_epoch="stream-epoch-1"
            ),
            "binding_replay_is_contiguous_complete_and_causally_applied",
            id="replay-stream-epoch",
        ),
        pytest.param(
            "replay-response-causality",
            lambda rows: next(row for row in rows if row.get("type") == "replay").update(
                response_received_ns=2_000_000_000
            ),
            "binding_replay_is_contiguous_complete_and_causally_applied",
            id="replay-response-causality",
        ),
        pytest.param(
            "missing-replay-apply",
            lambda rows: rows.remove(
                next(
                    row
                    for row in rows
                    if row.get("type") == "event_apply"
                    and row.get("replayed") is True
                    and row.get("sequence") == 1
                )
            ),
            "binding_replay_is_contiguous_complete_and_causally_applied",
            id="missing-replay-apply",
        ),
        pytest.param(
            "restore-digest",
            lambda rows: next(
                row
                for row in rows
                if row.get("type") == "state_snapshot" and row.get("phase") == "final"
            ).update(index_digest="d" * 64),
            "final_state_rebuilds_the_authoritative_200_replica_digest",
            id="restore-digest",
        ),
        pytest.param(
            "recovery-digest",
            lambda rows: next(
                row
                for row in rows
                if row.get("type") == "state_snapshot" and row.get("phase") == "recovery"
            ).update(index_digest="d" * 64),
            "recovery_snapshot_rebuilds_rotated_epoch_at_replay_completion",
            id="recovery-digest",
        ),
        pytest.param(
            "same-process",
            lambda rows: next(
                row
                for row in rows
                if row.get("type") == "process_identity" and row.get("phase") == "restored"
            ).update(pool_id=999),
            "pool_router_index_publisher_and_subscriber_never_restart",
            id="same-process",
        ),
        pytest.param(
            "frozen-schedule",
            lambda rows: next(row for row in rows if row.get("type") == "schedule")[
                "ordinals"
            ].reverse(),
            "schedule_matches_frozen_plan",
            id="frozen-schedule",
        ),
        pytest.param(
            "missing-churn",
            lambda rows: rows.remove(next(row for row in rows if row.get("type") == "churn")),
            "all_planned_churn_operations_execute_once_with_new_generations",
            id="missing-churn",
        ),
        pytest.param(
            "logical-sequence",
            lambda rows: next(
                row
                for row in rows
                if row.get("type") == "logical_apply" and row.get("sequence") == 0
            ).update(sequence=7),
            "logical_feeds_are_sequenced_synchronized_and_heartbeat_bound",
            id="logical-sequence",
        ),
        pytest.param(
            "physical-epoch-rotation",
            lambda rows: next(
                row for row in rows if row.get("type") == "stream" and row.get("action") == "rotate"
            ).update(stream_epoch="stream-epoch-1"),
            "physical_feed_rotates_once_and_is_contiguous_per_cache_lifetime",
            id="physical-epoch-rotation",
        ),
        pytest.param(
            "route-tick",
            lambda rows: next(row for row in rows if row.get("type") == "route").update(
                scheduled_ns=123
            ),
            "route_ticks_are_absolute_complete_and_monotonic",
            id="route-tick",
        ),
    ],
)
def test_semantic_tampering_fails_after_rehash_and_manifest_recompute(
    tmp_path: Path,
    source_verification: None,
    name: str,
    mutate,
    expected_check: str,
) -> None:
    rows = copy.deepcopy(_passing_rows())
    mutate(rows)
    manifest = _write_artifact(tmp_path / name, rows)

    result = f2b.verify_artifact(manifest)

    assert not result["passed"]
    assert result["checks"][expected_check] is False


def test_manifest_metrics_cannot_replace_raw_replay(
    tmp_path: Path,
    source_verification: None,
) -> None:
    manifest = _write_artifact(
        tmp_path / "manifest-tamper",
        _passing_rows(),
        update_manifest=False,
    )

    result = f2b.verify_artifact(manifest)

    assert not result["passed"]
    assert result["checks"]["manifest_metrics_recomputed_exactly"] is False


def test_raw_bytes_are_bound_before_semantic_replay(
    tmp_path: Path,
    source_verification: None,
) -> None:
    manifest = _write_artifact(tmp_path / "raw-hash", _passing_rows())
    raw_path = manifest.parent / f2b.RAW_NAME
    raw_path.write_text(
        raw_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    result = f2b.verify_artifact(manifest)

    assert not result["passed"]
    assert result["checks"]["raw_sha256_matches"] is False


def test_retained_replay_requires_byte_identical_current_sources(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "retained"
    artifact.mkdir()
    source = {
        "files": [
            {
                "path": path,
                "sha256": f2b._sha256_file(f2b._REPO_ROOT / path),
            }
            for path in f2b._SOURCE_PATHS
        ]
    }
    manifest = artifact / f2b.MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "config": asdict(f2b.profile_config("smoke")),
                "source": source,
            }
        ),
        encoding="utf-8",
    )

    assert f2b.artifact_source_matches_current(artifact)
    assert f2b.artifact_source_matches_current(
        artifact,
        required_profile="smoke",
    )
    assert not f2b.artifact_source_matches_current(
        artifact,
        required_profile="formal",
    )

    source["files"][0]["sha256"] = "0" * 64
    manifest.write_text(
        json.dumps(
            {
                "config": asdict(f2b.profile_config("smoke")),
                "source": source,
            }
        ),
        encoding="utf-8",
    )
    assert not f2b.artifact_source_matches_current(artifact)


def test_verify_artifact_enforces_required_profile_and_current_source(
    tmp_path: Path,
    source_verification: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_artifact(tmp_path / "policy", _passing_rows())

    rejected = f2b.verify_artifact(
        manifest,
        required_profile="formal",
        require_current_source=True,
    )

    assert not rejected["passed"]
    assert rejected["checks"]["required_profile_matches"] is False
    assert rejected["checks"]["artifact_sources_match_current_worktree"] is False

    monkeypatch.setattr(
        f2b,
        "artifact_source_matches_current",
        lambda _path, *, required_profile=None: required_profile == "smoke",
    )
    accepted = f2b.verify_artifact(
        manifest,
        required_profile="smoke",
        require_current_source=True,
    )
    assert accepted["passed"], accepted["errors"]


def test_fresh_replay_requires_same_local_environment_not_github_identity(
    tmp_path: Path,
    source_verification: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _passing_rows()
    measured_environment = _environment(_github_context())
    rows[0]["environment"] = measured_environment
    manifest = _write_artifact(tmp_path / "fresh", rows)
    monkeypatch.setattr(
        f2b,
        "environment_descriptor",
        lambda: {
            **measured_environment,
            "github": {"GITHUB_RUN_ID": "a-different-diagnostic-run"},
        },
    )

    accepted = f2b.verify_artifact(
        manifest,
        require_current_runner_context=True,
    )
    assert accepted["passed"], accepted["errors"]
    assert accepted["checks"]["artifact_local_environment_matches_current_runner"]

    monkeypatch.setattr(
        f2b,
        "environment_descriptor",
        lambda: {**measured_environment, "python": "3.13.0 different"},
    )
    rejected = f2b.verify_artifact(
        manifest,
        require_current_runner_context=True,
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["artifact_local_environment_matches_current_runner"]


def test_cli_forwards_replay_policy_to_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_verify(
        path: Path,
        *,
        required_profile: str | None,
        require_current_source: bool,
        require_current_runner_context: bool,
    ) -> dict[str, object]:
        captured.update(
            path=path,
            required_profile=required_profile,
            require_current_source=require_current_source,
            require_current_runner_context=require_current_runner_context,
        )
        return {"passed": True}

    monkeypatch.setattr(f2b, "verify_artifact", fake_verify)
    artifact = tmp_path / "artifact"

    assert (
        f2b.main(
            [
                "--verify-artifact",
                str(artifact),
                "--required-profile",
                "formal",
                "--require-current-source",
                "--require-current-runner-context",
            ]
        )
        == 0
    )
    assert captured == {
        "path": artifact,
        "required_profile": "formal",
        "require_current_source": True,
        "require_current_runner_context": True,
    }
    assert '"passed": true' in capsys.readouterr().out


def test_workflow_fresh_pr_policy_and_source_closure_are_explicit() -> None:
    direct_dependencies = {
        "kairyu/__init__.py",
        "kairyu/audit_io.py",
        "kairyu/engine/__init__.py",
        "kairyu/outputs.py",
        "kairyu/sampling_params.py",
        "kairyu/telemetry.py",
        "kairyu/orchestration/__init__.py",
        "kairyu/orchestration/features.py",
        "kairyu/orchestration/router.py",
    }
    workflow = (f2b._REPO_ROOT / ".github/workflows/f2b-kv-event-chaos.yml").read_text(
        encoding="utf-8"
    )
    bench_source = Path(f2b.__file__).read_text(encoding="utf-8")

    assert direct_dependencies <= set(f2b._SOURCE_PATHS)
    assert all(f'- "{path}"' in workflow for path in direct_dependencies)
    assert workflow.count("--require-current-source") == 2
    assert workflow.count("--require-current-runner-context") == 1
    assert "--actions-run-provenance" not in workflow
    assert '"bench/results/f2b-kv-event-retained/**"' in workflow
    assert "PROFILE: ${{ github.event_name == 'pull_request' && 'formal'" in workflow
    assert "F2B_EXPECTED_COMMIT:" not in workflow
    assert "F2B_GITHUB_HEAD_SHA:" not in workflow
    assert "actions: read" not in workflow
    assert "steps.mode.outputs" not in workflow
    assert "gh api" not in workflow
    assert "gh run download" not in workflow
    assert "github.event_name == 'pull_request' || inputs.mode == 'measure'" in workflow
    assert "github.event_name == 'workflow_dispatch' && inputs.mode == 'replay'" in workflow
    assert '--verify-artifact "${F2B_RESULTS_DIR}"' in workflow
    assert '--verify-artifact "${REPLAY_DIR}"' in workflow
    assert "membership_lock = asyncio.Lock()" in bench_source
    assert bench_source.count("async with membership_lock:") >= 3
