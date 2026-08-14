"""Fast CPU tests for the fail-closed G2 A9 crossover report."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from verification.l1.performance import dp_scaling_g2_a8_bench as a8
from verification.l1.performance import g2_a9_dp_tp_crossover_bench as a9


def _rate_metrics(orders: tuple[int, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rate, order in zip(a9.RATES_RPS, orders, strict=True):
        rows.append(
            {
                "rate_rps": rate,
                "dp_median_goodput_rps": 10.0 + order,
                "tp8_median_goodput_rps": 10.0,
                "dp_median_max_in_flight": float(rate),
                "tp8_median_max_in_flight": float(rate + 1),
            }
        )
    return rows


def _a8_manifest(
    *,
    passed: bool = False,
    checks: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "passed": passed,
        "checks": dict(
            checks
            if checks is not None
            else {a9.A8_ACCEPTED_FALSE_CHECK: False}
        ),
        "provenance": {
            "source_commit": a9.A8_SOURCE_COMMIT,
            "image_id": a9.A8_IMAGE_ID,
            "model": a9.MODEL,
            "model_revision": a9.MODEL_REVISION,
            "checkpoint": {
                "weights_sha256": a8.CHECKPOINT_WEIGHTS_SHA256,
                "index_sha256": a8.CHECKPOINT_INDEX_SHA256,
                "tokenizer_sha256": a8.PINNED_TOKENIZER_SHA256,
            },
        },
    }


def _metric_row(
    *,
    arm: str,
    repeat: int,
    rate_rps: int,
    sequence: int,
    origin_ns: int,
) -> dict[str, object]:
    scheduled_ns = origin_ns + round(sequence * 1_000_000_000 / rate_rps)
    started_ns = scheduled_ns + 1_000
    first_token_ns = started_ns + 10_000
    ended_ns = first_token_ns + 31 * 20_000
    return {
        "type": "request",
        "phase": "scaling",
        "arm": arm,
        "repeat": repeat,
        "rate_rps": rate_rps,
        "sequence": sequence,
        "scheduled_ns": scheduled_ns,
        "started_ns": started_ns,
        "first_token_ns": first_token_ns,
        "ended_ns": ended_ns,
        "ttft_ns": first_token_ns - started_ns,
        "completion_tokens": a9.OUTPUT_TOKENS,
        "status": "success",
    }


def _synthetic_a8_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repeat in range(a9.REPEATS):
        for rate_index, rate in enumerate(a9.RATES_RPS):
            origin_ns = 10**12 + repeat * 10**11 + rate_index * 10**10
            rows.extend(
                _metric_row(
                    arm="dp",
                    repeat=repeat,
                    rate_rps=rate,
                    sequence=sequence,
                    origin_ns=origin_ns,
                )
                for sequence in range(a9.REQUESTS_PER_CELL)
            )
    return rows


def _tp8_request(
    *,
    phase: str,
    repeat: int,
    sequence: int,
    scheduled_ns: int,
    response_id: str,
    rate_rps: int | None = None,
) -> dict[str, object]:
    started_ns = scheduled_ns + 1_000
    first_token_ns = started_ns + 10_000
    ended_ns = first_token_ns + 31 * 20_000
    row: dict[str, object] = {
        "type": "request",
        "phase": phase,
        "arm": "tp8",
        "namespace_arm": "dp",
        "repeat": repeat,
        "sequence": sequence,
        "scheduled_ns": scheduled_ns,
        "started_ns": started_ns,
        "first_token_ns": first_token_ns,
        "ended_ns": ended_ns,
        "ttft_ns": first_token_ns - started_ns,
        "completion_tokens": a9.OUTPUT_TOKENS,
        "status": "success",
        "response_request_id": response_id,
    }
    if rate_rps is not None:
        row["rate_rps"] = rate_rps
    return row


def _tp8_placement(request: Mapping[str, object]) -> dict[str, object]:
    started_ns = int(request["started_ns"])
    return {
        "type": "placement",
        "phase": request["phase"],
        "repeat": request["repeat"],
        "rate_rps": request.get("rate_rps"),
        "sequence": request["sequence"],
        "response_request_id": request["response_request_id"],
        "placement_started_ns": started_ns,
        "selected_at_ns": started_ns + 1_000,
        "placement_latency_ns": 1_000,
        "pool_size": 1,
        "eligible_size": 1,
        "replica_id": "0",
        "replica_generation": "1" * 32,
    }


def _synthetic_a9_rows(trace: Mapping[str, object]) -> list[dict[str, object]]:
    provenance = {
        "source_commit": "a" * 40,
        "gpus": [{"index": index} for index in range(8)],
        "containers": {"engine": "fixed"},
        "engine_backends": {"role": "engine-host"},
        "gateway_backends": {"role": "gateway"},
    }
    start = {
        "type": "run",
        "schema_version": a9.SCHEMA_VERSION,
        "gate": a9.GATE,
        "config": a9.formal_config(),
        "trace": copy.deepcopy(trace),
        "provenance": provenance,
        "started_unix_ns": 1,
        "started_monotonic_ns": 1,
    }
    requests: list[dict[str, object]] = []
    clock_ns = 2 * 10**12
    for repeat in range(a9.REPEATS):
        for sequence in range(a9.WARMUP_REQUESTS):
            requests.append(
                _tp8_request(
                    phase="warmup",
                    repeat=repeat,
                    sequence=sequence,
                    scheduled_ns=clock_ns + sequence * 1_000_000,
                    response_id=f"warmup-r{repeat}-{sequence}",
                )
            )
        clock_ns += 10**9
        for rate in a9.RATES_RPS:
            origin_ns = clock_ns
            requests.extend(
                _tp8_request(
                    phase="scaling",
                    repeat=repeat,
                    rate_rps=rate,
                    sequence=sequence,
                    scheduled_ns=(
                        origin_ns
                        + round(sequence * 1_000_000_000 / rate)
                    ),
                    response_id=f"scaling-r{repeat}-{rate}-{sequence}",
                )
                for sequence in range(a9.REQUESTS_PER_CELL)
            )
            clock_ns += 20 * 10**9
    placements = [_tp8_placement(request) for request in requests]
    end = {
        "type": "run_end",
        "completed": True,
        "source_commit": provenance["source_commit"],
        "gpu_inventory": provenance["gpus"],
        "containers": provenance["containers"],
        "engine_backends": provenance["engine_backends"],
        "gateway_backends": provenance["gateway_backends"],
        "request_count": len(requests),
        "placement_count": len(placements),
        "ended_unix_ns": 2,
        "ended_monotonic_ns": 10**18,
    }
    return [start, *requests, *placements, end]


def _patch_synthetic_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    a8_rows: list[dict[str, object]],
) -> None:
    monkeypatch.setattr(
        a9,
        "load_a8_dp_evidence",
        lambda _artifact_dir: (a8_rows, _a8_manifest(), {}),
    )
    monkeypatch.setattr(
        a9,
        "_validate_provenance",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        a9,
        "_validate_tp8_request",
        lambda _row, _trace: None,
    )


def _valid_container_provenance() -> dict[str, object]:
    services: dict[str, object] = {}
    for service, container_id in (
        ("tp8", "1" * 64),
        ("gateway", "2" * 64),
    ):
        services[service] = {
            "container_id": container_id,
            "image_id": a9.A8_IMAGE_ID,
            "running": True,
            "started_at": "2026-07-31T00:00:00Z",
            "compose_config_sha256": "3" * 64,
            "device_ids": (
                [str(index) for index in range(8)]
                if service == "tp8"
                else []
            ),
            "cpuset_cpus": (
                "0-14,16-30,32-46,48-62,64-78,80-94,96-110,112-126"
                if service == "tp8"
                else "15,31,47,63,79,95,111,127"
            ),
            "port": "8300" if service == "tp8" else "8301",
            "mounts": copy.deepcopy(
                a9._expected_normalized_mounts(service)
            ),
        }
    return {
        "project": "kairyu-qwen3-32b-a9-tp8",
        "compose_file": (
            "repo:deploy/verification/qwen3-32b-multi-gpu/a9-tp8-compose.yaml"
        ),
        "services": services,
    }


def _first_row(
    rows: list[dict[str, object]],
    *,
    row_type: str,
) -> dict[str, object]:
    return next(row for row in rows if row.get("type") == row_type)


def _insert_unknown_row(rows: list[dict[str, object]]) -> None:
    rows.insert(-1, {"type": "unknown"})


def _drop_request(rows: list[dict[str, object]]) -> None:
    rows.remove(_first_row(rows, row_type="request"))


def _drop_placement(rows: list[dict[str, object]]) -> None:
    rows.remove(_first_row(rows, row_type="placement"))


def _swap_first_two_requests(rows: list[dict[str, object]]) -> None:
    indices = [
        index for index, row in enumerate(rows) if row.get("type") == "request"
    ][:2]
    rows[indices[0]], rows[indices[1]] = rows[indices[1]], rows[indices[0]]


def _duplicate_response_id(rows: list[dict[str, object]]) -> None:
    requests = [row for row in rows if row.get("type") == "request"]
    requests[1]["response_request_id"] = requests[0]["response_request_id"]


def _orphan_placement(rows: list[dict[str, object]]) -> None:
    placement = _first_row(rows, row_type="placement")
    placement["response_request_id"] = "orphan"


def _break_placement_identity(rows: list[dict[str, object]]) -> None:
    placement = _first_row(rows, row_type="placement")
    placement["repeat"] = 99


@pytest.fixture(scope="module")
def a8_trace() -> dict[str, object]:
    raw_path = a9.A8_ARTIFACT_DIR / a8.RAW_NAME
    with raw_path.open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    trace = first["trace"]
    assert isinstance(trace, dict)
    return trace


@pytest.fixture(scope="module")
def synthetic_evidence(
    a8_trace: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return _synthetic_a9_rows(a8_trace), _synthetic_a8_rows()


def test_formal_config_is_fixed_and_returns_fresh_collections() -> None:
    config = a9.formal_config()

    assert config["arms"] == ["dp2-tp4", "tp8"]
    assert config["rates_rps"] == [4, 8, 16, 32, 64]
    assert config["repeats"] == 3
    assert config["requests_per_cell"] == 64
    assert config["output_tokens"] == 32
    assert config["tp8_warmup_requests_per_repeat"] == 8
    assert config["retry_count"] == 0
    assert config["tpot_method"] == a9.TPOT_METHOD
    assert config["verdict"] == "report-only; no numeric performance threshold"
    capacity = config["topology_capacity"]
    assert isinstance(capacity, dict)
    dp_capacity = capacity["dp2-tp4"]
    tp8_capacity = capacity["tp8"]
    assert isinstance(dp_capacity, dict)
    assert isinstance(tp8_capacity, dict)
    assert dp_capacity["aggregate_usable_kv_pages"] == 16384
    assert tp8_capacity["aggregate_usable_kv_pages"] == 8192
    assert dp_capacity["aggregate_max_num_seqs"] == 32
    assert tp8_capacity["aggregate_max_num_seqs"] == 16

    config["rates_rps"].append(128)
    config["arms"].reverse()
    assert a9.formal_config()["rates_rps"] == [4, 8, 16, 32, 64]
    assert a9.formal_config()["arms"] == ["dp2-tp4", "tp8"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.pop("tpot_method"),
        lambda config: config["rates_rps"].append(128),
        lambda config: config.__setitem__("repeats", 2),
        lambda config: config.__setitem__("extra", True),
    ],
)
def test_formal_config_rejects_every_contract_change(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    config = a9.formal_config()
    mutation(config)

    with pytest.raises(a9.GateEvidenceError, match="formal config differs"):
        a9._validate_formal_config(config)


@pytest.mark.parametrize(
    ("orders", "first_status", "first_rate", "bracket"),
    [
        (
            (-1, -1, -1, -1, -1),
            "none_in_measured_range",
            None,
            None,
        ),
        (
            (1, 1, 1, 1, 1),
            "dp_noninferior_from_min_rate",
            4,
            None,
        ),
        (
            (0, 0, 0, 0, 0),
            "dp_noninferior_from_min_rate",
            4,
            None,
        ),
        (
            (-1, -1, 0, 1, 1),
            "dp_first_noninferior_within_range",
            16,
            [8, 16],
        ),
        (
            (-1, 1, -1, 0, 1),
            "dp_first_noninferior_within_range",
            8,
            [4, 8],
        ),
    ],
)
def test_derive_crossover_covers_every_verdict(
    orders: tuple[int, ...],
    first_status: str,
    first_rate: int | None,
    bracket: list[int] | None,
) -> None:
    result = a9.derive_crossover(_rate_metrics(orders))

    assert result["first_dp_noninferior_status"] == first_status
    assert result["first_dp_noninferior_rate_rps"] == first_rate
    assert result["arrival_rate_bracket_rps"] == bracket
    assert [row["order"] for row in result["order_by_rate"]] == list(
        orders
    )
    expected_transitions = sum(
        current != previous
        for previous, current in zip(orders, orders[1:], strict=False)
    )
    assert len(result["all_order_transitions"]) == expected_transitions
    assert result["has_order_transition"] is bool(expected_transitions)
    assert result["no_order_transition_in_measured_range"] is (
        not bool(expected_transitions)
    )
    assert result["status"] == (
        "order_transition_observed"
        if expected_transitions
        else "no_order_transition_in_measured_range"
    )
    transition_indices = [
        index
        for index in range(1, len(orders))
        if orders[index] != orders[index - 1]
    ]
    for transition, index in zip(
        result["all_order_transitions"],
        transition_indices,
        strict=True,
    ):
        assert transition["lower_observed_concurrency"] == {
            "dp_median_max_in_flight": float(a9.RATES_RPS[index - 1]),
            "tp8_median_max_in_flight": float(
                a9.RATES_RPS[index - 1] + 1
            ),
        }
        assert transition["upper_observed_concurrency"] == {
            "dp_median_max_in_flight": float(a9.RATES_RPS[index]),
            "tp8_median_max_in_flight": float(a9.RATES_RPS[index] + 1),
        }
    assert result["interpolation"] == "none"
    if first_rate is None:
        assert result["observed_concurrency_at_first_noninferior_rate"] is None
    else:
        assert result["observed_concurrency_at_first_noninferior_rate"] == {
            "dp_median_max_in_flight": float(first_rate),
            "tp8_median_max_in_flight": float(first_rate + 1),
        }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            _rate_metrics((-1, -1, -1, -1, -1))[:-1],
            "complete arrival grid",
        ),
        (
            [
                {**row, "rate_rps": 4 if row["rate_rps"] == 8 else row["rate_rps"]}
                for row in _rate_metrics((-1, -1, -1, -1, -1))
            ],
            "arrival grid differs",
        ),
        (
            [
                {**row, "rate_rps": 128 if row["rate_rps"] == 64 else row["rate_rps"]}
                for row in _rate_metrics((-1, -1, -1, -1, -1))
            ],
            "arrival grid differs",
        ),
        (
            [
                {
                    **row,
                    "dp_median_goodput_rps": (
                        float("nan")
                        if row["rate_rps"] == 4
                        else row["dp_median_goodput_rps"]
                    ),
                }
                for row in _rate_metrics((-1, -1, -1, -1, -1))
            ],
            "must be finite",
        ),
    ],
)
def test_derive_crossover_rejects_incomplete_or_invalid_rows(
    rows: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(a9.GateEvidenceError, match=message):
        a9.derive_crossover(rows)


def test_fixed_a8_artifact_replays_with_only_the_accepted_deviation() -> None:
    rows, manifest, descriptor = a9.load_a8_dp_evidence()

    assert len(rows) == a9.REPEATS * len(a9.RATES_RPS) * a9.REQUESTS_PER_CELL
    assert all(
        row.get("phase") == "scaling" and row.get("arm") == "dp"
        for row in rows
    )
    assert manifest["passed"] is False
    assert {
        name for name, passed in manifest["checks"].items() if passed is not True
    } == {a9.A8_ACCEPTED_FALSE_CHECK}
    assert descriptor == {
        "path": a9.A8_ARTIFACT_DIR.as_posix(),
        "raw_sha256": a9.A8_RAW_SHA256,
        "manifest_sha256": a9.A8_MANIFEST_SHA256,
        "placements_sha256": a9.A8_PLACEMENTS_SHA256,
        "source_commit": a9.A8_SOURCE_COMMIT,
        "image_id": a9.A8_IMAGE_ID,
    }


@pytest.mark.parametrize(
    ("tampered_name", "message"),
    [
        (a8.RAW_NAME, "raw SHA-256 differs"),
        (a8.MANIFEST_NAME, "manifest SHA-256 differs"),
        ("placements.jsonl", "placements SHA-256 differs"),
    ],
)
def test_a8_descriptor_rejects_each_tampered_file(
    monkeypatch: pytest.MonkeyPatch,
    tampered_name: str,
    message: str,
) -> None:
    hashes = {
        a8.RAW_NAME: a9.A8_RAW_SHA256,
        a8.MANIFEST_NAME: a9.A8_MANIFEST_SHA256,
        "placements.jsonl": a9.A8_PLACEMENTS_SHA256,
    }

    def fake_sha256_file(path: Path) -> str:
        return "0" * 64 if path.name == tampered_name else hashes[path.name]

    monkeypatch.setattr(a9, "sha256_file", fake_sha256_file)
    with pytest.raises(a9.GateEvidenceError, match=message):
        a9._a8_descriptor(a9.A8_ARTIFACT_DIR)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            _a8_manifest(passed=True),
            "verdict differs",
        ),
        (
            _a8_manifest(
                passed=False,
                checks={a9.A8_ACCEPTED_FALSE_CHECK: True},
            ),
            "verdict differs",
        ),
        (
            _a8_manifest(
                checks={
                    a9.A8_ACCEPTED_FALSE_CHECK: False,
                    "unaccepted_failure": False,
                }
            ),
            "may retain only",
        ),
    ],
)
def test_a8_reuse_rejects_any_verdict_drift(
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, object],
    message: str,
) -> None:
    row_count = a9.REPEATS * len(a9.RATES_RPS) * a9.REQUESTS_PER_CELL
    monkeypatch.setattr(a9, "_a8_descriptor", lambda _path: {})
    monkeypatch.setattr(
        a8,
        "verify_artifact",
        lambda _path, *, assert_gate: copy.deepcopy(manifest),
    )
    monkeypatch.setattr(
        a8,
        "_read_jsonl",
        lambda _path: [
            {"type": "request", "phase": "scaling", "arm": "dp"}
            for _ in range(row_count)
        ],
    )

    with pytest.raises(a9.GateEvidenceError, match=message):
        a9.load_a8_dp_evidence(Path("unused"))


def test_a8_reuse_accepts_an_improved_fully_passing_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = a9.REPEATS * len(a9.RATES_RPS) * a9.REQUESTS_PER_CELL
    manifest = _a8_manifest(
        passed=True,
        checks={a9.A8_ACCEPTED_FALSE_CHECK: True},
    )
    monkeypatch.setattr(a9, "_a8_descriptor", lambda _path: {})
    monkeypatch.setattr(
        a8,
        "verify_artifact",
        lambda _path, *, assert_gate: copy.deepcopy(manifest),
    )
    monkeypatch.setattr(
        a8,
        "_read_jsonl",
        lambda _path: [
            {"type": "request", "phase": "scaling", "arm": "dp"}
            for _ in range(row_count)
        ],
    )

    rows, verified, _descriptor = a9.load_a8_dp_evidence(Path("unused"))

    assert len(rows) == row_count
    assert verified["passed"] is True


def test_expected_runtime_package_is_derived_from_the_pinned_a8_source() -> None:
    package = a9._expected_runtime_package()

    assert package["source_commit"] == a9.A8_SOURCE_COMMIT
    assert package["source_archive_sha256"] == a9.A8_SOURCE_ARCHIVE_SHA256
    assert package["file_count"] == a9.A8_RUNTIME_FILE_COUNT == 199
    assert package["files_sha256"] == a9.A8_RUNTIME_FILES_SHA256


def test_delegated_a8_helper_matches_the_measured_source_blob() -> None:
    assert a9._a8_helper_reference(a9.A8_SOURCE_COMMIT) == {
        "path": a9.A8_HELPER_PATH.as_posix(),
        "source_commit": a9.A8_SOURCE_COMMIT,
        "sha256": a9.A8_HELPER_SHA256,
    }


def test_delegated_a8_helper_rejects_current_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_blob(commit: str, _path: Path) -> str:
        return (
            a9.A8_HELPER_SHA256
            if commit == a9.A8_SOURCE_COMMIT
            else "0" * 64
        )

    monkeypatch.setattr(a9, "_git_blob_sha256", fake_blob)
    with pytest.raises(a9.GateEvidenceError, match="delegated A8 helper"):
        a9._a8_helper_reference("a" * 40)


def test_container_provenance_accepts_the_exact_mount_matrix() -> None:
    a9._validate_container_provenance(_valid_container_provenance())


@pytest.mark.parametrize(
    ("service", "destination", "field", "value"),
    [
        ("tp8", "/models", "type", "bind"),
        ("tp8", "/models", "name", "wrong-volume"),
        ("gateway", "/evidence", "rw", False),
        (
            "gateway",
            "/etc/kairyu/config.yaml",
            "source",
            "repo:wrong.yaml",
        ),
    ],
)
def test_container_provenance_rejects_mount_tampering(
    service: str,
    destination: str,
    field: str,
    value: object,
) -> None:
    provenance = _valid_container_provenance()
    services = provenance["services"]
    assert isinstance(services, dict)
    descriptor = services[service]
    assert isinstance(descriptor, dict)
    mounts = descriptor["mounts"]
    assert isinstance(mounts, list)
    mount = next(
        item
        for item in mounts
        if isinstance(item, dict)
        and item.get("destination") == destination
    )
    mount[field] = value

    with pytest.raises(a9.GateEvidenceError, match="mount evidence invalid"):
        a9._validate_container_provenance(provenance)


def test_container_provenance_rejects_source_bind_injected_into_raw() -> None:
    provenance = _valid_container_provenance()
    services = provenance["services"]
    assert isinstance(services, dict)
    tp8 = services["tp8"]
    assert isinstance(tp8, dict)
    mounts = tp8["mounts"]
    assert isinstance(mounts, list)
    mounts.append(
        {
            "type": "bind",
            "source": "repo:kairyu",
            "destination": "/app/kairyu",
            "name": None,
            "rw": False,
        }
    )

    with pytest.raises(a9.GateEvidenceError, match="mount evidence invalid"):
        a9._validate_container_provenance(provenance)


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "revision",
        "weights_sha256",
        "index_sha256",
        "checkpoint_evidence_sha256",
        "tokenizer_sha256",
        "a7_raw_sha256",
        "a7_manifest_sha256",
        "live_volume_attestation",
    ],
)
def test_checkpoint_descriptor_requires_every_a8_field(field: str) -> None:
    a8_checkpoint = {
        "model": a9.MODEL,
        "revision": a9.MODEL_REVISION,
        "weights_sha256": "1" * 64,
        "index_sha256": "2" * 64,
        "checkpoint_evidence_sha256": "3" * 64,
        "tokenizer_sha256": "4" * 64,
        "a7_raw_sha256": "5" * 64,
        "a7_manifest_sha256": "6" * 64,
        "live_volume_attestation": {"weights": []},
        "attested_container_id": "7" * 64,
    }
    container_id = "8" * 64
    measured = {
        **a8_checkpoint,
        "attested_container_id": container_id,
    }
    assert a9._validate_checkpoint_descriptor(
        measured,
        a8_checkpoint=a8_checkpoint,
        container_id=container_id,
    ) == measured

    tampered = copy.deepcopy(measured)
    tampered.pop(field)
    with pytest.raises(
        a9.GateEvidenceError,
        match="checkpoint descriptor differs",
    ):
        a9._validate_checkpoint_descriptor(
            tampered,
            a8_checkpoint=a8_checkpoint,
            container_id=container_id,
        )


def test_tp8_request_validation_accepts_the_exact_a8_dp_contract(
    a8_trace: dict[str, object],
) -> None:
    scaling = a8_trace["scaling"]
    assert isinstance(scaling, list)
    trace_scaling = {
        int(row["sequence"]): row for row in scaling if isinstance(row, dict)
    }
    row = _metric_row(
        arm="dp",
        repeat=0,
        rate_rps=4,
        sequence=0,
        origin_ns=10**12,
    )
    row.update(
        {
            "arm": "dp",
            "prompt_index": 0,
            "prompt_sha256": trace_scaling[0]["prompt_sha256"],
            "retry_count": 0,
            "http_status": 200,
            "error_type": None,
            "saw_done": True,
            "finish_reason": "length",
            "total_ns": int(row["ended_ns"]) - int(row["started_ns"]),
            "output_sha256": "1" * 64,
            "response_request_id": "request-0",
        }
    )
    row["namespace_sha256"] = a8._expected_namespace_sha256(row)
    row["namespace_token_id"] = a8._expected_namespace_token_id(row)
    row["arm"] = "tp8"
    row["namespace_arm"] = "dp"

    a9._validate_tp8_request(row, trace_scaling)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("retry_count", 1, "retry_count must be zero"),
        ("completion_tokens", 31, "exactly 32 tokens"),
        ("status", "failure", "must succeed"),
        ("response_request_id", "", "X-Request-ID missing"),
        ("namespace_arm", "single", "namespace ownership is invalid"),
    ],
)
def test_tp8_request_validation_rejects_semantic_tampering(
    a8_trace: dict[str, object],
    field: str,
    value: object,
    message: str,
) -> None:
    scaling = a8_trace["scaling"]
    assert isinstance(scaling, list)
    trace_scaling = {
        int(row["sequence"]): row for row in scaling if isinstance(row, dict)
    }
    row = _metric_row(
        arm="dp",
        repeat=0,
        rate_rps=4,
        sequence=0,
        origin_ns=10**12,
    )
    row.update(
        {
            "arm": "dp",
            "prompt_index": 0,
            "prompt_sha256": trace_scaling[0]["prompt_sha256"],
            "retry_count": 0,
            "http_status": 200,
            "error_type": None,
            "saw_done": True,
            "finish_reason": "length",
            "total_ns": int(row["ended_ns"]) - int(row["started_ns"]),
            "output_sha256": "1" * 64,
            "response_request_id": "request-0",
        }
    )
    row["namespace_sha256"] = a8._expected_namespace_sha256(row)
    row["namespace_token_id"] = a8._expected_namespace_token_id(row)
    row["arm"] = "tp8"
    row["namespace_arm"] = "dp"
    row[field] = value

    with pytest.raises(a9.GateEvidenceError, match=message):
        a9._validate_tp8_request(row, trace_scaling)


def test_terminal_tpot_uses_first_content_to_stream_terminal_arithmetic() -> None:
    samples: list[dict[str, object]] = []
    for sequence in range(a9.REQUESTS_PER_CELL):
        row = _metric_row(
            arm="tp8",
            repeat=0,
            rate_rps=4,
            sequence=sequence,
            origin_ns=10**12,
        )
        row["ended_ns"] = int(row["first_token_ns"]) + 31 * 7_000
        samples.append(row)

    metrics = a9._cell_metrics(
        samples,
        arm="tp8",
        repeat=0,
        rate_rps=4,
    )

    assert metrics["tpot_mean_ns"] == 7_000
    assert metrics["tpot_p50_ns"] == 7_000
    assert metrics["tpot_p99_ns"] == 7_000
    assert a9.TPOT_METHOD == "stream-terminal-token-v1"
    assert "stream_terminal_ns" in a9.TPOT_DEFINITION


def test_complete_synthetic_raw_recomputes_a_report_only_manifest(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_evidence: tuple[
        list[dict[str, object]], list[dict[str, object]]
    ],
) -> None:
    rows, a8_rows = synthetic_evidence
    _patch_synthetic_dependencies(monkeypatch, a8_rows)

    manifest = a9.recompute_manifest(
        copy.deepcopy(rows),
        raw_sha256="f" * 64,
    )

    assert manifest["passed"] is True
    assert manifest["report_only"] is True
    assert len(manifest["cell_metrics"]) == 30
    assert len(manifest["rate_metrics"]) == 5
    assert manifest["raw"] == {
        "tp8_sha256": "f" * 64,
        "tp8_request_count": 984,
        "tp8_placement_count": 984,
        "a8_dp_scaling_request_count": 960,
    }
    assert manifest["crossover"]["status"] == (
        "no_order_transition_in_measured_range"
    )
    assert manifest["crossover"]["first_dp_noninferior_status"] == (
        "dp_noninferior_from_min_rate"
    )


@pytest.mark.parametrize(
    ("row_index", "field"),
    [
        (0, "started_unix_ns"),
        (0, "started_monotonic_ns"),
        (-1, "ended_unix_ns"),
        (-1, "ended_monotonic_ns"),
    ],
)
def test_run_clock_fields_require_exact_integers(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_evidence: tuple[
        list[dict[str, object]], list[dict[str, object]]
    ],
    row_index: int,
    field: str,
) -> None:
    source_rows, a8_rows = synthetic_evidence
    rows = copy.deepcopy(source_rows)
    rows[row_index][field] = True
    _patch_synthetic_dependencies(monkeypatch, a8_rows)

    with pytest.raises(
        a9.GateEvidenceError,
        match=rf"run {field} must be an integer",
    ):
        a9.recompute_manifest(rows, raw_sha256="f" * 64)


def test_run_end_monotonic_clock_cannot_precede_run_start(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_evidence: tuple[
        list[dict[str, object]], list[dict[str, object]]
    ],
) -> None:
    source_rows, a8_rows = synthetic_evidence
    rows = copy.deepcopy(source_rows)
    rows[0]["started_monotonic_ns"] = 100
    rows[-1]["ended_monotonic_ns"] = 99
    _patch_synthetic_dependencies(monkeypatch, a8_rows)

    with pytest.raises(
        a9.GateEvidenceError,
        match="run ended_monotonic_ns must be an integer >= 100",
    ):
        a9.recompute_manifest(rows, raw_sha256="f" * 64)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scheduled_ns", 0),
        ("started_ns", 10**18 + 1),
        ("first_token_ns", 10**18 + 1),
        ("ended_ns", 10**18 + 1),
    ],
)
def test_every_tp8_request_timestamp_must_stay_inside_run_interval(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_evidence: tuple[
        list[dict[str, object]], list[dict[str, object]]
    ],
    field: str,
    value: int,
) -> None:
    source_rows, a8_rows = synthetic_evidence
    rows = copy.deepcopy(source_rows)
    request = _first_row(rows, row_type="request")
    request[field] = value
    _patch_synthetic_dependencies(monkeypatch, a8_rows)

    with pytest.raises(a9.GateEvidenceError, match="escape the run bounds"):
        a9.recompute_manifest(rows, raw_sha256="f" * 64)


def test_measurement_paths_require_the_exact_placement_log_name(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    placement_log = output_dir / "placements.jsonl"

    assert a9._resolve_measurement_paths(
        gateway_placement_log=placement_log,
        output_dir=output_dir,
    ) == (output_dir.resolve(), placement_log.resolve())


def test_compose_preflight_supplies_every_required_fixed_environment(
    tmp_path: Path,
) -> None:
    evidence_dir = (tmp_path / "run").resolve()
    environment = a9._compose_environment(
        a9.A8_IMAGE_ID,
        evidence_dir=evidence_dir,
    )

    assert environment["KAIRYU_A9_IMAGE"] == a9.A8_IMAGE_ID
    assert environment["KAIRYU_A9_IMAGE_ID"] == a9.A8_IMAGE_ID
    assert environment["KAIRYU_A9_RUN_DIR"] == str(evidence_dir)
    assert environment["KAIRYU_A9_RUNTIME_DIR"] == str(
        a9._runtime_source_dir(evidence_dir)
    )


@pytest.mark.parametrize(
    "placement_log",
    [
        Path("run/other.jsonl"),
        Path("run/nested/placements.jsonl"),
        Path("placements.jsonl"),
    ],
)
def test_measurement_paths_reject_any_other_resolved_path(
    tmp_path: Path,
    placement_log: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="must resolve to --output-dir/placements.jsonl",
    ):
        a9._resolve_measurement_paths(
            gateway_placement_log=tmp_path / placement_log,
            output_dir=tmp_path / "run",
        )


def test_runtime_identity_allows_gpu_memory_accounting_to_change(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_evidence: tuple[
        list[dict[str, object]], list[dict[str, object]]
    ],
) -> None:
    source_rows, a8_rows = synthetic_evidence
    rows = copy.deepcopy(source_rows)
    final_gpus = copy.deepcopy(rows[-1]["gpu_inventory"])
    assert isinstance(final_gpus, list)
    assert isinstance(final_gpus[0], dict)
    final_gpus[0]["memory_used_mib"] = 12345
    rows[-1]["gpu_inventory"] = final_gpus
    _patch_synthetic_dependencies(monkeypatch, a8_rows)

    assert a9.recompute_manifest(rows, raw_sha256="f" * 64)["passed"] is True


def test_runtime_identity_rejects_physical_gpu_change(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_evidence: tuple[
        list[dict[str, object]], list[dict[str, object]]
    ],
) -> None:
    source_rows, a8_rows = synthetic_evidence
    rows = copy.deepcopy(source_rows)
    final_gpus = copy.deepcopy(rows[-1]["gpu_inventory"])
    assert isinstance(final_gpus, list)
    assert isinstance(final_gpus[0], dict)
    final_gpus[0]["uuid"] = "GPU-replaced"
    rows[-1]["gpu_inventory"] = final_gpus
    _patch_synthetic_dependencies(monkeypatch, a8_rows)

    with pytest.raises(a9.GateEvidenceError, match="runtime identity changed"):
        a9.recompute_manifest(rows, raw_sha256="f" * 64)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_insert_unknown_row, "unknown row type"),
        (_drop_request, "request matrix is incomplete"),
        (_drop_placement, "must have one placement row"),
        (_swap_first_two_requests, "request order differs"),
        (_duplicate_response_id, "response IDs are not unique"),
        (_orphan_placement, "does not join exactly one"),
        (_break_placement_identity, "placement/request identity mismatch"),
    ],
)
def test_synthetic_raw_rejects_unknown_missing_or_tampered_rows(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_evidence: tuple[
        list[dict[str, object]], list[dict[str, object]]
    ],
    mutation: Callable[[list[dict[str, object]]], None],
    message: str,
) -> None:
    source_rows, a8_rows = synthetic_evidence
    rows = copy.deepcopy(source_rows)
    mutation(rows)
    _patch_synthetic_dependencies(monkeypatch, a8_rows)

    with pytest.raises(a9.GateEvidenceError, match=message):
        a9.recompute_manifest(rows, raw_sha256="f" * 64)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("", "evidence is empty"),
        ('{"row_index":0}', "JSONL framing"),
        ('{"row_index":1}\n', "indices are not contiguous"),
        ('["not-an-object"]\n', "is not an object"),
    ],
)
def test_read_jsonl_rejects_missing_or_malformed_rows(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(payload, encoding="utf-8")

    with pytest.raises(a9.GateEvidenceError, match=message):
        a9._read_jsonl(raw_path)
